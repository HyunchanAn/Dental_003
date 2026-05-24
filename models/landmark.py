import os
import urllib.request
import torch
import numpy as np
import cv2
from typing import Dict, Tuple

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("WARNING: segment-anything 패키지가 설치되지 않았습니다. pip install git+https://github.com/facebookresearch/segment-anything.git 를 실행하세요.")

class PerioLandmarkPredictor:
    def __init__(self, device: str = "cuda"):
        """
        SAM(Segment Anything Model) 기반 랜드마크 예측 모델을 초기화합니다.
        """
        self.device = device
        self.model_type = "vit_b"
        self.checkpoint_path = "models/sam_vit_b_01ec64.pth"
        
        self._ensure_checkpoint_exists()
        
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def _ensure_checkpoint_exists(self):
        """SAM 모델 가중치가 없으면 자동으로 다운로드합니다."""
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            print(f"Downloading SAM checkpoint to {self.checkpoint_path}...")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, self.checkpoint_path)
            print("Download complete.")

    def predict_landmarks(self, image_rgb: np.ndarray, bbox: list) -> Dict[str, Tuple[float, float]]:
        """
        SAM을 이용하여 치아 마스크를 추출하고, 기하학적 휴리스틱을 통해 5개의 핵심 랜드마크를 추론합니다.
        
        Args:
            image_rgb: 원본 RGB 이미지 배열
            bbox: YOLOv11 OBB 포맷 [cx, cy, w, h, angle] 또는 일반 BBox
            
        Returns:
            Dictionary of landmarks: 
            {
                "mesial_cej": (x, y),
                "distal_cej": (x, y),
                "mesial_crest": (x, y),
                "distal_crest": (x, y),
                "root_apex": (x, y)
            }
        """
        cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # SAM은 [x_min, y_min, x_max, y_max] 포맷의 박스 프롬프트를 받습니다.
        x_min = max(0, cx - w / 2)
        y_min = max(0, cy - h / 2)
        x_max = min(image_rgb.shape[1], cx + w / 2)
        y_max = min(image_rgb.shape[0], cy + h / 2)
        input_box = np.array([x_min, y_min, x_max, y_max])

        # SAM 이미지 인코딩
        self.predictor.set_image(image_rgb)
        
        # 마스크 추론
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        mask = masks[0].astype(np.uint8) * 255
        
        # 마스크의 윤곽선(Contour) 추출
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # 마스크를 찾지 못한 경우 박스 기반 임의의 점 반환
            return self._get_fallback_landmarks(cx, cy, w, h)
            
        largest_contour = max(contours, key=cv2.contourArea)
        points = largest_contour.squeeze(1) # [N, 2]
        
        # 악궁 상하 위치 판단 (Maxillary vs Mandibular)
        # 이미지의 위쪽 절반에 있으면 상악(뿌리가 위로 향함), 아래쪽에 있으면 하악(뿌리가 아래로 향함)
        img_h = image_rgb.shape[0]
        is_upper = cy < (img_h / 2)
        
        if is_upper:
            # 상악: Root Apex는 Y 좌표가 가장 작은 점 (가장 위쪽)
            apex_idx = np.argmin(points[:, 1])
            apex = tuple(points[apex_idx])
            # 치관(Crown) 쪽은 Y 좌표가 가장 큰 점 (가장 아래쪽)
            crown_y = np.max(points[:, 1])
            root_y = apex[1]
        else:
            # 하악: Root Apex는 Y 좌표가 가장 큰 점 (가장 아래쪽)
            apex_idx = np.argmax(points[:, 1])
            apex = tuple(points[apex_idx])
            crown_y = np.min(points[:, 1])
            root_y = apex[1]
            
        # CEJ와 Crest는 기하학적 휴리스틱 적용 (전체 치아 길이의 특정 비율)
        total_length = abs(crown_y - root_y)
        
        # 대략 치관에서 뿌리 방향으로 30% 지점을 CEJ, 40% 지점을 Crest로 추정
        cej_y_target = crown_y - 0.3 * total_length if is_upper else crown_y + 0.3 * total_length
        crest_y_target = crown_y - 0.4 * total_length if is_upper else crown_y + 0.4 * total_length
        
        # 윤곽선 상에서 타겟 Y 좌표와 가장 가까운 좌/우 점 탐색
        mesial_cej, distal_cej = self._find_left_right_points(points, cej_y_target, cx)
        mesial_crest, distal_crest = self._find_left_right_points(points, crest_y_target, cx)
        
        return {
            "mesial_cej": mesial_cej,
            "distal_cej": distal_cej,
            "mesial_crest": mesial_crest,
            "distal_crest": distal_crest,
            "root_apex": (float(apex[0]), float(apex[1]))
        }
        
    def _find_left_right_points(self, points: np.ndarray, target_y: float, cx: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        # target_y 와 y값이 가장 비슷한 점들 필터링
        y_dists = np.abs(points[:, 1] - target_y)
        close_indices = np.argsort(y_dists)[:20] # 가장 가까운 20개 점
        close_points = points[close_indices]
        
        # cx를 기준으로 좌(Mesial/Distal) 우(Distal/Mesial) 분리
        left_points = close_points[close_points[:, 0] < cx]
        right_points = close_points[close_points[:, 0] >= cx]
        
        if len(left_points) > 0:
            left_pt = left_points[np.argmin(np.abs(left_points[:, 1] - target_y))]
        else:
            left_pt = np.array([cx - 20, target_y])
            
        if len(right_points) > 0:
            right_pt = right_points[np.argmin(np.abs(right_points[:, 1] - target_y))]
        else:
            right_pt = np.array([cx + 20, target_y])
            
        return (float(left_pt[0]), float(left_pt[1])), (float(right_pt[0]), float(right_pt[1]))

    def _get_fallback_landmarks(self, cx, cy, w, h):
        return {
            "mesial_cej": (cx - w/4, cy),
            "distal_cej": (cx + w/4, cy),
            "mesial_crest": (cx - w/4, cy + h/10),
            "distal_crest": (cx + w/4, cy + h/10),
            "root_apex": (cx, cy + h/2)
        }
