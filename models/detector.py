"""
Tooth Detection and Numbering Module using YOLOv11-OBB.

이 모듈은 방사선 이미지(Panoramic/Periapical)를 입력받아
FDI 시스템(11~18, 21~28, 31~38, 41~48) 기반의 치아 번호, 
Oriented Bounding Box(OBB) 좌표 및 신뢰도 점수를 반환합니다.
"""

import torch
import numpy as np
from typing import List, Dict, Union
from ultralytics import YOLO

class ToothDetector:
    def __init__(self, weights_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        YOLOv11 OBB 모델을 로드하여 치아 검출기를 초기화합니다.
        
        Args:
            weights_path: 학습된 가중치 파일 경로 (.pt)
            device: 실행 디바이스 (cuda 또는 cpu)
        """
        self.device = device
        self.model = YOLO(weights_path).to(self.device)

    def predict(self, image: Union[np.ndarray, str]) -> List[Dict[str, Union[int, List[float], float]]]:
        """
        방사선 영상을 입력받아 치아 검출 결과를 반환합니다.
        
        Args:
            image: numpy array (RGB) 또는 이미지 경로
            
        Returns:
            List of dictionary containing:
              - tooth_number: FDI 치아 번호 (int)
              - bbox: [x_center, y_center, width, height, angle]
              - confidence: 신뢰도 (float)
        """
        results = self.model(image, conf=0.45, iou=0.45, verbose=False)
        return self._postprocess(results)

    def _postprocess(self, model_outputs) -> List[Dict[str, Union[int, List[float], float]]]:
        """
        추론 결과에 Non-Maximum Suppression(NMS) 및 OBB 포맷 파싱을 적용합니다.
        중복된 Bounding Box를 세밀하게 제어하여 인접한 치아나 겹친 뿌리에서의 
        오검출을 방지합니다.
        """
        parsed_results = []
        for result in model_outputs:
            if result.boxes is None and (not hasattr(result, 'obb') or result.obb is None):
                continue
                
            # OBB 포맷
            if hasattr(result, 'obb') and result.obb is not None:
                boxes = result.obb.xywhr.cpu().numpy() # [x, y, w, h, angle]
                confs = result.obb.conf.cpu().numpy()
                classes = result.obb.cls.cpu().numpy()
            else:
                boxes = result.boxes.xywh.cpu().numpy() # [x, y, w, h]
                boxes = np.concatenate([boxes, np.zeros((boxes.shape[0], 1))], axis=1) # angle 0.0 추가
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
            names = result.names
            
            for i in range(len(boxes)):
                class_id = int(classes[i])
                class_name = names[class_id]
                
                try:
                    tooth_number = int(class_name)
                except ValueError:
                    continue # 치아 번호가 아닌 클래스는 무시
                    
                parsed_results.append({
                    "tooth_number": tooth_number,
                    "bbox": boxes[i].tolist(),
                    "confidence": float(confs[i])
                })
        return parsed_results
