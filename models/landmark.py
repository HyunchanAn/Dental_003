import os
import urllib.request
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import torch
from utils.preprocess import crop_roi

try:
    from segment_anything import SamPredictor, sam_model_registry
    HAS_SAM = True
except ImportError:
    HAS_SAM = False


class PerioLandmarkPredictor:
    """
    SAM(Segment Anything Model) 치아 윤곽선 분할과 방사선 불투과도(Radiodensity) 프로파일 분석을 결합하여
    백악법랑경계(CEJ), 치간 치조골정(Alveolar Crest), 치근단(Root Apex) 5대 치주 랜드마크를 검출합니다.
    """

    def __init__(self, device: str = None, checkpoint_path: str = None, lazy_sam: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "vit_b"
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            checkpoint_path,
            os.path.join(base_dir, "models", "sam_vit_b_01ec64.pth"),
            os.path.join(base_dir, "weights", "sam_b.pt"),
            "models/sam_vit_b_01ec64.pth",
        ]
        resolved = next((p for p in candidates if p and os.path.exists(p)), None)
        self.checkpoint_path = resolved or checkpoint_path or os.path.join(base_dir, "models", "sam_vit_b_01ec64.pth")
        self.predictor = None

        if not lazy_sam:
            self._init_sam()

    def _init_sam(self):
        if self.predictor is not None or not HAS_SAM:
            return
        self._ensure_checkpoint_exists()
        if os.path.exists(self.checkpoint_path):
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)

    def _ensure_checkpoint_exists(self):
        """SAM 모델 가중치가 없으면 자동으로 다운로드합니다."""
        ckpt_dir = os.path.dirname(self.checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, self.checkpoint_path)

    @staticmethod
    def _to_clahe_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _extract_radiographic_landmarks(
        self,
        patch_gray: np.ndarray,
        tooth_mask: np.ndarray,
        points: np.ndarray,
        local_cx: float,
        is_upper: bool,
        all_teeth_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], float]:
        """
        치아 마스크 윤곽선과 치간부 방사선 밀도(CLAHE luminance) 프로파일을 분석하여
        실제 CEJ(법랑질-상아질 경계 및 치경부 협착점)와 치간 치조골정(Alveolar Crest)을 산출합니다.
        """
        ph, pw = patch_gray.shape[:2]
        exclusion_mask = all_teeth_mask if all_teeth_mask is not None else tooth_mask

        if is_upper:
            apex_idx = int(np.argmin(points[:, 1]))
            apex = (float(points[apex_idx, 0]), float(points[apex_idx, 1]))
            crown_y = float(np.max(points[:, 1]))
            root_y = float(apex[1])
        else:
            apex_idx = int(np.argmax(points[:, 1]))
            apex = (float(points[apex_idx, 0]), float(points[apex_idx, 1]))
            crown_y = float(np.min(points[:, 1]))
            root_y = float(apex[1])

        total_length = abs(crown_y - root_y)
        if total_length < 12.0:
            cej_y = crown_y - 0.28 * total_length if is_upper else crown_y + 0.28 * total_length
            crest_y = crown_y - 0.32 * total_length if is_upper else crown_y + 0.32 * total_length
            m_cej, d_cej = self._find_left_right_points_from_mask(tooth_mask, points, cej_y, local_cx)
            m_crest, d_crest = self._find_left_right_points_from_mask(tooth_mask, points, crest_y, local_cx)
            return m_cej, d_cej, m_crest, d_crest, apex, 0.55

        def t_to_y(t_val: float) -> float:
            return crown_y - t_val * total_length if is_upper else crown_y + t_val * total_length

        # 1. Axial profile sampling (t from 0.14 to 0.86)
        t_grid = np.linspace(0.14, 0.86, 37)
        left_x = np.zeros_like(t_grid)
        right_x = np.zeros_like(t_grid)
        y_grid = np.zeros_like(t_grid)
        internal_left_intensity = np.zeros_like(t_grid)
        internal_right_intensity = np.zeros_like(t_grid)
        interprox_left_intensity = np.zeros_like(t_grid)
        interprox_right_intensity = np.zeros_like(t_grid)
        non_tooth_left_ratio = np.zeros_like(t_grid)
        non_tooth_right_ratio = np.zeros_like(t_grid)

        strip_pad = max(12, min(22, int(round(pw * 0.10))))

        for idx, t_val in enumerate(t_grid):
            yk = t_to_y(float(t_val))
            y_grid[idx] = yk
            l_pt, r_pt = self._find_left_right_points_from_mask(tooth_mask, points, yk, local_cx)
            lx, rx = l_pt[0], r_pt[0]
            left_x[idx] = lx
            right_x[idx] = rx

            yi_min = max(0, int(round(yk)) - 2)
            yi_max = min(ph, int(round(yk)) + 3)
            if yi_max <= yi_min:
                continue

            # Internal peripheral intensity (enamel vs dentin transition inside tooth contour)
            il_x1 = max(0, min(pw - 1, int(round(lx)) + 1))
            il_x2 = max(il_x1 + 1, min(pw, int(round(lx)) + 7))
            internal_left_intensity[idx] = float(np.mean(patch_gray[yi_min:yi_max, il_x1:il_x2]))

            ir_x2 = max(1, min(pw, int(round(rx)) - 1))
            ir_x1 = max(0, min(ir_x2 - 1, int(round(rx)) - 7))
            internal_right_intensity[idx] = float(np.mean(patch_gray[yi_min:yi_max, ir_x1:ir_x2]))

            # External interproximal strip outside root contour, excluding any tooth mask
            el_x2 = max(1, min(pw, int(round(lx)) - 2))
            el_x1 = max(0, min(el_x2 - 1, int(round(lx)) - strip_pad))
            strip_l = patch_gray[yi_min:yi_max, el_x1:el_x2]
            mask_l = exclusion_mask[yi_min:yi_max, el_x1:el_x2] == 0
            if strip_l.size > 0:
                non_tooth_left_ratio[idx] = float(np.mean(mask_l))
                if np.any(mask_l):
                    interprox_left_intensity[idx] = float(np.mean(strip_l[mask_l]))
                else:
                    interprox_left_intensity[idx] = float(np.mean(strip_l))

            er_x1 = max(0, min(pw - 1, int(round(rx)) + 2))
            er_x2 = max(er_x1 + 1, min(pw, int(round(rx)) + strip_pad))
            strip_r = patch_gray[yi_min:yi_max, er_x1:er_x2]
            mask_r = exclusion_mask[yi_min:yi_max, er_x1:er_x2] == 0
            if strip_r.size > 0:
                non_tooth_right_ratio[idx] = float(np.mean(mask_r))
                if np.any(mask_r):
                    interprox_right_intensity[idx] = float(np.mean(strip_r[mask_r]))
                else:
                    interprox_right_intensity[idx] = float(np.mean(strip_r))

        widths = np.maximum(4.0, right_x - left_x)

        # 2. Radiographic CEJ Detection within cervical window t in [0.20, 0.36]
        cervical_mask = (t_grid >= 0.20) & (t_grid <= 0.36)
        cervical_indices = np.where(cervical_mask)[0]

        d_width = -np.gradient(widths)
        d_int_l = -np.gradient(internal_left_intensity)
        d_int_r = -np.gradient(internal_right_intensity)

        def _norm_vec(v: np.ndarray) -> np.ndarray:
            span = float(np.ptp(v))
            return (v - float(np.min(v))) / span if span > 1e-5 else np.zeros_like(v)

        w_score = _norm_vec(d_width[cervical_indices])
        l_cej_score = 0.65 * _norm_vec(d_int_l[cervical_indices]) + 0.35 * w_score
        r_cej_score = 0.65 * _norm_vec(d_int_r[cervical_indices]) + 0.35 * w_score

        idx_cej_l = int(cervical_indices[int(np.argmax(l_cej_score))])
        idx_cej_r = int(cervical_indices[int(np.argmax(r_cej_score))])

        # 3. Interproximal Alveolar Crest Detection on left and right sides
        def _detect_side_crest(
            inter_profile: np.ndarray,
            open_ratio: np.ndarray,
            idx_cej: int,
        ) -> Tuple[int, int, float]:
            start_idx = min(len(t_grid) - 4, idx_cej + 1)
            trough_end = min(len(t_grid) - 3, int(len(t_grid) * 0.52))
            end_idx = min(len(t_grid) - 2, int(len(t_grid) * 0.78))

            if end_idx <= start_idx + 2 or trough_end <= start_idx:
                return idx_cej, start_idx, 0.75

            # Check if interproximal space is genuinely open (not occluded by tight adjacent tooth contact)
            cervical_open = float(np.mean(open_ratio[start_idx : max(start_idx + 2, trough_end)]))
            if cervical_open < 0.35:
                return idx_cej, start_idx, 0.80

            smoothed = np.convolve(inter_profile, np.array([0.25, 0.5, 0.25]), mode="same")

            # Locate the darkest radiolucent trough in the coronal-to-mid root window [start_idx, trough_end]
            trough_slice = smoothed[start_idx:trough_end]
            valid_open = open_ratio[start_idx:trough_end] >= 0.25
            if not np.any(valid_open):
                return idx_cej, start_idx, 0.78

            masked_trough = np.where(valid_open, trough_slice, 255.0)
            rel_trough = int(np.argmin(masked_trough))
            idx_trough = start_idx + rel_trough
            i_trough = float(smoothed[idx_trough])

            deep_start = max(idx_trough + 2, int(len(t_grid) * 0.42))
            if deep_start >= end_idx:
                return idx_cej, start_idx, 0.76

            deep_slice = smoothed[deep_start:end_idx]
            deep_open = open_ratio[deep_start:end_idx] >= 0.25
            if np.any(deep_open):
                deep_bone_val = float(np.percentile(deep_slice[deep_open], 75))
            else:
                deep_bone_val = float(np.percentile(deep_slice, 75))

            contrast = deep_bone_val - i_trough
            # A true periodontal resorption defect requires a radiolucent trough (i_trough < 62)
            # followed by a distinct radiopaque alveolar bone rise (contrast >= 28 and deep_bone_val >= 75)
            if i_trough >= 62.0 or deep_bone_val < 75.0 or contrast < 28.0:
                return idx_cej, start_idx, 0.80

            bone_thresh = i_trough + 0.55 * contrast
            crest_idx = idx_trough + 1
            for k in range(idx_trough + 1, end_idx):
                if open_ratio[k] >= 0.20 and float(smoothed[k]) >= bone_thresh:
                    crest_idx = k
                    break

            # On tilted molars where the crown overhangs the defect, refine CEJ to just above the trough
            refined_cej = max(idx_cej, idx_trough - 2) if idx_trough > idx_cej + 2 else idx_cej
            snr_conf = float(np.clip(0.74 + min(contrast, 60.0) / 250.0, 0.74, 0.94))
            return refined_cej, crest_idx, snr_conf

        idx_cej_l, idx_crest_l, conf_l = _detect_side_crest(
            interprox_left_intensity, non_tooth_left_ratio, idx_cej_l
        )
        idx_cej_r, idx_crest_r, conf_r = _detect_side_crest(
            interprox_right_intensity, non_tooth_right_ratio, idx_cej_r
        )

        t_cej_l = float(t_grid[idx_cej_l])
        t_cej_r = float(t_grid[idx_cej_r])
        t_crest_l = float(t_grid[idx_crest_l])
        t_crest_r = float(t_grid[idx_crest_r])

        mesial_cej, _ = self._find_left_right_points_from_mask(tooth_mask, points, t_to_y(t_cej_l), local_cx)
        _, distal_cej = self._find_left_right_points_from_mask(tooth_mask, points, t_to_y(t_cej_r), local_cx)
        mesial_crest, _ = self._find_left_right_points_from_mask(tooth_mask, points, t_to_y(t_crest_l), local_cx)
        _, distal_crest = self._find_left_right_points_from_mask(tooth_mask, points, t_to_y(t_crest_r), local_cx)
        avg_conf = round((conf_l + conf_r) / 2.0, 4)

        return mesial_cej, distal_cej, mesial_crest, distal_crest, apex, avg_conf

    def predict_landmarks_from_contour(
        self,
        image_rgb: np.ndarray,
        contour_pts: np.ndarray,
        bbox: list,
        tooth_number: int = None,
        external_teeth_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        치아 분할 컨투어(Dental_008 등)와 원본 파노라마 영상을 결합하여
        SAM 재호출 없이 즉시 방사선 밀도 기반 5대 치주 랜드마크를 산출합니다.
        """
        if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            # [x1, y1, x2, y2] -> [cx, cy, w, h]
            x1, y1, x2, y2 = bbox
            cx, cy, w, h = (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
        else:
            cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        cropped_image, offset_x, offset_y = crop_roi(image_rgb, [cx, cy, w, h], padding=28)
        ph, pw = cropped_image.shape[:2]
        if ph < 8 or pw < 8:
            return self._get_fallback_landmarks(cx, cy, w, h)

        local_cx = cx - offset_x
        pts = np.asarray(contour_pts, dtype=np.float32).reshape(-1, 2).copy()
        if pts.shape[0] < 6:
            return self._get_fallback_landmarks(cx, cy, w, h)

        pts[:, 0] = np.clip(pts[:, 0] - offset_x, 0, pw - 1)
        pts[:, 1] = np.clip(pts[:, 1] - offset_y, 0, ph - 1)

        tooth_mask = np.zeros((ph, pw), dtype=np.uint8)
        cv2.fillPoly(tooth_mask, [pts.astype(np.int32)], 255)
        patch_gray = self._to_clahe_gray(cropped_image)

        all_teeth_patch = None
        if external_teeth_mask is not None:
            all_teeth_patch = external_teeth_mask[offset_y : offset_y + ph, offset_x : offset_x + pw]

        if tooth_number is not None:
            is_upper = (10 <= tooth_number < 30) or (50 <= tooth_number < 70)
        else:
            is_upper = cy < (image_rgb.shape[0] / 2.0)

        m_cej, d_cej, m_crest, d_crest, apex, conf = self._extract_radiographic_landmarks(
            patch_gray, tooth_mask, pts, local_cx, is_upper, all_teeth_mask=all_teeth_patch
        )

        def to_global(pt: Tuple[float, float]) -> Tuple[float, float]:
            return (float(pt[0] + offset_x), float(pt[1] + offset_y))

        return {
            "mesial_cej": to_global(m_cej),
            "distal_cej": to_global(d_cej),
            "mesial_crest": to_global(m_crest),
            "distal_crest": to_global(d_crest),
            "root_apex": to_global(apex),
            "confidence": conf,
        }

    def predict_landmarks(
        self, image_rgb: np.ndarray, bbox: list, tooth_number: int = None
    ) -> Dict[str, Tuple[float, float]]:
        """
        SAM을 이용하여 치아 마스크를 추출하고, 방사선 불투과도 경사 및 치경부 형태 분석을 통해
        5개의 핵심 치주 랜드마크를 추론합니다.
        """
        self._init_sam()
        cx, cy, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        cropped_image, offset_x, offset_y = crop_roi(image_rgb, bbox, padding=20)
        ph, pw = cropped_image.shape[:2]
        if ph < 8 or pw < 8:
            return self._get_fallback_landmarks(cx, cy, w, h)

        local_cx = cx - offset_x
        local_cy = cy - offset_y

        x_min = max(0, local_cx - w / 2)
        y_min = max(0, local_cy - h / 2)
        x_max = min(pw, local_cx + w / 2)
        y_max = min(ph, local_cy + h / 2)
        input_box = np.array([x_min, y_min, x_max, y_max])

        if self.predictor is not None:
            self.predictor.set_image(cropped_image)
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            mask = masks[0].astype(np.uint8) * 255
        else:
            # Otsu + morphological contour fallback if segment-anything is not installed
            gray_crop = self._to_clahe_gray(cropped_image)
            _, mask = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._get_fallback_landmarks(cx, cy, w, h)

        largest_contour = max(contours, key=cv2.contourArea)
        points = largest_contour.squeeze(1)
        if points.ndim != 2 or points.shape[0] < 6:
            return self._get_fallback_landmarks(cx, cy, w, h)

        if tooth_number is not None:
            is_upper = (10 <= tooth_number < 30) or (50 <= tooth_number < 70)
        else:
            is_upper = cy < (image_rgb.shape[0] / 2.0)

        patch_gray = self._to_clahe_gray(cropped_image)
        m_cej, d_cej, m_crest, d_crest, apex, conf = self._extract_radiographic_landmarks(
            patch_gray, mask, points, local_cx, is_upper
        )

        def to_global(pt: Tuple[float, float]) -> Tuple[float, float]:
            return (float(pt[0] + offset_x), float(pt[1] + offset_y))

        return {
            "mesial_cej": to_global(m_cej),
            "distal_cej": to_global(d_cej),
            "mesial_crest": to_global(m_crest),
            "distal_crest": to_global(d_crest),
            "root_apex": to_global(apex),
            "confidence": conf,
        }

    def _find_left_right_points_from_mask(
        self,
        tooth_mask: np.ndarray,
        points: np.ndarray,
        target_y: float,
        cx: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        ph = tooth_mask.shape[0]
        yi_min = max(0, int(round(target_y)) - 2)
        yi_max = min(ph, int(round(target_y)) + 3)
        if yi_max > yi_min:
            cols = np.where(tooth_mask[yi_min:yi_max, :] > 0)[1]
            if cols.size > 0:
                return (float(np.min(cols)), float(target_y)), (float(np.max(cols)), float(target_y))
        return self._find_left_right_points(points, target_y, cx)

    def _find_left_right_points(
        self, points: np.ndarray, target_y: float, cx: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        y_dists = np.abs(points[:, 1] - target_y)
        close_indices = np.argsort(y_dists)[:20]
        close_points = points[close_indices]

        if len(close_points) >= 2:
            left_pt = close_points[int(np.argmin(close_points[:, 0]))]
            right_pt = close_points[int(np.argmax(close_points[:, 0]))]
            return (float(left_pt[0]), float(target_y)), (float(right_pt[0]), float(target_y))

        return (float(cx - 15.0), float(target_y)), (float(cx + 15.0), float(target_y))

    def _get_fallback_landmarks(self, cx: float, cy: float, w: float, h: float):
        return {
            "mesial_cej": (float(cx - w / 4.0), float(cy)),
            "distal_cej": (float(cx + w / 4.0), float(cy)),
            "mesial_crest": (float(cx - w / 4.0), float(cy + h / 12.0)),
            "distal_crest": (float(cx + w / 4.0), float(cy + h / 12.0)),
            "root_apex": (float(cx), float(cy + h / 2.0)),
            "confidence": 0.50,
        }
