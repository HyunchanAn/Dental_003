from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from dental_core.schemas.panorama_contract import (
    PERMANENT_FDI,
    BoneLossAnnotation,
    PanoramaHiTLDocument,
    ToothAnnotation,
    compute_tooth_axis_and_apex,
)
from models.landmark import PerioLandmarkPredictor
from services.staging import determine_patient_stage, determine_site_stage
from utils.calibration import CalibrationManager
from utils.geometry import calculate_axial_rbl


def _extract_teeth_list(
    doc_or_teeth: Union[PanoramaHiTLDocument, Dict[Any, Any], Sequence[Any]]
) -> List[ToothAnnotation]:
    if isinstance(doc_or_teeth, PanoramaHiTLDocument):
        return list(doc_or_teeth.teeth)

    out: List[ToothAnnotation] = []
    if isinstance(doc_or_teeth, dict):
        for k, val in doc_or_teeth.items():
            try:
                fdi_int = int(k)
            except (TypeError, ValueError):
                continue
            if not isinstance(val, dict):
                continue
            bbox = val.get("bbox_xyxy") or val.get("bbox")
            poly = val.get("polygon_px") or val.get("contour") or []
            if bbox is None and len(poly) >= 3:
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                bbox = [
                    float(np.min(pts[:, 0])),
                    float(np.min(pts[:, 1])),
                    float(np.max(pts[:, 0])),
                    float(np.max(pts[:, 1])),
                ]
            if bbox is None or len(bbox) < 4:
                continue
            pts_list = np.asarray(poly, dtype=np.float32).reshape(-1, 2).tolist() if len(poly) >= 3 else []
            out.append(
                ToothAnnotation(
                    fdi=fdi_int,
                    bbox_xyxy=[float(x) for x in bbox[:4]],
                    polygon_px=pts_list,
                    confidence=float(val.get("confidence", val.get("conf", 1.0))),
                    restoration_tags=list(val.get("restoration_tags", [])),
                )
            )
        return out

    for item in doc_or_teeth:
        if isinstance(item, ToothAnnotation):
            out.append(item)
        elif isinstance(item, dict):
            fdi_int = int(item.get("fdi", item.get("tooth_number", 0)))
            bbox = item.get("bbox_xyxy") or item.get("bbox")
            poly = item.get("polygon_px") or item.get("contour") or []
            if bbox is None or len(bbox) < 4:
                continue
            pts_list = np.asarray(poly, dtype=np.float32).reshape(-1, 2).tolist() if len(poly) >= 3 else []
            out.append(
                ToothAnnotation(
                    fdi=fdi_int,
                    bbox_xyxy=[float(x) for x in bbox[:4]],
                    polygon_px=pts_list,
                    confidence=float(item.get("confidence", item.get("conf", 1.0))),
                    restoration_tags=list(item.get("restoration_tags", [])),
                )
            )
    return out


def measure_rbl_from_008_hitl_doc(
    image_rgb: np.ndarray,
    doc_or_teeth: Union[PanoramaHiTLDocument, Dict[Any, Any], Sequence[Any]],
    pixel_spacing_mm: Optional[Sequence[float]] = None,
    is_calibrated: Optional[bool] = None,
    nominal_mm_per_px: float = 0.1,
    landmark_predictor: Optional[PerioLandmarkPredictor] = None,
    teeth_lost_due_to_perio: int = 0,
) -> Dict[str, Any]:
    """
    Compute tooth-axis-projected Radiographic Bone Loss (RBL %) and physical bone loss (mm)
    directly from Dental_008 tooth segmentation (`PanoramaHiTLDocument` or `008_tooth_data`)
    and Dental_Core calibration metadata (`pixel_spacing_mm`, `is_calibrated`).
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("image_rgb must be a non-empty image array.")

    img_h, img_w = image_rgb.shape[:2]
    calibrator = CalibrationManager()

    if isinstance(doc_or_teeth, PanoramaHiTLDocument) and pixel_spacing_mm is None:
        calibrator.set_scale_from_panorama_meta(
            doc_or_teeth.image_meta, nominal_mm_per_px=nominal_mm_per_px
        )
    elif pixel_spacing_mm is not None and (is_calibrated is None or is_calibrated):
        if not calibrator.set_scale_from_dicom(pixel_spacing_mm):
            calibrator.set_scale_from_panorama_meta(None, nominal_mm_per_px=nominal_mm_per_px)
    else:
        calibrator.set_scale_from_panorama_meta(None, nominal_mm_per_px=nominal_mm_per_px)

    teeth = _extract_teeth_list(doc_or_teeth)
    predictor = landmark_predictor or PerioLandmarkPredictor(device="cpu", lazy_sam=True)

    all_teeth_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for t in teeth:
        poly = t.ensure_polygon()
        if len(poly) >= 3:
            pts_i32 = np.round(np.asarray(poly, dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(all_teeth_mask, [pts_i32], 255)

    tooth_results: List[Dict[str, Any]] = []
    staging_inputs: List[Dict[str, float]] = []
    bone_loss_annotations: List[BoneLossAnnotation] = []
    skipped_implants: List[int] = []

    for tooth in teeth:
        fdi = int(tooth.fdi)
        if fdi not in PERMANENT_FDI:
            continue
        if "Implant" in tooth.restoration_tags:
            skipped_implants.append(fdi)
            continue

        poly = tooth.ensure_polygon()
        contour_arr = np.asarray(poly, dtype=np.float32)
        axis_info = compute_tooth_axis_and_apex(fdi, tooth.bbox_xyxy, poly)
        axis_vec = axis_info["axis_unit_vec"]

        landmarks = predictor.predict_landmarks_from_contour(
            image_rgb=image_rgb,
            contour_pts=contour_arr,
            bbox=tooth.bbox_xyxy,
            tooth_number=fdi,
            external_teeth_mask=all_teeth_mask,
        )

        m_cej = landmarks["mesial_cej"]
        d_cej = landmarks["distal_cej"]
        m_crest = landmarks["mesial_crest"]
        d_crest = landmarks["distal_crest"]
        apex = landmarks["root_apex"]
        lm_conf = float(landmarks.get("confidence", 0.75))

        try:
            m_rbl, m_loss_px, _ = calculate_axial_rbl(m_cej, m_crest, apex, axis_unit_vec=axis_vec)
            d_rbl, d_loss_px, _ = calculate_axial_rbl(d_cej, d_crest, apex, axis_unit_vec=axis_vec)
        except ValueError:
            continue

        if m_rbl >= d_rbl:
            max_rbl = m_rbl
            worst_loss_px = m_loss_px
        else:
            max_rbl = d_rbl
            worst_loss_px = d_loss_px

        loss_mm = round(calibrator.pixel_to_mm(worst_loss_px), 2)
        site_stage = determine_site_stage(max_rbl).value
        quad_poly = [
            [round(float(m_cej[0]), 2), round(float(m_cej[1]), 2)],
            [round(float(d_cej[0]), 2), round(float(d_cej[1]), 2)],
            [round(float(d_crest[0]), 2), round(float(d_crest[1]), 2)],
            [round(float(m_crest[0]), 2), round(float(m_crest[1]), 2)],
        ]

        staging_inputs.append({"tooth": fdi, "max_rbl": max_rbl})
        entry = {
            "fdi": fdi,
            "mesial_rbl_percent": m_rbl,
            "distal_rbl_percent": d_rbl,
            "rbl_percent": max_rbl,
            "loss_mm": loss_mm,
            "mm_calibrated": calibrator.is_calibrated,
            "tilt_deg": axis_info["tilt_deg"],
            "stage": site_stage,
            "confidence": round(lm_conf, 4),
            "polygon_px": quad_poly,
            "landmarks": {
                "mesial_cej": [round(float(m_cej[0]), 2), round(float(m_cej[1]), 2)],
                "distal_cej": [round(float(d_cej[0]), 2), round(float(d_cej[1]), 2)],
                "mesial_crest": [round(float(m_crest[0]), 2), round(float(m_crest[1]), 2)],
                "distal_crest": [round(float(d_crest[0]), 2), round(float(d_crest[1]), 2)],
                "root_apex": [round(float(apex[0]), 2), round(float(apex[1]), 2)],
            },
        }
        tooth_results.append(entry)

        bone_loss_annotations.append(
            BoneLossAnnotation(
                fdi=fdi,
                polygon_px=quad_poly,
                rbl_percent=max_rbl,
                loss_mm=loss_mm,
                mm_calibrated=calibrator.is_calibrated,
                stage=site_stage,
                confidence=lm_conf,
            )
        )

    patient_stage, patient_extent = determine_patient_stage(
        staging_inputs,
        has_severe_complexity=False,
        teeth_lost_due_to_perio=teeth_lost_due_to_perio,
    )

    if isinstance(doc_or_teeth, PanoramaHiTLDocument):
        doc_or_teeth.bone_loss = bone_loss_annotations

    return {
        "measurement_method": "axial_projected_densitometric_cej_crest_v3",
        "mm_calibrated": calibrator.is_calibrated,
        "pixels_per_mm": round(calibrator.pixels_per_mm, 4),
        "patient_stage": patient_stage.value,
        "patient_extent": patient_extent.value,
        "skipped_implant_fdi": sorted(skipped_implants),
        "teeth_metrics": tooth_results,
    }
