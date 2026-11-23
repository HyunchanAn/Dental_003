import io
import os
import sys
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api.main import app
from dental_core.schemas.panorama_contract import (
    PanoramaHiTLDocument,
    PanoramaImageMeta,
    ToothAnnotation,
)
from models.landmark import PerioLandmarkPredictor
from services.rbl_008_linker import measure_rbl_from_008_hitl_doc
from services.staging import Extent, Stage, determine_patient_stage
from utils.calibration import CalibrationManager
from utils.geometry import calculate_axial_rbl, calculate_distance, calculate_rbl


def test_geometry_distance_and_rbl():
    assert calculate_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert calculate_rbl(cej=(100.0, 100.0), crest=(100.0, 100.0), apex=(100.0, 200.0)) == pytest.approx(0.0)
    assert calculate_rbl(cej=(100.0, 100.0), crest=(100.0, 115.0), apex=(100.0, 200.0)) == pytest.approx(15.0)
    assert calculate_rbl(cej=(100.0, 100.0), crest=(100.0, 145.0), apex=(100.0, 200.0)) == pytest.approx(45.0)


def test_staging_classification():
    metrics = [
        {"tooth": 11, "max_rbl": 10.0},
        {"tooth": 16, "max_rbl": 28.0},
        {"tooth": 26, "max_rbl": 42.0},
    ]
    stage, extent = determine_patient_stage(metrics, has_severe_complexity=False, teeth_lost_due_to_perio=0)
    assert stage == Stage.STAGE_III
    assert extent in (Extent.LOCALIZED, Extent.GENERALIZED)


def test_predict_landmarks_from_contour_radiodensity():
    predictor = PerioLandmarkPredictor(device="cpu", lazy_sam=True)
    img_rgb = np.full((300, 300, 3), 40, dtype=np.uint8)
    cv2.rectangle(img_rgb, (120, 80), (180, 140), (235, 235, 235), -1)
    cv2.rectangle(img_rgb, (130, 140), (170, 230), (190, 190, 190), -1)
    cv2.rectangle(img_rgb, (80, 175), (220, 250), (165, 165, 165), -1)

    contour = np.array(
        [[120, 80], [180, 80], [170, 230], [150, 240], [130, 230], [120, 140]],
        dtype=np.float32,
    )
    landmarks = predictor.predict_landmarks_from_contour(
        image_rgb=img_rgb,
        contour_pts=contour,
        bbox=[120, 80, 180, 240],
        tooth_number=36,
    )
    for key in ("mesial_cej", "distal_cej", "mesial_crest", "distal_crest", "root_apex", "confidence"):
        assert key in landmarks
    assert landmarks["root_apex"][1] > landmarks["mesial_cej"][1]
    assert 0.5 <= landmarks["confidence"] <= 1.0


def test_fastapi_predict_pano_smoke_and_503_handling():
    client = TestClient(app)
    dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", dummy_img)
    assert ok

    mock_detector = MagicMock()
    mock_detector.predict.return_value = [
        {"tooth_number": 36, "bbox": [80.0, 80.0, 40.0, 80.0]}
    ]
    mock_landmark = MagicMock()
    mock_landmark.predict_landmarks.return_value = {
        "mesial_cej": (65.0, 50.0),
        "distal_cej": (95.0, 50.0),
        "mesial_crest": (65.0, 85.0),
        "distal_crest": (95.0, 80.0),
        "root_apex": (80.0, 140.0),
        "confidence": 0.85,
    }

    with patch("api.main.get_models", return_value=(mock_detector, mock_landmark)):
        resp = client.post(
            "/predict/pano",
            files={"file": ("pano.png", io.BytesIO(buf.tobytes()), "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["measurement_method"] == "radiographic_densitometric_cej_crest_v2"
        assert len(body["tooth_level_results"]) == 1
        assert body["tooth_level_results"][0]["tooth_number"] == 36


def test_axial_rbl_projection_prevents_horizontal_offset_inflation():
    # Vertical axis (0, 1): CEJ=(100, 100), Apex=(100, 200), Crest=(130, 120) -> horizontal offset 30px, axial loss 20px/100px = 20%
    rbl_pct, loss_px, root_px = calculate_axial_rbl(
        cej=(100.0, 100.0),
        crest=(130.0, 120.0),
        apex=(100.0, 200.0),
        axis_unit_vec=[0.0, 1.0],
    )
    assert rbl_pct == pytest.approx(20.0)
    assert loss_px == pytest.approx(20.0)
    assert root_px == pytest.approx(100.0)
    # Euclidean RBL would have been ~36.06% due to horizontal interproximal offset
    assert calculate_rbl((100.0, 100.0), (130.0, 120.0), (100.0, 200.0)) > 35.0


def test_measure_rbl_from_008_hitl_doc_calibrated_and_uncalibrated():
    img_rgb = np.full((320, 480, 3), 45, dtype=np.uint8)
    cv2.rectangle(img_rgb, (120, 80), (180, 140), (235, 235, 235), -1)
    cv2.rectangle(img_rgb, (130, 140), (170, 230), (190, 190, 190), -1)
    cv2.rectangle(img_rgb, (80, 175), (220, 250), (165, 165, 165), -1)

    doc_cal = PanoramaHiTLDocument(
        image_meta=PanoramaImageMeta(
            filename="calibrated_pano.dcm",
            width=480,
            height=320,
            source_format="DICOM",
            pixel_spacing_mm=(0.08, 0.08),
            is_calibrated=True,
        ),
        teeth=[
            ToothAnnotation(
                fdi=36,
                bbox_xyxy=[120.0, 80.0, 180.0, 240.0],
                polygon_px=[[120, 80], [180, 80], [170, 230], [150, 240], [130, 230], [120, 140]],
            ),
            ToothAnnotation(
                fdi=46,
                bbox_xyxy=[300.0, 85.0, 355.0, 235.0],
                restoration_tags=["Implant"],
            ),
        ],
    )

    res_cal = measure_rbl_from_008_hitl_doc(img_rgb, doc_cal)
    assert res_cal["mm_calibrated"] is True
    assert res_cal["pixels_per_mm"] == pytest.approx(12.5)
    assert res_cal["skipped_implant_fdi"] == [46]
    assert len(res_cal["teeth_metrics"]) == 1
    assert res_cal["teeth_metrics"][0]["fdi"] == 36
    assert res_cal["teeth_metrics"][0]["mm_calibrated"] is True
    assert len(doc_cal.bone_loss) == 1
    assert doc_cal.bone_loss[0].mm_calibrated is True

    # Uncalibrated PNG fallback check
    cal_mgr = CalibrationManager()
    assert cal_mgr.set_scale_from_panorama_meta({"pixel_spacing_mm": None, "is_calibrated": False}) is False
    assert cal_mgr.is_calibrated is False
