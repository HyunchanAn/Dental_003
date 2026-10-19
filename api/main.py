"""
FastAPI Server Module for Pano_BoneLoss_Measurement.

방사선 이미지를 업로드받아 치아 검출 -> 랜드마크 추출 -> RBL 계산 -> 병기 판별
전체 파이프라인을 실행하고 결과를 JSON 형태로 반환합니다.
"""

import os
import sys
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# 내부 모듈 임포트 (패키지 절대/상대 경로 조정)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.detector import ToothDetector
from models.landmark import PerioLandmarkPredictor
from utils.geometry import calculate_rbl
from services.staging import determine_patient_stage

app = FastAPI(
    title="Pano_BoneLoss_Measurement API",
    version="1.0.0",
    description="Perio Staging & Bone Loss Auto-Measurement System API",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
_detector: Optional[ToothDetector] = None
_landmark_predictor: Optional[PerioLandmarkPredictor] = None


def resolve_detector_weights() -> Optional[str]:
    """
    로컬 가중치 경로를 우선 탐색하고, 부재 시 Hugging Face Hub에서 가중치를 다운로드합니다.
    더미 가중치(dummy_yolo.pt)는 절대 사용하지 않습니다.
    """
    local_candidates = [
        os.path.join(BASE_DIR, "weights", "boneloss_yolov8m_obb.pt"),
        os.path.join(BASE_DIR, "weights", "best.onnx"),
        os.path.join(BASE_DIR, "weights", "best.pt"),
        os.path.join(BASE_DIR, "runs", "detect", "models", "detector_train", "weights", "best.onnx"),
        os.path.join(BASE_DIR, "runs", "detect", "models", "detector_train", "weights", "best.pt"),
    ]
    for path in local_candidates:
        if os.path.exists(path):
            return path

    if os.environ.get("DENTAL_003_SKIP_HF_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return None

    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id="chemahc94/Dental-AI-Models",
            filename="Dental_003/weights/boneloss_yolov8m_obb.pt",
        )
    except Exception:
        pass

    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id="chemahc94/pano-boneloss-weights",
            filename="best.pt",
        )
    except Exception:
        return None


def resolve_sam_weights() -> Optional[str]:
    """
    로컬 SAM 체크포인트(sam_b.pt 또는 sam_vit_b_01ec64.pth)를 탐색합니다.
    """
    local_candidates = [
        os.path.join(BASE_DIR, "weights", "sam_b.pt"),
        os.path.join(BASE_DIR, "models", "sam_vit_b_01ec64.pth"),
        os.path.join(BASE_DIR, "sam_b.pt"),
    ]
    for path in local_candidates:
        if os.path.exists(path):
            return path

    if os.environ.get("DENTAL_003_SKIP_HF_DOWNLOAD", "").lower() in ("1", "true", "yes"):
        return None

    return os.path.join(BASE_DIR, "models", "sam_vit_b_01ec64.pth")


def get_models() -> Tuple[ToothDetector, PerioLandmarkPredictor]:
    """
    요청 시점에 모델을 지연 로드(lazy-load)하며, 가중치 부재 시 HTTP 503 예외를 발생시킵니다.
    """
    global _detector, _landmark_predictor

    if _detector is not None and _landmark_predictor is not None:
        return _detector, _landmark_predictor

    yolo_weight_path = resolve_detector_weights()
    sam_weight_path = resolve_sam_weights()

    if not yolo_weight_path or not os.path.exists(yolo_weight_path) or not sam_weight_path:
        raise HTTPException(status_code=503, detail="Model weights unavailable")

    try:
        if _detector is None:
            _detector = ToothDetector(weights_path=yolo_weight_path, device=device)
        if _landmark_predictor is None:
            _landmark_predictor = PerioLandmarkPredictor(
                device=device,
                checkpoint_path=sam_weight_path,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model weights unavailable: {str(exc)}",
        ) from exc

    return _detector, _landmark_predictor


async def _run_panoramic_inference(file: UploadFile) -> JSONResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detector, landmark_predictor = get_models()

    try:
        detections = detector.predict(img_rgb)

        tooth_metrics = []
        detailed_results = []

        for det in detections:
            tooth_num = det["tooth_number"]
            bbox = det["bbox"]
            landmarks = landmark_predictor.predict_landmarks(
                img_rgb,
                bbox,
                tooth_number=tooth_num,
            )

            mesial_rbl = calculate_rbl(
                cej=landmarks["mesial_cej"],
                crest=landmarks["mesial_crest"],
                apex=landmarks["root_apex"],
            )
            distal_rbl = calculate_rbl(
                cej=landmarks["distal_cej"],
                crest=landmarks["distal_crest"],
                apex=landmarks["root_apex"],
            )

            max_rbl = max(mesial_rbl, distal_rbl)
            tooth_metrics.append({"tooth": tooth_num, "max_rbl": max_rbl})

            detailed_results.append({
                "tooth_number": tooth_num,
                "bbox": bbox,
                "landmarks": landmarks,
                "mesial_rbl_percent": round(mesial_rbl, 2),
                "distal_rbl_percent": round(distal_rbl, 2),
                "max_rbl_percent": round(max_rbl, 2),
            })

        final_stage, extent = determine_patient_stage(
            tooth_metrics=tooth_metrics,
            has_severe_complexity=False,
            teeth_lost_due_to_perio=0,
        )

        return JSONResponse(content={
            "status": "success",
            "measurement_method": "geometric_heuristic_30_40_rule",
            "model_version": "yolov8m-obb+sam-b",
            "global_diagnosis": {
                "stage": final_stage.value,
                "extent": extent.value,
            },
            "tooth_level_results": detailed_results,
        })

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}") from exc


@app.post("/predict")
async def predict_root(file: UploadFile = File(...)):
    """
    파노라마 방사선 이미지를 입력받아 치아별 골소실과 환자의 치주염 병기를 판별합니다.
    """
    return await _run_panoramic_inference(file)


@app.post("/predict/pano")
async def predict_panoramic(file: UploadFile = File(...)):
    """
    파노라마 방사선 이미지를 입력받아 치아별 골소실과 환자의 치주염 병기를 판별합니다.
    """
    return await _run_panoramic_inference(file)


@app.post("/predict/periapical")
async def predict_periapical(file: UploadFile = File(...)):
    """
    치근단(Periapical) 방사선 이미지를 분석합니다.
    """
    return {"message": "Periapical analysis endpoint is initialized for future integration."}
