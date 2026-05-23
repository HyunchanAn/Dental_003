"""
Streamlit Cloud Deployment UI for Pano_BoneLoss_Measurement.

이 모듈은 웹 브라우저 상에서 방사선 이미지를 업로드받고, 
내부 딥러닝 모듈과 기하학 연산 로직을 호출하여 결과를 시각화합니다.
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import time

from models.detector import ToothDetector
from models.landmark import PerioLandmarkPredictor
from utils.geometry import calculate_rbl
from services.staging import determine_patient_stage, Stage, Extent

# =====================================================================
# 페이지 설정 및 CSS
# =====================================================================
st.set_page_config(
    page_title="Pano BoneLoss Measurement",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 모델 초기화 (Streamlit Caching 활용)
# =====================================================================
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = ToothDetector(weights_path="dummy_yolo.pt", device=device)
    landmark_predictor = PerioLandmarkPredictor(backbone_type="swin").to(device)
    landmark_predictor.eval()
    return detector, landmark_predictor, device

detector, landmark_predictor, device = load_models()

# =====================================================================
# UI 레이아웃
# =====================================================================
st.title("🦷 Pano BoneLoss Measurement System")
st.markdown("---")

# 사이드바: 컨트롤 패널
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "방사선 이미지 업로드 (Panoramic/Periapical)", 
        type=["png", "jpg", "jpeg", "dicom"]
    )
    
    st.markdown("---")
    st.info("""
    **💡 판별 근거**
    - 이 시스템은 학습된 딥러닝 모델의 가중치(.pt)가 연동되어 동작하는 것을 전제로 설계되었습니다.
    - 현재 UI 구조 상에서는 **더미 텐서 모델(YOLOv11, Swin Transformer 구조 기반)**을 호출하여 임의의 예측 좌표를 반환하고 있습니다.
    - 실제 현장 도입 시 `models/detector.py` 등에 실제 학습된 PyTorch 모델 로드 코드를 연결해야 합니다.
    """)

# 메인 화면: 결과 표출
if uploaded_file is not None:
    # 1. 이미지 로드 및 렌더링
    image_bytes = uploaded_file.read()
    # OpenCV 처리를 위해 numpy 배열로 변환
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    
    st.subheader("Original Radiograph")
    st.image(img_rgb, use_container_width=True)
    
    with st.spinner('딥러닝 모델 추론 및 랜드마크 분석 중...'):
        time.sleep(1) # 더미 로딩 시간
        
        # 2. 모델 추론 파이프라인 (실제 텐서 변환 로직 필요)
        dummy_tensor = torch.randn(1, 3, 1024, 2048).to(device)
        detections = detector.predict(dummy_tensor)
        
        tooth_metrics = []
        table_data = []
        
        # 시각화를 위한 복제 이미지
        overlay_img = img_rgb.copy()
        
        for det in detections:
            tooth_num = det["tooth_number"]
            bbox = det["bbox"] # [x_center, y_center, w, h, angle]
            
            # 랜드마크 추출
            crop_tensor = torch.randn(1, 3, 224, 224).to(device)
            landmarks = landmark_predictor.predict_landmarks(crop_tensor)
            
            # RBL 연산
            mesial_rbl = calculate_rbl(landmarks["mesial_cej"], landmarks["mesial_crest"], landmarks["root_apex"])
            distal_rbl = calculate_rbl(landmarks["distal_cej"], landmarks["distal_crest"], landmarks["root_apex"])
            max_rbl = max(mesial_rbl, distal_rbl)
            
            tooth_metrics.append({"tooth": tooth_num, "max_rbl": max_rbl})
            
            table_data.append({
                "Tooth (FDI)": tooth_num,
                "Mesial RBL (%)": round(mesial_rbl, 1),
                "Distal RBL (%)": round(distal_rbl, 1),
                "Max RBL (%)": round(max_rbl, 1),
                "Status": "Normal" if max_rbl == 0 else ("Warning" if max_rbl < 33 else "Severe")
            })
            
            # [시각화] 원본 이미지 해상도가 고정되어 있지 않으므로 
            # 실제 구현에서는 좌표를 원본 스케일로 역변환해야 합니다.
            # 여기서는 더미 시각화로 대체합니다.
            cx, cy = int(bbox[0]), int(bbox[1])
            cv2.putText(overlay_img, f"#{tooth_num}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 3. 환자 병기 판별
        final_stage, extent = determine_patient_stage(tooth_metrics)
        
    st.markdown("---")
    st.subheader("Analysis Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Global Periodontitis Stage", value=final_stage.value)
    with col2:
        st.metric(label="Disease Extent", value=extent.value)
        
    st.markdown("#### Tooth-level Metrics")
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)
    
    st.markdown("#### Detection Overlay")
    st.image(overlay_img, caption="Tooth Bounding Boxes & Landmarks (Dummy Visualization)", use_container_width=True)

else:
    st.info("왼쪽 사이드바에서 분석할 방사선 이미지를 업로드해 주세요.")
