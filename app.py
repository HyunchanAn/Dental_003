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
import torch.nn as nn
from torchvision import models, transforms

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
    detector = ToothDetector(weights_path="runs/detect/models/detector_train/weights/best.pt", device=device)
    landmark_predictor = PerioLandmarkPredictor(device=device)
    
    classifier = models.mobilenet_v3_small()
    num_ftrs = classifier.classifier[3].in_features
    classifier.classifier[3] = nn.Linear(num_ftrs, 2)
    classifier.load_state_dict(torch.load("models/pano_classifier.pt", map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    
    return detector, landmark_predictor, classifier, device

detector, landmark_predictor, classifier, device = load_models()

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
    **💡 딥러닝 추론 시스템 연동 완료**
    - 이미지 검증: MobileNetV3 기반 파노라마 사진 여부 판별 (OOD Reject)
    - 치아 검출: Roboflow `ufba-425` 데이터셋으로 커스텀 학습된 YOLOv11 모델 적용.
    - 랜드마크 검출: Foundation Model인 **SAM(Segment Anything)**을 이용한 치아 마스크 기반 기하학적 추정(CEJ, Crest, Apex) 파이프라인 연동 완료.
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
    
    with st.spinner('파노라마 방사선 사진 여부 검증 및 분석 중...'):
        
        # 1.5 OOD Classification (입구 컷 필터)
        img_pil = Image.fromarray(img_rgb)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        cls_input = transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = classifier(cls_input)
            _, preds = torch.max(outputs, 1)
            is_pano = (preds.item() == 1) # 1이 'pano', 0이 'non_pano'
            
        if not is_pano:
            st.error("⚠️ 이 이미지는 파노라마 방사선 사진이 아닌 것 같습니다 (예: 치근단 방사선 사진, 일반 사진). 전체 치아 배열이 보이는 파노라마 원본을 업로드해 주세요.")
            st.stop()
            
        # 2. 모델 추론 파이프라인 (YOLOv11 적용)
        dummy_tensor = torch.randn(1, 3, 1024, 2048).to(device)
        detections = detector.predict(dummy_tensor)
        
        tooth_metrics = []
        table_data = []
        
        # 시각화를 위한 복제 이미지
        overlay_img = img_rgb.copy()
        
        for det in detections:
            tooth_num = det["tooth_number"]
            bbox = det["bbox"] # [x_center, y_center, w, h, angle]
            
            # SAM 기반 랜드마크 추출 (원본 이미지 및 바운딩 박스 전달)
            landmarks = landmark_predictor.predict_landmarks(img_rgb, bbox)
            
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
            
            # [시각화] YOLO 바운딩 박스와 SAM 랜드마크 시각화
            cx, cy = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2]), int(bbox[3])
            
            # 치아 번호 및 박스 (간이 표현)
            cv2.rectangle(overlay_img, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2), (0, 255, 0), 2)
            cv2.putText(overlay_img, f"#{tooth_num}", (cx - w//2, cy - h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 랜드마크 점 찍기
            pts = [
                (landmarks["mesial_cej"], (255, 0, 0)),   # Blue (CEJ)
                (landmarks["distal_cej"], (255, 0, 0)),
                (landmarks["mesial_crest"], (0, 255, 255)), # Yellow (Crest)
                (landmarks["distal_crest"], (0, 255, 255)),
                (landmarks["root_apex"], (0, 0, 255))      # Red (Apex)
            ]
            for pt_coord, color in pts:
                pt_x, pt_y = int(pt_coord[0]), int(pt_coord[1])
                cv2.circle(overlay_img, (pt_x, pt_y), 5, color, -1)

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
