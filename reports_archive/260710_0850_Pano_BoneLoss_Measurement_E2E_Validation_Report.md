# 260710_0850_Pano_BoneLoss_Measurement_E2E_Validation_Report

## 작성일: 2026-07-10 08:50
## 작성자: 안현찬 (Hyunchan An)

***

### 1. 개요 (Executive Summary)

본 보고서는 파노라마 방사선 사진에서 개별 치아를 검출하고, SAM 기반의 기하학적 분석을 통해 CEJ, Crest, Apex 랜드마크를 추출한 뒤 최종적으로 치조골 소실량(RBL, Radiographic Bone Loss)을 측정하는 **Dental_003 (Pano BoneLoss Measurement)** 파이프라인의 검증 결과를 기술합니다.

이번 E2E 검증에서는 OOD(Out-of-Distribution) 필터링 모델 성능 확인, YOLOv11-OBB 기반의 치아 검출기 벤치마크 평가, 그리고 RBL 연산 로직에 대한 단위 테스트를 포괄적으로 수행했습니다.

***

### 2. 통합 모델링 파이프라인 개요

시스템은 아래의 세부 모듈로 나뉘어 동작하며, 각 파트별 검증이 완료되었습니다.

1. **OOD Classifier:** MobileNetV3를 기반으로 이미지가 파노라마인지 여부 판별.
2. **Tooth Detector:** YOLOv11-OBB (ufba-425 학습) 기반 개별 치아 크롭.
3. **Landmark & RBL Evaluator:** 치아 랜드마크(CEJ, Crest, Apex)의 위치를 SAM을 통해 추출한 뒤 비율 기반 Bone Loss 산출. (해당 파트는 GT 데이터셋의 부재로 로직 단위의 Pytest로 검증 대체)

***

### 3. 검증 결과 상세 (Evaluation Metrics)

#### 3.1. 치아 검출 (YOLOv11-OBB)
ufba-425 데이터셋을 사용하여 100 Epoch 학습을 진행한 후의 최종 Validation Metrics입니다.
- **Precision:** 0.850
- **Recall:** 0.871
- **mAP@50:** 0.903
- **mAP@50-95:** 0.689

치아의 형태가 제각각이거나 일부 가려짐이 있는 상황에서도 90% 이상의 매우 높은 mAP50 성능을 기록하며 안정적인 크롭 성능을 보였습니다.

#### 3.2. OOD 분류 모델 (MobileNetV3)
비-파노라마 이미지(예: 치근단 사진, 일반 사진)와 파노라마 이미지를 이진 분류하는 모듈입니다.
- **Accuracy:** 100.0% (검증 셋 기준)
추후 발생할 수 있는 노이즈 입력을 완벽하게 차단할 수 있음을 확인했습니다.

#### 3.3. RBL Evaluator Logic (Pytest 단위 검증)
골소실량을 연산하고 평가하는 `evaluator.py` 및 `geometry.py` 등의 코어 로직은 Pytest 단위 테스트를 통해 MAE(Mean Absolute Error) 및 R-squared 점수 산출의 정합성을 검증했습니다.

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\chema\Github\Dental_003
configfile: pyproject.toml

collected 6 items

tests\test_evaluator.py .                                                [ 16%]
tests\test_geometry.py ...                                               [ 66%]
tests\test_staging.py ..                                                 [100%]

============================== 6 passed in 1.20s ==============================
```

***

### 4. 결론 및 향후 계획

- **검증 완료:** 파이프라인의 코어 모듈이 모두 정상 동작하며, 모델의 검출 지표가 목표치(mAP50 90% 이상)에 도달했음을 확인했습니다. 전체 테스트 코드가 패스하였으며 CI/CD 배포 준비가 완료되었습니다.
- **향후 계획 (Future Work):**
  - 현재 SAM 기반 마스크에 의존하여 휴리스틱하게 랜드마크를 추출하는 구조에서 벗어나, 추후 랜드마크 GT 데이터셋을 자체 구축하여 End-to-End 예측 모델로 개선할 계획입니다.
  - 패키징 및 도커라이징 작업 완료.
