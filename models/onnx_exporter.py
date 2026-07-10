import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

def export_classifier(pt_path="models/pano_classifier.pt", onnx_path="models/pano_classifier.onnx"):
    """
    MobileNetV3 PyTorch 가중치를 ONNX 포맷으로 내보냅니다.
    """
    print(f"Exporting Classifier {pt_path} to {onnx_path}...")
    try:
        classifier = models.mobilenet_v3_small()
        num_ftrs = classifier.classifier[3].in_features
        classifier.classifier[3] = nn.Linear(num_ftrs, 2)
        
        # Load weights on CPU
        classifier.load_state_dict(torch.load(pt_path, map_location="cpu"))
        classifier.eval()
        
        # Dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        torch.onnx.export(
            classifier, 
            dummy_input, 
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("Classifier export success!")
    except Exception as e:
        print(f"Failed to export classifier: {e}")

def export_yolo(pt_path="runs/detect/models/detector_train/weights/best.pt"):
    """
    YOLOv11 PyTorch 가중치를 ONNX 포맷으로 내보냅니다.
    """
    print(f"Exporting YOLO {pt_path} to ONNX...")
    try:
        model = YOLO(pt_path)
        # Ultralytics built-in export
        # This will create best.onnx in the same directory
        model.export(format="onnx", opset=12, simplify=True)
        print("YOLO export success!")
    except Exception as e:
        print(f"Failed to export YOLO: {e}")

if __name__ == "__main__":
    export_classifier()
    export_yolo()
