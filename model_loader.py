import torch
from ultralytics import YOLO
from src.model import CRNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

crnn_model = None
charset = None
yolo_model = None


def load_models():
    global crnn_model, charset, yolo_model

    if crnn_model is None:
        checkpoint = torch.load("crnn.pth", map_location=DEVICE)
        charset = checkpoint["charset"]

        crnn_model = CRNN(num_classes=len(charset) + 1)
        crnn_model.load_state_dict(checkpoint["model_state"])
        crnn_model.to(DEVICE)
        crnn_model.eval()

    if yolo_model is None:
        yolo_model = YOLO("best.pt")

    return crnn_model, charset, yolo_model