import cv2
import torch
import numpy as np

from preprocess import preprocess_line
from model_loader import load_models
from src.dataset import decode_predictions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sort_boxes(boxes):
    return sorted(boxes, key=lambda x: x[1])


def remove_duplicate_boxes(boxes, y_thresh=10):

    filtered = []

    for box in boxes:
        y1 = box[1]

        keep = True

        for fbox in filtered:
            fy1 = fbox[1]
            if abs(y1 - fy1) < y_thresh:
                keep = False
                break

        if keep:
            filtered.append(box)

    return filtered


def crop_with_padding(img, box):
    x1, y1, x2, y2 = map(int, box)

    x1 = 0
    x2 = img.shape[1]

    y1 = max(0, y1 - 5)
    y2 = min(img.shape[0], y2 + 5)

    return img[y1:y2, x1:x2]


def ocr_image(img):

    crnn, charset, yolo = load_models()

    results = yolo(img)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    boxes = sort_boxes(boxes)
    boxes = remove_duplicate_boxes(boxes)

    lines = []

    for box in boxes:

        crop = crop_with_padding(img, box)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        input_tensor = preprocess_line(gray).to(DEVICE)

        with torch.no_grad():
            output = crnn(input_tensor)
            pred = decode_predictions(output, charset)

        text = pred[0]
        lines.append(text)

    return "\n".join(lines)