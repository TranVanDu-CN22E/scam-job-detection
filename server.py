import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File
import uvicorn

from src.model import CRNN
from src.dataset import decode_predictions
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# =========================
# PREPROCESS
# =========================
def preprocess_line(img):

    h, w = img.shape

    new_h = 32
    new_w = int(w * (new_h / h))

    img = cv2.resize(img, (new_w, new_h))

    min_w = 100
    if new_w < min_w:
        pad = np.ones((32, min_w - new_w), dtype=np.uint8) * 255
        img = np.concatenate([img, pad], axis=1)

    img = Image.fromarray(img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    return transform(img).unsqueeze(0)


# =========================
# LOAD MODELS (LOAD 1 LẦN)
# =========================
crnn = None
charset = None
yolo = None
phobert_tokenizer = None
phobert_model = None


def load_models():
    global crnn, charset, yolo, phobert_tokenizer, phobert_model

    if crnn is None:
        checkpoint = torch.load("crnn.pth", map_location=DEVICE)

        charset = checkpoint["charset"]

        crnn = CRNN(num_classes=len(charset) + 1)
        crnn.load_state_dict(checkpoint["model_state"])
        crnn.to(DEVICE)
        crnn.eval()

    if yolo is None:
        yolo = YOLO("best.pt")

    if phobert_model is None:

        MODEL_PATH = "phobert-scam"

        phobert_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        phobert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

        phobert_model.to(DEVICE)
        phobert_model.eval()


# =========================
# SORT BOX
# =========================
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


# =========================
# OCR CORE
# =========================
def ocr_image(img):

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

def detect_scam(text):

    inputs = phobert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = phobert_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    score = probs[0][pred].item()

    label = "SCAM" if pred == 1 else "SAFE"

    return label, score

import re

def clean_text(text):
    # Thay thế xuống dòng bằng khoảng trắng và xóa khoảng trắng thừa ở 2 đầu
    text = text.replace("\n", " ").strip()
    # Xóa các ký tự đặc biệt không cần thiết
    text = re.sub(r'["!\'?#\^\*\[\]\{\}\|\.\,]', '', text)
    
    return text


def process_lines(raw_text):
    raw_lines = raw_text.split("\n")
    final_lines = []
    buffer = ""
    in_parentheses = False

    for line in raw_lines:
        line = line.strip()
        if not line: continue # Bỏ qua dòng trống
        
        if "(" in line and ")" not in line:
            # Bắt đầu mở ngoặc nhưng chưa đóng trên cùng 1 dòng
            buffer = line
            in_parentheses = True
        elif ")" in line and in_parentheses:
            # Tìm thấy dấu đóng ngoặc cho buffer đang chờ
            buffer += " " + line
            final_lines.append(clean_text(buffer))
            buffer = ""
            in_parentheses = False
        elif in_parentheses:
            # Đang nằm giữa cặp ngoặc, tiếp tục cộng dồn
            buffer += " " + line
        else:
            # Dòng bình thường hoặc dòng có cặp ngoặc đóng mở đầy đủ
            final_lines.append(clean_text(line))
            
    return final_lines


import random

def build_windows(lines):
    windows = []

    # =========================
    # 1. 3 câu liên tiếp
    # =========================
    for i in range(len(lines) - 3):
        windows.append(" ".join(lines[i:i+4]))
    # =========================
    # 2. random 3 câu
    # =========================
    """
    if len(lines) >= 3:
        seen = set()

        # số lần random (bạn có thể chỉnh)
        num_samples = min(20, len(lines)*2)

        for _ in range(num_samples):
            combo = tuple(sorted(random.sample(lines, 3)))

            if combo not in seen:
                seen.add(combo)
                windows.append(" ".join(combo))"""
    return windows

def get_combined_sensitive_text(lines):
    """
    Lọc các câu chứa từ khóa và số, sau đó nối lại thành 1 dòng duy nhất.
    """
    # Pattern tìm: tuyển, thưởng, típ hoặc số 0-9
    pattern = re.compile(r'(tuyển|thưởng|típ|[0-9])', re.IGNORECASE)
    
    # Lọc ra danh sách các câu thỏa điều kiện
    filtered_list = [line.strip() for line in lines if pattern.search(line)]
    
    # Nối lại thành 1 dòng duy nhất, ngăn cách bởi dấu cách
    # Ní có thể thay " " bằng ". " nếu muốn phân tách câu rõ hơn
    combined_text = " ".join(filtered_list)
    
    return combined_text


# =========================
# API
# =========================
@app.on_event("startup")
def startup():
    load_models()
    print("OCR Models loaded")


@app.get("/")
def root():
    return {"status": "OCR server running"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # OCR -> nhiều dòng
    text = ocr_image(img)

    #lines = [clean_text(x) for x in text.split("\n") if x.strip()]
    lines = process_lines(text)

    if len(lines) == 0:
        return {
            "success": True,
            "text": "",
            "scam": "SAFE",
            "confidence": 0.0
        }
    sensitive_lines = get_combined_sensitive_text(lines)
    windows = build_windows(lines)

    results = []

    for w in windows:
        label, score = detect_scam(w)

        results.append({
            "text": w,
            "label": label,
            "score": score
        })
    label, score = detect_scam(sensitive_lines)

    results.append({
        "text": sensitive_lines,
        "label": label,
        "score": score
    })

    # =========================
    # ƯU TIÊN SCAM
    # =========================
    scams = [r for r in results if r["label"] == "SCAM"]

    if len(scams) > 0:
        best = max(scams, key=lambda x: x["score"])
    else:
        # lấy SAFE có confidence thấp nhất
        safes = [r for r in results if r["label"] == "SAFE"]
        best = min(safes, key=lambda x: x["score"])

    return {
        "success": True,
        "text": lines,
        "scam": best["label"],
        "confidence": float(best["score"]),
        "lines": results
    }


# =========================
# RUN
# =========================
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)