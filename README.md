# OCR FastAPI Server

API OCR sử dụng FastAPI để nhận ảnh và trả về text sau khi nhận diện.
Server chạy local và cung cấp endpoint `/ocr`.

---

# 🚀 Setup môi trường

Tạo virtual environment (Python 3.11)

```bash
py -3.11 -m venv venv
```

Activate môi trường:

```bash
venv\Scripts\activate
```

---

# 📦 Cài đặt dependencies

Cài FastAPI + xử lý ảnh

```bash
pip install fastapi uvicorn python-multipart numpy opencv-python
```

Cài NLP model (PhoBERT / transformers)

```bash
pip install transformers datasets evaluate accelerate
```

Cài YOLO (Ultralytics)

```bash
pip install ultralytics
```

Cài PyTorch GPU (CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

# ▶️ Chạy server

```bash
python server.py
```

Server sẽ chạy tại:

```
http://localhost:8000
```

---

# 📚 Swagger UI

Test trực tiếp trên trình duyệt:

```
http://localhost:8000/docs
```

Hoặc:

```
http://localhost:8000/docs#/default/ocr_ocr_post
```

---

# 🧠 Pipeline OCR

Pipeline xử lý:

```
Image
  ↓
YOLO detect text region
  ↓
Crop text boxes
  ↓
CRNN OCR
  ↓
Text lines
  ↓
PhoBERT classify scam / safe
```

---

# 📁 Project Structure

```
CheckScam/
├─ phobert-scam/
│  ├─ added_tokens.json
│  ├─ bpe.codes
│  ├─ config.json
│  ├─ model.safetensors
│  ├─ tokenizer_config.json
│  ├─training_args.bin
│  └─ vocab.txt
├─ src/
│  ├─ collate.py
│  ├─ dataset.py
│  └─ model.py
├─ venv/
├─ .gitignore
├─ best.pt
├─ charset.txt
├─ crnn.pth
├─ model_loader.py
├─ ocr_engine.py
├─ preprocess.py
├─ README.md
└─ server.py

```

---

# ⚡ Notes

* Yêu cầu GPU để chạy nhanh
* Hỗ trợ batch OCR
* API trả về từng dòng text
* Có thể tích hợp vào C# / .NET / React frontend

---

# 👨‍💻 Author

OCR + Scam Detection Pipeline
FastAPI + YOLO + CRNN + PhoBERT
