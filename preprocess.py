import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


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