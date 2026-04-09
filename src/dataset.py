import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class OCRDataset(Dataset):
    def __init__(self, root_dir, labels_path):
        self.root_dir = root_dir

        with open(labels_path, "r", encoding="utf-8") as f:
            self.samples = [line.strip().split("\t") for line in f]

        self.transform = transforms.Compose([
            #transforms.Resize((32, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        return image, label
class CTCLabelConverter:
    def __init__(self, charset_path):
        with open(charset_path, "r", encoding="utf-8") as f:
            chars = [line.strip("\n") for line in f]

        self.characters = chars
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(chars)}

        self.blank_idx = 0  # CTC blank

    def encode(self, texts):
        lengths = []
        result = []

        for text in texts:
            encoded = [self.char_to_idx[c] for c in text]
            lengths.append(len(encoded))
            result.extend(encoded)

        return torch.LongTensor(result), torch.LongTensor(lengths)

    def decode(self, preds):
        texts = []
        for pred in preds:
            char_list = []
            for i in range(len(pred)):
                if pred[i] != 0 and (i == 0 or pred[i] != pred[i-1]):
                    idx = int(pred[i].item())
                    char_list.append(self.idx_to_char[idx])
            texts.append("".join(char_list))
        return texts
def decode_predictions(preds, charset):
    """
    Greedy CTC decoding
    """
    #preds = preds.permute(1, 0, 2)  # (batch, seq_len, num_classes)
    preds = torch.argmax(preds, dim=2)

    results = []

    for pred in preds:
        string = ""
        previous_char = None

        for p in pred:
            p = p.item()
            if p != 0 and p != previous_char:
                #print("Index:", p, "Char:", charset[p - 1])
                string += charset[p - 1]
            previous_char = p

        results.append(string)

    return results
def load_charset(path="charset.txt"):
    charset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line == "":
                continue
            charset.append(line)
    return "".join(charset)

CHARSET = load_charset()