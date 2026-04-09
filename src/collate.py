import torch

def ocr_collate_fn(batch):
    images, labels = zip(*batch)

    widths = [img.shape[2] for img in images]
    max_w = max(widths)

    padded_images = []
    for img in images:
        pad_w = max_w - img.shape[2]
        padded = torch.nn.functional.pad(img, (0, pad_w), value=1.0)
        padded_images.append(padded)

    images = torch.stack(padded_images)

    return images, labels