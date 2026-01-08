import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from Program.Models.thai_cnn3 import Thai_CNN3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "Dataset", "train")
PREDICT_DIR = os.path.join(BASE_DIR, "Data Independen")
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")
MODEL_PATH = os.path.join(BASE_DIR, "Results", "models", "thai_best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def otsu_threshold(img):
    img_np = np.array(img)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Folder TRAIN tidak ditemukan: {TRAIN_DIR}")

class_names = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])
num_classes = len(class_names)

print("====================================")
print(f"[INFO] Jumlah kelas terbaca: {num_classes}")
for idx, c in enumerate(class_names):
    print(f"  Index {idx}: {c}")
print("====================================\n")


print("[INFO] Memuat model...")

model = Thai_CNN3(num_classes=num_classes).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()

print("[INFO] Model siap digunakan.\n")

img_size = (32, 32)
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize(img_size),
    transforms.Lambda(lambda img: otsu_threshold(img)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict_image(image_path):
    img = Image.open(image_path).convert("L")
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        idx = probs.argmax(1).item()
        confidence = probs[0][idx].item()

    return class_names[idx], confidence


if __name__ == "__main__":
    summary_path = os.path.join(BASE_DIR, "hasil_prediksi.txt")
    summary_file = open(summary_path, "w", encoding="utf-8")

    print("\n=== MEMULAI PROSES PREDIKSI ===\n")

    for filename in os.listdir(PREDICT_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(PREDICT_DIR, filename)
        pred, conf = predict_image(img_path)

        print("===================================")
        print(f"Gambar     : {filename}")
        print(f"Prediksi   : {pred}")
        print(f"Akurasi    : {conf*100:.2f}%")
        print("===================================\n")

        summary_file.write(f"{filename} | Prediksi: {pred} | Akurasi: {conf*100:.2f}%\n")

        out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Gambar   : {filename}\n")
            f.write(f"Prediksi : {pred}\n")
            f.write(f"Akurasi  : {conf*100:.2f}%\n")

    summary_file.close()

    print(f"[INFO] Ringkasan disimpan di: {summary_path}")
    print(f"[INFO] Output per gambar: {OUTPUT_DIR}")
