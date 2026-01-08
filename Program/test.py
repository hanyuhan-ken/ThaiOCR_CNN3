import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
from Program.Models.thai_cnn3 import Thai_CNN3
from Utils.get_data import load_data_for_trainingtesting
from matplotlib import font_manager, rcParams

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "Dataset", "test")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
MODEL_PATH = os.path.join(RESULTS_DIR, "models", "thai_best_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULTS_DIR, exist_ok=True)

font_path = r"C:\Windows\Fonts\angsana.ttc"  
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = font_prop.get_name()
    rcParams['axes.unicode_minus'] = False
    print(f"[INFO] Font Thai aktif: {font_prop.get_name()}")
else:
    print("[WARNING] Font Angsana New tidak ditemukan.")
    font_prop = None

_, test_loader, num_classes, class_names = load_data_for_trainingtesting(
    test_dir=TEST_DIR, batch_size=64
)
print(f"[INFO] Loaded testing data: {len(test_loader.dataset)} images from '{TEST_DIR}'")

model = Thai_CNN3(num_classes=num_classes).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

print(f"[LOADED] Model dari: {MODEL_PATH}")
print(f"[INFO] Evaluasi menggunakan device: {device}")

y_true, y_pred = [], []
criterion = torch.nn.CrossEntropyLoss()
total_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

avg_loss = total_loss / len(test_loader.dataset)
acc = np.mean(np.array(y_true) == np.array(y_pred))

print("\n HASIL EVALUASI MODEL")
print("===================================")
print(f" Test Accuracy : {acc * 100:.2f}%")
print(f" Test Loss     : {avg_loss:.4f}\n")

report = classification_report(
    y_true, y_pred, target_names=class_names, zero_division=0
)
print("Classification Report:")
print(report)

report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(" HASIL EVALUASI MODEL\n")
    f.write("===================================\n")
    f.write(f"Test Accuracy : {acc * 100:.2f}%\n")
    f.write(f"Test Loss     : {avg_loss:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"[SAVED] Laporan evaluasi disimpan di: {report_path}")

cm = confusion_matrix(y_true, y_pred)

annot_labels = np.where(cm == 0, "", cm)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=annot_labels,
    fmt="",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar=False
)

plt.title("Confusion Matrix (Test Set)", fontproperties=font_prop)
plt.xlabel("Predicted Labels", fontproperties=font_prop)
plt.ylabel("True Labels", fontproperties=font_prop)
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_test.png")
plt.savefig(cm_path, dpi=300)
plt.close()

print(f"[SAVED] Confusion matrix disimpan di: {cm_path}")
print("[DONE] Evaluasi model selesai!\n")
