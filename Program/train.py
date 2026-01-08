import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from Program.Models.thai_cnn3 import Thai_CNN3
from Utils.get_data import load_data_for_trainingtesting

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "Dataset", "train")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

learning_rate = 0.0001
epochs = 20
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, _, num_classes, class_names = load_data_for_trainingtesting(train_dir=TRAIN_DIR, batch_size=batch_size)

model = Thai_CNN3(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_loss_list, train_acc_list = [], []

print(f"[INFO] Training on {device} | Classes: {num_classes}")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}\n")

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    accuracy = correct / total

    train_loss_list.append(avg_loss)
    train_acc_list.append(accuracy)

    print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f} | Training Accuracy: {accuracy*100:.2f}%")

model_path = os.path.join(MODEL_DIR, "thai_best_model.pth")
torch.save(model.state_dict(), model_path)
print(f"\n[SAVED] Model disimpan di: {model_path}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_loss_list, label="Training Loss", color="orange")
plt.plot(range(1, epochs + 1), train_acc_list, label="Training Accuracy", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Loss and Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.savefig(os.path.join(RESULTS_DIR, "training_plot.png"))
plt.close()

print("\n Training selesai!")
