import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2

def is_image_valid(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def otsu_threshold(img):
    img_np = np.array(img)  

    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(binary)  


def load_data_for_trainingtesting(train_dir=None, test_dir=None, batch_size=64, img_size=(32, 32)):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.Lambda(lambda img: otsu_threshold(img)),   
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader, test_loader = None, None
    num_classes, class_names = 0, []

    if train_dir and os.path.exists(train_dir):
        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=transform,
            is_valid_file=is_image_valid
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes

        print(f"[INFO] Loaded training data: {len(train_dataset)} images from '{train_dir}'")

    if test_dir and os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(
            root=test_dir,
            transform=transform,
            is_valid_file=is_image_valid
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if not num_classes:
            num_classes = len(test_dataset.classes)
            class_names = test_dataset.classes

        print(f"[INFO] Loaded testing data: {len(test_dataset)} images from '{test_dir}'")

    return train_loader, test_loader, num_classes, class_names


def load_predict_data(predict_dir, img_size=(32, 32)):
    if not os.path.exists(predict_dir):
        raise FileNotFoundError(f"[ERROR] Folder Predict tidak ditemukan: {predict_dir}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.Lambda(lambda img: otsu_threshold(img)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_tensors = []
    image_paths = []

    for filename in os.listdir(predict_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(predict_dir, filename)

            if is_image_valid(path):
                img = Image.open(path).convert("L")
                tensor_img = transform(img)
                image_tensors.append(tensor_img)
                image_paths.append(path)

    print(f"[INFO] Loaded Predict data: {len(image_tensors)} images from '{predict_dir}'")

    return image_paths, image_tensors
