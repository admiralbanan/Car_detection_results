
import os
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import random


DATA_DIR = Path(r"D:\Diploma AI\dataset")
BATCH_SIZE = 25
EPOCHS = 100
LR = 0.0015
PATIENCE = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 509563

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed(SEED)

print(f"Используем устройство: {DEVICE}")


transform_train = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.RandomHorizontalFlip(p=0.001),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_data = datasets.ImageFolder(DATA_DIR / "train", transform=transform_train)
val_data = datasets.ImageFolder(DATA_DIR / "val", transform=transform_val)

print(f"Найдено классов: {len(train_data.classes)} → {train_data.classes}")
print(f"Образцы: train={len(train_data)}, val={len(val_data)}")

if len(train_data) == 0 or len(val_data) == 0:
    raise ValueError("Датасет пустой! Проверь пути.")

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


model = efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_data.classes))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


train_losses = []
val_accuracies = []
best_acc = 0.0
patience_counter = 0
best_epoch = 0
all_preds = []
all_labels = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    model.eval()
    correct = total = 0
    epoch_preds = []
    epoch_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
    acc = correct / total
    val_accuracies.append(acc)

    if acc > best_acc:
        all_preds = epoch_preds
        all_labels = epoch_labels
        best_acc = acc
        best_epoch = epoch + 1
        patience_counter = 0


        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': train_data.classes,        
            'val_accuracy': best_acc,
            'epoch': best_epoch,
            'train_samples': len(train_data),
            'val_samples': len(val_data)
        }, "best_model_with_classes.pth") 

        print("Модель сохранена с классами!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping!")
            break

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_losses[-1]:.4f} | Val Acc: {acc:.4f} | Best: {best_acc:.4f}")


model.load_state_dict(torch.load("best_model_with_classes.pth")['model_state_dict'])
final_acc = best_acc
print(f"Финальная точность: {final_acc:.4f} (на эпохе {best_epoch})")


cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_data.classes, yticklabels=train_data.classes)
plt.xlabel("Предсказано")
plt.ylabel("Истинно")
plt.title(f"Матрица ошибок (Val Acc: {best_acc:.1%})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print(classification_report(all_labels, all_preds, target_names=train_data.classes))


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss", marker='o', color='tab:red')
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.title("Потери")

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Val Accuracy", marker='o', color='tab:green')
plt.axhline(y=best_acc, color='gray', linestyle='--', label=f'Лучшая: {best_acc:.1%}')
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.title("Точность")
plt.legend()
plt.suptitle("Прогресс обучения")
plt.tight_layout()
plt.savefig("training_progress.png", dpi=300)
plt.show()


def plot_samples(dataset, class_names, n=6):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    shown = {c: 0 for c in class_names}
    for img, label in dataset:
        cls = class_names[label]
        if shown[cls] < n // len(class_names):
            ax = axes[sum(shown.values())]
            img = img.permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(cls)
            ax.axis('off')
            shown[cls] += 1
        if sum(shown.values()) >= n:
            break
    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=300)
    plt.show()

plot_samples(train_data, train_data.classes)


if (DATA_DIR / "test").exists():
    test_data = datasets.ImageFolder(DATA_DIR / "test", transform=transform_val)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"ТЕСТ: {correct/total:.4f}")


with open("results.txt", "w", encoding="utf-8") as f:
    f.write(f"Классы: {train_data.classes}\n")
    f.write(f"Val Acc: {best_acc:.4f}\n")
    f.write(f"Модель: best_model_with_classes.pth\n")

