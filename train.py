import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

# ====================== 配置 ======================
# 数据集路径（请修改为你自己的路径）
TRAIN_DIR = "./chest_xray/train"
TEST_DIR = "./chest_xray/test"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ====================== 数据加载与预处理 ======================
class PneumoniaDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 读取文件路径和标签
def load_data_from_dir(dir_path):
    file_paths = []
    labels = []
    # 0: NORMAL, 1: PNEUMONIA
    for label, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
        class_dir = os.path.join(dir_path, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(label)
    return file_paths, labels

# 加载并划分训练/验证集（8:2）
train_paths, train_labels = load_data_from_dir(TRAIN_DIR)
test_paths, test_labels = load_data_from_dir(TEST_DIR)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.2, stratify=train_labels, random_state=42
)

# 打印数据分布
print(f"Train set: {len(train_paths)} samples, Normal: {train_labels.count(0)}, Pneumonia: {train_labels.count(1)}")
print(f"Val set: {len(val_paths)} samples, Normal: {val_labels.count(0)}, Pneumonia: {val_labels.count(1)}")
print(f"Test set: {len(test_paths)} samples, Normal: {test_labels.count(0)}, Pneumonia: {test_labels.count(1)}")

# 数据增强与预处理
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = PneumoniaDataset(train_paths, train_labels, transform=train_transform)
val_dataset = PneumoniaDataset(val_paths, val_labels, transform=val_test_transform)
test_dataset = PneumoniaDataset(test_paths, test_labels, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ====================== 模型构建（MobileNetV2 迁移学习） ======================
def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # 冻结底层
    for param in model.parameters():
        param.requires_grad = False
    # 替换顶层分类器
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 1)  # 二分类输出一个logit
    )
    return model.to(DEVICE)

model = build_model()
criterion = nn.BCEWithLogitsLoss()  # 等价于binary_crossentropy
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# ====================== 训练与验证 ======================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def val_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_labels, all_preds

train_losses, train_accs = [], []
val_losses, val_accs = [], []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, _, _ = val_epoch(model, val_loader, criterion)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ====================== 测试集评估 ======================
test_loss, test_acc, test_labels_true, test_preds = val_epoch(model, test_loader, criterion)
precision = precision_score(test_labels_true, test_preds)
recall = recall_score(test_labels_true, test_preds)
f1 = f1_score(test_labels_true, test_preds)
cm = confusion_matrix(test_labels_true, test_preds)

print("\n===== Test Set Metrics =====")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# ====================== 绘图保存（自动存到figures/） ======================
os.makedirs("figures", exist_ok=True)

# 训练/验证 Loss & Accuracy 曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.tight_layout()
plt.savefig("figures/train_curves.png")
plt.close()

# 混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("figures/confusion_matrix.png")
plt.close()

print("Plots saved to figures/ directory.")
