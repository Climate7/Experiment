import os
import json
import math
import argparse
import time
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

ENV_NAME = "lk"

# defined constants
OUTPUT_DIR = r"d:/Goat/Courses/JuniorYear_FirstSemester/Machine_Vision/Experiment/output/exp3"
DATA_ROOT = r"d:/Goat/Courses/JuniorYear_FirstSemester/Machine_Vision/Experiment/data/mnist"
INFER_IMG_PATH = r"d:/Goat/Courses/JuniorYear_FirstSemester/Machine_Vision/Experiment/data/exp3/student_id5.jpg"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 输入特征来自 64 x 14 x 14 (28x28 经一次 2x2 池化)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = F.max_pool2d(self.bn2(F.relu(self.conv2(x))), 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(batch_size=128, num_workers=2):
    tf_train = transforms.Compose([
        transforms.RandomAffine(5, translate=(0.05,0.05), scale=(0.95,1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    root = DATA_ROOT
    os.makedirs(root, exist_ok=True)
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=tf_train)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=tf_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def train(model, train_loader, test_loader, device, epochs=12, lr=1e-3, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for images, targets in pbar:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pred = outputs.argmax(1)
            correct += (pred == targets).sum().item()
            total += images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct / total:.4f}"})
        train_loss = running_loss / total
        train_acc = correct / total
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        scheduler.step()
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"Epoch [{epoch}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model_state": model.state_dict(), "acc": best_acc, "epoch": epoch}, os.path.join(out_dir, "mnist_cnn_best.pth"))
    torch.save({"model_state": model.state_dict(), "history": history}, os.path.join(out_dir, "mnist_cnn_last.pth"))
    with open(os.path.join(out_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return history, best_acc

def evaluate(model, data_loader, device, criterion=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item() * images.size(0)
            pred = outputs.argmax(1)
            correct += (pred == targets).sum().item()
            total += images.size(0)
    if criterion is None:
        return correct / total
    return running_loss / total, correct / total

def test_report(model, test_loader, device, out_dir=OUTPUT_DIR):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            pred = outputs.argmax(1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(targets.numpy().tolist())
    report = classification_report(y_true, y_pred, digits=4, output_dict=False)
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report

def pad_and_resize_digit(img, size=28, pad=4):
    h, w = img.shape
    scale = (size - 2 * pad) / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def preprocess_id_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = 255 - th
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- Debug Visualization Start ---
    debug_img = cv2.imread(image_path)
    # --- Debug Visualization End ---

    boxes = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < (h * w) * 0.001: # Increased threshold from 0.0005 to 0.001 to reduce noise
            continue
        ar = cw / (ch + 1e-6)
        if ar < 0.1 or ar > 1.5: # Slightly relaxed aspect ratio
            continue
        boxes.append((x, y, cw, ch))
        # Draw box on debug image
        cv2.rectangle(debug_img, (x, y), (x+cw, y+ch), (0, 0, 255), 2)

    # Save debug image
    debug_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "debug_contours.jpg"), debug_img)

    if not boxes:
        return []
    # Sort strictly by X for single line text
    boxes = sorted(boxes, key=lambda b: b[0]) 
    
    digits = []
    for i, bx in enumerate(boxes):
        x, y, cw, ch = bx
        y0 = max(0, y - ch//5)
        x0 = max(0, x - cw//5)
        y1 = min(h, y + ch + ch//5)
        x1 = min(w, x + cw + cw//5)
        roi = th[y0:y1, x0:x1]
        roi = pad_and_resize_digit(roi, 28, 4)
        
        # Save individual digit for debug
        cv2.imwrite(os.path.join(debug_dir, f"digit_{i}.jpg"), roi)
        
        # Reverted: Additional morphological cleanup degraded performance
        # kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        # roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel_clean)

        roi = roi.astype(np.float32) / 255.0
        roi = (roi - 0.1307) / 0.3081
        digits.append(roi)
    return digits

def predict_digits(model, digits, device):
    if not digits:
        return []
    arr = np.stack(digits, axis=0)
    t = torch.from_numpy(arr).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(t)
        pred = logits.argmax(1).cpu().numpy().tolist()
    return pred

def predict_id_string(model, image_path, device):
    digits = preprocess_id_image(image_path)
    preds = predict_digits(model, digits, device)
    s = "".join(str(p) for p in preds)
    return s

def save_experiment_report(out_dir, history, best_acc, env_name, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("实验名称：基于深度学习的手写数字识别")
    lines.append(f"环境名称：{env_name}")
    lines.append(f"时间：{ts}")
    lines.append(f"训练轮数：{len(history['epoch'])}")
    lines.append(f"最佳测试准确率：{best_acc:.4f}")
    if extra and "id_result" in extra:
        lines.append(f"学号识别结果：{extra['id_result']}")
        lines.append(f"学号图片路径：{extra.get('image_path','')}")
    lines.append("算法说明：")
    lines.append("1. 使用卷积神经网络在MNIST上训练，并进行轻量数据增强")
    lines.append("2. 学号照片通过阈值与形态学处理提取数字轮廓")
    lines.append("3. 逐数字裁剪归一化到28x28，并按MNIST均值方差标准化")
    lines.append("4. 模型逐个预测数字并拼接为学号字符串")
    with open(os.path.join(out_dir, f"report_{env_name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def load_model(model_path, device):
    model = Net().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    out_dir = OUTPUT_DIR
    
    if args.train:
        train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
        model = Net().to(device)
        history, best_acc = train(model, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr, out_dir=out_dir)
        test_report(model, test_loader, device, out_dir=out_dir)
        save_experiment_report(out_dir, history, best_acc, ENV_NAME)
        
    infer_path = INFER_IMG_PATH
    model_path = os.path.join(out_dir, "mnist_cnn_best.pth")
    
    if os.path.isfile(infer_path) and os.path.isfile(model_path):
        os.makedirs(out_dir, exist_ok=True)
        model = load_model(model_path, device)
        sid = predict_id_string(model, infer_path, device)
        print(sid)
        with open(os.path.join(out_dir, "infer_result.json"), "w", encoding="utf-8") as f:
            json.dump({"image_path": infer_path, "id_result": sid}, f, ensure_ascii=False, indent=2)
        # Preserve original behavior: overwrite report with inference result
        save_experiment_report(out_dir, {"epoch":[],"train_loss":[],"train_acc":[],"test_loss":[],"test_acc":[]}, 0.0, ENV_NAME, extra={"id_result": sid, "image_path": infer_path})

if __name__ == "__main__":
    main()