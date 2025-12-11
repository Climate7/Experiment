import os
import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np

# Environment Name (Student Initials)
ENV_NAME = "lk"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data/exp4")
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "output", "exp4")

# COCO Class names (Index 2 is bicycle)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def load_model(device):
    print("Loading Faster R-CNN model...")
    # Attempt to use modern weights API, fallback to pretrained=True for older torchvision
    try:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
    except ImportError:
        print("Note: Using older torchvision syntax (pretrained=True)")
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    model.to(device)
    model.eval()
    return model

def detect_bicycles(model, image_path, device, threshold=0.5):
    print(f"Processing {os.path.basename(image_path)}...")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Failed to load image: {image_path}")
        return None, []

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # Convert to Tensor (0-1 float)
    image_tensor = F.to_tensor(image_rgb).to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    
    # Filter results
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    results = []
    
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
            
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        
        # We are only interested in bicycles
        if class_name == 'bicycle':
            results.append((box, score))
            
            # Draw box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label
            text = f"{class_name}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(original_image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(original_image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return original_image, results

def generate_report(results_summary):
    report_path = os.path.join(OUTPUT_DIR, f"report_{ENV_NAME}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("实验报告：校园共享单车目标检测\n")
        f.write("=" * 40 + "\n")
        f.write(f"学生环境名称：{ENV_NAME}\n")
        f.write("实验任务：从校园道路、停车区图像中检测共享单车\n")
        f.write("\n")
        
        f.write("一、环境配置过程\n")
        f.write("-" * 30 + "\n")
        f.write(f"1. 创建虚拟环境：conda create -n {ENV_NAME} python=3.8\n")
        f.write(f"2. 激活环境：conda activate {ENV_NAME}\n")
        f.write("3. 安装依赖库：\n")
        f.write("   - PyTorch & Torchvision: 用于加载预训练深度学习模型 (Faster R-CNN)\n")
        f.write("   - OpenCV (opencv-python): 用于图像读取与结果绘制\n")
        f.write("   - NumPy: 用于矩阵运算\n")
        f.write("\n")

        f.write("二、算法分析\n")
        f.write("-" * 30 + "\n")
        f.write("1. 核心算法：Faster R-CNN (Region-based Convolutional Neural Networks)\n")
        f.write("   - 特征提取 (Backbone)：使用 ResNet50 + FPN (特征金字塔) 提取多尺度图像特征。\n")
        f.write("   - 目标定位 (RPN)：区域生成网络 (Region Proposal Network) 生成可能包含物体的候选框。\n")
        f.write("   - 分类与回归 (Head)：对候选框进行具体类别分类 (Softmax) 和坐标微调 (BBox Regression)。\n")
        f.write("2. 训练集：Microsoft COCO Dataset (包含 'bicycle' 类别)。\n")
        f.write("3. 检测逻辑：\n")
        f.write("   - 加载预训练模型权重。\n")
        f.write("   - 对输入图像进行 Tensor 转换与归一化。\n")
        f.write("   - 筛选置信度 (Confidence Score) > 0.5 且 类别为 'bicycle' 的检测框。\n")
        f.write("\n")

        f.write("三、模型检测结果统计\n")
        f.write("-" * 30 + "\n")
        for img_name, count in results_summary.items():
            f.write(f"图片: {img_name:<20} | 检测到单车数量: {count}\n")
    
    print(f"\nReport generated: {report_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model = load_model(device)
    
    # Process all images in data directory
    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    results_summary = {}
    
    for img_file in image_files:
        img_path = os.path.join(DATA_DIR, img_file)
        result_img, detections = detect_bicycles(model, img_path, device)
        
        if result_img is not None:
            # Save result image
            save_path = os.path.join(OUTPUT_DIR, f"detected_{img_file}")
            cv2.imwrite(save_path, result_img)
            results_summary[img_file] = len(detections)
            if len(detections) > 0:
                print(f"  -> Found {len(detections)} bicycles. Saved to {save_path}")
            else:
                print(f"  -> No bicycles found.")
            
    generate_report(results_summary)
    print("\nAll processing complete.")

if __name__ == "__main__":
    main()