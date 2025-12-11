import cv2
import numpy as np
import os

def process_image(image_path, output_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    height, width = img.shape[:2]

    # 1. 灰度化 (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 高斯模糊 (Gaussian Blur) - 抑制噪声
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny 边缘检测 (Edge Detection)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur, low_threshold, high_threshold)
    
    # 保存边缘检测结果用于报告
    edge_path = output_path.replace("result_", "edges_")
    cv2.imwrite(edge_path, edges)

    # 4. 区域生长/感兴趣区域 (ROI Selection)
    # 车道线通常位于图像下半部分，我们定义一个梯形区域
    mask = np.zeros_like(edges)
    
    roi_height_top = int(height * 0.6) # 顶部高度位置（图像中上部）
    
    # 定义梯形顶点 (根据一般车载摄像头视角调整)
    vertices = np.array([[
        (0, height),                            # 左下
        (int(width * 0.45), roi_height_top),    # 左上
        (int(width * 0.55), roi_height_top),    # 右上
        (width, height)                         # 右下
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 保存ROI结果
    roi_path = output_path.replace("result_", "roi_")
    cv2.imwrite(roi_path, masked_edges)

    # 5. 霍夫变换 (Hough Transform)
    # 使用概率霍夫变换 HoughLinesP，直接返回线段端点
    rho = 1              # 距离分辨率 (像素)
    theta = np.pi / 180  # 角度分辨率 (弧度)
    threshold = 15       # 最小投票数 (交点数)
    min_line_len = 40    # 最小线段长度
    max_line_gap = 20    # 允许的线段最大间隙
    
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), 
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # 6. 车道线拟合与优化 (Line Fitting & Averaging)
    # 创建一个全黑图像用于绘制车道线
    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    debug_img = img.copy() # 调试图

    if lines is not None:
        left_lines = []
        right_lines = []
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 绘制所有检测到的线段到调试图
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

                if x2 == x1:
                    continue # 忽略垂直线 (避免斜率无穷大)
                
                slope = (y2 - y1) / (x2 - x1)
                
                # 过滤斜率：
                # 1. 去除接近水平的线 (噪音)
                # 2. 根据斜率正负区分左右车道 (图像坐标系y向下，左车道斜率为负，右车道斜率为正)
                if abs(slope) < 0.5:
                    continue
                    
                if slope < 0:
                    left_lines.append((slope, y1 - slope * x1)) # 存储 (斜率, 截距)
                else:
                    right_lines.append((slope, y1 - slope * x1))
        
        # 保存调试图（显示所有候选线段）
        debug_path = output_path.replace("result_", "debug_")
        cv2.imwrite(debug_path, debug_img)

        def draw_lane(lines_data, color):
            if not lines_data:
                return
            
            # 对斜率和截距取平均值，得到一条平滑的直线
            avg_slope = np.mean([l[0] for l in lines_data])
            avg_intercept = np.mean([l[1] for l in lines_data])
            
            # 计算绘制直线的端点 (y = mx + b => x = (y - b) / m)
            y1 = height
            y2 = int(roi_height_top) + 20 # 稍微延伸一点
            
            try:
                x1 = int((y1 - avg_intercept) / avg_slope)
                x2 = int((y2 - avg_intercept) / avg_slope)
                # 绘制粗直线
                cv2.line(line_img, (x1, y1), (x2, y2), color, 10)
            except OverflowError:
                pass 
            
        # 绘制左车道 (绿色)
        draw_lane(left_lines, (0, 255, 0)) 
        # 绘制右车道 (绿色)
        draw_lane(right_lines, (0, 255, 0))

    # 7. 图像叠加
    # 将车道线图像叠加到原始图像上
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    
    cv2.imwrite(output_path, result)
    print(f"Processed {os.path.basename(image_path)} -> Saved to {os.path.basename(output_path)}")

def main():
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "exp2")
    output_dir = os.path.join(base_dir, "output", "exp2")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    # 获取所有图片文件
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(valid_exts)]
    
    if not image_files:
        print(f"No images found in {data_dir}")
        return
    
    print(f"Found {len(image_files)} images. Starting processing...")
    
    for img_file in image_files:
        input_path = os.path.join(data_dir, img_file)
        output_path = os.path.join(output_dir, f"result_{img_file}")
        process_image(input_path, output_path)
        
    print("All tasks completed.")

if __name__ == "__main__":
    main()