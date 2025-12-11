import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows下常用SimHei(黑体)
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

def manual_convolve2d(image, kernel):
    """
    使用NumPy手动执行二维卷积操作。
    """
    if len(image.shape) == 3:
        h, w, c = image.shape
        # 独立处理每个颜色通道
        output = np.zeros((h, w, c), dtype=np.float32)
        for i in range(c):
            output[:, :, i] = manual_convolve2d_channel(image[:, :, i], kernel)
        return output
    else:
        return manual_convolve2d_channel(image, kernel)

def manual_convolve2d_channel(image, kernel): 
    """
    单通道卷积的辅助函数。
    使用填充保持输出图像尺寸与输入一致。
    """
    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_h = k_h // 2  # 垂直方向填充量（核高度的一半）
    pad_w = k_w // 2  # 水平方向填充量（核宽度的一半）
    
    # 对图像进行零填充
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    output = np.zeros_like(image, dtype=np.float32)  # 初始化输出图像
    
    # 卷积的数学定义要求翻转核
    # 但在图像处理中，"滤波"通常指的是互相关操作
    # 题目要求是"滤波"，在计算机视觉中通常采用互相关（不翻转核）
    # 考虑到题目中给出的核是边缘检测算子，CV中标准实现（如cv2.filter2D）使用互相关
    # 因此此处采用互相关（不翻转核）
    
    # 注：使用滑动窗口的向量化实现会更快
    # 但为了严格遵循"不使用现成函数包"的思路，这里使用切片优化的循环
    # 兼顾代码的可读性和教育意义
    
    for i in range(h):
        for j in range(w):
            # 提取感兴趣区域（与核大小一致）
            roi = padded_image[i:i+k_h, j:j+k_w]
            # 元素-wise相乘后求和（互相关操作）
            output[i, j] = np.sum(roi * kernel)
            
    return output

def compute_sobel(image):
    """
    使用手动卷积计算Sobel梯度幅值。
    """
    # 边缘检测通常在灰度图上进行，若输入是彩色图则先转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 标准Sobel算子
    # Gx：水平方向梯度检测（边缘垂直）
    Gx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)
    # Gy：垂直方向梯度检测（边缘水平）
    Gy_kernel = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=np.float32)
    
    # 计算两个方向的梯度
    grad_x = manual_convolve2d(gray, Gx_kernel)
    grad_y = manual_convolve2d(gray, Gy_kernel)
    
    # 计算梯度幅值：sqrt(Gx² + Gy²)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 归一化到0-255范围并转换为8位整数
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return magnitude

def compute_given_kernel(image, kernel):
    """
    使用指定的自定义核对图像进行滤波。
    """
    # 滤波通常在灰度图上进行，若输入是彩色图则先转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # 应用自定义核进行滤波
    result = manual_convolve2d(gray, kernel)
    
    # 取绝对值（梯度可能为负）以便可视化
    result = np.abs(result)
    # 归一化到0-255范围并转换为8位整数
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def manual_color_histogram(image):
    """
    手动计算彩色图像的直方图。
    返回B（蓝）、G（绿）、R（红）三个通道的直方图。
    """
    # 初始化三个通道的直方图（每个通道0-255共256个 bins）
    hist_b = np.zeros(256, dtype=int)
    hist_g = np.zeros(256, dtype=int)
    hist_r = np.zeros(256, dtype=int)
    
    h, w, c = image.shape  # 获取图像高度、宽度和通道数
    
    # 统计像素值分布 - 使用NumPy优化迭代效率
    # 避免使用cv2.calcHist等现成函数
    
    # 将每个通道的像素值展平为一维数组，便于遍历
    flat_b = image[:,:,0].flatten()  # B通道
    flat_g = image[:,:,1].flatten()  # G通道
    flat_r = image[:,:,2].flatten()  # R通道
    
    # 使用np.add.at进行"手动"计数，避免Python循环的效率问题
    np.add.at(hist_b, flat_b, 1)
    np.add.at(hist_g, flat_g, 1)
    np.add.at(hist_r, flat_r, 1)
            
    return hist_b, hist_g, hist_r

def compute_glcm_features(image):
    """
    手动计算灰度共生矩阵（GLCM）纹理特征。
    提取的特征：对比度（Contrast）、同质性（Homogeneity）、能量（Energy）、相关性（Correlation）
    """
    # 纹理特征提取基于灰度图，若输入是彩色图则先转灰度
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # GLCM参数设置
    levels = 256  # 灰度级别（0-255）
    d = 1  # 像素对之间的距离
    theta = 0  # 角度（0度表示水平方向：当前像素与其右侧邻居）
    
    h, w = gray.shape
    glcm = np.zeros((levels, levels), dtype=np.float32)  # 初始化GLCM矩阵
    
    # 计算GLCM（水平方向：(i,j) 和 (i,j+1) 像素对）
    # 使用切片操作提高效率，替代嵌套循环
    left = gray[:, :-d].flatten()  # 左像素：所有行，列0到w-2
    right = gray[:, d:].flatten()  # 右像素：所有行，列d到w-1
    
    # 填充GLCM矩阵
    # 对于像素对(l, r)，在glcm[l, r]位置加1
    # 使用展平索引提高效率
    np.add.at(glcm.reshape(-1), left * levels + right, 1)
    
    # 归一化GLCM（转换为概率分布）
    glcm_sum = np.sum(glcm)
    if glcm_sum > 0:
        glcm /= glcm_sum
        
    # 计算纹理特征
    # 创建索引矩阵（i和j分别对应行列索引）
    i_matrix, j_matrix = np.indices((levels, levels))
    
    # 对比度：sum(P(i,j) * (i-j)²) - 衡量像素灰度差异的强度
    contrast = np.sum(glcm * (i_matrix - j_matrix)**2)
    
    # 同质性：sum(P(i,j) / (1 + (i-j)²)) - 衡量像素灰度的均匀性
    homogeneity = np.sum(glcm / (1 + (i_matrix - j_matrix)**2))
    
    # 能量：sum(P(i,j)²) - 衡量图像灰度分布的均匀性（能量越大越均匀）
    energy = np.sum(glcm**2)
    
    # 相关性：sum((i-μi)(j-μj)P(i,j)) / (σiσj) - 衡量像素对的线性相关程度
    # 计算均值和标准差
    mu_i = np.sum(glcm * i_matrix)  # i的均值
    mu_j = np.sum(glcm * j_matrix)  # j的均值
    sigma_i = np.sqrt(np.sum(glcm * (i_matrix - mu_i)**2))  # i的标准差
    sigma_j = np.sqrt(np.sum(glcm * (j_matrix - mu_j)**2))  # j的标准差
    
    if sigma_i * sigma_j == 0:
        correlation = 0  # 避免除以零
    else:
        correlation = np.sum(glcm * (i_matrix - mu_i) * (j_matrix - mu_j)) / (sigma_i * sigma_j)
        
    return {
        "对比度(Contrast)": contrast,
        "同质性(Homogeneity)": homogeneity,
        "能量(Energy)": energy,
        "相关性(Correlation)": correlation
    }

def main():
    # 1. 加载图像
    image_path = '../data/red_door.jpg'
    if not os.path.exists(image_path):
        print(f"错误：未找到图像文件 {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("错误：图像加载失败")
        return
    
    print(f"成功加载图像：{image_path}，图像尺寸：{img.shape}")

    # 2. Sobel边缘检测
    print("正在执行Sobel滤波...")
    sobel_img = compute_sobel(img)
    cv2.imwrite('../output/sobel_output.jpg', sobel_img)
    
    # 3. 自定义核滤波
    print("正在执行自定义核滤波...")
    # 题目中给出的自定义核（水平方向边缘检测算子）
    custom_kernel = np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]], dtype=np.float32)
    custom_filtered_img = compute_given_kernel(img, custom_kernel)
    cv2.imwrite('../output/custom_kernel_output.jpg', custom_filtered_img)
    
    # 4. 彩色直方图计算
    print("正在计算彩色直方图...")
    hist_b, hist_g, hist_r = manual_color_histogram(img)
    
    # 可视化直方图
    plt.figure(figsize=(10, 6))
    plt.title("彩色直方图")
    plt.xlabel("像素灰度级（Bins）")
    plt.ylabel("像素数量（# of Pixels）")
    plt.plot(hist_r, color="red", label="红色通道")
    plt.plot(hist_g, color="green", label="绿色通道")
    plt.plot(hist_b, color="blue", label="蓝色通道")
    plt.legend()
    plt.grid(alpha=0.3)  # 网格透明度
    plt.savefig('../output/histogram_output.jpg')
    plt.close()
    
    # 5. 纹理特征提取
    print("正在提取纹理特征...")
    features = compute_glcm_features(img)
    print("纹理特征结果：", features)
    
    # 保存纹理特征到npy文件
    np.save('../output/texture_features.npy', features)
    
    print("所有任务执行完成！")
    print("输出文件已保存：sobel_output.jpg（Sobel边缘检测）、custom_kernel_output.jpg（自定义核滤波）、histogram_output.jpg（彩色直方图）、texture_features.npy（纹理特征）")

if __name__ == "__main__":
    main()