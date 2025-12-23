import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

smooth_deep = 9  # 平滑深度，即卷积操作的次数
def convolution_smooth(image_path, output_path, kernel_size=3):
    """
    对图像进行卷积平滑处理，保持原图大小不变。
    
    Args:
        image_path: 原图路径
        output_path: 保存路径
        kernel_size: 卷积核大小 (必须是奇数，如 3, 5)。3 为轻微平滑，5 为中度平滑。
    """
    
    # 1. 读取图像
    # 注意：cv2.imread 默认为 BGR 格式
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"错误：找不到文件 {image_path}")
        return

    # 2. 定义卷积核 (Kernel)
    # 这里使用均值滤波器：创建一个 3x3 的矩阵，所有值均为 1，然后除以 9 (归一化)
    # 数学含义：当前像素值 = 周围 3x3 区域像素的平均值
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    print(f"正在使用 {kernel_size}x{kernel_size} 卷积核进行处理...")
    print(f"卷积核内容:\n{kernel}")

    # 3. 进行卷积操作 (cv2.filter2D)
    # src: 原图
    # ddepth: -1 表示输出图像深度与原图相同 (保持 uint8)
    # kernel: 我们定义的卷积核
    # 这一步会自动处理边缘填充，因此输出大小 = 输入大小
    smoothed_img = img
    for _ in range(smooth_deep):
        smoothed_img = cv2.filter2D(smoothed_img, -1, kernel)

    # 4. 保存结果
    cv2.imwrite(output_path, smoothed_img)
    print(f"已保存平滑后的图片至: {output_path}")

    # 5. 可视化对比 (可选)
    #为了 matplotlib 显示正常，需要将 BGR 转为 RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    smooth_rgb = cv2.cvtColor(smoothed_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image (原图)")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Smoothed Image ({kernel_size}x{kernel_size} Convolution)")
    plt.imshow(smooth_rgb)
    plt.axis('off')

    plt.show()

# --- 使用示例 ---
if __name__ == "__main__":
    # 替换为你想要处理的图片路径
    input_file = "out.png"  
    output_file = "out_smooth.png"
    
    # kernel_size=3 表示非常轻微的平滑 (推荐用于去噪且不想丢失太多细节)
    # kernel_size=5 会更模糊一点
    convolution_smooth(input_file, output_file, kernel_size=3)