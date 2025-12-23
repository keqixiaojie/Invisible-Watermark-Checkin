import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_y_histogram(input_folder, output_folder):
    """
    分析指定文件夹中每张图片的亮度(Y通道)直方图并保存结果
    """
    # 1. 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"已创建输出文件夹: {output_folder}")

    # 支持的图片格式
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # 2. 获取所有文件
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    count = 0
    for file_name in files:
        # 检查扩展名
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in valid_extensions:
            continue

        file_path = os.path.join(input_folder, file_name)
        
        # 3. 读取图片 (OpenCV 默认为 BGR)
        img_bgr = cv2.imread(file_path)
        
        if img_bgr is None:
            print(f"无法读取文件: {file_name}")
            continue

        # 4. 颜色空间转换
        # 用于显示的 RGB 图
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 用于分析的 YUV 图 (Y: 亮度, U: 色度, V: 浓度)
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        
        # 提取 Y 通道 (第0个通道)
        y_channel = img_yuv[:, :, 0]

        # 5. 计算直方图
        # cv2.calcHist(images, channels, mask, histSize, ranges)
        hist_y = cv2.calcHist([y_channel], [0], None, [256], [0, 256])

        # 6. 可视化绘图
        plt.figure(figsize=(12, 5))
        
        # 左图：显示原图
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title(f"Original: {file_name}")
        plt.axis('off')

        # 右图：显示亮度直方图
        plt.subplot(1, 2, 2)
        plt.plot(hist_y, color='black')
        plt.xlim([0, 256])
        plt.title("Luminance (Y-Channel) Histogram")
        plt.xlabel("Pixel Intensity (0=Black, 255=White)")
        plt.ylabel("Pixel Count")
        plt.grid(True, alpha=0.3)
        
        # 填充曲线下方的区域，看起来更直观
        plt.fill_between(range(256), hist_y.flatten(), color='gray', alpha=0.5)

        # 7. 保存结果图
        save_name = f"hist_{os.path.splitext(file_name)[0]}.png"
        save_path = os.path.join(output_folder, save_name)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() # 关闭画布释放内存

        print(f"已处理并保存: {save_name}")
        count += 1

    print(f"\n处理完成！共分析了 {count} 张图片。")
    print(f"结果保存在: {output_folder}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 请将此处修改为你的图片文件夹路径
    input_dir = "./pic/output"       # 你的图片所在的文件夹
    output_dir = "analysis_result" # 结果保存的文件夹
    
    # 简单的容错：如果输入文件夹不存在，提示用户
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"错误：找不到输入文件夹 '{input_dir}'。已为您创建该空文件夹，请放入图片后重试。")
    else:
        analyze_y_histogram(input_dir, output_dir)