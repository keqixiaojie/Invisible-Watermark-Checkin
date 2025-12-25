# utils/qr_generator.py
import qrcode
import cv2
import numpy as np
import os
from config import QRMOAEL_FOLDER as model_folder

# print(model_folder)
class QRGeometricCorrector:
    def __init__(self, model_dir=model_folder):
        # 初始化微信二维码识别引擎
        # 请确保 model_dir 文件夹内包含 4 个模型文件
        
        detect_pro = os.path.join(model_dir, "detect.prototxt")
        detect_caf = os.path.join(model_dir, "detect.caffemodel")
        sr_pro = os.path.join(model_dir, "sr.prototxt")
        sr_caf = os.path.join(model_dir, "sr.caffemodel")
        print(detect_pro)
        
        try:
            # 检查文件是否存在
            if not all([os.path.exists(p) for p in [detect_pro, detect_caf, sr_pro, sr_caf]]):
                 print("错误：模型文件夹中缺少必要文件，请检查路径。")
                 exit()
            self.detector = cv2.wechat_qrcode_WeChatQRCode(detect_pro, detect_caf, sr_pro, sr_caf)
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit()
            
    def align_and_crop(self, image_input, output_size=1024):
        """
        对输入的图像执行透视校正
        :param image_input: 可以是图片路径(str)，也可以是已读取的图像(numpy array)
        :param output_size: 目标画布大小，默认 1024
        :return: (校正后的图像, 识别出的二维码内容)
        """
        # 支持路径输入或直接的 numpy 数组输入
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input

        if img is None:
            return None, None

        # 1. 基础预处理：中值滤波去除传感器噪点
        img_blur = cv2.medianBlur(img, 3)

        # 2. 使用微信引擎定位四个关键顶点
        contents, points = self.detector.detectAndDecode(img_blur)

        if len(points) > 0:
            # 微信引擎返回点序: [左上, 右上, 右下, 左下]
            src_pts = points[0].astype("float32")
            
            # 3. 定义 1024x1024 标准画布上的映射点
            margin = 40  # 留白边距，防止边缘剪裁导致水印信息丢失
            dst_pts = np.array([
                [margin, margin],
                [output_size - 1 - margin, margin],
                [output_size - 1 - margin, output_size - 1 - margin],
                [margin, output_size - 1 - margin]
            ], dtype="float32")

            # 4. 计算并执行透视变换
            # getPerspectiveTransform 会处理旋转问题，将二维码“扶正”
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # 使用 INTER_LANCZOS4 插值保持频域信息完整，这对 DFT 水印至关重要
            warped = cv2.warpPerspective(img, matrix, (output_size, output_size), flags=cv2.INTER_LANCZOS4)
            
            # 返回校正后的图及其内容（内容可用于从数据库查找 metadata）
            return warped, contents[0]
        else:
            return None, None
        
    def process_image(self, img, output_size=1024):
        img = cv2.medianBlur(img, 3)
        if img is None: return None

        # 使用微信引擎检测
        # res: 识别出的字符串内容列表
        # points: 对应的四个顶点坐标列表，核心在于引擎已经帮我们排好序了
        res, points = self.detector.detectAndDecode(img)

        if len(points) > 0:
            # --- 关键修改点开始 ---
            
            # 获取引擎返回的原始点序。微信引擎返回的顺序通常是：
            # 索引0: 左上角定位符中心
            # 索引1: 右上角定位符中心
            # 索引2: 右下角(无定位符区域)
            # 索引3: 左下角定位符中心
            src_pts = points[0].astype("float32")
            
            # 定义目标 1024x1024 画布上的四个对应点
            # 顺序必须与 src_pts 严格保持一致：[左上, 右上, 右下, 左下]
            margin = 40 # 留白边距
            dst_pts = np.array([
                [margin, margin],                         # 映射到目标画布的左上
                [output_size - 1 - margin, margin],       # 映射到目标画布的右上
                [output_size - 1 - margin, output_size - 1 - margin], # 映射到目标画布的右下
                [margin, output_size - 1 - margin]        # 映射到目标画布的左下
            ], dtype="float32")

            # 计算透视变换矩阵。
            # 因为 src_pts 里的第一个点（二维码结构的左上角）被映射到了 dst_pts 的第一个点（画布的左上角），
            # 所以即使原图中二维码是倒着的，变换后也会自动正过来。
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # 执行变换，使用高质量插值
            warped = cv2.warpPerspective(img, M, (output_size, output_size), flags=cv2.INTER_LANCZOS4)
            
            # --- 关键修改点结束 ---
            
            return warped
        else:
            return None




# def generate_base_qr(content: str, save_path: str):
#     """生成基础二维码"""
#     qr = qrcode.QRCode(version=1, box_size=10, border=4)
#     qr.add_data(content)
#     qr.make(fit=True)
    
#     img = qr.make_image(fill_color="black", back_color="white")
#     # PIL -> Numpy
#     img_np = np.array(img.convert('L')) # 转为灰度 Numpy 数组
#     processor = QRGeometricCorrector(model_folder)
#     img = processor.process_image(img_np)
#     # img.save(save_path)
#     cv2.imwrite(save_path, img)
#     print(f"二维码已保存到: {save_path}")
    
def generate_base_qr(data: str, save_path: str):
    """
    生成基础二维码并保存
    """
    # 1. 配置二维码参数
    qr = qrcode.QRCode(
        version=1, 
        box_size=10, 
        border=4,
        error_correction=qrcode.constants.ERROR_CORRECT_H # 高容错率，适合加水印
    )
    qr.add_data(data)
    qr.make(fit=True)

    # 2. 生成 PIL 图片
    pil_img = qr.make_image(fill_color="black", back_color="white")
    
    # 3. 【核心修复】将 PIL Image 转换为 OpenCV (NumPy) 格式
    # PIL 是 RGB，OpenCV 需要 BGR 或 灰度
    # 先转为 RGB 模式的 numpy 数组
    open_cv_image = np.array(pil_img.convert('RGB')) 
    
    # Convert RGB to BGR (OpenCV 标准格式) 
    # [:, :, ::-1] 是将通道顺序反转
    open_cv_image = open_cv_image[:, :, ::-1].copy() 

    # 4. 如果你有后处理逻辑 (比如 process_image)，现在传入 numpy 数组就不会报错了
    # 假设你原本的代码逻辑是这样的：
    processor = QRGeometricCorrector(model_folder)
    # if processor:
    #     open_cv_image = processor.process_image(open_cv_image)
    open_cv_image = processor.process_image(open_cv_image)
    
    # 5. 保存图片
    cv2.imwrite(save_path, open_cv_image)
    print(f"二维码已保存到: {save_path}")