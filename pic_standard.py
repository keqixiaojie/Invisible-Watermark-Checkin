import cv2
import numpy as np
import os

class QRPreprocessor:
    def __init__(self, model_dir):
        # 初始化微信二维码识别引擎
        # 请确保 model_dir 文件夹内包含 4 个模型文件
        detect_pro = os.path.join(model_dir, "detect.prototxt")
        detect_caf = os.path.join(model_dir, "detect.caffemodel")
        sr_pro = os.path.join(model_dir, "sr.prototxt")
        sr_caf = os.path.join(model_dir, "sr.caffemodel")
        
        try:
            # 检查文件是否存在
            if not all([os.path.exists(p) for p in [detect_pro, detect_caf, sr_pro, sr_caf]]):
                 print("错误：模型文件夹中缺少必要文件，请检查路径。")
                 exit()
            self.detector = cv2.wechat_qrcode_WeChatQRCode(detect_pro, detect_caf, sr_pro, sr_caf)
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit()

    def process_image(self, img_path, output_size=1024):
        img = cv2.imread(img_path)
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

def batch_process(input_folder, output_folder, model_folder):
    processor = QRPreprocessor(model_folder)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    supported_ext = ('.jpg', '.png', '.jpeg', '.bmp')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_ext):
            print(f"正在处理: {filename}...")
            result = processor.process_image(os.path.join(input_folder, filename))
            
            if result is not None:
                # 保存为高质量 PNG 以避免压缩噪点
                output_path = os.path.join(output_folder, f"fixed_{filename.rsplit('.', 1)[0]}.png")
                cv2.imwrite(output_path, result)
                print(f"  -> 已保存: {output_path}")
            else:
                print(f"  -> 失败: 无法识别/定位二维码: {filename}")

if __name__ == "__main__":
    # --- 配置区域 ---
    # 1. 请确保下载了 detect.prototxt, detect.caffemodel, sr.prototxt, sr.caffemodel 并放入此文件夹
    MODEL_DIR = "./QR_model" 
    # 2. 原始图片文件夹
    INPUT_DIR = "./pic/orig_code"
    # 3. 输出文件夹
    OUTPUT_DIR = "./pic/prep_code"
    
    batch_process(INPUT_DIR, OUTPUT_DIR, MODEL_DIR)