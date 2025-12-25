import onnxruntime as ort
import numpy as np
import cv2
import os
from config import DeepModel_FOLDER

# 拼接相对路径: utils/digit_recognizer.onnx
# 使用 .as_posix() 将反斜杠 \ 转换为正斜杠 /，这对 C++ 库更友好
model_path_obj = DeepModel_FOLDER / "digit_recognizer.onnx"
model_path = model_path_obj.as_posix()

class FastDigitRecognizer:
    def __init__(self, model_file=model_path):
        print(f"[Init] Loading ONNX Model (Relative): {model_file}")
        
        # 1. 检查文件是否存在 (这一步 Python 处理，不会报错)
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"ONNX模型文件未找到: {model_file}")

        # 2. 尝试创建 Session
        try:
            # 方案 A: 直接传相对路径字符串
            # 这里的 providers 强制指定 CPU，避免 CUDA 带来的额外路径问题
            self.session = ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
            print("[Init] Session created successfully (Path Mode)")
            
        except Exception as e:
            print(f"[Warning] Path load failed ({e}), trying Memory Load...")
            try:
                # 方案 B (保底): 读取二进制流加载
                # 如果相对路径字符串还是让 C++ 库崩溃，我们直接把文件读进内存传给它
                with open(model_file, "rb") as f:
                    model_bytes = f.read()
                self.session = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
                print("[Init] Session created successfully (Memory Mode)")
            except Exception as e2:
                print(f"[Fatal] Model loading failed completely.")
                raise e2

        self.input_name = self.session.get_inputs()[0].name

    def predict_array(self, img_array):
        """直接接收内存中的 Numpy 数组进行识别"""
        
        # 1. 预处理：确保为 64x64 灰度图
        if len(img_array.shape) == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            img = img_array
            
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
        
        # 2. 归一化并增加 batch/channel 维度
        img = img.astype(np.float32) / 255.0
        # 模型输入维度通常是 (1, 1, 64, 64)
        img = img[None, None, :, :]
        
        # 3. 推理
        outputs = self.session.run(None, {self.input_name: img})
        logits = outputs[0][0]
        
        # 4. 返回识别出的数字字符串
        return f"{int(np.argmax(logits)):02d}"