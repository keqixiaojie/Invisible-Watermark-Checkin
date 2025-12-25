import cv2
import numpy as np
import os
from pathlib import Path
import traceback

def diagnose_wechat_qrcode(model_paths):
    """
    详细诊断 WeChatQRCode 问题
    """
    print("=" * 60)
    print("WeChatQRCode 深度诊断")
    print("=" * 60)
    
    # 1. 检查 OpenCV 版本和编译信息
    print(f"\n1. OpenCV 版本信息:")
    print(f"  版本: {cv2.__version__}")
    print(f"  构建信息: {cv2.getBuildInformation()[:500]}...")
    
    # 2. 检查模型文件
    print(f"\n2. 模型文件检查:")
    for name, path in model_paths.items():
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"  {name}:")
        print(f"    路径: {path}")
        print(f"    存在: {'✓' if exists else '✗'}")
        if exists:
            print(f"    大小: {size:,} 字节 ({size/1024/1024:.2f} MB)")
            
            # 尝试读取文件头
            try:
                with open(path, 'rb') as f:
                    header = f.read(100)
                    print(f"    文件头: {header[:20].hex()}...")
            except:
                print("    无法读取文件头")
    
    # 3. 尝试不同初始化方式
    print(f"\n3. 尝试初始化...")
    
    # 方法1：4个参数
    try:
        print("  尝试方法1 (4参数)...")
        detector = cv2.wechat_qrcode_WeChatQRCode(
            model_paths['detect_prototxt'],
            model_paths['detect_caffemodel'],
            model_paths['sr_prototxt'],
            model_paths['sr_caffemodel']
        )
        print("  ✓ 方法1 成功")
        return detector
    except Exception as e1:
        print(f"  ✗ 方法1 失败: {e1}")
    
    # 方法2：2个参数
    try:
        print("  尝试方法2 (2参数)...")
        detector = cv2.wechat_qrcode_WeChatQRCode(
            model_paths['detect_prototxt'],
            model_paths['detect_caffemodel']
        )
        print("  ✓ 方法2 成功")
        return detector
    except Exception as e2:
        print(f"  ✗ 方法2 失败: {e2}")
    
    # 方法3：使用 cv2.wechat_qrcode_WeChatQRCode.create()
    try:
        print("  尝试方法3 (create方法)...")
        detector = cv2.wechat_qrcode_WeChatQRCode.create(
            model_paths['detect_prototxt'],
            model_paths['detect_caffemodel'],
            model_paths['sr_prototxt'],
            model_paths['sr_caffemodel']
        )
        print("  ✓ 方法3 成功")
        return detector
    except Exception as e3:
        print(f"  ✗ 方法3 失败: {e3}")
    
    # 4. 测试文件完整性
    print(f"\n4. 测试文件完整性...")
    for name, path in model_paths.items():
        try:
            with open(path, 'rb') as f:
                content = f.read()
                print(f"  {name}: 可读取，大小 {len(content):,} 字节")
                
                # 检查是否是有效的 Caffe 模型（以 'Caff' 开头）
                if name.endswith('.caffemodel'):
                    if content[:4] == b'Caff':
                        print(f"    ✓ 有效的 Caffe 模型文件")
                    else:
                        print(f"    ⚠ 文件头不是 Caffe 格式: {content[:4]}")
                        
        except Exception as e:
            print(f"  {name}: 读取失败 - {e}")
    
    return None

# 使用示例
if __name__ == "__main__":
    # 你的模型路径
    MODEL_DIR = Path(r"C:\Users\31855\Desktop\works\数字图像处理\Invisible-Watermark-Checkin\QR_watermark\QR_model")
    
    model_paths = {
        'detect_prototxt': str(MODEL_DIR / "detect.prototxt"),
        'detect_caffemodel': str(MODEL_DIR / "detect.caffemodel"),
        'sr_prototxt': str(MODEL_DIR / "sr.prototxt"),
        'sr_caffemodel': str(MODEL_DIR / "sr.caffemodel")
    }
    
    detector = diagnose_wechat_qrcode(model_paths)
    
    if detector:
        print("\n✅ 检测器初始化成功！")
        # 测试检测
        test_image = "test_qr.png"
        if os.path.exists(test_image):
            img = cv2.imread(test_image)
            if img is not None:
                results = detector.detectAndDecode(img)
                print(f"检测结果: {results}")
            else:
                print(f"无法读取测试图片: {test_image}")
    else:
        print("\n❌ 检测器初始化失败")