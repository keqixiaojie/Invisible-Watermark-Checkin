import requests
import time
import base64
import cv2
import numpy as np
import json
import os
from utils.watermark_core import WatermarkEncoder
from utils.qr_generator import generate_base_qr
from config import BASE_DIR

# --- 配置 ---
HOST = "http://127.0.0.1:8000"
COURSE_ID = 7  # 【注意】改成你正在运行的那个 Course ID
GROUP_ID = 0   # 默认第一组

def run_test():
    print(f"🚀 开始测试课程 {COURSE_ID} 的验证流程...")

    # 1. 获取 Session 信息 (模拟获取当前时间)
    # 我们需要知道 Session 什么时候开始的，才能算出现在的 index
    # 这一步通常前端通过 active_session 接口拿，或者扫码得到的
    try:
        resp = requests.get(f"{HOST}/courses/{COURSE_ID}/active_session")
        data = resp.json()
        if not data['active']:
            print("❌ 错误：请先在教师后台【开启大屏签到】！")
            return
        
        session_id = data['session_id']
        # 后端没直接返回 start_time，我们假设大屏已经开了一会儿
        # 这里我们需要去数据库看一眼 start_time，或者...
        # 为了测试方便，我们直接假设误差在允许范围内，
        # 我们用“现在”作为提交时间，倒推 index
        
        # 但是！脚本生成图片需要 seed，seed 需要读取本地文件
        meta_path = BASE_DIR / "static" / "courses" / str(COURSE_ID) / f"group_{GROUP_ID}" / "group_config.json"
        if not meta_path.exists():
            print(f"❌ 本地找不到配置文件: {meta_path}")
            return
            
        with open(meta_path, "r") as f:
            config = json.load(f)
            
        seed_wm = config['seed_wm']
        seed_dct = config['seed_dct']
        
        print(f"✅ 读取到种子: WM={seed_wm}, DCT={seed_dct}")

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 2. 生成模拟图片
    # 我们不知道确切的 start_time，无法算出精确的 Theoretical ID。
    # 这是一个“黑盒测试”的难点。
    # 变通方法：我们生成 3 帧图片，分别对应 ID=0, ID=5, ID=10
    # 发送给后端，看后端返回的 Log (Verify Debug) 里说理论 ID 是多少
    # 如果其中有一帧报错“ID误差X”，我们就能推算出正确的 ID。
    
    # 或者，我们直接暴力一点：
    # 假设大屏刚开不久，我们测试生成第 0, 1, 2 张图
    
    print("📸 生成模拟帧...")
    
    # 假设大屏刚开不久，我们测试生成第 0, 1, 2 张图
    target_indices = [0, 1, 2, 3, 4] 
    
    frames = []
    base_qr = "temp_qr.png"
    # 这里不需要指定固定的 logo 了，要在循环里动态指定
    
    generate_base_qr(f"test?cid={COURSE_ID}&gid={GROUP_ID}", base_qr)
    
    # 初始化编码器 (mod=40)
    # encoder = WatermarkEncoder(seed_wm=seed_wm, seed_dct=seed_dct, mod=4096) 
    
    for idx in target_indices:
        save_name = f"test_frame_{idx}.png"
        
        # 【修正点】读取对应的数字水印图
        # 假设你的 assets 文件夹就在项目根目录的 static/assets 下
        logo_filename = f"{idx}.png"
        logo_path = BASE_DIR / "static" / "assets" / logo_filename
        
        if not logo_path.exists():
            print(f"❌ 找不到测试水印图: {logo_path}")
            return

        # 模拟动态种子 (tasks.py 里的逻辑)
        # 重新实例化以确保状态纯净
        temp_enc = WatermarkEncoder(seed_wm=seed_wm, seed_dct=seed_dct, mod=4096)
        
        # 嵌入：将数字图片(logo_path) 嵌入到 二维码(base_qr) 中
        temp_enc.encode(base_qr, str(logo_path), save_name)
        
        with open(save_name, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
            frames.append(b64)
            
        print(f"   已生成第 {idx} 帧 (嵌入了 {logo_filename})")
        
        # os.remove(save_name)

    # 3. 发送请求
    print(f"📤 发送 {len(frames)} 帧到后端验证...")
    verify_data = {
        "course_id": COURSE_ID,
        "frames": frames
    }
    
    res = requests.post(f"{HOST}/student/verify", json=verify_data)
    print("\n🔍 后端返回结果:")
    print(json.dumps(res.json(), indent=2, ensure_ascii=False))
    
    if res.json().get("success"):
        print("\n🎉🎉🎉 验证通过！逻辑闭环达成！")
    else:
        print("\n⚠️ 验证失败。请检查：")
        print("1. 大屏是否开启？(必须 active)")
        print("2. 脚本里的 COURSE_ID 和大屏一致吗？")
        print("3. 时间误差：看后端控制台输出的 [验证调试] 信息，理论ID是多少？")

if __name__ == "__main__":
    run_test()