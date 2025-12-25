# utils/verification_service.py

import time
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

# 引入你的模块
import utils
from utils.WatermarkRecognizer import FastDigitRecognizer
from utils.watermark_core import WatermarkDecoder
from utils.qr_generator import QRGeometricCorrector
from config import BASE_DIR # 使用统一的 BASE_DIR
print(f"🔥 [真相大白] 正在加载的 Recognizer 文件路径: {utils.WatermarkRecognizer.__file__}")

class SignInVerificationService:
    def __init__(self):
        # 基础路径指向 static/courses
        self.base_path = BASE_DIR / "static" / "courses"
        
        self.corrector = QRGeometricCorrector() 
        self.recognizer = FastDigitRecognizer() 
        
        # 【关键修改】必须与大屏前端 FRAME_DURATION = 3000 保持一致
        self.scroll_speed = 2.0  

    def _get_metadata(self, course_id: int, gid: int) -> Dict:
        """
        路径必须与 tasks.py 生成逻辑严格匹配：
        static/courses/{id}/group_{gid}
        """
        group_path = self.base_path / str(course_id) / f"group_{gid}"
        meta_dir = group_path 
        
        if not meta_dir.exists():
            raise FileNotFoundError(f"元数据目录不存在: {meta_dir}")

        # 加载 .npy
        ignore_mask = np.load(str(meta_dir / "ignore_mask.npy"))
        global_levels = np.load(str(meta_dir / "global_levels.npy"))
        
        # 加载配置
        with open(meta_dir / "group_config.json", "r") as f:
            config = json.load(f)
            
        return {
            "ignore_mask": ignore_mask,
            "global_levels": global_levels,
            "seed_wm": config["seed_wm"],
            "seed_dct": config["seed_dct"],
            "wm_shape": tuple(config["wm_shape"])
        }

    def verify_sign_in(self, user_id: str, course_id: int, frames: List[np.ndarray], start_time: float, submit_time: float) -> Dict:
        report = {
            "user_id": user_id,
            "success": False,
            "reason": "",
            "extracted_ids": [],
            "timestamp": time.time()
        }
        
        extracted_sequence = []
        qr_gid = None

        print(f"\n🔍 [开始验证] 收到 {len(frames)} 帧图片") # <--- Debug
        # 1. 遍历所有帧进行提取
        frames = frames[::2]  # 抽帧，减轻服务器压力
        
        for i, frame in enumerate(frames):
            # A. 几何校正
            aligned, qr_content = self.corrector.align_and_crop(frame)
            if aligned is None:
                print(f"❌ [帧 {2*i}] 几何校正失败：未检测到二维码或定位点") # <--- Debug
                continue 
            
            print(f"✅ [帧 {2*i}] 定位成功，内容: {qr_content}") # <--- Debug
            # B. 简单的二维码内容校验
            if f"cid={course_id}" not in qr_content:
                continue # 拍错课程了，忽略这一帧

            # C. 解析 gid (只解析一次即可)
            if qr_gid is None:
                try:
                    import urllib.parse as urlparse
                    parsed = urlparse.urlparse(qr_content)
                    qr_gid = int(urlparse.parse_qs(parsed.query).get('gid', [0])[0])
                except:
                    qr_gid = 0

            # D. 获取元数据并解码
            try:
                metadata = self._get_metadata(course_id, qr_gid)
                decoder = WatermarkDecoder(seed_wm=metadata['seed_wm'], seed_dct=metadata['seed_dct'])
                
                # 解码水印
                wm_img = decoder.decode(aligned, metadata)
                
                # 识别数字
                digit_str = self.recognizer.predict_array(wm_img)
                extracted_sequence.append(int(digit_str))
                print(f"🔢 [帧 {i*2}] 水印提取结果: '{digit_str}'") # <--- Debug: 看看提取出了什么？
                # 如果识别器返回 None 或 空字符串，说明提取出的图太烂了
                if digit_str is not None and str(digit_str).isdigit():
                    extracted_sequence.append(int(digit_str))
                else:
                    print(f"⚠️ [帧 {i*2}] 无法识别为数字")
                
            except Exception as e:
                print(f"❌ 【帧 {i*2}】 处理失败: {e}")
                continue

        # 2. 最终判定
        report["extracted_ids"] = extracted_sequence
        
        if not extracted_sequence:
            report["reason"] = "未能从任何图像中提取有效水印，请靠近大屏重试"
            return report

        is_passed, fail_reason = self._final_check(extracted_sequence, start_time, submit_time)
        
        report["success"] = is_passed
        report["reason"] = fail_reason
        
        return report

    # def _final_check(self, sequence: List[int], t_start: float, t_submit: float) -> Tuple[bool, str]:
    #     # A. 序列一致性校验
    #     # 学生连拍5张，通常应该得到 [5, 5, 5, 5, 5] (图片没变) 或者 [5, 5, 6, 6, 6] (刚好切换)
    #     # 我们取众数作为“提取到的ID”
    #     from collections import Counter
    #     counts = Counter(sequence)
    #     most_common_id, _ = counts.most_common(1)[0]

    #     # B. 时间对齐校验 (核心防伪)
    #     # 理论ID = (提交时间 - 开始时间) / 3.0
    #     elapsed = t_submit - t_start
    #     theoretical_id = int((elapsed / self.scroll_speed) % 100)
        
    #     # 计算环形距离 (因为 99 后面是 00)
    #     dist = abs(most_common_id - theoretical_id)
    #     time_error = min(dist, 100 - dist)
        
    #     # 【关键】宽容度设置
    #     # 大屏缓冲可能延迟 1-3秒，网络传输 1秒，学生举起手机 2秒
    #     # 允许 ±3 帧 (即 ±6秒) 的误差是合理的
    #     TOLERANCE_FRAMES = 3
        
    #     print(f"[验证调试] 耗时:{elapsed:.1f}s | 理论ID:{theoretical_id} | 提取ID:{most_common_id} | 误差:{time_error}")

    #     if time_error > TOLERANCE_FRAMES:
    #         return False, f"验证超时或非实时拍摄 (ID误差 {time_error})"

    #     return True, "验证通过"
    def _final_check(self, sequence: List[int], t_start: float, t_submit: float) -> Tuple[bool, str]:
        # 1. 序列去重并排序，看看提取到了几个不同的 ID
        unique_ids = sorted(list(set(sequence)))
        
        # 2. 计算理论 ID (Theoretical ID)
        # 注意：这里我们取这批图片的“中间时刻”作为基准，或者直接用 submit_time
        elapsed = t_submit - t_start
        theoretical_id = int((elapsed / self.scroll_speed) % 100)
        
        print(f"[验证逻辑] 序列:{sequence} | 理论ID:{theoretical_id} (耗时{elapsed:.1f}s)")

        # 3. 策略判断
        
        # 策略 A: 必须包含理论 ID (允许 ±2 误差)
        # 只要识别出的数字里，有一个落在 [理论值-2, 理论值+2] 的区间内，就说明拍到了对的图
        hit = False
        for uid in unique_ids:
            # 环形距离计算
            dist = abs(uid - theoretical_id)
            err = min(dist, 100 - dist)
            if err <= 2: # 误差容忍度
                hit = True
                break
        
        if not hit:
            # 如果所有的数字都离谱地远 -> 可能是拍了别人的照片/录像，或者大屏时间严重不同步
            return False, f"时间验证未通过 (期待 {theoretical_id} 附近, 实际 {unique_ids})"

        # 策略 B: 静态 vs 动态
        # 如果只拍到了 1 个 ID (例如 [5, 5, 5])
        if len(unique_ids) == 1:
            # 虽然时间对上了，但没有跨帧。
            # 为了严谨，我们应该要求跨帧。但在实际体验中，如果要求跨帧，学生可能要举很久。
            # 折中方案：如果时间对得很准，且图片清晰，也可以过。
            # 或者返回特定错误码让前端继续采。
            
            # 这里我们选择【严格模式】：必须要有动态变化才能证明是“活体”
            return False, "信号静止，请继续保持扫描..." 
            
            # # 或者【宽松模式】：只要时间对上就算过 (适合网速极慢的情况)
            # return True, "验证通过 (静态帧匹配)"

        # 如果拍到了多个 ID (例如 [5, 6]) -> 完美！
        return True, "验证通过 (动态帧匹配)"