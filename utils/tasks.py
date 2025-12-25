import os
import json
import shutil
import random
import numpy as np
import asyncio
import traceback
from pathlib import Path
from typing import List, Dict

# 引入线程池工具
from starlette.concurrency import run_in_threadpool

from utils.watermark_core import WatermarkEncoder
from utils.qr_generator import generate_base_qr
from config import BASE_DIR

# 全局进度存储
TASK_PROGRESS: Dict[int, dict] = {}

def get_unique_seeds(count: int, used_seeds: set) -> List[int]:
    seeds = []
    while len(seeds) < count:
        s = random.randint(1000, 999999)
        if s not in used_seeds:
            seeds.append(s)
            used_seeds.add(s)
    return seeds

async def generate_course_watermarks(course_id: int, server_domain: str):
    """
    后台任务：为课程生成3组水印二维码序列
    [性能优化版] 使用 run_in_threadpool 释放主线程
    """
    try:
        TASK_PROGRESS[course_id] = {
            "status": "processing",
            "progress": 0,
            "message": "正在初始化..."
        }
        
        print(f"--- [后台任务] 开始为课程 {course_id} 生成水印序列 ---")
        
        # course_root = BASE_DIR / "static" / "courses" / str(course_id)
        # if course_root.exists():
        #     shutil.rmtree(course_root)
        # course_root.mkdir(parents=True)
        # 1. 准备主目录
        course_root = BASE_DIR / "static" / "courses" / str(course_id)
        if course_root.exists():
            try:
                # ignore_errors=True 防止因为某个文件被占用导致整个删除失败
                shutil.rmtree(course_root, ignore_errors=True)
            except Exception as e:
                print(f"清理旧目录失败: {e} (但这通常不影响覆盖写入)")
                
        # exist_ok=True 防止上面没删干净导致这里报错
        course_root.mkdir(parents=True, exist_ok=True)

        wm_path = BASE_DIR / "static" / "assets" 
        if not wm_path.exists():
            error_msg = "错误：水印素材不存在"
            print(error_msg)
            TASK_PROGRESS[course_id] = {"status": "error", "progress": 0, "message": error_msg}
            return

        all_groups_info = []
        used_seeds = set()
        
        total_groups = 3
        images_per_group = 100 
        total_steps = total_groups * images_per_group
        current_step = 0

        for g_idx in range(total_groups):
            group_dir = course_root / f"group_{g_idx}"
            img_dir = group_dir / "images"
            meta_dir = group_dir 
            # 【修改点】加上 exist_ok=True，如果文件夹存在就不报错，直接用
            img_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)

            # 生成基础二维码 (这也是IO操作，但这步很快，可以直接跑)
            # 注意：请确保这里的 IP 是你局域网 IP，否则手机扫不开
            
            # 【修改这里】使用传入的 server_domain
            # 确保 server_domain 没有末尾的斜杠，且路径拼写正确
            if not server_domain: server_domain = "http://192.168.1.100:8000"
            base_qr_content = f"{server_domain}/pages/student_scan.html?cid={course_id}&gid={g_idx}"
            
            base_qr_path = group_dir / "base_qr.png"
            generate_base_qr(base_qr_content, str(base_qr_path))

            seeds = get_unique_seeds(2, used_seeds)
            seed_wm, seed_dct = seeds[0], seeds[1]

            encoder = WatermarkEncoder(seed_wm=seed_wm, seed_dct=seed_dct, mod=4096)

            group_image_urls = []
            shared_metadata = None

            for i in range(images_per_group):
                file_name = f"{i:02d}.png"
                logo_path = wm_path / f"{i}.png"
                if not logo_path.exists(): logo_path = wm_path / "wm_logo.png"
                
                save_abs_path = img_dir / file_name
                
                # 【核心优化】放到线程池里跑！
                # 这一步是最耗时的，现在它不会阻塞主线程了
                metadata = await run_in_threadpool(
                    encoder.encode, 
                    str(base_qr_path), 
                    str(logo_path), 
                    str(save_abs_path)
                )
                
                if i == 0: shared_metadata = metadata
                
                rel_path = f"/pages/courses/{course_id}/group_{g_idx}/images/{file_name}"
                group_image_urls.append(rel_path)

                # 更新进度
                current_step += 1
                if i % 4 == 0: # 稍微降低更新频率，减少 dict 操作开销
                    percent = int((current_step / total_steps) * 100)
                    TASK_PROGRESS[course_id] = {
                        "status": "processing",
                        "progress": percent,
                        "message": f"第 {g_idx+1} 组: {i}/{images_per_group}"
                    }
                    # 这里的 sleep 依然保留，给 Event Loop 喘息机会
                    await asyncio.sleep(0.01)

            # 保存 Metadata
            mask_path = meta_dir / "ignore_mask.npy"
            levels_path = meta_dir / "global_levels.npy"
            np.save(str(mask_path), shared_metadata["ignore_mask"])
            np.save(str(levels_path), shared_metadata["global_levels"])

            group_config = {
                "group_id": g_idx,
                "seed_wm": seed_wm,
                "seed_dct": seed_dct,
                "wm_shape": shared_metadata["wm_shape"],
                "base_qr": f"/pages/courses/{course_id}/group_{g_idx}/base_qr.png",
                "mask_file": str(mask_path.relative_to(BASE_DIR)),
                "levels_file": str(levels_path.relative_to(BASE_DIR)),
                "images": group_image_urls
            }
            
            with open(meta_dir / "group_config.json", "w") as f:
                json.dump(group_config, f)
            all_groups_info.append(group_config)

        master_meta = {
            "course_id": course_id,
            "total_groups": total_groups,
            "groups": all_groups_info
        }
        with open(course_root / "metadata.json", "w") as f:
            json.dump(master_meta, f)
            
        print(f"--- [任务完成] 课程 {course_id} ---")
        
        TASK_PROGRESS[course_id] = {
            "status": "completed",
            "progress": 100,
            "message": "完成"
        }
        
    except Exception as e:
        print(f"后台任务出错: {e}")
        traceback.print_exc()
        TASK_PROGRESS[course_id] = {
            "status": "error",
            "progress": 0,
            "message": f"失败: {str(e)}"
        }