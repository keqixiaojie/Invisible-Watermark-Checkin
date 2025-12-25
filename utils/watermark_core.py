# 重构后的水印核心代码


import numpy as np
import cv2
from pywt import dwt2, idwt2
import os
from collections import defaultdict
from scipy.signal import find_peaks
import concurrent.futures
import multiprocessing


# ==========================================
# 基础辅助函数
# ==========================================
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv_img

def cv_imwrite(path, img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0, 255).astype(np.uint8)
    suffix = os.path.splitext(path)[-1] or '.png'
    try:
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder): os.makedirs(folder)
        is_success, buffer = cv2.imencode(suffix, img)
        if is_success: buffer.tofile(path)
    except Exception as e: print(f"保存出错: {e}")


# ==========================================
# 优化版：局部亮度重组器
# ==========================================
class BrightnessRankQuantizer:
    def __init__(self, n_levels=3, block_shape=(4,4), context_blocks=13):
        self.n_levels = n_levels
        self.block_shape = block_shape
        self.context_blocks = context_blocks if context_blocks % 2 != 0 else context_blocks + 1
        self.global_levels = None 

    def fit(self, img_y, ignore_mask=None):
        """分析亮度基准 (用于嵌入端生成 metadata)"""
        gray = img_y.astype(np.uint8)
        if ignore_mask is not None:
            if ignore_mask.shape != gray.shape:
                mask_resized = cv2.resize(ignore_mask.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                valid_pixels = gray[~mask_resized]
            else:
                valid_pixels = gray[~ignore_mask]
        else:
            valid_pixels = gray.flatten()

        hist, _ = np.histogram(valid_pixels, bins=256, range=[0, 256])
        temp_hist = hist.copy()
        temp_hist[:50] = 0 
        temp_hist[255] = 0 
        
        peaks, _ = find_peaks(temp_hist, height=np.max(temp_hist)*0.1, distance=20)
        candidates = sorted(list(set(list(peaks) + [255])))
        
        white_levels = np.array(candidates[-self.n_levels:]) if len(candidates) >= self.n_levels else np.linspace(candidates[0], 255, self.n_levels)
        self.global_levels = np.concatenate(([0], white_levels)).astype(int)
        return self.global_levels

    @staticmethod
    def _worker_apply_strip(args):
        """并行分片处理"""
        (padded_img, start_r, end_r, new_w, block_shape, context_blocks, global_levels) = args
        bh, bw = block_shape
        pad_h = (context_blocks // 2) * bh
        pad_w = (context_blocks // 2) * bw
        real_levels = global_levels[1:]
        
        processed_strip = np.zeros((end_r - start_r, new_w), dtype=np.float32)
        PIXEL_THRESH = 100
        
        for r in range(start_r, end_r, bh):
            strip_row = r - start_r
            for c in range(0, new_w, bw):
                roi = padded_img[r : r + context_blocks*bh, c : c + context_blocks*bw]
                center = roi[pad_h : pad_h + bh, pad_w : pad_w + bw]
                ctx_min, ctx_max = np.percentile(roi, [0, 100])
                
                mask = center >= PIXEL_THRESH
                out_block = center.copy().astype(np.float32)
                
                if not np.any(mask):
                    processed_strip[strip_row:strip_row+bh, c:c+bw] = out_block
                    continue
                
                if (ctx_max - ctx_min) < 10:
                    target = global_levels[(np.abs(global_levels - np.mean(center))).argmin()]
                    out_block[mask] = target
                else:
                    norm = np.clip((center - ctx_min) / (ctx_max - ctx_min + 1e-5), 0, 1) * 255.0
                    # 向量化寻找最近能级
                    dist = np.abs(norm[..., None] - real_levels[None, None, :])
                    out_block[mask] = real_levels[np.argmin(dist, axis=2)][mask]

                processed_strip[strip_row:strip_row+bh, c:c+bw] = out_block
        return processed_strip

    def apply(self, img_y, global_levels):
        """执行重组 (用于提取端)"""
        h, w = img_y.shape[:2]
        bh, bw = self.block_shape
        new_h, new_w = h - (h % bh), w - (w % bw)
        gray = img_y[:new_h, :new_w]
        
        pad_h, pad_w = (self.context_blocks // 2) * bh, (self.context_blocks // 2) * bw
        padded = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=10)
        
        num_cores = min(multiprocessing.cpu_count(), 8)
        chunk_h = max((new_h // bh // num_cores) * bh, bh)
        
        tasks = [(padded, r, min(r + chunk_h, new_h), new_w, self.block_shape, self.context_blocks, global_levels) 
                 for r in range(0, new_h, chunk_h)]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as ex:
            results = list(ex.map(self._worker_apply_strip, tasks))
            
        return np.vstack(results)



class WatermarkBase:
    """基础共有逻辑"""
    def __init__(self, mod=4096, block_shape=(4,4), dwt_deep=1):
        self.mod = mod
        self.block_shape = block_shape
        self.dwt_deep = dwt_deep

    def get_dwt_ll(self, img_y):
        ll = dwt2(img_y, 'haar')[0]
        for _ in range(self.dwt_deep - 1):
            ll = dwt2(ll, 'haar')[0]
        return ll

    def get_block_indices(self, ll_shape):
        bh, bw = self.block_shape
        rows, cols = ll_shape[0] // bh, ll_shape[1] // bw
        idx0, idx1 = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        return idx0.flatten(), idx1.flatten()

# ==========================================
# 嵌入端：由教师端调用
# ==========================================
class WatermarkEncoder(WatermarkBase):
    def __init__(self, seed_wm, seed_dct, mod=4096, alpha=0.9, center_alpha=0.8):
        """
        初始化嵌入器
        :param seed_wm: 水印像素打乱种子
        :param seed_dct: 频域分块顺序随机种子
        :param mod: 调制强度 (建议 4096)
        """
        super().__init__(mod)
        self.seed_wm = seed_wm
        self.seed_dct = seed_dct
        self.alpha = alpha
        self.center_alpha = center_alpha
        self.quantizer = BrightnessRankQuantizer(block_shape=self.block_shape)

    def _idwt_channel(self, ha, coeffs_list):
        """执行逆小波变换还原通道"""
        res = ha
        for i in range(len(coeffs_list)):
            res = idwt2((res, coeffs_list[-(i+1)]), 'haar')
        return res

    def encode(self, ori_path, wm_path, output_path):
        """
        执行水印嵌入并返回 Metadata
        """
        # 1. 加载并转换色彩空间
        img = cv_imread(ori_path).astype(np.float32)
        ori_h, ori_w = img.shape[:2]
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
        # 2. 生成忽略掩膜 (Ignore Mask)
        # 针对二维码黑块，阈值设为 100 以识别暗区
        _, thr = cv2.threshold(yuv[:,:,0].astype(np.uint8), 100, 255, cv2.THRESH_BINARY_INV)
        factor = 2 ** self.dwt_deep
        small_mask = cv2.resize(thr, (yuv.shape[1] // factor, yuv.shape[0] // factor), 
                                interpolation=cv2.INTER_AREA) > 127
        
        # 3. 准备水印数据
        wm = cv_imread(wm_path)[:,:,0]
        wm_shape = wm.shape
        wm_flat = wm.flatten()
        if self.seed_wm:
            np.random.RandomState(self.seed_wm).shuffle(wm_flat)
        
        # 4. 执行小波分解 (DWT)
        def get_dwt(data):
            coeffs = dwt2(data, 'haar')
            ll, coeffs_list = coeffs[0], [coeffs[1]]
            for _ in range(self.dwt_deep - 1):
                coeffs = dwt2(ll, 'haar')
                ll, coeffs_list.append(coeffs[1])
            return ll, coeffs_list

        ha_Y, coeffs_Y = get_dwt(yuv[:,:,0])
        ha_U, coeffs_U = get_dwt(yuv[:,:,1])
        ha_V, coeffs_V = get_dwt(yuv[:,:,2])
        
        # 5. 分块处理逻辑
        bh, bw = self.block_shape
        idx0, idx1 = self.get_block_indices(ha_Y.shape)
        new_ha_Y = ha_Y.copy()
        
        rnd_dct = np.random.RandomState(self.seed_dct)
        dct_idx = np.arange(bh * bw) # 用于消耗随机种子状态
        
        # 遍历所有分块执行 DFT 调制
        for i in range(len(idx0)):
            rnd_dct.shuffle(dct_idx) # 必须保持与原算法一致的随机数消耗
            r, c = idx0[i], idx1[i]
            
            # 检查当前块是否在忽略名单中 (黑块不嵌)
            if np.mean(small_mask[r*bh:(r+1)*bh, c*bw:(c+1)*bw]) > 0.5:
                continue
            
            # 提取 4x4 块
            block = ha_Y[r*bh:(r+1)*bh, c*bw:(c+1)*bw].copy()
            
            # 安全预处理: Shift to Center
            target_center = 128 + self.center_alpha * (np.mean(block) - 128)
            block = (block - target_center) * self.alpha + target_center
            
            # DFT 变换并调制幅度
            dft = cv2.dft(block, flags=cv2.DFT_COMPLEX_OUTPUT)
            mag = np.sqrt(dft[0,0,0]**2 + dft[0,0,1]**2)
            
            wm_bit = wm_flat[i % wm_flat.size]
            # 根据水印位决定量化落点 (0.75 或 0.25)
            target_mag = mag - mag % self.mod + (0.75 if wm_bit >= 128 else 0.25) * self.mod
            
            # 更新 DC 分量系数
            if mag > 1e-6:
                dft[0, 0, 0] *= (target_mag / mag)
                dft[0, 0, 1] *= (target_mag / mag)
            else:
                dft[0, 0, 0], dft[0, 0, 1] = target_mag, 0
            
            # IDFT 还原块
            new_ha_Y[r*bh:(r+1)*bh, c*bw:(c+1)*bw] = cv2.idft(dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        # 6. 逆小波变换还原图像 (IDWT)
        res_y = self._idwt_channel(new_ha_Y, coeffs_Y)
        res_u = self._idwt_channel(ha_U, coeffs_U)
        res_v = self._idwt_channel(ha_V, coeffs_V)
        
        # 合并通道并裁剪回原尺寸
        res_yuv = np.zeros((*res_y.shape, 3), dtype=np.float32)
        res_yuv[:,:,0], res_yuv[:,:,1], res_yuv[:,:,2] = res_y, res_u, res_v
        res_yuv = res_yuv[:ori_h, :ori_w]
        
        # 转回 BGR 并限制像素范围
        res_bgr = cv2.cvtColor(res_yuv, cv2.COLOR_YUV2BGR)
        res_bgr = np.clip(res_bgr, 0, 255).astype(np.uint8)
        
        # 7. 生成 Metadata (这一步非常重要，供提取端 Decoder 使用)
        # 使用嵌入后的 Y 通道进行亮度分析，得出基准能级
        levels = self.quantizer.fit(res_yuv[:,:,0], ignore_mask=small_mask)
        
        cv_imwrite(output_path, res_bgr)
        
        return {
            "ignore_mask": small_mask,
            "global_levels": levels,
            "wm_shape": wm_shape
        }

# ==========================================
# 提取端：由学生端/校验后端调用
# ==========================================
class WatermarkDecoder(WatermarkBase):
    def __init__(self, seed_wm, seed_dct, mod=4096):
        super().__init__(mod)
        self.seed_wm = seed_wm
        self.seed_dct = seed_dct
        self.quantizer = BrightnessRankQuantizer(block_shape=self.block_shape)

    def decode(self, img_array, metadata):
        """
        metadata: 包含 ignore_mask, global_levels, wm_shape
        """
        img = img_array.astype(np.float32)
        y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
        
        # 1. 屏摄补偿 (针对真实场景)
        y_clean = self.quantizer.apply(y, metadata['global_levels'])
        
        # 2. 小波分解
        ll = self.get_dwt_ll(y_clean)
        bh, bw = self.block_shape
        idx0, idx1 = self.get_block_indices(ll.shape)
        
        # 3. 向量化提取：性能提升的关键
        # 将 LL 层重排为 (N, bh, bw) 的形式，一次性做 batch DFT
        strides = ll.strides[0]*bh, ll.strides[1]*bw, ll.strides[0], ll.strides[1]
        blocks = np.lib.stride_tricks.as_strided(ll, (ll.shape[0]//bh, ll.shape[1]//bw, bh, bw), strides)
        
        # 提取逻辑
        votes = defaultdict(list)
        total_bits = metadata['wm_shape'][0] * metadata['wm_shape'][1]
        
        # 随机种子状态同步
        rnd_dct = np.random.RandomState(self.seed_dct)
        mask = metadata['ignore_mask']
        
        for i in range(len(idx0)):
            rnd_dct.shuffle(np.arange(bh*bw))
            r, c = idx0[i], idx1[i]
            
            # 跳过 Ignore Mask 区域
            if np.mean(mask[r*bh:(r+1)*bh, c*bw:(c+1)*bw]) > 0.5:
                continue
                
            block = ll[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            # 仅做 DC 分量的 DFT 计算 (极其高效)
            dc_mag = np.abs(np.sum(block)) # DFT(0,0) 的幅度等于像素之和
            
            wm_bit = 255 if (dc_mag % self.mod) > (self.mod / 2) else 0
            votes[i % total_bits].append(wm_bit)
            
        # 4. 投票结果汇总
        extracted_flat = np.array([np.median(votes[i]) if i in votes else 0 for i in range(total_bits)])
        
        # 5. 逆混淆随机化
        wm_idx = np.arange(total_bits)
        np.random.RandomState(self.seed_wm).shuffle(wm_idx)
        final_wm = np.zeros(total_bits)
        final_wm[wm_idx] = extracted_flat
        
        return final_wm.reshape(metadata['wm_shape']).astype(np.uint8)
    


if __name__ == "__main__":
    # Windows 环境下使用多进程提取需此行
    import multiprocessing
    multiprocessing.freeze_support()

    # --- 配置区 ---
    ORI_IMAGE = ".\pic\prep_code/fixed_QR-code3.png"   # 原始二维码
    WATERMARK = "../datasets/watermarks/68.png"      # 原水印
    # ENCODED_QR = "./pic\output\out_code\out_code_test.png"      # 生成的水印二维码
    ENCODED_QR = "./pic/prep_code/fixed_test.png"      # 生成的水印二维码(屏摄)

    EXTRACTED_WM = "./pic/output/out_real_wm/out_real_wm_test.png"   # 提取出来的结果

    # --- 步骤 1: 教师端嵌入 (Encoder) ---
    # 这里模拟生成 3 组中的一组，使用特定随机种子
    encoder = WatermarkEncoder(seed_wm=4399, seed_dct=2333, mod=4096)
    
    # encode 方法应返回包含 ignore_mask 和 global_levels 的字典
    print(">>> 正在生成水印二维码...")
    metadata = encoder.encode(ORI_IMAGE, WATERMARK, ENCODED_QR)
    
    # # 在实际系统中，你会把 metadata 存入数据库
    print(f"Metadata 已生成。亮度能级为: {metadata['global_levels']}")

    # --- 步骤 2: 学生端/后端提取 (Decoder) ---
    # 模拟从数据库取出种子和 metadata 进行提取
    decoder = WatermarkDecoder(seed_wm=4399, seed_dct=2333, mod=4096)
    
    print(">>> 正在提取水印...")
    # 模拟真实场景 (is_real=True 会触发并行光照重组)
    # 这里直接用生成的图测试，如果测试拍照图，请更换路径
    result_wm = decoder.decode(ENCODED_QR, metadata)
    
    # 保存提取结果进行查看
    cv_imwrite(EXTRACTED_WM, result_wm)
    print(f">>> 提取完成！请查看: {EXTRACTED_WM}")