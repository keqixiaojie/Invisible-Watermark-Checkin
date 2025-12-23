import numpy as np 
import cv2
from pywt import dwt2,idwt2
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import find_peaks

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),cv2.IMREAD_COLOR)
    return cv_img

def cv_imwrite(path, img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img, 0, 255).astype(np.uint8)
    suffix = os.path.splitext(path)[-1] or '.png'
    if not suffix: path += suffix
    try:
        is_success, buffer = cv2.imencode(suffix, img)
        if is_success: buffer.tofile(path)
    except Exception as e: print(f"保存出错: {e}")

# ==========================================
# 类：基于排序的局部亮度重组器 (支持忽略区域)
# ==========================================
class BrightnessRankQuantizer:
    def __init__(self, n_levels=4, block_shape=(4,4), context_blocks=13):
        if context_blocks % 2 == 0: context_blocks += 1
        self.n_levels = n_levels
        self.block_shape = block_shape
        self.context_blocks = context_blocks 
        self.global_levels = None 

    def fit(self, img, ignore_mask=None):
        """
        Fit: 寻找白色区域的层级，但强制加入 0 作为黑色基准。
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = img.astype(np.uint8)

        # 1. 提取有效像素
        if ignore_mask is not None:
            if ignore_mask.shape != gray.shape:
                mask_resized = cv2.resize(ignore_mask.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                valid_pixels = gray[~mask_resized]
            else:
                valid_pixels = gray[~ignore_mask]
        else:
            valid_pixels = gray.flatten()

        if len(valid_pixels) < 100: valid_pixels = gray.flatten()

        # 2. 计算直方图
        hist, bins = np.histogram(valid_pixels, bins=256, range=[0, 256])
        
        # 3. 寻找白色波峰
        temp_hist = hist.copy()
        temp_hist[:50] = 0 # 忽略暗部
        temp_hist[255] = 0 
        
        peaks, _ = find_peaks(temp_hist, height=np.max(temp_hist)*0.1, distance=20)
        
        candidates = list(peaks)
        candidates.append(255) # 强制加最亮白
        
        sorted_candidates = sorted(list(set(candidates)))
        
        # 取最亮的 n_levels 个作为白色基准
        if len(sorted_candidates) < self.n_levels:
            min_p = sorted_candidates[0] if len(sorted_candidates) > 0 else 128
            white_levels = np.linspace(min_p, 255, self.n_levels)
        else:
            white_levels = np.array(sorted_candidates[-self.n_levels:])
            
        # 【关键修改】最终基准 = [0] + [白色基准]
        # 这样量化器就知道，如果像素很暗，应该归类为 0，而不是归类为最暗的白(例如200)
        self.global_levels = np.concatenate(([0], white_levels)).astype(int)
            
        print(f"【Fit 结果】基准层级 (含黑): {self.global_levels}")

    def apply(self, img, ignore_mask=None):
        if self.global_levels is None: self.fit(img, ignore_mask)
            
        h, w = img.shape[:2]
        bh, bw = self.block_shape
        new_h = h - (h % bh)
        new_w = w - (w % bw)
        
        if len(img.shape) == 3:
            src = img[:new_h, :new_w]
            gray = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_BGR2GRAY) if src.shape[2]==3 else src
        else:
            gray = img[:new_h, :new_w]

        pad_b_count = self.context_blocks // 2 
        pad_h = pad_b_count * bh
        pad_w = pad_b_count * bw
        padded_img = cv2.copyMakeBorder(gray, pad_h, pad_h, pad_w, pad_w, value=10, borderType=cv2.BORDER_CONSTANT)
        
        processed_img = np.zeros((new_h, new_w), dtype=np.float32)
        ctx_pixel_h = self.context_blocks * bh
        ctx_pixel_w = self.context_blocks * bw

        # 你的逻辑：只处理亮度大于阈值的像素
        # 注意：这里的阈值是对 原图(屏摄图) 判断的，可能不够稳健
        # 更好的方式是用 Context Percentile 相对判断，但为了遵从你的逻辑，这里保留硬阈值
        PIXEL_THRESH = 100 

        for r in range(0, new_h, bh):
            for c in range(0, new_w, bw):
                
                # --- 获取上下文 ---
                context_roi = padded_img[r : r + ctx_pixel_h, c : c + ctx_pixel_w]
                center_block = context_roi[pad_h : pad_h + bh, pad_w : pad_w + bw]
                
                # 1. 统计全量信息的 min/max (包含黑白所有像素)
                ctx_min, ctx_max = np.percentile(context_roi, [0, 100])
                
                # 2. 创建更新掩码 (只更新原图中比较亮的点)
                # 这保留了你在"黑白边缘"不想误伤黑色像素的意图
                brightness_mask = center_block >= PIXEL_THRESH
                
                # 默认先复制原图
                quantized_block = gray[r:r+bh, c:c+bw].copy().astype(np.float32)
                
                if not np.any(brightness_mask):
                    processed_img[r:r+bh, c:c+bw] = quantized_block
                    continue
                
                # 3. 归一化与映射
                if (ctx_max - ctx_min) < 10:
                    # 低反差区域：整体吸附
                    mean_val = np.mean(center_block)
                    idx = (np.abs(self.global_levels - mean_val)).argmin()
                    target = self.global_levels[idx]
                    # 只更新亮部
                    quantized_block[brightness_mask] = target
                else:
                    # 高反差区域：拉伸
                    # 【核心修正】：映射到 0~255，而不是 g_min~g_max
                    # 这样 0.0 (最暗) -> 0, 1.0 (最亮) -> 255
                    
                    # 归一化 0.0 ~ 1.0
                    norm_block = (center_block - ctx_min) / (ctx_max - ctx_min + 1e-5)
                    norm_block = np.clip(norm_block, 0.0, 1.0)
                    
                    # 映射到全动态范围 0 ~ 255
                    mapped_block = norm_block * 255.0
                    
                    # 逐像素量化
                    for i in range(bh):
                        for j in range(bw):
                            if brightness_mask[i, j]:
                                val = mapped_block[i, j]
                                # 找最近的层级 (现在 global_levels 包含了 0 和 白阶)
                                # 比如 val=30 -> 离0最近 -> 变成0
                                # 比如 val=210 -> 离200最近 -> 变成200
                                real_levels = self.global_levels[1:] # 忽略 0
                                idx = (np.abs(real_levels - val)).argmin()+1
                                quantized_block[i, j] = self.global_levels[idx]
                                
                processed_img[r:r+bh, c:c+bw] = quantized_block
                
        return processed_img

# ==========================================
# 主水印类
# ==========================================
class watermark():
    def __init__(self, random_seed_wm, random_seed_dct, mod, mod2=None, wm_shape=None, block_shape=(4,4), color_mod='YUV', dwt_deep=1, alpha=0.9, center_alpha=0.8, is_smooth=True):
        self.block_shape = block_shape 
        self.random_seed_wm = random_seed_wm
        self.random_seed_dct = random_seed_dct
        self.mod = mod
        self.mod2 = mod2
        self.wm_shape = wm_shape
        self.color_mod = color_mod
        self.dwt_deep = dwt_deep
        self.alpha = alpha
        self.center_alpha = center_alpha
        self.is_smooth = is_smooth
        self.smooth_deep = 1
        
        self.rank_quantizer = BrightnessRankQuantizer(n_levels=3, block_shape=block_shape)
        
        # 新增：黑名单 Mask (存储在 DWT LL 层的分辨率下)
        self.ignore_mask = None 

    def init_block_add_index(self, img_shape):
        shape0_int, shape1_int = int(img_shape[0]/self.block_shape[0]), int(img_shape[1]/self.block_shape[1])
        if not shape0_int*shape1_int >= self.wm_shape[0]*self.wm_shape[1]:
            print("水印的大小超过图片的容量")
        self.part_shape = (shape0_int*self.block_shape[0], shape1_int*(self.block_shape[1]))
        self.block_add_index0, self.block_add_index1 = np.meshgrid(np.arange(shape0_int), np.arange(shape1_int))
        self.block_add_index0, self.block_add_index1 = self.block_add_index0.flatten(), self.block_add_index1.flatten()
        self.length = self.block_add_index0.size
        
    def convolution_smooth(self, img, kernel_size=3):
        if img is None: return None
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        smoothed_img = img
        for _ in range(self.smooth_deep):
            smoothed_img = cv2.filter2D(smoothed_img, -1, kernel)
        return smoothed_img
        
    def generate_ignore_mask(self, img):
        """
        根据原图生成忽略掩膜 (黑色区域)
        img: 应该是原始尺寸的 Y 通道图像
        """
        # 简单的阈值分割：像素 < 30 认为是黑色区域
        # 为了更准确，可以使用大津法或自适应阈值
        _, mask = cv2.threshold(img.astype(np.uint8), 100, 255, cv2.THRESH_BINARY_INV)
        
        # 缩小 Mask 到 DWT LL 层大小
        # DWT 会把尺寸缩小 2^deep 倍
        factor = 2 ** self.dwt_deep
        # 使用 INTER_MAX 确保只要区域内有黑，缩小后也倾向于黑 (或者用 Mean)
        # 这里我们用 Mean，如果一个 Block 区域内大部分是黑，则 mask 为 True
        h, w = mask.shape
        small_h = h // factor
        small_w = w // factor
        
        mask_resized = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
        # 归一化为 bool: True 表示忽略 (黑色)
        self.ignore_mask = mask_resized > 127
        
        # 保存一下 Mask 看看对不对
        cv_imwrite("debug_ignore_mask.png", self.ignore_mask.astype(np.uint8)*255)

    def read_ori_img(self, filename):
        ori_img = cv_imread(filename).astype(np.float32)
        if self.is_smooth:
            ori_img = self.convolution_smooth(ori_img)
        self.ori_img_shape = ori_img.shape[:2]
        
        if self.color_mod == 'RGB':
            self.ori_img_YUV = ori_img
        elif self.color_mod == 'YUV':
            self.ori_img_YUV = cv2.cvtColor(ori_img, cv2.COLOR_BGR2YUV)

        # 生成忽略 Mask (基于 Y 通道)
        self.generate_ignore_mask(self.ori_img_YUV[:,:,0])

        # Padding
        if not self.ori_img_YUV.shape[0]%(2**self.dwt_deep)==0:
            temp = (2**self.dwt_deep)-self.ori_img_YUV.shape[0]%(2**self.dwt_deep)
            self.ori_img_YUV = np.concatenate((self.ori_img_YUV,np.zeros((temp,self.ori_img_YUV.shape[1],3))),axis=0)
        if not self.ori_img_YUV.shape[1]%(2**self.dwt_deep)==0:
            temp = (2**self.dwt_deep)-self.ori_img_YUV.shape[1]%(2**self.dwt_deep)
            self.ori_img_YUV = np.concatenate((self.ori_img_YUV,np.zeros((self.ori_img_YUV.shape[0],temp,3))),axis=1)

        # DWT 分解 (简写)
        def get_dwt(data):
            coeffs = dwt2(data, 'haar')
            ll = coeffs[0]
            coeffs_list = [coeffs[1]]
            for i in range(self.dwt_deep-1):
                coeffs = dwt2(ll, 'haar')
                ll = coeffs[0]
                coeffs_list.append(coeffs[1])
            return ll, coeffs_list

        self.ha_Y, self.coeffs_Y = get_dwt(self.ori_img_YUV[:,:,0])
        self.ha_U, self.coeffs_U = get_dwt(self.ori_img_YUV[:,:,1])
        self.ha_V, self.coeffs_V = get_dwt(self.ori_img_YUV[:,:,2])
        
        # 分块
        self.ha_block_shape = (int(self.ha_Y.shape[0]/self.block_shape[0]),int(self.ha_Y.shape[1]/self.block_shape[1]),self.block_shape[0],self.block_shape[1])
        strides = self.ha_Y.itemsize*(np.array([self.ha_Y.shape[1]*self.block_shape[0],self.block_shape[1],self.ha_Y.shape[1],1]))
        
        self.ha_Y_block = np.lib.stride_tricks.as_strided(self.ha_Y.copy(),self.ha_block_shape,strides)
        self.ha_U_block = np.lib.stride_tricks.as_strided(self.ha_U.copy(),self.ha_block_shape,strides)
        self.ha_V_block = np.lib.stride_tricks.as_strided(self.ha_V.copy(),self.ha_block_shape,strides)

    def read_wm(self, filename):
        self.wm = cv_imread(filename)[:,:,0]
        self.wm_shape = self.wm.shape[:2]
        self.init_block_add_index(self.ha_Y.shape)
        self.wm_flatten = self.wm.flatten()
        if self.random_seed_wm:
            self.random_wm = np.random.RandomState(self.random_seed_wm)
            self.random_wm.shuffle(self.wm_flatten)

    def block_add_wm(self, block, index, i):
        # 【关键】检查是否在忽略名单中
        # 获取当前 Block 在 LL 层中的坐标
        r_idx = self.block_add_index0[i]
        c_idx = self.block_add_index1[i]
        
        # 获取对应位置的 Mask
        # self.ignore_mask 已经是 DWT LL 层分辨率
        # 但它是像素级的，我们需要看这个 Block 覆盖的区域
        # 实际上 block_add_index0/1 是 Block 的索引
        # 对应 mask 中的坐标应该是 r_idx * block_h : (r_idx+1)*block_h
        bh, bw = self.block_shape
        r_start = r_idx * bh
        c_start = c_idx * bw
        
        # 注意边界检查 (Padding后的尺寸可能比 mask 大一点点，mask 需要 padding 或 resize)
        # 这里为了稳健，如果越界就默认为 False
        if r_start < self.ignore_mask.shape[0] and c_start < self.ignore_mask.shape[1]:
            local_mask = self.ignore_mask[r_start:r_start+bh, c_start:c_start+bw]
            # 如果 Block 内黑色像素占比超过一半，则跳过
            if np.mean(local_mask) > 0.5:
                return block # 原样返回，不嵌入
        
        # 1. 安全预处理 (Shift to Center)
        target_center = 128 + self.center_alpha * (block.mean() - 128)
        block = (block - target_center) * self.alpha + target_center
        
        # 2. 准备嵌入数据
        i = i % (self.wm_shape[0] * self.wm_shape[1])
        wm_1 = self.wm_flatten[i]

        # 3. DFT 变换
        dft = cv2.dft(block.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
        current_mag = mag[0, 0]
        
        # 5. 量化调制
        target_mag = (current_mag - current_mag % self.mod + 3/4 * self.mod) if wm_1 >= 128 else (current_mag - current_mag % self.mod + 1/4 * self.mod)
        
        if current_mag > 1e-6:
            scale = target_mag / current_mag
            dft[0, 0, 0] *= scale
            dft[0, 0, 1] *= scale
        else:
            dft[0, 0, 0] = target_mag
            dft[0, 0, 1] = 0

        block_back = cv2.idft(dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        return block_back

    def embed(self, filename):
        embed_ha_Y_block=self.ha_Y_block.copy()
        
        self.random_dct = np.random.RandomState(self.random_seed_dct)
        index = np.arange(self.block_shape[0]*self.block_shape[1])

        for i in range(self.length):
            self.random_dct.shuffle(index)
            embed_ha_Y_block[self.block_add_index0[i],self.block_add_index1[i]] = self.block_add_wm(embed_ha_Y_block[self.block_add_index0[i],self.block_add_index1[i]],index,i)

        embed_ha_Y_part = np.concatenate(embed_ha_Y_block,1)
        embed_ha_Y_part = np.concatenate(embed_ha_Y_part,1)
        embed_ha_Y = self.ha_Y.copy()
        embed_ha_Y[:self.part_shape[0],:self.part_shape[1]] = embed_ha_Y_part

        # IDWT
        def idwt_channel(ha, coeffs_list):
            res = ha
            for i in range(len(coeffs_list)):
                res = idwt2((res, coeffs_list[-(i+1)]), 'haar')
            return res

        embed_img_Y = idwt_channel(embed_ha_Y, self.coeffs_Y)
        temp_U = idwt_channel(self.ha_U, self.coeffs_U)
        temp_V = idwt_channel(self.ha_V, self.coeffs_V)
        
        embed_img_YUV = np.zeros((embed_img_Y.shape[0], embed_img_Y.shape[1], 3), dtype=np.float32)
        embed_img_YUV[:,:,0] = embed_img_Y
        embed_img_YUV[:,:,1] = temp_U
        embed_img_YUV[:,:,2] = temp_V

        embed_img_YUV=embed_img_YUV[:self.ori_img_shape[0],:self.ori_img_shape[1]]
        
        if self.color_mod == 'RGB':
            embed_img = embed_img_YUV
        elif self.color_mod == 'YUV':
            embed_img = cv2.cvtColor(embed_img_YUV,cv2.COLOR_YUV2BGR)

        embed_img[embed_img>255]=255
        embed_img[embed_img<0]=0

        # Fit 量化器 (只对有效白色区域)
        # 注意：这里我们传入 self.ignore_mask，这要求 mask 尺寸与 embed_img 匹配
        # 但 self.ignore_mask 是 DWT LL 尺寸。
        # 所以 Fit 时最好重新生成原图尺寸的 mask，或者在 fit 内部处理缩放
        # 为了简单，我们在 fit 内部会检查，如果需要 resize 会自动做
        # 传入原始尺寸的 Y 通道给 fit
        y_channel = embed_img_YUV[:,:,0]
        # 需要把 DWT 尺寸的 mask 放大回去
        mask_full = cv2.resize(self.ignore_mask.astype(np.uint8), (y_channel.shape[1], y_channel.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        self.rank_quantizer.fit(y_channel, ignore_mask=mask_full)
        
        cv_imwrite(filename, embed_img)

    def block_get_wm(self, block, index):
        dft = cv2.dft(block.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
        max_s = mag[0, 0]
        wm_1 = 255 if max_s % self.mod > self.mod / 2 else 0
        return wm_1

    def extract(self, filename, out_wm_name, debug_path="rank_reconstruct_debug.png", is_real=False):
        if not self.wm_shape:
            print("水印的形状未设定")
            return 0
        
        embed_img = cv_imread(filename).astype(np.float32)
        if self.color_mod == 'RGB':
            embed_img_YUV = embed_img
        elif self.color_mod == 'YUV':
            embed_img_YUV = cv2.cvtColor(embed_img, cv2.COLOR_BGR2YUV)

        # Padding
        if not embed_img_YUV.shape[0]%(2**self.dwt_deep)==0:
            temp = (2**self.dwt_deep)-embed_img_YUV.shape[0]%(2**self.dwt_deep)
            embed_img_YUV = np.concatenate((embed_img_YUV,np.zeros((temp,embed_img_YUV.shape[1],3))),axis=0)
        if not embed_img_YUV.shape[1]%(2**self.dwt_deep)==0:
            temp = (2**self.dwt_deep)-embed_img_YUV.shape[1]%(2**self.dwt_deep)
            embed_img_YUV = np.concatenate((embed_img_YUV,np.zeros((embed_img_YUV.shape[0],temp,3))),axis=1)
        
        embed_img_Y = embed_img_YUV[:,:,0].astype(np.float32)
        
        # 重新生成或加载 Mask
        # 注意：如果是盲提取，我们应该从当前的 embed_img 重新计算 Mask
        # 因为屏摄后位置可能微变，或者黑度变化
        # 这里为了保持一致性，我们再次调用 generate_ignore_mask
        self.generate_ignore_mask(embed_img_Y)
        
        if is_real:
            print(">>> 正在进行 Block 级光照重构与像素重组...")
            # 传入 Mask，让 quantizer 知道哪些地方忽略
            # 注意 Mask 需要放大
            mask_full = cv2.resize(self.ignore_mask.astype(np.uint8), (embed_img_Y.shape[1], embed_img_Y.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            clean_Y = self.rank_quantizer.apply(embed_img_Y, ignore_mask=mask_full)
            cv_imwrite(debug_path, clean_Y)
            embed_img_Y = clean_Y
        
        # DWT 分解
        coeffs_Y = dwt2(embed_img_Y,'haar')
        ha_Y = coeffs_Y[0]
        for i in range(self.dwt_deep-1):
            coeffs_Y = dwt2(ha_Y,'haar')
            ha_Y = coeffs_Y[0]

        self.init_block_add_index(ha_Y.shape)
        
        ha_block_shape = (int(ha_Y.shape[0]/self.block_shape[0]),int(ha_Y.shape[1]/self.block_shape[1]),self.block_shape[0],self.block_shape[1])
        strides = ha_Y.itemsize*(np.array([ha_Y.shape[1]*self.block_shape[0],self.block_shape[1],ha_Y.shape[1],1]))
        ha_Y_block = np.lib.stride_tricks.as_strided(ha_Y.copy(),ha_block_shape,strides)

        self.wm_votes = defaultdict(list)
        self.random_dct = np.random.RandomState(self.random_seed_dct)
        index = np.arange(self.block_shape[0]*self.block_shape[1])
        total_wm_bits = self.wm_shape[0] * self.wm_shape[1]

        bh, bw = self.block_shape

        for i in range(self.length):
            self.random_dct.shuffle(index)
            
            # 【关键】提取时检查 Mask
            r_idx = self.block_add_index0[i]
            c_idx = self.block_add_index1[i]
            r_start = r_idx * bh
            c_start = c_idx * bw
            
            is_ignored = False
            if r_start < self.ignore_mask.shape[0] and c_start < self.ignore_mask.shape[1]:
                local_mask = self.ignore_mask[r_start:r_start+bh, c_start:c_start+bw]
                if np.mean(local_mask) > 0.5:
                    is_ignored = True
            
            if is_ignored:
                # 忽略该投票，或者投弃权票 (不加入 list)
                continue

            wm_Y = self.block_get_wm(ha_Y_block[r_idx, c_idx], index)
            wm_idx = i % total_wm_bits
            self.wm_votes[wm_idx].append(wm_Y)

        extract_wm = np.zeros(total_wm_bits)
        for i in range(total_wm_bits):
            if i in self.wm_votes and len(self.wm_votes[i]) > 0:
                extract_wm[i] = np.median(self.wm_votes[i])
            else:
                extract_wm[i] = 0 # 默认

        wm_index = np.arange(extract_wm.size)
        self.random_wm = np.random.RandomState(self.random_seed_wm)
        self.random_wm.shuffle(wm_index)
        
        final_wm = np.zeros_like(extract_wm)
        final_wm[wm_index] = extract_wm
        final_wm[final_wm >= 127] = 255
        final_wm[final_wm < 127] = 0

        cv_imwrite(out_wm_name, final_wm.reshape(self.wm_shape[0],self.wm_shape[1]))

    def diagnose_detailed(self, embed_img_path, ori_img_path, wm_path, save_vis_path="error_vis.png", is_real=False):
        print("="*30 + " DFT 深度诊断报告 " + "="*30)
        
        # 1. 准备可视化画布 (原图)
        vis_img = cv_imread(ori_img_path)
        scale_factor = 2 ** self.dwt_deep
        
        # 2. 准备标准水印
        wm_ori = cv_imread(wm_path)[:,:,0]
        wm_flatten = wm_ori.flatten()
        if self.random_seed_wm:
            np.random.RandomState(self.random_seed_wm).shuffle(wm_flatten)

        # 辅助读取函数 (复用逻辑)
        def get_ll_blocks(path, is_embed_img=False):
            img = cv_imread(path).astype(np.float32)
            if self.color_mod == 'YUV': img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            
            # Padding
            if not img.shape[0]%(2**self.dwt_deep)==0:
                temp = (2**self.dwt_deep)-img.shape[0]%(2**self.dwt_deep)
                img = np.concatenate((img,np.zeros((temp,img.shape[1],3))),axis=0)
            if not img.shape[1]%(2**self.dwt_deep)==0:
                temp = (2**self.dwt_deep)-img.shape[1]%(2**self.dwt_deep)
                img = np.concatenate((img,np.zeros((img.shape[0],temp,3))),axis=1)
                
            img_Y = img[:,:,0].astype(np.float32)
            
            # 如果是嵌入后的图(或屏摄图)，且开启了真实场景模式，应用重组
            if is_embed_img and is_real:
                # 重新生成mask以防万一
                self.generate_ignore_mask(img_Y)
                mask_full = cv2.resize(self.ignore_mask.astype(np.uint8), (img_Y.shape[1], img_Y.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                clean_Y = self.rank_quantizer.apply(img_Y, ignore_mask=mask_full)
                img_Y = clean_Y
            
            # DWT
            ll = dwt2(img_Y, 'haar')[0] 
            for _ in range(self.dwt_deep - 1): ll = dwt2(ll, 'haar')[0]
            
            # 分块
            h_bl, w_bl = ll.shape[0] // self.block_shape[0], ll.shape[1] // self.block_shape[1]
            strides = ll.itemsize * np.array([ll.shape[1]*self.block_shape[0], self.block_shape[1], ll.shape[1], 1])
            blocks = np.lib.stride_tricks.as_strided(ll.copy(), (h_bl, w_bl, self.block_shape[0], self.block_shape[1]), strides)
            return blocks, ll.shape

        try:
            # 原图 blocks (用于对比基准)
            blocks_ori, ll_shape = get_ll_blocks(ori_img_path, is_embed_img=False)
            # 嵌入图 blocks (可能是屏摄图)
            blocks_embed, _ = get_ll_blocks(embed_img_path, is_embed_img=True)
        except Exception as e:
            print(f"诊断出错: {e}")
            import traceback
            traceback.print_exc()
            return

        self.init_block_add_index(ll_shape)
        
        # 确保 Mask 存在 (使用原图生成最准)
        if self.ignore_mask is None:
            temp_ori = cv_imread(ori_img_path).astype(np.float32)
            if self.color_mod == 'YUV': temp_ori = cv2.cvtColor(temp_ori, cv2.COLOR_BGR2YUV)
            self.generate_ignore_mask(temp_ori[:,:,0])

        header = f"{'ID':<6} | {'Exp':<3} {'Get':<3} | {'Mag_Ori':<8} {'Mag_New':<8} {'Diff':<6} | {'Ignored':<7} | {'Reason'}"
        print(header); print("-" * 100)

        error_count = 0
        valid_blocks_count = 0
        total_blocks = self.length
        bh, bw = self.block_shape
        
        for i in range(total_blocks):
            # 获取 Block 坐标
            r_idx = self.block_add_index0[i]
            c_idx = self.block_add_index1[i]
            r_start = r_idx * bh
            c_start = c_idx * bw
            
            # 检查忽略 Mask
            is_ignored = False
            if r_start < self.ignore_mask.shape[0] and c_start < self.ignore_mask.shape[1]:
                local_mask = self.ignore_mask[r_start:r_start+bh, c_start:c_start+bw]
                if np.mean(local_mask) > 0.5:
                    is_ignored = True
            
            # 如果被忽略，跳过详细计算，或者打印一条忽略信息
            if is_ignored:
                # 可选：绘制灰色框表示忽略区域
                y = int(r_idx * bh * scale_factor)
                x = int(c_idx * bw * scale_factor)
                h = int(bh * scale_factor)
                w = int(bw * scale_factor)
                # cv2.rectangle(vis_img, (x, y), (x + w, y + h), (128, 128, 128), 1) 
                continue

            valid_blocks_count += 1
            
            # 获取数据
            b_ori = blocks_ori[r_idx, c_idx]
            b_embed = blocks_embed[r_idx, c_idx]
            
            # 为了计算 Mag_Ori 准确，也需要对原图做预处理模拟
            # 但为了看真实的物理值，这里直接算
            dft_ori = cv2.dft(b_ori.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
            mag_ori = cv2.magnitude(dft_ori[:,:,0], dft_ori[:,:,1])[0,0]

            dft_embed = cv2.dft(b_embed.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
            mag_new = cv2.magnitude(dft_embed[:,:,0], dft_embed[:,:,1])[0,0]

            expected = wm_flatten[i % wm_flatten.size]
            mod_res = mag_new % self.mod
            extracted = 255 if mod_res > self.mod / 2 else 0
            
            is_error = (expected >= 128 and extracted == 0) or (expected < 128 and extracted == 255)
            
            if is_error:
                error_count += 1
                diff_val = mag_new - mag_ori
                
                # 绘制红框
                y = int(r_idx * bh * scale_factor)
                x = int(c_idx * bw * scale_factor)
                h = int(bh * scale_factor)
                w = int(bw * scale_factor)
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                print(f"{i:<6} | {1 if expected>128 else 0:<3} {1 if extracted>128 else 0:<3} | {mag_ori:<8.1f} {mag_new:<8.1f} {diff_val:<6.1f} | {'No':<7} | Error")

        print("-" * 100)
        if valid_blocks_count > 0:
            print(f"有效块数: {valid_blocks_count} / {total_blocks}")
            print(f"有效区域错误率: {error_count/valid_blocks_count:.2%}")
        else:
            print("警告：没有检测到有效块（全黑？）")
            
        cv_imwrite(save_vis_path, vis_img)
        
if __name__=="__main__":
    ori_path = "pic/prep_code"
    out_path = "pic/output"
    wm_path = "pic/watermark"

    bwm1 = watermark(4399,2333,2048)
    bwm1.read_ori_img(ori_path + "/fixed_Wechat2.png")
    bwm1.read_wm(wm_path + "/wm.png")
    bwm1.embed(out_path + "/out.png")
    bwm1.extract(out_path + "/out.png", out_path + "/out_wm.png")
    bwm1.extract(out_path + "/fixed_reality.png", out_path + "/real_wm.png", is_real=True)
    # bwm1.diagnose_detailed(out_path + "/out.png", ori_path + "/fixed_Wechat2.png", wm_path + "/wm.png")
    bwm1.diagnose_detailed(out_path + "/fixed_reality.png", ori_path + "/fixed_Wechat2.png", wm_path + "/wm.png", is_real=True)