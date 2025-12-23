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

# def cv_imwrite(path,img):
#     suffix = os.path.splitext(path)[-1]
#     cv2.imencode(suffix, img)[1].tofile(path)
def cv_imwrite(path, img):
    # 检查输入是否为浮点数
    if img.dtype == np.float32 or img.dtype == np.float64:
        # 1. 限制范围在 0-255 之间 (防止溢出，比如 -5 变成 251)
        img = np.clip(img, 0, 255)
        # 2. 安全转换为无符号8位整数
        img = img.astype(np.uint8)
        
    suffix = os.path.splitext(path)[-1]
    # 如果没有后缀名，默认给个 .png，防止报错
    if not suffix:
        suffix = '.png'
        path = path + suffix
        
    try:
        # cv2.imencode 返回 (retval, buffer)
        is_success, buffer = cv2.imencode(suffix, img)
        if is_success:
            buffer.tofile(path)
        else:
            print(f"保存失败: {path}，可能是编码格式不支持")
    except Exception as e:
        print(f"保存出错: {e}")

# ==========================================
# 新增类：基于排序的局部亮度重组器
# ==========================================
class BrightnessRankQuantizer:
    def __init__(self, n_levels=4, block_shape=(4,4)):
        self.n_levels = n_levels
        self.block_shape = block_shape
        self.global_levels = None # 存储全图的5个基准亮度

    def fit(self, img):
        """
        第一步：分析全图，找到即使经过拍照破坏，依然存在的5个'山头'
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = img.astype(np.uint8)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        
        # 寻找波峰
        peaks, _ = find_peaks(hist, height=np.max(hist)*0.05, distance=20)
        
        # 强制包含“最黑”和“最白”的区域中心作为锚点
        # 我们不能简单用0和255，因为拍照后黑可能变40，白可能变200
        # 所以我们取直方图有值的最小端和最大端
        non_zeros = np.where(hist > 0)[0]
        if len(non_zeros) > 0:
            min_val = non_zeros[0] + 10 # 稍微往里一点
            max_val = non_zeros[-1] - 10
        else:
            min_val, max_val = 0, 255

        # 收集候选点
        candidates = set(peaks)
        candidates.add(min_val)
        candidates.add(max_val)
        
        # 筛选出最显著的5个层级
        # 如果不够5个，就用线性插值补齐
        sorted_candidates = sorted(list(candidates))
        
        if len(sorted_candidates) > self.n_levels:
            # 简单的均匀采样，或者取直方图最高的
            # 这里为了稳健，直接均匀取样
            indices = np.linspace(0, len(sorted_candidates)-1, self.n_levels).astype(int)
            self.global_levels = np.array([sorted_candidates[i] for i in indices])
        elif len(sorted_candidates) < self.n_levels:
            # 补齐
            self.global_levels = np.linspace(min_val, max_val, self.n_levels)
        else:
            self.global_levels = np.array(sorted_candidates)
            
        print(f"【全局基准】检测到的光照环境层级: {self.global_levels.astype(int)}")

    def process_block(self, block):
        """
        核心算法：Block内部排序重组
        """
        # 1. 扁平化并记录原始索引
        flat = block.flatten()
        original_indices = np.argsort(flat) # 记住原来的位置
        sorted_vals = flat[original_indices] # 排序后的像素值
        
        # 2. 聚类：在Block内部找阶梯
        # 简单的一维聚类：找数值跳变最大的地方
        # 但考虑到只有16个点，直接用K-Means或者简单距离归类
        # 为了极速，我们用“映射到最近的Global Level”但保留相对顺序的策略
        
        # 策略升级：
        # 计算该Block的动态范围
        b_min, b_max = sorted_vals[0], sorted_vals[-1]
        
        # 如果Block是纯色（反差极小），直接整体映射到最近的一个Global Level
        if (b_max - b_min) < 30: 
            # 找整体均值离哪个Global最近
            mean_val = np.mean(sorted_vals)
            idx = (np.abs(self.global_levels - mean_val)).argmin()
            target_val = self.global_levels[idx]
            return np.full(self.block_shape, target_val, dtype=np.float32)

        # 如果是有内容的Block：
        # 我们将 block 内部的像素值线性拉伸(Normalize)到 Global 的范围，然后做最近邻
        # 这就是“局部对比度拉伸”
        
        # 1. 归一化到 0-1
        norm_vals = (flat - b_min) / (b_max - b_min + 1e-5)
        
        # 2. 映射到 Global 的最大最小范围
        # Global中最暗的级是 global_levels[0]，最亮是 global_levels[-1]
        g_min, g_max = self.global_levels[0], self.global_levels[-1]
        
        mapped_vals = norm_vals * (g_max - g_min) + g_min
        
        # 3. 最后一步：吸附到具体的5个层级上 (Quantize)
        quantized_vals = np.zeros_like(mapped_vals)
        for i in range(len(mapped_vals)):
            idx = (np.abs(self.global_levels - mapped_vals[i])).argmin()
            quantized_vals[i] = self.global_levels[idx]
            
        return quantized_vals.reshape(self.block_shape)

    def apply(self, img):
        """
        遍历图像块进行处理
        """
        if self.global_levels is None:
            self.fit(img)
            
        h, w = img.shape[:2]
        # 确保尺寸是块的倍数
        new_h = h - (h % self.block_shape[0])
        new_w = w - (w % self.block_shape[1])
        
        processed_img = np.zeros((new_h, new_w), dtype=np.float32)
        
        if len(img.shape) == 3:
            # 如果是彩色，先转灰度处理，或者只处理Y通道
            # 这里为了简单，假设输入已经是单通道(Y)或者只处理第一通道
            src = img[:new_h, :new_w]
            if src.shape[2] == 3: # 如果是BGR
                # 警告：这里只处理单通道逻辑，如果是彩色需要外部拆分
                gray = cv2.cvtColor(src.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                gray = src
        else:
            gray = img[:new_h, :new_w]

        # 遍历所有Block
        bh, bw = self.block_shape
        for r in range(0, new_h, bh):
            for c in range(0, new_w, bw):
                block = gray[r:r+bh, c:c+bw]
                processed_img[r:r+bh, c:c+bw] = self.process_block(block)
                
        return processed_img


# ==========================================
# 主水印类
# ==========================================

class watermark():
    def __init__(self,random_seed_wm,random_seed_dct,mod,mod2=None,wm_shape=None,block_shape=(4,4),color_mod = 'YUV',dwt_deep=1,alpha = 0.9,center_alpha = 0.8,is_smooth=True):
        # self.wm_per_block = 1
        self.block_shape = block_shape  #2^n
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
        # 实例化新的排序重组器
        self.rank_quantizer = BrightnessRankQuantizer(n_levels=5, block_shape=block_shape)


    def init_block_add_index(self,img_shape):
        #假设原图长宽均为2的整数倍,同时假设水印为64*64,则32*32*4
        #分块并DCT
        shape0_int,shape1_int = int(img_shape[0]/self.block_shape[0]),int(img_shape[1]/self.block_shape[1])
        if not shape0_int*shape1_int>=self.wm_shape[0]*self.wm_shape[1]:
            print("水印的大小超过图片的容量")
        self.part_shape = (shape0_int*self.block_shape[0],shape1_int*(self.block_shape[1]))
        self.block_add_index0,self.block_add_index1 = np.meshgrid(np.arange(shape0_int),np.arange(shape1_int))
        self.block_add_index0,self.block_add_index1 = self.block_add_index0.flatten(),self.block_add_index1.flatten()
        self.length = self.block_add_index0.size
        #验证没有意义,但是我不验证不舒服斯基
        assert self.block_add_index0.size==self.block_add_index1.size
        
    def convolution_smooth(self, img, kernel_size=3):
        """
        对图像进行卷积平滑处理，保持原图大小不变。
        
        Args:
            image_path: 原图路径
            output_path: 保存路径
            kernel_size: 卷积核大小 (必须是奇数，如 3, 5)。3 为轻微平滑，5 为中度平滑。
        """
        
        # 1. 读取图像
        # 注意：cv2.imread 默认为 BGR 格式
        
        if img is None:
            print(f"错误：找不到文件")
            return

        # 2. 定义卷积核 (Kernel)
        # 这里使用均值滤波器：创建一个 3x3 的矩阵，所有值均为 1，然后除以 9 (归一化)
        # 数学含义：当前像素值 = 周围 3x3 区域像素的平均值
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        print(f"正在使用 {kernel_size}x{kernel_size} 卷积核进行处理...")
        print(f"卷积核内容:\n{kernel}")

        # 3. 进行卷积操作 (cv2.filter2D)
        # src: 原图
        # ddepth: -1 表示输出图像深度与原图相同 (保持 uint8)
        # kernel: 我们定义的卷积核
        # 这一步会自动处理边缘填充，因此输出大小 = 输入大小
        smoothed_img = img
        for _ in range(self.smooth_deep):
            smoothed_img = cv2.filter2D(smoothed_img, -1, kernel)

        # 4. 保存结果
        return smoothed_img
        

    def read_ori_img(self,filename):
        #傻逼opencv因为数组类型不会变,输入是uint8输出也是uint8,而UV可以是负数且uint8会去掉小数部分
        ori_img = cv_imread(filename).astype(np.float32)
        if self.is_smooth:
            ori_img = self.convolution_smooth(ori_img)
        self.ori_img_shape = ori_img.shape[:2]
        if self.color_mod == 'RGB':
            self.ori_img_YUV = ori_img
        elif self.color_mod == 'YUV':
            self.ori_img_YUV = cv2.cvtColor(ori_img, cv2.COLOR_BGR2YUV)

        if not self.ori_img_YUV.shape[0]%(2**self.dwt_deep)==0:
            temp = (2**self.dwt_deep)-self.ori_img_YUV.shape[0]%(2**self.dwt_deep)
            self.ori_img_YUV = np.concatenate((self.ori_img_YUV,np.zeros((temp,self.ori_img_YUV.shape[1],3))),axis=0)
        if not self.ori_img_YUV.shape[1]%(2**self.dwt_deep)==0:
            temp = (2**self.dwt_deep)-self.ori_img_YUV.shape[1]%(2**self.dwt_deep)
            self.ori_img_YUV = np.concatenate((self.ori_img_YUV,np.zeros((self.ori_img_YUV.shape[0],temp,3))),axis=1)
        assert self.ori_img_YUV.shape[0]%(2**self.dwt_deep)==0
        assert self.ori_img_YUV.shape[1]%(2**self.dwt_deep)==0

        if self.dwt_deep==1:
            coeffs_Y = dwt2(self.ori_img_YUV[:,:,0],'haar')
            ha_Y = coeffs_Y[0]
            coeffs_U = dwt2(self.ori_img_YUV[:,:,1],'haar')
            ha_U = coeffs_U[0]
            coeffs_V = dwt2(self.ori_img_YUV[:,:,2],'haar')
            ha_V = coeffs_V[0]
            self.coeffs_Y = [coeffs_Y[1]]
            self.coeffs_U = [coeffs_U[1]]
            self.coeffs_V = [coeffs_V[1]]

        elif self.dwt_deep>=2:
            #不希望使用太多级的dwt,2,3次就行了
            coeffs_Y = dwt2(self.ori_img_YUV[:,:,0],'haar')
            ha_Y = coeffs_Y[0]
            coeffs_U = dwt2(self.ori_img_YUV[:,:,1],'haar')
            ha_U = coeffs_U[0]
            coeffs_V = dwt2(self.ori_img_YUV[:,:,2],'haar')
            ha_V = coeffs_V[0]
            self.coeffs_Y = [coeffs_Y[1]]
            self.coeffs_U = [coeffs_U[1]]
            self.coeffs_V = [coeffs_V[1]]
            for i in range(self.dwt_deep-1):
                coeffs_Y = dwt2(ha_Y,'haar')
                ha_Y = coeffs_Y[0]
                coeffs_U = dwt2(ha_U,'haar')
                ha_U = coeffs_U[0]
                coeffs_V = dwt2(ha_V,'haar')
                ha_V = coeffs_V[0]
                self.coeffs_Y.append(coeffs_Y[1])
                self.coeffs_U.append(coeffs_U[1])
                self.coeffs_V.append(coeffs_V[1])
        self.ha_Y = ha_Y
        self.ha_U = ha_U
        self.ha_V = ha_V
        
        # 1. 合并三个通道 (H, W) -> (H, W, 3)
        ll_merged = np.dstack((self.ha_Y, self.ha_U, self.ha_V))

        # 2. 处理数值范围 (DWT后数值可能是float，需要clip回0-255并转uint8以供显示)
        # 注意：如果不clip，浮点数可能会导致显示全白或全黑
        ll_vis = np.clip(ll_merged, 0, 255).astype(np.uint8)

        # 3. 如果是YUV模式，为了人眼看着舒服，转回RGB显示
        if self.color_mod == 'YUV':
            # OpenCV的YUV转RGB
            ll_vis = cv2.cvtColor(ll_vis, cv2.COLOR_YUV2RGB)
        elif self.color_mod == 'RGB':
            # Matplotlib默认接受RGB，但要注意OpenCV读入通常是BGR，
            # 如果你前面cv_imread没转RGB，这里可能需要 cv2.cvtColor(ll_vis, cv2.COLOR_BGR2RGB)
            # 假设你目前的self.ori_img_YUV已经是RGB顺序：
            pass 

        plt.figure(figsize=(6, 6))
        plt.title(f'Level {self.dwt_deep} DWT LL Approximation')
        plt.imshow(ll_vis)
        plt.axis('off') # 不显示坐标轴
        plt.show()
        # ================== 可视化部分 End ==================

        self.ha_block_shape = (int(self.ha_Y.shape[0]/self.block_shape[0]),int(self.ha_Y.shape[1]/self.block_shape[1]),self.block_shape[0],self.block_shape[1])
        strides = self.ha_Y.itemsize*(np.array([self.ha_Y.shape[1]*self.block_shape[0],self.block_shape[1],self.ha_Y.shape[1],1]))
        
        self.ha_Y_block = np.lib.stride_tricks.as_strided(self.ha_Y.copy(),self.ha_block_shape,strides)
        self.ha_U_block = np.lib.stride_tricks.as_strided(self.ha_U.copy(),self.ha_block_shape,strides)
        self.ha_V_block = np.lib.stride_tricks.as_strided(self.ha_V.copy(),self.ha_block_shape,strides)

        

    def read_wm(self,filename):
        self.wm = cv_imread(filename)[:,:,0]
        self.wm_shape = self.wm.shape[:2]

        #初始化块索引数组,因为需要验证块是否足够存储水印信息,所以才放在这儿
        self.init_block_add_index(self.ha_Y.shape)

        self.wm_flatten = self.wm.flatten()
        if self.random_seed_wm:
            self.random_wm = np.random.RandomState(self.random_seed_wm)
            self.random_wm.shuffle(self.wm_flatten)


    def block_add_wm(self,block,index,i):
        # 安全预处理
        target_center = 128 + self.center_alpha * (block.mean() - 128)
        block = (block - target_center) * self.alpha + target_center
        
        flag = 0
        if i == 1197:
            flag = 1
        i = i%(self.wm_shape[0]*self.wm_shape[1])

        wm_1 = self.wm_flatten[i]
        block_dct = cv2.dct(block)
        # print("dct系数：", block_dct)
        block_dct_flatten = block_dct.flatten().copy()
        
        block_dct_flatten = block_dct_flatten[index]
        block_dct_shuffled = block_dct_flatten.reshape(self.block_shape)
        U,s,V = np.linalg.svd(block_dct_shuffled)
        max_s = s[0]
        s[0] = (max_s-max_s%self.mod+3/4*self.mod) if wm_1>=128 else (max_s-max_s%self.mod+1/4*self.mod)
        
        if flag:
            print("最大奇异值：", max_s,"更新后的奇异值值：",s[0])
            
        if self.mod2:
            max_s = s[1]
            s[1] = (max_s-max_s%self.mod2+3/4*self.mod2) if wm_1>=128 else (max_s-max_s%self.mod2+1/4*self.mod2)
        # s[1] = (max_s-max_s%self.mod2+3/4*self.mod2) if wm_1<128 else (max_s-max_s%self.mod2+1/4*self.mod2)

        ###np.dot(U[:, :k], np.dot(np.diag(sigma[:k]),v[:k, :]))
        block_dct_shuffled = np.dot(U,np.dot(np.diag(s),V))

        block_dct_flatten = block_dct_shuffled.flatten()
   
        block_dct_flatten[index] = block_dct_flatten.copy()
        block_dct  = block_dct_flatten.reshape(self.block_shape)

        return cv2.idct(block_dct)



    def embed(self,filename):

        embed_ha_Y_block=self.ha_Y_block.copy()
        embed_ha_U_block=self.ha_U_block.copy()
        embed_ha_V_block=self.ha_V_block.copy()

        self.random_dct = np.random.RandomState(self.random_seed_dct)
        index = np.arange(self.block_shape[0]*self.block_shape[1])

        for i in range(self.length):

            self.random_dct.shuffle(index)
            # if i==0:
            #     print("Y:")
            embed_ha_Y_block[self.block_add_index0[i],self.block_add_index1[i]] = self.block_add_wm(embed_ha_Y_block[self.block_add_index0[i],self.block_add_index1[i]],index,i)
            # embed_ha_U_block[self.block_add_index0[i],self.block_add_index1[i]] = self.block_add_wm(embed_ha_U_block[self.block_add_index0[i],self.block_add_index1[i]],index,i)
            # embed_ha_V_block[self.block_add_index0[i],self.block_add_index1[i]] = self.block_add_wm(embed_ha_V_block[self.block_add_index0[i],self.block_add_index1[i]],index,i)

        
        
        embed_ha_Y_part = np.concatenate(embed_ha_Y_block,1)
        embed_ha_Y_part = np.concatenate(embed_ha_Y_part,1)
        # embed_ha_U_part = np.concatenate(embed_ha_U_block,1)
        # embed_ha_U_part = np.concatenate(embed_ha_U_part,1)
        # embed_ha_V_part = np.concatenate(embed_ha_V_block,1)
        # embed_ha_V_part = np.concatenate(embed_ha_V_part,1)

        embed_ha_Y = self.ha_Y.copy()
        embed_ha_Y[:self.part_shape[0],:self.part_shape[1]] = embed_ha_Y_part
        # embed_ha_U = self.ha_U.copy()
        # embed_ha_U[:self.part_shape[0],:self.part_shape[1]] = embed_ha_U_part
        # embed_ha_V = self.ha_V.copy()
        # embed_ha_V[:self.part_shape[0],:self.part_shape[1]] = embed_ha_V_part


        for i in range(self.dwt_deep):
            (cH, cV, cD) = self.coeffs_Y[-1*(i+1)]
            embed_ha_Y = idwt2((embed_ha_Y.copy(), (cH, cV, cD)),"haar") #其idwt得到父级的ha
            # (cH, cV, cD) = self.coeffs_U[-1*(i+1)]
            # embed_ha_U = idwt2((embed_ha_U.copy(), (cH, cV, cD)),"haar") #其idwt得到父级的ha
            # (cH, cV, cD) = self.coeffs_V[-1*(i+1)]
            # embed_ha_V = idwt2((embed_ha_V.copy(), (cH, cV, cD)),"haar") #其idwt得到父级的ha
            #最上级的ha就是嵌入水印的图,即for运行完的ha


        embed_img_YUV = np.zeros(self.ori_img_YUV.shape,dtype=np.float32)
        embed_img_YUV[:,:,0] = embed_ha_Y
        embed_img_YUV[:,:,1] = self.ori_img_YUV[:,:,1]
        embed_img_YUV[:,:,2] = self.ori_img_YUV[:,:,2]
        # embed_img_YUV[:,:,1] = embed_ha_U
        # embed_img_YUV[:,:,2] = embed_ha_V

        embed_img_YUV=embed_img_YUV[:self.ori_img_shape[0],:self.ori_img_shape[1]]
        if self.color_mod == 'RGB':
            embed_img = embed_img_YUV
        elif self.color_mod == 'YUV':
            embed_img = cv2.cvtColor(embed_img_YUV,cv2.COLOR_YUV2BGR)

        embed_img[embed_img>255]=255
        embed_img[embed_img<0]=0
        
        # 1. 【核心步骤】使用排序重组器处理
        # 这一步会把每个 Block 内部的像素重新分配到 5 个理想的层级上
        # 自动消除暗角、反光等光照不均问题
        self.rank_quantizer.fit(embed_img)

        cv_imwrite(filename,embed_img)

    def block_get_wm(self,block,index):
        block_dct = cv2.dct(block)
        block_dct_flatten = block_dct.flatten().copy()
        block_dct_flatten = block_dct_flatten[index]
        block_dct_shuffled = block_dct_flatten.reshape(self.block_shape)

        U,s,V = np.linalg.svd(block_dct_shuffled)
        max_s = s[0]
        wm_1 = 255 if max_s%self.mod >self.mod/2 else 0
        if self.mod2:
            max_s = s[1]
            wm_2 = 255 if max_s%self.mod2 >self.mod2/2 else 0
            wm = (wm_1*3+wm_2*1)/4
        else:
            wm = wm_1
        return wm

    def extract(self,filename,out_wm_name,debug_path="rank_reconstruct_debug.png"):
        if not self.wm_shape:
            print("水印的形状未设定")
            return 0
        
        #读取图片
        embed_img = cv_imread(filename).astype(np.float32)
        if self.color_mod == 'RGB':
            embed_img_YUV = embed_img
        elif self.color_mod == 'YUV':
            embed_img_YUV = cv2.cvtColor(embed_img, cv2.COLOR_BGR2YUV)
            

        if not embed_img_YUV.shape[0]%(2**self.dwt_deep)==0:
            temp = (2**self.dwt_deep)-embed_img_YUV.shape[0]%(2**self.dwt_deep)
            embed_img_YUV = np.concatenate((embed_img_YUV,np.zeros((temp,embed_img_YUV.shape[1],3))),axis=0)
        if not embed_img_YUV.shape[1]%(2**self.dwt_deep)==0:
            temp = (2**self.dwt_deep)-embed_img_YUV.shape[1]%(2**self.dwt_deep)
            embed_img_YUV = np.concatenate((embed_img_YUV,np.zeros((embed_img_YUV.shape[0],temp,3))),axis=1)

        assert embed_img_YUV.shape[0]%(2**self.dwt_deep)==0
        assert embed_img_YUV.shape[1]%(2**self.dwt_deep)==0

        print(">>> 正在进行 Block 级光照重构与像素重组...")
        # 注意：我们只处理 Y 通道用于解密
        # embed_img_Y = embed_img_YUV[:,:,0]
        embed_img_Y = embed_img_YUV[:,:,0].astype(np.float32)
        embed_img_U = embed_img_YUV[:,:,1]
        embed_img_V = embed_img_YUV[:,:,2]
        
        # 对 Y 通道应用重组
        clean_Y = self.rank_quantizer.apply(embed_img_Y)
        
        # 保存重组后的图看看 (你会发现图片变得像马赛克一样干净)
        cv_imwrite(debug_path, clean_Y)
        print(f">>> 重组完成，中间结果已保存: {debug_path}")

        embed_img_Y = clean_Y
        
        coeffs_Y = dwt2(embed_img_Y,'haar')
        coeffs_U = dwt2(embed_img_U,'haar')
        coeffs_V = dwt2(embed_img_V,'haar')
        ha_Y = coeffs_Y[0]
        ha_U = coeffs_U[0]
        ha_V = coeffs_V[0]
        #对ha进一步进行小波变换,并把下一级ha保存到ha中
        for i in range(self.dwt_deep-1):
            coeffs_Y = dwt2(ha_Y,'haar')
            ha_Y = coeffs_Y[0]
            coeffs_U = dwt2(ha_U,'haar')
            ha_U = coeffs_U[0]
            coeffs_V = dwt2(ha_V,'haar')
            ha_V = coeffs_V[0]
        
        
        #初始化块索引数组
        try :
            if self.ha_Y.shape == ha_Y.shape :
                self.init_block_add_index(ha_Y.shape)
            else:
                print('你现在要解水印的图片与之前读取的原图的形状不同,这是不被允许的')
        except:
            self.init_block_add_index(ha_Y.shape)


        ha_block_shape = (int(ha_Y.shape[0]/self.block_shape[0]),int(ha_Y.shape[1]/self.block_shape[1]),self.block_shape[0],self.block_shape[1])
        strides = ha_Y.itemsize*(np.array([ha_Y.shape[1]*self.block_shape[0],self.block_shape[1],ha_Y.shape[1],1]))
        
        ha_Y_block = np.lib.stride_tricks.as_strided(ha_Y.copy(),ha_block_shape,strides)
        ha_U_block = np.lib.stride_tricks.as_strided(ha_U.copy(),ha_block_shape,strides)
        ha_V_block = np.lib.stride_tricks.as_strided(ha_V.copy(),ha_block_shape,strides)


        extract_wm   = np.array([])
        extract_wm_Y = np.array([])
        extract_wm_U = np.array([])
        extract_wm_V = np.array([])
        self.random_dct = np.random.RandomState(self.random_seed_dct)

        index = np.arange(self.block_shape[0]*self.block_shape[1])
        for i in range(self.length):
            self.random_dct.shuffle(index)
            wm_Y = self.block_get_wm(ha_Y_block[self.block_add_index0[i],self.block_add_index1[i]],index)
            wm_U = self.block_get_wm(ha_U_block[self.block_add_index0[i],self.block_add_index1[i]],index)
            wm_V = self.block_get_wm(ha_V_block[self.block_add_index0[i],self.block_add_index1[i]],index)
            # wm = round((wm_Y+wm_U+wm_V)/3)
            wm = wm_Y  # 暂时只信任 Y 通道
            

            #else情况是对循环嵌入的水印的提取
            if i<self.wm_shape[0]*self.wm_shape[1]:
                extract_wm   = np.append(extract_wm,wm)
                extract_wm_Y = np.append(extract_wm_Y,wm_Y)
                extract_wm_U = np.append(extract_wm_U,wm_U)
                extract_wm_V = np.append(extract_wm_V,wm_V)
            else:
                times = int(i/(self.wm_shape[0]*self.wm_shape[1]))
                ii = i%(self.wm_shape[0]*self.wm_shape[1])
                extract_wm[ii]   = (extract_wm[ii]*times +   wm  )/(times+1)
                extract_wm_Y[ii] = (extract_wm_Y[ii]*times + wm_Y)/(times+1)
                extract_wm_U[ii] = (extract_wm_U[ii]*times + wm_U)/(times+1)
                extract_wm_V[ii] = (extract_wm_V[ii]*times + wm_V)/(times+1)

        for i in range(extract_wm.size):
            extract_wm[i] = 0 if extract_wm[i] < 128 else 255

        wm_index = np.arange(extract_wm.size)
        self.random_wm = np.random.RandomState(self.random_seed_wm)
        self.random_wm.shuffle(wm_index)
        extract_wm[wm_index]   = extract_wm.copy()
        extract_wm_Y[wm_index] = extract_wm_Y.copy()
        extract_wm_U[wm_index] = extract_wm_U.copy()
        extract_wm_V[wm_index] = extract_wm_V.copy()
        cv_imwrite(out_wm_name,extract_wm.reshape(self.wm_shape[0],self.wm_shape[1]))

        path,file_name = os.path.split(out_wm_name)
        if not os.path.isdir(os.path.join(path,'Y_U_V')):
            os.mkdir(os.path.join(path,'Y_U_V'))
        cv_imwrite(os.path.join(path,'Y_U_V','Y'+file_name),extract_wm_Y.reshape(self.wm_shape[0],self.wm_shape[1]))
        cv_imwrite(os.path.join(path,'Y_U_V','U'+file_name),extract_wm_U.reshape(self.wm_shape[0],self.wm_shape[1]))
        cv_imwrite(os.path.join(path,'Y_U_V','V'+file_name),extract_wm_V.reshape(self.wm_shape[0],self.wm_shape[1]))
    
    # def diagnose_detailed(self, embed_img_path, ori_img_path, wm_path):
    #     print("="*30 + " 深度诊断报告 " + "="*30)
        
    #     # --- 1. 准备标准答案 (水印) ---
    #     wm_ori = cv_imread(wm_path)[:,:,0]
    #     wm_flatten = wm_ori.flatten()
    #     if self.random_seed_wm:
    #         np.random.RandomState(self.random_seed_wm).shuffle(wm_flatten)

    #     # --- 2. 准备图片数据 (原图 & 嵌入图) ---
    #     # 辅助函数：读取并获取 LL 层分块数据
    #     def get_ll_blocks(path):
    #         img = cv_imread(path).astype(np.float32)
    #         if self.color_mod == 'YUV':
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            
    #         # 补零/填充逻辑 (保持与你 embed 时一致，这里简化为根据倍数裁切或假设尺寸正确)
    #         if not img.shape[0]%(2**self.dwt_deep)==0:
    #             temp = (2**self.dwt_deep)-img.shape[0]%(2**self.dwt_deep)
    #             img = np.concatenate((img,np.zeros((temp,img.shape[1],3))),axis=0)
    #         if not img.shape[1]%(2**self.dwt_deep)==0:
    #             temp = (2**self.dwt_deep)-img.shape[1]%(2**self.dwt_deep)
    #             img = np.concatenate((img,np.zeros((img.shape[0],temp,3))),axis=1)
    #         assert img.shape[0]%(2**self.dwt_deep)==0
    #         assert img.shape[1]%(2**self.dwt_deep)==0
    #         # 建议：为了诊断准确，最好确保输入图片已经是处理好尺寸的，或者在这里复制你完整的 padding 逻辑
    #         h, w = img.shape[:2]
    #         new_h = h - (h % (2**self.dwt_deep))
    #         new_w = w - (w % (2**self.dwt_deep))
    #         img = img[:new_h, :new_w] 

    #         # DWT 变换
    #         coeffs = dwt2(img[:,:,0], 'haar')
    #         ll = coeffs[0]
    #         for _ in range(self.dwt_deep - 1):
    #             coeffs = dwt2(ll, 'haar')
    #             ll = coeffs[0]
            
    #         # 分块
    #         h_bl, w_bl = ll.shape[0] // self.block_shape[0], ll.shape[1] // self.block_shape[1]
    #         strides = ll.itemsize * np.array([ll.shape[1]*self.block_shape[0], self.block_shape[1], ll.shape[1], 1])
    #         blocks = np.lib.stride_tricks.as_strided(ll.copy(), (h_bl, w_bl, self.block_shape[0], self.block_shape[1]), strides)
    #         return blocks, ll.shape

    #     try:
    #         blocks_ori, ll_shape = get_ll_blocks(ori_img_path)
    #         blocks_embed, _ = get_ll_blocks(embed_img_path)
    #     except Exception as e:
    #         print(f"图片读取或DWT处理失败: {e}")
    #         return

    #     # --- 3. 初始化索引 ---
    #     self.init_block_add_index(ll_shape)
    #     self.random_dct = np.random.RandomState(self.random_seed_dct)
    #     dct_shuffle_index = np.arange(self.block_shape[0]*self.block_shape[1])
        
    #     # --- 4. 打印表头 ---
    #     # ID: 块编号, Exp: 期望比特, Get: 提取比特
    #     # S_ori: 原奇异值, S_new: 现奇异值, Diff: 变化量
    #     # Mean: 块平均亮度, Std: 块标准差(纹理丰富度)
    #     # Dist: 距离翻转阈值的安全距离 (越小越危险)
    #     header = f"{'ID':<6} | {'Exp':<3} {'Get':<3} | {'S_ori':<8} {'S_new':<8} {'Diff':<6} | {'Mean':<6} {'Std':<6} | {'Dist':<6} | {'Reason'}"
    #     print(header)
    #     print("-" * 100)

    #     error_count = 0
    #     total_blocks = min(self.length, wm_flatten.size)

    #     for i in range(total_blocks):
    #         # 随机打乱 DCT 系数索引
    #         self.random_dct.shuffle(dct_shuffle_index)
            
    #         # 获取对应位置的块
    #         b_ori = blocks_ori[self.block_add_index0[i], self.block_add_index1[i]]
    #         b_embed = blocks_embed[self.block_add_index0[i], self.block_add_index1[i]]
            
    #         # --- 原图分析 ---
    #         # 统计特征：亮度与纹理
    #         val_mean = np.mean(b_ori)
    #         val_std = np.std(b_ori)

    #         # 计算原奇异值
    #         dct_ori = cv2.dct(b_ori).flatten()[dct_shuffle_index].reshape(self.block_shape)
    #         _, s_o, _ = np.linalg.svd(dct_ori)
    #         s_ori_val = s_o[0]

    #         # --- 嵌入图分析 ---
    #         # 计算现奇异值
    #         dct_embed = cv2.dct(b_embed).flatten()[dct_shuffle_index].reshape(self.block_shape)
    #         _, s_e, _ = np.linalg.svd(dct_embed)
    #         s_new_val = s_e[0]

    #         if i==838:
    #             print("block_ori:",b_ori)
    #             print("block_embed:",b_embed)
    #             print("dct_ori:",dct_ori)
    #             print("dct_embed:",dct_embed)
    #         # --- 判定逻辑 ---
    #         expected = wm_flatten[i] # 0 or 255 (归一化前)
    #         # 模拟提取逻辑
    #         mod_res = s_new_val % self.mod
    #         extracted = 255 if mod_res > self.mod / 2 else 0
            
    #         # --- 深度计算 ---
    #         diff_val = s_new_val - s_ori_val # 奇异值变了多少
    #         # 安全距离：离 mod/2 这个悬崖有多远。如果 dist 接近 0，说明是一个"灰色像素"候选者
    #         safe_dist = abs(mod_res - self.mod/2)

    #         # --- 输出筛选 ---
    #         # 只输出错误的，或者极其危险的块 (安全距离 < mod 的 10%)
    #         is_error = (expected >= 128 and extracted == 0) or (expected < 128 and extracted == 255)
    #         is_risky = safe_dist < (self.mod * 0.1)

    #         if is_error or is_risky:
    #             if is_error: error_count += 1
                
    #             # 构造原因字符串
    #             reasons = []
    #             if val_std < 2.0: reasons.append("纯色区域")
    #             elif val_std < 5.0: reasons.append("低纹理")
    #             if val_mean < 5 or val_mean > 250: reasons.append("极暗/极亮")
    #             if is_risky: reasons.append("边缘徘徊")
    #             if abs(diff_val) > self.mod: reasons.append("剧烈变化")
                
    #             reason_str = ",".join(reasons) if reasons else "未知干扰"
                
    #             # 格式化输出
    #             # 把 expected 从 0/255 转为 0/1 显示更直观
    #             exp_bit = 1 if expected > 128 else 0
    #             get_bit = 1 if extracted > 128 else 0
                
    #             print(f"{i:<6} | {exp_bit:<3} {get_bit:<3} | {s_ori_val:<8.1f} {s_new_val:<8.1f} {diff_val:<6.1f} | {val_mean:<6.1f} {val_std:<6.1f} | {safe_dist:<6.1f} | {reason_str}")
                
    #             # if error_count > 30: # 限制输出行数
    #             #     print("... (错误太多，中断输出)")
    #             #     break

    #     print("-" * 100)
    #     print(f"总检查块数: {total_blocks}, 错误数: {error_count}, 错误率: {error_count/total_blocks:.2%}")

    def diagnose_detailed(self, embed_img_path, ori_img_path, wm_path, save_vis_path="error_vis.png"):
        print("="*30 + " 深度诊断报告 " + "="*30)
        
        # --- 0. 准备可视化画布 ---
        # 读取原图用于画框，注意保持为 BGR 格式以便 cv2 画图
        vis_img = cv_imread(ori_img_path)
        # 计算坐标映射的缩放因子：LL层 1个像素 = 原图 2^deep 个像素
        scale_factor = 2 ** self.dwt_deep

        # --- 1. 准备标准答案 (水印) ---
        wm_ori = cv_imread(wm_path)[:,:,0]
        wm_flatten = wm_ori.flatten()
        if self.random_seed_wm:
            np.random.RandomState(self.random_seed_wm).shuffle(wm_flatten)

        # --- 2. 准备图片数据 ---
        # 辅助函数：读取并获取 LL 层分块数据
        def get_ll_blocks(path):
            img = cv_imread(path).astype(np.float32)

            if self.color_mod == 'YUV':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            
            # 简单的尺寸修正 (假设输入图尺寸已经大体符合要求，这里做微调)
            h, w = img.shape[:2]
            new_h = h - (h % (2**self.dwt_deep))
            new_w = w - (w % (2**self.dwt_deep))
            img = img[:new_h, :new_w] 

            # DWT 变换
            coeffs = dwt2(img[:,:,0], 'haar')
            ll = coeffs[0]
            for _ in range(self.dwt_deep - 1):
                coeffs = dwt2(ll, 'haar')
                ll = coeffs[0]
            
            # 分块
            h_bl, w_bl = ll.shape[0] // self.block_shape[0], ll.shape[1] // self.block_shape[1]
            strides = ll.itemsize * np.array([ll.shape[1]*self.block_shape[0], self.block_shape[1], ll.shape[1], 1])
            blocks = np.lib.stride_tricks.as_strided(ll.copy(), (h_bl, w_bl, self.block_shape[0], self.block_shape[1]), strides)
            return blocks, ll.shape

        try:
            blocks_ori, ll_shape = get_ll_blocks(ori_img_path)
            blocks_embed, _ = get_ll_blocks(embed_img_path)
        except Exception as e:
            print(f"图片读取或DWT处理失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # --- 3. 初始化索引 ---
        self.init_block_add_index(ll_shape)
        self.random_dct = np.random.RandomState(self.random_seed_dct)
        dct_shuffle_index = np.arange(self.block_shape[0]*self.block_shape[1])
        
        # --- 4. 打印表头 ---
        header = f"{'ID':<6} | {'Exp':<3} {'Get':<3} | {'S_ori':<8} {'S_upd':<8} {'S_new':<8} {'Diff':<6} | {'Mean':<6} {'Std':<6} | {'Dist':<6} | {'Reason'}"
        print(header)
        print("-" * 100)

        error_count = 0
        total_blocks = self.length # 检查所有块

        example = []
        ex_idx = 9350
        for i in range(total_blocks):
            self.random_dct.shuffle(dct_shuffle_index)
            
            # 获取块，原图需要预处理
            b_ori = blocks_ori[self.block_add_index0[i], self.block_add_index1[i]]
            # 安全预处理
            target_center = 128 + self.center_alpha * (b_ori.mean() - 128)
            b_ori = (b_ori - target_center) * self.alpha + target_center
            
            b_embed = blocks_embed[self.block_add_index0[i], self.block_add_index1[i]]
            
            # 统计特征
            val_mean = np.mean(b_ori)
            val_std = np.std(b_ori)

            # 计算奇异值
            dct_ori = cv2.dct(b_ori).flatten()[dct_shuffle_index].reshape(self.block_shape)
            _, s_o, _ = np.linalg.svd(dct_ori)
            s_ori_val = s_o[0]

            dct_embed = cv2.dct(b_embed).flatten()[dct_shuffle_index].reshape(self.block_shape)
            _, s_e, _ = np.linalg.svd(dct_embed)
            s_new_val = s_e[0]
            
    
            # 使用取模运算，循环获取水印位
            expected = wm_flatten[i % wm_flatten.size]
            s_ori_new = (s_ori_val-s_ori_val%self.mod+3/4*self.mod) if expected>=128 else (s_ori_val-s_ori_val%self.mod+1/4*self.mod)
            #测试案例
            if i == ex_idx:
                example.append(s_o[0])
                example.append(s_ori_new)
                example.append(s_e[0])
                example.append(b_ori)
                example.append(b_embed)
                example.append(dct_ori)
                example.append(dct_embed)
                
            
            mod_res = s_new_val % self.mod
            extracted = 255 if mod_res > self.mod / 2 else 0
            
            diff_val = s_new_val - s_ori_new
            safe_dist = abs(mod_res - self.mod/2)

            is_error = (expected >= 128 and extracted == 0) or (expected < 128 and extracted == 255)
            is_risky = safe_dist < (self.mod * 0.1)

            # --- 可视化绘制部分 Start ---
            if is_error or is_risky:
                # 1. 获取块在 LL 层的左上角坐标 (行, 列)
                r_idx = self.block_add_index0[i]
                c_idx = self.block_add_index1[i]
                
                # 2. 映射回原图坐标
                # 坐标 = 块索引 * 块大小 * DWT缩放倍数
                y = int(r_idx * self.block_shape[0] * scale_factor)
                x = int(c_idx * self.block_shape[1] * scale_factor)
                h = int(self.block_shape[0] * scale_factor)
                w = int(self.block_shape[1] * scale_factor)

                # 3. 绘制矩形
                if is_error:
                    # 红色框：提取错误
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                elif is_risky:
                    # 黄色框：提取正确但非常危险 (边缘徘徊)
                    # 可选：如果你只想要报错的，可以注释掉这个elif
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 255), 1)
            # --- 可视化绘制部分 End ---

            if is_error:
                if is_error: error_count += 1
                reasons = []
                if val_std < 2.0: reasons.append("纯色区域")
                elif val_std < 5.0: reasons.append("低纹理")
                if val_mean < 5 or val_mean > 250: reasons.append("极暗/极亮")
                if is_risky: reasons.append("边缘徘徊")
                if abs(diff_val) > self.mod: reasons.append("剧烈变化")
                reason_str = ",".join(reasons) if reasons else "未知干扰"
                
                exp_bit = 1 if expected > 128 else 0
                get_bit = 1 if extracted > 128 else 0
                
                print(f"{i:<6} | {exp_bit:<3} {get_bit:<3} | {s_ori_val:<8.1f} {s_ori_new:<8.1f} {s_new_val:<8.1f} {diff_val:<6.1f} | {val_mean:<6.1f} {val_std:<6.1f} | {safe_dist:<6.1f} | {reason_str}")

        print("-" * 100)
        print(f"总检查块数: {total_blocks}, 错误数: {error_count}, 错误率: {error_count/total_blocks:.2%}")
        
        # 测试案例
        print("s_ori:",example[0])
        print("s_upd:",example[1])
        print("s_new:",example[2])
        print("block_ori:",example[3])
        print("block_new:",example[4])
        print("dct_ori:",example[5])
        print("dct_new:",example[6])
        
        # 保存可视化结果
        cv_imwrite(save_vis_path, vis_img)
        print(f"可视化诊断图已保存至: {save_vis_path}")
        
        # 如果在 Jupyter/Colab 中，可以显示出来
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.title(f"Error Visualization (Red=Error, Yellow=Risky)")
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        except:
            pass

if __name__=="__main__":
    bwm1 = watermark(4399,2333,64)
    bwm1.read_ori_img("Wechat2.png")
    bwm1.read_wm("clock_wm.png")
    bwm1.embed('out.png')
    bwm1.extract("out.png","./out_wm.png")

    # 诊断
    bwm1.diagnose_detailed('out.png', 'Wechat2.png', 'clock_wm.png')
    
    # bwm2 = watermark(7373,1024,22,12)
    # bwm2.read_ori_img('out.png')
    # bwm2.read_wm('pic/wm2.png')
    # bwm2.embed('out2.png')
    # bwm2.extract('out2.png','./out_wm2.png')

    # bwm1.extract('out2.png','./bwm1_out2.png')
    # bwm2.extract('out.png','./bwm2_out.png')

        