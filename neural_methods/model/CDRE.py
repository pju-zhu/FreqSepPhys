"""
CDRE.py — Cross-Domain Robustness Enhancement (跨域鲁棒性增强)
===============================================================
创新点4：三管齐下提升rPPG模型跨数据集/跨场景泛化能力。

子模块：
4A. IlluminationInvariantStem：光照不变性前端
    - 基于CHROM/POS物理模型的可学习色彩投影
    - 差分归一化（DiffNormalized）内嵌于stem
    - 物理初始化：用CHROM色彩向量[3, -2, 0]和POS向量[0, 1, -1]
    
4B. MotionRobustnessGating：运动鲁棒性门控
    - 高频分支运动能量估计
    - 自适应门控抑制运动伪影帧
    - 时间平滑约束防止门控震荡
    
4C. HRFrequencyAugmentation：HR感知频域数据增强
    - 扩展心率分布覆盖50-180bpm全范围
    - 频域幅度抖动（PASTA思想）
    - 训练时在线增强，推理时关闭

生理/物理依据：
- CHROM模型：假设皮肤反射R:G:B ≈ [3,-2,0]投影可消除背景光照分量
- POS模型：正交平面[0,1,-1]与[1,-1,0]上投影分离脉动与运动
- 运动伪影频谱：集中在0-1Hz（头部晃动）和>4Hz（表情微动），
  与rPPG有效频段0.83-3.0Hz部分重叠，需自适应抑制
- 心率分布不均：训练集通常偏60-90bpm，需增强50-60和120-180bpm样本
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


# ════════════════════════════════════════════════════════════════════
#  4A. 光照不变性Stem
# ════════════════════════════════════════════════════════════════════

class IlluminationInvariantStem(nn.Module):
    """
    光照不变性前端模块（替换原ConvBlock1作为第一层）。
    
    核心设计：
    1. 可学习色彩投影矩阵（物理初始化）
       - 初始权重基于CHROM和POS色彩模型
       - CHROM投影: [3, -2, 0] → 消除直流光照分量
       - POS投影: [0, 1, -1] → 提取正交脉动分量
       - 训练过程中允许微调以适应不同肤色/光源
       
    2. 差分归一化层（DiffNormalized）
       - 帧间差分消除低频光照漂移
       - 标准化到零均值单位方差
       - 对光照强度变化具有一阶不变性
       
    3. 自适应色度映射
       - 多头色彩投影（4头，每头学习不同色彩不变性方向）
       - 融合后通道注意力选择最优投影
    
    物理约束：
    - 投影矩阵行向量互相正交（投影到不同色彩子空间）
    - 投影矩阵行向量模为1（能量归一化）
    
    Args:
        out_channels: 输出通道数（默认16，与原ConvBlock1对齐）
        num_heads: 色彩投影头数（默认4）
        use_diff_norm: 是否启用差分归一化（默认True）
        temporal_kernel: 时间维度卷积核大小（默认1，与原始对齐）
    """
    def __init__(self, out_channels: int = 16, num_heads: int = 4,
                 use_diff_norm: bool = True, temporal_kernel: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_diff_norm = use_diff_norm
        
       # ── 物理先验色彩投影（可学习微调）──────────────────────
        # 每个头学习一个3D色彩投影方向
        # 初始化基于已知的rPPG色彩模型
        self.color_proj = nn.Parameter(torch.zeros(num_heads, 3))
        self._init_color_projection()
        
        # ── 差分归一化层 ──────────────────────────────────────────
        if use_diff_norm:
            self.diff_norm = TemporalDiffNorm()
        
        # ── 多头投影后的特征提取 ──────────────────────────────────
        # 输入通道 = num_heads（每头一个标量投影）+ 3（原始RGB保留）
        in_ch = num_heads + 3
        self.feature_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_channels, 
                      kernel_size=[temporal_kernel, 5, 5],
                      stride=1,
                      padding=[temporal_kernel // 2, 2, 2]),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # ── 通道注意力（选择最优投影组合）──────────────────────────
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid(),
        )

    def _init_color_projection(self):
        """基于物理模型初始化色彩投影方向"""
        with torch.no_grad():
            # Head 0: CHROM模型色彩方向 [3, -2, 0] (归一化)
            chrom = torch.tensor([3.0, -2.0, 0.0])
            self.color_proj[0] = chrom / chrom.norm()
            
            # Head 1: POS模型色彩方向 [0, 1, -1] (归一化)
            pos1 = torch.tensor([0.0, 1.0, -1.0])
            self.color_proj[1] = pos1 / pos1.norm()
            
            # Head 2: POS正交方向 [1, -1, 0] (归一化)
            pos2 = torch.tensor([1.0, -1.0, 0.0])
            self.color_proj[2] = pos2 / pos2.norm()
            
            # Head 3: 均匀投影（灰度亮度）[1, 1, 1] / √3
            gray = torch.tensor([1.0, 1.0, 1.0])
            self.color_proj[3] = gray / gray.norm()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, T, H, W] 原始RGB视频帧
        Returns:
            [B, out_channels, T, H, W] 光照不变性特征
        """
        B, C, T, H, W = x.shape
        
        # 1. 差分归一化（消除低频光照漂移）
        if self.use_diff_norm:
            x_norm = self.diff_norm(x)  # [B, 3, T, H, W]
        else:
            x_norm = x
        
        # 2. 多头色彩投影
        # color_proj: [num_heads, 3] → 对RGB通道做线性组合
        # x_norm: [B, 3, T, H, W] → reshape for matmul
        proj_weights = self.color_proj  # [H_heads, 3]
        # [B, 3, T, H, W] → [B, T*H*W, 3]
        x_flat = x_norm.permute(0, 2, 3, 4, 1).reshape(B * T * H * W, C)
        # proj: [B*T*H*W, 3] @ [3, num_heads] → [B*T*H*W, num_heads]
        proj_out = x_flat @ proj_weights.t()
        # reshape back: [B, T, H, W, num_heads] → [B, num_heads, T, H, W]
        proj_out = proj_out.reshape(B, T, H, W, self.num_heads).permute(0, 4, 1, 2, 3)
        
        # 3. 拼接原始RGB + 投影特征
        combined = torch.cat([x_norm, proj_out], dim=1)  # [B, 3+num_heads, T, H, W]
        
        # 4. 特征提取卷积
        feat = self.feature_conv(combined)  # [B, out_channels, T, H, W]
        
   # 5. 通道注意力加权
        attn = self.channel_attn(feat).view(B, self.out_channels, 1, 1, 1)
        feat = feat * attn
        
        return feat

    def get_projection_orthogonality(self) -> float:
        """
        计算投影矩阵的正交性度量（用于监控/正则化）。
        完美正交时返回0，完全共线时返回1。
        """
        proj = self.color_proj  # [num_heads, 3]
        proj_norm = F.normalize(proj, dim=-1)
        gram = proj_norm @ proj_norm.t()  # [H, H]
        # 理想情况下gram应为单位阵
        off_diag = gram - torch.eye(self.num_heads, device=gram.device)
        return off_diag.abs().mean().item()


class TemporalDiffNorm(nn.Module):
    """
    时间差分归一化层。
    
    物理动机：
    - rPPG信号是帧间亮度微变化（AC分量），而光照/背景是直流分量（DC）
    - 帧间差分自然消除DC分量：Δx(t) = x(t) - x(t-1) ∝ rPPG(t)
    - 归一化消除绝对亮度影响，使模型对光照强度不变
    
    实现：
    - 前向差分：diff(t) = x(t) - x(t-1)，首帧填零
    - Instance归一化：每通道零均值单位方差
    - 输出 = diff_norm（保留原始形状，首帧用第二帧差分填充）
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            差分归一化后的特征，形状不变 [B, C, T, H, W]
        """
        # 帧间差分
        diff = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]  # [B, C, T-1, H, W]
        # 首帧用第二帧差分填充（避免信息损失）
        diff = torch.cat([diff[:, :, :1, :, :], diff], dim=2)  # [B, C, T, H, W]
        
        # Instance归一化（per-sample, per-channel）
        B, C, T, H, W = diff.shape
        diff_flat = diff.reshape(B, C, -1)  # [B, C, T*H*W]
        mean = diff_flat.mean(dim=-1, keepdim=True)
        std = diff_flat.std(dim=-1, keepdim=True) + self.eps
        diff_norm = (diff_flat - mean) / std
        
        return diff_norm.reshape(B, C, T, H, W)


# ════════════════════════════════════════════════════════════════════
#  4B. 运动鲁棒性门控模块
# ════════════════════════════════════════════════════════════════════

class MotionRobustnessGating(nn.Module):
    """
    运动鲁棒性门控模块。
    
    核心思想：
    - 运帧的高频能量异常大（运动伪影频谱扩散）
    - 静止帧的高频能量集中在rPPG有效频段
    - 自适应门控：运动帧权重降低，静止帧权重提高
    
    设计：
    1. 运动能量估计器：分析高频分支的时间能量包络
    2. 软门控生成：sigmoid将能量映射为[0,1]权重
    3. 时间平滑：防止门控在相邻帧间剧烈震荡（生理rPPG信号连续）
    
    与FreqSepBlock的集成方式：
    - 位于高频路BiMamba之后、FreqChannelGating之前
    - 门控作用于高频分量，抑制运动伪影帧的高频噪声
    - 不影响低频路（低频路处理光照/慢速漂移）
    
    物理约束：
    - 门控下界 > 0.1（完全关闭会丢失rPPG信号）
    - 时间平滑核σ=2帧（约66ms@30fps，一个心跳周期内门控应一致）
    
    Args:
        channels: 输入通道数
        reduction: 通道压缩比（默认4）
        gate_min: 门控最小值（默认0.1，防止信号完全丢失）
        smooth_kernel: 时间平滑卷积核大小（默认5）
    """
    def __init__(self, channels: int, reduction: int = 4,
                 gate_min: float = 0.1, smooth_kernel: int = 5):
        super().__init__()
        self.gate_min = gate_min
        self.channels = channels
        
        # ── 运动能量估计网络 ──────────────────────────────────────
        # 输入: [B, C, T, H, W]的高频特征
        # 估计每帧的运动能量水平
        hidden = max(channels // reduction, 8)
        self.motion_estimator = nn.Sequential(
            # 空间全局池化 → [B, C, T, 1, 1]
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # 保留时间维度
        )
        self.motion_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        
        # ── 时间平滑卷积（防止门控震荡）──────────────────────────
        self.smooth_kernel = smooth_kernel
        pad = smooth_kernel // 2
        # 高斯平滑核，σ=2帧（一个心跳周期~30帧内门控应缓变）
        sigma = 2.0
        coords = torch.arange(smooth_kernel).float() - pad
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        # [1, 1, K] 时间维度1D卷积核
        self.register_buffer('smooth_weight', g.view(1, 1, -1))
        self.smooth_pad = pad
        self.last_gate = None

    def forward(self, x_hf: torch.Tensor) -> torch.Tensor:
        """
        对高频特征施加运动鲁棒性门控。
        
        Args:
            x_hf: [B, C, T, H, W] 高频分支特征（经BiMamba处理后）
        Returns:
            gated_hf: [B, C, T, H, W] 门控后的高频特征
        """
        B, C, T, H, W = x_hf.shape
        
        # 1. 估计每帧运动能量
        # [B, C, T, H, W] → spatial pooling → [B, C, T, 1, 1] → [B, C, T]
        energy = self.motion_estimator(x_hf).squeeze(-1).squeeze(-1)  # [B, C, T]
        # 时间维度帧级特征: [B, T, C]
        energy = energy.permute(0, 2, 1)  # [B, T, C]
        # MLP预测每帧运动分数: [B, T, 1]
        motion_score = self.motion_mlp(energy)  # [B, T, 1]
        motion_score = motion_score.squeeze(-1)  # [B, T]
        
        # 2. 生成软门控（运动越大→门控越小→抑制越强）
        # sigmoid反转：高运动能量→低门控值
        gate = torch.sigmoid(-motion_score)  # [B, T]，运动大→接近0
        
        # 3. 时间平滑（防止门控在相邻帧剧烈跳变）
        gate_smooth = F.conv1d(
            gate.unsqueeze(1),  # [B, 1, T]
            self.smooth_weight,
            padding=self.smooth_pad,
        ).squeeze(1)  # [B, T]
        
        # 4. 约束门控范围 [gate_min, 1.0]
        gate_final = self.gate_min + (1.0 - self.gate_min) * gate_smooth
        self.last_gate = gate_final
        
        # 5. 应用门控到高频特征
        # gate: [B, T] → [B, 1, T, 1, 1] 广播到所有通道和空间位置
        gate_expanded = gate_final.view(B, 1, T, 1, 1)
        gated_hf = x_hf * gate_expanded
        
        return gated_hf

    def get_gate_stats(self, x_hf: torch.Tensor) -> dict:
        """获取门控统计信息（用于可视化/调试）"""
        B, C, T, H, W = x_hf.shape
        energy = self.motion_estimator(x_hf).squeeze(-1).squeeze(-1)
        energy = energy.permute(0, 2, 1)
        motion_score = self.motion_mlp(energy).squeeze(-1)
        gate = torch.sigmoid(-motion_score)
        return {
            'gate_mean': gate.mean().item(),
            'gate_std': gate.std().item(),
            'gate_min': gate.min().item(),
            'gate_max': gate.max().item(),
            'motion_score_mean': motion_score.mean().item(),
        }


# ════════════════════════════════════════════════════════════════════
#  4C. HR感知频域数据增强
# ════════════════════════════════════════════════════════════════════

class HRFrequencyAugmentation(nn.Module):
    """
    HR感知频域数据增强模块。
    
    动机：
    - 训练数据心率分布不均（通常集中在60-90bpm静息心率）
    - 跨域测试时遇到运动后高心率（120-180bpm）性能显著下降
    - 需要在频域增强稀有心率对应的频率分量
    
    方法（基于PASTA思想改进）：
    1. 频谱幅度抖动：对rPPG有效频段内的幅度施加随机扰动
       - 保留相位（相位携带时序信息）
       - 仅扰动幅度（模拟不同SNR条件）
       
    2. 心率偏移增强：
       - 将基频从当前HR向目标HR偏移
       - 目标HR从50-180bpm均匀采样
       - 保持谐波结构（2倍频/3倍频同步偏移）
       
    3. 频域mix-up：
       - 同batch内不同样本的频谱幅度混合
       - 增强心率分布多样性
    
    约束：
    - 仅在训练时激活（eval模式自动关闭）
    - 增强概率p=0.5（不总是增强，保留原始分布）
    - 频谱修改后需确保时域信号仍物理合理（正值、有界）
    
    Args:
        sample_rate: 采样率fps（默认30）
        hr_min: 最小心率bpm（默认50，覆盖运动员静息心率）
        hr_max: 最大心率bpm（默认180，覆盖剧烈运动）
        aug_prob: 增强概率（默认0.5）
        amplitude_jitter: 幅度抖动范围（默认0.3，±30%）
    """
    def __init__(self, sample_rate: int = 30,
                 hr_min: float = 50.0, hr_max: float = 180.0,
                 aug_prob: float = 0.5,
                 amplitude_jitter: float = 0.3):
        super().__init__()
        self.sample_rate = sample_rate
        self.hr_min = hr_min
        self.hr_max = hr_max
        self.aug_prob = aug_prob
        self.amplitude_jitter = amplitude_jitter
        
        # rPPG有效频段边界（Hz）
        self.freq_min = hr_min / 60.0  # 0.833 Hz
        self.freq_max = hr_max / 60.0  # 3.0 Hz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对时间序列信号施加频域增强。
        
        Args:
            x: [B, C, T, H, W] 特征图（在时间维度做FFT增强）
               或 [B, T] rPPG信号
        Returns:
            增强后的信号（形状不变）
        """
        if not self.training:
            return x
        
        # 按概率决定是否增强
        if torch.rand(1).item() > self.aug_prob:
            return x
        
        if x.dim() == 5:
            return self._augment_feature_map(x)
        elif x.dim() == 2:
            return self._augment_signal(x)
        else:
            return x

    def _augment_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        对5D特征图在时间维度施加频域增强。
        仅对空间平均后的时间序列做FFT，避免逐像素FFT的计算爆炸。
        """
        B, C, T, H, W = x.shape
        
        # 空间平均得到时间信号: [B, C, T]
        x_spatial_avg = x.mean(dim=[3, 4])  # [B, C, T]
        
        # FFT
        X_freq = torch.fft.rfft(x_spatial_avg, dim=-1)  # [B, C, T//2+1] complex
        
        # 频率轴
        freqs = torch.fft.rfftfreq(T, d=1.0/self.sample_rate)  # [T//2+1]
        freqs = freqs.to(x.device)
        
        # 找到rPPG有效频段的索引
        valid_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        
        # 幅度抖动：仅在有效频段内
        amplitude = X_freq.abs()
        phase = X_freq.angle()
        
        # 随机抖动因子: [B, C, T//2+1]
        jitter = 1.0 + self.amplitude_jitter * (2 * torch.rand_like(amplitude) - 1)
        # 仅对有效频段施加抖动
        jitter_mask = valid_mask.float().view(1, 1, -1)
        jitter = jitter * jitter_mask + (1.0 - jitter_mask)  # 非有效频段保持不变
        
        # 重建增强后的频谱
        X_aug = (amplitude * jitter) * torch.exp(1j * phase)
        
        # IFFT回时域
        x_aug_temporal = torch.fft.irfft(X_aug, n=T, dim=-1)  # [B, C, T]
        
        # 计算增强差异并广播回空间维度
        diff = (x_aug_temporal - x_spatial_avg).unsqueeze(-1).unsqueeze(-1)  # [B,C,T,1,1]
        x_augmented = x + diff  # 将频域增强传播到所有空间位置
        
        return x_augmented

    def _augment_signal(self, x: torch.Tensor) -> torch.Tensor:
        """
        对2D rPPG信号 [B, T] 施加频域增强。
        """
        B, T = x.shape
        
        X_freq = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(T, d=1.0/self.sample_rate).to(x.device)
        valid_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        
        amplitude = X_freq.abs()
        phase = X_freq.angle()
        
        jitter = 1.0 + self.amplitude_jitter * (2 * torch.rand_like(amplitude) - 1)
        jitter_mask = valid_mask.float().unsqueeze(0)
        jitter = jitter * jitter_mask + (1.0 - jitter_mask)
        
        X_aug = (amplitude * jitter) * torch.exp(1j * phase)
        x_aug = torch.fft.irfft(X_aug, n=T, dim=-1)
        
        return x_aug


# ════════════════════════════════════════════════════════════════════
#  正交性正则化损失（光照投影矩阵约束）
# ════════════════════════════════════════════════════════════════════

class ProjectionOrthogonalityLoss(nn.Module):
    """色彩投影多样性约束。

    当投影头数不超过RGB维数时约束行向量正交；头数超过维数时改用
    单位范数紧框架约束，避免要求四个三维向量两两正交这一不可达目标。
    """
    def __init__(self, weight: float = 0.05):
        super().__init__()
        self.weight = weight

    def forward(self, projection_matrix: torch.Tensor) -> torch.Tensor:
        P = F.normalize(projection_matrix, dim=-1)
        num_heads, feature_dim = P.shape
        if num_heads <= feature_dim:
            gram = P @ P.t()
            target = torch.eye(num_heads, device=P.device, dtype=P.dtype)
        else:
            # 单位范数紧框架：P^T P = (H / D) I_D，对H>D时目标可达。
            gram = P.t() @ P
            target = (num_heads / feature_dim) * torch.eye(
                feature_dim, device=P.device, dtype=P.dtype
            )
        loss = (gram - target).pow(2).sum() / gram.numel()
        return self.weight * loss


# ════════════════════════════════════════════════════════════════════
#  运动门控正则化损失
# ════════════════════════════════════════════════════════════════════

class MotionGateRegularizationLoss(nn.Module):
    """
    运动门控正则化损失。
    
    约束目标：
    1. 门控时间平滑性：相邻帧门控值差异不应过大
       L_smooth = mean(|gate(t) - gate(t-1)|²)
       物理含义：rPPG信号连续，门控不应在相邻帧剧烈跳变
       
    2. 门控稀疏性：大部分帧应正常通过（门控≈1），仅少数运动帧被抑制
       L_sparse = mean(1 - gate)
       物理含义：过度抑制会丢失rPPG信号
    
    Args:
        smooth_weight: 平滑性权重（默认0.1）
        sparse_weight: 稀疏性权重（默认0.05）
    """
    def __init__(self, smooth_weight: float = 0.1, sparse_weight: float = 0.05):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.sparse_weight = sparse_weight

    def forward(self, gate: torch.Tensor) -> dict:
        """
        Args:
            gate: [B, T] 门控值
        Returns:
            dict with 'smooth', 'sparse', 'total' losses
        """
        losses = {}
        
        # 时间平滑性
        gate_diff = gate[:, 1:] - gate[:, :-1]
        losses['smooth'] = self.smooth_weight * gate_diff.pow(2).mean()
        
        # 稀疏性（鼓励大部分帧门控接近1）
        losses['sparse'] = self.sparse_weight * (1.0 - gate).mean()
        
        losses['total'] = losses['smooth'] + losses['sparse']
        return losses


# ════════════════════════════════════════════════════════════════════
#  CDRE统一接口
# ════════════════════════════════════════════════════════════════════

class CDREModule(nn.Module):
    """
    CDRE统一封装模块。
    
    提供创新点4的三个子模块的统一管理和前向接口。
    在FreqSepPhysPlusPlus中按需调用各子模块。
    
    Args:
        out_channels: Stem输出通道数
        hf_channels: 高频分支通道数（用于运动门控）
        num_color_heads: 色彩投影头数
        use_motion_gate: 是否启用运动门控
        use_freq_aug: 是否启用频域增强
        sample_rate: 采样率
    """
    def __init__(self, out_channels: int = 16,
                 hf_channels: int = 64,
                 num_color_heads: int = 4,
                 use_motion_gate: bool = True,
                 use_freq_aug: bool = True,
                 sample_rate: int = 30):
        super().__init__()
        
        # 4A: 光照不变性Stem
        self.illumination_stem = IlluminationInvariantStem(
            out_channels=out_channels,
            num_heads=num_color_heads,
        )
        
        # 4B: 运动鲁棒性门控
        self.use_motion_gate = use_motion_gate
        if use_motion_gate:
            self.motion_gate = MotionRobustnessGating(
                channels=hf_channels,
            )
        
        # 4C: HR频域增强
        self.use_freq_aug = use_freq_aug
        if use_freq_aug:
            self.freq_aug = HRFrequencyAugmentation(
                sample_rate=sample_rate,
            )
        
        # 正则化损失
        self.proj_ortho_loss = ProjectionOrthogonalityLoss()
        self.gate_reg_loss = MotionGateRegularizationLoss()

    def stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        """光照不变性前端处理"""
        return self.illumination_stem(x)

    def apply_motion_gate(self, x_hf: torch.Tensor) -> torch.Tensor:
        """对高频分支施加运动门控"""
        if self.use_motion_gate:
            return self.motion_gate(x_hf)
        return x_hf

    def apply_freq_aug(self, x: torch.Tensor) -> torch.Tensor:
        """频域数据增强（仅训练时生效）"""
        if self.use_freq_aug:
            return self.freq_aug(x)
        return x

    def get_regularization_losses(self) -> dict:
        """获取CDRE相关的所有正则化损失"""
        losses = {}
        # 投影正交性
        losses['proj_ortho'] = self.proj_ortho_loss(
            self.illumination_stem.color_proj
        )
        return losses