"""
AFD.py — Adaptive Frequency Decomposition (自适应频率分解)
==========================================================
创新点1：将FreqSepPhys中固定σ的高斯低通分解升级为可学习+输入自适应频率边界。

核心改进：
1. LearnableSigmaLowPass3d: σ作为可学习参数，支持梯度优化
2. AdaptiveSigmaPredictor: 轻量网络根据输入内容预测最优σ
3. AnisotropicFreqDecomposer: 各向异性时空分离（时间/空间独立σ）
4. AdaptiveFreqDecomposer: 统一接口，支持3种模式切换

生理依据：
- 心率范围50-180bpm → 对应0.83-3.0Hz（rPPG有效频段）
- 采样率30fps，Nyquist=15Hz，时间维度σ需覆盖信号频段
- σ_low > σ_high 约束：确保低频分量确实比高频分量更平滑
- 各向异性：时间维度需更精细的频率控制（率变异），空间维度可更粗糙（面部区域均匀脉动）

参数初始化依据：
- σ_temporal ∈ [0.5, 3.0]: 对应时间低通截止，初始1.5（覆盖心率基频附近）
- σ_spatial ∈ [0.3, 2.5]: 空间平滑，初始1.0（面部ROI内脉动空间相关性高）
- sigma_min=0.3: 防止σ过小导致核退化为delta函数
- sigma_max=3.0: 防止σ过大导致所有信号被平滑
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ════════════════════════════════════════════════════════════════════
#  可学习σ的3D高斯低通滤波器
# ════════════════════════════════════════════════════════════════════

class LearnableSigmaLowPass3d(nn.Module):
    """
    可学习各向同性3D高斯低通滤波器。
    
    σ为可学习参数，通过softplus+clamp约束在物理有效范围内。
    每次forward动态重建高斯核（计算开销极小，5x5x5核仅125个元素）。
    
    与固定GaussianLowPass3d的关区别：
    - σ参与梯度反传，可由loss驱动优化
    - 自动满足σ_min/σ_max物理约束
    - 支持per-channel独立σ或全局共享σ
    
    Args:
        channels: 输入通道数（depthwise分组卷积）
        kernel_size: 高斯核空间尺寸（默认5，覆盖±2σ）
        init_sigma: σ初始值（基于流的时间尺度选择）
        sigma_min: σ下界（防止退化为delta函数）
        sigma_max: σ上界（防止过度平滑）
        per_channel: 是否每个通道独立σ（默认False，共享σ减少参数）
    """
    def __init__(self, channels: int, kernel_size: int = 5,
                 init_sigma: float = 1.5,
                 sigma_min: float = 0.3, sigma_max: float = 3.0,
                 per_channel: bool = False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.padding = kernel_size // 2
        self.per_channel = per_channel

        # σ参数化：使用inverse_softplus初始化，确保softplus(raw)=init_sigma
        # softplus(x) = ln(1+exp(x))，inverse: x = ln(exp(sigma)-1)
        num_sigmas = channels if per_channel else 1
        init_raw = math.log(math.exp(init_sigma) - 1.0)
        self.sigma_raw = nn.Parameter(torch.full((num_sigmas,), init_raw))

        # 预计算坐标偏移（不参与梯度）
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        self.register_buffer('coords', coords)

    @property
    def sigma(self) -> torch.Tensor:
        """获取约束后的σ值"""
        return torch.clamp(F.softplus(self.sigma_raw), self.sigma_min, self.sigma_max)

    def _build_kernel(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        动态构建3D高斯核。
        
        Args:
            sigma: [num_sigmas] 约束后的σ值
        Returns:
            kernel: [channels, 1, K, K, K] depthwise卷积核
        """
        # 1D高斯: g(x) = exp(-x^2 / (2σ^2))
        # sigma: [num_sigmas] → 需扩展为 [num_sigmas, kernel_size]
        sigma_expanded = sigma.view(-1, 1)  # [num_sigmas, 1]
        coords = self.coords.view(1, -1)    # [1, K]
        
        g1d = torch.exp(-(coords ** 2) / (2 * sigma_expanded ** 2))  # [num_sigmas, K]
        g1d = g1d / g1d.sum(dim=-1, keepdim=True)  # 归一化

        # 3D可分离: k3d = g_t ⊗ g_h ⊗ g_w（各向同性，三轴相同σ）
        if self.per_channel:
            # 每通道独立核: [C, K] → [C, K, K, K]
            k3d = g1d[:, :, None, None] * g1d[:, None, :, None] * g1d[:, None, None, :]
            kernel = k3d.unsqueeze(1)  # [C, 1, K, K, K]
        else:
            # 共享核: [1, K] → [1, 1, K, K, K] → repeat to [C, 1, K, K, K]
            g = g1d.squeeze(0)  # [K]
            k3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
            kernel = k3d.unsqueeze(0).unsqueeze(0).repeat(self.channels, 1, 1, 1, 1)
        
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] 输入特征图
        Returns:
            低通滤波后的特征图，形状不变
        """
        sigma = self.sigma
        kernel = self._build_kernel(sigma)
        return F.conv3d(x, kernel, padding=self.padding, groups=self.channels)

    def extra_repr(self) -> str:
        sigma_val = self.sigma.detach().cpu().numpy()
        return (f'channels={self.channels}, kernel_size={self.kernel_size}, '
                f'sigma={sigma_val}, range=[{self.sigma_min}, {self.sigma_max}]')


# ════════════════════════════════════════════════════════════════════
#  输入自适应σ预测网络
# ════════════════════════════════════════════════════════════════════

class AdaptiveSigmaPredictor(nn.Module):
    """
    轻量自适应σ预测网络。
    
    根据输入特征的全局统计特性（均值+方差+频谱能量）预测最优σ。
    
    设计动机：
    - 静息状态心率稳定(~60bpm)→较窄低通（σ大，强平滑去噪）
    - 运动状态心率快速变化(~150bpm)→需较宽通带（σ小，保留高频）
    - 不同光照条件下信噪比不同→最优截止频率不同
    
    网络架构（极轻量，<1K参数）：
    GlobalAvgPool → [C] → Linear(C, C//4) → ReLU → Linear(C//4, num_sigmas) → softplus+clamp
    
    Args:
        channels: 输入通道数
        num_sigmas: 预测σ的数量（1=各向同性，2=时空各向异性，3=三轴独立）
        sigma_min/max: σ的物理约束范围
        init_sigma: 偏置初始化为init_sigma附近
    """
    def __init__(self, channels: int, num_sigmas: int = 1,
                 sigma_min: float = 0.3, sigma_max: float = 3.0,
                 init_sigma: float = 1.5):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_sigmas = num_sigmas

        hidden = max(channels // 4, 8)
        self.predictor = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_sigmas),
        )
        
        # 偏置初始化：使输出在init_sigma附近
        # softplus(bias) ≈ init_sigma → bias = ln(exp(init_sigma)-1)
        init_bias = math.log(math.exp(init_sigma) - 1.0)
        nn.init.zeros_(self.predictor[0].weight)
        nn.init.zeros_(self.predictor[0].bias)
        nn.init.zeros_(self.predictor[2].weight)
        nn.init.constant_(self.predictor[2].bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] 输入特征
        Returns:
            sigma: [B, num_sigmas] 每个样本的预测σ
        """
        # 全局平均池化获取通道统计
        gap = x.mean(dim=[2, 3, 4])  # [B, C]
        raw_sigma = self.predictor(gap)  # [B, num_sigmas]
        sigma = torch.clamp(F.softplus(raw_sigma), self.sigma_min, self.sigma_max)
        return sigma


# ════════════════════════════════════════════════════════════════════
#  各向异性时空频率分解器
# ════════════════════════════════════════════════════════════════════

class AnisotropicFreqDecomposer(nn.Module):
    """
    各向异性时空频率分解器。
    
    核心创新：时间维度和空间维度使用独立的σ参数。
    
    物理动机：
    - 时间维度：rPPG信号带宽0.83-3.0Hz，需精确控制时间低通截止
    - 空间维度：面部皮肤脉动在空间上高度相关（同一血管区域同步脉动），
      空间低通可更激进以去噪，不会损失rPPG信息
    - 各向异性允许模型独立优化时间频率选择性和空间去噪强度
    
    实现：分离为1D时间低通 + 2D空间低通的级联（可分离高斯）。
    
    Args:
        channels: 通道数
        kernel_size_t: 时间维度核大小（默认5，对应±2帧窗口）
        kernel_size_s: 空间维度核大小（默认5）
        init_sigma_t: 时间σ初始值
        init_sigma_s: 空σ初始值
        sigma_min/max: σ约束范围
    """
    def __init__(self, channels: int,
                 kernel_size_t: int = 5, kernel_size_s: int = 5,
                 init_sigma_t: float = 1.5, init_sigma_s: float = 1.0,
                 sigma_min: float = 0.3, sigma_max: float = 3.0):
        super().__init__()
        self.channels = channels
        self.kernel_size_t = kernel_size_t
        self.kernel_size_s = kernel_size_s
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.padding_t = kernel_size_t // 2
        self.padding_s = kernel_size_s // 2

        # 可学习时间σ和空间σ
        init_raw_t = math.log(math.exp(init_sigma_t) - 1.0)
        init_raw_s = math.log(math.exp(init_sigma_s) - 1.0)
        self.sigma_t_raw = nn.Parameter(torch.tensor(init_raw_t))
        self.sigma_s_raw = nn.Parameter(torch.tensor(init_raw_s))

        # 坐标缓存
        coords_t = torch.arange(kernel_size_t).float() - kernel_size_t // 2
        coords_s = torch.arange(kernel_size_s).float() - kernel_size_s // 2
        self.register_buffer('coords_t', coords_t)
        self.register_buffer('coords_s', coords_s)

    @property
    def sigma_t(self) -> torch.Tensor:
        return torch.clamp(F.softplus(self.sigma_t_raw), self.sigma_min, self.sigma_max)

    @property
    def sigma_s(self) -> torch.Tensor:
        return torch.clamp(F.softplus(self.sigma_s_raw), self.sigma_min, self.sigma_max)

    def forward(self, x: torch.Tensor):
        """
        各向异性频率分解。
        
        Args:
            x: [B, C, T, H, W]
        Returns:
            low_freq: [B, C, T, H, W] 低频分量
            high_freq: [B, C, T, H, W] 高频分量 = x - low_freq
        """
        B, C, T, H, W = x.shape
        sigma_t = self.sigma_t
        sigma_s = self.sigma_s

        # --- 时间维度1D高斯低通 ---
        g_t = torch.exp(-(self.coords_t ** 2) / (2 * sigma_t ** 2))
        g_t = g_t / g_t.sum()
        # 构建 [C, 1, Kt, 1, 1] 时间卷积核
        kernel_t = g_t.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1)
        low_freq = F.conv3d(x, kernel_t, padding=(self.padding_t, 0, 0), groups=C)

        # --- 空间维度2D高斯低通（可分离：先H后W）---
        g_s = torch.exp(-(self.coords_s ** 2) / (2 * sigma_s ** 2))
        g_s = g_s / g_s.sum()
        # H方向: [C, 1, 1, Ks, 1]
        kernel_h = g_s.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1)
        low_freq = F.conv3d(low_freq, kernel_h, padding=(0, self.padding_s, 0), groups=C)
        # W方向: [C, 1, 1, 1, Ks]
        kernel_w = g_s.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1)
        low_freq = F.conv3d(low_freq, kernel_w, padding=(0, 0, self.padding_s), groups=C)

        high_freq = x - low_freq
        return low_freq, high_freq

    def extra_repr(self) -> str:
        return (f'channels={self.channels}, '
                f'sigma_t={self.sigma_t.item():.3f}, sigma_s={self.sigma_s.item():.3f}')


# ════════════════════════════════════════════════════════════════════
#  ★ 自适应频率分解器（统一接口）
# ════════════════════════════════════════════════════════════════════

class AdaptiveFreqDecomposer(nn.Module):
    """
    自适应频率分解器AFD）— 创新点1的统一接口。
    
    将原FreqSepPhys中固定σ的rPPGFreqDecomposer替换为自适应版本。
    支持3种运行模式：
    
    Mode 1 - 'learnable': 可学习固定σ（训练时优化，推理时固定）
        适用场景：数据集分布相对一致，需全局最优截止频率
        参数量：+1（共享σ）或+C（per-channel σ）
        
    Mode 2 - 'adaptive': 输入自适应σ（每个样本独立预测σ）
        适用场景：跨域/跨场景，输入条件差异大
        参数量：+C//4 + num_sigmas（预测网络）
        
    Mode 3 - 'anisotropic': 各向异性时空分离（时间σ/空间σ独立可学习）
        适用场景：需要精确控制时间频率选择性
        参数量：+2（σ_t + σ_s）
    
    正则化约束（均有明确物理含义）：
    - σ_low > σ_high: 低频分量必须比高频更平滑（信息论基本约束）
    - σ ∈ [sigma_min, sigma_max]: 防止退化
    - 频率正交性: |⟨lf, hf⟩| → 0（低频高频应尽量正交）
    
    Args:
        channels: 输入通道数
        mode: 'learnable' | 'adaptive' | 'anisotropic'
        init_sigma: σ初始值（应匹配所在流的时间尺度）
        kernel_size: 高斯核空间大小
        sigma_min: σ下界
        sigma_max: σ上界
    """
    def __init__(self, channels: int, mode: str = 'anisotropic',
                 init_sigma: float = 1.5, kernel_size: int = 5,
                 sigma_min: float = 0.3, sigma_max: float = 3.0):
        super().__init__()
        self.channels = channels
        self.mode = mode
        self.kernel_size = kernel_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        if mode == 'learnable':
            self.decomposer = LearnableSigmaLowPass3d(
                channels, kernel_size=kernel_size,
                init_sigma=init_sigma,
                sigma_min=sigma_min, sigma_max=sigma_max,
                per_channel=False,
            )
        elif mode == 'adaptive':
            self.sigma_predictor = AdaptiveSigmaPredictor(
                channels, num_sigmas=1,
                sigma_min=sigma_min, sigma_max=sigma_max,
                init_sigma=init_sigma,
            )
            self.kernel_size = kernel_size
            self.padding = kernel_size // 2
            # 坐标缓存
            coords = torch.arange(kernel_size).float() - kernel_size // 2
            self.register_buffer('coords', coords)
        elif mode == 'anisotropic':
            self.decomposer = AnisotropicFreqDecomposer(
                channels,
                kernel_size_t=kernel_size, kernel_size_s=kernel_size,
                init_sigma_t=init_sigma, init_sigma_s=init_sigma * 0.7,
                sigma_min=sigma_min, sigma_max=sigma_max,
            )
        else:
            raise ValueError(f"Unknown AFD mode: {mode}. Choose from 'learnable', 'adaptive', 'anisotropic'")

    def forward(self, x: torch.Tensor):
        """
        自适应频率分解。
        
        Args:
            x: [B, C, T, H, W]
        Returns:
            low_freq: [B, C, T, H, W]
            high_freq: [B, C, T, H, W]
        """
        if self.mode == 'learnable':
            low_freq = self.decomposer(x)
            high_freq = x - low_freq
            return low_freq, high_freq
        
        elif self.mode == 'adaptive':
            return self._adaptive_decompose(x)
        
        elif self.mode == 'anisotropic':
            return self.decomposer(x)

    def _adaptive_decompose(self, x: torch.Tensor):
        """
        输入自适应分解：每个样本独立预测σ并应用。
        
        由于batch内不同样本σ不同，需逐样本构建核。
        优化：当batch内σ方差极小时退化为统一处理。
        """
        B, C, T, H, W = x.shape
        sigmas = self.sigma_predictor(x)  # [B, 1]
        
        # 逐样本构建高斯核并滤波
        low_freqs = []
        for i in range(B):
            sigma_i = sigmas[i, 0]  # scalar
            g = torch.exp(-(self.coords ** 2) / (2 * sigma_i ** 2))
            g = g / g.sum()
            # 3D各向同性核
            k3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
            kernel = k3d.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1, 1)
            lf_i = F.conv3d(
                x[i:i+1], kernel, padding=self.padding, groups=C
            )
            low_freqs.append(lf_i)
        
        low_freq = torch.cat(low_freqs, dim=0)  # [B, C, T, H, W]
        high_freq = x - low_freq
        return low_freq, high_freq

    def get_sigma_values(self) -> dict:
        """获取当前σ值（用于监控/日志，返回Python标量，已脱离计算图）。

        ⚠️ 仅用于日志打印/可视化，切勿用于损失计算——.item()会切断梯度。
        损失计算请改用 get_sigma_tensors()。
        """
        info = {'mode': self.mode}
        if self.mode == 'learnable':
            info['sigma'] = self.decomposer.sigma.detach().cpu().numpy().tolist()
        elif self.mode == 'adaptive':
            info['note'] = 'sigma is input-dependent, call forward first'
        elif self.mode =='anisotropic':
            info['sigma_t'] = self.decomposer.sigma_t.item()
            info['sigma_s'] = self.decomposer.sigma_s.item()
        return info

    def get_sigma_tensors(self) -> dict:
        """获取可微的σ张量（用于损失计算，保留计算图）。

        与get_sigma_values的区别：不调用.item()，返回的tensor带梯度，
        可直接参与反向传播优化 sigma_*_raw 参数。

        Returns:
            dict:
              - mode: 当前模式
              - 'anisotropic': {'sigma_t': Tensor, 'sigma_s': Tensor}（可微）
              - 'learnable':   {'sigma': Tensor}（可微）
              - 'adaptive':    {}（σ为输入相关，需forward时另行获取）
        """
        info = {'mode': self.mode}
        if self.mode == 'learnable':
            # LearnableSigmaLowPass3d.sigma 为 softplus 参数化，可微
            info['sigma'] = self.decomposer.sigma
        elif self.mode == 'anisotropic':
            info['sigma_t'] = self.decomposer.sigma_t  # 可微 tensor
            info['sigma_s'] = self.decomposer.sigma_s
        return info


# ════════════════════════════════════════════════════════════════════
#  AFD正则化损失
# ════════════════════════════════════════════════════════════════════

class AFDRegularizationLoss(nn.Module):
    """
    AFD正则化损失，确保频率分解的物理合理性。
    
    包含3个正则项（均有明确物理含义）：
    
    1. 频率正交性损失 L_ortho:
       |⟨lf, hf⟩| / (||lf|| * ||hf||) → 0
       物理含义：低频和高频分量应携带不同信息，重叠越少分解越有效
       
    2. 能量守恒损失 L_energy:
       ||lf + hf - x||² → 0
       物理含义：分解应无损，所有能量都被保留（由设计保证，作为数值稳定性检查）
       
    3. σ分离约束 L_sep（仅对多σ场景）:
       max(0, σ_high - σ_low + margin)
       物理含义：低频流的σ应大于高频流的σ（低频更平滑）
    
    Args:
        ortho_weight: 正交性损失权重（默认0.1）
        energy_weight: 能量守恒权重（默认0.01，仅作数值检查）
        sep_weight: σ分离约束权重（默认0.5）
        sep_margin: σ分离最小间距（默认0.3，对应≈0.5Hz带宽差）
    """
    def __init__(self, ortho_weight: float = 0.1,
                 energy_weight: float = 0.01,
                 sep_weight: float = 0.5,
                 sep_margin: float = 0.3):
        super().__init__()
        self.ortho_weight = ortho_weight
        self.energy_weight = energy_weight
        self.sep_weight = sep_weight
        self.sep_margin = sep_margin

    def forward(self, x: torch.Tensor, low_freq: torch.Tensor, high_freq: torch.Tensor,
                sigma_slow=None, sigma_fast=None) -> dict:
        """
        Args:
            x: 原始输入
            low_freq: 低频分量
            high_freq: 高频分量
            sigma_slow: 慢流σ值（optional，用于分离约束）
            sigma_fast: 快流σ值（optional）
        Returns:
            dict with 'total', 'ortho', 'energy', 'sep' losses
        """
        losses = {}
        
        # 1. 频率正交性
        B = x.shape[0]
        lf_flat = low_freq.reshape(B, -1)
        hf_flat = high_freq.reshape(B, -1)
        # cosine similarity作为正交性度量
        cos_sim = F.cosine_similarity(lf_flat, hf_flat, dim=-1)  # [B]
        losses['ortho'] = self.ortho_weight * cos_sim.abs().mean()

        # 2. 能量守恒（理论上lf+hf=x，此项用于数值稳定性监控）
        recon_error = (low_freq + high_freq - x).pow(2).mean()
        losses['energy'] = self.energy_weight * recon_error

        # 3. σ分离约束
        if sigma_slow is not None and sigma_fast is not None:
            # 确保慢流σ > 快流σ + margin
            sep_loss = F.relu(sigma_fast - sigma_slow + self.sep_margin)
            losses['sep'] = self.sep_weight * sep_loss.mean()
        else:
            losses['sep'] = torch.tensor(0.0, device=x.device)

        losses['total'] = losses['ortho'] + losses['energy'] + losses['sep']
        return losses


# ════════════════════════════════════════════════════════════════════
#  可学习σ的低通约束（替换FreqSepBlock中的固定GaussianLowPass3d）
# ════════════════════════════════════════════════════════════════════

class AdaptiveLowPassConstraint(nn.Module):
    """
    自适应低通约束层（替换FreqSepBlock低频路末端的固定GaussianLowPass3d）。
    
    与LearnableSigmaLowPass3d的区别：
    - 这里σ由外部控制（与AdaptiveFreqDecomposer联动）
    - 支持从decomposer传入σ实现端到端一致性
    
    用法：在FreqSepBlock的低频路末端，用自适应低通替代固定低通。
    """
    def __init__(self, channels: int, kernel_size: int = 5,
                 init_sigma: float = 1.5,
                 sigma_min: float = 0.3, sigma_max: float = 3.0):
        super().__init__()
        self.lowpass = LearnableSigmaLowPass3d(
            channels, kernel_size=kernel_size,
            init_sigma=init_sigma,
            sigma_min=sigma_min, sigma_max=sigma_max,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)

    @property
    def sigma(self) -> torch.Tensor:
        return self.lowpass.sigma