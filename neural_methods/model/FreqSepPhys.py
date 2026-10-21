"""
freqsepphys.py
==============
FreqSepPhys — Frequency-Separated rPPG Network

核心贡献（研究动机A：频率分离感知）：

rPPG 信号是皮肤表面的微弱光学脉动（0.5~3 Hz），
而运动伪影和光照漂移主要表现为低频趋势干扰。
现有方法对慢流/快流使用统一处理策略，未能显式区分信号频率成分。

本文在双流架构每个处理块内引入：
1. rPPGFreqDecomposer
   显式分解：低频分量 = LowPass(x)，高频分量 = x - 低频
   低频 → 光照漂移/运动背景（干扰），高频 → 皮肤脉动微变色（rPPG信号载体）

2. InvertibleFreqExchange
   可逆跨频率信息交换：
   低频列 ← 高频信息 / 高频列 ← 低频信息（加法耦合，精确可逆）
   实现频率间信息无损双向流动

3. FreqSepBlock（替换原 _build_block）
   ┌─ 分解 ────────────────────────────────────────────────────┐
   │  x → lf = LowPass(x)                                     │
   │    → hf = x - lf                                         │
   ├─ 双路 CDC_T + BiMambaEncoder + FreqChannelGating ─────────┤
   │  lf_path: CDC_T → BN+ReLU → BiMamba → Gating → LowPass  │  ← 慢流大σ / 快流小σ
   │  hf_path: CDC_T → BN+ReLU → BiMamba → Gating            │  ← 无额外平滑，保留rPPG细节
   ├─ InvertibleFreqExchange ──────────────────────────────────┤
   │  lf ← lf + F(hf),   hf ← hf + G(lf)                    │
   └─ 残差求和输出 lf + hf ────────────────────────────────────┘

4. 尺度自适应截止频率（SLOW/FAST_STREAM_SIGMA）
   慢流（时间压缩×4，粗尺度）：sigma=1.8 → 强低通，净化低频背景漂移
   快流（时间压缩×2，细尺度）：sigma=0.8 → 轻低通，保留rPPG高频脉动细节
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from mamba_ssm import Mamba
from torch.nn import functional as F


# ════════════════════════════════════════════════════════════════════
#  低通滤波基础组件
# ════════════════════════════════════════════════════════════════════

class GaussianLowPass3d(nn.Module):
    """
    固定（不可学习）各向同性 3D 高斯低通滤波器（depthwise）。
    register_buffer 注册，不参与梯度，零额外参数。
    用于在时空域施加频率截止约束，抑制高频噪声或低频漂移。
    """
    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 1.5):
        super().__init__()
        self.channels = channels
        pad = kernel_size // 2
        coords = torch.arange(kernel_size).float() - pad
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        k3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        k3d = k3d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1, 1)
        self.register_buffer('kernel', k3d)
        self.padding = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self.kernel, padding=self.padding, groups=self.channels)


class LowPassRegressionHead(nn.Module):
    """输出回归头：Conv → 输出 → 低通约束（硬约束，防止高频毛刺）"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, smooth_sigma: float = 1.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.1),
            nn.Conv3d(32, out_channels, kernel_size, padding=kernel_size // 2),
        )
        self.lowpass = GaussianLowPass3d(out_channels, kernel_size=9, sigma=smooth_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(self.conv(x))


class LowPassUpsampleBlock(nn.Module):
    """上采样精化块：Upsample → Conv3d → BN → ELU → 低通约束"""
    def __init__(self, in_channels: int, out_channels: int, smooth_sigma: float = 1.0):
        super().__init__()
        self.up      = nn.Upsample(scale_factor=(2, 1, 1))
        self.conv    = nn.Conv3d(in_channels, out_channels,
                                 kernel_size=[3, 1, 1], stride=1, padding=(1, 0, 0))
        self.bn      = nn.BatchNorm3d(out_channels)
        self.act     = nn.ELU(inplace=True)
        self.lowpass = GaussianLowPass3d(out_channels, kernel_size=5, sigma=smooth_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(self.act(self.bn(self.conv(self.up(x)))))


class LowPassConvBlock(nn.Module):
    """精化卷积块：Conv3d → BN → 激活 → 低通约束（快流末端抗混叠）"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride=1, padding=0,
                 activation: str = 'elu', smooth_sigma: float = 0.8):
        super().__init__()
        self.conv    = nn.Conv3d(in_channels, out_channels,
                                 kernel_size, stride=stride, padding=padding)
        self.bn      = nn.BatchNorm3d(out_channels)
        self.act     = nn.ELU(inplace=True) if activation == 'elu' else nn.ReLU(inplace=True)
        self.lowpass = GaussianLowPass3d(out_channels, kernel_size=3, sigma=smooth_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(self.act(self.bn(self.conv(x))))


# ════════════════════════════════════════════════════════════════════
#  ★ 频率分离核心组件
# ════════════════════════════════════════════════════════════════════

class rPPGFreqDecomposer(nn.Module):
    """
    rPPG 频率分解器

    将输入特征显式分离为两个频率分量：
      low_freq  = GaussianLowPass(x)   ← 低频：光照漂移、慢速运动背景（干扰）
      high_freq = x - low_freq         ← 高频：皮肤微血管脉动色彩变化（rPPG信号载体）

    sigma 与所在流的时间尺度匹配：
      慢流（粗时间尺度）→ 大 sigma（1.8），强低通，净化运动伪影
      快流（细时间尺度）→ 小 sigma（0.8），轻低通，保留 rPPG 高频细节
    """
    def __init__(self, channels: int, sigma: float = 1.5, kernel_size: int = 5):
        super().__init__()
        self.lowpass_filter = GaussianLowPass3d(channels, kernel_size=kernel_size, sigma=sigma)

    def forward(self, x: torch.Tensor):
        low_freq  = self.lowpass_filter(x)   # 低频分量（干扰载体）
        high_freq = x - low_freq              # 高频分量（rPPG信号载体）
        return low_freq, high_freq


class InvertibleFreqExchange(nn.Module):
    """
    可逆频率信息交换层

    在低频分量和高频分量之间实现双向无损信息流动：

    正向（低频列 ← 高频信息 / 高频列 ← 低频信息）：
        y_lf = x_lf + F_net(x_hf)    ← 高频rPPG信息增强低频判别力
        y_hf = x_hf + G_net(y_lf)    ← 低频背景上下文稳定高频估计

    精确逆向（用于一致性验证）：
        x_hf = y_hf - G_net(y_lf)
        x_lf = y_lf - F_net(x_hf)

    轻量 1×1 投影，无空间结构改变，加法耦合保证信息无损。
    """
    def __init__(self, channels: int):
        super().__init__()
        # F_net: 高频信息流入低频列
        self.F_net = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, 1, bias=False),
        )
        # G_net: 低频信息流入高频列
        self.G_net = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, 1, bias=False),
        )

    def forward(self, x_lf: torch.Tensor, x_hf: torch.Tensor):
        """正向：双向频率信息交换"""
        y_lf = x_lf + self.F_net(x_hf)   # 高频信息注入低频列
        y_hf = x_hf + self.G_net(y_lf)   # 低频信息（更新后）注入高频列
        return y_lf, y_hf

    def inverse(self, y_lf: torch.Tensor, y_hf: torch.Tensor):
        """精确逆向（理论误差 ≈ 机器精度）"""
        x_hf = y_hf - self.G_net(y_lf)
        x_lf = y_lf - self.F_net(x_hf)
        return x_lf, x_hf


# ════════════════════════════════════════════════════════════════════
#  序列建模与注意力组件
# ════════════════════════════════════════════════════════════════════

class BiMambaEncoder(nn.Module):
    """
    双向 Mamba 编码器（参考 Vim / Vision Mamba, 2024）：
    前向扫描 + 后向扫描（flip→Mamba→flip），可学习加权融合。
    分别作用于低频路和高频路，独立捕捉各频段的时序依赖。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        self.dim    = dim
        self.norm1  = nn.LayerNorm(dim)
        self.norm2  = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.alpha     = nn.Parameter(torch.ones(1))
        self.beta      = nn.Parameter(torch.ones(1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_patch_token(self, x):
        B, d_model, *img_dims = x.shape
        n_tokens = 1
        for s in img_dims:
            n_tokens *= s
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm1(x_flat)
        x_fwd  = self.mamba_fwd(x_norm)
        x_bwd  = self.mamba_bwd(x_norm.flip(1)).flip(1)
        x_bi   = self.alpha * x_fwd + self.beta * x_bwd
        x_out  = self.norm2(x_flat + self.drop_path(x_bi))
        return x_out.transpose(-1, -2).reshape(B, d_model, *img_dims)

    def forward(self, x):
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
        return self.forward_patch_token(x)


class FreqChannelGating(nn.Module):
    """
    频率通道门控注意力（Channel Attention）
    分别作用于低频路和高频路，自适应地强化各频段中的判别性通道。
    低频路：突出与光照/运动相关的通道（用于抑制）
    高频路：突出与脉动相关的通道（用于增强 rPPG 信号）
    """
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class CrossStreamFusion(nn.Module):
    """
    跨流融合层（慢流 ← 快流）
    通过时间步长卷积将快流特征对齐到慢流时间尺度，
    实现细时间尺度高频信息对粗时间尺度慢流的补充。
    """
    def __init__(self, fast_channels=32, slow_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, [3, 1, 1],
                      stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(),
        )

    def forward(self, slow_path, fast_path):
        return self.conv(fast_path) + slow_path


class CDC_T(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.2):
        super().__init__()
        self.conv  = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation,
                               groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta) < 1e-8 or self.conv.weight.shape[2] <= 1:
            return out_normal
        kernel_diff = (
            self.conv.weight[:, :, 0, :, :].sum(2).sum(2) +
            self.conv.weight[:, :, 2, :, :].sum(2).sum(2)
        )[:, :, None, None, None]
        out_diff = F.conv3d(x, kernel_diff, bias=self.conv.bias,
                            stride=self.conv.stride, padding=0,
                            dilation=self.conv.dilation, groups=self.conv.groups)
        return out_normal - self.theta * out_diff


def conv_block(in_channels, out_channels, kernel_size, stride, padding,
               bn=True, activation='relu'):
    layers = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
    if bn:
        layers.append(nn.BatchNorm3d(out_channels))
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'elu':
        layers.append(nn.ELU(inplace=True))
    return nn.Sequential(*layers)


# ════════════════════════════════════════════════════════════════════
#  ★ 核心处理块：频率分离处理块
# ════════════════════════════════════════════════════════════════════

class FreqSepBlock(nn.Module):
    """
    频率分离处理块（FreqSepPhys 的核心模块）

    研究动机实现：
      rPPG 有效信号（高频）和干扰（低频漂移）在特征空间中混叠，
      统一处理会导致模型无法区分信号与噪声。
      FreqSepBlock 在每个处理块内显式分离两类频率成分，分别建模后再交换信息。

    内部流程：
    ┌─ 输入 x ─────────────────────────────────────────────────────┐
    │                                                              │
    │  ① rPPGFreqDecomposer → low_freq, high_freq                │
    │                                                              │
    │  ② 低频路（光照/运动背景）：                                 │
    │     CDC_T → BN+ReLU → BiMambaEncoder → FreqChannelGating    │
    │     → GaussianLowPass（σ=STREAM_SIGMA）                     │
    │     慢流 σ=1.8：强低通，净化低频漂移趋势                    │
    │     快流 σ=0.8：轻低通，避免削弱中低频 rPPG 能量            │
    │                                                              │
    │  ③ 高频路（rPPG 脉动信号）：                                │
    │     CDC_T → BN+ReLU → BiMambaEncoder → FreqChannelGating    │
    │     （无额外平滑，完整保留皮肤微变色高频细节）               │
    │                                                              │
    │  ④ InvertibleFreqExchange                                   │
    │     lf ← lf + F(hf)   高频rPPG信息增强低频判别力           │
    │     hf ← hf + G(lf)   低频上下文稳定高频信号估计            │
    │                                                              │
    │  ⑤ 输出 = low_freq + high_freq（残差相加）                  │
    └──────────────────────────────────────────────────────────────┘
    """
    def __init__(self, channels: int, theta: float,
                 stream_sigma: float = 1.5,
                 lf_kernel_size: int = 5,
                 decomp_cutoff: float = 1.5):
        super().__init__()
        # ── 频率分解 ─────────────────────────────────────────
        self.freq_decomposer = rPPGFreqDecomposer(
            channels, sigma=decomp_cutoff, kernel_size=lf_kernel_size
        )

        # ── 低频路（干扰建模路）──────────────────────────────
        self.cdc_lf    = CDC_T(channels, channels, theta=theta)
        self.bn_lf     = nn.BatchNorm3d(channels)
        self.relu_lf   = nn.ReLU(inplace=True)
        self.mamba_lf  = BiMambaEncoder(dim=channels)
        self.gating_lf = FreqChannelGating(in_channels=channels, reduction=2)
        # 低频路输出施加低通约束（σ 由流的时间尺度决定）
        self.lowpass_lf = GaussianLowPass3d(channels, kernel_size=lf_kernel_size, sigma=stream_sigma)

        # ── 高频路（rPPG 信号建模路）────────────────────────
        self.cdc_hf    = CDC_T(channels, channels, theta=theta)
        self.bn_hf     = nn.BatchNorm3d(channels)
        self.relu_hf   = nn.ReLU(inplace=True)
        self.mamba_hf  = BiMambaEncoder(dim=channels)
        self.gating_hf = FreqChannelGating(in_channels=channels, reduction=2)
        # 高频路：不施加低通，完整保留 rPPG 脉动高频

        # ── 可逆频率信息交换 ─────────────────────────────────
        self.freq_exchange = InvertibleFreqExchange(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ① 频率分解
        low_freq, high_freq = self.freq_decomposer(x)

        # ② 低频路处理（光照/运动背景）
        low_freq = self.relu_lf(self.bn_lf(self.cdc_lf(low_freq)))
        low_freq = self.mamba_lf(low_freq)
        low_freq = self.gating_lf(low_freq)
        low_freq = self.lowpass_lf(low_freq)       # 低通约束（硬约束）

        # ③ 高频路处理（rPPG 脉动信号）
        high_freq = self.relu_hf(self.bn_hf(self.cdc_hf(high_freq)))
        high_freq = self.mamba_hf(high_freq)
        high_freq = self.gating_hf(high_freq)
        # 无低通：保留 rPPG 微弱色彩高频

        # ④ 可逆频率信息交换
        low_freq, high_freq = self.freq_exchange(low_freq, high_freq)

        # ⑤ 残差相加输出
        return low_freq + high_freq


# ════════════════════════════════════════════════════════════════════
#  主模型：FreqSepPhys
# ════════════════════════════════════════════════════════════════════

class FreqSepPhys(nn.Module):
    """
    FreqSepPhys: Frequency-Separated rPPG Network

    研究动机：
      rPPG 信号（0.5~3 Hz）是皮肤表面微弱光学脉动，
      运动伪影和光照漂移主要表现为低频趋势干扰。
      现有双流方法对慢流/快流使用统一特征处理，
      未能显式区分信号频率成分，导致跨域泛化能力不足。

    核心设计：
      在双流（慢流×4 / 快流×2 时间压缩）每个处理块内：
      ① 显式高低频分解（rPPGFreqDecomposer）
      ② 双路独立建模（低频路/高频路，各含 CDC_T + BiMambaEncoder + FreqChannelGating）
      ③ 尺度自适应截止频率（慢流 σ=1.8 强低通 / 快流 σ=0.8 轻低通）
      ④ 可逆频率信息交换（InvertibleFreqExchange，无损双向耦合）

    与原 EfficientPhys 的差异：
    ┌──────────────────┬──────────────────┬──────────────────────────────────┐
    │ 模块             │ 原版             │ FreqSepPhys                      │
    ├──────────────────┼──────────────────┼──────────────────────────────────┤
    │ 慢流 Block1~3    │ 单路 CDC+Mamba   │ FreqSepBlock(σ=1.8)              │
    │ 快流 Block4~6    │ 单路 CDC+Mamba   │ FreqSepBlock(σ=0.8)              │
    │ Block 内部       │ 统一处理         │ lf路+hf路+InvertibleFreqExchange │
    │ ConvBlock6       │ ELU Conv         │ LowPassConvBlock(σ=0.8)          │
    │ upsample1/2      │ nn.Sequential    │ LowPassUpsampleBlock             │
    │ 输出头           │ nn.Conv3d        │ LowPassRegressionHead            │
    └──────────────────┴──────────────────┴──────────────────────────────────┘
    """

    # 尺度自适应截止频率（σ 越大，低通越强）
    SLOW_STREAM_SIGMA  = 1.8   # 慢流低频路输出截止（强低通，净化背景漂移）
    FAST_STREAM_SIGMA  = 0.8   # 快流低频路输出截止（轻低通，保留rPPG细节）
    SLOW_DECOMP_CUTOFF = 1.8   # 慢流分解边界（高低频分割点）
    FAST_DECOMP_CUTOFF = 0.8   # 快流分解边界

    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5, frames=128):
        super().__init__()

        # ── 共享特征提取（保持不变）────────────────────────────
        self.ConvBlock1 = conv_block(3,  16, [1, 5, 5], stride=1, padding=[0, 2, 2])
        self.ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)

        # ── 快流末端精化（低通约束）────────────────────────────
        self.ConvBlock6 = LowPassConvBlock(
            32, 32, kernel_size=[3, 1, 1], stride=1,
            padding=[1, 0, 0], activation='elu', smooth_sigma=0.8,
        )

        # ── 慢流：FreqSepBlock（强低通 σ=1.8）─────────────────
        # 时间压缩×4，粗时间尺度，需强低通净化背景漂移
        self.Block1 = self._build_slow_freqsep_block(64, theta)
        self.Block2 = self._build_slow_freqsep_block(64, theta)
        self.Block3 = self._build_slow_freqsep_block(64, theta)

        # ── 快流：FreqSepBlock（轻低通 σ=0.8）─────────────────
        # 时间压缩×2，细时间尺度，轻低通保留 rPPG 高频脉动
        self.Block4 = self._build_fast_freqsep_block(32, theta)
        self.Block5 = self._build_fast_freqsep_block(32, theta)
        self.Block6 = self._build_fast_freqsep_block(32, theta)

        # ── 上采样精化（低通约束）──────────────────────────────
        self.upsample1 = LowPassUpsampleBlock(64, 64, smooth_sigma=1.0)
        self.upsample2 = LowPassUpsampleBlock(96, 48, smooth_sigma=1.0)

        # ── 输出回归头（低通约束）──────────────────────────────
        self.ConvBlockLast = LowPassRegressionHead(
            in_channels=48, out_channels=1,
            kernel_size=9, smooth_sigma=1.5,
        )

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.cross_fuse_1 = CrossStreamFusion(32, 64)
        self.cross_fuse_2 = CrossStreamFusion(32, 64)
        self.drop_1 = nn.Dropout(drop_rate1)
        self.drop_2 = nn.Dropout(drop_rate1)
        self.drop_3 = nn.Dropout(drop_rate2)
        self.drop_4 = nn.Dropout(drop_rate2)
        self.drop_5 = nn.Dropout(drop_rate2)
        self.drop_6 = nn.Dropout(drop_rate2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def _build_slow_freqsep_block(self, channels: int, theta: float) -> FreqSepBlock:
        """
        慢流频率分离块：强低通（σ=1.8）
        粗时间尺度需强低通净化宏观运动漂移和光照变化。
        """
        return FreqSepBlock(
            channels=channels, theta=theta,
            stream_sigma=self.SLOW_STREAM_SIGMA,
            decomp_cutoff=self.SLOW_DECOMP_CUTOFF,
        )

    def _build_fast_freqsep_block(self, channels: int, theta: float) -> FreqSepBlock:
        """
        快流频率分离块：轻低通（σ=0.8）
        细时间尺度只做轻微低通，避免削弱 rPPG 脉动高频能量。
        """
        return FreqSepBlock(
            channels=channels, theta=theta,
            stream_sigma=self.FAST_STREAM_SIGMA,
            decomp_cutoff=self.FAST_DECOMP_CUTOFF,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 共享特征提取 ──────────────────────────────────────
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpa(x)

        # ── 双流分叉 ──────────────────────────────────────────
        s_x = self.ConvBlock4(x)    # 慢流：时间压缩×4，64ch
        f_x = self.ConvBlock5(x)    # 快流：时间压缩×2，32ch

        # ── 第一组：频率分离处理 + 空间下采样 + 跨流融合 ──────
        s_x1 = self.drop_1(self.MaxpoolSpa(self.Block1(s_x)))
        f_x1 = self.drop_2(self.MaxpoolSpa(self.Block4(f_x)))
        s_x1 = self.cross_fuse_1(s_x1, f_x1)   # 快流高频细节补充慢流

        # ── 第二组 ────────────────────────────────────────────
        s_x2 = self.drop_3(self.MaxpoolSpa(self.Block2(s_x1)))
        f_x2 = self.drop_4(self.MaxpoolSpa(self.Block5(f_x1)))
        s_x2 = self.cross_fuse_2(s_x2, f_x2)

        # ── 第三组 + 各流精化 ─────────────────────────────────
        s_x3 = self.drop_5(self.upsample1(self.Block3(s_x2)))
        f_x3 = self.drop_6(self.ConvBlock6(self.Block6(f_x2)))

        # ── 融合 → 上采样精化 → 全局池化 → 输出 ──────────────
        x_final = self.upsample2(torch.cat((f_x3, s_x3), dim=1))
        x_final = self.ConvBlockLast(self.poolspa(x_final))
        return x_final.view(-1, x_final.shape[2])

    @torch.no_grad()
    def verify_freq_exchange_invertibility(self, x: torch.Tensor, block_idx: int = 0) -> dict:
        """
        验证指定 FreqSepBlock 的 InvertibleFreqExchange 可逆性。
        理论误差应接近机器精度（~1e-6）。

        Args:
            x: 任意中间特征图 [B, C, T, H, W]
            block_idx: 0~5，对应 Block1~Block6
        """
        block = [self.Block1, self.Block2, self.Block3,
                 self.Block4, self.Block5, self.Block6][block_idx]
        lf, hf = block.freq_decomposer(x)
        lf_proc = block.lowpass_lf(block.gating_lf(block.mamba_lf(
            block.relu_lf(block.bn_lf(block.cdc_lf(lf))))))
        hf_proc = block.gating_hf(block.mamba_hf(
            block.relu_hf(block.bn_hf(block.cdc_hf(hf)))))
        y_lf, y_hf = block.freq_exchange(lf_proc, hf_proc)
        r_lf, r_hf = block.freq_exchange.inverse(y_lf, y_hf)
        return {
            'block': f'Block{block_idx + 1}',
            'lf_reconstruction_error': (r_lf - lf_proc).abs().max().item(),
            'hf_reconstruction_error': (r_hf - hf_proc).abs().max().item(),
        }


# # ════════════════════════════════════════════════════════════════════
# #  快速测试
# # ════════════════════════════════════════════════════════════════════
#
# if __name__ == '__main__':
#     model = FreqSepPhys(theta=0.5, frames=128)
#     model.eval()
#
#     dummy = torch.randn(2, 3, 128, 64, 64)
#     with torch.no_grad():
#         out = model(dummy)
#     print(f"Input : {dummy.shape}")
#     print(f"Output: {out.shape}  (expected [2, 128])")
#
#     # 验证可逆频率交换层精度
#     feat_slow = torch.randn(2, 64, 8, 8, 8)
#     info = model.verify_freq_exchange_invertibility(feat_slow, block_idx=0)
#     print(f"\n可逆频率交换验证 — {info['block']}:")
#     print(f"  lf 重建误差: {info['lf_reconstruction_error']:.2e}")
#     print(f"  hf 重建误差: {info['hf_reconstruction_error']:.2e}")
#     print("（理论上应接近机器精度 ~1e-6）")
#
#     total = sum(p.numel() for p in model.parameters())
#     print(f"\n总参数量: {total:,}")