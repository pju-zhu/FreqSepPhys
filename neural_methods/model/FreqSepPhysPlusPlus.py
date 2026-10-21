"""
FreqSepPhysPlusPlus.py — FreqSepPhys++ 主模型整合
====================================================
将4个创新模块深度融合到FreqSepPhys架构中：

创新点1 AFD → 替换FreqSepBlock中的固定σ rPPGFreqDecomposer → AdaptiveFreqDecomposer
         替换低频路末端固定GaussianLowPass3d → AdaptiveLowPassConstraint
创新点2 FreqTTA → 不在主模型中（测试时外挂模块，由Trainer调用）
创新点3 SCP → 主模型暴露encode()接口供SCP预训练调用
创新点4 CDRE → 替换ConvBlock1为IlluminationInvariantStem
         在FreqSepBlock高频路插入MotionRobustnessGating
         训练时在输入端插入HRFrequencyAugmentation

融合策略（非简单堆砌）：
- AFD与FreqSepBlock深度耦合：每个block的分解器+低通约束联动
- CDRE Stem替换原始ConvBlock1，输出16通道（与原始一致）
- MotionGate插入高频路BiMamba之后、Gating之前
- SCP的encode()接口复用主模型backbone（到poolspa之前）
- FreqTTA在推理时由Trainer包装

维度链路（B=4, T=128, H=W=128）：
Input[B,3,128,128,128]
→ CDRE Stem → [B,16,128,64,64] (替换ConvBlock1+Maxpool)
→ ConvBlock2 → [B,32,128,64,64]
→ ConvBlock3 → [B,64,128,64,64] → Maxpool → [B,64,128,32,32]
→ 慢流ConvBlock4(stride[4,1,1]) → [B,64,32,32,32]
→ 快流ConvBlock5(stride[2,1,1]) → [B,32,64,32,32]
→ 3组(FreqSepBlock++ + Maxpool + CrossFusion)
→ Cat[B,96,128,8,8] → Upsample[B,48,128,8,8] → PoolSpa[B,48,128,1,1]
→ RegressionHead → [B,1,128,1,1] → view → [B,128]
"""

import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from mamba_ssm import Mamba
from torch.nn import functional as F

# 导入原始架构组件
from neural_methods.model.FreqSepPhys import (
    GaussianLowPass3d,
    LowPassRegressionHead,
    LowPassUpsampleBlock,
    LowPassConvBlock,
    InvertibleFreqExchange,
    BiMambaEncoder,
    FreqChannelGating,
    CrossStreamFusion,
    CDC_T,
    conv_block,
)

# 导入创新模块
from neural_methods.model.AFD import (
    AdaptiveFreqDecomposer,
    AdaptiveLowPassConstraint,
    AFDRegularizationLoss,
)
from neural_methods.model.CDRE import (
    IlluminationInvariantStem,
    MotionRobustnessGating,
    HRFrequencyAugmentation,
    ProjectionOrthogonalityLoss,
    MotionGateRegularizationLoss,
    CDREModule,
)


# ════════════════════════════════════════════════════════════════════
#  ★ FreqSepBlock++ — 融合AFD + MotionGate的增强频率分离块
# ════════════════════════════════════════════════════════════════════

class FreqSepBlockPlusPlus(nn.Module):
    """
    增强频率分离处理块（融合创新点1 AFD + 创新点4 MotionGate）。

    相比原始FreqSepBlock的改进：
    1. 频率分解器：固定σ → AdaptiveFreqDecomposer（可学习/自适应/各向异性）
    2. 低频路低通约束：固定GaussianLowPass3d → AdaptiveLowPassConstraint
    3. 高频路新增MotionRobustnessGating（BiMamba之后、Gating之前）

    内部流程：
    ┌─ 输入 x ─────────────────────────────────────────────────────┐
    │  ① AdaptiveFreqDecomposer → low_freq, high_freq            │
    │  ② 低频路：CDC_T→BN→ReLU→BiMamba→Gating→AdaptiveLowPass   │
    │  ③ 高频路：CDC_T→BN→ReLU→BiMamba→MotionGate→Gating       │
    │  ④ InvertibleFreqExchange                                  │
    │  ⑤ 残差相加输出                                             │
    └──────────────────────────────────────────────────────────────┘

    Args:
        channels: 通道数
        theta: CDC_T的时间差分系数
        stream_sigma: 低频路低通σ初始值（慢流1.8/快流0.8）
        afd_mode: AFD模式 'learnable'|'adaptive'|'anisotropic'
        use_motion_gate: 是否启用运动门控
        kernel_size: 高斯核大小
    """
    def __init__(self, channels: int, theta: float,
                 stream_sigma: float = 1.5,
                 afd_mode: str = 'anisotropic',
                 use_motion_gate: bool = True,
                 kernel_size: int = 5):
        super().__init__()

        # 原始分解结果缓存（供AFD正交/能量正则化使用）
        self._cached_lf = None
        self._cached_hf = None
        self._cached_input = None

        # ── 创新点1：自适应频率分解 ──────────────────────────
        self.freq_decomposer = AdaptiveFreqDecomposer(
            channels=channels,
            mode=afd_mode,
            init_sigma=stream_sigma,
            kernel_size=kernel_size,
        )

        # ── 低频路（干扰建模路）──────────────────────────────
        self.cdc_lf = CDC_T(channels, channels, theta=theta)
        self.bn_lf = nn.BatchNorm3d(channels)
        self.relu_lf = nn.ReLU(inplace=True)
        self.mamba_lf = BiMambaEncoder(dim=channels)
        self.gating_lf = FreqChannelGating(in_channels=channels, reduction=2)
        # 创新点1：可学习低通约束（替换固定GaussianLowPass3d）
        self.lowpass_lf = AdaptiveLowPassConstraint(
            channels=channels,
            kernel_size=kernel_size,
            init_sigma=stream_sigma,
        )

        # ── 高频路（rPPG信号建模路）────────────────────────
        self.cdc_hf = CDC_T(channels, channels, theta=theta)
        self.bn_hf = nn.BatchNorm3d(channels)
        self.relu_hf = nn.ReLU(inplace=True)
        self.mamba_hf = BiMambaEncoder(dim=channels)
        # 创新点4B：运动鲁棒性门控（BiMamba之后、Gating之前）
        self.use_motion_gate = use_motion_gate
        if use_motion_gate:
            self.motion_gate = MotionRobustnessGating(channels=channels)
        self.gating_hf = FreqChannelGating(in_channels=channels, reduction=2)

        # ── 可逆频率信息交换 ─────────────────────────────────
        self.freq_exchange = InvertibleFreqExchange(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ① 自适应频率分解
        low_freq, high_freq = self.freq_decomposer(x)

        # 缓存原始分解结果（供AFD正交/能量正则化使用，保留计算图）。
        # 仅训练时缓存，推理时置None以省显存；正则化读取后应及时清理。
        if self.training:
            self._cached_lf = low_freq
            self._cached_hf = high_freq
            self._cached_input = x
        else:
            self._cached_lf = None
            self._cached_hf = None
            self._cached_input = None

        # ② 低频路处理
        low_freq = self.relu_lf(self.bn_lf(self.cdc_lf(low_freq)))
        low_freq = self.mamba_lf(low_freq)
        low_freq = self.gating_lf(low_freq)
        low_freq = self.lowpass_lf(low_freq)  # 自适应低通约束

        # ③ 高频路处理
        high_freq = self.relu_hf(self.bn_hf(self.cdc_hf(high_freq)))
        high_freq = self.mamba_hf(high_freq)
        if self.use_motion_gate:
            high_freq = self.motion_gate(high_freq)  # 运动门控
        high_freq = self.gating_hf(high_freq)

        # ④ 可逆频率信息交换
        low_freq, high_freq = self.freq_exchange(low_freq, high_freq)

        # ⑤ 残差相加
        return low_freq + high_freq


# ════════════════════════════════════════════════════════════════════
#  ★ FreqSepPhysPlusPlus 主模型
# ════════════════════════════════════════════════════════════════════

class FreqSepPhysPlusPlus(nn.Module):
    """
    FreqSepPhys++ — 四创新点融合的完整rPPG模型。

    架构改进总览（vs FreqSepPhys）：
    ┌──────────────────────┬──────────────────────┬───────────────────────────────┐
    │ 组件                 │ 原版                 │ FreqSepPhys++                 │
    ├──────────────────────┼──────────────────────┼───────────────────────────────┤
    │ 第一层ConvBlock1     │ 标准Conv3d           │ IlluminationInvariantStem(4B) │
    │ FreqSepBlock分解器   │ 固定σ高斯            │ AdaptiveFreqDecomposer(AFD)   │
    │ 低频路低通           │ 固定GaussianLowPass  │ AdaptiveLowPassConstraint(AFD)│
    │ 高频路               │ 无运动处理            │ +MotionRobustnessGating(4B)   │
    │ 数据增强             │ 无                   │ HRFrequencyAugmentation(4C)   │
    │ 预训练支持           │ 无                   │ encode()接口供SCP使用         │
    │ 测试时适应           │ 无                   │ 兼容FreqTTA包装               │
    └──────────────────────┴──────────────────────┴───────────────────────────────┘

    Args:
        theta: CDC_T时间差分系数（默认0.5）
        drop_rate1: 第一组Dropout率（默认0.25）
        drop_rate2: 第二组Dropout率（默认0.5）
        frames: 时间帧数（默认128）
        afd_mode: AFD模式（默认'anisotropic'）
        use_motion_gate: 是否启用运动门控（默认True）
        use_cdre_stem: 是否使用CDRE光照不变Stem（默认True）
        use_freq_aug: 是否启用频域增强（默认True）
        sample_rate: 采样率fps（默认30）
        use_diff_norm: Stem内是否再做帧间差分归一化（默认False）。
            当数据预处理DATA_TYPE已为DiffNormalized时必须保持False，
            否则会造成二次差分破坏rPPG生理信号；仅Raw/Standardized输入时可设True。
    """

    # 尺度自适应截止频率（与原版一致，作为AFD初始值）
    SLOW_STREAM_SIGMA = 1.8
    FAST_STREAM_SIGMA = 0.8

    def __init__(self, theta=0.5, drop_rate1=0.25, drop_rate2=0.5,
                 frames=128, afd_mode='anisotropic',
                 use_motion_gate=True, use_cdre_stem=True,
                 use_freq_aug=True, sample_rate=30,
                 use_diff_norm=False):
        super().__init__()
        self.frames = frames
        self.use_cdre_stem = use_cdre_stem
        self.sample_rate = sample_rate
        self.use_diff_norm = use_diff_norm

        # ── 创新点4A：光照不变性Stem（替换ConvBlock1）─────────
        if use_cdre_stem:
            self.stem = IlluminationInvariantStem(
                out_channels=16,
                num_heads=4,
                use_diff_norm=use_diff_norm,
            )
        else:
            self.stem = conv_block(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2])

        # ── 共享特征提取（ConvBlock2-3保持不变）────────────────
        self.ConvBlock2 = conv_block(16, 32, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock3 = conv_block(32, 64, [3, 3, 3], stride=1, padding=1)
        self.ConvBlock4 = conv_block(64, 64, [4, 1, 1], stride=[4, 1, 1], padding=0)
        self.ConvBlock5 = conv_block(64, 32, [2, 1, 1], stride=[2, 1, 1], padding=0)

        # ── 快流末端精化 ──────────────────────────────────────
        self.ConvBlock6 = LowPassConvBlock(
            32, 32, kernel_size=[3, 1, 1], stride=1,
            padding=[1, 0, 0], activation='elu', smooth_sigma=0.8,
        )

        # ── 慢流：FreqSepBlock++（AFD + MotionGate，σ=1.8）───
        self.Block1 = FreqSepBlockPlusPlus(
            64, theta, stream_sigma=self.SLOW_STREAM_SIGMA,
            afd_mode=afd_mode, use_motion_gate=use_motion_gate,
        )
        self.Block2 = FreqSepBlockPlusPlus(
            64, theta, stream_sigma=self.SLOW_STREAM_SIGMA,
            afd_mode=afd_mode, use_motion_gate=use_motion_gate,
        )
        self.Block3 = FreqSepBlockPlusPlus(
            64, theta, stream_sigma=self.SLOW_STREAM_SIGMA,
            afd_mode=afd_mode, use_motion_gate=use_motion_gate,
        )

        # ── 快流：FreqSepBlock++（AFD + MotionGate，σ=0.8）───
        self.Block4 = FreqSepBlockPlusPlus(
            32, theta, stream_sigma=self.FAST_STREAM_SIGMA,
            afd_mode=afd_mode, use_motion_gate=use_motion_gate,
        )
        self.Block5 = FreqSepBlockPlusPlus(
            32, theta, stream_sigma=self.FAST_STREAM_SIGMA,
            afd_mode=afd_mode, use_motion_gate=use_motion_gate,
        )
        self.Block6 = FreqSepBlockPlusPlus(
            32, theta, stream_sigma=self.FAST_STREAM_SIGMA,
            afd_mode=afd_mode, use_motion_gate=use_motion_gate,
        )

        # ── 上采样精化 ────────────────────────────────────────
        self.upsample1 = LowPassUpsampleBlock(64, 64, smooth_sigma=1.0)
        self.upsample2 = LowPassUpsampleBlock(96, 48, smooth_sigma=1.0)

        # ── 输出回归头 ────────────────────────────────────────
        self.ConvBlockLast = LowPassRegressionHead(
            in_channels=48, out_channels=1,
            kernel_size=9, smooth_sigma=1.5,
        )

        # ── 池化/融合/Dropout ─────────────────────────────────
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

        # ── 创新点4C：HR频域增强（训练时在线增强）──────────────
        self.use_freq_aug = use_freq_aug
        if use_freq_aug:
            self.freq_aug = HRFrequencyAugmentation(sample_rate=sample_rate)

        # ── 正则化损失模块 ────────────────────────────────────
        self.afd_reg_loss = AFDRegularizationLoss()
        self.proj_ortho_loss = ProjectionOrthogonalityLoss()
        self.gate_reg_loss = MotionGateRegularizationLoss()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码器前向（供SCP预训练使用）。
        返回poolspa之前的特征，不含回归头。

        Args:
            x: [B, 3, T, H, W]
        Returns:
            features: [B, 48, T, 1, 1] 融合后特征
        """
        # ── 共享特征提取 ──────────────────────────────────────
        x = self.stem(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpa(x)

        # ── 双流分叉 ──────────────────────────────────────────
        s_x = self.ConvBlock4(x)
        f_x = self.ConvBlock5(x)

        # ── 三组处理 ──────────────────────────────────────────
        s_x1 = self.drop_1(self.MaxpoolSpa(self.Block1(s_x)))
        f_x1 = self.drop_2(self.MaxpoolSpa(self.Block4(f_x)))
        s_x1 = self.cross_fuse_1(s_x1, f_x1)

        s_x2 = self.drop_3(self.MaxpoolSpa(self.Block2(s_x1)))
        f_x2 = self.drop_4(self.MaxpoolSpa(self.Block5(f_x1)))
        s_x2 = self.cross_fuse_2(s_x2, f_x2)

        s_x3 = self.drop_5(self.upsample1(self.Block3(s_x2)))
        f_x3 = self.drop_6(self.ConvBlock6(self.Block6(f_x2)))

        # ── 融合 → 上采样精化 ─────────────────────────────────
        x_final = self.upsample2(torch.cat((f_x3, s_x3), dim=1))
        return x_final  # [B, 48, T, H', W']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        完整前向传播。

        Args:
            x: [B, 3, T, H, W] 输入视频
        Returns:
            pred: [B, T] rPPG预测信号
        """
        # 创新点4C：训练时频域增强
        if self.use_freq_aug and self.training:
            x = self.freq_aug(x)

        # 编码
        features = self.encode(x)  # [B, 48, T, H', W']

        # 全局池化 → 回归头
        x_final = self.ConvBlockLast(self.poolspa(features))
        return x_final.view(-1, x_final.shape[2])

    def get_regularization_losses(self, x: torch.Tensor = None) -> dict:
        """
        获取所有创新模块的正则化损失。
        在Trainer中与主损失加权求和。

        Returns:
            dict with:
            - 'proj_ortho': 色彩投影正交性损失
            - 'afd_sigma_info': AFD的σ值信息（用于监控）
        """
        losses = {}

        # 色彩投影正交性（仅CDRE Stem启用时）
        if self.use_cdre_stem:
            losses['proj_ortho'] = self.proj_ortho_loss(self.stem.color_proj)

        # AFD σ值监控信息
        sigma_info = {}
        for name, block in [('Block1', self.Block1), ('Block4', self.Block4)]:
            sigma_info[name] = block.freq_decomposer.get_sigma_values()
        losses['afd_sigma_info'] = sigma_info

        return losses

    def get_afd_regularization(self, x: torch.Tensor) -> dict:
        """
        计算AFD正则化损失（使用可微σ张量 + block缓存的原始分解结果）。

        修复要点（相对旧实现）：
        1. σ改用 get_sigma_tensors() 返回的可微张量（不再用.item()切断计算图），
           使 afd_sep 对 sigma_*_raw 参数产生真实梯度。
        2. 接入 FreqSepBlock.forward 缓存的原始 lf/hf，恢复正交性/能量守恒两项损失，
           不再是"只剩失效sep约束"的空壳。
        3. sep_margin 由 AFDRegularizationLoss 统一控制（默认对慢/快流σ做分离约束）。

        Args:
            x: [B, 3, T, H, W] 原始输入（仅用于device/占位）
        Returns:
            dict with 'afd_sep' / 'afd_ortho' / 'afd_energy' / 'afd_total'
        """
        device = x.device
        zero = torch.tensor(0.0, device=device)

        # 1. 聚合慢流Block1-3与快流Block4-6的可微时间σ
        slow_blocks = (self.Block1, self.Block2, self.Block3)
        fast_blocks = (self.Block4, self.Block5, self.Block6)
        all_blocks = slow_blocks + fast_blocks

        def temporal_sigma(block):
            sigma = block.freq_decomposer.get_sigma_tensors()
            if sigma.get('mode') == 'anisotropic':
                return sigma.get('sigma_t')
            if sigma.get('mode') == 'learnable':
                return sigma.get('sigma')
            return None

        slow_sigmas = [s for s in (temporal_sigma(b) for b in slow_blocks) if s is not None]
        fast_sigmas = [s for s in (temporal_sigma(b) for b in fast_blocks) if s is not None]
        sigma_slow = torch.stack([s.mean() for s in slow_sigmas]).mean() if slow_sigmas else None
        sigma_fast = torch.stack([s.mean() for s in fast_sigmas]).mean() if fast_sigmas else None

        # 2. 聚合六个块缓存的原始分解结果
        ortho_terms = []
        energy_terms = []
        for block in all_blocks:
            lf = getattr(block, '_cached_lf', None)
            hf = getattr(block, '_cached_hf', None)
            cached_x = getattr(block, '_cached_input', None)
            if lf is not None and hf is not None and cached_x is not None:
                reg = self.afd_reg_loss(cached_x, lf, hf)
                ortho_terms.append(reg['ortho'])
                energy_terms.append(reg['energy'])

        afd_ortho = torch.stack(ortho_terms).mean() if ortho_terms else zero
        afd_energy = torch.stack(energy_terms).mean() if energy_terms else zero
        if sigma_slow is not None and sigma_fast is not None:
            afd_sep = self.afd_reg_loss.sep_weight * F.relu(
                sigma_fast - sigma_slow + self.afd_reg_loss.sep_margin
            )
        else:
            afd_sep = zero

        # 3. 读取后清理全部缓存，避免跨step悬挂计算图
        for block in all_blocks:
            block._cached_lf = None
            block._cached_hf = None
            block._cached_input = None

        return {
            'afd_sep': afd_sep,
            'afd_ortho': afd_ortho,
            'afd_energy': afd_energy,
            'afd_total': afd_ortho + afd_energy + afd_sep,
        }

    def get_motion_gate_regularization(self, x: torch.Tensor) -> dict:
        """聚合六个频率分离块最近一次前向产生的运动门控正则。"""
        blocks = (self.Block1, self.Block2, self.Block3,
                  self.Block4, self.Block5, self.Block6)
        zero = x.sum() * 0.0
        smooth_terms = []
        sparse_terms = []
        for block in blocks:
            gate_module = getattr(block, 'motion_gate', None)
            gate = getattr(gate_module, 'last_gate', None)
            if gate is not None:
                reg = self.gate_reg_loss(gate)
                smooth_terms.append(reg['smooth'])
                sparse_terms.append(reg['sparse'])

        smooth = torch.stack(smooth_terms).mean() if smooth_terms else zero
        sparse = torch.stack(sparse_terms).mean() if sparse_terms else zero
        return {
            'motion_smooth': smooth,
            'motion_sparse': sparse,
            'motion_total': smooth + sparse,
        }



# ════════════════════════════════════════════════════════════════════
#  快速维度验证
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    model = FreqSepPhysPlusPlus(theta=0.5, frames=128)
    model.eval()

    dummy = torch.randn(2, 3, 128, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}  (expected [2, 128])")

    # encode接口测试
    with torch.no_grad():
        feat = model.encode(dummy)
    print(f"Encode: {feat.shape}")

    # 正则化损失
    reg = model.get_regularization_losses()
    print(f"Reg keys: {list(reg.keys())}")

    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total:,}")