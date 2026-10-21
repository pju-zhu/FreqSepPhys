"""
SCP.py — Spectral Contrastive Pre-training (频谱对比预训练)
============================================================
创新点3：在频谱嵌入空间进行对比学习，建立rPPG信号的频率先验表示。

核心思想：
- rPPG信号具有强频率结构先验：能量集中在心率基频及其谐波
- 同一受试者不同时间窗口的频谱应相似（正样本对）
- 不同受试者/不同心率的频谱应不同（负样本对）
- 通过对比学习，使编码器学会频谱感知的特征表示

方法：
1. SpectralProjectionHead: 将模型中间特征投影到频谱嵌入空间
2. SpectralAugmentation: 频域数据增强生成正样本对
3. NTXentSpectralLoss: NT-Xent对比损失在频谱嵌入上计算
4. SCPFramework: 预训练框架（编码器+投影头+对比损失）

技术路线：
- Phase 1 预训练：无标签数据 + 频谱对比学习
- Phase 2 微调：冻结编码器，训练回归头

与BeatFormer(2025)的区别：
- BeatFormer在原始信号空间做SCL，我们在频谱嵌入空间
- 我们使用更丰富的频域增强（幅度+相位+频率偏移）
- 我们引入频率一致性约束（正样本对应相同心率频段）

生理依据：
- 正样本定义：同一视频≤10s内的不同窗口（心率短期内稳定）
- 负样本定义：不同视频/相隔>30s的窗口（心率可能不同）
- 频率集中性先验：健康人rPPG功率谱>60%集中在基频±0.2Hz
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ════════════════════════════════════════════════════════════════════
#  频谱投影头
# ════════════════════════════════════════════════════════════════════

class SpectralProjectionHead(nn.Module):
    """
    频谱投影头：将编码器特征投影到频谱嵌入空间。
    
    设计：
    1. 时间维度全局池化 → 频域特征提取（FFT）
    2. 频域MLP投影到低维嵌入空间
    3. L2归一化到单位超球面（对比学习标准做法）
    
    为什么在频域做投影（而非时域）：
    - rPPG的核心信息在频域（心率=频率位置）
    - 频域表示对时间平移不变（正样本对的关键需求）
    - 频域幅度谱对相位噪声鲁棒
    
    Args:
        in_channels: 编码器输出通道数
        temporal_dim: 时间维度大小
        embed_dim: 对比学习嵌入维度（默认128）
        hidden_dim: MLP隐层维度（默认256）
        sample_rate: 采样率（用于频率轴计算）
    """
    def __init__(self, in_channels: int, temporal_dim: int = 128,
                 embed_dim: int = 128, hidden_dim: int = 256,
                 sample_rate: int = 30):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_dim = temporal_dim
        self.embed_dim = embed_dim
        self.sample_rate = sample_rate
        
        # rPPG有效频段
        self.freq_min = 0.7   # Hz（42bpm，覆盖极低心率）
        self.freq_max = 4.0   # Hz（240bpm，覆盖极高心率+一次谐波）
        
        # 计算有效频率bin数量
        freq_resolution = sample_rate / temporal_dim  # Hz per bin
        self.n_valid_bins = int((self.freq_max - self.freq_min) / freq_resolution) + 1
        
        # 空间全局池化
        self.spatial_pool = nn.AdaptiveAvgPool3d((temporal_dim,1, 1))
        
        # 频谱特征提取MLP
        # 输入: in_channels * n_valid_bins (频域幅度谱的有效频段)
        spectral_input_dim = in_channels * self.n_valid_bins
        
        self.spectral_mlp = nn.Sequential(
            nn.Linear(spectral_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] 编码器中间特征
        Returns:
            embed: [B, embed_dim] L2归一化的频谱嵌入
        """
        B, C, T, H, W = x.shape
        
        # 1. 空间池化 → [B, C, T, 1, 1] → [B, C, T]
        x_pooled = self.spatial_pool(x).squeeze(-1).squeeze(-1)  # [B, C, T]
        
        # 2. FFT → 频域幅度谱
        X_freq = torch.fft.rfft(x_pooled, dim=-1)  # [B, C, T//2+1] complex
        amplitude = X_freq.abs()  # [B, C, T//2+1]
        
        # 3. 提取rPPG有效频段
        freq_resolution = self.sample_rate / T
        bin_min = max(1, int(self.freq_min / freq_resolution))
        bin_max = min(amplitude.shape[-1] - 1, int(self.freq_max / freq_resolution))
        amplitude_valid = amplitude[:, :, bin_min:bin_max+1]  # [B, C, n_bins]
        
        # 4. 展平 → MLP投影
        amp_flat = amplitude_valid.reshape(B, -1)  # [B, C*n_bins]
        
        # 动态适配维度（不同T会导致n_bins变化）
        if amp_flat.shape[1] != self.spectral_mlp[0].in_features:
           # 使用自适应池化对齐维度
            amp_flat = F.adaptive_avg_pool1d(
                amp_flat.unsqueeze(1), self.spectral_mlp[0].in_features
            ).squeeze(1)
        
        embed = self.spectral_mlp(amp_flat)  # [B, embed_dim]
        
        # 5. L2归一化到单位超球面
        embed = F.normalize(embed, dim=-1)
        
        return embed


# ════════════════════════════════════════════════════════════════════
#  频谱数据增强（生成对比学习正样本对）
# ════════════════════════════════════════════════════════════════════

class SpectralAugmentation(nn.Module):
    """
    频域数据增强，生成对比学习的正样本对。
    
    增强策略（保持rPPG信号的核心频率结构不变）：
    
    1. 幅度缩放（Amplitude Scaling）：
       - 对整体幅度施加随机缩放
       - 模拟不同信噪比条件
       - 不改变频率位置（心率不变）
       
    2. 频谱噪声注入（Spectral Noise）：
       - 在非心率频段注入高斯噪声
       - 模拟真实场景中的频域干扰
       
    3. 相位扰动（Phase Perturbation）：
       - 对相位施加小幅随机扰动
       - 模拟不同时间起点的窗口
       - 不改变幅度谱（频率结构不变）
       
    4. 频率微偏移（Frequency Micro-shift）：
       - 将频谱整体微移±1-2 bins
       - 模拟心率的自然波动（±2-3bpm）
    
    所有增强都保持rPPG信号的核心特征：
    - 基频位置基本不变（心率不变）
    - 能量集中性不变（频谱结构保持）
    - 谐波关系不变（基频-二倍频比例保持）
    
    Args:
        amplitude_scale_range: 幅度缩放范围 (默认(0.7, 1.3))
        phase_noise_std: 相位噪声标准差 (默认0.1 rad)
        freq_shift_max: 最大频率偏移bins数 (默认2)
        noise_level: 频谱噪声水平 (默认0.1)
    """
    def __init__(self, amplitude_scale_range=(0.7, 1.3),
                 phase_noise_std: float = 0.1,
                 freq_shift_max: int = 2,
                 noise_level: float = 0.1,
                 sample_rate: int = 30):
        super().__init__()
        self.amp_min, self.amp_max = amplitude_scale_range
        self.phase_noise_std = phase_noise_std
        self.freq_shift_max = freq_shift_max
        self.noise_level = noise_level
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入施加保持计算图的随机频域增强。"""
        B, C, T, H, W = x.shape
        x_temporal = x.mean(dim=[3, 4])
        X = torch.fft.rfft(x_temporal, dim=-1)
        amp = X.abs()
        phase = X.angle()

        # 随机量本身不需要梯度，但增强运算必须保留输入梯度。
        with torch.no_grad():
            scale = torch.empty(B, 1, 1, device=x.device).uniform_(
                self.amp_min, self.amp_max
            )
            phase_noise = torch.randn_like(phase) * self.phase_noise_std
            shifts = torch.randint(
                -self.freq_shift_max,
                self.freq_shift_max + 1,
                (B,),
                device=x.device,
            ) if self.freq_shift_max> 0 else torch.zeros(B, dtype=torch.long, device=x.device)

        amp_aug = amp * scale
        # 仅在rPPG频带外注入噪声，避免直接破坏生理主频。
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sample_rate).to(x.device)
        non_hr_mask = ((freqs < 40.0 / 60.0) | (freqs > 200.0 / 60.0)).view(1, 1, -1)
        noise = torch.randn_like(amp) * self.noise_level * amp.mean(dim=-1, keepdim=True)
        amp_aug = (amp_aug + noise * non_hr_mask).clamp_min(0)
        phase_aug = phase + phase_noise
        X_aug = amp_aug * torch.exp(1j * phase_aug)

        # 每个样本独立做小幅频移，并清除roll产生的环绕频点。
        shifted = []
        for sample, shift_tensor in zip(X_aug, shifts):
            shift = int(shift_tensor.item())
            sample_shifted = torch.roll(sample, shifts=shift, dims=-1)
            if shift > 0:
                sample_shifted = torch.cat([
                    torch.zeros_like(sample_shifted[..., :shift]),
                    sample_shifted[..., shift:],
                ], dim=-1)
            elif shift < 0:
                sample_shifted = torch.cat([
                    sample_shifted[..., :shift],
                    torch.zeros_like(sample_shifted[..., shift:]),
                ], dim=-1)
            shifted.append(sample_shifted)
        X_aug = torch.stack(shifted, dim=0)

        x_aug_temporal = torch.fft.irfft(X_aug, n=T, dim=-1)
        diff = (x_aug_temporal - x_temporal).unsqueeze(-1).unsqueeze(-1)
        return x + diff


# ════════════════════════════════════════════════════════════════════
#  NT-Xent对比损失（频谱版）
# ════════════════════════════════════════════════════════════════════

class NTXentSpectralLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    用于频谱对比预训练。
    
    对比学习标准损失：
    L = -log( exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) )
    
    其中：
    - z_i, z_j: 同一样本两种增强视图的嵌入（正样本对）
 - z_k: 其他所有样本的嵌入（负样本）
    - τ: 温度参数（控制对比难度）
    - sim: 余弦相似度
    
    温度选择依据：
    - τ=0.07: SimCLR默认，适合大batch
    - τ=0.1: 适合小batch（rPPG数据量有限）
    - τ越小，对比越"hard"，需要更多负样本
    
    Args:
        temperature: 温度参数（默认0.1，适合rPPG小batch场景）
        gather_distributed: 是否收集分布式训练的负样本
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        计算NT-Xent损失。
        
        Args:
            z_i: [B, D] 一视图的嵌入（L2归一化后）
            z_j: [B, D] 第二视图的嵌入（L2归一化后）
        Returns:
            loss: 标量对比损失
        """
        B = z_i.shape[0]
        
        # 拼接所有嵌入: [2B, D]
        z = torch.cat([z_i, z_j], dim=0)
        
        # 计算所有对的余弦相似度矩阵: [2B, 2B]
        sim = z @ z.t() / self.temperature
        
        # 构建正样本对mask
        # 对于样本i(view1)，其正样本是i+B(view2)，反之亦然
        pos_mask = torch.zeros(2 * B, 2 * B, device=z.device, dtype=torch.bool)
        for i in range(B):
            pos_mask[i, i + B] = True
            pos_mask[i + B, i] = True
        
        # 自身不参与对比（mask对角线）
        self_mask = ~torch.eye(2 * B, device=z.device, dtype=torch.bool)
        
        # 正样本相似度
        pos_sim = sim[pos_mask].view(2 * B)
        
        # 分母：所有非自身样本（正+负）
        denominator = sim.masked_fill(~self_mask, float('-inf'))
        log_sum_exp = torch.logsumexp(denominator, dim=-1)
        
        # InfoNCE loss
        loss = (-pos_sim + log_sum_exp).mean()
        
        return loss


# ════════════════════════════════════════════════════════════════════
#  频谱一致性损失（辅助正则项）
# ════════════════════════════════════════════════════════════════════

class SpectralConsistencyLoss(nn.Module):
    """
    频谱一致性损失：确保对比学习不破坏频率结构先验。
    
    约束：
    1. 频谱集中性（Spectral Concentration）：
       rPPG功率谱应集中在少数频率bins
       L_conc = 1 - max(PSD) / sum(PSD)  → 鼓励尖锐峰
       
    2. 谐波一致性（Harmonic Consistency）：
       二倍频幅度应与基频幅度成比例
       L_harm = |A(2f0) - α*A(f0)|²，α∈[0.2, 0.5]
    
    这些约束确保对比学习的嵌入保留了rPPG信号的物理特征。
    
    Args:
       concentration_weight: 集中性权重（默认0.1）
        harmonic_weight: 谐波一致性权重（默认0.05）
        sample_rate: 采样率
        hr_min/hr_max: 有效心率范围
    """
    def __init__(self, concentration_weight: float = 0.1,
                 harmonic_weight: float = 0.05,
                 sample_rate: int = 30,
                 hr_min: float = 40.0, hr_max: float = 200.0):
        super().__init__()
        self.concentration_weight = concentration_weight
        self.harmonic_weight = harmonic_weight
        self.sample_rate = sample_rate
        self.freq_min = hr_min / 60.0
        self.freq_max = hr_max / 60.0

    def forward(self, pred_signal: torch.Tensor) -> dict:
        """
        Args:
            pred_signal: [B, T] 模型预测的rPPG信号
        Returns:
            dict with 'concentration', 'harmonic', 'total'
        """
        B, T = pred_signal.shape
        losses = {}
        
        # FFT
        X = torch.fft.rfft(pred_signal, dim=-1)
        psd = X.abs().pow(2)  # [B, T//2+1] 功率谱密度
        freqs = torch.fft.rfftfreq(T, d=1.0/self.sample_rate).to(pred_signal.device)
        
        # 有效频段mask
        valid_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        psd_valid = psd[:, valid_mask]  # [B, n_valid]
        
        # 1. 频谱集中性
        psd_max = psd_valid.max(dim=-1, keepdim=True)[0]
        psd_sum = psd_valid.sum(dim=-1, keepdim=True) + 1e-8
        concentration = (psd_max / psd_sum).squeeze(-1)  # [B]，越大越集中
        losses['concentration'] = self.concentration_weight * (1.0 - concentration).mean()
        
        # 2. 谐波一致性：在真实频率轴上定位2f0，避免有效频带偏移导致索引错误。
        valid_freqs = freqs[valid_mask]
        peak_idx = psd_valid.argmax(dim=-1)
        batch_idx = torch.arange(B, device=pred_signal.device)
        fundamental_freq = valid_freqs[peak_idx]
        harmonic_freq = 2.0 * fundamental_freq
        valid_harmonic = harmonic_freq <= self.freq_max

        if valid_harmonic.any():
            harmonic_idx = (valid_freqs.unsqueeze(0) - harmonic_freq.unsqueeze(1)).abs().argmin(dim=-1)
            fund_amp = psd_valid[batch_idx, peak_idx].sqrt()
            harm_amp = psd_valid[batch_idx, harmonic_idx].sqrt()
            ratio = harm_amp / (fund_amp + 1e-8)
            harm_loss = F.relu(ratio - 0.5) + F.relu(0.2 - ratio)
            valid_count = valid_harmonic.float().sum().clamp_min(1.0)
            losses['harmonic'] = self.harmonic_weight * (
                harm_loss * valid_harmonic.float()
            ).sum() / valid_count
        else:
            losses['harmonic'] = pred_signal.sum() * 0.0
        
        losses['total'] = losses['concentration'] + losses['harmonic']
        return losses


# ════════════════════════════════════════════════════════════════════
#  ★ SCP预训练框架
# ═══════════════════════════════════════════════════════════════════

class SCPFramework(nn.Module):
    """
    Spectral Contrastive Pre-training (SCP) 完整框架。
    
    训练范式：
    ┌─────────────────────────────────────────────────────────┐
    │ Phase 1: 预训练（无标签）                               │
    │   Input → [Encoder] → features                         │
    │   features → [Aug1] → view1 → [ProjHead] → z1         │
    │   features → [Aug2] → view2 → [ProjHead] → z2         │
    │   Loss = NT-Xent(z1, z2) + λ*SpectralConsistency       │
    ├─────────────────────────────────────────────────────────┤
    │ Phase 2: 微调（有标签）                                 │
    │   Input → [Encoder(frozen/finetune)] → features        │
    │   features → [RegressionHead] → rPPG signal            │
    │   Loss = NegPearson + λ*SpectralConsistency             │
    └─────────────────────────────────────────────────────────┘
    
    Args:
        encoder: FreqSepPhysPlusPlus编码器（不含回归头）
        feature_channels: 编码器输出通道数
        temporal_dim: 时间维度
        embed_dim: 对比嵌入维度
        temperature: NT-Xent温度
        sample_rate: 采样率
    """
    def __init__(self, encoder: nn.Module,
                 feature_channels: int = 48,
                 temporal_dim: int = 128,
                 embed_dim: int = 128,
                 temperature: float = 0.1,
                 sample_rate: int = 30):
        super().__init__()
        
        self.encoder = encoder
        
        # 频谱投影头（预训练时使用，微调时丢弃）
        self.projection_head = SpectralProjectionHead(
            in_channels=feature_channels,
            temporal_dim=temporal_dim,
            embed_dim=embed_dim,
            sample_rate=sample_rate,
        )
        
        # 频域增强器
        self.augmentation = SpectralAugmentation(sample_rate=sample_rate)
        
        # 对比损失
        self.ntxent_loss = NTXentSpectralLoss(temperature=temperature)
        
        # 频谱一致性正则
        self.spectral_consistency = SpectralConsistencyLoss(
            sample_rate=sample_rate,
        )

        # 时间窗口正样本采样器（接入预训练，替代"同一窗口两次增强"）
        # 生理先验：同一视频时间邻近子窗心率结构高度相似，构成更强的正样本对
        self.temporal_sampler = TemporalPositiveSampler(
            chunk_length=max(temporal_dim // 2, 32),
            max_temporal_gap=min(temporal_dim, 300),
            sample_rate=sample_rate,
        )
        self.use_temporal_sampling = True

        self.is_pretraining = True

    def set_pretrain_mode(self):
        """设置为预训练模式"""
        self.is_pretraining = True
        self.encoder.train()
        self.projection_head.train()

    def set_finetune_mode(self, freeze_encoder: bool = False):
        """设置为微调模式"""
        self.is_pretraining = False
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def pretrain_forward(self, x: torch.Tensor) -> dict:
        """
        预训练前向传播。
        
        Args:
            x: [B, 3, T, H, W] 输入视频
        Returns:
            dict with 'loss', 'ntxent_loss', 'consistency_loss', 'z1', 'z2'
        """
        # 编码器提取特征（需要encoder暴露中间特征接口）
        features = self.encoder.encode(x)  # [B, C, T, H, W]

        # 修复：接入时间窗口正样本采样。
        # 原实现对同一 features 做两次增强，正样本对仅差随机增强，先验较弱；
        # 现改为在时间维采样两个邻近子窗（生理上心率结构相似），再各自增强，
        # 使 TemporalPositiveSampler 的时间邻近先验真正生效。
        if self.use_temporal_sampling and features.shape[2] >= 2 * self.temporal_sampler.chunk_length:
            T = features.shape[2]
            L = self.temporal_sampler.chunk_length
            s1, s2 = self.temporal_sampler.sample_positive_pair(T)
            feat1 = features[:, :, s1:s1 + L]
            feat2 = features[:, :, s2:s2 + L]
            view1 = self.augmentation(feat1)
            view2 = self.augmentation(feat2)
        else:
            # 时间长度不足以切两个子窗时，回退到同窗两次增强
            view1 = self.augmentation(features)
            view2 = self.augmentation(features)
        
        # 投影到嵌入空间
        z1 = self.projection_head(view1)  # [B, embed_dim]
        z2 = self.projection_head(view2)  # [B, embed_dim]
        
        # NT-Xent对比损失
        ntxent = self.ntxent_loss(z1, z2)
        
        # 频谱一致性（可选，如果encoder有rPPG信号输出）
        # 这里用特征的时间平均作为粗略信号估计
        pseudo_signal = features.mean(dim=[1, 3, 4])  # [B, T]
        consistency = self.spectral_consistency(pseudo_signal)
        
        total_loss = ntxent + 0.1 * consistency['total']
        
        return {
            'loss': total_loss,
            'ntxent_loss': ntxent,
            'consistency_loss': consistency['total'],
            'z1': z1.detach(),
            'z2': z2.detach(),
        }

    def finetune_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        微调前向传播（标准rPPG回归）。
        
        Args:
            x: [B, 3, T, H, W]
        Returns:
           pred: [B, T] rPPG预测信号
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor):
        """根据当前模式选择前向路径"""
        if self.is_pretraining:
            return self.pretrain_forward(x)
        else:
            return self.finetune_forward(x)

    def get_pretrain_params(self):
        """获取预训练阶段的可训练参数"""
        return list(self.encoder.parameters()) + list(self.projection_head.parameters())

    def get_finetune_params(self, freeze_encoder: bool = False):
        """获取微调阶段的可训练参数"""
        if freeze_encoder:
            return [p for p in self.encoder.parameters() if p.requires_grad]
        return list(self.encoder.parameters())


# ════════════════════════════════════════════════════════════════════
#  时间窗口正样本采样器
# ════════════════════════════════════════════════════════════════════

class TemporalPositiveSampler:
    """
    时间窗口正样本采策略。
    
    为对比学习生成正样本对的采样策略：
    - 正样本：同一视频中时间间隔<10s的不同窗口
    - 负样本：batch内其他所样本自动构成负样本
    
    生理依据：
    - 静息心率变异率(HRV)约4-8%，即60bpm基线±5bpm
    - 10s内心率变化通常<5bpm，频谱结构高度相似
    - >30s时心率可能因呼吸/姿态变化产生显著差异
    
    使用方式：
    - 在DataLoader中调用，返回同一视频的两个相邻窗口
    - 这不是nn.Module，而是数据预处理逻辑
    """
    def __init__(self, chunk_length: int = 128,
                 max_temporal_gap: int = 300,  # 10s * 30fps
                 sample_rate: int = 30):
        self.chunk_length = chunk_length
        self.max_temporal_gap = max_temporal_gap
        self.sample_rate = sample_rate

    def sample_positive_pair(self, video_length: int) -> tuple:
        """
        从视频中采样正样本对的起始索引。
        
        Args:
            video_length: 视频总帧数
        Returns:
            (start1, start2): 两个窗口的起始帧索引
        """
        import random
        max_start = video_length - self.chunk_length
        if max_start <= 0:
            return 0, 0
        
        start1 = random.randint(0, max_start)
        
        # 正样本：在start1附近max_temporal_gap范围内采样
        gap_min = self.chunk_length  # 至少不重叠
        gap_max = min(self.max_temporal_gap, max_start - start1)
        
        if gap_max <= gap_min:
            start2 = max(0, start1 - self.chunk_length)
        else:
            offset = random.randint(gap_min, gap_max)
            start2 = start1 + offset
            if start2 + self.chunk_length > video_length:
                start2 = max(0, start1 - offset)
        
        return start1, start2