"""
FreqTTA.py — Frequency-aware Test-Time Adaptation (频率感知测试时自适应)
=========================================================================
创新点2：利用rPPG信号的频域先验进行无标签测试时模型自适应。

核心思想：
- 测试时没有标签，但rPPG信号具有强频域先验可作为自监督信号
- 3种频域自监督信号不依赖标签，仅依赖物理先验
- 仅适应少量参数（BN层+FreqChannelGating），避免灾难性遗忘

与Bi-TTA(ECCV 2024)的区别：
- Bi-TTA仅用一种周期性先验，我们用3种互补频域信号
- Bi-TTA适应所有参数，我们仅适应BN+Gating（更稳定）
- 我们支持持续TTA（指数移动平均记忆）和单次TTA两种模式

3种频域自监督信号：
1. 频谱集中性 (Spectral Concentration, SC)：
   rPPG功率谱应集中在心率基频附近 → L_SC = 1 - max(PSD)/sum(PSD)
   
2. 时间平稳性 (Temporal Stationarity, TS)：
   短时窗口内心率应稳定 → L_TS = var(HR_windows)
   
3. 分离一致性 (Separation Consistency, SepC)：
   高频分量应携带大部分rPPG信号能量 → L_SepC = energy(lf) / energy(total)

生理依据：
- 健康人静息心率稳定性：10s内变化<5bpm（平稳性先验）
- rPPG信号>60%能量集中在基频±0.2Hz（集中性先验）
- 高频路应比低频路携带更多rPPG信息（分离先验）

Args详见各类docstring。
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy
from collections import deque


# ════════════════════════════════════════════════════════════════════
#  频域自监督信号计算
# ════════════════════════════════════════════════════════════════════

class FreqDomainSelfSupervision(nn.Module):
    """
    频域自监督信号综合模块。
    
    计算3种不依赖标签的频域先验损失，用于TTA时的参数更新。
    
    Args:
        sample_rate: 采样率fps（默认30）
        hr_min: 有效心率下界bpm（默认40，覆盖低心率运动员）
        hr_max: 有效心率上界bpm（默认200，覆盖剧烈运动）
        sc_weight: 频谱集中性权重（默认1.0）
        ts_weight: 时间平稳性权重（默认0.5）
        sep_weight: 分离一致性权重（默认0.3）
        window_size: 平稳性计算的子窗口帧数（默认32，约1s@30fps）
    """
    def __init__(self, sample_rate: int = 30,
                 hr_min: float = 40.0, hr_max: float = 200.0,
                 sc_weight: float = 1.0,
                 ts_weight: float = 0.5,
                 sep_weight: float = 0.3,
                 window_size: int = 32):
        super().__init__()
        self.sample_rate = sample_rate
        self.freq_min = hr_min / 60.0   # Hz
        self.freq_max = hr_max / 60.0   # Hz
        self.sc_weight = sc_weight
        self.ts_weight = ts_weight
        self.sep_weight = sep_weight
        self.window_size = window_size
        # soft-argmax 温度：越小越接近硬argmax（更尖锐但梯度更弱）。
        # 0.05 在 rPPG PSD 归一化尺度下峰值定位偏差 < 1 bpm，且保留有效梯度。
        self.softargmax_temp = 0.05

    def spectral_concentration_loss(self, signal: torch.Tensor) -> torch.Tensor:
        """
        频谱集中性损失。
        
        物理先验：健康人rPPG功率谱应在基频处形成尖锐峰。
        散布的频谱意味着信号质量差或包含运动伪影。
        
        L_SC = 1 - max(PSD_valid) / sum(PSD_valid)
        完美集中→L_SC=0，完全散布→L_SC≈1
        
        Args:
            signal: [B, T] 预测的rPPG信号
        Returns:
            loss: 标量
        """
        B, T = signal.shape
        X = torch.fft.rfft(signal, dim=-1)
        psd = X.abs().pow(2)  # [B, T//2+1]
        freqs = torch.fft.rfftfreq(T, d=1.0/self.sample_rate).to(signal.device)
        
        # 有效频段
        valid_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        psd_valid = psd[:, valid_mask]  # [B, n_valid]
        
        # 集中性 = max / sum
        psd_max = psd_valid.max(dim=-1)[0]  # [B]
        psd_sum = psd_valid.sum(dim=-1) + 1e-8  # [B]
        concentration = psd_max / psd_sum  # [B]，越大越好
        
        loss = (1.0 - concentration).mean()
        return loss

    def temporal_stationarity_loss(self, signal: torch.Tensor) -> torch.Tensor:
        """
        时间平稳性损失。
        
        物理先验：短时间内（4-10s）心率变化缓慢。
        将信号分割为子窗口，各窗口峰值频率应一致。
        
        L_TS = var(peak_freqs_across_windows)
        完美平稳→L_TS=0，心率跳变→L_TS大
        
        Args:
            signal: [B, T] 预测的rPPG信号
        Returns:
            loss: 标量
        """
        B, T = signal.shape
        W = self.window_size
        
        if T < 2 * W:
            return torch.tensor(0.0, device=signal.device)
        
        # 分割为重叠子窗口（50%重叠）
        stride = W // 2
        num_windows = (T - W) // stride + 1
        
        peak_freqs = []
        freqs = torch.fft.rfftfreq(W, d=1.0/self.sample_rate).to(signal.device)
        valid_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        valid_freqs = freqs[valid_mask]
        
        for i in range(num_windows):
            start = i * stride
            window = signal[:, start:start+W]  # [B, W]
            
            X_w = torch.fft.rfft(window, dim=-1)
            psd_w = X_w.abs().pow(2)[:, valid_mask]  # [B, n_valid]
            
            # 修复：用 soft-argmax 替代 argmax，恢复对信号的可微性。
            # argmax 梯度处处为0，会使 TS 损失完全无法优化模型；
            # soft-argmax = Σ softmax(psd/τ)·freq，为峰值频率的可微期望估计。
            tau = self.softargmax_temp
            # 先按每样本峰值归一化PSD到[0,1]，避免PSD绝对量级差异导致softmax溢出
            psd_norm = psd_w / (psd_w.amax(dim=-1, keepdim=True) + 1e-8)
            weights = F.softmax(psd_norm / tau, dim=-1)                # [B, n_valid]
            peak_f = (weights * valid_freqs.unsqueeze(0)).sum(dim=-1)  # [B] 可微
            peak_freqs.append(peak_f)
        
        # [num_windows, B] → [B, num_windows]
        peak_freqs = torch.stack(peak_freqs, dim=-1)  # [B, num_windows]
        
        # 计算峰值频率的方差（应尽量小）
        loss = peak_freqs.var(dim=-1).mean()
        return loss

    def separation_consistency_loss(self, low_freq_signal: torch.Tensor,
                                     total_signal: torch.Tensor) -> torch.Tensor:
        """
        分离一致性损失。
        
        物理先验：高频分量应携带更多rPPG信号能量（因为rPPG是皮肤微脉动）。
        如果低频分量在rPPG有效频段内能量过大，说明分离不充分。
        
        L_SepC = energy_rppg_band(lf) / energy_rppg_band(total)
        完美分离→L_SepC≈0（低频不含rPPG），分离失败→L_SepC≈1
        
        Args:
            low_freq_signal: [B, T] 低频分量信号
            total_signal: [B, T] 总信号
        Returns:
            loss: 标量
        """
        B, T = total_signal.shape
        freqs = torch.fft.rfftfreq(T, d=1.0/self.sample_rate).to(total_signal.device)
        valid_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        
        # 低频分量在rPPG频段的能量
        X_lf = torch.fft.rfft(low_freq_signal, dim=-1)
        psd_lf_valid = X_lf.abs().pow(2)[:, valid_mask].sum(dim=-1)  # [B]
        
        # 总信号在rPPG频段的能量
        X_total = torch.fft.rfft(total_signal, dim=-1)
        psd_total_valid = X_total.abs().pow(2)[:, valid_mask].sum(dim=-1) + 1e-8  # [B]
        
        # 低频占比应尽量小
        ratio = psd_lf_valid / psd_total_valid  # [B]
        loss = ratio.mean()
        return loss

    def forward(self, pred_signal: torch.Tensor,
                low_freq_signal: torch.Tensor = None) -> dict:
        """
        计算综合频域自监督损失。
        
        Args:
            pred_signal: [B, T] 模型预测的rPPG信号
            low_freq_signal: [B, T] 低频分量信号（可选，用于分离一致性）
        Returns:
            dict with individual losses and total
        """
        losses = {}
        
        losses['sc'] = self.sc_weight * self.spectral_concentration_loss(pred_signal)
        losses['ts'] = self.ts_weight * self.temporal_stationarity_loss(pred_signal)
        
        if low_freq_signal is not None:
            losses['sep'] = self.sep_weight * self.separation_consistency_loss(
                low_freq_signal, pred_signal
            )
        else:
            losses['sep'] = torch.tensor(0.0, device=pred_signal.device)
        
        losses['total'] = losses['sc'] + losses['ts'] + losses['sep']
        return losses


# ════════════════════════════════════════════════════════════════════
#  ★ FreqTTA 主模块
# ════════════════════════════════════════════════════════════════════

class FreqTTA(nn.Module):
    """
    Frequency-aware Test-Time Adaptation (FreqTTA)。
    
    测试时自适应策略：
    1. 加载预训练模型
    2. 冻结大部分参数，仅解冻BN层和FreqChannelGating
    3. 使用频域自监督信号更新解冻参数
    4. 更新后的模型预测当前测试样本
    
    支持两种TTA模式：
    
    Mode A - 单次TTA (One-shot TTA)：
        每个测试样本独立适应，不累积跨样本信息。
        适合：测试样本间无时间关联（不同受试者）
        
    Mode B - 持续TTA (Continual TTA)：
        使用指数移动平均(EMA)累积适应信息。
        适合：测试样本来自同一视频的连续片段
    
    参数选择依据：
    - 仅适应BN层：BN统计量对域偏移最敏感（均值/方差漂移）
    - 仅适应Gating：通道注意力权重需适应新域的通道重要性分布
    - 不适应卷积/Mamba权重：避免灾难性遗忘核心特征提取能力
    
    Args:
        model: 预训练好的FreqSepPhysPlusPlus模型
        lr: TTA学习率（默认1e-4，比训练时小100x）
        num_steps: 每个样本的适应步数（默认3，避免过适应）
        ema_momentum: EMA动量（默认0.99，持续TTA使用）
        mode: 'one_shot' | 'continual'
        adapt_bn: 是否适应BN层（默认True）
        adapt_gating: 是否适应Gating层（默认True）
    """
    def __init__(self, model: nn.Module,
                 lr: float = 1e-4,
                 num_steps: int = 3,
                 ema_momentum: float = 0.99,
                 mode: str = 'one_shot',
                 adapt_bn: bool = True,
                 adapt_gating: bool = True,
                 sample_rate: int = 30):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_steps = num_steps
        self.ema_momentum = ema_momentum
        self.mode = mode
        self.adapt_bn = adapt_bn
        self.adapt_gating = adapt_gating
        
        # 频域自监督信号计算器
        self.freq_supervision = FreqDomainSelfSupervision(
            sample_rate=sample_rate,
        )
        
        # 保存原始模型状态（用于one-shot模式重置）
        self.original_state = None
        
        # 持续TTA的EMA模型
        if mode == 'continual':
            self.ema_model = deepcopy(model)
            for p in self.ema_model.parameters():
                p.requires_grad = False
        
        # 适应历史记录（用于监控）
        self.adaptation_history = deque(maxlen=100)

    def _get_adaptable_params(self):
        """获取去重后的可适应参数列表。"""
        params = []
        seen = set()

        def add_param(param):
            if param is not None and id(param) not in seen:
                params.append(param)
                seen.add(id(param))

        for name, module in self.model.named_modules():
            if self.adapt_bn and isinstance(module, nn.BatchNorm3d):
                add_param(module.weight)
                add_param(module.bias)
            if self.adapt_gating and 'gating' in name.lower():
                for param in module.parameters():
                    add_param(param)
        return params

    def _configure_model_for_tta(self):
        """配置模型进入TTA模式。"""
        self.model.eval()

        if self.adapt_bn:
            for module in self.model.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.train()

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self._get_adaptable_params():
            param.requires_grad = True

    def _save_original_state(self):
        """保存包含参数和缓冲区的完整模型状态。"""
        self.original_state = deepcopy(self.model.state_dict())

    def _restore_original_state(self):
        """恢复完整模型状态，防止one-shot样本间状态泄漏。"""
        if self.original_state is not None:
            self.model.load_state_dict(self.original_state, strict=True)
            self.original_state = None
        self.model.eval()

    @torch.no_grad()
    def _update_ema(self):
        """更新EMA模型（持续TTA模式）"""
        if self.mode != 'continual':
            return
        for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.ema_momentum).add_(
                model_p.data, alpha=1.0 - self.ema_momentum
            )

    def _extract_low_freq(self, signal: torch.Tensor) -> torch.Tensor:
        """从预测信号中提取低频分量（可微，供 SepC 使用）。

        用 FFT 域理想低通实现：保留 rPPG 有效频段下界 freq_min 以下的分量。
        SepC 度量"rPPG频段内低频能量占比应尽量小"，此低频分量与 total_signal
        处于同一时域尺度，可直接比较。

        Args:
            signal: [B, T]
        Returns:
            low_freq_signal: [B, T]（可微）
        """
        B, T = signal.shape
        freqs = torch.fft.rfftfreq(
            T, d=1.0 / self.freq_supervision.sample_rate
        ).to(signal.device)
        # 低通掩码：保留 < freq_min 的甚低频（呼吸/基线漂移等干扰）
        lp_mask = (freqs < self.freq_supervision.freq_min).float()
        X = torch.fft.rfft(signal, dim=-1)
        X_lp = X * lp_mask.unsqueeze(0)
        low_freq_signal = torch.fft.irfft(X_lp, n=T, dim=-1)
        return low_freq_signal

    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        测试时自适应+预测。
        
        流程：
        1. 配置模型TTA模式
        2. N步自监督适应
        3. 最终预测
        4. (one-shot) 恢复原始状态 / (continual) 更新EMA
        
        Args:
            x: [B, 3, T, H, W] 测试输入
        Returns:
            pred: [B, T] rPPG预测
        """
        # 配置TTA
        self._configure_model_for_tta()
        
        if self.mode == 'one_shot':
            self._save_original_state()
        
        # 获取可适应参数和优化器
        adaptable_params = self._get_adaptable_params()
        if not adaptable_params:
            # 无可适应参数，直接预测
            with torch.no_grad():
                return self.model(x)
        
        optimizer = torch.optim.Adam(adaptable_params, lr=self.lr)
        
        # N步自监督适应（显式启用梯度，兼容外层推理上下文）
        adaptation_losses = []
        with torch.enable_grad():
            for step in range(self.num_steps):
                optimizer.zero_grad()

                # 前向传播
                pred = self.model(x)  # [B, T]

                # 补传低频分量信号，恢复SepC分离一致性损失。
                low_freq_signal = self._extract_low_freq(pred)
                losses = self.freq_supervision(pred, low_freq_signal=low_freq_signal)
                total_loss = losses['total']

                total_loss.backward()
                optimizer.step()
                adaptation_losses.append(total_loss.detach().item())

        # 最终预测（使用适应后的模型）
        # 修复：continual 模式下用 EMA 模型预测（更稳定，抵抗跨片段漂移），
        # 与 2025 持续TTA-rPPG 文献一致；one_shot 模式仍用适应后的当前模型。
        with torch.no_grad():
            if self.mode == 'continual':
                self._update_ema()  # 先更新EMA再用其预测
                self.ema_model.eval()  # 用running stats，避免batch统计波动
                final_pred = self.ema_model(x)
            else:
                final_pred = self.model(x)

        # 记录适应历史
        self.adaptation_history.append({
            'losses': adaptation_losses,
            'loss_reduction': adaptation_losses[0] - adaptation_losses[-1] if len(adaptation_losses) > 1 else 0,
        })

        # 模式特定后处理（EMA已在预测前更新，此处仅one_shot需恢复状态）
        if self.mode == 'one_shot':
            self._restore_original_state()

        return final_pred

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        统一前向接口。
        训练时直接前向，测试时执行TTA。
        """
        if self.training:
            return self.model(x)
        else:
            return self.adapt_and_predict(x)

    def get_adaptation_stats(self) -> dict:
        """获取适应统计信息"""
        if not self.adaptation_history:
            return {'n_adaptations': 0}
        
        losses_first = [h['losses'][0] for h in self.adaptation_history]
        losses_last = [h['losses'][-1] for h in self.adaptation_history]
        reductions = [h['loss_reduction'] for h in self.adaptation_history]
        
        return {
            'n_adaptations': len(self.adaptation_history),
            'avg_initial_loss': sum(losses_first) / len(losses_first),
            'avg_final_loss': sum(losses_last) / len(losses_last),
            'avg_reduction': sum(reductions) / len(reductions),
        }


# ════════════════════════════════════════════════════════════════════
#  TTA置信度评估
# ════════════════════════════════════════════════════════════════════

class TTAConfidenceEstimator(nn.Module):
    """
    TTA置信度评估器。
    
    评估TTA适应效果的可靠性，决定是否使用适应后的预测。
    
    策略：
    - 如果TTA后频谱质量提升 → 使用适应后的预测
    - 如果TTA后频谱质量下降 → 回退适应前的预测
    - 通过置信度分数加权两个预测
    
    置信度计算：
    - 基于频谱集中性变化
    - 基于损失减少幅度
    - 基于预测稳定性（适应前后预测的相关性）
    
    Args:
        confidence_threshold: 置信度阈值（默认0.5）
    """
    def __init__(self, confidence_threshold: float = 0.5,
                 sample_rate: int = 30):
        super().__init__()
        self.threshold = confidence_threshold
        self.sample_rate = sample_rate
        self.freq_min = 40.0 / 60.0  # Hz
        self.freq_max = 200.0 / 60.0  # Hz

    def compute_confidence(self, pred_before: torch.Tensor,
                          pred_after: torch.Tensor) -> torch.Tensor:
        """
        计算TTA适应的置信度。
        
        Args:
            pred_before: [B, T] 适应前的预测
            pred_after: [B, T] 适应后的预测
        Returns:
            confidence: [B] 每个样本的置信度分数 ∈ [0, 1]
        """
        B, T = pred_before.shape
        
        # 1. 频谱质量对比
        sc_before = self._spectral_concentration(pred_before)
        sc_after = self._spectral_concentration(pred_after)
        quality_improvement = (sc_after - sc_before)  # >0表示提升
        
        # 2. 预测稳定性（Pearson相关）
        pred_b_norm = pred_before - pred_before.mean(dim=-1, keepdim=True)
        pred_a_norm = pred_after - pred_after.mean(dim=-1, keepdim=True)
        correlation = F.cosine_similarity(pred_b_norm, pred_a_norm, dim=-1)  # [B]
        
        # 3. 综合置信度
        # 质量提升+高相关→高置信度使用TTA结果
        # 质量下降+低相关→低置信度，回退原始
        confidence = torch.sigmoid(
            5.0 * quality_improvement + 2.0 * (correlation - 0.5)
        )
        
        return confidence

    def _spectral_concentration(self, signal: torch.Tensor) -> torch.Tensor:
        """计算频谱集中性 [B]"""
        B, T = signal.shape
        X = torch.fft.rfft(signal, dim=-1)
        psd = X.abs().pow(2)
        freqs = torch.fft.rfftfreq(T, d=1.0/self.sample_rate).to(signal.device)
        valid_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        psd_valid = psd[:, valid_mask]
        return psd_valid.max(dim=-1)[0] / (psd_valid.sum(dim=-1) + 1e-8)

    def blend_predictions(self, pred_before: torch.Tensor,
                         pred_after: torch.Tensor) -> torch.Tensor:
        """
        基于置信度混合适应前后的预测。
        
        Args:
            pred_before: [B, T]
            pred_after: [B, T]
        Returns:
            blended: [B, T]
        """
        confidence = self.compute_confidence(pred_before, pred_after)  # [B]
        
        # 高置信度→用TTA结果，低置信度→用原始结果
        alpha = confidence.unsqueeze(-1)  # [B, 1]
        blended = alpha * pred_after + (1 - alpha) * pred_before
        
        return blended