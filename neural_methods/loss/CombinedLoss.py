"""
CombinedLoss.py — FreqSepPhys++ 组合损失函数
=================================================
统一管理主任务损失 + 所有创新模块的正则化损失。

损失组成：
L_total = L_task                                    # 主任务：NegPearson
        + λ_proj   · L_proj_ortho                   # CDRE：色彩投影正交性
        + λ_afd    · L_afd_sep                      # AFD：σ分离约束
        + λ_scp    · L_scp                          # SCP：频谱对比损失（预训练阶段）
        + λ_freq   · L_freq_consistency             # FreqTTA：频域自监督（TTA阶段）

各权重的设计依据：
- λ_proj=0.1:  正交性是软约束，过强会限制色彩变换的表达力
- λ_afd=0.3:  σ分离需足够强以保证频率不混叠，但不主导主损失
- λ_scp=1.0:  预训练阶段为主损失；微调阶段设0
- λ_freq=0.5: TTA阶段自监督信号，需平衡适应与稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_methods.loss.NegPearsonLoss import NegPearsonLoss


class CombinedLoss(nn.Module):
    """
    FreqSepPhys++ 组合损失管理器。

    使用方式：
        criterion = CombinedLoss()
        # 在训练循环中：
        pred = model(x)
        loss_dict = criterion(pred, label, model=model, x=x)
        loss_dict['total'].backward()
    """

    def __init__(self,
                 lambda_proj: float = 0.1,
                 lambda_afd: float = 0.3,
                 lambda_motion: float = 0.01,
                 lambda_scp: float = 0.0,
                 lambda_freq: float = 0.0):
        super().__init__()
        # 主任务损失
        self.neg_pearson = NegPearsonLoss()

        # 正则化权重
        self.lambda_proj = lambda_proj
        self.lambda_afd = lambda_afd
        self.lambda_motion = lambda_motion
        self.lambda_scp = lambda_scp
        self.lambda_freq = lambda_freq

    def forward(self, pred: torch.Tensor, label: torch.Tensor,
                model: nn.Module = None, x: torch.Tensor = None,
                scp_loss: torch.Tensor = None,
                freq_self_sup_loss: torch.Tensor = None) -> dict:
        """
        计算组合损失。

        Args:
            pred: [B, T] 模型预测rPPG信号
            label: [B, T] 真实rPPG标签
            model: FreqSepPhysPlusPlus模型（用于获取正则化损失）
            x: [B, 3, T, H, W] 原始输入（用于AFD正则化）
            scp_loss: SCP预训练损失（预训练阶段传入）
            freq_self_sup_loss: FreqTTA自监督损失（TTA阶段传入）

        Returns:
            dict with 'total', 'task', 'proj_ortho', 'afd_sep', 'scp', 'freq'
        """
        losses = {}
        total = torch.tensor(0.0, device=pred.device)

        # ── 主任务损失 ────────────────────────────────────────
        task_loss = self.neg_pearson(pred, label)
        losses['task'] = task_loss
        total = total + task_loss

        # ── CDRE：色彩投影正交性 ─────────────────────────────
        if model is not None and self.lambda_proj > 0:
            if hasattr(model, 'use_cdre_stem') and model.use_cdre_stem:
                proj_loss = model.proj_ortho_loss(model.stem.color_proj)
                losses['proj_ortho'] = proj_loss
                total = total + self.lambda_proj * proj_loss

        # ── AFD：频率分解正则化（正交性 + 能量守恒 + σ分离）──────
        # 修复：改用 afd_total（含 ortho/energy/sep 三项，均可微），
        # 不再仅取失效的 afd_sep。afd_reg_loss 内部已对各项加权，
        # 此处 lambda_afd 作为AFD正则整体的全局缩放。
        if model is not None and self.lambda_afd > 0 and x is not None:
            afd_reg = model.get_afd_regularization(x)
            afd_total = afd_reg.get('afd_total', afd_reg.get('afd_sep'))
            if afd_total is not None:
                losses['afd_sep'] = afd_reg.get('afd_sep')
                losses['afd_ortho'] = afd_reg.get('afd_ortho')
                losses['afd_energy'] = afd_reg.get('afd_energy')
                losses['afd_total'] = afd_total
                total = total + self.lambda_afd * afd_total

        # ── MotionGate：时间平滑性 + 稀疏通过约束 ───────────────
        if model is not None and self.lambda_motion > 0 and x is not None:
            motion_reg = model.get_motion_gate_regularization(x)
            motion_total = motion_reg.get('motion_total')
            if motion_total is not None:
                losses['motion_smooth'] = motion_reg.get('motion_smooth')
                losses['motion_sparse'] = motion_reg.get('motion_sparse')
                losses['motion_total'] = motion_total
                total = total + self.lambda_motion * motion_total

        # ── SCP：频谱对比损失（预训练阶段）────────────────────
        if scp_loss is not None and self.lambda_scp > 0:
            losses['scp'] = scp_loss
            total = total + self.lambda_scp * scp_loss

        # ── FreqTTA：频域自监督损失（TTA阶段）─────────────────
        if freq_self_sup_loss is not None and self.lambda_freq > 0:
            losses['freq'] = freq_self_sup_loss
            total = total + self.lambda_freq * freq_self_sup_loss

        losses['total'] = total
        return losses

    def update_weights(self, **kwargs):
        """动态更新损失权重（用于多阶段训练）。"""
        for key, val in kwargs.items():
            if hasattr(self, f'lambda_{key}'):
                setattr(self, f'lambda_{key}', val)