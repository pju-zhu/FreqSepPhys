"""
FreqSepPhysPlusPlusTrainer.py — 多阶段训练器
================================================
支持3阶段训练流程：
  阶段1: SCP频谱对比预训练（无标签，学习通用频率表征）
  阶段2: 监督微调（有标签，NegPearson + 正则化损失）
  阶段3: FreqTTA测试时适应（无标签，频域自监督适应）

使用方式：
  # 阶段1：SCP预训练
  trainer = FreqSepPhysPlusPlusTrainer(config, data_loader)
  trainer.pretrain_scp(data_loader)

  # 阶段2：监督微调
  trainer.finetune_supervised(data_loader)

  # 阶段3：测试时适应
  trainer.test_with_tta(data_loader)

  # 或一步到位
  trainer.train(data_loader)  # 自动执行阶段1→2
  trainer.test(data_loader)   # 自动执行阶段3
"""

import os
import math
import numpy as np
import torch
import torch.optim as optim
from collections import OrderedDict
from tqdm import tqdm
from scipy.signal import welch

from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.model.FreqSepPhysPlusPlus import FreqSepPhysPlusPlus
from neural_methods.model.SCP import SCPFramework
from neural_methods.model.FreqTTA import FreqTTA
from neural_methods.loss.CombinedLoss import CombinedLoss
from evaluation.metrics import calculate_metrics


def _normalize_per_sample(signal: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """沿时间维逐样本标准化，避免跨样本统计污染和零方差。"""
    mean = signal.mean(dim=-1, keepdim=True)
    std = signal.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (signal - mean) / std


class FreqSepPhysPlusPlusTrainer(BaseTrainer):
    """
    FreqSepPhys++ 多阶段训练器。

    训练流程：
    ┌────────────────────────────────────────────────────────────────┐
    │ 阶段1: SCP预训练                                               │
    │   - 损失: NTXent频谱对比 + 频谱一致性                          │
    │   - 冻结: 无（全参数训练）                                      │
    │   - 目的: 学习通用频率表征，提升跨域泛化                       │
    │                                                                │
    │ 阶段2: 监督微调                                                │
    │   - 损失: NegPearson + λ_proj·正交性 + λ_afd·σ分离             │
    │   - 加载: 阶段1预训练权重                                      │
    │   - 目的: 任务特异性优化                                       │
    │                                                                │
    │ 阶段3: FreqTTA测试时适应                                       │
    │   - 损失: 频域自监督（SC + TS + SepC）                        │
    │   - 适应: 仅BN层 + FreqChannelGating参数                     │
    │   - 目的: 无标签目标域适应                                     │
    └────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1

        # 数据是否已在预处理阶段做过帧间差分（DiffNormalized）。
        # 若已差分，则Stem内的TemporalDiffNorm必须关闭，避免二次差分破坏rPPG信号。
        _data_types = getattr(config.TRAIN.DATA.PREPROCESS, 'DATA_TYPE', [])
        stem_use_diff_norm = "DiffNormalized" not in _data_types
        self.frame_rate = config.TRAIN.DATA.FS

        # ── 创建FreqSepPhys++模型 ─────────────────────────────
        self.model = FreqSepPhysPlusPlus(
            theta=0.5,
            frames=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH,
            sample_rate=config.TRAIN.DATA.FS,
            afd_mode=getattr(config.TRAIN, 'AFD_MODE', 'anisotropic'),
            use_motion_gate=getattr(config.TRAIN, 'USE_MOTION_GATE', True),
            use_cdre_stem=getattr(config.TRAIN, 'USE_CDRE_STEM', True),
            use_freq_aug=getattr(config.TRAIN, 'USE_FREQ_AUG', True),
            use_diff_norm=stem_use_diff_norm,
        ).to(self.device)

        if self.num_of_gpu > 1:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
            )

        # ── 多阶段配置 ────────────────────────────────────────
        self.scp_pretrain_epochs = getattr(config.TRAIN, 'SCP_PRETRAIN_EPOCHS', 10)
        self.use_scp_pretrain = getattr(config.TRAIN, 'USE_SCP_PRETRAIN', True)
        self.use_tta = getattr(config.TEST, 'USE_TTA', False)

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])

            # 组合损失
            self.criterion = CombinedLoss(
                lambda_proj=getattr(config.TRAIN, 'LAMBDA_PROJ', 0.1),
                lambda_afd=getattr(config.TRAIN, 'LAMBDA_AFD', 0.3),
                lambda_motion=getattr(config.TRAIN, 'LAMBDA_MOTION', 0.01),
                lambda_scp=0.0,  # 微调阶段关闭SCP损失
                lambda_freq=0.0,  # 训练阶段关闭TTA损失
            )

            # 优化器：分层学习率（AFD的σ参数用较小lr）
            self.optimizer = self._build_optimizer(config)

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches,
            )

            # SCP框架（阶段1使用）
            if self.use_scp_pretrain:
                self.scp_framework = SCPFramework(
                    encoder=self.model,
                    feature_channels=48,
                    temporal_dim=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH,
                    embed_dim=128,
                    temperature=0.1,
                    sample_rate=config.TRAIN.DATA.FS,
                ).to(self.device)
                self.scp_optimizer = optim.AdamW(
                    self.scp_framework.parameters(),
                    lr=getattr(config.TRAIN, 'SCP_LR', 1e-3),
                    weight_decay=1e-4,
                )

        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("Incorrect toolbox mode!")

    def _build_optimizer(self, config):
        """分层学习率优化器：AFD σ参数用较小lr防止振荡。"""
        sigma_params = []
        base_params = []
        for name, param in self.model.named_parameters():
            if 'sigma' in name.lower() or 'lowpass' in name.lower():
                sigma_params.append(param)
            else:
                base_params.append(param)

        param_groups = [
            {'params': base_params, 'lr': config.TRAIN.LR},
        ]
        if sigma_params:
            param_groups.append({
                'params': sigma_params,
                'lr': config.TRAIN.LR * 0.1,  # σ参数用1/10学习率
                'name': 'sigma',
            })

        return optim.Adam(param_groups, lr=config.TRAIN.LR, weight_decay=0.0005)

    # ════════════════════════════════════════════════════════════════
    #  阶段1：SCP频谱对比预训练
    # ════════════════════════════════════════════════════════════════

    def pretrain_scp(self, data_loader):
        """SCP频谱对比预训练阶段。"""
        if not self.use_scp_pretrain:
            print("跳过SCP预训练（USE_SCP_PRETRAIN=False）")
            return

        print(f"\n{'='*60}")
        print(f"阶段1: SCP频谱对比预训练 ({self.scp_pretrain_epochs} epochs)")
        print(f"{'='*60}")

        self.scp_framework.set_pretrain_mode()

        for epoch in range(self.scp_pretrain_epochs):
            print(f"\n--- SCP Pretrain Epoch: {epoch} ---")
            self.scp_framework.train()
            loss_avg = []
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description(f"SCP epoch {epoch}")
                data = batch[0].float().to(self.device)

                self.scp_optimizer.zero_grad()
                output = self.scp_framework.pretrain_forward(data)
                loss = output['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.scp_framework.parameters(), 1.0)
                self.scp_optimizer.step()

                loss_avg.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            print(f"SCP Epoch {epoch} avg loss: {np.mean(loss_avg):.4f}")

        # 保存预训练权重
        scp_checkpoint = os.path.join(
            self.model_dir, self.model_file_name + '_SCP_pretrain.pth'
        )
        os.makedirs(self.model_dir, exist_ok=True)
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), scp_checkpoint)
        else:
            torch.save(self.model.state_dict(), scp_checkpoint)
        print(f"SCP预训练权重已保存: {scp_checkpoint}")

        # 切换到微调模式
        self.scp_framework.set_finetune_mode(freeze_encoder=False)

    # ════════════════════════════════════════════════════════════════
    #  阶段2：监督微调
    # ════════════════════════════════════════════════════════════════

    def finetune_supervised(self, data_loader):
        """监督微调阶段（与原始train流程一致，但使用CombinedLoss）。"""
        print(f"\n{'='*60}")
        print(f"阶段2: 监督微调 ({self.max_epoch_num} epochs)")
        print(f"{'='*60}")

        for epoch in range(self.max_epoch_num):
            print(f"\n====Training Epoch: {epoch}====")
            self.model.train()
            loss_rPPG_avg = []
            running_loss = 0.0

            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description(f"Train epoch {epoch}")
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape
                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)

                # 预测与标签均沿时间维逐样本标准化。
                pred_ppg = _normalize_per_sample(pred_ppg)
                labels = _normalize_per_sample(labels)

                # 组合损失
                model_ref = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                loss_dict = self.criterion(
                    pred_ppg, labels,
                    model=model_ref, x=data,
                )
                loss = loss_dict['total']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                running_loss += loss.item()

                if idx % 100 == 99:
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())

            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                print(f'validation loss: {valid_loss:.4f}')
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")

            torch.cuda.empty_cache()

        if not self.config.TEST.USE_LAST_EPOCH:
            print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")

    # ════════════════════════════════════════════════════════════════
    #  统一train入口（阶段1→2）
    # ════════════════════════════════════════════════════════════════

    def train(self, data_loader):
        """完整训练流程：SCP预训练 → 监督微调。"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        # 阶段1：SCP预训练
        if self.use_scp_pretrain:
            self.pretrain_scp(data_loader)

        # 阶段2：监督微调
        self.finetune_supervised(data_loader)

    # ════════════════════════════════════════════════════════════════
    #  验证
    # ════════════════════════════════════════════════════════════════

    def valid(self, data_loader):
        """验证集评估。"""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("\n ====Validing===")
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_batch in vbar:
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(torch.float32).to(self.device)
                rPPG = self.model(valid_batch[0].to(torch.float32).to(self.device))
                rPPG = _normalize_per_sample(rPPG)
                BVP_label = _normalize_per_sample(BVP_label)
                loss_ecg = self.criterion.neg_pearson(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                vbar.set_postfix(loss=loss_ecg.item())
        return np.mean(valid_loss)

    # ════════════════════════════════════════════════════════════════
    #  阶段3：测试 + FreqTTA
    # ════════════════════════════════════════════════════════════════

    def test(self, data_loader):
        """测试（可选FreqTTA适应）。"""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("===Testing===")
        predictions = dict()
        labels = dict()

        # 加载模型
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error!")
            state_dict = torch.load(self.config.INFERENCE.MODEL_PATH)
            model_ref = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            model_ref.load_state_dict(state_dict)
            print(f"Loaded: {self.config.INFERENCE.MODEL_PATH}")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth'
                )
            else:
                model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth'
                )
            state_dict = torch.load(model_path)
            model_ref = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            model_ref.load_state_dict(state_dict)
            print(f"Loaded: {model_path}")

        self.model = self.model.to(self.config.DEVICE)

        # ── FreqTTA包装 ────────────────────────────────────────
        if self.use_tta:
            print("启用FreqTTA测试时适应...")
            tta_model = FreqTTA(
                model=model_ref,
                lr=1e-4,
                num_steps=3,
                ema_momentum=0.99,
                mode='one_shot',
                adapt_bn=True,
                adapt_gating=True,
                sample_rate=self.frame_rate,
            ).to(self.device)
            tta_model.eval()
        else:
            self.model.eval()

        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)

                if self.use_tta:
                    pred_ppg_test = tta_model.adapt_and_predict(data)
                else:
                    pred_ppg_test = self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print('测试完毕')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

    # ════════════════════════════════════════════════════════════════
    #  工具方法
    # ════════════════════════════════════════════════════════════════

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth'
        )
        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), model_path)
        else:
            torch.save(self.model.state_dict(), model_path)
        print(f'Saved Model Path: {model_path}')

    def get_hr(self, y, sr=30, min_hr=30, max_hr=180):
        """从rPPG信号估计心率。"""
        p, q = welch(y, sr, nfft=1e5 / sr, nperseg=np.min((len(y) - 1, 256)))
        mask = (p > min_hr / 60) & (p < max_hr / 60)
        return p[mask][np.argmax(q[mask])] * 60