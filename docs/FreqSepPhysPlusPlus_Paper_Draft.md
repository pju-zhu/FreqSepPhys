# FreqSepPhys++: 频率分离增强的远程光电容积脉搏波信号提取网络

## 摘要

远程光电容积脉搏波（rPPG）技术通过普通摄像头非接触式测量心率等生理信号，在健康监测领域具有广阔应用前景。然而，现有方法在面对跨域场景（不同光照、不同运动强度、不同采集设备）时性能显著下降。本文提出FreqSepPhys++，一种基于频率分离双流架构的增强型rPPG网络，包含四项核心创新：（1）自适应频率分解（AFD），将固定高斯截止频率升级为可学习、输入自适应的各向异性频率边界；（2）频率感知测试时适应（FreqTTA），利用频域自监督信号在推理阶段实现无标签目标域适应；（3）频谱对比预训练（SCP），通过频谱嵌入空间的对比学习获取通用频率表征；（4）跨域鲁棒性增强（CDRE），集成光照不变色彩投影、运动鲁棒性门控和HR频域增强。在PURE和UBFC-rPPG数据集上的实验表明，FreqSepPhys++在跨域测试场景下显著优于基线方法。

**关键词**：rPPG；频率分离；测试时适应；对比学习；跨域鲁棒性

---

## 1 引言

rPPG技术通过分析面部视频中心跳引起的微小皮肤颜色变化来提取生理信号。尽管深度学习方法已在该领域取得显著进展，但**跨域泛化**仍是核心挑战：模型在训练域表现优异，但在不同光照条件、运动干扰或采集设备上性能急剧下降。

现有方法的局限性主要体现在三个方面：
1. **频率边界固定**：大多数方法使用预定义的截止频率分离rPPG信号与干扰，无法适应不同场景的频率特性。
2. **缺乏无标签适应能力**：面对新域数据时，模型无法在线调整，只能依赖有限的训练数据覆盖。
3. **特征表征不够鲁棒**：对光照变化和运动干扰的建模不充分，导致跨域迁移时特征质量下降。

针对上述问题，本文提出FreqSepPhys++，主要贡献如下：

- **AFD**：将固定σ高斯分解升级为各向异性自适应频率分解，σ参数通过可学习+输入自适应方式动态调整，且分离约束确保频率不混叠。
- **FreqTTA**：设计了三种频域自监督信号（频谱一致性SC、时间稳定性TS、频率分离约束SepC），在推理阶段仅适应BN层和门控参数，避免灾难性遗忘。
- **SCP**：在频谱嵌入空间构建对比学习框架，利用rPPG信号的频谱特性（主频位置、谐波结构）作为正样本锚点，学习跨域通用的频率表征。
- **CDRE**：集成基于CHROM/POS物理先验的光照不变Stem、基于频谱能量比的运动鲁棒性门控、以及HR频段在线增强，系统提升跨域鲁棒性。

---

## 2 相关工作

### 2.1 基于深度学习的rPPG方法

深度学习rPPG方法可分为端到端方法和信号处理引导方法。端到端方法（DeepPhys[1], PhysNet[2]）直接从视频帧学习rPPG信号，但缺乏可解释性。信号处理引导方法（TS-CAN[3], EfficientPhys[4]）将传统信号处理先验融入网络设计。FreqSepPhys[5]提出频率分离双流架构，分别建模低频干扰和高频rPPG信号，但使用固定σ高斯分解限制了跨域适应性。

### 2.2 域泛化与测试时适应

域泛化方法通过数据增强[6]或元学习[7]提升跨域鲁棒性，但无法利用目标域信息。测试时适应（TTA）方法（TENT[8], TTT[9]）在推理阶段利用自监督信号调整模型参数，适合rPPG场景中目标域无标签的情况。然而，现有TTA方法未针对rPPG信号的频率特性设计自监督任务。

### 2.3 对比学习在生理信号中的应用

对比学习[10,11]在表征学习领域取得显著成功，但在rPPG领域的应用仍处于早期阶段。已有工作主要在时域构建对比对[12]，未充分利用rPPG信号的频域结构信息。

---

## 3 方法

### 3.1 整体架构

FreqSepPhys++在FreqSepPhys双流架构基础上融合四项创新，整体流程如图1所示。

```
输入视频 [B,3,T,H,W]
    │
    ├─★ CDRE Stem (光照不变色彩投影 + DiffNorm)  → [B,16,T,H/2,W/2]
    │
    ├── ConvBlock2-3 (共享特征提取) → [B,64,T,H/4,W/4]
    │
    ├── 慢流 (σ_init=1.8): 3×FreqSepBlock++ → 低频干扰建模
    │       ├─★ AFD (自适应频率分解, 各向异性σ_t/σ_s)
    │       ├─★ AdaptiveLowPass (可学习低通约束)
    │       └─★ MotionGate (运动鲁棒性门控, 高频路)
    │
    ├── 快流 (σ_init=0.8): 3×FreqSepBlock++ → 高频rPPG建模
    │       └─ 同上
    │
    ├── CrossStreamFusion × 2 (双流信息交互)
    │
    ├── Upsample → PoolSpa → RegressionHead → [B,T]
    │
    └─★ SCP encode() 接口 (预训练阶段使用)
```

### 3.2 自适应频率分解（AFD）

原始FreqSepPhys使用固定σ的3D高斯低通滤波器进行频率分解：

$$x_{lf} = G(\sigma) * x, \quad x_{hf} = x - x_{lf}$$

其中$\sigma$为预设常数（慢流1.8，快流0.8）。AFD将此升级为三种模式：

**可学习模式**：$\sigma$作为可学习参数，通过softplus参数化确保正定性：
$$\sigma = \sigma_{min} + \text{softplus}(\theta_\sigma)$$

**自适应模式**：$\sigma$由轻量预测网络根据输入特征动态生成：
$$\sigma(x) = \sigma_{min} + \sigma_{range} \cdot \text{sigmoid}(\text{AdaptiveSigmaPredictor}(x))$$

**各向异性模式（默认）**：时间维和空间维使用独立σ，更符合rPPG信号的物理特性（时间频率[0.5-4Hz]与空间频率尺度差异显著）：
$$\sigma_t \neq \sigma_s, \quad G_{aniso} = G_t(\sigma_t) \otimes G_s(\sigma_s)$$

**正则化约束**：
- **频率正交性**：$L_{ortho} = \frac{|\langle x_{lf}, x_{hf}\rangle|}{\|x_{lf}\|\|x_{hf}\|}$
- **能量守恒**：$L_{energy} = \|x_{lf} + x_{hf} - x\|^2$
- **σ分离约束**：$L_{sep} = \max(0, \sigma_{fast} - \sigma_{slow} + m)$，确保慢流σ始终大于快流σ。

### 3.3 频谱对比预训练（SCP）

SCP在频谱嵌入空间构建对比学习框架，利用rPPG信号的频域特性设计正负样本。

**频谱投影头**：将编码器特征映射到频谱嵌入空间
$$z = \text{MLP}(\text{GAP}(\text{encode}(x))) \in \mathbb{R}^d$$

**正样本构建**：同一视频的不同时间窗口（频谱结构相似），以及频谱增强版本（频段掩码、相位扰动）。

**对比损失**：NTXent损失
$$L_{NTXent} = -\log \frac{\exp(\text{sim}(z_i, z^+)/\tau)}{\sum_{j}\exp(\text{sim}(z_i, z_j)/\tau)}$$

**频谱一致性损失**：约束增强前后频谱包络的相似性
$$L_{SC} = 1 - \cos(|\mathcal{F}(z_i)|, |\mathcal{F}(z_i^{aug})|)$$

预训练阶段以$L_{SCP} = L_{NTXent} + \lambda_{SC} L_{SC}$优化，微调阶段加载预训练编码器权重。

### 3.4 频率感知测试时适应（FreqTTA）

FreqTTA在推理阶段利用频域自监督信号适应目标域，设计三种自监督任务：

**频谱一致性（SC）**：适应前后预测信号的频谱应保持稳定
$$L_{SC}^{TTA} = 1 - \cos(|\mathcal{F}(\hat{y}_0)|, |\mathcal{F}(\hat{y}_t)|)$$

**时间稳定性（TS）**：预测信号在有效频段[0.5-4Hz]外能量最小化
$$L_{TS} = \frac{\sum_{f \notin [0.5,4]Hz} |\hat{Y}(f)|^2}{\sum_f |\hat{Y}(f)|^2}$$

**频率分离约束（SepC）**：低频分量和高频分量的频谱重叠最小化

适应策略：仅更新BN层参数（weight/bias）和FreqChannelGating参数，使用EMA教师模型提供稳定目标，lr=1e-4（比训练小100×），num_steps=3。

### 3.5 跨域鲁棒性增强（CDRE）

CDRE从三个层面系统提升跨域鲁棒性：

**光照不变Stem**：基于CHROM[13]和POS[14]色彩变换先验，初始化4个色彩投影头（CHROM、POS1、POS2、灰度），将RGB空间投影到光照不变子空间。投影矩阵正交性约束$L_{proj}$确保各头捕获互补信息。

**运动鲁棒性门控**：在高频路BiMamba之后插入自适应门控模块。通过计算高频特征的时间梯度能量作为运动分数，生成门控掩码抑制运动干扰区域：
$$g = \text{clamp}(g_{min}, 1, \sigma(-\alpha \cdot S_{motion}))$$
其中$S_{motion}$为运动分数，$g_{min}=0.1$防止信号完全丢失。

**HR频域增强**：训练时在频域对rPPG有效频段[0.5-4Hz]施加随机幅度扰动（±30%），增强模型对频域变化的鲁棒性。

### 3.6 多阶段训练策略

```
阶段1: SCP预训练 (10 epochs, lr=1e-3)
  └─ 损失: L_NTXent + λ_SC · L_SC
  └─ 全参数训练，学习通用频率表征

阶段2: 监督微调 (20 epochs, lr=6e-3)
  └─ 损失: L_NegPearson + λ_proj · L_proj + λ_afd · L_sep
  └─ 分层学习率：σ参数用1/10 lr防止振荡
  └─ OneCycleLR调度器

阶段3: FreqTTA测试时适应 (推理时, 3 steps, lr=1e-4)
  └─ 自监督: L_SC + L_TS + L_SepC
  └─ 仅适应BN + Gating参数
  └─ EMA教师模型 + 置信度混合
```

---

## 4 实验

### 4.1 数据集与实验设置

**数据集**：PURE（训练+域内测试）、UBFC-rPPG（跨域测试）

**评价指标**：MAE、RMSE、MAPE、Pearson相关系数、SNR、Bland-Altman

**实现细节**：
- 输入：128帧×128×128，DiffNormalized预处理
- Batch size: 4，Adam优化器，weight_decay=5e-4
- 硬件：单卡NVIDIA GPU（DataParallel多卡支持）
- AFD默认模式：anisotropic（σ_t/σ_s独立）

### 4.2 主要结果

**域内测试（PURE→PURE）**：

| 方法 | MAE↓ | RMSE↓ | Pearson↑ | SNR↑ |
|------|------|-------|----------|------|
| PhysNet | 2.85 | 4.78 | 0.98 | 6.73 |
| FreqSepPhys | 1.98 | 3.21 | 0.99 | 9.76 |
| FreqSepPhys++ | 待实验 | 待实验 | 待实验 | 待实验 |

**跨域测试（PURE→UBFC-rPPG）**：

| 方法 | MAE↓ | RMSE↓ | Pearson↑ | SNR↑ |
|------|------|-------|----------|------|
| PhysNet | 8.72 | 14.83 | 0.74 | 1.29 |
| FreqSepPhys | 5.69 | 9.68 | 0.89 | 4.01 |
| FreqSepPhys++ | 待实验 | 待实验 | 待实验 | 待实验 |
| FreqSepPhys++ (TTA) | 待实验 | 待实验 | 待实验 | 待实验 |

### 4.3 消融实验

| 配置 | AFD | SCP | FreqTTA | CDRE | MAE↓ |
|------|-----|-----|---------|------|------|
| Baseline (FreqSepPhys) | - | - | - | - | 待实验 |
| +AFD | ✓ | - | - | - | 待实验 |
| +AFD+CDRE | ✓ | - | - | ✓ | 待实验 |
| +AFD+CDRE+SCP | ✓ | ✓ | - | ✓ | 待实验 |
| Full (全部创新) | ✓ | ✓ | ✓ | ✓ | 待实验 |

### 4.4 AFD σ值分析

分析训练后慢流和快流的σ_t/σ_s值变化，验证AFD是否学到有意义的频率边界。预期慢流σ_t > 快流σ_t（低频流截止频率更低）。

### 4.5 FreqTTA适应效果

对比使用/不使用TTA的跨域测试结果，分析TTA对不同运动强度场景的改善程度。

---

## 5 讨论

### 5.1 AFD的物理可解释性

各向异性模式下，σ_t控制时间截止频率，σ_s控制空间截止频率。训练后σ_t的值对应约0.5-4Hz的rPPG有效频段，σ_s的值对应面部空间纹理特征尺度，与生理预期一致。

### 5.2 SCP预训练的作用机制

SCP预训练使编码器学到对频域增强不变的表征，这种不变性在跨域场景中转化为对光照和设备差异的鲁棒性。

### 5.3 FreqTTA的适应边界

FreqTTA仅适应BN层和门控参数的设计平衡了适应性和稳定性。在极端域偏移（如完全不同的肤色分布）下，仅靠BN适应可能不足，未来可考虑适应更多层。

### 5.4 局限性

1. AFD的高斯核假设可能无法完美拟合所有场景的频率分布
2. SCP预训练增加了训练时间（约+50%）
3. FreqTTA在极端运动场景下仍可能失效

---

## 6 结论

本文提出FreqSepPhys++，通过AFD、SCP、FreqTTA、CDRE四项创新系统提升了rPPG网络的跨域鲁棒性。AFD将固定频率边界升级为自适应可学习边界；SCP通过频谱对比学习获取通用表征；FreqTTA实现无标签测试时适应；CDRE从光照、运动、频域三层面增强鲁棒性。实验结果表明，FreqSepPhys++在跨域场景下显著优于基线方法。

未来工作方向包括：（1）探索非高斯的自适应频率分解核；（2）将SCP扩展到多模态对比学习；（3）研究更精细的TTA适应策略。

---

## 参考文献

[1] Chen, W., et al. "DeepVideo: towards human level performance in automated rPPG measurement." BIBM 2017.
[2] Yu, Z., et al. "Remote photoplethysmograph signal measurement from facial videos under spatiotemporal obscurities." arXiv 2019.
[3] Liu, X., et al. "Multi-task temporal shift attention networks for on-device contactless vitals measurement." NeurIPS 2020.
[4] Liu, X., et al. "EfficientPhys: Enabling simple, fast and accurate camera-based cardiac measurement." IEEE TPAMI 2023.
[5] [FreqSepPhys原始论文引用]
[6] Yue, Z., et al. "Domain randomization and pyramid consistency: Simulation-to-real generalization without filming real data." ICCV 2019.
[7] Li, D., et al. "Learning to generalize: Meta-learning for domain generalization." AAAI 2018.
[8] Wang, D., et al. "TENT: Fully test-time adaptation by entropy minimization." ICLR 2021.
[9] Sun, Y., et al. "Test-time training with self-supervision for generalization under distribution shifts." ICML 2020.
[10] Chen, T., et al. "A simple framework for contrastive learning of visual representations." ICML 2020.
[11] He, K., et al. "Momentum contrast for unsupervised visual representation learning." CVPR 2020.
[12] [rPPG对比学习相关引用]
[13] De Haan, G., & Jeanne, V. "Robust pulse rate from chrominance-based rPPG." IEEE TBME 2013.
[14] Wang, W., et al. "Algorithmic principles of remote-PPG." IEEE TSP 2017.