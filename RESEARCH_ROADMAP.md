# Infinity-RoPE 研究路线图：从入门到PhD申请 (修订版)

> **目标：** 暑假3个月（~12周），每周30小时，深入理解视频生成与长序列自回归生成方向，为PhD/MPhil申请建立扎实的研究基础。
>
> **起点：** 已完成 Stanford CS231n（CNN + CV 基础）和 MIT 6.S184（Generative AI，含 labs），对深度学习、计算机视觉、生成模型有扎实的实践基础。
>
> **终点：** 能够理解 Infinity-RoPE 的全部技术细节及其在学术版图中的位置，拥有可展示的 side project，并能独立阅读和复现最前沿的视频生成论文。

---

## 零、已完成基础（CS231n + MIT 6.S184）

以下内容已通过课程 labs 动手实现，**不需要重新学，只需在用到时回顾**：

| 已完成 | 来源 | 掌握程度 |
|--------|------|----------|
| ODE/SDE 数值模拟 (Euler, Euler-Maruyama) | 6.S184 Lab 1 | 手写实现过 |
| Brownian motion, OU process, Langevin dynamics | 6.S184 Lab 1 | 手写实现过 |
| Flow Matching (Gaussian + Linear paths) | 6.S184 Lab 2 | 完整实现过 conditional probability paths |
| Score Matching | 6.S184 Lab 2 | 推导过 marginal score 与 marginal flow 的关系 |
| Classifier-Free Guidance (CFG) | 6.S184 Lab 3 | 实现过 CFG 训练循环 |
| Diffusion Transformer (DiT) | 6.S184 Lab 3 | 从零写过 Fourier encoder, patchifier, transformer, depatchifier |
| VAE (Encoder, Decoder, Residual/Attn blocks) | 6.S184 Lab 3 | 从零写过完整 VAE |
| Latent Diffusion Model | 6.S184 Lab 3 | 在 VAE latent space 训练过 DiT |

**这意味着：Flow Matching / Score Matching / SDE 理论 / DiT 架构 / VAE / CFG 这六个模块已经是你的已知领域。**

---

## 一、领域全景图（已掌握标记 ✓）

```
                    ┌─────────────────────────────┐
                    │     Text-to-Video 生成       │
                    │   (Wan2.1, Sora, CogVideo)   │
                    └──────────────┬──────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                         │
  ┌───────▼────────┐    ┌─────────▼──────────┐   ┌─────────▼──────────┐
  │ 扩散/流匹配理论 │    │  因果自回归生成      │   │   模型蒸馏/加速     │
  │ ✓FM ✓Score ✓SDE│    │ CausVid,SelfForcing │   │ DMD2, SiD, CD      │
  │  □DDPM □DDIM   │    └─────────┬──────────┘   └────────────────────┘
  └────────────────┘              │
                    ┌─────────────┼─────────────┐
                    │             │              │
          ┌────────▼──────┐ ┌────▼──────┐ ┌─────▼──────────┐
          │ 位置编码(RoPE) │ │  KV Cache │ │ 高效注意力机制   │
          │ YaRN, NTK      │ │ Attention │ │ FlashAttn,      │
          │                │ │ Sink,SWA  │ │ FlexAttention   │
          └───────────────┘ └───────────┘ └────────────────┘
```

---

## 二、12周详细计划（修订版）

### 第1周：DDPM 补课 + Wan2.1 跑通

**目标：** 补充 DDPM/DDIM 历史视角（你已有 Flow Matching 基础，读起来会很快），跑通 Wan2.1 推理。

> 你已从 6.S184 深入理解了 Flow Matching 的 inference/sampling。DDPM 的加噪-去噪范式是历史路径——理解它有助于看懂早期论文的 related work，但不需要像 FM 那样深入实现。

#### 论文

| # | 论文 | 级别 | 关键点 | 预计时间 |
|---|------|------|--------|----------|
| 1 | **DDPM** (Ho et al., 2020) | **L2** | 扩散模型的原始框架：正向加噪、反向去噪、U-Net、simple loss。**重点对比 DDPM 的 stochastic sampling 和你熟悉的 Flow Matching ODE sampling。** | 6h |
| 2 | **DDIM** (Song et al., 2021) | **L1→L2** | 确定性采样、跳步加速。**你会发现 DDIM 其实就是 ODE 视角下的扩散采样——和你 Lab 2 里的 ODE solver 本质一致。** | 3h |
| 3 | **Score-Based SDE** (Song et al., 2021) | **L1** | 这个论文的理论部分你已经在 Lab 1 里用代码实现过了（Euler-Maruyama, Langevin, SDE→ODE 转换）。快速浏览确认真相。 | 3h |
| 4 | **Wan2.1 技术报告** | **L2** | DiT 架构你已经熟悉。新东西是：3D VAE（空间+时间压缩）、umT5 text encoder、flow matching scheduler 的具体参数。**重点理解 3D VAE 的 temporal compression 是怎么做的。** | 6h |

#### 练手

- [ ] **跑通 Wan2.1 推理**（`Wan-Video/Wan2.1`）——生成 81 帧视频
  - 理解 3D VAE 的 latent shape `(C, T, H, W)`——对比你 Lab 3 的 2D VAE `(C, H, W)`
  - 理解 Flow Matching scheduler——应该和 Lab 2 的 ODE solver 非常眼熟
- [ ] **对比 DDPM sampling vs Flow Matching sampling**：各用 5 行伪代码写出采样循环，标注核心区别
- [ ] **可选：** 如果你想体验一下 DDPM U-Net 的手感，可以在 CIFAR-10 上训练一个极简版 (2h)

**预计投入：** 30h（论文 18h + 动手 12h）

---

### 第2周：视频扩散模型专攻

**目标：** 理解从图像生成到视频生成的核心挑战（时序建模），熟悉 Wan2.1 内部结构。

#### 论文

| # | 论文 | 级别 | 关键点 | 预计时间 |
|---|------|------|--------|----------|
| 5 | **Video Diffusion Models** (Ho et al., 2022) | **L2** | 第一个大规模视频扩散。3D U-Net、factorized space-time attention。**关键 insight：空间 attention 和时间 attention 可以分开做。** | 5h |
| 6 | **Latent Diffusion / Stable Diffusion** (Rombach et al., 2022) | **L1** | 你在 Lab 3 已经实现了 latent diffusion 的核心逻辑。快速浏览论文确认真相即可。 | 3h |
| 7 | **Stable Video Diffusion** (Blattmann et al., 2023) | **L2** | 从图像扩散模型微调出视频模型。**关键 insight：高质量视频模型可以从图像模型初始化（temporal layers 从零学）。** | 5h |
| 8 | **DiT** (Peebles & Xie, 2023) | **L1** | 你在 Lab 3 已经写过 DiT 的 patchifier + transformer + depatchifier。快速对比你写的版本和官方版本，重点关注 adaLN 你 Lab 里可能没实现的部分。 | 3h |

#### 练手

- [ ] **阅读 Wan2.1 模型代码**：`wan/modules/model.py` (bidirectional 版本)
  - 画出完整架构图：3D VAE encoder → latent → DiT blocks (full attention + cross attention) → 3D VAE decoder
  - 标注每个 tensor 的 shape 变化
- [ ] **阅读 Infinity-RoPE 推理入口**：`inference.py` + `pipeline/causal_pipeline.py`
  - 理解 `CausalInferencePipeline` 的整体流程
  - 理解 prompt 解析逻辑（duration、`|`、`#`、`;`）
- [ ] **写一个 Wan2.1 的最小推理脚本**（不依赖 Infinity-RoPE，直接用原版 Wan2.1 生成视频）

**预计投入：** 30h（论文 16h + 动手 14h）

---

### 第3-4周：位置编码 + 因果注意力 + Long Context（Infinity-RoPE 核心 I）

**目标：** 这是 Infinity-RoPE 最关键的技术创新。深入理解 RoPE 及其长序列变体、KV cache 管理策略、因果注意力在视频扩散中的应用。

#### 论文

| # | 论文 | 级别 | 关键点 | 预计时间 |
|---|------|------|--------|----------|
| 9 | **RoPE** (Su et al., 2023) — RoFormer | **L2** | 旋转位置编码的数学推导。复数旋转 ↔ 矩阵形式的等价性。**你 Lab 3 的 DiT 用的是什么 position encoding？对比一下。** | 6h |
| 10 | **YaRN** (Peng et al., 2023) | **L2** | NTK-aware scaling、position interpolation、extrapolation vs interpolation。**理解为什么直接外推 RoPE 会失败。** | 5h |
| 11 | **Flash Attention 1+2** (Dao et al., 2022/2023) | **L1** | IO-aware attention、tiling、recomputation。不用实现，理解为什么快就行。 | 3h |
| 12 | **StreamingLLM** (Xiao et al., 2024) | **L2** | Attention sink 现象 + 滑动窗口 KV cache。**Infinity-RoPE 的 sink_size + local_attn_size 直接来自这篇。** | 5h |
| 13 | **CausVid** (Tian et al., 2024) | **L2** | 第一个将因果注意力引入视频扩散的工作。block-wise causal mask。**Infinity-RoPE 的 causal inference 是 CausVid 的直接进化。** | 6h |
| 14 | **Infinity-RoPE** (本项目, CVPR 2026) | **L3 开始** | Section 3 (Method) 精读两遍：block-relative RoPE、rope_cut、双级 KV eviction、scene transition。 | 12h |

#### 练手

- [ ] **从头实现 RoPE (1D + 2D)**：对比复数形式 vs 矩阵形式
- [ ] **写一个 toy causal transformer + KV cache**：
  - prefill vs decode 阶段的 attention mask 区别
  - sliding window + attention sink eviction
  - 可视化 attention score——观察 attention sink 现象（前 4 个 token 的 attention 权重异常高）
- [ ] **精读 `wan/modules/causal_model.py`**：
  - `block_relativistic_rope` 函数——对比你写的标准 RoPE
  - `rope_cut` 场景切换逻辑
  - `CausalWanSelfAttention.forward` 的 KV cache 增删改查
- [ ] **跑通 Infinity-RoPE 推理**：生成一个 30s+ 的长视频，观察 scene transition 效果

**预计投入：** 60h（论文 37h + 动手 23h）

---

### 第5-6周：模型蒸馏 + 自回归训练（Infinity-RoPE 核心 II）

**目标：** 理解训练方法。Inference-only 的 causal attention hack 很多项目都有，但 Infinity-RoPE 区别于它们的关键是 **data-free self-forcing training**——这让你不是在做 hack，而是在训练一个真正的 causal model。

#### 论文

| # | 论文 | 级别 | 关键点 | 预计时间 |
|---|------|------|--------|----------|
| 15 | **Consistency Models** (Song et al., 2023) | **L2** | 一步生成、consistency distillation/training。**与 DMD 互补的蒸馏思路。** | 6h |
| 16 | **DMD / DMD2** (Yin et al., 2024) | **L2** | KL divergence gradient、real/fake score networks、distribution matching loss。**Infinity-RoPE 训练的核心 loss。** | 8h |
| 17 | **SiD** (Zhou et al., 2024) | **L1** | 了解与 DMD 的对比、SiD loss formulation。Infinity-RoPE 代码里有 SiD 变体 (`model/sid.py`)。 | 3h |
| 18 | **Self-Forcing** (Hu et al., 2025) | **L2** | Backward simulation、data-free training、ODE initialization。**Infinity-RoPE 的训练框架直接基于这篇。** | 8h |
| 19 | **ODE Regression / Causal Forcing** | **L1** | 将 bidirectional 模型通过 ODE regression 转换为 causal 模型的初始 checkpoint。理解为什么需要这一步。 | 3h |

#### 练手

- [ ] **实现一个极简 DMD 训练循环**（2D toy distribution）：
  - 理解 fake_score / real_score 各自扮演的角色
  - 可视化 KL gradient 的方向——这和 score matching 的 gradient 有什么不同？
- [ ] **阅读 Infinity-RoPE 训练代码**：`train.py` + `model/dmd.py`
  - `_consistency_backward_simulation` 的实现
  - gradient normalization (`clip_grad`)
  - FSDP 配置（`torch.distributed`）
- [ ] **尝试小规模训练复现**（如果 GPU >= 24GB）：
  - 用 `configs/dmd.yaml` 跑 50-100 iterations
  - 观察 loss curve (wandb)
  - 对比训练前后的视频质量

**预计投入：** 60h（论文 28h + 动手 32h）

---

### 第7-8周：横向拓展 + 基础设施

**目标：** 了解视频生成领域的版图，同时补足 GPU 工程能力（这对 PhD 面试也很有用）。

#### 论文

| # | 论文 | 级别 | 关键点 | 预计时间 |
|---|------|------|--------|----------|
| 20 | **Sora 技术报告** (OpenAI, 2024) | **L1** | Scaling 思路、spacetime patches | 3h |
| 21 | **CogVideoX** (Zhipu AI, 2024) | **L1** | 开源视频 DiT、3D causal VAE、expert transformer | 4h |
| 22 | **HunyuanVideo** (Tencent, 2024) | **L1** | 开源最强之一、bilingual text encoder | 4h |
| 23 | **LongLive** (NVIDIA, 2025) | **L1** | 直接引用 Infinity-RoPE 的后续工作——说明你的研究在领域内的位置 | 4h |
| 24 | **MovieGen** (Meta, 2024) | **L1** | 音视频联合生成 | 3h |

#### 练手：GPU 工程能力

- [ ] **学习 torch.compile + FlexAttention**：
  - 给 `CausalWanSelfAttention` 的 `forward` 加上 `torch.compile`
  - 理解 `flex_attention` + `BlockMask` 的用法
- [ ] **理解 FSDP**：
  - 阅读 `train.py` 中的 FSDP 配置
  - 写一个 toy FSDP 例子（用 `torch.distributed.fsdp` 包装一个小模型，跑 forward/backward）
- [ ] **了解 FP8 训练**（`torchao`）：
  - 阅读 Infinity-RoPE 中 FP8 的使用方式
  - 如果硬件支持，跑一次 FP8 forward 对比 speed/memory

**预计投入：** 60h（论文 18h + 动手 42h）

---

### 第9-10周：Side Project

**目标：** 选一个方向深入，产出可展示的研究成果。

#### Side Project 选题建议

**难度 ★★★（适合产出 workshop paper，2周可完成）**

- **A. RoPE 变体对比实验**：在视频生成场景下对比 standard RoPE, YaRN, NTK-aware, block-relative RoPE 的长视频质量（FVD, CLIP score, 人工评估）
- **B. KV Cache 策略消融**：对比不同 sink_size / window_size / eviction 策略对 temporal consistency 的影响，找出最优配置

**难度 ★★★★（适合产出 conference submission，4-6周）**

- **C. 将 Causal Attention + KV Cache 迁移到另一个 video DiT（如 CogVideoX）**
  - Core change: modify attention mask + KV cache + block-relative RoPE
  - Verify if it improves long video quality on a different architecture
- **D. 改进 scene transition 机制**
  - 替换 KV flush 为 soft scene boundary（跨场景 information propagation）
  - 加入 text-conditioned scene boundary detection

**难度 ★★★★★（主攻项目，可做整个暑假）**

- **E. Multi-modal action control**：将 action-controllable prompting 扩展到 speech + music
- **F. Efficient distillation**：在 consumer GPU（<24GB）上完成 data-free training（<2 GPU-hours）

> **建议：** 第 7-8 周做完横向拓展后对领域有更全面判断，那时再最终决定 side project 方向。先做 A 或 B 练手（2周能出结果），再看是否 upgrade 到 C/D。

**预计投入：** 60h（论文调研 8h + Side Project 52h）

---

### 第11-12周：输出 + 申请准备

**目标：** 将积累转化为 PhD/MPhil 申请的可见成果。

#### 输出目标

- [ ] **Research Statement 草稿**（2-3 页）
  - 你对 video generation / long-context generative models 的理解
  - Side Project 的 motivation → method → findings
  - Future directions（从你的 Side Project 自然延伸出去的方向最有说服力）

- [ ] **技术博客或 Technical Report**
  - 例如："Understanding Block-Relative RoPE for Autoregressive Video Generation" 或 "A Survey of Causal Attention in Video Diffusion Models"
  - 或将 Side Project 写成 4-6 页 workshop-style paper（arXiv 发布）

- [ ] **GitHub Portfolio**
  - Side Project 代码：README, demo video, config, reproducible pipeline
  - 为 Infinity-RoPE 贡献 PR（找 `good first issue` 或改进文档/修复 bug）

- [ ] **目标导师调研（8-12 位）**
  - 阅读每位导师近 2 年代表作
  - 记录 lab website、招生状态
  - 针对 2-3 位最心仪的导师，读他们的 paper 并思考 "我能在这个 lab 做什么"

#### 推荐研究组

| 学校 | 导师 | 方向 |
|------|------|------|
| Stanford | Stefano Ermon, Fei-Fei Li | Diffusion models, video generation |
| MIT | Phillip Isola, William Freeman | Generative models, image/video synthesis |
| UC Berkeley | Alexei Efros, Angjoo Kanazawa | Video generation, world models |
| CMU | Jun-Yan Zhu, Katerina Fragkiadaki | Controllable generation, video |
| NYU | Saining Xie, Rob Fergus | Diffusion models, flow matching |
| Oxford | Andrew Zisserman, João F. Henriques | Video understanding & generation |
| CUHK | Dahua Lin, Bolei Zhou | Open-source video models (CogVideo) |
| NUS | Mike Zheng Shou | Video generation, Show-1, Tune-A-Video |
| KAIST | Jaegul Choo | Video diffusion, distillation |
| Tsinghua | Jun Zhu | DMD, diffusion theory |
| HKUST | Qifeng Chen | Efficient generative models |

**预计投入：** 60h（输出 40h + 申请准备 20h）

---

## 三、新增：12周时间重新分配对比

| 阶段 | 原计划 | 新计划 | 变化原因 |
|------|--------|--------|----------|
| 扩散基础 | 2周(1-2) | **1周(1)** | 6.S184 已覆盖 FM/Score/SDE/CFG |
| 视频扩散 | 2周(3-4) | **1周(2)** | 6.S184 已覆盖 DiT/VAE/LDM |
| RoPE + Causal Attn | 2周(5-6) | **2周(3-4)** | 无变化，这是核心 |
| 蒸馏 + 训练 | 2周(7-8) | **2周(5-6)** | 无变化，这是核心 |
| 横向 + 工程 | 2周(9-10) | **2周(7-8)** | 新增 GPU 工程能力板块 |
| Side Project | (含在9-10) | **2周(9-10)** | 独立出来 |
| 输出 + 申请 | 2周(11-12) | **2周(11-12)** | 无变化 |

**关键变化：** 原计划前 4 周被压缩到 2 周，释放了 2 周时间——1 周给了 GPU 工程能力（第7-8周有了更多时间），1 周给了 Side Project（第9-10周更专注）。

---

## 四、论文分级总览

| 级别 | 数量 | 说明 |
|------|------|------|
| **L3** (批判复现) | 1 | Infinity-RoPE |
| **L2** (方法掌握) | 11 | DDPM, Wan2.1, VDM, SVD, RoPE, YaRN, StreamingLLM, CausVid, Consistency Models, DMD2, Self-Forcing |
| **L1** (框架理解) | 12 | DDIM, Score-SDE, FlashAttn, DiT, LD/SD, SiD, ODE Reg, Sora, CogVideoX, HunyuanVideo, LongLive, MovieGen |

论文阅读总时间：约 110h（原计划 165h，节省的 55h 来自已掌握内容的 L1 化）

---

## 五、开源项目清单

### 核心项目（必须跑通）

| 项目 | Repo | 用途 |
|------|------|------|
| **Infinity-RoPE** | 当前 repo | 主线 |
| **Wan2.1** | `Wan-Video/Wan2.1` | 基座模型 |
| **Self-Forcing** | `guandeh17/Self-Forcing` | 训练框架前身 |
| **CausVid** | `tianweiy/CausVid` | 因果视频扩散先驱 |
| **diffusers** | `huggingface/diffusers` | 扩散模型标准库 |

### 练习项目（提升代码能力）

| 项目 | 用途 |
|------|------|
| `lucidrains/denoising-diffusion-pytorch` | DDPM 教学实现（作为参照对比你的 FM 理解） |
| `facebookresearch/DiT` | 对比你 Lab 3 的实现和官方实现 |
| `Dao-AILab/flash-attention` | Flash Attention 官方实现 |

### 前沿项目（了解趋势）

| 项目 | Repo | 用途 |
|------|------|------|
| CogVideoX | `THUDM/CogVideo` | Side Project C 的迁移目标 |
| HunyuanVideo | `Tencent/HunyuanVideo` | 横向对比 |
| LongLive | `NVlabs/LongLive` | Infinity-RoPE 后续 |
| Cosmos | `NVIDIA/Cosmos` | NVIDIA 世界模型 |

---

## 六、每周作息建议

```
周一-周五：每天 5h（上午 2.5h 精读论文/推导公式，下午 2.5h 动手写代码）
周六：     5h 总结本周所学、写笔记、更新 GitHub
周日：     休息或机动

每周笔记模板（存到 GitHub repo 的 /notes 目录）：
  - 本周读了哪些论文（一句话总结核心 idea）
  - 推导了哪些公式（拍照手写推导过程）
  - 代码练习进展（附 commit link）
  - 本周困惑 / 下周计划
```

---

## 七、里程碑检查点

| 时间 | 检查内容 |
|------|----------|
| 第1周末 | 能画出 Wan2.1 完整架构图；能写出 DDPM 和 Flow Matching 的采样伪代码并标注核心区别 |
| 第2周末 | 能解释 3D VAE 和 2D VAE 的本质区别；跑通 Wan2.1 推理，生成第一个视频 |
| 第4周末 | 能手写 RoPE 公式和 KV cache eviction 算法；跑通 Infinity-RoPE 推理，生成 30s+ 长视频 |
| 第6周末 | 能解释 DMD KL gradient 与 score matching 的区别；理解 backward simulation 消除 train-test mismatch 的原理 |
| 第8周末 | 能用 torch.compile / FlexAttention / FSDP 做基本操作；读完所有 L1 论文，对视频生成领域有全局认知 |
| 第10周末 | Side Project 有初步实验数据（至少一条有意义的 insight） |
| 第12周末 | 完成 Research Statement + 技术博客；确定目标导师名单 |

---

## 八、工具与资源

- **论文管理：** Zotero + annotated bibliography
- **数学推导：** iPad/手写笔记，重点推导（RoPE rotation, KL gradient, flow ODE）
- **GPU：** RTX 4090 (24GB) 跑推理 + 小规模训练；集群/cloud 用于大训练
- **社区：** Twitter/X (follow video gen researchers), GitHub Discussions, Discord (diffusers, CUDA MODE)
- **会议跟踪：** arXiv cs.CV daily, CVPR/ICCV/NeurIPS/ICLR proceedings
- **推荐课程（补充视角）：**
  - Stanford CS236 (Deep Generative Models) — 讲 diffusion 的部分
  - CUDA MODE — GPU 编程与性能优化（第7-8周看）

---

## 九、从 Side Project 到论文

1. **Analysis paper (最可行，2-4周)：** 系统性消融实验，发现反直觉结论。投 CVPR/ICCV workshop。
2. **Method paper (有挑战，4-8周)：** 迁移 block-relative RoPE + causal attention 到新 domain。
3. **Efficiency paper (实用，4-8周)：** 降低 data-free distillation 训练成本到 consumer GPU 可承受范围。

*先用 2 周 exploration（跑 baseline、确认 idea 可行），再决定主攻方向。*

---

> **你的优势：** CS231n + MIT 6.S184 的组合让你在 diffusion/flow matching/DiT 这些基础上比大多数暑期实习生强。不要浪费这个基础——不要在第 1-2 周重复你已经会的东西，直接跳到你还不会的（视频 + RoPE + causal attention + distillation）。三个月后，你应该成为视频生成领域最懂 RoPE 和 causal attention 的人之一。Good luck!
