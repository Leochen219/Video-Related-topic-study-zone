# Infinity-RoPE 研究路线图：从入门到PhD申请

> **目标：** 暑假3个月（~12周），每周30小时，深入理解视频生成与长序列自回归生成方向，为PhD/MPhil申请建立扎实的研究基础。
>
> **起点：** 已完成 Stanford CS231n（含 labs），对深度学习与计算机视觉有基本认知。
>
> **终点：** 能够理解 Infinity-RoPE 的全部技术细节及其在学术版图中的位置，拥有可展示的 side project，并能独立阅读和复现最前沿的视频生成论文。

---

## 一、领域全景图

Infinity-RoPE (CVPR 2026) 解决的问题是 **无限长、可控的视频生成**，其技术栈横跨以下子方向：

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
  │ DDPM, FM, SDE  │    │ CausVid,SelfForcing │   │ DMD2, SiD, CD      │
  └────────────────┘    └─────────┬──────────┘   └────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │              │
          ┌────────▼──────┐ ┌────▼──────┐ ┌─────▼──────────┐
          │ 位置编码(RoPE) │ │  KV Cache │ │ 高效注意力机制   │
          │ YaRN, NTK      │ │ Attention │ │ FlashAttn,      │
          │                │ │ Sink,SWA  │ │ FlexAttention   │
          └───────────────┘ └───────────┘ └────────────────┘
```

---

## 二、12周详细计划

### 第1-2周：扩散模型基础 + 动手热身

**目标：** 从 CS231n 的 CNN 基础过渡到生成模型，深入理解扩散模型原理。

#### 必读论文（按顺序）

| # | 论文 | 关键点 | 预计时间 |
|---|------|--------|----------|
| 1 | **DDPM** (Ho et al., 2020) — Denoising Diffusion Probabilistic Models | 扩散模型的原始框架：正向加噪、反向去噪、U-Net、simple loss | 8h |
| 2 | **DDIM** (Song et al., 2021) — Denoising Diffusion Implicit Models | 确定性采样、加速采样、与score-based的联系 | 5h |
| 3 | **Score-Based SDE** (Song et al., 2021) — Score-Based Generative Modeling through SDEs | SDE/ODE统一框架、PC sampler、概率流 | 8h |
| 4 | **Flow Matching** (Lipman et al., 2023) — Flow Matching for Generative Modeling | 连续归一化流、条件流匹配、与扩散模型的联系 | 8h |
| 5 | **Classifier-Free Guidance** (Ho & Salimans, 2022) | CFG公式推导，理解guidance scale | 3h |

#### 练手项目

- [ ] **从头实现 DDPM**（参考 `lucidrains/denoising-diffusion-pytorch`）：在 CIFAR-10 上训练一个小型 U-Net 扩散模型
  - 理解 noise schedule、timestep embedding、U-Net 结构
  - 实现 DDPM 采样和 DDIM 采样，对比效果
  - 实现 CFG sampling

- [ ] **阅读 diffusers 库源码**：选读 `DDPMPipeline`, `DDPMScheduler`, `UNet2DModel`
  - Repo: `huggingface/diffusers`

- [ ] **阅读 CS231n 后续课程**：Stanford CS236 (Deep Generative Models) 前几讲

**预计投入：** 60h（论文 32h + 动手 28h）

---

### 第3-4周：视频扩散模型 + Wan2.1 深入

**目标：** 理解将图像扩散模型扩展到视频的核心挑战（时序一致性、3D attention），跑通 Wan2.1 推理。

#### 必读论文

| # | 论文 | 关键点 | 预计时间 |
|---|------|--------|----------|
| 6 | **Video Diffusion Models** (Ho et al., 2022) | 首个大规模视频扩散模型、3D U-Net、factorized attention | 6h |
| 7 | **Latent Diffusion / Stable Diffusion** (Rombach et al., 2022) | VAE latent space、cross-attention conditioning | 6h |
| 8 | **Stable Video Diffusion** (Blattmann et al., 2023) | 从图像扩散模型微调到视频、latent video training | 5h |
| 9 | **Wan2.1** (Alibaba, 2025) — 官方技术报告 | DiT (Diffusion Transformer) 架构、3D VAE、umT5 text encoder、flow matching scheduler | 8h |
| 10 | **DiT** (Peebles & Xie, 2023) — Scalable Diffusion Models with Transformers | Transformer替代U-Net做扩散、adaLN、patchify | 6h |

#### 练手项目

- [ ] **跑通 Wan2.1 推理**（`Wan-Video/Wan2.1`）：在自己的 GPU 上生成 81 帧视频
  - 理解 3D VAE 的 latent 表示 (C, T, H, W)
  - 理解 Flow Matching scheduler 的采样过程
  - 画出 forward/reverse process 的框架图

- [ ] **阅读 Infinity-RoPE 推理代码**：`inference.py` + `pipeline/causal_pipeline.py`
  - 理解 `CausalInferencePipeline` 的整体流程
  - 理解 prompt parsing (时长、场景切换)

- [ ] **实现一个简单的 DiT block**（参考 DiT 论文 Figure 3）
  - adaLN-modulation / adaLN-zero
  - 理解 patch embedding + position embedding

**预计投入：** 60h（论文 31h + 动手 29h）

---

### 第5-6周：位置编码 + 因果注意力 + Long Context

**目标：** 这是 Infinity-RoPE 的核心创新之一。深入理解 RoPE 及其变体，以及如何让 transformer 处理超长序列。

#### 必读论文

| # | 论文 | 关键点 | 预计时间 |
|---|------|--------|----------|
| 11 | **RoPE** (Su et al., 2023) — RoFormer: Enhanced Transformer with Rotary Position Embedding | 旋转位置编码的数学推导、与绝对/相对位置编码的对比 | 6h |
| 12 | **YaRN** (Peng et al., 2023) — Yet another RoPE extensioN method | NTK-aware scaling、位置插值、外推 vs 内插 | 5h |
| 13 | **StreamingLLM** (Xiao et al., 2024) — Efficient Streaming Language Models with Attention Sinks | Attention sink 现象、KV cache 滑动窗口 + sink 策略 | 5h |
| 14 | **Flash Attention** (Dao et al., 2022) + **Flash Attention 2** (Dao, 2023) | IO-aware attention、tiling、重计算 | 6h |
| 15 | **CausVid** (Tian et al., 2024) — Causal Video Diffusion with Temporal Attention | 第一个将因果注意力引入视频扩散模型的工作，block-wise causal mask | 6h |
| 16 | **Infinity-RoPE** (本项目, CVPR 2026) — 论文精读 | block-relative RoPE、scene cut时的position reset、双级KV eviction | 10h |

#### 练手项目

- [ ] **从头实现 RoPE**：在 PyTorch 中实现 1D 和 2D RoPE
  - 理解复数旋转的等价矩阵形式
  - 验证 `rope_apply` 与标准实现的等价性

- [ ] **实现简易 KV Cache**：搭建一个 toy causal transformer
  - 实现 prefill vs decode 阶段
  - 实现 sliding window attention + attention sink eviction
  - 可视化 attention score 分布（观察 attention sink 现象）

- [ ] **阅读 Infinity-RoPE 核心代码**：`wan/modules/causal_model.py`
  - `block_relativistic_rope` 函数
  - `rope_cut` 场景切换逻辑
  - `CausalWanSelfAttention.forward` 的 KV cache 管理

**预计投入：** 60h（论文 38h + 动手 22h）

---

### 第7-8周：模型蒸馏 + 自回归训练

**目标：** 理解 Infinity-RoPE 的训练方法（data-free distillation + self-forcing），这是该工作与单纯 inference hack 的分水岭。

#### 必读论文

| # | 论文 | 关键点 | 预计时间 |
|---|------|--------|----------|
| 17 | **Consistency Models** (Song et al., 2023) | 一步生成、consistency distillation/training、与扩散模型的关系 | 6h |
| 18 | **DMD / DMD2** (Yin et al., 2024) — Distribution Matching Distillation | KL divergence gradient、两个score network (real/fake)、distribution matching loss | 8h |
| 19 | **SiD** (Zhou et al., 2024) — Score identity Distillation | 与 DMD 对比、SiD loss formulation | 5h |
| 20 | **Self-Forcing** (Hu et al., 2025) — 代码与论文 | backward simulation、data-free training、ODE initialization | 8h |
| 21 | **ODE Regression / iDEM** — 相关技术报告 | 将bidirectional模型转换为一阶ODE solver的初始checkpoint | 4h |

#### 练手项目

- [ ] **实现一个极简 DMD 训练循环**：在 2D toy distribution 上
  - 理解 fake_score / real_score 的作用
  - KL gradient 的直观理解

- [ ] **阅读 Infinity-RoPE 训练代码**：`train.py` + `model/dmd.py`
  - 理解 backward simulation 的实现
  - 理解 gradient normalization (`clip_grad`)
  - 理解 FSDP 分布式训练配置

- [ ] **尝试复现小规模训练**（如果 GPU 足够，至少24GB VRAM）
  - 用 `configs/dmd.yaml` 跑一小段训练
  - 观察 loss curve、track wandb metrics

**预计投入：** 60h（论文 31h + 动手 29h）

---

### 第9-10周：横向拓展 + Side Project

**目标：** 了解视频生成领域的其他重要工作，选择一个方向深入做 mini-project。

#### 横向必读论文

| # | 论文 | 关键点 | 预计时间 |
|---|------|--------|----------|
| 22 | **Sora 技术报告** (OpenAI, 2024) — Video Generation Models as World Simulators | 大规模视频生成的 scaling 思路、时空 patch | 3h |
| 23 | **CogVideoX** (Zhipu AI, 2024) | 开源视频 DiT、3D causal VAE、expert transformer | 5h |
| 24 | **MovieGen** (Meta, 2024) | 音视频联合生成、flow matching | 4h |
| 25 | **HunyuanVideo** (Tencent, 2024) | 开源最强之一、bilingual text encoder、3D VAE设计 | 5h |
| 26 | **LongLive** (NVIDIA, 2025) — 引用 Infinity-RoPE 的后续工作 | 长到无限视频生成 | 5h |

#### Side Project 选题建议（选一个）

**难度 ★★★ (适合产出 workshop paper)**
- **A. RoPE 变体对比实验**：在视频生成场景下对比 standard RoPE, YaRN, NTK-aware, block-relative RoPE 的长视频质量
- **B. KV Cache 策略消融**：对比不同 sink size / window size / eviction 策略对长视频一致性（CLIP score, FVD）的影响

**难度 ★★★★ (适合产出 conference submission)**
- **C. 将 Causal Attention + KV Cache 迁移到另一个 video DiT（如 CogVideoX）**
  - 核心改动：修改 attention mask + 实现 KV cache + block-relative RoPE
  - 验证是否可以提升该模型的 long video quality
- **D. 改进 scene transition 机制**
  - 替换简单的 KV flush 为更平滑的 cross-scene information propagation
  - 加入 text-conditioned scene boundary detection

**难度 ★★★★★ (适合主攻项目)**
- **E. 将 action-controllable prompting 扩展到多模态（speech + text + music）**
- **F. 设计更高效的蒸馏方法以减少训练开销（<2 GPU-hours on consumer GPU）**

**预计投入：** 60h（论文 22h + Side Project 38h）

---

### 第11-12周：总结输出 + 申请准备

**目标：** 将积累的知识和研究结果转化为 PhD/MPhil 申请的可见成果。

#### 输出目标

- [ ] **Research Statement 草稿**（2-3页）
  - 陈述对 video generation / long-context generative models 的理解
  - 总结 Side Project 的动机、方法、发现
  - 展望未来研究方向（如：real-time interactive video generation, 3D-consistent long video, embodied video generation）

- [ ] **技术博客或 Technical Report**
  - 写一篇关于 "Understanding Block-Relative RoPE for Autoregressive Video Generation" 的深度技术文章
  - 或者将 Side Project 写成 4-6 页的 workshop-style paper

- [ ] **GitHub Portfolio 整理**
  - 确保 Side Project 代码有 README、demo video、config 说明
  - 为 Infinity-RoPE 贡献代码（找 issues labeled `good first issue` 或改进文档）

- [ ] **目标导师调研**
  - 列出 8-12 位在 video generation / generative models 方向活跃的导师
  - 阅读每位导师近 2 年的代表性工作
  - 记录他们的 lab website、招生状态

#### 推荐关注的研究组

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

## 三、开源项目清单（日常参考）

### 核心项目（必须熟悉）

| 项目 | Repo | 用途 |
|------|------|------|
| **Infinity-RoPE** | `hongyu-chen03/infinity-rope` (当前) | 主线项目，反复阅读 |
| **Wan2.1** | `Wan-Video/Wan2.1` | 基座模型 |
| **Self-Forcing** | `guandeh17/Self-Forcing` | 训练框架前身 |
| **CausVid** | `tianweiy/CausVid` | 因果视频扩散先驱 |
| **diffusers** | `huggingface/diffusers` | 扩散模型标准库 |

### 练习项目（入门练手）

| 项目 | Repo | 用途 |
|------|------|------|
| DDPM PyTorch | `lucidrains/denoising-diffusion-pytorch` | DDPM 教学实现 |
| DiT | `facebookresearch/DiT` | Diffusion Transformer 官方实现 |
| Flow Matching | `facebookresearch/flow_matching` | Meta 的流匹配实现 |

### 前沿项目（了解趋势）

| 项目 | Repo | 用途 |
|------|------|------|
| CogVideoX | `THUDM/CogVideo` | 开源视频 DiT |
| HunyuanVideo | `Tencent/HunyuanVideo` | 开源视频 DiT |
| Open-Sora | `hpcaitech/Open-Sora` | Sora 复现 |
| Cosmos | `NVIDIA/Cosmos` | NVIDIA 世界模型 |
| LongLive | `NVlabs/LongLive` | Infinity-RoPE 后续 |
| FastVideo | `hao-ai-lab/FastVideo` | 视频扩散加速 |

---

## 四、每周作息建议

```
周一-周五：每天 5h（上午 2.5h 精读论文/推导公式，下午 2.5h 动手写代码）
周六：     5h 总结本周所学、写笔记、更新 GitHub
周日：     休息或机动

每周笔记模板：
  - 本周读了哪些论文（1-2句话总结核心idea）
  - 推导了哪些公式（附手写笔记照片）
  - 代码练习进展（附GitHub commit link）
  - 本周困惑/下周计划
```

---

## 五、里程碑检查点

| 时间 | 检查内容 |
|------|----------|
| 第2周末 | 能完整推导 DDPM 的 ELBO 和 DDIM 的跳步采样公式；CIFAR-10 DDPM 模型能生成看得过去的样本 |
| 第4周末 | 能画出 Wan2.1 完整架构图（VAE → DiT blocks → text encoder）；理解 Flow Matching 与 DDPM 的数学关系 |
| 第6周末 | 能手写 RoPE 的前向/反向公式；理解为什么 sliding window + attention sink 能让 KV cache 不爆炸 |
| 第8周末 | 能解释 DMD 的 KL gradient 与 standard score matching 的区别；理解 backward simulation 消除 train-test mismatch 的原理 |
| 第10周末 | Side Project 有初步实验数据（至少一条有意义的 insight） |
| 第12周末 | 完成 Research Statement 和技术博客；确定目标导师名单 |

---

## 六、工具与资源

- **论文管理：** Zotero + 为每篇论文写 3-5 句话的 annotated bibliography
- **数学推导：** iPad/手写笔记，重点推导（ELBO, score function, RoPE rotation, KL gradient）
- **GPU 资源：** 自己的 RTX 4090 (24GB) 足够跑 Wan2.1 推理和小规模训练；大训练需要学校集群或 cloud（AutoDL, Lambda Labs, RunPod）
- **社区：** Twitter/X 关注视频生成方向的研究者；GitHub Discussions in Wan2.1 / diffusers；Discord (diffusers, CUDA MODE)
- **会议跟踪：** CVPR/ICCV/NeurIPS/ICLR proceedings, arXiv cs.CV (daily)
- **推荐课程：**
  - Stanford CS236 (Deep Generative Models) — 扩散模型部分
  - UC Berkeley CS294 (Deep Unsupervised Learning) — 生成模型
  - Yannic Kilcher / Aleksa Gordić 的论文讲解 YouTube 频道
  - CUDA MODE — GPU 编程与性能优化

---

## 七、从 Side Project 到论文

一个好的 PhD 申请通常需要至少一个 "statement-worthy project"。以下是几种路径：

1. **Analysis paper (最可行)：** 对 Infinity-RoPE 的各个组件做系统性消融实验，发现反直觉的结论，提出改进建议。适合投 CVPR/ICCV workshop。
2. **Method paper (有挑战)：** 将 block-relative RoPE + causal attention 迁移到另一个 domain（如 image-to-video, 3D Gaussian splatting，或 robotics video prediction）。
3. **Efficiency paper (实用)：** 大幅降低 data-free distillation 的训练成本，使其在消费级 GPU 上可行。

*建议先用 2 周做 exploration（跑 baseline、确认 idea 可行），再决定主攻哪个方向。*

---

> **最后提醒：** 不要试图读完所有论文才开始动手。最好的学习循环是：读 1-2 篇论文 → 立刻跑相关代码 → 发现不懂的细节 → 回论文找答案 → 继续跑代码。三个月足够从 CS231n 水平成长为能在视频生成方向做独立研究的 PhD 候选人。Good luck!
