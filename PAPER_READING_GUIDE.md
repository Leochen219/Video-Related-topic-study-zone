# 论文阅读方法论：从零到深度掌握

> 配套 [RESEARCH_ROADMAP.md](./RESEARCH_ROADMAP.md)，解答三个核心问题：
> 1. 论文应该读到什么程度才算"读懂了"？
> 2. 不同论文应该花多少时间、读到什么深度？
> 3. 具体怎么读？

---

## 一、论文的三个阅读深度

不是所有论文都需要从头读到尾。根据论文对你的研究的重要性，分为三个层级：

### Level 1：框架理解（1-3小时/篇）

**适用对象：** 横向拓展论文（如 Sora tech report、HunyuanVideo、MovieGen）

**掌握标准：**
- 能用 2-3 句话说出这篇论文的核心 contribution 是什么
- 能画出系统的高层架构图（哪些模块、怎么连接）
- 知道它和你的研究方向（Infinity-RoPE）的 **一个** 具体联系点
- **不需要** 能推导公式，**不需要** 能复现

**具体操作：**
```
20% 时间读 Abstract + Introduction + Conclusion
50% 时间读 Figures + Tables（图比文字更重要）
20% 时间浏览 Method 的小标题结构，不深究细节
10% 时间扫一眼 Experiments 的关键数字
```

**检验方式：** 关掉论文，能在一页纸上画出架构图，并标注出 1 个 "这个设计跟 Infinity-RoPE 的 XXX 类似/不同" 的点。

---

### Level 2：方法掌握（8-12小时/篇）

**适用对象：** 直接相关论文（如 DDPM、Flow Matching、Wan2.1、DMD2、CausVid、Self-Forcing）

**掌握标准：**
- 能完整推导核心公式（从第一行到最终 loss）
- 能解释每个设计选择的 motivation（为什么用这个不用那个？）
- 能手写核心算法的伪代码
- 能跑通官方代码，并能修改一个超参数看效果变化
- 知道论文的 **局限** 是什么（Failure cases、assumptions）

**具体操作：**
```
第一遍 (2h)：   Abstract → Conclusion → Figures → Method 小节标题
                （建立整体认知地图：这篇论文要解决什么？怎么解决的？结果怎么样？）

第二遍 (4h)：   Method 逐段精读 + 同步推导公式
                （纸笔在手边，每个公式自己推一遍，推不动的地方就是没懂）

第三遍 (2h)：   读 Experiments + 看代码
                （打开 GitHub repo，对着论文的实验设置看 config yaml）

第四遍 (2h)：   总结 + 关联
                （用自己的话写 summary，尝试和之前读过的论文建立联系）
```

**检验方式：**
- 给一个完全不懂的同学讲 15 分钟，对方能理解核心 idea
- 能不看论文推导出关键公式
- 能指出官方代码中哪些实现细节跟论文描述不完全一致（这种事情经常发生）

---

### Level 3：批判复现（20-40小时/篇）

**适用对象：** 你的核心论文（Infinity-RoPE），以及 Side Project 的 baseline

**掌握标准：**
- Level 2 的全部标准 +
- 能独立复现核心实验（或在小规模设定下验证关键 claim）
- 能指出论文中模糊、省略或不一致的细节
- 能提出一个有根据的改进方向，并设计实验验证
- 对论文的每个超参数都知道它影响什么

**具体操作：**
```
包含 Level 2 全部步骤 +

第五遍 (4h+)：  逐行阅读核心代码（如 causal_model.py）
                （理解每一个 if-else 分支的动机、每一个 reshape/permute 的维度含义）

第六遍 (8h+)：  在自己环境上复现最小实验
                （比如只用 1 个 GPU、small model size、few steps）

第七遍 (4h+)：  对照论文写 code review 笔记
                （"这里论文说 X，但代码做了 Y，可能是因为 Z"）
```

**检验方式：**
- 你的复现代码能跑通，且输出与论文报告的结果趋势一致
- 能回答 "如果去掉 XXX 组件，会发生什么" 这类消融问题

---

## 二、ROADMAP 中论文的分级

| 论文 | 级别 | 预计时间 | 备注 |
|------|------|----------|------|
| DDPM | **L2** | 8h | 扩散模型必修课 |
| DDIM | **L2** | 5h | 互补 DDPM |
| Score-Based SDE | **L2** | 8h | 统一框架视角 |
| Flow Matching | **L2** | 8h | Wan2.1 的数学基础 |
| Classifier-Free Guidance | **L2** | 3h | 短但关键 |
| Video Diffusion Models | **L2** | 6h | 视频扩散开山之作 |
| Latent Diffusion / SD | **L2** | 6h | VAE + cross-attention |
| Stable Video Diffusion | **L2** | 5h | 图像→视频微调范式 |
| Wan2.1 | **L2** | 8h | 你的基座模型 |
| DiT | **L2** | 6h | Transformer 替代 U-Net |
| RoPE | **L2** | 6h | 核心组件推导 |
| YaRN | **L2** | 5h | RoPE 外推经典方法 |
| StreamingLLM | **L2** | 5h | Attention sink 机制 |
| Flash Attention 1+2 | **L1** | 3h | 知道原理即可，不用实现 |
| CausVid | **L2** | 6h | 因果视频扩散先驱 |
| **Infinity-RoPE** | **L3** | 30h | 你的核心论文 |
| Consistency Models | **L2** | 6h | 蒸馏基础 |
| DMD2 | **L2** | 8h | 蒸馏方法 |
| SiD | **L1** | 3h | 了解与 DMD 的对比 |
| Self-Forcing | **L2** | 8h | 训练框架 |
| ODE Regression | **L1** | 3h | 理解初始化目的 |
| Sora | **L1** | 3h | 横向 |
| CogVideoX | **L1** | 3h | 横向 |
| MovieGen | **L1** | 2h | 横向 |
| HunyuanVideo | **L1** | 3h | 横向 |
| LongLive | **L1** | 3h | 横向 |

**时间汇总：** L3 × 1 = 30h, L2 × 17 ≈ 115h, L1 × 7 ≈ 20h，总计约 **165 小时**论文阅读，占 360 总小时（30h×12周）的 46%。

---

## 三、单篇论文的标准阅读流程

这是一个通用 SOP，根据 Level 调整深度。

### Step 1: 预读 (5 min)

在精读前问三个问题，写在论文第一页空白处：
1. 这篇论文要解决什么问题？（Problem）
2. 它为什么比现有方法好？（Claim）
3. 它跟我的研究方向有什么关系？（Relevance）

**如果三个问题都答不上来，先读 Abstract 和 Introduction 的最后一段。**

### Step 2: 骨架提取 (15 min)

不看正文，只看：
- All figures and captions
- All table headers
- Section titles and subsection titles

这一步的目标是建立 "这篇论文长什么样" 的空间认知。画一张手绘的论文结构图。

### Step 3: 正向速读 (30 min)

按顺序通读一遍，不纠结任何细节：
- 遇到不懂的公式 → 用铅笔圈出来，继续往下读
- 遇到陌生的术语 → 写下来，读完再查
- **坚决不回头读**

这一步的目标是理解 narrative flow：作者是怎么讲故事的。

### Step 4: 逆向精读 (核心步骤, 2-6h)

从 Conclusion 往前读：

```
Conclusion → 作者声称达成了什么？
    ↓
Experiments → 这些声称有数据支持吗？关键实验是什么？
    ↓
Method → 方法是怎么设计的？每个公式为什么长这样？
    ↓
Introduction → 现在重读，你就能理解作者的每一句铺垫了
```

**逆向读的好处：** 带着"结论是什么"去读方法，不会被中间的细节绕晕。你知道每个公式最终要服务于什么实验结果。

### Step 5: 公式推导 (1-3h)

对于 L2/L3 论文，在纸上从头推到尾：

- 第一遍：誊写论文中的推导过程
- 第二遍：合上论文，自己推一遍
- 第三遍：标出自己卡住的步骤，回去看论文是怎么处理的

**关键习惯：** 每个符号第一次出现时，在旁边标注它的维度（shape）。深度学习论文的大部分困惑来自于不知道每个 tensor 的 shape 是什么。

### Step 6: 代码对照 (1-4h)

打开官方代码仓库：
1. 找到论文的核心算法（通常在某个 model 文件的 forward 函数里）
2. 逐行对照论文的 Algorithm box
3. 标注论文和代码不一致的地方（这很常见，也是潜在的研究机会）
4. 修改一个超参数，跑一遍，观察输出变化

### Step 7: 总结输出 (30 min)

写一段结构化的阅读笔记：

```markdown
## [论文标题]

**一句话：** [用你自己的话，一句话说清这篇论文]

**核心公式：** [手写拍照或 LaTeX，只放最重要的 1-2 个]

**关键 insight：** [这篇论文最反直觉或最巧妙的设计]

**局限：** [作者自己承认的 + 你自己发现的]

**与我的研究的关系：** [具体到代码或公式层面]

**可跟进的方向：** [读了这篇论文后想到的 1-2 个 idea]
```

---

## 四、常见陷阱与对策

### 陷阱 1: "我要把每个公式都搞懂"

**现实：** 很多公式是工程细节，不影响对论文核心贡献的理解。比如 DiT 的 adaLN 的具体参数化方式，不需要逐行推导，理解 "用一个 MLP 把 timestep embedding 映射为 scale/shift" 就够了。

**对策：** 区分 "核心公式" 和 "工程公式"。核心公式通常出现在 Method 的开头部分，定义了 loss 或采样过程。工程公式是具体的参数化细节——先跳过，需要时再回来看。

### 陷阱 2: "读完一篇接一篇，从不回头看"

**现实：** 连续读 5 篇之后，第 1 篇已经忘了一大半。

**对策：** 每读完 3 篇 L2 论文，停下来 1 天做回顾。画一张 cross-paper comparison 表：

|  | DDPM | Flow Matching | Score SDE |
|---|---|---|---|
| Forward process | 逐步加噪 (Markov) | 线性插值 (OT path) | SDE |
| Loss | 噪声预测 MSE | 速度场预测 MSE | Score matching |
| Sampling | 逐步去噪 | ODE solver | PC sampler / ODE |
| 与 Inf-RoPE 的关系 | 基础框架 | Wan2.1 用这个 | — |

这种交叉对比比单独笔记有用得多。

### 陷阱 3: "代码跑不通就不跑了"

**现实：** 官方代码经常有环境问题、缺依赖、hardcode 路径。

**对策：**
- 先尝试 15 分钟解决环境问题（Google error message）
- 如果 15 分钟搞不定，**Debug 日记**不是 "代码跑不通" 四个字，而是：报错信息是什么？你尝试了什么？卡在哪一步？
- 可以考虑不看代码直接读——不是所有论文都需要跑代码
- 对于 L3 论文必须跑通（可以求助 Claude/GPT 帮你修环境）

### 陷阱 4: "只读论文不写代码" VS "只写代码不读论文"

**现实：** 纯读论文会让人产生 "我懂了" 的幻觉；纯写代码会忽略理论动机。

**对策：** 严格遵守 50/50 的时间分配。读完一篇 L2 论文后，至少花等量时间写相关代码（哪怕是 toy example）。最简单的形式：

```python
# 读完 DDPM 后，一个 50 行的 toy diffusion 能验证你理解了：
# 1. noise schedule
# 2. forward diffusion (x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps)
# 3. loss = MSE(eps, eps_theta(x_t, t))
# 4. reverse sampling loop
```

写不出来 toy example = 没真懂。

---

## 五、建立你的论文知识图谱

推荐用 **Obsidian** 或 **Notion** 维护一个论文知识图谱。每篇论文一个节点，每条关键概念（如 "RoPE", "Flow Matching", "Attention Sink"）也是一个节点，用双向链接连接：

```
DDPM ──→ Score Matching ──→ Score SDE
  │                              │
  └──→ Diffusion Framework ←─────┘
         │
         ├──→ Flow Matching ──→ Wan2.1
         │
         └──→ Latent Diffusion ──→ Stable Video Diffusion
                                       │
                                       └──→ CausVid ──→ Infinity-RoPE
```

这样做的好处：当你写 Research Statement 时，你不需要 "回忆" 论文之间的关系——你已经把它们画出来了。

---

## 六、每周论文阅读清单模板

每周日开始前填写，周末对照检查：

```markdown
## 第 X 周论文计划

### 精读 (L2/L3)
- [ ] [论文名] — 预计 Xh, 实际 ___h
  - 核心公式推导完成？ □
  - 代码对照完成？ □
  - 总结笔记写完？ □

### 速读 (L1)
- [ ] [论文名] — □ 架构图 □ 一句话总结

### 本周交叉对比
- 这周读的 X 篇论文之间有什么联系？
- 哪个 idea 反复出现？

### 本周最大困惑
- （写下来，下周带着困惑继续读）
```

---

## 七、关键提醒

1. **阅读顺序很重要。** DDPM → Flow Matching → Wan2.1 → CausVid → Infinity-RoPE 是一条因果链。跳着读只会反复回来补课，浪费时间。

2. **"读懂了"的最高标准不是能复述，而是能修改。** 能说出 "如果改 XXX，预期 YYY 会变好/变坏" 并给出理由，才是真懂。

3. **论文不是圣经。** 顶级会议的论文也有错误、模糊和过度声称。随着阅读量增加，你会越来越擅长发现这些问题——这正是做研究的能力。

4. **你的第一个月会很慢。** DDPM 可能花 10 小时才 "感觉懂了"，这是正常的。第 3-4 篇扩散论文的速度会明显加快，因为概念在重复。到第 6 周，你读一篇新论文的 Method section 可能只需要首次阅读一半的时间。

5. **用 AI 辅助但不依赖它。** 当一个公式看不懂时，可以问 Claude/GPT "请用直观的语言解释这个公式每个符号的含义"。但对 L2 及以上论文，最终必须自己推导一遍——AI 的解释给你 intuition，自己的推导给你 certainty。
