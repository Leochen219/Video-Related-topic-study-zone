# 科研代码方法论：如何用 AI 辅助但不被 AI 耽误

> 配套 [RESEARCH_ROADMAP.md](./RESEARCH_ROADMAP.md) 和 [PAPER_READING_GUIDE.md](./PAPER_READING_GUIDE.md)
>
> 核心问题：暑假 12 周、每周 ~15h 写代码，AI 应该扮演什么角色？

---

## 一、核心原则：AI 是加速器，不是替代品

```
你的大脑                           AI
─────────                      ─────────
理解为什么        ←──→         帮你写怎么做的草稿
做设计决策                     填充 boilerplate
判断对错                       解释报错信息
形成直觉                       找文档/API 用法
```

**一个简单的判断标准：** 如果一行代码你删掉之后不知道会发生什么，那这行代码不应该出现在你的项目里——无论它是不是 AI 写的。

---

## 二、AI 使用的三区模型

把写代码分成三个区域，AI 的使用策略完全不同：

```
┌──────────────────────────────────────────────────────┐
│  🚫 禁区：自己写                                       │
│  - 核心算法的 forward/backward                        │
│  - loss function 的实现                               │
│  - attention mechanism (RoPE, KV cache)               │
│  - sampling loop                                      │
│                                                       │
│  原因：这些是你要写到 Research Statement 里的东西。      │
│  如果 AI 帮你写了，你永远学不会。                        │
├──────────────────────────────────────────────────────┤
│  🟡 协作区：AI 写草稿，你改                             │
│  - 数据加载 (Dataset class, DataLoader)               │
│  - 配置文件解析 (YAML/argparse)                        │
│  - logging, checkpointing, wandb 集成                  │
│  - 可视化代码 (plot loss curves, render video)         │
│  - 单元测试                                            │
│                                                       │
│  策略：让 AI 写第一版 → 你逐行读懂 → 改到你的需求        │
├──────────────────────────────────────────────────────┤
│  🟢 放心区：交给 AI                                    │
│  - 环境配置 (conda env, pip install, Docker)           │
│  - 报错排查 ("这个 CUDA error 什么意思")                │
│  - 代码格式化和 linting                                 │
│  - Shell 脚本、git 操作                                │
│  - 文档字符串、type hints 生成                          │
│  - 已知算法的查漏补缺 ("帮我校验这段 ddpm sampler        │
│    和论文 Algorithm 1 是否一致")                         │
└──────────────────────────────────────────────────────┘
```

---

## 三、ROADMAP 练手项目的 AI 使用指南

### 练手项目 1：从头实现 DDPM (第1-2周)

| 代码模块 | 区域 | AI 使用方式 |
|----------|------|------------|
| `forward_diffusion(x_0, t)` — 加噪公式 | 🚫 禁区 | 完全自己写，对照论文公式 |
| `UNet2D` 或 `DiTBlock` | 🚫 禁区 | 自己画图设计，自己写 forward |
| `DDPM Scheduler` (noise schedule, beta schedule) | 🚫 禁区 | 论文有明确公式，自己实现 |
| `training_loop` (optimizer, lr_scheduler, gradient clipping) | 🟡 协作 | 让 AI 搭框架，你填核心逻辑 |
| `sample()` — reverse diffusion loop | 🚫 禁区 | 这个不会写 = 没懂扩散模型 |
| 可视化 grid of generated images | 🟢 放心 | 交给 AI |
| 配置文件 (argparse / yaml) | 🟢 放心 | 交给 AI |

### 练手项目 2：实现 RoPE + KV Cache (第5-6周)

| 代码模块 | 区域 | AI 使用方式 |
|----------|------|------------|
| `rope_apply(q, k, cos, sin)` — 旋转操作 | 🚫 禁区 | 自己推公式，自己写 |
| `apply_rotary_pos_emb` 的各种变体 | 🚫 禁区 | 核心能力 |
| `KVCache` class (insert, evict, roll) | 🚫 禁区 | Infinity-RoPE 的核心 |
| `block_causal_mask` 生成 | 🟡 协作 | 维度容易错，让 AI 帮你检查 |
| 可视化 attention score (heatmap) | 🟡 协作 | AI 写 matplotlib，你定需求 |
| 对比实验脚本 (baseline vs sliding window) | 🟡 协作 | AI 搭实验框架 |
| toy transformer 的其余部分 (FFN, LayerNorm) | 🟢 放心 | 标准组件 |

### Side Project (第9-10周)

| 代码模块 | 区域 | AI 使用方式 |
|----------|------|------------|
| **核心创新部分** (你的 contribution) | 🚫 禁区 | 必须自己写 |
| Baseline 实现 | 🟡 协作 | 让 AI 帮你从论文翻译到代码 |
| 评估指标 (FVD, CLIP score, FID) | 🟢 放心 | 调库 |
| 实验管理 (wandb sweeps, grid search) | 🟢 放心 | 交给 AI |
| Demo / Web UI | 🟢 放心 | 交给 AI |

---

## 四、什么情况下绝对不能用 AI

这些场景下用 AI 写代码是在欺骗自己：

### 1. 你不理解的数据结构

```python
# AI 写了一段 RoPE 代码，里面有 rotate_half、complex_multiply 等函数
# 如果你不能回答：
#   - 为什么用复数旋转？
#   - rotate_half 做了什么 reshape？
#   - cos/sin 的 frequency 是怎么计算的？
# → 删掉重写，直到你能解释每一行
```

### 2. 你看不懂的维度变换

```python
x = x.reshape(B, T, H, D).permute(0, 2, 1, 3).contiguous()
# 如果你不能在纸上画出每一步后的 tensor shape
# → AI 写对了也算你没懂
```

### 3. Loss 函数和训练逻辑

Loss 是研究工作的核心。如果你的 Side Project 的 loss 是 AI 写的而你不知道它为什么对（或者根本不对），你的论文会被 reviewer 在 method 部分直接拒掉。

### 4. 任何你要放进论文的实验

实验代码有任何你不理解的成分 → 你的 ablation study 就无法解释 anomaly → reviewer 会质疑你的可信度。

---

## 五、AI 的最佳使用时机

### 时机 1: "我已经会了，但不想手敲"（效率型）

当你已经在小项目上自己写过一遍 DDPM sampler，第二个项目可以理所当然地让 AI 照论文公式给你生成一个 draft，你只需要检查正确性。

**判断标准：** 关掉 AI，你能否在白板上手写出这段代码的核心逻辑？能 → 可以交给 AI。

### 时机 2: "卡住了，需要 Rubber Duck"（调试型）

```
你不是这么用 AI：
  "我写了 diffusion sampler 但结果全黑，帮我看看代码哪里错了"
  （把 300 行全贴进去）

你应该这么用 AI：
  "我写了 DDPM sampler，每步的 x_t 的 std 在不断衰减到接近 0。
   我期望 std 在最后几步接近 1（纯噪声的 std = 1）。
   我的 noise schedule 是 linear beta from 1e-4 to 0.02。
   可能是什么原因？"
  （描述现象 + 你的预期 + 你已经排查过的方向）
```

### 时机 3: "需要翻译论文到代码"（翻译型）

```
好用法：
  "帮我把 CausVid 的 Algorithm 2 (KV-cached causal inference) 
   翻译成 PyTorch 伪代码。
   输入是 [B, T, H, D] 的 q, k, v 和一个 KVCache 对象。
   我自己会写完整的 forward，你先给我一个骨架让我对照论文。"
```

### 时机 4: "写 boilerplate 浪费时间"（基础设施型）

数据加载、配置管理、日志记录、实验追踪——这些可以放心交给 AI。它们不是你的核心竞争力。

### 时机 5: "帮我 code review"（审查型）

```
你写好了 causal attention 的 forward，让 AI 对照论文检查：
  "这是我的 block-relative RoPE 实现。请对照 Infinity-RoPE 
   论文 Section 3.2 和 Algorithm 1，检查：
   1. KV cache 的索引逻辑是否正确
   2. sink_size 的边界条件
   3. scene cut 时的 token rotation 逻辑
   如果有问题，指出具体哪一行和论文哪里不一致。"
```

---

## 六、阅读开源项目的正确姿势

Roadmap 中列了 15+ 开源项目，你不会全读。区分三种读法：

### A. 泛读：理解架构边界（1-2h）

**适用：** CogVideoX, HunyuanVideo, Open-Sora 等横向项目

```
只看：
  - README 的 architecture diagram
  - 顶层 __init__.py 或 model.py 的 class 定义
  - config yaml 的核心参数名
  - pipeline 的 __call__ 方法

目标：
  - 能说出这个项目和 Wan2.1/Infinity-RoPE 在设计上的 1 个关键区别
```

### B. 精读：理解一个组件（4-8h）

**适用：** diffusers 的 scheduler, Wan2.1 的 VAE

```
流程：
  1. 找到你要读的那个文件（如 scheduling_ddpm.py）
  2. 从头到尾逐行读一遍，写边注
  3. 关掉代码，自己重写核心逻辑（不是抄，是回忆着写）
  4. 对比源代码，标记你漏了什么
```

### C. 钻读：完整跑通 + 修改（20h+）

**适用：** Infinity-RoPE, Self-Forcing

```
流程：
  1. 先跑通推理（inference.py），不修改任何东西
  2. 在关键位置插 pdb.set_trace() / print shape
  3. 修改一个小参数（如 sink_size, local_attn_size），观察输出变化
  4. 尝试加入你自己的代码（如一个新的 eviction 策略）
  5. 如果新加的功能能跑通，你就真正理解了这个项目
```

---

## 七、科研代码的 5 条铁律

### 铁律 1: 可复现性从第一天开始

```yaml
# 每个实验都记录：
seed: 42
env: conda env export > env.yaml
config: 完整的 yaml，不依赖命令行参数
log: wandb / tensorboard，不要 print 到 stdout
```

坏习惯：改了一个超参数跑出好结果，但忘了改的是什么。

好习惯：每个实验一个 config yaml，git commit 它的 hash 写在 wandb log 里。

### 铁律 2: 先用 toy setting 验证，再上规模

```
验证扩散模型：CIFAR-10 (32×32, 5000 steps) → 确认 loss 下降趋势对 → 上大图
验证 RoPE：     seq_len=16, dim=64 → 验证 rotary 不改变 norm → 上长序列
验证 KV cache： fake tensor, 手动构造 key/value → 验证 evict 索引对 → 上真实模型
```

很多人一上来就在 81 帧 Wan2.1 上 debug，一次 forward 等 2 分钟，一个下午只能尝试 10 个改动。先用 `torch.randn` 构造假的 tensor 在 2 秒内验证逻辑，确认无误再到真实模型上跑。

### 铁律 3: 写代码前先写检查点

```python
def causal_attention(q, k, v, kv_cache):
    """
    实现 Infinity-RoPE Section 3.2, Algorithm 1.
    
    关键检查点：
    ✓ q: [B, 1, H, D] (decode 时 T=1)
    ✓ kv_cache.tokens: int, 当前缓存多少 token
    ✓ 返回的 attn_output shape 必须和 q 一致
    ✓ 当 kv_cache.tokens >= local_attn_size 时触发 eviction
    ✓ sink_size 个 token 永远不被 evict
    ✓ scene_cut 时 kv_cache 被 flush 到 sink_size
    """
    # 实现...
```

先写检查点（你想验证什么），再写代码。写完对着检查点逐条通过。

### 铁律 4: 代码不是一次写对的——写三版

```
V1 (探索版)：能跑就行，各种 hardcode，各种 print debug。
            目的：验证 idea 在代码上可行。

V2 (整理版)：去掉 hardcode，抽象成 config，清理 print。
            目的：可以给别人看，可以跑 ablation。

V3 (发布版)：type hints, docstring, unit tests, README。
            目的：可以放到 GitHub portfolio / 论文 supplementary。
```

很多学生卡在 "想一次写出 V3" 的完美主义里，结果什么都没写出来。先写 V1，允许自己是丑陋的。

### 铁律 5: Git 就是你的实验笔记本

```bash
# 每个实验一个 branch，不要都在 main 上改
git checkout -b exp/dmd-fp8-training
git checkout -b exp/larger-sink-size

# Commit message 写实验结论，不写改了什么文件
"exp: sink_size=4 → FVD=120, sink_size=8 → FVD=95 (better temporal consistency)"
# 不是 "modified causal_model.py"

# 失败的实验也要 commit（标注 FAILED）
"FAILED: fp16 training diverged after iter 300, grad norm exploded to 1000+"
# 三个月后你写论文时，这些失败记录非常宝贵
```

---

## 八、AI 使用清单：开工前问自己

每次打开编辑器前，快速自问三个问题：

1. **这段代码的核心逻辑我能在白板上画出来吗？**
   - 能 → AI 可以辅助写
   - 不能 → 必须先自己搞懂，再写代码

2. **如果这段代码有 bug，我能在 10 分钟内定位到问题行吗？**
   - 能 → AI 写的不影响你的 debug 能力
   - 不能 → 你对这段代码的理解不够，别让 AI 写

3. **这段代码会出现在我的论文/Research Statement 里吗？**
   - 会 → 🚫 禁区，自己写
   - 不会，且是标准组件 → 🟢 放心交给 AI
   - 不会，但涉及实验逻辑 → 🟡 协作区

---

## 九、12 周代码能力成长曲线

```
Week 1-2:  在 AI 帮助下写出第一个可训练的 DDPM (CIFAR-10)
           感觉自己很依赖 AI，经常要问 "这段代码对吗"
           
Week 3-4:  能自己写 Wan2.1 推理脚本的 wrapper
           AI 帮你搭框架，你填细节
           
Week 5-6:  能自己写 RoPE + KV cache 的完整实现（200行）
           开始能在 AI 给的代码中发现 bug
           
Week 7-8:  能读懂 DMD 训练的 80% 代码
           能修改训练超参数并预测影响
           
Week 9-10: AI 主要帮你写 boilerplate 和 debug
           核心逻辑自己写，偶尔让 AI code review
           
Week 11-12: 在写 Side Project 的核心创新时，根本不打开 AI
            成品代码有 type hints, tests, config, README
```

第一周 80% 的代码可能是 AI 写的，但第十二周这个比例应该降到 20%——剩下的 80% 的核心逻辑是你自己写的。**这个比例的下降就是你真正的成长。**

---

## 十、推荐工具链

| 用途 | 工具 | 为什么 |
|------|------|--------|
| AI 编程助手 | Claude Code (当前) | 能读整个 repo 上下文，理解项目结构 |
| IDE | VS Code + Python/Pylance | 类型检查、跳转定义、autocomplete |
| 环境管理 | conda / mamba | 每个项目一个 env，不打乱依赖 |
| 实验追踪 | wandb | 免费 academic plan，自动记录所有 metrics |
| 代码托管 | GitHub | Portfolio 的核心，每个项目一个干净 repo |
| 格式化 | ruff (替代 flake8+isort) | 快，一个工具搞定 lint + format |
| 类型检查 | mypy (可选) | 规范代码质量，但不是第一优先级 |
| 论文复现 | 复制官方 repo，不改动，另开 repo 做改动 | 保持 clean baseline |
| GPU 云 | AutoDL / Lambda Labs / RunPod | 学校集群排队时备用 |

---

> **最后的话：** 暑假结束后的你，真正有竞争力的不是 "我用 AI 做了 X"，而是 "我能解释 X 的每个细节，AI 只是帮我省了打字时间"。PhD 导师面试时一眼就能看出谁是真的懂、谁是 AI 帮他装的懂。让 AI 帮你跑得更快，但路必须自己走。
