"""
RoPE (Rotary Position Embedding) 手写练习
==========================================
一步步理解 RoPE 编码的原理和实现。

核心直觉:
  在标准的 Transformer 中，位置编码是"加"到 token embedding 上的:
    x = embed(token) + pos_encoding(pos)

  RoPE 不同: 它通过"旋转" query 和 key 来注入位置信息。
  旋转的好处是 → Q 和 K 的内积天然只依赖于"相对位置" (m-n)，而非绝对位置。

  类比: 时钟上的指针。
  - 指针长度 = 语义信息 (token 是什么)
  - 指针角度 = 位置信息 (token 在第几个位置)
  - 两个指针的夹角只取决于它们的"相对角度差"，不取决于绝对指向

公式总览:
  对于位置 pos，维度对 i (i = 0, 1, ..., d/2 - 1):
    θ_i = 1 / (10000^(2i/d))            ← 基础频率 (高频→低频)
    旋转角度 = pos * θ_i                 ← 位置越远，转得越多

  对向量对 (x_2i, x_2i+1) 旋转 pos*θ_i 弧度:
    x_2i'   = x_2i * cos(pos*θ_i) - x_2i+1 * sin(pos*θ_i)
    x_2i+1' = x_2i * sin(pos*θ_i) + x_2i+1 * cos(pos*θ_i)

参考资料: RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
"""

import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt


# ============================================================================
# 第 1 步: 2D 旋转 — 感受"向量被旋转"
# ============================================================================
# 这是 RoPE 最底层的操作。一个 2D 向量 (x, y) 逆时针旋转 θ 弧度:
#   x' = x*cos(θ) - y*sin(θ)
#   y' = x*sin(θ) + y*cos(θ)
#
# 几何直觉: 把你的手臂伸直指向某个方向，然后以肩膀为轴转动手臂。
#          手臂长度不变，但指向的方向变了。

def rotate_2d(x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor):
    """
    对 2D 向量 (x, y) 施加旋转角度 theta。

    Args:
        x, y: shape (...,), 向量的两个分量
        theta: shape (...,), 旋转角度 (弧度)
    Returns:
        x_rotated, y_rotated
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_rotated = x * cos_theta - y * sin_theta
    y_rotated = x * sin_theta + y * cos_theta
    return x_rotated, y_rotated


def demo_step1():
    """可视化: 同一个向量旋转不同角度后的位置"""
    x = torch.tensor([1.0])   # 初始向量的 x 分量
    y = torch.tensor([0.0])   # 初始向量的 y 分量 (单位向量指向正x轴)

    angles = torch.linspace(0, 2 * math.pi, 13)[:-1]  # 12个角度, 不包含2π(和0重复)

    xs_rot, ys_rot = rotate_2d(
        x.expand_as(angles),   # 把标量扩展到和 angles 一样的 shape
        y.expand_as(angles),
        angles
    )

    plt.figure(figsize=(6, 6))
    plt.axhline(0, color='gray', alpha=0.3)
    plt.axvline(0, color='gray', alpha=0.3)
    plt.quiver(
        torch.zeros_like(xs_rot), torch.zeros_like(ys_rot),
        xs_rot, ys_rot,
        angles=angles.numpy(), scale=1, scale_units='xy', cmap='hsv'
    )
    plt.xlim(-1.5, 1.5); plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal')
    plt.title('Step 1: 2D 旋转 — 同一个向量旋转不同角度')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

    print("观察: 向量长度(=1)保持不变，只有方向改变。这就是 RoPE 的'保距'性质。")
    print("这意味着语义信息(长度)被保留了，只有位置信息(方向)被编码。")


# ============================================================================
# 第 2 步: 计算旋转频率 θ_i
# ============================================================================
# RoPE 对不同的维度对用不同的频率。低维度用高频(转得快)，高维度用低频(转得慢)。
# 类比: 二进制计数 — 最低位变化最快，最高位变化最慢。类似地，
#       RoPE 用不同频率的组合来唯一标识每个位置。

def compute_freqs(dim: int, theta_base: float = 10000.0):
    """
    计算每一对维度的基础频率 θ_i。

    θ_i = 1 / (theta_base^(2i/d))

    Args:
        dim: 总维度 (必须是偶数)
        theta_base: 基础频率参数, 论文默认 10000
    Returns:
        freqs: shape (dim//2,), 每对维度的基础频率
    """
    # TODO: 你的实现
    # 提示:
    #   1. i 从 0 到 dim//2 - 1
    #   2. exponent = 2*i / dim
    #   3. θ_i = theta_base^(-exponent) = 1 / theta_base^exponent
    # arange(dim//2) 产生 0, 1, ..., dim//2-1，共 dim//2 个值
    i = torch.arange(dim//2, dtype=torch.float32)
    exponent = 2 * i / dim
    theta_i = theta_base ** (-exponent)  # 等价于 1 / theta_base^exponent
    return theta_i


def demo_step2():
    """可视化: 不同维度的旋转频率"""
    dim = 64
    freqs = compute_freqs(dim)

    # 频率应当随维度递减 (低频→高频，对应 低维→高维)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(freqs.numpy(), 'b.')
    plt.title(f'Step 2a: 各维度对的频率 (dim={dim})')
    plt.xlabel('维度对索引 i'); plt.ylabel('θ_i (基础频率)')
    plt.grid(True, alpha=0.3)

    # 看看各维度在位置 0~100 时的旋转角度
    plt.subplot(1, 2, 2)
    positions = torch.arange(100).float()
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (100, dim//2)
    # 取几个代表性的维度
    for idx in [0, dim//4, dim//2 - 1]:
        plt.plot(positions.numpy(), angles[:, idx].numpy(),
                 label=f'dim_pair={idx}')
    plt.title('Step 2b: 旋转角度随位置变化')
    plt.xlabel('位置 pos'); plt.ylabel('旋转角度 (弧度)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("观察: 低维度对(蓝色)转得极快，高维度对(绿色)转得极慢。")
    print("这意味着相邻位置的 token 在低频维度上几乎相同，在高频维度上差异明显。")


# ============================================================================
# 第 3 步: RoPE 的核心 — 对一个序列施加旋转位置编码
# ============================================================================
# 有两种等价的实现方式:
#   方式 A (直观但慢): 逐对旋转
#     对每一对维度 (x_2i, x_2i+1)，用 2D 旋转矩阵旋转角度 pos*θ_i
#
#   方式 B (高效实际用): 复数乘法
#     把每对视为复数 z = x_2i + j*x_2i+1
#     旋转等价于 z * e^(j*pos*θ_i) = z * (cos(pos*θ_i) + j*sin(pos*θ_i))
#     这可以用向量化操作一次性完成整个序列

def rope_naive(x: torch.Tensor, freqs: torch.Tensor):
    """
    方式 A: 朴素实现 — 逐对旋转 (帮助理解，不用于生产)

    Args:
        x: shape (seq_len, dim), 输入的 token 序列
        freqs: shape (dim//2,), 基础频率 θ_i
    Returns:
        x_rotated: shape (seq_len, dim)
    """
    seq_len, dim = x.shape
    assert dim % 2 == 0, "dim 必须是偶数"

    x_rotated = x.clone()
    # TODO: 你的实现
    # 对每个位置 pos 和每对维度 i:
    #   取 x[pos, 2*i] 和 x[pos, 2*i+1] 这对
    #   旋转角度 = pos * freqs[i]
    #   用 step1 的 rotate_2d 做旋转
    #   写回 x_rotated
    for pos in range(seq_len):
        for i in range(dim//2):
            angle=pos* freqs[i]
            x_rotated[pos,2*i],x_rotated[pos,2*i+1]=rotate_2d(x[pos,2*i],x[pos,2*i+1],angle)


    return x_rotated


def rope_efficient(x: torch.Tensor, freqs: torch.Tensor):

    """

    方式 B: 高效实现 — 复数乘法 (工业界实际使用)



    原理: 把向量视为 d/2 个复数的实部和虚部

      z_i = x[2i] + j * x[2i+1]

      z_i' = z_i * e^(j * pos * θ_i)

           = z_i * (cos(pos*θ_i) + j * sin(pos*θ_i))



    然后分别取实部和虚部即可。



    Args:

        x: shape (seq_len, dim)

        freqs: shape (dim//2,)

    Returns:

        x_rotated: shape (seq_len, dim)

    """

    seq_len, dim = x.shape

    positions = torch.arange(seq_len, dtype=x.dtype, device=x.device)



    # Step 3a: 计算每个位置 x 每个维度对 的旋转角度

    # positions: (seq_len,) → (seq_len, 1)

    # freqs: (dim//2,) → (1, dim//2)

    # angles: (seq_len, dim//2)

    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)



    # Step 3b: 一次性计算所有 cos 和 sin

    cos_vals = torch.cos(angles)  # (seq_len, dim//2)

    sin_vals = torch.sin(angles)  # (seq_len, dim//2)



    # Step 3c: 将 x 拆成偶数维和奇数维

    # x: (seq_len, dim) → 每对 (x_even, x_odd)

    # TODO: 你的实现

    # 提示: x_even = x[:, 0::2], x_odd = x[:, 1::2]

    # 它们的 shape 都是 (seq_len, dim//2)

    x_even=x[:,0::2]

    x_odd=x[:,1::2]





    # 实际上 cos_vals 还需要在 dim 维度上交错排列:

    # cos_vals 是 (seq_len, dim//2)，我们需要把它复制一份 → (seq_len, dim)

    # 可以用 repeat_interleave 或 stack+reshape



    # Step 3d: 复数乘法的实部和虚部

    # z' = (x_even + j*x_odd) * (cos + j*sin)

    #    = (x_even*cos - x_odd*sin) + j*(x_even*sin + x_odd*cos)

    # ROI: 请解释为什么这个复数乘法等价于 2D 旋转矩阵？

    #   (写在注释里)

    x_even_rotated = x_even * cos_vals - x_odd * sin_vals

    x_odd_rotated = x_even * sin_vals + x_odd * cos_vals



    # Step 3e: 把偶数/奇数维度交错合并回 (seq_len, dim)

    # TODO: 你的实现

    # 提示: 可以用 stack(dim=-1) 然后 reshape

    x_rotated=torch.stack([x_even_rotated,x_odd_rotated],dim=-1).reshape(seq_len,dim)



    return x_rotated

# ============================================================================
# 第 4 步: 将 RoPE 应用到 Attention 的 Q 和 K
# ============================================================================
# 在实际 Transformer 中:
#   Q = W_q @ x      ← 先做线性投影
#   K = W_k @ x
#   Q' = RoPE(Q)     ← 再施加旋转位置编码
#   K' = RoPE(K)
#   V 不做 RoPE      ← 位置信息通过 QK 内积传递，V 不需要

def apply_rope_to_qk(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor):
    """
    对 Q 和 K 分别施加 RoPE。

    输入 shape: (batch, num_heads, seq_len, head_dim)
    rope_efficient 期望: (seq_len, dim) — 2D 输入

    核心思路 — 为什么 reshape 可行:
      RoPE 对每个 batch 样本、每个 head 的操作是完全一样的！
      因为位置(seq_len)只和序列维度有关，和 batch/head 无关。
      所以把 (batch, num_heads) 压平成一个大"批次":

        (batch, num_heads, seq_len, head_dim)
              ↓ view 合并前两维
        (batch * num_heads, seq_len, head_dim)
              ↓ 逐 slice 调用 rope_efficient
        (batch * num_heads, seq_len, head_dim)
              ↓ view 恢复
        (batch, num_heads, seq_len, head_dim)
    """
    batch, num_heads, seq_len, head_dim = q.shape
    assert head_dim % 2 == 0, "head_dim 必须是偶数"

    q_flat = q.view(batch * num_heads, seq_len, head_dim)
    k_flat = k.view(batch * num_heads, seq_len, head_dim)


    q_rope_list, k_rope_list = [], []
    for i in range(batch * num_heads):
        q_rope_list.append(rope_efficient(q_flat[i], freqs))
        k_rope_list.append(rope_efficient(k_flat[i], freqs))

    q_rope = torch.stack(q_rope_list, dim=0).view(batch, num_heads, seq_len, head_dim)
    k_rope = torch.stack(k_rope_list, dim=0).view(batch, num_heads, seq_len, head_dim)

    return q_rope, k_rope


def demo_step4():
    """验证 apply_rope_to_qk 的 shape 和保范性"""
    batch, num_heads, seq_len, head_dim = 2, 4, 8, 16
    freqs = compute_freqs(head_dim)

    torch.manual_seed(99)
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads, seq_len, head_dim)

    q_rope, k_rope = apply_rope_to_qk(q, k, freqs)

    print("Step 4: apply_rope_to_qk")
    print(f"Q shape: {q.shape} → {q_rope.shape} [OK]")
    print(f"K shape: {k.shape} → {k_rope.shape} [OK]")

    norm_before = q.norm(dim=-1).mean()
    norm_after = q_rope.norm(dim=-1).mean()
    print(f"Q norm 旋转前: {norm_before:.4f}, 旋转后: {norm_after:.4f}")
    print("如果接近，说明多头场景也保距 [OK]")


def demo_step3():
    """验证 naive 和 efficient 两种实现输出一致"""
    dim = 8
    seq_len = 4
    freqs = compute_freqs(dim)

    # 随机输入
    torch.manual_seed(42)
    x = torch.randn(seq_len, dim)

    result_naive = rope_naive(x, freqs)
    result_efficient = rope_efficient(x, freqs)

    print("Step 3: 验证两种实现")
    print(f"输入 shape: {x.shape}")
    print(f"Naive 输出:\n{result_naive}")
    print(f"Efficient 输出:\n{result_efficient}")
    print(f"最大差异: {(result_naive - result_efficient).abs().max().item():.2e}")
    print("如果差异接近 0，说明两种实现等价 [OK]")

    # 验证保范性: ||旋转后|| ≈ ||旋转前||
    norm_before = x.norm(dim=-1)
    norm_after = result_efficient.norm(dim=-1)
    print(f"\n旋转前 norm: {norm_before}")
    print(f"旋转后 norm: {norm_after}")
    print("如果接近，说明 RoPE 只旋转不缩放(保距性质) [OK]")





    

    


# ============================================================================
# 第 5 步: 验证核心性质 — 内积只依赖于相对位置
# ============================================================================
# 这是 RoPE 最核心的理论性质:
#   RoPE(q, m) · RoPE(k, n) = g(q, k, m-n)
#
# 即: 位置 m 的 Q 和位置 n 的 K 做内积，结果只和相对位置 (m-n) 有关。
# 这意味着模型天然理解"两个 token 之间隔了多少个位置"。

def verify_relative_property():
    """
    验证 RoPE 的核心性质: Q_m · K_n 只依赖于相对位置 (m-n)。

    原理推导 (复数形式):
      设 q 和 k 是原始向量，在位置 m 和 n 施加 RoPE:
        RoPE(q, m) = q * e^{j*m*θ}     (逐元素复数乘法)
        RoPE(k, n) = k * e^{j*n*θ}

      内积 = Re( Σ (q_i * e^{j*m*θ_i}) · conj(k_i * e^{j*n*θ_i}) )
           = Re( Σ q_i * conj(k_i) * e^{j*(m-n)*θ_i} )

      可以看到内积只和 (m-n) 有关！绝对位置 m 和 n 被消掉了。
    """
    dim = 64
    freqs = compute_freqs(dim)

    # 创建固定的一对向量 (模拟 Q 和 K 的语义内容)
    torch.manual_seed(123)
    q_vec = torch.randn(dim)  # query 的语义
    k_vec = torch.randn(dim)  # key 的语义

    # ------------------------------------------------------------------
    # Test 1: 相同相对位置，不同绝对位置 → 内积应相同
    # ------------------------------------------------------------------
    print("=" * 50)
    print("Test 1: 相同相对位置 → 内积相等")

    # 我们需要一个辅助函数: 把向量放在序列的某个位置并施加 RoPE
    def rope_at_position(vec, pos, freqs):
        """把 vec 放在序列位置 pos 处，施加 RoPE 后取回该位置的结果"""
        # 构造一个序列，把 vec 放在 pos 位置 (其他位置填 0，不影响结果)
        seq = torch.zeros(pos + 1, dim)
        seq[pos] = vec
        rope_seq = rope_efficient(seq, freqs)
        return rope_seq[pos]

    # 三组不同的绝对位置，但相对位置都是 3 (即 q_pos - k_pos = 3)
    pairs = [
        (5, 2),    # q 在 5, k 在 2,  相对 = 5-2 = 3
        (10, 7),   # q 在 10, k 在 7, 相对 = 10-7 = 3
        (100, 97), # q 在 100, k 在 97, 相对 = 100-97 = 3
    ]

    dot_products = []
    for q_pos, k_pos in pairs:
        q_rope = rope_at_position(q_vec, q_pos, freqs)
        k_rope = rope_at_position(k_vec, k_pos, freqs)
        dot = (q_rope * k_rope).sum().item()
        dot_products.append(dot)

    for (q_pos, k_pos), dot in zip(pairs, dot_products):
        print(f"  q_pos={q_pos:3d}, k_pos={k_pos:3d}, 相对位置={q_pos-k_pos:2d} → Q·K = {dot:.6f}")

    max_diff = max(dot_products) - min(dot_products)
    print(f"  三组内积的最大差异: {max_diff:.2e}")
    if max_diff < 1e-4:
        print("  [OK] 通过! 相同相对位置 → 内积几乎完全相等")
    else:
        print("  [FAIL] 失败! 内积差异过大，检查实现")

    # ------------------------------------------------------------------
    # Test 2: 遍历所有相对位置，验证内积只随 (m-n) 变化
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Test 2: 内积只随相对位置变化 (遍历验证)")

    # 固定绝对位置 q_pos=20, 变化 k_pos 从 0 到 40
    q_pos_fixed = 20
    q_rope_fixed = rope_at_position(q_vec, q_pos_fixed, freqs)

    rel_positions = []
    dots_by_rel = {}
    for k_pos in range(41):
        rel = q_pos_fixed - k_pos  # q_pos - k_pos
        k_rope = rope_at_position(k_vec, k_pos, freqs)
        dot = (q_rope_fixed * k_rope).sum().item()
        rel_positions.append(rel)
        if rel not in dots_by_rel:
            dots_by_rel[rel] = []
        dots_by_rel[rel].append(dot)

    # 对每个相对位置，如果出现多次 (因为 q_pos 不同但 rel 相同)，
    # 检查多次的值是否一致。当前测试只用了 q_pos=20 固定，所以每个 rel 只有一条。
    # 我们换一种方式: 用不同的绝对位置但相同 rel，验证一致性。

    rel_to_check = 5  # 相对位置 = 5
    abs_pairs_for_check = [(10, 5), (20, 15), (50, 45)]
    dots_for_same_rel = []
    for qp, kp in abs_pairs_for_check:
        qr = rope_at_position(q_vec, qp, freqs)
        kr = rope_at_position(k_vec, kp, freqs)
        dots_for_same_rel.append((qr * kr).sum().item())
    max_d = max(dots_for_same_rel) - min(dots_for_same_rel)
    print(f"  相对位置={rel_to_check}，不同绝对位置的 {len(abs_pairs_for_check)} 组:")
    for (qp, kp), d in zip(abs_pairs_for_check, dots_for_same_rel):
        print(f"    q={qp:2d}, k={kp:2d} → Q·K = {d:.6f}")
    print(f"  最大差异: {max_d:.2e}")
    if max_d < 1e-4:
        print("  [OK] 通过!")
    else:
        print("  [FAIL] 失败!")

    # ------------------------------------------------------------------
    # Test 3: 可视化 — 内积随相对位置的变化
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Test 3: 可视化内积 vs 相对位置")

    max_pos = 50
    q_rope_25 = rope_at_position(q_vec, 25, freqs)

    dots_vs_rel = []
    for k_pos in range(max_pos + 1):
        k_rope = rope_at_position(k_vec, k_pos, freqs)
        dot = (q_rope_25 * k_rope).sum().item()
        dots_vs_rel.append(dot)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    rels = [25 - k for k in range(max_pos + 1)]
    plt.plot(rels, dots_vs_rel, 'b-', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='相对位置=0')
    plt.xlabel('相对位置 (q_pos - k_pos)')
    plt.ylabel('Q · K 内积')
    plt.title('RoPE: Q·K 内积随相对位置变化')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 同时展示随机向量的情况 (不施加 RoPE 的话，内积不随位置变化)
    plt.subplot(1, 2, 2)
    # 无 RoPE 时的原始内积
    raw_dot = (q_vec * k_vec).sum().item()
    # 绝对位置变化时的内积 (固定 k 在 0，q 在不同位置)
    dots_vs_abs = []
    for q_pos in range(max_pos + 1):
        qr = rope_at_position(q_vec, q_pos, freqs)
        kr = rope_at_position(k_vec, 0, freqs)  # k 固定在位置 0
        dots_vs_abs.append((qr * kr).sum().item())

    plt.plot(range(max_pos + 1), dots_vs_abs, 'g-', alpha=0.7)
    plt.axhline(y=raw_dot, color='orange', linestyle='--', alpha=0.7,
                label=f'原始内积 (无RoPE) = {raw_dot:.4f}')
    plt.xlabel('绝对位置 (q_pos, k 固定在 0)')
    plt.ylabel('Q · K 内积')
    plt.title('RoPE 后: 内积随绝对位置仍有变化')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n观察左图:")
    print("  - 当相对位置=0 (即 q 和 k 在同一位置)，内积最大")
    print("  - 相对位置偏离 0 越远，内积通常衰减 (但不单调，因为有不同频率)")
    print("  - 这说明 RoPE 天然给近距离 token 更高的 attention score")
    print("\n总结: RoPE 的核心性质 — Q·K 取决于 (m-n)，编码了相对位置信息 [OK]")


# ============================================================================
# 第 6 步 (思考题): 与实际生产代码对比
# ============================================================================
# 读一读下面这些来自真实项目的 RoPE 实现，看看和你写的有何异同:
#
# 1. HuggingFace transformers 的 LlamaRotaryEmbedding:
#    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
#    搜索 LlamaRotaryEmbedding 类
#
# 2. Meta 官方 Llama 实现:
#    https://github.com/meta-llama/llama/blob/main/llama/model.py
#    搜索 apply_rotary_emb
#
# 3. FlashAttention 中的 RoPE:
#    https://github.com/Dao-AILab/flash-attention
#    搜索 rotary_embedding
#
# 思考:
#   - 为什么真实实现通常用 "precompute cos/sin table" 而不是每次计算？
#   - 真实实现中 "interleaved" vs "non-interleaved" RoPE 有什么区别？
#   - YaRN / NTK-aware scaling 等 RoPE 扩展是为了解决什么问题？


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RoPE 手写练习 — 运行所有验证步骤")
    print("=" * 60)

    print("\n>>> Step 1: 感受 2D 旋转")
    demo_step1()

    print("\n>>> Step 2: 计算频率并可视化")
    demo_step2()

    print("\n>>> Step 3: 验证 naive 和 efficient 两种实现")
    demo_step3()

    print("\n>>> Step 4: 验证多头 Q/K 的 RoPE 应用")
    demo_step4()

    print("\n>>> Step 5: 验证核心性质 — 相对位置编码")
    verify_relative_property()
