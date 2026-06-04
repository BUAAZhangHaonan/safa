



## SAFA

### 1. 问题形式化

设原始人脸图像为 $x_0 \in \mathcal{X}$。冻结的情感编码器 $E_0: \mathcal{X} \to \mathbb{S}^{d-1}$ 将其映射为单位超球面上的归一化向量

$$
z_0 = E_0(x_0), \quad \|z_0\|_2 = 1.
$$

生成器 $G_\theta$ 以噪声 $\xi \sim p_0$ 和条件信号 $c$ 为输入，输出人脸图像

$$
\hat{x} = G_\theta(\xi; c).
$$

本方法的目标是生成一张落在自然人脸流形上的图像 $\hat{x}$，使其在 $E_0$ 的表征空间中与 $x_0$ 保持近似一致：

$$
E_0(\hat{x}) \approx z_0, \quad \hat{x} \in \mathcal{M},
$$

其中 $\mathcal{M}$ 表示真实人脸图像所在的低维流形。关键约束在于：生成器不得以 $z_0$ 为条件学习原图重建 $G_\theta(\xi; z_0) \approx x_0$，否则将导致身份信息泄漏。

### 2. 两阶段训练框架

**阶段一：空条件人脸先验**

生成器首先以空条件 $c_\varnothing$ 学习无条件人脸生成。采用流匹配目标，定义线性插值路径

$$
x_t = (1 - t)\xi + t x_1, \quad t \sim \mathcal{U}[0,1],
$$

其中 $x_1 \sim p_{\text{face}}$ 为真实人脸，$\xi \sim \mathcal{N}(0,I)$ 为噪声。目标速度场为

$$
u_t = x_1 - \xi.
$$

对应的流匹配损失为

$$
\mathcal{L}_{\text{FM}}(\theta)
= \mathbb{E}_{x_1,\xi,t}
\left[
\bigl\|v_\theta(x_t,t;c_\varnothing)-u_t\bigr\|^2
\right].
$$

训练完成后得到参数 $\theta^*$。此时 $G_{\theta^*}(\xi;c_\varnothing)$ 学到自然人脸先验，但尚未注入任何与原图相关的表征信息。

**阶段二：受约束的表征控制**

阶段二引入真实表征 $z_0$ 作为控制信号，生成

$$
\hat{x} = G_\theta(\xi;c(z_0)).
$$

随后施加表征保持约束。核心问题是：在推进 $E_0(\hat{x}) \to z_0$ 的同时，不能破坏阶段一已学到的人脸生成能力。

### 3. 表征保持损失

记生成图像的再编码为

$$
z = E_0(\hat{x}).
$$

点级表征保持损失定义为

$$
\mathcal{L}_{\text{repr}}(\theta)
= \mathbb{E}_{x_0,\xi}
\left[
1 - E_0\bigl(G_\theta(\xi;c(z_0))\bigr)^\top z_0
\right].
$$

当 $z \approx z_0$ 时，设球面夹角为

$$
\phi = \arccos(z^\top z_0).
$$

则有局部展开

$$
1 - z^\top z_0
= 1 - \cos\phi
= \frac{1}{2}\phi^2 + O(\phi^4).
$$

因此，点级余弦距离、球面测地距离平方 $\phi^2$ 和切空间误差 $\sin^2\phi$ 在小角度区域局部二阶等价。该结论说明，单纯替换这些点级距离形式并不会改变主要优化结构；方法的关键不在于改写点级距离，而在于如何让表征目标与人脸先验目标解耦优化。

### 4. 约束优化与投影解耦

阶段二不采用无约束加权目标 $\mathcal{L}_{\text{FM}}+\lambda\mathcal{L}_{\text{repr}}$，而将人脸生成质量作为约束：

$$
\min_\theta \mathcal{L}_{\text{repr}}(\theta)
\quad \text{s.t.} \quad
\mathcal{L}_{\text{FM}}(\theta)
\le \mathcal{L}_{\text{FM}}(\theta^*) + \rho,
$$

其中 $\rho \ge 0$ 为允许的先验退化预算。该形式表示：阶段二的主要目标是表征保持，但更新方向必须位于不破坏人脸先验的一阶可行区域内。

在某次迭代中，记

$$
\nabla_{\text{FM}}
= \nabla_\theta \mathcal{L}_{\text{FM}}(\theta),
\quad
\nabla_{\text{repr}}
= \nabla_\theta \mathcal{L}_{\text{repr}}(\theta).
$$

对参数更新方向 $v$ 作一阶展开。为了避免 FM 损失在一阶项上升，要求

$$
\nabla_{\text{FM}}^\top v \le 0.
$$

在该约束下，表征更新方向由以下二次子问题给出：

$$
v^*
= \arg\min_v
\left\{
\nabla_{\text{repr}}^\top v
+ \frac{1}{2\eta}\|v\|^2
\right\}
\quad \text{s.t.} \quad
\nabla_{\text{FM}}^\top v \le 0.
$$

下面假设 $\|\nabla_{\text{FM}}\|_2 > 0$。若

$$
\nabla_{\text{repr}}^\top \nabla_{\text{FM}} \ge 0,
$$

则无约束最优解 $v^*=-\eta\nabla_{\text{repr}}$ 已满足 FM 可行性，因为

$$
\nabla_{\text{FM}}^\top v^*
= -\eta\nabla_{\text{FM}}^\top\nabla_{\text{repr}}
\le 0.
$$

若

$$
\nabla_{\text{repr}}^\top \nabla_{\text{FM}} < 0,
$$

则直接沿 $-\nabla_{\text{repr}}$ 更新会使 FM 损失出现一阶上升。此时最优解落在约束边界上，并等价于将 $\nabla_{\text{repr}}$ 投影到 $\nabla_{\text{FM}}$ 的正交补：

$$
v^*
= -\eta
\left(
\nabla_{\text{repr}}
- \frac{
\nabla_{\text{repr}}^\top\nabla_{\text{FM}}
}{
\|\nabla_{\text{FM}}\|^2
}
\nabla_{\text{FM}}
\right)
= -\eta
P_{\perp\nabla_{\text{FM}}}(\nabla_{\text{repr}}).
$$

因此，两种情形可以统一写成

$$
v^* = -\eta d,
$$

其中

$$
d =
\begin{cases}
\nabla_{\text{repr}},
& \nabla_{\text{repr}}^\top\nabla_{\text{FM}} \ge 0, \\[4pt]
P_{\perp\nabla_{\text{FM}}}(\nabla_{\text{repr}}),
& \nabla_{\text{repr}}^\top\nabla_{\text{FM}} < 0.
\end{cases}
$$

该方向满足

$$
\nabla_{\text{FM}}^\top v^* \le 0.
$$

特别地，在发生投影的冲突情形下，有

$$
\nabla_{\text{FM}}^\top v^* = 0.
$$

所以，表征更新不会使 FM 目标产生一阶上升。若 $\mathcal{L}_{\text{FM}}$ 在当前邻域内为 $L$-光滑，则有

$$
\mathcal{L}_{\text{FM}}(\theta + v^*)
- \mathcal{L}_{\text{FM}}(\theta)
\le \frac{L\eta^2}{2}\|d\|^2.
$$

同时，表征损失的一阶变化为

$$
\nabla_{\text{repr}}^\top v^*
= -\eta\nabla_{\text{repr}}^\top d.
$$

在非冲突情形下，$\nabla_{\text{repr}}^\top d = \|\nabla_{\text{repr}}\|^2$；在冲突情形下，$\nabla_{\text{repr}}^\top d = \|d\|^2$。因此只要 $d \ne 0$，投影表征步就使表征损失一阶下降。

### 5. 两子步更新

为同时优化人脸先验与表征保持，每次迭代拆分为两个子步。

**子步一：流匹配步**

首先沿 FM 目标下降：

$$
\theta_{t+\frac{1}{2}}
= \theta_t
- \eta_1
\nabla_\theta\mathcal{L}_{\text{FM}}(\theta_t).
$$

该子步主动降低流匹配损失，从而继续维护人脸生成能力。

**子步二：投影表征步**

在 $\theta_{t+\frac{1}{2}}$ 处重新计算梯度

$$
\tilde{\nabla}_{\text{FM}}
= \nabla_\theta\mathcal{L}_{\text{FM}}(\theta_{t+\frac{1}{2}}),
\quad
\tilde{\nabla}_{\text{repr}}
= \nabla_\theta\mathcal{L}_{\text{repr}}(\theta_{t+\frac{1}{2}}).
$$

定义

$$
\tilde{d}
=
\begin{cases}
\tilde{\nabla}_{\text{repr}},
& \tilde{\nabla}_{\text{repr}}^\top\tilde{\nabla}_{\text{FM}} \ge 0, \\[4pt]
P_{\perp\tilde{\nabla}_{\text{FM}}}(\tilde{\nabla}_{\text{repr}}),
& \tilde{\nabla}_{\text{repr}}^\top\tilde{\nabla}_{\text{FM}} < 0.
\end{cases}
$$

然后更新

$$
\theta_{t+1}
= \theta_{t+\frac{1}{2}}
- \eta_2\tilde{d}.
$$

该两子步机制给出明确的分工：流匹配步负责推动人脸先验继续下降，投影表征步负责在 FM 一阶可行方向内推动表征保持。由此，FM 与表征目标不再通过一个手工权重直接相加，而是在参数空间中以一阶约束的方式解耦。

### 6. 冻结先验场景的推广

当使用预训练人脸生成模型作为先验时，可以冻结生成器主体，仅训练轻量适配器 $\phi$ 将 $z_0$ 映射为条件信号。此时需要保护的对象不再是训练中的 FM 损失，而是条件注入后对冻结先验的偏移。

定义先验一致性代理为

$$
\mathcal{L}_{\text{prior}}(\phi)
= \mathbb{E}
\left[
\left\|
v_{\theta^*}(x_t,t;c_\phi(z_0))
- v_{\theta^*}(x_t,t;c_\varnothing)
\right\|^2
\right].
$$

于是冻结先验下的约束优化形式为

$$
\min_\phi \mathcal{L}_{\text{repr}}(\phi)
\quad \text{s.t.} \quad
\mathcal{L}_{\text{prior}}(\phi)
\le \rho.
$$

对应的投影规则只需将 $\nabla_{\text{FM}}$ 替换为

$$
\nabla_\phi \mathcal{L}_{\text{prior}}(\phi),
$$

并将训练变量从 $\theta$ 替换为 $\phi$。因此，同一套投影解耦框架可以同时覆盖两种场景：从头训练生成器时保护 FM 目标，使用冻结预训练先验时保护先验一致性。