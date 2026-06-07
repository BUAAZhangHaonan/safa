# SAFA

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

其中 $\mathcal{M}$ 表示真实人脸图像所在的低维流形。关键约束在于：生成器**不得**以 $z_0$ 为条件学习原图重建 $G_\theta(\xi; z_0) \approx x_0$，否则将导致身份信息泄漏。

### 2. 两阶段训练框架

**阶段一：空条件人脸先验**

生成器首先以空条件 $c_\varnothing$ 学习无条件人脸生成。采用流匹配目标，定义线性插值路径

$$
x_t = (1 - t) \xi + t\, x_1, \quad t \sim \mathcal{U}[0,1],
$$

其中 $x_1 \sim p_{\text{face}}$ 为真实人脸，$\xi \sim \mathcal{N}(0, I)$ 为噪声。目标速度场为 $u_t = x_1 - \xi$，训练损失为

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{x_1, \xi, t}\left[ \|v_\theta(x_t, t; c_\varnothing) - u_t\|^2 \right].
$$

训练完成后得到参数 $\theta^*$，此时 $G_{\theta^*}(\xi; c_\varnothing)$ 能生成自然人脸，但尚未注入任何与原图相关的表征信息。

**阶段二：受约束的表征控制**

阶段二引入真实表征条件 $z_0$ 作为控制信号，生成

$$
\hat{x} = G_\theta(\xi; c(z_0)),
$$

并施加表征保持约束。核心问题在于：如何在推进 $E_0(\hat{x}) \to z_0$ 的同时，不破坏阶段一已学到的人脸生成能力。

### 3. 表征保持损失

记生成图像的再编码为 $z = E_0(\hat{x})$。表征保持损失基于超球面上的点级余弦距离：

$$
\mathcal{L}_{\text{repr}}(\theta) = \mathbb{E}_{x_0, \xi}\left[ 1 - E_0\big(G_\theta(\xi; c(z_0))\big)^\top z_0 \right].
$$

> **注**：当 $z \approx z_0$ 时，设球面夹角 $\phi = \arccos(z^\top z_0)$，有
>
> $$
> 1 - z^\top z_0 = 1 - \cos\phi = \tfrac{1}{2}\phi^2 + O(\phi^4).
> $$
>
> 球面测地距离平方 $\phi^2$ 和切空间误差 $\sin^2\!\phi$ 在局部具有相同的二阶展开。因此，将余弦替换为测地距离或切空间误差并不能在根本上改变优化几何——这是本方法以点级余弦为基础、将结构保持留作可选扩展的理论依据。

### 4. 约束优化与投影解耦

阶段二的训练目标不是线性组合 $\mathcal{L}_{\text{FM}} + \lambda \mathcal{L}_{\text{repr}}$，而是将生成质量作为硬约束：

$$
\min_\theta \mathcal{L}_{\text{repr}}(\theta) \quad \text{s.t.} \quad \mathcal{L}_{\text{FM}}(\theta) \le \mathcal{L}_{\text{FM}}(\theta^*) + \rho,
$$

其中 $\rho \ge 0$ 为允许的先验退化预算。

在每次迭代中，记 $\nabla_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{FM}}$，$\nabla_{\text{repr}} = \nabla_\theta \mathcal{L}_{\text{repr}}$。考虑参数更新方向 $v$ 的一阶近似约束

$$
\mathcal{L}_{\text{FM}}(\theta + v) \le \mathcal{L}_{\text{FM}}(\theta) + \epsilon \;\Longrightarrow\; \nabla_{\text{FM}}^\top v \le \epsilon.
$$

当步长充分小时（$\epsilon \to 0$），此约束简化为 $\nabla_{\text{FM}}^\top v \le 0$。在此约束下最小化表征损失的一阶近似与正则项：

$$
v^* = \arg\min_v \left[ \nabla_{\text{repr}}^\top v + \frac{1}{2\eta}\|v\|^2 \right] \quad \text{s.t.} \quad \nabla_{\text{FM}}^\top v \le 0.
$$

该问题有闭式解。若 $\nabla_{\text{repr}}^\top \nabla_{\text{FM}} \ge 0$，无约束最优解 $-\eta \nabla_{\text{repr}}$ 自动满足约束；若二者冲突（内积为负），则需将 $\nabla_{\text{repr}}$ 投影到 $\nabla_{\text{FM}}$ 的正交补上：

$$
v^* = -\eta \left( \nabla_{\text{repr}} - \frac{\nabla_{\text{repr}}^\top \nabla_{\text{FM}}}{\|\nabla_{\text{FM}}\|^2} \nabla_{\text{FM}} \right) = -\eta \, P_{\perp \nabla_{\text{FM}}}(\nabla_{\text{repr}}).
$$

在发生投影的冲突情形下，此更新方向的关键性质为 $\nabla_{\text{FM}}^\top v^* = 0$；在非冲突情形下有 $\nabla_{\text{FM}}^\top v^* \le 0$。因此对充分小的学习率 $\eta$，表征更新不会对 FM 目标造成一阶增加：

$$
\mathcal{L}_{\text{FM}}(\theta + v^*) \le \mathcal{L}_{\text{FM}}(\theta) + O(\eta^2).
$$

这意味着表征保持的更新对生成质量没有一阶伤害。若进一步假设 $\mathcal{L}_{\text{FM}}$ 在当前邻域内 $L$-光滑，则可得到一步漂移的显式上界：

$$
\mathcal{L}_{\text{FM}}(\theta + v^*) - \mathcal{L}_{\text{FM}}(\theta) \le \frac{L \eta^2}{2} \big\|P_{\perp \nabla_{\text{FM}}}(\nabla_{\text{repr}})\big\|^2.
$$

### 5. 两子步更新

为使阶段二的训练同时推进生成质量与表征保持，每步迭代拆分为两个子步。

**子步一（流匹配步）：**

$$
\theta_{t + \frac{1}{2}} = \theta_t - \eta_1 \nabla_\theta \mathcal{L}_{\text{FM}}(\theta_t).
$$

此步沿空条件先验方向继续优化，用于维持并约束人脸生成能力不退化。

**子步二（投影表征步）：** 在更新后的参数点重新计算梯度 $\tilde{\nabla}_{\text{FM}}$ 与 $\tilde{\nabla}_{\text{repr}}$，按上述规则确定更新方向。若 $\tilde{\nabla}_{\text{repr}}^\top \tilde{\nabla}_{\text{FM}} \ge 0$，则直接沿表征梯度下降；否则使用正交投影：

$$
\theta_{t+1} = \theta_{t + \frac{1}{2}} - \eta_2 \cdot \begin{cases} \tilde{\nabla}_{\text{repr}}, & \text{若非冲突} \\ P_{\perp \tilde{\nabla}_{\text{FM}}}(\tilde{\nabla}_{\text{repr}}), & \text{若冲突} \end{cases}
$$

此设计保证：子步一使流匹配损失一阶下降；子步二在投影后仍保留非零表征下降分量时，使表征损失一阶下降，且在冲突方向对生成质量无额外一阶损害。

### 6. 冻结先验场景的推广

当使用预训练人脸生成模型作为先验时，生成器主体被冻结，仅训练轻量适配器 $\phi$ 将 $z_0$ 映射为条件信号。此时保护对象从 $\mathcal{L}_{\text{FM}}$ 替换为先验一致性代理：

$$
\mathcal{L}_{\text{prior}}(\phi) = \mathbb{E}\left[ \big\|v_{\theta^*}(x_t, t; c_\phi(z_0)) - v_{\theta^*}(x_t, t; c_\varnothing)\big\|^2 \right].
$$

约束优化形式与投影更新规则完全一致，仅将 $\nabla_{\text{FM}}$ 替换为 $\nabla_\phi \mathcal{L}_{\text{prior}}$，训练变量从 $\theta$ 换为 $\phi$。此统一性意味着解耦机制不依赖"是在训练整个生成器还是仅微调适配器"——只要存在一个需要保护的人脸先验，投影框架就以相同方式运作。
