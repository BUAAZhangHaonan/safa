# SAFA

[English README](README.md)

SAFA 研究逐样本情感人脸匿名化。给定一张源人脸图像 $x_0$、一个冻结的情感编码器 $E_0$，以及它的归一化嵌入

$$
z_0 = E_0(x_0), \qquad \|z_0\|_2 = 1,
$$

目标是生成一张匿名化后的人脸 $\hat{x}$。这张图像应当保持在人脸图像流形上，同时尽量保留冻结情感表征：

$$
E_0(\hat{x}) \approx z_0, \qquad \hat{x} \in \mathcal{M}_{\text{face}}.
$$

这个方法不应被理解为带身份监督的人脸编辑。代码库当前没有使用身份损失、ArcFace 损失、关键点、分割，或 3D 人脸条件来训练。身份隐私只会在生成结果先通过效用和单人脸约束后，再单独做评估。

## 当前实现状态

这个仓库目前是实验性原型，不是完成版的隐私发布系统。

当前的 medium-v1 Stage 1 生成器是一个条件 flow-matching 模型。它用 AffectNet 配对数据 $(x_0, E_0(x_0))$ 训练。这个版本适合用来检验：一个小的人脸先验加上表征约束，是否至少能训练起来。但它还不能算是隐私上干净的先验，因为条件信号仍然和源样本绑定。最近的实验因此也加入了 null-condition probe 和 projected-update diagnostic。更强的无条件先验或预训练人脸先验，仍然是下一步的重要方向。

当前更偏好的表征损失是逐点 cosine 目标。Gram relation loss 已经实现并测试过，它作为一个 $O(B^2)$ 的批内几何诊断项存在，但目前结果还没有清楚说明它比逐点损失更有优势。

## Flow-Matching 生成器

生成器被训练成一个条件 flow-matching 模型。对噪声 $\xi \sim \mathcal{N}(0,I)$、目标图像 $x_1$，以及 $t \sim \mathcal{U}[0,1]$，插值路径为

$$
x_t = (1-t)\xi + t x_1,
$$

对应的目标速度为

$$
u_t = x_1 - \xi.
$$

flow-matching 损失为

$$
\mathcal{L}_{\text{FM}}(\theta)
= \mathbb{E}_{x_1,\xi,t}
\left[\|v_\theta(x_t,t;c)-u_t\|^2\right].
$$

在当前原型里，$c$ 可以是冻结的情感嵌入。在 null-condition 和未来基于先验的设定里，$c$ 也可以是空条件，或者是一个不和源身份绑定的可学习条件。

## 表征保持

对生成图像 $\hat{x}=G_\theta(\xi;c(z_0))$，定义

$$
z = E_0(\hat{x}), \qquad \|z\|_2 = 1.
$$

逐点表征损失为

$$
\mathcal{L}_{\text{point}}
= \mathbb{E}_{x_0,\xi}\left[1-z^\top z_0\right].
$$

当 $z$ 接近 $z_0$ 时，若球面角
$\phi=\arccos(z^\top z_0)$，则

$$
1-z^\top z_0 = 1-\cos\phi = \tfrac{1}{2}\phi^2 + O(\phi^4).
$$

平方测地距离和切空间误差，在局部二阶意义下都和这个逐点 cosine 损失等价。这并不证明所有表征损失在训练中的行为都一样。它只说明，简单的 cosine 损失可以作为一个合理的基线。

当前实现的 Gram 诊断项还会额外比较批级关系矩阵

$$
K_0 = Z_0Z_0^\top, \qquad K = ZZ^\top,
$$

并且只使用非对角项。这样会把批内约束数量从 $O(B)$ 的逐点项增加到 $O(B^2)$ 的成对项，但当前实验还没有显示这会改善 SAFA 的收敛。

## 为什么 Stage 2 需要解耦

像下面这样的加权和

$$
\mathcal{L}_{\text{FM}} + \lambda \mathcal{L}_{\text{point}}
$$

实现起来很直接，但它把两个不同目标混在一起。在 medium-v1 M0 中，单人脸生成是稳定的，但 latent cosine 仍低于正式隐私门槛，而且 Stage 2 期间图像质量会下降。梯度日志也显示，FM 和表征目标的方向冲突经常出现。

这推动了一个 projected two-step update 的想法。这里的目标不是声称已经得到一个全局约束解。当前实现的 M3 实验只是在测试：一个局部一阶的表征步，是否能在一阶意义下避免增加 mini-batch 的 FM 目标。

## 投影式两步更新

在参数点 $\theta$ 处，记

$$
g_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{FM}}(\theta),
\qquad
g_{\text{repr}} = \nabla_\theta \mathcal{L}_{\text{repr}}(\theta).
$$

理想化的联合更新有两部分。

第一步，先做一个 FM 更新：

$$
\theta_{t+\frac{1}{2}}
= \theta_t - \eta_{\text{FM}} g_{\text{FM}}(\theta_t).
$$

第二步，在 $\theta_{t+\frac{1}{2}}$ 处重新计算梯度：

$$
\tilde{g}_{\text{FM}}
= \nabla_\theta \mathcal{L}_{\text{FM}}(\theta_{t+\frac{1}{2}}),
\qquad
\tilde{g}_{\text{repr}}
= \nabla_\theta \mathcal{L}_{\text{repr}}(\theta_{t+\frac{1}{2}}).
$$

如果 $\tilde{g}_{\text{repr}}^\top\tilde{g}_{\text{FM}} \ge 0$，那么这个表征步在一阶意义下已经对 FM 可行：

$$
v^* = -\eta_{\text{repr}}\tilde{g}_{\text{repr}}.
$$

如果 $\tilde{g}_{\text{repr}}^\top\tilde{g}_{\text{FM}} < 0$，那么会把表征梯度投影到 FM 梯度的正交补上：

$$
v^*
= -\eta_{\text{repr}}
\left(
\tilde{g}_{\text{repr}}
- \frac{\tilde{g}_{\text{repr}}^\top \tilde{g}_{\text{FM}}}
{\|\tilde{g}_{\text{FM}}\|^2}
\tilde{g}_{\text{FM}}
\right).
$$

在发生投影冲突的情形里，$\tilde{g}_{\text{FM}}^\top v^* = 0$。在没有冲突的情形里，$\tilde{g}_{\text{FM}}^\top v^* \le 0$。所以在常见的小步长一阶近似下，这个表征子步不会在一阶意义上增大 mini-batch FM 损失。

这个结论是局部的。它本身并不保证全局图像质量、多步稳定性，或隐私一定成功。如果投影后的表征分量几乎为零，那么即便这个投影在数学上是对的，表征目标也可能停滞。

## 评估协议

主要的效用指标包括：

- $E_0(\hat{x})$ 和 $E_0(x_0)$ 之间的 latent cosine；
- 冻结 $E_0$ 分类器下的源预测保持率；
- 生成标签准确率，作为辅助指标；
- single-face rate、zero-face rate 和 multi-face rate。

图像质量会用分布指标和无参考指标来跟踪，比如 FID、KID 和 NIQE。相对源图像的 PSNR 和 SSIM 不被当作主要质量指标，因为生成图像不应重建源身份。

正式隐私评估是分阶段开启的。一个 checkpoint 必须先满足效用和单人脸阈值。临时性的身份探针可以用于调试，但不会被当作正式隐私通过来报告。

## 研究方向

当前结果说明，从零开始训练的小型 FM 先验很可能是主要瓶颈。下一版更强的 SAFA 应当把人脸先验和源样本分得更开，比如用 null-condition prior、更大的 flow 或 diffusion backbone，或者用冻结的预训练人脸先验再配合轻量条件适配器。

投影式更新仍然可以作为诊断工具和潜在的优化工具，但它的实际价值仍需要靠实验判断，不能只靠一阶推导下结论。
