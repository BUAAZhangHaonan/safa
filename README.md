# SAFA

[中文版本](README.zh-CN.md)

SAFA studies sample-wise affective face anonymization. Given a source face
image $x_0$, a frozen affect encoder $E_0$, and its normalized embedding

$$
z_0 = E_0(x_0), \qquad \|z_0\|_2 = 1,
$$

the goal is to generate an anonymized face $\hat{x}$ that stays on the face
image manifold while preserving the frozen affect representation:

$$
E_0(\hat{x}) \approx z_0, \qquad \hat{x} \in \mathcal{M}_{\text{face}}.
$$

The method must not be read as identity-supervised face editing. The codebase
does not train with identity loss, ArcFace loss, landmarks, segmentation, or 3D
face conditions. Identity privacy is evaluated only after the generated image
passes utility and single-face guards.

## Current Implementation Status

The repository is an experimental prototype, not a finished privacy release.

The current medium-v1 Stage 1 generator is a conditional flow-matching model
trained from AffectNet pairs $(x_0, E_0(x_0))$. This is useful for testing
whether a small face prior plus representation constraints can train at all, but
it is not a privacy-clean prior: the conditioning signal is still tied to the
source sample. Recent experiments therefore also include null-condition probes
and projected-update diagnostics. A stronger unconditional or pretrained face
prior remains an important next step.

The current preferred representation loss is the point-wise cosine objective.
The Gram relation loss was implemented and tested as an $O(B^2)$ batch
geometry diagnostic, but the current results do not show a clear benefit over
the point-only loss.

## Flow-Matching Generator

The generator is trained as a conditional flow-matching model. For noise
$\xi \sim \mathcal{N}(0,I)$, target image $x_1$, and
$t \sim \mathcal{U}[0,1]$, the interpolation path is

$$
x_t = (1-t)\xi + t x_1,
$$

with target velocity

$$
u_t = x_1 - \xi.
$$

The flow-matching loss is

$$
\mathcal{L}_{\text{FM}}(\theta)
= \mathbb{E}_{x_1,\xi,t}
\left[\|v_\theta(x_t,t;c)-u_t\|^2\right].
$$

In the current prototype, $c$ can be the frozen affect embedding. In the
null-condition and future prior-based setting, $c$ can be an empty or learned
condition that is not tied to the source identity.

## Representation Preservation

For a generated image $\hat{x}=G_\theta(\xi;c(z_0))$, define

$$
z = E_0(\hat{x}), \qquad \|z\|_2 = 1.
$$

The point-wise representation loss is

$$
\mathcal{L}_{\text{point}}
= \mathbb{E}_{x_0,\xi}\left[1-z^\top z_0\right].
$$

When $z$ is close to $z_0$, with spherical angle
$\phi=\arccos(z^\top z_0)$,

$$
1-z^\top z_0 = 1-\cos\phi = \tfrac{1}{2}\phi^2 + O(\phi^4).
$$

Squared geodesic distance and tangent-space error are locally second-order
equivalent to this point-wise cosine loss. This does not prove that all
representation losses behave the same in training. It only explains why the
simple cosine loss is a reasonable baseline.

The implemented Gram diagnostic additionally compares batch-level relation
matrices

$$
K_0 = Z_0Z_0^\top, \qquad K = ZZ^\top,
$$

using only off-diagonal entries. It increases the number of batch constraints
from $O(B)$ point terms to $O(B^2)$ pair terms, but current experiments have
not shown that this improves convergence for SAFA.

## Why Stage 2 Needs Decoupling

A weighted sum such as

$$
\mathcal{L}_{\text{FM}} + \lambda \mathcal{L}_{\text{point}}
$$

is easy to implement, but it mixes two different goals. In medium-v1 M0,
single-face generation was stable, but latent cosine stayed below the formal
privacy guard and image quality degraded during Stage 2. Gradient logs also
showed frequent FM-vs-representation direction conflict.

This motivates a projected two-step update. The goal is not to claim a global
constraint solution has already been achieved. The implemented M3 experiment
tests whether a local first-order representation step can avoid increasing the
mini-batch FM objective to first order.

## Projected Two-Step Update

At a parameter point $\theta$, let

$$
g_{\text{FM}} = \nabla_\theta \mathcal{L}_{\text{FM}}(\theta),
\qquad
g_{\text{repr}} = \nabla_\theta \mathcal{L}_{\text{repr}}(\theta).
$$

The idealized joint update has two parts.

First, take an FM step:

$$
\theta_{t+\frac{1}{2}}
= \theta_t - \eta_{\text{FM}} g_{\text{FM}}(\theta_t).
$$

Second, recompute gradients at $\theta_{t+\frac{1}{2}}$:

$$
\tilde{g}_{\text{FM}}
= \nabla_\theta \mathcal{L}_{\text{FM}}(\theta_{t+\frac{1}{2}}),
\qquad
\tilde{g}_{\text{repr}}
= \nabla_\theta \mathcal{L}_{\text{repr}}(\theta_{t+\frac{1}{2}}).
$$

If $\tilde{g}_{\text{repr}}^\top\tilde{g}_{\text{FM}} \ge 0$, the
representation step is already FM-feasible to first order:

$$
v^* = -\eta_{\text{repr}}\tilde{g}_{\text{repr}}.
$$

If $\tilde{g}_{\text{repr}}^\top\tilde{g}_{\text{FM}} < 0$, the
representation gradient is projected onto the orthogonal complement of the FM
gradient:

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

In the projected conflict case,
$\tilde{g}_{\text{FM}}^\top v^* = 0$. In the non-conflict case,
$\tilde{g}_{\text{FM}}^\top v^* \le 0$. Thus, under the usual small-step
first-order approximation, the representation substep does not increase the
mini-batch FM loss to first order.

This statement is local. It does not by itself guarantee global image quality,
multi-step stability, or successful privacy. If the projected representation
component is nearly zero, the representation objective can stall even though the
projection is mathematically correct.

## Evaluation Protocol

The main utility metrics are:

- latent cosine between $E_0(\hat{x})$ and $E_0(x_0)$;
- source prediction preservation under the frozen $E_0$ classifier;
- generated label accuracy as an auxiliary measure;
- single-face rate, zero-face rate, and multi-face rate.

Image quality is tracked with distribution and no-reference metrics such as FID,
KID, and NIQE. PSNR and SSIM against the source image are not treated as primary
quality metrics because the generated image should not reconstruct the source
identity.

Formal privacy evaluation is gated. A checkpoint must first satisfy utility and
single-face thresholds. Ad-hoc identity probes can be useful for debugging, but
they are not reported as formal privacy passes.

## Research Direction

The current results suggest that the small from-scratch FM prior is a major
bottleneck. The next stronger version of SAFA should separate the face prior from
the source sample more cleanly, for example with a null-condition prior, a larger
flow/diffusion backbone, or a pretrained frozen face prior plus lightweight
conditioning adapters.

The projected update remains useful as a diagnostic and possible optimization
tool, but its practical value must be judged by experiments, not by the
first-order derivation alone.
