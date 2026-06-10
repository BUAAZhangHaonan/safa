# MEDIUM V2 阶段性实验分析（2026-06-08）

文档时间：2026-06-08。  
本文只整理已有 artifact、日志和资源状态，不引用不存在的新实验。

## 结论

FM-only + adaptive rerank 是当前阶段最实用的方案。它的质量指标接近固定 K48，single-face 为 1.0，候选计算量比固定 K48 少 45.238%。它的 generated-image guard 仍未通过，但 latent cosine 只差 0.000547。

固定 K48 的 generated-image guard 通过了，但它还不是 clean formal privacy pass。formal privacy 被 crop protocol blocker 阻塞，crop variant ArcFace 检出 0 faces，所以 `privacy_skipped=true`。

CAGrad、FAMO、硬投影没有达到当前阶段有效结果标准。CAGrad/FAMO 能把 latent cosine 拉高，但 FID 明显恶化；hard projection 的 FID 也明显偏高。它们不能同时保持质量和 utility。

frozen FM conditioning weights-only 在 epoch1 后仍明显不达标，并已停止。epoch1 的 raw latent cosine 和 FID 都没有比 epoch0 变好，该 5M FM 从头/conditioning-only run 未显示阶段性改善。

formal privacy 还没有 clean pass。

## 关键 artifact

- Adaptive K8->K48：`artifacts/eval/g_medium_v2_stage2_fm_only_probe_gpu5_6_bs48_20260606/formal_candidate_rerank_adaptive_k8_to_k48_20260608`
- Fixed K48 privacy-clean：`artifacts/eval/g_medium_v2_stage2_fm_only_probe_gpu5_6_bs48_20260606/formal_candidate_rerank_k48_privacy_clean_arcface_20260607`
- Frozen FM conditioning weights-only：`artifacts/checkpoints/g_medium_v2_stage2_frozen_fm_conditioning_only_probe_gpu0_bs24_weights_only_20260608`

## 指标表

| 实验 | FID | KID mean/std | NIQE | latent cosine | source preserved | single-face | privacy 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Adaptive K8->K48 | 56.080898 | 0.043123 / 0.008121 | 5.875467 | 0.949453 | 0.898211 | 1.0 | `privacy_guard_pass=false`, `privacy_skipped=true` |
| Fixed K48 privacy-clean | 56.673965 | 0.044617 / 0.007356 | 5.884801 | 0.954156 | 0.903730 | 1.0 | `privacy_guard_pass=true`, `privacy_skipped=true`, `skip_reason=privacy_protocol_blocker` |
| K8 | 55.918 | 0.0421 / n/a | 5.876 | 0.899 | 0.821 | 1.0 | privacy fail |
| FM-only base | 72.203 | 0.0632 / n/a | 4.943 | raw 0.546 / EMA 0.643 | n/a | n/a | 不承担 formal privacy 结论 |
| Frozen FM conditioning weights-only, epoch1 / `stage_epoch=1` | 99.5531 | n/a | n/a | raw 0.56843 / EMA 0.60394 | n/a | n/a | 已停止 |
| Frozen FM conditioning weights-only, epoch0 | 97.086 | n/a | n/a | raw 0.617 / EMA 0.624 | n/a | n/a | 对照点 |

## Adaptive K8->K48 候选成本

| 项 | 数值 |
| --- | ---: |
| mean evaluated candidates | 26.2857 |
| fixed K48 total | 190512 |
| actual total | 104328 |
| saved ratio | 45.238% |
| latent cosine guard threshold | 0.950000 |
| latent cosine mean | 0.949453 |
| guard miss | 0.000547 |

Adaptive K8->K48 的核心价值是用较少候选接近 fixed K48 的 utility 和质量。它没有形成 formal pass，因为 generated-image guard 仍为 false；这个 false 来自 latent cosine 均值略低于 0.95，而不是 face detection 问题。该 run 的 single-face 为 1.0，source preserved 为 0.898211。

## Fixed K48 formal privacy 状态

Fixed K48 privacy-clean 的 generated-image guard 是 true，latent cosine mean 为 0.954156，source preserved 为 0.903730，single-face 为 1.0。

这不是 clean formal pass。原因是 `privacy_skipped=true`，`skip_reason=privacy_protocol_blocker`。阻塞点是 crop variant ArcFace 检出 0 faces，formal privacy metrics 没有完整跑完。这个结果只说明 fixed K48 的 generated-image guard 和 utility 指标成立，不能写成 formal privacy clean pass。

## 负控指标

| 实验 | checkpoint / epoch | FID | latent cosine | 主要事实 |
| --- | --- | ---: | ---: | --- |
| CAGrad | e10 | 268.419 | raw 0.972 / EMA 0.991 | latent 很高，但质量显著变差 |
| FAMO | e4 | 221.758 | n/a | FID 显著偏高 |
| FAMO floor03 | e4 | 202.639 | n/a | FID 仍显著偏高 |
| hard projection | e5 | 135.289 | n/a | 质量仍明显差于 rerank 路线 |

这些负控说明，强投影和多目标梯度方法没有达到当前阶段有效结果标准。它们要么把 latent 拉高但严重伤质量，要么质量仍明显偏高，和 FM-only rerank 的平衡状态不在同一水平。

## Frozen FM conditioning 结果

Frozen FM conditioning weights-only 使用路径：

`artifacts/checkpoints/g_medium_v2_stage2_frozen_fm_conditioning_only_probe_gpu0_bs24_weights_only_20260608`

epoch0 的 raw/EMA latent cosine 为 0.617/0.624，FID 为 97.086。epoch1 / `stage_epoch=1` 的 raw latent cosine 为 0.56843，EMA latent cosine 为 0.60394，raw FID 为 99.5531。

这个结果没有显示出有效改善。它比 rerank 方案的 latent cosine 和 FID 都差很多，也没有比 epoch0 更好。frozen FM conditioning 已在 epoch1 后停止。

## 资源和 git 状态

旧后卡实验已停。GPU0-6 复核时没有 compute 进程，显存占用都是 3 MiB，GPU util 都是 0%。进程列表只看到 monitor 或 stale tmux 命令行，没有正在占用 GPU 的训练或评估进程。

最新代码 commit 至少到：

`9129f3d eval: summarize adaptive rerank candidate cost`

工作区状态里已有未跟踪 `logs/`。本文没有清理它。

## 分析结论

FM-only + adaptive rerank 是当前最实用方案。它用 54.762% 的 fixed K48 候选量，拿到接近 fixed K48 的 FID、KID、NIQE 和 utility。它还没有 formal pass，但 guard miss 只有 0.000547。

固定 K48 的 generated-image guard 和 utility 指标成立，但 formal privacy 还没有 clean pass。它的 blocker 是 crop protocol，不是 generated-image single-face 或 latent cosine。

K8 的质量很好，但 latent cosine 只有 0.899，source preserved 只有 0.821，privacy guard 失败。它不能单独作为当前阶段的可用方案。

CAGrad、FAMO、硬投影没有达到当前阶段有效结果标准。它们不能保持 FM-only rerank 的质量/utility 平衡。

frozen FM conditioning weights-only 的 epoch1 结果表明该 run 在当前 5M 设置下不达标。停止记录属于已有阶段性事实。
