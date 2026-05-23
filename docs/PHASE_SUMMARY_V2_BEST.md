# V2 Best 阶段性总结

> 更新时间：2026-05-23  
> 范围：最近一轮 V2 best 相关代码修复、Stage1/Stage2 训练结果、basic eval、full privacy eval 状态。  
> 结论只按已有证据写，不把未完成项写成完成。

## 结论一览

| 项目 | 当前结论 | 证据 |
|---|---|---|
| 当前可报告 checkpoint | `artifacts/checkpoints/g_v2_best/best.pt` | Stage2 `stage_epoch=2`，也是 `best_stage2.pt` |
| 不能用作最佳的 checkpoint | `artifacts/checkpoints/g_v2_best/last.pt` | Stage2 `stage_epoch=3`，训练 loss 继续降，但验证 utility 回落 |
| Stage2 最佳验证表现 | 可报告 | `validation_latent_cosine_mean=0.9740784373`，`validation_source_prediction_preserved=0.916015625`，`validation_face_detection_rate=1.0` |
| Basic eval | 已完成 | `artifacts/eval/g_v2_best_basic_val.json`，4000 samples，face guard passed |
| Full privacy eval | 未完成，不能写 privacy 通过 | `artifacts/logs/eval_g_v2_best.log` 中 ArcFace 对 source 检到 2 张脸并抛错 |
| P1 梯度冲突 logging | 代码已实现 | `src/safa/training/g_loop.py` 已写入相关指标；但当前 checkpoint 没有这些指标 |
| 测试状态 | 最近证据为通过 | `87 passed, 32 subtests passed` |

## 已完成的代码修复

| 类别 | 已完成内容 | 影响 |
|---|---|---|
| P0 训练指标 | 修复 `e0_loop.py` 训练 metric key | 避免训练日志/指标字段错位 |
| P0 维度配置 | 移除训练和 cache 路径中的硬编码维度 | G/E0/cache 的 feature dim 不再靠隐式默认值 |
| P0 cache schema | feature cache 使用严格 schema | cache 的 `feature_dim`、`sample_ids`、`labels` 会被校验 |
| P0 source audit | source audit 覆盖 training/models 代码 | 减少关键路径漏扫 |
| P0 采样确定性 | generator sampling 支持 `x_init`、按 `sample_id` 稳定 seed、`clamp_output` 控制 | 同一个样本在训练、验证、eval 中可复现采样初值 |
| P0 V2 best 配置 | 增加 V2 best Stage1/Stage2 入口配置 | 训练入口可复现 |
| P0 Stage2 batch | Stage2 batch size 降到 16 | 降低 Stage2 显存压力 |
| Eval/visualization | checkpoint 必须显式带 `model_config` | 不再使用 `get("model_config", {})` 这种隐式兜底 |
| Eval/visualization | 移除 eval/visualization 的 224 fallback | image size 必须来自 metadata |
| Eval/visualization | 从 metadata/cache 校验 feature dim | 避免 E0、G、cache 维度不一致时静默运行 |
| P1 梯度冲突 logging | `gradient_cosine_fm_cycle`、`gradient_norm_fm`、`gradient_norm_cycle`、`gradient_conflict_count` 已实现 | 重新跑 Stage2 后可以分析 FM 与 cycle 的梯度关系 |

## 已完成的实验结果

### 训练结果

| 阶段 | checkpoint | stage_epoch | loss | flow_matching_mse | cycle | grad_norm | val latent cosine | val source preserved | val face det |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage1 best | `artifacts/checkpoints/g_v2_best_stage1/best.pt` | 3 | 0.0669606805 | 0.0669606805 | 0.0 | 0.2443749139 | 0.4981328957 | 0.404296875 | 0.994140625 |
| Stage2 best | `artifacts/checkpoints/g_v2_best/best.pt` | 2 | 0.0646579628 | 0.0644469934 | 0.0210969492 | 0.1624730787 | 0.9740784373 | 0.916015625 | 1.0 |
| Stage2 last | `artifacts/checkpoints/g_v2_best/last.pt` | 3 | 0.0639350024 | 0.0637634170 | 0.0171585372 | 0.1500611733 | 0.9539825357 | 0.869140625 | 0.998046875 |

**解释：** Stage2 epoch 3 的训练 loss 仍然下降，但验证侧 utility 回落。当前最佳仍是 Stage2 epoch 2 的 `best.pt`，不是 epoch 3 的 `last.pt`。

### Basic eval

| 项目 | 数值 |
|---|---:|
| 文件 | `artifacts/eval/g_v2_best_basic_val.json` |
| checkpoint | `artifacts/checkpoints/g_v2_best/best.pt` |
| 样本数 | 4000 |
| `latent_cosine.mean` | 0.9682712732 |
| `latent_cosine.p10` | 0.9517689705 |
| `latent_angle_rad.mean` | 0.2106874043 |
| `source_prediction_preserved.mean` | 0.91425 |
| `label_accuracy_generated.mean` | 0.54 |
| `logit_l2_drift.mean` | 0.6839543516 |
| face guard | passed，detection rate 1.0 |
| per-sample 输出 | 4000 rows |
| sample dir | 4 pngs |

### Full privacy eval

| 项目 | 当前状态 |
|---|---|
| 日志 | `artifacts/logs/eval_g_v2_best.log` |
| 结果 | 失败，privacy pass 没有完成 |
| 失败点 | `RuntimeError: Recognizer arcface expected exactly one face, detected 2` |
| 含义 | ArcFace 在 source 图上检到 2 张脸，所以 full privacy eval 中断 |
| 可写结论 | 只能写“full privacy eval incomplete” |
| 不能写结论 | 不能写 privacy 通过，不能写 anonymization privacy 指标已验证 |

## 不能声称完成的内容

| 不能声称 | 原因 | 正确写法 |
|---|---|---|
| `last.pt` 是最佳模型 | `last.pt` 是 Stage2 epoch 3，验证指标低于 epoch 2 | 当前可报告最佳是 `best.pt` / `best_stage2.pt`，Stage2 epoch 2 |
| Full privacy eval 通过 | ArcFace 对 source 检到 2 张脸后报错 | full privacy eval 失败，privacy 指标未完成 |
| 已有梯度冲突结论 | 当前 checkpoint 没有 `gradient_cosine_fm_cycle` 等指标 | logging 代码已实现，需要重新跑 Stage2 才能得到结论 |
| 老的长训练可作为完整结果 | 老训练停在 epoch 4 partial，且当前没有 active tmux | 不作为本轮完整实验结果 |
| 5M 模型代表最终图像质量能力 | 这是小型 FM 原型，不是强 image-quality prior | 只能用于方法验证，后续仍需要更强 prior/backbone |

## 梯度冲突 logging 状态

| 项目 | 状态 |
|---|---|
| 代码位置 | `src/safa/training/g_loop.py` |
| 配置 | `configs/train_g_v2_best.yaml` 中 `stages.stage2.gradient_conflict.enabled=true`，`interval=50` |
| 指标 | `gradient_cosine_fm_cycle`、`gradient_norm_fm`、`gradient_norm_cycle`、`gradient_conflict_count` |
| 当前 checkpoint 是否有指标 | 没有 |
| 原因 | 该 logging 是本轮代码修复后加入的，当前可报告 checkpoint 来自加入前的 Stage2 训练结果 |
| 下一步 | 重新跑 Stage2，才可以判断 FM 与 cycle 是否有梯度冲突 |

## 训练速度慢的原因和 5M 模型定位

| 点 | 事实 | 含义 |
|---|---|---|
| 模型规模 | base_channels=32 的 G/FM 是约 5M 量级 | 这不是大模型训练；它是 prototype / method-validation FM |
| 数据规模 | train 287,651，val 4,000 | 每个 epoch 本身就不小 |
| Stage2 step 数 | global batch 64 over 4 GPUs，对应约 4,495 steps/epoch | 慢主要来自每个 epoch 的 step 数和每步计算量 |
| 每步计算 | 每个 batch 计算 FM + differentiable multi-step `generator.sample` + E0 ResNet50 backprop for cycle | cycle 不是轻量验证项，而是训练图里的反向传播 |
| 采样步数 | schedule 包含 `[4, 8, 16, 32]`，Heun 会增加 vector-field eval 次数 | 步数越高，每个 batch 越慢 |
| face detector | 只在 validation 中使用，不在 main train batch 中使用 | 训练慢不能主要归因于 face detector |
| 图像质量定位 | 5M 量级 FM 不是最终 image-quality prior | 如果目标是更好的图像质量，还需要更强 prior/backbone |

## 下一步建议

| 优先级 | 建议 | 目的 |
|---|---|---|
| 1 | 用当前代码重新跑 Stage2，并保留 gradient conflict 指标 | 得到 FM 与 cycle 梯度关系的真实结论 |
| 2 | 修复或隔离 full privacy eval 中 source 多脸样本问题 | 让 ArcFace privacy pass 能完整跑完 |
| 3 | 报告时固定使用 `artifacts/checkpoints/g_v2_best/best.pt` | 避免把 epoch 3 `last.pt` 当成最佳 |
| 4 | 后续图像质量改进时换更强 prior/backbone | 当前 5M 量级 FM 只适合方法验证 |

## 可追溯提交

| commit | 说明 |
|---|---|
| `6c7dfae` | fix: require explicit generator metadata in eval paths |
| `70329fe` | feat: log stage2 objective gradient cosine |
| `3e99a53` | fix: reduce v2 best stage2 batch size |
| `2500806` | chore: add v2 best stage1 config |
| `cfa8185` | fix: harden training audit entrypoint |
| `03c7b57` | chore: add reproducible v2 best training entrypoint |
| `eef62be` | fix: tighten deterministic sampling contracts |
| `a734b87` | feat: stabilize generator sampling by sample id |
| `72eb9e5` | feat: support deterministic generator sampling |
| `e31368f` | fix: enforce strict feature cache schema |
| `584f5e0` | refactor: make model dimensions configurable |
| `b45d971` | fix: align e0 train metric key |
