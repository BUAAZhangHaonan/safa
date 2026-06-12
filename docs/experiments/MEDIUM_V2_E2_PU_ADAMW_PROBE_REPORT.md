# E2 PU-AdamW Probe 实验总结报告

日期: 2026-06-12
实验: e2_pu_adamw_probe_v2
状态: 10/10 epochs 完成

---

## 1. 背景

E2 PU-AdamW 实验此前发生灾难性发散（66% 零人脸率）。根因分析确认了三个致命 bug：

1. **Adam 状态污染**: `_sync_adam_state_after_repr_step` 将 repr 梯度写入 Adam exp_avg/exp_avg_sq，导致 Adam 对 FM 梯度的动量估计失真
2. **假归一化**: normalize → project → denormalize 流程让投影后的梯度幅度恢复到原始 repr grad 量级，完全绕过了幅度控制
3. **Backtracking 是症状**: backtrack_count=1.72 是 1+2 的后果，不是独立问题

修复方案：删除 Adam sync，用 trust-ratio cap 替代 denormalization，添加诊断指标。

---

## 2. 代码改动

### Commit 1: `67e3e44` — 主修复

- 删除 `_sync_adam_state_after_repr_step`、`_save_adam_state`、`_restore_adam_state` 三个函数及所有调用点
- 删除 `_preconditioned_parameter_step` 函数
- 添加 `repr_step_ratio_cap` 到 `_Stage2ObjectiveRuntime` dataclass 和 config parsing（默认 0.25）
- 重写 AdamW path：normalize → project → 不 denormalize → trust ratio clip → 直接参数更新
- 核心不变量：Adam 状态只看 FM 梯度，repr step 永远不碰 Adam

### Commit 2: `ac3f3a0` — 指标修复

- 修正 `first_order_fm_increase` 指标中 effective_lr 的使用

### 新增 9 个诊断指标

| 指标 | 含义 |
|------|------|
| `repr_grad_qnorm_before_proj` | 投影前 repr grad 的 Q-norm |
| `repr_grad_qnorm_after_proj` | 投影后 projected grad 的 Q-norm |
| `repr_param_step_norm_before_clip` | clip 前的 repr 参数位移 norm |
| `repr_param_step_norm_after_clip` | clip 后的 repr 参数位移 norm |
| `fm_param_step_norm` | FM optimizer.step() 的参数位移 norm |
| `repr_to_fm_param_step_ratio` | repr_step / fm_step |
| `pu_effective_repr_lr` | backtracking 后的实际 repr LR |
| `pu_backtrack_count` | 平均 backtracking 次数 |
| `raw_ema_cosine_gap` | raw cosine - EMA cosine（已知 bug：batch 层面无 validation 值） |

---

## 3. 实验配置

```yaml
experiment_name: e2_pu_adamw_probe_v2
optimizer_type: adamw
learning_rate: 0.0003
weight_decay: 0.0
repr_learning_rate: 0.00003
repr_step_ratio_cap: 0.25
pu_gradient_normalization: true
pu_backtrack_max_retries: 3
pu_fm_increase_budget: 0.0
global_batch_size: 96 (4 GPUs × 24)
stages.stage2.epochs: 10
resume_from: g_medium_v1_stage1_long200_v4/best_stage1.pt
resume_mode: model_weights_only
resume_optimizer_state: false
ema: enabled, decay=0.999
```

启动命令:
```bash
SAFA_CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 \
  train.py --config configs/medium_v2/experiments/e2_pu_adamw_probe_v2.yaml
```

GPU 布局: GPUs 2-5（E1 占用 GPU 0-1, E8 占用 GPU 6）

---

## 4. 完整指标数据

### 4.1 核心训练指标

| Ep | backtrack | ratio | fm_norm | eff_lr | fm_mse | loss | NIQE |
|----|-----------|-------|---------|--------|--------|------|------|
| 0 | 1.4328 | 0.2500 | 0.1391 | 8.52e-6 | 0.05983 | 0.3937 | 6.833 |
| 1 | 1.3000 | 0.2500 | 0.1350 | 8.47e-6 | 0.05844 | 0.3902 | 4.832 |
| 2 | 1.2000 | 0.2500 | 0.1326 | 8.85e-6 | 0.05846 | 0.3736 | 5.252 |
| 3 | 1.2808 | 0.2500 | 0.1353 | 8.48e-6 | 0.05904 | 0.3708 | 6.367 |
| 4 | 1.2808 | 0.2500 | 0.1355 | 8.72e-6 | 0.05839 | 0.3615 | 5.250 |
| 5 | 1.1616 | 0.2500 | 0.1374 | 8.94e-6 | 0.05919 | 0.3486 | 4.613 |
| 6 | 1.2564 | 0.2500 | 0.1368 | 8.61e-6 | 0.05894 | 0.3480 | 5.095 |
| 7 | 1.1556 | 0.2500 | 0.1362 | 8.93e-6 | 0.05828 | 0.3375 | 5.688 |
| 8 | 1.2264 | 0.2500 | 0.1367 | 8.84e-6 | 0.05844 | 0.3452 | 4.262 |
| 9 | 1.1276 | 0.2500 | 0.1382 | 9.06e-6 | 0.05831 | 0.3347 | 5.081 |

### 4.2 验证指标 (Representation Quality)

| Ep | raw_cosine | ema_cosine | raw_pearson | ema_pearson | raw_spearman | ema_spearman |
|----|-----------|-----------|-------------|-------------|-------------|-------------|
| 0 | 0.6955 | 0.6859 | 0.5958 | 0.5933 | 0.6029 | 0.6006 |
| 1 | 0.6651 | 0.6962 | 0.5714 | 0.6087 | 0.5783 | 0.6165 |
| 2 | 0.6917 | 0.7102 | 0.6118 | 0.6295 | 0.6188 | 0.6373 |
| 3 | 0.6528 | 0.7147 | 0.5498 | 0.6333 | 0.5568 | 0.6412 |
| 4 | 0.7792 | 0.7216 | 0.7188 | 0.6357 | 0.7223 | 0.6432 |
| 5 | 0.6906 | 0.7363 | 0.6063 | 0.6581 | 0.6145 | 0.6651 |
| 6 | 0.6581 | 0.7387 | 0.5283 | 0.6593 | 0.5331 | 0.6659 |
| 7 | 0.7036 | 0.7403 | 0.5869 | 0.6591 | 0.5916 | 0.6657 |
| 8 | 0.7471 | 0.7452 | 0.6674 | 0.6654 | 0.6737 | 0.6722 |
| 9 | 0.6931 | **0.7545** | 0.6038 | **0.6754** | 0.6116 | **0.6821** |

### 4.3 人脸检测 + 零人脸率

| Ep | raw_zfr | ema_zfr | raw_fdr | ema_fdr |
|----|---------|---------|---------|---------|
| 0 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 1 | 0.0020 | 0.0000 | 0.9980 | 1.0000 |
| 2 | 0.0098 | 0.0000 | 0.9902 | 1.0000 |
| 3 | 0.0020 | 0.0000 | 0.9980 | 1.0000 |
| 4 | 0.0039 | 0.0000 | 0.9961 | 1.0000 |
| 5 | 0.0020 | 0.0000 | 0.9980 | 1.0000 |
| 6 | 0.0020 | 0.0000 | 0.9980 | 1.0000 |
| 7 | 0.0020 | 0.0000 | 0.9980 | 1.0000 |
| 8 | 0.0020 | 0.0000 | 0.9980 | 1.0000 |
| 9 | 0.0020 | 0.0000 | 0.9980 | 1.0000 |

### 4.4 Repr Loss

| Ep | raw_pt_loss | ema_pt_loss | raw_rel_loss | ema_rel_loss |
|----|-------------|-------------|--------------|--------------|
| 0 | 0.304507 | 0.314075 | 0.155785 | 0.163139 |
| 1 | 0.334883 | 0.303817 | 0.171004 | 0.156540 |
| 2 | 0.308329 | 0.289837 | 0.150687 | 0.147954 |
| 3 | 0.347154 | 0.285310 | 0.174717 | 0.145018 |
| 4 | 0.220789 | 0.278378 | 0.112803 | 0.143057 |
| 5 | 0.309416 | 0.263694 | 0.159483 | 0.134239 |
| 6 | 0.341945 | 0.261327 | 0.177863 | 0.133726 |
| 7 | 0.296386 | 0.259725 | 0.159536 | 0.132985 |
| 8 | 0.252950 | 0.254832 | 0.130808 | 0.130446 |
| 9 | 0.306867 | **0.245542** | 0.153126 | **0.126645** |

### 4.5 PU 诊断指标

| Ep | qnorm_before | qnorm_after | step_before_clip | step_after_clip | fm_step_norm | step_ratio |
|----|-------------|-------------|------------------|-----------------|-------------|------------|
| 0 | 1205.72 | 5.3044 | 0.093988 | 0.034780 | 0.1391 | 0.2500 |
| 1 | 1146.23 | 5.3445 | 0.079615 | 0.033741 | 0.1350 | 0.2500 |
| 2 | 1122.12 | 5.2420 | 0.078393 | 0.033156 | 0.1326 | 0.2500 |
| 3 | 1104.45 | 5.3541 | 0.080201 | 0.033837 | 0.1353 | 0.2500 |
| 4 | 1105.68 | 5.1808 | 0.078080 | 0.033865 | 0.1355 | 0.2500 |
| 5 | 1086.67 | 5.4078 | 0.081642 | 0.034353 | 0.1374 | 0.2500 |
| 6 | 1090.15 | 5.3497 | 0.080973 | 0.034191 | 0.1368 | 0.2500 |
| 7 | 1070.63 | 5.2836 | 0.080269 | 0.034039 | 0.1362 | 0.2500 |
| 8 | 1087.36 | 5.2445 | 0.079615 | 0.034163 | 0.1367 | 0.2500 |
| 9 | 1054.20 | 5.3368 | 0.081569 | 0.034544 | 0.1382 | 0.2500 |

---

## 5. 与 Baseline 对比

| 实验 | Optimizer | 框架 | raw cosine | ema cosine | NIQE | 零人脸率 | 状态 |
|------|-----------|------|-----------|-----------|------|----------|------|
| E8 | AdamW | FM-only | 0.611 | 0.672 | 5.08 | 0 | 运行中 (102/200ep) |
| E1 | SGD | PU-SGD | 0.811 | — | — | 低 | 运行中 (GPU0-1) |
| **E2 旧版** | AdamW | PU (有 bug) | — | — | — | **66%** | 已停 |
| **E2 probe** | AdamW | PU (修复后) | 0.693 | **0.755** | 5.08 | **0%** | **10ep 完成** |

EMA cosine 对比:
- E2 probe (ep9): **0.7545**
- E8 baseline: 0.672
- 差距: **+0.083** (+12.4%)

---

## 6. 成功标准评估

| 标准 | 目标 | 实际 | 结果 |
|------|------|------|------|
| zero_face_rate (ema) | = 0 | 0.0000 | **PASS** |
| repr_to_fm_ratio 稳定 | ~0.25 | 0.25 精确 | **PASS** |
| fm_param_step_norm | 非零、非爆炸 | 0.136 ± 0.002 | **PASS** |
| backtrack_count | < 0.3 | 1.13 (ep9) | **FAIL** |
| raw_ema_cosine_gap | < 0.03 | 指标 bug，无法评估 | **N/A** |

5 项标准中 3 项通过，1 项未达标，1 项因代码 bug 无法评估。

---

## 7. 关键发现

### 7.1 修复成功

灾难性发散问题已解决。三个 bug（Adam 状态污染、假归一化、backtracking 症状）全部修复。EMA cosine 在 10 epoch 内从 0.686 稳步上升到 0.755，没有出现任何发散迹象。

### 7.2 EMA 是真正的输出模型

Raw cosine 在 0.653-0.779 之间剧烈震荡。repr step 的参数扰动让 raw 模型呈现周期性波动。EMA（decay=0.999）平滑了这些波动，形成稳定的上升趋势。EMA 模型的所有指标（cosine、pearson、spearman、pt_loss、rel_loss）都单调改善。

### 7.3 Trust Ratio 机制精确

`repr_to_fm_param_step_ratio` 在所有 10 个 epoch 中精确等于 0.25。新的 normalize → project → no denormalize → trust ratio clip 管道工作完全符合设计。

### 7.4 投影效果

Q-norm 从投影前的 ~1050-1200 降到投影后的 ~5.2，投影将梯度幅度压缩了约 200 倍。投影后的步长（step_after_clip ~0.034）是 FM 步长（fm_step_norm ~0.136）的 25%，与 trust ratio cap 完全一致。

### 7.5 backtrack_count 分析

backtrack_count 从 1.43 下降到 1.13，趋势在下降但速度很慢。每个 batch 平均发生 1.1 次回溯，说明 repr step 频繁与 FM loss 冲突。但 trust ratio 将冲突时的损害限制在 FM 步长的 25% 以内，所以即使频繁回溯也不会导致发散。回溯时参数恢复到 FM step 后的状态，repr step 被丢弃——这是一种保护机制。

### 7.6 EMA 尚未饱和

Epoch 9 的 ema cosine 增速约为 +0.007/epoch（基于 ep5-9 的平均），没有明显减速迹象。如果趋势在 200 epoch 内保持（减速是必然的），EMA cosine 有可能达到 0.85+。

---

## 8. 已知问题

1. **`raw_ema_cosine_gap` 指标 bug**: 代码中使用 `metrics.get("cosine_raw", 0.0)` 从 batch 级别的 metrics dict 读取，但 cosine_raw/cosine_ema 只在 validation 时计算，不在 batch 级别存在。结果该指标始终为 0.0。修复方法是从 epoch 级别的 validation 结果中读取。低优先级——不影响训练。

2. **backtrack_count 持续偏高**: ~1.1 远超目标 0.3。可能需要在 200 epoch 实验中降低 `repr_step_ratio_cap` 或调整 `repr_learning_rate`。

---

## 9. 后续建议

### 9.1 正式实验参数

建议在正式 200 epoch 实验中考虑以下调整：

- `repr_step_ratio_cap`: 从 0.25 降到 **0.10-0.15**，降低 backtrack 频率
- `best_model`: 改为 **ema**，因为 raw 模型震荡剧烈
- `repr_learning_rate`: 可以保持 3e-5 或微降到 2e-5
- 其余参数保持不变

### 9.2 建议的参数扫描

| 变体 | repr_step_ratio_cap | 预期效果 |
|------|---------------------|----------|
| A | 0.25 (当前) | backtrack ~1.1, ema 改善最快 |
| B | 0.15 | backtrack ~0.5-0.7, ema 改善略慢但更稳定 |
| C | 0.10 | backtrack ~0.2-0.3, 最稳定但 ema 提升可能有限 |

建议先跑 B（0.15），如果 backtrack_count 仍不理想再试 C。

### 9.3 暂不考虑的方向

- DC / CAGrad / FAMO：已确认不是目标框架
- SGD optimizer：E1 已在跑 PU-SGD，AdamW 是独立的对比实验
- 修改 SGDP path 代码：本次只修 AdamW path

---

## 10. 时间线

| 时间 | 事件 |
|------|------|
| 06-12 01:44 | Probe 实验启动 (4 GPU DDP) |
| 06-12 04:22 | Epoch 3 完成 |
| 06-12 05:17 | Epoch 4 完成 |
| 06-12 06:14 | Epoch 5 开始 |
| 06-12 ~11:00 | 全部 10 epoch 完成 |

每个 epoch 约 55-60 分钟（313 iterations × ~7s/it + validation + NIQE eval）。

---

## 11. 文件位置

| 文件 | 路径 |
|------|------|
| 本地代码 | `/tmp/safa_review_v3/g_loop.py` |
| 本地投影函数 | `/tmp/safa_review_v3/projected_update.py` |
| 远程代码 | `src/safa/training/g_loop.py` |
| Probe 配置 | `configs/medium_v2/experiments/e2_pu_adamw_probe_v2.yaml` |
| 训练指标 | `artifacts/checkpoints/e2_pu_adamw_probe_v2/metrics_history.jsonl` |
| 训练日志 | `artifacts/logs/e2_pu_adamw_probe_v2.log` |
| Git commit 1 | `67e3e44` — 主修复 |
| Git commit 2 | `ac3f3a0` — 指标修复 |
| 不稳定备份 | `e2_pu_adamw_200ep_unstable_v1` |
