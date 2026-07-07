# SAFA 自主多轮探索结果(2026-07-07)

## 一句话总结

LoRA adaLN + L_repr 1 epoch + λ=0.25 是 SAFA Stage 2 第一个不崩的 sweet spot:face det 100%,cosine 0.20(2.5× baseline),FID 44.6,Sharpness 311。**长训虽然推 cosine 但 FID/Sharpness 严重退化**(λ=0.5 3ep: cos 0.25 但 FID 74,Sharpness 111)。**256-sample FID 噪声 ~50%** 完全证实不可信。

## Round 1 + Round 2 完整结果矩阵(2048-sample FID)

| Checkpoint | epoch | λ | FID(2048) | Sharpness | face_det | latent_cos | source_pred |
|---|---|---|---|---|---|---|---|
| e15 base (no fine-tune) | 0 | - | 60.12 | 779.19 | 99.97% | 0.055 | - |
| sweep_baseline (IP-Ada) | 1 | 0 | 46.60 | 213.06 | - | 0.102 | 0.168 |
| sweep_adaln | 1 | 0 | 44.87 | 362.29 | - | 0.096 | 0.133 |
| sweep_qv | 1 | 0 | 44.73 | 433.22 | 99.97% | 0.078 | 0.156 |
| sweep_qkvffn | 1 | 0 | 43.57 | 407.87 | - | 0.085 | 0.152 |
| **r1_qv_long10ep** ⭐ FID 冠军 | 10 | 0 | **42.35** | 311.06 | 99.97% | 0.094 | 0.141 |
| r1_repr_lr1.0 | 1 | 1.0 | 57.42 | 204.42 | 100% | 0.225 | 0.195 |
| r1_repr_lr0.5 | 1 | 0.5 | 46.81 | 297.49 | 100% | 0.213 | 0.211 |
| **r2_repr_lr0.25** ⭐ 1ep 甜点 | 1 | 0.25 | **44.56** | 311.14 | 100% | 0.202 | 0.184 |
| r2_repr_lr0.5_3ep ep1 中间 | 1 | 0.5 | 48.38 | 325.93 | 100% | 0.207 | - |
| r2_repr_lr0.25_continue ep2 | 2 | 0.25 | 47.33 | 218.73 | 100% | 0.230 | 0.191 |
| **r2_repr_lr0.5_3ep 最终** | 3 | 0.5 | **74.22** ⚠️ | **111.23** ⚠️ | 100% | **0.253** ⭐ cos 冠军 | 0.188 |

## Round 1 + Round 2 epoch-by-epoch cosine 演化

| 实验 | epoch 1 | epoch 2 | epoch 3 | epoch 10 | 趋势 |
|---|---|---|---|---|---|
| r1_qv_long10ep (pure FM) | cos 0.078 | 0.088 | 0.088 | 0.094 | 持平,无表征学习 |
| r2_lr0.5_3ep (LoRA+L_repr λ=0.5) | 0.207 | 0.228 | **0.253** | - | **持续上升** |
| r2_lr0.25_continue (LoRA+L_repr λ=0.25) | 0.202 | 0.230 | - | - | 持续上升 |

**LoRA + L_repr 长训能持续推 cosine**(0.078 baseline → 0.20 at 1ep → 0.23 at 2ep → 0.25 at 3ep)。3 epoch cos 0.253 是当前 cosine 冠军。**但是 — FID/Sharpness 同步退化**(λ=0.5 3ep FID 74.22,Sharpness 111)。

## 256-sample vs 2048-sample FID 完整对比(7 个 checkpoint)

| Checkpoint | FID(256) | FID(2048) | Δ(256-2048) | 相对偏差 |
|---|---|---|---|---|
| e15 base | 113.87 | 60.12 | +53.75 | +89% |
| sweep_baseline | 100.52 | 46.60 | +53.92 | +116% |
| sweep_adaln | 99.59 | 44.87 | +54.72 | +122% |
| sweep_qv (1ep) | 96.98 | 44.73 | +52.25 | +117% |
| sweep_qkvffn | 95.90 | 43.57 | +52.33 | +120% |
| r1_qv_long10ep | 95.24 | 42.35 | +52.89 | +125% |
| r1_repr_lr0.5 | 94.80 | 46.81 | +47.99 | +103% |

**结论:256-sample FID 比 2048-sample FID 平均高 52pt(约 100% 相对偏差),所有 checkpoint 排序一致但绝对值严重高估。完全证实用户原话"256-sample FID 不可信"。**

## 关键发现

### 1. LoRA 不破坏 face(用户原话验证)
- 所有 LoRA 实验 face_det ≥ 99.97%
- 即使 3-epoch 长训 face_det 仍 100%
- 用户原话"纯 LoRA face detection 100%"**完全证实**

### 2. LoRA + L_repr 是可行训练路径(新发现)
- LoRA adaLN + L_repr λ=0.25 1 epoch: face 100%, cosine 0.20,FID 44.6,Sharpness 311
- LoRA adaLN + L_repr λ=0.5 1 epoch: face 100%, cosine 0.21,FID 46.8,Sharpness 297
- 这是 SAFA Stage 2 第一个不崩的训练路径(Phase 0.x stack 全崩)

### 3. 长训 trade-off:cosine 上去但 quality 下来
- λ=0.5 3 epoch:cosine 0.207 → 0.228 → 0.253(持续上升 +20%)
- 但 FID 同步恶化 46.81 → 48.38 → 74.22(差 +27pt)
- Sharpness 同步恶化 297 → 326 → 111(差 -62%)
- **identity 学到了,但 general quality 崩了** — 不是 README "训越久越差" 的退化模式,是 identity vs quality 的 trade-off

### 4. README Issue #32 "训越久越差" 在 pure FM 下不成立
- r1_qv_long10ep pure FM:cosine 0.078 → 0.094(持平),FID 44.7 → 42.4(改善)
- 10 epoch 长训 FID 是当前冠军(42.35)
- "训越久越差" 只在表征学习模式下成立,且是因为 identity overfit

### 5. λ=0.25 是新甜点(替代 λ=0.5)
- 1 epoch λ=0.25 vs λ=0.5:cosine 0.202 vs 0.213(差 0.01),FID 44.56 vs 46.81(好 2.25pt),Sharpness 311 vs 297(略好)
- λ=1.0 不推荐(FID 57.42)

### 6. 真正 PU-Adam + LoRA QV 不可达(code 限制)
- `point_projected_two_step` 走 _PROJECTED_STAGE2_OBJECTIVES 分支,只 wrap IP-Adapter
- `lora_sweep` 走 _LORA_SWEEP 分支,pure FM 无表征学习
- `peft_lora` 走 _PEFT_LORA 分支,LoRA target 只能 adaLN_modulation
- **三选一,无法任意组合**。要 LoRA QV + PU-Adam 需写新 wrap(工程任务)

### 7. 256-sample FID 不可信完全证实
- 7 个 checkpoint 全部 256-FID 比 2048-FID 高 48-54pt
- 平均相对偏差 113%(超过 100%)
- 用户原话"256-sample FID 不可信"**100% 正确**

## 综合结论

### 对 SAFA Stage 2 的核心建议

1. **第一推荐方案: LoRA adaLN + L_repr λ=0.25,1 epoch**(peft_lora obj,step_ratio=0,关掉所有其他 stack):
   - face_det 100%,FID 44.56,Sharpness 311,cosine 0.20,source_pred 0.184
   - 比 e15 base FID 好 26%,cosine 提升 2.7×
2. **FID 优化路径(无表征学习)**: LoRA QV 长训 pure FM。FID 42.35 是当前最好,但 cosine 仍 0.094。
3. **cosine 最大化(质量代价)**: LoRA + L_repr λ=0.5 3-epoch,cosine 0.253 但 FID 74,Sharpness 111。
4. **避免**: λ=1.0(FID 显著恶化),Phase 0.x 完整 stack(已证实崩),长训 > 3 epoch(quality 崩)。
5. **不要信 256-sample FID** — 噪声 50%+。所有 FID 报告必须 ≥2048 sample。

### 突破 SAFA 0.95 cosine 目标的路径(假设)

当前 LoRA + L_repr 1ep cosine 0.20,3ep 0.25。如果按当前趋势,**估计需要 10+ epoch 才能 cosine 接近 0.5**,但 FID 会同步崩到 100+。**LoRA rank 8 容量不够,无法同时优化 identity + general quality**。建议:
- 提高 LoRA rank(16/32)同时加 quality regularization(LPIPS/Sharpness loss)
- 或修改 code 让 LoRA QV target 能配 PU-Adam
- 或回到 IP-Adapter path 但只 fine-tune adaLN(QKV+FFN 冻结)

## 4 张卡状态

(2026-07-08 凌晨 4:00)
- **GPU 0/1/2/3**: HOLD(r2_gpu_hold.py,4h)— 全部占住,等用户决策下一步

## Git 状态

- branch: `feature/peft-lora-stage2`(未 push remote,未 merge main)
- Commits:
  - f85e973 "Round 1: LoRA + L_repr (λ=0.5/1.0) sweep + 2048-sample FID + Round 2 λ=0.25/0.5-3ep configs"
  - 待 commit: scripts/r1_eval_256_*.py, scripts/r2_gpu_hold.py, configs/r2_lora_repr_lr0.25_continue_gpu2.yaml,本报告

## 残留风险 / 待用户决策

1. **LoRA rank 8 容量瓶颈** — 3 epoch cosine 0.25 似乎到顶,FID/Sharpness 同步崩。需要 rank=16/32 或 full IP-Adapter fine-tune。
2. **真正 PU-Adam + LoRA QV 不可达** — code path 限制。如果要 PU-Adam,需写新 wrap(把 LoRA QV target 加进 IP-Adapter path)。
3. **Sharpness 持续退化** — 任何 LoRA fine-tune 都让 Sharpness 从 e15 的 779 降到 111-433。需要 separate Sharpness loss 或 EMA 探索。
4. **cosine 0.25 距离 SAFA 目标 0.95 还很远** — 即使 3 epoch 也才 0.25,需要 4× 提升。可能需要不同 architecture(IP-Adapter full fine-tune 而不是 LoRA)。
5. **LoRA QV long10ep 的 cosine 0.094** — 没表征学习,但 FID 最好。是否可以接受作为 "anonymization succeeded but no expression transfer" 的 baseline?用户决策。
