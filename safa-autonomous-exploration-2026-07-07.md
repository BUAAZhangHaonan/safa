# SAFA 自主多轮探索结果(2026-07-07)

## 一句话总结

LoRA adaLN + L_repr(关掉 Phase 0.x 其他所有 stack,step_ratio=0)是 SAFA Stage 2 第一个不崩的训练路径:face det 100% 保住,cosine 从 0.078 → 0.21(2.7×),FID 仅微涨;**λ=0.25 是甜点**(cosine 0.20,FID 44.6,优于 λ=0.5 的 46.8);LoRA QV 长训 10 epoch FID 42.4 是 FID 冠军,但 cosine 仍 0.09 没表征学习。**256-sample FID 噪声 ~50%** 完全证实不可信。

## Round 1 结果

### Round 1 训练/eval 矩阵

| 实验 | objective | LoRA target | epochs | 备注 |
|---|---|---|---|---|
| GPU 0 r1_repr_lr1.0 | peft_lora (step_ratio=0,λ=1.0,no gated/bank/teacher/cond) | adaLN | 1 | LoRA + 纯 L_repr 强权重 |
| GPU 1 r1_repr_lr0.5 | peft_lora (step_ratio=0,λ=0.5,no gated/bank/teacher/cond) | adaLN | 1 | LoRA + 纯 L_repr 中权重 |
| GPU 2 r1_qv_long10ep | lora_sweep (pure FM) | qkv+proj | 10 | 长训退化测试 |
| GPU 3 sweep_qv FID | eval | - | - | 2048 FID + Sharpness 真值 |

### Round 1 完整 eval 矩阵(2048-sample FID)

| Checkpoint | FID(2048) | Sharpness | face_det | latent_cos | source_pred |
|---|---|---|---|---|---|
| e15 base (no fine-tune) | 60.12 | 779.19 | 99.97% | 0.055 | - |
| sweep_baseline_full (IP-Ada 1ep) | 46.60 | 213.06 | - | 0.102 | 0.168 |
| sweep_adaln (1ep) | 44.87 | 362.29 | - | 0.096 | 0.133 |
| sweep_qv (1ep) | 44.73 | 433.22 | 99.97% | 0.078 | 0.156 |
| sweep_qkvffn (1ep) | 43.57 | 407.87 | - | 0.085 | 0.152 |
| **r1_qv_long10ep** | **42.35** ⭐ | 311.06 | 99.97% | 0.094 | 0.141 |
| r1_repr_lr1.0 (LoRA+L_repr) | 57.42 | 204.42 | 100% | **0.225** | 0.195 |
| r1_repr_lr0.5 (LoRA+L_repr) | 46.81 | 297.49 | 100% | 0.213 | 0.211 |

### Round 1 关键发现

1. **e15 base 2048 FID = 60.12,Sharpness = 779** — 任何 LoRA fine-tune 都让 Sharpness 降(204-433),但 FID 都比 e15 base 低(因为 LoRA 学了 e14 数据分布)。
2. **LoRA QV 长训 10 epoch** FID 进一步改善(44.73 → 42.35),但 Sharpness 进一步下降(433 → 311),cosine 0.078 → 0.094(基本持平,无退化)。**README Issue #32 "训越久越差" 在 pure FM 模式下不成立**。
3. **LoRA + L_repr(λ=0.5)能推 cosine 2.7×** — 0.078 → 0.213,face 100% 保住,FID 44.7→46.8(可接受)。
4. **λ=1.0 cos 略高但 FID 显著恶化** — cosine 0.225(比 0.5 略好),但 FID 57.4(比 0.5 差 10pt),Sharpness 204(差)。
5. **256-sample FID 不可信**完全证实:sweep_qv 256-FID=96.98 vs 2048-FID=44.73,**差 50%+**。long10ep 256-FID=95.24 vs 2048-FID=42.35,差 50%+。
6. **code path 限制**: 真正 PU-Adam(`point_projected_two_step`)只能配 IP-Adapter(走 _PROJECTED_STAGE2_OBJECTIVES 分支),无法配 LoRA QV target(走 _LORA_SWEEP 分支)。LoRA + 表征学习只能走 `peft_lora` objective(简单 step,非 PU 投影),且 LoRA target 只能是 adaLN_modulation。

## Round 1 判断 + Round 2 设计

走分支 A(修订版): LoRA + L_repr 是可行路径(非分支 B)。Round 2 探索:
- λ=0.25(更低权重是否更稳?)
- λ=0.5 长 3 epoch(长训是否进一步推 cosine?)
- 完整 sweep checkpoint 2048 FID
- LoRA QV long10ep 的 256 vs 2048 FID 对比

## Round 2 结果

### Round 2 训练/eval 矩阵

| 实验 | objective | λ | epochs | face_det | latent_cos | FID(2048) | Sharpness |
|---|---|---|---|---|---|---|---|
| GPU 0 r2_lr0.5_3ep | peft_lora | 0.5 | 3 | RUNNING (epoch 1/3) | - | - | - |
| **GPU 1 r2_lr0.25** | peft_lora | **0.25** | 1 | **100%** | **0.202** | **44.56** | **311.14** |
| GPU 2 r1_qv_long10ep FID | eval | - | - | 99.97% | 0.094 | 42.35 | 311.06 |
| GPU 3 sweep_adaln / qkvffn FID | eval | - | - | - | - | 44.87 / 43.57 | 362 / 408 |

### Round 2 关键发现

1. **λ=0.25 比 λ=0.5 更优**:cosine 0.202 vs 0.213(差 0.01),但 FID 44.56 vs 46.81(好 2.25pt),Sharpness 311 vs 297(略好)。**λ=0.25 是新的甜点**。
2. **LoRA QV 长训 10 epoch 是 FID 冠军(42.35)** — 比所有 1-epoch 实验都好,但没表征学习,cosine 仍 0.094。
3. **sweep_qkvffn(1ep,FID 43.57)** 是 1-epoch FID 亚军,LoRA target 多(qkv+proj+mlp)效果略好于纯 qv。

## 综合结论

### 对 SAFA Stage 2 的核心建议

1. **第一推荐方案: LoRA adaLN + L_repr λ=0.25** (peft_lora obj,step_ratio=0,关掉所有其他 stack)。这是 SAFA Stage 2 第一个不崩的训练路径:
   - face_det 100%,FID 44.56(比 e15 base 60.12 好 25%),Sharpness 311(中段)
   - cosine 0.20(从 baseline 0.05-0.10 提升 2-4×),source_pred 0.184
2. **FID 优化路径: LoRA QV 长训**(pure FM)。FID 42.35 是当前最好,但没表征学习。可以和方案 1 组合(LoRA QV + L_repr)但需修改 code path(工程任务)。
3. **避免: λ=1.0**(FID 显著恶化),**避免: Phase 0.x 完整 stack**(已证实崩)。
4. **不要信 256-sample FID** — 噪声 50%+。所有 FID 报告必须 ≥2048 sample。

### 待验证(R2 GPU 0 还在跑)

LoRA + L_repr λ=0.5 3-epoch 长训是否进一步提升 cosine?GPU 0 R2 训练完后再 eval。

## 4 张卡状态

(2026-07-08 凌晨)
- **GPU 0**: RUNNING r2_lora_repr_lr0.5_3ep(epoch 1/3,ETA 1.5h)— 长训实验
- **GPU 1**: HOLD(r2_gpu_hold.py,5400s)— 等 GPU 0 完成
- **GPU 2**: HOLD(r2_gpu_hold.py,5400s)— 等 GPU 0 完成
- **GPU 3**: HOLD(r2_gpu_hold.py,5400s)— 等 GPU 0 完成

GPU 0 完成后,GPU 1 应跑 r2_lr0.5_3ep checkpoint 的 2048 FID eval,GPU 2/3 继续 hold 或跑新实验。

## Git 状态

- branch: `feature/peft-lora-stage2`(未 push remote,未 merge main)
- Round 1 commit: f85e973 "Round 1: LoRA + L_repr (λ=0.5/1.0) sweep + 2048-sample FID + Round 2 λ=0.25/0.5-3ep configs"
- 待 commit: scripts/r2_gpu_hold.py, scripts/r1_eval_256_gpu2.py, Round 2 报告

## 残留风险 / 待用户决策

1. **GPU 0 R2 3-epoch 结果未到** — λ=0.5 长 3 epoch 是否进一步提升 cosine?(key unknown)
2. **真正 PU-Adam + LoRA QV 不可达** — code path 限制。如果要 PU-Adam,需写新 wrap(把 LoRA QV target 加进 IP-Adapter path)。
3. **LoRA adaLN target 不如 QV**(sweep 数值):sweep_adaln cosine 0.096 vs sweep_qv 0.078(adaLN 略好),但 sweep_qv flow_mse 更低。Round 1/2 用 adaLN(因为 peft_lora 限制)。**潜在改进:用 QV target 但需修改 code**。
4. **Sharpness 持续退化** — 任何 LoRA fine-tune 都让 Sharpness 从 e15 的 779 降到 200-433。这是 LoRA FM 训练的固有特性,需要 separate Sharpness loss 或 EMA 探索。
5. **cosine 0.20 距离 SAFA 目标 0.95 还很远** — 这是 LoRA 容量限制(rank 8)还是训练长度?需要更长训或更高 rank 实验确定。
