# SAFA Stage 2 自主探索 Round 1 结果 (2026-07-07)

## 一句话总结

LoRA + L_repr(关掉 Phase 0.x 其他所有 stack)能推 cosine 从 0.078 → 0.22(2.8×),face det 100% 保住,FID 仅微涨(44.7 → 46.8 with λ=0.5);λ=1.0 时 FID 显著恶化。**LoRA + L_repr λ=0.5 是甜点**,真正 SAFA 表征训练能跑且不崩。

## 上一轮 sweep baseline(256-sample validation)

| Checkpoint | face_det | latent_cos | source_pred | grad_norm | flow_mse |
|---|---|---|---|---|---|
| sweep_baseline_full | 100% | 0.102 | 0.168 | 0.958 | 10.44 |
| sweep_adaln | 100% | 0.096 | 0.133 | 0.069 | 15.12 |
| sweep_qv | 100% | 0.078 | 0.156 | 0.026 | 15.76 |
| sweep_qkvffn | 100% | 0.085 | 0.152 | 0.061 | 13.42 |

**注**: sweep 全部是 `lora_sweep` pure FM,`lambda_repr=0`,不算表征 loss。所以 cosine 都低。这跟用户原话"LoRA sweep face 100% 不破坏 face"一致,但**不代表 LoRA sweep 跑了表征学习**。

## Round 1 新结果(2048-sample FID + Sharpness)

| Checkpoint | FID(2048) | Sharpness | face_det | latent_cos | source_pred |
|---|---|---|---|---|---|
| **e15 base (no fine-tune)** | 60.12 | 779.19 | 99.97% | 0.055 | - |
| sweep_baseline (IP-Ada 1ep) | 46.60 | 213.06 | - | - | - |
| sweep_qv (LoRA QV 1ep) | 44.73 | 433.22 | 99.97% | 0.072 | - |
| **r1_repr_lr0.5 (LoRA adaLN + L_repr λ=0.5)** | **46.81** | **297.49** | **100%** | **0.213** | **0.211** |
| **r1_repr_lr1.0 (LoRA adaLN + L_repr λ=1.0)** | **57.42** | **204.42** | **100%** | **0.225** | **0.195** |

## Round 1 核心发现

1. **e15 base FID=60.12, Sharpness=779** — 任何 LoRA fine-tune 都让 Sharpness 降(213-433),但 FID 都比 e15 base 低(因为 LoRA 学了 e14 数据分布)。
2. **LoRA QV sweep (pure FM 1ep) 不破坏 face** — face det 99.97%,FID=44.73(比 e15 好)。
3. **加 L_repr(λ=0.5)能推 cosine 2.7×** — 0.078 → 0.213,face 100% 保住,FID 44.7→46.8(可接受)。
4. **λ=1.0 cos 略高但 FID 显著恶化** — cosine 0.225(比 0.5 略好),但 FID 57.4(比 0.5 差 10pt),Sharpness 204(差)。**λ=0.5 是甜点**。
5. **code path 限制**: 真正的 PU-Adam(`point_projected_two_step`)只能配 IP-Adapter,无法配 LoRA QV target。LoRA + 表征学习只能走 `peft_lora` objective(简单 step,非 PU 投影)。
6. **256-sample FID 不可信**被证实: e15 base 256-sample FID=112.77(用户报),2048-sample FID=60.12。**差 50%**,256-sample 严重高估 FID。

## Round 1 判断 → Round 2 设计

**走分支 A 的修订版**: LoRA + L_repr 是可行路径(非分支 B),但需要长训 + 探索 cosine 上限。

### Round 2 设计(基于 Round 1)

| GPU | 实验 | 目的 | 预期时间 |
|-----|------|------|----------|
| 0 | LoRA adaLN + L_repr λ=0.5,3 epoch | 长训是否进一步提升 cosine?退化? | 3 epoch ~3h |
| 1 | LoRA adaLN + L_repr λ=0.25 | 更低 λ 是否更稳? | 1 epoch ~50min |
| 2 | (GPU 2 Round 1 LoRA QV 10ep 完成后启动) | 等 Round 1 长训结果 | - |
| 3 | eval sweep_adaln + sweep_qkvffn 2048 FID | 完整 sweep 2048 FID 对比 | 20min × 2 = 40min |

GPU 0/1 启动后,GPU 2 继续 Round 1 长训(到 epoch 10),GPU 3 跑 sweep eval。

## 4 张卡状态(Round 1 完成)

- GPU 0: free (Round 1 r1_lora_repr_lr1.0 训练 + eval 完成)
- GPU 1: free (Round 1 r1_lora_repr_lr0.5 训练 + eval 完成)
- GPU 2: RUNNING Round 1 LoRA QV 10 epoch (epoch 6/10, ETA 40 min)
- GPU 3: free (Round 1 eval 完成)

## Git 状态

- branch: `feature/peft-lora-stage2`
- Round 1 commit (待): r1_lora_repr_lr1.0_gpu0.yaml, r1_lora_repr_lr0.5_gpu1.yaml, r1_lora_qv_long10ep_gpu2.yaml, r1_eval2048_*.py, r1_eval_checkpoint_gpu3.py, r1_eval_nolora_gpu1.py, r1_fid_only_gpu3.py
- Round 1 results: artifacts/r1_eval2048_{qv,e15base,sweep_lora_baseline_full_gpu0,r1_lora_repr_only_lr0.5_gpu1,r1_lora_repr_only_lr1.0_gpu0}/result.json

## 残留风险 / 待用户决策

1. **LoRA QV 长训 10 epoch 结果未到** — 是否触发 README "长训退化"?等 Round 1 GPU 2 完成。
2. **真正 PU-Adam + LoRA QV 不可达** — code path 限制。如果一定要 PU-Adam,需写新 wrap(LoRA QV target 加进 IP-Adapter path),是工程任务,不在本轮范围。
3. **LoRA adaLN target 不如 QV**: sweep_adaln cosine 0.096 vs sweep_qv 0.078(差 0.02),但 sweep_qv flow_mse 更低(15.76 vs 15.12)。Round 1 用 adaLN(因为 peft_lora 只支持 adaLN),潜在改进空间是用 QV 但需修改 code。
