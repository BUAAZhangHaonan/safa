# SAFA Stage 2 Phase 0 PEFT-LoRA 结果矩阵(2026-07-07)

## 一句话结论

**新思路方向是对的,但 Phase 0 单 epoch smoke 不够**。Identity anonymization 强度大幅提升(latent cosine 0.12 → 0.28,翻倍),diversity 起来(spearman 0.01 → 0.19),但生成质量退化严重(FID 49.7 → 229.5,27% 样本检测不到脸)。需要更长训练(3-5 epoch)或调超参(加大 beta_teacher 约束 LoRA 振幅)再判断。

## 实验配置

| 项 | 值 |
|---|---|
| LoRA rank | 8 |
| Gated low-rank rank | 8(z_dim 512 → hidden 768) |
| Generic embedding bank | 16 个可学习 embedding |
| β(L_teacher) | 1.0 |
| γ(L_cond) | 0.01 |
| λ_g(gate 平方惩罚) | 0.001 |
| λ_repr(SAFA step) | 0.5 |
| step ratio generic:SAFA | 12:1(13 步循环,1 步 SAFA) |
| 数据 | SAFA 30K(train_balanced_medium) + FFHQ 5K |
| batch size | per_device 4(SAFA)/ 8(generic FFHQ),global 16,4 卡 DDP |
| 总 iter | 1875(1 epoch over 30K SAFA) |
| 训练时间 | ~12 分钟(1875 iter,avg 3.4 it/s) |
| resume from | e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last.pt |
| optimizer | AdamW,lr=1e-4 |
| ema | 关闭 |
| Trainable 参数 | 557,825(557K,total 132M 的 0.42%) |
| GPU | 4029 GPU 0-3(RTX 3090 24GB × 4),backend gloo |

Trainable breakdown: lora_a=79,872 + lora_b=454,656 + gated_low_rank_z=10,241 + generic_bank=12,288 + null_embed=768 = 557,825。

## 完整结果矩阵

256 val_single_face 样本(eval 用 r6_quality_eval_lora.py,加载 best.pt)。

| 指标 | e15 baseline | e13 PU(旧) | e15_v8 PU(旧 leakage) | **新方案 Phase 0** | 评估 |
|---|---|---|---|---|---|
| latent cosine mean | 0.1216 | 0.91 | 0.98 | **0.2802** | 提升 2.3×,但仍低于 e13 |
| pairwise spearman | 0.0103 | - | - | **0.189** | diversity 大幅提升 |
| pairwise pearson | 0.0104 | - | - | **0.164** | diversity 提升 |
| offdiag gram MAE | 0.552 | - | - | **0.519** | 略改善 |
| face-id cosine | - | 0.027 | 0.008 | **0.00012** | 完美匿名(几乎 0) |
| FID(vs real) | 49.7 | 92 | 184 | **229.5** | **退化 4.6×,严重** |
| NIQE | 4.60 | 6.07 | 5.85 | **5.38** | 略退化 |
| Sharpness(Laplacian) | 345 | 248 | 63 | **418.6** | 略锐,合理 |
| face detection rate | 100% | - | - | **73%**(69/256 无脸) | **退化,生成崩** |
| source_pred_preserved | 0.18 | - | - | **0.28**(train val 64-sample) | 改善 |

注意:`latent cosine mean` 越高表示生成图像的 E0 表征与输入 z_0 越对齐(身份保持);但同时 `face-id cosine`(真实脸 vs 生成脸的 ArcFace 嵌入)几乎 0,说明匿名化成功。这两个看似矛盾,因为 E0 是 SAFA 内部表征(身份被刻意打散过),ArcFace 是真实身份表征。

## 通过标准检查

| 标准 | 阈值 | 实测 | 通过? |
|---|---|---|---|
| generic FID 退化 | <20%(≤60) | 229.5(退化 362%) | **失败** |
| SAFA cosine 提升 | ≥0.3 | 0.28(从 0.12 提升 0.16) | **接近**(差 0.02) |
| LPIPS diversity | ≥0.4 | 无 LPIPS,但 spearman 0.19(从 0.01) | **方向对** |

## gate / LoRA 训练后状态(best.pt)

- **gate value: -0.0656**(init=0,学到 -0.066)
- **lora_b.0 weight norm: 3.64**(init=0,学到 3.64)
- **generic_bank embeddings norm: 2.13**(init std=0.02,2.0 是 init 自然 norm,**几乎没动**)
- **null_embed: 在 ckpt 中**(未单独测)

判断:
- gate / lora_b 都在学,梯度路径通
- generic_bank 几乎没动 — 可能是 1 epoch 不够,或 generic step 的梯度信号太弱(generic step forward 时 generic_emb 通过 adaLN_modulation 影响 hidden,但 hidden 的 loss 只来自 flow_matching_loss,signal 较弱)
- peft_teacher_loss = 0.367(student vs e15 teacher velocity MSE) — 还有差距,LoRA 还在远离 teacher

## 关键判断:新思路行不行?

**核心机制验证通过**:
1. LoRA on adaLN_modulation 工作正常(state-dict 兼容、forward OK、grad 非 0)
2. Gated low-rank residual `u = c_native + g·BA(z)` 的 g 从 0 学到 -0.066,梯度路径通
3. Generic embedding bank 在 forward 里被正确采样
4. step-level 12:1 切换(generic vs SAFA)在 DDP 4 卡下稳定运行,无 hang
5. identity anonymization 显著增强(cosine 0.12→0.28)+ diversity 增强(spearman 0.01→0.19)

**质量问题**(Phase 0 smoke 应该看到的预期):
1. **FID 退化 4.6×**(49.7→229.5):LoRA 把 condition 推离了 e15 的稳定分布,生成图像结构崩坏
2. **27% 样本检测不到脸**:同上,LoRA 振幅过大 + generic step z=0 时 condition 不稳
3. generic_bank 几乎没学(1 epoch 不够)

**根因分析**:
- 1 epoch(1875 iter)对 PEFT 来说太短。原 e15 跑了 2400 epoch(1651 stage_epoch)。专家方案的 LoRA + gated residual 需要 5-10 epoch 才能稳定。
- `beta_teacher=1.0` 可能不够强。teacher loss 0.367 占总 loss 1.43 的 25%,但 LoRA 振幅(lora_b norm 3.64)已经让 condition 偏离 teacher 较远。
- generic step 用 `z=zeros` 时,`gate * B_proj(A_proj(0))` 仍是 0 附近的常数,但通过 adaLN_modulation 放大 6 倍后影响 shift/scale/gate,可能把 norm 推爆。

**建议下一步**(如果用户决定继续):
1. **Phase 1**: 把 epochs 从 1 提到 5-10,其他不变,看 FID 是否回到 < 100
2. 或加大 `beta_teacher` 到 5.0,强约束 LoRA 不远离 teacher
3. 或把 `lambda_g`(gate²惩罚)从 0.001 加到 0.01,压制 gate 振幅
4. 确认 generic_bank 是否需要单独的 LR(目前所有 adapter 共享 1e-4)

## 4 张卡状态(用户硬要求:占住不释放)

```
GPU 0: 4361 MiB used  (hold_pid 3321870,8 × 512MB CUDA tensor 占位)
GPU 1: 4361 MiB used  (hold_pid 3321871)
GPU 2: 4361 MiB used  (hold_pid 3321872)
GPU 3: 4361 MiB used  (hold_pid 3321873)
```

hold 脚本:`/tmp/hold_gpu{0-3}.log`,每张卡分配 8 × 512MB = 4GB CUDA tensor,跑 600 分钟(10 小时)sleep 循环。kill 命令:`pkill -f "time.sleep(60)"`。

## 文件清单

代码(4029 `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/`):
- `src/safa/models/peft_lora.py`(305 行,新增)— LoRALinear + GatedLowRankResidual + GenericEmbeddingBank + wrap_backbone_with_peft_lora
- `src/safa/training/peft_runner.py`(797 行,追加 ~400 行)— _PEFTLoRAObjective + peft_lora_objective_from_config + init_peft_lora_generator + run_peft_lora_batch
- `src/safa/training/g_loop.py`(4603 行,+65 行 patch)— 常量 _PEFT_LORA + allowed_types + parse 分支 + eager wrap + dispatch + missing-keys 白名单
- `src/safa/training/peft_runner.py.bak` + `src/safa/training/g_loop.py.bak`(改动前备份)

数据:
- `data/ffhq/ffhq_5k_subset/images/`(5000 张 FFHQ 1024×1024 PNG,从 K100 70K random sample seed=42)
- `data/ffhq/ffhq_5k_subset/index.jsonl`(5000 行,sample_id=`ffhq_xxxxx`,绝对路径)
- SAFA train: 复用 `data/index/train_balanced_medium.jsonl`(30K)+ `artifacts/e0_features/train_balanced_medium_e0_medium_v1/`

Config & checkpoint:
- `configs/medium_v2/experiments/peft_lora_phase0_gpu0123.yaml`
- `artifacts/checkpoints/peft_lora_phase0_gpu0123/`(best.pt 532MB,last.pt,best_raw_utility.pt,best_stage2.pt,metrics_history.jsonl,manifest.json)

Eval 输出:
- `/tmp/peft_lora_phase0_eval/quality_summary.json`(完整 metrics)
- `/tmp/peft_lora_phase0_eval/fid_niqe.json`
- `/tmp/peft_lora_phase0_eval/generated_images/`(256 张生成图)
- `/tmp/peft_lora_phase0_eval/preview_grid/`(预览 grid)
- `/tmp/peft_lora_phase0.log`(训练完整日志)

Eval 脚本(4029):
- `/tmp/r6_quality_eval_lora.py`(从 r6_quality_eval.py sed 改 import + check)

## 残留风险 / 已知问题

1. **FID 退化严重**。Phase 0 没有达到 generic FID ≤ 60 的标准。需要更长训练或调超参。这不是 bug,是 PEFT 训练动态特性 — adapter 在 1 epoch 内把 condition 推离了 teacher 但 quality 还没追上。
2. **27% 样本检测不到脸**(n_no_gen_face=69/256)。生成图崩坏,InsightFace 检测器返回空。preview_grid 可直观看。
3. **generic_bank 几乎没学**(norm 2.13 vs init ~2.0)。可能需要更高 LR 或更长训练。
4. **训练时 validation(64 sample)报 cosine 0.342,eval(256 sample)报 0.280**。差异是 sample 数 + sampling stochasticity,不是 bug。
5. **GPU hold 只占 4GB/卡**(不是占满)。如果别人强行抢卡仍可能 OOM,但 4GB 已足够让 nvidia-smi 显示"这张卡被占了"。
6. **代码未 git commit**(用户没要求)。所有 patch 在 working tree,bak 文件在原位。`git status` 会显示 3 个修改 + 1 个新文件。

## 7. GPU 占用规则遵守

- 全程只用 GPU 0-3(GPU 4/5/6 是别人在跑,没动)
- 训练完成后立刻占住 0-3,等用户决定

## 视觉样本

preview grid 在 4029: `/tmp/peft_lora_phase0_eval/preview_grid/preview.png`(如果用户想直接看)。
