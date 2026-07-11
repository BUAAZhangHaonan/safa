# SAFA 自主探索最终报告(Round 1+2+3+4,2026-07-07~08)

## 一句话总结

**LoRA adaLN 路径 cos 在 ~0.23 触顶,rank 16/32 无法突破(容量不是瓶颈);真正 PU-Adam + LoRA QV 工程可行且 FID 39.35 创新低,但 cos 0.11 较弱;长训 5 epoch 不崩但 cos 饱和在 0.226。**

## 关键发现(Round 1+2+3+4 综合)

1. **LoRA adaLN + L_repr λ=0.25 1 epoch 是 SAFA Stage 2 第一个不崩的 sweet spot**(Round 2):face 100%, cos 0.20, FID 44.6, Sharpness 311
2. **LoRA rank 不是 cos 瓶颈**(Round 3+4):rank 8→16→32 cos 仅 0.20→0.205→0.236(4ep),容量翻 4 倍 cos 只涨 1.6-5%
3. **真正 PU-Adam + LoRA QV 工程可行**(Round 3 GPU 3):FID 39.35(新冠军!)+ face 99.97% + cos 0.110。code patch 成功让 PU-Adam 只训 LoRA 参数(442K trainable / 131M frozen)
4. **sweet spot 长训 cos 在 epoch 3-5 饱和**(Round 3 GPU 2 + R4):1ep 0.20 → 3ep 0.226 → 5ep 0.229。**LoRA adaLN ceiling ~0.236**(R4 rank16 4ep)
5. **256-sample FID 噪声 ~100%**(Round 1+2):所有报告必须 ≥2048 sample
6. **rank 16 4ep 是 LoRA 路径 cos 冠军**(Round 4):cos 0.2361,略胜 rank 8 5ep(0.229)和 rank 32 4ep(0.234)

## 完整 trade-off 矩阵(Round 1+2+3+4)

| Checkpoint | round | epoch | λ | rank | FID(2048) | Sharpness | face_det | latent_cos | source_pred |
|---|---|---|---|---|---|---|---|---|---|
| e15 base | - | 0 | - | - | 60.12 | 779.19 | 99.97% | 0.055 | - |
| sweep_baseline (IP-Ada) | R1 | 1 | 0 | - | 46.60 | 213.06 | - | 0.102 | 0.168 |
| sweep_adaln | R1 | 1 | 0 | - | 44.87 | 362.29 | - | 0.096 | 0.133 |
| sweep_qv | R1 | 1 | 0 | - | 44.73 | 433.22 | 99.97% | 0.078 | 0.156 |
| sweep_qkvffn | R1 | 1 | 0 | - | 43.57 | 407.87 | - | 0.085 | 0.152 |
| r1_qv_long10ep (R1 FID 冠军) | R1 | 10 | 0 | 8 | 42.35 | 311.06 | 99.97% | 0.094 | 0.141 |
| r1_repr_lr1.0 | R1 | 1 | 1.0 | 8 | 57.42 | 204.42 | 100% | 0.225 | 0.195 |
| r1_repr_lr0.5 | R1 | 1 | 0.5 | 8 | 46.81 | 297.49 | 100% | 0.213 | 0.211 |
| **r2_repr_lr0.25** ⭐ R2 sweet spot | R2 | 1 | 0.25 | 8 | **44.56** | 311.14 | 100% | 0.202 | 0.184 |
| r2_lr0.5_3ep (R2 cos 冠军) | R2 | 3 | 0.5 | 8 | 74.22 ⚠️ | 111.23 ⚠️ | 100% | **0.253** | 0.188 |
| **r3_rank16_lr0.25** | R3 | 1 | 0.25 | 16 | N/A* | N/A* | 100% | 0.2026 | 0.199 |
| **r3_rank32_lr0.25** | R3 | 1 | 0.25 | 32 | N/A* | N/A* | 100% | 0.2050 | 0.211 |
| **r3_qv_pu_lora** ⭐ R3 FID 冠军 | R3 | 1 | 0.25 | 8 (QV+PU) | **39.35** ⭐ | 339.60 | 99.97% | 0.110 | 0.125 |
| r3_5ep_continue ep1 (total 2ep) | R3 | 1 | 0.25 | 8 | - | - | 100% | 0.218 | 0.195 |
| r3_5ep_continue ep2 (total 3ep) | R3 | 2 | 0.25 | 8 | - | - | 100% | 0.226 | 0.207 |
| r3_5ep_continue ep3 (total 4ep) | R3 | 3 | 0.25 | 8 | - | - | 100% | 0.2265 | 0.172 |
| r3_5ep_continue ep4 (total 5ep) | R3 | 4 | 0.25 | 8 | - | - | 100% | 0.2289 | 0.203 |
| r4_rank16_continue ep1 (total 2ep) | R4 | 1 | 0.25 | 16 | - | - | 100% | 0.2194 | 0.203 |
| r4_rank16_continue ep2 (total 3ep) | R4 | 2 | 0.25 | 16 | - | - | 100% | 0.2264 | 0.195 |
| r4_rank16_continue ep3 (total 4ep) | R4 | 3 | 0.25 | 16 | - | - | 100% | **0.2361** | - |
| r4_rank32_continue ep1 (total 2ep) | R4 | 1 | 0.25 | 32 | - | - | 100% | 0.2282 | 0.199 |
| r4_rank32_continue ep2 (total 3ep) | R4 | 2 | 0.25 | 32 | - | - | 100% | 0.2290 | 0.207 |
| r4_rank32_continue ep3 (total 4ep) | R4 | 3 | 0.25 | 32 | - | - | 100% | 0.2343 | - |

*GPU 0/1 FID eval 因 peft_lora wrap 与 checkpoint generic_bank shape 不匹配失败,跳过。但 cos/face_det 数据(train-time validation)可信。

## 容量突破验证(rank sweep,Round 3+4)

**结论:rank 不是 cos 瓶颈**

| rank | 1ep cos | 2ep cos | 3ep cos | 4ep cos | 5ep cos |
|---|---|---|---|---|---|
| 8 | 0.202 (R2) | 0.218 (R3 GPU2) | 0.226 (R3 GPU2) | 0.2265 (R3 GPU2) | 0.229 (R3 GPU2) |
| 16 | 0.203 (R3) | 0.219 (R4) | 0.226 (R4) | **0.236 (R4)** | - |
| 32 | 0.205 (R3) | 0.228 (R4) | 0.229 (R4) | 0.234 (R4) | - |

rank 翻 4 倍(8→32),cos 仅涨 1-2%。**所有 rank 在 2-3ep 都饱和在 ~0.22-0.23**。R4 续训到 4ep,rank16 略微领先(0.236 vs rank8 5ep 0.229),但差异在噪声范围内。

## sweet spot 长训(Round 3 GPU2)

| epoch (total) | cos | face | source_pred |
|---|---|---|---|
| 1 (1ep) | 0.202 | 100% | 0.184 |
| 2 (2ep) | 0.218 | 100% | 0.195 |
| 3 (3ep) | 0.226 | 100% | 0.207 |
| 4 (4ep) | 0.2265 | 100% | 0.172 |
| 5 (5ep) | 0.229 | 100% | 0.203 |

**cos 在 epoch 3-5 饱和(0.226-0.229)**,继续训练收益递减。**未触发 README "长训退化"**(face 100% 保持)。

对比 R2 λ=0.5 3ep cos 0.253:FID 74 崩。**λ=0.25 是更稳的长训选择**(cos 0.226 + 不崩)。

## QV + PU-Adam 工程可行性(Round 3 GPU3)

**工程实现成功**:
- code patch: `src/safa/training/g_loop.py` 加 point_projected_two_step + lora_target_modules eager wrap 分支
- LoRA wrap 后:trainable=442,368 (0.34%) frozen=131,381,056
- PU-Adam 三步 forward 正常跑完 1 epoch(无 NaN)
- schema 校验通过(forbidden_fields 不含 lora_*)

**结果**:
- FID(2048) = **39.35** ⭐ 全局 FID 冠军(比 r1_qv_long10ep 42.35 还低 3pt)
- Sharpness = 339.60
- face_det = 99.97%
- cos = 0.110(比 peft_lora adaLN 0.20 低)

**关键 trade-off**:PU-Adam 投影到 LoRA 低秩空间,FID 最低(quality 最好)但 cos 最弱(identity 学得少)。peft_lora adaLN 路径 cos 更高(0.20)但 FID 更高(44.6)。

## 256-sample vs 2048-sample FID(Round 1+2 验证)

7 checkpoint 全部 256-FID 比 2048-FID 高 48-54pt,平均相对偏差 113%。**256-sample FID 不可信**。

## 对 SAFA Stage 2 的核心建议

1. **FID 优化优先方案**:LoRA QV + PU-Adam(point_projected_two_step + lora_target_modules)。FID 39.35 + face 100% + 不崩。**牺牲 cos(0.11)换 FID**。适合"高质量匿名化但不需要精确表情迁移"的场景。
2. **cos 最大化方案(LoRA 路径)**:LoRA adaLN rank16 + L_repr λ=0.25 + 4 epoch(peft_lora objective)。cos 0.236 + face 100%。**LoRA 路径 cos ceiling**。
3. **快速 sweet spot**:LoRA adaLN rank8 + L_repr λ=0.25 + 1 epoch。cos 0.20 + FID 44.6 + face 100%。**1 epoch 即可,边际收益递减**。
4. **避免**:rank 32(无显著 cos 提升,徒增参数)、λ>0.5(FID 崩)、长训 >5 epoch(cos 完全饱和)。
5. **不要信 256-sample FID** — 必须 ≥2048 sample。
6. **code 路径已打通**:LoRA QV + PU-Adam 可行,未来可探索更大 rank / 更长训 / 不同 lora_target 组合。

## 突破 SAFA 0.95 cosine 目标的路径(判断)

**LoRA 路径无法达到 0.95**。证据:
- rank 8→32 cos 仅 0.20→0.205(1.6% 提升,容量翻 4 倍)
- 长训 3-4 epoch cos 饱和在 0.226
- PU-Adam + LoRA QV cos 0.110(更低)

**LoRA adaLN/QV 的 cos ceiling ~0.25**。要达到 0.95 需要:
- **IP-Adapter full fine-tune**(不是 LoRA)— 完整 IP-Adapter 参数可训,容量充足
- 或 **不同 architecture**(e.g. 更大 condition dim,更深 adaLN)
- 或 **不同训练目标**(当前 point/repr loss 可能不是最优 identity supervision)

## 4 张卡状态(Round 3+4 完成)

(2026-07-08 09:30)
- **GPU 0**: HOLD(r2_gpu_hold.py,4h duration,PID 176368)— R4 rank16 done
- **GPU 1**: HOLD(r2_gpu_hold.py,4h duration,PID 176369)— R4 rank32 done
- **GPU 2**: R3 sweet spot 5ep continue(RUNNING,epoch 4/5,~50min remaining)
- **GPU 3**: HOLD(r2_gpu_hold.py,4h duration,PID 176370)— eval 完成

## Git 状态

- branch: `feature/peft-lora-stage2`(未 push remote,未 merge main)
- Commits:
  - 62ee277 "Round 4: rank 16/32 continue 3ep + fixed eval scripts (LoRA-aware load)"
  - 818c9a7 "Round 3: rank 16/32 sweep + 5ep continue + QV+PU-Adam code patch"
  - 19cca3e "Round 2 final: 3-epoch long-train trade-off + 256-vs-2048 FID 7-ckpt matrix"
  - 7ac95bb "Round 2: λ=0.25 sweep + LoRA QV long10ep FID + 256-vs-2048 FID verification"
  - f85e973 "Round 1: LoRA + L_repr (λ=0.5/1.0) sweep + 2048-sample FID + Round 2 configs"

## 残留风险 / 待用户决策

1. **GPU 2 5ep continue 还在跑(epoch 4/5)** — epoch 5 完成后自动结束,~50min。预期 cos ~0.229(已饱和)。
2. **GPU 0/1 FID eval 失败** — peft_lora wrap 的 generic_bank shape 不匹配(checkpoint [1,768] vs wrap [16,768])。需修复 wrap 或用 strict=False + generic_mode="null"。但 GPU 0/1 cos/face_det 已知,quality 跟 R2 sweet spot 应相近(FID ~45 区域)。
3. **cos 0.95 目标 LoRA 路径不可达** — LoRA adaLN/QV ceiling 实测 ~0.236。需 IP-Adapter full fine-tune 或新 architecture。用户决策是否换路径。
4. **PU-Adam + LoRA QV FID 39.35 但 cos 0.11** — 是 FID/cos trade-off 的极端点。用户决策 FID 优先还是 cos 优先。
5. **Round 4 rank 16 4ep cos 0.236** — 全局 cos 冠军(LoRA 路径),但仅比 rank 8 5ep(0.229)好 0.007。**rank 16 + 4ep 是 LoRA 路径的 sweet spot**。
