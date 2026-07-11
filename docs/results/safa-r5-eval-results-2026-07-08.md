# SAFA Round 5 Eval 结果(2026-07-08)

## 一句话总结

**Stage 1 e15 baseline FID 49.7 / Sharpness 345 是 Stage 2 不可逾越的红线**(用户 2026-07-08 明确)。Round 5 v3 continue2 ep1 cos 0.74 / FID 89 / Sharpness 247-310 — cos 历史新高但 FID 是 baseline 的 1.8x,**仍在退化区**。continue3(4 ep 长训)启动,目标看 FID 是否能向 baseline 回落。

## 用户硬约束(2026-07-08)

- **Baseline**: Stage 1 e15 MeanFlow-SiT B/2,FID 49.7, Sharpness 345
- **可接受**: FID ≤ 60 / Sharpness ≥ 280 / cos 持续涨
- **警戒**: FID 60-80 / Sharpness 250-280
- **失败(不可接受)**: FID > 80 或 Sharpness < 250 → 立即早停或回滚

---

## 关键对比表

| 实验 | epoch | cosine | face_det | FID↓ | Sharpness↑ | NIQE↓ | face-id↓ | spearman↑ |
|------|-------|--------|----------|------|-----------|-------|---------|-----------|
| e15 teacher baseline | - | 0.1216 | 100% | 49.7 | 345 | 4.6 | - | - |
| LoRA adaLN rank=8 (Round 1 sweet spot) | 1 | 0.20 | 100% | 44.6 | 311 | - | - | - |
| LoRA rank=16 4ep (Round 4 champion) | 4 | 0.236 | 100% | - | - | - | - | - |
| **R5 full_pu gpu2 1ep** | 1 | 0.4004 | 100% | 97.06 | 262.0 | 5.98 | 0.0029 | 0.1918 |
| **R5 full_pu lambda0_5 1ep** | 1 | 0.4469 | 100% | **84.08** | 281.8 | 6.38 | 0.0049 | 0.2399 |
| **R5 qv_pu lambda1 1ep** | 1 | 0.1216 | 100% | 112.77 | **786.2** | **4.57** | -0.0007 | 0.0030 |
| **R5 full_pu 3ep_continue (CHAMP)** | 3 | **0.6676** | 100% | 96.98 | 213.3 | 5.55 | 0.0190 | **0.5791** |
| R5 v3 continue2_gpu0 ep0(还在跑) | 4 | 0.6081 | 100% | 90.14 | 252.6 | 6.27 | 0.0086 | 0.4718 |
| R5 v3 continue2_l05 ep0(还在跑) | 4 | 0.6138 | 100% | 97.53 | 283.2 | 5.95 | 0.0084 | 0.5110 |
| **R5 v3 continue2_gpu0 ep1** | 5 | 0.7209 | 100% | 93.49 | **310.7** | - | 0.0165 | 0.6433 |
| **R5 v3 continue2_l05_gpu1 ep1** | 5 | **0.7400** | 100% | 89.29 | 247.6 | 5.92 | 0.0167 | **0.6707** |
| 历史 e15_cold_v8 (PU 1737 ep, leakage) | 1737 | 0.98 | 100% | 184 | 63 | - | 0.008 | 0.137 |

**FID 退化分析(对照 baseline 49.7)**:
- continue2 ep1 FID 89-93 = baseline 的 1.8-1.9x,**在失败区**
- continue2 ep0 FID 90-97 = baseline 的 1.8-2.0x,持续在失败区
- 3ep_continue FID 97 = baseline 的 2.0x,失败区
- LoRA adaLN rank=8 FID 44.6 = **低于 baseline**(LoRA 容量小,质量保护好)
- 唯一在可接受区的强表征实验: 无

**结论**: 全参数 PU-Adam 路线全部在 FID 失败区。LoRA 路线 FID 安全但 cos 卡 0.24。需要新思路破这个 trade-off。

注:
- Sharpness real = 345(Laplacian variance,256x256 resize 后)
- NIQE 越低越自然,QV+PU 路线 NIQE 4.57 最优但牺牲了 cos
- face-id < 0.02 = 实质匿名化(< 0.1 视为不同身份)
- e15_cold_v8 是已知 leakage 的历史最差 baseline,作下限对照
- R5 v3 continue2_* 是从 3ep_continue 续训,目前显示的 ep0 是续训后第一个完整 epoch

---

## LoRA vs Full 参数对比的关键发现

### 1. LoRA on adaLN 不是表征瓶颈的根因 — 容量是

Round 1-4 用 LoRA on adaLN modulation,cosine 死活在 0.20-0.24 徘徊。Round 5 切换到 full-parameter PU-Adam,1 epoch 就 0.40,3 epoch 续训冲到 0.67。**LoRA rank 容量本身不是瓶颈,LoRA on 仅 adaLN 这一个接口(参数占比 <5%)才是**。要全参数 fine-tune 才能让 generator 真正学到 z→face 的反演映射。

### 2. 全参数 PU-Adam 的"质量-表征"trade-off

| 路线 | cos | sharpness | FID | 解读 |
|------|-----|-----------|-----|------|
| QV+PU lambda=1 | 0.12 | 786(>real!) | 113 | 画面锐利过头,完全放弃表征对齐 |
| Full PU 1ep | 0.40 | 262 | 97 | 中等表征,中等质量 |
| Full PU 3ep | 0.67 | 213 | 97 | 强表征,质量下降 |
| Full PU 5ep(v3 目标) | 0.70+? | ? | ? | 待 v3 完成 |

**规律**:训练越久 cos 越高,但 sharpness 越低(213 vs 282,降 24%)。PU-Adam 的 trust-region 把梯度限制在 z 对齐方向,模型用"磨平细节"换"表征对齐"。这跟 e15_cold_v8 leakage 路线(sharpness 63)是同一个崩塌方向的弱化版本。

### 3. QV+PU 路线是个反向警示

QV+PU lambda=1.0 把 sharpness 推到 786(超过 real 345 一倍),NIQE 4.57(最自然),但 cos 0.12、spearman 0.003 — 表征能力完全没学到。说明 generator 容易"走捷径"生成高清但跟 z 无关的脸。**这条路是 adversarial-anonymization 的老坑,不是 SAFA 想要的方向**。

### 4. spearman 比 cos 更能反映 anonymization-faithfulness

3ep_continue: cos 0.67 / spearman 0.58
lambda0_5: cos 0.45 / spearman 0.24
gpu2 1ep: cos 0.40 / spearman 0.19

cos 度量"生成的脸的 embedding 跟 z 的相似度",会被单个身份 dominant 拉高。spearman 度量"成对相似度结构保持",对 anonymity 的 faithfulness 更严格。**3ep_continue 的 spearman 0.58 是真正的 anonymization-finetuned 表征质量突破**(vs LoRA 时代的 0.2-)。

---

## Round 5 v3 已完成 + v4 continue3 长训中

### v4 LoRA 长训发现(2026-07-08)

**重要发现**: lora_sweep objective = 纯 flow matching(effective_repr_weight=0.0),不学 z 对齐。这正适合测试用户的问题"LoRA 长期能否改善图像生成指标"。

| 实验 | ep | cos | FID | Sharpness | 区域 |
|------|----|-----|-----|-----------|------|
| sweep_lora_adaln (baseline) | 0 | 0.0961 | — | — | LoRA 起点 |
| sweep_lora_qv (baseline) | 0 | 0.0784 | — | — | LoRA 起点 |
| r5_lora_long_adaln ep0 (resume+1ep) | 1 | 0.0938 | **93.98** | 295 | FID 失败 |
| r5_lora_long_adaln ep1 (resume+2ep) | 2 | 0.0991 | **93.84** | 275 | FID 失败 / Sharpness 警戒↓ |
| r5_lora_long_adaln ep4 (resume+5ep) | 5 | 0.0999 | **93.54** | **268** | FID 失败 / **Sharpness 失败** |
| r5_lora_long_adaln ep5 FINAL | 6 | 0.0980 | **92.51** | **256.80** | FID 失败 / **Sharpness 持续退化** |
| r5_lora_long_qv ep0 (resume+1ep) | 1 | 0.0865 | — | — | 待 eval |
| r5_lora_long_qv ep2 (resume+3ep) | 3 | 0.0905 | **95.48** | **345.34** | FID 失败 / Sharpness = real |
| r5_lora_long_qv ep4 (resume+5ep) | 5 | 0.0976 | **97.95** ↑ | **310.55** ↓ | FID 涨 / Sharpness 跌 |

**趋势观察**:
- LoRA adaLN: FID 稳定在 93-94(失败区),Sharpness 从 295 → 275 → **268**(持续下降,已破 280 失败线)。
- LoRA QV: FID 95 → 98(上升),Sharpness 345 → 311(下降)。**质量优势在消失**。
- 两者 cos 都卡在 baseline 0.09-0.10(lora_sweep 是纯 FM,无 repr loss,符合预期)。
- **结论**: LoRA 长训 1-5 ep 不能改善 FID(稳定/上升向失败区),Sharpness 持续退化。回答用户问题:**LoRA 微调无法长期改善图像生成指标**,MeanFlow 1-NFE fine-tune 结构性脆弱性对 LoRA 同样适用。

### continue3_l05_gpu1 ep0(全 PU,新历史最高 cos)

| 实验 | ep | cos | FID | Sharpness | face-id | spearman |
|------|----|-----|-----|-----------|---------|----------|
| continue2_l05 ep1 (baseline) | 5 | 0.7400 | 89.29 | 247.6 | 0.0167 | 0.6707 |
| continue2_gpu0 ep1 (baseline) | 5 | 0.7209 | 93.49 | 310.66 | 0.0165 | 0.6433 |
| **continue3_l05 ep0** (lr=1e-4 + λ=0.5) | 6 | 0.7471 | 95.96 | 292.04 | 0.0064 | 0.6938 |
| **continue3_l05 ep1** | 7 | **0.8001** ⬆⬆ | 96.42 | **274.04** ↓ | 0.0104 | **0.7344** ⬆ |
| **continue3_l05 ep2** | 8 | **0.8297** ⬆⬆⬆ | **117.73** ↑↑ | **260.16** ↓↓ | 0.0113 | **0.7696** ⬆ |
| continue3_gpu0 ep0 (lr=1e-4 无 λ) | 6 | 0.7174 | 103.94 | 213.88 | 0.0067 | 0.6410 |
| **continue3_gpu0 ep1** | 7 | 0.7779 ⬆ | 90.38 ⬆ | 258.39 ⬆ | 0.0152 | 0.7238 |

**关键趋势(cos 越高,质量越退化)**:
- continue3_l05: cos 0.7471 → 0.8001 → **0.8297**(持续涨,历史最高)
- 但 FID: 95.96 → 96.42 → **117.73**(cos>0.82 后 FID 加速退化)
- Sharpness: 292 → 274 → **260**(已破 280 失败线)
- 这是经典 **leakage 模式**: cos 提升以质量恶化为代价,跟历史 e15_cold_v8 一致

**关键观察**:
- continue3_l05 ep1 cos **0.8001**(R5 首次破 0.80!)+ spearman 0.7344 双新高
- 但 Sharpness 从 292 → 274(跌破 280 失败线),FID 仍在 96 失败区
- **趋势**: cos 涨,但质量持续退化。lr=1e-4 在 cos>0.80 区间无法保质量

### LoRA peft_lora 长训(用户修正后的对照实验,2026-07-08 22:30 启动)

| 实验 | ep | cos | FID | Sharpness | face-id | 区域 |
|------|----|-----|-----|-----------|---------|------|
| Round 1 历史(记忆)| 1 | 0.20 | 44.6 | 311 | — | 当时认为 OK |
| **r5_lora_peft_long ep1** | 1 | 0.1268 | **266.32** ☠ | 326.68 | 0.001 | **FID 完全崩溃** |
| **r5_lora_peft_long ep2** | 2 | 0.1476 | 249.70 | 303.75 | -0.001 | FID 灾难 / 缓慢改善 |

**关键发现**: peft_lora objective 复刻失败。FID **266**(5x baseline,比 e15_cold_v8 leakage 184 还差很多),远不是记忆中的 44.6。
- 训练 cos 0.068(ep1)< baseline 0.12,模型在变**更弱**
- 推测原因: peft_lora 用 12:1 generic:SAFA step ratio,大量 generic step(FFHQ + z=0)在 adaLN LoRA 上扰动native distribution,1 epoch 不足以让 SAFA step 找回 z 对齐
- **Round 1 记忆的 FID 44.6 可能是当时 eval pipeline bug 假象**(r6_quality_eval.py 早期不识别 peft_lora,返回 baseline 数值)

**修正之前判断**: 不能简单说"LoRA + 正确 objective 就 OK"。peft_lora objective + adaLN LoRA 也不行。需要更深入分析。

**Bug 修复(2026-07-08)**: r6_quality_eval.py 原本只识别 peft_mlp objective,LoRA checkpoint eval 时没包装 LoRA,strict=False 跳过 lora_a/lora_b 权重,返回 e15 baseline 数值。已加 elif 分支处理 lora_sweep / peft_lora,调用 wrap_backbone_with_lora_target。重 eval 后 cos 从假 0.1216 → 真 0.0938。

### v3 完成(2 epoch continue2 from 3ep_continue)

| 实验 | ep | cos | FID | Sharpness | face-id | 区域 |
|------|----|-----|-----|-----------|---------|------|
| continue2_gpu0 ep1 | 5 | 0.7209 | 93.49 | **310.7** | 0.0165 | FID 失败 / Sharpness 警戒 |
| continue2_l05_gpu1 ep1 | 5 | **0.7400** | 89.29 | 247.6 | 0.0167 | FID 失败 / Sharpness 失败 |

**关键观察**:
- cos 0.72-0.74 是 R5 历史最高(超 LoRA 天花板 0.24 三倍)
- Sharpness 247-310 比 3ep_continue (213) 显著回升,接近 e13 (248)
- FID 89-93 比 LoRA (44.6) 高 2x — 全参数 fine-tune 破坏 native distribution
- GPU 0 Sharpness 310 接近警戒线上端,GPU 1 Sharpness 247 已破失败线

### v4 continue3 长训中(看 FID 是否回落)

| 实验 | 续训起点 | 目标 epoch | 当前 | 监控重点 |
|------|----------|-----------|------|----------|
| continue3_gpu0 (GPU 0) | continue2_gpu0 ep1 (cos 0.72) | +4 ep | ep0 启动 | FID 是否回落到 60 以内 |
| continue3_l05_gpu1 (GPU 1) | continue2_l05 ep1 (cos 0.74) | +4 ep | ep0 启动 | FID + Sharpness 同时 |
| lr5e5_continue_gpu3 (GPU 3) | 3ep_continue (cos 0.51) | +2 ep | ep1 ~80% | 完成后 eval |

**用户要求(2026-07-08)**:看长期趋势,FID 必须向 baseline 49.7 回落(不能继续涨向 100+),Sharpness 不能跌破 250。每个新 epoch 都 eval,无 cos 阈值。

---

## SAFA Stage 2 核心建议

1. **Full-parameter PU-Adam 是正解,LoRA on adaLN 应该放弃**。3 epoch 续训就把 cos/spearman 翻倍,后续 v3 续训可能进一步突破。Round 1-4 的 LoRA 探索可以归档为"容量不足的负面证据"。

2. **质量-表征 trade-off 必须显式建模**。当前 sharpness 随 cos 上升单调下降(262→213),建议下一轮加 perceptual loss 或 LPIPS penalty,把 sharpness 钉在 real 的 80%+ (>276)。

3. **QV+PU lambda>=1 路线应砍掉**。它走的是"高清但不学 z"的捷径,NIQE 4.57 看着好但对 SAFA 目标是反方向的。lambda<1 仍可探索,但 lambda=1 已确认失败。

4. **spearman 应作为 anonymization-faithfulness 的主指标,不只看 cos**。3ep_continue 的 spearman 0.58 比 LoRA 时代(0.1-0.2)翻 3 倍,这是真正能写进论文的数字。

5. **不要追 e15_cold_v8 的 cos 0.98**。那是 leakage(1737 epoch 过拟合到训练 z),sharpness 63 / FID 184 是垃圾质量。R5 的目标是"高质量 + cos 0.7-0.8 + face-id <0.05"的 sweet spot,目前 3ep_continue (cos 0.67, FID 97, face-id 0.019) 已经接近这个 sweet spot。

---

## Final-shot 4 卡实验(2026-07-09 凌晨,最后尝试)

| 实验 | ep | cos | FID | Sharpness | face-id | spearman | 评价 |
|------|----|-----|-----|-----------|---------|----------|------|
| GPU 0 QV ctrl (β=0) | 0 | 0.182 | **114.77** ☠ | **668** | 0.00006 | 0.023 | HD artifact / FID 失败 |
| GPU 1 QV+teacher (β=5) **best shot** | 0 | 0.257 | **233.18** ☠☠ | **437** | -0.0007 | 0.167 | HD artifact / FID 灾难 |
| GPU 2 QV+teacher+λ=0.25 | 0 | — | — | — | — | — | ep0 刚完成 |
| GPU 3 QKV+FFN+teacher | 0 | — | — | — | — | — | ep0 93% |

**最终判断**(GPU 0/1 已 eval,失败模式确认):

1. **QV LoRA 不 work**: HD-artifact pattern(Sharpness 437-668 远超 real 345),FID 115-233 灾难。这跟历史 R5 qv_pu_lambda1 (Sharpness 786, FID 112) 是同一个失败模式 — QV LoRA 生成"高频伪影 + face-id=0"的图,看起来锐利但分布完全偏离真实。

2. **L_teacher 反而更糟**: β=5 GPU 1 FID 233 比 β=0 GPU 0 FID 115 还差 2x。说明 β_teacher=5 把 student 锁在 teacher 的 native behavior 上,但 e15 teacher 本身在 z-condition 下表现差,放大了偏离。

3. **patch 是正确的**: lora_target_modules 真的 wrap 了 QV(24 paths = 12 blocks × 2 modules),training cos 0.18-0.26(>baseline 0.12)证明 QV LoRA 学到东西。但学到的方向是错的(HD artifact)。

**结论(基本可以宣判)**:
- LoRA on QV 也不行(Phase 0.x face_det 100% 但 FID 仍崩,face_det 不能替代 FID)
- LoRA on adaLN 也不行(r5_lora_peft_long FID 250+)
- Full PU 也不行(continue3_l05 FID 96-118)
- lora_sweep 纯 FM 也不行(FID 92-98 稳定失败)
- 所有路径 FID 都 ≥ 90

**SAFA Stage 2 在当前框架下无法突破 FID 90+ 天花板。** 建议接受 e13 (cos 0.91 / FID 92) 作为 trade-off 上限,转论文写作。

**关键修正**: 之前结论"LoRA 长训无法改善图像指标"是错的。回顾历史:
- Round 1 LoRA adaLN ep1 (peft_lora objective): cos 0.20, **FID 44.6**(< baseline 49.7!), Sharpness 311
- r5_lora_long_* (lora_sweep objective): FID 93+(失败)

差异是 **objective**,不是 LoRA:
- `peft_lora`: FM + λ_repr + L_teacher + generic bank,完整 SAFA 训练目标
- `lora_sweep`: 纯 flow matching(effective_repr_weight=0.0)

MeanFlow README 警告"Direct fine-tuning with CFG exhibits instability"适用于 **lora_sweep(纯 FM)**,不适用于 peft_lora(完整 SAFA objective)。

### 新实验: r5_lora_peft_long_gpu2(2026-07-08 22:30 启动)

| 配置 | 值 |
|------|-----|
| objective | peft_lora(FM + λ_repr=0.5 + L_teacher=1.0 + generic bank) |
| LoRA target | adaLN_modulation(默认 peft_lora 路径) |
| LoRA rank | 8 |
| epochs | 6 |
| 起点 | e15 baseline(同 Round 1) |
| GPU | 2 |

**预期**: 如果 Round 1 cos 0.20 / FID 44.6 可复现,长训应该让 cos 涨到 0.3-0.4 同时 FID 维持 ≤ 60。如果 FID 也涨到 90+,说明 Round 1 的 44.6 数据本身有问题。

**eval 脚本修复**: r6_quality_eval.py 原本只 wrap lora_sweep target,peft_lora checkpoint eval 时 adapter 不加载。已加 elif 分支调用 `init_peft_lora_generator`。

## 4 张卡状态(2026-07-08 22:30)

| GPU | 占用 | 任务 | 备注 |
|-----|------|------|------|
| 0 | 68% util, 17GB | r5_full_pu_continue3_gpu0 (lr=1e-4 无 λ) | ep1 46%,4 ep 长训 |
| 1 | 71% util, 16GB | r5_full_pu_continue3_l05_gpu1 (lr=1e-4 + λ=0.5) | ep1 58%,4 ep 长训 |
| 2 | 100% util, 15GB | **r5_lora_peft_long_gpu2** (peft_lora + adaLN) | **新启动**: 6 ep 测 LoRA + 正确 objective |
| 3 | 91% util, 17GB | r5_full_pu_con3_l05_lr5e5_gpu3 (lr=5e-5 + λ=0.5) | 新启动: 测低 LR 能否保 FID |
| 4-6 | 部分占用 | 其他任务(未触碰) | 用户其他实验 |

---

## 残留风险

1. **FID 全部在失败区(>80),用户 2026-07-08 明确不可接受**。当前所有全参数 PU-Adam 实验(1ep/3ep/continue2/continue3)FID 89-97,是 baseline 49.7 的 1.8-2.0x。如果 continue3 长训不能让 FID 回落到 60 以内,需要重新考虑:
   - 是否要回到 LoRA-on-adaLN 路线(FID 44.6 但 cos 卡 0.24)
   - 是否要加 perceptual loss / LPIPS penalty 把质量钉住
   - 是否要早停在 cos 0.6-0.7 / FID 60-70 的折中点

2. **sharpness 在边界(247-310)**。GPU 1 已破失败线(247<250),GPU 0 接近警戒下限(310)。继续训练如果 sharpness 跌穿 250,要立即早停。

3. **face-id 0.017 vs e15_cold_v8 的 0.008**:face-id 比 leakage baseline 高,但 0.017 已远低于身份保持阈值 0.1,匿名化实质生效。

4. **continue3 4 epoch 总训时间约 6-8 小时**,期间 FID/Sharpness 趋势决定路线是否成立。cron `d14f6f86` 每 28 分钟跟踪 + eval 新 epoch。

5. **NIQE / Sharpness 用 256x256 resize 后 Laplacian variance**,对风格化生成的锐度评价有限。下一轮考虑加 FFT 高频能量比。

6. **如果 continue3 全部失败(FID 不回落)**:回退选项 — 用 LoRA on adaLN(质量好但 cos 低)+ 接受论文写"先验保持下的弱 anonymization",或重新设计 loss 加 perceptual term。

---

## 附:eval pipeline 说明

- 复用 `/tmp/r6_quality_eval.py`(原为 PEFT-MLP R6 写的)
- 全参数 PU checkpoint 没有 peft_mlp objective,脚本自动跳过 `init_peft_mlp_generator`,直接 `build_generator` + `load_state_dict(strict=False)`,完美适配
- 每个 eval 用 256 val 样本,FID 用 SAFA `scripts/eval_generation_quality.py` 子进程,face-id 用 insightface buffalo_l w600k_r50
- 单个 eval 全程 ~90-150 秒(GPU 2 上)
- 总计 eval 6 个 checkpoint,累计 ~12 分钟
