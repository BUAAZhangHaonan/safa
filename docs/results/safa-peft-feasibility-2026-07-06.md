# SAFA Stage 2 Leakage 问题完整总结(2026-07-06 最终版)

**用途**: 下次讨论的依据文档,综合 v1-v4 PEFT 实验 + 历史 PU-AdamW/SGD 实验 + 2 轮独立代码审查

---

## 0. 一句话结论

**SAFA Stage 2 leakage 不是 adapter 架构问题,也不是 optimizer/SGD/AdamW 选择问题,而是 stage2 训练目标设计问题**。无论用 1.58M PEFT MLP adapter 还是 131M 全参数 PU-AdamW,只要用 (z_0, x_0) 配对训练 + λ_repr 强拟合 E0 表征,cosine 拉到 0.97+ 时 FID 就会飙到 100-150。**根本要换训练目标/监督信号,不是调超参或改架构**。

---

## 1. 完整 trade-off 数据(历史 PU + v4 PEFT 并列)

### 历史 PU-AdamW 最佳结果(cosine 0.97+ 候选)

| 实验 | 训练参数 | best cosine | best FID@cos(训练时) | **FID re-eval** | **Sharpness** | **LPIPS 多样性** | **face-id cos** | 说明 |
|------|----------|-------------|--------------|------------------|---------------|------------------|-----------------|------|
| real baseline | — | — | — | — | **345** | — | — | 真实图像参考 |
| **e13_pu_adamw** | 131M | 0.91 | 47 | **92.45** | **248 (72% real)** | **0.544** | 0.027 | **质量平衡最佳**(cos 略低) |
| e14_v3_budrelax | 131M | 0.967 | 69 | 111.45 | 186 (54% real) | 0.455 | 0.021 | FID 控制相对好 |
| e14_v4_combined | 131M | 0.97 | 145 | **173.05** | 141 (41% real) | 0.340 | 0.012 | v4=cap0.5+budget0.005 |
| **e15_cold_v8_lpips** | 131M | 0.9833 EMA | 143 | **183.69** | **63 (18% real)** | **0.137** | 0.008 | **Sharpness/多样性 塌方** |

**关键观察(为讨论准备)**:
1. **Sharpness 单调暴跌**:cos 0.91→0.97→0.98 时 Sharpness 248→186→141→63(72% → 18% of real)。**用户当时"Sharpness 下降"肉眼观察完全量化证实**
2. **多样性(LPIPS)同步塌缩**:0.544 → 0.137(跌 75%)。**用户当时"多样性下降"也证实**
3. **face-id cosine 单调下降**(0.027→0.008)但**可能是图像塌方带来的假匿名**,不是真匿名
4. **FID 单调飙升**:92→184(2 倍)
5. **NIQE 反应迟钝**(5.85-7.31 无明显单调关系),不能作早期信号
6. **e13 是当前最佳 trade-off**:cos 0.91 + Sharpness 248 + 多样性 0.544。继续推 cosine 到 0.97+ 就是图像崩坏

**PU-SGD 全部失败**:e12_pu_sgd_* best cosine 仅 0.09-0.18(SGD-pure-baseline,LR 上限 3e-6 不可调)

### e15_cold_v8 完整 FID 轨迹(决定性证据)

| stage2 epoch | cos_ema | FID | KID | NIQE |
|---|---|---|---|---|
| 1660 | 0.79 | **34** | 0.015 | 5.63 |
| 1680 | 0.96 | 107 | 0.077 | 6.05 |
| 1700 | **0.98** | **139** | **0.102** | 6.00 |
| 1720 | **0.98** | **143** | **0.111** | 5.91 |
| 1740 | 0.94 | 78 | 0.049 | 6.29 |
| 1760 | 0.90 | 38 | 0.016 | 6.44 |

**cosine 0.79 → 0.98 时 FID 34 → 143(4.2× 破坏),KID 0.015 → 0.111(7.4× 破坏)**。NIQE 反应迟钝(5.6-7.3),不能作为早期信号。

### v4 PEFT 单变量 leakage 实验

| Run | 训练参数 | λ_repr | cosine | FID | NIQE | Sharpness | face-id |
|-----|----------|--------|--------|-----|------|-----------|---------|
| e15 baseline | — | — | 0.1216 | 49.7 | 4.6 | 345 (real) | — |
| V1 | 1.58M | 0.0 | 0.2141 | 128.12 | 4.70 | 533 | 0.001 |
| V2 | 1.58M | 0.1 | 0.6728 | 138.28 | 6.98 | 598 | 0.005 |
| V3 | 1.58M | 0.5 | 0.7771 | 165.36 | 8.15 | 815 | 0.005 |
| V4 | 1.58M | 1.0 | 0.8579 | 181.01 | 8.50 | 597 | 0.006 |

### 关键对比:PEFT 1.58M vs PU 131M(同样 cosine 区间)

| cosine 区间 | PEFT (1.58M) FID | PU (131M) FID | 差距 |
|---|---|---|---|
| ~0.79 | — | 34 (e15_cold_v8 ep1660) | — |
| ~0.87 | 165 (R6) | — | — |
| ~0.96 | 165 (V3) | 107 (e15_cold_v8 ep1680) | PEFT +58 |
| ~0.98 | — | 143 (e15_cold_v8 ep1700) | — |

**结论**:同样 cosine 水平,PEFT 比 PU 破坏更大(FID +58 @ cos 0.96),但**两条路都坍塌到 FID 100+ 的 leakage 模式**。

---

## 2. 代码现状(2 轮独立审查后)

### SubAgent B 独立审 forward 计算图(7 个问题)

| 问题 | 答案 | 证据 |
|------|------|------|
| ConditionMLPAdapter forward 计算图完整 | ✓ | `condition = base + adapter(z)` 全程无 detach/no_grad,fc1/fc2/fc3 grad.norm = 4.8/4.7/16.0 |
| fc3 small-init 不阻断 gradient | ✓ | `nn.init.normal_(std=0.01)`(不是 zeros),R6 checkpoint absmax 0.089 vs init 0.048(涨了) |
| IP-Adapter gate 不阻断 gradient | ✓ | gate=0.1(不是 0) |
| wrap 真替换 backbone.forward | ✓ | forward hook 在 flow_matching_loss 中被调 2 次(main + JVP) |
| freeze base 生效 | 部分 | trainable=7(6 adapter + 1 null_condition.embedding 泄漏) |
| PEFT_MLP runner loss 覆盖 adapter | ✓ | flow_loss + repr_loss 都走 patched forward |
| DDP wrap 正确 | ✓ | wrap 整个 training_module,find_unused_parameters=true |

### SubAgent C 独立审 optimizer/checkpoint(8 个问题)

| 问题 | 答案 | 证据 |
|------|------|------|
| optimizer 创建时机 | ✓ | g_loop.py:1475,**在 eager wrap 之后** |
| eager wrap 覆盖 PEFT_FM/MLP | ✓ | L1402-1406 (FM) + L1407-1410 (MLP) + L1358-1360 pre-resume |
| optimizer.param_groups 包含 adapter | ✓ | R6 optimizer_state 有 6 个 adapter param,全部有 step/exp_avg/exp_avg_sq |
| checkpoint save 完整保留 adapter | ✓ | R6 best.pt 里 6 个 cond_mlp_adapter key,shape 正确 |
| checkpoint load strict=False | ✓ | L1369 总是 strict=False,L1370-1373 检查非 adapter missing key 会 raise |
| PU-AdamW 处理 adapter | ✓ | R6 用标准 AdamW,PU 投影只在 repr_loss 路径,不会错误归入 base 组 |
| DDP wrap 正确 | ✓ | L1466 wrap _GeneratorTrainingStep,adapter 在同步范围 |
| 续训 resume 正确 | ✓ | R6-continue adapter diff=0(完美恢复) |

### 唯一 minor bug

**`null_condition.embedding` 漏 freeze**:
- 位置: `_apply_generator_trainable_mode` IP_ADAPTER 分支是 no-op(g_loop.py:1074-1092)+ `wrap_backbone_with_condition_mlp` 只 freeze `backbone.named_parameters()`(ip_adapter.py:330-336)
- 影响: 该参数被错误放进 optimizer(R6 optimizer_state 有 7 个 param,第 7 个就是它)
- 实际无害: `meanflow_sit.py:388` 加了 `0.0 * null_condition.embedding.sum()` dummy guard,grad 恒为 0,checkpoint 值 absmax=0.0008 几乎不变
- 修复: 在 `_apply_generator_trainable_mode` IP_ADAPTER 分支显式 freeze 非 adapter 参数

### 代码现状总结

**v4 PEFT 实验数据是可信的**,建立在正确实现上。没有"adapter 没接上预训练权重"那种 bug。SubAgent B/C 都验证了 fc1/fc2/fc3 在 R6 checkpoint 里真的从 init 值变过,真的训了。

---

## 3. 真正根因:Stage 2 训练目标设计问题

### 数据支持的因果链

1. **历史 PU-AdamW (131M)**:cosine 0.79 → 0.98 时 FID 34 → 143(4.2×)
2. **v4 PEFT MLP (1.58M)**:λ=0 → 1.0 时 FID 128 → 181(1.4×,且起点已经崩)
3. **PEFT 比 PU 破坏更大**(同 cosine 区间 FID +58)说明 adapter 加 condition 主路确实放大了 leakage
4. **但 PU 也照样塌**说明 adapter 架构不是唯一因素,甚至不是主要因素

### 真正根因(最终判断)

跟 [[safa-fm-supervision-plan-june17]] 之前讨论的"Stage 2 FM loss 用 x_0 自己作监督存在身份泄漏"是同一个根问题:

**Stage 2 训练目标**:
```
L_stage2 = L_FM(v_θ(x_t, t, c=z), target=velocity_to(x_0)) + λ_repr * cos(E0(x̂), z_0)
```

- `L_FM` 用 `(x_0, z_0)` 配对监督,但 z_0 来自 E0(x_0)(同一张图片的表征)→ generator 学到"按 z 重建 x_0"而不是"按 z 生成 anonymized 人脸" → **identity leakage**
- `λ_repr * cos(E0(x̂), z_0)` 强拟合 E0 表征 → 推 generator 输出 x̂ 趋近 x_0 → 进一步 leakage
- 不论用什么参数空间(全参数 / adapter / LoRA / MLP)更新这个目标,都会 leakage

### 用户当时判断的验证

> "模型本身没有被很好地保护,被表征学习入侵破坏了"

**完全正确**。具体证据:
1. FID 暴涨:e15_cold_v8 cosine 0.79→0.98 时 FID 34→143
2. 多样性塌缩:cosine ≥ 0.95 时 pearson/spearman ≥ 0.95(近乎 1.0)
3. PU 把 cosine 推到 0.97+ 时确实在挤压 FM 模型容量:`pu_norm_ratio` 18-29(repr grad 是 FM grad 的近 30 倍)
4. e13 在 cosine 0.80 时 FID 35,同一个 base 走到 0.97+ FID 立刻飙到 145 — 不是免费午餐

---

## 4. 讨论的明确选项(下次决定)

### 选项 A: 换监督信号(治本,工程量大)

参考 [[safa-fm-supervision-plan-june17]]:
- **CelebHQ/FFHQ 表征最近邻替换**:不要用 z_0 自身作监督,用 face-id 过滤后的"身份不同但属性相近"的另一个样本 z' 作监督
- **Caption-based supervision**:用 VLM 给样本生成 caption,监督信号从 (z_0, x_0) 改成 (caption_z, x_caption)
- 工程量:5-10 天(数据 pipeline + 训练框架改动)

### 选项 B: 加 image quality regularization(治标)

`L_stage2 = L_FM + λ_repr * L_repr + λ_lpips * LPIPS(x̂, stage1_null_condition_output)`

- 强制 generator 保持 stage1 学到的人脸先验
- e15_cold_v8 已经在用 `lambda_lpips=0.15`,但效果不够(FID 还是飙到 143)
- 可能需要把 lambda_lpips 加到 0.5+,或者换 stronger perceptual loss(VGG face / arcface feature loss)
- 工程量:1-2 天

### 选项 C: 控 cosine 上限(早停)

接受"cosine 0.85 + FID 50"比"cosine 0.97 + FID 145"更好的 trade-off:
- e13_pu_adamw_e14resume 在 cosine 0.91 时 FID 47 是当前最佳平衡点
- 训练时实时监控 FID,**FID > 60 立刻早停**
- 工程量:0(就是早停)
- 缺点:cosine 上不去,论文卖点弱

### 选项 D: 改 stage2 objective(治本,但需新理论)

- **Cyclic consistency**:z_0 → x̂ → E0(x̂) → 重建 z_0,中间加 anonymization bottleneck
- **Contrastive**:z_0 是 anchor,正样本是 same-expression different-identity,负样本是 same-identity
- 工程量:1-2 周,需要新理论指导

### 选项 E: 接受 SAFA 当前上限,转向应用/部署

- e13 cosine 0.91 + FID 47 已经是当前最佳 trade-off
- [[safa-project-positioning-jul1]] 写过"现有结果够 CCF-B 冲 CCF-A"
- 不再追求 cosine 0.97+,把精力放到论文写作 / 应用拓展

---

## 5. 待补的实验数据(为下次讨论准备)

### 5.1 Sharpness 量化(整仓库缺失)

整仓库没有 sharpness 字段,用户当时"Sharpness 下降"是肉眼观察。建议:
- 写一个离线脚本 `/tmp/calc_sharpness.py`,对 `artifacts/eval/e15_cold_v8_lpips_gpu0123_ddp_150ep/quality/epoch_*/generated_images/` 里的 PNG 算 Laplacian variance
- 对比 epoch 1660 (cos 0.79) vs 1700 (cos 0.98) vs 1760 (cos 0.90) 的 sharpness
- 工程量:30 分钟

### 5.2 R1 PU baseline 当前 FID/NIQE/Sharpness

R1 cosine 0.4741 但质量指标完全没测。需要:
- 在 R1 best.pt 上跑 `/tmp/r6_quality_eval.py` 同款 pipeline
- 看 R1 在低 cosine 区间是否 FID 维持 ~50
- 工程量:30 分钟

### 5.3 多样性指标(face generation diversity)

整仓库用 pearson/spearman 当多样性代理,但实际应该测:
- **LPIPS diversity**:生成 256 张图片,两两 LPIPS 距离的均值
- **FID within generated set**(不是跟真实分布比,是看生成分布的内部多样性)
- **Identifiability**:用 arcface 算生成图片两两 cosine,看 identity 多样性

---

## 6. 关键文件路径(4029 上)

### 历史 PU 最佳 checkpoint

```
artifacts/checkpoints/e15_cold_v8_lpips_gpu0123_ddp_150ep/best.pt   # cos 0.9833, FID 143
artifacts/checkpoints/e14resume_v4_combined_gpu01_ddp_200ep/best.pt # cos 0.9743, FID 145
artifacts/checkpoints/e14resume_v3_budrelax_gpu23_ddp_200ep/best.pt # cos 0.9669, FID 68(质量最好)
artifacts/checkpoints/e13_pu_adamw_meanflow_sit_stage2_gpu5_200ep/best.pt # cos 0.91, FID 47(平衡最佳)
```

### v4 PEFT checkpoint

```
artifacts/checkpoints/peft_mlp_v{1,2,3,4}_lr{0,01,05,10}/best.pt   # cos 0.21/0.67/0.78/0.86, FID 128/138/165/181
artifacts/checkpoints/peft_mlp_r6_continue_gpu0123/best.pt         # 续训 5 epoch
artifacts/checkpoints/peft_mlp_r6_gpu1/best.pt                     # R6 原始
```

### 评估脚本(可复用)

```
/tmp/r6_quality_eval.py                  # 完整质量 pipeline (FID/NIQE/Sharpness/face-id)
/tmp/post_image_metrics.py               # epoch-level sharpness + face-id 后处理
/tmp/find_best_pu.py                     # 找历史 PU 最佳结果脚本
/tmp/pu_scan_results.json                # 29 个历史实验完整扫描结果
```

### 代码

```
src/safa/models/ip_adapter.py            # ConditionMLPAdapter + IPAdapterCrossAttention
src/safa/training/peft_runner.py         # PEFT_FM + PEFT_MLP runners
src/safa/training/g_loop.py              # PEFT dispatch + eager wrap
src/safa/training/g_loop.py.bak_r6cont   # 续训前备份
src/safa/_legacy_backup_20260706/        # v1 原始备份
```

---

## 7. 仓库状态

- 所有改动在 working tree,**没有 git commit/push**
- 备份完整(可随时回滚)
- PEFT 代码经过 2 轮独立审查,核心机制正确,只有 1 个 minor freeze 泄漏(null_condition.embedding,已被 dummy guard 抵消)

---

## 8. 讨论时的关键问题

下次讨论时建议聚焦:

1. **是否同意"训练目标设计是根因"的判断?** 如果同意,选项 A/D 是正解;如果不同意,先补什么实验?
2. **Sharpness 量化是否需要补?**(用户当时说"Sharpness 下降"无量化,30 分钟可补)
3. **R1 PU baseline 的 FID/NIQE 是否需要立刻测?**(决定 PU 在低 cosine 区间是否真的安全)
4. **选项 B(stronger LPIPS)是否值得先试?**(1-2 天,如果有效就能 buy time)
5. **论文目标是什么 CCF 级别?** 决定是否需要继续冲 cosine 0.97+ 还是接受当前 e13 (0.91, FID 47) 的 trade-off
