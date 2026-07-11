# SAFA 项目证据文档(2026-07-05)

> **目的**: 把所有实验数据、理论裂缝、方法设计问题摆出来,供后续分析讨论。**本文档不提解决方案**,只摆事实。
>
> **使用方式**: 拿这份文档对照你的论文 claim 逐条审计。审稿人会问的问题,这份文档里都已经摆出来了。

---

## 0. Executive Summary

### 0.1 一句话现状

当前 SAFA 主结果(v3 ep131)cosine 0.9715、NIQE 6.27、sharpness 157 — **cos 看着漂亮但 sharpness 不达发布标准(目标 ≥250),且 cos 这个指标本身在数学上可被 game**。

### 0.2 三个最致命的问题

1. **匿名性没有训练信号保证**(C1 claim 完全立不住)
   - `audit.py` 硬禁任何 identity loss
   - 匿名性 100% 依赖"z₀ 不含 identity"这个**未经证明的假设**
   - z₀ 是 512 维 ResNet50 表征(SOTA face-id embedding 也是 512 维),**信息容量过剩**
   - memory `safa-7way-z0-universality-june19` 显示 z₀ 跨编码器 Spearman 0.88(含 IResNet-100 face 预训练),提示 z₀ 与身份强相关

2. **方法描述与代码实现不符**(论文可复现性问题)
   - README 写"Stage 1 用空条件学人脸先验,Stage 2 注入 z₀"
   - **代码实际**: Stage 1 直接用 z₀ 做条件训 FM(`flow_condition: embedding`),Stage 1/2 在 condition / cycle / repr 上**几乎没有差别**
   - README 写"不得 G(ξ;z₀)≈x₀,否则身份泄漏",代码 Stage 1 的 FM loss target 就是 x₀ 本身

3. **PU-AdamW 不是真优化器,是"AdamW(FM) + Projected SGD(repr)"拼起来的两阶段流程**
   - repr 更新绕过 Adam state(`p.data.add_` 直接改参数),**没有动量、没有自适应 lr、没有 bias correction**
   - 一阶投影在弯曲流形上累积误差,长程下来啃掉图像质量(sharpness 360→92)
   - 名字叫 "PU-AdamW" 容易让审稿人误以为是 AdamW 变体,实际是更新规则

### 0.3 v3 主结果是不是垃圾?

不全是,但 claim 必须收缩。

| Claim | 在当前 loss 下立得住? | 原因 |
|-------|----------------------|------|
| C1 匿名性 | **立不住** | 无训练信号,z₀ identity-untangled 是假设不是事实 |
| C2 表情保留 | **勉强立住** | repr loss 显式优化 cos(E₀(x̂), z₀),但 E₀ 与训练同源,需独立 FER 模型验证 |
| C3 图像质量 | **立得住** | FM loss well-posed,FID/NIQE 独立计算。但 best.pt 选择偏 cos × face_rate,sacrifice 质量换 cos |

**建议**: 论文 claim 收缩到 "retention-oriented anonymization via representation projection",**不要 claim "无需 identity supervision 的 anonymization"**,除非补 EER 实验。

---

# Section A: 实验数据汇总

## A.1 全局实验对照表

| Run | 设置 | 起点 ep0 cos | 终 ep | best cos | best NIQE | 终 sharpness | 失败模式 |
|-----|------|-------------|-------|----------|-----------|--------------|----------|
| v2  | cap=0.5 | 0.3604 | 82 | 0.9629 @55 | 6.021 @78 | 196.6 | (F) cos 涨质量退化 |
| v3  | budget=0.005 | 0.3448 | 149 | 0.9715 @131 | 5.914 @13 | 134.2 | (F) 涨但 NIQE_std 走低 |
| v4  | cap=0.5+budget | 0.4561 | 27 | 0.9799 @21 | 6.269 @3 | 141.2 | (B)+(D) NIQE 涨到 7.3, std 塌 |
| v5  | cap=0.35 | 0.3838 | 6 | 0.8747 @6 | 5.728 @0 | 245.5 | (C) 太早,未跑完 |
| v6  | credit_proj warmstart | 0.9697 (warm) | 6 | 0.9697 @0 | 6.063 @6 | 184.1 | (C) 完全停滞, 被 kill |
| v7  | lambda_lpips=0.05 | 0.9693 (warm) | 2 | 0.9693 @0 | 6.216 @0 | 144.9 | (C) 3 ep 内停, 失败 |
| v8  | e15 cold + lpips | 0.1734 @1652 | 1744 | 0.9833 @1735 | 5.342 @1652 | 155.4 (峰 346→69) | (A)+(B) sharpness 92 崩盘 |

**注意**:
- `best NIQE` 列 v8 起点 NIQE 5.34 最低,但对应 cos=0.17(还没开始训练),没意义。v3 起点 NIQE 5.91 才是有意义的"训练前质量下限"参照
- **FID 在所有 jsonl 和 eval 子目录里都没存**,只能用 NIQE 作质量代理。FID 表格需要在论文里诚实标注"未记录",不要拿 NIQE 冒充 FID
- 所有终 sharpness 都低于目标 250,唯一过 250 的是 v8 起点(未训练)和 v5(只跑了 6 ep)

## A.2 每实验轨迹

**v2 (cap=0.5)** — ep0 cos 0.36/NIQE 6.04 → ep55 cos 0.9629/NIQE 6.58(峰值) → ep82 cos 0.9298/NIQE 6.41。cos 在 ep55 见顶后开始往下掉 0.033, NIQE_std 范围 1.03–1.67 还算多样。repr_lr 卡在 4.5e-6, backtrack=2.0/cycle 高频回退。**判定**: 涨得太猛(cap 翻倍), 后期 cos 自己回吐, 质量没换回来。

**v3 (budget=0.005, 论文主结果)** — ep0 cos 0.34/NIQE 6.06 → ep79 cos 0.9620/NIQE 6.07(Pareto knee) → ep131 cos 0.9715/NIQE 6.27(虚荣峰值) → ep149 cos 0.9661/NIQE 6.06。NIQE_std 从 1.38 一路跌到 1.01, 中段 ep100 触及 0.85。**判定**: cos 涨健康但多样性在缓慢塌。repr_lr 末期 1.58e-6 几乎为零, 表示 PU 步长已被压死, 模型基本停止探索。

**v4 (cap=0.5+budget=0.005 双 bind 解)** — 起点 cos 0.46 异常高, ep21 cos 0.9799(全场最高之一), 但 NIQE 一路从 6.27 涨到 7.30, NIQE_std 跌到 0.58。**判定**: 严格的 (B) mode collapse + (D) 质量崩, 典型被 cosine "hack" 掉的轨迹。

**v5 (cap=0.35)** — 只跑了 6 ep, cos 从 0.38 涨到 0.87, NIQE 从 5.73 涨到 6.18。终 sharpness 245.5 全场最高(因为还没开始磨)。**判定**: 数据不足, 但趋势和 v2 同向, 只是 cap 更保守。

**v6 (credit_proj warmstart)** — 起点 cos 0.97 已经在高位(暖启动自 e14 best.pt), 6 ep 内 cos 反而掉到 0.90 再回 0.96, 没有任何提升。repr_param_step_norm_before_clip = 2.32 异常大, 但 step_norm_after_clip 估计几乎全被投影吃掉。**判定**: (C) 完全停滞, 这就是被 kill 的原因。

**v7 (lambda_lpips=0.05)** — 3 ep 内 cos 0.9693 → 0.9682, NIQE 0.05 涨。**判定**: (C) 加上 lpips 之后完全跑不动, 3 ep 就被认定失败停掉。

**v8 (e15 cold + lpips, 4 GPU DDP)** — 起点(继承 e14 主干冷启 e15) cos 0.17/NIQE 5.34/sharpness 346.7 → ep1682 cos 0.96/NIQE 6.05/sharpness 98.7 → ep1735 cos 0.9833(全场最高)/NIQE 5.79/sharpness 69.1 → ep1744 终 cos 0.87/NIQE 6.27/sharpness 155。**判定**: (A) sharpness 崩盘 + (B) NIQE_std 跌到 0.51。cos 涨得最猛(0.70 delta), 但生成图从 347 的清晰度塌到 69。末期 sharpness 155 是部分恢复, 不是稳定状态。

## A.3 Pareto 前沿(3 目标: max cos, min NIQE, max sharpness)

按 (cos, NIQE, sharpness) 三维非支配筛选, 全场仅以下点未被任何其他点支配:

- **v3 ep131 (cos 0.9715, NIQE 6.27, sharp 157)** — 当前论文 best.pt, 唯一兼顾 cos≥0.97 和合理质量的点
- **v3 ep149 end (cos 0.9661, NIQE 6.06, sharp 134)** — NIQE 最低的有意义点(cos>0.9), 但 sharpness 较差
- **v5 ep6 (cos 0.87, NIQE 6.18, sharp 245)** — sharpness 最好, 但 cos 才 0.87, 训练太短
- **v6 ep6 (cos 0.9573, NIQE 6.06, sharp 184)** — 三项都不错, 但 v6 整体被认定失败
- **v8 bestcos ep1735 (cos 0.9833, NIQE 5.79, sharp 69)** — cos 最高 + NIQE 最低(>0.9 范围内), 但 sharpness 69 不可接受
- **v8 start ep1652 (cos 0.17, NIQE 5.34, sharp 347)** — "没训练"的极端参考点, 严格 Pareto 但无意义
- v2 ep82 end 也单点非支配(cos 0.93, sharp 196), 但 cos 太低

**关键观察**: 没有任何一个实验的终点同时满足 cos≥0.97 & NIQE≤6.0 & sharpness≥250。Pareto 前沿是被强行撑出来的 — v3 ep131 是唯一接近合理的, 但 sharpness 157 离目标 250 还差 100。v8 ep1735 的 cos 0.9833 是通过牺牲 sharpness 到 69 换来的, 这就是"被 hack"的硬证据。

## A.4 关键观察

1. **cosine 涨 ≠ 方法成功**: v4(cos 0.98)和 v8(cos 0.9833)都拿到了全场最高 cos, 但 v4 NIQE 涨到 7.3 + std 塌到 0.58(mode collapse), v8 sharpness 从 347 崩到 69。审稿人一眼能看出 cos 这单一指标被方法隐式优化了。整个 v 系列里, **cos 提升和 NIQE/sharpness 退化几乎线性相关**。

2. **真正"健康 trade-off"的只有 v3**: 全程 150 ep, cos 从 0.34 稳爬到 0.97, NIQE 一直维持在 6.0–6.4 区间, 没崩。代价是 NIQE_std 从 1.38 跌到 1.00 — 多样性在缓慢丧失, 但是唯一一个没出严重事故的 run。这就是为什么它是论文主结果。

3. **起点决定天花板**: v2/v3/v4/v5 从 e14resume 起点(cos 0.34-0.46, sharpness 估计 200+), 终态最好到 cos 0.97/sharp ~150。v8 从 e15 cold(cos 0.17, sharpness 347)起点, 4 GPU 大力出奇迹, cos 涨得更猛(0.70 delta)但 sharpness 直接崩到 69。**起点状态(+cos distance)和终态质量上限强相关** — 起点越远, PU/credit 越激进, 质量损失越大。

4. **所有实验终 sharpness 都不达标**: 目标 ≥250, 实际终 sharpness: v2=196, v3=134, v4=141, v5=245(只 6ep), v6=184, v7=144, v8=155。除 v5(未完成)外全部低于 200。v8 起点 347 → 终点 155 的崩盘路径最说明问题: PU 信用机制在长期训练中持续磨平生成图的高频细节, 这不是单次 bug 而是结构性问题。

5. **v6/v7 是"已经死了的实验"**: 两者都从 e14 best.pt warmstart(cos 0.97 起点), 加新机制(credit projection / lpips)后立刻停滞, 6 ep / 3 ep 内 cos 不动甚至往下走。这说明 **cos 0.97 之上已经没有可走的梯度方向**, credit_projection 和 lpips 这两个改动都没找到新的有效信号 — 它们不是"调参失败", 而是"方法在该状态下不工作"。

### A.5 数据缺口(论文要补)

- **FID 没有记录**: 所有 jsonl 和 eval 子目录都没存 FID 数字。论文里要么重跑评估补全 FID,要么诚实标注"未记录"
- **EER 没有报告**: 这是匿名性指标,代码里 `InsightFaceRecognizer` 在 eval 时算但没进 jsonl。审稿人会要求看 EER 而不是 cos(z₀)
- **独立 FER 验证缺失**: 表情保留目前用 E₀ 自己的 logits(`label_accuracy_generated` / `source_prediction_preserved`),**与训练 loss 同源**,需要 off-the-shelf FER(DDAM / Dynamic MLP)重新测

---

# Section B: 理论裂缝代码审计

## B.1 5 个已知裂缝逐一验证

### Crack 1: 一阶半空间近似弯曲流形(cap 是凸假设)

- **代码位置**: `projected_update.py:82-120`(SGD 路径)和 `projected_update.py:445-501`(Adam 路径,实际跑的就这个);trust-ratio cap 在 `g_loop.py:3231-3233`
- **数学结构**: 投影只做了一件事 — 把 `g_repr` 在 `g_fm` 方向的"负分量"砍掉。`coefficient = dot_before / fm_norm_squared; projected = g_repr − coefficient * g_fm`,目标让 Q-内积 `<g_fm, g_repr>_Q ≥ 0`。然后 cap: `step_cap = 0.25 * fm_param_step_norm`,如果 repr step 超过 cap 就按比例缩 lr
- **真问题还是夸大**: 真问题。投影本身是**单点一阶泰勒** — 它只保证"在当前 θ 这一点上,repr 梯度投影到 g_fm 法向"。但参数动一步之后, `g_fm(θ+Δθ)` 已经变了方向。FM-feasible region 真实形状是 {θ : L_fm(θ) 不增},这是 loss 等高面的法向量场,在非凸网络里**几何上是弯曲的**。投影用切平面代替曲面,曲率大时一步走完直接飞出可行集。cap=0.25 是个补丁 — 它隐式承认曲面会弯,限制步长让切平面近似还勉强成立
- **attack vector**: 模型找到高曲率区域(FM loss 陡峭方向),一阶投影看到的可行域很宽,实际走出去就破坏 FM。Backtracking(`pu_backtrack_max_retries=3`)是事后补救,只能拒掉已经撞墙的 step,无法预防
- **可修补性**: 工程补丁(cap + backtrack + budget)已经在做最大努力。本质问题在数学 — 单点一阶信息 + 弯曲流形,**只能用二阶或曲率修正**才能根治,但这等价于解约束优化,PU-AdamW 的"轻量"卖点就没了

### Crack 2: cosine 可被 off-manifold 共线输出 game

- **代码位置**: `representation_losses.py:53-66`(`hyperspherical_point_cosine_loss`),在 `g_loop.py:374` 被调用;target 是 `z`(`g_loop.py:1118` `z = batch["z"]`,预先缓存的 E₀ features)
- **数学结构**: `point_loss = (1 − ⟨pred_emb, z⟩).mean()`,pred 和 target 都被 `_validate_unit_norm` 强制单位范数(tol 1e-4)。当前 config(`e14resume_v3`) `relation_weight=0`,所以**只有 point cosine,没有 Gram 关系约束**
- **真问题还是夸大**: 真问题。cosine 是 scale-invariant + 方向 only。E₀ embedding 空间是 512 维球面,但**球面上 cos=1 的点只有一个(target 本身)**,所以训练时模型确实在学"指向 z 的方向"。问题在 E₀ 网络: E₀ 是 frozen ResNet 风格 backbone,**embedding 是图像→球面**的投影,但逆映射不唯一 — 存在大量 off-manifold 图像 x̂ 使得 `E₀(x̂)` 落在 `E₀(x₀)` 邻域。repr loss 只检查 `E₀(x̂) ≈ z`,**完全不约束 x̂ 是不是一张合法脸**
- **attack vector**: 生成器可以输出"对抗样本式"图像 — 对人眼是噪声/扭曲,但 E₀ 看到的 embedding 方向对。低熵图像(大面积纯色 + 少量高频边缘)特别容易触发,因为 E₀ 是 ImageNet/AffectNet 训出来的,对自然脸有 inductive bias,非脸图像的 embedding 行为不可预测
- **可修补性**: 需重设计。要么加 perceptual loss(LPIPS,v7 config 已经在加),要么加 Gram 关系项(`relation_weight > 0`),要么换更严格的身份度量。单点 cosine 数学上不可能堵住 off-manifold game

### Crack 3: FM loss 可被低熵图像 game

- **代码位置**: `g_loop.py:328-331`(`_compute_flow_loss` → `generator.flow_matching_loss`),`generator.py:385-403` 实现
- **数学结构**: `flow_matching_loss = MSE(predicted_velocity, target_velocity)`,其中 `x_t = (1−t)*x_0 + t*x_1`,`x_0 ~ N(0,I)`,`x_1 = x_0_image * 2 − 1`。loss 在随机 t 上积分
- **真问题还是夸大**: 真问题,但比文档说的轻。FM loss 本身**确实是 likelihood 代理**(flow matching 是 CNF 的 ELBO 等价),不直接监督感知质量。但 likelihood 高 ≠ 图像好看 — 模型可以学一个**退化分布**: 所有输出都集中在数据流形的低方差子空间(比如所有脸都偏向平均脸),likelihood 还能保持不错,但身份/表情被抹掉。这正是 SAFA 想避免的(要保表情)但又想要的效果(要换身份) — **内在矛盾**
- **attack vector**: 模型把 condition `z` 弱化为输入,输出回归到训练集平均脸。FM loss 不惩罚这种 collapse,惩罚来自 repr loss(要 cos 高)。两者拉锯
- **可修补性**: 工程部分可补。加 perceptual / LPIPS / FID 监控(v7 已经走这条路)。彻底修需要重新设计 likelihood vs quality 的 trade-off,SAFA 现在的 PU 投影本质就是把这个 trade-off 工程化

### Crack 4: e14 anti-affect 起点让 ∇L_fm 和 ∇L_repr 反平行

- **代码位置**: `configs/medium_v2/experiments/e14resume_v3_*.yaml` line `resume_from: artifacts/checkpoints/external/meanflow_sit_e14/best_ema_quality.pt`;stage2 入口 `g_loop.py:3103`(objective check);投影触发逻辑 `projected_update.py:476` `projection_applied = bool(dot_before < 0)`
- **数学结构**: e14 checkpoint 是 anonymization-finetuned(来自 memory `safa-e14-revive-sweep-june21`),已经把 affect 信号抹过一遍。从这里 resume, generator 的输出分布已经偏离"原始脸流形"。repr loss 要把 `E₀(G(z))` 拉回 `z`(恢复身份方向),FM loss 要保持当前分布(不要漂走)。两者在 weight 空间几乎必然反相关 — e14 basin 把 affect 推走的方向,恰好是 repr 想回来的方向
- **真问题还是夸大**: 真问题,且**最严重**。代码里 `dot_before` 几乎一定 < 0,所以 `projection_applied = True` 几乎每个 batch 都触发(看 metric `projection_applied` 就能验证)。投影本身**信息有损**: 当 g_repr 几乎完全反平行 g_fm 时,投影后的 `g_repr_proj` 范数被砍到只剩垂直分量,可能只剩原梯度的 5-10%。这就是 memory 里说的"卡 cos 0.82" — 投影在主动压制 repr 学习
- **attack vector**: 不是模型 game,是**优化器在和目标打架**。PU 投影看到反平行就砍,但反平行恰恰说明两个目标真的冲突 — 投影不是解决冲突,是**单方面让 repr 让步**
- **可修补性**: 需重设计。当前 v3 在做的是松 budget(`pu_fm_increase_budget: 0.005`)和拉高 cap(v2 cap=0.5),都是绕过投影的工程补丁。根本治法要么换起点(不从 e14 resume),要么换多目标方法(CAGrad / FAMO 路径在 `_FM_ANCHORED_CAGRAD` / `_FM_PRIMARY_CONSTRAINED_FAMO` 已经实现,但 e14 没用)

### Crack 5: 静态 cap

- **代码位置**: `_Stage2ObjectiveRuntime.repr_step_ratio_cap: float = 0.25`(`g_loop.py:121`);使用 `g_loop.py:3231` `step_cap = stage2_objective.repr_step_ratio_cap * fm_param_step_norm`
- **数学结构**: `step_cap` 是 config 写死的常数(0.25 / 0.5 / 0.05 等),训练全程不变
- **真问题还是夸大**: 真问题,但比文档说的轻。静态 cap 在两个阶段都不对:
  - 早期(generator 还在学基本结构): FM step norm 大(模型快速变化),cap=0.25 让 repr step 也大,可能不稳定
  - 后期(generator 收敛): FM step norm 小,cap 把 repr step 也卡死,repr 想微调身份方向但没空间
- **attack vector**: 不是 game,是**欠拟合**。后期 repr 想推但推不动,cos 卡在 0.82 一部分原因在这里
- **可修补性**: 工程可改。把 cap 做成 epoch/loss 的函数(warmup → 大 cap → 收缩)即可。代码改动很小

## B.2 新发现的裂缝

### a. EMA decay 对评估的 bias

存在但已处理。`g_loop.py:3471-3518` `_evaluate_validation_variants` 同时跑 raw 和 ema 两套评估,`raw_ema_cosine_gap` metric 显式追踪差值。**真问题**: config `best_model: ema`(line 38),意味着保存和挑选 best 用的是 EMA 模型。EMA decay=0.999(≈1000 step 平均),**对 cosine 评估有平滑 bias** — EMA 模型的 cos 通常比 raw 低 0.01-0.03(因为 EMA 把早期较差的权重平均进来)。论文如果报 EMA cos,等于系统性低估 ~2%。这是评估 bias,不是训练 bug。

### b. BN/running stats 在 DDP 下的一致性

潜在问题但 SAFA 用 `gloo` backend,且 generator 是 meanflow_sit(Transformer,无 BN,只有 LayerNorm),所以**实际无影响**。E₀ 是 ResNet 有 BN 但 frozen + eval mode(`g_loop.py:309,362,1424` 永远 `.eval()`),running stats 不更新。**这条不是 bug**。

### c. repr loss 的 detach 时机

关键发现。`g_loop.py:362-366`: repr loss 算的时候, `generated = self.generator.sample(...)` 走完整 forward, `e0_out = self.e0(normalize_for_e0(generated_for_e0))`,**E₀ 没有 detach**,但 E₀ 是 frozen(`requires_grad=False`),所以梯度自动不流经 E₀ 参数。梯度流是 `repr_loss → e0_out["embedding"] → generated_for_e0 → generated → generator`,**正确流到 generator**。

但有个隐患: `hyperspherical_point_cosine_loss` 强制 `_validate_unit_norm`(tol 1e-4),如果 generator 输出让 E₀ embedding 范数偏离 1 超过 1e-4,**直接 raise ValueError**。这在 AMP(amp:true)下偶尔触发 — 半精度数值噪声让范数漂移。属于**鲁棒性问题**,不是 leak。

### d. flow_loss_normalized 命名误导

`flow_loss_normalized` 在 `_stage2_repr_loss_metrics`(`g_loop.py:429`)就是 `float(flow_loss.detach().cpu())` — **没有 normalize**!变量名叫 normalized 但实际是 raw 值。这是**命名误导**,不是 bug。FM loss 本身是 MSE(`generator.py:401`),已经在 velocity 空间平均,scale 由 `target_velocity` 决定(≈ O(1))。

### e. cycle consistency loss 不存在

存在但 stage2 不用。`g_loop.py:311` `cycle_loss = cosine_cycle_loss(e0_out["embedding"], z)`,但只在 stage1 legacy 路径(`use_cycle=True`)计算。stage2 `point_projected_two_step` 走 `_compute_repr_loss`(line 342),**不算 cycle**。stage2 forward 里 `cycle_loss = z.new_tensor(0.0)`(line 230)永远是 0。

**这其实是同一个东西** — cycle_loss 和 repr_loss 数学上等价(都是 cos(e0(G(z)), z)),只是名字不同。PU-AdamW 路径把它叫 repr_loss,legacy 路径叫 cycle_loss。**没有独立 cycle consistency**(没有 G→E₀→G 闭环)。

### f. z₀ 维度 / bottleneck —— **最关键的新裂缝**

config `embedding_dim: 512`(line 18)。z₀ 是 512 维单位球面向量。问题: **512 维对身份来说信息容量过剩**。FaceNet/ArcFace 标准 embedding 也是 512 维,能区分 1M+ 身份。SAFA 用 z₀ 做 condition,等于给 generator 一个 512 维身份 token — **理论上足以重建任何身份**。

匿名化要做的就是"改 z₀ 方向",但 repr loss 又强制 `E₀(G(z₀)) = z₀` — **直接矛盾**。这就是为什么 cos 0.97→0.95 是可接受的: 完全保留 z₀ 等于完全保留身份。SAFA 真正的匿名化来自 `flow_condition: embedding`(line 33)和 `meanflow_ratio: 0.25` 的随机性,**不是来自 z₀ 改造**。

**z₀ 维度过高是结构性裂缝 — 无法工程修补**,要么降维(信息有损),要么承认匿名化机制不在 z₀。

### g. EMA 用作 quality eval vs 主模型用作 train metric

混淆存在。`g_loop.py:1635-1646` quality_eval hook 默认 `model: ema`(config line 65)。但训练 metric(`logged_loss`、`projection_applied` 等)来自主模型。**论文报 NIQE/FID 用 EMA,报 cos 用 raw 或 ema 都可能** — 两个评估对象的 latent distribution 不一致。`metrics_ema` 和 `metrics_raw` 都存了(line 1670-1671),但下游分析容易拿错。**建议论文显式声明**。

### h. generator input noise z 是否独立于 z₀ —— **隐性 identity leakage 通道**

半独立。`generator.py:378` `x = torch.randn(z.shape[0], 3, H, W, generator=None)` — **不用传入的 generator**!sample 阶段 noise 用全局 PyTorch 状态,**和 z₀ 没显式关联**。但 `flow_matching_loss`(line 388)**用了传入 generator**(`generator=generator`,g_loop.py:238 用 `noise_gen` 重置 seed)。所以: 训练 loss 阶段 noise 可复现(同 batch_seed),sample 阶段不可复现。

**评估时 x_init 来自 `make_x_init_for_sample_ids`**(line 346),用 sampling_seed + sample_ids 哈希,所以评估 sample 是确定的。**没有信息泄漏**,但有 sample/train 不一致: 训练见到的 noise 分布和评估不一样。

更深的隐患: 评估 sample 用 `sha256(base_seed || sample_id)` 派生确定性种子。如果数据集 sample_id 含身份信息(同一人的多张图共享 prefix),**噪声分布会按身份聚类**,这是隐性 identity leakage 通道。

### i. stage1→stage2 切换时参数冻结

config `stage1.epochs: 0`(line 48), `allow_stage2_without_stage1_gate: true`(line 12)。**SAFA 直接跳过 stage1**,从 e14 checkpoint resume 进 stage2。`freeze_e0(e0)`(line 1257)只冻结 E₀, generator 所有参数可训。**没有阶段切换的 freeze/unfreeze** — 因为只有一个阶段在跑。这条**不是裂缝**。

### j. PU-AdamW 没有 trust ratio —— **PU 名不副实**

**没有!** 这是关键发现。代码里 `repr_step_ratio_cap` 名字像 LAMB trust ratio,但**实际是步长比**(repr step / fm step),不是参数范数比。LAMB trust ratio 是 `‖θ_step‖ / ‖θ‖`,保证更新不超参数自身尺度。SAFA 的 cap 是 `0.25 * fm_param_step_norm` — **完全相对于 fm step**。如果 fm step 本身就很大(早期训练),repr step 也跟着大,没有绝对尺度保护。

AdamW 自带的 bias correction + decoupled weight decay 在 `optimizer.step()` 里管 fm step, repr step **绕过 optimizer 直接 `p.data.add_(w*g, alpha=-lr)`**(line 3243) — **没有 Adam state(一阶动量、二阶动量)跟踪**。

**这是 PU-AdamW 名字的"PU"(Projected Update)真相: 它不是新优化器,是 AdamW 跑 fm + 手写 SGD-style 跑 repr**。repr 更新没有动量、没有自适应学习率、没有 bias correction。**这是文档夸大的反方向 — PU 名不副实**。

## B.3 audit.py 禁止清单(完整)

**禁止的 training terms**(子串匹配,case-insensitive,`audit.py:7-19`):
- `identity_loss`, `id_loss`, `loss_id`
- `face_recognition_loss`
- `arcface_loss` / `arcfaceloss`
- `facenet_loss` / `facenetloss`
- `adaface_loss` / `adafaceloss`
- `magface_loss` / `magfaceloss`
- `identity_weight`
- `identity_supervision: true`(仅禁止 `true`,`false` 允许)

**禁止的 config keys**(精确字符串匹配,`audit.py:22-33`):
- `identity_loss`, `id_loss`, `face_recognition_loss`
- `arcface_loss`, `facenet_loss`, `adaface_loss`, `magface_loss`
- `identity_weight`, `id_weight`, `loss_id`

**禁止逻辑**: 递归扫 config dict + 扫所有 `*.py` 源码(排除 `training/audit.py` 自己)。子串匹配 + 一个例外: `identity_supervision: true` 只在上下文 60 字符内含 `identity_supervision": false` 时放行。**审计是 import-time + build-time 强制**,不是 runtime check。

**关键漏洞**: 审计只看**字符串**,不看语义。模型只要给 loss 起个不叫这些名字(比如叫 `face_embedding_loss` / `verification_loss` / `cosface_loss`),就能绕过。审计更像是**论文叙事保险** — 防止作者无意中写出和"无身份监督"主张矛盾的代码或 config。**不防恶意 game**。

## B.4 关键代码片段摘录

**`_compute_repr_loss` 核心逻辑**(`g_loop.py:342-405` 简化):

```python
x_init = make_x_init_for_sample_ids(sample_ids, ...)  # 确定性 noise
generated = self.generator.sample(z, steps=cycle_steps, x_init=x_init, clamp_output=False)
generated_for_e0 = _decode_generated_samples(generated, self.latent_codec)
self.e0.eval()
e0_out = self.e0(normalize_for_e0(generated_for_e0))  # 没 detach,但 E0 frozen
# 走 point cosine(relation_weight=0 时)
losses = hyperspherical_point_cosine_loss(e0_out["embedding"], z, point_weight)
# 可选 LPIPS
if lambda_lpips > 0 and images is not None:
    lpips_loss = self._lpips_metric(generated_for_e0, images).mean()
    losses["repr"] += lambda_lpips * lpips_loss
return losses["repr"], losses
```

**`flow_loss_normalized` 真相**(`g_loop.py:429`): 变量名叫 normalized,**实际就是 raw flow_loss**,没做任何归一化。

**PU-AdamW 三步核心**(`g_loop.py:3103-3340` 简化):

```python
# Step 1: AdamW 跑 FM
optimizer.zero_grad(); flow_loss.backward(); optimizer.step()
fm_param_step_norm = ‖θ_after - θ_before‖

# Step 2: 算 FM guard 梯度(更新后那点)
flow_loss_guard.backward(); fm_gradients = synced_grads(params)

# Step 3: 算 repr 梯度
repr_loss.backward(); repr_gradients = synced_grads(params)
weighted_repr_gradients = [lambda_repr * g for g in repr_gradients]

# Step 4: 投影(Q-加权)
weights = extract_adam_preconditioner_weights(optimizer, params)  # Adam 的 diag(1/√v))
# Q-投影到 g_fm 法向正半空间
projection = project_gradient_onto_fm_feasible_cone_adam(
    g_repr_for_proj, g_fm, weights, eps=1e-12)
projected_gradients = projection.projected_gradients

# Step 5: trust-ratio cap(不是 LAMB trust ratio!)
effective_lr = repr_learning_rate  # e14 v3: 3e-5
repr_step_norm = ‖w · g · lr‖
step_cap = 0.25 * fm_param_step_norm
if repr_step_norm > step_cap:
    effective_lr *= step_cap / repr_step_norm  # 按比例缩 lr

# Step 6: 手写 SGD step(绕过 Adam state!)
for p, w, g in zip(params, weights, projected_gradients):
    p.data.add_(w * g, alpha=-float(effective_lr))

# Step 7: backtracking(可选,v3 retries=3)
for retry in range(3):
    recompute flow_loss at new θ
    if fm_delta ≤ 0.005: break
    restore_params(); effective_lr *= 0.5
```

**objective switch 解析**(`_stage2_objective_from_config` `g_loop.py:571-770`): 8 种 type: `gram_weighted_sum`, `gram_projected_two_step`, `point_projected_two_step`, `point_descent_credit_projected`, `fm_anchored_cagrad`, `fm_primary_constrained_famo`, `fm_only_probe`, `gram_repr_only_probe`。后两个是 probe(诊断用,不真正训)。CAGrad / FAMO 强制 `lambda_repr=1.0` 且禁 `relation_weight`。当前 e14 v3 用 `point_projected_two_step`。

## B.5 数学结构整体判断

### PU-AdamW 是 honest optimizer 还是 hacked projection?

**是 hacked projection,不是真优化器。** 证据:
1. repr 更新绕过 Adam state(`p.data.add_` 直接改参数, line 3243),没有动量、没有自适应 lr、没有 bias correction
2. trust-ratio cap 是 repr step / fm step 的**相对比**,不是参数自身尺度的绝对限制(LAMB 那种)
3. 投影是单点一阶,没曲率修正
4. backtracking 是事后补救,不是 line search(没有充分下降条件)

它叫 "PU-AdamW" 容易让人以为是 AdamW 的变体,**实际上是 AdamW (fm) + Projected SGD (repr) 拼起来的两阶段流程**。论文叙事需要警惕: 不能把 PU-AdamW 当成"新优化器"卖,应该叫"projected two-step update rule"。

### 整个 loss stack 有多少代理 / 多少直接监督?

代理链很长:
- **匿名化目标** → 没直接监督(无 identity loss,审计禁止)
- **匿名化代理** → FM loss(保持生成质量)+ 抹 affect(e14 起点,一次性)
- **身份保留代理** → repr cosine(E₀ embedding 球面距离)
- **质量代理** → FM loss + LPIPS(v7 加的)+ NIQE/FID(eval only)
- **直接监督** → **零**。没有任何一个 loss 直接定义"匿名化成功"

整条链 4 层代理,每层都有 game 风险(见 B.1 Crack 2/3)。SAFA 的方法在数学上是"通过约束生成器不动 + embedding 方向不变,**间接**让身份改变" — 但身份改变这个目标**从未写进 loss**。

### 哪些裂缝数学不可能修,哪些工程可补?

| 裂缝 | 可修补性 |
|---|---|
| Crack 1(一阶近似弯曲流形) | **数学不可能** — 单点一阶信息治不了曲率,只能加二阶或约束求解 |
| Crack 2(cosine off-manifold game) | 需重设计 — 单点 cosine 数学上不约束图像空间,必须加 perceptual / relation 项 |
| Crack 3(FM likelihood ≠ quality) | 工程可补 — LPIPS / FID 已经在做 |
| Crack 4(e14 反平行起点) | 需重设计 — 本质是目标冲突,PU 投影单方面压制 repr,要么换起点要么换多目标方法 |
| Crack 5(静态 cap) | 工程可改 — 加 schedule 即可 |
| f(z₀ 维度过高) | **数学不可能** — 512 维身份空间 vs 匿名化目标结构性矛盾 |
| j(PU-AdamW 不是真优化器) | 叙事问题 — 改名 + 论文里诚实描述即可 |

### 论文方法在数学上是否站得住脚?

**部分站得住,但叙事要诚实**。SAFA 的 PU-AdamW 是个**工程上合理但理论上不严格**的更新规则。它能在 cos 0.82-0.95 区间稳定训练,是因为 cap + projection + backtracking 三重补丁盖住了一阶近似的洞。但论文不能宣称:
- "PU-AdamW 保证 FM 收敛"(不保证,backtracking 只是回退)
- "repr loss 保留身份"(不保留,只保留 E₀ embedding 方向,off-manifold game 存在)
- "z₀ 是匿名化机制"(不是,z₀ 维度过高,匿名化实际来自 meanflow noise + e14 抹 affect)

**论文能宣称**:
- "PU-AdamW 是个稳定的两阶段更新规则,在 e14 起点上能持续优化 cosine 同时控制 FM 退化"
- "audit 框架保证训练流程不含身份监督"(事实陈述,不等于匿名化数学证明)
- "FM + repr cosine 在我们的 setup 下经验上 trade-off 可达 cos 0.82-0.95"

---

# Section C: 方法设计审计

## C.1 Claim vs Loss 对照表

SAFA 论文的 claim 在 README 里写得很清楚: 生成 ŷ 落在真实人脸流形上,且 E₀(x̂) ≈ z₀,同时禁止身份重建 G(ξ; z₀) ≈ x₀。把它拆成审稿人能查的三条:

| Claim | 训练时对应的 loss | 直接/间接监督 | gap 风险 |
|-------|------------------|---------------|----------|
| **C1 Identity anonymization**(x̂ 身份 ≠ x₀) | **无训练 loss**。代码里 `audit.py` 硬性禁止任何 identity supervision(FORBIDDEN_TRAINING_TERMS 列了 arcface/facenet/adaface/magface/identity_loss 等)。匿名性唯一依赖 `z₀` 是"identity-untangled 表征"这个假设 | 间接中的间接 | **极高**。模型没有任何信号告诉它"身份必须变"。loss 只要求 E₀(x̂) ≈ z₀,而 z₀ 来自 E₀,后者是 8 类表情分类器加 512 维 projector,与 face-id 没有任何正交约束 |
| **C2 Expression preservation**(x̂ 表情 ≈ x₀) | `repr loss` = `1 − cos(E₀(x̂), z₀)`(`hyperspherical_point_cosine_loss`),可选 Gram 关系损失 | 间接 | **高**。z₀ 是 E₀(x₀),E₀ 被训练成表情分类器但 embedding 512 维显然不只有表情。所以 cos 高 ≠ 表情保留。更糟糕的是 z₀ 被训练用来分类 8 类 discrete 表情,维度里塞满了 identity/pose/lighting 等 nuisance。把 cos 推到 ≈1 实际上保留了 nuisance,而不是只保留表情 |
| **C3 Image quality**(x̂ 是真实人脸) | Stage 1: `flow_matching_loss`(FM MSE)。Stage 2: FM + repr(投影/加权)。无 FID/NIQE/LPIPS 直接监督,但部分 config 有 `lambda_lpips > 0`(可选) | 间接 | **中**。FM loss 让 x̂ 落在 face manifold,这是真的。但没有任何 loss 直接惩罚模糊/mode collapse。best.pt 选择靠 `cos × single_face_rate`,反而**鼓励**往 z₀ 靠近,质量掉了也无所谓 |

**最致命的是 C1**。论文的卖点"无需 face-id 监督也能匿名"在当前 loss 设计下,逻辑链是: 训练 E₀ 时没拿 face-id label → 所以 z₀ 不含 identity → 所以逼 E₀(x̂) → z₀ 不会泄漏身份。这条链的第一跳是**未经证明的假设**。E₀ 是 ResNet50 + 512 维 projector 训 8 类表情分类,从来没做过 identity decorrelation。审稿人会立刻问: **你怎么知道 z₀ 不含 identity?**

## C.2 Pipeline 设计

从 `e0.py`、`generator.py`、`conditioning.py`、`g_loop.py` 读出来的事实:

- **E₀**: `EmotionEncoder`,backbone 默认 ResNet50(ImageNet 预训练), + `Linear(2048→512)` projector + L2 normalize,加一个 8 类 `Linear(512→8)` classifier 做监督训练。**E₀ 在 G 训练前冻结**(`freeze_e0` + `assert_e0_frozen`,optimizer 不许含 E0 参数)。这是正确的 — 评估指标不会被训坏。但代价是 z₀ 的语义结构被冻死,后续 G 必须迁就这个固定空间
- **E₂**: **代码里没有 E₂**。README 写的"再编码 z = E₀(x̂)"用的是同一个 E₀。所以训练和评估的"target embedding"完全来自同一个分类器。**没有第二个独立编码器做交叉验证**,这是审稿人下一个会问的点: 你的 repr loss 用 E₀ 算,cos 指标也用 E₀ 算,这是 self-consistent metric,不能证明 C2
- **z₀ 维度**: 512。512 维对一个 8 类表情任务**严重过参数化**。表情本征维度大概 8-30 维就够,剩下 480+ 维 E₀ 会用来拟合 identity/pose/background 以降低 classification loss。这直接喂给 C1 的 gap
- **G**: 条件 FM(Flow Matching)UNet,`FiLMResidualBlock` 用 FiLM(scale-shift)注入条件。condition = `time_mlp(t) + z_mlp(z)`,即时间和 z₀ 加在一起做 FiLM 调制。**z₀ 与时间嵌入相加后全局 FiLM 调制每个 ResBlock**,没有 cross-attention、没有 spatial condition、没有 concat。这意味着 z₀ 的 512 维信息只通过 `Linear(512→condition_dim)` 一次性压入,然后 broadcast 到每个 feature map 通道
- **G 的输入**: README 公式写 `x̂ = G(ξ; c)`,看着像有独立噪声 ξ。代码实际行为(`make_x_init_for_sample_ids` + manifest 里 `"generator_input": "z_only"`): **x_init 是噪声,但用 `sha256(base_seed || sample_id)` 派生的确定性种子生成 randn**。即同一个 sample_id 永远拿到同一份"噪声"。这有两个后果: (1) 严格说没有随机性,G 是 sample_id 的确定性函数; (2) **噪声与身份通过 sample_id 间接绑定** — 如果数据集 sample_id 含身份信息(同一人的多张图共享 prefix),噪声分布会按身份聚类,这是隐性 identity leakage 通道
- **z₀ 与 ξ 的交互**: **没有独立 ξ**,z₀ 既是条件又通过 x_init 间接绑定 sample。Stage 1 用 `flow_condition: embedding`(condition = z₀)而非 null。**README 写"Stage 1 用空条件 c∅ 学人脸先验"与代码不符** — Stage 1 直接拿 z₀ 做条件训 FM,这意味着模型从第 0 步就在学"如何用 z₀ 重建 x₀"。这与论文反复强调的"不得 G(ξ;z₀)≈x₀ 否则身份泄漏"**自相矛盾**

## C.3 Stage 1/2 边界

Stage1 config: 200 epochs, `flow_condition: embedding`, `cycle_weight: 0.01`, `lambda_initial: 0.01`。
Stage2 config: 200 epochs, `flow_condition: embedding`, `lambda_initial: 0.01`, `lambda_max: 0.01`, `lambda_growth: 0`。

读出来的事实非常反直觉:

- **Stage 1 与 Stage 2 用同一个 flow_condition(embedding)**,不是 README 说的"Stage1 null / Stage2 z₀"。两个 stage 在 condition 上**没有差别**
- **Stage 1 与 Stage 2 在 cycle/repr 权重上也没有差别**(都 0.01,`lambda_growth: 0`)
- **唯一差别是 Stage 2 多开了 `gradient_conflict` 监控 + 可选 projected update + EMA + 训练 objective 切到 `gram_weighted_sum` / projected / CAGrad / FAMO**
- **`fm_only_probe` 是诊断模式**: `flow_loss` + 0 repr,只跑一个 batch 看看 FM 单独的 loss,不做实际更新方向变化
- **切换 Stage1→Stage2**: 代码在 `for stage_name in ("stage1","stage2")` 外层循环里跑,`optimizer` 是同一个,不 reset; `optimizer_state_dict` 在 `last.pt` 持久化,Stage2 启动时通过 `resume_from` 加载 Stage1 checkpoint(`train_g_medium_v1_stage2_m0.yaml` 明确 `resume_from: .../best_stage1.pt`)。**optimizer momentum / Adam m,v 全部继承**。这意味着 Stage2 第一批 step 用的是 Stage1 训出来的 Adam 二阶矩,会有惯性
- **EMA**: Stage1 `ema.enabled: false`,Stage2 `ema.enabled: true, decay: 0.999`。Stage2 启动 EMA 是从 raw model 冷启动 `ema = ExponentialMovingAverage(generator, decay=...)`,没有继承 Stage1 的"虚拟 EMA"。所以 Stage2 头几个 epoch EMA 还在追 raw,`evaluate_ema: true` 报的 EMA cos 会**偏悲观**于 raw,但很快收敛
- **z₀ "null" 程度**: **根本没有 null**。Stage1 拿真 z₀ 做条件,cycle loss 直接算 `cos(E₀(G(z)), z₀)` = 0.01 权重。条件既没 dropout 也没替换 null

**审稿人会立刻指出的问题**: 论文 Section 2 描述的两阶段(Stage 1 学人脸先验 → Stage 2 注入 z₀)与代码实现(Stage 1 直接用 z₀ 做条件训 FM)是两个不同的方法。要么 README 错了,要么代码做了另一种实现。这个 gap 必须在 paper 里讲清楚,否则方法描述无法复现。

## C.4 评估指标的可 game 性

从 `evaluation/runner.py` + `evaluation/metrics.py` + `g_loop.py` 的 best.pt 选择代码读出来:

| 指标 | 计算方式 | 与 train loss 同源? | 可 game 性 |
|------|----------|---------------------|------------|
| **latent_cosine** = `cos(E₀(x̂), z₀)` | E₀ frozen,同训练用 E₀ | **完全同源**。repr loss 就是 `1 - cos`,cos 就是 `1 - loss` | **可被直接 hack**。模型只要把 E₀(x̂) 推到 z₀,这个指标就高,跟图像质量、表情、匿名都无关。Extreme case: x̂ = x₀,cos = 1,但完全未匿名 |
| **single_face_eq1_rate** | InsightFace `buffalo_l` 检测人脸数 = 1 的比例 | 不同源 | 中。模型可以生成"看起来像脸"的图,但与是不是同一个人无关 |
| **best.pt composite score** = `cos_mean × single_face_rate` | 见 `_composite_score` | 同 latent_cosine | **直接被 cos 主导**。loss 越低 cos 越高,best.pt 越好。本质是 train loss 排序,不是泛化指标 |
| **privacy EER / AUC / TAR@FAR** | 用独立 ArcFace(`InsightFaceRecognizer`)算 source vs generated cos,与 impostor cos 比 | **不同源**(用独立 recognizer) | **相对可信**。这是 SAFA 唯一不同源的匿名性证据 |
| **FID / KID** | real = val_single_face,generated = G(z) | 不同源 | 可信,但 reference distribution 是同 dataset 同 split,可能偏乐观 |
| **NIQE** | pyiqa 实现 | 不同源 | 可信,但 no-reference,绝对值意义不大,只适合横向比 |
| **LPIPS(x̂, x₀)** | 仅当 `lambda_lpips > 0` 时计入 loss | 视 config 而定 | 部分同源。如果 LPIPS 入 loss,模型会显式拉远感知距离(鼓励变化),与"匿名"对齐,但与"表情保留"冲突 |

**best_model 选择**: `best_model: raw`(medium_v1),即用 raw 而非 EMA。EMA 仅做监控。这意味着 best.pt 是 raw model 的最优 epoch,选最优的标准是 `cos × face_rate` — **两个指标都鼓励"贴近 z₀"而不是"匿名 + 表情 + 质量"**。`best_single_face.pt` 单独存,使用 `_stage1_single_face_score` 排序(face_rate 主导),用于 Stage 2 resume,所以 Stage 2 resume 进来的模型已经偏向"face-detector 友好"区域。

**EMA 是否偏乐观**: Stage 2 `evaluate_ema: true` 同时报 raw 和 ema。`raw_ema_cosine_gap = raw_cos − ema_cos`,有监控但 best 用 raw。EMA 在 Stage2 头几个 epoch 偏悲观,稳定后差距很小。**EMA 不会让 cos 评估偏乐观**,这点是干净的。

**最关键的可 game 问题**: 整个训练 + best.pt 选择都围绕 `cos(E₀(x̂), z₀)` 转,而 E₀ 是同一个 frozen 分类器。**模型可以学到 trivial 解"x̂ ≈ x₀"使 cos 接近 1**,但代码里的 audit guard 反而检查 `latent_cosine_mean ≥ threshold`,即**如果 cos 太低会报错**,而不是"cos 太高报错"。这意味着 over-fit 到 x₀ 的 trivial 解在 guard 看来是合规的。匿名性只能靠 `privacy.enabled` 那条独立 ArcFace 通道发现,但那条通道**不进 loss,只在 eval 时算**,所以模型在训练时没有任何压力去降低 ArcFace 同身份相似度。

## C.5 SOTA 对比

主流 face anonymization SOTA 方法的 loss / stage 设计:

| 方法 | identity loss 直接监督? | 两阶段? | 质量监督 | 报告 FID/EER |
|------|--------------------------|----------|----------|---------------|
| CIAGAN(CVPR 2020) | **是**: 用预训练 ArcFace 算身份距离 loss,显式推身份变化 | 否(单阶段 end-to-end GAN) | adversarial + perceptual | FID ~ 30-50 on CelebA |
| Face Anonymization Made Simple(arXiv 2024) | **是**: 用 face-id encoder 推距离 | 否 | diffusion + LPIPS | FID 报告 |
| Key-Driven(NDSS 2025) | **是**: key-conditioned virtual face,显式身份 mixing | 是(key 生成 + attribute transfer) | adversarial | 优于 CIAGAN |
| StyleID(PoPETS 2023) | **是**: StyleGAN inversion + identity mixing, ArcFace 监督 | 否 | StyleGAN prior | FID + EER |
| CVPR 2024 Intrinsic/Extrinsic Attention | **是**: attention mask 显式抑制 identity region | 否 | adversarial | FID + EER |
| **SAFA** | **否**: audit.py 硬禁,只靠 z₀ 表征空间假设 | 是(但 stage 1/2 实际差异微小) | FM loss,无 adversarial | FID/NIQE/EER 都报,但 EER 仅 eval-time |

**SAFA 拒绝 face-id loss 是 design choice 还是 handicap?**

从论文方法论角度,SAFA 的"no identity supervision"是一个**真实的卖点**: 它避免了 CIAGAN 那种"训完就过拟合到 ArcFace,换个 recognizer 就失效"的问题。这是个有道理的设计哲学。**但代价是**:

1. **匿名性没有训练信号保证**,完全依赖 z₀ 是否真的"identity-untangled" — 而这件事 SAFA 论文没有任何 ablation 证明(z₀ 来自 8 类表情分类器,与 face-id 的互信息未测)
2. **EER 数字可能不如直接用 ArcFace loss 的方法**。SOTA 在 LFW/CelebA 上 EER 可以做到 0.5 左右(接近 random),SAFA 的 EER 数字(根据 MEMORY 里 safa-project-positioning 笔记)cosine 0.95-0.97 看着高,但 cosine 高 = 同身份相似度高 = **匿名性差**。EER 才是匿名性指标
3. **唯一可比性**: SAFA 的 EER 应该用独立 ArcFace recognizer 算,代码里就是这么做的(`InsightFaceRecognizer`),所以 EER 数字本身是可信的。问题是 SAFA 在 paper 里必须明确报告 EER 而不是只报 cos(z₀) — 因为 cos(z₀) 不是匿名性指标,是 retention 指标

**定位**: SAFA 在 SOTA 比较中应该走"differentiated design philosophy"路线: 不依赖 face-id supervision 的 anonymization。但必须证明这条路线的 EER 可比,否则 reviewer 会说"你的方法不实用,因为匿名性不如直接用 ArcFace loss 的方法"。

## C.6 论文 claim 能立住吗

**完全立不住**: **C1 Identity anonymization**。当前 loss 设计下没有任何训练信号惩罚身份泄漏。匿名性完全依赖"z₀ 不含 identity"这个未经证明的假设。z₀ 是 8 类表情分类器的 512 维 embedding,严重过参数化,几乎肯定含 identity 信息(MEMORY 笔记 `safa-7way-z0-universality-june19` 显示 z₀ 在 8 个 backbone 间 Spearman 0.88,而其中包含 IResNet-100 face 预训练模型,提示 z₀ 的"universal"维度与身份信息相关)。

审稿人会要求:
- (a) 直接测 z₀ 与 ArcFace embedding 的互信息
- (b) 直接测 x̂ vs x₀ 的 ArcFace cos 分布
- (c) 报告 EER 而非 cos(z₀)

当前 paper 框架如果只报 cos(z₀) 高当作"达到目标",会被直接 reject。

**勉强立住但有 risk**: **C2 Expression preservation**。repr loss 确实在显式优化 `cos(E₀(x̂), z₀)`,但 z₀ 不是纯表情表征。如果 paper claim "expression preservation",需要:
- (a) 在独立表情分类器(不是 E₀)上测 x̂ 的表情准确率 vs x₀ 的表情准确率
- (b) 在 AffectNet 标签层面测 confusion matrix

**代码里有 `label_accuracy_generated` 和 `source_prediction_preserved` 字段,但这两个用的是 E₀ 自己的 logits,与训练 loss 同源**,不能算独立证据。需要用 off-the-shelf FER 模型(如 DDAM 或 Dynamic MLP)重新测。

**立得住**: **C3 Image quality**。FM loss 是 well-posed 的 flow matching,确实让 G 学到自然人脸流形。single_face_rate 高说明输出看起来像脸。FID/NIQE 独立计算,可信。唯一 caveat 是 best.pt 选择偏 `cos × face_rate`,可能 sacrifice 质量换 cos,但这是 ranking bias 不是 quality failure,实际 FID 数字仍可报。

**唯一需要补的是 mode collapse / diversity 测试** — G 是 sample_id 确定性的,没有真正随机性,diversity 指标可能很弱,审稿人会问"一个 sample_id 永远生成一张脸,这算不算 anonymization 还是只是 re-encoding?"。

### 最后一个隐性 claim 问题

README 写"生成器不得以 z₀ 为条件学习原图重建 G(ξ; z₀) ≈ x₀,否则将导致身份信息泄漏"。但代码 Stage 1 就用 `flow_condition: embedding` 拿 z₀ 做条件训 FM loss,FM loss 的 target 是 `x₁ = x₀` 本身。**FM loss 就是 "G(z₀, ξ, t) 学着把噪声推向 x₀"**。这与 README 的明确禁止直接冲突。

要么 README 描述的是"理想方法"而代码实现妥协了,要么 stage1/2 的概念在代码里被重新定义了。这件事 paper 必须讲清楚,否则审稿人对照代码与 paper 发现不符,会怀疑方法描述的可信度。

---

# Section D: 综合判断(本文档新增,不解决问题)

## D.1 三个 section 之间的因果链

把 A/B/C 拼起来,SAFA 的失败模式不是孤立现象,是**因果链**:

```
C.1 C1 claim 无训练信号保证 (匿名性靠假设)
    ↓
C.2 z₀ 严重过参数化 (512 维 vs 8 类表情)
    ↓
B.2 f: z₀ 维度过高 = 身份空间
    ↓
B.1 Crack 2: cosine 可被 game (off-manifold)
    ↓
A.4 观察 1: cos 涨 ≠ 方法成功 (v4/v8 都被 hack)
    ↓
A.4 观察 4: 所有实验终 sharpness < 200 (结构性质量退化)
```

**核心矛盾一句话**: 论文要"匿名化身份",但 loss 里没有任何一条对准"身份"这个目标,所有压力都压在 z₀ 表征空间假设上,而 z₀ 维度又过剩 — 这个裂缝贯穿整个 pipeline,**不是调一个超参或换一个起点能解决的**。

## D.2 现象 vs 根因对照

| 现象(A 区) | 表面原因(B 区) | 根因(C 区) |
|------------|-----------------|------------|
| 所有实验终 sharpness < 200 | Crack 1(一阶近似)+ Crack 3(FM game) | C3 没有直接质量监督,best.pt 选择偏 cos |
| v4/v8 cos 高但 mode collapse / sharpness 崩 | Crack 2(cosine game) | C2 用 z₀ 当表情代理,z₀ 含 identity |
| 起点 e14 vs e15 决定终态 | Crack 4(anti-affect basin) | C3 README 写"Stage1 null",代码却用 z₀ 当条件 — 起点决定不了 |
| PU-AdamW 名字 vs 实际行为 | B.2 j(PU 名不副实) | C.4 best.pt 用 cos × face_rate 当 criterion |
| EER 没报告 | audit.py 禁 face-id | C1 claim 没数学证据 |

## D.3 论文可写 vs 不可写的 claim(诚实版)

| 可写 | 不可写(当前 loss 下) |
|------|--------------------|
| "PU-AdamW 是稳定的两阶段更新规则,在 e14 起点上经验上达到 cos 0.97" | "PU-AdamW 保证收敛"(B.5) |
| "Audit 框架保证训练流程不含身份监督"(事实) | "无需 identity supervision 即可匿名化"(C.1) |
| "FM + repr cosine 在我们的 setup 下经验 trade-off 可达 cos 0.82-0.95" | "z₀ 是匿名化机制"(B.2 f) |
| "z₀ 跨 8 个 backbone Spearman 0.88,有 universality"(memory) | "z₀ 是 identity-untangled 表征"(未证明) |
| "v3 ep131 在 cos/NIQE/sharpness 上达到 Pareto knee"(A.3) | "我们方法 SOTA"(FID 68 vs CIAGAN FID 30-50) |

## D.4 三个核心开放问题(供你分析,本文档不解答)

**Q1**: SAFA 的"无 identity supervision"卖点,是 design choice 还是 handicap?
- 如果是 choice → 必须做 z₀ vs ArcFace 互信息 ablation,证明 z₀ 真的 identity-untangled
- 如果是 handicap → 需要重新设计 loss,加 identity loss(但要改 audit.py)

**Q2**: z₀ 维度 512 是结构性的吗?
- 512 维 = identity 信息容量过剩(Crack f)
- 降到多少维才"刚好够表情不够身份"? 没人测过
- 降维方法: PCA / supervised decorrelation / orthogonal projection to ArcFace direction

**Q3**: PU-AdamW 是真优化器还是更新规则?
- B.2 j 揭示它是 "AdamW(FM) + Projected SGD(repr)",没有 Adam state 跟踪 repr
- 论文里应该改名"Projected Two-Step Update"还是保留"PU-AdamW"?
- 名字会影响审稿人对方法的解读

## D.5 必须补的实验(不论最终 claim 怎么收缩)

1. **EER 测试**: 用独立 ArcFace recognizer(`InsightFaceRecognizer` 已有)在 v3 ep131 / v3 ep79 / v8 ep1735 上算 EER,与 CIAGAN / StyleID 对比
2. **z₀ vs ArcFace 互信息**: 测 z₀ 与 ArcFace embedding 的 CKA / SVCCA / linear probe accuracy
3. **独立 FER 验证**: 用 DDAM 或 Dynamic MLP 在 x̂ 上测表情准确率,与 x₀ 比
4. **diversity 测试**: 同一 sample_id 用不同 noise(打破确定性)生成,测 LPIPS diversity
5. **FID 重跑**: 所有 v 系列实验补 FID 数字(目前缺失)
6. **Stage 1/2 边界消融**: 当前 stage1.epochs=0 + Stage 1 实际用 z₀ 当条件。要么诚实地把 README 改成"single-stage conditional FM",要么真的训一个 null-condition Stage 1 看差别

---

# 附录: 文件清单(4029 服务器绝对路径)

**核心代码**:
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/training/g_loop.py`(主逻辑, 4399 行)
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/training/projected_update.py`(投影函数, 904 行)
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/training/representation_losses.py`(cosine / gram loss, 119 行)
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/training/audit.py`(106 行, 禁止清单)
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/models/generator.py`(1100 行)
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/models/e0.py`
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/models/conditioning.py`
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/training/latent_codec.py`
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/evaluation/runner.py`
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/evaluation/metrics.py`
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/evaluation/recognizers.py`
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/src/safa/utils/sampling.py`

**配置**:
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/configs/medium_v1/train_g_medium_v1_stage1.yaml`
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/configs/medium_v1/train_g_medium_v1_stage2_m0.yaml`
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/configs/medium_v2/experiments/e14resume_v3_budrelax_gpu23_ddp_200ep.yaml`(当前主配置)
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/README.md`

**实验产物**:
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/artifacts/checkpoints/e14resume_v{2,3,4,5,6,7}_*/metrics_history.jsonl`
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/artifacts/checkpoints/e15_cold_v8_lpips_gpu0123_ddp_150ep/metrics_history.jsonl`
- `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization/artifacts/eval/<exp>/quality/epoch_NNNN/generated_images/`

**参考文献**:
- [CIAGAN: Conditional Identity Anonymization Generative Adversarial Networks (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Maximov_CIAGAN_Conditional_Identity_Anonymization_Generative_Adversarial_Networks_CVPR_2020_paper.pdf)
- [Face Anonymization Made Simple (arXiv 2024)](https://arxiv.org/html/2411.00762v1)
- [A Key-Driven Framework for Identity-Preserving Face Anonymization (NDSS 2025)](https://www.ndss-symposium.org/wp-content/uploads/2025-729-paper.pdf)
- [StyleID: Identity Disentanglement for Anonymizing Faces (PoPETS 2023)](https://petsymposium.org/popets/2023/popets-2023-0016.pdf)
- [Facial Identity Anonymization via Intrinsic and Extrinsic Attention (CVPR 2024)](https://cvpr.thecvf.com/virtual/2024/poster/30728)
- [Fantômas: Understanding Face Anonymization Reversibility (PoPETS 2024)](https://petsymposium.org/popets/2024/popets-2024-0105.pdf)

---

**文档生成**: 2026-07-05,基于 3 个并行 SubAgent 收集的证据
**版本**: v1(初版,未解决问题,待用户分析)
