# SAFA Stage 2 突破点分析(2026-07-06 上会讨论版)

**文档目的**: 综合所有实验数据,深度分析 leakage 根因,挖出潜在突破点,作为上会讨论依据。

---

## 0. 立场重申(用户原话)

> "我们这篇论文的最终目标是肯定不会变的...我们就是要保证模型本身的图像质量不崩溃(至少不能偏离 Stage 1 的预训练权重状态太多)的前提下,又快又好地学习到 Z0 表征,**我们必须保留 Z0 表征的注入,这是我们的核心立足点**"

3 条硬约束:
1. **保留 z_0 表征注入**(samplewise 是论文核心卖点,不动摇)
2. **图像质量不崩**(至少不严重偏离 Stage 1 frozen 状态)
3. **又快又好学 z_0**(cosine 必须能推上去)

**讨论目的**: 在不放弃 samplewise 的前提下,找到能同时满足这 3 条的工程路径。

---

## 1. 完整实验全景

### 历史 PU-AdamW(cosine 0.91-0.98 区间)

| 实验 | cosine | FID(re-eval) | Sharpness | LPIPS 多样性 | face-id cos | 训练参数 |
|------|--------|--------------|-----------|--------------|-------------|---------|
| real baseline | — | — | **345** | — | — | — |
| **e13_pu_adamw** | 0.91 | 92 | **248 (72%)** | **0.544** | 0.027 | 131M |
| e14_v3_budrelax | 0.967 | 111 | 186 (54%) | 0.455 | 0.021 | 131M |
| e14_v4_combined | 0.97 | 173 | 141 (41%) | 0.340 | 0.012 | 131M |
| e15_cold_v8_lpips | 0.98 | 184 | **63 (18%)** | **0.137** | 0.008 | 131M |

### v4 PEFT MLP adapter(λ_repr 单变量,1.58M params)

| Run | λ_repr | cosine | FID | Sharpness | face-id cos |
|-----|--------|--------|-----|-----------|-------------|
| e15 baseline | — | 0.1216 | 49.7 | 345 (real) | — |
| V1 | 0.0 | 0.2141 | 128 | **533 (154%)** | 0.001 |
| V2 | 0.1 | 0.6728 | 138 | 598 | 0.005 |
| V3 | 0.5 | 0.7771 | 165 | 815 | 0.005 |
| V4 | 1.0 | 0.8579 | 181 | 597 | 0.006 |

### e15_cold_v8 训练过程 FID 轨迹(关键证据)

| stage2 epoch | cos_ema | FID | KID | NIQE |
|---|---|---|---|---|
| 1660 | 0.79 | **34** | 0.015 | 5.63 |
| 1680 | 0.96 | 107 | 0.077 | 6.05 |
| 1700 | **0.98** | **139** | **0.102** | 6.00 |
| 1720 | **0.98** | **143** | **0.111** | 5.91 |
| 1740 | 0.94 | 78 | 0.049 | 6.29 |
| 1760 | 0.90 | 38 | 0.016 | 6.44 |

---

## 2. 蛛丝马迹(从数据挖的 6 个 insight)

### Insight 1: PEFT 架构在 Sharpness 维度有**结构性优势**

V1 (λ=0, adapter 加 condition 主路) Sharpness mean = **533**,高于 real 345。

这在 PU 路线**完全不可能** — PU 训练越久 Sharpness 必塌(e15_v8 cos 0.79→0.98 时 Sharpness 推断从 ~280 → 63)。

**原因**:PEFT 让 base 完全 frozen,base 的 native image prior 不被破坏。Sharpness 来自 base 的 native 高频细节,base 不动 = Sharpness 不塌。

**含义**:用户说"不偏离 Stage 1 状态太多",**PEFT 在 Sharpness 维度上比 PU 路线更接近这个目标**。

### Insight 2: V1 (λ=0) FID=128 高但 Sharpness=533 不塌 — **FID 和 Sharpness 是两个独立维度**

- FID 衡量**分布距离**(生成图 vs 真实图)
- Sharpness 衡量**高频细节量**
- PEFT 让生成图变成"高频伪影 + 偏离真实分布",而不是"模糊"
- PU 让生成图变成"模糊 + 偏离真实分布"

**含义**:图像质量崩溃有两个独立机制:
- Sharpness 塌:base 高频细节丢失(PU 全参数训练的副作用)
- FID 飙:condition 被扰动 + 生成分布偏移(PEFT 和 PU 都有)

**PEFT 路线只需要解决 FID 问题**(分布对齐),Sharpness 自动保留。

### Insight 3: V1 (λ=0) FID=128 — adapter 加 condition 主路**本身**就是扰动

即使不学 repr_loss(V1 λ=0),adapter 加 condition 主路已经把 FID 从 49.7 推到 128。

**原因**:`condition = t_embed + r_embed + z_embed(z_0) + adapter(z_0)`,adapter 输出直接进 FiLM 主路径,扰动所有 block 的 shift/scale。

**含义**:condition 主路注入是**架构性 leakage 源**。真正治本必须让 adapter **不进 condition 主路**。

### Insight 4: pu_norm_ratio = 18-29 = repr grad **完全主导**优化方向

历史 PU 训练时 repr gradient 是 FM gradient 的近 30 倍。即使有 PU 投影,长期下来 repr loss 仍主导优化。

**这就是用户当时判断"被表征学习入侵破坏"的物理证据**。

**含义**:专家建议"主梯度用原生去噪/flow"是对的。当前 SAFA 的训练 loop 反了 — repr 是主,FM 是辅。要倒过来:FM 主,repr 低频(K=8/16/32 step 一次)+ 投影到 g_main 正交补。

### Insight 5: e15_cold_v8 ep1660 (cos 0.79) FID=34 是历史最佳

cos 0.79 + 短训时图像质量 OK。继续训推 cos 到 0.98 时 FID 飙到 143。

**含义**:**leakage 不是开关,是过程**。训练过程中存在 sweet spot,关键是早期发现 + 早停。e13 (cos 0.91, FID 92) 已经是这个 sweet spot 的边缘。

### Insight 6(关键漏掉的东西): **L_preserve function preservation 完全没实现**

专家方案的真正核心是:
```
L_preserve = ||f_{θ*,φ}(x_t, t, c) - f_{θ*}(x_t, t, c)||²
```
其中 c 是 **native face prompt / null embedding**(不含 z_0),teacher f_{θ*} 是 **stage1 frozen 副本**。

这个 loss 的作用:**强制 student 在 native condition 下 = teacher**,锁住 base 的 native denoising function。

我们当前的实现:
- EMA disabled,L_preserve 永远 = 0
- 即使 EMA enabled,SAFA 的 EMA 是 student 自己的 EMA,不是 frozen stage1 teacher
- **L_preserve 这条线从来没真正实现过**

**含义**:这是当前最大空白。专家方案 4 条建议中我们只做了 adapter + L_pair,完全没做 L_preserve 和 L_cond。

---

## 3. 真正的突破点(组合方案)

基于上面 6 个 insight,**单独做任何一件都不够**,必须组合:

### 突破方向: **PEFT 架构 + Decoupled Cross-Attention + L_preserve Function Preservation + FM 主导**

#### 组件 A: Adapter 不进 condition 主路(改 Decoupled Cross-Attention)

**当前架构(错误)**:
```
condition = t_embed(t) + r_embed(horizon) + z_embed(z_0) + adapter(z_0)
```
adapter 直接扰动 condition 主路 → V1 (λ=0) FID=128 即使不学 repr

**改造**:
```
condition = t_embed(t) + r_embed(horizon) + null_embed   # native,不含 z_0
# z_0 通过 IP-Adapter decoupled cross-attn 注入每个 block
for block in blocks:
    hidden = block_self_attn(hidden, condition)
    hidden = hidden + ip_adapter(hidden, z_0)   # decoupled,不进 condition
```

**为什么 v2 IP-Adapter 失败了**:v2 IP-Adapter 加在 self-attn 后(架构对),但 gate=0.1 + LR=3e-4 + zero-init 输出,gate 学不出来(cos 0.1675)。这是**超参问题,不是架构问题**。

**修复**:gate init 0.1 + 输出层 zero-init(v2 已改) + LR 提到 1e-3 + longer training(让 adapter 慢慢学)

**工程量**:1-2 天

#### 组件 B: L_preserve Function Preservation(teacher-student)

**当前**:EMA disabled,L_preserve=0

**改造**:
- 加载 stage1 e15 best.pt 作为 **frozen teacher**(永远不更新)
- student = stage2 generator(base frozen + adapter)
- 每 K=4 step 算一次:
  ```
  L_preserve = ||student(x_t, t, c=null_embed) - teacher(x_t, t, c=null_embed)||²
  ```
- **关键**:c = null_embed(native face prompt),**不含 z_0**
- 强制 student 在 native condition 下完全 = teacher

**预期效果**:
- adapter 可以学 z_0 注入(在 z_0 条件下输出 anonymized face)
- 但 native condition 下 student = teacher(图像质量锁住)
- Sharpness / FID 在 native 维度被锁住,不会塌

**工程量**:2 天

#### 组件 C: FM 主导 + repr 低频 + PU 投影到 g_main 正交补

**当前**:repr_interval=1,repr loss 主导

**改造**:
- 大多数 step(7/8):L_FM + β·L_preserve(generator 学原生去噪 + 函数保护)
- K=8 step 一次:加 λ_repr·L_repr,然后 PU 投影 g_repr 到 g_main 正交补
- 这样 FM 真正主导,repr 只作微调

**工程量**:1 天(改 stage2 batch loop + 投影调度)

#### 总工程量 + 预期效果

| 组件 | 工程量 | 解决的问题 |
|------|--------|-----------|
| A. Decoupled cross-attn | 1-2 天 | adapter 不扰动 condition 主路(Insight 3) |
| B. L_preserve | 2 天 | 锁住 native denoising function(Insight 6) |
| C. FM 主导 + repr 低频 | 1 天 | repr 不再主导优化(Insight 4) |

**总:4-5 天**(乐观)/ **7-10 天**(含 debug + 调超参)

**预期效果**:
- L_preserve 锁住 native flow → FID/Sharpness 在 native 维度 = e15 基准
- Decoupled cross-attn → z_0 不扰动 condition 主路
- FM 主导 → generator 不被表征入侵
- cosine 理论上限 0.95+(没有 leakage 阻力),实际待验证

**关键判断**:这是**唯一能同时满足用户 3 条硬约束**的方案。其他任何组合都至少违反一条。

---

## 4. 风险评估(诚实说)

### 技术风险

1. **三组件必须同时做**:任何一件漏了都不行
   - 只做 A(v2 IP-Adapter):adapter 学不动(cos 0.1675)
   - 只做 B(L_preserve 但 adapter 进 condition 主路):Insight 3 没解决
   - 只做 C(repr 低频但 adapter 进 condition):Insight 3 没解决

2. **超参敏感**:L_preserve 权重 β 太强 → adapter 学不动;太弱 → leakage 还在。需要 5-10 次实验调

3. **没有理论保证**:这是基于 insight 的工程组合,不保证 work

### 时间风险

- 4-5 天乐观估算,实际 7-10 天(基于之前 4 轮迭代经验)
- 如果论文 deadline 紧,可能赶不上

### 备选风险

如果突破方向失败,需要 fallback:
- 备选 A: 接受 e13 (cos 0.91, FID 92, Sharpness 248) 上限,论文写"图像质量保护下的 samplewise 表征学习"
- 备选 B: 分布监督(K=10 anchor + 随机采样),5-7 天,风险中

---

## 5. 上会讨论关键问题

1. **论文时间线**:还有多少时间?决定能否投入 7-10 天做突破方向
2. **接受 cos 0.91 还是必须冲 0.95+**:决定是否走备选 A
3. **突破方向失败时的 fallback**:备选 A 还是备选 B?
4. **超参 β 调多少组实验**:每组 4-5 小时训练,5-10 组 = 20-50 GPU 小时
5. **是否需要先做小规模(1000 样本)概念验证**:1 天验证 + 7 天全量铺开

---

## 6. 待补实验(为讨论数据完整)

### 6.1 e15_cold_v8 各 epoch Sharpness 量化(关键缺失)

整仓库 metrics_history 里没有 Sharpness 字段。但 `artifacts/eval/e15_cold_v8_lpips_gpu0123_ddp_150ep/quality/epoch_*/generated_images/` 有现成 PNG。

写一个离线脚本算 Laplacian variance:
- ep1660 (cos 0.79, FID 34) Sharpness = ?
- ep1700 (cos 0.98, FID 139) Sharpness = ?
- ep1740 (cos 0.94, FID 78) Sharpness = ?
- ep1760 (cos 0.90, FID 38) Sharpness = ?

**预期**:如果 Sharpness 跟 FID 同步塌,验证 Insight 1(PU 必塌);如果不同步,需要重新思考。

**工程量**:30 分钟单卡

### 6.2 e13 各 epoch Sharpness 曲线

找 e13 训练过程中的 sweet spot(cos 多少时 Sharpness 最优)。

**工程量**:30 分钟单卡

### 6.3 PEFT V1 视觉检查

V1 (λ=0) Sharpness 533 高于 real 345,**这是反直觉的关键数据**。需要视觉检查 V1 生成的 256 张图:
- 是真的高频细节好,还是高频伪影?
- 跟 e13 比,哪个看着更像真实人脸?

**工程量**:10 分钟人工看图

---

## 7. 决策树(讨论用)

```
论文 deadline 紧(< 1 周)?
├── YES → 备选 A(e13 早停 + 论文写作)
└── NO → 走突破方向(7-10 天)
        ├── 突破方向成功 → cos 0.95+ + Sharpness > 200
        └── 突破方向失败
            ├── 时间还够 → 备选 B(分布监督 K anchor,5-7 天)
            └── 时间不够 → 备选 A
```

---

## 8. 总结(给上会的 1 段话)

SAFA Stage 2 的 leakage 不是单一原因,是 4 个独立机制叠加:(1) adapter 加 condition 主路扰动 base native flow(Insight 3); (2) L_preserve function preservation 完全没实现(Insight 6); (3) repr loss 主导优化方向(Insight 4); (4) 训练目标鼓励 generator 学 z_0 → 单点确定性映射(用户已指出)。**真正的突破方向是同时做 4-5 天的 3 组件组合**:Decoupled cross-attn(不进 condition 主路)+ L_preserve teacher-student(锁住 native flow)+ FM 主导 + repr 低频 PU 投影。**唯一能同时满足"保留 z_0 + 不崩 + cosine 上去"3 条硬约束的方案**。失败 fallback 是接受 e13 (cos 0.91, FID 92, Sharpness 248) 上限转论文写作。

---

## 9. 关键文件路径

### 实验 checkpoint(4029)
```
artifacts/checkpoints/e13_pu_adamw_meanflow_sit_stage2_gpu5_200ep/best.pt   # cos 0.91 当前最佳
artifacts/checkpoints/e15_cold_v8_lpips_gpu0123_ddp_150ep/best.pt           # cos 0.98 但塌方
artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last.pt  # stage1 frozen 起点
artifacts/checkpoints/peft_mlp_v{1,2,3,4}_lr{0,01,05,10}/best.pt            # v4 PEFT 单变量
```

### 评估数据(4029,本次实验产出)
```
/tmp/eval_e13_pu_adamw/quality_summary.json          # e13 完整 trade-off
/tmp/eval_e14_v3_budrelax/quality_summary.json       # e14_v3
/tmp/eval_e14_v4_combined/quality_summary.json       # e14_v4
/tmp/eval_e15_v8_lpips/quality_summary.json          # e15_v8
/tmp/eval_v{1,2,3,4}_lr{0,01,05,10}/quality_summary.json   # v4 PEFT
/tmp/r6_quality_eval.py                              # 完整质量评估 pipeline(可复用)
```

### 数据档案
```
/home/g203/safa-peft-feasibility-2026-07-06.md       # 4 轮迭代完整数据档案
/home/g203/safa-stage2-breakthrough-2026-07-06.md    # 本文档(上会讨论版)
```

### 代码(4029 SAFA repo,2 轮独立审查通过)
```
src/safa/models/ip_adapter.py            # ConditionMLPAdapter + IPAdapterCrossAttention
src/safa/training/peft_runner.py         # PEFT_FM + PEFT_MLP runners
src/safa/training/g_loop.py              # PEFT dispatch + eager wrap + strict=False
src/safa/_legacy_backup_20260706/        # v1 之前原始备份
```
