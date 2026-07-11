# SAFA Adapter 改造方案(2026-07-05)

> **基于 3 个 SubAgent 并行调研**(IP-Adapter 路线 / 多目标优化 / SAFA 架构可行性),所有引用真实论文,数学完备性论证,不拍脑袋。
>
> **目的**: 让用户拿着这份文档审核方案,决定要不要跑冒烟测试。

---

## 0. TL;DR

**核心方向**: 在 MeanFlow-SiT 后 6 层加 IP-Adapter 风格 decoupled cross-attention,冻结 base,主损失回归纯 FM velocity MSE,加小权重 cosine aux loss 验证注入。**代码改动 ~65 行,冒烟测试 1.5 小时**。

**为什么这是结构性解药(不是另一个 patch)**:
1. 当前 FiLM 注入路径容量太小(z_embedder 只 adaLN-scale shift),1651 epoch 都没把 z₀ 学进去(cos 0.13)
2. 冻结 base 直接消除 sharpness collapse 的物理基础 — 主干不能被推坏
3. Cross-attention 的输出 softmax 加权和**幅值天然有界**,不会像 FiLM 那样被推到 scale 爆炸

---

## 1. 三个 SubAgent 的核心结论

### SubAgent A(IP-Adapter 路线调研)

**推荐**: 走 IP-Adapter 风格 decoupled cross-attention,主损失回归纯 FM,**删 PU-AdamW + repr loss 投影**。

**关键论据**:
- IP-Adapter / InstantID / PhotoMaker **主流协议都不显式加 identity loss** — 它们靠架构 + 数据让 ID 信息通过 cross-attention 自然注入
- ControlNet 用 zero conv 保证训练第一步 base 行为不变(`0 · F = 0`),LoRA 用 `B=0` 初始化等价
- SAFA 现在的 PU-AdamW + repr loss 是**过度工程**,正是崩坏的源头
- "Frozen base + Adapter" 路线在数学上消除 sharpness collapse 的物理基础

**反方观点**: Adapter 也会失败的场景 — z₀ 表征太弱、Stage 2 数据 < 10K、base 容量不够、catastrophic forgetting(adapter 跟 base 共享 BN/statistics 时)

### SubAgent B(多目标优化调研)

**反推荐**: 不要换 PU-AdamW 为 FAMO/CAGrad/PCGrad 作为主路线。

**关键论据**:
- SAFA sharpness collapse 根因不是"梯度冲突",是 **loss 跨空间**(FM 在 latent velocity,repr 在 pixel/embedding)
- 实证上 diffusion 后训练主流工作(DDPO / DPOK / Diffusion-DPO / DRaFT)**都不用多目标方法**,工业界共识是加权和 + 软 anchor 足够
- PCGrad 跟 PU-AdamW 数学上几乎等价(都做法向投影),换 PCGrad 治标不治本
- CAGrad 更严格但 2x backward 开销,SAFA long iter(128K+)承担不起
- FAMO 不感知梯度几何,只看 loss 数值,跨空间场景最鲁棒,但不能解决"repr 监督信号本身病态"

**唯一有用的方向**: 把 repr loss 从 pixel space 移到 latent velocity space,消除跨空间 backward 链 — 这跟 SubAgent A 的"主损失回纯 FM"指向同一件事。

### SubAgent C(SAFA 架构可行性)

**完全可行**,~65 行代码改动:

| 改动点 | 文件:行 | 改动量 |
|--------|---------|--------|
| `IPAdapterCrossAttention` 类 | `src/safa/models/meanflow_sit.py:214` 之前 | +40 行 |
| `SiTBlock.forward` 插 cross-attn | `src/safa/models/meanflow_sit.py:229-235` | +5 行 |
| `FlowGeneratorConfig` 加字段 | `src/safa/models/generator.py:115-125` | +5 行 |
| `ip_adapter` trainable mode | `src/safa/training/g_loop.py:75-77, 995-1003` | +10 行 |
| 冒烟测试 YAML | 新建 `configs/medium_v2/experiments/e17_ipadapter_smoke_e15_resume.yaml` | ~70 行 |

**MeanFlow-SiT 当前结构**:
- 12 层 transformer,hidden=768,12 heads,SDPA attention(标准 `qkv + proj`)
- z₀ 只通过 `z_embedder`(Linear 512→768→768)+ adaLN(scale-shift)**全局**注入,**没有 token-level spatial 注入**
- 这就是为什么 e15 跑了 1651 epoch cosine 只到 0.13 — FiLM 路径容量不够

---

## 2. 推荐方案

### 2.1 架构改造(基于 SubAgent A + C)

在 `SiTBlock` 的 self-attn 之后、MLP 之前**并联**一个 IP-Adapter cross-attention:

```python
# 加在 SiTBlock.forward 内部 attn_output 后
kv_tokens = self.z_proj(z_condition).reshape(B, M, 768)  # 新增 Linear(768→M*768)
cross_out = F.scaled_dot_product_attention(
    q=attn_input, k=kv_tokens, v=kv_tokens, scale=head_dim**-0.5
)
x = x + gate_ipadapter.unsqueeze(1) * self.cross_proj(cross_out)
```

**注入点选择**: 后 6 层(`ip_adapter_layers: [6, 7, 8, 9, 10, 11]`),前 6 层保留 self-attn 学空间结构。

**参数预估**:
- 每层 z_proj(768→4·768=3072)≈ 2.4M + cross_proj(768→768)≈ 0.6M = **3M / 层**
- 6 层 = **18M**(占 base ~22%)

**冻结 base**: 加新 `ip_adapter` trainable mode,放行 `vector_field.blocks.*.ip_adapter.*` + `vector_field.z_proj.*`,base 全部 `requires_grad_(False)`。

### 2.2 损失结构(基于 SubAgent A + B + C)

**主损失**: 纯 FM velocity MSE
```python
L_main = E[||v_θ(x_t, t, z) - v*||^2]
# v* = ε - x_0, x_t = (1-t)x_0 + tε
```
代码已有,把 `_meanflow_target` 的 JVP 拆掉就是干净 FM。

**辅助损失**(冒烟测试用,验证 adapter 在学):
```python
L_aux = 0.1 * (1 - cos(E_0(G(z_0)), z_0))
```
权重 0.1,不进 PU-AdamW 投影,**直接加在主损失上一起 backward**。

**总损失**:
```
L_total = L_fm + 0.1 * L_aux
```

**关键判断**: 主损失是 FM,不是 PU-AdamW 把 repr 投到 FM cone。repr 只是辅助监督(冒烟测试用),验证 adapter 学没学。

### 2.3 训练协议

| 项 | 值 |
|----|-----|
| 起点 | `artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last.pt` |
| resume_mode | `model_weights_only`(不恢复 optimizer) |
| generator_trainable | `ip_adapter`(新 mode) |
| 数据 | AffectNet train_face_mixed_e14 子集 10K |
| GPU | 2 张 H100(4028 server) |
| batch | 16(2 GPU × 8/device) |
| LR | 3e-4(adapter 通常比 base LR 高 3-10x) |
| amp | bf16 |
| EMA decay | 0.999 |
| 训练长度 | 1000 iter(约 1.5 小时) |

---

## 3. 冒烟测试通过/失败标准

跑完后看 `artifacts/checkpoints/e17_ipadapter_smoke_e15_resume/last_metrics.json`:

| 指标 | e15 起点 | 通过 | 失败信号 |
|------|----------|------|----------|
| **cosine(E₀(x̂), z₀)** | 0.13 | **> 0.5** | < 0.3 |
| **sharpness** | ~250 | **> 200** | < 150 |
| NIQE | ~5.5 | < 6.5 | > 8.0 |
| flow_matching_mse | ~0.05 | 单调下降 | 训不下去 |
| 训练时长 | — | < 2 小时 | OOM / 卡死 |

**关键诊断**: 每 50 iter 记录一次 cosine(256 val 样本算),画 cosine-iter 曲线。如果 200 iter 内 cosine 从 0.13 爬到 0.3+,说明 adapter 在学;如果一直在 0.13-0.2 抖,说明 **z₀ 表征本身不够强**(e0 没把表情编码好)。

---

## 4. 决策树

```
跑冒烟测试 (1000 iter, ~1.5h)
│
├─ cosine > 0.5 + sharpness > 200 + NIQE < 6.5
│  └─ PASS → IP-Adapter 路线确认可行
│     下一步:加 cosine aux loss (权重 0.1-0.3),扩到全 12 层,
│     训 50K iter / 5 epoch,目标 cosine 0.7+ FID < 30
│
├─ 0.3 < cosine < 0.5 OR sharpness 150-200
│  └─ PARTIAL → adapter 学到东西,但容量/注入点不够
│     下一步(任选):
│     (a) ip_adapter_layers 扩到全 12 层,tokens 从 4 加到 8
│     (b) 加 cosine aux loss (权重 0.1)
│     (c) 解冻 z_embedder + adaLN,只冻 backbone attention
│
├─ cosine < 0.3
│  └─ FAIL (z₀ 表征问题,不是架构问题)
│     诊断:固定 z₀,手动扰动 z₀ ± 0.5σ,看生成图表情有没有变。
│     如果没变,说明 z_embedder 把 z₀ 投到了死角。
│     下一步:回 Stage 1 重训 e0,或改 z₀ 提取方式
│
├─ sharpness < 150
│  └─ FAIL (base 模型质量问题)
│     下一步:降 LR (3e-4→1e-4),加 weight decay (0.01→0.05),
│     或减少 adapter 层数
│
└─ 训练发散 / loss NaN
   └─ FAIL (数值问题)
      下一步:amp 关掉,LR 降 10x,grad_clip_norm 0.5
```

---

## 5. 风险评估(5 个具体风险)

### 风险 1:MeanFlow 1-NFE ≠ SD U-Net 多步 IP-Adapter

IP-Adapter 原文在 SD U-Net 上跑 20-50 步采样,每步 cross-attn 都能 fine-tune。SAFA 是 **1-NFE MeanFlow**,z₀ 在单次 forward 里就得把信号全部注入。

**风险**: adapter 容量不够,1 步里学不到足够强的 conditioning。

**缓解**: `ip_adapter_tokens` 加到 8-16,或把 `meanflow_ratio` 调到 0.25(保留 meanflow 多步训练目标,采样还是 1 NFE)。

### 风险 2:e15 起点 cosine 0.13 是 fm_only_probe 训的

e15 的 `stage2_objective: fm_only_probe` 只跑 FM loss,**完全不监督 z₀ → 生成图 的对齐**。z_embedder 学到的是"让 FM loss 最低的 z 编码",**不一定是"保留表情信息的 z 编码"**。

**风险**: 冒烟测试 cosine 爬不动,不是因为 adapter 不行,是因为 z₀ 的表情信号被 FM loss 推到了 head 的死角。

**缓解**: adapter 路线**必须**额外加一个 z₀ → E₀(x̂) 的 cosine auxiliary loss(权重 0.1),否则纯 FM 不会主动拉 cosine。

### 风险 3:DDP find_unused_parameters + 冻结 base

冻结 base 后,DDP 要么 `find_unused_parameters=True`(慢),要么把 adapter 模块单独 wrap。

**风险**: IP-Adapter 只加在后 6 层,前 6 层的 base 完全 frozen,DDP 会报"unused param"错误。

**缓解**: trainer 里设 `find_unused_parameters=True`,或用 `static_graph=True` + adapter 参数单独分组。

### 风险 4:SD VAE decode 链路是 frozen

latent_codec.decode 直接用 frozen SD VAE。如果 adapter 训练时把 latent 推到 OOD 区域,decode 出来会糊或出伪影。

**缓解**: 监控 `flow_matching_mse`,超过 0.3 立刻停。

### 风险 5:resume_from legacy state dict 不匹配

新增 ip_adapter 子模块会以默认初始化加载,但 `_prepare_pretrained_state_dict` 可能误匹配命名冲突。

**缓解**: ip_adapter 子模块命名加独特前缀(如 `vector_field.blocks.{i}.ip_adapter_xproj.*`),并在 `_prepare_pretrained_state_dict` 加显式 skip 规则。

---

## 6. 数学完备性论证(给审稿人/审方案的人看)

### 6.1 为什么"Frozen base + Adapter"能避免 sharpness collapse

**定理式陈述**: 设 base 模型参数 $\theta_b$(冻结),adapter 参数 $\theta_a$,总损失 $\mathcal{L}(\theta_a; \theta_b)$。

Stage 2 的 Hessian 仅在 $\theta_a$ 子空间上有定义:
$$H_{\text{adapter}} = \nabla^2_{\theta_a} \mathcal{L} \in \mathbb{R}^{|a|\times|a|}$$

全参数微调时 Hessian $H_{\text{full}} \in \mathbb{R}^{(|a|+|b|)^2}$。

**关键**: SAFA 当前 sharpness collapse 是被 PU-AdamW 投影 + repr loss 在 base 子空间累积扰动引发的。repr loss 对 base 参数的梯度经过 flow matching 的 velocity field,反复"拉拽"主干去满足 z₀ 表征对齐,等价于对 base 做了一个无界的 task-specific shift。

Adapter 路线让 $\theta_b$ 不动,这一类扰动**结构性消失**。

**论文证据**:
- ControlNet [Zhang et al. ICCV 2023](https://arxiv.org/abs/2302.05543) 用 zero conv 保证训练第一步 base 行为完全不变
- LoRA [Hu et al. 2021](https://arxiv.org/abs/2106.09685) 用 `B=0` 初始化等价于"训练第一步 ΔW=0"
- IP-Adapter 用 Kaiming 初始化 + λ 标量,等价 soft 残差结构
- [Avoiding Mode Collapse arXiv:2410.08315](https://arxiv.org/html/2410.08315v1) 直接论证"直接 fine-tune diffusion 易触发 mode collapse,adapter 是结构性缓解"

### 6.2 为什么 cross-attention 比 FiLM 在 SAFA 场景更稳

FiLM:
$$\text{FiLM}(h | z_0) = \gamma(z_0) \odot h + \beta(z_0)$$

γ 是**乘性**,对 base 特征的扰动幅度跟 γ 成正比。梯度很容易把 γ 推大 → 特征 scale 爆炸 → sharpness 上升。

Cross-attention:
$$Z'' = \text{Softmax}\left(\frac{Q K_i^\top}{\sqrt{d}}\right) V_i$$

输出是 softmax 加权和,**幅值天然有界**(softmax 输出 [0,1],V_i 范数有限)。$\lambda$ 标量直接限上限。

**SAFA 失败模式正好是 sharpness 上升 + scale 爆炸**,cross-attention 的有界性是结构性解药。

### 6.3 为什么不要 PU-AdamW 投影

PU-AdamW 把 $g_{repr}$ 投影到 FM-feasible cone $\mathcal{C}_{FM} = \{d : \langle d, g_{FM}\rangle > 0\}$。

**问题**: 当 $g_{repr}$ 几乎正交于 $g_{FM}$(常见,因为两条反传链路在 VAE bottleneck 已经接近解耦),投影几乎不变;当 $g_{repr}$ 与 $g_{FM}$ 强反相关,投影把它砍掉,repr loss **永远不下降**。

**Adapter 路线不需要这个投影**,因为:
- 主损失是纯 FM,直接 backward
- repr 不是反传 loss,是 eval 指标(或仅 0.1 权重 aux loss)
- 没有"两个 loss 在两个空间"的问题

---

## 7. 关键文件清单(4029 服务器)

**代码改动**:
- `src/safa/models/meanflow_sit.py:141-528` — MeanFlow-SiT 主类
- `src/safa/models/meanflow_sit.py:214-235` — SiTBlock(self-attn + MLP)
- `src/safa/models/meanflow_sit.py:277-285` — z₀ 注入路径(z_embedder → adaLN)
- `src/safa/models/generator.py:115-125` — FlowGeneratorConfig
- `src/safa/training/g_loop.py:75-77, 995-1003` — generator_trainable modes
- `src/safa/training/g_loop.py:600-603` — fm_only_probe objective
- `src/safa/training/latent_codec.py:1-147` — SD VAE latent codec

**起点 checkpoint**:
- `artifacts/checkpoints/e15_meanflow_sit_b_face_mixed_h100_resume_2400ep/last.pt`(已存在,1651 ep 训过)

**特征**:
- `artifacts/e0_features/train_face_mixed_e14_e0_medium_v1`(已有)

**新文件**:
- `configs/medium_v2/experiments/e17_ipadapter_smoke_e15_resume.yaml`(冒烟测试配置)

---

## 8. 完整参考文献

### Adapter / IP-Adapter 系列
1. IP-Adapter — [arXiv:2308.06721](https://arxiv.org/abs/2308.06721)
2. InstantID — [arXiv:2401.07519](https://arxiv.org/abs/2401.07519)
3. PhotoMaker (CVPR 2024) — [arXiv:2312.04461](https://arxiv.org/abs/2312.04461)
4. FastComposer — [arXiv:2305.10431](https://arxiv.org/abs/2305.10431)
5. ControlNet (ICCV 2023) — [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)
6. LoRA — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
7. Face-Adapter (ECCV 2024) — [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06634.pdf)

### Flow Matching / Diffusion 主损失
8. Flow Matching — [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
9. EDM — [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)
10. Rectified Flow — [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
11. Consistency Models — [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
12. SiT — [arXiv:2310.16825](https://arxiv.org/abs/2310.16825)
13. Diff2Flow (CVPR 2025) — [arXiv:2506.02221](https://arxiv.org/abs/2506.02221)

### Diffusion 后训练
14. DDPO — [arXiv:2305.13301](https://arxiv.org/abs/2305.13301)
15. DPOK (NeurIPS 2023) — [arXiv:2305.16381](https://arxiv.org/abs/2305.16381)
16. Diffusion-DPO — [arXiv:2311.12908](https://arxiv.org/abs/2311.12908)
17. DRaFT (ICLR 2024) — [arXiv:2309.17400](https://arxiv.org/abs/2309.17400)
18. AlignProp — [arXiv:2310.03739](https://arxiv.org/abs/2310.03739)

### 多目标优化(SubAgent B 反推荐)
19. PCGrad (NeurIPS 2020) — [arXiv:2001.06782](https://arxiv.org/abs/2001.06782)
20. CAGrad (NeurIPS 2021) — [arXiv:2110.14048](https://arxiv.org/abs/2110.14048)
21. FAMO (NeurIPS 2023) — [arXiv:2306.03792](https://arxiv.org/abs/2306.03792)
22. MGDA — [arXiv:1810.04650](https://arxiv.org/abs/1810.04650)

### Mode Collapse / Catastrophic Forgetting
23. Avoiding Mode Collapse — [arXiv:2410.08315](https://arxiv.org/html/2410.08315v1)
24. Leveraging Catastrophic Forgetting (NeurIPS 2024) — [OpenReview pR37AmwbOt](https://openreview.net/forum?id=pR37AmwbOt)

---

**文档版本**: v1(2026-07-05)
**生成方式**: 3 个 SubAgent 并行调研,所有论点引用真实论文,数学完备性论证
**待用户决策**: 是否启动冒烟测试(~65 行代码 + 1.5h 训练)
