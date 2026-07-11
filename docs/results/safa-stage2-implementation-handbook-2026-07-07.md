# SAFA Stage 2 突破方向 — 实施手册(2026-07-07 最终版)

**用途**: 基于专家完整回应 + 4 轮 PEFT 迭代 + 历史 PU 数据,整理成可直接进 Phase 0 实现的工程手册。

---

## 0. 一句话规则

> **对 MeanFlow-SiT(adaLN-native backbone),第一版就做: `LoRA on adaLN_modulation + gated low-rank z residual`,用 FFHQ 做 generic main loop,SAFA 做 sparse repr loop,step ratio 从 12:1 起,不再碰 IP-Adapter-after-self-attn 这条线。**

---

## 1. 核心架构决策(基于专家 #10)

### 1.1 v2 IP-Adapter 失败根因 = 架构错配

DiT 论文明确: 在 adaLN / cross-attn / extra-token 三种条件注入里,**adaLN works best**。MeanFlow-SiT 用 adaLN modulation(单向量 condition → chunk(6) → FiLM 所有 block),**不是 token-cross-attn-native backbone**。

IP-Adapter 的 decoupled cross-attn 需要原生 text token cross-attn 接口才能接,MeanFlow-SiT 没有这个接口,硬插在 self-attn 后只是盲目残差,改不到 condition 主路径。**这是 v2 cosine 0.1675 的真正根因,不是超参问题**。

### 1.2 新架构: LoRA + Gated Low-Rank Residual

```
# 1. z 注入路径(gated, init g=0)
delta_z = g · BA(z)              # g: scalar gate init=0; B/A: low-rank matrices
u = c_native + delta_z           # c_native = t_embed(t) + r_embed(horizon) + null_embed

# 2. adaLN_modulation 内部加 LoRA(无 gate,small init)
m_i(u) = W_i · u + LoRA_i(u)     # 每个 block i 的 adaLN_modulation

# 3. drop(z) 状态(generic main loop)
# z 全关 = g 路径 forward 但 g=0 → delta_z = 0 → u = c_native
# 这就是专家说的 "drop 整个 z 分支输出 = teacher/native path"
```

**LoRA 加在哪**:
- ✓ 每个 block 的 `adaLN_modulation` 内部 Linear
- ✗ 不动 `t_embedder / r_embedder / z_embedder`(第一版冻结)
- ✗ 不动 self-attn / mlp / norm

### 1.3 init 策略

| 参数 | init | 理由 |
|------|------|------|
| `g` (gate) | **0** | step 0 不破坏 base,慢慢长出来(ControlNet / Perfusion 范式) |
| `B` (low-rank output) | xavier_uniform | 标准初始化 |
| `A` (low-rank input) | xavier_uniform | 标准初始化 |
| `LoRA_a / LoRA_b` (adaLN 内部) | `a: kaiming, b: 0` | 标准 LoRA init(b=0 保证 step 0 LoRA 输出 = 0) |

**关键**: gate=0 + LoRA_b=0,step 0 整个 adapter 输出严格 = 0,完全等于 teacher。gradient 流通因为 `delta_z = g · BA(z)`,∂delta_z/∂g = BA(z) 非零,gate 能学;∂m_i/∂LoRA_b = LoRA_a(z) 非零,LoRA 能学。

### 1.4 参数量估算

- LoRA on adaLN_modulation: rank=8 × 12 blocks × (768 × 6 × 768) ≈ 600K
- gated low-rank residual: rank=8 × (512 × 768) ≈ 6K
- 总 adapter 参数: **~600K**(比 v4 ConditionMLP 1.58M 还小 60%)

---

## 2. 训练循环(step-level 切换,不 batch 内混合)

### 2.1 generic:SAFA = 12:1 step ratio

```
for step in range(total_steps):
    if step % 13 != 0:
        # generic main step (12/13)
        x_ffhq, prompt = sample_ffhq_batch()
        loss = L_native(x_ffhq, prompt, z=0)
              + beta * L_teacher(x_ffhq, prompt)
              + gamma * L_cond()
    else:
        # SAFA sparse step (1/13)
        x_0, z_0 = sample_safa_batch()
        x_hat = generate(x_0, z_0)        # 真实采样,过 E0
        loss_repr = L_repr(x_hat, z_0)
        g_repr = ∇ L_repr
        g_main = ∇(L_native + beta*L_teacher + gamma*L_cond)  # 重算 g_main
        g_repr_projected = PU_project(g_repr, g_main)         # 投影到正交补
        update_with(g_repr_projected)
```

### 2.2 generic main step 详细

```python
def generic_step():
    # 1. 采样 FFHQ batch + uniform sample prompt
    x_ffhq, prompt = ffhq_loader.sample()
    
    # 2. z 全关(g 路径 forward 但 g=0 → delta_z=0)
    # 注意:不传 z,直接走 c_native path
    c_native = t_embedder(t) + r_embedder(horizon) + null_embed
    
    # 3. forward student + teacher
    x_t = add_noise(x_ffhq, t, eps)
    v_student = generator(x_t, t, c_native)   # adapter LoRA + gated low-rank 都在
    with torch.no_grad():
        v_teacher = teacher_generator(x_t, t, c_native)   # stage1 frozen
    
    # 4. target = flow matching velocity target on x_ffhq
    v_target = velocity_target(x_ffhq, t, eps)
    
    # 5. losses
    L_native = mse(v_student, v_target)
    L_teacher = mse(v_student, v_teacher)
    L_cond = ||BA(z)||² + lambda_g * g²    # z 不注入但仍 regularize 防止 BA 漂移
                # 实际上 generic step z=0, BA(0)=0, 这一项=0
                # 但 g 仍然要 regularize 防漂移
    
    loss = L_native + beta * L_teacher + gamma * lambda_g * g²
    loss.backward()
    optimizer.step()
```

### 2.3 SAFA sparse step 详细

```python
def safa_step():
    # 1. 采样 SAFA batch
    x_0, z_0 = safa_loader.sample()
    
    # 2. 完整 forward(z 注入)
    c_native = t_embedder(t) + r_embedder(horizon) + null_embed
    delta_z = g * BA(z_0)           # g 可能非 0
    u = c_native + delta_z
    
    x_hat = generate(x_0, z_0)      # 真实采样(1 NFE MeanFlow)
    
    # 3. L_repr
    z_hat = E0(x_hat)
    L_repr = 1 - cosine(z_hat, z_0)
    
    # 4. PU 投影
    g_repr = torch.autograd.grad(L_repr, adapter_params)
    g_main = torch.autograd.grad(L_native_total + beta*L_teacher + gamma*L_cond, adapter_params)
    g_repr_proj = PU_project(g_repr, g_main)   # 投影到 g_main 正交补
    
    # 5. 应用 projected gradient
    apply_grads(adapter_params, g_repr_proj)
    optimizer.step()
```

---

## 3. Loss 函数详细

### 3.1 L_native(主损失,FFHQ 上)

```python
L_native = E_{x_ffhq, prompt, t, eps} || f_φ(x_t, t; c_prompt, z=0) - v_target(x_ffhq, t, eps) ||²
```

- `x_t = (1-t)·encode(x_ffhq) + t·eps`(latent space flow matching)
- `v_target = eps - encode(x_ffhq)`(velocity target)
- `c_prompt` 来自 16 条 prompt bank(均匀采样 + text encoder 编码)
- 注意: base MeanFlow-SiT 的 condition 不包含 text prompt!需要确认 base 模型是否接受 text prompt,或者 prompt bank 转 embedding 后注入哪个位置

**关键待确认**: MeanFlow-SiT 当前 condition 是 `(t, r, z_embed)` 三项加和,**没有 text prompt 接口**。需要决定:
- (a) Prompt bank 转 embedding 直接加到 condition vector(类似 z_embed)
- (b) 第一版不用 prompt,直接用 null_embed 当 c_native(简化版)

### 3.2 L_teacher(prior preservation)

```python
L_teacher = E_{x_ffhq, prompt, t} || f_φ(x_t, t; c_prompt, z=0) - f_teacher(x_t, t; c_prompt) ||²
```

- teacher = stage1 e15 best.pt frozen(永远不更新)
- student 在 z=0 时应该 = teacher
- **c_prompt 是 native prompt(来自 16 条 bank),不含 z**

### 3.3 L_cond(条件残差正则)

```python
L_cond = ||BA(z)||² + lambda_g * g²

# 可选 L_scale(让 scale 留在 native condition norm 的小比例内)
L_scale = relu(||g * BA(z)|| - tau)²    # tau = 0.1 * ||c_native||
```

- `gamma` 起点让 L_cond 占 L_native 的 1%-5%
- 不要 cosine margin(专家明确反对)

### 3.4 L_repr(只在 SAFA step)

```python
L_repr = 1 - cosine(E0(x_hat), z_0)
```

- x_hat = student 完整 forward(z 注入)生成的图
- E0 是 SAFA face emotion encoder(frozen)
- z_0 是 SAFA 配对数据的 ground truth

---

## 4. Generic Face Prompt Bank(16 条,专家原版)

只服务 generic main loop,均匀采样,**不做 per-sample caption reconstruction**。

1. `a high-quality close-up portrait photo of a young adult person, neutral expression, frontal view, soft daylight, plain background`
2. `a high-quality close-up portrait photo of a young adult person, slight smile, frontal view, soft daylight, plain background`
3. `a high-quality close-up portrait photo of a young adult person, serious expression, three-quarter view, studio lighting, plain background`
4. `a high-quality close-up portrait photo of a young adult person, calm expression, three-quarter view, indoor lighting, simple background`
5. `a high-quality close-up portrait photo of a middle-aged woman, neutral expression, frontal view, soft daylight, plain background`
6. `a high-quality close-up portrait photo of a middle-aged woman, smiling expression, frontal view, natural indoor lighting, plain background`
7. `a high-quality close-up portrait photo of a middle-aged woman, serious expression, three-quarter view, studio lighting, simple background`
8. `a high-quality close-up portrait photo of a middle-aged woman, calm expression, three-quarter view, window light, simple background`
9. `a high-quality close-up portrait photo of a middle-aged man, neutral expression, frontal view, soft daylight, plain background`
10. `a high-quality close-up portrait photo of a middle-aged man, slight smile, frontal view, indoor lighting, plain background`
11. `a high-quality close-up portrait photo of a middle-aged man, serious expression, three-quarter view, studio lighting, simple background`
12. `a high-quality close-up portrait photo of a middle-aged man, calm expression, three-quarter view, window light, simple background`
13. `a high-quality close-up portrait photo of an older adult person, neutral expression, frontal view, soft daylight, plain background`
14. `a high-quality close-up portrait photo of an older adult person, gentle smile, frontal view, indoor lighting, plain background`
15. `a high-quality close-up portrait photo of an older adult person, serious expression, three-quarter view, studio lighting, simple background`
16. `a high-quality close-up portrait photo of an older adult person, calm expression, three-quarter view, soft side lighting, simple background`

**禁止**: 姓名 / 纹身 / 疤痕 / 稀有饰品等身份细节。

---

## 5. 数据 pipeline

### 5.1 数据源选择

| 数据集 | 规模 | 用途 | 优先级 |
|--------|------|------|--------|
| **FFHQ** | 70K(60K train / 10K val NVIDIA 推荐) | generic main loop | **第一选择** |
| CelebAMask-HQ | 30K(512×512) | 备选 / 算力受限 fallback | 第二 |
| CASIA-WebFace | 500K / 10K 身份 | ✗ 不用(recognition-first,不适合生成 anchor) | 不选 |
| SAFA train_face_mixed_e14 | 看本地 jsonl 行数 | sparse repr loop | 必用 |

### 5.2 三阶段数据规模

| Phase | FFHQ | SAFA | ratio | 时长 |
|-------|------|------|-------|------|
| Phase 0 smoke | 5K | 1K | 12:1 | 1-2 天 |
| Phase 1 中等 | 20K | 2-4K | 12:1 | 3-5 天 |
| Phase 2 全量 | 60K(train split) | 全量 | 12:1(可调 8:1 或 16:1) | 5-10 天 |

---

## 6. 三阶段 rollout + 通过标准

### Phase 0: smoke test(1-2 天)

**目的**: 验证主梯度架构对不对,不是追求 SOTA

**配置**:
- FFHQ 5K + SAFA 1K + step ratio 12:1
- LoRA rank=8,gate init=0
- β(L_teacher)=1.0, γ(L_cond)=0.01(让 L_cond 占 1%-5%)
- 单 GPU 或 4 卡 DDP 都行

**通过标准**:
1. **generic held-out 质量不明显劣化**: FID/NIQE/Sharpness 相对 e15 teacher 退化 < 20%
2. **SAFA cosine 有可见提升**: ≥ 0.3(不要求一步到位,但不能不动)
3. **diversity 不像 e15_v8 自由落体**: LPIPS diversity ≥ 0.4

**Phase 0 失败 fallback**: 
- 如果 cosine 不动 → 检查 gate 是否学出来(g 应该非 0)
- 如果 FID 退化 > 50% → 加强 L_teacher(β 加到 5.0)
- 如果 diversity 塌 → 检查 SAFA step 是否真的稀疏(应该 1/13)

### Phase 1: 中等规模(3-5 天)

**目的**: 调超参,找最佳配置

**配置**:
- FFHQ 20K + SAFA 2-4K + 12:1
- 5-10 组超参 sweep:
  - β = {0.5, 1.0, 2.0, 5.0}
  - γ = {0.001, 0.01, 0.1}
  - LoRA rank = {4, 8, 16}
  - gate init = {0, 0.01, 0.1}

**通过标准**: cosine ≥ 0.7 + FID ≤ 80 + Sharpness ≥ 200

### Phase 2: 全量(5-10 天)

**配置**:
- FFHQ 60K train + 全量 SAFA
- 用 Phase 1 最佳超参
- ratio 可调: Phase 1 cosine 涨得慢 → 8:1;质量抖 → 16:1

**目标**: cosine ≥ 0.9 + FID ≤ 70 + Sharpness ≥ 250 + LPIPS ≥ 0.45

---

## 7. 实施时间线

| 阶段 | 工程量 | 累计 |
|------|--------|------|
| 数据 pipeline(FFHQ 下载 + 5K/20K/60K 索引) | 1-2 天 | 2 天 |
| 代码实现(LoRA + gated low-rank + 双数据 loader + step 切换 + 4 个 loss + PU 投影) | 3-4 天 | 6 天 |
| Phase 0 smoke + 通过标准验证 | 1-2 天 | 8 天 |
| Phase 1 中等规模 + 超参 sweep | 3-5 天 | 13 天 |
| Phase 2 全量 + 最终验证 | 5-10 天 | 23 天 |

**总: 18-23 天**(用户已确认工程量不是问题)

---

## 8. 关键代码改动点(4029 SAFA repo)

### 8.1 新文件

- `src/safa/models/peft_lora.py` (~200 行)
  - `LoRALayer` (rank-r low-rank adapter for nn.Linear)
  - `GatedLowRankResidual` (z injection: `g · BA(z)`)
  - `wrap_backbone_with_peft_lora(backbone, lora_rank, z_dim, hidden_size)`
- `data/index/generic_face_prompts.txt` (16 条 prompt)
- `data/index/ffhq_train_5k.jsonl` / `ffhq_train_20k.jsonl` / `ffhq_train_60k.jsonl`
- `configs/medium_v2/experiments/peft_lora_phase{0,1,2}_*.yaml`

### 8.2 修改文件

- `src/safa/training/peft_runner.py` 加:
  - `run_peft_lora_generic_step(runtime, batch, objective)` 
  - `run_peft_lora_safa_step(runtime, batch, objective)`
- `src/safa/training/g_loop.py` 加:
  - `_PEFT_LORA = "peft_lora"` 枚举
  - 双数据 loader 切换(step-level)
  - `step % 13` 调度逻辑

### 8.3 数据下载

- FFHQ 70K: 需要 NVIDIA 官方下载(可能要申请,或用镜像)
- FFHQ thumbnails 128×128 也行(Phase 0 验证概念用)
- 4029 上检查 `data/ffhq/` 是否已存在

---

## 9. 4 条核心规则(专家明确,不可动摇)

1. **主梯度来自通用数据(FFHQ),不是 SAFA 配对**
2. **`g_main = ∇(L_native + β·L_teacher + γ·L_cond)` 总梯度,不是只用 L_native**
3. **通用主循环不算 z,不用 E0(x_ffhq)**
4. **主循环不训 z 路径,z 分支只在 SAFA 稀疏循环训;很小、零初始化、带 gate**

---

## 10. 待澄清的剩余问题(0 个)

所有 12 条问题专家都明确回答了。剩余的工程细节(LoRA rank 数值 / βγ 起始值 / batch size)在 Phase 1 sweep 调。

---

## 11. 关联文档

- `/home/g203/safa-stage2-breakthrough-2026-07-06.md` — 突破方向分析(上会讨论版,本次手册的前身)
- `/home/g203/safa-peft-feasibility-2026-07-06.md` — 4 轮 PEFT 迭代 + 历史 PU 完整数据档案
- memory: `safa-stage2-leakage-breakthrough-july6` / `safa-fm-supervision-plan-june17` / `feedback-image-quality-priority`

---

## 12. 下一步(等用户决策)

1. **立刻进 Phase 0 实现**?(派 SubAgent 写代码 + 跑 smoke test)
2. **再等一轮专家 review**?(本文档给专家 review 后再实现)
3. **数据 pipeline 先行**?(派 SubAgent 下载 FFHQ + 建索引,并行写代码)
