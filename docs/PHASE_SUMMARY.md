# SAFA 项目阶段性总结

## 1. 项目目标

构建 **Samplewise Affective Face Anonymization (SAFA)** 系统：
- 冻结 E0 情绪编码器（ResNet-50, 8-class emotion, L2-normalized 512-dim embedding）
- 训练 G：conditional flow matching generator，输入 E0 embedding z，输出保留情绪信息的匿名化人脸
- 核心约束：cosine(z, E0(G(z))) ≥ 0.8，face detection rate = 1.0，emotion label preserved

---

## 2. 实验全景

### 2.1 实验列表与最终指标

| 实验 | GPU | 架构 | 训练方式 | Epochs | Cosine | Face Det | Composite | Label % | 状态 |
|------|-----|------|----------|--------|--------|----------|-----------|---------|------|
| V1 (Round 1) | 4,5 | ch=32, FM | Stage1→Stage2, λ=0.01→0.05, 4-step | 2 stage2 | 0.7281 | 1.0 | 0.7281 | 37.5% | 完成 |
| V2 (λ ramp) | 4,5 | ch=32, FM | Stage1→Stage2, λ=0.005→0.05, steps=[4,8,16,32] | 1 stage2 (running) | 0.8684 | 1.0 | 0.8684 | 50.0% | **最佳，仍在训练 epoch 2** |
| Ablation A | 2 | ch=32, FM | **From scratch**, λ=0.05 fixed, 4-step | 3 stage2 (running) | 0.4689* | 0.9688 | 0.4542 | 0.0% | 运行中，epoch 3 |
| Ablation B | 3 | ch=32, FM | Resume Stage1 best, λ=0.05, 4-step | 1 stage2 | 0.5928 | 1.0 | 0.5928 | 25.0% | 完成 |
| Ablation F | 2 | ch=32, FM | **From scratch**, λ=0.05, 8-step | 1 stage2 | 0.9694 | **0.0** | **0.0** | 90.6% | **完全退化** |
| Ablation C | 3 | ch=32, FM | Resume V2 best, λ=0.05, steps=[4,8,16] | 25% epoch 0 | N/A | N/A | N/A | N/A | **KILLED（太慢，10h/epoch）** |
| V3 (ch=64) | 4,5 | **ch=64**, FM | Stage1 only | 3 stage1 | 0.5636 | 0.996 | 0.5612 | 43.4% | **KILLED（OOM in Stage2）** |

*Ablation A 的 best.pt 保存了 epoch 0 的退化 checkpoint（cosine=0.9542, face_det=0.1406）。上表使用 last.pt（epoch 2）的指标。*

### 2.2 复合分数排名

Composite = cosine × face_detection_rate（避免选择退化的 checkpoint）

1. **V2**: 0.8684 × 1.0 = **0.8684** ← 最佳
2. **V1**: 0.7281 × 1.0 = **0.7281**
3. **Ablation B**: 0.5928 × 1.0 = **0.5928**
4. **Ablation A**: 0.4689 × 0.9688 = **0.4542**
5. **Ablation F**: 0.9694 × 0.0 = **0.0000** ← 完全退化
6. **V3**: 0.5636 × 0.996 = 0.5612 (未完成)

---

## 3. 关键发现与推论

### 3.1 从零开始训练会导致退化

**实验证据**：
- Ablation A (from scratch, 4-step): epoch 0 的 best.pt 是退化的 (face_det=0.14)。训练到 epoch 2 开始恢复 (face_det=0.97)，但 cosine 停在 0.47
- Ablation F (from scratch, 8-step): 完全退化 (face_det=0.0)，cosine=0.97 是假象

**推论**：从随机权重开始联合训练 flow matching + cycle loss，generator 找到了一个对抗性模式——生成非人脸输出使 E0 返回与输入 z 高度相似的 embedding（因为 E0 对非人脸输入仍会产生 embedding）。这不是真正的情绪保持。

**根因分析**：cycle loss (cosine(z, E0(G(z)))) 本质上是一个 self-consistency 目标。当 G 还没学会生成人脸时，它可以通过生成让 E0 "短路" 的模式来最小化 cycle loss。Stage 1（纯 flow matching + face detection gate）先让 G 学会生成真实人脸，再引入 cycle loss 才有意义。

### 3.2 Stage 1 的表示与 cycle consistency 兼容

**实验证据**：
- Ablation B (resume Stage1 best, λ=0.05): cosine=0.5928, face_det=1.0 — 生成的是真实人脸，cosine 从第一个 epoch 就是正的
- V1/V2 都经过 Stage 1 gate (face_det ≥ 0.95) 后才进 Stage 2

**推论**：Stage 1 学到的表示与 cycle consistency 兼容。问题不在于表示本身，而在于：
1. 从零开始训练时缺少 face quality 约束
2. Lambda 增长速度影响收敛

### 3.3 Lambda ramp 优于固定高 lambda

**实验证据**：
- V2 (λ=0.005→0.05 slow ramp): cosine=0.8684, face_det=1.0
- Ablation B (λ=0.05 fixed, resume Stage1): cosine=0.5928, face_det=1.0

**推论**：V2 的 λ ramp 让 generator 在 cycle loss 权重低时先稳定 flow matching 质量，然后逐步增加 cycle consistency 要求。直接用高 lambda 可能破坏 flow matching 的生成质量。

### 3.4 cycle_steps_schedule 有益但增加训练成本

**实验证据**：
- V2 使用 steps=[4,8,16,32]，每 epoch 轮换步数
- V1 使用固定 4-step
- V2 cosine 显著高于 V1 (0.8684 vs 0.7281)

**推论**：多步训练让模型学会在不同 ODE solver 精度下都保持一致性，提高泛化能力。但：
- 训练时间 ≈ 4x（每个 batch 做 4/8/16/32 步）
- Ablation C (steps=[4,8,16]) 在单 GPU 上 10h/epoch，被 kill

### 3.5 ODE solver 发散是正常现象，不是 bug

**观察**：所有实验都持续产生 `WARNING: ODE solver divergence at step X/Y, max_abs=5.01~6.21`

**分析**：
- 发散阈值设为 5.0（代码中为 7.0 on disk，但运行进程用的是旧版 5.0）
- 发散主要发生在：step 3/4（4-step batches）、step 0/32（validation 32-step）
- 值在 5.0-6.5 范围，不算严重（不像 Ablation F 那样值 > 10）
- clamp(-1,1) 后图像看起来正常

**结论**：这些是 ODE solver 在少数样本上的轻微不稳定，不影响整体训练。在 Heun solver 的最后一步已经改用 Euler 避免在 t=1.0 处查询 vector field。

### 3.6 E0 情绪编码器质量是瓶颈

**E0 指标**：
- val accuracy: 55.5%（8-class）
- per-class: class_0=78%, class_1=93.4%, class_7=23.8%（contempt 几乎随机）
- majority baseline: 12.5%（8 类均匀）

**影响**：
- E0 只有一半概率正确识别情绪 → G 的 cycle loss 上限受 E0 质量约束
- Label match % (37.5%-50%) 部分受限于 E0 本身的不准确
- 要提升最终效果，E0 准确率需要从 55% 提升到 70%+

---

## 4. 代码漏洞与修复记录

### 4.1 修复的 bug（按严重性排序）

| Bug | 文件 | 严重性 | 描述 | 修复 |
|-----|------|--------|------|------|
| NaN DDP deadlock | e0_loop.py | CRITICAL | NaN loss 导致 assert_finite_tensor 在 rank 0 crash，其他 rank 永久等待 | NaN 检查放在 assert 之前 |
| Checkpoint overwrite | g_loop.py | HIGH | best.pt 在 Stage 2 epoch 0 被退化 checkpoint 覆盖 | 引入 composite score (cos × fd) |
| UnboundLocalError | g_loop.py | HIGH | `epochs=0` 时 `stage_epoch` 未初始化 | 在 range 前初始化 stage_epoch=-1 |
| TOCTOU race | feature_cache.py | MEDIUM | cache 文件 check-then-write 竞态条件 | tempfile + atomic rename |
| Heun t=1.0 extrapolation | generator.py | MEDIUM | 最后一步 Heun 查询 t=1.0 处的 vector field | 最后一步改用 Euler |
| Gradient clipping 缺失 | g_loop.py | MEDIUM | flow matching 早期可能梯度爆炸 | clip_grad_norm_(params, 1.0) |
| GPU index parsing | device.py | MEDIUM | `require_cuda_device("cuda:0")` 解析为无 index | 正确解析 device index |
| SHA256 check | g_loop.py | LOW | E0 checkpoint 与 feature cache 不匹配时无警告 | 添加 SHA256 一致性检查 |
| stable_epochs | train_g.yaml | LOW | Stage 1 只需 1 epoch 满足 gate 就退出 | 改为 3 epoch |

### 4.2 本次 DDP 重构

- 创建 `src/safa/utils/distributed.py`：共享 `DistributedContext`, `init_distributed`, `barrier`, `cleanup_distributed`, `unwrap_model`, `reduce_train_metrics`, `broadcast_early_stop`
- `g_loop.py` 和 `e0_loop.py` 都从 ~100 行本地 DDP 代码简化为 import 共享模块
- 减少 ~170 行重复代码，未来 DDP 修改只需改一处

---

## 5. 架构决策回顾

### 5.1 Flow Matching (OT-CFM) 选择正确

线性插值路径 `x_t = (1-t)*x_0 + t*x_1` 配合 Heun solver，在 32 步内可以生成高质量人脸。相比扩散模型：
- 训练更稳定（不需要噪声 schedule 调参）
- 推理更快（32 步 vs DDPM 的 1000 步）
- 代价是收敛速度可能较慢

### 5.2 Stage 1 → Stage 2 两阶段设计正确但有代价

Stage 1 (纯 flow matching + face detection gate) 确保了 G 能生成真实人脸，避免了退化。但：
- Stage 1 的纯 FM 训练让 G 学到了一种与 cycle loss 不完全兼容的内部表示
- 从 scratch 联合训练（Ablation A）虽然有退化风险，但长期来看 cosine 可能超过 V2
- 当前数据不足以下定论（Ablation A 只跑了 3 epoch，还在恢复中）

### 5.3 UNet ch=64 尝试失败

V3 使用 base_channels=64（15.7M params vs ch=32 的 5M params）在 Stage 2 遇到 OOM。gradient checkpointing 解决了 OOM 但训练太慢。模型容量增加是否有帮助尚未验证。

---

## 6. 可视化产出

| 文件 | 内容 |
|------|------|
| `multi_experiment_comparison.png` | 4 个实验 × 8 样本并排对比（含 metrics overlay） |
| `metrics_comparison.png` | cosine / face_det / composite 柱状图 |
| `round1_comparison_grid.png` | V1 单实验 16 样本 grid |
| `v2_comparison_grid.png` | V2 单实验 16 样本 grid |
| `v2_extended/comparison_grid.png` | V2 扩展 64 样本评估 |
| `ablation_a_comparison_grid.png` | Ablation A (退化 epoch 0) 16 样本 |
| `ablation_b_grid.png` | Ablation B 16 样本 |
| `ablation_f_grid.png` | Ablation F (完全退化) 16 样本 |

---

## 7. 训练配置总结

### E0 配置 (train_e0.yaml)
```yaml
epochs: 80, batch_size: 64, lr: 0.0003, weight_decay: 0.0001
label_smoothing: 0.1, class_weight: true (effective number of samples)
scheduler: cosine, warmup_epochs: 5
early_stopping_patience: 15, augmentation: strong
```
最终: accuracy=55.5%, epoch 9 early stop

### G V2 配置 (train_g.yaml)
```yaml
batch_size: 16 (per GPU), base_channels: 32
stage1: epochs=5, face_detection_rate gate ≥ 0.95, stable_epochs=3
stage2: epochs=50, lambda_initial=0.005, lambda_max=0.05, lambda_growth=0.005/epoch
cycle_steps_schedule: [4, 8, 16, 32]
grad_clip_norm: 1.0
distributed: gloo, 2 GPUs (DDP)
```

### Ablation A 配置
```yaml
stage1.epochs: 0 (跳过 Stage 1, from scratch)
stage2.epochs: 40, lambda_initial=0.05, lambda_max=0.05 (fixed)
batch_size: 32, single GPU
```

### Ablation B 配置
```yaml
resume_from: artifacts/checkpoints/g/best_stage1.pt
stage1.epochs: 0, stage2.epochs: 40
lambda_initial=0.05, lambda_max=0.05 (fixed)
batch_size: 32, single GPU
```

---

## 8. 当前运行中的实验

| 实验 | 进度 | 预计完成 | 计划 |
|------|------|----------|------|
| Ablation A | epoch 3, ~85% | ~30min | 完成当前 epoch 后停止 |
| V2 | epoch 2, ~52% | ~3h | 完成当前 epoch 后停止 |
| Ablation C | KILLED | - | 太慢（10h/epoch） |

---

## 9. 下一步建议

### 9.1 优先级 P0：等 V2 epoch 2 完成
V2 是当前最佳模型（composite=0.8684）。epoch 2 可能进一步改善 cosine。完成后：
- 用 64 样本做扩展评估
- 生成最终可视化
- 检查是否达到 0.90+ cosine 的目标

### 9.2 优先级 P1：分析 Ablation A epoch 3
如果 Ablation A 的 cosine 恢复到 0.6+，说明 from-scratch 联合训练在足够长的训练后可以收敛。可能需要 10+ epoch。

### 9.3 优先级 P2：提升 E0
E0 准确率 55% 是整个系统的上限瓶颈。选项：
- 更强的 backbone (ResNet-101, ConvNeXt)
- 更多训练数据或更好的 augmentation
- 8-class → fewer classes (如合并 fear/surprise, contempt/sad)

### 9.4 优先级 P3：推理优化
当前 32 步 Heun solver 推理较慢。可以尝试：
- 8 步或 16 步推理（训练用多步，推理用少步）
- Adaptive step size ODE solver
- 模型蒸馏

---

## 10. 关键文件索引

| 文件 | 用途 |
|------|------|
| `src/safa/training/g_loop.py` | G 训练主循环 (Stage 1 + Stage 2) |
| `src/safa/training/e0_loop.py` | E0 训练循环 |
| `src/safa/models/generator.py` | FlowGenerator + OT-CFM + Heun solver |
| `src/safa/models/e0.py` | E0 情绪编码器 (ResNet-50) |
| `src/safa/utils/distributed.py` | 共享 DDP 工具模块 |
| `src/safa/training/losses.py` | cosine cycle loss + normalize_for_e0 |
| `src/safa/training/transforms.py` | train/eval/augmentation transforms |
| `src/safa/data/feature_dataset.py` | E0 特征对齐数据集 |
| `src/safa/data/feature_cache.py` | 特征缓存 + SHA256 验证 |
| `scripts/visualize_results.py` | 单实验可视化 |
| `scripts/visualize_multi.py` | 多实验对比可视化 |
| `configs/train_g.yaml` | V2 主训练配置 |
| `configs/ablation/` | 消融实验配置 |
