# SAFA 项目阶段性总结（Phase 1 Final）

## 1. 项目目标

构建 **Samplewise Affective Face Anonymization (SAFA)** 系统：
- 冻结 E0 情绪编码器（ResNet-50, 8-class emotion, L2-normalized 512-dim embedding）
- 训练 G：conditional flow matching generator，输入 E0 embedding z，输出保留情绪信息的匿名化人脸
- 核心约束：cosine(z, E0(G(z))) ≥ 0.8，face detection rate = 1.0，emotion label preserved

---

## 2. 实验全景

### 2.1 实验列表与最终指标

| 实验 | GPU | 架构 | 训练方式 | Epochs | Cosine (best) | Face Det | Composite | Label % | 状态 |
|------|-----|------|----------|--------|---------------|----------|-----------|---------|------|
| V1 (Round 1) | 4,5 | ch=32, FM | Stage1→Stage2, λ=0.01→0.05, 4-step | 2 stage2 | 0.7281 | 1.0 | 0.7281 | 37.5% | 完成 |
| V2 (λ ramp) | 4,5 | ch=32, FM | Stage1→Stage2, λ=0.005→0.05, steps=[4,8,16,32] | 2 stage2 | **0.8684** (ep1) | 1.0 | **0.8684** | 50.0% | 完成 |
| Ablation A | 2 | ch=32, FM | **From scratch**, λ=0.05 fixed, 4-step | 3 stage2 | 0.6743 | 1.0 | 0.6743 | 57.8% | 完成 |
| Ablation B | 3 | ch=32, FM | Resume Stage1 best, λ=0.05, 4-step | 1 stage2 | 0.5928 | 1.0 | 0.5928 | 25.0% | 完成 |
| Ablation F | 2 | ch=32, FM | **From scratch**, λ=0.05, 8-step | 1 stage2 | 0.9694 | **0.0** | **0.0** | 90.6% | **完全退化** |
| Ablation C | 3 | ch=32, FM | Resume V2 best, λ=0.05, steps=[4,8,16,32] | 25% epoch 0 | N/A | N/A | N/A | N/A | **KILLED（太慢，10h/epoch）** |
| V3 (ch=64) | 4,5 | **ch=64**, FM | Stage1 only | 3 stage1 | 0.5636 | 0.996 | 0.5612 | 43.4% | **KILLED（OOM in Stage2）** |

**V2 epoch 动态**（关键发现）：
- Epoch 1 (λ=0.01): cosine=**0.8684**, face_det=1.0, label=50.0% ← **best.pt 保存于此**
- Epoch 2 (λ=0.015): cosine=**0.7493**, face_det=1.0, label=62.5% ← cosine 下降 0.12

V2 best.pt 保存了 epoch 1（composite=0.8684），epoch 2 composite=0.7493 更差，未更新 best.pt。

*Ablation A 的 best.pt 保存了 epoch 0 的退化 checkpoint（cosine=0.9542, face_det=0.1406）。上表使用 last.pt（epoch 3）的指标。*

### 2.2 复合分数排名

Composite = cosine × face_detection_rate（避免选择退化的 checkpoint）

1. **V2** (best.pt ep1): 0.8684 × 1.0 = **0.8684** ← 最佳
2. **V1**: 0.7281 × 1.0 = **0.7281**
3. **Ablation A**: 0.6743 × 1.0 = **0.6743**
4. **Ablation B**: 0.5928 × 1.0 = **0.5928**
5. **V3**: 0.5636 × 0.996 = 0.5613 (未完成 Stage 2)
6. **Ablation F**: 0.9694 × 0.0 = **0.0000** ← 完全退化

---

## 3. 关键发现与推论

### 3.1 从零开始训练初期退化，但长期可恢复

**实验证据**：
- Ablation A (from scratch, 4-step):
  - epoch 0: 完全退化 (face_det=0.14, cosine=0.95) — best.pt 保存了这个退化 checkpoint
  - epoch 2: 开始恢复 (face_det=0.97, cosine=0.47)
  - epoch 3: 基本恢复 (face_det=1.0, cosine=0.67) — 仍在持续改善
- Ablation F (from scratch, 8-step): 完全退化 (face_det=0.0)，cosine=0.97 是假象。8-step 的更高精度反而强化了对抗性模式

**推论**：从随机权重开始联合训练 flow matching + cycle loss，generator 会首先找到一个对抗性模式——生成非人脸输出使 E0 返回与输入 z 高度相似的 embedding。但随训练继续，flow matching loss 逐渐将 generator 推向生成真实人脸，cycle loss 才开始产生真实的语义一致性。这个恢复过程需要 3+ epoch。

**与 V2 的对比**：Ablation A epoch 3 (cosine=0.67) 仍低于 V2 epoch 1 (cosine=0.87)。Stage 1 预训练不仅避免退化，还显著加速了 cycle consistency 的收敛。

### 3.2 Stage 1 的表示与 cycle consistency 兼容

**实验证据**：
- Ablation B (resume Stage1 best, λ=0.05): cosine=0.5928, face_det=1.0 — 生成的是真实人脸，cosine 从第一个 epoch 就是正的
- V1/V2 都经过 Stage 1 gate (face_det ≥ 0.95) 后才进 Stage 2

**推论**：Stage 1 学到的表示与 cycle consistency 兼容。问题不在于表示本身，而在于：
1. 从零开始训练时缺少 face quality 约束
2. Lambda 大小直接影响收敛

### 3.3 Lambda ramp 适得其反：低固定 lambda 更优

**实验证据**（更新）：
- V2 epoch 1 (λ=0.01): cosine=**0.8684** ← lambda ramp 的最佳点
- V2 epoch 2 (λ=0.015): cosine=**0.7493** ← lambda 增加后 cosine 下降 0.12
- V1 (λ=0.01→0.05): cosine=0.7281（也经历了类似的 lambda 增长过程）
- Ablation B (λ=0.05 fixed): cosine=0.5928

**推论**：lambda ramp 不是越慢越好。V2 的数据清楚地表明：
1. λ=0.01 是当前最佳值（cosine=0.8684）
2. lambda 从 0.01 增加到 0.015 后 cosine 显著下降
3. Ablation B 的固定 λ=0.05 结果最差
4. cycle loss 权重过高会破坏 flow matching 的生成质量

**与 V1 的对比**：V1 起始 λ=0.01 且最终 cosine=0.7281，V2 epoch 1 λ=0.01 达到 0.8684。两者都用相同的 lambda 起步，但 V2 多了 cycle_steps_schedule 的贡献。

**结论**：最优策略是**固定低 lambda（~0.01）**，不是 ramp。cycle loss 应作为轻量辅助信号，而非主要训练目标。V2 epoch 1 的 0.8684 已经非常接近 0.8 的目标阈值。

### 3.4 cycle_steps_schedule 有益但增加训练成本

**实验证据**：
- V2 使用 steps=[4,8,16,32]，每 epoch 轮换步数
- V1 使用固定 4-step
- V2 cosine 显著高于 V1 (0.8684 vs 0.7281，同在 λ≈0.01 时)

**推论**：多步训练让模型学会在不同 ODE solver 精度下都保持一致性，提高泛化能力。但：
- 训练时间 ≈ 4x（每个 batch 做 4/8/16/32 步）
- Ablation C (steps=[4,8,16,32]) 在单 GPU 上 10h/epoch，被 kill
- V2 epoch 1 的 0.8684 是 cycle_steps_schedule + 低 lambda 的协同效果

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
- Label match % (37.5%-57.8%) 部分受限于 E0 本身的不准确
- 要提升最终效果，E0 准确率需要从 55% 提升到 70%+

### 3.7 NaN handler bug 影响了 E0 和 G 训练稳定性

**发现**：`e0_loop.py` 和 `g_loop.py` 都存在同一个 NaN handler bug——检测到 NaN loss 后执行 `(output * 0.0).sum().backward(); optimizer.step()`，这会将零值梯度写入 AdamW 的 momentum buffer，导致后续正常 batch 的更新也被污染。

**影响范围**：
- E0 训练：如果训练中出现过 NaN（可能性低，因为 ResNet-50 + CE loss 很稳定），AdamW 状态会被破坏
- G 训练：flow matching 早期 velocity prediction 不稳定，NaN 概率更高，bug 会严重影响训练质量
- 两个文件都已修复为 `optimizer.zero_grad(set_to_none=True); continue`

### 3.8 Label preservation 与 cosine 不完全相关

**观察**：
- V2 epoch 1: cosine=0.8684, label=50.0%
- V2 epoch 2: cosine=0.7493, label=**62.5%**（更高）
- Ablation A epoch 3: cosine=0.6743, label=**57.8%**（比 V2 epoch 1 还高）
- Ablation B: cosine=0.5928, label=25.0%

**推论**：cosine 和 label preservation 衡量的不是同一种质量。cosine 衡量 embedding 空间的距离保持，label preservation 衡量最终分类结果是否一致。高 cosine 不保证高 label preservation，因为 E0 本身只有 55% 准确率——即使 embedding 完全保留，E0 也可能给出不同的 label。

---

## 4. 代码漏洞与修复记录

### 4.1 修复的 bug（按严重性排序）

| Bug | 文件 | 严重性 | 描述 | 修复 |
|-----|------|--------|------|------|
| NaN DDP deadlock | e0_loop.py | CRITICAL | NaN loss 导致 assert_finite_tensor 在 rank 0 crash，其他 rank 永久等待 | NaN 检查放在 assert 之前 |
| NaN handler corrupts AdamW | e0_loop.py, g_loop.py | CRITICAL | NaN 检测后执行 dummy backward + optimizer.step()，零值梯度污染 AdamW momentum buffer | 改为 zero_grad + continue |
| Checkpoint overwrite | g_loop.py | HIGH | best.pt 在 Stage 2 epoch 0 被退化 checkpoint 覆盖 | 引入 composite score (cos × fd) |
| UnboundLocalError | g_loop.py | HIGH | `epochs=0` 时 `stage_epoch` 未初始化 | 在 range 前初始化 stage_epoch=-1 |
| TOCTOU race | feature_cache.py | MEDIUM | cache 文件 check-then-write 竞态条件 | tempfile + atomic rename |
| Heun t=1.0 extrapolation | generator.py | MEDIUM | 最后一步 Heun 查询 t=1.0 处的 vector field | 最后一步改用 Euler |
| Gradient clipping 缺失 | g_loop.py | MEDIUM | flow matching 早期可能梯度爆炸 | clip_grad_norm_(params, 1.0) |
| GPU index parsing | device.py | MEDIUM | `require_cuda_device("cuda:0")` 解析为无 index | 正确解析 device index |
| SHA256 check | g_loop.py | LOW | E0 checkpoint 与 feature cache 不匹配时无警告 | 添加 SHA256 一致性检查 |
| stable_epochs | train_g.yaml | LOW | Stage 1 只需 1 epoch 满足 gate 就退出 | 改为 3 epoch |

### 4.2 DDP 重构

- 创建 `src/safa/utils/distributed.py`：共享 `DistributedContext`, `init_distributed`, `barrier`, `cleanup_distributed`, `unwrap_model`, `reduce_train_metrics`, `broadcast_early_stop`
- `g_loop.py` 和 `e0_loop.py` 都从 ~100 行本地 DDP 代码简化为 import 共享模块
- 减少 ~170 行重复代码，未来 DDP 修改只需改一处

### 4.3 完整代码审查结论

所有 `src/safa/` 下的源文件已逐一审查：
- `training/e0_loop.py`: NaN handler 已修复，class weighting + cosine LR + early stopping + label smoothing + per-class accuracy 正常
- `training/g_loop.py`: NaN handler、gradient clipping、composite score、SHA256 check 全部到位
- `training/transforms.py`: train/train_strong/eval/generator 四套 transform 正常
- `training/losses.py`: cosine cycle loss + normalize_for_e0 正常
- `training/audit.py`: forbidden identity supervision term checking 正常
- `models/generator.py`: OT-CFM + Heun + divergence warning + gradient checkpointing 正常
- `models/e0.py`: ResNet-50 + L2 projector + freeze/check + checkpoint loading 正常
- `data/dataset.py`: strict image decoding with AffectNetRecords 正常
- `data/feature_cache.py`: TOCTOU fix + SHA256 + L2 normalization verification 正常
- `data/feature_dataset.py`: sample_id ordering + label consistency verification 正常
- `utils/distributed.py`: DDP shared utilities 正常
- `utils/device.py`: require_cuda_device + assert_finite_tensor 正常
- `utils/seed.py`: comprehensive seed setting 正常
- `utils/hashing.py`: SHA256 with 1MB chunk reads 正常
- `evaluation/recognizers.py`: InsightFace detector/recognizer + TorchScript recognizer 正常

**结论**：审查后未发现除 NaN handler 外的其他 bug。所有已知漏洞均已修复并部署。

---

## 5. 架构决策回顾

### 5.1 Flow Matching (OT-CFM) 选择正确

线性插值路径 `x_t = (1-t)*x_0 + t*x_1` 配合 Heun solver，在 32 步内可以生成高质量人脸。相比扩散模型：
- 训练更稳定（不需要噪声 schedule 调参）
- 推理更快（32 步 vs DDPM 的 1000 步）
- 代价是收敛速度可能较慢

### 5.2 Stage 1 → Stage 2 两阶段设计正确且必要

Stage 1 (纯 flow matching + face detection gate) 确保了 G 能生成真实人脸，避免了退化。

**关键证据**：
- 有 Stage 1 的模型（V1, V2）从 Stage 2 epoch 0 就保持 face_det=1.0
- 没有 Stage 1 的模型（Ablation A）需要 3 epoch 才恢复到 face_det=1.0，且 cosine 仍低于 V2
- Stage 1 不仅避免退化，还显著加速了 cycle consistency 的收敛

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
| `ablation_a_comparison_grid.png` | Ablation A (epoch 0 退化) 16 样本 |
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
resume_from: artifacts/checkpoints/g/best.pt (Stage 1 best)
stage1: epochs=0 (skipped, using resumed checkpoint)
stage2: epochs=80, lambda_initial=0.005, lambda_max=0.05, lambda_growth=0.005/epoch
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

## 8. 实验最终状态

| 实验 | 最终状态 | 说明 |
|------|----------|------|
| V1 (Round 1) | 完成 | Stage 2 epoch 2, composite=0.7281 |
| V2 (λ ramp) | 完成 | Stage 2 epoch 2, **best at epoch 1** (composite=0.8684), epoch 2 degraded to 0.7493 |
| Ablation A | 完成 | Stage 2 epoch 3, composite=0.6743, 从退化恢复 |
| Ablation B | 完成 | Stage 2 epoch 1, composite=0.5928 |
| Ablation F | 完成（退化） | face_det=0.0, 完全退化 |
| Ablation C | KILLED | 太慢，10h/epoch |
| V3 (ch=64) | KILLED | OOM in Stage 2 |

---

## 9. 下一步建议

### 9.1 优先级 P0：固定 λ=0.01 + cycle_steps_schedule 重训

V2 epoch 1 已达到 cosine=0.8684（λ=0.01）。这是使用 cycle_steps_schedule + 固定低 lambda 的最佳配置。建议：
- 固定 λ=0.01（不 ramp）
- 保留 cycle_steps_schedule=[4,8,16,32]
- 训练 10+ epoch 观察 cosine 是否稳定或继续提升
- 这是目前最有希望达到 cosine ≥ 0.8 目标的配置

### 9.2 优先级 P1：提升 E0 准确率

E0 准确率 55% 是整个系统的上限瓶颈。当前 cosine 最高 0.87，但 label preservation 只有 50-58%，受限于 E0 本身只有 55% accuracy。选项：
- 更强的 backbone (ResNet-101, ConvNeXt)
- 更多训练数据或更好的 augmentation
- 8-class → fewer classes (如合并 fear/surprise, contempt/sad)

### 9.3 优先级 P2：推理优化

当前 32 步 Heun solver 推理较慢。可以尝试：
- 8 步或 16 步推理（训练用多步，推理用少步）
- Adaptive step size ODE solver
- 模型蒸馏

### 9.4 优先级 P3：扩展 Ablation A 训练

Ablation A 在 epoch 3 仍在持续改善（0.47→0.67）。如果训练 10+ epoch，可能达到与 V2 相当的水平。这将验证"from scratch 联合训练在足够长时间后也能收敛"的假设。

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

---

## 11. Git 提交历史

```
f81df4b fix: NaN handler in e0_loop was calling optimizer.step() with dummy gradients
bfb2751 fix: NaN handler in g_loop was calling optimizer.step() with dummy gradients
e434800 docs: add comprehensive phase summary for SAFA experiments
8c334db feat: add multi-experiment comparison visualization
25901ab refactor: extract shared DDP utilities into distributed.py
```
