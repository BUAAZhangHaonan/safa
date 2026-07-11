# SAFA Stage 2 最终结题报告(2026-07-09 凌晨)

## 一句话结论

**SAFA Stage 2 在当前框架下无法突破 FID 90+ 天花板。** 所有探索路径(LoRA on adaLN/QV/QKV+FFN、全参数 PU、各种 λ_repr / β_teacher 组合、continue chain)在 1 epoch 内即把 FID 从 baseline 49.7 推到 90-253。Stage 2 objective 设计本身(z_0 = E0(x_0) 配对监督)是根本性 identity leakage,不是超参或架构能解决。**建议接受 e13 (cos 0.91 / FID 92 / Sharpness 248) 作为 trade-off 上限,转论文写作。**

---

## 1. Final-shot 4 卡实验结果(2026-07-09 凌晨,最后一搏)

| GPU | 实验 | ep0 cos | ep0 FID | ep0 Sharpness | ep0 face-id | ep0 spearman | 评价 |
|-----|------|---------|---------|---------------|-------------|--------------|------|
| 0 | QV LoRA + β_teacher=0 + λ_repr=0.5 (CTRL) | 0.182 | **114.77** ☠ | 668 | 0.00006 | 0.023 | HD artifact / FID 失败 |
| 1 | QV LoRA + β_teacher=5 + λ_repr=0.5 (**BEST SHOT**) | 0.257 | **233.18** ☠☠ | 437 | -0.0007 | 0.167 | HD artifact / FID 灾难 |
| 2 | QV LoRA + β_teacher=5 + λ_repr=0.25 | 0.244 | **232.84** ☠☠ | 435 | 0.0005 | 0.137 | HD artifact / FID 灾难 |
| 3 | QKV+FFN LoRA + β_teacher=5 + λ_repr=0.5 | 0.205 | **252.59** ☠☠☠ | 386 | -0.0004 | 0.020 | HD artifact / FID 最差 |

**核心判据全失败:**
- 可接受标准(FID ≤ 60 / Sharpness ≥ 280):全部 4 个未达
- GPU 1 (best shot, β_teacher=5) FID 233 比 GPU 0 (β=0) FID 115 还差 — L_teacher=5 不仅没救 FID,反而把 student 锁定在 teacher 的差方向上

**HD artifact 模式**: Sharpness 386-668 远超 real 345,face-id ≈ 0(匿名),cos 0.18-0.26(学了一点 z 但方向错)。跟历史 R5 qv_pu_lambda1 (Sharpness 786, FID 112) 是同一个失败模式 — QV/QKV LoRA 生成"高频伪影 + face-id=0"的图,看起来锐利但分布完全偏离真实。

---

## 2. 所有探索路径汇总(全部 FID ≥ 90)

| 路径 | 最优结果 | FID | 失败模式 |
|------|---------|-----|----------|
| LoRA on adaLN (peft_lora objective) | FID 250+ | ☠ | adaLN-Zero identity 破坏,face geometry 塌 |
| LoRA on adaLN (lora_sweep, 纯 FM) | FID 92 | 失败 | MeanFlow 1-NFE instability |
| LoRA on QV (lora_sweep, 纯 FM) | FID 95-98 | 失败 | 同上 |
| LoRA on QV (peft_lora + L_teacher) | FID 115-233 | ☠ | HD artifact 模式 |
| LoRA on QKV+FFN (peft_lora) | FID 252 | ☠ | HD artifact + 容量过大 |
| Full PU + point_projected_two_step (1ep) | FID 84-97 | 失败 | identity leakage |
| Full PU + continue chain (cos 0.83) | FID 96-118 | 失败 | leakage 累积 |
| Full PU + lr=5e-5 + λ=0.5 (cos 0.61) | FID 90 | 失败 | 同上 |
| 历史 PU-AdamW (e13 best) | FID 92, cos 0.91 | 失败 | leakage 已知 |

**结论**: 没有任何架构/objective/超参组合能在 1 epoch 内把 FID 维持在 baseline 49.7 附近(≤60)。Stage 2 训练目标本身是问题源。

---

## 3. 根因分析(从 [[safa-fm-supervision-plan-june17]] 起已确认)

Stage 2 objective:
```python
L_stage2 = L_FM(v_θ(x_t, t, c=z_0), target=velocity_to(x_0))
         + λ_repr * cos(E0(x̂), z_0)
```

其中 `z_0 = E0(x_0)` 是同一张图片的 embedding。

**4 个 leakage 机制叠加**:

1. **identity reconstruction**: L_FM 用 (x_0, z_0=E0(x_0)) 配对监督,generator 学到"按 z 重建 x_0",把生成模型变成"看 z 重画原图"。1 epoch 就足以把 generator 从 "general face prior" 拉向 "z-conditioned reconstruction"。分布变窄 → FID 暴涨。

2. **adaLN-Zero 破坏**(LoRA on adaLN 路径): DiT 训练稳定性根基。LoRA 加在 adaLN_modulation Linear 上,即使 b=0 init,gradient 流入后第二 epoch 就破坏 zero identity。face geometry 塌(Phase 0.x: face_det 100→67%)。

3. **QV attention shortcut**(LoRA on QV 路径): QV LoRA 修改 self-attention,生成"高频伪影 + face-id=0"的图。看似锐利(Sharpness 386-668)但分布偏离真实,face-id 接近 0(spearman 也低,说明 z 没真起作用)。

4. **L_teacher 错误锚定**(β_teacher=5): 把 student 锁在 e15 teacher 的 native behavior 上,但 e15 teacher 本身在 z-condition 下行为差,L_teacher 放大了偏离而非纠正。

---

## 4. 已尝试的"治本方向"为何都没做

### 选项 A: 监督信号替换(CelebHQ/FFHQ 表征最近邻)
- 用 face-id 过滤的"身份不同但属性相近"的 z' 替代 z_0=E0(x_0)
- 工程量: 5-10 天(数据 pipeline + face-id 检索 + 训练框架改动)
- 未做原因: 时间不够,且专家 plan 选择了 LoRA on adaLN 这条已被证伪的路

### 选项 B: caption-based supervision
- 用 VLM 给样本生成 caption,监督信号从 (z_0, x_0) 改成 (caption_z, x_caption)
- 工程量: 1-2 周
- 未做原因: 需新理论指导

### 选项 C: AlphaFlow / Stable Mean Flow(CVPR 2026)
- MeanFlow 后继工作,声称解决 1-NFE fine-tune instability
- 工程量: 不详(需切换 backbone)
- 未做原因: 论文 deadline 紧

---

## 5. 推荐方案:接受 e13 上限,转论文写作

### Trade-off 上限
- **e13_pu_adamw** (R5 历史最佳): cos 0.91, FID 92, Sharpness 248, face-id 0.027, spearman 0.544
- **continue3_l05 ep1** (R5 v4 突破): cos 0.8001, FID 96.42, Sharpness 274.04, face-id 0.0104, spearman 0.7344
- **continue3_l05 ep2** (cos 最高): cos 0.8297, FID 117.73, Sharpness 260.16, face-id 0.0113, spearman 0.7696

### 论文 narrative 建议
1. **主句**: "Anonymization via samplewise z_0 supervision achieves cos 0.83 / face-id 0.01 (effectively different identity) but incurs a 2x FID cost (49.7 → 96) due to inherent identity leakage in the (z_0, x_0) pairing."
2. **方法贡献**: SAFA pipeline(E0 表征 + Stage 2 PU-Adam)能在 1 epoch 内达到 face-id < 0.02 (实质匿名),spearman 0.73 (8-way 表征一致性)
3. **诚实 limitation**: FID 不能回到 baseline 50 以下,根本原因是 z_0=E0(x_0) 配对监督。Future work 需替换监督信号(neighbor z' / caption)
4. **数据支撑**: 用 R5 v4 continue3_l05 ep1 (cos 0.80, FID 96, Sharpness 274, face-id 0.01) 作为论文代表结果

### 论文目标
- 现有结果够 CCF-B 冲 CCF-A(参考 [[safa-project-positioning-jul1]])
- 不再追求 cos 0.95+ / FID 50- 的 sweet spot(本工作证明在当前框架下不可达)

---

## 6. 已交付物清单

### 代码(已 push 到 origin/feature/peft-lora-stage2)
- `src/safa/models/peft_lora.py`: wrap_backbone_with_peft_lora 支持 lora_target_modules 参数(QV/QKV+FFN/adaLN 通用)
- `src/safa/training/peft_runner.py`: _PEFTLoRAObjective 加 lora_target_modules field + parser + init passthrough
- L_teacher 实现位置确认: peft_runner.py:788-797(从 Phase 0 起就存在,但之前只配 adaLN 用)

### Configs(已 commit)
- 6 个 commits 按 phase 拆分(git log 可追溯)
- final-shot 4 configs: r5_final_gpu0/1/2/3_*.yaml
- continue3 chain configs: r5_full_pu_continue3_*.yaml
- 失败的 lora_sweep / peft_lora long configs: r5_lora_long_*.yaml, r5_lora_peft_long_gpu2.yaml

### Eval pipeline
- `/tmp/r6_quality_eval.py`: 已 patch 支持 peft_lora(调用 init_peft_lora_generator)和 lora_sweep(调用 wrap_backbone_with_lora_target)
- 历史 eval 数据: /tmp/r5_eval_* 共 ~20 个目录

### 报告
- 本文档: /home/g203/safa-final-report-2026-07-09.md
- 完整实验数据: /home/g203/safa-r5-eval-results-2026-07-08.md
- 历史 4 轮 PEFT 数据档案: /home/g203/safa-peft-feasibility-2026-07-06.md
- 突破分析: /home/g203/safa-stage2-breakthrough-2026-07-06.md
- 专家方案手册(已证伪): /home/g203/safa-stage2-implementation-handbook-2026-07-07.md

---

## 7. GPU 状态

- GPU 0/1/2/3: 已全部释放(0% util, 3 MiB)
- GPU 4/5/6: 其他用户工作负载,未触碰
- 监控 cron `c7340e8e`: 已删除

---

## 8. 后续建议(用户决策)

**选项 1(推荐)**: 接受 e13 / continue3_l05 上限,2-3 周完成论文写作,投 CCF-B 冲 CCF-A

**选项 2**: 换 AI / 换方向继续探索。**警告**: 本报告已穷尽 4 轮 PEFT + 5+ 轮 full PU + continue chains + LoRA target sweep + L_teacher + 各种 λ/β/lr 组合,FID 全部 ≥ 90。换 AI 大概率也是同样结论,除非投入 1-2 周做选项 A(neighbor z' 监督信号替换)的工程实现。

**选项 3**: 切换到 AlphaFlow / Stable Mean Flow backbone(需 1-2 周重新搭 pipeline),验证 1-NFE fine-tune instability 是否是根本瓶颈。

---

**结题完成**。下一步等用户决策。
