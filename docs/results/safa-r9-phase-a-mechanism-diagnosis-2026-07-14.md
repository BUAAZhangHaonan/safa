# SAFA R9 Phase A 机制诊断结果（2026-07-14）

## 结论

Phase A 已完成，正式 gate 结论为 `continue`。39 个 run（702 个 sample-run）全部完成，三次重复逐文件、逐指标和逐步诊断位级一致；所有诊断值有限，合同一致。三类各晋级一个 arm：

- `flow_map2_normalized_eta_0p125`
- `paper_eta_0p125`
- `paper_eta_0p25_disable_i2`

这个结论不能解释为三个 arm 已通过后续质量门。Phase A 的 E0、Edev 和视觉 severe 在正式恢复策略中是 `report_only` / `observation_only`，只用于类内排序。Phase A 的硬门只有位级确定性、诊断有限性和合同一致性。因此，gate 中 12 个候选的 `passed=true` 表示技术合同通过，不表示 E0 或视觉阈值通过。B、C、D 阶段仍使用数值、视觉、质量和隐私硬门。

最直接的机制结论是：增大 guidance 强度会继续提高 E0/Edev，但 paper-split 的结构崩坏也会快速增加。`paper_eta_0p375` 的 E0 达到 0.784581、Edev 增量达到 0.566004，同时 severe 已增至 13/18。因此，不能沿着高 E0 单指标继续增大 eta。

## 执行与确定性

- Campaign ID：`r9-report-only-formal-v2`
- Phase：`diagnose`
- 样本：9 个 R8 困难样本 + 9 个 matched native-E0 对照
- 逻辑 run：13 个 arm（含 native）x 3 次重复 = 39
- Sample-run：39 x 18 = 702
- 候选视觉 review：12 个候选 x 3 次重复 = 36，全部通过官方 review validator
- Visual review aggregate SHA-256：`c9092a78176d5dc3e67acad1f114e5a71877caef4e528d8cd4df6f0c94565b25`
- 三次重复的 PNG、cosine、loss history、route diagnostics 和逐步诊断完全一致
- 每个候选的三个 `run_sha256` 相同，且 `diagnostics_finite=true`
- 常量 guidance arm 的 algorithm/diagnostic NFE 为 8/9；单区间 ablation 为 7/8；matched native NFE 为 1

正式 resume 使用 tmux `r9-a-final-resume-v3`，pane PID 1506926，退出码 0。Driver 成功时没有 stdout，因此 `/tmp/r9-a-final-resume-v3.log` 为 0 字节。Resume 只物化 phase results 和 gate，没有启动新的 generation worker。

## 12 个候选结果

`severe` 写成 `总数（困难+对照）`。Edev 是相对 matched native 的增量。三次重复完全一致，表内数值是该 arm 的正式聚合观察值。

| Family | Arm | Severe | E0 mean | Delta Edev | Algorithm/diagnostic NFE | 类内结果 |
|---|---|---:|---:|---:|---:|---|
| flow-map2 | `flow_map2_normalized_eta_0p125` | 2（2+0） | 0.634158 | 0.377379 | 8/9 | 晋级 |
| flow-map2 | `flow_map2_normalized_eta_0p1875` | 3（2+1） | 0.681325 | 0.460180 | 8/9 | 未晋级 |
| flow-map2 | `flow_map2_normalized_eta_0p25` | 4（3+1） | 0.707722 | 0.509579 | 8/9 | 未晋级 |
| paper constant | `paper_eta_0p125` | 2（2+0） | 0.689248 | 0.433683 | 8/9 | 晋级 |
| paper constant | `paper_eta_0p1875` | 4（4+0） | 0.749222 | 0.529760 | 8/9 | 未晋级 |
| paper constant | `paper_eta_0p25` | 6（6+0） | 0.747876 | 0.489039 | 8/9 | 未晋级 |
| paper constant | `paper_eta_0p3125` | 9（7+2） | 0.775538 | 0.558448 | 8/9 | 未晋级 |
| paper constant | `paper_eta_0p375` | 13（9+4） | 0.784581 | 0.566004 | 8/9 | 未晋级 |
| paper constant | `paper_eta_0p5` | 16（9+7） | 0.731315 | 0.529803 | 8/9 | 未晋级 |
| interval ablation | `paper_eta_0p25_disable_i1` | 5（4+1） | 0.593718 | 0.423076 | 7/8 | 未晋级 |
| interval ablation | `paper_eta_0p25_disable_i2` | 4（4+0） | 0.556967 | 0.434994 | 7/8 | 晋级 |
| interval ablation | `paper_eta_0p25_disable_i3` | 4（4+0） | 0.532441 | 0.434574 | 7/8 | 未晋级 |

## 逐步诊断摘要

下表是每个 arm 的 repeat 0 在 18 个样本、三个区间上的均值。其余两次重复位级相同。`active` 是实际启用修正的区间行数；ablation 的关闭区间仍记录 residual 和 transport，但 correction、gradient 和 ratio 为 0。

| Arm | Active | E0 loss before -> after | Gradient norm | Velocity norm | Transport norm | Correction norm | Correction/transport | Grad-velocity cosine | Local semigroup residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `flow_map2_normalized_eta_0p125` | 54/54 | 0.888507 -> 0.614190 | 0.725917 | 67.588583 | 16.897146 | 2.112143 | 0.125000 | 0.010298 | 0.100655 |
| `flow_map2_normalized_eta_0p1875` | 54/54 | 0.900722 -> 0.589767 | 0.674065 | 67.850923 | 16.962731 | 3.180512 | 0.187500 | 0.000871 | 0.097855 |
| `flow_map2_normalized_eta_0p25` | 54/54 | 0.867626 -> 0.548659 | 0.676171 | 68.030247 | 17.007562 | 4.251890 | 0.250000 | -0.005669 | 0.097624 |
| `paper_eta_0p125` | 54/54 | 0.833678 -> 0.520562 | 0.494585 | 67.573773 | 16.893443 | 2.111680 | 0.125000 | 0.004227 | 0.099811 |
| `paper_eta_0p1875` | 54/54 | 0.822346 -> 0.493953 | 0.493384 | 67.615824 | 16.903956 | 3.169492 | 0.187500 | 0.001754 | 0.099325 |
| `paper_eta_0p25` | 54/54 | 0.806591 -> 0.483627 | 0.481603 | 67.658804 | 16.914701 | 4.228675 | 0.250000 | 0.002409 | 0.099599 |
| `paper_eta_0p3125` | 54/54 | 0.803964 -> 0.478182 | 0.487671 | 67.666951 | 16.916738 | 5.286481 | 0.312500 | 0.000728 | 0.099127 |
| `paper_eta_0p375` | 54/54 | 0.786286 -> 0.461181 | 0.472661 | 67.667182 | 16.916796 | 6.343798 | 0.375000 | -0.001913 | 0.098871 |
| `paper_eta_0p5` | 54/54 | 0.768285 -> 0.457841 | 0.471053 | 67.696454 | 16.924114 | 8.462057 | 0.500000 | -0.000585 | 0.098538 |
| `paper_eta_0p25_disable_i1` | 36/54 | 1.026697 -> 0.753201 | 0.298403 | 67.399666 | 16.849917 | 2.803482 | 0.166667 | 0.000504 | 0.103940 |
| `paper_eta_0p25_disable_i2` | 36/54 | 0.895287 -> 0.640563 | 0.410343 | 67.660409 | 16.915102 | 2.776956 | 0.166667 | 0.002675 | 0.099500 |
| `paper_eta_0p25_disable_i3` | 36/54 | 0.806591 -> 0.556115 | 0.387642 | 67.658804 | 16.914701 | 2.860817 | 0.166667 | 0.004475 | 0.099599 |

这些诊断支持三个判断：

1. `correction/transport` 与注册 eta 精确一致，说明实现没有靠裁剪、后处理或隐藏降级改变算法。
2. Eta 增大时 correction norm 近似线性增加，而 transport、velocity 和 local residual 基本不变。结构崩坏主要跟修正幅度上升一起出现，不像是基础 transport 本身失稳。
3. 关闭任一区间都会明显降低 E0。关闭 I2 和 I3 的 severe 都是 4；I2 仅凭更高的 Edev 增量（0.434994 对 0.434574）赢得类内排序。这只是机制探针，不表示 I2 ablation 已满足 B 阶段硬门。

## 类内排序

正式排序键是 severe、Edev、E0、arm ID；每类最多取一个，不补位。

- Flow-map2：`.125 > .1875 > .25`
- Paper constant：`.125 > .1875 > .25 > .3125 > .375 > .5`
- Interval ablation：`disable I2 > disable I3 > disable I1`

Phase A 给出的实际方向是“保留小 eta 的视觉安全候选，再到 64 样本 x 3 seeds 用硬门淘汰”，不是继续追求最高 E0。三个晋级 arm 都可能在 B 阶段因 E0、质量、重复 severe 或 ArcFace privacy 失败。

## Artifact 和合同摘要

| Artifact | File SHA-256 | 内部合同 SHA-256 |
|---|---|---|
| `campaign_runtime.json` | `5f4ac90f8dca4a5dc1c492e79389e30f85a2b5142893016a4a5e60996f08f8aa` | `048bfcba49a80070086baa366082aaf776e8d20e18ae367e761e16a474cef867` |
| `diagnose/automatic_evidence.json` | `a9784169a20191a7bf73979b66ba7996d3a0ff0f1374095c9730b8a8f324f7e0` | `385af1c62a52943ac58051a8fd8ef196496e613506429297cab1565ee23fa8d5` |
| `diagnose/phase_results.json` | `558b35cefdfdd9f321114bf21b2a71b4c372b31993306665fd049e7dd76d3d1f` | `54bc6ba437986aedc695016c53cb2a82d69c545f443ad4c068502dedab4c882f` |
| `diagnose/gate_contract.json` | `716a27f1f685d8e07b085643e9fb120e529d5c5d9a7f1bc31706ae30b932aa5b` | `748ff3a78157db3cce0c5161dc8a209d204a6f356fa19325b7e92a01e40d5cef` |

其他绑定摘要：

- Manifest SHA-256：`e41b68999939f3ca53a60cb4d7a2452f75c4cd103a8bd6ac7136a0a5f08a0aa3`
- Manifest contracts SHA-256：`19ef690a20f80463b700075721367ce1c24437204fbc2d95cd1a32f55bf18236`
- Run plan SHA-256：`bf1fec2a95fb8e173d77c3c8f025572f6114cddd1d4a1f7dafab80867e243bb4`
- Checkpoint SHA-256：`4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d`
- Evaluator evidence SHA-256：`060f85f1ce3912305f74b433af50f19de3f89a8c775684548c82804614406b59`
- Preflight closure seal contract SHA-256：`88d83d2f47c67ecc84405e5332b8436185f2e6b7a542b3e829d16e7b0f54aa28`
- Recovery policy contract SHA-256：`b82d3725658b04b7dd7c5cd2a3061be6bca37acfbfd797617001e0c4221a0d23`
- Gate verdict：`continue`，`failures=[]`

R8 contracts 和 artifacts 没有写入。R9 generation inventory 仍是 39 个 `generation_result.json` 和 1,350 张 generation-only PNG；原 inventory digest 为 `659b0bf85104ed9e19e8be7e3924ee149debdd732a842d4a0f99e0633ac6a14e`。本次另按明确可复现的定义重算：以 `diagnose` 为根，只取直接 `__repeat_` run 下的 `generated_images/*.png` 和 `native_images/*.png`，逐文件记录 `{relative_path,size_bytes,mtime_ns,sha256}`，按路径排序后用 compact、sorted-key JSON 编码；新审计 digest 为 `e6fcc17b6ba7f2a314f535d705dda6eb77d721a204fa1c2826bffb74b3760247`。新摘要不冒充原摘要的序列化。39 个 `session_history.jsonl` 合计仍只有 39 行，最新 `generation_result.json` 时间是 17:48:48，早于 18:23 的 phase materialization。Resume 后真实 Python generation worker 数为 0。

## 资源审计

- 长任务运行在 tmux，GPU 使用 0-3。
- 调度实际发放 16 个 slot，每卡 4 个。
- 单 slot VRAM claim：4,938,792,960 B；每卡另留 2 GiB。
- 单 slot RAM claim：3,691,032,576 B。
- 观察到的系统 RAM 峰值约 40%，低于 85% admission 和 90% kill 线。
- 没有 OOM、非有限值、合同不一致、自动降 batch、算法替换或静默重试。
- 本阶段生成所基于的代码提交：`5159d761d34dc60a9bc7bdce3cd1ef7c20dc9702`。

## 下一阶段边界

B 阶段只运行这三个候选和 matched native，固定 seeds 1337、2027、3407。A 阶段的 report-only 规则不得传播到 B/C/D。B 阶段仍必须使用 severe、FID、KID、NIQE、Sharpness、E0/Edev、ArcFace exact-one 和 10,000 次 paired cluster bootstrap privacy 的硬门；在门禁实现与逐 ID 聚合测试全部通过前，不启动 B。
