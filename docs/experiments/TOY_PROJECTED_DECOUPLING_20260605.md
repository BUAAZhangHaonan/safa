# Toy FM/CL projected decoupling experiments, 2026-06-05

## 1. 目标

本轮 toy 实验用于验证 Stage 2 的优化框架，而不是验证图像生成质量。

核心问题是：

1. FM 作为主任务时，CL 更新能否在不破坏 FM 的前提下有效下降。
2. 投影解耦是不是过于保守，导致 CL 收敛慢。
3. 是否存在比固定加权、PCGrad、line-search 更快、更自适应的更新方式。

toy 任务使用同一个 2D flow-matching / representation-control 设置，重点看：

- `valid_fm_loss`
- `repr_cosine_mean`
- `conflict_fraction`
- credit / projection diagnostics

## 2. 已测试方法

| 方法 | 含义 | 是否手调权重 | 主要问题 |
| --- | --- | --- | --- |
| `weighted_sum` | `FM + lambda * CL` | 是 | 能拉高 cosine，但 lambda 敏感，会损 FM |
| `pcgrad` | 对称梯度手术 | 是 | 高 cosine 需要较大 lambda，FM loss 变差 |
| `projected_two_step` | FM step + hard projected CL step | 否 | 稳，但 cosine 卡在约 0.70 |
| `soft_margin_projected` | 带软余量的 projected update | 是 | 有改善空间，但引入 margin 超参 |
| `budgeted_cl_line_search` | full CL gradient + 实际回评 line-search | 否 | 太慢，几分钟内没有到第一个 eval |
| `descent_credit_projected` | 用 FM step 赚到的下降量作为 CL 一阶预算 | 否 | cosine 略好，但 FM 不如 scaled 稳 |
| `descent_credit_scaled` | 不投影方向，只按 FM credit 缩放 CL 步长 | 否 | FM 最稳，但 CL 更慢 |

## 3. 关键实验产物

| 实验 | 路径 |
| --- | --- |
| fast sweep | `artifacts/toy_fm_cl_projected/main_sweep_gpu0_fast_seed1337/summary.json` |
| descent-credit projected | `artifacts/toy_fm_cl_projected_descent_credit_20260605/delta45_descent_credit_projected_gpu6_bs8192_20260605/summary.json` |
| descent-credit scaled GPU5 | `artifacts/toy_fm_cl_projected_descent_scaled_20260605/delta45_descent_credit_scaled_gpu5_bs8192_20260605/summary.json` |
| descent-credit scaled GPU6 | `artifacts/toy_fm_cl_projected_descent_scaled_20260605/delta45_descent_credit_scaled_gpu6_bs8192_20260605/summary.json` |

相关代码提交：

- `387dd90 Add descent-credit projected toy baseline`
- `205100b Strengthen descent-credit toy diagnostics tests`
- `86b3c4f Add descent-credit scaled toy baseline`

## 4. 主要结果

### 4.1 fast sweep, delta=45

按 cosine 排序：

| 方法 | lambda / margin | cosine | FM loss | conflict |
| --- | ---: | ---: | ---: | ---: |
| `repr_only` | 0.01 / 0 | 1.0000 | 16.6905 | 0.000 |
| `pcgrad` | 1.0 / 0 | 0.8259 | 0.2142 | 1.000 |
| `pcgrad` | 0.3 / 0 | 0.7761 | 0.1276 | 1.000 |
| `pcgrad` | 0.1 / 0 | 0.7382 | 0.0973 | 1.000 |
| `weighted_sum` | 1.0 / 0 | 0.7300 | 0.1050 | 1.000 |
| `weighted_sum` | 0.3 / 0 | 0.7120 | 0.0738 | 0.988 |

按低 FM loss 且 cosine > 0.70 排序：

| 方法 | lambda / margin | cosine | FM loss | conflict |
| --- | ---: | ---: | ---: | ---: |
| `projected_two_step` | 1.0 / 0 | 0.7048 | 0.0659 | 0.483 |
| `soft_margin_projected` | 1.0 / 0.05 | 0.7095 | 0.0688 | 0.448 |
| `weighted_sum` | 0.03 / 0 | 0.7058 | 0.0698 | 0.549 |
| `projected_two_step` | 0.03 / 0 | 0.7040 | 0.0725 | 0.475 |
| `weighted_sum` | 0.3 / 0 | 0.7120 | 0.0738 | 0.988 |

结论：高 cosine 可以靠 `repr_only`、高 lambda、PCGrad 得到，但 FM 明显变差。低 FM loss 区间里，hard projected / soft projected 更稳。

### 4.2 descent-credit projected

最终结果，GPU6，step 5000：

| 指标 | 数值 |
| --- | ---: |
| `repr_cosine_mean` | 0.7109 |
| `valid_fm_loss` | 0.0909 |
| `repr_point_loss` | 0.2891 |
| `repr_relation_loss` | 0.00125 |
| `conflict_fraction` | 0.469 |
| `projected_repr_norm_ratio` | 0.9963 |
| `fm_descent_credit` | 1.508e-4 |
| `credit_budget_used_fraction` | 0.0167 |
| `net_fm_delta_after_two_step` | -1.143e-4 |

解读：

1. 该方法没有手动 lambda/margin。
2. CL step 几乎没有被大幅投影，`projected_repr_norm_ratio` 约 0.996。
3. FM credit 大部分没有被消耗，`credit_budget_used_fraction` 只有约 1.7%。
4. cosine 略高于 fast sweep 中的 hard projected，但 FM loss 也更高。

### 4.3 descent-credit scaled

最终结果，两张卡复现：

| 实验 | cosine | FM loss | conflict | credit scale | net FM delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU5 | 0.7057 | 0.0839 | 0.452 | 0.8691 | -1.202e-4 |
| GPU6 | 0.7055 | 0.0839 | 0.459 | 0.8721 | -1.219e-4 |

解读：

1. 两次复现很一致。
2. scaled 方法比 projected 更保护 FM。
3. scaled 方法的 cosine 更低，说明只缩放 CL 步长会让 CL 更慢。
4. 平均 `credit_scale` 约 0.87，说明该方法确实在约束 CL 步长，而不是退化成 full CL。

## 5. 当前判断

### 5.1 最快

`descent_credit_projected` 和 `descent_credit_scaled` 都比 `budgeted_cl_line_search` 快得多。

`budgeted_cl_line_search` 每步要多次真实回评 candidate，GPU 利用率高但 wall-clock 很慢，几分钟内没有到第一个 eval。它不适合作为主线。

### 5.2 最好

如果主指标是 cosine，当前最好仍是手调类方法：

- `pcgrad(lambda=1.0)`: cosine 0.8259, FM loss 0.2142
- `weighted_sum(lambda=1.0)`: cosine 0.7300, FM loss 0.1050

但这两者都不是当前目标，因为它们不满足“无手调、保护 FM”的要求。

在无手调方法里：

- `descent_credit_projected` 的 cosine 更高，0.7109。
- `descent_credit_scaled` 的 FM 更好，0.0839。

### 5.3 最自适应

当前最干净的自适应机制是 FM descent credit：

1. 先做 FM step。
2. 计算该 step 在当前 batch 上实际降低了多少 normalized FM loss。
3. 将这部分下降量作为 CL step 可用的一阶预算。

它比固定 lambda、PCGrad、soft margin 更符合 SAFA 的非对称目标。

## 6. 暴露出的问题

1. `descent_credit_projected` 并没有明显突破 hard projected 的 cosine 上限。它说明 credit 预算不是当前主要瓶颈。
2. `descent_credit_scaled` 更保护 FM，但进一步压慢了 CL。
3. `conflict_fraction` 仍在 0.45 左右。解耦算法可以控制 FM 一阶损害，但不能让 CL 监督本身变得更容易。
4. 当前 toy 中，FM 很快降到低 loss，CL cosine 很快到 0.70 附近后停滞。这和真实 SAFA 的现象一致：主要问题不是是否能防止 FM 被破坏，而是 CL 可达性和表示约束本身较难。

## 7. 下一步建议

推荐继续保留 `descent_credit_projected` 作为最小无手调主线候选。

备选是 `descent_credit_scaled`，用于需要更强 FM 保护的场景。

不建议继续投入 `budgeted_cl_line_search`，因为它太慢，且它解决的是“真实回评接受性”问题，不是当前 cosine 卡住的问题。

下一步更值得测的是真实 SAFA 中的小规模 smoke：

1. 用 `descent_credit_projected` 替代 M3 hard projected repr step。
2. 保持同一个 Stage1 checkpoint、同一 batch、同一 fixed16。
3. 只跑短程 5 到 10 epoch，先看 cosine 是否比原 M3 快。
4. 如果仍卡住，说明瓶颈在生成器/表征目标可达性，而不是 toy 级投影规则。

