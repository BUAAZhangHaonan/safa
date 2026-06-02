# MEDIUM V2 Interim Analysis

文档时间：2026-06-02T12:53:22+08:00。  
只读数据检查时间：2026-06-02T04:53:22Z。  
本文只整理已有 artifact 和汇总文档，没有启动、停止或修改任何训练 job。

## 结论

M3 projected update 仍然是下一步主线。原因有三点：

1. Stage1 和 null-FM 能给出可用但偏弱的生成先验，FID 还没有到最终发布质量。
2. M2 weighted-sum 没有同时解决质量和 latent cosine，epoch80 的 FID 明显变差，梯度冲突仍在。
3. CL-only 可以抬高 cosine，但 face rate 直接崩掉。它只能作为诊断，不是可用生成器训练方式。

## 实验背景

Stage1 是条件 flow-matching 先验训练。它主要看单脸稳定性和分布质量，不承担隐私结论。

M0 是 Stage2 基线。它从 Stage1 继续训练，加入 Stage2 目标后 latent cosine 上升，但图像分布质量没有同步变好。M0 只适合作为比较锚点。

M2 是 weighted-sum 版本。它把 flow-matching 和 representation/Gram relation 损失放在同一个加权目标里。这个实验想看 Gram relation 是否能改善表示几何，同时保住生成质量。

null-FM 是固定 null condition 的 FM-only probe。它用来检查没有条件信息时的生成先验强度。它不是严格从零开始的 null-FM，因为它从已有条件 Stage1 权重继续。

Point-only CL-only 和 Point+Gram CL-only 都把 FM 关掉，只训练表示目标。它们用于诊断 representation loss 的方向，不用于最终生成器训练。

Relation metrics 是验证集上的几何检查。这里记录点损失、off-diagonal Gram MAE/MSE、pairwise Pearson/Spearman。它们只说明特征几何关系，不是隐私指标。

M3 projected update 的核心是把 FM 更新和 representation 更新分开处理。在 representation 更新与 FM 方向冲突时，M3 用投影控制冲突。这个方法更贴合当前证据：weighted-sum 有冲突，CL-only 会破坏生成，直接加权不够稳。

## Stage1 结果

FID/KID 使用 3969 generated images 对 3969 real single-face validation images。NIQE 使用 512 generated images。

### long200_v4 quality series

| Epoch | FID | KID mean | KID std | NIQE mean | NIQE std |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 88.63896942138672 | 0.07938392460346222 | 0.009279227815568447 | 5.9737976509406 | 1.205115213818254 |
| 40 | 77.76925659179688 | 0.0619264580309391 | 0.008338884450495243 | 5.873527599569037 | 1.0356054430422246 |
| 60 | 72.20774841308594 | 0.06319732218980789 | 0.009427825920283794 | 6.172364075373091 | 1.2965566596924467 |
| 80 | 75.86714172363281 | 0.0637371763586998 | 0.009754758328199387 | 5.298254521417006 | 1.1004706817592862 |
| 100 | 85.77922058105469 | 0.07788431644439697 | 0.010812146589159966 | 4.63071785985953 | 0.8241131357009724 |
| 120 | 80.82064819335938 | 0.06868883222341537 | 0.00983670074492693 | 4.383053843429592 | 1.0612918559333553 |
| 140 | 64.225830078125 | 0.05397951975464821 | 0.007536903955042362 | 5.628161913783976 | 1.3516087347843182 |
| 160 | 89.18077850341797 | 0.08050089329481125 | 0.00973923783749342 | 4.906436352287958 | 0.9214128422976436 |
| 180 | 55.054893493652344 | 0.045338477939367294 | 0.0071295201778411865 | 6.162786119033053 | 1.3817374423637965 |
| 200 | 49.21614074707031 | 0.03554704040288925 | 0.006894022226333618 | 6.109232051007189 | 1.33176137794584 |

### long200_v4 final validation

| Metric | Value |
| --- | ---: |
| stage_epoch_1based | 200 |
| loss | 0.05814494377523661 |
| flow_loss_raw | 0.05814494377523661 |
| cycle_loss_raw | 0.0 |
| grad_norm | 0.07980940988858541 |
| validation_latent_cosine_mean | 0.6387721505016088 |
| validation_source_prediction_preserved | 0.4921875 |
| validation_single_face_eq1_rate | 0.998046875 |
| validation_face_detect_ge1_rate | 0.998046875 |
| validation_zero_face_rate | 0.001953125 |
| validation_multi_face_rate | 0.0 |

### long1000_continue quality series

| Epoch | FID | KID mean | KID std | NIQE mean | NIQE std |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 220 | 58.0914306640625 | 0.04677475243806839 | 0.007509959861636162 | 6.011440572683938 | 1.2392725810030756 |
| 240 | 67.54428100585938 | 0.054612282663583755 | 0.008440935052931309 | 4.873286446113866 | 1.0837409656983146 |
| 260 | 51.21743392944336 | 0.0401373989880085 | 0.0070038242265582085 | 7.397486525347456 | 1.5882240691897962 |
| 280 | 60.963809967041016 | 0.05217462405562401 | 0.009170569479465485 | 5.804594429009608 | 1.4355013171073496 |
| 300 | 66.92613983154297 | 0.05836018547415733 | 0.008267347700893879 | 5.908099011877438 | 1.1567420995187243 |
| 320 | 52.791038513183594 | 0.040357161313295364 | 0.00804689060896635 | 5.687790199889922 | 1.6168151671347255 |
| 340 | 57.523258209228516 | 0.049211643636226654 | 0.007879093289375305 | 6.313626262903672 | 1.608162711382338 |
| 360 | 60.05561065673828 | 0.0508994534611702 | 0.0075064767152071 | 5.522132173294199 | 1.2566435874969835 |
| 380 | 56.737735748291016 | 0.045413482934236526 | 0.007718169130384922 | 5.36126743619532 | 1.1834918257433937 |

Best collected FID for long1000_continue is epoch260, with FID 51.21743392944336. It did not beat long200_v4 epoch200 FID 49.21614074707031.

### long1000_continue latest validation

| Metric | Value |
| --- | ---: |
| stage_epoch_1based | 385 |
| loss | 0.05743709391355514 |
| flow_loss_raw | 0.05743709391355514 |
| cycle_loss_raw | 0.0 |
| grad_norm | 0.07477237764000892 |
| quality_raw_niqe_mean | 4.932585817712659 |
| quality_raw_niqe_std | 0.9055755718341233 |
| validation_latent_cosine_mean | 0.6352042593061924 |
| validation_source_prediction_preserved | 0.4921875 |
| validation_single_face_eq1_rate | 0.994140625 |
| validation_face_detect_ge1_rate | 0.994140625 |
| validation_zero_face_rate | 0.005859375 |
| validation_multi_face_rate | 0.0 |
| FID/KID at epoch385 | unavailable |

## M0 和旧 V2 best

### M0 epoch100

| Metric | Value |
| --- | ---: |
| stage_epoch_1based | 100 |
| loss | 0.058011283247172835 |
| flow_loss_raw | 0.05787898943275213 |
| cycle_loss_raw | 0.013229383102804422 |
| grad_norm | 0.08047994378805161 |
| raw latent cosine mean | 0.9238038249313831 |
| raw source prediction preserved | 0.859375 |
| raw single_face_eq1 rate | 1.0 |
| raw face_detect_ge1 rate | 1.0 |
| raw zero_face rate | 0.0 |
| raw multi_face rate | 0.0 |
| EMA latent cosine mean | 0.9226631131023169 |
| EMA source prediction preserved | 0.85546875 |
| EMA single_face_eq1 rate | 1.0 |
| NIQE mean | 7.167329834326688 |
| NIQE std | 1.703940598670236 |
| FID | 126.25408172607422 |
| KID mean | 0.11806682497262955 |
| KID std | 0.019659586250782013 |

M0 keeps face rate and utility, but FID 126.25408172607422 is much worse than Stage1 long200_v4 epoch200 FID 49.21614074707031. It is a baseline, not the main method.

### old V2 best and single-face recheck

| Source | Metric | Value |
| --- | --- | ---: |
| Stage2 best checkpoint | stage_epoch | 2 |
| Stage2 best checkpoint | loss | 0.0646579628 |
| Stage2 best checkpoint | flow_matching_mse | 0.0644469934 |
| Stage2 best checkpoint | cycle | 0.0210969492 |
| Stage2 best checkpoint | grad_norm | 0.1624730787 |
| Stage2 best checkpoint | val latent cosine | 0.9740784373 |
| Stage2 best checkpoint | val source preserved | 0.916015625 |
| Stage2 best checkpoint | val face det | 1.0 |
| Basic eval | samples | 4000 |
| Basic eval | latent_cosine.mean | 0.9682712732 |
| Basic eval | latent_cosine.p10 | 0.9517689705 |
| Basic eval | source_prediction_preserved.mean | 0.91425 |
| Basic eval | label_accuracy_generated.mean | 0.54 |
| Single-face recheck | generated image count | 3969 |
| Single-face recheck | latent_cosine.mean | 0.9682674512753305 |
| Single-face recheck | latent_cosine.p10 | 0.9517428636550903 |
| Single-face recheck | source_prediction_preserved.mean | 0.9135802469135802 |
| Single-face recheck | label_accuracy_generated.mean | 0.5399344923154447 |
| Single-face recheck | face_detect_ge1_rate | 1.0 |
| Single-face recheck | single_face_eq1_rate | 1.0 |
| Single-face recheck | zero_face_rate | 0.0 |
| Single-face recheck | multi_face_rate | 0.0 |
| Single-face recheck quality | FID | 144.543212890625 |
| Single-face recheck quality | KID mean | 0.13880713284015656 |
| Single-face recheck quality | KID std | 0.012893247418105602 |
| Single-face recheck quality | NIQE mean | 5.422948851819098 |
| Single-face recheck quality | NIQE std | 1.0067968732400285 |
| Privacy status | privacy_guard_pass | true |
| Privacy status | privacy_skipped | true |
| Privacy status | skip_reason | privacy_protocol_blocker |

旧 V2 best 的 utility 高，但 quality 很弱。single-face recheck 通过 generated-image guard，不等于 formal privacy pass。ArcFace 在 source-side privacy recognizer stage 检到 2 张脸，所以 privacy metrics 没有完成。

## M2 gram-weighted

M2 training had been manually stopped based on the operational snapshot, but no preserved tmux snapshot is included in this document. Artifacts show last metric epoch88: `last_metrics.json` mtime 是 2026-06-01 18:51:06 +0800。monitor status/report 文件可能是 stale/live pointers，不能单独证明最终 session 状态。

| Point | loss | flow_loss_raw | repr_point_loss | repr_relation_loss | repr_loss | val latent cosine | val source preserved | single_face_eq1 | NIQE | FID | KID mean | KID std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epoch20, best collected FID | 0.059213842205703256 | 0.058989731176197525 | 0.017108474335446953 | 0.00530262389825657 | 0.02241109821945429 | 0.9226016290485859 | 0.85546875 | 0.998046875 | 6.755033784026385 | 84.60601043701172 | 0.07090625911951065 | 0.012409738264977932 |
| epoch40 | 0.05955418538153172 | 0.05936947254985571 | 0.014346947176009416 | 0.004124336596531794 | 0.018471283777058124 | 0.9110979028046131 | 0.83984375 | 1.0 | 6.450528413413472 | 102.98348999023438 | 0.08712911605834961 | 0.014585566706955433 |
| epoch60 | 0.05926766955703497 | 0.05912686118334532 | 0.011454407953470946 | 0.0026264279290568082 | 0.01408083588965237 | 0.8761210441589355 | 0.77734375 | 1.0 | 5.67648759026854 | 86.63194274902344 | 0.06540318578481674 | 0.007656523026525974 |
| epoch80 | 0.05838578795343637 | 0.05820593472123146 | 0.01394630338959396 | 0.004039020881569013 | 0.017985324261710046 | 0.9806453622877598 | 0.955078125 | 1.0 | 7.460613698503903 | 191.86097717285156 | 0.22683677077293396 | 0.023960590362548828 |
| epoch88 last metric | 0.058988171719014645 | 0.05883315653204918 | 0.01233259941637516 | 0.0031689265507273377 | 0.01550152594782412 | 0.9064574800431728 | 0.8359375 | 1.0 | 6.913431037916532 | unavailable | unavailable | unavailable |

M2 did not solve quality and cosine together. Epoch80 cosine is high at 0.9806453622877598, but FID is 191.86097717285156. Epoch88 cosine falls to 0.9064574800431728. The latest gradient diagnostics also show conflict: `gradient_conflict_fraction=0.453125`, `gradient_cosine_fm_repr=0.0068998365447200186`, and `weighted_repr_to_fm_ratio=0.10835632761291356`.

## null-FM epoch120

| Metric | Value |
| --- | ---: |
| stage_epoch_1based | 120 |
| loss | 0.05961719523966313 |
| flow_loss_raw | 0.05961719523966313 |
| cycle_loss_raw | 0.0 |
| repr_loss | 0.0 |
| grad_norm | 0.10415833313465118 |
| validation_latent_cosine_mean | 0.10446457890793681 |
| validation_source_prediction_preserved | 0.140625 |
| validation_single_face_eq1_rate | 1.0 |
| validation_face_detect_ge1_rate | 1.0 |
| validation_zero_face_rate | 0.0 |
| validation_multi_face_rate | 0.0 |
| EMA validation_latent_cosine_mean | 0.1060156364692375 |
| EMA validation_source_prediction_preserved | 0.162109375 |
| NIQE mean | 5.944306197968555 |
| NIQE std | 0.9479649005228142 |
| FID | 80.23278045654297 |
| KID mean | 0.07296038419008255 |
| KID std | 0.009125943295657635 |

null-FM 的单脸率是 1.0，latent cosine 很低。这符合 null condition 会丢掉条件信息的预期。FID 80.23278045654297 仍不足以作为最终 prior。这个 run 也不是严格 from-scratch null-FM，因为它从条件 Stage1 权重继续。

## CL-only ablation

这两个 run 只说明 representation loss 的方向。它们不是可用的 generator training mode，因为生成端 face rate 已经崩掉。

注意：下面 Point-only 和 Point+Gram 的 metric 表是 2026-06-02 12:46:44 CST (UTC+08) 采集的 snapshot。live `last_metrics.json` 和 relation summary 文件之后可能继续前进，所以表中数值不保证和当前 live path 完全一致。

### Point-only CL-only epoch12 snapshot

Point-only 的 `last_metrics.json` 是 live pointer。此表固定为采集时的 epoch12 snapshot；NIQE 对应 epoch-specific quality file `artifacts/eval/g_medium_v2_stage2_point_only_cl_only/quality/epoch_0012/stage2_epoch_0012_raw_niqe.json`。

| Metric | Value |
| --- | ---: |
| stage_epoch_1based | 12 |
| loss | 0.007577866327017546 |
| flow_loss_raw | 0.0 |
| cycle_loss_raw | 0.0 |
| repr_loss | 0.007577866327017546 |
| repr_point_loss | 0.007577866327017546 |
| repr_relation_loss | 0.0008530394764151425 |
| grad_norm | 0.061241768193244936 |
| validation_latent_cosine_mean | 0.945625577121973 |
| validation_source_prediction_preserved | 0.8828125 |
| validation_single_face_eq1_rate | 0.0 |
| validation_face_detect_ge1_rate | 0.0 |
| validation_zero_face_rate | 1.0 |
| validation_multi_face_rate | 0.0 |
| EMA validation_latent_cosine_mean | 0.935131648555398 |
| EMA validation_source_prediction_preserved | 0.865234375 |
| NIQE mean | 10.17065530155901 |
| NIQE std | 1.7787748747866814 |
| FID/KID | unavailable |

### Point+Gram CL-only epoch16

Point+Gram 的 `last_metrics.json` 也是 live pointer。此表固定为采集时的 epoch16 snapshot；NIQE 对应 epoch-specific quality file `artifacts/eval/g_medium_v2_stage2_point_gram_cl_only/quality/epoch_0016/stage2_epoch_0016_raw_niqe.json`。

| Metric | Value |
| --- | ---: |
| stage_epoch_1based | 16 |
| loss | 0.010166356929019094 |
| flow_loss_raw | 0.0 |
| cycle_loss_raw | 0.0 |
| repr_loss | 0.010166356929019094 |
| repr_point_loss | 0.009250228849053382 |
| repr_relation_loss | 0.0009161280749831348 |
| grad_norm | 0.10481874302625656 |
| validation_latent_cosine_mean | 0.8527187872678041 |
| validation_source_prediction_preserved | 0.712890625 |
| validation_single_face_eq1_rate | 0.0 |
| validation_face_detect_ge1_rate | 0.0 |
| validation_zero_face_rate | 1.0 |
| validation_multi_face_rate | 0.0 |
| EMA validation_latent_cosine_mean | 0.8664827272295952 |
| EMA validation_source_prediction_preserved | 0.732421875 |
| NIQE mean | 18.100111318552372 |
| NIQE std | 7.818117630514779 |
| FID/KID | unavailable |

### Validation relation metrics

Relation metric table 是 2026-06-02 12:46:44 CST (UTC+08) 写入本文的 captured snapshot，不是 live summary state。Point-only 行保留 first captured epoch12 summary，summary created_at 是 2026-06-02T04:33:51Z；同一个 live summary path 后来已经前进到 epoch13。Point+Gram 行是 epoch16 summary，created_at 是 2026-06-02T04:33:50Z。两者都使用 `sample_count=512.0`。

| Run | Variant | repr_point_loss | repr_relation_loss | offdiag_gram_mae | offdiag_gram_mse | pairwise_pearson | pairwise_spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Point-only snapshot epoch12 | raw | 0.05437569891608954 | 0.027412507510142667 | 0.11997485410016771 | 0.027412507510142667 | 0.9389362005475402 | 0.9356569921625505 |
| Point-only snapshot epoch12 | EMA | 0.06486598720832965 | 0.03180814061995583 | 0.13157931520682672 | 0.03180814061995583 | 0.9281059284409674 | 0.9234335481149079 |
| Point+Gram snapshot epoch16 | raw | 0.14728770012389478 | 0.05806031405358421 | 0.18977813837254368 | 0.05806031405358421 | 0.9102106091085927 | 0.9148816969651131 |
| Point+Gram snapshot epoch16 | EMA | 0.13350961247703336 | 0.051077603723501806 | 0.17470303276246218 | 0.051077603723501806 | 0.9136417281408618 | 0.917070384817023 |

在 2026-06-02 12:46:44 CST (UTC+08) 的 snapshot 中，P+G 没有超过 P。它在 point metrics 和 relation metrics 上都更差：raw repr_point_loss 更高，offdiag Gram 误差更高，pairwise correlation 更低。所以现在不能写 Gram 已经有帮助。

### 2026-06-03 pre-stop P/P+G snapshot

这次只读检查没有停止训练、没有停止 watcher，也没有启动 M3。latest `last_metrics.json` 的落盘时间是 2026-06-03 01:11:05 CST for Point-only 和 2026-06-03 01:40:18 CST for Point+Gram。relation summary 的落盘时间是 2026-06-03 01:15:08 CST 和 2026-06-03 01:50:49 CST。

| Metric | Point-only CL-only | Point+Gram CL-only |
| --- | ---: | ---: |
| stage_epoch_1based | 22 | 28 |
| loss | 0.006282156319543719 | 0.0070958162331953645 |
| repr_point_loss | 0.006282156319543719 | 0.006655988897569478 |
| repr_relation_loss | 0.0004995158842764795 | 0.000439827333856374 |
| grad_norm | 0.033481443786621094 | 0.0502570367872715 |
| raw latent cosine mean | 0.9536038227379322 | 0.894861675798893 |
| raw source prediction preserved | 0.8984375 | 0.76171875 |
| raw single_face_eq1 rate | 0.0 | 0.001953125 |
| raw face_detect_ge1 rate | 0.0 | 0.001953125 |
| raw zero_face rate | 1.0 | 0.998046875 |
| raw multi_face rate | 0.0 | 0.0 |
| EMA latent cosine mean | 0.9507150780409575 | 0.8854449465870857 |
| EMA source prediction preserved | 0.884765625 | 0.736328125 |
| EMA single_face_eq1 rate | 0.0 | 0.0 |
| EMA face_detect_ge1 rate | 0.0 | 0.0 |
| NIQE mean | 9.12577004070587 | 10.822569135586313 |
| NIQE std | 1.0474253714863426 | 3.098093102891729 |
| FID/KID | missing | missing |

Latest validation relation metrics still favor Point-only. Both summaries use `sample_count=512.0`.

| Run | Variant | repr_point_loss | repr_relation_loss | offdiag_gram_mae | offdiag_gram_mse | pairwise_pearson | pairwise_spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Point-only epoch22 | raw | 0.0463919224129043 | 0.022804933895805086 | 0.11016760002496145 | 0.022804933895805086 | 0.9450920872419294 | 0.9409817234161033 |
| Point-only epoch22 | EMA | 0.04928911969992582 | 0.024106661564289776 | 0.11363663295576282 | 0.024106661564289776 | 0.9424966109763318 | 0.9382520285892924 |
| Point+Gram epoch28 | raw | 0.10513666144989699 | 0.040115730819789155 | 0.15543402880915425 | 0.040115730819789155 | 0.9303443795852158 | 0.9343045251284067 |
| Point+Gram epoch28 | EMA | 0.11455198297192151 | 0.04207123885029448 | 0.15717286437673514 | 0.04207123885029448 | 0.926595788475514 | 0.9305572203646052 |

Point-only is better than Point+Gram in this pre-stop snapshot. It has lower loss, better raw latent cosine, better raw source prediction preservation, lower NIQE, lower validation point loss, lower off-diagonal Gram error, and higher pairwise correlations. Point+Gram has a smaller training-time `repr_relation_loss`, but its validation relation summary is worse, so that does not prove a useful Gram gain.

Gram has not proved an O(B^2) acceleration. The Gram term adds O(B^2) pairwise constraints inside a batch, but the current evidence does not show faster convergence or better validation geometry than Point-only.

CL-only remains a convergence diagnostic only. Point-only has raw and EMA face rates at 0.0. Point+Gram has raw `single_face_eq1_rate=0.001953125`, which is only 1/512, and EMA face rate is 0.0. This is still a face-generation collapse, so these runs cannot be used as generator-quality evidence.

Pair visualizations were backfilled on 2026-06-03 02:00 CST with `CUDA_VISIBLE_DEVICES=1` and a one-shot wrapper around `scripts/watch_medium_v2_checkpoint_visuals.py`; no watcher or M3 job was started. The latest exact checkpoint-pair paths are:

| Run | Latest pair visualization | Manifest |
| --- | --- | --- |
| Point-only CL-only epoch22 | `artifacts/plots/medium_v2/stage2_point_only_cl_only/epoch_0022_checkpoint_pairs.png` | `artifacts/plots/medium_v2/stage2_point_only_cl_only/epoch_0022_checkpoint_pairs_manifest.json` |
| Point+Gram CL-only epoch28 | `artifacts/plots/medium_v2/stage2_point_gram_cl_only/epoch_0028_checkpoint_pairs.png` | `artifacts/plots/medium_v2/stage2_point_gram_cl_only/epoch_0028_checkpoint_pairs_manifest.json` |

Backfilled visualization coverage is now complete up to those latest landed metrics: Point-only has checkpoint-pair PNGs for epochs 1-22, and Point+Gram has checkpoint-pair PNGs for epochs 1-28. Because per-epoch checkpoint files were not present for the lagging epochs, Point-only epochs 19-21 use the latest epoch22 checkpoint, and Point+Gram epochs 22-27 use the latest epoch28 checkpoint. The manifests record `backfilled_from_latest_checkpoint=true` for those rows.

## M0 epoch100 ad-hoc privacy probe

这个 probe 只能作为粗略信号。它不是 formal privacy pass。metadata 必须保留：`ad_hoc_ignore_guard=true`，`not_formal_privacy_pass=true`，`stage_epoch_1based=100`，`generated_image_count=512`，`num_pairs=512`。

| Recognizer | AUC | EER | TAR@FAR=1e-3 | TAR@FAR=1e-4 |
| --- | ---: | ---: | ---: | ---: |
| adaface | 0.5686988830566406 | 0.453125 | 0.0 | 0.0 |
| arcface | 0.5475692749023438 | 0.46875 | 0.001953125 | 0.001953125 |
| facenet | 0.6074790954589844 | 0.43359375 | 0.009765625 | 0.009765625 |

| Guard metric | Value |
| --- | ---: |
| face_detect_ge1_rate | 1.0 |
| single_face_eq1_rate | 1.0 |
| zero_face_rate | 0.0 |
| multi_face_rate | 0.0 |
| latent_cosine_mean | 0.9238019218901172 |

这些数不能写成 privacy 通过。正式 privacy 仍然没有通过或不可用。

## 主要分析

Stage1/FMs 仍然弱。long200_v4 最好 FID 是 49.21614074707031，long1000_continue 最好 FID 是 51.21743392944336。继续 Stage1 没有明显突破。5M 级小 prior 不太可能直接达到最终发布质量。

null-FM 能稳定出单脸，但它牺牲条件保留。epoch120 latent cosine 只有 0.10446457890793681。它可以作为 prior 诊断，不能作为最终 anonymization generator。

CL-only 抬高了 cosine，但直接破坏 face generation。Point-only epoch12 face_detect_ge1_rate 是 0.0，zero_face_rate 是 1.0。Point+Gram epoch16 也是 face_detect_ge1_rate 0.0，zero_face_rate 1.0。这个方向不能单独训练生成器。

在这次 P/P+G snapshot 中，P+G 没有带来 relation 优势。Point-only raw pairwise_pearson 是 0.9389362005475402，P+G raw 是 0.9102106091085927。Point-only raw offdiag_gram_mse 是 0.027412507510142667，P+G raw 是 0.05806031405358421。

M2 weighted-sum 没有解决冲突。epoch20 是已收集的 best FID，FID 84.60601043701172。epoch80 FID 变成 191.86097717285156。epoch88 的 latest gradient_conflict_fraction 是 0.453125。这说明简单加权还没有稳定地平衡 FM 和 representation 目标。

正式 privacy 不可写通过。旧 V2 single-face recheck 是 protocol blocker，M0 privacy probe 是 ad-hoc，M2/null-FM/CL-only 没有 formal privacy pass。

## 不能证明

- 不能证明任何当前方法已经通过正式 privacy evaluation。
- 不能证明 M2 weighted-sum 同时改善 quality 和 cosine。
- 不能证明 Gram relation 已经优于 Point-only。
- 不能证明 null-FM 是严格 from-scratch null prior。
- 不能证明 5M prior 已经足够最终发布质量。
- 不能用 CL-only 的高 cosine 证明生成器可用，因为 face rate 已经为 0.0。

## 下一步

1. 继续 Point-only 和 Point+Gram 到更多 epoch，只看趋势，不把它们当可用生成器。
2. 除非 P+G 后续显示明确 relation advantage，否则先用 Point-only 作为 representation 诊断基线。
3. 准备 M3 projected update，从 Stage1/null-FM prior 出发，重点记录 projection_applied_fraction、projection_removed_norm_mean、projected_repr_norm_mean 和最终 quality/utility。
4. 同步考虑更强或预训练 FM prior。当前 Stage1/FMs 的质量上限偏低。
5. 只有在 face/latent guard 先通过后，再加 formal privacy eval。未通过 guard 前不写 privacy pass。

## Artifact paths

Live artifact caveat: paths ending in `last_metrics.json` and `validation_relation_metrics_summary.json` are live pointers. P/P+G values in this document are snapshot values collected at 2026-06-02 12:46:44 CST (UTC+08), so later live files may not match the tables above. For quality metrics, prefer the epoch-specific files listed below where available.

| Area | Paths |
| --- | --- |
| Stage1 long200_v4 docs | `docs/experiments/MEDIUM_V1_STAGE1.md` |
| Stage1 long200_v4 metrics | `artifacts/checkpoints/g_medium_v1_stage1_long200_v4/last_metrics.json`; `artifacts/eval/g_medium_v1_stage1_long200_v4/quality`; `artifacts/plots/medium_v1/stage1_long200_v4_metrics_timeseries.json` |
| Stage1 long1000_continue metrics | `artifacts/checkpoints/g_medium_v2_stage1_long1000_continue/last_metrics.json`; `artifacts/eval/g_medium_v2_stage1_long1000_continue/quality`; `artifacts/plots/medium_v2/stage1_long1000_continue` |
| M0 docs and metrics | `docs/experiments/MEDIUM_V1_STAGE2_M0_EPOCH100.md`; `artifacts/checkpoints/g_medium_v1_stage2_m0/last_metrics.json`; `artifacts/eval/g_medium_v1_stage2_m0/quality` |
| M0 ad-hoc privacy | `artifacts/privacy/medium_v1_m0_epoch100_probe_gpu1` |
| old V2 best | `docs/PHASE_SUMMARY_V2_BEST.md`; `artifacts/eval/g_v2_best_basic_val.json`; `artifacts/checkpoints/g_v2_best/best.pt` |
| old V2 single-face recheck | `docs/experiments/G_V2_BEST_SINGLE_FACE_RECHECK.md`; `artifacts/eval/g_v2_best_single_face_recheck/result.json`; `artifacts/eval/g_v2_best_single_face_recheck/generation_quality.json`; `artifacts/eval/g_v2_best_single_face_recheck/generated_images` |
| M2 gram-weighted | `artifacts/checkpoints/g_medium_v2_stage2_m2_gram_weighted/last_metrics.json`; `artifacts/monitor/medium_v2_stage2_m2_gram_weighted_status.json`; `artifacts/monitor/medium_v2_m2_epoch_0080_report.json`; `artifacts/eval/g_medium_v2_stage2_m2_gram_weighted/quality`; `artifacts/plots/medium_v2/m2` |
| null-FM | `artifacts/checkpoints/g_medium_v2_stage2_null_fm/last_metrics.json`; `artifacts/eval/g_medium_v2_stage2_null_fm/quality`; `artifacts/plots/medium_v2/stage2_null_fm` |
| Point-only CL-only | `artifacts/checkpoints/g_medium_v2_stage2_point_only_cl_only/last_metrics.json` (live; table captured epoch12 at 2026-06-02 12:46:44 CST); `artifacts/eval/g_medium_v2_stage2_point_only_cl_only/quality/epoch_0012/stage2_epoch_0012_raw_niqe.json`; `artifacts/metrics/medium_v2/stage2_point_only_cl_only/validation_relation_metrics/validation_relation_metrics_summary.json` (live; relation table keeps first epoch12 snapshot); `artifacts/plots/medium_v2/stage2_point_only_cl_only` |
| Point+Gram CL-only | `artifacts/checkpoints/g_medium_v2_stage2_point_gram_cl_only/last_metrics.json` (live; table captured epoch16 at 2026-06-02 12:46:44 CST); `artifacts/eval/g_medium_v2_stage2_point_gram_cl_only/quality/epoch_0016/stage2_epoch_0016_raw_niqe.json`; `artifacts/metrics/medium_v2/stage2_point_gram_cl_only/validation_relation_metrics/validation_relation_metrics_summary.json` (live; table captured epoch16 snapshot); `artifacts/plots/medium_v2/stage2_point_gram_cl_only` |
