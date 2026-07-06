# SAFA 代码审查与清理记录 2026-07-05

## 范围

本次审查覆盖 `configs/`、`scripts/`、`docs/`、`reports/`、`tests/`、`data/index/`、`src/safa/cli/`、`src/safa/data/`、`src/safa/models/`、`src/safa/training/`、`src/safa/evaluation/`。

本次不判断数学框架是否成立。身份隐私不做形式化保证；项目应按事后验证处理，即用外部 face recognizer / face verification 网络比较生成图 `X` 和源图 `X0`，确认生成图不保留源身份。

## 清理结果

已清理的内容只限 ignored 生成缓存：

- `.pytest_cache/`
- `scripts/**/__pycache__/`
- `src/safa/**/__pycache__/`
- `tests/__pycache__/`

这些文件都是 Python 或 pytest 生成物，已被 `.gitignore` 覆盖，可重新生成。

本轮没有删除 tracked 源码、配置、索引、checkpoint、pretrained 权重、实验日志或评估产物。`data/index/*.jsonl` 虽然被 `.gitignore` 匹配，但当前仍是 tracked 文件，并被 configs/tests/docs/manifests 多处引用，所以不能在没有迁移方案时直接删除。

## 人工确认后再清理的候选

- `.vscode/`：个人 IDE 配置，已 ignored。
- `logs/`：已 ignored，但可能有实验追踪价值。
- `artifacts/reports/sync_backups/20260606T083416Z_candidate_rerank_4029/`：像旧同步备份，但可能仍是审计材料。
- `data/index/train_face_mixed_e14.jsonl` 和 manifest：用户已确认这是 K100 服务器上训练得到的文件，本轮不在 4029 上重建或迁移。

## 外部资产/跨机器状态，当前不处理

- `data/index/train_face_mixed_e14.jsonl` 和 manifest：路径指向 `/home/k100/Datasets/Face/...`，这是 K100 训练资产，不作为 4029 本轮 BUG 处理。
- `artifacts/e0_features/train_face_mixed_e14_e0_medium_v1` 和 `artifacts/e0_features/val_face_mixed_e14_e0_medium_v1`：用户已确认 E14 feature cache 来自 K100 服务器，本轮不处理。
- `zhuyu_sit_l_2_imagenet256.pt` 和 `zhuyu_sit_b_2_imagenet256.pt`：用户已确认 B/2、L/2 MeanFlow/SiT 权重正在其他服务器训练，本轮不处理。

## E0 checkpoint 校对结果

本轮没有修改 `artifacts/`。在 4029 原项目 `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization` 的 `artifacts/checkpoints/` 下，发现这些 E0 候选目录：

- `artifacts/checkpoints/e0_convnext_tiny`：有 `best.pt` 和 `manifest.json`，manifest 记录 `backbone=convnext_tiny`，`accuracy=0.55725`。
- `artifacts/checkpoints/e0_dinov2_large`：有 `best.pt`，没有 manifest。
- `artifacts/checkpoints/e0_dinov2_large_v2`：有 `best.pt` 和 `manifest.json`，manifest 记录 `backbone=dinov2_large`，`accuracy=0.561`。
- `artifacts/checkpoints/e0_dinov3_vitl16`：有 `best.pt` 和 `manifest.json`，manifest 记录 `backbone=dinov3_vitl16`，`accuracy=0.54425`。
- `artifacts/checkpoints/e0_iresnet100`：有 `best.pt` 和 `manifest.json`，manifest 记录 `backbone=iresnet100`，`accuracy=0.54975`。
- `artifacts/checkpoints/e0_medium_v1`：有 `best.pt` 和 `manifest.json`，manifest 记录 `accuracy=0.563114134542706`，但没有 `backbone` 字段。
- `artifacts/checkpoints/e0_mobilenetv3`：有 `best.pt` 和 `manifest.json`，manifest 记录 `backbone=mobilenetv3_large`，`accuracy=0.54575`。
- `artifacts/checkpoints/e0_resnet18`：有 `best.pt` 和 `manifest.json`，manifest 记录 `backbone=resnet18`，`accuracy=0.551`。
- `artifacts/checkpoints/e0_resnet18_v4`：有 `best.pt` 和 `manifest.json`，manifest 记录 `backbone=resnet18`，`accuracy=0.55375`。
- `artifacts/checkpoints/e0_swin_tiny`：有 `best.pt` 和 `manifest.json`，manifest 记录 `backbone=swin_tiny`，`accuracy=0.54675`。
- `artifacts/checkpoints/e0_vgg16`：有 `best.pt` 和 `manifest.json`，manifest 记录 `backbone=vgg16`，`accuracy=0.546`。
- `artifacts/checkpoints/e0_vgg16_v4`：有 `best.pt` 和 `manifest.json`，manifest 记录 `backbone=vgg16`，`accuracy=0.5475`。

已检查 `configs/cache_e0.yaml:7`、`configs/eval.yaml:8`、`configs/train_g.yaml:13`、`configs/smoke.yaml:9`。这些 legacy config 只引用 `artifacts/checkpoints/e0/best.pt`，上下文没有写 E0 backbone、实验名或版本。相关旧 feature manifest `artifacts/e0_features/train/manifest.json`、`artifacts/e0_features/val/manifest.json`、`artifacts/e0_features/val_single_face_e0/manifest.json` 也只记录原路径和 `encoder_checkpoint_sha256=5f165c520fad315dd1550676c6515c3480585e8ea0dcf1841fd678c8f1963e0f`。对 12 个候选 `best.pt` 计算 sha256 后，没有一个匹配该值；也未发现可判别的 E0 README。

结论：不能从文件名、manifest 或 README 唯一推断这四个 legacy config 应指向哪个 E0。本轮未修改 `configs/cache_e0.yaml`、`configs/eval.yaml`、`configs/train_g.yaml`、`configs/smoke.yaml`。需要人工选择目标 E0，或恢复 sha256 为 `5f165c520fad315dd1550676c6515c3480585e8ea0dcf1841fd678c8f1963e0f` 的原 `artifacts/checkpoints/e0/best.pt`。

## 已确认 BUG（非资产，已修复）

以下非资产问题已完成代码或测试修复，本报告保留问题记录。

### 1. latent rerank 评估未传 `latent_codec`

- 严重度：高
- 位置：`src/safa/evaluation/runner.py:42`、`:78`、`:90`、`:331`、`:462`、`:562`；`src/safa/training/latent_codec.py:62`
- 现象：`latent_training` 和 `candidate_rerank.enabled` 同时启用时，rerank 路径会把 latent tensor 直接送进 E0、face detector、anti-steg 评估。
- 根因/影响：普通采样路径会传 `latent_codec`，但 rerank 和 adaptive rerank 路径没有传。latent SiT 输出是 `[B,4,H,W]`，不是图像，会导致通道数崩溃，或让评估指标基于错误输入。
- 建议：给 `_sample_reranked_generated_for_eval` 和 `_sample_adaptive_reranked_generated_for_eval` 增加 `latent_codec` 参数，并把它传给内部所有 `_sample_generated_for_eval(...)`。

### 2. eval 落盘 JSON 缺 `out_json`

- 严重度：中
- 位置：`src/safa/evaluation/runner.py:237`、`:248`、`:249`；`src/safa/cli/eval.py:73`
- 现象：返回对象里有 `out_json`，但写出的 JSON 文件里没有 `out_json`。
- 根因/影响：代码先写 JSON，再设置 `result["out_json"]`。CLI 输出和 artifact 不一致，下游脚本读取落盘 JSON 时缺结果路径。
- 建议：在 `write_text(json.dumps(...))` 前设置 `result["out_json"]`。

### 3. 单脸 rerank 约束可静默失效

- 严重度：中
- 位置：`src/safa/evaluation/runner.py:348`、`:546`、`:599`、`:640`、`:668`
- 现象：`candidate_rerank.adaptive_k.require_single_face: true` 在没有启用 face detector 时不会真正执行。
- 根因/影响：face count 只在 detector 存在时收集。没有 detector 时逻辑只能标记缺指标，最终仍可能按 latent cosine 选样本。配置要求和实际选择不一致。
- 建议：配置校验中要求 `require_single_face` 或 `single_face_priority` 启用时必须启用 face detection；否则直接报错。

### 4. smoke 测试引用了错误文档路径

- 严重度：高
- 位置：`tests/test_smoke_cli.py:116`
- 现象：测试读取 `docs/4029_runbook.md`，但实际文件是 `docs/experiments/4029_runbook.md`。
- 根因/影响：文档移动后测试未更新，导致测试直接失败。
- 建议：改测试路径，或恢复对应文档位置。

### 5. supervisor 会误判 privacy skip

- 严重度：高
- 位置：`scripts/supervise_medium_v2_stages.py:390-404`；`src/safa/evaluation/runner.py:222-225`
- 现象：`privacy_guard_pass=true` 可能和 `privacy_skipped=true`、`skip_reason=privacy_protocol_blocker` 同时出现，但 supervisor 会把它当成不需要记录 skip。
- 根因/影响：`privacy_guard_pass` 只是 pre-privacy guard，不代表正式 privacy protocol 已通过。实验汇总可能把被 blocker 跳过的结果误当成可用。
- 建议：判断应同时要求 `privacy_skipped is False`、`skip_reason is None`，并确认 `metrics.privacy` 非空。

## 疑点

- `src/safa/training/g_loop.py:701`、`:1358`、`:3207`：`stages.stage2.stage2_objective.optimizer_type` 和顶层 `config.optimizer_type` 双位置配置。已见配置基本一致；如果不一致，可能出现真实 optimizer 和 projected update 逻辑错配。
- `src/safa/utils/config.py:7` 以及 CLI 入口：相对路径按当前工作目录解析，不按配置文件所在目录解析。runbook 要求从项目根运行，所以这是复现风险，不单独列为 BUG。
- `configs/paths.yaml:1-2`：硬编码 4029 项目路径和 AffectNet 路径。当前机器存在，但可移植性弱。
- 多个 latent training 配置使用 `vae_model: stabilityai/sd-vae-ft-ema`，本地已有 `artifacts/checkpoints/external/sd-vae-ft-ema`。离线复现时可能应改成本地 `vae_path`。
- K100/H100 脚本硬编码 `/home/k100/miniconda3/...` 等路径。如果这些脚本只在对应机器跑，可以保留；如果要在 4029 跑，需要参数化。
- `g_loop.py` 使用 seed，但也启用了 `torch.backends.cudnn.benchmark = True`。严格确定性不足。

## 当前验证

本轮资产校对只运行了轻量文件检查、manifest 读取、`sha256sum`、`git status` 和 `git diff`。修复完成后运行了完整测试：`PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest tests -q`，结果 `612 passed, 20 warnings, 143 subtests passed in 31.30s`。未跑长训练。

清理 worktree 运行过：

```bash
PYTHONPATH=src /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m pytest tests -q
```

结果：`593 passed, 16 failed`，耗时约 44.27 秒。

失败不是清理引入的，因为清理前后 `git diff --stat` 为空。失败集中在测试/配置合同问题：eval contract 缺 `model_type`、E0 配置新增必需键、`_save_generator` 需要 `validation`、缺少 `docs/4029_runbook.md`。
