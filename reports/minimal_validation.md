# Minimal Validation Report

Status: implementation is synchronized to 4029 at `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization`. The closed-loop code path runs through `E0`, feature cache, smoke, and one-epoch `G` training. Full privacy evaluation is structurally blocked because generated images are not detected as faces by ArcFace.

## Implemented Chain

- AffectNet strict index builder with required fields and 8-class label validation.
- `E0` ResNet-50 emotion encoder training path with 512-d L2-normalized embedding.
- `E0` feature cache with manifest hashes for index, checkpoint, and shard.
- `G(z)->x_hat` training path where `z` is the only generator input.
- Explicit no-identity-supervision audit for generator training config/source.
- Evaluation for affective preservation, privacy recognizer similarity, and anti-steganography perturbations.
- Per-sample eval rows at `artifacts/eval/per_sample.jsonl`, keyed by `sample_id`.
- Dataset-level deterministic impostor pairs, independent of eval batch size.
- Recognizer asset preflight and SHA256 recording for TorchScript FaceNet/AdaFace checkpoints.
- `tmux` scripts for long-running smoke, `E0`, cache, `G`, and eval jobs.
- RAM guard for long scripts; jobs stop if server RAM reaches 90%.
- Explicit GPU mapping through `SAFA_CUDA_VISIBLE_DEVICES`, with the selected environment injected into the `tmux` job command.

## Local Validation

- `python -m unittest tests.test_index_schema tests.test_dataset_smoke` passed.
- `python -m unittest tests.test_feature_cache tests.test_no_identity_audit tests.test_eval_contract` passed.
- `python -m compileall -q src tests` passed.
- Torch-dependent tests are present but were skipped locally because this Windows environment does not have `torch` and `torchvision` installed.

## Remote Environment Validation

- Repository cloned on 4029 at `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization`.
- The configured HTTP/HTTPS proxy reaches PyPI from 4029.
- Initial validation used base Anaconda Python at `/home/hdd3/zhanghaonan/anaconda3/bin/python`. The default runtime has since been migrated to `/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python`.
- Remote `PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 /home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python -m unittest discover tests` passed all 22 tests.
- `insightface` required pinning `numpy==1.26.4` to match the existing `scikit-image` ABI; `cv2` import was verified after the pin.
- The project `.venv` on 4029 was deleted; the `safa` conda environment is the runtime target.

## Data Repair and Index State

- AffectNet was received as official archives and extracted with `unrar` under `/home/hdd3/zhanghaonan/AffectNet/Manually_Annotated_Images`.
- `training.csv` contained exactly one row whose image path did not exist after extraction:
  `2/9db2af5a1da8bd77355e8c6a655da519a899ecc42641bf254107bfc0.jpg`.
- The row was deleted from `/home/hdd3/zhanghaonan/AffectNet/training.csv` with an audit backup at `artifacts/data_fixes/training.csv.before_missing_row_removal.bak` and row record at `artifacts/data_fixes/removed_missing_training_row.json`.
- Rebuilt strict 8-class indices:
  - `data/index/train.jsonl`: 287651 records.
  - `data/index/val.jsonl`: 4000 records, 500 per class.

## Remote Execution Results

- AffectNet strict train/val indices were rebuilt after CSV repair.
- `E0` training completed. Best validation accuracy: `0.4685`; majority baseline: `0.125`; `passes_majority_baseline=true`.
- Feature cache completed:
  - train: `287651` samples, 512-d L2 features.
  - val: `4000` samples, 512-d L2 features.
- Smoke completed on 32 balanced validation samples. Smoke `G` metrics: loss `1.4315`, cycle `0.7562`, semantic CE `2.7011`, TV `0.0428`.
- Full `G` one-epoch training completed. Metrics: loss `0.1759`, cycle `0.0124`, semantic CE `0.6538`, TV `0.0183`.
- FaceNet and AdaFace privacy checkpoints were exported through the 4029 proxy:
  - FaceNet SHA256: `38855589cae480268e8c64c2556c7898c449428043aacce761fe43389c6a8c72`.
  - AdaFace SHA256: `9953ec6b93d2bb7e771dcd2717f1e9b15ecd4760983557b006cbe00e598618d7`.
- FaceNet is exported in the separate `/home/hdd3/zhanghaonan/anaconda3/envs/facenet` environment with torch `2.2.2+cu121`, torchvision `0.17.2+cu121`, and `facenet-pytorch==2.6.0`; the `safa` runtime loads only the exported TorchScript recognizer.
- Affective-only diagnostic eval completed on val:
  - latent cosine mean `0.9869`, median `0.9891`.
  - angle mean `0.1534` rad, median `0.1481` rad.
  - generated label accuracy `0.45925`.
  - source prediction preservation `0.92`.
  - logit L2 drift mean `0.4733`.
  - anti-steg diagnostics were written for `jpeg`, `blur`, `downsample`, `crop`, and `noise`.
- Full multi-recognizer privacy eval did not complete. ArcFace failed on generated images with `expected exactly one face, detected 0`.
- ArcFace detection diagnostic on the first 64 validation samples:
  - source: 64/64 had exactly one detected face.
  - generated: 64/64 had zero detected faces.
- Server RAM stayed below the 90% limit in observed runs; typical observed usage was 14-20 GiB out of 251 GiB after the training/cache stages.

## First Remote Acceptance Target

The first success criterion is a complete closed loop, not a hard latent-cosine threshold:

- `E0` training runs and beats majority-class validation baseline.
- Feature cache writes a valid manifest.
- Smoke completes and writes `artifacts/smoke/smoke_result.json`.
- `G` short run writes a checkpoint and finite losses.
- Affective diagnostic eval writes `artifacts/eval/g_val_affective_only.json` and `artifacts/eval/per_sample_affective_only.jsonl`.
- Full privacy eval will write `artifacts/eval/g_val.json` and `artifacts/eval/per_sample.jsonl` only after generated faces are detectable by ArcFace.

## Structural Issue

The current `G` objective preserves the frozen affective representation well, but it does not force outputs onto a face-image manifold strongly enough for ArcFace detection. This is not a privacy success. It means the first generator is finding a machine-feature shortcut rather than producing valid anonymized faces. The next method-level fix should add a faithful non-identity image prior or generator prior, not a post-processing detector workaround.
