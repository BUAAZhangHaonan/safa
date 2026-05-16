# Minimal Validation Report

Status: implementation prepared locally; remote login works through remote access setup; remote repository synchronization pending path correction to `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization`.

## Implemented Chain

- AffectNet strict index builder with required fields and 8-class label validation.
- `E0` ResNet-50 emotion encoder training path with 512-d L2-normalized embedding.
- `E0` feature cache with manifest hashes for index, checkpoint, and shard.
- `G(z)->x_hat` training path where `z` is the only generator input.
- Explicit no-identity-supervision audit for generator training config/source.
- Evaluation for affective preservation, privacy recognizer similarity, and anti-steganography perturbations.
- `tmux` scripts for long-running smoke, `E0`, and `G` jobs.

## Local Validation

- `python -m unittest tests.test_index_schema tests.test_dataset_smoke` passed.
- `python -m unittest tests.test_feature_cache tests.test_no_identity_audit tests.test_eval_contract` passed.
- `python -m compileall -q src tests` passed.
- Torch-dependent tests are present but were skipped locally because this Windows environment does not have `torch` and `torchvision` installed.

## Remote Environment Validation

- Repository cloned on 4029 at `/home/hdd3/zhanghaonan/projects/samplewise-affective-face-anonymization`.
- Proxy `http://<proxy-host>:<proxy-port>` reaches PyPI from 4029.
- Anaconda Python at `/home/hdd3/zhanghaonan/anaconda3/bin/python` has torch `2.11.0+cu128`, torchvision `0.26.0+cu128`, CUDA visible on GPU 0, insightface, and onnxruntime installed.
- Remote `PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 /home/hdd3/zhanghaonan/anaconda3/bin/python -m unittest discover tests` passed all 18 tests.
- `insightface` required pinning `numpy==1.26.4` to match the existing `scikit-image` ABI; `cv2` import was verified after the pin.

## Data Repair and Index State

- AffectNet was received as official archives and extracted with `unrar` under `/home/hdd3/zhanghaonan/AffectNet/Manually_Annotated_Images`.
- `training.csv` contained exactly one row whose image path did not exist after extraction:
  `2/9db2af5a1da8bd77355e8c6a655da519a899ecc42641bf254107bfc0.jpg`.
- The row was deleted from `/home/hdd3/zhanghaonan/AffectNet/training.csv` with an audit backup at `artifacts/data_fixes/training.csv.before_missing_row_removal.bak` and row record at `artifacts/data_fixes/removed_missing_training_row.json`.
- Rebuilt strict 8-class indices:
  - `data/index/train.jsonl`: 287651 records.
  - `data/index/val.jsonl`: 4000 records, 500 per class.

## Remote Execution Blockers

- OpenSSH batch login to `4029` still fails without remote access setup or SSH key. Paramiko remote access setup was used for setup.
- AffectNet root exists at `/home/hdd3/zhanghaonan/AffectNet`; full label/layout validation is pending.
- External recognizer assets for FaceNet and AdaFace must be placed at the configured paths or the evaluation must stop.

## First Remote Acceptance Target

The first success criterion is a complete closed loop, not a hard latent-cosine threshold:

- `E0` training runs and beats majority-class validation baseline.
- Feature cache writes a valid manifest.
- Smoke completes and writes `artifacts/smoke/smoke_result.json`.
- `G` short run writes a checkpoint and finite losses.
- Eval writes `artifacts/eval/g_val.json` with latent, classification, privacy, and perturbation metric distributions.
