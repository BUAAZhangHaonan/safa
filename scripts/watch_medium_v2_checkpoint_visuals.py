#!/usr/bin/env python3
"""Watch M2 checkpoints and render fixed validation source/generated pairs."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CHECKPOINT_DIR = Path("artifacts/checkpoints/g_medium_v2_stage2_m2_gram_weighted")
DEFAULT_METRICS = DEFAULT_CHECKPOINT_DIR / "last_metrics.json"
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "last.pt"
DEFAULT_CONFIG = Path("configs/medium_v2/train_g_medium_v2_stage2_m2_gram_weighted.yaml")
DEFAULT_INDEX = Path("data/index/val_single_face.jsonl")
DEFAULT_FEATURES = Path("artifacts/e0_features/val_single_face_e0_medium_v1")
DEFAULT_OUT_DIR = Path("artifacts/plots/medium_v2/m2")
DEFAULT_EVENTS = DEFAULT_OUT_DIR / "checkpoint_visuals_events.jsonl"
DEFAULT_LOG = DEFAULT_OUT_DIR / "checkpoint_visuals.log"
DEFAULT_STATE = DEFAULT_OUT_DIR / "checkpoint_visuals_state.json"
DEFAULT_PYTHON = "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"

IMAGE_SIZE = 132
TILE_WIDTH = 360
TILE_HEIGHT = 312
TITLE_HEIGHT = 76
GRID_COLUMNS = 4
EMOTION_LABELS = (
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "anger",
    "contempt",
)


class WatcherPaths(NamedTuple):
    metrics: Path = DEFAULT_METRICS
    checkpoint: Path = DEFAULT_CHECKPOINT
    config: Path = DEFAULT_CONFIG
    index: Path = DEFAULT_INDEX
    features: Path = DEFAULT_FEATURES
    out_dir: Path = DEFAULT_OUT_DIR
    events: Path = DEFAULT_EVENTS
    log: Path = DEFAULT_LOG
    state: Path = DEFAULT_STATE


GenerateFunc = Callable[..., None]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def append_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            detail = row.get("out_path") or row.get("error") or row.get("reason") or ""
            handle.write(f"{row['time']} {row['type']}: {detail}\n")


def read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_stage_epoch(metrics_path: Path) -> int:
    payload = read_json(metrics_path)
    if not isinstance(payload, dict) or "stage_epoch_1based" not in payload:
        raise ValueError(f"metrics missing stage_epoch_1based: {metrics_path}")
    value = payload["stage_epoch_1based"]
    if isinstance(value, bool):
        raise ValueError(f"stage_epoch_1based must be a positive integer, got {value!r}")
    try:
        epoch = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stage_epoch_1based must be a positive integer, got {value!r}") from exc
    if epoch <= 0:
        raise ValueError(f"stage_epoch_1based must be a positive integer, got {value!r}")
    return epoch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(payload)
    if not rows:
        raise ValueError(f"index contains no rows: {path}")
    return rows


def select_samples(index_path: Path, *, num_samples: int, sample_seed: int | None) -> list[dict[str, Any]]:
    if isinstance(num_samples, bool) or int(num_samples) <= 0:
        raise ValueError(f"num_samples must be a positive integer, got {num_samples!r}")
    rows = read_jsonl(index_path)
    selected = list(rows)
    if sample_seed is not None:
        random.Random(int(sample_seed)).shuffle(selected)
    selected = selected[: int(num_samples)]
    parsed: list[dict[str, Any]] = []
    for row in selected:
        sample_id = row.get("sample_id")
        image_path = row.get("image_path")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"index row missing sample_id: {row!r}")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"index row {sample_id!r} missing image_path")
        parsed.append(
            {
                "sample_id": sample_id,
                "label": row.get("label"),
                "image_path": image_path,
                "split": row.get("split"),
            }
        )
    return parsed


def validate_required_inputs(paths: WatcherPaths) -> None:
    if not paths.config.is_file():
        raise FileNotFoundError(f"config not found: {paths.config}")
    if not paths.index.is_file():
        raise FileNotFoundError(f"index not found: {paths.index}")
    if not paths.features.is_dir():
        raise FileNotFoundError(f"features dir not found: {paths.features}")
    for name in ("features.pt", "manifest.json"):
        required = paths.features / name
        if not required.is_file():
            raise FileNotFoundError(f"feature cache file not found: {required}")


def config_sampling_seed(config_path: Path) -> int:
    try:
        from safa.utils.config import load_yaml
        from safa.utils.sampling import sampling_base_seed_from_config
    except Exception as exc:
        raise RuntimeError("safa config utilities are required to read sampling_seed") from exc

    config = load_yaml(config_path)
    return int(sampling_base_seed_from_config(config))


def output_paths(out_dir: Path, epoch: int) -> tuple[Path, Path]:
    return (
        out_dir / f"epoch_{epoch:04d}_checkpoint_pairs.png",
        out_dir / f"epoch_{epoch:04d}_checkpoint_pairs_manifest.json",
    )


def build_manifest(
    *,
    epoch: int,
    paths: WatcherPaths,
    samples: list[dict[str, Any]],
    out_path: Path,
    device: str,
    sampling_seed: int,
    metrics: list[dict[str, Any]],
    note: str,
) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "epoch": int(epoch),
        "output": str(out_path),
        "device": device,
        "sampling_seed": int(sampling_seed),
        "selection": {
            "source": "data/index/val_single_face.jsonl",
            "method": "first_n_or_fixed_seed_shuffle",
            "num_samples": len(samples),
        },
        "inputs": {
            "metrics": str(paths.metrics),
            "checkpoint": str(paths.checkpoint),
            "config": str(paths.config),
            "index": str(paths.index),
            "features": str(paths.features),
        },
        "samples": samples,
        "metrics": metrics,
        "note": note,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json_atomic(path, manifest)


def validate_cuda_visible_devices(value: str | None) -> None:
    if value is None or value == "":
        return
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            index = int(raw)
        except ValueError:
            continue
        if index in {3, 4, 5, 6}:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES includes GPU3-6, which are reserved for M2: {value}")


def query_nvidia_smi() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def guard_gpu0_available(nvidia_smi_output: str, *, max_memory_mb: int, max_util_pct: int) -> None:
    for line in nvidia_smi_output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4 or parts[0] != "0":
            continue
        util = int(float(parts[1]))
        memory_used = int(float(parts[2]))
        if util > int(max_util_pct) or memory_used > int(max_memory_mb):
            raise RuntimeError(
                f"GPU0 is busy: utilization={util}% memory_used={memory_used}MiB "
                f"(limits {max_util_pct}%/{max_memory_mb}MiB)"
            )
        return
    raise RuntimeError("GPU0 not found in nvidia-smi output")


def wait_until_gpu0_available(*, max_memory_mb: int, max_util_pct: int, interval_seconds: int, wait: bool) -> None:
    validate_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    while True:
        try:
            guard_gpu0_available(query_nvidia_smi(), max_memory_mb=max_memory_mb, max_util_pct=max_util_pct)
            return
        except RuntimeError:
            if not wait:
                raise
            time.sleep(interval_seconds)


def load_config(path: Path) -> dict[str, Any]:
    from safa.utils.config import load_yaml

    return load_yaml(path)


def generate_checkpoint_pairs(
    *,
    epoch: int,
    paths: WatcherPaths,
    out_path: Path,
    manifest_path: Path,
    num_samples: int,
    sample_seed: int | None,
    device: str,
    sampling_seed: int,
) -> None:
    if device.startswith("cuda"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        wait_until_gpu0_available(max_memory_mb=1024, max_util_pct=50, interval_seconds=60, wait=True)

    import torch
    import torch.nn.functional as F

    from safa.data.dataset import load_rgb_image_strict
    from safa.data.feature_cache import load_feature_cache
    from safa.models.e0 import freeze_e0, load_e0_checkpoint
    from safa.models.generator import build_generator, require_generator_model_config
    from safa.training.losses import normalize_for_e0
    from safa.training.transforms import eval_transform
    from safa.utils.sampling import make_x_init_for_sample_ids

    config = load_config(paths.config)
    e0_checkpoint = Path(str(config["e0_checkpoint"]))
    checkpoint = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    model_config = require_generator_model_config(checkpoint, str(paths.checkpoint))
    generator = build_generator(model_config).to(device)
    generator.load_state_dict(checkpoint["model_state_dict"])
    generator.eval()

    payload, _manifest = load_feature_cache(paths.features, paths.index, e0_checkpoint)
    features = payload["features"]
    feature_by_sample_id = {sample_id: features[index] for index, sample_id in enumerate(payload["sample_ids"])}

    e0, _e0_payload = load_e0_checkpoint(e0_checkpoint, device=device)
    e0.to(device)
    freeze_e0(e0)

    image_size = int(model_config["image_size"])
    sample_steps = int(model_config.get("sample_steps", 32))
    e0_transform = eval_transform(image_size)
    samples = select_samples(paths.index, num_samples=num_samples, sample_seed=sample_seed)
    sample_ids = [row["sample_id"] for row in samples]
    missing = [sample_id for sample_id in sample_ids if sample_id not in feature_by_sample_id]
    if missing:
        raise KeyError(f"selected sample_ids missing from feature cache: {missing[:5]}")

    z = torch.stack([feature_by_sample_id[sample_id] for sample_id in sample_ids], dim=0).to(device)
    x_init = make_x_init_for_sample_ids(sample_ids, sampling_seed, image_size, z.device, z.dtype)
    entries: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    with torch.no_grad():
        generated = generator.sample(z, steps=sample_steps, x_init=x_init).detach().cpu().clamp(0.0, 1.0)
        for index, row in enumerate(samples):
            source_path = Path(str(row["image_path"]))
            source_pil = load_rgb_image_strict(source_path)
            source_for_e0 = e0_transform(source_pil).unsqueeze(0).to(device)
            generated_for_e0 = normalize_for_e0(generated[index].unsqueeze(0).to(device))
            source_out = e0(source_for_e0)
            generated_out = e0(generated_for_e0)
            latent_cosine = float(F.cosine_similarity(generated_out["embedding"], z[index].unsqueeze(0), dim=1).item())
            source_pred = int(source_out["logits"].argmax(dim=1).item())
            generated_pred = int(generated_out["logits"].argmax(dim=1).item())
            generated_pil = tensor_to_pil(generated[index])
            entries.append(
                {
                    "sample_id": row["sample_id"],
                    "label": row.get("label"),
                    "source_path": source_path,
                    "generated_image": generated_pil,
                    "latent_cosine": latent_cosine,
                    "source_pred": source_pred,
                    "generated_pred": generated_pred,
                }
            )
            metrics.append(
                {
                    "sample_id": row["sample_id"],
                    "label": row.get("label"),
                    "latent_cosine": latent_cosine,
                    "source_pred": source_pred,
                    "generated_pred": generated_pred,
                    "source_pred_name": label_name(source_pred),
                    "generated_pred_name": label_name(generated_pred),
                }
            )

    draw_checkpoint_pair_grid(entries, out_path, epoch=epoch, checkpoint_path=paths.checkpoint)
    write_manifest(
        manifest_path,
        build_manifest(
            epoch=epoch,
            paths=paths,
            samples=samples,
            out_path=out_path,
            device=device,
            sampling_seed=sampling_seed,
            metrics=metrics,
            note="latent_cosine/source_pred/generated_pred computed with E0 from config.e0_checkpoint",
        ),
    )


def tensor_to_pil(tensor) -> Image.Image:
    import numpy as np

    array = tensor.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def draw_checkpoint_pair_grid(entries: list[dict[str, Any]], out_path: Path, *, epoch: int, checkpoint_path: Path) -> Path:
    if not entries:
        raise ValueError("no entries to draw")
    columns = min(GRID_COLUMNS, len(entries))
    rows = int(math.ceil(len(entries) / columns))
    canvas = Image.new("RGB", (columns * TILE_WIDTH, TITLE_HEIGHT + rows * TILE_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(21)
    small_font = font(11)
    header_font = font(12)
    title = f"epoch {epoch:04d} checkpoint pairs"
    subtitle = f"checkpoint: {checkpoint_path}"
    draw.text((canvas.width // 2, 18), title, fill=(20, 20, 20), font=title_font, anchor="ma")
    draw.text((canvas.width // 2, 48), short_text(subtitle, 150), fill=(70, 70, 70), font=small_font, anchor="ma")
    for item_index, entry in enumerate(entries):
        row_index = item_index // columns
        col_index = item_index % columns
        x0 = col_index * TILE_WIDTH
        y0 = TITLE_HEIGHT + row_index * TILE_HEIGHT
        source_x = x0 + 32
        generated_x = x0 + 194
        image_y = y0 + 38
        draw.rectangle((x0, y0, x0 + TILE_WIDTH - 1, y0 + TILE_HEIGHT - 1), outline=(226, 226, 226))
        draw.text((source_x + IMAGE_SIZE // 2, y0 + 18), "source X0", fill=(65, 65, 65), font=header_font, anchor="ma")
        draw.text((generated_x + IMAGE_SIZE // 2, y0 + 18), "generated X", fill=(65, 65, 65), font=header_font, anchor="ma")
        paste_entry_image(canvas, entry, "source", source_x, image_y)
        paste_entry_image(canvas, entry, "generated", generated_x, image_y)
        for line_index, line in enumerate(entry_lines(entry, epoch)):
            draw.text((x0 + 18, y0 + 188 + line_index * 18), line, fill=(25, 25, 25), font=small_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


def paste_entry_image(canvas: Image.Image, entry: dict[str, Any], kind: str, x: int, y: int) -> None:
    image_value = entry.get(f"{kind}_image")
    if isinstance(image_value, Image.Image):
        image = image_value.convert("RGB")
    else:
        path = Path(entry[f"{kind}_path"])
        with Image.open(path) as opened:
            image = opened.convert("RGB")
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    image.thumbnail((IMAGE_SIZE, IMAGE_SIZE), resample)
    frame = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (245, 245, 245))
    frame.paste(image, ((IMAGE_SIZE - image.width) // 2, (IMAGE_SIZE - image.height) // 2))
    canvas.paste(frame, (x, y))


def entry_lines(entry: dict[str, Any], epoch: int) -> list[str]:
    lines = [
        f"id {short_text(str(entry['sample_id']), 42)}",
        f"label {entry.get('label')}  epoch {epoch:04d}",
    ]
    latent_cosine = entry.get("latent_cosine")
    source_pred = entry.get("source_pred")
    generated_pred = entry.get("generated_pred")
    if latent_cosine is not None:
        lines.append(f"latent cosine {float(latent_cosine):.4f}")
    else:
        lines.append("latent cosine n/a")
    if source_pred is not None and generated_pred is not None:
        lines.append(f"pred {label_name(source_pred)} -> {label_name(generated_pred)}")
    else:
        lines.append("pred n/a")
    return lines


def short_text(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return "..." + compact[-(limit - 3) :]


def label_name(value: int | None) -> str:
    if value is None:
        return "n/a"
    index = int(value)
    if 0 <= index < len(EMOTION_LABELS):
        return f"{index}:{EMOTION_LABELS[index]}"
    return str(index)


def font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def run_once(
    paths: WatcherPaths,
    *,
    num_samples: int = 16,
    sample_seed: int | None = None,
    device: str = "cuda:0",
    generate_func: GenerateFunc = generate_checkpoint_pairs,
) -> int:
    validate_required_inputs(paths)
    now = utc_now()
    state = read_state(paths.state)
    completed = dict(state.get("completed", {})) if isinstance(state.get("completed"), dict) else {}
    events: list[dict[str, Any]] = []

    if not paths.metrics.is_file():
        events.append({"time": now, "type": "checkpoint_visual_waiting", "reason": f"metrics missing: {paths.metrics}"})
        write_state(paths, completed, events)
        return 0
    if not paths.checkpoint.is_file():
        events.append({"time": now, "type": "checkpoint_visual_waiting", "reason": f"checkpoint missing: {paths.checkpoint}"})
        write_state(paths, completed, events)
        return 0

    epoch = read_stage_epoch(paths.metrics)
    epoch_key = f"{epoch:04d}"
    out_path, manifest_path = output_paths(paths.out_dir, epoch)
    if epoch_key in completed and out_path.is_file() and manifest_path.is_file():
        write_state(paths, completed, events)
        return 0

    sampling_seed = config_sampling_seed(paths.config)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    generate_func(
        epoch=epoch,
        paths=paths,
        out_path=out_path,
        manifest_path=manifest_path,
        num_samples=num_samples,
        sample_seed=sample_seed,
        device=device,
        sampling_seed=sampling_seed,
    )
    completed[epoch_key] = str(out_path)
    events.append(
        {
            "time": now,
            "type": "checkpoint_visual_created",
            "epoch": epoch,
            "out_path": str(out_path),
            "manifest_path": str(manifest_path),
        }
    )
    write_state(paths, completed, events)
    return 0


def write_state(paths: WatcherPaths, completed: dict[str, Any], events: list[dict[str, Any]]) -> None:
    payload = {
        "time": utc_now(),
        "metrics": str(paths.metrics),
        "checkpoint": str(paths.checkpoint),
        "config": str(paths.config),
        "index": str(paths.index),
        "features": str(paths.features),
        "out_dir": str(paths.out_dir),
        "completed": completed,
        "new_event_count": len(events),
    }
    write_json_atomic(paths.state, payload)
    append_jsonl(paths.events, events)
    append_log(paths.log, events)


def loop(
    paths: WatcherPaths,
    *,
    num_samples: int,
    sample_seed: int | None,
    interval_seconds: int,
    device: str,
) -> int:
    while True:
        try:
            run_once(paths, num_samples=num_samples, sample_seed=sample_seed, device=device)
        except Exception as exc:
            now = utc_now()
            events = [{"time": now, "type": "checkpoint_visual_error", "error": f"{type(exc).__name__}: {exc}"}]
            append_jsonl(paths.events, events)
            append_log(paths.log, events)
        time.sleep(interval_seconds)


def build_tmux_command(
    *,
    session_name: str,
    python_exe: str,
    script: str,
    interval: int,
    device: str,
    cuda_visible_devices: str,
) -> list[str]:
    script_command = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"CUDA_VISIBLE_DEVICES={shlex.quote(cuda_visible_devices)} PYTHONPATH=src "
        f"{shlex.quote(python_exe)} {shlex.quote(script)} "
        f"--interval {int(interval)} --device {shlex.quote(device)}"
    )
    return ["tmux", "new-session", "-d", "-s", session_name, script_command]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch medium v2 M2 last.pt and render checkpoint pair visuals.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device.startswith("cuda"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        validate_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    paths = WatcherPaths(
        metrics=args.metrics,
        checkpoint=args.checkpoint,
        config=args.config,
        index=args.index,
        features=args.features,
        out_dir=args.out_dir,
        events=args.events,
        log=args.log,
        state=args.state,
    )
    if args.once:
        return run_once(paths, num_samples=args.num_samples, sample_seed=args.sample_seed, device=args.device)
    return loop(
        paths,
        num_samples=args.num_samples,
        sample_seed=args.sample_seed,
        interval_seconds=args.interval,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
