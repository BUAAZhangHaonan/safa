from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import re

from safa.data.feature_dataset import FeatureAlignedAffectNet
from safa.evaluation.metrics import face_count_rates, flatten_finite_numbers, summarize
from safa.evaluation.perturbations import perturbation_map
from safa.evaluation.recognizers import InsightFaceDetector, build_recognizers, describe_recognizer_assets, validate_recognizer_configs
from safa.models.e0 import freeze_e0, load_e0_checkpoint
from safa.models.generator import build_generator, require_generator_model_config
from safa.training.losses import normalize_for_e0
from safa.training.transforms import generator_image_transform
from safa.utils.device import assert_finite_tensor, require_cuda_device
from safa.utils.hashing import sha256_file
from safa.utils.sampling import make_x_init_for_sample_ids, sampling_base_seed_from_config
from safa.utils.seed import set_seed


def run_eval_from_config(config: dict) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    privacy_cfg, face_detection_cfg, anti_cfg = _eval_monitor_configs(config)
    set_seed(int(config["seed"]))
    device = require_cuda_device(str(config["device"]))
    e0, e0_checkpoint = load_e0_checkpoint(config["e0_checkpoint"], device=str(device))
    e0.to(device)
    freeze_e0(e0)
    generator = _load_generator(config["g_checkpoint"], config, str(device))
    sampling_seed = sampling_base_seed_from_config(config)
    dataset = FeatureAlignedAffectNet(
        config["index"],
        config["features"],
        config["e0_checkpoint"],
        transform=generator_image_transform(int(config["image_size"])),
    )
    feature_metadata = _feature_metadata_for_eval(dataset, generator, e0_checkpoint, config["features"])
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), pin_memory=True)
    detector = _build_face_detector(face_detection_cfg, str(device))
    recognizer_assets = []
    perturbations = perturbation_map(anti_cfg, int(config["seed"])) if anti_cfg["enabled"] else {}

    rows: list[dict] = []
    generated_chunks = [] if privacy_cfg["enabled"] else None
    sample_dir = Path(config["sample_dir"])
    sample_dir.mkdir(parents=True, exist_ok=True)
    generated_image_dir = _generated_image_output_dir(config)
    generated_image_count = 0
    saved_samples = 0
    e0.eval()
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            source = batch["image"].to(device, non_blocking=True)
            z = batch["z"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            sample_ids = list(batch["sample_id"])
            generated = _sample_generated_for_eval(generator, z, sample_ids, sampling_seed, int(config["image_size"]))
            assert_finite_tensor("eval_generated", generated)
            if generated_chunks is not None:
                generated_chunks.append(generated.detach().cpu())
            source_out = e0(normalize_for_e0(source))
            generated_out = e0(normalize_for_e0(generated))
            batch_rows = _make_affective_rows(sample_ids, labels, source_out, generated_out, z)
            if detector is not None:
                _attach_face_detection_rows(batch_rows, detector.detect_counts(generated))
            rows.extend(batch_rows)
            row_start = len(rows) - len(batch_rows)
            if generated_image_dir is not None:
                for i, sample_id in enumerate(sample_ids):
                    _save_generated_image_for_eval(
                        generated[i],
                        generated_image_dir,
                        global_index=row_start + i,
                        sample_id=sample_id,
                        row=rows[row_start + i],
                    )
                    generated_image_count += 1
            for name, perturb in perturbations.items():
                perturbed = perturb(generated)
                assert_finite_tensor(f"perturbed_{name}", perturbed)
                perturbed_out = e0(normalize_for_e0(perturbed))
                _attach_perturbed_affective_rows(rows, row_start, name, perturbed_out, z)
            if saved_samples < 16:
                save_image(generated[: min(4, generated.shape[0])].detach().cpu(), sample_dir / f"generated_{saved_samples:04d}.png", nrow=4)
                saved_samples += int(min(4, generated.shape[0]))
    summarized = _summarize_rows(rows)
    guard = _guard_result(summarized, face_detection_cfg)
    privacy_skipped = bool(privacy_cfg["enabled"] and not guard["passed"])
    if privacy_cfg["enabled"] and guard["passed"]:
        recognizer_assets = describe_recognizer_assets(privacy_cfg["recognizers"])
        recognizers = build_recognizers(privacy_cfg["recognizers"], str(device))
        privacy_store = _empty_privacy_store(recognizers, perturbations)
        _run_privacy_pass(config, loader, generated_chunks, recognizers, perturbations, privacy_store, device)
        _attach_privacy_rows(rows, privacy_store)
        summarized = _summarize_rows(rows)
    result = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {"name": "AffectNet", "num_samples": len(dataset), "index": config["index"]},
        "checkpoints": {
            "e0": config["e0_checkpoint"],
            "e0_sha256": sha256_file(config["e0_checkpoint"]),
            "g": config["g_checkpoint"],
            "g_sha256": sha256_file(config["g_checkpoint"]),
        },
        "features": feature_metadata,
        "recognizer_assets": recognizer_assets,
        "face_detection_guard": guard,
        "privacy_skipped": privacy_skipped,
        "metrics": summarized,
        "artifacts": {
            "sample_dir": str(sample_dir),
            "per_sample_jsonl": config["per_sample_jsonl"],
            "generated_image_dir": str(generated_image_dir) if generated_image_dir is not None else None,
            "generated_image_count": generated_image_count,
        },
        "sampling": {"base_seed": sampling_seed, "stable_x_init": True},
    }
    flatten_finite_numbers(result["metrics"])
    if len(rows) != len(dataset):
        raise RuntimeError(f"Per-sample eval row count mismatch: rows={len(rows)} dataset={len(dataset)}")
    per_sample_jsonl = Path(config["per_sample_jsonl"])
    per_sample_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with per_sample_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    out_json = Path(config["out_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    result["out_json"] = str(out_json)
    if privacy_skipped:
        raise RuntimeError(
            "Privacy evaluation skipped because generation guard failed: "
            f"face_detection_rate={guard['face_detection_rate']} "
            f"latent_cosine_mean={guard['latent_cosine_mean']} "
            f"thresholds=({guard['face_detection_threshold']}, {guard['latent_cosine_threshold']})"
        )
    return result


def _load_generator(checkpoint_path: str, config: dict, device: str):
    import torch

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Generator checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device)
    model_config = require_generator_model_config(payload, str(path))
    state_dict = _generator_state_dict_for_eval(payload, config, str(path))
    generator = build_generator(model_config)
    generator.load_state_dict(state_dict)
    return generator.to(device).eval()


def _generator_state_dict_for_eval(payload: dict, config: dict, checkpoint_path: str):
    source = _eval_checkpoint_model_source(payload, config)
    if source == "raw":
        return payload["model_state_dict"]
    if source == "ema":
        state_dict = payload.get("ema_model_state_dict")
        if state_dict is None:
            raise ValueError(f"Generator checkpoint requested EMA weights but missing ema_model_state_dict: {checkpoint_path}")
        return state_dict
    raise ValueError(f"checkpoint_model must be 'raw' or 'ema', got {source!r}")


def _eval_checkpoint_model_source(payload: dict, config: dict) -> str:
    if "checkpoint_model" in config:
        source = config["checkpoint_model"]
        if source not in ("raw", "ema"):
            raise ValueError(f"checkpoint_model must be 'raw' or 'ema', got {source!r}")
        return str(source)
    training_config = payload.get("training_config") if isinstance(payload, dict) else None
    if isinstance(training_config, dict) and training_config.get("best_model") == "ema":
        return "ema"
    return "raw"


def _feature_metadata_for_eval(dataset, generator, e0_checkpoint: dict, cache_path: str) -> dict:
    manifest = getattr(dataset, "manifest", None)
    feature_dim = _positive_int_metadata(getattr(manifest, "feature_dim", None), "feature cache manifest feature_dim")
    if getattr(manifest, "l2_normalized", True) is not True:
        raise RuntimeError("Feature cache manifest must declare l2_normalized=true for eval")
    generator_config = getattr(generator, "config", None)
    generator_dim = _positive_int_metadata(getattr(generator_config, "embedding_dim", None), "generator model_config.embedding_dim")
    e0_config = e0_checkpoint.get("model_config") if isinstance(e0_checkpoint, dict) else None
    if not isinstance(e0_config, dict) or "embedding_dim" not in e0_config:
        raise ValueError("E0 checkpoint missing model_config.embedding_dim")
    e0_dim = _positive_int_metadata(e0_config["embedding_dim"], "E0 model_config.embedding_dim")
    if feature_dim != generator_dim or feature_dim != e0_dim:
        raise RuntimeError(
            "feature_dim mismatch: "
            f"cache={feature_dim}, generator={generator_dim}, e0={e0_dim}"
        )
    return {"dim": feature_dim, "l2_normalized": True, "cache": cache_path}


def _positive_int_metadata(value, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer, got bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer, got {value!r}")
    return parsed


def _sample_generated_for_eval(generator, z, sample_ids, sampling_seed: int, image_size: int):
    x_init = make_x_init_for_sample_ids(sample_ids, sampling_seed, image_size, z.device, z.dtype)
    return generator.sample(z, x_init=x_init)


_SAFE_SAMPLE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _generated_image_output_dir(config: dict) -> Path | None:
    generated_image_dir = config.get("generated_image_dir")
    if generated_image_dir is not None and str(generated_image_dir).strip():
        return Path(str(generated_image_dir))
    save_generated_images = config.get("save_generated_images", False)
    if not isinstance(save_generated_images, bool):
        raise ValueError("save_generated_images must be true or false")
    if save_generated_images:
        return Path(config["sample_dir"]) / "generated_images"
    return None


def _safe_sample_id(sample_id) -> str:
    raw = str(sample_id).replace(chr(92), "_").replace("/", "_")
    safe = _SAFE_SAMPLE_ID_RE.sub("_", raw)
    safe = re.sub(r"_+", "_", safe).strip("._-")
    return safe or "sample"


def _save_generated_image_for_eval(image, output_dir: Path, *, global_index: int, sample_id, row: dict) -> Path:
    from torchvision.utils import save_image

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{int(global_index):08d}__{_safe_sample_id(sample_id)}.png"
    if path.exists():
        raise FileExistsError(f"Generated image already exists: {path}")
    save_image(image.detach().cpu(), path)
    row.setdefault("artifacts", {})["generated_image_path"] = str(path)
    return path


def _eval_monitor_configs(config: dict) -> tuple[dict, dict, dict]:
    privacy_cfg = _require_config_block(config, "privacy")
    face_detection_cfg = _require_config_block(config, "face_detection")
    anti_cfg = _require_config_block(config, "anti_steg")
    _validate_privacy_config(privacy_cfg)
    _validate_face_detection_config(face_detection_cfg)
    _validate_anti_steg_config(anti_cfg)
    return privacy_cfg, face_detection_cfg, anti_cfg


def _require_config_block(config: dict, name: str) -> dict:
    if name not in config:
        raise ValueError(f"eval config requires explicit {name} block")
    block = config[name]
    if not isinstance(block, dict):
        raise ValueError(f"eval config {name} block must be a mapping")
    _require_enabled_flag(block, name)
    return block


def _require_enabled_flag(config: dict, context: str) -> bool:
    if "enabled" not in config:
        raise ValueError(f"{context}.enabled is required")
    enabled = config["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError(f"{context}.enabled must be true or false")
    return enabled


def _require_fields(config: dict, fields: tuple[str, ...], context: str) -> None:
    for field in fields:
        if field not in config:
            raise ValueError(f"{context}.{field} is required")


def _validate_privacy_config(config: dict) -> None:
    if not _require_enabled_flag(config, "privacy"):
        return
    if "recognizers" not in config:
        raise ValueError("privacy.recognizers is required")
    if not config["recognizers"]:
        raise ValueError("privacy.recognizers must not be empty when privacy is enabled")
    validate_recognizer_configs(config["recognizers"])


def _validate_face_detection_config(config: dict) -> None:
    if not _require_enabled_flag(config, "face_detection"):
        return
    _require_fields(config, ("model_name", "threshold", "latent_cosine_threshold"), "face_detection")


def _validate_anti_steg_config(config: dict) -> None:
    if not _require_enabled_flag(config, "anti_steg"):
        return
    _require_fields(
        config,
        ("jpeg_quality", "blur_radius", "downsample_scale", "crop_fraction", "noise_std"),
        "anti_steg",
    )


def _build_face_detector(config: dict, device: str):
    if not _require_enabled_flag(config, "face_detection"):
        return None
    return InsightFaceDetector(str(config["model_name"]), device)


def _empty_privacy_store(recognizers, perturbations):
    store = {}
    for recognizer in recognizers:
        store[recognizer.name] = {"source": [], "generated": {"clean": []}}
        for name in perturbations:
            store[recognizer.name]["generated"][name] = []
    return store


def _make_affective_rows(sample_ids, labels, source_out, generated_out, z) -> list[dict]:
    import torch
    import torch.nn.functional as F

    cosine = F.cosine_similarity(generated_out["embedding"], z, dim=1).clamp(-1, 1)
    angle = torch.acos(cosine)
    generated_pred = generated_out["logits"].argmax(dim=1)
    source_pred = source_out["logits"].argmax(dim=1)
    drift = (generated_out["logits"] - source_out["logits"]).float().norm(dim=1)
    rows = []
    for i, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": str(sample_id),
                "label": int(labels[i].detach().cpu()),
                "affective": {
                    "latent_cosine": float(cosine[i].detach().cpu()),
                    "latent_angle_rad": float(angle[i].detach().cpu()),
                    "label_accuracy_generated": float((generated_pred[i] == labels[i]).detach().cpu()),
                    "source_prediction_preserved": float((generated_pred[i] == source_pred[i]).detach().cpu()),
                    "logit_l2_drift": float(drift[i].detach().cpu()),
                },
                "face_detection": {},
                "anti_steg": {},
                "privacy": {},
            }
        )
    return rows


def _attach_face_detection_rows(rows: list[dict], counts: list[int]) -> None:
    if len(rows) != len(counts):
        raise RuntimeError(f"Face detection count mismatch: rows={len(rows)} counts={len(counts)}")
    for row, count in zip(rows, counts):
        rates = face_count_rates([count])
        row["face_detection"] = {
            "count": int(count),
            "detected": rates["face_detect_ge1_rate"],
            **rates,
        }


def _attach_perturbed_affective_rows(rows: list[dict], row_start: int, name: str, perturbed_out, z) -> None:
    import torch
    import torch.nn.functional as F

    cosine = F.cosine_similarity(perturbed_out["embedding"], z, dim=1).clamp(-1, 1)
    angle = torch.acos(cosine)
    for i in range(cosine.shape[0]):
        rows[row_start + i]["anti_steg"][name] = {
            "latent_cosine": float(cosine[i].detach().cpu()),
            "latent_angle_rad": float(angle[i].detach().cpu()),
        }


def _collect_privacy_embeddings(store: dict, recognizers, source, generated, variant: str) -> None:
    for recognizer in recognizers:
        if variant == "clean":
            store[recognizer.name]["source"].append(recognizer.embed(source).detach().cpu())
        store[recognizer.name]["generated"][variant].append(recognizer.embed(generated).detach().cpu())


def _run_privacy_pass(config: dict, loader, generated_chunks, recognizers, perturbations, privacy_store: dict, device) -> None:
    import torch

    if generated_chunks is None:
        raise RuntimeError("Privacy pass requires cached generated images from the guard pass")
    batch_count = 0
    with torch.no_grad():
        for batch, generated_cpu in zip(loader, generated_chunks):
            batch_count += 1
            source = batch["image"].to(device, non_blocking=True)
            generated = generated_cpu.to(device=device, dtype=source.dtype)
            _collect_privacy_embeddings(privacy_store, recognizers, source, generated, variant="clean")
            for name, perturb in perturbations.items():
                perturbed = perturb(generated)
                assert_finite_tensor(f"privacy_perturbed_{name}", perturbed)
                _collect_privacy_embeddings(privacy_store, recognizers, source, perturbed, variant=name)
    if batch_count != len(generated_chunks) or batch_count != len(loader):
        raise RuntimeError(f"Privacy pass batch mismatch: loader={len(loader)} generated_chunks={len(generated_chunks)} consumed={batch_count}")


def deterministic_impostor_indices(num_samples: int) -> list[int]:
    if num_samples < 2:
        raise ValueError("Privacy evaluation needs at least 2 samples for an impostor baseline")
    offset = max(1, num_samples // 2)
    return [int((index + offset) % num_samples) for index in range(num_samples)]


def _attach_privacy_rows(rows: list[dict], store: dict) -> None:
    import torch
    import torch.nn.functional as F

    impostor_indices = torch.tensor(deterministic_impostor_indices(len(rows)), dtype=torch.long)
    for recognizer_name, payload in store.items():
        source = torch.cat(payload["source"], dim=0)
        if source.shape[0] != len(rows):
            raise RuntimeError(f"Recognizer {recognizer_name} source embedding count mismatch")
        clean_same = None
        for variant, chunks in payload["generated"].items():
            generated = torch.cat(chunks, dim=0)
            if generated.shape[0] != len(rows):
                raise RuntimeError(f"Recognizer {recognizer_name} generated embedding count mismatch for {variant}")
            same = F.cosine_similarity(source, generated, dim=1)
            impostor = F.cosine_similarity(source, generated[impostor_indices], dim=1)
            if variant == "clean":
                clean_same = same
                same_key = "same_similarity"
                impostor_key = "impostor_similarity"
            else:
                same_key = f"same_similarity_{variant}"
                impostor_key = f"impostor_similarity_{variant}"
            for i, row in enumerate(rows):
                privacy = row["privacy"].setdefault(recognizer_name, {})
                privacy[same_key] = float(same[i])
                privacy[impostor_key] = float(impostor[i])
                if variant != "clean":
                    if clean_same is None:
                        raise RuntimeError(f"Clean privacy scores must be computed before perturbation {variant}")
                    delta = same[i] - clean_same[i]
                    privacy[f"same_similarity_delta_{variant}"] = float(delta)
                    privacy[f"same_similarity_rebound_{variant}"] = float(delta > 0.0)


def _privacy_score_array(values, context: str):
    import numpy as np

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"Privacy ROC metrics require non-empty {context}")
    if not np.isfinite(array).all():
        raise ValueError(f"Privacy ROC metrics require finite {context}")
    return array


def _tar_at_far(same_scores, impostor_scores, target_far: float) -> float:
    import numpy as np

    thresholds = [float("inf"), float(np.nextafter(impostor_scores.max(), np.inf))]
    thresholds.extend(float(value) for value in np.unique(impostor_scores)[::-1])
    best_tar = 0.0
    for threshold in thresholds:
        far = float(np.mean(impostor_scores >= threshold))
        if far <= target_far + 1e-12:
            best_tar = max(best_tar, float(np.mean(same_scores >= threshold)))
    return best_tar


def _privacy_auc(same_scores, impostor_scores) -> float:
    import numpy as np

    sorted_impostor = np.sort(impostor_scores)
    wins = np.searchsorted(sorted_impostor, same_scores, side="left")
    ties = np.searchsorted(sorted_impostor, same_scores, side="right") - wins
    return float(np.sum(wins + 0.5 * ties) / (same_scores.size * impostor_scores.size))


def _privacy_eer(same_scores, impostor_scores) -> float:
    import numpy as np

    unique_scores = np.unique(np.concatenate([same_scores, impostor_scores]))[::-1]
    thresholds = [float("inf")]
    for value in unique_scores:
        thresholds.append(float(np.nextafter(value, np.inf)))
        thresholds.append(float(value))
    thresholds.append(float("-inf"))
    best = 1.0
    best_gap = float("inf")
    for threshold in thresholds:
        far = float(np.mean(impostor_scores >= threshold))
        fnr = float(np.mean(same_scores < threshold))
        gap = abs(far - fnr)
        if gap < best_gap:
            best_gap = gap
            best = (far + fnr) / 2.0
    return float(best)


def _privacy_roc_metrics(same_values, impostor_values) -> dict[str, float]:
    same_scores = _privacy_score_array(same_values, "same_similarity")
    impostor_scores = _privacy_score_array(impostor_values, "impostor_similarity")
    return {
        "same_identity_similarity_mean": float(same_scores.mean()),
        "tar_at_far_1e-3": _tar_at_far(same_scores, impostor_scores, 1e-3),
        "tar_at_far_1e-4": _tar_at_far(same_scores, impostor_scores, 1e-4),
        "eer": _privacy_eer(same_scores, impostor_scores),
        "auc": _privacy_auc(same_scores, impostor_scores),
    }


def _summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize zero eval rows")
    affective_keys = rows[0]["affective"].keys()
    summarized = {key: summarize(row["affective"][key] for row in rows) for key in affective_keys}
    # Optional: rows omit face_detection when the face detector monitor is disabled.
    face_keys = sorted({key for row in rows for key in row.get("face_detection", {})})
    summarized["face_detection"] = {
        key: summarize(row["face_detection"][key] for row in rows if key in row.get("face_detection", {}))
        for key in face_keys
    }
    anti_names = sorted({name for row in rows for name in row["anti_steg"]})
    summarized["anti_steg"] = {
        name: {
            metric: summarize(row["anti_steg"][name][metric] for row in rows)
            for metric in rows[0]["anti_steg"][name].keys()
        }
        for name in anti_names
    }
    recognizer_names = sorted({name for row in rows for name in row["privacy"]})
    summarized["privacy"] = {}
    for recognizer_name in recognizer_names:
        metric_names = sorted({metric for row in rows for metric in row["privacy"].get(recognizer_name, {})})
        if "same_similarity" not in metric_names or "impostor_similarity" not in metric_names:
            raise ValueError(
                f"Privacy ROC metrics for {recognizer_name} require both same_similarity and impostor_similarity"
            )
        summarized["privacy"][recognizer_name] = {
            metric: summarize(row["privacy"][recognizer_name][metric] for row in rows)
            for metric in metric_names
        }
        try:
            same_values = [row["privacy"][recognizer_name]["same_similarity"] for row in rows]
            impostor_values = [row["privacy"][recognizer_name]["impostor_similarity"] for row in rows]
        except KeyError as exc:
            raise ValueError(
                f"Privacy ROC metrics for {recognizer_name} require both same_similarity and impostor_similarity"
            ) from exc
        summarized["privacy"][recognizer_name].update(_privacy_roc_metrics(same_values, impostor_values))
    return summarized


def _require_summary_mean(metrics: dict, field_path: tuple[str, ...]) -> float:
    context = ".".join(field_path)
    current = metrics
    for field in field_path:
        if not isinstance(current, dict) or field not in current:
            raise RuntimeError(f"Face detection guard requires {context}")
        current = current[field]
    if isinstance(current, bool):
        raise RuntimeError(f"Face detection guard requires numeric {context}, got bool")
    try:
        return float(current)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Face detection guard requires numeric {context}, got {current!r}") from exc


def _guard_result(metrics: dict, config: dict) -> dict:
    if not _require_enabled_flag(config, "face_detection"):
        return {"enabled": False, "passed": True, "reason": "disabled"}
    _validate_face_detection_config(config)
    face_threshold = float(config["threshold"])
    cosine_threshold = float(config["latent_cosine_threshold"])
    face_detection_rate = _require_summary_mean(metrics, ("face_detection", "detected", "mean"))
    face_rates = {
        "face_detect_ge1_rate": _require_summary_mean(metrics, ("face_detection", "face_detect_ge1_rate", "mean")),
        "single_face_eq1_rate": _require_summary_mean(metrics, ("face_detection", "single_face_eq1_rate", "mean")),
        "zero_face_rate": _require_summary_mean(metrics, ("face_detection", "zero_face_rate", "mean")),
        "multi_face_rate": _require_summary_mean(metrics, ("face_detection", "multi_face_rate", "mean")),
    }
    latent_cosine_mean = _require_summary_mean(metrics, ("latent_cosine", "mean"))
    passed = bool(face_detection_rate >= face_threshold and latent_cosine_mean >= cosine_threshold)
    return {
        "enabled": True,
        "passed": passed,
        "face_detection_rate": face_detection_rate,
        **face_rates,
        "latent_cosine_mean": latent_cosine_mean,
        "face_detection_threshold": face_threshold,
        "latent_cosine_threshold": cosine_threshold,
    }
