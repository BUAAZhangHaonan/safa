from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

from safa.data.feature_dataset import FeatureAlignedAffectNet
from safa.evaluation.metrics import flatten_finite_numbers, summarize
from safa.evaluation.perturbations import perturbation_map
from safa.evaluation.recognizers import InsightFaceDetector, build_recognizers, describe_recognizer_assets
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
    privacy_cfg = config.get("privacy", {"enabled": False})
    face_detection_cfg = config.get("face_detection", _default_face_detection_config(privacy_cfg))
    detector = _build_face_detector(face_detection_cfg, str(device))
    recognizer_assets = []
    anti_cfg = config.get("anti_steg", {"enabled": False})
    perturbations = perturbation_map(anti_cfg, int(config["seed"])) if anti_cfg.get("enabled") else {}

    rows: list[dict] = []
    generated_chunks = [] if privacy_cfg.get("enabled") else None
    sample_dir = Path(config["sample_dir"])
    sample_dir.mkdir(parents=True, exist_ok=True)
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
    privacy_skipped = bool(privacy_cfg.get("enabled") and not guard["passed"])
    if privacy_cfg.get("enabled") and guard["passed"]:
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
        "artifacts": {"sample_dir": str(sample_dir), "per_sample_jsonl": config["per_sample_jsonl"]},
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
            f"face_detection_rate={guard.get('face_detection_rate')} "
            f"latent_cosine_mean={guard.get('latent_cosine_mean')} "
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
    generator = build_generator(model_config)
    generator.load_state_dict(payload["model_state_dict"])
    return generator.to(device).eval()


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


def _default_face_detection_config(privacy_cfg: dict) -> dict:
    return {"enabled": bool(privacy_cfg.get("enabled")), "model_name": "buffalo_l", "threshold": 0.95, "latent_cosine_threshold": 0.95}


def _build_face_detector(config: dict, device: str):
    if not config.get("enabled", False):
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
        row["face_detection"] = {"count": int(count), "detected": float(count >= 1)}


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


def _summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize zero eval rows")
    affective_keys = rows[0]["affective"].keys()
    summarized = {key: summarize(row["affective"][key] for row in rows) for key in affective_keys}
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
        summarized["privacy"][recognizer_name] = {
            metric: summarize(row["privacy"][recognizer_name][metric] for row in rows)
            for metric in metric_names
        }
    return summarized


def _guard_result(metrics: dict, config: dict) -> dict:
    if not config.get("enabled", False):
        return {"enabled": False, "passed": True, "reason": "disabled"}
    face_threshold = float(config.get("threshold", 0.95))
    cosine_threshold = float(config.get("latent_cosine_threshold", 0.95))
    face_detection_rate = metrics.get("face_detection", {}).get("detected", {}).get("mean")
    latent_cosine_mean = metrics.get("latent_cosine", {}).get("mean")
    if face_detection_rate is None:
        raise RuntimeError("Face detection guard is enabled but no face_detection metrics were produced")
    if latent_cosine_mean is None:
        raise RuntimeError("Face detection guard is enabled but no latent_cosine metrics were produced")
    passed = bool(face_detection_rate >= face_threshold and latent_cosine_mean >= cosine_threshold)
    return {
        "enabled": True,
        "passed": passed,
        "face_detection_rate": face_detection_rate,
        "latent_cosine_mean": latent_cosine_mean,
        "face_detection_threshold": face_threshold,
        "latent_cosine_threshold": cosine_threshold,
    }
