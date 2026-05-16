from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

from safa.data.feature_dataset import FeatureAlignedAffectNet
from safa.evaluation.metrics import flatten_finite_numbers, summarize
from safa.evaluation.perturbations import perturbation_map
from safa.evaluation.recognizers import build_recognizers, describe_recognizer_assets
from safa.models.e0 import freeze_e0, load_e0_checkpoint
from safa.models.generator import ZOnlyGenerator
from safa.training.losses import normalize_for_e0
from safa.training.transforms import generator_image_transform
from safa.utils.device import assert_finite_tensor, require_cuda_device
from safa.utils.hashing import sha256_file
from safa.utils.seed import set_seed


def run_eval_from_config(config: dict) -> dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    set_seed(int(config["seed"]))
    device = require_cuda_device(str(config["device"]))
    e0, _ = load_e0_checkpoint(config["e0_checkpoint"], device=str(device))
    e0.to(device)
    freeze_e0(e0)
    generator = _load_generator(config["g_checkpoint"], config, str(device))
    dataset = FeatureAlignedAffectNet(
        config["index"],
        config["features"],
        config["e0_checkpoint"],
        transform=generator_image_transform(int(config["image_size"])),
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config["num_workers"]), pin_memory=True)
    privacy_cfg = config.get("privacy", {"enabled": False})
    recognizer_assets = describe_recognizer_assets(privacy_cfg["recognizers"]) if privacy_cfg.get("enabled") else []
    recognizers = build_recognizers(privacy_cfg["recognizers"], str(device)) if privacy_cfg.get("enabled") else []
    anti_cfg = config.get("anti_steg", {"enabled": False})
    perturbations = perturbation_map(anti_cfg, int(config["seed"])) if anti_cfg.get("enabled") else {}

    rows: list[dict] = []
    privacy_store = _empty_privacy_store(recognizers, perturbations)
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
            generated = generator(z)
            assert_finite_tensor("eval_generated", generated)
            source_out = e0(normalize_for_e0(source))
            generated_out = e0(normalize_for_e0(generated))
            batch_rows = _make_affective_rows(sample_ids, labels, source_out, generated_out, z)
            rows.extend(batch_rows)
            row_start = len(rows) - len(batch_rows)
            if recognizers:
                _collect_privacy_embeddings(privacy_store, recognizers, source, generated, variant="clean")
            for name, perturb in perturbations.items():
                perturbed = perturb(generated)
                assert_finite_tensor(f"perturbed_{name}", perturbed)
                perturbed_out = e0(normalize_for_e0(perturbed))
                _attach_perturbed_affective_rows(rows, row_start, name, perturbed_out, z)
                if recognizers:
                    _collect_privacy_embeddings(privacy_store, recognizers, source, perturbed, variant=name)
            if saved_samples < 16:
                save_image(generated[: min(4, generated.shape[0])].detach().cpu(), sample_dir / f"generated_{saved_samples:04d}.png", nrow=4)
                saved_samples += int(min(4, generated.shape[0]))
    if recognizers:
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
        "features": {"dim": 512, "l2_normalized": True, "cache": config["features"]},
        "recognizer_assets": recognizer_assets,
        "metrics": summarized,
        "artifacts": {"sample_dir": str(sample_dir), "per_sample_jsonl": config["per_sample_jsonl"]},
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
    return result


def _load_generator(checkpoint_path: str, config: dict, device: str):
    import torch

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Generator checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device)
    model_config = payload.get("model_config", {})
    generator = ZOnlyGenerator(
        embedding_dim=int(model_config.get("embedding_dim", config.get("embedding_dim", 512))),
        image_size=int(model_config.get("image_size", config["image_size"])),
    )
    generator.load_state_dict(payload["model_state_dict"])
    return generator.to(device).eval()


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
                "anti_steg": {},
                "privacy": {},
            }
        )
    return rows


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
