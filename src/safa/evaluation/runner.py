from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json

from safa.data.feature_dataset import FeatureAlignedAffectNet
from safa.evaluation.metrics import flatten_finite_numbers, summarize
from safa.evaluation.perturbations import perturbation_map
from safa.evaluation.recognizers import build_recognizers
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
    recognizers = build_recognizers(privacy_cfg["recognizers"], str(device)) if privacy_cfg.get("enabled") else []
    anti_cfg = config.get("anti_steg", {"enabled": False})
    perturbations = perturbation_map(anti_cfg, int(config["seed"])) if anti_cfg.get("enabled") else {}

    metrics = _empty_metrics(recognizers, perturbations)
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
            generated = generator(z)
            assert_finite_tensor("eval_generated", generated)
            source_out = e0(normalize_for_e0(source))
            generated_out = e0(normalize_for_e0(generated))
            _collect_affective(metrics, source_out, generated_out, z, labels)
            if recognizers:
                _collect_privacy(metrics, recognizers, source, generated, suffix="")
            for name, perturb in perturbations.items():
                perturbed = perturb(generated)
                assert_finite_tensor(f"perturbed_{name}", perturbed)
                perturbed_out = e0(normalize_for_e0(perturbed))
                _collect_perturbed_affective(metrics, name, perturbed_out, z)
                if recognizers:
                    _collect_privacy(metrics, recognizers, source, perturbed, suffix=f"_{name}")
            if saved_samples < 16:
                save_image(generated[: min(4, generated.shape[0])].detach().cpu(), sample_dir / f"generated_{saved_samples:04d}.png", nrow=4)
                saved_samples += int(min(4, generated.shape[0]))
    summarized = _summarize_metrics(metrics)
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
        "metrics": summarized,
        "artifacts": {"sample_dir": str(sample_dir)},
    }
    flatten_finite_numbers(result["metrics"])
    out_json = Path(config["out_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
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


def _empty_metrics(recognizers, perturbations):
    metrics = {
        "latent_cosine": [],
        "latent_angle_rad": [],
        "label_accuracy_generated": [],
        "source_prediction_preserved": [],
        "logit_l2_drift": [],
        "anti_steg": {name: {"latent_cosine": [], "latent_angle_rad": []} for name in perturbations},
        "privacy": {},
    }
    for recognizer in recognizers:
        metrics["privacy"][recognizer.name] = {"same_similarity": [], "impostor_similarity": []}
        for name in perturbations:
            metrics["privacy"][recognizer.name][f"same_similarity_{name}"] = []
            metrics["privacy"][recognizer.name][f"impostor_similarity_{name}"] = []
    return metrics


def _collect_affective(metrics, source_out, generated_out, z, labels):
    import torch
    import torch.nn.functional as F

    cosine = F.cosine_similarity(generated_out["embedding"], z, dim=1).clamp(-1, 1)
    angle = torch.acos(cosine)
    metrics["latent_cosine"].extend([float(item) for item in cosine.detach().cpu()])
    metrics["latent_angle_rad"].extend([float(item) for item in angle.detach().cpu()])
    generated_pred = generated_out["logits"].argmax(dim=1)
    source_pred = source_out["logits"].argmax(dim=1)
    metrics["label_accuracy_generated"].extend([float(item) for item in (generated_pred == labels).detach().cpu()])
    metrics["source_prediction_preserved"].extend([float(item) for item in (generated_pred == source_pred).detach().cpu()])
    drift = (generated_out["logits"] - source_out["logits"]).float().norm(dim=1)
    metrics["logit_l2_drift"].extend([float(item) for item in drift.detach().cpu()])


def _collect_perturbed_affective(metrics, name, perturbed_out, z):
    import torch
    import torch.nn.functional as F

    cosine = F.cosine_similarity(perturbed_out["embedding"], z, dim=1).clamp(-1, 1)
    angle = torch.acos(cosine)
    metrics["anti_steg"][name]["latent_cosine"].extend([float(item) for item in cosine.detach().cpu()])
    metrics["anti_steg"][name]["latent_angle_rad"].extend([float(item) for item in angle.detach().cpu()])


def _collect_privacy(metrics, recognizers, source, generated, suffix: str):
    import torch
    import torch.nn.functional as F

    for recognizer in recognizers:
        source_emb = recognizer.embed(source)
        generated_emb = recognizer.embed(generated)
        same = F.cosine_similarity(source_emb, generated_emb, dim=1)
        if generated_emb.shape[0] < 2:
            raise ValueError("Privacy evaluation needs batch_size >= 2 for impostor baseline")
        impostor = F.cosine_similarity(source_emb, torch.roll(generated_emb, shifts=1, dims=0), dim=1)
        metrics["privacy"][recognizer.name][f"same_similarity{suffix}"].extend([float(item) for item in same.detach().cpu()])
        metrics["privacy"][recognizer.name][f"impostor_similarity{suffix}"].extend([float(item) for item in impostor.detach().cpu()])


def _summarize_metrics(metrics: dict) -> dict:
    summarized = {}
    for key, value in metrics.items():
        if key == "anti_steg":
            summarized[key] = {name: {metric: summarize(values) for metric, values in payload.items()} for name, payload in value.items()}
        elif key == "privacy":
            summarized[key] = {
                recognizer: {metric: summarize(values) for metric, values in payload.items()} for recognizer, payload in value.items()
            }
        else:
            summarized[key] = summarize(value)
    return summarized

