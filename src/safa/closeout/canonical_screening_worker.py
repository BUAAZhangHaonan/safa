"""Independent worker for one immutable canonical screening request."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
    build_run_claim,
    build_run_result,
    canonical_json,
    load_json,
    load_jsonl,
    sha256_file,
    validate_candidate_manifest,
    validate_checkpoint_plan,
    validate_policy,
    validate_run_request,
    validate_run_result,
    write_exclusive_json,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_bound_file(value: Mapping[str, Any], label: str) -> Path:
    path = Path(str(value.get("path", ""))).resolve()
    expected = value.get("sha256")
    if not path.is_file() or sha256_file(path) != expected:
        raise CanonicalScreeningError(f"{label} file binding mismatch: {path}")
    return path


def _request_repo_root(request: Mapping[str, Any]) -> Path:
    worker = _assert_bound_file(request["screening_worker"], "screening worker")
    try:
        root = worker.parents[3]
    except IndexError as exc:
        raise CanonicalScreeningError(
            "screening worker path cannot determine repository root"
        ) from exc
    if not (root / "pyproject.toml").is_file():
        raise CanonicalScreeningError(
            "screening worker binding is not under a SAFA repository"
        )
    return root


def _ordered_manifest_ids(binding: Mapping[str, Any]) -> list[str]:
    path = _assert_bound_file(binding, "sample manifest")
    rows = load_jsonl(path, "sample manifest")
    ids = [row.get("sample_id") for row in rows]
    if (
        any(not isinstance(item, str) or not item for item in ids)
        or len(ids) != len(set(ids))
        or len(ids) != binding.get("sample_count")
    ):
        raise CanonicalScreeningError("sample manifest coverage differs")
    return [str(item) for item in ids]


def _mean(values: Sequence[float], label: str) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise CanonicalScreeningError(f"{label} values are empty or non-finite")
    return float(statistics.fmean(values))


def _assert_runtime_cuda_binding(
    request: Mapping[str, Any],
    physical_gpu_index: int,
    physical_gpu_uuid: str,
) -> dict[str, Any]:
    registry = {
        row["physical_gpu_index"]: row["physical_gpu_uuid"]
        for row in request["authorized_gpu_registry"]
    }
    expected_uuid = registry.get(physical_gpu_index)
    if expected_uuid is None or physical_gpu_uuid != expected_uuid:
        raise CanonicalScreeningError(
            "worker physical GPU index/UUID differs from admission"
        )
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise CanonicalScreeningError(
            "worker CUDA_DEVICE_ORDER must be PCI_BUS_ID"
        )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != expected_uuid:
        raise CanonicalScreeningError(
            "worker CUDA_VISIBLE_DEVICES differs from the authorized GPU UUID"
        )

    import torch

    if torch.cuda.device_count() != 1:
        raise CanonicalScreeningError(
            "worker CUDA visibility must contain exactly one device"
        )
    properties = torch.cuda.get_device_properties(0)
    runtime_uuid = properties.uuid
    if isinstance(runtime_uuid, bytes):
        runtime_uuid = runtime_uuid.decode("ascii")
    runtime_uuid = str(runtime_uuid)
    if runtime_uuid != expected_uuid:
        raise CanonicalScreeningError(
            "worker runtime CUDA UUID differs from the authorized GPU UUID"
        )
    torch.cuda.set_device(0)
    return {
        "physical_gpu_index": physical_gpu_index,
        "physical_gpu_uuid": expected_uuid,
        "logical_cuda_index": 0,
        "runtime_cuda_uuid": runtime_uuid,
        "cuda_visible_devices": visible_devices,
    }


def _load_arcface_contract(request: Mapping[str, Any]) -> dict[str, Any]:
    from safa.evaluation.r9_evaluator_worker import _validate_arcface_contract

    declared = dict(request["arcface"])
    probe_binding = dict(declared.pop("execution_probe"))
    probe_path = _assert_bound_file(probe_binding, "ArcFace execution probe")
    probe = load_json(probe_path, "ArcFace execution probe")
    execution = probe.get("execution")
    if not isinstance(execution, Mapping):
        raise CanonicalScreeningError("ArcFace execution probe omits execution")
    declared["execution"] = dict(execution)
    declared["execution_probe"] = probe_binding
    source_root = Path(str(request["source_index"]["path"])).resolve()
    repo_root = next(
        (
            parent
            for parent in source_root.parents
            if (parent / "pyproject.toml").is_file()
        ),
        None,
    )
    if repo_root is None:
        raise CanonicalScreeningError("cannot resolve repository root from source index")
    return _validate_arcface_contract(declared, repo_root=repo_root)


def _embedding_cosine(left: Any, right: Any) -> float:
    import numpy as np

    lhs = np.asarray(left, dtype=np.float64).reshape(-1)
    rhs = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise CanonicalScreeningError("ArcFace embedding norm is invalid")
    value = float(np.dot(lhs, rhs) / denominator)
    if not math.isfinite(value):
        raise CanonicalScreeningError("ArcFace cosine is non-finite")
    return value


def _tensor_sample_sha256(value: Any) -> str:
    import torch

    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_source_pixel_batch(
    source_paths: Sequence[Path], image_size: int, device: str
):
    import numpy as np
    from PIL import Image
    import torch

    tensors = []
    for path in source_paths:
        with Image.open(path) as image:
            array = np.asarray(
                image.convert("RGB").resize(
                    (image_size, image_size), resample=Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
        tensors.append(torch.from_numpy(array).permute(2, 0, 1) / 255.0)
    return torch.stack(tensors, dim=0).to(device)


def _representation_cosines(
    generated_e0: Any,
    target_z: Any,
    generated_edev: Any,
    source_edev: Any,
) -> tuple[Any, Any]:
    import torch.nn.functional as F

    if generated_e0.shape != target_z.shape:
        raise CanonicalScreeningError(
            "generated E0 embedding and locked conditioning target z differ in shape"
        )
    return (
        F.cosine_similarity(generated_e0, target_z, dim=1),
        F.cosine_similarity(generated_edev, source_edev, dim=1),
    )


def _run_generation(
    request: Mapping[str, Any], gpu_index: int, output_dir: Path
) -> tuple[list[dict[str, Any]], list[Path], list[Path]]:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.utils import save_image

    from safa.data.feature_dataset import FeatureAlignedAffectNet
    from safa.evaluation.checkpoint_preflight import strict_load_generator_checkpoint
    from safa.closeout.generator_output_contract import (
        build_bound_decoder,
        validate_native_generator_output,
        validate_rgb_unit_interval,
    )
    from safa.models.e0 import freeze_e0, load_e0_checkpoint
    from safa.models.generator import generator_sample_channels
    from safa.training.losses import normalize_for_e0
    from safa.training.transforms import generator_image_transform
    from safa.utils.device import assert_finite_tensor
    from safa.utils.sampling import make_x_init_for_sample_ids
    from safa.utils.seed import set_seed

    device = "cuda:0"
    if not torch.cuda.is_available():
        raise CanonicalScreeningError("CUDA is required for canonical screening")
    set_seed(4549)
    checkpoint = dict(request["candidate"])
    checkpoint_path = Path(str(checkpoint["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = _request_repo_root(request) / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise CanonicalScreeningError(f"candidate checkpoint does not exist: {checkpoint_path}")
    generator, preflight = strict_load_generator_checkpoint(
        checkpoint_path,
        str(checkpoint["checkpoint_model"]),
        device,
        expected_checkpoint_sha256=str(checkpoint["checkpoint_sha256"]),
        compute_sha256=True,
        smoke_samples=0,
        output_decoder_registry=request["output_decoder_registry"],
    )
    if preflight["status"] != "valid" or preflight["sha256_binding"] != "expected_exact":
        raise CanonicalScreeningError("worker checkpoint preflight is not exact and valid")
    if preflight["output_contract"] != request["output_contract"]:
        raise CanonicalScreeningError(
            "worker checkpoint output contract differs from run request"
        )
    generator.eval()
    codec = build_bound_decoder(
        request["output_contract"],
        request["output_decoder_registry"],
        device,
    )

    e0_path = _assert_bound_file(request["e0"], "E0")
    edev_path = _assert_bound_file(request["edev"], "Edev")
    e0, _ = load_e0_checkpoint(e0_path, device="cpu")
    edev, _ = load_e0_checkpoint(edev_path, device="cpu")
    for encoder in (e0, edev):
        encoder.to(device)
        freeze_e0(encoder)
        encoder.eval()

    source_index = _assert_bound_file(request["source_index"], "source index")
    feature_directory = Path(str(request["features"]["directory"])).resolve()
    _assert_bound_file(request["features"]["manifest"], "feature manifest")
    _assert_bound_file(request["features"]["shard"], "feature shard")
    dataset = FeatureAlignedAffectNet(
        source_index,
        feature_directory,
        e0_path,
        transform=generator_image_transform(224),
    )
    manifest_ids = _ordered_manifest_ids(request["sample_manifest"])
    position = {record.sample_id: index for index, record in enumerate(dataset.records)}
    if any(sample_id not in position for sample_id in manifest_ids):
        missing = [sample_id for sample_id in manifest_ids if sample_id not in position]
        raise CanonicalScreeningError(f"manifest IDs absent from source index: {missing[:8]}")
    subset = Subset(dataset, [position[sample_id] for sample_id in manifest_ids])
    loader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    image_dir = output_dir / "generated"
    image_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    candidate_paths: list[Path] = []
    source_paths: list[Path] = []
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device, non_blocking=True)
            sample_ids = [str(item) for item in batch["sample_id"]]
            config = getattr(generator, "config", None)
            channels = generator_sample_channels(config) if config is not None else 3
            image_size = int(config.image_size) if config is not None else 224
            x_init = make_x_init_for_sample_ids(
                sample_ids,
                4549,
                image_size,
                z.device,
                z.dtype,
                channels=channels,
            )
            generated_native = generator.sample(
                z,
                x_init=x_init,
                clamp_output=True,
            )
            validate_native_generator_output(
                generated_native,
                request["output_contract"],
                request["output_decoder_registry"],
            )
            native_sha256 = [
                _tensor_sample_sha256(generated_native[index])
                for index in range(len(sample_ids))
            ]
            generated = (
                generated_native
                if codec is None
                else codec.decode(generated_native)
            )
            validate_rgb_unit_interval(
                generated,
                "canonical screening RGB before E0/Edev/quality",
                request["output_contract"]["rgb_contract"],
            )
            assert_finite_tensor("canonical_screening_generated", generated)
            generated_e0 = e0(normalize_for_e0(generated))["embedding"]
            batch_source_paths = [
                Path(dataset.records[position[sample_id]].image_path).resolve()
                for sample_id in sample_ids
            ]
            source_pixels = _load_source_pixel_batch(
                batch_source_paths,
                int(request["pixel_image_size"]),
                device,
            )
            source_edev = edev(normalize_for_e0(source_pixels))["embedding"]
            generated_edev = edev(normalize_for_e0(generated))["embedding"]
            e0_cosine, edev_cosine = _representation_cosines(
                generated_e0, z, generated_edev, source_edev
            )
            assert_finite_tensor("canonical_screening_e0_cosine", e0_cosine)
            assert_finite_tensor("canonical_screening_edev_cosine", edev_cosine)
            for local_index, sample_id in enumerate(sample_ids):
                global_index = len(rows)
                candidate_path = image_dir / f"{global_index:06d}.png"
                save_image(generated[local_index].detach().cpu(), candidate_path)
                if not candidate_path.is_file():
                    raise CanonicalScreeningError("generated image was not materialized")
                record = dataset.records[position[sample_id]]
                source_path = Path(record.image_path).resolve()
                candidate_paths.append(candidate_path.resolve())
                source_paths.append(source_path)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "run_request_sha256": request["run_request_sha256"],
                        "checkpoint_sha256": request["candidate"][
                            "checkpoint_sha256"
                        ],
                        "checkpoint_model": request["candidate"][
                            "checkpoint_model"
                        ],
                        "source_path": str(source_path),
                        "source_sha256": sha256_file(source_path),
                        "candidate_path": str(candidate_path.resolve()),
                        "candidate_sha256": sha256_file(candidate_path),
                        "output_contract_sha256": request["output_contract"][
                            "output_contract_sha256"
                        ],
                        "output_contract_type": request["output_contract"][
                            "contract_type"
                        ],
                        "decoder_registry_sha256": request[
                            "output_decoder_registry"
                        ]["decoder_registry_sha256"],
                        "output_space": request["output_contract"]["capability"][
                            "output_space"
                        ],
                        "native_output_sha256": native_sha256[local_index],
                        "native_output_shape": list(
                            generated_native[local_index].shape
                        ),
                        "native_rgb_shape": list(generated[local_index].shape),
                        "native_rgb_size": list(request["native_rgb_size"]),
                        "quality_protocol_family": request[
                            "quality_protocol_family"
                        ],
                        "nfe": request["nfe"],
                        "e0_cosine": float(e0_cosine[local_index].cpu()),
                        "edev_cosine": float(edev_cosine[local_index].cpu()),
                    }
                )
    if [row["sample_id"] for row in rows] != manifest_ids:
        raise CanonicalScreeningError("generation rows do not exactly follow the manifest")
    return rows, source_paths, candidate_paths


def _run_arcface(
    request: Mapping[str, Any],
    gpu_index: int,
    rows: list[dict[str, Any]],
    source_paths: Sequence[Path],
    candidate_paths: Sequence[Path],
) -> dict[str, Any]:
    from safa.evaluation.r9_evaluator_worker import (
        _arcface_observation,
        _production_face_analyzer_factory,
    )

    contract = _load_arcface_contract(request)
    analyzer = _production_face_analyzer_factory(contract, "cuda:0")
    complete = 0
    values: list[float] = []
    for row, source, candidate in zip(
        rows, source_paths, candidate_paths, strict=True
    ):
        source_count, source_embedding = _arcface_observation(analyzer, source)
        candidate_count, candidate_embedding = _arcface_observation(analyzer, candidate)
        row["arcface_source_face_count"] = source_count
        row["arcface_candidate_face_count"] = candidate_count
        if source_count == candidate_count == 1:
            value = _embedding_cosine(source_embedding, candidate_embedding)
            row["arcface_source_candidate_cosine"] = value
            complete += 1
            values.append(value)
        else:
            row["arcface_source_candidate_cosine"] = None
    return {
        "coverage": complete,
        "sample_count": len(rows),
        "mean_source_candidate_cosine": _mean(values, "ArcFace cosine")
        if values
        else None,
    }


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = b"".join(canonical_json(row) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    complete = False
    try:
        view = memoryview(content)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
        complete = True
    finally:
        os.close(descriptor)
        if not complete:
            path.unlink(missing_ok=True)


def _write_validated_run_result(
    path: Path,
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    claim: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    validate_run_result(result, request, claim, policy)
    write_exclusive_json(path, result)


def _run_quality(
    request: Mapping[str, Any],
    gpu_index: int,
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from safa.evaluation.r9_evaluator_worker import _production_quality_backend
    from safa.closeout.canonical_quality import evaluate_locked_kid

    per_sample = output_dir / "generated_per_sample.jsonl"
    _write_jsonl_exclusive(
        per_sample,
        [
            {"sample_id": row["sample_id"], "generated": row["candidate_path"]}
            for row in rows
        ],
    )
    output = output_dir / "quality.json"
    quality = dict(
        _production_quality_backend(
            quality_script=request["quality_script"],
            real_index=Path(str(request["source_index"]["path"])),
            generated_dir=output_dir / "generated",
            output=output,
            iqa_method="niqe",
            metrics=("fid", "niqe", "sharpness"),
            max_generated=None,
            max_real=None,
            subset_seed=4549,
            device="cuda:0",
            sample_id_manifest=Path(str(request["sample_manifest"]["path"])),
            per_sample_jsonl=per_sample,
            generation_result=None,
            reuse_valid_output=False,
        )
    )
    kid = evaluate_locked_kid(
        quality_script=request["quality_script"],
        real_index=Path(str(request["source_index"]["path"])),
        generated_dir=output_dir / "generated",
        sample_id_manifest=Path(str(request["sample_manifest"]["path"])),
        per_sample_jsonl=per_sample,
        subset_seed=4549,
        subset_size=request["kid_subset_size"],
        device="cuda:0",
    )
    for field in ("kid_mean", "kid_std", "kid_subset_size"):
        quality[field] = kid[field]
    quality["canonical_kid"] = kid
    return quality


def execute_screening_request(
    request_path: Path,
    gpu_index: int,
    gpu_uuid: str,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    request = validate_run_request(
        load_json(request_path, "screening run request"), policy
    )
    if "TMUX" not in os.environ:
        raise CanonicalScreeningError("screening execution must run inside tmux")
    for name, binding in request["implementations"].items():
        _assert_bound_file(binding, f"{name} implementation")
    candidate_manifest_path = _assert_bound_file(
        request["candidate_manifest"], "candidate manifest"
    )
    candidate_manifest = load_json(candidate_manifest_path, "candidate manifest")
    plan_binding = candidate_manifest.get("checkpoint_plan")
    if not isinstance(plan_binding, Mapping):
        raise CanonicalScreeningError("candidate manifest omits checkpoint plan binding")
    plan_path = _assert_bound_file(plan_binding, "checkpoint plan")
    plan = load_json(plan_path, "checkpoint plan")
    preflight_root = Path(str(plan.get("preflight_result_root", ""))).resolve()
    validate_checkpoint_plan(
        plan,
        repo_root=_request_repo_root(request),
        policy=policy,
        preflight_root=preflight_root,
    )
    validate_candidate_manifest(
        candidate_manifest,
        policy=policy,
        plan=plan,
        plan_path=plan_path,
        repo_root=_request_repo_root(request),
        preflight_root=preflight_root,
    )
    if request["candidate"] not in candidate_manifest["candidates"]:
        raise CanonicalScreeningError(
            "run request candidate is not bound by the immutable candidate manifest"
        )
    policy_path = _assert_bound_file(request["policy"], "screening policy")
    if validate_policy(
        _request_repo_root(request),
        policy_path,
        verify_historical_output_evidence=False,
    ) != dict(policy):
        raise CanonicalScreeningError("worker policy differs from current validated policy")
    cuda_binding = _assert_runtime_cuda_binding(request, gpu_index, gpu_uuid)
    output_dir = Path(str(request["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    claim = build_run_claim(
        request,
        policy,
        cuda_binding["physical_gpu_index"],
        cuda_binding["physical_gpu_uuid"],
        cuda_binding["runtime_cuda_uuid"],
        cuda_binding["cuda_visible_devices"],
        os.getpid(),
        _utc_now(),
    )
    write_exclusive_json(output_dir / "claim.json", claim)
    try:
        rows, source_paths, candidate_paths = _run_generation(
            request, gpu_index, output_dir
        )
        arcface = _run_arcface(
            request, gpu_index, rows, source_paths, candidate_paths
        )
        quality = _run_quality(request, gpu_index, output_dir, rows)
        _write_jsonl_exclusive(output_dir / "per_sample.jsonl", rows)
        evidence = {
            "mode": request["mode"],
            "replicate": request["replicate"],
            "seed": request["seed"],
            "batch_size": request["batch_size"],
            "sample_count": len(rows),
            "sample_manifest_sha256": request["sample_manifest"]["sha256"],
            "candidate_manifest_sha256": request["candidate_manifest"][
                "canonical_sha256"
            ],
            "policy_sha256": policy["policy_sha256"],
            "implementations": dict(policy["implementations"]),
            "checkpoint_sha256": request["candidate"]["checkpoint_sha256"],
            "checkpoint_model": request["candidate"]["checkpoint_model"],
            "output_contract_sha256": request["output_contract"][
                "output_contract_sha256"
            ],
            "output_contract_type": request["output_contract"]["contract_type"],
            "decoder_registry_sha256": request["output_decoder_registry"][
                "decoder_registry_sha256"
            ],
            "output_space": request["output_contract"]["capability"][
                "output_space"
            ],
            "native_rgb_size": list(request["native_rgb_size"]),
            "quality_protocol_family": request["quality_protocol_family"],
            "nfe": request["nfe"],
            "pixel_image_size": request["pixel_image_size"],
            "pixel_protocol_config_sha256": request["pixel_protocol_config"][
                "sha256"
            ],
            "kid_subset_size": request["kid_subset_size"],
            "e0_mean": _mean([row["e0_cosine"] for row in rows], "E0 cosine"),
            "edev_mean": _mean([row["edev_cosine"] for row in rows], "Edev cosine"),
            "arcface": arcface,
            "quality": quality,
            "per_sample_sha256": sha256_file(output_dir / "per_sample.jsonl"),
        }
        result = build_run_result(
            request,
            claim,
            policy,
            status="completed",
            completed_at=_utc_now(),
            evidence=evidence,
        )
        _write_validated_run_result(
            output_dir / "result.json", result, request, claim, policy
        )
        return result
    except BaseException as exc:
        result = build_run_result(
            request,
            claim,
            policy,
            status="failed",
            completed_at=_utc_now(),
            failure={
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        _write_validated_run_result(
            output_dir / "result.json", result, request, claim, policy
        )
        raise
