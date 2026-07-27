"""Independent worker for one immutable canonical screening request."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
    WORKER_READY_CONTRACT,
    WORKER_EXTERNAL_GPU_RACE_CONTRACT,
    WORKER_PRE_CUDA_VERIFICATION_ORDER,
    build_run_claim,
    build_run_result,
    canonical_digest,
    canonicalize_nvidia_gpu_uuid,
    canonical_json,
    load_json,
    load_jsonl,
    publish_exclusive_json,
    sha256_file,
    validate_candidate_manifest,
    validate_checkpoint_plan,
    validate_final_release_admission,
    validate_policy,
    validate_worker_ready_value,
    validate_worker_release_value,
    validate_run_request,
    validate_run_result,
    write_exclusive_json,
)


PRE_CUDA_VERIFICATION_ORDER = WORKER_PRE_CUDA_VERIFICATION_ORDER
EXTERNAL_GPU_RACE_CONTRACT = WORKER_EXTERNAL_GPU_RACE_CONTRACT
HEAVY_MODULE_ROOTS = (
    "torch",
    "torchvision",
    "onnxruntime",
    "diffusers",
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


def validate_pre_cuda_request(
    request_path: Path,
    policy: Mapping[str, Any],
    final_release_admission: Mapping[str, Any],
    *,
    config_path: Path,
    require_heavy_modules_absent: bool = True,
) -> dict[str, Any]:
    loaded_heavy_modules = sorted(
        name for name in HEAVY_MODULE_ROOTS if name in sys.modules
    )
    if require_heavy_modules_absent and loaded_heavy_modules:
        raise CanonicalScreeningError(
            "pre-CUDA request validation started after heavy import: "
            f"{loaded_heavy_modules}"
        )
    request = validate_run_request(
        load_json(request_path, "screening run request"), policy
    )
    repo_root = _request_repo_root(request)
    resolved_config = config_path.resolve()
    if (
        resolved_config
        != Path(str(request["policy"]["path"])).resolve()
        or sha256_file(resolved_config) != request["policy"]["sha256"]
    ):
        raise CanonicalScreeningError(
            "worker config differs from the request policy binding"
        )
    implementations = {}
    for name, binding in request["implementations"].items():
        path = _assert_bound_file(binding, f"{name} implementation")
        implementations[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }
    candidate_manifest_path = _assert_bound_file(
        request["candidate_manifest"], "candidate manifest"
    )
    candidate_manifest = load_json(
        candidate_manifest_path, "candidate manifest"
    )
    plan_binding = candidate_manifest.get("checkpoint_plan")
    if not isinstance(plan_binding, Mapping):
        raise CanonicalScreeningError(
            "candidate manifest omits checkpoint plan binding"
        )
    plan_path = _assert_bound_file(plan_binding, "checkpoint plan")
    plan = load_json(plan_path, "checkpoint plan")
    preflight_root = Path(str(plan.get("preflight_result_root", ""))).resolve()
    validate_checkpoint_plan(
        plan,
        repo_root=repo_root,
        policy=policy,
        preflight_root=preflight_root,
    )
    validate_candidate_manifest(
        candidate_manifest,
        policy=policy,
        plan=plan,
        plan_path=plan_path,
        repo_root=repo_root,
        preflight_root=preflight_root,
    )
    if request["candidate"] not in candidate_manifest["candidates"]:
        raise CanonicalScreeningError(
            "run request candidate is not bound by the immutable candidate manifest"
        )
    checkpoint_path = Path(
        str(request["candidate"]["checkpoint_path"])
    ).resolve()
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path)
        != request["candidate"]["checkpoint_sha256"]
    ):
        raise CanonicalScreeningError(
            "candidate checkpoint changed before worker release"
        )
    asset_verification_audit: list[dict[str, Any]] = []
    if validate_policy(
        repo_root,
        resolved_config,
        verify_historical_output_evidence=False,
        asset_verification_audit=asset_verification_audit,
    ) != dict(policy):
        raise CanonicalScreeningError(
            "worker policy differs from current validated policy"
        )
    validate_final_release_admission(
        final_release_admission, request, policy
    )
    _assert_ready_barrier(request, policy)
    rehashed_bindings = {
        "config": {
            "path": str(resolved_config),
            "sha256": sha256_file(resolved_config),
        },
        "implementations": implementations,
        "request": {
            "path": str(request_path.resolve()),
            "sha256": sha256_file(request_path),
            "canonical_sha256": request["run_request_sha256"],
        },
        "candidate_manifest": dict(request["candidate_manifest"]),
        "checkpoint_plan": dict(plan_binding),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": request["candidate"]["checkpoint_sha256"],
        },
        "data_and_evaluators": {
            name: request[name]
            for name in (
                "sample_manifest",
                "source_index",
                "features",
                "e0",
                "edev",
                "quality_script",
                "pixel_protocol_config",
                "arcface",
            )
        },
        "final_release": dict(final_release_admission),
        "controller_ready": dict(request["controller_ready"]),
        "observer_ready": dict(request["observer_ready"]),
    }
    return {
        "request": request,
        "repo_root": str(repo_root),
        "verification_order": list(PRE_CUDA_VERIFICATION_ORDER),
        "rehashed_bindings": rehashed_bindings,
        "rehashed_bindings_sha256": hashlib.sha256(
            canonical_json(rehashed_bindings)
        ).hexdigest(),
        "heavy_modules_absent": loaded_heavy_modules == [],
        "loaded_heavy_modules": loaded_heavy_modules,
        "asset_content_verification": asset_verification_audit[0],
    }


def _wait_worker_release(
    release_path: Path,
    ready: Mapping[str, Any],
    ready_binding: Mapping[str, Any],
    request: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while not release_path.is_file():
        if time.monotonic() >= deadline:
            raise CanonicalScreeningError(
                "worker CUDA release token timed out"
            )
        time.sleep(0.05)
    release = validate_worker_release_value(
        load_json(release_path, "worker CUDA release token"),
        request,
        policy,
        expected_worker_pid=os.getpid(),
    )
    validate_worker_ready_value(
        ready,
        request,
        policy,
        expected_worker_pid=os.getpid(),
    )
    if release["worker_ready"] != dict(ready_binding):
        raise CanonicalScreeningError(
            "worker CUDA release token binds a different ready artifact"
        )
    return release


def _assert_ready_barrier(
    request: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    controller_binding = request["controller_ready"]
    observer_binding = request["observer_ready"]
    controller_path = _assert_bound_file(
        controller_binding, "controller ready"
    )
    observer_path = _assert_bound_file(observer_binding, "observer ready")
    controller = load_json(controller_path, "controller ready")
    observer = load_json(observer_path, "observer ready")
    admission_sha256 = request["admission"]["canonical_sha256"]
    if (
        controller.get("contract_type")
        != "safa_canonical_gpu_controller_ready_v1"
        or controller.get("policy_sha256") != policy["policy_sha256"]
        or controller.get("phase") != request["mode"]
        or controller.get("admission_sha256") != admission_sha256
        or controller.get("controller_ready_sha256")
        != controller_binding["canonical_sha256"]
        or canonical_digest(controller, "controller_ready_sha256")
        != controller_binding["canonical_sha256"]
    ):
        raise CanonicalScreeningError("worker controller ready binding mismatch")
    if (
        observer.get("contract_type")
        != "safa_canonical_gpu_observer_ready_v1"
        or observer.get("policy_sha256") != policy["policy_sha256"]
        or observer.get("phase") != request["mode"]
        or observer.get("admission_sha256") != admission_sha256
        or observer.get("controller_ready_sha256")
        != controller_binding["canonical_sha256"]
        or observer.get("observer_ready_sha256")
        != observer_binding["canonical_sha256"]
        or canonical_digest(observer, "observer_ready_sha256")
        != observer_binding["canonical_sha256"]
    ):
        raise CanonicalScreeningError("worker observer ready binding mismatch")
    for field, digest_field in {
        "controller_claim": "controller_claim_sha256",
        "wrapper_claim": "wrapper_claim_sha256",
        "observer_launch": "observer_launch_sha256",
        "admission": "admission_sha256",
        "request_intent_manifest": "request_intent_manifest_sha256",
        "internal_monitor": "monitor_sample_sha256",
        "runtime_guard_first_sample": "resource_window_sha256",
        "resource_recheck": "resource_recheck_sha256",
    }.items():
        binding = controller.get(field)
        if not isinstance(binding, Mapping):
            raise CanonicalScreeningError(
                f"worker controller ready omits {field} binding"
            )
        artifact_path = _assert_bound_file(
            binding, f"controller ready {field}"
        )
        artifact = load_json(artifact_path, f"controller ready {field}")
        if (
            artifact.get(digest_field) != binding.get("canonical_sha256")
            or canonical_digest(artifact, digest_field)
            != binding.get("canonical_sha256")
        ):
            raise CanonicalScreeningError(
                f"worker controller ready {field} canonical mismatch"
            )
    for field, digest_field in {
        "observer_claim": "observer_claim_sha256",
        "wrapper_claim": "wrapper_claim_sha256",
        "observer_launch": "observer_launch_sha256",
        "controller_ready": "controller_ready_sha256",
        "admission": "admission_sha256",
        "first_observer_sample": "monitor_sample_sha256",
    }.items():
        binding = observer.get(field)
        if not isinstance(binding, Mapping):
            raise CanonicalScreeningError(
                f"worker observer ready omits {field} binding"
            )
        artifact_path = _assert_bound_file(
            binding, f"observer ready {field}"
        )
        artifact = load_json(artifact_path, f"observer ready {field}")
        if (
            artifact.get(digest_field) != binding.get("canonical_sha256")
            or canonical_digest(artifact, digest_field)
            != binding.get("canonical_sha256")
        ):
            raise CanonicalScreeningError(
                f"worker observer ready {field} canonical mismatch"
            )
    if (
        controller["controller_claim"]["canonical_sha256"]
        != controller.get("controller_claim_sha256")
        or controller["wrapper_claim"]["canonical_sha256"]
        != controller.get("wrapper_claim_sha256")
        or controller["observer_launch"]["canonical_sha256"]
        != controller.get("observer_launch_sha256")
        or controller["admission"]["canonical_sha256"]
        != controller.get("admission_sha256")
        or observer["observer_claim"]["canonical_sha256"]
        != observer.get("observer_claim_sha256")
        or observer["wrapper_claim"] != controller["wrapper_claim"]
        or observer["observer_launch"] != controller["observer_launch"]
        or observer["controller_ready"]["canonical_sha256"]
        != observer.get("controller_ready_sha256")
        or observer["admission"]["canonical_sha256"]
        != observer.get("admission_sha256")
    ):
        raise CanonicalScreeningError("worker ready primary binding mismatch")


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
    if expected_uuid is None:
        raise CanonicalScreeningError(
            "worker physical GPU index differs from admission"
        )
    expected = canonicalize_nvidia_gpu_uuid(
        expected_uuid, "admission GPU UUID"
    )
    supplied = canonicalize_nvidia_gpu_uuid(
        physical_gpu_uuid, "worker physical GPU UUID"
    )
    if supplied["canonical"] != expected["canonical"]:
        raise CanonicalScreeningError(
            "worker physical GPU index/UUID differs from admission"
        )
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise CanonicalScreeningError(
            "worker CUDA_DEVICE_ORDER must be PCI_BUS_ID"
        )
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        raise CanonicalScreeningError("worker CUDA_VISIBLE_DEVICES is missing")
    visible = canonicalize_nvidia_gpu_uuid(
        visible_devices, "worker CUDA_VISIBLE_DEVICES"
    )
    if visible["canonical"] != expected["canonical"]:
        raise CanonicalScreeningError(
            "worker CUDA_VISIBLE_DEVICES differs from the authorized GPU UUID"
        )

    import torch

    if torch.cuda.device_count() != 1:
        raise CanonicalScreeningError(
            "worker CUDA visibility must contain exactly one device"
        )
    properties = torch.cuda.get_device_properties(0)
    runtime_raw_uuid = properties.uuid
    runtime = canonicalize_nvidia_gpu_uuid(
        runtime_raw_uuid, "worker runtime CUDA UUID"
    )
    raw_evidence = {
        "admission": expected,
        "worker_argument": supplied,
        "cuda_visible_devices": visible,
        "runtime_cuda_uuid": runtime,
    }
    evidence = {
        "physical_gpu_index": physical_gpu_index,
        "physical_gpu_uuid": expected["canonical"],
        "logical_cuda_index": 0,
        "runtime_cuda_uuid": runtime["canonical"],
        "cuda_visible_devices": visible["canonical"],
        "uuid_evidence": raw_evidence,
    }
    print(
        json.dumps(
            {"event": "cuda_device_binding", **evidence},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    if runtime["canonical"] != expected["canonical"]:
        error = CanonicalScreeningError(
            "worker runtime CUDA UUID differs from the authorized GPU UUID"
        )
        error.cuda_device_binding = evidence  # type: ignore[attr-defined]
        raise error
    torch.cuda.set_device(0)
    return evidence


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


def prepare_screening_request_for_cuda(
    request_path: Path,
    gpu_index: int,
    gpu_uuid: str,
    policy: Mapping[str, Any],
    final_release_admission: Mapping[str, Any],
    worker_ready_path: Path,
    worker_release_path: Path,
) -> dict[str, Any]:
    """Complete the independently validated CPU-only worker handshake."""
    if "TMUX" not in os.environ:
        raise CanonicalScreeningError("screening execution must run inside tmux")
    pre_cuda = validate_pre_cuda_request(
        request_path,
        policy,
        final_release_admission,
        config_path=Path(str(policy["policy_file"]["path"])),
    )
    request = pre_cuda["request"]
    ready = {
        "schema_version": 1,
        "contract_type": WORKER_READY_CONTRACT,
        "policy_sha256": policy["policy_sha256"],
        "phase": request["mode"],
        "worker_pid": os.getpid(),
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "run_request_sha256": request["run_request_sha256"],
        "request": pre_cuda["rehashed_bindings"]["request"],
        "final_release": dict(final_release_admission),
        "verification_order": pre_cuda["verification_order"],
        "rehashed_bindings": pre_cuda["rehashed_bindings"],
        "rehashed_bindings_sha256": pre_cuda[
            "rehashed_bindings_sha256"
        ],
        "controller_claim": load_json(
            Path(str(request["controller_ready"]["path"])),
            "worker controller ready",
        )["controller_claim"],
        "screening_worker_sha256": request["implementations"][
            "screening_worker"
        ]["sha256"],
        "controller_implementation_sha256": request["implementations"][
            "controller"
        ]["sha256"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "heavy_modules_absent": pre_cuda["heavy_modules_absent"],
        "loaded_heavy_modules": pre_cuda["loaded_heavy_modules"],
        "asset_content_verification": pre_cuda[
            "asset_content_verification"
        ],
        "external_gpu_race_contract": EXTERNAL_GPU_RACE_CONTRACT,
        "ready_at": _utc_now(),
    }
    ready["worker_ready_sha256"] = canonical_digest(
        ready, "worker_ready_sha256"
    )
    validate_worker_ready_value(
        ready,
        request,
        policy,
        expected_worker_pid=os.getpid(),
        expected_gpu_index=gpu_index,
        expected_gpu_uuid=gpu_uuid,
    )
    publish_exclusive_json(worker_ready_path, ready)
    ready_binding = {
        "path": str(worker_ready_path.resolve()),
        "sha256": sha256_file(worker_ready_path),
        "canonical_sha256": ready["worker_ready_sha256"],
    }
    release = _wait_worker_release(
        worker_release_path,
        ready,
        ready_binding,
        request,
        policy,
        timeout_seconds=180.0,
    )
    post_release = validate_pre_cuda_request(
        request_path,
        policy,
        final_release_admission,
        config_path=Path(str(policy["policy_file"]["path"])),
    )
    if (
        post_release["rehashed_bindings_sha256"]
        != pre_cuda["rehashed_bindings_sha256"]
    ):
        raise CanonicalScreeningError(
            "worker bindings changed after CUDA release publication"
        )
    release_binding = {
        "path": str(worker_release_path.resolve()),
        "sha256": sha256_file(worker_release_path),
        "canonical_sha256": release["worker_release_sha256"],
    }
    return {
        "request": request,
        "pre_cuda": pre_cuda,
        "post_release": post_release,
        "worker_ready": ready,
        "worker_ready_binding": ready_binding,
        "worker_release": release,
        "worker_release_binding": release_binding,
        "next_stage": "runtime_cuda_binding",
    }


def _execute_screening_request_impl(
    request_path: Path,
    gpu_index: int,
    gpu_uuid: str,
    policy: Mapping[str, Any],
    final_release_admission: Mapping[str, Any],
    worker_ready_path: Path,
    worker_release_path: Path,
) -> dict[str, Any]:
    prepared = prepare_screening_request_for_cuda(
        request_path,
        gpu_index,
        gpu_uuid,
        policy,
        final_release_admission,
        worker_ready_path,
        worker_release_path,
    )
    request = prepared["request"]
    ready_binding = prepared["worker_ready_binding"]
    release_binding = prepared["worker_release_binding"]
    cuda_binding = _assert_runtime_cuda_binding(request, gpu_index, gpu_uuid)
    output_dir = Path(str(request["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    claim = build_run_claim(
        request,
        policy,
        final_release_admission,
        ready_binding,
        release_binding,
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


def execute_screening_request(
    request_path: Path,
    gpu_index: int,
    gpu_uuid: str,
    policy: Mapping[str, Any],
    final_release_admission: Mapping[str, Any],
    worker_ready_path: Path,
    worker_release_path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    started_at = _utc_now()
    try:
        result = _execute_screening_request_impl(
            request_path,
            gpu_index,
            gpu_uuid,
            policy,
            final_release_admission,
            worker_ready_path,
            worker_release_path,
        )
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        def terminal_binding(
            path: Path, digest_field: str
        ) -> dict[str, Any] | None:
            if not path.is_file():
                return None
            canonical_sha256 = None
            try:
                value = load_json(path, f"{digest_field} terminal binding")
            except (CanonicalScreeningError, OSError):
                value = {}
            if isinstance(value, Mapping):
                candidate = value.get(digest_field)
                if isinstance(candidate, str):
                    canonical_sha256 = candidate
            return {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "canonical_sha256": canonical_sha256,
            }

        request_binding = terminal_binding(
            request_path, "run_request_sha256"
        )
        claim_binding = None
        result_binding = None
        try:
            terminal_request = load_json(
                request_path, "worker terminal request"
            )
            terminal_output_dir = Path(
                str(terminal_request["output_dir"])
            ).resolve()
            claim_binding = terminal_binding(
                terminal_output_dir / "claim.json",
                "run_claim_sha256",
            )
            result_binding = terminal_binding(
                terminal_output_dir / "result.json",
                "run_result_sha256",
            )
        except (CanonicalScreeningError, KeyError, TypeError, OSError):
            pass
        terminal = {
            "schema_version": 1,
            "contract_type": "safa_canonical_worker_terminal_v1",
            "policy_sha256": policy["policy_sha256"],
            "worker_pid": os.getpid(),
            "request": request_binding,
            "claim": claim_binding,
            "result": result_binding,
            "worker_ready": terminal_binding(
                worker_ready_path, "worker_ready_sha256"
            ),
            "worker_release": terminal_binding(
                worker_release_path, "worker_release_sha256"
            ),
            "status": "completed" if result is not None else "failed",
            "failure": (
                None
                if failure is None
                else {
                    "type": type(failure).__name__,
                    "message": str(failure),
                }
            ),
            "started_at": started_at,
            "completed_at": _utc_now(),
        }
        terminal["worker_terminal_sha256"] = canonical_digest(
            terminal, "worker_terminal_sha256"
        )
        publish_exclusive_json(
            worker_ready_path.parent / "worker_terminal.json", terminal
        )
