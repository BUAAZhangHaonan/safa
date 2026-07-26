from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from safa.models.generator import (
    FlowGeneratorConfig,
    GENERATOR_MODEL_TYPE_FLOW,
    GENERATOR_MODEL_TYPE_MEANFLOW_SIT,
    SIT_DATA_SPACE_LATENT,
)
from safa.utils.hashing import sha256_file


SCHEMA_VERSION = 1
CAPABILITY_CONTRACT = "safa_generator_output_capability_v1"
OUTPUT_CONTRACT = "safa_generator_output_contract_v1"
DECODER_REGISTRY_CONTRACT = "safa_canonical_output_decoder_registry_v1"
LATENT_DECODER_TYPE = "r9_frozen_sd_vae_ft_ema"
PIXEL_DECODER_TYPE = "native_rgb_unit_interval"


class GeneratorOutputContractError(ValueError):
    pass


def _canonical_digest(
    value: Mapping[str, Any],
    excluded_field: str | None = None,
) -> str:
    payload = {
        key: item
        for key, item in dict(value).items()
        if key != excluded_field
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decoder_registry_digest(value: Mapping[str, Any]) -> str:
    return _canonical_digest(value, "decoder_registry_sha256")


def canonical_runtime_model_config(
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize the strict-load runtime config through one canonical path."""
    runtime_input = dict(_mapping(model_config, "generator model_config"))
    # Pretrained SiT weights are an initialization-only input.  A checkpoint's
    # serialized state is authoritative during strict reconstruction, so the
    # path must be cleared before parsing and canonical serialization.  Parsing
    # after clearing also lets FlowGeneratorConfig.to_dict() omit the empty key.
    runtime_input["sit_pretrained_path"] = ""
    return FlowGeneratorConfig.from_dict(runtime_input).to_dict()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeneratorOutputContractError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise GeneratorOutputContractError(
            f"{label} fields differ: "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _file_binding(value: Any, label: str) -> Mapping[str, Any]:
    binding = _mapping(value, label)
    _exact_keys(binding, {"path", "sha256"}, label)
    digest = binding["sha256"]
    if (
        not isinstance(binding["path"], str)
        or not binding["path"]
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise GeneratorOutputContractError(f"{label} binding differs")
    return binding


def validate_decoder_registry(
    decoder_registry: Mapping[str, Any],
) -> dict[str, Any]:
    registry = dict(_mapping(decoder_registry, "output decoder registry"))
    _exact_keys(
        registry,
        {
            "schema_version",
            "contract_type",
            "pixel",
            "latent",
            "decoder_registry_sha256",
        },
        "output decoder registry",
    )
    if (
        registry["schema_version"] != SCHEMA_VERSION
        or registry["contract_type"] != DECODER_REGISTRY_CONTRACT
        or registry["decoder_registry_sha256"]
        != _canonical_digest(registry, "decoder_registry_sha256")
    ):
        raise GeneratorOutputContractError(
            "output decoder registry type/digest differs"
        )
    pixel = _mapping(registry["pixel"], "pixel output registry")
    _exact_keys(
        pixel,
        {
            "decoder_type",
            "output_range",
            "channels",
            "height",
            "width",
            "model_type",
            "sampler",
            "sample_steps",
            "model_space",
            "sample_api",
            "clamp_output",
            "postprocess",
            "decoder_forbidden",
            "sampling_implementation",
        },
        "pixel output registry",
    )
    _file_binding(
        pixel["sampling_implementation"],
        "pixel sampling implementation",
    )
    if (
        pixel["decoder_type"] != PIXEL_DECODER_TYPE
        or pixel["output_range"] != [0.0, 1.0]
        or pixel["channels"] != 3
        or pixel["height"] != 224
        or pixel["width"] != 224
        or pixel["model_type"] != GENERATOR_MODEL_TYPE_FLOW
        or pixel["sampler"] != "heun"
        or pixel["sample_steps"] != 32
        or pixel["model_space"] != "rgb_neg1_pos1"
        or pixel["sample_api"] != "clamp_output=true"
        or pixel["clamp_output"] is not True
        or pixel["postprocess"]
        != "in_generator_clamp_minus1_1_then_affine_then_clamp_unit_interval"
        or pixel["decoder_forbidden"] is not True
    ):
        raise GeneratorOutputContractError(
            "pixel output registry semantics differ"
        )
    latent = _mapping(registry["latent"], "latent decoder registry")
    _exact_keys(
        latent,
        {
            "decoder_type",
            "vae_source_path",
            "directory",
            "config",
            "weights",
            "scaling_factor",
            "implementation",
            "trusted_runtime_config",
            "trusted_runner",
            "trusted_reference_checkpoint",
            "trusted_resolved_config",
            "trusted_generation_result",
            "environment",
            "directory_digest_algorithm",
            "asset_digest_cache",
            "asset_digest_cache_algorithm",
            "latent_shape",
            "decoded_rgb_shape",
            "output_range",
        },
        "latent decoder registry",
    )
    directory = _mapping(latent["directory"], "latent decoder directory")
    _exact_keys(directory, {"path", "digest"}, "latent decoder directory")
    for name in (
        "config",
        "weights",
        "implementation",
        "trusted_runtime_config",
        "trusted_runner",
        "trusted_reference_checkpoint",
        "trusted_resolved_config",
        "trusted_generation_result",
        "asset_digest_cache_algorithm",
    ):
        _file_binding(latent[name], f"latent decoder {name}")
    cache = _mapping(latent["asset_digest_cache"], "latent asset digest cache")
    _exact_keys(cache, {"path"}, "latent asset digest cache")
    environment = _mapping(latent["environment"], "decoder environment")
    _exact_keys(
        environment,
        {
            "provenance_snapshot",
            "packages_sha256",
            "python_version",
            "torch_version",
            "diffusers_version",
        },
        "decoder environment",
    )
    _file_binding(
        environment["provenance_snapshot"],
        "decoder environment provenance",
    )
    if (
        latent["decoder_type"] != LATENT_DECODER_TYPE
        or latent["vae_source_path"]
        != "artifacts/checkpoints/external/sd-vae-ft-ema"
        or latent["scaling_factor"] != 0.18215
        or latent["directory_digest_algorithm"]
        != "sha256_relative_posix_nul_content_nul_v1"
        or latent["latent_shape"] != ["B", 4, 32, 32]
        or latent["decoded_rgb_shape"] != ["B", 3, 256, 256]
        or latent["output_range"] != [0.0, 1.0]
        or environment["packages_sha256"]
        != "35196c0c7f5a8a2db3dcb31a67c0102fbd713db6d67af72eacfffe8f8b82be7b"
        or environment["python_version"] != "3.12.13"
        or environment["torch_version"] != "2.11.0+cu128"
        or environment["diffusers_version"] != "0.38.0"
    ):
        raise GeneratorOutputContractError(
            "latent decoder registry semantics differ"
        )
    return registry


def resolve_checkpoint_output_capability(
    checkpoint_payload: Mapping[str, Any],
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    payload = _mapping(checkpoint_payload, "checkpoint payload")
    model_config = _mapping(payload.get("model_config"), "checkpoint model_config")
    model_type = model_config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise GeneratorOutputContractError(
            "checkpoint model_config.model_type must be explicit"
        )
    training = payload.get("training_config")
    if training is None:
        training = {}
    training = _mapping(training, "checkpoint training_config")
    latent_training = training.get("latent_training")
    if latent_training is not None and not isinstance(latent_training, bool):
        raise GeneratorOutputContractError(
            "checkpoint training_config.latent_training must be boolean when present"
        )

    if model_type == GENERATOR_MODEL_TYPE_MEANFLOW_SIT:
        data_space = model_config.get("sit_data_space")
        if data_space != SIT_DATA_SPACE_LATENT:
            raise GeneratorOutputContractError(
                "registered MeanFlow-SiT capability requires latent data space"
            )
        if (
            model_config.get("sit_input_channels") != 4
            or model_config.get("image_size") != 32
            or model_config.get("sampler") != "meanflow"
            or model_config.get("sample_steps") != 1
        ):
            raise GeneratorOutputContractError(
                "MeanFlow-SiT latent capability fields differ"
            )
        output_space = SIT_DATA_SPACE_LATENT
        resolution_source = "checkpoint.model_config.sit_data_space"
        nfe = 1
    elif model_type == GENERATOR_MODEL_TYPE_FLOW:
        if (
            model_config.get("sampler") != "heun"
            or model_config.get("sample_steps") != 32
            or model_config.get("image_size") != 224
        ):
            raise GeneratorOutputContractError(
                "registered conditional-flow pixel capability fields differ"
            )
        output_space = "pixel"
        resolution_source = "checkpoint.model_config.model_type"
        forbidden = {
            key: model_config[key]
            for key in ("sit_data_space", "sit_input_channels")
            if key in model_config
        }
        if forbidden:
            raise GeneratorOutputContractError(
                f"pixel-native model carries incompatible SiT fields: {forbidden}"
            )
        nfe = 2 * int(model_config["sample_steps"]) - 1
    else:
        raise GeneratorOutputContractError(
            f"generator model_type has no registered output capability: {model_type}"
        )

    expected_latent_training = output_space == SIT_DATA_SPACE_LATENT
    if (
        latent_training is not None
        and latent_training is not expected_latent_training
    ):
        raise GeneratorOutputContractError(
            "checkpoint training_config.latent_training contradicts model capability"
        )
    training_decoder = {
        key: {
            "present": key in training,
            "value": training.get(key),
        }
        for key in ("vae_path", "vae_model", "vae_scaling_factor")
    }
    training_observation = {
        "latent_training": {
            "present": "latent_training" in training,
            "value": latent_training,
        },
        "decoder_fields": training_decoder,
    }
    runtime_model_config = canonical_runtime_model_config(model_config)
    if output_space == SIT_DATA_SPACE_LATENT:
        generator_output_tensor = {
            "dtype_family": "floating",
            "rank": 4,
            "channels": 4,
            "height": 32,
            "width": 32,
            "semantic_space": "scaled_vae_latent",
            "value_contract": "finite_unbounded",
            "postprocess": "identity",
        }
        resolution_source = "checkpoint_model_config_explicit_latent_space"
    else:
        generator_output_tensor = {
            "dtype_family": "floating",
            "rank": 4,
            "channels": 3,
            "height": 224,
            "width": 224,
            "semantic_space": "rgb_unit_interval",
            "value_contract": "finite_closed_unit_interval",
            "postprocess": (
                "in_generator_clamp_minus1_1_then_affine_then_"
                "clamp_unit_interval"
            ),
        }
        resolution_source = "registered_model_type_native_pixel_semantics"
    capability = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": CAPABILITY_CONTRACT,
        "checkpoint_sha256": checkpoint_sha256,
        "model_type": model_type,
        "output_space": output_space,
        "resolution_source": resolution_source,
        "model_config_sha256": _canonical_digest(model_config),
        "runtime_model_config_sha256": _canonical_digest(
            runtime_model_config
        ),
        "sampler": model_config.get("sampler"),
        "sample_steps": model_config.get("sample_steps"),
        "model_image_size": model_config.get("image_size"),
        "native_output_size": [
            model_config.get("image_size"),
            model_config.get("image_size"),
        ],
        "generator_output_tensor": generator_output_tensor,
        "nfe": nfe,
        "checkpoint_training_config_observation": training_observation,
    }
    capability["capability_sha256"] = _canonical_digest(capability)
    return capability


def bind_output_contract(
    capability: Mapping[str, Any],
    decoder_registry: Mapping[str, Any],
) -> dict[str, Any]:
    capability = validate_output_capability(capability)
    registry = validate_decoder_registry(decoder_registry)
    if capability["checkpoint_sha256"] is None:
        raise GeneratorOutputContractError(
            "bound output contract requires exact checkpoint SHA256"
        )
    if capability["output_space"] == SIT_DATA_SPACE_LATENT:
        decoder = _mapping(registry.get("latent"), "latent decoder registry")
        if decoder.get("decoder_type") != LATENT_DECODER_TYPE:
            raise GeneratorOutputContractError("latent decoder type differs")
        observation = capability["checkpoint_training_config_observation"]
        latent_observation = observation["latent_training"]
        decoder_observations = observation["decoder_fields"]
        if (
            latent_observation["present"]
            and latent_observation["value"] is not True
        ):
            raise GeneratorOutputContractError(
                "checkpoint latent_training observation contradicts latent output"
            )
        expected_source = decoder["vae_source_path"]
        source_observation = decoder_observations["vae_path"]
        if (
            source_observation["present"]
            and source_observation["value"] != expected_source
        ):
            raise GeneratorOutputContractError(
                "checkpoint vae_path observation contradicts frozen decoder"
            )
        model_observation = decoder_observations["vae_model"]
        if model_observation["present"] and model_observation["value"] not in {
            None,
            "",
        }:
            raise GeneratorOutputContractError(
                "checkpoint vae_model observation contradicts path-pinned decoder"
            )
        scale_observation = decoder_observations["vae_scaling_factor"]
        if (
            scale_observation["present"]
            and float(scale_observation["value"])
            != float(decoder["scaling_factor"])
        ):
            raise GeneratorOutputContractError(
                "checkpoint VAE scale observation contradicts frozen decoder"
            )
        bound_decoder: dict[str, Any] | None = dict(decoder)
        decoder_mode = "required"
    else:
        pixel = _mapping(registry.get("pixel"), "pixel output registry")
        if (
            pixel.get("decoder_type") != PIXEL_DECODER_TYPE
            or pixel.get("output_range") != [0.0, 1.0]
            or pixel.get("channels") != 3
            or pixel.get("height") != 224
            or pixel.get("width") != 224
            or pixel.get("model_type") != GENERATOR_MODEL_TYPE_FLOW
            or pixel.get("sampler") != "heun"
            or pixel.get("sample_steps") != 32
            or pixel.get("model_space") != "rgb_neg1_pos1"
            or pixel.get("sample_api") != "clamp_output=true"
            or pixel.get("clamp_output") is not True
            or pixel.get("postprocess")
            != "in_generator_clamp_minus1_1_then_affine_then_clamp_unit_interval"
            or pixel.get("decoder_forbidden") is not True
            or set(pixel)
            != {
                "decoder_type",
                "output_range",
                "channels",
                "height",
                "width",
                "model_type",
                "sampler",
                "sample_steps",
                "model_space",
                "sample_api",
                "clamp_output",
                "postprocess",
                "decoder_forbidden",
                "sampling_implementation",
            }
        ):
            raise GeneratorOutputContractError("pixel output registry differs")
        observation = capability["checkpoint_training_config_observation"]
        latent_observation = observation["latent_training"]
        if (
            latent_observation["present"]
            and latent_observation["value"] is True
        ) or any(
            item["present"]
            for item in observation["decoder_fields"].values()
        ):
            raise GeneratorOutputContractError(
                "pixel checkpoint records latent decoder training fields"
            )
        bound_decoder = None
        decoder_mode = "forbidden"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "contract_type": OUTPUT_CONTRACT,
        "capability": capability,
        "decoder_mode": decoder_mode,
        "decoder": bound_decoder,
        "quality_protocol_family": (
            "native_rgb_256_r9_frozen_vae"
            if decoder_mode == "required"
            else "native_rgb_224_cfm"
        ),
        "rgb_contract": (
            {
                "dtype_family": "floating",
                "rank": 4,
                "channels": 3,
                "height": 256,
                "width": 256,
                "semantic_space": "rgb_unit_interval",
                "value_contract": "finite_closed_unit_interval",
                "output_range": [0.0, 1.0],
                "finite": True,
                "postprocess": "frozen_vae_decode_then_affine_then_clamp_unit_interval",
            }
            if bound_decoder is not None
            else {
                "dtype_family": "floating",
                "rank": 4,
                "channels": 3,
                "height": 224,
                "width": 224,
                "semantic_space": "rgb_unit_interval",
                "value_contract": "finite_closed_unit_interval",
                "output_range": [0.0, 1.0],
                "finite": True,
                "postprocess": "identity_after_registered_generator_sample",
            }
        ),
    }
    contract["output_contract_sha256"] = _canonical_digest(contract)
    return contract


def validate_output_capability(
    capability: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(_mapping(capability, "output capability"))
    required = {
        "schema_version",
        "contract_type",
        "checkpoint_sha256",
        "model_type",
        "output_space",
        "resolution_source",
        "model_config_sha256",
        "runtime_model_config_sha256",
        "sampler",
        "sample_steps",
        "model_image_size",
        "native_output_size",
        "generator_output_tensor",
        "nfe",
        "checkpoint_training_config_observation",
        "capability_sha256",
    }
    if set(value) != required:
        raise GeneratorOutputContractError("output capability fields differ")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["contract_type"] != CAPABILITY_CONTRACT
        or value["output_space"] not in {"pixel", SIT_DATA_SPACE_LATENT}
        or value["capability_sha256"]
        != _canonical_digest(value, "capability_sha256")
    ):
        raise GeneratorOutputContractError("output capability contract differs")
    checkpoint_sha256 = value["checkpoint_sha256"]
    if checkpoint_sha256 is not None and (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise GeneratorOutputContractError(
            "output capability checkpoint_sha256 is invalid"
        )
    for field in ("model_config_sha256", "runtime_model_config_sha256"):
        digest = value[field]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise GeneratorOutputContractError(
                f"output capability {field} is invalid"
            )
    tensor = _mapping(
        value["generator_output_tensor"],
        "generator output tensor capability",
    )
    tensor_keys = {
        "dtype_family",
        "rank",
        "channels",
        "height",
        "width",
        "semantic_space",
        "value_contract",
        "postprocess",
    }
    if set(tensor) != tensor_keys:
        raise GeneratorOutputContractError(
            "generator output tensor capability fields differ"
        )
    expected_tensor = (
        {
            "dtype_family": "floating",
            "rank": 4,
            "channels": 4,
            "height": 32,
            "width": 32,
            "semantic_space": "scaled_vae_latent",
            "value_contract": "finite_unbounded",
            "postprocess": "identity",
        }
        if value["output_space"] == SIT_DATA_SPACE_LATENT
        else {
            "dtype_family": "floating",
            "rank": 4,
            "channels": 3,
            "height": 224,
            "width": 224,
            "semantic_space": "rgb_unit_interval",
            "value_contract": "finite_closed_unit_interval",
            "postprocess": (
                "in_generator_clamp_minus1_1_then_affine_then_"
                "clamp_unit_interval"
            ),
        }
    )
    expected_resolution_source = (
        "checkpoint_model_config_explicit_latent_space"
        if value["output_space"] == SIT_DATA_SPACE_LATENT
        else "registered_model_type_native_pixel_semantics"
    )
    expected_nfe = 1 if value["output_space"] == SIT_DATA_SPACE_LATENT else 63
    if (
        dict(tensor) != expected_tensor
        or value["native_output_size"]
        != [expected_tensor["height"], expected_tensor["width"]]
        or value["resolution_source"] != expected_resolution_source
        or value["nfe"] != expected_nfe
    ):
        raise GeneratorOutputContractError(
            "generator output tensor semantics differ"
        )
    expected_model_fields = (
        {
            "model_type": GENERATOR_MODEL_TYPE_MEANFLOW_SIT,
            "sampler": "meanflow",
            "sample_steps": 1,
            "model_image_size": 32,
        }
        if value["output_space"] == SIT_DATA_SPACE_LATENT
        else {
            "model_type": GENERATOR_MODEL_TYPE_FLOW,
            "sampler": "heun",
            "sample_steps": 32,
            "model_image_size": 224,
        }
    )
    if any(value[field] != expected for field, expected in expected_model_fields.items()):
        raise GeneratorOutputContractError(
            "generator model/output capability combination differs"
        )
    observation = _mapping(
        value["checkpoint_training_config_observation"],
        "checkpoint training config observation",
    )
    if set(observation) != {"latent_training", "decoder_fields"}:
        raise GeneratorOutputContractError(
            "checkpoint training config observation fields differ"
        )
    latent_observation = _mapping(
        observation["latent_training"],
        "checkpoint latent training observation",
    )
    if set(latent_observation) != {"present", "value"}:
        raise GeneratorOutputContractError(
            "checkpoint latent training observation fields differ"
        )
    if not isinstance(latent_observation["present"], bool) or (
        latent_observation["value"] is not None
        and not isinstance(latent_observation["value"], bool)
    ):
        raise GeneratorOutputContractError(
            "checkpoint latent training observation value differs"
        )
    decoder_observations = _mapping(
        observation["decoder_fields"],
        "checkpoint decoder field observations",
    )
    if set(decoder_observations) != {
        "vae_path",
        "vae_model",
        "vae_scaling_factor",
    }:
        raise GeneratorOutputContractError(
            "checkpoint decoder observation fields differ"
        )
    for name, observation_value in decoder_observations.items():
        item = _mapping(observation_value, f"checkpoint decoder observation {name}")
        if set(item) != {"present", "value"}:
            raise GeneratorOutputContractError(
                f"checkpoint decoder observation {name} fields differ"
            )
        if not isinstance(item["present"], bool):
            raise GeneratorOutputContractError(
                f"checkpoint decoder observation {name}.present must be boolean"
            )
    return value


def validate_output_contract(
    contract: Mapping[str, Any],
    decoder_registry: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(_mapping(contract, "generator output contract"))
    required = {
        "schema_version",
        "contract_type",
        "capability",
        "decoder_mode",
        "decoder",
        "quality_protocol_family",
        "rgb_contract",
        "output_contract_sha256",
    }
    if set(value) != required:
        raise GeneratorOutputContractError("generator output contract fields differ")
    expected = bind_output_contract(value["capability"], decoder_registry)
    if value != expected:
        raise GeneratorOutputContractError(
            "generator output contract differs from frozen decoder registry"
        )
    if (
        value["decoder_mode"] == "required"
        and not isinstance(value["decoder"], Mapping)
    ) or (
        value["decoder_mode"] == "forbidden"
        and value["decoder"] is not None
    ):
        raise GeneratorOutputContractError(
            "generator output decoder mode/object binding differs"
        )
    return value


def digest_asset_directory(path: Path) -> str:
    if path.is_symlink() or not path.is_dir():
        raise GeneratorOutputContractError(
            f"decoder directory is missing or symlinked: {path}"
        )
    files = sorted(
        item for item in path.rglob("*") if item.is_file() or item.is_symlink()
    )
    if not files:
        raise GeneratorOutputContractError(
            f"decoder directory contains no files: {path}"
        )
    digest = hashlib.sha256()
    for file_path in files:
        if file_path.is_symlink():
            raise GeneratorOutputContractError(
                f"decoder directory contains a symlink: {file_path}"
            )
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _assert_decoder_assets(decoder: Mapping[str, Any]) -> Path:
    directory = _mapping(decoder.get("directory"), "decoder directory binding")
    path = Path(str(directory.get("path"))).resolve()
    cache = _mapping(
        decoder.get("asset_digest_cache"),
        "decoder asset digest cache",
    )
    from safa.evaluation.meanflow_guidance_runner import cached_asset_digest

    try:
        actual = cached_asset_digest(
            path,
            str(directory.get("digest")),
            Path(str(cache.get("path"))).resolve(),
        )
    except Exception as exc:
        raise GeneratorOutputContractError(
            f"decoder directory cached digest verification failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if actual != directory.get("digest"):
        raise GeneratorOutputContractError("decoder directory digest mismatch")
    for name in ("implementation",):
        binding = _mapping(decoder.get(name), f"decoder {name} binding")
        asset = Path(str(binding.get("path"))).resolve()
        if not asset.is_file() or sha256_file(asset) != binding.get("sha256"):
            raise GeneratorOutputContractError(
                f"decoder {name} SHA256 mismatch: {asset}"
            )
    environment = _mapping(decoder.get("environment"), "decoder environment")
    import diffusers
    import torch

    runtime = {
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        "torch_version": str(torch.__version__),
        "diffusers_version": str(diffusers.__version__),
    }
    for field, actual in runtime.items():
        if environment.get(field) != actual:
            raise GeneratorOutputContractError(
                f"decoder runtime {field} mismatch: "
                f"expected={environment.get(field)!r}, actual={actual!r}"
            )
    return path


def validate_loaded_generator_capability(
    generator: Any,
    capability: Mapping[str, Any],
) -> None:
    value = validate_output_capability(capability)
    config = getattr(generator, "config", None)
    if config is None:
        raise GeneratorOutputContractError(
            "strict-loaded generator has no explicit config"
        )
    actual = {
        "model_type": getattr(config, "model_type", None),
        "sampler": getattr(config, "sampler", None),
        "sample_steps": getattr(config, "sample_steps", None),
        "model_image_size": getattr(config, "image_size", None),
    }
    expected = {
        "model_type": value["model_type"],
        "sampler": value["sampler"],
        "sample_steps": value["sample_steps"],
        "model_image_size": value["model_image_size"],
    }
    if actual != expected:
        raise GeneratorOutputContractError(
            f"strict-loaded generator capability differs: "
            f"expected={expected}, actual={actual}"
        )
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise GeneratorOutputContractError(
            "strict-loaded generator config cannot be canonically serialized"
        )
    actual_runtime_config = canonical_runtime_model_config(to_dict())
    if (
        _canonical_digest(actual_runtime_config)
        != value["runtime_model_config_sha256"]
    ):
        raise GeneratorOutputContractError(
            "strict-loaded generator canonical config digest differs"
        )
    output_tensor = value["generator_output_tensor"]
    if value["output_space"] == SIT_DATA_SPACE_LATENT:
        latent_actual = {
            "sit_data_space": getattr(config, "sit_data_space", None),
            "sit_input_channels": getattr(config, "sit_input_channels", None),
        }
        latent_expected = {
            "sit_data_space": SIT_DATA_SPACE_LATENT,
            "sit_input_channels": output_tensor["channels"],
        }
        if latent_actual != latent_expected:
            raise GeneratorOutputContractError(
                f"strict-loaded latent capability differs: "
                f"expected={latent_expected}, actual={latent_actual}"
            )
    elif output_tensor["channels"] != 3:
        raise GeneratorOutputContractError(
            "strict-loaded pixel capability must declare three channels"
        )


def build_bound_decoder(
    output_contract: Mapping[str, Any],
    decoder_registry: Mapping[str, Any],
    device: Any,
):
    contract = validate_output_contract(output_contract, decoder_registry)
    if contract["decoder_mode"] == "forbidden":
        pixel = _mapping(decoder_registry.get("pixel"), "pixel output registry")
        implementation = _mapping(
            pixel.get("sampling_implementation"),
            "pixel sampling implementation binding",
        )
        implementation_path = Path(str(implementation.get("path"))).resolve()
        if (
            not implementation_path.is_file()
            or sha256_file(implementation_path) != implementation.get("sha256")
        ):
            raise GeneratorOutputContractError(
                "pixel sampling implementation SHA256 mismatch: "
                f"{implementation_path}"
            )
        return None
    decoder = _mapping(contract["decoder"], "bound latent decoder")
    source = _assert_decoder_assets(decoder)
    from safa.training.latent_codec import build_latent_codec_from_train_config

    codec = build_latent_codec_from_train_config(
        {
            "latent_training": True,
            "vae_path": str(source),
            "vae_scaling_factor": float(decoder["scaling_factor"]),
        },
        device,
    )
    if codec is None:
        raise GeneratorOutputContractError(
            "frozen latent decoder builder returned no codec"
        )
    codec.vae.eval()
    return codec


def validate_rgb_unit_interval(
    images: Any,
    context: str,
    rgb_contract: Mapping[str, Any],
):
    import torch

    if not isinstance(images, torch.Tensor):
        raise GeneratorOutputContractError(
            f"{context} must be a torch.Tensor"
        )
    expected = (
        int(rgb_contract["channels"]),
        int(rgb_contract["height"]),
        int(rgb_contract["width"]),
    )
    if images.ndim != 4 or tuple(images.shape[1:]) != expected:
        raise GeneratorOutputContractError(
            f"{context} must have shape [B,{expected[0]},{expected[1]},"
            f"{expected[2]}], got {tuple(images.shape)}"
        )
    if not torch.is_floating_point(images):
        raise GeneratorOutputContractError(f"{context} must be floating point")
    if not torch.isfinite(images).all():
        raise GeneratorOutputContractError(f"{context} contains non-finite values")
    minimum = float(images.min().detach().cpu())
    maximum = float(images.max().detach().cpu())
    if minimum < 0.0 or maximum > 1.0:
        raise GeneratorOutputContractError(
            f"{context} must be in [0,1], got min={minimum}, max={maximum}"
        )
    return images


def validate_native_generator_output(
    generated: Any,
    output_contract: Mapping[str, Any],
    decoder_registry: Mapping[str, Any],
):
    import torch

    contract = validate_output_contract(output_contract, decoder_registry)
    capability = contract["capability"]
    if not isinstance(generated, torch.Tensor):
        raise GeneratorOutputContractError(
            "native generator output must be a torch.Tensor"
        )
    expected = (
        int(capability["generator_output_tensor"]["channels"]),
        int(capability["generator_output_tensor"]["height"]),
        int(capability["generator_output_tensor"]["width"]),
    )
    if generated.ndim != 4 or tuple(generated.shape[1:]) != expected:
        raise GeneratorOutputContractError(
            "native generator output shape differs: "
            f"expected=[B,{expected[0]},{expected[1]},{expected[2]}], "
            f"actual={tuple(generated.shape)}"
        )
    if not torch.is_floating_point(generated):
        raise GeneratorOutputContractError(
            "native generator output must be floating point"
        )
    if not torch.isfinite(generated).all():
        raise GeneratorOutputContractError(
            "native generator output contains non-finite values"
        )
    if capability["output_space"] == "pixel":
        validate_rgb_unit_interval(
            generated,
            "native pixel generator output",
            contract["rgb_contract"],
        )
    return generated


def decode_generator_output(
    generated: Any,
    output_contract: Mapping[str, Any],
    decoder_registry: Mapping[str, Any],
    device: Any,
):
    contract = validate_output_contract(output_contract, decoder_registry)
    codec = build_bound_decoder(contract, decoder_registry, device)
    rgb = generated if codec is None else codec.decode(generated)
    return validate_rgb_unit_interval(
        rgb,
        "decoded generator RGB output",
        contract["rgb_contract"],
    )
