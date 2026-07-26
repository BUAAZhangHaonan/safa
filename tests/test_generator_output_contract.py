from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from safa.closeout.generator_output_contract import (
    GeneratorOutputContractError,
    bind_output_contract,
    build_bound_decoder,
    decoder_registry_digest,
    resolve_checkpoint_output_capability,
    validate_decoder_registry,
    validate_native_generator_output,
    validate_rgb_unit_interval,
)
from safa.evaluation.meanflow_guidance_runner import cached_asset_digest


ROOT = Path(__file__).parents[1]


def _binding(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _registry(tmp_path: Path) -> dict:
    implementation = tmp_path / "implementation.py"
    implementation.write_text("pass\n", encoding="utf-8")
    bound = _binding(implementation)
    registry = {
        "schema_version": 1,
        "contract_type": "safa_canonical_output_decoder_registry_v1",
        "pixel": {
            "decoder_type": "native_rgb_unit_interval",
            "output_range": [0.0, 1.0],
            "channels": 3,
            "height": 224,
            "width": 224,
            "model_type": "conditional_flow_matching",
            "sampler": "heun",
            "sample_steps": 32,
            "model_space": "rgb_neg1_pos1",
            "sample_api": "clamp_output=true",
            "clamp_output": True,
            "postprocess": (
                "in_generator_clamp_minus1_1_then_affine_then_"
                "clamp_unit_interval"
            ),
            "decoder_forbidden": True,
            "sampling_implementation": bound,
        },
        "latent": {
            "decoder_type": "r9_frozen_sd_vae_ft_ema",
            "vae_source_path": "artifacts/checkpoints/external/sd-vae-ft-ema",
            "directory": {"path": str(tmp_path), "digest": "a" * 64},
            "config": bound,
            "weights": bound,
            "scaling_factor": 0.18215,
            "implementation": bound,
            "trusted_runtime_config": bound,
            "trusted_runner": bound,
            "trusted_reference_checkpoint": bound,
            "trusted_resolved_config": bound,
            "trusted_generation_result": bound,
            "environment": {
                "provenance_snapshot": bound,
                "packages_sha256": (
                    "35196c0c7f5a8a2db3dcb31a67c0102"
                    "fbd713db6d67af72eacfffe8f8b82be7b"
                ),
                "python_version": "3.12.13",
                "torch_version": "2.11.0+cu128",
                "diffusers_version": "0.38.0",
            },
            "directory_digest_algorithm": (
                "sha256_relative_posix_nul_content_nul_v1"
            ),
            "asset_digest_cache": {"path": str(tmp_path / "cache.json")},
            "asset_digest_cache_algorithm": bound,
            "latent_shape": ["B", 4, 32, 32],
            "decoded_rgb_shape": ["B", 3, 256, 256],
            "output_range": [0.0, 1.0],
        },
        "decoder_registry_sha256": "",
    }
    registry["decoder_registry_sha256"] = decoder_registry_digest(registry)
    return registry


def _pixel_payload() -> dict:
    return {
        "model_config": {
            "model_type": "conditional_flow_matching",
            "embedding_dim": 512,
            "image_size": 224,
            "base_channels": 32,
            "channel_multipliers": [1, 2, 4, 4],
            "condition_dim": 512,
            "sample_steps": 32,
            "train_cycle_steps": 8,
            "sampler": "heun",
        },
        "training_config": {},
    }


def _latent_payload() -> dict:
    return {
        "model_config": {
            "model_type": "meanflow_sit",
            "embedding_dim": 512,
            "image_size": 32,
            "base_channels": 32,
            "channel_multipliers": [1, 2, 4, 4],
            "condition_dim": 512,
            "sample_steps": 1,
            "train_cycle_steps": 8,
            "sampler": "meanflow",
            "sit_input_channels": 4,
            "sit_data_space": "latent",
        },
        "training_config": {},
    }


def test_registered_pixel_and_latent_capabilities_are_exact(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    pixel = bind_output_contract(
        resolve_checkpoint_output_capability(_pixel_payload(), "1" * 64),
        registry,
    )
    latent = bind_output_contract(
        resolve_checkpoint_output_capability(_latent_payload(), "2" * 64),
        registry,
    )
    assert pixel["decoder_mode"] == "forbidden"
    assert pixel["decoder"] is None
    assert pixel["capability"]["nfe"] == 63
    assert pixel["capability"]["native_output_size"] == [224, 224]
    assert pixel["quality_protocol_family"] == "native_rgb_224_cfm"
    assert latent["decoder_mode"] == "required"
    assert latent["decoder"] is not None
    assert latent["capability"]["nfe"] == 1
    assert latent["capability"]["native_output_size"] == [32, 32]
    assert latent["rgb_contract"]["height"] == 256


def test_pixel_backend_validates_native_rgb_without_decoder(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    contract = bind_output_contract(
        resolve_checkpoint_output_capability(_pixel_payload(), "3" * 64),
        registry,
    )
    output = torch.rand(8, 3, 224, 224)
    assert validate_native_generator_output(output, contract, registry) is output
    assert build_bound_decoder(contract, registry, "cpu") is None


def test_missing_or_wrong_decoder_contract_fails_closed(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    capability = resolve_checkpoint_output_capability(
        _latent_payload(),
        "4" * 64,
    )
    missing = json.loads(json.dumps(registry))
    del missing["latent"]
    missing["decoder_registry_sha256"] = decoder_registry_digest(missing)
    with pytest.raises(GeneratorOutputContractError):
        bind_output_contract(capability, missing)
    wrong = json.loads(json.dumps(registry))
    wrong["latent"]["scaling_factor"] = 0.5
    wrong["decoder_registry_sha256"] = decoder_registry_digest(wrong)
    with pytest.raises(
        GeneratorOutputContractError,
        match="latent decoder registry semantics",
    ):
        bind_output_contract(capability, wrong)
    tampered = json.loads(json.dumps(registry))
    tampered["pixel"]["sample_steps"] = 31
    with pytest.raises(GeneratorOutputContractError, match="digest"):
        validate_decoder_registry(tampered)


def test_native_tensor_and_rgb_range_fail_closed(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    contract = bind_output_contract(
        resolve_checkpoint_output_capability(_pixel_payload(), "5" * 64),
        registry,
    )
    with pytest.raises(GeneratorOutputContractError, match="shape"):
        validate_native_generator_output(
            torch.zeros(8, 4, 224, 224),
            contract,
            registry,
        )
    with pytest.raises(GeneratorOutputContractError, match=r"\[0,1\]"):
        validate_rgb_unit_interval(
            torch.full((8, 3, 224, 224), 1.1),
            "pixel test",
            contract["rgb_contract"],
        )


def test_asset_digest_cache_survives_metadata_change_and_rejects_bytes(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.bin"
    cache = tmp_path / "cache.json"
    asset.write_bytes(b"frozen")
    expected = hashlib.sha256(b"frozen").hexdigest()
    assert cached_asset_digest(asset, expected, cache) == expected
    stat = asset.stat()
    os.utime(asset, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    assert cached_asset_digest(asset, expected, cache) == expected
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset digest mismatch"):
        cached_asset_digest(asset, expected, cache)


@pytest.mark.skipif(
    os.environ.get("SAFA_RUN_REAL_OUTPUT_CONTRACT_INTEGRATION") != "1",
    reason="explicit real frozen-VAE CPU integration only",
)
def test_real_r9_frozen_vae_decodes_eight_latents_on_cpu() -> None:
    from safa.closeout.canonical_screening import validate_policy

    policy_path = ROOT / "configs/closeout/canonical_screening_512_v1.json"
    policy = validate_policy(
        ROOT,
        policy_path,
        verify_historical_output_evidence=False,
    )
    registry = policy["output_decoder_registry"]
    contract = bind_output_contract(
        resolve_checkpoint_output_capability(_latent_payload(), "6" * 64),
        registry,
    )
    decoder = build_bound_decoder(contract, registry, "cpu")
    assert decoder is not None
    rgb = decoder.decode(torch.zeros(8, 4, 32, 32))
    validate_rgb_unit_interval(rgb, "real R9 VAE CPU decode", contract["rgb_contract"])
    assert list(rgb.shape) == [8, 3, 256, 256]


@pytest.mark.skipif(
    os.environ.get("SAFA_RUN_REAL_OUTPUT_CONTRACT_INTEGRATION") != "1",
    reason="explicit real historical checkpoint CPU integration only",
)
def test_real_e2_checkpoint_strictly_binds_pixel_backend_on_cpu() -> None:
    from safa.closeout.canonical_screening import validate_policy
    from safa.evaluation.checkpoint_preflight import (
        strict_load_generator_checkpoint,
    )

    policy = validate_policy(
        ROOT,
        ROOT / "configs/closeout/canonical_screening_512_v1.json",
        verify_historical_output_evidence=False,
    )
    checkpoint = (
        ROOT
        / "artifacts/checkpoints/e2_pu_adamw_200ep_gpu2_3_v2/"
        "best_raw_quality.pt"
    )
    generator, result = strict_load_generator_checkpoint(
        checkpoint,
        "raw",
        "cpu",
        expected_checkpoint_sha256=(
            "10b333a6c4b88e464be7c380758ae5a1"
            "40a10305c6708d9594517f102ceb929a"
        ),
        compute_sha256=True,
        output_decoder_registry=policy["output_decoder_registry"],
    )
    assert result["status"] == "valid"
    contract = result["output_contract"]
    assert contract["decoder_mode"] == "forbidden"
    assert contract["quality_protocol_family"] == "native_rgb_224_cfm"
    assert contract["capability"]["nfe"] == 63
    assert build_bound_decoder(
        contract,
        policy["output_decoder_registry"],
        "cpu",
    ) is None
    assert generator.config.model_type == "conditional_flow_matching"
