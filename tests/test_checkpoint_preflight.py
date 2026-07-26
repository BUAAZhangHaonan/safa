from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

from safa.evaluation.checkpoint_preflight import (  # noqa: E402
    preflight_generator_checkpoint,
    strict_load_generator_checkpoint,
)
from safa.models.generator import build_generator  # noqa: E402
from safa.models.ip_adapter import (  # noqa: E402
    wrap_backbone_with_condition_mlp,
    wrap_backbone_with_ip_adapter,
)
from safa.models.peft_lora import wrap_backbone_with_lora_target  # noqa: E402
from safa.training.peft_runner import (  # noqa: E402
    init_peft_lora_generator,
    peft_lora_objective_from_config,
)
from safa.utils.hashing import sha256_file  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_checkpoint_preflight.py"
SPEC = importlib.util.spec_from_file_location("run_checkpoint_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _config() -> dict:
    return {
        "model_type": "meanflow_sit",
        "embedding_dim": 16,
        "image_size": 16,
        "base_channels": 4,
        "channel_multipliers": [1],
        "time_embedding_dim": 8,
        "condition_dim": 16,
        "sample_steps": 1,
        "train_cycle_steps": 1,
        "sampler": "meanflow",
        "learned_null_condition": True,
        "meanflow_ratio": 0.25,
        "meanflow_ratio_r_not_equal_t": 0.75,
        "meanflow_adaptive_weighting": True,
        "meanflow_norm_p": 1.0,
        "meanflow_norm_eps": 0.001,
        "meanflow_jvp_mode": "torch_func",
        "sit_input_channels": 3,
        "sit_patch_size": 4,
        "sit_hidden_size": 32,
        "sit_depth": 2,
        "sit_num_heads": 4,
        "sit_mlp_ratio": 2.0,
        "sit_time_embedding_dim": 32,
    }


def _objective(**overrides) -> dict:
    objective = {
        "type": "peft_lora",
        "ffhq_index": "unused-by-preflight.jsonl",
        "enable_lora": True,
        "enable_gated_low_rank": False,
        "enable_generic_bank": False,
        "generic_mode": "null",
        "freeze_null_embed": True,
        "lora_rank": 4,
        "num_generic_embeddings": 16,
        "lora_blocks": "all",
        "step_ratio": 0,
    }
    objective.update(overrides)
    return objective


def _payload(generator, objective: dict | None = None, **extra) -> dict:
    payload = {
        "model_config": generator.config.to_dict(),
        "model_state_dict": generator.state_dict(),
    }
    if objective is not None:
        payload["training_config"] = {
            "stages": {"stage2": {"stage2_objective": objective}}
        }
    payload.update(extra)
    return payload


def _lora_payload(objective: dict | None = None) -> dict:
    objective = _objective() if objective is None else objective
    generator = build_generator(_config())
    parsed = peft_lora_objective_from_config(objective, "test.objective")
    init_peft_lora_generator(generator, parsed)
    return _payload(generator, objective)


def _other_adapter_payload(objective: dict) -> dict:
    generator = build_generator(_config())
    objective_type = objective["type"]
    if objective_type == "peft_fm":
        wrap_backbone_with_ip_adapter(
            generator.vector_field,
            ip_adapter_layers=objective["ip_adapter_layers"],
            num_z_tokens=objective["ip_adapter_num_tokens"],
        )
    elif objective_type == "peft_mlp":
        wrap_backbone_with_condition_mlp(generator.vector_field)
    elif objective_type in {"lora_sweep", "point_projected_two_step"}:
        wrap_backbone_with_lora_target(
            generator.vector_field,
            target_modules=objective["lora_target_modules"],
            rank=objective["lora_rank"],
            alpha=objective["lora_alpha"],
        )
    else:
        raise AssertionError(f"unsupported test objective {objective_type}")
    return _payload(generator, objective)


def test_plain_checkpoint_strict_load_and_cpu_smoke_8(tmp_path: Path) -> None:
    path = tmp_path / "plain.pt"
    torch.save(_payload(build_generator(_config())), path)

    generator, result = strict_load_generator_checkpoint(
        path,
        "raw",
        "cpu",
        compute_sha256=True,
        smoke_samples=8,
    )

    assert result["status"] == "valid"
    assert result["checkpoint_sha256"]
    assert result["selector_binding"] == "single_available_state_dict"
    assert result["adapter"]["type"] == "none"
    assert result["missing_keys"] == []
    assert result["unexpected_keys"] == []
    assert result["smoke"]["executed_sample_count"] == 8
    assert result["smoke"]["output_shape"] == [8, 3, 16, 16]
    assert generator._safa_checkpoint_preflight["status"] == "valid"


def test_expected_sha256_is_bound_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "plain.pt"
    torch.save(_payload(build_generator(_config())), path)
    actual_sha256 = sha256_file(path)

    _, valid = strict_load_generator_checkpoint(
        path,
        "raw",
        "cpu",
        expected_checkpoint_sha256=actual_sha256,
    )
    assert valid["checkpoint_sha256"] == actual_sha256
    assert valid["expected_checkpoint_sha256"] == actual_sha256
    assert valid["sha256_binding"] == "expected_exact"

    def must_not_deserialize(*args, **kwargs):
        raise AssertionError("SHA mismatch must fail before torch.load")

    monkeypatch.setattr(torch, "load", must_not_deserialize)
    invalid = preflight_generator_checkpoint(
        path,
        "raw",
        expected_checkpoint_sha256="0" * 64,
        compute_sha256=False,
    )
    assert invalid["status"] == "invalid"
    assert invalid["failure_code"] == "checkpoint_sha256_mismatch"
    assert invalid["checkpoint_sha256"] == actual_sha256
    assert invalid["expected_checkpoint_sha256"] == "0" * 64
    assert invalid["sha256_binding"] == "expected_mismatch"


def test_invalid_expected_sha256_fails_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "plain.pt"
    torch.save(_payload(build_generator(_config())), path)

    def must_not_deserialize(*args, **kwargs):
        raise AssertionError("invalid expected SHA must fail before torch.load")

    monkeypatch.setattr(torch, "load", must_not_deserialize)
    result = preflight_generator_checkpoint(
        path,
        "raw",
        expected_checkpoint_sha256="NOT-A-SHA",
    )
    assert result["failure_code"] == "invalid_expected_checkpoint_sha256"
    assert result["checkpoint_sha256"] is None


def test_lora_checkpoint_reproduces_bare_loader_bug_then_mounts_strictly(
    tmp_path: Path,
) -> None:
    payload = _lora_payload()
    path = tmp_path / "lora.pt"
    torch.save(payload, path)

    bare = build_generator(_config())
    incompatible = bare.load_state_dict(payload["model_state_dict"], strict=False)
    assert any("lora_a" in key for key in incompatible.unexpected_keys)
    with pytest.raises(RuntimeError):
        build_generator(_config()).load_state_dict(
            payload["model_state_dict"],
            strict=True,
        )

    generator, result = strict_load_generator_checkpoint(path, "raw", "cpu")
    assert result["status"] == "valid"
    assert result["adapter"]["type"] == "peft_lora"
    assert result["adapter"]["mounted"] is True
    assert result["adapter"]["state_key_count"] > 0
    assert result["adapter"]["mounted_key_count"] == result["adapter"]["state_key_count"]
    assert getattr(generator.vector_field, "_peft_lora_wrapped") is True


@pytest.mark.parametrize(
    ("objective", "expected_adapter"),
    [
        (
            {
                "type": "peft_fm",
                "ip_adapter_layers": [0, 1],
                "ip_adapter_num_tokens": 2,
            },
            "ip_adapter",
        ),
        (
            {"type": "peft_mlp", "lambda_repr": 1.0, "repr_interval": 1},
            "peft_mlp",
        ),
        (
            {
                "type": "lora_sweep",
                "lora_target_modules": ["attn.qkv"],
                "lora_rank": 4,
                "lora_alpha": 2.0,
            },
            "lora_target",
        ),
        (
            {
                "type": "point_projected_two_step",
                "lora_target_modules": ["attn.proj"],
                "lora_rank": 4,
                "lora_alpha": 2.0,
            },
            "lora_target",
        ),
    ],
)
def test_every_recorded_adapter_loader_mounts_exactly(
    tmp_path: Path,
    objective: dict,
    expected_adapter: str,
) -> None:
    path = tmp_path / f"{objective['type']}.pt"
    torch.save(_other_adapter_payload(objective), path)

    _, result = strict_load_generator_checkpoint(path, "raw", "cpu")

    assert result["status"] == "valid"
    assert result["adapter"]["type"] == expected_adapter
    assert result["adapter"]["mounted"] is True
    assert result["adapter"]["state_key_count"] > 0
    assert result["adapter"]["mounted_key_count"] == result["adapter"]["state_key_count"]


def test_adapter_state_without_training_contract_hard_fails(tmp_path: Path) -> None:
    payload = _lora_payload()
    payload.pop("training_config")
    path = tmp_path / "missing-contract.pt"
    torch.save(payload, path)

    result = preflight_generator_checkpoint(path, "raw")

    assert result["status"] == "invalid"
    assert result["failure_code"] == "adapter_configuration_missing"


def test_wrong_adapter_contract_exposes_missing_and_unexpected_keys(
    tmp_path: Path,
) -> None:
    payload = _lora_payload()
    payload["training_config"]["stages"]["stage2"]["stage2_objective"][
        "generic_mode"
    ] = "bank"
    path = tmp_path / "wrong-contract.pt"
    torch.save(payload, path)

    result = preflight_generator_checkpoint(path, "raw")

    assert result["status"] == "invalid"
    assert result["failure_code"] == "state_dict_shape_mismatch"
    assert result["shape_mismatches"]


def test_nonfinite_tensor_hard_fails(tmp_path: Path) -> None:
    payload = _payload(build_generator(_config()))
    key = next(
        key
        for key, value in payload["model_state_dict"].items()
        if torch.is_floating_point(value)
    )
    payload["model_state_dict"][key].view(-1)[0] = float("nan")
    path = tmp_path / "nonfinite.pt"
    torch.save(payload, path)

    result = preflight_generator_checkpoint(path, "raw")

    assert result["status"] == "invalid"
    assert result["failure_code"] == "nonfinite_tensor"
    assert result["nonfinite_keys"] == [key]


def test_selector_mismatch_and_missing_ema_are_distinct(tmp_path: Path) -> None:
    generator = build_generator(_config())
    declared = tmp_path / "declared.pt"
    torch.save(_payload(generator, checkpoint_model="ema"), declared)
    missing = tmp_path / "missing-ema.pt"
    torch.save(_payload(generator), missing)

    declared_result = preflight_generator_checkpoint(declared, "raw")
    missing_result = preflight_generator_checkpoint(missing, "ema")

    assert declared_result["failure_code"] == "selector_mismatch"
    assert missing_result["failure_code"] == "selector_state_missing"


def test_raw_and_ema_both_available_bind_to_explicit_request(tmp_path: Path) -> None:
    generator = build_generator(_config())
    path = tmp_path / "both.pt"
    torch.save(
        _payload(
            generator,
            ema_model_state_dict=generator.state_dict(),
        ),
        path,
    )

    raw = preflight_generator_checkpoint(path, "raw")
    ema = preflight_generator_checkpoint(path, "ema")

    assert raw["status"] == "valid"
    assert ema["status"] == "valid"
    assert raw["selector_binding"] == "explicit_request_with_multiple_states"
    assert ema["selector_binding"] == "explicit_request_with_multiple_states"


def test_unexplained_key_mismatch_hard_fails(tmp_path: Path) -> None:
    payload = _payload(build_generator(_config()))
    payload["model_state_dict"]["unexplained.weight"] = torch.zeros(1)
    path = tmp_path / "unexpected.pt"
    torch.save(payload, path)

    result = preflight_generator_checkpoint(path, "raw")

    assert result["status"] == "invalid"
    assert result["failure_code"] == "state_dict_key_mismatch"
    assert result["unexpected_keys"] == ["unexplained.weight"]


def test_cli_emits_machine_readable_invalid_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(_payload(build_generator(_config())), path)

    exit_code = runner.main(
        [
            "--checkpoint",
            str(path),
            "--checkpoint-model",
            "ema",
            "--expected-sha256",
            sha256_file(path),
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert result["contract_type"] == "safa_generator_checkpoint_preflight_v1"
    assert result["status"] == "invalid"
    assert result["failure_code"] == "selector_state_missing"
    assert result["sha256_binding"] == "expected_exact"


def test_cli_valid_lora_status_is_one_json_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "lora.pt"
    torch.save(_lora_payload(), path)

    exit_code = runner.main(
        [
            "--checkpoint",
            str(path),
            "--checkpoint-model",
            "raw",
            "--expected-sha256",
            sha256_file(path),
        ]
    )
    rendered = capsys.readouterr().out
    result = json.loads(rendered)

    assert exit_code == 0
    assert rendered.count("\n") == 1
    assert result["status"] == "valid"
    assert result["adapter"]["mounted"] is True
    assert result["checkpoint_sha256"] == sha256_file(path)
    assert result["sha256_binding"] == "expected_exact"


def test_cli_skip_sha256_is_diagnostic_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "plain.pt"
    torch.save(_payload(build_generator(_config())), path)

    with pytest.raises(SystemExit, match="only allowed with --diagnostic"):
        runner.main(
            [
                "--checkpoint",
                str(path),
                "--checkpoint-model",
                "raw",
                "--skip-sha256",
            ]
        )
    with pytest.raises(SystemExit, match="required for formal preflight"):
        runner.main(
            [
                "--checkpoint",
                str(path),
                "--checkpoint-model",
                "raw",
            ]
        )
    assert (
        runner.main(
            [
                "--checkpoint",
                str(path),
                "--checkpoint-model",
                "raw",
                "--diagnostic",
                "--skip-sha256",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["checkpoint_sha256"] is None
    assert result["sha256_binding"] is None


def test_cli_formal_rejects_invalid_expected_sha256_without_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "plain.pt"
    torch.save(_payload(build_generator(_config())), path)
    with pytest.raises(SystemExit, match="lowercase SHA256"):
        runner.main(
            [
                "--checkpoint",
                str(path),
                "--checkpoint-model",
                "raw",
                "--expected-sha256",
                "not-a-sha",
            ]
        )
    assert capsys.readouterr().out == ""


def test_cli_out_json_is_immutable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "plain.pt"
    output = tmp_path / "result.json"
    torch.save(_payload(build_generator(_config())), checkpoint)
    arguments = [
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-model",
        "raw",
        "--expected-sha256",
        sha256_file(checkpoint),
        "--out-json",
        str(output),
    ]

    assert runner.main(arguments) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        runner.main(arguments)
    assert output.read_bytes() == original
