from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from safa.evaluation.r9_semigroup_contracts import (
    build_r9_semigroup_gate_contract,
    canonical_r9_semigroup_preflight_digest,
    canonical_r9_semigroup_preflight_payload,
    finalize_r9_semigroup_preflight,
    validate_r9_locked_schedule_bindings,
    validate_r9_semigroup_gate_contract,
    validate_r9_semigroup_preflight_config,
)
from safa.evaluation.r9_determinism import (
    canonical_json_sha256,
    canonical_r9_arm_config_digest,
    validate_r9_execution_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
R9_CONFIG = (
    REPO_ROOT
    / "configs"
    / "medium_v2"
    / "experiments"
    / "r9_meanflow_semigroup_preflight.yaml"
)


def _config() -> dict:
    payload = yaml.safe_load(R9_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_r9_preflight_contract_binds_determinism_backend_assets_and_schedule() -> None:
    config = _config()
    payload = validate_r9_semigroup_preflight_config(config)

    assert canonical_r9_semigroup_preflight_digest(config) == config[
        "semigroup_preflight_contract_sha256"
    ]
    assert payload["attention_backend"] == "native"
    assert payload["determinism_policy"] == config["determinism_policy"]
    assert payload["checkpoint"]["sha256"] == config["checkpoint_sha256"]
    assert payload["sample_manifest"]["sha256"] == config[
        "sample_id_manifest_sha256"
    ]
    assert payload["sample_manifest"]["sample_count"] == 64
    assert payload["schedule"] == {
        "registered_t_cut_candidates": [0.75, 0.5, 0.25],
        "split_times": [0.25, 0.5, 0.75],
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_sha256", "0" * 64),
        ("sample_id_manifest_sha256", "1" * 64),
        ("max_samples", 63),
        ("split_times", [0.25, 0.5]),
        ("registered_t_cut_candidates", [0.25, 0.5, 0.75]),
    ),
)
def test_r9_preflight_contract_rejects_bound_field_tampering(field: str, value) -> None:
    config = {**_config(), field: value}

    with pytest.raises(ValueError):
        validate_r9_semigroup_preflight_config(config)


def test_r9_gate_contract_binds_preflight_report_and_selected_schedule() -> None:
    config = _config()
    gate = build_r9_semigroup_gate_contract(
        config,
        effective_config_sha256="c" * 64,
        semigroup_report_sha256="a" * 64,
        gate_passed=True,
        selected_t_cut=0.25,
        schedule_contract_sha256="b" * 64,
    )

    assert validate_r9_semigroup_gate_contract(gate, config) == gate
    assert gate["preflight_contract_sha256"] == config[
        "semigroup_preflight_contract_sha256"
    ]
    assert gate["determinism_policy_sha256"] == config[
        "determinism_policy_sha256"
    ]
    assert gate["checkpoint_sha256"] == config["checkpoint_sha256"]
    assert gate["sample_id_manifest_sha256"] == config[
        "sample_id_manifest_sha256"
    ]
    assert gate["selected_t_cut"] == 0.25
    assert gate["schedule_contract_sha256"] == "b" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("attention_backend", "auto"),
        ("determinism_policy_sha256", "0" * 64),
        ("checkpoint_sha256", "1" * 64),
        ("sample_id_manifest_sha256", "2" * 64),
        ("split_times", [0.25, 0.5]),
        ("selected_t_cut", 0.5),
        ("schedule_contract_sha256", "3" * 64),
        ("gate_contract_sha256", "4" * 64),
    ),
)
def test_r9_gate_contract_rejects_any_binding_tamper(field: str, value) -> None:
    config = _config()
    gate = build_r9_semigroup_gate_contract(
        config,
        effective_config_sha256="c" * 64,
        semigroup_report_sha256="a" * 64,
        gate_passed=True,
        selected_t_cut=0.25,
        schedule_contract_sha256="b" * 64,
    )
    tampered = deepcopy(gate)
    tampered[field] = value

    with pytest.raises(ValueError):
        validate_r9_semigroup_gate_contract(tampered, config)


def test_failed_r9_gate_cannot_claim_a_selected_schedule() -> None:
    config = _config()
    with pytest.raises(ValueError, match="must not bind"):
        build_r9_semigroup_gate_contract(
            config,
            effective_config_sha256="c" * 64,
            semigroup_report_sha256="a" * 64,
            gate_passed=False,
            selected_t_cut=0.25,
            schedule_contract_sha256="b" * 64,
        )


def test_canonical_preflight_payload_does_not_include_declared_digest() -> None:
    config = _config()
    payload = canonical_r9_semigroup_preflight_payload(config)

    assert "semigroup_preflight_contract_sha256" not in payload


def _sample_digest(sample_ids: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode()
    ).hexdigest()


def _write_merge_fixture(tmp_path: Path) -> tuple[dict, list[Path]]:
    sample_ids = [f"sample-{index:02d}" for index in range(64)]
    manifest = tmp_path / "calibration_64.jsonl"
    manifest.write_text(
        "".join(json.dumps({"sample_id": sample_id}) + "\n" for sample_id in sample_ids),
        encoding="utf-8",
    )
    config = {
        **_config(),
        "sample_id_manifest": str(manifest),
        "sample_id_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "calibration_sample_id_manifest": str(manifest),
        "calibration_sample_id_manifest_sha256": hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest(),
    }
    config["semigroup_preflight_contract_sha256"] = (
        canonical_r9_semigroup_preflight_digest(config)
    )
    arm_sha256 = canonical_r9_arm_config_digest(config)
    execution = validate_r9_execution_config(config)
    effective = {**config, "arm_config_sha256": arm_sha256}
    shard_dirs = []
    for shard_index in range(4):
        shard_dir = tmp_path / "shards" / f"shard_{shard_index}"
        shard_dir.mkdir(parents=True)
        ids = sample_ids[shard_index::4]
        rows = []
        for sample_id in ids:
            rows.append(
                {
                    "sample_id": sample_id,
                    "splits": {
                        split: {
                            "latent_residual": 0.05,
                            "endpoint_e0_cosine": 0.99,
                            "decoded_pixel_l1": 0.01,
                            "decoded_psnr": 35.0,
                        }
                        for split in ("0.25", "0.5", "0.75")
                    },
                }
            )
        (shard_dir / "semigroup.json").write_text(
            json.dumps(
                {
                    "mode": "semigroup",
                    "split_times": [0.25, 0.5, 0.75],
                    "rows": rows,
                }
            ),
            encoding="utf-8",
        )
        generation = {
            "status": "complete",
            "mode": "semigroup",
            "sample_count": 16,
            "sample_id_sha256": _sample_digest(ids),
            "shard": {"index": shard_index, "count": 4},
            "arm_config_sha256": arm_sha256,
            "r9_execution_contract": execution,
            "checkpoint": {
                "sha256": config["checkpoint_sha256"],
                "attention_backend_requested": "native",
                "attention_backend_resolved": "native",
            },
            "config": effective,
            "resume_contract": {
                "input_sample_manifest": {
                    "sha256": config["sample_id_manifest_sha256"]
                }
            },
        }
        (shard_dir / "generation_result.json").write_text(
            json.dumps(generation), encoding="utf-8"
        )
        shard_dirs.append(shard_dir)
    return config, shard_dirs


def test_finalize_r9_preflight_produces_and_consumes_bound_gate(
    tmp_path: Path,
) -> None:
    config, shard_dirs = _write_merge_fixture(tmp_path)
    output = tmp_path / "final"

    finalized = finalize_r9_semigroup_preflight(
        config,
        shard_dirs,
        output_dir=output,
        visual_pass_by_split={"0.25": True, "0.5": True, "0.75": True},
    )

    assert finalized["gate_passed"] is True
    assert finalized["report"]["sample_count"] == 64
    assert finalized["report"]["selected_t_cut"] == 0.25
    assert finalized["gate_contract"]["attention_backend_requested"] == "native"
    assert finalized["gate_contract"]["attention_backend_resolved"] == "native"
    assert finalized["gate_contract"]["arm_config_sha256"] == canonical_r9_arm_config_digest(
        config
    )
    assert finalized["gate_contract"]["effective_config_sha256"] == canonical_json_sha256(
        {**config, "arm_config_sha256": canonical_r9_arm_config_digest(config)}
    )
    schedule = finalized["schedule"]
    guided_config = {
        **config,
        "mode": "paper_algorithm_split",
        "phase": "diagnose",
        "semigroup_report": schedule["semigroup_report"],
        "semigroup_sample_id_manifest": schedule["semigroup_sample_id_manifest"],
        "semigroup_sample_id_manifest_sha256": schedule[
            "semigroup_sample_id_manifest_sha256"
        ],
        "semigroup_preflight_contract": schedule["semigroup_preflight_contract"],
        "r9_semigroup_gate_contract": schedule["r9_semigroup_gate_contract"],
        "r9_semigroup_gate_contract_sha256": schedule[
            "r9_semigroup_gate_contract_sha256"
        ],
    }
    assert validate_r9_locked_schedule_bindings(guided_config, schedule) == finalized[
        "gate_contract"
    ]


def test_merge_rejects_resolved_backend_tampering(tmp_path: Path) -> None:
    config, shard_dirs = _write_merge_fixture(tmp_path)
    generation_path = shard_dirs[2] / "generation_result.json"
    payload = json.loads(generation_path.read_text(encoding="utf-8"))
    payload["checkpoint"]["attention_backend_resolved"] = "sdpa"
    generation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="attention_backend_resolved"):
        finalize_r9_semigroup_preflight(
            config,
            shard_dirs,
            output_dir=tmp_path / "final",
            visual_pass_by_split={"0.25": True, "0.5": True, "0.75": True},
        )


@pytest.mark.parametrize(
    "digest_field",
    (
        "semigroup_preflight_contract_sha256",
        "r9_semigroup_gate_contract_sha256",
    ),
)
def test_guided_consumer_rejects_config_digest_that_disagrees_with_schedule(
    tmp_path: Path, digest_field: str
) -> None:
    config, shard_dirs = _write_merge_fixture(tmp_path)
    finalized = finalize_r9_semigroup_preflight(
        config,
        shard_dirs,
        output_dir=tmp_path / "final",
        visual_pass_by_split={"0.25": True, "0.5": True, "0.75": True},
    )
    schedule = finalized["schedule"]
    guided_config = {
        **config,
        "mode": "paper_algorithm_split",
        "phase": "diagnose",
        "semigroup_report": schedule["semigroup_report"],
        "semigroup_sample_id_manifest": schedule["semigroup_sample_id_manifest"],
        "semigroup_sample_id_manifest_sha256": schedule[
            "semigroup_sample_id_manifest_sha256"
        ],
        "semigroup_preflight_contract": schedule["semigroup_preflight_contract"],
        "semigroup_preflight_contract_sha256": schedule[
            "semigroup_preflight_contract_sha256"
        ],
        "r9_semigroup_gate_contract": schedule["r9_semigroup_gate_contract"],
        "r9_semigroup_gate_contract_sha256": schedule[
            "r9_semigroup_gate_contract_sha256"
        ],
    }
    guided_config[digest_field] = "9" * 64

    with pytest.raises(ValueError, match=f"config {digest_field}"):
        validate_r9_locked_schedule_bindings(guided_config, schedule)
