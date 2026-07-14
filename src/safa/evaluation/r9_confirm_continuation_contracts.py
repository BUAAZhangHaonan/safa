from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from safa.evaluation.r9_calibration_selection_contracts import (
    CHILD_CAMPAIGN_ID,
    SOURCE_CAMPAIGN_ID,
    SOURCE_GATE_SHA256,
    SOURCE_PHASE_RESULTS_SHA256,
    SOURCE_REPAIR_SHA256,
    calibration_selection_contract_binding,
    build_calibration_report_only_selection_contract,
    validate_calibration_report_only_selection_contract,
)


SEMIGROUP_CLOSURE_CAMPAIGN_ID = "r9-report-only-formal-v2"
SOURCE_CONTINUATION_SHA256 = (
    "b4a532681de748eb8ac93cd0480b854bc76de26896ca88d8b5f29fd9ff9ca4ba"
)


class ConfirmContinuationContractError(ValueError):
    """Raised when the frozen B-to-C continuation chain is invalid."""


def build_confirm_continuation_contract(
    *,
    repo_root: Path,
    selection: Mapping[str, Any] | None = None,
    child_campaign_id: str = CHILD_CAMPAIGN_ID,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if child_campaign_id != CHILD_CAMPAIGN_ID:
        raise ConfirmContinuationContractError("confirm continuation child must be v7")
    locked_selection = validate_calibration_report_only_selection_contract(
        (
            build_calibration_report_only_selection_contract(repo_root=root)
            if selection is None
            else selection
        ),
        repo_root=root,
    )
    selection_path, selection_content, selection_binding = (
        calibration_selection_contract_binding(locked_selection, repo_root=root)
    )
    source_root = (
        root
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / SOURCE_CAMPAIGN_ID
    )
    runtime_path = source_root / "campaign_runtime.json"
    runtime = _read_contract(runtime_path, "campaign_runtime_sha256")
    if runtime.get("campaign_id") != SOURCE_CAMPAIGN_ID:
        raise ConfirmContinuationContractError("source runtime campaign changed")
    source_binding = _mapping(runtime.get("continuation"), "source continuation")
    source_continuation_path = _repo_file(
        root, source_binding.get("path"), "source continuation"
    )
    if _file_sha256(source_continuation_path) != _sha(
        source_binding.get("file_sha256"), "source continuation file SHA256"
    ):
        raise ConfirmContinuationContractError("source continuation file changed")
    source_continuation = _read_contract(
        source_continuation_path, "continuation_contract_sha256"
    )
    if (
        source_binding.get("contract_sha256") != SOURCE_CONTINUATION_SHA256
        or source_continuation.get("continuation_contract_sha256")
        != SOURCE_CONTINUATION_SHA256
        or source_continuation.get("child_campaign_id") != SOURCE_CAMPAIGN_ID
        or _mapping(source_continuation.get("parent"), "source parent").get(
            "campaign_id"
        )
        != SEMIGROUP_CLOSURE_CAMPAIGN_ID
    ):
        raise ConfirmContinuationContractError("source continuation chain changed")
    source = _mapping(locked_selection.get("source"), "selection source")
    if (
        source.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or source.get("gate_contract_sha256") != SOURCE_GATE_SHA256
        or source.get("phase_results_sha256") != SOURCE_PHASE_RESULTS_SHA256
        or source.get("evaluation_repair_contract_sha256") != SOURCE_REPAIR_SHA256
    ):
        raise ConfirmContinuationContractError("selection source changed")
    selected_arms = locked_selection.get("selected_arms")
    if not isinstance(selected_arms, list) or len(selected_arms) != 2:
        raise ConfirmContinuationContractError("confirm continuation requires two arms")
    manifests = _mapping(runtime.get("manifests"), "source manifests")
    payload = {
        "schema_version": 1,
        "contract_type": "safa_r9_confirm_continuation_contract_v1",
        "child_campaign_id": CHILD_CAMPAIGN_ID,
        "start_phase": "confirm512",
        "source_campaign_id": SOURCE_CAMPAIGN_ID,
        "semigroup_closure_campaign_id": SEMIGROUP_CLOSURE_CAMPAIGN_ID,
        "selection": {
            **selection_binding,
            "path": str(selection_path.relative_to(root)),
            "prospective_file_sha256": hashlib.sha256(selection_content).hexdigest(),
        },
        "source": {
            "runtime": _binding(
                root,
                runtime_path,
                _sha(runtime.get("campaign_runtime_sha256"), "source runtime SHA256"),
            ),
            "continuation": _binding(
                root, source_continuation_path, SOURCE_CONTINUATION_SHA256
            ),
            "calibrate_gate_contract_sha256": SOURCE_GATE_SHA256,
            "calibrate_phase_results_sha256": SOURCE_PHASE_RESULTS_SHA256,
            "evaluation_repair_contract_sha256": SOURCE_REPAIR_SHA256,
        },
        "selected_arms": [dict(row) for row in selected_arms],
        "bindings": {
            "manifest_contracts_sha256": _sha(
                runtime.get("manifest_contracts_sha256"),
                "manifest contracts SHA256",
            ),
            "manifests": {
                name: dict(_mapping(manifests.get(name), f"manifest {name}"))
                for name in sorted(manifests)
            },
            "checkpoint": dict(_mapping(runtime.get("checkpoint"), "checkpoint")),
            "determinism_policy_sha256": _sha(
                runtime.get("determinism_policy_sha256"),
                "determinism policy SHA256",
            ),
            "attention_backend": runtime.get("attention_backend"),
            "schedule": dict(_mapping(runtime.get("schedule"), "schedule")),
            "semigroup_gate": dict(
                _mapping(runtime.get("semigroup_gate"), "semigroup gate")
            ),
            "implementations": {
                path: _actual_file_binding(root, path)
                for path in (
                    "scripts/run_r9_meanflow_campaign.py",
                    "scripts/run_r9_phase_evaluator.py",
                    "src/safa/evaluation/r9_evaluator_worker.py",
                    "src/safa/evaluation/r9_phase_results.py",
                    "src/safa/evaluation/r9_calibration_selection_contracts.py",
                    "src/safa/evaluation/r9_confirm_continuation_contracts.py",
                )
            },
        },
        "policy": {
            "source_gate_mutation": False,
            "source_gate_usage": "superseded_evidence_only",
            "selection_role": "report_only_promotion_decision",
            "allowed_phases": ["confirm512", "full"],
            "cli_algorithm_overrides": False,
            "retry_count": 0,
        },
    }
    if payload["bindings"]["attention_backend"] != "native":
        raise ConfirmContinuationContractError("confirm continuation requires native attention")
    payload["confirm_continuation_sha256"] = _canonical_digest(
        payload, "confirm_continuation_sha256"
    )
    return payload


def validate_confirm_continuation_contract(
    value: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    normalized = _mapping(value, "confirm continuation")
    declared = _sha(
        normalized.get("confirm_continuation_sha256"),
        "confirm continuation SHA256",
    )
    if declared != _canonical_digest(normalized, "confirm_continuation_sha256"):
        raise ConfirmContinuationContractError("confirm continuation digest mismatch")
    expected = build_confirm_continuation_contract(repo_root=repo_root)
    if normalized != expected:
        raise ConfirmContinuationContractError(
            "confirm continuation disagrees with frozen evidence"
        )
    return normalized


def confirm_continuation_contract_binding(
    payload: Mapping[str, Any], *, repo_root: Path
) -> tuple[Path, bytes, dict[str, str]]:
    normalized = _mapping(payload, "confirm continuation")
    declared = _sha(
        normalized.get("confirm_continuation_sha256"),
        "confirm continuation SHA256",
    )
    if declared != _canonical_digest(normalized, "confirm_continuation_sha256"):
        raise ConfirmContinuationContractError("confirm continuation digest mismatch")
    root = Path(repo_root).resolve()
    path = (
        root
        / "artifacts/r9_meanflow_flow_map_guidance/campaigns"
        / CHILD_CAMPAIGN_ID
        / "confirm_continuation_contract.json"
    )
    content = _contract_bytes(normalized)
    return path, content, {
        "path": str(path.relative_to(root)),
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "contract_sha256": declared,
    }


def materialize_confirm_continuation_contract(
    *, repo_root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    selection = build_calibration_report_only_selection_contract(repo_root=repo_root)
    payload = build_confirm_continuation_contract(
        repo_root=repo_root, selection=selection
    )
    path, content, binding = confirm_continuation_contract_binding(
        payload, repo_root=repo_root
    )
    _write_exclusive(path, content)
    return payload, binding


def _actual_file_binding(root: Path, relative: str) -> dict[str, str]:
    path = _repo_file(root, relative, "implementation")
    return {"path": relative, "sha256": _file_sha256(path)}


def _binding(root: Path, path: Path, contract_sha256: str) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "file_sha256": _file_sha256(path),
        "contract_sha256": contract_sha256,
    }


def _repo_file(root: Path, value: Any, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute():
        raise ConfirmContinuationContractError(f"{label} path must be relative")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise ConfirmContinuationContractError(f"{label} is not a regular repo file")
    return path


def _read_contract(path: Path, field: str) -> dict[str, Any]:
    value = _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    declared = _sha(value.get(field), field)
    if declared != _canonical_digest(value, field):
        raise ConfirmContinuationContractError(f"{field} digest mismatch")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _contract_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise ConfirmContinuationContractError("confirm continuation already differs")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
        os.link(temporary, path)
    except FileExistsError as error:
        raise ConfirmContinuationContractError("confirm continuation creation raced") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfirmContinuationContractError(f"{label} must be a mapping")
    return dict(value)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ConfirmContinuationContractError(f"{label} must be lowercase SHA256")
    return value
