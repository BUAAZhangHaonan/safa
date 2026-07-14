from __future__ import annotations

import copy
from pathlib import Path

import pytest

from safa.evaluation.r9_confirm_continuation_contracts import (
    CHILD_CAMPAIGN_ID,
    SEMIGROUP_CLOSURE_CAMPAIGN_ID,
    ConfirmContinuationContractError,
    build_confirm_continuation_contract,
    confirm_continuation_contract_binding,
    validate_confirm_continuation_contract,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUNTIME = ROOT / (
    "artifacts/r9_meanflow_flow_map_guidance/campaigns/"
    "r9-report-only-formal-v6/campaign_runtime.json"
)
requires_frozen_b = pytest.mark.skipif(
    not SOURCE_RUNTIME.is_file(), reason="frozen on-server Phase-B evidence is absent"
)


@requires_frozen_b
def test_live_confirm_continuation_binds_exact_two_arms() -> None:
    result = build_confirm_continuation_contract(repo_root=ROOT)
    assert result["child_campaign_id"] == CHILD_CAMPAIGN_ID
    assert result["start_phase"] == "confirm512"
    assert result["semigroup_closure_campaign_id"] == SEMIGROUP_CLOSURE_CAMPAIGN_ID
    assert [row["arm_id"] for row in result["selected_arms"]] == [
        "paper_eta_0p125",
        "flow_map2_normalized_eta_0p125",
    ]
    validated = validate_confirm_continuation_contract(result, repo_root=ROOT)
    assert validated == result
    path, content, binding = confirm_continuation_contract_binding(
        result, repo_root=ROOT
    )
    assert path.name == "confirm_continuation_contract.json"
    assert len(content) > 0
    assert binding["contract_sha256"] == result["confirm_continuation_sha256"]


@requires_frozen_b
def test_confirm_continuation_rejects_selection_tamper() -> None:
    result = build_confirm_continuation_contract(repo_root=ROOT)
    tampered = copy.deepcopy(result)
    tampered["selected_arms"].reverse()
    with pytest.raises(ConfirmContinuationContractError, match="digest mismatch"):
        validate_confirm_continuation_contract(tampered, repo_root=ROOT)
