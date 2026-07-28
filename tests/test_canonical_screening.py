from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import shutil
import sys
import threading
import time
import types
from typing import Any, Mapping

import pytest

from safa.closeout.canonical_screening import (
    CanonicalScreeningError,
    CONTROLLER_LAUNCH_REHASH_CONTRACT,
    WORKER_EXTERNAL_GPU_RACE_CONTRACT,
    WORKER_PRE_CUDA_VERIFICATION_ORDER,
    WORKER_READY_CONTRACT,
    WORKER_RELEASE_CONTRACT,
    _require_no_repo_path_component_symlinks,
    _require_tree_without_symlinks,
    _validate_6b_failed_probe_root_identity,
    _validate_ram_probe_artifact_seal,
    _validate_ram_slot_budget_source,
    build_candidate_manifest,
    build_checkpoint_plan,
    build_preflight_result,
    build_run_claim,
    build_run_request,
    build_run_result,
    canonical_digest,
    canonical_gpu_registry,
    canonicalize_nvidia_gpu_uuid,
    canonical_json,
    hash_asset_directory_content,
    load_json,
    publish_exclusive_json,
    ram_probe_admission_evidence_digest,
    ram_probe_contract_digest,
    ram_probe_execution_digest,
    sha256_file,
    validate_arcface_execution_probe_binding,
    validate_candidate_manifest,
    validate_checkpoint_plan,
    validate_preflight_result,
    validate_run_request,
    validate_run_result,
    validate_controller_launch_rehash_value,
    validate_worker_ready_value,
    validate_worker_release_value,
    validate_worker_terminal_value,
    validate_supersession_evidence,
    validate_policy,
    write_exclusive_json,
)
from safa.closeout.canonical_screening_worker import (
    _assert_ready_barrier,
    _assert_runtime_cuda_binding,
    _load_arcface_contract,
    _load_source_pixel_batch,
    _representation_cosines,
    _wait_worker_release,
    _write_validated_run_result,
    execute_screening_request,
)
import safa.closeout.canonical_screening_worker as screening_worker_module
from safa.closeout.canonical_quality import evaluate_locked_kid
from safa.closeout.generator_output_contract import (
    bind_output_contract,
    decoder_registry_digest,
    resolve_checkpoint_output_capability,
)
from safa.closeout.preflight_launch_contract import (
    PreflightLaunchContractError,
    build_bound_lifecycle_evidence as build_preflight_bound_lifecycle_evidence,
    build_claim_v3 as build_preflight_claim_v3,
    build_file_identity as build_preflight_file_identity,
    build_gate_ready as build_preflight_gate_ready,
    build_launch_accepted as build_preflight_launch_accepted,
    build_ownership_release as build_preflight_ownership_release,
    build_ownership_terminal as build_preflight_ownership_terminal,
    build_pane_owner_seal as build_preflight_pane_owner_seal,
    build_process_identity as build_preflight_process_identity,
    build_tmux_server_identity as build_preflight_tmux_server_identity,
    build_tmux_started as build_preflight_tmux_started,
    build_verified_implementations as build_preflight_verified_implementations,
    build_wrapper_started as build_preflight_wrapper_started,
    canonical_digest as preflight_launch_digest,
    shared_contract_types,
    shared_schema_keysets,
    validate_claim_v3 as validate_preflight_claim_v3,
)
import safa.closeout.preflight_launch_contract as preflight_launch_contract_module


def _gpu_uuid(index: int) -> str:
    return f"GPU-0000000{index}-0000-0000-0000-00000000000{index}"


def _raw_controller_module():
    path = Path(__file__).parents[1] / "scripts" / "run_canonical_checkpoint_screening.py"
    spec = importlib.util.spec_from_file_location("canonical_controller_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _controller_module():
    module = _raw_controller_module()
    module._install_verified_contract_api(
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json",
        verify_historical_output_evidence=False,
    )
    return module


def _wrapper_module():
    path = Path(__file__).parents[1] / "scripts" / "run_canonical_preflight_wrapper.py"
    spec = importlib.util.spec_from_file_location("canonical_wrapper_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._install_verified_preflight_apis(
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    return module


def _launcher_module():
    path = (
        Path(__file__).parents[1]
        / "scripts/run_canonical_preflight_launcher.py"
    )
    spec = importlib.util.spec_from_file_location(
        "canonical_preflight_launcher_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._install_verified_preflight_apis(
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    return module


def _shared_binding(name: str) -> dict[str, str]:
    return {
        "path": f"/contract/{name}.json",
        "sha256": hashlib.sha256(f"{name}:file".encode()).hexdigest(),
        "canonical_sha256": hashlib.sha256(
            f"{name}:canonical".encode()
        ).hexdigest(),
    }


def _shared_file_identity(name: str) -> dict[str, Any]:
    return {
        "path": f"/contract/{name}",
        "device": 7,
        "inode": 11,
        "mode": 33188,
        "size": 101,
    }


def _shared_process(pid: int, *, ppid: int) -> dict[str, int]:
    return {
        "pid": pid,
        "ppid": ppid,
        "pgid": pid,
        "sid": pid,
        "start_ticks": pid * 10,
    }


def _shared_launch_receipt_v4() -> dict[str, Any]:
    namespace = "/contract/attempt/pane_fault_consumer"
    launcher_sha = hashlib.sha256(b"launcher").hexdigest()
    artifacts = {
        name: f"{namespace}/consumer_{name}.json"
        for name in (
            "ready",
            "started",
            "offer",
            "accepted",
            "commit",
            "active",
            "reader_release",
            "release_observed",
            "controller_cleanup",
            "terminal",
            "join",
            "cleanup",
            "wait_supervisor_ready",
        )
    }
    artifacts.update(
        {
            "attempt": f"{namespace}/consumer_attempt.json",
            "log": f"{namespace}/consumer.log",
            "self_fault_channel": (
                f"{namespace}/consumer_self_fault.channel"
            ),
            "lifecycle_wait_channel": (
                f"{namespace}/consumer_lifecycle_wait.channel"
            ),
        }
    )
    attempt_id = hashlib.sha256(b"receipt-attempt").hexdigest()
    gate_worker_arguments = [
        "/contract/python",
        "-B",
        "-u",
        "/contract/launcher.py",
        "__pane_gate__",
        "--attempt-root",
        "/contract/attempt",
        "--release-path",
        "/contract/attempt/pane_gate_release.json",
        "--log-path",
        "/contract/attempt/pane.log",
        "--wrapper-arguments-json",
        '["wrapper"]',
    ]
    supervisor_arguments = [
        "/contract/python",
        "-B",
        "-u",
        "/contract/launcher.py",
        "__gate_wait_supervisor__",
        "--launch-receipt",
        "/contract/attempt/launch_receipt.json",
        "--attempt-id",
        attempt_id,
        "--wait-channel-path",
        "/contract/attempt/gate_lifecycle_wait.channel",
        "--gate-worker-arguments-json",
        json.dumps(gate_worker_arguments, separators=(",", ":")),
    ]
    consumer_worker_arguments = [
        "/contract/python",
        "-B",
        "-u",
        "/contract/launcher.py",
        "__pane_fault_consumer__",
        "--attempt-path",
        f"{namespace}/consumer_attempt.json",
        "--config",
        "/contract/config.json",
    ]
    consumer_supervisor_arguments = [
        "/contract/python",
        "-B",
        "-u",
        "/contract/launcher.py",
        "__consumer_wait_supervisor__",
        "--attempt-path",
        f"{namespace}/consumer_attempt.json",
        "--config",
        "/contract/config.json",
        "--wait-channel-path",
        f"{namespace}/consumer_lifecycle_wait.channel",
        "--consumer-worker-arguments-json",
        json.dumps(consumer_worker_arguments, separators=(",", ":")),
    ]
    receipt = {
        "schema_version": 4,
        "contract_type": (
            "safa_canonical_preflight_launch_receipt_v4"
        ),
        "attempt_id": attempt_id,
        "started_registry": _shared_binding("started-registry"),
        "policy_sha256": hashlib.sha256(b"receipt-policy").hexdigest(),
        "git": {},
        "bindings": {"config": {"path": "/contract/config.json"}},
        "verified_implementations": {},
        "python_executable": {},
        "controller_session": "controller",
        "controller_owner_nonce": hashlib.sha256(b"owner").hexdigest(),
        "observer_session": "observer",
        "wrapper_arguments": ["wrapper"],
        "gate_lifecycle_wait_channel": {
            "path": "/contract/attempt/gate_lifecycle_wait.channel",
            "device": 7,
            "inode": 19,
            "mode": 33152,
            "uid": 1000,
            "nlink": 1,
            "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "directory_device": 7,
            "directory_inode": 17,
        },
        "gate_lifecycle_wait_publisher": {
            "path": "/contract/launcher.py",
            "sha256": launcher_sha,
            "file_identity": {
                **_shared_file_identity("launcher.py"),
                "path": "/contract/launcher.py",
            },
            "role": "gate_lifecycle_wait_supervisor",
        },
        "gate_lifecycle_wait_supervisor_arguments": (
            supervisor_arguments
        ),
        "gate_lifecycle_wait_supervisor_ready_path": (
            "/contract/attempt/gate_wait_supervisor_ready.json"
        ),
        "gate_lifecycle_wait_status_path": (
            "/contract/attempt/gate_lifecycle_wait.channel"
        ),
        "gate_worker_arguments": gate_worker_arguments,
        "consumer_lifecycle_wait_channel": {
            "path": (
                f"{namespace}/consumer_lifecycle_wait.channel"
            ),
            "device": 7,
            "inode": 23,
            "mode": 33152,
            "uid": 1000,
            "nlink": 1,
            "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "directory_device": 7,
            "directory_inode": 17,
        },
        "consumer_lifecycle_wait_publisher": {
            "path": "/contract/launcher.py",
            "sha256": launcher_sha,
            "file_identity": {
                **_shared_file_identity("launcher.py"),
                "path": "/contract/launcher.py",
            },
            "role": "consumer_lifecycle_wait_supervisor",
        },
        "consumer_lifecycle_wait_supervisor_arguments": (
            consumer_supervisor_arguments
        ),
        "consumer_lifecycle_wait_supervisor_ready_path": (
            f"{namespace}/consumer_wait_supervisor_ready.json"
        ),
        "consumer_lifecycle_wait_status_path": (
            f"{namespace}/consumer_lifecycle_wait.channel"
        ),
        "consumer_worker_arguments": consumer_worker_arguments,
        "consumer_session": (
            "safa-pane-fault-consumer-" + attempt_id
        ),
        "consumer_owner_nonce": hashlib.sha256(
            b"consumer-owner"
        ).hexdigest(),
        "consumer_tmux_arguments": [
            "tmux",
            "new-session",
            *consumer_supervisor_arguments,
        ],
        "tmux_arguments": ["tmux", *supervisor_arguments],
        "shell": False,
        "pane_log": {},
        "fault_channel": {},
        "pane_gate_fault_channel": {},
        "pane_gate_fault_publisher": {},
        "pane_fault_consumer": {
            "schema_version": 1,
            "contract_type": (
                "safa_pane_fault_consumer_receipt_registration_v1"
            ),
            "namespace": namespace,
            "artifacts": artifacts,
            "publishers": {
                "launcher": {
                    "path": "/contract/launcher.py",
                    "sha256": launcher_sha,
                    "role": (
                        "launcher_pane_fault_consumer_handoff"
                    ),
                },
                "consumer": {
                    "path": "/contract/launcher.py",
                    "sha256": launcher_sha,
                    "role": "pane_fault_consumer",
                },
            },
        },
        "wrapper_claim_path": "/contract/claim.json",
        "wrapper_started_path": "/contract/wrapper-started.json",
        "gate_execution_terminal_path": "/contract/gate-terminal.json",
        "started_at": "2026-07-28T00:00:00+00:00",
    }
    receipt["launch_receipt_sha256"] = preflight_launch_digest(
        receipt, "launch_receipt_sha256"
    )
    return receipt


def test_launch_receipt_v4_exact_schema_accepted_by_consumers() -> None:
    receipt = _shared_launch_receipt_v4()
    for module in (
        preflight_launch_contract_module,
        _launcher_module(),
        _wrapper_module(),
    ):
        assert (
            module.validate_launch_receipt_schema(
                receipt,
                expected_gate_worker_arguments=receipt[
                    "gate_worker_arguments"
                ],
                expected_consumer_worker_arguments=receipt[
                    "consumer_worker_arguments"
                ],
            )
            == receipt
        )


def _shared_launch_receipt_v5() -> dict[str, Any]:
    receipt_v4 = _shared_launch_receipt_v4()
    receipt_v4["verified_implementations"] = (
        _test_verified_preflight_implementations()
    )
    receipt_v4["launch_receipt_sha256"] = (
        preflight_launch_digest(
            receipt_v4, "launch_receipt_sha256"
        )
    )
    fault_channel = {
        **receipt_v4["gate_lifecycle_wait_channel"],
        "path": (
            "/contract/attempt/"
            "gate_lifecycle_wait_publish_fault.channel"
        ),
        "inode": 29,
    }
    return preflight_launch_contract_module.build_launch_receipt_v5(
        launch_receipt_v4=receipt_v4,
        gate_lifecycle_wait_publish_fault_channel=fault_channel,
        gate_lifecycle_wait_publish_fault_publisher=receipt_v4[
            "gate_lifecycle_wait_publisher"
        ],
        expected_gate_worker_arguments=receipt_v4[
            "gate_worker_arguments"
        ],
        expected_consumer_worker_arguments=receipt_v4[
            "consumer_worker_arguments"
        ],
    )


def test_launch_receipt_v5_roundtrip_is_exact_and_non_mixing() -> None:
    receipt = _shared_launch_receipt_v5()
    assert (
        preflight_launch_contract_module.validate_launch_receipt_v5(
            receipt,
            expected_gate_worker_arguments=receipt[
                "gate_worker_arguments"
            ],
            expected_consumer_worker_arguments=receipt[
                "consumer_worker_arguments"
            ],
        )
        == receipt
    )
    assert receipt["postclaim_finalization_profile"] == (
        preflight_launch_contract_module
        .build_postclaim_finalization_profile_v1()
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_key",
        "missing_key",
        "v4_discriminator",
        "wrong_schema",
        "wrong_type",
        "profile_component_version",
        "profile_component_type",
        "profile_mixed_stack",
        "profile_digest",
        "attempt_id",
        "fault_path",
        "fault_inode",
        "fault_publisher_role",
        "fault_publisher_binding",
        "verified_binding",
        "digest",
    ),
)
def test_launch_receipt_v5_mutations_fail_closed(
    mutation: str,
) -> None:
    receipt = copy.deepcopy(_shared_launch_receipt_v5())
    if mutation == "extra_key":
        receipt["legacy_stack"] = True
    elif mutation == "missing_key":
        receipt.pop("postclaim_finalization_profile")
    elif mutation == "v4_discriminator":
        receipt["schema_version"] = 4
        receipt["contract_type"] = (
            preflight_launch_contract_module
            .LAUNCH_RECEIPT_CONTRACT_TYPE
        )
    elif mutation == "wrong_schema":
        receipt["schema_version"] = 6
    elif mutation == "wrong_type":
        receipt["contract_type"] = "foreign"
    elif mutation == "profile_component_version":
        receipt["postclaim_finalization_profile"]["raw_wait"][
            "schema_version"
        ] = 2
    elif mutation == "profile_component_type":
        receipt["postclaim_finalization_profile"][
            "consumer_terminal"
        ]["contract_type"] = "foreign"
    elif mutation == "profile_mixed_stack":
        receipt["postclaim_finalization_profile"][
            "mixed_stack_allowed"
        ] = True
    elif mutation == "profile_digest":
        receipt["postclaim_finalization_profile"][
            "postclaim_finalization_profile_sha256"
        ] = "0" * 64
    elif mutation == "attempt_id":
        receipt["attempt_id"] = "bad"
    elif mutation == "fault_path":
        receipt[
            "gate_lifecycle_wait_publish_fault_channel"
        ]["path"] = receipt["gate_lifecycle_wait_channel"]["path"]
    elif mutation == "fault_inode":
        receipt[
            "gate_lifecycle_wait_publish_fault_channel"
        ]["inode"] = receipt["gate_lifecycle_wait_channel"]["inode"]
    elif mutation == "fault_publisher_role":
        receipt[
            "gate_lifecycle_wait_publish_fault_publisher"
        ]["role"] = "foreign"
    elif mutation == "fault_publisher_binding":
        receipt[
            "gate_lifecycle_wait_publish_fault_publisher"
        ]["sha256"] = "0" * 64
    elif mutation == "verified_binding":
        receipt["verified_implementations"][
            "preflight_launch_contract"
        ]["sha256"] = "bad"
    elif mutation == "digest":
        receipt["launch_receipt_sha256"] = "0" * 64
    if mutation != "digest":
        receipt["launch_receipt_sha256"] = (
            preflight_launch_digest(
                receipt, "launch_receipt_sha256"
            )
        )
    with pytest.raises(PreflightLaunchContractError):
        preflight_launch_contract_module.validate_launch_receipt_v5(
            receipt,
            expected_gate_worker_arguments=receipt[
                "gate_worker_arguments"
            ],
            expected_consumer_worker_arguments=receipt[
                "consumer_worker_arguments"
            ],
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "v1_impersonation",
        "v2_impersonation",
        "v3_impersonation",
        "missing_registration",
        "extra_registration",
        "missing_artifact",
        "wrong_artifact_path",
        "wrong_publisher_role",
        "wrong_publisher_sha",
        "missing_wait_channel",
        "missing_wait_publisher",
        "missing_supervisor_arguments",
        "missing_supervisor_ready_path",
        "missing_wait_status_path",
        "missing_gate_worker_arguments",
        "wait_channel_identity_drift",
        "publisher_identity_drift",
        "publisher_wrong_argv_index",
        "supervisor_mode_drift",
        "tmux_supervisor_suffix_drift",
        "worker_inherits_wait_channel",
        "worker_argv_binding_drift",
        "coherent_worker_tail_drift",
        "missing_consumer_wait_channel",
        "missing_consumer_wait_publisher",
        "missing_consumer_supervisor_arguments",
        "missing_consumer_supervisor_ready_path",
        "missing_consumer_wait_status_path",
        "missing_consumer_worker_arguments",
        "missing_consumer_session",
        "missing_consumer_owner_nonce",
        "missing_consumer_tmux_arguments",
        "consumer_worker_argv_binding_drift",
        "coherent_consumer_worker_tail_drift",
        "wrong_digest",
    ),
)
def test_launch_receipt_registration_mutations_rejected(
    mutation: str,
) -> None:
    receipt = json.loads(json.dumps(_shared_launch_receipt_v4()))
    expected_gate_worker_arguments = list(
        receipt["gate_worker_arguments"]
    )
    expected_consumer_worker_arguments = list(
        receipt["consumer_worker_arguments"]
    )
    if mutation == "v1_impersonation":
        receipt["schema_version"] = 1
        receipt["contract_type"] = (
            "safa_canonical_preflight_launch_receipt_v1"
        )
    elif mutation == "v2_impersonation":
        receipt["schema_version"] = 2
        receipt["contract_type"] = (
            "safa_canonical_preflight_launch_receipt_v2"
        )
    elif mutation == "v3_impersonation":
        receipt["schema_version"] = 3
        receipt["contract_type"] = (
            "safa_canonical_preflight_launch_receipt_v3"
        )
    elif mutation == "missing_registration":
        receipt.pop("pane_fault_consumer")
    elif mutation == "extra_registration":
        receipt["pane_fault_consumer"]["extra"] = True
    elif mutation == "missing_artifact":
        receipt["pane_fault_consumer"]["artifacts"].pop("active")
    elif mutation == "wrong_artifact_path":
        receipt["pane_fault_consumer"]["artifacts"]["active"] = (
            "/contract/elsewhere/consumer_active.json"
        )
    elif mutation == "wrong_publisher_role":
        receipt["pane_fault_consumer"]["publishers"]["consumer"][
            "role"
        ] = "launcher_pane_fault_consumer_handoff"
    elif mutation == "wrong_publisher_sha":
        receipt["pane_fault_consumer"]["publishers"]["consumer"][
            "sha256"
        ] = hashlib.sha256(b"other").hexdigest()
    elif mutation == "missing_wait_channel":
        receipt.pop("gate_lifecycle_wait_channel")
    elif mutation == "missing_wait_publisher":
        receipt.pop("gate_lifecycle_wait_publisher")
    elif mutation == "missing_supervisor_arguments":
        receipt.pop("gate_lifecycle_wait_supervisor_arguments")
    elif mutation == "missing_supervisor_ready_path":
        receipt.pop("gate_lifecycle_wait_supervisor_ready_path")
    elif mutation == "missing_wait_status_path":
        receipt.pop("gate_lifecycle_wait_status_path")
    elif mutation == "missing_gate_worker_arguments":
        receipt.pop("gate_worker_arguments")
    elif mutation == "wait_channel_identity_drift":
        receipt["gate_lifecycle_wait_channel"]["inode"] = 0
    elif mutation == "publisher_identity_drift":
        receipt["gate_lifecycle_wait_publisher"]["file_identity"][
            "path"
        ] = "/contract/other.py"
    elif mutation == "publisher_wrong_argv_index":
        receipt["gate_lifecycle_wait_supervisor_arguments"][3] = (
            "/contract/not-the-publisher.py"
        )
    elif mutation == "supervisor_mode_drift":
        receipt["gate_lifecycle_wait_supervisor_arguments"][4] = (
            "__pane_gate__"
        )
    elif mutation == "tmux_supervisor_suffix_drift":
        receipt["tmux_arguments"][-1] = "different"
    elif mutation == "worker_inherits_wait_channel":
        receipt["gate_worker_arguments"][6] = receipt[
            "gate_lifecycle_wait_channel"
        ]["path"]
    elif mutation == "worker_argv_binding_drift":
        receipt["gate_worker_arguments"][-1] = '["different"]'
    elif mutation == "coherent_worker_tail_drift":
        receipt["gate_worker_arguments"][-1] = '["different"]'
        encoded_worker = json.dumps(
            receipt["gate_worker_arguments"], separators=(",", ":")
        )
        receipt["gate_lifecycle_wait_supervisor_arguments"][-1] = (
            encoded_worker
        )
        receipt["tmux_arguments"][-1] = encoded_worker
    elif mutation == "missing_consumer_wait_channel":
        receipt.pop("consumer_lifecycle_wait_channel")
    elif mutation == "missing_consumer_wait_publisher":
        receipt.pop("consumer_lifecycle_wait_publisher")
    elif mutation == "missing_consumer_supervisor_arguments":
        receipt.pop("consumer_lifecycle_wait_supervisor_arguments")
    elif mutation == "missing_consumer_supervisor_ready_path":
        receipt.pop("consumer_lifecycle_wait_supervisor_ready_path")
    elif mutation == "missing_consumer_wait_status_path":
        receipt.pop("consumer_lifecycle_wait_status_path")
    elif mutation == "missing_consumer_worker_arguments":
        receipt.pop("consumer_worker_arguments")
    elif mutation == "missing_consumer_session":
        receipt.pop("consumer_session")
    elif mutation == "missing_consumer_owner_nonce":
        receipt.pop("consumer_owner_nonce")
    elif mutation == "missing_consumer_tmux_arguments":
        receipt.pop("consumer_tmux_arguments")
    elif mutation == "consumer_worker_argv_binding_drift":
        receipt["consumer_worker_arguments"][-1] = (
            "/contract/other-config.json"
        )
    elif mutation == "coherent_consumer_worker_tail_drift":
        receipt["consumer_worker_arguments"][-1] = (
            "/contract/other-config.json"
        )
        receipt[
            "consumer_lifecycle_wait_supervisor_arguments"
        ][-1] = json.dumps(
            receipt["consumer_worker_arguments"],
            separators=(",", ":"),
        )
        receipt["consumer_tmux_arguments"][-1] = receipt[
            "consumer_lifecycle_wait_supervisor_arguments"
        ][-1]
    else:
        receipt["started_at"] = "changed"
    if mutation != "wrong_digest":
        receipt["launch_receipt_sha256"] = preflight_launch_digest(
            receipt, "launch_receipt_sha256"
        )
    for module in (
        preflight_launch_contract_module,
        _launcher_module(),
        _wrapper_module(),
    ):
        with pytest.raises(module.PreflightLaunchContractError):
            module.validate_launch_receipt_schema(
                receipt,
                expected_gate_worker_arguments=(
                    expected_gate_worker_arguments
                ),
                expected_consumer_worker_arguments=(
                    expected_consumer_worker_arguments
                ),
            )


def _shared_launch_contract_values() -> dict[str, Any]:
    gate_process = _shared_process(101, ppid=1)
    wrapper_process = _shared_process(202, ppid=101)
    receipt = _shared_binding("receipt")
    receipt_identity = _shared_file_identity("receipt")
    verified_implementations = (
        _test_verified_preflight_implementations()
    )
    pane_fault_consumer_chain = {
        "consumer_started": _shared_binding("consumer-started"),
        "consumer_active": _shared_binding("consumer-active"),
        "consumer_reader_release": _shared_binding(
            "consumer-reader-release"
        ),
        "consumer_release_observed": _shared_binding(
            "consumer-release-observed"
        ),
    }
    gate_ready = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_pane_gate_ready_v1",
        "launch_receipt": receipt,
        "launch_receipt_identity": receipt_identity,
        "verified_implementations": verified_implementations,
        "process": gate_process,
        "wrapper_arguments": ["/usr/bin/python", "wrapper.py"],
        "ready_at": "2026-07-28T00:00:00+00:00",
    }
    gate_ready["pane_gate_ready_sha256"] = preflight_launch_digest(
        gate_ready, "pane_gate_ready_sha256"
    )
    wrapper_started = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_wrapper_started_v1"
        ),
        "launch_receipt": receipt,
        "launch_receipt_identity": receipt_identity,
        "verified_implementations": verified_implementations,
        "pane_gate_ready": _shared_binding("gate-ready"),
        "pane_gate_process": gate_process,
        "wrapper_arguments": ["/usr/bin/python", "wrapper.py"],
        "wrapper_process": wrapper_process,
        "wrapper_executable": _shared_file_identity("python"),
        "started_at": "2026-07-28T00:00:01+00:00",
    }
    wrapper_started["wrapper_started_sha256"] = (
        preflight_launch_digest(
            wrapper_started, "wrapper_started_sha256"
        )
    )
    server_process = _shared_process(303, ppid=1)
    pane = {
        "session": "safa-screening-preflight-controller",
        "pane": "%1",
        "pane_pid": gate_process["pid"],
        "pane_dead": False,
        "pane_dead_status": None,
    }
    tmux_server = {
        "server_pid": server_process["pid"],
        "server_process": server_process,
        "socket_path": "/tmp/tmux.sock",
        "socket_device": 7,
        "socket_inode": 13,
    }
    owner_seal = build_preflight_pane_owner_seal(
        server_pid=server_process["pid"],
        server_start_ticks=server_process["start_ticks"],
        socket_path=tmux_server["socket_path"],
        socket_device=tmux_server["socket_device"],
        socket_inode=tmux_server["socket_inode"],
        session=pane["session"],
        pane=pane["pane"],
        pane_pid=pane["pane_pid"],
        pane_process=gate_process,
        owner_nonce=hashlib.sha256(b"owner").hexdigest(),
        tmux_identity=pane,
        tmux_server=tmux_server,
    )
    tmux_started = build_preflight_tmux_started(
        launch_receipt=receipt,
        launch_receipt_identity=receipt_identity,
        verified_implementations=verified_implementations,
        pane_gate_ready=_shared_binding("gate-ready"),
        tmux_client={"returncode": 0, "stdout": "", "stderr": ""},
        owner_seal=owner_seal,
        started_at="2026-07-28T00:00:00+00:00",
        tmux_identity=pane,
        tmux_server=tmux_server,
    )
    claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_wrapper_claim_v3",
        "attempt_id": hashlib.sha256(b"attempt").hexdigest(),
        "preflight_launch_receipt": receipt,
        "preflight_launch_receipt_identity": receipt_identity,
        "verified_implementations": verified_implementations,
        "pane_gate_ready": _shared_binding("gate-ready"),
        "preflight_launch_tmux_started": _shared_binding("tmux-started"),
        "preflight_wrapper_started": _shared_binding("wrapper-started"),
        "pane_gate_process": gate_process,
        "wrapper_arguments": ["/usr/bin/python", "wrapper.py"],
        "wrapper_executable": _shared_file_identity("python"),
        "pane_log": _shared_file_identity("pane-log"),
        "git": {"sha": hashlib.sha256(b"git").hexdigest()},
        "policy_sha256": hashlib.sha256(b"policy").hexdigest(),
        "config": {
            "path": "/contract/config.json",
            "sha256": hashlib.sha256(b"config").hexdigest(),
        },
        "checkpoint_plan": _shared_binding("plan"),
        "preflight_request_manifest": _shared_binding("requests"),
        "controller_session": "safa-screening-preflight-controller",
        "controller_tmux": pane,
        "controller_tmux_server": tmux_server,
        "observer_session": "safa-screening-preflight-monitor-abc",
        "command": ["/usr/bin/python", "controller.py"],
        "observer_command": ["/usr/bin/python", "controller.py", "monitor"],
        "wrapper_pid": wrapper_process["pid"],
        "wrapper_process": wrapper_process,
        "wrapper_launch_process": wrapper_process,
        "started_at": "2026-07-28T00:00:01+00:00",
        "external_timeout_seconds": None,
        "pane_fault_consumer_chain": pane_fault_consumer_chain,
    }
    claim["wrapper_claim_sha256"] = preflight_launch_digest(
        claim, "wrapper_claim_sha256"
    )
    accepted = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_launch_accepted_v1",
        "attempt_id": claim["attempt_id"],
        "launch_receipt": receipt,
        "launch_receipt_identity": receipt_identity,
        "verified_implementations": verified_implementations,
        "wrapper_claim": _shared_binding("claim"),
        "tmux_started": _shared_binding("tmux-started"),
        "pane": pane,
        "pane_log_path": "/contract/pane-log",
        "startup_window_closed": False,
        "started_at": "2026-07-28T00:00:00+00:00",
        "accepted_at": "2026-07-28T00:00:02+00:00",
        "pane_fault_consumer_chain": pane_fault_consumer_chain,
    }
    accepted["launch_accepted_sha256"] = preflight_launch_digest(
        accepted, "launch_accepted_sha256"
    )
    terminal = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_launch_terminal_v1",
        "launch_receipt": receipt,
        "launch_receipt_identity": receipt_identity,
        "verified_implementations": verified_implementations,
        "launch_accepted": _shared_binding("accepted"),
        "wrapper_claim": _shared_binding("claim"),
        "tmux_started": _shared_binding("tmux-started"),
        "status": "ownership_transferred",
        "failure": None,
        "tmux_client": None,
        "pane": pane,
        "pane_log": _shared_file_identity("pane-log"),
        "session_residual": True,
        "started_at": accepted["started_at"],
        "completed_at": "2026-07-28T00:00:03+00:00",
        "pane_fault_consumer_chain": pane_fault_consumer_chain,
    }
    terminal["launch_terminal_sha256"] = preflight_launch_digest(
        terminal, "launch_terminal_sha256"
    )
    release = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_ownership_release_v1"
        ),
        "launch_receipt": receipt,
        "launch_receipt_identity": receipt_identity,
        "verified_implementations": verified_implementations,
        "launch_accepted": _shared_binding("accepted"),
        "launch_terminal": _shared_binding("terminal"),
        "wrapper_claim": _shared_binding("claim"),
        "startup_window_closed": True,
        "released_at": "2026-07-28T00:00:04+00:00",
        "pane_fault_consumer_chain": pane_fault_consumer_chain,
    }
    release["launch_ownership_release_sha256"] = (
        preflight_launch_digest(
            release, "launch_ownership_release_sha256"
        )
    )
    return {
        "gate_ready": gate_ready,
        "tmux_started": tmux_started,
        "tmux_identity": pane,
        "tmux_server": tmux_server,
        "wrapper_started": wrapper_started,
        "claim": claim,
        "accepted": accepted,
        "terminal": terminal,
        "release": release,
        "receipt": receipt,
        "receipt_identity": receipt_identity,
        "verified_implementations": verified_implementations,
        "wrapper_binding": _shared_binding("claim"),
        "accepted_binding": _shared_binding("accepted"),
        "terminal_binding": _shared_binding("terminal"),
        "pane_fault_consumer_chain": pane_fault_consumer_chain,
    }


def test_preflight_launch_contract_is_pure_and_consumers_share_schemas():
    repo_root = Path(__file__).parents[1]
    contract_path = (
        repo_root / "src/safa/closeout/preflight_launch_contract.py"
    )
    contract_tree = ast.parse(contract_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(contract_tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert not imported_roots.intersection(
        {"io", "os", "pathlib", "signal", "subprocess"}
    )
    consumer_expectations = {
        "scripts/run_canonical_preflight_launcher.py": {
                "build_bound_lifecycle_evidence",
                "build_invalid_claim_evidence",
                "build_terminal_failure",
                "build_tmux_started",
            "validate_claim_v3",
            "validate_gate_ready",
            "validate_ownership_chain",
            "validate_wrapper_started",
        },
        "scripts/run_canonical_preflight_wrapper.py": {
            "build_claim_v3",
            "validate_gate_ready",
            "validate_ownership_chain",
            "validate_pane_owner_seal",
            "validate_tmux_started",
            "validate_wrapper_started",
        },
        "scripts/run_canonical_checkpoint_screening.py": {
            "validate_claim_v3",
            "validate_ownership_chain",
            "validate_pane_owner_seal",
            "validate_tmux_started",
        },
    }
    shared_sets = set(shared_schema_keysets().values())
    shared_sets.update(
        {
            frozenset({"server_pid", "socket_path"}),
            frozenset({"pid", "pgid", "start_ticks"}),
            frozenset({"path", "device", "inode", "mode"}),
            frozenset({"argv0", "sys_executable", "proc_executable"}),
        }
    )
    legacy_field_names = {
        "argv0",
        "sys_executable",
        "proc_executable",
    }
    bootstrap_schema_functions = {
        "scripts/run_canonical_preflight_launcher.py": {
            "build_file_identity",
            "build_process_identity",
            "_bootstrap_read_file",
            "_json_binding",
            "_reverify_verified_loader",
        },
        "scripts/run_canonical_preflight_wrapper.py": {
            "_bootstrap_read_file",
            "_reverify_verified_loader",
        },
        "scripts/run_canonical_checkpoint_screening.py": {
            "_install_verified_preflight_contract_api",
        },
    }
    for relative, expected_calls in consumer_expectations.items():
        consumer_tree = ast.parse(
            (repo_root / relative).read_text(encoding="utf-8")
        )
        calls = {
            node.func.id
            for node in ast.walk(consumer_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        assert expected_calls <= calls
        bootstrap_nodes = {
            id(descendant)
            for function in consumer_tree.body
            if isinstance(function, ast.FunctionDef)
            and function.name
            in bootstrap_schema_functions.get(relative, set())
            for descendant in ast.walk(function)
        }
        string_literals = {
            node.value
            for node in ast.walk(consumer_tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }
        assert not string_literals.intersection(shared_contract_types())
        assert not string_literals.intersection(legacy_field_names)
        for node in ast.walk(consumer_tree):
            if id(node) in bootstrap_nodes:
                continue
            if isinstance(node, ast.Dict):
                elements = node.keys
            elif isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                elements = node.elts
            else:
                continue
            literal = frozenset(
                item.value
                for item in elements
                if isinstance(item, ast.Constant)
                and isinstance(item.value, str)
            )
            if len(literal) == len(elements):
                assert literal not in shared_sets
    helper_tree = ast.parse(
        (
            repo_root / "tests/helpers/preflight_lifecycle_helper.py"
        ).read_text(encoding="utf-8")
    )
    production_builder = next(
        node
        for node in helper_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_supervised_launcher_contract"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch_preflight"
        for node in ast.walk(production_builder)
    )
    assert not any(
        isinstance(node, ast.Constant)
        and node.value in {
            2,
            "pane_gate_arguments",
            "safa_canonical_preflight_launch_tmux_started_v1",
        }
        for node in ast.walk(production_builder)
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "schema_v2",
        "missing_ppid",
        "missing_sid",
        "reduced_only",
        "full_mismatch",
        "parent_spoof",
        "identity_extra",
        "chain_swap",
        "chain_digest",
        "chain_reference",
    ),
)
def test_shared_claim_mutations_rejected_by_all_consumers(mutation: str):
    modules = (
        _launcher_module(),
        _wrapper_module(),
        _controller_module(),
    )
    values = _shared_launch_contract_values()
    claim = json.loads(json.dumps(values["claim"]))
    if mutation == "missing":
        claim.pop("policy_sha256")
    elif mutation == "extra":
        claim["unshared_schema_field"] = True
    elif mutation == "schema_v2":
        claim["schema_version"] = 2
    elif mutation == "missing_ppid":
        claim["wrapper_process"].pop("ppid")
    elif mutation == "missing_sid":
        claim["wrapper_launch_process"].pop("sid")
    elif mutation == "reduced_only":
        claim["wrapper_process"] = {
            key: claim["wrapper_process"][key]
            for key in ("pid", "pgid", "start_ticks")
        }
    elif mutation == "full_mismatch":
        claim["wrapper_process"]["start_ticks"] += 1
    elif mutation == "parent_spoof":
        claim["wrapper_launch_process"]["ppid"] = 1
    elif mutation == "identity_extra":
        claim["wrapper_process"]["extra"] = True
    elif mutation == "chain_swap":
        chain = claim["pane_fault_consumer_chain"]
        chain["consumer_started"], chain["consumer_active"] = (
            chain["consumer_active"],
            chain["consumer_started"],
        )
    elif mutation == "chain_digest":
        claim["pane_fault_consumer_chain"][
            "consumer_active"
        ]["canonical_sha256"] = hashlib.sha256(
            b"wrong-active"
        ).hexdigest()
    else:
        claim["pane_fault_consumer_chain"][
            "consumer_release_observed"
        ]["path"] = "/contract/unregistered-observed.json"
    if mutation not in {"missing", "extra"}:
        claim["wrapper_claim_sha256"] = preflight_launch_digest(
            claim, "wrapper_claim_sha256"
        )
    for module in modules:
        expected_error = getattr(
            module,
            "PreflightLaunchContractError",
            PreflightLaunchContractError,
        )
        with pytest.raises(expected_error):
            module.validate_claim_v3(
                claim,
                verified_implementations=values[
                    "verified_implementations"
                ],
                    gate_ready=values["gate_ready"],
                wrapper_started=values["wrapper_started"],
                pane_fault_consumer_chain=values[
                    "pane_fault_consumer_chain"
                ],
            )


def test_shared_full_process_claim_accepted_by_all_consumers():
    values = _shared_launch_contract_values()
    for module in (
        _launcher_module(),
        _wrapper_module(),
        _controller_module(),
    ):
        assert (
            module.validate_claim_v3(
                values["claim"],
                verified_implementations=values[
                    "verified_implementations"
                ],
                gate_ready=values["gate_ready"],
                wrapper_started=values["wrapper_started"],
                pane_fault_consumer_chain=values[
                    "pane_fault_consumer_chain"
                ],
            )
            == values["claim"]
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "wrong_chain", "wrong_owner"),
)
def test_shared_tmux_started_mutations_are_rejected(mutation: str):
    values = _shared_launch_contract_values()
    tmux_started = json.loads(json.dumps(values["tmux_started"]))
    if mutation == "missing":
        tmux_started.pop("pane_gate_ready")
    elif mutation == "extra":
        tmux_started["unshared_schema_field"] = True
    elif mutation == "wrong_chain":
        tmux_started["verified_implementations"][
            "verified_loader"
        ]["sha256"] = hashlib.sha256(b"wrong-loader").hexdigest()
    else:
        tmux_started["owner_seal"]["pane_pid"] += 1
    if mutation not in {"missing", "extra"}:
        tmux_started["launch_tmux_started_sha256"] = (
            preflight_launch_digest(
                tmux_started, "launch_tmux_started_sha256"
            )
        )
    with pytest.raises(PreflightLaunchContractError):
        preflight_launch_contract_module.validate_tmux_started(
            tmux_started,
            verified_implementations=values[
                "verified_implementations"
            ],
            tmux_identity=values["tmux_identity"],
            tmux_server=values["tmux_server"],
        )


def _shared_preclaim_failure_intent(
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = _shared_launch_contract_values()
    deadline_reached = reason == "claim_timeout"
    wrapper_claim_path = "/contract/wrapper-claim.json"
    intent = preflight_launch_contract_module.build_preclaim_failure_intent(
        attempt_id=values["claim"]["attempt_id"],
        launch_receipt=values["receipt"],
        launch_receipt_identity=values["receipt_identity"],
        verified_implementations=values[
            "verified_implementations"
        ],
        wrapper_claim_path=wrapper_claim_path,
        pane_fault_consumer_chain=values[
            "pane_fault_consumer_chain"
        ],
        controller_owner_seal=values["tmux_started"]["owner_seal"],
        reason=reason,
        stage=(
            "wrapper_claim_validation"
            if reason == "invalid_claim"
            else "wrapper_claim_wait_deadline"
        ),
        deadline_observation={
            "started_monotonic_ns": 1_000,
            "deadline_monotonic_ns": 2_000,
            "observed_monotonic_ns": (
                1_500 if not deadline_reached else 2_000
            ),
            "deadline_reached": deadline_reached,
        },
        invalid_claim_evidence=(
            preflight_launch_contract_module.build_invalid_claim_evidence(
                raw_content_sha256=hashlib.sha256(
                    b"invalid raw claim"
                ).hexdigest(),
                file_identity=_shared_file_identity(
                    "invalid-claim"
                )
                | {"path": wrapper_claim_path},
            )
            if reason == "invalid_claim"
            else None
        ),
        observed_at="2026-07-28T00:00:05+00:00",
        tmux_identity=values["tmux_identity"],
        tmux_server=values["tmux_server"],
    )
    return intent, values


@pytest.mark.parametrize(
    "mutation", ("none", "missing", "extra", "sha", "identity")
)
def test_invalid_claim_evidence_builder_roundtrip_and_mutations(
    mutation: str,
) -> None:
    raw_content_sha256 = hashlib.sha256(
        b"invalid raw claim"
    ).hexdigest()
    file_identity = _shared_file_identity("invalid-claim")
    expected = {
        "raw_content_sha256": raw_content_sha256,
        "file_identity": file_identity,
    }
    built = (
        preflight_launch_contract_module.build_invalid_claim_evidence(
            raw_content_sha256=raw_content_sha256,
            file_identity=file_identity,
        )
    )
    assert built == expected
    mutated = copy.deepcopy(built)
    if mutation == "missing":
        mutated.pop("raw_content_sha256")
    elif mutation == "extra":
        mutated["extra"] = True
    elif mutation == "sha":
        mutated["raw_content_sha256"] = "not-a-sha"
    elif mutation == "identity":
        mutated["file_identity"]["size"] = -1
    if mutation == "none":
        assert (
            preflight_launch_contract_module.validate_invalid_claim_evidence(
                mutated
            )
            == expected
        )
    else:
        with pytest.raises(PreflightLaunchContractError):
            preflight_launch_contract_module.validate_invalid_claim_evidence(
                mutated
            )


@pytest.mark.parametrize("reason", ("invalid_claim", "claim_timeout"))
def test_preclaim_failure_intent_exact_schema_accepted(
    reason: str,
) -> None:
    intent, values = _shared_preclaim_failure_intent(reason)
    for module in (
        preflight_launch_contract_module,
        _launcher_module(),
    ):
        assert (
            module.validate_preclaim_failure_intent(
                intent,
                verified_implementations=values[
                    "verified_implementations"
                ],
                expected_wrapper_claim_path=(
                    "/contract/wrapper-claim.json"
                ),
                tmux_identity=values["tmux_identity"],
                tmux_server=values["tmux_server"],
                expected_receipt=values["receipt"],
                expected_receipt_identity=values[
                    "receipt_identity"
                ],
                expected_consumer_chain=values[
                    "pane_fault_consumer_chain"
                ],
            )
            == intent
        )


def _preclaim_failure_publisher_fixture(
    launcher: Any,
    tmp_path: Path,
    *,
    reason: str,
) -> tuple[Path, dict[str, Any]]:
    values = _shared_launch_contract_values()
    wrapper_claim_path = tmp_path / "wrapper_claim.json"
    if reason == "invalid_claim":
        wrapper_claim_path.write_bytes(b'{"invalid":"claim"}\n')
        wrapper_claim_path.chmod(0o644)
    deadline_reached = reason == "claim_timeout"
    intent_path = tmp_path / "preclaim_failure_intent.json"
    return intent_path, {
        "attempt_id": values["claim"]["attempt_id"],
        "launch_receipt": values["receipt"],
        "launch_receipt_identity": values["receipt_identity"],
        "verified_implementations": values[
            "verified_implementations"
        ],
        "wrapper_claim_path": wrapper_claim_path,
        "pane_fault_consumer_chain": values[
            "pane_fault_consumer_chain"
        ],
        "tmux_owner_seal": values["tmux_started"]["owner_seal"],
        "reason": reason,
        "deadline_observation": {
            "started_monotonic_ns": 1_000,
            "deadline_monotonic_ns": 2_000,
            "observed_monotonic_ns": (
                2_000 if deadline_reached else 1_500
            ),
            "deadline_reached": deadline_reached,
        },
        "observed_at": "2026-07-28T00:00:05+00:00",
        "tmux_identity": values["tmux_identity"],
        "tmux_server": values["tmux_server"],
    }


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux descriptor publication semantics",
)
@pytest.mark.parametrize("reason", ("invalid_claim", "claim_timeout"))
def test_preclaim_failure_intent_publisher_resumes_exact_intent(
    tmp_path: Path,
    reason: str,
) -> None:
    launcher = _launcher_module()
    path, arguments = _preclaim_failure_publisher_fixture(
        launcher, tmp_path, reason=reason
    )
    first = launcher._publish_or_resume_preclaim_failure_intent(
        path, **arguments
    )
    first_bytes = path.read_bytes()
    first_stat = path.stat()
    second = launcher._publish_or_resume_preclaim_failure_intent(
        path, **arguments
    )
    assert second == first
    assert path.read_bytes() == first_bytes
    assert (path.stat().st_dev, path.stat().st_ino) == (
        first_stat.st_dev,
        first_stat.st_ino,
    )
    assert second["intent"]["controller_owner_seal"] == arguments[
        "tmux_owner_seal"
    ]
    assert second["artifact"] == {
        "path": str(path),
        "sha256": hashlib.sha256(first_bytes).hexdigest(),
        "canonical_sha256": second["intent"][
            "preclaim_failure_intent_sha256"
        ],
    }


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux descriptor publication semantics",
)
@pytest.mark.parametrize("cutpoint", ("before_intent", "after_intent"))
def test_preclaim_failure_intent_publisher_crash_cutpoints_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cutpoint: str,
) -> None:
    launcher = _launcher_module()
    path, arguments = _preclaim_failure_publisher_fixture(
        launcher, tmp_path, reason="claim_timeout"
    )
    original_write = launcher._write_exclusive

    class SimulatedCrash(BaseException):
        pass

    def crash_at_intent(
        target: Path, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        if cutpoint == "before_intent":
            raise SimulatedCrash()
        original_write(target, value)
        raise SimulatedCrash()

    monkeypatch.setattr(launcher, "_write_exclusive", crash_at_intent)
    with pytest.raises(SimulatedCrash):
        launcher._publish_or_resume_preclaim_failure_intent(
            path, **arguments
        )
    assert path.exists() is (cutpoint == "after_intent")
    published_bytes = path.read_bytes() if path.exists() else None
    monkeypatch.setattr(launcher, "_write_exclusive", original_write)
    resumed = launcher._publish_or_resume_preclaim_failure_intent(
        path, **arguments
    )
    assert path.is_file()
    if published_bytes is not None:
        assert path.read_bytes() == published_bytes
    assert resumed["artifact"]["sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux descriptor publication semantics",
)
@pytest.mark.parametrize(
    "collision", ("partial", "foreign", "hardlink_residual")
)
def test_preclaim_failure_intent_publisher_collision_fails_closed(
    tmp_path: Path,
    collision: str,
) -> None:
    launcher = _launcher_module()
    path, arguments = _preclaim_failure_publisher_fixture(
        launcher, tmp_path, reason="claim_timeout"
    )
    if collision == "partial":
        path.write_bytes(b'{"schema_version":')
        path.chmod(0o644)
    elif collision == "foreign":
        launcher._publish_or_resume_preclaim_failure_intent(
            path, **arguments
        )
        arguments["observed_at"] = "2026-07-28T00:00:06+00:00"
    else:
        launcher._publish_or_resume_preclaim_failure_intent(
            path, **arguments
        )
        residual = tmp_path / ".preclaim_failure_intent.residual"
        os.link(path, residual)
        assert path.stat().st_nlink == 2
    before = path.read_bytes()
    before_stat = path.stat()
    if collision == "hardlink_residual":
        with pytest.raises(
            RuntimeError, match="named identity changed"
        ) as raised:
            launcher._publish_or_resume_preclaim_failure_intent(
                path, **arguments
            )
        assert not isinstance(
            raised.value, launcher.LauncherExclusivePublishError
        )
        assert residual.read_bytes() == before
    else:
        with pytest.raises(
            launcher.LauncherExclusivePublishError
        ) as raised:
            launcher._publish_or_resume_preclaim_failure_intent(
                path, **arguments
            )
        assert raised.value.commit_state == "collision"
    assert path.read_bytes() == before
    assert (path.stat().st_dev, path.stat().st_ino) == (
        before_stat.st_dev,
        before_stat.st_ino,
    )


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux descriptor publication semantics",
)
def test_preclaim_failure_intent_publisher_fsync_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    path, arguments = _preclaim_failure_publisher_fixture(
        launcher, tmp_path, reason="claim_timeout"
    )
    original_fsync = launcher.os.fsync
    counts = {"file": 0, "directory": 0}

    def counting_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        key = "directory" if stat.S_ISDIR(mode) else "file"
        counts[key] += 1
        original_fsync(descriptor)

    monkeypatch.setattr(launcher.os, "fsync", counting_fsync)
    launcher._publish_or_resume_preclaim_failure_intent(
        path, **arguments
    )
    assert counts == {"file": 2, "directory": 3}
    counts.update(file=0, directory=0)
    launcher._publish_or_resume_preclaim_failure_intent(
        path, **arguments
    )
    assert counts == {"file": 3, "directory": 3}


def test_preclaim_failure_intent_publisher_ast_boundaries() -> None:
    launcher = _launcher_module()
    invalid_reader = ast.parse(
        inspect.getsource(
            launcher._sealed_invalid_wrapper_claim_evidence
        )
    )
    calls = [
        (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
        )
        for node in ast.walk(invalid_reader)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    ]
    for forbidden in (
        "_load_json",
        "_json_binding",
        "_opened_file_identity",
        "open",
        "read_bytes",
        "read_text",
    ):
        assert forbidden not in calls
    assert calls.count("_launcher_read_publication_file") == 1

    source = Path(launcher.__file__).read_text(encoding="utf-8")
    module_tree = ast.parse(source)
    production_calls = [
        node
        for node in ast.walk(module_tree)
        if isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id
                == "_publish_or_resume_preclaim_failure_intent"
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                == "_publish_or_resume_preclaim_failure_intent"
            )
        )
    ]
    function_names = {
        node.name
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert production_calls == []
    assert "_start_preclaim_wait_deadline" not in function_names
    assert "_observe_preclaim_wait_state" not in function_names




def _production_preclaim_finalization_evidence_fixture(
    tmp_path: Path,
    state: str,
    *,
    hide_controller_after_gate_terminal: bool = False,
) -> tuple[Any, dict[str, Any]]:
    launcher, repo_root, config, _fake_wrapper, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    fixture_provenance = hashlib.sha256(
        f"{tmp_path.resolve()}:{state}".encode()
    ).hexdigest()
    attempt_id = hashlib.sha256(
        f"p3a:{fixture_provenance}".encode()
    ).hexdigest()
    campaign_root = tmp_path / "campaign"
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts/by_policy"
        / policy_sha256
        / attempt_id
    )
    gate_terminal_probe = attempt_root / "gate_execution_terminal.json"
    observer_suffix = hashlib.sha256(
        f"p3a-observer:{fixture_provenance}".encode()
    ).hexdigest()
    observer_session = launcher.OBSERVER_SESSION_PREFIX + observer_suffix
    original_tmux_pane = launcher._tmux_pane
    hidden_controller_observations = 0

    def observed_tmux_pane(session: str) -> Any:
        nonlocal hidden_controller_observations
        if (
            hide_controller_after_gate_terminal
            and session == launcher.CONTROLLER_SESSION
            and gate_terminal_probe.is_file()
        ):
            hidden_controller_observations += 1
            return None
        return original_tmux_pane(session)

    launcher._tmux_pane = observed_tmux_pane
    lock_descriptor = os.open(
        "/tmp/safa_preclaim_finalization_real_tmux.lock",
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC,
        0o600,
    )
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        for session in (
            launcher.CONTROLLER_SESSION,
            observer_session,
        ):
            assert launcher._tmux_pane(session) is None
        try:
            result = launcher.launch_preflight(
                repo_root=repo_root,
                config=config,
                campaign_root=campaign_root,
                policy_sha256=policy_sha256,
                python=sys.executable,
                startup_timeout_seconds=10,
                attempt_id=attempt_id,
                owner_nonce=hashlib.sha256(
                    f"p3a-owner:{fixture_provenance}".encode()
                ).hexdigest(),
                observer_suffix=observer_suffix,
                wrapper_arguments_override=[
                    sys.executable,
                    "-B",
                    "-u",
                    "-c",
                    "raise SystemExit(7)",
                ],
            )
        finally:
            launcher._tmux_pane = original_tmux_pane
        receipt_probe = load_json(
            attempt_root / "launch_receipt.json",
            "P3a locked launch receipt",
        )
        consumer_attempt_probe = load_json(
            Path(
                str(
                    receipt_probe["pane_fault_consumer"][
                        "artifacts"
                    ]["attempt"]
                )
            ),
            "P3a locked consumer attempt",
        )
        for session in (
            launcher.CONTROLLER_SESSION,
            observer_session,
            consumer_attempt_probe["consumer_session"],
        ):
            assert launcher._tmux_pane(session) is None
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)
    assert result["status"] == "wrapper_exited_before_claim"
    if hide_controller_after_gate_terminal:
        assert hidden_controller_observations == 1
    receipt_path = attempt_root / "launch_receipt.json"
    receipt = load_json(receipt_path, "P3a launch receipt")
    assert isinstance(receipt["started_at"], str)
    receipt_started_at = datetime.fromisoformat(
        receipt["started_at"]
    )
    assert receipt["started_at"].endswith("+00:00")
    assert receipt_started_at.tzinfo is not None
    assert receipt_started_at.utcoffset() == timezone.utc.utcoffset(
        None
    )
    assert receipt_started_at.isoformat() == receipt["started_at"]
    receipt_identity = launcher._opened_file_identity(receipt_path)
    artifacts = receipt["pane_fault_consumer"]["artifacts"]
    consumer_attempt_path = Path(str(artifacts["attempt"]))
    consumer_attempt = load_json(
        consumer_attempt_path, "P3a consumer attempt"
    )
    controller_cleanup_path = Path(
        str(artifacts["controller_cleanup"])
    )
    controller_cleanup = load_json(
        controller_cleanup_path, "P3a controller cleanup"
    )
    consumer_terminal_path = Path(str(artifacts["terminal"]))
    consumer_terminal = load_json(
        consumer_terminal_path, "P3a consumer terminal"
    )
    join_path = Path(str(artifacts["join"]))
    join = load_json(join_path, "P3a consumer join")
    cleanup_path = Path(str(artifacts["cleanup"]))
    consumer_chain = {
        key: consumer_terminal[key]
        for key in (
            "consumer_started",
            "consumer_active",
            "consumer_reader_release",
            "consumer_release_observed",
        )
    }
    runtime_gate_owner = consumer_attempt["gate_owner_seal"]
    runtime_tmux_server = runtime_gate_owner["tmux_server"]
    runtime_tmux_identity = {
        "session": runtime_gate_owner["session"],
        "pane": runtime_gate_owner["pane"],
        "pane_pid": runtime_gate_owner["pane_pid"],
    }
    formal_gate_owner = (
        preflight_launch_contract_module.build_pane_owner_seal(
            server_pid=runtime_tmux_server["server_pid"],
            server_start_ticks=runtime_tmux_server[
                "server_process"
            ]["start_ticks"],
            socket_path=runtime_tmux_server["socket_path"],
            socket_device=runtime_tmux_server["socket_device"],
            socket_inode=runtime_tmux_server["socket_inode"],
            session=runtime_gate_owner["session"],
            pane=runtime_gate_owner["pane"],
            pane_pid=runtime_gate_owner["pane_pid"],
            pane_process=runtime_gate_owner["pane_process"],
            owner_nonce=runtime_gate_owner["owner_nonce"],
            tmux_identity=runtime_tmux_identity,
            tmux_server=runtime_tmux_server,
        )
    )
    assert set(formal_gate_owner) == {
        "server_pid",
        "server_start_ticks",
        "socket_path",
        "socket_device",
        "socket_inode",
        "session",
        "pane",
        "pane_pid",
        "pane_process",
        "owner_nonce",
    }
    assert formal_gate_owner["session"] == runtime_gate_owner[
        "session"
    ]
    assert formal_gate_owner["pane"] == runtime_gate_owner["pane"]
    assert formal_gate_owner["pane_pid"] == runtime_gate_owner[
        "pane_pid"
    ]
    assert formal_gate_owner["pane_process"] == runtime_gate_owner[
        "pane_process"
    ]
    assert formal_gate_owner["owner_nonce"] == runtime_gate_owner[
        "owner_nonce"
    ]
    intent_path = attempt_root / "preclaim_failure_intent.json"
    intent_publication = (
        launcher._publish_or_resume_preclaim_failure_intent(
            intent_path,
            attempt_id=attempt_id,
            launch_receipt=launcher._json_binding(
                receipt_path, "launch_receipt_sha256"
            ),
            launch_receipt_identity=receipt_identity,
            verified_implementations=receipt[
                "verified_implementations"
            ],
            wrapper_claim_path=Path(
                str(receipt["wrapper_claim_path"])
            ),
            pane_fault_consumer_chain=consumer_chain,
            tmux_owner_seal=formal_gate_owner,
            reason="claim_timeout",
            deadline_observation={
                "started_monotonic_ns": 1_000,
                "deadline_monotonic_ns": 2_000,
                "observed_monotonic_ns": 2_000,
                "deadline_reached": True,
            },
            observed_at="2026-07-28T00:00:05+00:00",
            tmux_identity=runtime_tmux_identity,
            tmux_server=runtime_tmux_server,
        )
    )
    intent = intent_publication["intent"]
    dead_gate_owner = controller_cleanup["dead_owner_seal"]
    live_consumer_owner = consumer_terminal["supervisor_owner_seal"]
    dead_consumer_owner = {
        **live_consumer_owner,
        "pane_dead": True,
        "pane_dead_status": join["retired_pane"][
            "pane_dead_status"
        ],
        "pane_process": None,
    }
    v2_path = attempt_root / "launch_terminal_v2.json"
    formal_gate = launcher._read_formal_gate_lifecycle_status(
        attempt_root=attempt_root,
        pane=dead_gate_owner,
    )
    formal_consumer = launcher._read_formal_consumer_lifecycle_status(
        attempt_path=consumer_attempt_path,
        pane=dead_consumer_owner,
    )

    def bound(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "artifact": {
                "path": snapshot["channel_authority"]["path"],
                "sha256": snapshot["sha256"],
                "canonical_sha256": snapshot["record"][
                    "lifecycle_wait_status_sha256"
                ],
            },
            "record": snapshot["record"],
        }

    terminal_v2 = (
        preflight_launch_contract_module.build_launch_terminal_v2(
            attempt_id=attempt_id,
            preclaim_failure_intent=intent,
            preclaim_failure_intent_binding=intent_publication[
                "artifact"
            ],
            launch_receipt=intent["launch_receipt"],
            launch_receipt_identity=receipt_identity,
            verified_implementations=intent[
                "verified_implementations"
            ],
            pane_fault_consumer_chain=consumer_chain,
            gate_execution_terminal=launcher._json_binding(
                attempt_root / "gate_execution_terminal.json",
                "gate_execution_terminal_sha256",
            ),
            gate_lifecycle=bound(formal_gate["snapshot"]),
            controller_cleanup=launcher._json_binding(
                controller_cleanup_path,
                "consumer_controller_cleanup_sha256",
            ),
            consumer_terminal=launcher._json_binding(
                consumer_terminal_path, "consumer_terminal_sha256"
            ),
            consumer_lifecycle=bound(formal_consumer["snapshot"]),
            consumer_join=launcher._json_binding(
                join_path, "consumer_join_sha256"
            ),
            consumer_cleanup=launcher._json_binding(
                cleanup_path, "consumer_cleanup_sha256"
            ),
            status="wrapper_claim_timeout",
            failure={
                "reason": "claim_timeout",
                "stage": "wrapper_claim_wait_deadline",
                "type": "WrapperClaimTimeout",
                "message": "wrapper claim deadline reached",
            },
            started_at=receipt["started_at"],
            completed_at=receipt["started_at"],
        )
    )
    assert (
        datetime.fromisoformat(terminal_v2["completed_at"])
        >= receipt_started_at
    )
    launcher._write_exclusive(v2_path, terminal_v2)
    sealed_terminal_v2, _terminal_v2_binding, _terminal_v2_identity = (
        launcher._sealed_finalization_json(
            v2_path,
            digest_field="launch_terminal_sha256",
            label="P3a launch terminal v2",
        )
    )
    assert sealed_terminal_v2 == terminal_v2
    all_paths = {
        "gate": attempt_root / "gate_execution_terminal.json",
        "controller": controller_cleanup_path,
        "terminal": consumer_terminal_path,
        "join": join_path,
        "cleanup": cleanup_path,
        "launch": v2_path,
    }
    keep = {
        "intent_only": set(),
        "gate_evidence": {"gate"},
        "controller_cleanup_present": {"gate", "controller"},
        "consumer_terminal_chain": {
            "gate",
            "controller",
            "terminal",
        },
        "consumer_join_present": {
            "gate",
            "controller",
            "terminal",
            "join",
        },
        "consumer_cleanup_present": {
            "gate",
            "controller",
            "terminal",
            "join",
            "cleanup",
        },
        "launch_terminal": set(all_paths),
        "terminal_without_cleanup": {"gate", "terminal"},
        "join_without_terminal": {"gate", "controller", "join"},
        "cleanup_without_join": {
            "gate",
            "controller",
            "terminal",
            "cleanup",
        },
        "launch_terminal_without_cleanup": {"gate", "launch"},
        "foreign_owner": set(),
    }[state]
    for name, path in all_paths.items():
        if name not in keep:
            path.rename(path.with_name(path.name + ".hidden"))
    arguments = {
        "attempt_root": attempt_root,
        "intent_publication": intent_publication,
        "launch_receipt": receipt,
        "launch_receipt_identity": receipt_identity,
        "expected_gate_owner_seal": dead_gate_owner,
        "expected_consumer_owner_seal": dead_consumer_owner,
        "launch_terminal_path": v2_path,
    }
    if state == "foreign_owner":
        arguments["expected_consumer_owner_seal"] = {
            **dead_consumer_owner,
            "owner_nonce": "f" * 64,
        }
    return launcher, arguments


@pytest.mark.parametrize(
    ("fixture_state", "expected_state"),
    (
        ("intent_only", "INTENT_ONLY"),
        ("gate_evidence", "GATE_EVIDENCE"),
        (
            "controller_cleanup_present",
            "CONTROLLER_CLEANUP_PRESENT",
        ),
        ("consumer_terminal_chain", "CONSUMER_TERMINAL_CHAIN"),
        ("consumer_join_present", "CONSUMER_JOIN_PRESENT"),
        ("consumer_cleanup_present", "CONSUMER_CLEANUP_PRESENT"),
        ("launch_terminal", "LAUNCH_TERMINAL"),
    ),
)
def test_preclaim_finalization_evidence_reader_states(
    tmp_path: Path,
    fixture_state: str,
    expected_state: str,
) -> None:
    launcher, arguments = (
        _production_preclaim_finalization_evidence_fixture(
            tmp_path, fixture_state
        )
    )
    assert (
        launcher._load_validate_preclaim_finalization_evidence(
            **arguments
        )
        is getattr(
            launcher.PreclaimFinalizationEvidenceState,
            expected_state,
        )
    )


def test_wrapper_early_exit_continues_from_cleanup_after_pane_disappears(
    tmp_path: Path,
) -> None:
    launcher, arguments = (
        _production_preclaim_finalization_evidence_fixture(
            tmp_path,
            "intent_only",
            hide_controller_after_gate_terminal=True,
        )
    )
    assert (
        launcher._load_validate_preclaim_finalization_evidence(
            **arguments
        )
        is launcher.PreclaimFinalizationEvidenceState.INTENT_ONLY
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "attempt",
        "owner_nonce",
        "session",
        "tmux_server",
        "session_residual",
    ),
)
def test_wrapper_early_exit_resigned_cleanup_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    launcher, arguments = (
        _production_preclaim_finalization_evidence_fixture(
            tmp_path, "launch_terminal"
        )
    )
    attempt_root = arguments["attempt_root"]
    receipt_path = attempt_root / "launch_receipt.json"
    receipt = arguments["launch_receipt"]
    artifacts = receipt["pane_fault_consumer"]["artifacts"]
    consumer_attempt = load_json(
        Path(str(artifacts["attempt"])),
        "early-exit mutation consumer attempt",
    )
    consumer_terminal = load_json(
        Path(str(artifacts["terminal"])),
        "early-exit mutation consumer terminal",
    )
    cleanup_path = Path(str(artifacts["controller_cleanup"]))
    cleanup = load_json(
        cleanup_path, "early-exit mutation controller cleanup"
    )
    if mutation == "attempt":
        cleanup["attempt_id"] = "f" * 64
    elif mutation == "session_residual":
        cleanup["session_residual"] = True
    else:
        dead_owner = cleanup["dead_owner_seal"]
        if mutation == "owner_nonce":
            dead_owner["owner_nonce"] = "f" * 64
        elif mutation == "session":
            dead_owner["session"] = "foreign-session"
        else:
            dead_owner["tmux_server"]["socket_inode"] += 1
    cleanup["consumer_controller_cleanup_sha256"] = canonical_digest(
        cleanup, "consumer_controller_cleanup_sha256"
    )
    cleanup_path.write_text(
        json.dumps(cleanup, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    expected_consumer_chain = {
        key: consumer_terminal[key]
        for key in (
            "consumer_started",
            "consumer_active",
            "consumer_reader_release",
            "consumer_release_observed",
        )
    }
    with pytest.raises(RuntimeError):
        launcher._continue_wrapper_early_exit_from_durable_cleanup(
            attempt_root=attempt_root,
            launch_receipt=receipt,
            launch_receipt_path=receipt_path,
            launch_receipt_identity=launcher._opened_file_identity(
                receipt_path
            ),
            gate_ready_path=attempt_root / "pane_gate_ready.json",
            live_gate_owner_seal=consumer_attempt["gate_owner_seal"],
            expected_consumer_chain=expected_consumer_chain,
            pane_fault_consumer={},
            config=Path(__file__).parents[1]
            / "configs/closeout/canonical_screening_512_v1.json",
            deadline=time.monotonic() + 1.0,
            startup_timeout_seconds=1.0,
        )


def test_wrapper_early_exit_missing_cleanup_fails_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    artifacts = {
        "attempt": str(tmp_path / "consumer_attempt.json"),
        "controller_cleanup": str(
            tmp_path / "consumer_controller_cleanup.json"
        ),
    }
    monkeypatch.setattr(
        launcher,
        "_require_post_handoff_pane_fault_consumer",
        lambda *_args: None,
    )
    monkeypatch.setattr(launcher.time, "monotonic", lambda: 2.0)
    with pytest.raises(
        launcher.PaneFaultConsumerReservationError,
        match="preclaim formal consumer timed out",
    ):
        launcher._continue_wrapper_early_exit_from_durable_cleanup(
            attempt_root=tmp_path,
            launch_receipt={
                "pane_fault_consumer": {"artifacts": artifacts}
            },
            launch_receipt_path=tmp_path / "launch_receipt.json",
            launch_receipt_identity={},
            gate_ready_path=tmp_path / "pane_gate_ready.json",
            live_gate_owner_seal={},
            expected_consumer_chain={},
            pane_fault_consumer={},
            config=tmp_path / "config.json",
            deadline=1.0,
            startup_timeout_seconds=1.0,
        )


@pytest.mark.parametrize(
    "partial",
    (
        "terminal_without_cleanup",
        "join_without_terminal",
        "cleanup_without_join",
        "launch_terminal_without_cleanup",
    ),
)
def test_preclaim_finalization_evidence_partial_chain_fails_closed(
    tmp_path: Path,
    partial: str,
) -> None:
    launcher, arguments = (
        _production_preclaim_finalization_evidence_fixture(
            tmp_path, partial
        )
    )
    with pytest.raises(RuntimeError):
        launcher._load_validate_preclaim_finalization_evidence(
            **arguments
        )


def test_preclaim_finalization_evidence_foreign_owner_fails_closed(
    tmp_path: Path,
) -> None:
    launcher, arguments = (
        _production_preclaim_finalization_evidence_fixture(
            tmp_path, "foreign_owner"
        )
    )
    with pytest.raises(RuntimeError, match="consumer owner differs"):
        launcher._load_validate_preclaim_finalization_evidence(
            **arguments
        )
    for mutation in ("tmux_identity", "tmux_server"):
        mutated = copy.deepcopy(arguments)
        if mutation == "tmux_identity":
            mutated["expected_gate_owner_seal"]["pane"] = (
                "%foreign"
            )
        else:
            mutated["expected_gate_owner_seal"]["tmux_server"][
                "socket_inode"
            ] += 1
        with pytest.raises(
            RuntimeError, match="gate owner seal differs"
        ):
            launcher._load_validate_preclaim_finalization_evidence(
                **mutated
            )


@pytest.mark.parametrize(
    ("mutation", "state"),
    (
        ("gate_extra", "gate_evidence"),
        ("field", "consumer_terminal_chain"),
        ("owner", "consumer_terminal_chain"),
        ("attempt", "controller_cleanup_present"),
        ("lifecycle", "consumer_join_present"),
        ("adjudication", "consumer_cleanup_present"),
    ),
)
def test_preclaim_finalization_production_artifact_resigned_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
    state: str,
) -> None:
    launcher, arguments = (
        _production_preclaim_finalization_evidence_fixture(
            tmp_path, state
        )
    )
    receipt = arguments["launch_receipt"]
    artifacts = receipt["pane_fault_consumer"]["artifacts"]
    if mutation == "gate_extra":
        path = arguments["attempt_root"] / (
            "gate_execution_terminal.json"
        )
        digest_field = "gate_execution_terminal_sha256"
        value = load_json(path, "mutated gate terminal")
        value["extra"] = True
    elif mutation in {"field", "owner"}:
        path = Path(str(artifacts["terminal"]))
        digest_field = "consumer_terminal_sha256"
        value = load_json(path, "mutated consumer terminal")
        if mutation == "field":
            value["status"] = "foreign_status"
        else:
            value["consumer_owner_nonce"] = "f" * 64
    elif mutation == "attempt":
        path = Path(str(artifacts["controller_cleanup"]))
        digest_field = "consumer_controller_cleanup_sha256"
        value = load_json(path, "mutated controller cleanup")
        value["attempt_id"] = "f" * 64
    elif mutation == "lifecycle":
        path = Path(str(artifacts["join"]))
        digest_field = "consumer_join_sha256"
        value = load_json(path, "mutated consumer join")
        value["consumer_lifecycle"]["record"]["returncode"] = 0
    else:
        path = Path(str(artifacts["cleanup"]))
        digest_field = "consumer_cleanup_sha256"
        value = load_json(path, "mutated consumer cleanup")
        value["adjudicated_outcome"] = "foreign_outcome"
    value[digest_field] = launcher._canonical_digest(
        value, digest_field
    )
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        launcher._load_validate_preclaim_finalization_evidence(
            **arguments
        )


def test_preclaim_finalization_evidence_reader_has_no_production_callsite() -> None:
    launcher = _launcher_module()
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    callers = {
        function.name
        for function in tree.body
        if isinstance(function, ast.FunctionDef)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "_load_validate_preclaim_finalization_evidence"
            for node in ast.walk(function)
        )
    }
    reader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_load_validate_preclaim_finalization_evidence"
    )
    forbidden = {
        "_kill_exact_session",
        "_terminate_exact_wrapper_child",
        "_write_exclusive",
        "_publish_terminal",
        "join_pane_fault_consumer",
    }
    launch_preflight = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "launch_preflight"
    )
    assert callers == {"_read_preclaim_finalizer_state"}
    assert not {
        node.func.id
        for node in ast.walk(launch_preflight)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }.intersection(
        {
            "_load_validate_preclaim_finalization_evidence",
            "_read_preclaim_finalizer_state",
            "_resume_or_finalize_preclaim_failure",
        }
    )
    assert not {
        node.func.id
        for node in ast.walk(reader)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }.intersection(forbidden)
    join_source = inspect.getsource(
        launcher.join_pane_fault_consumer
    )
    for validator in (
        "_validate_consumer_attempt_evidence",
        "_validate_controller_cleanup_evidence",
        "_validate_consumer_terminal_evidence",
        "_validate_consumer_join_evidence",
        "_validate_consumer_cleanup_evidence",
    ):
        assert validator in join_source
    assert "set(controller_cleanup)" not in join_source
    assert "set(terminal)" not in join_source
    assert "set(value)" not in join_source
    gate_path_source = inspect.getsource(
        launcher._validate_gate_execution_terminal
    )
    assert gate_path_source.count(
        "_sealed_finalization_json("
    ) == 1
    assert "_load_json(" not in gate_path_source
    assert ".resolve(" not in gate_path_source
    formal_gate_source = inspect.getsource(
        launcher._read_formal_gate_lifecycle_status
    )
    assert "_validate_gate_execution_terminal_value(" in (
        formal_gate_source
    )
    assert "_validate_gate_execution_terminal(" not in (
        formal_gate_source
    )
    join_validator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_consumer_join_evidence"
    )
    join_validator_calls = {
        node.func.id
        for node in ast.walk(join_validator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }
    assert not join_validator_calls.intersection(
        {
            "_json_binding",
            "_sealed_finalization_json",
            "_load_json",
            "_read_sealed_json_artifact_at",
            "open",
        }
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "wrong_digest",
        "receipt_drift",
        "receipt_identity_drift",
        "verified_implementation_drift",
        "reason_stage_mismatch",
        "observed_at_non_utc",
        "consumer_chain_drift",
        "owner_seal_drift",
        "deadline_order",
        "deadline_flag",
        "timeout_with_claim",
        "timeout_before_deadline",
        "invalid_without_claim",
        "invalid_claim_sha",
        "invalid_claim_identity",
        "invalid_claim_path",
    ),
)
def test_preclaim_failure_intent_mutations_rejected(
    mutation: str,
) -> None:
    reason = (
        "claim_timeout"
        if mutation.startswith("timeout_")
        else "invalid_claim"
    )
    intent, values = _shared_preclaim_failure_intent(reason)
    mutated = copy.deepcopy(intent)
    if mutation == "missing":
        mutated.pop("stage")
    elif mutation == "extra":
        mutated["extra"] = True
    elif mutation == "wrong_digest":
        mutated["preclaim_failure_intent_sha256"] = hashlib.sha256(
            b"wrong intent digest"
        ).hexdigest()
    elif mutation == "receipt_drift":
        mutated["launch_receipt"] = _shared_binding("other-receipt")
    elif mutation == "receipt_identity_drift":
        mutated["launch_receipt_identity"]["inode"] += 1
    elif mutation == "verified_implementation_drift":
        mutated["verified_implementations"]["verified_loader"][
            "sha256"
        ] = hashlib.sha256(b"other loader").hexdigest()
    elif mutation == "reason_stage_mismatch":
        mutated["stage"] = "wrapper_claim_wait_deadline"
    elif mutation == "observed_at_non_utc":
        mutated["observed_at"] = "2026-07-28T08:00:05+08:00"
    elif mutation == "consumer_chain_drift":
        mutated["pane_fault_consumer_chain"][
            "consumer_active"
        ] = _shared_binding("other-active")
    elif mutation == "owner_seal_drift":
        mutated["controller_owner_seal"]["pane_pid"] += 1
    elif mutation == "deadline_order":
        mutated["deadline_observation"][
            "deadline_monotonic_ns"
        ] = 999
    elif mutation == "deadline_flag":
        mutated["deadline_observation"]["deadline_reached"] = (
            not mutated["deadline_observation"]["deadline_reached"]
        )
    elif mutation == "timeout_with_claim":
        mutated["invalid_claim_evidence"] = {
            "raw_content_sha256": hashlib.sha256(b"raw").hexdigest(),
            "file_identity": _shared_file_identity("raw"),
        }
    elif mutation == "timeout_before_deadline":
        mutated["deadline_observation"][
            "observed_monotonic_ns"
        ] = 1_500
        mutated["deadline_observation"]["deadline_reached"] = False
    elif mutation == "invalid_without_claim":
        mutated["invalid_claim_evidence"] = None
    elif mutation == "invalid_claim_sha":
        mutated["invalid_claim_evidence"][
            "raw_content_sha256"
        ] = "not-a-sha"
    elif mutation == "invalid_claim_identity":
        mutated["invalid_claim_evidence"]["file_identity"][
            "size"
        ] = -1
    elif mutation == "invalid_claim_path":
        mutated["invalid_claim_evidence"]["file_identity"][
            "path"
        ] = "/contract/other-claim.json"
    if mutation != "wrong_digest":
        mutated["preclaim_failure_intent_sha256"] = (
            preflight_launch_digest(
                mutated, "preclaim_failure_intent_sha256"
            )
        )
    with pytest.raises(PreflightLaunchContractError):
        preflight_launch_contract_module.validate_preclaim_failure_intent(
            mutated,
            verified_implementations=values[
                "verified_implementations"
            ],
            expected_wrapper_claim_path=(
                "/contract/wrapper-claim.json"
            ),
            tmux_identity=values["tmux_identity"],
            tmux_server=values["tmux_server"],
            expected_receipt=values["receipt"],
            expected_receipt_identity=values["receipt_identity"],
            expected_consumer_chain=values[
                "pane_fault_consumer_chain"
            ],
        )


def _shared_bound_lifecycle(
    *,
    role: str,
    attempt_id: str,
) -> dict[str, Any]:
    channel_path = f"/contract/{role}_lifecycle_wait.channel"
    channel = {
        "path": channel_path,
        "device": 7,
        "inode": 301 if role == "gate" else 302,
        "mode": 33152,
        "uid": 1000,
        "nlink": 1,
        "size": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "directory_device": 7,
        "directory_inode": 300,
    }
    record = _lifecycle_wait_record(
        preflight_launch_contract_module,
        channel,
        role=role,
    )
    record["attempt_id"] = attempt_id
    record["lifecycle_wait_status_sha256"] = (
        preflight_launch_digest(
            record, "lifecycle_wait_status_sha256"
        )
    )
    artifact = _shared_binding(f"{role}-lifecycle")
    artifact["path"] = channel_path
    artifact["canonical_sha256"] = record[
        "lifecycle_wait_status_sha256"
    ]
    return build_preflight_bound_lifecycle_evidence(
        artifact=artifact,
        record=record,
        role=role,
        attempt_id=attempt_id,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "outer_missing",
        "outer_extra",
        "artifact_sha",
        "record_attempt",
        "record_role",
        "canonical_binding",
    ),
)
def test_bound_lifecycle_evidence_shared_contract_rejects_mutations(
    mutation: str,
):
    attempt_id = hashlib.sha256(b"bound-attempt").hexdigest()
    value = _shared_bound_lifecycle(
        role="gate",
        attempt_id=attempt_id,
    )
    mutated = copy.deepcopy(value)
    if mutation == "outer_missing":
        mutated.pop("artifact")
    elif mutation == "outer_extra":
        mutated["unexpected"] = None
    elif mutation == "artifact_sha":
        mutated["artifact"]["sha256"] = "not-a-sha"
    elif mutation == "record_attempt":
        mutated["record"]["attempt_id"] = hashlib.sha256(
            b"other-attempt"
        ).hexdigest()
        mutated["record"]["lifecycle_wait_status_sha256"] = (
            preflight_launch_digest(
                mutated["record"],
                "lifecycle_wait_status_sha256",
            )
        )
    elif mutation == "record_role":
        mutated["record"]["role"] = "consumer"
        mutated["record"]["lifecycle_wait_status_sha256"] = (
            preflight_launch_digest(
                mutated["record"],
                "lifecycle_wait_status_sha256",
            )
        )
    elif mutation == "canonical_binding":
        mutated["artifact"]["canonical_sha256"] = hashlib.sha256(
            b"other-record"
        ).hexdigest()
    with pytest.raises(PreflightLaunchContractError):
        preflight_launch_contract_module.validate_bound_lifecycle_evidence(
            mutated,
            role="gate",
            attempt_id=attempt_id,
            label="test bound lifecycle evidence",
        )


@pytest.mark.parametrize(
    "mutation",
    ("missing", "extra", "reason", "stage", "type", "message"),
)
def test_terminal_failure_shared_contract_rejects_mutations(
    mutation: str,
):
    value = preflight_launch_contract_module.build_terminal_failure(
        reason="invalid_claim",
        stage="wrapper_claim_validation",
        failure_type="InvalidWrapperClaim",
        message="wrapper claim validation failed",
    )
    mutated = copy.deepcopy(value)
    if mutation == "missing":
        mutated.pop("message")
    elif mutation == "extra":
        mutated["unexpected"] = None
    elif mutation == "reason":
        mutated["reason"] = "claim_timeout"
    elif mutation == "stage":
        mutated["stage"] = "other"
    elif mutation == "type":
        mutated["type"] = ""
    elif mutation == "message":
        mutated["message"] = ""
    with pytest.raises(PreflightLaunchContractError):
        preflight_launch_contract_module.validate_terminal_failure(
            mutated,
            reason="invalid_claim",
            stage="wrapper_claim_validation",
            label="test terminal failure",
        )


def _shared_launch_terminal_v2(
    reason: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    intent, values = _shared_preclaim_failure_intent(reason)
    intent_binding = _shared_binding("preclaim-failure-intent")
    intent_binding["canonical_sha256"] = intent[
        "preclaim_failure_intent_sha256"
    ]
    attempt_id = intent["attempt_id"]
    namespace = values["pane_fault_consumer_chain"][
        "consumer_started"
    ]["path"].rsplit("/", 1)[0]
    status = {
        "invalid_claim": "launcher_failed",
        "claim_timeout": "wrapper_claim_timeout",
    }[reason]
    terminal = (
        preflight_launch_contract_module.build_launch_terminal_v2(
            attempt_id=attempt_id,
            preclaim_failure_intent=intent,
            preclaim_failure_intent_binding=intent_binding,
            launch_receipt=values["receipt"],
            launch_receipt_identity=values["receipt_identity"],
            verified_implementations=values[
                "verified_implementations"
            ],
            pane_fault_consumer_chain=values[
                "pane_fault_consumer_chain"
            ],
            gate_execution_terminal=_shared_binding(
                "gate-execution-terminal"
            ),
            gate_lifecycle=_shared_bound_lifecycle(
                role="gate", attempt_id=attempt_id
            ),
            controller_cleanup={
                **_shared_binding("controller-cleanup"),
                "path": (
                    f"{namespace}/consumer_controller_cleanup.json"
                ),
            },
            consumer_terminal={
                **_shared_binding("consumer-terminal"),
                "path": f"{namespace}/consumer_terminal.json",
            },
            consumer_lifecycle=_shared_bound_lifecycle(
                role="consumer", attempt_id=attempt_id
            ),
            consumer_join={
                **_shared_binding("consumer-join"),
                "path": f"{namespace}/consumer_join.json",
            },
            consumer_cleanup={
                **_shared_binding("consumer-cleanup"),
                "path": f"{namespace}/consumer_cleanup.json",
            },
            status=status,
            failure={
                "reason": reason,
                "stage": intent["stage"],
                "type": (
                    "InvalidWrapperClaim"
                    if reason == "invalid_claim"
                    else "WrapperClaimTimeout"
                ),
                "message": f"formal {reason.replace('_', ' ')}",
            },
            started_at="2026-07-28T00:00:00+00:00",
            completed_at="2026-07-28T00:00:06+00:00",
        )
    )
    return terminal, intent, intent_binding, values


@pytest.mark.parametrize("reason", ("invalid_claim", "claim_timeout"))
def test_launch_terminal_v2_exact_schema_accepted(
    reason: str,
) -> None:
    terminal, intent, intent_binding, values = (
        _shared_launch_terminal_v2(reason)
    )
    for module in (
        preflight_launch_contract_module,
        _launcher_module(),
    ):
        assert (
            module.validate_launch_terminal_v2(
                terminal,
                preclaim_failure_intent=intent,
                preclaim_failure_intent_binding=intent_binding,
                verified_implementations=values[
                    "verified_implementations"
                ],
            )
            == terminal
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "wrong_digest",
        "attempt_drift",
        "intent_binding_drift",
        "receipt_drift",
        "receipt_identity_drift",
        "verified_implementation_drift",
        "consumer_chain_drift",
        "gate_lifecycle_artifact_drift",
        "gate_lifecycle_record_drift",
        "consumer_lifecycle_role_drift",
        "lifecycle_policy_drift",
        "controller_cleanup_path_drift",
        "consumer_terminal_path_drift",
        "ownership_present",
        "session_residual",
        "process_residual",
        "status_drift",
        "failure_reason_drift",
        "duplicate_artifact_path",
        "started_at_non_utc",
        "completed_at_malformed",
        "completed_at_noncanonical",
        "started_at_wrong_type",
        "time_reversed",
    ),
)
def test_launch_terminal_v2_mutations_rejected(
    mutation: str,
) -> None:
    terminal, intent, intent_binding, values = (
        _shared_launch_terminal_v2("invalid_claim")
    )
    mutated = copy.deepcopy(terminal)
    if mutation == "missing":
        mutated.pop("status")
    elif mutation == "extra":
        mutated["extra"] = True
    elif mutation == "wrong_digest":
        mutated["launch_terminal_sha256"] = hashlib.sha256(
            b"wrong terminal digest"
        ).hexdigest()
    elif mutation == "attempt_drift":
        mutated["attempt_id"] = hashlib.sha256(
            b"other attempt"
        ).hexdigest()
    elif mutation == "intent_binding_drift":
        mutated["preclaim_failure_intent"] = _shared_binding(
            "other-intent"
        )
    elif mutation == "receipt_drift":
        mutated["launch_receipt"] = _shared_binding("other-receipt")
    elif mutation == "receipt_identity_drift":
        mutated["launch_receipt_identity"]["inode"] += 1
    elif mutation == "verified_implementation_drift":
        mutated["verified_implementations"]["verified_loader"][
            "sha256"
        ] = hashlib.sha256(b"other loader").hexdigest()
    elif mutation == "consumer_chain_drift":
        mutated["pane_fault_consumer_chain"][
            "consumer_active"
        ] = _shared_binding("other-active")
    elif mutation == "gate_lifecycle_artifact_drift":
        mutated["gate_lifecycle"]["artifact"][
            "canonical_sha256"
        ] = hashlib.sha256(b"other lifecycle").hexdigest()
    elif mutation == "gate_lifecycle_record_drift":
        mutated["gate_lifecycle"]["record"]["attempt_id"] = (
            hashlib.sha256(b"other lifecycle attempt").hexdigest()
        )
    elif mutation == "consumer_lifecycle_role_drift":
        mutated["consumer_lifecycle"]["record"]["role"] = "gate"
    elif mutation == "lifecycle_policy_drift":
        lifecycle = mutated["consumer_lifecycle"]
        lifecycle["record"]["policy_sha256"] = hashlib.sha256(
            b"other policy"
        ).hexdigest()
        lifecycle["record"]["lifecycle_wait_status_sha256"] = (
            preflight_launch_digest(
                lifecycle["record"],
                "lifecycle_wait_status_sha256",
            )
        )
        lifecycle["artifact"]["canonical_sha256"] = lifecycle[
            "record"
        ]["lifecycle_wait_status_sha256"]
    elif mutation == "controller_cleanup_path_drift":
        mutated["controller_cleanup"]["path"] = (
            "/contract/other_controller_cleanup.json"
        )
    elif mutation == "consumer_terminal_path_drift":
        mutated["consumer_terminal"]["path"] = (
            "/contract/other_consumer_terminal.json"
        )
    elif mutation == "ownership_present":
        mutated["ownership"]["wrapper_claim"] = _shared_binding("claim")
    elif mutation == "session_residual":
        mutated["session_residual"] = True
    elif mutation == "process_residual":
        mutated["process_residual"] = True
    elif mutation == "status_drift":
        mutated["status"] = "wrapper_claim_timeout"
    elif mutation == "failure_reason_drift":
        mutated["failure"]["reason"] = "claim_timeout"
    elif mutation == "duplicate_artifact_path":
        mutated["consumer_cleanup"]["path"] = mutated[
            "consumer_join"
        ]["path"]
    elif mutation == "started_at_non_utc":
        mutated["started_at"] = "2026-07-28T08:00:00+08:00"
    elif mutation == "completed_at_malformed":
        mutated["completed_at"] = "not-a-timestamp"
    elif mutation == "completed_at_noncanonical":
        mutated["completed_at"] = "2026-07-28 00:00:06+00:00"
    elif mutation == "started_at_wrong_type":
        mutated["started_at"] = 1
    elif mutation == "time_reversed":
        mutated["completed_at"] = "2026-07-27T23:59:59+00:00"
    if mutation != "wrong_digest":
        mutated["launch_terminal_sha256"] = (
            preflight_launch_digest(
                mutated, "launch_terminal_sha256"
            )
        )
    with pytest.raises(PreflightLaunchContractError):
        preflight_launch_contract_module.validate_launch_terminal_v2(
            mutated,
            preclaim_failure_intent=intent,
            preclaim_failure_intent_binding=intent_binding,
            verified_implementations=values[
                "verified_implementations"
            ],
        )


def _shared_post_handoff_finalization_failure(
    reason: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    int,
]:
    terminal, intent, intent_binding, values = (
        _shared_launch_terminal_v2(reason)
    )
    target_path = "/contract/launch_terminal.json"
    content_sha256 = hashlib.sha256(
        json.dumps(
            terminal, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    payload_size = 4096
    evidence = (
        preflight_launch_contract_module
        .build_post_handoff_finalization_failure(
            attempted_launch_terminal=terminal,
            preclaim_failure_intent=intent,
            preclaim_failure_intent_binding=intent_binding,
            verified_implementations=values[
                "verified_implementations"
            ],
            publish_target_path=target_path,
            attempted_content_sha256=content_sha256,
            attempted_payload_size=payload_size,
            failure={
                "outer": {
                    "type": "LauncherTerminalPublishError",
                    "message": (
                        "terminal publication failed for "
                        f"{target_path}: injected terminal write"
                    ),
                    "path": target_path,
                    "secondary_failures": [
                        {
                            "stage": "external_tmux_cleanup",
                            "type": "RuntimeError",
                            "message": "cleanup evidence retained",
                        }
                    ],
                },
                "inner": (
                    preflight_launch_contract_module
                    .build_finalization_inner_failure(
                        OSError(5, "injected terminal write")
                    )
                ),
            },
            started_at="2026-07-28T00:00:00+00:00",
            completed_at="2026-07-28T00:00:07+00:00",
        )
    )
    return (
        evidence,
        terminal,
        intent,
        intent_binding,
        content_sha256,
        payload_size,
    )


@pytest.mark.parametrize("reason", ("invalid_claim", "claim_timeout"))
def test_post_handoff_finalization_failure_exact_schema_accepted(
    reason: str,
) -> None:
    (
        evidence,
        terminal,
        intent,
        intent_binding,
        content_sha256,
        payload_size,
    ) = _shared_post_handoff_finalization_failure(reason)
    values = _shared_launch_contract_values()
    for module in (
        preflight_launch_contract_module,
        _launcher_module(),
    ):
        assert (
            module.validate_post_handoff_finalization_failure(
                evidence,
                attempted_launch_terminal=terminal,
                preclaim_failure_intent=intent,
                preclaim_failure_intent_binding=intent_binding,
                verified_implementations=values[
                    "verified_implementations"
                ],
                publish_target_path="/contract/launch_terminal.json",
                attempted_content_sha256=content_sha256,
                attempted_payload_size=payload_size,
            )
            == evidence
        )


@pytest.mark.parametrize(
    "failure",
    (
        OSError(5, "ordinary OSError"),
        PermissionError(13, "permission denied"),
    ),
)
def test_post_handoff_finalization_failure_accepts_os_error_subtypes(
    failure: OSError,
) -> None:
    (
        evidence,
        terminal,
        intent,
        intent_binding,
        content_sha256,
        payload_size,
    ) = _shared_post_handoff_finalization_failure("invalid_claim")
    values = _shared_launch_contract_values()
    evidence["failure"]["inner"] = (
        preflight_launch_contract_module
        .build_finalization_inner_failure(failure)
    )
    evidence[
        "post_handoff_finalization_failure_sha256"
    ] = preflight_launch_digest(
        evidence,
        "post_handoff_finalization_failure_sha256",
    )
    for module in (
        preflight_launch_contract_module,
        _launcher_module(),
    ):
        assert (
            module.validate_post_handoff_finalization_failure(
                evidence,
                attempted_launch_terminal=terminal,
                preclaim_failure_intent=intent,
                preclaim_failure_intent_binding=intent_binding,
                verified_implementations=values[
                    "verified_implementations"
                ],
                publish_target_path="/contract/launch_terminal.json",
                attempted_content_sha256=content_sha256,
                attempted_payload_size=payload_size,
            )
            == evidence
        )


@pytest.mark.parametrize(
    "failure",
    (
        RuntimeError("runtime"),
        ValueError("value"),
        type(
            "PaneFaultConsumerReservationError",
            (RuntimeError,),
            {},
        )("reservation"),
    ),
)
def test_finalization_inner_failure_builder_rejects_non_os_errors(
    failure: BaseException,
) -> None:
    with pytest.raises(PreflightLaunchContractError):
        (
            preflight_launch_contract_module
            .build_finalization_inner_failure(failure)
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "wrong_digest",
        "attempt_drift",
        "intent_binding_drift",
        "target_drift",
        "canonical_digest_drift",
        "content_digest_drift",
        "payload_size_zero",
        "payload_size_wrong_type",
        "invented_file_identity",
        "gate_terminal_drift",
        "gate_lifecycle_drift",
        "controller_cleanup_drift",
        "consumer_terminal_drift",
        "consumer_lifecycle_drift",
        "consumer_join_drift",
        "consumer_cleanup_drift",
        "ownership_present",
        "session_residual",
        "process_residual",
        "stage_drift",
        "reservation_outer_type",
        "reservation_inner_type",
        "runtime_inner_type",
        "value_inner_type",
        "outer_path_drift",
        "outer_secondary_extra",
        "inner_type_missing",
        "inner_errno_wrong_type",
        "legacy_close_fault_shape",
        "started_non_utc",
        "completed_malformed",
        "completed_noncanonical",
        "started_wrong_type",
        "time_reversed",
    ),
)
def test_post_handoff_finalization_failure_mutations_rejected(
    mutation: str,
) -> None:
    (
        evidence,
        terminal,
        intent,
        intent_binding,
        content_sha256,
        payload_size,
    ) = _shared_post_handoff_finalization_failure("invalid_claim")
    values = _shared_launch_contract_values()
    mutated = copy.deepcopy(evidence)
    if mutation == "missing":
        mutated.pop("stage")
    elif mutation == "extra":
        mutated["extra"] = True
    elif mutation == "wrong_digest":
        mutated[
            "post_handoff_finalization_failure_sha256"
        ] = hashlib.sha256(b"wrong finalization digest").hexdigest()
    elif mutation == "attempt_drift":
        mutated["attempt_id"] = hashlib.sha256(
            b"other finalization attempt"
        ).hexdigest()
    elif mutation == "intent_binding_drift":
        mutated["preclaim_failure_intent"] = _shared_binding(
            "other-finalization-intent"
        )
    elif mutation == "target_drift":
        mutated["attempted_launch_terminal"]["target_path"] = (
            "/contract/other_terminal.json"
        )
    elif mutation == "canonical_digest_drift":
        mutated["attempted_launch_terminal"][
            "canonical_sha256"
        ] = hashlib.sha256(b"other canonical payload").hexdigest()
    elif mutation == "content_digest_drift":
        mutated["attempted_launch_terminal"][
            "content_sha256"
        ] = hashlib.sha256(b"other content payload").hexdigest()
    elif mutation == "payload_size_zero":
        mutated["attempted_launch_terminal"]["size"] = 0
    elif mutation == "payload_size_wrong_type":
        mutated["attempted_launch_terminal"]["size"] = "4096"
    elif mutation == "invented_file_identity":
        mutated["attempted_launch_terminal"]["file_identity"] = (
            _shared_file_identity("invented-terminal")
        )
    elif mutation == "gate_terminal_drift":
        mutated["gate_execution_terminal"] = _shared_binding(
            "other-gate-terminal"
        )
    elif mutation == "gate_lifecycle_drift":
        mutated["gate_lifecycle"]["artifact"][
            "content_sha256"
        ] = hashlib.sha256(b"extra").hexdigest()
    elif mutation == "controller_cleanup_drift":
        mutated["controller_cleanup"] = _shared_binding(
            "other-controller-cleanup"
        )
    elif mutation == "consumer_terminal_drift":
        mutated["consumer_terminal"] = _shared_binding(
            "other-consumer-terminal"
        )
    elif mutation == "consumer_lifecycle_drift":
        mutated["consumer_lifecycle"]["record"]["attempt_id"] = (
            hashlib.sha256(b"other consumer attempt").hexdigest()
        )
    elif mutation == "consumer_join_drift":
        mutated["consumer_join"] = _shared_binding(
            "other-consumer-join"
        )
    elif mutation == "consumer_cleanup_drift":
        mutated["consumer_cleanup"] = _shared_binding(
            "other-consumer-cleanup"
        )
    elif mutation == "ownership_present":
        mutated["ownership"]["launch_accepted"] = _shared_binding(
            "accepted"
        )
    elif mutation == "session_residual":
        mutated["session_residual"] = True
    elif mutation == "process_residual":
        mutated["process_residual"] = True
    elif mutation == "stage_drift":
        mutated["stage"] = "external_tmux_cleanup"
    elif mutation == "reservation_outer_type":
        mutated["failure"]["outer"]["type"] = (
            "PaneFaultConsumerReservationError"
        )
    elif mutation == "reservation_inner_type":
        mutated["failure"]["inner"]["exception_type"] = (
            "PaneFaultConsumerReservationError"
        )
    elif mutation == "runtime_inner_type":
        mutated["failure"]["inner"]["exception_type"] = "RuntimeError"
    elif mutation == "value_inner_type":
        mutated["failure"]["inner"]["exception_type"] = "ValueError"
    elif mutation == "outer_path_drift":
        mutated["failure"]["outer"]["path"] = (
            "/contract/other_terminal.json"
        )
    elif mutation == "outer_secondary_extra":
        mutated["failure"]["outer"]["secondary_failures"][0][
            "extra"
        ] = True
    elif mutation == "inner_type_missing":
        mutated["failure"]["inner"]["exception_type"] = ""
    elif mutation == "inner_errno_wrong_type":
        mutated["failure"]["inner"]["error_number"] = "5"
    elif mutation == "legacy_close_fault_shape":
        mutated["failure"]["inner"] = {
            "type": "OSError",
            "message": "legacy close fault",
            "errno": "5",
        }
    elif mutation == "started_non_utc":
        mutated["started_at"] = "2026-07-28T08:00:00+08:00"
    elif mutation == "completed_malformed":
        mutated["completed_at"] = "not-a-timestamp"
    elif mutation == "completed_noncanonical":
        mutated["completed_at"] = "2026-07-28 00:00:07+00:00"
    elif mutation == "started_wrong_type":
        mutated["started_at"] = 1
    elif mutation == "time_reversed":
        mutated["completed_at"] = "2026-07-27T23:59:59+00:00"
    if mutation != "wrong_digest":
        mutated[
            "post_handoff_finalization_failure_sha256"
        ] = preflight_launch_digest(
            mutated,
            "post_handoff_finalization_failure_sha256",
        )
    with pytest.raises(PreflightLaunchContractError):
        (
            preflight_launch_contract_module
            .validate_post_handoff_finalization_failure(
                mutated,
                attempted_launch_terminal=terminal,
                preclaim_failure_intent=intent,
                preclaim_failure_intent_binding=intent_binding,
                verified_implementations=values[
                    "verified_implementations"
                ],
                publish_target_path="/contract/launch_terminal.json",
                attempted_content_sha256=content_sha256,
                attempted_payload_size=payload_size,
            )
        )


@pytest.mark.parametrize(
    "error_kind",
    (
        "exclusive_publish",
        "terminal_publish",
        "gate_fault",
        "consumer_reservation",
    ),
)
@pytest.mark.parametrize("mutation", ("none", "missing", "extra"))
def test_launcher_typed_errors_share_secondary_failure_contract(
    error_kind: str,
    mutation: str,
) -> None:
    launcher = _launcher_module()
    primary = OSError(5, "primary")
    errors = {
        "exclusive_publish": launcher.LauncherExclusivePublishError(
            "precommit_failed_clean",
            "primary",
            stage="publish",
            directory_seal={},
            payload={},
            temporary=None,
            error_number=5,
            quarantined=False,
        ),
        "terminal_publish": launcher.LauncherTerminalPublishError(
            Path("/contract/launch_terminal.json"), primary
        ),
        "gate_fault": launcher.LauncherGateFaultError(
            "gate_fault", snapshot=None, failure=primary
        ),
        "consumer_reservation": (
            launcher.PaneFaultConsumerReservationError(primary)
        ),
    }
    error = errors[error_kind]
    secondary = RuntimeError("secondary failure")
    error.add_secondary_failure(
        stage="external_tmux_cleanup", failure=secondary
    )
    expected = launcher.build_finalization_secondary_failure(
        stage="external_tmux_cleanup",
        failure_type="RuntimeError",
        message="secondary failure",
    )
    assert error.secondary_failures == [expected]
    mutated = copy.deepcopy(expected)
    if mutation == "missing":
        mutated.pop("stage")
    elif mutation == "extra":
        mutated["extra"] = True
    if mutation == "none":
        assert (
            launcher.validate_finalization_secondary_failure(
                mutated
            )
            == expected
        )
    else:
        with pytest.raises(launcher.PreflightLaunchContractError):
            launcher.validate_finalization_secondary_failure(mutated)


_VERIFIED_IMPLEMENTATION_CONTRACT_CALLS = frozenset(
    {
        "build_gate_ready",
        "validate_gate_ready",
        "build_tmux_started",
        "validate_tmux_started",
        "build_wrapper_started",
        "validate_wrapper_started",
        "build_preclaim_failure_intent",
        "validate_preclaim_failure_intent",
        "build_launch_terminal_v2",
        "validate_launch_terminal_v2",
        "build_post_handoff_finalization_failure",
        "validate_post_handoff_finalization_failure",
        "build_claim_v3",
        "validate_claim_v3",
        "build_launch_accepted",
        "build_ownership_terminal",
        "build_ownership_release",
        "validate_ownership_chain",
    }
)


def test_shared_verified_implementation_chain_is_required_by_every_api():
    for name in sorted(_VERIFIED_IMPLEMENTATION_CONTRACT_CALLS):
        function = getattr(preflight_launch_contract_module, name)
        signature = inspect.signature(function)
        parameter = signature.parameters.get("verified_implementations")
        assert parameter is not None, name
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is inspect.Parameter.empty, name
        arguments = {
            item.name: None
            for item in signature.parameters.values()
            if item.default is inspect.Parameter.empty
            and item.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            and item.name != "verified_implementations"
        }
        with pytest.raises(TypeError, match="verified_implementations"):
            function(**arguments)


def test_all_verified_implementation_contract_calls_pass_chain_explicitly():
    repo_root = Path(__file__).parents[1]
    missing: list[str] = []
    for source_root in (
        repo_root / "src",
        repo_root / "scripts",
        repo_root / "tests",
    ):
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                else:
                    continue
                if name not in _VERIFIED_IMPLEMENTATION_CONTRACT_CALLS:
                    continue
                if not any(
                    keyword.arg == "verified_implementations"
                    for keyword in node.keywords
                ):
                    missing.append(
                        f"{path.relative_to(repo_root)}:{node.lineno}:{name}"
                    )
    assert missing == []


@pytest.mark.parametrize("mutation", ("missing", "extra", "schema_v2"))
def test_shared_ownership_mutations_rejected_by_all_consumers(
    mutation: str,
):
    modules = (
        _launcher_module(),
        _wrapper_module(),
        _controller_module(),
    )
    values = _shared_launch_contract_values()
    accepted = json.loads(json.dumps(values["accepted"]))
    if mutation == "missing":
        accepted.pop("attempt_id")
    elif mutation == "extra":
        accepted["unshared_schema_field"] = True
    else:
        accepted["schema_version"] = 2
        accepted["launch_accepted_sha256"] = preflight_launch_digest(
            accepted, "launch_accepted_sha256"
        )
    for module in modules:
        expected_error = getattr(
            module,
            "PreflightLaunchContractError",
            PreflightLaunchContractError,
        )
        with pytest.raises(expected_error):
            module.validate_ownership_chain(
                accepted,
                values["terminal"],
                values["release"],
                receipt_binding=values["receipt"],
                receipt_identity=values["receipt_identity"],
                wrapper_binding=values["wrapper_binding"],
                accepted_binding=values["accepted_binding"],
                terminal_binding=values["terminal_binding"],
                verified_implementations=values[
                    "verified_implementations"
                ],
                pane_fault_consumer_chain=values[
                    "pane_fault_consumer_chain"
                ],
            )


def _gpu_wrapper_module():
    path = Path(__file__).parents[1] / "scripts" / "run_canonical_gpu_wrapper.py"
    spec = importlib.util.spec_from_file_location("canonical_gpu_wrapper_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ram_probe_module():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_canonical_screening_ram_probe.py"
    )
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(
        "canonical_ram_probe_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json(row) for row in rows))


def _bound(path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(b"x")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _test_executable_identity(
    path: str | Path = sys.executable,
) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    value = resolved.stat()
    return build_preflight_file_identity(
        path=str(resolved),
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        size=int(value.st_size),
    )


def _test_process_identity(
    pid: int,
    *,
    ppid: int = 1,
    pgid: int | None = None,
    sid: int | None = None,
    start_ticks: int = 77,
) -> dict[str, int]:
    return build_preflight_process_identity(
        pid=pid,
        ppid=ppid,
        pgid=pid if pgid is None else pgid,
        sid=pid if sid is None else sid,
        start_ticks=start_ticks,
    )


def _test_tmux_server_identity(
    path: Path,
    *,
    server_pid: int,
    server_process: Mapping[str, Any] | None = None,
    socket_device: int = 1,
    socket_inode: int = 2,
) -> dict[str, Any]:
    process = (
        _test_process_identity(server_pid)
        if server_process is None
        else dict(server_process)
    )
    return build_preflight_tmux_server_identity(
        server_pid=server_pid,
        server_process=process,
        socket_path=str(path.resolve()),
        socket_device=socket_device,
        socket_inode=socket_inode,
    )


def _test_file_identity(
    path: Path,
    *,
    device: int = 1,
    inode: int = 2,
    mode: int = 0o100644,
    size: int = 1,
) -> dict[str, Any]:
    return build_preflight_file_identity(
        path=str(path.resolve()),
        device=device,
        inode=inode,
        mode=mode,
        size=size,
    )


def _test_verified_preflight_implementations() -> dict[str, Any]:
    root = Path(__file__).parents[1]

    def implementation(path: Path) -> dict[str, Any]:
        resolved = path.resolve(strict=True)
        value = resolved.stat()
        return {
            "path": str(resolved),
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            "file_identity": build_preflight_file_identity(
                path=str(resolved),
                device=int(value.st_dev),
                inode=int(value.st_ino),
                mode=int(value.st_mode),
                size=int(value.st_size),
            ),
        }

    return build_preflight_verified_implementations(
        verified_loader=implementation(
            root
            / "src/safa/closeout/verified_preflight_module_loader.py"
        ),
        preflight_launch_contract=implementation(
            root / "src/safa/closeout/preflight_launch_contract.py"
        ),
    )


def _test_validated_preflight_launch(
    *,
    tmp_path: Path,
    attempt_id: str,
    receipt_binding: Mapping[str, Any],
    receipt_identity: Mapping[str, Any],
    gate_ready_binding: Mapping[str, Any],
    tmux_started_binding: Mapping[str, Any],
    wrapper_started_binding: Mapping[str, Any],
    gate_supervisor_process: Mapping[str, Any],
    gate_process: Mapping[str, Any],
    wrapper_process: Mapping[str, Any],
    wrapper_arguments: list[str],
    wrapper_executable: Mapping[str, Any],
    pane_log: Mapping[str, Any],
) -> dict[str, Any]:
    verified_implementations = (
        _test_verified_preflight_implementations()
    )
    gate_ready = build_preflight_gate_ready(
        launch_receipt=receipt_binding,
        launch_receipt_identity=receipt_identity,
        verified_implementations=verified_implementations,
        process=gate_process,
        wrapper_arguments=wrapper_arguments,
        ready_at="2026-07-28T00:00:00+00:00",
    )
    wrapper_started = build_preflight_wrapper_started(
        launch_receipt=receipt_binding,
        launch_receipt_identity=receipt_identity,
        verified_implementations=verified_implementations,
        pane_gate_ready=gate_ready_binding,
        pane_gate_process=gate_process,
        wrapper_arguments=wrapper_arguments,
        wrapper_process=wrapper_process,
        wrapper_executable=wrapper_executable,
        started_at="2026-07-28T00:00:01+00:00",
        gate_ready=gate_ready,
    )
    pane_fault_consumer_chain = {
        "consumer_started": _shared_binding(
            f"{attempt_id}-consumer-started"
        ),
        "consumer_active": _shared_binding(
            f"{attempt_id}-consumer-active"
        ),
        "consumer_reader_release": _shared_binding(
            f"{attempt_id}-consumer-reader-release"
        ),
        "consumer_release_observed": _shared_binding(
            f"{attempt_id}-consumer-release-observed"
        ),
    }
    receipt = _shared_launch_receipt_v4()
    preflight_launch_contract_module.validate_launch_receipt_schema(
        receipt,
        expected_gate_worker_arguments=receipt[
            "gate_worker_arguments"
        ],
        expected_consumer_worker_arguments=receipt[
            "consumer_worker_arguments"
        ],
        label="validated launch fixture receipt v4",
    )
    return {
        "attempt_id": attempt_id,
        "receipt": receipt,
        "receipt_binding": dict(receipt_binding),
        "receipt_identity": dict(receipt_identity),
        "verified_implementations": verified_implementations,
        "gate_ready_binding": dict(gate_ready_binding),
        "tmux_started_binding": dict(tmux_started_binding),
        "wrapper_started_binding": dict(wrapper_started_binding),
        "gate_supervisor_process": dict(
            gate_supervisor_process
        ),
        "gate_process": dict(gate_process),
        "wrapper_launch_process": dict(wrapper_process),
        "accepted_path": tmp_path / "fixture-launch-accepted.json",
        "release_path": tmp_path / "fixture-launch-release.json",
        "wrapper_arguments": list(wrapper_arguments),
        "wrapper_executable": dict(wrapper_executable),
        "pane_log": dict(pane_log),
        "git": {},
        "gate_ready": gate_ready,
        "wrapper_started": wrapper_started,
        "pane_fault_consumer_chain": pane_fault_consumer_chain,
        "pane_fault_consumer_registration": {
            **receipt["pane_fault_consumer"],
        },
    }


def _decoder_registry(tmp_path: Path) -> dict:
    bound = _bound(tmp_path / "decoder-bound.bin")
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
            "sampling_implementation": dict(bound),
        },
        "latent": {
            "decoder_type": "r9_frozen_sd_vae_ft_ema",
            "vae_source_path": "artifacts/checkpoints/external/sd-vae-ft-ema",
            "directory": {"path": str(tmp_path), "digest": "a" * 64},
            "config": dict(bound),
            "weights": dict(bound),
            "scaling_factor": 0.18215,
            "implementation": dict(bound),
            "trusted_runtime_config": dict(bound),
            "trusted_runner": dict(bound),
            "trusted_reference_checkpoint": dict(bound),
            "trusted_resolved_config": dict(bound),
            "trusted_generation_result": dict(bound),
            "environment": {
                "provenance_snapshot": dict(bound),
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
            "asset_digest_cache_algorithm": dict(bound),
            "latent_shape": ["B", 4, 32, 32],
            "decoded_rgb_shape": ["B", 3, 256, 256],
            "output_range": [0.0, 1.0],
        },
        "decoder_registry_sha256": "",
    }
    registry["decoder_registry_sha256"] = decoder_registry_digest(registry)
    return registry


def test_asset_content_hash_rejects_same_size_restored_time_and_forged_cache(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "vae"
    asset.mkdir()
    weights = asset / "weights.bin"
    weights.write_bytes(b"AAAA")
    expected = hashlib.sha256(
        b"weights.bin\0" + b"AAAA" + b"\0"
    ).hexdigest()
    verification = hash_asset_directory_content(asset, expected)
    original_mtime_ns = weights.stat().st_mtime_ns
    weights.write_bytes(b"BBBB")
    os.utime(weights, ns=(original_mtime_ns, original_mtime_ns))
    current = weights.stat()
    forged_cache = {
        "forged": {
            "path": str(asset.resolve()),
            "expected_digest": expected,
            "stat_fingerprint": [
                {
                    "relative": "weights.bin",
                    "device": int(current.st_dev),
                    "inode": int(current.st_ino),
                    "size": int(current.st_size),
                    "mtime_ns": int(current.st_mtime_ns),
                    "ctime_ns": int(current.st_ctime_ns),
                }
            ],
            "digest": expected,
        }
    }
    (tmp_path / "forged-cache.json").write_text(
        json.dumps(forged_cache), encoding="utf-8"
    )
    assert verification["total_bytes"] == 4
    with pytest.raises(
        CanonicalScreeningError,
        match="asset directory content digest differs",
    ):
        hash_asset_directory_content(asset, expected)


def _pixel_output_contract(checkpoint_sha256: str, registry: dict) -> dict:
    capability = resolve_checkpoint_output_capability(
        {
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
        },
        checkpoint_sha256,
    )
    return bind_output_contract(capability, registry)


def _asset_content_verification(policy: dict) -> dict:
    directory = policy["output_decoder_registry"]["latent"]["directory"]
    return {
        "schema_version": 1,
        "contract_type": "safa_canonical_asset_content_verification_v1",
        "path": directory["path"],
        "digest_algorithm": "sha256_relative_posix_nul_content_nul_v1",
        "expected_digest": directory["digest"],
        "observed_digest": directory["digest"],
        "file_count": 2,
        "total_bytes": 2,
        "elapsed_seconds": 0.01,
        "started_at": "2026-07-27T00:00:00+00:00",
        "completed_at": "2026-07-27T00:00:00+00:00",
    }


def _policy(tmp_path: Path, ledger: Path) -> tuple[dict, Path, dict]:
    bound = _bound(tmp_path / "bound.bin")
    smoke_manifest = tmp_path / "smoke8.jsonl"
    screen_manifest = tmp_path / "screen512.jsonl"
    _write_jsonl(
        smoke_manifest,
        [{"sample_id": f"s{index}"} for index in range(8)],
    )
    _write_jsonl(
        screen_manifest,
        [{"sample_id": f"s{index}"} for index in range(512)],
    )
    implementations = {
        name: dict(bound)
        for name in (
            "checkpoint_preflight",
            "arcface_evaluator",
            "e0_loader",
            "canonical_quality",
            "screening_contracts",
            "screening_worker",
            "controller",
            "ram_probe_launcher",
            "preflight_launcher",
            "preflight_wrapper",
            "generator_sampling",
            "meanflow_sampling",
            "latent_codec",
            "output_contract",
        )
    }
    root = Path(__file__).parents[1]
    for name, relative in (
        (
            "preflight_verified_loader",
            "src/safa/closeout/verified_preflight_module_loader.py",
        ),
        (
            "preflight_launch_contract",
            "src/safa/closeout/preflight_launch_contract.py",
        ),
    ):
        path = (root / relative).resolve(strict=True)
        implementations[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    policy = {
        "campaign_id": "historical-canonical-512-v1",
        "supersedes_policy_sha256": "f7d9b8e263bdd54af7754889c7e7ce92d3ec7212d3784ac11c819fc3c07381cd",
        "python": "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python",
        "policy_sha256": "1" * 64,
        "source": {
            "ledger": {
                "path": str(ledger.resolve()),
                "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            }
        },
        "protocol": {
            "seed": 4549,
            "batch_size": 2,
            "manifests": {
                "smoke8": {**_bound(smoke_manifest), "sample_count": 8},
                "screen512": {
                    **_bound(screen_manifest),
                    "sample_count": 512,
                },
            },
            "source_index": bound,
            "features": {"directory": str(tmp_path), "manifest": bound, "shard": bound},
            "e0": bound,
            "edev": bound,
            "quality_script": bound,
            "pixel_image_size": 256,
            "pixel_protocol_config": bound,
            "kid_subset_sizes": {"smoke8": 8, "screen512": 50},
            "metrics": [],
        },
        "resources": {
            "physical_gpus": [0, 1, 2, 3],
            "workers_per_gpu": 2,
            "ram_budget_status": "sealed",
            "ram_slot_budget_bytes": 1100,
            "ram_slot_budget_source": {
                "contract_type": "safa_canonical_screening_ram_budget_source_v1",
                "method": (
                    "ceil(single_worker_process_tree_peak_rss_bytes*11/10)"
                ),
                "measurement_factor_numerator": 11,
                "measurement_factor_denominator": 10,
                "peak_process_tree_rss_bytes": 1000,
                "ram_slot_budget_bytes": 1100,
                "probe_result": bound,
            },
            "gpu_headroom_bytes": 2 * 1024**3,
            "cpu_admission_percent": 90,
            "cpu_hard_limit_percent": 90,
            "cpu_window_seconds": 60,
            "cpu_consecutive_hard_windows": 2,
            "resource_poll_seconds": 10,
            "swap_consecutive_hard_intervals": 3,
            "ram_admission_percent": 85,
            "ram_hard_limit_percent": 90,
            "disk_admission_percent": 85,
            "disk_hard_limit_percent": 90,
            "retry_count": 0,
            "require_tmux": True,
            "global_lock_root": str(tmp_path / "locks"),
        },
        "implementations": implementations,
        "arcface": {"model_name": "buffalo_l"},
        "output_decoder_registry": _decoder_registry(tmp_path),
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"policy":"fixture"}\n', encoding="utf-8")
    admission_value = {
        "contract_type": "safa_canonical_resource_admission_v1",
        "policy_sha256": policy["policy_sha256"],
        "snapshot": _admission_snapshot(policy),
    }
    admission_value["admission_sha256"] = canonical_digest(
        admission_value, "admission_sha256"
    )
    admission_path = tmp_path / "admission.json"
    write_exclusive_json(admission_path, admission_value)
    admission = {
        "path": str(admission_path.resolve()),
        "sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "canonical_sha256": admission_value["admission_sha256"],
    }
    return policy, policy_path, admission


def _admission_snapshot(policy: dict) -> dict:
    slot_count = (
        len(policy["resources"]["physical_gpus"])
        * int(policy["resources"]["workers_per_gpu"])
    )
    slot_budget_bytes = int(
        policy["resources"]["ram_slot_budget_bytes"]
    )
    reserved_bytes = slot_count * slot_budget_bytes
    memory_total_bytes = max(100000, reserved_bytes * 10)
    memory_used_bytes = memory_total_bytes // 10
    projected_used_bytes = memory_used_bytes + reserved_bytes
    return {
        "gpus": [],
        "compute_processes": [],
        "authorized_gpu_registry": [
            {
                "physical_gpu_index": index,
                "physical_gpu_uuid": _gpu_uuid(index),
            }
            for index in range(4)
        ],
        "ram_reservation": {
            "slot_count": slot_count,
            "slot_budget_bytes": slot_budget_bytes,
            "reserved_bytes": reserved_bytes,
            "memory_total_bytes": memory_total_bytes,
            "memory_used_bytes": memory_used_bytes,
            "projected_used_bytes": projected_used_bytes,
            "projected_used_percent": (
                100.0 * projected_used_bytes / memory_total_bytes
            ),
            "admission_limit_percent": policy["resources"][
                "ram_admission_percent"
            ],
            "budget_source": policy["resources"]["ram_slot_budget_source"],
        },
    }


def _row(run_id: str, sha: str | None, selector: str = "raw", path: str | None = None) -> dict:
    return {
        "run_id": run_id,
        "status": "config_only_never_started" if sha is None else "started_incomplete",
        "logical_experiment_id": "R6",
        "protocol_family": "family",
        "comparability_group": "group",
        "evidence_level": "strong_provenance_historical_baseline",
        "checkpoint": {
            "files": [] if sha is None else [{
                "path": path or f"artifacts/{run_id}.pt",
                "sha256": sha,
                "size_bytes": 10,
            }],
            "selector": selector,
        },
    }


def _strict_preflight(
    sha: str,
    selector: str,
    registry: dict,
    status: str = "valid",
) -> dict:
    valid = status == "valid"
    return {
        "schema_version": 1,
        "contract_type": "safa_generator_checkpoint_preflight_v1",
        "status": status,
        "checkpoint_path": "/checkpoint",
        "checkpoint_sha256": sha,
        "expected_checkpoint_sha256": sha,
        "sha256_binding": "expected_exact",
        "checkpoint_model": selector,
        "declared_checkpoint_model": None,
        "available_state_dict_fields": ["model_state_dict"],
        "selector_binding": "single_available_state_dict",
        "state_dict_field": "model_state_dict",
        "tensor_count": 2,
        "finite_tensor_count": 2,
        "nonfinite_keys": [],
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "reconstruction_messages": [],
        "adapter": {
            "type": "none",
            "objective_type": None,
            "configuration_source": None,
            "state_key_count": 0,
            "mounted_key_count": 0,
            "mounted": False,
        },
        "output_capability": (
            _pixel_output_contract(sha, registry)["capability"]
            if valid
            else None
        ),
        "output_contract": (
            _pixel_output_contract(sha, registry)
            if valid
            else None
        ),
        "smoke": {"requested_sample_count": 0, "executed_sample_count": 0, "output_shape": None},
        "failure_code": None if valid else "strict_load_failed",
        "failure_message": None if valid else "cannot reconstruct",
    }


def _complete_plan(tmp_path: Path, rows: list[dict]) -> tuple[dict, dict, Path]:
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, rows)
    policy, _, _ = _policy(tmp_path, ledger)
    result_root = tmp_path / "results"
    pending = build_checkpoint_plan(tmp_path, policy, result_root)
    for request in pending["preflight_requests"]:
        strict = _strict_preflight(
            request["checkpoint_sha256"],
            request["checkpoint_model"],
            policy["output_decoder_registry"],
        )
        envelope = build_preflight_result(request, policy, strict)
        write_exclusive_json(
            result_root
            / f"{request['checkpoint_sha256']}__{request['checkpoint_model']}.json",
            envelope,
        )
    return build_checkpoint_plan(tmp_path, policy, result_root), policy, result_root


def _manifest_fixture(tmp_path: Path) -> tuple[dict, dict, Path, Path, dict, dict]:
    checkpoint = tmp_path / "artifacts" / "candidate.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"canonical-screening-fixture-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    plan, policy, result_root = _complete_plan(
        tmp_path,
        [
            _row(
                "candidate",
                checkpoint_sha256,
                path=str(checkpoint.resolve()),
            )
        ],
    )
    plan_path = tmp_path / "plan.json"
    write_exclusive_json(plan_path, plan)
    manifest = build_candidate_manifest(
        policy,
        plan,
        plan_path=plan_path,
        repo_root=tmp_path,
        preflight_root=result_root,
    )
    manifest_path = tmp_path / "manifest.json"
    write_exclusive_json(manifest_path, manifest)
    policy_path = tmp_path / "policy.json"
    admission_value = {
        "contract_type": "safa_canonical_resource_admission_v1",
        "policy_sha256": policy["policy_sha256"],
        "snapshot": _admission_snapshot(policy),
    }
    admission_value["admission_sha256"] = canonical_digest(
        admission_value, "admission_sha256"
    )
    admission_path = tmp_path / "admission2.json"
    write_exclusive_json(admission_path, admission_value)
    admission = {
        "path": str(admission_path.resolve()),
        "sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "canonical_sha256": admission_value["admission_sha256"],
    }
    return policy, manifest, manifest_path, policy_path, admission, plan


def test_plan_counts_real_reference_semantics_and_dedup(tmp_path: Path) -> None:
    sha = "4" * 64
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        ledger,
        [
            _row("raw_a", sha, path="artifacts/a.pt"),
            _row("raw_b", sha, path="artifacts/b.pt"),
            _row("config", None),
        ],
    )
    policy, _, _ = _policy(tmp_path, ledger)
    plan = build_checkpoint_plan(tmp_path, policy, tmp_path / "results")
    counts = plan["counts"]
    assert counts["checkpoint_references"] == 2
    assert counts["raw_checkpoint_references"] == 2
    assert counts["ema_checkpoint_references"] == 0
    assert counts["distinct_checkpoint_sha256"] == 1
    assert counts["distinct_raw_checkpoint_sha256"] == 1
    assert counts["distinct_ema_checkpoint_sha256"] == 0
    assert counts["duplicate_checkpoint_references"] == 1
    assert counts["selector_conflicts"] == 0


def test_old_unbound_preflight_result_is_rejected(tmp_path: Path) -> None:
    sha = "5" * 64
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", sha)])
    policy, _, _ = _policy(tmp_path, ledger)
    results = tmp_path / "results"
    write_exclusive_json(
        results / f"{sha}__raw.json",
        _strict_preflight(sha, "raw", policy["output_decoder_registry"]),
    )
    with pytest.raises(CanonicalScreeningError, match="fields differ"):
        build_checkpoint_plan(tmp_path, policy, results)


def test_preflight_result_binds_request_policy_ledger_and_tool(tmp_path: Path) -> None:
    sha = "6" * 64
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", sha)])
    policy, _, _ = _policy(tmp_path, ledger)
    pending = build_checkpoint_plan(tmp_path, policy, tmp_path / "results")
    request = pending["preflight_requests"][0]
    envelope = build_preflight_result(
        request,
        policy,
        _strict_preflight(
            sha,
            "raw",
            policy["output_decoder_registry"],
        ),
    )
    assert validate_preflight_result(envelope, request, policy)[0] is True
    tampered = json.loads(json.dumps(envelope))
    tampered["policy_sha256"] = "7" * 64
    tampered["preflight_result_sha256"] = canonical_digest(
        tampered, "preflight_result_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="binding mismatch"):
        validate_preflight_result(tampered, request, policy)


@pytest.mark.parametrize("mutation", ("digest", "drop", "count", "policy"))
def test_plan_validator_rederives_and_rejects_tamper(tmp_path: Path, mutation: str) -> None:
    plan, policy, result_root = _complete_plan(
        tmp_path, [_row("candidate", "9" * 64)]
    )
    changed = json.loads(json.dumps(plan))
    if mutation == "digest":
        changed["checkpoint_plan_sha256"] = "a" * 64
    elif mutation == "drop":
        changed["eligible"] = []
        changed["checkpoint_plan_sha256"] = canonical_digest(
            changed, "checkpoint_plan_sha256"
        )
    elif mutation == "count":
        changed["counts"]["eligible_candidates"] = 2
        changed["checkpoint_plan_sha256"] = canonical_digest(
            changed, "checkpoint_plan_sha256"
        )
    else:
        changed["policy_sha256"] = "b" * 64
        changed["checkpoint_plan_sha256"] = canonical_digest(
            changed, "checkpoint_plan_sha256"
        )
    with pytest.raises(CanonicalScreeningError):
        validate_checkpoint_plan(
            changed,
            repo_root=tmp_path,
            policy=policy,
            preflight_root=result_root,
        )


def test_candidate_manifest_exactly_binds_validated_plan(tmp_path: Path) -> None:
    policy, manifest, manifest_path, _, _, plan = _manifest_fixture(tmp_path)
    result_root = tmp_path / "results"
    actual_plan_path = Path(manifest["checkpoint_plan"]["path"])
    assert validate_candidate_manifest(
        manifest,
        policy=policy,
        plan=plan,
        plan_path=actual_plan_path,
        repo_root=tmp_path,
        preflight_root=result_root,
    ) == manifest
    changed = json.loads(json.dumps(manifest))
    changed["candidate_count"] = 0
    changed["candidate_manifest_sha256"] = canonical_digest(
        changed, "candidate_manifest_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="differs"):
        validate_candidate_manifest(
            changed,
            policy=policy,
            plan=plan,
            plan_path=actual_plan_path,
            repo_root=tmp_path,
            preflight_root=result_root,
        )
    assert manifest_path.is_file()


def _run_fixture(tmp_path: Path, mode: str = "smoke8", replicate: str = "primary"):
    policy, manifest, manifest_path, policy_path, admission, _ = _manifest_fixture(tmp_path)
    candidate = manifest["candidates"][0]
    controller_ready, observer_ready = _ready_bindings(
        tmp_path, policy, admission, mode
    )
    request = build_run_request(
        policy,
        policy_path,
        manifest,
        manifest_path,
        candidate,
        mode,
        replicate,
        tmp_path / "runs",
        admission,
        controller_ready,
        observer_ready,
    )
    return policy, request


def _real_policy_run_fixture(
    tmp_path: Path,
    module,
) -> tuple[dict, Path, dict, Path]:
    repo_root = Path(__file__).parents[1].resolve()
    config = (
        repo_root
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    policy = validate_policy(
        repo_root,
        config,
        verify_historical_output_evidence=False,
    )
    preflight_root = tmp_path / "real-policy-preflight"
    pending = build_checkpoint_plan(repo_root, policy, preflight_root)
    available = []
    for request in pending["preflight_requests"]:
        raw_path = Path(str(request["checkpoint_path"]))
        checkpoint_path = (
            raw_path if raw_path.is_absolute() else repo_root / raw_path
        ).resolve()
        if checkpoint_path.is_file():
            available.append(
                (checkpoint_path.stat().st_size, checkpoint_path, request)
            )
    for _, checkpoint_path, preflight_request in sorted(
        available, key=lambda item: (item[0], str(item[1]))
    ):
        if sha256_file(checkpoint_path) == preflight_request[
            "checkpoint_sha256"
        ]:
            break
    else:
        raise AssertionError(
            "real policy has no SHA-exact checkpoint for CPU integration"
        )
    selected_sha256 = preflight_request["checkpoint_sha256"]
    for item in pending["preflight_requests"]:
        selected = item["checkpoint_sha256"] == selected_sha256
        strict = _strict_preflight(
            item["checkpoint_sha256"],
            item["checkpoint_model"],
            policy["output_decoder_registry"],
            status="valid" if selected else "invalid",
        )
        strict["checkpoint_path"] = (
            str(checkpoint_path)
            if selected
            else str(item["checkpoint_path"])
        )
        preflight_result = build_preflight_result(
            item, policy, strict
        )
        write_exclusive_json(
            preflight_root
            / (
                f"{item['checkpoint_sha256']}__"
                f"{item['checkpoint_model']}.json"
            ),
            preflight_result,
        )
    plan = build_checkpoint_plan(repo_root, policy, preflight_root)
    assert plan["counts"]["eligible_candidates"] == 1
    plan_path = tmp_path / "real-policy-plan.json"
    write_exclusive_json(plan_path, plan)
    manifest = build_candidate_manifest(
        policy,
        plan,
        plan_path=plan_path,
        repo_root=repo_root,
        preflight_root=preflight_root,
    )
    manifest_path = tmp_path / "real-policy-manifest.json"
    write_exclusive_json(manifest_path, manifest)
    admission_value = {
        "contract_type": "safa_canonical_resource_admission_v1",
        "policy_sha256": policy["policy_sha256"],
        "snapshot": _admission_snapshot(policy),
    }
    admission_value["admission_sha256"] = canonical_digest(
        admission_value, "admission_sha256"
    )
    admission_path = tmp_path / "real-policy-admission.json"
    write_exclusive_json(admission_path, admission_value)
    admission = {
        **_bound(admission_path),
        "canonical_sha256": admission_value["admission_sha256"],
    }
    controller_ready, observer_ready = _production_ready_bindings(
        tmp_path, module, policy, manifest, admission
    )
    request = build_run_request(
        policy,
        config,
        manifest,
        manifest_path,
        manifest["candidates"][0],
        "smoke8",
        "primary",
        tmp_path / "real-policy-runs",
        admission,
        controller_ready,
        observer_ready,
    )
    request_path = tmp_path / "real-policy-run-request.json"
    write_exclusive_json(request_path, request)
    return policy, config, request, request_path


def _production_ready_bindings(
    tmp_path: Path,
    module,
    policy: dict,
    manifest: dict,
    admission: dict,
) -> tuple[dict, dict]:
    paths = module._paths(
        tmp_path / "real-ready-campaign", policy["policy_sha256"]
    )
    wrapper, observer_launch = _wrapper_bindings(
        tmp_path, policy, "smoke8"
    )
    claim, claim_path = module._write_gpu_controller_claim(
        policy, paths, "smoke8", wrapper, observer_launch
    )
    intent, intent_path = module._write_request_intent_manifest(
        policy,
        paths,
        "smoke8",
        ("primary", "repeat"),
        manifest,
        admission,
    )

    def artifact(name: str, digest_field: str) -> tuple[dict, Path]:
        value = {
            "kind": name,
            "policy_sha256": policy["policy_sha256"],
        }
        value[digest_field] = canonical_digest(value, digest_field)
        path = tmp_path / "real-ready-artifacts" / f"{name}.json"
        write_exclusive_json(path, value)
        return value, path

    internal, internal_path = artifact(
        "internal", "monitor_sample_sha256"
    )
    first_guard, first_guard_path = artifact(
        "runtime_guard", "resource_window_sha256"
    )
    recheck, recheck_path = artifact(
        "resource_recheck", "resource_recheck_sha256"
    )
    controller, _, controller_binding = module._write_controller_ready(
        policy,
        paths,
        "smoke8",
        claim,
        admission,
        intent,
        intent_path,
        internal,
        internal_path,
        first_guard,
        first_guard_path,
        recheck,
        recheck_path,
        claim_path,
    )
    observer_claim, observer_claim_path = artifact(
        "observer_claim", "observer_claim_sha256"
    )
    observer_sample, observer_sample_path = artifact(
        "observer_sample", "monitor_sample_sha256"
    )
    observer = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_ready_v1",
        "campaign_id": policy["campaign_id"],
        "phase": "smoke8",
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "controller_ready_sha256": controller[
            "controller_ready_sha256"
        ],
        "observer_claim_sha256": observer_claim[
            "observer_claim_sha256"
        ],
        "wrapper_claim_sha256": wrapper["canonical_sha256"],
        "observer_launch_sha256": observer_launch[
            "canonical_sha256"
        ],
        "observer_claim": module._artifact_binding(
            observer_claim_path,
            observer_claim["observer_claim_sha256"],
        ),
        "wrapper_claim": wrapper,
        "observer_launch": observer_launch,
        "controller_ready": controller_binding,
        "admission": admission,
        "first_observer_sample": module._artifact_binding(
            observer_sample_path,
            observer_sample["monitor_sample_sha256"],
        ),
    }
    observer["observer_ready_sha256"] = canonical_digest(
        observer, "observer_ready_sha256"
    )
    observer_path = (
        paths["gpu_control"] / "smoke8" / "observer_ready.json"
    )
    write_exclusive_json(observer_path, observer)
    observer_binding = module._artifact_binding(
        observer_path, observer["observer_ready_sha256"]
    )
    module._validate_observer_ready(
        observer, policy, "smoke8", controller, admission
    )
    return controller_binding, observer_binding


def _final_release_for_single_request(
    tmp_path: Path,
    policy: dict,
    request: dict,
    request_path: Path,
) -> dict:
    controller_ready = load_json(
        Path(request["controller_ready"]["path"]),
        "real pre-CUDA controller ready",
    )
    release = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_final_release_admission_v1",
        "campaign_id": policy["campaign_id"],
        "phase": request["mode"],
        "policy_sha256": policy["policy_sha256"],
        "initial_admission_sha256": request["admission"][
            "canonical_sha256"
        ],
        "controller_ready_sha256": request["controller_ready"][
            "canonical_sha256"
        ],
        "observer_ready_sha256": request["observer_ready"][
            "canonical_sha256"
        ],
        "wrapper_claim": controller_ready["wrapper_claim"],
        "wrapper_claim_sha256": controller_ready[
            "wrapper_claim_sha256"
        ],
        "observer_launch": controller_ready["observer_launch"],
        "observer_launch_sha256": controller_ready[
            "observer_launch_sha256"
        ],
        "authorized_gpu_registry": request[
            "authorized_gpu_registry"
        ],
        "request_count": 1,
        "requests": [
            {
                **_bound(request_path),
                "canonical_sha256": request["run_request_sha256"],
            }
        ],
        "snapshot": {
            "authorized_gpu_registry": request[
                "authorized_gpu_registry"
            ],
            "compute_processes": [],
        },
        "released_at": "2026-07-27T00:00:00+00:00",
    }
    release["final_release_admission_sha256"] = canonical_digest(
        release, "final_release_admission_sha256"
    )
    release_path = tmp_path / "real-policy-final-release.json"
    write_exclusive_json(release_path, release)
    return {
        **_bound(release_path),
        "canonical_sha256": release[
            "final_release_admission_sha256"
        ],
    }


def _ready_bindings(
    tmp_path: Path, policy: dict, admission: dict, mode: str
) -> tuple[dict, dict]:
    wrapper_claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_wrapper_claim_v1",
        "policy_sha256": policy["policy_sha256"],
        "phase": mode,
    }
    wrapper_claim["wrapper_claim_sha256"] = canonical_digest(
        wrapper_claim, "wrapper_claim_sha256"
    )
    wrapper_claim_path = tmp_path / "ready" / mode / "wrapper_claim.json"
    write_exclusive_json(wrapper_claim_path, wrapper_claim)
    wrapper_binding = {
        **_bound(wrapper_claim_path),
        "canonical_sha256": wrapper_claim["wrapper_claim_sha256"],
    }
    observer_launch = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_launch_v2",
        "policy_sha256": policy["policy_sha256"],
        "phase": mode,
        "status": "launched",
        "failure": None,
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": wrapper_claim["wrapper_claim_sha256"],
    }
    observer_launch["observer_launch_sha256"] = canonical_digest(
        observer_launch, "observer_launch_sha256"
    )
    observer_launch_path = (
        tmp_path / "ready" / mode / "observer_launch.json"
    )
    write_exclusive_json(observer_launch_path, observer_launch)
    observer_launch_binding = {
        **_bound(observer_launch_path),
        "canonical_sha256": observer_launch["observer_launch_sha256"],
    }
    controller_claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_controller_claim_v1",
        "campaign_id": policy["campaign_id"],
        "phase": mode,
        "policy_sha256": policy["policy_sha256"],
        "wrapper_claim": wrapper_binding,
        "observer_launch": observer_launch_binding,
        "controller_pid": 77,
        "started_at": "2026-07-27T00:00:00+00:00",
    }
    controller_claim["controller_claim_sha256"] = canonical_digest(
        controller_claim, "controller_claim_sha256"
    )
    controller_claim_path = (
        tmp_path / "ready" / mode / "controller_claim.json"
    )
    write_exclusive_json(controller_claim_path, controller_claim)
    controller_claim_binding = {
        **_bound(controller_claim_path),
        "canonical_sha256": controller_claim[
            "controller_claim_sha256"
        ],
    }
    controller = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_controller_ready_v1",
        "campaign_id": policy["campaign_id"],
        "phase": mode,
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "controller_claim": controller_claim_binding,
        "controller_claim_sha256": controller_claim[
            "controller_claim_sha256"
        ],
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": wrapper_claim["wrapper_claim_sha256"],
        "observer_launch": observer_launch_binding,
        "observer_launch_sha256": observer_launch[
            "observer_launch_sha256"
        ],
    }
    controller["controller_ready_sha256"] = canonical_digest(
        controller, "controller_ready_sha256"
    )
    controller_path = tmp_path / "ready" / mode / "controller_ready.json"
    write_exclusive_json(controller_path, controller)
    controller_binding = {
        **_bound(controller_path),
        "canonical_sha256": controller["controller_ready_sha256"],
    }
    observer = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_ready_v1",
        "campaign_id": policy["campaign_id"],
        "phase": mode,
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "controller_ready_sha256": controller["controller_ready_sha256"],
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": wrapper_claim["wrapper_claim_sha256"],
        "observer_launch": observer_launch_binding,
        "observer_launch_sha256": observer_launch[
            "observer_launch_sha256"
        ],
    }
    observer["observer_ready_sha256"] = canonical_digest(
        observer, "observer_ready_sha256"
    )
    observer_path = tmp_path / "ready" / mode / "observer_ready.json"
    write_exclusive_json(observer_path, observer)
    observer_binding = {
        **_bound(observer_path),
        "canonical_sha256": observer["observer_ready_sha256"],
    }
    return controller_binding, observer_binding


def _wrapper_bindings(
    tmp_path: Path, policy: dict, mode: str
) -> tuple[dict, dict]:
    wrapper = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_wrapper_claim_v1",
        "policy_sha256": policy["policy_sha256"],
        "phase": mode,
    }
    wrapper["wrapper_claim_sha256"] = canonical_digest(
        wrapper, "wrapper_claim_sha256"
    )
    wrapper_path = tmp_path / "wrapper" / mode / "wrapper_claim.json"
    write_exclusive_json(wrapper_path, wrapper)
    wrapper_binding = {
        **_bound(wrapper_path),
        "canonical_sha256": wrapper["wrapper_claim_sha256"],
    }
    launch = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_launch_v2",
        "policy_sha256": policy["policy_sha256"],
        "phase": mode,
        "status": "launched",
        "failure": None,
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": wrapper["wrapper_claim_sha256"],
        "command": ["controller", "--monitor-target", mode, "--execute"],
    }
    launch["observer_launch_sha256"] = canonical_digest(
        launch, "observer_launch_sha256"
    )
    launch_path = tmp_path / "wrapper" / mode / "observer_launch.json"
    write_exclusive_json(launch_path, launch)
    return wrapper_binding, {
        **_bound(launch_path),
        "canonical_sha256": launch["observer_launch_sha256"],
    }


def _mock_controller_claim(path: Path, claim: dict) -> tuple[dict, Path]:
    write_exclusive_json(path, claim)
    return claim, path


def _evidence(policy: dict, request: dict) -> dict:
    return {
        "mode": request["mode"],
        "replicate": request["replicate"],
        "seed": 4549,
        "batch_size": 2,
        "sample_count": request["sample_count"],
        "sample_manifest_sha256": request["sample_manifest"]["sha256"],
        "candidate_manifest_sha256": request["candidate_manifest"]["canonical_sha256"],
        "policy_sha256": policy["policy_sha256"],
        "implementations": policy["implementations"],
        "checkpoint_sha256": request["candidate"]["checkpoint_sha256"],
        "checkpoint_model": request["candidate"]["checkpoint_model"],
        "output_contract_sha256": request["output_contract"][
            "output_contract_sha256"
        ],
        "output_contract_type": request["output_contract"]["contract_type"],
        "decoder_registry_sha256": request["output_decoder_registry"][
            "decoder_registry_sha256"
        ],
        "output_space": request["output_contract"]["capability"]["output_space"],
        "native_rgb_size": request["native_rgb_size"],
        "quality_protocol_family": request["quality_protocol_family"],
        "nfe": request["nfe"],
        "pixel_image_size": 256,
        "pixel_protocol_config_sha256": policy["protocol"]["pixel_protocol_config"]["sha256"],
        "kid_subset_size": policy["protocol"]["kid_subset_sizes"][request["mode"]],
        "e0_mean": 0.8,
        "edev_mean": 0.7,
        "arcface": {"coverage": request["sample_count"]},
        "quality": {"kid_mean": 0.01},
        "per_sample_sha256": "c" * 64,
    }


def _handshake_fixture(
    policy: dict,
    request: dict,
    gpu_index: int = 0,
    worker_pid: int = 123,
) -> dict:
    gpu_uuid = request["authorized_gpu_registry"][gpu_index][
        "physical_gpu_uuid"
    ]
    request_path = (
        Path(request["policy"]["path"]).parent
        / "release_requests"
        / f"{request['mode']}-{request['replicate']}.json"
    )
    write_exclusive_json(request_path, request)
    release = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_final_release_admission_v1",
        "campaign_id": policy["campaign_id"],
        "phase": request["mode"],
        "policy_sha256": policy["policy_sha256"],
        "initial_admission_sha256": request["admission"][
            "canonical_sha256"
        ],
        "controller_ready_sha256": request["controller_ready"][
            "canonical_sha256"
        ],
        "observer_ready_sha256": request["observer_ready"][
            "canonical_sha256"
        ],
        "wrapper_claim": load_json(
            Path(request["controller_ready"]["path"]), "controller ready"
        )["wrapper_claim"],
        "wrapper_claim_sha256": load_json(
            Path(request["controller_ready"]["path"]), "controller ready"
        )["wrapper_claim_sha256"],
        "observer_launch": load_json(
            Path(request["controller_ready"]["path"]), "controller ready"
        )["observer_launch"],
        "observer_launch_sha256": load_json(
            Path(request["controller_ready"]["path"]), "controller ready"
        )["observer_launch_sha256"],
        "authorized_gpu_registry": request["authorized_gpu_registry"],
        "request_count": 1,
        "requests": [
            {
                **_bound(request_path),
                "canonical_sha256": request["run_request_sha256"],
            }
        ],
        "snapshot": {
            "authorized_gpu_registry": request["authorized_gpu_registry"],
            "compute_processes": [],
        },
        "released_at": "2026-07-26T00:00:00+00:00",
    }
    release["final_release_admission_sha256"] = canonical_digest(
        release, "final_release_admission_sha256"
    )
    release_path = (
        Path(request["policy"]["path"]).parent
        / "release_requests"
        / f"{request['mode']}-{request['replicate']}-release.json"
    )
    write_exclusive_json(release_path, release)
    release_binding = {
        **_bound(release_path),
        "canonical_sha256": release["final_release_admission_sha256"],
    }
    manifest = load_json(
        Path(request["candidate_manifest"]["path"]),
        "handshake candidate manifest",
    )
    checkpoint_path = Path(request["candidate"]["checkpoint_path"]).resolve()
    rehashed_bindings = {
        "config": {
            "path": str(Path(request["policy"]["path"]).resolve()),
            "sha256": request["policy"]["sha256"],
        },
        "implementations": {
            name: {
                "path": str(Path(binding["path"]).resolve()),
                "sha256": binding["sha256"],
            }
            for name, binding in request["implementations"].items()
        },
        "request": {
            "path": str(request_path.resolve()),
            "sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "canonical_sha256": request["run_request_sha256"],
        },
        "candidate_manifest": dict(request["candidate_manifest"]),
        "checkpoint_plan": dict(manifest["checkpoint_plan"]),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
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
        "final_release": dict(release_binding),
        "controller_ready": dict(request["controller_ready"]),
        "observer_ready": dict(request["observer_ready"]),
    }
    controller_ready = load_json(
        Path(request["controller_ready"]["path"]),
        "handshake controller ready",
    )
    worker_ready = {
        "schema_version": 1,
        "contract_type": WORKER_READY_CONTRACT,
        "policy_sha256": policy["policy_sha256"],
        "phase": request["mode"],
        "worker_pid": worker_pid,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "run_request_sha256": request["run_request_sha256"],
        "request": rehashed_bindings["request"],
        "final_release": release_binding,
        "verification_order": list(WORKER_PRE_CUDA_VERIFICATION_ORDER),
        "rehashed_bindings": rehashed_bindings,
        "rehashed_bindings_sha256": hashlib.sha256(
            canonical_json(rehashed_bindings)
        ).hexdigest(),
        "controller_claim": controller_ready["controller_claim"],
        "screening_worker_sha256": request["implementations"][
            "screening_worker"
        ]["sha256"],
        "controller_implementation_sha256": request["implementations"][
            "controller"
        ]["sha256"],
        "cuda_visible_devices": gpu_uuid,
        "heavy_modules_absent": True,
        "loaded_heavy_modules": [],
        "asset_content_verification": _asset_content_verification(policy),
        "external_gpu_race_contract": WORKER_EXTERNAL_GPU_RACE_CONTRACT,
        "ready_at": "2026-07-27T00:00:01+00:00",
    }
    worker_ready["worker_ready_sha256"] = canonical_digest(
        worker_ready, "worker_ready_sha256"
    )
    worker_ready_path = (
        Path(request["policy"]["path"]).parent
        / "release_requests"
        / f"{request['mode']}-{request['replicate']}-worker-ready.json"
    )
    write_exclusive_json(worker_ready_path, worker_ready)
    worker_ready_binding = {
        **_bound(worker_ready_path),
        "canonical_sha256": worker_ready["worker_ready_sha256"],
    }
    gpus = [
        {
            "index": row["physical_gpu_index"],
            "uuid": row["physical_gpu_uuid"],
            "memory_total_mib": 24576,
            "memory_used_mib": 3,
            "memory_free_mib": 24573,
            "temperature_c": 35,
        }
        for row in request["authorized_gpu_registry"]
    ]
    controller_resource_snapshot = {
        "observed_at": "2026-07-27T00:00:02+00:00",
        "cpu_load_percent": 1.0,
        "memory_percent": 2.0,
        "disk_percent": 3.0,
        "swap_pages": {"in": 0, "out": 0},
        "gpus": gpus,
        "authorized_gpu_registry": request["authorized_gpu_registry"],
        "ram_reservation": {"validated": True},
        "compute_processes": [],
    }
    controller_rehash = {
        "schema_version": 1,
        "contract_type": CONTROLLER_LAUNCH_REHASH_CONTRACT,
        "policy_sha256": policy["policy_sha256"],
        "run_request_sha256": request["run_request_sha256"],
        "worker_pid": worker_pid,
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "worker_ready": worker_ready_binding,
        "verification_order": list(WORKER_PRE_CUDA_VERIFICATION_ORDER),
        "rehashed_bindings": rehashed_bindings,
        "rehashed_bindings_sha256": worker_ready[
            "rehashed_bindings_sha256"
        ],
        "resource_snapshot": controller_resource_snapshot,
        "asset_content_verification": _asset_content_verification(policy),
        "external_gpu_race_contract": WORKER_EXTERNAL_GPU_RACE_CONTRACT,
        "validated_at": "2026-07-27T00:00:02+00:00",
    }
    controller_rehash["controller_launch_rehash_sha256"] = canonical_digest(
        controller_rehash, "controller_launch_rehash_sha256"
    )
    controller_rehash_path = (
        Path(request["policy"]["path"]).parent
        / "release_requests"
        / f"{request['mode']}-{request['replicate']}-controller-rehash.json"
    )
    write_exclusive_json(controller_rehash_path, controller_rehash)
    controller_rehash_binding = {
        **_bound(controller_rehash_path),
        "canonical_sha256": controller_rehash[
            "controller_launch_rehash_sha256"
        ],
    }
    worker_release = {
        "schema_version": 1,
        "contract_type": WORKER_RELEASE_CONTRACT,
        "policy_sha256": policy["policy_sha256"],
        "phase": request["mode"],
        "worker_pid": worker_pid,
        "run_request_sha256": request["run_request_sha256"],
        "worker_ready": worker_ready_binding,
        "controller_launch_rehash": controller_rehash_binding,
        "resource_snapshot": {
            "admission": controller_resource_snapshot,
            "runtime_guard": {
                "schema_version": 1,
                "contract_type":
                    "safa_canonical_worker_release_resource_snapshot_v2",
                "policy_sha256": policy["policy_sha256"],
                "observed_at": "2026-07-27T00:00:03+00:00",
                "runtime_gpu_registry": request[
                    "authorized_gpu_registry"
                ],
                "compute_processes": [],
                "unknown_compute_processes": [],
                "cpu_load_percent": 1.0,
                "memory_percent": 2.0,
                "disk_percent": 3.0,
                "swap_pages_before": {"in": 0, "out": 0},
                "swap_pages_after": {"in": 0, "out": 0},
                "swap_io_delta": {"in": 0, "out": 0},
                "swap_consecutive_io": 0,
                "gpu": gpus,
                "active_worker_pids": [worker_pid],
                "hard_limits": {
                    "cpu_percent": 90,
                    "ram_percent": 90,
                    "disk_percent": 90,
                    "gpu_memory_percent": 90.0,
                    "gpu_temperature_c": 85,
                    "gpu_free_mib": 2048,
                    "swap_io_delta_pages": 0,
                    "swap_consecutive_io": 0,
                },
                "guard_thread_failure": None,
                "guard_violation_reason": None,
            },
        },
        "external_gpu_race_contract": WORKER_EXTERNAL_GPU_RACE_CONTRACT,
        "released_at": "2026-07-27T00:00:03+00:00",
    }
    worker_release["worker_release_sha256"] = canonical_digest(
        worker_release, "worker_release_sha256"
    )
    worker_release_path = (
        Path(request["policy"]["path"]).parent
        / "release_requests"
        / f"{request['mode']}-{request['replicate']}-worker-release.json"
    )
    write_exclusive_json(worker_release_path, worker_release)
    worker_release_binding = {
        **_bound(worker_release_path),
        "canonical_sha256": worker_release["worker_release_sha256"],
    }
    validate_worker_ready_value(
        worker_ready,
        request,
        policy,
        expected_worker_pid=worker_pid,
        expected_gpu_index=gpu_index,
        expected_gpu_uuid=gpu_uuid,
    )
    validate_controller_launch_rehash_value(
        controller_rehash, request, policy
    )
    validate_worker_release_value(
        worker_release,
        request,
        policy,
        expected_worker_pid=worker_pid,
    )
    return {
        "final_release": release_binding,
        "worker_ready": worker_ready,
        "worker_ready_binding": worker_ready_binding,
        "controller_rehash": controller_rehash,
        "controller_rehash_binding": controller_rehash_binding,
        "worker_release": worker_release,
        "worker_release_binding": worker_release_binding,
    }


def _run_claim(
    policy: dict, request: dict, gpu_index: int = 0, worker_pid: int = 123
) -> dict:
    gpu_uuid = request["authorized_gpu_registry"][gpu_index][
        "physical_gpu_uuid"
    ]
    handshake = _handshake_fixture(
        policy,
        request,
        gpu_index=gpu_index,
        worker_pid=worker_pid,
    )
    return build_run_claim(
        request,
        policy,
        handshake["final_release"],
        handshake["worker_ready_binding"],
        handshake["worker_release_binding"],
        gpu_index,
        gpu_uuid,
        gpu_uuid,
        gpu_uuid,
        worker_pid,
        "2026-07-26T00:00:00+00:00",
    )


def _completed_worker_terminal_fixture(
    policy: dict, request: dict, worker_pid: int = 123
) -> dict:
    handshake = _handshake_fixture(
        policy, request, worker_pid=worker_pid
    )
    gpu_uuid = request["authorized_gpu_registry"][0][
        "physical_gpu_uuid"
    ]
    claim = build_run_claim(
        request,
        policy,
        handshake["final_release"],
        handshake["worker_ready_binding"],
        handshake["worker_release_binding"],
        0,
        gpu_uuid,
        gpu_uuid,
        gpu_uuid,
        worker_pid,
        "2026-07-26T00:00:00+00:00",
    )
    output_dir = Path(request["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_path = output_dir / "claim.json"
    write_exclusive_json(claim_path, claim)
    result = build_run_result(
        request,
        claim,
        policy,
        status="completed",
        completed_at="2026-07-26T00:01:00+00:00",
        evidence=_evidence(policy, request),
    )
    result_path = output_dir / "result.json"
    write_exclusive_json(result_path, result)
    request_path = Path(
        handshake["worker_ready"]["request"]["path"]
    ).resolve()
    terminal = {
        "schema_version": 1,
        "contract_type": "safa_canonical_worker_terminal_v1",
        "policy_sha256": policy["policy_sha256"],
        "worker_pid": worker_pid,
        "request": {
            **_bound(request_path),
            "canonical_sha256": request["run_request_sha256"],
        },
        "claim": {
            **_bound(claim_path),
            "canonical_sha256": claim["run_claim_sha256"],
        },
        "result": {
            **_bound(result_path),
            "canonical_sha256": result["run_result_sha256"],
        },
        "worker_ready": handshake["worker_ready_binding"],
        "worker_release": handshake["worker_release_binding"],
        "status": "completed",
        "failure": None,
        "started_at": "2026-07-26T00:00:00+00:00",
        "completed_at": "2026-07-26T00:01:00+00:00",
    }
    terminal["worker_terminal_sha256"] = canonical_digest(
        terminal, "worker_terminal_sha256"
    )
    terminal_path = (
        Path(handshake["worker_ready_binding"]["path"]).parent
        / "worker_terminal.json"
    )
    write_exclusive_json(terminal_path, terminal)
    return {
        "request_path": request_path,
        "claim_path": claim_path,
        "result_path": result_path,
        "terminal_path": terminal_path,
        "claim": claim,
        "result": result,
        "terminal": terminal,
    }


@pytest.mark.parametrize(
    "target",
    (
        "request",
        "claim",
        "result",
        "terminal",
        "missing_claim",
        "missing_result",
        "missing_terminal",
    ),
)
def test_completion_rejects_post_exit_artifact_tamper(
    tmp_path: Path, target: str
) -> None:
    module = _controller_module()
    policy, request = _run_fixture(tmp_path)
    artifacts = _completed_worker_terminal_fixture(policy, request)
    validate_worker_terminal_value(
        artifacts["terminal"],
        artifacts["request_path"],
        policy,
        expected_worker_pid=123,
        require_completed=True,
    )
    if target == "request":
        value = load_json(artifacts["request_path"], "request")
        value["replicate"] = "post-exit-tamper"
        value["run_request_sha256"] = canonical_digest(
            value, "run_request_sha256"
        )
        artifacts["request_path"].write_bytes(canonical_json(value))
    elif target == "claim":
        value = load_json(artifacts["claim_path"], "claim")
        value["started_at"] = "2026-07-26T00:00:01+00:00"
        value["run_claim_sha256"] = canonical_digest(
            value, "run_claim_sha256"
        )
        artifacts["claim_path"].write_bytes(canonical_json(value))
    elif target == "result":
        value = load_json(artifacts["result_path"], "result")
        value["completed_at"] = "2026-07-26T00:01:01+00:00"
        value["run_result_sha256"] = canonical_digest(
            value, "run_result_sha256"
        )
        artifacts["result_path"].write_bytes(canonical_json(value))
    elif target == "terminal":
        value = load_json(artifacts["terminal_path"], "terminal")
        value["result"]["sha256"] = "f" * 64
        value["worker_terminal_sha256"] = canonical_digest(
            value, "worker_terminal_sha256"
        )
        artifacts["terminal_path"].write_bytes(canonical_json(value))
    else:
        artifacts[target.removeprefix("missing_") + "_path"].unlink()
    with pytest.raises((CanonicalScreeningError, FileNotFoundError)):
        module._build_gpu_completion_summary(
            policy,
            request["mode"],
            module._paths(tmp_path / "campaign", policy["policy_sha256"]),
            [artifacts["request_path"]],
            {},
            {},
            tmp_path / "monitor.jsonl",
            {},
        )


def test_run_request_rejects_stale_policy_and_wrong_kid_subset(tmp_path: Path) -> None:
    policy, request = _run_fixture(tmp_path)
    assert request["kid_subset_size"] == 8
    assert validate_run_request(request, policy) == request
    changed = json.loads(json.dumps(request))
    changed["kid_subset_size"] = 50
    changed["run_request_sha256"] = canonical_digest(changed, "run_request_sha256")
    with pytest.raises(CanonicalScreeningError, match="frozen"):
        validate_run_request(changed, policy)
    stale = json.loads(json.dumps(request))
    stale["policy"]["canonical_sha256"] = "d" * 64
    stale["run_request_sha256"] = canonical_digest(stale, "run_request_sha256")
    with pytest.raises(CanonicalScreeningError, match="policy binding"):
        validate_run_request(stale, policy)


def test_screen512_locks_kid_subset_50(tmp_path: Path) -> None:
    policy, request = _run_fixture(tmp_path, mode="screen512")
    assert request["kid_subset_size"] == 50
    assert validate_run_request(request, policy) == request


def test_run_result_binds_smoke_manifest_policy_tool_and_admission(tmp_path: Path) -> None:
    policy, request = _run_fixture(tmp_path)
    claim = _run_claim(policy, request, gpu_index=3)
    result = build_run_result(
        request,
        claim,
        policy,
        status="completed",
        completed_at="2026-07-26T00:01:00+00:00",
        evidence=_evidence(policy, request),
    )
    assert validate_run_result(result, request, claim, policy) == result
    changed = json.loads(json.dumps(result))
    changed["evidence"]["candidate_manifest_sha256"] = "e" * 64
    changed["run_result_sha256"] = canonical_digest(changed, "run_result_sha256")
    with pytest.raises(CanonicalScreeningError, match="candidate_manifest"):
        validate_run_result(changed, request, claim, policy)


def test_run_request_and_claim_reject_gpu_uuid_tampering(tmp_path: Path) -> None:
    policy, request = _run_fixture(tmp_path)
    changed_request = json.loads(json.dumps(request))
    changed_request["authorized_gpu_registry"][0]["physical_gpu_uuid"] = (
        "GPU-tampered"
    )
    changed_request["run_request_sha256"] = canonical_digest(
        changed_request, "run_request_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="GPU UUID registry"):
        validate_run_request(changed_request, policy)

    claim = _run_claim(policy, request)
    changed_claim = json.loads(json.dumps(claim))
    changed_claim["runtime_cuda_uuid"] = "GPU-tampered"
    changed_claim["run_claim_sha256"] = canonical_digest(
        changed_claim, "run_claim_sha256"
    )
    with pytest.raises(CanonicalScreeningError, match="CUDA/RAM"):
        build_run_result(
            request,
            changed_claim,
            policy,
            status="failed",
            completed_at="2026-07-26T00:01:00+00:00",
            failure={"type": "probe", "message": "probe"},
        )


def test_worker_cuda_binding_refuses_remap_and_runtime_uuid_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    policy, request = _run_fixture(tmp_path)
    expected_uuid = request["authorized_gpu_registry"][0]["physical_gpu_uuid"]
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(CanonicalScreeningError, match="CUDA_VISIBLE_DEVICES"):
        _assert_runtime_cuda_binding(request, 0, expected_uuid)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", expected_uuid)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: types.SimpleNamespace(uuid=_gpu_uuid(1)),
    )
    with pytest.raises(CanonicalScreeningError, match="runtime CUDA UUID"):
        _assert_runtime_cuda_binding(request, 0, expected_uuid)

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: types.SimpleNamespace(uuid=expected_uuid),
    )
    selected: list[int] = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    binding = _assert_runtime_cuda_binding(request, 0, expected_uuid)
    assert binding["physical_gpu_index"] == 0
    assert binding["logical_cuda_index"] == 0
    assert binding["physical_gpu_uuid"] == expected_uuid
    assert binding["runtime_cuda_uuid"] == expected_uuid
    assert binding["cuda_visible_devices"] == expected_uuid
    assert {
        binding["uuid_evidence"][name]["canonical"]
        for name in (
            "admission",
            "worker_argument",
            "cuda_visible_devices",
            "runtime_cuda_uuid",
        )
    } == {expected_uuid}
    assert selected == [0]


@pytest.mark.parametrize(
    "raw",
    [
        "GPU-7BA69FC7-12AC-3DFB-8265-3476CE2504B6",
        "7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
        b"GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
    ],
)
def test_gpu_uuid_canonicalizer_accepts_verified_representations(
    raw: str | bytes,
) -> None:
    assert canonicalize_nvidia_gpu_uuid(raw, "fixture")["canonical"] == (
        "GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6"
    )


@pytest.mark.parametrize(
    "raw",
    [
        " GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
        "GPU-7ba69fc7",
        "7ba69fc712ac3dfb82653476ce2504b6",
        "MIG-7ba69fc7-12ac-3dfb-8265-3476ce2504b6",
        b"\xff",
        object(),
    ],
)
def test_gpu_uuid_canonicalizer_rejects_malformed_values(raw: object) -> None:
    with pytest.raises(CanonicalScreeningError):
        canonicalize_nvidia_gpu_uuid(raw, "fixture")  # type: ignore[arg-type]


def test_gpu_registry_rejects_duplicate_canonical_uuid() -> None:
    with pytest.raises(CanonicalScreeningError, match="duplicate UUIDs"):
        canonical_gpu_registry(
            [
                {
                    "physical_gpu_index": 0,
                    "physical_gpu_uuid": _gpu_uuid(0),
                },
                {
                    "physical_gpu_index": 1,
                    "physical_gpu_uuid": _gpu_uuid(0).removeprefix("GPU-"),
                },
            ]
        )


def test_failed_probe_root_identity_rejects_root_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "clone"
    target.mkdir()
    declared = (
        tmp_path
        / "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__6b088236579f7311"
    )
    declared.parent.mkdir(parents=True)
    declared.symlink_to(target, target_is_directory=True)
    with pytest.raises(CanonicalScreeningError, match="root identity"):
        _validate_6b_failed_probe_root_identity(
            tmp_path,
            {
                "path": (
                    "artifacts/closeout/historical-canonical-512-v1/"
                    "ram_probe__6b088236579f7311"
                ),
                "digest": "0" * 64,
                "digest_algorithm": (
                    "sha256_relative_posix_nul_content_nul_v1"
                ),
            },
        )


def test_ram_probe_contract_and_execution_digests_are_split() -> None:
    module = _ram_probe_module()
    static = {
        "schema_version": 1,
        "contract_type": module.PROBE_CONTRACT,
        "sample_count": 8,
        "authorized_gpu_registry": None,
        "admission": None,
        "probe_contract_sha256": None,
        "probe_execution_sha256": None,
    }
    contract = module._probe_contract_digest(static)
    registry = [
        {
            "physical_gpu_index": 0,
            "physical_gpu_uuid": _gpu_uuid(0),
        }
    ]
    live = {
        **static,
        "authorized_gpu_registry": registry,
        "admission": {"path": "/bound", "sha256": "0" * 64},
        "probe_contract_sha256": contract,
        "probe_execution_sha256": "1" * 64,
    }
    assert module._probe_contract_digest(live) == contract
    admission = {
        "schema_version": 1,
        "contract_type": module.PROBE_ADMISSION_CONTRACT,
        "probe_contract_sha256": contract,
        "host": {"ram_used_percent": 20.0},
        "gpu_snapshot": [{"index": 0, "uuid": _gpu_uuid(0)}],
        "authorized_gpu_registry": registry,
        "observed_at": "2026-07-27T00:00:00+00:00",
    }
    evidence = module._admission_evidence_digest(admission)
    execution = module._probe_execution_digest(contract, registry, evidence)
    changed_registry = [{**registry[0], "physical_gpu_uuid": _gpu_uuid(1)}]
    assert (
        module._probe_execution_digest(contract, changed_registry, evidence)
        != execution
    )
    tampered = {
        **admission,
        "host": {"ram_used_percent": 21.0},
    }
    assert module._admission_evidence_digest(tampered) != evidence


def test_ram_probe_build_spec_separates_admission_binding_and_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _ram_probe_module()
    config = tmp_path / "policy.json"
    manifest_path = tmp_path / "manifest.json"
    smoke = tmp_path / "smoke.jsonl"
    for path in (config, manifest_path, smoke):
        path.write_text("{}\n", encoding="utf-8")
    policy = {
        "policy_sha256": "1" * 64,
        "protocol": {"manifests": {"smoke8": _bound(smoke)}},
        "implementations": {"worker": _bound(smoke)},
    }
    manifest = {"candidate_manifest_sha256": "2" * 64}
    monkeypatch.setattr(module, "_select_probe_candidates", lambda _value: [])
    dry = module._build_spec(
        policy,
        config,
        manifest,
        manifest_path,
        tmp_path / "probe",
        None,
    )
    binding = {
        "path": str((tmp_path / "admission.json").resolve()),
        "sha256": "3" * 64,
        "canonical_sha256": "4" * 64,
    }
    execution = "5" * 64
    live = module._build_spec(
        policy,
        config,
        manifest,
        manifest_path,
        tmp_path / "probe",
        [
            {
                "physical_gpu_index": 0,
                "physical_gpu_uuid": _gpu_uuid(0),
            }
        ],
        binding,
        execution,
    )
    assert live["admission"] == binding
    assert live["probe_execution_sha256"] == execution
    assert live["probe_contract_sha256"] == dry["probe_contract_sha256"]
    with pytest.raises(CanonicalScreeningError, match="must be paired"):
        module._build_spec(
            policy,
            config,
            manifest,
            manifest_path,
            tmp_path / "probe",
            [],
            binding,
            None,
        )
    with pytest.raises(CanonicalScreeningError, match="fields differ"):
        module._build_spec(
            policy,
            config,
            manifest,
            manifest_path,
            tmp_path / "probe",
            [],
            {**binding, "probe_execution_sha256": execution},
            execution,
        )


def test_ram_probe_controller_writes_one_preworker_failure_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _ram_probe_module()
    artifact_root = tmp_path / "probe"
    claim = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_screening_ram_probe_controller_claim_v1"
        ),
        "controller_claim_sha256": "6" * 64,
    }

    def fail_after_claim(*_args: object, **_kwargs: object) -> None:
        artifact_root.mkdir()
        write_exclusive_json(artifact_root / "controller_claim.json", claim)
        (artifact_root / "input_policy.json").write_text(
            "{}\n", encoding="utf-8"
        )
        write_exclusive_json(artifact_root / "admission.json", {})
        raise KeyError("probe_execution_sha256")

    monkeypatch.setattr(module, "_run_controller_once", fail_after_claim)
    with pytest.raises(KeyError, match="probe_execution_sha256"):
        module._run_controller({}, tmp_path / "p", {}, tmp_path / "m", artifact_root)
    terminal = load_json(
        artifact_root / "controller_terminal.json", "controller terminal"
    )
    assert terminal["status"] == "failed"
    assert terminal["stage"] == "admission_to_spec"
    assert terminal["exception"]["type"] == "KeyError"
    assert terminal["retry_count"] == 0
    assert terminal["worker_started"] is False
    assert not (artifact_root / "probe_result.json").exists()
    before = (artifact_root / "controller_terminal.json").read_bytes()
    with pytest.raises(FileExistsError):
        module._write_controller_failure_terminal(
            artifact_root, RuntimeError("collision")
        )
    assert (artifact_root / "controller_terminal.json").read_bytes() == before


def test_ram_probe_controller_positive_mock_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _ram_probe_module()
    root = Path(__file__).parents[1]
    config = root / "configs/closeout/canonical_screening_512_v1.json"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    artifact_root = tmp_path / "probe"
    policy = validate_policy(root, config)
    policy["resources"] = {
        key: value
        for key, value in policy["resources"].items()
        if key not in {"ram_slot_budget_bytes", "ram_slot_budget_source"}
    }
    policy["resources"]["ram_budget_status"] = "probe_required"
    manifest = {"candidate_manifest_sha256": "2" * 64}
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(module, "_select_probe_candidates", lambda _value: [])
    monkeypatch.setattr(
        module,
        "assert_cpu_resource_admission",
        lambda *_args: {"memory_percent": 20.0},
    )
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: [
            {
                "index": index,
                "uuid": _gpu_uuid(index),
                "memory_free_mib": 24_000,
            }
            for index in range(4)
        ],
    )
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    monkeypatch.setattr(module, "_worker_environment", lambda _uuid: {})

    class Guard:
        def __init__(self, *_args: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> dict:
            return {
                "violated": False,
                "violation_reason": None,
                "thread_failure": None,
            }

    class Process:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

    def monitor(
        _process: object, _policy: dict, _guard: object
    ) -> tuple[int, int, None, None]:
        spec = load_json(artifact_root / "probe_spec.json", "probe spec")
        worker = {
            "schema_version": 1,
            "contract_type": module.PROBE_WORKER_RESULT_CONTRACT,
            "status": "succeeded",
            "probe_contract_sha256": spec["probe_contract_sha256"],
            "probe_execution_sha256": spec["probe_execution_sha256"],
            "purpose": spec["purpose"],
            "device_binding": {"physical_gpu_index": 0},
            "steps": [],
            "worker_vmhwm_bytes": 900,
            "failure": None,
            "completed_at": "2026-07-27T00:01:00+00:00",
        }
        worker["worker_result_sha256"] = canonical_digest(
            worker, "worker_result_sha256"
        )
        write_exclusive_json(artifact_root / "worker_result.json", worker)
        return 1000, 0, None, None

    monkeypatch.setattr(module, "RuntimeResourceGuard", Guard)
    monkeypatch.setattr(module.subprocess, "Popen", Process)
    monkeypatch.setattr(module, "_monitor_probe_process", monitor)
    result = module._run_controller(
        policy, config, manifest, manifest_path, artifact_root
    )
    spec = load_json(artifact_root / "probe_spec.json", "probe spec")
    admission = load_json(artifact_root / "admission.json", "admission")
    assert result["status"] == "succeeded"
    assert spec["admission"] == {
        "path": str((artifact_root / "admission.json").resolve()),
        "sha256": hashlib.sha256(
            (artifact_root / "admission.json").read_bytes()
        ).hexdigest(),
        "canonical_sha256": admission["admission_sha256"],
    }
    assert (
        spec["probe_execution_sha256"]
        == admission["probe_execution_sha256"]
    )
    assert (artifact_root / "controller_claim.json").is_file()
    assert not (artifact_root / "controller_terminal.json").exists()
    assert (artifact_root / "probe_result.json").is_file()


def test_worker_environment_overrides_inherited_cuda_remap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _controller_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2,1,0")
    env = module._worker_environment("GPU-authorized")
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-authorized"
    with pytest.raises(CanonicalScreeningError, match="UUID"):
        module._worker_environment("0")


def test_ram_projection_accepts_84_99_and_rejects_exact_85() -> None:
    module = _controller_module()
    source = {"contract_type": "fixture"}
    accepted = module._ram_reservation_projection(
        total_bytes=1_000_000,
        used_bytes=0,
        slot_budget_bytes=849_900,
        slot_count=1,
        admission_limit_percent=85,
        budget_source=source,
    )
    assert accepted["projected_used_percent"] == 84.99
    with pytest.raises(CanonicalScreeningError, match="RAM reservation"):
        module._ram_reservation_projection(
            total_bytes=1_000_000,
            used_bytes=0,
            slot_budget_bytes=850_000,
            slot_count=1,
            admission_limit_percent=85,
            budget_source=source,
        )


def test_sealed_4d_ram_probe_artifact_tree_and_source() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = validate_policy(
        root,
        root / "configs/closeout/canonical_screening_512_v1.json",
        verify_historical_output_evidence=False,
    )
    resources = policy["resources"]
    source = resources["ram_slot_budget_source"]
    seal = source["probe_artifact_seal"]
    assert resources["ram_budget_status"] == "sealed"
    assert resources["ram_slot_budget_bytes"] == 3_768_299_111
    assert source["peak_sampled_process_tree_rss_bytes"] == 3_275_694_080
    assert source["worker_vmhwm_bytes"] == 3_425_726_464
    assert source["ram_budget_basis_bytes"] == 3_425_726_464
    assert seal["file_count"] == 28
    assert seal["directory_count"] == 5
    assert seal["symlink_count"] == 0
    assert len([name for name in seal["files"] if name.endswith(".png")]) == 16
    assert seal["controller_terminal"] == "absent_by_contract"
    assert seal["scientific_result_reuse"] == "forbidden"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("controller_terminal", "present"),
        ("scientific_result_reuse", "allowed"),
        ("file_count", 27),
    ],
)
def test_sealed_4d_ram_probe_rejects_contract_tamper(
    field: str, value: object
) -> None:
    root = Path(__file__).resolve().parents[1]
    raw = load_json(
        root / "configs/closeout/canonical_screening_512_v1.json",
        "canonical policy",
    )
    seal = dict(raw["resources"]["ram_slot_budget_source"]["probe_artifact_seal"])
    seal[field] = value
    with pytest.raises(CanonicalScreeningError, match="artifact tree"):
        _validate_ram_probe_artifact_seal(root, seal)


@pytest.mark.parametrize("symlink_kind", ["root", "ancestor"])
def test_sealed_4d_ram_probe_rejects_path_component_symlinks(
    tmp_path: Path, symlink_kind: str
) -> None:
    source_repo = Path(__file__).resolve().parents[1]
    raw = load_json(
        source_repo / "configs/closeout/canonical_screening_512_v1.json",
        "canonical policy",
    )
    seal = raw["resources"]["ram_slot_budget_source"]["probe_artifact_seal"]
    source_root = (
        source_repo
        / "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__4d0345b6fc29cc8e"
    )
    expected_root = (
        tmp_path
        / "artifacts/closeout/historical-canonical-512-v1/"
        "ram_probe__4d0345b6fc29cc8e"
    )
    if symlink_kind == "root":
        expected_root.parent.mkdir(parents=True)
        expected_root.symlink_to(source_root, target_is_directory=True)
    else:
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/closeout").symlink_to(
            source_repo / "artifacts/closeout",
            target_is_directory=True,
        )
    with pytest.raises(CanonicalScreeningError, match="symlink"):
        _validate_ram_probe_artifact_seal(tmp_path, seal)


def test_eight_worker_ram_projection_is_strictly_below_85_percent() -> None:
    module = _controller_module()
    total = 100_000_000_000
    slot_budget = 3_768_299_111
    slots = 8
    reserved = slots * slot_budget
    exact_85_used = 85_000_000_000 - reserved
    source = {"contract_type": "safa_canonical_screening_ram_budget_source_v2"}
    accepted = module._ram_reservation_projection(
        total_bytes=total,
        used_bytes=exact_85_used - 1,
        slot_budget_bytes=slot_budget,
        slot_count=slots,
        admission_limit_percent=85,
        budget_source=source,
    )
    assert accepted["slot_count"] == 8
    assert accepted["reserved_bytes"] == 30_146_392_888
    assert accepted["projected_used_bytes"] == 84_999_999_999
    assert accepted["projected_used_percent"] < 85
    with pytest.raises(CanonicalScreeningError, match="RAM reservation"):
        module._ram_reservation_projection(
            total_bytes=total,
            used_bytes=exact_85_used,
            slot_budget_bytes=slot_budget,
            slot_count=slots,
            admission_limit_percent=85,
            budget_source=source,
        )


def test_ram_probe_selects_largest_checkpoint_per_output_space(
    tmp_path: Path,
) -> None:
    module = _ram_probe_module()
    candidates = []
    for output_space, sizes in (("latent", (3, 7)), ("pixel", (5, 2))):
        for index, size in enumerate(sizes):
            checkpoint = tmp_path / f"{output_space}-{index}.pt"
            checkpoint.write_bytes(bytes([index + 1]) * size)
            candidates.append(
                {
                    "candidate_id": f"{output_space}-{index}",
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                    "checkpoint_model": "raw",
                    "output_contract": {
                        "output_contract_sha256": str(index) * 64,
                        "capability": {"output_space": output_space},
                    },
                }
            )
    selected = module._select_probe_candidates({"candidates": candidates})
    assert [
        (row["output_space"], row["candidate_id"], row["checkpoint_size_bytes"])
        for row in selected
    ] == [("latent", "latent-1", 7), ("pixel", "pixel-0", 5)]


def test_ram_probe_manifest_uses_current_plan_and_candidate_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _ram_probe_module()
    plan_path = tmp_path / "plan.json"
    plan = {"preflight_result_root": str((tmp_path / "preflight").resolve())}
    write_exclusive_json(plan_path, plan)
    manifest = {
        "checkpoint_plan": _bound(plan_path),
        "candidates": [{"candidate_id": "candidate"}],
    }
    manifest_path = tmp_path / "manifest.json"
    write_exclusive_json(manifest_path, manifest)
    policy = {"policy_sha256": "1" * 64}
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        module,
        "validate_checkpoint_plan",
        lambda value, **kwargs: calls.append(
            ("plan", (value, kwargs["policy"]))
        )
        or value,
    )
    monkeypatch.setattr(
        module,
        "validate_candidate_manifest",
        lambda value, **kwargs: calls.append(
            ("manifest", (value, kwargs["policy"]))
        )
        or value,
    )
    assert module._validate_manifest_envelope(manifest_path, policy) == manifest
    assert calls == [
        ("plan", (plan, policy)),
        ("manifest", (manifest, policy)),
    ]
    manifest["checkpoint_plan"]["sha256"] = "0" * 64
    manifest_path.write_bytes(canonical_json(manifest))
    with pytest.raises(CanonicalScreeningError, match="plan binding"):
        module._validate_manifest_envelope(manifest_path, policy)


def _sealed_ram_probe_fixture(tmp_path: Path) -> tuple[dict, int]:
    purpose = "resource_measurement_only_scientific_reuse_forbidden"
    gpu_uuid = _gpu_uuid(0)
    input_file = tmp_path / "input.json"
    input_file.write_text("{}\n", encoding="utf-8")
    input_binding = _bound(input_file)
    policy_snapshot = tmp_path / "input_policy.json"
    policy_snapshot.write_text(
        '{"resources":{"ram_budget_status":"probe_required"}}\n',
        encoding="utf-8",
    )
    policy_binding = {
        "path": str(
            (
                tmp_path / "configs/closeout/canonical_screening_512_v1.json"
            ).resolve()
        ),
        "sha256": hashlib.sha256(policy_snapshot.read_bytes()).hexdigest(),
        "canonical_sha256": "1" * 64,
        "snapshot": _bound(policy_snapshot),
    }
    selected = [
        {
            "candidate_id": f"{space}-candidate",
            "checkpoint_sha256": str(index + 2) * 64,
            "checkpoint_model": "raw",
            "checkpoint_size_bytes": 100 + index,
            "output_space": space,
            "output_contract_sha256": str(index + 4) * 64,
        }
        for index, space in enumerate(("latent", "pixel"))
    ]
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": (
                gpu_uuid if index == 0 else _gpu_uuid(index)
            ),
        }
        for index in range(4)
    ]
    spec = {
        "schema_version": 1,
        "contract_type": "safa_canonical_screening_ram_probe_v1",
        "purpose": purpose,
        "policy": policy_binding,
        "candidate_manifest": input_binding,
        "selected_candidates": selected,
        "sample_manifest": input_binding,
        "sample_count": 8,
        "seed": 4549,
        "batch_size": 2,
        "authorized_gpu_registry": registry,
        "artifact_root": str(tmp_path.resolve()),
        "implementations": {"worker": input_binding},
        "retry_count": 0,
        "probe_sha256": None,
    }
    spec["probe_sha256"] = canonical_digest(spec, "probe_sha256")
    write_exclusive_json(tmp_path / "probe_spec.json", spec)
    admission = {
        "schema_version": 1,
        "contract_type": "safa_canonical_screening_ram_probe_admission_v1",
        "probe_sha256": spec["probe_sha256"],
        "host": {},
        "gpu_snapshot": [],
        "authorized_gpu_registry": registry,
        "observed_at": "2026-07-26T00:00:00+00:00",
    }
    admission["admission_sha256"] = canonical_digest(
        admission, "admission_sha256"
    )
    write_exclusive_json(tmp_path / "admission.json", admission)
    worker = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_screening_ram_probe_worker_result_v1"
        ),
        "probe_sha256": spec["probe_sha256"],
        "purpose": purpose,
        "device_binding": {
            "physical_gpu_index": 0,
            "physical_gpu_uuid": gpu_uuid,
            "logical_cuda_index": 0,
            "runtime_cuda_uuid": gpu_uuid,
            "cuda_visible_devices": gpu_uuid,
        },
        "steps": [
            {
                **descriptor,
                "sample_count": 8,
                "generated_png_manifest_sha256": "8" * 64,
            }
            for descriptor in selected
        ],
        "worker_vmhwm_bytes": 900,
        "completed_at": "2026-07-26T00:01:00+00:00",
    }
    worker["worker_result_sha256"] = canonical_digest(
        worker, "worker_result_sha256"
    )
    write_exclusive_json(tmp_path / "worker_result.json", worker)
    worker_log = tmp_path / "worker.log"
    worker_log.write_text("probe\n", encoding="utf-8")
    peak = 1000
    budget = 1100
    method = (
        "ceil(max(peak_sampled_process_tree_rss_bytes,"
        "worker_vmhwm_bytes)*11/10);sampled_tree_every_0.1s_"
        "plus_worker_vmhwm_not_a_mathematical_instantaneous_tree_peak"
    )
    result = {
        "schema_version": 1,
        "contract_type": "safa_canonical_screening_ram_probe_result_v1",
        "status": "succeeded",
        "purpose": purpose,
        "probe_sha256": spec["probe_sha256"],
        "admission_sha256": admission["admission_sha256"],
        "worker_result_sha256": worker["worker_result_sha256"],
        "worker_log_sha256": hashlib.sha256(worker_log.read_bytes()).hexdigest(),
        "worker_returncode": 0,
        "termination": None,
        "peak_sampled_process_tree_rss_bytes": peak,
        "worker_vmhwm_bytes": 900,
        "ram_budget_basis_bytes": peak,
        "ram_slot_budget_bytes": budget,
        "budget_method": method,
        "measurement_factor_numerator": 11,
        "measurement_factor_denominator": 10,
        "runtime_resource_guard": {
            "violated": False,
            "violation_reason": None,
            "thread_failure": None,
        },
        "failure": None,
        "retry_count": 0,
        "completed_at": "2026-07-26T00:02:00+00:00",
    }
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    result_path = tmp_path / "probe_result.json"
    write_exclusive_json(result_path, result)
    source = {
        "contract_type": "safa_canonical_screening_ram_budget_source_v1",
        "method": method,
        "measurement_factor_numerator": 11,
        "measurement_factor_denominator": 10,
        "peak_sampled_process_tree_rss_bytes": peak,
        "worker_vmhwm_bytes": 900,
        "ram_budget_basis_bytes": peak,
        "ram_slot_budget_bytes": budget,
        "probe_result": _bound(result_path),
    }
    return source, budget


def _reseal_ram_probe_fixture(
    tmp_path: Path, source: dict
) -> None:
    snapshot_path = tmp_path / "input_policy.json"
    spec_path = tmp_path / "probe_spec.json"
    spec = load_json(spec_path, "RAM probe spec")
    snapshot_binding = _bound(snapshot_path)
    spec["policy"]["sha256"] = snapshot_binding["sha256"]
    spec["policy"]["snapshot"] = snapshot_binding
    spec["probe_sha256"] = canonical_digest(spec, "probe_sha256")
    spec_path.write_bytes(canonical_json(spec))

    admission_path = tmp_path / "admission.json"
    admission = load_json(admission_path, "RAM probe admission")
    admission["probe_sha256"] = spec["probe_sha256"]
    admission["admission_sha256"] = canonical_digest(
        admission, "admission_sha256"
    )
    admission_path.write_bytes(canonical_json(admission))

    worker_path = tmp_path / "worker_result.json"
    worker = load_json(worker_path, "RAM probe worker result")
    worker["probe_sha256"] = spec["probe_sha256"]
    worker["worker_result_sha256"] = canonical_digest(
        worker, "worker_result_sha256"
    )
    worker_path.write_bytes(canonical_json(worker))

    result_path = tmp_path / "probe_result.json"
    result = load_json(result_path, "RAM probe result")
    result["probe_sha256"] = spec["probe_sha256"]
    result["admission_sha256"] = admission["admission_sha256"]
    result["worker_result_sha256"] = worker["worker_result_sha256"]
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    result_path.write_bytes(canonical_json(result))
    source["probe_result"] = _bound(result_path)


def _upgrade_ram_probe_fixture_to_v2(tmp_path: Path, source: dict) -> None:
    spec_path = tmp_path / "probe_spec.json"
    spec = load_json(spec_path, "RAM probe spec")
    spec.pop("probe_sha256")
    spec["contract_type"] = "safa_canonical_screening_ram_probe_v2"
    spec["admission"] = None
    spec["probe_contract_sha256"] = None
    spec["probe_execution_sha256"] = None
    contract = ram_probe_contract_digest(spec)

    admission_path = tmp_path / "admission.json"
    admission = load_json(admission_path, "RAM probe admission")
    admission.pop("probe_sha256")
    admission["contract_type"] = (
        "safa_canonical_screening_ram_probe_admission_v2"
    )
    admission["probe_contract_sha256"] = contract
    admission["admission_evidence_sha256"] = (
        ram_probe_admission_evidence_digest(admission)
    )
    execution = ram_probe_execution_digest(
        contract,
        admission["authorized_gpu_registry"],
        admission["admission_evidence_sha256"],
    )
    admission["probe_execution_sha256"] = execution
    admission["admission_sha256"] = canonical_digest(
        admission, "admission_sha256"
    )
    admission_path.write_bytes(canonical_json(admission))

    spec["probe_contract_sha256"] = contract
    spec["probe_execution_sha256"] = execution
    spec["admission"] = {
        **_bound(admission_path),
        "canonical_sha256": admission["admission_sha256"],
    }
    spec_path.write_bytes(canonical_json(spec))

    worker_path = tmp_path / "worker_result.json"
    worker = load_json(worker_path, "RAM probe worker result")
    worker.pop("probe_sha256")
    worker["contract_type"] = (
        "safa_canonical_screening_ram_probe_worker_result_v2"
    )
    worker["status"] = "succeeded"
    worker["failure"] = None
    worker["probe_contract_sha256"] = contract
    worker["probe_execution_sha256"] = execution
    worker["worker_result_sha256"] = canonical_digest(
        worker, "worker_result_sha256"
    )
    worker_path.write_bytes(canonical_json(worker))

    result_path = tmp_path / "probe_result.json"
    result = load_json(result_path, "RAM probe result")
    result.pop("probe_sha256")
    result["contract_type"] = "safa_canonical_screening_ram_probe_result_v2"
    result["probe_contract_sha256"] = contract
    result["probe_execution_sha256"] = execution
    result["worker_device_binding"] = worker["device_binding"]
    result["admission_sha256"] = admission["admission_sha256"]
    result["worker_result_sha256"] = worker["worker_result_sha256"]
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    result_path.write_bytes(canonical_json(result))
    source["probe_result"] = _bound(result_path)


@pytest.mark.parametrize(
    ("artifact", "field"),
    [
        ("probe_spec.json", "probe_contract_sha256"),
        ("probe_spec.json", "probe_execution_sha256"),
        ("admission.json", "authorized_gpu_registry"),
        ("worker_result.json", "probe_execution_sha256"),
        ("probe_result.json", "probe_contract_sha256"),
    ],
)
def test_sealed_ram_probe_v2_rejects_digest_chain_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    field: str,
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    _upgrade_ram_probe_fixture_to_v2(tmp_path, source)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: {"policy_sha256": "1" * 64},
    )
    assert (
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )["ram_slot_budget_bytes"]
        == budget
    )
    path = tmp_path / artifact
    changed = load_json(path, "tampered v2 RAM probe artifact")
    if field == "authorized_gpu_registry":
        changed[field][0]["physical_gpu_uuid"] = _gpu_uuid(1)
    else:
        changed[field] = "0" * 64
    own_digest = {
        "admission.json": "admission_sha256",
        "worker_result.json": "worker_result_sha256",
        "probe_result.json": "probe_result_sha256",
    }.get(artifact)
    if own_digest is not None:
        changed[own_digest] = canonical_digest(changed, own_digest)
    path.write_bytes(canonical_json(changed))
    if artifact == "probe_result.json":
        source["probe_result"] = _bound(path)
    with pytest.raises(CanonicalScreeningError, match="evidence chain"):
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )


@pytest.mark.parametrize(
    ("artifact", "field", "value", "message"),
    (
        ("probe_result.json", "status", "failed", "semantics"),
        ("input_policy.json", "policy", "forged", "snapshot binding"),
        ("probe_spec.json", "purpose", "scientific", "evidence chain"),
        ("admission.json", "probe_sha256", "0" * 64, "evidence chain"),
        (
            "worker_result.json",
            "device_binding",
            {
                "physical_gpu_index": 0,
                "physical_gpu_uuid": "GPU-tampered",
                "logical_cuda_index": 0,
                "runtime_cuda_uuid": "GPU-tampered",
                "cuda_visible_devices": "GPU-tampered",
            },
            "evidence chain",
        ),
    ),
)
def test_sealed_ram_probe_chain_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    field: str,
    value: object,
    message: str,
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: {"policy_sha256": "1" * 64},
    )
    assert (
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )["ram_slot_budget_bytes"]
        == budget
    )
    path = tmp_path / artifact
    changed = load_json(path, "tampered RAM probe artifact")
    changed[field] = value
    digest_fields = {
        "probe_result.json": "probe_result_sha256",
        "probe_spec.json": "probe_sha256",
        "admission.json": "admission_sha256",
        "worker_result.json": "worker_result_sha256",
    }
    if artifact in digest_fields:
        changed[digest_fields[artifact]] = canonical_digest(
            changed, digest_fields[artifact]
        )
    path.write_bytes(canonical_json(changed))
    if artifact == "probe_result.json":
        source["probe_result"] = _bound(path)
    with pytest.raises(CanonicalScreeningError, match=message):
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )


def test_ram_probe_sampler_exception_terminates_and_reaps_worker_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 123

        def poll(self):
            return None

    class Guard:
        def raise_if_violated(self) -> None:
            return None

    cleanup_calls: list[int] = []
    monkeypatch.setattr(
        module, "_process_group_members", lambda _group: ((123, 1),)
    )
    monkeypatch.setattr(
        module,
        "_sample_or_reap_process_tree",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("sampler injected")),
    )
    monkeypatch.setattr(
        module,
        "_terminate_process_group",
        lambda process, **_kwargs: (
            cleanup_calls.append(process.pid)
            or {
                "term_sent": True,
                "kill_sent": False,
                "reaped_returncode": -15,
            }
        ),
    )
    peak, returncode, failure, termination = module._monitor_probe_process(
        Process(), {}, Guard()
    )
    assert peak == 0
    assert returncode == -15
    assert failure == "RuntimeError: sampler injected"
    assert termination["term_sent"] is True
    assert cleanup_calls == [123]


def test_ram_probe_descendant_appearance_terminates_and_reaps_worker_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 321

        def poll(self):
            return None

    class Guard:
        def raise_if_violated(self) -> None:
            return None

    cleanup_calls: list[int] = []
    monkeypatch.setattr(
        module, "_process_descendants", lambda _root: ((654, 2),)
    )
    monkeypatch.setattr(
        module,
        "_terminate_process_group",
        lambda process, **_kwargs: (
            cleanup_calls.append(process.pid)
            or {
                "term_sent": True,
                "kill_sent": False,
                "reaped_returncode": -15,
            }
        ),
    )
    peak, returncode, failure, termination = module._monitor_probe_process(
        Process(), {}, Guard()
    )
    assert peak == 0
    assert returncode == -15
    assert "forbidden descendant processes" in failure
    assert termination["term_sent"] is True
    assert cleanup_calls == [321]


def test_ram_probe_descendant_scan_tracks_children_that_escape_process_group(
    tmp_path: Path,
) -> None:
    module = _ram_probe_module()
    proc_root = tmp_path / "proc"

    def write_stat(
        pid: int, *, parent: int, process_group: int, start_time: int
    ) -> None:
        directory = proc_root / str(pid)
        directory.mkdir(parents=True)
        fields = (
            ["S", str(parent), str(process_group)]
            + ["0"] * 16
            + [str(start_time)]
        )
        (directory / "stat").write_text(
            f"{pid} (fixture worker) {' '.join(fields)}\n",
            encoding="utf-8",
        )

    write_stat(100, parent=1, process_group=100, start_time=10)
    write_stat(101, parent=100, process_group=999, start_time=11)
    write_stat(102, parent=101, process_group=998, start_time=12)
    write_stat(200, parent=1, process_group=200, start_time=20)

    assert module._process_group_members(100, proc_root=proc_root) == (
        (100, 10),
    )
    assert module._process_descendants(100, proc_root=proc_root) == (
        (101, 11),
        (102, 12),
    )


def test_ram_probe_process_group_cleanup_escalates_to_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 456
        returncode = None
        waits = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout: int):
            assert timeout == 10
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("probe", timeout)
            self.returncode = -9
            return self.returncode

    process = Process()
    signals: list[tuple[int, object]] = []
    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module,
        "_process_group_members",
        lambda _group: (
            ((process.pid, 1),) if process.returncode is None else ()
        ),
    )
    monotonic = iter((0.0, 11.0, 11.0, 11.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(module, "_live_process_identities", lambda _ids: set())
    monkeypatch.setattr(module, "_signal_process_identities", lambda *_args: 0)

    def killpg(pgid, sig):
        signals.append((pgid, sig))
        if sig == module.signal.SIGKILL:
            process.returncode = -9

    monkeypatch.setattr(
        module.os, "killpg", killpg
    )
    result = module._terminate_process_group(process)
    assert result == {
        "term_sent": True,
        "kill_sent": True,
        "reaped_returncode": -9,
    }
    assert signals == [
        (process.pid, module.signal.SIGTERM),
        (process.pid, module.signal.SIGKILL),
    ]


def test_ram_probe_cleanup_reaps_initial_zombie_before_group_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 457
        returncode = None

        def poll(self):
            self.returncode = 0
            return self.returncode

    process = Process()
    group_scans: list[int] = []
    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module,
        "_process_group_members",
        lambda group: group_scans.append(group) or (),
    )
    monkeypatch.setattr(module, "_live_process_identities", lambda _ids: set())
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("reaped zombie group must not be signalled")
        ),
    )
    assert module._terminate_process_group(process) == {
        "term_sent": False,
        "kill_sent": False,
        "reaped_returncode": 0,
    }
    assert group_scans == [process.pid]


def test_ram_probe_cleanup_accepts_esrch_only_after_empty_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 458
        returncode = None
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls > 1:
                self.returncode = 0
            return self.returncode

    process = Process()
    member_samples = iter((((process.pid, 1),), ()))
    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module, "_process_group_members", lambda _group: next(member_samples)
    )
    monkeypatch.setattr(module, "_live_process_identities", lambda _ids: set())
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    result = module._terminate_process_group(process)
    assert result == {
        "term_sent": False,
        "kill_sent": False,
        "reaped_returncode": 0,
    }


def test_ram_probe_cleanup_esrch_with_live_members_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 459
        returncode = None

        def poll(self):
            return None

    process = Process()
    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module,
        "_process_group_members",
        lambda _group: ((process.pid, 1),),
    )
    monkeypatch.setattr(module, "_live_process_identities", lambda _ids: set())
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: process.pid)

    def killpg(_group, sig):
        if sig == module.signal.SIGKILL:
            raise ProcessLookupError()

    monkeypatch.setattr(module.os, "killpg", killpg)
    monotonic = iter((0.0, 11.0, 11.0, 22.0))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic))
    with pytest.raises(CanonicalScreeningError, match="survived SIGKILL"):
        module._terminate_process_group(process)


def test_ram_probe_root_exit_cleans_residual_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 777
        returncode = 0

        def poll(self):
            return 0

    class Guard:
        def raise_if_violated(self) -> None:
            return None

    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module, "_process_group_members", lambda _group: ((888, 2),)
    )
    monkeypatch.setattr(
        module,
        "_terminate_process_group",
        lambda _process, **_kwargs: {
            "term_sent": True,
            "kill_sent": False,
            "reaped_returncode": 0,
        },
    )
    peak, returncode, failure, termination = module._monitor_probe_process(
        Process(), {}, Guard()
    )
    assert peak == 0 and returncode == 0
    assert "residual process-group members" in failure
    assert termination["term_sent"] is True


def test_ram_probe_sampling_exit_cleans_residual_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _ram_probe_module()

    class Process:
        pid = 778
        returncode = 0

        def poll(self):
            return None

    class Guard:
        def raise_if_violated(self) -> None:
            return None

    monkeypatch.setattr(module, "_process_descendants", lambda _root: ())
    monkeypatch.setattr(
        module, "_process_group_members", lambda _group: ((889, 2),)
    )
    monkeypatch.setattr(
        module, "_sample_or_reap_process_tree", lambda *_args: (None, 0)
    )
    monkeypatch.setattr(
        module,
        "_terminate_process_group",
        lambda _process, **_kwargs: {
            "term_sent": True,
            "kill_sent": False,
            "reaped_returncode": 0,
        },
    )
    peak, returncode, failure, termination = module._monitor_probe_process(
        Process(), {}, Guard()
    )
    assert peak == 0 and returncode == 0
    assert "exited during RSS sampling" in failure
    assert termination["term_sent"] is True


def test_sealed_ram_budget_uses_higher_worker_vmhwm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    result_path = tmp_path / "probe_result.json"
    result = load_json(result_path, "RAM probe result")
    result["peak_sampled_process_tree_rss_bytes"] = 500
    result["worker_vmhwm_bytes"] = 1000
    result["ram_budget_basis_bytes"] = 1000
    worker_path = tmp_path / "worker_result.json"
    worker = load_json(worker_path, "RAM probe worker result")
    worker["worker_vmhwm_bytes"] = 1000
    worker["worker_result_sha256"] = canonical_digest(
        worker, "worker_result_sha256"
    )
    worker_path.write_bytes(canonical_json(worker))
    result["worker_result_sha256"] = worker["worker_result_sha256"]
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    result_path.write_bytes(canonical_json(result))
    source["peak_sampled_process_tree_rss_bytes"] = 500
    source["worker_vmhwm_bytes"] = 1000
    source["ram_budget_basis_bytes"] = 1000
    source["probe_result"] = _bound(result_path)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: {"policy_sha256": "1" * 64},
    )
    validated = _validate_ram_slot_budget_source(
        tmp_path,
        source,
        declared_budget_bytes=budget,
        expected_predecessor_policy_sha256="1" * 64,
    )
    assert validated["ram_budget_basis_bytes"] == 1000


def test_sealed_ram_budget_rejects_vmhwm_chain_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    result_path = tmp_path / "probe_result.json"
    result = load_json(result_path, "RAM probe result")
    result["peak_sampled_process_tree_rss_bytes"] = 500
    result["worker_vmhwm_bytes"] = 1000
    result["ram_budget_basis_bytes"] = 1000
    result["probe_result_sha256"] = canonical_digest(
        result, "probe_result_sha256"
    )
    result_path.write_bytes(canonical_json(result))
    source["peak_sampled_process_tree_rss_bytes"] = 500
    source["worker_vmhwm_bytes"] = 1000
    source["ram_budget_basis_bytes"] = 1000
    source["probe_result"] = _bound(result_path)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: {"policy_sha256": "1" * 64},
    )
    with pytest.raises(CanonicalScreeningError, match="evidence chain"):
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )


def test_sealed_ram_budget_rejects_forged_snapshot_canonical_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: {"policy_sha256": "2" * 64},
    )
    with pytest.raises(CanonicalScreeningError, match="predecessor policy"):
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )


def test_sealed_ram_budget_rejects_recursive_sealed_snapshot_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, budget = _sealed_ram_probe_fixture(tmp_path)
    snapshot_path = tmp_path / "input_policy.json"
    snapshot_path.write_text(
        '{"resources":{"ram_budget_status":"sealed"}}\n',
        encoding="utf-8",
    )
    _reseal_ram_probe_fixture(tmp_path, source)
    monkeypatch.setattr(
        sys.modules[_validate_ram_slot_budget_source.__module__],
        "validate_policy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("recursive policy validation must not start")
        ),
    )
    with pytest.raises(CanonicalScreeningError, match="probe-required"):
        _validate_ram_slot_budget_source(
            tmp_path,
            source,
            declared_budget_bytes=budget,
            expected_predecessor_policy_sha256="1" * 64,
        )


def test_screen512_gate_requires_exact_primary_repeat_smoke(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    policy, manifest, manifest_path, policy_path, admission, _ = _manifest_fixture(
        tmp_path
    )
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    candidate = manifest["candidates"][0]
    controller_ready, observer_ready = _ready_bindings(
        tmp_path, policy, admission, "smoke8"
    )
    baseline_rows = {}
    baseline_results = {}
    for replicate in ("primary", "repeat"):
        request = build_run_request(
            policy,
            policy_path,
            manifest,
            manifest_path,
            candidate,
            "smoke8",
            replicate,
            paths["runs"],
            admission,
            controller_ready,
            observer_ready,
        )
        request_path = (
            paths["run_requests"]
            / f"smoke8_{replicate}"
            / f"{candidate['candidate_id']}.json"
        )
        write_exclusive_json(request_path, request)
        output = Path(request["output_dir"])
        output.mkdir(parents=True)
        generated = output / "generated"
        generated.mkdir()
        rows = []
        for index in range(8):
            source_path = tmp_path / "sources" / f"{index:06d}.png"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(f"source{index}".encode())
            candidate_path = generated / f"{index:06d}.png"
            candidate_path.write_bytes(f"png{index}".encode())
            rows.append(
                {
                    "sample_id": f"s{index}",
                    "run_request_sha256": request["run_request_sha256"],
                    "checkpoint_sha256": request["candidate"][
                        "checkpoint_sha256"
                    ],
                    "checkpoint_model": request["candidate"][
                        "checkpoint_model"
                    ],
                    "source_path": str(source_path.resolve()),
                    "source_sha256": hashlib.sha256(
                        source_path.read_bytes()
                    ).hexdigest(),
                    "candidate_path": str(candidate_path.resolve()),
                    "candidate_sha256": hashlib.sha256(
                        candidate_path.read_bytes()
                    ).hexdigest(),
                    "native_output_sha256": hashlib.sha256(
                        f"native{index}".encode()
                    ).hexdigest(),
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
                    "native_output_shape": [3, 224, 224],
                    "native_rgb_shape": [3, 224, 224],
                    "native_rgb_size": [224, 224],
                    "quality_protocol_family": request[
                        "quality_protocol_family"
                    ],
                    "nfe": request["nfe"],
                    "e0_cosine": 0.8,
                    "edev_cosine": 0.7,
                    "arcface_source_face_count": 1,
                    "arcface_candidate_face_count": 1,
                    "arcface_source_candidate_cosine": 0.1,
                }
            )
        per_sample = output / "per_sample.jsonl"
        _write_jsonl(per_sample, rows)
        claim = _run_claim(
            policy,
            request,
            worker_pid=100 + (replicate == "repeat"),
        )
        write_exclusive_json(output / "claim.json", claim)
        evidence = _evidence(policy, request)
        evidence["per_sample_sha256"] = hashlib.sha256(
            per_sample.read_bytes()
        ).hexdigest()
        result = build_run_result(
            request,
            claim,
            policy,
            status="completed",
            completed_at="2026-07-26T00:01:00+00:00",
            evidence=evidence,
        )
        write_exclusive_json(output / "result.json", result)
        baseline_rows[replicate] = json.loads(json.dumps(rows))
        baseline_results[replicate] = json.loads(json.dumps(result))
    module._require_smoke_success(policy, manifest, paths)
    mutations = (
        lambda rows: rows[0].__setitem__("sample_id", "tampered"),
        lambda rows: rows[1].__setitem__("sample_id", rows[0]["sample_id"]),
        lambda rows: (
            rows[0].__setitem__("sample_id", "s1"),
            rows[1].__setitem__("sample_id", "s0"),
        ),
        lambda rows: rows[0].__setitem__(
            "native_rgb_shape", [3, 256, 256]
        ),
        lambda rows: rows[0].__setitem__(
            "output_contract_sha256", "f" * 64
        ),
    )
    for mutate in mutations:
        for replicate in ("primary", "repeat"):
            output = (
                paths["runs"]
                / f"smoke8_{replicate}"
                / candidate["candidate_id"]
            )
            changed_rows = json.loads(json.dumps(baseline_rows[replicate]))
            mutate(changed_rows)
            per_sample = output / "per_sample.jsonl"
            per_sample.unlink()
            _write_jsonl(per_sample, changed_rows)
            changed_result = json.loads(
                json.dumps(baseline_results[replicate])
            )
            changed_result["evidence"]["per_sample_sha256"] = hashlib.sha256(
                per_sample.read_bytes()
            ).hexdigest()
            changed_result["run_result_sha256"] = canonical_digest(
                changed_result,
                "run_result_sha256",
            )
            (output / "result.json").write_bytes(
                canonical_json(changed_result)
            )
        with pytest.raises(CanonicalScreeningError, match="smoke8 per-sample"):
            module._require_smoke_success(policy, manifest, paths)
        for replicate in ("primary", "repeat"):
            output = (
                paths["runs"]
                / f"smoke8_{replicate}"
                / candidate["candidate_id"]
            )
            per_sample = output / "per_sample.jsonl"
            per_sample.unlink()
            _write_jsonl(per_sample, baseline_rows[replicate])
            (output / "result.json").write_bytes(
                canonical_json(baseline_results[replicate])
            )


def test_e0_cosine_uses_locked_target_z_not_source_embedding() -> None:
    torch = pytest.importorskip("torch")
    generated = torch.tensor([[1.0, 0.0]])
    target_z = torch.tensor([[1.0, 0.0]])
    source_e0 = torch.tensor([[0.0, 1.0]])
    generated_edev = torch.tensor([[0.0, 1.0]])
    source_edev = torch.tensor([[0.0, 1.0]])
    e0_cosine, edev_cosine = _representation_cosines(
        generated, target_z, generated_edev, source_edev
    )
    assert e0_cosine.item() == pytest.approx(1.0)
    assert edev_cosine.item() == pytest.approx(1.0)
    assert torch.nn.functional.cosine_similarity(generated, source_e0).item() == 0.0


def test_edev_source_loader_is_locked_to_256(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    black = tmp_path / "black.png"
    white = tmp_path / "white.png"
    image_module.new("RGB", (17, 23), color=(0, 0, 0)).save(black)
    image_module.new("RGB", (17, 23), color=(255, 255, 255)).save(white)
    batch = _load_source_pixel_batch([black, white], 256, "cpu")
    assert tuple(batch.shape) == (2, 3, 256, 256)
    assert float(batch.min()) == 0.0
    assert float(batch.max()) == 1.0
    assert float(batch[0].max()) == 0.0
    assert float(batch[1].min()) == 1.0


def test_kid_subset_8_accepts_eight_real_and_fake_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import safa.closeout.canonical_quality as canonical_quality
    torch = pytest.importorskip("torch")
    root = tmp_path
    real_paths = [root / f"real_{index}.png" for index in range(8)]
    generated_paths = [root / f"generated_{index}.png" for index in range(8)]
    manifest = root / "canonical_kid_test_manifest.jsonl"
    per_sample = root / "canonical_kid_test_per_sample.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    per_sample.write_text("{}\n", encoding="utf-8")

    class FakeKid:
        def __init__(self, subset_size: int, normalize: bool) -> None:
            self.subset_size = subset_size
            self.real = 0
            self.fake = 0

        def update(self, _image, *, real: bool) -> None:
            if real:
                self.real += 1
            else:
                self.fake += 1

        def compute(self):
            assert self.real >= self.subset_size
            assert self.fake >= self.subset_size
            return 0.1, 0.01

    fake_quality = types.SimpleNamespace(
        manifest_image_paths=lambda **_kwargs: (
            [f"s{index}" for index in range(8)],
            real_paths,
            generated_paths,
        ),
        quality_eval_device=lambda _device: torch.device("cpu"),
        prepare_metric_for_device=lambda metric, device: (metric, device),
        load_image_uint8=lambda _path: torch.zeros(1, 3, 4, 4, dtype=torch.uint8),
        image_to_device=lambda image, _device: image,
        seed_metric_randomness=lambda _seed, _device: None,
        metric_scalar=float,
        asset_manifest_digest=lambda paths, labels: hashlib.sha256(
            canonical_json([str(path) for path in paths] + list(labels))
        ).hexdigest(),
    )
    monkeypatch.setattr(canonical_quality, "_load_quality_module", lambda _binding: fake_quality)
    torchmetrics = types.ModuleType("torchmetrics")
    image = types.ModuleType("torchmetrics.image")
    kid = types.ModuleType("torchmetrics.image.kid")
    kid.KernelInceptionDistance = FakeKid
    monkeypatch.setitem(sys.modules, "torchmetrics", torchmetrics)
    monkeypatch.setitem(sys.modules, "torchmetrics.image", image)
    monkeypatch.setitem(sys.modules, "torchmetrics.image.kid", kid)
    result = evaluate_locked_kid(
        quality_script={"path": "/locked", "sha256": "a" * 64},
        real_index=root / "index.jsonl",
        generated_dir=root,
        sample_id_manifest=manifest,
        per_sample_jsonl=per_sample,
        subset_seed=4549,
        subset_size=8,
        device="cpu",
    )
    assert result["kid_mean"] == 0.1
    assert result["kid_subset_size"] == 8


def test_invalid_result_validation_leaves_no_immutable_result(tmp_path: Path) -> None:
    policy, request = _run_fixture(tmp_path)
    claim = _run_claim(policy, request)
    result = build_run_result(
        request,
        claim,
        policy,
        status="completed",
        completed_at="2026-07-26T00:01:00+00:00",
        evidence=_evidence(policy, request),
    )
    result["evidence"]["policy_sha256"] = "f" * 64
    result["run_result_sha256"] = canonical_digest(result, "run_result_sha256")
    path = tmp_path / "result.json"
    with pytest.raises(CanonicalScreeningError, match="policy_sha256"):
        _write_validated_run_result(path, result, request, claim, policy)
    assert not path.exists()


def test_free_slot_pool_reuses_exact_out_of_order_completion() -> None:
    module = _controller_module()
    pool = module.FreeSlotPool([(0, 0), (0, 1), (1, 0)])
    assert pool.acquire() == (0, 0)
    second = pool.acquire()
    third = pool.acquire()
    assert (second, third) == ((0, 1), (1, 0))
    pool.release(third)
    assert pool.acquire() == third
    pool.release(second)
    with pytest.raises(CanonicalScreeningError, match="invalid GPU slot release"):
        pool.release(second)


def test_controller_cleanup_terminates_workers_and_releases_owned_lock(
    tmp_path: Path,
) -> None:
    module = _controller_module()

    class Process:
        def __init__(self) -> None:
            self.pid = 4242
            self.terminated = False
            self.waited = False

        def poll(self):
            return None if not self.terminated else -15

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None) -> int:
            self.waited = True
            return -15

    class Log:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    pool = module.FreeSlotPool([(0, 0)])
    slot = pool.acquire()
    lock = tmp_path / "owned.lock"
    lock.write_text("owned", encoding="utf-8")
    process = Process()
    log = Log()
    active = [{
        "process": process,
        "request": tmp_path / "request.json",
        "lock": lock,
        "log_handle": log,
        "slot": slot,
    }]
    guard = types.SimpleNamespace(unregister_worker_pid=lambda _pid: None)
    module._cleanup_active_workers(active, pool, guard)
    assert active == []
    assert process.terminated and process.waited and log.closed
    assert not lock.exists()
    assert pool.free_count == 1


def test_monitor_is_append_only_and_audit_reconstructable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 12.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 20.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 30.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (1, 2))
    gpu_rows = [
        {
            "index": index,
            "uuid": _gpu_uuid(index),
            "temperature_c": 40,
        }
        for index in range(4)
    ]
    monkeypatch.setattr(module, "_gpu_snapshot", lambda: gpu_rows)
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    admission = module._write_admission(
        policy,
        paths,
        "smoke8",
        {
            **_admission_snapshot(policy),
            "gpus": gpu_rows,
        },
    )
    path = module._append_monitor_sample(
        policy, paths, "smoke8", admission=admission
    )
    module._append_monitor_sample(
        policy, paths, "smoke8", terminal=True, admission=admission
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["terminal"] is False and rows[1]["terminal"] is True
    assert rows[0]["gpus"][0]["uuid"] == _gpu_uuid(0)
    assert rows[0]["artifacts"]["generated_png"] == 0


def test_cpu_admission_never_depends_on_gpu_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    policy["resources"].update({
        "cpu_admission_percent": 90,
        "ram_admission_percent": 85,
        "disk_admission_percent": 85,
    })
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 20.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 30.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (1, 2))
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("GPU must not be queried")),
    )
    snapshot = module.assert_cpu_resource_admission(policy, tmp_path)
    assert snapshot["admission_kind"] == "cpu_only"


@pytest.mark.parametrize(
    ("cpu_percent", "should_pass"),
    [(89.1, True), (90.0, False)],
)
def test_cpu_startup_admission_uses_strict_below_90_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_percent: float,
    should_pass: bool,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    policy["resources"]["cpu_admission_percent"] = 90
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: cpu_percent)
    monkeypatch.setattr(module, "_memory_percent", lambda: 20.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 30.0)
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("GPU must not be queried")),
    )
    if should_pass:
        assert module.assert_cpu_resource_admission(
            policy, tmp_path
        )["cpu_load_percent"] == cpu_percent
    else:
        with pytest.raises(CanonicalScreeningError, match="CPU admission failed"):
            module.assert_cpu_resource_admission(policy, tmp_path)


def test_cpu_window_requires_two_consecutive_windows_and_latches() -> None:
    module = _controller_module()
    single = module.CpuWindowState(90.0, 2)
    assert single.record(93.0) is False
    assert single.record(10.0) is False
    assert single.consecutive_high == 0
    exact = module.CpuWindowState(90.0, 2)
    assert exact.record(90.0) is False
    assert exact.consecutive_high == 1
    assert exact.record(90.0) is True
    consecutive = module.CpuWindowState(90.0, 2)
    assert consecutive.record(91.0) is False
    assert consecutive.record(92.0) is True
    assert consecutive.record(10.0) is True
    assert consecutive.violated is True


def test_runtime_guard_preserves_ram_disk_and_swap_hard_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    monkeypatch.setattr(module, "_cpu_times", lambda: (100, 50))

    class FiniteWait:
        def __init__(self, intervals: int) -> None:
            self.intervals = intervals
            self.calls = 0

        def wait(self, _seconds: int) -> bool:
            self.calls += 1
            return self.calls > self.intervals

    def run_case(
        name: str,
        *,
        memory_percent: float,
        disk_percent: float,
        swaps: list[tuple[int, int]],
        intervals: int,
    ) -> str:
        monotonic_values = iter(float(10 * index) for index in range(intervals + 1))
        swap_values = iter(swaps)
        monkeypatch.setattr(module.time, "monotonic", lambda: next(monotonic_values))
        monkeypatch.setattr(module, "_memory_percent", lambda: memory_percent)
        monkeypatch.setattr(module, "_disk_percent", lambda _path: disk_percent)
        monkeypatch.setattr(module, "_swap_pages", lambda: next(swap_values))
        guard = module.RuntimeResourceGuard(
            policy, tmp_path / f"{name}.jsonl", tmp_path
        )
        guard._stop = FiniteWait(intervals)
        guard._run()
        with pytest.raises(CanonicalScreeningError) as error:
            guard.raise_if_violated()
        return str(error.value)

    assert "RAM runtime hard stop" in run_case(
        "ram",
        memory_percent=90.0,
        disk_percent=10.0,
        swaps=[(0, 0), (0, 0)],
        intervals=1,
    )
    assert "disk runtime hard stop" in run_case(
        "disk",
        memory_percent=10.0,
        disk_percent=90.0,
        swaps=[(0, 0), (0, 0)],
        intervals=1,
    )
    assert "sustained swap I/O" in run_case(
        "swap",
        memory_percent=10.0,
        disk_percent=10.0,
        swaps=[(0, 0), (1, 0), (2, 0), (3, 0)],
        intervals=3,
    )


def test_runtime_guard_exposes_monitor_thread_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    monkeypatch.setattr(
        module,
        "_cpu_times",
        lambda: (_ for _ in ()).throw(RuntimeError("proc stat injected")),
    )
    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "guard.jsonl", tmp_path
    )
    guard._run()
    with pytest.raises(CanonicalScreeningError, match="proc stat injected"):
        guard.raise_if_violated()


def test_runtime_guard_hard_stops_unknown_gpu_pid_during_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    monkeypatch.setattr(module, "_cpu_times", lambda: (100, 50))
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 10.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (0, 0))
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: [
            {"index": row["physical_gpu_index"], "uuid": row["physical_gpu_uuid"]}
            for row in registry
        ],
    )
    monkeypatch.setattr(
        module,
        "_gpu_compute_processes",
        lambda: [
            {
                "gpu_uuid": registry[0]["physical_gpu_uuid"],
                "pid": 99991,
                "process_name": "foreign",
            }
        ],
    )

    class OneSample:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, _seconds: int) -> bool:
            self.calls += 1
            return self.calls > 1

        def set(self) -> None:
            self.calls = 2

    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "unknown_pid.jsonl", tmp_path, registry
    )
    launched = False

    def forbidden_factory():
        nonlocal launched
        launched = True
        return types.SimpleNamespace(pid=99991)

    with pytest.raises(CanonicalScreeningError, match="unknown compute PID"):
        guard.launch_authorized_worker(forbidden_factory)
    assert launched is False
    guard._stop = OneSample()
    guard._run()
    with pytest.raises(CanonicalScreeningError, match="unknown compute PID"):
        guard.raise_if_violated()
    sample = json.loads(
        (tmp_path / "unknown_pid.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert sample["unknown_compute_processes"][0]["pid"] == 99991


def test_runtime_guard_allows_atomically_registered_worker_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    monkeypatch.setattr(module, "_cpu_times", lambda: (100, 50))
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 10.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (0, 0))
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: [
            {"index": row["physical_gpu_index"], "uuid": row["physical_gpu_uuid"]}
            for row in registry
        ],
    )
    launched = False

    def compute_processes():
        return (
            [{"gpu_uuid": registry[0]["physical_gpu_uuid"], "pid": 4242}]
            if launched
            else []
        )

    monkeypatch.setattr(module, "_gpu_compute_processes", compute_processes)

    class OneSample:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, _seconds: int) -> bool:
            self.calls += 1
            return self.calls > 1

        def set(self) -> None:
            self.calls = 2

    process = types.SimpleNamespace(pid=4242)
    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "allowed_pid.jsonl", tmp_path, registry
    )

    def launch_worker():
        nonlocal launched
        launched = True
        return process

    assert guard.launch_authorized_worker(launch_worker) is process
    guard._stop = OneSample()
    guard._run()
    guard.raise_if_violated()
    guard.unregister_worker_pid(4242)
    assert guard.stop()["final_active_worker_pids"] == []


def test_cpu_worker_handshake_orders_rehash_launch_register_check_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 10.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (0, 0))
    monkeypatch.setattr(module, "_gpu_snapshot", lambda: [])
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    events = []
    process = types.SimpleNamespace(pid=5151)
    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "handshake.jsonl", tmp_path
    )

    def initial_validator():
        events.append("initial_rehash")
        return {"stage": "initial"}

    def factory():
        events.append("popen")
        return process

    launched, validation = guard.launch_cpu_worker(
        factory, initial_validator
    )
    assert launched is process
    assert validation == {"stage": "initial"}

    def final_validator():
        assert 5151 in guard._active_worker_pids
        events.append("final_rehash")
        return {"stage": "final"}

    def publisher(_validation, snapshot):
        assert snapshot["active_worker_pids"] == [5151]
        events.append("release")

    guard.release_worker_after_handshake(
        5151, final_validator, publisher
    )
    assert events == [
        "initial_rehash",
        "popen",
        "final_rehash",
        "release",
    ]
    guard.unregister_worker_pid(5151)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("cpu", "CPU release hard gate"),
        ("ram", "RAM release hard gate"),
        ("disk", "disk release hard gate"),
        ("swap", "swap I/O release hard gate"),
        ("gpu_memory", "release memory hard gate"),
        ("gpu_temperature", "release temperature hard gate"),
        ("unknown_pid", "unknown compute PID"),
        ("thread_failure", "guard failed before worker release"),
    ),
)
def test_release_lock_hard_gates_fresh_resource_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    monkeypatch.setattr(
        module,
        "_cpu_load_percent",
        lambda: 90.0 if case == "cpu" else 10.0,
    )
    monkeypatch.setattr(
        module,
        "_memory_percent",
        lambda: 90.0 if case == "ram" else 10.0,
    )
    monkeypatch.setattr(
        module,
        "_disk_percent",
        lambda _path: 90.0 if case == "disk" else 10.0,
    )
    swaps = iter(
        ((0, 0), (1, 0))
        if case == "swap"
        else ((0, 0), (0, 0))
    )
    monkeypatch.setattr(module, "_swap_pages", lambda: next(swaps))
    gpus = [
        {
            "index": row["physical_gpu_index"],
            "uuid": row["physical_gpu_uuid"],
            "memory_total_mib": 24576,
            "memory_used_mib": (
                23000
                if case == "gpu_memory"
                and row["physical_gpu_index"] == 0
                else 3
            ),
            "memory_free_mib": (
                1576
                if case == "gpu_memory"
                and row["physical_gpu_index"] == 0
                else 24573
            ),
            "temperature_c": (
                86
                if case == "gpu_temperature"
                and row["physical_gpu_index"] == 0
                else 35
            ),
        }
        for row in registry
    ]
    monkeypatch.setattr(module, "_gpu_snapshot", lambda: gpus)
    monkeypatch.setattr(
        module,
        "_gpu_compute_processes",
        lambda: (
            [
                {
                    "gpu_uuid": registry[0]["physical_gpu_uuid"],
                    "pid": 9999,
                    "process_name": "foreign",
                }
            ]
            if case == "unknown_pid"
            else []
        ),
    )
    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "release.jsonl", tmp_path, registry
    )
    if case == "thread_failure":
        guard._thread_failure = RuntimeError("injected guard failure")
    published = False

    def publisher(*_args):
        nonlocal published
        published = True

    with pytest.raises(CanonicalScreeningError, match=message):
        guard.release_worker_after_handshake(
            5151, lambda: {"validated": True}, publisher
        )
    assert published is False
    assert 5151 not in guard._active_worker_pids


def test_worker_release_pid_tamper_fails_before_heavy_import(
    tmp_path: Path,
) -> None:
    policy, request = _run_fixture(tmp_path, mode="screen512")
    handshake = _handshake_fixture(
        policy, request, worker_pid=os.getpid()
    )
    ready = handshake["worker_ready"]
    ready_binding = handshake["worker_ready_binding"]
    release = json.loads(json.dumps(handshake["worker_release"]))
    release["worker_pid"] = 999999
    release["worker_release_sha256"] = canonical_digest(
        release, "worker_release_sha256"
    )
    release_path = tmp_path / "tampered-worker-release.json"
    write_exclusive_json(release_path, release)
    with pytest.raises(CanonicalScreeningError, match="contract mismatch"):
        _wait_worker_release(
            release_path,
            ready,
            ready_binding,
            request,
            policy,
            timeout_seconds=0.01,
        )


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "contract_type",
        "policy_sha256",
        "phase",
        "worker_pid",
        "gpu_index",
        "gpu_uuid",
        "run_request_sha256",
        "request",
        "final_release",
        "verification_order",
        "rehashed_bindings",
        "rehashed_bindings_sha256",
        "controller_claim",
        "screening_worker_sha256",
        "controller_implementation_sha256",
        "cuda_visible_devices",
        "heavy_modules_absent",
        "loaded_heavy_modules",
        "asset_content_verification",
        "external_gpu_race_contract",
        "ready_at",
        "worker_ready_sha256",
    ),
)
def test_worker_ready_rejects_each_missing_field(
    tmp_path: Path, field: str
) -> None:
    policy, request = _run_fixture(tmp_path)
    ready = json.loads(
        json.dumps(_handshake_fixture(policy, request)["worker_ready"])
    )
    ready.pop(field)
    if field != "worker_ready_sha256":
        ready["worker_ready_sha256"] = canonical_digest(
            ready, "worker_ready_sha256"
        )
    with pytest.raises(CanonicalScreeningError):
        validate_worker_ready_value(ready, request, policy)


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "contract_type",
        "policy_sha256",
        "run_request_sha256",
        "worker_pid",
        "gpu_index",
        "gpu_uuid",
        "worker_ready",
        "verification_order",
        "rehashed_bindings",
        "rehashed_bindings_sha256",
        "resource_snapshot",
        "asset_content_verification",
        "external_gpu_race_contract",
        "validated_at",
        "controller_launch_rehash_sha256",
    ),
)
def test_controller_rehash_rejects_each_missing_field(
    tmp_path: Path, field: str
) -> None:
    policy, request = _run_fixture(tmp_path)
    value = json.loads(
        json.dumps(
            _handshake_fixture(policy, request)["controller_rehash"]
        )
    )
    value.pop(field)
    if field != "controller_launch_rehash_sha256":
        value["controller_launch_rehash_sha256"] = canonical_digest(
            value, "controller_launch_rehash_sha256"
        )
    with pytest.raises(CanonicalScreeningError):
        validate_controller_launch_rehash_value(value, request, policy)


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "contract_type",
        "policy_sha256",
        "phase",
        "worker_pid",
        "run_request_sha256",
        "worker_ready",
        "controller_launch_rehash",
        "resource_snapshot",
        "external_gpu_race_contract",
        "released_at",
        "worker_release_sha256",
    ),
)
def test_worker_release_rejects_each_missing_field(
    tmp_path: Path, field: str
) -> None:
    policy, request = _run_fixture(tmp_path)
    value = json.loads(
        json.dumps(_handshake_fixture(policy, request)["worker_release"])
    )
    value.pop(field)
    if field != "worker_release_sha256":
        value["worker_release_sha256"] = canonical_digest(
            value, "worker_release_sha256"
        )
    with pytest.raises(CanonicalScreeningError):
        validate_worker_release_value(value, request, policy)


@pytest.mark.parametrize(
    "mutation",
    (
        "heavy_false",
        "loaded_torch",
        "gpu_uuid",
        "external_race",
        "nested_request",
        "asset_digest",
        "extra_field",
    ),
)
def test_worker_ready_rejects_semantic_tamper(
    tmp_path: Path, mutation: str
) -> None:
    policy, request = _run_fixture(tmp_path)
    value = json.loads(
        json.dumps(_handshake_fixture(policy, request)["worker_ready"])
    )
    if mutation == "heavy_false":
        value["heavy_modules_absent"] = False
    elif mutation == "loaded_torch":
        value["loaded_heavy_modules"] = ["torch"]
    elif mutation == "gpu_uuid":
        value["gpu_uuid"] = _gpu_uuid(1)
    elif mutation == "external_race":
        value["external_gpu_race_contract"]["compute_mode_changed"] = True
    elif mutation == "nested_request":
        value["rehashed_bindings"]["request"]["sha256"] = "f" * 64
        value["rehashed_bindings_sha256"] = hashlib.sha256(
            canonical_json(value["rehashed_bindings"])
        ).hexdigest()
    elif mutation == "asset_digest":
        value["asset_content_verification"]["observed_digest"] = "f" * 64
    else:
        value["unexpected"] = True
    value["worker_ready_sha256"] = canonical_digest(
        value, "worker_ready_sha256"
    )
    with pytest.raises(CanonicalScreeningError):
        validate_worker_ready_value(value, request, policy)


@pytest.mark.parametrize(
    "mutation",
    (
        "ready_rehash",
        "admission",
        "runtime_registry",
        "unknown_pid",
        "active_pid",
        "cpu_limit",
        "swap_delta",
        "gpu_memory",
        "gpu_headroom",
        "gpu_temperature",
        "guard_failure",
        "external_race",
        "extra_field",
    ),
)
def test_worker_release_rejects_semantic_tamper(
    tmp_path: Path, mutation: str
) -> None:
    policy, request = _run_fixture(tmp_path)
    handshake = _handshake_fixture(policy, request)
    value = json.loads(json.dumps(handshake["worker_release"]))
    if mutation == "ready_rehash":
        rehash = json.loads(json.dumps(handshake["controller_rehash"]))
        rehash["rehashed_bindings_sha256"] = "e" * 64
        rehash["controller_launch_rehash_sha256"] = canonical_digest(
            rehash, "controller_launch_rehash_sha256"
        )
        path = tmp_path / "tampered-controller-rehash.json"
        write_exclusive_json(path, rehash)
        value["controller_launch_rehash"] = {
            **_bound(path),
            "canonical_sha256": rehash[
                "controller_launch_rehash_sha256"
            ],
        }
    elif mutation == "admission":
        value["resource_snapshot"]["admission"]["cpu_load_percent"] = 4.0
    elif mutation == "runtime_registry":
        value["resource_snapshot"]["runtime_guard"][
            "runtime_gpu_registry"
        ] = list(reversed(request["authorized_gpu_registry"]))
    elif mutation == "unknown_pid":
        value["resource_snapshot"]["runtime_guard"][
            "unknown_compute_processes"
        ] = [{"pid": 999}]
    elif mutation == "active_pid":
        value["resource_snapshot"]["runtime_guard"][
            "active_worker_pids"
        ] = []
    elif mutation == "cpu_limit":
        value["resource_snapshot"]["runtime_guard"][
            "cpu_load_percent"
        ] = 90.0
    elif mutation == "swap_delta":
        value["resource_snapshot"]["runtime_guard"][
            "swap_io_delta"
        ]["in"] = 1
    elif mutation == "gpu_memory":
        gpu = value["resource_snapshot"]["runtime_guard"]["gpu"][0]
        gpu["memory_used_mib"] = 23000
        gpu["memory_free_mib"] = 1576
    elif mutation == "gpu_headroom":
        gpu = value["resource_snapshot"]["runtime_guard"]["gpu"][0]
        gpu["memory_used_mib"] = 23000
        gpu["memory_free_mib"] = 1576
    elif mutation == "gpu_temperature":
        value["resource_snapshot"]["runtime_guard"]["gpu"][0][
            "temperature_c"
        ] = 86
    elif mutation == "guard_failure":
        value["resource_snapshot"]["runtime_guard"][
            "guard_thread_failure"
        ] = {"type": "RuntimeError"}
    elif mutation == "external_race":
        value["external_gpu_race_contract"]["compute_mode_changed"] = True
    else:
        value["unexpected"] = True
    value["worker_release_sha256"] = canonical_digest(
        value, "worker_release_sha256"
    )
    with pytest.raises(CanonicalScreeningError):
        validate_worker_release_value(value, request, policy)


def test_real_cpu_subprocess_completes_production_pre_cuda_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy, config, request, request_path = _real_policy_run_fixture(
        tmp_path, module
    )
    final_release = _final_release_for_single_request(
        tmp_path, policy, request, request_path
    )
    gpu_index = 0
    gpu_uuid = request["authorized_gpu_registry"][0][
        "physical_gpu_uuid"
    ]
    ready_path = tmp_path / "real-handshake/worker_ready.json"
    release_path = tmp_path / "real-handshake/worker_release.json"
    controller_rehash_path = (
        tmp_path / "real-handshake/controller_rehash.json"
    )
    gpu_rows = [
        {
            "index": row["physical_gpu_index"],
            "uuid": row["physical_gpu_uuid"],
            "memory_total_mib": 24576,
            "memory_used_mib": 3,
            "memory_free_mib": 24573,
            "temperature_c": 35,
        }
        for row in request["authorized_gpu_registry"]
    ]
    controller_resources = {
        "observed_at": "2026-07-27T00:00:00+00:00",
        "cpu_load_percent": 1.0,
        "memory_percent": 2.0,
        "disk_percent": 3.0,
        "swap_pages": {"in": 0, "out": 0},
        "gpus": gpu_rows,
        "authorized_gpu_registry": request[
            "authorized_gpu_registry"
        ],
        "ram_reservation": _admission_snapshot(policy)[
            "ram_reservation"
        ],
        "compute_processes": [],
    }
    monkeypatch.setattr(
        module,
        "assert_resource_admission",
        lambda *_args, **_kwargs: json.loads(
            json.dumps(controller_resources)
        ),
    )
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 1.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 2.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 3.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (0, 0))
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: json.loads(json.dumps(gpu_rows)),
    )
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    child_code = """
import json
from pathlib import Path
import sys
from safa.closeout.canonical_screening import load_json, validate_policy
from safa.closeout.canonical_screening_worker import (
    HEAVY_MODULE_ROOTS,
    prepare_screening_request_for_cuda,
)
repo_root = Path(sys.argv[1]).resolve()
config = Path(sys.argv[2]).resolve()
request_path = Path(sys.argv[3]).resolve()
final_release_path = Path(sys.argv[4]).resolve()
ready_path = Path(sys.argv[5]).resolve()
release_path = Path(sys.argv[6]).resolve()
gpu_index = int(sys.argv[7])
gpu_uuid = sys.argv[8]
policy = validate_policy(
    repo_root, config, verify_historical_output_evidence=False
)
prepared = prepare_screening_request_for_cuda(
    request_path,
    gpu_index,
    gpu_uuid,
    policy,
    {
        "path": str(final_release_path),
        "sha256": __import__("hashlib").sha256(
            final_release_path.read_bytes()
        ).hexdigest(),
        "canonical_sha256": load_json(
            final_release_path, "child final release"
        )["final_release_admission_sha256"],
    },
    ready_path,
    release_path,
)
print(json.dumps({
    "next_stage": prepared["next_stage"],
    "worker_ready_sha256": prepared["worker_ready"]["worker_ready_sha256"],
    "worker_release_sha256": prepared["worker_release"]["worker_release_sha256"],
    "pre_asset_digest": prepared["pre_cuda"]["asset_content_verification"][
        "observed_digest"
    ],
    "post_asset_digest": prepared["post_release"][
        "asset_content_verification"
    ]["observed_digest"],
    "heavy_modules": sorted(
        name for name in HEAVY_MODULE_ROOTS if name in sys.modules
    ),
}, sort_keys=True))
"""
    child_env = {
        **os.environ,
        "TMUX": "canonical-pre-cuda-integration",
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    child_env.pop("CUDA_DEVICE_ORDER", None)
    guard = module.RuntimeResourceGuard(
        policy,
        tmp_path / "real-handshake/resource.jsonl",
        tmp_path,
        request["authorized_gpu_registry"],
    )
    process = None
    try:
        process, initial_rehash = guard.launch_cpu_worker(
            lambda: subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    child_code,
                    str(Path(__file__).parents[1]),
                    str(config),
                    str(request_path),
                    str(final_release["path"]),
                    str(ready_path),
                    str(release_path),
                    str(gpu_index),
                    gpu_uuid,
                ],
                cwd=Path(__file__).parents[1],
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ),
            lambda: module._validate_launch_integrity(
                policy,
                config,
                module._paths(
                    tmp_path / "real-campaign",
                    policy["policy_sha256"],
                ),
                request_path,
                final_release,
            ),
        )
        assert initial_rehash["worker_pid"] is None
        ready, ready_binding = module._wait_worker_ready(
            process,
            ready_path,
            request_path,
            request,
            policy,
            gpu_index,
            gpu_uuid,
            timeout_seconds=120.0,
        )
        final_rehash = None

        def publish_release(validation, guard_snapshot):
            nonlocal final_rehash
            final_rehash = dict(validation)
            validate_controller_launch_rehash_value(
                final_rehash, request, policy
            )
            publish_exclusive_json(
                controller_rehash_path, final_rehash
            )
            controller_rehash_binding = {
                **_bound(controller_rehash_path),
                "canonical_sha256": final_rehash[
                    "controller_launch_rehash_sha256"
                ],
            }
            module._publish_worker_release(
                release_path,
                policy,
                request,
                process.pid,
                ready_binding,
                controller_rehash_binding,
                {
                    "admission": final_rehash[
                        "resource_snapshot"
                    ],
                    "runtime_guard": dict(guard_snapshot),
                },
            )

        guard.release_worker_after_handshake(
            process.pid,
            lambda: module._validate_launch_integrity(
                policy,
                config,
                module._paths(
                    tmp_path / "real-campaign",
                    policy["policy_sha256"],
                ),
                request_path,
                final_release,
                worker_pid=process.pid,
                gpu_index=gpu_index,
                gpu_uuid=gpu_uuid,
                worker_ready=ready_binding,
            ),
            publish_release,
        )
        stdout, stderr = process.communicate(timeout=120.0)
        assert process.returncode == 0, stderr
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.pid in guard._active_worker_pids:
                guard.unregister_worker_pid(process.pid)
    payload = json.loads(stdout)
    release = validate_worker_release_value(
        load_json(release_path, "real worker release"),
        request,
        policy,
        expected_worker_pid=process.pid,
    )
    assert final_rehash is not None
    assert payload["next_stage"] == "runtime_cuda_binding"
    assert payload["heavy_modules"] == []
    assert payload["worker_ready_sha256"] == ready[
        "worker_ready_sha256"
    ]
    assert payload["worker_release_sha256"] == release[
        "worker_release_sha256"
    ]
    expected_asset_digest = policy["output_decoder_registry"]["latent"][
        "directory"
    ]["digest"]
    assert payload["pre_asset_digest"] == expected_asset_digest
    assert payload["post_asset_digest"] == expected_asset_digest
    controller_ready = load_json(
        Path(request["controller_ready"]["path"]),
        "real controller ready",
    )
    assert ready["controller_claim"] == controller_ready[
        "controller_claim"
    ]
    assert release["controller_launch_rehash"][
        "canonical_sha256"
    ] == final_rehash["controller_launch_rehash_sha256"]
    output_dir = Path(request["output_dir"])
    assert not output_dir.exists()
    assert not (output_dir / "claim.json").exists()


def test_malformed_release_still_writes_atomic_worker_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    write_exclusive_json(request_path, {"request": "fixture"})
    ready_path = tmp_path / "handshake" / "worker_ready.json"
    release_path = tmp_path / "handshake" / "worker_release.json"
    write_exclusive_json(
        ready_path,
        {"worker_ready_sha256": "a" * 64},
    )
    write_exclusive_json(release_path, {"malformed": True})

    def reject_malformed(*_args, **_kwargs):
        release = load_json(release_path, "malformed release")
        if "worker_release_sha256" not in release:
            raise CanonicalScreeningError(
                "malformed worker release is missing its digest"
            )
        raise AssertionError("malformed release unexpectedly passed")

    monkeypatch.setattr(
        screening_worker_module,
        "_execute_screening_request_impl",
        reject_malformed,
    )
    with pytest.raises(CanonicalScreeningError, match="malformed"):
        execute_screening_request(
            request_path,
            0,
            _gpu_uuid(0),
            {"policy_sha256": "b" * 64},
            {},
            ready_path,
            release_path,
        )
    terminal_path = ready_path.parent / "worker_terminal.json"
    terminal = load_json(terminal_path, "malformed release terminal")
    assert terminal["status"] == "failed"
    assert terminal["failure"]["type"] == "CanonicalScreeningError"
    assert terminal["worker_release"]["canonical_sha256"] is None
    assert terminal["worker_terminal_sha256"] == canonical_digest(
        terminal, "worker_terminal_sha256"
    )
    assert terminal_path.read_bytes().endswith(b"\n")


def test_launch_bootstrap_rejects_invalid_config_before_resource_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy, request = _run_fixture(tmp_path)
    request_path = tmp_path / "bootstrap-request.json"
    write_exclusive_json(request_path, request)
    sampled = False

    def forbidden_sample(*_args, **_kwargs):
        nonlocal sampled
        sampled = True
        raise AssertionError("resource sample must follow rehash")

    monkeypatch.setattr(module, "assert_resource_admission", forbidden_sample)
    with pytest.raises(
        module.ControllerBootstrapError,
        match="omits implementations",
    ):
        module._validate_launch_integrity(
            policy,
            Path(request["policy"]["path"]),
            module._paths(tmp_path / "campaign", policy["policy_sha256"]),
            request_path,
            {},
        )
    assert sampled is False


def test_real_controller_main_cpu_dry_run_uses_current_policy_without_writes(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).parents[1]
    script = repo_root / "scripts/run_canonical_checkpoint_screening.py"
    config = (
        repo_root
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    campaign = tmp_path / "dry-run-campaign"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--phase",
            "plan",
            "--campaign-root",
            str(campaign),
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    payload = json.loads(completed.stdout)
    assert payload["phase"] == "plan"
    assert payload["execute"] is False
    assert payload["policy_sha256"]
    assert not campaign.exists()


def test_real_worker_policy_bootstrap_failure_writes_atomic_terminal(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).parents[1]
    script = repo_root / "scripts/run_canonical_checkpoint_screening.py"
    source_config = (
        repo_root
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    raw = json.loads(source_config.read_text(encoding="utf-8"))
    raw["implementations"]["controller"]["sha256"] = "0" * 64
    config = tmp_path / "tampered-policy.json"
    config.write_bytes(canonical_json(raw))
    request_path = tmp_path / "request.json"
    request_path.write_text('{"malformed":true}\n', encoding="utf-8")
    ready_path = tmp_path / "handshake/worker_ready.json"
    release_path = tmp_path / "handshake/worker_release.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--phase",
            "smoke8",
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--execute",
            "--request",
            str(request_path),
            "--worker-ready-path",
            str(ready_path),
            "--worker-release-path",
            str(release_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    terminal_path = (
        ready_path.parent / "worker_bootstrap_terminal.json"
    )
    terminal = load_json(terminal_path, "worker bootstrap terminal")
    assert terminal["stage"] == "policy_bootstrap"
    assert terminal["status"] == "failed"
    assert terminal["worker_pid"] > 0
    assert terminal["request"] == {
        "observed_path": str(request_path.resolve()),
        "observed_sha256": hashlib.sha256(
            request_path.read_bytes()
        ).hexdigest(),
        "observed_canonical_sha256": None,
    }
    assert terminal["worker_bootstrap_terminal_sha256"] == canonical_digest(
        terminal, "worker_bootstrap_terminal_sha256"
    )
    assert not ready_path.exists()
    assert not release_path.exists()
    assert not (ready_path.parent / "worker_terminal.json").exists()


def test_preflight_monitor_never_queries_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 10.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 20.0)
    monkeypatch.setattr(module, "_disk_percent", lambda _path: 30.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (1, 2))
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: (_ for _ in ()).throw(AssertionError("GPU must not be queried")),
    )
    monkeypatch.setattr(
        module,
        "_gpu_compute_processes",
        lambda: (_ for _ in ()).throw(AssertionError("GPU must not be queried")),
    )
    sample = module._monitor_sample(
        policy, paths, "preflight", terminal=False
    )
    assert sample["gpus"] is None
    assert sample["compute_processes"] is None


def test_supersession_evidence_binds_ea7_failed_smoke_chain(
    tmp_path: Path,
) -> None:
    old_policy = (
        "ea7ae71fd662526b9a45bf3cc6d283884"
        "aefc380b292c8f273169a35f42ffc28"
    )
    policy_root = (
        tmp_path
        / "artifacts/closeout/historical-canonical-512-v1/by_policy"
        / old_policy
    )
    primary_requests = policy_root / "run_requests/smoke8_primary"
    repeat_requests = policy_root / "run_requests/smoke8_repeat"
    run_requests = []
    run_claims = []
    failed_results = []
    worker_logs = []
    failure_message = (
        "The size of tensor a (4) must match the size of tensor b (3) "
        "at non-singleton dimension 1"
    )
    for index in range(193):
        candidate_id = f"g_{index:016x}_raw"
        request_path = primary_requests / f"{candidate_id}.json"
        if index < 8:
            request = {
                "contract_type": "safa_canonical_screening_run_request_v1",
                "mode": "smoke8",
                "replicate": "primary",
                "sample_count": 8,
                "batch_size": 2,
                "seed": 4549,
                "policy": {"canonical_sha256": old_policy},
                "candidate": {"candidate_id": candidate_id},
            }
            request["run_request_sha256"] = canonical_digest(
                request, "run_request_sha256"
            )
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_bytes(canonical_json(request))
            run_dir = policy_root / "runs/smoke8_primary" / candidate_id
            claim = {
                "contract_type": "safa_canonical_screening_run_claim_v1",
                "run_request_sha256": request["run_request_sha256"],
            }
            claim["run_claim_sha256"] = canonical_digest(
                claim, "run_claim_sha256"
            )
            claim_path = run_dir / "claim.json"
            claim_path.parent.mkdir(parents=True, exist_ok=True)
            claim_path.write_bytes(canonical_json(claim))
            result = {
                "contract_type": "safa_canonical_screening_run_result_v1",
                "run_request_sha256": request["run_request_sha256"],
                "run_claim_sha256": claim["run_claim_sha256"],
                "status": "failed",
                "failure": {
                    "type": "RuntimeError",
                    "message": failure_message,
                },
            }
            result["run_result_sha256"] = canonical_digest(
                result, "run_result_sha256"
            )
            result_path = run_dir / "result.json"
            result_path.write_bytes(canonical_json(result))
            log_path = policy_root / "logs" / f"smoke8_primary__{candidate_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(failure_message, encoding="utf-8")
            run_requests.append(_bound(request_path))
            run_claims.append(_bound(claim_path))
            failed_results.append(_bound(result_path))
            worker_logs.append(_bound(log_path))
        else:
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text("{}\n", encoding="utf-8")
        repeat_path = repeat_requests / f"{candidate_id}.json"
        repeat_path.parent.mkdir(parents=True, exist_ok=True)
        repeat_path.write_text("{}\n", encoding="utf-8")
    monitor = _bound(policy_root / "logs/smoke8__monitor.jsonl")
    runtime = _bound(policy_root / "logs/smoke8__runtime_resource_windows.jsonl")
    summary_path = policy_root / "summaries/smoke8__failed.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes(
        canonical_json(
            {
                "phase": "smoke8",
                "reason": "worker_nonzero_exit",
                "failures": [
                    f"{binding['path']}: exit_code=1"
                    for binding in run_requests
                ],
                "monitor_log": monitor,
                "runtime_resource_guard": {
                    "samples": runtime,
                    "violated": False,
                    "violation_reason": None,
                    "thread_failure": None,
                    "final_cpu_consecutive_high": 0,
                    "final_swap_consecutive_io": 0,
                },
            }
        )
    )
    supersedes = {
        "policy_sha256": old_policy,
        "classification": "started_incomplete",
        "phase": "smoke8",
        "request_count": 386,
        "primary_failed_count": 8,
        "repeat_result_count": 0,
        "screen512_result_count": 0,
        "generated_png_count": 0,
        "failed_summary": _bound(summary_path),
        "run_requests": run_requests,
        "run_claims": run_claims,
        "failed_results": failed_results,
        "worker_logs": worker_logs,
        "resource_monitor": monitor,
        "runtime_resource_windows": runtime,
    }
    assert (
        validate_supersession_evidence(tmp_path, supersedes)["classification"]
        == "started_incomplete"
    )
    tampered = json.loads(json.dumps(supersedes))
    tampered["run_claims"][0]["sha256"] = "f" * 64
    with pytest.raises(CanonicalScreeningError, match="SHA256 mismatch"):
        validate_supersession_evidence(tmp_path, tampered)


def test_current_arcface_binding_reaches_official_validator_and_probe_request() -> None:
    root = Path(__file__).parents[1]
    policy_path = root / "configs/closeout/canonical_screening_512_v1.json"
    policy = validate_policy(root, policy_path)
    expected_fields = {
        "path",
        "sha256",
        "bootstrap_claim_path",
        "bootstrap_claim_sha256",
        "bootstrap_claim_file_sha256",
        "bootstrap_result_path",
        "bootstrap_result_sha256",
        "bootstrap_result_file_sha256",
    }
    assert set(policy["arcface"]["execution_probe"]) == expected_fields
    contract = _load_arcface_contract(
        {
            "arcface": policy["arcface"],
            "source_index": policy["protocol"]["source_index"],
        }
    )
    assert set(contract["execution_probe"]) == expected_fields

    module = _ram_probe_module()
    manifest_path = (
        root
        / "artifacts/closeout/historical-canonical-512-v1/"
        "candidate_manifest__4c5ecb55501fa6b0.json"
    )
    manifest = load_json(manifest_path, "4c5 candidate manifest")
    request = module._probe_request(
        policy,
        manifest,
        manifest["candidates"][0],
        {"probe_execution_sha256": "e" * 64},
    )
    assert set(request["arcface"]["execution_probe"]) == expected_fields


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "probe_file_sha",
        "claim_canonical_sha",
        "claim_file_sha",
        "result_canonical_sha",
        "result_file_sha",
        "path_escape",
    ],
)
def test_arcface_execution_probe_binding_tamper_fails_closed(
    mutation: str,
) -> None:
    root = Path(__file__).parents[1]
    raw = load_json(
        root / "configs/closeout/canonical_screening_512_v1.json",
        "current policy",
    )
    arcface = json.loads(json.dumps(raw["arcface"]))
    binding = arcface["execution_probe"]
    if mutation == "missing":
        binding.pop("bootstrap_claim_sha256")
    elif mutation == "extra":
        binding["unexpected"] = "x"
    elif mutation == "probe_file_sha":
        binding["sha256"] = "0" * 64
    elif mutation == "claim_canonical_sha":
        binding["bootstrap_claim_sha256"] = "0" * 64
    elif mutation == "claim_file_sha":
        binding["bootstrap_claim_file_sha256"] = "0" * 64
    elif mutation == "result_canonical_sha":
        binding["bootstrap_result_sha256"] = "0" * 64
    elif mutation == "result_file_sha":
        binding["bootstrap_result_file_sha256"] = "0" * 64
    elif mutation == "path_escape":
        binding["bootstrap_claim_path"] = "../outside.json"
    else:
        raise AssertionError(mutation)
    with pytest.raises(CanonicalScreeningError):
        validate_arcface_execution_probe_binding(
            root, binding, arcface_contract=arcface
        )


def test_arcface_binding_rejects_ancestor_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "probe.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(CanonicalScreeningError, match="must not be symlinks"):
        _require_no_repo_path_component_symlinks(
            tmp_path, "alias/probe.json", "ArcFace probe"
        )


def test_arcface_binding_delegates_coherent_semantic_tamper(
    tmp_path: Path,
) -> None:
    from safa.evaluation.r9_evaluator_worker import _canonical_digest

    root = Path(__file__).parents[1]
    raw = load_json(
        root / "configs/closeout/canonical_screening_512_v1.json",
        "current policy",
    )
    arcface = json.loads(json.dumps(raw["arcface"]))
    original = arcface["execution_probe"]
    probe = tmp_path / "probe.json"
    claim_path = tmp_path / "claim.json"
    result_path = tmp_path / "result.json"
    probe.write_bytes((root / original["path"]).read_bytes())
    claim = load_json(root / original["bootstrap_claim_path"], "claim")
    claim["kind"] = "not_arcface_profile"
    claim["probe_output"] = str(probe.resolve())
    claim["bootstrap_claim_sha256"] = _canonical_digest(
        claim, "bootstrap_claim_sha256"
    )
    write_exclusive_json(claim_path, claim)
    result = load_json(root / original["bootstrap_result_path"], "result")
    result["bootstrap_claim_sha256"] = claim["bootstrap_claim_sha256"]
    result["bootstrap_result_sha256"] = _canonical_digest(
        result, "bootstrap_result_sha256"
    )
    write_exclusive_json(result_path, result)
    arcface["execution_probe"] = {
        "path": str(probe.resolve()),
        "sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        "bootstrap_claim_path": str(claim_path.resolve()),
        "bootstrap_claim_sha256": claim["bootstrap_claim_sha256"],
        "bootstrap_claim_file_sha256": hashlib.sha256(
            claim_path.read_bytes()
        ).hexdigest(),
        "bootstrap_result_path": str(result_path.resolve()),
        "bootstrap_result_sha256": result["bootstrap_result_sha256"],
        "bootstrap_result_file_sha256": hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest(),
    }
    with pytest.raises(CanonicalScreeningError, match="claim policy mismatch"):
        validate_arcface_execution_probe_binding(
            tmp_path,
            arcface["execution_probe"],
            arcface_contract=arcface,
        )


def test_supersession_tree_rejects_recursive_symlink(
    tmp_path: Path,
) -> None:
    (tmp_path / "root/nested").mkdir(parents=True)
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    (tmp_path / "root/nested/alias").symlink_to(target)
    with pytest.raises(CanonicalScreeningError, match="must not contain symlinks"):
        _require_tree_without_symlinks(
            tmp_path / "root", "4c5 failed root"
        )


def test_worker_rejects_truncated_arcface_probe_request() -> None:
    from safa.evaluation.r9_evaluator_worker import R9EvaluatorError

    root = Path(__file__).parents[1]
    policy = validate_policy(
        root, root / "configs/closeout/canonical_screening_512_v1.json"
    )
    arcface = json.loads(json.dumps(policy["arcface"]))
    arcface["execution_probe"].pop("bootstrap_result_file_sha256")
    with pytest.raises(
        R9EvaluatorError,
        match="provenance fields are not canonical",
    ):
        _load_arcface_contract(
            {
                "arcface": arcface,
                "source_index": policy["protocol"]["source_index"],
            }
        )


def test_current_policy_binds_9300_zero_result_preflight_supersession() -> None:
    root = Path(__file__).parents[1]
    policy_path = root / "configs/closeout/canonical_screening_512_v1.json"

    policy = validate_policy(root, policy_path)

    supersedes = policy["supersedes"]
    assert supersedes["policy_sha256"] == (
        "9300a01c5f308840918dca8717f06bd6684e3a52967478950b5a9146b8f62508"
    )
    assert supersedes["previous_policy_sha256"] == (
        "5dbb82fdb1c89d8f7afd463a2f0b40743f42abd7b0f07dcefab144a32787c7af"
    )
    assert (
        supersedes["classification"]
        == "prepared_execution_barrier_not_crossed_superseded"
    )
    assert supersedes["counts"]["preflight_request_count"] == 193
    assert supersedes["counts"]["preflight_result_count"] == 0
    assert supersedes["counts"]["controller_artifact_count"] == 0
    assert supersedes["counts"]["generated_png_count"] == 0
    assert supersedes["absence_evidence"]["preflight_control"] == "absent"
    assert (
        supersedes["absence_evidence"]["preflight_request_manifest"]
        == "absent"
    )
    assert supersedes["scientific_result_reuse"] == "forbidden"
    assert supersedes["successor_execution"] == "fresh_full_193_preflight"
    assert supersedes["ram_budget_source_policy_sha256"] == (
        "4d0345b6fc29cc8ec50ddc0255188a466ae78edae2e472fed9deda461cf76cbc"
    )
    assert "gpu_wrapper" in policy["implementations"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("request_count", "counts differ"),
        ("reuse", "status differs"),
        ("root_digest", "evidence root differs"),
        ("request_digest", "request set differs"),
        ("absence", "absence status differs"),
        ("successor", "status differs"),
        ("ram_lineage", "status differs"),
    ],
)
def test_9300_zero_result_supersession_tampering_fails_closed(
    mutation: str,
    match: str,
) -> None:
    root = Path(__file__).parents[1]
    raw = load_json(
        root / "configs/closeout/canonical_screening_512_v1.json",
        "current policy",
    )
    supersedes = json.loads(json.dumps(raw["supersedes"]))
    if mutation == "request_count":
        supersedes["counts"]["preflight_request_count"] = 192
    elif mutation == "reuse":
        supersedes["scientific_result_reuse"] = "allowed"
    elif mutation == "root_digest":
        supersedes["evidence_root"]["digest"] = "0" * 64
    elif mutation == "request_digest":
        supersedes["request_set"]["digest"] = "0" * 64
    elif mutation == "absence":
        supersedes["absence_evidence"]["preflight_control"] = "present"
    elif mutation == "successor":
        supersedes["successor_execution"] = "fresh_full_193_preflight_and_smoke8"
    elif mutation == "ram_lineage":
        supersedes["ram_budget_source_policy_sha256"] = "0" * 64
    else:
        raise AssertionError(mutation)

    with pytest.raises(CanonicalScreeningError, match=match):
        validate_supersession_evidence(root, supersedes)


def test_unknown_supersession_policy_sha_fails_closed() -> None:
    with pytest.raises(CanonicalScreeningError, match="unknown supersession"):
        validate_supersession_evidence(
            Path(__file__).parents[1],
            {"policy_sha256": "0" * 64},
        )


def _materialize_observer_ready_binding(
    module: Any,
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, str]:
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_observer_ready_v1",
        "verified_implementations": (
            module._verified_preflight_implementations(
                policy["implementations"]
            )
        ),
    }
    value["observer_ready_sha256"] = canonical_digest(
        value, "observer_ready_sha256"
    )
    path = paths["preflight_control"] / "observer_ready.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_exclusive_json(path, value)
    return module._artifact_binding(
        path, value["observer_ready_sha256"]
    )


def test_preflight_attempt_failure_writes_claim_and_terminal_without_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", "a" * 64)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    plan = build_checkpoint_plan(tmp_path, policy, paths["preflight_results"])
    request = plan["preflight_requests"][0]
    request_path = (
        paths["preflight_requests"]
        / f"{request['checkpoint_sha256']}__{request['checkpoint_model']}.json"
    )
    write_exclusive_json(request_path, request)
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module,
        "assert_cpu_resource_admission",
        lambda *_args: {"admission_kind": "cpu_only"},
    )
    monkeypatch.setattr(
        module,
        "preflight_generator_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    monkeypatch.setattr(
        module, "_assert_preflight_observer_live", lambda *_args: None
    )
    guard = types.SimpleNamespace(raise_if_violated=lambda: None)
    observer_ready = _materialize_observer_ready_binding(
        module, policy, paths
    )
    with pytest.raises(RuntimeError, match="injected"):
        module.materialize_preflights(
            policy, paths, guard, "d" * 64, observer_ready
        )
    attempts = paths["preflight_control"] / "attempts"
    claim = load_json(next(attempts.glob("*.claim.json")), "attempt claim")
    ready = load_json(
        paths["preflight_control"] / "observer_ready.json",
        "materialization observer ready",
    )
    assert (
        claim["verified_implementations"]
        == ready["verified_implementations"]
    )
    terminal = load_json(next(attempts.glob("*.terminal.json")), "attempt terminal")
    assert terminal["attempt_claim_sha256"] == claim["attempt_claim_sha256"]
    assert terminal["status"] == "failed"
    assert terminal["failure"]["type"] == "RuntimeError"
    assert list(paths["preflight_results"].glob("*.json")) == []


def test_runtime_stop_before_result_writes_one_failed_attempt_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", "a" * 64)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    plan = build_checkpoint_plan(tmp_path, policy, paths["preflight_results"])
    request = plan["preflight_requests"][0]
    write_exclusive_json(
        paths["preflight_requests"]
        / f"{request['checkpoint_sha256']}__{request['checkpoint_model']}.json",
        request,
    )
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module, "preflight_generator_checkpoint", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        module, "_assert_preflight_observer_live", lambda *_args: None
    )
    observer_ready = _materialize_observer_ready_binding(
        module, policy, paths
    )

    class Guard:
        def __init__(self) -> None:
            self.calls = 0

        def raise_if_violated(self) -> None:
            self.calls += 1
            if self.calls == 2:
                raise CanonicalScreeningError("CPU runtime hard stop")

    with pytest.raises(CanonicalScreeningError, match="CPU runtime hard stop"):
        module.materialize_preflights(
            policy,
            paths,
            Guard(),
            "d" * 64,
            observer_ready,
        )
    attempts = paths["preflight_control"] / "attempts"
    terminals = list(attempts.glob("*.terminal.json"))
    assert len(terminals) == 1
    terminal = load_json(terminals[0], "attempt terminal")
    assert terminal["status"] == "failed"
    assert terminal["failure"]["message"] == "CPU runtime hard stop"
    assert list(paths["preflight_results"].glob("*.json")) == []


def test_controller_failure_persists_log_and_global_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    admission_calls = 0

    def admit(*_args, **_kwargs):
        nonlocal admission_calls
        admission_calls += 1
        return _admission_snapshot(policy)

    class FakeGuard:
        def __init__(
            self,
            _policy,
            sample_path: Path,
            _disk_path: Path,
            authorized_gpu_registry: list[dict],
        ) -> None:
            self.started = False
            self.sample_path = sample_path
            self.policy_sha256 = _policy["policy_sha256"]
            self.authorized_gpu_registry = authorized_gpu_registry

        def start(self) -> None:
            self.started = True
            sample = {
                "schema_version": 1,
                "contract_type": (
                    "safa_canonical_runtime_resource_window_v1"
                ),
                "policy_sha256": self.policy_sha256,
                "sequence": 1,
                "violated": False,
            }
            sample["resource_window_sha256"] = canonical_digest(
                sample, "resource_window_sha256"
            )
            _write_jsonl(self.sample_path, [sample])

        def wait_first_sample(self, _timeout: float) -> dict:
            return module.load_jsonl(self.sample_path, "resource")[0]

        def stop(self) -> dict:
            return {
                "started": self.started,
                "thread_failure": None,
                "violation_reason": None,
                "violated": False,
                "samples": {
                    "path": str(self.sample_path.resolve()),
                    "sha256": hashlib.sha256(
                        self.sample_path.read_bytes()
                    ).hexdigest(),
                },
            }

    fake_binding = {
        "path": str((tmp_path / "bound.json").resolve()),
        "sha256": "a" * 64,
        "canonical_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        module,
        "_current_tmux_session",
        lambda expected, _label: expected,
    )
    monkeypatch.setattr(
        module,
        "_process_identity",
        lambda pid: {"pid": pid, "pgid": pid, "start_ticks": 1},
    )
    monkeypatch.setattr(
        module,
        "_validate_preflight_wrapper_provenance",
        lambda *_args: (
            {"checkpoint_plan": fake_binding},
            fake_binding,
            {"request_count": 0},
            fake_binding,
            fake_binding,
            fake_binding,
        ),
    )
    monkeypatch.setattr(module, "assert_resource_admission", admit)
    monkeypatch.setattr(module, "RuntimeResourceGuard", FakeGuard)
    monkeypatch.setattr(
        module,
        "_wait_preflight_observer_ready",
        lambda *_args: (
            {"observer_ready_sha256": "c" * 64},
            fake_binding,
        ),
    )
    monkeypatch.setattr(
        module, "_assert_preflight_observer_live", lambda *_args: None
    )
    monkeypatch.setattr(
        module,
        "materialize_preflights",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("controller injected")),
    )
    with pytest.raises(RuntimeError, match="controller injected"):
        module._execute_preflight_controller(policy, paths)
    control = paths["preflight_control"]
    terminal = load_json(control / "controller_terminal.json", "controller terminal")
    assert admission_calls == 1
    assert terminal["status"] == "failed"
    assert terminal["failure"]["message"] == "controller injected"
    assert terminal["runtime_resource_guard"]["started"] is True
    samples = terminal["runtime_resource_guard"]["samples"]
    assert Path(samples["path"]).is_file()
    assert samples["sha256"] == hashlib.sha256(
        Path(samples["path"]).read_bytes()
    ).hexdigest()
    assert "controller_exception" in (control / "controller.log").read_text(
        encoding="utf-8"
    )


def _prepare_wrapper_contract_inputs(wrapper, policy_root: Path) -> None:
    missing = []
    trusted_root = policy_root
    while not trusted_root.exists():
        missing.append(trusted_root.name)
        trusted_root = trusted_root.parent
    wrapper._ensure_secure_leaf_directories(
        trusted_root, tuple(reversed(missing))
    )
    wrapper._ensure_secure_leaf_directories(
        policy_root, ("checkpoint_preflight",)
    )
    plan = {"schema_version": 1}
    plan["checkpoint_plan_sha256"] = wrapper._canonical_digest(
        plan, "checkpoint_plan_sha256"
    )
    wrapper._write_exclusive(policy_root / "checkpoint_plan.json", plan)
    manifest = {"schema_version": 1}
    manifest[
        "preflight_request_manifest_sha256"
    ] = wrapper._canonical_digest(
        manifest, "preflight_request_manifest_sha256"
    )
    wrapper._write_exclusive(
        policy_root
        / "checkpoint_preflight"
        / "preflight_request_manifest.json",
        manifest,
    )


def _patch_wrapper_tmux(
    wrapper, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate_supervisor_pid = os.getpid() + 100000
    gate_pid = os.getpid() + 100001
    fake_terminal = tmp_path / "fake_observer_terminal.json"
    fake_terminal.write_text(
        json.dumps({"status": "completed"}) + "\n", encoding="utf-8"
    )
    fake_terminal_binding = {
        "path": str(fake_terminal),
        "sha256": hashlib.sha256(fake_terminal.read_bytes()).hexdigest(),
        "canonical_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        wrapper, "_tmux_session", lambda: wrapper.CONTROLLER_SESSION
    )
    monkeypatch.setattr(
        wrapper,
        "_tmux_identity",
        lambda session: {
            "session": session,
            "pane": (
                "%0" if session == wrapper.CONTROLLER_SESSION else "%1"
            ),
            "pane_pid": (
                gate_supervisor_pid
                if session == wrapper.CONTROLLER_SESSION
                else os.getpid()
            ),
            "pane_current_command": "python",
        },
    )
    monkeypatch.setattr(
        wrapper,
        "_tmux_pane_identity",
        lambda pane: {
            "session": (
                wrapper.CONTROLLER_SESSION
                if pane == "%0"
                else wrapper.OBSERVER_SESSION
            ),
            "pane": pane,
            "pane_pid": (
                gate_supervisor_pid
                if pane == "%0"
                else os.getpid()
            ),
            "pane_current_command": "python",
        },
    )
    monkeypatch.setattr(
        wrapper,
        "_tmux_server_identity",
        lambda _target=None: _test_tmux_server_identity(
            tmp_path / "tmux.sock",
            server_pid=os.getpid(),
            server_process=fixture_process,
        ),
    )
    fixture_tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%1",
        "pane_pid": os.getpid(),
        "pane_current_command": "python",
    }
    fixture_process = wrapper._require_process_identity(
        os.getpid(), "fixture"
    )
    fixture_process["ppid"] = gate_pid
    fixture_process["pgid"] = fixture_process["pid"]
    fixture_process["sid"] = fixture_process["pid"]
    fixture_server = _test_tmux_server_identity(
        tmp_path / "tmux.sock",
        server_pid=os.getpid(),
        server_process=fixture_process,
    )
    real_process_identity = wrapper._process_identity
    real_read_process_stat = wrapper._read_process_stat
    monkeypatch.setattr(
        wrapper,
        "_process_identity",
        lambda pid: (
            dict(fixture_process)
            if pid == fixture_process["pid"]
            else _test_process_identity(
                gate_pid,
                ppid=gate_supervisor_pid,
                start_ticks=77,
            )
            if pid == gate_pid
            else _test_process_identity(
                gate_supervisor_pid, start_ticks=76
            )
            if pid == gate_supervisor_pid
            else real_process_identity(pid)
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_read_process_stat",
        lambda pid: (
            (dict(fixture_process), "S")
            if pid == fixture_process["pid"]
            else real_read_process_stat(pid)
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_build_tmux_owner_seal",
        lambda tmux, server, owner_nonce: _test_tmux_owner_seal(
            tmux,
            server,
            owner_nonce=owner_nonce,
            server_start_ticks=fixture_process["start_ticks"],
            pane_process=fixture_process,
        ),
    )
    launch_binding = dict(fake_terminal_binding)
    wrapper_arguments = wrapper._process_command(os.getpid())
    wrapper_executable = str(
        Path(os.readlink(f"/proc/{os.getpid()}/exe")).resolve()
    )
    fixture_launch = _test_validated_preflight_launch(
        tmp_path=tmp_path,
        attempt_id="f" * 64,
        receipt_binding=launch_binding,
        receipt_identity=_test_file_identity(
            Path(launch_binding["path"])
        ),
        gate_ready_binding=launch_binding,
        tmux_started_binding=launch_binding,
        wrapper_started_binding=launch_binding,
        gate_supervisor_process=_test_process_identity(
            gate_supervisor_pid, start_ticks=76
        ),
        gate_process=_test_process_identity(
            gate_pid,
            ppid=gate_supervisor_pid,
            start_ticks=77,
        ),
        wrapper_process=_test_process_identity(
            os.getpid(),
            ppid=gate_pid,
            start_ticks=fixture_process["start_ticks"],
        ),
        wrapper_arguments=wrapper_arguments,
        wrapper_executable=_test_executable_identity(
            wrapper_executable
        ),
        pane_log=_test_file_identity(
            tmp_path / "fixture-pane.log"
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_validate_preflight_launch_receipt",
        lambda **_kwargs: dict(fixture_launch),
    )
    monkeypatch.setattr(
        wrapper,
        "_validate_pane_fault_consumer_runtime",
        lambda **_kwargs: {
            "chain": dict(
                fixture_launch["pane_fault_consumer_chain"]
            )
        },
    )
    monkeypatch.setattr(
        wrapper,
        "_wait_preflight_launch_release",
        lambda **_kwargs: None,
    )

    def fake_wait_identity(
        _session,
        owner_nonce,
        bootstrap_path,
        **wait_kwargs,
    ):
        bootstrap = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_observer_bootstrap_v1"
            ),
            "policy_sha256": wait_kwargs["policy_sha256"],
            "verified_implementations": (
                wrapper._reverify_verified_preflight_apis()
            ),
            "wrapper_claim": dict(wait_kwargs["wrapper_binding"]),
            "observer_session": wrapper.OBSERVER_SESSION,
            "owner_nonce": owner_nonce,
            "process": dict(fixture_process),
            "executable": sys.executable,
            "executable_identity": _test_executable_identity(),
            "command": list(wait_kwargs["expected_command"]),
            "tmux": dict(fixture_tmux),
            "published_at": wrapper._utc_now(),
        }
        bootstrap["observer_bootstrap_sha256"] = (
            wrapper._canonical_digest(
                bootstrap, "observer_bootstrap_sha256"
            )
        )
        wrapper._write_exclusive(bootstrap_path, bootstrap)
        return (
            dict(fixture_tmux),
            dict(fixture_server),
            _test_tmux_owner_seal(
                fixture_tmux,
                fixture_server,
                owner_nonce=owner_nonce,
                server_start_ticks=fixture_process["start_ticks"],
                pane_process=fixture_process,
            ),
            dict(fixture_process),
            bootstrap,
        )

    monkeypatch.setattr(
        wrapper,
        "_wait_tmux_process_identity",
        fake_wait_identity,
    )
    def fake_gate_launch(
        *,
        ready_path,
        release_path,
        bootstrap_path,
        policy_sha256,
        wrapper_binding,
        owner_nonce,
        observer_command,
        **_kwargs,
    ):
        ready = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_observer_gate_ready_v1"
            ),
            "policy_sha256": policy_sha256,
            "verified_implementations": (
                wrapper._reverify_verified_preflight_apis()
            ),
            "wrapper_claim": dict(wrapper_binding),
            "observer_session": wrapper.OBSERVER_SESSION,
            "owner_nonce": owner_nonce,
            "process": dict(fixture_process),
            "gate_executable": sys.executable,
            "gate_command": [sys.executable, "gate"],
            "tmux": dict(fixture_tmux),
            "tmux_server": dict(fixture_server),
            "release_path": str(release_path.resolve()),
            "bootstrap_path": str(bootstrap_path.resolve()),
            "observer_command": list(observer_command),
            "published_at": wrapper._utc_now(),
        }
        ready["observer_gate_ready_sha256"] = (
            wrapper._canonical_digest(
                ready, "observer_gate_ready_sha256"
            )
        )
        wrapper._write_exclusive(ready_path, ready)
        return (
            {
                "status": "exact_ready",
                "tmux": dict(fixture_tmux),
                "tmux_server": dict(fixture_server),
                "tmux_owner_seal": _test_tmux_owner_seal(
                    fixture_tmux,
                    fixture_server,
                    owner_nonce=owner_nonce,
                    server_start_ticks=fixture_process[
                        "start_ticks"
                    ],
                    pane_process=fixture_process,
                ),
                "process": dict(fixture_process),
                "gate_ready": ready,
                "failure": None,
                "session_residual": True,
            },
            {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "failure": None,
                "command": ["tmux", "new-session"],
            },
        )

    monkeypatch.setattr(
        wrapper,
        "_launch_and_probe_observer_gate",
        fake_gate_launch,
    )
    monkeypatch.setattr(
        wrapper, "_set_observer_remain_on_exit", lambda _seal: None
    )
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout="", stderr=""
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_wait_observer_terminal",
        lambda *_args, **_kwargs: (
            {"status": "completed", "observer_stop": None},
            dict(fake_terminal_binding),
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_read_observer_terminal",
        lambda *_args, **_kwargs: (
            {"status": "completed", "observer_stop": None},
            dict(fake_terminal_binding),
        ),
    )
    monkeypatch.setattr(
        wrapper, "_wait_bound_observer_exit", lambda *_args: True
    )
    monkeypatch.setattr(
        wrapper,
        "_terminate_bound_observer",
        lambda *_args, **_kwargs: {
            "session": wrapper.OBSERVER_SESSION,
            "sealed_tmux": {
                "session": wrapper.OBSERVER_SESSION,
                "pane": "%1",
                "pane_pid": os.getpid(),
                "pane_current_command": "python",
            },
            "sealed_tmux_server": dict(fixture_server),
            "sealed_tmux_owner": _test_tmux_owner_seal(
                fixture_tmux,
                fixture_server,
                owner_nonce="a" * 64,
                server_start_ticks=fixture_process["start_ticks"],
                pane_process=fixture_process,
            ),
            "sealed_process": dict(fixture_process),
            "status": "closed_terminal_observer",
            "session_residual": False,
            "process_residual": False,
            "started_at": wrapper._utc_now(),
            "completed_at": wrapper._utc_now(),
        },
    )


def _run_provisional_observer_launch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation: str,
    cleanup_mode: str = "executed",
    gate_mode: str = "exact_ready",
    remain_failure: bool = False,
    gate_failure_message: str | None = None,
) -> tuple[Any, dict[str, Any], Path, dict[str, Any]]:
    wrapper = _wrapper_module()
    policy_sha256 = "7" * 64
    policy_root = tmp_path / "campaign" / "by_policy" / policy_sha256
    config = (
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    _prepare_wrapper_contract_inputs(wrapper, policy_root)
    observer_command = [
        sys.executable,
        "controller.py",
        "--config",
        str(config),
    ]
    controller_command = [sys.executable, "-c", "raise SystemExit(0)"]
    control = policy_root / "preflight_control"
    controller_process = wrapper._require_process_identity(
        os.getpid(), "fixture controller"
    )
    gate_supervisor_pid = 300
    gate_pid = 301
    controller_process["ppid"] = gate_pid
    controller_process["pgid"] = controller_process["pid"]
    controller_process["sid"] = controller_process["pid"]
    observer_process = _test_process_identity(
        401, ppid=os.getpid(), start_ticks=88
    )
    server = _test_tmux_server_identity(
        tmp_path / "tmux.sock",
        server_pid=os.getpid(),
        server_process=controller_process,
    )
    controller_tmux = {
        "session": wrapper.CONTROLLER_SESSION,
        "pane": "%0",
        "pane_pid": gate_supervisor_pid,
        "pane_current_command": "python",
    }
    observer_tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%1",
        "pane_pid": observer_process["pid"],
        "pane_current_command": "python",
    }
    owner_nonce = "a" * 64
    owner_seal = _test_tmux_owner_seal(
        observer_tmux,
        server,
        owner_nonce=owner_nonce,
        server_start_ticks=controller_process["start_ticks"],
    )
    state: dict[str, Any] = {
        "owner": "sealed",
        "kill_calls": 0,
        "popen_calls": 0,
        "tmux_commands": [],
    }
    monkeypatch.setattr(
        wrapper.secrets, "token_hex", lambda _size: owner_nonce
    )

    monkeypatch.setattr(
        wrapper, "_tmux_session", lambda: wrapper.CONTROLLER_SESSION
    )

    def tmux_identity(session: str) -> dict[str, Any]:
        if session == wrapper.CONTROLLER_SESSION:
            return dict(controller_tmux)
        if state["owner"] == "absent":
            raise wrapper.TmuxTargetAbsent("observer absent")
        if state["owner"] == "foreign":
            return {
                **observer_tmux,
                "pane_pid": 999,
                "pane_current_command": "bash",
            }
        return dict(observer_tmux)

    def pane_identity(pane: str) -> dict[str, Any]:
        if pane == controller_tmux["pane"]:
            return dict(controller_tmux)
        if state["owner"] == "absent":
            raise wrapper.TmuxTargetAbsent("observer pane absent")
        if state["owner"] == "foreign":
            return {
                **observer_tmux,
                "pane_pid": 999,
                "pane_current_command": "bash",
            }
        return dict(observer_tmux)

    monkeypatch.setattr(wrapper, "_tmux_identity", tmux_identity)
    monkeypatch.setattr(wrapper, "_tmux_pane_identity", pane_identity)
    monkeypatch.setattr(
        wrapper,
        "_tmux_server_identity",
        lambda _target=None: dict(server),
    )
    monkeypatch.setattr(
        wrapper,
        "_build_tmux_owner_seal",
        lambda tmux, observed_server, expected_nonce: (
            dict(owner_seal)
            if (
                tmux == observer_tmux
                and observed_server == server
                and expected_nonce == owner_nonce
            )
            else (_ for _ in ()).throw(
                AssertionError("provisional owner inputs differ")
            )
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_tmux_owner_nonce",
        lambda _pane, _socket: (
            "b" * 64
            if state["owner"] in {"foreign", "unsealed"}
            else owner_nonce
        ),
    )

    def process_identity(pid: int):
        if pid == os.getpid():
            return dict(controller_process)
        if pid == gate_pid:
            return _test_process_identity(
                gate_pid,
                ppid=gate_supervisor_pid,
                start_ticks=77,
            )
        if pid == gate_supervisor_pid:
            return _test_process_identity(
                gate_supervisor_pid, start_ticks=76
            )
        if pid == observer_process["pid"] and state["owner"] == "sealed":
            return dict(observer_process)
        return None

    def process_stat(pid: int):
        identity = process_identity(pid)
        return None if identity is None else (identity, "S")

    monkeypatch.setattr(wrapper, "_process_identity", process_identity)
    monkeypatch.setattr(wrapper, "_read_process_stat", process_stat)
    wrapper_arguments = wrapper._process_command(os.getpid())
    wrapper_executable = str(
        Path(os.readlink(f"/proc/{os.getpid()}/exe")).resolve()
    )
    launch_binding = {
        "path": str((tmp_path / "fixture-launch.json").resolve()),
        "sha256": "a" * 64,
        "canonical_sha256": "b" * 64,
    }
    fixture_launch = _test_validated_preflight_launch(
        tmp_path=tmp_path,
        attempt_id="f" * 64,
        receipt_binding=launch_binding,
        receipt_identity=_test_file_identity(
            Path(launch_binding["path"])
        ),
        gate_ready_binding=launch_binding,
        tmux_started_binding=launch_binding,
        wrapper_started_binding=launch_binding,
        gate_supervisor_process=_test_process_identity(
            gate_supervisor_pid, start_ticks=76
        ),
        gate_process=_test_process_identity(
            gate_pid,
            ppid=gate_supervisor_pid,
            start_ticks=77,
        ),
        wrapper_process=_test_process_identity(
            os.getpid(),
            ppid=gate_pid,
            start_ticks=controller_process["start_ticks"],
        ),
        wrapper_arguments=wrapper_arguments,
        wrapper_executable=_test_executable_identity(
            wrapper_executable
        ),
        pane_log=_test_file_identity(
            tmp_path / "fixture-pane.log"
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_validate_preflight_launch_receipt",
        lambda **_kwargs: dict(fixture_launch),
    )
    monkeypatch.setattr(
        wrapper,
        "_wait_preflight_launch_release",
        lambda **_kwargs: None,
    )
    original_wrapper_readlink = wrapper.os.readlink
    monkeypatch.setattr(
        wrapper.os,
        "readlink",
        lambda path: (
            sys.executable
            if path == f"/proc/{observer_process['pid']}/exe"
            else original_wrapper_readlink(path)
        ),
    )
    real_process_executable_identity = (
        wrapper._process_executable_identity
    )
    monkeypatch.setattr(
        wrapper,
        "_process_executable_identity",
        lambda pid: (
            _test_executable_identity()
            if pid == observer_process["pid"]
            else real_process_executable_identity(pid)
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_process_command",
        lambda pid: (
            list(observer_command)
            if pid == observer_process["pid"]
            else (_ for _ in ()).throw(
                AssertionError("unexpected process command PID")
            )
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_process_command_bytes",
        lambda pid: (
            wrapper._command_bytes(observer_command)
            if pid == observer_process["pid"]
            else (_ for _ in ()).throw(
                AssertionError("unexpected raw process command PID")
            )
        ),
    )
    monkeypatch.setattr(wrapper, "OBSERVER_IDENTITY_WAIT_SECONDS", 0.0)

    def fake_tmux_run(command, **_kwargs):
        state["tmux_commands"].append(list(command))
        if mutation != "never_publish":
            wrapper_value = json.loads(
                (control / "wrapper_claim.json").read_text(
                    encoding="utf-8"
                )
            )
            wrapper_binding = {
                "path": str((control / "wrapper_claim.json").resolve()),
                "sha256": hashlib.sha256(
                    (control / "wrapper_claim.json").read_bytes()
                ).hexdigest(),
                "canonical_sha256": wrapper_value[
                    "wrapper_claim_sha256"
                ],
            }
            bootstrap = {
                "schema_version": 1,
                "contract_type": (
                    "safa_canonical_preflight_observer_bootstrap_v1"
                ),
                "policy_sha256": policy_sha256,
                "verified_implementations": (
                    wrapper._reverify_verified_preflight_apis()
                ),
                "wrapper_claim": wrapper_binding,
                "observer_session": wrapper.OBSERVER_SESSION,
                "owner_nonce": owner_nonce,
                "process": dict(observer_process),
                "executable": sys.executable,
                "executable_identity": _test_executable_identity(),
                "command": list(observer_command),
                "tmux": dict(observer_tmux),
                "published_at": wrapper._utc_now(),
            }
            if mutation == "wrapper":
                bootstrap["wrapper_claim"] = {
                    **wrapper_binding,
                    "canonical_sha256": "0" * 64,
                }
            elif mutation == "command":
                bootstrap["command"] = [sys.executable, "-c", "pass # changed"]
            elif mutation == "process":
                bootstrap["process"] = {
                    **observer_process,
                    "start_ticks": observer_process["start_ticks"] + 1,
                }
            elif mutation == "executable_identity":
                bootstrap["executable_identity"] = {
                    **bootstrap["executable_identity"],
                    "inode": (
                        bootstrap["executable_identity"]["inode"] + 1
                    ),
                }
            bootstrap["observer_bootstrap_sha256"] = (
                wrapper._canonical_digest(
                    bootstrap, "observer_bootstrap_sha256"
                )
            )
            if mutation == "canonical":
                bootstrap["observer_bootstrap_sha256"] = "0" * 64
            wrapper._write_exclusive(
                control / "observer_bootstrap.json", bootstrap
            )
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(wrapper.subprocess, "run", fake_tmux_run)
    def failed_bootstrap_gate_launch(
        *,
        ready_path,
        release_path,
        bootstrap_path,
        policy_sha256,
        wrapper_binding,
        owner_nonce,
        observer_command,
        **_kwargs,
    ):
        if gate_mode.startswith("sealed_then_"):
            final_owner = gate_mode.removeprefix("sealed_then_")
            state["owner"] = final_owner
            final_tmux = (
                None
                if final_owner == "absent"
                else {
                    **observer_tmux,
                    **(
                        {
                            "pane_pid": 999,
                            "pane_current_command": "bash",
                        }
                        if final_owner == "foreign"
                        else {}
                    ),
                }
            )
            return (
                {
                    "status": (
                        "absent"
                        if final_owner == "absent"
                        else "owner_unsealed_unknown"
                        if final_owner == "unsealed"
                        else "foreign_or_incomplete_owner"
                    ),
                    "tmux": final_tmux,
                    "tmux_server": (
                        None if final_owner == "absent" else dict(server)
                    ),
                    "tmux_owner_seal": None,
                    "process": None,
                    "process_probe": {"status": "not_observed"},
                    "gate_ready": None,
                    "failure": {
                        "type": "FixtureWeakLaterProbe",
                        "message": gate_mode,
                    },
                    "session_residual": final_owner != "absent",
                    "best_tmux": dict(observer_tmux),
                    "best_tmux_server": dict(server),
                    "best_tmux_owner_seal": dict(owner_seal),
                    "best_process": dict(observer_process),
                    "best_process_probe": {
                        "status": "live",
                        "pid": observer_process["pid"],
                        "state": "S",
                        "identity": dict(observer_process),
                    },
                },
                {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "failure": None,
                    "command": ["tmux", "new-session"],
                },
            )
        if gate_mode != "exact_ready":
            if gate_mode == "foreign_or_incomplete_owner":
                state["owner"] = "foreign"
            exact_owner = gate_mode.startswith("exact_owner_")
            process_probe = (
                {"status": "not_observed"}
                if not exact_owner
                else
                {
                    "status": "error",
                    "pid": observer_process["pid"],
                    "failure": {
                        "type": "OSError",
                        "message": (
                            gate_failure_message
                            or "fixture process stat failed"
                        ),
                    },
                }
                if gate_mode == "exact_owner_process_probe_failed"
                else {
                    "status": "live",
                    "pid": observer_process["pid"],
                    "state": "S",
                    "identity": dict(observer_process),
                }
            )
            return (
                {
                    "status": gate_mode,
                    "tmux": (
                        {
                            **observer_tmux,
                            "pane_pid": 999,
                            "pane_current_command": "bash",
                        }
                        if gate_mode == "foreign_or_incomplete_owner"
                        else dict(observer_tmux)
                    ),
                    "tmux_server": dict(server),
                    "tmux_owner_seal": (
                        dict(owner_seal) if exact_owner else None
                    ),
                    "process": (
                        None
                        if gate_mode
                        == "exact_owner_process_probe_failed"
                        else dict(observer_process)
                        if exact_owner
                        else None
                    ),
                    "process_probe": process_probe,
                    "gate_ready": None,
                    "failure": {
                        "type": "FixtureProbeFailure",
                        "message": (
                            gate_failure_message or gate_mode
                        ),
                    },
                    "session_residual": True,
                },
                {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "failure": None,
                    "command": [
                        "tmux",
                        "new-session",
                        "exec gate",
                    ],
                },
            )
        ready = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_observer_gate_ready_v1"
            ),
            "policy_sha256": policy_sha256,
            "wrapper_claim": dict(wrapper_binding),
            "observer_session": wrapper.OBSERVER_SESSION,
            "owner_nonce": owner_nonce,
            "process": dict(observer_process),
            "gate_executable": sys.executable,
            "gate_command": [sys.executable, "gate"],
            "tmux": dict(observer_tmux),
            "tmux_server": dict(server),
            "release_path": str(release_path.resolve()),
            "bootstrap_path": str(bootstrap_path.resolve()),
            "observer_command": list(observer_command),
            "published_at": wrapper._utc_now(),
        }
        ready["observer_gate_ready_sha256"] = (
            wrapper._canonical_digest(
                ready, "observer_gate_ready_sha256"
            )
        )
        wrapper._write_exclusive(ready_path, ready)
        fake_tmux_run(["fixture-bootstrap"])
        return (
            {
                "status": "exact_ready",
                "tmux": dict(observer_tmux),
                "tmux_server": dict(server),
                "tmux_owner_seal": dict(owner_seal),
                "process": dict(observer_process),
                "gate_ready": ready,
                "failure": None,
                "session_residual": True,
            },
            {
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "failure": None,
                "command": ["tmux", "new-session"],
            },
        )

    monkeypatch.setattr(
        wrapper,
        "_launch_and_probe_observer_gate",
        failed_bootstrap_gate_launch,
    )
    monkeypatch.setattr(
        wrapper,
        "_set_observer_remain_on_exit",
        (
            lambda _seal: (_ for _ in ()).throw(
                RuntimeError("fixture remain-on-exit failed")
            )
            if remain_failure
            else lambda _seal: None
        ),
    )

    def forbidden_popen(*_args, **_kwargs):
        state["popen_calls"] += 1
        raise AssertionError("controller process must remain not_started")

    monkeypatch.setattr(wrapper.subprocess, "Popen", forbidden_popen)

    def conditional(seal):
        assert dict(seal) == owner_seal
        state["kill_calls"] += 1
        if cleanup_mode == "executed":
            state["owner"] = "absent"
            return (
                "executed",
                types.SimpleNamespace(
                    returncode=0, stdout="", stderr=""
                ),
            )
        if cleanup_mode in {"foreign", "reject"}:
            if cleanup_mode == "foreign":
                state["owner"] = "foreign"
            return (
                "condition_rejected",
                types.SimpleNamespace(
                    returncode=0,
                    stdout=wrapper.TMUX_CONDITIONAL_KILL_REJECTED,
                    stderr="",
                ),
            )
        return (
            "command_failed",
            types.SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="fixture conditional kill failed",
            ),
        )

    monkeypatch.setattr(
        wrapper, "_conditional_kill_tmux_owner", conditional
    )
    value = wrapper.run_wrapped_controller(
        repo_root=tmp_path,
        policy_root=policy_root,
        policy_sha256=policy_sha256,
        config=config,
        observer_command=observer_command,
        command=controller_command,
    )
    return wrapper, value, policy_root, state


@pytest.mark.parametrize(
    "mutation",
    (
        "never_publish",
        "canonical",
        "wrapper",
        "command",
        "process",
        "executable_identity",
    ),
)
def test_wrapper_provisional_owner_closes_each_bootstrap_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    wrapper, value, policy_root, state = (
        _run_provisional_observer_launch_failure(
            tmp_path, monkeypatch, mutation=mutation
        )
    )
    control = policy_root / "preflight_control"
    launch = load_json(control / "observer_launch.json", "failed launch")
    cleanup = load_json(control / "observer_cleanup.json", "launch cleanup")
    process_start = load_json(
        control / "controller_process_start.json", "not-started controller"
    )
    process_exit = load_json(
        control / "controller_process_exit.json", "not-started exit"
    )
    assert value["exit_code"] != 0
    assert value["controller_exit_code"] is None
    assert launch["contract_type"].endswith("observer_launch_failed_v1")
    assert launch["status"] == "failed"
    assert launch["provisional_tmux_owner_seal"] is not None
    assert launch["tmux"] is None
    assert cleanup["status"] == "closed_provisional_observer"
    assert cleanup["session_residual"] is False
    assert cleanup["process_residual"] is False
    assert process_start["status"] == "not_started"
    assert process_start["process"] is None
    assert process_exit["status"] == "not_started"
    assert process_exit["controller_pid"] is None
    assert process_exit["exit_code"] is None
    assert state["kill_calls"] == 1
    assert state["popen_calls"] == 0
    attempts = policy_root / "preflight_control/attempts"
    results = policy_root / "checkpoint_preflight/results"
    gpu_control = policy_root / "gpu_control"
    execution_counts = {
        "controller_process_starts": state["popen_calls"],
        "preflight_request_executions": (
            len(list(attempts.glob("*.claim.json")))
            if attempts.exists()
            else 0
        ),
        "preflight_results": (
            len(list(results.glob("*.json"))) if results.exists() else 0
        ),
        "generator_outputs": len(
            list(
                (
                    policy_root / "checkpoint_preflight"
                ).rglob("*.png")
            )
        ),
        "gpu_control_artifacts": (
            len(list(gpu_control.rglob("*")))
            if gpu_control.exists()
            else 0
        ),
    }
    assert execution_counts == {
        "controller_process_starts": 0,
        "preflight_request_executions": 0,
        "preflight_results": 0,
        "generator_outputs": 0,
        "gpu_control_artifacts": 0,
    }
    assert not (control / "controller_process.log").exists()
    assert value["wrapper_exit_sha256"] == wrapper._canonical_digest(
        value, "wrapper_exit_sha256"
    )


def test_wrapper_provisional_cleanup_refuses_foreign_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, value, policy_root, state = (
        _run_provisional_observer_launch_failure(
            tmp_path,
            monkeypatch,
            mutation="canonical",
            cleanup_mode="foreign",
        )
    )
    cleanup = load_json(
        policy_root / "preflight_control/observer_cleanup.json",
        "foreign launch cleanup",
    )
    assert value["exit_code"] != 0
    assert cleanup["status"] == "identity_replaced_not_terminated"
    assert cleanup["session_residual"] is False
    assert cleanup["foreign_session_residual"] is True
    assert cleanup["foreign_tmux"]["pane_pid"] == 999
    assert state["owner"] == "foreign"
    assert state["popen_calls"] == 0


def test_wrapper_provisional_cleanup_failure_is_durable_with_live_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, value, policy_root, state = (
        _run_provisional_observer_launch_failure(
            tmp_path,
            monkeypatch,
            mutation="canonical",
            cleanup_mode="command_failed",
        )
    )
    cleanup = load_json(
        policy_root / "preflight_control/observer_cleanup.json",
        "failed launch cleanup",
    )
    assert value["exit_code"] != 0
    assert cleanup["status"] == "conditional_kill_command_failed"
    assert cleanup["session_residual"] is True
    assert cleanup["process_residual"] is True
    assert cleanup["failure"]["type"] == "TmuxConditionalKillCommandError"
    assert state["owner"] == "sealed"
    assert state["popen_calls"] == 0


def test_wrapper_gate_post_seal_option_failure_closes_exact_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper, value, policy_root, state = (
        _run_provisional_observer_launch_failure(
            tmp_path,
            monkeypatch,
            mutation="never_publish",
            remain_failure=True,
        )
    )
    control = policy_root / "preflight_control"
    launch = load_json(control / "observer_launch.json", "partial launch")
    cleanup = load_json(control / "observer_cleanup.json", "partial cleanup")
    process_start = load_json(
        control / "controller_process_start.json", "partial not-started"
    )
    process_exit = load_json(
        control / "controller_process_exit.json", "partial exit"
    )
    assert value["exit_code"] != 0
    assert "remain-on-exit failed" in launch["failure"]["message"]
    assert launch["provisional_tmux_owner_seal"] is not None
    assert cleanup["status"] == "closed_provisional_observer"
    assert cleanup["session_residual"] is False
    assert process_start["status"] == "not_started"
    assert process_exit["status"] == "not_started"
    assert state["kill_calls"] == 1
    assert state["popen_calls"] == 0
    assert not (policy_root / "checkpoint_preflight/results").exists()
    assert not (policy_root / "preflight_control/attempts").exists()
    assert not (policy_root / "gpu_control").exists()
    assert not list(policy_root.rglob("*.png"))
    assert value["wrapper_exit_sha256"] == wrapper._canonical_digest(
        value, "wrapper_exit_sha256"
    )


@pytest.mark.parametrize(
    "gate_mode",
    (
        "foreign_or_incomplete_owner",
        "owner_unsealed_unknown",
    ),
)
def test_wrapper_gate_unowned_probe_never_kills_and_closes_durably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_mode: str,
) -> None:
    _, value, policy_root, state = (
        _run_provisional_observer_launch_failure(
            tmp_path,
            monkeypatch,
            mutation="never_publish",
            gate_mode=gate_mode,
        )
    )
    control = policy_root / "preflight_control"
    launch = load_json(control / "observer_launch.json", "unowned launch")
    cleanup = load_json(control / "observer_cleanup.json", "unowned cleanup")
    process_start = load_json(
        control / "controller_process_start.json", "unowned not-started"
    )
    process_exit = load_json(
        control / "controller_process_exit.json", "unowned exit"
    )
    assert value["exit_code"] != 0
    assert launch["observer_gate_client"]["returncode"] == 0
    assert launch["observer_gate_probe"]["status"] == gate_mode
    assert launch["provisional_tmux_owner_seal"] is None
    assert cleanup["status"] == "observer_owner_not_sealed"
    assert cleanup["session_residual"] is True
    assert state["kill_calls"] == 0
    assert state["popen_calls"] == 0
    assert process_start["status"] == "not_started"
    assert process_start["process"] is None
    assert process_exit["status"] == "not_started"
    assert process_exit["controller_pid"] is None
    assert not (policy_root / "checkpoint_preflight/results").exists()
    assert not (policy_root / "preflight_control/attempts").exists()
    assert not (policy_root / "gpu_control").exists()
    assert not list(policy_root.rglob("*.png"))
    if gate_mode == "foreign_or_incomplete_owner":
        assert state["owner"] == "foreign"


@pytest.mark.parametrize("later_owner", ("absent", "unsealed", "foreign"))
def test_wrapper_later_weak_probe_uses_best_seal_without_killing_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    later_owner: str,
) -> None:
    wrapper, value, policy_root, state = (
        _run_provisional_observer_launch_failure(
            tmp_path,
            monkeypatch,
            mutation="never_publish",
            gate_mode=f"sealed_then_{later_owner}",
            cleanup_mode="reject",
        )
    )
    control = policy_root / "preflight_control"
    launch = load_json(control / "observer_launch.json", "weak later launch")
    cleanup = load_json(control / "observer_cleanup.json", "weak later cleanup")
    process_start = load_json(
        control / "controller_process_start.json", "weak later start"
    )
    process_exit = load_json(
        control / "controller_process_exit.json", "weak later exit"
    )
    assert value["exit_code"] != 0
    assert launch["observer_gate_probe"]["status"] == (
        "absent"
        if later_owner == "absent"
        else "owner_unsealed_unknown"
        if later_owner == "unsealed"
        else "foreign_or_incomplete_owner"
    )
    assert launch["provisional_tmux_owner_seal"] is not None
    assert (
        launch["provisional_tmux_owner_seal"]["owner_nonce"]
        == "a" * 64
    )
    assert cleanup["tmux_kill_status"] == "condition_rejected"
    assert cleanup["status"] == "identity_replaced_not_terminated"
    assert state["kill_calls"] == 1
    assert state["owner"] == later_owner
    assert state["popen_calls"] == 0
    assert process_start["status"] == "not_started"
    assert process_exit["status"] == "not_started"
    assert not (policy_root / "preflight_control/attempts").exists()
    assert not (policy_root / "checkpoint_preflight/results").exists()
    assert not (policy_root / "gpu_control").exists()
    assert not list(policy_root.rglob("*.png"))
    assert value["wrapper_exit_sha256"] == wrapper._canonical_digest(
        value, "wrapper_exit_sha256"
    )


@pytest.mark.parametrize(
    ("gate_mode", "failure_message", "expected_cleanup_status"),
    (
        (
            "exact_owner_ready_invalid",
            "ready canonical digest differs",
            "closed_provisional_observer",
        ),
        (
            "exact_owner_ready_invalid",
            "ready process identity differs",
            "closed_provisional_observer",
        ),
        (
            "exact_owner_process_probe_failed",
            "process stat permission denied",
            "cleanup_indeterminate_process_residual",
        ),
    ),
)
def test_wrapper_gate_post_seal_probe_failure_keeps_exact_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_mode: str,
    failure_message: str,
    expected_cleanup_status: str,
) -> None:
    _, value, policy_root, state = (
        _run_provisional_observer_launch_failure(
            tmp_path,
            monkeypatch,
            mutation="never_publish",
            gate_mode=gate_mode,
            gate_failure_message=failure_message,
        )
    )
    control = policy_root / "preflight_control"
    launch = load_json(control / "observer_launch.json", "sealed failure launch")
    cleanup = load_json(
        control / "observer_cleanup.json", "sealed failure cleanup"
    )
    process_start = load_json(
        control / "controller_process_start.json",
        "sealed failure not-started",
    )
    process_exit = load_json(
        control / "controller_process_exit.json",
        "sealed failure exit",
    )
    assert value["exit_code"] != 0
    assert launch["observer_gate_probe"]["status"] == gate_mode
    assert failure_message in launch["failure"]["message"]
    assert launch["provisional_tmux_owner_seal"] is not None
    assert cleanup["status"] == expected_cleanup_status
    assert cleanup["session_residual"] is False
    if gate_mode == "exact_owner_process_probe_failed":
        assert cleanup["process_residual"] is None
        assert cleanup["process_probe_failure"]["message"] == failure_message
    else:
        assert cleanup["process_residual"] is False
    assert state["kill_calls"] == 1
    assert state["owner"] == "absent"
    assert state["popen_calls"] == 0
    assert process_start["status"] == "not_started"
    assert process_start["process"] is None
    assert process_exit["status"] == "not_started"
    assert process_exit["controller_pid"] is None
    assert process_exit["exit_code"] is None
    assert not (policy_root / "preflight_control/attempts").exists()
    assert not (policy_root / "checkpoint_preflight/results").exists()
    assert not (policy_root / "gpu_control").exists()
    assert not list(policy_root.rglob("*.png"))


@pytest.mark.parametrize(
    ("failure_kind", "expected_status"),
    (
        ("ready_canonical", "exact_owner_ready_invalid"),
        ("ready_identity", "exact_owner_ready_invalid"),
        ("ready_executable", "exact_owner_ready_invalid"),
        ("ready_command", "exact_owner_ready_invalid"),
        ("process_stat", "exact_owner_process_probe_failed"),
    ),
)
def test_wrapper_probe_owner_seal_is_monotonic_after_exact_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_status: str,
) -> None:
    wrapper = _wrapper_module()
    monkeypatch.setattr(wrapper, "OBSERVER_IDENTITY_WAIT_SECONDS", 0.0)
    tmux_identity = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%71",
        "pane_pid": 701,
        "pane_current_command": "python",
    }
    tmux_server = {
        "server_pid": 601,
        "socket_path": str((tmp_path / "tmux.sock").resolve()),
    }
    process = {"pid": 701, "pgid": 701, "start_ticks": 88}
    owner_nonce = "a" * 64
    owner_seal = _test_tmux_owner_seal(
        tmux_identity,
        tmux_server,
        owner_nonce=owner_nonce,
    )
    monkeypatch.setattr(
        wrapper, "_tmux_identity", lambda _session: dict(tmux_identity)
    )
    monkeypatch.setattr(
        wrapper,
        "_tmux_server_identity",
        lambda _target=None: dict(tmux_server),
    )
    monkeypatch.setattr(
        wrapper, "_tmux_owner_nonce_raw", lambda *_args: owner_nonce
    )
    monkeypatch.setattr(
        wrapper,
        "_build_tmux_owner_seal",
        lambda *_args: dict(owner_seal),
    )
    if failure_kind == "process_stat":
        monkeypatch.setattr(
            wrapper,
            "_read_process_stat",
            lambda _pid: (_ for _ in ()).throw(
                PermissionError("process stat permission denied")
            ),
        )
    else:
        monkeypatch.setattr(
            wrapper,
            "_read_process_stat",
            lambda _pid: (dict(process), "S"),
        )
    ready_path = tmp_path / "observer_gate_ready.json"
    release_path = tmp_path / "observer_gate_release.json"
    bootstrap_path = tmp_path / "observer_bootstrap.json"
    wrapper_binding = {
        "path": str((tmp_path / "wrapper.json").resolve()),
        "sha256": "b" * 64,
        "canonical_sha256": "c" * 64,
    }
    observer_command = [sys.executable, "-c", "pass"]
    if failure_kind != "process_stat":
        ready_process = (
            {**process, "start_ticks": process["start_ticks"] + 1}
            if failure_kind == "ready_identity"
            else dict(process)
        )
        ready = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_preflight_observer_gate_ready_v1"
            ),
            "policy_sha256": "d" * 64,
            "wrapper_claim": wrapper_binding,
            "observer_session": wrapper.OBSERVER_SESSION,
            "owner_nonce": owner_nonce,
            "process": ready_process,
            "gate_executable": (
                "/wrong/executable"
                if failure_kind == "ready_executable"
                else sys.executable
            ),
            "gate_command": (
                [sys.executable, "wrong-gate"]
                if failure_kind == "ready_command"
                else [sys.executable, "gate"]
            ),
            "tmux": tmux_identity,
            "tmux_server": tmux_server,
            "release_path": str(release_path.resolve()),
            "bootstrap_path": str(bootstrap_path.resolve()),
            "observer_command": observer_command,
            "published_at": wrapper._utc_now(),
        }
        ready["observer_gate_ready_sha256"] = (
            wrapper._canonical_digest(
                ready, "observer_gate_ready_sha256"
            )
        )
        if failure_kind == "ready_canonical":
            ready["observer_gate_ready_sha256"] = "0" * 64
        wrapper._write_exclusive(ready_path, ready)
        monkeypatch.setattr(
            wrapper.os, "readlink", lambda _path: sys.executable
        )
        monkeypatch.setattr(
            wrapper,
            "_process_command",
            lambda _pid: [sys.executable, "gate"],
        )
        monkeypatch.setattr(
            wrapper,
            "_process_command_bytes",
            lambda _pid: wrapper._command_bytes(
                [sys.executable, "gate"]
            ),
        )
    probe = wrapper._probe_observer_gate(
        ready_path=ready_path,
        release_path=release_path,
        bootstrap_path=bootstrap_path,
        policy_sha256="d" * 64,
        wrapper_binding=wrapper_binding,
        owner_nonce=owner_nonce,
        observer_command=observer_command,
    )
    assert probe["status"] == expected_status
    assert probe["tmux_owner_seal"] == owner_seal
    if failure_kind == "process_stat":
        assert probe["process_probe"]["status"] == "error"
        assert probe["process"] is None
    else:
        assert probe["process_probe"]["status"] == "live"
        assert probe["process"] == process


def test_wrapper_probe_allows_diagnostic_command_transition_for_same_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    owner_nonce = "a" * 64
    initial_tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%75",
        "pane_pid": 705,
        "pane_current_command": "bash",
    }
    ready_tmux = {
        **initial_tmux,
        "pane_current_command": "python",
    }
    tmux_server = {
        "server_pid": 605,
        "socket_path": str((tmp_path / "tmux.sock").resolve()),
    }
    process = {"pid": 705, "pgid": 705, "start_ticks": 89}
    owner_seal = _test_tmux_owner_seal(
        ready_tmux, tmux_server, owner_nonce=owner_nonce
    )
    ready_path = tmp_path / "observer_gate_ready.json"
    release_path = tmp_path / "observer_gate_release.json"
    bootstrap_path = tmp_path / "observer_bootstrap.json"
    wrapper_binding = {
        "path": str((tmp_path / "wrapper.json").resolve()),
        "sha256": "b" * 64,
        "canonical_sha256": "c" * 64,
    }
    observer_command = [sys.executable, "-c", "pass"]
    ready = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_observer_gate_ready_v1"
        ),
        "policy_sha256": "d" * 64,
        "verified_implementations": (
            wrapper._reverify_verified_preflight_apis()
        ),
        "wrapper_claim": wrapper_binding,
        "observer_session": wrapper.OBSERVER_SESSION,
        "owner_nonce": owner_nonce,
        "process": process,
        "gate_executable": sys.executable,
        "gate_command": [sys.executable, "gate"],
        "tmux": ready_tmux,
        "tmux_server": tmux_server,
        "release_path": str(release_path.resolve()),
        "bootstrap_path": str(bootstrap_path.resolve()),
        "observer_command": observer_command,
        "published_at": wrapper._utc_now(),
    }
    ready["observer_gate_ready_sha256"] = (
        wrapper._canonical_digest(
            ready, "observer_gate_ready_sha256"
        )
    )
    calls = {"identity": 0}

    def identity(_session: str) -> dict[str, Any]:
        calls["identity"] += 1
        if calls["identity"] == 2:
            wrapper._write_exclusive(ready_path, ready)
        return dict(
            initial_tmux
            if calls["identity"] == 1
            else ready_tmux
        )

    monkeypatch.setattr(wrapper, "OBSERVER_IDENTITY_WAIT_SECONDS", 1.0)
    monkeypatch.setattr(wrapper, "_tmux_identity", identity)
    monkeypatch.setattr(
        wrapper,
        "_tmux_server_identity",
        lambda _target=None: dict(tmux_server),
    )
    monkeypatch.setattr(
        wrapper, "_tmux_owner_nonce_raw", lambda *_args: owner_nonce
    )
    monkeypatch.setattr(
        wrapper,
        "_build_tmux_owner_seal",
        lambda *_args: dict(owner_seal),
    )
    monkeypatch.setattr(
        wrapper,
        "_read_process_stat",
        lambda _pid: (dict(process), "S"),
    )
    monkeypatch.setattr(
        wrapper.os, "readlink", lambda _path: sys.executable
    )
    monkeypatch.setattr(
        wrapper,
        "_process_command",
        lambda _pid: [sys.executable, "gate"],
    )
    monkeypatch.setattr(
        wrapper,
        "_process_command_bytes",
        lambda _pid: wrapper._command_bytes(
            [sys.executable, "gate"]
        ),
    )
    probe = wrapper._probe_observer_gate(
        ready_path=ready_path,
        release_path=release_path,
        bootstrap_path=bootstrap_path,
        policy_sha256="d" * 64,
        wrapper_binding=wrapper_binding,
        owner_nonce=owner_nonce,
        observer_command=observer_command,
    )
    assert probe["status"] == "exact_ready"
    assert probe["tmux"] == ready_tmux
    assert probe["best_process"] == process
    assert calls["identity"] >= 2


@pytest.mark.parametrize(
    "mutation",
    (
        "pane",
        "pane_pid",
        "process_ppid",
        "process_sid",
        "process_start_ticks",
        "server",
        "owner_nonce",
    ),
)
def test_wrapper_probe_rejects_stable_owner_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    wrapper = _wrapper_module()
    owner_nonce = "a" * 64
    tmux_identity = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%76",
        "pane_pid": 706,
        "pane_current_command": "bash",
    }
    tmux_server = {
        "server_pid": 606,
        "socket_path": str((tmp_path / "tmux.sock").resolve()),
    }
    process = {"pid": 706, "pgid": 706, "start_ticks": 90}
    owner_seal = _test_tmux_owner_seal(
        tmux_identity, tmux_server, owner_nonce=owner_nonce
    )
    calls = {"identity": 0}

    def identity(_session: str) -> dict[str, Any]:
        calls["identity"] += 1
        value = dict(tmux_identity)
        if calls["identity"] >= 2:
            if mutation == "pane":
                value["pane"] = "%77"
            elif mutation == "pane_pid":
                value["pane_pid"] += 1
        return value

    def server(_target=None) -> dict[str, Any]:
        value = dict(tmux_server)
        if calls["identity"] >= 2 and mutation == "server":
            value["server_pid"] += 1
        return value

    def seal(*_args) -> dict[str, Any]:
        value = dict(owner_seal)
        if calls["identity"] >= 2:
            if mutation == "pane":
                value["pane"] = "%77"
            elif mutation == "pane_pid":
                value["pane_pid"] += 1
            elif mutation == "server":
                value["server_pid"] += 1
            elif mutation == "owner_nonce":
                value["owner_nonce"] = "b" * 64
            elif mutation in {"process_ppid", "process_sid"}:
                value["pane_process"] = dict(
                    value["pane_process"]
                )
                field = (
                    "ppid"
                    if mutation == "process_ppid"
                    else "sid"
                )
                value["pane_process"][field] += 1
        return value

    def process_stat(_pid: int):
        value = dict(process)
        if (
            calls["identity"] >= 2
            and mutation == "process_start_ticks"
        ):
            value["start_ticks"] += 1
        return value, "S"

    monkeypatch.setattr(wrapper, "OBSERVER_IDENTITY_WAIT_SECONDS", 5.0)
    monkeypatch.setattr(wrapper.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wrapper, "_tmux_identity", identity)
    monkeypatch.setattr(wrapper, "_tmux_server_identity", server)
    monkeypatch.setattr(
        wrapper, "_tmux_owner_nonce_raw", lambda *_args: owner_nonce
    )
    monkeypatch.setattr(wrapper, "_build_tmux_owner_seal", seal)
    monkeypatch.setattr(wrapper, "_read_process_stat", process_stat)
    probe = wrapper._probe_observer_gate(
        ready_path=tmp_path / "observer_gate_ready.json",
        release_path=tmp_path / "observer_gate_release.json",
        bootstrap_path=tmp_path / "observer_bootstrap.json",
        policy_sha256="d" * 64,
        wrapper_binding={
            "path": str((tmp_path / "wrapper.json").resolve()),
            "sha256": "b" * 64,
            "canonical_sha256": "c" * 64,
        },
        owner_nonce=owner_nonce,
        observer_command=[sys.executable, "-c", "pass"],
    )
    assert probe["status"] == "exact_owner_evidence_conflict"
    assert probe["failure"]["type"] in {
        "TmuxOwnerEvidenceConflict",
        "ProcessOwnerEvidenceConflict",
    }
    assert probe["session_residual"] is True


@pytest.mark.parametrize(
    ("later_observation", "expected_status"),
    (
        ("absent", "absent"),
        ("unsealed", "owner_unsealed_unknown"),
        ("foreign", "foreign_or_incomplete_owner"),
    ),
)
def test_wrapper_probe_retains_best_exact_evidence_after_weaker_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    later_observation: str,
    expected_status: str,
) -> None:
    wrapper = _wrapper_module()
    tmux_identity = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%81",
        "pane_pid": 801,
        "pane_current_command": "python",
    }
    foreign_tmux = {
        **tmux_identity,
        "pane": "%82",
        "pane_pid": 802,
        "pane_current_command": "bash",
    }
    tmux_server = {
        "server_pid": 701,
        "socket_path": str((tmp_path / "tmux.sock").resolve()),
    }
    process = {"pid": 801, "pgid": 801, "start_ticks": 99}
    owner_nonce = "a" * 64
    owner_seal = _test_tmux_owner_seal(
        tmux_identity, tmux_server, owner_nonce=owner_nonce
    )
    calls = {"identity": 0, "nonce": 0}

    def identity(_session: str) -> dict[str, Any]:
        calls["identity"] += 1
        if calls["identity"] == 1:
            return dict(tmux_identity)
        if later_observation == "absent":
            raise wrapper.TmuxTargetAbsent("later observer absent")
        if later_observation == "foreign":
            return dict(foreign_tmux)
        return dict(tmux_identity)

    def nonce(*_args) -> str:
        calls["nonce"] += 1
        if calls["nonce"] == 1:
            return owner_nonce
        if later_observation == "unsealed":
            raise RuntimeError("later owner environment is absent")
        return "b" * 64

    clock = iter((0.0, 0.1, 1.0))
    monkeypatch.setattr(wrapper, "OBSERVER_IDENTITY_WAIT_SECONDS", 0.5)
    monkeypatch.setattr(wrapper.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(wrapper.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(wrapper, "_tmux_identity", identity)
    monkeypatch.setattr(
        wrapper,
        "_tmux_server_identity",
        lambda _target=None: dict(tmux_server),
    )
    monkeypatch.setattr(wrapper, "_tmux_owner_nonce_raw", nonce)
    monkeypatch.setattr(
        wrapper,
        "_build_tmux_owner_seal",
        lambda *_args: dict(owner_seal),
    )
    monkeypatch.setattr(
        wrapper,
        "_read_process_stat",
        lambda _pid: (dict(process), "S"),
    )
    probe = wrapper._probe_observer_gate(
        ready_path=tmp_path / "observer_gate_ready.json",
        release_path=tmp_path / "observer_gate_release.json",
        bootstrap_path=tmp_path / "observer_bootstrap.json",
        policy_sha256="d" * 64,
        wrapper_binding={
            "path": str((tmp_path / "wrapper.json").resolve()),
            "sha256": "b" * 64,
            "canonical_sha256": "c" * 64,
        },
        owner_nonce=owner_nonce,
        observer_command=[sys.executable, "-c", "pass"],
    )
    assert probe["status"] == expected_status
    assert probe["best_tmux"] == tmux_identity
    assert probe["best_tmux_server"] == tmux_server
    assert probe["best_tmux_owner_seal"] == owner_seal
    assert probe["best_process"] == process
    assert probe["best_process_probe"]["status"] == "live"
    assert probe["tmux_owner_seal"] is None
    assert probe["process"] is None


def test_wrapper_gate_new_session_is_one_exec_command() -> None:
    wrapper = _wrapper_module()
    config_path = (
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    command = wrapper._observer_gate_command(
        ready_path=Path("/tmp/ready.json"),
        release_path=Path("/tmp/release.json"),
        bootstrap_path=Path("/tmp/bootstrap.json"),
        policy_sha256="a" * 64,
        wrapper_binding={
            "path": "/tmp/wrapper.json",
            "sha256": "b" * 64,
            "canonical_sha256": "c" * 64,
        },
        owner_nonce="d" * 64,
        observer_command=[
            sys.executable,
            "controller.py",
            "--config",
            str(config_path),
        ],
    )
    shell_command = "exec " + wrapper.shlex.join(command)
    assert shell_command.startswith("exec ")
    assert ";" not in shell_command
    assert "set-option" not in shell_command
    assert "remain-on-exit" not in shell_command


def test_wrapper_gate_creation_binds_nonce_atomically_before_replacement_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    owner_nonce = "a" * 64
    tmux_commands: list[list[str]] = []

    def run(command: list[str], **kwargs):
        tmux_commands.append(list(command))
        assert kwargs == {"capture_output": True, "text": True}
        return types.SimpleNamespace(
            returncode=0, stdout="", stderr=""
        )

    replacement_probe = {
        "status": "foreign_or_incomplete_owner",
        "tmux": {
            "session": wrapper.OBSERVER_SESSION,
            "pane": "%92",
            "pane_pid": 902,
            "pane_current_command": "bash",
        },
        "tmux_server": {
            "server_pid": 901,
            "socket_path": "/tmp/tmux-test/default",
        },
        "tmux_owner_seal": None,
        "process": None,
        "process_probe": {"status": "not_observed"},
        "gate_ready": None,
        "failure": {
            "type": "TmuxOwnerMarkerMismatch",
            "message": "replacement changed the session environment",
        },
        "session_residual": True,
    }
    monkeypatch.setattr(wrapper.subprocess, "run", run)
    monkeypatch.setattr(
        wrapper,
        "_probe_observer_gate",
        lambda **_kwargs: dict(replacement_probe),
    )
    probe, client = wrapper._launch_and_probe_observer_gate(
        repo_root=tmp_path,
        ready_path=tmp_path / "ready.json",
        release_path=tmp_path / "release.json",
        bootstrap_path=tmp_path / "bootstrap.json",
        policy_sha256="d" * 64,
        wrapper_binding={
            "path": str((tmp_path / "wrapper.json").resolve()),
            "sha256": "b" * 64,
            "canonical_sha256": "c" * 64,
        },
        owner_nonce=owner_nonce,
        observer_command=[
            sys.executable,
            "controller.py",
            "--config",
            str(
                Path(__file__).parents[1]
                / "configs/closeout/canonical_screening_512_v1.json"
            ),
        ],
    )
    assert probe == replacement_probe
    assert client["returncode"] == 0
    assert len(tmux_commands) == 1
    command = tmux_commands[0]
    assert command[:4] == ["tmux", "new-session", "-d", "-s"]
    assert command[4] == wrapper.OBSERVER_SESSION
    assert command.count("-e") == 2
    assert (
        f"{wrapper.TMUX_OWNER_ENV}={owner_nonce}" in command
    )
    assert (
        f"{wrapper.OBSERVER_SESSION_ENV}={wrapper.OBSERVER_SESSION}"
        in command
    )
    assert command[-1].startswith("exec ")
    assert "set-option" not in command
    assert len(wrapper.OBSERVER_SESSION) == (
        len(wrapper.OBSERVER_SESSION_PREFIX) + 1 + 64
    )
    assert wrapper.OBSERVER_SESSION.startswith(
        f"{wrapper.OBSERVER_SESSION_PREFIX}-"
    )


def _test_process_stat(
    pid: int,
    *,
    state: str = "S",
    pgid: int | None = None,
    start_ticks: int = 777,
    command: str = "python worker",
) -> str:
    resolved_pgid = pid if pgid is None else pgid
    fields = (
        [state, "1", str(resolved_pgid), str(resolved_pgid)]
        + ["0"] * 15
        + [str(start_ticks)]
    )
    return f"{pid} ({command}) {' '.join(fields)}\n"


def test_wrapper_process_identity_initial_stat_disappearance_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    pid = os.getpid()
    monkeypatch.setattr(
        wrapper.Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError(pid)
        ),
    )
    assert wrapper._process_identity_state(pid) is None
    assert wrapper._process_identity(pid) is None


def test_wrapper_process_identity_zombie_skips_executable_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    pid = 401
    raw_stat = _test_process_stat(
        pid, state="Z", pgid=401, start_ticks=88
    )
    monkeypatch.setattr(
        wrapper.Path,
        "read_text",
        lambda *_args, **_kwargs: raw_stat,
    )
    monkeypatch.setattr(
        wrapper.os,
        "readlink",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("zombie executable must not be probed")
        ),
    )
    monkeypatch.setattr(
        wrapper.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(
            AssertionError("zombie process group must not be probed")
        ),
    )
    assert wrapper._process_identity_state(pid) == (
        _test_process_identity(
            pid,
            ppid=1,
            pgid=401,
            sid=401,
            start_ticks=88,
        ),
        "Z",
    )


class ProductionV4Harness:
    def __init__(
        self,
        *,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        policy_sha256: str,
        wrapper: Any,
        command: list[str],
    ) -> None:
        self.monkeypatch = monkeypatch
        self.policy_sha256 = policy_sha256
        self.wrapper = wrapper
        self.launcher = _launcher_module()
        self.repo_root = Path(__file__).parents[1]
        self.config = (
            self.repo_root
            / "configs/closeout/canonical_screening_512_v1.json"
        )
        self.campaign_root = tmp_path / "campaign"
        self.policy_root = (
            self.campaign_root / "by_policy" / policy_sha256
        )
        self.attempt_id = hashlib.sha256(
            (
                f"production-v4-harness:{policy_sha256}:"
                f"{tmp_path.resolve()}"
            ).encode()
        ).hexdigest()
        self.controller_owner_nonce = hashlib.sha256(
            f"production-v4-owner:{self.attempt_id}".encode()
        ).hexdigest()
        self.observer_suffix = hashlib.sha256(
            f"production-v4-observer:{self.attempt_id}".encode()
        ).hexdigest()
        self.consumer_session = (
            self.launcher.PANE_FAULT_CONSUMER_SESSION_PREFIX
            + self.attempt_id
        )
        if "--supervised-child" in command:
            raise AssertionError(
                "ProductionV4Harness owns supervised-child injection"
            )
        self.wrapper_arguments = [*command, "--supervised-child"]
        self.stage_timings: list[dict[str, Any]] = []
        self.receipt_path = (
            self.campaign_root
            / "preflight_launch_attempts"
            / "by_policy"
            / policy_sha256
            / self.attempt_id
            / "launch_receipt.json"
        )

    def _begin_stage(
        self, name: str, budget_seconds: float
    ) -> dict[str, Any]:
        started = time.monotonic()
        if (
            self.stage_timings
            and self.stage_timings[-1]["ended"] > started
        ):
            raise AssertionError(
                "ProductionV4Harness phase clocks overlap"
            )
        stage = {
            "name": name,
            "started": started,
            "deadline": started + budget_seconds,
            "ended": None,
        }
        self.stage_timings.append(stage)
        return stage

    @staticmethod
    def _end_stage(stage: dict[str, Any]) -> None:
        ended = time.monotonic()
        stage["ended"] = ended
        if ended > stage["deadline"]:
            raise AssertionError(
                "ProductionV4Harness phase exceeded its own budget: "
                f"{stage['name']}"
            )

    def wait_json(
        self, path: Path, label: str, *, timeout_seconds: float
    ) -> dict[str, Any]:
        self.wait_path(
            path,
            label,
            timeout_seconds=timeout_seconds,
        )
        return load_json(path, label)

    def wait_path(
        self,
        path: Path,
        label: str,
        *,
        timeout_seconds: float,
        require_nonempty: bool = False,
    ) -> Path:
        stage = self._begin_stage(label, timeout_seconds)
        while not (
            path.is_file()
            and (
                not require_nonempty
                or path.stat().st_size > 0
            )
        ):
            if time.monotonic() >= stage["deadline"]:
                stage["ended"] = time.monotonic()
                raise AssertionError(f"{label} timed out")
            time.sleep(0.05)
        self._end_stage(stage)
        return path

    def __enter__(self) -> "ProductionV4Harness":
        self.launcher._install_verified_preflight_apis(
            self.config
        )

        def verified_git_state(root: Path) -> dict[str, str]:
            values: dict[str, str] = {}
            for name, arguments in (
                ("head_sha", ("rev-parse", "HEAD")),
                (
                    "origin_master_sha",
                    ("rev-parse", "origin/master"),
                ),
                ("branch", ("branch", "--show-current")),
            ):
                result = subprocess.run(
                    ["git", "-C", str(root), *arguments],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                values[name] = result.stdout.strip()
            return values

        self.monkeypatch.setattr(
            self.launcher,
            "_verified_git_state",
            verified_git_state,
        )
        for session in (
            self.wrapper.CONTROLLER_SESSION,
            self.wrapper.OBSERVER_SESSION,
            self.consumer_session,
        ):
            if self.launcher._tmux_pane(session) is not None:
                raise AssertionError(
                    "ProductionV4Harness refuses a pre-existing "
                    f"session: {session}"
                )
        stage = self._begin_stage("production v4 launch", 10.0)
        self.launcher.launch_preflight(
            repo_root=self.repo_root,
            config=self.config,
            campaign_root=self.campaign_root,
            policy_sha256=self.policy_sha256,
            python=sys.executable,
            startup_timeout_seconds=10.0,
            attempt_id=self.attempt_id,
            owner_nonce=self.controller_owner_nonce,
            observer_suffix=self.observer_suffix,
            wrapper_arguments_override=self.wrapper_arguments,
        )
        self._end_stage(stage)
        receipt = load_json(
            self.receipt_path,
            "ProductionV4Harness launch receipt",
        )
        self.launcher.validate_launch_receipt_schema(
            receipt,
            expected_gate_worker_arguments=receipt[
                "gate_worker_arguments"
            ],
            expected_consumer_worker_arguments=receipt[
                "consumer_worker_arguments"
            ],
            label="ProductionV4Harness launch receipt v4",
        )
        return self

    def _owned_consumer_attempt(
        self,
    ) -> tuple[Path, dict[str, Any]] | None:
        if not self.receipt_path.is_file():
            return None
        receipt = load_json(
            self.receipt_path,
            "ProductionV4Harness teardown receipt",
        )
        attempt_path = Path(
            receipt["pane_fault_consumer"]["artifacts"][
                "attempt"
            ]
        )
        if not attempt_path.is_file():
            return None
        return (
            attempt_path,
            load_json(
                attempt_path,
                "ProductionV4Harness consumer attempt",
            ),
        )

    def _teardown(self) -> None:
        consumer = self._owned_consumer_attempt()
        if consumer is not None:
            attempt_path, attempt = consumer
            terminal_path = Path(attempt["artifacts"]["terminal"])
            if terminal_path.is_file():
                self.launcher.join_pane_fault_consumer(
                    attempt_path=attempt_path,
                    config=self.config,
                    timeout_seconds=5.0,
                )
            elif (
                self.launcher._tmux_pane(
                    attempt["consumer_session"]
                )
                is not None
            ):
                self.launcher._cleanup_failed_pane_fault_consumer(
                    attempt["consumer_session"],
                    attempt["consumer_owner_nonce"],
                )
        observer_pane = self.launcher._tmux_pane(
            self.wrapper.OBSERVER_SESSION
        )
        if observer_pane is not None:
            launch_path = (
                self.policy_root
                / "preflight_control/observer_launch.json"
            )
            if not launch_path.is_file():
                raise AssertionError(
                    "owned observer residual has no launch seal"
                )
            launch = load_json(
                launch_path,
                "ProductionV4Harness observer launch",
            )
            cleanup = self.wrapper._terminate_bound_observer(
                launch["tmux"],
                launch["tmux_server"],
                launch["owner_seal"],
                launch["process"],
            )
            if (
                cleanup["session_residual"]
                or cleanup["process_residual"]
            ):
                raise AssertionError(
                    "ProductionV4Harness observer cleanup residual"
                )
        controller_pane = self.launcher._tmux_pane(
            self.wrapper.CONTROLLER_SESSION
        )
        if controller_pane is not None:
            if consumer is None:
                raise AssertionError(
                    "owned controller residual has no consumer seal"
                )
            gate_owner_seal = consumer[1]["gate_owner_seal"]
            current_owner = self.launcher._tmux_owner_seal(
                self.wrapper.CONTROLLER_SESSION,
                gate_owner_seal["owner_nonce"],
            )
            if any(
                current_owner[key] != gate_owner_seal[key]
                for key in (
                    "session",
                    "pane",
                    "pane_pid",
                    "owner_nonce",
                    "tmux_server",
                )
            ):
                raise AssertionError(
                    "ProductionV4Harness preserves foreign controller "
                    "owner"
                )
            self.launcher._kill_exact_session(
                self.wrapper.CONTROLLER_SESSION,
                gate_owner_seal["owner_nonce"],
                current_owner,
            )
        residual = {
            session: self.launcher._tmux_pane(session)
            for session in (
                self.wrapper.CONTROLLER_SESSION,
                self.wrapper.OBSERVER_SESSION,
                self.consumer_session,
            )
        }
        if any(value is not None for value in residual.values()):
            raise AssertionError(
                f"ProductionV4Harness residual differs: {residual}"
            )

    def __exit__(
        self,
        exc_type: Any,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, exc, traceback
        stage = self._begin_stage(
            "ProductionV4Harness teardown", 5.0
        )
        self._teardown()
        self._end_stage(stage)
        return False


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux /proc zombie semantics",
)
def test_wrapper_real_zombie_with_missing_executable_keeps_identity() -> None:
    wrapper = _wrapper_module()
    process = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5.0
        snapshot = None
        while time.monotonic() < deadline:
            snapshot = wrapper._read_process_stat(process.pid)
            if snapshot is not None and snapshot[1] == "Z":
                break
            time.sleep(0.01)
        assert snapshot is not None
        assert snapshot[1] == "Z"
        with pytest.raises(FileNotFoundError):
            os.readlink(f"/proc/{process.pid}/exe")
        assert wrapper._process_identity_state(process.pid) == snapshot
    finally:
        process.wait(timeout=5.0)


def test_wrapper_process_identity_live_missing_executable_stat_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    pid = 402
    reads: list[str | BaseException] = [
        _test_process_stat(pid, start_ticks=89),
        FileNotFoundError(pid),
    ]

    def read_stat(*_args, **_kwargs):
        value = reads.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(wrapper.Path, "read_text", read_stat)
    monkeypatch.setattr(
        wrapper.os,
        "readlink",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError(pid)),
    )
    assert wrapper._process_identity_state(pid) is None
    assert reads == []


def test_wrapper_process_identity_live_missing_executable_same_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    pid = 403
    raw_stat = _test_process_stat(pid, start_ticks=90)
    monkeypatch.setattr(
        wrapper.Path, "read_text", lambda *_args, **_kwargs: raw_stat
    )
    monkeypatch.setattr(
        wrapper.os,
        "readlink",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError(pid)),
    )
    with pytest.raises(
        RuntimeError, match="executable is absent.*remains live"
    ):
        wrapper._process_identity_state(pid)


def test_wrapper_process_identity_live_missing_executable_pid_reuse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    pid = 404
    reads = iter(
        (
            _test_process_stat(pid, start_ticks=91),
            _test_process_stat(pid, start_ticks=92),
        )
    )
    monkeypatch.setattr(
        wrapper.Path,
        "read_text",
        lambda *_args, **_kwargs: next(reads),
    )
    monkeypatch.setattr(
        wrapper.os,
        "readlink",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError(pid)),
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        wrapper._process_identity_state(pid)


@pytest.mark.parametrize("second_state", ("absent", "same_live"))
def test_wrapper_process_identity_getpgid_esrch_revalidates_stat(
    monkeypatch: pytest.MonkeyPatch,
    second_state: str,
) -> None:
    wrapper = _wrapper_module()
    pid = 405
    raw_stat = _test_process_stat(pid, start_ticks=93)
    reads: list[str | BaseException] = [
        raw_stat,
        (
            FileNotFoundError(pid)
            if second_state == "absent"
            else raw_stat
        ),
    ]

    def read_stat(*_args, **_kwargs):
        value = reads.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(wrapper.Path, "read_text", read_stat)
    monkeypatch.setattr(wrapper.os, "readlink", lambda _path: "/python")
    monkeypatch.setattr(
        wrapper.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(ProcessLookupError(pid)),
    )
    if second_state == "absent":
        assert wrapper._process_identity_state(pid) is None
    else:
        with pytest.raises(
            RuntimeError, match="process group is absent.*remains live"
        ):
            wrapper._process_identity_state(pid)
    assert reads == []


def test_wrapper_process_identity_final_stat_pid_reuse_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    pid = 406
    reads = iter(
        (
            _test_process_stat(pid, start_ticks=94),
            _test_process_stat(pid, start_ticks=95),
        )
    )
    monkeypatch.setattr(
        wrapper.Path,
        "read_text",
        lambda *_args, **_kwargs: next(reads),
    )
    monkeypatch.setattr(wrapper.os, "readlink", lambda _path: "/python")
    monkeypatch.setattr(wrapper.os, "getpgid", lambda _pid: pid)
    with pytest.raises(RuntimeError, match="identity changed during snapshot"):
        wrapper._process_identity_state(pid)


def test_wrapper_process_identity_parses_command_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    pid = 407
    raw_stat = _test_process_stat(
        pid,
        pgid=400,
        start_ticks=96,
        command="tmux: server worker",
    )
    monkeypatch.setattr(
        wrapper.Path, "read_text", lambda *_args, **_kwargs: raw_stat
    )
    monkeypatch.setattr(wrapper.os, "readlink", lambda _path: "/tmux")
    monkeypatch.setattr(wrapper.os, "getpgid", lambda _pid: 400)
    assert wrapper._process_identity_state(pid) == (
        _test_process_identity(
            pid,
            ppid=1,
            pgid=400,
            sid=400,
            start_ticks=96,
        ),
        "S",
    )


@pytest.mark.parametrize(
    "raw_stat",
    (
        "malformed",
        "408 (python) S 1",
        _test_process_stat(409, start_ticks=97),
        _test_process_stat(408, state="SS", start_ticks=97),
    ),
)
def test_wrapper_process_identity_snapshot_parse_error_fails(
    monkeypatch: pytest.MonkeyPatch,
    raw_stat: str,
) -> None:
    wrapper = _wrapper_module()
    pid = 408
    monkeypatch.setattr(
        wrapper.Path,
        "read_text",
        lambda *_args, **_kwargs: raw_stat,
    )
    with pytest.raises(RuntimeError, match="stat is malformed"):
        wrapper._process_identity_state(pid)


@pytest.mark.parametrize("stage", ("stat", "executable", "process_group"))
def test_wrapper_process_identity_permission_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    wrapper = _wrapper_module()
    pid = 410
    raw_stat = _test_process_stat(pid, start_ticks=98)

    def read_stat(*_args, **_kwargs):
        if stage == "stat":
            raise PermissionError(pid)
        return raw_stat

    def read_executable(_path):
        if stage == "executable":
            raise PermissionError(pid)
        return "/python"

    def read_process_group(_pid):
        if stage == "process_group":
            raise PermissionError(pid)
        return pid

    monkeypatch.setattr(
        wrapper.Path,
        "read_text",
        read_stat,
    )
    monkeypatch.setattr(wrapper.os, "readlink", read_executable)
    monkeypatch.setattr(wrapper.os, "getpgid", read_process_group)
    with pytest.raises(RuntimeError, match="permission denied"):
        wrapper._process_identity_state(pid)


def test_controller_process_identity_parses_parenthesized_command_spaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _controller_module()
    fields = ["S", "1", "301", "301"] + ["0"] * 15 + ["777"]
    raw_stat = f"301 (tmux: server) {' '.join(fields)}\n"
    monkeypatch.setattr(
        module.Path,
        "read_text",
        lambda *_args, **_kwargs: raw_stat,
    )
    assert module._process_identity(301) == _test_process_identity(
        301,
        ppid=1,
        pgid=301,
        sid=301,
        start_ticks=777,
    )


@pytest.mark.parametrize(
    ("reference", "field"),
    tuple(
        (reference, field)
        for reference in (
            "controller_process_exit",
            "observer_claim",
            "observer_ready",
        )
        for field in ("path", "sha256", "canonical_sha256")
    ),
)
def test_wrapper_observer_terminal_rejects_reference_binding_tamper(
    tmp_path: Path,
    reference: str,
    field: str,
) -> None:
    wrapper = _wrapper_module()
    policy_sha256 = "7" * 64
    observer_process = {"pid": 41, "pgid": 41, "start_ticks": 99}
    observer_launch_binding = {
        "path": str((tmp_path / "observer_launch.json").resolve()),
        "sha256": "2" * 64,
        "canonical_sha256": "3" * 64,
    }

    def artifact(
        name: str, digest_field: str, body: dict[str, Any]
    ) -> tuple[Path, dict[str, str]]:
        value = dict(body)
        value[digest_field] = wrapper._canonical_digest(
            value, digest_field
        )
        artifact_path = tmp_path / f"{name}.json"
        wrapper._write_exclusive(artifact_path, value)
        return artifact_path, {
            "path": str(artifact_path.resolve()),
            "sha256": wrapper._sha256_file(artifact_path),
            "canonical_sha256": value[digest_field],
        }

    process_exit_path, process_exit_binding = artifact(
        "controller_process_exit",
        "controller_process_exit_sha256",
        {"contract_type": "fixture_process_exit"},
    )
    _, claim_binding = artifact(
        "observer_claim",
        "observer_claim_sha256",
        {
            "contract_type": "safa_canonical_preflight_observer_claim_v1",
            "phase": "preflight",
            "policy_sha256": policy_sha256,
            "observer_launch": observer_launch_binding,
            "observer_session": wrapper.OBSERVER_SESSION,
            "observer_pid": observer_process["pid"],
            "observer_process": observer_process,
        },
    )
    _, ready_binding = artifact(
        "observer_ready",
        "observer_ready_sha256",
        {
            "contract_type": "safa_canonical_preflight_observer_ready_v1",
            "phase": "preflight",
            "policy_sha256": policy_sha256,
            "observer_claim": claim_binding,
            "observer_claim_sha256": claim_binding["canonical_sha256"],
            "observer_launch": observer_launch_binding,
            "observer_session": wrapper.OBSERVER_SESSION,
            "observer_pid": observer_process["pid"],
            "observer_process": observer_process,
        },
    )
    bindings = {
        "controller_process_exit": process_exit_binding,
        "observer_claim": claim_binding,
        "observer_ready": ready_binding,
    }
    changed = dict(bindings[reference])
    changed[field] = (
        str((tmp_path / "other.json").resolve())
        if field == "path"
        else ("0" if field == "sha256" else "1") * 64
    )
    bindings[reference] = changed
    terminal = {
        "contract_type": "safa_canonical_preflight_observer_terminal_v1",
        "policy_sha256": policy_sha256,
        "status": "completed",
        "failure": None,
        **bindings,
    }
    terminal["observer_terminal_sha256"] = wrapper._canonical_digest(
        terminal, "observer_terminal_sha256"
    )
    terminal_path = tmp_path / "observer_terminal.json"
    wrapper._write_exclusive(terminal_path, terminal)
    with pytest.raises(RuntimeError, match="observer terminal"):
        wrapper._read_observer_terminal(
            terminal_path,
            process_exit_path,
            policy_sha256=policy_sha256,
            observer_launch_binding=observer_launch_binding,
            observer_process=observer_process,
        )


def _completed_terminal_fixture(
    tmp_path: Path,
) -> tuple[
    Any,
    Path,
    Path,
    str,
    dict[str, str],
    dict[str, int],
]:
    wrapper = _wrapper_module()
    policy_sha256 = "8" * 64
    observer_process = {"pid": 81, "pgid": 81, "start_ticks": 181}
    observer_launch_binding = {
        "path": str((tmp_path / "observer_launch.json").resolve()),
        "sha256": "2" * 64,
        "canonical_sha256": "3" * 64,
    }

    def artifact(
        name: str, digest_field: str, body: dict[str, Any]
    ) -> tuple[Path, dict[str, str], dict[str, Any]]:
        value = dict(body)
        value[digest_field] = wrapper._canonical_digest(
            value, digest_field
        )
        artifact_path = tmp_path / f"{name}.json"
        wrapper._write_exclusive(artifact_path, value)
        return artifact_path, {
            "path": str(artifact_path.resolve()),
            "sha256": wrapper._sha256_file(artifact_path),
            "canonical_sha256": value[digest_field],
        }, value

    process_exit_path, process_exit_binding, _ = artifact(
        "controller_process_exit",
        "controller_process_exit_sha256",
        {"contract_type": "fixture_process_exit"},
    )
    claim_path, claim_binding, claim = artifact(
        "observer_claim",
        "observer_claim_sha256",
        {
            "contract_type": "safa_canonical_preflight_observer_claim_v1",
            "phase": "preflight",
            "policy_sha256": policy_sha256,
            "observer_launch": observer_launch_binding,
            "observer_session": wrapper.OBSERVER_SESSION,
            "observer_pid": observer_process["pid"],
            "observer_process": observer_process,
        },
    )
    ready_path, ready_binding, _ = artifact(
        "observer_ready",
        "observer_ready_sha256",
        {
            "contract_type": "safa_canonical_preflight_observer_ready_v1",
            "phase": "preflight",
            "policy_sha256": policy_sha256,
            "observer_claim": claim_binding,
            "observer_claim_sha256": claim["observer_claim_sha256"],
            "observer_launch": observer_launch_binding,
            "observer_session": wrapper.OBSERVER_SESSION,
            "observer_pid": observer_process["pid"],
            "observer_process": observer_process,
        },
    )
    terminal = {
        "contract_type": "safa_canonical_preflight_observer_terminal_v1",
        "policy_sha256": policy_sha256,
        "status": "completed",
        "failure": None,
        "controller_process_exit": process_exit_binding,
        "observer_claim": claim_binding,
        "observer_ready": ready_binding,
    }
    terminal["observer_terminal_sha256"] = wrapper._canonical_digest(
        terminal, "observer_terminal_sha256"
    )
    terminal_path = tmp_path / "observer_terminal.json"
    wrapper._write_exclusive(terminal_path, terminal)
    assert claim_path.is_file() and ready_path.is_file()
    return (
        wrapper,
        terminal_path,
        process_exit_path,
        policy_sha256,
        observer_launch_binding,
        observer_process,
    )


def test_wrapper_completed_observer_terminal_with_full_evidence_passes(
    tmp_path: Path,
) -> None:
    (
        wrapper,
        terminal_path,
        process_exit_path,
        policy_sha256,
        observer_launch_binding,
        observer_process,
    ) = _completed_terminal_fixture(tmp_path)
    value, binding = wrapper._read_observer_terminal(
        terminal_path,
        process_exit_path,
        policy_sha256=policy_sha256,
        observer_launch_binding=observer_launch_binding,
        observer_process=observer_process,
    )
    assert value["status"] == "completed"
    assert binding["path"] == str(terminal_path.resolve())


@pytest.mark.parametrize(
    ("evidence", "mutation"),
    (
        ("observer_claim", "drop"),
        ("observer_ready", "drop"),
        ("observer_claim", "null"),
        ("observer_ready", "null"),
        ("observer_claim", "file_missing"),
        ("observer_ready", "file_missing"),
    ),
)
def test_wrapper_completed_observer_terminal_requires_claim_and_ready(
    tmp_path: Path,
    evidence: str,
    mutation: str,
) -> None:
    (
        wrapper,
        terminal_path,
        process_exit_path,
        policy_sha256,
        observer_launch_binding,
        observer_process,
    ) = _completed_terminal_fixture(tmp_path)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    evidence_path = (
        None
        if terminal.get(evidence) is None
        else Path(terminal[evidence]["path"])
    )
    terminal_path.unlink()
    if mutation == "drop":
        terminal.pop(evidence)
    elif mutation == "null":
        terminal[evidence] = None
    else:
        assert evidence_path is not None
        evidence_path.unlink()
    terminal["observer_terminal_sha256"] = wrapper._canonical_digest(
        terminal, "observer_terminal_sha256"
    )
    wrapper._write_exclusive(terminal_path, terminal)
    with pytest.raises(RuntimeError):
        wrapper._read_observer_terminal(
            terminal_path,
            process_exit_path,
            policy_sha256=policy_sha256,
            observer_launch_binding=observer_launch_binding,
            observer_process=observer_process,
        )


def test_wrapper_late_completed_terminal_requires_full_evidence(
    tmp_path: Path,
) -> None:
    (
        wrapper,
        terminal_path,
        process_exit_path,
        policy_sha256,
        observer_launch_binding,
        observer_process,
    ) = _completed_terminal_fixture(tmp_path)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_path.unlink()
    terminal["observer_ready"] = None
    terminal["observer_terminal_sha256"] = wrapper._canonical_digest(
        terminal, "observer_terminal_sha256"
    )
    wrapper._write_exclusive(terminal_path, terminal)
    with pytest.raises(RuntimeError):
        wrapper._wait_observer_terminal(
            terminal_path,
            process_exit_path,
            policy_sha256=policy_sha256,
            observer_launch_binding=observer_launch_binding,
            observer_process=observer_process,
        )


def test_wrapper_failed_terminal_without_ready_remains_valid(
    tmp_path: Path,
) -> None:
    (
        wrapper,
        terminal_path,
        process_exit_path,
        policy_sha256,
        observer_launch_binding,
        observer_process,
    ) = _completed_terminal_fixture(tmp_path)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_path.unlink()
    terminal["status"] = "failed"
    terminal["failure"] = {"type": "FixtureFailure", "message": "failed"}
    terminal["observer_claim"] = None
    terminal["observer_ready"] = None
    terminal["observer_terminal_sha256"] = wrapper._canonical_digest(
        terminal, "observer_terminal_sha256"
    )
    wrapper._write_exclusive(terminal_path, terminal)
    value, _ = wrapper._read_observer_terminal(
        terminal_path,
        process_exit_path,
        policy_sha256=policy_sha256,
        observer_launch_binding=observer_launch_binding,
        observer_process=observer_process,
    )
    assert value["status"] == "failed"


@pytest.mark.parametrize(
    "stderr",
    (
        "no server running on /tmp/tmux-1/default",
        "can't find session: missing",
        "can't find window: missing",
        "can't find pane: missing",
    ),
)
def test_wrapper_tmux_identity_classifies_only_explicit_absence(
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
) -> None:
    wrapper = _wrapper_module()
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=1, stdout="", stderr=stderr
        ),
    )
    with pytest.raises(wrapper.TmuxTargetAbsent):
        wrapper._tmux_identity("missing")


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode"),
    (
        (
            "s\t%1\t1\tpython\ns\t%2\t2\tpython\n",
            "",
            0,
        ),
        ("malformed\n", "", 0),
        ("", "permission denied", 1),
    ),
)
def test_wrapper_tmux_identity_non_absence_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    stderr: str,
    returncode: int,
) -> None:
    wrapper = _wrapper_module()
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        ),
    )
    with pytest.raises(RuntimeError) as failure:
        wrapper._tmux_identity("s")
    assert not isinstance(failure.value, wrapper.TmuxTargetAbsent)


def _test_tmux_owner_seal(
    tmux: Mapping[str, Any],
    server: Mapping[str, Any],
    *,
    owner_nonce: str = "a" * 64,
    server_start_ticks: int = 55,
    pane_process: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sealed_server = (
        dict(server)
        if set(server) == {
            "server_pid",
            "server_process",
            "socket_path",
            "socket_device",
            "socket_inode",
        }
        else _test_tmux_server_identity(
            Path(str(server["socket_path"])),
            server_pid=int(server["server_pid"]),
            server_process=_test_process_identity(
                int(server["server_pid"]),
                start_ticks=server_start_ticks,
            ),
        )
    )
    sealed_pane_process = (
        _test_process_identity(
            int(tmux["pane_pid"]), start_ticks=77
        )
        if pane_process is None
        else dict(pane_process)
    )
    return build_preflight_pane_owner_seal(
        server_pid=int(sealed_server["server_pid"]),
        server_start_ticks=server_start_ticks,
        socket_path=str(sealed_server["socket_path"]),
        socket_device=int(sealed_server["socket_device"]),
        socket_inode=int(sealed_server["socket_inode"]),
        session=str(tmux["session"]),
        pane=str(tmux["pane"]),
        pane_pid=int(tmux["pane_pid"]),
        pane_process=sealed_pane_process,
        owner_nonce=owner_nonce,
        tmux_identity=tmux,
        tmux_server=sealed_server,
    )


def _write_preflight_wrapper_v3_fixture(
    module: Any,
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    gate_pid: int,
    wrapper_pid: int,
    controller_tmux: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    control = paths["preflight_control"]
    verified_implementations = (
        _test_verified_preflight_implementations()
    )
    module.validate_tmux_server_identity(
        tmux_server, "fixture tmux server"
    )
    gate_process = _test_process_identity(
        gate_pid, start_ticks=30
    )
    wrapper_launch_process = _test_process_identity(
        wrapper_pid, ppid=gate_pid, start_ticks=31
    )
    gate_arguments = [str(Path(sys.executable).resolve()), "fixture-gate"]
    wrapper_arguments = [
        str(Path(sys.executable).resolve()),
        "fixture-wrapper",
    ]
    executable_identity = _test_executable_identity()
    attempt_id = "f" * 64
    attempt_root = (
        paths["root"]
        / "preflight_launch_attempts"
        / "by_policy"
        / str(policy["policy_sha256"])
        / attempt_id
    )
    pane_log_path = attempt_root / "pane.log"
    pane_log_path.parent.mkdir(parents=True, exist_ok=True)
    pane_log_path.write_bytes(b"")
    pane_log_stat = pane_log_path.stat()
    pane_log = build_preflight_file_identity(
        path=str(pane_log_path.resolve()),
        device=int(pane_log_stat.st_dev),
        inode=int(pane_log_stat.st_ino),
        mode=int(pane_log_stat.st_mode),
        size=int(pane_log_stat.st_size),
    )
    started_registry = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_started_registry_v1"
        ),
        "attempt_id": attempt_id,
        "policy_sha256": policy["policy_sha256"],
        "started_at": "2026-07-28T00:00:00+00:00",
    }
    started_registry["launch_started_registry_sha256"] = (
        canonical_digest(
            started_registry, "launch_started_registry_sha256"
        )
    )
    started_registry_path = (
        paths["root"]
        / "preflight_launch_attempts"
        / "started"
        / f"{attempt_id}.json"
    )
    write_exclusive_json(started_registry_path, started_registry)
    started_registry_binding = module._artifact_binding(
        started_registry_path,
        started_registry["launch_started_registry_sha256"],
    )
    wrapper_path = control / "wrapper_claim.json"
    wrapper_started_path = attempt_root / "wrapper_started.json"
    receipt = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_receipt_v1"
        ),
        "attempt_id": attempt_id,
        "policy_sha256": policy["policy_sha256"],
        "verified_implementations": verified_implementations,
        "pane_gate_arguments": gate_arguments,
        "wrapper_arguments": wrapper_arguments,
        "python_executable": {
            "path": executable_identity["path"],
            "sha256": hashlib.sha256(
                Path(executable_identity["path"]).read_bytes()
            ).hexdigest(),
        },
        "started_registry": started_registry_binding,
        "pane_log": pane_log,
        "wrapper_claim_path": str(wrapper_path.resolve()),
        "wrapper_started_path": str(wrapper_started_path.resolve()),
        "gate_execution_terminal_path": str(
            (attempt_root / "gate_execution_terminal.json").resolve()
        ),
        "git": {},
        "started_at": "2026-07-28T00:00:00+00:00",
    }
    receipt["launch_receipt_sha256"] = canonical_digest(
        receipt, "launch_receipt_sha256"
    )
    receipt_path = attempt_root / "launch_receipt.json"
    write_exclusive_json(receipt_path, receipt)
    receipt_stat = receipt_path.stat()
    receipt_identity = {
        "path": str(receipt_path.resolve()),
        "device": int(receipt_stat.st_dev),
        "inode": int(receipt_stat.st_ino),
        "mode": int(receipt_stat.st_mode),
        "size": int(receipt_stat.st_size),
    }
    receipt_binding = module._artifact_binding(
        receipt_path, receipt["launch_receipt_sha256"]
    )
    gate_ready = build_preflight_gate_ready(
        launch_receipt=receipt_binding,
        launch_receipt_identity=receipt_identity,
        verified_implementations=verified_implementations,
        process=gate_process,
        wrapper_arguments=wrapper_arguments,
        ready_at="2026-07-28T00:00:01+00:00",
    )
    gate_ready_path = attempt_root / "pane_gate_ready.json"
    write_exclusive_json(gate_ready_path, gate_ready)
    gate_ready_binding = module._artifact_binding(
        gate_ready_path, gate_ready["pane_gate_ready_sha256"]
    )
    owner_seal = build_preflight_pane_owner_seal(
        server_pid=int(tmux_server["server_pid"]),
        server_start_ticks=int(
            tmux_server["server_process"]["start_ticks"]
        ),
        socket_path=str(tmux_server["socket_path"]),
        socket_device=int(tmux_server["socket_device"]),
        socket_inode=int(tmux_server["socket_inode"]),
        session=str(controller_tmux["session"]),
        pane=str(controller_tmux["pane"]),
        pane_pid=int(controller_tmux["pane_pid"]),
        pane_process=gate_process,
        owner_nonce="a" * 64,
        tmux_identity=controller_tmux,
        tmux_server=tmux_server,
    )
    tmux_started = build_preflight_tmux_started(
        launch_receipt=receipt_binding,
        launch_receipt_identity=receipt_identity,
        verified_implementations=verified_implementations,
        pane_gate_ready=gate_ready_binding,
        tmux_client={"returncode": 0, "stdout": "", "stderr": ""},
        owner_seal=owner_seal,
        started_at="2026-07-28T00:00:01+00:00",
        tmux_identity=controller_tmux,
        tmux_server=tmux_server,
    )
    tmux_started_path = attempt_root / "launch_tmux_started.json"
    write_exclusive_json(tmux_started_path, tmux_started)
    wrapper_started = build_preflight_wrapper_started(
        launch_receipt=receipt_binding,
        launch_receipt_identity=receipt_identity,
        verified_implementations=verified_implementations,
        pane_gate_ready=gate_ready_binding,
        pane_gate_process=gate_process,
        wrapper_arguments=wrapper_arguments,
        wrapper_process=wrapper_launch_process,
        wrapper_executable=executable_identity,
        started_at="2026-07-28T00:00:02+00:00",
        gate_ready=gate_ready,
    )
    write_exclusive_json(wrapper_started_path, wrapper_started)
    if not paths["checkpoint_plan"].is_file():
        plan = build_checkpoint_plan(
            paths["root"].parent, policy, paths["preflight_results"]
        )
        write_exclusive_json(paths["checkpoint_plan"], plan)
        request_paths = module.write_preflight_requests(
            plan, paths["preflight_requests"]
        )
        module._build_preflight_request_manifest(
            policy, paths, plan, request_paths
        )
    checkpoint_plan = module._artifact_binding(
        paths["checkpoint_plan"],
        load_json(
            paths["checkpoint_plan"], "fixture checkpoint plan"
        )["checkpoint_plan_sha256"],
    )
    preflight_request_manifest = module._artifact_binding(
        paths["preflight_request_manifest"],
        load_json(
            paths["preflight_request_manifest"],
            "fixture preflight request manifest",
        )["preflight_request_manifest_sha256"],
    )
    wrapper = build_preflight_claim_v3(
        attempt_id=attempt_id,
        preflight_launch_receipt=receipt_binding,
        preflight_launch_receipt_identity=receipt_identity,
        verified_implementations=verified_implementations,
        pane_gate_ready=gate_ready_binding,
        preflight_launch_tmux_started=module._artifact_binding(
            tmux_started_path,
            tmux_started["launch_tmux_started_sha256"],
        ),
        preflight_wrapper_started=module._artifact_binding(
            wrapper_started_path,
            wrapper_started["wrapper_started_sha256"],
        ),
        pane_gate_process=gate_process,
        wrapper_arguments=wrapper_arguments,
        wrapper_executable=executable_identity,
        pane_log=pane_log,
        git={},
        policy_sha256=policy["policy_sha256"],
        config=policy["policy_file"],
        checkpoint_plan=checkpoint_plan,
        preflight_request_manifest=preflight_request_manifest,
        controller_session=module.PREFLIGHT_CONTROLLER_SESSION,
        controller_tmux=controller_tmux,
        controller_tmux_server=tmux_server,
        observer_session=module.PREFLIGHT_OBSERVER_SESSION,
        command=module._expected_preflight_controller_command(
            policy, paths
        ),
        observer_command=module._expected_preflight_observer_command(
            policy, paths
        ),
        wrapper_pid=wrapper_pid,
        wrapper_process=wrapper_launch_process,
        wrapper_launch_process=wrapper_launch_process,
        started_at="2026-07-28T00:00:03+00:00",
        external_timeout_seconds=None,
        gate_ready=gate_ready,
        wrapper_started=wrapper_started,
    )
    write_exclusive_json(wrapper_path, wrapper)
    wrapper_binding = module._artifact_binding(
        wrapper_path, wrapper["wrapper_claim_sha256"]
    )
    pane = {
        "session": controller_tmux["session"],
        "pane": controller_tmux["pane"],
        "pane_pid": controller_tmux["pane_pid"],
        "pane_dead": False,
        "pane_dead_status": None,
    }
    tmux_started_binding = module._artifact_binding(
        tmux_started_path,
        tmux_started["launch_tmux_started_sha256"],
    )
    accepted = build_preflight_launch_accepted(
        attempt_id=attempt_id,
        launch_receipt=receipt_binding,
        launch_receipt_identity=receipt_identity,
        verified_implementations=verified_implementations,
        wrapper_claim=wrapper_binding,
        tmux_started=tmux_started_binding,
        pane=pane,
        pane_log_path=pane_log["path"],
        started_at="2026-07-28T00:00:00+00:00",
        accepted_at="2026-07-28T00:00:04+00:00",
    )
    accepted_path = attempt_root / "launch_accepted.json"
    write_exclusive_json(accepted_path, accepted)
    accepted_binding = module._artifact_binding(
        accepted_path, accepted["launch_accepted_sha256"]
    )
    terminal = build_preflight_ownership_terminal(
        launch_receipt=receipt_binding,
        launch_receipt_identity=receipt_identity,
        verified_implementations=verified_implementations,
        launch_accepted=accepted_binding,
        wrapper_claim=wrapper_binding,
        tmux_started=tmux_started_binding,
        pane=pane,
        pane_log=pane_log,
        started_at="2026-07-28T00:00:00+00:00",
        completed_at="2026-07-28T00:00:05+00:00",
    )
    terminal_path = attempt_root / "launch_terminal.json"
    write_exclusive_json(terminal_path, terminal)
    terminal_binding = module._artifact_binding(
        terminal_path,
        terminal["launch_terminal_sha256"],
    )
    release = build_preflight_ownership_release(
        launch_receipt=receipt_binding,
        launch_receipt_identity=receipt_identity,
        verified_implementations=verified_implementations,
        wrapper_claim=wrapper_binding,
        launch_accepted=accepted_binding,
        launch_terminal=terminal_binding,
        released_at="2026-07-28T00:00:06+00:00",
    )
    write_exclusive_json(
        attempt_root / "launch_ownership_release.json", release
    )
    return wrapper, wrapper_path, {
        "gate": gate_process,
        "wrapper": wrapper_launch_process,
        "gate_arguments": gate_arguments,
        "wrapper_arguments": wrapper_arguments,
    }


def _mock_preflight_wrapper_process_seals(
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    seals: Mapping[str, Any],
) -> None:
    original_readlink = module.os.readlink
    identities = {
        seals["gate"]["pid"]: seals["gate"],
        seals["wrapper"]["pid"]: seals["wrapper"],
    }
    commands = {
        seals["gate"]["pid"]: module._command_bytes(
            seals["gate_arguments"]
        ),
        seals["wrapper"]["pid"]: module._command_bytes(
            seals["wrapper_arguments"]
        ),
    }
    monkeypatch.setattr(
        module,
        "_launch_process_identity",
        lambda pid: dict(identities[pid]),
    )
    monkeypatch.setattr(
        module,
        "_process_command_bytes",
        lambda pid: commands[pid],
    )
    monkeypatch.setattr(
        module,
        "_process_command",
        lambda pid: (
            list(seals["gate_arguments"])
            if pid == seals["gate"]["pid"]
            else list(seals["wrapper_arguments"])
            if pid == seals["wrapper"]["pid"]
            else (_ for _ in ()).throw(AssertionError(pid))
        ),
    )
    monkeypatch.setattr(
        module.os,
        "readlink",
        lambda path: str(Path(sys.executable).resolve())
        if path
        in {
            f"/proc/{seals['gate']['pid']}/exe",
            f"/proc/{seals['wrapper']['pid']}/exe",
        }
        else original_readlink(path),
    )


def _write_preflight_observer_provenance_fixture(
    module: Any,
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    wrapper: Mapping[str, Any],
    wrapper_path: Path,
    observer_tmux: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
    observer_process: Mapping[str, int],
) -> tuple[dict[str, Any], Path]:
    verified_implementations = (
        _test_verified_preflight_implementations()
    )
    wrapper_binding = module._artifact_binding(
        wrapper_path, wrapper["wrapper_claim_sha256"]
    )
    gate_ready_path = (
        paths["preflight_control"] / "observer_gate_ready.json"
    )
    gate_release_path = (
        paths["preflight_control"] / "observer_gate_release.json"
    )
    bootstrap_path = (
        paths["preflight_control"] / "observer_bootstrap.json"
    )
    observer_command = module._expected_preflight_observer_command(
        policy, paths
    )
    gate_ready = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_observer_gate_ready_v1"
        ),
        "policy_sha256": policy["policy_sha256"],
        "verified_implementations": verified_implementations,
        "wrapper_claim": wrapper_binding,
        "observer_session": module.PREFLIGHT_OBSERVER_SESSION,
        "owner_nonce": "a" * 64,
        "process": dict(observer_process),
        "gate_executable": sys.executable,
        "gate_command": [sys.executable, "gate"],
        "tmux": dict(observer_tmux),
        "tmux_server": dict(tmux_server),
        "release_path": str(gate_release_path.resolve()),
        "bootstrap_path": str(bootstrap_path.resolve()),
        "observer_command": observer_command,
        "published_at": module._utc_now(),
    }
    gate_ready["observer_gate_ready_sha256"] = canonical_digest(
        gate_ready, "observer_gate_ready_sha256"
    )
    write_exclusive_json(gate_ready_path, gate_ready)
    gate_ready_binding = module._artifact_binding(
        gate_ready_path, gate_ready["observer_gate_ready_sha256"]
    )
    gate_release = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_observer_gate_release_v1"
        ),
        "policy_sha256": policy["policy_sha256"],
        "verified_implementations": verified_implementations,
        "wrapper_claim": wrapper_binding,
        "observer_gate_ready": gate_ready_binding,
        "observer_session": module.PREFLIGHT_OBSERVER_SESSION,
        "owner_nonce": "a" * 64,
        "observer_command": observer_command,
        "released_at": module._utc_now(),
    }
    gate_release["observer_gate_release_sha256"] = canonical_digest(
        gate_release, "observer_gate_release_sha256"
    )
    write_exclusive_json(gate_release_path, gate_release)
    bootstrap = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_observer_bootstrap_v1"
        ),
        "policy_sha256": policy["policy_sha256"],
        "verified_implementations": verified_implementations,
        "wrapper_claim": wrapper_binding,
        "observer_session": module.PREFLIGHT_OBSERVER_SESSION,
        "owner_nonce": "a" * 64,
        "process": dict(observer_process),
        "executable": sys.executable,
        "executable_identity": _test_executable_identity(),
        "command": observer_command,
        "tmux": dict(observer_tmux),
        "published_at": module._utc_now(),
    }
    bootstrap["observer_bootstrap_sha256"] = canonical_digest(
        bootstrap, "observer_bootstrap_sha256"
    )
    write_exclusive_json(bootstrap_path, bootstrap)
    launch = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_observer_launch_v3",
        "policy_sha256": policy["policy_sha256"],
        "verified_implementations": verified_implementations,
        "wrapper_claim": wrapper_binding,
        "wrapper_claim_sha256": wrapper["wrapper_claim_sha256"],
        "observer_session": module.PREFLIGHT_OBSERVER_SESSION,
        "command": observer_command,
        "observer_gate_ready": gate_ready_binding,
        "observer_gate_release": module._artifact_binding(
            gate_release_path,
            gate_release["observer_gate_release_sha256"],
        ),
        "status": "launched",
        "failure": None,
        "tmux": dict(observer_tmux),
        "tmux_server": dict(tmux_server),
        "tmux_owner_seal": _test_tmux_owner_seal(
            observer_tmux, tmux_server, server_start_ticks=20
        ),
        "observer_bootstrap": module._artifact_binding(
            bootstrap_path, bootstrap["observer_bootstrap_sha256"]
        ),
        "process": dict(observer_process),
    }
    launch["observer_launch_sha256"] = canonical_digest(
        launch, "observer_launch_sha256"
    )
    launch_path = paths["preflight_control"] / "observer_launch.json"
    write_exclusive_json(launch_path, launch)
    return launch, launch_path


def _write_preflight_process_start_fixture(
    module: Any,
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    pid: int = 100,
) -> tuple[dict[str, Any], Path]:
    process_start = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_controller_process_start_v1"
        ),
        "policy_sha256": policy["policy_sha256"],
        "verified_implementations": (
            _test_verified_preflight_implementations()
        ),
        "process": {"pid": pid, "pgid": pid, "start_ticks": 10},
    }
    process_start["controller_process_start_sha256"] = canonical_digest(
        process_start, "controller_process_start_sha256"
    )
    path = paths["preflight_control"] / "controller_process_start.json"
    write_exclusive_json(path, process_start)
    return process_start, path


def _write_preflight_process_exit_fixture(
    module: Any,
    policy: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    wrapper: Mapping[str, Any],
    observer_launch: Mapping[str, Any],
    observer_launch_path: Path,
    process_start: Mapping[str, Any],
    process_start_path: Path,
    exit_code: int,
    controller_terminal: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], Path]:
    process_exit = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_controller_process_exit_v2"
        ),
        "policy_sha256": policy["policy_sha256"],
        "wrapper_claim_sha256": wrapper["wrapper_claim_sha256"],
        "observer_launch": module._artifact_binding(
            observer_launch_path,
            observer_launch["observer_launch_sha256"],
        ),
        "controller_process_start": module._artifact_binding(
            process_start_path,
            process_start["controller_process_start_sha256"],
        ),
        "observer_stop": None,
        "controller_pid": process_start["process"]["pid"],
        "command": module._expected_preflight_controller_command(
            policy, paths
        ),
        "exit_code": exit_code,
        "controller_terminal": (
            None if controller_terminal is None else dict(controller_terminal)
        ),
        "signal": None,
        "launch_failure": None,
        "controller_process_log": None,
        "controller_claim": None,
        "completed_at": module._utc_now(),
    }
    process_exit["controller_process_exit_sha256"] = canonical_digest(
        process_exit, "controller_process_exit_sha256"
    )
    path = paths["preflight_control"] / "controller_process_exit.json"
    write_exclusive_json(path, process_exit)
    return process_exit, path


@pytest.mark.parametrize(
    ("error", "absent_after", "expected"),
    (
        (ProcessLookupError(), True, "cleaned_detached_process_absent"),
        (ProcessLookupError(), False, "raises"),
        (PermissionError(), False, "raises"),
    ),
)
def test_wrapper_killpg_error_classification(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    absent_after: bool,
    expected: str,
) -> None:
    wrapper = _wrapper_module()
    sealed = _test_process_identity(401, start_ticks=77)
    state = {"kill_called": False}
    monkeypatch.setattr(
        wrapper,
        "_tmux_identity",
        lambda _session: (_ for _ in ()).throw(
            wrapper.TmuxTargetAbsent("missing")
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_tmux_server_identity",
        lambda _target=None: (_ for _ in ()).throw(
            wrapper.TmuxTargetAbsent("missing")
        ),
    )

    def identity(_pid: int):
        if state["kill_called"] and absent_after:
            return None
        return dict(sealed)

    monkeypatch.setattr(wrapper, "_process_identity", identity)
    monkeypatch.setattr(
        wrapper,
        "_process_identity_state",
        lambda _pid: (dict(sealed), "S"),
    )
    monkeypatch.setattr(
        wrapper,
        "_read_process_stat",
        lambda _pid: (
            None
            if state["kill_called"] and absent_after
            else (dict(sealed), "S")
        ),
    )

    def killpg(_pgid: int, _signal: int):
        state["kill_called"] = True
        raise error

    monkeypatch.setattr(wrapper.os, "killpg", killpg)
    tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%1",
        "pane_pid": 401,
        "pane_current_command": "python",
    }
    server = _test_tmux_server_identity(
        Path("/tmp/tmux.sock"),
        server_pid=301,
        server_process=_test_process_identity(301, start_ticks=55),
    )
    if expected == "raises":
        with pytest.raises(RuntimeError):
            wrapper._terminate_bound_observer(
                tmux,
                server,
                _test_tmux_owner_seal(tmux, server),
                sealed,
            )
    else:
        result = wrapper._terminate_bound_observer(
            tmux,
            server,
            _test_tmux_owner_seal(tmux, server),
            sealed,
        )
        assert result["status"] == expected
        assert result["process_residual"] is False


def test_wrapper_cleanup_permission_failure_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    config = (
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    policy_root = tmp_path / "campaign" / "by_policy" / ("6" * 64)
    _prepare_wrapper_contract_inputs(wrapper, policy_root)
    _patch_wrapper_tmux(wrapper, monkeypatch, tmp_path)
    monkeypatch.setattr(
        wrapper,
        "_terminate_bound_observer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("fixture permission denied")
        ),
    )
    value = wrapper.run_wrapped_controller(
        repo_root=tmp_path,
        policy_root=policy_root,
        policy_sha256="6" * 64,
        config=config,
        observer_command=[sys.executable, "-c", "pass"],
        command=[sys.executable, "-c", "raise SystemExit(0)"],
    )
    cleanup = load_json(
        Path(value["observer_cleanup"]["path"]),
        "durable permission cleanup",
    )
    assert value["exit_code"] != 0
    assert cleanup["status"] == "cleanup_failed"
    assert cleanup["failure"]["type"] == "PermissionError"
    assert cleanup["session_residual"] is True
    assert cleanup["process_residual"] is True


@pytest.mark.parametrize(
    "mutation",
    ("named_pane", "extra_server_field"),
)
def test_wrapper_public_tmux_identity_remains_four_field_opaque(
    mutation: str,
) -> None:
    wrapper = _wrapper_module()
    identity = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%17",
        "pane_pid": 401,
        "pane_current_command": "python",
    }
    if mutation == "named_pane":
        identity["pane"] = "monitor:0.0"
    else:
        identity["server_pid"] = 99
    with pytest.raises(RuntimeError, match="public tmux identity"):
        wrapper._validate_tmux_identity(
            identity, wrapper.OBSERVER_SESSION
        )


def test_wrapper_kill_pane_check_to_kill_replacement_preserves_foreign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    sealed_tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%17",
        "pane_pid": 401,
        "pane_current_command": "python",
    }
    sealed_server = _test_tmux_server_identity(
        Path("/tmp/tmux-test/default"),
        server_pid=301,
        server_process=_test_process_identity(301, start_ticks=55),
    )
    sealed_process = _test_process_identity(401, start_ticks=77)
    sealed_owner = _test_tmux_owner_seal(
        sealed_tmux,
        sealed_server,
        server_start_ticks=55,
        pane_process=sealed_process,
    )
    foreign_tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%18",
        "pane_pid": 402,
        "pane_current_command": "python",
    }
    state = {"replaced": False, "foreign_alive": True}
    commands: list[list[str]] = []
    monkeypatch.setattr(
        wrapper,
        "_tmux_server_identity",
        lambda _target=None: dict(sealed_server),
    )
    monkeypatch.setattr(
        wrapper,
        "_tmux_identity",
        lambda _session: (
            dict(foreign_tmux) if state["replaced"] else dict(sealed_tmux)
        ),
    )

    def pane_identity(_pane: str):
        if state["replaced"]:
            raise wrapper.TmuxTargetAbsent("can't find pane: %17")
        return dict(sealed_tmux)

    monkeypatch.setattr(wrapper, "_tmux_pane_identity", pane_identity)
    monkeypatch.setattr(
        wrapper,
        "_process_identity",
        lambda _pid: (
            None if state["replaced"] else dict(sealed_process)
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_process_identity_state",
        lambda _pid: (
            None if state["replaced"] else (dict(sealed_process), "S")
        ),
    )
    monkeypatch.setattr(
        wrapper,
        "_read_process_stat",
        lambda _pid: (
            None
            if state["replaced"]
            else (dict(sealed_process), "S")
        ),
    )

    def conditional(owner):
        assert dict(owner) == sealed_owner
        commands.append(["conditional-kill"])
        state["replaced"] = True
        return (
            "condition_rejected",
            types.SimpleNamespace(
                returncode=0,
                stdout=wrapper.TMUX_CONDITIONAL_KILL_REJECTED,
                stderr="",
            ),
        )

    monkeypatch.setattr(wrapper, "_conditional_kill_tmux_owner", conditional)
    result = wrapper._terminate_bound_observer(
        sealed_tmux,
        sealed_server,
        sealed_owner,
        sealed_process,
    )
    assert commands == [["conditional-kill"]]
    assert state["foreign_alive"] is True
    assert result["session_residual"] is False
    assert result["process_residual"] is False
    assert (
        result["tmux_kill_failure"]["type"]
        == "TmuxConditionalKillRejected"
    )


def test_wrapper_kill_pane_failure_with_live_seal_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    sealed_tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%21",
        "pane_pid": 501,
        "pane_current_command": "python",
    }
    sealed_server = _test_tmux_server_identity(
        Path("/tmp/tmux-test/default"),
        server_pid=301,
        server_process=_test_process_identity(301, start_ticks=55),
    )
    sealed_process = _test_process_identity(501, start_ticks=88)
    sealed_owner = _test_tmux_owner_seal(
        sealed_tmux,
        sealed_server,
        pane_process=sealed_process,
    )
    monkeypatch.setattr(
        wrapper,
        "_tmux_server_identity",
        lambda _target=None: dict(sealed_server),
    )
    monkeypatch.setattr(
        wrapper, "_tmux_identity", lambda _session: dict(sealed_tmux)
    )
    monkeypatch.setattr(
        wrapper, "_tmux_pane_identity", lambda _pane: dict(sealed_tmux)
    )
    monkeypatch.setattr(
        wrapper, "_process_identity", lambda _pid: dict(sealed_process)
    )
    monkeypatch.setattr(
        wrapper,
        "_process_identity_state",
        lambda _pid: (dict(sealed_process), "S"),
    )
    monkeypatch.setattr(
        wrapper,
        "_conditional_kill_tmux_owner",
        lambda _owner: (
            "command_failed",
            types.SimpleNamespace(
            returncode=1,
            stdout="",
                stderr="permission denied",
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="remained live"):
        wrapper._terminate_bound_observer(
            sealed_tmux,
            sealed_server,
            sealed_owner,
            sealed_process,
        )


@pytest.mark.parametrize(
    ("case", "live_identity", "expected_status"),
    (
        (
            "exact_owner",
            {
                "server_pid": 301,
                "pane": "%17",
                "pane_pid": 401,
                "owner_nonce": "a" * 64,
            },
            "executed",
        ),
        (
            "same_server_pane_replacement",
            {
                "server_pid": 301,
                "pane": "%18",
                "pane_pid": 402,
                "owner_nonce": "a" * 64,
            },
            "condition_rejected",
        ),
        (
            "same_pane_id_process_replacement",
            {
                "server_pid": 301,
                "pane": "%17",
                "pane_pid": 402,
                "owner_nonce": "a" * 64,
            },
            "condition_rejected",
        ),
        (
            "replacement_server_reuses_pid_socket_name_and_pane",
            {
                "server_pid": 301,
                "pane": "%17",
                "pane_pid": 401,
                "owner_nonce": "b" * 64,
            },
            "condition_rejected",
        ),
    ),
)
def test_wrapper_atomic_tmux_owner_kill_is_nonce_and_pane_bound(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    live_identity: dict[str, Any],
    expected_status: str,
) -> None:
    wrapper = _wrapper_module()
    sealed_tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%17",
        "pane_pid": 401,
        "pane_current_command": "python",
    }
    sealed_server = _test_tmux_server_identity(
        Path("/tmp/tmux-test/default"),
        server_pid=301,
        server_process=_test_process_identity(301, start_ticks=55),
    )
    owner = _test_tmux_owner_seal(sealed_tmux, sealed_server)
    calls: list[list[str]] = []
    foreign_killed = {"value": False}
    monkeypatch.setattr(
        wrapper,
        "_validate_tmux_owner_host_identity",
        lambda actual: actual == owner
        or (_ for _ in ()).throw(AssertionError("owner differs")),
    )

    def run(command: list[str], **kwargs):
        calls.append(list(command))
        assert kwargs == {"capture_output": True, "text": True}
        assert command[:8] == [
            "tmux",
            "-S",
            owner["socket_path"],
            "if-shell",
            "-t",
            owner["pane"],
            "-F",
            command[7],
        ]
        condition = command[7]
        assert f"#{{==:#{{pid}},{owner['server_pid']}}}" in condition
        assert (
            f"#{{==:#{{session_name}},{owner['session']}}}"
            in condition
        )
        assert f"#{{==:#{{pane_id}},{owner['pane']}}}" in condition
        assert f"#{{==:#{{pane_pid}},{owner['pane_pid']}}}" in condition
        assert (
            f"#{{==:#{{E:{wrapper.TMUX_OWNER_ENV}}},"
            f"{owner['owner_nonce']}}}"
        ) in condition
        assert command[8] == f"kill-pane -t {owner['pane']}"
        assert command[9] == (
            "display-message -p "
            f"{wrapper.TMUX_CONDITIONAL_KILL_REJECTED}"
        )
        exact = live_identity == {
            "server_pid": owner["server_pid"],
            "pane": owner["pane"],
            "pane_pid": owner["pane_pid"],
            "owner_nonce": owner["owner_nonce"],
        }
        if exact:
            foreign_killed["value"] = True
            return types.SimpleNamespace(
                returncode=0, stdout="", stderr=""
            )
        return types.SimpleNamespace(
            returncode=0,
            stdout=wrapper.TMUX_CONDITIONAL_KILL_REJECTED + "\n",
            stderr="",
        )

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    status, _ = wrapper._conditional_kill_tmux_owner(owner)
    assert status == expected_status, case
    assert len(calls) == 1
    assert foreign_killed["value"] is (expected_status == "executed")


def test_wrapper_remain_on_exit_replacement_is_rejected_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    sealed_tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%27",
        "pane_pid": 501,
        "pane_current_command": "python",
    }
    sealed_server = _test_tmux_server_identity(
        Path("/tmp/tmux-test/default"),
        server_pid=401,
        server_process=_test_process_identity(401, start_ticks=55),
    )
    owner = _test_tmux_owner_seal(sealed_tmux, sealed_server)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wrapper,
        "_validate_tmux_owner_host_identity",
        lambda actual: actual == owner
        or (_ for _ in ()).throw(AssertionError("owner differs")),
    )

    def run(command: list[str], **kwargs):
        calls.append(list(command))
        assert kwargs == {"capture_output": True, "text": True}
        assert command[:7] == [
            "tmux",
            "-S",
            owner["socket_path"],
            "if-shell",
            "-t",
            owner["pane"],
            "-F",
        ]
        condition = command[7]
        assert (
            f"#{{==:#{{session_name}},{owner['session']}}}"
            in condition
        )
        assert (
            f"#{{==:#{{E:{wrapper.TMUX_OWNER_ENV}}},"
            f"{owner['owner_nonce']}}}"
        ) in condition
        assert command[8] == (
            f"set-window-option -t {owner['pane']} "
            "remain-on-exit on"
        )
        assert command[9] == (
            "display-message -p "
            f"{wrapper.TMUX_CONDITIONAL_REMAIN_REJECTED}"
        )
        return types.SimpleNamespace(
            returncode=0,
            stdout=wrapper.TMUX_CONDITIONAL_REMAIN_REJECTED + "\n",
            stderr="",
        )

    monkeypatch.setattr(wrapper.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="owner condition rejected"):
        wrapper._set_observer_remain_on_exit(owner)
    assert len(calls) == 1
    assert "set-window-option" not in calls[0][:8]


@pytest.mark.parametrize("failure", ("server_start_ticks", "socket_inode"))
def test_wrapper_tmux_owner_host_precheck_failure_issues_no_command(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    wrapper = _wrapper_module()
    tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%17",
        "pane_pid": 401,
        "pane_current_command": "python",
    }
    server = _test_tmux_server_identity(
        Path("/tmp/tmux-test/default"),
        server_pid=301,
        server_process=_test_process_identity(301, start_ticks=55),
    )
    owner = _test_tmux_owner_seal(tmux, server)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wrapper,
        "_validate_tmux_owner_host_identity",
        lambda _owner: (_ for _ in ()).throw(
            RuntimeError(f"tmux owner {failure} differs")
        ),
    )
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(list(command)),
    )
    with pytest.raises(RuntimeError, match=failure):
        wrapper._conditional_kill_tmux_owner(owner)
    assert calls == []


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_wrapper_real_tmux_nonce_change_rejects_without_killing_pane(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper_module()
    socket_path = tmp_path / "owner-test.sock"
    session = "safa-owner-atomic-test"
    old_nonce = "a" * 64
    new_nonce = "b" * 64
    try:
        subprocess.run(
            [
                "tmux",
                "-S",
                str(socket_path),
                "new-session",
                "-d",
                "-s",
                session,
                "-e",
                f"{wrapper.TMUX_OWNER_ENV}={old_nonce}",
                sys.executable,
                "-c",
                "import time;time.sleep(30)",
            ],
            check=True,
        )
        identity_row = subprocess.run(
            [
                "tmux",
                "-S",
                str(socket_path),
                "display-message",
                "-p",
                "-t",
                session,
                "#{pid}\t#{pane_id}\t#{pane_pid}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().split("\t")
        socket_value = os.lstat(socket_path)
        server_pid = int(identity_row[0])
        server_process = wrapper._require_process_identity(
            server_pid, "real tmux owner test server"
        )
        owner = {
            "server_pid": server_pid,
            "server_start_ticks": server_process["start_ticks"],
            "socket_path": str(socket_path),
            "socket_device": int(socket_value.st_dev),
            "socket_inode": int(socket_value.st_ino),
            "session": session,
            "pane": identity_row[1],
            "pane_pid": int(identity_row[2]),
            "pane_process": wrapper._launch_process_identity(
                int(identity_row[2])
            ),
            "owner_nonce": old_nonce,
        }
        subprocess.run(
            [
                "tmux",
                "-S",
                str(socket_path),
                "set-environment",
                "-t",
                session,
                wrapper.TMUX_OWNER_ENV,
                new_nonce,
            ],
            check=True,
        )
        status, command_result = wrapper._conditional_kill_tmux_owner(
            owner
        )
        assert status == "condition_rejected"
        assert command_result.returncode == 0
        assert subprocess.run(
            [
                "tmux",
                "-S",
                str(socket_path),
                "has-session",
                "-t",
                session,
            ],
            capture_output=True,
            text=True,
        ).returncode == 0
    finally:
        subprocess.run(
            ["tmux", "-S", str(socket_path), "kill-server"],
            capture_output=True,
            text=True,
        )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_wrapper_real_tmux_replacement_rejects_remain_on_exit(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper_module()
    socket_path = tmp_path / "remain-owner-test.sock"
    session = "safa-remain-owner-atomic-test"
    old_nonce = "a" * 64
    new_nonce = "b" * 64
    try:
        subprocess.run(
            [
                "tmux",
                "-S",
                str(socket_path),
                "new-session",
                "-d",
                "-s",
                session,
                "-e",
                f"{wrapper.TMUX_OWNER_ENV}={old_nonce}",
                sys.executable,
                "-c",
                "import time;time.sleep(30)",
            ],
            check=True,
        )
        identity_row = subprocess.run(
            [
                "tmux",
                "-S",
                str(socket_path),
                "display-message",
                "-p",
                "-t",
                session,
                "#{pid}\t#{pane_id}\t#{pane_pid}",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().split("\t")
        socket_value = os.lstat(socket_path)
        server_pid = int(identity_row[0])
        server_process = wrapper._require_process_identity(
            server_pid, "real tmux remain owner server"
        )
        owner = {
            "server_pid": server_pid,
            "server_start_ticks": server_process["start_ticks"],
            "socket_path": str(socket_path),
            "socket_device": int(socket_value.st_dev),
            "socket_inode": int(socket_value.st_ino),
            "session": session,
            "pane": identity_row[1],
            "pane_pid": int(identity_row[2]),
            "pane_process": wrapper._launch_process_identity(
                int(identity_row[2])
            ),
            "owner_nonce": old_nonce,
        }
        subprocess.run(
            [
                "tmux",
                "-S",
                str(socket_path),
                "set-window-option",
                "-t",
                session,
                "remain-on-exit",
                "off",
            ],
            check=True,
        )
        subprocess.run(
            [
                "tmux",
                "-S",
                str(socket_path),
                "set-environment",
                "-t",
                session,
                wrapper.TMUX_OWNER_ENV,
                new_nonce,
            ],
            check=True,
        )
        with pytest.raises(RuntimeError, match="owner condition rejected"):
            wrapper._set_observer_remain_on_exit(owner)
        remain_value = subprocess.run(
            [
                "tmux",
                "-S",
                str(socket_path),
                "show-window-options",
                "-v",
                "-t",
                session,
                "remain-on-exit",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert remain_value == "off"
        assert subprocess.run(
            [
                "tmux",
                "-S",
                str(socket_path),
                "has-session",
                "-t",
                session,
            ],
            capture_output=True,
            text=True,
        ).returncode == 0
    finally:
        subprocess.run(
            ["tmux", "-S", str(socket_path), "kill-server"],
            capture_output=True,
            text=True,
        )


def test_wrapper_server_replacement_is_never_terminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_module()
    sealed_tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%31",
        "pane_pid": 601,
        "pane_current_command": "python",
    }
    sealed_server = _test_tmux_server_identity(
        Path("/tmp/tmux-test/default"),
        server_pid=301,
        server_process=_test_process_identity(301, start_ticks=55),
    )
    foreign_server = _test_tmux_server_identity(
        Path("/tmp/tmux-test/default"),
        server_pid=302,
        server_process=_test_process_identity(302, start_ticks=56),
    )
    foreign_tmux = {
        "session": wrapper.OBSERVER_SESSION,
        "pane": "%32",
        "pane_pid": 602,
        "pane_current_command": "python",
    }
    sealed_process = _test_process_identity(601, start_ticks=99)
    sealed_owner = _test_tmux_owner_seal(
        sealed_tmux,
        sealed_server,
        pane_process=sealed_process,
    )
    monkeypatch.setattr(
        wrapper,
        "_tmux_server_identity",
        lambda _target=None: dict(foreign_server),
    )
    monkeypatch.setattr(
        wrapper, "_tmux_identity", lambda _session: dict(foreign_tmux)
    )
    monkeypatch.setattr(wrapper, "_process_identity", lambda _pid: None)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(list(command)),
    )
    result = wrapper._terminate_bound_observer(
        sealed_tmux,
        sealed_server,
        sealed_owner,
        sealed_process,
    )
    assert commands == []
    assert result["status"] == "identity_replaced_not_terminated"
    assert result["observed_tmux"] == foreign_tmux
    assert result["session_residual"] is False
    assert result["process_residual"] is False


def test_wrapper_records_native_stderr_and_sigkill_without_controller_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    config = (
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    policy_root = tmp_path / "campaign" / "by_policy" / ("1" * 64)
    _prepare_wrapper_contract_inputs(wrapper, policy_root)
    _patch_wrapper_tmux(wrapper, monkeypatch, tmp_path)
    value = wrapper.run_wrapped_controller(
        repo_root=tmp_path,
        policy_root=policy_root,
        policy_sha256="1" * 64,
        config=config,
        observer_command=[sys.executable, "-c", "pass"],
        command=[
            sys.executable,
            "-c",
            (
                "import os,signal;"
                "os.write(2,b'native-before-kill\\n');"
                "os.kill(os.getpid(),signal.SIGKILL)"
            ),
        ],
    )
    assert value["exit_code"] == 137
    assert value["signal"] == 9
    assert value["controller_claim"] is None
    assert value["controller_terminal"] is None
    log_path = Path(value["controller_process_log"]["path"])
    assert log_path.read_bytes() == b"native-before-kill\n"
    assert load_json(
        policy_root / "preflight_control" / "wrapper_exit.json", "wrapper exit"
    ) == value


def test_wrapper_records_pre_main_failure_without_controller_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    config = (
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    policy_root = tmp_path / "campaign" / "by_policy" / ("2" * 64)
    _prepare_wrapper_contract_inputs(wrapper, policy_root)
    _patch_wrapper_tmux(wrapper, monkeypatch, tmp_path)
    value = wrapper.run_wrapped_controller(
        repo_root=tmp_path,
        policy_root=policy_root,
        policy_sha256="2" * 64,
        config=config,
        observer_command=[sys.executable, "-c", "pass"],
        command=[
            sys.executable,
            "-c",
            "import os,sys;os.write(1,b'pre-main\\n');sys.exit(2)",
        ],
    )
    assert value["exit_code"] == 2
    assert value["signal"] is None
    assert value["controller_claim"] is None
    assert value["controller_terminal"] is None
    assert Path(value["controller_process_log"]["path"]).read_bytes() == b"pre-main\n"


def test_preflight_tmux_launcher_has_gate_terminal_and_no_timeout(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    commands = module._tmux_commands(
        policy, policy_path, tmp_path / "campaign", "preflight"
    )
    controller = " ".join(commands["controller"])
    assert "run_canonical_preflight_launcher.py" in controller
    assert "run_canonical_preflight_wrapper.py" not in controller
    assert "--policy-sha256" in controller
    assert "timeout" not in controller.lower()


def test_current_policy_preflight_refuses_partial_result_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", "c" * 64)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    plan = build_checkpoint_plan(tmp_path, policy, paths["preflight_results"])
    request = plan["preflight_requests"][0]
    write_exclusive_json(
        paths["preflight_requests"]
        / f"{request['checkpoint_sha256']}__{request['checkpoint_model']}.json",
        request,
    )
    write_exclusive_json(paths["preflight_results"] / "partial.json", {"partial": True})
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module, "_assert_preflight_observer_live", lambda *_args: None
    )
    guard = types.SimpleNamespace(raise_if_violated=lambda: None)
    with pytest.raises(CanonicalScreeningError, match="refuses result reuse"):
        module.materialize_preflights(
            policy, paths, guard, "d" * 64, {"sha256": "e" * 64}
        )


def test_write_exclusive_json_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    write_exclusive_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_exclusive_json(path, {"value": 2})


def test_atomic_publish_exposes_only_complete_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ready.json"
    value = {"contract": "ready", "rows": list(range(1000))}
    original_link = __import__("os").link
    observed = []

    def inspect_link(source, target):
        assert Path(target) == path
        assert not path.exists()
        observed.append(load_json(Path(source), "temporary publication"))
        return original_link(source, target)

    module = sys.modules[publish_exclusive_json.__module__]
    monkeypatch.setattr(module.os, "link", inspect_link)
    publish_exclusive_json(path, value)
    assert observed == [value]
    assert load_json(path, "published ready") == value
    assert list(tmp_path.glob(".ready.json.publish-*")) == []


def test_atomic_publish_race_has_one_winner_and_valid_final(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observer_ready.json"
    barrier = threading.Barrier(2)
    successes = []
    failures = []

    def publish(value: dict) -> None:
        barrier.wait()
        try:
            publish_exclusive_json(path, value)
            successes.append(value)
        except CanonicalScreeningError as exc:
            failures.append(exc)

    values = [{"writer": 1}, {"writer": 2}]
    threads = [
        threading.Thread(target=publish, args=(value,)) for value in values
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(successes) == 1
    assert len(failures) == 1
    assert load_json(path, "race winner") == successes[0]
    assert list(tmp_path.glob(".observer_ready.json.publish-*")) == []


def test_preflight_wrapper_exclusive_publish_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready", "rows": list(range(1000))}
    original_link = wrapper.os.link
    observed = []

    def inspect_link(source, target, **kwargs):
        assert target == path.name
        assert kwargs["src_dir_fd"] == kwargs["dst_dir_fd"]
        assert kwargs["follow_symlinks"] is False
        assert not path.exists()
        observed.append(
            load_json(
                tmp_path / source, "wrapper temporary publication"
            )
        )
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(wrapper.os, "link", inspect_link)
    wrapper._write_exclusive(path, value)
    assert observed == [value]
    assert load_json(path, "wrapper published ready") == value
    assert list(tmp_path.glob(".observer_gate_ready.json.publish-*")) == []
    monkeypatch.setattr(wrapper.os, "link", original_link)
    with pytest.raises(
        wrapper.ExclusivePublishError, match="collision"
    ):
        wrapper._write_exclusive(path, {"contract": "replacement"})
    assert load_json(path, "wrapper original ready") == value
    assert list(tmp_path.glob(".observer_gate_ready.json.publish-*")) == []


def test_preflight_launcher_creates_presealed_fault_channel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    path = tmp_path / "wrapper_fault.channel"
    original_fsync = launcher.os.fsync
    fsync_identities = []

    def record_fsync(descriptor: int) -> None:
        value = os.fstat(descriptor)
        fsync_identities.append(
            (
                stat.S_IFMT(value.st_mode),
                int(value.st_dev),
                int(value.st_ino),
            )
        )
        original_fsync(descriptor)

    monkeypatch.setattr(launcher.os, "fsync", record_fsync)
    binding = launcher._create_fault_channel(path)
    opened = path.stat()
    assert binding["path"] == str(path)
    assert binding["device"] == opened.st_dev
    assert binding["inode"] == opened.st_ino
    assert binding["uid"] == os.geteuid()
    assert binding["nlink"] == 1
    assert binding["size"] == 0
    assert binding["sha256"] == hashlib.sha256(b"").hexdigest()
    assert stat.S_IMODE(opened.st_mode) == 0o600
    assert path.read_bytes() == b""
    assert [item[0] for item in fsync_identities] == [
        stat.S_IFREG,
        stat.S_IFDIR,
    ]
    with pytest.raises(FileExistsError):
        launcher._create_fault_channel(path)
    assert path.read_bytes() == b""


def _fault_channel_receipt(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "attempt_id": "a" * 64,
        "controller_owner_nonce": "b" * 64,
        "launch_receipt_sha256": "c" * 64,
        "bindings": {
            "wrapper": {
                "path": "/verified/preflight_wrapper.py",
                "sha256": "d" * 64,
            }
        },
        "fault_channel": dict(binding),
    }


def test_preflight_fault_channel_fd_is_sealed_without_path_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    wrapper = _wrapper_module()
    tmp_path.chmod(0o755)
    path = tmp_path / "wrapper_fault.channel"
    binding = launcher._create_fault_channel(path)
    descriptor = launcher._open_presealed_fault_channel(
        tmp_path, binding
    )
    try:
        assert os.get_inheritable(descriptor) is False
        os.set_inheritable(descriptor, True)
        monkeypatch.setenv(
            wrapper.FAULT_CHANNEL_FD_ENV, str(descriptor)
        )
        monkeypatch.setattr(
            wrapper.os,
            "open",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("wrapper reopened fault channel path")
            ),
        )
        context = wrapper._bind_inherited_fault_channel(
            _fault_channel_receipt(binding)
        )
        assert context["descriptor"] == descriptor
        assert context["binding"] == binding
        assert context["attempt_id"] == "a" * 64
        assert context["owner_nonce"] == "b" * 64
        assert context["launch_receipt_sha256"] == "c" * 64
        assert context["publisher"]["sha256"] == "d" * 64
        assert os.get_inheritable(descriptor) is False
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "raw_descriptor",
    ("-1", "0", "1", "2", "+3", "03", "3x", ""),
)
def test_preflight_fault_channel_rejects_noncanonical_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    raw_descriptor: str,
) -> None:
    wrapper = _wrapper_module()
    monkeypatch.setenv(
        wrapper.FAULT_CHANNEL_FD_ENV, raw_descriptor
    )
    with pytest.raises(RuntimeError, match="descriptor"):
        wrapper._bind_inherited_fault_channel(
            _fault_channel_receipt({})
        )


def test_preflight_fault_channel_rejects_uninherited_or_wrong_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    wrapper = _wrapper_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "wrapper_fault.channel"
    )
    descriptor = launcher._open_presealed_fault_channel(
        tmp_path, binding
    )
    try:
        monkeypatch.setenv(
            wrapper.FAULT_CHANNEL_FD_ENV, str(descriptor)
        )
        with pytest.raises(RuntimeError, match="identity differs"):
            wrapper._bind_inherited_fault_channel(
                _fault_channel_receipt(binding)
            )
        os.set_inheritable(descriptor, True)
        wrong = dict(binding)
        wrong["inode"] += 1
        with pytest.raises(RuntimeError, match="identity differs"):
            wrapper._bind_inherited_fault_channel(
                _fault_channel_receipt(wrong)
            )
        wrong = dict(binding)
        wrong["uid"] += 1
        with pytest.raises(RuntimeError, match="identity differs"):
            wrapper._bind_inherited_fault_channel(
                _fault_channel_receipt(wrong)
            )
        wrong_receipt = _fault_channel_receipt(binding)
        wrong_receipt["attempt_id"] = "wrong"
        with pytest.raises(
            RuntimeError, match="receipt binding differs"
        ):
            wrapper._bind_inherited_fault_channel(
                wrong_receipt
            )
    finally:
        os.close(descriptor)


def test_preflight_gate_fault_channel_close_error_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()

    def fail_close(_descriptor: int) -> None:
        raise OSError(5, "fixture fault channel close failure")

    monkeypatch.setattr(launcher.os, "close", fail_close)
    failure = launcher._close_fault_channel(123)
    assert failure == {
        "type": "OSError",
        "message": "[Errno 5] fixture fault channel close failure",
        "errno": "5",
    }


def _fault_publish_error(wrapper) -> Any:
    return wrapper.ExclusivePublishError(
        "durability_unknown_quarantined",
        "fixture directory fsync failed",
        stage="final_link_directory_fsync",
        directory_seal={
            "device": 1,
            "inode": 2,
            "uid": os.geteuid(),
            "mode": 0o755,
        },
        payload={"size": 17, "sha256": "e" * 64},
        temporary={
            "device": 1,
            "inode": 3,
            "ctime_ns": 4,
            "size": 17,
            "sha256": "f" * 64,
            "nlink": 2,
        },
        error_number=5,
    )


def test_preflight_fault_channel_frame_round_trip_short_and_eintr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    wrapper = _wrapper_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "wrapper_fault.channel"
    )
    descriptor = launcher._open_presealed_fault_channel(
        tmp_path, binding
    )
    original_pwrite = wrapper.os.pwrite
    calls = 0

    def interrupted_then_short(fd, content, offset):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError("fixture fault EINTR")
        if calls == 2:
            return original_pwrite(fd, content[:13], offset)
        return original_pwrite(fd, content, offset)

    try:
        os.set_inheritable(descriptor, True)
        monkeypatch.setenv(
            wrapper.FAULT_CHANNEL_FD_ENV, str(descriptor)
        )
        receipt = _fault_channel_receipt(binding)
        context = wrapper._bind_inherited_fault_channel(receipt)
        monkeypatch.setattr(
            wrapper.os, "pwrite", interrupted_then_short
        )
        record = wrapper._write_fault_channel_record(
            context, _fault_publish_error(wrapper)
        )
        snapshot = launcher._read_fault_channel(
            descriptor,
            binding,
            attempt_id=receipt["attempt_id"],
            owner_nonce=receipt["controller_owner_nonce"],
            launch_receipt_sha256=receipt[
                "launch_receipt_sha256"
            ],
            publisher=receipt["bindings"]["wrapper"],
        )
        assert calls >= 3
        assert snapshot["state"] == "valid_fault"
        assert snapshot["record"] == record
        assert (
            snapshot["record"]["failure"]["commit_state"]
            == "durability_unknown_quarantined"
        )
    finally:
        os.close(descriptor)


def test_preflight_fault_channel_empty_is_distinct_from_partial(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "wrapper_fault.channel"
    )
    descriptor = launcher._open_presealed_fault_channel(
        tmp_path, binding
    )
    receipt = _fault_channel_receipt(binding)
    try:
        snapshot = launcher._read_fault_channel(
            descriptor,
            binding,
            attempt_id=receipt["attempt_id"],
            owner_nonce=receipt["controller_owner_nonce"],
            launch_receipt_sha256=receipt[
                "launch_receipt_sha256"
            ],
            publisher=receipt["bindings"]["wrapper"],
        )
        assert snapshot["state"] == "empty"
        os.pwrite(descriptor, b"SAFA-PARTIAL", 0)
        os.fsync(descriptor)
        with pytest.raises(RuntimeError, match="magic|partial"):
            launcher._read_fault_channel(
                descriptor,
                binding,
                attempt_id=receipt["attempt_id"],
                owner_nonce=receipt["controller_owner_nonce"],
                launch_receipt_sha256=receipt[
                    "launch_receipt_sha256"
                ],
                publisher=receipt["bindings"]["wrapper"],
            )
    finally:
        os.close(descriptor)


def test_preflight_fault_channel_write_zero_and_fsync_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    wrapper = _wrapper_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "wrapper_fault.channel"
    )
    descriptor = launcher._open_presealed_fault_channel(
        tmp_path, binding
    )
    receipt = _fault_channel_receipt(binding)
    try:
        os.set_inheritable(descriptor, True)
        monkeypatch.setenv(
            wrapper.FAULT_CHANNEL_FD_ENV, str(descriptor)
        )
        context = wrapper._bind_inherited_fault_channel(receipt)
        monkeypatch.setattr(
            wrapper.os, "pwrite", lambda *_args: 0
        )
        with pytest.raises(RuntimeError, match="no progress"):
            wrapper._write_fault_channel_record(
                context, _fault_publish_error(wrapper)
            )
        assert os.fstat(descriptor).st_size == 0
        monkeypatch.undo()
        os.set_inheritable(descriptor, False)
        monkeypatch.setattr(
            wrapper.os,
            "fsync",
            lambda _fd: (_ for _ in ()).throw(
                OSError(5, "fixture fault fsync failure")
            ),
        )
        with pytest.raises(OSError, match="fault fsync"):
            wrapper._write_fault_channel_record(
                context, _fault_publish_error(wrapper)
            )
        assert os.fstat(descriptor).st_size > 0
    finally:
        os.close(descriptor)


def test_preflight_fault_channel_uses_sealed_inode_after_path_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    wrapper = _wrapper_module()
    tmp_path.chmod(0o755)
    path = tmp_path / "wrapper_fault.channel"
    binding = launcher._create_fault_channel(path)
    descriptor = launcher._open_presealed_fault_channel(
        tmp_path, binding
    )
    receipt = _fault_channel_receipt(binding)
    sealed_name = tmp_path / "sealed-detached.channel"
    path.rename(sealed_name)
    path.write_bytes(b"foreign")
    try:
        os.set_inheritable(descriptor, True)
        monkeypatch.setenv(
            wrapper.FAULT_CHANNEL_FD_ENV, str(descriptor)
        )
        context = wrapper._bind_inherited_fault_channel(receipt)
        wrapper._write_fault_channel_record(
            context, _fault_publish_error(wrapper)
        )
        assert path.read_bytes() == b"foreign"
        assert sealed_name.stat().st_ino == binding["inode"]
        assert sealed_name.stat().st_size > 0
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "mutation",
    ("partial", "trailing", "oversize", "hash", "schema", "binding"),
)
def test_preflight_fault_channel_reader_rejects_invalid_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    launcher = _launcher_module()
    wrapper = _wrapper_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "wrapper_fault.channel"
    )
    descriptor = launcher._open_presealed_fault_channel(
        tmp_path, binding
    )
    receipt = _fault_channel_receipt(binding)
    try:
        os.set_inheritable(descriptor, True)
        monkeypatch.setenv(
            wrapper.FAULT_CHANNEL_FD_ENV, str(descriptor)
        )
        context = wrapper._bind_inherited_fault_channel(receipt)
        record = wrapper._write_fault_channel_record(
            context, _fault_publish_error(wrapper)
        )
        frame = os.pread(
            descriptor, os.fstat(descriptor).st_size, 0
        )
        if mutation == "partial":
            mutated = frame[:-1]
        elif mutation == "trailing":
            mutated = frame + b"x"
        elif mutation == "oversize":
            mutated = b"x" * (
                launcher.FAULT_CHANNEL_MAX_RECORD_BYTES + 1
            )
        elif mutation == "hash":
            replacement = (
                b"1" if frame[-2:-1] == b"0" else b"0"
            )
            mutated = frame[:-2] + replacement + b"\n"
            assert mutated != frame
        else:
            changed = dict(record)
            if mutation == "schema":
                changed["schema_version"] = 2
            else:
                changed["attempt_id"] = "0" * 64
            changed["fault_record_sha256"] = (
                wrapper._canonical_digest(
                    changed, "fault_record_sha256"
                )
            )
            mutated = wrapper._fault_channel_frame(
                wrapper._canonical_json(changed)
            )
        os.ftruncate(descriptor, 0)
        os.pwrite(descriptor, mutated, 0)
        os.fsync(descriptor)
        with pytest.raises(RuntimeError):
            launcher._read_fault_channel(
                descriptor,
                binding,
                attempt_id=receipt["attempt_id"],
                owner_nonce=receipt["controller_owner_nonce"],
                launch_receipt_sha256=receipt[
                    "launch_receipt_sha256"
                ],
                publisher=receipt["bindings"]["wrapper"],
            )
    finally:
        os.close(descriptor)


def test_preflight_fault_channel_valid_fault_has_absolute_precedence() -> None:
    launcher = _launcher_module()
    reads = 0

    def forbidden_wrapper_exit():
        nonlocal reads
        reads += 1
        raise AssertionError("typed fault must not read wrapper exit")

    outcome = launcher._evaluate_wrapper_outcome(
        returncode=0,
        fault_snapshot={
            "state": "valid_fault",
            "record": {
                "failure": {
                    "commit_state": (
                        "durability_unknown_quarantined"
                    )
                }
            },
        },
        fault_validation_failure=None,
        fault_close_failure=None,
        wrapper_exit_reader=forbidden_wrapper_exit,
    )
    assert outcome["status"] == "typed_publish_failure"
    assert outcome["exit_code"] == 125
    assert reads == 0


@pytest.mark.parametrize("returncode", (123, 122, 121, 7, -9))
def test_preflight_fault_channel_nonzero_never_reads_wrapper_exit(
    returncode: int,
) -> None:
    launcher = _launcher_module()
    reads = 0

    def forbidden_wrapper_exit():
        nonlocal reads
        reads += 1
        raise AssertionError("failed child must not read wrapper exit")

    outcome = launcher._evaluate_wrapper_outcome(
        returncode=returncode,
        fault_snapshot={"state": "empty"},
        fault_validation_failure=None,
        fault_close_failure=None,
        wrapper_exit_reader=forbidden_wrapper_exit,
    )
    assert outcome["status"] == "wrapper_child_failed"
    assert outcome["exit_code"] != 0
    assert reads == 0


def test_preflight_fault_channel_empty_exact_exit_zero_is_only_success() -> None:
    launcher = _launcher_module()
    success_exit = {
        "value": {
            "exit_code": 0,
            "controller_exit_code": 0,
            "launch_failure": None,
        },
        "binding": {"path": "/exact/wrapper_exit.json"},
    }
    outcome = launcher._evaluate_wrapper_outcome(
        returncode=0,
        fault_snapshot={"state": "empty"},
        fault_validation_failure=None,
        fault_close_failure=None,
        wrapper_exit_reader=lambda: success_exit,
    )
    assert outcome == {
        "status": "success",
        "exit_code": 0,
        "failure": None,
        "wrapper_exit": success_exit["binding"],
    }
    invalid = launcher._evaluate_wrapper_outcome(
        returncode=0,
        fault_snapshot={"state": "empty"},
        fault_validation_failure={"type": "Partial", "message": "partial"},
        fault_close_failure=None,
        wrapper_exit_reader=lambda: success_exit,
    )
    assert invalid["status"] == "invalid_fault_channel"
    assert invalid["exit_code"] == 125
    missing = launcher._evaluate_wrapper_outcome(
        returncode=0,
        fault_snapshot={"state": "empty"},
        fault_validation_failure=None,
        fault_close_failure=None,
        wrapper_exit_reader=lambda: (_ for _ in ()).throw(
            RuntimeError("wrapper exit absent")
        ),
    )
    assert missing["status"] == "wrapper_exit_invalid"
    assert missing["exit_code"] == 125


@pytest.mark.parametrize(
    ("returncode", "status", "exit_code"),
    (
        (0, "success", 0),
        (2, "wrapper_child_failed", 2),
        (124, "wrapper_child_failed", 124),
        (143, "wrapper_child_failed", 143),
        (-signal.SIGTERM, "wrapper_child_failed", 143),
        (-signal.SIGKILL, "wrapper_child_failed", 137),
    ),
)
def test_shared_gate_outcome_maps_exit_and_signal_exactly(
    returncode: int,
    status: str,
    exit_code: int,
) -> None:
    launcher = _launcher_module()
    wrapper_exit = {
        "value": {
            "exit_code": 0,
            "controller_exit_code": 0,
            "launch_failure": None,
        },
        "binding": {"path": "/exact/wrapper_exit.json"},
    }
    outcome = launcher._evaluate_gate_outcome(
        returncode=returncode,
        exec_failure=None,
        fault_snapshot={"state": "empty"},
        fault_validation_failure=None,
        fault_close_failure=None,
        wrapper_exit_reader=lambda: wrapper_exit,
    )
    assert outcome["status"] == status
    assert outcome["exit_code"] == exit_code
    assert outcome["failure"] is None
    assert outcome["wrapper_exit"] == (
        wrapper_exit["binding"] if returncode == 0 else None
    )


@pytest.mark.parametrize(
    ("mode", "value", "wait_code", "wait_status"),
    (
        ("exit", 0, "exited", 0),
        ("exit", 2, "exited", 2),
        ("exit", 117, "exited", 117),
        ("exit", 118, "exited", 118),
        ("signal", signal.SIGTERM, "killed", signal.SIGTERM),
        ("signal", signal.SIGKILL, "killed", signal.SIGKILL),
    ),
)
def test_lifecycle_waitid_preserves_kernel_exit_and_signal(
    mode: str,
    value: int,
    wait_code: str,
    wait_status: int,
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import time; time.sleep(30)"
                if mode == "signal"
                else f"raise SystemExit({value})"
            ),
        ],
        start_new_session=True,
    )
    if mode == "signal":
        os.killpg(child.pid, value)
    info = os.waitid(
        os.P_PID,
        child.pid,
        os.WEXITED | os.WNOWAIT,
    )
    code_names = {
        os.CLD_EXITED: "exited",
        os.CLD_KILLED: "killed",
        os.CLD_DUMPED: "dumped",
    }
    assert code_names[info.si_code] == wait_code
    assert info.si_status == wait_status
    waited_pid, wait_status_raw = os.waitpid(child.pid, 0)
    outcome = launcher.derive_lifecycle_wait_outcome(
        wait_code=wait_code,
        wait_status=int(info.si_status),
    )
    assert waited_pid == child.pid == info.si_pid
    assert (
        os.waitstatus_to_exitcode(wait_status_raw)
        == outcome["returncode"]
    )
    child.returncode = outcome["returncode"]
    assert outcome["core_dumped"] is False
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / f"real_{mode}_{value}.channel"
    )
    template = _lifecycle_wait_record(
        launcher, binding, role="consumer"
    )
    child_process = {
        **template["child_process"],
        "pid": child.pid,
        "pgid": child.pid,
        "sid": child.pid,
    }
    build_arguments = dict(
        role="consumer",
        policy_sha256=template["policy_sha256"],
        attempt_id=template["attempt_id"],
        source_artifact=template["source_artifact"],
        wait_channel=template["wait_channel"],
        publisher=template["publisher"],
        supervisor_owner_seal=template[
            "supervisor_owner_seal"
        ],
        supervisor_process=template["supervisor_process"],
        supervisor_executable=template[
            "supervisor_executable"
        ],
        supervisor_command=template["supervisor_command"],
        worker_started=template["worker_started"],
        child_process=child_process,
        child_executable=template["child_executable"],
        child_command=template["child_command"],
        terminal=(
            None if mode == "signal" else template["terminal"]
        ),
        waitid_si_pid=int(info.si_pid),
        waitid_si_code=int(info.si_code),
        waitid_si_status=int(info.si_status),
        waited_pid=waited_pid,
        wait_status_raw=wait_status_raw,
        started_at=template["started_at"],
        completed_at=template["completed_at"],
    )
    if mode == "exit" and value != 118:
        with pytest.raises(
            launcher.PreflightLaunchContractError,
            match="lifecycle wait schema differs",
        ):
            launcher.build_lifecycle_wait_status(
                **build_arguments
            )
        return
    record = launcher.build_lifecycle_wait_status(
        **build_arguments
    )
    assert record["child_process"]["pid"] == waited_pid
    assert record["wait_status_raw"] == wait_status_raw


def _lifecycle_wait_record(
    launcher,
    binding: Mapping[str, Any],
    *,
    role: str = "gate",
    wait_code: str = "exited",
    wait_status: int | None = None,
    include_terminal: bool = True,
) -> dict[str, Any]:
    process = {
        "pid": 11,
        "ppid": 10,
        "pgid": 11,
        "sid": 11,
        "start_ticks": 100,
    }
    role_kinds = {
        "gate": (
            "launch_receipt",
            "gate_worker_started",
            "gate_execution_terminal",
        ),
        "consumer": (
            "consumer_attempt",
            "consumer_worker_started",
            "consumer_terminal",
        ),
    }

    def sealed_artifact(
        kind: str, path: str, character: str, inode: int
    ) -> dict[str, Any]:
        return {
            "kind": kind,
            "binding": {
                "path": path,
                "sha256": character * 64,
                "canonical_sha256": "f" * 64,
            },
            "file_identity": {
                "path": path,
                "device": 1,
                "inode": inode,
                "mode": 0o100600,
                "size": 3,
            },
        }

    if wait_status is None:
        wait_status = 117 if role == "gate" else 118
    if wait_code == "exited":
        waitid_si_code = os.CLD_EXITED
        wait_status_raw = wait_status << 8
    elif wait_code == "killed":
        waitid_si_code = os.CLD_KILLED
        wait_status_raw = wait_status
    else:
        waitid_si_code = os.CLD_DUMPED
        wait_status_raw = wait_status | 0x80
    source_kind, started_kind, terminal_kind = role_kinds[role]
    return launcher.build_lifecycle_wait_status(
        role=role,
        policy_sha256="a" * 64,
        attempt_id="b" * 64,
        source_artifact=sealed_artifact(
            source_kind, "/exact/source.json", "c", 3
        ),
        wait_channel=dict(binding),
        publisher={
            "path": "/exact/launcher.py",
            "sha256": "e" * 64,
            "file_identity": {
                "path": "/exact/launcher.py",
                "device": 1,
                "inode": 5,
                "mode": 0o100644,
                "size": 3,
            },
            "role": f"{role}_lifecycle_wait_supervisor",
        },
        supervisor_owner_seal={
            "session": "fixture",
            "pane": "%1",
            "pane_pid": 10,
            "pane_dead": False,
            "pane_dead_status": None,
            "pane_process": {
                **process,
                "pid": 10,
                "pgid": 10,
                "sid": 10,
            },
            "owner_nonce": "f" * 64,
            "tmux_server": {
                "server_pid": 20,
                "server_process": {
                    "pid": 20,
                    "ppid": 1,
                    "pgid": 20,
                    "sid": 20,
                    "start_ticks": 50,
                },
                "socket_path": "/tmp/tmux-fixture",
                "socket_device": 1,
                "socket_inode": 2,
            },
        },
        supervisor_process={
            **process,
            "pid": 10,
            "pgid": 10,
            "sid": 10,
        },
        supervisor_executable={
            "path": "/usr/bin/python3",
            "device": 1,
            "inode": 6,
            "mode": 0o100755,
            "size": 3,
        },
        supervisor_command=(
            [
                "/env/bin/python",
                "-B",
                "-u",
                "/exact/launcher.py",
                "__gate_wait_supervisor__",
            ]
            if role == "gate"
            else [
                "/env/bin/python",
                "-B",
                "-u",
                "/exact/launcher.py",
                "__consumer_wait_supervisor__",
            ]
        ),
        worker_started=sealed_artifact(
            started_kind, "/exact/worker_started.json", "1", 7
        ),
        child_process=process,
        child_executable={
            "path": "/usr/bin/python3",
            "device": 1,
            "inode": 2,
            "mode": 0o100755,
            "size": 3,
        },
        child_command=["/usr/bin/python3", "-c", "pass"],
        terminal=(
            sealed_artifact(
                terminal_kind, "/exact/terminal.json", "2", 9
            )
            if include_terminal
            else None
        ),
        waitid_si_pid=process["pid"],
        waitid_si_code=waitid_si_code,
        waitid_si_status=wait_status,
        waited_pid=process["pid"],
        wait_status_raw=wait_status_raw,
        started_at="2026-07-28T00:00:00+00:00",
        completed_at="2026-07-28T00:00:01+00:00",
    )


def _read_lifecycle_test_status(
    launcher: Any,
    descriptor: int,
    directory_descriptor: int,
    binding: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    expected_bindings = {
        key: record[key]
        for key in launcher.LIFECYCLE_WAIT_EXPECTED_BINDING_KEYS
    }
    return launcher._read_lifecycle_wait_status(
        descriptor,
        directory_descriptor,
        binding,
        role=role,
        expected_bindings=expected_bindings,
    )


@pytest.mark.parametrize("exit_code", (0, 2, 118))
def test_gate_lifecycle_rejects_non_adjudicated_controlled_exit(
    tmp_path: Path, exit_code: int
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / f"gate_exit_{exit_code}.channel"
    )
    with pytest.raises(
        launcher.PreflightLaunchContractError,
        match="lifecycle wait schema differs",
    ):
        _lifecycle_wait_record(
            launcher,
            binding,
            role="gate",
            wait_status=exit_code,
        )


def test_gate_lifecycle_requires_terminal_for_adjudicated_exit(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "gate_exit_117.channel"
    )
    assert _lifecycle_wait_record(
        launcher, binding, role="gate", wait_status=117
    )["returncode"] == 117
    with pytest.raises(
        launcher.PreflightLaunchContractError,
        match="lifecycle wait schema differs",
    ):
        _lifecycle_wait_record(
            launcher,
            binding,
            role="gate",
            wait_status=117,
            include_terminal=False,
        )


@pytest.mark.parametrize("exit_code", (0, 2, 117))
def test_consumer_lifecycle_rejects_non_adjudicated_controlled_exit(
    tmp_path: Path, exit_code: int
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / f"consumer_exit_{exit_code}.channel"
    )
    with pytest.raises(
        launcher.PreflightLaunchContractError,
        match="lifecycle wait schema differs",
    ):
        _lifecycle_wait_record(
            launcher,
            binding,
            role="consumer",
            wait_status=exit_code,
        )


def test_consumer_lifecycle_requires_terminal_for_exit_118(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "consumer_exit_118.channel"
    )
    assert _lifecycle_wait_record(
        launcher,
        binding,
        role="consumer",
        wait_status=118,
    )["returncode"] == 118
    with pytest.raises(
        launcher.PreflightLaunchContractError,
        match="lifecycle wait schema differs",
    ):
        _lifecycle_wait_record(
            launcher,
            binding,
            role="consumer",
            wait_status=118,
            include_terminal=False,
        )


@pytest.mark.parametrize("signum", (signal.SIGTERM, signal.SIGKILL))
def test_consumer_lifecycle_signal_can_record_missing_terminal(
    tmp_path: Path, signum: int
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / f"consumer_signal_{signum}.channel"
    )
    record = _lifecycle_wait_record(
        launcher,
        binding,
        role="consumer",
        wait_code="killed",
        wait_status=int(signum),
        include_terminal=False,
    )
    assert record["returncode"] == -signum
    assert record["terminal"] is None


@pytest.mark.parametrize("signum", (signal.SIGTERM, signal.SIGKILL))
def test_gate_lifecycle_signal_can_record_missing_terminal(
    tmp_path: Path, signum: int
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / f"gate_signal_{signum}.channel"
    )
    record = _lifecycle_wait_record(
        launcher,
        binding,
        role="gate",
        wait_code="killed",
        wait_status=int(signum),
        include_terminal=False,
    )
    assert record["returncode"] == -signum
    assert record["terminal"] is None


@pytest.mark.parametrize(
    ("mode", "value"),
    (
        ("exit", 0),
        ("exit", 2),
        ("exit", 117),
        ("signal", signal.SIGTERM),
        ("signal", signal.SIGKILL),
    ),
)
def test_gate_wait_capture_uses_waitid_then_same_waitpid(
    mode: str, value: int
) -> None:
    launcher = _launcher_module()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import time; time.sleep(30)"
                if mode == "signal"
                else f"raise SystemExit({value})"
            ),
        ],
        start_new_session=True,
    )
    if mode == "signal":
        os.killpg(child.pid, value)
    info, waited_pid, raw_status = launcher._waitid_then_waitpid(
        child
    )
    assert info.si_pid == waited_pid == child.pid
    if mode == "exit":
        assert info.si_code == os.CLD_EXITED
        assert info.si_status == value
        assert raw_status == value << 8
        assert child.returncode == value
    else:
        assert info.si_code in {os.CLD_KILLED, os.CLD_DUMPED}
        assert info.si_status == value
        assert os.WTERMSIG(raw_status) == value
        assert child.returncode == -value


def test_gate_worker_does_not_inherit_lifecycle_writer(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "gate_writer.channel"
    )
    descriptor, directory_descriptor = (
        launcher._open_presealed_lifecycle_wait_channel(
            tmp_path, binding, name="gate_writer.channel"
        )
    )
    environment = dict(os.environ)
    environment["EXPECTED_CHANNEL_DEVICE"] = str(binding["device"])
    environment["EXPECTED_CHANNEL_INODE"] = str(binding["inode"])
    script = (
        "import os;"
        "d=int(os.environ['EXPECTED_CHANNEL_DEVICE']);"
        "i=int(os.environ['EXPECTED_CHANNEL_INODE']);"
        "found=False;"
        "\nfor name in os.listdir('/proc/self/fd'):\n"
        "  try:\n"
        "    s=os.fstat(int(name)); found=found or "
        "(s.st_dev==d and s.st_ino==i)\n"
        "  except OSError:\n"
        "    pass\n"
        "raise SystemExit(99 if found else 117)"
    )
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", script],
            close_fds=True,
            preexec_fn=launcher._wrapper_child_setup,
            env=environment,
        )
        info, waited_pid, raw_status = (
            launcher._waitid_then_waitpid(child)
        )
        assert info.si_pid == waited_pid == child.pid
        assert raw_status == 117 << 8
        assert child.returncode == 117
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def test_consumer_worker_does_not_inherit_lifecycle_writer(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "consumer_writer.channel"
    )
    descriptor, directory_descriptor = (
        launcher._open_presealed_lifecycle_wait_channel(
            tmp_path,
            binding,
            name="consumer_writer.channel",
        )
    )
    environment = dict(os.environ)
    environment["EXPECTED_CONSUMER_CHANNEL_DEVICE"] = str(
        binding["device"]
    )
    environment["EXPECTED_CONSUMER_CHANNEL_INODE"] = str(
        binding["inode"]
    )
    script = (
        "import os;"
        "d=int(os.environ['EXPECTED_CONSUMER_CHANNEL_DEVICE']);"
        "i=int(os.environ['EXPECTED_CONSUMER_CHANNEL_INODE']);"
        "found=False;"
        "\nfor name in os.listdir('/proc/self/fd'):\n"
        "  try:\n"
        "    s=os.fstat(int(name)); found=found or "
        "(s.st_dev==d and s.st_ino==i)\n"
        "  except OSError:\n"
        "    pass\n"
        "raise SystemExit(99 if found else 118)"
    )
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", script],
            close_fds=True,
            preexec_fn=launcher._wrapper_child_setup,
            env=environment,
        )
        info, waited_pid, raw_status = (
            launcher._waitid_then_waitpid(child)
        )
        assert info.si_pid == waited_pid == child.pid
        assert raw_status == 118 << 8
        assert child.returncode == 118
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def test_gate_worker_pdeathsig_kills_child_with_supervisor(
    tmp_path: Path,
) -> None:
    launcher_path = Path(
        _launcher_module().__file__
    ).resolve()
    middle_script = (
        "import importlib.util,subprocess,sys,time;"
        f"p={str(launcher_path)!r};"
        "s=importlib.util.spec_from_file_location('gate_supervisor_test',p);"
        "m=importlib.util.module_from_spec(s);"
        "s.loader.exec_module(m);"
        "c=subprocess.Popen([sys.executable,'-c',"
        "'import time; time.sleep(30)'],"
        "preexec_fn=m._wrapper_child_setup,close_fds=True);"
        "print(c.pid,flush=True);time.sleep(30)"
    )
    supervisor = subprocess.Popen(
        [sys.executable, "-c", middle_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert supervisor.stdout is not None
    raw_pid = supervisor.stdout.readline().strip()
    assert raw_pid.isdigit()
    child_pid = int(raw_pid)
    os.kill(supervisor.pid, signal.SIGKILL)
    supervisor.wait(timeout=2.0)
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{child_pid}").exists():
        if time.monotonic() >= deadline:
            os.kill(child_pid, signal.SIGKILL)
            pytest.fail("gate worker survived supervisor SIGKILL")
        time.sleep(0.01)


def test_lifecycle_wait_channel_is_write_once_and_exact(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    path = tmp_path / "gate_wait_status.channel"
    binding = launcher._create_fault_channel(path)
    descriptor, directory_descriptor = (
        launcher._open_presealed_lifecycle_wait_channel(
        tmp_path, binding, name=path.name
        )
    )
    try:
        record = _lifecycle_wait_record(launcher, binding)
        assert (
            launcher._write_lifecycle_wait_status(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
            == record
        )
        snapshot = _read_lifecycle_test_status(
            launcher,
            descriptor,
            directory_descriptor,
            binding,
            record,
            role="gate",
        )
        assert snapshot["state"] == "valid_wait_status"
        assert snapshot["record"] == record
        assert snapshot["channel_authority"] == {
            "path": binding["path"],
            "device": binding["device"],
            "inode": binding["inode"],
            "mode": binding["mode"],
            "uid": binding["uid"],
            "nlink": 1,
            "size": snapshot["size"],
            "directory_device": binding["directory_device"],
            "directory_inode": binding["directory_inode"],
        }
        with pytest.raises(
            RuntimeError, match="named identity differs"
        ):
            launcher._write_lifecycle_wait_status(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def test_gate_lifecycle_wait_registration_is_presealed_and_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    path = tmp_path / "gate_lifecycle_wait.channel"
    fsync_kinds: list[str] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        fsync_kinds.append(
            "directory" if stat.S_ISDIR(mode) else "file"
        )
        real_fsync(descriptor)

    monkeypatch.setattr(launcher.os, "fsync", recording_fsync)
    binding = launcher._create_fault_channel(path)
    observed = path.stat()
    directory = tmp_path.stat()
    assert binding == {
        "path": str(path),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": observed.st_mode,
        "uid": observed.st_uid,
        "nlink": 1,
        "size": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "directory_device": directory.st_dev,
        "directory_inode": directory.st_ino,
    }
    assert stat.S_IMODE(observed.st_mode) == 0o600
    assert fsync_kinds == ["file", "directory"]
    descriptor, directory_descriptor = (
        launcher._open_presealed_lifecycle_wait_channel(
            tmp_path, binding, name=path.name
        )
    )
    try:
        assert fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_DSYNC
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)
    with pytest.raises(FileExistsError):
        launcher._create_fault_channel(path)
    assert path.stat().st_ino == binding["inode"]
    assert path.read_bytes() == b""
    drifted_binding = dict(binding)
    drifted_binding["inode"] += 1
    with pytest.raises(
        RuntimeError,
        match="presealed lifecycle wait channel identity differs",
    ):
        launcher._open_presealed_lifecycle_wait_channel(
            tmp_path, drifted_binding, name=path.name
        )
    foreign = tmp_path / "foreign.channel"
    foreign.write_bytes(b"foreign")
    foreign.chmod(0o600)
    foreign_identity = foreign.stat()
    with pytest.raises(FileExistsError):
        launcher._create_fault_channel(foreign)
    assert foreign.read_bytes() == b"foreign"
    assert foreign.stat().st_ino == foreign_identity.st_ino


def test_lifecycle_wait_channel_rejects_partial_and_path_replace(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    path = tmp_path / "consumer_wait_status.channel"
    binding = launcher._create_fault_channel(path)
    descriptor, directory_descriptor = (
        launcher._open_presealed_lifecycle_wait_channel(
        tmp_path, binding, name=path.name
        )
    )
    try:
        record = _lifecycle_wait_record(
            launcher, binding, role="consumer"
        )
        os.pwrite(
            descriptor,
            launcher.LIFECYCLE_WAIT_CHANNEL_PREFIX + b"00000010\n{",
            0,
        )
        os.fsync(descriptor)
        with pytest.raises(RuntimeError, match="partial"):
            _read_lifecycle_test_status(
                launcher,
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="consumer",
            )
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)
    moved = tmp_path / "moved.channel"
    path.rename(moved)
    path.write_bytes(b"")
    path.chmod(0o600)
    with pytest.raises(
        RuntimeError,
        match="presealed lifecycle wait channel identity differs",
    ):
        channel = launcher._open_presealed_lifecycle_wait_channel(
            tmp_path, binding, name=path.name
        )
        os.close(channel[0])
        os.close(channel[1])


def test_lifecycle_wait_channel_short_write_and_eintr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    path = tmp_path / "wait_status.channel"
    binding = launcher._create_fault_channel(path)
    descriptor, directory_descriptor = (
        launcher._open_presealed_lifecycle_wait_channel(
        tmp_path, binding, name=path.name
        )
    )
    original_pwrite = launcher.os.pwrite
    calls = 0

    def interrupted_short_write(fd: int, data: bytes, offset: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError
        return original_pwrite(fd, data[:7], offset)

    monkeypatch.setattr(
        launcher.os, "pwrite", interrupted_short_write
    )
    try:
        record = _lifecycle_wait_record(launcher, binding)
        launcher._write_lifecycle_wait_status(
            descriptor,
            directory_descriptor,
            binding,
            record,
            role="gate",
        )
        assert calls > 2
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def _open_lifecycle_test_writer(
    launcher: Any, tmp_path: Path, name: str
) -> tuple[Path, dict[str, Any], int, int]:
    tmp_path.chmod(0o755)
    path = tmp_path / name
    binding = launcher._create_fault_channel(path)
    descriptor, directory_descriptor = (
        launcher._open_presealed_lifecycle_wait_channel(
        tmp_path, binding, name=name
        )
    )
    return path, binding, descriptor, directory_descriptor


def test_lifecycle_wait_channel_syscall_order_and_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
        launcher, tmp_path, "ordered_wait.channel"
        )
    )
    assert (
        launcher.fcntl.fcntl(descriptor, launcher.fcntl.F_GETFL)
        & os.O_DSYNC
    )
    original_pwrite = launcher.os.pwrite
    original_fsync = launcher.os.fsync
    events: list[str] = []

    def ordered_pwrite(fd: int, data: bytes, offset: int) -> int:
        events.append(
            "commit"
            if data.startswith(
                launcher.LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
            )
            else "body"
        )
        return original_pwrite(fd, data, offset)

    def ordered_fsync(fd: int) -> None:
        events.append("fsync")
        original_fsync(fd)

    monkeypatch.setattr(launcher.os, "pwrite", ordered_pwrite)
    monkeypatch.setattr(launcher.os, "fsync", ordered_fsync)
    record = _lifecycle_wait_record(launcher, binding)
    launcher._write_lifecycle_wait_status(
        descriptor,
        directory_descriptor,
        binding,
        record,
        role="gate",
    )
    assert events == ["body", "fsync", "commit"]
    os.close(descriptor)
    os.close(directory_descriptor)
    reader, reader_directory = (
        launcher._open_lifecycle_wait_channel_reader(
            tmp_path, binding, name=path.name
        )
    )
    try:
        assert (
            fcntl.fcntl(reader, fcntl.F_GETFL) & os.O_ACCMODE
        ) == os.O_RDONLY
        snapshot = _read_lifecycle_test_status(
            launcher,
            reader,
            reader_directory,
            binding,
            record,
            role="gate",
        )
        assert snapshot["record"] == record
    finally:
        os.close(reader)
        os.close(reader_directory)


def test_lifecycle_wait_channel_requires_commit_marker(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
        launcher, tmp_path, "uncommitted_wait.channel"
        )
    )
    record = _lifecycle_wait_record(launcher, binding)
    body, _commit, _frame = (
        launcher._build_lifecycle_wait_channel_frame(record)
    )
    os.pwrite(descriptor, body, 0)
    os.fsync(descriptor)
    os.close(descriptor)
    os.close(directory_descriptor)
    reader, reader_directory = (
        launcher._open_lifecycle_wait_channel_reader(
            tmp_path, binding, name=path.name
        )
    )
    try:
        with pytest.raises(RuntimeError, match="uncommitted"):
            _read_lifecycle_test_status(
                launcher,
                reader,
                reader_directory,
                binding,
                record,
                role="gate",
            )
    finally:
        os.close(reader)
        os.close(reader_directory)


def test_lifecycle_wait_channel_fsync_failure_cannot_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
        launcher, tmp_path, "fsync_failure_wait.channel"
        )
    )
    record = _lifecycle_wait_record(launcher, binding)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError(5, "injected fsync failure")

    monkeypatch.setattr(launcher.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        launcher._write_lifecycle_wait_status(
            descriptor,
            directory_descriptor,
            binding,
            record,
            role="gate",
        )
    content = os.pread(
        descriptor,
        launcher.LIFECYCLE_WAIT_CHANNEL_MAX_RECORD_BYTES,
        0,
    )
    assert (
        launcher.LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
        not in content
    )
    os.close(descriptor)
    os.close(directory_descriptor)
    reader, reader_directory = (
        launcher._open_lifecycle_wait_channel_reader(
            tmp_path, binding, name=path.name
        )
    )
    try:
        with pytest.raises(RuntimeError, match="uncommitted"):
            _read_lifecycle_test_status(
                launcher,
                reader,
                reader_directory,
                binding,
                record,
                role="gate",
            )
    finally:
        os.close(reader)
        os.close(reader_directory)


@pytest.mark.parametrize("zero_stage", ("body", "commit"))
def test_lifecycle_wait_channel_zero_write_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zero_stage: str,
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, f"zero_{zero_stage}_wait.channel"
        )
    )
    original_pwrite = launcher.os.pwrite

    def zero_write(fd: int, data: bytes, offset: int) -> int:
        stage = (
            "commit"
            if data.startswith(
                launcher.LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
            )
            else "body"
        )
        if stage == zero_stage:
            return 0
        return original_pwrite(fd, data, offset)

    monkeypatch.setattr(launcher.os, "pwrite", zero_write)
    record = _lifecycle_wait_record(launcher, binding)
    with pytest.raises(RuntimeError, match="made no progress"):
        launcher._write_lifecycle_wait_status(
            descriptor,
            directory_descriptor,
            binding,
            record,
            role="gate",
        )
    os.close(descriptor)
    os.close(directory_descriptor)
    reader, reader_directory = (
        launcher._open_lifecycle_wait_channel_reader(
            tmp_path, binding, name=path.name
        )
    )
    try:
        if zero_stage == "body":
            empty = _read_lifecycle_test_status(
                launcher,
                reader,
                reader_directory,
                binding,
                record,
                role="gate",
            )
            assert empty["state"] == "empty"
            assert empty["channel_authority"]["inode"] == (
                binding["inode"]
            )
        else:
            with pytest.raises(RuntimeError, match="uncommitted"):
                _read_lifecycle_test_status(
                    launcher,
                    reader,
                    reader_directory,
                    binding,
                    record,
                    role="gate",
                )
    finally:
        os.close(reader)
        os.close(reader_directory)


def test_lifecycle_wait_channel_binding_mismatch_is_prewrite(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    _path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, "bound_wait.channel"
        )
    )
    other_path = tmp_path / "other_wait.channel"
    other_binding = launcher._create_fault_channel(other_path)
    record = _lifecycle_wait_record(launcher, other_binding)
    with pytest.raises(
        RuntimeError, match="binding differs before write"
    ):
        launcher._write_lifecycle_wait_status(
            descriptor,
            directory_descriptor,
            binding,
            record,
            role="gate",
        )
    assert os.fstat(descriptor).st_size == 0
    assert other_path.read_bytes() == b""
    os.close(descriptor)
    os.close(directory_descriptor)


def test_lifecycle_wait_channel_second_writer_is_excluded(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, "exclusive_wait.channel"
        )
    )
    try:
        with pytest.raises(BlockingIOError):
            launcher._open_presealed_lifecycle_wait_channel(
                tmp_path, binding, name=path.name
            )
        assert path.read_bytes() == b""
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


@pytest.mark.parametrize(
    "mutation", ("replace", "unlink", "hardlink", "symlink")
)
def test_lifecycle_wait_channel_rejects_live_name_mutation(
    tmp_path: Path, mutation: str
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, f"{mutation}_wait.channel"
        )
    )
    bound_backup = tmp_path / f"{mutation}_bound.channel"
    foreign = tmp_path / f"{mutation}_foreign"
    foreign.write_bytes(b"FOREIGN")
    foreign.chmod(0o600)
    foreign_before = (
        foreign.stat().st_ino,
        foreign.stat().st_mode,
        foreign.read_bytes(),
    )
    if mutation == "replace":
        path.rename(bound_backup)
        path.write_bytes(b"REPLACEMENT")
        path.chmod(0o600)
    elif mutation == "unlink":
        path.unlink()
    elif mutation == "hardlink":
        os.link(path, bound_backup)
    else:
        path.rename(bound_backup)
        path.symlink_to(foreign)
    record = _lifecycle_wait_record(launcher, binding)
    try:
        with pytest.raises(
            (RuntimeError, FileNotFoundError),
            match="named identity differs|No such file",
        ):
            launcher._write_lifecycle_wait_status(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
        assert os.fstat(descriptor).st_size == 0
        assert (
            foreign.stat().st_ino,
            foreign.stat().st_mode,
            foreign.read_bytes(),
        ) == foreign_before
        if mutation == "replace":
            assert path.read_bytes() == b"REPLACEMENT"
        elif mutation == "symlink":
            assert path.is_symlink()
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def test_lifecycle_wait_channel_rejects_parent_swap(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    attempt = tmp_path / "attempt"
    attempt.mkdir(mode=0o755)
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, attempt, "parent_wait.channel"
        )
    )
    moved = tmp_path / "moved_attempt"
    attempt.rename(moved)
    attempt.mkdir(mode=0o755)
    replacement = attempt / path.name
    replacement.write_bytes(b"FOREIGN")
    replacement.chmod(0o600)
    replacement_before = (
        replacement.stat().st_ino,
        replacement.read_bytes(),
    )
    record = _lifecycle_wait_record(launcher, binding)
    try:
        with pytest.raises(
            RuntimeError, match="named identity differs"
        ):
            launcher._write_lifecycle_wait_status(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
        assert os.fstat(descriptor).st_size == 0
        assert (
            replacement.stat().st_ino,
            replacement.read_bytes(),
        ) == replacement_before
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def test_lifecycle_wait_channel_rejects_name_swap_after_body_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, "post_fsync_swap.channel"
        )
    )
    record = _lifecycle_wait_record(launcher, binding)
    moved = tmp_path / "post_fsync_bound.channel"
    original_fsync = launcher.os.fsync

    def fsync_then_swap(fd: int) -> None:
        original_fsync(fd)
        path.rename(moved)
        path.write_bytes(b"FOREIGN")
        path.chmod(0o600)

    monkeypatch.setattr(launcher.os, "fsync", fsync_then_swap)
    try:
        with pytest.raises(
            RuntimeError, match="named identity differs"
        ):
            launcher._write_lifecycle_wait_status(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
        assert (
            launcher.LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
            not in moved.read_bytes()
        )
        assert path.read_bytes() == b"FOREIGN"
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def test_lifecycle_wait_writer_rejects_swap_after_final_pread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, "post_pread_swap.channel"
        )
    )
    record = _lifecycle_wait_record(launcher, binding)
    moved = tmp_path / "post_pread_bound.channel"
    original_pread = launcher.os.pread
    calls = 0

    def pread_then_swap(fd: int, size: int, offset: int) -> bytes:
        nonlocal calls
        content = original_pread(fd, size, offset)
        calls += 1
        if calls == 2:
            path.rename(moved)
            path.write_bytes(b"FOREIGN")
            path.chmod(0o600)
        return content

    monkeypatch.setattr(launcher.os, "pread", pread_then_swap)
    try:
        with pytest.raises(
            RuntimeError, match="named identity differs"
        ):
            launcher._write_lifecycle_wait_status(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
        assert (
            launcher.LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
            in moved.read_bytes()
        )
        assert path.read_bytes() == b"FOREIGN"
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


@pytest.mark.parametrize(
    "cutpoint",
    (
        "before_write",
        "mid_body",
        "before_body_fsync",
        "after_body_fsync",
        "mid_commit",
        "after_commit",
    ),
)
def test_lifecycle_wait_channel_crash_cutpoints(
    tmp_path: Path, cutpoint: str
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, f"crash_{cutpoint}.channel"
        )
    )
    record = _lifecycle_wait_record(launcher, binding)
    child_pid = os.fork()
    if child_pid == 0:
        original_pwrite = launcher.os.pwrite
        original_fsync = launcher.os.fsync

        def crash() -> None:
            os.kill(os.getpid(), signal.SIGKILL)
            os._exit(255)

        def crash_pwrite(
            fd: int, data: bytes, offset: int
        ) -> int:
            is_commit = data.startswith(
                launcher.LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
            )
            if cutpoint == "mid_body" and not is_commit:
                original_pwrite(fd, data[:7], offset)
                crash()
            if cutpoint == "after_body_fsync" and is_commit:
                crash()
            if cutpoint == "mid_commit" and is_commit:
                original_pwrite(fd, data[:7], offset)
                crash()
            if cutpoint == "after_commit" and is_commit:
                original_pwrite(fd, data, offset)
                crash()
            return original_pwrite(fd, data, offset)

        def crash_fsync(fd: int) -> None:
            if cutpoint == "before_body_fsync":
                crash()
            original_fsync(fd)

        launcher.os.pwrite = crash_pwrite
        launcher.os.fsync = crash_fsync
        if cutpoint == "before_write":
            crash()
        launcher._write_lifecycle_wait_status(
            descriptor,
            directory_descriptor,
            binding,
            record,
            role="gate",
        )
        os._exit(0)
    os.close(descriptor)
    os.close(directory_descriptor)
    waited_pid, waited_status = os.waitpid(child_pid, 0)
    assert waited_pid == child_pid
    assert os.WIFSIGNALED(waited_status)
    assert os.WTERMSIG(waited_status) == signal.SIGKILL
    reader, reader_directory = (
        launcher._open_lifecycle_wait_channel_reader(
            tmp_path, binding, name=path.name
        )
    )
    try:
        if cutpoint == "after_commit":
            snapshot = _read_lifecycle_test_status(
                launcher,
                reader,
                reader_directory,
                binding,
                record,
                role="gate",
            )
            assert snapshot["record"] == record
        elif cutpoint == "before_write":
            assert _read_lifecycle_test_status(
                launcher,
                reader,
                reader_directory,
                binding,
                record,
                role="gate",
            )["state"] == "empty"
        else:
            with pytest.raises(
                RuntimeError,
                match="uncommitted|partial|magic or bound",
            ):
                _read_lifecycle_test_status(
                    launcher,
                    reader,
                    reader_directory,
                    binding,
                    record,
                    role="gate",
                )
    finally:
        os.close(reader)
        os.close(reader_directory)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra_key",
        "missing_key",
        "role",
        "source",
        "channel_uid",
        "publisher",
        "owner_nested",
        "owner_session",
        "owner_pane",
        "child_ppid",
        "child_pgid",
        "child_sid",
        "wait_tuple",
        "waitid_pid",
        "waited_pid",
        "raw_wait",
        "raw_stopped",
        "raw_continued",
        "raw_exit_core_byte",
        "raw_exit_high_core",
        "dumped_without_core",
        "killed_with_core",
        "signal_255",
        "returncode_bool",
        "exit_code_bool",
        "signal_number_bool",
        "core_dumped_int",
        "source_kind",
        "publisher_identity",
        "publisher_command",
        "supervisor_command",
        "terminal_kind",
        "success_without_terminal",
        "naive_time",
        "mixed_time",
        "reversed_time",
    ),
)
def test_lifecycle_wait_status_contract_rejects_mutations(
    tmp_path: Path, mutation: str
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / f"{mutation}.channel"
    )
    record = copy.deepcopy(
        _lifecycle_wait_record(launcher, binding)
    )
    if mutation == "extra_key":
        record["extra"] = True
    elif mutation == "missing_key":
        del record["completed_at"]
    elif mutation == "role":
        record["role"] = "consumer"
    elif mutation == "source":
        record["source_artifact"]["binding"][
            "path"
        ] = "relative/source.json"
    elif mutation == "channel_uid":
        record["wait_channel"]["uid"] = -1
    elif mutation == "publisher":
        record["publisher"]["role"] = "foreign"
    elif mutation == "owner_nested":
        del record["supervisor_owner_seal"]["tmux_server"][
            "socket_inode"
        ]
    elif mutation == "owner_session":
        record["supervisor_owner_seal"]["session"] = ""
    elif mutation == "owner_pane":
        record["supervisor_owner_seal"]["pane"] = ""
    elif mutation == "child_ppid":
        record["child_process"]["ppid"] = 99
    elif mutation == "child_pgid":
        record["child_process"]["pgid"] = 99
    elif mutation == "child_sid":
        record["child_process"]["sid"] = 99
    elif mutation == "wait_tuple":
        record["waitid_si_status"] = 2
    elif mutation == "waitid_pid":
        record["waitid_si_pid"] = 99
    elif mutation == "waited_pid":
        record["waited_pid"] = 99
    elif mutation == "raw_wait":
        record["wait_status_raw"] = 2 << 8
    elif mutation == "raw_stopped":
        record["wait_status_raw"] = 0x7F
    elif mutation == "raw_continued":
        record["wait_status_raw"] = 0xFFFF
    elif mutation == "raw_exit_core_byte":
        record["wait_status_raw"] = 0x80
    elif mutation == "raw_exit_high_core":
        record.update(
            {
                "waitid_si_status": 255,
                "wait_status_raw": 0xFF80,
                "returncode": 255,
                "exit_code": 255,
            }
        )
    elif mutation == "dumped_without_core":
        record.update(
            {
                "waitid_si_code": os.CLD_DUMPED,
                "waitid_si_status": int(signal.SIGABRT),
                "wait_status_raw": int(signal.SIGABRT),
                "wait_code": "dumped",
                "returncode": -int(signal.SIGABRT),
                "exit_kind": "signal",
                "exit_code": None,
                "signal_number": int(signal.SIGABRT),
                "core_dumped": True,
            }
        )
    elif mutation == "killed_with_core":
        record.update(
            {
                "waitid_si_code": os.CLD_KILLED,
                "waitid_si_status": int(signal.SIGABRT),
                "wait_status_raw": int(signal.SIGABRT) | 0x80,
                "wait_code": "killed",
                "returncode": -int(signal.SIGABRT),
                "exit_kind": "signal",
                "exit_code": None,
                "signal_number": int(signal.SIGABRT),
                "core_dumped": False,
            }
        )
    elif mutation == "signal_255":
        record.update(
            {
                "waitid_si_code": os.CLD_KILLED,
                "waitid_si_status": 255,
                "wait_status_raw": 255,
                "wait_code": "killed",
                "returncode": -255,
                "exit_kind": "signal",
                "exit_code": None,
                "signal_number": 255,
                "core_dumped": False,
            }
        )
    elif mutation in {"returncode_bool", "exit_code_bool"}:
        record.update(
            {
                "waitid_si_status": 1,
                "wait_status_raw": 1 << 8,
                "returncode": (
                    True if mutation == "returncode_bool" else 1
                ),
                "exit_code": (
                    True if mutation == "exit_code_bool" else 1
                ),
            }
        )
    elif mutation == "signal_number_bool":
        record.update(
            {
                "waitid_si_code": os.CLD_KILLED,
                "waitid_si_status": 1,
                "wait_status_raw": 1,
                "wait_code": "killed",
                "returncode": -1,
                "exit_kind": "signal",
                "exit_code": None,
                "signal_number": True,
                "core_dumped": False,
            }
        )
    elif mutation == "core_dumped_int":
        record["core_dumped"] = 0
    elif mutation == "source_kind":
        record["source_artifact"]["kind"] = "consumer_attempt"
    elif mutation == "publisher_identity":
        record["publisher"]["file_identity"][
            "path"
        ] = "/other/launcher.py"
    elif mutation == "publisher_command":
        record["publisher"]["path"] = "/other/launcher.py"
        record["publisher"]["file_identity"][
            "path"
        ] = "/other/launcher.py"
    elif mutation == "supervisor_command":
        record["supervisor_command"] = []
    elif mutation == "terminal_kind":
        record["terminal"]["kind"] = "consumer_terminal"
    elif mutation == "success_without_terminal":
        record["terminal"] = None
    elif mutation == "naive_time":
        record["started_at"] = "2026-07-28T00:00:00"
    elif mutation == "mixed_time":
        record["completed_at"] = "2026-07-28T00:00:01"
    else:
        record["completed_at"] = "2026-07-27T23:59:59+00:00"
    record["lifecycle_wait_status_sha256"] = (
        launcher._canonical_digest(
            record, "lifecycle_wait_status_sha256"
        )
    )
    with pytest.raises(launcher.PreflightLaunchContractError):
        launcher.validate_lifecycle_wait_status(
            record, role="gate", label="mutated wait status"
        )


def test_lifecycle_wait_status_allows_raw_argv_symlink_name(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "argv.channel"
    )
    record = _lifecycle_wait_record(launcher, binding)
    record["child_command"][0] = "/env/bin/python"
    record["lifecycle_wait_status_sha256"] = (
        launcher._canonical_digest(
            record, "lifecycle_wait_status_sha256"
        )
    )
    assert launcher.validate_lifecycle_wait_status(
        record, role="gate", label="symlink argv wait status"
    ) == record


def test_lifecycle_wait_status_allows_root_channel_uid(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "root_uid.channel"
    )
    record = _lifecycle_wait_record(launcher, binding)
    record["wait_channel"]["uid"] = 0
    record["lifecycle_wait_status_sha256"] = (
        launcher._canonical_digest(
            record, "lifecycle_wait_status_sha256"
        )
    )
    assert launcher.validate_lifecycle_wait_status(
        record, role="gate", label="root uid wait status"
    ) == record


def test_lifecycle_wait_status_allows_missing_crash_terminal(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "missing_terminal.channel"
    )
    record = _lifecycle_wait_record(
        launcher,
        binding,
        wait_code="killed",
        wait_status=int(signal.SIGKILL),
    )
    record["terminal"] = None
    record["lifecycle_wait_status_sha256"] = (
        launcher._canonical_digest(
            record, "lifecycle_wait_status_sha256"
        )
    )
    assert launcher.validate_lifecycle_wait_status(
        record, role="gate", label="crash wait status"
    ) == record


@pytest.mark.parametrize(
    "semantic",
    (
        "policy",
        "attempt",
        "source",
        "publisher",
        "owner",
        "supervisor",
        "child",
        "executable",
        "command",
    ),
)
def test_lifecycle_wait_reader_rejects_semantic_binding_drift(
    tmp_path: Path, semantic: str
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, f"semantic_{semantic}.channel"
        )
    )
    expected_record = _lifecycle_wait_record(launcher, binding)
    record = copy.deepcopy(expected_record)
    if semantic == "policy":
        record["policy_sha256"] = "0" * 64
    elif semantic == "attempt":
        record["attempt_id"] = "0" * 64
    elif semantic == "source":
        record["source_artifact"]["binding"]["sha256"] = "0" * 64
    elif semantic == "publisher":
        record["publisher"]["sha256"] = "0" * 64
    elif semantic == "owner":
        record["supervisor_owner_seal"]["owner_nonce"] = "0" * 64
    elif semantic == "supervisor":
        record["supervisor_process"]["start_ticks"] = 101
        record["supervisor_owner_seal"]["pane_process"][
            "start_ticks"
        ] = 101
    elif semantic == "child":
        record["child_process"]["start_ticks"] = 101
    elif semantic == "executable":
        record["child_executable"]["path"] = "/other/python3"
    else:
        record["child_command"][-1] = (
            "raise SystemExit(0)"
            if semantic == "command"
            else record["child_command"][-1]
        )
    record["lifecycle_wait_status_sha256"] = (
        launcher._canonical_digest(
            record, "lifecycle_wait_status_sha256"
        )
    )
    launcher._write_lifecycle_wait_status(
        descriptor,
        directory_descriptor,
        binding,
        record,
        role="gate",
    )
    os.close(descriptor)
    os.close(directory_descriptor)
    reader, reader_directory = (
        launcher._open_lifecycle_wait_channel_reader(
            tmp_path, binding, name=path.name
        )
    )
    try:
        with pytest.raises(
            RuntimeError, match="semantic bindings differ"
        ):
            launcher._read_lifecycle_wait_status(
                reader,
                reader_directory,
                binding,
                role="gate",
                expected_bindings={
                    key: expected_record[key]
                    for key in (
                        launcher.LIFECYCLE_WAIT_EXPECTED_BINDING_KEYS
                    )
                },
            )
    finally:
        os.close(reader)
        os.close(reader_directory)


@pytest.mark.parametrize(
    "frame_mutation",
    (
        "partial_commit",
        "bad_commit",
        "trailing",
        "noncanonical",
        "oversized",
    ),
)
def test_lifecycle_wait_channel_rejects_frame_mutations(
    tmp_path: Path, frame_mutation: str
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, f"frame_{frame_mutation}.channel"
        )
    )
    record = _lifecycle_wait_record(launcher, binding)
    body, commit, frame = (
        launcher._build_lifecycle_wait_channel_frame(record)
    )
    if frame_mutation == "partial_commit":
        content = body + commit[:7]
    elif frame_mutation == "bad_commit":
        content = (
            body
            + launcher.LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
            + b"0" * 64
            + b"\n"
        )
    elif frame_mutation == "trailing":
        content = frame + b"x"
    elif frame_mutation == "noncanonical":
        payload = (
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        noncanonical_body = (
            launcher.LIFECYCLE_WAIT_CHANNEL_PREFIX
            + f"{len(payload):08x}\n".encode("ascii")
            + payload
            + launcher.FAULT_CHANNEL_SHA_PREFIX
            + hashlib.sha256(payload).hexdigest().encode("ascii")
            + b"\n"
        )
        content = (
            noncanonical_body
            + launcher.LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
            + hashlib.sha256(noncanonical_body).hexdigest().encode(
                "ascii"
            )
            + b"\n"
        )
    else:
        content = (
            b"x"
            * (
                launcher.LIFECYCLE_WAIT_CHANNEL_MAX_RECORD_BYTES
                + 1
            )
        )
    launcher._pwrite_all(
        descriptor, content, 0, label="test raw frame"
    )
    os.fsync(descriptor)
    os.close(descriptor)
    os.close(directory_descriptor)
    if frame_mutation == "oversized":
        with pytest.raises(
            RuntimeError, match="reader identity differs"
        ):
            channel = launcher._open_lifecycle_wait_channel_reader(
                tmp_path, binding, name=path.name
            )
            os.close(channel[0])
            os.close(channel[1])
        return
    reader, reader_directory = (
        launcher._open_lifecycle_wait_channel_reader(
            tmp_path, binding, name=path.name
        )
    )
    try:
        with pytest.raises(
            RuntimeError,
            match=(
                "uncommitted|partial|trailing|commit differs|"
                "not canonical"
            ),
        ):
            _read_lifecycle_test_status(
                launcher,
                reader,
                reader_directory,
                binding,
                record,
                role="gate",
            )
    finally:
        os.close(reader)
        os.close(reader_directory)


def test_lifecycle_wait_reader_rejects_mid_read_name_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, "mid_read_swap.channel"
        )
    )
    record = _lifecycle_wait_record(launcher, binding)
    launcher._write_lifecycle_wait_status(
        descriptor,
        directory_descriptor,
        binding,
        record,
        role="gate",
    )
    os.close(descriptor)
    os.close(directory_descriptor)
    reader, reader_directory = (
        launcher._open_lifecycle_wait_channel_reader(
            tmp_path, binding, name=path.name
        )
    )
    moved = tmp_path / "mid_read_bound.channel"
    original_pread = launcher.os.pread
    swapped = False

    def swap_name(fd: int, size: int, offset: int) -> bytes:
        nonlocal swapped
        content = original_pread(fd, size, offset)
        if not swapped:
            swapped = True
            path.rename(moved)
            path.write_bytes(b"FOREIGN")
            path.chmod(0o600)
        return content

    monkeypatch.setattr(launcher.os, "pread", swap_name)
    try:
        with pytest.raises(
            RuntimeError, match="named identity differs"
        ):
            _read_lifecycle_test_status(
                launcher,
                reader,
                reader_directory,
                binding,
                record,
                role="gate",
            )
        assert path.read_bytes() == b"FOREIGN"
        assert moved.stat().st_ino == binding["inode"]
    finally:
        os.close(reader)
        os.close(reader_directory)


def test_lifecycle_wait_reader_final_seal_follows_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher, tmp_path, "post_validation_swap.channel"
        )
    )
    record = _lifecycle_wait_record(launcher, binding)
    launcher._write_lifecycle_wait_status(
        descriptor,
        directory_descriptor,
        binding,
        record,
        role="gate",
    )
    os.close(descriptor)
    os.close(directory_descriptor)
    reader, reader_directory = (
        launcher._open_lifecycle_wait_channel_reader(
            tmp_path, binding, name=path.name
        )
    )
    moved = tmp_path / "post_validation_bound.channel"
    original_validate = launcher.validate_lifecycle_wait_status

    def validate_then_swap(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = original_validate(*args, **kwargs)
        path.rename(moved)
        path.write_bytes(b"FOREIGN")
        path.chmod(0o600)
        return value

    monkeypatch.setattr(
        launcher,
        "validate_lifecycle_wait_status",
        validate_then_swap,
    )
    try:
        with pytest.raises(
            RuntimeError, match="named identity differs"
        ):
            _read_lifecycle_test_status(
                launcher,
                reader,
                reader_directory,
                binding,
                record,
                role="gate",
            )
        assert path.read_bytes() == b"FOREIGN"
        assert moved.stat().st_ino == binding["inode"]
    finally:
        os.close(reader)
        os.close(reader_directory)


def test_preflight_wrapper_typed_failure_uses_dedicated_recorded_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    wrapper = _wrapper_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "wrapper_fault.channel"
    )
    gate_descriptor = launcher._open_presealed_fault_channel(
        tmp_path, binding
    )
    wrapper_descriptor = os.dup(gate_descriptor)
    os.set_inheritable(wrapper_descriptor, True)
    monkeypatch.setenv(
        wrapper.FAULT_CHANNEL_FD_ENV, str(wrapper_descriptor)
    )
    receipt = _fault_channel_receipt(binding)
    context = wrapper._bind_inherited_fault_channel(receipt)
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        raise _fault_publish_error(wrapper)

    try:
        value, code, detail = wrapper._execute_with_fault_reporting(
            context, fail_once
        )
        assert value is None
        assert code == wrapper.FAULT_REPORTED_EXIT_CODE == 123
        assert detail["status"] == "typed_publish_failure_reported"
        assert calls == 1
        snapshot = launcher._read_fault_channel(
            gate_descriptor,
            binding,
            attempt_id=receipt["attempt_id"],
            owner_nonce=receipt["controller_owner_nonce"],
            launch_receipt_sha256=receipt[
                "launch_receipt_sha256"
            ],
            publisher=receipt["bindings"]["wrapper"],
        )
        assert snapshot["state"] == "valid_fault"
    finally:
        os.close(gate_descriptor)


def test_preflight_wrapper_fault_channel_write_failure_is_dedicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    wrapper = _wrapper_module()
    tmp_path.chmod(0o755)
    binding = launcher._create_fault_channel(
        tmp_path / "wrapper_fault.channel"
    )
    descriptor = launcher._open_presealed_fault_channel(
        tmp_path, binding
    )
    os.set_inheritable(descriptor, True)
    monkeypatch.setenv(
        wrapper.FAULT_CHANNEL_FD_ENV, str(descriptor)
    )
    context = wrapper._bind_inherited_fault_channel(
        _fault_channel_receipt(binding)
    )
    calls = 0

    def fail_once():
        nonlocal calls
        calls += 1
        raise _fault_publish_error(wrapper)

    monkeypatch.setattr(
        wrapper,
        "_write_fault_channel_record",
        lambda *_args: (_ for _ in ()).throw(
            OSError(5, "fixture channel write failed")
        ),
    )
    value, code, detail = wrapper._execute_with_fault_reporting(
        context, fail_once
    )
    assert value is None
    assert code == wrapper.FAULT_CHANNEL_WRITE_FAILED_EXIT_CODE == 122
    assert detail["status"] == "fault_channel_write_failed"
    assert calls == 1


def test_preflight_wrapper_total_publisher_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    calls = 0
    failure = _fault_publish_error(wrapper)

    def fail_once(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(wrapper, "_write_exclusive", fail_once)
    with pytest.raises(wrapper.ExclusivePublishError) as observed:
        wrapper._publish_wrapper_exit_total(
            tmp_path / "wrapper_exit.json",
            {"exit_code": 125},
        )
    assert observed.value is failure
    assert calls == 1


def test_preflight_launcher_reads_exact_canonical_wrapper_exit(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    tmp_path.chmod(0o755)
    path = tmp_path / "wrapper_exit.json"
    value = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_wrapper_exit_v4",
        "policy_sha256": "a" * 64,
        "exit_code": 0,
        "controller_exit_code": 0,
        "launch_failure": None,
    }
    value["wrapper_exit_sha256"] = launcher._canonical_digest(
        value, "wrapper_exit_sha256"
    )
    path.write_bytes(launcher._canonical_json_bytes(value))
    path.chmod(0o644)
    snapshot = launcher._read_exact_wrapper_exit(
        path, policy_sha256="a" * 64
    )
    assert snapshot["value"] == value
    path.write_bytes(
        json.dumps(value, indent=2).encode("utf-8") + b"\n"
    )
    with pytest.raises(RuntimeError, match="schema|canonical"):
        launcher._read_exact_wrapper_exit(
            path, policy_sha256="a" * 64
        )


def test_preflight_wrapper_concurrent_reader_never_sees_partial_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready", "rows": list(range(10000))}
    first_write = threading.Event()
    release_write = threading.Event()
    writer_done = threading.Event()
    original_write = wrapper.os.write
    writer_failures = []
    reader_failures = []
    observed = []
    write_calls = 0

    def pause_after_partial_temporary_write(descriptor: int, content) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            written = original_write(descriptor, content[:17])
            first_write.set()
            assert release_write.wait(timeout=5.0)
            return written
        return original_write(descriptor, content)

    def publish() -> None:
        try:
            wrapper._write_exclusive(path, value)
        except BaseException as exc:
            writer_failures.append(exc)
        finally:
            writer_done.set()

    def read_while_publishing() -> None:
        try:
            while not writer_done.is_set():
                if path.exists():
                    observed.append(load_json(path, "concurrent wrapper reader"))
                time.sleep(0.001)
            if path.exists():
                observed.append(load_json(path, "completed wrapper reader"))
        except BaseException as exc:
            reader_failures.append(exc)

    monkeypatch.setattr(
        wrapper.os, "write", pause_after_partial_temporary_write
    )
    writer = threading.Thread(target=publish)
    reader = threading.Thread(target=read_while_publishing)
    writer.start()
    assert first_write.wait(timeout=5.0)
    assert not path.exists()
    reader.start()
    time.sleep(0.05)
    assert not path.exists()
    release_write.set()
    writer.join(timeout=5.0)
    reader.join(timeout=5.0)
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert writer_failures == []
    assert reader_failures == []
    assert observed
    assert all(item == value for item in observed)
    assert list(tmp_path.glob(".observer_gate_ready.json.publish-*")) == []


def test_preflight_launcher_partial_temporary_is_never_final_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "launch_ownership_release.json"
    value = {"contract": "release", "rows": list(range(10000))}
    first_partial_write = threading.Event()
    release_write = threading.Event()
    failures: list[BaseException] = []
    original_write = launcher.os.write
    write_calls = 0

    def pause_partial_temporary_write(
        descriptor: int, content: bytes
    ) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            written = original_write(descriptor, content[:19])
            first_partial_write.set()
            assert release_write.wait(timeout=5.0)
            return written
        return original_write(descriptor, content)

    def publish() -> None:
        try:
            launcher._write_exclusive(path, value)
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(
        launcher.os, "write", pause_partial_temporary_write
    )
    writer = threading.Thread(target=publish)
    writer.start()
    assert first_partial_write.wait(timeout=5.0)
    assert not path.exists()
    temporary = list(
        tmp_path.glob(".launch_ownership_release.json.publish-*")
    )
    assert len(temporary) == 1
    assert temporary[0].stat().st_size == 19
    release_write.set()
    writer.join(timeout=5.0)
    assert not writer.is_alive()
    assert failures == []
    assert load_json(path, "atomic launcher publication") == value
    assert list(
        tmp_path.glob(".launch_ownership_release.json.publish-*")
    ) == []


def test_preflight_launcher_high_frequency_reader_sees_absent_or_full(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "launch_ownership_release.json"
    value = {"contract": "release", "rows": list(range(100000))}
    expected = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    failures: list[BaseException] = []
    observed: list[bytes] = []

    def publish() -> None:
        try:
            launcher._write_exclusive(path, value)
        except BaseException as exc:
            failures.append(exc)

    writer = threading.Thread(target=publish)
    writer.start()
    while writer.is_alive():
        if path.is_file():
            observed.append(path.read_bytes())
    writer.join(timeout=5.0)
    assert not writer.is_alive()
    observed.append(path.read_bytes())
    assert failures == []
    assert observed
    assert all(content == expected for content in observed)


@pytest.mark.parametrize("same_content", (True, False))
def test_preflight_launcher_concurrent_publishers_are_exact(
    tmp_path: Path,
    same_content: bool,
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "launch_ownership_release.json"
    values = [
        {"writer": 1},
        {"writer": 1 if same_content else 2},
    ]
    barrier = threading.Barrier(2)
    results: list[dict[str, Any]] = []
    failures: list[BaseException] = []

    def publish(value: dict[str, int]) -> None:
        barrier.wait()
        try:
            results.append(launcher._write_exclusive(path, value))
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=publish, args=(value,))
        for value in values
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert all(not thread.is_alive() for thread in threads)
    if same_content:
        assert failures == []
        assert {result["status"] for result in results} == {
            "committed",
            "already_committed_exact",
        }
    else:
        assert len(results) == 1
        assert results[0]["status"] == "committed"
        assert len(failures) == 1
        assert isinstance(
            failures[0], launcher.LauncherExclusivePublishError
        )
        assert failures[0].commit_state == "collision"
    assert load_json(path, "concurrent launcher publication") in values
    assert list(
        tmp_path.glob(".launch_ownership_release.json.publish-*")
    ) == []


def test_preflight_launcher_partial_write_failure_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "launch_ownership_release.json"
    original_write = launcher.os.write
    calls = 0

    def partial_then_fail(
        descriptor: int, content: bytes
    ) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(descriptor, content[:11])
        raise OSError(5, "fixture launcher partial write failure")

    monkeypatch.setattr(launcher.os, "write", partial_then_fail)
    with pytest.raises(
        launcher.LauncherExclusivePublishError
    ) as failure:
        launcher._write_exclusive(path, {"contract": "release"})
    assert failure.value.commit_state == "precommit_failed_clean"
    assert failure.value.quarantined is False
    assert not path.exists()
    assert list(
        tmp_path.glob(".launch_ownership_release.json.publish-*")
    ) == []


def test_preflight_launcher_commit_fsync_failure_quarantines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "launch_ownership_release.json"

    def fail_commit_fsync(_descriptor: int) -> None:
        raise OSError(5, "fixture launcher commit fsync failure")

    monkeypatch.setattr(
        launcher, "_launcher_fsync_dirfd", fail_commit_fsync
    )
    with pytest.raises(
        launcher.LauncherExclusivePublishError
    ) as failure:
        launcher._write_exclusive(path, {"contract": "release"})
    assert (
        failure.value.commit_state
        == "durability_unknown_quarantined"
    )
    assert failure.value.stage == "final_link_directory_fsync"
    assert failure.value.quarantined is True
    temporary = list(
        tmp_path.glob(".launch_ownership_release.json.publish-*")
    )
    assert len(temporary) == 1
    assert path.stat().st_ino == temporary[0].stat().st_ino
    assert path.stat().st_nlink == 2


def test_preflight_launcher_committed_cleanup_failure_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "launch_ownership_release.json"

    def fail_cleanup(*_args, **_kwargs) -> None:
        raise OSError(5, "fixture launcher cleanup failure")

    monkeypatch.setattr(
        launcher, "_launcher_cleanup_temporary", fail_cleanup
    )
    with pytest.raises(
        launcher.LauncherExclusivePublishError
    ) as failure:
        launcher._write_exclusive(path, {"contract": "release"})
    assert failure.value.commit_state == "committed_cleanup_error"
    assert failure.value.stage == "committed_temporary_cleanup"
    assert failure.value.quarantined is True
    assert path.is_file()
    temporary = list(
        tmp_path.glob(".launch_ownership_release.json.publish-*")
    )
    assert len(temporary) == 1
    assert path.stat().st_ino == temporary[0].stat().st_ino


def test_preflight_launcher_quarantine_forbids_target_directory_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "launch_ownership_release.json"
    target_descriptor: int | None = None
    fault_latched = False
    post_fault_calls: list[tuple[str, tuple, dict]] = []
    original_open = launcher.os.open
    original_stat = launcher.os.stat
    original_link = launcher.os.link
    original_unlink = launcher.os.unlink
    original_read = launcher._launcher_read_publication_file

    def fail_commit_fsync(descriptor: int) -> None:
        nonlocal target_descriptor, fault_latched
        target_descriptor = descriptor
        fault_latched = True
        raise OSError(5, "fixture launcher commit fsync failure")

    def record(name, function):
        def instrumented(*args, **kwargs):
            descriptors = {
                kwargs.get("dir_fd"),
                kwargs.get("src_dir_fd"),
                kwargs.get("dst_dir_fd"),
            }
            if (
                fault_latched
                and target_descriptor is not None
                and target_descriptor in descriptors
            ):
                post_fault_calls.append((name, args, dict(kwargs)))
            return function(*args, **kwargs)

        return instrumented

    def record_read(directory_descriptor, *args, **kwargs):
        if (
            fault_latched
            and directory_descriptor == target_descriptor
        ):
            post_fault_calls.append(
                (
                    "_launcher_read_publication_file",
                    (directory_descriptor, *args),
                    dict(kwargs),
                )
            )
        return original_read(directory_descriptor, *args, **kwargs)

    monkeypatch.setattr(
        launcher.os, "open", record("open", original_open)
    )
    monkeypatch.setattr(
        launcher.os, "stat", record("stat", original_stat)
    )
    monkeypatch.setattr(
        launcher.os, "link", record("link", original_link)
    )
    monkeypatch.setattr(
        launcher.os, "unlink", record("unlink", original_unlink)
    )
    monkeypatch.setattr(
        launcher, "_launcher_read_publication_file", record_read
    )
    monkeypatch.setattr(
        launcher, "_launcher_fsync_dirfd", fail_commit_fsync
    )
    with pytest.raises(
        launcher.LauncherExclusivePublishError
    ) as failure:
        launcher._write_exclusive(path, {"contract": "release"})
    assert (
        failure.value.commit_state
        == "durability_unknown_quarantined"
    )
    assert post_fault_calls == []
    assert target_descriptor is not None
    temporary = list(
        tmp_path.glob(".launch_ownership_release.json.publish-*")
    )
    assert len(temporary) == 1
    assert path.stat().st_ino == temporary[0].stat().st_ino


def test_preflight_launcher_cleanup_dir_fsync_failure_is_namespace_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "launch_ownership_release.json"
    original_fsync = launcher._launcher_fsync_dirfd
    calls = 0

    def fail_cleanup_dir_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(5, "fixture launcher cleanup fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(
        launcher, "_launcher_fsync_dirfd", fail_cleanup_dir_fsync
    )
    with pytest.raises(
        launcher.LauncherExclusivePublishError
    ) as failure:
        launcher._write_exclusive(path, {"contract": "release"})
    assert failure.value.commit_state == "committed_cleanup_error"
    assert failure.value.quarantined is True
    assert calls == 2
    assert path.is_file()
    assert path.stat().st_nlink == 1
    assert list(
        tmp_path.glob(".launch_ownership_release.json.publish-*")
    ) == []


def test_preflight_launcher_exact_existing_fsyncs_file_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "launch_ownership_release.json"
    value = {"contract": "release"}
    launcher._write_exclusive(path, value)
    original_fsync = launcher.os.fsync
    fsync_kinds: list[str] = []

    def record_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        fsync_kinds.append(
            "directory" if stat.S_ISDIR(mode) else "file"
        )
        original_fsync(descriptor)

    monkeypatch.setattr(launcher.os, "fsync", record_fsync)
    result = launcher._write_exclusive(path, value)
    assert result["status"] == "already_committed_exact"
    assert fsync_kinds.count("file") >= 2
    assert fsync_kinds.count("directory") >= 2


def test_preflight_launcher_directory_close_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "launch_ownership_release.json"
    original_close = launcher._launcher_checked_close
    directory_close_calls = 0

    def fail_directory_close(descriptor: int, label: str) -> None:
        nonlocal directory_close_calls
        if label == "launcher publication directory":
            directory_close_calls += 1
            os.close(descriptor)
            raise RuntimeError("fixture launcher directory close failure")
        original_close(descriptor, label)

    monkeypatch.setattr(
        launcher, "_launcher_checked_close", fail_directory_close
    )
    with pytest.raises(
        launcher.LauncherExclusivePublishError
    ) as failure:
        launcher._write_exclusive(path, {"contract": "release"})
    assert failure.value.commit_state == "committed_cleanup_error"
    assert failure.value.stage == "directory_close"
    assert directory_close_calls == 1


@pytest.mark.skipif(sys.platform != "linux", reason="requires os.fork")
def test_preflight_wrapper_kill_before_link_never_exposes_final(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready", "rows": list(range(1000))}
    pid = os.fork()
    if pid == 0:
        def kill_before_link(*_args, **_kwargs):
            os.kill(os.getpid(), signal.SIGKILL)
            raise AssertionError("SIGKILL returned")

        wrapper.os.link = kill_before_link
        wrapper._write_exclusive(path, value)
        os._exit(0)
    waited_pid, status = os.waitpid(pid, 0)
    assert waited_pid == pid
    assert os.WIFSIGNALED(status)
    assert os.WTERMSIG(status) == signal.SIGKILL
    assert not path.exists()
    temporary = list(
        tmp_path.glob(".observer_gate_ready.json.publish-*")
    )
    assert len(temporary) == 1
    assert load_json(temporary[0], "killed writer temporary") == value
    temporary[0].unlink()


@pytest.mark.skipif(sys.platform != "linux", reason="requires os.fork")
@pytest.mark.parametrize(
    ("window", "expected_ack"),
    (
        ("post_link_pre_fsync", b"L"),
        ("post_commit_pre_cleanup", b"C"),
    ),
)
def test_preflight_wrapper_sigkill_window_is_signal_terminal_even_with_exact_final(
    tmp_path: Path,
    window: str,
    expected_ack: bytes,
) -> None:
    launcher = _launcher_module()
    wrapper = _wrapper_module()
    attempt_root = tmp_path / f"attempt-{window}"
    attempt_root.mkdir(mode=0o755)
    os.chmod(attempt_root, 0o755)
    target_root = attempt_root / "target"
    target_root.mkdir(mode=0o755)
    os.chmod(target_root, 0o755)
    wrapper_exit_path = target_root / "wrapper_exit.json"
    policy_sha256 = hashlib.sha256(
        f"policy:{window}".encode()
    ).hexdigest()
    wrapper_exit = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_wrapper_exit_v4",
        "policy_sha256": policy_sha256,
        "exit_code": 0,
        "controller_exit_code": 0,
        "launch_failure": None,
    }
    wrapper_exit["wrapper_exit_sha256"] = wrapper._canonical_digest(
        wrapper_exit, "wrapper_exit_sha256"
    )
    fault_path = attempt_root / "wrapper_fault.channel"
    fault_binding = launcher._create_fault_channel(fault_path)
    fault_descriptor = launcher._open_presealed_fault_channel(
        attempt_root, fault_binding
    )
    ack_read, ack_write = os.pipe()
    block_read, block_write = os.pipe()
    pid = os.fork()
    waited = False
    status: int | None = None
    if pid == 0:
        os.close(ack_read)
        os.close(block_write)
        os.close(fault_descriptor)
        if window == "post_link_pre_fsync":
            original_fsync_dirfd = wrapper._fsync_dirfd
            fsync_calls = 0

            def pause_before_first_directory_fsync(
                descriptor: int,
            ) -> None:
                nonlocal fsync_calls
                fsync_calls += 1
                if fsync_calls == 1:
                    os.write(ack_write, b"L")
                    os.read(block_read, 1)
                    raise AssertionError(
                        "post-link SIGKILL block was released"
                    )
                original_fsync_dirfd(descriptor)

            wrapper._fsync_dirfd = pause_before_first_directory_fsync
        else:
            original_fsync_dirfd = wrapper._fsync_dirfd
            commit_fsync_completed = False

            def record_commit_directory_fsync(
                descriptor: int,
            ) -> None:
                nonlocal commit_fsync_completed
                original_fsync_dirfd(descriptor)
                commit_fsync_completed = True

            def pause_before_committed_cleanup(
                directory_descriptor: int,
                temporary_name: str,
                temporary_seal: Mapping[str, Any],
            ) -> None:
                if not commit_fsync_completed:
                    os._exit(91)
                os.write(ack_write, b"C")
                os.read(block_read, 1)
                raise AssertionError(
                    "post-commit SIGKILL block was released"
                )

            wrapper._fsync_dirfd = record_commit_directory_fsync
            wrapper._cleanup_sealed_temporary = (
                pause_before_committed_cleanup
            )
        wrapper._write_exclusive(wrapper_exit_path, wrapper_exit)
        os._exit(0)
    os.close(ack_write)
    os.close(block_read)
    try:
        readable, _, _ = select.select([ack_read], [], [], 10.0)
        assert readable == [ack_read]
        assert os.read(ack_read, 1) == expected_ack
        first_identity = launcher._process_identity(pid)
        second_identity = launcher._process_identity(pid)
        assert first_identity["pid"] == pid
        assert second_identity["pid"] == pid
        assert (
            first_identity["start_ticks"]
            == second_identity["start_ticks"]
        )
        os.kill(pid, 0)
        temporary = list(
            target_root.glob(".wrapper_exit.json.publish-*")
        )
        assert len(temporary) == 1
        final_stat = wrapper_exit_path.stat()
        temporary_stat = temporary[0].stat()
        assert final_stat.st_ino == temporary_stat.st_ino
        assert final_stat.st_dev == temporary_stat.st_dev
        assert final_stat.st_nlink == 2
        assert temporary_stat.st_nlink == 2
        before_kill_snapshot = launcher._read_fault_channel(
            fault_descriptor,
            fault_binding,
            attempt_id="a" * 64,
            owner_nonce="b" * 64,
            launch_receipt_sha256="c" * 64,
            publisher={"path": "wrapper", "sha256": "d" * 64},
        )
        assert before_kill_snapshot["state"] == "empty"
        assert fault_path.stat().st_size == 0
        os.kill(pid, signal.SIGKILL)
        waited_pid, status = os.waitpid(pid, 0)
        waited = True
        assert waited_pid == pid
        assert os.WIFSIGNALED(status)
        assert os.WTERMSIG(status) == signal.SIGKILL
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        fault_snapshot = launcher._read_fault_channel(
            fault_descriptor,
            fault_binding,
            attempt_id="a" * 64,
            owner_nonce="b" * 64,
            launch_receipt_sha256="c" * 64,
            publisher={"path": "wrapper", "sha256": "d" * 64},
        )
        assert fault_snapshot["state"] == "empty"
        wrapper_exit_read_calls = 0

        def forbidden_wrapper_exit_reader():
            nonlocal wrapper_exit_read_calls
            wrapper_exit_read_calls += 1
            raise AssertionError(
                "signal terminal read target wrapper_exit"
            )

        outcome = launcher._evaluate_wrapper_outcome(
            returncode=-signal.SIGKILL,
            fault_snapshot=fault_snapshot,
            fault_validation_failure=None,
            fault_close_failure=None,
            wrapper_exit_reader=forbidden_wrapper_exit_reader,
        )
        gate_terminal = {
            "returncode": -signal.SIGKILL,
            "exit_kind": "signal",
            "exit_code": None,
            "signal_number": signal.SIGKILL,
            "fault_channel": fault_binding,
            "fault_channel_snapshot": fault_snapshot,
            "fault_channel_validation_failure": None,
            "fault_channel_close_failure": None,
            "wrapper_outcome": outcome,
        }
        assert gate_terminal["signal_number"] == signal.SIGKILL
        assert gate_terminal["fault_channel_snapshot"]["state"] == "empty"
        assert gate_terminal["wrapper_outcome"] == {
            "status": "wrapper_child_failed",
            "exit_code": 128 + signal.SIGKILL,
            "failure": None,
            "wrapper_exit": None,
        }
        assert wrapper_exit_read_calls == 0
        assert wrapper_exit_path.read_bytes() == wrapper._canonical_json(
            wrapper_exit
        )
        with pytest.raises(
            RuntimeError, match="identity or size differs"
        ):
            launcher._read_exact_wrapper_exit(
                wrapper_exit_path,
                policy_sha256=policy_sha256,
            )
        assert len(temporary) == 1
    finally:
        if not waited:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        os.close(ack_read)
        os.close(block_write)
        os.close(fault_descriptor)


def test_preflight_wrapper_dir_fsync_failure_keeps_complete_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready", "rows": list(range(1000))}
    original_fsync_directory = wrapper._fsync_dirfd
    calls = 0

    def fail_after_link(directory_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert path.exists()
            assert load_json(path, "linked wrapper publication") == value
            raise OSError("fixture directory fsync failure")
        original_fsync_directory(directory_descriptor)

    monkeypatch.setattr(wrapper, "_fsync_dirfd", fail_after_link)
    with pytest.raises(
        wrapper.ExclusivePublishError,
        match="durability_unknown_quarantined",
    ) as failure:
        wrapper._write_exclusive(path, value)
    assert (
        failure.value.commit_state
        == "durability_unknown_quarantined"
    )
    assert failure.value.stage == "final_link_directory_fsync"
    assert failure.value.quarantined is True
    assert failure.value.payload == {
        "size": len(wrapper._canonical_json(value)),
        "sha256": hashlib.sha256(
            wrapper._canonical_json(value)
        ).hexdigest(),
    }
    assert calls == 1
    assert load_json(path, "complete wrapper publication") == value
    temporary = list(
        tmp_path.glob(".observer_gate_ready.json.publish-*")
    )
    assert len(temporary) == 1
    assert temporary[0].stat().st_ino == path.stat().st_ino
    temporary[0].unlink()


def test_preflight_wrapper_quarantine_forbids_target_directory_syscalls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready", "rows": list(range(1000))}
    target_descriptor = None
    fault_latched = False
    post_fault_calls = []
    original_fsync_dirfd = wrapper._fsync_dirfd
    original_open = wrapper.os.open
    original_stat = wrapper.os.stat
    original_link = wrapper.os.link
    original_unlink = wrapper.os.unlink
    original_read_dirfd_regular = wrapper._read_dirfd_regular

    def fail_commit_fsync(descriptor: int) -> None:
        nonlocal target_descriptor, fault_latched
        target_descriptor = descriptor
        fault_latched = True
        raise OSError(5, "fixture commit directory fsync failure")

    def record(name, function):
        def instrumented(*args, **kwargs):
            descriptors = {
                kwargs.get("dir_fd"),
                kwargs.get("src_dir_fd"),
                kwargs.get("dst_dir_fd"),
            }
            if (
                fault_latched
                and target_descriptor is not None
                and target_descriptor in descriptors
            ):
                post_fault_calls.append((name, args, dict(kwargs)))
            return function(*args, **kwargs)

        return instrumented

    def record_read(directory_descriptor, *args, **kwargs):
        if fault_latched and directory_descriptor == target_descriptor:
            post_fault_calls.append(
                (
                    "_read_dirfd_regular",
                    (directory_descriptor, *args),
                    dict(kwargs),
                )
            )
        return original_read_dirfd_regular(
            directory_descriptor, *args, **kwargs
        )

    monkeypatch.setattr(wrapper.os, "open", record("open", original_open))
    monkeypatch.setattr(wrapper.os, "stat", record("stat", original_stat))
    monkeypatch.setattr(wrapper.os, "link", record("link", original_link))
    monkeypatch.setattr(
        wrapper.os, "unlink", record("unlink", original_unlink)
    )
    monkeypatch.setattr(
        wrapper, "_read_dirfd_regular", record_read
    )
    monkeypatch.setattr(wrapper, "_fsync_dirfd", fail_commit_fsync)
    with pytest.raises(wrapper.ExclusivePublishError) as failure:
        wrapper._write_exclusive(path, value)
    assert (
        failure.value.commit_state
        == "durability_unknown_quarantined"
    )
    assert failure.value.stage == "final_link_directory_fsync"
    assert post_fault_calls == []
    assert target_descriptor is not None
    assert path.exists()
    temporary = list(
        tmp_path.glob(".observer_gate_ready.json.publish-*")
    )
    assert len(temporary) == 1
    monkeypatch.setattr(wrapper, "_fsync_dirfd", original_fsync_dirfd)
    temporary[0].unlink()


def test_preflight_wrapper_cleanup_dir_fsync_failure_quarantines_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready"}
    original_fsync_dirfd = wrapper._fsync_dirfd
    calls = 0

    def fail_cleanup_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(5, "fixture cleanup directory fsync failure")
        original_fsync_dirfd(descriptor)

    monkeypatch.setattr(wrapper, "_fsync_dirfd", fail_cleanup_fsync)
    with pytest.raises(wrapper.ExclusivePublishError) as failure:
        wrapper._write_exclusive(path, value)
    assert failure.value.commit_state == "committed_cleanup_error"
    assert failure.value.stage == "committed_temporary_cleanup"
    assert failure.value.quarantined is True
    assert calls == 2
    assert load_json(path, "cleanup fsync committed final") == value
    assert list(tmp_path.glob(f".{path.name}.publish-*")) == []


def test_preflight_wrapper_rollback_dir_fsync_failure_quarantines_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready"}
    original_read = wrapper._read_dirfd_regular
    original_fsync_dirfd = wrapper._fsync_dirfd
    fsync_calls = 0

    def fail_final_read(directory_descriptor, name, **kwargs):
        if name == path.name:
            raise OSError(5, "fixture final verification failure")
        return original_read(directory_descriptor, name, **kwargs)

    def fail_rollback_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        raise OSError(5, "fixture rollback directory fsync failure")

    monkeypatch.setattr(
        wrapper, "_read_dirfd_regular", fail_final_read
    )
    monkeypatch.setattr(wrapper, "_fsync_dirfd", fail_rollback_fsync)
    with pytest.raises(wrapper.ExclusivePublishError) as failure:
        wrapper._write_exclusive(path, value)
    assert (
        failure.value.commit_state
        == "durability_unknown_quarantined"
    )
    assert failure.value.stage == "linked_final_read_rollback"
    assert failure.value.quarantined is True
    assert fsync_calls == 1
    assert not path.exists()
    temporary = list(
        tmp_path.glob(f".{path.name}.publish-*")
    )
    assert len(temporary) == 1
    monkeypatch.setattr(wrapper, "_fsync_dirfd", original_fsync_dirfd)
    temporary[0].unlink()


def test_preflight_wrapper_exclusive_publish_hides_partial_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    original_write = wrapper.os.write
    write_calls = 0

    def fail_after_partial_write(descriptor: int, content) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return original_write(descriptor, content[:7])
        raise OSError("fixture partial publication failure")

    monkeypatch.setattr(wrapper.os, "write", fail_after_partial_write)
    with pytest.raises(
        wrapper.ExclusivePublishError,
        match="precommit_failed_clean",
    ) as failure:
        wrapper._write_exclusive(
            path, {"contract": "ready", "rows": list(range(1000))}
        )
    assert failure.value.commit_state == "precommit_failed_clean"
    assert not path.exists()
    assert list(tmp_path.glob(".observer_gate_ready.json.publish-*")) == []


def test_preflight_wrapper_exact_republish_is_idempotent(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready", "rows": list(range(1000))}
    first = wrapper._write_exclusive(path, value)
    second = wrapper._write_exclusive(path, value)
    assert first["status"] == "committed"
    assert second["status"] == "already_committed_exact"
    assert first["payload_sha256"] == second["payload_sha256"]
    assert load_json(path, "idempotent wrapper publication") == value
    assert list(tmp_path.glob(".observer_gate_ready.json.publish-*")) == []


@pytest.mark.parametrize("collision_type", ("symlink", "directory"))
def test_preflight_wrapper_rejects_unsafe_existing_final(
    tmp_path: Path,
    collision_type: str,
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    target = tmp_path / "foreign.json"
    target.write_text('{"foreign":true}\n', encoding="utf-8")
    if collision_type == "symlink":
        path.symlink_to(target)
    else:
        path.mkdir()
    with pytest.raises(
        wrapper.ExclusivePublishError, match="collision"
    ):
        wrapper._write_exclusive(path, {"contract": "ready"})
    if collision_type == "symlink":
        assert path.is_symlink()
        assert target.read_text(encoding="utf-8") == '{"foreign":true}\n'
    else:
        assert path.is_dir()
    assert list(tmp_path.glob(".observer_gate_ready.json.publish-*")) == []


def test_preflight_wrapper_rejects_insecure_publication_directory(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper_module()
    directory = tmp_path / "insecure"
    directory.mkdir(mode=0o777)
    directory.chmod(0o777)
    path = directory / "observer_gate_ready.json"
    with pytest.raises(RuntimeError, match="permissions differ"):
        wrapper._write_exclusive(path, {"contract": "ready"})
    assert not path.exists()


def test_preflight_wrapper_new_leaf_directory_is_explicitly_safe(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper_module()
    directory = tmp_path / "new" / "publication"
    path = directory / "observer_gate_ready.json"
    previous_umask = os.umask(0o002)
    try:
        created = wrapper._ensure_secure_leaf_directories(
            tmp_path, ("new", "publication")
        )
        result = wrapper._write_exclusive(
            path, {"contract": "ready"}
        )
    finally:
        os.umask(previous_umask)
    assert result["status"] == "committed"
    assert created == directory
    assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    assert load_json(path, "safe leaf publication") == {
        "contract": "ready"
    }


def test_preflight_wrapper_detects_temporary_name_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    original_link = wrapper.os.link
    swapped_names = []

    def swap_before_link(source, target, **kwargs):
        directory_descriptor = kwargs["src_dir_fd"]
        os.unlink(source, dir_fd=directory_descriptor)
        descriptor = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=directory_descriptor,
        )
        try:
            os.write(descriptor, b'{"swapped":true}\n')
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        swapped_names.append(source)
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(wrapper.os, "link", swap_before_link)
    with pytest.raises(
        wrapper.ExclusivePublishError,
        match="collision",
    ) as failure:
        wrapper._write_exclusive(path, {"contract": "ready"})
    assert failure.value.commit_state == "collision"
    assert not path.exists()
    assert len(swapped_names) == 1
    swapped = tmp_path / swapped_names[0]
    assert load_json(swapped, "foreign swapped temporary") == {
        "swapped": True
    }
    swapped.unlink()


def test_preflight_wrapper_secure_reader_high_frequency_visibility(
    tmp_path: Path,
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready", "rows": list(range(10000))}
    content = wrapper._canonical_json(value)
    failures = []
    complete_reads = []

    def publish() -> None:
        try:
            wrapper._write_exclusive(path, value)
        except BaseException as exc:
            failures.append(exc)

    writer = threading.Thread(target=publish)
    writer.start()
    while writer.is_alive():
        try:
            read = wrapper._secure_read_file(path, missing_ok=True)
            if read is not None:
                complete_reads.append(read[0])
        except BaseException as exc:
            failures.append(exc)
            break
    writer.join(timeout=5.0)
    assert not writer.is_alive()
    final = wrapper._secure_read_file(path)
    assert final is not None
    complete_reads.append(final[0])
    assert failures == []
    assert complete_reads
    assert all(item == content for item in complete_reads)


def test_preflight_wrapper_write_retries_eintr_and_short_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready", "rows": list(range(1000))}
    original_write = wrapper.os.write
    calls = 0

    def interrupted_then_short(descriptor: int, content) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InterruptedError("fixture EINTR")
        if calls == 2:
            return original_write(descriptor, content[:11])
        return original_write(descriptor, content)

    monkeypatch.setattr(wrapper.os, "write", interrupted_then_short)
    result = wrapper._write_exclusive(path, value)
    assert result["status"] == "committed"
    assert calls >= 3
    assert load_json(path, "EINTR wrapper publication") == value


def test_preflight_wrapper_zero_write_fails_before_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    monkeypatch.setattr(wrapper.os, "write", lambda *_args: 0)
    with pytest.raises(
        wrapper.ExclusivePublishError,
        match="precommit_failed_clean",
    ) as failure:
        wrapper._write_exclusive(path, {"contract": "ready"})
    assert failure.value.commit_state == "precommit_failed_clean"
    assert not path.exists()
    assert list(tmp_path.glob(".observer_gate_ready.json.publish-*")) == []


def test_preflight_wrapper_committed_cleanup_error_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    value = {"contract": "ready", "rows": list(range(1000))}
    original_unlink = wrapper.os.unlink
    temporary_names = []

    def fail_temporary_unlink(name, **kwargs):
        if str(name).startswith(f".{path.name}.publish-"):
            temporary_names.append(str(name))
            raise OSError("fixture temporary cleanup failure")
        return original_unlink(name, **kwargs)

    monkeypatch.setattr(wrapper.os, "unlink", fail_temporary_unlink)
    with pytest.raises(
        wrapper.ExclusivePublishError,
        match="committed_cleanup_error",
    ) as failure:
        wrapper._write_exclusive(path, value)
    assert failure.value.commit_state == "committed_cleanup_error"
    assert failure.value.status == "committed_cleanup_error"
    assert failure.value.quarantined is True
    assert load_json(path, "committed wrapper final") == value
    assert len(set(temporary_names)) == 1
    monkeypatch.setattr(wrapper.os, "unlink", original_unlink)
    (tmp_path / temporary_names[0]).unlink()


@pytest.mark.parametrize("same_content", (True, False))
def test_preflight_wrapper_concurrent_publishers_are_exact(
    tmp_path: Path,
    same_content: bool,
) -> None:
    wrapper = _wrapper_module()
    path = tmp_path / "observer_gate_ready.json"
    barrier = threading.Barrier(2)
    values = [
        {"writer": 1},
        {"writer": 1 if same_content else 2},
    ]
    results = []
    failures = []

    def publish(value: dict) -> None:
        barrier.wait()
        try:
            results.append(wrapper._write_exclusive(path, value))
        except BaseException as exc:
            failures.append(exc)

    threads = [
        threading.Thread(target=publish, args=(value,))
        for value in values
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert all(not thread.is_alive() for thread in threads)
    if same_content:
        assert failures == []
        assert {result["status"] for result in results} == {
            "committed",
            "already_committed_exact",
        }
        assert load_json(path, "same-content race final") == values[0]
    else:
        assert len(results) == 1
        assert results[0]["status"] == "committed"
        assert len(failures) == 1
        assert isinstance(
            failures[0], wrapper.ExclusivePublishError
        )
        assert failures[0].status == "collision"
        assert load_json(path, "different-content race final") in values
    assert list(tmp_path.glob(".observer_gate_ready.json.publish-*")) == []


def test_gpu_pre_ready_admission_failure_writes_terminal_without_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "1" * 64}
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}\n", encoding="utf-8")
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monkeypatch.setattr(
        module,
        "assert_resource_admission",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CanonicalScreeningError("fixture admission race")
        ),
    )
    claim = {
        "controller_claim_sha256": "3" * 64,
        "wrapper_claim": {"canonical_sha256": "4" * 64},
        "observer_launch": {"canonical_sha256": "5" * 64},
    }
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module,
        "_validate_gpu_wrapper_provenance",
        lambda *_args: (
            claim["wrapper_claim"],
            claim["observer_launch"],
        ),
    )
    monkeypatch.setattr(
        module,
        "_write_gpu_controller_claim",
        lambda *_args: _mock_controller_claim(
            tmp_path / "claim.json", claim
        ),
    )
    with pytest.raises(CanonicalScreeningError, match="admission race"):
        module._run_gpu_phase(
            policy, policy_path, paths, "screen512"
        )
    terminal = load_json(
        paths["gpu_control"] / "screen512" / "controller_terminal.json",
        "GPU terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["stage"] == "startup_admission"
    assert not paths["run_requests"].exists()


def test_gpu_preclaim_failure_writes_only_bootstrap_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "6" * 64}
    config = tmp_path / "policy.json"
    config.write_text("{}\n", encoding="utf-8")
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monkeypatch.delenv("TMUX", raising=False)
    with pytest.raises(CanonicalScreeningError, match="inside tmux"):
        module._run_gpu_phase(policy, config, paths, "screen512")
    terminal = load_json(
        paths["gpu_control"] / "screen512" / "bootstrap_terminal.json",
        "bootstrap terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["stage"] == "tmux_bootstrap"
    assert terminal["controller_claim"] is None
    assert not (
        paths["gpu_control"] / "screen512" / "controller_terminal.json"
    ).exists()


def test_observer_ready_timeout_is_bounded_and_writes_no_requests(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "2" * 64}
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    with pytest.raises(CanonicalScreeningError, match="timed out"):
        module._wait_observer_ready(
            policy,
            paths,
            "screen512",
            {"controller_ready_sha256": "3" * 64},
            {"canonical_sha256": "4" * 64},
            timeout_seconds=0.01,
        )
    assert not paths["run_requests"].exists()


def test_duplicate_observer_claim_fails_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "5" * 64}
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    claim_path = (
        paths["gpu_control"] / "smoke8" / "observer_claim.json"
    )
    write_exclusive_json(claim_path, {"occupied": True})
    monkeypatch.setenv("TMUX", "fixture")
    with pytest.raises(CanonicalScreeningError, match="already exists"):
        module._run_monitor(policy, paths, "smoke8")


def test_gpu_resource_recheck_rejects_uuid_registry_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "6" * 64}
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    original_snapshot = {
        "authorized_gpu_registry": [
            {"physical_gpu_index": 0, "physical_gpu_uuid": _gpu_uuid(0)}
        ],
        "compute_processes": [],
        "gpus": [{"temperature_c": 40}],
    }
    admission_value = {
        "policy_sha256": policy["policy_sha256"],
        "snapshot": original_snapshot,
    }
    admission_value["admission_sha256"] = canonical_digest(
        admission_value, "admission_sha256"
    )
    admission_path = tmp_path / "admission.json"
    write_exclusive_json(admission_path, admission_value)
    admission = {
        **_bound(admission_path),
        "canonical_sha256": admission_value["admission_sha256"],
    }
    raced_snapshot = json.loads(json.dumps(original_snapshot))
    raced_snapshot["authorized_gpu_registry"][0]["physical_gpu_uuid"] = _gpu_uuid(1)
    monkeypatch.setattr(
        module,
        "assert_resource_admission",
        lambda *_args, **_kwargs: raced_snapshot,
    )
    with pytest.raises(CanonicalScreeningError, match="differs"):
        module._write_gpu_resource_recheck(
            policy,
            paths,
            "screen512",
            admission,
            {
                "violated": False,
                "swap_consecutive_io": 0,
                "resource_window_sha256": "7" * 64,
            },
        )


def test_final_request_set_rejects_partial_intent_coverage(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    policy, request = _run_fixture(tmp_path)
    request_path = tmp_path / "request.json"
    write_exclusive_json(request_path, request)
    candidate = request["candidate"]
    base = {
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "checkpoint_model": candidate["checkpoint_model"],
        "mode": "smoke8",
        "sample_count": 8,
        "seed": 4549,
        "batch_size": 2,
        "admission_sha256": request["admission"]["canonical_sha256"],
    }
    intents = {
        "request_count": 2,
        "requests": [
            {
                **base,
                "candidate_id": candidate["candidate_id"],
                "replicate": "primary",
            },
            {
                **base,
                "candidate_id": "missing-candidate",
                "replicate": "repeat",
            },
        ],
    }
    with pytest.raises(CanonicalScreeningError, match="coverage"):
        module._validate_final_requests_against_intents(
            [request_path],
            intents,
            policy,
            request["controller_ready"],
            request["observer_ready"],
        )


def test_runtime_guard_first_sample_exposes_monitor_thread_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    monkeypatch.setattr(
        module,
        "_cpu_times",
        lambda: (_ for _ in ()).throw(RuntimeError("fixture thread failure")),
    )
    guard = module.RuntimeResourceGuard(
        policy, tmp_path / "guard.jsonl", tmp_path
    )
    guard.start()
    try:
        with pytest.raises(CanonicalScreeningError, match="thread failure"):
            guard.wait_first_sample(1.0)
    finally:
        summary = guard.stop()
    assert summary["thread_failure"]["type"] == "RuntimeError"


def test_worker_revalidates_ready_files_before_cuda(
    tmp_path: Path,
) -> None:
    policy, request = _run_fixture(tmp_path)
    controller_path = Path(request["controller_ready"]["path"])
    controller_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CanonicalScreeningError, match="file binding"):
        _assert_ready_barrier(request, policy)


def _legacy_archive_v2_fixture(
    tmp_path: Path,
) -> tuple[Any, dict[str, Any]]:
    launcher = _launcher_module()
    campaign_root = (tmp_path / "campaign").resolve()
    campaign_root.mkdir(mode=0o755)
    os.chmod(campaign_root, 0o755)
    policy_sha256 = hashlib.sha256(b"legacy-policy").hexdigest()
    policy_root = (
        campaign_root / "by_policy" / policy_sha256
    )
    request_root = (
        policy_root / "checkpoint_preflight" / "requests"
    )
    request_root.mkdir(parents=True)

    def write(path: Path, value: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    plan_requests: list[dict[str, Any]] = []
    manifest_requests: list[dict[str, str]] = []
    for index, checkpoint_model in enumerate(("raw", "raw", "ema")):
        checkpoint_sha256 = hashlib.sha256(
            f"checkpoint:{index}".encode()
        ).hexdigest()
        request = {
            "schema_version": 1,
            "contract_type": (
                "safa_canonical_checkpoint_preflight_request_v1"
            ),
            "policy_sha256": policy_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_model": checkpoint_model,
        }
        request["preflight_request_sha256"] = (
            launcher._canonical_digest(
                request, "preflight_request_sha256"
            )
        )
        request_path = (
            request_root
            / f"{checkpoint_sha256}__{checkpoint_model}.json"
        )
        write(request_path, request)
        plan_requests.append(request)
        manifest_requests.append(
            {
                "checkpoint_model": checkpoint_model,
                "checkpoint_sha256": checkpoint_sha256,
                "path": str(request_path),
                "preflight_request_sha256": request[
                    "preflight_request_sha256"
                ],
                "sha256": hashlib.sha256(
                    request_path.read_bytes()
                ).hexdigest(),
            }
        )
    counts = {
        "preflight_requests": 3,
        "distinct_checkpoint_sha256": 3,
        "distinct_raw_checkpoint_sha256": 2,
        "distinct_ema_checkpoint_sha256": 1,
    }
    plan = {
        "schema_version": 1,
        "contract_type": "safa_canonical_checkpoint_plan_v1",
        "policy_sha256": policy_sha256,
        "counts": counts,
        "preflight_requests": plan_requests,
    }
    plan["checkpoint_plan_sha256"] = launcher._canonical_digest(
        plan, "checkpoint_plan_sha256"
    )
    plan_path = policy_root / "checkpoint_plan.json"
    write(plan_path, plan)
    manifest = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_request_manifest_v1"
        ),
        "policy_sha256": policy_sha256,
        "checkpoint_plan": {
            "path": str(plan_path),
            "sha256": hashlib.sha256(
                plan_path.read_bytes()
            ).hexdigest(),
            "canonical_sha256": plan["checkpoint_plan_sha256"],
        },
        "request_count": 3,
        "requests": manifest_requests,
    }
    manifest["preflight_request_manifest_sha256"] = (
        launcher._canonical_digest(
            manifest, "preflight_request_manifest_sha256"
        )
    )
    manifest_path = (
        policy_root
        / "checkpoint_preflight"
        / "preflight_request_manifest.json"
    )
    write(manifest_path, manifest)
    fixed_mtime_ns = 1_785_180_247_323_155_491
    os.utime(
        manifest_path,
        ns=(fixed_mtime_ns, fixed_mtime_ns),
    )
    tree = launcher._legacy_policy_tree_snapshot(policy_root)
    kwargs = {
        "campaign_root": campaign_root,
        "policy_sha256": policy_sha256,
        "controller_owner_nonce": hashlib.sha256(
            b"legacy-owner"
        ).hexdigest(),
        "observer_session": (
            launcher.OBSERVER_SESSION_PREFIX
            + hashlib.sha256(b"legacy-observer").hexdigest()
        ),
        "prepare_completion_path": manifest_path,
        "prepare_completion_file_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "prepare_completion_canonical_sha256": manifest[
            "preflight_request_manifest_sha256"
        ],
        "prepare_completion_mtime_ns": fixed_mtime_ns,
        "old_policy_tree_sha256": tree["sha256"],
    }
    return launcher, kwargs


def test_legacy_archive_v2_is_deterministic_idempotent_and_honest(
    tmp_path: Path,
) -> None:
    launcher, kwargs = _legacy_archive_v2_fixture(tmp_path)
    policy_root = (
        kwargs["campaign_root"]
        / "by_policy"
        / kwargs["policy_sha256"]
    )
    old_tree = launcher._legacy_policy_tree_snapshot(policy_root)
    old_files = {
        path.relative_to(policy_root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_ino,
            path.stat().st_mtime_ns,
        )
        for path in policy_root.rglob("*")
        if path.is_file()
    }
    first = launcher.archive_legacy_untracked_failure_v2(
        **kwargs,
        archived_at="2026-07-28T01:00:00+00:00",
    )
    second = launcher.archive_legacy_untracked_failure_v2(
        **kwargs,
        archived_at="2026-07-28T02:00:00+00:00",
    )
    assert first["created"] is True
    assert second["created"] is False
    assert first["archive_id"] == second["archive_id"]
    assert first["archive"] == second["archive"]
    assert (
        second["value"]["archived_at"]
        == "2026-07-28T01:00:00+00:00"
    )
    value = launcher.validate_legacy_untracked_failure_archive_v2(
        first["value"]
    )
    evidence = value["immutable_evidence"]
    assert evidence["original_attempt_id"] is None
    assert evidence["original_started_registry"] is None
    assert evidence["launch_receipt"] is None
    assert evidence["wrapper_claim"] is None
    assert evidence["failure_stage"] == "launch_before_ownership"
    assert evidence["evidence_level"] == (
        "operator_observed_unsealed"
    )
    assert evidence["occurrence_time"]["exact"] is None
    assert evidence["occurrence_time"]["not_after"] is None
    assert evidence["occurrence_time"]["precision"] == (
        "lower_bound_only_or_unknown"
    )
    completion = evidence["occurrence_time"]["not_before"][
        "prepare_completion_artifact"
    ]
    assert completion["mtime_ns"] == (
        kwargs["prepare_completion_mtime_ns"]
    )
    assert evidence["prepared_policy"]["request_set"] == {
        "derivation": launcher.LEGACY_REQUEST_SET_DERIVATION,
        "sha256": evidence["prepared_policy"]["request_set"][
            "sha256"
        ],
        "request_count": 3,
        "raw_count": 2,
        "ema_count": 1,
    }
    assert evidence["absence_snapshot"] == {
        "policy_root": str(policy_root),
        "preflight_results": 0,
        "attempt_claims": 0,
        "attempt_terminals": 0,
        "preflight_control_files": 0,
        "generated_png": 0,
        "run_requests": 0,
        "scientific_execution": 0,
        "scientific_execution_started": False,
    }
    archive_path = Path(first["archive"]["path"])
    assert archive_path == (
        kwargs["campaign_root"]
        / "untracked_failure_archives"
        / "by_policy"
        / kwargs["policy_sha256"]
        / f"{first['archive_id']}.json"
    )
    assert not (
        kwargs["campaign_root"] / "preflight_launch_attempts"
    ).exists()
    assert (
        launcher._legacy_policy_tree_snapshot(policy_root)
        == old_tree
    )
    assert {
        path.relative_to(policy_root).as_posix(): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_ino,
            path.stat().st_mtime_ns,
        )
        for path in policy_root.rglob("*")
        if path.is_file()
    } == old_files


def test_legacy_archive_v2_reader_sees_absent_then_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher, kwargs = _legacy_archive_v2_fixture(tmp_path)
    first_partial_write = threading.Event()
    release_write = threading.Event()
    failures: list[BaseException] = []
    results: list[dict[str, Any]] = []
    original_write = launcher.os.write
    write_calls = 0

    def pause_archive_temporary(
        descriptor: int, content: bytes
    ) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            written = original_write(descriptor, content[:23])
            first_partial_write.set()
            assert release_write.wait(timeout=5.0)
            return written
        return original_write(descriptor, content)

    def archive() -> None:
        try:
            results.append(
                launcher.archive_legacy_untracked_failure_v2(
                    **kwargs,
                    archived_at="2026-07-28T01:00:00+00:00",
                )
            )
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(
        launcher.os, "write", pause_archive_temporary
    )
    writer = threading.Thread(target=archive)
    writer.start()
    assert first_partial_write.wait(timeout=5.0)
    archive_root = (
        kwargs["campaign_root"]
        / "untracked_failure_archives/by_policy"
        / kwargs["policy_sha256"]
    )
    assert list(archive_root.glob("*.json")) == []
    temporary = list(archive_root.glob(".*.json.publish-*"))
    assert len(temporary) == 1
    assert temporary[0].stat().st_size == 23
    release_write.set()
    writer.join(timeout=5.0)
    assert not writer.is_alive()
    assert failures == []
    assert len(results) == 1
    archive_path = Path(results[0]["archive"]["path"])
    assert archive_path.is_file()
    assert launcher.validate_legacy_untracked_failure_archive_v2(
        load_json(archive_path, "atomic legacy archive")
    ) == results[0]["value"]
    assert list(archive_root.glob(".*.json.publish-*")) == []


def test_legacy_archive_v2_collision_difference_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher, kwargs = _legacy_archive_v2_fixture(tmp_path)
    first = launcher.archive_legacy_untracked_failure_v2(**kwargs)
    monkeypatch.setattr(
        launcher,
        "_derive_legacy_failure_archive_id",
        lambda _evidence: first["archive_id"],
    )
    different = dict(kwargs)
    different["controller_owner_nonce"] = hashlib.sha256(
        b"different-owner"
    ).hexdigest()
    with pytest.raises(RuntimeError, match="collision differs"):
        launcher.archive_legacy_untracked_failure_v2(**different)


def test_legacy_archive_v2_rejects_path_and_symlink_ambiguity(
    tmp_path: Path,
) -> None:
    launcher, kwargs = _legacy_archive_v2_fixture(tmp_path)
    relative = dict(kwargs)
    relative["campaign_root"] = Path("campaign")
    with pytest.raises(RuntimeError, match="not absolute"):
        launcher.archive_legacy_untracked_failure_v2(**relative)
    wrong_path = dict(kwargs)
    wrong_path["prepare_completion_path"] = (
        kwargs["campaign_root"] / "outside.json"
    )
    with pytest.raises(RuntimeError, match="path differs"):
        launcher.archive_legacy_untracked_failure_v2(**wrong_path)
    policy_root = (
        kwargs["campaign_root"]
        / "by_policy"
        / kwargs["policy_sha256"]
    )
    (policy_root / "ambiguous-link").symlink_to(
        policy_root / "checkpoint_plan.json"
    )
    with pytest.raises(RuntimeError, match="contains a symlink"):
        launcher.archive_legacy_untracked_failure_v2(**kwargs)
    (policy_root / "ambiguous-link").unlink()
    outside = (tmp_path / "outside-archive").resolve()
    outside.mkdir()
    (
        kwargs["campaign_root"] / "untracked_failure_archives"
    ).symlink_to(outside, target_is_directory=True)
    with pytest.raises(
        (RuntimeError, NotADirectoryError),
        match="contains a non-directory|Not a directory",
    ):
        launcher.archive_legacy_untracked_failure_v2(**kwargs)


def test_legacy_archive_v2_rejects_nonzero_scientific_evidence(
    tmp_path: Path,
) -> None:
    launcher, kwargs = _legacy_archive_v2_fixture(tmp_path)
    policy_root = (
        kwargs["campaign_root"]
        / "by_policy"
        / kwargs["policy_sha256"]
    )
    result = (
        policy_root
        / "checkpoint_preflight"
        / "results"
        / "unexpected.json"
    )
    result.parent.mkdir()
    result.write_text("{}\n", encoding="utf-8")
    kwargs["old_policy_tree_sha256"] = (
        launcher._legacy_policy_tree_snapshot(policy_root)["sha256"]
    )
    with pytest.raises(
        RuntimeError, match="not zero scientific execution"
    ):
        launcher.archive_legacy_untracked_failure_v2(**kwargs)


def test_legacy_archive_v2_id_cannot_be_reused_as_launch_attempt(
    tmp_path: Path,
) -> None:
    launcher, kwargs = _legacy_archive_v2_fixture(tmp_path)
    archived = launcher.archive_legacy_untracked_failure_v2(**kwargs)
    with pytest.raises(
        RuntimeError, match="collides with a legacy archive ID"
    ):
        launcher.launch_preflight(
            repo_root=tmp_path.resolve(),
            config=(tmp_path / "missing.json").resolve(),
            campaign_root=kwargs["campaign_root"],
            policy_sha256=kwargs["policy_sha256"],
            python=sys.executable,
            attempt_id=archived["archive_id"],
            owner_nonce=hashlib.sha256(b"owner").hexdigest(),
            observer_suffix=hashlib.sha256(b"observer").hexdigest(),
        )
    assert not (
        kwargs["campaign_root"] / "preflight_launch_attempts"
    ).exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "original_attempt_id",
        "exact_time",
        "sealed_operator_evidence",
        "owner_seal_field",
    ),
)
def test_legacy_archive_v2_rejects_dishonest_evidence_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    launcher, kwargs = _legacy_archive_v2_fixture(tmp_path)
    archived = launcher.archive_legacy_untracked_failure_v2(**kwargs)
    value = json.loads(json.dumps(archived["value"]))
    evidence = value["immutable_evidence"]
    if mutation == "original_attempt_id":
        evidence["original_attempt_id"] = "0" * 64
    elif mutation == "exact_time":
        evidence["occurrence_time"]["exact"] = (
            "2026-07-28T01:00:00+00:00"
        )
    elif mutation == "sealed_operator_evidence":
        evidence["evidence_level"] = "ownership_sealed"
    else:
        evidence["owner_seal"] = {"controller_owner_nonce": "0" * 64}
    value["archive_id"] = (
        launcher._derive_legacy_failure_archive_id(evidence)
    )
    value["legacy_failure_archive_sha256"] = (
        launcher._canonical_digest(
            value, "legacy_failure_archive_sha256"
        )
    )
    with pytest.raises(RuntimeError, match="evidence"):
        launcher.validate_legacy_untracked_failure_archive_v2(value)


@pytest.mark.parametrize(
    "field",
    (
        "prepare_completion_file_sha256",
        "prepare_completion_canonical_sha256",
        "prepare_completion_mtime_ns",
        "old_policy_tree_sha256",
    ),
)
def test_legacy_archive_v2_rejects_prepare_binding_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    launcher, kwargs = _legacy_archive_v2_fixture(tmp_path)
    if field == "prepare_completion_mtime_ns":
        kwargs[field] += 1
    else:
        kwargs[field] = "0" * 64
    with pytest.raises(RuntimeError, match="differs"):
        launcher.archive_legacy_untracked_failure_v2(**kwargs)


def test_legacy_archive_v2_cli_publishes_only_archive_namespace(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launcher, kwargs = _legacy_archive_v2_fixture(tmp_path)
    result = launcher.main(
        [
            launcher.ARCHIVE_LEGACY_FAILURE_V2_MODE,
            "--campaign-root",
            str(kwargs["campaign_root"]),
            "--policy-sha256",
            kwargs["policy_sha256"],
            "--controller-owner-nonce",
            kwargs["controller_owner_nonce"],
            "--observer-session",
            kwargs["observer_session"],
            "--prepare-completion-path",
            str(kwargs["prepare_completion_path"]),
            "--prepare-completion-file-sha256",
            kwargs["prepare_completion_file_sha256"],
            "--prepare-completion-canonical-sha256",
            kwargs["prepare_completion_canonical_sha256"],
            "--prepare-completion-mtime-ns",
            str(kwargs["prepare_completion_mtime_ns"]),
            "--old-policy-tree-sha256",
            kwargs["old_policy_tree_sha256"],
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["created"] is True
    assert output["archive_id"] == Path(
        output["archive"]["path"]
    ).stem
    assert not (
        kwargs["campaign_root"] / "preflight_launch_attempts"
    ).exists()


def _prepare_preflight_launcher_fixture(
    tmp_path: Path,
) -> tuple[Any, Path, Path, Path, str]:
    (tmp_path / "campaign").mkdir(mode=0o755)
    source_root = Path(__file__).parents[1]
    repo_root = tmp_path / "launcher-repo"
    scripts = repo_root / "scripts"
    scripts.mkdir(parents=True)
    launcher_path = scripts / "run_canonical_preflight_launcher.py"
    wrapper_path = scripts / "run_canonical_preflight_wrapper.py"
    controller_path = scripts / "run_canonical_checkpoint_screening.py"
    closeout = repo_root / "src/safa/closeout"
    closeout.mkdir(parents=True)
    verified_loader_path = (
        closeout / "verified_preflight_module_loader.py"
    )
    launch_contract_path = (
        closeout / "preflight_launch_contract.py"
    )
    shutil.copy2(
        source_root / "scripts/run_canonical_preflight_launcher.py",
        launcher_path,
    )
    shutil.copy2(
        source_root
        / "src/safa/closeout/verified_preflight_module_loader.py",
        verified_loader_path,
    )
    shutil.copy2(
        source_root / "src/safa/closeout/preflight_launch_contract.py",
        launch_contract_path,
    )
    wrapper_path.write_text("# bound wrapper fixture\n", encoding="utf-8")
    controller_path.write_text(
        "# bound controller fixture\n", encoding="utf-8"
    )
    fake_wrapper = scripts / "fake_wrapper.py"
    fake_wrapper.write_text(
        """
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    "mode",
    choices=(
        "claim",
        "claim_wrong_policy",
        "claim_wrong_pid",
        "claim_wrong_ppid",
        "claim_wrong_pgid",
        "claim_wrong_start_ticks",
        "claim_wrong_pane",
        "claim_wrong_gate",
        "claim_wrong_argv",
        "claim_wrong_receipt",
        "claim_wrong_started",
        "claim_wrong_executable",
        "claim_replace_receipt_before_claim",
        "claim_malformed",
        "claim_late",
        "claim_hold",
    ),
)
args = parser.parse_args()
receipt_path = Path(os.environ["SAFA_PREFLIGHT_LAUNCH_RECEIPT_PATH"])
receipt_descriptor = os.open(
    receipt_path, os.O_RDONLY | os.O_NOFOLLOW
)
try:
    receipt_stat = os.fstat(receipt_descriptor)
    with os.fdopen(receipt_descriptor, "rb") as receipt_handle:
        receipt_descriptor = -1
        receipt_bytes = receipt_handle.read()
finally:
    if receipt_descriptor >= 0:
        os.close(receipt_descriptor)
receipt = json.loads(receipt_bytes.decode("utf-8"))
receipt_identity = {
    "path": str(receipt_path.resolve()),
    "device": int(receipt_stat.st_dev),
    "inode": int(receipt_stat.st_ino),
    "mode": int(receipt_stat.st_mode),
    "size": int(receipt_stat.st_size),
}
claim_path = Path(receipt["wrapper_claim_path"])
claim_path.parent.mkdir(parents=True, exist_ok=True)
os.chmod(claim_path.parent, 0o755)
command = [
    item.decode("utf-8")
    for item in Path(f"/proc/{os.getpid()}/cmdline")
    .read_bytes()
    .split(b"\\0")
    if item
]
proc_executable_path = Path(
    os.readlink(f"/proc/{os.getpid()}/exe")
).resolve()
proc_executable_stat = proc_executable_path.stat()
log_path = Path(os.environ["SAFA_PREFLIGHT_PANE_LOG_PATH"]).resolve()
log_stat = log_path.stat()
attempt_root = receipt_path.parent
gate_ready_path = attempt_root / "pane_gate_ready.json"
tmux_started_path = attempt_root / "launch_tmux_started.json"
wrapper_started_path = Path(receipt["wrapper_started_path"])
deadline = time.monotonic() + 5
while not wrapper_started_path.is_file():
    if time.monotonic() >= deadline:
        raise SystemExit(92)
    time.sleep(0.005)
def artifact_binding(path, field):
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_sha256": value[field],
    }
def process_identity(pid):
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    fields = raw[raw.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "start_ticks": int(fields[19]),
    }
def digest(value, excluded):
    payload = {k: v for k, v in value.items() if k != excluded}
    return hashlib.sha256(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\\n"
        ).encode()
    ).hexdigest()
tmux_started = json.loads(
    tmux_started_path.read_text(encoding="utf-8")
)
tmux_owner_seal = tmux_started["owner_seal"]
tmux_server_process = process_identity(
    tmux_owner_seal["server_pid"]
)
controller_tmux_server = {
    "server_pid": tmux_owner_seal["server_pid"],
    "server_process": tmux_server_process,
    "socket_path": tmux_owner_seal["socket_path"],
    "socket_device": tmux_owner_seal["socket_device"],
    "socket_inode": tmux_owner_seal["socket_inode"],
}
consumer_artifacts = receipt["pane_fault_consumer"]["artifacts"]
pane_fault_consumer_chain = {
    "consumer_started": artifact_binding(
        Path(consumer_artifacts["started"]),
        "consumer_started_sha256",
    ),
    "consumer_active": artifact_binding(
        Path(consumer_artifacts["active"]),
        "consumer_active_sha256",
    ),
    "consumer_reader_release": artifact_binding(
        Path(consumer_artifacts["reader_release"]),
        "consumer_reader_release_sha256",
    ),
    "consumer_release_observed": artifact_binding(
        Path(consumer_artifacts["release_observed"]),
        "consumer_release_observed_sha256",
    ),
}
claim = {
    "schema_version": 1,
    "contract_type": "safa_canonical_preflight_wrapper_claim_v3",
    "attempt_id": receipt["attempt_id"],
    "preflight_launch_receipt": {
        "path": str(receipt_path.resolve()),
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "canonical_sha256": receipt["launch_receipt_sha256"],
    },
    "preflight_launch_receipt_identity": receipt_identity,
    "pane_gate_ready": artifact_binding(
        gate_ready_path, "pane_gate_ready_sha256"
    ),
    "preflight_launch_tmux_started": artifact_binding(
        tmux_started_path, "launch_tmux_started_sha256"
    ),
    "preflight_wrapper_started": artifact_binding(
        wrapper_started_path,
        "wrapper_started_sha256",
    ),
    "pane_gate_process": json.loads(
        gate_ready_path.read_text(encoding="utf-8")
    )["process"],
    "wrapper_arguments": command,
    "wrapper_executable": {
        "path": str(proc_executable_path),
        "device": int(proc_executable_stat.st_dev),
        "inode": int(proc_executable_stat.st_ino),
        "mode": int(proc_executable_stat.st_mode),
        "size": int(proc_executable_stat.st_size),
    },
    "pane_log": {
        "path": str(log_path),
        "device": int(log_stat.st_dev),
        "inode": int(log_stat.st_ino),
        "mode": int(log_stat.st_mode),
        "size": int(log_stat.st_size),
    },
    "git": receipt["git"],
    "policy_sha256": receipt["policy_sha256"],
    "config": receipt["bindings"]["config"],
    "verified_implementations": receipt["verified_implementations"],
    "checkpoint_plan": artifact_binding(
        wrapper_started_path, "wrapper_started_sha256"
    ),
    "preflight_request_manifest": artifact_binding(
        wrapper_started_path, "wrapper_started_sha256"
    ),
    "controller_session": "safa-screening-preflight-controller",
    "wrapper_pid": os.getpid(),
    "controller_tmux": {
        "pane_pid": tmux_owner_seal["pane_pid"]
    },
    "controller_tmux_server": controller_tmux_server,
    "observer_session": receipt["observer_session"],
    "command": command,
    "observer_command": command,
    "wrapper_process": process_identity(os.getpid()),
    "wrapper_launch_process": process_identity(os.getpid()),
    "started_at": "fixture",
    "external_timeout_seconds": None,
    "pane_fault_consumer_chain": pane_fault_consumer_chain,
}
if args.mode == "claim_wrong_policy":
    claim["policy_sha256"] = "0" * 64
elif args.mode == "claim_wrong_pid":
    claim["wrapper_pid"] += 1
elif args.mode == "claim_wrong_ppid":
    claim["wrapper_launch_process"]["ppid"] += 1
elif args.mode == "claim_wrong_pgid":
    claim["wrapper_launch_process"]["pgid"] += 1
elif args.mode == "claim_wrong_start_ticks":
    claim["wrapper_launch_process"]["start_ticks"] += 1
elif args.mode == "claim_wrong_pane":
    claim["controller_tmux"]["pane_pid"] += 1
elif args.mode == "claim_wrong_gate":
    claim["pane_gate_process"]["start_ticks"] += 1
elif args.mode == "claim_wrong_argv":
    claim["wrapper_arguments"] = [*command, "unexpected"]
elif args.mode == "claim_wrong_receipt":
    claim["preflight_launch_receipt"]["canonical_sha256"] = "0" * 64
elif args.mode == "claim_wrong_started":
    claim["preflight_wrapper_started"]["canonical_sha256"] = "0" * 64
elif args.mode == "claim_wrong_executable":
    claim["wrapper_executable"]["path"] = "/wrong/executable"
elif args.mode == "claim_replace_receipt_before_claim":
    replacement_path = receipt_path.with_name(
        f".replacement-{os.getpid()}.json"
    )
    descriptor = os.open(
        replacement_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        os.write(descriptor, receipt_bytes)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(replacement_path, receipt_path)
if args.mode == "claim_late":
    time.sleep(2)
claim["wrapper_claim_sha256"] = digest(
    claim, "wrapper_claim_sha256"
)
payload = (
    json.dumps(claim, sort_keys=True, indent=2, allow_nan=False)
    + "\\n"
).encode()
if args.mode == "claim_malformed":
    payload = b"{"
descriptor = os.open(
    claim_path,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    0o644,
)
try:
    os.write(descriptor, payload)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
release = Path(os.environ["SAFA_PREFLIGHT_LAUNCH_RELEASE_PATH"])
deadline = time.monotonic() + 10
while not release.is_file():
    if time.monotonic() >= deadline:
        raise SystemExit(91)
    time.sleep(0.02)
if args.mode != "claim_hold":
    wrapper_exit = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_wrapper_exit_v4",
        "policy_sha256": receipt["policy_sha256"],
        "exit_code": 0,
        "controller_exit_code": 0,
        "launch_failure": None,
    }
    wrapper_exit["wrapper_exit_sha256"] = digest(
        wrapper_exit, "wrapper_exit_sha256"
    )
    wrapper_exit_path = claim_path.parent / "wrapper_exit.json"
    wrapper_exit_payload = (
        json.dumps(
            wrapper_exit,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\\n"
    ).encode()
    wrapper_exit_descriptor = os.open(
        wrapper_exit_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        os.write(wrapper_exit_descriptor, wrapper_exit_payload)
        os.fsync(wrapper_exit_descriptor)
    finally:
        os.close(wrapper_exit_descriptor)
if args.mode == "claim_hold":
    time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )
    (repo_root / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n",
        encoding="utf-8",
    )
    config = repo_root / "policy.json"

    def binding(path: Path) -> dict[str, str]:
        return {
            "path": str(path.relative_to(repo_root).as_posix()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    config.write_text(
        json.dumps(
            {
                "implementations": {
                    "preflight_launcher": binding(launcher_path),
                    "preflight_verified_loader": binding(
                        verified_loader_path
                    ),
                    "preflight_launch_contract": binding(
                        launch_contract_path
                    ),
                    "preflight_wrapper": binding(wrapper_path),
                    "controller": binding(controller_path),
                }
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init", "-q", "-b", "master"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "launcher@test.invalid"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Launcher Test"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "update-ref",
            "refs/remotes/origin/master",
            "HEAD",
        ],
        cwd=repo_root,
        check=True,
    )
    spec = importlib.util.spec_from_file_location(
        f"launcher_fixture_{tmp_path.name}", launcher_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy_sha256 = hashlib.sha256(
        str(tmp_path).encode()
    ).hexdigest()
    return module, repo_root, config, fake_wrapper, policy_sha256


def _prepare_verified_preflight_entry_fixture(
    tmp_path: Path,
    entry: str,
) -> tuple[Any, Path, Path, Path, Path]:
    source_root = Path(__file__).parents[1]
    repo_root = tmp_path / f"{entry}-verified-repo"
    scripts = repo_root / "scripts"
    closeout = repo_root / "src/safa/closeout"
    scripts.mkdir(parents=True)
    closeout.mkdir(parents=True)
    entry_relative = {
        "launcher": "scripts/run_canonical_preflight_launcher.py",
        "wrapper": "scripts/run_canonical_preflight_wrapper.py",
    }[entry]
    implementation_name = {
        "launcher": "preflight_launcher",
        "wrapper": "preflight_wrapper",
    }[entry]
    entry_path = repo_root / entry_relative
    loader_path = closeout / "verified_preflight_module_loader.py"
    contract_path = closeout / "preflight_launch_contract.py"
    shutil.copy2(source_root / entry_relative, entry_path)
    shutil.copy2(
        source_root
        / "src/safa/closeout/verified_preflight_module_loader.py",
        loader_path,
    )
    shutil.copy2(
        source_root / "src/safa/closeout/preflight_launch_contract.py",
        contract_path,
    )

    def binding(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(repo_root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    config_path = repo_root / "policy.json"
    config_path.write_text(
        json.dumps(
            {
                "implementations": {
                    implementation_name: binding(entry_path),
                    "preflight_verified_loader": binding(loader_path),
                    "preflight_launch_contract": binding(contract_path),
                }
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        f"verified_{entry}_{tmp_path.name}", entry_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, repo_root, config_path, loader_path, contract_path


@pytest.mark.parametrize("entry", ("launcher", "wrapper"))
def test_preflight_entry_raw_import_executes_no_verified_module(
    entry: str,
) -> None:
    root = Path(__file__).parents[1]
    script = root / {
        "launcher": "scripts/run_canonical_preflight_launcher.py",
        "wrapper": "scripts/run_canonical_preflight_wrapper.py",
    }[entry]
    code = (
        "import importlib.util,json,sys;"
        f"p={str(script)!r};"
        "s=importlib.util.spec_from_file_location('raw_entry',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps(sorted(x for x in ("
        "'safa.closeout.verified_preflight_module_loader',"
        "'safa.closeout.preflight_launch_contract') if x in sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


@pytest.mark.parametrize("entry", ("launcher", "wrapper"))
def test_preflight_entry_ignores_ambient_verified_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    module, _, config, _, _ = (
        _prepare_verified_preflight_entry_fixture(tmp_path, entry)
    )
    ambient_loader = types.ModuleType(
        "safa.closeout.verified_preflight_module_loader"
    )
    ambient_contract = types.ModuleType(
        "safa.closeout.preflight_launch_contract"
    )
    monkeypatch.setitem(
        sys.modules,
        "safa.closeout.verified_preflight_module_loader",
        ambient_loader,
    )
    monkeypatch.setitem(
        sys.modules,
        "safa.closeout.preflight_launch_contract",
        ambient_contract,
    )
    module._install_verified_preflight_apis(config)
    assert module._VERIFIED_LOADER_HANDLE["module"] is not ambient_loader
    assert module._SHARED_CONTRACT_HANDLE["module"] is not ambient_contract
    assert (
        sys.modules["safa.closeout.verified_preflight_module_loader"]
        is ambient_loader
    )
    assert (
        sys.modules["safa.closeout.preflight_launch_contract"]
        is ambient_contract
    )


@pytest.mark.parametrize("entry", ("launcher", "wrapper"))
@pytest.mark.parametrize(
    ("target", "fault"),
    (
        ("loader", "wrong_path"),
        ("loader", "wrong_sha"),
        ("loader", "symlink"),
        ("loader", "same_content_inode_replacement"),
        ("loader", "post_load_tamper"),
        ("loader", "api_missing"),
        ("loader", "execution_failure"),
        ("contract", "wrong_path"),
        ("contract", "wrong_sha"),
        ("contract", "symlink"),
        ("contract", "same_content_inode_replacement"),
        ("contract", "post_load_tamper"),
        ("contract", "api_missing"),
        ("contract", "execution_failure"),
    ),
)
def test_preflight_entry_verified_module_fault_matrix(
    tmp_path: Path,
    entry: str,
    target: str,
    fault: str,
) -> None:
    module, repo_root, config_path, loader_path, contract_path = (
        _prepare_verified_preflight_entry_fixture(tmp_path, entry)
    )
    target_path = (
        loader_path if target == "loader" else contract_path
    )
    implementation = (
        "preflight_verified_loader"
        if target == "loader"
        else "preflight_launch_contract"
    )
    if fault in {
        "same_content_inode_replacement",
        "post_load_tamper",
    }:
        module._install_verified_preflight_apis(config_path)
        if fault == "same_content_inode_replacement":
            replacement = target_path.with_name(
                f"{target_path.name}.replacement"
            )
            replacement.write_bytes(target_path.read_bytes())
            os.replace(replacement, target_path)
        else:
            target_path.write_bytes(
                target_path.read_bytes() + b"\n# post-load tamper\n"
            )
        with pytest.raises(RuntimeError):
            module._reverify_verified_preflight_apis()
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    binding = config["implementations"][implementation]
    if fault == "wrong_path":
        binding["path"] = "scripts/induced_target.py"
    elif fault == "wrong_sha":
        binding["sha256"] = "0" * 64
    elif fault == "symlink":
        replacement = repo_root / f"{target}.real.py"
        replacement.write_bytes(target_path.read_bytes())
        target_path.unlink()
        target_path.symlink_to(replacement)
    elif fault == "api_missing":
        target_path.write_text(
            "UNRELATED_API = object()\n", encoding="utf-8"
        )
        binding["sha256"] = hashlib.sha256(
            target_path.read_bytes()
        ).hexdigest()
    else:
        target_path.write_text(
            "raise RuntimeError('induced execution failure')\n",
            encoding="utf-8",
        )
        binding["sha256"] = hashlib.sha256(
            target_path.read_bytes()
        ).hexdigest()
    config_path.write_text(
        json.dumps(config, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        module._install_verified_preflight_apis(config_path)


def test_preflight_launcher_bootstrap_failure_is_minimal_and_pre_tmux(
    tmp_path: Path,
) -> None:
    launcher, repo_root, config_path, _, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["implementations"]["preflight_verified_loader"][
        "sha256"
    ] = "0" * 64
    config_path.write_text(
        json.dumps(config, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    attempt_id = hashlib.sha256(b"bootstrap-failure").hexdigest()
    campaign_root = tmp_path / "campaign"
    with pytest.raises(RuntimeError):
        launcher.launch_preflight(
            repo_root=repo_root,
            config=config_path,
            campaign_root=campaign_root,
            policy_sha256=policy_sha256,
            python=sys.executable,
            startup_timeout_seconds=1,
            attempt_id=attempt_id,
            owner_nonce=hashlib.sha256(b"owner").hexdigest(),
            observer_suffix=hashlib.sha256(b"observer").hexdigest(),
        )
    setup_path = (
        campaign_root
        / "preflight_launch_attempts/setup_terminals"
        / f"{attempt_id}.json"
    )
    setup = load_json(setup_path, "launcher setup terminal")
    assert set(setup) == {
        "schema_version",
        "contract_type",
        "attempt_id",
        "policy_sha256",
        "started_registry",
        "stage",
        "failure",
        "tmux_execution_count",
        "scientific_execution_started",
        "started_at",
        "completed_at",
        "launch_setup_terminal_sha256",
    }
    assert setup["contract_type"] == (
        "safa_canonical_preflight_launch_setup_terminal_v1"
    )
    assert setup["stage"] == "verified_implementation_bootstrap"
    assert setup["tmux_execution_count"] == 0
    assert setup["scientific_execution_started"] is False
    assert setup["contract_type"] not in shared_contract_types()
    with pytest.raises(PreflightLaunchContractError):
        validate_preflight_claim_v3(
            setup,
            verified_implementations=(
                _test_verified_preflight_implementations()
            ),
            gate_ready={},
            wrapper_started={},
        )
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts/by_policy"
        / policy_sha256
        / attempt_id
    )
    for name in (
        "launch_receipt.json",
        "pane.log",
        "pane_gate_ready.json",
        "launch_tmux_started.json",
        "wrapper_started.json",
        "launch_accepted.json",
        "launch_terminal.json",
        "launch_ownership_release.json",
    ):
        assert not (attempt_root / name).exists()
    assert not (
        campaign_root
        / "by_policy"
        / policy_sha256
        / "preflight_control/wrapper_claim.json"
    ).exists()


def test_preflight_wrapper_bootstrap_failure_precedes_claim_and_process(
    tmp_path: Path,
) -> None:
    wrapper, repo_root, config_path, _, _ = (
        _prepare_verified_preflight_entry_fixture(tmp_path, "wrapper")
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["implementations"]["preflight_launch_contract"][
        "sha256"
    ] = "0" * 64
    config_path.write_text(
        json.dumps(config, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    policy_root = tmp_path / "policy"
    emergency_state: dict[str, Any] = {}
    with pytest.raises(RuntimeError):
        wrapper._run_wrapped_controller_owned(
            repo_root=repo_root,
            policy_root=policy_root,
            policy_sha256=hashlib.sha256(b"policy").hexdigest(),
            config=config_path,
            command=[sys.executable, "-c", "raise SystemExit(93)"],
            observer_command=[
                sys.executable,
                "-c",
                "raise SystemExit(94)",
            ],
            emergency_state=emergency_state,
        )
    assert emergency_state == {}
    control = policy_root / "preflight_control"
    for name in (
        "wrapper_claim.json",
        "controller_process_start.json",
        "controller_process_exit.json",
        "observer_launch.json",
        "observer_bootstrap.json",
        "observer_gate_ready.json",
    ):
        assert not (control / name).exists()


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    ("target", "field", "replacement"),
    (
        ("live", "pane_dead", True),
        ("live", "pane_dead_status", 0),
        ("live_process", "start_ticks", 0),
        ("dead", "session", "foreign"),
        ("dead", "pane", "%99"),
        ("dead", "pane_pid", 99),
        ("dead", "owner_nonce", "f" * 64),
        ("dead", "tmux_server", {}),
        ("dead", "pane_dead", False),
        ("dead", "pane_process", {"pid": 41}),
    ),
)
def test_pane_owner_lifecycle_transition_mutations_fail_closed(
    target: str,
    field: str,
    replacement: Any,
) -> None:
    launcher = _launcher_module()
    process = {
        "pid": 41,
        "ppid": 7,
        "pgid": 41,
        "sid": 41,
        "start_ticks": 123,
    }
    server = {
        "server_pid": 7,
        "server_process": {
            "pid": 7,
            "ppid": 1,
            "pgid": 7,
            "sid": 7,
            "start_ticks": 77,
        },
        "socket_path": "/tmp/tmux-owner",
        "socket_device": 1,
        "socket_inode": 2,
    }
    live = {
        "session": "consumer",
        "pane": "%1",
        "pane_pid": 41,
        "pane_dead": False,
        "pane_dead_status": None,
        "pane_process": process,
        "owner_nonce": "a" * 64,
        "tmux_server": server,
    }
    dead = json.loads(json.dumps(live))
    dead["pane_dead"] = True
    dead["pane_dead_status"] = 0
    dead["pane_process"] = None
    assert (
        launcher._validate_pane_owner_lifecycle_transition(
            live, dead, label="fixture"
        )
        == dead
    )
    mutated_live = json.loads(json.dumps(live))
    mutated_dead = json.loads(json.dumps(dead))
    if target == "live":
        mutated_live[field] = replacement
    elif target == "live_process":
        mutated_live["pane_process"][field] = replacement
    else:
        mutated_dead[field] = replacement
    with pytest.raises(
        RuntimeError,
        match="lifecycle transition differs",
    ):
        launcher._validate_pane_owner_lifecycle_transition(
            mutated_live, mutated_dead, label="fixture"
        )


def test_pane_owner_dead_status_is_not_lifecycle_verdict(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    process = {
        "pid": 41,
        "ppid": 7,
        "pgid": 41,
        "sid": 41,
        "start_ticks": 123,
    }
    server = {
        "server_pid": 7,
        "server_process": {
            "pid": 7,
            "ppid": 1,
            "pgid": 7,
            "sid": 7,
            "start_ticks": 77,
        },
        "socket_path": "/tmp/tmux-owner",
        "socket_device": 1,
        "socket_inode": 2,
    }
    live = {
        "session": "consumer",
        "pane": "%1",
        "pane_pid": 41,
        "pane_dead": False,
        "pane_dead_status": None,
        "pane_process": process,
        "owner_nonce": "a" * 64,
        "tmux_server": server,
    }
    dead_without_tmux_status = {
        **live,
        "pane_dead": True,
        "pane_dead_status": None,
        "pane_process": None,
    }
    assert (
        launcher._validate_pane_owner_lifecycle_transition(
            live,
            dead_without_tmux_status,
            label="fixture without tmux exit status",
        )
        == dead_without_tmux_status
    )
    path, binding, descriptor, directory_descriptor = (
        _open_lifecycle_test_writer(
            launcher,
            tmp_path,
            "pane_owner_formal_lifecycle.channel",
        )
    )
    record = _lifecycle_wait_record(
        launcher,
        binding,
        role="gate",
        wait_status=117,
    )
    launcher._write_lifecycle_wait_status(
        descriptor,
        directory_descriptor,
        binding,
        record,
        role="gate",
    )
    os.close(descriptor)
    os.close(directory_descriptor)
    reader, reader_directory = (
        launcher._open_lifecycle_wait_channel_reader(
            tmp_path,
            binding,
            name=path.name,
        )
    )
    try:
        snapshot = _read_lifecycle_test_status(
            launcher,
            reader,
            reader_directory,
            binding,
            record,
            role="gate",
        )
        assert snapshot["record"]["waitid_si_status"] == 117
        assert snapshot["record"]["wait_status_raw"] == (117 << 8)
        assert snapshot["record"]["terminal"] is not None
    finally:
        os.close(reader)
        os.close(reader_directory)


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_preflight_launcher_claim_gate_is_evidence_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher, repo_root, config, fake_wrapper, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    campaign_root = tmp_path / "campaign"
    attempt_id = hashlib.sha256(b"launcher-success").hexdigest()
    owner_nonce = hashlib.sha256(b"launcher-owner").hexdigest()
    try:
        result = launcher.launch_preflight(
            repo_root=repo_root,
            config=config,
            campaign_root=campaign_root,
            policy_sha256=policy_sha256,
            python=sys.executable,
            startup_timeout_seconds=10,
            attempt_id=attempt_id,
            owner_nonce=owner_nonce,
            observer_suffix=hashlib.sha256(
                b"launcher-observer"
            ).hexdigest(),
            wrapper_arguments_override=[
                sys.executable,
                "-B",
                "-u",
                str(fake_wrapper),
                "claim",
            ],
        )
        assert result["contract_type"] == (
            "safa_canonical_preflight_launch_ownership_release_v1"
        )
        attempt_root = (
            campaign_root
            / "preflight_launch_attempts/by_policy"
            / policy_sha256
            / attempt_id
        )
        receipt = load_json(
            attempt_root / "launch_receipt.json", "launch receipt"
        )
        assert receipt["schema_version"] == 4
        assert receipt["contract_type"] == (
            "safa_canonical_preflight_launch_receipt_v4"
        )
        supervisor_ready = load_json(
            Path(
                receipt[
                    "gate_lifecycle_wait_supervisor_ready_path"
                ]
            ),
            "gate wait supervisor ready",
        )
        assert (
            launcher.validate_launch_receipt_schema(
                receipt,
                expected_gate_worker_arguments=supervisor_ready[
                    "gate_worker_arguments"
                ],
                expected_consumer_worker_arguments=receipt[
                    "consumer_worker_arguments"
                ],
            )
            == receipt
        )
        assert receipt["shell"] is False
        assert isinstance(receipt["tmux_arguments"], list)
        assert receipt["pane_log"]["inode"] > 0
        started = load_json(
            attempt_root / "launch_tmux_started.json",
            "launch tmux started",
        )
        assert started["remain_on_exit"] == "on"
        assert started["owner_seal"]["owner_nonce"] == owner_nonce
        gate_ready = load_json(
            attempt_root / "pane_gate_ready.json",
            "pane gate ready",
        )
        wrapper_started = load_json(
            attempt_root / "wrapper_started.json",
            "wrapper started",
        )
        assert (
            started["owner_seal"]["pane_process"]
            == supervisor_ready["supervisor_process"]
        )
        assert (
            supervisor_ready["gate_worker_process"]
            == gate_ready["process"]
        )
        assert (
            gate_ready["process"]["ppid"]
            == supervisor_ready["supervisor_process"]["pid"]
        )
        assert (
            wrapper_started["wrapper_process"]["ppid"]
            == gate_ready["process"]["pid"]
        )
        assert len(
            {
                supervisor_ready["supervisor_process"]["pid"],
                gate_ready["process"]["pid"],
                wrapper_started["wrapper_process"]["pid"],
            }
        ) == 3
        accepted = load_json(
            attempt_root / "launch_accepted.json",
            "launch accepted",
        )
        assert accepted["startup_window_closed"] is False
        terminal = load_json(
            attempt_root / "launch_terminal.json",
            "launch terminal",
        )
        assert terminal["status"] == "ownership_transferred"
        assert terminal["failure"] is None
        assert result["startup_window_closed"] is True
        consumer_artifacts = receipt["pane_fault_consumer"][
            "artifacts"
        ]
        consumer_chain = {
            "consumer_started": launcher._json_binding(
                Path(consumer_artifacts["started"]),
                "consumer_started_sha256",
            ),
            "consumer_active": launcher._json_binding(
                Path(consumer_artifacts["active"]),
                "consumer_active_sha256",
            ),
            "consumer_reader_release": launcher._json_binding(
                Path(consumer_artifacts["reader_release"]),
                "consumer_reader_release_sha256",
            ),
            "consumer_release_observed": launcher._json_binding(
                Path(consumer_artifacts["release_observed"]),
                "consumer_release_observed_sha256",
            ),
        }
        claim = load_json(
            Path(receipt["wrapper_claim_path"]),
            "launch wrapper claim",
        )
        assert claim["pane_fault_consumer_chain"] == consumer_chain
        assert accepted["pane_fault_consumer_chain"] == consumer_chain
        assert terminal["pane_fault_consumer_chain"] == consumer_chain
        assert result["pane_fault_consumer_chain"] == consumer_chain
        consumer_attempt = load_json(
            Path(consumer_artifacts["attempt"]),
            "pane fault consumer attempt",
        )
        consumer_session = consumer_attempt["consumer_session"]
        assert (
            subprocess.run(
                ["tmux", "has-session", "-t", consumer_session],
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        execution_path = attempt_root / "gate_execution_terminal.json"
        deadline = time.monotonic() + 5
        while not execution_path.is_file():
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "accepted wrapper execution terminal timed out"
                )
            time.sleep(0.01)
        execution = load_json(
            execution_path, "accepted wrapper execution terminal"
        )
        assert execution["exit_kind"] == "exit"
        assert execution["exit_code"] == 0
        assert execution["signal_number"] is None
        assert execution["launch_terminal"] == launcher._json_binding(
            attempt_root / "launch_terminal.json",
            "launch_terminal_sha256",
        )
        assert execution["launch_accepted"] == launcher._json_binding(
            attempt_root / "launch_accepted.json",
            "launch_accepted_sha256",
        )
        assert execution["launch_ownership_release"] == launcher._json_binding(
            attempt_root / "launch_ownership_release.json",
            "launch_ownership_release_sha256",
        )
        supervisor_process = supervisor_ready[
            "supervisor_process"
        ]
        deadline = time.monotonic() + 5
        while True:
            try:
                live_supervisor = launcher._process_identity(
                    supervisor_process["pid"]
                )
            except (OSError, RuntimeError):
                live_supervisor = None
            if live_supervisor != supervisor_process:
                break
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "gate wait supervisor did not retire"
                )
            time.sleep(0.01)
        original_lifecycle_reader = (
            launcher._open_lifecycle_wait_channel_reader
        )
        formal_reader_access_modes: list[int] = []

        def record_formal_reader_access_mode(*args, **kwargs):
            descriptor, directory_descriptor = (
                original_lifecycle_reader(*args, **kwargs)
            )
            formal_reader_access_modes.append(
                fcntl.fcntl(descriptor, fcntl.F_GETFL)
                & os.O_ACCMODE
            )
            return descriptor, directory_descriptor

        monkeypatch.setattr(
            launcher,
            "_open_lifecycle_wait_channel_reader",
            record_formal_reader_access_mode,
        )
        formal_gate = launcher._read_formal_gate_lifecycle_status(
            attempt_root=attempt_root,
            pane={
                "pane_pid": supervisor_process["pid"],
                "pane_dead": True,
                "pane_dead_status": None,
            },
        )
        wait_record = formal_gate["snapshot"]["record"]
        assert wait_record["returncode"] == 117
        assert wait_record["terminal"]["binding"] == (
            launcher._json_binding(
                execution_path,
                "gate_execution_terminal_sha256",
            )
        )
        assert (
            formal_gate["snapshot"]["channel_authority"]["inode"]
            == receipt["gate_lifecycle_wait_channel"]["inode"]
        )
        assert (
            formal_gate["adjudication"]["adjudicated_outcome"]
            == "completed"
        )
        assert formal_reader_access_modes == [os.O_RDONLY]
        assert formal_gate["ownership_chain_state"] == "bound"
        failed_after_claim = json.loads(json.dumps(execution))
        failed_after_claim["returncode"] = 1
        failed_after_claim["exit_code"] = 1
        failed_after_claim["wrapper_outcome"] = {
            "status": "wrapper_child_failed",
            "exit_code": 1,
            "failure": None,
            "wrapper_exit": None,
        }
        failed_adjudication = (
            launcher._adjudicate_gate_execution_outcome(
                failed_after_claim,
                wrapper_exit_path=attempt_root / "wrapper_exit.json",
                policy_sha256=policy_sha256,
            )
        )
        failed_chain_state, _ = (
            launcher._validate_gate_execution_ownership_chain(
                receipt_path=attempt_root / "launch_receipt.json",
                gate_execution=failed_after_claim,
            )
        )
        assert failed_adjudication["adjudicated_outcome"] == (
            "controller_failed"
        )
        assert failed_chain_state == "bound"
        consumer_terminal_path = Path(
            consumer_artifacts["terminal"]
        )
        deadline = time.monotonic() + 5
        while not consumer_terminal_path.is_file():
            if time.monotonic() >= deadline:
                gate_pane = launcher._tmux_pane(
                    launcher.CONTROLLER_SESSION
                )
                consumer_pane = launcher._tmux_pane(
                    consumer_session
                )
                raise AssertionError(
                    "pane fault consumer terminal timed out: "
                    f"gate={gate_pane}, "
                    f"consumer={consumer_pane}, "
                    "self_size="
                    f"{Path(consumer_artifacts['self_fault_channel']).stat().st_size}, "
                    f"gate_terminal={execution}"
                )
            time.sleep(0.01)
        consumer_terminal = load_json(
            consumer_terminal_path,
            "pane fault consumer terminal",
        )
        assert consumer_terminal["status"] == "completed"
        assert consumer_terminal["exit_code"] == 0
        original_write = launcher._write_exclusive
        cleanup_failure_injected = False

        def interrupt_before_cleanup(path, value):
            nonlocal cleanup_failure_injected
            if (
                not cleanup_failure_injected
                and Path(path)
                == Path(consumer_artifacts["cleanup"])
            ):
                cleanup_failure_injected = True
                raise RuntimeError(
                    "fixture interruption before cleanup publish"
                )
            return original_write(path, value)

        monkeypatch.setattr(
            launcher, "_write_exclusive", interrupt_before_cleanup
        )
        with pytest.raises(
            RuntimeError,
            match="fixture interruption before cleanup publish",
        ):
            launcher.join_pane_fault_consumer(
                attempt_path=Path(consumer_artifacts["attempt"]),
                config=config,
                timeout_seconds=5,
            )
        assert Path(consumer_artifacts["join"]).is_file()
        assert not Path(consumer_artifacts["cleanup"]).exists()
        assert (
            subprocess.run(
                ["tmux", "has-session", "-t", consumer_session],
                capture_output=True,
                text=True,
            ).returncode
            != 0
        )
        foreign_nonce = hashlib.sha256(
            b"foreign-consumer-session"
        ).hexdigest()
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                consumer_session,
                "-c",
                str(repo_root),
                "-e",
                f"{launcher.TMUX_OWNER_ENV}={foreign_nonce}",
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            ],
            check=True,
        )
        foreign_pane = launcher._tmux_pane(consumer_session)
        assert foreign_pane is not None
        foreign_process = launcher._process_identity(
            foreign_pane["pane_pid"]
        )
        monkeypatch.setattr(
            launcher, "_write_exclusive", original_write
        )
        with pytest.raises(
            RuntimeError,
            match="join found foreign session",
        ):
            launcher.join_pane_fault_consumer(
                attempt_path=Path(consumer_artifacts["attempt"]),
                config=config,
                timeout_seconds=5,
            )
        assert (
            subprocess.run(
                ["tmux", "has-session", "-t", consumer_session],
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        assert launcher._tmux_pane(consumer_session) == foreign_pane
        assert (
            launcher._process_identity(foreign_pane["pane_pid"])
            == foreign_process
        )
        subprocess.run(
            ["tmux", "kill-session", "-t", consumer_session],
            check=True,
        )
        consumer_cleanup = launcher.join_pane_fault_consumer(
            attempt_path=Path(consumer_artifacts["attempt"]),
            config=config,
            timeout_seconds=5,
        )
        assert consumer_cleanup["status"] == "cleaned"
        assert consumer_cleanup["session_residual"] is False
        assert Path(consumer_artifacts["join"]).is_file()
        assert Path(consumer_artifacts["cleanup"]).is_file()
        join_before = Path(consumer_artifacts["join"]).read_bytes()
        cleanup_before = Path(
            consumer_artifacts["cleanup"]
        ).read_bytes()
        join_stat_before = Path(
            consumer_artifacts["join"]
        ).stat()
        cleanup_stat_before = Path(
            consumer_artifacts["cleanup"]
        ).stat()
        assert (
            launcher.join_pane_fault_consumer(
                attempt_path=Path(consumer_artifacts["attempt"]),
                config=config,
                timeout_seconds=5,
            )
            == consumer_cleanup
        )
        join_stat_after = Path(consumer_artifacts["join"]).stat()
        cleanup_stat_after = Path(
            consumer_artifacts["cleanup"]
        ).stat()
        assert (
            Path(consumer_artifacts["join"]).read_bytes()
            == join_before
        )
        assert (
            Path(consumer_artifacts["cleanup"]).read_bytes()
            == cleanup_before
        )
        assert (
            join_stat_after.st_ino,
            join_stat_after.st_mtime_ns,
        ) == (
            join_stat_before.st_ino,
            join_stat_before.st_mtime_ns,
        )
        assert (
            cleanup_stat_after.st_ino,
            cleanup_stat_after.st_mtime_ns,
        ) == (
            cleanup_stat_before.st_ino,
            cleanup_stat_before.st_mtime_ns,
        )
        artifact_mutations = (
            "join_extra",
            "join_digest",
            "join_owner",
            "join_pane",
            "join_status",
            "join_exit",
            "cleanup_extra",
            "cleanup_digest",
            "cleanup_owner",
            "cleanup_status",
            "cleanup_join",
            "cleanup_residual",
            "terminal_digest",
            "terminal_status",
            "terminal_exit",
            "terminal_owner",
            "gate_digest",
            "gate_wrapper_outcome",
            "controller_cleanup_digest",
            "controller_cleanup_status",
            "controller_cleanup_exit",
        )
        mutation_paths = {
            "join": Path(consumer_artifacts["join"]),
            "cleanup": Path(consumer_artifacts["cleanup"]),
            "terminal": consumer_terminal_path,
            "gate": execution_path,
            "controller_cleanup": Path(
                consumer_artifacts["controller_cleanup"]
            ),
        }
        digest_fields = {
            "join": "consumer_join_sha256",
            "cleanup": "consumer_cleanup_sha256",
            "terminal": "consumer_terminal_sha256",
            "gate": "gate_execution_terminal_sha256",
            "controller_cleanup": (
                "consumer_controller_cleanup_sha256"
            ),
        }
        for mutation in artifact_mutations:
            target = next(
                name
                for name in mutation_paths
                if mutation.startswith(f"{name}_")
            )
            path = mutation_paths[target]
            original = path.read_bytes()
            value = load_json(path, f"{mutation} source")
            action = mutation.removeprefix(f"{target}_")
            if action == "extra":
                value["unexpected"] = True
            elif action == "digest":
                value[digest_fields[target]] = "0" * 64
            elif action == "owner":
                if target == "join":
                    value["consumer_owner_nonce"] = "f" * 64
                elif target == "cleanup":
                    value["controller_owner_seal"] = {}
                else:
                    value["supervisor_owner_seal"]["pane"] = (
                        "%999999"
                    )
            elif action == "pane":
                value["retired_pane"]["pane"] = "%999999"
            elif action == "status":
                value["status"] = "mutated"
            elif action == "exit":
                if target == "controller_cleanup":
                    value["controller_exit_code"] = 2
                elif target == "join":
                    value["consumer_adjudicated_exit"] = 2
                else:
                    value["exit_code"] = 2
            elif action == "join":
                value["consumer_join"]["canonical_sha256"] = (
                    "0" * 64
                )
            elif action == "residual":
                value["session_residual"] = True
            elif action == "wrapper_outcome":
                value["wrapper_outcome"]["status"] = (
                    "wrapper_child_failed"
                )
            else:
                raise AssertionError(
                    f"unhandled artifact mutation: {mutation}"
                )
            if action != "digest":
                value[digest_fields[target]] = (
                    launcher._canonical_digest(
                        value, digest_fields[target]
                    )
                )
            path.write_bytes(canonical_json(value))
            try:
                with pytest.raises(RuntimeError):
                    launcher.join_pane_fault_consumer(
                        attempt_path=Path(
                            consumer_artifacts["attempt"]
                        ),
                        config=config,
                        timeout_seconds=0,
                    )
            finally:
                path.write_bytes(original)
        lifecycle_path = Path(
            consumer_attempt["consumer_lifecycle_wait_channel"][
                "path"
            ]
        )
        lifecycle_original = lifecycle_path.read_bytes()
        joined = load_json(
            Path(consumer_artifacts["join"]),
            "consumer join lifecycle mutation source",
        )
        lifecycle_record = joined["consumer_lifecycle"]["record"]
        lifecycle_mutations = (
            "signal_without_terminal",
            "exit_0_with_terminal",
            "exit_117_with_terminal",
            "exit_118_without_terminal",
            "expected_binding_drift",
        )
        for mutation in lifecycle_mutations:
            record = copy.deepcopy(lifecycle_record)
            if mutation == "signal_without_terminal":
                record.update(
                    {
                        "waitid_si_code": os.CLD_KILLED,
                        "waitid_si_status": int(signal.SIGTERM),
                        "wait_status_raw": int(signal.SIGTERM),
                        "wait_code": "killed",
                        "returncode": -int(signal.SIGTERM),
                        "exit_kind": "signal",
                        "exit_code": None,
                        "signal_number": int(signal.SIGTERM),
                        "core_dumped": False,
                        "terminal": None,
                    }
                )
            elif mutation in {
                "exit_0_with_terminal",
                "exit_117_with_terminal",
            }:
                exit_code = (
                    0 if mutation == "exit_0_with_terminal" else 117
                )
                record.update(
                    {
                        "waitid_si_status": exit_code,
                        "wait_status_raw": exit_code << 8,
                        "returncode": exit_code,
                        "exit_code": exit_code,
                    }
                )
            elif mutation == "exit_118_without_terminal":
                record["terminal"] = None
            elif mutation == "expected_binding_drift":
                record["child_command"] = [
                    *record["child_command"],
                    "--mutated",
                ]
            else:
                raise AssertionError(
                    f"unhandled lifecycle mutation: {mutation}"
                )
            record["lifecycle_wait_status_sha256"] = (
                launcher._canonical_digest(
                    record, "lifecycle_wait_status_sha256"
                )
            )
            lifecycle_path.write_bytes(
                launcher._build_lifecycle_wait_channel_frame(record)[0]
            )
            try:
                with pytest.raises(RuntimeError):
                    launcher.join_pane_fault_consumer(
                        attempt_path=Path(
                            consumer_artifacts["attempt"]
                        ),
                        config=config,
                        timeout_seconds=0,
                    )
            finally:
                lifecycle_path.write_bytes(lifecycle_original)
        for invalid_frame in (
            b"",
            launcher.LIFECYCLE_WAIT_CHANNEL_PREFIX + b"00000010\n{",
        ):
            lifecycle_path.write_bytes(invalid_frame)
            try:
                with pytest.raises(RuntimeError):
                    launcher.join_pane_fault_consumer(
                        attempt_path=Path(
                            consumer_artifacts["attempt"]
                        ),
                        config=config,
                        timeout_seconds=0,
                    )
            finally:
                lifecycle_path.write_bytes(lifecycle_original)
        missing_terminal = consumer_terminal_path.with_name(
            "consumer_terminal.missing"
        )
        consumer_terminal_path.rename(missing_terminal)
        try:
            with pytest.raises(RuntimeError):
                launcher.join_pane_fault_consumer(
                    attempt_path=Path(
                        consumer_artifacts["attempt"]
                    ),
                    config=config,
                    timeout_seconds=0,
                )
        finally:
            missing_terminal.rename(consumer_terminal_path)
        self_channel_path = Path(
            consumer_attempt["consumer_self_fault_channel"][
                "path"
            ]
        )
        assert self_channel_path.read_bytes() == b""
        self_channel_path.write_bytes(b"{")
        try:
            with pytest.raises(RuntimeError):
                launcher.join_pane_fault_consumer(
                    attempt_path=Path(
                        consumer_artifacts["attempt"]
                    ),
                    config=config,
                    timeout_seconds=0,
                )
        finally:
            self_channel_path.write_bytes(b"")
        self_descriptor = (
            launcher._open_presealed_fault_channel(
                self_channel_path.parent,
                consumer_attempt["consumer_self_fault_channel"],
                name=self_channel_path.name,
            )
        )
        try:
            failure = launcher.LauncherExclusivePublishError(
                "precommit_failed_clean",
                "fixture join self-channel fault",
                stage="fixture_join_self_channel",
                directory_seal={},
                payload={},
                temporary=None,
                error_number=None,
                quarantined=False,
            )
            launcher._write_launcher_fault_channel_record(
                self_descriptor,
                consumer_attempt["consumer_self_fault_channel"],
                attempt_id=consumer_attempt["attempt_id"],
                owner_nonce=consumer_attempt[
                    "consumer_owner_nonce"
                ],
                launch_receipt_sha256=consumer_attempt[
                    "launch_receipt"
                ]["canonical_sha256"],
                publisher=consumer_attempt[
                    "consumer_self_fault_publisher"
                ],
                failure=failure,
            )
        finally:
            os.close(self_descriptor)
        try:
            with pytest.raises(
                RuntimeError,
                match=(
                    "presealed fault channel identity differs"
                    "|self channel is not empty"
                ),
            ):
                launcher.join_pane_fault_consumer(
                    attempt_path=Path(
                        consumer_artifacts["attempt"]
                    ),
                    config=config,
                    timeout_seconds=0,
                )
        finally:
            self_channel_path.write_bytes(b"")
        assert (
            subprocess.run(
                ["tmux", "has-session", "-t", consumer_session],
                capture_output=True,
                text=True,
            ).returncode
            != 0
        )
    finally:
        subprocess.run(
            [
                "tmux",
                "kill-session",
                "-t",
                launcher.CONTROLLER_SESSION,
            ],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "tmux",
                "kill-session",
                "-t",
                (
                    launcher.PANE_FAULT_CONSUMER_SESSION_PREFIX
                    + attempt_id
                ),
            ],
            capture_output=True,
            text=True,
        )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    "fault_mode",
    (
        "gate_valid",
        "gate_partial",
        "self_valid",
        "self_partial",
        "consumer_death",
    ),
)
def test_preflight_launcher_post_handoff_consumer_faults_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_mode: str,
) -> None:
    launcher, repo_root, config, fake_wrapper, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    campaign_root = tmp_path / "campaign"
    attempt_id = hashlib.sha256(
        f"launcher-consumer-fault:{fault_mode}".encode()
    ).hexdigest()
    original_transfer = launcher._transfer_pane_fault_consumer

    def inject_after_transfer(**kwargs):
        result = original_transfer(**kwargs)
        consumer = kwargs["consumer"]
        attempt = consumer["attempt"]
        if fault_mode == "consumer_death":
            launcher._cleanup_failed_pane_fault_consumer(
                consumer["session"], consumer["owner_nonce"]
            )
            return result
        channel_name = (
            "pane_fault_channel"
            if fault_mode.startswith("gate_")
            else "consumer_self_fault_channel"
        )
        binding = attempt[channel_name]
        descriptor = launcher._open_presealed_fault_channel(
            Path(binding["path"]).parent,
            binding,
            name=Path(binding["path"]).name,
        )
        try:
            if fault_mode.endswith("_partial"):
                assert os.pwrite(descriptor, b"{", 0) == 1
                os.fsync(descriptor)
            else:
                publisher = (
                    attempt["pane_fault_publisher"]
                    if fault_mode == "gate_valid"
                    else attempt["consumer_self_fault_publisher"]
                )
                owner = (
                    attempt["gate_owner_seal"]["owner_nonce"]
                    if fault_mode == "gate_valid"
                    else attempt["consumer_owner_nonce"]
                )
                failure = launcher.LauncherExclusivePublishError(
                    "precommit_failed_clean",
                    f"injected {fault_mode}",
                    stage=f"fixture_{fault_mode}",
                    directory_seal={},
                    payload={"fault_mode": fault_mode},
                    temporary=None,
                    error_number=None,
                    quarantined=False,
                )
                launcher._write_launcher_fault_channel_record(
                    descriptor,
                    binding,
                    attempt_id=attempt["attempt_id"],
                    owner_nonce=owner,
                    launch_receipt_sha256=attempt[
                        "launch_receipt"
                    ]["canonical_sha256"],
                    publisher=publisher,
                    failure=failure,
                )
        finally:
            os.close(descriptor)
        if fault_mode.startswith("gate_"):
            deadline = time.monotonic() + 2
            self_path = Path(
                attempt["consumer_self_fault_channel"]["path"]
            )
            while self_path.stat().st_size == 0:
                if time.monotonic() >= deadline:
                    raise AssertionError(
                        "consumer did not relay gate fault"
                    )
                time.sleep(0.01)
        return result

    monkeypatch.setattr(
        launcher,
        "_transfer_pane_fault_consumer",
        inject_after_transfer,
    )
    consumer_session = (
        launcher.PANE_FAULT_CONSUMER_SESSION_PREFIX + attempt_id
    )
    try:
        if fault_mode == "consumer_death":
            failed_result = launcher.launch_preflight(
                repo_root=repo_root,
                config=config,
                campaign_root=campaign_root,
                policy_sha256=policy_sha256,
                python=sys.executable,
                startup_timeout_seconds=10,
                attempt_id=attempt_id,
                owner_nonce=hashlib.sha256(
                    f"launcher-owner:{fault_mode}".encode()
                ).hexdigest(),
                observer_suffix=hashlib.sha256(
                    f"launcher-observer:{fault_mode}".encode()
                ).hexdigest(),
                wrapper_arguments_override=[
                    sys.executable,
                    "-B",
                    "-u",
                    str(fake_wrapper),
                    "claim",
                ],
            )
            assert failed_result["status"] == "launcher_failed"
            assert failed_result["failure"]["type"] == (
                "PaneFaultConsumerReservationError"
            )
        else:
            with pytest.raises(
                launcher.PaneFaultConsumerReservationError
            ):
                launcher.launch_preflight(
                    repo_root=repo_root,
                    config=config,
                    campaign_root=campaign_root,
                    policy_sha256=policy_sha256,
                    python=sys.executable,
                    startup_timeout_seconds=10,
                    attempt_id=attempt_id,
                    owner_nonce=hashlib.sha256(
                        f"launcher-owner:{fault_mode}".encode()
                    ).hexdigest(),
                    observer_suffix=hashlib.sha256(
                        f"launcher-observer:{fault_mode}".encode()
                    ).hexdigest(),
                    wrapper_arguments_override=[
                        sys.executable,
                        "-B",
                        "-u",
                        str(fake_wrapper),
                        "claim",
                    ],
                )
        attempt_root = (
            campaign_root
            / "preflight_launch_attempts/by_policy"
            / policy_sha256
            / attempt_id
        )
        assert not (attempt_root / "launch_accepted.json").exists()
        assert not (
            attempt_root / "launch_ownership_release.json"
        ).exists()
        assert (
            subprocess.run(
                ["tmux", "has-session", "-t", consumer_session],
                capture_output=True,
                text=True,
            ).returncode
            != 0
        )
        assert (
            subprocess.run(
                [
                    "tmux",
                    "has-session",
                    "-t",
                    launcher.CONTROLLER_SESSION,
                ],
                capture_output=True,
                text=True,
            ).returncode
            != 0
        )
    finally:
        for session in (
            launcher.CONTROLLER_SESSION,
            consumer_session,
        ):
            subprocess.run(
                ["tmux", "kill-session", "-t", session],
                capture_output=True,
                text=True,
            )


def _finalize_exact_fixture_consumer(
    launcher,
    *,
    attempt_root: Path,
    config: Path,
    timeout_seconds: float = 5.0,
) -> dict[str, str]:
    receipt_path = attempt_root / "launch_receipt.json"
    if not receipt_path.is_file():
        return {
            "consumer": "not_reserved",
            "controller": "not_reserved",
        }
    receipt = load_json(
        receipt_path, "fixture cleanup launch receipt"
    )
    attempt_path = Path(
        receipt["pane_fault_consumer"]["artifacts"]["attempt"]
    )
    if not attempt_path.is_file():
        if (
            launcher._tmux_pane(launcher.CONTROLLER_SESSION)
            is not None
        ):
            raise AssertionError(
                "fixture controller exists without consumer attempt"
            )
        return {
            "consumer": "not_reserved",
            "controller": "absent",
        }
    attempt = load_json(
        attempt_path, "fixture cleanup consumer attempt"
    )
    terminal_path = Path(attempt["artifacts"]["terminal"])
    join_path = Path(attempt["artifacts"]["join"])
    cleanup_path = Path(attempt["artifacts"]["cleanup"])
    consumer_session = attempt["consumer_session"]
    consumer_pane = launcher._tmux_pane(consumer_session)
    deadline = time.monotonic() + timeout_seconds
    while (
        not terminal_path.is_file()
        and not cleanup_path.is_file()
        and consumer_pane is not None
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
        consumer_pane = launcher._tmux_pane(consumer_session)
    if terminal_path.is_file() and (
        consumer_pane is not None
        or join_path.is_file()
        or cleanup_path.is_file()
    ):
        launcher.join_pane_fault_consumer(
            attempt_path=attempt_path,
            config=config,
            timeout_seconds=timeout_seconds,
        )
        consumer_status = "formal_join"
    elif terminal_path.is_file():
        terminal = load_json(
            terminal_path, "fixture cleanup consumer terminal"
        )
        if (
            terminal.get("consumer_attempt")
            != launcher._json_binding(
                attempt_path, "consumer_attempt_sha256"
            )
            or terminal.get("attempt_id") != attempt["attempt_id"]
            or terminal.get("consumer_session")
            != consumer_session
            or terminal.get("consumer_owner_nonce")
            != attempt["consumer_owner_nonce"]
            or terminal.get("consumer_terminal_sha256")
            != launcher._canonical_digest(
                terminal, "consumer_terminal_sha256"
            )
            or launcher._tmux_pane(
                consumer_session
            )
            is not None
            or join_path.is_file()
            or cleanup_path.is_file()
        ):
            raise AssertionError(
                "absent fixture consumer terminal binding differs"
            )
        consumer_status = "absent_without_join"
    elif cleanup_path.is_file() or join_path.is_file():
        raise AssertionError(
            "fixture consumer finalization lacks typed terminal"
        )
    elif consumer_pane is not None:
        launcher._cleanup_failed_pane_fault_consumer(
            consumer_session,
            attempt["consumer_owner_nonce"],
        )
        consumer_status = "timeout_exact_cleanup"
    else:
        consumer_status = "absent_without_terminal"
    controller_pane = launcher._tmux_pane(
        launcher.CONTROLLER_SESSION
    )
    if controller_pane is not None:
        current_owner = launcher._tmux_owner_seal(
            launcher.CONTROLLER_SESSION,
            receipt["controller_owner_nonce"],
        )
        expected_owner = attempt["gate_owner_seal"]
        if current_owner["pane_dead"]:
            launcher._validate_pane_owner_lifecycle_transition(
                expected_owner,
                current_owner,
                label="fixture controller",
            )
        elif current_owner != expected_owner:
            raise AssertionError(
                "live fixture controller owner differs"
            )
        launcher._kill_exact_session(
            launcher.CONTROLLER_SESSION,
            receipt["controller_owner_nonce"],
            current_owner,
        )
        controller_status = "exact_cleanup"
    else:
        controller_status = "absent"
    assert (
        launcher._tmux_pane(consumer_session) is None
    )
    assert (
        launcher._tmux_pane(launcher.CONTROLLER_SESSION)
        is None
    )
    return {
        "consumer": consumer_status,
        "controller": controller_status,
    }


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    ("phase", "gate_signal"),
    (
        ("preclaim", signal.SIGTERM),
        ("preclaim", signal.SIGKILL),
        ("postclaim", signal.SIGTERM),
        ("postclaim", signal.SIGKILL),
    ),
)
def test_preflight_gate_signal_has_no_wrapper_orphan(
    tmp_path: Path,
    phase: str,
    gate_signal: signal.Signals,
    request: pytest.FixtureRequest,
) -> None:
    launcher, repo_root, config, fake_wrapper, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    provenance = (
        f"{tmp_path.resolve()}:gate-signal:{phase}:"
        f"{gate_signal.value}"
    )
    attempt_id = hashlib.sha256(provenance.encode()).hexdigest()
    campaign_root = tmp_path / "campaign"
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts/by_policy"
        / policy_sha256
        / attempt_id
    )
    request.addfinalizer(
        lambda: _finalize_exact_fixture_consumer(
            launcher,
            attempt_root=attempt_root,
            config=config,
        )
    )
    arguments = {
        "repo_root": repo_root,
        "config": config,
        "campaign_root": campaign_root,
        "policy_sha256": policy_sha256,
        "python": sys.executable,
        "startup_timeout_seconds": 5,
        "attempt_id": attempt_id,
        "owner_nonce": hashlib.sha256(
            f"{provenance}:gate-owner".encode()
        ).hexdigest(),
        "observer_suffix": hashlib.sha256(
            f"{provenance}:gate-observer".encode()
        ).hexdigest(),
        "wrapper_arguments_override": (
            [
                sys.executable,
                "-B",
                "-u",
                "-c",
                "import time; time.sleep(30)",
            ]
            if phase == "preclaim"
            else [
                sys.executable,
                "-B",
                "-u",
                str(fake_wrapper),
                "claim_hold",
            ]
        ),
    }
    result_box: dict[str, Any] = {}
    failure_box: list[BaseException] = []

    def launch() -> None:
        try:
            result_box["value"] = launcher.launch_preflight(
                **arguments
            )
        except BaseException as exc:
            failure_box.append(exc)

    thread: threading.Thread | None = None
    if phase == "preclaim":
        thread = threading.Thread(target=launch, daemon=True)
        thread.start()
    else:
        launch()
        assert not failure_box
        assert result_box["value"]["startup_window_closed"] is True
    started_path = attempt_root / "wrapper_started.json"
    deadline = time.monotonic() + 5
    while not started_path.is_file():
        if time.monotonic() >= deadline:
            raise AssertionError("wrapper-start evidence timed out")
        time.sleep(0.01)
    started = load_json(started_path, "wrapper started")
    gate_pid = started["pane_gate_process"]["pid"]
    wrapper_pid = started["wrapper_process"]["pid"]
    os.kill(gate_pid, gate_signal)
    if thread is not None:
        thread.join(timeout=5)
        assert not thread.is_alive()
        if failure_box:
            assert gate_signal == signal.SIGKILL
            assert len(failure_box) == 1
            reservation = failure_box[0]
            assert isinstance(
                reservation,
                launcher.PaneFaultConsumerReservationError,
            )
            assert isinstance(reservation.failure, RuntimeError)
            assert (
                "wrapper process exited before publishing a durable "
                "claim"
                in str(reservation.failure)
            )
            assert result_box == {}
        else:
            result = result_box["value"]
            assert (
                result["status"]
                == "wrapper_exited_before_claim"
            )
        assert not (attempt_root / "launch_accepted.json").exists()
        assert not (
            attempt_root / "launch_ownership_release.json"
        ).exists()
        consumer_session = (
            launcher.PANE_FAULT_CONSUMER_SESSION_PREFIX
            + attempt_id
        )
        assert launcher._tmux_pane(consumer_session) is None
    deadline = time.monotonic() + 5
    while Path(f"/proc/{wrapper_pid}").exists():
        if time.monotonic() >= deadline:
            raise AssertionError("wrapper child survived gate death")
        time.sleep(0.01)
    execution_path = attempt_root / "gate_execution_terminal.json"
    if gate_signal == signal.SIGTERM:
        deadline = time.monotonic() + 5
        while not execution_path.is_file():
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "forwarded gate signal terminal timed out"
                )
            time.sleep(0.01)
        execution = load_json(
            execution_path, "gate signal execution terminal"
        )
        assert execution["exit_kind"] == "signal"
        assert execution["signal_number"] == signal.SIGTERM
        assert signal.SIGTERM in execution["supervisor_signals"]
        if phase == "postclaim":
            assert execution["launch_terminal"] is not None
            assert execution["launch_ownership_release"] is not None
    else:
        assert not execution_path.exists()
    deadline = time.monotonic() + 5
    while launcher._tmux_pane(launcher.CONTROLLER_SESSION) is not None:
        if time.monotonic() >= deadline:
            raise AssertionError("gate session survived signal")
        time.sleep(0.01)
    teardown = _finalize_exact_fixture_consumer(
        launcher,
        attempt_root=attempt_root,
        config=config,
    )
    if phase == "preclaim" and gate_signal == signal.SIGKILL:
        assert teardown["consumer"] == "absent_without_terminal"
    else:
        assert teardown["consumer"] == "formal_join"
    assert teardown["controller"] == "absent"
    consumer_session = (
        launcher.PANE_FAULT_CONSUMER_SESSION_PREFIX + attempt_id
    )
    assert launcher._tmux_pane(consumer_session) is None


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux prctl and /proc",
)
def test_wrapper_pdeathsig_covers_pre_evidence_parent_death(
    tmp_path: Path,
) -> None:
    launcher_path = (
        Path(__file__).parents[1]
        / "scripts/run_canonical_preflight_launcher.py"
    )
    child_pid_path = tmp_path / "child.pid"
    parent_code = """
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
spec = importlib.util.spec_from_file_location("launcher", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
child = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(30)"],
    shell=False,
    preexec_fn=module._wrapper_child_setup,
)
Path(sys.argv[2]).write_text(str(child.pid), encoding="utf-8")
time.sleep(30)
"""
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            parent_code,
            str(launcher_path),
            str(child_pid_path),
        ]
    )
    try:
        deadline = time.monotonic() + 5
        while not child_pid_path.is_file():
            if time.monotonic() >= deadline:
                raise AssertionError("child PID publication timed out")
            time.sleep(0.005)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        assert Path(f"/proc/{child_pid}").exists()
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while Path(f"/proc/{child_pid}").exists():
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "PDEATHSIG child survived pre-evidence parent death"
                )
            time.sleep(0.005)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    (
        "mode",
        "expected_kind",
        "expected_returncode",
        "expected_exit_code",
        "expected_signal",
    ),
    (
        ("bad_argparse", "exit", 2, 2, None),
        ("bad_python", "exec_error", None, None, None),
        ("bad_script", "exit", 2, 2, None),
        ("bad_import", "exit", 1, 1, None),
        ("early_exit", "exit", 7, 7, None),
        ("signal_exit", "signal", -15, None, 15),
    ),
)
def test_preflight_launcher_preclaim_failures_are_terminal(
    tmp_path: Path,
    mode: str,
    expected_kind: str,
    expected_returncode: int | None,
    expected_exit_code: int | None,
    expected_signal: int | None,
) -> None:
    launcher, repo_root, config, fake_wrapper, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    campaign_root = tmp_path / "campaign"
    commands = {
        "bad_argparse": [
            sys.executable,
            "-B",
            "-u",
            str(fake_wrapper),
            "--not-valid",
        ],
        "bad_python": [
            str(tmp_path / "missing-python"),
            str(fake_wrapper),
            "claim",
        ],
        "bad_script": [
            sys.executable,
            "-B",
            "-u",
            str(tmp_path / "missing-script.py"),
        ],
        "bad_import": [
            sys.executable,
            "-B",
            "-u",
            "-c",
            "import deliberately_missing_preflight_module",
        ],
        "early_exit": [
            sys.executable,
            "-B",
            "-u",
            "-c",
            "raise SystemExit(7)",
        ],
        "signal_exit": [
            sys.executable,
            "-B",
            "-u",
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ],
    }
    result = launcher.launch_preflight(
        repo_root=repo_root,
        config=config,
        campaign_root=campaign_root,
        policy_sha256=policy_sha256,
        python=sys.executable,
        startup_timeout_seconds=10,
        attempt_id=hashlib.sha256(mode.encode()).hexdigest(),
        owner_nonce=hashlib.sha256(
            f"owner:{mode}".encode()
        ).hexdigest(),
        observer_suffix=hashlib.sha256(
            f"observer:{mode}".encode()
        ).hexdigest(),
        wrapper_arguments_override=commands[mode],
    )
    assert result["status"] == "wrapper_exited_before_claim"
    execution = result["gate_execution"]
    assert execution["exit_kind"] == expected_kind
    assert execution["returncode"] == expected_returncode
    assert execution["exit_code"] == expected_exit_code
    assert execution["signal_number"] == expected_signal
    assert execution["launch_accepted"] is None
    assert execution["launch_terminal"] is None
    assert execution["launch_ownership_release"] is None
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts/by_policy"
        / policy_sha256
        / hashlib.sha256(mode.encode()).hexdigest()
    )
    receipt = load_json(
        attempt_root / "launch_receipt.json",
        f"{mode} launch receipt",
    )
    consumer_terminal = load_json(
        Path(
            receipt["pane_fault_consumer"]["artifacts"]["terminal"]
        ),
        f"{mode} pane fault consumer terminal",
    )
    consumer_cleanup = load_json(
        Path(
            receipt["pane_fault_consumer"]["artifacts"][
                "controller_cleanup"
            ]
        ),
        f"{mode} pane fault consumer cleanup",
    )
    assert consumer_terminal["ownership_chain_state"] == "absent"
    assert consumer_cleanup["ownership_chain_state"] == "absent"
    if mode == "bad_argparse":
        partial = json.loads(json.dumps(execution))
        partial["launch_accepted"] = {
            "path": str(attempt_root / "launch_accepted.json"),
            "sha256": hashlib.sha256(b"missing").hexdigest(),
            "canonical_sha256": hashlib.sha256(
                b"missing-canonical"
            ).hexdigest(),
        }
        partial["gate_execution_terminal_sha256"] = (
            launcher._canonical_digest(
                partial, "gate_execution_terminal_sha256"
            )
        )
        partial_path = attempt_root / "partial_gate_terminal.json"
        write_exclusive_json(partial_path, partial)
        with pytest.raises(RuntimeError, match="terminal differs"):
            launcher._validate_gate_execution_terminal(
                partial_path,
                receipt_binding=launcher._json_binding(
                    attempt_root / "launch_receipt.json",
                    "launch_receipt_sha256",
                ),
                receipt_identity=launcher._opened_file_identity(
                    attempt_root / "launch_receipt.json"
                ),
                gate_ready_binding=launcher._json_binding(
                    attempt_root / "pane_gate_ready.json",
                    "pane_gate_ready_sha256",
                ),
                wrapper_arguments=receipt["wrapper_arguments"],
            )
    if mode == "bad_python":
        assert execution["fault_channel"] == receipt["fault_channel"]
        assert execution["fault_channel_snapshot"] == {
            "state": "empty",
            "record": None,
            "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
        assert (
            execution["fault_channel_validation_failure"] is None
        )
        assert execution["fault_channel_close_failure"] is None
        assert execution["wrapper_outcome"]["status"] == "exec_error"
        assert execution["wrapper_outcome"]["exit_code"] == 126
        assert execution["wrapper_outcome"]["wrapper_exit"] is None
        assert execution["wrapper_outcome"]["failure"]["type"] in {
            "FileNotFoundError",
            "OSError",
        }
        wrapper_claim_path = Path(
            receipt["wrapper_claim_path"]
        )
        assert not (wrapper_claim_path.parent / "wrapper_exit.json").exists()
    assert not (
        execution["exit_code"] is None
        and execution["signal_number"] is None
        and execution["exit_kind"] != "exec_error"
    )
    assert result["pane_log"] is not None
    assert Path(result["pane_log"]["path"]).is_file()
    assert (
        subprocess.run(
            [
                "tmux",
                "has-session",
                "-t",
                launcher.CONTROLLER_SESSION,
            ],
            capture_output=True,
            text=True,
        ).returncode
        != 0
    )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    ("mode", "expected_status"),
    (
        ("claim_wrong_policy", "launcher_failed"),
        ("claim_wrong_pid", "launcher_failed"),
        ("claim_wrong_ppid", "launcher_failed"),
        ("claim_wrong_pgid", "launcher_failed"),
        ("claim_wrong_start_ticks", "launcher_failed"),
        ("claim_wrong_pane", "launcher_failed"),
        ("claim_wrong_gate", "launcher_failed"),
        ("claim_wrong_argv", "launcher_failed"),
        ("claim_wrong_receipt", "launcher_failed"),
        ("claim_wrong_started", "launcher_failed"),
        ("claim_wrong_executable", "launcher_failed"),
        ("claim_malformed", "launcher_failed"),
        ("claim_late", "wrapper_claim_timeout"),
    ),
)
def test_preflight_launcher_rejects_wrong_or_late_claim(
    tmp_path: Path,
    mode: str,
    expected_status: str,
) -> None:
    launcher, repo_root, config, fake_wrapper, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    result = launcher.launch_preflight(
        repo_root=repo_root,
        config=config,
        campaign_root=tmp_path / "campaign",
        policy_sha256=policy_sha256,
        python=sys.executable,
        startup_timeout_seconds=0.5,
        attempt_id=hashlib.sha256(mode.encode()).hexdigest(),
        owner_nonce=hashlib.sha256(
            f"owner:{mode}".encode()
        ).hexdigest(),
        observer_suffix=hashlib.sha256(
            f"observer:{mode}".encode()
        ).hexdigest(),
        wrapper_arguments_override=[
            sys.executable,
            "-B",
            "-u",
            str(fake_wrapper),
            mode,
        ],
    )
    assert result["status"] == expected_status
    assert result["session_residual"] is False
    attempt_root = (
        tmp_path
        / "campaign/preflight_launch_attempts/by_policy"
        / policy_sha256
        / hashlib.sha256(mode.encode()).hexdigest()
    )
    assert not (
        attempt_root / "launch_ownership_release.json"
    ).exists()
    assert not (attempt_root / "launch_accepted.json").exists()
    assert (
        subprocess.run(
            [
                "tmux",
                "has-session",
                "-t",
                launcher.CONTROLLER_SESSION,
            ],
            capture_output=True,
            text=True,
        ).returncode
        != 0
    )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_preflight_launcher_rejects_same_content_receipt_inode_replacement_before_claim(
    tmp_path: Path,
) -> None:
    launcher, repo_root, config, fake_wrapper, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    attempt_id = hashlib.sha256(
        b"receipt-replaced-before-claim"
    ).hexdigest()
    campaign_root = tmp_path / "campaign"
    result = launcher.launch_preflight(
        repo_root=repo_root,
        config=config,
        campaign_root=campaign_root,
        policy_sha256=policy_sha256,
        python=sys.executable,
        startup_timeout_seconds=5,
        attempt_id=attempt_id,
        owner_nonce=hashlib.sha256(
            b"receipt-before-owner"
        ).hexdigest(),
        observer_suffix=hashlib.sha256(
            b"receipt-before-observer"
        ).hexdigest(),
        wrapper_arguments_override=[
            sys.executable,
            "-B",
            "-u",
            str(fake_wrapper),
            "claim_replace_receipt_before_claim",
        ],
    )
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts/by_policy"
        / policy_sha256
        / attempt_id
    )
    receipt = load_json(
        attempt_root / "launch_receipt.json",
        "replaced launch receipt",
    )
    assert receipt["launch_receipt_sha256"] == (
        launcher._canonical_digest(
            receipt, "launch_receipt_sha256"
        )
    )
    assert result["status"] == "launcher_failed"
    assert "identity" in result["failure"]["message"]
    assert result["session_residual"] is False
    assert not (attempt_root / "launch_accepted.json").exists()
    assert not (
        attempt_root / "launch_ownership_release.json"
    ).exists()
    assert launcher._tmux_pane(launcher.CONTROLLER_SESSION) is None


def test_preflight_launcher_rechecks_receipt_identity_after_claim_before_accept(
    tmp_path: Path,
) -> None:
    launcher, *_unused = _prepare_preflight_launcher_fixture(
        tmp_path
    )
    source = inspect.getsource(launcher.launch_preflight)
    claim_validation = source.index("_validate_wrapper_claim(")
    identity_recheck = source.index(
        "if _opened_file_identity(receipt_path) != receipt_identity:",
        claim_validation,
    )
    accepted_publication = source.index(
        "_publish_accepted(", identity_recheck
    )
    assert claim_validation < identity_recheck < accepted_publication


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_preflight_launcher_rejects_same_content_receipt_inode_replacement_after_claim_and_preserves_foreign_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher, repo_root, config, fake_wrapper, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    real_validate = launcher._validate_wrapper_claim
    replacement: dict[str, Any] = {}
    foreign_nonce = hashlib.sha256(
        b"receipt-after-foreign"
    ).hexdigest()

    def replace_after_valid_claim(
        path: Path, **kwargs: Any
    ) -> dict[str, Any]:
        claim = real_validate(path, **kwargs)
        receipt_path = Path(
            str(kwargs["receipt_binding"]["path"])
        )
        original_bytes = receipt_path.read_bytes()
        original = receipt_path.stat()
        replacement_path = receipt_path.with_name(
            ".same-content-after-claim.json"
        )
        descriptor = os.open(
            replacement_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        try:
            os.write(descriptor, original_bytes)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(replacement_path, receipt_path)
        current = receipt_path.stat()
        replacement.update(
            {
                "original_inode": int(original.st_ino),
                "current_inode": int(current.st_ino),
                "content": receipt_path.read_bytes(),
                "expected_content": original_bytes,
            }
        )
        subprocess.run(
            [
                "tmux",
                "kill-session",
                "-t",
                launcher.CONTROLLER_SESSION,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                launcher.CONTROLLER_SESSION,
                "-e",
                f"{launcher.TMUX_OWNER_ENV}={foreign_nonce}",
                "sleep",
                "30",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return claim

    monkeypatch.setattr(
        launcher,
        "_validate_wrapper_claim",
        replace_after_valid_claim,
    )
    attempt_id = hashlib.sha256(
        b"receipt-replaced-after-claim"
    ).hexdigest()
    campaign_root = tmp_path / "campaign"
    try:
        result = launcher.launch_preflight(
            repo_root=repo_root,
            config=config,
            campaign_root=campaign_root,
            policy_sha256=policy_sha256,
            python=sys.executable,
            startup_timeout_seconds=5,
            attempt_id=attempt_id,
            owner_nonce=hashlib.sha256(
                b"receipt-after-owner"
            ).hexdigest(),
            observer_suffix=hashlib.sha256(
                b"receipt-after-observer"
            ).hexdigest(),
            wrapper_arguments_override=[
                sys.executable,
                "-B",
                "-u",
                str(fake_wrapper),
                "claim_hold",
            ],
        )
        attempt_root = (
            campaign_root
            / "preflight_launch_attempts/by_policy"
            / policy_sha256
            / attempt_id
        )
        assert replacement["original_inode"] != replacement[
            "current_inode"
        ]
        assert replacement["content"] == replacement[
            "expected_content"
        ]
        assert result["status"] == "launcher_failed"
        assert "identity" in result["failure"]["message"]
        assert result["session_residual"] is True
        assert not (attempt_root / "launch_accepted.json").exists()
        assert not (
            attempt_root / "launch_ownership_release.json"
        ).exists()
        pane = launcher._tmux_pane(launcher.CONTROLLER_SESSION)
        assert pane is not None
        foreign_owner = subprocess.run(
            [
                "tmux",
                "show-environment",
                "-t",
                launcher.CONTROLLER_SESSION,
                launcher.TMUX_OWNER_ENV,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert foreign_owner == (
            f"{launcher.TMUX_OWNER_ENV}={foreign_nonce}"
        )
    finally:
        subprocess.run(
            [
                "tmux",
                "kill-session",
                "-t",
                launcher.CONTROLLER_SESSION,
            ],
            capture_output=True,
            text=True,
        )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_preflight_launcher_duplicate_session_is_foreign_safe(
    tmp_path: Path,
) -> None:
    launcher, repo_root, config, _, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            launcher.CONTROLLER_SESSION,
            "sleep",
            "30",
        ],
        check=True,
    )
    before = launcher._tmux_pane(launcher.CONTROLLER_SESSION)
    assert before is not None
    before_process = launcher._process_identity(before["pane_pid"])
    try:
        result = launcher.launch_preflight(
            repo_root=repo_root,
            config=config,
            campaign_root=tmp_path / "campaign",
            policy_sha256=policy_sha256,
            python=sys.executable,
            attempt_id=hashlib.sha256(b"duplicate").hexdigest(),
            owner_nonce=hashlib.sha256(
                b"duplicate-owner"
            ).hexdigest(),
            observer_suffix=hashlib.sha256(
                b"duplicate-observer"
            ).hexdigest(),
            wrapper_arguments_override=[
                sys.executable,
                "-c",
                "raise SystemExit(0)",
            ],
        )
        assert result["status"] == "tmux_launch_failed"
        assert result["session_residual"] is True
        assert (
            subprocess.run(
                [
                    "tmux",
                    "has-session",
                    "-t",
                    launcher.CONTROLLER_SESSION,
                ],
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
        after = launcher._tmux_pane(launcher.CONTROLLER_SESSION)
        assert after == before
        assert launcher._process_identity(after["pane_pid"]) == before_process
    finally:
        subprocess.run(
            [
                "tmux",
                "kill-session",
                "-t",
                launcher.CONTROLLER_SESSION,
            ],
            capture_output=True,
            text=True,
        )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_preflight_launcher_attempt_id_is_never_reused(
    tmp_path: Path,
) -> None:
    launcher, repo_root, config, _, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    attempt_id = hashlib.sha256(b"one-start-only").hexdigest()
    arguments = {
        "repo_root": repo_root,
        "config": config,
        "campaign_root": tmp_path / "campaign",
        "policy_sha256": policy_sha256,
        "python": sys.executable,
        "startup_timeout_seconds": 10,
        "attempt_id": attempt_id,
        "owner_nonce": hashlib.sha256(b"reuse-owner").hexdigest(),
        "observer_suffix": hashlib.sha256(
            b"reuse-observer"
        ).hexdigest(),
        "wrapper_arguments_override": [
            sys.executable,
            "-c",
            "raise SystemExit(9)",
        ],
    }
    first = launcher.launch_preflight(**arguments)
    assert first["status"] == "wrapper_exited_before_claim"
    registry = (
        tmp_path
        / "campaign/preflight_launch_attempts/started"
        / f"{attempt_id}.json"
    )
    attempt_root = (
        tmp_path
        / "campaign/preflight_launch_attempts/by_policy"
        / policy_sha256
        / attempt_id
    )
    evidence = [
        registry,
        attempt_root / "pane.log",
        attempt_root / "launch_receipt.json",
        attempt_root / "pane_gate_ready.json",
        attempt_root / "launch_tmux_started.json",
        attempt_root / "pane_gate_release.json",
        attempt_root / "launch_terminal.json",
    ]
    evidence_sha = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence
    }
    with pytest.raises(
        launcher.LauncherExclusivePublishError
    ) as collision:
        launcher.launch_preflight(**arguments)
    assert collision.value.commit_state == "collision"
    assert collision.value.stage == "existing_final_verify"
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence
    } == evidence_sha


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    ("stage", "expected_setup_terminal"),
    (
        ("registry_create", False),
        ("pane_log_create", True),
        ("receipt_create", True),
    ),
)
def test_preflight_launcher_create_faults_never_reach_tmux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    expected_setup_terminal: bool,
) -> None:
    launcher, repo_root, config, _, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    attempt_id = hashlib.sha256(stage.encode()).hexdigest()
    campaign_root = tmp_path / "campaign"
    original_write = launcher._write_exclusive
    original_open = launcher.os.open
    original_run = launcher.subprocess.run
    tmux_called = False

    def guarded_run(arguments, *args, **kwargs):
        nonlocal tmux_called
        if arguments and arguments[0] == "tmux":
            tmux_called = True
        return original_run(arguments, *args, **kwargs)

    def failing_write(path, value):
        if (
            stage == "registry_create"
            and path.name == f"{attempt_id}.json"
            and path.parent.name == "started"
        ) or (
            stage == "receipt_create"
            and path.name == "launch_receipt.json"
        ):
            raise OSError(f"injected {stage}")
        return original_write(path, value)

    def failing_open(path, flags, mode=0o777, **kwargs):
        if stage == "pane_log_create" and Path(path).name == "pane.log":
            raise OSError("injected pane_log_create")
        return original_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", guarded_run)
    monkeypatch.setattr(launcher, "_write_exclusive", failing_write)
    monkeypatch.setattr(launcher.os, "open", failing_open)
    with pytest.raises(OSError, match=f"injected {stage}"):
        launcher.launch_preflight(
            repo_root=repo_root,
            config=config,
            campaign_root=campaign_root,
            policy_sha256=policy_sha256,
            python=sys.executable,
            attempt_id=attempt_id,
            owner_nonce=hashlib.sha256(
                f"owner:{stage}".encode()
            ).hexdigest(),
            observer_suffix=hashlib.sha256(
                f"observer:{stage}".encode()
            ).hexdigest(),
        )
    assert tmux_called is False
    setup_terminal = (
        campaign_root
        / "preflight_launch_attempts/setup_terminals"
        / f"{attempt_id}.json"
    )
    assert setup_terminal.exists() is expected_setup_terminal
    if expected_setup_terminal:
        value = load_json(setup_terminal, "setup terminal")
        assert value["contract_type"] == (
            "safa_canonical_preflight_launch_setup_terminal_v1"
        )
        assert value["tmux_execution_count"] == 0
        assert value["scientific_execution_started"] is False


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    "target",
    ("started", "pane.log", "launch_receipt.json"),
)
def test_preflight_launcher_fsync_faults_are_durable_and_pre_tmux(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    launcher, repo_root, config, _, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    attempt_id = hashlib.sha256(f"fsync:{target}".encode()).hexdigest()
    campaign_root = tmp_path / "campaign"
    original_fsync = launcher.os.fsync
    original_write = launcher._write_exclusive
    original_run = launcher.subprocess.run
    injected = False
    tmux_called = False
    active_publication = False

    def guarded_run(arguments, *args, **kwargs):
        nonlocal tmux_called
        if arguments and arguments[0] == "tmux":
            tmux_called = True
        return original_run(arguments, *args, **kwargs)

    def failing_fsync(descriptor):
        nonlocal injected
        descriptor_stat = os.fstat(descriptor)
        descriptor_path = Path(
            os.readlink(f"/proc/self/fd/{descriptor}")
        )
        matches = (
            active_publication
            and stat.S_ISREG(descriptor_stat.st_mode)
        ) or (
            target == "pane.log"
            and descriptor_path.name == target
        )
        if matches and not injected:
            injected = True
            raise OSError(f"injected fsync:{target}")
        return original_fsync(descriptor)

    def failing_publication(path, value):
        nonlocal active_publication
        matches = (
            target == "started"
            and path.name == f"{attempt_id}.json"
            and path.parent.name == "started"
        ) or (
            target == "launch_receipt.json"
            and path.name == target
        )
        if not matches:
            return original_write(path, value)
        active_publication = True
        try:
            return original_write(path, value)
        finally:
            active_publication = False

    monkeypatch.setattr(launcher.subprocess, "run", guarded_run)
    monkeypatch.setattr(launcher.os, "fsync", failing_fsync)
    monkeypatch.setattr(
        launcher, "_write_exclusive", failing_publication
    )
    expected_failure = (
        OSError
        if target == "pane.log"
        else launcher.LauncherExclusivePublishError
    )
    with pytest.raises(expected_failure, match="injected fsync") as failure:
        launcher.launch_preflight(
            repo_root=repo_root,
            config=config,
            campaign_root=campaign_root,
            policy_sha256=policy_sha256,
            python=sys.executable,
            attempt_id=attempt_id,
            owner_nonce=hashlib.sha256(
                f"owner:fsync:{target}".encode()
            ).hexdigest(),
            observer_suffix=hashlib.sha256(
                f"observer:fsync:{target}".encode()
            ).hexdigest(),
        )
    if target != "pane.log":
        assert failure.value.commit_state == "precommit_failed_clean"
        assert failure.value.stage == "publication"
        assert failure.value.quarantined is False
    assert injected is True
    assert tmux_called is False
    setup_terminal = (
        campaign_root
        / "preflight_launch_attempts/setup_terminals"
        / f"{attempt_id}.json"
    )
    if target in {"started", "launch_receipt.json"}:
        assert not setup_terminal.exists()
        if target == "started":
            started_root = (
                campaign_root / "preflight_launch_attempts/started"
            )
            assert not (
                started_root / f"{attempt_id}.json"
            ).exists()
            assert list(
                started_root.glob(f".{attempt_id}.json.publish-*")
            ) == []
        else:
            attempt_root = (
                campaign_root
                / "preflight_launch_attempts/by_policy"
                / policy_sha256
                / attempt_id
            )
            assert not (
                attempt_root / "launch_receipt.json"
            ).exists()
            assert list(
                attempt_root.glob(
                    ".launch_receipt.json.publish-*"
                )
            ) == []
        return
    value = load_json(setup_terminal, "setup terminal")
    assert value["contract_type"] == (
        "safa_canonical_preflight_launch_setup_terminal_v1"
    )
    assert value["tmux_execution_count"] == 0
    assert value["scientific_execution_started"] is False


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    ("fault", "expected_contract", "expected_status"),
    (
        (
            "remain_set",
            "safa_canonical_preflight_launch_terminal_v1",
            "launcher_failed",
        ),
        (
            "remain_verify",
            "safa_canonical_preflight_launch_terminal_v1",
            "launcher_failed",
        ),
        (
            "gate_release_write",
            "safa_canonical_preflight_launch_terminal_v1",
            "launcher_failed",
        ),
        (
            "accepted_write",
            "safa_canonical_preflight_launch_terminal_v1",
            "launcher_failed",
        ),
        (
            "ownership_terminal_write",
            "safa_canonical_preflight_launch_terminal_v1",
            "launcher_failed",
        ),
        (
            "ownership_release_write",
            (
                "safa_canonical_preflight_launch_post_terminal_"
                "failure_v1"
            ),
            "launch_release_failed",
        ),
    ),
)
def test_preflight_launcher_ownership_faults_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    expected_contract: str,
    expected_status: str,
) -> None:
    launcher, repo_root, config, fake_wrapper, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    original_write = launcher._write_exclusive
    original_set = launcher._set_remain_on_exit
    original_verify = launcher._verify_remain_on_exit
    injection_count = 0

    def failing_write(path, value):
        nonlocal injection_count
        matches = {
            "gate_release_write": "pane_gate_release.json",
            "accepted_write": "launch_accepted.json",
            "ownership_terminal_write": "launch_terminal.json",
            "ownership_release_write": "launch_ownership_release.json",
        }
        if (
            fault in matches
            and path.name == matches[fault]
            and (
                fault != "ownership_terminal_write"
                or value.get("status") == "ownership_transferred"
            )
        ):
            injection_count += 1
            raise OSError(f"injected {fault}")
        return original_write(path, value)

    def failing_set(pane, enabled):
        nonlocal injection_count
        if fault == "remain_set" and enabled:
            injection_count += 1
            raise OSError(f"injected {fault}")
        return original_set(pane, enabled)

    def failing_verify(pane, expected):
        nonlocal injection_count
        if fault == "remain_verify" and expected == "on":
            injection_count += 1
            raise OSError("injected remain_verify")
        return original_verify(pane, expected)

    monkeypatch.setattr(launcher, "_write_exclusive", failing_write)
    monkeypatch.setattr(launcher, "_set_remain_on_exit", failing_set)
    monkeypatch.setattr(launcher, "_verify_remain_on_exit", failing_verify)
    attempt_id = hashlib.sha256(fault.encode()).hexdigest()
    attempt_root = (
        tmp_path
        / "campaign/preflight_launch_attempts/by_policy"
        / policy_sha256
        / attempt_id
    )
    try:
        result = launcher.launch_preflight(
            repo_root=repo_root,
            config=config,
            campaign_root=tmp_path / "campaign",
            policy_sha256=policy_sha256,
            python=sys.executable,
            startup_timeout_seconds=2,
            attempt_id=attempt_id,
            owner_nonce=hashlib.sha256(
                f"owner:{fault}".encode()
            ).hexdigest(),
            observer_suffix=hashlib.sha256(
                f"observer:{fault}".encode()
            ).hexdigest(),
            wrapper_arguments_override=[
                sys.executable,
                "-B",
                "-u",
                str(fake_wrapper),
                "claim",
            ],
        )
        assert injection_count == 1
        assert result["contract_type"] == expected_contract
        assert result["status"] == expected_status
        assert result["session_residual"] is False
        assert result["pane_log"]["size"] >= 0
        assert len(result["pane_log"]["sha256"]) == 64
        assert not (
            attempt_root / "launch_ownership_release.json"
        ).exists()
        assert (
            subprocess.run(
                [
                    "tmux",
                    "has-session",
                    "-t",
                    launcher.CONTROLLER_SESSION,
                ],
                capture_output=True,
                text=True,
            ).returncode
            != 0
        )
    finally:
        _finalize_exact_fixture_consumer(
            launcher,
            attempt_root=attempt_root,
            config=config,
        )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_preflight_launcher_terminal_first_write_fault_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher, repo_root, config, _, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    original_write = launcher._write_exclusive
    terminal_calls = 0

    def fail_first_terminal(path, value):
        nonlocal terminal_calls
        if path.name == "launch_terminal.json":
            terminal_calls += 1
            raise OSError("injected first terminal write")
        return original_write(path, value)

    monkeypatch.setattr(launcher, "_write_exclusive", fail_first_terminal)
    campaign_root = tmp_path / "campaign"
    attempt_id = hashlib.sha256(b"terminal-first-write").hexdigest()
    with pytest.raises(
        launcher.LauncherTerminalPublishError,
        match="injected first terminal write",
    ) as published:
        launcher.launch_preflight(
            repo_root=repo_root,
            config=config,
            campaign_root=campaign_root,
            policy_sha256=policy_sha256,
            python=sys.executable,
            startup_timeout_seconds=2,
            attempt_id=attempt_id,
            owner_nonce=hashlib.sha256(
                b"terminal-first-owner"
            ).hexdigest(),
            observer_suffix=hashlib.sha256(
                b"terminal-first-observer"
            ).hexdigest(),
            wrapper_arguments_override=[
                sys.executable,
                "-c",
                "raise SystemExit(13)",
            ],
        )
    assert isinstance(published.value.failure, OSError)
    assert str(published.value.failure) == (
        "injected first terminal write"
    )
    assert published.value.secondary_failures == []
    assert terminal_calls == 1
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts/by_policy"
        / policy_sha256
        / attempt_id
    )
    assert not (attempt_root / "launch_terminal.json").exists()
    assert not (
        attempt_root / "launch_terminal_emergency.json"
    ).exists()


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_post_handoff_terminal_error_preserves_identity_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher, repo_root, config, _, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    launcher._install_verified_preflight_apis(config.resolve())
    campaign_root = tmp_path / "campaign"
    attempt_id = hashlib.sha256(
        b"post-handoff-terminal-identity"
    ).hexdigest()
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts/by_policy"
        / policy_sha256
        / attempt_id
    )
    expected = launcher.LauncherTerminalPublishError(
        attempt_root / "launch_terminal.json",
        OSError("injected post-handoff terminal publish"),
    )
    expected.add_secondary_failure(
        stage="external_tmux_cleanup",
        failure=RuntimeError("retained secondary"),
    )
    expected_secondary = copy.deepcopy(
        expected.secondary_failures
    )
    kill_sessions: list[str] = []
    poison_calls = 0
    original_kill = launcher._kill_exact_session
    original_poison = (
        launcher._poison_and_cleanup_pane_fault_consumer
    )

    def tracked_kill(session, owner_nonce, owner_seal):
        kill_sessions.append(session)
        return original_kill(session, owner_nonce, owner_seal)

    def tracked_poison(consumer, failure):
        nonlocal poison_calls
        poison_calls += 1
        return original_poison(consumer, failure)

    def fail_terminal(path, **_kwargs):
        assert path == expected.path
        raise expected

    monkeypatch.setattr(
        launcher, "_kill_exact_session", tracked_kill
    )
    monkeypatch.setattr(
        launcher,
        "_poison_and_cleanup_pane_fault_consumer",
        tracked_poison,
    )
    monkeypatch.setattr(launcher, "_publish_terminal", fail_terminal)
    with pytest.raises(
        launcher.LauncherTerminalPublishError
    ) as published:
        launcher.launch_preflight(
            repo_root=repo_root,
            config=config,
            campaign_root=campaign_root,
            policy_sha256=policy_sha256,
            python=sys.executable,
            startup_timeout_seconds=2,
            attempt_id=attempt_id,
            owner_nonce=hashlib.sha256(
                b"post-handoff-terminal-owner"
            ).hexdigest(),
            observer_suffix=hashlib.sha256(
                b"post-handoff-terminal-observer"
            ).hexdigest(),
            wrapper_arguments_override=[
                sys.executable,
                "-c",
                "raise SystemExit(19)",
            ],
        )
    consumer_session = (
        launcher.PANE_FAULT_CONSUMER_SESSION_PREFIX + attempt_id
    )
    assert published.value is expected
    assert published.value.secondary_failures == expected_secondary
    assert kill_sessions == [consumer_session]
    assert poison_calls == 0
    assert launcher._tmux_pane(launcher.CONTROLLER_SESSION) is None
    assert launcher._tmux_pane(consumer_session) is None


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_preflight_launcher_unwritable_terminal_preserves_attempt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher, repo_root, config, _, policy_sha256 = (
        _prepare_preflight_launcher_fixture(tmp_path)
    )
    original_write = launcher._write_exclusive
    attempt_id = hashlib.sha256(b"terminal-unwritable").hexdigest()
    campaign_root = tmp_path / "campaign"

    terminal_calls = 0

    def fail_all_terminals(path, value):
        nonlocal terminal_calls
        if path.name == "launch_terminal.json":
            terminal_calls += 1
            raise OSError("injected terminal permanently unwritable")
        return original_write(path, value)

    monkeypatch.setattr(launcher, "_write_exclusive", fail_all_terminals)
    arguments = {
        "repo_root": repo_root,
        "config": config,
        "campaign_root": campaign_root,
        "policy_sha256": policy_sha256,
        "python": sys.executable,
        "startup_timeout_seconds": 2,
        "attempt_id": attempt_id,
        "owner_nonce": hashlib.sha256(
            b"terminal-unwritable-owner"
        ).hexdigest(),
        "observer_suffix": hashlib.sha256(
            b"terminal-unwritable-observer"
        ).hexdigest(),
        "wrapper_arguments_override": [
            sys.executable,
            "-c",
            "raise SystemExit(17)",
        ],
    }
    with pytest.raises(
        launcher.LauncherTerminalPublishError,
        match="terminal permanently unwritable",
    ) as published:
        launcher.launch_preflight(**arguments)
    assert isinstance(published.value.failure, OSError)
    assert str(published.value.failure) == (
        "injected terminal permanently unwritable"
    )
    assert published.value.secondary_failures == []
    assert terminal_calls == 1
    attempt_root = (
        campaign_root
        / "preflight_launch_attempts/by_policy"
        / policy_sha256
        / attempt_id
    )
    evidence = [
        campaign_root
        / "preflight_launch_attempts/started"
        / f"{attempt_id}.json",
        attempt_root / "pane.log",
        attempt_root / "launch_receipt.json",
        attempt_root / "pane_gate_ready.json",
        attempt_root / "launch_tmux_started.json",
        attempt_root / "pane_gate_release.json",
    ]
    evidence_sha = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence
    }
    with pytest.raises(
        launcher.LauncherExclusivePublishError
    ) as collision:
        launcher.launch_preflight(**arguments)
    assert collision.value.commit_state == "collision"
    assert collision.value.stage == "existing_final_verify"
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence
    } == evidence_sha
    assert not (attempt_root / "launch_accepted.json").exists()
    assert not (
        attempt_root / "launch_ownership_release.json"
    ).exists()


def test_preflight_launcher_never_overwrites_primary_terminal(
    tmp_path: Path,
) -> None:
    launcher, _, config, _, _ = _prepare_preflight_launcher_fixture(
        tmp_path
    )
    launcher._install_verified_preflight_apis(config)
    receipt_path = tmp_path / "launch_receipt.json"
    receipt = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_receipt_v1"
        ),
    }
    receipt["launch_receipt_sha256"] = launcher._canonical_digest(
        receipt, "launch_receipt_sha256"
    )
    launcher._write_exclusive(receipt_path, receipt)
    log_path = tmp_path / "pane.log"
    log_path.write_text("sealed log\n", encoding="utf-8")
    terminal_path = tmp_path / "launch_terminal.json"
    first = launcher._publish_terminal(
        terminal_path,
        receipt_path=receipt_path,
        receipt_identity=launcher._opened_file_identity(
            receipt_path
        ),
        status="first_failure",
        failure_type="FirstFailure",
        message="primary",
        client=None,
        pane=None,
        tmux_started_path=None,
        log_path=log_path,
        session_residual=False,
        started_at="2026-07-28T00:00:00+00:00",
    )
    primary_sha = hashlib.sha256(
        terminal_path.read_bytes()
    ).hexdigest()
    with pytest.raises(
        launcher.LauncherExclusivePublishError
    ) as collision:
        launcher._publish_terminal(
            terminal_path,
            receipt_path=receipt_path,
            receipt_identity=launcher._opened_file_identity(
                receipt_path
            ),
            status="second_failure",
            failure_type="SecondFailure",
            message="must not replace primary",
            client=None,
            pane=None,
            tmux_started_path=None,
            log_path=log_path,
            session_residual=False,
            started_at="2026-07-28T00:00:01+00:00",
        )
    assert first["status"] == "first_failure"
    assert collision.value.commit_state == "collision"
    assert collision.value.stage == "existing_final_verify"
    assert (
        hashlib.sha256(terminal_path.read_bytes()).hexdigest()
        == primary_sha
    )
    assert not (
        tmp_path / "launch_terminal_emergency.json"
    ).exists()


def test_preflight_gate_terminal_first_write_fault_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    path = tmp_path / "gate_execution_terminal.json"
    original_write = launcher._write_exclusive
    terminal_calls = 0

    def fail_first(target, value):
        nonlocal terminal_calls
        if target == path:
            terminal_calls += 1
            raise OSError("injected gate terminal first write")
        return original_write(target, value)

    monkeypatch.setattr(launcher, "_write_exclusive", fail_first)
    value = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_gate_execution_terminal_v1"
        ),
        "publication_failures": [],
    }
    with pytest.raises(
        OSError, match="injected gate terminal first write"
    ):
        launcher._publish_gate_execution_terminal(path, value)
    assert terminal_calls == 1
    assert not path.exists()


def test_preflight_gate_terminal_permanent_fault_preserves_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    registry = tmp_path / "started.json"
    registry.write_text("reserved\n", encoding="utf-8")
    registry_sha = hashlib.sha256(registry.read_bytes()).hexdigest()

    def fail_terminal(_path, _value):
        raise OSError("injected gate terminal permanently unwritable")

    monkeypatch.setattr(launcher, "_write_exclusive", fail_terminal)
    with pytest.raises(
        OSError, match="gate terminal permanently unwritable"
    ):
        launcher._publish_gate_execution_terminal(
            tmp_path / "gate_execution_terminal.json",
            {
                "schema_version": 1,
                "contract_type": (
                    "safa_canonical_preflight_"
                    "gate_execution_terminal_v1"
                ),
                "publication_failures": [],
            },
        )
    assert (
        hashlib.sha256(registry.read_bytes()).hexdigest()
        == registry_sha
    )


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux /proc",
)
def test_preflight_pane_gate_rejects_replaced_log_before_redirect(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir(mode=0o755)
    os.chmod(attempt_root, 0o755)
    log_path = attempt_root / "pane.log"
    log_path.write_bytes(b"")
    receipt = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_receipt_v1"
        ),
        "pane_gate_fault_channel": (
            launcher._create_fault_channel(
                attempt_root / "pane_gate_fault.channel"
            )
        ),
        "pane_gate_fault_publisher": {
            "path": str(Path(launcher.__file__).resolve()),
            "sha256": hashlib.sha256(
                Path(launcher.__file__).read_bytes()
            ).hexdigest(),
            "role": "launcher_pane_gate",
        },
        "pane_log": {
            **launcher._file_identity(log_path),
            "inode": log_path.stat().st_ino + 1,
        },
    }
    receipt["launch_receipt_sha256"] = launcher._canonical_digest(
        receipt, "launch_receipt_sha256"
    )
    launcher._write_exclusive(
        attempt_root / "launch_receipt.json", receipt
    )
    with pytest.raises(RuntimeError, match="pane log identity differs"):
        launcher._pane_gate(
            attempt_root=attempt_root,
            release_path=attempt_root / "pane_gate_release.json",
            log_path=log_path,
            wrapper_arguments=[sys.executable, "-c", "pass"],
        )
    assert not (attempt_root / "pane_gate_ready.json").exists()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires os.fork",
)
@pytest.mark.parametrize(
    ("commit_state", "quarantined"),
    (
        ("precommit_failed_clean", False),
        ("durability_unknown_quarantined", True),
        ("committed_cleanup_error", True),
        ("collision", False),
    ),
)
def test_preflight_pane_gate_first_publish_reports_typed_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit_state: str,
    quarantined: bool,
) -> None:
    launcher = _launcher_module()
    config = (
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    launcher._install_verified_preflight_apis(config)
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir(mode=0o755)
    channel = launcher._create_fault_channel(
        attempt_root / "pane_gate_fault.channel"
    )
    attempt_id = hashlib.sha256(
        f"pane:{commit_state}".encode()
    ).hexdigest()
    owner_nonce = hashlib.sha256(
        f"owner:{commit_state}".encode()
    ).hexdigest()
    publisher = {
        "path": str(Path(launcher.__file__).resolve()),
        "sha256": hashlib.sha256(
            Path(launcher.__file__).read_bytes()
        ).hexdigest(),
        "role": "launcher_pane_gate",
    }
    receipt = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_launch_receipt_v1"
        ),
        "attempt_id": attempt_id,
        "controller_owner_nonce": owner_nonce,
        "pane_gate_fault_channel": channel,
        "pane_gate_fault_publisher": publisher,
    }
    receipt["launch_receipt_sha256"] = (
        launcher._canonical_digest(
            receipt, "launch_receipt_sha256"
        )
    )
    launcher._write_exclusive(
        attempt_root / "launch_receipt.json", receipt
    )

    def typed_first_publish(**_kwargs):
        raise launcher.LauncherExclusivePublishError(
            commit_state,
            "fixture pane first publication",
            stage="fixture_pane_gate_ready",
            directory_seal={
                "device": channel["directory_device"],
                "inode": channel["directory_inode"],
            },
            payload={"path": str(attempt_root)},
            temporary=None,
            error_number=5,
            quarantined=quarantined,
        )

    monkeypatch.setattr(
        launcher, "_pane_gate_owned", typed_first_publish
    )
    descriptor = launcher._open_presealed_fault_channel(
        attempt_root,
        channel,
        name="pane_gate_fault.channel",
    )
    pid = os.fork()
    if pid == 0:
        os.close(descriptor)
        code = launcher._pane_gate(
            attempt_root=attempt_root,
            release_path=attempt_root / "pane_gate_release.json",
            log_path=attempt_root / "pane.log",
            wrapper_arguments=[sys.executable, "-c", "pass"],
        )
        os._exit(code)
    waited_pid, status = os.waitpid(pid, 0)
    assert waited_pid == pid
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 123
    try:
        snapshot = launcher._read_fault_channel(
            descriptor,
            channel,
            attempt_id=attempt_id,
            owner_nonce=owner_nonce,
            launch_receipt_sha256=receipt[
                "launch_receipt_sha256"
            ],
            publisher=publisher,
        )
    finally:
        os.close(descriptor)
    assert snapshot["state"] == "valid_fault"
    assert (
        snapshot["record"]["failure"]["commit_state"]
        == commit_state
    )
    assert (
        snapshot["record"]["failure"]["quarantined"]
        is quarantined
    )
    assert not (attempt_root / "pane_gate_ready.json").exists()
    assert not (attempt_root / "pane_fault_consumer").exists()


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    "interruption",
    (
        None,
        "kill_ready",
        "kill_accepted",
        "kill_active",
        "kill_observed",
        "abort_offer",
        "abort_commit",
        "abort_intent",
    ),
)
def test_pane_fault_consumer_reserve_spawn_ready_live_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: str | None,
) -> None:
    launcher = _launcher_module()
    repo_root = Path(__file__).parents[1]
    config = (
        repo_root
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    launcher._install_verified_preflight_apis(config)
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir(mode=0o755)
    attempt_id = hashlib.sha256(
        b"receipt-attempt"
    ).hexdigest()
    policy_sha256 = hashlib.sha256(
        b"pane-fault-consumer-policy"
    ).hexdigest()
    gate_owner_nonce = hashlib.sha256(
        b"pane-fault-consumer-gate-owner"
    ).hexdigest()
    channel = launcher._create_fault_channel(
        attempt_root / "pane_gate_fault.channel"
    )
    publisher = {
        "path": str(Path(launcher.__file__).resolve()),
        "sha256": hashlib.sha256(
            Path(launcher.__file__).read_bytes()
        ).hexdigest(),
        "role": "launcher_pane_gate",
    }
    registration = (
        launcher._build_pane_fault_consumer_registration(
            attempt_root=attempt_root,
            launcher_binding={
                "path": publisher["path"],
                "sha256": publisher["sha256"],
            },
        )
    )
    consumer_attempt_path = Path(
        registration["artifacts"]["attempt"]
    )
    Path(registration["namespace"]).mkdir(mode=0o755)
    consumer_worker_arguments = [
        sys.executable,
        "-B",
        "-u",
        str(Path(launcher.__file__).resolve()),
        launcher.PANE_FAULT_CONSUMER_MODE,
        "--attempt-path",
        str(consumer_attempt_path),
        "--config",
        str(config),
    ]
    consumer_lifecycle_wait_channel = (
        launcher._create_fault_channel(
            Path(registration["artifacts"]["lifecycle_wait_channel"])
        )
    )
    consumer_supervisor_arguments = [
        sys.executable,
        "-B",
        "-u",
        str(Path(launcher.__file__).resolve()),
        launcher.CONSUMER_WAIT_SUPERVISOR_MODE,
        "--attempt-path",
        str(consumer_attempt_path),
        "--config",
        str(config),
        "--wait-channel-path",
        consumer_lifecycle_wait_channel["path"],
        "--consumer-worker-arguments-json",
        json.dumps(
            consumer_worker_arguments, separators=(",", ":")
        ),
    ]
    consumer_session = (
        launcher.PANE_FAULT_CONSUMER_SESSION_PREFIX + attempt_id
    )
    consumer_owner_nonce = hashlib.sha256(
        b"pane-fault-consumer-owner"
    ).hexdigest()
    receipt_path = attempt_root / "launch_receipt.json"
    receipt = _shared_launch_receipt_v4()
    receipt.update(
        {
        "policy_sha256": policy_sha256,
        "consumer_session": consumer_session,
        "consumer_owner_nonce": consumer_owner_nonce,
        "consumer_worker_arguments": consumer_worker_arguments,
        "consumer_lifecycle_wait_channel": (
            consumer_lifecycle_wait_channel
        ),
        "consumer_lifecycle_wait_publisher": {
            "path": publisher["path"],
            "sha256": publisher["sha256"],
            "file_identity": launcher._opened_file_identity(
                Path(publisher["path"])
            ),
            "role": "consumer_lifecycle_wait_supervisor",
        },
        "consumer_lifecycle_wait_supervisor_arguments": (
            consumer_supervisor_arguments
        ),
        "consumer_lifecycle_wait_supervisor_ready_path": (
            registration["artifacts"]["wait_supervisor_ready"]
        ),
        "consumer_lifecycle_wait_status_path": (
            registration["artifacts"]["lifecycle_wait_channel"]
        ),
        "consumer_tmux_arguments": [
            "tmux",
            "new-session",
            "-d",
            "-s",
            consumer_session,
            "-c",
            str(repo_root),
            "-e",
            f"{launcher.TMUX_OWNER_ENV}={consumer_owner_nonce}",
            *consumer_supervisor_arguments,
        ],
        "pane_fault_consumer": registration,
        }
    )
    receipt["bindings"]["config"]["path"] = str(config)
    receipt["launch_receipt_sha256"] = (
        launcher._canonical_digest(
            receipt, "launch_receipt_sha256"
        )
    )
    launcher._write_exclusive(receipt_path, receipt)
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            launcher.CONTROLLER_SESSION,
            "-e",
            f"{launcher.TMUX_OWNER_ENV}={gate_owner_nonce}",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        check=True,
    )
    consumer: dict[str, Any] | None = None
    launcher_gate_reader: dict[str, Any] | None = None
    try:
        gate_owner_seal = launcher._tmux_owner_seal(
            launcher.CONTROLLER_SESSION, gate_owner_nonce
        )
        consumer = (
            launcher._reserve_spawn_ready_pane_fault_consumer(
                repo_root=repo_root,
                config=config,
                attempt_root=attempt_root,
                policy_sha256=policy_sha256,
                attempt_id=attempt_id,
                receipt_path=receipt_path,
                receipt_identity=launcher._opened_file_identity(
                    receipt_path
                ),
                gate_owner_seal=gate_owner_seal,
                pane_fault_channel=channel,
                pane_fault_publisher=publisher,
                python=sys.executable,
                ready_timeout_seconds=5.0,
                registration=registration,
            )
        )
        assert consumer["ready"]["supervisor_process"]["pid"] == (
            consumer["owner_seal"]["pane_pid"]
        )
        assert consumer["ready"]["worker_process"]["ppid"] == (
            consumer["ready"]["supervisor_process"]["pid"]
        )
        assert (
            consumer["ready"]["worker_process"]["pid"]
            != consumer["ready"]["supervisor_process"]["pid"]
        )
        assert (
            consumer["ready"]["pane_fault_channel"]
            == channel
        )
        assert (
            consumer["attempt"]["gate_owner_seal"]
            == gate_owner_seal
        )
        assert (
            consumer["started"]["consumer_self_fault_channel"]
            == consumer["self_fault_channel"]
        )
        assert (
            subprocess.run(
                [
                    "tmux",
                    "show-window-options",
                    "-v",
                    "-t",
                    consumer["owner_seal"]["pane"],
                    "remain-on-exit",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == "on"
        )
        launcher_gate_reader = {
            "descriptor": launcher._open_presealed_fault_channel(
                attempt_root,
                channel,
                name="pane_gate_fault.channel",
            ),
            "closed": False,
        }
        original_wait = (
            launcher._wait_for_pane_fault_consumer_artifact
        )
        original_write = launcher._write_exclusive

        def kill_consumer() -> None:
            launcher._kill_exact_session(
                consumer["session"],
                consumer["owner_nonce"],
                consumer["owner_seal"],
            )

        def interrupt_wait(**kwargs):
            value = original_wait(**kwargs)
            if (
                interruption == "kill_accepted"
                and kwargs["label"] == "transfer accepted"
            ) or (
                interruption == "kill_active"
                and kwargs["label"] == "transfer active"
            ) or (
                interruption == "kill_observed"
                and kwargs["label"] == "reader release observed"
            ):
                kill_consumer()
            return value

        def interrupt_write(path, value):
            target = {
                "abort_offer": "offer",
                "abort_commit": "commit",
                "abort_intent": "reader_release",
            }.get(interruption)
            if (
                target is not None
                and Path(path)
                == Path(consumer["attempt"]["artifacts"][target])
            ):
                raise RuntimeError(
                    f"injected launcher abort before {target}"
                )
            return original_write(path, value)

        monkeypatch.setattr(
            launcher,
            "_wait_for_pane_fault_consumer_artifact",
            interrupt_wait,
        )
        monkeypatch.setattr(
            launcher, "_write_exclusive", interrupt_write
        )
        if interruption == "kill_ready":
            kill_consumer()
        if interruption is None:
            transfer = launcher._transfer_pane_fault_consumer(
                consumer=consumer,
                launcher_gate_reader=launcher_gate_reader,
                timeout_seconds=5.0,
            )
        else:
            with pytest.raises(RuntimeError):
                launcher._transfer_pane_fault_consumer(
                    consumer=consumer,
                    launcher_gate_reader=launcher_gate_reader,
                    timeout_seconds=5.0,
                )
            assert launcher_gate_reader["closed"] is False
            os.fstat(launcher_gate_reader["descriptor"])
            launcher._kill_exact_session(
                consumer["session"],
                consumer["owner_nonce"],
                consumer["owner_seal"],
            )
            assert (
                launcher._tmux_pane(consumer["session"]) is None
            )
            deadline = time.monotonic() + 5
            for process in (
                consumer["ready"]["supervisor_process"],
                consumer["ready"]["worker_process"],
            ):
                while True:
                    try:
                        live = launcher._process_identity(
                            int(process["pid"])
                        )
                    except (
                        FileNotFoundError,
                        ProcessLookupError,
                    ):
                        live = None
                    if live != process:
                        break
                    if time.monotonic() >= deadline:
                        raise AssertionError(
                            "consumer process survived exact kill"
                        )
                    time.sleep(0.01)
            boundary = {
                "kill_ready": 0,
                "abort_offer": 0,
                "kill_accepted": 2,
                "abort_commit": 2,
                "kill_active": 4,
                "abort_intent": 4,
                "kill_observed": 6,
            }[interruption]
            names = (
                "offer",
                "accepted",
                "commit",
                "active",
                "reader_release",
                "release_observed",
            )
            for name in names[boundary:]:
                assert not Path(
                    consumer["attempt"]["artifacts"][name]
                ).exists()
            assert not (
                attempt_root / "pane_gate_release.json"
            ).exists()
            return
        assert launcher_gate_reader["closed"] is True
        with pytest.raises(OSError):
            os.fstat(launcher_gate_reader["descriptor"])
        assert (
            transfer["reader_release"][
                "launcher_gate_reader_release_intent"
            ]
            is True
        )
        assert (
            transfer["release_observed"][
                "consumer_reader_release"
            ]
            == launcher._json_binding(
                transfer["reader_release_path"],
                "consumer_reader_release_sha256",
            )
        )
    finally:
        if (
            launcher_gate_reader is not None
            and not launcher_gate_reader["closed"]
        ):
            os.close(launcher_gate_reader["descriptor"])
        if consumer is not None:
            try:
                launcher._kill_exact_session(
                    consumer["session"],
                    consumer["owner_nonce"],
                    consumer["owner_seal"],
                )
            finally:
                os.close(
                    consumer["self_fault_reader_descriptor"]
                )
        subprocess.run(
            [
                "tmux",
                "kill-session",
                "-t",
                launcher.CONTROLLER_SESSION,
            ],
            capture_output=True,
            text=True,
        )


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux descriptor semantics",
)
def test_pane_fault_consumer_second_channel_open_closes_first_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    attempt_path = tmp_path / "consumer_attempt.json"
    attempt = {
        "schema_version": 1,
        "contract_type": "safa_pane_fault_consumer_attempt_v1",
        "consumer_log": {
            "path": str(tmp_path / "consumer.log"),
        },
        "pane_fault_channel": {
            "path": str(tmp_path / "pane_gate_fault.channel"),
        },
        "consumer_self_fault_channel": {
            "path": str(
                tmp_path / "consumer_self_fault.channel"
            ),
        },
    }
    attempt["consumer_attempt_sha256"] = (
        launcher._canonical_digest(
            attempt, "consumer_attempt_sha256"
        )
    )
    log_descriptor = os.open(os.devnull, os.O_WRONLY)
    gate_descriptor = os.open(os.devnull, os.O_RDONLY)
    opened = 0
    closed: list[int] = []
    original_close = os.close

    def open_channel(*_args, **_kwargs):
        nonlocal opened
        opened += 1
        if opened == 1:
            return gate_descriptor
        raise OSError("injected self-channel open failure")

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    monkeypatch.setattr(
        launcher,
        "_install_verified_preflight_apis",
        lambda _config: None,
    )
    monkeypatch.setattr(
        launcher,
        "_load_json",
        lambda _path, _label: attempt,
    )
    monkeypatch.setattr(
        launcher,
        "_open_consumer_log",
        lambda _path, _binding: log_descriptor,
    )
    monkeypatch.setattr(
        launcher, "_open_presealed_fault_channel", open_channel
    )
    monkeypatch.setattr(launcher.os, "dup2", lambda *_args: None)
    monkeypatch.setattr(launcher.os, "close", tracked_close)
    monkeypatch.setattr(
        launcher,
        "_write_exclusive",
        lambda *_args, **_kwargs: pytest.fail(
            "channel-open failure published an artifact"
        ),
    )

    with pytest.raises(
        OSError, match="injected self-channel open failure"
    ):
        launcher._pane_fault_consumer(
            attempt_path=attempt_path,
            config=tmp_path / "config.json",
        )
    assert opened == 2
    assert closed.count(log_descriptor) == 1
    assert closed.count(gate_descriptor) == 1
    with pytest.raises(OSError):
        os.fstat(gate_descriptor)


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_pane_fault_consumer_spawn_failure_keeps_evidence_and_cleans_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir(mode=0o755)
    attempt_id = hashlib.sha256(
        b"pane-fault-consumer-spawn-failure"
    ).hexdigest()
    namespace = attempt_root / "pane_fault_consumer"
    consumer_attempt_path = namespace / "consumer_attempt.json"
    config = tmp_path / "config.json"
    consumer_worker_arguments = [
        sys.executable,
        "-B",
        "-u",
        str(Path(launcher.__file__).resolve()),
        launcher.PANE_FAULT_CONSUMER_MODE,
        "--attempt-path",
        str(consumer_attempt_path),
        "--config",
        str(config),
    ]
    session = launcher.PANE_FAULT_CONSUMER_SESSION_PREFIX + attempt_id
    consumer_owner_nonce = hashlib.sha256(
        b"spawn-failure-consumer-owner"
    ).hexdigest()
    receipt_path = attempt_root / "launch_receipt.json"
    receipt = {
        "schema_version": 4,
        "contract_type": (
            "safa_canonical_preflight_launch_receipt_v4"
        ),
        "attempt_id": attempt_id,
        "consumer_session": session,
        "consumer_owner_nonce": consumer_owner_nonce,
        "consumer_worker_arguments": consumer_worker_arguments,
        "consumer_lifecycle_wait_channel": {
            "path": str(
                namespace / "consumer_lifecycle_wait.channel"
            )
        },
        "consumer_lifecycle_wait_publisher": {
            "role": "consumer_lifecycle_wait_supervisor"
        },
        "consumer_lifecycle_wait_supervisor_arguments": [
            "fixture-consumer-supervisor"
        ],
        "consumer_lifecycle_wait_supervisor_ready_path": str(
            namespace / "consumer_wait_supervisor_ready.json"
        ),
        "consumer_lifecycle_wait_status_path": str(
            namespace / "consumer_lifecycle_wait.channel"
        ),
        "consumer_tmux_arguments": ["fixture-tmux"],
    }
    receipt["launch_receipt_sha256"] = (
        launcher._canonical_digest(
            receipt, "launch_receipt_sha256"
        )
    )
    launcher._write_exclusive(receipt_path, receipt)
    channel = launcher._create_fault_channel(
        attempt_root / "pane_gate_fault.channel"
    )
    opened_readers: list[int] = []
    original_open = launcher._open_presealed_fault_channel

    def tracked_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        opened_readers.append(descriptor)
        return descriptor

    def spawn_then_fail(**kwargs):
        result = subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                kwargs["consumer_session"],
                "-e",
                (
                    f"{launcher.TMUX_OWNER_ENV}="
                    f"{kwargs['consumer_owner_nonce']}"
                ),
                "sleep",
                "30",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        raise RuntimeError("injected failure after consumer spawn")

    monkeypatch.setattr(
        launcher, "_open_presealed_fault_channel", tracked_open
    )
    monkeypatch.setattr(
        launcher, "_spawn_ready_pane_fault_consumer", spawn_then_fail
    )
    with pytest.raises(
        launcher.PaneFaultConsumerReservationError,
        match="injected failure after consumer spawn",
    ) as raised:
        launcher._reserve_spawn_ready_pane_fault_consumer(
            repo_root=Path(__file__).parents[1],
                config=config,
            attempt_root=attempt_root,
            policy_sha256="1" * 64,
            attempt_id=attempt_id,
            receipt_path=receipt_path,
            receipt_identity=launcher._opened_file_identity(
                receipt_path
            ),
            gate_owner_seal={"fixture": "gate-owner"},
            pane_fault_channel=channel,
            pane_fault_publisher={"role": "fixture"},
            python=sys.executable,
            ready_timeout_seconds=1.0,
            registration=(
                launcher._build_pane_fault_consumer_registration(
                    attempt_root=attempt_root,
                    launcher_binding={
                        "path": str(
                            Path(launcher.__file__).resolve()
                        ),
                        "sha256": hashlib.sha256(
                            Path(launcher.__file__).read_bytes()
                        ).hexdigest(),
                    },
                )
            ),
        )
    assert raised.value.secondary_failures == []
    assert launcher._tmux_pane(session) is None
    assert len(opened_readers) == 1
    with pytest.raises(OSError):
        os.fstat(opened_readers[0])
    namespace = attempt_root / "pane_fault_consumer"
    assert (namespace / "consumer_attempt.json").is_file()
    assert (namespace / "consumer_self_fault.channel").is_file()
    assert (namespace / "consumer.log").is_file()
    assert not (namespace / "consumer_ready.json").exists()
    assert not (namespace / "consumer_started.json").exists()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux descriptor semantics",
)
@pytest.mark.parametrize(
    "cutpoint",
    (
        "ready_offer",
        "offer_accepted",
        "accepted_commit",
        "commit_active",
        "active_intent",
        "intent_observed",
        "observed_close",
    ),
)
@pytest.mark.parametrize(
    "failure_kind", ("valid_fault", "partial_invalid", "consumer_abort")
)
def test_pane_fault_consumer_transfer_cutpoints_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cutpoint: str,
    failure_kind: str,
) -> None:
    launcher = _launcher_module()
    attempt_root = tmp_path / "attempt"
    namespace = attempt_root / "pane_fault_consumer"
    namespace.mkdir(parents=True, mode=0o755)
    attempt_root.chmod(0o755)
    namespace.chmod(0o755)
    attempt_id = hashlib.sha256(
        f"{cutpoint}:{failure_kind}".encode()
    ).hexdigest()
    policy_sha256 = "1" * 64
    gate_owner_nonce = "2" * 64
    consumer_owner_nonce = "3" * 64
    receipt_path = attempt_root / "launch_receipt.json"
    receipt = {
        "schema_version": 1,
        "contract_type": "fixture_launch_receipt",
        "attempt_id": attempt_id,
    }
    receipt["launch_receipt_sha256"] = (
        launcher._canonical_digest(
            receipt, "launch_receipt_sha256"
        )
    )
    launcher._write_exclusive(receipt_path, receipt)
    gate_channel = launcher._create_fault_channel(
        attempt_root / "pane_gate_fault.channel"
    )
    self_channel = launcher._create_fault_channel(
        namespace / "consumer_self_fault.channel"
    )
    gate_publisher = {
        "path": str(Path(launcher.__file__).resolve()),
        "sha256": hashlib.sha256(
            Path(launcher.__file__).read_bytes()
        ).hexdigest(),
        "role": "launcher_pane_gate",
    }
    self_publisher = {
        **gate_publisher,
        "role": "pane_fault_consumer",
    }
    artifacts = {
        name: str(namespace / f"consumer_{name}.json")
        for name in (
            "ready",
            "started",
            "offer",
            "accepted",
            "commit",
            "active",
            "reader_release",
            "release_observed",
            "controller_cleanup",
            "terminal",
            "join",
            "cleanup",
        )
    }
    attempt_path = namespace / "consumer_attempt.json"
    attempt = {
        "schema_version": 1,
        "contract_type": "safa_pane_fault_consumer_attempt_v1",
        "policy_sha256": policy_sha256,
        "attempt_id": attempt_id,
        "launch_receipt": launcher._json_binding(
            receipt_path, "launch_receipt_sha256"
        ),
        "launch_receipt_identity": (
            launcher._opened_file_identity(receipt_path)
        ),
        "gate_owner_seal": {
            "owner_nonce": gate_owner_nonce,
        },
        "pane_fault_channel": gate_channel,
        "pane_fault_publisher": gate_publisher,
        "consumer_self_fault_channel": self_channel,
        "consumer_self_fault_publisher": self_publisher,
        "consumer_session": "fixture-consumer",
        "consumer_owner_nonce": consumer_owner_nonce,
        "artifacts": artifacts,
    }
    attempt["consumer_attempt_sha256"] = (
        launcher._canonical_digest(
            attempt, "consumer_attempt_sha256"
        )
    )
    launcher._write_exclusive(attempt_path, attempt)
    ready = {
        "schema_version": 1,
        "contract_type": "safa_pane_fault_consumer_ready_v1",
        "supervisor_process": {
            "pid": 100,
            "start_ticks": 200,
        },
        "worker_process": {
            "pid": 101,
            "ppid": 100,
            "start_ticks": 202,
        },
    }
    ready["consumer_ready_sha256"] = (
        launcher._canonical_digest(
            ready, "consumer_ready_sha256"
        )
    )
    ready_path = Path(artifacts["ready"])
    launcher._write_exclusive(ready_path, ready)
    owner_seal = {
        "session": "fixture-consumer",
        "pane": "%fixture",
        "pane_pid": 100,
        "pane_dead": False,
        "pane_dead_status": "",
        "pane_process": ready["supervisor_process"],
        "owner_nonce": consumer_owner_nonce,
        "tmux_server": {"fixture": "server"},
    }
    started = {
        "schema_version": 1,
        "contract_type": "safa_pane_fault_consumer_started_v1",
        "owner_seal": owner_seal,
    }
    started["consumer_started_sha256"] = (
        launcher._canonical_digest(
            started, "consumer_started_sha256"
        )
    )
    started_path = Path(artifacts["started"])
    launcher._write_exclusive(started_path, started)
    gate_reader = launcher._open_presealed_fault_channel(
        attempt_root,
        gate_channel,
        name="pane_gate_fault.channel",
    )
    self_reader = launcher._open_presealed_fault_channel(
        namespace,
        self_channel,
        name="consumer_self_fault.channel",
    )
    consumer = {
        "attempt": attempt,
        "attempt_path": attempt_path,
        "ready": ready,
        "ready_path": ready_path,
        "started": started,
        "started_path": started_path,
        "owner_seal": owner_seal,
        "session": "fixture-consumer",
        "owner_nonce": consumer_owner_nonce,
        "self_fault_reader_descriptor": self_reader,
    }
    launcher_gate_reader = {
        "descriptor": gate_reader,
        "closed": False,
    }
    original_require = (
        launcher._require_empty_pane_fault_consumer_channels
    )
    injected = False

    def inject_failure() -> None:
        nonlocal injected
        if injected:
            pytest.fail("cutpoint failure injected more than once")
        injected = True
        if failure_kind == "consumer_abort":
            raise RuntimeError(
                "pane fault consumer changed at injected cutpoint"
            )
        writer = os.open(
            gate_channel["path"],
            os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            if failure_kind == "partial_invalid":
                os.pwrite(writer, b"partial-frame", 0)
                os.fsync(writer)
            else:
                publication_failure = (
                    launcher.LauncherExclusivePublishError(
                        "precommit_failed_clean",
                        "injected valid gate fault",
                        stage="fixture_cutpoint",
                        directory_seal={"fixture": 1},
                        payload={},
                        temporary=None,
                        error_number=None,
                        quarantined=False,
                    )
                )
                launcher._write_launcher_fault_channel_record(
                    writer,
                    gate_channel,
                    attempt_id=attempt_id,
                    owner_nonce=gate_owner_nonce,
                    launch_receipt_sha256=receipt[
                        "launch_receipt_sha256"
                    ],
                    publisher=gate_publisher,
                    failure=publication_failure,
                )
        finally:
            os.close(writer)
        original_require(
            attempt=attempt,
            fault_descriptor=gate_reader,
            self_fault_descriptor=self_reader,
        )
        pytest.fail("fault channel injection was accepted as empty")

    def stage_for_require() -> str | None:
        offer_exists = Path(artifacts["offer"]).exists()
        accepted_exists = Path(artifacts["accepted"]).exists()
        commit_exists = Path(artifacts["commit"]).exists()
        active_exists = Path(artifacts["active"]).exists()
        intent_exists = Path(artifacts["reader_release"]).exists()
        observed_exists = Path(
            artifacts["release_observed"]
        ).exists()
        if not offer_exists:
            return "ready_offer"
        if accepted_exists and not commit_exists:
            return "accepted_commit"
        if active_exists and not intent_exists:
            return "active_intent"
        if observed_exists:
            return "observed_close"
        return None

    def guarded_require(**kwargs):
        if not injected and stage_for_require() == cutpoint:
            inject_failure()
        return original_require(**kwargs)

    def fake_wait(*, path, label, **_kwargs):
        if (
            not injected
            and (
                (
                    cutpoint == "offer_accepted"
                    and label == "transfer accepted"
                )
                or (
                    cutpoint == "commit_active"
                    and label == "transfer active"
                )
                or (
                    cutpoint == "intent_observed"
                    and label == "reader release observed"
                )
            )
        ):
            inject_failure()
        if label == "transfer accepted":
            offer_path = Path(artifacts["offer"])
            value = {
                "schema_version": 1,
                "contract_type": (
                    "safa_pane_fault_consumer_transfer_accepted_v1"
                ),
                "policy_sha256": policy_sha256,
                "attempt_id": attempt_id,
                "consumer_attempt": launcher._json_binding(
                    attempt_path, "consumer_attempt_sha256"
                ),
                "consumer_ready": launcher._json_binding(
                    ready_path, "consumer_ready_sha256"
                ),
                "consumer_started": launcher._json_binding(
                    started_path, "consumer_started_sha256"
                ),
                "consumer_offer": launcher._json_binding(
                    offer_path, "consumer_offer_sha256"
                ),
                "consumer_session": "fixture-consumer",
                "consumer_owner_nonce": consumer_owner_nonce,
                    "owner_seal": owner_seal,
                    "supervisor_process": ready[
                        "supervisor_process"
                    ],
                    "worker_process": ready["worker_process"],
                "pane_fault_channel": gate_channel,
                "consumer_self_fault_channel": self_channel,
                "accepted_at": launcher._utc_now(),
            }
            digest_field = "consumer_accepted_sha256"
        elif label == "transfer active":
            commit_path = Path(artifacts["commit"])
            commit = launcher._load_json(
                commit_path, "fixture commit"
            )
            value = {
                "schema_version": 1,
                "contract_type": (
                    "safa_pane_fault_consumer_transfer_active_v1"
                ),
                "policy_sha256": policy_sha256,
                "attempt_id": attempt_id,
                "consumer_attempt": launcher._json_binding(
                    attempt_path, "consumer_attempt_sha256"
                ),
                "consumer_accepted": commit[
                    "consumer_accepted"
                ],
                "consumer_commit": launcher._json_binding(
                    commit_path, "consumer_commit_sha256"
                ),
                "consumer_session": "fixture-consumer",
                "consumer_owner_nonce": consumer_owner_nonce,
                    "owner_seal": owner_seal,
                    "supervisor_process": ready[
                        "supervisor_process"
                    ],
                    "worker_process": ready["worker_process"],
                "pane_fault_channel": gate_channel,
                "consumer_self_fault_channel": self_channel,
                "active_at": launcher._utc_now(),
            }
            digest_field = "consumer_active_sha256"
        else:
            release_path = Path(artifacts["reader_release"])
            release = launcher._load_json(
                release_path, "fixture reader release"
            )
            value = {
                "schema_version": 1,
                "contract_type": (
                    "safa_pane_fault_consumer_release_observed_v1"
                ),
                "policy_sha256": policy_sha256,
                "attempt_id": attempt_id,
                "consumer_attempt": launcher._json_binding(
                    attempt_path, "consumer_attempt_sha256"
                ),
                "consumer_active": release["consumer_active"],
                "consumer_reader_release": launcher._json_binding(
                    release_path,
                    "consumer_reader_release_sha256",
                ),
                "consumer_session": "fixture-consumer",
                "consumer_owner_nonce": consumer_owner_nonce,
                    "owner_seal": owner_seal,
                    "supervisor_process": ready[
                        "supervisor_process"
                    ],
                    "worker_process": ready["worker_process"],
                "release_observed_at": launcher._utc_now(),
            }
            digest_field = "consumer_release_observed_sha256"
        value[digest_field] = launcher._canonical_digest(
            value, digest_field
        )
        launcher._write_exclusive(path, value)
        return value

    monkeypatch.setattr(
        launcher,
        "_require_empty_pane_fault_consumer_channels",
        guarded_require,
    )
    monkeypatch.setattr(
        launcher,
        "_wait_for_pane_fault_consumer_artifact",
        fake_wait,
    )
    monkeypatch.setattr(
        launcher,
        "_require_live_pane_fault_consumer",
        lambda *_args, **_kwargs: None,
    )
    try:
        expected = (
            launcher.LauncherGateFaultError
            if failure_kind != "consumer_abort"
            else RuntimeError
        )
        with pytest.raises(expected) as raised:
            launcher._transfer_pane_fault_consumer(
                consumer=consumer,
                launcher_gate_reader=launcher_gate_reader,
                timeout_seconds=1.0,
            )
        assert injected is True
        if failure_kind == "valid_fault":
            assert (
                raised.value.status
                == "pane_gate_typed_publish_failure"
            )
        elif failure_kind == "partial_invalid":
            assert (
                raised.value.status
                == "pane_gate_fault_channel_invalid"
            )
        assert launcher_gate_reader["closed"] is False
        os.fstat(gate_reader)
        stage_order = (
            "offer",
            "accepted",
            "commit",
            "active",
            "reader_release",
            "release_observed",
        )
        boundary = {
            "ready_offer": 0,
            "offer_accepted": 1,
            "accepted_commit": 2,
            "commit_active": 3,
            "active_intent": 4,
            "intent_observed": 5,
            "observed_close": 6,
        }[cutpoint]
        for name in stage_order[boundary:]:
            assert not Path(artifacts[name]).exists()
        assert not (attempt_root / "pane_gate_release.json").exists()
    finally:
        os.close(gate_reader)
        os.close(self_reader)


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_preflight_cleanup_refuses_replaced_session(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    nonce = hashlib.sha256(b"cleanup-owner").hexdigest()
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            launcher.CONTROLLER_SESSION,
            "-e",
            f"{launcher.TMUX_OWNER_ENV}={nonce}",
            "sleep",
            "30",
        ],
        check=True,
    )
    seal = launcher._tmux_owner_seal(
        launcher.CONTROLLER_SESSION, nonce
    )
    subprocess.run(
        [
            "tmux",
            "kill-session",
            "-t",
            launcher.CONTROLLER_SESSION,
        ],
        check=True,
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            launcher.CONTROLLER_SESSION,
            "-e",
            f"{launcher.TMUX_OWNER_ENV}={nonce}",
            "sleep",
            "30",
        ],
        check=True,
    )
    try:
        with pytest.raises(
            RuntimeError, match="refuses to kill a replaced tmux owner"
        ):
            launcher._kill_exact_session(
                launcher.CONTROLLER_SESSION, nonce, seal
            )
        assert launcher._tmux_pane(launcher.CONTROLLER_SESSION) is not None
    finally:
        subprocess.run(
            [
                "tmux",
                "kill-session",
                "-t",
                launcher.CONTROLLER_SESSION,
            ],
            capture_output=True,
            text=True,
        )


def test_cpu_preflight_request_manifest_binds_plan_and_request_files(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", "a" * 64)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    plan = build_checkpoint_plan(tmp_path, policy, paths["preflight_results"])
    write_exclusive_json(paths["checkpoint_plan"], plan)
    request_paths = module.write_preflight_requests(
        plan, paths["preflight_requests"]
    )
    manifest = module._build_preflight_request_manifest(
        policy, paths, plan, request_paths
    )
    assert manifest["request_count"] == 1
    assert (
        module._validate_preflight_request_manifest(
            manifest, policy, paths
        )
        == manifest
    )
    request_path = request_paths[0]
    request_path.write_bytes(request_path.read_bytes() + b" ")
    with pytest.raises(
        CanonicalScreeningError, match="file binding mismatch"
    ):
        module._validate_preflight_request_manifest(
            manifest, policy, paths
        )


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="requires Linux openat and O_NOFOLLOW",
)
def test_preflight_exact_owned_entry_rejects_path_aliases_and_inode_spoof(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    exact_root = tmp_path / "campaign/preflight_launch_attempts/attempt"
    exact_root.mkdir(parents=True)
    receipt_path = exact_root / "launch_receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    identity = module._require_exact_owned_regular_entry(
        str(receipt_path),
        receipt_path,
        "fixture receipt",
    )
    assert identity["path"] == str(receipt_path)
    with pytest.raises(
        CanonicalScreeningError, match="exact path differs"
    ):
        module._require_exact_owned_regular_entry(
            str(exact_root / "nested/../launch_receipt.json"),
            receipt_path,
            "fixture receipt",
        )
    spoofed_identity = dict(identity)
    spoofed_identity["inode"] += 1
    with pytest.raises(
        CanonicalScreeningError, match="inode identity differs"
    ):
        module._require_exact_owned_regular_entry(
            str(receipt_path),
            receipt_path,
            "fixture receipt",
            identity=spoofed_identity,
        )
    real_parent = tmp_path / "real-attempt"
    real_parent.mkdir()
    (real_parent / "launch_receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    alias_parent = tmp_path / "alias-attempt"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    alias_receipt = alias_parent / "launch_receipt.json"
    with pytest.raises(
        CanonicalScreeningError, match="non-symlink directory"
    ):
        module._require_exact_owned_regular_entry(
            str(alias_receipt),
            alias_receipt,
            "fixture receipt",
        )


def test_preflight_two_layer_process_contract_rejects_old_pane_wrapper_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _controller_module()
    gate_pid = 101
    wrapper_pid = 202
    gate_arguments = ["/sealed/python", "gate"]
    wrapper_arguments = ["/sealed/python", "wrapper"]
    gate = {
        "pid": gate_pid,
        "ppid": 1,
        "pgid": gate_pid,
        "sid": gate_pid,
        "start_ticks": 1001,
    }
    child = {
        "pid": wrapper_pid,
        "ppid": gate_pid,
        "pgid": wrapper_pid,
        "sid": wrapper_pid,
        "start_ticks": 1002,
    }
    receipt = {
        "pane_gate_arguments": gate_arguments,
        "wrapper_arguments": wrapper_arguments,
        "python_executable": {"path": "/sealed/python"},
    }
    wrapper = {
        "pane_gate_process": gate,
        "wrapper_launch_process": child,
        "wrapper_executable": build_preflight_file_identity(
            path="/sealed/python",
            device=1,
            inode=2,
            mode=0o100755,
            size=1,
        ),
    }
    identities = {gate_pid: gate, wrapper_pid: child}
    commands = {
        gate_pid: module._command_bytes(gate_arguments),
        wrapper_pid: module._command_bytes(wrapper_arguments),
    }
    monkeypatch.setattr(
        module,
        "_launch_process_identity",
        lambda pid: dict(identities[pid]),
    )
    monkeypatch.setattr(
        module, "_process_command_bytes", lambda pid: commands[pid]
    )
    monkeypatch.setattr(
        module.os,
        "readlink",
        lambda path: "/sealed/python"
        if path in {f"/proc/{gate_pid}/exe", f"/proc/{wrapper_pid}/exe"}
        else (_ for _ in ()).throw(AssertionError(path)),
    )
    module._validate_live_preflight_process_layers(
        wrapper, receipt, "fixture"
    )
    old_single_process = dict(wrapper)
    old_single_process["wrapper_launch_process"] = dict(gate)
    with pytest.raises(
        CanonicalScreeningError,
        match="gate/wrapper exact process seal differs",
    ):
        module._validate_live_preflight_process_layers(
            old_single_process, receipt, "fixture"
        )
    wrong_parent = dict(wrapper)
    wrong_parent["wrapper_launch_process"] = {
        **child,
        "ppid": gate_pid + 1,
    }
    with pytest.raises(
        CanonicalScreeningError,
        match="gate/wrapper exact process seal differs",
    ):
        module._validate_live_preflight_process_layers(
            wrong_parent, receipt, "fixture"
        )
    commands[wrapper_pid] = b"/sealed/python\0different-wrapper\0"
    with pytest.raises(
        CanonicalScreeningError,
        match="gate/wrapper exact process seal differs",
    ):
        module._validate_live_preflight_process_layers(
            wrapper, receipt, "fixture"
        )


def test_preflight_tmux_assertions_never_bind_pane_to_wrapper_process() -> None:
    module = _raw_controller_module()
    tree = ast.parse(inspect.getsource(module))
    violations = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_assert_tmux_process_identity"
            and len(node.args) >= 4
        ):
            for child in ast.walk(node.args[3]):
                if (
                    isinstance(child, ast.Constant)
                    and child.value == "wrapper_process"
                ):
                    violations.append(node.lineno)
    assert violations == []


def test_cpu_preflight_monitor_dispatch_never_uses_gpu_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _controller_module()
    called = []
    monkeypatch.setattr(
        module,
        "_run_preflight_monitor",
        lambda *_args: called.append("cpu") or {"samples": 1},
    )
    monkeypatch.setattr(
        module,
        "_run_gpu_monitor",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("GPU observer was called")
        ),
    )
    assert module._run_monitor({}, {}, "preflight") == {"samples": 1}
    assert called == ["cpu"]
    with pytest.raises(
        CanonicalScreeningError, match="target is invalid"
    ):
        module._run_monitor({}, {}, "not-a-phase")


def test_preflight_artifact_progress_closes_all_193_requests(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    paths = module._paths(tmp_path / "campaign", "1" * 64)
    attempts = paths["preflight_control"] / "attempts"
    for index in range(193):
        stem = f"{index:064x}__raw"
        write_exclusive_json(
            attempts / f"{stem}.claim.json",
            {"sequence": index + 1},
        )
        write_exclusive_json(
            paths["preflight_results"] / f"{stem}.json",
            {"valid": index % 2 == 0},
        )
        write_exclusive_json(
            attempts / f"{stem}.terminal.json",
            {
                "status": "completed",
                "valid": index % 2 == 0,
            },
        )
    assert module._preflight_progress(paths, 193) == {
        "request_count": 193,
        "result_count": 193,
        "attempt_claim_count": 193,
        "attempt_terminal_count": 193,
        "completed": 193,
        "failed": 0,
        "valid": 97,
        "invalid": 96,
        "pending": 0,
    }


def test_preflight_observer_early_exit_fails_ready_barrier(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    write_exclusive_json(
        paths["preflight_control"] / "observer_terminal.json",
        {
            "contract_type": (
                "safa_canonical_preflight_observer_terminal_v1"
            ),
            "failure": {
                "type": "RuntimeError",
                "message": "observer exited",
            },
        },
    )
    with pytest.raises(
        CanonicalScreeningError, match="terminated before ready"
    ):
        module._wait_preflight_observer_ready(
            policy,
            paths,
            {
                "controller_ready_sha256": "a" * 64,
                "request_count": 1,
            },
        )


def test_preflight_tmux_is_launcher_managed_without_external_monitor(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    commands = module._tmux_commands(
        policy, policy_path, tmp_path / "campaign", "preflight"
    )
    assert commands["monitor"] == []
    assert any(
        item.endswith("run_canonical_preflight_launcher.py")
        for item in commands["controller"]
    )
    assert not any(
        item.endswith("run_canonical_preflight_wrapper.py")
        for item in commands["controller"]
    )


def test_cpu_preflight_monitor_completes_without_gpu_control_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("candidate", "a" * 64)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    policy["policy_file"] = {
        "path": str(policy_path.resolve()),
        "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    plan = build_checkpoint_plan(tmp_path, policy, paths["preflight_results"])
    write_exclusive_json(paths["checkpoint_plan"], plan)
    request_paths = module.write_preflight_requests(
        plan, paths["preflight_requests"]
    )
    manifest = module._build_preflight_request_manifest(
        policy, paths, plan, request_paths
    )
    control = paths["preflight_control"]
    sealed_pid = os.getpid()
    sealed_process = _test_process_identity(
        sealed_pid,
        ppid=os.getppid(),
        pgid=sealed_pid,
        sid=sealed_pid,
        start_ticks=20,
    )
    gate_pid = sealed_pid + 10000
    wrapper_pid = sealed_pid + 20000
    controller_tmux = {
        "session": module.PREFLIGHT_CONTROLLER_SESSION,
        "pane": "%0",
        "pane_pid": gate_pid,
        "pane_current_command": "python",
    }
    observer_tmux = {
        **controller_tmux,
        "session": module.PREFLIGHT_OBSERVER_SESSION,
        "pane": "%1",
        "pane_pid": sealed_pid,
    }
    tmux_server = _test_tmux_server_identity(
        tmp_path / "tmux.sock",
        server_pid=sealed_pid,
        server_process=sealed_process,
    )
    wrapper, wrapper_path, wrapper_seals = (
        _write_preflight_wrapper_v3_fixture(
            module,
            policy,
            paths,
            gate_pid=gate_pid,
            wrapper_pid=wrapper_pid,
            controller_tmux=controller_tmux,
            tmux_server=tmux_server,
        )
    )
    observer_launch, observer_launch_path = (
        _write_preflight_observer_provenance_fixture(
            module,
            policy,
            paths,
            wrapper=wrapper,
            wrapper_path=wrapper_path,
            observer_tmux=observer_tmux,
            tmux_server=tmux_server,
            observer_process=sealed_process,
        )
    )
    process_start, process_start_path = (
        _write_preflight_process_start_fixture(module, policy, paths)
    )
    controller_claim = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_controller_claim_v2",
        "policy_sha256": policy["policy_sha256"],
    }
    controller_claim["controller_claim_sha256"] = canonical_digest(
        controller_claim, "controller_claim_sha256"
    )
    controller_claim_path = control / "controller_claim.json"
    write_exclusive_json(controller_claim_path, controller_claim)
    admission = module._write_admission(
        policy,
        paths,
        "preflight",
        _admission_snapshot(policy),
    )
    controller_ready = {
        "schema_version": 1,
        "contract_type": "safa_canonical_preflight_controller_ready_v1",
        "policy_sha256": policy["policy_sha256"],
        "verified_implementations": (
            _test_verified_preflight_implementations()
        ),
        "controller_session": module.PREFLIGHT_CONTROLLER_SESSION,
        "observer_session": module.PREFLIGHT_OBSERVER_SESSION,
        "request_count": 1,
        "controller_pid": 100,
        "controller_process": process_start["process"],
        "controller_claim": module._artifact_binding(
            controller_claim_path,
            controller_claim["controller_claim_sha256"],
        ),
        "observer_launch": module._artifact_binding(
            observer_launch_path,
            observer_launch["observer_launch_sha256"],
        ),
        "controller_process_start": module._artifact_binding(
            process_start_path,
            process_start["controller_process_start_sha256"],
        ),
        "checkpoint_plan": module._artifact_binding(
            paths["checkpoint_plan"], plan["checkpoint_plan_sha256"]
        ),
        "preflight_request_manifest": module._artifact_binding(
            paths["preflight_request_manifest"],
            manifest["preflight_request_manifest_sha256"],
        ),
        "startup_admission": admission,
    }
    controller_ready["controller_ready_sha256"] = canonical_digest(
        controller_ready, "controller_ready_sha256"
    )
    controller_ready_path = control / "controller_ready.json"
    write_exclusive_json(controller_ready_path, controller_ready)
    request_stem = request_paths[0].stem
    write_exclusive_json(
        control / "attempts" / f"{request_stem}.claim.json",
        {"sequence": 1},
    )
    write_exclusive_json(
        paths["preflight_results"] / request_paths[0].name,
        {"valid": True},
    )
    write_exclusive_json(
        control / "attempts" / f"{request_stem}.terminal.json",
        {"status": "completed", "valid": True},
    )
    progress = module._preflight_progress(paths, 1)
    controller_terminal = {
        "schema_version": 1,
        "contract_type": (
            "safa_canonical_preflight_controller_terminal_v2"
        ),
        "policy_sha256": policy["policy_sha256"],
        "status": "completed",
        "failure": None,
        "progress": progress,
    }
    controller_terminal["controller_terminal_sha256"] = canonical_digest(
        controller_terminal, "controller_terminal_sha256"
    )
    controller_terminal_path = control / "controller_terminal.json"
    write_exclusive_json(controller_terminal_path, controller_terminal)
    process_exit, _ = _write_preflight_process_exit_fixture(
        module,
        policy,
        paths,
        wrapper=wrapper,
        observer_launch=observer_launch,
        observer_launch_path=observer_launch_path,
        process_start=process_start,
        process_start_path=process_start_path,
        exit_code=0,
        controller_terminal={
            "path": str(controller_terminal_path.resolve()),
            "sha256": hashlib.sha256(
                controller_terminal_path.read_bytes()
            ).hexdigest(),
        },
    )

    class FakeGuard:
        def __init__(
            self,
            _policy: dict,
            sample_path: Path,
            _disk_path: Path,
            authorized_gpu_registry: list[dict],
        ) -> None:
            self.sample_path = sample_path
            self.policy_sha256 = _policy["policy_sha256"]
            self.authorized_gpu_registry = authorized_gpu_registry

        def start(self) -> None:
            sample = {
                "schema_version": 1,
                "contract_type": (
                    "safa_canonical_runtime_resource_window_v1"
                ),
                "policy_sha256": self.policy_sha256,
                "sequence": 1,
                "violated": False,
            }
            sample["resource_window_sha256"] = canonical_digest(
                sample, "resource_window_sha256"
            )
            _write_jsonl(self.sample_path, [sample])

        def wait_first_sample(self, _timeout: float) -> dict:
            return module.load_jsonl(self.sample_path, "resource")[0]

        def raise_if_violated(self) -> None:
            return None

        def stop(self) -> dict:
            return {
                "started": True,
                "violated": False,
                "violation_reason": None,
                "thread_failure": None,
            }

    monkeypatch.setattr(
        module, "_current_tmux_session", lambda *_args: "monitor"
    )
    monkeypatch.setattr(
        module,
        "_tmux_identity",
        lambda session: {
            "session": session,
            "pane": "%0",
            "pane_pid": sealed_pid,
            "pane_current_command": "python",
        },
    )
    monkeypatch.setattr(
        module,
        "_tmux_pane_identity",
        lambda pane: (
            dict(controller_tmux)
            if pane == controller_tmux["pane"]
            else dict(observer_tmux)
        ),
    )
    monkeypatch.setattr(
        module,
        "_tmux_server_identity",
        lambda _target: dict(tmux_server),
    )
    monkeypatch.setattr(
        module, "_validate_tmux_owner_seal", lambda *_args: None
    )
    monkeypatch.setattr(
        module,
        "_process_identity",
        lambda pid: (
            dict(wrapper_seals["gate"])
            if pid == gate_pid
            else dict(sealed_process)
            if pid == sealed_pid
            else _test_process_identity(pid, start_ticks=20)
        ),
    )
    _mock_preflight_wrapper_process_seals(
        module, monkeypatch, wrapper_seals
    )
    monkeypatch.setattr(module, "RuntimeResourceGuard", FakeGuard)
    monkeypatch.setattr(
        module,
        "_hold_preflight_observer_for_wrapper_close",
        lambda: None,
    )
    result = module._run_preflight_monitor(policy, paths)
    assert result["samples"] == 2
    observer_terminal = load_json(
        control / "observer_terminal.json", "observer terminal"
    )
    assert observer_terminal["status"] == "completed"
    assert observer_terminal["controller_terminal"] is not None
    assert observer_terminal["controller_process_exit"] is not None
    assert not paths["gpu_control"].exists()


@pytest.mark.parametrize("mutation", ["session", "pid"])
def test_preflight_wrapper_wrong_session_or_pid_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    policy["policy_file"] = {
        "path": str(policy_path.resolve()),
        "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    parent_pid = os.getppid()
    wrapper_pid = parent_pid + (1 if mutation == "pid" else 0)
    gate_pid = max(1, parent_pid - 1)
    wrapper_process = {
        "pid": wrapper_pid,
        "ppid": gate_pid,
        "pgid": wrapper_pid,
        "sid": wrapper_pid,
        "start_ticks": 2,
    }
    wrapper = _shared_launch_contract_values()["claim"]
    wrapper.update(
        {
            "policy_sha256": policy["policy_sha256"],
            "config": policy["policy_file"],
            "controller_session": (
                "wrong"
                if mutation == "session"
                else module.PREFLIGHT_CONTROLLER_SESSION
            ),
            "controller_tmux": {
                "session": module.PREFLIGHT_CONTROLLER_SESSION,
                "pane": "%0",
                "pane_pid": gate_pid,
                "pane_current_command": "python",
            },
            "observer_session": module.PREFLIGHT_OBSERVER_SESSION,
            "command": module._expected_preflight_controller_command(
                policy, paths
            ),
            "observer_command": module._expected_preflight_observer_command(
                policy, paths
            ),
            "wrapper_pid": wrapper_pid,
            "wrapper_process": wrapper_process,
            "wrapper_launch_process": dict(wrapper_process),
            "pane_gate_process": {
                "pid": gate_pid,
                "ppid": 1,
                "pgid": gate_pid,
                "sid": gate_pid,
                "start_ticks": 1,
            },
        }
    )
    wrapper["wrapper_claim_sha256"] = canonical_digest(
        wrapper, "wrapper_claim_sha256"
    )
    write_exclusive_json(
        paths["preflight_control"] / "wrapper_claim.json", wrapper
    )
    with pytest.raises(
        CanonicalScreeningError, match="wrapper claim contract mismatch"
    ):
        module._validate_preflight_wrapper_provenance(policy, paths)


def test_preflight_controller_exit_without_terminal_writes_failed_observer_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    policy["policy_file"] = {
        "path": str(policy_path.resolve()),
        "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    }
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    control = paths["preflight_control"]
    sealed_pid = os.getpid()
    sealed_process = _test_process_identity(
        sealed_pid,
        ppid=os.getppid(),
        pgid=sealed_pid,
        sid=sealed_pid,
        start_ticks=20,
    )
    gate_pid = sealed_pid + 10000
    wrapper_pid = sealed_pid + 20000
    tmux_server = _test_tmux_server_identity(
        tmp_path / "tmux.sock",
        server_pid=sealed_pid,
        server_process=sealed_process,
    )
    controller_tmux = {
        "session": module.PREFLIGHT_CONTROLLER_SESSION,
        "pane": "%0",
        "pane_pid": gate_pid,
        "pane_current_command": "python",
    }
    wrapper, wrapper_path, wrapper_seals = (
        _write_preflight_wrapper_v3_fixture(
            module,
            policy,
            paths,
            gate_pid=gate_pid,
            wrapper_pid=wrapper_pid,
            controller_tmux=controller_tmux,
            tmux_server=tmux_server,
        )
    )
    observer_tmux = {
        "session": module.PREFLIGHT_OBSERVER_SESSION,
        "pane": "%1",
        "pane_pid": sealed_pid,
        "pane_current_command": "python",
    }
    launch, launch_path = _write_preflight_observer_provenance_fixture(
        module,
        policy,
        paths,
        wrapper=wrapper,
        wrapper_path=wrapper_path,
        observer_tmux=observer_tmux,
        tmux_server=tmux_server,
        observer_process=sealed_process,
    )
    process_start, process_start_path = (
        _write_preflight_process_start_fixture(module, policy, paths)
    )
    _write_preflight_process_exit_fixture(
        module,
        policy,
        paths,
        wrapper=wrapper,
        observer_launch=launch,
        observer_launch_path=launch_path,
        process_start=process_start,
        process_start_path=process_start_path,
        exit_code=2,
        controller_terminal=None,
    )
    monkeypatch.setattr(
        module, "_current_tmux_session", lambda *_args: "monitor"
    )
    monkeypatch.setattr(
        module,
        "_process_identity",
        lambda pid: (
            dict(wrapper_seals["gate"])
            if pid == gate_pid
            else dict(sealed_process)
            if pid == sealed_pid
            else _test_process_identity(pid, start_ticks=20)
        ),
    )
    monkeypatch.setattr(
        module,
        "_tmux_identity",
        lambda session: {
            "session": session,
            "pane": "%0",
            "pane_pid": sealed_pid,
            "pane_current_command": "python",
        },
    )
    monkeypatch.setattr(
        module,
        "_tmux_pane_identity",
        lambda pane: (
            dict(controller_tmux)
            if pane == "%0"
            else {
                "session": module.PREFLIGHT_OBSERVER_SESSION,
                "pane": pane,
                "pane_pid": sealed_pid,
                "pane_current_command": "python",
            }
        ),
    )
    monkeypatch.setattr(
        module,
        "_tmux_server_identity",
        lambda _target: dict(tmux_server),
    )
    monkeypatch.setattr(
        module, "_validate_tmux_owner_seal", lambda *_args: None
    )
    _mock_preflight_wrapper_process_seals(
        module, monkeypatch, wrapper_seals
    )
    monkeypatch.setattr(
        module,
        "_hold_preflight_observer_for_wrapper_close",
        lambda: None,
    )
    with pytest.raises(
        CanonicalScreeningError, match="exited before ready"
    ):
        module._run_preflight_monitor(policy, paths)
    terminal = load_json(
        control / "observer_terminal.json", "observer terminal"
    )
    assert terminal["status"] == "failed"
    assert terminal["controller_terminal"] is None


def test_preflight_observer_provenance_timeout_writes_durable_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    times = iter([0.0, module.PREFLIGHT_BARRIER_TIMEOUT_SECONDS + 1.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        module, "_current_tmux_session", lambda *_args: "monitor"
    )
    monkeypatch.setattr(
        module,
        "_process_identity",
        lambda pid: {"pid": pid, "pgid": pid, "start_ticks": 1},
    )
    monkeypatch.setattr(
        module,
        "_tmux_identity",
        lambda session: {
            "session": session,
            "pane": "%0",
            "pane_pid": os.getpid(),
            "pane_current_command": "python",
        },
    )
    with pytest.raises(
        CanonicalScreeningError, match="provenance barrier timed out"
    ):
        module._run_preflight_monitor(policy, paths)
    terminal = load_json(
        paths["preflight_control"] / "observer_terminal.json",
        "observer terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["failure"]["type"] == "CanonicalScreeningError"


def test_preflight_observer_resource_stop_hard_stops_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monkeypatch.setattr(
        module,
        "_process_identity",
        lambda pid: {"pid": pid, "pgid": pid, "start_ticks": 1},
    )
    monkeypatch.setattr(
        module,
        "_tmux_identity",
        lambda session: {
            "session": session,
            "pane": "%0",
            "pane_pid": os.getpid(),
            "pane_current_command": "python",
        },
    )
    stop = module._publish_preflight_observer_stop(
        policy,
        paths,
        None,
        {
            "type": "CanonicalScreeningError",
            "message": "RAM runtime hard stop: 90.00% >= 90%",
        },
    )
    assert stop["contract_type"] == "safa_canonical_preflight_observer_stop_v2"
    assert stop["observer_process"]["start_ticks"] == 1
    assert stop["controller_process"] is None
    assert stop["observer_stop_sha256"] == canonical_digest(
        stop, "observer_stop_sha256"
    )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    (
        "observer_mode",
        "controller_seconds",
        "controller_exit",
        "expected_exit",
        "expected_cleanup",
    ),
    (
        ("success", 0.2, 0, 0, True),
        ("stop", 30.0, 0, 143, True),
        ("timeout", 0.2, 0, 124, True),
        ("failure", 0.1, 2, 2, True),
    ),
)
def test_preflight_wrapper_real_tmux_subprocess_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observer_mode: str,
    controller_seconds: float,
    controller_exit: int,
    expected_exit: int,
    expected_cleanup: bool,
) -> None:
    wrapper = _wrapper_module()
    launcher = _launcher_module()
    repo_root = Path(__file__).parents[1]
    helper = repo_root / "tests/helpers/preflight_lifecycle_helper.py"
    policy_sha256 = hashlib.sha256(
        f"integration:{observer_mode}".encode()
    ).hexdigest()
    controller_owner_nonce = hashlib.sha256(
        f"integration-owner:{observer_mode}".encode()
    ).hexdigest()
    policy_root = tmp_path / "campaign" / "by_policy" / policy_sha256
    config = repo_root / "configs/closeout/canonical_screening_512_v1.json"
    _prepare_wrapper_contract_inputs(wrapper, policy_root)
    sessions = (
        wrapper.CONTROLLER_SESSION,
        wrapper.OBSERVER_SESSION,
    )
    for session in sessions:
        assert (
            subprocess.run(
                ["tmux", "has-session", "-t", session],
                capture_output=True,
                text=True,
            ).returncode
            != 0
        )
    command = [
        sys.executable,
        str(helper),
        "wrapper",
        "--wrapper-module",
        str(repo_root / "scripts/run_canonical_preflight_wrapper.py"),
        "--repo-root",
        str(repo_root),
        "--policy-root",
        str(policy_root),
        "--policy",
        policy_sha256,
        "--config",
        str(config),
        "--observer-mode",
        observer_mode,
        "--controller-seconds",
        str(controller_seconds),
        "--controller-exit",
        str(controller_exit),
        "--terminal-timeout",
        "1.0",
        "--defer-pane-fault-consumer-cleanup",
        "--controller-owner-nonce",
        controller_owner_nonce,
    ]
    started = time.monotonic()
    consumer_attempt_path: Path | None = None
    consumer_session: str | None = None
    consumer_owner_nonce: str | None = None
    gate_owner_seal: dict[str, Any] | None = None
    monkeypatch.setattr(
        launcher,
        "_verified_git_state",
        lambda root: {
            "head_sha": subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "origin_master_sha": subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "rev-parse",
                    "origin/master",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "branch": "master",
        },
    )
    try:
        launcher.launch_preflight(
            repo_root=repo_root,
            config=config,
            campaign_root=tmp_path / "campaign",
            policy_sha256=policy_sha256,
            python=sys.executable,
            startup_timeout_seconds=10,
            attempt_id=hashlib.sha256(
                f"integration-attempt:{observer_mode}".encode()
            ).hexdigest(),
            owner_nonce=controller_owner_nonce,
            observer_suffix=hashlib.sha256(
                f"integration-observer:{observer_mode}".encode()
            ).hexdigest(),
            wrapper_arguments_override=[
                *command,
                "--supervised-child",
            ],
        )
        exit_path = policy_root / "preflight_control" / "wrapper_exit.json"
        deadline = time.monotonic() + 30.0
        while not exit_path.is_file():
            if time.monotonic() >= deadline:
                raise AssertionError("real wrapper lifecycle timed out")
            time.sleep(0.05)
        value = load_json(exit_path, "real wrapper exit")
        assert value["contract_type"] == "safa_canonical_preflight_wrapper_exit_v4"
        assert value["exit_code"] == expected_exit
        assert (value["observer_cleanup"] is not None) is expected_cleanup
        cleanup = load_json(
            Path(value["observer_cleanup"]["path"]),
            "real wrapper observer cleanup",
        )
        if observer_mode == "timeout":
            assert cleanup["reason"] == "observer_terminal_timeout"
        else:
            assert cleanup["reason"] == "observer_terminal_consumed"
            assert cleanup["status"] == "closed_terminal_observer"
        assert cleanup["session_residual"] is False
        assert cleanup["process_residual"] is False
        launch = load_json(
            policy_root / "preflight_control" / "observer_launch.json",
            "real observer launch",
        )
        assert launch["tmux"]["pane_pid"] == launch["process"]["pid"]
        assert launch["process"]["pgid"] == launch["process"]["pid"]
        assert launch["process"]["start_ticks"] > 0
        process_exit = load_json(
            policy_root / "preflight_control" / "controller_process_exit.json",
            "real process exit",
        )
        assert (
            process_exit["contract_type"]
            == "safa_canonical_preflight_controller_process_exit_v2"
        )
        if observer_mode == "stop":
            assert process_exit["observer_stop"] is not None
            assert time.monotonic() - started < 10.0
        receipt_paths = list(
            (
                tmp_path
                / "campaign"
                / "preflight_launch_attempts"
                / "by_policy"
            ).glob("*/*/launch_receipt.json")
        )
        assert len(receipt_paths) == 1
        receipt = load_json(
            receipt_paths[0], "real wrapper launch receipt"
        )
        consumer_attempt_path = Path(
            receipt["pane_fault_consumer"]["artifacts"]["attempt"]
        )
        consumer_attempt = load_json(
            consumer_attempt_path,
            "real wrapper pane fault consumer attempt",
        )
        consumer_session = consumer_attempt["consumer_session"]
        consumer_owner_nonce = consumer_attempt[
            "consumer_owner_nonce"
        ]
        gate_owner_seal = consumer_attempt["gate_owner_seal"]
        consumer_artifacts = consumer_attempt["artifacts"]
        consumer_terminal_path = Path(
            consumer_artifacts["terminal"]
        )
        consumer_self_fault_path = Path(
            consumer_attempt["consumer_self_fault_channel"]["path"]
        )
        pane_fault_path = Path(
            consumer_attempt["pane_fault_channel"]["path"]
        )
        deadline = time.monotonic() + 10.0
        while not consumer_terminal_path.is_file():
            if consumer_self_fault_path.stat().st_size:
                raise AssertionError(
                    "real wrapper consumer published self fault: "
                    f"{consumer_self_fault_path.read_text()}"
                )
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "real wrapper consumer terminal timed out: "
                    f"controller={launcher._tmux_pane(wrapper.CONTROLLER_SESSION)}, "
                    f"consumer={launcher._tmux_pane(consumer_session)}"
                )
            time.sleep(0.01)
        consumer_terminal = load_json(
            consumer_terminal_path,
            "real wrapper pane fault consumer terminal",
        )
        controller_cleanup = load_json(
            Path(consumer_artifacts["controller_cleanup"]),
            "real wrapper controller cleanup",
        )
        expected_outcome = (
            "completed"
            if expected_exit == 0
            else "controller_failed"
        )
        assert controller_cleanup["dead_owner_seal"][
            "pane_dead"
        ] is True
        assert (
            controller_cleanup["controller_exit_code"]
            == expected_exit
        )
        assert consumer_terminal["status"] == expected_outcome
        assert (
            consumer_terminal["controller_exit_code"]
            == expected_exit
        )
        assert consumer_terminal["exit_code"] == 0
        assert consumer_self_fault_path.stat().st_size == 0
        assert pane_fault_path.stat().st_size == 0
        assert (
            launcher._tmux_pane(wrapper.CONTROLLER_SESSION)
            is None
        )
        consumer_cleanup = launcher.join_pane_fault_consumer(
            attempt_path=consumer_attempt_path,
            config=config,
            timeout_seconds=5.0,
        )
        assert (
            consumer_cleanup["adjudicated_outcome"]
            == expected_outcome
        )
        assert consumer_cleanup["status"] == "cleaned"
        assert consumer_cleanup["session_residual"] is False
        assert launcher._tmux_pane(consumer_session) is None
        assert launcher._tmux_pane(wrapper.OBSERVER_SESSION) is None
    finally:
        if (
            consumer_session is not None
            and consumer_owner_nonce is not None
            and launcher._tmux_pane(consumer_session) is not None
        ):
            launcher._cleanup_failed_pane_fault_consumer(
                consumer_session, consumer_owner_nonce
            )
        controller_pane = launcher._tmux_pane(
            wrapper.CONTROLLER_SESSION
        )
        if controller_pane is not None:
            if gate_owner_seal is None:
                raise AssertionError(
                    "controller residual has no durable owner seal"
                )
            current_owner = launcher._tmux_owner_seal(
                wrapper.CONTROLLER_SESSION,
                str(gate_owner_seal["owner_nonce"]),
            )
            if any(
                current_owner[key] != gate_owner_seal[key]
                for key in (
                    "session",
                    "pane",
                    "pane_pid",
                    "owner_nonce",
                    "tmux_server",
                )
            ):
                raise AssertionError(
                    "controller residual is a foreign owner"
                )
            launcher._kill_exact_session(
                wrapper.CONTROLLER_SESSION,
                str(gate_owner_seal["owner_nonce"]),
                current_owner,
            )
        for session in sessions:
            assert (
                subprocess.run(
                    ["tmux", "has-session", "-t", session],
                    capture_output=True,
                    text=True,
                ).returncode
                != 0
            )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    "observer_mode",
    (
        "terminal_process_exit_null",
        "terminal_process_exit_path",
        "terminal_process_exit_sha",
        "terminal_process_exit_canonical",
        "terminal_malformed",
        "terminal_validator_exception",
    ),
)
def test_preflight_wrapper_terminal_validation_failure_closes_durably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observer_mode: str,
) -> None:
    wrapper = _wrapper_module()
    repo_root = Path(__file__).parents[1]
    helper = repo_root / "tests/helpers/preflight_lifecycle_helper.py"
    policy_sha256 = hashlib.sha256(
        f"terminal-validation:{observer_mode}".encode()
    ).hexdigest()
    policy_root = tmp_path / "campaign" / "by_policy" / policy_sha256
    config = repo_root / "configs/closeout/canonical_screening_512_v1.json"
    _prepare_wrapper_contract_inputs(wrapper, policy_root)
    command = [
        sys.executable,
        str(helper),
        "wrapper",
        "--wrapper-module",
        str(repo_root / "scripts/run_canonical_preflight_wrapper.py"),
        "--repo-root",
        str(repo_root),
        "--policy-root",
        str(policy_root),
        "--policy",
        policy_sha256,
        "--config",
        str(config),
        "--observer-mode",
        observer_mode,
        "--controller-seconds",
        "0.2",
        "--controller-exit",
        "0",
        "--terminal-timeout",
        "1.0",
    ]
    with ProductionV4Harness(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        policy_sha256=policy_sha256,
        wrapper=wrapper,
        command=command,
    ) as harness:
        wrapper_exit_path = (
            policy_root / "preflight_control/wrapper_exit.json"
        )
        wrapper_exit = harness.wait_json(
            wrapper_exit_path,
            "terminal-validation wrapper exit",
            timeout_seconds=20.0,
        )
        assert wrapper_exit["exit_code"] != 0
        assert wrapper_exit["controller_exit_code"] == 0
        assert wrapper_exit["observer_terminal"] is None
        assert (
            wrapper_exit["observer_terminal_validation_failure"]
            is not None
        )
        cleanup = load_json(
            Path(wrapper_exit["observer_cleanup"]["path"]),
            "terminal-validation cleanup",
        )
        assert cleanup["reason"] == "observer_terminal_validation_failed"
        assert cleanup["observer_terminal_validation_failure"] is not None
        assert cleanup["session_residual"] is False
        assert cleanup["process_residual"] is False
        assert not (
            policy_root / "wrapper_fixture_error.log"
        ).exists()


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    ("observer_mode", "current_status", "expect_success", "expect_valid"),
    (
        ("snapshot_completed_to_failed", "failed", True, True),
        ("snapshot_failed_to_completed", "completed", False, True),
        ("snapshot_delete", None, True, True),
        ("snapshot_exception_replacement", "failed", False, False),
    ),
)
def test_preflight_wrapper_uses_first_strict_terminal_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observer_mode: str,
    current_status: str | None,
    expect_success: bool,
    expect_valid: bool,
) -> None:
    wrapper = _wrapper_module()
    repo_root = Path(__file__).parents[1]
    helper = repo_root / "tests/helpers/preflight_lifecycle_helper.py"
    policy_sha256 = hashlib.sha256(
        f"terminal-snapshot:{observer_mode}".encode()
    ).hexdigest()
    policy_root = tmp_path / "campaign" / "by_policy" / policy_sha256
    config = repo_root / "configs/closeout/canonical_screening_512_v1.json"
    _prepare_wrapper_contract_inputs(wrapper, policy_root)
    command = [
        sys.executable,
        str(helper),
        "wrapper",
        "--wrapper-module",
        str(repo_root / "scripts/run_canonical_preflight_wrapper.py"),
        "--repo-root",
        str(repo_root),
        "--policy-root",
        str(policy_root),
        "--policy",
        policy_sha256,
        "--config",
        str(config),
        "--observer-mode",
        observer_mode,
        "--controller-seconds",
        "0.2",
        "--controller-exit",
        "0",
        "--terminal-timeout",
        "1.0",
    ]
    with ProductionV4Harness(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        policy_sha256=policy_sha256,
        wrapper=wrapper,
        command=command,
    ) as harness:
        wrapper_exit_path = (
            policy_root / "preflight_control/wrapper_exit.json"
        )
        wrapper_exit = harness.wait_json(
            wrapper_exit_path,
            "terminal snapshot wrapper exit",
            timeout_seconds=20.0,
        )
        snapshot = wrapper_exit["observer_terminal_snapshot"]
        assert snapshot is not None
        assert (wrapper_exit["exit_code"] == 0) is expect_success
        assert (
            wrapper_exit["observer_terminal"] == snapshot
        ) is expect_valid
        cleanup = load_json(
            Path(wrapper_exit["observer_cleanup"]["path"]),
            "terminal snapshot cleanup",
        )
        assert cleanup["reason"] == (
            "observer_terminal_consumed"
            if expect_valid
            else "observer_terminal_validation_failed"
        )
        terminal_path = (
            policy_root / "preflight_control/observer_terminal.json"
        )
        assert terminal_path.exists() is (current_status is not None)
        if current_status is not None:
            current = load_json(
                terminal_path, "replacement observer terminal"
            )
            assert current["status"] == current_status
            assert hashlib.sha256(
                terminal_path.read_bytes()
            ).hexdigest() != snapshot["sha256"]
        assert cleanup["session_residual"] is False
        assert cleanup["process_residual"] is False


def test_preflight_wrapper_has_no_post_validation_terminal_path_read() -> None:
    source = inspect.getsource(
        _wrapper_module()._run_wrapped_controller_owned
    )
    assert "observer_terminal[\"path\"]" not in source
    assert ".read_text(" not in source
    assert source.count("_wait_observer_terminal(") == 1
    assert source.count("_read_observer_terminal(") == 1


@pytest.mark.parametrize(
    ("fault", "expected_stage"),
    (
        ("initial_identity", "termination_initial_identity"),
        ("initial_pgid", "termination_initial_identity"),
        ("sigterm", "termination_sigterm"),
        ("sigkill_recheck", "termination_sigkill"),
        ("sigkill_wait", "termination_sigkill_wait"),
    ),
)
def test_controller_process_closure_faults_are_structured_and_reaped(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    expected_stage: str,
) -> None:
    wrapper = _wrapper_module()

    class Process:
        pid = 4242

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.wait_calls = 0

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -wrapper.signal.SIGTERM

        def kill(self) -> None:
            self.returncode = -wrapper.signal.SIGKILL

        def wait(self, timeout: float | None = None) -> int:
            self.wait_calls += 1
            if (
                fault
                in {"sigterm", "sigkill_recheck", "sigkill_wait"}
                and self.wait_calls == 1
            ):
                raise subprocess.TimeoutExpired("fixture", timeout)
            if fault == "sigkill_wait" and self.wait_calls == 2:
                raise RuntimeError("fixture SIGKILL wait failure")
            assert self.returncode is not None
            return self.returncode

    process = Process()
    identity = {"pid": process.pid, "pgid": process.pid, "start_ticks": 1}

    def assert_identity(
        _identity: Mapping[str, int], label: str
    ) -> None:
        if fault == "initial_identity" and "termination" in label:
            raise RuntimeError("fixture initial identity failure")
        if fault == "sigkill_recheck" and "SIGKILL" in label:
            raise RuntimeError("fixture SIGKILL identity failure")

    monkeypatch.setattr(wrapper, "_assert_process_identity", assert_identity)

    def getpgid(_pid: int) -> int:
        if fault == "initial_pgid":
            raise RuntimeError("fixture PGID failure")
        return process.pid

    monkeypatch.setattr(wrapper.os, "getpgid", getpgid)

    def killpg(_pid: int, sig: int) -> None:
        if fault == "sigterm" and sig == wrapper.signal.SIGTERM:
            raise RuntimeError("fixture SIGTERM failure")
        if sig == wrapper.signal.SIGKILL:
            process.returncode = -wrapper.signal.SIGKILL

    monkeypatch.setattr(wrapper.os, "killpg", killpg)
    return_code, closure = wrapper._close_owned_controller_process(
        process, identity, terminate=True
    )
    assert return_code in {
        -wrapper.signal.SIGTERM,
        -wrapper.signal.SIGKILL,
    }
    assert closure["status"] == "reaped"
    assert closure["wait_observed"] is True
    assert closure["process_residual"] is False
    assert expected_stage in {
        failure["stage"] for failure in closure["failures"]
    }
    assert closure["controller_process_closure_sha256"] == (
        canonical_digest(
            closure, "controller_process_closure_sha256"
        )
    )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    ("fault_mode", "commit_state"),
    (
        (
            "controller_fault_start_typed_precommit",
            "precommit_failed_clean",
        ),
        (
            "controller_fault_start_typed_unknown",
            "durability_unknown_quarantined",
        ),
        (
            "controller_fault_start_typed_cleanup",
            "committed_cleanup_error",
        ),
        (
            "controller_fault_start_typed_collision",
            "collision",
        ),
    ),
)
def test_preflight_wrapper_process_start_typed_fault_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_mode: str,
    commit_state: str,
) -> None:
    wrapper = _wrapper_module()
    launcher = _launcher_module()
    repo_root = Path(__file__).parents[1]
    helper = (
        repo_root / "tests/helpers/preflight_lifecycle_helper.py"
    )
    policy_sha256 = hashlib.sha256(
        fault_mode.encode()
    ).hexdigest()
    campaign_root = tmp_path / "campaign"
    policy_root = (
        campaign_root / "by_policy" / policy_sha256
    )
    config = (
        repo_root
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    _prepare_wrapper_contract_inputs(wrapper, policy_root)
    command = [
        sys.executable,
        str(helper),
        "wrapper",
        "--wrapper-module",
        str(
            repo_root
            / "scripts/run_canonical_preflight_wrapper.py"
        ),
        "--repo-root",
        str(repo_root),
        "--policy-root",
        str(policy_root),
        "--policy",
        policy_sha256,
        "--config",
        str(config),
        "--observer-mode",
        fault_mode,
        "--controller-seconds",
        "30",
        "--terminal-timeout",
        "0.5",
    ]
    with ProductionV4Harness(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        policy_sha256=policy_sha256,
        wrapper=wrapper,
        command=command,
    ) as harness:
        attempt_root = harness.receipt_path.parent
        harness.wait_path(
            attempt_root / "wrapper_fault.channel",
            "typed process-start fault channel",
            timeout_seconds=20.0,
            require_nonempty=True,
        )
        receipt = load_json(
            attempt_root / "launch_receipt.json",
            "typed process-start launch receipt",
        )
        descriptor = os.open(
            attempt_root / "wrapper_fault.channel",
            os.O_RDONLY | os.O_NOFOLLOW,
        )
        try:
            snapshot = launcher._read_fault_channel(
                descriptor,
                receipt["fault_channel"],
                attempt_id=receipt["attempt_id"],
                owner_nonce=receipt[
                    "controller_owner_nonce"
                ],
                launch_receipt_sha256=receipt[
                    "launch_receipt_sha256"
                ],
                publisher=receipt["bindings"]["wrapper"],
            )
        finally:
            os.close(descriptor)
        assert snapshot["state"] == "valid_fault"
        failure = snapshot["record"]["failure"]
        assert failure["commit_state"] == commit_state
        assert failure["stage"] == (
            "fixture_controller_process_start_" + commit_state
        )
        assert failure["payload"][
            "post_poison_target_io"
        ] == []
        control = policy_root / "preflight_control"
        assert not (
            control / "controller_process_closure.json"
        ).exists()
        assert not (
            control / "controller_process_exit.json"
        ).exists()
        assert not (control / "wrapper_exit.json").exists()


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    "fault_mode",
    (
        "controller_fault_identity",
        "controller_fault_pgid",
        "controller_fault_start_write",
        "controller_fault_monitor",
        "controller_fault_log_fsync",
        "controller_fault_log_close",
        "controller_fault_monitor_fsync",
        "controller_fault_log_fsync_close",
        "controller_fault_start_write_process_exit_write",
        "controller_fault_observer_cleanup_write",
        "controller_fault_monitor_cleanup_write",
        "controller_fault_process_exit_binding",
        "controller_fault_final_binding",
        "controller_fault_wrapper_exit_write",
    ),
)
def test_preflight_wrapper_post_popen_faults_close_durably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_mode: str,
) -> None:
    wrapper = _wrapper_module()
    repo_root = Path(__file__).parents[1]
    helper = repo_root / "tests/helpers/preflight_lifecycle_helper.py"
    policy_sha256 = hashlib.sha256(
        f"controller-close:{fault_mode}".encode()
    ).hexdigest()
    policy_root = tmp_path / "campaign" / "by_policy" / policy_sha256
    config = repo_root / "configs/closeout/canonical_screening_512_v1.json"
    _prepare_wrapper_contract_inputs(wrapper, policy_root)
    command = [
        sys.executable,
        str(helper),
        "wrapper",
        "--wrapper-module",
        str(repo_root / "scripts/run_canonical_preflight_wrapper.py"),
        "--repo-root",
        str(repo_root),
        "--policy-root",
        str(policy_root),
        "--policy",
        policy_sha256,
        "--config",
        str(config),
        "--observer-mode",
        fault_mode,
        "--controller-seconds",
        (
            "0.2"
            if fault_mode
            in {
                "controller_fault_monitor",
                "controller_fault_log_fsync",
                "controller_fault_log_close",
                "controller_fault_monitor_fsync",
                "controller_fault_log_fsync_close",
                "controller_fault_observer_cleanup_write",
                "controller_fault_monitor_cleanup_write",
                "controller_fault_process_exit_binding",
                "controller_fault_final_binding",
                "controller_fault_wrapper_exit_write",
            }
            else "30"
        ),
        "--controller-exit",
        "0",
        "--terminal-timeout",
        "0.5",
    ]
    with ProductionV4Harness(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        policy_sha256=policy_sha256,
        wrapper=wrapper,
        command=command,
    ) as harness:
        wrapper_exit_path = (
            policy_root / "preflight_control/wrapper_exit.json"
        )
        if fault_mode == "controller_fault_wrapper_exit_write":
            launcher = _launcher_module()
            attempt_root = harness.receipt_path.parent
            receipt = load_json(
                attempt_root / "launch_receipt.json",
                "typed wrapper-exit fault receipt",
            )
            channel_path = attempt_root / "wrapper_fault.channel"
            harness.wait_path(
                channel_path,
                "typed wrapper-exit fault channel",
                timeout_seconds=20.0,
                require_nonempty=True,
            )
            descriptor = os.open(
                channel_path, os.O_RDONLY | os.O_NOFOLLOW
            )
            try:
                snapshot = launcher._read_fault_channel(
                    descriptor,
                    receipt["fault_channel"],
                    attempt_id=receipt["attempt_id"],
                    owner_nonce=receipt[
                        "controller_owner_nonce"
                    ],
                    launch_receipt_sha256=receipt[
                        "launch_receipt_sha256"
                    ],
                    publisher=receipt["bindings"]["wrapper"],
                )
            finally:
                os.close(descriptor)
            assert snapshot["state"] == "valid_fault"
            assert (
                snapshot["record"]["failure"]["commit_state"]
                == "precommit_failed_clean"
            )
            assert (
                snapshot["record"]["failure"]["stage"]
                == "fixture_wrapper_exit_write"
            )
            assert not wrapper_exit_path.exists()
            assert (
                policy_root
                / "preflight_control/controller_process_closure.json"
            ).is_file()
            assert (
                policy_root
                / "preflight_control/observer_cleanup.json"
            ).is_file()
            return
        wrapper_exit = harness.wait_json(
            wrapper_exit_path,
            f"controller closure: {fault_mode}",
            timeout_seconds=20.0,
        )
        assert wrapper_exit["exit_code"] != 0
        assert wrapper_exit["launch_failure"] is not None
        failure_stages = {
            wrapper_exit["launch_failure"]["stage"],
            *(
                failure["stage"]
                for failure in wrapper_exit["launch_failure"][
                    "secondary_failures"
                ]
            ),
        }
        expected_stages = {
            "controller_fault_monitor_fsync": {
                "controller_monitor",
                "controller_log_fsync",
            },
            "controller_fault_log_fsync_close": {
                "controller_log_fsync",
                "controller_log_close",
            },
            "controller_fault_start_write_process_exit_write": {
                "controller_launch_or_start",
                "controller_process_exit_write",
            },
            "controller_fault_observer_cleanup_write": {
                "observer_cleanup_write",
            },
            "controller_fault_monitor_cleanup_write": {
                "controller_monitor",
                "observer_cleanup_write",
            },
            "controller_fault_process_exit_binding": {
                "controller_process_exit_binding",
            },
            "controller_fault_final_binding": {
                "wrapper_process_log_binding",
            },
            "controller_fault_wrapper_exit_write": {
                "wrapper_exit_write",
            },
        }
        if fault_mode in expected_stages:
            assert expected_stages[fault_mode] <= failure_stages
        closure_binding = wrapper_exit["controller_process_closure"]
        assert closure_binding is not None
        closure = load_json(
            Path(closure_binding["path"]), "controller process closure"
        )
        assert closure["wait_observed"] is True
        assert closure["process_residual"] is False
        process_exit_binding = wrapper_exit["controller_process_exit"]
        if fault_mode in {
            "controller_fault_start_write_process_exit_write",
            "controller_fault_process_exit_binding",
        }:
            assert process_exit_binding is None
        else:
            assert process_exit_binding is not None
            process_exit = load_json(
                Path(process_exit_binding["path"]),
                "controller process exit",
            )
            assert process_exit["exit_code"] == (
                closure["wait_return_code"]
                if closure["wait_return_code"] >= 0
                else 128 - closure["wait_return_code"]
            )
        if fault_mode in {
            "controller_fault_monitor",
            "controller_fault_log_fsync",
            "controller_fault_log_close",
            "controller_fault_monitor_fsync",
            "controller_fault_log_fsync_close",
            "controller_fault_observer_cleanup_write",
            "controller_fault_monitor_cleanup_write",
            "controller_fault_process_exit_binding",
            "controller_fault_final_binding",
            "controller_fault_wrapper_exit_write",
        }:
            assert closure["wait_return_code"] == 0
            assert wrapper_exit["controller_exit_code"] == 0
            assert wrapper_exit["observer_terminal"] is not None
            assert wrapper_exit["observer_terminal_validation_failure"] is None
        cleanup_binding = wrapper_exit["observer_cleanup"]
        if fault_mode in {
            "controller_fault_observer_cleanup_write",
            "controller_fault_monitor_cleanup_write",
        }:
            assert cleanup_binding is None
        else:
            assert cleanup_binding is not None
            cleanup = load_json(
                Path(cleanup_binding["path"]),
                "controller fault cleanup",
            )
            assert cleanup["session_residual"] is False
            assert cleanup["process_residual"] is False
            assert cleanup["foreign_session_residual"] is not True
            assert cleanup["foreign_pane_residual"] is not True


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    "fault_mode",
    (
        "controller_fault_observer_launch_write",
        "controller_fault_observer_launch_binding",
        "controller_fault_process_log_mkdir",
        "controller_fault_after_exact_owner_seal",
    ),
)
def test_preflight_wrapper_observer_launch_write_fault_closes_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_mode: str,
) -> None:
    wrapper = _wrapper_module()
    repo_root = Path(__file__).parents[1]
    helper = repo_root / "tests/helpers/preflight_lifecycle_helper.py"
    policy_sha256 = hashlib.sha256(fault_mode.encode()).hexdigest()
    policy_root = tmp_path / "campaign" / "by_policy" / policy_sha256
    config = repo_root / "configs/closeout/canonical_screening_512_v1.json"
    _prepare_wrapper_contract_inputs(wrapper, policy_root)
    command = [
        sys.executable,
        str(helper),
        "wrapper",
        "--wrapper-module",
        str(repo_root / "scripts/run_canonical_preflight_wrapper.py"),
        "--repo-root",
        str(repo_root),
        "--policy-root",
        str(policy_root),
        "--policy",
        policy_sha256,
        "--config",
        str(config),
        "--observer-mode",
        fault_mode,
        "--controller-seconds",
        "30",
        "--controller-exit",
        "0",
        "--terminal-timeout",
        "0.5",
    ]
    with ProductionV4Harness(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        policy_sha256=policy_sha256,
        wrapper=wrapper,
        command=command,
    ) as harness:
        wrapper_exit_path = (
            policy_root / "preflight_control/wrapper_exit.json"
        )
        wrapper_exit = harness.wait_json(
            wrapper_exit_path,
            "observer launch write wrapper exit",
            timeout_seconds=20.0,
        )
        assert wrapper_exit["exit_code"] != 0
        assert (
            wrapper_exit["observer_launch"] is not None
        ) is (
            fault_mode
            in {
                "controller_fault_observer_launch_binding",
                "controller_fault_process_log_mkdir",
                "controller_fault_after_exact_owner_seal",
            }
        )
        stages = {
            wrapper_exit["launch_failure"]["stage"],
            *(
                failure["stage"]
                for failure in wrapper_exit["launch_failure"][
                    "secondary_failures"
                ]
            ),
        }
        if fault_mode == "controller_fault_observer_launch_write":
            assert "observer_launch_write" in stages
        elif fault_mode == "controller_fault_observer_launch_binding":
            assert "observer_launch" in stages
            assert (
                "observer launch binding hash failure"
                in wrapper_exit["launch_failure"]["message"]
            )
        elif fault_mode == "controller_fault_after_exact_owner_seal":
            assert "observer_launch" in stages
            assert (
                "failure after exact owner seal"
                in wrapper_exit["launch_failure"]["message"]
            )
        else:
            assert "controller_launch_or_start" in stages
            assert "strict_binding_validation" in stages
            assert (
                "controller process log open failure"
                in wrapper_exit["launch_failure"]["message"]
            )
        cleanup_binding = wrapper_exit["observer_cleanup"]
        assert cleanup_binding is not None
        cleanup = load_json(
            Path(cleanup_binding["path"]),
            "observer launch write cleanup",
        )
        assert cleanup["session_residual"] is False
        assert cleanup["process_residual"] is False


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
@pytest.mark.parametrize(
    "mode",
    (
        "success",
        "resource_stop",
        "early_exit",
        "terminal_timeout",
        "late_terminal_race",
        "late_terminal_foreign_replacement",
        "proc_snapshot_absent",
        "identity_replacement",
        "process_exit_delay",
        "process_exit_barrier_timeout",
        "late_snapshot_replacement",
        "late_snapshot_delete",
    ),
)
def test_preflight_real_production_controller_observer_chain(
    tmp_path: Path,
    mode: str,
) -> None:
    module = _controller_module()
    wrapper = _wrapper_module()
    repo_root = Path(__file__).parents[1]
    helper = repo_root / "tests/helpers/preflight_lifecycle_helper.py"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"production-chain-checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        ledger,
        [
            _row(
                "integration",
                checkpoint_sha256,
                path=str(checkpoint.resolve()),
            )
        ],
    )
    policy, policy_path, _ = _policy(tmp_path, ledger)
    trust_config = (
        repo_root
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    policy["resources"]["resource_poll_seconds"] = 1
    policy["resources"]["cpu_window_seconds"] = 1
    policy["policy_file"] = {
        "path": str(trust_config.resolve()),
        "sha256": hashlib.sha256(
            trust_config.read_bytes()
        ).hexdigest(),
    }
    campaign_root = tmp_path / "campaign"
    paths = module._paths(campaign_root, policy["policy_sha256"])
    plan = build_checkpoint_plan(tmp_path, policy, paths["preflight_results"])
    write_exclusive_json(paths["checkpoint_plan"], plan)
    request_paths = module.write_preflight_requests(
        plan, paths["preflight_requests"]
    )
    module._build_preflight_request_manifest(
        policy, paths, plan, request_paths
    )
    strict = _strict_preflight(
        checkpoint_sha256,
        "raw",
        policy["output_decoder_registry"],
    )
    terminal_timeout_seconds = (
        0.05
        if mode in {
            "terminal_timeout",
            "late_terminal_race",
            "late_terminal_foreign_replacement",
            "proc_snapshot_absent",
            "late_snapshot_replacement",
            "late_snapshot_delete",
        }
        else 2.0
    )
    process_termination_wait_seconds = 10.0
    controller_start_exit_margin_seconds = 10.0
    wrapper_completion_timeout_seconds = (
        2.0 * process_termination_wait_seconds
        + terminal_timeout_seconds
        + controller_start_exit_margin_seconds
    )
    fixture_path = tmp_path / "production_fixture.json"
    controller_command = [
        sys.executable,
        str(helper),
        "production-role",
        "--controller-module",
        str(repo_root / "scripts/run_canonical_checkpoint_screening.py"),
        "--fixture",
        str(fixture_path),
        "--config",
        str(trust_config),
        "--role",
        "controller",
    ]
    observer_command = [
        sys.executable,
        str(helper),
        "production-role",
        "--controller-module",
        str(repo_root / "scripts/run_canonical_checkpoint_screening.py"),
        "--fixture",
        str(fixture_path),
        "--config",
        str(trust_config),
        "--role",
        "observer",
    ]
    fixture_path.write_text(
        json.dumps(
            {
                "policy": policy,
                "campaign_root": str(campaign_root.resolve()),
                "controller_command": controller_command,
                "observer_command": observer_command,
                "strict_preflight": strict,
                "resource_stop": mode == "resource_stop",
                "mode": mode,
                "process_termination_wait": (
                    process_termination_wait_seconds
                ),
                "checkpoint_delay": (
                    0.2
                    if mode in {
                        "terminal_timeout",
                        "late_terminal_race",
                        "late_terminal_foreign_replacement",
                        "proc_snapshot_absent",
                        "late_snapshot_replacement",
                        "late_snapshot_delete",
                    }
                    else 3.0
                    if mode == "identity_replacement"
                    else 0.0
                ),
                "terminal_timeout": terminal_timeout_seconds,
                "barrier_timeout": 2.0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    policy_root = paths["policy_root"]
    sessions = (
        wrapper.CONTROLLER_SESSION,
        wrapper.OBSERVER_SESSION,
    )
    for session in sessions:
        assert (
            subprocess.run(
                ["tmux", "has-session", "-t", session],
                capture_output=True,
                text=True,
            ).returncode
            != 0
        )
    wrapper_command = [
        sys.executable,
        str(helper),
        "production-wrapper",
        "--wrapper-module",
        str(repo_root / "scripts/run_canonical_preflight_wrapper.py"),
        "--repo-root",
        str(repo_root),
        "--policy-root",
        str(policy_root),
        "--policy",
        policy["policy_sha256"],
            "--config",
            str(trust_config),
        "--fixture",
        str(fixture_path),
    ]
    started = time.monotonic()
    replacement_identity: dict[str, Any] | None = None
    try:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                wrapper.CONTROLLER_SESSION,
                "-e",
                (
                    f"{wrapper.OBSERVER_SESSION_ENV}="
                    f"{wrapper.OBSERVER_SESSION}"
                ),
                "-c",
                str(repo_root),
                *wrapper_command,
            ],
            check=True,
        )
        if mode == "identity_replacement":
            ready_path = (
                policy_root / "preflight_control" / "observer_ready.json"
            )
            ready_deadline = time.monotonic() + 10.0
            while not ready_path.is_file():
                if time.monotonic() >= ready_deadline:
                    raise AssertionError(
                        "production observer ready barrier timed out"
                    )
                time.sleep(0.02)
            subprocess.run(
                [
                    "tmux",
                    "kill-session",
                    "-t",
                    wrapper.OBSERVER_SESSION,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    wrapper.OBSERVER_SESSION,
                    sys.executable,
                    "-c",
                    "import time;time.sleep(30)",
                ],
                check=True,
            )
            current = subprocess.run(
                [
                    "tmux",
                    "list-panes",
                    "-t",
                    wrapper.OBSERVER_SESSION,
                    "-F",
                    "#{session_name}\t#{pane_id}\t#{pane_pid}\t#{pane_current_command}",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().split("\t")
            replacement_identity = {
                "session": current[0],
                "pane": current[1],
                "pane_pid": int(current[2]),
                "pane_current_command": current[3],
            }
        wrapper_exit_path = (
            policy_root / "preflight_control" / "wrapper_exit.json"
        )
        deadline = (
            time.monotonic() + wrapper_completion_timeout_seconds
        )
        while not wrapper_exit_path.is_file():
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "production controller/observer chain timed out"
                )
            time.sleep(0.05)
        wrapper_exit = load_json(
            wrapper_exit_path, "production chain wrapper exit"
        )
        launch = load_json(
            policy_root / "preflight_control" / "observer_launch.json",
            "production chain observer launch",
        )
        gate_ready = load_json(
            Path(launch["observer_gate_ready"]["path"]),
            "production chain observer gate ready",
        )
        gate_release = load_json(
            Path(launch["observer_gate_release"]["path"]),
            "production chain observer gate release",
        )
        assert (
            launch["contract_type"]
            == "safa_canonical_preflight_observer_launch_v3"
        )
        assert gate_ready["process"] == launch["process"]
        assert gate_ready["process"]["pid"] == gate_ready["process"]["pgid"]
        assert gate_ready["tmux"] == launch["tmux"]
        assert gate_ready["tmux_server"] == launch["tmux_server"]
        assert (
            gate_ready["owner_nonce"]
            == launch["tmux_owner_seal"]["owner_nonce"]
        )
        assert gate_ready["observer_command"] == observer_command
        assert gate_ready["gate_command"] != observer_command
        assert gate_release["observer_gate_ready"] == (
            launch["observer_gate_ready"]
        )
        assert gate_release["observer_command"] == observer_command
        assert gate_release["owner_nonce"] == gate_ready["owner_nonce"]
        try:
            current_observer_process = wrapper._process_identity(
                int(launch["process"]["pid"])
            )
        except (FileNotFoundError, ProcessLookupError):
            current_observer_process = None
        assert current_observer_process != launch["process"]
        observer_terminal_path = (
            policy_root / "preflight_control" / "observer_terminal.json"
        )
        observer_terminal = (
            load_json(
                observer_terminal_path,
                "production chain observer terminal",
            )
            if observer_terminal_path.is_file()
            else None
        )
        controller_terminal_path = (
            policy_root / "preflight_control" / "controller_terminal.json"
        )
        if mode in {
            "resource_stop",
            "early_exit",
            "terminal_timeout",
            "late_terminal_race",
            "late_terminal_foreign_replacement",
            "proc_snapshot_absent",
            "identity_replacement",
            "process_exit_barrier_timeout",
            "late_snapshot_replacement",
            "late_snapshot_delete",
        }:
            assert wrapper_exit["exit_code"] != 0
            if mode in {
                "resource_stop",
                "early_exit",
                "process_exit_barrier_timeout",
            }:
                assert observer_terminal is not None
                assert observer_terminal["status"] == "failed"
                if mode != "process_exit_barrier_timeout":
                    assert wrapper_exit["observer_stop"] is not None
            elif mode not in {
                "late_terminal_race",
                "late_terminal_foreign_replacement",
                "late_snapshot_replacement",
                "late_snapshot_delete",
            }:
                assert observer_terminal is None
            if mode in {
                "terminal_timeout",
                "proc_snapshot_absent",
            }:
                cleanup = load_json(
                    Path(wrapper_exit["observer_cleanup"]["path"]),
                    "production timeout cleanup",
                )
                assert cleanup["status"] in {
                    "cleaned_process_killed",
                    "cleaned_process_already_absent",
                    "cleaned_process_absent",
                    "cleaned_process_zombie",
                    "cleaned_detached_process_killed",
                    "cleaned_tmux_killed",
                    "cleaned_tmux_already_absent",
                    "cleaned_tmux_absent",
                    "cleaned_tmux_zombie",
                }
                assert cleanup["session_residual"] is False
                assert cleanup["process_residual"] is False
                assert not observer_terminal_path.exists()
                time.sleep(0.5)
                assert not observer_terminal_path.exists()
            if mode == "late_terminal_race":
                assert observer_terminal is not None
                assert observer_terminal["status"] == "completed"
                assert wrapper_exit["observer_terminal"] is None
                late_binding = wrapper_exit["late_observer_terminal"]
                assert late_binding is not None
                assert late_binding == {
                    "path": str(observer_terminal_path.resolve()),
                    "sha256": hashlib.sha256(
                        observer_terminal_path.read_bytes()
                    ).hexdigest(),
                    "canonical_sha256": observer_terminal[
                        "observer_terminal_sha256"
                    ],
                }
                cleanup = load_json(
                    Path(wrapper_exit["observer_cleanup"]["path"]),
                    "production late-terminal cleanup",
                )
                assert cleanup["reason"] == "observer_terminal_timeout"
                assert cleanup["late_observer_terminal"] == late_binding
                assert cleanup["session_residual"] is False
                assert cleanup["process_residual"] is False
                assert wrapper_exit["observer_stop"] is None
                race = load_json(
                    policy_root
                    / "preflight_control/late_terminal_race_window.json",
                    "production late-terminal race window",
                )
                assert race["terminal_absent_after_wait"] is True
                assert race["race_window_sha256"] == canonical_digest(
                    race, "race_window_sha256"
                )
            if mode == "late_terminal_foreign_replacement":
                assert observer_terminal is not None
                assert observer_terminal["status"] == "completed"
                assert wrapper_exit["observer_terminal"] is None
                late_binding = wrapper_exit["late_observer_terminal"]
                assert late_binding is not None
                assert late_binding == {
                    "path": str(observer_terminal_path.resolve()),
                    "sha256": hashlib.sha256(
                        observer_terminal_path.read_bytes()
                    ).hexdigest(),
                    "canonical_sha256": observer_terminal[
                        "observer_terminal_sha256"
                    ],
                }
                replacement = load_json(
                    policy_root
                    / (
                        "preflight_control/"
                        "late_terminal_foreign_replacement.json"
                    ),
                    "late-terminal foreign replacement",
                )
                assert replacement[
                    "foreign_replacement_sha256"
                ] == canonical_digest(
                    replacement, "foreign_replacement_sha256"
                )
                assert replacement["observer_terminal"] == late_binding
                replacement_identity = replacement["foreign_tmux"]
                cleanup = load_json(
                    Path(wrapper_exit["observer_cleanup"]["path"]),
                    "production late-terminal foreign cleanup",
                )
                assert cleanup["reason"] == "observer_terminal_timeout"
                assert cleanup["late_observer_terminal"] == late_binding
                assert cleanup["session_residual"] is False
                assert cleanup["process_residual"] is False
                assert cleanup["foreign_session_residual"] is True
                assert cleanup["foreign_pane_residual"] is True
                assert cleanup["foreign_tmux"] == replacement_identity
                assert (
                    cleanup["foreign_tmux_server"]
                    == replacement["foreign_tmux_server"]
                )
                assert wrapper_exit["late_observer_terminal"] == (
                    cleanup["late_observer_terminal"]
                )
                assert wrapper_exit["exit_code"] != 0
            if mode in {
                "late_snapshot_replacement",
                "late_snapshot_delete",
            }:
                late_snapshot = wrapper_exit[
                    "late_observer_terminal_snapshot"
                ]
                assert late_snapshot is not None
                assert wrapper_exit[
                    "late_observer_terminal"
                ] == late_snapshot
                cleanup = load_json(
                    Path(wrapper_exit["observer_cleanup"]["path"]),
                    "late snapshot cleanup",
                )
                assert cleanup["reason"] == "observer_terminal_timeout"
                assert cleanup[
                    "late_observer_terminal_snapshot"
                ] == late_snapshot
                if mode == "late_snapshot_replacement":
                    current = load_json(
                        observer_terminal_path,
                        "late replacement observer terminal",
                    )
                    assert current["status"] == "failed"
                    assert hashlib.sha256(
                        observer_terminal_path.read_bytes()
                    ).hexdigest() != late_snapshot["sha256"]
                else:
                    assert not observer_terminal_path.exists()
            if mode == "identity_replacement":
                cleanup = load_json(
                    Path(wrapper_exit["observer_cleanup"]["path"]),
                    "production replacement cleanup",
                )
                assert cleanup["status"] == "identity_replaced_not_terminated"
                assert replacement_identity is not None
                assert cleanup["observed_tmux"] == replacement_identity
                assert cleanup["process_residual"] is False
            if mode == "process_exit_barrier_timeout":
                barrier = load_json(
                    policy_root
                    / "preflight_control/process_exit_barrier.json",
                    "production process-exit timeout barrier",
                )
                assert barrier[
                    "controller_terminal_before_process_exit"
                ] is True
                assert barrier[
                    "observer_terminal_before_process_exit"
                ] is True
                assert barrier["observer_status_before_process_exit"] == (
                    "failed"
                )
                assert barrier[
                    "process_exit_barrier_sha256"
                ] == canonical_digest(
                    barrier, "process_exit_barrier_sha256"
                )
                cleanup = load_json(
                    Path(wrapper_exit["observer_cleanup"]["path"]),
                    "process-exit timeout cleanup",
                )
                assert (
                    cleanup["reason"]
                    == "observer_terminal_validation_failed"
                )
                assert (
                    wrapper_exit[
                        "observer_terminal_validation_failure"
                    ]
                    is not None
                )
                assert cleanup["session_residual"] is False
                assert cleanup["process_residual"] is False
            assert (
                time.monotonic() - started
                < wrapper_completion_timeout_seconds
            )
        else:
            assert wrapper_exit["exit_code"] == 0
            assert wrapper_exit["observer_cleanup"] is not None
            cleanup = load_json(
                Path(wrapper_exit["observer_cleanup"]["path"]),
                "production success cleanup",
            )
            assert cleanup["reason"] == "observer_terminal_consumed"
            assert cleanup["status"] == "closed_terminal_observer"
            assert cleanup["session_residual"] is False
            assert cleanup["process_residual"] is False
            assert observer_terminal is not None
            assert observer_terminal["status"] == "completed"
            controller_terminal = load_json(
                controller_terminal_path,
                "production chain controller terminal",
            )
            assert controller_terminal["status"] == "completed"
            assert controller_terminal["progress"]["completed"] == 1
            if mode == "process_exit_delay":
                barrier = load_json(
                    policy_root
                    / "preflight_control/process_exit_barrier.json",
                    "production process-exit barrier",
                )
                assert barrier[
                    "controller_terminal_before_process_exit"
                ] is True
                assert barrier[
                    "observer_terminal_before_process_exit"
                ] is False
                assert barrier[
                    "process_exit_barrier_sha256"
                ] == canonical_digest(
                    barrier, "process_exit_barrier_sha256"
                )
    finally:
        if replacement_identity is not None:
            current = subprocess.run(
                [
                    "tmux",
                    "list-panes",
                    "-t",
                    wrapper.OBSERVER_SESSION,
                    "-F",
                    "#{session_name}\t#{pane_id}\t#{pane_pid}\t#{pane_current_command}",
                ],
                capture_output=True,
                text=True,
            )
            if current.returncode == 0:
                row = current.stdout.strip().split("\t")
                current_identity = {
                    "session": row[0],
                    "pane": row[1],
                    "pane_pid": int(row[2]),
                    "pane_current_command": row[3],
                }
                assert current_identity == replacement_identity
                subprocess.run(
                    [
                        "tmux",
                        "kill-pane",
                        "-t",
                        replacement_identity["pane"],
                    ],
                    check=True,
                )
        exit_deadline = time.monotonic() + 3.0
        while (
            subprocess.run(
                [
                    "tmux",
                    "has-session",
                    "-t",
                    wrapper.CONTROLLER_SESSION,
                ],
                capture_output=True,
                text=True,
            ).returncode
            == 0
            and time.monotonic() < exit_deadline
        ):
            time.sleep(0.02)
        for session in sessions:
            assert (
                subprocess.run(
                    ["tmux", "has-session", "-t", session],
                    capture_output=True,
                    text=True,
                ).returncode
                != 0
            )


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("tmux") is None,
    reason="requires Linux /proc and tmux",
)
def test_preflight_real_production_observer_provenance_timeout(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    wrapper = _wrapper_module()
    repo_root = Path(__file__).parents[1]
    helper = repo_root / "tests/helpers/preflight_lifecycle_helper.py"
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, _, _ = _policy(tmp_path, ledger)
    trust_config = (
        repo_root
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    policy["policy_file"] = {
        "path": str(trust_config.resolve()),
        "sha256": hashlib.sha256(trust_config.read_bytes()).hexdigest(),
    }
    campaign_root = tmp_path / "campaign"
    paths = module._paths(campaign_root, policy["policy_sha256"])
    fixture_path = tmp_path / "provenance_fixture.json"
    observer_command = [
        sys.executable,
        str(helper),
        "production-role",
        "--controller-module",
        str(repo_root / "scripts/run_canonical_checkpoint_screening.py"),
        "--fixture",
        str(fixture_path),
        "--config",
        str(trust_config),
        "--role",
        "observer",
    ]
    fixture_path.write_text(
        json.dumps(
            {
                "policy": policy,
                "campaign_root": str(campaign_root.resolve()),
                "controller_command": [],
                "observer_command": observer_command,
                "strict_preflight": {},
                "resource_stop": False,
                "barrier_timeout": 0.5,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            ["tmux", "has-session", "-t", wrapper.OBSERVER_SESSION],
            capture_output=True,
            text=True,
        ).returncode
        != 0
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            wrapper.OBSERVER_SESSION,
            "-e",
            (
                f"{wrapper.OBSERVER_SESSION_ENV}="
                f"{wrapper.OBSERVER_SESSION}"
            ),
            "-c",
            str(repo_root),
            *observer_command,
        ],
        check=True,
    )
    launched = subprocess.run(
        [
            "tmux",
            "list-panes",
            "-t",
            wrapper.OBSERVER_SESSION,
            "-F",
            "#{pane_pid}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        terminal_path = (
            paths["preflight_control"] / "observer_terminal.json"
        )
        deadline = time.monotonic() + 10.0
        while not terminal_path.is_file():
            if time.monotonic() >= deadline:
                raise AssertionError(
                    "production provenance timeout did not publish terminal"
                )
            time.sleep(0.02)
        terminal = load_json(
            terminal_path, "production provenance timeout terminal"
        )
        stop = load_json(
            paths["preflight_control"] / "observer_stop.json",
            "production provenance timeout stop",
        )
        assert terminal["status"] == "failed"
        assert "provenance barrier timed out" in terminal["failure"]["message"]
        assert stop["wrapper_claim"] is None
        assert stop["observer_launch"] is None
        assert stop["controller_process_start"] is None
        exit_deadline = time.monotonic() + 5.0
        while (
            subprocess.run(
                ["tmux", "has-session", "-t", wrapper.OBSERVER_SESSION],
                capture_output=True,
                text=True,
            ).returncode
            == 0
        ):
            if time.monotonic() >= exit_deadline:
                raise AssertionError(
                    "production provenance observer tmux did not exit"
                )
            time.sleep(0.02)
    finally:
        current = subprocess.run(
            [
                "tmux",
                "list-panes",
                "-t",
                wrapper.OBSERVER_SESSION,
                "-F",
                "#{pane_pid}",
            ],
            capture_output=True,
            text=True,
        )
        if current.returncode == 0:
            assert current.stdout.strip() == launched
            subprocess.run(
                [
                    "tmux",
                    "kill-session",
                    "-t",
                    wrapper.OBSERVER_SESSION,
                ],
                check=True,
            )
        assert (
            subprocess.run(
                ["tmux", "has-session", "-t", wrapper.OBSERVER_SESSION],
                capture_output=True,
                text=True,
            ).returncode
            != 0
        )


def test_controller_raw_import_executes_no_policy_bound_module() -> None:
    root = Path(__file__).parents[1]
    controller = root / "scripts/run_canonical_checkpoint_screening.py"
    code = (
        "import importlib.util,json,sys;"
        f"p={str(controller)!r};"
        "s=importlib.util.spec_from_file_location('raw_controller',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps(sorted(x for x in ("
        "'safa.closeout.canonical_screening',"
        "'safa.closeout.preflight_launch_contract',"
        "'safa.closeout.canonical_screening_worker') if x in sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


@pytest.mark.parametrize(
    "implementation",
    (
        "checkpoint_preflight",
        "arcface_evaluator",
        "e0_loader",
        "canonical_quality",
        "screening_contracts",
        "screening_worker",
        "preflight_verified_loader",
        "preflight_launch_contract",
        "controller",
        "ram_probe_launcher",
        "preflight_wrapper",
        "gpu_wrapper",
        "generator_sampling",
        "meanflow_sampling",
        "latent_codec",
        "output_contract",
    ),
)
def test_controller_stdlib_bootstrap_rejects_each_implementation_tamper(
    tmp_path: Path, implementation: str
) -> None:
    module = _raw_controller_module()
    root = Path(__file__).parents[1]
    config = json.loads(
        (
            root / "configs/closeout/canonical_screening_512_v1.json"
        ).read_text(encoding="utf-8")
    )
    config["implementations"][implementation]["sha256"] = "0" * 64
    path = tmp_path / f"{implementation}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        module.ControllerBootstrapError,
        match="implementation digest differs",
    ):
        module._stdlib_validate_implementation_bindings(path)


def test_controller_tampered_worker_fails_before_dynamic_import(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    controller = root / "scripts/run_canonical_checkpoint_screening.py"
    config = json.loads(
        (
            root / "configs/closeout/canonical_screening_512_v1.json"
        ).read_text(encoding="utf-8")
    )
    config["implementations"]["screening_worker"]["sha256"] = "0" * 64
    config_path = tmp_path / "tampered-worker-policy.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    code = (
        "import importlib.util,json,sys;"
        f"p={str(controller)!r};c={str(config_path)!r};"
        "s=importlib.util.spec_from_file_location('raw_controller',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "\ntry:m._install_verified_contract_api(__import__('pathlib').Path(c))\n"
        "except m.ControllerBootstrapError:pass\n"
        "else:raise SystemExit(7)\n"
        "print(json.dumps(sorted(x for x in ("
        "'safa.closeout.canonical_screening',"
        "'safa.closeout.canonical_screening_worker') if x in sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_controller_preflight_contract_ignores_ambient_same_name_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _raw_controller_module()
    ambient = types.ModuleType(
        "safa.closeout.preflight_launch_contract"
    )
    monkeypatch.setitem(
        sys.modules,
        "safa.closeout.preflight_launch_contract",
        ambient,
    )
    policy_path = (
        Path(__file__).parents[1]
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    module._install_verified_preflight_contract_api(policy_path)
    assert module._PREFLIGHT_CONTRACT_MODULE is not ambient
    assert module._PREFLIGHT_CONTRACT_MODULE.__file__ == str(
        (
            Path(__file__).parents[1]
            / "src/safa/closeout/preflight_launch_contract.py"
        ).resolve()
    )


def test_controller_preflight_contract_rejects_noncanonical_repo_path(
    tmp_path: Path,
) -> None:
    module = _raw_controller_module()
    root = Path(__file__).parents[1]
    config = json.loads(
        (
            root / "configs/closeout/canonical_screening_512_v1.json"
        ).read_text(encoding="utf-8")
    )
    wrong_path = root / "src/safa/closeout/canonical_screening.py"
    config["implementations"]["preflight_launch_contract"] = {
        "path": str(wrong_path.resolve()),
        "sha256": hashlib.sha256(wrong_path.read_bytes()).hexdigest(),
    }
    config_path = tmp_path / "wrong-preflight-path.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        module.ControllerBootstrapError,
        match="exact repository path",
    ):
        module._install_verified_preflight_contract_api(config_path)


@pytest.mark.parametrize(
    ("source", "message"),
    (
        (b"raise RuntimeError('broken import')\n", "import failed"),
        (
            b"PreflightLaunchContractError = RuntimeError\n",
            "omit controller API",
        ),
    ),
)
def test_controller_preflight_contract_failure_is_terminal_before_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: bytes,
    message: str,
) -> None:
    module = _raw_controller_module()
    root = Path(__file__).parents[1]
    policy_path = (
        root / "configs/closeout/canonical_screening_512_v1.json"
    )
    campaign_root = tmp_path / "campaign"
    ready_path = tmp_path / "observer_bootstrap.json"
    original_reader = module._stdlib_read_sealed_module_source
    _actual_source, seal = original_reader(
        root / "src/safa/closeout/preflight_launch_contract.py",
        json.loads(policy_path.read_text(encoding="utf-8"))[
            "implementations"
        ]["preflight_launch_contract"]["sha256"],
    )
    monkeypatch.setattr(
        module,
        "_stdlib_read_sealed_module_source",
        lambda _path, _sha256: (source, seal),
    )
    monkeypatch.setenv(
        module.OBSERVER_BOOTSTRAP_PATH_ENV, str(ready_path)
    )
    monkeypatch.setenv(
        module.OBSERVER_BOOTSTRAP_POLICY_ENV, "1" * 64
    )
    monkeypatch.setenv(
        module.OBSERVER_BOOTSTRAP_WRAPPER_ENV,
        json.dumps(_shared_binding("wrapper")),
    )
    monkeypatch.setenv(
        module.OBSERVER_BOOTSTRAP_NONCE_ENV, "2" * 64
    )
    with pytest.raises(
        module.ControllerBootstrapError, match=message
    ):
        module.main(
            [
                "--config",
                str(policy_path),
                "--phase",
                "plan",
                "--campaign-root",
                str(campaign_root),
                "--execute",
            ]
    )
    assert not ready_path.exists()
    assert not (
        campaign_root / "preflight_control/wrapper_claim.json"
    ).exists()
    terminal_path = (
        campaign_root
        / "bootstrap_control/plan/main_bootstrap_terminal.json"
    )
    terminal = load_json(
        terminal_path, "preflight contract bootstrap terminal"
    )
    assert terminal["status"] == "failed"
    assert message in terminal["failure"]["message"]


def test_controller_rechecks_preflight_contract_after_screening_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _raw_controller_module()
    root = Path(__file__).parents[1]
    policy_path = (
        root / "configs/closeout/canonical_screening_512_v1.json"
    )
    original_reader = module._stdlib_read_sealed_module_source
    calls = {"count": 0}

    def mutate_after_install(
        path: Path, expected_sha256: str
    ) -> tuple[bytes, tuple[Any, ...]]:
        source, seal = original_reader(path, expected_sha256)
        calls["count"] += 1
        if calls["count"] >= 3:
            seal = (*seal[:-2], int(seal[-2]) + 1, seal[-1])
        return source, seal

    monkeypatch.setattr(
        module,
        "_stdlib_read_sealed_module_source",
        mutate_after_install,
    )
    with pytest.raises(
        module.ControllerBootstrapError,
        match="changed after verified import",
    ):
        module._install_verified_contract_api(
            policy_path,
            verify_historical_output_evidence=False,
        )
    assert calls["count"] >= 3


def test_real_worker_bootstrap_imports_no_heavy_modules() -> None:
    root = Path(__file__).parents[1]
    controller = (
        root
        / "scripts"
        / "run_canonical_checkpoint_screening.py"
    )
    policy_path = root / "configs/closeout/canonical_screening_512_v1.json"
    code = (
        "import importlib.util,json,sys;"
        f"p={str(controller)!r};"
        "s=importlib.util.spec_from_file_location('worker_bootstrap',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"m._install_verified_contract_api(__import__('pathlib').Path({str(policy_path)!r}),"
        "verify_historical_output_evidence=False);"
        "print(json.dumps(sorted(x for x in "
        "('torch','torchvision','onnxruntime','diffusers') if x in sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_gpu_wrapper_records_sigkill_before_controller_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _gpu_wrapper_module()
    monkeypatch.setattr(
        wrapper,
        "_launch_observer",
        lambda **_kwargs: {
            "session": "fixture-monitor",
            "command": [
                "monitor",
                "--monitor-target",
                "screen512",
                "--execute",
            ],
            "launched_at": "2026-07-27T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(
        wrapper,
        "_wait_observer_terminal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("fixture observer exited")
        ),
    )
    config = tmp_path / "policy.json"
    config.write_text("{}\n", encoding="utf-8")
    policy_root = tmp_path / "campaign" / "by_policy" / ("8" * 64)
    value = wrapper.run_wrapped_controller(
        repo_root=tmp_path,
        policy_root=policy_root,
        policy_sha256="8" * 64,
        config=config,
        campaign_root=tmp_path / "campaign",
        phase="screen512",
        python=sys.executable,
        command=[
            sys.executable,
            "-c",
            "import os,signal;os.kill(os.getpid(),signal.SIGKILL)",
        ],
    )
    assert value["exit_code"] == 137
    assert value["signal"] == 9
    assert value["controller_claim"] is None
    assert value["controller_terminal"] is None
    assert load_json(
        policy_root / "gpu_control/screen512/wrapper_exit.json",
        "GPU wrapper exit",
    ) == value
    terminal = load_json(
        policy_root / "gpu_control/screen512/wrapper_terminal.json",
        "GPU wrapper terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["failure"]["type"] == "ControllerExit"


def test_gpu_wrapper_preclaim_failure_still_writes_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapper = _gpu_wrapper_module()
    config = tmp_path / "policy.json"
    config.write_text("{}\n", encoding="utf-8")
    policy_root = tmp_path / "campaign" / "by_policy" / ("7" * 64)
    claim_path = (
        policy_root / "gpu_control/screen512/wrapper_claim.json"
    )
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text("{}\n", encoding="utf-8")
    popen_calls = 0

    def forbidden_popen(*_args, **_kwargs):
        nonlocal popen_calls
        popen_calls += 1
        raise AssertionError("controller Popen must not run")

    monkeypatch.setattr(wrapper.subprocess, "Popen", forbidden_popen)
    monkeypatch.setattr(
        wrapper.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=1, stderr="", stdout=""
        ),
    )
    with pytest.raises((FileExistsError, RuntimeError)):
        wrapper.run_wrapped_controller(
            repo_root=tmp_path,
            policy_root=policy_root,
            policy_sha256="7" * 64,
            config=config,
            campaign_root=tmp_path / "campaign",
            phase="screen512",
            python=sys.executable,
            command=[sys.executable, "-c", "raise SystemExit(0)"],
        )
    assert popen_calls == 0
    terminal = load_json(
        policy_root / "gpu_control/screen512/wrapper_terminal.json",
        "wrapper preclaim terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["wrapper_claim"] is not None


def test_gpu_wrapper_validates_observer_terminal_barrier(
    tmp_path: Path,
) -> None:
    wrapper = _gpu_wrapper_module()
    ready = {
        "contract_type": "safa_canonical_gpu_observer_ready_v1",
        "policy_sha256": "9" * 64,
        "phase": "screen512",
    }
    ready["observer_ready_sha256"] = wrapper._canonical_digest(
        ready, "observer_ready_sha256"
    )
    ready_path = tmp_path / "observer_ready.json"
    wrapper._write_exclusive(ready_path, ready)
    terminal = {
        "contract_type": "safa_canonical_gpu_observer_terminal_v1",
        "policy_sha256": "9" * 64,
        "phase": "screen512",
        "status": "completed",
        "failure": None,
        "observer_ready": {
            "path": str(ready_path.resolve()),
            "sha256": wrapper._sha256_file(ready_path),
            "canonical_sha256": ready["observer_ready_sha256"],
        },
    }
    terminal["observer_terminal_sha256"] = wrapper._canonical_digest(
        terminal, "observer_terminal_sha256"
    )
    path = tmp_path / "observer_terminal.json"
    wrapper._write_exclusive(path, terminal)
    assert (
        wrapper._wait_observer_terminal(
            path, "9" * 64, "screen512", timeout_seconds=0.1
        )
        == terminal
    )
    failed = dict(terminal)
    failed["status"] = "failed"
    failed["failure"] = {"type": "RuntimeError", "message": "fixture"}
    failed["observer_terminal_sha256"] = wrapper._canonical_digest(
        failed, "observer_terminal_sha256"
    )
    failed_path = tmp_path / "failed_observer_terminal.json"
    wrapper._write_exclusive(failed_path, failed)
    with pytest.raises(RuntimeError, match="observer terminal contract"):
        wrapper._wait_observer_terminal(
            failed_path, "9" * 64, "screen512", timeout_seconds=0.1
        )


def test_gpu_tmux_command_uses_durable_wrapper_and_managed_observer(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    commands = module._tmux_commands(
        policy, policy_path, tmp_path / "campaign", "screen512"
    )
    assert "run_canonical_gpu_wrapper.py" in " ".join(commands["controller"])
    assert commands["monitor"] == []


def test_controller_rejects_observer_death_after_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy, request = _run_fixture(tmp_path)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    ready = load_json(Path(request["observer_ready"]["path"]), "observer ready")
    ready["admission"] = request["admission"]
    ready["observer_ready_sha256"] = canonical_digest(
        ready, "observer_ready_sha256"
    )
    ready_path = tmp_path / "observer_liveness_ready.json"
    write_exclusive_json(ready_path, ready)
    ready_binding = {
        **_bound(ready_path),
        "canonical_sha256": ready["observer_ready_sha256"],
    }
    admission_value = load_json(
        Path(request["admission"]["path"]), "liveness admission"
    )
    monkeypatch.setattr(
        module,
        "_gpu_snapshot",
        lambda: [
            {
                "index": row["physical_gpu_index"],
                "uuid": row["physical_gpu_uuid"],
            }
            for row in admission_value["snapshot"]["authorized_gpu_registry"]
        ],
    )
    monkeypatch.setattr(module, "_cpu_load_percent", lambda: 1.0)
    monkeypatch.setattr(module, "_memory_percent", lambda: 2.0)
    monkeypatch.setattr(module, "_disk_percent", lambda *_args: 3.0)
    monkeypatch.setattr(module, "_swap_pages", lambda: (0, 0))
    monkeypatch.setattr(module, "_gpu_compute_processes", lambda: [])
    heartbeat = module._monitor_sample(
        policy,
        paths,
        "smoke8",
        terminal=False,
        admission=request["admission"],
    )
    assert "observed_at" in heartbeat
    assert "completed_at" not in heartbeat
    _write_jsonl(paths["logs"] / "smoke8__observer.jsonl", [heartbeat])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=1),
    )
    with pytest.raises(CanonicalScreeningError, match="observer tmux died"):
        module._assert_observer_live(
            policy, paths, "smoke8", ready_binding
        )


def test_controller_has_final_observer_liveness_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monitor_path = tmp_path / "monitor.jsonl"
    monitor_path.write_text("{}\n", encoding="utf-8")
    guard = types.SimpleNamespace(
        raise_if_violated=lambda: None,
        stop=lambda: {
            "violation_reason": None,
            "thread_failure": None,
        },
    )
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    monkeypatch.setenv("TMUX", "fixture")
    wrapper, observer_launch = _wrapper_bindings(
        tmp_path, policy, "screen512"
    )
    monkeypatch.setattr(
        module,
        "_validate_gpu_wrapper_provenance",
        lambda *_args: (wrapper, observer_launch),
    )
    monkeypatch.setattr(
        module,
        "_write_gpu_controller_claim",
        lambda *_args: _mock_controller_claim(
            tmp_path / "controller_claim.json",
            {
                "controller_claim_sha256": "b" * 64,
                "wrapper_claim": wrapper,
                "observer_launch": observer_launch,
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "_prepare_gpu_ready_barrier",
        lambda *_args: {
            "admission_snapshot": {"authorized_gpu_registry": registry},
            "admission": {"canonical_sha256": "a" * 64},
            "requests": [],
            "resource_guard": guard,
            "monitor_path": monitor_path,
            "observer_ready": {"path": "fixture"},
            "controller_ready": {"path": "fixture"},
            "claim": {"controller_claim_sha256": "b" * 64},
        },
    )
    monkeypatch.setattr(
        module, "_append_monitor_sample", lambda *_args, **_kwargs: monitor_path
    )
    monkeypatch.setattr(
        module, "_write_gpu_controller_terminal", lambda *_args, **_kwargs: None
    )
    calls = 0

    def check_observer(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CanonicalScreeningError("fixture final observer death")

    monkeypatch.setattr(module, "_assert_observer_live", check_observer)
    with pytest.raises(CanonicalScreeningError, match="final observer death"):
        module._run_gpu_phase(policy, policy_path, paths, "screen512")
    assert calls == 2


def test_final_release_gpu_pid_race_blocks_first_worker_popen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monitor_path = tmp_path / "monitor.jsonl"
    monitor_path.write_text("{}\n", encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text("{}\n", encoding="utf-8")
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    guard = types.SimpleNamespace(
        raise_if_violated=lambda: None,
        stop=lambda: {
            "violation_reason": None,
            "thread_failure": None,
            "final_active_worker_pids": [],
        },
    )
    wrapper, launch = _wrapper_bindings(tmp_path, policy, "screen512")
    claim = {
        "controller_claim_sha256": "b" * 64,
        "wrapper_claim": wrapper,
        "observer_launch": launch,
    }
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module,
        "_validate_gpu_wrapper_provenance",
        lambda *_args: (wrapper, launch),
    )
    monkeypatch.setattr(
        module,
        "_write_gpu_controller_claim",
        lambda *_args: _mock_controller_claim(
            tmp_path / "claim.json", claim
        ),
    )
    monkeypatch.setattr(
        module,
        "_prepare_gpu_ready_barrier",
        lambda *_args: {
            "admission_snapshot": {"authorized_gpu_registry": registry},
            "admission": {"canonical_sha256": "a" * 64},
            "requests": [request_path],
            "resource_guard": guard,
            "monitor_path": monitor_path,
            "observer_ready": {"path": "observer"},
            "controller_ready": {"path": "controller"},
        },
    )
    monkeypatch.setattr(module, "_assert_observer_live", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "_write_final_release_admission",
        lambda *_args: (_ for _ in ()).throw(
            CanonicalScreeningError(
                "unknown compute PID observed at final release"
            )
        ),
    )
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker Popen must not run")
        ),
    )
    monkeypatch.setattr(
        module, "_append_monitor_sample", lambda *_args, **_kwargs: monitor_path
    )
    with pytest.raises(CanonicalScreeningError, match="unknown compute PID"):
        module._run_gpu_phase(policy, policy_path, paths, "screen512")
    terminal = load_json(
        paths["gpu_control"] / "screen512" / "controller_terminal.json",
        "race terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["stage"] == "final_release_admission"


def test_release_ready_tamper_fails_before_artifact_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    policy = {"campaign_id": "fixture", "policy_sha256": "1" * 64}
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    ready_path = tmp_path / "tampered_ready.json"
    ready_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_validate_controller_ready",
        lambda *_args: (_ for _ in ()).throw(
            CanonicalScreeningError("release controller ready tampered")
        ),
    )
    with pytest.raises(CanonicalScreeningError, match="tampered"):
        module._write_final_release_admission(
            policy,
            paths,
            "screen512",
            {"canonical_sha256": "2" * 64},
            {"path": str(ready_path.resolve())},
            {"path": str(ready_path.resolve())},
            [],
            types.SimpleNamespace(raise_if_violated=lambda: None),
        )
    assert not (
        paths["gpu_control"]
        / "screen512"
        / "final_release_admission.json"
    ).exists()


def test_post_worker_summary_exception_writes_failed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _controller_module()
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(ledger, [_row("config", None)])
    policy, policy_path, _ = _policy(tmp_path, ledger)
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    monitor_path = tmp_path / "monitor.jsonl"
    monitor_path.write_text("{}\n", encoding="utf-8")
    registry = [
        {
            "physical_gpu_index": index,
            "physical_gpu_uuid": _gpu_uuid(index),
        }
        for index in range(4)
    ]
    stop_summary = {
        "violation_reason": None,
        "thread_failure": None,
        "final_active_worker_pids": [],
    }
    guard = types.SimpleNamespace(
        raise_if_violated=lambda: None,
        stop=lambda: stop_summary,
    )
    wrapper, launch = _wrapper_bindings(tmp_path, policy, "screen512")
    claim = {
        "controller_claim_sha256": "b" * 64,
        "wrapper_claim": wrapper,
        "observer_launch": launch,
    }
    release = {
        "path": str((tmp_path / "release.json").resolve()),
        "sha256": "c" * 64,
        "canonical_sha256": "d" * 64,
    }
    monkeypatch.setenv("TMUX", "fixture")
    monkeypatch.setattr(
        module,
        "_validate_gpu_wrapper_provenance",
        lambda *_args: (wrapper, launch),
    )
    monkeypatch.setattr(
        module,
        "_write_gpu_controller_claim",
        lambda *_args: _mock_controller_claim(
            tmp_path / "claim.json", claim
        ),
    )
    monkeypatch.setattr(
        module,
        "_prepare_gpu_ready_barrier",
        lambda *_args: {
            "admission_snapshot": {"authorized_gpu_registry": registry},
            "admission": {"canonical_sha256": "a" * 64},
            "requests": [],
            "resource_guard": guard,
            "monitor_path": monitor_path,
            "observer_ready": {"path": "observer"},
            "controller_ready": {"path": "controller"},
        },
    )
    monkeypatch.setattr(module, "_assert_observer_live", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "_write_final_release_admission",
        lambda *_args: ({}, release),
    )
    monkeypatch.setattr(
        module, "_append_monitor_sample", lambda *_args, **_kwargs: monitor_path
    )
    monkeypatch.setattr(
        module,
        "_build_gpu_completion_summary",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("summary injected")
        ),
    )
    with pytest.raises(RuntimeError, match="summary injected"):
        module._run_gpu_phase(policy, policy_path, paths, "screen512")
    terminal = load_json(
        paths["gpu_control"] / "screen512" / "controller_terminal.json",
        "summary failure terminal",
    )
    assert terminal["status"] == "failed"
    assert terminal["stage"] == "completion_summary"
    assert "summary injected" in terminal["failure"]["message"]


def test_gpu_ready_barrier_positive_contract_chain(
    tmp_path: Path,
) -> None:
    module = _controller_module()
    policy, manifest, manifest_path, policy_path, admission, _ = _manifest_fixture(
        tmp_path
    )
    paths = module._paths(tmp_path / "campaign", policy["policy_sha256"])
    wrapper, observer_launch = _wrapper_bindings(
        tmp_path, policy, "smoke8"
    )
    claim, claim_path = module._write_gpu_controller_claim(
        policy, paths, "smoke8", wrapper, observer_launch
    )
    candidates = []
    template = manifest["candidates"][0]
    for index in range(193):
        candidates.append(
            {
                "candidate_id": f"candidate-{index:03d}",
                "checkpoint_sha256": f"{index:064x}",
                "checkpoint_model": template["checkpoint_model"],
            }
        )
    intent, intent_path = module._write_request_intent_manifest(
        policy,
        paths,
        "smoke8",
        ("primary", "repeat"),
        {
            "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
            "candidates": candidates,
        },
        admission,
    )

    def write_artifact(
        name: str, digest_field: str, value: dict
    ) -> tuple[dict, Path]:
        value[digest_field] = canonical_digest(value, digest_field)
        path = tmp_path / "barrier" / f"{name}.json"
        write_exclusive_json(path, value)
        return value, path

    internal, internal_path = write_artifact(
        "internal",
        "monitor_sample_sha256",
        {"kind": "internal", "policy_sha256": policy["policy_sha256"]},
    )
    guard, guard_path = write_artifact(
        "guard",
        "resource_window_sha256",
        {"kind": "guard", "policy_sha256": policy["policy_sha256"]},
    )
    recheck, recheck_path = write_artifact(
        "recheck",
        "resource_recheck_sha256",
        {"kind": "recheck", "policy_sha256": policy["policy_sha256"]},
    )
    controller, controller_path, controller_binding = (
        module._write_controller_ready(
            policy,
            paths,
            "smoke8",
            claim,
            admission,
            intent,
            intent_path,
            internal,
            internal_path,
            guard,
            guard_path,
            recheck,
            recheck_path,
            claim_path,
        )
    )
    assert (
        module._validate_controller_ready(
            controller, policy, "smoke8", admission
        )
        == controller
    )
    observer_claim, observer_claim_path = write_artifact(
        "observer_claim",
        "observer_claim_sha256",
        {"kind": "observer_claim", "policy_sha256": policy["policy_sha256"]},
    )
    observer_sample, observer_sample_path = write_artifact(
        "observer_sample",
        "monitor_sample_sha256",
        {"kind": "observer_sample", "policy_sha256": policy["policy_sha256"]},
    )
    observer = {
        "schema_version": 1,
        "contract_type": "safa_canonical_gpu_observer_ready_v1",
        "campaign_id": policy["campaign_id"],
        "phase": "smoke8",
        "policy_sha256": policy["policy_sha256"],
        "admission_sha256": admission["canonical_sha256"],
        "controller_ready_sha256": controller["controller_ready_sha256"],
        "observer_claim_sha256": observer_claim["observer_claim_sha256"],
        "wrapper_claim_sha256": wrapper["canonical_sha256"],
        "observer_launch_sha256": observer_launch["canonical_sha256"],
        "observer_claim": module._artifact_binding(
            observer_claim_path, observer_claim["observer_claim_sha256"]
        ),
        "wrapper_claim": wrapper,
        "observer_launch": observer_launch,
        "controller_ready": controller_binding,
        "admission": admission,
        "first_observer_sample": module._artifact_binding(
            observer_sample_path, observer_sample["monitor_sample_sha256"]
        ),
    }
    observer["observer_ready_sha256"] = canonical_digest(
        observer, "observer_ready_sha256"
    )
    observer_path = (
        paths["gpu_control"] / "smoke8" / "observer_ready.json"
    )
    write_exclusive_json(observer_path, observer)
    observer_binding = module._artifact_binding(
        observer_path, observer["observer_ready_sha256"]
    )
    assert (
        module._validate_observer_ready(
            observer, policy, "smoke8", controller, admission
        )
        == observer
    )
    request = build_run_request(
        policy,
        policy_path,
        manifest,
        manifest_path,
        manifest["candidates"][0],
        "smoke8",
        "primary",
        tmp_path / "runs",
        admission,
        controller_binding,
        observer_binding,
    )
    assert validate_run_request(request, policy) == request
    _assert_ready_barrier(request, policy)
    internal_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CanonicalScreeningError, match="file binding"):
        _assert_ready_barrier(request, policy)


def _patch_preclaim_terminal_publisher_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    launcher: Any,
    *,
    cleanup_completed_at: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    attempt_root = Path("/sealed/attempt")
    artifacts = {
        "attempt": str(attempt_root / "consumer_attempt.json"),
        "controller_cleanup": str(
            attempt_root / "controller_cleanup.json"
        ),
        "terminal": str(attempt_root / "consumer_terminal.json"),
        "join": str(attempt_root / "consumer_join.json"),
        "cleanup": str(attempt_root / "consumer_cleanup.json"),
    }
    receipt = {
        "started_at": "2026-07-29T00:00:00.000000Z",
        "gate_execution_terminal_path": str(
            attempt_root / "gate_terminal.json"
        ),
        "pane_fault_consumer": {"artifacts": artifacts},
    }
    intent = {
        "attempt_id": "a" * 64,
        "reason": "invalid_claim",
        "stage": "wrapper_claim_validation",
        "launch_receipt": _shared_binding("launch_receipt"),
        "verified_implementations": {"contract": "sealed"},
        "pane_fault_consumer_chain": {"consumer": "sealed"},
    }
    publication = {
        "intent": intent,
        "artifact": _shared_binding("preclaim_intent"),
    }
    written: list[dict[str, Any]] = []

    def sealed(
        path: Path, *, digest_field: str, label: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        del label
        if path.name == "consumer_cleanup.json":
            value = {"completed_at": cleanup_completed_at}
        elif path.name == "launch_terminal.json":
            assert written
            value = written[-1]
        else:
            value = {"sealed": path.name}
        return (
            value,
            {
                "path": str(path),
                "sha256": f"{digest_field}:file",
                "canonical_sha256": f"{digest_field}:canonical",
            },
            {"path": str(path), "sealed": True},
        )

    def lifecycle(role: str, inode: int) -> dict[str, Any]:
        channel = {
            "path": f"/sealed/{role}_lifecycle_wait.channel",
            "device": 7,
            "inode": inode,
            "mode": 0o100600,
            "uid": 1000,
            "nlink": 1,
            "size": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "directory_device": 7,
            "directory_inode": 300,
        }
        record = _lifecycle_wait_record(
            launcher,
            channel,
            role=role,
        )
        return {
            "snapshot": {
                "channel_authority": channel,
                "sha256": hashlib.sha256(
                    f"{role}-lifecycle".encode()
                ).hexdigest(),
                "record": record,
            }
        }

    gate_lifecycle = lifecycle("gate", 301)
    consumer_lifecycle = lifecycle("consumer", 302)
    monkeypatch.setattr(launcher, "_sealed_finalization_json", sealed)
    monkeypatch.setattr(
        launcher,
        "_read_formal_gate_lifecycle_status",
        lambda **_kwargs: gate_lifecycle,
    )
    monkeypatch.setattr(
        launcher,
        "_read_formal_consumer_lifecycle_status",
        lambda **_kwargs: consumer_lifecycle,
    )

    def build_terminal(**kwargs: Any) -> dict[str, Any]:
        return {
            "launch_terminal_sha256": "terminal:canonical",
            "completed_at": kwargs["completed_at"],
        }

    monkeypatch.setattr(
        launcher, "build_launch_terminal_v2", build_terminal
    )
    return receipt, publication, written


def test_preclaim_finalizer_actions_have_independent_fixed_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    state = launcher.PreclaimFinalizationEvidenceState.GATE_EVIDENCE
    gate = {"owner": "gate"}
    consumer = {"owner": "consumer"}
    observed_actions: list[Mapping[str, Any]] = []
    monkeypatch.setattr(
        launcher,
        "_read_preclaim_finalizer_state",
        lambda **kwargs: (
            observed_actions.append(kwargs["action"])
            or (state, gate, consumer)
        ),
    )
    ticks = iter((10.0, 10.1, 10.2, 10.3, 20.0, 20.1, 20.2, 20.3))
    monkeypatch.setattr(
        launcher.time, "monotonic", lambda: next(ticks)
    )

    timings = []
    for name in ("formal_gate_lifecycle", "consumer_chain"):
        result = launcher._run_preclaim_finalization_action(
            action_name=name,
            timeout_seconds=5.0,
            expected_states={state},
            attempt_root=Path("/sealed/attempt"),
            intent_publication={},
            launch_receipt={},
            launch_receipt_identity={},
            live_gate_owner_seal=gate,
            live_consumer_owner_seal=consumer,
            launch_terminal_path=Path(
                "/sealed/attempt/launch_terminal.json"
            ),
            operation=lambda action, _gate, _consumer: (
                observed_actions.append(action)
            ),
        )
        timings.append(result[3])

    assert timings == [
        {
            "action": "formal_gate_lifecycle",
            "started": 10.0,
            "deadline": 15.0,
            "ended": 10.3,
        },
        {
            "action": "consumer_chain",
            "started": 20.0,
            "deadline": 25.0,
            "ended": 20.3,
        },
    ]
    assert observed_actions[0] is observed_actions[1]
    assert observed_actions[1] is observed_actions[2]
    assert observed_actions[3] is observed_actions[4]
    assert observed_actions[4] is observed_actions[5]
    assert observed_actions[0] is not observed_actions[3]


def test_preclaim_finalizer_runs_six_actions_with_p3a_rereads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    states = iter(
        (
            launcher.PreclaimFinalizationEvidenceState.INTENT_ONLY,
            launcher.PreclaimFinalizationEvidenceState.INTENT_ONLY,
            launcher.PreclaimFinalizationEvidenceState.INTENT_ONLY,
            launcher.PreclaimFinalizationEvidenceState.INTENT_ONLY,
            launcher.PreclaimFinalizationEvidenceState.GATE_EVIDENCE,
            launcher.PreclaimFinalizationEvidenceState.GATE_EVIDENCE,
            launcher.PreclaimFinalizationEvidenceState.GATE_EVIDENCE,
            launcher.PreclaimFinalizationEvidenceState.GATE_EVIDENCE,
            (
                launcher.PreclaimFinalizationEvidenceState
                .CONSUMER_TERMINAL_CHAIN
            ),
            (
                launcher.PreclaimFinalizationEvidenceState
                .CONSUMER_TERMINAL_CHAIN
            ),
            (
                launcher.PreclaimFinalizationEvidenceState
                .CONSUMER_CLEANUP_PRESENT
            ),
            (
                launcher.PreclaimFinalizationEvidenceState
                .CONSUMER_CLEANUP_PRESENT
            ),
            launcher.PreclaimFinalizationEvidenceState.LAUNCH_TERMINAL,
        )
    )
    gate = {"owner": "gate"}
    consumer = {"owner": "consumer"}
    reread_actions: list[str] = []

    def read_state(**kwargs: Any) -> tuple[Any, Any, Any]:
        reread_actions.append(str(kwargs["action"]["action"]))
        return next(states), gate, consumer

    monkeypatch.setattr(
        launcher, "_read_preclaim_finalizer_state", read_state
    )
    tick = {"value": 0.0}

    def monotonic() -> float:
        tick["value"] += 0.01
        return tick["value"]

    monkeypatch.setattr(launcher.time, "monotonic", monotonic)
    operations: list[str] = []
    monkeypatch.setattr(
        launcher,
        "_terminate_exact_wrapper_child",
        lambda *_args: operations.append("exact_terminate"),
    )
    monkeypatch.setattr(
        launcher,
        "_wait_preclaim_gate_terminal_and_dead",
        lambda **_kwargs: operations.append("gate_wait_dead"),
    )
    monkeypatch.setattr(
        launcher,
        "_wait_preclaim_controller_cleanup",
        lambda **_kwargs: operations.append("controller_cleanup_wait"),
    )

    def consumer_wait(**_kwargs: Any) -> tuple[Any, Any]:
        operations.append("consumer_chain_dead")
        return gate, consumer

    monkeypatch.setattr(
        launcher,
        "_wait_preclaim_consumer_terminal_and_dead",
        consumer_wait,
    )
    monkeypatch.setattr(
        launcher,
        "_read_formal_gate_lifecycle_status",
        lambda **_kwargs: operations.append("formal_gate"),
    )
    monkeypatch.setattr(
        launcher,
        "_read_formal_consumer_lifecycle_status",
        lambda **_kwargs: operations.append("formal_consumer"),
    )
    monkeypatch.setattr(
        launcher,
        "join_pane_fault_consumer",
        lambda **_kwargs: operations.append("join_cleanup"),
    )
    monkeypatch.setattr(
        launcher,
        "_publish_preclaim_launch_terminal_v2",
        lambda **_kwargs: operations.append("terminal_v2"),
    )
    receipt = {
        "pane_fault_consumer": {
            "artifacts": {"attempt": "/sealed/consumer_attempt.json"}
        }
    }
    result = launcher._resume_or_finalize_preclaim_failure(
        config=Path("/sealed/config.json"),
        timeout_seconds=1.0,
        attempt_root=Path("/sealed/attempt"),
        intent_publication={},
        launch_receipt=receipt,
        launch_receipt_identity={},
        live_gate_owner_seal=gate,
        live_consumer_owner_seal=consumer,
        launch_terminal_path=Path(
            "/sealed/attempt/launch_terminal.json"
        ),
    )
    assert operations == [
        "exact_terminate",
        "gate_wait_dead",
        "formal_gate",
        "controller_cleanup_wait",
        "consumer_chain_dead",
        "formal_gate",
        "formal_consumer",
        "join_cleanup",
        "terminal_v2",
    ]
    expected_action_names = [
        "exact_wrapper_termination",
        "gate_terminal_and_dead",
        "formal_gate_lifecycle",
        "consumer_terminal_chain_and_formal_dead",
        "consumer_join_and_cleanup",
        "launch_terminal_v2_publication",
    ]
    assert [
        timing["action"] for timing in result["action_timings"]
    ] == expected_action_names
    assert reread_actions == [
        "state_discovery",
        *[
            action_name
            for action_name in expected_action_names
            for _ in range(2)
        ],
    ]


def test_preclaim_launch_terminal_uses_sealed_cleanup_completed_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    sealed_completed_at = "2026-07-29T00:00:07.000000Z"
    receipt, publication, written = (
        _patch_preclaim_terminal_publisher_dependencies(
            monkeypatch,
            launcher,
            cleanup_completed_at=sealed_completed_at,
        )
    )
    monkeypatch.setattr(
        launcher,
        "_write_exclusive",
        lambda _path, value: written.append(dict(value)),
    )
    terminal = launcher._publish_preclaim_launch_terminal_v2(
        attempt_root=Path("/sealed/attempt"),
        intent_publication=publication,
        launch_receipt=receipt,
        launch_receipt_identity={"sealed": "receipt"},
        gate_owner_seal={"sealed": "gate"},
        consumer_owner_seal={"sealed": "consumer"},
        launch_terminal_path=Path(
            "/sealed/attempt/launch_terminal.json"
        ),
    )
    assert written[0]["completed_at"] == sealed_completed_at
    assert terminal["completed_at"] == sealed_completed_at


def test_preclaim_missing_session_waits_for_sealed_cleanup_until_typed_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    paths = {
        "gate_terminal": Path("/sealed/gate.json"),
        "controller_cleanup": Path("/sealed/controller_cleanup.json"),
        "consumer_terminal": Path("/sealed/consumer_terminal.json"),
        "consumer_join": Path("/sealed/consumer_join.json"),
        "consumer_cleanup": Path("/sealed/consumer_cleanup.json"),
        "launch_terminal": Path("/sealed/launch_terminal.json"),
    }
    monkeypatch.setattr(
        launcher,
        "_preclaim_finalizer_artifact_paths",
        lambda **_kwargs: paths,
    )
    polls: list[str] = []

    def partial(_paths: Mapping[str, Path]) -> dict[str, bool]:
        polls.append("sealed-evidence-poll")
        return {name: False for name in paths}

    monkeypatch.setattr(
        launcher, "_require_preclaim_finalizer_partial_order", partial
    )
    monkeypatch.setattr(launcher, "_tmux_pane", lambda _session: None)
    monkeypatch.setattr(
        launcher,
        "_tmux_owner_seal",
        lambda *_args: pytest.fail(
            "missing session must not synthesize a live owner"
        ),
    )
    monkeypatch.setattr(launcher.time, "sleep", lambda _seconds: None)
    ticks = iter((0.5, 1.0))
    monkeypatch.setattr(
        launcher.time, "monotonic", lambda: next(ticks)
    )
    action = {
        "action": "gate_terminal_and_dead",
        "started": 0.0,
        "deadline": 1.0,
    }
    with pytest.raises(
        launcher.PreclaimFinalizationTimeoutError
    ) as captured:
        launcher._preclaim_finalizer_owner_seals(
            launch_receipt={},
            launch_terminal_path=paths["launch_terminal"],
            live_gate_owner_seal={
                "session": "missing-gate",
                "owner_nonce": "g" * 64,
            },
            live_consumer_owner_seal={
                "session": "missing-consumer",
                "owner_nonce": "c" * 64,
            },
            action=action,
        )
    assert polls == [
        "sealed-evidence-poll",
        "sealed-evidence-poll",
    ]
    assert captured.value.action == "gate_terminal_and_dead"
    assert captured.value.started == 0.0
    assert captured.value.deadline == 1.0
    assert captured.value.ended == 1.0


def test_preclaim_terminal_publish_error_identity_and_secondary_failures_survive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    receipt, publication, written = (
        _patch_preclaim_terminal_publisher_dependencies(
            monkeypatch,
            launcher,
            cleanup_completed_at="2026-07-29T00:00:07.000000Z",
        )
    )
    assert written == []
    failure = launcher.LauncherTerminalPublishError(
        Path("/sealed/attempt/launch_terminal.json"),
        OSError("durability failed"),
    )
    failure.add_secondary_failure(
        stage="post_handoff_failure_publication",
        failure=OSError("secondary durability failed"),
    )
    secondary = copy.deepcopy(failure.secondary_failures)

    def fail_publish(_path: Path, _value: Mapping[str, Any]) -> None:
        raise failure

    monkeypatch.setattr(launcher, "_write_exclusive", fail_publish)
    with pytest.raises(
        launcher.LauncherTerminalPublishError
    ) as captured:
        launcher._publish_preclaim_launch_terminal_v2(
            attempt_root=Path("/sealed/attempt"),
            intent_publication=publication,
            launch_receipt=receipt,
            launch_receipt_identity={"sealed": "receipt"},
            gate_owner_seal={"sealed": "gate"},
            consumer_owner_seal={"sealed": "consumer"},
            launch_terminal_path=Path(
                "/sealed/attempt/launch_terminal.json"
            ),
        )
    assert captured.value is failure
    assert captured.value.failure is failure.failure
    assert captured.value.secondary_failures == secondary
