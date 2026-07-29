from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

import safa.closeout.preflight_launch_contract as contract


def _channel(name: str) -> dict[str, Any]:
    inode = 17 + int(hashlib.sha256(name.encode()).hexdigest()[:6], 16)
    return {
        "path": f"/contract/{name}.channel",
        "device": 7,
        "inode": inode,
        "mode": 0o100600,
        "uid": 1000,
        "nlink": 1,
        "size": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "directory_device": 7,
        "directory_inode": 11,
    }


def _process(pid: int, ppid: int) -> dict[str, int]:
    return {
        "pid": pid,
        "ppid": ppid,
        "pgid": pid,
        "sid": pid,
        "start_ticks": pid * 10,
    }


def _owner() -> dict[str, Any]:
    return {
        "session": "safa-fixture",
        "pane": "%1",
        "pane_pid": 10,
        "pane_dead": False,
        "pane_dead_status": None,
        "pane_process": _process(10, 1),
        "owner_nonce": "a" * 64,
        "tmux_server": {
            "server_pid": 20,
            "server_process": _process(20, 1),
            "socket_path": "/tmp/tmux-fixture",
            "socket_device": 3,
            "socket_inode": 4,
        },
    }


def _source(role: str) -> dict[str, Any]:
    kind = "launch_receipt" if role == "gate" else "consumer_attempt"
    path = f"/contract/{kind}.json"
    return {
        "kind": kind,
        "binding": {
            "path": path,
            "sha256": "b" * 64,
            "canonical_sha256": "c" * 64,
        },
        "file_identity": {
            "path": path,
            "device": 7,
            "inode": 19,
            "mode": 0o100600,
            "size": 97,
        },
    }


def _publisher(role: str) -> dict[str, Any]:
    return {
        "path": "/contract/launcher.py",
        "sha256": "d" * 64,
        "file_identity": {
            "path": "/contract/launcher.py",
            "device": 7,
            "inode": 23,
            "mode": 0o100644,
            "size": 101,
        },
        "role": f"{role}_lifecycle_wait_supervisor",
    }


def _raw_wait(
    *,
    role: str = "gate",
    signal_number: int | None = None,
) -> dict[str, Any]:
    child = _process(11, 10)
    if signal_number is None:
        waitid_code = 1
        waitid_status = 117 if role == "gate" else 118
        raw_status = waitid_status << 8
    else:
        waitid_code = 2
        waitid_status = signal_number
        raw_status = signal_number
    return contract.build_lifecycle_raw_wait_v3(
        role=role,
        policy_sha256="e" * 64,
        attempt_id="f" * 64,
        source_artifact=_source(role),
        wait_channel=_channel(f"{role}-wait"),
        publisher=_publisher(role),
        supervisor_owner_seal=_owner(),
        child_process=child,
        waitid_si_pid=child["pid"],
        waitid_si_code=waitid_code,
        waitid_si_status=waitid_status,
        waited_pid=child["pid"],
        wait_status_raw=raw_status,
        started_at="2026-07-29T00:00:00+00:00",
        reaped_at="2026-07-29T00:00:01+00:00",
    )


@pytest.mark.parametrize(
    ("role", "signal_number", "exit_kind", "returncode"),
    (
        ("gate", None, "exit", 117),
        ("consumer", None, "exit", 118),
        ("gate", 15, "signal", -15),
        ("consumer", 9, "signal", -9),
    ),
)
def test_lifecycle_raw_wait_v3_preserves_only_structural_wait(
    role: str,
    signal_number: int | None,
    exit_kind: str,
    returncode: int,
) -> None:
    value = _raw_wait(role=role, signal_number=signal_number)
    assert value["schema_version"] == 3
    assert value["exit_kind"] == exit_kind
    assert value["returncode"] == returncode
    assert "worker_started" not in value
    assert "terminal" not in value
    assert "ownership_chain" not in value


@pytest.mark.parametrize(
    "mutation",
    (
        "extra",
        "missing",
        "role",
        "source_kind",
        "publisher_role",
        "owner_dead",
        "owner_process",
        "child_parent",
        "waitid_pid",
        "waited_pid",
        "waitid_status",
        "raw_status",
        "derived",
        "naive_time",
        "reversed_time",
        "digest",
    ),
)
def test_lifecycle_raw_wait_v3_mutations_fail_closed(
    mutation: str,
) -> None:
    value = copy.deepcopy(_raw_wait())
    if mutation == "extra":
        value["terminal"] = None
    elif mutation == "missing":
        value.pop("reaped_at")
    elif mutation == "role":
        value["role"] = "consumer"
    elif mutation == "source_kind":
        value["source_artifact"]["kind"] = "consumer_attempt"
    elif mutation == "publisher_role":
        value["publisher"]["role"] = "consumer_lifecycle_wait_supervisor"
    elif mutation == "owner_dead":
        value["supervisor_owner_seal"]["pane_dead"] = True
    elif mutation == "owner_process":
        value["supervisor_owner_seal"]["pane_process"]["pid"] = 99
    elif mutation == "child_parent":
        value["child_process"]["ppid"] = 99
    elif mutation == "waitid_pid":
        value["waitid_si_pid"] = 12
    elif mutation == "waited_pid":
        value["waited_pid"] = 12
    elif mutation == "waitid_status":
        value["waitid_si_status"] = 116
    elif mutation == "raw_status":
        value["wait_status_raw"] = 116 << 8
    elif mutation == "derived":
        value["returncode"] = 0
    elif mutation == "naive_time":
        value["reaped_at"] = "2026-07-29T00:00:01"
    elif mutation == "reversed_time":
        value["reaped_at"] = "2026-07-28T23:59:59+00:00"
    elif mutation == "digest":
        value["lifecycle_raw_wait_sha256"] = "0" * 64
    if mutation != "digest":
        value["lifecycle_raw_wait_sha256"] = contract.canonical_digest(
            value, "lifecycle_raw_wait_sha256"
        )
    with pytest.raises(contract.PreflightLaunchContractError):
        contract.validate_lifecycle_raw_wait_v3(
            value, role="gate", label="mutated raw wait"
        )


def _publish_failure_record() -> dict[str, Any]:
    return contract.build_publish_failure_record(
        commit_state="precommit_failed_clean",
        stage="write_body",
        message="write failed",
        directory_seal={"device": 7, "inode": 11},
        payload={"target": "/contract/gate-wait.channel"},
        temporary=None,
        error_number=5,
        secondary_failures=[],
    )


def _raw_publish_fault() -> dict[str, Any]:
    return contract.build_lifecycle_raw_wait_publish_failure_v1(
        role="gate",
        policy_sha256="e" * 64,
        attempt_id="f" * 64,
        source_artifact=_source("gate"),
        target_channel=_channel("gate-wait"),
        fault_channel=_channel("gate-wait-publish-fault"),
        publisher=_publisher("gate"),
        supervisor_owner_seal=_owner(),
        child_process=_process(11, 10),
        intended_raw_wait_sha256="1" * 64,
        publish_failure=_publish_failure_record(),
        recorded_at="2026-07-29T00:00:02+00:00",
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "extra",
        "missing",
        "role",
        "source",
        "target",
        "fault_channel",
        "publisher",
        "owner",
        "child",
        "intended",
        "publish_failure",
        "time",
        "digest",
    ),
)
def test_raw_wait_publish_failure_v1_mutations_fail_closed(
    mutation: str,
) -> None:
    value = copy.deepcopy(_raw_publish_fault())
    if mutation == "extra":
        value["raw_wait"] = {}
    elif mutation == "missing":
        value.pop("publish_failure")
    elif mutation == "role":
        value["role"] = "consumer"
    elif mutation == "source":
        value["source_artifact"]["kind"] = "consumer_attempt"
    elif mutation == "target":
        value["target_channel"]["path"] = "relative"
    elif mutation == "fault_channel":
        value["fault_channel"] = value["target_channel"]
    elif mutation == "publisher":
        value["publisher"]["role"] = "foreign"
    elif mutation == "owner":
        value["supervisor_owner_seal"]["owner_nonce"] = "bad"
    elif mutation == "child":
        value["child_process"]["ppid"] = 99
    elif mutation == "intended":
        value["intended_raw_wait_sha256"] = "bad"
    elif mutation == "publish_failure":
        value["publish_failure"]["commit_state"] = "unknown"
    elif mutation == "time":
        value["recorded_at"] = ""
    elif mutation == "digest":
        value["lifecycle_raw_wait_publish_failure_sha256"] = "0" * 64
    if mutation != "digest":
        value["lifecycle_raw_wait_publish_failure_sha256"] = (
            contract.canonical_digest(
                value,
                "lifecycle_raw_wait_publish_failure_sha256",
            )
        )
    with pytest.raises(contract.PreflightLaunchContractError):
        contract.validate_lifecycle_raw_wait_publish_failure_v1(
            value, role="gate", label="mutated raw publish fault"
        )


def test_postclaim_v5_profile_is_one_exact_non_mixing_stack() -> None:
    profile = contract.build_postclaim_finalization_profile_v1()
    assert profile == contract.validate_postclaim_finalization_profile_v1(
        profile
    )
    assert profile["raw_wait"] == {
        "contract_type": contract.LIFECYCLE_RAW_WAIT_V3_CONTRACT_TYPE,
        "schema_version": 3,
    }
    assert profile["controller_cleanup"]["schema_version"] == 3
    assert profile["consumer_terminal"]["schema_version"] == 3
    assert profile["consumer_join"]["schema_version"] == 4
    assert (
        contract.LAUNCH_RECEIPT_V5_CONTRACT_TYPE
        in contract.shared_contract_types()
    )
    assert "postclaim_finalization_profile" in (
        contract.shared_schema_keysets()["launch_receipt_v5"]
    )


def test_postclaim_contract_is_pure_and_node2_is_unreachable() -> None:
    path = Path(contract.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert not imported_roots.intersection(
        {"io", "os", "pathlib", "signal", "subprocess"}
    )
    launcher_path = path.parents[3] / (
        "scripts/run_canonical_preflight_launcher.py"
    )
    launcher_tree = ast.parse(
        launcher_path.read_text(encoding="utf-8")
    )
    all_calls = {
        node.func.id
        for node in ast.walk(launcher_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "build_lifecycle_raw_wait_v3",
            "validate_lifecycle_raw_wait_v3",
            "build_lifecycle_raw_wait_publish_failure_v1",
            "validate_lifecycle_raw_wait_publish_failure_v1",
            "build_launch_receipt_v5",
            "validate_launch_receipt_v5",
        }
    }
    assert all_calls == {
        "build_lifecycle_raw_wait_v3",
        "validate_lifecycle_raw_wait_v3",
        "build_lifecycle_raw_wait_publish_failure_v1",
        "validate_lifecycle_raw_wait_publish_failure_v1",
        "validate_launch_receipt_v5",
    }
    assert "build_launch_receipt_v5" not in all_calls
    functions = {
        node.name: node
        for node in launcher_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    edges = {
        name: {
            call.func.id
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in functions
        }
        for name, node in functions.items()
    }
    reachable: set[str] = set()
    pending = ["main", "launch_preflight"]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(edges.get(name, set()) - reachable)
    node2_functions = {
        "_write_lifecycle_raw_wait_v3",
        "_write_lifecycle_raw_wait_publish_failure_v1",
        "_require_empty_lifecycle_raw_wait_publish_fault_channel",
        "_publish_gate_raw_wait_after_reap",
        "_gate_wait_supervisor_v5_reap_and_publish_unconnected",
    }
    assert node2_functions.isdisjoint(reachable)
