from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import safa.closeout.preflight_launch_contract as contract


def _launcher_module() -> Any:
    path = (
        Path(__file__).parents[1]
        / "scripts/run_canonical_preflight_launcher.py"
    )
    spec = importlib.util.spec_from_file_location(
        "postclaim_gate_raw_wait_launcher_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in module._SHARED_CONTRACT_EXPORTS:
        setattr(module, name, getattr(contract, name))
    return module


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
        "session": "safa-node2",
        "pane": "%1",
        "pane_pid": 10,
        "pane_dead": False,
        "pane_dead_status": None,
        "pane_process": _process(10, 1),
        "owner_nonce": "a" * 64,
        "tmux_server": {
            "server_pid": 20,
            "server_process": _process(20, 1),
            "socket_path": "/tmp/tmux-node2",
            "socket_device": 3,
            "socket_inode": 4,
        },
    }


def _source() -> dict[str, Any]:
    return {
        "kind": "launch_receipt",
        "binding": {
            "path": "/contract/launch_receipt.json",
            "sha256": "b" * 64,
            "canonical_sha256": "c" * 64,
        },
        "file_identity": {
            "path": "/contract/launch_receipt.json",
            "device": 7,
            "inode": 19,
            "mode": 0o100600,
            "size": 97,
        },
    }


def _publisher() -> dict[str, Any]:
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
        "role": "gate_lifecycle_wait_supervisor",
    }


def _raw_wait(
    binding: dict[str, Any],
    *,
    signal_number: int | None = None,
) -> dict[str, Any]:
    child = _process(11, 10)
    if signal_number is None:
        waitid_code = os.CLD_EXITED
        waitid_status = 117
        raw_status = 117 << 8
    else:
        waitid_code = os.CLD_KILLED
        waitid_status = int(signal_number)
        raw_status = int(signal_number)
    return contract.build_lifecycle_raw_wait_v3(
        role="gate",
        policy_sha256="e" * 64,
        attempt_id="f" * 64,
        source_artifact=_source(),
        wait_channel=binding,
        publisher=_publisher(),
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


def _writer(
    launcher: Any,
    tmp_path: Path,
    name: str,
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


@pytest.mark.parametrize(
    ("signal_number", "exit_kind", "returncode"),
    (
        (None, "exit", 117),
        (signal.SIGTERM, "signal", -signal.SIGTERM),
        (signal.SIGKILL, "signal", -signal.SIGKILL),
    ),
)
def test_raw_wait_v3_writer_preserves_exit_and_signal(
    tmp_path: Path,
    signal_number: int | None,
    exit_kind: str,
    returncode: int,
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "raw_wait.channel"
    )
    record = _raw_wait(binding, signal_number=signal_number)
    try:
        assert (
            launcher._write_lifecycle_raw_wait_v3(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
            == record
        )
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)
    assert record["exit_kind"] == exit_kind
    assert record["returncode"] == returncode
    assert path.read_bytes().endswith(
        launcher.LIFECYCLE_WAIT_CHANNEL_COMMIT_PREFIX
        + hashlib.sha256(
            launcher._build_lifecycle_wait_channel_frame(record)[0]
        ).hexdigest().encode("ascii")
        + b"\n"
    )


def test_raw_wait_v3_writer_resumes_exact_committed_record(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    _path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "collision.channel"
    )
    record = _raw_wait(binding)
    try:
        launcher._write_lifecycle_raw_wait_v3(
            descriptor,
            directory_descriptor,
            binding,
            record,
            role="gate",
        )
        assert (
            launcher._write_lifecycle_raw_wait_v3(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
            == record
        )
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def test_raw_wait_v3_writer_precommit_failure_is_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "precommit.channel"
    )
    record = _raw_wait(binding)

    def fail_before_write(
        _descriptor: int,
        _data: bytes,
        _offset: int,
        *,
        label: str,
    ) -> None:
        assert label == "lifecycle raw wait channel body"
        raise OSError(5, "injected precommit failure")

    monkeypatch.setattr(launcher, "_pwrite_all", fail_before_write)
    try:
        with pytest.raises(
            launcher.LifecycleRawWaitPublishError
        ) as raised:
            launcher._write_lifecycle_raw_wait_v3(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
        assert (
            raised.value.commit_state
            == "precommit_failed_clean"
        )
        assert path.read_bytes() == b""
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def test_raw_wait_v3_writer_body_fsync_failure_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "quarantined.channel"
    )
    record = _raw_wait(binding)
    real_fsync = launcher.os.fsync

    def fail_channel_fsync(fd: int) -> None:
        if fd == descriptor:
            raise OSError(5, "injected body fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(launcher.os, "fsync", fail_channel_fsync)
    try:
        with pytest.raises(
            launcher.LifecycleRawWaitPublishError
        ) as raised:
            launcher._write_lifecycle_raw_wait_v3(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
        assert (
            raised.value.commit_state
            == "durability_unknown_quarantined"
        )
        assert path.stat().st_size > 0
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def test_raw_wait_v3_writer_postcommit_failure_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    _path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "postcommit.channel"
    )
    record = _raw_wait(binding)
    real_require = launcher._require_named_lifecycle_wait_channel
    calls = 0

    def fail_after_commit(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError(5, "injected postcommit verify failure")
        return real_require(*args, **kwargs)

    monkeypatch.setattr(
        launcher,
        "_require_named_lifecycle_wait_channel",
        fail_after_commit,
    )
    try:
        with pytest.raises(
            launcher.LifecycleRawWaitPublishError
        ) as raised:
            launcher._write_lifecycle_raw_wait_v3(
                descriptor,
                directory_descriptor,
                binding,
                record,
                role="gate",
            )
        assert (
            raised.value.commit_state
            == "committed_cleanup_error"
        )
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)


def test_raw_wait_missing_only_exact_keeps_fault_channel_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    _path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "raw.channel"
    )
    fault_path = tmp_path / "raw_publish_fault.channel"
    fault_binding = launcher._create_fault_channel(fault_path)
    fault_descriptor = launcher._open_presealed_fault_channel(
        tmp_path,
        fault_binding,
        name=fault_path.name,
    )
    record = _raw_wait(binding)
    receipt = {
        "policy_sha256": record["policy_sha256"],
        "gate_lifecycle_wait_channel": binding,
        "gate_lifecycle_wait_publisher": record["publisher"],
        "gate_lifecycle_wait_publish_fault_channel": fault_binding,
    }
    monkeypatch.setattr(
        launcher,
        "_sealed_lifecycle_artifact",
        lambda *args, **kwargs: record["source_artifact"],
    )
    monkeypatch.setattr(
        launcher,
        "build_lifecycle_raw_wait_v3",
        lambda **kwargs: record,
    )
    try:
        launcher._write_lifecycle_raw_wait_v3(
            descriptor,
            directory_descriptor,
            binding,
            record,
            role="gate",
        )
        assert launcher._publish_gate_raw_wait_after_reap(
            channel_descriptor=descriptor,
            directory_descriptor=directory_descriptor,
            fault_descriptor=fault_descriptor,
            receipt_path=tmp_path / "launch_receipt.json",
            receipt=receipt,
            attempt_id=record["attempt_id"],
            owner_seal=record["supervisor_owner_seal"],
            child_process=record["child_process"],
            info=SimpleNamespace(
                si_pid=record["waitid_si_pid"],
                si_code=record["waitid_si_code"],
                si_status=record["waitid_si_status"],
            ),
            waited_pid=record["waited_pid"],
            raw_status=record["wait_status_raw"],
            started_at=record["started_at"],
            reaped_at=record["reaped_at"],
        ) == record
        assert fault_path.read_bytes() == b""
    finally:
        os.close(fault_descriptor)
        os.close(descriptor)
        os.close(directory_descriptor)


@pytest.mark.parametrize("existing_state", ("different", "partial"))
def test_nonexact_existing_raw_never_creates_valid_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_state: str,
) -> None:
    launcher = _launcher_module()
    raw_path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "nonexact_raw.channel"
    )
    fault_path = tmp_path / "nonexact_fault.channel"
    fault_binding = launcher._create_fault_channel(fault_path)
    fault_descriptor = launcher._open_presealed_fault_channel(
        tmp_path, fault_binding, name=fault_path.name
    )
    intended = _raw_wait(binding, signal_number=signal.SIGTERM)
    if existing_state == "different":
        existing = _raw_wait(binding)
        launcher._write_lifecycle_raw_wait_v3(
            descriptor,
            directory_descriptor,
            binding,
            existing,
            role="gate",
        )
        expected_raw = raw_path.read_bytes()
    else:
        expected_raw = b"partial-raw-wait"
        os.pwrite(descriptor, expected_raw, 0)
        os.fsync(descriptor)
    receipt = {
        "policy_sha256": intended["policy_sha256"],
        "gate_lifecycle_wait_channel": binding,
        "gate_lifecycle_wait_publisher": intended["publisher"],
        "gate_lifecycle_wait_publish_fault_channel": fault_binding,
    }
    monkeypatch.setattr(
        launcher,
        "_sealed_lifecycle_artifact",
        lambda *args, **kwargs: intended["source_artifact"],
    )
    monkeypatch.setattr(
        launcher,
        "build_lifecycle_raw_wait_v3",
        lambda **kwargs: intended,
    )
    try:
        with pytest.raises(
            launcher.LifecycleRawWaitPublishError
        ) as raised:
            launcher._publish_gate_raw_wait_after_reap(
                channel_descriptor=descriptor,
                directory_descriptor=directory_descriptor,
                fault_descriptor=fault_descriptor,
                receipt_path=tmp_path / "launch_receipt.json",
                receipt=receipt,
                attempt_id=intended["attempt_id"],
                owner_seal=intended["supervisor_owner_seal"],
                child_process=intended["child_process"],
                info=SimpleNamespace(
                    si_pid=intended["waitid_si_pid"],
                    si_code=intended["waitid_si_code"],
                    si_status=intended["waitid_si_status"],
                ),
                waited_pid=intended["waited_pid"],
                raw_status=intended["wait_status_raw"],
                started_at=intended["started_at"],
                reaped_at=intended["reaped_at"],
            )
        assert raised.value.commit_state == "collision"
        assert raised.value.stage == "prewrite_nonempty"
        assert raw_path.read_bytes() == expected_raw
        assert fault_path.read_bytes() == b""
    finally:
        os.close(fault_descriptor)
        os.close(descriptor)
        os.close(directory_descriptor)


def test_durability_unknown_raw_never_creates_valid_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    raw_path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "unknown_raw.channel"
    )
    fault_path = tmp_path / "unknown_fault.channel"
    fault_binding = launcher._create_fault_channel(fault_path)
    fault_descriptor = launcher._open_presealed_fault_channel(
        tmp_path, fault_binding, name=fault_path.name
    )
    record = _raw_wait(binding)
    receipt = {
        "policy_sha256": record["policy_sha256"],
        "gate_lifecycle_wait_channel": binding,
        "gate_lifecycle_wait_publisher": record["publisher"],
        "gate_lifecycle_wait_publish_fault_channel": fault_binding,
    }
    monkeypatch.setattr(
        launcher,
        "_sealed_lifecycle_artifact",
        lambda *args, **kwargs: record["source_artifact"],
    )
    monkeypatch.setattr(
        launcher,
        "build_lifecycle_raw_wait_v3",
        lambda **kwargs: record,
    )
    real_fsync = launcher.os.fsync

    def fail_raw_fsync(fd: int) -> None:
        if fd == descriptor:
            raise OSError(5, "injected raw durability failure")
        real_fsync(fd)

    monkeypatch.setattr(launcher.os, "fsync", fail_raw_fsync)
    try:
        with pytest.raises(
            launcher.LifecycleRawWaitPublishError
        ) as raised:
            launcher._publish_gate_raw_wait_after_reap(
                channel_descriptor=descriptor,
                directory_descriptor=directory_descriptor,
                fault_descriptor=fault_descriptor,
                receipt_path=tmp_path / "launch_receipt.json",
                receipt=receipt,
                attempt_id=record["attempt_id"],
                owner_seal=record["supervisor_owner_seal"],
                child_process=record["child_process"],
                info=SimpleNamespace(
                    si_pid=record["waitid_si_pid"],
                    si_code=record["waitid_si_code"],
                    si_status=record["waitid_si_status"],
                ),
                waited_pid=record["waited_pid"],
                raw_status=record["wait_status_raw"],
                started_at=record["started_at"],
                reaped_at=record["reaped_at"],
            )
        assert (
            raised.value.commit_state
            == "durability_unknown_quarantined"
        )
        assert raw_path.stat().st_size > 0
        assert fault_path.read_bytes() == b""
    finally:
        os.close(fault_descriptor)
        os.close(descriptor)
        os.close(directory_descriptor)


def test_committed_raw_verify_failure_keeps_fault_empty_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    raw_path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "committed_raw.channel"
    )
    fault_path = tmp_path / "committed_fault.channel"
    fault_binding = launcher._create_fault_channel(fault_path)
    fault_descriptor = launcher._open_presealed_fault_channel(
        tmp_path, fault_binding, name=fault_path.name
    )
    record = _raw_wait(binding)
    receipt = {
        "policy_sha256": record["policy_sha256"],
        "gate_lifecycle_wait_channel": binding,
        "gate_lifecycle_wait_publisher": record["publisher"],
        "gate_lifecycle_wait_publish_fault_channel": fault_binding,
    }
    monkeypatch.setattr(
        launcher,
        "_sealed_lifecycle_artifact",
        lambda *args, **kwargs: record["source_artifact"],
    )
    monkeypatch.setattr(
        launcher,
        "build_lifecycle_raw_wait_v3",
        lambda **kwargs: record,
    )
    real_require = launcher._require_named_lifecycle_wait_channel
    calls = 0

    def fail_after_commit(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError(5, "injected committed verify failure")
        return real_require(*args, **kwargs)

    monkeypatch.setattr(
        launcher,
        "_require_named_lifecycle_wait_channel",
        fail_after_commit,
    )

    def publish() -> dict[str, Any]:
        return launcher._publish_gate_raw_wait_after_reap(
            channel_descriptor=descriptor,
            directory_descriptor=directory_descriptor,
            fault_descriptor=fault_descriptor,
            receipt_path=tmp_path / "launch_receipt.json",
            receipt=receipt,
            attempt_id=record["attempt_id"],
            owner_seal=record["supervisor_owner_seal"],
            child_process=record["child_process"],
            info=SimpleNamespace(
                si_pid=record["waitid_si_pid"],
                si_code=record["waitid_si_code"],
                si_status=record["waitid_si_status"],
            ),
            waited_pid=record["waited_pid"],
            raw_status=record["wait_status_raw"],
            started_at=record["started_at"],
            reaped_at=record["reaped_at"],
        )

    try:
        with pytest.raises(
            launcher.LifecycleRawWaitPublishError
        ) as raised:
            publish()
        assert (
            raised.value.commit_state
            == "committed_cleanup_error"
        )
        assert fault_path.read_bytes() == b""
        assert raw_path.read_bytes() == (
            launcher._build_lifecycle_wait_channel_frame(record)[2]
        )
        monkeypatch.setattr(
            launcher,
            "_require_named_lifecycle_wait_channel",
            real_require,
        )
        assert publish() == record
        assert fault_path.read_bytes() == b""
    finally:
        os.close(fault_descriptor)
        os.close(descriptor)
        os.close(directory_descriptor)


def test_fault_writer_failure_preserves_primary_and_attempts_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    _raw_path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "primary_raw.channel"
    )
    fault_path = tmp_path / "primary_fault.channel"
    fault_binding = launcher._create_fault_channel(fault_path)
    fault_descriptor = launcher._open_presealed_fault_channel(
        tmp_path, fault_binding, name=fault_path.name
    )
    record = _raw_wait(binding)
    receipt = {
        "policy_sha256": record["policy_sha256"],
        "gate_lifecycle_wait_channel": binding,
        "gate_lifecycle_wait_publisher": record["publisher"],
        "gate_lifecycle_wait_publish_fault_channel": fault_binding,
    }
    primary = launcher.LifecycleRawWaitPublishError(
        "precommit_failed_clean",
        "primary raw publication failure",
        stage="raw_wait_body",
        directory_seal={"device": 1, "inode": 2},
        payload={"size": 3, "sha256": "4" * 64},
        temporary=None,
        error_number=5,
        quarantined=False,
    )
    attempts = {"raw": 0, "fault": 0}

    def fail_raw(*args: Any, **kwargs: Any) -> dict[str, Any]:
        attempts["raw"] += 1
        raise primary

    def fail_fault(*args: Any, **kwargs: Any) -> dict[str, Any]:
        attempts["fault"] += 1
        raise OSError(28, "injected fault channel failure")

    monkeypatch.setattr(
        launcher, "_write_lifecycle_raw_wait_v3", fail_raw
    )
    monkeypatch.setattr(
        launcher,
        "_write_lifecycle_raw_wait_publish_failure_v1",
        fail_fault,
    )
    monkeypatch.setattr(
        launcher,
        "_sealed_lifecycle_artifact",
        lambda *args, **kwargs: record["source_artifact"],
    )
    monkeypatch.setattr(
        launcher,
        "build_lifecycle_raw_wait_v3",
        lambda **kwargs: record,
    )
    try:
        with pytest.raises(
            launcher.LifecycleRawWaitPublishError
        ) as raised:
            launcher._publish_gate_raw_wait_after_reap(
                channel_descriptor=descriptor,
                directory_descriptor=directory_descriptor,
                fault_descriptor=fault_descriptor,
                receipt_path=tmp_path / "launch_receipt.json",
                receipt=receipt,
                attempt_id=record["attempt_id"],
                owner_seal=record["supervisor_owner_seal"],
                child_process=record["child_process"],
                info=SimpleNamespace(
                    si_pid=record["waitid_si_pid"],
                    si_code=record["waitid_si_code"],
                    si_status=record["waitid_si_status"],
                ),
                waited_pid=record["waited_pid"],
                raw_status=record["wait_status_raw"],
                started_at=record["started_at"],
                reaped_at=record["reaped_at"],
            )
        assert raised.value is primary
        assert raised.value.commit_state == "precommit_failed_clean"
        assert raised.value.stage == "raw_wait_body"
        assert raised.value.payload == {
            "size": 3,
            "sha256": "4" * 64,
        }
        assert raised.value.secondary_failures == [
            {
                "stage": "raw_wait_publish_fault",
                "type": "OSError",
                "message": (
                    "[Errno 28] injected fault channel failure"
                ),
            }
        ]
        assert attempts == {"raw": 1, "fault": 1}
        assert fault_path.read_bytes() == b""
    finally:
        os.close(fault_descriptor)
        os.close(descriptor)
        os.close(directory_descriptor)


def test_committed_fault_prevents_later_raw_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    raw_path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "fault_first_raw.channel"
    )
    fault_path = tmp_path / "fault_first_fault.channel"
    fault_binding = launcher._create_fault_channel(fault_path)
    fault_descriptor = launcher._open_presealed_fault_channel(
        tmp_path, fault_binding, name=fault_path.name
    )
    record = _raw_wait(binding)
    receipt = {
        "policy_sha256": record["policy_sha256"],
        "gate_lifecycle_wait_channel": binding,
        "gate_lifecycle_wait_publisher": record["publisher"],
        "gate_lifecycle_wait_publish_fault_channel": fault_binding,
    }
    primary = launcher.LifecycleRawWaitPublishError(
        "precommit_failed_clean",
        "injected clean raw failure",
        stage="raw_wait_body",
        directory_seal={"device": 1, "inode": 2},
        payload={"size": 3, "sha256": "4" * 64},
        temporary=None,
        error_number=5,
        quarantined=False,
    )
    raw_attempts = 0

    def fail_raw_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal raw_attempts
        raw_attempts += 1
        raise primary

    monkeypatch.setattr(
        launcher, "_write_lifecycle_raw_wait_v3", fail_raw_once
    )
    monkeypatch.setattr(
        launcher,
        "_sealed_lifecycle_artifact",
        lambda *args, **kwargs: record["source_artifact"],
    )
    monkeypatch.setattr(
        launcher,
        "build_lifecycle_raw_wait_v3",
        lambda **kwargs: record,
    )

    def publish() -> dict[str, Any]:
        return launcher._publish_gate_raw_wait_after_reap(
            channel_descriptor=descriptor,
            directory_descriptor=directory_descriptor,
            fault_descriptor=fault_descriptor,
            receipt_path=tmp_path / "launch_receipt.json",
            receipt=receipt,
            attempt_id=record["attempt_id"],
            owner_seal=record["supervisor_owner_seal"],
            child_process=record["child_process"],
            info=SimpleNamespace(
                si_pid=record["waitid_si_pid"],
                si_code=record["waitid_si_code"],
                si_status=record["waitid_si_status"],
            ),
            waited_pid=record["waited_pid"],
            raw_status=record["wait_status_raw"],
            started_at=record["started_at"],
            reaped_at=record["reaped_at"],
        )

    try:
        with pytest.raises(
            launcher.LifecycleRawWaitPublishError
        ) as first:
            publish()
        assert first.value is primary
        committed_fault = fault_path.read_bytes()
        assert committed_fault
        assert raw_path.read_bytes() == b""
        with pytest.raises(
            launcher.LifecycleRawWaitPublishError
        ) as second:
            publish()
        assert second.value.commit_state == "collision"
        assert second.value.stage == "preexisting_publish_fault"
        assert raw_attempts == 1
        assert raw_path.read_bytes() == b""
        assert fault_path.read_bytes() == committed_fault
    finally:
        os.close(fault_descriptor)
        os.close(descriptor)
        os.close(directory_descriptor)


def test_unconnected_v5_gate_reap_to_raw_order_is_structural() -> None:
    launcher = _launcher_module()
    tree = ast.parse(
        inspect.getsource(
            launcher
            ._gate_wait_supervisor_v5_reap_and_publish_unconnected
        )
    )
    statement_lists = [
        value
        for node in ast.walk(tree.body[0])
        for _field, value in ast.iter_fields(node)
        if isinstance(value, list)
        and all(isinstance(item, ast.stmt) for item in value)
    ]
    statements, wait_index = next(
        (items, index)
        for items in statement_lists
        for index, statement in enumerate(items)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_waitid_then_waitpid"
    )
    next_statement = statements[wait_index + 1]
    assert isinstance(next_statement, ast.Assign)
    assert isinstance(next_statement.value, ast.Call)
    assert isinstance(next_statement.value.func, ast.Name)
    assert (
        next_statement.value.func.id
        == "_publish_gate_raw_wait_after_reap"
    )
    later_calls = [
        node.func.id
        for statement in statements[wait_index + 2 :]
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    ]
    assert "post_raw_adjudicator" in later_calls


def test_unconnected_v5_gate_reap_raw_adjudication_spy_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    events: list[str] = []
    info = SimpleNamespace(
        si_pid=11,
        si_code=os.CLD_EXITED,
        si_status=117,
    )
    monkeypatch.setattr(
        launcher,
        "validate_launch_receipt_v5",
        lambda *args, **kwargs: events.append("validate"),
    )
    monkeypatch.setattr(
        launcher,
        "_expected_consumer_worker_arguments_from_receipt",
        lambda receipt: ["consumer"],
    )
    monkeypatch.setattr(
        launcher,
        "_waitid_then_waitpid",
        lambda child: (
            events.append("reap") or (info, 11, 117 << 8)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_publish_gate_raw_wait_after_reap",
        lambda **kwargs: events.append("raw") or {"raw": True},
    )

    def adjudicate(
        _info: Any, _waited_pid: int, _raw_status: int
    ) -> str:
        events.append("adjudicate")
        return "done"

    result = (
        launcher
        ._gate_wait_supervisor_v5_reap_and_publish_unconnected(
            child=object(),
            channel_descriptor=1,
            directory_descriptor=2,
            fault_descriptor=3,
            receipt_path=Path("/contract/launch_receipt.json"),
            receipt={"attempt_id": "f" * 64},
            attempt_id="f" * 64,
            owner_seal={},
            child_process={},
            gate_worker_arguments=["gate"],
            started_at="2026-07-29T00:00:00+00:00",
            post_raw_adjudicator=adjudicate,
        )
    )
    assert events == ["validate", "reap", "raw", "adjudicate"]
    assert result[-1] == "done"


def test_reaped_signal_publish_failure_precedes_adjudication_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _launcher_module()
    raw_path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "signal_raw.channel"
    )
    fault_path = tmp_path / "signal_fault.channel"
    fault_binding = launcher._create_fault_channel(fault_path)
    fault_descriptor = launcher._open_presealed_fault_channel(
        tmp_path, fault_binding, name=fault_path.name
    )
    record = _raw_wait(binding, signal_number=signal.SIGKILL)
    receipt = {
        "policy_sha256": record["policy_sha256"],
        "gate_lifecycle_wait_channel": binding,
        "gate_lifecycle_wait_publisher": record["publisher"],
        "gate_lifecycle_wait_publish_fault_channel": fault_binding,
    }
    events: list[str] = []

    def fail_raw(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("raw")
        raise launcher.LifecycleRawWaitPublishError(
            "precommit_failed_clean",
            "injected",
            stage="raw",
            directory_seal={},
            payload={"size": 1, "sha256": "0" * 64},
            temporary=None,
            error_number=None,
            quarantined=False,
        )

    real_fault = launcher._write_lifecycle_raw_wait_publish_failure_v1

    def write_fault(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("fault")
        return real_fault(*args, **kwargs)

    monkeypatch.setattr(launcher, "_write_lifecycle_raw_wait_v3", fail_raw)
    monkeypatch.setattr(
        launcher,
        "_write_lifecycle_raw_wait_publish_failure_v1",
        write_fault,
    )
    monkeypatch.setattr(
        launcher,
        "_sealed_lifecycle_artifact",
        lambda *args, **kwargs: record["source_artifact"],
    )
    monkeypatch.setattr(
        launcher,
        "build_lifecycle_raw_wait_v3",
        lambda **kwargs: record,
    )
    try:
        with pytest.raises(
            launcher.LifecycleRawWaitPublishError
        ):
            launcher._publish_gate_raw_wait_after_reap(
                channel_descriptor=descriptor,
                directory_descriptor=directory_descriptor,
                fault_descriptor=fault_descriptor,
                receipt_path=tmp_path / "launch_receipt.json",
                receipt=receipt,
                attempt_id=record["attempt_id"],
                owner_seal=record["supervisor_owner_seal"],
                child_process=record["child_process"],
                info=SimpleNamespace(
                    si_pid=11,
                    si_code=os.CLD_KILLED,
                    si_status=signal.SIGKILL,
                ),
                waited_pid=11,
                raw_status=int(signal.SIGKILL),
                started_at=record["started_at"],
                reaped_at=record["reaped_at"],
            )
        assert events == ["raw", "fault"]
        assert raw_path.read_bytes() == b""
        assert fault_path.stat().st_size > 0
    finally:
        os.close(fault_descriptor)
        os.close(descriptor)
        os.close(directory_descriptor)


def test_v5_reap_to_raw_crash_window_is_explicitly_unrecoverable() -> None:
    profile = contract.build_postclaim_finalization_profile_v1()
    assert profile["reap_to_raw_crash_policy"] == (
        "fail_closed_unrecoverable"
    )
    launcher = _launcher_module()
    source = inspect.getsource(
        launcher._gate_wait_supervisor_v5_reap_and_publish_unconnected
    )
    assert source.index("_publish_gate_raw_wait_after_reap") < (
        source.index("post_raw_adjudicator(")
    )


def test_v4_gate_supervisor_has_no_v5_callsite_or_dispatch() -> None:
    launcher = _launcher_module()
    source = inspect.getsource(launcher._gate_wait_supervisor)
    assert "validate_launch_receipt_schema(" in source
    assert "_write_lifecycle_wait_status(" in source
    assert "receipt_is_v5" not in source
    assert "_publish_gate_raw_wait_after_reap" not in source
    main_source = inspect.getsource(launcher.main)
    assert (
        "_gate_wait_supervisor_v5_reap_and_publish_unconnected"
        not in main_source
    )


def test_legacy_gate_and_verified_launcher_sha_are_frozen() -> None:
    launcher = _launcher_module()
    source = inspect.getsource(launcher._gate_wait_supervisor)
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == (
        "31750376be7239641a1e9af1557ca779292420b5003ed2fc38891174fd8c2203"
    )
    launcher_path = Path(launcher.__file__).resolve()
    config_path = (
        launcher_path.parents[1]
        / "configs/closeout/canonical_screening_512_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = config["implementations"]["preflight_launcher"][
        "sha256"
    ]
    assert hashlib.sha256(launcher_path.read_bytes()).hexdigest() == (
        expected
    )


def test_real_wait_then_raw_publication_uses_observed_sigkill(
    tmp_path: Path,
) -> None:
    launcher = _launcher_module()
    _path, binding, descriptor, directory_descriptor = _writer(
        launcher, tmp_path, "dynamic_signal.channel"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    os.killpg(child.pid, signal.SIGKILL)
    info, waited_pid, raw_status = launcher._waitid_then_waitpid(child)
    owner = _owner()
    owner["pane_pid"] = os.getpid()
    owner["pane_process"] = _process(os.getpid(), os.getppid())
    child_process = _process(child.pid, os.getpid())
    record = contract.build_lifecycle_raw_wait_v3(
        role="gate",
        policy_sha256="e" * 64,
        attempt_id="f" * 64,
        source_artifact=_source(),
        wait_channel=binding,
        publisher=_publisher(),
        supervisor_owner_seal=owner,
        child_process=child_process,
        waitid_si_pid=int(info.si_pid),
        waitid_si_code=int(info.si_code),
        waitid_si_status=int(info.si_status),
        waited_pid=waited_pid,
        wait_status_raw=raw_status,
        started_at="2026-07-29T00:00:00+00:00",
        reaped_at="2026-07-29T00:00:01+00:00",
    )
    try:
        launcher._write_lifecycle_raw_wait_v3(
            descriptor,
            directory_descriptor,
            binding,
            record,
            role="gate",
        )
    finally:
        os.close(descriptor)
        os.close(directory_descriptor)
    assert record["signal_number"] == signal.SIGKILL
    assert record["returncode"] == -signal.SIGKILL
