"""Pure schema contracts for the canonical CPU-preflight launch chain."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


class PreflightLaunchContractError(RuntimeError):
    """Raised when a durable launch artifact violates the shared contract."""


GATE_READY_CONTRACT_TYPE = (
    "safa_canonical_preflight_pane_gate_ready_v1"
)
TMUX_STARTED_CONTRACT_TYPE = (
    "safa_canonical_preflight_launch_tmux_started_v1"
)
WRAPPER_STARTED_CONTRACT_TYPE = (
    "safa_canonical_preflight_wrapper_started_v1"
)
CLAIM_V3_CONTRACT_TYPE = "safa_canonical_preflight_wrapper_claim_v3"
LAUNCH_ACCEPTED_CONTRACT_TYPE = (
    "safa_canonical_preflight_launch_accepted_v1"
)
LAUNCH_TERMINAL_CONTRACT_TYPE = (
    "safa_canonical_preflight_launch_terminal_v1"
)
OWNERSHIP_RELEASE_CONTRACT_TYPE = (
    "safa_canonical_preflight_launch_ownership_release_v1"
)


_ARTIFACT_BINDING_KEYS = {"path", "sha256", "canonical_sha256"}
_FILE_IDENTITY_KEYS = {"path", "device", "inode", "mode", "size"}
_PROCESS_IDENTITY_KEYS = {
    "pid",
    "ppid",
    "pgid",
    "sid",
    "start_ticks",
}
_EXECUTABLE_IDENTITY_KEYS = _FILE_IDENTITY_KEYS
_TMUX_SERVER_IDENTITY_KEYS = {
    "server_pid",
    "server_process",
    "socket_path",
    "socket_device",
    "socket_inode",
}
_VERIFIED_IMPLEMENTATION_KEYS = {
    "path",
    "sha256",
    "file_identity",
}
_VERIFIED_IMPLEMENTATIONS_KEYS = {
    "verified_loader",
    "preflight_launch_contract",
}
_PANE_OWNER_SEAL_KEYS = {
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
_GATE_READY_KEYS = {
    "schema_version",
    "contract_type",
    "launch_receipt",
    "launch_receipt_identity",
    "verified_implementations",
    "process",
    "wrapper_arguments",
    "ready_at",
    "pane_gate_ready_sha256",
}
_TMUX_STARTED_KEYS = {
    "schema_version",
    "contract_type",
    "launch_receipt",
    "launch_receipt_identity",
    "verified_implementations",
    "pane_gate_ready",
    "tmux_client",
    "owner_seal",
    "remain_on_exit",
    "started_at",
    "launch_tmux_started_sha256",
}
_TMUX_CLIENT_KEYS = {"returncode", "stdout", "stderr"}
_WRAPPER_STARTED_KEYS = {
    "schema_version",
    "contract_type",
    "launch_receipt",
    "launch_receipt_identity",
    "verified_implementations",
    "pane_gate_ready",
    "pane_gate_process",
    "wrapper_arguments",
    "wrapper_process",
    "wrapper_executable",
    "started_at",
    "wrapper_started_sha256",
}
_CLAIM_V3_KEYS = {
    "schema_version",
    "contract_type",
    "attempt_id",
    "preflight_launch_receipt",
    "preflight_launch_receipt_identity",
    "verified_implementations",
    "pane_gate_ready",
    "preflight_launch_tmux_started",
    "preflight_wrapper_started",
    "pane_gate_process",
    "wrapper_arguments",
    "wrapper_executable",
    "pane_log",
    "git",
    "policy_sha256",
    "config",
    "checkpoint_plan",
    "preflight_request_manifest",
    "controller_session",
    "controller_tmux",
    "controller_tmux_server",
    "observer_session",
    "command",
    "observer_command",
    "wrapper_pid",
    "wrapper_process",
    "wrapper_launch_process",
    "started_at",
    "external_timeout_seconds",
    "wrapper_claim_sha256",
}
_ACCEPTED_KEYS = {
    "schema_version",
    "contract_type",
    "attempt_id",
    "launch_receipt",
    "launch_receipt_identity",
    "verified_implementations",
    "wrapper_claim",
    "tmux_started",
    "pane",
    "pane_log_path",
    "startup_window_closed",
    "started_at",
    "accepted_at",
    "launch_accepted_sha256",
}
_OWNERSHIP_TERMINAL_KEYS = {
    "schema_version",
    "contract_type",
    "launch_receipt",
    "launch_receipt_identity",
    "verified_implementations",
    "launch_accepted",
    "wrapper_claim",
    "tmux_started",
    "status",
    "failure",
    "tmux_client",
    "pane",
    "pane_log",
    "session_residual",
    "started_at",
    "completed_at",
    "launch_terminal_sha256",
}
_OWNERSHIP_RELEASE_KEYS = {
    "schema_version",
    "contract_type",
    "launch_receipt",
    "launch_receipt_identity",
    "verified_implementations",
    "launch_accepted",
    "launch_terminal",
    "wrapper_claim",
    "startup_window_closed",
    "released_at",
    "launch_ownership_release_sha256",
}


def canonical_digest(
    value: Mapping[str, Any], excluded_field: str
) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key != excluded_field
    }
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightLaunchContractError(f"{label} is not a mapping")
    return dict(value)


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise PreflightLaunchContractError(
            f"{label} exact keys differ"
        )


def _hex64(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PreflightLaunchContractError(f"{label} is not hex64")
    return value


def validate_artifact_binding(
    raw: Any, label: str
) -> dict[str, str]:
    value = _mapping(raw, label)
    _exact_keys(value, _ARTIFACT_BINDING_KEYS, label)
    if (
        not isinstance(value["path"], str)
        or not value["path"]
        or not value["path"].startswith("/")
    ):
        raise PreflightLaunchContractError(
            f"{label} path is not absolute"
        )
    _hex64(value["sha256"], f"{label} sha256")
    _hex64(
        value["canonical_sha256"],
        f"{label} canonical sha256",
    )
    return value  # type: ignore[return-value]


def build_artifact_binding(
    *, path: str, sha256: str, canonical_sha256: str
) -> dict[str, str]:
    return validate_artifact_binding(
        {
            "path": path,
            "sha256": sha256,
            "canonical_sha256": canonical_sha256,
        },
        "artifact binding",
    )


def validate_file_identity(
    raw: Any, label: str
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _FILE_IDENTITY_KEYS, label)
    if (
        not isinstance(value["path"], str)
        or not value["path"].startswith("/")
        or any(
            type(value[field]) is not int or int(value[field]) < 0
            for field in ("device", "inode", "mode", "size")
        )
        or int(value["inode"]) <= 0
    ):
        raise PreflightLaunchContractError(f"{label} is invalid")
    return value


def build_file_identity(
    *, path: str, device: int, inode: int, mode: int, size: int
) -> dict[str, Any]:
    return validate_file_identity(
        {
            "path": path,
            "device": device,
            "inode": inode,
            "mode": mode,
            "size": size,
        },
        "file identity",
    )


def validate_verified_implementations(
    raw: Any, label: str = "verified implementations"
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _VERIFIED_IMPLEMENTATIONS_KEYS, label)
    normalized: dict[str, Any] = {}
    for name in sorted(_VERIFIED_IMPLEMENTATIONS_KEYS):
        binding = _mapping(value[name], f"{label} {name}")
        _exact_keys(
            binding,
            _VERIFIED_IMPLEMENTATION_KEYS,
            f"{label} {name}",
        )
        if (
            not isinstance(binding["path"], str)
            or not binding["path"].startswith("/")
        ):
            raise PreflightLaunchContractError(
                f"{label} {name} path differs"
            )
        _hex64(binding["sha256"], f"{label} {name} SHA-256")
        identity = validate_file_identity(
            binding["file_identity"],
            f"{label} {name} file identity",
        )
        if identity["path"] != binding["path"]:
            raise PreflightLaunchContractError(
                f"{label} {name} identity path differs"
            )
        normalized[name] = {
            "path": binding["path"],
            "sha256": binding["sha256"],
            "file_identity": dict(identity),
        }
    return normalized


def build_verified_implementations(
    *,
    verified_loader: Mapping[str, Any],
    preflight_launch_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_verified_implementations(
        {
            "verified_loader": dict(verified_loader),
            "preflight_launch_contract": dict(
                preflight_launch_contract
            ),
        }
    )


def validate_process_identity(
    raw: Any, label: str
) -> dict[str, int]:
    """Validate the full parent/session-bearing process identity."""

    value = _mapping(raw, label)
    _exact_keys(value, _PROCESS_IDENTITY_KEYS, label)
    if any(
        type(value[field]) is not int or int(value[field]) <= 0
        for field in _PROCESS_IDENTITY_KEYS
    ):
        raise PreflightLaunchContractError(f"{label} is invalid")
    return value  # type: ignore[return-value]


def build_process_identity(
    *,
    pid: int,
    ppid: int,
    pgid: int,
    sid: int,
    start_ticks: int,
) -> dict[str, int]:
    """Build a full authority-bearing identity from one proc snapshot."""

    return validate_process_identity(
        {
            "pid": pid,
            "ppid": ppid,
            "pgid": pgid,
            "sid": sid,
            "start_ticks": start_ticks,
        },
        "process identity",
    )


def validate_executable_identity(
    raw: Any, label: str
) -> dict[str, Any]:
    value = validate_file_identity(raw, label)
    _exact_keys(value, _EXECUTABLE_IDENTITY_KEYS, label)
    return value


def validate_tmux_server_identity(
    raw: Any, label: str
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _TMUX_SERVER_IDENTITY_KEYS, label)
    server_process = validate_process_identity(
        value["server_process"],
        f"{label} server process",
    )
    if (
        type(value["server_pid"]) is not int
        or int(value["server_pid"]) <= 1
        or server_process["pid"] != value["server_pid"]
        or not isinstance(value["socket_path"], str)
        or not value["socket_path"].startswith("/")
        or any(
            type(value[field]) is not int or int(value[field]) <= 0
            for field in ("socket_device", "socket_inode")
        )
    ):
        raise PreflightLaunchContractError(f"{label} is invalid")
    return value


def build_tmux_server_identity(
    *,
    server_pid: int,
    server_process: Mapping[str, Any],
    socket_path: str,
    socket_device: int,
    socket_inode: int,
) -> dict[str, Any]:
    return validate_tmux_server_identity(
        {
            "server_pid": server_pid,
            "server_process": dict(server_process),
            "socket_path": socket_path,
            "socket_device": socket_device,
            "socket_inode": socket_inode,
        },
        "tmux server identity",
    )


def validate_pane_owner_seal(
    raw: Any,
    *,
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _PANE_OWNER_SEAL_KEYS, label)
    server = validate_tmux_server_identity(
        tmux_server, f"{label} tmux server"
    )
    pane_process = validate_process_identity(
        value["pane_process"],
        f"{label} pane process",
    )
    if (
        value["server_pid"] != server["server_pid"]
        or value["server_start_ticks"]
        != server["server_process"]["start_ticks"]
        or value["socket_path"] != server["socket_path"]
        or value["socket_device"] != server["socket_device"]
        or value["socket_inode"] != server["socket_inode"]
        or value["session"] != tmux_identity.get("session")
        or value["pane"] != tmux_identity.get("pane")
        or value["pane_pid"] != tmux_identity.get("pane_pid")
        or pane_process["pid"] != value["pane_pid"]
        or any(
            type(value[field]) is not int or int(value[field]) <= 0
            for field in (
                "server_pid",
                "server_start_ticks",
                "socket_device",
                "socket_inode",
                "pane_pid",
            )
        )
    ):
        raise PreflightLaunchContractError(f"{label} relation differs")
    _hex64(value["owner_nonce"], f"{label} owner nonce")
    return value


def build_pane_owner_seal(
    *,
    server_pid: int,
    server_start_ticks: int,
    socket_path: str,
    socket_device: int,
    socket_inode: int,
    session: str,
    pane: str,
    pane_pid: int,
    pane_process: Mapping[str, Any],
    owner_nonce: str,
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_pane_owner_seal(
        {
            "server_pid": server_pid,
            "server_start_ticks": server_start_ticks,
            "socket_path": socket_path,
            "socket_device": socket_device,
            "socket_inode": socket_inode,
            "session": session,
            "pane": pane,
            "pane_pid": pane_pid,
            "pane_process": dict(pane_process),
            "owner_nonce": owner_nonce,
        },
        tmux_identity=tmux_identity,
        tmux_server=tmux_server,
        label="pane owner seal",
    )


def _validate_digested_contract(
    raw: Any,
    *,
    expected_keys: set[str],
    contract_type: str,
    digest_field: str,
    label: str,
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, expected_keys, label)
    if (
        value.get("schema_version") != 1
        or value.get("contract_type") != contract_type
        or value.get(digest_field)
        != canonical_digest(value, digest_field)
    ):
        raise PreflightLaunchContractError(f"{label} contract differs")
    return value


def validate_gate_ready(
    raw: Any,
    *,
    verified_implementations: Mapping[str, Any],
    label: str = "pane gate ready",
) -> dict[str, Any]:
    value = _validate_digested_contract(
        raw,
        expected_keys=_GATE_READY_KEYS,
        contract_type=GATE_READY_CONTRACT_TYPE,
        digest_field="pane_gate_ready_sha256",
        label=label,
    )
    validate_artifact_binding(value["launch_receipt"], "launch receipt")
    validate_file_identity(
        value["launch_receipt_identity"],
        "launch receipt identity",
    )
    normalized_implementations = validate_verified_implementations(
        verified_implementations,
        f"{label} expected verified implementations",
    )
    validate_process_identity(
        value["process"], "pane gate process"
    )
    if (
        value["verified_implementations"] != normalized_implementations
        or not isinstance(value["wrapper_arguments"], list)
        or not value["wrapper_arguments"]
        or any(
            not isinstance(item, str)
            for item in value["wrapper_arguments"]
        )
        or not isinstance(value["ready_at"], str)
        or not value["ready_at"]
    ):
        raise PreflightLaunchContractError(
            f"{label} wrapper arguments differ"
        )
    return value


def build_gate_ready(
    *,
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    process: Mapping[str, Any],
    wrapper_arguments: list[str],
    ready_at: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": GATE_READY_CONTRACT_TYPE,
        "launch_receipt": dict(launch_receipt),
        "launch_receipt_identity": dict(launch_receipt_identity),
        "verified_implementations": dict(verified_implementations),
        "process": dict(process),
        "wrapper_arguments": list(wrapper_arguments),
        "ready_at": ready_at,
    }
    value["pane_gate_ready_sha256"] = canonical_digest(
        value, "pane_gate_ready_sha256"
    )
    return validate_gate_ready(
        value,
        verified_implementations=verified_implementations,
    )


def validate_tmux_started(
    raw: Any,
    *,
    verified_implementations: Mapping[str, Any],
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
    label: str = "launch tmux started",
) -> dict[str, Any]:
    value = _validate_digested_contract(
        raw,
        expected_keys=_TMUX_STARTED_KEYS,
        contract_type=TMUX_STARTED_CONTRACT_TYPE,
        digest_field="launch_tmux_started_sha256",
        label=label,
    )
    validate_artifact_binding(value["launch_receipt"], "launch receipt")
    validate_file_identity(
        value["launch_receipt_identity"],
        "launch receipt identity",
    )
    normalized_implementations = validate_verified_implementations(
        verified_implementations,
        f"{label} expected verified implementations",
    )
    validate_artifact_binding(
        value["pane_gate_ready"], "pane gate ready"
    )
    validate_pane_owner_seal(
        value["owner_seal"],
        tmux_identity=tmux_identity,
        tmux_server=tmux_server,
        label=f"{label} owner seal",
    )
    tmux_client = _mapping(value["tmux_client"], f"{label} tmux client")
    _exact_keys(tmux_client, _TMUX_CLIENT_KEYS, f"{label} tmux client")
    if (
        value["verified_implementations"] != normalized_implementations
        or type(tmux_client["returncode"]) is not int
        or tmux_client["returncode"] != 0
        or not isinstance(tmux_client["stdout"], str)
        or not isinstance(tmux_client["stderr"], str)
        or value["remain_on_exit"] != "on"
        or not isinstance(value["started_at"], str)
        or not value["started_at"]
    ):
        raise PreflightLaunchContractError(f"{label} relation differs")
    return value


def build_tmux_started(
    *,
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    pane_gate_ready: Mapping[str, Any],
    tmux_client: Mapping[str, Any],
    owner_seal: Mapping[str, Any],
    started_at: str,
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": TMUX_STARTED_CONTRACT_TYPE,
        "launch_receipt": dict(launch_receipt),
        "launch_receipt_identity": dict(launch_receipt_identity),
        "verified_implementations": dict(verified_implementations),
        "pane_gate_ready": dict(pane_gate_ready),
        "tmux_client": dict(tmux_client),
        "owner_seal": dict(owner_seal),
        "remain_on_exit": "on",
        "started_at": started_at,
    }
    value["launch_tmux_started_sha256"] = canonical_digest(
        value, "launch_tmux_started_sha256"
    )
    return validate_tmux_started(
        value,
        verified_implementations=verified_implementations,
        tmux_identity=tmux_identity,
        tmux_server=tmux_server,
    )


def validate_wrapper_started(
    raw: Any,
    *,
    verified_implementations: Mapping[str, Any],
    gate_ready: Mapping[str, Any] | None = None,
    label: str = "wrapper started",
) -> dict[str, Any]:
    value = _validate_digested_contract(
        raw,
        expected_keys=_WRAPPER_STARTED_KEYS,
        contract_type=WRAPPER_STARTED_CONTRACT_TYPE,
        digest_field="wrapper_started_sha256",
        label=label,
    )
    validate_artifact_binding(value["launch_receipt"], "launch receipt")
    validate_file_identity(
        value["launch_receipt_identity"],
        "launch receipt identity",
    )
    normalized_implementations = validate_verified_implementations(
        verified_implementations,
        f"{label} expected verified implementations",
    )
    validate_artifact_binding(value["pane_gate_ready"], "pane gate ready")
    gate_process = validate_process_identity(
        value["pane_gate_process"],
        "pane gate process",
    )
    wrapper_process = validate_process_identity(
        value["wrapper_process"],
        "wrapper process",
    )
    validate_executable_identity(
        value["wrapper_executable"], "wrapper executable"
    )
    if (
        value["verified_implementations"] != normalized_implementations
        or wrapper_process["pid"] == gate_process["pid"]
        or wrapper_process["ppid"] != gate_process["pid"]
        or wrapper_process["pgid"] != wrapper_process["pid"]
        or wrapper_process["sid"] != wrapper_process["pid"]
        or not isinstance(value["wrapper_arguments"], list)
        or not value["wrapper_arguments"]
        or any(
            not isinstance(item, str)
            for item in value["wrapper_arguments"]
        )
        or not isinstance(value["started_at"], str)
        or not value["started_at"]
        or (
            gate_ready is not None
            and gate_process != gate_ready.get("process")
        )
    ):
        raise PreflightLaunchContractError(
            f"{label} process relation differs"
        )
    return value


def build_wrapper_started(
    *,
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    pane_gate_ready: Mapping[str, Any],
    pane_gate_process: Mapping[str, Any],
    wrapper_arguments: list[str],
    wrapper_process: Mapping[str, Any],
    wrapper_executable: Mapping[str, Any],
    started_at: str,
    gate_ready: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": WRAPPER_STARTED_CONTRACT_TYPE,
        "launch_receipt": dict(launch_receipt),
        "launch_receipt_identity": dict(launch_receipt_identity),
        "verified_implementations": dict(verified_implementations),
        "pane_gate_ready": dict(pane_gate_ready),
        "pane_gate_process": dict(pane_gate_process),
        "wrapper_arguments": list(wrapper_arguments),
        "wrapper_process": dict(wrapper_process),
        "wrapper_executable": dict(wrapper_executable),
        "started_at": started_at,
    }
    value["wrapper_started_sha256"] = canonical_digest(
        value, "wrapper_started_sha256"
    )
    return validate_wrapper_started(
        value,
        verified_implementations=verified_implementations,
        gate_ready=gate_ready,
    )


def validate_claim_v3(
    raw: Any,
    *,
    verified_implementations: Mapping[str, Any],
    gate_ready: Mapping[str, Any] | None = None,
    wrapper_started: Mapping[str, Any] | None = None,
    label: str = "wrapper claim v3",
) -> dict[str, Any]:
    value = _validate_digested_contract(
        raw,
        expected_keys=_CLAIM_V3_KEYS,
        contract_type=CLAIM_V3_CONTRACT_TYPE,
        digest_field="wrapper_claim_sha256",
        label=label,
    )
    _hex64(value["attempt_id"], f"{label} attempt ID")
    _hex64(value["policy_sha256"], f"{label} policy SHA")
    for field in (
        "preflight_launch_receipt",
        "pane_gate_ready",
        "preflight_launch_tmux_started",
        "preflight_wrapper_started",
        "config",
        "checkpoint_plan",
        "preflight_request_manifest",
    ):
        binding = _mapping(value[field], f"{label} {field}")
        if field == "config":
            if set(binding) != {"path", "sha256"}:
                raise PreflightLaunchContractError(
                    f"{label} config binding differs"
                )
        else:
            validate_artifact_binding(binding, f"{label} {field}")
    config_binding = _mapping(value["config"], f"{label} config")
    if (
        not isinstance(config_binding["path"], str)
        or not config_binding["path"].startswith("/")
    ):
        raise PreflightLaunchContractError(
            f"{label} config path differs"
        )
    _hex64(config_binding["sha256"], f"{label} config SHA")
    validate_file_identity(
        value["preflight_launch_receipt_identity"],
        f"{label} receipt identity",
    )
    normalized_implementations = validate_verified_implementations(
        verified_implementations,
        f"{label} expected verified implementations",
    )
    gate_process = validate_process_identity(
        value["pane_gate_process"],
        f"{label} gate process",
    )
    wrapper_process = validate_process_identity(
        value["wrapper_process"],
        f"{label} wrapper process",
    )
    wrapper_launch = validate_process_identity(
        value["wrapper_launch_process"],
        f"{label} wrapper launch process",
    )
    validate_executable_identity(
        value["wrapper_executable"], f"{label} wrapper executable"
    )
    validate_file_identity(value["pane_log"], f"{label} pane log")
    validate_tmux_server_identity(
        value["controller_tmux_server"],
        f"{label} controller tmux server",
    )
    for field in ("wrapper_arguments", "command", "observer_command"):
        sequence = value[field]
        if (
            not isinstance(sequence, list)
            or not sequence
            or any(not isinstance(item, str) for item in sequence)
        ):
            raise PreflightLaunchContractError(
                f"{label} {field} differs"
            )
    if (
        not isinstance(value["git"], Mapping)
        or not isinstance(value["controller_tmux"], Mapping)
        or not isinstance(value["controller_session"], str)
        or not value["controller_session"]
        or not isinstance(value["observer_session"], str)
        or not value["observer_session"]
        or not isinstance(value["started_at"], str)
        or not value["started_at"]
        or value["external_timeout_seconds"] is not None
    ):
        raise PreflightLaunchContractError(
            f"{label} field type differs"
        )
    if (
        value["verified_implementations"] != normalized_implementations
        or wrapper_process != wrapper_launch
        or value["wrapper_pid"] != wrapper_launch["pid"]
        or wrapper_launch["pid"] == gate_process["pid"]
        or wrapper_launch["ppid"] != gate_process["pid"]
        or wrapper_launch["pgid"] != wrapper_launch["pid"]
        or wrapper_launch["sid"] != wrapper_launch["pid"]
        or (
            gate_ready is not None
            and gate_process != gate_ready.get("process")
        )
        or (
            wrapper_started is not None
            and (
                gate_process
                != wrapper_started.get("pane_gate_process")
                or wrapper_launch
                != wrapper_started.get("wrapper_process")
            )
        )
    ):
        raise PreflightLaunchContractError(
            f"{label} process relation differs"
        )
    return value


def build_claim_v3(
    *,
    attempt_id: str,
    preflight_launch_receipt: Mapping[str, Any],
    preflight_launch_receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    pane_gate_ready: Mapping[str, Any],
    preflight_launch_tmux_started: Mapping[str, Any],
    preflight_wrapper_started: Mapping[str, Any],
    pane_gate_process: Mapping[str, Any],
    wrapper_arguments: list[str],
    wrapper_executable: Mapping[str, Any],
    pane_log: Mapping[str, Any],
    git: Mapping[str, Any],
    policy_sha256: str,
    config: Mapping[str, Any],
    checkpoint_plan: Mapping[str, Any],
    preflight_request_manifest: Mapping[str, Any],
    controller_session: str,
    controller_tmux: Mapping[str, Any],
    controller_tmux_server: Mapping[str, Any],
    observer_session: str,
    command: list[str],
    observer_command: list[str],
    wrapper_pid: int,
    wrapper_process: Mapping[str, Any],
    wrapper_launch_process: Mapping[str, Any],
    started_at: str,
    external_timeout_seconds: None,
    gate_ready: Mapping[str, Any] | None = None,
    wrapper_started: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": CLAIM_V3_CONTRACT_TYPE,
        "attempt_id": attempt_id,
        "preflight_launch_receipt": dict(preflight_launch_receipt),
        "preflight_launch_receipt_identity": dict(
            preflight_launch_receipt_identity
        ),
        "verified_implementations": dict(verified_implementations),
        "pane_gate_ready": dict(pane_gate_ready),
        "preflight_launch_tmux_started": dict(
            preflight_launch_tmux_started
        ),
        "preflight_wrapper_started": dict(preflight_wrapper_started),
        "pane_gate_process": dict(pane_gate_process),
        "wrapper_arguments": list(wrapper_arguments),
        "wrapper_executable": dict(wrapper_executable),
        "pane_log": dict(pane_log),
        "git": dict(git),
        "policy_sha256": policy_sha256,
        "config": dict(config),
        "checkpoint_plan": dict(checkpoint_plan),
        "preflight_request_manifest": dict(
            preflight_request_manifest
        ),
        "controller_session": controller_session,
        "controller_tmux": dict(controller_tmux),
        "controller_tmux_server": dict(controller_tmux_server),
        "observer_session": observer_session,
        "command": list(command),
        "observer_command": list(observer_command),
        "wrapper_pid": wrapper_pid,
        "wrapper_process": dict(wrapper_process),
        "wrapper_launch_process": dict(wrapper_launch_process),
        "started_at": started_at,
        "external_timeout_seconds": external_timeout_seconds,
    }
    value["wrapper_claim_sha256"] = canonical_digest(
        value, "wrapper_claim_sha256"
    )
    return validate_claim_v3(
        value,
        verified_implementations=verified_implementations,
        gate_ready=gate_ready,
        wrapper_started=wrapper_started,
    )


def validate_ownership_chain(
    raw_accepted: Any,
    raw_terminal: Any,
    raw_release: Any,
    *,
    receipt_binding: Mapping[str, Any],
    receipt_identity: Mapping[str, Any],
    wrapper_binding: Mapping[str, Any],
    accepted_binding: Mapping[str, Any],
    terminal_binding: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    label: str = "launch ownership chain",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    accepted = _validate_digested_contract(
        raw_accepted,
        expected_keys=_ACCEPTED_KEYS,
        contract_type=LAUNCH_ACCEPTED_CONTRACT_TYPE,
        digest_field="launch_accepted_sha256",
        label=f"{label} accepted",
    )
    terminal = _validate_digested_contract(
        raw_terminal,
        expected_keys=_OWNERSHIP_TERMINAL_KEYS,
        contract_type=LAUNCH_TERMINAL_CONTRACT_TYPE,
        digest_field="launch_terminal_sha256",
        label=f"{label} terminal",
    )
    release = _validate_digested_contract(
        raw_release,
        expected_keys=_OWNERSHIP_RELEASE_KEYS,
        contract_type=(
            OWNERSHIP_RELEASE_CONTRACT_TYPE
        ),
        digest_field="launch_ownership_release_sha256",
        label=f"{label} release",
    )
    normalized_receipt = validate_artifact_binding(
        receipt_binding, f"{label} receipt binding"
    )
    normalized_identity = validate_file_identity(
        receipt_identity, f"{label} receipt identity"
    )
    normalized_wrapper = validate_artifact_binding(
        wrapper_binding, f"{label} wrapper binding"
    )
    normalized_accepted = validate_artifact_binding(
        accepted_binding, f"{label} accepted binding"
    )
    normalized_terminal = validate_artifact_binding(
        terminal_binding, f"{label} terminal binding"
    )
    normalized_implementations = validate_verified_implementations(
        verified_implementations,
        f"{label} verified implementations",
    )
    _hex64(accepted["attempt_id"], f"{label} accepted attempt ID")
    validate_artifact_binding(
        accepted["tmux_started"], f"{label} accepted tmux started"
    )
    validate_artifact_binding(
        terminal["tmux_started"], f"{label} terminal tmux started"
    )
    validate_file_identity(
        terminal["pane_log"], f"{label} terminal pane log"
    )
    if (
        not isinstance(accepted["pane"], Mapping)
        or not isinstance(terminal["pane"], Mapping)
        or not isinstance(accepted["pane_log_path"], str)
        or not accepted["pane_log_path"].startswith("/")
        or accepted["tmux_started"] != terminal["tmux_started"]
        or accepted["pane"] != terminal["pane"]
        or accepted["pane_log_path"] != terminal["pane_log"]["path"]
        or terminal["tmux_client"] is not None
        or not isinstance(accepted["started_at"], str)
        or not accepted["started_at"]
        or not isinstance(accepted["accepted_at"], str)
        or not accepted["accepted_at"]
        or terminal["started_at"] != accepted["started_at"]
        or not isinstance(terminal["completed_at"], str)
        or not terminal["completed_at"]
        or not isinstance(release["released_at"], str)
        or not release["released_at"]
    ):
        raise PreflightLaunchContractError(
            f"{label} field type or temporal relation differs"
        )
    if (
        accepted["launch_receipt"] != normalized_receipt
        or accepted["launch_receipt_identity"] != normalized_identity
        or accepted["verified_implementations"]
        != normalized_implementations
        or accepted["wrapper_claim"] != normalized_wrapper
        or accepted["startup_window_closed"] is not False
        or terminal["launch_receipt"] != normalized_receipt
        or terminal["launch_receipt_identity"] != normalized_identity
        or terminal["verified_implementations"]
        != normalized_implementations
        or terminal["launch_accepted"] != normalized_accepted
        or terminal["wrapper_claim"] != normalized_wrapper
        or terminal["status"] != "ownership_transferred"
        or terminal["failure"] is not None
        or terminal["session_residual"] is not True
        or release["launch_receipt"] != normalized_receipt
        or release["launch_receipt_identity"] != normalized_identity
        or release["verified_implementations"]
        != normalized_implementations
        or release["launch_accepted"] != normalized_accepted
        or release["launch_terminal"] != normalized_terminal
        or release["wrapper_claim"] != normalized_wrapper
        or release["startup_window_closed"] is not True
    ):
        raise PreflightLaunchContractError(
            f"{label} reference order or relation differs"
        )
    return accepted, terminal, release


def build_launch_accepted(
    *,
    attempt_id: str,
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    wrapper_claim: Mapping[str, Any],
    tmux_started: Mapping[str, Any],
    pane: Mapping[str, Any],
    pane_log_path: str,
    started_at: str,
    accepted_at: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": LAUNCH_ACCEPTED_CONTRACT_TYPE,
        "attempt_id": attempt_id,
        "launch_receipt": dict(launch_receipt),
        "launch_receipt_identity": dict(launch_receipt_identity),
        "verified_implementations": dict(verified_implementations),
        "wrapper_claim": dict(wrapper_claim),
        "tmux_started": dict(tmux_started),
        "pane": dict(pane),
        "pane_log_path": pane_log_path,
        "startup_window_closed": False,
        "started_at": started_at,
        "accepted_at": accepted_at,
    }
    value["launch_accepted_sha256"] = canonical_digest(
        value, "launch_accepted_sha256"
    )
    return _validate_digested_contract(
        value,
        expected_keys=_ACCEPTED_KEYS,
        contract_type=LAUNCH_ACCEPTED_CONTRACT_TYPE,
        digest_field="launch_accepted_sha256",
        label="launch accepted",
    )


def build_ownership_terminal(
    *,
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    launch_accepted: Mapping[str, Any],
    wrapper_claim: Mapping[str, Any],
    tmux_started: Mapping[str, Any],
    pane: Mapping[str, Any],
    pane_log: Mapping[str, Any],
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": LAUNCH_TERMINAL_CONTRACT_TYPE,
        "launch_receipt": dict(launch_receipt),
        "launch_receipt_identity": dict(launch_receipt_identity),
        "verified_implementations": dict(verified_implementations),
        "launch_accepted": dict(launch_accepted),
        "wrapper_claim": dict(wrapper_claim),
        "tmux_started": dict(tmux_started),
        "status": "ownership_transferred",
        "failure": None,
        "tmux_client": None,
        "pane": dict(pane),
        "pane_log": dict(pane_log),
        "session_residual": True,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    value["launch_terminal_sha256"] = canonical_digest(
        value, "launch_terminal_sha256"
    )
    return _validate_digested_contract(
        value,
        expected_keys=_OWNERSHIP_TERMINAL_KEYS,
        contract_type=LAUNCH_TERMINAL_CONTRACT_TYPE,
        digest_field="launch_terminal_sha256",
        label="ownership terminal",
    )


def build_ownership_release(
    *,
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    launch_accepted: Mapping[str, Any],
    launch_terminal: Mapping[str, Any],
    wrapper_claim: Mapping[str, Any],
    released_at: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": OWNERSHIP_RELEASE_CONTRACT_TYPE,
        "launch_receipt": dict(launch_receipt),
        "launch_receipt_identity": dict(launch_receipt_identity),
        "verified_implementations": dict(verified_implementations),
        "launch_accepted": dict(launch_accepted),
        "launch_terminal": dict(launch_terminal),
        "wrapper_claim": dict(wrapper_claim),
        "startup_window_closed": True,
        "released_at": released_at,
    }
    value["launch_ownership_release_sha256"] = canonical_digest(
        value, "launch_ownership_release_sha256"
    )
    return _validate_digested_contract(
        value,
        expected_keys=_OWNERSHIP_RELEASE_KEYS,
        contract_type=(
            OWNERSHIP_RELEASE_CONTRACT_TYPE
        ),
        digest_field="launch_ownership_release_sha256",
        label="ownership release",
    )


def shared_schema_keysets() -> dict[str, frozenset[str]]:
    """Expose immutable exact schemas for static duplicate-literal tests."""

    return {
        "artifact_binding": frozenset(_ARTIFACT_BINDING_KEYS),
        "file_identity": frozenset(_FILE_IDENTITY_KEYS),
        "verified_implementation": frozenset(
            _VERIFIED_IMPLEMENTATION_KEYS
        ),
        "verified_implementations": frozenset(
            _VERIFIED_IMPLEMENTATIONS_KEYS
        ),
        "process_identity": frozenset(_PROCESS_IDENTITY_KEYS),
        "tmux_server": frozenset(_TMUX_SERVER_IDENTITY_KEYS),
        "pane_owner_seal": frozenset(_PANE_OWNER_SEAL_KEYS),
        "gate_ready": frozenset(_GATE_READY_KEYS),
        "tmux_started": frozenset(_TMUX_STARTED_KEYS),
        "wrapper_started": frozenset(_WRAPPER_STARTED_KEYS),
        "claim_v3": frozenset(_CLAIM_V3_KEYS),
        "accepted": frozenset(_ACCEPTED_KEYS),
        "ownership_terminal": frozenset(_OWNERSHIP_TERMINAL_KEYS),
        "ownership_release": frozenset(_OWNERSHIP_RELEASE_KEYS),
    }


def shared_contract_types() -> frozenset[str]:
    return frozenset(
        {
            GATE_READY_CONTRACT_TYPE,
            TMUX_STARTED_CONTRACT_TYPE,
            WRAPPER_STARTED_CONTRACT_TYPE,
            CLAIM_V3_CONTRACT_TYPE,
            LAUNCH_ACCEPTED_CONTRACT_TYPE,
            LAUNCH_TERMINAL_CONTRACT_TYPE,
            OWNERSHIP_RELEASE_CONTRACT_TYPE,
        }
    )


def shared_schema_tokens() -> frozenset[str]:
    """Expose schema field names for static duplicate-literal tests."""

    return frozenset(
        _FILE_IDENTITY_KEYS
        | _PANE_OWNER_SEAL_KEYS
        | _GATE_READY_KEYS
        | _TMUX_STARTED_KEYS
        | _WRAPPER_STARTED_KEYS
        | _CLAIM_V3_KEYS
        | _ACCEPTED_KEYS
        | _OWNERSHIP_TERMINAL_KEYS
        | _OWNERSHIP_RELEASE_KEYS
    )
