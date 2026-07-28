"""Pure schema contracts for the canonical CPU-preflight launch chain."""

from __future__ import annotations

import hashlib
import json
import stat
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


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
FAULT_RECORD_CONTRACT_TYPE = (
    "safa_canonical_preflight_wrapper_fault_v2"
)
LAUNCH_RECEIPT_CONTRACT_TYPE = (
    "safa_canonical_preflight_launch_receipt_v4"
)
LAUNCH_RECEIPT_V5_CONTRACT_TYPE = (
    "safa_canonical_preflight_launch_receipt_v5"
)
PANE_FAULT_CONSUMER_REGISTRATION_CONTRACT_TYPE = (
    "safa_pane_fault_consumer_receipt_registration_v1"
)
LIFECYCLE_WAIT_STATUS_CONTRACT_TYPE = (
    "safa_preflight_lifecycle_wait_status_v2"
)
LIFECYCLE_RAW_WAIT_V3_CONTRACT_TYPE = (
    "safa_preflight_lifecycle_raw_wait_v3"
)
LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_CONTRACT_TYPE = (
    "safa_preflight_lifecycle_raw_wait_publish_failure_v1"
)
POSTCLAIM_FINALIZATION_PROFILE_V1_CONTRACT_TYPE = (
    "safa_postclaim_finalization_profile_v1"
)
PANE_FAULT_CONSUMER_CONTROLLER_CLEANUP_V3_CONTRACT_TYPE = (
    "safa_pane_fault_consumer_controller_cleanup_v3"
)
PANE_FAULT_CONSUMER_TERMINAL_V3_CONTRACT_TYPE = (
    "safa_pane_fault_consumer_terminal_v3"
)
PANE_FAULT_CONSUMER_JOIN_V4_CONTRACT_TYPE = (
    "safa_pane_fault_consumer_join_v4"
)
PRECLAIM_FAILURE_INTENT_CONTRACT_TYPE = (
    "safa_canonical_preflight_preclaim_failure_intent_v1"
)
LAUNCH_TERMINAL_V2_CONTRACT_TYPE = (
    "safa_canonical_preflight_launch_terminal_v2"
)
POST_HANDOFF_FINALIZATION_FAILURE_CONTRACT_TYPE = (
    "safa_canonical_preflight_post_handoff_finalization_failure_v1"
)
OS_ERROR_TYPE_TOKENS = frozenset(
    {
        "OSError",
        "BlockingIOError",
        "ChildProcessError",
        "ConnectionError",
        "BrokenPipeError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "FileExistsError",
        "FileNotFoundError",
        "InterruptedError",
        "IsADirectoryError",
        "NotADirectoryError",
        "PermissionError",
        "ProcessLookupError",
        "TimeoutError",
    }
)


_PUBLISH_FAILURE_KEYS = {
    "commit_state",
    "stage",
    "message",
    "directory_seal",
    "payload",
    "temporary",
    "errno",
    "quarantined",
    "secondary_failures",
}
_SECONDARY_FAILURE_KEYS = {
    "stage",
    "type",
    "message",
    "errno",
    "identity",
}
_PUBLISH_COMMIT_STATES = {
    "precommit_failed_clean",
    "durability_unknown_quarantined",
    "committed_cleanup_error",
    "collision",
}


def validate_publish_failure_record(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    secondary = value.get("secondary_failures")
    if (
        set(value) != _PUBLISH_FAILURE_KEYS
        or value.get("commit_state")
        not in _PUBLISH_COMMIT_STATES
        or not isinstance(value.get("stage"), str)
        or not 0 < len(value["stage"]) <= 256
        or not isinstance(value.get("message"), str)
        or len(value["message"]) > 4096
        or (
            value.get("directory_seal") is not None
            and not isinstance(value["directory_seal"], dict)
        )
        or not isinstance(value.get("payload"), dict)
        or (
            value.get("temporary") is not None
            and not isinstance(value["temporary"], dict)
        )
        or (
            value.get("errno") is not None
            and type(value["errno"]) is not int
        )
        or value.get("quarantined")
        is not (
            value.get("commit_state")
            in {
                "durability_unknown_quarantined",
                "committed_cleanup_error",
            }
        )
        or not isinstance(secondary, list)
        or len(secondary) > 8
        or any(
            not isinstance(item, dict)
            or set(item) != _SECONDARY_FAILURE_KEYS
            or not isinstance(item.get("stage"), str)
            or not 0 < len(item["stage"]) <= 256
            or not isinstance(item.get("type"), str)
            or not 0 < len(item["type"]) <= 256
            or not isinstance(item.get("message"), str)
            or len(item["message"]) > 4096
            or (
                item.get("errno") is not None
                and type(item["errno"]) is not int
            )
            or (
                item.get("identity") is not None
                and not isinstance(item["identity"], dict)
            )
            for item in secondary
        )
    ):
        raise PreflightLaunchContractError(
            "typed publication failure record differs"
        )
    return {
        **dict(value),
        "secondary_failures": [
            {
                **dict(item),
                "identity": (
                    None
                    if item["identity"] is None
                    else dict(item["identity"])
                ),
            }
            for item in secondary
        ],
    }


def build_publish_failure_record(
    *,
    commit_state: str,
    stage: str,
    message: str,
    directory_seal: Mapping[str, Any] | None,
    payload: Mapping[str, Any],
    temporary: Mapping[str, Any] | None,
    error_number: int | None,
    secondary_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    return validate_publish_failure_record(
        {
            "commit_state": commit_state,
            "stage": stage,
            "message": message,
            "directory_seal": (
                None
                if directory_seal is None
                else dict(directory_seal)
            ),
            "payload": dict(payload),
            "temporary": (
                None if temporary is None else dict(temporary)
            ),
            "errno": error_number,
            "quarantined": commit_state
            in {
                "durability_unknown_quarantined",
                "committed_cleanup_error",
            },
            "secondary_failures": [
                dict(item) for item in secondary_failures
            ],
        }
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
_PANE_FAULT_CONSUMER_ARTIFACT_NAMES = {
    "attempt",
    "log",
    "self_fault_channel",
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
    "lifecycle_wait_channel",
}
_PANE_FAULT_CONSUMER_PUBLISHER_KEYS = {
    "path",
    "sha256",
    "role",
}
_PANE_FAULT_CONSUMER_REGISTRATION_KEYS = {
    "schema_version",
    "contract_type",
    "namespace",
    "artifacts",
    "publishers",
}
_PANE_FAULT_CONSUMER_CHAIN_KEYS = {
    "consumer_started",
    "consumer_active",
    "consumer_reader_release",
    "consumer_release_observed",
}
_DEADLINE_OBSERVATION_KEYS = {
    "started_monotonic_ns",
    "deadline_monotonic_ns",
    "observed_monotonic_ns",
    "deadline_reached",
}
_INVALID_CLAIM_EVIDENCE_KEYS = {
    "raw_content_sha256",
    "file_identity",
}
_PRECLAIM_FAILURE_INTENT_KEYS = {
    "schema_version",
    "contract_type",
    "attempt_id",
    "launch_receipt",
    "launch_receipt_identity",
    "verified_implementations",
    "wrapper_claim_path",
    "pane_fault_consumer_chain",
    "controller_owner_seal",
    "reason",
    "stage",
    "deadline_observation",
    "invalid_claim_evidence",
    "observed_at",
    "preclaim_failure_intent_sha256",
}
_BOUND_LIFECYCLE_EVIDENCE_KEYS = {"artifact", "record"}
_OWNERSHIP_ABSENT_KEYS = {
    "launch_accepted",
    "launch_ownership_release",
    "wrapper_claim",
}
_TERMINAL_FAILURE_KEYS = {
    "reason",
    "stage",
    "type",
    "message",
}
_LAUNCH_TERMINAL_V2_KEYS = {
    "schema_version",
    "contract_type",
    "attempt_id",
    "preclaim_failure_intent",
    "launch_receipt",
    "launch_receipt_identity",
    "verified_implementations",
    "pane_fault_consumer_chain",
    "gate_execution_terminal",
    "gate_lifecycle",
    "controller_cleanup",
    "consumer_terminal",
    "consumer_lifecycle",
    "consumer_join",
    "consumer_cleanup",
    "ownership",
    "status",
    "failure",
    "session_residual",
    "process_residual",
    "started_at",
    "completed_at",
    "launch_terminal_sha256",
}
_ATTEMPTED_TERMINAL_PAYLOAD_KEYS = {
    "target_path",
    "canonical_sha256",
    "content_sha256",
    "size",
}
_FINALIZATION_SECONDARY_FAILURE_KEYS = {
    "stage",
    "type",
    "message",
}
_LAUNCHER_TERMINAL_PUBLISH_ERROR_KEYS = {
    "type",
    "message",
    "path",
    "secondary_failures",
}
_FINALIZATION_INNER_FAILURE_KEYS = {
    "exception_type",
    "exception_message",
    "error_number",
}
_POST_HANDOFF_FAILURE_KEYS = {
    "outer",
    "inner",
}
_POST_HANDOFF_FINALIZATION_FAILURE_KEYS = {
    "schema_version",
    "contract_type",
    "attempt_id",
    "preclaim_failure_intent",
    "attempted_launch_terminal",
    "gate_execution_terminal",
    "gate_lifecycle",
    "controller_cleanup",
    "consumer_terminal",
    "consumer_lifecycle",
    "consumer_join",
    "consumer_cleanup",
    "ownership",
    "stage",
    "failure",
    "session_residual",
    "process_residual",
    "started_at",
    "completed_at",
    "post_handoff_finalization_failure_sha256",
}
_LAUNCH_RECEIPT_KEYS = {
    "schema_version",
    "contract_type",
    "attempt_id",
    "started_registry",
    "policy_sha256",
    "git",
    "bindings",
    "verified_implementations",
    "python_executable",
    "controller_session",
    "controller_owner_nonce",
    "observer_session",
    "wrapper_arguments",
    "gate_lifecycle_wait_channel",
    "gate_lifecycle_wait_publisher",
    "gate_lifecycle_wait_supervisor_arguments",
    "gate_lifecycle_wait_supervisor_ready_path",
    "gate_lifecycle_wait_status_path",
    "gate_worker_arguments",
    "consumer_lifecycle_wait_channel",
    "consumer_lifecycle_wait_publisher",
    "consumer_lifecycle_wait_supervisor_arguments",
    "consumer_lifecycle_wait_supervisor_ready_path",
    "consumer_lifecycle_wait_status_path",
    "consumer_worker_arguments",
    "consumer_session",
    "consumer_owner_nonce",
    "consumer_tmux_arguments",
    "tmux_arguments",
    "shell",
    "pane_log",
    "fault_channel",
    "pane_gate_fault_channel",
    "pane_gate_fault_publisher",
    "pane_fault_consumer",
    "wrapper_claim_path",
    "wrapper_started_path",
    "gate_execution_terminal_path",
    "started_at",
    "launch_receipt_sha256",
}
_LAUNCH_RECEIPT_V5_KEYS = _LAUNCH_RECEIPT_KEYS | {
    "postclaim_finalization_profile",
    "gate_lifecycle_wait_publish_fault_channel",
    "gate_lifecycle_wait_publish_fault_publisher",
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
    "pane_fault_consumer_chain",
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
    "pane_fault_consumer_chain",
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
    "pane_fault_consumer_chain",
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
    "pane_fault_consumer_chain",
    "launch_ownership_release_sha256",
}


def validate_pane_fault_consumer_registration(
    raw: Any,
    *,
    expected_namespace: str | None = None,
    label: str = "pane fault consumer registration",
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(
        value,
        _PANE_FAULT_CONSUMER_REGISTRATION_KEYS,
        label,
    )
    namespace = value["namespace"]
    artifacts = _mapping(value["artifacts"], f"{label} artifacts")
    publishers = _mapping(
        value["publishers"], f"{label} publishers"
    )
    if (
        value["schema_version"] != 1
        or value["contract_type"]
        != PANE_FAULT_CONSUMER_REGISTRATION_CONTRACT_TYPE
        or not isinstance(namespace, str)
        or not namespace.startswith("/")
        or (
            expected_namespace is not None
            and namespace != expected_namespace
        )
        or set(artifacts) != _PANE_FAULT_CONSUMER_ARTIFACT_NAMES
        or any(
            not isinstance(path, str)
            or path
            != (
                f"{namespace}/consumer_{name}.json"
                if name
                not in {
                    "attempt",
                    "log",
                    "self_fault_channel",
                    "lifecycle_wait_channel",
                }
                else {
                    "attempt": f"{namespace}/consumer_attempt.json",
                    "log": f"{namespace}/consumer.log",
                    "self_fault_channel": (
                        f"{namespace}/consumer_self_fault.channel"
                    ),
                    "lifecycle_wait_channel": (
                        f"{namespace}/consumer_lifecycle_wait.channel"
                    ),
                }[name]
            )
            for name, path in artifacts.items()
        )
        or set(publishers) != {"launcher", "consumer"}
    ):
        raise PreflightLaunchContractError(
            f"{label} schema or path relation differs"
        )
    normalized_publishers: dict[str, dict[str, str]] = {}
    for name, raw_publisher in publishers.items():
        publisher = _mapping(
            raw_publisher, f"{label} {name} publisher"
        )
        if (
            set(publisher) != _PANE_FAULT_CONSUMER_PUBLISHER_KEYS
            or not isinstance(publisher["path"], str)
            or not publisher["path"].startswith("/")
            or not isinstance(publisher["role"], str)
            or publisher["role"]
            != {
                "launcher": "launcher_pane_fault_consumer_handoff",
                "consumer": "pane_fault_consumer",
            }[name]
        ):
            raise PreflightLaunchContractError(
                f"{label} {name} publisher differs"
            )
        _hex64(
            publisher["sha256"],
            f"{label} {name} publisher SHA",
        )
        normalized_publishers[name] = {
            key: str(publisher[key])
            for key in _PANE_FAULT_CONSUMER_PUBLISHER_KEYS
        }
    if (
        normalized_publishers["launcher"]["path"]
        != normalized_publishers["consumer"]["path"]
        or normalized_publishers["launcher"]["sha256"]
        != normalized_publishers["consumer"]["sha256"]
    ):
        raise PreflightLaunchContractError(
            f"{label} publisher implementation differs"
        )
    return {
        **value,
        "artifacts": {
            name: str(path) for name, path in artifacts.items()
        },
        "publishers": normalized_publishers,
    }


def build_pane_fault_consumer_registration(
    *,
    namespace: str,
    artifacts: Mapping[str, Any],
    publishers: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_pane_fault_consumer_registration(
        {
            "schema_version": 1,
            "contract_type": (
                PANE_FAULT_CONSUMER_REGISTRATION_CONTRACT_TYPE
            ),
            "namespace": namespace,
            "artifacts": dict(artifacts),
            "publishers": dict(publishers),
        },
        expected_namespace=namespace,
    )


def validate_pane_fault_consumer_chain(
    raw: Any,
    *,
    registration: Mapping[str, Any] | None = None,
    label: str = "pane fault consumer chain",
) -> dict[str, dict[str, str]]:
    value = _mapping(raw, label)
    _exact_keys(value, _PANE_FAULT_CONSUMER_CHAIN_KEYS, label)
    normalized = {
        name: validate_artifact_binding(
            _mapping(binding, f"{label} {name}"),
            f"{label} {name}",
        )
        for name, binding in value.items()
    }
    if registration is not None:
        registered = validate_pane_fault_consumer_registration(
            registration,
            label=f"{label} registration",
        )
        expected_paths = {
            "consumer_started": registered["artifacts"]["started"],
            "consumer_active": registered["artifacts"]["active"],
            "consumer_reader_release": registered["artifacts"][
                "reader_release"
            ],
            "consumer_release_observed": registered["artifacts"][
                "release_observed"
            ],
        }
        if any(
            normalized[name]["path"] != path
            for name, path in expected_paths.items()
        ):
            raise PreflightLaunchContractError(
                f"{label} registered path relation differs"
            )
    return normalized


def build_pane_fault_consumer_chain(
    *,
    consumer_started: Mapping[str, Any],
    consumer_active: Mapping[str, Any],
    consumer_reader_release: Mapping[str, Any],
    consumer_release_observed: Mapping[str, Any],
    registration: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    return validate_pane_fault_consumer_chain(
        {
            "consumer_started": dict(consumer_started),
            "consumer_active": dict(consumer_active),
            "consumer_reader_release": dict(
                consumer_reader_release
            ),
            "consumer_release_observed": dict(
                consumer_release_observed
            ),
        },
        registration=registration,
    )


def _validate_deadline_observation(
    raw: Any,
    *,
    label: str,
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _DEADLINE_OBSERVATION_KEYS, label)
    started = value["started_monotonic_ns"]
    deadline = value["deadline_monotonic_ns"]
    observed = value["observed_monotonic_ns"]
    if (
        type(started) is not int
        or type(deadline) is not int
        or type(observed) is not int
        or started < 0
        or deadline <= started
        or observed < started
        or type(value["deadline_reached"]) is not bool
        or value["deadline_reached"] is not (observed >= deadline)
    ):
        raise PreflightLaunchContractError(
            f"{label} monotonic relation differs"
        )
    return dict(value)


def validate_invalid_claim_evidence(
    raw: Any,
    *,
    label: str = "invalid claim evidence",
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _INVALID_CLAIM_EVIDENCE_KEYS, label)
    _hex64(value["raw_content_sha256"], f"{label} raw content SHA")
    identity = validate_file_identity(
        value["file_identity"], f"{label} file identity"
    )
    return {
        "raw_content_sha256": str(value["raw_content_sha256"]),
        "file_identity": identity,
    }


def build_invalid_claim_evidence(
    *,
    raw_content_sha256: str,
    file_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return validate_invalid_claim_evidence(
        {
            "raw_content_sha256": raw_content_sha256,
            "file_identity": dict(file_identity),
        }
    )


def validate_preclaim_failure_intent(
    raw: Any,
    *,
    verified_implementations: Mapping[str, Any],
    expected_wrapper_claim_path: str,
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
    expected_receipt: Mapping[str, Any] | None = None,
    expected_receipt_identity: Mapping[str, Any] | None = None,
    expected_consumer_chain: Mapping[str, Any] | None = None,
    label: str = "preclaim failure intent",
) -> dict[str, Any]:
    value = _validate_digested_contract(
        raw,
        expected_keys=_PRECLAIM_FAILURE_INTENT_KEYS,
        contract_type=PRECLAIM_FAILURE_INTENT_CONTRACT_TYPE,
        digest_field="preclaim_failure_intent_sha256",
        label=label,
    )
    _hex64(value["attempt_id"], f"{label} attempt ID")
    receipt = validate_artifact_binding(
        value["launch_receipt"], f"{label} launch receipt"
    )
    receipt_identity = validate_file_identity(
        value["launch_receipt_identity"],
        f"{label} launch receipt identity",
    )
    normalized_implementations = validate_verified_implementations(
        verified_implementations,
        f"{label} expected verified implementations",
    )
    if (
        not isinstance(expected_wrapper_claim_path, str)
        or not expected_wrapper_claim_path.startswith("/")
    ):
        raise PreflightLaunchContractError(
            f"{label} expected wrapper claim path differs"
        )
    consumer_chain = validate_pane_fault_consumer_chain(
        value["pane_fault_consumer_chain"],
        label=f"{label} pane fault consumer chain",
    )
    owner_seal = validate_pane_owner_seal(
        value["controller_owner_seal"],
        tmux_identity=tmux_identity,
        tmux_server=tmux_server,
        label=f"{label} controller owner seal",
    )
    deadline = _validate_deadline_observation(
        value["deadline_observation"],
        label=f"{label} deadline observation",
    )
    reason = value["reason"]
    expected_stage = {
        "invalid_claim": "wrapper_claim_validation",
        "claim_timeout": "wrapper_claim_wait_deadline",
    }.get(reason)
    invalid_evidence = value["invalid_claim_evidence"]
    if (
        expected_stage is None
        or value["stage"] != expected_stage
        or value["verified_implementations"]
        != normalized_implementations
        or value["wrapper_claim_path"]
        != expected_wrapper_claim_path
        or not isinstance(value["observed_at"], str)
        or not value["observed_at"].endswith("+00:00")
        or (
            reason == "claim_timeout"
            and (
                deadline["deadline_reached"] is not True
                or invalid_evidence is not None
            )
        )
        or (
            reason == "invalid_claim"
            and invalid_evidence is None
        )
    ):
        raise PreflightLaunchContractError(
            f"{label} failure relation differs"
        )
    normalized_invalid = (
        None
        if invalid_evidence is None
        else validate_invalid_claim_evidence(
            invalid_evidence,
            label=f"{label} invalid claim evidence",
        )
    )
    try:
        observed_at = datetime.fromisoformat(value["observed_at"])
    except ValueError as exc:
        raise PreflightLaunchContractError(
            f"{label} observed UTC timestamp differs"
        ) from exc
    if (
        observed_at.tzinfo is None
        or observed_at.utcoffset() != timezone.utc.utcoffset(None)
        or observed_at.isoformat() != value["observed_at"]
        or (
            normalized_invalid is not None
            and normalized_invalid["file_identity"]["path"]
            != expected_wrapper_claim_path
        )
    ):
        raise PreflightLaunchContractError(
            f"{label} observed or claim path relation differs"
        )
    if (
        expected_receipt is not None
        and receipt
        != validate_artifact_binding(
            expected_receipt, f"{label} expected launch receipt"
        )
    ) or (
        expected_receipt_identity is not None
        and receipt_identity
        != validate_file_identity(
            expected_receipt_identity,
            f"{label} expected launch receipt identity",
        )
    ) or (
        expected_consumer_chain is not None
        and consumer_chain
        != validate_pane_fault_consumer_chain(
            expected_consumer_chain,
            label=f"{label} expected pane fault consumer chain",
        )
    ):
        raise PreflightLaunchContractError(
            f"{label} expected binding differs"
        )
    return {
        **value,
        "launch_receipt": receipt,
        "launch_receipt_identity": receipt_identity,
        "verified_implementations": normalized_implementations,
        "pane_fault_consumer_chain": consumer_chain,
        "controller_owner_seal": owner_seal,
        "deadline_observation": deadline,
        "invalid_claim_evidence": normalized_invalid,
    }


def build_preclaim_failure_intent(
    *,
    attempt_id: str,
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    wrapper_claim_path: str,
    pane_fault_consumer_chain: Mapping[str, Any],
    controller_owner_seal: Mapping[str, Any],
    reason: str,
    stage: str,
    deadline_observation: Mapping[str, Any],
    invalid_claim_evidence: Mapping[str, Any] | None,
    observed_at: str,
    tmux_identity: Mapping[str, Any],
    tmux_server: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": PRECLAIM_FAILURE_INTENT_CONTRACT_TYPE,
        "attempt_id": attempt_id,
        "launch_receipt": dict(launch_receipt),
        "launch_receipt_identity": dict(launch_receipt_identity),
        "verified_implementations": dict(
            verified_implementations
        ),
        "wrapper_claim_path": wrapper_claim_path,
        "pane_fault_consumer_chain": dict(
            pane_fault_consumer_chain
        ),
        "controller_owner_seal": dict(controller_owner_seal),
        "reason": reason,
        "stage": stage,
        "deadline_observation": dict(deadline_observation),
        "invalid_claim_evidence": (
            None
            if invalid_claim_evidence is None
            else dict(invalid_claim_evidence)
        ),
        "observed_at": observed_at,
    }
    value["preclaim_failure_intent_sha256"] = canonical_digest(
        value, "preclaim_failure_intent_sha256"
    )
    return validate_preclaim_failure_intent(
        value,
        verified_implementations=verified_implementations,
        expected_wrapper_claim_path=wrapper_claim_path,
        tmux_identity=tmux_identity,
        tmux_server=tmux_server,
        expected_receipt=launch_receipt,
        expected_receipt_identity=launch_receipt_identity,
        expected_consumer_chain=pane_fault_consumer_chain,
    )


def validate_bound_lifecycle_evidence(
    raw: Any,
    *,
    role: str,
    attempt_id: str,
    label: str,
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _BOUND_LIFECYCLE_EVIDENCE_KEYS, label)
    artifact = validate_artifact_binding(
        value["artifact"], f"{label} artifact"
    )
    record = validate_lifecycle_wait_status(
        _mapping(value["record"], f"{label} record"),
        role=role,
        label=f"{label} record",
    )
    if (
        record["attempt_id"] != attempt_id
        or artifact["canonical_sha256"]
        != record["lifecycle_wait_status_sha256"]
    ):
        raise PreflightLaunchContractError(
            f"{label} artifact or attempt binding differs"
        )
    return {"artifact": artifact, "record": record}


def build_bound_lifecycle_evidence(
    *,
    artifact: Mapping[str, Any],
    record: Mapping[str, Any],
    role: str,
    attempt_id: str,
) -> dict[str, Any]:
    return validate_bound_lifecycle_evidence(
        {
            "artifact": dict(artifact),
            "record": dict(record),
        },
        role=role,
        attempt_id=attempt_id,
        label=f"{role} bound lifecycle evidence",
    )


def _validate_ownership_absent(
    raw: Any,
    *,
    label: str,
) -> dict[str, None]:
    value = _mapping(raw, label)
    _exact_keys(value, _OWNERSHIP_ABSENT_KEYS, label)
    if any(item is not None for item in value.values()):
        raise PreflightLaunchContractError(f"{label} differs")
    return {
        "launch_accepted": None,
        "launch_ownership_release": None,
        "wrapper_claim": None,
    }


def validate_terminal_failure(
    raw: Any,
    *,
    reason: str,
    stage: str,
    label: str,
) -> dict[str, str]:
    value = _mapping(raw, label)
    _exact_keys(value, _TERMINAL_FAILURE_KEYS, label)
    if (
        value["reason"] != reason
        or value["stage"] != stage
        or not isinstance(value["type"], str)
        or not value["type"]
        or len(value["type"]) > 256
        or not isinstance(value["message"], str)
        or not value["message"]
        or len(value["message"]) > 4096
    ):
        raise PreflightLaunchContractError(f"{label} differs")
    return {key: str(value[key]) for key in _TERMINAL_FAILURE_KEYS}


def build_terminal_failure(
    *,
    reason: str,
    stage: str,
    failure_type: str,
    message: str,
) -> dict[str, str]:
    return validate_terminal_failure(
        {
            "reason": reason,
            "stage": stage,
            "type": failure_type,
            "message": message,
        },
        reason=reason,
        stage=stage,
        label="terminal failure",
    )


def validate_launch_terminal_v2(
    raw: Any,
    *,
    preclaim_failure_intent: Mapping[str, Any],
    preclaim_failure_intent_binding: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    label: str = "preclaim launch terminal v2",
) -> dict[str, Any]:
    value = _validate_digested_contract(
        raw,
        expected_keys=_LAUNCH_TERMINAL_V2_KEYS,
        contract_type=LAUNCH_TERMINAL_V2_CONTRACT_TYPE,
        digest_field="launch_terminal_sha256",
        label=label,
    )
    intent = _mapping(
        preclaim_failure_intent,
        f"{label} preclaim failure intent",
    )
    _exact_keys(
        intent,
        _PRECLAIM_FAILURE_INTENT_KEYS,
        f"{label} preclaim failure intent",
    )
    if (
        intent.get("contract_type")
        != PRECLAIM_FAILURE_INTENT_CONTRACT_TYPE
        or intent.get("preclaim_failure_intent_sha256")
        != canonical_digest(
            intent, "preclaim_failure_intent_sha256"
        )
    ):
        raise PreflightLaunchContractError(
            f"{label} preclaim failure intent differs"
        )
    attempt_id = _hex64(
        value["attempt_id"], f"{label} attempt ID"
    )
    intent_binding = validate_artifact_binding(
        value["preclaim_failure_intent"],
        f"{label} preclaim failure intent binding",
    )
    expected_intent_binding = validate_artifact_binding(
        preclaim_failure_intent_binding,
        f"{label} expected preclaim failure intent binding",
    )
    receipt = validate_artifact_binding(
        value["launch_receipt"], f"{label} launch receipt"
    )
    receipt_identity = validate_file_identity(
        value["launch_receipt_identity"],
        f"{label} launch receipt identity",
    )
    normalized_implementations = validate_verified_implementations(
        verified_implementations,
        f"{label} expected verified implementations",
    )
    consumer_chain = validate_pane_fault_consumer_chain(
        value["pane_fault_consumer_chain"],
        label=f"{label} pane fault consumer chain",
    )
    gate_terminal = validate_artifact_binding(
        value["gate_execution_terminal"],
        f"{label} gate execution terminal",
    )
    gate_lifecycle = validate_bound_lifecycle_evidence(
        value["gate_lifecycle"],
        role="gate",
        attempt_id=attempt_id,
        label=f"{label} gate lifecycle",
    )
    controller_cleanup = validate_artifact_binding(
        value["controller_cleanup"],
        f"{label} controller cleanup",
    )
    consumer_terminal = validate_artifact_binding(
        value["consumer_terminal"],
        f"{label} consumer terminal",
    )
    consumer_lifecycle = validate_bound_lifecycle_evidence(
        value["consumer_lifecycle"],
        role="consumer",
        attempt_id=attempt_id,
        label=f"{label} consumer lifecycle",
    )
    consumer_join = validate_artifact_binding(
        value["consumer_join"], f"{label} consumer join"
    )
    consumer_cleanup = validate_artifact_binding(
        value["consumer_cleanup"], f"{label} consumer cleanup"
    )
    ownership = _validate_ownership_absent(
        value["ownership"], label=f"{label} ownership"
    )
    reason = intent["reason"]
    stage = intent["stage"]
    failure = validate_terminal_failure(
        value["failure"],
        reason=reason,
        stage=stage,
        label=f"{label} failure",
    )
    expected_status = {
        "invalid_claim": "launcher_failed",
        "claim_timeout": "wrapper_claim_timeout",
    }.get(reason)
    consumer_namespace = consumer_chain[
        "consumer_started"
    ]["path"].rsplit("/", 1)[0]
    expected_consumer_paths = {
        "controller_cleanup": (
            f"{consumer_namespace}/consumer_controller_cleanup.json"
        ),
        "consumer_terminal": (
            f"{consumer_namespace}/consumer_terminal.json"
        ),
        "consumer_join": f"{consumer_namespace}/consumer_join.json",
        "consumer_cleanup": (
            f"{consumer_namespace}/consumer_cleanup.json"
        ),
    }
    try:
        started_at = datetime.fromisoformat(value["started_at"])
        completed_at = datetime.fromisoformat(value["completed_at"])
    except (TypeError, ValueError) as exc:
        raise PreflightLaunchContractError(
            f"{label} UTC timestamp differs"
        ) from exc
    if (
        attempt_id != intent["attempt_id"]
        or intent_binding != expected_intent_binding
        or intent_binding["canonical_sha256"]
        != intent["preclaim_failure_intent_sha256"]
        or receipt != intent["launch_receipt"]
        or receipt_identity != intent["launch_receipt_identity"]
        or value["verified_implementations"]
        != normalized_implementations
        or normalized_implementations
        != intent["verified_implementations"]
        or consumer_chain != intent["pane_fault_consumer_chain"]
        or gate_lifecycle["record"]["policy_sha256"]
        != consumer_lifecycle["record"]["policy_sha256"]
        or value["status"] != expected_status
        or value["session_residual"] is not False
        or value["process_residual"] is not False
        or not isinstance(value["started_at"], str)
        or not value["started_at"].endswith("+00:00")
        or started_at.tzinfo is None
        or started_at.utcoffset()
        != timezone.utc.utcoffset(None)
        or started_at.isoformat() != value["started_at"]
        or not isinstance(value["completed_at"], str)
        or not value["completed_at"].endswith("+00:00")
        or completed_at.tzinfo is None
        or completed_at.utcoffset()
        != timezone.utc.utcoffset(None)
        or completed_at.isoformat() != value["completed_at"]
        or completed_at < started_at
        or controller_cleanup["path"]
        != expected_consumer_paths["controller_cleanup"]
        or consumer_terminal["path"]
        != expected_consumer_paths["consumer_terminal"]
        or consumer_join["path"]
        != expected_consumer_paths["consumer_join"]
        or consumer_cleanup["path"]
        != expected_consumer_paths["consumer_cleanup"]
        or len(
            {
                gate_terminal["path"],
                gate_lifecycle["artifact"]["path"],
                controller_cleanup["path"],
                consumer_terminal["path"],
                consumer_lifecycle["artifact"]["path"],
                consumer_join["path"],
                consumer_cleanup["path"],
            }
        )
        != 7
    ):
        raise PreflightLaunchContractError(
            f"{label} finalization relation differs"
        )
    return {
        **value,
        "preclaim_failure_intent": intent_binding,
        "launch_receipt": receipt,
        "launch_receipt_identity": receipt_identity,
        "verified_implementations": normalized_implementations,
        "pane_fault_consumer_chain": consumer_chain,
        "gate_execution_terminal": gate_terminal,
        "gate_lifecycle": gate_lifecycle,
        "controller_cleanup": controller_cleanup,
        "consumer_terminal": consumer_terminal,
        "consumer_lifecycle": consumer_lifecycle,
        "consumer_join": consumer_join,
        "consumer_cleanup": consumer_cleanup,
        "ownership": ownership,
        "failure": failure,
    }


def build_launch_terminal_v2(
    *,
    attempt_id: str,
    preclaim_failure_intent: Mapping[str, Any],
    preclaim_failure_intent_binding: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_receipt_identity: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    pane_fault_consumer_chain: Mapping[str, Any],
    gate_execution_terminal: Mapping[str, Any],
    gate_lifecycle: Mapping[str, Any],
    controller_cleanup: Mapping[str, Any],
    consumer_terminal: Mapping[str, Any],
    consumer_lifecycle: Mapping[str, Any],
    consumer_join: Mapping[str, Any],
    consumer_cleanup: Mapping[str, Any],
    status: str,
    failure: Mapping[str, Any],
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": LAUNCH_TERMINAL_V2_CONTRACT_TYPE,
        "attempt_id": attempt_id,
        "preclaim_failure_intent": dict(
            preclaim_failure_intent_binding
        ),
        "launch_receipt": dict(launch_receipt),
        "launch_receipt_identity": dict(launch_receipt_identity),
        "verified_implementations": dict(
            verified_implementations
        ),
        "pane_fault_consumer_chain": dict(
            pane_fault_consumer_chain
        ),
        "gate_execution_terminal": dict(gate_execution_terminal),
        "gate_lifecycle": dict(gate_lifecycle),
        "controller_cleanup": dict(controller_cleanup),
        "consumer_terminal": dict(consumer_terminal),
        "consumer_lifecycle": dict(consumer_lifecycle),
        "consumer_join": dict(consumer_join),
        "consumer_cleanup": dict(consumer_cleanup),
        "ownership": {
            "launch_accepted": None,
            "launch_ownership_release": None,
            "wrapper_claim": None,
        },
        "status": status,
        "failure": dict(failure),
        "session_residual": False,
        "process_residual": False,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    value["launch_terminal_sha256"] = canonical_digest(
        value, "launch_terminal_sha256"
    )
    return validate_launch_terminal_v2(
        value,
        preclaim_failure_intent=preclaim_failure_intent,
        preclaim_failure_intent_binding=(
            preclaim_failure_intent_binding
        ),
        verified_implementations=verified_implementations,
    )


def validate_finalization_secondary_failure(
    raw: Any,
    *,
    label: str = "finalization secondary failure",
) -> dict[str, str]:
    item = _mapping(raw, label)
    _exact_keys(item, _FINALIZATION_SECONDARY_FAILURE_KEYS, label)
    if any(
        not isinstance(item[key], str)
        or not item[key]
        or len(item[key]) > (4096 if key == "message" else 256)
        for key in _FINALIZATION_SECONDARY_FAILURE_KEYS
    ):
        raise PreflightLaunchContractError(f"{label} differs")
    return {
        key: str(item[key])
        for key in _FINALIZATION_SECONDARY_FAILURE_KEYS
    }


def build_finalization_secondary_failure(
    *,
    stage: str,
    failure_type: str,
    message: str,
) -> dict[str, str]:
    return validate_finalization_secondary_failure(
        {
            "stage": stage,
            "type": failure_type,
            "message": message,
        }
    )


def _validate_finalization_secondary_failures(
    raw: Any,
    *,
    label: str,
) -> list[dict[str, str]]:
    if not isinstance(raw, list) or len(raw) > 8:
        raise PreflightLaunchContractError(f"{label} differs")
    normalized: list[dict[str, str]] = []
    for index, item_raw in enumerate(raw):
        normalized.append(
            validate_finalization_secondary_failure(
                item_raw, label=f"{label} item {index}"
            )
        )
    return normalized


def _validate_post_handoff_failure(
    raw: Any,
    *,
    publish_target_path: str,
    label: str,
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _POST_HANDOFF_FAILURE_KEYS, label)
    outer = _mapping(value["outer"], f"{label} outer")
    inner = _mapping(value["inner"], f"{label} inner")
    _exact_keys(
        outer,
        _LAUNCHER_TERMINAL_PUBLISH_ERROR_KEYS,
        f"{label} outer",
    )
    _exact_keys(
        inner, _FINALIZATION_INNER_FAILURE_KEYS, f"{label} inner"
    )
    outer_secondary = _validate_finalization_secondary_failures(
        outer["secondary_failures"],
        label=f"{label} outer secondary failures",
    )
    if (
        outer["type"] != "LauncherTerminalPublishError"
        or not isinstance(outer["message"], str)
        or not outer["message"]
        or len(outer["message"]) > 4096
        or outer["path"] != publish_target_path
        or inner["exception_type"] not in OS_ERROR_TYPE_TOKENS
        or not isinstance(inner["exception_message"], str)
        or not inner["exception_message"]
        or len(inner["exception_message"]) > 4096
        or (
            inner["error_number"] is not None
            and type(inner["error_number"]) is not int
        )
    ):
        raise PreflightLaunchContractError(
            f"{label} typed failure relation differs"
        )
    return {
        "outer": {
            "type": str(outer["type"]),
            "message": str(outer["message"]),
            "path": str(outer["path"]),
            "secondary_failures": outer_secondary,
        },
        "inner": {
            "exception_type": str(inner["exception_type"]),
            "exception_message": str(inner["exception_message"]),
            "error_number": inner["error_number"],
        },
    }


def build_finalization_inner_failure(
    failure: BaseException,
) -> dict[str, Any]:
    if not isinstance(failure, OSError):
        raise PreflightLaunchContractError(
            "finalization inner failure is not an OSError"
        )
    exception_type = type(failure).__name__
    if exception_type not in OS_ERROR_TYPE_TOKENS:
        raise PreflightLaunchContractError(
            "finalization inner OSError type is not registered"
        )
    error_number = failure.errno
    if error_number is not None and type(error_number) is not int:
        raise PreflightLaunchContractError(
            "finalization inner OSError errno differs"
        )
    return {
        "exception_type": exception_type,
        "exception_message": str(failure),
        "error_number": error_number,
    }


def validate_post_handoff_finalization_failure(
    raw: Any,
    *,
    attempted_launch_terminal: Mapping[str, Any],
    preclaim_failure_intent: Mapping[str, Any],
    preclaim_failure_intent_binding: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    publish_target_path: str,
    attempted_content_sha256: str,
    attempted_payload_size: int,
    label: str = "post-handoff finalization failure",
) -> dict[str, Any]:
    value = _validate_digested_contract(
        raw,
        expected_keys=_POST_HANDOFF_FINALIZATION_FAILURE_KEYS,
        contract_type=POST_HANDOFF_FINALIZATION_FAILURE_CONTRACT_TYPE,
        digest_field="post_handoff_finalization_failure_sha256",
        label=label,
    )
    terminal = validate_launch_terminal_v2(
        attempted_launch_terminal,
        preclaim_failure_intent=preclaim_failure_intent,
        preclaim_failure_intent_binding=(
            preclaim_failure_intent_binding
        ),
        verified_implementations=verified_implementations,
        label=f"{label} attempted launch terminal",
    )
    attempt_id = _hex64(
        value["attempt_id"], f"{label} attempt ID"
    )
    intent_binding = validate_artifact_binding(
        value["preclaim_failure_intent"],
        f"{label} preclaim failure intent",
    )
    expected_intent_binding = validate_artifact_binding(
        preclaim_failure_intent_binding,
        f"{label} expected preclaim failure intent",
    )
    payload = _mapping(
        value["attempted_launch_terminal"],
        f"{label} attempted launch terminal payload",
    )
    _exact_keys(
        payload,
        _ATTEMPTED_TERMINAL_PAYLOAD_KEYS,
        f"{label} attempted launch terminal payload",
    )
    _hex64(
        payload["canonical_sha256"],
        f"{label} attempted terminal canonical SHA",
    )
    _hex64(
        payload["content_sha256"],
        f"{label} attempted terminal content SHA",
    )
    _hex64(
        attempted_content_sha256,
        f"{label} expected attempted terminal content SHA",
    )
    if (
        not isinstance(publish_target_path, str)
        or not publish_target_path.startswith("/")
        or type(attempted_payload_size) is not int
        or attempted_payload_size <= 0
    ):
        raise PreflightLaunchContractError(
            f"{label} expected attempted terminal payload differs"
        )
    gate_terminal = validate_artifact_binding(
        value["gate_execution_terminal"],
        f"{label} gate execution terminal",
    )
    gate_lifecycle = validate_bound_lifecycle_evidence(
        value["gate_lifecycle"],
        role="gate",
        attempt_id=attempt_id,
        label=f"{label} gate lifecycle",
    )
    controller_cleanup = validate_artifact_binding(
        value["controller_cleanup"],
        f"{label} controller cleanup",
    )
    consumer_terminal = validate_artifact_binding(
        value["consumer_terminal"],
        f"{label} consumer terminal",
    )
    consumer_lifecycle = validate_bound_lifecycle_evidence(
        value["consumer_lifecycle"],
        role="consumer",
        attempt_id=attempt_id,
        label=f"{label} consumer lifecycle",
    )
    consumer_join = validate_artifact_binding(
        value["consumer_join"], f"{label} consumer join"
    )
    consumer_cleanup = validate_artifact_binding(
        value["consumer_cleanup"], f"{label} consumer cleanup"
    )
    ownership = _validate_ownership_absent(
        value["ownership"], label=f"{label} ownership"
    )
    failure = _validate_post_handoff_failure(
        value["failure"],
        publish_target_path=publish_target_path,
        label=f"{label} failure",
    )
    try:
        started_at = datetime.fromisoformat(value["started_at"])
        completed_at = datetime.fromisoformat(value["completed_at"])
    except (TypeError, ValueError) as exc:
        raise PreflightLaunchContractError(
            f"{label} UTC timestamp differs"
        ) from exc
    if (
        attempt_id != terminal["attempt_id"]
        or intent_binding != expected_intent_binding
        or payload["target_path"] != publish_target_path
        or payload["canonical_sha256"]
        != terminal["launch_terminal_sha256"]
        or payload["content_sha256"] != attempted_content_sha256
        or payload["size"] != attempted_payload_size
        or type(payload["size"]) is not int
        or payload["size"] <= 0
        or gate_terminal != terminal["gate_execution_terminal"]
        or gate_lifecycle != terminal["gate_lifecycle"]
        or controller_cleanup != terminal["controller_cleanup"]
        or consumer_terminal != terminal["consumer_terminal"]
        or consumer_lifecycle != terminal["consumer_lifecycle"]
        or consumer_join != terminal["consumer_join"]
        or consumer_cleanup != terminal["consumer_cleanup"]
        or ownership != terminal["ownership"]
        or value["stage"] != "launch_terminal_publish"
        or value["session_residual"] is not False
        or value["process_residual"] is not False
        or not isinstance(value["started_at"], str)
        or not value["started_at"].endswith("+00:00")
        or started_at.tzinfo is None
        or started_at.utcoffset()
        != timezone.utc.utcoffset(None)
        or started_at.isoformat() != value["started_at"]
        or not isinstance(value["completed_at"], str)
        or not value["completed_at"].endswith("+00:00")
        or completed_at.tzinfo is None
        or completed_at.utcoffset()
        != timezone.utc.utcoffset(None)
        or completed_at.isoformat() != value["completed_at"]
        or completed_at < started_at
    ):
        raise PreflightLaunchContractError(
            f"{label} finalization relation differs"
        )
    return {
        **value,
        "preclaim_failure_intent": intent_binding,
        "attempted_launch_terminal": {
            "target_path": str(payload["target_path"]),
            "canonical_sha256": str(payload["canonical_sha256"]),
            "content_sha256": str(payload["content_sha256"]),
            "size": int(payload["size"]),
        },
        "gate_execution_terminal": gate_terminal,
        "gate_lifecycle": gate_lifecycle,
        "controller_cleanup": controller_cleanup,
        "consumer_terminal": consumer_terminal,
        "consumer_lifecycle": consumer_lifecycle,
        "consumer_join": consumer_join,
        "consumer_cleanup": consumer_cleanup,
        "ownership": ownership,
        "failure": failure,
    }


def build_post_handoff_finalization_failure(
    *,
    attempted_launch_terminal: Mapping[str, Any],
    preclaim_failure_intent: Mapping[str, Any],
    preclaim_failure_intent_binding: Mapping[str, Any],
    verified_implementations: Mapping[str, Any],
    publish_target_path: str,
    attempted_content_sha256: str,
    attempted_payload_size: int,
    failure: Mapping[str, Any],
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    terminal = validate_launch_terminal_v2(
        attempted_launch_terminal,
        preclaim_failure_intent=preclaim_failure_intent,
        preclaim_failure_intent_binding=(
            preclaim_failure_intent_binding
        ),
        verified_implementations=verified_implementations,
        label="post-handoff attempted launch terminal",
    )
    value = {
        "schema_version": 1,
        "contract_type": (
            POST_HANDOFF_FINALIZATION_FAILURE_CONTRACT_TYPE
        ),
        "attempt_id": terminal["attempt_id"],
        "preclaim_failure_intent": dict(
            preclaim_failure_intent_binding
        ),
        "attempted_launch_terminal": {
            "target_path": publish_target_path,
            "canonical_sha256": terminal[
                "launch_terminal_sha256"
            ],
            "content_sha256": attempted_content_sha256,
            "size": attempted_payload_size,
        },
        "gate_execution_terminal": dict(
            terminal["gate_execution_terminal"]
        ),
        "gate_lifecycle": dict(terminal["gate_lifecycle"]),
        "controller_cleanup": dict(terminal["controller_cleanup"]),
        "consumer_terminal": dict(terminal["consumer_terminal"]),
        "consumer_lifecycle": dict(terminal["consumer_lifecycle"]),
        "consumer_join": dict(terminal["consumer_join"]),
        "consumer_cleanup": dict(terminal["consumer_cleanup"]),
        "ownership": dict(terminal["ownership"]),
        "stage": "launch_terminal_publish",
        "failure": dict(failure),
        "session_residual": False,
        "process_residual": False,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    value["post_handoff_finalization_failure_sha256"] = (
        canonical_digest(
            value, "post_handoff_finalization_failure_sha256"
        )
    )
    return validate_post_handoff_finalization_failure(
        value,
        attempted_launch_terminal=terminal,
        preclaim_failure_intent=preclaim_failure_intent,
        preclaim_failure_intent_binding=(
            preclaim_failure_intent_binding
        ),
        verified_implementations=verified_implementations,
        publish_target_path=publish_target_path,
        attempted_content_sha256=attempted_content_sha256,
        attempted_payload_size=attempted_payload_size,
    )


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


def derive_lifecycle_wait_outcome(
    *, wait_code: str, wait_status: int
) -> dict[str, Any]:
    if type(wait_status) is not int or not 0 <= wait_status <= 255:
        raise PreflightLaunchContractError(
            "lifecycle wait status is outside uint8"
        )
    if wait_code == "exited":
        return {
            "returncode": wait_status,
            "exit_kind": "exit",
            "exit_code": wait_status,
            "signal_number": None,
            "core_dumped": False,
        }
    if (
        wait_code in {"killed", "dumped"}
        and 0 < wait_status <= 64
    ):
        return {
            "returncode": -wait_status,
            "exit_kind": "signal",
            "exit_code": None,
            "signal_number": wait_status,
            "core_dumped": wait_code == "dumped",
        }
    raise PreflightLaunchContractError(
        "lifecycle wait code and status differ"
    )


_SEALED_LIFECYCLE_ARTIFACT_KEYS = {
    "kind",
    "binding",
    "file_identity",
}
_LIFECYCLE_PUBLISHER_KEYS = {
    "path",
    "sha256",
    "file_identity",
    "role",
}
_LIFECYCLE_ROLE_ARTIFACT_KINDS = {
    "gate": {
        "source_artifact": "launch_receipt",
        "worker_started": "gate_worker_started",
        "terminal": "gate_execution_terminal",
    },
    "consumer": {
        "source_artifact": "consumer_attempt",
        "worker_started": "consumer_worker_started",
        "terminal": "consumer_terminal",
    },
}
_LIFECYCLE_WAIT_STATUS_KEYS = {
    "schema_version",
    "contract_type",
    "role",
    "policy_sha256",
    "attempt_id",
    "source_artifact",
    "wait_channel",
    "publisher",
    "supervisor_owner_seal",
    "supervisor_process",
    "supervisor_executable",
    "supervisor_command",
    "worker_started",
    "child_process",
    "child_executable",
    "child_command",
    "terminal",
    "waitid_si_pid",
    "waitid_si_code",
    "waitid_si_status",
    "waited_pid",
    "wait_status_raw",
    "wait_code",
    "returncode",
    "exit_kind",
    "exit_code",
    "signal_number",
    "core_dumped",
    "started_at",
    "completed_at",
    "lifecycle_wait_status_sha256",
}
_LIFECYCLE_RAW_WAIT_V3_KEYS = {
    "schema_version",
    "contract_type",
    "role",
    "policy_sha256",
    "attempt_id",
    "source_artifact",
    "wait_channel",
    "publisher",
    "supervisor_owner_seal",
    "child_process",
    "waitid_si_pid",
    "waitid_si_code",
    "waitid_si_status",
    "waited_pid",
    "wait_status_raw",
    "wait_code",
    "returncode",
    "exit_kind",
    "exit_code",
    "signal_number",
    "core_dumped",
    "started_at",
    "reaped_at",
    "lifecycle_raw_wait_sha256",
}
_LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_KEYS = {
    "schema_version",
    "contract_type",
    "role",
    "policy_sha256",
    "attempt_id",
    "source_artifact",
    "target_channel",
    "fault_channel",
    "publisher",
    "supervisor_owner_seal",
    "child_process",
    "intended_raw_wait_sha256",
    "publish_failure",
    "recorded_at",
    "lifecycle_raw_wait_publish_failure_sha256",
}
_POSTCLAIM_PROFILE_COMPONENT_KEYS = {
    "schema_version",
    "contract_type",
}
_POSTCLAIM_FINALIZATION_PROFILE_V1_KEYS = {
    "schema_version",
    "contract_type",
    "raw_wait",
    "raw_wait_publish_failure",
    "controller_cleanup",
    "consumer_terminal",
    "consumer_join",
    "mixed_stack_allowed",
    "reap_to_raw_crash_policy",
    "post_raw_resume_policy",
    "postclaim_finalization_profile_sha256",
}
_LINUX_CLD_EXITED = 1
_LINUX_CLD_KILLED = 2
_LINUX_CLD_DUMPED = 3


def _decode_linux_waitpid_status(
    wait_status_raw: int,
) -> dict[str, Any]:
    if (
        type(wait_status_raw) is not int
        or not 0 <= wait_status_raw <= 0xFFFF
    ):
        raise PreflightLaunchContractError(
            "lifecycle raw waitpid status differs"
        )
    low_seven = wait_status_raw & 0x7F
    if wait_status_raw & 0xFF == 0:
        return {
            "wait_code": "exited",
            "wait_status": (wait_status_raw >> 8) & 0xFF,
            "returncode": (wait_status_raw >> 8) & 0xFF,
            "core_dumped": False,
        }
    if (
        low_seven == 0x7F
        or wait_status_raw & ~0xFF
        or not 1 <= low_seven <= 64
    ):
        raise PreflightLaunchContractError(
            "lifecycle raw waitpid status is not terminal"
        )
    core_dumped = bool(wait_status_raw & 0x80)
    return {
        "wait_code": "dumped" if core_dumped else "killed",
        "wait_status": low_seven,
        "returncode": -low_seven,
        "core_dumped": core_dumped,
    }


def _validate_lifecycle_command(
    raw: Any, label: str
) -> list[str]:
    if (
        not isinstance(raw, list)
        or not raw
        or any(
            not isinstance(item, str)
            or not item
            or "\0" in item
            for item in raw
        )
    ):
        raise PreflightLaunchContractError(f"{label} differs")
    return list(raw)


def _validate_lifecycle_regular_identity(
    raw: Any,
    label: str,
    *,
    executable: bool,
) -> dict[str, Any]:
    value = validate_file_identity(raw, label)
    if (
        not stat.S_ISREG(value["mode"])
        or value["size"] <= 0
        or (executable and value["mode"] & 0o111 == 0)
    ):
        raise PreflightLaunchContractError(f"{label} differs")
    return value


def _validate_sealed_lifecycle_artifact(
    raw: Any,
    *,
    expected_kind: str,
    label: str,
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _SEALED_LIFECYCLE_ARTIFACT_KEYS, label)
    binding = validate_artifact_binding(
        value["binding"], f"{label} binding"
    )
    identity = _validate_lifecycle_regular_identity(
        value["file_identity"],
        f"{label} file identity",
        executable=False,
    )
    if (
        value["kind"] != expected_kind
        or binding["path"] != identity["path"]
    ):
        raise PreflightLaunchContractError(
            f"{label} relation differs"
        )
    return {
        "kind": value["kind"],
        "binding": dict(binding),
        "file_identity": dict(identity),
    }


def _derive_lifecycle_wait_evidence(
    *,
    child_pid: int,
    waitid_si_pid: int,
    waitid_si_code: int,
    waitid_si_status: int,
    waited_pid: int,
    wait_status_raw: int,
) -> dict[str, Any]:
    if (
        any(
            type(item) is not int
            for item in (
                child_pid,
                waitid_si_pid,
                waitid_si_code,
                waitid_si_status,
                waited_pid,
                wait_status_raw,
            )
        )
        or child_pid <= 0
        or waitid_si_pid != child_pid
        or waited_pid != child_pid
        or wait_status_raw < 0
    ):
        raise PreflightLaunchContractError(
            "lifecycle wait child identity differs"
        )
    code_to_name = {
        _LINUX_CLD_EXITED: "exited",
        _LINUX_CLD_KILLED: "killed",
        _LINUX_CLD_DUMPED: "dumped",
    }
    wait_code = code_to_name.get(waitid_si_code)
    if wait_code is None:
        raise PreflightLaunchContractError(
            "lifecycle waitid code differs"
        )
    outcome = derive_lifecycle_wait_outcome(
        wait_code=wait_code,
        wait_status=waitid_si_status,
    )
    raw = _decode_linux_waitpid_status(wait_status_raw)
    if (
        raw["wait_code"] != wait_code
        or raw["wait_status"] != waitid_si_status
        or raw["returncode"] != outcome["returncode"]
        or raw["core_dumped"] != outcome["core_dumped"]
    ):
        raise PreflightLaunchContractError(
            "lifecycle waitid and waitpid evidence differ"
        )
    return {
        "wait_code": wait_code,
        **outcome,
    }


def validate_lifecycle_wait_status(
    value: Mapping[str, Any],
    *,
    role: str,
    label: str,
) -> dict[str, Any]:
    if role not in _LIFECYCLE_ROLE_ARTIFACT_KINDS:
        raise PreflightLaunchContractError(
            f"{label} role differs"
        )
    role_kinds = _LIFECYCLE_ROLE_ARTIFACT_KINDS[role]
    source_artifact = _validate_sealed_lifecycle_artifact(
        value.get("source_artifact"),
        expected_kind=role_kinds["source_artifact"],
        label=f"{label} source artifact",
    )
    worker_started = _validate_sealed_lifecycle_artifact(
        value.get("worker_started"),
        expected_kind=role_kinds["worker_started"],
        label=f"{label} worker started",
    )
    terminal_raw = value.get("terminal")
    terminal = (
        None
        if terminal_raw is None
        else _validate_sealed_lifecycle_artifact(
            terminal_raw,
            expected_kind=role_kinds["terminal"],
            label=f"{label} terminal",
        )
    )
    channel = _mapping(
        value.get("wait_channel"), f"{label} wait channel"
    )
    _exact_keys(
        channel,
        {
            "path",
            "device",
            "inode",
            "mode",
            "uid",
            "nlink",
            "size",
            "sha256",
            "directory_device",
            "directory_inode",
        },
        f"{label} wait channel",
    )
    publisher = _mapping(
        value.get("publisher"), f"{label} publisher"
    )
    _exact_keys(
        publisher,
        _LIFECYCLE_PUBLISHER_KEYS,
        f"{label} publisher",
    )
    publisher_identity = _validate_lifecycle_regular_identity(
        publisher.get("file_identity"),
        f"{label} publisher file identity",
        executable=False,
    )
    supervisor = validate_process_identity(
        value.get("supervisor_process"),
        f"{label} supervisor process",
    )
    child = validate_process_identity(
        value.get("child_process"),
        f"{label} child process",
    )
    executable = validate_file_identity(
        value.get("child_executable"),
        f"{label} child executable",
    )
    supervisor_executable = (
        _validate_lifecycle_regular_identity(
            value.get("supervisor_executable"),
            f"{label} supervisor executable",
            executable=True,
        )
    )
    child_executable = _validate_lifecycle_regular_identity(
        executable,
        f"{label} child executable",
        executable=True,
    )
    owner = _mapping(
        value.get("supervisor_owner_seal"),
        f"{label} supervisor owner seal",
    )
    _exact_keys(
        owner,
        {
            "session",
            "pane",
            "pane_pid",
            "pane_dead",
            "pane_dead_status",
            "pane_process",
            "owner_nonce",
            "tmux_server",
        },
        f"{label} supervisor owner seal",
    )
    owner_process = validate_process_identity(
        owner.get("pane_process"),
        f"{label} supervisor owner process",
    )
    validate_tmux_server_identity(
        owner.get("tmux_server"),
        f"{label} supervisor tmux server",
    )
    supervisor_command = _validate_lifecycle_command(
        value.get("supervisor_command"),
        f"{label} supervisor command",
    )
    child_command = _validate_lifecycle_command(
        value.get("child_command"),
        f"{label} child command",
    )
    try:
        started_text = value.get("started_at")
        completed_text = value.get("completed_at")
        started_at = datetime.fromisoformat(started_text)
        completed_at = datetime.fromisoformat(completed_text)
        timestamps_differ = (
            started_at.tzinfo is None
            or completed_at.tzinfo is None
            or started_at.utcoffset()
            != timezone.utc.utcoffset(None)
            or completed_at.utcoffset()
            != timezone.utc.utcoffset(None)
            or started_at.isoformat() != started_text
            or completed_at.isoformat() != completed_text
            or completed_at < started_at
        )
    except (TypeError, ValueError) as exc:
        raise PreflightLaunchContractError(
            f"{label} timestamps differ"
        ) from exc
    if (
        set(value) != _LIFECYCLE_WAIT_STATUS_KEYS
        or value.get("schema_version") != 2
        or value.get("contract_type")
        != LIFECYCLE_WAIT_STATUS_CONTRACT_TYPE
        or value.get("role") != role
        or source_artifact != value["source_artifact"]
        or worker_started != value["worker_started"]
        or terminal != value["terminal"]
        or (
            not isinstance(channel["path"], str)
            or not channel["path"].startswith("/")
            or any(
                type(channel[field]) is not int
                or int(channel[field]) <= 0
                for field in (
                    "device",
                    "inode",
                    "mode",
                    "nlink",
                    "directory_device",
                    "directory_inode",
                )
            )
            or type(channel["uid"]) is not int
            or channel["uid"] < 0
            or channel["size"] != 0
            or channel["nlink"] != 1
            or channel["sha256"] != hashlib.sha256(b"").hexdigest()
        )
        or (
            not isinstance(publisher["path"], str)
            or not publisher["path"].startswith("/")
            or publisher["path"] != publisher_identity["path"]
            or publisher_identity != publisher["file_identity"]
            or supervisor_command.count(publisher["path"]) != 1
            or (
                role == "gate"
                and (
                    len(supervisor_command) < 5
                    or supervisor_command[3] != publisher["path"]
                    or supervisor_command[4]
                    != "__gate_wait_supervisor__"
                )
            )
            or (
                role == "consumer"
                and (
                    len(supervisor_command) < 5
                    or supervisor_command[3] != publisher["path"]
                    or supervisor_command[4]
                    != "__consumer_wait_supervisor__"
                )
            )
            or publisher["role"]
            != f"{role}_lifecycle_wait_supervisor"
        )
        or _hex64(
            publisher["sha256"], f"{label} publisher SHA"
        )
        != publisher["sha256"]
        or owner.get("pane_dead") is not False
        or owner.get("pane_dead_status") is not None
        or not isinstance(owner.get("session"), str)
        or not owner["session"]
        or not isinstance(owner.get("pane"), str)
        or not owner["pane"]
        or _hex64(
            owner.get("owner_nonce"),
            f"{label} supervisor owner nonce",
        )
        != owner["owner_nonce"]
        or owner_process != supervisor
        or owner.get("pane_pid") != supervisor["pid"]
        or supervisor_executable
        != value["supervisor_executable"]
        or supervisor_command != value["supervisor_command"]
        or child["ppid"] != supervisor["pid"]
        or child["pgid"] != child["pid"]
        or child["sid"] != child["pid"]
        or child_executable != value["child_executable"]
        or child_command != value["child_command"]
        or (
            role == "gate"
            and value.get("exit_kind") == "exit"
            and (
                value.get("returncode") != 117
                or terminal is None
            )
        )
        or (
            role == "consumer"
            and value.get("exit_kind") == "exit"
            and (
                value.get("returncode") != 118
                or terminal is None
            )
        )
        or timestamps_differ
    ):
        raise PreflightLaunchContractError(
            f"{label} lifecycle wait schema differs"
        )
    expected = _derive_lifecycle_wait_evidence(
        child_pid=child["pid"],
        waitid_si_pid=value.get("waitid_si_pid"),
        waitid_si_code=value.get("waitid_si_code"),
        waitid_si_status=value.get("waitid_si_status"),
        waited_pid=value.get("waited_pid"),
        wait_status_raw=value.get("wait_status_raw"),
    )
    if (
        any(
            type(value.get(key)) is not type(item)
            or value.get(key) != item
            for key, item in expected.items()
        )
        or value.get("lifecycle_wait_status_sha256")
        != canonical_digest(
            value, "lifecycle_wait_status_sha256"
        )
    ):
        raise PreflightLaunchContractError(
            f"{label} lifecycle wait outcome differs"
        )
    _hex64(value.get("policy_sha256"), f"{label} policy SHA")
    _hex64(value.get("attempt_id"), f"{label} attempt ID")
    return dict(value)


def build_lifecycle_wait_status(
    *,
    role: str,
    policy_sha256: str,
    attempt_id: str,
    source_artifact: Mapping[str, Any],
    wait_channel: Mapping[str, Any],
    publisher: Mapping[str, Any],
    supervisor_owner_seal: Mapping[str, Any],
    supervisor_process: Mapping[str, Any],
    supervisor_executable: Mapping[str, Any],
    supervisor_command: list[str],
    worker_started: Mapping[str, Any],
    child_process: Mapping[str, Any],
    child_executable: Mapping[str, Any],
    child_command: list[str],
    terminal: Mapping[str, Any] | None,
    waitid_si_pid: int,
    waitid_si_code: int,
    waitid_si_status: int,
    waited_pid: int,
    wait_status_raw: int,
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    wait_evidence = _derive_lifecycle_wait_evidence(
        child_pid=child_process.get("pid"),
        waitid_si_pid=waitid_si_pid,
        waitid_si_code=waitid_si_code,
        waitid_si_status=waitid_si_status,
        waited_pid=waited_pid,
        wait_status_raw=wait_status_raw,
    )
    value = {
        "schema_version": 2,
        "contract_type": LIFECYCLE_WAIT_STATUS_CONTRACT_TYPE,
        "role": role,
        "policy_sha256": policy_sha256,
        "attempt_id": attempt_id,
        "source_artifact": dict(source_artifact),
        "wait_channel": dict(wait_channel),
        "publisher": dict(publisher),
        "supervisor_owner_seal": dict(supervisor_owner_seal),
        "supervisor_process": dict(supervisor_process),
        "supervisor_executable": dict(supervisor_executable),
        "supervisor_command": list(supervisor_command),
        "worker_started": dict(worker_started),
        "child_process": dict(child_process),
        "child_executable": dict(child_executable),
        "child_command": list(child_command),
        "terminal": None if terminal is None else dict(terminal),
        "waitid_si_pid": waitid_si_pid,
        "waitid_si_code": waitid_si_code,
        "waitid_si_status": waitid_si_status,
        "waited_pid": waited_pid,
        "wait_status_raw": wait_status_raw,
        **wait_evidence,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    value["lifecycle_wait_status_sha256"] = canonical_digest(
        value, "lifecycle_wait_status_sha256"
    )
    return validate_lifecycle_wait_status(
        value,
        role=role,
        label=f"{role} lifecycle wait status",
    )


def _validate_lifecycle_channel_v3(
    raw: Any, *, label: str
) -> dict[str, Any]:
    channel = _mapping(raw, label)
    expected = {
        "path",
        "device",
        "inode",
        "mode",
        "uid",
        "nlink",
        "size",
        "sha256",
        "directory_device",
        "directory_inode",
    }
    _exact_keys(channel, expected, label)
    if (
        not isinstance(channel["path"], str)
        or not channel["path"].startswith("/")
        or any(
            type(channel[field]) is not int or channel[field] <= 0
            for field in expected
            - {"path", "uid", "size", "sha256"}
        )
        or type(channel["uid"]) is not int
        or channel["uid"] < 0
        or channel["size"] != 0
        or channel["nlink"] != 1
        or channel["sha256"] != hashlib.sha256(b"").hexdigest()
    ):
        raise PreflightLaunchContractError(f"{label} differs")
    return dict(channel)


def _validate_lifecycle_publisher_v3(
    raw: Any, *, role: str, label: str
) -> dict[str, Any]:
    publisher = _mapping(raw, label)
    _exact_keys(publisher, _LIFECYCLE_PUBLISHER_KEYS, label)
    identity = _validate_lifecycle_regular_identity(
        publisher.get("file_identity"),
        f"{label} file identity",
        executable=False,
    )
    if (
        not isinstance(publisher.get("path"), str)
        or not publisher["path"].startswith("/")
        or publisher["path"] != identity["path"]
        or publisher.get("role")
        != f"{role}_lifecycle_wait_supervisor"
    ):
        raise PreflightLaunchContractError(f"{label} differs")
    _hex64(publisher.get("sha256"), f"{label} SHA")
    return {**dict(publisher), "file_identity": identity}


def _validate_lifecycle_runtime_owner_v3(
    raw: Any, *, label: str
) -> dict[str, Any]:
    owner = _mapping(raw, label)
    _exact_keys(
        owner,
        {
            "session",
            "pane",
            "pane_pid",
            "pane_dead",
            "pane_dead_status",
            "pane_process",
            "owner_nonce",
            "tmux_server",
        },
        label,
    )
    process = validate_process_identity(
        owner.get("pane_process"), f"{label} process"
    )
    server = validate_tmux_server_identity(
        owner.get("tmux_server"), f"{label} tmux server"
    )
    if (
        owner.get("pane_dead") is not False
        or owner.get("pane_dead_status") is not None
        or not isinstance(owner.get("session"), str)
        or not owner["session"]
        or not isinstance(owner.get("pane"), str)
        or not owner["pane"]
        or owner.get("pane_pid") != process["pid"]
    ):
        raise PreflightLaunchContractError(f"{label} differs")
    _hex64(owner.get("owner_nonce"), f"{label} owner nonce")
    return {
        **dict(owner),
        "pane_process": process,
        "tmux_server": server,
    }


def _validate_ordered_utc_times(
    *, started_at: Any, completed_at: Any, label: str
) -> None:
    try:
        started = datetime.fromisoformat(started_at)
        completed = datetime.fromisoformat(completed_at)
        invalid = (
            started.tzinfo is None
            or completed.tzinfo is None
            or started.utcoffset() != timezone.utc.utcoffset(None)
            or completed.utcoffset() != timezone.utc.utcoffset(None)
            or started.isoformat() != started_at
            or completed.isoformat() != completed_at
            or completed < started
        )
    except (TypeError, ValueError) as exc:
        raise PreflightLaunchContractError(
            f"{label} timestamps differ"
        ) from exc
    if invalid:
        raise PreflightLaunchContractError(
            f"{label} timestamps differ"
        )


def validate_lifecycle_raw_wait_v3(
    value: Mapping[str, Any],
    *,
    role: str,
    label: str,
) -> dict[str, Any]:
    if role not in _LIFECYCLE_ROLE_ARTIFACT_KINDS:
        raise PreflightLaunchContractError(f"{label} role differs")
    source = _validate_sealed_lifecycle_artifact(
        value.get("source_artifact"),
        expected_kind=_LIFECYCLE_ROLE_ARTIFACT_KINDS[role][
            "source_artifact"
        ],
        label=f"{label} source artifact",
    )
    channel = _validate_lifecycle_channel_v3(
        value.get("wait_channel"), label=f"{label} wait channel"
    )
    publisher = _validate_lifecycle_publisher_v3(
        value.get("publisher"),
        role=role,
        label=f"{label} publisher",
    )
    owner = _validate_lifecycle_runtime_owner_v3(
        value.get("supervisor_owner_seal"),
        label=f"{label} supervisor owner",
    )
    child = validate_process_identity(
        value.get("child_process"), f"{label} child process"
    )
    _validate_ordered_utc_times(
        started_at=value.get("started_at"),
        completed_at=value.get("reaped_at"),
        label=label,
    )
    if (
        set(value) != _LIFECYCLE_RAW_WAIT_V3_KEYS
        or value.get("schema_version") != 3
        or value.get("contract_type")
        != LIFECYCLE_RAW_WAIT_V3_CONTRACT_TYPE
        or value.get("role") != role
        or source != value.get("source_artifact")
        or channel != value.get("wait_channel")
        or publisher != value.get("publisher")
        or owner != value.get("supervisor_owner_seal")
        or child["ppid"] != owner["pane_pid"]
        or child["pgid"] != child["pid"]
        or child["sid"] != child["pid"]
    ):
        raise PreflightLaunchContractError(
            f"{label} lifecycle raw wait schema differs"
        )
    expected = _derive_lifecycle_wait_evidence(
        child_pid=child["pid"],
        waitid_si_pid=value.get("waitid_si_pid"),
        waitid_si_code=value.get("waitid_si_code"),
        waitid_si_status=value.get("waitid_si_status"),
        waited_pid=value.get("waited_pid"),
        wait_status_raw=value.get("wait_status_raw"),
    )
    if (
        any(
            type(value.get(key)) is not type(expected_value)
            or value.get(key) != expected_value
            for key, expected_value in expected.items()
        )
        or value.get("lifecycle_raw_wait_sha256")
        != canonical_digest(value, "lifecycle_raw_wait_sha256")
    ):
        raise PreflightLaunchContractError(
            f"{label} lifecycle raw wait outcome differs"
        )
    _hex64(value.get("policy_sha256"), f"{label} policy SHA")
    _hex64(value.get("attempt_id"), f"{label} attempt ID")
    return dict(value)


def build_lifecycle_raw_wait_v3(
    *,
    role: str,
    policy_sha256: str,
    attempt_id: str,
    source_artifact: Mapping[str, Any],
    wait_channel: Mapping[str, Any],
    publisher: Mapping[str, Any],
    supervisor_owner_seal: Mapping[str, Any],
    child_process: Mapping[str, Any],
    waitid_si_pid: int,
    waitid_si_code: int,
    waitid_si_status: int,
    waited_pid: int,
    wait_status_raw: int,
    started_at: str,
    reaped_at: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 3,
        "contract_type": LIFECYCLE_RAW_WAIT_V3_CONTRACT_TYPE,
        "role": role,
        "policy_sha256": policy_sha256,
        "attempt_id": attempt_id,
        "source_artifact": dict(source_artifact),
        "wait_channel": dict(wait_channel),
        "publisher": dict(publisher),
        "supervisor_owner_seal": dict(supervisor_owner_seal),
        "child_process": dict(child_process),
        "waitid_si_pid": waitid_si_pid,
        "waitid_si_code": waitid_si_code,
        "waitid_si_status": waitid_si_status,
        "waited_pid": waited_pid,
        "wait_status_raw": wait_status_raw,
        **_derive_lifecycle_wait_evidence(
            child_pid=child_process.get("pid"),
            waitid_si_pid=waitid_si_pid,
            waitid_si_code=waitid_si_code,
            waitid_si_status=waitid_si_status,
            waited_pid=waited_pid,
            wait_status_raw=wait_status_raw,
        ),
        "started_at": started_at,
        "reaped_at": reaped_at,
    }
    value["lifecycle_raw_wait_sha256"] = canonical_digest(
        value, "lifecycle_raw_wait_sha256"
    )
    return validate_lifecycle_raw_wait_v3(
        value, role=role, label=f"{role} lifecycle raw wait"
    )


def validate_lifecycle_raw_wait_publish_failure_v1(
    value: Mapping[str, Any],
    *,
    role: str,
    label: str,
) -> dict[str, Any]:
    if role not in _LIFECYCLE_ROLE_ARTIFACT_KINDS:
        raise PreflightLaunchContractError(f"{label} role differs")
    source = _validate_sealed_lifecycle_artifact(
        value.get("source_artifact"),
        expected_kind=_LIFECYCLE_ROLE_ARTIFACT_KINDS[role][
            "source_artifact"
        ],
        label=f"{label} source artifact",
    )
    target = _validate_lifecycle_channel_v3(
        value.get("target_channel"),
        label=f"{label} target channel",
    )
    fault_channel = _validate_lifecycle_channel_v3(
        value.get("fault_channel"),
        label=f"{label} fault channel",
    )
    publisher = _validate_lifecycle_publisher_v3(
        value.get("publisher"),
        role=role,
        label=f"{label} publisher",
    )
    owner = _validate_lifecycle_runtime_owner_v3(
        value.get("supervisor_owner_seal"),
        label=f"{label} supervisor owner",
    )
    child = validate_process_identity(
        value.get("child_process"), f"{label} child process"
    )
    failure = validate_publish_failure_record(
        _mapping(
            value.get("publish_failure"),
            f"{label} publish failure",
        )
    )
    _validate_ordered_utc_times(
        started_at=value.get("recorded_at"),
        completed_at=value.get("recorded_at"),
        label=label,
    )
    if (
        set(value)
        != _LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_KEYS
        or value.get("schema_version") != 1
        or value.get("contract_type")
        != LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_CONTRACT_TYPE
        or value.get("role") != role
        or source != value.get("source_artifact")
        or target != value.get("target_channel")
        or fault_channel != value.get("fault_channel")
        or publisher != value.get("publisher")
        or owner != value.get("supervisor_owner_seal")
        or child["ppid"] != owner["pane_pid"]
        or child["pgid"] != child["pid"]
        or child["sid"] != child["pid"]
        or target["path"] == fault_channel["path"]
        or (target["device"], target["inode"])
        == (fault_channel["device"], fault_channel["inode"])
        or failure != value.get("publish_failure")
    ):
        raise PreflightLaunchContractError(f"{label} schema differs")
    _hex64(value.get("policy_sha256"), f"{label} policy SHA")
    _hex64(value.get("attempt_id"), f"{label} attempt ID")
    _hex64(
        value.get("intended_raw_wait_sha256"),
        f"{label} intended raw wait SHA",
    )
    if value.get(
        "lifecycle_raw_wait_publish_failure_sha256"
    ) != canonical_digest(
        value,
        "lifecycle_raw_wait_publish_failure_sha256",
    ):
        raise PreflightLaunchContractError(f"{label} digest differs")
    return dict(value)


def build_lifecycle_raw_wait_publish_failure_v1(
    *,
    role: str,
    policy_sha256: str,
    attempt_id: str,
    source_artifact: Mapping[str, Any],
    target_channel: Mapping[str, Any],
    fault_channel: Mapping[str, Any],
    publisher: Mapping[str, Any],
    supervisor_owner_seal: Mapping[str, Any],
    child_process: Mapping[str, Any],
    intended_raw_wait_sha256: str,
    publish_failure: Mapping[str, Any],
    recorded_at: str,
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": (
            LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_CONTRACT_TYPE
        ),
        "role": role,
        "policy_sha256": policy_sha256,
        "attempt_id": attempt_id,
        "source_artifact": dict(source_artifact),
        "target_channel": dict(target_channel),
        "fault_channel": dict(fault_channel),
        "publisher": dict(publisher),
        "supervisor_owner_seal": dict(supervisor_owner_seal),
        "child_process": dict(child_process),
        "intended_raw_wait_sha256": intended_raw_wait_sha256,
        "publish_failure": dict(publish_failure),
        "recorded_at": recorded_at,
    }
    value["lifecycle_raw_wait_publish_failure_sha256"] = (
        canonical_digest(
            value,
            "lifecycle_raw_wait_publish_failure_sha256",
        )
    )
    return validate_lifecycle_raw_wait_publish_failure_v1(
        value,
        role=role,
        label=f"{role} lifecycle raw wait publish failure",
    )


def validate_postclaim_finalization_profile_v1(
    raw: Any,
    *,
    label: str = "postclaim finalization profile",
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(
        value, _POSTCLAIM_FINALIZATION_PROFILE_V1_KEYS, label
    )
    components = {
        "raw_wait": (
            3,
            LIFECYCLE_RAW_WAIT_V3_CONTRACT_TYPE,
        ),
        "raw_wait_publish_failure": (
            1,
            LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_CONTRACT_TYPE,
        ),
        "controller_cleanup": (
            3,
            PANE_FAULT_CONSUMER_CONTROLLER_CLEANUP_V3_CONTRACT_TYPE,
        ),
        "consumer_terminal": (
            3,
            PANE_FAULT_CONSUMER_TERMINAL_V3_CONTRACT_TYPE,
        ),
        "consumer_join": (
            4,
            PANE_FAULT_CONSUMER_JOIN_V4_CONTRACT_TYPE,
        ),
    }
    for name, (version, contract_type) in components.items():
        component = _mapping(value.get(name), f"{label} {name}")
        _exact_keys(
            component,
            _POSTCLAIM_PROFILE_COMPONENT_KEYS,
            f"{label} {name}",
        )
        if component != {
            "schema_version": version,
            "contract_type": contract_type,
        }:
            raise PreflightLaunchContractError(
                f"{label} {name} differs"
            )
    if (
        value.get("schema_version") != 1
        or value.get("contract_type")
        != POSTCLAIM_FINALIZATION_PROFILE_V1_CONTRACT_TYPE
        or value.get("mixed_stack_allowed") is not False
        or value.get("reap_to_raw_crash_policy")
        != "fail_closed_unrecoverable"
        or value.get("post_raw_resume_policy") != "missing_only"
        or value.get("postclaim_finalization_profile_sha256")
        != canonical_digest(
            value, "postclaim_finalization_profile_sha256"
        )
    ):
        raise PreflightLaunchContractError(f"{label} differs")
    return dict(value)


def build_postclaim_finalization_profile_v1() -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "contract_type": (
            POSTCLAIM_FINALIZATION_PROFILE_V1_CONTRACT_TYPE
        ),
        "raw_wait": {
            "schema_version": 3,
            "contract_type": LIFECYCLE_RAW_WAIT_V3_CONTRACT_TYPE,
        },
        "raw_wait_publish_failure": {
            "schema_version": 1,
            "contract_type": (
                LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_CONTRACT_TYPE
            ),
        },
        "controller_cleanup": {
            "schema_version": 3,
            "contract_type": (
                PANE_FAULT_CONSUMER_CONTROLLER_CLEANUP_V3_CONTRACT_TYPE
            ),
        },
        "consumer_terminal": {
            "schema_version": 3,
            "contract_type": (
                PANE_FAULT_CONSUMER_TERMINAL_V3_CONTRACT_TYPE
            ),
        },
        "consumer_join": {
            "schema_version": 4,
            "contract_type": PANE_FAULT_CONSUMER_JOIN_V4_CONTRACT_TYPE,
        },
        "mixed_stack_allowed": False,
        "reap_to_raw_crash_policy": "fail_closed_unrecoverable",
        "post_raw_resume_policy": "missing_only",
    }
    value["postclaim_finalization_profile_sha256"] = (
        canonical_digest(
            value, "postclaim_finalization_profile_sha256"
        )
    )
    return validate_postclaim_finalization_profile_v1(value)


def validate_launch_receipt_v5(
    raw: Any,
    *,
    expected_gate_worker_arguments: Sequence[str],
    expected_consumer_worker_arguments: Sequence[str],
    label: str = "launch receipt v5",
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _LAUNCH_RECEIPT_V5_KEYS, label)
    if (
        value.get("schema_version") != 5
        or value.get("contract_type")
        != LAUNCH_RECEIPT_V5_CONTRACT_TYPE
        or value.get("launch_receipt_sha256")
        != canonical_digest(value, "launch_receipt_sha256")
    ):
        raise PreflightLaunchContractError(
            f"{label} discriminator or digest differs"
        )
    profile = validate_postclaim_finalization_profile_v1(
        value.get("postclaim_finalization_profile"),
        label=f"{label} postclaim finalization profile",
    )
    if profile != build_postclaim_finalization_profile_v1():
        raise PreflightLaunchContractError(
            f"{label} postclaim finalization stack differs"
        )
    fault_channel = _validate_lifecycle_channel_v3(
        value.get("gate_lifecycle_wait_publish_fault_channel"),
        label=f"{label} gate raw wait publish fault channel",
    )
    fault_publisher = _validate_lifecycle_publisher_v3(
        value.get("gate_lifecycle_wait_publish_fault_publisher"),
        role="gate",
        label=f"{label} gate raw wait publish fault publisher",
    )
    wait_channel = _validate_lifecycle_channel_v3(
        value.get("gate_lifecycle_wait_channel"),
        label=f"{label} gate raw wait channel",
    )
    expected_fault_path = (
        wait_channel["path"].rsplit("/", 1)[0]
        + "/gate_lifecycle_wait_publish_fault.channel"
    )
    if (
        fault_channel["path"] != expected_fault_path
        or fault_channel["path"] == wait_channel["path"]
        or (fault_channel["device"], fault_channel["inode"])
        == (wait_channel["device"], wait_channel["inode"])
        or fault_channel["directory_device"]
        != wait_channel["directory_device"]
        or fault_channel["directory_inode"]
        != wait_channel["directory_inode"]
        or fault_publisher
        != value.get("gate_lifecycle_wait_publisher")
    ):
        raise PreflightLaunchContractError(
            f"{label} gate raw wait publish fault authority differs"
        )
    verified = validate_verified_implementations(
        value.get("verified_implementations"),
        f"{label} verified implementations",
    )
    if verified != value.get("verified_implementations"):
        raise PreflightLaunchContractError(
            f"{label} verified implementations differ"
        )
    legacy = {
        key: value[key] for key in _LAUNCH_RECEIPT_KEYS
    }
    legacy["schema_version"] = 4
    legacy["contract_type"] = LAUNCH_RECEIPT_CONTRACT_TYPE
    legacy["launch_receipt_sha256"] = canonical_digest(
        legacy, "launch_receipt_sha256"
    )
    validate_launch_receipt_schema(
        legacy,
        expected_gate_worker_arguments=(
            expected_gate_worker_arguments
        ),
        expected_consumer_worker_arguments=(
            expected_consumer_worker_arguments
        ),
        label=f"{label} shared v4 fields",
    )
    _hex64(value.get("attempt_id"), f"{label} attempt ID")
    return dict(value)


def build_launch_receipt_v5(
    *,
    launch_receipt_v4: Mapping[str, Any],
    gate_lifecycle_wait_publish_fault_channel: Mapping[str, Any],
    gate_lifecycle_wait_publish_fault_publisher: Mapping[str, Any],
    expected_gate_worker_arguments: Sequence[str],
    expected_consumer_worker_arguments: Sequence[str],
) -> dict[str, Any]:
    legacy = validate_launch_receipt_schema(
        launch_receipt_v4,
        expected_gate_worker_arguments=(
            expected_gate_worker_arguments
        ),
        expected_consumer_worker_arguments=(
            expected_consumer_worker_arguments
        ),
        label="launch receipt v5 source v4",
    )
    value = {
        **legacy,
        "schema_version": 5,
        "contract_type": LAUNCH_RECEIPT_V5_CONTRACT_TYPE,
        "postclaim_finalization_profile": (
            build_postclaim_finalization_profile_v1()
        ),
        "gate_lifecycle_wait_publish_fault_channel": dict(
            gate_lifecycle_wait_publish_fault_channel
        ),
        "gate_lifecycle_wait_publish_fault_publisher": dict(
            gate_lifecycle_wait_publish_fault_publisher
        ),
    }
    value["launch_receipt_sha256"] = canonical_digest(
        value, "launch_receipt_sha256"
    )
    return validate_launch_receipt_v5(
        value,
        expected_gate_worker_arguments=(
            expected_gate_worker_arguments
        ),
        expected_consumer_worker_arguments=(
            expected_consumer_worker_arguments
        ),
    )


def validate_launch_receipt_schema(
    raw: Any,
    *,
    expected_gate_worker_arguments: Sequence[str],
    expected_consumer_worker_arguments: Sequence[str],
    label: str = "launch receipt",
) -> dict[str, Any]:
    value = _mapping(raw, label)
    _exact_keys(value, _LAUNCH_RECEIPT_KEYS, label)
    wait_channel = _mapping(
        value["gate_lifecycle_wait_channel"],
        f"{label} gate lifecycle wait channel",
    )
    _exact_keys(
        wait_channel,
        {
            "path",
            "device",
            "inode",
            "mode",
            "uid",
            "nlink",
            "size",
            "sha256",
            "directory_device",
            "directory_inode",
        },
        f"{label} gate lifecycle wait channel",
    )
    publisher = _mapping(
        value["gate_lifecycle_wait_publisher"],
        f"{label} gate lifecycle wait publisher",
    )
    _exact_keys(
        publisher,
        _LIFECYCLE_PUBLISHER_KEYS,
        f"{label} gate lifecycle wait publisher",
    )
    publisher_identity = _validate_lifecycle_regular_identity(
        publisher["file_identity"],
        f"{label} gate lifecycle wait publisher identity",
        executable=False,
    )
    supervisor_arguments = _validate_lifecycle_command(
        value["gate_lifecycle_wait_supervisor_arguments"],
        f"{label} gate lifecycle wait supervisor arguments",
    )
    gate_worker_arguments = _validate_lifecycle_command(
        value["gate_worker_arguments"],
        f"{label} gate worker arguments",
    )
    independently_expected_worker_arguments = (
        _validate_lifecycle_command(
            expected_gate_worker_arguments,
            f"{label} independently expected gate worker arguments",
        )
    )
    wrapper_arguments = _validate_lifecycle_command(
        value["wrapper_arguments"],
        f"{label} wrapper arguments",
    )
    consumer_wait_channel = _mapping(
        value["consumer_lifecycle_wait_channel"],
        f"{label} consumer lifecycle wait channel",
    )
    _exact_keys(
        consumer_wait_channel,
        {
            "path",
            "device",
            "inode",
            "mode",
            "uid",
            "nlink",
            "size",
            "sha256",
            "directory_device",
            "directory_inode",
        },
        f"{label} consumer lifecycle wait channel",
    )
    consumer_publisher = _mapping(
        value["consumer_lifecycle_wait_publisher"],
        f"{label} consumer lifecycle wait publisher",
    )
    _exact_keys(
        consumer_publisher,
        _LIFECYCLE_PUBLISHER_KEYS,
        f"{label} consumer lifecycle wait publisher",
    )
    consumer_publisher_identity = (
        _validate_lifecycle_regular_identity(
            consumer_publisher["file_identity"],
            f"{label} consumer lifecycle wait publisher identity",
            executable=False,
        )
    )
    consumer_supervisor_arguments = _validate_lifecycle_command(
        value["consumer_lifecycle_wait_supervisor_arguments"],
        f"{label} consumer lifecycle wait supervisor arguments",
    )
    consumer_worker_arguments = _validate_lifecycle_command(
        value["consumer_worker_arguments"],
        f"{label} consumer worker arguments",
    )
    independently_expected_consumer_worker_arguments = (
        _validate_lifecycle_command(
            expected_consumer_worker_arguments,
            f"{label} independently expected consumer worker arguments",
        )
    )
    wait_path = wait_channel.get("path")
    if not isinstance(wait_path, str) or not wait_path.startswith("/"):
        raise PreflightLaunchContractError(
            f"{label} gate lifecycle wait channel path differs"
        )
    attempt_root = wait_path.rsplit("/", 1)[0]
    receipt_path = f"{attempt_root}/launch_receipt.json"
    consumer_wait_path = consumer_wait_channel.get("path")
    consumer_registration = _mapping(
        value["pane_fault_consumer"],
        f"{label} pane fault consumer registration",
    )
    consumer_artifacts = _mapping(
        consumer_registration.get("artifacts"),
        f"{label} pane fault consumer artifacts",
    )
    expected_consumer_supervisor_prefix = [
        consumer_supervisor_arguments[0],
        "-B",
        "-u",
        consumer_publisher.get("path"),
        "__consumer_wait_supervisor__",
        "--attempt-path",
        consumer_artifacts.get("attempt"),
        "--config",
        value["bindings"]["config"]["path"],
        "--wait-channel-path",
        consumer_wait_path,
        "--consumer-worker-arguments-json",
    ]
    expected_supervisor_prefix = [
        supervisor_arguments[0],
        "-B",
        "-u",
        publisher.get("path"),
        "__gate_wait_supervisor__",
        "--launch-receipt",
        receipt_path,
        "--attempt-id",
        value["attempt_id"],
        "--wait-channel-path",
        wait_path,
        "--gate-worker-arguments-json",
    ]
    if (
        value["schema_version"] != 4
        or value["contract_type"] != LAUNCH_RECEIPT_CONTRACT_TYPE
        or value["launch_receipt_sha256"]
        != canonical_digest(value, "launch_receipt_sha256")
        or any(
            type(wait_channel[field]) is not int
            or wait_channel[field] <= 0
            for field in (
                "device",
                "inode",
                "mode",
                "nlink",
                "directory_device",
                "directory_inode",
            )
        )
        or type(wait_channel["uid"]) is not int
        or wait_channel["uid"] < 0
        or wait_channel["nlink"] != 1
        or wait_channel["size"] != 0
        or wait_channel["sha256"] != hashlib.sha256(b"").hexdigest()
        or publisher.get("path") != publisher_identity["path"]
        or publisher.get("file_identity") != publisher_identity
        or publisher.get("role")
        != "gate_lifecycle_wait_supervisor"
        or _hex64(
            publisher.get("sha256"),
            f"{label} gate lifecycle wait publisher SHA",
        )
        != publisher["sha256"]
        or len(supervisor_arguments) != 13
        or supervisor_arguments[:12] != expected_supervisor_prefix
        or supervisor_arguments[12]
        != json.dumps(
            independently_expected_worker_arguments,
            separators=(",", ":"),
        )
        or len(gate_worker_arguments) != 13
        or gate_worker_arguments
        != independently_expected_worker_arguments
        or len(independently_expected_worker_arguments) != 13
        or independently_expected_worker_arguments[1:5]
        != ["-B", "-u", publisher.get("path"), "__pane_gate__"]
        or independently_expected_worker_arguments[5:7]
        != ["--attempt-root", attempt_root]
        or independently_expected_worker_arguments[7:9]
        != [
            "--release-path",
            f"{attempt_root}/pane_gate_release.json",
        ]
        or independently_expected_worker_arguments[9:11]
        != ["--log-path", f"{attempt_root}/pane.log"]
        or independently_expected_worker_arguments[11]
        != "--wrapper-arguments-json"
        or independently_expected_worker_arguments[12]
        != json.dumps(wrapper_arguments, separators=(",", ":"))
        or supervisor_arguments[0] != gate_worker_arguments[0]
        or wait_path in gate_worker_arguments
        or len(value["tmux_arguments"]) < len(supervisor_arguments)
        or value["tmux_arguments"][-len(supervisor_arguments) :]
        != supervisor_arguments
        or value["gate_lifecycle_wait_supervisor_ready_path"]
        != f"{attempt_root}/gate_wait_supervisor_ready.json"
        or value["gate_lifecycle_wait_status_path"]
        != wait_path
        or not isinstance(consumer_wait_path, str)
        or consumer_wait_path
        != consumer_artifacts.get("lifecycle_wait_channel")
        or any(
            type(consumer_wait_channel[field]) is not int
            or consumer_wait_channel[field] <= 0
            for field in (
                "device",
                "inode",
                "mode",
                "nlink",
                "directory_device",
                "directory_inode",
            )
        )
        or type(consumer_wait_channel["uid"]) is not int
        or consumer_wait_channel["uid"] < 0
        or consumer_wait_channel["nlink"] != 1
        or consumer_wait_channel["size"] != 0
        or consumer_wait_channel["sha256"]
        != hashlib.sha256(b"").hexdigest()
        or consumer_publisher.get("path")
        != consumer_publisher_identity["path"]
        or consumer_publisher.get("file_identity")
        != consumer_publisher_identity
        or consumer_publisher.get("role")
        != "consumer_lifecycle_wait_supervisor"
        or _hex64(
            consumer_publisher.get("sha256"),
            f"{label} consumer lifecycle wait publisher SHA",
        )
        != consumer_publisher["sha256"]
        or len(consumer_worker_arguments) != 9
        or consumer_worker_arguments
        != independently_expected_consumer_worker_arguments
        or consumer_worker_arguments[1:5]
        != [
            "-B",
            "-u",
            consumer_publisher.get("path"),
            "__pane_fault_consumer__",
        ]
        or consumer_worker_arguments[5:7]
        != ["--attempt-path", consumer_artifacts.get("attempt")]
        or consumer_worker_arguments[7:9]
        != ["--config", value["bindings"]["config"]["path"]]
        or len(consumer_supervisor_arguments) != 13
        or consumer_supervisor_arguments[:12]
        != expected_consumer_supervisor_prefix
        or consumer_supervisor_arguments[12]
        != json.dumps(
            independently_expected_consumer_worker_arguments,
            separators=(",", ":"),
        )
        or consumer_supervisor_arguments[0]
        != consumer_worker_arguments[0]
        or consumer_wait_path in consumer_worker_arguments
        or value["consumer_lifecycle_wait_supervisor_ready_path"]
        != consumer_artifacts.get("wait_supervisor_ready")
        or value["consumer_lifecycle_wait_status_path"]
        != consumer_wait_path
        or value["consumer_session"]
        != f"safa-pane-fault-consumer-{value['attempt_id']}"
        or _hex64(
            value["consumer_owner_nonce"],
            f"{label} consumer owner nonce",
        )
        != value["consumer_owner_nonce"]
        or not isinstance(value["consumer_tmux_arguments"], list)
        or len(value["consumer_tmux_arguments"])
        < len(consumer_supervisor_arguments)
        or value["consumer_tmux_arguments"][
            -len(consumer_supervisor_arguments) :
        ]
        != consumer_supervisor_arguments
    ):
        raise PreflightLaunchContractError(
            f"{label} schema or digest differs"
        )
    _hex64(value["attempt_id"], f"{label} attempt ID")
    _hex64(value["policy_sha256"], f"{label} policy SHA")
    validate_pane_fault_consumer_registration(
        value["pane_fault_consumer"],
        label=f"{label} pane fault consumer",
    )
    return value


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


def build_sealed_lifecycle_artifact(
    *,
    kind: str,
    binding: Mapping[str, Any],
    file_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(kind, str) or not kind:
        raise PreflightLaunchContractError(
            "sealed lifecycle artifact kind differs"
        )
    validated_binding = _mapping(
        binding, "sealed lifecycle artifact binding"
    )
    _exact_keys(
        validated_binding,
        {"path", "sha256", "canonical_sha256"},
        "sealed lifecycle artifact binding",
    )
    if (
        not isinstance(validated_binding["path"], str)
        or not validated_binding["path"].startswith("/")
        or _hex64(
            validated_binding["sha256"],
            "sealed lifecycle artifact content SHA",
        )
        != validated_binding["sha256"]
        or _hex64(
            validated_binding["canonical_sha256"],
            "sealed lifecycle artifact canonical SHA",
        )
        != validated_binding["canonical_sha256"]
    ):
        raise PreflightLaunchContractError(
            "sealed lifecycle artifact binding differs"
        )
    identity = validate_file_identity(
        file_identity, "sealed lifecycle artifact file identity"
    )
    if identity["path"] != validated_binding["path"]:
        raise PreflightLaunchContractError(
            "sealed lifecycle artifact paths differ"
        )
    return dict(
        kind=kind,
        binding=dict(validated_binding),
        file_identity=identity,
    )


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
    pane_fault_consumer_chain: Mapping[str, Any] | None = None,
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
    normalized_consumer_chain = validate_pane_fault_consumer_chain(
        value["pane_fault_consumer_chain"],
        label=f"{label} pane fault consumer chain",
    )
    if (
        pane_fault_consumer_chain is not None
        and normalized_consumer_chain
        != validate_pane_fault_consumer_chain(
            pane_fault_consumer_chain,
            label=f"{label} expected pane fault consumer chain",
        )
    ):
        raise PreflightLaunchContractError(
            f"{label} pane fault consumer reference differs"
        )
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
    pane_fault_consumer_chain: Mapping[str, Any],
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
        "pane_fault_consumer_chain": dict(
            pane_fault_consumer_chain
        ),
    }
    value["wrapper_claim_sha256"] = canonical_digest(
        value, "wrapper_claim_sha256"
    )
    return validate_claim_v3(
        value,
        verified_implementations=verified_implementations,
        gate_ready=gate_ready,
        wrapper_started=wrapper_started,
        pane_fault_consumer_chain=pane_fault_consumer_chain,
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
    pane_fault_consumer_chain: Mapping[str, Any],
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
    normalized_consumer_chain = (
        validate_pane_fault_consumer_chain(
            pane_fault_consumer_chain,
            label=f"{label} expected pane fault consumer chain",
        )
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
        or accepted["pane_fault_consumer_chain"]
        != normalized_consumer_chain
        or terminal["pane_fault_consumer_chain"]
        != normalized_consumer_chain
        or release["pane_fault_consumer_chain"]
        != normalized_consumer_chain
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
    pane_fault_consumer_chain: Mapping[str, Any],
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
        "pane_fault_consumer_chain": dict(
            pane_fault_consumer_chain
        ),
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
    pane_fault_consumer_chain: Mapping[str, Any],
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
        "pane_fault_consumer_chain": dict(
            pane_fault_consumer_chain
        ),
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
    pane_fault_consumer_chain: Mapping[str, Any],
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
        "pane_fault_consumer_chain": dict(
            pane_fault_consumer_chain
        ),
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
        "pane_fault_consumer_registration": frozenset(
            _PANE_FAULT_CONSUMER_REGISTRATION_KEYS
        ),
        "pane_fault_consumer_chain": frozenset(
            _PANE_FAULT_CONSUMER_CHAIN_KEYS
        ),
        "deadline_observation": frozenset(
            _DEADLINE_OBSERVATION_KEYS
        ),
        "invalid_claim_evidence": frozenset(
            _INVALID_CLAIM_EVIDENCE_KEYS
        ),
        "preclaim_failure_intent": frozenset(
            _PRECLAIM_FAILURE_INTENT_KEYS
        ),
        "bound_lifecycle_evidence": frozenset(
            _BOUND_LIFECYCLE_EVIDENCE_KEYS
        ),
        "ownership_absent": frozenset(
            _OWNERSHIP_ABSENT_KEYS
        ),
        "terminal_failure": frozenset(
            _TERMINAL_FAILURE_KEYS
        ),
        "launch_terminal_v2": frozenset(
            _LAUNCH_TERMINAL_V2_KEYS
        ),
        "attempted_terminal_payload": frozenset(
            _ATTEMPTED_TERMINAL_PAYLOAD_KEYS
        ),
        "finalization_secondary_failure": frozenset(
            _FINALIZATION_SECONDARY_FAILURE_KEYS
        ),
        "launcher_terminal_publish_error": frozenset(
            _LAUNCHER_TERMINAL_PUBLISH_ERROR_KEYS
        ),
        "finalization_inner_failure": frozenset(
            _FINALIZATION_INNER_FAILURE_KEYS
        ),
        "post_handoff_failure": frozenset(
            _POST_HANDOFF_FAILURE_KEYS
        ),
        "post_handoff_finalization_failure": frozenset(
            _POST_HANDOFF_FINALIZATION_FAILURE_KEYS
        ),
        "sealed_lifecycle_artifact": frozenset(
            _SEALED_LIFECYCLE_ARTIFACT_KEYS
        ),
        "lifecycle_publisher": frozenset(
            _LIFECYCLE_PUBLISHER_KEYS
        ),
        "lifecycle_wait_status": frozenset(
            _LIFECYCLE_WAIT_STATUS_KEYS
        ),
        "lifecycle_raw_wait_v3": frozenset(
            _LIFECYCLE_RAW_WAIT_V3_KEYS
        ),
        "lifecycle_raw_wait_publish_failure_v1": frozenset(
            _LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_KEYS
        ),
        "postclaim_finalization_profile_v1": frozenset(
            _POSTCLAIM_FINALIZATION_PROFILE_V1_KEYS
        ),
        "launch_receipt": frozenset(_LAUNCH_RECEIPT_KEYS),
        "launch_receipt_v5": frozenset(_LAUNCH_RECEIPT_V5_KEYS),
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
            LAUNCH_RECEIPT_CONTRACT_TYPE,
            LAUNCH_RECEIPT_V5_CONTRACT_TYPE,
            PANE_FAULT_CONSUMER_REGISTRATION_CONTRACT_TYPE,
            LIFECYCLE_WAIT_STATUS_CONTRACT_TYPE,
            LIFECYCLE_RAW_WAIT_V3_CONTRACT_TYPE,
            LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_CONTRACT_TYPE,
            POSTCLAIM_FINALIZATION_PROFILE_V1_CONTRACT_TYPE,
            PANE_FAULT_CONSUMER_CONTROLLER_CLEANUP_V3_CONTRACT_TYPE,
            PANE_FAULT_CONSUMER_TERMINAL_V3_CONTRACT_TYPE,
            PANE_FAULT_CONSUMER_JOIN_V4_CONTRACT_TYPE,
            PRECLAIM_FAILURE_INTENT_CONTRACT_TYPE,
            LAUNCH_TERMINAL_V2_CONTRACT_TYPE,
            POST_HANDOFF_FINALIZATION_FAILURE_CONTRACT_TYPE,
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
        | _PANE_FAULT_CONSUMER_REGISTRATION_KEYS
        | _PANE_FAULT_CONSUMER_CHAIN_KEYS
        | _DEADLINE_OBSERVATION_KEYS
        | _INVALID_CLAIM_EVIDENCE_KEYS
        | _PRECLAIM_FAILURE_INTENT_KEYS
        | _BOUND_LIFECYCLE_EVIDENCE_KEYS
        | _OWNERSHIP_ABSENT_KEYS
        | _TERMINAL_FAILURE_KEYS
        | _LAUNCH_TERMINAL_V2_KEYS
        | _ATTEMPTED_TERMINAL_PAYLOAD_KEYS
        | _FINALIZATION_SECONDARY_FAILURE_KEYS
        | _LAUNCHER_TERMINAL_PUBLISH_ERROR_KEYS
        | _FINALIZATION_INNER_FAILURE_KEYS
        | _POST_HANDOFF_FAILURE_KEYS
        | _POST_HANDOFF_FINALIZATION_FAILURE_KEYS
        | _SEALED_LIFECYCLE_ARTIFACT_KEYS
        | _LIFECYCLE_PUBLISHER_KEYS
        | _LIFECYCLE_WAIT_STATUS_KEYS
        | _LIFECYCLE_RAW_WAIT_V3_KEYS
        | _LIFECYCLE_RAW_WAIT_PUBLISH_FAILURE_V1_KEYS
        | _POSTCLAIM_FINALIZATION_PROFILE_V1_KEYS
        | _LAUNCH_RECEIPT_V5_KEYS
    )
