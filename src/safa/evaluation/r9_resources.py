from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Mapping, Protocol, Sequence


R9_MAX_GPU_SLOTS = 4
R9_GPU_SLOT_CLAIM_BYTES = 4_938_792_960
R9_GPU_HEADROOM_BYTES = 2 * 1024**3
R9_RAM_SLOT_MARGIN_NUMERATOR = 110
R9_RAM_SLOT_MARGIN_DENOMINATOR = 100
R9_RAM_ADMISSION_PERCENT = 85
R9_RAM_HARD_LIMIT_PERCENT = 90
_CAMPAIGN_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_ACTIVE_WORKER_STATES = frozenset({"admitted", "running"})
_TERMINAL_WORKER_STATES = frozenset({"succeeded", "failed", "terminated"})


class ResourceContractError(RuntimeError):
    """Raised when an R9 resource value violates the registered contract."""


class SlotUnavailableError(ResourceContractError):
    """Raised when a caller requires a slot that another process holds."""


class StaleSlotLeaseError(ResourceContractError):
    """Raised when an unlocked lease needs explicit terminal-peer evidence."""


class RamAdmissionReason(str, Enum):
    ADMITTED = "admitted"
    PROJECTED_LIMIT = "projected_ram_limit"
    HARD_LIMIT = "actual_ram_hard_limit"


class LockAcquireStatus(str, Enum):
    ACQUIRED = "acquired"
    RESUMED = "resumed"
    RECLAIMED = "reclaimed_terminal_peer"
    CONTENDED = "contended"
    STALE = "stale_unlocked_lease"


class AdmissionStatus(str, Enum):
    ADMITTED = "admitted"
    RESUMED = "resumed"
    RECLAIMED = "reclaimed_terminal_peer"
    RAM_LIMIT = "projected_ram_limit"
    GPU_LIMIT = "insufficient_vram"
    LOCK_CONTENTION = "slot_lock_contention"
    STALE_PEER = "stale_peer_not_terminal"


class FailureKind(str, Enum):
    OOM = "oom"
    NONFINITE = "nonfinite"
    CONTRACT_MISMATCH = "contract_mismatch"
    PEER_FAILURE = "peer_failure"
    RAM_HARD_LIMIT = "actual_ram_hard_limit"


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    uuid: str
    free_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.index, "GPU index")
        if not self.uuid:
            raise ResourceContractError("GPU UUID must be non-empty")
        _require_positive_int(self.total_bytes, "GPU total bytes")
        _require_nonnegative_int(self.free_bytes, "GPU free bytes")
        if self.free_bytes > self.total_bytes:
            raise ResourceContractError("GPU free bytes cannot exceed total bytes")


@dataclass(frozen=True)
class RamSnapshot:
    total_bytes: int
    available_bytes: int

    def __post_init__(self) -> None:
        _require_positive_int(self.total_bytes, "RAM total bytes")
        _require_nonnegative_int(self.available_bytes, "RAM available bytes")
        if self.available_bytes > self.total_bytes:
            raise ResourceContractError("RAM available bytes cannot exceed total bytes")

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.available_bytes


@dataclass(frozen=True)
class RamAdmission:
    allowed: bool
    reason: RamAdmissionReason
    used_bytes: int
    projected_used_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class SlotLease:
    campaign_id: str
    worker_id: str
    gpu_uuid: str
    slot_index: int
    resource_contract_sha256: str
    launch_ordinal: int
    ram_slot_budget_bytes: int

    def __post_init__(self) -> None:
        _require_nonempty(self.campaign_id, "campaign ID")
        _require_nonempty(self.worker_id, "worker ID")
        _require_nonempty(self.gpu_uuid, "GPU UUID")
        _require_nonnegative_int(self.slot_index, "slot index")
        if self.slot_index >= R9_MAX_GPU_SLOTS:
            raise ResourceContractError("slot index exceeds the R9 per-GPU limit")
        _require_sha256(self.resource_contract_sha256, "resource contract SHA256")
        _require_nonnegative_int(self.launch_ordinal, "launch ordinal")
        _require_positive_int(self.ram_slot_budget_bytes, "RAM slot budget bytes")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "worker_id": self.worker_id,
            "gpu_uuid": self.gpu_uuid,
            "slot_index": self.slot_index,
            "resource_contract_sha256": self.resource_contract_sha256,
            "launch_ordinal": self.launch_ordinal,
            "ram_slot_budget_bytes": self.ram_slot_budget_bytes,
            "gpu_slot_claim_bytes": R9_GPU_SLOT_CLAIM_BYTES,
            "gpu_headroom_bytes": R9_GPU_HEADROOM_BYTES,
        }

    @classmethod
    def from_payload(cls, payload: object) -> SlotLease:
        if not isinstance(payload, dict):
            raise ResourceContractError("slot lease payload must be an object")
        expected_keys = {
            "schema_version",
            "campaign_id",
            "worker_id",
            "gpu_uuid",
            "slot_index",
            "resource_contract_sha256",
            "launch_ordinal",
            "ram_slot_budget_bytes",
            "gpu_slot_claim_bytes",
            "gpu_headroom_bytes",
        }
        if set(payload) != expected_keys:
            raise ResourceContractError("slot lease payload fields disagree with R9")
        if payload["schema_version"] != 1:
            raise ResourceContractError("slot lease schema version mismatch")
        if payload["gpu_slot_claim_bytes"] != R9_GPU_SLOT_CLAIM_BYTES:
            raise ResourceContractError("slot lease GPU claim mismatch")
        if payload["gpu_headroom_bytes"] != R9_GPU_HEADROOM_BYTES:
            raise ResourceContractError("slot lease GPU headroom mismatch")
        return cls(
            campaign_id=str(payload["campaign_id"]),
            worker_id=str(payload["worker_id"]),
            gpu_uuid=str(payload["gpu_uuid"]),
            slot_index=_strict_int(payload["slot_index"], "slot index"),
            resource_contract_sha256=str(payload["resource_contract_sha256"]),
            launch_ordinal=_strict_int(payload["launch_ordinal"], "launch ordinal"),
            ram_slot_budget_bytes=_strict_int(
                payload["ram_slot_budget_bytes"], "RAM slot budget bytes"
            ),
        )


@dataclass(frozen=True)
class LockAcquireResult:
    status: LockAcquireStatus
    lease: SlotLease | None = None
    incumbent: SlotLease | None = None


@dataclass(frozen=True)
class WorkerRequest:
    worker_id: str
    gpu_index: int
    expected_gpu_uuid: str
    resource_contract_sha256: str
    launch_ordinal: int
    ram_slot_budget_bytes: int

    def __post_init__(self) -> None:
        _require_nonempty(self.worker_id, "worker ID")
        _require_nonnegative_int(self.gpu_index, "GPU index")
        _require_nonempty(self.expected_gpu_uuid, "expected GPU UUID")
        _require_sha256(self.resource_contract_sha256, "resource contract SHA256")
        _require_nonnegative_int(self.launch_ordinal, "launch ordinal")
        _require_positive_int(self.ram_slot_budget_bytes, "RAM slot budget bytes")


@dataclass(frozen=True)
class CampaignFailure:
    campaign_id: str
    kind: FailureKind
    worker_id: str | None
    terminate_worker_id: str | None
    reason: str
    campaign_failed: bool = True
    retry_allowed: bool = False
    batch_size_change_allowed: bool = False
    algorithm_switch_allowed: bool = False


@dataclass(frozen=True)
class AdmissionDecision:
    status: AdmissionStatus
    worker_id: str
    gpu_capacity: int
    ram: RamAdmission
    lease: SlotLease | None = None
    incumbent: SlotLease | None = None


class CampaignFailedError(ResourceContractError):
    def __init__(self, failure: CampaignFailure) -> None:
        self.failure = failure
        super().__init__(failure.reason)


class PeerStatusProbe(Protocol):
    def is_terminal(self, campaign_id: str, worker_id: str) -> bool: ...


class R9PeerStatusStore:
    """Durable worker liveness evidence shared by every R9 campaign."""

    def __init__(
        self,
        campaign_base_root: Path,
        *,
        campaign_id: str,
        proc_root: Path = Path("/proc"),
    ) -> None:
        _require_campaign_id(campaign_id)
        self.campaign_base_root = Path(campaign_base_root)
        self.campaign_id = campaign_id
        self.proc_root = Path(proc_root)

    def record_admitted(
        self, worker_id: str, *, controller_pid: int | None = None
    ) -> None:
        pid = os.getpid() if controller_pid is None else controller_pid
        self._transition(worker_id, state="admitted", pid=pid, require_previous=False)

    def record_running(self, worker_id: str, *, pid: int) -> None:
        self._transition(worker_id, state="running", pid=pid, require_previous=True)

    def record_terminal(self, worker_id: str, *, state: str) -> None:
        if state not in _TERMINAL_WORKER_STATES:
            raise ResourceContractError("worker terminal state is not registered")
        path = self.status_path(self.campaign_id, worker_id)
        previous = self._read_status(
            path, campaign_id=self.campaign_id, worker_id=worker_id
        )
        if previous["state"] not in _ACTIVE_WORKER_STATES:
            raise ResourceContractError(
                "terminal worker transition requires an active state"
            )
        payload = dict(previous)
        payload["state"] = state
        self._write_status(path, payload)

    def is_terminal(self, campaign_id: str, worker_id: str) -> bool:
        path = self.status_path(campaign_id, worker_id)
        status = self._read_status(path, campaign_id=campaign_id, worker_id=worker_id)
        state = str(status["state"])
        if state in _TERMINAL_WORKER_STATES:
            return True
        if state not in _ACTIVE_WORKER_STATES:
            raise ResourceContractError("worker status state is not registered")
        observed = _read_process_start_ticks(self.proc_root, int(status["pid"]))
        if observed is None:
            return True
        return observed != int(status["process_start_ticks"])

    def status_path(self, campaign_id: str, worker_id: str) -> Path:
        _require_campaign_id(campaign_id)
        _require_nonempty(worker_id, "worker ID")
        digest = hashlib.sha256(worker_id.encode("utf-8")).hexdigest()
        return (
            self.campaign_base_root / campaign_id / "worker_status" / f"{digest}.json"
        )

    def _transition(
        self,
        worker_id: str,
        *,
        state: str,
        pid: int,
        require_previous: bool,
    ) -> None:
        _require_positive_int(pid, "worker status PID")
        if state not in _ACTIVE_WORKER_STATES:
            raise ResourceContractError("worker active state is not registered")
        path = self.status_path(self.campaign_id, worker_id)
        if require_previous:
            previous = self._read_status(
                path, campaign_id=self.campaign_id, worker_id=worker_id
            )
            if previous["state"] != "admitted" or state != "running":
                raise ResourceContractError("worker status transition is not canonical")
        elif path.exists():
            raise ResourceContractError("worker admission status already exists")
        start_ticks = _read_process_start_ticks(self.proc_root, pid)
        if start_ticks is None:
            raise ResourceContractError("worker status PID is not live")
        payload: dict[str, object] = {
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "worker_id": worker_id,
            "state": state,
            "pid": pid,
            "process_start_ticks": start_ticks,
        }
        self._write_status(path, payload, exclusive=not require_previous)

    @staticmethod
    def _read_status(
        path: Path, *, campaign_id: str, worker_id: str
    ) -> dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ResourceContractError(
                "peer worker status evidence is missing"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ResourceContractError(
                "peer worker status evidence is corrupt"
            ) from error
        expected = {
            "schema_version",
            "campaign_id",
            "worker_id",
            "state",
            "pid",
            "process_start_ticks",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ResourceContractError("peer worker status fields are not canonical")
        if payload["schema_version"] != 1:
            raise ResourceContractError("peer worker status schema version mismatch")
        if payload["campaign_id"] != campaign_id or payload["worker_id"] != worker_id:
            raise ResourceContractError("peer worker status identity mismatch")
        state = payload["state"]
        if state not in _ACTIVE_WORKER_STATES | _TERMINAL_WORKER_STATES:
            raise ResourceContractError("peer worker status state is not registered")
        _require_positive_int(payload["pid"], "peer worker status PID")
        _require_positive_int(
            payload["process_start_ticks"], "peer worker process start ticks"
        )
        return payload

    @staticmethod
    def _write_status(
        path: Path, payload: Mapping[str, object], *, exclusive: bool = False
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        if exclusive:
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as error:
                raise ResourceContractError(
                    "worker admission status already exists"
                ) from error
            try:
                _write_all(fd, encoded)
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_directory(path.parent)
            return
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            _write_all(fd, encoded)
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            if fd >= 0:
                os.close(fd)
            temporary.unlink(missing_ok=True)


class ResourceProbe(Protocol):
    def gpu_snapshots(self) -> tuple[GpuSnapshot, ...]: ...

    def ram_snapshot(self) -> RamSnapshot: ...


class SlotLockBackend(Protocol):
    def try_acquire(self, lease: SlotLease) -> LockAcquireResult: ...

    def reclaim_terminal(
        self,
        lease: SlotLease,
        *,
        expected_stale: SlotLease,
        peer_status_probe: PeerStatusProbe,
    ) -> LockAcquireResult: ...

    def release(self, lease: SlotLease) -> None: ...


CommandRunner = Callable[[Sequence[str]], str]
MeminfoReader = Callable[[], str]


class SystemResourceProbe:
    """Linux probe with injectable command and proc readers for CPU tests."""

    def __init__(
        self,
        *,
        command_runner: CommandRunner | None = None,
        meminfo_reader: MeminfoReader | None = None,
    ) -> None:
        self._command_runner = command_runner or _run_command
        self._meminfo_reader = meminfo_reader or _read_proc_meminfo

    def gpu_snapshots(self) -> tuple[GpuSnapshot, ...]:
        output = self._command_runner(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            )
        )
        return parse_nvidia_smi_snapshots(output)

    def gpu_snapshot(self, index: int, *, expected_uuid: str) -> GpuSnapshot:
        _require_nonnegative_int(index, "GPU index")
        _require_nonempty(expected_uuid, "expected GPU UUID")
        matches = [item for item in self.gpu_snapshots() if item.index == index]
        if len(matches) != 1:
            raise ResourceContractError(
                f"GPU index {index} was not reported exactly once"
            )
        snapshot = matches[0]
        if snapshot.uuid != expected_uuid:
            raise ResourceContractError(
                f"GPU index {index} UUID {snapshot.uuid!r} does not match "
                f"{expected_uuid!r}"
            )
        return snapshot

    def ram_snapshot(self) -> RamSnapshot:
        return parse_proc_meminfo(self._meminfo_reader())


class FcntlSlotLockBackend:
    """Hold UUID-addressed slot leases with non-blocking cross-process flock."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._held: dict[tuple[str, int], tuple[int, SlotLease]] = {}

    def lock_path(self, *, gpu_uuid: str, slot_index: int) -> Path:
        _require_nonempty(gpu_uuid, "GPU UUID")
        _require_nonnegative_int(slot_index, "slot index")
        if slot_index >= R9_MAX_GPU_SLOTS:
            raise ResourceContractError("slot index exceeds the R9 per-GPU limit")
        uuid_digest = hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()[:24]
        return self.root / f"gpu_{uuid_digest}.slot_{slot_index}.lock"

    def try_acquire(self, lease: SlotLease) -> LockAcquireResult:
        key = (lease.gpu_uuid, lease.slot_index)
        local = self._held.get(key)
        if local is not None:
            incumbent = local[1]
            if incumbent == lease:
                return LockAcquireResult(LockAcquireStatus.RESUMED, lease=lease)
            if _same_worker(incumbent, lease):
                raise ResourceContractError(
                    "same worker attempted to change its active slot lease contract"
                )
            return LockAcquireResult(LockAcquireStatus.CONTENDED, incumbent=incumbent)
        fd = os.open(
            self.lock_path(gpu_uuid=lease.gpu_uuid, slot_index=lease.slot_index),
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        if not _try_flock(fd):
            os.close(fd)
            return LockAcquireResult(LockAcquireStatus.CONTENDED)
        try:
            incumbent = _read_lease_fd(fd)
        except Exception:
            _unlock_and_close(fd)
            raise
        if incumbent is None:
            _write_lease_fd(fd, lease)
            self._held[key] = (fd, lease)
            return LockAcquireResult(LockAcquireStatus.ACQUIRED, lease=lease)
        if incumbent == lease:
            self._held[key] = (fd, lease)
            return LockAcquireResult(LockAcquireStatus.RESUMED, lease=lease)
        _unlock_and_close(fd)
        if _same_worker(incumbent, lease):
            raise ResourceContractError(
                "resumed worker lease disagrees with its durable slot contract"
            )
        return LockAcquireResult(LockAcquireStatus.STALE, incumbent=incumbent)

    def reclaim_terminal(
        self,
        lease: SlotLease,
        *,
        expected_stale: SlotLease,
        peer_status_probe: PeerStatusProbe,
    ) -> LockAcquireResult:
        terminal = peer_status_probe.is_terminal(
            expected_stale.campaign_id, expected_stale.worker_id
        )
        if not isinstance(terminal, bool):
            raise ResourceContractError("peer status probe must return a bool")
        if not terminal:
            return LockAcquireResult(LockAcquireStatus.STALE, incumbent=expected_stale)
        key = (lease.gpu_uuid, lease.slot_index)
        if key in self._held:
            return LockAcquireResult(
                LockAcquireStatus.CONTENDED, incumbent=self._held[key][1]
            )
        fd = os.open(
            self.lock_path(gpu_uuid=lease.gpu_uuid, slot_index=lease.slot_index),
            os.O_RDWR | os.O_CREAT,
            0o600,
        )
        if not _try_flock(fd):
            os.close(fd)
            return LockAcquireResult(LockAcquireStatus.CONTENDED)
        try:
            actual = _read_lease_fd(fd)
            if actual != expected_stale:
                raise ResourceContractError(
                    "stale slot lease changed before terminal-peer reclaim"
                )
            _write_lease_fd(fd, lease)
        except Exception:
            _unlock_and_close(fd)
            raise
        self._held[key] = (fd, lease)
        return LockAcquireResult(LockAcquireStatus.RECLAIMED, lease=lease)

    def release(self, lease: SlotLease) -> None:
        key = (lease.gpu_uuid, lease.slot_index)
        local = self._held.get(key)
        if local is None:
            raise ResourceContractError("cannot release a slot lease not held locally")
        fd, incumbent = local
        if incumbent != lease or _read_lease_fd(fd) != lease:
            raise ResourceContractError("slot lease ownership changed before release")
        _clear_lease_fd(fd)
        _unlock_and_close(fd)
        del self._held[key]

    @contextmanager
    def lease_context(
        self,
        lease: SlotLease,
        *,
        peer_status_probe: PeerStatusProbe | None = None,
    ) -> Iterator[LockAcquireResult]:
        result = self.try_acquire(lease)
        if result.status is LockAcquireStatus.CONTENDED:
            raise SlotUnavailableError("GPU slot is held by another process")
        if result.status is LockAcquireStatus.STALE:
            incumbent = result.incumbent
            if incumbent is None:
                raise ResourceContractError("stale lease result omitted its incumbent")
            if peer_status_probe is None:
                raise StaleSlotLeaseError(
                    "unlocked slot lease requires explicit terminal-peer evidence"
                )
            result = self.reclaim_terminal(
                lease,
                expected_stale=incumbent,
                peer_status_probe=peer_status_probe,
            )
            if result.status is LockAcquireStatus.STALE:
                raise StaleSlotLeaseError(
                    "unlocked slot lease belongs to a non-terminal peer"
                )
            if result.status is LockAcquireStatus.CONTENDED:
                raise SlotUnavailableError("GPU slot was reacquired during reclaim")
        try:
            yield result
        finally:
            self.release(lease)


def parse_nvidia_smi_snapshots(output: str) -> tuple[GpuSnapshot, ...]:
    rows: list[GpuSnapshot] = []
    for line_number, line in enumerate(output.splitlines(), 1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ResourceContractError(
                f"nvidia-smi row {line_number} must have exactly four fields"
            )
        index = _decimal_int(fields[0], f"nvidia-smi row {line_number} index")
        total_mib = _decimal_int(fields[2], f"nvidia-smi row {line_number} total MiB")
        free_mib = _decimal_int(fields[3], f"nvidia-smi row {line_number} free MiB")
        rows.append(
            GpuSnapshot(
                index=index,
                uuid=fields[1],
                total_bytes=total_mib * 1024**2,
                free_bytes=free_mib * 1024**2,
            )
        )
    if not rows:
        raise ResourceContractError("nvidia-smi reported no GPUs")
    if len({row.index for row in rows}) != len(rows):
        raise ResourceContractError("nvidia-smi reported duplicate GPU indices")
    if len({row.uuid for row in rows}) != len(rows):
        raise ResourceContractError("nvidia-smi reported duplicate GPU UUIDs")
    return tuple(sorted(rows, key=lambda row: row.index))


def parse_proc_meminfo(output: str) -> RamSnapshot:
    values: dict[str, int] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        if key not in {"MemTotal", "MemAvailable"}:
            continue
        fields = raw_value.split()
        if len(fields) != 2 or fields[1] != "kB":
            raise ResourceContractError(f"/proc/meminfo {key} must use kB")
        if key in values:
            raise ResourceContractError(f"/proc/meminfo repeats {key}")
        values[key] = _decimal_int(fields[0], f"/proc/meminfo {key}") * 1024
    if set(values) != {"MemTotal", "MemAvailable"}:
        raise ResourceContractError(
            "/proc/meminfo must report MemTotal and MemAvailable"
        )
    return RamSnapshot(
        total_bytes=values["MemTotal"],
        available_bytes=values["MemAvailable"],
    )


class R9ResourceScheduler:
    """Deterministic R9 admission state machine with injected probes and locks."""

    def __init__(
        self,
        *,
        campaign_id: str,
        resource_contract_sha256: str,
        smoke_peak_rss_bytes: int,
        probe: ResourceProbe,
        lock_backend: SlotLockBackend,
        peer_status_probe: PeerStatusProbe | None = None,
        gpu_slot_claim_bytes: int = R9_GPU_SLOT_CLAIM_BYTES,
        gpu_headroom_bytes: int = R9_GPU_HEADROOM_BYTES,
        max_gpu_slots: int = R9_MAX_GPU_SLOTS,
        ram_slot_budget_bytes_override: int | None = None,
    ) -> None:
        _require_nonempty(campaign_id, "campaign ID")
        _require_sha256(resource_contract_sha256, "resource contract SHA256")
        self.campaign_id = campaign_id
        self.resource_contract_sha256 = resource_contract_sha256
        self.ram_slot_budget_bytes = (
            ram_slot_budget_bytes(smoke_peak_rss_bytes)
            if ram_slot_budget_bytes_override is None
            else ram_slot_budget_bytes_override
        )
        _require_positive_int(self.ram_slot_budget_bytes, "RAM slot budget bytes")
        _require_positive_int(gpu_slot_claim_bytes, "GPU slot claim bytes")
        _require_nonnegative_int(gpu_headroom_bytes, "GPU headroom bytes")
        _require_positive_int(max_gpu_slots, "max GPU slots")
        self.gpu_slot_claim_bytes = gpu_slot_claim_bytes
        self.gpu_headroom_bytes = gpu_headroom_bytes
        self.max_gpu_slots = max_gpu_slots
        if self.max_gpu_slots > R9_MAX_GPU_SLOTS:
            raise ResourceContractError("max GPU slots exceeds the R9 limit")
        self._probe = probe
        self._lock_backend = lock_backend
        self._peer_status_probe = peer_status_probe
        self._active: dict[str, SlotLease] = {}
        self._failure: CampaignFailure | None = None

    @property
    def failure(self) -> CampaignFailure | None:
        return self._failure

    @property
    def active_leases(self) -> tuple[SlotLease, ...]:
        return tuple(
            sorted(
                self._active.values(),
                key=lambda lease: (lease.launch_ordinal, lease.worker_id),
            )
        )

    def admit_worker(self, request: WorkerRequest) -> AdmissionDecision:
        self._raise_if_failed()
        if request.resource_contract_sha256 != self.resource_contract_sha256:
            self._fail(
                kind=FailureKind.CONTRACT_MISMATCH,
                worker_id=request.worker_id,
                reason="worker resource contract SHA256 disagrees with campaign",
            )
        ram_snapshot = self._probe.ram_snapshot()
        self._enforce_ram_hard_limit(ram_snapshot)
        existing = self._active.get(request.worker_id)
        if existing is not None:
            if not _lease_matches_request(existing, request, self.campaign_id):
                self._fail(
                    kind=FailureKind.CONTRACT_MISMATCH,
                    worker_id=request.worker_id,
                    reason="active worker resume request changed its lease contract",
                )
            return AdmissionDecision(
                status=AdmissionStatus.RESUMED,
                worker_id=request.worker_id,
                gpu_capacity=self._gpu_slot_capacity(
                    _select_gpu_snapshot(
                        self._probe.gpu_snapshots(),
                        index=request.gpu_index,
                        expected_uuid=request.expected_gpu_uuid,
                    )
                ),
                ram=self._ram_admission_for(
                    ram_snapshot,
                    worker_id=request.worker_id,
                    requested_budget_bytes=request.ram_slot_budget_bytes,
                ),
                lease=existing,
            )
        if any(
            lease.launch_ordinal == request.launch_ordinal
            for lease in self._active.values()
        ):
            self._fail(
                kind=FailureKind.CONTRACT_MISMATCH,
                worker_id=request.worker_id,
                reason="launch ordinal must be unique within the active campaign",
            )
        ram = self._ram_admission_for(
            ram_snapshot,
            worker_id=request.worker_id,
            requested_budget_bytes=request.ram_slot_budget_bytes,
        )
        if not ram.allowed:
            return AdmissionDecision(
                status=AdmissionStatus.RAM_LIMIT,
                worker_id=request.worker_id,
                gpu_capacity=0,
                ram=ram,
            )
        try:
            gpu = _select_gpu_snapshot(
                self._probe.gpu_snapshots(),
                index=request.gpu_index,
                expected_uuid=request.expected_gpu_uuid,
            )
        except ResourceContractError as error:
            self._fail(
                kind=FailureKind.CONTRACT_MISMATCH,
                worker_id=request.worker_id,
                reason=str(error),
            )
        capacity = self._gpu_slot_capacity(gpu)
        if capacity == 0:
            return AdmissionDecision(
                status=AdmissionStatus.GPU_LIMIT,
                worker_id=request.worker_id,
                gpu_capacity=0,
                ram=ram,
            )
        stale_incumbent: SlotLease | None = None
        for slot_index in range(capacity):
            lease = SlotLease(
                campaign_id=self.campaign_id,
                worker_id=request.worker_id,
                gpu_uuid=request.expected_gpu_uuid,
                slot_index=slot_index,
                resource_contract_sha256=request.resource_contract_sha256,
                launch_ordinal=request.launch_ordinal,
                ram_slot_budget_bytes=request.ram_slot_budget_bytes,
            )
            try:
                acquired = self._lock_backend.try_acquire(lease)
            except ResourceContractError as error:
                self._fail(
                    kind=FailureKind.CONTRACT_MISMATCH,
                    worker_id=request.worker_id,
                    reason=str(error),
                )
            if acquired.status in {
                LockAcquireStatus.ACQUIRED,
                LockAcquireStatus.RESUMED,
            }:
                self._active[request.worker_id] = lease
                status = (
                    AdmissionStatus.ADMITTED
                    if acquired.status is LockAcquireStatus.ACQUIRED
                    else AdmissionStatus.RESUMED
                )
                return AdmissionDecision(
                    status=status,
                    worker_id=request.worker_id,
                    gpu_capacity=capacity,
                    ram=ram,
                    lease=lease,
                )
            if acquired.status is LockAcquireStatus.STALE:
                incumbent = acquired.incumbent
                if incumbent is None:
                    self._fail(
                        kind=FailureKind.CONTRACT_MISMATCH,
                        worker_id=request.worker_id,
                        reason="stale lock result omitted its incumbent lease",
                    )
                stale_incumbent = incumbent
                if self._peer_status_probe is not None:
                    try:
                        reclaimed = self._lock_backend.reclaim_terminal(
                            lease,
                            expected_stale=incumbent,
                            peer_status_probe=self._peer_status_probe,
                        )
                    except ResourceContractError as error:
                        self._fail(
                            kind=FailureKind.CONTRACT_MISMATCH,
                            worker_id=request.worker_id,
                            reason=str(error),
                        )
                    if reclaimed.status is LockAcquireStatus.RECLAIMED:
                        self._active[request.worker_id] = lease
                        return AdmissionDecision(
                            status=AdmissionStatus.RECLAIMED,
                            worker_id=request.worker_id,
                            gpu_capacity=capacity,
                            ram=ram,
                            lease=lease,
                            incumbent=incumbent,
                        )
        return AdmissionDecision(
            status=(
                AdmissionStatus.STALE_PEER
                if stale_incumbent is not None
                else AdmissionStatus.LOCK_CONTENTION
            ),
            worker_id=request.worker_id,
            gpu_capacity=capacity,
            ram=ram,
            incumbent=stale_incumbent,
        )

    def enforce_actual_ram_limit(self) -> CampaignFailure | None:
        self._raise_if_failed()
        return self._enforce_ram_hard_limit(self._probe.ram_snapshot())

    def fail_worker(self, worker_id: str, *, kind: FailureKind) -> None:
        self._raise_if_failed()
        if kind not in {
            FailureKind.OOM,
            FailureKind.NONFINITE,
            FailureKind.CONTRACT_MISMATCH,
            FailureKind.PEER_FAILURE,
        }:
            raise ResourceContractError("unsupported explicit worker failure kind")
        lease = self._active.pop(worker_id, None)
        if lease is not None:
            self._lock_backend.release(lease)
        self._fail(
            kind=kind,
            worker_id=worker_id,
            reason=f"worker {worker_id!r} failed with {kind.value}",
        )

    def release_worker(self, worker_id: str) -> None:
        lease = self._active.get(worker_id)
        if lease is None:
            raise ResourceContractError("cannot release an unknown worker")
        self._lock_backend.release(lease)
        del self._active[worker_id]

    def release_all_workers_after_failure(self) -> tuple[str, ...]:
        """Release every active lease after a campaign-wide peer failure."""
        if self._failure is None:
            raise ResourceContractError(
                "bulk worker release requires a recorded campaign failure"
            )
        worker_ids = tuple(
            lease.worker_id
            for lease in sorted(
                self._active.values(),
                key=lambda lease: (lease.launch_ordinal, lease.worker_id),
            )
        )
        for worker_id in worker_ids:
            lease = self._active.pop(worker_id)
            self._lock_backend.release(lease)
        return worker_ids

    def _enforce_ram_hard_limit(self, snapshot: RamSnapshot) -> CampaignFailure | None:
        if snapshot.used_bytes * 100 < snapshot.total_bytes * R9_RAM_HARD_LIMIT_PERCENT:
            return None
        newest = newest_worker_for_termination(self.active_leases)
        worker_id = None if newest is None else newest.worker_id
        self._fail(
            kind=FailureKind.RAM_HARD_LIMIT,
            worker_id=worker_id,
            terminate_worker_id=worker_id,
            reason="actual system RAM reached the R9 90% hard limit",
        )

    def _ram_admission_for(
        self,
        snapshot: RamSnapshot,
        *,
        worker_id: str,
        requested_budget_bytes: int,
    ) -> RamAdmission:
        existing = self._active.get(worker_id)
        slot_budget_bytes = (
            requested_budget_bytes
            if existing is None
            else existing.ram_slot_budget_bytes
        )
        reserved_bytes = sum(
            lease.ram_slot_budget_bytes
            for active_worker_id, lease in self._active.items()
            if active_worker_id != worker_id
        )
        return evaluate_ram_admission(
            snapshot,
            slot_budget_bytes=slot_budget_bytes,
            reserved_bytes=reserved_bytes,
        )

    def _gpu_slot_capacity(self, snapshot: GpuSnapshot) -> int:
        usable_bytes = snapshot.free_bytes - self.gpu_headroom_bytes
        if usable_bytes <= 0:
            return 0
        return min(self.max_gpu_slots, usable_bytes // self.gpu_slot_claim_bytes)

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise CampaignFailedError(self._failure)

    def _fail(
        self,
        *,
        kind: FailureKind,
        worker_id: str | None,
        reason: str,
        terminate_worker_id: str | None = None,
    ) -> None:
        failure = CampaignFailure(
            campaign_id=self.campaign_id,
            kind=kind,
            worker_id=worker_id,
            terminate_worker_id=terminate_worker_id,
            reason=reason,
        )
        self._failure = failure
        raise CampaignFailedError(failure)


def newest_worker_for_termination(
    leases: Sequence[SlotLease],
) -> SlotLease | None:
    if not leases:
        return None
    ordinals = [lease.launch_ordinal for lease in leases]
    if len(set(ordinals)) != len(ordinals):
        raise ResourceContractError(
            "newest-worker selection requires unique launch ordinals"
        )
    return max(leases, key=lambda lease: lease.launch_ordinal)


def gpu_slot_capacity(snapshot: GpuSnapshot) -> int:
    """Return the exact R9 capacity derived from the current free VRAM."""
    usable_bytes = snapshot.free_bytes - R9_GPU_HEADROOM_BYTES
    if usable_bytes <= 0:
        return 0
    return min(R9_MAX_GPU_SLOTS, usable_bytes // R9_GPU_SLOT_CLAIM_BYTES)


def ram_slot_budget_bytes(smoke_peak_rss_bytes: int) -> int:
    """Apply the registered 10% RSS margin using exact integer arithmetic."""
    _require_positive_int(smoke_peak_rss_bytes, "smoke peak RSS bytes")
    numerator = smoke_peak_rss_bytes * R9_RAM_SLOT_MARGIN_NUMERATOR
    return (numerator + R9_RAM_SLOT_MARGIN_DENOMINATOR - 1) // (
        R9_RAM_SLOT_MARGIN_DENOMINATOR
    )


def evaluate_ram_admission(
    snapshot: RamSnapshot,
    *,
    slot_budget_bytes: int,
    reserved_bytes: int = 0,
) -> RamAdmission:
    """Evaluate exact RAM boundaries with active reservations counted once."""
    _require_positive_int(slot_budget_bytes, "RAM slot budget bytes")
    _require_nonnegative_int(reserved_bytes, "reserved RAM bytes")
    used_bytes = snapshot.used_bytes
    projected_used_bytes = used_bytes + reserved_bytes + slot_budget_bytes
    if used_bytes * 100 >= snapshot.total_bytes * R9_RAM_HARD_LIMIT_PERCENT:
        return RamAdmission(
            allowed=False,
            reason=RamAdmissionReason.HARD_LIMIT,
            used_bytes=used_bytes,
            projected_used_bytes=projected_used_bytes,
            total_bytes=snapshot.total_bytes,
        )
    if projected_used_bytes * 100 >= snapshot.total_bytes * R9_RAM_ADMISSION_PERCENT:
        return RamAdmission(
            allowed=False,
            reason=RamAdmissionReason.PROJECTED_LIMIT,
            used_bytes=used_bytes,
            projected_used_bytes=projected_used_bytes,
            total_bytes=snapshot.total_bytes,
        )
    return RamAdmission(
        allowed=True,
        reason=RamAdmissionReason.ADMITTED,
        used_bytes=used_bytes,
        projected_used_bytes=projected_used_bytes,
        total_bytes=snapshot.total_bytes,
    )


def _require_positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ResourceContractError(f"{label} must be a positive integer")


def _require_nonnegative_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceContractError(f"{label} must be a non-negative integer")


def _require_nonempty(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ResourceContractError(f"{label} must be a non-empty string")


def _require_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ResourceContractError(f"{label} must be a lowercase SHA256 digest")


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResourceContractError(f"{label} must be an integer")
    return value


def _decimal_int(value: str, label: str) -> int:
    if not value or any(character not in "0123456789" for character in value):
        raise ResourceContractError(f"{label} must be a base-10 integer")
    return int(value)


def _run_command(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        list(arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _read_proc_meminfo() -> str:
    return Path("/proc/meminfo").read_text(encoding="utf-8")


def _read_process_start_ticks(proc_root: Path, pid: int) -> int | None:
    _require_positive_int(pid, "process PID")
    try:
        raw = (Path(proc_root) / str(pid) / "stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    closing = raw.rfind(")")
    if closing < 0:
        raise ResourceContractError("process stat record is malformed")
    fields = raw[closing + 1 :].split()
    start_time_index = 22 - 3
    if len(fields) <= start_time_index:
        raise ResourceContractError("process stat record omits start time")
    raw_ticks = fields[start_time_index]
    if not raw_ticks or any(character not in "0123456789" for character in raw_ticks):
        raise ResourceContractError("process start time must be a base-10 integer")
    ticks = int(raw_ticks)
    _require_positive_int(ticks, "process start ticks")
    return ticks


def _require_campaign_id(value: str) -> None:
    _require_nonempty(value, "campaign ID")
    if _CAMPAIGN_ID_PATTERN.fullmatch(value) is None:
        raise ResourceContractError("campaign ID must be an immutable lowercase slug")


def _try_flock(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise
    return True


def _read_lease_fd(fd: int) -> SlotLease | None:
    os.lseek(fd, 0, os.SEEK_SET)
    payload = os.read(fd, 64 * 1024)
    if not payload:
        return None
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResourceContractError("slot lease file is corrupt") from error
    return SlotLease.from_payload(decoded)


def _write_all(fd: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("slot lease write made no progress")
        remaining = remaining[written:]


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_lease_fd(fd: int, lease: SlotLease) -> None:
    payload = (
        json.dumps(
            lease.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    _write_all(fd, payload)
    os.fsync(fd)


def _clear_lease_fd(fd: int) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.fsync(fd)


def _unlock_and_close(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _same_worker(left: SlotLease, right: SlotLease) -> bool:
    return left.campaign_id == right.campaign_id and left.worker_id == right.worker_id


def _select_gpu_snapshot(
    snapshots: Sequence[GpuSnapshot], *, index: int, expected_uuid: str
) -> GpuSnapshot:
    if len({snapshot.index for snapshot in snapshots}) != len(snapshots):
        raise ResourceContractError("resource probe reported duplicate GPU indices")
    if len({snapshot.uuid for snapshot in snapshots}) != len(snapshots):
        raise ResourceContractError("resource probe reported duplicate GPU UUIDs")
    matches = [snapshot for snapshot in snapshots if snapshot.index == index]
    if len(matches) != 1:
        raise ResourceContractError(f"GPU index {index} was not reported exactly once")
    snapshot = matches[0]
    if snapshot.uuid != expected_uuid:
        raise ResourceContractError(
            f"GPU index {index} UUID {snapshot.uuid!r} does not match {expected_uuid!r}"
        )
    return snapshot


def _lease_matches_request(
    lease: SlotLease, request: WorkerRequest, campaign_id: str
) -> bool:
    return (
        lease.campaign_id == campaign_id
        and lease.worker_id == request.worker_id
        and lease.gpu_uuid == request.expected_gpu_uuid
        and lease.resource_contract_sha256 == request.resource_contract_sha256
        and lease.launch_ordinal == request.launch_ordinal
        and lease.ram_slot_budget_bytes == request.ram_slot_budget_bytes
    )
