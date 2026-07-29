from __future__ import annotations

import json
import multiprocessing
import hashlib
from pathlib import Path

import pytest

from safa.evaluation.r9_resources import (
    AdmissionStatus,
    CampaignFailedError,
    FailureKind,
    FcntlSlotLockBackend,
    GpuSnapshot,
    LockAcquireStatus,
    R9_GPU_HEADROOM_BYTES,
    R9_GPU_SLOT_CLAIM_BYTES,
    R9PeerStatusStore,
    R9ResourceScheduler,
    RamAdmissionReason,
    RamSnapshot,
    ResourceContractError,
    SlotLease,
    SystemResourceProbe,
    WorkerRequest,
    evaluate_ram_admission,
    gpu_slot_capacity,
    newest_worker_for_termination,
    parse_nvidia_smi_snapshots,
    parse_proc_meminfo,
    ram_slot_budget_bytes,
)


CONTRACT_SHA = "a" * 64
MIB = 1024**2


class _Probe:
    def __init__(self, snapshots: tuple[GpuSnapshot, ...], ram: RamSnapshot) -> None:
        self.snapshots = snapshots
        self.ram = ram
        self.gpu_calls = 0
        self.ram_calls = 0

    def gpu_snapshots(self) -> tuple[GpuSnapshot, ...]:
        self.gpu_calls += 1
        return self.snapshots

    def ram_snapshot(self) -> RamSnapshot:
        self.ram_calls += 1
        return self.ram


class _PeerProbe:
    def __init__(self, terminal: bool) -> None:
        self.terminal = terminal
        self.calls: list[tuple[str, str]] = []

    def is_terminal(self, campaign_id: str, worker_id: str) -> bool:
        self.calls.append((campaign_id, worker_id))
        return self.terminal


def _child_lock_status(root: str, lease_payload: dict[str, object], queue) -> None:
    backend = FcntlSlotLockBackend(Path(root))
    lease = SlotLease.from_payload(lease_payload)
    result = backend.try_acquire(lease)
    queue.put(result.status.value)
    if result.status in {LockAcquireStatus.ACQUIRED, LockAcquireStatus.RESUMED}:
        backend.release(lease)


def _gpu(index: int, *, free_mib: int = 20_888) -> GpuSnapshot:
    return GpuSnapshot(
        index=index,
        uuid=f"GPU-{index}",
        total_bytes=24_576 * MIB,
        free_bytes=free_mib * MIB,
    )


def _request(
    worker: str,
    *,
    gpu_index: int = 0,
    uuid: str | None = None,
    ordinal: int = 0,
    digest: str = CONTRACT_SHA,
    ram_slot_budget_bytes: int = 110,
) -> WorkerRequest:
    return WorkerRequest(
        worker_id=worker,
        gpu_index=gpu_index,
        expected_gpu_uuid=uuid or f"GPU-{gpu_index}",
        resource_contract_sha256=digest,
        launch_ordinal=ordinal,
        ram_slot_budget_bytes=ram_slot_budget_bytes,
    )


def _lease(
    worker: str,
    *,
    slot: int = 0,
    ordinal: int = 0,
    campaign: str = "campaign",
) -> SlotLease:
    return SlotLease(
        campaign_id=campaign,
        worker_id=worker,
        gpu_uuid="GPU-0",
        slot_index=slot,
        resource_contract_sha256=CONTRACT_SHA,
        launch_ordinal=ordinal,
        ram_slot_budget_bytes=1_100,
    )


def _write_unlocked_lease(backend: FcntlSlotLockBackend, lease: SlotLease) -> None:
    path = backend.lock_path(gpu_uuid=lease.gpu_uuid, slot_index=lease.slot_index)
    path.write_text(
        json.dumps(lease.to_payload(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_proc_stat(proc_root: Path, pid: int, start_ticks: int) -> None:
    process_root = proc_root / str(pid)
    process_root.mkdir(parents=True, exist_ok=True)
    fields = ["S", *(["0"] * 18), str(start_ticks)]
    (process_root / "stat").write_text(
        f"{pid} (r9 worker) {' '.join(fields)}\n", encoding="utf-8"
    )


def test_peer_status_store_tracks_live_running_and_terminal_states(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_stat(proc_root, 101, 1_001)
    _write_proc_stat(proc_root, 202, 2_002)
    store = R9PeerStatusStore(
        tmp_path / "campaigns", campaign_id="campaign-a", proc_root=proc_root
    )
    store.record_admitted("worker:0", controller_pid=101)
    assert store.is_terminal("campaign-a", "worker:0") is False
    store.record_running("worker:0", pid=202)
    assert store.is_terminal("campaign-a", "worker:0") is False
    store.record_terminal("worker:0", state="succeeded")
    assert store.is_terminal("campaign-a", "worker:0") is True


def test_peer_status_store_detects_exit_and_pid_reuse(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_stat(proc_root, 303, 3_003)
    store = R9PeerStatusStore(
        tmp_path / "campaigns", campaign_id="campaign-a", proc_root=proc_root
    )
    store.record_admitted("exited", controller_pid=303)
    (proc_root / "303" / "stat").unlink()
    assert store.is_terminal("campaign-a", "exited") is True

    _write_proc_stat(proc_root, 404, 4_004)
    store.record_admitted("reused", controller_pid=404)
    _write_proc_stat(proc_root, 404, 4_005)
    assert store.is_terminal("campaign-a", "reused") is True


def test_peer_status_store_requires_explicit_status_evidence(tmp_path: Path) -> None:
    store = R9PeerStatusStore(
        tmp_path / "campaigns", campaign_id="campaign-a", proc_root=tmp_path / "proc"
    )
    with pytest.raises(ResourceContractError, match="evidence is missing"):
        store.is_terminal("campaign-b", "unknown")


def test_global_slot_lock_contends_across_campaigns_and_reclaims_crash(
    tmp_path: Path,
) -> None:
    locks = tmp_path / "global-locks"
    first = FcntlSlotLockBackend(locks)
    second = FcntlSlotLockBackend(locks)
    incumbent = _lease("first", campaign="campaign-a")
    challenger = _lease("second", campaign="campaign-b")
    assert first.try_acquire(incumbent).status is LockAcquireStatus.ACQUIRED
    assert second.try_acquire(challenger).status is LockAcquireStatus.CONTENDED
    first.release(incumbent)

    _write_unlocked_lease(first, incumbent)
    proc_root = tmp_path / "proc"
    _write_proc_stat(proc_root, 505, 5_005)
    incumbent_status = R9PeerStatusStore(
        tmp_path / "campaigns", campaign_id="campaign-a", proc_root=proc_root
    )
    incumbent_status.record_admitted("first", controller_pid=505)
    challenger_status = R9PeerStatusStore(
        tmp_path / "campaigns", campaign_id="campaign-b", proc_root=proc_root
    )
    stale = second.try_acquire(challenger)
    assert stale.status is LockAcquireStatus.STALE
    not_reclaimed = second.reclaim_terminal(
        challenger,
        expected_stale=incumbent,
        peer_status_probe=challenger_status,
    )
    assert not_reclaimed.status is LockAcquireStatus.STALE
    (proc_root / "505" / "stat").unlink()
    reclaimed = second.reclaim_terminal(
        challenger,
        expected_stale=incumbent,
        peer_status_probe=challenger_status,
    )
    assert reclaimed.status is LockAcquireStatus.RECLAIMED
    second.release(challenger)


def test_gpu_slot_formula_exact_20887_20888_mib_boundary_and_clamp() -> None:
    assert R9_GPU_SLOT_CLAIM_BYTES == 4_938_792_960 == 4_710 * MIB
    assert R9_GPU_HEADROOM_BYTES == 2_048 * MIB
    assert gpu_slot_capacity(_gpu(0, free_mib=2_047)) == 0
    assert gpu_slot_capacity(_gpu(0, free_mib=2_048 + 4_710)) == 1
    assert gpu_slot_capacity(_gpu(0, free_mib=20_887)) == 3
    assert gpu_slot_capacity(_gpu(0, free_mib=20_888)) == 4
    assert gpu_slot_capacity(_gpu(0, free_mib=24_576)) == 4


def test_ram_slot_budget_is_peak_rss_times_1p10_rounded_up() -> None:
    assert ram_slot_budget_bytes(10) == 11
    assert ram_slot_budget_bytes(11) == 13
    assert ram_slot_budget_bytes(4_938_792_960) == 5_432_672_256


def test_ram_boundaries_are_84p99_85_89p99_and_90_percent() -> None:
    total = 100_000
    below_admission = evaluate_ram_admission(
        RamSnapshot(total, 15_011), slot_budget_bytes=1
    )
    at_admission = evaluate_ram_admission(
        RamSnapshot(total, 15_001), slot_budget_bytes=1
    )
    below_hard = evaluate_ram_admission(RamSnapshot(total, 10_010), slot_budget_bytes=1)
    at_hard = evaluate_ram_admission(RamSnapshot(total, 10_000), slot_budget_bytes=1)

    assert below_admission.allowed
    assert below_admission.projected_used_bytes == 84_990
    assert at_admission.reason is RamAdmissionReason.PROJECTED_LIMIT
    assert at_admission.projected_used_bytes == 85_000
    assert below_hard.reason is RamAdmissionReason.PROJECTED_LIMIT
    assert below_hard.used_bytes == 89_990
    assert at_hard.reason is RamAdmissionReason.HARD_LIMIT
    assert at_hard.used_bytes == 90_000


def test_nvidia_smi_probe_parses_four_gpus_and_hard_binds_uuid() -> None:
    output = "\n".join(
        f"{index}, GPU-{index}, 24576, {20888 + index}" for index in range(4)
    )
    calls: list[tuple[str, ...]] = []

    def runner(arguments) -> str:
        calls.append(tuple(arguments))
        return output

    probe = SystemResourceProbe(
        command_runner=runner,
        meminfo_reader=lambda: "MemTotal: 1000 kB\nMemAvailable: 500 kB\n",
    )
    snapshots = probe.gpu_snapshots()

    assert [snapshot.index for snapshot in snapshots] == [0, 1, 2, 3]
    assert [gpu_slot_capacity(snapshot) for snapshot in snapshots] == [4, 4, 4, 4]
    assert probe.gpu_snapshot(2, expected_uuid="GPU-2") == snapshots[2]
    with pytest.raises(ResourceContractError, match="does not match"):
        probe.gpu_snapshot(2, expected_uuid="GPU-wrong")
    assert calls[0][0] == "nvidia-smi"


def test_linux_ram_probe_requires_memavailable_without_fallback() -> None:
    snapshot = parse_proc_meminfo(
        "MemTotal:       262144 kB\n"
        "MemFree:         65536 kB\n"
        "MemAvailable:   131072 kB\n"
    )
    assert snapshot == RamSnapshot(262_144 * 1024, 131_072 * 1024)
    with pytest.raises(ResourceContractError, match="MemAvailable"):
        parse_proc_meminfo("MemTotal: 262144 kB\nMemFree: 65536 kB\n")


@pytest.mark.parametrize(
    "bad_output",
    (
        "",
        "0, GPU-0, 24576",
        "0, GPU-0, 24576, NaN",
        "0, GPU-0, 24576, 20000\n1, GPU-0, 24576, 20000",
    ),
)
def test_gpu_probe_rejects_malformed_or_nonfinite_data(bad_output: str) -> None:
    with pytest.raises(ResourceContractError):
        parse_nvidia_smi_snapshots(bad_output)


def test_fcntl_context_holds_cross_backend_lock_and_releases(tmp_path: Path) -> None:
    first = FcntlSlotLockBackend(tmp_path)
    second = FcntlSlotLockBackend(tmp_path)
    lease = _lease("worker-0")

    with first.lease_context(lease) as acquired:
        assert acquired.status is LockAcquireStatus.ACQUIRED
        contention = second.try_acquire(_lease("worker-1"))
        assert contention.status is LockAcquireStatus.CONTENDED

    acquired = second.try_acquire(_lease("worker-1"))
    assert acquired.status is LockAcquireStatus.ACQUIRED
    second.release(_lease("worker-1"))


def test_fcntl_slot_lease_is_contended_by_a_real_peer_process(
    tmp_path: Path,
) -> None:
    backend = FcntlSlotLockBackend(tmp_path)
    owner = _lease("owner")
    contender = _lease("contender")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()

    with backend.lease_context(owner):
        process = context.Process(
            target=_child_lock_status,
            args=(str(tmp_path), contender.to_payload(), queue),
        )
        process.start()
        process.join(timeout=10)
        assert process.exitcode == 0
        assert queue.get(timeout=1) == LockAcquireStatus.CONTENDED.value


def test_exact_durable_lease_resumes_but_changed_contract_fails(
    tmp_path: Path,
) -> None:
    backend = FcntlSlotLockBackend(tmp_path)
    lease = _lease("worker-0", ordinal=7)
    _write_unlocked_lease(backend, lease)

    resumed = backend.try_acquire(lease)
    assert resumed.status is LockAcquireStatus.RESUMED
    backend.release(lease)

    _write_unlocked_lease(backend, lease)
    changed = SlotLease(
        **{
            **lease.__dict__,
            "resource_contract_sha256": "b" * 64,
        }
    )
    with pytest.raises(ResourceContractError, match="durable slot contract"):
        backend.try_acquire(changed)


def test_stale_lease_requires_terminal_peer_before_exact_reclaim(
    tmp_path: Path,
) -> None:
    backend = FcntlSlotLockBackend(tmp_path)
    old = _lease("old-worker", campaign="old-campaign")
    new = _lease("new-worker")
    _write_unlocked_lease(backend, old)

    stale = backend.try_acquire(new)
    assert stale.status is LockAcquireStatus.STALE
    assert stale.incumbent == old

    reclaimed = backend.reclaim_terminal(
        new,
        expected_stale=old,
        peer_status_probe=_PeerProbe(True),
    )
    assert reclaimed.status is LockAcquireStatus.RECLAIMED
    backend.release(new)


def test_legacy_recovery_writes_receipt_before_clearing_exact_stale_lease(
    tmp_path: Path,
) -> None:
    backend = FcntlSlotLockBackend(tmp_path / "locks")
    stale = _lease("evaluator-smoke:arcface", campaign="r9-report-only-formal-v9")
    _write_unlocked_lease(backend, stale)
    lock_path = backend.lock_path(gpu_uuid=stale.gpu_uuid, slot_index=stale.slot_index)
    raw_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    receipt_path = tmp_path / "receipts" / "legacy.json"

    recovered = backend.recover_legacy_lease(
        expected_stale=stale,
        expected_raw_sha256=raw_sha256,
        legacy_pid=707,
        legacy_process_start_ticks=7_070,
        receipt_path=receipt_path,
        proc_root=tmp_path / "proc",
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert recovered.raw_lease_sha256 == raw_sha256
    assert receipt["raw_lease_sha256"] == raw_sha256
    assert receipt["stale_lease"] == stale.to_payload()
    assert lock_path.read_bytes() == b""


def test_scheduler_admits_four_slots_on_each_of_four_uuid_bound_gpus(
    tmp_path: Path,
) -> None:
    probe = _Probe(
        tuple(_gpu(index) for index in range(4)),
        RamSnapshot(total_bytes=1_000_000, available_bytes=900_000),
    )
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=probe,
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )

    for gpu_index in range(4):
        for slot_index in range(4):
            ordinal = gpu_index * 4 + slot_index
            decision = scheduler.admit_worker(
                _request(
                    f"worker-{ordinal}",
                    gpu_index=gpu_index,
                    ordinal=ordinal,
                )
            )
            assert decision.status is AdmissionStatus.ADMITTED
            assert decision.lease is not None
            assert decision.lease.slot_index == slot_index
            assert decision.lease.gpu_uuid == f"GPU-{gpu_index}"

    blocked = scheduler.admit_worker(_request("worker-16", gpu_index=0, ordinal=16))
    assert blocked.status is AdmissionStatus.LOCK_CONTENTION
    assert len(scheduler.active_leases) == 16
    for lease in tuple(scheduler.active_leases):
        scheduler.release_worker(lease.worker_id)


def test_scheduler_cumulative_ram_reservations_queue_exact_85_percent(
    tmp_path: Path,
) -> None:
    probe = _Probe(
        (_gpu(0),),
        RamSnapshot(total_bytes=100_000, available_bytes=15_440),
    )
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=probe,
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )

    projected = []
    for ordinal in range(3):
        decision = scheduler.admit_worker(
            _request(f"worker-{ordinal}", ordinal=ordinal)
        )
        assert decision.status is AdmissionStatus.ADMITTED
        projected.append(decision.ram.projected_used_bytes)

    blocked = scheduler.admit_worker(_request("queued", ordinal=3))
    blocked_again = scheduler.admit_worker(_request("queued", ordinal=3))

    assert projected == [84_670, 84_780, 84_890]
    assert blocked.status is AdmissionStatus.RAM_LIMIT
    assert blocked.ram.reason is RamAdmissionReason.PROJECTED_LIMIT
    assert blocked.ram.projected_used_bytes == 85_000
    assert blocked_again == blocked
    assert [lease.worker_id for lease in scheduler.active_leases] == [
        "worker-0",
        "worker-1",
        "worker-2",
    ]
    assert probe.gpu_calls == 3
    for lease in tuple(scheduler.active_leases):
        scheduler.release_worker(lease.worker_id)


def test_scheduler_uses_each_workers_explicit_smoke_bound_ram_budget(
    tmp_path: Path,
) -> None:
    probe = _Probe(
        (_gpu(0),),
        RamSnapshot(total_bytes=100_000, available_bytes=16_000),
    )
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=probe,
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )

    first = scheduler.admit_worker(
        _request("quality", ordinal=0, ram_slot_budget_bytes=200)
    )
    second = scheduler.admit_worker(
        _request("arcface", ordinal=1, ram_slot_budget_bytes=500)
    )
    blocked = scheduler.admit_worker(
        _request("generation", ordinal=2, ram_slot_budget_bytes=400)
    )

    assert first.status is AdmissionStatus.ADMITTED
    assert first.lease is not None
    assert first.lease.ram_slot_budget_bytes == 200
    assert first.ram.projected_used_bytes == 84_200
    assert second.status is AdmissionStatus.ADMITTED
    assert second.lease is not None
    assert second.lease.ram_slot_budget_bytes == 500
    assert second.ram.projected_used_bytes == 84_700
    assert blocked.status is AdmissionStatus.RAM_LIMIT
    assert blocked.ram.projected_used_bytes == 85_100
    for lease in tuple(scheduler.active_leases):
        scheduler.release_worker(lease.worker_id)


def test_scheduler_rejects_resume_with_changed_ram_budget(tmp_path: Path) -> None:
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=_Probe(
            (_gpu(0),),
            RamSnapshot(total_bytes=100_000, available_bytes=90_000),
        ),
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )
    scheduler.admit_worker(_request("worker", ram_slot_budget_bytes=200))

    with pytest.raises(CampaignFailedError) as caught:
        scheduler.admit_worker(_request("worker", ram_slot_budget_bytes=201))

    assert caught.value.failure.kind is FailureKind.CONTRACT_MISMATCH
    assert "resume request changed" in caught.value.failure.reason


def test_scheduler_release_returns_one_ram_reservation_to_admission_pool(
    tmp_path: Path,
) -> None:
    probe = _Probe(
        (_gpu(0),),
        RamSnapshot(total_bytes=100_000, available_bytes=15_440),
    )
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=probe,
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )
    for ordinal in range(3):
        scheduler.admit_worker(_request(f"worker-{ordinal}", ordinal=ordinal))
    assert (
        scheduler.admit_worker(_request("queued", ordinal=3)).status
        is AdmissionStatus.RAM_LIMIT
    )

    scheduler.release_worker("worker-1")
    admitted = scheduler.admit_worker(_request("queued", ordinal=3))

    assert admitted.status is AdmissionStatus.ADMITTED
    assert admitted.ram.projected_used_bytes == 84_890
    assert [lease.worker_id for lease in scheduler.active_leases] == [
        "worker-0",
        "worker-2",
        "queued",
    ]
    for lease in tuple(scheduler.active_leases):
        scheduler.release_worker(lease.worker_id)


def test_scheduler_active_resume_counts_each_ram_reservation_once(
    tmp_path: Path,
) -> None:
    probe = _Probe(
        (_gpu(0),),
        RamSnapshot(total_bytes=100_000, available_bytes=15_440),
    )
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=probe,
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )
    first = scheduler.admit_worker(_request("worker-0", ordinal=0))
    scheduler.admit_worker(_request("worker-1", ordinal=1))

    resumed = scheduler.admit_worker(_request("worker-0", ordinal=0))
    third = scheduler.admit_worker(_request("worker-2", ordinal=2))

    assert resumed.status is AdmissionStatus.RESUMED
    assert resumed.lease == first.lease
    assert resumed.ram.projected_used_bytes == 84_780
    assert third.status is AdmissionStatus.ADMITTED
    assert third.ram.projected_used_bytes == 84_890
    assert len(scheduler.active_leases) == 3
    for lease in tuple(scheduler.active_leases):
        scheduler.release_worker(lease.worker_id)


def test_scheduler_uuid_mismatch_is_campaign_hard_failure_without_lock(
    tmp_path: Path,
) -> None:
    probe = _Probe((_gpu(0),), RamSnapshot(total_bytes=100_000, available_bytes=90_000))
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=probe,
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )

    with pytest.raises(CampaignFailedError) as caught:
        scheduler.admit_worker(_request("worker", uuid="GPU-wrong"))

    assert caught.value.failure.kind is FailureKind.CONTRACT_MISMATCH
    assert not scheduler.active_leases
    assert not caught.value.failure.retry_allowed


@pytest.mark.parametrize("terminal", (False, True))
def test_scheduler_stale_peer_reclaim_is_explicit(
    tmp_path: Path, terminal: bool
) -> None:
    backend = FcntlSlotLockBackend(tmp_path)
    old = _lease("old-worker", campaign="old-campaign")
    _write_unlocked_lease(backend, old)
    peer = _PeerProbe(terminal)
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=_Probe(
            (_gpu(0, free_mib=2_048 + 4_710),),
            RamSnapshot(total_bytes=100_000, available_bytes=90_000),
        ),
        lock_backend=backend,
        peer_status_probe=peer,
    )

    decision = scheduler.admit_worker(_request("new-worker"))

    assert decision.status is (
        AdmissionStatus.RECLAIMED if terminal else AdmissionStatus.STALE_PEER
    )
    assert peer.calls == [("old-campaign", "old-worker")]
    if terminal:
        scheduler.release_worker("new-worker")


def test_scheduler_resumes_only_an_exact_persisted_lease(tmp_path: Path) -> None:
    backend = FcntlSlotLockBackend(tmp_path)
    persisted = SlotLease(
        campaign_id="campaign",
        worker_id="worker",
        gpu_uuid="GPU-0",
        slot_index=0,
        resource_contract_sha256=CONTRACT_SHA,
        launch_ordinal=5,
        ram_slot_budget_bytes=110,
    )
    _write_unlocked_lease(backend, persisted)
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=_Probe(
            (_gpu(0),),
            RamSnapshot(total_bytes=100_000, available_bytes=90_000),
        ),
        lock_backend=backend,
    )

    decision = scheduler.admit_worker(_request("worker", ordinal=5))

    assert decision.status is AdmissionStatus.RESUMED
    assert decision.lease == persisted
    scheduler.release_worker("worker")


def test_ram_hard_limit_selects_newest_worker_and_fails_campaign(
    tmp_path: Path,
) -> None:
    probe = _Probe((_gpu(0),), RamSnapshot(total_bytes=100_000, available_bytes=90_000))
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=probe,
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )
    scheduler.admit_worker(_request("older", ordinal=3))
    scheduler.admit_worker(_request("newest", ordinal=9))
    probe.ram = RamSnapshot(total_bytes=100_000, available_bytes=10_000)

    with pytest.raises(CampaignFailedError) as caught:
        scheduler.enforce_actual_ram_limit()

    failure = caught.value.failure
    assert failure.kind is FailureKind.RAM_HARD_LIMIT
    assert failure.terminate_worker_id == "newest"
    assert failure.campaign_failed
    assert not failure.retry_allowed
    assert not failure.batch_size_change_allowed
    assert not failure.algorithm_switch_allowed
    for lease in tuple(scheduler.active_leases):
        scheduler.release_worker(lease.worker_id)


@pytest.mark.parametrize(
    "kind",
    (FailureKind.OOM, FailureKind.NONFINITE, FailureKind.PEER_FAILURE),
)
def test_worker_failures_are_terminal_with_no_retry_or_degradation(
    tmp_path: Path, kind: FailureKind
) -> None:
    probe = _Probe((_gpu(0),), RamSnapshot(total_bytes=100_000, available_bytes=90_000))
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=probe,
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )
    scheduler.admit_worker(_request("worker"))
    gpu_calls_before = probe.gpu_calls
    ram_calls_before = probe.ram_calls

    with pytest.raises(CampaignFailedError) as caught:
        scheduler.fail_worker("worker", kind=kind)

    failure = caught.value.failure
    assert failure.kind is kind
    assert not failure.retry_allowed
    assert not failure.batch_size_change_allowed
    assert not failure.algorithm_switch_allowed
    with pytest.raises(CampaignFailedError):
        scheduler.admit_worker(_request("retry", ordinal=1))
    assert probe.gpu_calls == gpu_calls_before
    assert probe.ram_calls == ram_calls_before


def test_campaign_failure_bulk_release_clears_every_peer_lease(
    tmp_path: Path,
) -> None:
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=_Probe(
            (_gpu(0),),
            RamSnapshot(total_bytes=100_000, available_bytes=90_000),
        ),
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )
    scheduler.admit_worker(_request("failed", ordinal=0))
    scheduler.admit_worker(_request("peer-1", ordinal=1))
    scheduler.admit_worker(_request("peer-2", ordinal=2))

    with pytest.raises(CampaignFailedError):
        scheduler.fail_worker("failed", kind=FailureKind.PEER_FAILURE)

    assert scheduler.release_all_workers_after_failure() == ("peer-1", "peer-2")
    assert scheduler.active_leases == ()
    assert scheduler.release_all_workers_after_failure() == ()


def test_bulk_release_is_forbidden_before_campaign_failure(tmp_path: Path) -> None:
    scheduler = R9ResourceScheduler(
        campaign_id="campaign",
        resource_contract_sha256=CONTRACT_SHA,
        smoke_peak_rss_bytes=100,
        probe=_Probe(
            (_gpu(0),),
            RamSnapshot(total_bytes=100_000, available_bytes=90_000),
        ),
        lock_backend=FcntlSlotLockBackend(tmp_path),
    )
    with pytest.raises(ResourceContractError, match="recorded campaign failure"):
        scheduler.release_all_workers_after_failure()


def test_newest_worker_selection_refuses_ambiguous_ordinals() -> None:
    with pytest.raises(ResourceContractError, match="unique launch ordinals"):
        newest_worker_for_termination(
            (_lease("one", ordinal=4), _lease("two", slot=1, ordinal=4))
        )
