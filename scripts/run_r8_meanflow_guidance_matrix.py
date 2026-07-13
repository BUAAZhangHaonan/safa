#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shlex
import shutil
import stat
import statistics
import subprocess
import time
from typing import Any, Iterator, Mapping, Sequence

import yaml

from safa.evaluation.meanflow_guidance_runner import validate_guidance_config
from safa.evaluation.r8_arm_contracts import (
    canonical_arm_config_digest,
    require_arm_config_digest,
)
from safa.evaluation.r8_visual_evidence import (
    build_visual_evidence_contract,
    validate_visual_review_arm,
    write_contact_sheets,
)


DEFAULT_PYTHON = "/home/hdd3/zhanghaonan/anaconda3/envs/safa/bin/python"
ROOT = Path("artifacts/r8_meanflow_flow_map_guidance")
CONFIG_ROOT = Path("configs/medium_v2/experiments")
SEMIGROUP_CONFIG = CONFIG_ROOT / "r8_meanflow_semigroup_preflight.yaml"
NATIVE_CONFIG = CONFIG_ROOT / "r8_meanflow_native_ema.yaml"
FLOW_MAP1_CONFIG = CONFIG_ROOT / "r8_meanflow_official_xt_flow_map1_gpu0.yaml"
FLOW_MAP2_CONFIG = CONFIG_ROOT / "r8_meanflow_official_xt_flow_map2_gpu1.yaml"
PAPER_CONFIG = CONFIG_ROOT / "r8_meanflow_paper_split_gpu2.yaml"
NOISE_CONFIGS = (
    CONFIG_ROOT / "r8_meanflow_noise_fixed_eta025.yaml",
    CONFIG_ROOT / "r8_meanflow_noise_fixed_eta05.yaml",
    CONFIG_ROOT / "r8_meanflow_noise_shell_eta1.yaml",
    CONFIG_ROOT / "r8_meanflow_noise_shell_eta2.yaml",
)
CALIBRATION_MANIFEST = ROOT / "manifests/calibration_64.jsonl"
FULL_MANIFEST = ROOT / "manifests/full_2048.jsonl"
CALIBRATION_MANIFEST_SHA256 = "ffc1f04f671533ee1498f4b03565826920afcc4e5c6ab244fc6f9b7aa680f964"
FULL_MANIFEST_SHA256 = "7f830ad3f84089bcf83d092fbffaf2b5c3335cf68a4b397f04b65f362f79ae5b"
CHECKPOINT_SHA256 = "4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d"
SEMIGROUP_GATE = ROOT / "semigroup/semigroup_gate.json"
SEMIGROUP_GATE_DRAFT = ROOT / "semigroup/semigroup_gate_draft.json"
SCHEDULE_MANIFEST = ROOT / "semigroup/locked_schedule_manifest.json"
SEMIGROUP_VISUAL_REVIEW = ROOT / "semigroup/visual_review.json"
SEMIGROUP_VISUAL_EVIDENCE = ROOT / "semigroup/visual_evidence.json"
SELECTION = ROOT / "selection.json"
VISUAL_REVIEW = ROOT / "visual_review.json"
CALIBRATION_VISUAL_EVIDENCE = ROOT / "calibration/visual_evidence.json"
FULL_VISUAL_REVIEW = ROOT / "full/visual_review.json"
FULL_VISUAL_EVIDENCE = ROOT / "full/visual_evidence.json"
QUALITY_SCRIPT = Path("scripts/eval_generation_quality.py")
GUIDANCE_SCRIPT = Path("scripts/run_meanflow_flow_map_guidance.py")
MIN_FREE_MEMORY_MIB = 12000
PROCESS_POLL_INTERVAL_SECONDS = 0.05
HOST_GPU_LOCK_DIR = Path("/tmp") / f"safa-r8-meanflow-guidance-{os.getuid()}"


@dataclass(frozen=True)
class GpuState:
    index: int
    uuid: str
    free_memory_mib: int
    compute_processes: tuple[str, ...]


@dataclass(frozen=True)
class RunContract:
    physical_gpu: int
    gpu_uuid: str
    config: Path
    source_configs: tuple[Path, ...]
    family: str
    arm_ids: tuple[str, ...]
    shard_index: int
    num_shards: int
    sample_count: int
    sample_manifest: Path
    sample_manifest_sha256: str
    output_dir: Path
    log_path: Path
    env: Mapping[str, str]
    command: tuple[str, ...]
    runtime_config: Path | None = None


@dataclass(frozen=True)
class MatrixPlan:
    repo_root: Path
    python: str
    phase: str
    campaign_id: str | None
    campaign_root: Path | None
    execute: bool
    allow_busy_gpus: bool
    status_dir: Path
    lock_path: Path
    runs: tuple[RunContract, ...]
    schedule_manifest: Path | None
    sample_manifest: Path
    sample_manifest_sha256: str
    external_compute_processes: Mapping[int, tuple[str, ...]]
    full_contract: Mapping[str, Any] | None = None
    calibration_gate: Mapping[str, Any] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible four-GPU R8 frozen MeanFlow guidance matrix.",
        epilog="Dry-run is the default; GPU work requires explicit --execute.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument(
        "--phase", choices=("semigroup", "calibrate", "full", "all"), default="all"
    )
    parser.add_argument("--campaign-id")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-busy-gpus", action="store_true")
    args = parser.parse_args(argv)
    if args.phase in {"calibrate", "all"}:
        if args.campaign_id is None:
            parser.error("--campaign-id is required for calibration")
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.campaign_id) is None:
            parser.error("--campaign-id must be a lowercase slug")
    elif args.campaign_id is not None:
        parser.error("--campaign-id is only valid for calibration")
    return args


def build_matrix_plan(
    args: argparse.Namespace,
    *,
    semigroup_gate: Mapping[str, Any] | None = None,
    full_contract: Mapping[str, Any] | None = None,
) -> MatrixPlan:
    repo_root = Path(args.repo_root).resolve()
    phase = str(args.phase)
    if phase == "all":
        raise ValueError("build_matrix_plan requires one concrete phase, not 'all'")
    common = {
        "repo_root": repo_root,
        "python": str(args.python),
        "phase": phase,
        "campaign_id": None,
        "campaign_root": None,
        "execute": bool(args.execute),
        "allow_busy_gpus": bool(args.allow_busy_gpus),
        "status_dir": ROOT / phase,
        "lock_path": ROOT / f".{phase}.lock",
        "external_compute_processes": {},
    }
    if phase == "semigroup":
        runs = tuple(_semigroup_run(repo_root, str(args.python), gpu) for gpu in range(4))
        return MatrixPlan(
            **common,
            runs=runs,
            schedule_manifest=None,
            sample_manifest=CALIBRATION_MANIFEST,
            sample_manifest_sha256=CALIBRATION_MANIFEST_SHA256,
        )
    if phase == "calibrate":
        campaign_id = str(args.campaign_id)
        if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", campaign_id) is None:
            raise ValueError("calibration requires a valid campaign_id")
        campaign_root = ROOT / "campaigns" / campaign_id
        common.update(
            campaign_id=campaign_id,
            campaign_root=campaign_root,
            status_dir=campaign_root / "status",
            lock_path=campaign_root / ".calibrate.lock",
        )
        gate = dict(semigroup_gate) if semigroup_gate is not None else _load_required_json(
            repo_root / SEMIGROUP_GATE, "semigroup gate"
        )
        _validate_gate_identity(gate)
        if gate.get("gate_passed") is True:
            runs = _guided_calibration_runs(repo_root, str(args.python), gate, campaign_root)
            schedule = SCHEDULE_MANIFEST
        else:
            runs = _fallback_calibration_runs(repo_root, str(args.python), campaign_root)
            schedule = None
        return MatrixPlan(
            **common,
            runs=runs,
            schedule_manifest=schedule,
            sample_manifest=CALIBRATION_MANIFEST,
            sample_manifest_sha256=CALIBRATION_MANIFEST_SHA256,
            calibration_gate=gate,
        )
    contract = dict(full_contract) if full_contract is not None else load_full_contract(
        repo_root / SELECTION, repo_root / VISUAL_REVIEW
    )
    _validate_full_contract(contract)
    runs = tuple(_full_run(repo_root, str(args.python), gpu, contract) for gpu in range(4))
    return MatrixPlan(
        **common,
        runs=runs,
        schedule_manifest=SCHEDULE_MANIFEST
        if _winner_is_fmrg(contract)
        else None,
        sample_manifest=FULL_MANIFEST,
        sample_manifest_sha256=FULL_MANIFEST_SHA256,
        full_contract=contract,
    )


def _base_env(repo_root: Path) -> dict[str, str]:
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONPATH": "src",
        "SAFA_REPO_ROOT": str(repo_root),
    }


def _semigroup_run(repo_root: Path, python: str, gpu: int) -> RunContract:
    output = ROOT / f"semigroup/shards/shard_{gpu}"
    generation = (
        python,
        str(GUIDANCE_SCRIPT),
        "--config",
        str(SEMIGROUP_CONFIG),
        "--output-dir",
        str(output),
        "--shard-index",
        str(gpu),
        "--num-shards",
        "4",
        "--mode",
        "semigroup",
        "--max-samples",
        "64",
    )
    log = ROOT / f"semigroup/logs/gpu{gpu}.log"
    shell = _skip_completed_generation(output, generation)
    command = ("/bin/bash", "-lc", f"{{ {shell}; }} > {shlex.quote(str(log))} 2>&1")
    return RunContract(
        physical_gpu=gpu,
        gpu_uuid="",
        config=SEMIGROUP_CONFIG,
        source_configs=(SEMIGROUP_CONFIG,),
        family="semigroup",
        arm_ids=(f"semigroup_shard_{gpu}",),
        shard_index=gpu,
        num_shards=4,
        sample_count=16,
        sample_manifest=CALIBRATION_MANIFEST,
        sample_manifest_sha256=CALIBRATION_MANIFEST_SHA256,
        output_dir=output,
        log_path=log,
        env=_base_env(repo_root),
        command=command,
    )


def _guided_calibration_runs(
    repo_root: Path, python: str, gate: Mapping[str, Any], campaign_root: Path
) -> tuple[RunContract, ...]:
    t_cut = float(gate.get("selected_t_cut", gate.get("t_cut")))
    specs = (
        (0, FLOW_MAP1_CONFIG, "official_flow_map1"),
        (1, FLOW_MAP2_CONFIG, "official_flow_map2"),
        (2, PAPER_CONFIG, "paper_algorithm_split"),
        (3, NOISE_CONFIGS[0], "initial_noise_oracle"),
    )
    runs = []
    for gpu, config, family in specs:
        if gpu in (0, 1):
            candidates = [
                (f"{family}_adam_step{value:g}", ["--optimization-mode", "official_adam", "--step-size", str(value)])
                for value in (1.0, 3.0)
            ]
            candidates.extend(
                (
                    f"{family}_normalized_eta{value:g}",
                    [
                        "--optimization-mode",
                        "paper_normalized_direct_autograd",
                        "--step-size",
                        str(value),
                    ],
                )
                for value in (0.25, 0.5, 1.0, 2.0)
            )
        elif gpu == 2:
            candidates = [
                (f"paper_split_eta{value:g}", ["--step-size", str(value)])
                for value in (0.25, 0.5, 1.0, 2.0)
            ]
        else:
            candidates = [
                (
                    f"noise_{_config_stem(noise_config)}",
                    ["--config-override", str(noise_config)],
                )
                for noise_config in NOISE_CONFIGS
            ]
        for arm_id, overrides in candidates:
            candidate_config = config
            if overrides[:1] == ["--config-override"]:
                candidate_config = Path(overrides[1])
                overrides = []
            output = campaign_root / "calibration" / arm_id
            generation = _generation_command(
                python=python,
                config=candidate_config,
                output_dir=output,
                shard_index=0,
                num_shards=1,
                overrides=overrides,
                t_cut=t_cut if gpu < 3 else None,
            )
            quality = build_quality_command(
                python=python, output_dir=output, manifest=CALIBRATION_MANIFEST
            )
            shell = " && ".join(
                (_skip_completed_generation(output, generation), _skip_completed_quality(output, quality))
            )
            log = campaign_root / f"logs/gpu{gpu}_{arm_id}.log"
            runs.append(
                RunContract(
                    physical_gpu=gpu,
                    gpu_uuid="",
                    config=candidate_config,
                    source_configs=(candidate_config,),
                    family=family,
                    arm_ids=(arm_id,),
                    shard_index=0,
                    num_shards=1,
                    sample_count=64,
                    sample_manifest=CALIBRATION_MANIFEST,
                    sample_manifest_sha256=CALIBRATION_MANIFEST_SHA256,
                    output_dir=output,
                    log_path=log,
                    env=_base_env(repo_root),
                    command=(
                        "/bin/bash",
                        "-lc",
                        f"{{ {shell}; }} > {shlex.quote(str(log))} 2>&1",
                    ),
                )
            )
    runs.append(_native_unguided_calibration_run(repo_root, python, campaign_root))
    return tuple(runs)


def _native_unguided_calibration_run(
    repo_root: Path, python: str, campaign_root: Path
) -> RunContract:
    config = _load_yaml(repo_root / NATIVE_CONFIG)
    _require_sampling_seed_1337(config, "native unguided calibration config")
    arm_id = "native_unguided_64"
    output = campaign_root / "calibration" / arm_id
    runtime_config = campaign_root / "runtime_configs/native_unguided_64.yaml"
    generation = _generation_command(
        python=python,
        config=runtime_config,
        output_dir=output,
        shard_index=0,
        num_shards=1,
        overrides=("--mode", "native"),
    )
    quality = build_quality_command(
        python=python, output_dir=output, manifest=CALIBRATION_MANIFEST
    )
    shell = " && ".join(
        (_skip_completed_generation(output, generation), _skip_completed_quality(output, quality))
    )
    log = campaign_root / "logs/gpu3_native_unguided_64.log"
    return RunContract(
        physical_gpu=3,
        gpu_uuid="",
        config=NATIVE_CONFIG,
        source_configs=(NATIVE_CONFIG,),
        family="native_unguided",
        arm_ids=(arm_id,),
        shard_index=0,
        num_shards=1,
        sample_count=64,
        sample_manifest=CALIBRATION_MANIFEST,
        sample_manifest_sha256=CALIBRATION_MANIFEST_SHA256,
        output_dir=output,
        log_path=log,
        env=_base_env(repo_root),
        command=(
            "/bin/bash",
            "-lc",
            f"{{ {shell}; }} > {shlex.quote(str(log))} 2>&1",
        ),
        runtime_config=runtime_config,
    )


def _fallback_calibration_runs(
    repo_root: Path, python: str, campaign_root: Path
) -> tuple[RunContract, ...]:
    runs = []
    for gpu, config in enumerate(NOISE_CONFIGS):
        arm_id = f"fallback_{_config_stem(config)}"
        output = campaign_root / "calibration" / arm_id
        generation = _generation_command(
            python=python,
            config=config,
            output_dir=output,
            shard_index=0,
            num_shards=1,
        )
        quality = build_quality_command(
            python=python, output_dir=output, manifest=CALIBRATION_MANIFEST
        )
        log = campaign_root / f"logs/gpu{gpu}.log"
        shell = " && ".join(
            (_skip_completed_generation(output, generation), _skip_completed_quality(output, quality))
        )
        runs.append(
            RunContract(
                physical_gpu=gpu,
                gpu_uuid="",
                config=config,
                source_configs=(config,),
                family="initial_noise_fallback",
                arm_ids=(arm_id,),
                shard_index=0,
                num_shards=1,
                sample_count=64,
                sample_manifest=CALIBRATION_MANIFEST,
                sample_manifest_sha256=CALIBRATION_MANIFEST_SHA256,
                output_dir=output,
                log_path=log,
                env=_base_env(repo_root),
                command=(
                    "/bin/bash",
                    "-lc",
                    f"{{ {shell}; }} > {shlex.quote(str(log))} 2>&1",
                ),
            )
        )
    return tuple(runs)


def _full_run(
    repo_root: Path, python: str, gpu: int, contract: Mapping[str, Any]
) -> RunContract:
    native_runtime = ROOT / "full/runtime_configs/native.yaml"
    winner_runtime = ROOT / "full/runtime_configs/winner.yaml"
    winner_source = Path(str(contract["winner"]["config"]))
    output_root = ROOT / f"full/shards/shard_{gpu}"
    commands = []
    for arm_id, runtime_config in (("native", native_runtime), ("winner", winner_runtime)):
        output = output_root / arm_id
        generation = _generation_command(
            python=python,
            config=runtime_config,
            output_dir=output,
            shard_index=gpu,
            num_shards=4,
            t_cut=_winner_t_cut(contract) if arm_id == "winner" else None,
        )
        commands.append(_skip_completed_generation(output, generation))
    log = ROOT / f"full/logs/gpu{gpu}.log"
    return RunContract(
        physical_gpu=gpu,
        gpu_uuid="",
        config=winner_runtime,
        source_configs=(NATIVE_CONFIG, winner_source),
        family="full_native_winner",
        arm_ids=("native", "winner"),
        shard_index=gpu,
        num_shards=4,
        sample_count=512,
        sample_manifest=FULL_MANIFEST,
        sample_manifest_sha256=FULL_MANIFEST_SHA256,
        output_dir=output_root,
        log_path=log,
        env=_base_env(repo_root),
        command=(
            "/bin/bash",
            "-lc",
            f"{' && '.join(commands)} > {shlex.quote(str(log))} 2>&1",
        ),
    )


def _generation_command(
    *,
    python: str,
    config: Path,
    output_dir: Path,
    shard_index: int,
    num_shards: int,
    overrides: Sequence[str] = (),
    t_cut: float | None = None,
) -> tuple[str, ...]:
    command = [
        python,
        str(GUIDANCE_SCRIPT),
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
    ]
    command.extend(str(value) for value in overrides)
    if t_cut is not None:
        command.extend(
            [
                "--semigroup-report",
                str(SEMIGROUP_GATE),
                "--schedule-manifest",
                str(SCHEDULE_MANIFEST),
                "--t-cut",
                str(t_cut),
            ]
        )
    return tuple(command)


def _skip_completed_generation(output: Path, command: Sequence[str]) -> str:
    completion = output / "completion.json"
    return (
        f"if test -f {shlex.quote(str(completion))}; then :; else "
        f"{shlex.join(command)}; fi"
    )


def _skip_completed_quality(output: Path, command: Sequence[str]) -> str:
    del output
    return shlex.join(command)


def build_quality_command(
    *, python: str, output_dir: Path, manifest: Path
) -> tuple[str, ...]:
    return (
        python,
        str(QUALITY_SCRIPT),
        "--real-index",
        "data/index/val_face_mixed_e14.jsonl",
        "--generated-dir",
        str(output_dir / "generated_images"),
        "--per-sample-jsonl",
        str(output_dir / "per_sample.jsonl"),
        "--sample-id-manifest",
        str(manifest),
        "--output",
        str(output_dir / "quality.json"),
        "--generation-result",
        str(output_dir / "generation_result.json"),
        "--reuse-valid-output",
        "--seed",
        "1337",
        "--device",
        "cuda:0",
        "--metrics",
        "fid",
        "kid",
        "niqe",
        "sharpness",
    )


def validate_config_assets(repo_root: Path, config: Mapping[str, Any]) -> None:
    for field in (
        "checkpoint",
        "e0_checkpoint",
        "edev_checkpoint",
        "heldout_e1_checkpoint",
        "heldout_e2_checkpoint",
        "vae_path",
        "index",
        "features",
    ):
        value = config.get(field)
        path = Path(str(value)) if value else Path("")
        resolved = path if path.is_absolute() else repo_root / path
        if not value or not resolved.exists():
            raise FileNotFoundError(f"R8 config references missing {field}: {value!r}")
    validate_guidance_config(config)


def validate_preflight(
    plan: MatrixPlan,
    *,
    gpu_states: Mapping[int, GpuState] | None = None,
) -> MatrixPlan:
    if not plan.repo_root.is_dir():
        raise FileNotFoundError(f"repo root does not exist: {plan.repo_root}")
    python = Path(plan.python)
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Python executable not found or not executable: {python}")
    for script in (GUIDANCE_SCRIPT, QUALITY_SCRIPT):
        if not (plan.repo_root / script).is_file():
            raise FileNotFoundError(f"required R8 script does not exist: {script}")
    physical_gpus = [run.physical_gpu for run in plan.runs]
    if plan.phase == "calibrate":
        if set(physical_gpus) != {0, 1, 2, 3}:
            raise ValueError("R8 calibration must use physical GPUs 0,1,2,3")
    elif physical_gpus != [0, 1, 2, 3]:
        raise ValueError("R8 matrix must pin physical GPUs 0,1,2,3 exactly once")
    checked: set[Path] = set()
    for run in plan.runs:
        for config_path in run.source_configs:
            if config_path in checked:
                continue
            checked.add(config_path)
            payload = _load_yaml(plan.repo_root / config_path)
            validate_config_assets(plan.repo_root, payload)
    states = dict(gpu_states) if gpu_states is not None else query_gpu_states()
    _validate_gpu_states(states, allow_busy_gpus=plan.allow_busy_gpus)
    return _bind_gpu_uuids(plan, states)


def validate_artifact_paths(plan: MatrixPlan) -> None:
    for run in plan.runs:
        output = _resolve(plan.repo_root, run.output_dir)
        if not output.exists():
            continue
        if run.family == "full_native_winner":
            _validate_full_shard_root(output, run, plan)
            continue
        if (output / "resume_contract.json").is_file() or (output / "completion.json").is_file():
            continue
        raise FileExistsError(f"refusing to overwrite existing R8 output directory: {output}")


def _validate_full_shard_root(output: Path, run: RunContract, plan: MatrixPlan) -> None:
    if not output.is_dir():
        raise FileExistsError(f"full shard output is not a directory: {output}")
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in {"native", "winner"})
    if unexpected:
        raise FileExistsError(f"full shard output contains unowned entries: {unexpected!r}")
    for arm_id in ("native", "winner"):
        child = output / arm_id
        if not child.exists():
            continue
        completion_path = child / "completion.json"
        resume_path = child / "resume_contract.json"
        if not completion_path.is_file() and not resume_path.is_file():
            raise FileExistsError(
                f"full shard child lacks a resume/completion contract: {child}"
            )
        if completion_path.is_file():
            completion = _read_json(completion_path, "full shard completion")
            if completion.get("status") != "complete":
                raise ValueError(f"full shard completion is not complete: {completion_path}")
            expected_arm_digest = _full_arm_config_digest(plan, arm_id)
            if completion.get("arm_config_sha256") != expected_arm_digest:
                raise ValueError(
                    f"full {arm_id} completion disagrees with the locked winner arm config"
                )
        if resume_path.is_file():
            resume = _read_json(resume_path, "full shard resume contract")
            shard = resume.get("shard")
            if not isinstance(shard, Mapping) or (
                shard.get("index") != run.shard_index or shard.get("count") != run.num_shards
            ):
                raise ValueError(f"full shard resume contract has the wrong shard owner: {resume_path}")
            expected_mode = "native" if arm_id == "native" else None
            if expected_mode is not None and resume.get("mode") != expected_mode:
                raise ValueError(f"full native resume contract has the wrong mode: {resume_path}")
            if resume.get("arm_config_sha256") != _full_arm_config_digest(plan, arm_id):
                raise ValueError(
                    f"full {arm_id} resume disagrees with the locked winner arm config"
                )


def _full_arm_config_digest(plan: MatrixPlan, arm_id: str) -> str:
    if arm_id == "winner":
        if plan.full_contract is None:
            raise ValueError("full winner requires a locked full contract")
        return require_arm_config_digest(
            plan.full_contract["winner"].get("arm_config_sha256"),
            "winner arm config SHA256",
        )
    return canonical_arm_config_digest(_load_yaml(plan.repo_root / NATIVE_CONFIG))


def query_gpu_states() -> dict[int, GpuState]:
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states: dict[int, GpuState] = {}
    uuid_to_index: dict[str, int] = {}
    for line in gpu_result.stdout.splitlines():
        if not line.strip():
            continue
        index_text, uuid, memory_text = (part.strip() for part in line.split(",", maxsplit=2))
        index = int(index_text)
        uuid_to_index[uuid] = index
        states[index] = GpuState(index, uuid, int(memory_text), ())
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes: dict[int, list[str]] = {index: [] for index in states}
    for line in process_result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_name = (part.strip() for part in line.split(",", maxsplit=2))
        index = uuid_to_index.get(uuid)
        if index is not None:
            processes[index].append(f"pid={pid} {process_name}")
    return {
        index: replace(state, compute_processes=tuple(processes[index]))
        for index, state in states.items()
    }


def _validate_gpu_states(
    states: Mapping[int, GpuState], *, allow_busy_gpus: bool
) -> None:
    for index in range(4):
        state = states.get(index)
        if state is None:
            raise RuntimeError(f"nvidia-smi did not report required GPU {index}")
        if state.free_memory_mib < MIN_FREE_MEMORY_MIB:
            raise RuntimeError(
                f"GPU {index} requires at least {MIN_FREE_MEMORY_MIB} MiB free memory, "
                f"got {state.free_memory_mib} MiB"
            )
        if state.compute_processes and not allow_busy_gpus:
            raise RuntimeError(
                f"GPU {index} has an external compute process: {', '.join(state.compute_processes)}"
            )


def _bind_gpu_uuids(plan: MatrixPlan, states: Mapping[int, GpuState]) -> MatrixPlan:
    runs = []
    for run in plan.runs:
        state = states[run.physical_gpu]
        env = dict(run.env)
        env["CUDA_VISIBLE_DEVICES"] = state.uuid
        runs.append(replace(run, gpu_uuid=state.uuid, env=env))
    external = {
        index: states[index].compute_processes
        for index in range(4)
        if states[index].compute_processes
    }
    return replace(plan, runs=tuple(runs), external_compute_processes=external)


def render_dry_run(plan: MatrixPlan) -> str:
    lines = ["DRY RUN: no commands executed and no files or directories written."]
    lines.extend(
        (
            f"phase: {plan.phase}",
            f"sample_manifest: {plan.sample_manifest}",
            f"sample_manifest_sha256: {plan.sample_manifest_sha256}",
            f"allow_busy_gpus: {str(plan.allow_busy_gpus).lower()}",
        )
    )
    for run in plan.runs:
        env = " ".join(f"{key}={value}" for key, value in sorted(run.env.items()))
        lines.append(
            f"[gpu{run.physical_gpu} shard={run.shard_index}/{run.num_shards} "
            f"family={run.family}] {env}"
        )
        lines.append(f"  {shlex.join(run.command)}")
    return "\n".join(lines) + "\n"


def launch_matrix(plan: MatrixPlan) -> int:
    status_dir = _resolve(plan.repo_root, plan.status_dir)
    status_dir.mkdir(parents=True, exist_ok=True)
    for run in plan.runs:
        _resolve(plan.repo_root, run.log_path).parent.mkdir(parents=True, exist_ok=True)
    started_at = _timestamp()
    queues: dict[int, list[tuple[int, RunContract]]] = {gpu: [] for gpu in range(4)}
    for index, run in enumerate(plan.runs):
        queues[run.physical_gpu].append((index, run))
    active: dict[int, tuple[int, RunContract, Any, str]] = {}
    rows_by_index: dict[int, dict[str, Any]] = {}

    def start_next(gpu: int) -> None:
        if not queues[gpu]:
            return
        index, run = queues[gpu].pop(0)
        env = dict(os.environ)
        env.update(run.env)
        run_started = _timestamp()
        process = subprocess.Popen(
            run.command,
            cwd=plan.repo_root,
            env=env,
            start_new_session=True,
        )
        active[gpu] = (index, run, process, run_started)

    def terminate_active(status: str) -> None:
        terminated = []
        for gpu, (index, run, process, run_started) in list(active.items()):
            exit_code = _terminate_process_group(process)
            terminated.append((index, run, process, run_started, exit_code))
            del active[gpu]
        for index, run, process, run_started, exit_code in terminated:
            rows_by_index[index] = _status_row(
                run,
                pid=getattr(process, "pid", None),
                started_at=run_started,
                exit_code=exit_code,
                status=status,
            )

    try:
        for gpu in range(4):
            start_next(gpu)
        peer_failed = False
        while active:
            completed = []
            for gpu, (index, run, process, run_started) in list(active.items()):
                exit_code = process.poll()
                if exit_code is not None:
                    completed.append(
                        (gpu, index, run, process, run_started, int(exit_code))
                    )
            if not completed:
                time.sleep(PROCESS_POLL_INTERVAL_SECONDS)
                continue
            for gpu, _, _, process, _, _ in completed:
                process.wait()
                del active[gpu]
            if any(exit_code != 0 for _, _, _, _, _, exit_code in completed):
                peer_failed = True
                terminate_active("terminated_after_peer_failure")
            for gpu, index, run, process, run_started, exit_code in completed:
                rows_by_index[index] = _status_row(
                    run,
                    pid=getattr(process, "pid", None),
                    started_at=run_started,
                    exit_code=exit_code,
                    status="passed" if exit_code == 0 else "failed",
                )
            if peer_failed:
                break
            for gpu, *_ in completed:
                start_next(gpu)
    except BaseException as exc:
        terminate_active("terminated_after_launch_error")
        represented = set(rows_by_index)
        for index, run in enumerate(plan.runs):
            if index in represented:
                continue
            rows_by_index[index] = _status_row(
                run,
                pid=None,
                started_at=None,
                exit_code=None,
                status="not_started",
            )
        _write_matrix_status(
            status_dir,
            plan,
            [rows_by_index[index] for index in sorted(rows_by_index)],
            started_at=started_at,
            overall_status="failed",
            launch_error=f"{type(exc).__name__}: {exc}",
        )
        if not isinstance(exc, Exception):
            raise
        return 1
    for index, run in enumerate(plan.runs):
        if index not in rows_by_index:
            rows_by_index[index] = _status_row(
                run,
                pid=None,
                started_at=None,
                exit_code=None,
                status="not_started",
            )
    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    failed = any(row["exit_code"] != 0 for row in rows)
    _write_matrix_status(
        status_dir,
        plan,
        rows,
        started_at=started_at,
        overall_status="failed" if failed else "children_passed_pending_finalize",
    )
    return 1 if failed or peer_failed else 0


def _terminate_process_group(process: Any, *, terminate_timeout: float = 5.0) -> int:
    """Terminate the session created for one matrix command, including its children."""
    pid = getattr(process, "pid", None)
    try:
        if not isinstance(pid, int) or pid <= 0:
            raise ProcessLookupError
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
    try:
        try:
            return int(process.wait(timeout=terminate_timeout))
        except TypeError:
            return int(process.wait())
    except subprocess.TimeoutExpired:
        try:
            if not isinstance(pid, int) or pid <= 0:
                raise ProcessLookupError
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
        try:
            return int(process.wait(timeout=terminate_timeout))
        except TypeError:
            return int(process.wait())


def _status_row(
    run: RunContract,
    *,
    pid: int | None,
    started_at: str | None,
    exit_code: int | None,
    status: str,
) -> dict[str, Any]:
    return {
        "physical_gpu": run.physical_gpu,
        "gpu_uuid": run.gpu_uuid,
        "family": run.family,
        "arm_ids": list(run.arm_ids),
        "pid": pid,
        "started_at": started_at,
        "ended_at": _timestamp(),
        "exit_code": exit_code,
        "status": status,
        "peak_memory": run_peak_memory(run),
        "command": list(run.command),
        "output_dir": str(run.output_dir),
        "log": str(run.log_path),
    }


def run_peak_memory(run: RunContract) -> dict[str, int | None]:
    allocated: list[int] = []
    reserved: list[int] = []
    root = Path(run.output_dir)
    paths = [root / "generation_result.json"]
    if root.is_dir():
        paths.extend(root.glob("**/generation_result.json"))
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        payload = _read_json(path, "generation result")
        memory = payload.get("max_memory", {})
        if isinstance(memory, Mapping):
            if isinstance(memory.get("allocated_bytes"), int):
                allocated.append(int(memory["allocated_bytes"]))
            if isinstance(memory.get("reserved_bytes"), int):
                reserved.append(int(memory["reserved_bytes"]))
    return {
        "allocated": max(allocated) if allocated else None,
        "reserved": max(reserved) if reserved else None,
    }


def _write_matrix_status(
    status_dir: Path,
    plan: MatrixPlan,
    rows: list[dict[str, Any]],
    *,
    started_at: str,
    overall_status: str,
    launch_error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": plan.phase,
        "overall_status": overall_status,
        "started_at": started_at,
        "ended_at": _timestamp(),
        "allow_busy_gpus": plan.allow_busy_gpus,
        "external_compute_processes": {
            str(index): list(processes)
            for index, processes in sorted(plan.external_compute_processes.items())
        },
        "schedule_manifest": None
        if plan.schedule_manifest is None
        else str(plan.schedule_manifest),
        "sample_id_manifest": str(plan.sample_manifest),
        "sample_id_manifest_sha256": plan.sample_manifest_sha256,
        "runs": rows,
    }
    if launch_error is not None:
        payload["launch_error"] = launch_error
    _atomic_write_json(status_dir / "matrix_status.json", payload)


def merge_semigroup_shards(
    shard_paths: Sequence[Path],
    *,
    manifest_path: Path,
    thresholds: Mapping[str, float],
    visual_pass_by_split: Mapping[str, bool],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    manifest_ids = _read_manifest_ids(manifest_path)
    if len(manifest_ids) != 64:
        raise ValueError("semigroup manifest must contain exactly 64 ordered sample IDs")
    if len(shard_paths) != 4:
        raise ValueError("semigroup merge requires exactly four shard files")
    registered_splits = {"0.25", "0.5", "0.75"}
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    duplicate_ids: list[str] = []
    shard_contracts = []
    shard_rows: list[list[Mapping[str, Any]]] = []
    for shard_index, path in enumerate(shard_paths):
        payload = _read_json(path, "semigroup shard")
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"semigroup shard missing rows: {path}")
        if len(rows) != 16:
            raise ValueError(
                f"missing semigroup rows: shard {shard_index} must contain exactly 16 rows"
            )
        validated_rows: list[Mapping[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("sample_id"), str):
                raise ValueError(f"invalid semigroup row in {path}")
            splits = row.get("splits")
            if not isinstance(splits, Mapping) or set(map(str, splits)) != registered_splits:
                raise ValueError(
                    "every semigroup row must contain exactly the registered split set "
                    "{0.25,0.5,0.75}"
                )
            sample_id = str(row["sample_id"])
            if sample_id in rows_by_id:
                duplicate_ids.append(sample_id)
            rows_by_id[sample_id] = row
            validated_rows.append(row)
        shard_rows.append(validated_rows)
    if duplicate_ids:
        raise ValueError(f"duplicate semigroup sample IDs: {sorted(set(duplicate_ids))!r}")
    for shard_index, (path, rows) in enumerate(zip(shard_paths, shard_rows, strict=True)):
        actual_ids = [str(row["sample_id"]) for row in rows]
        expected_ids = manifest_ids[shard_index::4]
        if actual_ids != expected_ids:
            raise ValueError(
                f"semigroup shard {shard_index} IDs do not match the exact modulo-4 order"
            )
        shard_contracts.append(
            {
                "shard_index": shard_index,
                "sample_count": 16,
                "ordered_sample_id_sha256": _sample_id_digest(actual_ids),
                "semigroup_sha256": _sha256_file(path),
            }
        )
    missing = sorted(set(manifest_ids) - set(rows_by_id))
    extra = sorted(set(rows_by_id) - set(manifest_ids))
    if missing or extra:
        raise ValueError(f"missing semigroup IDs {missing!r}; extra IDs {extra!r}")
    candidates = []
    split_keys = sorted(registered_splits, key=float)
    for split_key in split_keys:
        split_rows = [rows_by_id[sample_id]["splits"][split_key] for sample_id in manifest_ids]
        residuals = [_finite_metric(row["latent_residual"], "latent_residual") for row in split_rows]
        cosines = [
            _finite_metric(row["endpoint_e0_cosine"], "endpoint_e0_cosine")
            for row in split_rows
        ]
        pixel_l1 = [_finite_metric(row["decoded_pixel_l1"], "decoded_pixel_l1") for row in split_rows]
        psnr = [_finite_metric(row["decoded_psnr"], "decoded_psnr") for row in split_rows]
        median = float(statistics.median(residuals))
        p90 = _percentile(residuals, 0.90)
        cosine = float(statistics.median(cosines))
        visual_pass = visual_pass_by_split.get(split_key) is True
        passed = (
            median <= float(thresholds["median"])
            and p90 <= float(thresholds["p90"])
            and cosine >= float(thresholds["endpoint_e0_cosine"])
            and visual_pass
        )
        candidates.append(
            {
                "t_cut": float(split_key),
                "median": median,
                "p90": p90,
                "endpoint_e0_cosine_median": cosine,
                "decoded_pixel_l1_median": float(statistics.median(pixel_l1)),
                "decoded_psnr_median": float(statistics.median(psnr)),
                "visual_pass": visual_pass,
                "passed": passed,
            }
        )
    candidates.sort(key=lambda row: row["t_cut"])
    passed = [row for row in candidates if row["passed"]]
    selected = passed[0]["t_cut"] if passed else None
    return {
        "schema_version": 1,
        "gate_passed": selected is not None,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_t_cut": selected,
        "t_cut": selected,
        "sample_count": len(manifest_ids),
        "sample_id_manifest": str(manifest_path),
        "sample_id_manifest_sha256": _sha256_file(manifest_path),
        "ordered_sample_id_sha256": _sample_id_digest(manifest_ids),
        "shards": shard_contracts,
        "selection_rule": "smallest_numeric_t_cut_passing_all_registered_thresholds",
        "thresholds": dict(thresholds),
        "candidates": candidates,
    }


def load_full_contract(selection_path: Path, visual_review_path: Path) -> dict[str, Any]:
    selection = _load_required_json(selection_path, "locked winner selection")
    visual = _load_required_json(visual_review_path, "visual_review")
    if int(visual.get("reviewed_sample_count", -1)) != 64:
        raise ValueError("visual_review must contain exactly 64 reviewed samples")
    if visual.get("passed") is not True:
        raise ValueError("visual_review must pass before the full run")
    evidence_path = selection_path.parent / "calibration/visual_evidence.json"
    evidence = _load_required_json(evidence_path, "calibration visual evidence")
    _validate_multi_arm_review(visual, evidence, require_passed=True)
    winner = selection.get("winner")
    if not isinstance(winner, Mapping) or not winner.get("config"):
        raise ValueError("selection must contain a locked winner config")
    _require_sha256(winner.get("config_sha256"), "winner config SHA256")
    require_arm_config_digest(winner.get("arm_config_sha256"), "winner arm config SHA256")
    manifest_count = int(selection.get("full_sample_count", 2048))
    manifest_sha = str(selection.get("full_sample_id_manifest_sha256", FULL_MANIFEST_SHA256))
    return {
        "winner": dict(winner),
        "visual_review": visual,
        "manifest_count": manifest_count,
        "manifest_sha256": manifest_sha,
    }


def _build_calibration_visual_evidence(plan: MatrixPlan) -> dict[str, Any]:
    if plan.campaign_root is None:
        raise ValueError("calibration visual evidence requires a campaign root")
    manifest_path = plan.repo_root / CALIBRATION_MANIFEST
    arm_ids = sorted({arm_id for run in plan.runs for arm_id in run.arm_ids})
    arms = {}
    for arm_id in arm_ids:
        arm_root = plan.repo_root / plan.campaign_root / "calibration" / arm_id
        rows = _read_jsonl(arm_root / "per_sample.jsonl")
        visual_rows = [
            {
                "sample_id": row.get("sample_id"),
                "source": row.get("source"),
                "native": row.get("native"),
                "candidate": row.get("generated"),
            }
            for row in rows
        ]
        page_manifest = _read_json(
            arm_root / "contact_sheet_columns.json", f"{arm_id} contact sheet manifest"
        )
        if page_manifest.get("columns") != ["source", "native", "candidate"]:
            raise ValueError(f"calibration arm {arm_id} contact sheet columns disagree")
        arms[arm_id] = build_visual_evidence_contract(
            manifest_path=manifest_path,
            rows=visual_rows,
            pages=page_manifest.get("pages", ()),
            columns=("source", "native", "candidate"),
            expected_count=64,
        )
    payload = {
        "schema_version": 1,
        "sample_count": 64,
        "sample_id_manifest": str(manifest_path),
        "sample_id_manifest_sha256": _sha256_file(manifest_path),
        "arms": arms,
    }
    _write_or_validate_json(
        plan.repo_root / plan.campaign_root / "calibration/visual_evidence.json", payload
    )
    return payload


def _build_semigroup_visual_evidence(plan: MatrixPlan) -> dict[str, Any]:
    manifest_path = plan.repo_root / CALIBRATION_MANIFEST
    manifest_ids = _read_manifest_ids(manifest_path)
    rows_by_id = {}
    for gpu in range(4):
        path = plan.repo_root / ROOT / f"semigroup/shards/shard_{gpu}/per_sample.jsonl"
        for row in _read_jsonl(path):
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in rows_by_id:
                raise ValueError("semigroup visual rows contain an invalid or duplicate sample ID")
            rows_by_id[sample_id] = row
    if set(rows_by_id) != set(manifest_ids):
        raise ValueError("semigroup visual rows do not cover the locked 64-sample manifest")
    arms = {}
    for split_key in ("0.25", "0.5", "0.75"):
        visual_rows = []
        for sample_id in manifest_ids:
            row = rows_by_id[sample_id]
            splits = row.get("semigroup")
            if not isinstance(splits, Mapping) or split_key not in splits:
                raise ValueError(f"semigroup visual row is missing split {split_key}")
            visual_rows.append(
                {
                    "sample_id": sample_id,
                    "source": row.get("source"),
                    "native": row.get("generated"),
                    "candidate": splits[split_key].get("decoded_image"),
                }
            )
        page_dir = (
            plan.repo_root
            / ROOT
            / "semigroup/visual_evidence/contact_sheets"
            / f"t_cut_{split_key.replace('.', 'p')}"
        )
        pages = write_contact_sheets(
            page_dir,
            visual_rows,
            columns=("source", "native", "candidate"),
        )
        arms[split_key] = build_visual_evidence_contract(
            manifest_path=manifest_path,
            rows=visual_rows,
            pages=pages,
            columns=("source", "native", "candidate"),
            expected_count=64,
        )
    payload = {
        "schema_version": 1,
        "sample_count": 64,
        "sample_id_manifest": str(manifest_path),
        "sample_id_manifest_sha256": _sha256_file(manifest_path),
        "arms": arms,
    }
    _write_or_validate_json(plan.repo_root / SEMIGROUP_VISUAL_EVIDENCE, payload)
    return payload


def _validate_multi_arm_review(
    review: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    require_passed: bool,
) -> dict[str, Any]:
    if int(review.get("reviewed_sample_count", -1)) != int(evidence.get("sample_count", -2)):
        raise ValueError("visual review count disagrees with locked visual evidence")
    review_arms = review.get("arms")
    evidence_arms = evidence.get("arms")
    if not isinstance(review_arms, Mapping) or not isinstance(evidence_arms, Mapping):
        raise ValueError("visual review and evidence must contain arm mappings")
    if set(review_arms) != set(evidence_arms):
        raise ValueError("visual review arm IDs must exactly match visual evidence arm IDs")
    normalized = {
        arm_id: validate_visual_review_arm(review_arms[arm_id], evidence_arms[arm_id])
        for arm_id in sorted(evidence_arms)
    }
    all_passed = all(arm["passed"] for arm in normalized.values())
    if review.get("passed") is not all_passed:
        raise ValueError("visual review top-level passed field disagrees with arm decisions")
    if require_passed and not all_passed:
        raise ValueError("visual review must pass before continuing")
    return {
        "reviewed_sample_count": int(evidence["sample_count"]),
        "passed": all_passed,
        "arms": normalized,
    }


def _validate_full_contract(contract: Mapping[str, Any]) -> None:
    winner = contract.get("winner")
    if not isinstance(winner, Mapping) or not winner.get("config"):
        raise ValueError("full phase requires a locked winner config")
    require_arm_config_digest(winner.get("arm_config_sha256"), "winner arm config SHA256")
    visual = contract.get("visual_review")
    if not isinstance(visual, Mapping) or int(visual.get("reviewed_sample_count", -1)) != 64:
        raise ValueError("full phase requires a 64-sample visual review")
    if visual.get("passed") is not True:
        raise ValueError("full phase requires visual review pass")
    if int(contract.get("manifest_count", -1)) != 2048:
        raise ValueError("full phase requires exactly 2048 samples")
    if contract.get("manifest_sha256") != FULL_MANIFEST_SHA256:
        raise ValueError("full phase manifest digest does not match the registered 2048 IDs")
    if _winner_is_fmrg(contract):
        _winner_t_cut(contract)
        schedule_value = winner.get("schedule_manifest")
        if Path(str(schedule_value)) != SCHEDULE_MANIFEST:
            raise ValueError("FMRG winner schedule_manifest must be the locked R8 schedule")
        _require_sha256(winner.get("schedule_manifest_sha256"), "winner schedule manifest SHA256")
        _require_sha256(winner.get("schedule_contract_sha256"), "winner schedule contract SHA256")


def materialize_locked_manifests(repo_root: Path) -> None:
    index_path = repo_root / "data/index/val_face_mixed_e14.jsonl"
    index_rows = _read_jsonl(index_path)
    sample_ids = []
    seen: set[str] = set()
    for row in index_rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("validation index contains an invalid sample_id")
        if sample_id in seen:
            raise ValueError(f"validation index contains duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        sample_ids.append(sample_id)
    for relative, count, expected in (
        (CALIBRATION_MANIFEST, 64, CALIBRATION_MANIFEST_SHA256),
        (FULL_MANIFEST, 2048, FULL_MANIFEST_SHA256),
    ):
        if len(sample_ids) < count:
            raise ValueError(f"validation index has fewer than {count} unique sample IDs")
        content = "".join(
            json.dumps({"sample_id": value}, sort_keys=True, separators=(",", ":")) + "\n"
            for value in sample_ids[:count]
        )
        _write_locked_text(repo_root / relative, content, expected)


def materialize_full_runtime_configs(plan: MatrixPlan) -> None:
    if plan.phase != "full" or plan.full_contract is None:
        return
    native = _load_yaml(plan.repo_root / NATIVE_CONFIG)
    winner_contract = plan.full_contract["winner"]
    winner_path = Path(str(winner_contract["config"]))
    winner_source = _resolve(plan.repo_root, winner_path)
    expected_arm_config_sha256 = require_arm_config_digest(
        winner_contract.get("arm_config_sha256"), "winner arm config SHA256"
    )
    expected_config_sha = winner_contract.get("config_sha256")
    if expected_config_sha is not None and _sha256_file(winner_source) != _require_sha256(
        expected_config_sha, "winner config SHA256"
    ):
        raise ValueError("locked winner config SHA256 disagrees with the selection contract")
    winner = _load_yaml(winner_source)
    if canonical_arm_config_digest(winner) != expected_arm_config_sha256:
        raise ValueError("locked winner canonical arm config SHA256 disagrees with selection")
    if _winner_is_fmrg(plan.full_contract):
        schedule_path = _resolve(plan.repo_root, Path(str(winner_contract["schedule_manifest"])))
        expected_schedule_sha = _require_sha256(
            winner_contract["schedule_manifest_sha256"], "winner schedule manifest SHA256"
        )
        if _sha256_file(schedule_path) != expected_schedule_sha:
            raise ValueError("locked schedule file SHA256 disagrees with the selection contract")
        schedule = _read_json(schedule_path, "locked schedule manifest")
        schedule_contract_sha = _schedule_contract_digest(schedule)
        if schedule_contract_sha != _require_sha256(
            winner_contract["schedule_contract_sha256"], "winner schedule contract SHA256"
        ) or schedule_contract_sha != schedule.get("schedule_contract_sha256"):
            raise ValueError("locked schedule contract digest disagrees with the selection contract")
        t_cut = _winner_t_cut(plan.full_contract)
        if not math.isclose(
            _finite_open_unit(schedule.get("t_cut"), "locked schedule t_cut"),
            float(t_cut),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("locked schedule t_cut disagrees with the selection contract")
        winner.update(
            {
                "t_cut": t_cut,
                "schedule_manifest": str(SCHEDULE_MANIFEST),
                "semigroup_report": str(SEMIGROUP_GATE),
                "semigroup_sample_id_manifest": schedule[
                    "semigroup_sample_id_manifest"
                ],
                "semigroup_sample_id_manifest_sha256": schedule[
                    "semigroup_sample_id_manifest_sha256"
                ],
                "schedule_contract_sha256": schedule_contract_sha,
            }
        )
    runtime_dir = plan.repo_root / ROOT / "full/runtime_configs"
    for arm_id, payload in (("native", native), ("winner", winner)):
        resolved = dict(payload)
        resolved.update(
            {
                "phase": "full",
                "max_samples": 2048,
                "sample_id_manifest": str(FULL_MANIFEST),
                "sample_id_manifest_sha256": FULL_MANIFEST_SHA256,
                "asset_digest_cache": str(ROOT / f"shared/{arm_id}_asset_digests.json"),
                "contact_sheets": False,
            }
        )
        arm_config_sha256 = canonical_arm_config_digest(resolved)
        if arm_id == "winner" and arm_config_sha256 != expected_arm_config_sha256:
            raise ValueError("full runtime winner arm config SHA256 changed during materialization")
        resolved["arm_config_sha256"] = arm_config_sha256
        validate_guidance_config(resolved)
        _write_locked_text(
            runtime_dir / f"{arm_id}.yaml",
            yaml.safe_dump(resolved, sort_keys=False),
            None,
        )


def _native_unguided_runtime_config(plan: MatrixPlan, run: RunContract) -> dict[str, Any]:
    if run.family != "native_unguided" or run.runtime_config is None:
        raise ValueError("native unguided runtime config requires its registered run")
    resolved = _load_yaml(_resolve(plan.repo_root, run.config))
    _require_sampling_seed_1337(resolved, "native unguided calibration config")
    resolved.pop("schedule_manifest", None)
    resolved.update(
        {
            "out_dir": str(run.output_dir),
            "mode": "native",
            "phase": "calibration",
            "sample_id_manifest": str(CALIBRATION_MANIFEST),
            "sample_id_manifest_sha256": CALIBRATION_MANIFEST_SHA256,
            "max_samples": 64,
            "contact_sheets": True,
        }
    )
    validate_guidance_config(resolved)
    return resolved


def materialize_calibration_runtime_configs(plan: MatrixPlan) -> None:
    if plan.phase != "calibrate":
        return
    validate_campaign_path_safety(plan)
    for run in plan.runs:
        if run.runtime_config is None:
            continue
        resolved = _native_unguided_runtime_config(plan, run)
        _write_locked_text(
            _resolve(plan.repo_root, run.runtime_config),
            yaml.safe_dump(resolved, sort_keys=False),
            None,
        )


def build_campaign_contract(plan: MatrixPlan) -> dict[str, Any]:
    if plan.phase != "calibrate" or plan.campaign_id is None or plan.campaign_root is None:
        raise ValueError("campaign contracts are only defined for calibration plans")
    manifest_path = _resolve(plan.repo_root, plan.sample_manifest)
    runs = []
    for run in plan.runs:
        config_path = _resolve(plan.repo_root, run.config)
        config = _load_yaml(config_path)
        sampling_seed = _require_sampling_seed_1337(config, f"{run.arm_ids[0]} config")
        runtime_config = None
        if run.runtime_config is not None:
            resolved_runtime = _native_unguided_runtime_config(plan, run)
            runtime_content = yaml.safe_dump(resolved_runtime, sort_keys=False)
            runtime_config = {
                "path": str(run.runtime_config),
                "sha256": hashlib.sha256(runtime_content.encode("utf-8")).hexdigest(),
                "arm_config_sha256": canonical_arm_config_digest(resolved_runtime),
            }
        sources = []
        for source in run.source_configs:
            source_path = _resolve(plan.repo_root, source)
            sources.append(
                {
                    "path": str(source),
                    "sha256": _sha256_file(source_path),
                }
            )
        runs.append(
            {
                "physical_gpu": run.physical_gpu,
                "family": run.family,
                "arm_ids": list(run.arm_ids),
                "config": str(run.config),
                "config_sha256": _sha256_file(config_path),
                "arm_config_sha256": canonical_arm_config_digest(config),
                "sampling_seed": sampling_seed,
                "runtime_config": runtime_config,
                "source_configs": sources,
                "shard_index": run.shard_index,
                "num_shards": run.num_shards,
                "sample_count": run.sample_count,
                "sample_id_manifest": str(run.sample_manifest),
                "sample_id_manifest_sha256": run.sample_manifest_sha256,
                "output_dir": str(run.output_dir),
                "command": list(run.command),
            }
        )
    schedule = None
    if plan.schedule_manifest is not None:
        schedule_path = _resolve(plan.repo_root, plan.schedule_manifest)
        schedule = {
            "path": str(plan.schedule_manifest),
            "sha256": _sha256_file(schedule_path),
        }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "phase": plan.phase,
        "sample_id_manifest": str(plan.sample_manifest),
        "sample_id_manifest_sha256": plan.sample_manifest_sha256,
        "sample_id_manifest_file_sha256": _sha256_file(manifest_path),
        "schedule_manifest": schedule,
        "semigroup_gate": plan.calibration_gate,
        "runs": runs,
    }
    payload["campaign_contract_sha256"] = _canonical_contract_digest(
        payload, "campaign_contract_sha256"
    )
    return payload


def ensure_campaign_contract(plan: MatrixPlan) -> dict[str, Any]:
    if plan.phase != "calibrate" or plan.campaign_root is None:
        raise ValueError("campaign contract requires a calibration campaign root")
    validate_campaign_path_safety(plan)
    campaigns_root = plan.repo_root / ROOT / "campaigns"
    campaign_root = _resolve(plan.repo_root, plan.campaign_root)
    _require_contained(campaigns_root, campaign_root, "calibration campaign")
    contract_path = campaign_root / "campaign_contract.json"
    if campaign_root.exists() and not campaign_root.is_dir():
        raise FileExistsError(f"calibration campaign root is not a directory: {campaign_root}")
    if contract_path.is_symlink():
        raise ValueError(f"campaign contract must not be a symlink: {contract_path}")
    if not contract_path.exists() and campaign_root.exists():
        unexpected = sorted(
            path.name
            for path in campaign_root.iterdir()
            if path != _resolve(plan.repo_root, plan.lock_path)
        )
        if unexpected:
            raise FileExistsError(
                "refusing calibration campaign root with entries but no contract: "
                f"{unexpected!r}"
            )
    payload = build_campaign_contract(plan)
    _write_immutable_json(contract_path, payload)
    return payload


def _acquire_kernel_lease(path: Path, root: Path, label: str) -> int:
    _reject_symlink_components(root, path.parent, label)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _reject_symlink_components(root, path.parent, label)
    _reject_lstat_symlink(path, label)
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"{label} must not be a symlink: {path}") from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RuntimeError(f"{label} is held by another process: {path}") from exc
            raise
        metadata = f"pid={os.getpid()} label={label}\n".encode()
        os.ftruncate(fd, 0)
        os.write(fd, metadata)
        os.fsync(fd)
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def execution_leases(plan: MatrixPlan) -> Iterator[None]:
    campaign_lock = _resolve(plan.repo_root, plan.lock_path)
    lease_specs = [
        (campaign_lock, plan.repo_root, f"R8 {plan.phase} campaign lease"),
        *[
            (
                HOST_GPU_LOCK_DIR / f"gpu{gpu}.lock",
                HOST_GPU_LOCK_DIR.parent,
                f"R8 physical GPU {gpu} lease",
            )
            for gpu in sorted({run.physical_gpu for run in plan.runs})
        ],
    ]
    leases: list[int] = []
    try:
        for path, root, label in lease_specs:
            leases.append(_acquire_kernel_lease(path, root, label))
        yield
    finally:
        for fd in reversed(leases):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def execute_plan(plan: MatrixPlan) -> int:
    if plan.phase == "calibrate":
        validate_campaign_path_safety(plan)
    with execution_leases(plan):
        if plan.phase == "calibrate":
            validate_campaign_path_safety(plan)
            validate_artifact_paths(plan)
            bound = validate_preflight(plan)
            materialize_locked_manifests(plan.repo_root)
            ensure_campaign_contract(bound)
        else:
            validate_artifact_paths(plan)
            bound = validate_preflight(plan)
            materialize_locked_manifests(plan.repo_root)
        materialize_calibration_runtime_configs(bound)
        materialize_full_runtime_configs(plan)
        validate_campaign_path_safety(bound)
        result = launch_matrix(bound)
        if result != 0:
            return result
        try:
            finalize_result = finalize_phase(bound)
        except Exception as exc:
            _update_matrix_status(
                bound,
                overall_status="failed",
                finalize_error=f"{type(exc).__name__}: {exc}",
            )
            return 1
        if finalize_result == 0:
            _update_matrix_status(bound, overall_status="passed")
        elif finalize_result == 2:
            _update_matrix_status(bound, overall_status="awaiting_direct_visual_review")
        else:
            _update_matrix_status(
                bound,
                overall_status="failed",
                finalize_error=f"finalize_phase returned {finalize_result}",
            )
        return finalize_result


def _update_matrix_status(
    plan: MatrixPlan, *, overall_status: str, finalize_error: str | None = None
) -> None:
    path = _resolve(plan.repo_root, plan.status_dir) / "matrix_status.json"
    payload = _read_json(path, "matrix status")
    payload["overall_status"] = overall_status
    payload["finalized_at"] = _timestamp()
    if finalize_error is not None:
        payload["finalize_error"] = finalize_error
    else:
        payload.pop("finalize_error", None)
    _atomic_write_json(path, payload)


def finalize_phase(plan: MatrixPlan) -> int:
    if plan.phase == "semigroup":
        paths = [
            plan.repo_root / ROOT / f"semigroup/shards/shard_{gpu}/semigroup.json"
            for gpu in range(4)
        ]
        visual_path = plan.repo_root / SEMIGROUP_VISUAL_REVIEW
        evidence = _build_semigroup_visual_evidence(plan)
        visual = _read_json(visual_path, "semigroup visual review") if visual_path.is_file() else {}
        normalized_visual = (
            _validate_multi_arm_review(visual, evidence, require_passed=False)
            if visual_path.is_file()
            else {"arms": {}}
        )
        visual_pass = {
            str(key): bool(value["passed"])
            for key, value in normalized_visual["arms"].items()
        }
        config = _load_yaml(plan.repo_root / SEMIGROUP_CONFIG)
        report = merge_semigroup_shards(
            paths,
            manifest_path=plan.repo_root / CALIBRATION_MANIFEST,
            thresholds=config["semigroup_thresholds"],
            visual_pass_by_split=visual_pass,
            checkpoint_sha256=CHECKPOINT_SHA256,
        )
        if not visual_path.is_file():
            draft = {
                **report,
                "gate_passed": None,
                "selected_t_cut": None,
                "t_cut": None,
                "status": "awaiting_direct_visual_review",
            }
            _atomic_write_json(plan.repo_root / SEMIGROUP_GATE_DRAFT, draft)
            return 2
        gate_path = plan.repo_root / SEMIGROUP_GATE
        _atomic_write_json(gate_path, report, exclusive=True)
        if report["gate_passed"]:
            _atomic_write_json(
                plan.repo_root / SCHEDULE_MANIFEST,
                _schedule_payload(
                    report,
                    semigroup_report_sha256=_sha256_file(gate_path),
                ),
                exclusive=True,
            )
        return 0
    if plan.phase == "calibrate":
        if plan.campaign_root is None:
            raise ValueError("calibration finalization requires a campaign root")
        evidence = _build_calibration_visual_evidence(plan)
        visual_path = plan.repo_root / plan.campaign_root / "visual_review.json"
        if not visual_path.is_file():
            return 2
        visual = _read_json(visual_path, "calibration visual review")
        _validate_multi_arm_review(visual, evidence, require_passed=True)
        return 0
    if plan.phase == "full":
        return finalize_full_outputs(plan)
    return 0


def finalize_full_outputs(plan: MatrixPlan) -> int:
    arm_completions: dict[str, Mapping[str, Any]] = {}
    for arm_id in ("native", "winner"):
        combined = plan.repo_root / ROOT / f"full/merged/{arm_id}"
        arm_completions[arm_id] = _merge_full_arm(
            plan.repo_root,
            arm_id,
            combined,
            expected_arm_config_sha256=_full_arm_config_digest(plan, arm_id),
        )
        quality_path = combined / "quality.json"
        command = build_quality_command(
            python=plan.python,
            output_dir=ROOT / f"full/merged/{arm_id}",
            manifest=FULL_MANIFEST,
        )
        subprocess.run(
            command,
            cwd=plan.repo_root,
            check=True,
            env={**os.environ, "PYTHONPATH": "src"},
        )
        _read_json(quality_path, f"full {arm_id} quality result")
    evidence = _build_full_visual_evidence(plan)
    visual_path = plan.repo_root / FULL_VISUAL_REVIEW
    if not visual_path.is_file():
        return 2
    visual = _read_json(visual_path, "full visual review")
    normalized_visual = _validate_multi_arm_review(visual, evidence, require_passed=True)
    completion = {
        "schema_version": 1,
        "status": "complete",
        "sample_id_manifest_sha256": _sha256_file(plan.repo_root / FULL_MANIFEST),
        "visual_evidence_sha256": _sha256_file(plan.repo_root / FULL_VISUAL_EVIDENCE),
        "visual_review_sha256": _sha256_file(visual_path),
        "visual_review": normalized_visual,
        "arms": {
            arm_id: {
                "merge_contract_sha256": arm_completions[arm_id]["merge_contract_sha256"],
                "arm_config_sha256": arm_completions[arm_id]["arm_config_sha256"],
                "quality_sha256": _sha256_file(
                    plan.repo_root / ROOT / f"full/merged/{arm_id}/quality.json"
                ),
            }
            for arm_id in ("native", "winner")
        },
    }
    _write_or_validate_json(
        plan.repo_root / ROOT / "full/finalization_completion.json", completion
    )
    return 0


def _build_full_visual_evidence(plan: MatrixPlan) -> dict[str, Any]:
    manifest_path = plan.repo_root / CALIBRATION_MANIFEST
    review_ids = _read_manifest_ids(manifest_path)
    full_ids = _read_manifest_ids(plan.repo_root / FULL_MANIFEST)
    if full_ids[:64] != review_ids:
        raise ValueError("full visual review IDs must be the locked first 64 full-manifest IDs")
    native_rows = {
        str(row.get("sample_id", "")): row
        for row in _read_jsonl(plan.repo_root / ROOT / "full/merged/native/per_sample.jsonl")
    }
    winner_rows = {
        str(row.get("sample_id", "")): row
        for row in _read_jsonl(plan.repo_root / ROOT / "full/merged/winner/per_sample.jsonl")
    }
    visual_rows = []
    for sample_id in review_ids:
        native = native_rows.get(sample_id)
        winner = winner_rows.get(sample_id)
        if native is None or winner is None or native.get("source") != winner.get("source"):
            raise ValueError("full visual native/winner source bindings disagree")
        visual_rows.append(
            {
                "sample_id": sample_id,
                "source": winner.get("source"),
                "native": native.get("generated"),
                "candidate": winner.get("generated"),
            }
        )
    visual_root = plan.repo_root / ROOT / "full/visual_evidence"
    _reject_symlink_components(plan.repo_root, visual_root, "full visual evidence output")
    _reject_symlink_tree(visual_root, "full visual evidence output")
    page_dir = visual_root / "contact_sheets"
    pages = write_contact_sheets(
        page_dir,
        visual_rows,
        columns=("source", "native", "candidate"),
    )
    winner_id = str(plan.full_contract["winner"]["arm_id"])
    arm = build_visual_evidence_contract(
        manifest_path=manifest_path,
        rows=visual_rows,
        pages=pages,
        columns=("source", "native", "candidate"),
        expected_count=64,
    )
    payload = {
        "schema_version": 1,
        "sample_count": 64,
        "sample_id_manifest": str(manifest_path),
        "sample_id_manifest_sha256": _sha256_file(manifest_path),
        "full_sample_id_manifest": str(plan.repo_root / FULL_MANIFEST),
        "full_sample_id_manifest_sha256": _sha256_file(plan.repo_root / FULL_MANIFEST),
        "arms": {winner_id: arm},
    }
    evidence_path = plan.repo_root / FULL_VISUAL_EVIDENCE
    _reject_symlink_components(plan.repo_root, evidence_path, "full visual evidence contract")
    if evidence_path.is_symlink():
        raise ValueError(f"full visual evidence contract must not be a symlink: {evidence_path}")
    _write_or_validate_json(evidence_path, payload)
    return payload


def validate_campaign_path_safety(plan: MatrixPlan) -> None:
    if plan.phase != "calibrate":
        return
    if plan.campaign_root is None:
        raise ValueError("calibration campaign path safety requires a campaign root")
    campaigns_root = plan.repo_root / ROOT / "campaigns"
    campaign_root = _resolve(plan.repo_root, plan.campaign_root)
    _reject_symlink_components(plan.repo_root, campaigns_root, "calibration campaigns root")
    _reject_symlink_components(plan.repo_root, campaign_root, "calibration campaign")
    _require_contained(campaigns_root, campaign_root, "calibration campaign")
    owned_paths = [
        _resolve(plan.repo_root, plan.status_dir),
        _resolve(plan.repo_root, plan.lock_path),
        campaign_root / "campaign_contract.json",
        campaign_root / "visual_review.json",
        campaign_root / "calibration/visual_evidence.json",
    ]
    for run in plan.runs:
        owned_paths.extend(
            (
                _resolve(plan.repo_root, run.output_dir),
                _resolve(plan.repo_root, run.log_path),
            )
        )
        if run.runtime_config is not None:
            owned_paths.append(_resolve(plan.repo_root, run.runtime_config))
    for path in owned_paths:
        _reject_symlink_components(plan.repo_root, path, "calibration campaign artifact")
        _require_contained(campaign_root, path, "calibration campaign artifact")


def _reject_symlink_components(root: Path, path: Path, label: str) -> None:
    root_absolute = root.absolute()
    path_absolute = path.absolute()
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository: {path}") from exc
    current = root_absolute
    _reject_lstat_symlink(current, label)
    for part in relative.parts:
        current /= part
        _reject_lstat_symlink(current, label)


def _reject_lstat_symlink(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode):
        raise ValueError(f"{label} has a symlink path component: {path}")


def _reject_symlink_tree(root: Path, label: str) -> None:
    if root.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {root}")
    if not root.exists():
        return
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*dirnames, *filenames):
            path = base / name
            if path.is_symlink():
                raise ValueError(f"{label} contains a symlink: {path}")


def _require_contained(root: Path, path: Path, label: str) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError(f"{label} escapes its owned directory: {path}") from exc


def _atomic_copy_file(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as source_handle, temporary.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_owned_temporaries(directory: Path, target_names: Sequence[str]) -> None:
    if not directory.is_dir():
        return
    prefixes = tuple(f".{name}." for name in target_names)
    for path in directory.iterdir():
        if path.name.startswith(prefixes) and path.name.endswith(".tmp"):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"owned temporary output has an invalid type: {path}")
            path.unlink()


def _write_recoverable_json(
    path: Path, payload: Mapping[str, Any], *, completed: bool
) -> None:
    expected = json.loads(json.dumps(payload, allow_nan=False))
    try:
        matches = path.is_file() and _read_json(path, path.name) == expected
    except (ValueError, json.JSONDecodeError):
        matches = False
    if matches:
        return
    if completed:
        raise ValueError(f"existing owned artifact disagrees with its contract: {path}")
    _atomic_write_json(path, payload)


def _write_recoverable_text(path: Path, content: str, *, completed: bool) -> None:
    try:
        matches = path.is_file() and path.read_text(encoding="utf-8") == content
    except UnicodeError:
        matches = False
    if matches:
        return
    if completed:
        raise ValueError(f"existing owned artifact disagrees with its contract: {path}")
    _atomic_write_bytes(path, content.encode("utf-8"))


def _merge_full_arm(
    repo_root: Path,
    arm_id: str,
    combined: Path,
    *,
    expected_arm_config_sha256: str,
) -> dict[str, Any]:
    expected_arm_config_sha256 = require_arm_config_digest(
        expected_arm_config_sha256, f"full {arm_id} arm config SHA256"
    )
    expected_ids = _read_manifest_ids(repo_root / FULL_MANIFEST)
    shards_root = repo_root / ROOT / "full/shards"
    _reject_symlink_components(repo_root, shards_root, f"full {arm_id} shard tree")
    _reject_symlink_tree(shards_root, f"full {arm_id} shard tree")
    merge_root = repo_root / ROOT / "full/merged"
    _reject_symlink_components(repo_root, merge_root, f"full {arm_id} merge tree")
    _reject_symlink_tree(merge_root, f"full {arm_id} merge tree")
    _require_contained(merge_root, combined, f"full {arm_id} merged output")
    rows_by_id: dict[str, dict[str, Any]] = {}
    shard_root_by_id: dict[str, Path] = {}
    shard_contracts = []
    generation_contracts = []
    for gpu in range(4):
        shard_root = repo_root / ROOT / f"full/shards/shard_{gpu}/{arm_id}"
        _reject_symlink_tree(shard_root, f"full {arm_id} shard {gpu}")
        completion_path = shard_root / "completion.json"
        completion = _read_json(completion_path, f"full {arm_id} shard completion")
        if completion.get("status") != "complete":
            raise ValueError(f"full {arm_id} shard {gpu} completion is not complete")
        path = shard_root / "per_sample.jsonl"
        shard_rows = _read_jsonl(path)
        expected_shard_ids = expected_ids[gpu::4]
        actual_shard_ids = [str(row.get("sample_id", "")) for row in shard_rows]
        if actual_shard_ids != expected_shard_ids:
            raise ValueError(
                f"full {arm_id} shard {gpu} does not match the exact modulo-4 manifest order"
            )
        if int(completion.get("sample_count", -1)) != len(shard_rows):
            raise ValueError(f"full {arm_id} shard {gpu} completion sample count disagrees")
        if completion.get("arm_config_sha256") != expected_arm_config_sha256:
            raise ValueError(f"full {arm_id} shard {gpu} locked arm config SHA256 disagrees")
        generation_path = shard_root / "generation_result.json"
        generation = _read_json(generation_path, f"full {arm_id} shard generation")
        if generation.get("status") != "complete" or generation.get("sample_count") != len(
            shard_rows
        ):
            raise ValueError(f"full {arm_id} shard {gpu} generation contract is incomplete")
        if generation.get("sample_id_sha256") != _sample_id_digest(actual_shard_ids):
            raise ValueError(f"full {arm_id} shard {gpu} generation sample digest disagrees")
        checkpoint = generation.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != CHECKPOINT_SHA256:
            raise ValueError(f"full {arm_id} shard {gpu} checkpoint contract disagrees")
        config = generation.get("config")
        if not isinstance(config, Mapping) or config.get("sampling_seed", config.get("seed")) != 1337:
            raise ValueError(f"full {arm_id} shard {gpu} sampling seed disagrees")
        if config.get("mode") != generation.get("mode"):
            raise ValueError(f"full {arm_id} shard {gpu} generation/config modes disagree")
        if generation.get("arm_config_sha256") != expected_arm_config_sha256 or (
            canonical_arm_config_digest(config) != expected_arm_config_sha256
        ):
            raise ValueError(f"full {arm_id} shard {gpu} canonical arm config SHA256 disagrees")
        generation_contracts.append(generation)
        shard_contracts.append(
            {
                "shard_index": gpu,
                "completion_sha256": _sha256_file(completion_path),
                "per_sample_sha256": _sha256_file(path),
                "generation_result_sha256": _sha256_file(generation_path),
                "ordered_sample_id_sha256": _sample_id_digest(actual_shard_ids),
            }
        )
        for row in shard_rows:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in rows_by_id:
                raise ValueError(f"duplicate or invalid full {arm_id} sample ID: {sample_id!r}")
            rows_by_id[sample_id] = row
            shard_root_by_id[sample_id] = shard_root
    if set(rows_by_id) != set(expected_ids):
        raise ValueError(f"full {arm_id} shards do not cover exactly the locked manifest IDs")
    first_generation = generation_contracts[0]
    for generation in generation_contracts[1:]:
        for field in ("mode", "checkpoint", "schedule", "config"):
            if generation.get(field) != first_generation.get(field):
                raise ValueError(f"full {arm_id} shard generation {field} contracts disagree")
    source_rows = []
    for sample_id in expected_ids:
        source_value = rows_by_id[sample_id].get(
            "generated", rows_by_id[sample_id].get("generated_image_path")
        )
        source = Path(str(source_value))
        if not source.is_absolute():
            source = repo_root / source
        _require_contained(shard_root_by_id[sample_id], source, f"full {arm_id} source image")
        if source.is_symlink():
            raise ValueError(f"full {arm_id} source image must not be a symlink: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"full {arm_id} generated image does not exist: {source}")
        source_rows.append((sample_id, source, _sha256_file(source)))
    source_manifest_sha256 = hashlib.sha256(
        "".join(
            f"{sample_id}\t{source}\t{digest}\n"
            for sample_id, source, digest in source_rows
        ).encode("utf-8")
    ).hexdigest()
    merge_contract = {
        "schema_version": 1,
        "arm_id": arm_id,
        "arm_config_sha256": expected_arm_config_sha256,
        "sample_count": len(expected_ids),
        "sample_id_manifest_sha256": _sha256_file(repo_root / FULL_MANIFEST),
        "ordered_sample_id_sha256": _sample_id_digest(expected_ids),
        "ordered_source_image_manifest_sha256": source_manifest_sha256,
        "shards": shard_contracts,
    }
    merge_contract["merge_contract_sha256"] = _canonical_contract_digest(
        merge_contract, "merge_contract_sha256"
    )
    _reject_symlink_tree(combined, f"full {arm_id} merged output")
    if combined.exists() and not combined.is_dir():
        raise FileExistsError(f"merged full arm output is not a directory: {combined}")
    allowed_top_level = {
        "merge_contract.json",
        "generated_images",
        "per_sample.jsonl",
        "generation_result.json",
        "completion.json",
        "quality.json",
    }
    if combined.is_dir():
        _remove_owned_temporaries(combined, tuple(allowed_top_level))
        extras = sorted(path.name for path in combined.iterdir() if path.name not in allowed_top_level)
        if extras:
            raise FileExistsError(f"merged full arm contains unowned entries: {extras!r}")
    completion_path = combined / "completion.json"
    completed = False
    if completion_path.is_file():
        try:
            existing_completion = _read_json(completion_path, f"full {arm_id} merge completion")
        except ValueError:
            existing_completion = None
        if existing_completion is not None:
            if (
                existing_completion.get("status") != "complete"
                or existing_completion.get("arm_id") != arm_id
                or existing_completion.get("arm_config_sha256") != expected_arm_config_sha256
                or existing_completion.get("merge_contract_sha256")
                != merge_contract["merge_contract_sha256"]
            ):
                raise ValueError(f"existing full {arm_id} completion contract disagrees")
            completed = True
    combined.mkdir(parents=True, exist_ok=True)
    _write_recoverable_json(combined / "merge_contract.json", merge_contract, completed=completed)
    generated = combined / "generated_images"
    _require_contained(combined, generated, f"full {arm_id} generated output")
    generated.mkdir(parents=True, exist_ok=True)
    expected_image_names = {
        f"{ordinal:06d}{source.suffix.lower() or '.png'}"
        for ordinal, (_, source, _) in enumerate(source_rows)
    }
    _remove_owned_temporaries(generated, tuple(expected_image_names))
    generated_extras = sorted(
        path.name for path in generated.iterdir() if path.name not in expected_image_names
    )
    if generated_extras:
        raise FileExistsError(f"merged full arm generated images contain unowned entries: {generated_extras!r}")
    output_rows = []
    image_manifest_lines = []
    for ordinal, (sample_id, source, source_sha256) in enumerate(source_rows):
        row = rows_by_id[sample_id]
        suffix = source.suffix.lower() or ".png"
        target = generated / f"{ordinal:06d}{suffix}"
        _require_contained(generated, target, f"full {arm_id} merged image")
        if target.is_symlink():
            raise ValueError(f"merged full arm image must not be a symlink: {target}")
        if target.exists():
            if not target.is_file() or _sha256_file(target) != source_sha256:
                if completed:
                    raise ValueError(f"existing merged image disagrees with its source: {target}")
                _atomic_copy_file(source, target)
        else:
            _atomic_copy_file(source, target)
        output_rows.append({**row, "generated": str(target)})
        image_manifest_lines.append(f"{sample_id}\t{target.name}\t{source_sha256}\n")
    per_sample_content = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in output_rows
    )
    _write_recoverable_text(combined / "per_sample.jsonl", per_sample_content, completed=completed)
    candidate_values = [
        _finite_metric(row.get("candidate_cosine"), "candidate cosine") for row in output_rows
    ]
    native_values = [
        _finite_metric(row.get("native_cosine"), "native cosine") for row in output_rows
    ]
    merged_generation = {
        "schema_version": 1,
        "status": "complete",
        "mode": first_generation["mode"],
        "arm_config_sha256": expected_arm_config_sha256,
        "checkpoint": first_generation["checkpoint"],
        "sample_count": len(expected_ids),
        "sample_id_sha256": _sample_id_digest(expected_ids),
        "sample_id_manifest": str(repo_root / FULL_MANIFEST),
        "sample_id_manifest_sha256": _sha256_file(repo_root / FULL_MANIFEST),
        "seed": 1337,
        "schedule": first_generation.get("schedule"),
        "config": first_generation["config"],
        "cosine": {
            "candidate_e0_target": _finite_summary(candidate_values),
            "native_e0_target": _finite_summary(native_values),
        },
        "shards": shard_contracts,
    }
    _write_recoverable_json(
        combined / "generation_result.json", merged_generation, completed=completed
    )
    completion = {
        "schema_version": 1,
        "status": "complete",
        "arm_id": arm_id,
        "arm_config_sha256": expected_arm_config_sha256,
        "sample_count": len(expected_ids),
        "merge_contract_sha256": merge_contract["merge_contract_sha256"],
        "per_sample_sha256": hashlib.sha256(per_sample_content.encode("utf-8")).hexdigest(),
        "generation_result_sha256": _sha256_file(combined / "generation_result.json"),
        "ordered_image_manifest_sha256": hashlib.sha256(
            "".join(image_manifest_lines).encode("utf-8")
        ).hexdigest(),
    }
    _write_recoverable_json(completion_path, completion, completed=completed)
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    phases = ("semigroup", "calibrate", "full") if args.phase == "all" else (args.phase,)
    if not args.execute:
        states = query_gpu_states()
        rendered = []
        for phase in phases:
            phase_args = argparse.Namespace(**{**vars(args), "phase": phase})
            if phase == "calibrate":
                plan = build_matrix_plan(phase_args, semigroup_gate=_dry_run_gate())
            elif phase == "full":
                plan = build_matrix_plan(phase_args, full_contract=_dry_run_full_contract())
            else:
                plan = build_matrix_plan(phase_args)
            plan = validate_preflight(plan, gpu_states=states)
            rendered.append(render_dry_run(plan))
        print("\n".join(rendered), end="")
        return 0
    for phase in phases:
        phase_args = argparse.Namespace(**{**vars(args), "phase": phase})
        plan = build_matrix_plan(phase_args)
        result = execute_plan(plan)
        if result != 0:
            return result
    return 0


def _dry_run_gate() -> dict[str, Any]:
    return {
        "gate_passed": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_t_cut": 0.25,
        "sample_id_manifest_sha256": CALIBRATION_MANIFEST_SHA256,
    }


def _dry_run_full_contract() -> dict[str, Any]:
    config = _load_yaml(Path(__file__).resolve().parents[1] / NOISE_CONFIGS[0])
    return {
        "winner": {
            "arm_id": "dry_run_winner",
            "config": str(NOISE_CONFIGS[0]),
            "arm_config_sha256": canonical_arm_config_digest(config),
        },
        "visual_review": {"reviewed_sample_count": 64, "passed": True},
        "manifest_count": 2048,
        "manifest_sha256": FULL_MANIFEST_SHA256,
    }


def _validate_gate_identity(gate: Mapping[str, Any]) -> None:
    if gate.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("semigroup gate checkpoint SHA256 mismatch")
    manifest_digest = gate.get("sample_id_manifest_sha256")
    if manifest_digest not in (None, CALIBRATION_MANIFEST_SHA256):
        raise ValueError("semigroup gate sample-ID manifest digest mismatch")
    if gate.get("gate_passed") is True:
        value = gate.get("selected_t_cut", gate.get("t_cut"))
        _finite_open_unit(value, "semigroup selected t_cut")


def _winner_is_fmrg(contract: Mapping[str, Any]) -> bool:
    value = str(contract.get("winner", {}).get("mode", ""))
    return value in {"official_head_current_xt", "paper_algorithm_split"}


def _winner_t_cut(contract: Mapping[str, Any]) -> float | None:
    if not _winner_is_fmrg(contract):
        return None
    value = contract.get("winner", {}).get("t_cut")
    if value is None:
        raise ValueError("FMRG winner is missing the locked t_cut")
    return _finite_open_unit(value, "winner t_cut")


def _schedule_payload(
    report: Mapping[str, Any], *, semigroup_report_sha256: str
) -> dict[str, Any]:
    t_cut = _finite_open_unit(report["selected_t_cut"], "selected t_cut")
    guided = [1.0 - index * (1.0 - t_cut) / 3.0 for index in range(4)]
    guided[-1] = t_cut
    payload = {
        "schema_version": 2,
        "gate_passed": True,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "semigroup_report": str(SEMIGROUP_GATE),
        "semigroup_report_sha256": semigroup_report_sha256,
        "semigroup_sample_id_manifest": str(report["sample_id_manifest"]),
        "semigroup_sample_id_manifest_sha256": report["sample_id_manifest_sha256"],
        "t_cut": t_cut,
        "guided_steps": 3,
        "guided_times": guided,
        "unguided_tail_intervals": 2,
        "unguided_times": [t_cut, t_cut / 2.0, 0.0],
        "selection_rule": report["selection_rule"],
    }
    payload["schedule_contract_sha256"] = _schedule_contract_digest(payload)
    return payload


def _schedule_contract_digest(payload: Mapping[str, Any]) -> str:
    contract = dict(payload)
    contract.pop("schedule_contract_sha256", None)
    return _canonical_json_sha256(contract)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"R8 YAML config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"R8 YAML config must contain a mapping: {path}")
    return payload


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON at {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _load_required_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required {label} does not exist: {path}")
    return _read_json(path, label)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSONL does not exist: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def _read_manifest_ids(path: Path) -> list[str]:
    ids = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"manifest contains invalid sample_id: {sample_id!r}")
        if sample_id in seen:
            raise ValueError(f"manifest contains duplicate sample_id: {sample_id!r}")
        seen.add(sample_id)
        ids.append(sample_id)
    if not ids:
        raise ValueError(f"manifest contains no IDs: {path}")
    return ids


def _write_locked_text(path: Path, content: str, expected_sha256: str | None) -> None:
    digest = hashlib.sha256(content.encode()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"locked content digest mismatch for {path}: {digest}")
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite disagreeing locked file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, 0o644)
    try:
        os.write(fd, content.encode())
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = False) -> None:
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to overwrite existing R8 artifact: {path}")
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    expected = json.loads(json.dumps(payload, allow_nan=False))
    if path.exists():
        if not path.is_file() or _read_json(path, path.name) != expected:
            raise ValueError(f"existing owned artifact disagrees with its contract: {path}")
        return
    _atomic_write_json(path, payload, exclusive=True)


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    expected = json.loads(json.dumps(payload, allow_nan=False))
    if path.exists():
        if not path.is_file() or _read_json(path, path.name) != expected:
            raise ValueError(f"existing owned artifact disagrees with its contract: {path}")
        return
    content = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or _read_json(path, path.name) != expected:
                raise ValueError(f"existing owned artifact disagrees with its contract: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate_text(path: Path, content: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            raise ValueError(f"existing owned artifact disagrees with its contract: {path}")
        return
    _write_locked_text(path, content, None)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_contract_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    contract = dict(payload)
    contract.pop(digest_field, None)
    return _canonical_json_sha256(contract)


def _sample_id_digest(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")
    ).hexdigest()


def _finite_metric(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite semigroup {label}: {value!r}")
    return result


def _finite_open_unit(value: Any, label: str) -> float:
    result = _finite_metric(value, label)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{label} must be within (0,1), got {result!r}")
    return result


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return text


def _require_sampling_seed_1337(config: Mapping[str, Any], label: str) -> int:
    value = config.get("sampling_seed")
    if isinstance(value, bool) or not isinstance(value, int) or value != 1337:
        raise ValueError(f"{label} sampling_seed must be the registered integer 1337")
    return value


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of empty values")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _finite_summary(values: Sequence[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("cannot summarize empty or non-finite full cosine values")
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _config_stem(path: Path) -> str:
    value = path.stem
    return value.removeprefix("r8_meanflow_")


if __name__ == "__main__":
    raise SystemExit(main())
