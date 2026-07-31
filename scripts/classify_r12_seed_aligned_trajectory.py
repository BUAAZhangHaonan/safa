#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from safa.evaluation.triangle_screening import ArmResult, evaluate_arms


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARATION = REPO_ROOT / "artifacts/r12_seed_aligned_trajectory/preparation_v1"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/r12_seed_aligned_trajectory/results_v1"
RUNS_ROOT = REPO_ROOT / "artifacts/r12_seed_aligned_trajectory/runs_v1"
EVALUATION_ROOT = REPO_ROOT / "artifacts/r12_seed_aligned_trajectory/evaluation_v1"
DATASET_ARMS = {
    "regular32": ("u12_regular32", "u16_regular32"),
    "sharpness_tail32": ("u12_tail32", "u16_tail32"),
}
FACE_PRIVACY_GATES = {
    "source_exact_one",
    "native_exact_one",
    "candidate_exact_one",
    "arcface_privacy_available",
    "arcface_delta",
}
REPRESENTATION_GATES = {"e0", "delta_e0", "delta_edev"}
QUALITY_GATES = {"niqe", "sharpness"}


class R12ClassificationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise R12ClassificationError(f"{label} is missing: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            R12ClassificationError(f"{label} contains non-finite JSON: {token}")
        ),
    )
    if not isinstance(value, Mapping):
        raise R12ClassificationError(f"{label} must be an object")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        raise R12ClassificationError(f"{label} is missing: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise R12ClassificationError(f"{label} row {line_number} is invalid")
        value = json.loads(
            line,
            parse_constant=lambda token: (_ for _ in ()).throw(
                R12ClassificationError(
                    f"{label} row {line_number} contains non-finite JSON: {token}"
                )
            ),
        )
        if not isinstance(value, dict):
            raise R12ClassificationError(f"{label} row {line_number} is invalid")
        rows.append(value)
    return rows


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R12ClassificationError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise R12ClassificationError(f"{label} is not finite")
    return result


def unique_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise R12ClassificationError(f"{label} sample IDs are invalid")
        result[sample_id] = row
    if len(result) != 32:
        raise R12ClassificationError(f"{label} requires exactly 32 rows")
    return result


def require_trajectory_prefix(
    u12_route: Mapping[str, Any], u16_route: Mapping[str, Any], label: str
) -> None:
    if u12_route.get("loss_history") != u16_route.get("loss_history", [])[:13]:
        raise R12ClassificationError(
            f"protocol_binding_failure: {label} u12 is not the exact u16 prefix"
        )
    if u12_route.get("initial_norm") != u16_route.get("initial_norm"):
        raise R12ClassificationError(
            f"protocol_binding_failure: {label} initial noise differs"
        )


def quality_rows(path: Path, label: str) -> dict[str, Mapping[str, Any]]:
    value = read_json(path, label)
    if value.get("metrics") != ["niqe", "sharpness"]:
        raise R12ClassificationError(f"{label} must contain only NIQE/sharpness")
    per_sample = value.get("per_sample_metrics")
    if (
        not isinstance(per_sample, Mapping)
        or per_sample.get("metric_fields") != ["niqe", "sharpness"]
        or not isinstance(per_sample.get("rows"), list)
    ):
        raise R12ClassificationError(f"{label} per-sample metrics differ")
    return unique_rows(per_sample["rows"], label)


def arcface_rows(path: Path, label: str) -> dict[str, Mapping[str, Any]]:
    value = read_json(path, label)
    if (
        value.get("contract_type") != "safa_r11_arcface_evaluator_output_v1"
        or not isinstance(value.get("result"), list)
    ):
        raise R12ClassificationError(f"{label} ArcFace contract differs")
    return unique_rows(value["result"], label)


def _native_binding(preparation: Mapping[str, Any], dataset_id: str) -> tuple[Path, dict[str, Mapping[str, Any]]]:
    bindings = preparation.get("formal_native_bindings")
    binding = bindings.get(dataset_id) if isinstance(bindings, Mapping) else None
    if not isinstance(binding, Mapping):
        raise R12ClassificationError(f"{dataset_id} native binding is missing")
    path = Path(str(binding.get("path", ""))).resolve()
    if sha256(path) != binding.get("sha256"):
        raise R12ClassificationError(f"{dataset_id} native binding differs")
    return path, unique_rows(read_jsonl(path, f"{dataset_id} native binding"), dataset_id)


def materialize_dataset(
    preparation: Mapping[str, Any], dataset_id: str
) -> list[ArmResult]:
    arm_ids = DATASET_ARMS[dataset_id]
    binding_path, native = _native_binding(preparation, dataset_id)
    selection = (
        REPO_ROOT / "artifacts/r10_triangle_exploration/preparation_v1/prefix32.jsonl"
        if dataset_id == "regular32"
        else REPO_ROOT
        / "artifacts/r11_initial_noise_sharpness_probe/preparation_v1/tail32.jsonl"
    )
    expected_ids = [row["sample_id"] for row in read_jsonl(selection, dataset_id)]
    if list(native) != expected_ids:
        raise R12ClassificationError(f"{dataset_id} native order differs")
    native_quality = quality_rows(
        EVALUATION_ROOT / dataset_id / "quality/native/quality.json",
        f"{dataset_id} native quality",
    )
    inputs = []
    trajectory_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    for arm_id in arm_ids:
        per_sample = unique_rows(
            read_jsonl(RUNS_ROOT / arm_id / "per_sample.jsonl", arm_id), arm_id
        )
        if list(per_sample) != expected_ids:
            raise R12ClassificationError(f"{arm_id} generation order differs")
        trajectory_rows[arm_id] = per_sample
        candidate_quality = quality_rows(
            EVALUATION_ROOT / dataset_id / "quality" / arm_id / "quality.json",
            f"{arm_id} quality",
        )
        arcface = arcface_rows(
            EVALUATION_ROOT / dataset_id / "arcface" / arm_id / "result.json",
            f"{arm_id} ArcFace",
        )
        rows = []
        expected_updates = 12 if arm_id.startswith("u12_") else 16
        for sample_id in expected_ids:
            generated = per_sample[sample_id]
            formal = native[sample_id]
            regenerated_native = Path(str(generated.get("native", ""))).resolve()
            if (
                not regenerated_native.is_file()
                or sha256(regenerated_native) != formal.get("formal_native_sha256")
            ):
                raise R12ClassificationError(
                    f"{arm_id} regenerated native differs from formal seed7919: {sample_id}"
                )
            if not math.isclose(
                finite(generated.get("native_cosine"), f"{arm_id} native e0"),
                finite(formal.get("e0_cosine"), f"{dataset_id} formal e0"),
                rel_tol=0.0,
                abs_tol=1.0e-7,
            ) or not math.isclose(
                finite(generated.get("native_edev_cosine"), f"{arm_id} native edev"),
                finite(formal.get("edev_cosine"), f"{dataset_id} formal edev"),
                rel_tol=0.0,
                abs_tol=1.0e-7,
            ):
                raise R12ClassificationError(
                    f"{arm_id} native representation differs from formal seed7919: {sample_id}"
                )
            route = generated.get("route_diagnostics")
            if (
                not isinstance(route, Mapping)
                or route.get("projection") != "fixed_radius"
                or route.get("eta") != 0.5
                or route.get("num_updates") != expected_updates
                or generated.get("candidate_nfe") != expected_updates + 1
                or not isinstance(route.get("loss_history"), list)
                or len(route["loss_history"]) != expected_updates + 1
            ):
                raise R12ClassificationError(f"{arm_id} trajectory contract differs")
            initial_norm = finite(route.get("initial_norm"), f"{arm_id} initial norm")
            final_norm = finite(route.get("final_norm"), f"{arm_id} final norm")
            if not math.isclose(initial_norm, final_norm, rel_tol=1.0e-6, abs_tol=1.0e-5):
                raise R12ClassificationError(f"{arm_id} fixed radius differs")
            qn = native_quality[sample_id]
            qc = candidate_quality[sample_id]
            af = arcface[sample_id]
            rows.append(
                {
                    "sample_id": sample_id,
                    "native_e0": finite(formal["e0_cosine"], "native_e0"),
                    "candidate_e0": finite(generated.get("candidate_cosine"), "candidate_e0"),
                    "native_edev": finite(formal["edev_cosine"], "native_edev"),
                    "candidate_edev": finite(generated.get("edev_cosine"), "candidate_edev"),
                    "native_niqe": finite(qn.get("niqe"), "native_niqe"),
                    "candidate_niqe": finite(qc.get("niqe"), "candidate_niqe"),
                    "native_sharpness": finite(qn.get("sharpness"), "native_sharpness"),
                    "candidate_sharpness": finite(qc.get("sharpness"), "candidate_sharpness"),
                    "source_face_count": af.get("source_face_count"),
                    "native_face_count": af.get("native_face_count"),
                    "candidate_face_count": af.get("candidate_face_count"),
                    "source_native_cosine": af.get("source_native_cosine"),
                    "source_candidate_cosine": af.get("source_candidate_cosine"),
                }
            )
        inputs.append({"arm_id": arm_id, "rows": rows, "fid": None, "kid": None})
    u12_rows = trajectory_rows[arm_ids[0]]
    u16_rows = trajectory_rows[arm_ids[1]]
    for sample_id in expected_ids:
        u12_route = u12_rows[sample_id]["route_diagnostics"]
        u16_route = u16_rows[sample_id]["route_diagnostics"]
        require_trajectory_prefix(u12_route, u16_route, f"{dataset_id} {sample_id}")
    return evaluate_arms(
        inputs,
        stage=32,
        native_fid=None,
        native_kid=None,
        baseline_arm_id=arm_ids[0],
        expected_sample_ids=expected_ids,
    )


def outcome(results: Mapping[str, Mapping[str, ArmResult]]) -> dict[str, Any]:
    regular = results["regular32"]
    tail = results["sharpness_tail32"]
    u12 = (regular["u12_regular32"], tail["u12_tail32"])
    u16 = (regular["u16_regular32"], tail["u16_tail32"])
    u12_pass = all(item.hard_gate_pass for item in u12)
    u16_pass = all(item.hard_gate_pass for item in u16)
    def pair_score(items: Sequence[ArmResult]) -> float | None:
        values = [
            value
            for item in items
            for value in (item.r_margin, item.q_margin, item.p_margin)
        ]
        return None if any(value is None for value in values) else min(values)

    pair_scores = {"u12": pair_score(u12), "u16": pair_score(u16)}
    if u12_pass and u16_pass:
        status = "early_stop_not_needed_at32"
        assert pair_scores["u12"] is not None and pair_scores["u16"] is not None
        selected = max(("u12", "u16"), key=lambda key: (pair_scores[key], key == "u12"))
    elif u12_pass:
        u16_failed = set().union(*(set(item.failed_gates) for item in u16))
        status = (
            "early_stop_quality_recovery"
            if u16_failed and u16_failed <= QUALITY_GATES
            else "early_stop_gate_recovery"
        )
        selected = "u12"
    elif u16_pass:
        status = "full_horizon_required"
        selected = "u16"
    else:
        selected = None
        u12_failed = set().union(*(set(item.failed_gates) for item in u12))
        if u12_failed & FACE_PRIVACY_GATES:
            status = "face_or_privacy_limited"
        elif regular["u12_regular32"].hard_gate_pass and not tail["u12_tail32"].hard_gate_pass:
            status = "tail_fragility"
        elif u12_failed & REPRESENTATION_GATES:
            status = "early_stop_representation_limited"
        elif all(set(item.failed_gates) & QUALITY_GATES for item in (*u12, *u16)):
            status = "initial_noise_quality_limited"
        elif any(
            regular[f"u{updates}_regular32"].hard_gate_pass
            and not tail[f"u{updates}_tail32"].hard_gate_pass
            for updates in (12, 16)
        ):
            status = "tail_fragility"
        elif set().union(*(set(item.failed_gates) for item in (*u12, *u16))) & FACE_PRIVACY_GATES:
            status = "face_or_privacy_limited"
        else:
            status = "initial_noise_quality_limited"
    return {
        "outcome": status,
        "u12_pair_pass": u12_pass,
        "u16_pair_pass": u16_pass,
        "selected_horizon": selected,
        "advance_arm_ids": (
            [f"{selected}_regular32", f"{selected}_tail32"] if selected else []
        ),
        "stop_required": selected is None,
        "pair_scores": pair_scores,
        "selection_rule": "maximum_worst_dataset_min_R_Q_P_tie_u12",
    }


def classify(preparation_root: Path, output_dir: Path) -> Mapping[str, Any]:
    preparation = read_json(preparation_root / "preparation_manifest.json", "R12 preparation")
    if preparation.get("status") != "prepared_not_launched":
        raise R12ClassificationError("R12 preparation status differs")
    evaluated = {
        dataset_id: materialize_dataset(preparation, dataset_id)
        for dataset_id in DATASET_ARMS
    }
    by_id = {
        dataset_id: {result.arm_id: result for result in results}
        for dataset_id, results in evaluated.items()
    }
    decision = outcome(by_id)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite R12 results: {output_dir}")
    output_dir.mkdir(parents=True)
    summary = {
        "schema_version": 1,
        "contract_type": "safa_r12_seed_aligned_result_v1",
        "stage": 32,
        "formal_native_byte_check": {
            "required": True,
            "checked_job_rows": 128,
            "status": "passed",
        },
        "datasets": {
            dataset_id: [result.as_dict() for result in results]
            for dataset_id, results in evaluated.items()
        },
        "decision": decision,
        "tail_sharpness_descriptive": {
            horizon: {
                "retention_ratio": by_id["sharpness_tail32"][f"{horizon}_tail32"].sharpness
                / by_id["sharpness_tail32"][f"{horizon}_tail32"].native_sharpness,
                "gate_deficit": max(
                    0.0,
                    max(
                        300.0,
                        0.95
                        * by_id["sharpness_tail32"][f"{horizon}_tail32"].native_sharpness,
                    )
                    - by_id["sharpness_tail32"][f"{horizon}_tail32"].sharpness,
                ),
                "partial_recovery_is_not_promotion": True,
            }
            for horizon in ("u12", "u16")
        },
        "legacy_tail_scope": {
            "selection_basis": "full_image_laplacian_sharpness",
            "known_confound": "background_high_frequency_content",
            "allowed_claim": "full_image_sharpness_gate_recovery_or_failure",
            "face_detail_restoration_claim_requires_agreeing_roi_evidence": True,
            "post_hoc_roi_tail_reselection": "forbidden",
        },
        "fid_kid_interpretation": "forbidden",
        "formal_winner": None,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output_dir / "arms.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset_id", "arm_id", "hard_gate_pass", "failed_gates", "R", "Q", "P"),
        )
        writer.writeheader()
        for dataset_id, results in evaluated.items():
            for result in results:
                writer.writerow(
                    {
                        "dataset_id": dataset_id,
                        "arm_id": result.arm_id,
                        "hard_gate_pass": result.hard_gate_pass,
                        "failed_gates": ";".join(result.failed_gates),
                        "R": result.r_margin,
                        "Q": result.q_margin,
                        "P": result.p_margin,
                    }
                )
    failures = {
        dataset_id: {
            result.arm_id: list(result.failed_gates)
            for result in results
            if any(gate in FACE_PRIVACY_GATES for gate in result.failed_gates)
        }
        for dataset_id, results in evaluated.items()
    }
    with (output_dir / "arcface_failures.json").open("x", encoding="utf-8") as handle:
        json.dump(failures, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    (output_dir / "conclusion.md").write_text(
        "# R12 seed-aligned trajectory conclusion\n\n"
        f"- Outcome: `{decision['outcome']}`\n"
        f"- Selected horizon: `{decision['selected_horizon']}`\n"
        f"- Stop required: `{str(decision['stop_required']).lower()}`\n"
        "- Tail32 supports only a full-image sharpness-gate claim; it does not by itself prove face-detail restoration.\n"
        "- Stage32 is exploratory; no formal winner is declared.\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify the bounded R12 seed-aligned trajectory matrix.")
    parser.add_argument("--preparation-root", type=Path, default=DEFAULT_PREPARATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = classify(args.preparation_root.resolve(), args.output_dir.resolve())
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
