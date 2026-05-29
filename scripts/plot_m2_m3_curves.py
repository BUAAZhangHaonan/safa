#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


BASE_REQUIRED_FIELDS = (
    'loss',
    'flow_loss_raw',
    'cycle_loss_raw',
    'validation_raw_latent_cosine_mean',
    'validation_raw_single_face_eq1_rate',
    'validation_raw_source_prediction_preserved',
    'validation_raw_face_detect_ge1_rate',
    'validation_raw_zero_face_rate',
    'validation_raw_multi_face_rate',
)
GRAM_REQUIRED_FIELDS = ('repr_point_loss', 'repr_relation_loss', 'repr_loss')
M3_PROJECTION_FIELDS = (
    'projection_applied_fraction',
    'projection_removed_norm_mean',
    'projected_repr_norm_mean',
    'repr_descent_inner_product_mean',
)
HISTORY_CANDIDATES = ('history.json', 'metrics_history.json', 'last_metrics_history.json')


class RunSeries:
    def __init__(self, label: str, run_dir: Path, rows: list[dict[str, Any]], quality: dict[int, dict[str, float]]):
        self.label = label
        self.run_dir = run_dir
        self.rows = rows
        self.quality = quality


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('matplotlib is required to plot M2/M3 curves') from exc
    return plt


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f'required JSON is missing: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _history_path(run_dir: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    for name in HISTORY_CANDIDATES:
        path = run_dir / name
        if path.is_file():
            return path
    candidates = ', '.join(HISTORY_CANDIDATES)
    raise FileNotFoundError(f'{run_dir}: missing required history JSON; expected one of {candidates}')


def _load_history(path: Path, label: str) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if isinstance(payload, dict):
        history = payload.get('history', payload.get('epochs'))
    else:
        history = payload
    if not isinstance(history, list) or not history:
        raise ValueError(f'{label}: {path} must contain a non-empty history or epochs list')
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            raise ValueError(f'{label}: history[{index}] must be an object')
        rows.append(dict(row))
    return rows


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f'{label} must be numeric, got bool')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be numeric, got {value!r}') from exc
    if not math.isfinite(number):
        raise ValueError(f'{label} must be finite, got {value!r}')
    return number


def _epoch(row: dict[str, Any], fallback_index: int, label: str) -> int:
    for field in ('stage_epoch_1based', 'epoch'):
        if field in row:
            value = _finite(row[field], f'{label}.{field}')
            if int(value) != value or value <= 0:
                raise ValueError(f'{label}.{field} must be a positive integer, got {row[field]!r}')
            return int(value)
    if 'stage_epoch' in row:
        value = _finite(row['stage_epoch'], f'{label}.stage_epoch')
        if int(value) != value or value < 0:
            raise ValueError(f"{label}.stage_epoch must be a non-negative integer, got {row['stage_epoch']!r}")
        return int(value) + 1
    return fallback_index


def _require_fields(rows: list[dict[str, Any]], fields: tuple[str, ...], label: str) -> None:
    for index, row in enumerate(rows):
        row_label = f'{label}.history[{index}]'
        for field in fields:
            if field not in row:
                raise ValueError(f'{row_label}: missing required metric {field!r}')
            _finite(row[field], f'{row_label}.{field}')


def _quality_epoch_from_path(path: Path) -> int | None:
    for text in (path.parent.name, path.name):
        match = re.search(r'epoch_(\d{4,})', text)
        if match:
            return int(match.group(1))
    return None


def _resolve_quality_dir(run_dir: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    direct = run_dir / 'quality'
    if direct.is_dir():
        return direct
    parts = run_dir.parts
    if 'checkpoints' in parts:
        root = Path(*parts[: parts.index('checkpoints')])
        candidate = root / 'eval' / run_dir.name / 'quality'
        if candidate.is_dir():
            return candidate
    return direct


def _load_quality(quality_dir: Path, label: str) -> dict[int, dict[str, float]]:
    if not quality_dir.is_dir():
        raise FileNotFoundError(f'{label}: required quality directory is missing: {quality_dir}')
    rows: dict[int, dict[str, float]] = {}
    for path in sorted(quality_dir.glob('epoch_*/*.json')):
        epoch = _quality_epoch_from_path(path)
        if epoch is None:
            continue
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f'{path} must contain a quality metrics object')
        row = rows.setdefault(epoch, {})
        iqa = payload.get('iqa')
        if isinstance(iqa, dict) and str(iqa.get('method', '')).lower() == 'niqe':
            row['niqe'] = _finite(iqa.get('mean'), f'{path}.iqa.mean')
            if iqa.get('std') is not None:
                row['niqe_std'] = _finite(iqa.get('std'), f'{path}.iqa.std')
        for source, target in (('fid', 'fid'), ('kid_mean', 'kid_mean'), ('kid_std', 'kid_std')):
            if source in payload:
                row[target] = _finite(payload[source], f'{path}.{source}')
    if not rows:
        raise ValueError(f'{label}: no quality JSON files found in {quality_dir}')
    if not any('niqe' in row for row in rows.values()):
        raise ValueError(f'{label}: quality JSON has no NIQE values')
    return rows


def _load_run(label: str, run_dir: Path, *, history_json: Path | None = None, quality_dir: Path | None = None) -> RunSeries:
    rows = _load_history(_history_path(run_dir, history_json), label)
    _require_fields(rows, BASE_REQUIRED_FIELDS, label)
    quality = _load_quality(_resolve_quality_dir(run_dir, quality_dir), label)
    return RunSeries(label, run_dir, rows, quality)


def _series(run: RunSeries, field: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for index, row in enumerate(run.rows, start=1):
        if field not in row:
            raise ValueError(f'{run.label}.history[{index - 1}]: missing required metric {field!r}')
        xs.append(_epoch(row, index, run.label))
        ys.append(_finite(row[field], f'{run.label}.{field}'))
    return xs, ys


def _quality_series(run: RunSeries, field: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    history_epochs = {_epoch(row, index, run.label) for index, row in enumerate(run.rows, start=1)}
    for epoch in sorted(history_epochs):
        value = run.quality.get(epoch, {}).get(field)
        if value is not None:
            xs.append(epoch)
            ys.append(_finite(value, f'{run.label}.quality[{epoch}].{field}'))
    if not xs:
        raise ValueError(f'{run.label}: missing required quality metric {field!r}')
    return xs, ys


def _utility(run: RunSeries) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for index, row in enumerate(run.rows, start=1):
        xs.append(_epoch(row, index, run.label))
        cosine = _finite(row['validation_raw_latent_cosine_mean'], f'{run.label}.latent_cosine')
        single = _finite(row['validation_raw_single_face_eq1_rate'], f'{run.label}.single_face')
        ys.append(cosine * single)
    return xs, ys


def _plot_axes(output: Path, title: str, panels: list[tuple[str, list[tuple[str, list[int], list[float]]]]]) -> None:
    plt = _import_pyplot()
    if not panels:
        raise ValueError(f'no panels provided for {title}')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, max(4, 2.4 * len(panels))), sharex=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (ylabel, curves) in zip(axes, panels):
        if not curves:
            raise ValueError(f'{title}: no curves for {ylabel}')
        for label, xs, ys in curves:
            if not xs:
                raise ValueError(f'{title}: empty curve {label} for {ylabel}')
            ax.plot(xs, ys, marker='o', linewidth=1.5, markersize=3, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].set_title(title)
    axes[-1].set_xlabel('epoch')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_run(run: RunSeries, output: Path) -> None:
    _require_fields(run.rows, GRAM_REQUIRED_FIELDS, run.label)
    _plot_axes(
        output,
        f'{run.label} curves',
        [
            ('loss', [(run.label, *_series(run, 'loss'))]),
            ('flow/cycle', [(f'{run.label} flow', *_series(run, 'flow_loss_raw')), (f'{run.label} cycle', *_series(run, 'cycle_loss_raw'))]),
            ('repr loss', [(f'{run.label} point', *_series(run, 'repr_point_loss')), (f'{run.label} relation', *_series(run, 'repr_relation_loss')), (f'{run.label} total', *_series(run, 'repr_loss'))]),
            ('utility', [(run.label, *_utility(run))]),
            ('quality', [(f'{run.label} NIQE', *_quality_series(run, 'niqe')), (f'{run.label} FID', *_quality_series(run, 'fid'))]),
        ],
    )


def _plot_comparison(runs: list[RunSeries], output: Path) -> None:
    _plot_axes(
        output,
        'M0/M2/M3 comparison',
        [
            ('loss', [(run.label, *_series(run, 'loss')) for run in runs]),
            ('utility', [(run.label, *_utility(run)) for run in runs]),
            ('source preserved', [(run.label, *_series(run, 'validation_raw_source_prediction_preserved')) for run in runs]),
            ('NIQE', [(run.label, *_quality_series(run, 'niqe')) for run in runs]),
            ('FID', [(run.label, *_quality_series(run, 'fid')) for run in runs]),
        ],
    )


def _plot_projection(run: RunSeries, output: Path) -> None:
    _require_fields(run.rows, M3_PROJECTION_FIELDS, run.label)
    _plot_axes(
        output,
        'M3 projection diagnostics',
        [
            ('applied fraction', [(run.label, *_series(run, 'projection_applied_fraction'))]),
            ('removed norm', [(run.label, *_series(run, 'projection_removed_norm_mean'))]),
            ('projected norm', [(run.label, *_series(run, 'projected_repr_norm_mean'))]),
            ('descent inner product', [(run.label, *_series(run, 'repr_descent_inner_product_mean'))]),
        ],
    )


def plot_m2_m3_curves(
    *,
    m0_run: Path,
    m2_run: Path,
    m3_run: Path,
    out_dir: Path,
    only: str | None = None,
    m0_history_json: Path | None = None,
    m2_history_json: Path | None = None,
    m3_history_json: Path | None = None,
    m0_quality_dir: Path | None = None,
    m2_quality_dir: Path | None = None,
    m3_quality_dir: Path | None = None,
) -> list[Path]:
    if only is not None and only not in {'m2', 'm3'}:
        raise ValueError(f"only must be 'm2', 'm3', or None; got {only!r}")

    if only == 'm3':
        m3 = _load_run('m3', m3_run, history_json=m3_history_json, quality_dir=m3_quality_dir)
        outputs = [out_dir / 'm3_curves.png', out_dir / 'm3_projection_diagnostics.png']
        _plot_run(m3, outputs[0])
        _plot_projection(m3, outputs[1])
        return outputs

    m2 = _load_run('m2', m2_run, history_json=m2_history_json, quality_dir=m2_quality_dir)
    if only == 'm2':
        output = out_dir / 'm2_curves.png'
        _plot_run(m2, output)
        return [output]

    m3 = _load_run('m3', m3_run, history_json=m3_history_json, quality_dir=m3_quality_dir)
    m0 = _load_run('m0', m0_run, history_json=m0_history_json, quality_dir=m0_quality_dir)
    outputs = [
        out_dir / 'm2_curves.png',
        out_dir / 'm3_curves.png',
        out_dir / 'm0_m2_m3_comparison.png',
        out_dir / 'm3_projection_diagnostics.png',
    ]
    _plot_run(m2, outputs[0])
    _plot_run(m3, outputs[1])
    _plot_comparison([m0, m2, m3], outputs[2])
    _plot_projection(m3, outputs[3])
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot medium_v2 M2/M3 curves from strict JSON metrics.')
    parser.add_argument('--only', choices=('m2', 'm3'), help='plot only one stage and do not require the other stage inputs')
    parser.add_argument('--m0-run', type=Path, default=Path('artifacts/checkpoints/g_medium_v1_stage2_m0'))
    parser.add_argument('--m2-run', type=Path, default=Path('artifacts/checkpoints/g_medium_v2_stage2_m2_gram_weighted'))
    parser.add_argument('--m3-run', type=Path, default=Path('artifacts/checkpoints/g_medium_v2_stage2_m3_gram_projected'))
    parser.add_argument('--m0-history-json', type=Path)
    parser.add_argument('--m2-history-json', type=Path)
    parser.add_argument('--m3-history-json', type=Path)
    parser.add_argument('--m0-quality-dir', type=Path)
    parser.add_argument('--m2-quality-dir', type=Path)
    parser.add_argument('--m3-quality-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('artifacts/plots/medium_v2'))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = plot_m2_m3_curves(
            m0_run=args.m0_run,
            m2_run=args.m2_run,
            m3_run=args.m3_run,
            out_dir=args.out_dir,
            only=args.only,
            m0_history_json=args.m0_history_json,
            m2_history_json=args.m2_history_json,
            m3_history_json=args.m3_history_json,
            m0_quality_dir=args.m0_quality_dir,
            m2_quality_dir=args.m2_quality_dir,
            m3_quality_dir=args.m3_quality_dir,
        )
    except Exception as exc:
        print(f'error: {exc}')
        return 1
    for path in outputs:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
