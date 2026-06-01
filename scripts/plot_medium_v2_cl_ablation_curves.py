#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from plot_m2_m3_curves import (
    GRAM_REQUIRED_FIELDS,
    _load_run,
    _plot_axes,
    _plot_run,
    _quality_series,
    _require_fields,
    _series,
    _utility,
)


def plot_medium_v2_cl_ablation_curves(
    *,
    point_only_run: Path,
    point_gram_run: Path,
    out_dir: Path,
    point_only_history_json: Path | None = None,
    point_gram_history_json: Path | None = None,
    point_only_quality_dir: Path | None = None,
    point_gram_quality_dir: Path | None = None,
) -> list[Path]:
    point_only = _load_run(
        'point-only CL-only',
        point_only_run,
        history_json=point_only_history_json,
        quality_dir=point_only_quality_dir,
    )
    point_gram = _load_run(
        'point+Gram CL-only',
        point_gram_run,
        history_json=point_gram_history_json,
        quality_dir=point_gram_quality_dir,
    )
    for run in (point_only, point_gram):
        _require_fields(run.rows, GRAM_REQUIRED_FIELDS, run.label)

    outputs = [
        out_dir / 'point_only_cl_only_curves.png',
        out_dir / 'point_gram_cl_only_curves.png',
        out_dir / 'point_only_vs_point_gram_cl_only.png',
    ]
    _plot_run(point_only, outputs[0])
    _plot_run(point_gram, outputs[1])
    _plot_axes(
        outputs[2],
        'medium_v2 CL-only ablation',
        [
            ('loss', [(point_only.label, *_series(point_only, 'loss')), (point_gram.label, *_series(point_gram, 'loss'))]),
            (
                'repr point',
                [
                    (point_only.label, *_series(point_only, 'repr_point_loss')),
                    (point_gram.label, *_series(point_gram, 'repr_point_loss')),
                ],
            ),
            (
                'repr relation',
                [
                    (point_only.label, *_series(point_only, 'repr_relation_loss')),
                    (point_gram.label, *_series(point_gram, 'repr_relation_loss')),
                ],
            ),
            ('repr total', [(point_only.label, *_series(point_only, 'repr_loss')), (point_gram.label, *_series(point_gram, 'repr_loss'))]),
            ('utility', [(point_only.label, *_utility(point_only)), (point_gram.label, *_utility(point_gram))]),
            (
                'source preserved',
                [
                    (point_only.label, *_series(point_only, 'validation_raw_source_prediction_preserved')),
                    (point_gram.label, *_series(point_gram, 'validation_raw_source_prediction_preserved')),
                ],
            ),
            ('NIQE', [(point_only.label, *_quality_series(point_only, 'niqe')), (point_gram.label, *_quality_series(point_gram, 'niqe'))]),
            ('FID', [(point_only.label, *_quality_series(point_only, 'fid')), (point_gram.label, *_quality_series(point_gram, 'fid'))]),
        ],
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot medium_v2 point-only vs point+Gram CL-only ablation curves.')
    parser.add_argument('--point-only-run', type=Path, default=Path('artifacts/checkpoints/g_medium_v2_stage2_point_only_cl_only'))
    parser.add_argument('--point-gram-run', type=Path, default=Path('artifacts/checkpoints/g_medium_v2_stage2_point_gram_cl_only'))
    parser.add_argument('--point-only-history-json', type=Path)
    parser.add_argument('--point-gram-history-json', type=Path)
    parser.add_argument('--point-only-quality-dir', type=Path)
    parser.add_argument('--point-gram-quality-dir', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('artifacts/plots/medium_v2_cl_ablation'))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outputs = plot_medium_v2_cl_ablation_curves(
            point_only_run=args.point_only_run,
            point_gram_run=args.point_gram_run,
            out_dir=args.out_dir,
            point_only_history_json=args.point_only_history_json,
            point_gram_history_json=args.point_gram_history_json,
            point_only_quality_dir=args.point_only_quality_dir,
            point_gram_quality_dir=args.point_gram_quality_dir,
        )
    except Exception as exc:
        print(f'error: {exc}')
        return 1
    for path in outputs:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
