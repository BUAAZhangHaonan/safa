from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_plot_m2_m3_curves_writes_expected_pngs(tmp_path: Path) -> None:
    from scripts.plot_m2_m3_curves import plot_m2_m3_curves

    m0 = _write_run(tmp_path, 'm0', _rows('m0'), include_quality=True)
    m2 = _write_run(tmp_path, 'm2', _rows('m2'), include_quality=True)
    m3 = _write_run(tmp_path, 'm3', _rows('m3', include_projection=True), include_quality=True)
    out_dir = tmp_path / 'plots'

    outputs = plot_m2_m3_curves(m0_run=m0, m2_run=m2, m3_run=m3, out_dir=out_dir)

    expected = {
        out_dir / 'm2_curves.png',
        out_dir / 'm3_curves.png',
        out_dir / 'm0_m2_m3_comparison.png',
        out_dir / 'm3_projection_diagnostics.png',
    }
    assert set(outputs) == expected
    for path in expected:
        assert path.is_file()
        assert path.stat().st_size > 0


def test_plot_only_m2_does_not_require_m0_or_m3_inputs(tmp_path: Path) -> None:
    from scripts.plot_m2_m3_curves import plot_m2_m3_curves

    m2 = _write_run(tmp_path, 'm2', _rows('m2'), include_quality=True)
    missing_m0 = tmp_path / 'missing_m0'
    missing_m3 = tmp_path / 'missing_m3'
    out_dir = tmp_path / 'plots'

    outputs = plot_m2_m3_curves(
        m0_run=missing_m0,
        m2_run=m2,
        m3_run=missing_m3,
        out_dir=out_dir,
        only='m2',
    )

    assert outputs == [out_dir / 'm2_curves.png']
    assert outputs[0].is_file()
    assert outputs[0].stat().st_size > 0
    assert not (out_dir / 'm3_curves.png').exists()
    assert not (out_dir / 'm0_m2_m3_comparison.png').exists()


def test_plot_only_m3_does_not_require_m0_or_m2_inputs(tmp_path: Path) -> None:
    from scripts.plot_m2_m3_curves import plot_m2_m3_curves

    m3 = _write_run(tmp_path, 'm3', _rows('m3', include_projection=True), include_quality=True)
    missing_m0 = tmp_path / 'missing_m0'
    missing_m2 = tmp_path / 'missing_m2'
    out_dir = tmp_path / 'plots'

    outputs = plot_m2_m3_curves(
        m0_run=missing_m0,
        m2_run=missing_m2,
        m3_run=m3,
        out_dir=out_dir,
        only='m3',
    )

    expected = [out_dir / 'm3_curves.png', out_dir / 'm3_projection_diagnostics.png']
    assert outputs == expected
    for path in expected:
        assert path.is_file()
        assert path.stat().st_size > 0
    assert not (out_dir / 'm2_curves.png').exists()
    assert not (out_dir / 'm0_m2_m3_comparison.png').exists()


def test_plot_m2_m3_curves_fails_when_required_metric_is_missing(tmp_path: Path) -> None:
    from scripts.plot_m2_m3_curves import plot_m2_m3_curves

    bad_rows = _rows('m2')
    del bad_rows[0]['validation_raw_latent_cosine_mean']
    m0 = _write_run(tmp_path, 'm0', _rows('m0'), include_quality=True)
    m2 = _write_run(tmp_path, 'm2', bad_rows, include_quality=True)
    m3 = _write_run(tmp_path, 'm3', _rows('m3', include_projection=True), include_quality=True)

    with pytest.raises(ValueError, match='validation_raw_latent_cosine_mean'):
        plot_m2_m3_curves(m0_run=m0, m2_run=m2, m3_run=m3, out_dir=tmp_path / 'plots')


def test_plot_m2_m3_curves_fails_when_no_history_json_exists(tmp_path: Path) -> None:
    from scripts.plot_m2_m3_curves import plot_m2_m3_curves

    m0 = _write_run(tmp_path, 'm0', _rows('m0'), include_quality=True)
    m2 = tmp_path / 'm2'
    m2.mkdir()
    m3 = _write_run(tmp_path, 'm3', _rows('m3', include_projection=True), include_quality=True)

    with pytest.raises(FileNotFoundError, match='history JSON'):
        plot_m2_m3_curves(m0_run=m0, m2_run=m2, m3_run=m3, out_dir=tmp_path / 'plots')


def _rows(run: str, *, include_projection: bool = False) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    base = {'m0': 0.75, 'm2': 0.70, 'm3': 0.68}[run]
    for epoch in range(1, 4):
        row: dict[str, float | int | str] = {
            'stage_epoch_1based': epoch,
            'loss': base - epoch * 0.01,
            'flow_loss_raw': 0.05 + epoch * 0.001,
            'cycle_loss_raw': 0.010 + epoch * 0.001,
            'validation_raw_latent_cosine_mean': 0.80 + epoch * 0.01,
            'validation_raw_single_face_eq1_rate': 1.0,
            'validation_raw_source_prediction_preserved': 0.70 + epoch * 0.01,
            'validation_raw_zero_face_rate': 0.0,
            'validation_raw_multi_face_rate': 0.0,
            'validation_raw_face_detect_ge1_rate': 1.0,
            'repr_point_loss': 0.20 / epoch,
            'repr_relation_loss': 0.30 / epoch,
            'repr_loss': 0.50 / epoch,
        }
        if include_projection:
            row.update(
                {
                    'projection_applied_fraction': 0.25 * epoch,
                    'projection_removed_norm_mean': 0.03 * epoch,
                    'projected_repr_norm_mean': 0.10 * epoch,
                    'repr_descent_inner_product_mean': 0.02 * epoch,
                }
            )
        rows.append(row)
    return rows


def _write_run(root: Path, name: str, rows: list[dict[str, float | int | str]], *, include_quality: bool) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / 'history.json').write_text(json.dumps({'history': rows}), encoding='utf-8')
    (run_dir / 'last_metrics.json').write_text(json.dumps(rows[-1]), encoding='utf-8')
    if include_quality:
        quality_dir = run_dir / 'quality'
        for row in rows:
            epoch = int(row['stage_epoch_1based'])
            epoch_dir = quality_dir / f'epoch_{epoch:04d}'
            epoch_dir.mkdir(parents=True)
            (epoch_dir / f'stage2_epoch_{epoch:04d}_raw_niqe.json').write_text(
                json.dumps({'iqa': {'method': 'niqe', 'mean': 6.0 + epoch, 'std': 0.1}}),
                encoding='utf-8',
            )
            (epoch_dir / f'stage2_epoch_{epoch:04d}_raw_distribution.json').write_text(
                json.dumps({'fid': 100.0 - epoch, 'kid_mean': 0.1 + epoch * 0.001, 'kid_std': 0.01}),
                encoding='utf-8',
            )
    return run_dir
