# Medium V2 M2 Follow-up Plan

Goal: keep M2 monitoring light, then block any M3 launch until M2 completion, documentation, curves, and an explicit marker are present.

## Active monitor

- Monitor session: `monitor_medium_v2_stage2_m2_gram_weighted`
- Status JSON: `artifacts/monitor/medium_v2_stage2_m2_gram_weighted_status.json`
- Events JSONL: `artifacts/monitor/medium_v2_stage2_m2_gram_weighted_events.jsonl`
- Human log: `artifacts/monitor/medium_v2_stage2_m2_gram_weighted.log`

## M2 completion gate

M2 counts as complete only when `artifacts/checkpoints/g_medium_v2_stage2_m2_gram_weighted/last_metrics.json` or history shows `stage_epoch_1based >= 120`, and expected final checkpoints are present.

## Required work before M3

1. Write the M2 run document with config, checkpoint, metrics, and known issues.
2. Generate M2 curves from history and quality JSONs.
3. Review the M2 document and curves.
4. Create explicit marker file `artifacts/monitor/medium_v2_stage2_m2_docs_and_curves_approved.marker`.

## Supervisor policy

No M2-to-M3 supervisor is running now.

If a supervisor is added later, it must only start M3 after all of these are true:

- M2 complete at 120 epochs.
- M2 document exists.
- M2 curves exist.
- `artifacts/monitor/medium_v2_stage2_m2_docs_and_curves_approved.marker` exists.
- No `train_g_medium_v2_stage2_m3_*` tmux session is already running.
