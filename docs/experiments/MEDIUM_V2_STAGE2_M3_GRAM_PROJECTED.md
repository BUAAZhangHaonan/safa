# MEDIUM V2 Stage2 M3 Gram Projected

Date: 2026-05-30

## Status

Pending. No completed M3 metrics artifact was found under the current checkpoint or eval artifact tree when this template was written.

## Intended Question

Does the projected Gram update keep the M2 representation-geometry benefit while reducing harmful gradient conflict with the flow-matching objective?

## Theory Note

Do not write that relation loss provides a first-order gradient advantage. The Gram relation term is useful because it applies O(B^2) pairwise geometry constraints within a batch. Those constraints can improve batch-local representation geometry discriminability and optimization conditions.

For M3, the projection diagnostic should answer whether the representation update was adjusted when it conflicted with the flow-matching update. The projection is an optimization-control mechanism, not a privacy metric by itself.

## Metrics To Fill

- Checkpoint dir: pending.
- History JSON: pending.
- Quality dir: pending.
- Privacy eval: pending and must be formal before any privacy pass claim.
- Curves: m3_curves.png and m3_projection_diagnostics.png from scripts/plot_m2_m3_curves.py.
- Quality cadence: NIQE every epoch; FID/KID every 20 epochs.

Required table once data exists:

| Epoch | loss | flow_loss_raw | cycle_loss_raw | repr_point_loss | repr_relation_loss | repr_loss | projection_applied_fraction | projection_removed_norm_mean | projected_repr_norm_mean | repr_descent_inner_product_mean | raw latent cosine | raw source preserved | single_face_eq1 | NIQE | FID |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Completion Gate

This document remains pending until the history JSON, last metrics JSON, quality JSON, and projection diagnostic fields are present and the curve script can plot M3 without missing-field errors.
