# MEDIUM V2 Stage2 M2 Gram Weighted

Date: 2026-05-30

## Status

Pending. No completed M2 metrics artifact was found under the current checkpoint or eval artifact tree when this template was written.

## Intended Question

Does adding a weighted hyperspherical Gram relation term improve Stage2 utility and quality over M0 without harming the face-detection guard?

## Theory Note

Do not describe the relation term as providing a separate first-order gradient signal. The useful point is different: the Gram relation term adds O(B^2) pairwise geometry constraints inside each batch. These constraints can improve batch-local representation geometry discriminability and optimization conditions because the model sees how samples relate to each other, not only how each sample matches its own target.

## Metrics To Fill

- Checkpoint dir: pending.
- History JSON: pending.
- Quality dir: pending.
- Privacy eval: pending and must be formal before any privacy pass claim.
- Curves: m2_curves.png from scripts/plot_m2_m3_curves.py.

Required table once data exists:

| Epoch | loss | flow_loss_raw | cycle_loss_raw | gram_point_loss | gram_relation_loss | gram_total_loss | raw latent cosine | raw source preserved | single_face_eq1 | NIQE | FID | KID mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Completion Gate

This document remains pending until the history JSON, last metrics JSON, and quality JSON are present and the curve script can plot M2 without missing-field errors.
