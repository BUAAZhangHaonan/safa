# SAFA R9 Phase B: three-seed calibration results (2026-07-14)

## Scope and immutable evidence

Phase B evaluated the three Phase-A promotions and a seed-matched native run on
the fixed 64-image calibration manifest at seeds 1337, 2027, and 3407. This is
12 logical runs and 768 sample-runs. The frozen generation inventory contains
1,344 PNG files and has SHA-256
`e40516b8dc852c6b6930e38b89b656b28274c91f40003884ab688d8768ab145a`.
Generation used GPUs 0--3 concurrently with three admitted slots per GPU.

Evaluation recovery did not rerun generation. The immutable repair chain is:

- v1 `fb14adb2543c62e4a320f152b0b11c96a78a4d1b91c8870846d01f89a76629ca`:
  corrected the source-index binding, then stopped at the pre-existing
  `manifest_ids` failure.
- v2 `4b6e5c5ae8235d2b084d9d1141e32ec55fd93a61de8f05dbc928ed0cafff57b6`:
  corrected that failure, then stopped because raw evidence still targeted the
  original immutable namespace.
- v3 `716355ccf9171d3b6d35f51c124139e110b99986393ed7e2b397c02d7c0fb355`:
  isolated repaired evaluator evidence in its own digest-bound namespace and
  completed 12 quality evaluations and nine ArcFace evaluations.

All nine required 64-image visual reviews were completed. Severe counts by
seed were:

| Arm | 1337 | 2027 | 3407 | Total |
| --- | ---: | ---: | ---: | ---: |
| `paper_eta_0p125` | 1 | 0 | 0 | 1 |
| `paper_eta_0p25_disable_i2` | 2 | 1 | 1 | 4 |
| `flow_map2_normalized_eta_0p125` | 2 | 2 | 1 | 5 |

No sample was severe in two or more seeds for any arm.

## Post-hoc metric summary

The campaign declared numerical metrics as `report_only` and visual outcomes as
`observation_only`. Values below are means over the three seeds; deltas are
candidate minus matched native, so lower FID, KID, and NIQE are better.

| Arm | E0 (delta) | Edev (delta) | FID (delta) | KID (delta) | NIQE (delta) | Sharpness (delta) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 0.16078 | 0.37515 | 153.233 | 0.03547 | 4.7183 | 695.50 |
| `paper_eta_0p125` | 0.77561 (+0.61483) | 0.61918 (+0.24404) | 140.682 (-12.551) | 0.02407 (-0.01139) | 4.6158 (-0.1025) | 610.94 (-84.56) |
| `flow_map2_normalized_eta_0p125` | 0.74044 (+0.57966) | 0.64032 (+0.26518) | 143.362 (-9.871) | 0.02556 (-0.00991) | 4.7482 (+0.0299) | 645.88 (-49.61) |
| `paper_eta_0p25_disable_i2` | 0.71031 (+0.54953) | 0.61315 (+0.23800) | 140.515 (-12.719) | 0.02492 (-0.01055) | 4.6007 (-0.1176) | 579.02 (-116.47) |

The preregistered 10,000-resample paired cluster bootstrap gave these main
results:

| Arm | E0 delta lower 95% | Edev delta lower 95% | NIQE delta 95% interval | Sharpness delta 95% interval |
| --- | ---: | ---: | ---: | ---: |
| `paper_eta_0p125` | +0.55877 | +0.20257 | [-0.18354, -0.01989] | [-124.81, -44.99] |
| `flow_map2_normalized_eta_0p125` | +0.52391 | +0.22584 | [-0.04488, +0.10509] | [-93.92, -2.84] |
| `paper_eta_0p25_disable_i2` | +0.49435 | +0.19530 | [-0.20957, -0.02638] | [-170.84, -62.64] |

## Coverage exception and report-only selection

The immutable Phase-B gate is
`84c4aa802965601bfeccc03fa0e9da2baef25d8cc98cb9dbbc536058037520b9`.
Its formal verdict is `stop_zero_candidates` with no selected arm. The only
failure for every candidate is
`seed_2027:arcface_not_exactly_one_face_per_image`. Candidate and source images
have exact-one coverage for all 192 observations per arm, while the matched
native has 191/192. The single exception is
`val:Manually_Annotated_Images/1003/5a46f394c9709f851bdb273c33f8ef136fe8c1c384b0975b8047c47b.jpg`, for which
the native image produced two detections at seed 2027. This is a native
reference coverage miss, not a candidate failure.

A clearly labelled post-hoc complete-case analysis excluded that one native
miss uniformly, leaving 63 IDs times three seeds, or 189 paired observations.
The ArcFace source-candidate minus source-native mean delta and one-sided 95%
upper bound were:

| Arm | Mean delta | Upper 95% |
| --- | ---: | ---: |
| `paper_eta_0p125` | +0.00321 | +0.00861 |
| `flow_map2_normalized_eta_0p125` | -0.00090 | +0.00402 |
| `paper_eta_0p25_disable_i2` | +0.00513 | +0.01062 |

These complete-case results are descriptive and do not replace the frozen
gate. Under the requested report-only policy, the balanced post-hoc selection
is `paper_eta_0p125` first and `flow_map2_normalized_eta_0p125` second. The
former has one severe image, the best E0 and KID, and broad quality gains. The
latter has the best Edev and retains more sharpness. Disabling I2 has a slightly
better FID and NIQE but loses more representation utility and sharpness.

## Resource observations for Phase C

The Phase-B generator kept all four GPUs active, but used three slots per GPU.
The evaluator dispatch path remained serial even though each evaluator was
globally admitted. Phase C therefore requires preregistered multi-shards with a
deterministic four-GPU rotation, a true parallel evaluator dispatcher, and a
measured batch-size contract. Batch size four must not be enabled unless it is
bitwise equivalent to batch size two for PNGs, metrics, and final latents and
its measured peak fits the free-VRAM budget with the fixed 2 GiB headroom.

