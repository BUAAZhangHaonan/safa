# SAFA R9 semigroup preflight failure report (2026-07-14)

## Decision

The campaign-specific R9 semigroup preflight failed. None of the three registered split schedules passed the complete quantitative-plus-visual gate. Therefore:

- `selected_t_cut` is `null` and no locked schedule exists.
- No `closure_seal.json` was created.
- Formal campaign `r9-node2-semigroup-formal-v1` was not created.
- Phase A was not started.
- R9 remains stopped at preflight. The failed bootstrap evidence must not be reused as a passing closure.

This report records the immutable development attempts, the successful v5 execution, the independent blind review, the failed gate, and the terminal closure behavior. R8 contracts and artifacts were not modified.

## Immutable attempt ledger

| Bootstrap CID | Immutable campaign tree SHA256 | Runtime config SHA256 | Outcome |
| --- | --- | --- | --- |
| `r9-node2-semigroup-bootstrap-v1` | `ff3ea5b605a41ea0e4a938eb6c838e67f5f414c7a1f24f601b98cf7fa0624fd3` | `81ad45c831ae465722aded3b4fa80b9fa4c10b3d3832166c09daae9cf96a2919` | Stale bootstrap development attempt; preserved and not reused. |
| `r9-node2-semigroup-bootstrap-v2` | `eb4e11301c3e3f19683ed68ee1b564328928194ea8065cd12f915fa92c519b28` | `d562dfd76add2645d9b30b22d08e079ad325d73f8683f35721562b9eb0c11f81` | Resource-smoke generation completed, then the parent observed a reaped process without a unique `VmRSS`; failed closed. Log SHA256 `33d8227701730ffaf2785f0f8cb3f9c7633685926bcef14f83299cd16ebd320e`. |
| `r9-node2-semigroup-bootstrap-v3` | `d2366f1739f4418be35da5f845c892f962b10e81c162b23a8daaedccbce6d02c` | `5d840100e0f6852c8d9dc728647633944f732b838249dd0a4d9cf8627374e855` | Resource-smoke lifecycle completed, then strict completion validation rejected the producer/validator path contract; failed closed. Log SHA256 `2fedab693cbc910201684732daae3c3d0e27e932d72730f79493f0c721ec975a`. |
| `r9-node2-semigroup-bootstrap-v4` | `5325815b62810a0e1eae72d5d0b38413a5c3b674a006a606c2c03c83ef6b12ba` | `5b546ac3fb7b97774c0a14aeb527a00da6ca2d5fb5e8d568df15e60dfac0e5bb` | Resource smoke completed and was sealed, then the eight-field ArcFace declaration rejected a derived `execution` field; failed closed. Log SHA256 `3c917f2526df5ebcc9eb009604a8476f420b138107162ae5478a7fadd0d86f08`. |
| `r9-node2-semigroup-bootstrap-v5` | Bound by evidence contract `e535958e12905a12c0f719f6c8a1882358a1529567a5c6c961eed5e55ac20f7d` | Executed preflight YAML file SHA256 `a5a239ad69c3b04284e8a53705d2ab087e4e13bba125ae4f92a80331d25d3ca5` | Four shards completed successfully; the full gate failed after independent blind review. |

Every failed CID was preserved. No attempt rewrote an earlier immutable campaign directory. The common nonzero exit-file SHA256 for v2-v4 and the first v5 finalizer is `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` (`1\n`).

## v5 execution and resource bindings

- Campaign ID: `r9-node2-semigroup-bootstrap-v5`
- Formal target ID: `r9-node2-semigroup-formal-v1`
- Campaign runtime contract SHA256: `52196679a312af4dc6324cb6bf776106d11d82ae007b9179e73504d62471e385`
- Campaign runtime file SHA256: `262de54169819218bbb2da570b4318ab1249fcdaea74a672d186254c00242135`
- Executed configuration contract SHA256: `628f543e7965e2369cdb54c99387fb377f1f720fe56fb209d4b8283441d42d88`
- Arm configuration SHA256: `1822161cc9b674243ff3cc6a4b904dd3512c17b90942d4c85e50e5eb1d5449c9`
- Determinism policy SHA256: `ea6a4e81627a993066d9b1a3ca4ae791a0bcb3e21e399a5d2cb27811aa22147f`
- Checkpoint SHA256: `4690717781db58a6021d57d124300a9b212f0a5043cf3028fb5de4d9c835cc4d`
- Calibration manifest file SHA256: `ffc1f04f671533ee1498f4b03565826920afcc4e5c6ab244fc6f9b7aa680f964`
- Ordered 64-sample ID SHA256: `0e9dc5bd1da3c265efe4d66959cdc6649a6b60b82c29058adf0dab843b7c1df3`
- Attention backend: requested `native`, resolved `native`
- Execution: 1 logical run, 4 shards, 64 unique samples, 64 sample-runs, seed 1337
- Generation log SHA256: `4abeaad9e099c4f871bb5882c4919b48207f8773ba7ca70441ad9c1843d5eaa8`
- Generation exit-file SHA256: `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` (`0\n`)

The v5 run reused the successful immutable v4 resource smoke. It did not launch another resource worker:

- Resource-smoke result file SHA256: `3118dbad0cd922145c03f9c4b3f6b4d9643a968f64c70bd3f72a588d23fe8ede`
- Resource-smoke contract SHA256: `92bfd288ef663b171940150d5fcb423d4b6ae948ce4f7cb352d1fab8d13723c7`
- Resource contract SHA256: `269eb2c877bbe9848dce2b4cae5db99589956933a1d57146fb90b7fd7eb0da54`
- Measured peak process-tree RSS: 3,355,484,160 bytes
- RAM slot budget: 3,691,032,576 bytes (measured peak plus 10%)
- GPU UUID: `GPU-7ba69fc7-12ac-3dfb-8265-3476ce2504b6`
- Four admitted slots: 0, 1, 2, 3
- Per-worker maximum allocated GPU memory: 1,658,522,624 bytes
- Per-worker maximum reserved GPU memory: 1,937,768,448 bytes

The four shard generation-result SHA256 values are, in shard order:

1. `0ce1a495742c6d10ae3efa88e934725c2ca491060bc4d1a9a11804f1b21edb5c`
2. `25e4118e82d292e7f9d3236ebf9b841cc468422e191a0a37b772098fe05560a8`
3. `5962d8417683fb25847ae754648be2581826819e8070d4802f703ae7aa14ac07`
4. `7cfb8a3f4bbeafbf367bac55127fa61fb6ccccccd8d858c48a2579f54d6e56c4`

The corresponding semigroup evidence SHA256 values are:

1. `b9e5002a7f4d4f98697cb7839ab920bd736fc69cfcf5ef6488a61bc62d69370a`
2. `32d7f0587fd37b7b0444745a352098a688a47f1ebfa1ea224d4bc86ec585c264`
3. `a794906f9c26b335907e8d1d874cb7a59eee5e1699c623b200a4dcc81455945b`
4. `768771e54564a66184e74572a0d68c536287d08f880c81cd9448fd4264ea7b5b`

The complete evidence manifest has file SHA256 `71fd256fb2546496b36d3b863e56ab075cb64cfed04d1119fe7db11bf09b1e8b` and contract SHA256 `e535958e12905a12c0f719f6c8a1882358a1529567a5c6c961eed5e55ac20f7d`. Its aggregate bindings are:

- 320 output PNGs: `0c3e9c8506eb89cfa630eb8805becfc05be899379463b9814f5c1db62ea73e0b`
- 192 latent diagnostic values: `6a3a703184b52a51a2902be8e9e76ede0dc7e60213e4de815378f9817bfecd02`
- 192 full diagnostic values: `dbb659ed9af5431a1087ef7df197a08ab9e47f2910ff3c1394a6cdd3bee9f231`

## Independent blind review

The immutable assignment covered all 64 samples for all three opaque conditions across 24 contact sheets. The assignment file SHA256 is `64e366555f11daf38f956915ed94f0b000038b01da3775ec61787233bc9cb860`; its contract SHA256 is `7e0bbb204371df4b27709a5f5f4d58269067937f28b757ea2d910c19b09925a0`. The independent decision file SHA256 is `19beaf6353d11f5482a498668e80cbbd7a3f334293df5b00f139133055ef8df0`; its contract SHA256 is `005382d6dd745317ece6230186e0465475a138d11faf9af27fd698c5cb8ab3a8`.

After review completion, the first finalizer invocation revealed this mapping:

| Opaque condition | Split | Severe | Passed |
| --- | ---: | ---: | --- |
| `condition_26a3c0b02ea2` | 0.25 | 1/64 | false |
| `condition_7f0ded397890` | 0.5 | 1/64 | false |
| `condition_795eb42f27fb` | 0.75 | 1/64 | false |

All three conditions marked the same sample as severe:

`val:Manually_Annotated_Images/1009/c85be880e25f2a48ad2e3a33cb2d0a5c8c02907fd0d7c3778b3e8bc1.jpg`

The review rule is exact: a condition passes if and only if its severe count is zero. Thus all three registered schedules fail the visual gate.

## Gate reconstruction

| `t_cut` | Latent residual median (<=0.1) | Latent residual p90 (<=0.2) | Endpoint E0 median (>=0.95) | Visual | Overall |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.25 | 0.0801515207, pass | 0.1109797888, pass | 0.9900583327, pass | fail | fail |
| 0.5 | 0.1449673325, fail | 0.2049319074, fail | 0.9700527787, pass | fail | fail |
| 0.75 | 0.1568036452, fail | 0.2474973917, fail | 0.9690026045, pass | fail | fail |

The selection rule is `smallest_numeric_t_cut_passing_all_registered_thresholds`. The reconstructed result is `gate_passed=false`, `selected_t_cut=null`. Split 0.25 is quantitatively admissible but cannot pass because its blind visual review found one severe case. Splits 0.5 and 0.75 fail both visual review and latent-residual thresholds.

## Severe-sample byte forensics

The earlier pre-campaign semigroup output and the v5 campaign output are byte-identical for every image used to inspect the severe sample:

| Image role | Old SHA256 | v5 SHA256 | Equal |
| --- | --- | --- | --- |
| Generated direct | `b8c862f66895bcb79efba9b332978c1e92fe4c078aa4287cecffb772aba65abf` | `b8c862f66895bcb79efba9b332978c1e92fe4c078aa4287cecffb772aba65abf` | yes |
| Matched native | `b8c862f66895bcb79efba9b332978c1e92fe4c078aa4287cecffb772aba65abf` | `b8c862f66895bcb79efba9b332978c1e92fe4c078aa4287cecffb772aba65abf` | yes |
| Split 0.25 | `d9ec33c75a71bca232a2ca58ba7fe13a077f20305a646449e2002f06c4074d28` | `d9ec33c75a71bca232a2ca58ba7fe13a077f20305a646449e2002f06c4074d28` | yes |
| Split 0.5 | `783f8e6851923b39b13b2881625a6a4339d3d4d100050d42291ffb3e49395193` | `783f8e6851923b39b13b2881625a6a4339d3d4d100050d42291ffb3e49395193` | yes |
| Split 0.75 | `8db8da2979f8143962b76faa00cbd0333e91c0982eb7a078b8469f77f6c491eb` | `8db8da2979f8143962b76faa00cbd0333e91c0982eb7a078b8469f77f6c491eb` | yes |

The source JPEG SHA256 is `ca8db2e16862c0547ad1273011ef9c13191a3c06e93f86639e6750a148252136`. This comparison is forensic only and is not part of the gate. It shows that the severe finding is not caused by v5 output-path handling, multi-slot scheduling, or rerun drift.

## First-finalizer implementation defect and terminal state

The first finalizer invocation exited 1 with:

`CampaignSemigroupClosureError: semigroup gate failed; a passing locked schedule cannot be sealed`

The finalizer log SHA256 is `df9144e8fc5e41dcba40fa50d2e5589f00117d7d691e1def9f3feb2acc26021b`; its exit-file SHA256 is `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` (`1\n`). It correctly created neither the requested closure directory nor a seal or formal campaign.

However, that version of the finalizer changed the blinding map from mode `000` to mode `0400` before it evaluated the failed gate. This is an implementation defect. The v5 map is now revealed and cannot honestly be made blind again. The revealed map file currently has mode `0400`, file SHA256 `703578f3145657131c610eafc7af3beb9ae5826b15717373bbbd520c0e963100`, and embedded contract SHA256 `c916cab3af0a6641033b1b67aa4059476c5a5ba59689b1ebe51a89744187fe44`. The v5 artifacts must remain unchanged; changing the mode back to `000` would only hide an already revealed map and would misstate the audit history.

The repaired terminal path validates the complete locked review without reading or changing the map, keeps a failed campaign's map at `000` in future runs, and creates exactly one immutable `closure_failure.json` via exclusive creation. The v5 terminal run completed with `terminal_path_read_map=false`. The closure directory contains only that read-only failure contract. It contains no seal, schedule, gate, report, or copied map, and the formal campaign remains absent.

- Terminal failure artifact: `artifacts/r9_meanflow_flow_map_guidance/semigroup_campaign_closures/r9-node2-semigroup-bootstrap-v5__for__r9-node2-semigroup-formal-v1/closure_failure.json`
- Terminal failure mode: `0444`
- Terminal failure file SHA256: `1c9ce4bd345267a1ea391f441dcca11aa9d297d93d4d95f5e03b5e2313517a82`
- Terminal failure contract SHA256: `fabee668cda6bc2f149035e433ebef10d3474d03ddc2f9596a80492a6008e3b0`
- Terminal failure reason: `all_blinded_visual_conditions_failed`
- Terminal path read map: `false`

The immutable failure contract binds the non-map campaign, runtime, evidence, assignment, and source-review chain. It does not contain a `map_state_at_materialization` field or old-finalizer log/exit bindings. Therefore, `terminal_path_read_map=false` proves only that the repaired terminal invocation did not inspect the map; it does not prove that no earlier invocation revealed it. The historical reveal is an explicit audit gap in the failure-contract schema. It is bound only by this Git-tracked report through the exact old map mode `0400`, map file/contract SHA values, and first-finalizer log/exit SHA values above. The terminal artifact cannot be rewritten to add fields without violating its exclusive-create and immutability contract.

The terminal resolver revalidated the failure contract and failed closed with `formal campaign semigroup closure is a terminal failure: all_blinded_visual_conditions_failed`.

## Stop conclusion

This preflight supplies no admissible schedule. The only quantitatively passing split, 0.25, fails the preregistered visual safety rule on one of 64 samples. The correct action is to stop R9 before Phase A. Re-running the same CID, altering the review, selecting a split from quantitative metrics alone, or treating the revealed map as newly blinded would violate the campaign contract.
