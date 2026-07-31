# R12 seed-aligned trajectory conclusion

- Audited result: `no_paired_survivor`; the locked classifier label is `face_or_privacy_limited` and no horizon is selected.
- All four arms have source/native/candidate exact-one coverage `32/32`; this is not a face-detection failure.
- On regular32, u12 and u16 fail only the ArcFace privacy point gate (`0.029659` and `0.021727`, required `<=0.02`).
- On the disjoint legacy tail32 set, u12 and u16 pass privacy but fail only full-image sharpness (retention `0.6520` and `0.6926`, required `>=0.95`). This does not by itself prove face-detail loss because the legacy selection is background-high-frequency confounded.
- The horizon direction is unstable across the two disjoint sets, so neither early stopping nor the full horizon shows a consistent recovery signal.
- The fixed affect-only `initial_noise + fixed_radius`, eta `0.5`, update-horizon line stops at stage32. FID/KID are not interpreted, no next stage is launched, and no formal winner is declared.
