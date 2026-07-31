# R12 face-frequency diagnostic

The contraction magnitude is large, but the aligned face ROI misses the predeclared monotonic-count rule.

- Tail32 monotonic count: `21/32` (required `>= 24`).
- Tail32 median NFE5/NFE1 aligned-ROI high-frequency ratio: `0.676985` (required `<= 0.8`).
- Classification: `face_roi_sampler_low_pass_not_confirmed`.
- Tail32 NFE5 median high-frequency ratio, ROI/full: `0.676985/0.539195`.
- Prefix128 NFE5 median aligned-ROI high-frequency ratio: `0.940782`.

Do not call this a global sampler low-pass confirmation. The ROI and full-image controls both lose high frequency on tail32, while prefix128 is near neutral; the next diagnostic should explain the non-monotonic tail exceptions rather than open another eta or schedule grid.

This is a diagnostic over existing images. It is not a screening survivor, privacy result, Full gate, or formal winner.
