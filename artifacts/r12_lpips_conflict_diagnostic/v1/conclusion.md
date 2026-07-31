# R12 native-anchor LPIPS conflict diagnostic

Native-anchor LPIPS conflict projection is rejected as ungrounded by the predeclared evidence rule.

- `regular_privacy` licensed: `false`.
  - u12: Spearman `0.189150` (95% bootstrap CI `-0.167292, 0.503430`), top-outlier enrichment `1.000x`, top-8 failure-group ROI LPIPS mean gap `0.015233`.
  - u16: Spearman `-0.052786` (95% bootstrap CI `-0.415248, 0.315190`), top-outlier enrichment `0.500x`, top-8 failure-group ROI LPIPS mean gap `-0.043327`.
- `tail_sharpness` licensed: `false`.
  - u12: Spearman `0.331378` (95% bootstrap CI `-0.040272, 0.638738`), top-outlier enrichment `2.000x`, top-8 failure-group ROI LPIPS mean gap `0.095624`.
  - u16: Spearman `0.312317` (95% bootstrap CI `-0.084833, 0.625230`), top-outlier enrichment `2.000x`, top-8 failure-group ROI LPIPS mean gap `0.139568`.

LPIPS and quality distances use candidate and exact formal native pixels only. Source pixels are never read by the quality-distance path; source information enters only through archived ArcFace cosine scalars.

This diagnostic does not promote an arm and is not privacy proof, a Full gate, or a formal winner.
