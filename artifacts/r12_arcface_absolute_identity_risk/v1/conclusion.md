# R12 ArcFace absolute identity-risk diagnostic

Predeclared classification: `identity_risk_inconclusive`.

- `u12`: locked relative delta mean `0.029659`; candidate Recall@1/5 `0.0000/0.1875`; candidate-minus-native percentile delta `0.152778` (95% CI `0.070437, 0.238591`); top-8 Recall@5/positive-margin enrichment `0.667x/0.000x`.
- `u16`: locked relative delta mean `0.021727`; candidate Recall@1/5 `0.0312/0.1250`; candidate-minus-native percentile delta `0.103671` (95% CI `0.020337, 0.194940`); top-8 Recall@5/positive-margin enrichment `1.000x/4.000x`.

The locked source-candidate minus source-native gate remains unchanged at `<= 0.02`.
This closed-set diagnostic only distinguishes baseline-conditioned relative-metric geometry from supported retrieval leakage on the existing 64-source gallery.

Source pixels are read only by the locked buffalo_l identity analyzer. No image was generated and no model was trained. This result is not a privacy proof, a Full gate, or a formal winner.
