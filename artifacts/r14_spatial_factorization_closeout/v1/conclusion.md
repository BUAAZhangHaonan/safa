# R12-R14 spatial-factorization closeout

The next direction is one versioned face-region inpainting feasibility probe. It is not a survivor or winner.

## Why the mainline changed

- R12 ended with no_paired_survivor. Regular32 failed the ArcFace privacy point gate, while tail32 failed full-image sharpness. No horizon passed both datasets. Native-anchor LPIPS projection and latent Fourier-shell projection were separately rejected by their preregistered rules.
- R13 did not launch formal Control/LPL training. The matched 8-step probe found weighted LPL at 22,420.0 times the flow-matching objective and pre-clip gradients at 59,014.6 times control. This is a scale-transfer NO-GO, not proof that LPL itself is scientifically invalid.
- R14 localized the tail failure. Background contributes 0.726997 at u12, 95% CI [0.677615, 0.780220], and 0.691517 at u16, CI [0.589180, 0.774769], of positive centered-Laplacian deficit. Both lower bounds exceed the preregistered 0.5 threshold. Multiscale gradients agree. Regular32 confidence intervals cross 0.5, so this is tail-specific architecture support, not a universal blur claim.

## The one allowed probe

Create a new meanflow_sit_inpaint model without changing E15 semantics. Use the unexpanded AffectNet face bbox as the feasibility mask. Remove original face pixels before context encoding, define flow and loss only inside the mask, project outside-mask latent back to context after every flow map, and force outside-mask output pixels to remain bit-exact. Training must pair embedding identity A with same-expression, different-identity target B and assert source_sample_id != target_sample_id.

Run smoke8, then one E15-EMA arm for 256 optimizer steps on GPU0-3 DDP with batch 2 per GPU. Do not search mask, learning rate, loss weight, or steps.

Stop immediately if smoke8 does not prove face removal from context, outside-mask bit-exactness, deterministic seeds, and finite masked loss/gradients. On regular32, require ArcFace exact-one 32/32, provisional representation/privacy gates, both full-image and detector-face ROI NIQE/sharpness gates, outside-mask bit-exactness, and severe 0/8. A full-image pass with ROI failure is copied-background metric inflation and is an immediate NO-GO. Any failed gate ends the route without tuning. Only an all-pass regular32 result may run one stage128; it may not jump to 512.

Milestone commits: R12 result ac4ade2, LPIPS rejection 73bc919, Fourier rejection 4bb9fbf; R13 NO-GO 8047731; R14 implementation/result 944138c, 2f110d2, 4898593.
