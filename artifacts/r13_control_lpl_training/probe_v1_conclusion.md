# R13 control/LPL 8-step probe conclusion

Decision: **NO-GO for formal training**.

Both one-attempt probes completed 8 optimizer steps and passed the cross-arm contract. They used the same locked E15 EMA source, first 32 samples, and byte-identical flow RNG ledger. All 30 allowed tensors and Adam states were finite and nonzero, no tensor outside the allowlist changed, and the VAE was not trainable.

The blocker is objective scale. LPL raw loss was 7,473.3 times the FM objective; after weight 3 it was 22,420.0 times FM. Mean pre-clip gradient norm was 52,419.49 versus 0.88825 for control, a 59,014.6-fold ratio. Per-step clipping at 1.0 therefore controls magnitude while LPL controls direction. The high-resolution `up_block_2` and `up_block_3` terms contribute 72.0% of raw LPL.

This is a valid negative probe, not a training failure. Formal control/LPL training was not launched. The control run has a launch-provenance caveat: its exact ledger process completed outside tmux after a shell quoting error; it was not retried. The LPL run completed once in its direct tmux with exit status 0.
