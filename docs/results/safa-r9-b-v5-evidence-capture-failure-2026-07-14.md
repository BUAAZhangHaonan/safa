# R9 B v5 evidence-capture failure

Campaign `r9-report-only-formal-v5` is invalid for continuation because both evaluator smoke controller logs were redirected to `/tmp` at process launch and copied into the artifact roots only after completion. The evaluator computations succeeded, but the copied logs do not satisfy the immutable direct-capture evidence contract.

- ArcFace source log mtime: `2026-07-14 21:31:27.890727639 +0800`; copied artifact log mtime: `2026-07-14 21:33:32.722703687 +0800`.
- Quality source log mtime: `2026-07-14 21:32:16.354718341 +0800`; copied artifact log mtime: `2026-07-14 21:33:32.722703687 +0800`.
- ArcFace result SHA256: `e0570a51e40a28c9e3043909691839844b15a369bf02ee591ed786d40726dba6`.
- Quality result SHA256: `9121ba1e465d48b4bc5deabbe936f35b8c500e1d14a687aeb77d2e08d2ce79d2`.
- A continuation contract was materialized before the capture failure was noticed. Its contract SHA256 is `f7cc494fb1baa9d98e06af1c3935534b6f33192e373cc9530ba64a11b8e97a7a`; its file SHA256 is `97da33c7e9d720d2b979259b94ddd3967167a7171f7d58d7d8f21773bc5e3804`.
- The frozen v5 root contains 16 files. Its sorted path-and-content inventory SHA256 is `f21bb726fd2ad176d997dd20fd37de8ef382bea1d5785e4d15f20855bdefb6b1`.

All v5 artifacts are read-only, no v5 process remains active, and v5 must never be resumed. Campaign v6 supersedes it with new artifact roots whose `controller.log` files are opened by the tmux launch command before each smoke process starts.
