#!/usr/bin/env python3
"""Persistent GPU hold: allocate buffer + light compute to claim GPU memory."""
import os
import time
import torch

DURATION = int(os.environ.get("HOLD_DURATION", "3600"))
print(f"[gpu_hold] duration={DURATION}s", flush=True)

device = torch.device("cuda:0")
# Allocate ~1GB persistent tensor
buf = torch.randn(8192, 8192, device=device, dtype=torch.float32)
print(f"[gpu_hold] allocated {buf.numel()*4/1e9:.2f} GB", flush=True)

end = time.time() + DURATION
i = 0
while time.time() < end:
    buf = buf * 1.0001 + 0.0001
    if i % 100 == 0:
        torch.cuda.synchronize()
    i += 1
print("[gpu_hold] DONE", flush=True)
