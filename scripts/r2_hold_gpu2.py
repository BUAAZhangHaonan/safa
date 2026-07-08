#!/usr/bin/env python3
"""Round 2 hold sessions: simple sleep to occupy GPUs without wasting compute.
Useful when 4 GPUs need to stay claimed but no immediate experiment pending."""
import os
import time

GPU = os.environ.get("HOLD_GPU", "0")
DURATION = int(os.environ.get("HOLD_DURATION", "3600"))  # default 1h

print(f"[hold] GPU={GPU} duration={DURATION}s")
end = time.time() + DURATION
while time.time() < end:
    time.sleep(60)
print(f"[hold] DONE")
