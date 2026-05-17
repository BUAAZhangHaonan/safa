from __future__ import annotations

import unittest

from safa.training.g_loop import _assert_stage1_gate_allows_stage2


class StageGateTests(unittest.TestCase):
    def test_blocks_stage2_when_detection_rate_missing(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 1}}
        with self.assertRaises(RuntimeError):
            _assert_stage1_gate_allows_stage2(stages, stable_hits=0, detection_rate=None, allow_bypass=False)

    def test_blocks_stage2_when_detection_gate_not_stable(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 2}}
        with self.assertRaises(RuntimeError):
            _assert_stage1_gate_allows_stage2(stages, stable_hits=1, detection_rate=0.99, allow_bypass=False)

    def test_allows_stage2_after_gate(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 2}}
        _assert_stage1_gate_allows_stage2(stages, stable_hits=2, detection_rate=0.99, allow_bypass=False)

    def test_smoke_bypass_is_explicit(self) -> None:
        stages = {"stage1": {"require_face_detection_gate": True, "face_detection_threshold": 0.95, "stable_epochs": 1}}
        _assert_stage1_gate_allows_stage2(stages, stable_hits=0, detection_rate=0.0, allow_bypass=True)


if __name__ == "__main__":
    unittest.main()
