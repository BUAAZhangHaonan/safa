"""Frozen-generator guidance algorithms for SAFA."""

from safa.guidance.meanflow_flow_map import (
    CountedFlowMap,
    GuidanceResult,
    assert_guidance_stack_frozen,
    freeze_guidance_stack,
    normalize_per_sample_to_velocity_norm,
    optimize_initial_noise,
    project_fixed_radius,
    project_gaussian_typical_shell,
    sample_official_head_current_xt,
    sample_paper_algorithm_split,
    select_t_cut,
    semigroup_probe,
    symmetric_relative_l2,
)

__all__ = [
    "CountedFlowMap",
    "GuidanceResult",
    "assert_guidance_stack_frozen",
    "freeze_guidance_stack",
    "normalize_per_sample_to_velocity_norm",
    "optimize_initial_noise",
    "project_fixed_radius",
    "project_gaussian_typical_shell",
    "sample_official_head_current_xt",
    "sample_paper_algorithm_split",
    "select_t_cut",
    "semigroup_probe",
    "symmetric_relative_l2",
]
