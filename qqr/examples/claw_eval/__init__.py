"""claw-eval example module for qqr.

RL training targeting the **claw-eval** benchmark (an external repo pointed to
by the ``CLAW_EVAL_ROOT`` env var). The agent loop, tool dispatch, E2B sandbox,
mock services and simulated user are reused from claw-eval via the
``qqr.tools.claw_eval`` bridge; this module adapts them into the qqr rollout / reward
pipeline.

Public interface (referenced by yaml configs / training scripts):
  qqr.examples.claw_eval.rollout.generate
      – per-turn rollout entry point (SGLang-backed agent loop)
  qqr.examples.claw_eval.reward_model.group_reward_by_official_grader
      – group reward model (official claw-eval grader score per trajectory)
  qqr.examples.claw_eval.reward_model.correct_group_norm
      – per-group GRPO advantage normalization post-process
  qqr.examples.claw_eval.group_rollout.generate_and_rm_group_with_rti
      – two-stage RTI self-correction group rollout (optional, RTI_ENABLED)

Submodules are imported lazily (yaml uses fully-qualified paths), so importing
this package stays cheap and free of heavy side effects.
"""

__all__ = [
    "generate",
    "generate_and_rm_group_with_rti",
]


def __getattr__(name):
    # Lazy re-export to keep package import side-effect free.
    if name == "generate":
        from .rollout import generate

        return generate
    if name == "generate_and_rm_group_with_rti":
        from .group_rollout import generate_and_rm_group_with_rti

        return generate_and_rm_group_with_rti
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
