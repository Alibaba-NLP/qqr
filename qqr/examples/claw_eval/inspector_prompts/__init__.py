"""claw-eval specific inspector prompt set.

Modeled after ``qqr/examples/navigation/inspector_prompts`` but tailored to
the claw-eval multi-service sandbox agent domain (crm / helpdesk / calendar /
contacts / gmail / kb / scheduler ... driven by Qwen3.5-style tool calls).

Modules:
  * corrector_draft  - the "silent calibrator": rewrites the current step into an
                       improved, style-aligned, leakage-free next action using the
                       hidden groundtruth as guidance. (primary, selected via
                       custom-inspector-prompt-path in the shared RTI pipeline.)
  * origin_prompt     - counterfactual baseline (no privileged info); the ref-KL gate scores
                       the corrected step under this vs. the calibrator prompt.
  * deleak           - strips any reference/groundtruth/feedback leakage from a reasoning draft.
  * judge_validity   - pre-check: is the current step a reasonable next action?
  * postcheck_judge  - post-check: is the corrected step acceptable (no leak / no
                       hallucinated tool args)?

Each module exposes a ``build(...)`` function whose signature matches the shared RTI
pipeline/agents call sites (qqr/off_policy/rti).
"""

from . import corrector_draft, deleak, judge_validity, origin_prompt, postcheck_judge

__all__ = ["corrector_draft", "deleak", "judge_validity", "origin_prompt", "postcheck_judge"]
