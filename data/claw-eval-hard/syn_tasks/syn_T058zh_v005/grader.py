"""syn_T058zh_v005 grader — synthetic variant of T058zh_painting_identification."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T058zh_painting_identification")


class SyntheticT058zhv005Grader(_Base):
    """Synthetic variant — oracle fixture is task-local; no overrides needed."""

    pass
