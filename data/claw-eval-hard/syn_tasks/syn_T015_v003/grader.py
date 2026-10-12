"""syn_T015_v003 grader — synthetic variant of T015zh_kb_search."""

from claw_eval.graders.base import load_peer_grader

_Base = load_peer_grader("T015zh_kb_search")


class SyntheticT015Grader(_Base):
    """Synthetic variant — overrides generated for this task."""

    # set required: peer grade() computes `articles_read & RELEVANT_ARTICLES`
    # (set intersection); a list would raise TypeError and zero the score.
    RELEVANT_ARTICLES = {"kb_001", "kb_003", "kb_005", "kb_006", "kb_007"}
