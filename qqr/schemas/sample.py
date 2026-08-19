from dataclasses import dataclass, field

from slime.utils.types import Sample as BaseSample


@dataclass
class Sample(BaseSample):
    """The sample generated"""

    messages: list[dict[str, str]] = field(default_factory=list)
    response_message: dict[str, str] = None
