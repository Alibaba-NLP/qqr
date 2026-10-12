from dataclasses import dataclass, field

from slime.utils.types import Sample as BaseSample


@dataclass
class Sample(BaseSample):
    """The sample generated"""

    messages: list[dict[str, str]] = field(default_factory=list)
    response_message: dict[str, str] = None

    def to_dict(self):
        keys = [
            "group_index",
            "index",
            "messages",
            "prompt",
            "response",
            "reward",
            "response_message",
            "label",
            "status",
            "metadata",
            "train_metadata",
        ]
        value = self.__dict__.copy()
        value["status"] = self.status.value
        value = {k: value[k] for k in keys if value[k] is not None}
        return value

    @staticmethod
    def from_dict(data: dict):
        data["status"] = BaseSample.Status(data["status"])
        data["spec_info"] = BaseSample.SpecInfo.from_dict(data.get("spec_info", {}))
        return Sample(**data)
