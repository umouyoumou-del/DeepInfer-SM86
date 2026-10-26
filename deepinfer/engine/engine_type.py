import enum


class EngineType(enum.Enum):
    """
    LLMEngine type

    请和 csrc/engine/scheduler.hpp 中的 enum class EngineType 保持一致
    """

    NORMAL = 0  # process prompts & completions
    PROMPT = enum.auto()  # only process prompts
    COMPLETION = enum.auto()  # only process completions

    @property
    def default_zero_overhead_scheduling(self) -> bool:
        return self in (EngineType.NORMAL, EngineType.COMPLETION)
