"""Generic paged continuous-batching scheduler.

The scheduler uses only the runtime surface ``model``, ``tokenizer``,
``device``, ``max_cache``, ``forward_logits`` and ``_sample``; it is shared by
the DeepSeek MoE runtime and other DeepInfer runtimes.
"""

from deepinfer.scheduler.paged import (
    GenerationResult,
    PagedBatchScheduler,
    PagedScheduledRequest,
)

__all__ = ["GenerationResult", "PagedBatchScheduler", "PagedScheduledRequest"]
