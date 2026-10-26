from __future__ import annotations

from pathlib import Path

from deepinfer import AttrDict


def load_model_config(model: str) -> AttrDict:
    """加载 HF model_cfg（AttrDict）。

    `model` 必须指向 HF 目录；配置与权重来自同一目录。

    返回的 model_cfg 至少包含：
    - model_cfg.architecture：HF architectures 中声明且通过注册表校验的架构名
    - model_cfg.gpt：模型结构与运行时字段
    - model_cfg.vl_aligner：DeepSeek V4.1 多模态模型才包含；Qwen2 为纯文本模型
    """
    # HF 配置解析只处理数据，不导入模型实现。
    from deepinfer.hugging_face import load_hf_model_config

    return load_hf_model_config(Path(model) / "config.json")
