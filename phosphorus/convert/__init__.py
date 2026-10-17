"""Phosphorus → HuggingFace 格式转换（T5-C8）。

仅暴露 `hf_export.export_to_hf`；torch 为惰性导入（包导入零 torch 依赖）。
"""

from .hf_export import export_to_hf

__all__ = ["export_to_hf"]
