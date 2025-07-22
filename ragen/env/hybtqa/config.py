from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class HybTQAEnvConfig:
    """Configuration for FrozenLake environment"""
    # Map config
    dataset_path: str = field(default="./data/hybtqa/train.json")
    retriever_path: str = field(default="/root/autodl-tmp/qwen3-embedding-0.6b")
    # cache_dir:str = field(default="./data")
    # split: str = field(default="train")
