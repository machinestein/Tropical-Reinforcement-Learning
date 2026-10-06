from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class StaticEnvConfig:
    """Configuration for StaticEnv environment"""
    dataset_name: str = field(default="metamathqa") 
    cache_dir: str = field(default="./data")
    split: Optional[str] = field(default=None)