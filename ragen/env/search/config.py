"""
Configuration for the Search (HotpotQA) environment.

The search environment is adapted from the RLLM project:
  https://github.com/rllm-org/rllm
  License: Apache-2.0
"""

from dataclasses import dataclass


@dataclass
class SearchEnvConfig:
    """Configuration for SearchEnv.

    Fields under env_config in config/envs.yaml map directly to these fields.
    """

    dataset_name: str = "hotpotqa"
    train_path: str = "data/search/train.parquet"
    max_instances: int = 20000

    retrieval_server_url: str = "http://127.0.0.1:8000"
    retrieval_timeout: float = 30.0
    max_search_results: int = 5
    max_total_chars: int = 4000  

    max_steps: int = 10       
    render_mode: str = "text"
    mock_mode: bool = False   

    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    f1_threshold: float = 0.3
