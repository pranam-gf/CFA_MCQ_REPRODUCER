"""
Configuration loader for different prompting strategies.
"""

from .default_config import ALL_MODEL_CONFIGS as DEFAULT_CONFIGS
from .cot_config import ALL_MODEL_CONFIGS_COT as COT_CONFIGS
from .self_discover_config import SELF_DISCOVER_CONFIGS

__all__ = [
    'DEFAULT_CONFIGS',
    'COT_CONFIGS',
    'SELF_DISCOVER_CONFIGS'
]