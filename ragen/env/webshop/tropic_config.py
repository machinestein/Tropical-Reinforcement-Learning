"""Opt-in WebShop settings; the original environment config is untouched."""

from dataclasses import dataclass

from .config import WebShopEnvConfig


@dataclass
class WebShopTropicEnvConfig(WebShopEnvConfig):
    max_steps: int = 9
