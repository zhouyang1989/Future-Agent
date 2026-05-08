"""
config_loader.py
配置文件加载模块。支持嵌套键访问，自动回退到模板配置。
"""
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """配置包装器，支持点号分隔的嵌套键访问。"""

    def __init__(self, raw: Dict[str, Any]) -> None:
        self._raw = raw if raw is not None else {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        支持嵌套键，例如：
            config.get("knowledge_source.obsidian.vault_path")
            config.get("arxiv.categories", ["cs.CL"])
        """
        keys = key.split(".")
        value = self._raw
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    @property
    def raw(self) -> Dict[str, Any]:
        return self._raw


def load_config(path: Optional[str] = None) -> Config:
    """
    加载 YAML 配置，优先级：
        1. 显式传入的 path
        2. ./config.local.yaml（用户私有，已被 .gitignore 保护）
        3. ./config.yaml（仓库模板）
    """
    if path:
        config_path = Path(path)
    else:
        local_path = Path("config.local.yaml")
        template_path = Path("config.yaml")
        config_path = local_path if local_path.exists() else template_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件。请复制 config.yaml 为 config.local.yaml 并修改。\n"
            f"查找路径: {config_path.absolute()}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(raw=raw)