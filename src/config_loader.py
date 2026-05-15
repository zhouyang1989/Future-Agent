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


def project_root() -> Path:
    """基于当前文件位置推导项目根目录。"""
    return Path(__file__).resolve().parent.parent


def resolve_config_path(path_str: Optional[str]) -> Optional[Path]:
    """
    解析配置文件中的路径。
    相对路径基于项目根目录解析，绝对路径或用户主目录缩写保持不变。
    """
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = project_root() / p
    return p.resolve()


def load_config(path: Optional[str] = None) -> Config:
    """
    加载 YAML 配置，优先级：
        1. 显式传入的 path
        2. <项目根目录>/config.local.yaml（用户私有，已被 .gitignore 保护）
        3. <项目根目录>/config.yaml（仓库模板）
    """
    if path:
        config_path = Path(path)
    else:
        root = project_root()
        local_path = root / "config.local.yaml"
        template_path = root / "config.yaml"
        config_path = local_path if local_path.exists() else template_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件。请复制 config.yaml 为 config.local.yaml 并修改。\n"
            f"查找路径: {config_path.absolute()}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(raw=raw)