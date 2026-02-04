"""Playground 注册表

提供装饰器机制，用于注册各 agent 的自定义 Playground 类。

使用示例：
    from evomaster.core import BasePlayground, register_playground

    @register_playground("my-agent")
    class MyAgentPlayground(BasePlayground):
        pass

    # 在 run.py 中
    from evomaster.core import get_playground_class
    playground = get_playground_class("my-agent", config_dir=...)
"""

import logging
from typing import Dict, Type, Optional
from pathlib import Path

# 全局注册表：存储 agent_name -> Playground 类的映射
_PLAYGROUND_REGISTRY: Dict[str, Type] = {}

logger = logging.getLogger(__name__)


def register_playground(agent_name: str):
    """装饰器：注册 Playground 类

    使用示例：
        @register_playground("agent-builder")
        class AgentBuilderPlayground(BasePlayground):
            pass

    Args:
        agent_name: Agent 名称（如 "minimal", "agent-builder"）
                   必须与 playground 目录名一致（使用连字符）

    Returns:
        装饰器函数
    """
    def decorator(cls):
        if agent_name in _PLAYGROUND_REGISTRY:
            logger.warning(
                f"Playground '{agent_name}' 已被注册，将被覆盖: "
                f"{_PLAYGROUND_REGISTRY[agent_name].__name__} -> {cls.__name__}"
            )

        _PLAYGROUND_REGISTRY[agent_name] = cls
        logger.debug(f"Registered playground: {agent_name} -> {cls.__name__}")
        return cls

    return decorator


def get_playground_class(agent_name: str, config_dir: Optional[Path] = None, config_path: Optional[Path] = None):
    """获取注册的 Playground 类实例

    如果 agent 有注册的自定义 Playground 类，则使用自定义类；
    否则回退到 BasePlayground。

    Args:
        agent_name: Agent 名称
        config_path: 配置文件完整路径（推荐使用）

    Returns:
        Playground 实例

    Raises:
        ImportError: 如果 BasePlayground 无法导入（内部错误）
    """
    from .playground import BasePlayground

    playground_class = _PLAYGROUND_REGISTRY.get(agent_name)

    if playground_class:
        # 使用注册的自定义类
        logger.info(f"Using custom Playground: {agent_name} -> {playground_class.__name__}")
        return playground_class(config_dir=config_dir, config_path=config_path)
    else:
        # 回退到 BasePlayground
        logger.info(f"Using BasePlayground for agent '{agent_name}' (no custom implementation registered)")
        return BasePlayground(config_dir=config_dir, config_path=config_path)


def list_registered_playgrounds():
    """列出所有注册的 Playground

    Returns:
        已注册的 agent 名称列表
    """
    return list(_PLAYGROUND_REGISTRY.keys())


def get_registry_info():
    """获取注册表的详细信息

    Returns:
        字典，格式为 {agent_name: class_name}
    """
    return {
        agent_name: cls.__name__
        for agent_name, cls in _PLAYGROUND_REGISTRY.items()
    }
