"""多智能体 Playground 实现

展示如何使用多个Agent协作完成任务。
包含Planning Agent和Coding Agent的工作流。
"""

import logging
import sys
from pathlib import Path

# 确保可以导入evomaster模块
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground
from evomaster.agent import copy_agent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evomaster.agent import Agent

from .exp import MultiAgentExp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import List, Any, Callable

@register_playground("minimal_multi_agent_parallel")
class MultiAgentParallelPlayground(BasePlayground):
    """多智能体 Playground

    实现Planning Agent和Coding Agent的协作工作流：
    1. Planning Agent分析任务并制定计划
    2. Coding Agent根据计划执行代码任务

    使用方式：
        # 通过统一入口
        python run.py --agent minimal_multi_agent --task "任务描述"

        # 或使用独立入口
        python playground/minimal_multi_agent/main.py
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        """初始化多智能体 Playground

        Args:
            config_dir: 配置目录路径，默认为 configs/minimal_multi_agent/
            config_path: 配置文件完整路径（如果提供，会覆盖 config_dir）
        """
        if config_path is None and config_dir is None:
            # 默认配置目录
            config_dir = Path(__file__).parent.parent.parent.parent / "configs" / "agent" / "minimal_multi_agent"

        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 存储多个Agent
        self.planning_agent = None
        self.coding_agent = None
        
        # 从配置中读取并行配置
        session_config = self.config.session.get("local", {})
        parallel_config = session_config.get("parallel", {})
        if parallel_config.get("enabled", False):
            self.max_workers = parallel_config.get("max_parallel", 3)
        else:
            self.max_workers = 3
        
        # 初始化mcp_manager（BasePlayground.cleanup需要）
        self.mcp_manager = None

    def setup(self) -> None:
        """初始化所有组件

        覆盖基类方法，复用基类的公共方法来创建多个Agent。
        每个Agent使用独立的LLM实例，确保日志记录独立。
        """
        self.logger.info("Setting up multi-agent playground...")

        # 1. 准备 LLM 配置（每个Agent会创建独立的LLM实例）
        llm_config_dict = self._setup_llm_config()
        self._llm_config_dict = llm_config_dict  # 保存配置供后续使用

        # 2. 创建 Session（所有Agent共享）
        self._setup_session()

        # 3. 加载 Skills（如果启用）
        skill_registry = None
        config_dict = self.config.model_dump()
        skills_config = config_dict.get("skills", {})
        if skills_config.get("enabled", False):
            self.logger.info("Skills enabled, loading skill registry...")
            from pathlib import Path
            from evomaster.skills import SkillRegistry

            skills_root = Path(skills_config.get("skills_root", "evomaster/skills"))
            skill_registry = SkillRegistry(skills_root)
            self.logger.info(f"Loaded {len(skill_registry.get_all_skills())} skills")

        # 4. 创建工具注册表并初始化 MCP 工具（传入 skill_registry）
        self._setup_tools(skill_registry)

        # 5. 创建多个Agent（每个Agent使用独立的LLM实例）
        agents_config = getattr(self.config, 'agents', {})
        if not agents_config:
            raise ValueError(
                "No agents configuration found. "
                "Please add 'agents' section to config.yaml"
            )

        # 创建Planning Agent（使用独立的LLM实例）
        if 'planning' in agents_config:
            planning_config = agents_config['planning']
            self.planning_agent = self._create_agent(
                name="planning",
                agent_config=planning_config,
                enable_tools=planning_config.get('enable_tools', False),
                llm_config_dict=llm_config_dict,
                skill_registry=skill_registry,  # 传递 skill_registry
            )
            self.logger.info("Planning Agent created")

        # 创建Coding Agent（使用独立的LLM实例）
        if 'coding' in agents_config:
            coding_config = agents_config['coding']
            self.coding_agent = self._create_agent(
                name="coding",
                agent_config=coding_config,
                enable_tools=coding_config.get('enable_tools', True),
                llm_config_dict=llm_config_dict,
                skill_registry=skill_registry,  # 传递 skill_registry
            )
            self.logger.info("Coding Agent created")

        self.logger.info("Multi-agent playground setup complete")

    def _create_exp(self, exp_index):
        """创建多智能体实验实例

        覆盖基类方法，创建 MultiAgentExp 实例。
        为每个 exp 创建独立的 Agent 副本，确保并行运行时上下文不冲突。

        Args:
            exp_index: 实验索引

        Returns:
            MultiAgentExp 实例
        """
        # 为每个 exp 创建独立的 Agent 副本
        # 这些副本共享 llm, session, tools, skill_registry 等配置
        # 但拥有独立的上下文（context_manager, current_dialog, trajectory 等）
        planning_agent_copy = copy_agent(
            self.planning_agent, 
            new_agent_name=f"planning_exp_{exp_index}"
        ) if self.planning_agent else None
        
        coding_agent_copy = copy_agent(
            self.coding_agent, 
            new_agent_name=f"coding_exp_{exp_index}"
        ) if self.coding_agent else None
        
        exp = MultiAgentExp(
            planning_agent=planning_agent_copy,
            coding_agent=coding_agent_copy,
            config=self.config,
            exp_index=exp_index
        )
        # 传递 run_dir 给 Exp
        if self.run_dir:
            exp.set_run_dir(self.run_dir)
        return exp

    def execute_parallel_tasks(self, tasks: List[Callable], max_workers: int = 3) -> List[Any]:
            """通用并行任务执行器

            Args:
                tasks: 这里的每个元素应该是一个可调用的对象。
                    如果是带参数的函数，请使用 functools.partial 封装。
                    例如: [partial(exp1.run, task="A"), partial(exp2.run, task="B")]
                max_workers: 最大并行线程数

            Returns:
                List[Any]: 按照输入 tasks 的顺序返回结果列表。
                        如果任务抛出异常，结果列表中对应位置将是该 Exception 对象。
            """
            self.logger.info(f"Starting parallel execution of {len(tasks)} tasks with {max_workers} workers.")
            
            results = [None] * len(tasks)
            
            # 检查是否启用了并行资源分配
            session_config = self.config.session.get("local", {})
            parallel_config = session_config.get("parallel", {})
            parallel_enabled = parallel_config.get("enabled", False)
            
            # 检查是否启用了 split_workspace_for_exp
            split_workspace = parallel_config.get("split_workspace_for_exp", False)
            
            # 包装任务函数，设置并行索引和独立工作空间
            def wrap_task(task_func, parallel_index):
                def wrapped():
                    try:
                        # 如果启用了并行资源分配，设置 session 的并行索引
                        if parallel_enabled and self.session is not None:
                            from evomaster.agent.session.local import LocalSession
                            if isinstance(self.session, LocalSession):
                                self.session.set_parallel_index(parallel_index)
                                self.logger.debug(f"设置并行索引: {parallel_index}")
                                
                                # 如果启用了 split_workspace_for_exp，为当前 exp 创建独立工作空间
                                if split_workspace:
                                    import os
                                    main_workspace = self.session.config.workspace_path
                                    exp_workspace = os.path.join(main_workspace, f"exp_{parallel_index}")
                                    # 通过 env 创建 exp 工作空间（含软链接）
                                    self.session._env.setup_exp_workspace(exp_workspace)
                                    # 设置线程本地的工作空间路径
                                    self.session.set_workspace_path(exp_workspace)
                                    self.logger.info(
                                        f"Exp {parallel_index} 使用独立工作空间: {exp_workspace}"
                                    )
                        return task_func()
                    finally:
                        # 清理线程本地状态
                        if parallel_enabled and self.session is not None:
                            from evomaster.agent.session.local import LocalSession
                            if isinstance(self.session, LocalSession):
                                self.session.set_parallel_index(None)
                                if split_workspace:
                                    self.session.set_workspace_path(None)
                return wrapped
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务，建立 future 到 index 的映射，以保证返回顺序
                wrapped_tasks = [wrap_task(task, i) for i, task in enumerate(tasks)]
                future_to_index = {executor.submit(wrapped_task): i for i, wrapped_task in enumerate(wrapped_tasks)}

                # 处理完成的任务
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        # 获取返回值
                        result = future.result()
                        results[index] = result
                    except Exception as exc:
                        self.logger.error(f"Task {index} generated an exception: {exc}")
                        # 将异常对象作为结果返回，避免打断其他任务
                        results[index] = exc

            self.logger.info("Parallel execution completed.")
            return results

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        """运行工作流（覆盖基类方法）

        Args:
            task_description: 任务描述
            output_file: 结果保存文件（可选，如果设置了 run_dir 则自动保存到 trajectories/）

        Returns:
            运行结果
        """
        try:
            self.setup()
            self._setup_trajectory_file(output_file)
            task_description_1 = task_description
            task_description_2 = task_description
            task_description_3 = task_description
            # task_description_2 = "通过python代码读取和打印系统环境变量CUDA_VISIBLE_DEVICES的值以及目前可见的cpu的数量，然后结束。注意，不要尝试修改环境变量，你只能读取和打印。"
            # task_description_3 = "通过python代码读取和打印系统环境变量CUDA_VISIBLE_DEVICES的值以及目前可见的cpu的数量，然后结束。注意，不要尝试修改环境变量，你只能读取和打印。"   
            # --- 关键步骤：创建任务列表 ---
            task_descriptions = [task_description_1, task_description_2, task_description_3]
            tasks = []
            for i in range(self.max_workers):
                exp = self._create_exp(exp_index=i)
                
                task_func = partial(exp.run, task_description=task_descriptions[i])
                
                tasks.append(task_func)
            
            # --- 调用封装好的并行函数 ---
            results = self.execute_parallel_tasks(tasks, max_workers=self.max_workers)
            
            result = {
                "status": "completed",
                "steps": 0,
            }
            return result

        finally:
            self.cleanup()

    # def run(self, task_description: str, output_file: str | None = None) -> dict:
    #     """运行工作流（覆盖基类方法）

    #     Args:
    #         task_description: 任务描述
    #         output_file: 结果保存文件（可选，如果设置了 run_dir 则自动保存到 trajectories/）

    #     Returns:
    #         运行结果
    #     """
    #     try:
    #         self.setup()

    #         # 设置轨迹文件路径
    #         self._setup_trajectory_file(output_file)

    #         # 创建并运行实验
    #         exp = self._create_exp(exp_index=0)

    #         self.logger.info("Running experiment...")
    #         # 如果有 task_id，传递给 exp.run()
    #         task_id = getattr(self, 'task_id', None)
    #         if task_id:
    #             result = exp.run(task_description, task_id=task_id)
    #         else:
    #             result = exp.run(task_description)

    #         return result

    #     finally:
    #         self.cleanup()

