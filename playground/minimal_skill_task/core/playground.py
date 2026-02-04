"""Minimal Skill Task Playground：Analyze → Plan → Search → Summarize 四 Agent 流程"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evomaster.core import BasePlayground, register_playground

from .utils.rag_utils import get_db_from_description, resolve_db_to_absolute_paths
from .exp import AnalyzeExp, SearchExp, SummarizeExp


@register_playground("minimal_skill_task")
class MinimalSkillTaskPlayground(BasePlayground):
    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "agent" / "minimal_skill_task"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.analyze_agent = None
        self.plan_agent = None
        self.search_agent = None
        self.summarize_agent = None

    def setup(self) -> None:
        self.logger.info("Setting up minimal skill task playground...")

        llm_config_dict = self._setup_llm_config()
        self._llm_config_dict = llm_config_dict

        self._setup_session()

        config_dict = self.config.model_dump()
        skills_config = config_dict.get("skills", {})
        skill_registry = None
        if skills_config.get("enabled", False):
            self.logger.info("Skills enabled, loading skill registry...")
            from evomaster.skills import SkillRegistry
            skills_root = Path(skills_config.get("skills_root", "evomaster/skills"))
            skill_registry = SkillRegistry(skills_root)
            self.logger.info(f"Loaded {len(skill_registry.get_all_skills())} skills")
        self._setup_tools(skill_registry)

        agents_config = getattr(self.config, "agents", {})
        if not agents_config:
            raise ValueError("No agents configuration found. Add 'agents' section to config.")

        for name in ["analyze", "plan", "search", "summarize"]:
            if name not in agents_config:
                raise ValueError(f"缺少 agent 配置: {name}")
            cfg = agents_config[name]
            enable_tools = cfg.get("enable_tools", True)
            agent = self._create_agent(
                name=name,
                agent_config=cfg,
                enable_tools=enable_tools,
                llm_config_dict=llm_config_dict,
                skill_registry=skill_registry,
            )
            setattr(self, f"{name}_agent", agent)
            self.logger.info(f"{name.capitalize()} Agent created")

        self.logger.info("Minimal skill task playground setup complete")

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        try:
            self.setup()

            self._setup_trajectory_file(output_file)

            db = get_db_from_description(task_description)
            db = resolve_db_to_absolute_paths(db)  # vec_dir、nodes_data、model 转为绝对路径
            task_id = getattr(self, "task_id", "task_0")

            self.logger.info("============================================================")
            self.logger.info("AnalyzeExp")
            self.logger.info("============================================================")
            analyze_exp = AnalyzeExp(self.analyze_agent, self.config)
            analyze_output, analyze_traj = analyze_exp.run(task_description, db, task_id=task_id)

            self.logger.info("============================================================")
            self.logger.info("SearchExp (Plan + Search, 2 rounds)")
            self.logger.info("============================================================")
            search_exp = SearchExp(self.plan_agent, self.search_agent, self.config)
            search_results, search_trajs = search_exp.run(
                task_description, analyze_output, db, task_id=task_id
            )

            self.logger.info("============================================================")
            self.logger.info("SummarizeExp")
            self.logger.info("============================================================")
            summarize_exp = SummarizeExp(self.summarize_agent, self.config)
            summarize_output, summarize_traj = summarize_exp.run(
                task_description, search_results, db, task_id=task_id
            )

            total_steps = (
                len(getattr(analyze_traj, "steps", []))
                + sum(len(getattr(t, "steps", [])) for t in search_trajs)
                + len(getattr(summarize_traj, "steps", []))
            )

            result = {
                "status": "completed",
                "steps": total_steps,
                "analyze_output": analyze_output,
                "search_results": search_results,
                "summarize_output": summarize_output,
            }
            return result

        except Exception as e:
            self.logger.error(f"Minimal skill task failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "steps": 0,
                "error": str(e),
            }
        finally:
            self.cleanup()
