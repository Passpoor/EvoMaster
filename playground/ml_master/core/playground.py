""" ML-Master Playground：调用Draft/Debug/Improve 三个EXP"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
from evomaster.core import BasePlayground, register_playground
from evomaster.agent import Agent

from .utils.grading import validate_submission
from .utils.uct import UCTSearchConfig, UCTDecayConfig, UCTSearchManager
from .utils.data_preview import generate as generate_data_preview
from .utils.playground_helpers import (
    append_trajectory,
    build_review,
    copy_submission,
    save_best,
    save_node_snapshot,
)
from .exp.draft_exp import DraftExp
from .exp.debug_exp import DebugExp
from .exp.improve_exp import ImproveExp

logger = logging.getLogger(__name__)


@register_playground("ml_master")
class MLMasterPlayground(BasePlayground):
    """ml-master 精简版：使用 BasePlayground Session/工具/Agent 创建"""

    def __init__(self, config_dir: Path | None = None, config_path: Path | None = None):
        if config_path is None and config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "configs" / "ml_master"
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents: dict[str, Agent] = {}
        self.trajectories: list[dict[str, Any]] = []

# --------------------------- 初始化 --------------------------- #
    def setup(self) -> None:
        self.logger.info("Setting up MLMasterPlayground using BasePlayground helpers...")

        llm_config_dict = self._setup_llm_config()
        self._llm_config_dict = llm_config_dict

        self._setup_session()
        self._setup_tools()

        agents_cfg = getattr(self.config, "agents", {})
        if not agents_cfg:
            raise ValueError("config.agents 未配置draft/debug/improve/metric")

        for name in ["draft", "debug", "improve", "metric"]:
            if name not in agents_cfg:
                raise ValueError(f"缺少 agent 配置: {name}")
            cfg = agents_cfg[name]
            enable_tools = cfg.get("enable_tools", False)
            agent = self._create_agent(
                name=name,
                agent_config=cfg,
                enable_tools=enable_tools,
                llm_config_dict=llm_config_dict,
            )
            self.agents[name] = agent
            self.logger.info("Agent created: %s", name)

        # 额外：baseline.json / grade.py 软链接
        exp_id = getattr(self.config, "exp_id", None)
        data_root = getattr(self.config, "data_root", None)
        ws = Path(self.session.config.workspace_path)
        if exp_id and data_root:
            prepared = Path(data_root) / exp_id / "prepared"
            src_base = prepared / "baseline.json"
            dst_base = ws / "input" / "baseline.json"
            if src_base.exists():
                dst_base.parent.mkdir(parents=True, exist_ok=True)
                if dst_base.exists():
                    dst_base.unlink()
                dst_base.symlink_to(src_base)
            src_grade = prepared / "grade.py"
            dst_grade = ws / "grade.py"
            if src_grade.exists():
                if dst_grade.exists():
                    dst_grade.unlink()
                dst_grade.symlink_to(src_grade)

    def cleanup(self) -> None:
        super().cleanup()

    def run(self, task_description: str, output_file: str | None = None) -> dict:
        try:
            self.setup()
            self._setup_trajectory_file(output_file)
            workspace = Path(self.session.config.workspace_path)
            data_preview = generate_data_preview(workspace)
            (workspace / "working").mkdir(parents=True, exist_ok=True)
            (workspace / "best_solution").mkdir(parents=True, exist_ok=True)
            (workspace / "best_submission").mkdir(parents=True, exist_ok=True)
            submission_dir = workspace / "submission"
            submission_dir.mkdir(parents=True, exist_ok=True)

            servers = getattr(self.config, "grading_servers", []) or []
            search_cfg = UCTSearchConfig()
            search_mgr = UCTSearchManager(
                search_cfg=search_cfg,
                decay_cfg=UCTDecayConfig(),
                grader=lambda exp_id, p: validate_submission(
                    exp_id,
                    p,
                    server_urls=servers,
                    dataset_root=getattr(self.config, "data_root", None),
                ),
                exp_id=getattr(self.config, "exp_id", "unknown"),
                submission_dir=submission_dir,
            )
            search_mgr.set_snapshot_fn(
                lambda node, sub, review, reward: save_node_snapshot(
                    self.run_dir,
                    Path(self.session.config.workspace_path),
                    node,
                    sub,
                    review,
                    reward,
                    search_mgr,
                )
            )

            results: dict = {"status": "completed", "draft": [], "debug": [], "improve": []}
            best_code: Optional[str] = None
            best_metric: Optional[float] = None
            best_node_id: Optional[str] = None

            max_steps = 40
            while search_mgr.current_step < max_steps:
                target = search_mgr.select_next()
                if target is None:
                    break

                if target.stage == "root":
                    stage = "draft"
                    prev_code = ""
                    term_out = ""
                elif target.is_buggy or target.metric.value is None:
                    stage = "debug"
                    prev_code = getattr(target, "code", "")
                    term_out = getattr(target, "stdout", "")
                else:
                    stage = "improve"
                    prev_code = getattr(target, "code", "")
                    term_out = getattr(target, "stdout", "")

                node = search_mgr.create_child(target, stage=stage, plan="", code="")
                if stage == "draft":
                    exp = DraftExp(self.agents["draft"], self.agents["metric"], self.session, workspace, getattr(self.config, "exp_id", None), data_preview, node, exp_index=search_mgr.current_step)
                    res = exp.run(task_description, memory=search_mgr.root.fetch_child_memory())
                elif stage == "debug":
                    exp = DebugExp(self.agents["debug"], self.agents["metric"], self.session, workspace, getattr(self.config, "exp_id", None), data_preview, node, exp_index=search_mgr.current_step)
                    res = exp.run(task_description, prev_code=prev_code, term_out=term_out, issue="")
                else:
                    exp = ImproveExp(self.agents["improve"], self.agents["metric"], self.session, workspace, getattr(self.config, "exp_id", None), data_preview, node, exp_index=search_mgr.current_step)
                    res = exp.run(task_description, best_code=best_code or prev_code, best_metric=best_metric, memory=target.fetch_child_memory(), term_out=term_out)

                node.code = res.get("code", "")
                node.plan = res.get("plan", "")
                node.stdout = res.get("exec", {}).get("stdout", "")
                node.exit_code = res.get("exec", {}).get("exit_code", None)

                copied = copy_submission(submission_dir, node.id)
                review = build_review(res, has_submission=copied is not None)
                reward = search_mgr.ingest_result(node, review)
                save_node_snapshot(self.run_dir, Path(self.session.config.workspace_path), node, copied, review, reward, search_mgr)

                trail = {
                    "ts": datetime.utcnow().isoformat(),
                    "step": search_mgr.current_step,
                    "stage": stage,
                    "node_id": node.id,
                    "parent": getattr(node.parent, "id", None),
                    "is_buggy": node.is_buggy,
                    "metric": getattr(node.metric, "value", None),
                    "has_submission": copied is not None,
                    "submission_file": str(copied) if copied else None,
                }
                append_trajectory(self, trail, logger=self.logger)
                results[stage].append(res)

                if search_mgr.best_node and search_mgr.best_node.id != best_node_id and search_mgr.best_node.metric.value is not None:
                    best_node_id = search_mgr.best_node.id
                    best_metric = search_mgr.best_node.metric.value
                    best_code = search_mgr.best_node.code
                    best_sub = submission_dir / f"submission_{best_node_id}.csv"
                    save_best(self.logger, workspace, best_code, best_sub if best_sub.exists() else copied)

            return results
        finally:
            self.cleanup()


