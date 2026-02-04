import logging
import json
from typing import Any
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from evomaster import TaskInstance
from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from .utils import strip_think_and_exec, extract_agent_response


class RewriteExp(BaseExp):
    """X-Master中Rewrite实验类实现

    实现Rewrite阶段工作流：汇总前一模块的所有答案，重写相同数量的答案
    """

    @property
    def exp_name(self) -> str:
        """返回实验阶段名称"""
        return "Rewriting"

    def __init__(self, rewriter_agent,  config, agent_num = 5, max_workers=5):
        """初始化RewriteExp实验类

        Args:
            rewriter_agent: Rewriter Agent 实例
            config: EvoMasterConfig 实例
            agent_num: 希望工作agent数量
            max_workers: 并行处理最大线程数， 如果不进行并行操作则将max_workers置为1
        """
        super().__init__(rewriter_agent, config)
        self.rewriter = rewriter_agent
        self.agent_num = agent_num
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_workers = max_workers


    def run(self, task_description:str,task_id:str = "exp_001",solutions:List[str]=None) -> dict:
        """运行rewriter实验

        工作流: agent_num个Rewriter Agent对前一个模块的所有答案进行汇总并重写

        Args:
            task_description: 任务描述
            task_id: 任务 ID
            solutions: 接收到来自前一个模块的所有答案
        Returns:
            执行结果字典
        """
        results = {
            'task_id':task_id,
            'steps':0,
            'task_description': task_description,
            'status': 'running',
        } 

        if solutions is None:
            self.logger.error(f"Rewriter-agent task execution failed: Solutions is None", exc_info=True)
            results['status'] = 'failed'
            results['error'] = "Solutions is None"
            return super().run(task_description, task_id)
        
        if len(solutions) != self.agent_num:
            error_msg = f"Number of solutions ({len(solutions)}) does not match number of rewriter agents ({self.agent_num})"
            self.logger.error(f"Rewriter-agent task execution failed: {error_msg}", exc_info=True)
            raise ValueError(error_msg)

        try:
            if self.rewriter:
                self.logger.info("="*60)
                self.logger.info(f"Rewrite : Rewriting {self.agent_num} solutions in parallel...")
                self.logger.info("=" * 60)

                rewriter_task = TaskInstance(
                    task_id = f"{task_id}_rewriter",
                    task_type = "rewriter",
                    description=task_description,
                    input_data={},
                )

                original_format_kwargs = self.rewriter._prompt_format_kwargs.copy()

                s_solutions = self._format_solutions_prompt(solutions)

                for i in range(self.agent_num):
                    task_index = i
                    try:
                        # 设置当前exp信息，用于trajectory记录
                        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=i)
                        self.rewriter._prompt_format_kwargs.update({
                            's_solutions':s_solutions
                        })
                        rewriter_trajectory = self.rewriter.run(rewriter_task)
                        results[f'rewriter_trajectory_{i}'] = rewriter_trajectory
                        rewriter_result = extract_agent_response(rewriter_trajectory)
                        results[f'rewriter_result_{i}'] = rewriter_result
                        self.rewriter.reset_context()


                    except Exception as e:
                        print(f"Task {i} failed: {e}")
                        results[f'rewriter_trajectory_{i}'] = None
                        results[f'rewriter_result_{i}'] = None

                # with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                #     future_to_index = {}
                #     for i in range(self.agent_num):
                #         task_index = i
                #         self.rewriter._prompt_format_kwargs.update({
                #             's_solutions':s_solutions
                #         })
                #         future = ex.submit(self._run_rewriter_task,rewriter_task)
                #         future_to_index[future] = task_index
                #     for future in as_completed(future_to_index):
                #         i = future_to_index[future]
                #         try:

                #             rewriter_trajectory = future.result()
                #             results[f'rewriter_trajectory_{i}'] = rewriter_trajectory
                #             rewriter_result = extract_agent_response(rewriter_trajectory)
                #             results[f'rewriter_result_{i}'] = rewriter_result
                #         except Exception as e:
                #             print(f"Task {i} failed: {e}")
                #             results[f'rewriter_trajectory_{i}'] = None
                #             results[f'rewriter_result_{i}'] = None
                
                self.rewriter._prompt_format_kwargs = original_format_kwargs

                self.logger.info("Rewriting completed")
            

            results['status'] = 'completed'
            self.logger.info("rewriter-agent task execution completed")

            self.results.append(results)
        except Exception as e:
            self.logger.error(f"rewriter-agent task execution failed: {e}", exc_info=True)
            results['status'] = 'failed'
            results['error'] = str(e)

            self.results.append(results)
        
        return results

    def _format_solutions_prompt(self, solutions:List[str]) -> str:
        """格式化解决方案列表为prompt

        Args:
            solutions: 方案列表
        Reture:
            返回的方案prompt:
            格式：
            ## Student 1's Solution
            {solution_1}
            ## Student 2's Solution
            {solution_2}
            ## Student 3's Solution
            {solution_3}
            ...
            ...
        """

        if not solutions:
            return "No solutions"

        prompt_lines = []
        for i, solution in enumerate(solutions,1):
            # 使用 strip_think_and_exec 清理每个 solution
            clean_solution = strip_think_and_exec(solution)
            if not clean_solution:
                clean_solution = "empty solution"
            prompt_lines.append(f"## Student {i}'s Solution")
            prompt_lines.append(clean_solution)
            prompt_lines.append("")

        return "\n".join(prompt_lines).strip()

    def _run_rewriter_task(self, rewrite_task: TaskInstance) -> str:
        """包装rewriter.run()以便在线程中执行

        Args:
            rewrite_task: 初始问题
        Return:
            重写后agent的解决方案
        """
        return self.rewriter.run(rewrite_task)


    def save_results(self, output_file: str):
        """保存实验结果

        Args:
            output_file: 输出文件路径
        """
        import json
        from pathlib import Path

        output_data = []
        
        for result in self.results:
            # 为每个任务创建一个记录，包含所有轨迹
            task_record = {
                "task_id": result.get("task_id", "unknown"),
                "status": result.get("status", "unknown"),
                "steps": 0,  # 先初始化为0
            }
            
            # 收集所有轨迹
            trajectories = {}
            results = {}
            total_steps = 0
            
            for i in range(self.agent_num):
                trajectory_key = f"rewriter_trajectory_{i}"
                result_key = f"rewriter_result_{i}"
                
                if trajectory_key in result and result[trajectory_key]:
                    traj = result[trajectory_key]
                    # 存储轨迹数据
                    if hasattr(traj, 'model_dump'):
                        trajectories[trajectory_key] = traj.model_dump()
                    else:
                        # 如果traj不是Model对象，直接存储
                        trajectories[trajectory_key] = traj
                    
                    # 更新总步数
                    total_steps += len(traj.steps) if hasattr(traj, 'steps') else 0
                    
                    # 更新任务状态（如果有成功的就用成功的状态）
                    if traj.status == "success" and task_record["status"] != "success":
                        task_record["status"] = traj.status
                    
                if result_key in result and result[result_key]:
                    res = result[result_key]
                    # 存储轨迹数据
                    if hasattr(res, 'model_dump'):
                        results[result_key] = res.model_dump()
                    else:
                        # 如果traj不是Model对象，直接存储
                        results[result_key] = res
            
            # 如果没有找到任何轨迹，创建空记录
            if not trajectories:
                task_record["trajectories"] = {}
                task_record["steps"] = result.get("steps", 0)
                
                # 添加任务的其他信息
                if "rewriter_result" in result:
                    task_record["rewriter_result"] = result.get("rewriter_result")
            else:
                task_record["trajectories"] = trajectories
                task_record["steps"] = total_steps


            if not results:
                task_record["results"] = {}
                
                # 添加任务的其他信息
                if "rewriter_result" in result:
                    task_record["rewriter_result"] = result.get("rewriter_result")
            else:
                task_record["results"] = results
            
            # 添加元数据
            task_record["agent_num"] = self.agent_num
            task_record["meta"] = {
                "agent_version": "1.0",
                "task_type": "multi_agent"
            }
            
            output_data.append(task_record)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info(f"Results saved to {output_file}")
