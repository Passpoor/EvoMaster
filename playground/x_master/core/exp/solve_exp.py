import logging
import json
from typing import Any
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from evomaster import TaskInstance
from evomaster.agent import BaseAgent
from evomaster.core.exp import BaseExp
from .utils import extract_agent_response

class SolveExp(BaseExp):
    """X-Master中Solve实验类实现

    实现Solve阶段工作流：分析任务并得到问题结果
    """

    @property
    def exp_name(self) -> str:
        """返回实验阶段名称"""
        return "Solving"

    def __init__(self, solver_agent, config, agent_num=5, max_workers=5):
        """初始化SolveExp实验类

        Args:
            solver_agent: Solver Agent 实例
            config: EvoMasterConfig 实例
            agent_num: 希望工作agent数量
            max_workers: 并行处理最大线程数， 如果不进行并行操作则将max_workers置为1
        """

        super().__init__(solver_agent, config)
        self.solver = solver_agent
        self.agent_num = agent_num
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_workers = max_workers

    def run(self, task_description:str, task_id:str = "exp_001") -> dict:
        """运行solver实验

        工作流: agent_num个Solver Agent对同一个原始问题进行分析并得到初始答案

        Args:
            task_description: 任务描述
            task_id: 任务 ID
        
        Returns:
            执行结果字典
        """
        self.logger.info("Starting XMaster task execution")
        self.logger.info(f"Task:{task_description}")

        results = {
            'task_id':task_id,
            'steps':0,
            'task_description': task_description,
            'status': 'running',
        } 

        try:
            if self.solver:
                self.logger.info("="*60)
                self.logger.info(f"Solver : Generating {self.agent_num} solutions in parallel...")
                self.logger.info("=" * 60)

                solver_task = TaskInstance(
                    task_id = f"{task_id}_solver",
                    task_type = "solver",
                    description=task_description,
                    input_data={},
                )
                
                ## 串行处理
                for i in range(self.agent_num):
                    task_index = i
                    try:
                        # 设置当前exp信息，用于trajectory记录
                        BaseAgent.set_exp_info(exp_name=self.exp_name, exp_index=i)
                        solver_trajectory = self.solver.run(solver_task)
                        results[f'solver_trajectory_{i}'] = solver_trajectory
                        solver_result = extract_agent_response(solver_trajectory)
                        results[f'solver_result_{i}'] = solver_result
                        self.solver.reset_context()

                    except Exception as e:
                        print(f"Task {i} failed: {e}")
                        results[f'solver_trajectory_{i}'] = None
                        results[f'solver_result_{i}'] = None

                ## 并行处理
                # with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                #     future_to_index = {}
                #     for i in range(self.agent_num):
                #         task_index = i
                #         future = ex.submit(self._run_solver_task,solver_task)
                #         future_to_index[future] = task_index
                    
                #     for future in as_completed(future_to_index):
                #         i = future_to_index[future]
                #         try:
                #             solver_trajectory = future.result()
                #             results[f'solver_trajectory_{i}'] = solver_trajectory
                #             solver_result = extract_agent_response(solver_trajectory)
                #             results[f'solver_result_{i}'] = solver_result
                #         except Exception as e:
                #             print(f"Task {i} failed: {e}")
                #             results[f'solver_trajectory_{i}'] = None
                #             results[f'solver_result_{i}'] = None

                self.logger.info("Solving completed")
            
            results['status'] = 'completed'
            self.logger.info("Solver-agent task execution completed")

            self.results.append(results)
        except Exception as e:
            self.logger.error(f"Solver-agent task execution failed: {e}", exc_info=True)
            results['status'] = 'failed'
            results['error'] = str(e)

            self.results.append(results)
        
        return results


    def _run_solver_task(self, solver_task: TaskInstance) -> str:
        """包装solver.run()以便在线程中执行

        Args:
            solver_task: 初始问题

        Return:
            agent的解决方案
        """
        return self.solver.run(solver_task)



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
                trajectory_key = f"solver_trajectory_{i}"
                result_key = f"solver_result_{i}"
                
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
                if "solver_result" in result:
                    task_record["solver_result"] = result.get("solver_result")
            else:
                task_record["trajectories"] = trajectories
                task_record["steps"] = total_steps


            if not results:
                task_record["results"] = {}
                
                # 添加任务的其他信息
                if "solver_result" in result:
                    task_record["solver_result"] = result.get("solver_result")
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
