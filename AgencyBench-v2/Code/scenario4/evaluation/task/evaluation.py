from evaluations.base.base_eval import BaseBenchmark
from typing import Dict, Any, Optional
import json
import os
from evaluations.base.data_classes import Config
from .scripts.evaluation_utils import (
    validate_files, 
    calculate_task_score
)

class TaskBenchmark(BaseBenchmark):
    """Advanced Mathematical Reasoning"""

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.task_types = [
            'geometry', 'graph_connectivity', 'graph_maxflow', 'math_convexity'
        ]
        self.output_dir_path = os.path.join(self.config.workspace, "data/outputs")
        self.reference_dir_path = os.path.join(self.config.eval_workspace, self.config.task_name, "data/references/test")

    def run_1(self) -> Dict[str, Any]:
        """
        运行完整的评估流程
        """
        try:
            # 第一部分：文件验证 (10分)
            total_files, valid_files, file_score = validate_files(
                self.output_dir_path, self.reference_dir_path, self.task_types
            )
            
            # 第二部分：任务得分 (90分)
            total_tasks_all, total_successful_all, task_score, task_details = calculate_task_score(
                self.output_dir_path, self.reference_dir_path, self.task_types
            )
            

            baseline_score = 0.0
            advanced_score = 0.0

            for task_type in self.task_types:
                baseline_score += task_details.get(f"{task_type}_total_tasks", 0) * self.config.get_basic_target(task_type)
                advanced_score += task_details.get(f"{task_type}_total_tasks", 0) * self.config.get_advanced_target(task_type)

            baseline_score = max(min(baseline_score / total_tasks_all, 100), 0)
            advanced_score = max(min(advanced_score / total_tasks_all, 100), 0)
            inference_score = total_successful_all / total_tasks_all * 100

            if inference_score < baseline_score:
                task_score = 0.0
            elif inference_score < advanced_score:
                task_score = 70.0 * (inference_score - baseline_score) / (advanced_score - baseline_score)
            else:
                task_score = 70.0 + 20.0 * (inference_score - advanced_score) / (100.0 - advanced_score)

            # total_score = score
            # 计算总分
            total_score = file_score + task_score
            
            # 计算总体成功率
            overall_success_rate = 0.0
            if total_tasks_all > 0:
                overall_success_rate = (total_successful_all / total_tasks_all) * 100
            
            
            return {
                "error": None,
                "score": total_score,
                # 文件验证信息
                "file_validation_score": file_score,
                "total_files": total_files,
                "valid_files": valid_files,
                # 各任务类型详细信息
                "acc_score": task_score,
                "overall_success_rate": overall_success_rate,
                "geometry_success_rate": task_details.get("geometry_success_rate", 0.0),
                "graph_connectivity_success_rate": task_details.get("graph_connectivity_success_rate", 0.0),
                "graph_maxflow_success_rate": task_details.get("graph_maxflow_success_rate", 0.0),
                "math_convexity_success_rate": task_details.get("math_convexity_success_rate", 0.0),
                "geometry_total_tasks": task_details.get("geometry_total_tasks", 0),
                "graph_connectivity_total_tasks": task_details.get("graph_connectivity_total_tasks", 0),
                "graph_maxflow_total_tasks": task_details.get("graph_maxflow_total_tasks", 0),
                "math_convexity_total_tasks": task_details.get("math_convexity_total_tasks", 0),
                "geometry_successful_tasks": task_details.get("geometry_successful_tasks", 0),
                "graph_connectivity_successful_tasks": task_details.get("graph_connectivity_successful_tasks", 0),
                "graph_maxflow_successful_tasks": task_details.get("graph_maxflow_successful_tasks", 0),
                "math_convexity_successful_tasks": task_details.get("math_convexity_successful_tasks", 0),
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "message": f"Error: {str(e)}",
                "score": 0
            } 