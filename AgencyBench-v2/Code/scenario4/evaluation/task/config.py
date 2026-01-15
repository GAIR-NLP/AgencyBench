from dataclasses import dataclass
from evaluations.base.data_classes import Config
import os

@dataclass
class TaskConfig(Config):
    """
    Advanced Mathematical Reasoning Task Configuration
    """
    
    task_name: str = "task27"
    max_eval_time: float = 6000.0  # 10分钟
    
    # Basic Performance Targets
    geometry_basic_target: float = 62.5
    graph_maxflow_basic_target: float = 25.0
    graph_connectivity_basic_target: float = 96.1
    function_convexity_basic_target: float = 87.2
    
    # Advanced Performance Targets
    geometry_advanced_target: float = 66.7
    graph_maxflow_advanced_target: float = 66.3
    graph_connectivity_advanced_target: float = 98.4
    function_convexity_advanced_target: float = 94.9
    

    
    @classmethod
    def from_dict(cls, data: dict) -> "TaskConfig":
        return cls(**data)
    
    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.__dataclass_fields__.keys()}
    
    def __str__(self):
        return f"TaskConfig(task_name={self.task_name}, task_output_dir={self.task_output_dir}, reference_dir={self.reference_dir}, max_eval_time={self.max_eval_time})"
    
    def __repr__(self):
        return self.__str__()
    
    def get_basic_target(self, task_type: str) -> float:
        """Get basic performance target for a specific task type"""
        targets = {
            'geometry': self.geometry_basic_target,
            'graph_maxflow': self.graph_maxflow_basic_target,
            'graph_connectivity': self.graph_connectivity_basic_target,
            'math_convexity': self.function_convexity_basic_target
        }
        return targets.get(task_type, 0.0)
    
    def get_advanced_target(self, task_type: str) -> float:
        """Get advanced performance target for a specific task type"""
        targets = {
            'geometry': self.geometry_advanced_target,
            'graph_maxflow': self.graph_maxflow_advanced_target,
            'graph_connectivity': self.graph_connectivity_advanced_target,
            'math_convexity': self.function_convexity_advanced_target
        }
        return targets.get(task_type, 0.0)
    
    def get_all_basic_targets(self) -> dict:
        """Get all basic performance targets"""
        return {
            'geometry': self.geometry_basic_target,
            'graph_maxflow': self.graph_maxflow_basic_target,
            'graph_connectivity': self.graph_connectivity_basic_target,
            'math_convexity': self.function_convexity_basic_target,
        }
    
    def get_all_advanced_targets(self) -> dict:
        """Get all advanced performance targets"""
        return {
            'geometry': self.geometry_advanced_target,
            'graph_maxflow': self.graph_maxflow_advanced_target,
            'graph_connectivity': self.graph_connectivity_advanced_target,
            'math_convexity': self.function_convexity_advanced_target,
        } 