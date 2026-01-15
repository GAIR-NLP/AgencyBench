from dataclasses import dataclass
from evaluations.base.data_classes import Config
import os

@dataclass
class TaskConfig(Config):
    """
    作业配置类
    """
    
    max_eval_time: float = 3000
    task_name: str = "task28"
    model_name: str = "azure/o3"
    batch_size: int = 10
    

    @classmethod
    def from_dict(cls, data: dict) -> "TaskConfig":
        return cls(**data)
    
    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self.__dataclass_fields__.keys()}
    
    def __str__(self):
        return f"TaskConfig(task_name={self.task_name}, task_output_dir={self.task_output_dir}, refernece_dir={self.refernece_dir}, max_eval_time={self.max_eval_time})"
    
    def __repr__(self):
        return self.__str__()