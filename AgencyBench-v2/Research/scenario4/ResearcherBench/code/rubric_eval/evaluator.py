#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core evaluation logic for AI model responses using rubrics (Parallel Version)
Enhanced with thread-safe operations and optimized for concurrent processing
"""

import os
import re
import json
import time
import threading
from typing import List, Dict, Any, Tuple
from openai import OpenAI

# Add project root to path for imports
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from code.prompt import get_single_rubric_evaluation_prompt
from code.utils import get_api_config


class ThreadSafeRubricEvaluator:
    """线程安全的标准评估器，支持并行处理"""
    
    # 类级别的锁，用于同步打印和统计
    _print_lock = threading.Lock()
    _stats_lock = threading.Lock()
    _global_stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0
    }
    
    def __init__(self, judge_model: str = "o3-mini", max_retries: int = 5):
        """
        初始化评估器
        
        Parameters:
        judge_model (str): 用于评估的模型
        max_retries (int): 最大重试次数
        """
        self.judge_model = judge_model
        self.max_retries = max_retries
        self.thread_id = threading.current_thread().ident
        self.client = self._setup_client()
        
        # 线程级别的统计
        self.thread_stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0
        }
    
    def _setup_client(self):
        """设置OpenAI客户端（每个线程独立）"""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        if not api_key:
            raise ValueError("需要设置 OPENAI_API_KEY 环境变量")
        
        # 如果提供了base_url，使用它；否则使用默认的OpenAI端点
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        else:
            return OpenAI(api_key=api_key)
    
    def _safe_print(self, message: str):
        """线程安全的打印函数"""
        with self._print_lock:
            print(f"[Thread-{self.thread_id}] {message}")
    
    def _update_global_stats(self, success: bool):
        """更新全局统计信息"""
        with self._stats_lock:
            self._global_stats["total_requests"] += 1
            if success:
                self._global_stats["successful_requests"] += 1
            else:
                self._global_stats["failed_requests"] += 1
    
    def _make_api_request(self, messages: List[Dict], operation_name: str) -> str:
        """
        统一的API请求处理函数，包含重试逻辑和统计
        
        Parameters:
        messages (List[Dict]): 消息列表
        operation_name (str): 操作名称，用于日志
        
        Returns:
        str: API响应内容
        """
        for retry_count in range(self.max_retries):
            try:
                self.thread_stats["requests"] += 1
                
                # 发送API请求
                completion = self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=messages
                )
                
                result = completion.choices[0].message.content.strip()
                self.thread_stats["successes"] += 1
                self._update_global_stats(True)
                
                return result
                
            except Exception as e:
                self.thread_stats["failures"] += 1
                error_msg = f"{operation_name} 失败 (尝试 {retry_count + 1}/{self.max_retries}): {e}"
                
                if retry_count == self.max_retries - 1:
                    self._safe_print(f"Reached maximum retry limit, {operation_name} failed")
                    self._update_global_stats(False)
                    raise e
                else:
                    self._safe_print(error_msg + " - Retrying...")
                    time.sleep(min(2 ** retry_count, 10))  # 指数退避，最大10秒
    
    def evaluate_single_rubric(self, question: str, rubric: Dict[str, Any], 
                              ai_response: str) -> Tuple[bool, str]:
        """
        评估单个标准项是否在AI回答中得到覆盖
        
        Parameters:
        question (str): 用户提出的研究问题
        rubric (dict): 单个标准项信息
        ai_response (str): 完整的AI回答文本
        
        Returns:
        tuple: (是否覆盖, 理由说明)
        """
        rubric_content = rubric['point']
        rubric_weight = rubric.get('weight', 1)
        
        # 获取评估提示词
        coverage_prompt = get_single_rubric_evaluation_prompt(
            question, rubric_content, rubric_weight, ai_response
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that helps determine if a rubric is covered in an AI response."},
            {"role": "user", "content": coverage_prompt}
        ]
        
        try:
            result = self._make_api_request(messages, "标准项覆盖评估")
            
            # 解析是否/否答案和理由说明
            if result.lower().startswith("yes:") or result.lower().startswith("yes "):
                is_covered = True
                justification = result[4:].strip() if len(result) > 3 and result[3] == ":" else result[4:].strip()
                return is_covered, justification
            elif result.lower().startswith("no:") or result.lower().startswith("no "):
                is_covered = False
                justification = result[3:].strip() if len(result) > 2 and result[2] == ":" else result[3:].strip()
                return is_covered, justification
            else:
                # 如果不以yes/no开头，检查是否包含yes/no
                if "yes" in result.lower():
                    return True, result
                elif "no" in result.lower():
                    return False, result
                else:
                    self._safe_print("Evaluation result does not contain clear yes/no answer, defaulting to not covered")
                    return False, "Unable to get clear answer, defaulting to not covered"
                        
        except Exception as e:
            error_msg = f"Error evaluating rubric coverage: {str(e)}"
            self._safe_print(error_msg)
            return False, error_msg
    
    def evaluate_coverage(self, question: str, reference_rubrics: List[Dict[str, Any]], 
                         ai_response: str) -> Tuple[list, int, int]:
        """
        评估AI回答对参考标准项的覆盖情况
        
        Parameters:
        question (str): 用户提出的研究问题
        reference_rubrics (list): 参考标准项列表
        ai_response (str): 完整的AI回答文本
        
        Returns:
        tuple: (覆盖结果, 加权覆盖数, 总权重)
        """
        coverage_results = []
        m_weighted = 0
        total_weight = 0
        
        self._safe_print(f"Starting rubric evaluation, total {len(reference_rubrics)} rubric items...")
        
        # 逐个评估每个标准项
        for i, rubric in enumerate(reference_rubrics):
            weight = rubric.get("weight", 1)
            total_weight += weight
            
            # 评估标准项是否被覆盖
            is_covered, justification = self.evaluate_single_rubric(
                question, rubric, ai_response
            )
            
            # 添加评估结果
            coverage_results.append({
                "point": rubric["point"],
                "weight": weight,
                "covered": is_covered,
                "justification": justification
            })
            
            # 如果被覆盖，加入加权总和
            if is_covered:
                m_weighted += weight
                self._safe_print(f"Rubric item {i+1}/{len(reference_rubrics)} covered (weight: {weight})")
            else:
                self._safe_print(f"Rubric item {i+1}/{len(reference_rubrics)} not covered (weight: {weight})")
        
        self._safe_print(f"Coverage evaluation completed. Covered weight: {m_weighted}, Total weight: {total_weight}")
        
        return coverage_results, m_weighted, total_weight
    
    def evaluate_response(self, question_id: int, model_name: str, question: str, 
                         ai_response: str, reference_rubrics: List[Dict[str, Any]]) -> Dict:
        """
        评估单个模型回答并返回结果
        
        Parameters:
        question_id (int): 问题ID
        model_name (str): 模型名称
        question (str): 研究问题
        ai_response (str): 要评估的AI生成回答文本
        reference_rubrics (list): 参考标准项列表
        
        Returns:
        dict: 包含评估结果的JSON对象
        """
        self._safe_print(f"Starting evaluation for question {question_id}, using evaluator: {self.judge_model}")
        
        # 评估标准覆盖情况
        coverage_results, m_weighted, total_weight = self.evaluate_coverage(
            question, reference_rubrics, ai_response
        )
        
        # 计算指标
        recall = m_weighted / total_weight if total_weight > 0 else 0
        
        self._safe_print(f"Question {question_id} evaluation completed - Coverage: {recall:.4f}")
        
        # 创建结果对象
        result_json = {
            "id": question_id,
            "model": model_name,
            "metric": "Evaluation",
            "evaluator": self.judge_model,
            "result": {
                "m_weighted": m_weighted,
                "total_weight": total_weight,
                "recall": round(recall, 4),
                "coverage_results": coverage_results
            },
            "thread_stats": self.thread_stats.copy()  # 包含线程级统计信息
        }
        
        return result_json
    
    @classmethod
    def get_global_stats(cls) -> Dict:
        """获取全局统计信息"""
        with cls._stats_lock:
            return cls._global_stats.copy()
    
    @classmethod
    def reset_global_stats(cls):
        """重置全局统计信息"""
        with cls._stats_lock:
            cls._global_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0
            }


# 为了向后兼容，保留原有类名
RubricEvaluator = ThreadSafeRubricEvaluator
