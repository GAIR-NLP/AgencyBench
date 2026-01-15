#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main evaluation script for AI model responses using rubrics (Parallel Version)
Enhanced with concurrent processing for improved performance
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from code.utils import (
    load_rubrics, load_model_responses, save_json_file, 
    create_output_directory, get_model_name_from_file,
    validate_rubric_format, validate_response_format
)
from evaluator import RubricEvaluator


class ParallelEvaluationCoordinator:
    """协调并行评估任务的类"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化并行评估协调器
        
        Parameters:
        max_workers (int): 最大并行工作线程数
        """
        self.max_workers = max_workers
        self.print_lock = Lock()  # For synchronized print output
        
    def safe_print(self, message: str):
        """线程安全的打印函数"""
        with self.print_lock:
            print(message)
    
    def evaluate_single_question(self, args: Tuple) -> Dict:
        """
        评估单个问题的函数，用于并行处理
        
        Parameters:
        args (Tuple): 包含评估所需参数的元组
        
        Returns:
        Dict: 评估结果或错误信息
        """
        (question_id, response_data, rubric_data, model_name, judge_model, thread_id) = args
        
        try:
            self.safe_print(f"[Thread-{thread_id}] Starting evaluation for question {question_id}...")
            
            # 为每个线程创建独立的评估器实例
            evaluator = RubricEvaluator(judge_model=judge_model)
            
            # 提取数据
            question = response_data["question"]
            ai_response = response_data["response"]
            reference_rubrics = rubric_data["rubric"]
            
            # 评估响应
            result = evaluator.evaluate_response(
                question_id=question_id,
                model_name=model_name,
                question=question,
                ai_response=ai_response,
                reference_rubrics=reference_rubrics
            )
            
            self.safe_print(f"[Thread-{thread_id}] Question {question_id} evaluation completed")
            return result
            
        except Exception as e:
            error_result = {
                "question_id": question_id,
                "error": str(e),
                "status": "failed"
            }
            self.safe_print(f"[Thread-{thread_id}] Question {question_id} evaluation failed: {e}")
            return error_result


def evaluate_single_model_parallel(model_file: str, rubrics_dict: Dict, result_dir: str, 
                                 judge_model: str = "o3-mini", max_workers: int = 4) -> str:
    """
    并行评估单个模型的响应
    
    Parameters:
    model_file (str): 模型响应文件路径
    rubrics_dict (Dict): 按问题ID组织的标准字典
    result_dir (str): 结果输出目录
    judge_model (str): 用于评估的模型
    max_workers (int): 最大并行工作线程数
    
    Returns:
    str: 输出文件路径
    """
    print(f"\n{'='*60}")
    print(f"Parallel evaluation for model: {model_file}")
    print(f"Maximum parallel workers: {max_workers}")
    print(f"{'='*60}")
    
    # 加载模型响应
    model_responses = load_model_responses(model_file)
    if not model_responses:
        print(f"Unable to load responses from {model_file}")
        return None
    
    # 从文件名提取模型名称
    model_name = get_model_name_from_file(model_file)
    
    # 初始化并行评估协调器
    coordinator = ParallelEvaluationCoordinator(max_workers=max_workers)
    
    # 准备评估任务
    evaluation_tasks = []
    thread_counter = 0
    
    for question_id, response_data in model_responses.items():
        # 验证响应格式
        if not validate_response_format(response_data):
            print(f"Skipping question {question_id}, invalid format")
            continue
        
        # 检查是否存在对应的标准
        if question_id not in rubrics_dict:
            print(f"Question {question_id} has no corresponding rubric, skipping...")
            continue
        
        rubric_data = rubrics_dict[question_id]
        
        # 验证标准格式
        if not validate_rubric_format(rubric_data):
            print(f"Skipping question {question_id}, invalid rubric format")
            continue
        
        # 准备任务参数
        task_args = (question_id, response_data, rubric_data, model_name, judge_model, thread_counter)
        evaluation_tasks.append(task_args)
        thread_counter += 1
    
    if not evaluation_tasks:
        print("No valid evaluation tasks")
        return None
    
    print(f"Preparing parallel evaluation for {len(evaluation_tasks)} questions...")
    
    # 执行并行评估
    all_results = []
    failed_tasks = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(coordinator.evaluate_single_question, task_args): task_args[0] 
            for task_args in evaluation_tasks
        }
        
        # 收集结果
        completed_count = 0
        for future in as_completed(future_to_task):
            question_id = future_to_task[future]
            try:
                result = future.result()
                if "error" in result:
                    failed_tasks.append(result)
                else:
                    all_results.append(result)
                
                completed_count += 1
                print(f"Progress: {completed_count}/{len(evaluation_tasks)} completed")
                
            except Exception as e:
                error_result = {
                    "question_id": question_id,
                    "error": str(e),
                    "status": "exception"
                }
                failed_tasks.append(error_result)
                print(f"Task exception - Question {question_id}: {e}")
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # 打印执行统计
    print(f"\n{'='*50}")
    print("Parallel Evaluation Statistics")
    print(f"{'='*50}")
    print(f"Total tasks: {len(evaluation_tasks)}")
    print(f"Successfully completed: {len(all_results)}")
    print(f"Failed tasks: {len(failed_tasks)}")
    print(f"Total time: {evaluation_time:.2f} seconds")
    print(f"Average time per question: {evaluation_time/len(evaluation_tasks):.2f} seconds")
    print(f"{'='*50}")
    
    # 如果有失败的任务，打印详情
    if failed_tasks:
        print("\nFailed task details:")
        for failed_task in failed_tasks:
            print(f"- Question {failed_task['question_id']}: {failed_task['error']}")
    
    # 创建模型特定的输出目录结构
    model_result_dir = os.path.join(result_dir,"rubric_eval", model_name)
    if not create_output_directory(model_result_dir):
        print(f"Unable to create model output directory: {model_result_dir}")
        return None
    
    # 保存结果
    output_file = os.path.join(model_result_dir, f"{model_name}_evaluation_results.json")
    if save_json_file(all_results, output_file):
        print(f"\nEvaluation results saved to: {model_result_dir}")
        
        # 打印汇总统计
        if all_results:
            print(f"\n{'='*50}")
            print("Evaluation Results Summary")
            print(f"{'='*50}")
            
            # 计算平均分数
            total_recall = sum(r["result"]["recall"] for r in all_results)
            count = len(all_results)
            
            avg_recall = total_recall / count
            
            print(f"Model: {model_name}")
            print(f"Successfully evaluated questions: {count}")
            print(f"Average coverage: {avg_recall:.4f}")
            print(f"{'='*50}")
            # 保存到txt文件
            with open(os.path.join(model_result_dir, f"{model_name}_evaluation_results.txt"), "w") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Average coverage: {avg_recall:.4f}\n")
                f.write(f"{'='*50}\n")
        
        return model_result_dir
    else:
        print(f"Failed to save results for {model_name}")
        return None


def main():
    """主函数，支持命令行参数解析"""
    parser = argparse.ArgumentParser(description='并行评估AI模型响应 (增强版标准评估)')
    
    parser.add_argument('--model_file', type=str, required=True,
                        help='模型响应文件路径')
    parser.add_argument('--rubrics_file', type=str, default='data/eval_data/rubric.json',
                        help='标准JSON文件路径')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='评估结果输出目录')
    parser.add_argument('--judge_model', type=str, default='o3-mini',
                        help='用于评估的模型')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='最大并行工作线程数 (默认: 4)')
    
    args = parser.parse_args()
    
    # 加载标准
    print("Loading evaluation rubrics...")
    rubrics_dict = load_rubrics(args.rubrics_file)
    if not rubrics_dict:
        print("Failed to load rubrics, exiting program.")
        return
    
    # 创建输出目录
    if not create_output_directory(args.result_dir):
        print("Failed to create output directory, exiting program.")
        return
    
    # 并行评估单个模型
    result_file = evaluate_single_model_parallel(
        model_file=args.model_file,
        rubrics_dict=rubrics_dict,
        result_dir=args.result_dir,
        judge_model=args.judge_model,
        max_workers=args.max_workers
    )
    
    # 打印最终汇总
    print(f"\n{'='*60}")
    print("Final Evaluation Summary")
    print(f"{'='*60}")
    
    if result_file:
        print(f"Evaluation completed successfully!")
        print(f"Results saved to: {result_file}")
        print(f"Using parallel workers: {args.max_workers}")
    else:
        print(f"Evaluation failed!")
        print(f"Please check the error logs above for detailed information.")


if __name__ == "__main__":
    main()