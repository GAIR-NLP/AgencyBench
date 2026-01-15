#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量评估指定题目和模型的脚本
根据题号-模型映射关系进行精准评估
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to path for imports
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from code.utils import (
    load_rubrics, load_model_responses, save_json_file, 
    create_output_directory
)
from code.rubric_eval.evaluator import RubricEvaluator


class TargetedEvaluationCoordinator:
    """针对特定题目的精准评估协调器"""
    
    def __init__(self, max_workers: int = 4):
        """
        初始化评估协调器
        
        Parameters:
        max_workers (int): 最大并行工作线程数
        """
        self.max_workers = max_workers
        self.print_lock = Lock()
        
        # 题号-模型映射关系
        self.question_model_mapping = {
            1: "OpenAI",
            3: "Google", 
            5: "Grok3",
            12: "Grok3deeper",
            22: "Perplexity",
            24: "Zhihu",
            34: "Yiyan",
            45: "OpenAI",
            50: "Google",
            55: "Grok3"
        }
        
        # 模型文件映射
        self.model_files = {
            "OpenAI": "data/user_data/OpenAI.json",
            "Google": "data/user_data/Google.json",
            "Grok3": "data/user_data/Grok3.json",
            "Grok3deeper": "data/user_data/Grok3deeper.json",
            "Perplexity": "data/user_data/Perplexity.json",
            "Zhihu": "data/user_data/Zhihu.json",
            "Yiyan": "data/user_data/Yiyan.json"
        }
        
    def safe_print(self, message: str):
        """线程安全的打印函数"""
        with self.print_lock:
            print(message)
    
    def load_all_model_responses(self) -> Dict[str, Dict]:
        """
        加载所有模型的响应数据
        
        Returns:
        Dict[str, Dict]: 模型名称到响应数据的映射
        """
        all_responses = {}
        
        for model_name, file_path in self.model_files.items():
            if os.path.exists(file_path):
                responses = load_model_responses(file_path)
                if responses:
                    all_responses[model_name] = responses
                    self.safe_print(f"成功加载 {model_name} 的响应数据: {len(responses)} 个问题")
                else:
                    self.safe_print(f"警告: 无法加载 {model_name} 的响应数据")
            else:
                self.safe_print(f"警告: 文件不存在 - {file_path}")
        
        return all_responses
    
    def prepare_evaluation_tasks(self, all_responses: Dict[str, Dict], 
                               rubrics_dict: Dict) -> List[Tuple]:
        """
        准备评估任务列表
        
        Parameters:
        all_responses (Dict): 所有模型的响应数据
        rubrics_dict (Dict): 评分标准数据
        
        Returns:
        List[Tuple]: 评估任务列表
        """
        evaluation_tasks = []
        
        for question_id, model_name in self.question_model_mapping.items():
            # 检查模型响应是否存在
            if model_name not in all_responses:
                self.safe_print(f"跳过 Q{question_id}: 模型 {model_name} 的响应数据不存在")
                continue
            
            # 检查具体问题的响应是否存在
            model_responses = all_responses[model_name]
            if question_id not in model_responses:
                self.safe_print(f"跳过 Q{question_id}: 模型 {model_name} 没有该问题的响应")
                continue
            
            # 检查评分标准是否存在
            if question_id not in rubrics_dict:
                self.safe_print(f"跳过 Q{question_id}: 没有对应的评分标准")
                continue
            
            # 准备任务参数
            response_data = model_responses[question_id]
            rubric_data = rubrics_dict[question_id]
            
            task_args = (question_id, model_name, response_data, rubric_data)
            evaluation_tasks.append(task_args)
            
            self.safe_print(f"准备评估任务: Q{question_id} - {model_name}")
        
        return evaluation_tasks
    
    def evaluate_single_task(self, args: Tuple, eval_model: str) -> Dict:
        """
        评估单个任务
        
        Parameters:
        args (Tuple): 任务参数
        eval_model (str): 评估模型
        
        Returns:
        Dict: 评估结果
        """
        question_id, model_name, response_data, rubric_data = args
        
        try:
            # 创建评估器实例
            evaluator = RubricEvaluator(eval_model=eval_model)
            
            # 提取数据
            question = response_data["question"]
            ai_response = response_data["response"]
            reference_rubrics = rubric_data["rubric"]
            
            # 执行评估
            result = evaluator.evaluate_response(
                question_id=question_id,
                model_name=model_name,
                question=question,
                ai_response=ai_response,
                reference_rubrics=reference_rubrics
            )
            
            self.safe_print(f"完成评估: Q{question_id} - {model_name}")
            return result
            
        except Exception as e:
            error_result = {
                "question_id": question_id,
                "model_name": model_name,
                "error": str(e),
                "status": "failed"
            }
            self.safe_print(f"评估失败: Q{question_id} - {model_name}: {e}")
            return error_result
    
    def run_targeted_evaluation(self, rubrics_file: str, output_dir: str, 
                               eval_model: str = "o3-mini") -> str:
        """
        运行针对性评估
        
        Parameters:
        rubrics_file (str): 评分标准文件路径
        output_dir (str): 输出目录
        eval_model (str): 评估模型
        
        Returns:
        str: 输出文件路径
        """
        print(f"\n{'='*60}")
        print("开始批量精准评估")
        print(f"评估模型: {eval_model}")
        print(f"最大并行线程数: {self.max_workers}")
        print(f"目标题目数量: {len(self.question_model_mapping)}")
        print(f"{'='*60}")
        
        # 加载评分标准
        self.safe_print("加载评分标准...")
        rubrics_dict = load_rubrics(rubrics_file)
        if not rubrics_dict:
            self.safe_print("加载评分标准失败")
            return None
        
        # 加载所有模型响应
        self.safe_print("加载模型响应数据...")
        all_responses = self.load_all_model_responses()
        if not all_responses:
            self.safe_print("没有加载到任何模型响应数据")
            return None
        
        # 准备评估任务
        self.safe_print("准备评估任务...")
        evaluation_tasks = self.prepare_evaluation_tasks(all_responses, rubrics_dict)
        
        if not evaluation_tasks:
            self.safe_print("没有有效的评估任务")
            return None
        
        self.safe_print(f"准备评估 {len(evaluation_tasks)} 个任务...")
        
        # 执行并行评估
        all_results = []
        failed_tasks = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self.evaluate_single_task, task_args, eval_model): task_args 
                for task_args in evaluation_tasks
            }
            
            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_task):
                task_args = future_to_task[future]
                try:
                    result = future.result()
                    if "error" in result:
                        failed_tasks.append(result)
                    else:
                        all_results.append(result)
                    
                    completed_count += 1
                    question_id, model_name = task_args[0], task_args[1]
                    self.safe_print(f"进度: {completed_count}/{len(evaluation_tasks)} - Q{question_id}({model_name})")
                    
                except Exception as e:
                    question_id, model_name = task_args[0], task_args[1]
                    error_result = {
                        "question_id": question_id,
                        "model_name": model_name,
                        "error": str(e),
                        "status": "exception"
                    }
                    failed_tasks.append(error_result)
                    self.safe_print(f"任务异常 - Q{question_id}({model_name}): {e}")
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        # 打印统计信息
        self.print_evaluation_summary(evaluation_tasks, all_results, failed_tasks, evaluation_time)
        
        # 保存结果
        return self.save_results(all_results, output_dir, eval_model)
    
    def print_evaluation_summary(self, tasks: List, results: List, 
                               failed: List, duration: float):
        """打印评估汇总信息"""
        print(f"\n{'='*50}")
        print("评估统计汇总")
        print(f"{'='*50}")
        print(f"总任务数: {len(tasks)}")
        print(f"成功完成: {len(results)}")
        print(f"失败任务: {len(failed)}")
        print(f"成功率: {len(results)/len(tasks)*100:.1f}%")
        print(f"总耗时: {duration:.2f} 秒")
        print(f"平均每题耗时: {duration/len(tasks):.2f} 秒")
        
        # 按模型统计
        model_stats = {}
        for result in results:
            model = result.get("model", "Unknown")
            if model not in model_stats:
                model_stats[model] = {"count": 0, "recall": []}
            
            model_stats[model]["count"] += 1
            model_stats[model]["recall"].append(result["result"]["recall"])
        
        print(f"\n按模型统计:")
        for model, stats in model_stats.items():
            if stats["count"] > 0:
                avg_recall = sum(stats["recall"]) / stats["count"]
                print(f"  {model}: {stats['count']}题 | 覆盖率:{avg_recall:.3f}")
        
        # 失败任务详情
        if failed:
            print(f"\n失败任务详情:")
            for failed_task in failed:
                qid = failed_task.get("question_id", "Unknown")
                model = failed_task.get("model_name", "Unknown")
                error = failed_task.get("error", "Unknown error")
                print(f"  Q{qid}({model}): {error}")
        
        print(f"{'='*50}")
    
    def save_results(self, results: List, output_dir: str, eval_model: str) -> str:
        """保存评估结果到指定的统一文件"""
        if not results:
            self.safe_print("没有结果需要保存")
            return None
        
        # 创建指定的输出目录结构: results_for_specific/targeted_evaluation/{eval_model}/
        targeted_output_dir = os.path.join(output_dir, "results_for_specific", "targeted_evaluation", eval_model.replace("/", "_"))
        if not create_output_directory(targeted_output_dir):
            self.safe_print(f"无法创建输出目录: {targeted_output_dir}")
            return None
        
        # 统一保存为 meta_eval.json
        output_file = os.path.join(targeted_output_dir, "meta_eval.json")
        
        # 计算汇总统计
        total_questions = len(results)
        avg_recall = sum(r["result"]["recall"] for r in results) / total_questions if total_questions > 0 else 0
        
        # 按模型分组统计
        model_stats = {}
        for result in results:
            model = result.get("model", "Unknown")
            if model not in model_stats:
                model_stats[model] = {
                    "question_count": 0,
                    "questions": [],
                    "avg_recall": 0,
                    "recalls": []
                }
            
            model_stats[model]["question_count"] += 1
            model_stats[model]["questions"].append(f"Q{result['id']}")
            model_stats[model]["recalls"].append(result["result"]["recall"])
        
        # 计算每个模型的平均值
        for model, stats in model_stats.items():
            if stats["question_count"] > 0:
                stats["avg_recall"] = sum(stats["recalls"]) / stats["question_count"]
                # 移除临时的列表数据，保持结果文件简洁
                del stats["recalls"]
        
        # 构建最终结果结构
        final_results = {
            "metadata": {
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "eval_model": eval_model,
                "total_questions": total_questions,
                "question_model_mapping": self.question_model_mapping,
                "summary_statistics": {
                    "overall_avg_recall": round(avg_recall, 4)
                },
                "model_statistics": model_stats
            },
            "detailed_results": results
        }
        
        if save_json_file(final_results, output_file):
            self.safe_print(f"\n✅ 所有结果已统一保存到: {output_file}")
            self.safe_print(f"📁 目录结构: {targeted_output_dir}")
            return output_file
        else:
            self.safe_print("❌ 保存结果失败")
            return None
    



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量评估指定题目和模型')
    
    parser.add_argument('--rubrics_file', type=str, default='data/eval_data/rubric.json',
                        help='评分标准JSON文件路径')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='评估结果输出目录')
    parser.add_argument('--eval_model', type=str, default='o3-mini',
                        help='用于评估的模型')
    parser.add_argument('--max_workers', type=int, default=4,
                        help='最大并行工作线程数 (默认: 4)')
    
    args = parser.parse_args()
    
    # 检查评分标准文件是否存在
    if not os.path.exists(args.rubrics_file):
        print(f"错误: 评分标准文件不存在 - {args.rubrics_file}")
        return
    
    # 创建输出目录
    if not create_output_directory(args.output_dir):
        print("创建输出目录失败，退出程序。")
        return
    
    # 创建评估协调器并运行评估
    coordinator = TargetedEvaluationCoordinator(max_workers=args.max_workers)
    
    result_file = coordinator.run_targeted_evaluation(
        rubrics_file=args.rubrics_file,
        output_dir=args.output_dir,
        eval_model=args.eval_model
    )
    
    # 最终汇总
    print(f"\n{'='*60}")
    print("批量精准评估完成")
    print(f"{'='*60}")
    
    if result_file:
        print(f"✅ 评估成功完成!")
        print(f"📁 结果文件: {result_file}")
        print(f"📊 统一保存: meta_eval.json")
        print(f"🧵 使用并行线程数: {args.max_workers}")
        print(f"🤖 评估模型: {args.eval_model}")
        print(f"📈 文件包含详细结果和汇总统计信息")
    else:
        print(f"❌ 评估失败!")
        print(f"请检查上述错误日志获取详细信息。")


if __name__ == "__main__":
    main()