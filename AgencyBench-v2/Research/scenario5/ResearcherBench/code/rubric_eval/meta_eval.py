#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估结果与人类标注结果对比分析脚本
计算F1分数和准确度（带权重和不带权重）
"""

import os
import sys
import json
import glob
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import statistics

# Add project root to path for imports
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from code.utils import load_json_file, save_json_file, create_output_directory


class ModelHumanComparison:
    """模型与人类标注结果对比分析器"""
    
    def __init__(self):
        self.human_annotations = {}  # 人类标注数据
        self.model_results = {}      # 模型评估数据
        self.rubric_weights = {}     # rubric权重数据
        self.comparison_results = {} # 对比结果
        
    def load_human_annotations(self, human_data_dir: str) -> bool:
        """
        加载人类标注数据
        
        Parameters:
        human_data_dir (str): 人类标注数据目录
        
        Returns:
        bool: 是否成功加载
        """
        print(f"正在加载人类标注数据从: {human_data_dir}")
        
        # 查找所有的人类标注文件 (Q*_result_*.json)
        pattern = os.path.join(human_data_dir, "Q*_result_*.json")
        annotation_files = glob.glob(pattern)
        
        if not annotation_files:
            print(f"错误: 在 {human_data_dir} 中未找到人类标注文件")
            return False
        
        for file_path in annotation_files:
            filename = os.path.basename(file_path)
            
            # 解析文件名获取题号和标注者
            # 格式: Q{id}_result_{annotator}.json
            try:
                parts = filename.replace('.json', '').split('_')
                question_id = int(parts[0][1:])  # Q1 -> 1
                annotator = parts[2]             # A, B, C, etc.
                
                # 加载标注数据
                annotation_data = load_json_file(file_path)
                if annotation_data:
                    if question_id not in self.human_annotations:
                        self.human_annotations[question_id] = {}
                    
                    self.human_annotations[question_id][annotator] = annotation_data
                    print(f"  ✅ 加载: Q{question_id} - 标注者{annotator}")
                else:
                    print(f"  ❌ 加载失败: {filename}")
                    
            except (IndexError, ValueError) as e:
                print(f"  ⚠️  跳过无效文件名: {filename} ({e})")
                continue
        
        total_questions = len(self.human_annotations)
        total_annotations = sum(len(annotators) for annotators in self.human_annotations.values())
        print(f"成功加载 {total_annotations} 个标注，涉及 {total_questions} 个题目")
        
        return total_annotations > 0
    
    def load_model_results(self, model_results_file: str) -> bool:
        """
        加载模型评估结果
        
        Parameters:
        model_results_file (str): 模型结果文件路径
        
        Returns:
        bool: 是否成功加载
        """
        print(f"正在加载模型评估结果从: {model_results_file}")
        
        model_data = load_json_file(model_results_file)
        if not model_data:
            print("错误: 无法加载模型评估结果")
            return False
        
        # 解析模型结果
        if "detailed_results" in model_data:
            results = model_data["detailed_results"]
        else:
            results = model_data.get("results", [])
        
        for result in results:
            question_id = result.get("id")
            if question_id:
                self.model_results[question_id] = result
                print(f"  ✅ 加载模型结果: Q{question_id}")
        
        print(f"成功加载 {len(self.model_results)} 个模型评估结果")
        return len(self.model_results) > 0
    
    def load_rubric_weights(self, rubric_file: str) -> bool:
        """
        加载rubric权重数据
        
        Parameters:
        rubric_file (str): rubric文件路径
        
        Returns:
        bool: 是否成功加载
        """
        print(f"正在加载rubric权重数据从: {rubric_file}")
        
        rubric_data = load_json_file(rubric_file)
        if not rubric_data:
            print("错误: 无法加载rubric权重数据")
            return False
        
        for item in rubric_data:
            question_id = item.get("id")
            if question_id and "rubric" in item:
                self.rubric_weights[question_id] = {}
                for rubric_item in item["rubric"]:
                    point = rubric_item.get("point", "")
                    weight = rubric_item.get("weight", 1)
                    # 使用point的前50个字符作为key来匹配
                    key = point[:50] if point else ""
                    self.rubric_weights[question_id][key] = weight
                
                print(f"  ✅ 加载权重: Q{question_id} ({len(item['rubric'])} 个rubric项)")
        
        print(f"成功加载 {len(self.rubric_weights)} 个题目的权重数据")
        return len(self.rubric_weights) > 0
    
    def find_rubric_weight(self, question_id: int, point_text: str) -> int:
        """
        查找rubric项的权重
        
        Parameters:
        question_id (int): 题目ID
        point_text (str): rubric点的文本
        
        Returns:
        int: 权重值，默认为1
        """
        if question_id not in self.rubric_weights:
            return 1
        
        weights = self.rubric_weights[question_id]
        
        # 精确匹配
        point_key = point_text[:50] if point_text else ""
        if point_key in weights:
            return weights[point_key]
        
        # 模糊匹配：查找包含关键词的权重
        for key, weight in weights.items():
            if key and point_text and (key in point_text or point_text in key):
                return weight
        
        return 1  # 默认权重
    
    def compare_single_question(self, question_id: int, annotator: str) -> Optional[Dict]:
        """
        比较单个题目的模型结果和人类标注
        
        Parameters:
        question_id (int): 题目ID
        annotator (str): 标注者ID
        
        Returns:
        Optional[Dict]: 比较结果，如果无法比较则返回None
        """
        # 检查数据是否存在
        if question_id not in self.human_annotations:
            print(f"  ⚠️  Q{question_id}: 无人类标注数据")
            return None
        
        if annotator not in self.human_annotations[question_id]:
            print(f"  ⚠️  Q{question_id}: 无标注者{annotator}的数据")
            return None
        
        if question_id not in self.model_results:
            print(f"  ⚠️  Q{question_id}: 无模型评估数据")
            return None
        
        human_data = self.human_annotations[question_id][annotator]
        model_data = self.model_results[question_id]
        
        # 获取人类标注的rubric评估
        human_rubric_eval = human_data.get("rubric_eval", [])
        
        # 获取模型的coverage_results
        model_coverage = model_data.get("result", {}).get("coverage_results", [])
        
        if not human_rubric_eval or not model_coverage:
            print(f"  ⚠️  Q{question_id}: 缺少rubric评估数据")
            return None
        
        # 建立映射关系（通过point文本匹配）
        comparisons = []
        matched_count = 0
        
        for human_item in human_rubric_eval:
            human_point = human_item.get("point", "")
            human_covered = human_item.get("covered", False)
            
            # 在模型结果中查找匹配的point
            model_covered = None
            model_weight = 1
            
            for model_item in model_coverage:
                model_point = model_item.get("point", "")
                
                # 文本相似性匹配（简单版本：检查是否包含相同的关键词）
                if self.is_point_match(human_point, model_point):
                    model_covered = model_item.get("covered", False)
                    model_weight = model_item.get("weight", 1)
                    matched_count += 1
                    break
            
            if model_covered is not None:
                # 获取权重
                weight = self.find_rubric_weight(question_id, human_point)
                
                comparisons.append({
                    "point": human_point[:100] + "..." if len(human_point) > 100 else human_point,
                    "human_covered": human_covered,
                    "model_covered": model_covered,
                    "weight": weight,
                    "match": human_covered == model_covered
                })
        
        if not comparisons:
            print(f"  ⚠️  Q{question_id}: 无法匹配任何rubric项")
            return None
        
        print(f"  ✅ Q{question_id}: 匹配了 {matched_count}/{len(human_rubric_eval)} 个rubric项")
        
        return {
            "question_id": question_id,
            "annotator": annotator,
            "total_items": len(comparisons),
            "matched_items": matched_count,
            "comparisons": comparisons
        }
    
    def is_point_match(self, human_point: str, model_point: str) -> bool:
        """
        判断两个rubric点是否匹配
        
        Parameters:
        human_point (str): 人类标注的点
        model_point (str): 模型评估的点
        
        Returns:
        bool: 是否匹配
        """
        if not human_point or not model_point:
            return False
        
        # 简单的文本匹配策略
        human_words = set(human_point.lower().split())
        model_words = set(model_point.lower().split())
        
        # 计算交集比例
        intersection = human_words.intersection(model_words)
        union = human_words.union(model_words)
        
        if len(union) == 0:
            return False
        
        similarity = len(intersection) / len(union)
        
        # 如果相似度超过阈值，认为是匹配的
        return similarity >= 0.3
    
    def calculate_metrics(self, comparisons: List[Dict]) -> Dict:
        """
        计算评估指标
        
        Parameters:
        comparisons (List[Dict]): 比较结果列表
        
        Returns:
        Dict: 计算出的指标
        """
        if not comparisons:
            return {
                "unweighted": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0},
                "weighted": {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
            }
        
        # 不带权重的计算
        tp = sum(1 for c in comparisons if c["human_covered"] and c["model_covered"])
        tn = sum(1 for c in comparisons if not c["human_covered"] and not c["model_covered"])
        fp = sum(1 for c in comparisons if not c["human_covered"] and c["model_covered"])
        fn = sum(1 for c in comparisons if c["human_covered"] and not c["model_covered"])
        
        total = len(comparisons)
        accuracy_unweighted = (tp + tn) / total if total > 0 else 0
        precision_unweighted = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_unweighted = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_unweighted = 2 * (precision_unweighted * recall_unweighted) / (precision_unweighted + recall_unweighted) if (precision_unweighted + recall_unweighted) > 0 else 0
        
        # 带权重的计算
        tp_weighted = sum(c["weight"] for c in comparisons if c["human_covered"] and c["model_covered"])
        tn_weighted = sum(c["weight"] for c in comparisons if not c["human_covered"] and not c["model_covered"])
        fp_weighted = sum(c["weight"] for c in comparisons if not c["human_covered"] and c["model_covered"])
        fn_weighted = sum(c["weight"] for c in comparisons if c["human_covered"] and not c["model_covered"])
        
        total_weight = sum(c["weight"] for c in comparisons)
        accuracy_weighted = (tp_weighted + tn_weighted) / total_weight if total_weight > 0 else 0
        precision_weighted = tp_weighted / (tp_weighted + fp_weighted) if (tp_weighted + fp_weighted) > 0 else 0
        recall_weighted = tp_weighted / (tp_weighted + fn_weighted) if (tp_weighted + fn_weighted) > 0 else 0
        f1_weighted = 2 * (precision_weighted * recall_weighted) / (precision_weighted + recall_weighted) if (precision_weighted + recall_weighted) > 0 else 0
        
        return {
            "unweighted": {
                "accuracy": round(accuracy_unweighted, 4),
                "precision": round(precision_unweighted, 4),
                "recall": round(recall_unweighted, 4),
                "f1": round(f1_unweighted, 4),
                "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
            },
            "weighted": {
                "accuracy": round(accuracy_weighted, 4),
                "precision": round(precision_weighted, 4),
                "recall": round(recall_weighted, 4),
                "f1": round(f1_weighted, 4),
                "confusion_matrix": {"tp": tp_weighted, "tn": tn_weighted, "fp": fp_weighted, "fn": fn_weighted}
            }
        }
    
    def run_comparison(self, target_annotator: str = "A") -> Dict:
        """
        运行完整的比较分析
        
        Parameters:
        target_annotator (str): 目标标注者ID，默认为"A"
        
        Returns:
        Dict: 完整的比较结果
        """
        print("\n" + "="*60)
        print(f"开始模型与人类标注结果对比分析 (标注者: {target_annotator})")
        print("="*60)
        
        all_comparisons = []
        detailed_results = {}
        processed_questions = 0
        
        # 遍历所有题目，只处理指定标注者的数据
        for question_id in sorted(self.human_annotations.keys()):
            if target_annotator not in self.human_annotations[question_id]:
                print(f"\n⚠️  Q{question_id}: 无标注者{target_annotator}的数据，跳过")
                continue
            
            print(f"\n处理 Q{question_id} - 标注者{target_annotator}:")
            
            result = self.compare_single_question(question_id, target_annotator)
            if result:
                detailed_results[question_id] = result
                all_comparisons.extend(result["comparisons"])
                processed_questions += 1
        
        if processed_questions == 0:
            print(f"\n❌ 错误: 没有找到标注者{target_annotator}的任何有效数据")
            return {}
        
        # 计算总体指标
        print(f"\n" + "="*50)
        print("计算总体评估指标")
        print("="*50)
        
        overall_metrics = self.calculate_metrics(all_comparisons)
        
        # 按题目计算指标
        question_metrics = {}
        for question_id, question_data in detailed_results.items():
            if question_data and "comparisons" in question_data:
                question_metrics[question_id] = self.calculate_metrics(question_data["comparisons"])
        
        # 汇总结果
        comparison_results = {
            "summary": {
                "annotator": target_annotator,
                "total_questions": len(detailed_results),
                "processed_questions": processed_questions,
                "total_comparisons": len(all_comparisons),
                "overall_metrics": overall_metrics
            },
            "question_metrics": question_metrics,
            "detailed_results": detailed_results
        }
        
        return comparison_results
    
    def print_summary_report(self, results: Dict):
        """打印汇总报告"""
        if not results:
            print("❌ 没有可用的比较结果")
            return
            
        print(f"\n" + "="*60)
        print("📊 模型与人类标注对比分析报告")
        print("="*60)
        
        summary = results["summary"]
        overall = summary["overall_metrics"]
        
        print(f"\n📈 总体统计:")
        print(f"  标注者: {summary['annotator']}")
        print(f"  处理题目数: {summary['processed_questions']}")
        print(f"  比较项目数: {summary['total_comparisons']}")
        
        print(f"\n🎯 总体性能指标:")
        print(f"  {'指标':<12} {'不带权重':<12} {'带权重':<12}")
        print(f"  {'-'*36}")
        print(f"  {'准确度':<12} {overall['unweighted']['accuracy']:<12} {overall['weighted']['accuracy']:<12}")
        print(f"  {'精确率':<12} {overall['unweighted']['precision']:<12} {overall['weighted']['precision']:<12}")
        print(f"  {'召回率':<12} {overall['unweighted']['recall']:<12} {overall['weighted']['recall']:<12}")
        print(f"  {'F1分数':<12} {overall['unweighted']['f1']:<12} {overall['weighted']['f1']:<12}")
        
        # 显示混淆矩阵
        unw_cm = overall['unweighted']['confusion_matrix']
        w_cm = overall['weighted']['confusion_matrix']
        print(f"\n📋 混淆矩阵:")
        print(f"  {'类型':<12} {'不带权重':<12} {'带权重':<12}")
        print(f"  {'-'*36}")
        print(f"  {'TP (正确覆盖)':<12} {unw_cm['tp']:<12} {w_cm['tp']:<12.1f}")
        print(f"  {'TN (正确未覆盖)':<12} {unw_cm['tn']:<12} {w_cm['tn']:<12.1f}")
        print(f"  {'FP (错误覆盖)':<12} {unw_cm['fp']:<12} {w_cm['fp']:<12.1f}")
        print(f"  {'FN (错误未覆盖)':<12} {unw_cm['fn']:<12} {w_cm['fn']:<12.1f}")
        
        # 按题目显示指标
        print(f"\n📋 分题目性能:")
        question_metrics = results["question_metrics"]
        if question_metrics:
            print(f"  {'题目':<6} {'F1(无权)':<10} {'F1(带权)':<10} {'准确度(无权)':<12} {'准确度(带权)':<12}")
            print(f"  {'-'*50}")
            
            for question_id in sorted(question_metrics.keys()):
                metrics = question_metrics[question_id]
                unw_f1 = metrics["unweighted"]["f1"]
                w_f1 = metrics["weighted"]["f1"]
                unw_acc = metrics["unweighted"]["accuracy"]
                w_acc = metrics["weighted"]["accuracy"]
                print(f"  Q{question_id:<5} {unw_f1:<10} {w_f1:<10} {unw_acc:<12} {w_acc:<12}")
        else:
            print("  无可用的分题目数据")
        
        # 性能评估
        f1_score = overall['weighted']['f1']
        print(f"\n🏆 性能评估:")
        if f1_score >= 0.8:
            print(f"  ✅ 模型表现优秀 (F1={f1_score:.3f})")
        elif f1_score >= 0.6:
            print(f"  🟡 模型表现良好，有改进空间 (F1={f1_score:.3f})")
        else:
            print(f"  🔴 模型表现需要显著改进 (F1={f1_score:.3f})")
        
        # 提供改进建议
        precision = overall['weighted']['precision']
        recall = overall['weighted']['recall']
        
        print(f"\n💡 改进建议:")
        if precision > recall + 0.1:
            print(f"  • 模型过于保守，建议降低判断阈值以提高召回率")
        elif recall > precision + 0.1:
            print(f"  • 模型过于激进，建议提高判断阈值以提高精确率")
        else:
            print(f"  • 精确率和召回率较为均衡，可考虑整体优化策略")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型评估结果与人类标注结果对比分析')
    
    parser.add_argument('--human_data_dir', type=str, 
                        default='data/meta_eval_data/meta_eval',
                        help='人类标注数据目录')
    parser.add_argument('--model_results_file', type=str,
                        default='results/results_for_specific/targeted_evaluation/google_gemini-2.5-flash/meta_eval.json',
                        help='模型评估结果文件')
    parser.add_argument('--rubric_file', type=str,
                        default='data/eval_data/rubric.json',
                        help='rubric权重文件')
    parser.add_argument('--output_dir', type=str,
                        default='results/comparison_analysis',
                        help='对比结果输出目录')
    parser.add_argument('--annotator', type=str,
                        default='A',
                        help='目标标注者ID (默认: A)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.human_data_dir):
        print(f"错误: 人类标注数据目录不存在 - {args.human_data_dir}")
        return
    
    if not os.path.exists(args.model_results_file):
        print(f"错误: 模型结果文件不存在 - {args.model_results_file}")
        return
    
    if not os.path.exists(args.rubric_file):
        print(f"错误: rubric文件不存在 - {args.rubric_file}")
        return
    
    # 创建输出目录
    create_output_directory(args.output_dir)
    
    # 创建比较分析器
    comparator = ModelHumanComparison()
    
    # 加载数据
    if not comparator.load_human_annotations(args.human_data_dir):
        print("加载人类标注数据失败，退出程序")
        return
    
    # 检查是否存在指定标注者的数据
    annotator_found = False
    for question_data in comparator.human_annotations.values():
        if args.annotator in question_data:
            annotator_found = True
            break
    
    if not annotator_found:
        print(f"错误: 未找到标注者 '{args.annotator}' 的任何数据")
        print(f"可用的标注者: {set(ann for q_data in comparator.human_annotations.values() for ann in q_data.keys())}")
        return
    
    if not comparator.load_model_results(args.model_results_file):
        print("加载模型评估结果失败，退出程序")
        return
    
    if not comparator.load_rubric_weights(args.rubric_file):
        print("加载rubric权重失败，退出程序")
        return
    
    # 运行比较分析
    results = comparator.run_comparison(target_annotator=args.annotator)
    
    if not results:
        print("比较分析失败，没有生成结果")
        return
    
    # 打印报告
    comparator.print_summary_report(results)
    
    # 保存结果
    output_file = os.path.join(args.output_dir,"gpt-4.1", f"model_human_comparison_annotator_{args.annotator}.json")
    if save_json_file(results, output_file):
        print(f"\n✅ 详细对比结果已保存到: {output_file}")
    else:
        print(f"\n❌ 保存对比结果失败")


if __name__ == "__main__":
    main()
