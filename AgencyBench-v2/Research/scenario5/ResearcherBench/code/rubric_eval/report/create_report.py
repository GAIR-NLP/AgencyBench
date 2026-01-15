#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用极简评估报告生成器 - 只输出三个核心指标
支持命令行参数或交互式输入
"""

import json
import os
import sys

def generate_simple_report(json_file_path):
    """
    生成极简报告
    
    Args:
        json_file_path (str): JSON文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件不存在 - {json_file_path}")
        return False
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            print("错误: 文件中没有评估结果")
            return False
        
        # 计算平均值
        recalls = [r["result"]["recall"] for r in results]
        
        avg_recall = sum(recalls) / len(recalls)
        
        # 生成报告内容
        report_content = f"""平均覆盖率: {avg_recall:.4f}"""
        
        # 保存报告
        output_file = json_file_path.replace('.json', '_simple_report.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 报告生成成功: {output_file}")
        print(f"📊 共处理 {len(results)} 条评估结果")
        print("\n📋 报告内容:")
        print(report_content)
        
        return True
        
    except Exception as e:
        print(f"❌ 生成报告时出错: {e}")
        return False

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 使用命令行参数指定的文件
        json_file = sys.argv[1]
        generate_simple_report(json_file)
    else:
        # 交互式输入
        print("🚀 极简评估报告生成器")
        print("=" * 30)
        
        # 检查默认文件
        default_file = r"F:\projects\DAIR_Benchmark_1\results\rubric_eval\o3-mini\Google_evaluation_results.json"
        if os.path.exists(default_file):
            print(f"📁 找到默认文件: {default_file}")
            choice = input("是否使用此文件? (y/n): ").lower().strip()
            
            if choice in ['y', 'yes', '']:
                generate_simple_report(default_file)
                return
        
        # 手动输入文件路径
        while True:
            file_path = input("\n请输入JSON文件路径 (或按Enter退出): ").strip()
            if not file_path:
                print("退出程序")
                break
            
            # 移除引号（如果用户复制粘贴路径带引号）
            file_path = file_path.strip('"\'')
            
            if generate_simple_report(file_path):
                break

if __name__ == "__main__":
    main()