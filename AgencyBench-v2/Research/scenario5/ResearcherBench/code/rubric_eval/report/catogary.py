#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键修改版分类分析
快速运行两类别分析：Open Consulting + Technical & Literature
"""

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def quick_modified_analysis():
    """一键运行修改版分类分析"""
    
    print("🚀 一键修改版分类分析")
    print("=" * 25)
    print("📋 分析类别: Open Consulting + Technical & Literature")
    print()
    
    # 1. 检查环境
    questions_file = "data/eval_data/questions.json"
    if not os.path.exists(questions_file):
        print(f"❌ 未找到文件: {questions_file}")
        return
    
    # 2. 加载问题分类
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        categories = {
            "Open Consulting": [],
            "Technical & Literature": []
        }
        
        for q in questions:
            category = q.get("category", "")
            if category == "Open Consulting":
                categories["Open Consulting"].append(q["id"])
            elif category in ["Technical Details", "Literature Review"]:
                categories["Technical & Literature"].append(q["id"])
        
        print(f"✅ 问题分类统计:")
        for cat, ids in categories.items():
            print(f"   • {cat}: {len(ids)} 题")
        
    except Exception as e:
        print(f"❌ 加载问题文件失败: {e}")
        return
    
    # 3. 查找评估结果
    result_files = glob.glob("results/rubric_eval/*_evaluation_results.json", recursive=True)
    if not result_files:
        result_files = glob.glob("*_evaluation_results.json")
    
    if not result_files:
        print("\n❌ 未找到评估结果文件")
        return
    
    print(f"\n📁 找到 {len(result_files)} 个评估结果文件")
    
    # 4. 创建输出目录
    os.makedirs("quick_modified_results", exist_ok=True)
    
    # 5. 分析每个模型
    for result_file in result_files:
        try:
            model_name = os.path.basename(result_file).replace('_evaluation_results.json', '')
            print(f"\n🔍 分析模型: {model_name}")
            
            # 加载评估结果
            with open(result_file, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
            
            # 分析两个类别
            model_stats = {}
            for category, question_ids in categories.items():
                category_results = []
                for result in evaluation_results:
                    if result["id"] in question_ids:
                        category_results.append(result["result"])
                
                if category_results:
                    recalls = [r["recall"] for r in category_results]
                    
                    model_stats[category] = {
                        "count": len(category_results),
                        "avg_recall": np.mean(recalls)
                    }
            
            # 生成简单报告
            if model_stats:
                generate_quick_report(model_name, model_stats)
                create_quick_chart(model_name, model_stats)
            
        except Exception as e:
            print(f"❌ 分析 {result_file} 时出错: {e}")
    
    print(f"\n🎉 修改版分析完成！")
    print(f"📁 结果保存在: quick_modified_results/")

def generate_quick_report(model_name, model_stats):
    """生成快速报告"""
    
    report_content = f"""{model_name} - 修改版分类分析报告
{'='*45}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 两大类别表现:
"""

    for category, stats in model_stats.items():
        report_content += f"""
{category}:
  题目数量: {stats['count']} 题
  平均覆盖率: {stats['avg_recall']:.4f}
"""

    # 找出更优的类别
    if len(model_stats) == 2:
        categories = list(model_stats.keys())
        stats1 = model_stats[categories[0]]
        stats2 = model_stats[categories[1]]
        
        comp1 = stats1['avg_recall']
        comp2 = stats2['avg_recall']
        
        better_category = categories[0] if comp1 > comp2 else categories[1]
        diff = abs(comp1 - comp2)
        
        report_content += f"""
🏆 表现更好的类别: {better_category}
📈 覆盖率差异: {diff:.4f}

💡 应用建议:
"""
        if better_category == "Open Consulting":
            report_content += "• 该模型更适合咨询类应用\n• 在开放性问题解答方面表现优秀\n"
        else:
            report_content += "• 该模型更适合技术文档和学术支持\n• 在专业知识和文献调研方面表现优秀\n"

    report_content += f"\n{'='*45}"
    
    # 保存报告
    report_file = f"quick_modified_results/{model_name}_quick_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📄 快速报告: {report_file}")

def create_quick_chart(model_name, model_stats):
    """创建快速图表"""
    
    if len(model_stats) != 2:
        return
    
    categories = list(model_stats.keys())
    recalls = [model_stats[cat]['avg_recall'] for cat in categories]
    counts = [model_stats[cat]['count'] for cat in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(categories))
    width = 0.4
    
    bars = ax.bar(x, recalls, width, label='覆盖率', color='#FF6B6B', alpha=0.8)
    
    ax.set_title(f'{model_name} - 修改版两类别覆盖率对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('问题类别')
    ax.set_ylabel('覆盖率')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.replace('&', '+') for cat in categories])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 添加题目数量
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax.text(i, -0.05, f'({count}题)', ha='center', va='top', 
               transform=ax.get_xaxis_transform(), fontsize=10, style='italic')
    
    plt.tight_layout()
    
    chart_file = f"quick_modified_results/{model_name}_quick_chart.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 快速图表: {chart_file}")

def main():
    """主函数"""
    try:
        quick_modified_analysis()
        
        print(f"\n💡 完整功能:")
        print(f"   • simple_category_analysis_modified.py - 详细的修改版分析")
        print(f"   • category_comparison_tool.py - 对比三类别vs两类别")
        print(f"   • category_analysis_updated.bat - 一键选择分析方式")
        
    except ImportError:
        print("❌ 需要安装: pip install matplotlib numpy")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    input("\n按Enter键退出...")

if __name__ == "__main__":
    main()