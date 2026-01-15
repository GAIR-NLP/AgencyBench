#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键生成三张单独图表
专为您的7个报告文件定制
"""

import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """一键生成覆盖率图表"""
    
    print("🚀 生成覆盖率图表")
    print("=" * 25)
    
    # 查找报告文件
    patterns = [
        r"F:\projects\DAIR_Benchmark_1\results\rubric_eval\o3-mini\*_simple_report.txt",
        r"*_simple_report.txt"
    ]
    
    files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            break
    
    if not files:
        print("❌ 未找到报告文件")
        input("按Enter退出...")
        return
    
    print(f"📊 找到 {len(files)} 个报告文件")
    
    # 解析数据
    models, recalls = [], []
    
    for file in files:
        try:
            model = os.path.basename(file).replace('_evaluation_results_simple_report.txt', '')
            
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            recall = float(re.search(r'平均覆盖率:\s*([\d.]+)', content).group(1))
            
            models.append(model)
            recalls.append(recall)
            
        except:
            continue
    
    if not models:
        print("❌ 解析失败")
        return
    
    # 生成覆盖率图表
    print(f"\n📊 生成覆盖率图表...")
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, recalls, color='#FF6B6B', alpha=0.8, edgecolor='black')
    plt.title('覆盖率对比', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('覆盖率', fontsize=12)
    plt.xlabel('模型', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, recalls):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(recalls)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('覆盖率图表.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ 覆盖率图表已保存: 覆盖率图表.png")
    
    print(f"\n🎉 图表生成完成！")
    print(f"📁 文件: 覆盖率图表.png")
    
    # 简单排名
    print(f"\n🏆 排名:")
    recall_best = models[recalls.index(max(recalls))]
    
    print(f"📈 覆盖率最高: {recall_best} ({max(recalls):.4f})")

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("❌ 需要安装: pip install matplotlib")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    input("\n按Enter键退出...")