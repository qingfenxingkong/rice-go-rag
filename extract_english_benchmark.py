#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从benchmark_go_ner_200.json中提取所有英文样本
生成新的测试数据集: benchmark_go_ner_english_only.json
"""

import json
from pathlib import Path

def extract_english_cases():
    # 文件路径
    input_file = Path('data/benchmark_go_ner_200.json')
    output_file = Path('data/benchmark_go_ner_en_and_mixed.json')
    
    print(f"正在读取: {input_file}")
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 过滤出所有英文和混合语言样本 (lang == "en" or lang == "mixed")
    english_and_mixed_cases = [case for case in data['cases'] if case['lang'] in ['en', 'mixed']]
    
    print(f"找到 {len(english_and_mixed_cases)} 个英文+混合语言样本")
    
    # 统计各语言类型的数量
    lang_stats = {}
    for case in english_and_mixed_cases:
        lang = case.get('lang', 'unknown')
        lang_stats[lang] = lang_stats.get(lang, 0) + 1
    
    print("\n语言类型分布:")
    for lang, count in sorted(lang_stats.items()):
        print(f"  {lang}: {count} 个样本")
    
    # 创建新的数据集结构
    filtered_dataset = {
        "name": "go_ner_benchmark_en_and_mixed",
        "description": "English and mixed-language subset extracted from benchmark_go_ner_200",
        "total_cases": len(english_and_mixed_cases),
        "source": "benchmark_go_ner_200.json",
        "filter_criteria": "lang in ['en', 'mixed']",
        "language_distribution": lang_stats,
        "cases": english_and_mixed_cases
    }
    
    # 保存为新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 新数据集已保存至: {output_file}")
    
    # 统计不同难度的样本数量
    difficulty_stats = {}
    for case in english_and_mixed_cases:
        diff = case.get('difficulty', 'unknown')
        difficulty_stats[diff] = difficulty_stats.get(diff, 0) + 1
    
    print("\n难度分布统计:")
    for diff, count in sorted(difficulty_stats.items()):
        percentage = (count / len(english_and_mixed_cases)) * 100
        print(f"  {diff:20s}: {count:3d} 个样本 ({percentage:5.1f}%)")
    
    # 显示前几个和后几个样本ID
    case_ids = [case['id'] for case in english_and_mixed_cases]
    print(f"\n样本ID范围: {min(case_ids)} - {max(case_ids)}")
    print(f"前10个ID: {case_ids[:10]}")
    print(f"后10个ID: {case_ids[-10:]}")
    
    return len(english_and_mixed_cases)

if __name__ == '__main__':
    try:
        count = extract_english_cases()
        print(f"\n✓ 成功提取 {count} 个英文样本!")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
