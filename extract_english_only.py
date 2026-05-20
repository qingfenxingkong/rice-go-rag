import json

# 读取原始数据
with open('C:/Users/yyq/Desktop/毕业设计/代码/data/benchmark_go_ner_200.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 过滤出所有英文样本 (lang == "en")
english_cases = [case for case in data['cases'] if case['lang'] == 'en']

# 创建新的数据集结构
english_dataset = {
    "name": "go_ner_benchmark_english_only",
    "description": "English-only subset extracted from benchmark_go_ner_200",
    "total_cases": len(english_cases),
    "source": "benchmark_go_ner_200.json",
    "filter_criteria": "lang == 'en'",
    "cases": english_cases
}

# 保存为新的JSON文件
output_path = 'C:/Users/yyq/Desktop/毕业设计/代码/data/benchmark_go_ner_english_only.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(english_dataset, f, ensure_ascii=False, indent=2)

print(f"成功提取 {len(english_cases)} 个英文样本")
print(f"新数据集已保存至: {output_path}")

# 统计不同难度的样本数量
difficulty_stats = {}
for case in english_cases:
    diff = case.get('difficulty', 'unknown')
    difficulty_stats[diff] = difficulty_stats.get(diff, 0) + 1

print("\n难度分布统计:")
for diff, count in sorted(difficulty_stats.items()):
    print(f"  {diff}: {count} 个样本")
