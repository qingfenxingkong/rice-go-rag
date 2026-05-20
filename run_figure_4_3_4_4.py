import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（根据您的系统环境可能需要调整，如 'SimHei' 或 'Microsoft YaHei'）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_figure_4_3():
    # 数据来源
    methods = ['Dict', 'NLTK', 'Vector', 'LLM(原始)', 'LLM(保守)', 'Ensemble']
    f1_scores = [0.255, 0.437, 0.396, 0.480, 0.548, 0.483]
    neg_fp = [1.2, 0.8, 15.4, 28.6, 5.7, 100.0]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # F1 分数柱状图
    color = 'steelblue'
    ax1.set_ylabel('F1 分数', color=color, fontsize=12)
    ax1.bar(methods, f1_scores, color=color, alpha=0.7, label='F1 分数')
    ax1.tick_params(axis='y', labelcolor=color)

    # 负例误报率折线图 (双轴)
    ax2 = ax1.twinx()
    color = 'crimson'
    ax2.set_ylabel('负例误报率 (NegFP%) - 对数坐标', color=color, fontsize=12)
    ax2.plot(methods, neg_fp, color=color, marker='o', linewidth=2, label='误报率')
    ax2.set_yscale('log')  # 误报率差异巨大，建议用对数坐标
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('图 4-3 各方法指标综合对比图', fontsize=14)
    fig.tight_layout()
    fig.savefig('figure_4_3.png', dpi=150, bbox_inches='tight')
    plt.show()


def generate_figure_4_4():
    # 数据来源
    labels = np.array(['精准率', '召回率', 'F1', '误报率(反向)'])
    # 归一化处理用于雷达图展示
    strict = [0.642, 0.310, 0.418, 0.995]  # 误报率取 1-NegFP
    balanced = [0.521, 0.485, 0.502, 0.876]
    llm_cons = [0.618, 0.492, 0.548, 0.943]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    def add_to_radar(data, label, color):
        values = data + data[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.25)

    add_to_radar(strict, 'Strict 模式', 'green')
    add_to_radar(balanced, 'Balanced 模式', 'blue')
    add_to_radar(llm_cons, 'LLM (保守模式)', 'orange')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title('图 4-4 不同集成预设与保守模式对比雷达图', fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    fig.savefig('figure_4_4.png', dpi=150, bbox_inches='tight')
    plt.show()


# 调用函数生成
generate_figure_4_3()
generate_figure_4_4()
