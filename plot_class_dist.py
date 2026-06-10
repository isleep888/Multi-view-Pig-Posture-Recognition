import pandas as pd
import matplotlib.pyplot as plt

# -------------------------- 全局样式配置 --------------------------
# 解决中文乱码，Windows默认雅黑字体
plt.rcParams["font.family"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
# 全局字体大小层级
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
# 线条和网格样式
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.alpha'] = 0.3

# -------------------------- 读取数据 --------------------------
df = pd.read_csv("D:/桌面/multiview_pig_posture_recognition/train2.csv")
dist = df["class_id"].value_counts().sort_index()

# -------------------------- 绘制深蓝色柱状图 --------------------------
# 16:9横版画布，完美适配PPT
plt.figure(figsize=(16, 9), dpi=150)

# 柱状图颜色强制使用你指定的 #00337C
bars = plt.bar(
    x=dist.index,
    height=dist.values,
    color='#00337C',  # 你指定的深蓝色，和train1完全统一
    width=0.6,
    edgecolor='white',
    linewidth=1
)

# 标题设置
plt.title('train2.csv 样本类别分布', fontweight='bold', pad=30)

# 坐标轴设置
plt.xlabel('姿态类别 class_id', labelpad=15)
plt.ylabel('样本数量', labelpad=15)
plt.xticks(dist.index)

# y轴范围和刻度，完美适配train2的数值
plt.ylim(0, 10500)
plt.yticks([0, 2000, 4000, 6000, 8000, 10000])

# 显示y轴网格线
plt.grid(axis='y', zorder=0)

# 柱子顶部标注数值，位置精准不溢出
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.,
        height + 150,
        f'{int(height)}',
        ha='center',
        va='bottom',
        fontsize=16,
        fontweight='medium'
    )

# 自动调整布局，无内容溢出
plt.tight_layout()

# 保存高清图到桌面，文件名清晰区分
plt.savefig('D:/桌面/train2_样本类别分布_深蓝色版.png', dpi=150, bbox_inches='tight')
print("✅ 深蓝色版train2柱状图已成功保存到桌面！")

# 显示图片
plt.show()