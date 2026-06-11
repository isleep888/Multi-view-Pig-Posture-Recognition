import pandas as pd
import matplotlib.pyplot as plt
import ast

# 修改为你的 test.csv 实际绝对路径
csv_path = r"D:\桌面\multiview_pig_posture_recognition\test.csv"

# 读取 CSV
df = pd.read_csv(csv_path)

# 解析 bbox 列
def extract_width(bbox_str):
    try:
        bbox_list = ast.literal_eval(bbox_str)
        return bbox_list[2]  # 宽度
    except:
        return None

df['width'] = df['bbox'].apply(extract_width)
df = df.dropna(subset=['width'])

# 统计信息
print(df['width'].describe())

# 绘图
plt.figure(figsize=(10, 6))
plt.hist(df['width'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
plt.xlabel('Bounding Box Width (pixels)')
plt.ylabel('Frequency')
plt.title('Distribution of BBox Width in test.csv')
plt.grid(axis='y', linestyle='--', alpha=0.5)

mean_w = df['width'].mean()
median_w = df['width'].median()
plt.axvline(mean_w, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_w:.1f}')
plt.axvline(median_w, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_w:.1f}')
plt.legend()

plt.savefig('bbox_width_distribution.png', dpi=150, bbox_inches='tight')
plt.show()