import pandas as pd
import matplotlib.pyplot as plt
import ast

# 你的 test.csv 绝对路径
csv_path = r"D:\桌面\multiview_pig_posture_recognition\test.csv"

# 读取 CSV 文件
df = pd.read_csv(csv_path)

# 解析 bbox 列，提取高度 (h)
def extract_height(bbox_str):
    try:
        bbox_list = ast.literal_eval(bbox_str)
        # bbox 格式为 [x, y, w, h]，高度是第四个元素，索引 3
        height = bbox_list[3]
        return height
    except:
        return None

df['height'] = df['bbox'].apply(extract_height)
df = df.dropna(subset=['height'])

# 打印统计信息
print("BBox 高度统计信息：")
print(df['height'].describe())

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(df['height'], bins=30, edgecolor='black', alpha=0.7, color='forestgreen')
plt.xlabel('Bounding Box Height (pixels)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of BBox Height in test.csv', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 添加均值和 median 参考线
mean_h = df['height'].mean()
median_h = df['height'].median()
plt.axvline(mean_h, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_h:.1f}')
plt.axvline(median_h, color='blue', linestyle='dashed', linewidth=1, label=f'Median: {median_h:.1f}')
plt.legend()

# 保存图片（保存到当前脚本所在目录，即 test_images 文件夹）
output_path = 'bbox_height_distribution.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"高度分布直方图已保存为 {output_path}")