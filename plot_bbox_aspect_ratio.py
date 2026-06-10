import pandas as pd
import matplotlib.pyplot as plt
import ast

# 你的 test.csv 绝对路径
csv_path = r"D:\桌面\multiview_pig_posture_recognition\test.csv"

# 读取 CSV 文件
df = pd.read_csv(csv_path)

# 解析 bbox 列，提取宽度和高度，计算长宽比
def extract_aspect_ratio(bbox_str):
    try:
        bbox_list = ast.literal_eval(bbox_str)
        width = bbox_list[2]
        height = bbox_list[3]
        # 避免除零错误（虽然实际数据中高度应该 > 0）
        if height > 0:
            aspect_ratio = width / height
        else:
            aspect_ratio = None
        return aspect_ratio
    except:
        return None

df['aspect_ratio'] = df['bbox'].apply(extract_aspect_ratio)
df = df.dropna(subset=['aspect_ratio'])

# 打印统计信息
print("BBox 长宽比统计信息：")
print(df['aspect_ratio'].describe())

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(df['aspect_ratio'], bins=30, edgecolor='black', alpha=0.7, color='purple')
plt.xlabel('Aspect Ratio (Width / Height)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of BBox Aspect Ratio in test.csv', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 添加均值和 median 参考线
mean_ar = df['aspect_ratio'].mean()
median_ar = df['aspect_ratio'].median()
plt.axvline(mean_ar, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_ar:.2f}')
plt.axvline(median_ar, color='blue', linestyle='dashed', linewidth=1, label=f'Median: {median_ar:.2f}')
plt.legend()

# 保存图片（保存到当前脚本所在目录，即 test_images 文件夹）
output_path = 'bbox_aspect_ratio_distribution.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"长宽比分布直方图已保存为 {output_path}")