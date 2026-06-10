# 多视角猪姿态识别
本项目面向真实养猪场场景，提供一套高鲁棒性的细粒度图像分类方案。针对场景中普遍存在的目标遮挡、光照变化、样本类别不均衡、标签噪声以及数据泄露等工业痛点，搭建了一套完整的计算机视觉系统，可应用于猪只智能健康监测与行为分析工作。

**🏆竞赛链接**：https://www.kaggle.com/competitions/multi-view-pig-posture-recognition

## 🐷 姿态类别说明
| 类别编号 | 类别名称 | 姿态描述 |
|---------|------------|-------------|
| 0 | Lateral_lying_left | 左侧躺 |
| 1 | Lateral_lying_right | 右侧躺 |
| 2 | Sitting | 坐姿 |
| 3 | Standing | 站姿 |
| 4 | Sternal_lying | 腹卧 |

## ⚙️ 环境依赖
```txt
Python >= 3.8
pandas==1.5.3
numpy==1.24.3
torch==2.0.1
torchvision==0.15.2
timm==0.9.2
matplotlib==3.7.1
seaborn==0.12.2
opencv-python==4.7.0
Pillow==9.5.0
scikit-learn==1.2.2
```
## 📂 项目目录结构
```python
├── data/
│   ├── train1.csv
│   ├── train2.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── pig_posture_classes.txt
│   └── changes.csv
├── figures/                  # 数据可视化结果存放目录
├── pig_timm_resnet_attention_train.py   # 模型主训练脚本
├── data_visualization/        
│   ├── python plot_bbox_width.py
│   ├── python plot_bbox_height.py
│   ├── python plot_bbox_aspect_ratio.py
│   ├── python plot_class_dist.py
├── README_CN.md
└── README.md
```
## 🔥 环境安装
```python
## 克隆代码仓库
git clone https://github.com/isleep888/Multi-view-Pig-Posture-Recognition.git
cd resnet_pig_package

# 创建 Python 3.8 运行环境
conda create -n pig-pose python=3.8
conda activate pig-pose
pip install -r requirements.txt
```

## 🔥 模型训练与推理
```python
python pig_timm_resnet_attention_train.py
```
该脚本可自动完成数据清洗、均衡采样、分组交叉验证、五折模型训练、测试时增强推理，并最终生成竞赛提交文件。

## 🔥 数据可视化与分析
```python
python plot_bbox_width.py        # 可视化边界框宽度分布
python plot_bbox_height.py       # 可视化边界框高度分布
python plot_bbox_aspect_ratio.py # 可视化边界框长宽比分布
python plot_class_dist.py        # 统计并绘制训练集样本类别分布
```

## 🏅 竞赛排名
本方案结合数据清洗、自适应数据增强、软隔离策略以及模型集成等多项优化手段，在 Kaggle 官方竞赛中取得了不错的成绩：

·竞赛最终排名：257 支参赛队伍中位列第 103 名

·核心优势：在复杂的真实养殖场景下具备稳定的泛化能力，有效解决了计算机视觉落地过程中的各类常见工业难题。
![竞赛排名](./kaggle_排名.png)

## 联系方式
如有问题欢迎邮件交流：202481313616@m.scnu.edu.cn。我会在业余时间持续维护本项目仓库。

## ⭐ Star & Fork
如果本仓库对你的学习或竞赛研究有所帮助，欢迎 Star 和 Fork，项目后续也会持续更新优化。
