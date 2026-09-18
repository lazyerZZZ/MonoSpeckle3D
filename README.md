# 基于深度学习的伪重叠散斑三维形貌重建

本项目对应毕业论文《基于深度学习伪重叠成像分离的形貌测量方法》，实现从单张伪重叠散斑图像到三维点云的完整处理流程。

仓库只保留论文最终采用的技术路线。Deblur、StrainNet、SGBM、ICGN、TV-L1 和 LK 光流等探索后未采用的方法不属于当前代码主线。

## 技术流程

```text
原始 blended clear blurred 三联图
  ↓ 按原始采集组划分训练集 验证集 测试集
256 × 256 对齐图像块
  ↓ 训练 DivideNet V3
单张伪重叠散斑图
  ↓ DivideNet V3 分离
清晰散斑图和模糊散斑图
  ↓ SIFT 稀疏匹配
Lowe 比率筛选和 RANSAC 几何筛选
  ↓ 线性插值
稠密水平和垂直位移场
  ↓ 中值偏差过滤和双边滤波
过滤后的位移场
  ↓ 去畸变和双目三角化
三维点云
  ↓ 统计离群点和深度范围过滤
最终 XYZ 点云
```

SIFT 负责产生稀疏匹配点，稠密位移场由后续线性插值得到。因此，本项目不将该步骤表述为“SIFT 直接生成稠密视差”。

## 项目结构

```text
Recoginition_Dic_Net/
├── main_reconstruction_pipeline.py     唯一命令行入口
├── requirements.txt                    Python 依赖
├── configs/
│   └── paper_calibration.example.json  标定和处理参数示例
├── pseudo_overlap/
│   ├── cli.py                          prepare train run 命令
│   ├── config.py                       配置读取和参数校验
│   ├── data.py                         三联图分组 划分和同步裁切
│   ├── model.py                        DivideNet V3 网络
│   ├── training.py                     模型训练
│   ├── separation.py                   大图分块推理和拼接
│   ├── matching.py                     SIFT RANSAC 和线性插值
│   ├── reconstruction.py               位移过滤 三角化和点云过滤
│   └── pipeline.py                     混合图到点云的完整编排
└── tests/
    ├── test_data.py                    数据划分与裁切测试
    └── test_reconstruction.py          三角化几何测试
```

## 环境要求

- Python 3.10
- PyTorch 2.0.1
- NumPy 1.26.4
- OpenCV 4.13.0
- SciPy 1.15.3
- Pillow 12.1.1

建议在独立虚拟环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

程序自动按 CUDA、Apple MPS、CPU 的顺序选择可用计算设备。

## 数据集和模型权重

论文附录提供了本项目的数据集和网络权重下载地址：

- 百度网盘：[数据集及网络权重](https://pan.baidu.com/s/1vp_XkQmVcg5ul4RGRNfglw?pwd=8888)
- 提取码：`8888`

下载后建议将原始三联图整理到 `data/raw/`，将 DivideNet V3 权重保存到 `checkpoints/`。论文链接提供的是项目实验材料；实际运行前仍需检查下载文件的目录结构、文件命名和权重文件名是否与下文命令一致。

## 数据要求

### 原始数据

`prepare` 命令读取一个只包含 BMP 图像的目录。文件按名称排序后，每连续三张必须依次对应：

1. 伪重叠图 `blended`
2. 清晰散斑图 `clear`
3. 模糊散斑图 `blurred`

三张图必须具有相同尺寸。原始文件总数必须是 3 的倍数。

例如：

```text
data/raw/
├── 001_blended.bmp
├── 001_clear.bmp
├── 001_blurred.bmp
├── 002_blended.bmp
├── 002_clear.bmp
└── 002_blurred.bmp
```

当前程序依赖文件名排序确定三张图的组内顺序。正式整理数据时，应确保排序结果确实为 `blended → clear → blurred`。

### 预处理数据

预处理后每个样本使用统一命名：

```text
{group_id}_{tile_id}_blended.png
{group_id}_{tile_id}_clear.png
{group_id}_{tile_id}_blurred.png
```

输出目录结构为：

```text
data/prepared/
├── train/
├── validation/
├── test/
└── split_manifest.json
```

数据先按原始采集组划分，再进行裁切。这样可以避免同一张原始大图的相邻图像块同时出现在训练集和验证集。

## 使用方法

项目只有三个按顺序使用的命令：`prepare`、`train` 和 `run`。

### 1 准备训练数据

```bash
python main_reconstruction_pipeline.py prepare \
  --input data/raw \
  --output data/prepared
```

固定处理参数：

- 训练集、验证集、测试集比例：7:2:1
- 随机种子：42
- 图像块尺寸：256 × 256
- 裁切步长：256

`split_manifest.json` 记录每个原始采集组所属的数据子集，可用于检查是否发生数据泄漏。

### 2 训练 DivideNet V3

```bash
python main_reconstruction_pipeline.py train \
  --dataset data/prepared \
  --checkpoints checkpoints
```

训练设置：

- 网络：DivideNet V3 双编码器双解码器
- Batch Size：128
- Epochs：200
- 优化器：Adam
- 初始学习率：`2e-4`
- 学习率调整：验证损失连续 10 个 epoch 未改善时减半
- 最佳权重：`checkpoints/best_model_v3.pth`

训练损失由像素 MSE、像素 L1、混合图重构约束和掩码互斥约束组成。原始代码中的 `predicted_clear + predicted_blurred ≈ blended` 在本项目中称为“简化的混合图重构约束”，不将其解释为严格的能量守恒关系。

### 3 执行完整重建

首先复制标定配置：

```bash
cp configs/paper_calibration.example.json configs/calibration.json
```

根据实际实验重新核对相机内参、畸变参数、旋转矩阵、平移向量和有效深度范围，然后运行：

```bash
python main_reconstruction_pipeline.py run \
  --input data/mixed.bmp \
  --checkpoint checkpoints/best_model_v3.pth \
  --config configs/calibration.json \
  --output outputs/reconstruction
```

参数说明：

- `--input`：待重建的单张伪重叠灰度图。
- `--checkpoint`：训练完成的 DivideNet V3 权重。
- `--config`：相机标定、SIFT 和过滤参数。
- `--output`：本次重建的输出目录。

该命令会连续执行图像分离、SIFT 匹配、RANSAC 筛选、线性插值、位移过滤、三角化和点云过滤，不需要单独运行中间脚本。

## 输出文件

```text
outputs/reconstruction/
├── clear.png             DivideNet V3 分离得到的清晰图
├── blurred.png           DivideNet V3 分离得到的模糊图
├── raw_disp_u.npy        插值后的原始水平位移场
├── raw_disp_v.npy        插值后的原始垂直位移场
├── filtered_disp_u.npy   过滤后的水平位移场
├── filtered_disp_v.npy   过滤后的垂直位移场
└── point_cloud.xyz       最终三维点云
```

`point_cloud.xyz` 每行依次记录一个点的 `X Y Z` 坐标。

## 配置文件

[paper_calibration.example.json](configs/paper_calibration.example.json) 包含三类参数：

- `calibration`：左右虚拟相机内参、畸变系数、相对旋转矩阵和平移向量。
- `matching`：SIFT 特征数量、Lowe 比率阈值、RANSAC 阈值和最少匹配点数量。
- `filtering`：中值窗口、突变阈值、双边滤波、ROI、深度范围和统计离群点参数。

示例参数来自论文对应版本，只用于说明配置结构。新的实验数据必须使用与该次拍摄对应的标定结果。

## 测试

运行单元测试：

```bash
python -m unittest discover -s tests -v
```

运行语法和导入检查：

```bash
python -m compileall -q pseudo_overlap tests main_reconstruction_pipeline.py
```

现有测试覆盖：

- 训练集、验证集和测试集的采集组互不重叠
- 三联图同步裁切后仍保持像素对齐
- 已知双目几何条件下的三角化深度正确

这里的 `tests/` 是代码单元测试，不是模型性能评估。当前仓库还没有批量读取 `test/` 数据集并计算 PSNR、SSIM 和 MSE 的评估命令。

## 当前限制

- 仓库不包含原始数据集和模型权重。
- 缺少论文测试集上的批量 PSNR、SSIM 和 MSE 评估脚本。
- 未提供可以直接复现论文最终点云的原始图像和完整实验标定文件。
- 标定参数、ROI 和有效深度范围与具体实验装置相关，不能直接套用到新的拍摄数据。

因此，代码可以验证处理逻辑和几何计算，但仅凭当前仓库不能宣称复现了论文报告的定量指标或最终实验结果。
