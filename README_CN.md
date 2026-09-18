# 基于深度学习的伪重叠散斑三维形貌重建

[![English](https://img.shields.io/badge/Language-English-lightgrey)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-blue)](README_CN.md)

本项目对应毕业论文《基于深度学习伪重叠成像分离的形貌测量方法》，实现从单张伪重叠散斑图像到三维点云的完整处理流程。

仓库只保留论文最终采用的技术路线。Deblur、StrainNet、SGBM、ICGN、TV-L1 和 LK 光流等探索后未采用的方法不属于当前代码主线。

原始毕设代码、探索路线和历史脚本完整保存在 [`backup/main-before-paper-refactor-20260918`](https://github.com/lazyerZZZ/Recoginition_Dic_Net/tree/backup/main-before-paper-refactor-20260918) 分支。如需追溯早期实现或与重构后的主线对照，请查看该分支。

## 项目定位

本项目起源于本科毕业设计，但实现的是一条完整、可拆分验证的光学测量与三维重建链路，可作为以下工作的基础：

- 伪重叠成像、图像分离、数字图像相关和三维形貌测量研究
- 图像分离、特征匹配、位移场插值和点云算法对比
- 光学测量系统的实验原型和工程可行性验证
- 面向特定相机、光路、材料或工件的工业二次开发

当前仓库属于研究原型，不等同于已经认证的工业产品。生产部署前需要针对实际设备重新采集数据和标定，并验证精度、重复性、鲁棒性、运行速度、异常处理和安全性。

## 技术流程

```text
原始 blended clear blurred 三联图
  → 按采集组划分并裁成对齐的 256 × 256 图像块
  → 训练 DivideNet V3
单张伪重叠散斑图
  → DivideNet V3 分离
清晰图和模糊图
  → SIFT 稀疏匹配
Lowe 比率筛选和 RANSAC 几何筛选
  → 线性插值得到稠密位移场
  → 中值偏差和双边滤波
  → 去畸变和双目三角化
  → 统计离群点和深度过滤
最终 XYZ 点云
```

SIFT 只产生稀疏匹配点，稠密位移场由后续线性插值得到。

## 项目结构

```text
Recoginition_Dic_Net/
├── main_reconstruction_pipeline.py     唯一命令行入口
├── requirements.txt                    Python 依赖
├── configs/
│   └── paper_calibration.example.json  标定和处理参数示例
├── pseudo_overlap/
│   ├── cli.py                          prepare train run 命令
│   ├── config.py                       配置读取和校验
│   ├── data.py                         三联图划分和同步裁切
│   ├── model.py                        DivideNet V3
│   ├── training.py                     模型训练
│   ├── separation.py                   大图分块推理
│   ├── matching.py                     SIFT RANSAC 和插值
│   ├── reconstruction.py               过滤 三角化和点云清理
│   └── pipeline.py                     混合图到点云的完整编排
└── tests/
    ├── test_data.py
    └── test_reconstruction.py
```

## 环境安装

要求 Python 3.10、PyTorch 2.0.1、NumPy 1.26.4、OpenCV 4.13.0、SciPy 1.15.3 和 Pillow 12.1.1。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

程序按 CUDA、Apple MPS、CPU 的顺序自动选择计算设备。

## 数据集和模型权重

论文附录提供了项目数据集和网络权重：

- 百度网盘：[数据集及网络权重](https://pan.baidu.com/s/1vp_XkQmVcg5ul4RGRNfglw?pwd=8888)
- 提取码：`8888`

建议将原始三联图放在 `data/raw/`，将 DivideNet V3 权重放在 `checkpoints/`。运行前需要核对压缩包的实际目录、文件名和权重名称。

## 数据格式

`prepare` 读取只包含 BMP 图像的目录。文件按名称排序后，每连续三张必须依次为 `blended`、`clear`、`blurred`，尺寸必须相同，文件总数必须是 3 的倍数。

```text
data/raw/
├── 001_blended.bmp
├── 001_clear.bmp
├── 001_blurred.bmp
├── 002_blended.bmp
├── 002_clear.bmp
└── 002_blurred.bmp
```

预处理样本命名为：

```text
{group_id}_{tile_id}_blended.png
{group_id}_{tile_id}_clear.png
{group_id}_{tile_id}_blurred.png
```

输出目录为：

```text
data/prepared/
├── train/
├── validation/
├── test/
└── split_manifest.json
```

程序先按原始采集组划分，再进行裁切，避免同一大图的相邻 patch 泄漏到不同子集。

## 使用方法

三个命令按顺序使用：`prepare`、`train`、`run`。

### 1 准备数据

```bash
python main_reconstruction_pipeline.py prepare \
  --input data/raw \
  --output data/prepared
```

固定采用 7:2:1 划分、随机种子 42、256 × 256 图像块和 256 像素步长。`split_manifest.json` 记录每个采集组所属的子集。

### 2 训练 DivideNet V3

```bash
python main_reconstruction_pipeline.py train \
  --dataset data/prepared \
  --checkpoints checkpoints
```

训练固定为 200 epochs、Batch Size 128、Adam 和初始学习率 `2e-4`。验证损失连续 10 个 epoch 未改善时学习率减半，最佳权重保存为 `checkpoints/best_model_v3.pth`。

损失由像素 MSE、像素 L1、混合图重构约束和掩码互斥约束组成。`predicted_clear + predicted_blurred ≈ blended` 仅称为简化的重构约束，不解释为严格能量守恒。

### 3 执行完整重建

```bash
cp configs/paper_calibration.example.json configs/calibration.json
```

核对实际实验的相机内参、畸变、相对位姿和有效深度范围，然后运行：

```bash
python main_reconstruction_pipeline.py run \
  --input data/mixed.bmp \
  --checkpoint checkpoints/best_model_v3.pth \
  --config configs/calibration.json \
  --output outputs/reconstruction
```

- `--input`：单张伪重叠灰度图
- `--checkpoint`：DivideNet V3 权重
- `--config`：标定、SIFT 和过滤参数
- `--output`：输出目录

该命令连续执行分离、SIFT、RANSAC、插值、位移过滤、三角化和点云过滤。

## 输出文件

```text
outputs/reconstruction/
├── clear.png
├── blurred.png
├── raw_disp_u.npy
├── raw_disp_v.npy
├── filtered_disp_u.npy
├── filtered_disp_v.npy
└── point_cloud.xyz
```

`point_cloud.xyz` 每行记录一个点的 `X Y Z` 坐标。

## 配置

[paper_calibration.example.json](configs/paper_calibration.example.json) 包含：

- `calibration`：相机内参、畸变、相对旋转和平移
- `matching`：SIFT、Lowe 比率、RANSAC 和最少匹配点参数
- `filtering`：中值、双边、ROI、深度范围和统计离群点参数

示例数值对应论文版本。新的采集任务必须使用自身实验装置的标定结果。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q pseudo_overlap tests main_reconstruction_pipeline.py
```

现有测试覆盖采集组隔离、三联图对齐裁切和已知双目几何下的三角化。`tests/` 是代码单元测试，不是模型性能评估；当前尚未提供整套测试集的 PSNR、SSIM 和 MSE 批量评估命令。

## 当前限制

- 仓库本身不包含数据集和权重，相关文件通过上方链接提供
- 尚未实现批量 PSNR、SSIM 和 MSE 评估
- 未随仓库提供复现论文最终点云所需的全部原图和标定文件
- 标定参数、ROI 和有效深度范围依赖具体实验装置

代码可用于检查和验证处理逻辑与重建几何，但仅凭当前仓库不能证明已经复现论文定量指标或最终实验点云。
