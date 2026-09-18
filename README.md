# 基于深度学习的伪重叠散斑三维形貌重建

本项目对应毕业论文《基于深度学习伪重叠成像分离的形貌测量方法》，实现从单张伪重叠散斑图像到三维点云的完整处理流程。

仓库只保留论文最终采用的技术路线。Deblur、StrainNet、SGBM、ICGN、TV-L1 和 LK 光流等探索后未采用的方法不属于当前代码主线。

## 项目定位

本项目起源于本科毕业设计，但实现的是一条完整、可拆分验证的光学测量与三维重建链路。因此，它不仅可用于毕业设计展示，也可作为以下工作的基础：

- 伪重叠成像、图像分离、数字图像相关和三维形貌测量研究
- DivideNet、特征匹配、位移场插值和点云处理算法对比
- 光学测量系统的实验原型和工程可行性验证
- 面向特定相机、光路、材料或工件的工业二次开发

本仓库目前属于研究原型，不等同于已经完成认证的工业产品。用于生产环境前，需要针对实际设备重新采集数据和标定，并完成精度、重复性、鲁棒性、运行速度、异常处理和安全性验证。

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

---

# Deep Learning Based 3D Morphology Reconstruction from Pseudo Overlapped Speckle Images

This project accompanies the undergraduate thesis *A Morphology Measurement Method Based on Deep Learning Separation of Pseudo Overlapped Images*. It implements the complete path from a single pseudo-overlapped speckle image to a three-dimensional point cloud.

The repository contains only the final technical route selected in the thesis. Deblur, StrainNet, SGBM, ICGN, TV-L1, and Lucas-Kanade optical flow were explored during development but are not part of the maintained pipeline.

## Project Scope

Although this project originated as undergraduate thesis code, it implements a complete optical measurement and 3D reconstruction chain whose stages can be tested independently. It can therefore serve as a foundation for:

- research on pseudo-overlapped imaging, image separation, digital image correlation, and 3D morphology measurement;
- comparisons of DivideNet, feature matching, displacement interpolation, and point-cloud processing methods;
- experimental prototypes and feasibility studies for optical measurement systems;
- application-specific industrial development for particular cameras, optical paths, materials, or workpieces.

The current repository is a research prototype, not a certified production system. Before industrial deployment, users must collect and calibrate data for the target equipment and validate accuracy, repeatability, robustness, runtime performance, error handling, and safety.

## Pipeline

```text
Raw blended clear blurred image triplets
  ↓ split by acquisition group into training validation and test sets
Aligned 256 × 256 image patches
  ↓ train DivideNet V3
One pseudo-overlapped speckle image
  ↓ DivideNet V3 separation
Clear and blurred speckle images
  ↓ sparse SIFT matching
Lowe ratio test and RANSAC geometric filtering
  ↓ linear interpolation
Dense horizontal and vertical displacement fields
  ↓ median-deviation and bilateral filtering
Filtered displacement fields
  ↓ undistortion and stereo triangulation
3D point cloud
  ↓ statistical outlier and depth-range filtering
Final XYZ point cloud
```

SIFT produces sparse feature matches. The dense displacement fields are generated by the following linear-interpolation stage; the project does not describe SIFT itself as a dense-disparity method.

## Repository Layout

```text
Recoginition_Dic_Net/
├── main_reconstruction_pipeline.py     single command-line entry point
├── requirements.txt                    Python dependencies
├── configs/
│   └── paper_calibration.example.json  calibration and processing example
├── pseudo_overlap/
│   ├── cli.py                          prepare train and run commands
│   ├── config.py                       configuration loading and validation
│   ├── data.py                         triplet grouping splitting and cropping
│   ├── model.py                        DivideNet V3
│   ├── training.py                     model training
│   ├── separation.py                   tiled full-image inference
│   ├── matching.py                     SIFT RANSAC and interpolation
│   ├── reconstruction.py               filtering triangulation and cleanup
│   └── pipeline.py                     complete image-to-cloud orchestration
└── tests/
    ├── test_data.py                    data split and crop tests
    └── test_reconstruction.py          triangulation geometry test
```

## Requirements

- Python 3.10
- PyTorch 2.0.1
- NumPy 1.26.4
- OpenCV 4.13.0
- SciPy 1.15.3
- Pillow 12.1.1

Installation in an isolated virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The program selects an available device in the following order: CUDA, Apple MPS, then CPU.

## Dataset and Model Weights

The thesis appendix provides the project dataset and network weights:

- Baidu Netdisk: [dataset and model weights](https://pan.baidu.com/s/1vp_XkQmVcg5ul4RGRNfglw?pwd=8888)
- Extraction code: `8888`

After downloading, place the raw image triplets under `data/raw/` and the DivideNet V3 checkpoint under `checkpoints/`. Check the actual archive layout, file names, and checkpoint name before using the commands below.

## Data Format

### Raw data

The `prepare` command reads a directory containing BMP images only. After lexical filename sorting, every three consecutive files must represent:

1. the pseudo-overlapped image, `blended`;
2. the clear speckle image, `clear`;
3. the blurred speckle image, `blurred`.

All three images must have identical dimensions, and the total number of BMP files must be divisible by three.

```text
data/raw/
├── 001_blended.bmp
├── 001_clear.bmp
├── 001_blurred.bmp
├── 002_blended.bmp
├── 002_clear.bmp
└── 002_blurred.bmp
```

The current implementation relies on filename sorting to determine the order within each triplet. Verify that the actual ordering is `blended → clear → blurred` before preprocessing.

### Prepared data

Each prepared sample uses the following names:

```text
{group_id}_{tile_id}_blended.png
{group_id}_{tile_id}_clear.png
{group_id}_{tile_id}_blurred.png
```

The generated directory layout is:

```text
data/prepared/
├── train/
├── validation/
├── test/
└── split_manifest.json
```

Acquisition groups are split before cropping, preventing neighboring patches from the same source image from leaking across training and validation subsets.

## Usage

The project exposes three commands that are used in order: `prepare`, `train`, and `run`.

### 1 Prepare the dataset

```bash
python main_reconstruction_pipeline.py prepare \
  --input data/raw \
  --output data/prepared
```

Fixed preprocessing settings:

- training, validation, and test ratio: 7:2:1;
- random seed: 42;
- patch size: 256 × 256 pixels;
- crop stride: 256 pixels.

The generated `split_manifest.json` records the subset assigned to every acquisition group and can be used to audit data leakage.

### 2 Train DivideNet V3

```bash
python main_reconstruction_pipeline.py train \
  --dataset data/prepared \
  --checkpoints checkpoints
```

Training configuration:

- network: dual-encoder, dual-decoder DivideNet V3;
- batch size: 128;
- epochs: 200;
- optimizer: Adam;
- initial learning rate: `2e-4`;
- scheduler: halve the learning rate after 10 epochs without validation improvement;
- best checkpoint: `checkpoints/best_model_v3.pth`.

The loss contains pixel MSE, pixel L1, a mixed-image reconstruction constraint, and a mask-based exclusion constraint. The relation `predicted_clear + predicted_blurred ≈ blended` is described as a simplified reconstruction constraint, not as strict conservation of energy.

### 3 Run the complete reconstruction pipeline

Copy the example calibration first:

```bash
cp configs/paper_calibration.example.json configs/calibration.json
```

Check the camera intrinsics, distortion coefficients, rotation matrix, translation vector, and valid depth range against the actual experiment, then run:

```bash
python main_reconstruction_pipeline.py run \
  --input data/mixed.bmp \
  --checkpoint checkpoints/best_model_v3.pth \
  --config configs/calibration.json \
  --output outputs/reconstruction
```

Arguments:

- `--input`: one pseudo-overlapped grayscale image;
- `--checkpoint`: trained DivideNet V3 weights;
- `--config`: camera calibration, SIFT, and filtering parameters;
- `--output`: output directory for this reconstruction.

This command runs separation, SIFT matching, RANSAC filtering, linear interpolation, displacement filtering, triangulation, and point-cloud cleanup in one sequence. No intermediate scripts are required.

## Outputs

```text
outputs/reconstruction/
├── clear.png             separated clear image
├── blurred.png           separated blurred image
├── raw_disp_u.npy        interpolated horizontal displacement field
├── raw_disp_v.npy        interpolated vertical displacement field
├── filtered_disp_u.npy   filtered horizontal displacement field
├── filtered_disp_v.npy   filtered vertical displacement field
└── point_cloud.xyz       final 3D point cloud
```

Each line in `point_cloud.xyz` stores one point as `X Y Z`.

## Configuration

[paper_calibration.example.json](configs/paper_calibration.example.json) contains three parameter groups:

- `calibration`: virtual left/right camera intrinsics, distortion coefficients, relative rotation, and translation;
- `matching`: SIFT feature count, Lowe ratio threshold, RANSAC threshold, and minimum match count;
- `filtering`: median window, jump threshold, bilateral filtering, ROI, depth range, and statistical-outlier parameters.

The example values correspond to the thesis version and primarily document the configuration structure. New experiments require calibration values obtained from their own image-acquisition setup.

## Tests

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Run syntax and import checks:

```bash
python -m compileall -q pseudo_overlap tests main_reconstruction_pipeline.py
```

The current tests verify that:

- acquisition groups do not overlap across training, validation, and test subsets;
- aligned triplets remain aligned after cropping;
- triangulation recovers the expected depth for known stereo geometry.

The `tests/` directory contains software unit tests, not a model-performance evaluation. The repository does not yet provide a command that evaluates the entire `test/` subset and reports PSNR, SSIM, and MSE.

## Current Limitations

- The repository itself does not contain the raw dataset or model weights; they are distributed through the link above.
- A batch evaluation command for PSNR, SSIM, and MSE on the thesis test subset is not yet included.
- The source images and complete calibration package required to reproduce the final thesis point cloud are not bundled in the repository.
- Calibration, ROI, and valid depth range depend on the physical setup and cannot be reused blindly for new acquisitions.

The code can be used to inspect and validate the processing logic and reconstruction geometry. The repository alone, however, is not sufficient evidence that the quantitative thesis results or final experimental point cloud have been reproduced.
