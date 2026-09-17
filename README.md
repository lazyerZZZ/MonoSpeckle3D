# 伪重叠散斑分离与三维形貌重建

本仓库以毕业论文《基于深度学习伪重叠成像分离的形貌测量方法》的最终技术路线为主线。主流程不再使用旧版入口中的示例 StrainNet 参数或不存在的 `DivideNet_V4`，而是明确分为数据准备、DivideNet V3 分离、SIFT 匹配、视差插值和标定三角化五个阶段。

## 主流程

```text
单目伪重叠散斑图
  → 按原始采集组划分 train validation test
  → 同步裁切 256 × 256 的 blended clear blurred 三联图
  → DivideNet V3 分离 clear 与 blurred 图像
  → SIFT 稀疏特征匹配
  → Lowe 比率筛选与 RANSAC 几何筛选
  → 线性插值生成稠密 u v 位移场
  → 中值偏差过滤与双边滤波
  → 去畸变和双目三角化
  → 统计离群点与深度范围过滤
  → XYZ 点云
```

SIFT 只产生稀疏匹配点，稠密位移场来自后续插值。论文中的“物理一致性损失”在原代码里直接约束 `clear + blurred ≈ blended`，但三个张量的数值范围并不匹配；当前重构保留已训练 V3 权重的兼容性，不把该项重新解释为严格能量守恒。

## 目录

```text
pseudo_overlap/
  config.py          配置与标定参数校验
  data.py            三联图发现、按组划分和同步裁切
  separation.py      DivideNet V3 整图分块推理
  matching.py        SIFT、比率筛选、RANSAC 和插值
  reconstruction.py  位移过滤、去畸变和三角化
  pipeline.py        从分离图像到点云的编排
  cli.py             命令行入口
configs/
  paper_calibration.example.json
tests/
legacy scripts       根目录中原有的探索脚本，暂时保留用于结果追溯
```

## 安装

建议使用 Python 3.10，并在独立虚拟环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

仓库不包含论文数据集和模型权重。没有这些文件时，可以运行静态检查和测试，但不能复现论文 PSNR、SSIM 或最终点云。

## 数据准备

输入目录中的 BMP 文件应按每组三张的采集顺序排列：blended、clear、blurred。下面的命令先按原始采集组做 7:2:1 划分，再裁切，避免同一大图的相邻 patch 同时进入训练集和验证集。

```bash
python main_reconstruction_pipeline.py prepare \
  --input /path/to/raw/Camera1 \
  --output /path/to/prepared \
  --tile-size 256 \
  --stride 256
```

输出包括 `train/`、`validation/`、`test/` 和可复核的 `split_manifest.json`。

## 图像分离

`pseudo_overlap.separation.load_dividenet_v3` 加载论文最终采用的 V3 网络，`separate_image` 对任意尺寸灰度图补边、分块推理并恢复原尺寸。命令行用法：

```bash
python main_reconstruction_pipeline.py separate \
  --input /path/to/mixed.bmp \
  --checkpoint checkpoints/V3_Final/best_model_v3.pth \
  --output outputs/separated
```

也可以在 Python 中调用：

```python
import cv2

from pseudo_overlap.separation import load_dividenet_v3, save_grayscale, separate_image

model, device = load_dividenet_v3("checkpoints/V3_Final/best_model_v3.pth")
mixed = cv2.imread("mixed.bmp", cv2.IMREAD_GRAYSCALE)
clear, blurred = separate_image(mixed, model, device)
save_grayscale(clear, "outputs/clear.png")
save_grayscale(blurred, "outputs/blurred.png")
```

## 三维重建

先复制并核对 `configs/paper_calibration.example.json`。其中数值来自论文对应版本，但真实实验前仍应使用本次实验的标定结果复核旋转矩阵、平移方向、图像尺寸和深度范围。

```bash
python main_reconstruction_pipeline.py reconstruct \
  --left outputs/clear.png \
  --right outputs/blurred.png \
  --config configs/paper_calibration.example.json \
  --output outputs/reconstruction
```

输出：

- `raw_disp_u.npy` 和 `raw_disp_v.npy`
- `filtered_disp_u.npy` 和 `filtered_disp_v.npy`
- `point_cloud.xyz`

## 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q pseudo_overlap tests main_reconstruction_pipeline.py
```

测试覆盖按组划分不泄漏、三联图同步裁切，以及已知标定几何下的三角化深度。

## 旧脚本说明

根目录中的 `sift.py`、`2DTrans3D.py`、`DivideNet_cut.py`、`reconstruct_large_image.py` 等文件保留为历史实验记录。它们包含硬编码路径、固定图像尺寸或探索路线，不再作为推荐入口。新的主入口是 `main_reconstruction_pipeline.py`，实际实现位于 `pseudo_overlap/`。
