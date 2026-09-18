# 伪重叠散斑三维形貌重建

这个仓库只保留毕业论文最终采用的一条主线：

```text
原始三联图
  → 按采集组划分 7:2:1 数据集并裁成 256 × 256
  → 训练 DivideNet V3
  → 将一张伪重叠图分离为清晰图和模糊图
  → SIFT 稀疏匹配
  → Lowe 比率筛选和 RANSAC 几何筛选
  → 线性插值得到稠密位移场
  → 中值偏差过滤和双边滤波
  → 标定去畸变和三角化
  → 统计离群点与深度过滤
  → XYZ 点云
```

仓库中不再包含 Deblur、StrainNet、SGBM、ICGN、TV-L1、LK 光流等论文探索后未选用的路线。

## 代码结构

```text
main_reconstruction_pipeline.py   唯一入口
configs/
  paper_calibration.example.json  标定和主线参数
pseudo_overlap/
  data.py                         数据分组和同步裁切
  model.py                        DivideNet V3
  training.py                     V3 训练
  separation.py                   整图分块分离
  matching.py                     SIFT RANSAC 线性插值
  reconstruction.py               视差过滤 三角化 点云过滤
  pipeline.py                     从混合图到点云的完整编排
  cli.py                          三个顺序执行阶段
tests/                            数据与几何测试
```

## 安装

```bash
python -m pip install -r requirements.txt
```

## 第一步 准备数据

输入 BMP 文件按每组三张的顺序排列：blended、clear、blurred。程序先按原始采集组划分，再裁切，避免同一张大图的相邻 patch 泄漏到不同数据集。

```bash
python main_reconstruction_pipeline.py prepare \
  --input /path/to/raw/Camera1 \
  --output data/prepared
```

固定设置：

- 训练集、验证集、测试集比例为 7:2:1
- 随机种子为 42
- patch 和步长均为 256 像素

## 第二步 训练 DivideNet V3

```bash
python main_reconstruction_pipeline.py train \
  --dataset data/prepared \
  --checkpoints checkpoints
```

训练固定为论文主线设置：Batch Size 128、200 epochs、初始学习率 `2e-4`，并根据验证损失降低学习率。最佳权重保存为 `checkpoints/best_model_v3.pth`。

原代码中的 `clear + blurred ≈ blended` 在这里仅称为简化的混合图重构约束，不把它表述为严格的能量守恒。

## 第三步 从混合图重建点云

先复制并核对标定配置：

```bash
cp configs/paper_calibration.example.json configs/calibration.json
```

然后执行完整主线：

```bash
python main_reconstruction_pipeline.py run \
  --input data/mixed.bmp \
  --checkpoint checkpoints/best_model_v3.pth \
  --config configs/calibration.json \
  --output outputs/reconstruction
```

一次运行依次完成图像分离、SIFT 匹配、插值、视差过滤、三角化和点云过滤，输出：

```text
clear.png
blurred.png
raw_disp_u.npy
raw_disp_v.npy
filtered_disp_u.npy
filtered_disp_v.npy
point_cloud.xyz
```

## 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q pseudo_overlap tests main_reconstruction_pipeline.py
```

仓库不包含原始数据和模型权重，因此仅凭仓库无法复现论文中的 PSNR、SSIM 和最终实验点云。
