# Face Recognition System

基于 PyTorch 的人脸识别系统，支持多种模型架构。

## 项目结构

```
facerecognize/
├── model.py              # ConvNet 模型
├── modelv2.py            # Facenet 模型
├── mobilefacenet.py      # MobileFaceNet 模型
├── vit.py                # Vision Transformer 模型
├── train.py              # 模型训练脚本
├── finetune.py           # MobileFaceNet微调脚本
├── compare.py            # 两张人脸图片比对
├── realtime_face_detection.py  # 实时人脸检测
├── lfw_eval*.py          # LFW 数据集评估脚本
├── export_model.py       # 模型导出脚本
├── facecrop.py           # 人脸裁剪工具
├── C++/                  # C++ 演示代码
│   ├── showcase.cpp
│   └── test_vit.cpp
└── face_detector/        # 人脸检测模型（opencv）
    ├── deploy.prototxt
    └── res10_300x300_ssd_iter_140000.caffemodel
```

## 模型架构

| 模型 | 特点 |
|------|------|
| ConvNet | 第一代卷积神经网络 |
| MobileFaceNet | 对照组模型 |
| ViT | Vision Transformer微调模型 |
| Facenet | 第二代卷积神经网络 |

## 环境依赖

- Python 3.10.11
- PyTorch
- OpenCV
- torchvision
- Numpy
- PIL

## 快速开始

### 安装依赖

```bash
pip install torch torchvision opencv-python numpy pillow
```

### 实时人脸检测

```bash
python realtime_face_detection.py
```

### 模型训练

```bash
python train.py
```

### LFW 评估

```bash
python lfw_eval.py
```

## 预训练模型

- `best_model.pth` - FaceNet 最佳模型
- `best_model_origin.pth` - 原始ConvNet模型
- `best_vit_model.pth` - ViT 最佳模型
- `mobilefacenet.ckpt` - MobileFaceNet 模型
- `mobilefacenet_finetuned.ckpt` - MobileFaceNet 微调模型
- `facenet_model.pt` - FaceNet 模型的TorchScript版本

## C++示例编译教程
- 安装cmake和一个C++编译器，我使用的是VS
- 从pytorch.org下载libtorch,解压到C++/libtorch下
- 下载opencv,解压到C++/opencv下
- 使用cmake编译

## 数据集

项目使用 CASIA FaceV5 数据集进行训练，评估采用 LFW (Labeled Faces in the Wild) 数据集。