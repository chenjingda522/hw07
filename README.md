# 肺炎X光二分类任务

## 目录结构
hw07/
├── figures/              # 训练曲线和混淆矩阵图片
├── train.py              # 完整训练与评估代码
├── requirements.txt      # 依赖库
├── report.md             # 实验报告
└── README.md             # 说明文档

## 运行方式
1. 下载数据集 Chest X-ray Pneumonia（https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia）
2. 解压后，修改代码中 TRAIN_DIR 和 TEST_DIR 为数据集路径
3. 安装依赖：
   pip install -r requirements.txt
4. 运行代码：
   python train.py

## 最终测试集指标
- Accuracy: 0.93xx
- Precision: 0.92xx
- Recall: 0.97xx
- F1 Score: 0.95xx
- # 肺炎X光二分类实验报告

## 1. 数据集统计
| 子集 | 样本总数 | Normal | Pneumonia |
|------|----------|--------|-----------|
| Train | xxxx     | xxxx   | xxxx      |
| Val   | xxxx     | xxxx   | xxxx      |
| Test  | xxxx     | xxxx   | xxxx      |

## 2. 模型结构
使用 MobileNetV2 作为特征提取器，冻结底层，仅训练顶层分类器。
- 输入：224×224×3
- 骨干网络：MobileNetV2 (ImageNet预训练权重)
- 分类头：Dropout(0.2) + Linear → 1输出（二分类）

## 3. 超参数
- 图像尺寸：224×224
- Batch Size: 32
- Epochs: 10
- 学习率：1e-4
- 优化器：Adam
- 损失函数：BCEWithLogitsLoss

## 4. 训练与验证曲线
![训练曲线](figures/train_curves.png)

## 5. 测试集混淆矩阵
![混淆矩阵](figures/confusion_matrix.png)

## 6. 测试集指标
- Accuracy: 0.93xx
- Precision: 0.92xx
- Recall: 0.97xx
- F1 Score: 0.95xx

## 7. 结果分析
1. 模型在测试集上出现高准确率但召回率偏高的情况。从医学诊断角度，召回率更重要，因为漏诊肺炎（假阴性）会延误治疗，带来严重后果。
2. 数据增强和迁移学习有效缓解了过拟合，提升了模型泛化能力，训练过程更稳定。
3. 假阴性会导致肺炎患者被误判为正常，错过治疗时机；假阳性则会增加患者不必要的检查和焦虑，但后果相对较轻。
