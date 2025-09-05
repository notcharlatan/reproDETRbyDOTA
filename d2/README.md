Detectron2的DETR封装接口
=======

我们为DETR提供了Detectron2封装接口，以便更好地将其集成到现有检测生态系统中。例如可以通过该接口便捷使用Detectron2提供的数据集或骨干网络。

当前封装仅支持目标框检测功能，实现方式尽可能贴近原始版本，我们已验证其结果一致性。需要注意的技术细节：
• 数据增强模块与DETR原始实现保持同步。为此我们修改了Detectron2的RandomCrop增强功能，因此需使用2020年6月24日或更新的master分支版本

• 为匹配DETR骨干网络初始化方式，我们使用torchvision训练的ImageNet预训练ResNet50权重。该网络默认使用的像素均值和标准差与多数Detectron2骨干网络不同，切换其他骨干网络时需特别注意。目前Detectron2暂未集成其他torchvision模型，未来可能更新

• 梯度裁剪模式采用"full_model"，这与Detectron2默认设置不同


使用说明

Detectron2安装请遵循[官方指南](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)。

模型评估

我们提供预训练模型转换脚本，可将主训练流程生成的模型转换为当前封装格式。转换ResNet50基准模型示例：

```
python converter.py --source_model https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_model converted_model.pth
```

转换后模型评估命令：
```
python train_net.py --eval-only --config configs/detr_256_6_6_torchvision.yaml  MODEL.WEIGHTS "converted_model.pth"
```


模型训练

单节点8卡训练基础模型：
```
python train_net.py --config configs/detr_256_6_6_torchvision.yaml --num-gpus 8
```

单节点8卡实例分割微调：
```
python train_net.py --config configs/detr_segm_256_6_6_torchvision.yaml --num-gpus 8 MODEL.DETR.FROZEN_WEIGHTS <模型路径>
```