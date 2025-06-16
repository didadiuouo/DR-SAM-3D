# Update record
2025-6-10

    DR-SAM3D 新增程序：多标签DICE指标迁移完成，验证单张图片，验证数据集多张图片。 
    DR-SAM3D 利用MedSAM3D的指标计算出训练最高达到print_dice: 0.5641294717788696, print_iou: 0.3928832411766052（result结果打印）
    MedSAM3D 训练结果展示在result文件夹内

2025-6-11

    MedSAM3D 训练结果更新在train_result文件夹内excel表中（medsam3d结果.xlsx）

2025-6-12

    DRsam3D和MedSAM3D训练过程结果更新在excel表内，出现过拟合的问题，不过训练与验证DICE达到0.54,IOU达到了0.37,仍需继续参数调整。

2025-6-13,14

    调整medsam3D学习率，训练验证DICE0.4 推论精度DICE0.18
    学习率越小，过拟合越慢
    1e-6，20次左右过拟合
