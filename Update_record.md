# Update record
2025-6-20

    FastSAM3D 跑通，
    MemorizingSAM3D跑通，出了结果--类似MedSAM3D的过拟合,展示在结果excel里。


2025-6-21

    验证了FastSAM3D，Medical-SAM-Adapter的结果，记录在结果xlsl里。


2025-6-23

    修改MedLSAM_SPL_Inference的读取数据和推论过程。MedLSAM是对比算法中唯一一个多标签分类的网络，其按标签将数据归为若干组，进行每个标签组的结果预测。我们的数据特点是部分标签不存在，
    导致MedLSAM无法完美运行，修改思路：遍历所有数据标签分布情况，生成标签对应map，利用medSAM读取标签map中信息进行标签组结果预测。

2025-6-25

    3D-unet探索分割多标签的方法，修改MedSAM3d，训练MedSAM3d
    ![image](https://github.com/user-attachments/assets/6e524e34-e759-4805-80f1-952bbe4753ec)




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

2025-6-16 

    分epoch验证过拟合情况，总结在“medsam3d结果.xlsx”里。最终调试的最高推论结果0.1888。
    
    <img src="https://github.com/user-attachments/assets/7ec6b86e-cf6e-4d26-b8f5-265f71554e40" width="150px"> <img src="https://github.com/user-attachments/assets/7045d6da-d198-4da8-89ff-d401783ad086" width="410px">

2025-6-17

    对比算法 3DSAM-adapter
    第一次训练，训练验证和测试结果如下，测试结果反而最高，程序待细查。
    Train metrics: 0.27631396
    Val metrics: 0.35554218
    Val metrics best: 0.3232891
    Case7.nii.gz - Dice 0.653682 | NSD 0.675991
    Case5.nii.gz - Dice 0.639147 | NSD 0.675235

2025-6-18

    对比算法微调测试 MedLSAM，Memorizing SAM，FastSAM3D
    现有SAM算法多是基于单标签分割，其二分类的问题和我们的多标签分割的有些许差异，拿到程序后，需要修改标签读取方式以及评价指标计算方式。稳步进行。。。

2025-6-19

    MedLSAM权重训练完成，MedLAM权重训练完成。{'valid_loss': 34.28057, 'train_loss': 25.599924}，等待推论验证。Memorizing SAM训练进行中。


