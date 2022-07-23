数据集：链接：https://pan.baidu.com/s/1PJQKffM8oHK_uZVgg07PJw  提取码：vw6v 
将数据解压至h5文件夹下
数据集是原NTU RGB+D数据集，经过处理转化为h5格式的文件，使用Cross Subject分割训练和测试数据集

论文原文：《Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks》
论文地址： https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_Modeling_Temporal_Dynamics_CVPR_2017_paper.html

未对数据进行论文所述的3D transformation 数据预处理

Temporal RNN使用的是Stacked RNN
Spatial RNN使用的是Chain Sequence遍历和Stacked RNN