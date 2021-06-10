# MDCNNforUV
Main code of MDCNN for UV and guidelines

主要依赖库:
tensorflow==2.1.0
keras==2.3.1
h5py==2.10.0

文件说明：
Backbone.py 为模型主要框架 
getPatches_offline.py 主要用于将整体大图划分为小块离线进行保存，并在本地进行预测。RAM足够且时间够多的情况下可修改代码，在线划分patches进行预测。 
imagePredict.py 用于对划分的patches进行逐个预测，并对应保存二分类结果，1为UV,0为non-UV
getMap.py 用于在总体大图上根据分类结果覆盖掩膜突出显示。
