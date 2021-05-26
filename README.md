
* result/
> 一些图像及可视化结果示意

* pytorch_metric_learning/
> 度量学习的一些工具，在本项目中主要用于计算度量学习方法下的识别准确率
> 可参考此工具进行一些其他度量学习工作的拓展

* pretrainedmodels/
> 在imagenet上一些常见模型的预训练参数的获得
> 在本项目中用这个包中的预训练模型进行cam及distcam的生成和评估

* distcam_eval.py
> 用pretrain得到的backbone进行embedding
> 用保存的prototype评估metric分类准确率

* faiss_demo.py
> faiss库用法示例

* distcam_generate.py 
> 以1shot任务为例，对单张图像进行dist_cam生成
> 多shot任务可在此基础上进行扩展
> imagenet上的度量学习任务，可先用make_prototype.py文件生成该backbone下1000个类的prototype，在distcam_generate上进行扩展，生成distcam可视化结果

* make_prototype.py.py 
(针对imagenet2012-trainset数据量太大的问题，对每个类别的样本单独embed然后计算prototype，保存在pickle中）
> 用pretrain得到的backbone进行embedding
> 保存计算出来的prototype

* dataset 
> imagenet验证集图像路径: /home/shenyq/data/ILSVRC2012_val/images/
> imagenet验证集bbox标签xml文件路径: /home/shenyq/data/ILSVRC2012_val/bbox/
> imagenet验证集class标签txt文件: /home/shenyq/data/ILSVRC2012_val/val.txt
> imagenet训练集图像路径: /home/shenyq/data/ILSVRC2012_train/
> imagenet训练集csv标签路径: /home/shenyq/data/ILSVRC2012_train/csv1000/


