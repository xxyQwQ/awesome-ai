# AI3610 类脑智能 脉冲神经网络作业

作业要求见 `Description.pdf`

## 文件结构

* Experiment：实验目录
  * checkpoint：运行结果目录，包含每次运行的训练日志、训练曲线、结果汇总、模型文件
  * dataset：数据集目录，由于文件过大，提交时已清空
  * model：模型定义目录，包括 CNN 和 SNN 的模型代码
  * utils：工具函数目录，包括日志和绘图等常用代码
  * run_cnn.py：CNN 的实验脚本
  * run_snn.py：SNN 的实验脚本
  
* Report：报告目录
  * Slides.pdf：讲解时所用的 PPT
  * Video.mp4：讲解视频


## 结果复现

运行如下命令可以复现 CNN 实验结果

```bash
python run_cnn.py -d gpu -b 256 -e 100
```

运行如下命令可以复现 SNN 实验结果

```bash
python run_snn.py -d gpu -b 256 -t 16 -e 100
```

**注：运行前需要手动补全数据集文件**

