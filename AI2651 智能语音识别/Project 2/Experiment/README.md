# 第二次大作业：LVCSR系统搭建

## 文件结构

* `conf/`：配置文件目录
* `local/`：本地脚本目录
* `cmd.sh`：设置提交命令
* `label.py`：生成标注文本
* `path.sh`：初始化环境变量
* `run.sh`：控制训练流程
* `submit.slurm`：提交超算作业

## 修改清单

* 在`cmd.sh`中，将所有`queue.pl`修改为`run.pl`，并指定内存为`--mem 16G`
    ```bash
    export train_cmd="run.pl --mem 16G"
    export decode_cmd="run.pl --mem 16G"
    export mkgraph_cmd="run.pl --mem 16G"
    ```
    由于提交作业数量严重受限，我们避免使用`slurm.pl`，而选择申请资源后直接运行
* 在`path.sh`中，对Kaldi根目录进行修改
    ```bash
    export KALDI_ROOT=/lustre/home/acct-stu/stu1718/kaldi
    ```
    同时另外添加一行
    ```bash
    . /lustre/home/acct-stu/stu1758/tools/env.sh`
    ```
    该目录包含所有自行安装的工具，包括`train_lm.sh`脚本和`SoX`音频处理工具
* 在`run.sh`中，修改数据集路径
    ```bash
    data=/lustre/home/acct-stu/stu1758/aishell_10h
    ```
    并注释掉下载步骤
    ```bash
    # local/download_and_untar.sh $data $data_url data_aishell || exit 1;
    # local/download_and_untar.sh $data $data_url resource_aishell || exit 1;
    ```
    实验所需的数据集位于`/lustre/home/acct-stu/stu1758/aishell_10h`目录下
* 在`local/nnet3/run_tdnn.sh`和`local/chain/run_tdnn.sh`中，修改训练进程数量
    ```bash
    num_jobs_initial=1
    num_jobs_final=1
    ```
    由于仅有单张GPU，进程超限会导致训练失败
* 在研究语料数量对模型性能的影响时，使用`train_large.txt`替换原有标注文件
    ```bash
    $ rm data/local/train/text
    $ mv train_large.txt data/local/train/text
    ```
    此修改在`run.sh`完成数据集准备后进行
* 编写脚本文件`submit.slurm`，用于提交超算作业
    ```bash
    #!/bin/bash
    #SBATCH --job-name=kaldi
    #SBATCH --partition=dgx2
    #SBATCH -n 1
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task=6
    #SBATCH --gres=gpu:1
    #SBATCH --output=%j.out
    #SBATCH --error=%j.err
    sh ./run.sh
    ```
    该脚本在`dgx2`分区上申请单张GPU和六核CPU，运行`run.sh`并保存输出日志
* 编写脚本文件`label.py`，用于生成标注文件
    ```python
    # convert the result of kaldi into the submission file
    score_path = "exp/nnet3/tdnn_sp/decode_test/scoring_kaldi/best_cer"
    with open(score_path, "r") as score:
        parameter = score.readline().strip().split('/')[-1]
    weight, penalty = parameter.split('_')[1:]
    source_path = "exp/nnet3/tdnn_sp/decode_test/scoring_kaldi/penalty_{}/{}.txt".format(penalty, weight)
    target_path = "label.csv"
    with open(source_path, "r") as source, open(target_path, "w") as target:
        target.write('uttid,result\n')
        for line in source:
            uttid, result = line.strip().split(' ', 1)
            target.write('{},{}\n'.format(uttid, result.replace(' ', '')))
    ```
    生成的标注文件保存在`label.csv`中，可以直接在Kaggle提交

## 使用方法

* 正确准备数据集并安装工具包，在`recipe`目录下直接执行
    ```bash
    $ sh ./run.sh
    ```
    即可完成各个模型的训练和解码，在日志中会输出最终的评分结果，在超算环境下应当使用
    ```bash
    $ sbatch submit.slurm
    ```
    代替上述命令，以提交超算作业的方式运行脚本
* 在测试集上进行推理，应当首先完成`run.sh`的全部流程，然后执行
    ```bash
    $ python3 label.py
    ```
    得到所需标注文件`label.csv`，完成推理
* 本实验中，`nnet3`模型的评分最高，我们将其选为预期的最终模型，所在路径应为
    ```bash
    exp/nnet3/tdnn_sp/final.mdl
    ```
    在$\pi 2.0$集群上，训练完成的最终模型位于
    ```bash
    /lustre/home/acct-stu/stu1758/aishell_large/s5/exp/nnet3/tdnn_sp/final.mdl
    ```
    解码过程中该模型需要与目录下其他文件配合使用

* 在$\pi 2.0$集群上，`recipe`所在对应目录应为

  ```bash
  /lustre/home/acct-stu/stu1758/aishell_large/s5
  ```