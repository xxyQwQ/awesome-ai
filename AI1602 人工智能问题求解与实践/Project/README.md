# 2022 聚类作业

一共有 4 道聚类题目，难度递增

需要聚类的数据我们将会以 `.npy`的格式打包给大家，附件中有一份 demo 代码，请查看本目录下的`load_data_example.py`

```python
import os
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

data = np.load(os.path.join('data', 'data_example.npy'))
label = np.load(os.path.join('data', 'label_example.npy'))

plt.scatter(data[:, 0], data[:, 1], c=label)
plt.axis('equal')
plt.savefig('data_example.jpg')

print([(data[i], label[i]) for i in range(10)])


```

## 题目类别 1:

给定**M** 个一维且`Size`为`（1）`的向量，将它们分为**2**类。

请将以下伪装代码补充完整

```python
import numpy as np
def cluster_task_1(data):
    """
    给定M个一维且Shape为(1,)的向量,将它们聚类成两类,一类的向量的ID为0,另一类的向量的ID为1。

    Args:
        data (np.ndarray): Shape: (M, )

    Returns:
        label (np.ndarray): Shape: (M, )
        	其中保存的是聚类的结果，请保证label与sample的index一致
    """
    label = np.zeros_like(data)

    # your code here

    return label

```

## 题目类别 2：

给定**M** 个一维且`Size`为`（S）`的向量，将它们分为**N**类, N 由自己尝试决定

```python
 def cluster_task_2(sample):
    """给定M个一维且Shape为(S)的向量,将它们聚类成N类,类别ID 从0 至 N-1。

    Args:
        sample (np.ndarray): Shape: (M, S)

    Returns:
        label (np.ndarray): Shape: (M, ),
        其中保存的是一维聚类的结果，请保证label与sample的index一致
    """
    label = np.zeros((sample.shape[0]))

    # your code here

    return label
```

# 题目要求:

请查看`Clustering.py`文件，里面一共有四个函数的原型。请填空。
完成后请运行

```
python Clustering.py
```

此代码会将你的结果保存至`output`文件夹内
并将你的`output`文件夹打包，以`学号+姓名.zip`的形式保存。
