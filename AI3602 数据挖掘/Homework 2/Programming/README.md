# Programming Assignment 2. Dimensionality Reduction
> TA Contact: Hangyu Ye, Borui Yang  
> E-mail: hangyu_ye@outlook.com, ybirua@sjtu.edu.cn


## Introduction
In this project, you are required to implement the dimensionality-reduction algorithm for the training procedure of Deep Neural Network with SVD. This project is inspired by The Low-dimensional Trajectory Hypothesis [1]:
> For a neural network with $n$ parameters, the parameters’ trajectory over training can be approximately covered by a $d$-dimensional space with $d \ll n$.

That is, a neural network could be trained in a low-dimensional subspace by projecting the parameters into this low-dimensional space. This could be useful in specific scenarios, e.g., the parameters (or gradient updates) of a neural network need to be transmitted over channels with limited bandwidths.

In this project, you are required to implement a dimensionality reduction algorithm (namely, PCA) as part of this low-dimensional training scheme.

You are given the training trajectory (training parameters sequences for the first 30 epochs) of `Resnet 8` trained on a subset (10%) of CIFAR-10's training set. You are required to calculate the dimensionality-reduction matrix (or project matrix) based on the given trajectory. This project matrix will be used to train `Resnet 8` from **scratch** on the **whole** training set of CIFAR-10.

Note that you **only need to implement the dimensionality reduction algorithm** (see [below](#task-for-the-project)). The remaining training/validation loops of the neural network have been included in the given code and you do NOT need to implement this part.

### Environment

```
python >= 3.8
pytorch >= 1.4
```

### Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) will be used for this assignment. The provided code should automatically download the dataset. However, if it fails, try manually downloading the dataset from <https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz> and unzip the dataset into `data/`.

## Task for the Project

### Algorithm Implementation
- Implement the PCA class in `main.py`
- The `PCA` class initializes a few variables in its constructor. You are required to implement the `fit()` method and assign the correct values to the following variables.
```
# in PCA.__init__()
self.components_ = None
self.explained_variance_ = None
self.explained_variance_ratio_ = None
```

The explained variance ($n$ is the number of samples): 
The amount of variance explained by each of the selected components.
$$
    \frac{S^2}{n - 1}
$$

The explained variance ratio: Percentage of variance explained by each of the selected components.



### Requirements
- Do NOT use any existing APIs other than `numpy`
- `np.linalg.svd` is suggested to use, but be careful about the `full_matrices` arg. 
  - > **full_matrices (bool, optional)**
    If True (default), u and vh have the shapes (..., M, M) and (..., N, N), respectively. Otherwise, the shapes are (..., M, K) and (..., K, N), respectively, where K = min(M, N).
  - Try it yourself setting the argument of `full_matrices` to True and False.
- Write your code between `TODO` and `End of TODO`. Any change in the remaining parts is not allowed.
- Do NOT use additional training trajectory. 
- CUDA or cpu only is both supported. Training with cpu is slower but should finish in acceptable time. If correctly implemented, the algorithm will converge in $3$ epochs.
- Do NOT directly copy others' code, whether from your classmates, previous years' solutions, GitHub, or generative AI.
- Ensure that your code is runnable and reproducible. The TAs will run your code and verify the results. A penalty will be imposed on your final score if there is a significant gap between your reported results or figures and our reproduced ones.
- The input and output code is provided in `main.py`. Do NOT modify any of them since the results will be scored automatically.



### Submission
Submit your assignment as a ZIP file, named as `[ChineseName]_[StudentID]_HW1.zip` (e.g., `张三_521030910000_HW1.zip`). Your submitted zip file should follow the structure specified as below

```
张三_521030910000_HW1.zip
├─ pca_ratio.pdf
├─ main.py
├─ README.md
```
- Your submission should include **(1) source code file of main.py** **(2)generated figure** (drawing code is given in `main.py`)
- You can append extra contents to this README file to provide any necessary description on how to run your project.

### Grading
- **Implementation Correctness.** Your code should be runnable and should produce a pdf `pca_ratio.pdf`. 
- **Accuracy** Your results will be scored by top-1 accuracy on CIFAR-10's test set. Full scores if your algorithm is able to achieve 62% and above accuracy consistently. 
  


## Hint
- The test accuracy of `Resnet 8` with model paramters `29.pt` in the given training procedure is about $0.59$
- NO hidden test set and NO ranking in this project.
- Make sure your data matrix is **zero-centered** before sending it to `np.linalg.svd` (Please refer to page 45 of the slide).
- You are not allowed to use `sklearn` in this project, but you can use `sklearn.decomposition.PCA` locally for reference (not required to be strictly the same). 
- Feel free to ask the TAs for help if you have any questions about this assignment.
## References
[1] Li T, Tan L, Huang Z, et al. Low dimensional trajectory hypothesis is true: Dnns can be trained in tiny subspaces[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022, 45(3): 3411-3420.