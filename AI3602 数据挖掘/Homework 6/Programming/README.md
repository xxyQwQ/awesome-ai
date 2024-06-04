# Programming Assignment 2 Phase 3 - Node2Vec

In this assignment, we implement the Node2Vec algorithm for node embedding, and use the learned embeddings for a Link Prediction task. We have provided the driver code and a scaffolding, and you are required to fill in some missing key components.

## 1. Introduction

### 1.1. Dataset

You are given an undirected network with 16,863 nodes and 46,116 edges. Your task is to learn node embeddings on the given graph using the Node2Vec algorithm and use the learned embeddings to perform link prediction on a given test dataset. The test set consists of 10,246 (src, dst) node pairs obtained from the original network. You need to predict the probability that a link exists between the two nodes using the learned node embeddings for each node pair.

The dataset files (under `p3_data/`) include

1. `p3.conf`, `p3_vertices.csv`, `p3_edges.csv`. These 3 files are the dataset. You need to convert them into a TuGraph database.
2. `p3_test.csv`. This is a csv file containing the test data, in the format of `id, src, dst`. For each (`src`, `dst`) pair, give the probability that a link between the two nodes.
3. `label_reference.csv`. This csv file contains 300 labels, which we release as a validation set to verify your algorithm. The file is in the format of `id, label`, where `id` corresponds to the `id` in `p3_test.csv`, and `label` is 1 if there is an edge, and 0 otherwise.

### 1.2. Environment

**TuGraph is required for this project.** You can refer to the documentations in HW2 Phase 1 for setting up the docker container for TuGraph database.

**Python 3.6 is required for this project.** You need to setup a Python 3.6 environment in the docker container. The provided docker image is shipped with Miniconda installed, so you can consider using `conda`. **Note that the Python version must be *exactly 3.6.x***.

Additionally, this project requires `torch`, `scikit-learn` and `tqdm`. Since Python 3.6 does not support the latest version of PyTorch, we will use PyTorch 1.10.

```sh
conda install pytorch==1.10.1 cpuonly -c pytorch
conda install scikit-learn
conda install tqdm
```

- Note that we are using the **cpu** version of PyTorch, because our docker does not support CUDA. Since the dataset and the model are relatively small, your CPU should be able to handle it.
- You are free to include other packages if you wish. However, **do NOT use existing Node2Vec modules**, including but not limited to `Word2Vec` from `gensim` and `Node2Vec` from `pytorch-geometric`.

### 1.3. Files

- `data_utils.py` contains a PyTorch `Dataset` for link prediction and a corresponding collator function.
- `loss.py` contains the implementation of Negative Sampling Loss. You need to complete its implementation.
- `walker.py` contains the implementation of a biased random walker. You need to complete its implementation.
- `metrics.py` contains a function for computing AUC scores.
- `model.py` contains a simple Node2Vec model and a Sigmoid classifier.
- `node2vec_trainer.py` contains the training process for the Node2Vec algorithm. The training process is already complete, but you can make adjustments if you need.
- `p3_main.py` is the entrance of the program. It loads the data, runs the Node2Vec algorithm to obtain node embeddings, uses the embeddings for link prediction and stores the results.

## 2. Task

This project consists of **4 progressive tasks**.

### 2.1. Converting the Provided Data into a TuGraph DB

> Related Files: `import_data.sh`.

1. **Convert the provided graph data into a TuGraph DB.**
   - We have provided the data in the format required by `lgraph_import`. Relevant files are available under the `p3_data/` directory.
   - You need to use `lgraph_import` to create a database under `/root/tugraph-db/build/outputs/`. The directory of the database should be `/root/tugraph-db/build/outputs/p3_db` and the name of the graph should be `default`.
   - Similar to phase 2, we have provided a shell script `import_data.sh` for your reference.

This task does not count toward your final score, but you have to complete it before you can proceed.

### 2.2. Implementing the Biased Random Walker (40%)

> Related Files: `walker.py`

The next step is to complete the biased random walk algorithm.

1. **Complete the `get_probs_biased()` method of `BiasedRandomWalker` in `walker.py`**.
   - This function computes the transition probability of the biased random walker.
   - Refer to the comments in the code for more instructions and hints.
2. **Complete the `walk()` method of `BiasedRandomWalker` in `walker.py`**.
   - This function performs a random walk and returns a trajectory.
   - Refer to the comments in the code for more instructions and hints.

In this task, you will need to use some of TuGraph's Python APIs. Relevant documentations have been provided in phase 1 and phase 2.

### 2.3. Implementing the Negative Sampling Loss (50%)

> Related Files: `loss.py`

1. **Complete the `forward()` method of `NegativeSamplingLoss` in `loss.py`**.
   - This module computes the negative sampling loss.

:information_source: **Note.** In practice, the loss sometimes goes to `NaN` or causes your model to degenerate, even if your implementation faithfully follow the mathematical formulas. We leave it up to you to figure out ways to mitigate such issues.

### 2.4. Hyper-parameter Tuning (10%)

> Related files: `p3_main.py`, `node2vec_trainer.py`, `run.sh`

We have provided a set of default hyper-parameters (e.g., optimizer, learning rate, length of random walk, parameter $p,q$ for biased random walk, etc.) and a very basic classifier (essentially a Sigmoid function) for performing link prediction. You can check the command-line arguments in `parse_args()` in `p3_main.py` for more details.

:warning: **The default hyper-parameters in the provided code are sub-optimal**. The expected AUC on the validation set using the default hyper-parameters is around 0.85.

**It is your task to experiment with different parameters and try to improve the performance**. Below we list a few potential improvements for your consideration.

1. **Try using different parameters for the random walker.** Consider changing the walk length, the window size, the parameters $p, q$ for the biased random walker. Especially, the default values for both $p$ and $q$ are set to $1.0$ (so no "bias" is applied by default at all!). Try tuning these parameters.
2. **Try configuring the number of negative samples.** Consider changing the number of negative samples used in your loss function. By default it is set to $1$.
3. **Try using different optimizers and parameters.** We have provided an `RMSProp` optimizer with learning rate `1e-2` (see `create_optimizer()` in `node2vec_trainer.py`). You can try changing the optimizer and/or adjusting the learning rates.
4. **Try using a more advanced classifier.** We have provided a rather simple classifier: dot-product followed by a Sigmoid function. You can try using other classifiers (such as using Logistic Regression from `scikit-learn`, or building an MLP with `torch`).
   - **NOTE.** If other hyper-parameters are set appropriately, you can still get a decent (>0.95) AUC even if you use this simple classifier.

It is suggested that you should update your `run.sh` and pass your hyper-parameters as command-line arguments, although other methods (e.g., directly modifying the code to change hyper-parameters) are also acceptable, so long as we are able to reproduce your results.

### Running your Algorithm

Similar to Phase 2, to run your algorithm, create a **symbolic link** of `p3_main.py` under `/root/tugraph-db/build/output/` and run `p3_main.py` under `/root/tugraph-db/build/output/`.

We have provided a script `run.sh` for ease of running your script.

```sh
# This should automatically create a symbolic link and run your p3_main.py
source run.sh
# or bash run.sh
```

## 3. Submission & Requirements

### 3.1. Submission

The provided code will automatically produce a `p3_prediction.csv` under `./p3_data`. The prediction csv file has two columns: `id` and `score`, where `id` corresponds to the `id` in `p3_test.csv` and score is the probability that an edge exists, rounded to 4 decimal places.

Submit your results and all source code as a ZIP file, named as `[StudentID]_[ChineseName]_HW2P3.zip` (e.g., `521030910000_张三_HW2P3.zip`). The following files are REQUIRED in your submission.

```txt
523030910000_张三_HW2P3/
├─ p3_data/
│  ├─ p3_prediction.csv   # predictions of your algorithm
├─ *.py                   # ALL related source code files
├─ run.sh                 # The script for running the project
```

The TAs will run you code by

```sh
bash run.sh                # For verifying your link prediction results
```

**Please ensure:** (1) Your submission is self-contained, i.e., it contains all necessary Python files for your project to run. (2) Your `run.sh` contains the required commandline arguments (if any) for the TAs to reproduce your results.

### 3.2. Requirements

1. **Do NOT use any existing Node2Vec implementations or link prediction APIs.** You can use any additional Python packages, but no direct uses of Node2Vec APIs are allowed.
2. **Do NOT use Graph Neural Networks (GNNs).** The focus of this assignment is the Node2Vec algorithm. GNN is a complete overkill for this simple task and it diverges from our goal.
3. **You are REQUIRED to use TuGraph APIs.** Do NOT read data directly from csv files. Do NOT use other graph data structures, whether from `networkx` or built from scratch.
4. **Do NOT change functions marked with `XXX: Do NOT change...`** Functions with this mark will be used for automated grading. Contact the TAs in advance if you do have the need to modify these functions.
5. **Other than the functions stated above, you can make any changes you wish.** You can also add new files or remove unused files as you need. However, please ensure your submission is self-contained.
6. **Do NOT directly copy others' code,** whether from your classmates, previous years' solutions, GitHub or generative AIs. If you refer to other's implementations, you need to explicitly acknowledge the source.
7. **Ensure that your code is runnable and reproducible.** We will run your code and verify the results. A penalty will be imposed on your final score if there is a significant gap between your reported results and our reproduced ones.

### 3.3. Other Notes

1. **You are encouraged (but not required) to follow good programming habits**, e.g., use meaningful variable names, write comments where necessary, avoid extremely long lines, etc. This, however, will not affect your score.

## 4. Grading

#### 4.1. Biased Random Walker (40%)

We have prepared a few test cases (not released) to test your implementation of `get_probs_biased()`. Your score will be given according to the results of our test cases. Full 40% if your code passes all our test cases, and deducted proportionally if some cases fail.

#### 4.2. Node2Vec Link Prediction (60%)

We will score your implementation based on the AUC of your algorithm on the full test set. You need to tune your hyper-parameters to get the optimal performance and full 100% score.

|                 Metrics                  | Score (Task 2.3 + Task 2.4) |
| :--------------------------------------: | :-------------------------: |
| Code runs without error. AUC above 0.93. |         50Pt + 10Pt         |
| Code runs without error. AUC above 0.85. |       50Pt + (0~10)Pt       |
| Code runs without error. AUC above 0.75. |       (40~50)Pt + 0Pt       |
| Code runs without error. AUC above 0.65. |       (20~40)Pt + 0Pt       |
|               Other cases.               |      Manually scored.       |

## References

1. Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.
2. Perozzi, Bryan, Rami Al-Rfou, and Steven Skiena. "Deepwalk: Online learning of social representations." Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.
3. [PyTorch Geometric | Node2Vec](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.Node2Vec.html).
