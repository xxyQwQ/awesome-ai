# Programming Assignment 1. Clustering

> TA Contact: Hangyu Ye, Borui Yang  
> E-mail: hangyu_ye@outlook.com, ybirua@sjtu.edu.cn

## 0. Introduction

In this project, you are given the feature vectors of 50000 papers, and each vector has 100 dimensions. These papers belong to 5 research areas: Data Mining, Knowledge Management, Operation Research, Information Retrieval and Natural Language Processing. We have already hidden the ground truths, and your task is to complete the given implementation of a k-means clustering algorithm in Python to divide these papers into 5 clusters.

### Environment

- This project requires Python 3 and Numpy.
- You can find the sample code in `main.py`.

### Dataset

The feature vectors are stored in a `features.csv` under `data/` directory. We have provided a utility function (`get_data()`) to load the features.

## 1. Task for the Project

### Algorithm Implementation

- You need to implement the K-Means algorithm and divide all papers into 5 categories.
- We have provided some utility functions and a scaffolding class. You are required to complete the code blocks marked with `# TODO`s.

## 2. Requirements

1. Do NOT use any existing clustering APIs. You are expected to implement the algorithm using only Numpy (or equivalent Python packages).
2. Do NOT directly copy others' code, whether from your classmates, previous years' solutions, GitHub, or generative AI.
3. Write your code between `TODO` and `End of TODO`. Any change in the remaining parts is not allowed.
4. Ensure that your code is runnable and reproducible. The TAs will run your code and verify the results. A penalty will be imposed on your final score if there is a significant gap between your reported results and our reproduced ones.
5. After performing the clustering, you should label each paper with their cluster indices.
6. The input and output code is provided in `main.py`. Do NOT modify any of them since the results will be scored automatically.
7. Don't worry about the order of labels. The given code will remap the labels to the ascending order of radius.

## 3. Submission & Grading

### Submission

Submit your assignment as a ZIP file, named as `[ChineseName]_[StudentID]_HW1.zip` (e.g., `张三_521030910000_HW1.zip`). Your submitted zip file should follow the structure specified as below

```txt
张三_521030910000_HW1.zip
├─ data/
│  ├─ features.csv
│  ├─ predictions.csv
├─ README.md
├─ main.py
```

- Your submission should include **(1) source code files** (`main.py`), **(2) all dataset files** (`data/features.csv`) and **(3) your clustering result** (`data/predictions.csv`).
- You can add, remove or rename files as you need. However, the entrance of your program must be named as `main.py`, and your results must be named as `predictions.csv` and stored under `data/` directory.
- You can append extra contents to this README file to provide any necessary description on how to run your project.
- Submit this project along with the other programming assignment and the written assignment, packed in a ZIP file.

### Grading

Your project will be graded in terms of **(1) Implementation correctness, (2) Accuracy of your submitted result**

1. **Implementation Correctness.** Your code should be runnable and should produce a CSV result in the correct format. The result produced by your code should be identical to your submitted one.
2. **Accuracy of the result.** Your result will be scored by **categorical accuracy**, given by
   
   $$CA = \frac{1}{N}\sum_{i=1}^N \mathbb{I}[p_i = y_i],$$
   
   where $N$ is the total number of samples, $p_i$ is your prediction (for the $i$-th sample) and $y_i$ is the ground truth label.
   - **NOTE.** Typically, clustering algorithms do not have "accuracy". We are just using the accuracy as a metric for the performance of your algorithm.

## *Hint*

- The radii of correct clustering results are in [2.0, 3.0].
- Feel free to ask the TAs for help if you have any questions about this assignment.
