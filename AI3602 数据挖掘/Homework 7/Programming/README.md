# Programming  3. Streaming Algorithm

> TA Contact: Hangyu Ye, Borui Yang  
> E-mail: hangyu_ye@outlook.com, ybirua@sjtu.edu.cn

## Introduction

In this project, you are required to implement the following streaming algorithm for different tasks:

- DGIM
- Bloom Filter
- Flajolet-Martin

## Task for this project

1. DGIM
   - DGIM is an efficient algorithm in processing large streams. When storing the flowing binary stream is infeasible, DGIM can estimate the number of 1-bits in the window. In this coding, you're given the `stream_data_dgim.txt` (binary stream), and you need to implement the DGIM algorithm to count the number of 1-bits. You are required to implement the following two parts of DGIM:
     - Update buckets when a new bit comes in
     - Count the number of 1-bits in the last k bits (k <= window size)
   - Besides `stream_data_dgim.txt`, two files, `edge1.txt` and `edge2.txt`, will be used as the edge-case tests for the implemented DGIM code. 
2. Bloom Filter
   - Bloom filter is a space-efficient probabilistic data structure. In this task, `nltk.corpus.words` is the list of keys $S$ and the words in `nltk.corpus.movie_reviews` is the stream data for query. You are required to implement the following part of Bloom Filter:
     - check whether a new item is in the `word_list`
3. Flajolet-Martin
   - You are required to estimate the number of distinct words occurred in `nltk.corpus.movie_reviews`. 

### Environment

You need to use Python 3.x to write the model, and we highly recommend you to use Python 3.8.0 for us to reproduce your code.

### Dataset

We will be using `stream_data_dgim.txt`,  `edge1.txt`, and  `edge2.txt` for task 1. However, you do NOT need to write any data loading functions. The provided code will handle all the inputs and outputs.

###  Requirements

- Do NOT use any libraries other than the provided ones.
- Write your code between `TODO` and `End of TODO`. Any change in the remaining parts is not allowed.
- Do NOT directly copy others' code, whether from your classmates, previous years' solutions, GitHub, or generative AI.
- Ensure that your code is runnable and reproducible. The TAs will run your code and verify the results. A penalty will be imposed on your final score if there is a significant gap between your reported results or figures and our reproduced ones.

## Submission

Submit your assignment as a ZIP file, named as `[ChineseName]_[StudentID]_HW3.zip` (e.g., `张三_521030910000_HW3.zip`). Your submitted zip file should follow the structure specified as below:

```txt
523030910000_张三_HW3/
├─dgim.py
├─flajolet_matrin.py   
└─bloom_filter.py 
```
