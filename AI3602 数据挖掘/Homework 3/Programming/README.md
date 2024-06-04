# Programming Assignment 3. Similar Item

> TA Contact: Hangyu Ye, Borui Yang  
> E-mail: hangyu_ye@outlook.com, ybirua@sjtu.edu.cn

## Introduction

This project demonstrates using the MinHash algorithm to search a large collection of documents to identify pairs of documents which have a lot of text in common. You need to complete the implementation of MinHash. The incomplete implementation is stored in `main.py`. The specific requirements are provided in main.py with a `TODO` mark.

## Task for this project

1. Convert each test file into a set of 3-shingles.
   - The shingles are formed by combining three consecutive words together.
   - Shingles are mapped to shingle IDs using the `sha1()` hash.
2. Calculate the MinHash signature for each document.
   - The MinHash algorithm should be implemented using the *random hash function trick* which avoids computing random permutations of all of the shingle IDs.
   - Check the comments in the provided code or the lecture slides for details.
3. Compare all MinHash signatures to one another.
   - Compare MinHash signatures by counting the number of components in which the signatures are equal. Divide the number of matching components by the signature length to get a similarity value.

### Environment

You need to use Python 3.x to write the model, and we highly recommend you to use "Python 3.8.0" for us to reproduce your code.

### Dataset

We will be using `./data/articles_1000.train` for this project. However, you do NOT need to write any data loading functions. The provided code will handle all the inputs/outputs.

For your information, the format of `articles_1000.train` is a plain text file. Each line contains an article. The first word of a line is the article ID (e.g. t3820), and the rest part is the content of the article.

###  Requirements

- Do NOT use any libraries other than the provided ones.
- Write your code between `TODO` and `End of TODO`. Any change in the remaining parts is not allowed.
- Do NOT directly copy others' code, whether from your classmates, previous years' solutions, GitHub, or generative AI.
- Ensure that your code is runnable and reproducible. The TAs will run your code and verify the results. A penalty will be imposed on your final score if there is a significant gap between your reported results or figures and our reproduced ones.
- The input and output code is provided in `main.py`. Do NOT modify any of them since the results will be scored automatically.

## Submission

Submit your assignment as a ZIP file, named as `[ChineseName]_[StudentID]_HW1.zip` (e.g., `张三_521030910000_HW1.zip`). Your submitted zip file should follow the structure specified as below

```txt
The project folder should contain:  
├─readme.md
│  
├─data   
│   articles_1000.train   
│   articles_1000.truth   
│   prediction.csv  
│  
└─src  
    main.py   
```
