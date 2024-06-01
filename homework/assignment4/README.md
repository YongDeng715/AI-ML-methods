# Clustering and Naive Bayes

## 实验目的

1. 掌握聚类算法原理，能够使用聚类算法对数据进行聚类。
2. 掌握高斯朴素贝叶斯算法原理，能够使用高斯朴素贝叶斯算法对数据进行分类。

## 实验内容

1. 读取 ```gmm``` 文件夹中的数据，使用 k_means 聚类算法对数据进行聚类，并绘制出聚类结果。

我们可以直接运行 ```src/```中的文件，实现对```gmm```中数据进行 k_means 聚类和 GMM 聚类 以及 高斯朴素贝叶斯分类。

```bash
.
├── README.md
├── assignment4.md
├── figs
│   ├── data-gmm3.png
│   ├── data-gmm4.png
│   ├── data-gmm6.png
│   ├── data-gmm8.png
│   ├── gmm-6.png
│   ├── gmm-8.png
│   ├── gnb-res6.png
│   ├── gnb-res8.png
│   ├── kMeans-6.png
│   └── kMeans-8.png
├── gmm
│   ├── GMM3.txt
│   ├── GMM4.txt
│   ├── GMM6.txt
│   └── GMM8.txt
├── scripts
│   ├── bayes.ipynb
│   └── cluster.ipynb
└── src
    ├── gmm.py
    ├── k_means.py
    └── naive_bayes.py
```