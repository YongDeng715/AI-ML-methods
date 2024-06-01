import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def load_file_data(file_path):
    X = []
    y = []
    text = np.loadtxt(file_path, skiprows=1)
    X.append(text[:, 1:])
    y.append(text[:, 0])
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

"""
包含 k 个随机质心的集合。随机质心在整个数据集的边界之内，可以通过找到数据集每一维的最小和最大值,
生成 0~1.0 之间的随机数并通过取值范围和最小值，以便确保随机点在数据的边界之内。
"""
def randCent(dataSet, k):
    n = dataSet.shape[1]         # 列的数量，即数据的特征个数
    centroids = np.zeros((k, n)) # 创建k个质心矩阵
    for j in range(n):           # 创建随机簇质心，并且在每一维的边界内
        minJ = np.min(dataSet[:, j])     # 最小值
        rangeJ = float(np.max(dataSet[:, j]) - minJ)    # 范围 = 最大值 - 最小值
        centroids[:, j] = minJ + rangeJ * np.random.rand(k)  
    return centroids
    
def kMeans_display(X, centroids, clusters, wcss_history, K=6):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(wcss_history, '.--')
    ax[0].set_title('Final WCSS')

    ax[1].scatter(X[:, 0], X[:, 1], c=clusters[:, 0], \
                  marker='o', s=5, cmap=plt.cm.get_cmap('Set1', K))
    ax[1].scatter(centroids[:, 0], centroids[:, 1], c=np.arange(K), \
                  marker='x', s=200, linewidths=3, cmap=plt.cm.get_cmap('Set1', K))
    ax[1].set_title('Final Iteration')

def kMeans(X, K, max_iters=100, if_display=True):
    n_samples, n_features = X.shape
    clusters = np.zeros((n_samples, n_features))  # 保存每个数据点的簇分配结果和平方误差
    centroids = randCent(X, K)
    cluster_changed = True 
    iter = 0      
    wcss_history = []

    plt.ion() 
    if if_display: # if display dynamic clustering procession
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    while cluster_changed and iter < max_iters:
        cluster_changed = False
        for i in range(n_samples):
            min_dist = np.inf
            min_index = -1
            for j in range(K):
                dist = np.sum((X[i] - centroids[j]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
                if clusters[i, 0] != min_index:
                    cluster_changed = True
            clusters[i, :] = min_index, min_dist
        
        wcss = np.sum(min_dist)
        wcss_history.append(wcss)
        # print(centroids)

        for cent in range(K): # Updata centroids
            Inclusters = X[np.nonzero(clusters[:, 0] == cent)[0]]
            centroids[cent, :] = np.mean(Inclusters, axis=0)   
        iter += 1
        
        if if_display:
            
            ax[0].cla()
            ax[0].plot(wcss_history, '.--')
            ax[0].set_title(f'WCSS: {wcss:.4f}')

            ax[1].cla()
            ax[1].scatter(X[:, 0], X[:, 1], c=clusters[:, 0], \
                        marker='o', s=5, cmap=plt.cm.get_cmap('Set1', K))
            ax[1].scatter(centroids[:, 0], centroids[:, 1], c=np.arange(K), \
                        marker='x', s=200, linewidths=3, cmap=plt.cm.get_cmap('Set1', K))
            ax[1].set_title(f'Iteration {iter}')
            plt.pause(0.2)  # 暂停一段时间以便动态显示
    
    kMeans_display(X, centroids, clusters, wcss_history, K=K)
    plt.ioff()
    plt.show()
    
    return centroids, clusters, wcss_history



if __name__ == '__main__':
    file_path = '../gmm/GMM6.txt'
    X, y = load_file_data(file_path)


    print(X.shape, y.shape)
    n_classes = int(np.max(np.unique(y))) + 1 # num of clusters or labels
    plt.scatter(X[:, 0], X[:, 1], c=y, \
                marker='o', s=5, cmap=plt.cm.get_cmap('Set1', n_classes))
    plt.colorbar()
    plt.title('Original data for clustering')
    plt.show()

    centroids, clusters, _ = kMeans(X, K=n_classes, max_iters=50, if_display=True)
    print(centroids.shape, clusters.shape)
    print(centroids)

    labels = clusters[:, 0] # k_means predicting labels
    # 计算 Rand 统计量
    rand_score = metrics.adjusted_rand_score(y, labels)
    # 计算 FM 指数
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(y, labels)
    # 计算轮廓系数
    silhouette_score = metrics.silhouette_score(X, labels)

    print(f"Rand Score: {rand_score:.4f}, FM Score: {fowlkes_mallows_score:.4f}, Silhouette Score: {silhouette_score:.4f}")
