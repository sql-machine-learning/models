import numpy as np

class DBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        '''
        :param eps: Neighborhood distance
        :param min_samples:
            The minimum number of samples required to form a cluster
        '''
        self.eps = eps
        self.min_samples = min_samples
        self.core_sample_indices_ = list()
        self.components_ = None
        self.labels_ = None

    def euclidean_distances(self, X, Y=None, Y_norm_squared=None, X_norm_squared=None):
        '''
        将数据的每行看做样本，计算两矩阵样本之间的欧氏距离
        :param X: matrix one
        :param Y: matrix two
        :param Y_norm_squared:
        :param X_norm_squared:
        :return: pairwise距离矩阵
        '''
        X = np.array(X)
        Y = np.array(Y) if Y else X  # 若未指定Y则令其为X

        dist_mat = np.dot(X, Y.T)

        X_squared = np.sum(np.square(X), axis=1).reshape((dist_mat.shape[0], -1))
        Y_squared = np.sum(np.square(Y), axis=1).reshape((-1, dist_mat.shape[1]))
        squared_dist = X_squared - 2 * dist_mat + Y_squared
        squared_dist[squared_dist < 0] = 0  # 在某些数据下可能出现负数，需要做截断处理

        return np.sqrt(squared_dist)

    def fit(self, X):
        dist_mat = self.euclidean_distances(X)
        dens_arr = list()  # 密度数组
        for row in dist_mat:
            dens = np.sum(row <= self.eps)  # 计算密度
            dens_arr.append(dens)
        dens_arr = np.array(dens_arr)
        visited_arr = [False for _ in range(len(X))]  # 访问标记数组
        self.labels_ = [-1 for _ in range(len(X))]  # 所属类别
        k = -1  # 第几个类别，初始默认所有样本点均为噪声点

        # 遍历样本点
        for idx in range(len(X)):
            if visited_arr[idx]:  # 已被访问则跳过
                continue

            visited_arr[idx] = True

            if dens_arr[idx] == 1 or dens_arr[idx] < self.min_samples:  # 噪声样本或边界样本
                continue

            else:  # 核心对象
                # 访问队列，会在循环中对其进行修改
                cores_q = [i for i in range(
                    len(X)) if dist_mat[i, idx] <= self.eps and dens_arr[i] >= self.min_samples]
                k += 1  # 新建类别
                self.labels_[idx] = k  # 为当前核心对象赋予类别

                while cores_q:  # BFS式访问密度相连的核心对象
                    cur_core = cores_q.pop(0)

                    # 对未被访问的核心对象操作，已被访问的核心直接跳过
                    if not visited_arr[cur_core]:
                        visited_arr[cur_core] = True
                        self.labels_[cur_core] = k

                        neighbors = [i for i in range(
                            len(X)) if dist_mat[i, cur_core] <= self.eps]  # 邻域内的所有样本点
                        neighbor_cores = [
                            i for i in neighbors if i not in cores_q and dens_arr[i] >= self.min_samples]  # 邻域内的所有核心对象
                        neighbor_boards = [
                            i for i in neighbors if dens_arr[i] < self.min_samples]  # 邻域内的所有边界样本

                        # 核心点加入队列等待访问
                        cores_q.extend(neighbor_cores)

                        # 边界点进行归类
                        for node_idx in neighbor_boards:
                            if self.labels_[node_idx] == -1:
                                self.labels_[node_idx] = k

    def call(self, inputs):
        self.fit(inputs)
        return self.labels_

    def sqlflow_train_loop(self, x, epochs=1, verbose=0):
        pass

    def display_model_info(self, verbose=0):
        pass

def loss(*args, **kwargs):
    return None

def optimizer(*args, **kwargs):
    return None


# if __name__ == '__main__':
#     from sklearn.datasets.samples_generator import make_blobs
#     from sklearn.preprocessing import StandardScaler
#
#     centers = [[1, 1], [-1, -1], [1, -1]]
#     X, Y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                       random_state=0)
#     X = StandardScaler().fit_transform(X)
#
#     db = DBSCAN(eps=0.3, min_samples=10)
#     db.call(X)
#
#     import matplotlib.pyplot as plt
#
#     plt.clf()
#     plt.scatter(X[:, 0], X[:, 1], c=db.labels_)
#     plt.show()
#
#     # 对比sklearn
#     del db
#     from sklearn.cluster import DBSCAN
#
#     db = DBSCAN(eps=0.3, min_samples=10)
#     db.fit(X)
#     plt.clf()
#     plt.scatter(X[:, 0], X[:, 1], c=db.labels_)
#     plt.show()