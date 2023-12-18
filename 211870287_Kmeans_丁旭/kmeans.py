import numpy as np
import matplotlib.pyplot as plt


class UnsupervisedLearning:

    def __init__(self, img_path):
        self.img = plt.imread(img_path)
        self.shape = self.img.shape
        self.images = [self.img]

    # k是分组数；tol‘中心点误差’；iter是迭代次数
    def kmeans(self, iter, k, tol):

        # 保存图片的行宽和列宽
        row = self.img.shape[0]
        col = self.img.shape[1]
        data = self.img.reshape(-1, 3)

        # 添加一列 之后用来存储离这个像素距离最近（也就是颜色最相近的簇心的下标j  簇心即为cluster_center[j]）
        data = np.column_stack((data, np.ones(row * col)))

        # 1.随机产生初始簇心
        cluster_center = data[np.random.choice(row * col, k)]

        # 2.分类
        distance = [[] for _ in range(k)]

        for i in range(iter):
            print("迭代次数：", i+1)
            # 2.1距离计算
            for j in range(k):
                # 这里采用了“广播”（broadcasting）的机制，让 cluster_center[j] 按照 data 的行数进行自动复制，使其变成一个与 data 形状相同的矩阵。
                distance[j] = np.sqrt(np.sum((data - cluster_center[j]) ** 2, axis=1))
            # 2.2归类
            data[:, 3] = np.argmin(distance, axis=0)
            # 3.计算新簇心
            pre_cluster_center = np.copy(cluster_center)
            for j in range(k):
                cluster_center[j] = np.mean(data[data[:, 3] == j], axis=0)

            # 4.停止条件
            move_distance = np.sqrt(np.sum((np.array(cluster_center) - np.array(pre_cluster_center)) ** 2))
            if move_distance < tol:
                print("移动距离小于阈值，算法终止。")
                break
        img_re = (data[:, 3]).reshape(row, col)
        self.images.append(img_re)

    def show(self):
        # 用来正常显示中文标签
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # 显示图像
        titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=3',
                  u'聚类图像 K=4', u'聚类图像 K=5', u'聚类图像 K=6']

        for i in range(len(self.images)):
            plt.subplot(2, 3, i + 1), plt.imshow(self.images[i]),
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == "__main__":
    unsupervisedLearning = UnsupervisedLearning('environment.jpg')
    for i in range(2, 7):
        unsupervisedLearning.kmeans(100, i, 0.001)
    unsupervisedLearning.show()
