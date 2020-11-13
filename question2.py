import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_cost(X, y, theta):
    # 相当于96行2列 * 2行1列
    # 矩阵.T为转置
    inner = np.power(((X * theta.T) - y), 2)
    # 取和除1/2m
    return np.sum(inner) / (2 * len(X))


def gradient_descent(X, y, theta, alpha, epoch):
    """return theta, cost"""

    cost = np.zeros(epoch)  # 初始化一个np.array，包含每次epoch的cost
    m = X.shape[0]  # 样本数量m

    for i in range(epoch):
        # 迭代次数
        # 利用向量化一步求解
        temp = theta - (alpha / m) * (X * theta.T - y).T * X

        theta = temp
        cost[i] = compute_cost(X, y, theta)

    return theta, cost


def multiple_linear_regression(alpha, data2, is_draw):
    # set X (training data) and y (target variable)
    cols = data2.shape[1]
    X2 = data2.iloc[:, 0:cols - 1]
    y2 = data2.iloc[:, cols - 1:cols]

    # convert to matrices and initialize theta
    X2 = np.matrix(X2.values)
    y2 = np.matrix(y2.values)
    theta2 = np.matrix(np.array([0, 0, 0]))

    epoch = 1500

    # perform linear regression on the data set
    g2, cost2 = gradient_descent(X2, y2, theta2, alpha, epoch)

    # get the cost (error) of the model
    compute_cost(X2, y2, g2)

    if is_draw:
        fig, ax = plt.subplots(figsize=(12, 8))
        # np.arange创建等差数组
        ax.plot(np.arange(epoch), cost2, 'r')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Cost')
        ax.set_title('Error vs. Training Epoch')
        plt.savefig("题目二拟合结果/alpha={}迭代次数-代价曲线.png".format(alpha))
        plt.show()
    return g2, cost2


if __name__ == "__main__":
    path = 'ex1data2.txt'
    data2 = pd.read_csv(path, names=['Size', 'Bedrooms', 'Price'])
    # print(data2.head())
    print(data2.describe())

    # 预处理步骤 - 特征归一化
    # Z-score
    # 减去平均值除以标准差,归一化后标准差为1
    data_mean = data2.mean()
    data_std = data2.std()
    data2 = (data2 - data2.mean()) / data2.std()
    print(data2.describe())

    # 归一化min-max

    # add ones column
    data2.insert(0, 'Ones', 1)
    # alpha小收敛速度慢，
    # 太大损失函数每次迭代后不一定能下降，可能会发散
    time_cost = []
    x = []

    # alpha 0.01 0.05 0.1
    alpha = 0.1
    final_theta, cost = multiple_linear_regression(alpha, data2, True)
    print("回归参数为:{}".format(final_theta * data_std["Size"] + data_mean["Bedrooms"]))
    print("最终代价为：{}".format(cost[-1] * data_std["Size"] + data_mean["Bedrooms"]))
    # alpha越大学习速度越快，但是拟合越不准确

    price = (final_theta[0, 0] + (final_theta[0, 1] * (1650 * data_std["Size"] + data_mean["Bedrooms"])) + (
                final_theta[0, 2] * (3 * data_std["Bedrooms"] + data_mean["Bedrooms"])))
    print("预测房屋面积为 1650 平方英尺，房间数量为 3 时，预测价格：{}".format(price))
