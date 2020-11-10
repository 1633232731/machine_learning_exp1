import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置


# 代价函数计算
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


def create_by_sklearn():
    city_population = []
    profit = []
    # 读入数据
    with open("ex1data1.txt", "r", encoding='utf-8') as file:
        for line in file.readlines():
            line = line.strip()
            city_population.append(float(line.split(",")[0]))
            profit.append(float(line.split(",")[1]))

    # 数组转换
    city_population = np.array(city_population)
    profit = np.array(profit)

    # 定义模型
    model = LinearRegression()
    city = city_population
    profit = profit

    # 变为一列
    x = city.reshape((-1, 1))
    y = profit

    # 拟合
    model.fit(x, y)
    print("sklearn回归参数为: {} \t {}".format(model.intercept_, model.coef_[0]))
    # print(model.intercept_)  # 截距
    # print(model.coef_)  # 线性模型的系数
    x2 = [[-1], [3.5], [7], [22.5]]  # 取两个预测值
    y2 = model.predict(x2)  # 进行预测
    print("sklearn线性回归 预测人口30000，利润：{}".format(y2[1]))
    print("sklearn线性回归 预测人口70000，利润：{}".format(y2[2]))

    plt.title("sklearn线性回归")
    plt.xlabel('city_population', fontsize=15, color='b')
    plt.ylabel('profit', fontsize=15, color='b')
    plt.plot(city, profit, 'k.')  # 黑点
    plt.plot(x2, y2, 'g-')  # 画出拟合曲线，绿色实线
    plt.savefig("题目一拟合结果/sklearn拟合.png")
    plt.show()

    plt.xlabel('city_population', fontsize=15, color='b')
    plt.ylabel('profit', fontsize=15, color='b')
    plt.title("sklearn线性回归残差")
    yr = model.predict(x)
    for index, x in enumerate(x):
        plt.plot([x, x], [y[index], yr[index]], 'r-')
    plt.plot(city, profit, 'k.')  # 黑点
    plt.plot(x2, y2, 'g-')  # 画出拟合曲线
    plt.savefig("题目一拟合结果/sklearn线性回归残差.png")
    plt.show()


def create_by_local():
    path = 'ex1data1.txt'
    # names添加列名，header用指定的行来作为标题，若原无标题且指定标题则设为None
    data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

    # 查看数据简介
    # print(data.describe())

    # 可视化数据
    data.plot(kind='scatter', x='Population', y='Profit', figsize=(8, 5))
    plt.savefig("题目一拟合结果/原始数据.png")
    plt.show()

    # print(data)
    # 在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度
    data.insert(0, 'Ones', 1)
    # print(data)

    # 列数
    cols = data.shape[1]

    # 取前cols-1列，即输入向量(ONE和人口列)
    X = data.iloc[:, 0:cols - 1]

    # 取最后一列，即目标向量(利润列）
    y = data.iloc[:, cols - 1:cols]

    X = np.matrix(X.values)
    y = np.matrix(y.values)

    # theta = [[0 0]]
    theta = np.matrix([0, 0])
    # 截距和斜率

    # 维度，可以进行输出
    X.shape, theta.shape, y.shape
    # ((97, 2), (1, 2), (97, 1))

    print("代价初始值为：{}".format(compute_cost(X, y, theta)))  # 代价初始值：32.072733877455676

    alpha = 0.01
    epoch = 1500

    # 梯度下降求参数
    final_theta, cost = gradient_descent(X, y, theta, alpha, epoch)
    print("回归参数为:{}".format(final_theta))
    print("最终代价为：{}".format(compute_cost(X, y, final_theta)))

    # np.linspace()在指定的间隔内返回均匀间隔的数字
    x = np.linspace(data.Population.min(), data.Population.max(), 100)  # 横坐标
    f = final_theta[0, 0] + (final_theta[0, 1] * x)  # 纵坐标，利润
    # 截距 + 斜率*横坐标

    # 画出拟合直线
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data['Population'], data.Profit, label='Traning Data')
    ax.legend(loc=2)  # 2表示在左上角
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    plt.savefig("题目一拟合结果/梯度下降拟合.png")
    plt.show()

    # 画出代价值随迭代次数的变化
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(epoch), cost, 'r')  # np.arange()返回等差数组
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Cost vs. Training Epoch')
    plt.savefig("题目一拟合结果/梯度下降拟合代价.png")
    plt.show()

    print("预测人口30000，利润：{}".format(final_theta[0, 0] + (final_theta[0, 1] * 3)))
    print("预测人口70000，利润：{}".format(final_theta[0, 0] + (final_theta[0, 1] * 7)))
    print()


if __name__ == "__main__":
    # 手写预测
    create_by_local()
    # 利用sklearn预测
    create_by_sklearn()
