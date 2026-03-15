import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

layer0_size = 728
layer1_size = 16
layer2_size = 16
layer3_size = 10

#初始化权重和偏置

w1 = np.random.randn(layer1_size, layer0_size)
b1 = np.random.randn(layer1_size, 1)

w2 = np.random.randn(layer2_size, layer1_size)
b2 = np.random.randn(layer2_size, 1)

w3 = np.random.randn(layer3_size, layer2_size)
b3 = np.random.randn(layer3_size, 1)

#计算
def forward_propagation(a0):
    z1 = np.dot(w1, a0) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = sigmoid(z3)
    return a0, a1, a2, a3

def one_hot(number):
    y = np.zeros((10,1))
    y[number] = 1.0
    return y

def compute_cost(a3,y_true):
    cost = np.sum(np.square(a3 - y_true))
    return cost

def sigmoid_derivative(a):
    return sigmoid(a) * (1 - a)

def backward_propagation(a0, a1, a2, a3,y_true):
    dz3 = (a3 - y_true) * sigmoid_derivative(a3)
    dw3 = np.dot(dz3, a2.T)
    db3 = dz3

    dz2 = np.dot(w3.T,dz3) * sigmoid_derivative(a2)
    dw2 = np.dot(dz2, a1.T)
    db2 = dz2

    dz1 = np.dot(w2.T,dz2) * sigmoid_derivative(a1)
    dw1 = np.dot(dz1, a0.T)
    db1 = dz1

    return dw1, dw2, dw3, db1, db2, db3


# 10. 梯度下降（更新旋钮）
def update_parameters(dw1, db1, dw2, db2, dw3, db3, learning_rate=0.1):
    global w1, b1, w2, b2, w3, b3

    w3 = w3 - learning_rate * dw3
    b3 = b3 - learning_rate * db3
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1


if __name__ == "__main__":
    # 1. 准备数据：随机生成一张固定图片，并设定它的真实身份是数字 3
    picture = np.random.rand(layer0_size, 1)
    number = 3
    y_true = one_hot(number)

    epochs = 1000  # 机器要反复看这张图片多少次
    learning_rate = 0.5  # 每次旋钮拧动的幅度 (学习率)

    print(f"开始训练，目标是让网络认出数字: {number}\n")

    # === 这就是让机器“学习”的核心循环 ===
    for i in range(epochs):
        # 步骤 1: 前向传播 (得出预测)
        a0, a1, a2, a3 = forward_propagation(picture)

        # 步骤 2: 计算代价 (看看错多离谱)
        current_cost = compute_cost(a3, y_true)

        # 步骤 3: 反向传播 (追究责任，计算出每个旋钮该怎么调)
        # 注意：这里的接收顺序我帮你调整为和函数内部对应了
        dw1, dw2, dw3, db1, db2, db3 = backward_propagation(a0, a1, a2, a3, y_true)

        # 步骤 4: 梯度下降 (真正动手拧旋钮)
        update_parameters(dw1, db1, dw2, db2, dw3, db3, learning_rate)

        # 每隔 100 次，我们用“示波器”看一下 Cost 信号的能量是不是在下降
        if i % 100 == 0:
            print(f"第 {i} 次迭代，Cost = {current_cost:.6f}")

    # === 训练结束，检验成果 ===
    print("\n--- 训练完成！看看现在的输出 ---")
    _, _, _, a3_final = forward_propagation(picture)

    # 保留 3 位小数打印，方便观察
    print(f"最终输出的 10 个概率分布:\n{np.round(a3_final.flatten(), 3)}")
    print(f"\n现在网络信誓旦旦地认为这个数字是: {np.argmax(a3_final)}")















