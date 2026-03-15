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



#随机生成图片a0
if __name__ == "__main__":
    picture = np.random.randn(layer0_size,1)
    a0, a1, a2, a3 = forward_propagation(picture)

    print("系统的最终输出 (10个数字的激活值):")
    print(a3)
    print(f"\n当前网络瞎猜这个数字是: {np.argmax(a3)}")















