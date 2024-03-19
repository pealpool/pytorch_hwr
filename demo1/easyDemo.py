import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 输入28*28的图片，即输入层有784个节点，输出64维向量，共3层
        # 这里的64是隐藏层的维度（节点），隐藏层节点数量是不好确定的，从1到100万都可以。
        # 一般网络层数越多越好，但练难度越大。
        # 但太多了很容易过拟合，效果反而差，因此需要合理设置隐藏层节点数量。
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        # 第4层输出10维向量，用于识别10个数字
        self.fc4 = torch.nn.Linear(64, 10)

    # 定义向前传播过程
    def forward(self, x):
        # self.fc1(x)先做全连接线性计算，然后用激活函数relu激活
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        # 最后softmax归一化，对数log是提升稳定性
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


# 导入数据集
def get_data_loader(is_train):
    # 对数据集进行转换成张量（to_tensor/多维数组），
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 下载MNIST数据集，is_train表示是否是训练集
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    # 返回一个DataLoader加载数据集，batch_size表示每次加载多少数据，shuffle表示是否打乱顺序
    return DataLoader(data_set, batch_size=15, shuffle=True)


# 评估模型准确率
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        # 按批次取出数据
        for (x, y) in test_data:
            # 计算神经网络的预测值
            outputs = net.forward(x.view(-1, 28 * 28))
            # 对批次中每个结果进行比较，累计预测正确的个数
            for i, output in enumerate(outputs):
                # argmax计算输出中最大值的索引，即预测的数字
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    # 返回准确率
    return n_correct / n_total


def easyDemo():
    # 定义训练集和测试集
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    # 初始化神经网络
    net = Net()

    # 打印未经训练的准确率
    print("initial accuracy:", evaluate(test_data, net))
    # 以下是pytorch训练神经网络的通用写法
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # 训练2个网络轮次，多轮训练可以提升模型的准确率
    for epoch in range(2):
        for (x, y) in train_data:
            # 初始化
            net.zero_grad()
            # 正向传播
            output = net.forward(x.view(-1, 28 * 28))
            # 计算差值，其中nll_loss是为了匹配之前的对数log_softmax
            loss = torch.nn.functional.nll_loss(output, y)
            # 反向误差传播
            loss.backward()
            # 优化网络参数
            optimizer.step()
        # 打印每轮训练的准确率
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    # 随机取出3个测试数据，显示预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()
