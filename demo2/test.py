import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader


# 模型搭建和参数优化
# 在顺利完成数据装载后，我们可以开始编写卷积神经网络模型的搭建和参数优化的代码
# 卷积层使用torch.nn.Conv2d类来搭建
# 激活层使用torch.nn.ReLU 类方法来搭建
# 池化层使用torch.nn.MaxPool2d类方法来搭建
# 全连接层使用 torch.nn.Linear 类方法来搭建

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2))

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )

    # 我们通过继承torch.nn.Modeule来构造网络，因为手写数字
    # 识别比较简单，我们只是用了两个卷积层，一个最大池化层，两个全连接层。
    # 在向前传播过程中进行x.view(-1, 14 * 14 * 128)
    # 对参数实现扁平化。最后通过自己self.dense定义的全连接层进行最后的分类
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


def testDemo2():
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 这里只使用一个均值和标准差，因为是灰度图像
    ])

    data_test = datasets.MNIST(root="", train=False, transform=transform)

    data_loader_test = DataLoader(dataset=data_test, batch_size=4, shuffle=True)

    # 在编写完搭建卷积神经网络模型的代码后，我们可以对模型进行训练和参数进行优化了
    # 首先 定义在训练之前使用哪种损失函数和优化函数
    # 下面定义了计算损失值的损失函数使用的是交叉熵
    # 优化函数使用的额是Adam自适应优化算法
    # model = Model()
    # 将所有的模型参数移动到GPU上
    # if torch.cuda.is_available():
    #     model.cuda()
    # cost = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    # print(model)

    # 卷积神经网络模型进行模型训练和参数优化的代码
    n_epochs = 5

    # 加载已训练的模型
    model = Model()
    model.load_state_dict(torch.load("cnn.pkl"))  # 加载模型参数
    # model.eval()  # 将模型设置为评估模式

    x_test, y_test = next(iter(data_loader_test))
    inputs = Variable(x_test)
    pred = model(inputs)
    _, pred = torch.max(pred, 1)

    # print("Predict Label is:", (i for i in pred))
    # print("Real Label is :", [i for i in y_test])
    print("Predict Label is:", [i.item() for i in pred])
    print("Real Label is :", [i.item() for i in y_test])

    # img = torchvision.utils.make_grid(x_test)
    # img = img.numpy().transpose(1, 2, 0)

    # std = [0.5, 0.5, 0.5]
    # mean = [0.5, 0.5, 0.5]
    # img = img * std + mean
    # cv2.imshow('win', img)
    # key_pressed = cv2.waitKey(0)

    # 显示图像
    img = torchvision.utils.make_grid(x_test, nrow=4, padding=1)
    img = img.permute(1, 2, 0).numpy()
    # 将图像数据从 [-1, 1] 范围缩放到 [0, 1] 范围
    img = (img + 1) / 2
    plt.imshow(img.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()
