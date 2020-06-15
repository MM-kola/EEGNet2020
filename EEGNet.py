import torch
from torch import nn
from torch.nn import functional as F

"""原始eegnet模型"""
class EEGNet1_0(nn.Module):

    func = 0
    learnRate = 5e-3
    batchSize = 75
    epoch = 300

    def __init__(self):
        super(EEGNet1_0, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 15, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d((2, 4))

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(4 * 4 * 4, 1)

    def forward(self, x):
        # Layer 1
        #  x = x.double()
        x = F.elu(self.conv1(x))
        #print(x.size())
        x = self.batchnorm1(x)
        x = x.permute(0, 3, 1, 2)
        #print(x.size())
        x = F.dropout(x, 0.15)

        # Layer 2
        x = self.padding1(x)
        #print(x.size())
        x = F.elu(self.conv2(x))
        #print(x.size())
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)
        #print(x.size())

        # Layer 3
        x = self.padding2(x)
        #print(x.size())
        x = F.elu(self.conv3(x))
        #print(x.size())
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)
        #print (x.size())
        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 4 * 4 * 4)  # for T=128
        x = torch.sigmoid(self.fc1(x))
        return x

#  vgg
"""原始eegnet基础上小卷积核代替大卷积核"""
class EEGNet1_1(nn.Module):

    func = 0
    learnRate = 5e-3
    batchSize = 75
    epoch = 300

    def __init__(self):
        super(EEGNet1_1, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.conv2_1 = nn.Conv2d(1, 4, (3, 3))
        self.batchnorm2_1 = nn.BatchNorm2d(4, False)
        self.conv2_2 = nn.Conv2d(4, 6, (3, 3))
        self.batchnorm2_2 = nn.BatchNorm2d(6, False)
        self.pooling2 = nn.MaxPool2d((2, 4))

        # Layer 3
        self.conv3_1 = nn.Conv2d(6, 8, (1, 4))
        self.batchnorm3_1 = nn.BatchNorm2d(8, False)
        self.conv3_2 = nn.Conv2d(8, 10, (1, 4))
        self.batchnorm3_2 = nn.BatchNorm2d(10, False)
        self.conv3_3 = nn.Conv2d(10, 12, (1, 4))
        self.batchnorm3_3 = nn.BatchNorm2d(12, False)
        self.pooling3 = nn.MaxPool2d((2, 2))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(12 * 3 * 4, 1)

    def forward(self, x):
        # Layer 1
        #  x = x.double()
        x = F.elu(self.conv1(x))
        #print(x.size())
        x = self.batchnorm1(x)
        x = x.permute(0, 3, 1, 2)
        #print(x.size())
        x = F.dropout(x, 0.25)

        # Layer 2
        #print(x.size())
        x = F.elu(self.conv2_1(x))
        #print(x.size())
        x = self.batchnorm2_1(x)
        x = F.dropout(x, 0.25)
        x = F.elu(self.conv2_2(x))
        # print(x.size())
        x = self.batchnorm2_2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        #print(x.size())

        # Layer 3
        # x = self.padding2(x)
        #print(x.size())
        x = F.elu(self.conv3_1(x))
        #print(x.size())
        x = self.batchnorm3_1(x)
        x = F.dropout(x, 0.25)
        x = F.elu(self.conv3_2(x))
        # print(x.size())
        x = self.batchnorm3_2(x)
        x = F.dropout(x, 0.25)
        x = F.elu(self.conv3_3(x))
        # print(x.size())
        x = self.batchnorm3_3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        #print (x.size())
        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 12 * 3 * 4)  # for T=128
        x = torch.sigmoid(self.fc1(x))
        return x


#  Fisrt signal then channel
"""先时域后空域"""
class EEGNet1_2(nn.Module):

    func = 0
    learnRate = 5e-3
    batchSize = 75
    epoch = 300

    def __init__(self):
        super(EEGNet1_2, self).__init__()
        #self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 32, (9, 1), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(32, False)

        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, (1, 64))
        self.batchnorm2 = nn.BatchNorm2d(64, False)

        # Layer 3
        self.conv3_1 = nn.Conv2d(1, 8, (3, 3))
        self.batchnorm3_1 = nn.BatchNorm2d(8, False)
        self.conv3_2 = nn.Conv2d(8, 16, (3, 3))
        self.batchnorm3_2 = nn.BatchNorm2d(16, False)
        self.pooling3 = nn.MaxPool2d((2, 2))

        # Layer 4
        self.conv4_1 = nn.Conv2d(16, 24, (3, 3))
        self.batchnorm4_1 = nn.BatchNorm2d(24, False)
        self.conv4_2 = nn.Conv2d(24, 32, (3, 3))
        self.batchnorm4_2 = nn.BatchNorm2d(32, False)
        self.conv4_3 = nn.Conv2d(32, 40, (3, 3))
        self.batchnorm4_3 = nn.BatchNorm2d(40, False)
        self.pooling4 = nn.MaxPool2d((2, 2))

        #Layer 5
        self.conv5_1 = nn.Conv2d(40, 48, (3, 3))
        self.batchnorm5_1 = nn.BatchNorm2d(48, False)
        self.conv5_2 = nn.Conv2d(48, 56, (3, 3))
        self.batchnorm5_2 = nn.BatchNorm2d(56, False)
        self.conv5_3 = nn.Conv2d(56, 30, (1, 1))
        self.batchnorm5_3 = nn.BatchNorm2d(30, False)

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(30 * 8 * 8, 1)

    def forward(self, x):
        # Layer 1
        #  x = x.double()
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)

        # Layer 2
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = x.permute(0, 3, 1, 2)
        x = F.dropout(x, 0.25)

        # Layer 3
        x = F.elu(self.conv3_1(x))
        x = self.batchnorm3_1(x)
        x = F.dropout(x, 0.25)
        x = F.elu(self.conv3_2(x))
        x = self.batchnorm3_2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # Layer 4
        x = F.elu(self.conv4_1(x))
        x = self.batchnorm4_1(x)
        x = F.dropout(x, 0.25)
        x = F.elu(self.conv4_2(x))
        x = self.batchnorm4_2(x)
        x = F.dropout(x, 0.25)
        x = F.elu(self.conv4_3(x))
        x = self.batchnorm4_3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling4(x)

        # Layer 4
        x = F.elu(self.conv5_1(x))
        x = self.batchnorm5_1(x)
        x = F.dropout(x, 0.25)
        x = F.elu(self.conv5_2(x))
        x = self.batchnorm5_2(x)
        x = F.dropout(x, 0.25)
        x = F.elu(self.conv5_3(x))
        x = self.batchnorm5_3(x)
        x = F.dropout(x, 0.25)

        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 30 * 8 * 8)
        out = torch.sigmoid(self.fc1(x))
        return out


"""
eeg的基础上对空域后的统一提取时域（效果最佳） 

 """
class EEGNet1_3(nn.Module):

    func = 0
    learnRate = 5e-3  # ERP 5e-3
    batchSize = 75  # erp 75 ST 500
    epoch = 250


    def __init__(self):
        super(EEGNet1_3, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64))
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.conv2 = nn.Conv2d(16, 16, (9, 1))
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.MaxPool2d((1, 2))

        # Layer 3
        self.conv3 = nn.Conv2d(1, 16, (3, 19))
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.pooling3 = nn.MaxPool2d((2, 2))

        # Layer 4
        self.conv4_1 = nn.Conv2d(16, 16, (2, 2))
        self.batchnorm4_1 = nn.BatchNorm2d(16, False)
        self.conv4_2 = nn.Conv2d(16, 8, 1, 1)
        self.batchnorm4_2 = nn.BatchNorm2d(8, False)
        self.pooling4 = nn.MaxPool2d((2, 2))
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(8 * 3 * 3, 1)

    def forward(self, x):
        # Layer 1

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)

        # Layer 2
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.dropout(x, 0.15)
        x = self.pooling2(x)

        # Layer 3
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        # x = F.dropout(x, 0.15)
        x = self.pooling3(x)

        # Layer 4
        x = F.elu(self.conv4_1(x))
        x = self.batchnorm4_1(x)
        # x = F.dropout(x, 0.15)
        x = F.elu(self.conv4_2(x))
        x = self.batchnorm4_2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling4(x)

        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 8 * 3 * 3)  # for T=128
        x = torch.sigmoid(self.fc1(x))
        return x


"""eeg的基础上对空域后的统一提取时域(VGG)"""
class EEGNet1_4(nn.Module):

    func = 0
    learnRate = 2e-4
    batchSize = 75
    epoch = 300

    def __init__(self):
        super(EEGNet1_4, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64))
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.conv2_1 = nn.Conv2d(16, 20, (5, 1))
        self.batchnorm2_1 = nn.BatchNorm2d(20, False)
        self.conv2_2 = nn.Conv2d(20, 40, (5, 1))
        self.batchnorm2_2 = nn.BatchNorm2d(40, False)
        self.conv2_3 = nn.Conv2d(40, 60, (5, 1))
        self.batchnorm2_3 = nn.BatchNorm2d(60, False)
        self.pooling2 = nn.MaxPool2d((2, 2))

        # Layer 3
        self.conv3_1 = nn.Conv2d(1, 10, (3, 3))
        self.batchnorm3_1 = nn.BatchNorm2d(10, False)
        self.conv3_2 = nn.Conv2d(10, 20, (3, 3))
        self.batchnorm3_2 = nn.BatchNorm2d(20, False)
        self.conv3_3 = nn.Conv2d(20, 30, (3, 3))
        self.batchnorm3_3 = nn.BatchNorm2d(30, False)
        self.pooling3 = nn.MaxPool2d((2, 2))

        # Layer 4
        self.conv4_1 = nn.Conv2d(30, 40, (3, 3))
        self.batchnorm4_1 = nn.BatchNorm2d(40, False)
        self.conv4_2 = nn.Conv2d(40, 50, (3, 3))
        self.batchnorm4_2 = nn.BatchNorm2d(50, False)
        self.conv4_3 = nn.Conv2d(50, 60, (3, 3))
        self.batchnorm4_3 = nn.BatchNorm2d(60, False)
        self.conv4_4 = nn.Conv2d(60, 20, (1, 1))
        self.batchnorm4_4 = nn.BatchNorm2d(20, False)


        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(20 * 6 * 6, 1)

    def forward(self, x):
        # Layer 1

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        #x = F.dropout(x, 0.25)

        # Layer 2
        x = F.elu(self.conv2_1(x))
        x = self.batchnorm2_1(x)
        x = F.dropout(x, 0.15)
        # print(x.shape)
        x = F.elu(self.conv2_2(x))
        x = self.batchnorm2_2(x)
        x = F.dropout(x, 0.15)
        # print(x.shape)
        x = F.elu(self.conv2_3(x))
        x = self.batchnorm2_3(x)
        x = F.dropout(x, 0.15)
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        x = self.pooling2(x)
        # print(x.shape)

        # Layer 3
        x = F.elu(self.conv3_1(x))
        x = self.batchnorm3_1(x)
        x = F.dropout(x, 0.15)
        x = F.elu(self.conv3_2(x))
        x = self.batchnorm3_2(x)
        x = F.dropout(x, 0.15)
        x = F.elu(self.conv3_3(x))
        x = self.batchnorm3_3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)

        # Layer 3
        x = F.elu(self.conv4_1(x))
        x = self.batchnorm4_1(x)
        x = F.dropout(x, 0.15)
        x = F.elu(self.conv4_2(x))
        x = self.batchnorm4_2(x)
        x = F.dropout(x, 0.15)
        x = F.elu(self.conv4_3(x))
        x = self.batchnorm4_3(x)
        x = F.dropout(x, 0.15)
        x = F.elu(self.conv4_4(x))
        x = self.batchnorm4_4(x)
        x = F.dropout(x, 0.15)


        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        logits = x.view(-1, 20 * 6 * 6)  # for T=128
        out = torch.sigmoid(self.fc1(logits))

        return out


"""eeg的基础上对空域后的统一提取时域（效果最佳）"""
class EEGNet1_5(nn.Module):

    func = 0
    learnRate = 8e-3
    batchSize = 1000
    epoch = 2500


    def __init__(self):
        super(EEGNet1_5, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 4))
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.conv2 = nn.Conv2d(16, 8, (41, 1))
        self.batchnorm2 = nn.BatchNorm2d(8, False)
        self.pooling2 = nn.MaxPool2d((1, 2))

        # Layer 3
        self.conv3 = nn.Conv2d(1, 8, (5, 5))
        self.batchnorm3 = nn.BatchNorm2d(8, False)
        self.pooling3 = nn.MaxPool2d((2, 2))

        # Layer 4
        # self.conv4_1 = nn.Conv2d(16, 24, (2, 2))
        # self.batchnorm4_1 = nn.BatchNorm2d(24, False)
        # self.conv4_2 = nn.Conv2d(24, 8, 1, 1)
        # self.batchnorm4_2 = nn.BatchNorm2d(8, False)
        # self.pooling4 = nn.MaxPool2d((2, 2))
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(8 * 2 * 6, 1)

    def forward(self, x):
        # Layer 1

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)

        # Layer 2
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = x.permute(0, 3, 1, 2)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)

        # Layer 3
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)

        # Layer 4
        # x = F.elu(self.conv4_1(x))
        # x = self.batchnorm4_1(x)
        # # x = F.dropout(x, 0.15)
        # x = F.elu(self.conv4_2(x))
        # x = self.batchnorm4_2(x)
        # x = F.dropout(x, 0.15)
        # x = self.pooling4(x)

        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 8 * 2 * 6)  # for T=128
        x = torch.sigmoid(self.fc1(x))
        return x


"""  EEGNet2均使用softmax作为输出，
      0：原始EEGNet结构；
      1：增加提取时域特征层；
      2：对1结构的vgg化；
    """
class EEGNet2_0(nn.Module):

    func = 1
    learnRate = 2e-4
    batchSize = 75
    epoch = 500

    def __init__(self):
        super(EEGNet2_0, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 15, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d((2, 4))

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x):
        # Layer 1
        #  x = x.double()
        x = F.elu(self.conv1(x))
        #print(x.size())
        x = self.batchnorm1(x)
        x = x.permute(0, 3, 1, 2)
        #print(x.size())
        x = F.dropout(x, 0.15)

        # Layer 2
        x = self.padding1(x)
        #print(x.size())
        x = F.elu(self.conv2(x))
        #print(x.size())
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)
        #print(x.size())

        # Layer 3
        x = self.padding2(x)
        #print(x.size())
        x = F.elu(self.conv3(x))
        #print(x.size())
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)
        #print (x.size())
        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 4 * 4 * 4)  # for T=128
        x = F.softmax(self.fc1(x), dim=1)
        return x


"""eeg的基础上对空域后的统一提取时域"""
class EEGNet2_1(nn.Module):

    func = 1
    learnRate = 2e-4
    batchSize = 75
    epoch = 500


    def __init__(self):
        super(EEGNet2_1, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64))
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.conv2 = nn.Conv2d(16, 16, (9, 1))
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.MaxPool2d((1, 2))

        # Layer 3
        self.conv3 = nn.Conv2d(1, 16, (3, 19))
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.pooling3 = nn.MaxPool2d((2, 2))

        # Layer 4
        self.conv4_1 = nn.Conv2d(16, 16, (2, 2))
        self.batchnorm4_1 = nn.BatchNorm2d(16, False)
        self.conv4_2 = nn.Conv2d(16, 8, 1, 1)
        self.batchnorm4_2 = nn.BatchNorm2d(8, False)
        self.pooling4 = nn.MaxPool2d((2, 2))
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(8 * 3 * 3, 2)

    def forward(self, x):
        # Layer 1

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)

        # Layer 2
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.dropout(x, 0.15)
        x = self.pooling2(x)

        # Layer 3
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        # x = F.dropout(x, 0.15)
        x = self.pooling3(x)

        # Layer 4
        x = F.elu(self.conv4_1(x))
        x = self.batchnorm4_1(x)
        # x = F.dropout(x, 0.15)
        x = F.elu(self.conv4_2(x))
        x = self.batchnorm4_2(x)
        # x = F.dropout(x, 0.15)
        x = self.pooling4(x)

        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 8 * 3 * 3)  # for T=128
        x = F.softmax(self.fc1(x), dim=1)
        return x


class EEGNet2_2(nn.Module):

    LearnRate = 5e-4
    BatchSize = 75
    Epoch = 200

    def __init__(self):
        super(EEGNet2_2, self).__init__()

        self.spatiotemporal_unit = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 64), stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64, False),
            nn.ZeroPad2d((0, 0, 4, 3)),
            nn.Conv2d(64, 64, kernel_size=(16, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64, False)
        )

        self.vgg_unit = nn.Sequential(

            # (B,1,64,64)
            nn.Conv2d(1, 8, kernel_size=(7, 7), stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(8, False),
            nn.Conv2d(8, 16, kernel_size=(7, 7), stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(16, False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # (B,64,32,32)

            nn.Conv2d(16, 32, kernel_size=(7, 7), stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32, False),
            nn.Conv2d(32, 32, kernel_size=(7, 7), stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32, False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # (B,128,16,16)

            nn.Conv2d(32, 64, kernel_size=(7, 7), stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64, False),
            nn.Conv2d(64, 64, kernel_size=(7, 7), stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64, False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # (B,256,8,8)

            nn.Conv2d(64, 128, kernel_size=(7, 7), stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128, False),
            nn.Conv2d(128, 128, kernel_size=(7, 7), stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128, False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            # (B,512,4,4)
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(128*4*4, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024, False),
            nn.Dropout(0.25),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128, False),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
            nn.BatchNorm1d(2, False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        x = self.spatiotemporal_unit(x)
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        x = self.vgg_unit(x)
        # print(x.shape)
        x = x.view(-1, 128*4*4)
        # print(x.shape)
        logit = self.fc_unit(x)
        # out = torch.sigmoid(logit)
        return logit


"""Xception 原版"""
class EEGNet2018(nn.Module):

    func = 0
    learnRate = 5e-2
    batchSize = 500
    epoch = 300

    def __init__(self):
        super(EEGNet2018, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 4, (37, 1), padding=(18, 0))
        self.batchnorm1 = nn.BatchNorm2d(4, False)

        # Layer 2
        # self.padding1 = nn.ZeroPad2d((16, 15, 0, 1))
        self.conv2 = nn.Conv2d(4, 8, (1, 64), groups=4)
        self.batchnorm2 = nn.BatchNorm2d(8, False)
        self.pooling2 = nn.MaxPool2d((4, 1))

        # Layer 3
        self.conv3 = nn.Conv2d(8, 8, (9, 1), padding=(4, 0), groups=8)
        self.batchnorm3 = nn.BatchNorm2d(8, False)
        self.convp = nn.Conv2d(8, 8, (1, 1), padding=0)
        self.pooling3 = nn.MaxPool2d((3, 1))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(8 * 6 * 1, 1)

    def forward(self, x):
        # Layer 1
        #  x = x.double()
        x = F.elu(self.conv1(x))
        #print(x.size())
        x = self.batchnorm1(x)
        # x = x.permute(0, 3, 1, 2)
        #print(x.size())
        # x = F.dropout(x, 0.15)

        # Layer 2
        # x = self.padding1(x)
        #print(x.size())
        x = F.elu(self.conv2(x))
        #print(x.size())
        x = self.batchnorm2(x)
        # x = F.dropout(x, 0.15)
        x = self.pooling2(x)
        #print(x.size())

        # Layer 3
        # x = self.padding2(x)
        #print(x.size())
        x = F.elu(self.conv3(x))
        x = F.elu(self.convp(x))
        #print(x.size())
        x = self.batchnorm3(x)
        # x = F.dropout(x, 0.15)
        x = self.pooling3(x)
        #print (x.size())
        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 8 * 6 * 1)  # for T=128
        x = torch.sigmoid(self.fc1(x))
        return x


"""Xception 原版"""
class EEGNet2018_42_200(nn.Module):

    func = 0
    learnRate = 1e-3
    batchSize = 150
    epoch = 300

    def __init__(self):
        super(EEGNet2018_42_200, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 4, (101, 1), padding=(50, 0))
        self.batchnorm1 = nn.BatchNorm2d(4, False)

        # Layer 2
        # self.padding1 = nn.ZeroPad2d((16, 15, 0, 1))
        self.conv2 = nn.Conv2d(4, 8, (1, 64), groups=4)
        self.batchnorm2 = nn.BatchNorm2d(8, False)
        self.pooling2 = nn.MaxPool2d((4, 1))

        # Layer 3
        self.conv3 = nn.Conv2d(8, 8, (25, 1), padding=(12, 0), groups=8)
        self.batchnorm3 = nn.BatchNorm2d(8, False)
        self.convp = nn.Conv2d(8, 8, (1, 1), padding=0)
        self.pooling3 = nn.MaxPool2d((10, 1))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(8 * 5 * 1, 1)

    def forward(self, x):
        # Layer 1
        #  x = x.double()
        x = F.elu(self.conv1(x))
        #print(x.size())
        x = self.batchnorm1(x)
        # x = x.permute(0, 3, 1, 2)
        #print(x.size())
        x = F.dropout(x, 0.15)

        # Layer 2
        # x = self.padding1(x)
        #print(x.size())
        x = F.elu(self.conv2(x))
        #print(x.size())
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)
        #print(x.size())

        # Layer 3
        # x = self.padding2(x)
        #print(x.size())
        x = F.elu(self.conv3(x))
        x = F.elu(self.convp(x))
        #print(x.size())
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)
        #print (x.size())
        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 8 * 5 * 1)  # for T=128
        x = torch.sigmoid(self.fc1(x))
        return x


"""Xception 原版"""
class EEGNet2018_82_200(nn.Module):

    func = 0
    learnRate = 1e-3
    batchSize = 150
    epoch = 300

    def __init__(self):
        super(EEGNet2018_82_200, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 8, (101, 1), padding=(50, 0))
        self.batchnorm1 = nn.BatchNorm2d(8, False)

        # Layer 2
        # self.padding1 = nn.ZeroPad2d((16, 15, 0, 1))
        self.conv2 = nn.Conv2d(8, 16, (1, 64), groups=4)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.MaxPool2d((4, 1))

        # Layer 3
        self.conv3 = nn.Conv2d(16, 16, (25, 1), padding=(12, 0), groups=8)
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.convp = nn.Conv2d(16, 16, (1, 1), padding=0)
        self.pooling3 = nn.MaxPool2d((10, 1))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(16 * 5 * 1, 1)

    def forward(self, x):
        # Layer 1
        #  x = x.double()
        x = F.elu(self.conv1(x))
        #print(x.size())
        x = self.batchnorm1(x)
        # x = x.permute(0, 3, 1, 2)
        #print(x.size())
        x = F.dropout(x, 0.15)

        # Layer 2
        # x = self.padding1(x)
        #print(x.size())
        x = F.elu(self.conv2(x))
        #print(x.size())
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)
        #print(x.size())

        # Layer 3
        # x = self.padding2(x)
        #print(x.size())
        x = F.elu(self.conv3(x))
        x = F.elu(self.convp(x))
        #print(x.size())
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)
        #print (x.size())
        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 16 * 5 * 1)  # for T=128
        x = torch.sigmoid(self.fc1(x))
        return x


"""eegnet2018 vgg版"""
class EEGNet2018_vgg_200(nn.Module):

    func = 0
    learnRate = 1e-3
    batchSize = 150
    epoch = 300

    def __init__(self):
        super(EEGNet2018_vgg_200, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1_1 = nn.Conv2d(1, 8, (61, 1), padding=(30, 0))
        self.batchnorm1_1 = nn.BatchNorm2d(8, False)
        self.conv1_2 = nn.Conv2d(8, 8, (61, 1), padding=(30, 0))
        self.batchnorm1_2 = nn.BatchNorm2d(8, False)
        self.conv1_3 = nn.Conv2d(8, 8, (61, 1), padding=(30, 0))
        self.batchnorm1_3 = nn.BatchNorm2d(8, False)

        # Layer 2
        # self.padding1 = nn.ZeroPad2d((16, 15, 0, 1))
        self.conv2 = nn.Conv2d(8, 16, (1, 64), groups=4)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.MaxPool2d((4, 1))

        # Layer 3
        self.conv3 = nn.Conv2d(16, 16, (25, 1), padding=(12, 0), groups=8)
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.convp = nn.Conv2d(16, 16, (1, 1), padding=0)
        self.pooling3 = nn.MaxPool2d((10, 1))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        self.fc1 = nn.Linear(16 * 5 * 1, 1)

    def forward(self, x):
        # Layer 1
        #  x = x.double()
        x = F.elu(self.conv1_1(x))
        #print(x.size())
        x = self.batchnorm1_1(x)
        x = F.elu(self.conv1_2(x))
        # print(x.size())
        x = self.batchnorm1_2(x)
        x = F.elu(self.conv1_3(x))
        # print(x.size())
        x = self.batchnorm1_3(x)
        # x = x.permute(0, 3, 1, 2)
        #print(x.size())
        x = F.dropout(x, 0.15)

        # Layer 2
        # x = self.padding1(x)
        #print(x.size())
        x = F.elu(self.conv2(x))
        #print(x.size())
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)
        #print(x.size())

        # Layer 3
        # x = self.padding2(x)
        #print(x.size())
        x = F.elu(self.conv3(x))
        x = F.elu(self.convp(x))
        #print(x.size())
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)
        #print (x.size())
        # FC Layer
        """-1代表无所谓是几，由机器做合适的判决，填充满足之后已知的情况下的数"""
        x = x.view(-1, 16 * 5 * 1)  # for T=128
        x = torch.sigmoid(self.fc1(x))
        return x


"""eegnet2018 lstm版"""
class EEGLstmNet_42_200(nn.Module):

    func = 0
    learnRate = 1e-3
    batchSize = 150
    epoch = 3000

    def __init__(self):
        super(EEGLstmNet_42_200, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 4, (101, 1), padding=(50, 0))
        self.batchnorm1 = nn.BatchNorm2d(4, False)

        # Layer 2
        # self.padding1 = nn.ZeroPad2d((16, 15, 0, 1))
        self.conv2 = nn.Conv2d(4, 8, (1, 64), groups=4)
        self.batchnorm2 = nn.BatchNorm2d(8, False)
        self.pooling2 = nn.MaxPool2d((4, 1))

        # Layer 3
        self.conv3 = nn.Conv2d(8, 8, (25, 1), padding=(12, 0), groups=8)
        self.batchnorm3 = nn.BatchNorm2d(8, False)
        self.convp = nn.Conv2d(8, 8, (1, 1), padding=0)
        self.pooling3 = nn.MaxPool2d((5, 1))

        # LSTM Layer
        self.rnn = nn.LSTM(
            input_size=8,
            hidden_size=160,# 8*10*x
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.out = nn.Linear(160 * 2, 1)

    def forward(self, x):
        # Layer 1

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.15)

        # Layer 2

        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)


        # Layer 3
        x = F.elu(self.conv3(x))
        x = F.elu(self.convp(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)

        # LSTM Layer
        x = x.view(-1, 8, 10)
        x = x.permute(0, 2, 1)
        x, (h_n, h_c) = self.rnn(x, None)
        x = self.out(x[:, -1, :])
        x = torch.sigmoid(x)
        return x


"""eegnet2018 lstm版"""
class EEGLstmNet_82_200(nn.Module):

    func = 0
    learnRate = 1e-3
    batchSize = 200
    epoch = 3000

    def __init__(self):
        super(EEGLstmNet_82_200, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 8, (101, 1), padding=(50, 0))
        self.batchnorm1 = nn.BatchNorm2d(8, False)
        # self.conv1_1 = nn.Conv2d(1, 8, (61, 1), padding=(30, 0))
        # self.batchnorm1_1 = nn.BatchNorm2d(8, False)
        # self.conv1_2 = nn.Conv2d(8, 8, (61, 1), padding=(30, 0))
        # self.batchnorm1_2 = nn.BatchNorm2d(8, False)
        # self.conv1_3 = nn.Conv2d(8, 8, (61, 1), padding=(30, 0))
        # self.batchnorm1_3 = nn.BatchNorm2d(8, False)

        # Layer 2
        # self.padding1 = nn.ZeroPad2d((16, 15, 0, 1))
        self.conv2 = nn.Conv2d(8, 16, (1, 64), groups=4)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.MaxPool2d((4, 1))

        # Layer 3
        self.conv3 = nn.Conv2d(16, 16, (25, 1), padding=(12, 0), groups=8)
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.convp = nn.Conv2d(16, 16, (1, 1), padding=0)
        self.pooling3 = nn.MaxPool2d((2, 1))

        # Layer 4
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 144 timepoints.
        # 4, 2, 487
        # self.fc1 = nn.Linear(16 * 5 * 1, 1)

        # LSTM Layer
        self.rnn = nn.LSTM(
            input_size=16,
            hidden_size=16*25*2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.out = nn.Linear(16*25*2*2, 1)


    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.15)
        # x = F.elu(self.conv1_1(x))
        # x = self.batchnorm1_1(x)
        # x = F.dropout(x, 0.15)
        # x = F.elu(self.conv1_2(x))
        # x = self.batchnorm1_2(x)
        # x = F.dropout(x, 0.15)
        # x = F.elu(self.conv1_3(x))
        # x = self.batchnorm1_3(x)
        # x = F.dropout(x, 0.15)

        # Layer 2
        # x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)

        # Layer 3
        # x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.convp(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)
        # LSTM Layer
        x = x.view(-1, 16, 25)
        x = x.permute(0, 2, 1)
        x, (h_n, h_c) = self.rnn(x, None)
        x = self.out(x[:, -1, :])
        x = torch.sigmoid(x)
        return x


"""eegnet2018 lstm版"""
class EEGLstmNet_42N_200(nn.Module):

    func = 0
    learnRate = 5e-3
    batchSize = 250
    epoch = 10000

    def __init__(self):
        super(EEGLstmNet_42N_200, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 4, (101, 1), padding=(50, 0))
        self.batchnorm1 = nn.BatchNorm2d(4, False)

        # Layer 2
        # self.padding1 = nn.ZeroPad2d((16, 15, 0, 1))
        self.conv2 = nn.Conv2d(4, 8, (1, 64), groups=4)
        self.batchnorm2 = nn.BatchNorm2d(8, False)
        self.pooling2 = nn.MaxPool2d((4, 1))

        # Layer 3
        self.conv3 = nn.Conv2d(8, 8, (25, 1), padding=(12, 0), groups=8)
        self.batchnorm3 = nn.BatchNorm2d(8, False)
        self.convp = nn.Conv2d(8, 8, (1, 1), padding=0)
        self.pooling3 = nn.MaxPool2d((5, 1))

        # LSTM Layer
        self.rnn0 = nn.LSTM(
            input_size=1,
            hidden_size=120,# 8*10*x
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout0 = nn.Sequential(
            nn.BatchNorm1d(240, False),
            nn.Linear(120 * 2, 1),
            nn.ReLU()
        )
        self.rnn1 = nn.LSTM(
            input_size=1,
            hidden_size=120,  # 8*10*x
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout1 = nn.Sequential(
            nn.BatchNorm1d(240, False),
            nn.Linear(120 * 2, 1),
            nn.ReLU()
        )
        self.rnn2 = nn.LSTM(
            input_size=1,
            hidden_size=120,  # 8*10*x
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout2 = nn.Sequential(
            nn.BatchNorm1d(240, False),
            nn.Linear(120 * 2, 1),
            nn.ReLU()
        )
        self.rnn3 = nn.LSTM(
            input_size=1,
            hidden_size=120,  # 8*10*x
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout3 = nn.Sequential(
            nn.BatchNorm1d(240, False),
            nn.Linear(120 * 2, 1),
            nn.ReLU()
        )
        self.rnn4 = nn.LSTM(
            input_size=1,
            hidden_size=120,  # 8*10*x
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout4 = nn.Sequential(
            nn.BatchNorm1d(240, False),
            nn.Linear(120 * 2, 1),
            nn.ReLU()
        )
        self.rnn5 = nn.LSTM(
            input_size=1,
            hidden_size=120,  # 8*10*x
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout5 = nn.Sequential(
            nn.BatchNorm1d(240, False),
            nn.Linear(120 * 2, 1),
            nn.ReLU()
        )
        self.rnn6 = nn.LSTM(
            input_size=1,
            hidden_size=120,  # 8*10*x
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout6 = nn.Sequential(
            nn.BatchNorm1d(240, False),
            nn.Linear(120 * 2, 1),
            nn.ReLU()
        )
        self.rnn7 = nn.LSTM(
            input_size=1,
            hidden_size=120,  # 8*10*x
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout7 = nn.Sequential(
            nn.BatchNorm1d(240, False),
            nn.Linear(120 * 2, 1),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.BatchNorm1d(8, False),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        # Layer 1

        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.15)

        # Layer 2

        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)

        # Layer 3
        x = F.elu(self.conv3(x))
        x = F.elu(self.convp(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)

        # LSTM Layer
        x = x.view(-1, 8, 10)
        x0 = x[:, 0, :].view(-1, 10, 1)
        x1 = x[:, 1, :].view(-1, 10, 1)
        x2 = x[:, 2, :].view(-1, 10, 1)
        x3 = x[:, 3, :].view(-1, 10, 1)
        x4 = x[:, 4, :].view(-1, 10, 1)
        x5 = x[:, 5, :].view(-1, 10, 1)
        x6 = x[:, 6, :].view(-1, 10, 1)
        x7 = x[:, 7, :].view(-1, 10, 1)
        x0, (h_n, h_c) = self.rnn0(x0, None)
        x1, (h_n, h_c) = self.rnn1(x1, None)
        x2, (h_n, h_c) = self.rnn2(x2, None)
        x3, (h_n, h_c) = self.rnn3(x3, None)
        x4, (h_n, h_c) = self.rnn4(x4, None)
        x5, (h_n, h_c) = self.rnn5(x5, None)
        x6, (h_n, h_c) = self.rnn6(x6, None)
        x7, (h_n, h_c) = self.rnn7(x7, None)
        x0 = self.lstmout0(x0[:, -1, :])
        x1 = self.lstmout1(x1[:, -1, :])
        x2 = self.lstmout2(x2[:, -1, :])
        x3 = self.lstmout3(x3[:, -1, :])
        x4 = self.lstmout4(x4[:, -1, :])
        x5 = self.lstmout5(x5[:, -1, :])
        x6 = self.lstmout6(x6[:, -1, :])
        x7 = self.lstmout7(x7[:, -1, :])
        x = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7), 1)
        x = self.out(x)
        return x


# 最好
class EEGLstmNet3fc_82_200(nn.Module):

    func = 0
    learnRate = 1e-3
    batchSize = 300
    epoch = 5000

    def __init__(self):
        super(EEGLstmNet3fc_82_200, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, (101, 1), padding=(50, 0)),
            nn.BatchNorm2d(8, False),
            nn.Sigmoid()
        )


        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, (1, 64), groups=8),
            nn.BatchNorm2d(16, False),
            nn.Sigmoid(),
            nn.AvgPool2d((4, 1))
        )

        # Layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, (25, 1), padding=(12, 0), groups=8),
            nn.BatchNorm2d(16, False),
            nn.Sigmoid(),
            nn.Conv2d(16, 16, (1, 1), padding=0),
            nn.BatchNorm2d(16, False),
            nn.Sigmoid(),
            nn.AvgPool2d((2, 1))
        )


        # LSTM Layer
        self.rnn = nn.LSTM(
            input_size=16,
            hidden_size=16 * 25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （batch,time_step,input）时是Ture
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(16 * 25 * 2 * 2, False),
            # nn.Dropout(0.15),
            nn.Linear(16 * 25 * 2 * 2, 800),
            nn.BatchNorm1d(800, False),
            nn.Sigmoid(),
            # nn.Linear(800, 400),
            # nn.Dropout(0.15),
            # nn.BatchNorm1d(800, False),
            # nn.Sigmoid(),
            nn.Linear(800, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = F.dropout(x, 0.15)

        # Layer 2
        x = self.conv2(x)
        x = F.dropout(x, 0.15)

        # Layer 3
        x = self.conv3(x)
        x = F.dropout(x, 0.15)
        #
        # # LSTM Layer
        x = x.view(-1, 16, 25)
        x = x.permute(0, 2, 1)
        x, (h_n, h_c) = self.rnn(x, None)
        x = self.fc(x[:, -1, :])
        return x


# 差
class EEGLstmNet_vggfc_82_200(nn.Module):

    func = 0
    learnRate = 1e-3
    batchSize = 300
    epoch = 3000

    def __init__(self):
        super(EEGLstmNet_vggfc_82_200, self).__init__()
        # self.T = 120

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, (31, 1), padding=(15, 0)),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm2d(8, False),
            nn.Conv2d(8, 8, (31, 1), padding=(15, 0)),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm2d(8, False),
            nn.Conv2d(8, 8, (31, 1), padding=(15, 0)),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm2d(8, False)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, (1, 64), groups=4),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm2d(16, False),
            nn.MaxPool2d((4, 1))
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, (9, 1), padding=(4, 0), groups=8),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm2d(16, False),
            nn.Conv2d(16, 16, (9, 1), padding=(4, 0), groups=8),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm2d(16, False),
            nn.Conv2d(16, 16, (9, 1), padding=(4, 0), groups=8),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm2d(16, False),
            nn.Conv2d(16, 16, (1, 1), padding=0),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.MaxPool2d((2, 1))
        )

        # LSTM Layer
        self.rnn = nn.LSTM(
            input_size=16,
            hidden_size=16 * 25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （batch,time_step,input）时是Ture
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(16 * 25 * 2 * 2, False),
            nn.Linear(16 * 25 * 2 * 2, 800),
            nn.Dropout(0.15),
            nn.Sigmoid(),
            nn.BatchNorm1d(800, False),
            nn.Linear(800, 1),
            nn.Dropout(0.15),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)

        # Layer 2
        x = self.layer2(x)

        # Layer 3
        x = self.layer3(x)
        #
        # # LSTM Layer
        x = x.view(-1, 16, 25)
        x = x.permute(0, 2, 1)
        x, (h_n, h_c) = self.rnn(x, None)
        x = self.fc(x[:, -1, :])
        return x


class EEGLstmNetfc_82_200(nn.Module):

    func = 0
    learnRate = 1e-3
    batchSize = 300
    epoch = 8000

    def __init__(self):
        super(EEGLstmNetfc_82_200, self).__init__()
        # self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 8, (101, 1), padding=(50, 0))
        self.batchnorm1 = nn.BatchNorm2d(8, False)

        # Layer 2
        self.conv2 = nn.Conv2d(8, 16, (1, 64), groups=4)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.AvgPool2d((4, 1))

        # Layer 3
        self.conv3 = nn.Conv2d(16, 16, (25, 1), padding=(12, 0), groups=8)
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.convp = nn.Conv2d(16, 16, (1, 1), padding=0)
        self.pooling3 = nn.AvgPool2d((2, 1))

        # LSTM Layer
        self.covlstm = nn.Sequential(
            nn.Conv2d(1, 4, (1, 16)),
            nn.ReLU()
            # nn.BatchNorm2d(16, False)
        )
        self.rnn = nn.LSTM(
            input_size=4,
            hidden_size=25 * 4 * 4,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （batch,time_step,input）时是Ture
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(25 * 4 * 4 * 2, False),
            nn.Linear(25 * 4 * 4 * 2, 400),
            nn.Sigmoid(),
            nn.BatchNorm1d(400, False),
            nn.Linear(400, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.15)

        # Layer 2
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)

        # Layer 3
        x = F.elu(self.conv3(x))
        x = F.elu(self.convp(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)
        #
        # # LSTM Layer
        x = x.permute(0, 3, 2, 1)
        x = self.covlstm(x)
        x = x.view(-1, 25, 4)
        x, (h_n, h_c) = self.rnn(x, None)
        x = self.fc(x[:, -1, :])
        return x


class EEGLstmNet_82N_200(nn.Module):

    learnRate = 2e-3
    batchSize = 250
    epoch = 10000

    def __init__(self):
        super(EEGLstmNet_82N_200, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 8, (101, 1), padding=(50, 0))
        self.batchnorm1 = nn.BatchNorm2d(8, False)

        # Layer 2
        self.conv2 = nn.Conv2d(8, 16, (1, 64), groups=4)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.MaxPool2d((4, 1))

        # Layer 3
        self.conv3 = nn.Conv2d(16, 16, (25, 1), padding=(12, 0), groups=8)
        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.convp = nn.Conv2d(16, 16, (1, 1), padding=0)
        self.pooling3 = nn.MaxPool2d((2, 1))

        # LSTM Layer
        self.rnn0 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout0 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn1 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout1 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn2 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout2 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn3 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout3 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn4 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout4 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn5 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout5 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn6 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout6 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn7 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout7 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn8 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout8 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn9 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout9 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn10 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout10 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn11 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout11 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn12 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout12 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn13 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout13 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn14 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout14 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.rnn15 = nn.LSTM(
            input_size=1,
            hidden_size=25 * 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # （time_step,batch,input）时是Ture
        )
        self.lstmout15 = nn.Sequential(
            nn.BatchNorm1d(25 * 2 * 2, False),
            nn.Linear(25 * 2 * 2, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(16, False),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.15)

        # Layer 2
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.15)
        x = self.pooling2(x)

        # Layer 3
        x = F.elu(self.conv3(x))
        x = F.elu(self.convp(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.15)
        x = self.pooling3(x)

        # # LSTM Layer
        x = x.view(-1, 16, 25)
        y0 = x[:, 0, :].view(-1, 25, 1)
        y0, (h_n, h_c) = self.rnn0(y0, None)
        y0 = self.lstmout0(y0[:, -1, :])
        y1 = x[:, 1, :].view(-1, 25, 1)
        y1, (h_n, h_c) = self.rnn1(y1, None)
        y1 = self.lstmout1(y1[:, -1, :])
        y2 = x[:, 2, :].view(-1, 25, 1)
        y2, (h_n, h_c) = self.rnn2(y2, None)
        y2 = self.lstmout2(y2[:, -1, :])
        y3 = x[:, 3, :].view(-1, 25, 1)
        y3, (h_n, h_c) = self.rnn3(y3, None)
        y3 = self.lstmout3(y3[:, -1, :])
        y4 = x[:, 4, :].view(-1, 25, 1)
        y4, (h_n, h_c) = self.rnn4(y4, None)
        y4 = self.lstmout4(y4[:, -1, :])
        y5 = x[:, 5, :].view(-1, 25, 1)
        y5, (h_n, h_c) = self.rnn5(y5, None)
        y5 = self.lstmout5(y5[:, -1, :])
        y6 = x[:, 6, :].view(-1, 25, 1)
        y6, (h_n, h_c) = self.rnn6(y6, None)
        y6 = self.lstmout6(y6[:, -1, :])
        y7 = x[:, 7, :].view(-1, 25, 1)
        y7, (h_n, h_c) = self.rnn7(y7, None)
        y7 = self.lstmout7(y7[:, -1, :])
        y8 = x[:, 8, :].view(-1, 25, 1)
        y8, (h_n, h_c) = self.rnn8(y8, None)
        y8 = self.lstmout8(y8[:, -1, :])
        y9 = x[:, 9, :].view(-1, 25, 1)
        y9, (h_n, h_c) = self.rnn9(y9, None)
        y9 = self.lstmout9(y9[:, -1, :])
        y10 = x[:, 10, :].view(-1, 25, 1)
        y10, (h_n, h_c) = self.rnn10(y10, None)
        y10 = self.lstmout10(y10[:, -1, :])
        y11 = x[:, 11, :].view(-1, 25, 1)
        y11, (h_n, h_c) = self.rnn11(y11, None)
        y11 = self.lstmout11(y11[:, -1, :])
        y12 = x[:, 12, :].view(-1, 25, 1)
        y12, (h_n, h_c) = self.rnn12(y12, None)
        y12 = self.lstmout12(y12[:, -1, :])
        y13 = x[:, 13, :].view(-1, 25, 1)
        y13, (h_n, h_c) = self.rnn13(y13, None)
        y13 = self.lstmout13(y13[:, -1, :])
        y14 = x[:, 14, :].view(-1, 25, 1)
        y14, (h_n, h_c) = self.rnn14(y14, None)
        y14 = self.lstmout14(y14[:, -1, :])
        y15 = x[:, 15, :].view(-1, 25, 1)
        y15, (h_n, h_c) = self.rnn15(y15, None)
        y15 = self.lstmout15(y15[:, -1, :])
        out = torch.cat((y0, y1, y2, y3,
                         y4, y5, y6, y7,
                         y8, y9, y10, y11,
                         y12, y13, y14, y15), 1)
        out = self.fc(out)
        return out


def main():
    x = torch.randn(3, 1, 200, 64)
    net = EEGLstmNet3fc_82_200()
    y = net(x)
    # para = list(net.parameters())
    # print(para.count())
    print(y.shape)
    # print(y)


if __name__ == '__main__':
    main()

