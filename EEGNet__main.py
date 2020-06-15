# %%
import numpy as np
import torch
from visdom import Visdom
import scipy
import scipy.io
import scipy.signal
# %%
import train_net

from EEGNet import EEGNet1_0
from EEGNet import EEGNet1_3
from EEGNet import EEGNet2_0
from EEGNet import EEGNet2_1
from EEGNet import EEGNet2018

def main():
    # 设定训练网络所用的硬件设备
    device = torch.device('cuda')
    # 选择网络模型
    # net = EEGNet1_3().to(device)
    net = EEGNet2018().to(device)

    # init all
    # 初始化总轮回数，用来显示训练曲线
    global_step = 0
    # 初始化网络权重
    net = train_net.weights_init(net)
    # 是否可视化，1 or 0
    vis = 1
    if vis == 1:
        viz = Visdom()
    if vis == 1:
        viz.line([0.], [0.], win='train_loss', update='append', opts={'title': 'train_loss'})
        # viz.line([0.], [0.], win='test_results', update='append', opts={'title': 'test_results'})

    # %%
    """loaddata"""
    dataset_A_train = {}
    targets_A_train = {}
    dataset_A_hold = {}
    targets_A_hold = {}
    filename_train = "G:\EEGNet\data\EEGDataset\ERP\TrainDatas/Circle/dataset_A_train.mat"
    filename_label = "G:\EEGNet\data\EEGDataset\ERP\TrainDatas/Circle/dataset_A_label.mat"
    filename_test = "G:\EEGNet\data\EEGDataset\ERP\TestDatas/Circle/dataset_A_test.mat"
    filename_test_label = "G:\EEGNet\data\EEGDataset\ERP\TestDatas/Circle/dataset_A_label.mat"

    # %%
    dataset_A_train[0] = scipy.io.loadmat(filename_train)['Tdata1']
    targets_A_train[0] = scipy.io.loadmat(filename_label)['Tlabel1']
    dataset_A_train[1] = scipy.io.loadmat(filename_train)['Tdata2']
    targets_A_train[1] = scipy.io.loadmat(filename_label)['Tlabel2']
    dataset_A_train[2] = scipy.io.loadmat(filename_train)['Tdata3']
    targets_A_train[2] = scipy.io.loadmat(filename_label)['Tlabel3']
    dataset_A_train[3] = scipy.io.loadmat(filename_train)['Tdata4']
    targets_A_train[3] = scipy.io.loadmat(filename_label)['Tlabel4']
    dataset_A_train[4] = scipy.io.loadmat(filename_train)['Tdata5']
    targets_A_train[4] = scipy.io.loadmat(filename_label)['Tlabel5']

    dataset_A_hold[0] = scipy.io.loadmat(filename_train)['Vdata1']
    targets_A_hold[0] = scipy.io.loadmat(filename_label)['Vlabel1']
    dataset_A_hold[1] = scipy.io.loadmat(filename_train)['Vdata2']
    targets_A_hold[1] = scipy.io.loadmat(filename_label)['Vlabel2']
    dataset_A_hold[2] = scipy.io.loadmat(filename_train)['Vdata3']
    targets_A_hold[2] = scipy.io.loadmat(filename_label)['Vlabel3']
    dataset_A_hold[3] = scipy.io.loadmat(filename_train)['Vdata4']
    targets_A_hold[3] = scipy.io.loadmat(filename_label)['Vlabel4']
    dataset_A_hold[4] = scipy.io.loadmat(filename_train)['Vdata5']
    targets_A_hold[4] = scipy.io.loadmat(filename_label)['Vlabel5']

    # %%
    dataset_test = scipy.io.loadmat(filename_test)['data']
    targets_test = scipy.io.loadmat(filename_test_label)['labels']

    X_test = np.reshape(dataset_test, [dataset_test.shape[0], 1, dataset_test.shape[1], dataset_test.shape[2]]).astype(
        'float32')
    y_test = np.reshape(targets_test, [targets_test.shape[0], 1]).astype('float32')

    for epoch1 in range(5):  # loop over the dataset multiple times
        X_train = np.reshape(dataset_A_train[epoch1],
                             [dataset_A_train[epoch1].shape[0], 1, dataset_A_train[epoch1].shape[1],
                              dataset_A_train[epoch1].shape[2]]).astype(
            'float32')
        y_train = np.reshape(targets_A_train[epoch1], [targets_A_train[epoch1].shape[0], 1]).astype('float32')

        X_val = np.reshape(dataset_A_hold[epoch1],
                           [dataset_A_hold[epoch1].shape[0], 1, dataset_A_hold[epoch1].shape[1],
                            dataset_A_hold[epoch1].shape[2]]).astype(
            'float32')
        y_val = np.reshape(targets_A_hold[epoch1], [targets_A_hold[epoch1].shape[0], 1]).astype('float32')

        for epoch2 in range(net.epoch):
            print("\nEpoch: ", epoch2, epoch1)
            print("\nglobal_step: ", global_step)
            # 训练网络：输出loss值、global_step用来绘制曲线，训练出的新网络net
            loss, global_step, net = train_net.train(net, X_train, y_train, net.batchSize, net.learnRate, global_step, vis)

            if vis == 1:
                viz.line([loss.item()], [global_step], win='train_loss', update='append', opts={'title':'train_loss'})
            # validation and test: 输出预测值，vali_results展示所有指标，test_loss用来绘制test曲线
            prediction, valid_results, valid_loss = train_net.valid(net, X_val, y_val)
            pred, test_results, test_loss = train_net.valid(net, X_test, y_test)
            if vis == 1:
                viz.line([test_results], [global_step], win='test_results', update='append', opts={'title':'test_results'})

            print('Parameters:["acc", "auc", "recall", "precision","fmeasure"]')
            print('validation_results', valid_results)
            print('test_results', test_results)


if __name__ == '__main__':
    main()
