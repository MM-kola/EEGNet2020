# %%
# win+r: python -m visdom.server
# 训练Single Trail



import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from visdom import Visdom
import scipy
# %%
import scipy.io
import matplotlib.pyplot as plt
import random
import scipy.signal
from EEGNet import *

device = torch.device('cuda')
net = EEGLstmNet3fc_82_200().to(device)

# train_filename = 'G:\EEGNet\data\EEGDataset\SingleTaril\TrainDatas\Circle/TrainDataset.mat'
# hold_filename = 'G:\EEGNet\data\EEGDataset\SingleTaril\TrainDatas\Circle/HoldDataset.mat'
train_filename = 'G:\EEGNet\data\EEGDataset\SingleTaril\TrainDatas/Normal/TrainDataset_new.mat'
hold_filename = 'G:\EEGNet\data\EEGDataset\SingleTaril\TrainDatas/Normal/HoldDataset_new.mat'
# test_filename = 'G:\EEGNet\data\EEGDataset\SingleTaril\TestDatas\Circle/test_set_A.mat'
# train_filename = 'G:/xDAWN/matlab/train_preproc.mat'
# hold_filename = 'G:/xDAWN/matlab/hold_preproc.mat'
vis = 1

lr = net.learnRate
print(lr)
batch_size = net.batchSize
Epochs = net.epoch

criterion = nn.BCELoss().to(device)  #bceloss 二分类交叉熵
running_loss_array = []
if vis == 1:
    viz = Visdom()
global_step = 0


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.xavier_normal_(m.weight.data)
        nn.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.xavier_normal_(m.weight.data)
        nn.constant_(m.bias.data, 0.0)


# %%
def train(X_train1, y_train1, batch_size, learnrate, global_step):

    optimizer = optim.Adam(net.parameters(), lr=learnrate)
    net.train()
    running_loss = 0.0
    # Mini batch training
    print('BS', batch_size)
    for i in range(0, len(X_train1), batch_size):
        # print ('Epocj,mini-batch',epoch,i)
        inputs = torch.from_numpy(X_train1[i:i + batch_size])
        labels = torch.FloatTensor(y_train1[i:i + batch_size] * 1.0)
        # wrap them in Variable
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs).to(device)
        # print ('op',outputs.size())
        # print('labels',labels.size())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        global_step += 1

        if vis == 1:
            viz.line([loss.item()], [global_step], win='train_loss', update='append')

        running_loss += loss.item()
        # for graphing puposes
        running_loss_array.append(loss.item())

        # print the current status of training
        # if(i % 128 == 0):
        #   print('Train Epoch',epoch,'Mini batch:',i, '\tLoss',loss.data[0])

    return running_loss_array, global_step

# %%
####  for validation and testing
def valid(net, X_val1, y_val1):
    net.eval()
    predicted_loss = []
    inputs = torch.from_numpy(X_val1)
    labels = torch.FloatTensor(y_val1 * 1.0)
    #inputs, labels = Variable(inputs), Variable(labels)
    inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
    '''
    output = net(data)
    valid_loss = valid_loss + F.nll_loss(output, target, size_average=False).data[0] 
    pred = output.data.max(1, keepdim=True)[1]
    correct = correct + pred.eq(target.data.view_as(pred)).cpu().sum() 
    '''
    ########################
    results = []
    predicted_loss_array = []

    # inputs = Variable(torch.from_numpy(X_val1))#.cuda(0))

    predicted = net(inputs)
    predicted = predicted.data.cpu().numpy()
    # from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
    # print ('predicted',predicted.shape)
    # print ('predicted',predicted)
    Y = labels.data.cpu().numpy()
    # print ('Y',Y)
    for param in ["acc", "auc", "recall", "precision", "fmeasure"]:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2 * precision * recall / (precision + recall))

    # print (len(results))
    return predicted, results, predicted_loss_array  # validation_loss_array

def main():
    # %%
    """loaddata"""
    # dataset_A_train = scipy.io.loadmat(train_filename)
    dataset_A_train = scipy.io.loadmat(train_filename)['data']
    targets_A_train = scipy.io.loadmat(train_filename)['label']

    dataset_A_hold = scipy.io.loadmat(hold_filename)['data']
    targets_A_hold = scipy.io.loadmat(hold_filename)['label']

    # dataset_A_train = scipy.io.loadmat(train_filename)['train_preproc']
    # targets_A_train = scipy.io.loadmat(train_filename)['train_label']
    #
    # dataset_A_hold = scipy.io.loadmat(hold_filename)['hold_preproc']
    # targets_A_hold = scipy.io.loadmat(hold_filename)['hold_label']

    # %%

    # %% test data
    # dataset_test = scipy.io.loadmat(test_filename)['data']
    # targets_test = scipy.io.loadmat(test_filename)['labels']
    #
    # X_test = np.reshape(dataset_test, [dataset_test.shape[0], 1, dataset_test.shape[1], dataset_test.shape[2]]).astype(
    #     'float32')
    # y_test = np.reshape(targets_test, [targets_test.shape[0], 1]).astype('float32')

    running_loss_array = []
    global_step = 0
    weights_init(net)
    if vis == 1:
        viz.line([0.], [0.], win='train_loss', update='append')

    X_train = np.reshape(dataset_A_train,
                         [dataset_A_train.shape[0], 1, dataset_A_train.shape[1],
                          dataset_A_train.shape[2]]).astype(
        'float32')
    y_train = np.reshape(targets_A_train, [targets_A_train.shape[0], 1]).astype('float32')
    permut = np.random.permutation(X_train.shape[0])
    shuffle_x = X_train[permut, :]
    shuffle_y = y_train[permut, :]
    # print(shuffle_x, shuffle_y)

    X_val = np.reshape(dataset_A_hold,
                       [dataset_A_hold.shape[0], 1, dataset_A_hold.shape[1],
                        dataset_A_hold.shape[2]]).astype(
        'float32')
    y_val = np.reshape(targets_A_hold, [targets_A_hold.shape[0], 1]).astype('float32')
    for epoch in range(Epochs):
        print("\nEpoch: ", epoch)
        print("\nglobal_step: ", global_step)
        # print ('range',int(len(X_train)/batch_size-1))
        train_loss, global_step = train(shuffle_x, shuffle_y, batch_size, lr, global_step)
        prediction, valid_results, valid_loss = valid(net, X_val, y_val)
        # pred, test_results, test_loss = valid(net, X_test, y_test)
        if vis == 1:
            viz.line([valid_results], [global_step], win='test_results', update='append')

        print('Parameters:["acc", "auc", "recall", "precision","fmeasure"]')
        print('validation_results', valid_results)
        # print('test_results', test_results)


if __name__ == '__main__':
    main()
