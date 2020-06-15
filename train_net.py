# %%
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torch.optim as optim
from visdom import Visdom



def init_net(net, vis):
    initNet = weights_init(net)
    device = torch.device('cuda')
    return initNet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.xavier_normal_(m.weight.data)
        nn.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.xavier_normal_(m.weight.data)
        nn.constant_(m.bias.data, 0.0)
    return m

# %%
def train(net, X_train1, y_train1, batch_size, learnrate, global_step, vis):

    optimizer = optim.Adam(net.parameters(), lr=learnrate)
    net.train()
    running_loss = 0.0
    # Mini batch training
    print('BS', batch_size)
    device = torch.device('cuda')

    if net.func == 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss().to(device)  # bceloss 二分类交叉熵



    for i in range(0, len(X_train1), batch_size):
        # print ('Epocj,mini-batch',epoch,i)
        inputs = torch.from_numpy(X_train1[i:i + batch_size])
        labels = torch.FloatTensor(y_train1[i:i + batch_size] * 1.0)
        # wrap them in Variable
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        # print ('op',outputs.size())
        # print('labels',labels.size())
        optimizer.zero_grad()
        if net.func == 1:
            loss = criterion(outputs, labels.long().squeeze())
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        global_step += 1

        # if vis == 1:
        #     Visdom().line([loss.item()], [global_step], win='train_loss', update='append')

        running_loss += loss.item()
        # for graphing puposes
        # running_loss_array.append(loss.item())

        # print the current status of training
        # if(i % 128 == 0):
        #   print('Train Epoch',epoch,'Mini batch:',i, '\tLoss',loss.data[0])

    return loss, global_step, net

# %%
####  for validation and testing
def valid(net, X_val1, y_val1):

    device = torch.device('cuda')

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

    if net.func == 1:
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
    else:
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
    return predicted, results, predicted_loss_array