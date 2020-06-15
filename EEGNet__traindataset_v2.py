'''
ERP TRAIN SET
'''


import numpy as np
import scipy
# %%
import scipy.io
import random
import scipy.signal
from scipy import signal
import mne
from sklearn import preprocessing
# %%
TRAIN_A = 'data/BCI_Comp_III_Wads_2004/Subject_A_Train1.mat'
TRAIN_B = 'data/BCI_Comp_III_Wads_2004/Subject_B_Train1.mat'

MATRIX = ['abcdef',
          'ghijkl',
          'mnopqr',
          'stuvwx',
          'yz1234',
          '56789_']
screen = [['A', 'B', 'C', 'D', 'E', 'F'],
          ['G', 'H', 'I', 'J', 'K', 'L'],
          ['M', 'N', 'O', 'P', 'Q', 'R'],
          ['S', 'T', 'U', 'V', 'W', 'X'],
          ['Y', 'Z', '1', '2', '3', '4'],
          ['5', '6', '7', '8', '9', '_']]

#%%loaddata
"""loaddata -> Signal_A_240, Flashing_A_240, StimulusCode_A_240, StimulusType_A_240, Target_A
               Signal_A_240:[实验个数 points channel],[85 7794 64]   
               Flashing_A_240:[实验个数 points(0,1)],[85 7794]
               StimulusCode_A_240:[实验个数 points(0,1~12)],[85 7794]
               StimulusType_A_240:[实验个数 points(0,1)],[85 7794]
               Target_A:StimulusType对应的字符"""
def load_dataset(SUBJECT, flag):
    """flag标志是否输出target"""
    data = scipy.io.loadmat(SUBJECT)

    print ('Subject A dataa',data)
    Signal = np.float32(data['Signal'])
    print ('signal',Signal, Signal.shape)

    Flashing = np.float32(data['Flashing'])
    print ('flashing',Flashing, Flashing.shape)

    StimulusCode = np.float32(data['StimulusCode'])
    print ('Stimulus COde',StimulusCode,StimulusCode.shape)
    if flag == 0:
        StimulusType = np.float32(data['StimulusType'])
        # print ('Stimulus type',StimulusType,StimulusType.shape)

        Target = data[
            'TargetChar']  # array([ 'EAEVQTDOJG8RBRGONCEDHCTUIDBPUHMEM6OUXOCFOUKWA4VJEFRZROLHYNQDW_EKTLBWXEPOUIKZERYOOTHQI'],4
        # print ('Target char for subjectA',Target)

        return Signal, Flashing, StimulusCode, StimulusType, Target

    else:
        return Signal, Flashing, StimulusCode
# %%
# %% Split training into hold and validation
def create_subset(NtrainIm, NtrainL):
    # Creating a hold out set and remaining set out of the new training set..
    NtrainIm_hold = []
    NtrainL_hold = []
    NtrainIm_rem = []
    NtrainL_rem = []
    # print(int(len(NtrainIm)/10))
    Random = random.sample(range(0, NtrainIm.shape[0]), np.int(0.12 * NtrainIm.shape[0]))
    for k in range(0, NtrainIm.shape[0]):
        if k in Random:
            NtrainIm_hold.append(NtrainIm[k, :])
            NtrainL_hold.append(NtrainL[k])
        else:
            NtrainIm_rem.append(NtrainIm[k, :])
            NtrainL_rem.append(NtrainL[k])
    NtrainIm_hold = np.array(NtrainIm_hold)
    NtrainL_hold = np.array(NtrainL_hold)
    NtrainIm_rem = np.array(NtrainIm_rem)
    NtrainL_rem = np.array(NtrainL_rem)

    return np.array(NtrainIm_rem), np.array(NtrainIm_hold), np.array(NtrainL_rem), np.array(NtrainL_hold)

def format_data(Signal, Flashing, StimulusCode, StimulusType, Target):

    # %%
    """ DEFINE P300 WIndow size"""
    #  FOR downsampled frequency of 120 HZ reduce sizes by 2 everywhere
    window = int(48 / 2)  # take a window to get no of datapoints corresponding to 600 ms after onset of stimuli
    T = int(3 * window)
    char_size = Signal.shape[0]  # 实验个数,85
    responses = np.zeros([char_size, 12, 15, T, 64])

    for epoch in range(0, Signal.shape[0]):
        rowcolcnt = np.zeros(12)
        for n in range(1, Signal.shape[1]):
            # detect location of sample immediately after the stimuli
            if Flashing[epoch, n] == 0 and Flashing[epoch, n - 1] == 1:
                rowcol = int(StimulusCode[epoch, n - 1]) - 1
                # print (Signal[epoch,n:n+window,:].shape)
                #print (rowcolcnt[int(rowcol)])

                responses[epoch, int(rowcol), int(rowcolcnt[int(rowcol)]), :, :] = Signal[epoch,
                                                                                   n - int(window / 2):n + int(
                                                                                       2.5 * window), :]
                rowcolcnt[rowcol] = rowcolcnt[rowcol] + 1
                # print (rowcolcnt)
        # print (epoch)
    print('Response for all characters', responses.shape)

    #####################################################################################################################
    ### Taking average over 15 instances of the 12 stimuli, comment to check performance and increase the dataset size- TO DO
    trainset = np.mean(responses, axis=2)
    print('trainset', trainset.shape)

    P300_dataset = np.zeros([85, 2, T, 64])
    non_P300_dataset = np.zeros([85, 10, T, 64])
    # trainset = responses
    # print('trainset', trainset.shape)


    #####################################################################################################################
    # target_ohe=np.zeros([len(Target[0]),36])
    stimulus_indices = []
    for n_char in range(0, len(Target[0])):  # character epochs

        # print (Target[0][n_char])
        # vv=np.where(screen==str(Target[0][n_char]))
        # print (vv)
        # [row,col]
        for row in range(0, 6):
            for col in range(0, 6):
                # print (screen[row][col])
                if (Target[0][n_char]) is (screen[row][col]):
                    ind = [row + 7, col + 1]
                    stimulus_indices.append(ind)
                    # print (ind)
    # print (len(stimulus_indices))
    print('Splitting P300 and non-P300 dataset...')
    # iterate over the 2nd dimension of trainset:trainset (train_char_size, 12, 42, 64) and split as train_char_size*2*42*64 and train_char_size*10*42*64
    for char_epoch in range(0, trainset.shape[0]):
        # choose the i,j out of the 2nd dimension of trainset where i,j comes from stimulus_indices[char_epoch]
        ind_1 = stimulus_indices[char_epoch][0]
        ind_2 = stimulus_indices[char_epoch][1]
        l = 0
        for index in range(0, 12):
            if index == ind_1 - 1 or index == ind_2 - 1:
                P300_dataset[char_epoch, 0, :, :] = trainset[char_epoch, ind_1 - 1, :, :]
                P300_dataset[char_epoch, 1, :, :] = trainset[char_epoch, ind_2 - 1, :, :]
            else:
                # print ('here')
                # print (index)
                non_P300_dataset[char_epoch, l, :, :] = trainset[char_epoch, index, :, :]
                # targets_A[char_epoch,index]=0
                l = l + 1
    # print (np.all(P300_dataset[0,0,:,:])==np.all(trainset[0,5,:,:]))
    print(P300_dataset.shape)
    print(non_P300_dataset.shape)
    return P300_dataset, non_P300_dataset


def Fdataset(P300_train, P300_hold, non_P300_train, P300_train_label, P300_hold_label, non_P300_train_label,
             non_P300_hold, non_P300_hold_label):
    train_data = np.zeros([P300_train.shape[0] + non_P300_train.shape[0], P300_train.shape[1], P300_train.shape[2]])
    train_label = np.zeros([P300_train_label.shape[0] + non_P300_train_label.shape[0], P300_train_label.shape[1]])
    hold_data = np.zeros([P300_hold.shape[0] + non_P300_hold.shape[0], P300_hold.shape[1], P300_hold.shape[2]])
    hold_label = np.zeros([P300_hold_label.shape[0] + non_P300_hold_label.shape[0], P300_hold_label.shape[1]])

    train_data[0:P300_train.shape[0], :, :] = P300_train[:, :, :]
    train_data[P300_train.shape[0]:P300_train.shape[0] + non_P300_train.shape[0], :, :] = non_P300_train[:, :, :]
    train_label[0:P300_train_label.shape[0], :] = P300_train_label[:, :]
    train_label[P300_train_label.shape[0]:P300_train_label.shape[0] + non_P300_train_label.shape[0],
    :] = non_P300_train_label[:, :]
    train_data, train_label = shuffle(train_data, train_label)

    hold_data[0:P300_hold.shape[0], :, :] = P300_hold[:, :, :]
    hold_data[P300_hold.shape[0]:P300_hold.shape[0] + non_P300_hold.shape[0], :, :] = non_P300_hold[:, :, :]
    hold_label[0:P300_hold_label.shape[0], :] = P300_hold_label[:, :]
    hold_label[P300_hold_label.shape[0]:P300_hold_label.shape[0] + non_P300_hold_label.shape[0],
    :] = non_P300_hold_label[:, :]
    hold_data, hold_label = shuffle(hold_data, hold_label)

    return train_data, train_label, hold_data, hold_label

# %% 打乱数据
def shuffle(trainIm_rem, trainL_rem):
    NtrainIm_hold = []
    NtrainL_hold = []

    R = random.sample(range(0, trainIm_rem.shape[0]), trainIm_rem.shape[0])

    for k in R:
        # print (k)
        NtrainIm_hold.append(trainIm_rem[k, :, :])
        NtrainL_hold.append(trainL_rem[k, :])

    return np.array(NtrainIm_hold), np.array(NtrainL_hold)

def pooling(array, num):
    # length = int(array.shape[1])
    poolarr = np.zeros((array.shape[0], int(np.floor(array.shape[1]/num))))
    for i in range(0, array.shape[0]):
        for j in range(0, int(np.floor(array.shape[1]/num))):
            poolarr[i, j] = array[i, 2*j]
    return poolarr

def balancedata(P300_train_, P300_hold_, P300_train_label_, P300_hold_label_, non_P300_train_, non_P300_train_label_, non_P300_hold_, non_P300_hold_label_):
    #  balance p300 and non-p300
    non_P300_train = np.zeros(P300_train_.shape)
    non_P300_hold = np.zeros(P300_hold_.shape)
    non_P300_train_label = np.zeros(P300_train_label_.shape)
    non_P300_hold_label = np.zeros(P300_hold_label_.shape)
    P300_train = P300_train_
    P300_hold = P300_hold_
    P300_train_label = P300_train_label_
    P300_hold_label = P300_hold_label_
    subprate = int(non_P300_train_.shape[0] / P300_train_.shape[0])
    for i in range(0, P300_train_.shape[0]):
        k = random.randint(1, subprate)
        non_P300_train[i, :, :] = non_P300_train_[i + k, :, :]
        P300_train[i] = preprocessing.scale(P300_train[i], axis=0)
        non_P300_train[i] = preprocessing.scale(non_P300_train[i], axis=0)
        non_P300_train_label[i, :] = non_P300_train_label_[i + k, :]

    for i in range(0, P300_hold_.shape[0]):
        k = random.randint(1, subprate)
        non_P300_hold[i, :, :] = non_P300_hold_[i + k, :, :]
        P300_hold[i] = preprocessing.scale(P300_hold[i], axis=0)
        non_P300_hold[i] = preprocessing.scale(non_P300_hold[i], axis=0)
        non_P300_hold_label[i, :] = non_P300_hold_label_[i + k, :]


    return P300_train, P300_hold, non_P300_train, P300_train_label, P300_hold_label, non_P300_train_label, non_P300_hold, non_P300_hold_label

def makeset():
    #  %% 从bci2003中提取数据
    subject = '0'
    if subject == '0':
        Signal_A_240, Flashing_A_240, StimulusCode_A_240, StimulusType_A_240, Target_A = load_dataset(TRAIN_A, 0)
        # %% 下采样
        """DOWNSAMPLING THE SIGNAL"""
        secs = Signal_A_240.shape[1] / 240  # Number of seconds in signal
        samps = int(secs * 120)  # Number of samples to downsample
        """初始化下采样后的Signal, Flashing, StimulusCode, StimulusType"""
        Signal_A = np.zeros([Signal_A_240.shape[0], samps, 64])
        Flashing_A = np.zeros([Signal_A_240.shape[0], samps])
        StimulusCode_A = np.zeros([Signal_A_240.shape[0], samps])
        StimulusType_A = np.zeros([Signal_A_240.shape[0], samps])
    else:
        Signal_B_240, Flashing_B_240, StimulusCode_B_240, StimulusType_B_240, Target_B = load_dataset(TRAIN_B, 0)
        # %% 下采样
        """DOWNSAMPLING THE SIGNAL"""
        secs = Signal_B_240.shape[1] / 240  # Number of seconds in signal
        samps = int(secs * 120)  # Number of samples to downsample
        """初始化下采样后的Signal, Flashing, StimulusCode, StimulusType"""
        Signal_B = np.zeros([Signal_B_240.shape[0], samps, 64])
        Flashing_B = np.zeros([Signal_B_240.shape[0], samps])
        StimulusCode_B = np.zeros([Signal_B_240.shape[0], samps])
        StimulusType_B = np.zeros([Signal_B_240.shape[0], samps])

    """从原始数据以samps的采样频率填充"""
    for i in range(0, Signal_A_240.shape[0]):
        if subject == '0':
            Signal_A[i, :, :] = scipy.signal.resample(Signal_A_240[i, :, :], int(samps))
            Flashing_A[i, :] = abs(np.round(scipy.signal.resample(Flashing_A_240[i, :], int(samps))))
            StimulusCode_A[i, :] = abs(np.floor(scipy.signal.resample(StimulusCode_A_240[i, :], int(samps)))).astype(
                'int8')
            StimulusType_A[i, :] = abs(np.floor(scipy.signal.resample(StimulusType_A_240[i, :], int(samps))))

        else:
            Signal_B[i, :, :] = scipy.signal.resample(Signal_B_240[i, :, :], int(samps))
            Flashing_B[i, :] = abs(np.round(scipy.signal.resample(Flashing_B_240[i, :], int(samps))))
            StimulusCode_B[i, :] = abs(np.floor(scipy.signal.resample(StimulusCode_B_240[i, :], int(samps))))
            StimulusType_B[i, :] = abs(np.floor(scipy.signal.resample(StimulusType_B_240[i, :], int(samps))))


    #      TO DO :
    #      REPEAT CODE IN THIS CELL FOR TRAINING SUBJECT B, ignore variables named as A for now
    #      this for loop is jsut for printing, either make a function or copy paste to get B data

    if subject == '0':
        print('For subject A')
        train_char_size = Signal_A.shape[0]  # train char size: 85
        P300_dataset_, non_P300_dataset_ = format_data(Signal_A, Flashing_A, StimulusCode_A, StimulusType_A, Target_A)
        # %%train_char_size

    else:
        print('For subject B')
        train_char_size = Signal_B.shape[0]  # train char size: 85
        P300_dataset_, non_P300_dataset_ = format_data(Signal_B, Flashing_B, StimulusCode_B, StimulusType_B, Target_B)
        # %%train_char_size

    targets_ = np.zeros([train_char_size * 15, 12])
    targets_[:, 0:2] = 1
    targets_[:, 2:12] = 0

    ## Create training and validation subset
    P300_train_, P300_hold_, P300_train_label_, P300_hold_label_ = create_subset(
        np.reshape(P300_dataset_, [train_char_size * 2, 72, 64]),
        np.reshape(targets_[:, 0:2], [train_char_size * 2 * 15, 1]))

    non_P300_train_, non_P300_hold_, non_P300_train_label_, non_P300_hold_label_ = create_subset(
        np.reshape(non_P300_dataset_, [train_char_size * 10, 72, 64]),
        np.reshape(targets_[:, 2:12], [train_char_size * 10 * 15, 1]))

    P300_train, P300_hold, non_P300_train, P300_train_label, P300_hold_label, non_P300_train_label, non_P300_hold, non_P300_hold_label = balancedata(
        P300_train_, P300_hold_, P300_train_label_, P300_hold_label_, non_P300_train_, non_P300_train_label_,
        non_P300_hold_, non_P300_hold_label_)

    train_data, train_label, hold_data, hold_label = Fdataset(
        P300_train, P300_hold, non_P300_train, P300_train_label, P300_hold_label, non_P300_train_label, non_P300_hold,
        non_P300_hold_label)
    return train_data, train_label, hold_data, hold_label


def main():
    train_data, train_label, hold_data, hold_label = makeset()
    # %%
    dataset_train = {};
    dataset_train['data'] = train_data
    print('train data size', train_data.shape)
    dataset_train['label'] = train_label
    print('train label size', train_label.shape)
    dataset_hold = {};
    dataset_hold['data'] = hold_data
    print('vail data size', hold_data.shape)
    dataset_hold['label'] = hold_label
    print('vail data size', hold_label.shape)

    scipy.io.savemat('TrainDataset_ERP.mat', dataset_train)
    scipy.io.savemat('HoldDataset_ERP.mat', dataset_hold)


if __name__ == '__main__':
    main()
    # arr = np.random.rand(3, 5)
    # print(arr)
    # pool = pooling(arr, 2)
    # print(pool)







