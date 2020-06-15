# %%
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
TRAIN_A = 'data/BCI_Comp_III_Wads_2004/Subject_A_Train1_reshape.mat'
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

    # print ('Subject A dataa',data)
    Signal = np.float32(data['Signal'])
    # print ('signal',Signal, Signal.shape)

    Flashing = np.float32(data['Flashing'])
    # print ('flashing',Flashing, Flashing.shape)

    StimulusCode = np.float32(data['StimulusCode'])
    # print ('Stimulus COde',StimulusCode,StimulusCode.shape)
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
    # 把所有下采样重新做一遍，可选择是否下采样
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
    char_size = Signal.shape[0]  # 实验个数,85
    responses = np.zeros([char_size, 12, 15, T, 64])

    for epoch in range(0, Signal.shape[0]):
        count = 1;
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

    P300_dataset = np.zeros([train_char_size, 2, T, 64])
    non_P300_dataset = np.zeros([train_char_size, 10, T, 64])

    for char_epoch in range(0, trainset.shape[0]):
        # choose the i,j out of the 2nd dimension of trainset where i,j comes from stimulus_indices[char_epoch]
        ind_1 = stimulus_indices[char_epoch][0]
        ind_2 = stimulus_indices[char_epoch][1]
        # print (ind_1,ind_2)
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
    '''
    for p in range(0,2):
        print ('next P300')
        for n_p in range(0,10):
            plt.figure()
            #plt.subplot(121)
            plt.plot(P300_dataset[4,p,:,5].T)
            #plt.figure()
            #plt.subplot(122)
            plt.plot(non_P300_dataset[4,n_p,:,5].T)
            plt.show()  
    '''
    return P300_dataset, non_P300_dataset
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

# AGAIN IGNORE THE NAMING CONVENTION WHICH IS BY DEFAULT A
def choose_subsection_non_P300(a, b, c, d, P300_train, non_P300_train, P300_hold, non_P300_hold):
    """non_P300_train[a:b] """
    print(a, b, c, d)
    t1 = 2 * P300_train.shape[0]  # + non_P300_train_A.shape[0] #909

    t2 = P300_train.shape[0]  # 152

    if b == non_P300_train.shape[0]:
        t1 = t1 - 2

    dataset_A_train = np.zeros([t1, T, 64])
    dataset_A_train[0:t2, :, :] = P300_train

    dataset_A_train[t2:t1, :, :] = non_P300_train[a:b, :, :]

    targets_A_train = np.zeros([t1, 1])
    targets_A_train[0:t2, :] = 1

    h1 = 2 * P300_hold.shape[0]  # + non_P300_hold_label_A.shape[0] #111
    h2 = P300_hold.shape[0]  # 18

    if d == non_P300_hold.shape[0]:
        h1 = h1 + 2

    dataset_A_hold = np.zeros([h1, T, 64])
    dataset_A_hold[0:h2, :, :] = P300_hold

    dataset_A_hold[h2:h1:, :] = non_P300_hold[c:d, :, :]
    targets_A_hold = np.zeros([h1, 1])
    targets_A_hold[0:h2, :] = 1

    # print(targets_A_hold.shape)

    ### SHUFFLE ABOVE DATASET and LABELS
    print('-------------------------HOLD AND TRAIN DATASET_CNN READY----------------------------------------------')
    dataset_A_hold, targets_A_hold = shuffle(dataset_A_hold, targets_A_hold)
    for i in range(0, dataset_A_hold.shape[0]):
        dataset_A_hold[i] = preprocessing.scale(dataset_A_hold[i], axis=0)
        # print('mean:', dataset_A_hold[i].mean(axis=0), 'std:', dataset_A_hold[i].std(axis=0))

    print(dataset_A_hold.shape, targets_A_hold.shape)

    dataset_A_train, targets_A_train = shuffle(dataset_A_train, targets_A_train)
    for i in range(0, dataset_A_hold.shape[0]):
        dataset_A_train[i] = preprocessing.scale(dataset_A_train[i], axis=0)
        # print('mean:', dataset_A_train[i].mean(axis=0), 'std:', dataset_A_train[i].std(axis=0))

    print(dataset_A_train.shape, targets_A_train.shape)
    print('--------------------------------------------------------------------------------------------------------')
    ################################

    return dataset_A_train, targets_A_train, dataset_A_hold, targets_A_hold

def create_balanced_dataset(P300_train_A, non_P300_train_A, P300_hold_A, non_P300_hold_A):
    #######################################################
    # Combine the dataset:
    # Part 1:
    a = 0;
    b = int(np.ceil(non_P300_train_A.shape[0] / 5));
    c = 0;
    d = int(non_P300_hold_A.shape[0] / 5)
    dataset_A_train_1, targets_A_train_1, dataset_A_hold_1, targets_A_hold_1 = choose_subsection_non_P300(a, b, c, d,
                                                                                                          P300_train_A,
                                                                                                          non_P300_train_A,
                                                                                                          P300_hold_A,
                                                                                                          non_P300_hold_A)

    # Part 2:
    a = int(np.ceil(non_P300_train_A.shape[0] / 5));
    b = 2 * int(np.ceil(non_P300_train_A.shape[0] / 5))
    c = int(non_P300_hold_A.shape[0] / 5);
    d = 2 * int(non_P300_hold_A.shape[0] / 5)
    dataset_A_train_2, targets_A_train_2, dataset_A_hold_2, targets_A_hold_2 = choose_subsection_non_P300(a, b, c, d,
                                                                                                          P300_train_A,
                                                                                                          non_P300_train_A,
                                                                                                          P300_hold_A,
                                                                                                          non_P300_hold_A)

    # Part 3:
    a = 2 * int(np.ceil(non_P300_train_A.shape[0] / 5));
    b = 3 * int(np.ceil(non_P300_train_A.shape[0] / 5))
    c = 2 * int(non_P300_hold_A.shape[0] / 5);
    d = 3 * int(non_P300_hold_A.shape[0] / 5)
    dataset_A_train_3, targets_A_train_3, dataset_A_hold_3, targets_A_hold_3 = choose_subsection_non_P300(a, b, c, d,
                                                                                                          P300_train_A,
                                                                                                          non_P300_train_A,
                                                                                                          P300_hold_A,
                                                                                                          non_P300_hold_A)

    # Part 4:
    a = 3 * int(np.ceil(non_P300_train_A.shape[0] / 5));
    b = 4 * int(np.ceil(non_P300_train_A.shape[0] / 5))
    c = 3 * int(non_P300_hold_A.shape[0] / 5);
    d = 4 * int(non_P300_hold_A.shape[0] / 5)
    dataset_A_train_4, targets_A_train_4, dataset_A_hold_4, targets_A_hold_4 = choose_subsection_non_P300(a, b, c, d,
                                                                                                          P300_train_A,
                                                                                                          non_P300_train_A,
                                                                                                          P300_hold_A,
                                                                                                          non_P300_hold_A)

    # Part 5:
    a = 4 * int(np.ceil(non_P300_train_A.shape[0] / 5));
    b = 5 * int(np.ceil(non_P300_train_A.shape[0] / 5)) - 2
    c = 4 * int(non_P300_hold_A.shape[0] / 5);
    d = 5 * int(non_P300_hold_A.shape[0] / 5) + 2
    dataset_A_train_5, targets_A_train_5, dataset_A_hold_5, targets_A_hold_5 = choose_subsection_non_P300(a, b, c, d,
                                                                                                          P300_train_A,
                                                                                                          non_P300_train_A,
                                                                                                          P300_hold_A,
                                                                                                          non_P300_hold_A)

    return dataset_A_train_1, targets_A_train_1, dataset_A_hold_1, targets_A_hold_1, dataset_A_train_2, targets_A_train_2, dataset_A_hold_2, targets_A_hold_2, dataset_A_train_3, targets_A_train_3, dataset_A_hold_3, targets_A_hold_3, dataset_A_train_4, targets_A_train_4, dataset_A_hold_4, targets_A_hold_4, dataset_A_train_5, targets_A_train_5, dataset_A_hold_5, targets_A_hold_5





# %%
Signal_A_240, Flashing_A_240, StimulusCode_A_240, StimulusType_A_240, Target_A = load_dataset(TRAIN_A, 0)
print('Signal A shape', Signal_A_240.shape)
#############################################################################################
Signal_B_240, Flashing_B_240, StimulusCode_B_240, StimulusType_B_240, Target_B = load_dataset(TRAIN_B, 0)

# %%
"""DOWNSAMPLING THE SIGNAL"""

secs = Signal_A_240.shape[1] / 240  # Number of seconds in signal
samps = int(secs * 120)  # Number of samples to downsample
#%%初始化下采样后的Signal, Flashing, StimulusCode, StimulusType
Signal_A = np.zeros([Signal_A_240.shape[0], samps, 64])
Flashing_A = np.zeros([Signal_A_240.shape[0], samps])
StimulusCode_A = np.zeros([Signal_A_240.shape[0], samps])
StimulusType_A = np.zeros([Signal_A_240.shape[0], samps])

Signal_B = np.zeros([Signal_B_240.shape[0], samps, 64])
Flashing_B = np.zeros([Signal_B_240.shape[0], samps])
StimulusCode_B = np.zeros([Signal_B_240.shape[0], samps])
StimulusType_B = np.zeros([Signal_B_240.shape[0], samps])

#%%从原始数据以samps的采样频率填充
for i in range(0, Signal_A_240.shape[0]):
    Signal_A[i, :, :] = scipy.signal.resample(Signal_A_240[i, :, :], int(samps))
    Signal_B[i, :, :] = scipy.signal.resample(Signal_B_240[i, :, :], int(samps))
    # print (Flashing_A_240[i,:],Flashing_A_240[i,:].shape)
    Flashing_A[i, :] = abs(np.round(scipy.signal.resample(Flashing_A_240[i, :], int(samps))))
    # print (Flashing_A[i,:],Flashing_A[i,:].shape)
    StimulusCode_A[i, :] = abs(np.floor(scipy.signal.resample(StimulusCode_A_240[i, :], int(samps)))).astype('int8')
    # print (StimulusCode_A[i,:])
    StimulusType_A[i, :] = abs(np.floor(scipy.signal.resample(StimulusType_A_240[i, :], int(samps))))
    # print (StimulusType_A[i,:])

    Flashing_B[i, :] = abs(np.round(scipy.signal.resample(Flashing_B_240[i, :], int(samps))))
    StimulusCode_B[i, :] = abs(np.floor(scipy.signal.resample(StimulusCode_B_240[i, :], int(samps))))
    StimulusType_B[i, :] = abs(np.floor(scipy.signal.resample(StimulusType_B_240[i, :], int(samps))))

# %%
""" DEFINE P300 WIndow size"""
#  FOR downsampled frequency of 120 HZ reduce sizes by 2 everywhere
window = int(48 / 2)  # take a window to get no of datapoints corresponding to 600 ms after onset of stimuli
T = int(3 * window)

# %%train_char_size
train_char_size = Signal_A.shape[0]
## TO DO :
### REPEAT CODE IN THIS CELL FOR TRAINING SUBJECT B, ignore variables named as A for now
# this for loop is jsut for printing, either make a function or copy paste to get B data

subject = '0'  # raw_input('Enter subject you want to train(A: 0 or B:1 ):  ')
if subject == '0':
    print('For subject A')
    P300_dataset_, non_P300_dataset_ = format_data(Signal_A, Flashing_A, StimulusCode_A, StimulusType_A, Target_A)
else:
    print('For subject B')
    P300_dataset_, non_P300_dataset_ = format_data(Signal_B, Flashing_B, StimulusCode_B, StimulusType_B, Target_B)

targets_ = np.zeros([train_char_size * 15, 12])
targets_[:, 0:2] = 1
targets_[:, 2:12] = 0

## Create training and validation subset
P300_train_, P300_hold_, P300_train_label_, P300_hold_label_ = create_subset(
    np.reshape(P300_dataset_, [train_char_size * 2, T, 64]),
    np.reshape(targets_[:, 0:2], [train_char_size * 2 * 15, 1]))

non_P300_train_, non_P300_hold_, non_P300_train_label_, non_P300_hold_label_ = create_subset(
    np.reshape(non_P300_dataset_, [train_char_size * 10, T, 64]),
    np.reshape(targets_[:, 2:12], [train_char_size * 10 * 15, 1]))
# print('-----------------------------------------------------------------------------------------------------------')
#
# print('Training set of P300 samples')
# print(P300_train_.shape, P300_train_label_.shape)
#
# print('\nValidation set of P300 samples')
# print(P300_hold_.shape, P300_hold_label_.shape)
#
# print('\nTraining set of non-P300 samples')
# print(non_P300_train_.shape, non_P300_train_label_.shape)
#
# print('\nValidation set of non- P300 samples')
# print(non_P300_hold_.shape, non_P300_hold_label_.shape)
#
# print('-----------------------------------------------------------------------------------------------------------')
# print('-----------------------------------------------------------------------------------------------------------')

dataset_A_train_1, targets_A_train_1, dataset_A_hold_1, targets_A_hold_1, dataset_A_train_2, targets_A_train_2, dataset_A_hold_2, targets_A_hold_2, dataset_A_train_3, targets_A_train_3, dataset_A_hold_3, targets_A_hold_3, dataset_A_train_4, targets_A_train_4, dataset_A_hold_4, targets_A_hold_4, dataset_A_train_5, targets_A_train_5, dataset_A_hold_5, targets_A_hold_5 = create_balanced_dataset(
    P300_train_, non_P300_train_, P300_hold_, non_P300_hold_)




A_train={};
A_train['Tdata1'] = dataset_A_train_1
A_train['Vdata1'] = dataset_A_hold_1
A_train['Tdata2'] = dataset_A_train_2
A_train['Vdata2'] = dataset_A_hold_2
A_train['Tdata3'] = dataset_A_train_3
A_train['Vdata3'] = dataset_A_hold_3
A_train['Tdata4'] = dataset_A_train_4
A_train['Vdata4'] = dataset_A_hold_4
A_train['Tdata5'] = dataset_A_train_5
A_train['Vdata5'] = dataset_A_hold_5
A_label={};
A_label['Tlabel1'] = targets_A_train_1
A_label['Vlabel1'] = targets_A_hold_1
A_label['Tlabel2'] = targets_A_train_2
A_label['Vlabel2'] = targets_A_hold_2
A_label['Tlabel3'] = targets_A_train_3
A_label['Vlabel3'] = targets_A_hold_3
A_label['Tlabel4'] = targets_A_train_4
A_label['Vlabel4'] = targets_A_hold_4
A_label['Tlabel5'] = targets_A_train_5
A_label['Vlabel5'] = targets_A_hold_5

scipy.io.savemat('G:\EEGNet\data\EEGDataset\ERP\TrainDatas/Circle/dataset_A_train.mat', A_train)
scipy.io.savemat('G:\EEGNet\data\EEGDataset\ERP\TrainDatas/Circle/dataset_A_label.mat', A_label)

