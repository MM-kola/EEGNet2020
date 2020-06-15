import numpy as np
import scipy
import scipy.io
import scipy.signal
import random
from scipy import signal
from sklearn import preprocessing

def load_dataset(SUBJECT, flag):
    """flag标志是否输出target"""
    data = scipy.io.loadmat(SUBJECT)
    print ('Subject A dataa',data)
    Signal = np.float32(data['Signal'])
    print('signal',Signal, Signal.shape)
    Flashing = np.float32(data['Flashing'])
    print('flashing',Flashing, Flashing.shape)
    StimulusCode = np.float32(data['StimulusCode'])
    print('Stimulus COde',StimulusCode,StimulusCode.shape)
    if flag == 0:
        StimulusType = np.float32(data['StimulusType'])
        # print ('Stimulus type',StimulusType,StimulusType.shape)

        Target = data[
            'TargetChar']  # array([ 'EAEVQTDOJG8RBRGONCEDHCTUIDBPUHMEM6OUXOCFOUKWA4VJEFRZROLHYNQDW_EKTLBWXEPOUIKZERYOOTHQI'],4
        # print ('Target char for subjectA',Target)

        return Signal, Flashing, StimulusCode, StimulusType, Target

    else:
        return Signal, Flashing, StimulusCode


def pooling(array, num):
    # length = int(array.shape[1])
    poolarr = np.zeros((array.shape[0], int(np.floor(array.shape[1]/num))))
    for i in range(0, array.shape[0]):
        for j in range(0, int(np.floor(array.shape[1]/num))):
            poolarr[i, j] = array[i, 2*j]
    return poolarr


def samp():
    return 0


def slipe():
    WIN = 200

    TRAIN_A = 'data/BCI_Comp_III_Wads_2004/Subject_A_Train1.mat'
    Signal_A_240, Flashing_A_240, StimulusCode_A_240, StimulusType_A_240, Target_A = load_dataset(TRAIN_A, 0)
    echo = Signal_A_240.shape[0]
    timeLength = Signal_A_240.shape[1]
    channel = Signal_A_240.shape[2]
    trigger = np.zeros(timeLength)
    last = 0
    p300set = list()
    nonp300set = list()
    for i in range(0, echo):
        for j in range(0, timeLength):
            if i == 0:
                if Flashing_A_240[0, j] - last == 1:
                    trigger[j] = True
                else:
                    trigger[j] = False
                last = Flashing_A_240[0, j]
            if trigger[j]:
                if StimulusType_A_240[i, j] == 1:
                    p300set.append(Signal_A_240[i, j:j+WIN])
                if StimulusType_A_240[i, j] == 0:
                    nonp300set.append(Signal_A_240[i, j:j+WIN])
    p300len = len(p300set)
    set_arr = np.zeros((p300len*2, WIN, channel))
    sum = p300len * 2
    holdnum = int(sum * 0.1)
    trainnum = int(sum * 0.9)
    train_data = np.zeros((trainnum, WIN, channel))
    train_label = np.zeros((trainnum, 1))
    hold_data = np.zeros((holdnum, WIN, channel))
    hold_label = np.zeros((holdnum, 1))
    label = np.zeros((sum, 1))
    for i in range(0, p300len):
        set_arr[i, :, :] = p300set.pop(p300len-1-i)[:, :]
        label[i, 0] = True
        set_arr[i*2, :, :] = nonp300set.pop(random.randint(0, len(nonp300set)-1-i))[:, :]
    permut = np.random.permutation(sum)
    shuffle_set = set_arr[permut, :]
    shuffle_label = label[permut, :]

    train_data[:, :, :] = shuffle_set[0:trainnum, :, :]
    train_label[:, 0] = shuffle_label[0:trainnum, 0]
    hold_data[:, :, :] = shuffle_set[trainnum:sum, :, :]
    hold_label[:, 0] = shuffle_label[trainnum:sum, 0]
    dataset_train = {};
    dataset_train['data'] = train_data
    dataset_train['label'] = train_label
    dataset_hold = {};
    dataset_hold['data'] = hold_data
    dataset_hold['label'] = hold_label

    scipy.io.savemat('G:\EEGNet\data\EEGDataset\SingleTaril\TrainDatas/Normal\TrainDataset_new.mat', dataset_train)
    scipy.io.savemat('G:\EEGNet\data\EEGDataset\SingleTaril\TrainDatas/Normal\HoldDataset_new.mat', dataset_hold)
    print('x')


if __name__ == '__main__':
    slipe()
