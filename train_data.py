import mne
import numpy as np
from sklearn.preprocessing import StandardScaler


def read_mark_txt(file_dir):
    """
    :读取.vmrk文件中mark信息
    :file_dir: 是.vmrk文件的绝对路径

    :返回
    :mark: 点击数字时发送到脑电帽的一个mark标记
    :point: 代表点击数字时已经记录了多少次脑电信号
    """
    mark = []
    point = []
    with open(file_dir, 'r') as f:
        for line in f:
            if line[0:2] == 'Mk':
                mark.append(int(line[3:].split(',')[1]))
                point.append(int(line[3:].split(',')[2]))

    return np.array(mark), np.array(point)


channels_index = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 'FC6', 'T7', 'T8', 'C3'
                    , 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1',
                      'O2']

LENGTH = 750  # 截取1.5s的数据段
WINDOWS = 150  # 滑动窗口长度为300ms
STRIDE = 75  # 步长为150ms

if __name__ == '__main__':
    set_file_dir = fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\preprocess\yangyifeng0808.set'
    raw = mne.io.read_raw_eeglab(set_file_dir, preload=True)
    raw_eeg = raw.get_data()

    print('raw_eeg.shape: ', raw_eeg.shape)

    mark, point = read_mark_txt(
        fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\original_record\yangyifeng0808.vmrk')
    print('mark.shape: ', mark.shape, 'point.shape: ', point.shape)

    rest_eeg = raw_eeg[:, 1:264894]
    test_eeg = raw_eeg[:, 264894:401749]

    print('rest_eeg.shape: ', rest_eeg.shape)
    print('test_eeg.shape: ', test_eeg.shape)

    norm_sequence = []
    abnorm_sequence = []

    for i in np.arange(len(mark) - 1):
        if mark[i + 1] != 0 and (point[i + 1] - point[i]) >= LENGTH:
            seq = raw_eeg[:, point[i+1] - LENGTH:point[i+1]]  # 点击时长超过1.5s的倒截取1.5s作为数据集
            for j in np.arange(int((seq.shape[1] - WINDOWS) / STRIDE) + 2):
                start = j * STRIDE
                end = (j + 2) * STRIDE
                if end < len(seq):
                    seq_ = StandardScaler().fit_transform(seq[:, start:end].T).T
                    norm_sequence.append(seq_)
                    abnorm_seq = seq_
                    abnorm_seq = np.hstack((abnorm_seq[:, STRIDE:], abnorm_seq[:, :STRIDE]))
                    abnorm_sequence.append(abnorm_seq)
                else:
                    seq_ = StandardScaler().fit_transform(seq[:, -WINDOWS:].T).T
                    norm_sequence.append(seq_)
                    abnorm_seq = seq_
                    abnorm_seq = np.hstack((abnorm_seq[:, STRIDE:], abnorm_seq[:, :STRIDE]))
                    abnorm_sequence.append(abnorm_seq)

    norm_sequence = np.array(norm_sequence)
    abnorm_sequence = np.array(abnorm_sequence)
    print('norm_sequence.shape: ', norm_sequence.shape, 'abnorm_sequence.shape: ', abnorm_sequence.shape)

    np.save(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\norm_sequence.npy', norm_sequence)
    np.save(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\abnorm_sequence.npy', abnorm_sequence)