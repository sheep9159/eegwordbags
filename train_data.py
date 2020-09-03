import mne
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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


def get_file_name(file_dir, file_type):
    """
    :遍历指定目录下的所有指定类型的数据文件
    :file_dir: 如有一目录下包含.eeg原始数据文件，.vhdr文件(含mark)和.vmrk文件，便可获得指定文件的完整路径
    :file_type: 指定需要找到的文件类型

    :返回
    :file_names: 指定文件的绝对路径
    """

    file_names = []

    for root, dirs, files in os.walk(file_dir, topdown=False):
        for file in files:
            if file_type in file:
                file_names.append(os.path.join(root, file))

    return file_names


channels_index = ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F8', 'F3', 'Fz', 'F4', 'FC5', 'FC6', 'T7', 'T8', 'C3'
                    , 'C4', 'CP5', 'CP6', 'P7', 'P8', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'PO3', 'PO4', 'O1',
                      'O2']


if __name__ == '__main__':
    name = 'weichuanxiang'
    set_file_dir = get_file_name(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\preprocess', '.set')
    vmrk_file_dir = get_file_name(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\segmentation', '.vmrk')

    for index, file in enumerate(set_file_dir):
        if not name in file: continue
        raw = mne.io.read_raw_eeglab(file, preload=True)
        raw_eeg = raw.get_data()

        print('raw_eeg.shape: ', raw_eeg.shape)

        print(vmrk_file_dir[index])
        mark, point = read_mark_txt(vmrk_file_dir[index])
        print('mark.shape: ', mark.shape, 'point.shape: ', point.shape)

        for i in np.arange(len(mark) - 1):
            if mark[i + 1] != 0 and (point[i+1] - point[i]) >= 150:
                seq = raw_eeg[:, point[i]:point[i+1]]
                np.save(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\train_data\{name}\{index}_{i}.npy', seq)

        print(f'{file} done!')

