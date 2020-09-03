import os
import numpy as np
import matplotlib.pyplot as plt


def get_file_name(file_dir, file_type):
    """
    :遍历指定目录下的所有指定类型的数据文件
    :file_dir: 此目录下包含.eeg原始数据文件，.vhdr文件(含mark)和.vmrk文件
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


vmrk_files = get_file_name(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\original_record', '.vmrk')
INDEX = [4, 10]

for index, file in enumerate(vmrk_files):
    mark, point = read_mark_txt(file)

    time1 = []
    time2 = []
    T = []
    for i in np.arange(len(mark) - 1):
        if mark[i+1] != 0:
            time1.append(point[i+1] - point[i])
            time2.append(point[i+1] - point[i])
        elif time2:
            T.append(np.mean(np.array(time2)))
            time2 = []
    T.append(np.mean(np.array(time2)))

    T = np.array(T)
    time1 = np.array(time1)
    T = T / 500
    time1 = time1 / 500
    min_time = np.min(time1) * 1000

    plt.figure(figsize=(8, 7))
    plt.subplot(2, 1, 1)
    plt.plot(T, 'o-')
    plt.ylabel('s')
    plt.xlabel(f'test order')
    plt.title(f'Average time for 6 groups of clicks', color='blue')

    plt.subplot(2, 1, 2)
    plt.plot(time1, 'o')
    plt.ylabel('s')
    plt.xlabel(f'test order')
    plt.title(f's{INDEX[index]}--min time is {min_time}ms', color='blue')
    plt.tight_layout()
    plt.savefig(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\点击时长统计\s{INDEX[index]}')
    plt.clf()
    plt.close()