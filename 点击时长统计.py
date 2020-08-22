import numpy as np
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


mark, point = read_mark_txt(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\original_record\yangyifeng0808.vmrk')

time = []
for i in np.arange(len(mark) - 1):
    if mark[i+1] != 0:
        time.append(point[i+1] - point[i])

time = np.array(time)
time = time / 500
min_time = np.min(time) * 1000

plt.plot(time, 'o')
plt.ylabel('s')
plt.xlabel(f'min time is {min_time}ms')
plt.title('YYF')
plt.show()