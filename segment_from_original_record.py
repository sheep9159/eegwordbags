import os
import struct


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


def read_eeg_file(eeg_file_dir, segmentation_dir, point_boundary, name_index, trail_indices):
    for trail_index in range(trail_indices):
        with open(eeg_file_dir, 'rb') as f:
            raw = f.read()
            raw_eeg = struct.unpack('{}f'.format(int(len(raw) / 4)), raw)
            data = raw_eeg[point_boundary[trail_index * 2] * 35: (point_boundary[trail_index * 2 + 1] + 1) * 35]
            data = struct.pack('{}f'.format(int(len(data))), *data)

        current_name_index = name_index[trail_index]

        if not os.path.exists(segmentation_dir): os.makedirs(segmentation_dir)
        f = open(fr'{segmentation_dir}\{current_name_index}.eeg', 'wb')
        f.write(data)
        f.close()


def read_vmrk_txt(vmrk_file_dir, segmentation_dir):
    mark = []
    point = []
    with open(vmrk_file_dir, 'r') as f:  # 把所有的mark与相应的point存储下来
        for line in f:
            if line[0:2] == 'Mk':
                mark.append(int(line[3:].split(',')[1]))
                point.append(int(line[3:].split(',')[2]))

    point_boundary = []  # 获取每个epoch的start与end时对应的point值
    for j in range(len(mark) - 1):
        if mark[j] == 0:  # epoch的start
            point_boundary.append(point[j])
        elif mark[j + 1] == 0:  # epoch的end
            point_boundary.append(point[j])

    point_boundary.append((point[-1]))  # 添加最后一个point作为最后一个epoch的end，因为上述循环没有遍历到最后一个边界
    point_boundary = point_boundary[1:]  # 去除第一个point，因为vmrk文件中的Mk1是无效的
    trail_indices = int(len(point_boundary) / 2)

    row = 0  # 用来记录读到第几行了
    txt = ''  # 先创建一个空字符串，用来存待会改写后的文本
    name_index = []  # 用于记录当前正在处理被试者的第几段子试验
    datas_length = []  # 用于记录每个子试验采集数据个数即points

    for trail_index in range(trail_indices):  # 为新的segmentation写新的vmrk文件
        with open(vmrk_file_dir, 'r') as f:
            for line in f:
                if row <= 5 and row != 3:
                    txt = txt + line
                if row == 3:
                    current_name_index = line[9:-5] + '_{0:0>2}'.format(trail_index + 1)  # 控制编号以_01， _02，形式
                    txt = txt + line[:9] + current_name_index + '.eeg' + '\n'
                row += 1
            row = 0
            trail_point = [i for i in point if i >= point_boundary[trail_index*2] and i <= point_boundary[trail_index*2+1]]
            for mk in range(len(trail_point)):
                txt = txt + 'Mk' + str(mk+1) + '=Stimulus,' + str(mk+1) + ',' + str(trail_point[mk]-trail_point[0]+1) + ',0,' + '\n'

        name_index.append(current_name_index)
        datas_length.append(trail_point[mk] - trail_point[0] + 1)

        if not os.path.exists(segmentation_dir): os.makedirs(segmentation_dir)
        f = open(fr'{segmentation_dir}\{current_name_index}.vmrk', 'w')
        f.write(txt)
        f.close()
        txt = ''

    return point_boundary, name_index, trail_indices, datas_length # point_boundary用于原始.eeg文件分段，trail_indices和datas_length用于改写.vhdr文件


def read_vhdr_file(vhdr_file_dir, segmentation_dir, name_index, trail_indices, datas_length):
    row = 0  # 用来记录读到第几行了
    txt = ''  # 先创建一个空字符串，用来存待会改写后的文本
    for trail_index in range(trail_indices):
        current_name_index = name_index[trail_index]
        with open(vhdr_file_dir, 'r') as f:
            for line in f:
                if row == 4:
                    txt = txt + line[:9] + current_name_index + '.eeg' + '\n'
                elif row == 5:
                    txt = txt + line[:11] + current_name_index + '.vmrk' + '\n'
                elif row == 10:
                    txt = txt + line[:11] + str(datas_length[trail_index]) + '\n'
                else:
                    txt = txt + line
                row += 1
            row = 0

        if not os.path.exists(segmentation_dir): os.makedirs(segmentation_dir)
        f = open(fr'{segmentation_dir}\{name_index[trail_index]}.vhdr', 'w')
        f.write(txt)
        f.close()
        txt = ''


if __name__ == '__main__':
    raw_eeg_dir = fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\original_record'
    segmentation_dir = fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\segmentation'

    mark_file_name = get_file_name(raw_eeg_dir, '.vmrk')
    vhdr_file_name = get_file_name(raw_eeg_dir, '.vhdr')
    eeg_file_name = get_file_name(raw_eeg_dir, '.eeg')

    for i in range(len(eeg_file_name)):

        point_boundary, name_index, trail_indices, datas_length = read_vmrk_txt(mark_file_name[i], segmentation_dir)
        read_vhdr_file(vhdr_file_name[i], segmentation_dir, name_index, trail_indices, datas_length)
        read_eeg_file(eeg_file_name[i], segmentation_dir, point_boundary, name_index, trail_indices)
