import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from deepcluster import DeepCluster, LENGTH, num_cluster, name
import matplotlib.cm as cm
import matplotlib.pyplot as plt


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

def keshihua():
    deepcluster = DeepCluster(train=False)
    deepcluster.load_state_dict(torch.load(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\模型\{name}-deepcluster.pt'))
    deepcluster.eval()

    sample = np.zeros(shape=(1, 28))
    order = np.zeros(0)

    data_dir = fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\train_data\{name}'
    data_dir = get_file_name(data_dir, 'npy')
    for dir in data_dir:
        data = np.load(dir).transpose((1, 0))
        l = len(data)
        order = np.hstack((order, np.arange(LENGTH)[-l:]))
        data = torch.from_numpy(data).float()
        data = F.normalize(data, dim=0)
        data_ = data.new_full((LENGTH, 28), float('-inf'))
        data_[:data.size()[0], :] = data[-LENGTH:, :]
        print(data_.size())
        feature = deepcluster(data_.unsqueeze(1))
        feature = feature.squeeze(1).detach().numpy()
        sample = np.vstack((sample, feature[:l, :]))
        # sample = np.vstack((sample, data[-LENGTH:, :]))
        print(order.shape)
        print(sample.shape)

    sample = sample[1:, :]
    print(order.shape)
    print((order[20000:20030]))
    print(sample.shape)

    tsne = TSNE(n_components=num_cluster, n_jobs=-1)
    low_dim_sample = tsne.fit_transform(sample)
    np.save(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\模型\{name}-low_dim_sample.npy', low_dim_sample)
    # low_dim_sample = np.load(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\模型\low_dim_sample.npy')
    plt.scatter(low_dim_sample[:, 0], low_dim_sample[:, 1], s=0.5, c=order, cmap=plt.cm.get_cmap("seismic"))

    plt.show()

if __name__ == '__main__':
    keshihua()




