import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn


EPOCH = 1000           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 16
LR = 0.001           # learning rate

LENGTH = 600
mask_probability = 0.2
num_cluster = 2

name = 'gaoyang'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class myBertDatasets(torch.utils.data.Dataset):
    def __init__(self):
        super(myBertDatasets).__init__()
        data_dir = fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\train_data\{name}'
        self.data = get_file_name(data_dir, '.npy')

    def __getitem__(self, item):  # 从PC中读取数据，并且，时间轴全部以用-inf填充的方式pad到同样的长度LENGTH
        data = np.load(self.data[item]).transpose((1, 0))  # 取得数据并reshape成[S，F]，S为时间轴长度，F为特征数
        data = torch.from_numpy(data).float()
        data = F.normalize(data, dim=0)  # 对每个特征列归一化
        out = data.new_full((LENGTH, 28), float('-inf'))  # pad到LENGTH长度
        out[:data.size()[0], :] = data[-LENGTH:, :]
        return out

    def __len__(self):
        return len(self.data)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len=1200, d_model=28):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class LearnedPositionEncoding(nn.Module):
    def __init__(self, train=True, max_len=1200, d_model=28):
        super(LearnedPositionEncoding, self).__init__()
        if train:
            self.weight = nn.parameter.Parameter(torch.empty(max_len, d_model), requires_grad=True).unsqueeze(1).to(device)
        else:
            self.weight = nn.parameter.Parameter(torch.empty(max_len, d_model), requires_grad=True).unsqueeze(1)
        nn.init.xavier_uniform_(self.weight)  # 初始化参数

    def forward(self, x):
        x = x + self.weight[:x.size(0), :]
        return x


def mask(inputs, mask_probability, src_key_padding_mask):
    masked_indices = torch.bernoulli(torch.full(inputs.size()[:-1], mask_probability)).to(device).bool()
    masked_indices = (masked_indices & ~src_key_padding_mask.transpose(1, 0))
    masked_indices_ = masked_indices.unsqueeze(2).expand(inputs.size())

    random_value = torch.randn_like(masked_indices_.float())
    inputs[masked_indices_] = random_value[masked_indices_]

    return inputs, masked_indices.float().transpose(1, 0)


class ClusteringLayer(nn.Module):
    def __init__(self, num_cluters, d_model):
        super(ClusteringLayer, self).__init__()
        self.num_clusters = num_cluters
        self.centriod = nn.parameter.Parameter(torch.Tensor(num_cluters, d_model))
        self.d_model = d_model

        nn.init.xavier_uniform_(self.centriod)

    def forward(self, x):
        qij = x.unsqueeze(1).expand(x.size()[0], self.num_clusters, self.d_model) - self.centriod

        qij_son = (1 + qij.norm(dim=2).pow(2)).pow(-1)
        qij_mother = qij_son.sum(dim=1, keepdim=True)
        Q = qij_son / qij_mother

        pij_son = Q.pow(2) / Q.sum(dim=0, keepdim=True)
        pij_mother = pij_son.sum(dim=1, keepdim=True)
        P = pij_son / pij_mother
        return Q, P


class EEGBert(nn.Module):
    def __init__(self, train=True, d_model=28, nhead=4, num_layers=2):
        super(EEGBert, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.position_encode = LearnedPositionEncoding(train)
        self.linear = nn.Linear(d_model, LENGTH)

        self.train_or_not = train

    def forward(self, x):
        if self.train_or_not:
            src_key_padding_mask = torch.ones_like(x).masked_fill(x != float('-inf'), 0.0)\
                .masked_fill(x == float('-inf'), 1.)
            src_key_padding_mask = torch.mean(src_key_padding_mask, dim=2).transpose(1, 0)
            src_key_padding_mask = src_key_padding_mask.bool()

            x = x.masked_fill(x == float('-inf'), 0.)

            # 添加mask，构建自监督学习, 20%的时间点位置被替换成了随机数.之后再对S维度归一化
            x, masked_indices = mask(x, mask_probability, src_key_padding_mask)

            x = F.pad(x, [0, 0, 0, 0, 1, 0])  # 在每个batch的最前面加入全0开头
            x = self.position_encode(x)  # 加入位置编码
            src_key_padding_mask = F.pad(src_key_padding_mask, [1, 0, 0, 0])

            output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

            pre_masked_indices = self.linear(output[0, :, :])
            return output[1:, :, :], masked_indices, pre_masked_indices, src_key_padding_mask[:, 1:]
        else:
            src_key_padding_mask = torch.ones_like(x).masked_fill(x != float('-inf'), 0.0) \
                .masked_fill(x == float('-inf'), 1.)
            src_key_padding_mask = torch.mean(src_key_padding_mask, dim=2).transpose(1, 0)
            src_key_padding_mask = src_key_padding_mask.bool()

            x = x.masked_fill(x == float('-inf'), 0.)

            x = F.pad(x, [0, 0, 0, 0, 1, 0])  # 在每个batch的最前面加入全0开头
            x = self.position_encode(x)  # 加入位置编码
            src_key_padding_mask = F.pad(src_key_padding_mask, [1, 0, 0, 0])

            output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            return output[1:, :, :]


class DeepCluster(nn.Module):
    def __init__(self, train=True):
        super(DeepCluster, self).__init__()
        self.eegbert = EEGBert(train=train)
        self.clusterlayer = ClusteringLayer(num_cluters=num_cluster, d_model=28)

        self.train_or_not = train

    def forward(self, x):
        if self.train_or_not:
            output, masked_indices, pre_masked_indices, src_key_padding_mask = self.eegbert(x)
            cluster_samples = output[~src_key_padding_mask.transpose(1, 0)]
            Q, P = self.clusterlayer(cluster_samples)
            Q= torch.log(Q)
            return output, masked_indices, pre_masked_indices, Q, P
        else:
            output = self.eegbert(x)
            return output

def xunlian():
    datasets = myBertDatasets()
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True)

    deepcluster = DeepCluster().to(device)
    loss_func1 = nn.MSELoss()
    loss_func2 = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(deepcluster.parameters(), lr=LR)
    # schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.85)

    for epoch in np.arange(EPOCH):
        for step, data in enumerate(dataloader):
            data = data.transpose(1, 0).to(device)
            _, masked_indices, pre_masked_indices, Q, P = deepcluster(data)

            loss1 = loss_func1(pre_masked_indices, masked_indices)
            loss2 = loss_func2(Q, P)
            loss = 0.4 * loss1 + 0.6 * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 and step % 20 == 0:
                print(f'epoch:{epoch} | step:{step} | loss1={loss1} loss2={loss2} --- loss={loss}')

        # schedule.step()

    torch.save(deepcluster.state_dict(), fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\模型\{name}-deepcluster.pt')

if __name__ == '__main__':
    xunlian()