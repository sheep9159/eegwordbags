import math
import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn


EPOCH = 200           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 32
LR = 0.01           # learning rate

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

    def __getitem__(self, item):
        data = np.load(self.data[item]).transpose((1, 0))
        data = torch.from_numpy(data).float()
        data = data[-1000:, :]
        data = nn.functional.normalize(data, dim=0)
        return data

    def __len__(self):
        return len(self.data)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len=2000, d_model=28):
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
    def __init__(self, max_len=2000, d_model=28):
        super(LearnedPositionEncoding, self).__init__()
        self.weight = nn.parameter.Parameter(torch.empty(max_len, d_model), requires_grad=True).unsqueeze(1).to(device)
        nn.init.xavier_uniform_(self.weight)  # 初始化参数

    def forward(self, x):
        x = x + self.weight[:x.size(0), :]
        return x
    

class EEGBert(nn.Module):
    def __init__(self, d_model=28, nhead=4, num_layers=2):
        super(EEGBert, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=32)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.position_encode = LearnedPositionEncoding()

    def forward(self, x):
        src_key_padding_mask = torch.ones_like(x).masked_fill(x != float('-inf'), 0.0)\
            .masked_fill(x == float('-inf'), 1.)
        src_key_padding_mask = torch.mean(src_key_padding_mask, dim=2).transpose(1, 0)
        src_key_padding_mask = src_key_padding_mask.bool()

        x = x.masked_fill(x == float('-inf'), 0.)
        x = self.position_encode(x)  # 加入位置编码

        # 添加mask，构建自监督学习, 15%的位置被替换成了0.
        masked_indices = torch.bernoulli(torch.full(x.size(), 0.15)).bool()
        x[masked_indices] = 0.

        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return output


datasets = myBertDatasets()
dataloader = torch.utils.data.DataLoader(datasets, batch_size=BATCH_SIZE, shuffle=True,
                                         collate_fn=lambda x: nn.utils.rnn.pad_sequence(x, padding_value=float('-inf')))

eegbert = EEGBert().to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(eegbert.parameters(), lr=LR)

for epoch in np.arange(EPOCH):
    for step, data in enumerate(dataloader):
        data = data.to(device)
        output = eegbert(data)

        data = data.masked_fill(data == float('-inf'), 0.)

        loss = loss_func(output, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 and step % 20 == 0:
            print(f'epoch:{epoch} | step:{step} | loss = {loss}')

torch.save(eegbert.state_dict(), fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\模型\eegbert.pt')