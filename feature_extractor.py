import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from train_data import LENGTH


EPOCH = 100           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
train_BATCH_SIZE = 1000  # 1000
test_BATCH_SIZE = 200
TIME_STEP = LENGTH      # rnn 时间步数 / 图片高度
INPUT_SIZE = 28     # rnn 每步输入值 / 图片每行像素
LR = 0.01           # learning rate

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class myLstmDatasets(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        data1 = np.load(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\norm_sequence.npy')
        data2 = np.load(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\abnorm_sequence.npy')
        self.data = torch.from_numpy(np.vstack((data1, data2)).transpose((0, 2, 1))).float().to(device)
        print('datasets.shape: ', self.data.size())

    def __getitem__(self, item):
        if item < self.data.size()[0] / 2:
            target = torch.tensor(0).to(device)
        else:
            target = torch.tensor(1).to(device)

        return self.data[item, :, :], target

    def __len__(self):
        return self.data.size()[0]


class eeglstm(nn.Module):
    def __init__(self):
        super(eeglstm, self).__init__()
        self.lstm = nn.LSTM(
            input_size=28,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.linear = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.out2 = nn.Linear(32, 2)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        feature = self.linear(r_out[:,-1,:])
        out = self.out2(feature)

        return out, feature


if __name__ == '__main__':
    LSTM = eeglstm().to(device)
    datasets = myLstmDatasets()

    train_datasets, test_datasets = torch.utils.data.random_split(
        dataset=datasets, lengths=[int(len(datasets) * 0.7), len(datasets) - int(len(datasets) * 0.7)])

    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=train_BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=test_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(LSTM.parameters(), lr=LR)  # optimize all parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.85)

    for epoch in np.arange(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data

            output, _ = LSTM(b_x)  # rnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if epoch % 5 == 0:
                with torch.no_grad():
                    for _, (test_x, test_y) in enumerate(test_loader):
                        test_output, _ = LSTM(test_x)
                        accuracy = (torch.argmax(test_output, dim=1) == test_y).float().cpu().numpy().sum() / \
                                   test_BATCH_SIZE * 100
                        print(
                            'epoch:{} | step:{} | loss:{:0.3f} | acc:{:0.3f}%'.format(epoch, step, loss, accuracy))
                        break
        schedule.step(epoch)

    torch.save(LSTM.cpu().state_dict(), fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\LSTM.pt')