from sklearn.cluster import KMeans
import joblib
import numpy as np
import torch
import mne
from feature_extractor import eeglstm
from train_data import LENGTH, WINDOWS, STRIDE, read_mark_txt
from sklearn.preprocessing import StandardScaler


dictionary = ['A', 'B', 'C', 'D', 'E']
range_n_clusters = [2, 3, 4, 5]

# datasets = np.load(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\datasets.npy')
set_file_dir = fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\preprocess\weichuanxiang.set'
raw = mne.io.read_raw_eeglab(set_file_dir, preload=True)
raw_eeg = raw.get_data()

print('raw_eeg.shape: ', raw_eeg.shape)

mark, point = read_mark_txt(
    fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\original_record\weichuanxiang.vmrk')
print('mark.shape: ', mark.shape, 'point.shape: ', point.shape)

LSTM = eeglstm()
LSTM.load_state_dict(torch.load(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\LSTM.pt'))
LSTM.eval()

for n_clusters in range_n_clusters:
    kmeans = joblib.load(fr'D:\SJTU\Study\MME_Lab\Teacher_Lu\click_number\EEG词袋模型\聚类效果图\特征\{n_clusters}.model')

    for i in np.arange(len(mark) - 1):
        encode = []
        if mark[i + 1] != 0 and (point[i+1] - point[i]) >= 120:
            seq = raw_eeg[:, point[i]:point[i+1]]
            for j in np.arange(int((seq.shape[1] - WINDOWS) / STRIDE) + 2):
                start = j * STRIDE
                end = (j + 2) * STRIDE
                if end < seq.shape[1]:
                    seq_ = StandardScaler().fit_transform(seq[:, start:end].T).T
                    encode.append(seq_)
                else:
                    seq_ = StandardScaler().fit_transform(seq[:, -WINDOWS:].T).T
                    encode.append(seq_)

            encode = np.array(encode)
            _, feature = LSTM(torch.from_numpy(encode.transpose((0, 2, 1))).float())
            label = kmeans.predict(feature.detach().numpy())
            print(label, '----', i, (point[i+1]-point[i])/500)

    print('------------------------------------------------------------')