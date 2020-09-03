import mne
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

# a = torch.arange(0, 20, 1).resize_(4, 5).float()
# b = torch.arange(20, 35, 1).resize_(3, 5).float()
# c = torch.arange(35, 40, 1).resize_(1, 5).float()
# print(a)
# print(b)
# print(c)
# l = [a, b, c]
#
# class a(torch.utils.data.Dataset):
#     def __getitem__(self, item):
#         return l[item]
#
#     def __len__(self):
#         return len(l)
#
# datasets = a()
# dataloader = torch.utils.data.DataLoader(datasets, batch_size=3, shuffle=True, collate_fn=lambda x: nn.utils.rnn.pad_packed_sequence(x, padding_value=float('-inf')))
# for index , data in enumerate(dataloader):
#     print(index, '\n', data)

encoder_layer = nn.TransformerEncoderLayer(d_model=3, nhead=1)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

src = torch.tensor([[[1, 2, 3, 11],[2, 3, 4, 12]],
                    [[3, 4, 5, 13],[4, 5, 6, 14]],
                    [[5, 6, 7, 15],[100, 100, 100, 16]]]).float()
print('src.size() is ', src.size())
print(src)

masked_indices = torch.bernoulli(torch.full(src.size()[:-1], 0.5)).bool()
print(masked_indices)
masked_indices = masked_indices.unsqueeze(2)
print(masked_indices)
masked_indices = masked_indices.expand(src.size())
print(masked_indices)
src[masked_indices] = 0.
print(src)
print(src[~masked_indices])
