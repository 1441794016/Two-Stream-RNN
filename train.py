import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from model.LSTM import TemporalRNN, SpatialRNN, TwoStreamRNN
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
from test import test
from matplotlib import pyplot as plt


train_dataset_path = "h5/NTURGB-D_train.h5"
test_dataset_path = "h5/NTURGB-D_test.h5"

model = TwoStreamRNN()

nll_loss = nn.NLLLoss()
# model中forward已经对结果进行了sotfmax处理，log_softmax()和NLLoss()的结合效果就是nn.CrossEntropyLoss()效果

lr = 0.002
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
# 每隔50个epoch lr变为原来的0.7

train_data = TrainDataset(train_dataset_path)
train_loader = DataLoader(train_data, batch_size=batch_size,
                          num_workers=0, shuffle=False, drop_last=True)
test_data = TestDataset(test_dataset_path)
test_loader = DataLoader(test_data, batch_size=batch_size,
                         num_workers=0, shuffle=False, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

epoch = 60

cs = [24, 25, 12, 11, 10, 9, 21, 5, 6, 7, 8, 23, 22, 4, 3, 21, 2, 1, 20, 19, 18, 17, 1, 13, 14, 15, 16]
# chain sequence

loss_list = []
train_acc_list = []
test_acc_list = []
for i in range(epoch):
    all_loss = 0
    train_acc = 0
    time_start = time.time()
    for inputs in train_loader:
        model.train()
        skeleton0 = inputs[-1][0].to(device).float()
        skeleton1 = inputs[-1][1].to(device).float()
        label = inputs[-1][2].to(device).long()

        sp_skeleton0 = torch.rand([batch_size, len(cs), 300])
        sp_skeleton1 = torch.rand([batch_size, len(cs), 300])
        for b in range(batch_size):
            sequence_idx = 0
            for j in cs:
                sp_skeleton0[b][sequence_idx] = skeleton0[b][:, 3 * (j - 1):3 * j].reshape(1, 300)
                sp_skeleton1[b][sequence_idx] = skeleton1[b][:, 3 * (j - 1):3 * j].reshape(1, 300)
                sequence_idx += 1

        sp_skeleton0 = sp_skeleton0.to(device).float()
        sp_skeleton1 = sp_skeleton1.to(device).float()

        pre = model(skeleton0, skeleton1, sp_skeleton0, sp_skeleton1)
        loss = nll_loss(pre, label)

        train_acc += np.sum(np.argmax(pre.cpu().detach().numpy(), axis=1) == label.cpu().detach().numpy())
        all_loss = all_loss + loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    test_acc = test(model, batch_size, device, test_data, test_loader)
    time_end = time.time()

    loss_list.append(all_loss.item())
    train_acc_list.append(train_acc / train_data.__len__())
    test_acc_list.append(test_acc)
    print(str(i + 1), '/', str(epoch), ':', str(all_loss))
    print("train_acc: ", str(train_acc / train_data.__len__()))
    print("test_acc: ", str(test_acc))
    print("time costing: ", str(time_end - time_start))

x_ = []
for i in range(epoch):
    x_.append(i + 1)
plt.plot(x_, loss_list)
plt.title('train loss')
plt.legend(['train'])
plt.savefig('train_loss.png')

# plt.plot(x_, train_acc_list)
# plt.plot(x_, test_acc_list)
# plt.title('accuracy')
# plt.legend(['train', 'test'])
# plt.savefig('accuracy.png')