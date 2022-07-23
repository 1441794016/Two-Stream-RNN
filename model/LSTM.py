import torch
import torch.nn as nn
import torch.nn.functional as f


class TemporalRNN(nn.Module):
    def __init__(self, input_size=25 * 3, hidden_size=512):
        # suggestion: num_layers = 2 即两层RNN堆叠
        # 时间序列 sequence 长度: T = 100 , 神经元数量: 512 , 分类种类 : 60
        super(TemporalRNN, self).__init__()
        self.LSTM1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        # batch_first = True 则输入的tensor为: (batch_size, seq_len, input_size)
        # 输出的tensor为: (batch_size, seq_len, hidden_size)
        self.LSTM2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size*100, 60)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, skeleton0_inputs, skeleton1_inputs):
        x0, _ = self.LSTM1(skeleton0_inputs)
        # dimension of skeleton0_inputs: (batch_size, seq_len, input_size)
        # dimension of x0(output) : (batch_size, seq_len, hidden_size)
        x0, _ = self.LSTM2(x0)
        x0 = x0.reshape(x0.shape[0], x0.shape[1] * x0.shape[2])
        # dimension of x0(output) : (batch_size, seq_len * hidden_size)
        x0 = self.fc(x0)

        x1, _ = self.LSTM1(skeleton1_inputs)
        x1, _ = self.LSTM2(x1)
        x1 = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2])
        x1 = self.fc(x1)

        return (f.log_softmax(x0, 1) + f.log_softmax(x1, 1)) / 2
        # 两个动作执行者的预测分数进行平均


class SpatialRNN(nn.Module):
    # chain sequence : [24, 25, 12, 11, 10, 9, 21, 5, 6, 7, 8, 23, 22,
    #                   4, 3, 21, 2, 1,
    #                   20, 19, 18, 17, 1, 13, 14, 15, 16]
    def __init__(self, input_size=3 * 100, hidden_size=512):
        # dimension of input: (batch_size, sequence=25, input_size=3*100)
        super(SpatialRNN, self).__init__()
        self.LSTM1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.LSTM2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size * 27, 60)
        # 27是chain sequence的长度
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, skeleton0, skeleton1):
        x0, _ = self.LSTM1(skeleton0)
        # dimension of skeleton0 : (batch, seq_len, hidden_size)
        x0, _ = self.LSTM2(x0)
        x0 = x0.reshape(x0.shape[0], x0.shape[1] * x0.shape[2])
        x0 = self.fc(x0)

        x1, _ = self.LSTM1(skeleton1)
        # dimension of x : (batch, seq_len, hidden_size)
        x1, _ = self.LSTM2(x1)
        x1 = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2])
        x1 = self.fc(x1)

        return (f.log_softmax(x0, 1) + f.log_softmax(x1, 1)) / 2


class TwoStreamRNN(nn.Module):
    def __init__(self):
        super(TwoStreamRNN, self).__init__()
        self.temporalRNN = TemporalRNN()
        self.spatialRNN = SpatialRNN()

    def forward(self, te_skeleton0, te_skeleton1, sp_skeleton0, sp_skeleton1):
        te_score = self.temporalRNN(te_skeleton0, te_skeleton1)
        sp_score = self.spatialRNN(sp_skeleton0, sp_skeleton1)

        return 0.9 * te_score + (1 - 0.9) * sp_score

