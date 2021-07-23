import torch.nn as nn
from layers import RNN,LSTM,GRU



class RNNs(nn.Module):
    def __init__(self, Flag, in_dim=1, nhid=32 ,out_dim=12, time_length=12):
        super(RNNs, self).__init__()

        self.start_conv = nn.Conv2d(in_dim, nhid, kernel_size=(1,1), stride=(1, 1), bias=True)

        if Flag == "RNN":
            self.RNNs1 = RNN(nhid,nhid)
            self.RNNs2 = RNN(nhid,nhid)
        if Flag == "LSTM":
            self.RNNs1 = LSTM(nhid,nhid)
            self.RNNs2 = LSTM(nhid,nhid)
        if Flag == "GRU":
            self.RNNs1 = GRU(nhid,nhid)
            self.RNNs2 = GRU(nhid,nhid)

        self.end_conv = nn.Conv2d(nhid, out_dim, kernel_size=(1, time_length), stride=(1, 1), bias=True)

    def forward(self, input):

        # [64,2,207,12]
        x = self.start_conv(input) # [64,32,207,12]
        x = self.RNNs1(x)
        x = self.RNNs2(x) # [64,32,207,12]
        x = self.end_conv(x) # [64,12,207,1]
        x = x.transpose(1, 3) # [64,1,207,12]
        return x




