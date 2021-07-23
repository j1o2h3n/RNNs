import torch
import torch.nn as nn
from torch.autograd import Variable


class Line_Transform(nn.Module):
    def __init__(self, c_in, c_out):
        super(Line_Transform, self).__init__()
        self.w = nn.Parameter(torch.rand(c_out, c_in), requires_grad=True).cuda()
        self.b = nn.Parameter(torch.rand(c_in, 1), requires_grad=True).cuda()

    def forward(self, x):
        x = torch.einsum('vf,bfn->bvn', (self.w, x)) + self.b
        return x.contiguous()


class RNN(nn.Module):
    def __init__(self, c_in, c_out):
        super(RNN, self).__init__()
        self.w1 = Line_Transform(c_in,c_out)
        self.w2 = Line_Transform(c_out,c_out)
        self.c_out = c_out

    def forward(self, x):
        shape = x.shape # b,f,n,t
        h = Variable(torch.zeros((shape[0], self.c_out, shape[2]))).cuda()
        out = []
        for t in range(shape[3]):
            input = x[:, :, :, t] # b,f,n
            new_h = torch.tanh(self.w1(input)+self.w2(h))
            h = new_h  # b,f,n
            out.append(new_h)
        x = torch.stack(out, -1) # b,f,n,t
        return x


class LSTM(nn.Module):
    def __init__(self, c_in, c_out):
        super(LSTM, self).__init__()
        self.w1 = Line_Transform(c_in,c_out)
        self.w2 = Line_Transform(c_out,c_out)
        self.w3 = Line_Transform(c_in,c_out)
        self.w4 = Line_Transform(c_out,c_out)
        self.w5 = Line_Transform(c_in, c_out)
        self.w6 = Line_Transform(c_out, c_out)
        self.w7 = Line_Transform(c_in, c_out)
        self.w8 = Line_Transform(c_out, c_out)
        self.c_out = c_out

    def forward(self, x):
        shape = x.shape # b,f,n,t
        h = Variable(torch.zeros((shape[0], self.c_out, shape[2]))).cuda()
        c = Variable(torch.zeros((shape[0], self.c_out, shape[2]))).cuda()
        out = []
        for t in range(shape[3]):
            input = x[:, :, :, t] # b,f,n
            i = torch.sigmoid(self.w1(input)+self.w2(h))
            f = torch.sigmoid(self.w3(input)+self.w4(h))
            o = torch.sigmoid(self.w5(input)+self.w6(h))
            g = torch.tanh(self.w7(input)+self.w8(h))
            new_c = f*c +i*g
            new_h = o*torch.tanh(new_c)
            c= new_c
            h = new_h  # b,f,n
            out.append(new_h)
        x = torch.stack(out, -1) # b,f,n,t
        return x


class GRU(nn.Module):
    def __init__(self, c_in, c_out):
        super(GRU, self).__init__()
        self.w1 = Line_Transform(c_in,c_out)
        self.w2 = Line_Transform(c_out,c_out)
        self.w3 = Line_Transform(c_in,c_out)
        self.w4 = Line_Transform(c_out,c_out)
        self.w5 = Line_Transform(c_in, c_out)
        self.w6 = Line_Transform(c_out, c_out)
        self.c_out = c_out

    def forward(self, x):
        shape = x.shape # b,f,n,t
        h = Variable(torch.zeros((shape[0], self.c_out, shape[2]))).cuda()
        out = []
        for t in range(shape[3]):
            input = x[:, :, :, t] # b,f,n
            z = torch.sigmoid(self.w1(input)+self.w2(h))
            r = torch.sigmoid(self.w3(input)+self.w4(h))
            h_ = torch.tanh(self.w5(input)+self.w6(r * h))
            new_h = z * h + (1 - z) * h_
            h = new_h  # b,f,n
            out.append(new_h)
        x = torch.stack(out, -1) # b,f,n,t
        return x
