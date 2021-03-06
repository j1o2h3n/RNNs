import torch.optim as optim
import torch
from model import *
import util

class trainer():
    def __init__(self, args, scaler, Flag, in_dim, nhid,out_dim, time_length):
        wdecay = args.weight_decay
        lrderate = args.lr_decay_rate
        lr_step_size = args.lr_step_size
        lrate = args.learning_rate
        self.model = RNNs(Flag, in_dim, nhid ,out_dim, time_length)
        self.model.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step_size, gamma=lrderate)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        real = torch.unsqueeze(real_val, dim=1)
        output = self.model(input)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        real = torch.unsqueeze(real_val, dim=1)
        output = self.model(input)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
