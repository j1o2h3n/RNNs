# RNNs

Realize RNNs series models (RNN, LSTM, GRU) based on Pytorch, and perform sequence prediction tasks.

The code implements three models of RNN, LSTM, and GRU. The experimental task is to predict traffic flow sequence. The experimental data set is PEMS08, a total of 170 observation nodes, 17856 time frames, one frame is recorded every five minutes, and 12 frames of historical data are used to predict 12 frames of future data. The PEMS08 data set is divided into training set, validation set, and test set according to the ratio of 6:2:2.

## Molde

The overall model is mainly composed of input layer + two RNNs layers + output layer.

## Requirements

- python 3
- numpy
- torch

## Train Commands

```
python train.py --Flag
```

Select the model used for training by selecting Flag. The options provided are RNN/LSTM/GRU, and the default is GRU.
