import torch
from torch import nn
import pdb

# Mostly a port of my Master thesis network for Speech enchancement
# https://github.com/dvisockas/norse/blob/master/norse/models/model.py
#
# SimpleNet is a 1D CNN with 2 layers, an avg pool layer and a fully connected layer. Total ~800 params
# SimpleNetV2 is 1D CNN with 4 layers, an avg pool layer and 2 fully connected layers. Total ~130k params
# Both networks use GELU as an activation function and use batch norm between a conv and activation layer.
#
class SimpleNet(nn.Module):
    def __init__(self, bs=0):
        self.bs = bs
        super(SimpleNet, self).__init__()

        padding_mode = 'reflect'

        out_channel_layer_1 = 16
        self.conv_1 = nn.Conv1d(1, out_channel_layer_1, 32, 2, 15, padding_mode=padding_mode)
        self.norm_1 = nn.BatchNorm1d(out_channel_layer_1)
        self.act_1 = nn.GELU()

        out_channel_layer_2 = 1
        self.conv_2 = nn.Conv1d(out_channel_layer_1, out_channel_layer_2, 1, padding_mode=padding_mode)
        self.norm_2 = nn.BatchNorm1d(out_channel_layer_2)
        self.act_2 = nn.GELU()


        output_dim = 256

        self.pool = nn.AdaptiveAvgPool1d(output_dim)
        self.fc_1 = nn.Linear(256, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        c_1 = self.act_1(self.norm_1(self.conv_1(x)))

        c_2 = self.act_2(self.norm_2(self.conv_2(c_1)))

        last = self.fc_1(self.pool(c_2))

        out = torch.sigmoid(last)

        return out

class SimpleNetV2(nn.Module):
    def __init__(self):
        super(SimpleNetV2, self).__init__()

        padding_mode = 'reflect'

        out_channel_layer_1 = 32
        self.conv_1 = nn.Conv1d(1, out_channel_layer_1, 32, 2, 15, padding_mode=padding_mode)
        self.norm_1 = nn.BatchNorm1d(out_channel_layer_1)
        self.act_1 = nn.GELU()

        out_channel_layer_2 = 32
        self.conv_2 = nn.Conv1d(out_channel_layer_1, out_channel_layer_2, 32, 2, 15, padding_mode=padding_mode)
        self.norm_2 = nn.BatchNorm1d(out_channel_layer_2)
        self.act_2 = nn.GELU()

        out_channel_layer_3 = 64
        self.conv_3 = nn.Conv1d(out_channel_layer_2, out_channel_layer_3, 32, 2, 15, padding_mode=padding_mode)
        self.norm_3 = nn.BatchNorm1d(out_channel_layer_3)
        self.act_3 = nn.GELU()

        out_channel_layer_4 = 1
        self.conv_4 = nn.Conv1d(out_channel_layer_3, out_channel_layer_4, 1, padding_mode=padding_mode)
        self.norm_4 = nn.BatchNorm1d(out_channel_layer_4)
        self.act_4 = nn.GELU()

        output_dim = 256
        fc_2_size = 128

        self.pool = nn.AdaptiveAvgPool1d(output_dim)
        self.fc_1 = nn.Linear(output_dim, fc_2_size)
        self.fc_2 = nn.Linear(fc_2_size, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x):
        c_1 = self.act_1(self.norm_1(self.conv_1(x)))

        c_2 = self.act_2(self.norm_2(self.conv_2(c_1)))

        c_3 = self.act_3(self.norm_3(self.conv_3(c_2)))

        c_4 = self.act_4(self.norm_4(self.conv_4(c_3)))

        fc_1_out = self.fc_1(self.pool(c_4))
        fc_2_out = self.fc_2(fc_1_out)

        last = fc_2_out

        out = torch.sigmoid(last)

        return out
