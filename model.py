import torch
import torch.nn as nn


def shift_div(t):
    x = torch.roll(t, 1, 0)
    t = t / x
    t = t[1:]
    return t


def batchify(x):
    return torch.unsqueeze(torch.unsqueeze(x, 0), 0)


def unbatchify(x):
    return torch.squeeze(torch.squeeze(x, 0), 0)


class TraderAI(nn.Module):

    def __init__(self, ins=270):
        super(TraderAI, self).__init__()
        #  self.embed = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=270, nhead=10),
        #                                   num_layers=2)
        self.brain = nn.LSTM(input_size=ins, hidden_size=512, num_layers=5)
        self.end = nn.LSTM(input_size=512, hidden_size=3, num_layers=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        #  x = self.embed(x)
        #  x = self.embed_more(x)
        x, _ = self.brain(x)
        x, _ = self.end(x)
        x = torch.softmax(x, 2)
        # print(x.size())  # (seq_size, 1 (batch), features (3 actions))
        return x


class Predictor(nn.Module):

    def __init__(self, siz):
        super(Predictor, self).__init__()
        self.transformer = nn.Sequential(
            nn.Linear(32, siz),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(siz, siz),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(siz, siz),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(siz, 1)
        )

    def forward(self, x):
        x = batchify(x)
        #  src = batchify(x[0:8])
        #  tgt = batchify(x[8:12])
        x = self.transformer((x - 1) * 10) + 1
        return unbatchify(x[0])  # src, tgt
