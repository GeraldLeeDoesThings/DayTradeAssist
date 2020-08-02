import torch
import os
import torch.utils.data as d


class StockData(d.Dataset):

    def __init__(self, eval=False):
        if eval:
            self.files = os.listdir("D:\\DayTradeAssist\\hidden")
            self.base = "D:\\DayTradeAssist\\hidden"
        else:
            self.files = os.listdir("D:\\DayTradeAssist\\stockdata")
            self.base = "D:\\DayTradeAssist\\stockdata"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        return torch.load(os.path.join(self.base, self.files[item]))

