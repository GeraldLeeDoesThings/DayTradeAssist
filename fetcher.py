from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import torch
import pandas as pd
import time

key = "INSERT_KEY_HERE"
ts = TimeSeries(key)


def fetch(sym):
    dat, meta = ts.get_intraday(symbol=sym, interval='15min', outputsize='full')
    li = []
    name = ""
    for x in dat:
        if name == "":
            name = x
        # print(dat[x]['2. high'])
        li.append(float(dat[x]['2. high']))
    li.reverse()
    li = torch.tensor(
        li, dtype=torch.float32
    )
    torch.save(li, 'D:\\DayTradeAssist\\stockdata\\' + sym + '.pt')
    return li


#  fetch("GOLD")  # LOW



df = pd.read_csv("companylist.csv")
total = 0
for v in df["Symbol"]:
    try:
        fetch(v)
        total += 1
        if total == 480:
            break
        print(total)
    except ValueError:
        print(v)
    finally:
        time.sleep(20)

#  fetch("NCLH")
