import torch
import torch.nn as nn
import torch.optim as op
import model
import stockloader
import utils


a = stockloader.StockData()
testset = stockloader.StockData(True)
pred = nn.DataParallel(model.TraderAI())
# predII = nn.DataParallel(model.TraderAI(273))
optim = op.Adadelta(pred.parameters())  #  + list(predII.parameters())
sched = op.lr_scheduler.ExponentialLR(optim, 0.64)
# pred.load_state_dict(torch.load("predictorMKII-A.pt"))
# predII.load_state_dict(torch.load("networkMVI-B.pt"))
if True:
    for i in range(100):
        s = 0
        for dp in a:
            #  pred(dp)
            short = utils.tema(dp.numpy(), 16)
            longt = utils.tema(dp.numpy(), 75)

            shorte = utils.ema(dp.numpy(), 50)
            longe = utils.ema(dp.numpy(), 200)

            conv_line, base_line, lead_span_a, lead_span_b = utils.ichimoku_cloud(dp.numpy())

            # import matplotlib.pyplot as plt

            #plt.plot(dp.numpy())
            #plt.plot(short)
            #plt.plot(longt)
            # plt.ylabel('some numbers')

            data_stream = [dp.clone(), short, longt, shorte, longe, conv_line, base_line, lead_span_a, lead_span_b]
            for i in range(0, len(data_stream)):
                data_stream[i] = utils.locked_diff(data_stream[0], data_stream[i])
            #  data_stream[0] = utils.change(data_stream[0])

            data_stream[0] = utils.relative(dp.numpy())

            clen = len(data_stream)

            for p in range(1, 30):
                for v in range(clen):
                    data_stream.append(utils.lookback(data_stream[v], p))

            for i in range(0, len(data_stream)):
                data_stream[i] = torch.tensor(data_stream[i], dtype=torch.float32)

            '''
            short = torch.tensor(short, dtype=torch.float32)
            longt = torch.tensor(longt, dtype=torch.float32)
            shorte = torch.tensor(shorte, dtype=torch.float32)
            longe = torch.tensor(longe, dtype=torch.float32)
            conv_line = torch.tensor(conv_line, dtype=torch.float32)
            base_line = torch.tensor(base_line, dtype=torch.float32)
            lead_span_a = torch.tensor(lead_span_a, dtype=torch.float32)
            lead_span_b = torch.tensor(lead_span_b, dtype=torch.float32)
            '''

            optim.zero_grad()
            actions = pred(torch.stack(data_stream, dim=1))

            '''
            again = torch.cat([torch.stack(data_stream, dim=1).cuda(), torch.squeeze(actions, 1)], dim=1)
            actions = predII(again)
            '''

            actions = torch.squeeze(actions, 1)
            money = torch.full([1], 10000.0, dtype=torch.float32, requires_grad=True).cuda()
            shares = torch.zeros([1], dtype=torch.float32, requires_grad=True).cuda()
            # add prev actions, 50 and 200 EMA, RSI
            last_price = 0
            for i in range(actions.size()[0]):
                r = torch.rand([1]).cuda()
                buy_action = torch.relu(actions[i][0] - r)
                sell_action = torch.relu(actions[i][1] - r + actions[i][0])
                hold_action = torch.relu(actions[i][2] - r + actions[i][0] + actions[i][1])
                if buy_action.item() != 0:
                    money = torch.relu(money - 10)
                    shares = shares + buy_action * money / dp[i]
                    money = money - buy_action * money
                elif sell_action.item() != 0:
                    money = money + sell_action * shares * dp[i] - 10
                    shares = shares - sell_action * shares
                else:
                    pass
                last_price = dp[i]
            ratio = (money + shares * last_price) / 10000
            (1 - ratio).backward()
            optim.step()
            s += 1
            print("{}/{}".format(s, len(a)))
            print(ratio.item())
        sched.step()
        torch.save(pred.state_dict(), "predictorMKIII-A.pt")
        #  torch.save(predII.state_dict(), "networkMVII-B.pt")
        #plt.show()

exit()
print("=" * 35)
pred = torch.load("networkMKV.pt")
pred.eval()


if True:
    s = 0
    for dp in a:
        #  pred(dp)
        short = utils.tema(dp.numpy(), 16)
        longt = utils.tema(dp.numpy(), 75)

        shorte = utils.ema(dp.numpy(), 50)
        longe = utils.ema(dp.numpy(), 200)

        conv_line, base_line, lead_span_a, lead_span_b = utils.ichimoku_cloud(dp.numpy())

        # import matplotlib.pyplot as plt

        #plt.plot(dp.numpy())
        #plt.plot(short)
        #plt.plot(longt)
        # plt.ylabel('some numbers')

        data_stream = [dp.clone(), short, longt, shorte, longe, conv_line, base_line, lead_span_a, lead_span_b]
        for i in range(1, len(data_stream)):
            data_stream[i] = utils.locked_diff(data_stream[0], data_stream[i])
        #  data_stream[0] = utils.change(data_stream[0])

        clen = len(data_stream)

        for p in range(1, 30):
            for v in range(clen):
                data_stream.append(utils.lookback(data_stream[v], p))


        for i in range(1, len(data_stream)):
            data_stream[i] = torch.tensor(data_stream[i], dtype=torch.float32)


        '''
        short = torch.tensor(short, dtype=torch.float32)
        longt = torch.tensor(longt, dtype=torch.float32)
        shorte = torch.tensor(shorte, dtype=torch.float32)
        longe = torch.tensor(longe, dtype=torch.float32)
        conv_line = torch.tensor(conv_line, dtype=torch.float32)
        base_line = torch.tensor(base_line, dtype=torch.float32)
        lead_span_a = torch.tensor(lead_span_a, dtype=torch.float32)
        lead_span_b = torch.tensor(lead_span_b, dtype=torch.float32)
        '''

        optim.zero_grad()
        actions = pred(torch.stack(data_stream, dim=1))

        again = torch.cat([torch.stack(data_stream, dim=1).cuda(), torch.squeeze(actions, 1)], dim=1)
        #  actions = predII(again)

        actions = torch.squeeze(actions, 1)
        money = torch.full([1], 10000.0, dtype=torch.float32, requires_grad=True).cuda()
        shares = torch.zeros([1], dtype=torch.float32, requires_grad=True).cuda()
        # add prev actions, 50 and 200 EMA, RSI
        last_price = 0
        for i in range(actions.size()[0]):
            r = torch.rand([1]).cuda()
            buy_action = torch.relu(actions[i][0] - r)
            sell_action = torch.relu(actions[i][1] - r + actions[i][0])
            hold_action = torch.relu(actions[i][2] - r + actions[i][0] + actions[i][1])
            if buy_action.item() != 0:
                shares = shares + buy_action * (money - 10) / dp[i]
                money = money - buy_action * (money - 10)
            elif sell_action.item() != 0:
                money = money + sell_action * shares * dp[i] - 10
                shares = shares - sell_action * shares
            else:
                pass
            last_price = dp[i]
        ratio = (money + shares * dp[i]) / 10000
        (1 - ratio).backward()
        #optim.step()
        s += 1
        print("{}/{}".format(s, len(a)))
        print(ratio.item())
    sched.step()
    #torch.save(pred.state_dict(), "networkMKVI.pt")
    #torch.save(predII.state_dict(), "networkMVI-B.pt")
    #plt.show()


for dp in testset:
    #  pred(dp)
    short = utils.tema(dp.numpy(), 16)
    longt = utils.tema(dp.numpy(), 75)

    shorte = utils.ema(dp.numpy(), 50)
    longe = utils.ema(dp.numpy(), 200)

    conv_line, base_line, lead_span_a, lead_span_b = utils.ichimoku_cloud(dp.numpy())

    # import matplotlib.pyplot as plt

    # plt.plot(dp.numpy())
    # plt.plot(short)
    # plt.plot(longt)
    # plt.ylabel('some numbers')

    data_stream = [dp.clone(), short, longt, shorte, longe, conv_line, base_line, lead_span_a, lead_span_b]
    for i in range(1, len(data_stream)):
        data_stream[i] = utils.locked_diff(data_stream[0], data_stream[i])
    #  data_stream[0] = utils.change(data_stream[0])

    clen = len(data_stream)

    for p in range(1, 30):
        for v in range(clen):
            data_stream.append(utils.lookback(data_stream[v], p))

    for i in range(1, len(data_stream)):
        data_stream[i] = torch.tensor(data_stream[i], dtype=torch.float32)

    '''
    short = torch.tensor(short, dtype=torch.float32)
    longt = torch.tensor(longt, dtype=torch.float32)
    shorte = torch.tensor(shorte, dtype=torch.float32)
    longe = torch.tensor(longe, dtype=torch.float32)
    conv_line = torch.tensor(conv_line, dtype=torch.float32)
    base_line = torch.tensor(base_line, dtype=torch.float32)
    lead_span_a = torch.tensor(lead_span_a, dtype=torch.float32)
    lead_span_b = torch.tensor(lead_span_b, dtype=torch.float32)
    '''



    actions = pred(torch.stack(data_stream, dim=1))

    actions = torch.squeeze(actions, 1)
    money = torch.full([1], 10000.0, dtype=torch.float32, requires_grad=True)
    shares = torch.zeros([1], dtype=torch.float32, requires_grad=True)
    fees = torch.zeros([1], dtype=torch.float32, requires_grad=True)
    buys = 0
    sells = 0
    for i in range(actions.size()[0]):
        r = torch.rand([1])
        buy_action = torch.relu(actions[i][0] - r)
        sell_action = torch.relu(actions[i][1] - r + actions[i][0])
        hold_action = torch.relu(actions[i][2] - r + actions[i][0] + actions[i][1])
        if buy_action.item() != 0:
            shares = shares + buy_action * money / dp[i]
            money = money - buy_action * money
            fees = fees + 10
            buys += 1
        elif sell_action.item() != 0:
            money = money + sell_action * shares * dp[i]
            shares = shares - sell_action * shares
            sells += 1
        else:
            pass
    ratio = ((money - fees) / 10000)
    print("Buys: " + str(buys))
    print("Sells: " + str(sells))
    print(ratio.item())


'''
embed = nn.Embedding(10, 3)
inp = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
print(embed(inp).size())
# torch.Size([2, 4, 3])

a = stockloader.StockData()
tr = model.Predictor(128)
opt = op.Adam(tr.parameters(), lr=0.01, betas=[0.999, 0.5])
for dp in a:
    s = model.shift_div(dp).size()[0]
    sd = model.shift_div(dp)
    lower = 0
    higher = 32
    if s >= 32:
        while higher <= s - 1:
            opt.zero_grad()
            out = tr(sd[lower:higher])
            tgt = sd[higher:(higher + 1)]
            loss = torch.sum(torch.abs(out - tgt))
            loss.backward()
            print("Out:")
            print(out)
            print("Target:")
            print(tgt)
            print("Loss:")
            print(loss)
            print("=" * 100)
            opt.step()
            lower += 1
            higher += 1
'''
