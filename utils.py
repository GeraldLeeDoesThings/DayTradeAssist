

'''
def ema(data, period, smoothing=2):
    length = len(data)
    if length < period:
        raise IndexError("kjfsdkjhn")
    result = [0] * (length - period)
    sumn = 0
    for i in range(period):
        sumn += data[i]
    result[0] = sumn / period
    alpha = smoothing / (period + 1)
    for i in range(length - period - 1):
        result[i + 1] = data[i + period] * alpha + result[i] * (1 - alpha)
    return result

a = [2, 4, 6, 8, 12, 14, 16, 2, 1]
b = ema(a, 4, 2)
print(b)
for i in range(len(a)):
    a[i] = a[i] - b[i]
print(a)
'''


def ema(data, period, smoothing=2):
    length = len(data)
    result = [0] * length
    result[0] = data[0]
    alpha = smoothing / (period + 1)
    for i in range(length - 1):
        result[i + 1] = data[i + 1] * alpha + result[i] * (1 - alpha)
    return result


def tema(data, period, smoothing=2):
    ema1 = ema(data, period, smoothing)
    ema2 = ema(ema1, period, smoothing)
    ema3 = ema(ema2, period, smoothing)
    result = [0] * len(ema1)
    for i in range(len(ema1)):
        result[i] = 3 * ema1[i] - 3 * ema2[i] + ema3[i]
    return result


def ph_pl_div2(data, period):
    out = []
    for i in range(len(data)):
        lower = max(0, i - period)
        upper = max(1, i)
        r = data[lower:upper]
        ph = max(r)
        pl = min(r)
        out.append((ph + pl) / 2)
    return out


def elemwise_average(data1, data2):
    out = []
    for i in range(len(data1)):
        out.append((data1[i] + data2[i]) / 2)
    return out


def ichimoku_cloud(data):
    conv_line = ph_pl_div2(data, 9)
    base_line = ph_pl_div2(data, 26)
    lead_span_a = elemwise_average(conv_line, base_line)
    lead_span_b = ph_pl_div2(data, 52)
    return conv_line, base_line, lead_span_a, lead_span_b


def locked_diff(base, der):
    for i in range(len(base)):
        der[i] = der[i] - base[i]
    return der


def change(data):
    for i in range(len(data)):
        if i == 0:
            pass
        else:
            data[i] = data[i] - data[i - 1]
    data[0] = 0
    return data

#  Use lookbacks


def lookback(data, distance):
    out = []
    for i in range(len(data)):
        if i - distance < 0:
            out.append(0)
        else:
            out.append(data[i])
    return out


def avg(data):
    sum = 0
    for i in data:
        sum += i
    return sum / len(data)


def relative(data):
    start = 0
    end = 6
    a = avg(data[start:end])
    o = []
    for point in range(len(data)):
        if point == end:
            start += 1
            end += 1
            a = avg(data[start:end])
        o.append(data[point] / a)
    return o
