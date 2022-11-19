import numpy as np
import torch.nn as nn
import torch

def wgn(x, snr):
    b, h, w = x.shape
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / (b*h*w)
    npower = xpower / snr
    return np.random.randn(b, h, w) * np.sqrt(npower)


def add_noise(data, snr_num):
    data_p = data

    rand_data = wgn(data, snr_num)

    data_n = data_p + rand_data

    return data_n

'''x = torch.ones(1, 1, 5)
conv = nn.Conv1d(1, 1, kernel_size=2, stride=2, padding=0)
y = conv(x)
print(y.shape)
deconv = nn.ConvTranspose1d(1, 1, 2, stride=2, padding=0, output_padding=1)
xx = deconv(y)
print(xx)'''