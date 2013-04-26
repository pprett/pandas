from __future__ import division

import numpy as np
import pylab as plt


def dewma(input, alpha, beta, adjust=False, init_trend=0):
    '''
    Compute double exponentially-weighted moving average.

    Adds a trend component to single exponentially-weighted moving average.

    Parameters
    ----------
    input : ndarray (float64 type)
    com : float64

    Returns
    -------
    y : ndarray
    '''

    N = len(input)

    output = np.empty(N, dtype=float)
    trend = np.empty(N, dtype=float)

    if N == 0:
        return output

    neww = alpha
    oldw = 1.0 - alpha
    newb = beta
    oldb = 1.0 - beta

    if adjust:
        output[0] = neww * input[0]
    else:
        output[0] = input[0]

    if init_trend == 0:
        trend[0] = input[1] - input[0]
    elif init_trend == 1:
        trend[0] = 0.0
        for i in range(3):
            trend[0] = input[i + 1] - input[i]
        trend[0] /= 3.0
    elif init_trend == 2:
        trend[0] = (input[N - 1] - input[0]) / (N - 1)

    for i in range(1, N):
        cur = input[i]
        prev_out = output[i - 1]
        prev_trend = trend[i - 1]

        if cur == cur:
            if prev_out == prev_out:
                output[i] = neww * cur + oldw * (prev_out + prev_trend)
                trend[i] = newb * (output[i] - prev_out) + oldb * prev_trend
            else:
                output[i] = neww * cur + oldw * prev_trend
        else:
            output[i] = prev_out

    return output


def tewma(input, alpha, beta, gamma, L):
    '''
    Compute triple exponentially-weighted moving average.

    Parameters
    ----------
    input : ndarray (float64 type)
    com : float64

    Returns
    -------
    y : ndarray
    '''

    N = len(input)

    output = np.empty(N, dtype=float)
    trend = np.empty(N, dtype=float)
    seaso = np.empty(N, dtype=float)

    if N == 0:
        return output

    neww = alpha
    oldw = 1.0 - alpha
    newb = beta
    oldb = 1.0 - beta
    newg = gamma
    oldg = 1.0 - gamma

    output[0] = input[0]

    # init trend based on two season cycles (2L points)
    trend[0] = 0.0
    for i in range(L):
        trend[0] += (input[L + i] - input[i]) / L
    trend[0] /= L

    # number of full cycles
    M = N // L

    # init the first L seaso coefs
    def norm(j):
        res = 0.0
        for i in range(L):
            res += input[L * j + i]
        return res / L

    for i in range(L):
        for j in range(M):
            s += (input[L * j + i] / norm(j))
        seaso[i] = s / M

    # compute output, trend, and seaso for reach time i
    for i in range(1, N):
        cur = input[i]
        prev_out = output[i - 1]
        prev_trend = trend[i - 1]
        prev_seaso = seaso[i - L]

        output[i] = (neww * (cur / prev_seaso) +
                     oldw * (prev_out + prev_trend))
        trend[i] = newb * (output[i] - prev_out) + oldb * prev_trend
        seaso[i] = newg * (cur / output[i]) + oldg * prev_seaso

    return output, trend, seaso



import statsmodels.api as sm
#data = sm.datasets.sunspots.load()

data = sm.datasets.copper.load_pandas()
wc = data.data[['WORLDCONSUMPTION']].values

x = np.arange(wc.shape[0])

plt.plot(x, wc, 'k-', label='wc')

for alpha, gamma in [(0.2, 0.5), (0.2, 0.1), (0.2, 0.7)]:
    sm = dewma(wc, alpha, gamma)
    plt.plot(x, sm, label='dewma %.1f %.1f' % (alpha, gamma))

plt.legend(loc='upper left')
plt.show()
