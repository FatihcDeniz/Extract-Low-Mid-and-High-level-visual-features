import numpy as np
import PIL
from functools import lru_cache
from numba import njit


def calc_bin(val, binsize, x_min, x_max):
    x_max = x_max + 1
    x_min = x_min + 1

    bin_num = np.ceil((x_max - x_min) / binsize).astype(int)
    bin_log_size = (np.log10(x_max) - np.log10(x_min)) / bin_num
    bin_mean_bar = np.zeros((2, bin_num), dtype=np.float64)
    bin_x_val = np.zeros(bin_num, dtype=np.float64)

    for i in range(bin_num):
        ii = i + 1

        a_x = np.log10(x_min) + (ii - 1) * bin_log_size
        b_x = np.log10(x_min) + ii * bin_log_size

        from_val = np.floor(10 ** a_x).astype(int)
        to_val = np.floor(10 ** b_x).astype(int)

        if to_val > len(val):
            to_val = len(val)

        bin_mean_bar[0, i] = np.mean(np.arange(from_val, to_val + 1))
        bin_mean_bar[1, i] = np.mean(val[np.arange(from_val - 1, to_val)])
        bin_x_val[i] = from_val

    return bin_mean_bar, bin_x_val


def var_eigen(log_m, log_n, log_x, log_y):
    fit = log_m * log_x + log_n
    diff = fit - log_y
    return np.var(diff)


@lru_cache(maxsize=None)
def _precompute_rotavg_lookup(N):
    """
    Precompute rho and sector-bin lookup for a given square size N.
    """
    X, Y = np.meshgrid(np.arange(-N // 2, N // 2), np.arange(-N // 2, N // 2))
    theta = np.arctan2(Y, X)
    rho = np.round(np.sqrt(X**2 + Y**2)).astype(np.int64)

    # same angular bins as original
    a = np.arange(0, np.pi + np.pi / 8, np.pi / 8)

    # sector index for k=0..7, default -1 if angle not in [0, pi)
    sector = np.full(theta.shape, -1, dtype=np.int64)
    for k in range(8):
        mask = (a[k] <= theta) & (theta < a[k + 1])
        sector[mask] = k

    return rho, sector


@njit(cache=True)
def _rotavg_numba(array, rho, sector):
    N = array.shape[0]
    I = np.zeros((N // 2 + 1, 9), dtype=np.float64)
    f = np.zeros((N // 2 + 1, 9), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            rh = rho[i, j]
            value = array[i, j]

            if rh <= N / 2:
                # "all angles" bin at index 8
                I[rh, 8] += 1.0
                if I[rh, 8] == 1.0:
                    f[rh, 8] = value
                else:
                    f[rh, 8] = f[rh, 8] + (value - f[rh, 8]) / I[rh, 8]

                # angle sector bin 0..7
                k = sector[i, j]
                if k >= 0:
                    I[rh, k] += 1.0
                    if I[rh, k] == 1.0:
                        f[rh, k] = value
                    else:
                        f[rh, k] = f[rh, k] + (value - f[rh, k]) / I[rh, k]

    return f


def rotavg_fast(array):
    N = array.shape[0]
    rho, sector = _precompute_rotavg_lookup(N)
    return _rotavg_numba(array.astype(np.float64), rho, sector)


def padding_and_resizing_to_square_1024_pixel(img):
    mean = np.round(np.mean(img)).astype(np.uint8)

    img = PIL.Image.fromarray(img)
    if img.size[0] >= img.size[1]:
        a = 1024 / float(img.size[0])
        img = img.resize(
            (int(img.size[0] * a), int(img.size[1] * a)),
            PIL.Image.Resampling.LANCZOS
        )
    else:
        a = 1024 / float(img.size[1])
        img = img.resize(
            (int(img.size[0] * a), int(img.size[1] * a)),
            PIL.Image.Resampling.LANCZOS
        )

    img = np.asarray(img)
    h, w = img.shape
    w_c = int(w / 2)
    h_c = int(h / 2)

    if h > w:
        img_pad = np.full((h, h), mean, dtype=img.dtype)
        if w % 2 == 1:
            img_pad[:, h_c - w_c: h_c + w_c + 1] = img
        else:
            img_pad[:, h_c - w_c: h_c + w_c] = img
        img = img_pad
    elif h < w:
        img_pad = np.full((w, w), mean, dtype=img.dtype)
        if h % 2 == 1:
            img_pad[w_c - h_c: w_c + h_c + 1, :] = img
        else:
            img_pad[w_c - h_c: w_c + h_c, :] = img
        img = img_pad

    return img


def fourier_redies(img_gray, bin_size=2, cycles_min=10, cycles_max=256):
    """
    Fast version of fourier_redies() with same algorithmic behavior.
    """
    img_gray_resized = padding_and_resizing_to_square_1024_pixel(img_gray)

    power = np.fft.fftshift(np.fft.fft2(img_gray_resized.astype(np.float64)))
    A = rotavg_fast(np.abs(power) ** 2)
    A = A[:, 8].copy()

    if cycles_max > len(A):
        cycles_max = len(A)

    rang = np.arange(cycles_min, cycles_max + 1) - 1

    x_min = rang[0]
    x_max = rang[-1]

    using_range = A[rang]
    log_rang = np.log10(rang + 1)
    log_using_range = np.log10(using_range)

    bin_mean_bar, bin_x_val = calc_bin(A, bin_size, x_min, x_max)

    X = np.vstack([np.ones(len(bin_mean_bar[1])), np.log10(bin_mean_bar[0])]).T
    y = np.log10(bin_mean_bar[1])
    param = np.linalg.lstsq(X, y, rcond=None)[0]

    log_m_bin = param[1]
    log_n_bin = param[0]

    SIGMA = var_eigen(log_m_bin, log_n_bin, log_rang, log_using_range)
    SLOPE = log_m_bin

    return SIGMA, SLOPE
