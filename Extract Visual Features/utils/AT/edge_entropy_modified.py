from PIL import Image
import numpy as np
import numpy as np
from scipy.ndimage import convolve
import PIL
import warnings
from functools import lru_cache
from numba import njit


def create_gabor(size, theta=0, octave=3):
    amplitude = 1.0
    phase = np.pi / 2.0
    frequency = 0.5 ** octave
    hrsf = 4
    sigma = (
        1 / (np.pi * frequency)
        * np.sqrt(np.log(2) / 2)
        * (2.0 ** hrsf + 1)
        / (2.0 ** hrsf - 1)
    )

    valsy = np.linspace(-size // 2 + 1, size // 2, size)
    valsx = np.linspace(-size // 2 + 1, size // 2, size)
    xgr, ygr = np.meshgrid(valsx, valsy)

    omega = 2 * np.pi * frequency
    gaussian = np.exp(-(xgr * xgr + ygr * ygr) / (2 * sigma * sigma))
    slant = xgr * (omega * np.sin(theta)) + ygr * (omega * np.cos(theta))

    gabor = np.round(gaussian, decimals=4) * amplitude * np.cos(slant + phase)
    return np.round(gabor, decimals=4)


@lru_cache(maxsize=None)
def create_filterbank_cached(flt_size=31, num_filters=24):
    flt_raw = np.zeros((num_filters, flt_size, flt_size), dtype=np.float64)
    bins_vec = np.linspace(0, 2 * np.pi, num_filters + 1)[:-1]
    for i in range(num_filters):
        flt_raw[i, :, :] = create_gabor(flt_size, theta=bins_vec[i], octave=3)
    return flt_raw


def run_filterbank(flt_raw, img):
    h, w = img.shape
    num_filters = flt_raw.shape[0]
    image_flt = np.empty((num_filters, h, w), dtype=np.float64)

    for i in range(num_filters):
        # keep scipy.ndimage.convolve to preserve behavior/results
        image_flt[i, :, :] = convolve(img, flt_raw[i, :, :])

    resp_bin = np.argmax(image_flt, axis=0)
    resp_val = np.max(image_flt, axis=0)
    return resp_bin, resp_val

@njit(cache=True)
def _do_counting_numba(ey, ex, orientations, values, CIRC_BINS=48, GABOR_BINS=24, MAX_DIAGONAL=500):
    n = ex.size
    counts = np.zeros((MAX_DIAGONAL, CIRC_BINS, GABOR_BINS), dtype=np.float64)

    for cp in range(n):
        ey_cp = ey[cp]
        ex_cp = ex[cp]
        ori_cp = orientations[cp]
        val_cp = values[cp]

        ori_shift = (ori_cp / float(GABOR_BINS)) * CIRC_BINS

        for j in range(n):
            dy = ey[j] - ey_cp
            dx = ex[j] - ex_cp

            # same logic as original
            dist = int(np.round(np.sqrt(dx * dx + dy * dy)))
            if dist >= MAX_DIAGONAL:
                dist = MAX_DIAGONAL - 1

            direction = int(np.round(np.arctan2(dy, dx) / (2.0 * np.pi) * CIRC_BINS + ori_shift))
            direction = (direction + CIRC_BINS) % CIRC_BINS

            ori_rel = orientations[j] - ori_cp
            ori_rel = (ori_rel + GABOR_BINS) % GABOR_BINS

            counts[dist, direction, ori_rel] += values[j] * val_cp

    return counts


def do_counting_fast(resp_val, resp_bin, CIRC_BINS=48, GABOR_BINS=24, MAX_DIAGONAL=500):
    """
    Same logic as original do_counting, but the heavy nested loop is JIT-compiled.
    """
    resp_val = resp_val.copy()

    flat = resp_val.ravel()
    # same cutoff value as sorting, but faster
    cutoff = np.partition(flat, flat.size - 10000)[flat.size - 10000]

    resp_val[resp_val < cutoff] = 0
    ey, ex = resp_val.nonzero()

    orientations = resp_bin[ey, ex].astype(np.int64)
    values = resp_val[ey, ex].astype(np.float64)

    counts = _do_counting_numba(
        ey.astype(np.int64),
        ex.astype(np.int64),
        orientations,
        values,
        CIRC_BINS=CIRC_BINS,
        GABOR_BINS=GABOR_BINS,
        MAX_DIAGONAL=MAX_DIAGONAL,
    )

    return counts, resp_val


def entropy(a):
    if np.sum(a) != 1.0 and np.sum(a) > 0:
        a = a / np.sum(a)
    v = a > 0.0
    return -np.sum(a[v] * np.log2(a[v]))


def do_statistics_fast(counts):
    counts_sum = np.sum(counts, axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized_counts = counts / counts_sum[:, :, np.newaxis]
        term = np.where(normalized_counts > 0, normalized_counts * np.log2(normalized_counts), 0.0)
        shannon = -np.sum(term, axis=2)

    shannon[counts_sum <= 1] = np.nan
    return shannon


def edge_resize(img_gray_np, max_pixels=300 * 400):
    if max_pixels is not None:
        img_gray_PIL = PIL.Image.fromarray(img_gray_np)
        s0, s1 = img_gray_PIL.size
        a = np.sqrt(max_pixels / float(s0 * s1))
        img_gray_PIL_rez = img_gray_PIL.resize((int(s0 * a), int(s1 * a)), PIL.Image.LANCZOS)
        img_gray_np = np.asarray(img_gray_PIL_rez, dtype="float")
    return img_gray_np

def edge_density(img_gray: np.array, GABOR_BINS: int =24) -> np.array:
    flt_raw = create_filterbank_cached(31, GABOR_BINS)
    img = edge_resize(img_gray)

    resp_bin, resp_val = run_filterbank(flt_raw, img)

    # edge density before resp_val is modified
    normalize_fac = float(resp_val.shape[0] * resp_val.shape[1])
    
    return np.sum(resp_val) / normalize_fac
    

def first_and_second_order_entropy(img_gray, GABOR_BINS=24):
    """
    Same outputs/logic as your original function, but substantially faster.
    """
    flt_raw = create_filterbank_cached(31, GABOR_BINS)
    img = edge_resize(img_gray)

    resp_bin, resp_val = run_filterbank(flt_raw, img)

    counts, resp_val = do_counting_fast(resp_val, resp_bin, GABOR_BINS=GABOR_BINS)

    # first order entropy (same logic)
    first_order_bin = np.zeros(GABOR_BINS, dtype=np.float64)
    for b in range(GABOR_BINS):
        first_order_bin[b] = np.sum(resp_val[resp_bin == b])
    first_order = entropy(first_order_bin)

    # second order entropy
    shannon_nan = do_statistics_fast(counts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        second_order = np.nanmean(np.nanmean(shannon_nan, axis=1)[20:240])

    return first_order, second_order


def first_and_second_order_entropy_and_edge_density(img_gray, GABOR_BINS=24):
    """
    Fast version of the original public function.
    """
    flt_raw = create_filterbank_cached(31, GABOR_BINS)
    img = edge_resize(img_gray)
    resp_bin, resp_val = run_filterbank(flt_raw, img)

    normalize_fac = float(resp_val.shape[0] * resp_val.shape[1])
    
    edge_d  = np.sum(resp_val) / normalize_fac


    counts, resp_val = do_counting_fast(resp_val, resp_bin, GABOR_BINS=GABOR_BINS)
    first_order_bin = np.zeros(GABOR_BINS, dtype=np.float64)
    for b in range(GABOR_BINS):
        first_order_bin[b] = np.sum(resp_val[resp_bin == b])
    first_order = entropy(first_order_bin)

    shannon_nan = do_statistics_fast(counts)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        second_order = np.nanmean(np.nanmean(shannon_nan, axis=1)[20:240])

    return first_order, second_order, edge_d
