""""
This is modified copy of CNN_qips.py from https://github.com/RBartho/Aesthetics-Toolbox/tree/main.

If you use this code please cite their original paper https://link.springer.com/article/10.3758/s13428-025-02632-3.

Compared to original code we vectorized and cached the code. In order to see comparison between original
and modifed versions please see "compare_results.ipynb".

"""
import numpy as np
from scipy.signal import correlate
from skimage.transform import resize
from numpy.lib.stride_tricks import sliding_window_view
from numba import njit


################################ helper functions #####################################


IMAGENET_MEAN = np.array([104.00698793, 116.66876762, 122.67891434], dtype=np.float64)


def resize_and_add_ImageNet_mean(img):
    """
    Keeps the same behavior as your original function:
    - resize to 512x512 with order=1
    - subtract ImageNet mean
    """
    img = resize(img, [512, 512], order=1)
    img = img - IMAGENET_MEAN
    return img


def preprocess_input_img(input_img):
    """
    Shared preprocessing:
    - RGB -> BGR
    - cast to float32 before resize (same as original)
    - resize and subtract ImageNet mean
    """
    input_img = input_img[:, :, (2, 1, 0)].astype(np.float32)
    input_img = resize_and_add_ImageNet_mean(input_img)
    return input_img


def conv2d_strict(input_img, kernel, bias):
    """
    Strictest version relative to your original:
    - keeps scipy.signal.correlate
    - keeps the same per-channel accumulation structure
    - only minor cleanup / refactoring

    This is the closest version if reproducibility matters most.
    """
    input_img = preprocess_input_img(input_img)

    in_height, in_width, in_channels = input_img.shape
    k_height, k_width, in_channels_k, out_channels = kernel.shape

    assert in_channels == in_channels_k

    out_height = int(np.ceil(float(in_height - k_height + 1) / float(4)))
    out_width = int(np.ceil(float(in_width - k_width + 1) / float(4)))

    output_data = np.zeros((out_height, out_width, out_channels), dtype=np.float64)

    for j in range(out_channels):
        for i in range(in_channels):
            output_data[:, :, j] += correlate(
                input_img[:, :, i],
                kernel[:, :, i, j],
                mode='valid'
            )[::4, ::4]

        output_data[:, :, j] += bias[j]

    output_data[output_data < 0] = 0

    output_data = np.swapaxes(output_data, 2, 0)
    output_data = np.swapaxes(output_data, 1, 2)

    return output_data


def conv2d_fast(input_img, kernel, bias):
    """
    Faster version using sliding windows + einsum.

    Same mathematical operation as your original correlation:
    - no kernel flipping (so this is correlation, not convolution)
    - stride 4 handled by slicing [::4, ::4]

    IMPORTANT:
    This should be numerically very close to the original, but because
    floating-point accumulation order changes, bit-for-bit identity is
    not guaranteed.
    """
    input_img = preprocess_input_img(input_img)

    in_height, in_width, in_channels = input_img.shape
    k_height, k_width, in_channels_k, out_channels = kernel.shape

    assert in_channels == in_channels_k

    # Create valid sliding windows over height and width
    # shape from sliding_window_view:
    # (out_h_valid, out_w_valid, channels, k_height, k_width)
    windows = sliding_window_view(input_img, (k_height, k_width), axis=(0, 1))

    # Reorder to: (out_h_valid, out_w_valid, k_height, k_width, channels)
    windows = np.transpose(windows, (0, 1, 3, 4, 2))

    # Apply stride 4 exactly like [::4, ::4] after valid correlation
    windows = windows[::4, ::4, :, :, :]

    # Correlation over spatial dims + channels:
    # windows: (oh, ow, kh, kw, c)
    # kernel : (kh, kw, c, oc)
    # result : (oh, ow, oc)
    output_data = np.einsum('xyhwc,hwco->xyo', windows, kernel, optimize=True)

    output_data += bias[np.newaxis, np.newaxis, :]

    # ReLU
    output_data[output_data < 0] = 0

    # Convert to shape (filters, dim1, dim2)
    output_data = np.transpose(output_data, (2, 0, 1))

    return output_data


@njit(cache=True)
def _max_pooling_numba(resp, h_starts, h_ends, w_starts, w_ends):
    i_filters, ih, iw = resp.shape
    patches = len(h_starts)

    max_pool_map = np.zeros((patches, patches, i_filters), dtype=np.float64)

    for h in range(patches):
        hs = h_starts[h]
        he = h_ends[h]
        for w in range(patches):
            ws = w_starts[w]
            we = w_ends[w]

            for b in range(i_filters):
                # initialize with first element in patch
                m = resp[b, hs, ws]
                for y in range(hs, he):
                    for x in range(ws, we):
                        v = resp[b, y, x]
                        if v > m:
                            m = v
                max_pool_map[h, w, b] = m

    return max_pool_map


def max_pooling_fast(resp, patches):
    """
    Same patch logic as your original max_pooling():
    patch_h = ih / float(patches), patch_w = iw / float(patches)
    slicing with int(ph):int(ph+patch_h), etc.
    """
    i_filters, ih, iw = resp.shape

    patch_h = ih / float(patches)
    patch_w = iw / float(patches)

    h_starts = np.array([int(h * patch_h) for h in range(patches)], dtype=np.int64)
    h_ends   = np.array([int(h * patch_h + patch_h) for h in range(patches)], dtype=np.int64)
    w_starts = np.array([int(w * patch_w) for w in range(patches)], dtype=np.int64)
    w_ends   = np.array([int(w * patch_w + patch_w) for w in range(patches)], dtype=np.int64)

    max_pool_map = _max_pooling_numba(
        resp.astype(np.float64),
        h_starts, h_ends, w_starts, w_ends
    )

    max_pool_map_sum = np.sum(max_pool_map, axis=2)
    normalized_max_pool_map = max_pool_map / max_pool_map_sum[:, :, np.newaxis]

    return max_pool_map, normalized_max_pool_map


def get_differences(max_pooling_map_orig, max_pooling_map_flip):
    assert max_pooling_map_orig.shape == max_pooling_map_flip.shape
    sum_abs = np.sum(np.abs(max_pooling_map_orig - max_pooling_map_flip))
    sum_max = np.sum(np.maximum(max_pooling_map_orig, max_pooling_map_flip))
    return 1.0 - sum_abs / sum_max
    
def CNN_symmetry(img, kernel, bias, conv_fn=conv2d_fast):
    """
    Fast symmetry computation.
    Use conv_fn=conv2d_strict if you want the closest possible behavior
    to your original implementation.
    """
    # original
    resp_orig = conv_fn(img, kernel, bias)
    max_pooling_map_orig, _ = max_pooling_fast(resp_orig, patches=17)

    # left-right
    img_lr = np.fliplr(img)
    resp_lr = conv_fn(img_lr, kernel, bias)
    max_pooling_map_lr, _ = max_pooling_fast(resp_lr, patches=17)
    sym_lr = get_differences(max_pooling_map_orig, max_pooling_map_lr)

    # up-down
    img_ud = np.flipud(img)
    resp_ud = conv_fn(img_ud, kernel, bias)
    max_pooling_map_ud, _ = max_pooling_fast(resp_ud, patches=17)
    sym_ud = get_differences(max_pooling_map_orig, max_pooling_map_ud)

    # left-right + up-down
    img_lrud = np.fliplr(np.flipud(img))
    resp_lrud = conv_fn(img_lrud, kernel, bias)
    max_pooling_map_lrud, _ = max_pooling_fast(resp_lrud, patches=17)
    sym_lrud = get_differences(max_pooling_map_orig, max_pooling_map_lrud)

    return sym_lr, sym_ud, sym_lrud


def CNN_Variance(img, kernel, bias):
    resp_orig = conv2d_fast(img, kernel, bias)
    _, normalized_max_pooling_map = max_pooling_fast(resp_orig, patches=22)
    sparseness = np.var(normalized_max_pooling_map)
    
    _, normalized_max_pooling_map = max_pooling_fast(resp_orig, patches=12)
    variability = np.median(np.var(normalized_max_pooling_map, axis=(0, 1)))

    return sparseness, variability


def CNN_selfsimilarity(img, kernel, bias):
    """
    Vectorized version of CNN_selfsimilarity().
    histogram_ground shape: (1,1,n) or compatible
    histogram_level shape : (ph,pw,n)
    """
    resp_orig = conv2d_fast(img, kernel, bias)
    _, normalized_max_pooling_map_8 = max_pooling_fast(resp_orig, patches=8)
    _, normalized_max_pooling_map_1 = max_pooling_fast(resp_orig, patches=1)
    # broadcast histogram_ground across all patches
    hiks = np.sum(np.minimum(normalized_max_pooling_map_1, normalized_max_pooling_map_8), axis=2)
    sesim = np.median(hiks)
    return [sesim]