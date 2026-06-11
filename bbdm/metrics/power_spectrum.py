import numpy as np


def _radial_bins(shape):
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    r_max = min(cx, cy)
    mask = r < r_max
    return r, mask, r_max


def _bin_average(values_2d, r, mask, r_max):
    counts = np.bincount(r[mask], minlength=r_max)
    sums = np.bincount(r[mask], weights=values_2d[mask], minlength=r_max)
    return sums[:r_max] / np.maximum(counts[:r_max], 1)


def azimuthal_average(power_2d):
    r, mask, r_max = _radial_bins(power_2d.shape)
    rapsd = _bin_average(power_2d, r, mask, r_max)
    freqs = np.arange(r_max) / r_max
    return rapsd, freqs


def _hanning_window(shape):
    return np.outer(np.hanning(shape[0]), np.hanning(shape[1]))


def compute_power_spectrum(patch):
    """patch: (H, W) numpy array"""
    window = _hanning_window(patch.shape)
    fft2 = np.fft.fftshift(np.fft.fft2(patch * window))
    power = np.abs(fft2) ** 2
    return azimuthal_average(power)


def transfer_function(c_ell_pred, c_ell_true):
    """T_ell = C_ell_pred / C_ell_true. Should be close to 1."""
    return c_ell_pred / (c_ell_true + 1e-20)


def cross_correlation(patch_pred, patch_true):

    window = _hanning_window(patch_pred.shape)

    fft_pred = np.fft.fftshift(np.fft.fft2(patch_pred * window))
    fft_true = np.fft.fftshift(np.fft.fft2(patch_true * window))

    cross = np.real(fft_pred * np.conj(fft_true))
    power_pred = np.abs(fft_pred) ** 2
    power_true = np.abs(fft_true) ** 2

    r, mask, r_max = _radial_bins(cross.shape)
    cross_ell = _bin_average(cross, r, mask, r_max)
    p_pred_ell = _bin_average(power_pred, r, mask, r_max)
    p_true_ell = _bin_average(power_true, r, mask, r_max)

    r_ell = cross_ell / (np.sqrt(p_pred_ell * p_true_ell) + 1e-20)
    freqs = np.arange(r_max) / r_max
    return r_ell, freqs