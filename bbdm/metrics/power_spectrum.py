import numpy as np


def azimuthal_average(power_2d):
    h, w   = power_2d.shape
    cy, cx = h // 2, w // 2
    y, x   = np.indices(power_2d.shape)
    r      = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    r_max  = min(cx, cy)
    mask   = r < r_max
    rapsd  = np.bincount(r[mask], weights=power_2d[mask]) / np.bincount(r[mask])
    freqs  = np.arange(r_max) / r_max
    return rapsd, freqs


def compute_power_spectrum(patch):
    """patch: (H, W) numpy array"""
    window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
    fft2   = np.fft.fftshift(np.fft.fft2(patch * window))
    power  = np.abs(fft2)**2
    return azimuthal_average(power)


def transfer_function(c_ell_pred, c_ell_true):
    """T_ell = C_ell_pred / C_ell_true. Должна быть близка к 1."""
    return c_ell_pred / (c_ell_true + 1e-20)


def cross_correlation(patch_pred, patch_true):

    fft_pred = np.fft.fftshift(np.fft.fft2(patch_pred))
    fft_true = np.fft.fftshift(np.fft.fft2(patch_true))

    cross      = np.real(fft_pred * np.conj(fft_true))
    power_pred = np.abs(fft_pred)**2
    power_true = np.abs(fft_true)**2

    r_2d = cross / (np.sqrt(power_pred * power_true) + 1e-20)

    h, w   = r_2d.shape
    cy, cx = h // 2, w // 2
    y, x   = np.indices(r_2d.shape)
    r      = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    r_max  = min(cx, cy)
    mask   = r < r_max

    r_ell = np.bincount(r[mask], weights=r_2d[mask]) / np.bincount(r[mask])
    freqs = np.arange(r_max) / r_max
    return r_ell, freqs
