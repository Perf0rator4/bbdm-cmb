"""Спектральные метрики: RAPSD, Transfer Function, кросс-корреляция.

Единица частоты во всех функциях -- доля частоты Найквиста: бин с радиусом
r в фурье-плоскости получает freq = r / r_max, где r_max = N / 2. То есть
freq = 1.0 соответствует 0.5 цикла/пиксель. Эта нормировка используется во
всех записанных прогонах, менять её -- значит сделать старые числа
несравнимыми с новыми.
"""

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


def _windowed_fft(patch):
    window = _hanning_window(patch.shape)
    return np.fft.fftshift(np.fft.fft2(patch * window))


def compute_power_spectrum(patch):
    """RAPSD одного патча.

    Args:
        patch: (H, W) numpy array.

    Returns:
        (rapsd, freqs), обе длины r_max = min(H, W) // 2.
    """
    power = np.abs(_windowed_fft(patch)) ** 2
    return azimuthal_average(power)


def transfer_function(c_ell_pred, c_ell_true):
    """T_ell = C_ell_pred / C_ell_true. В идеале == 1 на всех ell.

    Для усреднения по нескольким патчам усредняйте c_ell_pred и c_ell_true
    ПО ОТДЕЛЬНОСТИ, а отношение берите от средних (см. evaluate.py).
    Среднее от поштучных отношений разъезжается там, где у отдельного патча
    мощность цели близка к нулю.
    """
    return c_ell_pred / (c_ell_true + 1e-20)


def cross_spectrum_terms(patch_pred, patch_true):
    """Три слагаемых, из которых собирается r_ell, по отдельности.

    Возвращает (cross_ell, p_pred_ell, p_true_ell, freqs). Нужны отдельно,
    чтобы усреднять их по патчам ДО деления -- поштучное отношение
    смещено, ровно как и у Transfer Function.
    """
    fft_pred = _windowed_fft(patch_pred)
    fft_true = _windowed_fft(patch_true)

    cross = np.real(fft_pred * np.conj(fft_true))
    power_pred = np.abs(fft_pred) ** 2
    power_true = np.abs(fft_true) ** 2

    r, mask, r_max = _radial_bins(cross.shape)
    cross_ell = _bin_average(cross, r, mask, r_max)
    p_pred_ell = _bin_average(power_pred, r, mask, r_max)
    p_true_ell = _bin_average(power_true, r, mask, r_max)
    freqs = np.arange(r_max) / r_max

    return cross_ell, p_pred_ell, p_true_ell, freqs


def cross_correlation(patch_pred, patch_true):
    """r_ell = <Re(F_pred F_true*)>_ell / sqrt(<|F_pred|^2>_ell <|F_true|^2>_ell).

    Числитель и оба спектра биннятся ДО деления -- поштучное отношение
    Re(F_pred F_true*) / sqrt(|F_pred|^2 |F_true|^2) с последующим
    усреднением по бину даёт смещённую оценку. Окно Ханна применяется
    ко всем трём членам одинаково, как в compute_power_spectrum.

    Тождество для проверки: cross_correlation(p, p) == 1 для любого p.
    """
    cross_ell, p_pred_ell, p_true_ell, freqs = cross_spectrum_terms(
        patch_pred, patch_true
    )
    r_ell = cross_ell / (np.sqrt(p_pred_ell * p_true_ell) + 1e-20)
    return r_ell, freqs


def band_average(values, freqs, bands):
    """Среднее `values` по частотным полосам.

    Args:
        values: массив длины r_max (например, TF или r_ell).
        freqs: соответствующие частоты.
        bands: список пар (lo, hi), полосы полуоткрытые [lo, hi).

    Returns:
        Список средних по каждой полосе (nan, если полоса пуста).
    """
    out = []
    for lo, hi in bands:
        sel = (freqs >= lo) & (freqs < hi)
        out.append(float(np.mean(values[sel])) if sel.any() else float("nan"))
    return out


DEFAULT_BANDS = [(0.005, 0.03), (0.03, 0.10), (0.10, 0.30), (0.30, 1.00)]
