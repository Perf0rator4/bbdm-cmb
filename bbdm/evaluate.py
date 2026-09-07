"""Спектральная и попиксельная оценка, усреднённая по многим патчам.

Все спектральные величины усредняются по >= 30-60 патчам. Одиночный патч
для этого непригоден: на высоких ell мощность мала, и оценка спектра
одного патча гуляет так, что реальная систематика и шум оценки визуально
неотличимы.

Усреднение всегда идёт по схеме "среднее спектров, потом отношение", а не
"среднее отношений": во втором случае один патч с почти нулевой мощностью
цели в каком-нибудь бине уносит среднее в бесконечность.
"""

import contextlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from bbdm.metrics.power_spectrum import (
    DEFAULT_BANDS,
    band_average,
    compute_power_spectrum,
    cross_spectrum_terms,
)
from bbdm.sample import sample_batch


@contextlib.contextmanager
def _temporary_eta(bbdm, eta):
    """Временно подменяет bbdm.eta -- для абляций без переобучения."""
    if eta is None:
        yield
        return
    original = bbdm.eta
    bbdm.eta = float(eta)
    try:
        yield
    finally:
        bbdm.eta = original


def _iter_batches(dataset, indices, batch_size):
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        pairs = [dataset[i] for i in chunk]
        x0 = torch.stack([p[0] for p in pairs])
        y = torch.stack([p[1] for p in pairs])
        yield x0, y


@torch.no_grad()
def evaluate_spectra(
    bbdm,
    dataset,
    mu,
    sigma,
    n_patches=50,
    S=200,
    device="cuda",
    batch_size=4,
    eta=None,
    seed=0,
    indices=None,
    bands=DEFAULT_BANDS,
    progress=True,
):
    """Transfer Function и кросс-корреляция, усреднённые по n_patches патчам.

    Args:
        eta: если задано, временно подменяет bbdm.eta (абляция без
            переобучения -- меняется только обратный процесс).
        indices: явный список индексов патчей; по умолчанию первые
            n_patches. Один и тот же список обязателен при сравнении
            вариантов, иначе сравниваются разные куски неба.

    Returns:
        dict с ключами freqs, tf, r_ell, ps_pred, ps_target, ps_input,
        tf_bands / r_bands (среднее и std по патчам в полосах), bands.
    """
    from tqdm.auto import tqdm

    if indices is None:
        indices = list(range(min(n_patches, len(dataset))))

    bbdm = bbdm.to(device).eval()

    sum_pred = sum_target = sum_input = sum_cross = None
    freqs = None
    per_patch_tf_bands = []
    per_patch_r_bands = []

    batches = _iter_batches(dataset, indices, batch_size)
    if progress:
        n_batches = (len(indices) + batch_size - 1) // batch_size
        batches = tqdm(batches, total=n_batches, desc=f"eval (eta={eta})")

    with _temporary_eta(bbdm, eta):
        for b_i, (x0, y) in enumerate(batches):
            # Сид зависит от номера батча, но не от eta: варианты сравниваются
            # на одинаковых патчах и сопоставимом шуме.
            pred = sample_batch(
                bbdm, x0, mu, sigma, S=S, device=device, seed=seed + b_i
            )
            target = y[:, 0].numpy() * sigma + mu
            source = x0[:, 0].numpy() * sigma + mu

            for k in range(pred.shape[0]):
                ps_in, freqs = compute_power_spectrum(source[k])
                cross_ell, ps_pred, ps_target, _ = cross_spectrum_terms(
                    pred[k], target[k]
                )

                if sum_pred is None:
                    sum_pred = np.zeros_like(ps_pred)
                    sum_target = np.zeros_like(ps_target)
                    sum_input = np.zeros_like(ps_in)
                    sum_cross = np.zeros_like(cross_ell)

                sum_pred += ps_pred
                sum_target += ps_target
                sum_input += ps_in
                sum_cross += cross_ell

                tf_k = ps_pred / (ps_target + 1e-20)
                r_k = cross_ell / (np.sqrt(ps_pred * ps_target) + 1e-20)
                per_patch_tf_bands.append(band_average(tf_k, freqs, bands))
                per_patch_r_bands.append(band_average(r_k, freqs, bands))

    n = len(per_patch_tf_bands)
    if n == 0:
        raise ValueError("No patches evaluated")

    mean_pred = sum_pred / n
    mean_target = sum_target / n
    mean_input = sum_input / n
    mean_cross = sum_cross / n

    tf = mean_pred / (mean_target + 1e-20)
    r_ell = mean_cross / (np.sqrt(mean_pred * mean_target) + 1e-20)

    tf_bands = np.asarray(per_patch_tf_bands)
    r_bands = np.asarray(per_patch_r_bands)

    return {
        "freqs": freqs,
        "tf": tf,
        "r_ell": r_ell,
        "ps_pred": mean_pred,
        "ps_target": mean_target,
        "ps_input": mean_input,
        "bands": list(bands),
        # Полосные значения TF/r_ell от УСРЕДНЁННЫХ спектров -- основная цифра.
        "tf_bands": band_average(tf, freqs, bands),
        "r_bands": band_average(r_ell, freqs, bands),
        # Разброс поштучных полосных значений -- только как оценка
        # стабильности, не как доверительный интервал для tf_bands.
        "tf_bands_per_patch_mean": tf_bands.mean(axis=0).tolist(),
        "tf_bands_per_patch_std": tf_bands.std(axis=0).tolist(),
        "r_bands_per_patch_mean": r_bands.mean(axis=0).tolist(),
        "r_bands_per_patch_std": r_bands.std(axis=0).tolist(),
        "n_patches": n,
        "eta": bbdm.eta if eta is None else float(eta),
    }


def print_band_table(results, label=""):
    """Таблица TF и r_ell по частотным полосам."""
    header = f"  {'band':<14}{'TF':>10}{'TF ±':>9}{'r_ell':>10}{'r_ell ±':>9}"
    print(f"\n{label} (n={results['n_patches']} patches, eta={results['eta']})")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, (lo, hi) in enumerate(results["bands"]):
        print(
            f"  {lo:.3f}-{hi:.2f}   "
            f"{results['tf_bands'][i]:>9.3f}"
            f"{results['tf_bands_per_patch_std'][i]:>9.3f}"
            f"{results['r_bands'][i]:>10.3f}"
            f"{results['r_bands_per_patch_std'][i]:>9.3f}"
        )


def plot_spectral_comparison(results_list, labels, title=None):
    """RAPSD, Transfer Function и r_ell для нескольких вариантов рядом.

    Именно эту тройку графиков нужно смотреть вместе: TF ~ 1 сам по себе
    ничего не доказывает, если лишняя мощность некоррелирована с целью --
    тогда r_ell на тех же ell проседает.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(results_list)))

    ref = results_list[0]
    axes[0].loglog(ref["freqs"][1:], ref["ps_input"][1:], color="tab:blue",
                   lw=1.2, label="Planck (input)")
    axes[0].loglog(ref["freqs"][1:], ref["ps_target"][1:], color="tab:red",
                   lw=1.6, label="ACT+Planck (target)")

    for res, label, color in zip(results_list, labels, colors):
        axes[0].loglog(res["freqs"][1:], res["ps_pred"][1:], color=color,
                       lw=1.2, ls="--", label=f"pred, {label}")
        axes[1].semilogx(res["freqs"][1:], res["tf"][1:], color=color,
                         lw=1.4, label=label)
        axes[2].semilogx(res["freqs"][1:], res["r_ell"][1:], color=color,
                         lw=1.4, label=label)

    axes[0].set(xlabel="freq (Nyquist units)", ylabel="Power",
                title=f"RAPSD (mean over {ref['n_patches']} patches)")
    axes[1].axhline(1.0, color="gray", ls="--", alpha=0.7)
    axes[1].set(xlabel="freq (Nyquist units)", ylabel="T_ell",
                title="Transfer Function", ylim=(0, 2.5))
    axes[2].axhline(1.0, color="gray", ls="--", alpha=0.7)
    axes[2].axhline(0.0, color="gray", ls=":", alpha=0.5)
    axes[2].set(xlabel="freq (Nyquist units)", ylabel="r_ell",
                title="Cross-correlation", ylim=(-0.1, 1.05))

    for ax in axes:
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig


@torch.no_grad()
def evaluate_image_metrics(
    bbdm, dataset, mu, sigma, n_patches=50, S=200, device="cuda",
    batch_size=4, indices=None, seed=0, progress=True,
):
    """PSNR и SSIM по валидным (незамаскированным) пикселям.

    SSIM для одно-ко-многим стохастической генерации -- слабая метрика:
    она штрафует любую реализацию, отличную от конкретной наблюдённой,
    даже если её статистика идеальна. Приводится для сопоставимости с
    литературой, а не как основной критерий.
    """
    from skimage.metrics import structural_similarity as ssim_fn
    from tqdm.auto import tqdm

    if indices is None:
        indices = list(range(min(n_patches, len(dataset))))

    bbdm = bbdm.to(device).eval()
    psnr_list, ssim_list = [], []

    batches = _iter_batches(dataset, indices, batch_size)
    if progress:
        n_batches = (len(indices) + batch_size - 1) // batch_size
        batches = tqdm(batches, total=n_batches, desc="PSNR/SSIM")

    for b_i, (x0, y) in enumerate(batches):
        pred = sample_batch(bbdm, x0, mu, sigma, S=S, device=device, seed=seed + b_i)
        target = y[:, 0].numpy() * sigma + mu

        for k in range(pred.shape[0]):
            t_k, p_k = target[k], pred[k]
            valid = np.abs(t_k) > 0.5
            if valid.sum() < 1000:
                continue

            data_range = float(t_k[valid].max() - t_k[valid].min())
            if data_range < 1e-6:
                continue

            mse = float(np.mean((t_k[valid] - p_k[valid]) ** 2))
            psnr_list.append(10 * np.log10(data_range ** 2 / (mse + 1e-12)))

            masked_pred = np.where(valid, p_k, 0.0)
            ssim_list.append(
                ssim_fn(t_k, masked_pred, data_range=data_range)
            )

    return {
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std": float(np.std(psnr_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std": float(np.std(ssim_list)),
        "n_patches": len(psnr_list),
    }


# --------------------------------------------------------------- диагностика


@torch.no_grad()
def _mean_target_rapsd(bbdm, dataset, indices, batch_size, device):
    """Средний RAPSD цели в НОРМАЛИЗОВАННЫХ единицах, бинами BBDM.rapsd."""
    total, n = None, 0
    for _, y in _iter_batches(dataset, indices, batch_size):
        ps = bbdm.rapsd(y.to(device))
        total = ps if total is None else total + ps
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def bridge_snr(
    bbdm, dataset, n_patches=32, batch_size=8, device="cpu",
    bands=DEFAULT_BANDS, indices=None,
):
    """Сколько сигнала на каждом ell переживает шум моста, как функция t.

    Мост подмешивает **белый** шум с пиксельной дисперсией
    `delta_t = 2*s*m_t*(1-m_t)`, тогда как сигнальная компонента в `x_t`
    равна `(1-m_t)*y`. Спектр CMB+ACT падает на ~6 порядков от низких ell к
    высоким, поэтому плоский шумовой пол хоронит высокие ell почти при
    любом t. Доля t, при которых мода ещё различима, и есть доля обучающих
    шагов, на которых сеть вообще может чему-то научиться на этом масштабе.

    Это количественное обоснование пункта 7.2 (частотно-зависимое
    расписание `delta_t`): при скалярном расписании отношение сигнал/шум
    на высоких ell задано целиком спектром данных, а не выбором модели.

    Returns:
        dict с freqs, snr (матрица t x bin), t_values, полосными долями
        `frac_usable` и `best_snr`.
    """
    if indices is None:
        indices = list(range(min(n_patches, len(dataset))))

    bbdm = bbdm.to(device).eval()
    ps_target = _mean_target_rapsd(bbdm, dataset, indices, batch_size, device)
    ps_target = ps_target.cpu().numpy()

    # Белый шум с дисперсией v, пропущенный через то же окно Ханна и
    # ortho-FFT, даёт ПЛОСКИЙ спектр на уровне v * <w^2> (по Парсевалю).
    h, w = dataset[indices[0]][0].shape[-2:]
    window = bbdm._hann_window(h, w, torch.device(device), torch.float32)
    w2 = float((window ** 2).mean())

    t_values = np.arange(1, bbdm.T + 1)
    m = t_values / bbdm.T
    delta = 2 * bbdm.s * (m - m ** 2)

    signal = ((1 - m) ** 2)[:, None] * ps_target[None, :]
    noise = (delta * w2)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(noise > 0, signal / np.maximum(noise, 1e-300), 0.0)

    r_max = len(ps_target)
    freqs = np.arange(r_max) / r_max

    usable = (snr > 1.0).mean(axis=0)      # доля t на каждом бине
    return {
        "freqs": freqs,
        "t_values": t_values,
        "snr": snr,
        "ps_target": ps_target,
        "bands": list(bands),
        "frac_usable": band_average(usable, freqs, bands),
        "best_snr": band_average(snr.max(axis=0), freqs, bands),
    }


def print_bridge_snr(result):
    """Таблица: доля обучающих шагов, на которых полоса выше шума моста."""
    print(f"\n{'band':>14}{'usable t':>12}{'best SNR':>12}")
    print("  " + "-" * 36)
    for i, (lo, hi) in enumerate(result["bands"]):
        print(
            f"{lo:>7.3f}-{hi:<6.2f}"
            f"{result['frac_usable'][i] * 100:>10.1f}%"
            f"{result['best_snr'][i]:>12.3g}"
        )
    print("\n  'usable t' -- доля шагов 1..T, на которых сигнал полосы выше")
    print("  белого шума моста. Там, где она мала, сеть почти никогда не")
    print("  видит этот масштаб неиспорченным (см. §7.2).")


@torch.no_grad()
def diagnose_prediction_spectrum(
    bbdm, dataset, t_values=(1, 10, 50, 100, 250, 500, 750, 999),
    n_patches=16, batch_size=4, device="cuda", bands=DEFAULT_BANDS,
    indices=None, seed=0,
):
    """Мощность ОДНОШАГОВОГО предсказания model(x_t, t) относительно цели.

    Решающая проверка того, откуда берётся избыток мощности на высоких ell.

    Апостериорное среднее на моде с мощностью p равно `k_t * x_t` с
    `k_t = (1-m)p / ((1-m)^2 p + delta_t)`. На высоких ell, где p много
    меньше шума моста, `k_t ~ p/m` -- то есть сеть ОБЯЗАНА давить свой вход
    в десятки раз. Если она этого не делает, белый шум моста проходит на
    выход как есть, и `TF` выходит примерно на отношение "шум моста /
    мощность цели", а `r_ell` падает в ноль: этот избыток по построению
    независим от цели.

    Спектральный член лосса подталкивает ровно к этому. Он сравнивает
    спектры, усреднённые по батчу со СМЕШАННЫМИ t, то есть ограничивает
    только среднее по t; а самый дешёвый источник высокочастотной мощности
    для сети -- шум моста, уже присутствующий в её входе. Прекратить его
    давить дешевле, чем синтезировать структуру.

    `ideal` -- отношение для точного апостериорного среднего в режиме
    "вход не несёт информации" (справедливо на высоких ell). На низких ell
    это НИЖНЯЯ оценка: там Planck информативен и настоящее апостериорное
    среднее мощнее. Диагноз подтверждается, если на высоких ell `actual`
    много больше `ideal` и почти не убывает с ростом t.

    Returns:
        dict с t_values, ratio_bands (t x band), ideal_bands (t x band).
    """
    from tqdm.auto import tqdm

    if indices is None:
        indices = list(range(min(n_patches, len(dataset))))

    bbdm = bbdm.to(device).eval()
    ps_target = _mean_target_rapsd(bbdm, dataset, indices, batch_size, device)

    h, w = dataset[indices[0]][0].shape[-2:]
    window = bbdm._hann_window(h, w, torch.device(device), torch.float32)
    w2 = float((window ** 2).mean())

    ratio_bands, ideal_bands = [], []
    freqs = np.arange(len(ps_target)) / len(ps_target)
    ps_target_np = ps_target.cpu().numpy()

    for t_val in tqdm(list(t_values), desc="one-step spectrum"):
        total, n = None, 0
        for b_i, (x0, y) in enumerate(_iter_batches(dataset, indices, batch_size)):
            x0, y = x0.to(device), y.to(device)
            g = torch.Generator(device=x0.device)
            g.manual_seed(seed + b_i)
            t = torch.full((x0.shape[0],), int(t_val), device=x0.device,
                           dtype=torch.long)
            x_t, _ = bbdm.q_sample(x0, y, t, generator=g)
            ps = bbdm.rapsd(bbdm.model(x_t, t))
            total = ps if total is None else total + ps
            n += 1

        ratio = (total / max(n, 1)).cpu().numpy() / (ps_target_np + 1e-20)
        ratio_bands.append(band_average(ratio, freqs, bands))

        m = int(t_val) / bbdm.T
        delta = 2 * bbdm.s * (m - m ** 2)
        sig = (1 - m) ** 2 * ps_target_np
        ideal = sig / (sig + delta * w2 + 1e-20)
        ideal_bands.append(band_average(ideal, freqs, bands))

    return {
        "t_values": list(t_values),
        "bands": list(bands),
        "ratio_bands": ratio_bands,
        "ideal_bands": ideal_bands,
        "freqs": freqs,
    }


def print_prediction_spectrum(result):
    """Таблица "мощность предсказания / мощность цели" по t и полосам."""
    bands = result["bands"]
    head = f"{'t':>6}" + "".join(f"{f'{lo}-{hi}':>16}" for lo, hi in bands)
    print("\nOne-step prediction power / target power   (actual | ideal)")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for i, t in enumerate(result["t_values"]):
        row = "".join(
            f"{a:>8.2f} |{b:>6.2f}"
            for a, b in zip(result["ratio_bands"][i], result["ideal_bands"][i])
        )
        print(f"{t:>6}{row}")
    print("\n  'ideal' падает к 0 с ростом t -- при большом t апостериорное")
    print("  среднее ОБЯЗАНО быть гладким, потому что шум моста хоронит")
    print("  высокие ell. Если 'actual' там много больше 'ideal' и почти не")
    print("  убывает с t, сеть пропускает белый шум моста на выход вместо")
    print("  того, чтобы его давить: это и есть источник избытка мощности,")
    print("  и он по построению не коррелирует с целью (r_ell -> 0).")
