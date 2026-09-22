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
import math

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


def print_tf_across_S(results_by_S, bands=None):
    """TF и r_ell рядом для нескольких S (число шагов сэмплирования).

    Каждый `results_by_S[S]` -- результат `evaluate_spectra(..., S=S)` на
    ОДНОМ И ТОМ ЖЕ чекпоинте и тех же `indices`/`seed`; менять S не требует
    переобучения, только пересэмплирования.

    Диагностика: если избыток мощности на высоких ell накапливается вдоль
    цепочки обратного процесса (каждый шаг понемногу усиливает то, что
    сеть уже видит), TF должна заметно падать с уменьшением S -- цепочка
    короче, накопиться нечему. Если же избыток возникает независимо от
    траектории (сеть один раз "досочиняет" недостающую мощность), TF
    должна быть почти не чувствительна к S. Первый случай означает, что
    сэмплирование с меньшим S -- бесплатное немедленное смягчение, пока
    готовится исправление обучения (§7.2); второй случай означает, что S
    тут ни при чём и снижать его бессмысленно.
    """
    Ss = sorted(results_by_S)
    bands = bands or results_by_S[Ss[0]]["bands"]

    for label, key in (("Transfer Function", "tf_bands"), ("r_ell", "r_bands")):
        head = f"{'band':>14}" + "".join(f"{f'S={s}':>10}" for s in Ss)
        print(f"\n{label} vs число шагов сэмплирования S")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for i, (lo, hi) in enumerate(bands):
            row = "".join(f"{results_by_S[s][key][i]:>10.3f}" for s in Ss)
            print(f"{lo:.3f}-{hi:<6.2f}{row}")

    print("\n  Плоско по S -> избыток не зависит от длины траектории (сеть")
    print("  синтезирует мощность за один шаг, вне зависимости от S).")
    print("  Резко падает с уменьшением S -> избыток НАКАПЛИВАЕТСЯ вдоль")
    print("  цепочки -- тогда меньший S снижает его уже сейчас, без")
    print("  переобучения (но и без устранения первопричины).")


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


@torch.no_grad()
def diagnose_trajectory_spectrum(
    bbdm, dataset, t_probe=(999, 750, 500, 250, 100, 50, 10, 1),
    n_patches=16, batch_size=4, device="cuda", bands=DEFAULT_BANDS,
    indices=None, seed=0, S=200,
):
    """Мощность одношагового предсказания на РЕАЛЬНОЙ траектории сэмплера.

    `diagnose_prediction_spectrum` кормит сеть состояниями `q_sample(x0, y, t)`
    -- то есть распределением ОБУЧЕНИЯ. На инференсе сеть вместо этого видит
    то, что накопила цепочка к этому шагу, и это состояние может уже нести
    избыток, добавленный сетью же на предыдущих шагах. Эта функция
    прогоняет НАСТОЯЩИЙ обратный процесс (тот же код, что в `BBDM.sample()`)
    один раз на батч и записывает одношаговую мощность предсказания в
    шагах, ближайших к каждому запрошенному `t_probe`, -- чтобы сравнить
    напрямую с in-distribution диагностикой при (почти) тех же t.

    Цикл ниже ДУБЛИРУЕТ `BBDM.sample()`, а не вызывает его, потому что ему
    нужно читать пары (t, pred) в середине цепочки, которые `sample()` не
    отдаёт наружу. Это создаёт риск разъехаться с продакшн-кодом при
    следующей правке `sample()` -- поэтому
    `tests/test_bbdm_math.py::test_trajectory_diagnostic_matches_sampler_output`
    требует, чтобы при одинаковом сиде эта функция и `bbdm.sample()` давали
    ПОБИТОВО одинаковый финальный `x_t`; тест обязан падать при любом
    расхождении в копии цикла.

    Returns:
        dict с requested_t, actual_t (реально посещённый ближайший шаг),
        ratio_bands (мощность выхода / мощность цели, t x band) и
        input_ratio_bands (мощность ВХОДНОГО состояния / мощность цели,
        t x band) -- сравнение двух рядов показывает, усиливает ли сеть
        уже присутствующий во входе сигнал или синтезирует независимо от
        него.
    """
    from tqdm.auto import tqdm

    if indices is None:
        indices = list(range(min(n_patches, len(dataset))))

    bbdm = bbdm.to(device).eval()
    ps_target = _mean_target_rapsd(bbdm, dataset, indices, batch_size, device)
    freqs = np.arange(len(ps_target)) / len(ps_target)
    ps_target_np = ps_target.cpu().numpy()

    # Шаги траектории зависят только от T и S, не от входа -- считаем раз.
    steps = torch.linspace(bbdm.T, 1, S, device=device).round().long()
    steps = torch.unique(steps).flip(0)
    steps_list = steps.tolist()

    probe_idx = {}
    for want_t in t_probe:
        probe_idx[want_t] = min(
            range(len(steps_list)), key=lambda i: abs(steps_list[i] - want_t)
        )

    sums_out = {w: None for w in t_probe}
    sums_in = {w: None for w in t_probe}
    n = 0

    batches = _iter_batches(dataset, indices, batch_size)
    n_batches = (len(indices) + batch_size - 1) // batch_size
    batches = tqdm(batches, total=n_batches, desc="trajectory spectrum")

    for b_i, (x0, _y) in enumerate(batches):
        x0 = x0.to(device)
        B = x0.shape[0]
        g = torch.Generator(device=device)
        g.manual_seed(seed + b_i)

        y_cond = x0
        if bbdm.eta > 0:
            x_t = x0 + math.sqrt(bbdm.eta) * bbdm._randn_like(x0, g)
        else:
            x_t = x0.clone()

        for i, t_val in enumerate(steps):
            t_prev_val = steps[i + 1] if i + 1 < len(steps) else steps.new_zeros(())
            t = t_val.expand(B)
            t_prev = t_prev_val.expand(B)

            pred = bbdm.model(x_t, t)

            for want_t, probe_i in probe_idx.items():
                if probe_i == i:
                    ps_out = bbdm.rapsd(pred)
                    ps_in = bbdm.rapsd(x_t)
                    sums_out[want_t] = ps_out if sums_out[want_t] is None else sums_out[want_t] + ps_out
                    sums_in[want_t] = ps_in if sums_in[want_t] is None else sums_in[want_t] + ps_in

            c_x, c_y, c_e, d_tilde = bbdm._posterior_coeffs(t, t_prev)
            c_x = c_x.view(-1, 1, 1, 1)
            c_y = c_y.view(-1, 1, 1, 1)
            c_e = c_e.view(-1, 1, 1, 1)
            d_tilde = d_tilde.view(-1, 1, 1, 1)

            mean = (c_x - c_e) * x_t + c_e * pred + c_y * y_cond
            if bool((d_tilde > 0).any()):
                x_t = mean + d_tilde.clamp(min=0).sqrt() * bbdm._randn_like(x_t, g)
            else:
                x_t = mean
        n += 1
        # Финальное состояние ПОСЛЕДНЕГО батча -- только для сверки этого
        # цикла с продакшн-кодом BBDM.sample() (см. тест на побитовое
        # совпадение); ничего не агрегирует между батчами.
        last_batch_final_x_t = x_t.detach().cpu()

    ratio_bands, input_ratio_bands, actual_t = [], [], []
    for want_t in t_probe:
        ps_out = (sums_out[want_t] / max(n, 1)).cpu().numpy()
        ps_in = (sums_in[want_t] / max(n, 1)).cpu().numpy()
        ratio_bands.append(band_average(ps_out / (ps_target_np + 1e-20), freqs, bands))
        input_ratio_bands.append(band_average(ps_in / (ps_target_np + 1e-20), freqs, bands))
        actual_t.append(steps_list[probe_idx[want_t]])

    return {
        "requested_t": list(t_probe),
        "actual_t": actual_t,
        "bands": list(bands),
        "ratio_bands": ratio_bands,
        "input_ratio_bands": input_ratio_bands,
        "last_batch_final_x_t": last_batch_final_x_t,
    }


def print_trajectory_spectrum(result):
    """Таблица: мощность выхода / мощность цели на РЕАЛЬНОЙ траектории."""
    bands = result["bands"]
    head = f"{'t (real)':>10}" + "".join(f"{f'{lo}-{hi}':>14}" for lo, hi in bands)

    print("\nМощность выхода сети / мощность цели, на РЕАЛЬНОЙ траектории")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for i, act_t in enumerate(result["actual_t"]):
        row = "".join(f"{v:>14.2f}" for v in result["ratio_bands"][i])
        print(f"{act_t:>10}{row}")

    print("\nМощность ВХОДНОГО состояния x_t / мощность цели (для сравнения)")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for i, act_t in enumerate(result["actual_t"]):
        row = "".join(f"{v:>14.2f}" for v in result["input_ratio_bands"][i])
        print(f"{act_t:>10}{row}")

    print("\n  Если 'выход' близок к 'входу' на каждом t -- сеть в основном")
    print("  ПРОПУСКАЕТ то, что уже накопилось в x_t, а не порождает заново.")


def print_trajectory_vs_qsample(traj, qsample):
    """Одношаговое отношение мощности: q_sample (in-distribution) vs траектория.

    Требует, чтобы `traj` и `qsample` были посчитаны с ОДИНАКОВЫМ по
    порядку списком t (`diagnose_trajectory_spectrum(t_probe=...)` и
    `diagnose_prediction_spectrum(t_values=...)` с тем же кортежем) --
    строки сопоставляются по индексу, не по значению.

    Большой разрыв между колонками -- признак exposure bias: сеть ведёт
    себя иначе на состояниях, которые реально встречает при генерации, чем
    на состояниях из обучающего распределения при том же t.
    """
    if traj["requested_t"] != qsample["t_values"]:
        raise ValueError(
            "traj['requested_t'] и qsample['t_values'] не совпадают -- "
            "запустите обе функции с одним и тем же кортежем t, иначе "
            "строки нельзя сопоставить по индексу"
        )
    if traj["bands"] != qsample["bands"]:
        raise ValueError("bands должны совпадать в обеих диагностиках")

    bands = traj["bands"]
    print("\nМощность выхода / мощность цели: q_sample vs РЕАЛЬНАЯ траектория\n")
    for bi, (lo, hi) in enumerate(bands):
        print(f"band {lo}-{hi}:")
        print(f"  {'t':>10}{'q_sample':>12}{'trajectory':>14}{'(факт. t)':>12}")
        for i, want_t in enumerate(traj["requested_t"]):
            q = qsample["ratio_bands"][i][bi]
            tr = traj["ratio_bands"][i][bi]
            print(f"  {want_t:>10}{q:>12.2f}{tr:>14.2f}{traj['actual_t'][i]:>12}")
        print()


@torch.no_grad()
def compute_2d_power_spectrum(
    bbdm, dataset, n_patches=16, batch_size=4, device="cuda",
    indices=None, S=200, seed=0,
):
    """Средний ПОЛНЫЙ (не радиально усреднённый) 2D-спектр предсказания и цели.

    Радиальное усреднение в `evaluate_spectra` -- ровно то, что прячет
    НАПРАВЛЕННЫЙ артефакт вроде шахматки от `ConvTranspose2d(stride=2)` в
    `UpBlock`: два артефакта одного радиуса, но разного угла, усредняются
    в одно число. Здесь сохраняется полная карта (H, W), так что
    анизотропный избыток -- мощность, сконцентрированная вдоль осей
    изображения или в углах Найквиста, а не равномерно по кольцу, -- виден
    напрямую.

    Returns:
        dict с pred, target (обе (H, W), в нормализованных единицах,
        усреднены по патчам и батчам с весом по размеру батча) и ratio.
    """
    if indices is None:
        indices = list(range(min(n_patches, len(dataset))))

    bbdm = bbdm.to(device).eval()

    total_pred = total_target = None
    total_n = 0

    for b_i, (x0, y) in enumerate(_iter_batches(dataset, indices, batch_size)):
        x0, y = x0.to(device), y.to(device)
        g = torch.Generator(device=device)
        g.manual_seed(seed + b_i)

        pred = bbdm.sample(x0, S=S, generator=g)

        h, w = pred.shape[-2:]
        win = bbdm._hann_window(h, w, pred.device, pred.dtype)

        f_pred = torch.fft.fftshift(torch.fft.fft2(pred * win, norm="ortho"), dim=(-2, -1))
        f_true = torch.fft.fftshift(torch.fft.fft2(y * win, norm="ortho"), dim=(-2, -1))

        # Сумма (не среднее) по батчу и каналу -- усредняем по общему числу
        # патчей в конце, взвешенно, чтобы неполный последний батч не сместил
        # результат.
        p_pred_sum = (f_pred.real ** 2 + f_pred.imag ** 2).sum(dim=(0, 1))
        p_true_sum = (f_true.real ** 2 + f_true.imag ** 2).sum(dim=(0, 1))

        total_pred = p_pred_sum if total_pred is None else total_pred + p_pred_sum
        total_target = p_true_sum if total_target is None else total_target + p_true_sum
        total_n += x0.shape[0]

    mean_pred = (total_pred / max(total_n, 1)).cpu().numpy()
    mean_target = (total_target / max(total_n, 1)).cpu().numpy()

    return {
        "pred": mean_pred,
        "target": mean_target,
        "ratio": mean_pred / (mean_target + 1e-20),
        "n_patches": total_n,
    }


def plot_2d_power_spectrum(result, title=None):
    """Изображения (лог-шкала) 2D-спектра цели, предсказания и их отношения.

    Смотреть на мощность, НЕ равномерную по углу на фиксированном
    радиусе, -- яркую вдоль осей или сконцентрированную в четырёх углах
    Найквиста. Радиально усреднённый RAPSD такую картину усредняет и
    прячет. Это подпись артефакта апсемплинга со страйдом (шахматка), а
    не реальной структуры неба -- у неё нет выделенного направления.
    """
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    positive = result["target"][result["target"] > 0]
    vmin = float(np.percentile(positive, 1)) if positive.size else 1e-10
    vmax = float(np.percentile(result["target"], 99.9))

    for ax, key, label in [(axes[0], "target", "target"), (axes[1], "pred", "prediction")]:
        im = ax.imshow(result[key], norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap="inferno")
        ax.set_title(f"2D power spectrum: {label}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.8)

    im = axes[2].imshow(result["ratio"], norm=mcolors.LogNorm(vmin=0.1, vmax=10), cmap="RdBu_r")
    axes[2].set_title("ratio: pred / target")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], shrink=0.8)

    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f"n={result['n_patches']} patches")
    plt.tight_layout()
    plt.show()
    return fig


def axis_vs_diagonal_power(power_2d, bands, axis_half_width_deg=10.0):
    """Средняя мощность вдоль осей vs вдоль диагоналей, по частотным полосам.

    Отношение, далёкое от 1 на высоком радиусе, присутствующее у
    предсказания, но не у цели, -- количественная подпись направленного
    (шахматка / страйдовый апсемплинг) артефакта, а не изотропной мощности
    неба или шума.
    """
    h, w = power_2d.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    theta = np.degrees(np.arctan2(yy - cy, xx - cx)) % 180  # свернуть в [0, 180)

    def _near(angle, width):
        d = np.minimum(np.abs(theta - angle), 180 - np.abs(theta - angle))
        return d <= width

    axis_mask = _near(0, axis_half_width_deg) | _near(90, axis_half_width_deg)
    diag_mask = _near(45, axis_half_width_deg) | _near(135, axis_half_width_deg)

    r_max = min(cx, cy)
    out = []
    for lo, hi in bands:
        ring = (r >= lo * r_max) & (r < hi * r_max)
        a_sel, d_sel = ring & axis_mask, ring & diag_mask
        axis_power = float(power_2d[a_sel].mean()) if a_sel.any() else float("nan")
        diag_power = float(power_2d[d_sel].mean()) if d_sel.any() else float("nan")
        out.append(axis_power / diag_power if diag_power and diag_power > 0 else float("nan"))
    return out


def print_anisotropy_table(result, bands=DEFAULT_BANDS):
    """Таблица: отношение мощности вдоль осей к мощности по диагоналям."""
    axis_t = axis_vs_diagonal_power(result["target"], bands)
    axis_p = axis_vs_diagonal_power(result["pred"], bands)

    print(f"\n{'band':>14}{'axis/diag target':>19}{'axis/diag pred':>17}")
    print("  " + "-" * 48)
    for i, (lo, hi) in enumerate(bands):
        print(f"{lo:.3f}-{hi:<6.2f}{axis_t[i]:>19.3f}{axis_p[i]:>17.3f}")

    print("\n  ~1.0 -- изотропно (нет выделенного направления), как и должно")
    print("  быть у реальной структуры неба. Значение, далёкое от 1 у")
    print("  'pred', но не у 'target', и усиливающееся к высоким частотам,")
    print("  указывает на направленный артефакт сети (например, страйдовый")
    print("  апсемплинг / шахматку), а не на физическую анизотропию.")
