"""Brownian Bridge Diffusion Model для пары (Planck -> ACT+Planck).

Соглашение о направлении моста (НЕ инвертировать):

    t = 0  ->  y   -- ACT+Planck, то, что генерируем
    t = T  ->  x0  -- Planck, условный вход

То есть `x0` в коде означает "патч Planck", а не "состояние модели в t=0".

Расписание (s = S_VAR из конфига):

    m_t     = t / T
    delta_t = 2 * s * (m_t - m_t**2)

Прямой процесс:

    x_t = (1 - m_t) * y + m_t * x0 + sqrt(delta_t) * eps

Сеть обучается предсказывать чистый y (x0-параметризация), поэтому шаг
обратного процесса собирается как

    x_{t_prev} = (c_xt - c_et) * x_t + c_et * pred + c_yt * y_cond
                 + sqrt(delta_tilde) * z

Вычитание `c_et` обязательно: коэффициенты (c_xt, c_yt, c_et) взяты из
eps-параметризации (Algorithm 2 статьи), где c_xt уже алгебраически
содержит слагаемое +c_et. Добавление `+ c_et * pred` без этого вычитания
на каждом из ~200 шагов дублирует сигнал и раздувает амплитуду (проверено
oracle-тестом: x6.58 по амплитуде, ~x43 по мощности).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BBDM(nn.Module):
    """Обёртка над U-Net: прямой процесс, лосс и сэмплирование.

    Args:
        model: сеть, предсказывающая чистый y по (x_t, t).
        T: число шагов диффузии.
        s: масштаб дисперсии моста.
        eta: дисперсия шума в стартовом состоянии обратного процесса,
            x_T = planck + sqrt(eta) * randn. eta=0 -- как в статье.
        spectral_weight: вес спектрального члена в лоссе по умолчанию.
    """

    def __init__(self, model, T=1000, s=0.5, eta=0.0, spectral_weight=0.0):
        super().__init__()
        self.model = model
        self.T = int(T)
        self.s = float(s)
        self.eta = float(eta)
        self.spectral_weight = float(spectral_weight)
        # Кэш радиальных бинов и окон Ханна под конкретный (H, W, device).
        # Не буферы: не должны попадать в state_dict и переезжать с .to().
        self._bin_cache = {}
        self._window_cache = {}
        self._precompute_schedules()

    # ------------------------------------------------------------ расписание

    def _precompute_schedules(self):
        """m_t и delta_t для t = 1..T (используются в q_sample / loss)."""
        t_vals = torch.arange(1, self.T + 1, dtype=torch.float64)
        m_t = t_vals / self.T
        delta_t = 2 * self.s * (m_t - m_t ** 2)
        self.register_buffer("m_t", m_t.float())
        self.register_buffer("delta_t", delta_t.float())

    def _posterior_coeffs(self, t, t_prev):
        """Коэффициенты обратного шага для произвольного прыжка t -> t_prev.

        Поддерживает subsampled-инференс (S < T). При t = T величина
        m_t = 1 обнуляет delta_t и несколько коэффициентов вырождаются в
        0/0; clamp(max=1-1e-4) заменяет это корректным пределом
        x_{T-1} = m_{T-1} * planck + (1 - m_{T-1}) * pred + шум.
        """
        s = self.s
        m_t = (t.float() / self.T).clamp(max=1.0 - 1e-4)
        m_t1 = (t_prev.float() / self.T).clamp(max=1.0 - 1e-4)

        delta_t = (2 * s * (m_t - m_t ** 2)).clamp(min=1e-12)
        delta_t1 = 2 * s * (m_t1 - m_t1 ** 2)

        one_minus_m_t1 = (1 - m_t1).clamp(min=1e-4)
        shrink = (1 - m_t) / one_minus_m_t1

        delta_t_given_t1 = (delta_t - delta_t1 * shrink ** 2).clamp(min=0)
        delta_tilde = delta_t_given_t1 * delta_t1 / delta_t

        c_xt = (delta_t1 / delta_t) * shrink + (delta_t_given_t1 / delta_t) * one_minus_m_t1
        c_yt = m_t1 - m_t * shrink * (delta_t1 / delta_t)
        c_et = one_minus_m_t1 * (delta_t_given_t1 / delta_t)

        return c_xt, c_yt, c_et, delta_tilde

    # ------------------------------------------------------- прямой процесс

    @staticmethod
    def _randn_like(x, generator=None):
        """randn_like с опциональным генератором (для воспроизводимой валидации).

        Генератор обязан быть создан на том же устройстве, что и `x`.
        """
        if generator is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)

    def q_sample(self, x0, y, t, generator=None):
        """x_t = (1 - m_t) * y + m_t * x0 + sqrt(delta_t) * eps.

        В t = T даёт ровно x0 (Planck), в t -> 0 -- ровно y (ACT+Planck).
        """
        idx = t - 1
        m = self.m_t[idx].view(-1, 1, 1, 1)
        d = self.delta_t[idx].view(-1, 1, 1, 1)
        eps = self._randn_like(x0, generator)
        x_t = (1 - m) * y + m * x0 + d.sqrt() * eps
        return x_t, eps

    # ------------------------------------------------- спектральные утилиты

    def _hann_window(self, h, w, device, dtype):
        """Окно Ханна, идентичное np.outer(np.hanning(h), np.hanning(w))."""
        key = (h, w, device.type, device.index, dtype)
        win = self._window_cache.get(key)
        if win is None:
            wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
            wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
            win = torch.outer(wy, wx)
            self._window_cache[key] = win
        return win

    def _radial_bins(self, h, w, device):
        """Радиальные бины, совпадающие с metrics.power_spectrum._radial_bins."""
        key = (h, w, device.type, device.index)
        cached = self._bin_cache.get(key)
        if cached is None:
            cy, cx = h // 2, w // 2
            yy = torch.arange(h, device=device, dtype=torch.float32) - cy
            xx = torch.arange(w, device=device, dtype=torch.float32) - cx
            # .long() усекает так же, как .astype(int) в numpy-версии
            r = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2).long().reshape(-1)
            r_max = min(cx, cy)
            keep = r < r_max
            idx = r[keep]
            counts = torch.bincount(idx, minlength=r_max).clamp(min=1).float()
            cached = (keep, idx, counts, r_max)
            self._bin_cache[key] = cached
        return cached

    def rapsd(self, img):
        """Радиально усреднённый спектр мощности, усреднённый по батчу и каналам.

        Args:
            img: тензор (B, C, H, W).

        Returns:
            Тензор (r_max,). Окно и биннинг идентичны
            ``metrics.power_spectrum.compute_power_spectrum``, чтобы лосс
            штрафовал ровно ту величину, которая потом измеряется.
        """
        h, w = img.shape[-2:]
        win = self._hann_window(h, w, img.device, img.dtype)
        f = torch.fft.fftshift(torch.fft.fft2(img * win, norm="ortho"), dim=(-2, -1))
        # real^2 + imag^2, а не abs()**2: у abs() комплексного тензора
        # градиент в нуле -- NaN.
        power = (f.real ** 2 + f.imag ** 2).mean(dim=(0, 1)).reshape(-1)

        keep, idx, counts, r_max = self._radial_bins(h, w, img.device)
        sums = torch.zeros(r_max, device=img.device, dtype=power.dtype)
        sums = sums.index_add(0, idx, power[keep])
        return sums / counts

    def spectral_loss(self, pred, y, eps=1e-8):
        """L1 по логарифму RAPSD ансамбля: прямой штраф за TF != 1.

        Логарифм -- потому что мощность падает на порядки от низких ell к
        высоким, и невзвешенный спектральный лосс определялся бы только
        самыми низкими бинами. L1, а не L2 -- остатки log-мощности
        тяжелохвостые. Спектры усредняются по батчу до сравнения: цель --
        совпадение ансамблевой мощности (это и есть TF ~ 1), а не спектра
        каждой отдельной реализации, который сам по себе хи-квадрат-шумный.
        Бин 0 (DC-мода) отбрасывается -- средний уровень к спектральной
        точности отношения не имеет и покрыт MSE.
        """
        ps_pred = self.rapsd(pred)
        ps_true = self.rapsd(y)
        return F.l1_loss(
            torch.log(ps_pred[1:] + eps), torch.log(ps_true[1:] + eps)
        )

    # ------------------------------------------------------------------ лосс

    def loss(self, x0, y, spectral_weight=None, generator=None, return_terms=False):
        """MSE по чистому y + спектральный член.

        Args:
            x0: батч патчей Planck (B, 1, H, W), нормализованных.
            y: батч патчей ACT+Planck (B, 1, H, W), нормализованных.
            spectral_weight: перекрывает self.spectral_weight, если задан.
            generator: генератор для t и шума (воспроизводимая валидация).
            return_terms: вернуть также словарь со слагаемыми для логов.
        """
        w = self.spectral_weight if spectral_weight is None else float(spectral_weight)

        B = x0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=x0.device, generator=generator)
        x_t, _ = self.q_sample(x0, y, t, generator=generator)
        pred = self.model(x_t, t)

        mse = F.mse_loss(pred, y)
        if w == 0.0:
            total = mse
            spec = torch.zeros((), device=x0.device, dtype=mse.dtype)
        else:
            spec = self.spectral_loss(pred, y)
            total = mse + w * spec

        if return_terms:
            return total, {"mse": mse.detach(), "spec": spec.detach()}
        return total

    # -------------------------------------------------------- обратный процесс

    @torch.no_grad()
    def sample(self, planck, S=200, generator=None, progress=False):
        """Генерация ACT+Planck-патча из Planck-патча.

        Args:
            planck: (B, 1, H, W), нормализованный вход. Целевая карта сюда
                не передаётся и передаваться не должна.
            S: число шагов обратного процесса (S <= T).
            generator: генератор шума (для воспроизводимых сэмплов).
            progress: показывать tqdm-прогрессбар.
        """
        y_cond = planck  # чистый вход, условие на КАЖДОМ шаге
        if self.eta > 0:
            x_t = planck + math.sqrt(self.eta) * self._randn_like(planck, generator)
        else:
            x_t = planck.clone()

        steps = torch.linspace(self.T, 1, S, device=planck.device).round().long()
        # При S, близком к T, округление даёт повторы; такой шаг тождественный
        # (delta_{t|t_prev} = 0) и только тратит forward сети.
        steps = torch.unique(steps).flip(0)

        iterator = range(len(steps))
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="sampling", leave=False)

        B = planck.shape[0]
        for i in iterator:
            t_val = steps[i]
            t_prev_val = steps[i + 1] if i + 1 < len(steps) else steps.new_zeros(())

            t = t_val.expand(B)
            t_prev = t_prev_val.expand(B)

            pred = self.model(x_t, t)

            c_x, c_y, c_e, d_tilde = self._posterior_coeffs(t, t_prev)
            c_x = c_x.view(-1, 1, 1, 1)
            c_y = c_y.view(-1, 1, 1, 1)
            c_e = c_e.view(-1, 1, 1, 1)
            d_tilde = d_tilde.view(-1, 1, 1, 1)

            mean = (c_x - c_e) * x_t + c_e * pred + c_y * y_cond
            if bool((d_tilde > 0).any()):
                x_t = mean + d_tilde.clamp(min=0).sqrt() * self._randn_like(x_t, generator)
            else:
                x_t = mean

        return x_t
