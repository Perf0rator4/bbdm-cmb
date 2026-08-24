"""Проверки математики моста, сэмплера и спектральных метрик.

Запуск: `pytest tests/` или `python tests/test_bbdm_math.py`.
Данные не нужны -- всё на синтетике, работает на CPU за секунды.

Ключевой тест здесь -- oracle-тест сэмплера (test_sampler_oracle_*): в
BBDM.sample() подаётся фейковая "модель", возвращающая точную цель, и
проверяется, что амплитуда на выходе совпадает с целью ровно, а траектория
не разгоняется. Именно этот тест ловит смешение eps- и x0-параметризации,
которое раньше раздувало Transfer Function в десятки раз.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdm.metrics.power_spectrum import (  # noqa: E402
    compute_power_spectrum,
    cross_correlation,
    transfer_function,
)
from bbdm.model.bbdm import BBDM  # noqa: E402
from bbdm.model.unet import UNet  # noqa: E402

T_STEPS = 1000
S_VAR = 0.5


class OracleModel(nn.Module):
    """Идеальная модель: всегда возвращает точную цель.

    Попутно записывает std входного состояния на каждом шаге -- так видно,
    разгоняется траектория или держится на амплитуде моста.
    """

    def __init__(self, target):
        super().__init__()
        self.register_buffer("target", target)
        self.seen_std = []

    def forward(self, x, t):
        self.seen_std.append(float(x.std()))
        return self.target.expand_as(x)


def _make(target, eta=0.0):
    return BBDM(OracleModel(target), T=T_STEPS, s=S_VAR, eta=eta)


def _random_pair(b=2, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(b, 1, h, w, generator=g)          # ACT+Planck, цель
    x0 = torch.randn(b, 1, h, w, generator=g) * 0.8   # Planck, вход
    return x0, y


# --------------------------------------------------------------- q_sample --


def test_q_sample_endpoints():
    """t=T даёт ровно Planck, t=1 -- почти ровно ACT+Planck."""
    x0, y = _random_pair()
    bbdm = _make(y)

    t_T = torch.full((x0.shape[0],), T_STEPS, dtype=torch.long)
    x_T, _ = bbdm.q_sample(x0, y, t_T)
    assert torch.allclose(x_T, x0, atol=1e-6), "x_T должен быть ровно Planck"

    t_1 = torch.ones(x0.shape[0], dtype=torch.long)
    x_1, _ = bbdm.q_sample(x0, y, t_1)
    # m_1 = 1e-3, delta_1 = 1e-3 -> отклонение от y порядка 0.03 по std
    assert (x_1 - y).std() < 0.1, "x_1 должен быть близок к ACT+Planck"


def test_q_sample_moments_monte_carlo():
    """Эмпирические среднее и дисперсия x_t совпадают с формулой моста."""
    x0, y = _random_pair(b=1, h=8, w=8, seed=1)
    bbdm = _make(y)

    for t_val in (250, 500, 750):
        t = torch.full((1,), t_val, dtype=torch.long)
        draws = torch.stack([bbdm.q_sample(x0, y, t)[0] for _ in range(4000)])

        m = t_val / T_STEPS
        d = 2 * S_VAR * (m - m ** 2)
        expected_mean = (1 - m) * y + m * x0

        assert (draws.mean(0) - expected_mean).abs().max() < 0.05, f"mean @ t={t_val}"
        assert abs(float(draws.var(0).mean()) - d) < 0.02, f"var @ t={t_val}"


# --------------------------------------------------- коэффициенты обратного


def test_posterior_final_step_returns_prediction():
    """На последнем шаге (t_prev=0) обновление обязано вернуть ровно pred."""
    bbdm = _make(torch.zeros(1, 1, 4, 4))
    t = torch.tensor([1])
    t_prev = torch.tensor([0])
    c_x, c_y, c_e, d_tilde = bbdm._posterior_coeffs(t, t_prev)

    assert abs(float(c_x - c_e)) < 1e-5, "вклад x_t должен занулиться"
    assert abs(float(c_e) - 1.0) < 1e-5, "pred должен войти с весом 1"
    assert abs(float(c_y)) < 1e-5, "условие не должно добавляться поверх"
    assert float(d_tilde) < 1e-8, "шум на последнем шаге не добавляется"


def test_posterior_first_step_limit():
    """При t=T вырождение 0/0 должно давать корректный предел."""
    bbdm = _make(torch.zeros(1, 1, 4, 4))
    t = torch.tensor([T_STEPS])
    t_prev = torch.tensor([T_STEPS - 5])
    c_x, c_y, c_e, d_tilde = bbdm._posterior_coeffs(t, t_prev)

    m_prev = (T_STEPS - 5) / T_STEPS
    # предел: x_{T-1} = m_{T-1} * planck + (1 - m_{T-1}) * pred + шум
    assert abs(float(c_x - c_e) - m_prev) < 1e-3
    assert abs(float(c_e) - (1 - m_prev)) < 1e-3
    assert abs(float(c_y)) < 1e-3
    delta_prev = 2 * S_VAR * (m_prev - m_prev ** 2)
    assert abs(float(d_tilde) - delta_prev) < 1e-3


def test_posterior_first_step_limit_is_clamp_independent():
    """Значение clamp'а не должно влиять на предел (иначе это не предел)."""
    bbdm = _make(torch.zeros(1, 1, 4, 4))
    t_prev = torch.tensor([T_STEPS - 5])

    ref = [float(v) for v in bbdm._posterior_coeffs(torch.tensor([T_STEPS]), t_prev)]
    # t = T - 1 даёт m_t = 0.999 -- ниже clamp'а 1-1e-4, вырождения нет
    near = [float(v) for v in bbdm._posterior_coeffs(torch.tensor([T_STEPS - 1]), t_prev)]

    for a, b in zip(ref, near):
        assert abs(a - b) < 5e-3


# -------------------------------------------------------- oracle-тест сэмплера


def test_sampler_oracle_reproduces_target():
    """С идеальной моделью сэмплер обязан вернуть цель ровно (x1.0)."""
    x0, y = _random_pair(b=1, h=32, w=32, seed=2)
    bbdm = _make(y, eta=0.0).eval()

    out = bbdm.sample(x0, S=50)

    ratio = float(out.std() / y.std())
    assert abs(ratio - 1.0) < 1e-3, f"амплитуда x{ratio:.3f}, ожидалась x1.000"
    assert float((out - y).abs().max()) < 1e-4


def test_sampler_oracle_trajectory_does_not_blow_up():
    """Промежуточные состояния держатся на амплитуде моста, а не растут."""
    x0, y = _random_pair(b=1, h=32, w=32, seed=3)
    bbdm = _make(y, eta=0.0).eval()
    bbdm.sample(x0, S=50)

    seen = bbdm.model.seen_std
    assert len(seen) == 50
    # На мосту std(x_t) ~ 1 плюс не больше sqrt(max delta_t) = sqrt(0.25).
    assert max(seen) < 3.0, f"траектория разгоняется: max std = {max(seen):.2f}"


def test_buggy_update_inflates_amplitude():
    """Регрессия на Bug B: без вычитания c_et амплитуда раздувается.

    Тест намеренно воспроизводит СТАРУЮ формулу (c_x * x_t вместо
    (c_x - c_e) * x_t) и требует, чтобы она давала заметную инфляцию --
    если этот тест начнёт падать, значит проверка перестала быть
    осмысленной, а не что баг исчез.
    """
    x0, y = _random_pair(b=1, h=32, w=32, seed=4)
    bbdm = _make(y, eta=0.0).eval()

    steps = torch.linspace(T_STEPS, 1, 50).round().long()
    x_t = x0.clone()
    with torch.no_grad():
        for i, t_val in enumerate(steps):
            t_prev_val = steps[i + 1] if i + 1 < len(steps) else steps.new_zeros(())
            t = t_val.expand(1)
            t_prev = t_prev_val.expand(1)
            c_x, c_y, c_e, _ = bbdm._posterior_coeffs(t, t_prev)
            # старая (ошибочная) формула: c_x без вычитания c_e
            x_t = c_x.view(-1, 1, 1, 1) * x_t + c_e.view(-1, 1, 1, 1) * y \
                + c_y.view(-1, 1, 1, 1) * x0

    inflation = float(x_t.std() / y.std())
    assert inflation > 3.0, f"старая формула должна раздувать, получили x{inflation:.2f}"


def test_sampler_ignores_target_beyond_the_input():
    """sample() принимает только Planck: сигнатура не должна пускать цель."""
    import inspect

    params = set(inspect.signature(BBDM.sample).parameters)
    for forbidden in ("y", "y_true", "target", "act"):
        assert forbidden not in params


# ------------------------------------------------------------ спектральное


def test_torch_rapsd_matches_numpy():
    """RAPSD в лоссе и RAPSD в метриках должны совпадать до нормировки FFT."""
    g = torch.Generator().manual_seed(5)
    img = torch.randn(1, 1, 64, 64, generator=g)
    bbdm = _make(torch.zeros(1, 1, 64, 64))

    torch_ps = bbdm.rapsd(img).numpy()
    numpy_ps, _ = compute_power_spectrum(img[0, 0].numpy())

    # torch-версия считает с norm="ortho" (делит на sqrt(H*W)),
    # numpy-версия -- без нормировки: отношение ровно H*W.
    scaled = torch_ps * (64 * 64)
    assert np.allclose(scaled, numpy_ps, rtol=1e-4, atol=1e-6)


def test_spectral_loss_zero_on_identical_input():
    g = torch.Generator().manual_seed(6)
    img = torch.randn(2, 1, 32, 32, generator=g)
    bbdm = _make(torch.zeros(1, 1, 32, 32))
    assert float(bbdm.spectral_loss(img, img)) < 1e-6


def test_spectral_loss_detects_smoothing():
    """Заглаженное предсказание должно давать заметный спектральный штраф."""
    g = torch.Generator().manual_seed(7)
    img = torch.randn(4, 1, 64, 64, generator=g)
    blur = nn.AvgPool2d(3, stride=1, padding=1)
    smoothed = blur(img)

    bbdm = _make(torch.zeros(1, 1, 64, 64))
    assert float(bbdm.spectral_loss(smoothed, img)) > 0.5


def test_cross_correlation_identity():
    """r_ell(p, p) == 1 на всех бинах."""
    rng = np.random.default_rng(8)
    p = rng.standard_normal((64, 64))
    r_ell, _ = cross_correlation(p, p)
    assert np.allclose(r_ell, 1.0, atol=1e-8)


def test_transfer_function_identity():
    rng = np.random.default_rng(9)
    p = rng.standard_normal((64, 64))
    ps, _ = compute_power_spectrum(p)
    assert np.allclose(transfer_function(ps, ps), 1.0, atol=1e-8)


def test_cross_correlation_of_independent_maps_is_small():
    rng = np.random.default_rng(10)
    a = rng.standard_normal((128, 128))
    b = rng.standard_normal((128, 128))
    r_ell, freqs = cross_correlation(a, b)
    # на высоких ell в бине много мод, оценка должна быть близка к нулю
    assert abs(float(np.mean(r_ell[freqs > 0.3]))) < 0.1


# ---------------------------------------------------------------- сквозное


def test_unet_forward_shape_and_time_dim():
    net = UNet(in_ch=1, base_ch=8, time_dim=64, groups=4)
    x = torch.randn(2, 1, 64, 64)
    t = torch.tensor([3, 900])
    out = net(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_loss_backward_with_spectral_term():
    """Спектральный член обязан давать конечные градиенты."""
    net = UNet(in_ch=1, base_ch=8, time_dim=64, groups=4)
    bbdm = BBDM(net, T=T_STEPS, s=S_VAR, eta=0.0, spectral_weight=0.1)

    x0, y = _random_pair(b=2, h=64, w=64, seed=11)
    total, terms = bbdm.loss(x0, y, return_terms=True)
    total.backward()

    assert torch.isfinite(total)
    assert float(terms["spec"]) != 0.0
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_loss_is_reproducible_with_generator():
    """Один сид -> один и тот же лосс (иначе val-кривая неинформативна)."""
    net = UNet(in_ch=1, base_ch=8, time_dim=64, groups=4).eval()
    bbdm = BBDM(net, T=T_STEPS, s=S_VAR, eta=0.0, spectral_weight=0.1)
    x0, y = _random_pair(b=2, h=64, w=64, seed=12)

    values = []
    for _ in range(2):
        g = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            values.append(float(bbdm.loss(x0, y, generator=g)))
    assert values[0] == values[1]


def test_sample_step_deduplication():
    """При S близком к T округление даёт повторы -- их надо убирать."""
    x0, y = _random_pair(b=1, h=16, w=16, seed=13)
    bbdm = BBDM(OracleModel(y), T=20, s=S_VAR, eta=0.0).eval()
    bbdm.sample(x0, S=100)
    assert len(bbdm.model.seen_std) == 20, "шагов не больше, чем T"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
