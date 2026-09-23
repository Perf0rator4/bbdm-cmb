"""Проверки изменений Фазы 1: канал условия, апсемплинг, монитор, загрузка
чекпоинтов, защита от перезаписи.

Запуск: `pytest tests/` или `python tests/test_phase1.py`. CPU, секунды.
"""

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdm.evaluate import evaluate_spectra  # noqa: E402
from bbdm.model.bbdm import BBDM  # noqa: E402
from bbdm.model.unet import UNet  # noqa: E402
from bbdm.train import load_checkpoint, train  # noqa: E402

T_STEPS = 1000
S_VAR = 0.5


class _PairDataset:
    def __init__(self, n, size=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = [
            (torch.randn(1, size, size, generator=g),
             torch.randn(1, size, size, generator=g))
            for _ in range(n)
        ]
        self.mu, self.sigma = 0.0, 1.0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _steps(S, T=T_STEPS):
    s = torch.linspace(T, 1, S).round().long()
    return torch.unique(s).flip(0).tolist()


# -------------------------------------------------- почему нужен канал условия


def test_posterior_coeffs_c_x_plus_c_y_is_one():
    """c_xt + c_yt == 1 на каждом шаге.

    Отсюда: сеть-тождество (pred = x_t) оставляет Planck-коэффициент
    состояния равным 1 на всей цепочке -- то есть переносит Planck на выход
    целиком. На этом держится интерпретация полосы 0.1–0.3 в run 4.
    """
    bbdm = BBDM(nn.Identity(), T=T_STEPS, s=S_VAR)
    steps = _steps(200)
    for i, t in enumerate(steps):
        tp = steps[i + 1] if i + 1 < len(steps) else 0
        c_x, c_y, _, _ = bbdm._posterior_coeffs(torch.tensor([t]), torch.tensor([tp]))
        assert abs(float(c_x + c_y) - 1.0) < 1e-5, f"t={t}"


def _planck_coeff_at_output(bbdm, pred_planck_coeff):
    """Planck-коэффициент выхода, если pred несёт Planck с коэффициентом f(t, a_t)."""
    a = 1.0
    steps = _steps(200)
    for i, t in enumerate(steps):
        tp = steps[i + 1] if i + 1 < len(steps) else 0
        c_x, c_y, c_e, _ = (float(v) for v in bbdm._posterior_coeffs(
            torch.tensor([t]), torch.tensor([tp])))
        a = (c_x - c_e) * a + c_e * pred_planck_coeff(t, a) + c_y
    return a


def test_planck_leaks_without_cond_and_vanishes_with_it():
    """Мода, где Planck -- чистый шум мощности N = 6.29 P (полоса 0.1–0.3 в run 4).

    Сеть-тождество: на выходе весь Planck (коэффициент 1).
    Точное E[y | x_t] (сеть видит только x_t): заметный остаток.
    Точное E[y | x_t, x0] (сеть видит Planck отдельно): ровно ноль.
    """
    bbdm = BBDM(nn.Identity(), T=T_STEPS, s=S_VAR)
    P, N = 1.0, 6.29

    def delta(m):
        return 2 * S_VAR * (m - m * m)

    identity = _planck_coeff_at_output(bbdm, lambda t, a: a)

    def no_cond(t, a):
        m = t / T_STEPS
        k = (1 - m) * P / ((1 - m) ** 2 * P + m * m * N + delta(m) + 1e-30)
        return k * a

    def with_cond(t, a):
        m = t / T_STEPS
        k = (1 - m) * P / ((1 - m) ** 2 * P + delta(m) + 1e-30)
        return k * (a - m)

    a_no = _planck_coeff_at_output(bbdm, no_cond)
    a_cond = _planck_coeff_at_output(bbdm, with_cond)

    assert abs(identity - 1.0) < 1e-4
    assert a_no > 0.2, f"без условия Planck должен протекать, получили {a_no:.3f}"
    assert abs(a_cond) < 1e-3, f"с условием Planck должен уходить, получили {a_cond:.4f}"


def test_identity_denoiser_returns_planck_through_the_real_sampler():
    """То же на настоящем BBDM.sample(): тождество отдаёт Planck с коэффициентом ~1."""

    class Identity(nn.Module):
        def forward(self, x, t):
            return x

    g = torch.Generator().manual_seed(0)
    planck = torch.randn(4, 1, 32, 32, generator=g)
    out = BBDM(Identity(), T=T_STEPS, s=S_VAR, eta=0.0).sample(planck, S=50)

    coeff = float((out * planck).sum() / (planck * planck).sum())
    assert abs(coeff - 1.0) < 0.05, f"коэффициент Planck на выходе {coeff:.3f}"


# ---------------------------------------------------------------- архитектура


def test_resize_conv_unet_has_no_transposed_convs():
    net = UNet(in_ch=1, base_ch=8, time_dim=64, groups=4, upsample="resize_conv")
    assert not any(isinstance(m, nn.ConvTranspose2d) for m in net.modules())

    old = UNet(in_ch=1, base_ch=8, time_dim=64, groups=4, upsample="transpose")
    assert any(isinstance(m, nn.ConvTranspose2d) for m in old.modules())


def test_cond_unet_shapes_and_requires_cond():
    net = UNet(in_ch=1, base_ch=8, time_dim=64, groups=4, cond_ch=1)
    x = torch.randn(2, 1, 32, 32)
    t = torch.tensor([5, 900])
    out = net(x, t, cond=torch.randn(2, 1, 32, 32))
    assert out.shape == x.shape

    try:
        net(x, t)
    except ValueError:
        pass
    else:
        raise AssertionError("cond_ch > 0 без cond обязан падать, а не молча работать")


def test_unknown_upsample_mode_is_rejected():
    try:
        UNet(in_ch=1, base_ch=8, time_dim=64, groups=4, upsample="bilinear_magic")
    except ValueError:
        return
    raise AssertionError("неизвестный режим апсемплинга должен давать ValueError")


class _RecordingCondModel(nn.Module):
    """Сеть с каналом условия, записывающая, что ей передали как cond."""

    cond_ch = 1

    def __init__(self):
        super().__init__()
        self.seen = []
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, cond):
        self.seen.append(cond.detach().clone())
        return x * 0 + cond + self.w


def test_loss_and_sampler_pass_the_clean_planck_as_cond():
    """И лосс, и сэмплер обязаны передавать сети ЧИСТЫЙ Planck, а не x_t.

    Иначе модель на обучении видит одно условие, на инференсе -- другое.
    """
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(2, 1, 16, 16, generator=g)
    y = torch.randn(2, 1, 16, 16, generator=g)

    model = _RecordingCondModel()
    bbdm = BBDM(model, T=T_STEPS, s=S_VAR, eta=0.02)

    bbdm.loss(x0, y)
    assert torch.equal(model.seen[-1], x0), "лосс передал не x0"

    model.seen.clear()
    out = bbdm.sample(x0, S=10)
    assert len(model.seen) == 10
    assert all(torch.equal(c, x0) for c in model.seen), \
        "сэмплер обязан передавать чистый y_cond, а не зашумлённое стартовое состояние"
    # сеть возвращает cond, последний шаг возвращает pred -> выход == Planck
    assert torch.allclose(out, x0, atol=1e-5)


def test_r_input_is_one_when_output_is_the_input():
    """r_in (корреляция выхода со входом) == 1, если сэмплер возвращает сам Planck."""
    ds = _PairDataset(4, size=32)
    bbdm = BBDM(_RecordingCondModel(), T=T_STEPS, s=S_VAR, eta=0.0)
    res = evaluate_spectra(bbdm, ds, mu=0.0, sigma=1.0, n_patches=4, S=5,
                           device="cpu", batch_size=2, progress=False,
                           bands=[(0.1, 0.9)])
    assert abs(res["r_input_bands"][0] - 1.0) < 1e-6


# ---------------------------------------------------- чекпоинты и обучение


def _fake_checkpoint(path, unet, with_hparams):
    bbdm = BBDM(unet, T=T_STEPS, s=S_VAR, eta=0.0)
    state = {"model": bbdm.state_dict(), "ema": bbdm.model.state_dict(), "epoch": 0}
    if with_hparams:
        state["hparams"] = {"T": T_STEPS, "s": S_VAR, "eta": 0.0, "groups": 4}
    torch.save(state, path)
    return bbdm


def test_load_checkpoint_handles_old_and_new_architectures():
    """Один вызов открывает и run-3/4 (transpose, без условия, без hparams), и новые."""
    x = torch.randn(1, 1, 32, 32)
    t = torch.tensor([300])
    cond = torch.randn(1, 1, 32, 32)

    with tempfile.TemporaryDirectory() as d:
        # У чекпоинтов run 3/4 hparams про архитектуру нет, а число групп
        # GroupNorm по весам не восстановить -- оно берётся из config.GROUPS
        # (8, с ним и обучались run 3/4). Поэтому и здесь groups=8.
        old_path = os.path.join(d, "old.pt")
        torch.manual_seed(0)
        old = _fake_checkpoint(
            old_path, UNet(1, 16, 64, 8, cond_ch=0, upsample="transpose"),
            with_hparams=False)
        loaded_old, _ = load_checkpoint(old_path, device="cpu")
        assert loaded_old.model.upsample == "transpose"
        assert loaded_old.model.cond_ch == 0
        assert loaded_old.T == T_STEPS and abs(loaded_old.s - S_VAR) < 1e-6
        with torch.no_grad():
            assert torch.allclose(loaded_old.denoise(x, t, cond),
                                  old.eval().denoise(x, t, cond), atol=1e-6)

        new_path = os.path.join(d, "new.pt")
        torch.manual_seed(1)
        new = _fake_checkpoint(
            new_path, UNet(1, 8, 64, 4, cond_ch=1, upsample="resize_conv"),
            with_hparams=True)
        loaded_new, _ = load_checkpoint(new_path, device="cpu")
        assert loaded_new.model.upsample == "resize_conv"
        assert loaded_new.model.cond_ch == 1
        with torch.no_grad():
            assert torch.allclose(loaded_new.denoise(x, t, cond),
                                  new.eval().denoise(x, t, cond), atol=1e-6)


def test_train_refuses_to_overwrite_an_existing_run():
    ds = _PairDataset(4, size=16)
    bbdm = BBDM(UNet(1, 8, 64, 4, cond_ch=1), T=100, s=S_VAR)
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "best.pt"), "wb").close()
        try:
            train(bbdm, ds, ds, n_epochs=1, batch_size=2, device="cpu",
                  checkpoint_dir=d, num_workers=0)
        except FileExistsError:
            return
    raise AssertionError("train() перезаписал бы существующий best.pt")


def test_train_with_monitor_records_history_and_architecture():
    ds = _PairDataset(6, size=16)
    bbdm = BBDM(UNet(1, 8, 64, 4, cond_ch=1, upsample="resize_conv"), T=100, s=S_VAR)
    with tempfile.TemporaryDirectory() as d:
        train(bbdm, ds, ds, n_epochs=2, batch_size=2, lr=1e-3, ema_start=2,
              device="cpu", checkpoint_dir=d, num_workers=0,
              monitor_dataset=ds, monitor_every=1, monitor_n=4, monitor_S=5)
        ckpt = torch.load(os.path.join(d, "last.pt"), weights_only=False)

        assert len(ckpt["monitor"]) == 2
        entry = ckpt["monitor"][-1]
        assert len(entry["tf_bands"]) == len(entry["bands"])
        assert "r_input_bands" in entry
        assert ckpt["hparams"]["cond_ch"] == 1
        assert ckpt["hparams"]["upsample"] == "resize_conv"

        # resume подхватывает историю монитора, а не начинает её заново
        train(bbdm, ds, ds, n_epochs=3, batch_size=2, lr=1e-3, ema_start=2,
              device="cpu", checkpoint_dir=d, num_workers=0, resume=True,
              monitor_dataset=ds, monitor_every=1, monitor_n=4, monitor_S=5)
        ckpt = torch.load(os.path.join(d, "last.pt"), weights_only=False)
        assert len(ckpt["monitor"]) == 3

        loaded, _ = load_checkpoint(os.path.join(d, "best.pt"), device="cpu")
        assert loaded.model.cond_ch == 1


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
