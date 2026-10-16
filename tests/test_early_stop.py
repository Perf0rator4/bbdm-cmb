"""Останов и выбор чекпоинта по метрикам: логика EarlyStopper и её место в train().

Запуск: `pytest tests/` или `python tests/test_early_stop.py`. CPU, секунды.
"""

import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdm.early_stop import EarlyStopper  # noqa: E402
from bbdm.model.bbdm import BBDM  # noqa: E402
from bbdm.model.unet import UNet  # noqa: E402
from bbdm.train import train  # noqa: E402

BANDS = [(0.005, 0.03), (0.03, 0.10), (0.10, 0.30), (0.30, 1.00)]
NAN = float("nan")


def entry(epoch, psnr, tf=(1.0, 1.0, 0.9, 0.8), r=(1.0, 0.97, 0.45, 0.01)):
    return {"epoch": epoch, "bands": BANDS, "tf_bands": list(tf), "r_bands": list(r), "psnr": psnr}


def run(stopper, entries):
    """Прогоняет записи; возвращает (номер проверки останова или None, список решений)."""
    decs = []
    for i, e in enumerate(entries):
        d = stopper.update(e)
        decs.append(d)
        if d.stop:
            return i, decs
    return None, decs


def test_improving_psnr_never_stops():
    st = EarlyStopper(patience=3, warmup_checks=0)
    stop_at, _ = run(st, [entry(i, 20 + 0.5 * i) for i in range(30)])
    assert stop_at is None


def test_psnr_plateau_stops_after_patience():
    st = EarlyStopper(patience=4, warmup_checks=0, psnr_min_delta=0.05)
    seq = [entry(0, 25.0)] + [entry(i, 25.0) for i in range(1, 10)]
    stop_at, decs = run(st, seq)
    # проверка 0 -- улучшение (первое значение), затем 4 без улучшения -> стоп на 4-й
    assert stop_at == 4, stop_at
    assert "plateau" in decs[-1].reason


def test_gain_below_min_delta_does_not_reset_patience():
    """Рост на 0.01 дБ за проверку -- не улучшение: Keras-семантика, best не ползёт вверх."""
    st = EarlyStopper(patience=4, warmup_checks=0, psnr_min_delta=0.05)
    stop_at, _ = run(st, [entry(i, 25.0 + 0.01 * i) for i in range(12)])
    assert stop_at == 4, stop_at


def test_smaller_min_delta_counts_the_same_gain():
    st = EarlyStopper(patience=4, warmup_checks=0, psnr_min_delta=0.005)
    stop_at, _ = run(st, [entry(i, 25.0 + 0.01 * i) for i in range(12)])
    assert stop_at is None


def test_r_ell_growth_delays_the_stop_even_with_flat_psnr():
    """r_ell в информативной полосе растёт -- обучение не останавливается."""
    st = EarlyStopper(patience=3, warmup_checks=0, r_min_delta=0.01)
    seq = [entry(i, 25.0, r=(1.0, 0.97, 0.30 + 0.02 * i, 0.01)) for i in range(15)]
    stop_at, _ = run(st, seq)
    assert stop_at is None

    # то же без роста r_ell -- стоп
    st = EarlyStopper(patience=3, warmup_checks=0)
    stop_at, _ = run(st, [entry(i, 25.0) for i in range(15)])
    assert stop_at == 3


def test_r_ell_outside_tracked_bands_is_ignored():
    """Рост r_ell в полосе без информации (0.30-1.00) не продлевает обучение."""
    st = EarlyStopper(patience=3, warmup_checks=0)
    seq = [entry(i, 25.0, r=(1.0, 0.97, 0.45, 0.01 + 0.05 * i)) for i in range(10)]
    stop_at, _ = run(st, seq)
    assert stop_at == 3


def test_tf_guard_needs_consecutive_strikes():
    st = EarlyStopper(patience=100, warmup_checks=0, tf_max=2.0, tf_strikes=2)
    bad = (1.0, 1.0, 6.3, 9.5)       # run 4
    # одиночный выброс не останавливает
    stop_at, _ = run(st, [entry(0, 20), entry(1, 21, tf=bad), entry(2, 22), entry(3, 23)])
    assert stop_at is None
    # два подряд -- останавливает
    st = EarlyStopper(patience=100, warmup_checks=0, tf_max=2.0, tf_strikes=2)
    stop_at, decs = run(st, [entry(0, 20), entry(1, 21, tf=bad), entry(2, 22, tf=bad)])
    assert stop_at == 2
    assert "TF guard" in decs[-1].reason


def test_tf_below_one_in_uninformative_band_is_fine():
    """Выше луча Planck недобор мощности допустим (честная реализация, r_ell ~ 0)."""
    st = EarlyStopper(warmup_checks=0)
    d = st.update(entry(0, 25.0, tf=(1.0, 1.0, 0.9, 0.2)))
    assert d.tf_ok and d.is_best


def test_tf_off_in_signal_band_blocks_best_checkpoint_but_not_training():
    st = EarlyStopper(patience=100, warmup_checks=0, tf_tol=0.15)
    d1 = st.update(entry(0, 25.0))                                   # TF в норме
    d2 = st.update(entry(1, 28.0, tf=(1.0, 0.7, 0.9, 0.8)))          # PSNR выше, но TF 0.7 в 0.03-0.10
    assert d1.is_best and not d2.is_best and not d2.tf_ok
    assert st.best_eligible_epoch == 0 and st.best_eligible_psnr == 25.0
    assert d2.improved and not d2.stop


def test_best_metric_requires_strictly_higher_psnr():
    st = EarlyStopper(warmup_checks=0)
    assert st.update(entry(0, 25.0)).is_best
    assert not st.update(entry(1, 25.0)).is_best
    assert st.update(entry(2, 25.01)).is_best


def test_nan_psnr_and_nan_tf_bands_are_handled():
    st = EarlyStopper(patience=2, warmup_checks=0)
    d = st.update(entry(0, NAN, tf=(NAN, NAN, 0.9, 0.8), r=(NAN, 0.97, 0.45, 0.01)))
    assert d.tf_ok and not d.is_best          # нет PSNR -- не кандидат в best
    assert d.improved                          # но r_ell улучшился (первое значение)


def test_warmup_blocks_stopping():
    st = EarlyStopper(patience=1, warmup_checks=3, tf_strikes=1)
    bad = (1.0, 1.0, 6.3, 9.5)
    stop_at, _ = run(st, [entry(i, 25.0, tf=bad) for i in range(10)])
    assert stop_at == 3, "стоп не раньше (warmup_checks + 1)-й проверки"


def test_state_roundtrip_continues_identically():
    seq = [entry(i, 25.0 + (0.5 if i < 3 else 0.0)) for i in range(12)]
    full = EarlyStopper(patience=4, warmup_checks=0)
    stop_full, _ = run(full, seq)

    a = EarlyStopper(patience=4, warmup_checks=0)
    run(a, seq[:3])          # режем ДО остановки (полный прогон останавливается на 4)
    b = EarlyStopper(patience=4, warmup_checks=0)
    b.load_state_dict(a.state_dict())
    stop_b, _ = run(b, seq[3:])
    assert stop_full == 3 + stop_b
    assert b.stopped and b.stop_reason == full.stop_reason


def test_state_dict_is_a_snapshot_not_a_view():
    st = EarlyStopper(warmup_checks=0)
    st.update(entry(0, 25.0))
    snap = st.state_dict()
    st.update(entry(1, 40.0, r=(1.0, 0.99, 0.9, 0.5)))
    assert snap["n_checks"] == 1 and snap["best_r"] != st.best_r


# ------------------------------------------------------------ в цикле train()


class _DS:
    def __init__(self, n, size=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = [(torch.randn(1, size, size, generator=g), torch.randn(1, size, size, generator=g))
                      for _ in range(n)]
        self.mu, self.sigma = 0.0, 1.0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _fit(tmp, stopper, epochs, resume=False, monitor_every=1):
    ds = _DS(8)
    net = UNet(in_ch=1, base_ch=8, time_dim=64, groups=4, cond_ch=1)
    bbdm = BBDM(net, T=100, s=0.5)
    return train(bbdm, ds, ds, n_epochs=epochs, batch_size=2, lr=1e-3, ema_start=2,
                 device="cpu", checkpoint_dir=tmp, num_workers=0, resume=resume,
                 monitor_dataset=ds, monitor_every=monitor_every, monitor_n=4, monitor_S=3,
                 early_stop=stopper)


def test_train_stops_early_saves_reason_and_best_metric():
    with tempfile.TemporaryDirectory() as d:
        # tf_max < 0: любое положительное TF -- нарушение -> стоп после warmup
        st = EarlyStopper(tf_max=-1.0, tf_strikes=1, warmup_checks=0)
        _fit(d, st, epochs=6)
        ck = torch.load(os.path.join(d, "last.pt"), weights_only=False)
        assert ck["epoch"] == 0, f"ожидали стоп на первой проверке, эпоха {ck['epoch']}"
        assert "TF guard" in ck["stop_reason"]
        assert ck["early_stop"]["stopped"] is True
        assert len(ck["monitor"]) == 1 and "psnr" in ck["monitor"][0]
        assert math.isfinite(ck["monitor"][0]["psnr"]) or math.isnan(ck["monitor"][0]["psnr"])


def test_train_writes_best_metric_checkpoint():
    with tempfile.TemporaryDirectory() as d:
        # tf_tol огромный -> TF всегда «в норме»; PSNR есть -> первая же проверка = best
        st = EarlyStopper(tf_tol=1e9, tf_max=1e9, patience=100, warmup_checks=0)
        _fit(d, st, epochs=2)
        assert os.path.exists(os.path.join(d, "best_metric.pt"))
        ck = torch.load(os.path.join(d, "best_metric.pt"), weights_only=False)
        assert ck["early_stop"]["best_eligible_epoch"] is not None
        last = torch.load(os.path.join(d, "last.pt"), weights_only=False)
        assert last["stop_reason"] is None and last["epoch"] == 1


def test_resume_after_stop_does_not_train_further():
    with tempfile.TemporaryDirectory() as d:
        st = EarlyStopper(tf_max=-1.0, tf_strikes=1, warmup_checks=0)
        _fit(d, st, epochs=6)
        before = torch.load(os.path.join(d, "last.pt"), weights_only=False)

        st2 = EarlyStopper(tf_max=-1.0, tf_strikes=1, warmup_checks=0)
        _fit(d, st2, epochs=6, resume=True)
        after = torch.load(os.path.join(d, "last.pt"), weights_only=False)
        assert after["epoch"] == before["epoch"] and after["global_step"] == before["global_step"]


def test_early_stop_state_survives_resume_mid_run():
    with tempfile.TemporaryDirectory() as d:
        st = EarlyStopper(patience=100, warmup_checks=0, tf_tol=1e9, tf_max=1e9)
        _fit(d, st, epochs=2)
        n1 = torch.load(os.path.join(d, "last.pt"), weights_only=False)["early_stop"]["n_checks"]
        st2 = EarlyStopper(patience=100, warmup_checks=0, tf_tol=1e9, tf_max=1e9)
        _fit(d, st2, epochs=4, resume=True)
        n2 = torch.load(os.path.join(d, "last.pt"), weights_only=False)["early_stop"]["n_checks"]
        assert (n1, n2) == (2, 4), (n1, n2)


def test_monitor_every_controls_how_often_the_decision_is_made():
    with tempfile.TemporaryDirectory() as d:
        st = EarlyStopper(patience=100, warmup_checks=0, tf_tol=1e9, tf_max=1e9)
        _fit(d, st, epochs=4, monitor_every=2)
        ck = torch.load(os.path.join(d, "last.pt"), weights_only=False)
        assert ck["early_stop"]["n_checks"] == 2          # эпохи 2 и 4
        assert [m["epoch"] for m in ck["monitor"]] == [1, 3]


def test_early_stop_without_monitor_is_rejected():
    ds = _DS(4)
    bbdm = BBDM(UNet(in_ch=1, base_ch=8, time_dim=64, groups=4, cond_ch=1), T=100, s=0.5)
    with tempfile.TemporaryDirectory() as d:
        try:
            train(bbdm, ds, ds, n_epochs=1, batch_size=2, device="cpu", checkpoint_dir=d,
                  num_workers=0, early_stop=EarlyStopper())
        except ValueError:
            return
    raise AssertionError("early_stop без монитора должен давать ValueError, а не молча не работать")


def test_evaluate_spectra_psnr_matches_evaluate_image_metrics():
    """PSNR монитора и итоговый PSNR -- одна и та же величина."""
    from bbdm.evaluate import evaluate_image_metrics, evaluate_spectra

    ds = _DS(6, size=64)
    bbdm = BBDM(UNet(in_ch=1, base_ch=8, time_dim=64, groups=4, cond_ch=1), T=100, s=0.5).eval()
    kw = dict(mu=0.0, sigma=100.0, indices=list(range(6)), S=3, device="cpu",
              batch_size=3, progress=False)
    a = evaluate_spectra(bbdm, ds, with_psnr=True, **kw)
    b = evaluate_image_metrics(bbdm, ds, **kw)
    assert a["psnr_n"] == b["n_patches"] > 0
    assert abs(a["psnr_mean"] - b["psnr_mean"]) < 1e-9


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
