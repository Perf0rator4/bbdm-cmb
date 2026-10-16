"""Останов обучения и выбор чекпоинта по метрикам настоящего сэмплера.

Зачем. Val-лосс -- это MSE одного шага диффузии, и он не видит, что делает
весь обратный процесс: в run 4 он был в порядке все 100 эпох, пока TF на выходе
сэмплера росла до 9.5. Поэтому и останавливать обучение, и выбирать чекпоинт
нужно по метрикам, посчитанным настоящим сэмплером (их считает монитор в
`train.py` на EMA-весах): PSNR, Transfer Function и кросс-корреляция r_ell.

Что считается «улучшением». PSNR определяется крупными масштабами, которые сеть
выучивает быстро, а интересные для скоплений ell ~ 2000+ растут медленнее.
Остановка только по плато PSNR сработала бы, пока r_ell в информативных полосах
ещё растёт. Поэтому проверка считается улучшившейся, если выросло ЧТО-ТО ОДНО из:
PSNR (больше чем на `psnr_min_delta` дБ) или r_ell в любой из отслеживаемых
полос (больше чем на `r_min_delta`). Терпение сбрасывается любым из них.

Правила останова (срабатывает любое):
  * плато: `patience` проверок подряд без улучшения PSNR и r_ell;
  * защита по TF: TF выше `tf_max` (избыток мощности -- режим run 4) в
    `tf_strikes` проверках подряд;
  * (лимит эпох остаётся как раньше).
Раньше `warmup_checks` проверок не останавливает ничего: на первых эпохах
метрики ещё не значат ничего.

TF не требуется равным 1 везде. Выше луча Planck входных данных нет, и честная
реализация там -- TF ~ 1 при r_ell ~ 0, а недобор допустим. Поэтому для выбора
чекпоинта TF проверяется двумя разными условиями: |TF - 1| <= tf_tol в «сигнальных»
полосах (верхняя граница <= tf_signal_hi) и TF <= tf_max во всех.

Чекпоинт `best_metric.pt` -- максимум PSNR среди проверок, где TF в норме.
"""

import math
from dataclasses import dataclass, field


@dataclass
class StopDecision:
    stop: bool = False
    reason: str = None
    is_best: bool = False       # эта проверка -- лучший чекпоинт по метрике
    improved: bool = False      # PSNR или r_ell улучшились (терпение сброшено)
    tf_ok: bool = True
    details: dict = field(default_factory=dict)


def _finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


class EarlyStopper:
    """Состояние останова по метрикам; `update(entry)` вызывается после каждой
    проверки монитора и возвращает `StopDecision`.

    `entry` -- словарь монитора: `epoch`, `bands`, `tf_bands`, `r_bands`,
    `psnr` (nan допустим -- тогда PSNR просто не считается улучшением).
    """

    def __init__(
        self,
        psnr_min_delta=0.05,
        patience=8,
        r_min_delta=0.01,
        r_bands=((0.03, 0.10), (0.10, 0.30)),
        tf_max=2.0,
        tf_strikes=2,
        tf_tol=0.15,
        tf_signal_hi=0.10,
        warmup_checks=3,
    ):
        self.psnr_min_delta = float(psnr_min_delta)
        self.patience = int(patience)
        self.r_min_delta = float(r_min_delta)
        self.r_bands = tuple(tuple(b) for b in r_bands)
        self.tf_max = float(tf_max)
        self.tf_strikes = int(tf_strikes)
        self.tf_tol = float(tf_tol)
        self.tf_signal_hi = float(tf_signal_hi)
        self.warmup_checks = int(warmup_checks)

        # изменяемое состояние (см. state_dict)
        self.n_checks = 0
        self.stale = 0                  # проверок подряд без улучшения
        self.strikes = 0                # проверок подряд с TF > tf_max
        self.best_psnr = -math.inf      # по Keras-семантике: растёт только на >= min_delta
        self.best_psnr_epoch = None
        self.best_r = {}                # полоса -> лучшее r_ell (тоже на >= r_min_delta)
        self.best_eligible_psnr = -math.inf
        self.best_eligible_epoch = None
        self.stopped = False
        self.stop_reason = None

    # ------------------------------------------------------------ помощники

    def _tracked_indices(self, bands):
        """Индексы отслеживаемых полос r_ell в списке полос монитора."""
        idx = [
            i for i, (lo, hi) in enumerate(bands)
            if any(abs(lo - a) < 1e-9 and abs(hi - b) < 1e-9 for a, b in self.r_bands)
        ]
        # Если полосы монитора другие -- лучше следить за всеми, чем ни за чем.
        return idx or list(range(len(bands)))

    def tf_violation(self, entry):
        """(tf_ok, превышение_tf_max). nan-полосы (нет мод) пропускаются."""
        ok, over = True, False
        for (lo, hi), tf in zip(entry["bands"], entry["tf_bands"]):
            if not _finite(tf):
                continue
            if tf > self.tf_max:
                over = True
                ok = False
            elif hi <= self.tf_signal_hi + 1e-9 and abs(tf - 1.0) > self.tf_tol:
                ok = False
        return ok, over

    # ----------------------------------------------------------------- шаг

    def update(self, entry):
        self.n_checks += 1
        epoch = entry.get("epoch")
        psnr = entry.get("psnr")
        bands = entry["bands"]

        tf_ok, over = self.tf_violation(entry)
        self.strikes = self.strikes + 1 if over else 0

        improved = False
        if _finite(psnr) and psnr > self.best_psnr + self.psnr_min_delta:
            self.best_psnr, self.best_psnr_epoch = float(psnr), epoch
            improved = True
        for i in self._tracked_indices(bands):
            r = entry["r_bands"][i]
            key = f"{bands[i][0]:g}-{bands[i][1]:g}"
            if _finite(r) and r > self.best_r.get(key, -math.inf) + self.r_min_delta:
                self.best_r[key] = float(r)
                improved = True
        self.stale = 0 if improved else self.stale + 1

        # Лучший чекпоинт по метрике: строго больший PSNR среди проверок с TF в норме.
        is_best = bool(tf_ok and _finite(psnr) and psnr > self.best_eligible_psnr)
        if is_best:
            self.best_eligible_psnr, self.best_eligible_epoch = float(psnr), epoch

        dec = StopDecision(improved=improved, tf_ok=tf_ok, is_best=is_best)
        dec.details = {"stale": self.stale, "strikes": self.strikes, "n_checks": self.n_checks}

        if self.n_checks > self.warmup_checks:
            if self.strikes >= self.tf_strikes:
                dec.stop = True
                dec.reason = (
                    f"TF guard: TF > {self.tf_max:g} in {self.strikes} consecutive checks "
                    f"(excess power at the sampler output, the run-4 failure mode)"
                )
            elif self.stale >= self.patience:
                dec.stop = True
                best_r = ", ".join(f"{k}: {v:.3f}" for k, v in sorted(self.best_r.items()))
                dec.reason = (
                    f"plateau: no PSNR gain > {self.psnr_min_delta:g} dB and no r_ell gain > "
                    f"{self.r_min_delta:g} for {self.patience} consecutive checks "
                    f"(best PSNR {self.best_psnr:.2f} dB at epoch "
                    f"{(self.best_psnr_epoch + 1) if self.best_psnr_epoch is not None else '?'}; "
                    f"best r_ell {best_r})"
                )
        if dec.stop:
            self.stopped, self.stop_reason = True, dec.reason
        return dec

    # ---------------------------------------------------- сохранение в чекпоинт

    _STATE_KEYS = (
        "n_checks", "stale", "strikes", "best_psnr", "best_psnr_epoch", "best_r",
        "best_eligible_psnr", "best_eligible_epoch", "stopped", "stop_reason",
    )

    def state_dict(self):
        return {k: (dict(getattr(self, k)) if isinstance(getattr(self, k), dict)
                    else getattr(self, k)) for k in self._STATE_KEYS}

    def load_state_dict(self, state):
        for k in self._STATE_KEYS:
            if k in state:
                v = state[k]
                setattr(self, k, dict(v) if isinstance(v, dict) else v)
