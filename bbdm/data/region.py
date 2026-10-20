"""Сплит по области неба (тест = поле D56) и случайные кропы для train.

Геометрия. Тайлы режет `batching_planck.ipynb` (ячейка 23) из полных карт
ACT DR6 / Planck в CAR-проекции: shape (10320, 43200), 0.5'/px, RA = 180 при
x = 0 и убывает вправо, Dec = -63 при y = 0. Имя `tile_y{y}_x{x}.fits` -- это
смещение тайла в этой сетке, поэтому область неба переводится в прямоугольник
пикселей формулами

    x = 21600 - 120 * RA     (RA приведено к [-180, 180))
    y =  7560 + 120 * Dec

(центры пикселей, 0-based; CRPIX = (21601, 7561), CRVAL = (0, 0) в заголовке
карты; ячейка 35 того же ноутбука использует те же формулы). В каждом тайле
лежит его WCS, и `check_tile_wcs` сверяет с ним формулы на реальных файлах.

Сплит. test -- патчи-квадранты, целиком лежащие в прямоугольнике поля; патчи,
задевающие его границу, выбрасываются; val -- пространственные блоки среди
оставшихся; всё остальное небо -- train. Train читается не квадрантами, а
случайными кропами P x P с произвольным смещением внутри тайла
(`RandomCropDataset`), которые не задевают ни поле, ни val-патчи: уникальных
квадрантов на всё небо ~1000, а кропов с произвольным смещением -- сколько угодно.
"""

import math
import os

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset

from bbdm.data.dataset import AUGMENTATIONS, _is_main_process
from bbdm.data.splits import (
    _overlap_index,
    _overlapping,
    parse_tile_pos,
    unique_patch_positions,
)

# Сетка карт ACT DR6 / Planck, из которых нарезаны тайлы.
ACT_CAR = {"shape": (10320, 43200), "x_ra0": 21600.0, "y_dec0": 7560.0, "px_per_deg": 120.0}


def _signed_ra(ra):
    return ((np.asarray(ra, dtype=np.float64) + 180.0) % 360.0) - 180.0


def radec_to_pixel(ra, dec, grid=ACT_CAR):
    """(y, x) центра пикселя (0-based, float) для RA/Dec в градусах."""
    x = grid["x_ra0"] - grid["px_per_deg"] * _signed_ra(ra)
    y = grid["y_dec0"] + grid["px_per_deg"] * np.asarray(dec, dtype=np.float64)
    return y, x


def region_box_px(ra_min, ra_max, dec_min, dec_max, grid=ACT_CAR):
    """Полуоткрытый прямоугольник пикселей (y0, y1, x0, x1) области RA x Dec.

    RA можно задавать как 352 (= -8): поле D56 переходит через RA = 0, но в сетке
    карты это непрерывный кусок. Переход через RA = 180 (шов карты) не поддержан.
    """
    lo, hi = float(_signed_ra(ra_min)), float(_signed_ra(ra_max))
    if lo >= hi:
        raise ValueError(f"область RA {ra_min}..{ra_max} пересекает шов карты RA = 180")
    if dec_min >= dec_max:
        raise ValueError(f"dec_min {dec_min} >= dec_max {dec_max}")
    ya, xa = radec_to_pixel(hi, dec_min, grid)      # x убывает с ростом RA
    yb, xb = radec_to_pixel(lo, dec_max, grid)
    return int(math.ceil(ya)), int(math.floor(yb)) + 1, int(math.ceil(xa)), int(math.floor(xb)) + 1


def check_tile_wcs(header, name, grid=ACT_CAR):
    """Наибольшее расхождение (в пикселях) WCS тайла с формулами `radec_to_pixel`.

    Проверяет углы и центр тайла. Больше ~1 px -- тайлы нарезаны не из этой
    сетки, и прямоугольник поля указывает не туда.
    """
    from astropy.wcs import WCS

    ty, tx = parse_tile_pos(name)
    h, w = header["NAXIS2"], header["NAXIS1"]
    ys = np.array([0, 0, h - 1, h - 1, h // 2], dtype=np.float64)
    xs = np.array([0, w - 1, 0, w - 1, w // 2], dtype=np.float64)
    ra, dec = WCS(header).wcs_pix2world(xs, ys, 0)
    gy, gx = radec_to_pixel(ra, dec, grid)
    width = grid["shape"][1]
    dx = (gx - (tx + xs) + width / 2) % width - width / 2
    return float(max(np.abs(gy - (ty + ys)).max(), np.abs(dx).max()))


def _inside(pos, box, p):
    y0, y1, x0, x1 = box
    return y0 <= pos[0] and pos[0] + p <= y1 and x0 <= pos[1] and pos[1] + p <= x1


def _intersects(pos, box, p):
    y0, y1, x0, x1 = box
    return pos[0] < y1 and y0 < pos[0] + p and pos[1] < x1 and x0 < pos[1] + p


def grow_box(box, guard):
    y0, y1, x0, x1 = box
    return (y0 - guard, y1 + guard, x0 - guard, x1 + guard)


def region_splits(tiles, box, patch_size=480, val_ratio=0.1, block=2, seed=42, guard=0):
    """Сплит уникальных квадрантов: test внутри `box`, val блоками снаружи, train -- остальное.

    Args:
        tiles: имена тайлов (объединение по всем частотам -- сплит общий).
        box: (y0, y1, x0, x1) -- тестовое поле в пикселях карты (`region_box_px`).
        guard: ширина полосы (px) вокруг поля, в которую не заходят train/val.
            0 -- достаточно для отсутствия общих пикселей.

    Returns:
        (splits, info): splits = {"train"/"val"/"test": отсортированные позиции};
        info -- сколько патчей выброшено на границе поля и т.п.
    """
    p = patch_size
    positions = sorted(unique_patch_positions(tiles, p))
    gbox = grow_box(box, guard)
    test = [q for q in positions if _inside(q, box, p)]
    rest = [q for q in positions if not _intersects(q, gbox, p)]
    n_boundary = len(positions) - len(test) - len(rest)

    blocks = {}
    for q in rest:
        blocks.setdefault((q[0] // (block * p), q[1] // (block * p)), []).append(q)
    keys = sorted(blocks)
    np.random.default_rng(seed).shuffle(keys)
    n_val = int(round(len(rest) * val_ratio))
    val, train = [], []
    for k in keys:
        (val if len(val) < n_val else train).extend(blocks[k])

    # Квадранты верхнего ряда тайлов сдвинуты не на кратное P -- могут задевать val.
    vgrid = _overlap_index(val, p)
    n_val_overlap = sum(bool(_overlapping(q, vgrid, p)) for q in train)
    train = [q for q in train if not _overlapping(q, vgrid, p)]

    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    info = {
        "box_px": list(box), "guard_px": guard,
        "n_positions": len(positions),
        "n_dropped_on_region_boundary": n_boundary,
        "n_train_dropped_for_touching_val": n_val_overlap,
    }
    if _is_main_process():
        print(f"Region split: {len(positions)} unique patch positions from {len(tiles)} tiles; "
              f"test (inside the field) {len(test)}, dropped on the field boundary {n_boundary}, "
              f"val {len(val)}, train {len(train)} (+{n_val_overlap} dropped for touching val)",
              flush=True)
    return splits, info


class _RectIndex:
    """Быстрая проверка «окно P x P задевает хоть один запрещённый прямоугольник»."""

    def __init__(self, rects, cell):
        self.rects = [tuple(int(v) for v in r) for r in rects]
        self.cell = cell
        self.grid = {}
        for i, (y0, y1, x0, x1) in enumerate(self.rects):
            for cy in range(y0 // cell, (y1 - 1) // cell + 1):
                for cx in range(x0 // cell, (x1 - 1) // cell + 1):
                    self.grid.setdefault((cy, cx), []).append(i)

    def hits(self, py, px, p):
        c = self.cell
        for cy in range(py // c, (py + p - 1) // c + 1):
            for cx in range(px // c, (px + p - 1) // c + 1):
                for i in self.grid.get((cy, cx), ()):
                    if _intersects((py, px), self.rects[i], p):
                        return True
        return False


def forbidden_rects(splits, box, patch_size=480, guard=0):
    """Куда train-кропам нельзя: поле (с полосой guard) и все val-патчи."""
    p = patch_size
    return [grow_box(box, guard)] + [(y, y + p, x, x + p) for y, x in splits["val"]]


class RandomCropDataset(Dataset):
    """Train-пары: случайные кропы P x P с произвольным смещением внутри тайлов.

    Каждый вызов `__getitem__` даёт новый кроп (индекс не важен): тайл выбирается
    равновероятно, смещение -- равномерно в [0, tile - P]. Кроп отбрасывается и
    тянется заново, если задевает запрещённую область (тестовое поле, val-патчи)
    или не проходит те же фильтры, что `CMBPatchDataset` (доля нулей,
    несовпадение масок Planck/ACT). Затем случайная аугментация (флипы, поворот на 180).

    Тайлы не держатся в памяти: читаются только нужные P строк из FITS-файла
    (массив без BSCALE/BZERO, float32/64), так что ОЗУ не зависит от размера
    неба, а кэш страниц ОС общий для всех процессов на узле.

    Случайность: генератор создаётся лениво из `torch.initial_seed()` -- в
    процессе это сид, заданный скриптом (у каждого ранга свой), в воркерах
    DataLoader -- сид воркера. Поэтому ранги и воркеры не повторяют друг друга.
    """

    def __init__(self, planck_dir, act_dir, tile_list, mu, sigma, forbidden, n_samples,
                 patch_size=480, max_zero_frac=0.2, max_mask_mismatch_frac=0.01,
                 augment=True, max_tries=200):
        self.planck_dir = planck_dir
        self.act_dir = act_dir
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.patch_size = patch_size
        self.max_zero_frac = max_zero_frac
        self.max_mask_mismatch_frac = max_mask_mismatch_frac
        self.augmentations = AUGMENTATIONS if augment else AUGMENTATIONS[:1]
        self.n_samples = int(n_samples)
        self.max_tries = max_tries
        self.forbidden = _RectIndex(forbidden, patch_size)
        self._layout = {}
        self._rng = None
        self.n_rejected = {"forbidden": 0, "zero": 0, "mismatch": 0}

        tiles = sorted(tile_list)
        if not tiles:
            raise ValueError("RandomCropDataset: пустой список тайлов")
        hdr = fits.getheader(os.path.join(act_dir, tiles[0]))
        self.tile_size = (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))
        if min(self.tile_size) < patch_size:
            raise ValueError(f"тайл {self.tile_size} меньше патча {patch_size}")

        # Тайлы, в которых есть хоть одно разрешённое смещение (проверка на сетке).
        step = max(1, patch_size // 16)
        th, tw = self.tile_size
        oys = list(range(0, th - patch_size + 1, step)) + [th - patch_size]
        oxs = list(range(0, tw - patch_size + 1, step)) + [tw - patch_size]
        self.tiles = []
        for t in tiles:
            ty, tx = parse_tile_pos(t)
            if any(not self.forbidden.hits(ty + oy, tx + ox, patch_size) for oy in oys for ox in oxs):
                self.tiles.append((t, ty, tx))
        if not self.tiles:
            raise ValueError("ни в одном тайле нет места для train-кропа вне запрещённой области")
        self.tile_pos = [(ty, tx) for _, ty, tx in self.tiles]
        if _is_main_process():
            print(f"Random crops: {len(self.tiles)} of {len(tiles)} tiles can host a train crop; "
                  f"{self.n_samples} crops of {patch_size} px per epoch", flush=True)

    def __len__(self):
        return self.n_samples

    def normalize(self, patch):
        return (patch - self.mu) / self.sigma

    def denormalize(self, patch_norm):
        return patch_norm * self.sigma + self.mu

    def _get_rng(self):
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed())
        return self._rng

    def sample_window(self, rng):
        """(индекс тайла, oy, ox) разрешённого окна -- только геометрия, без чтения файлов."""
        p = self.patch_size
        th, tw = self.tile_size
        for _ in range(100 * self.max_tries):
            i = int(rng.integers(len(self.tiles)))
            oy = int(rng.integers(0, th - p + 1))
            ox = int(rng.integers(0, tw - p + 1))
            _, ty, tx = self.tiles[i]
            if not self.forbidden.hits(ty + oy, tx + ox, p):
                return i, oy, ox
            self.n_rejected["forbidden"] += 1
        raise RuntimeError("не удалось найти разрешённое окно для кропа")

    def _read_rows(self, directory, name, oy, ox):
        key = (directory, name)
        if key not in self._layout:
            path = os.path.join(directory, name)
            with fits.open(path, memmap=False) as hdul:
                hdu = hdul[0]
                bitpix = hdu.header["BITPIX"]
                if (bitpix not in (-32, -64) or hdu.header.get("BSCALE", 1) != 1
                        or hdu.header.get("BZERO", 0) != 0):
                    raise ValueError(f"{path}: ожидался несжатый float-массив (BITPIX={bitpix})")
                shape = (int(hdu.header["NAXIS2"]), int(hdu.header["NAXIS1"]))
                if shape != self.tile_size:
                    raise ValueError(f"{path}: размер {shape}, а у первого тайла {self.tile_size}")
                self._layout[key] = (path, hdu.fileinfo()["datLoc"],
                                     np.dtype(">f4" if bitpix == -32 else ">f8"))
        path, offset, dtype = self._layout[key]
        p, w = self.patch_size, self.tile_size[1]
        with open(path, "rb") as f:
            f.seek(offset + oy * w * dtype.itemsize)
            buf = f.read(p * w * dtype.itemsize)
        rows = np.frombuffer(buf, dtype=dtype).reshape(p, w)
        return np.ascontiguousarray(rows[:, ox:ox + p], dtype=np.float32)

    def read_crop(self, i, oy, ox):
        name = self.tiles[i][0]
        return (self._read_rows(self.planck_dir, name, oy, ox),
                self._read_rows(self.act_dir, name, oy, ox))

    def __getitem__(self, idx):
        rng = self._get_rng()
        for _ in range(self.max_tries):
            i, oy, ox = self.sample_window(rng)
            pp, pa = self.read_crop(i, oy, ox)
            valid_p, valid_a = pp != 0, pa != 0
            if (1.0 - valid_p.mean()) > self.max_zero_frac or (1.0 - valid_a.mean()) > self.max_zero_frac:
                self.n_rejected["zero"] += 1
                continue
            if np.logical_xor(valid_p, valid_a).mean() > self.max_mask_mismatch_frac:
                self.n_rejected["mismatch"] += 1
                continue
            aug = self.augmentations[int(rng.integers(len(self.augmentations)))]
            x0 = torch.from_numpy(self.normalize(aug(pp))).unsqueeze(0).float()
            y = torch.from_numpy(self.normalize(aug(pa))).unsqueeze(0).float()
            return x0, y
        raise RuntimeError(f"{self.max_tries} кропов подряд не прошли фильтры нулей/масок: "
                           f"{self.n_rejected}")


def crop_audit(ds, splits, n=20000, seed=0):
    """Геометрический аудит: сколько из n случайных train-окон задевают val/test-патчи.

    У корректного датасета оба числа -- ноль. Файлы не читаются.
    """
    p = ds.patch_size
    rng = np.random.default_rng(seed)
    grids = {k: _overlap_index(splits[k], p) for k in ("val", "test")}
    hits = {"val": 0, "test": 0}
    for _ in range(n):
        i, oy, ox = ds.sample_window(rng)
        ty, tx = ds.tile_pos[i]
        for k, g in grids.items():
            if _overlapping((ty + oy, tx + ox), g, p):
                hits[k] += 1
    return {"n_crops": n, "crops_touching_val": hits["val"], "crops_touching_test": hits["test"]}
