"""Разбиение тайлов на train/val/test и согласование разбиений между частотами."""

import json
import os
import random


def get_tile_splits(planck_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    tiles = sorted(os.listdir(planck_dir))
    tiles = [t for t in tiles if t.endswith(".fits")]

    random.seed(seed)
    random.shuffle(tiles)

    N = len(tiles)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_tiles = tiles[:n_train]
    val_tiles = tiles[n_train : n_train + n_val]
    test_tiles = tiles[n_train + n_val :]

    print(f"Total tiles: {N}")
    print(f"  Train: {len(train_tiles)} tiles -> {len(train_tiles)*4} patches")
    print(f"  Val:   {len(val_tiles)} tiles -> {len(val_tiles)*4} patches")
    print(f"  Test:  {len(test_tiles)} tiles -> {len(test_tiles)*4} patches")

    return train_tiles, val_tiles, test_tiles


def list_tiles(directory):
    """Отсортированные имена *.fits в папке."""
    return sorted(t for t in os.listdir(directory) if t.endswith(".fits"))


def inherit_splits(master, tiles, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Разбиение для другой частоты, согласованное с уже существующим.

    Зачем. Модели на 90, 150 и 220 ГГц обучаются на одном и том же небе.
    Если для каждой частоты тасовать свой список файлов независимо, тайл из
    test на 90 ГГц окажется в train на 150 ГГц, и любой многочастотный
    анализ (детекция по 90+150) будет оценивать модель на небе, которое
    одна из её частей уже видела. Заодно сплит 90 ГГц обязан остаться
    прежним -- по нему обучена и оценена модель run 5.

    Правило: тайл, уже лежащий в `master`, остаётся в том же сплите;
    тайлы, которых в `master` нет (валидные только на этой частоте, потому
    что `sigma_clip_tile` отбраковывает по порогу доли нулей у пары карт),
    раздаются по тем же долям детерминированно и только среди новых.

    Имена `tile_y{y}_x{x}` обозначают одно и то же небо лишь тогда, когда
    карты разных частот лежат на одной пиксельной сетке; это проверяет
    `compare_tile_grids`.

    Args:
        master: (train, val, test) -- списки имён главной частоты.
        tiles: имена тайлов, доступные на этой частоте.

    Returns:
        (train, val, test) для этой частоты, каждый отсортирован.
    """
    m_train, m_val, m_test = (set(s) for s in master)
    if (m_train & m_val) or (m_train & m_test) or (m_val & m_test):
        raise ValueError("master splits overlap")

    tiles = sorted(set(tiles))
    train = [t for t in tiles if t in m_train]
    val = [t for t in tiles if t in m_val]
    test = [t for t in tiles if t in m_test]

    known = m_train | m_val | m_test
    new = [t for t in tiles if t not in known]

    rng = random.Random(seed)
    rng.shuffle(new)
    n_train = int(len(new) * train_ratio)
    n_val = int(len(new) * val_ratio)
    train += new[:n_train]
    val += new[n_train : n_train + n_val]
    test += new[n_train + n_val :]

    print(
        f"Inherited splits: {len(tiles) - len(new)} tiles from master "
        f"(train/val/test = {len([t for t in tiles if t in m_train])}/"
        f"{len([t for t in tiles if t in m_val])}/{len([t for t in tiles if t in m_test])}), "
        f"{len(new)} new tiles split {n_train}/{n_val}/{len(new) - n_train - n_val}"
    )
    return sorted(train), sorted(val), sorted(test)


_GRID_KEYS = ("NAXIS1", "NAXIS2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2")


def compare_tile_grids(dir_a, dir_b, names=None, rel_tol=1e-6):
    """Сверка WCS-заголовков одноимённых тайлов из двух папок.

    Читает только заголовки (быстро). Возвращает список имён, у которых
    сетка отличается; пустой список -- имена в обеих папках обозначают одно
    и то же небо, и наследование сплита по имени корректно.
    """
    from astropy.io import fits

    if names is None:
        names = sorted(set(list_tiles(dir_a)) & set(list_tiles(dir_b)))

    def _differs(ha, hb):
        for k in _GRID_KEYS:
            va, vb = ha.get(k), hb.get(k)
            if va is None or vb is None:
                if va != vb:
                    return True
                continue
            if abs(va - vb) > rel_tol * max(1.0, abs(va), abs(vb)):
                return True
        return False

    bad = []
    for n in names:
        ha = fits.getheader(os.path.join(dir_a, n))
        hb = fits.getheader(os.path.join(dir_b, n))
        if _differs(ha, hb):
            bad.append(n)
    return bad


def save_splits(path, train, val, test, **extra):
    """Записывает сплит (и произвольные метаданные) в JSON."""
    with open(path, "w") as f:
        json.dump({"train": train, "val": val, "test": test, **extra}, f, indent=1)


def load_splits(path):
    with open(path) as f:
        d = json.load(f)
    return d["train"], d["val"], d["test"]
