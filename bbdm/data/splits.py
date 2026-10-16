"""Разбиение тайлов на train/val/test и согласование разбиений между частотами."""

import json
import os
import random


def split_tile_names(names, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Старый сплит по тайлам на СПИСКЕ имён (без файловой системы).

    Тот же алгоритм, что и `get_tile_splits` (сортировка, фильтр *.fits, `random.seed`,
    `random.shuffle`), поэтому на тех же именах даёт тот же сплит, что и run 5.
    """
    tiles = sorted(os.path.basename(t.strip()) for t in names if t.strip())
    tiles = [t for t in tiles if t.endswith(".fits")]

    random.seed(seed)
    random.shuffle(tiles)

    N = len(tiles)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    return tiles[:n_train], tiles[n_train : n_train + n_val], tiles[n_train + n_val :]


def get_tile_splits(planck_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    train_tiles, val_tiles, test_tiles = split_tile_names(
        os.listdir(planck_dir), train_ratio, val_ratio, seed
    )
    N = len(train_tiles) + len(val_tiles) + len(test_tiles)

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


# =====================================================================
# Сплит по ПОЗИЦИЯМ ПАТЧЕЙ (без утечки между train / val / test)
# =====================================================================
#
# Почему не по тайлам. Тайлы 960 px нарезаны с шагом 480 px, а патчи -- их
# квадранты 480x480. Правый нижний квадрант тайла (y, x) -- это ровно левый
# верхний квадрант тайла (y+480, x+480): одни и те же пиксели карты, бит в бит
# (различаются лишь границы sigma-clip, которые считаются для каждого тайла
# отдельно). Случайный сплит по ТАЙЛАМ этого не видит: на геометрии реальной
# ACT-карты (21 x 89 тайлов) около 70% val/test-патчей оказываются точными
# копиями какого-то train-патча. Метрики на таких патчах -- не вне выборки, а
# монитор, ранний останов и best_metric.pt по ним не видят переобучения.
#
# Поэтому: (1) патч однозначно определяется абсолютной позицией (py, px) в
# пиксельной сетке карты; (2) уникальные позиции делятся на блоки по
# `block` x `block` патчей, и блок целиком уходит в один сплит; (3) патчи,
# пересекающиеся по пикселям с патчем другого сплита (бывает у выровненных
# не по 480 px последних строк/столбцов тайлов), отбрасываются из сплита с
# меньшим приоритетом (test > val > train).
# Позиции -- в общей сетке ACT, поэтому ОДИН и тот же сплит годится для всех
# частот (если сетки совпадают; это проверяет `compare_tile_grids`).

import re

_TILE_RE = re.compile(r"tile_y(-?\d+)_x(-?\d+)")


def parse_tile_pos(name):
    """(y, x) из имени `tile_y{y}_x{x}.fits`."""
    m = _TILE_RE.search(os.path.basename(name))
    if m is None:
        raise ValueError(
            f"{name!r}: имя тайла не вида tile_y{{y}}_x{{x}}.fits -- позицию патча по нему не определить"
        )
    return int(m.group(1)), int(m.group(2))


def tile_patch_positions(name, patch_size=480):
    """Абсолютные позиции четырёх патчей тайла в порядке `tile_to_patches` (i*2 + j)."""
    y, x = parse_tile_pos(name)
    return [(y + i * patch_size, x + j * patch_size) for i in (0, 1) for j in (0, 1)]


def unique_patch_positions(tiles, patch_size=480):
    """Множество уникальных позиций патчей по списку тайлов."""
    out = set()
    for t in tiles:
        out.update(tile_patch_positions(t, patch_size))
    return out


def _overlap_index(positions, patch_size):
    """Сетка для быстрого поиска патчей, пересекающихся с данным."""
    grid = {}
    for p in positions:
        grid.setdefault((p[0] // patch_size, p[1] // patch_size), []).append(p)
    return grid


def _overlapping(pos, grid, patch_size):
    """Позиции из `grid`, пересекающие по пикселям патч `pos` (включая сам `pos`)."""
    cy, cx = pos[0] // patch_size, pos[1] // patch_size
    hits = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for q in grid.get((cy + dy, cx + dx), ()):
                if abs(q[0] - pos[0]) < patch_size and abs(q[1] - pos[1]) < patch_size:
                    hits.append(q)
    return hits


def get_patch_splits(tiles, patch_size=480, block=2, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Сплит уникальных патчей по пространственным блокам без утечки.

    Args:
        tiles: имена тайлов (можно объединение по всем частотам -- тогда сплит
            общий для всех моделей).
        block: сторона блока в патчах (2 -> 960 px = 8 градусов). Блок целиком
            попадает в один сплит, так что соседи по небу не разнесены между
            train и test внутри блока.

    Returns:
        dict {"train": [(py, px), ...], "val": [...], "test": [...]}, позиции
        отсортированы. Гарантия: ни одна пара патчей из разных сплитов не имеет
        общих пикселей (см. `patch_leakage_report`).
    """
    positions = sorted(unique_patch_positions(tiles, patch_size))
    blocks = {}
    for p in positions:
        blocks.setdefault((p[0] // (block * patch_size), p[1] // (block * patch_size)), []).append(p)

    keys = sorted(blocks)
    random.Random(seed).shuffle(keys)

    n = len(positions)
    n_test = int(round(n * (1.0 - train_ratio - val_ratio)))
    n_val = int(round(n * val_ratio))

    out = {"test": [], "val": [], "train": []}
    for k in keys:
        if len(out["test"]) < n_test:
            out["test"].extend(blocks[k])
        elif len(out["val"]) < n_val:
            out["val"].extend(blocks[k])
        else:
            out["train"].extend(blocks[k])

    # Пересечения между сплитами: у выровненных не по patch_size рядов тайлов
    # патчи могут перекрываться частично. Приоритет test > val > train.
    n_dropped = 0
    for low, high in (("train", "test"), ("train", "val"), ("val", "test")):
        grid = _overlap_index(out[high], patch_size)
        keep = []
        for p in out[low]:
            if _overlapping(p, grid, patch_size):
                n_dropped += 1
            else:
                keep.append(p)
        out[low] = keep

    out = {k: sorted(v) for k, v in out.items()}
    print(
        f"Patch splits: {n} unique patch positions from {len(tiles)} tiles "
        f"({4 * len(tiles)} patch instances), block {block}x{block} patches -> "
        f"train {len(out['train'])}, val {len(out['val'])}, test {len(out['test'])} "
        f"({n_dropped} dropped for overlapping a higher-priority split)"
    )
    return out


def patch_leakage_report(splits, patch_size=480):
    """Аудит: сколько патчей одного сплита совпадают или пересекаются с патчами другого.

    Args:
        splits: dict {"train": positions, "val": ..., "test": ...}.

    Returns:
        dict пар -> {"identical": точные копии, "overlapping": пересекающиеся по
        пикселям (включая копии)}. У корректного сплита всё равно нулю.
    """
    rep = {}
    for a, b in (("val", "train"), ("test", "train"), ("test", "val")):
        grid = _overlap_index(splits[b], patch_size)
        bset = set(splits[b])
        identical = sum(p in bset for p in splits[a])
        overlapping = sum(bool(_overlapping(p, grid, patch_size)) for p in splits[a])
        rep[f"{a}->{b}"] = {"n": len(splits[a]), "identical": identical, "overlapping": overlapping}
    return rep


def legacy_leakage_report(train_tiles, val_tiles, test_tiles, patch_size=480):
    """То же для СТАРОГО сплита по тайлам (`get_tile_splits`): утечка run 5.

    Считает только по именам файлов, данные читать не нужно.
    """
    tr = unique_patch_positions(train_tiles, patch_size)
    rep = {"unique_train_patches": len(tr), "train_patch_instances": 4 * len(train_tiles)}
    grid = _overlap_index(tr, patch_size)
    for name, tiles in (("val", val_tiles), ("test", test_tiles)):
        q = [p for t in tiles for p in tile_patch_positions(t, patch_size)]
        rep[name] = {
            "patch_instances": len(q),
            "identical_to_train": sum(p in tr for p in q),
            "overlapping_train": sum(bool(_overlapping(p, grid, patch_size)) for p in q),
        }
    return rep


def clean_eval_positions(train_tiles, eval_tiles, patch_size=480):
    """Разбивает патчи `eval_tiles` на невиденные и утёкшие относительно train.

    Returns:
        (clean, leaked): множества позиций; `clean` не пересекается по пикселям
        НИ с одним train-патчем -- на них метрики честные.
    """
    train_grid = _overlap_index(unique_patch_positions(train_tiles, patch_size), patch_size)
    clean, leaked = set(), set()
    for p in unique_patch_positions(eval_tiles, patch_size):
        (leaked if _overlapping(p, train_grid, patch_size) else clean).add(p)
    return clean, leaked


def save_patch_splits(path, splits, patch_size=480, **meta):
    with open(path, "w") as f:
        json.dump({"kind": "patch_positions", "patch_size": patch_size, **meta,
                   **{k: [list(p) for p in v] for k, v in splits.items()}}, f)


def load_patch_splits(path):
    with open(path) as f:
        d = json.load(f)
    if d.get("kind") != "patch_positions":
        raise ValueError(f"{path}: это не сплит по позициям патчей (kind={d.get('kind')!r})")
    return {k: [tuple(p) for p in d[k]] for k in ("train", "val", "test")}, d["patch_size"]
