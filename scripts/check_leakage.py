"""Сколько val/test-патчей старого (тайлового) сплита -- копии train-патчей.

Читает только ИМЕНА файлов в папке тайлов Planck, данные не нужны (секунды):

    python scripts/check_leakage.py --planck-dir /path/to/Diffusion/Planck/f100

Сплит воспроизводится ровно как у run 5: `get_tile_splits(planck_dir, 0.8, 0.1, 42)`.
Тайлы 960 px нарезаны с шагом 480 px, патчи -- их квадранты 480x480, так что один
и тот же кусок неба (бит в бит) лежит в нескольких тайлах и случайный сплит по
тайлам разносит копии между train и test.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdm.data import (  # noqa: E402
    get_patch_splits,
    get_tile_splits,
    legacy_leakage_report,
    list_tiles,
    patch_leakage_report,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planck-dir", required=True)
    ap.add_argument("--patch-size", type=int, default=480)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block", type=int, default=2)
    a = ap.parse_args(argv)

    tr, va, te = get_tile_splits(a.planck_dir, a.train_ratio, a.val_ratio, a.seed)
    rep = legacy_leakage_report(tr, va, te, a.patch_size)

    print("\n=== LEGACY split (by tiles, as run 5) ===")
    print(f"train: {rep['train_patch_instances']} patch instances, "
          f"{rep['unique_train_patches']} unique patches "
          f"(each unique patch is stored {rep['train_patch_instances'] / rep['unique_train_patches']:.2f}x)")
    for name in ("val", "test"):
        r = rep[name]
        pct = 100 * r["identical_to_train"] / max(r["patch_instances"], 1)
        pct_o = 100 * r["overlapping_train"] / max(r["patch_instances"], 1)
        print(f"{name:5s}: {r['identical_to_train']}/{r['patch_instances']} patches "
              f"({pct:.0f}%) are exact copies of a train patch; "
              f"{r['overlapping_train']} ({pct_o:.0f}%) share pixels with one")

    sp = get_patch_splits(list_tiles(a.planck_dir), a.patch_size, a.block,
                          a.train_ratio, a.val_ratio, a.seed)
    print("\n=== SPATIAL split (by patch position, blocks) ===")
    for k, v in patch_leakage_report(sp, a.patch_size).items():
        print(f"{k:12s} identical {v['identical']}/{v['n']}, overlapping {v['overlapping']}/{v['n']}")


if __name__ == "__main__":
    main()
