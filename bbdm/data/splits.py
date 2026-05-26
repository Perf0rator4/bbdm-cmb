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
