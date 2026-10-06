"""Build the train/val/test splits used to train DDPM-2 and DDPM-6 from the raw CAMELS files.

Inputs (CAMELS Multifield Dataset, IllustrisTNG LH set, z = 0):
    Maps_HI_IllustrisTNG_LH_z=0.00.npy   (15000, 256, 256) HI column density, 15 maps per simulation
    params_LH_IllustrisTNG.txt           (1000, 6) Omega_m, sigma_8, A_SN1, A_AGN1, A_SN2, A_AGN2

Processing (as in the original notebook that wrote the training files):
    1. x = log10(N_HI + 1e-8)
    2. min-max normalise to [0, 1] separately at each pixel position, using the min and max of
       that pixel over all 15000 maps (axis 0), not one global min and max
    3. map i gets the parameters of simulation i // 15
    4. shuffle with numpy.random.default_rng(42).permutation and cut 90 / 5 / 5 %
       -> 13500 train, 750 val, 750 test

Outputs:
    <out_dir>/params_2/{train,val,test}_LH.npy,   {train,val,test}_labels_LH_2.npy
    <out_dir>/params_6/{train,val,test}_LH_6.npy, {train,val,test}_labels_LH.npy
The images are identical in both folders; only the label columns differ.

Memory: the maps are processed in place, but the run needs roughly 2x the 3.9 GB raw array.
Disk: about 7.9 GB, since the images are written once per folder.
"""

import argparse
import os

import numpy as np

MAPS_PER_SIMULATION = 15
LOG_EPS = 1e-8
SPLIT_SEED = 42
TRAIN_FRACTION, VAL_FRACTION = 0.90, 0.05


def normalise_maps(maps):
    """log10, then per-pixel min-max to [0, 1]. Works in place on `maps` (float32) and returns it."""
    maps += LOG_EPS
    np.log10(maps, out=maps)
    lo = maps.min(axis=0)
    span = maps.max(axis=0) - lo
    if np.any(span == 0):
        raise ValueError("some pixel has the same value in every map; per-pixel normalisation is undefined")
    maps -= lo
    maps /= span
    return maps


def split_indices(n_maps):
    """Shuffled indices for the train, validation and test splits."""
    perm = np.random.default_rng(SPLIT_SEED).permutation(n_maps)
    n_train = int(TRAIN_FRACTION * n_maps)
    n_val = int(VAL_FRACTION * n_maps)
    return {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": perm[n_train + n_val:]}


def write_splits(maps, labels, out_dir):
    """Write every split for both label sets (2 and 6 parameters)."""
    dirs = {2: os.path.join(out_dir, "params_2"), 6: os.path.join(out_dir, "params_6")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    for split, idx in split_indices(len(maps)).items():
        split_maps = maps[idx]
        np.save(os.path.join(dirs[2], f"{split}_LH.npy"), split_maps)
        np.save(os.path.join(dirs[6], f"{split}_LH_6.npy"), split_maps)
        del split_maps
        np.save(os.path.join(dirs[2], f"{split}_labels_LH_2.npy"), labels[idx, :2])
        np.save(os.path.join(dirs[6], f"{split}_labels_LH.npy"), labels[idx])
        print(f"{split}: {len(idx)} maps")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maps", required=True, help="Maps_HI_IllustrisTNG_LH_z=0.00.npy")
    p.add_argument("--params", required=True, help="params_LH_IllustrisTNG.txt")
    p.add_argument("--out_dir", required=True, help="parent folder for params_2/ and params_6/")
    args = p.parse_args()

    maps = np.load(args.maps)
    params = np.loadtxt(args.params)
    if maps.ndim != 3 or params.ndim != 2 or params.shape[1] != 6:
        raise ValueError(f"unexpected shapes: maps {maps.shape}, params {params.shape}")
    if len(maps) != MAPS_PER_SIMULATION * len(params):
        raise ValueError(f"{len(maps)} maps for {len(params)} simulations; expected {MAPS_PER_SIMULATION} per simulation")
    if maps.dtype != np.float32:
        raise ValueError(f"expected float32 maps (as in CAMELS), got {maps.dtype}")

    labels = np.repeat(params, MAPS_PER_SIMULATION, axis=0)
    write_splits(normalise_maps(maps), labels, args.out_dir)
    print(f"done: {args.out_dir}")


if __name__ == "__main__":
    main()
