"""Mutagenesis library for a single K562 test-set seed.

100K random mutants @ 10% mutation rate. Only the 200bp variable region
[15:215] of the 230bp insert is mutagenized; the 15bp adapters on each side
are held at WT (matches the attribution window used downstream by SEAM_ism.py).
WT is stored at index 0. The seed is selected by --seq-idx (looked up in
seam_seeds.csv); default 3609 = peak27535_Reversed.

Usage:
    python SEAM_mutagenesis.py [--seq-idx 3609]
"""

import argparse
import csv
import numpy as np
import h5py
from pathlib import Path
import squid

# Config
LIB_SIZE = 100_000
MUT_RATE = 0.10
SEQ_LENGTH = 230
VAR_START, VAR_END = 15, 215   # 200bp variable region; adapters [0:15],[215:230] fixed
SEED = 42

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS_CSV = PROJ_ROOT / "single_seq/seq_subsets/seam_seeds.csv"
OUT_DIR = PROJ_ROOT / "results/single_seq/mutagenesis_lib"

ALPHA_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def load_seed(seq_idx):
    """Return (seq_id, wt_seq, mean_value) for the row with seq_idx in seam_seeds.csv.

    Trailing ':' is stripped from seq_id so it is filename-safe (matches the
    existing peak27535_Reversed convention).
    """
    with open(SEEDS_CSV) as f:
        for row in csv.DictReader(f):
            if int(row['seq_idx']) == seq_idx:
                return row['seq_id'].rstrip(':'), row['seq'], float(row['true'])
    raise SystemExit(f"seq_idx {seq_idx} not found in {SEEDS_CSV}")


def str_to_onehot(seq_str):
    ohe = np.zeros((len(seq_str), 4), dtype=np.float32)
    for j, base in enumerate(seq_str):
        if base in ALPHA_MAP:
            ohe[j, ALPHA_MAP[base]] = 1.0
    return ohe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-idx', type=int, default=3609,
                        help='seq_idx in seam_seeds.csv (default 3609 = peak27535_Reversed)')
    parser.add_argument('--lib-size', type=int, default=LIB_SIZE,
                        help=f'number of mutants incl. WT (default {LIB_SIZE})')
    parser.add_argument('--out-dir', type=str, default=str(OUT_DIR),
                        help='output mutagenesis-lib dir (default 100k)')
    parser.add_argument('--mut-rate', type=float, default=MUT_RATE,
                        help=f'per-position mutation rate (default {MUT_RATE})')
    args = parser.parse_args()
    mut_rate = args.mut_rate
    SEQ_ID, WT_SEQ, MEAN_VALUE = load_seed(args.seq_idx)
    lib_size = args.lib_size
    out_dir = Path(args.out_dir)

    assert len(WT_SEQ) == SEQ_LENGTH, f"WT_SEQ len {len(WT_SEQ)} != {SEQ_LENGTH}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{SEQ_ID}.h5"
    if out_file.exists():
        print(f"Already exists: {out_file}")
        return

    print(f"seq_idx={args.seq_idx}  seq_id={SEQ_ID}  mean_value={MEAN_VALUE}")
    print(f"Building {lib_size} mutants @ {mut_rate:.0%} on {SEQ_ID}")
    mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=mut_rate, seed=SEED)

    wt_onehot = str_to_onehot(WT_SEQ)                                # (230, 4)
    wt_var = wt_onehot[VAR_START:VAR_END]                            # (200, 4)
    var_mut = mut_generator(wt_var, num_sim=lib_size - 1)            # (lib_size-1, 200, 4)
    var_all = np.concatenate([wt_var[np.newaxis], var_mut], axis=0)  # (lib_size, 200, 4)
    x_all = np.broadcast_to(wt_onehot, (lib_size, SEQ_LENGTH, 4)).copy()
    x_all[:, VAR_START:VAR_END, :] = var_all

    with h5py.File(out_file, 'w') as f:
        f.create_dataset('sequences', data=x_all, dtype='float32',
                         compression='gzip', compression_opts=4)
        f.create_dataset('wt_sequence', data=wt_onehot, dtype='float32')
        f.attrs['seq_id'] = SEQ_ID
        f.attrs['wt_seq_str'] = WT_SEQ
        f.attrs['mean_value'] = float(MEAN_VALUE)
        f.attrs['cell_type'] = 'K562'
        f.attrs['n_mutants'] = lib_size
        f.attrs['mut_rate'] = mut_rate
        f.attrs['seq_length'] = SEQ_LENGTH
        f.attrs['var_start'] = VAR_START
        f.attrs['var_end'] = VAR_END
        f.attrs['alphabet'] = 'ACGT'

    print(f"Done: {out_file}  (shape={x_all.shape})")


if __name__ == '__main__':
    main()
