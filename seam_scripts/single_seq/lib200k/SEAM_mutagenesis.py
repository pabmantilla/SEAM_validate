"""Mutagenesis library for a single K562 test-set seed -- 200K variant.

Identical to seam_scripts/single_seq/SEAM_mutagenesis.py but builds 200K random
mutants and writes to results/single_seq/mutagenesis_lib_200k/, leaving the
100K libraries in place. Only the 200bp variable region [15:215] is mutagenized;
the 15bp adapters on each side are held at WT. WT is stored at index 0.

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
LIB_SIZE = 200_000
MUT_RATE = 0.10
SEQ_LENGTH = 230
VAR_START, VAR_END = 15, 215   # 200bp variable region; adapters [0:15],[215:230] fixed
SEED = 42

PROJ_ROOT = Path(__file__).resolve().parents[3]
SEEDS_CSV = PROJ_ROOT / "single_seq/seq_subsets/seam_seeds.csv"
OUT_DIR = PROJ_ROOT / "results/single_seq/mutagenesis_lib_200k"

ALPHA_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def load_seed(seq_idx):
    """Return (seq_id, wt_seq, mean_value) for the row with seq_idx in seam_seeds.csv."""
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
    args = parser.parse_args()
    SEQ_ID, WT_SEQ, MEAN_VALUE = load_seed(args.seq_idx)

    assert len(WT_SEQ) == SEQ_LENGTH, f"WT_SEQ len {len(WT_SEQ)} != {SEQ_LENGTH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{SEQ_ID}.h5"
    if out_file.exists():
        print(f"Already exists: {out_file}")
        return

    print(f"seq_idx={args.seq_idx}  seq_id={SEQ_ID}  mean_value={MEAN_VALUE}")
    print(f"Building {LIB_SIZE} mutants @ {MUT_RATE:.0%} on {SEQ_ID}")
    mut_generator = squid.mutagenizer.RandomMutagenesis(mut_rate=MUT_RATE, seed=SEED)

    wt_onehot = str_to_onehot(WT_SEQ)                                # (230, 4)
    wt_var = wt_onehot[VAR_START:VAR_END]                            # (200, 4)
    var_mut = mut_generator(wt_var, num_sim=LIB_SIZE - 1)            # (LIB_SIZE-1, 200, 4)
    var_all = np.concatenate([wt_var[np.newaxis], var_mut], axis=0)  # (LIB_SIZE, 200, 4)
    x_all = np.broadcast_to(wt_onehot, (LIB_SIZE, SEQ_LENGTH, 4)).copy()
    x_all[:, VAR_START:VAR_END, :] = var_all

    with h5py.File(out_file, 'w') as f:
        f.create_dataset('sequences', data=x_all, dtype='float32',
                         compression='gzip', compression_opts=4)
        f.create_dataset('wt_sequence', data=wt_onehot, dtype='float32')
        f.attrs['seq_id'] = SEQ_ID
        f.attrs['wt_seq_str'] = WT_SEQ
        f.attrs['mean_value'] = float(MEAN_VALUE)
        f.attrs['cell_type'] = 'K562'
        f.attrs['n_mutants'] = LIB_SIZE
        f.attrs['mut_rate'] = MUT_RATE
        f.attrs['seq_length'] = SEQ_LENGTH
        f.attrs['var_start'] = VAR_START
        f.attrs['var_end'] = VAR_END
        f.attrs['alphabet'] = 'ACGT'

    print(f"Done: {out_file}  (shape={x_all.shape})")


if __name__ == '__main__':
    main()
