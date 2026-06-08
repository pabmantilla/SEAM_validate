"""Merge per-chunk ISM outputs into the single file SEAM_explainer.py expects.

Concatenates ism_raw / ism_centered / predictions in start-order across all
{SEQ_ID}_chunk{start}-{end}.h5 files into {SEQ_ID}.h5. Verifies the chunks
tile [0, n_total) with no gaps/overlaps before writing.

Usage:
    python merge_ism_chunks.py [--seq-idx 3609]              # require full coverage
    python merge_ism_chunks.py [--seq-idx 3609] --if-complete  # no-op unless all chunks present
                                              # (called at the end of every ISM job;
                                              #  the last finisher triggers the merge)
"""

import argparse
import csv
import os
import re
import numpy as np
import h5py
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS_CSV = PROJ_ROOT / "single_seq/seq_subsets/seam_seeds.csv"
ISM_DIR = PROJ_ROOT / "results/single_seq/ism"


def seq_id_for_idx(seq_idx):
    """Resolve seq_idx -> filename-safe seq_id via seam_seeds.csv (trailing ':' stripped)."""
    with open(SEEDS_CSV) as f:
        for row in csv.DictReader(f):
            if int(row['seq_idx']) == seq_idx:
                return row['seq_id'].rstrip(':')
    raise SystemExit(f"seq_idx {seq_idx} not found in {SEEDS_CSV}")


def collect_chunks(seq_id):
    chunk_re = re.compile(rf"^{re.escape(seq_id)}_chunk(\d+)-(\d+)\.h5$")
    chunks = []
    for p in ISM_DIR.iterdir():
        m = chunk_re.match(p.name)
        if m:
            chunks.append((int(m.group(1)), int(m.group(2)), p))
    chunks.sort()
    return chunks


def coverage_complete(chunks):
    """True iff the chunks tile [0, n_total) contiguously with no gap/overlap."""
    if not chunks:
        return False, None
    n_total = None
    cursor = 0
    for start, end, p in chunks:
        with h5py.File(p, 'r') as f:
            nt = int(f.attrs['n_total'])
        n_total = nt if n_total is None else n_total
        if nt != n_total or start != cursor:
            return False, n_total
        cursor = end
    return cursor == n_total, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-idx', type=int, default=3609,
                        help='seq_idx in seam_seeds.csv (default 3609 = peak27535_Reversed)')
    parser.add_argument('--if-complete', action='store_true',
                        help='no-op unless all chunks are present (for end-of-job auto-merge)')
    args = parser.parse_args()
    SEQ_ID = seq_id_for_idx(args.seq_idx)

    out_path = ISM_DIR / f"{SEQ_ID}.h5"
    chunks = collect_chunks(SEQ_ID)
    complete, _ = coverage_complete(chunks)

    if args.if_complete:
        if not complete:
            covered = sum(e - s for s, e, _ in chunks)
            print(f"[if-complete] coverage not yet complete ({covered} seqs, {len(chunks)} chunks) - skipping merge")
            return
        if out_path.exists():
            print(f"[if-complete] {out_path.name} already exists - skipping merge")
            return
        # atomic lock: only the first finisher that wins the mkdir does the merge
        lock = ISM_DIR / f".{SEQ_ID}.merge.lock"
        try:
            os.mkdir(lock)
        except FileExistsError:
            print("[if-complete] another job is already merging - skipping")
            return
    else:
        assert complete, "chunks do not tile [0, n_total) - cannot merge"

    n_total = coverage_complete(chunks)[1]
    print(f"{len(chunks)} chunks tile [0:{n_total}] contiguously")

    ism_raw = np.zeros((n_total, 200, 4), dtype=np.float32)
    ism_centered = np.zeros((n_total, 200, 4), dtype=np.float32)
    predictions = np.zeros(n_total, dtype=np.float32)

    for start, end, p in chunks:
        with h5py.File(p, 'r') as f:
            ism_raw[start:end] = f['ism_raw'][:]
            ism_centered[start:end] = f['ism_centered'][:]
            predictions[start:end] = f['predictions'][:]
        print(f"  loaded {p.name}  [{start}:{end}]")

    tmp_path = ISM_DIR / f".{SEQ_ID}.h5.tmp"
    with h5py.File(tmp_path, 'w') as f:
        f.create_dataset('ism_raw', data=ism_raw, compression='gzip', compression_opts=4)
        f.create_dataset('ism_centered', data=ism_centered, compression='gzip', compression_opts=4)
        f.create_dataset('predictions', data=predictions)
        f.attrs['seq_id'] = SEQ_ID
        f.attrs['cell_type'] = 'K562'
        f.attrs['n_total'] = int(n_total)
        f.attrs['start'] = 0
        f.attrs['end'] = int(n_total)
        f.attrs['var_start'] = 15
        f.attrs['var_end'] = 215
        f.attrs['alphabet'] = 'ACGT'
        f.attrs['format'] = 'NLC'
        f.attrs['method'] = 'ISM (saturation single-position mutagenesis); ism_centered is mean-centered across channels'
        f.attrs['merged_from'] = f"{len(chunks)} chunks"

    os.replace(tmp_path, out_path)   # atomic publish
    if args.if_complete:
        os.rmdir(ISM_DIR / f".{SEQ_ID}.merge.lock")

    print(f"\nDone -> {out_path}")
    print(f"  ism_raw.shape={ism_raw.shape}  predictions.shape={predictions.shape}")


if __name__ == '__main__':
    main()
