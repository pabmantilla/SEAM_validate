"""SEAM clustering + MetaExplainer background separation for the 100K K562 lib.

Loads the pre-computed ISM attribution maps (from SEAM_ism.py), runs SEAM
K-means clustering (30 clusters) and MetaExplainer to extract scaled
foreground/background. Uses ism_centered as the attribution map (mean-centered
across channels, the analog of the hypothetical/gradient correction).

Usage:
    python SEAM_explainer.py [--seq-idx 3609]
"""

import argparse
import csv
import gc
import numpy as np
import h5py
from pathlib import Path

from seam import Compiler, Clusterer, MetaExplainer

CELL_TYPE = "K562"

DEFAULT_N_CLUSTERS = 50
MUT_RATE = 0.10
ALPHABET = ['A', 'C', 'G', 'T']
VAR_START, VAR_END = 15, 215

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS_CSV = PROJ_ROOT / "single_seq/seq_subsets/seam_seeds.csv"
MUT_LIB_DIR = PROJ_ROOT / "results/single_seq/mutagenesis_lib"
ISM_DIR = PROJ_ROOT / "results/single_seq/ism"
OUT_DIR = PROJ_ROOT / "results/single_seq/foregrounds"


def seq_id_for_idx(seq_idx):
    """Resolve seq_idx -> filename-safe seq_id via seam_seeds.csv (trailing ':' stripped)."""
    with open(SEEDS_CSV) as f:
        for row in csv.DictReader(f):
            if int(row['seq_idx']) == seq_idx:
                return row['seq_id'].rstrip(':')
    raise SystemExit(f"seq_idx {seq_idx} not found in {SEEDS_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-idx', type=int, default=3609,
                        help='seq_idx in seam_seeds.csv (default 3609 = peak27535_Reversed)')
    parser.add_argument('--n-clusters', type=int, default=DEFAULT_N_CLUSTERS,
                        help=f'K-means clusters (default {DEFAULT_N_CLUSTERS})')
    args = parser.parse_args()
    SEQ_ID = seq_id_for_idx(args.seq_idx)
    N_CLUSTERS = args.n_clusters

    seq_dir = OUT_DIR / SEQ_ID
    seq_dir.mkdir(parents=True, exist_ok=True)

    mut_path = MUT_LIB_DIR / f"{SEQ_ID}.h5"
    ism_path = ISM_DIR / f"{SEQ_ID}.h5"
    assert mut_path.exists(), f"missing mutagenesis lib: {mut_path}"
    assert ism_path.exists(), f"missing ism attributions: {ism_path}"

    with h5py.File(mut_path, 'r') as f:
        x_mut = f['sequences'][:, VAR_START:VAR_END, :]   # (N, 200, 4)
        wt_seq = f['wt_sequence'][VAR_START:VAR_END, :]   # (200, 4)

    with h5py.File(ism_path, 'r') as f:
        predictions = f['predictions'][:]      # (N,)
        attributions = f['ism_centered'][:]    # (N, 200, 4)

    print(f"N={len(x_mut)}  L={x_mut.shape[1]}  using ism_centered as attribution maps")

    clusterer = Clusterer(attributions, gpu=False)
    cluster_labels = clusterer.cluster(
        embedding=clusterer.maps,
        method='kmeans',
        n_clusters=N_CLUSTERS,
    )

    compiler = Compiler(
        x=x_mut,
        y=predictions,
        x_ref=wt_seq[np.newaxis],
        y_bg=None,
        alphabet=ALPHABET,
        gpu=False,
    )
    mave_df = compiler.compile()

    clusterer = Clusterer(attributions, gpu=False)
    clusterer.cluster_labels = cluster_labels

    meta = MetaExplainer(
        clusterer=clusterer,
        mave_df=mave_df,
        attributions=attributions,
        sort_method='median',
        ref_idx=0,
        mut_rate=MUT_RATE,
    )
    msm = meta.generate_msm(gpu=False)
    meta.compute_background(
        mut_rate=MUT_RATE,
        entropy_multiplier=0.5,
        adaptive_background_scaling=True,
        process_logos=False,
    )

    if meta.cluster_order is not None:
        mapping = {old: new for new, old in enumerate(meta.cluster_order)}
        meta.membership_df['Cluster_Sorted'] = meta.membership_df['Cluster'].map(mapping)
        ref_cluster = meta.membership_df.loc[0, 'Cluster_Sorted']
    else:
        ref_cluster = meta.membership_df.loc[0, 'Cluster']

    ref_cluster_avg = np.mean(meta.get_cluster_maps(ref_cluster), axis=0)
    bg_scale = meta.background_scaling[ref_cluster] if meta.background_scaling is not None else 1.0
    foreground_scaled = ref_cluster_avg - bg_scale * meta.background

    cluster_maps = np.stack([
        np.mean(meta.get_cluster_maps(k), axis=0) for k in range(N_CLUSTERS)
    ])

    for name, arr in [
        ('foreground_scaled', foreground_scaled),
        ('average_background', meta.background),
        ('average_background_scaled', bg_scale * meta.background),
        ('wt_attribution', attributions[0]),
        ('ref_cluster_avg', ref_cluster_avg),
        ('cluster_labels', cluster_labels),
        ('cluster_maps', cluster_maps),
        ('cluster_backgrounds', meta.cluster_backgrounds),
        ('ref_cluster_idx', np.array(ref_cluster)),
    ]:
        tmp = seq_dir / f'.{name}_tmp'
        np.save(tmp, arr)
        (seq_dir / f'.{name}_tmp.npy').rename(seq_dir / f'{name}.npy')

    print(f"bg_scale={bg_scale:.4f}, ref_cluster={ref_cluster}")
    print(f"Done: {seq_dir}")

    del x_mut, wt_seq, predictions, attributions, clusterer, compiler, mave_df, meta
    gc.collect()


if __name__ == '__main__':
    main()
