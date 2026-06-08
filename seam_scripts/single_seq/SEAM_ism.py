"""ISM attribution maps for the 100K K562 mutagenesis library.

For each of the 100K sequences in the mutagenesis library:
  - The 200bp variable region [15:215] is mutated to each of {A,C,G,T} at every
    position (800 single-position mutants per sequence; ISM = saturation
    single-position mutagenesis, L*4 inputs).
  - The 230bp insert is padded with the K562 model's construct suffix
    (36bp promoter + 15bp barcode) to 281bp before scoring.
  - ISM_raw[pos, base] = pred(mut) - pred(self)
  - ISM_centered = ISM_raw mean-centered across the 4 channels per position.

Output: results/single_seq/ism/{SEQ_ID}.h5
  ism_raw       (N, 200, 4)  float32
  ism_centered  (N, 200, 4)  float32
  predictions   (N,)         float32  prediction of each of the N base seqs

Usage:
    python SEAM_ism.py [--seq-idx 3609] [--start 0 --end 100000] [--batch-size 1024]
"""

import argparse
import csv
import gc
import sys
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
SEEDS_CSV = PROJ_ROOT / "single_seq/seq_subsets/seam_seeds.csv"
EI_DIR = PROJ_ROOT.parent / "Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/eigen-interactions"
sys.path.insert(0, str(EI_DIR))

from ag_deeplift_patches import patch_alphagenome
from alphagenome_encoder_ft import EncoderMPRAModel

patch_alphagenome()

CELL_TYPE = "K562"


def seq_id_for_idx(seq_idx):
    """Resolve seq_idx -> filename-safe seq_id via seam_seeds.csv (trailing ':' stripped)."""
    with open(SEEDS_CSV) as f:
        for row in csv.DictReader(f):
            if int(row['seq_idx']) == seq_idx:
                return row['seq_id'].rstrip(':')
    raise SystemExit(f"seq_idx {seq_idx} not found in {SEEDS_CSV}")

ENHANCER_LEN = 230
VAR_START, VAR_END = 15, 215
L_VAR = VAR_END - VAR_START            # 200
N_MUT_PER_SEQ = L_VAR * 4              # 800

CKPT_PATH = '/grid/koo/home/shared/models/alphagenome_encoder/torch/mpra_K562/finetuned_encoder.pt'

MUT_LIB_DIR = PROJ_ROOT / "results/single_seq/mutagenesis_lib"
OUT_DIR = PROJ_ROOT / "results/single_seq/ism"

ALPHA_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


class TransposeWrapper(nn.Module):
    """EncoderMPRAModel expects (B, L, 4); our pipeline holds (B, 4, L)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x.transpose(-1, -2))
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out


def str_to_onehot_cf(seq_str):
    ohe = np.zeros((4, len(seq_str)), dtype=np.float32)
    for j, base in enumerate(seq_str):
        if base in ALPHA_MAP:
            ohe[ALPHA_MAP[base], j] = 1.0
    return ohe


def load_model_and_suffix(device='cuda'):
    ck = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    cc = ck['construct_config']
    promoter_seq, barcode_seq = cc['promoter_seq'], cc['barcode_seq']
    assert len(promoter_seq) == 36 and len(barcode_seq) == 15
    suffix_cf = str_to_onehot_cf(promoter_seq + barcode_seq)   # (4, 51)
    base = EncoderMPRAModel.from_checkpoint(CKPT_PATH, device=device).eval()
    return TransposeWrapper(base).to(device).eval(), suffix_cf, promoter_seq, barcode_seq


def make_ism_inputs(x_insert_cf):
    """Build the (801, 4, 230) tensor for one base sequence.

    Index 0 is the original; indices 1..800 are the 200*4 mutants ordered (pos, base).
    """
    out = np.broadcast_to(x_insert_cf, (N_MUT_PER_SEQ + 1, 4, ENHANCER_LEN)).copy()
    pos_idx = np.repeat(np.arange(L_VAR), 4)
    base_idx = np.tile(np.arange(4), L_VAR)
    cols = (VAR_START + pos_idx).astype(np.int64)
    mut_rows = 1 + np.arange(N_MUT_PER_SEQ)
    out[mut_rows[:, None], np.arange(4)[None, :], cols[:, None]] = 0.0
    out[mut_rows, base_idx, cols] = 1.0
    return out


@torch.no_grad()
def predict_batched(model, x_cf, batch_size):
    out = np.empty(len(x_cf), dtype=np.float32)
    for i in range(0, len(x_cf), batch_size):
        t = torch.from_numpy(x_cf[i:i+batch_size]).float().cuda(non_blocking=True)
        out[i:i+batch_size] = model(t).squeeze(-1).cpu().numpy()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-idx', type=int, default=3609,
                        help='seq_idx in seam_seeds.csv (default 3609 = peak27535_Reversed)')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    SEQ_ID = seq_id_for_idx(args.seq_idx)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mut_path = MUT_LIB_DIR / f"{SEQ_ID}.h5"
    assert mut_path.exists(), f"missing mutagenesis lib: {mut_path}"

    with h5py.File(mut_path, 'r') as f:
        seqs_nlc = f['sequences'][:]                # (N, 230, 4)
    N_total = len(seqs_nlc)
    end = args.end if args.end is not None else N_total
    assert 0 <= args.start < end <= N_total
    print(f"library: {mut_path.name}  N={N_total}  processing [{args.start}:{end}] = {end - args.start}")

    print(f"loading {CELL_TYPE} model...")
    model, suffix_cf, promoter, barcode = load_model_and_suffix()
    print(f"  promoter ({len(promoter)}bp): {promoter}")
    print(f"  barcode  ({len(barcode)}bp): {barcode}")

    suffix_tiled = np.broadcast_to(
        suffix_cf, (N_MUT_PER_SEQ + 1, 4, suffix_cf.shape[1])
    )

    n_slice = end - args.start
    ism_raw = np.zeros((n_slice, L_VAR, 4), dtype=np.float32)
    ism_centered = np.zeros((n_slice, L_VAR, 4), dtype=np.float32)
    predictions = np.zeros(n_slice, dtype=np.float32)

    t0 = time.time()
    log_every = max(1, n_slice // 50)
    for j in range(n_slice):
        n = args.start + j
        x_insert_cf = seqs_nlc[n].transpose(1, 0).astype(np.float32, copy=False)   # (4, 230)
        ins_inputs = make_ism_inputs(x_insert_cf)                                  # (801, 4, 230)
        x_all = np.concatenate([ins_inputs, suffix_tiled], axis=2)                 # (801, 4, 281)
        preds = predict_batched(model, x_all, batch_size=args.batch_size)
        self_pred = preds[0]
        ism = (preds[1:] - self_pred).reshape(L_VAR, 4)
        ism_raw[j] = ism
        ism_centered[j] = ism - ism.mean(axis=-1, keepdims=True)
        predictions[j] = self_pred

        if (j + 1) % log_every == 0 or j == 0:
            elapsed = (time.time() - t0) / 60
            rate = (j + 1) / max(elapsed * 60, 1e-6)
            eta = (n_slice - (j + 1)) / max(rate, 1e-6) / 60
            print(f"  [{j+1:>6}/{n_slice}]  {elapsed:.1f}min  {rate:.1f} seq/s  ETA {eta:.1f}min  "
                  f"(self_pred={self_pred:+.3f}, ism range [{ism.min():+.3f}, {ism.max():+.3f}])")

    out_name = f"{SEQ_ID}.h5"
    if args.start != 0 or end != N_total:
        out_name = f"{SEQ_ID}_chunk{args.start}-{end}.h5"
    out_path = OUT_DIR / out_name

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('ism_raw', data=ism_raw,
                         compression='gzip', compression_opts=4)
        f.create_dataset('ism_centered', data=ism_centered,
                         compression='gzip', compression_opts=4)
        f.create_dataset('predictions', data=predictions)
        f.attrs['seq_id'] = SEQ_ID
        f.attrs['cell_type'] = CELL_TYPE
        f.attrs['mut_lib'] = str(mut_path)
        f.attrs['start'] = int(args.start)
        f.attrs['end'] = int(end)
        f.attrs['n_total'] = int(N_total)
        f.attrs['var_start'] = VAR_START
        f.attrs['var_end'] = VAR_END
        f.attrs['alphabet'] = 'ACGT'
        f.attrs['format'] = 'NLC'
        f.attrs['method'] = 'ISM (saturation single-position mutagenesis); ism_centered is mean-centered across channels'

    elapsed = (time.time() - t0) / 60
    print(f"\nDone in {elapsed:.1f}min  ->  {out_path}")
    print(f"  ism_raw.shape={ism_raw.shape}  ism_centered.shape={ism_centered.shape}")

    del seqs_nlc, ism_raw, ism_centered, predictions
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
