"""Train the temporal convolutional emission model on the external benchmark (PyTorch, GPU).

    python scripts/train_tcn.py --folds 5          # out-of-fold log-probabilities for tuning
    python scripts/train_tcn.py --final            # train on every game, write models/tcn.json

Only external games are used. Folds are grouped by game and stratified by dataset, so a game's
out-of-fold predictions come from a network that never saw any degraded copy of it. Training is
seeded but uses fast, non-deterministic GPU kernels (cuDNN autotuning, bfloat16 autocast), so a
rerun gives slightly different weights. That is fine for the competition: the committed JSON
weights are what the pipeline uses, and inference on them is exactly reproducible on the CPU.

Needs `pip install -r requirements-train.txt` (torch); inference (src/agamemnon/tcn.py) does not.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agamemnon import deadball, decoder, tcn  # noqa: E402
from agamemnon.external import CACHE_DIR, bench  # noqa: E402

DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WIDTH = 96
CROP = 4096
MARGIN = 512            # frames at each crop edge that see zero padding; not in the loss
BATCH = 64
STEPS = 6000
LR = 2e-3
TAG_DROPOUT = 0.3       # share of crops trained with every tag channel zeroed
DROPOUT = 0.25          # inside residual blocks; training only, inference is unaffected
WEIGHT_DECAY = 1e-3
SWAP = True             # augment by exchanging the roles of the two teams
SEED = 0
OOF_DIR = os.path.join(CACHE_DIR, "tcn_oof")
SWAP_ANTISYMMETRIC = ("dist", "close", "touch", "next_touch", "spread", "ball_field",
                      "centroid_field", "near", "tag")
# Input groups that --drop can remove for ablations.
GROUPS = {
    "proximity": ["dist", "close", "log_d_home", "log_d_away", "near"],
    "touches": ["touch", "next_touch"],
    "shape": ["spread", "ball_field", "centroid_field"],
    "stoppage": ["speed_1s", "speed_3s", "log_speed_3s", "q75_3s", "ball_still", "ball_speed"],
    "tags": ["tag", "setpiece", "post_goal"],
}
DROP = []


def channel(name):
    if name in decoder.TEAM_COLUMNS:
        return decoder.TEAM_COLUMNS.index(name) - 1
    return len(decoder.TEAM_COLUMNS) - 1 + deadball.DEAD_COLUMNS.index(name) - 1


def drop_channels(X):
    for group in DROP:
        for name in GROUPS[group]:
            X[:, channel(name)] = 0.0
    return X


TAG_CHANNELS = (decoder.TEAM_COLUMNS.index("tag") - 1,
                len(decoder.TEAM_COLUMNS) - 1 + deadball.DEAD_COLUMNS.index("setpiece") - 1,
                len(decoder.TEAM_COLUMNS) - 1 + deadball.DEAD_COLUMNS.index("post_goal") - 1)


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.inp = nn.Conv1d(n_in, WIDTH, 1)
        self.blocks = nn.ModuleList()
        for d in DILATIONS:
            block = nn.Module()
            block.conv = nn.Conv1d(WIDTH, WIDTH, 3, dilation=d, padding=d)
            block.mix = nn.Conv1d(WIDTH, WIDTH, 1)
            block.drop = nn.Dropout(DROPOUT)
            self.blocks.append(block)
        self.out = nn.Conv1d(WIDTH, 3, 1)

    def forward(self, x):
        h = self.inp(x)
        for block in self.blocks:
            h = h + block.mix(block.drop(torch.relu(block.conv(h))))
        return self.out(torch.relu(h))


def export(net, mean, std):
    params = {"mean": mean.tolist(), "std": std.tolist()}
    sd = {k: v.detach().cpu().double().numpy() for k, v in net.state_dict().items()}
    params["inp.w"], params["inp.b"] = sd["inp.weight"].tolist(), sd["inp.bias"].tolist()
    for i in range(len(DILATIONS)):
        for part in ("conv", "mix"):
            params["b{}.{}.w".format(i, part)] = sd["blocks.{}.{}.weight".format(i, part)].tolist()
            params["b{}.{}.b".format(i, part)] = sd["blocks.{}.{}.bias".format(i, part)].tolist()
    params["out.w"], params["out.b"] = sd["out.weight"].tolist(), sd["out.bias"].tolist()
    return {"dilations": list(DILATIONS), "params": params}


def load_data(keys):
    data = []
    for key in keys:
        cell = bench.Cell(*key)
        X = drop_channels(tcn.inputs(cell.Xt, cell.Xd).astype(np.float32))
        for start, stop in cell.periods:
            data.append((key, X[start:stop], cell.gt[start:stop].astype(np.int64)))
    return data


def train(train_keys, device, log_every=500):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    data = load_data(train_keys)
    stacked = np.concatenate([x for _, x, _ in data])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-3
    # The whole training set lives on the GPU, normalised, so a batch is one gather and the GPU
    # never waits for the CPU to cut crops.
    X = torch.tensor((stacked - mean) / std, dtype=torch.float32, device=device)
    del stacked
    Y = torch.tensor(np.concatenate([np.where(y < 3, y, -1) for _, _, y in data]),
                     dtype=torch.int64, device=device)
    offsets = np.cumsum([0] + [len(x) for _, x, _ in data])
    lengths = np.diff(offsets)
    # Each dataset contributes equally, whatever its number of games.
    datasets = sorted({k[0] for k, _, _ in data})
    by_ds = {d: np.array([i for i, (k, _, _) in enumerate(data) if k[0] == d and
                          lengths[i] > CROP]) for d in datasets}
    del data
    tag_mask = torch.ones(X.shape[1], device=device)
    tag_mask[list(TAG_CHANNELS)] = 0.0
    tag_zero = torch.tensor(-mean / std, dtype=torch.float32, device=device)  # raw value 0
    window = torch.arange(CROP, device=device)
    in_loss = torch.zeros(CROP, dtype=torch.bool, device=device)
    in_loss[MARGIN:CROP - MARGIN] = True

    net = Net(len(mean)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Team swap: every home-vs-away feature is antisymmetric (it changes sign when the teams
    # trade places), the two raw distances trade channels, dead-ball features do not change.
    sign = torch.ones(X.shape[1], device=device)
    for name in SWAP_ANTISYMMETRIC:
        sign[decoder.TEAM_COLUMNS.index(name) - 1] = -1.0
    perm = torch.arange(X.shape[1], device=device)
    i, j = decoder.TEAM_COLUMNS.index("log_d_home") - 1, decoder.TEAM_COLUMNS.index("log_d_away") - 1
    perm[i], perm[j] = j, i
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)
    label_swap = torch.tensor([1, 0, 2], device=device)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    t0 = time.time()
    for step in range(STEPS):
        ds = rng.integers(len(datasets), size=BATCH)
        piece = np.array([by_ds[datasets[d]][rng.integers(len(by_ds[datasets[d]]))] for d in ds])
        starts = offsets[piece] + rng.integers(0, lengths[piece] - CROP + 1)
        idx = torch.tensor(starts, device=device)[:, None] + window
        xb = X[idx]                                   # (B, CROP, C)
        drop = torch.tensor(rng.random(BATCH) < TAG_DROPOUT, device=device)[:, None, None]
        xb = torch.where(drop, xb * tag_mask + tag_zero * (1 - tag_mask), xb).transpose(1, 2)
        yb = torch.where(in_loss, Y[idx], torch.full_like(Y[idx], -1))
        if SWAP:
            swap = torch.tensor(rng.random(BATCH) < 0.5, device=device)
            raw = xb.transpose(1, 2) * std_t + mean_t          # back to raw units
            swapped = ((raw * sign)[:, :, perm] - mean_t) / std_t
            xb = torch.where(swap[:, None, None], swapped.transpose(1, 2), xb)
            yb = torch.where(swap[:, None] & (yb >= 0), label_swap[yb.clamp(min=0)], yb)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = net(xb)
        loss = loss_fn(logits.float(), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0:
            print("  step {} loss {:.4f} ({:.0f}s)".format(step + 1, loss.item(), time.time() - t0),
                  flush=True)
    return export(net, mean, std)


def folds_of(games, k):
    """Grouped, dataset-stratified folds: games of each dataset dealt round-robin."""
    out = [[] for _ in range(k)]
    i = 0
    for ds in sorted({d for d, _ in games}):
        for g in [g for g in games if g[0] == ds]:
            out[i % k].append(g)
            i += 1
    return out


def main():
    global WIDTH, STEPS, DILATIONS, OOF_DIR, DROPOUT, WEIGHT_DECAY, SWAP, DROP
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", default=tcn.WEIGHTS)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--depth", type=int, default=len(DILATIONS),
                        help="residual blocks; dilations double from 1")
    parser.add_argument("--name", default="tcn_oof", help="out-of-fold folder under the cache")
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--only-fold", type=int, default=None)
    parser.add_argument("--drop", nargs="*", default=[], choices=sorted(GROUPS),
                        help="ablation: zero these input groups in training and inference")
    args = parser.parse_args()
    WIDTH, STEPS = args.width, args.steps
    DILATIONS = tuple(2 ** i for i in range(args.depth))
    OOF_DIR = os.path.join(CACHE_DIR, args.name)
    DROPOUT, WEIGHT_DECAY, SWAP = args.dropout, args.weight_decay, not args.no_swap
    DROP = list(args.drop)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    games = bench.all_games()
    keys = lambda gs: [(d, m, lv) for d, m in gs for lv in bench.LEVELS]
    if args.final:
        model = train(keys(games), device)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(model, handle, separators=(",", ":"))
        print("wrote", args.out)
        return

    os.makedirs(OOF_DIR, exist_ok=True)
    for f, held in enumerate(folds_of(games, args.folds)):
        if args.only_fold is not None and f != args.only_fold:
            continue
        print("fold {} holds out {}".format(f, ["{}-{}".format(*g) for g in held]), flush=True)
        model = train(keys([g for g in games if g not in held]), device)
        with open(os.path.join(OOF_DIR, "fold{}.json".format(f)), "w") as handle:
            json.dump(model, handle, separators=(",", ":"))
        net = tcn.load(os.path.join(OOF_DIR, "fold{}.json".format(f)))
        for key in keys(held):
            cell = bench.Cell(*key)
            logp = tcn.log_probs(net, drop_channels(tcn.inputs(cell.Xt, cell.Xd)), cell.periods)
            notag = drop_channels(tcn.inputs(*cell.designs(tags=False)))
            logp_nt = tcn.log_probs(net, notag, cell.periods)
            np.savez_compressed(os.path.join(OOF_DIR, "{}-{}@{}.npz".format(*key)),
                                logp=logp.astype(np.float32), logp_notags=logp_nt.astype(np.float32))


if __name__ == "__main__":
    main()
