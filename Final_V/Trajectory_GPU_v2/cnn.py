# =============================================================================
# cnn.py -- 1D CNN on the closest-approach labels, all networks trained at the
#           same time on the GPU; then scored on the held-out test satellites
# =============================================================================
#
# Same network (DopplerCNN), inputs, two-stage early stopping and batched GPU
# training as Noise_Testing_GPU/cnn_trainer_v3.py, with one network per label:
#   linear labels: 1 output, target standardised, MSE loss
#   binary labels: 1 output (a logit), binary cross-entropy loss; predicted
#                  class = probability > 0.5
# All 8 networks train together as one batched network (grouped convolutions,
# batched matrix products); each has its own weights, loss, optimizer state,
# early stopping and stopping epoch. See the GPU notes in
# Noise_Testing_GPU/cnn_trainer_v3.py.
#
# WRITES (output/)
#   models/cnn/cnn__<target>.pt   weights + the numbers needed to use them
#   test_predictions_cnn.csv      sat_idx, target, model, y_true, y_pred (+ y_prob)
#   test_results_cnn.csv          one row per target (common.score)
#   cnn_COMPLETE.txt              written last, so run_all.py can tell it finished
# =============================================================================

import os
import time

# Needed for deterministic matrix products on CUDA; must be set before torch loads.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from common import ALL_TARGETS, kind, load_table, print_results, score
from splits import RANDOM_STATE, make_or_load_split

RR_SCALE = 8000.0          # [m/s] fixed input scale for the range-rate channel
EARLY_STOP_FRACTION = 0.2  # share of the training data held back in stage 1
MAX_EPOCHS = 2000
PATIENCE = 150             # epochs without improvement before stopping
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

MODEL_NAME = "cnn"
MODEL_DIR = "output/models/cnn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_CHUNK = 1024          # satellites per forward pass when scoring the early-stopping subset
PROGRESS_EVERY = 100       # print a progress line every this many epochs

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False         # full float32, as on the CPU
torch.backends.cuda.matmul.allow_tf32 = False


def device_name():
    if DEVICE.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(DEVICE)})"
    return "cpu (no GPU found -- this will be slow)"


def to_input(sub):
    """(N, 2, L) float32 array: channel 0 scaled range rate, channel 1 mask."""
    rr_cols = sorted(c for c in sub.columns if c.startswith("rr_c_"))
    mask_cols = sorted(c for c in sub.columns if c.startswith("mask_"))
    rr = sub[rr_cols].to_numpy(dtype=float) / RR_SCALE
    mask = sub[mask_cols].to_numpy(dtype=float)
    return np.stack([rr, mask], axis=1).astype(np.float32)


# -----------------------------------------------------------------------------
# The network, and M of them side by side
# -----------------------------------------------------------------------------
class DopplerCNN(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(32 * (seq_len // 2), 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.features(x))


def _layers(net):
    """The 5 layers of a DopplerCNN that hold weights."""
    return net.features[0], net.features[2], net.features[5], net.head[2], net.head[4]


class BatchedDopplerCNN(nn.Module):
    """M independent DopplerCNNs in one module: input (M, B, 2, L), output (M, B).
    Grouped convolutions and batched matrix products, so no weight or
    activation is shared between networks."""
    def __init__(self, nets):
        super().__init__()
        self.M = len(nets)
        layers = [_layers(n) for n in nets]

        def cat(i, attr):
            return nn.Parameter(torch.cat([getattr(l[i], attr).detach() for l in layers]))

        def stack(i, attr):
            return nn.Parameter(torch.stack([getattr(l[i], attr).detach() for l in layers]))

        self.c1w, self.c1b = cat(0, "weight"), cat(0, "bias")
        self.c2w, self.c2b = cat(1, "weight"), cat(1, "bias")
        self.c3w, self.c3b = cat(2, "weight"), cat(2, "bias")
        self.l1w, self.l1b = stack(3, "weight"), stack(3, "bias")   # (M, 64, F), (M, 64)
        self.l2w, self.l2b = stack(4, "weight"), stack(4, "bias")   # (M, 1, 64), (M, 1)

    def forward(self, x):
        M, B = x.shape[0], x.shape[1]
        h = x.transpose(0, 1).reshape(B, M * 2, -1)
        h = F.relu(F.conv1d(h, self.c1w, self.c1b, padding=2, groups=M))
        h = F.relu(F.conv1d(h, self.c2w, self.c2b, padding=2, groups=M))
        h = F.max_pool1d(h, kernel_size=2)
        h = F.relu(F.conv1d(h, self.c3w, self.c3b, padding=1, groups=M))
        h = h.reshape(B, M, -1).transpose(0, 1)                     # (M, B, 32*L//2), same order as Flatten
        h = F.dropout(h, DROPOUT, training=self.training)
        h = F.relu(torch.baddbmm(self.l1b.unsqueeze(1), h, self.l1w.transpose(1, 2)))
        return torch.baddbmm(self.l2b.unsqueeze(1), h, self.l2w.transpose(1, 2))[..., 0]

    def unpack(self, state, m, seq_len):
        """Network m of a saved state as a plain DopplerCNN on the CPU."""
        net = DopplerCNN(seq_len)
        c1, c2, c3, l1, l2 = _layers(net)
        with torch.no_grad():
            for layer, w, b in ((c1, "c1w", "c1b"), (c2, "c2w", "c2b"), (c3, "c3w", "c3b")):
                layer.weight.copy_(state[w].view(self.M, *layer.weight.shape)[m])
                layer.bias.copy_(state[b].view(self.M, -1)[m])
            for layer, w, b in ((l1, "l1w", "l1b"), (l2, "l2w", "l2b")):
                layer.weight.copy_(state[w][m])
                layer.bias.copy_(state[b][m])
        return net


def _keep_where(state, net, take):
    """state[name][m] = current weights of network m wherever take[m] is True."""
    M = net.M
    take = take.to(DEVICE).view(M, 1)
    with torch.no_grad():
        for name, p in net.named_parameters():
            s = state[name]
            s.copy_(torch.where(take, p.view(M, -1), s.view(M, -1)).view_as(s))


def _loss(out, y, is_bin):
    """(M, B) per-sample losses: binary cross-entropy for binary networks, squared error otherwise."""
    return torch.where(is_bin, F.binary_cross_entropy_with_logits(out, y, reduction="none"), (out - y) ** 2)


@torch.no_grad()
def _mean_loss(net, X, Y, is_bin):
    net.eval()
    total = torch.zeros(net.M, device=DEVICE)
    for start in range(0, X.shape[1], EVAL_CHUNK):
        total += _loss(net(X[:, start:start + EVAL_CHUNK]), Y[:, start:start + EVAL_CHUNK], is_bin).sum(dim=1)
    return (total / X.shape[1]).cpu()


def _train_many(X, Y, is_bin, seed, n_epochs, X_es=None, Y_es=None, label=""):
    """
    Train M fresh DopplerCNNs at once. X (M, n, 2, L), Y (M, n) on DEVICE,
    is_bin (M, 1) bool. n_epochs: one per network. With an early-stopping
    subset: stop each network after PATIENCE epochs without improvement and
    return (best weights, best epochs). Without: take network m's weights after
    n_epochs[m] epochs and return (those weights, n_epochs).
    """
    M, n, L = X.shape[0], X.shape[1], X.shape[3]
    nets = []
    for _ in range(M):                         # every network starts from the same seeded weights
        torch.manual_seed(seed)
        nets.append(DopplerCNN(L))
    torch.manual_seed(seed)                    # dropout stream
    gen = torch.Generator().manual_seed(seed)  # shuffle stream, CPU

    model = BatchedDopplerCNN(nets).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    state = {name: p.detach().clone() for name, p in model.named_parameters()}
    n_epochs_t = torch.tensor(n_epochs)
    early_stopping = X_es is not None
    if early_stopping:
        best_loss = torch.full((M,), float("inf"))
        best_epoch = torch.zeros(M, dtype=torch.long)
        since_best = torch.zeros(M, dtype=torch.long)
        active = torch.ones(M, dtype=torch.bool)

    t0 = time.time()
    for epoch in range(1, max(n_epochs) + 1):
        model.train()
        order = torch.randperm(n, generator=gen).to(DEVICE)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            loss = _loss(model(X[:, idx]), Y[:, idx], is_bin).mean(dim=1).sum()   # sum of each network's own loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        if early_stopping:
            es_loss = _mean_loss(model, X_es, Y_es, is_bin)
            improved = active & (es_loss < best_loss)
            best_loss = torch.where(improved, es_loss, best_loss)
            best_epoch[improved] = epoch
            since_best = torch.where(improved, 0, since_best + 1)
            _keep_where(state, model, improved)
            active &= since_best < PATIENCE
            still = int(active.sum())
        else:
            done_now = n_epochs_t == epoch
            if done_now.any():
                _keep_where(state, model, done_now)
            still = int((n_epochs_t > epoch).sum())

        if epoch % PROGRESS_EVERY == 0 or still == 0:
            print(f"    {label} epoch {epoch:>4}: {still} of {M} networks still training "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        if still == 0:
            break

    weights = [model.unpack(state, m, L) for m in range(M)]
    return weights, ([int(e) for e in best_epoch] if early_stopping else list(n_epochs))


# -----------------------------------------------------------------------------
# One model per label; fit_many fits several at once
# -----------------------------------------------------------------------------
class CNNModel:
    """fit / predict for one label. Linear: target standardised with the mean/std
    of the data being fit. Binary: 0/1 target, predict() returns the class and
    predict_proba() the probability of 1."""
    def __init__(self, target, seed=RANDOM_STATE):
        self.target, self.binary, self.seed = target, kind(target) == "binary", seed

    def _encode(self, y):
        y = np.asarray(y, dtype=float)
        return (y if self.binary else (y - self.y_mean_) / self.y_std_).astype(np.float32)

    def _raw(self, X):
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.from_numpy(X))[:, 0].numpy()

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-self._raw(X).astype(float)))

    def predict(self, X):
        if self.binary:
            return (self.predict_proba(X) > 0.5).astype(int)
        return self._raw(X) * self.y_std_ + self.y_mean_


def fit_many(models, X, ys):
    """Two-stage fit of several CNNModels on the same inputs X (N, 2, L), at once:
    stage 1 finds each network's epoch count on an early-stopping subset,
    stage 2 retrains on all of X for that many epochs."""
    seed = models[0].seed
    n = len(X)
    ys = [np.asarray(y, dtype=float) for y in ys]
    is_bin = torch.tensor([m.binary for m in models], device=DEVICE).view(-1, 1)
    print(f"  training {len(models)} networks together on {device_name()}, {n} satellites each", flush=True)

    perm = np.random.RandomState(seed).permutation(n)
    n_es = int(round(EARLY_STOP_FRACTION * n))
    es_idx, fit_idx = perm[:n_es].astype(np.int64), perm[n_es:].astype(np.int64)
    Xd = torch.from_numpy(X).to(DEVICE)

    def tile(idx):                             # (M, len(idx), 2, L): the same satellites for every network
        return Xd[torch.from_numpy(idx).to(DEVICE)].unsqueeze(0).expand(len(models), -1, -1, -1)

    # --- stage 1: each network's epoch count on an early-stopping subset ---
    for m, y in zip(models, ys):
        m.y_mean_, m.y_std_ = y[fit_idx].mean(), y[fit_idx].std()
    Y_fit = torch.from_numpy(np.stack([m._encode(y[fit_idx]) for m, y in zip(models, ys)])).to(DEVICE)
    Y_es = torch.from_numpy(np.stack([m._encode(y[es_idx]) for m, y in zip(models, ys)])).to(DEVICE)
    _, best = _train_many(tile(fit_idx), Y_fit, is_bin, seed, [MAX_EPOCHS] * len(models),
                          X_es=tile(es_idx), Y_es=Y_es, label="stage 1 (early stopping)")
    for m, e in zip(models, best):
        m.best_epoch_ = max(e, 1)

    # --- stage 2: retrain on ALL the training data for that many epochs ---
    for m, y in zip(models, ys):
        m.y_mean_, m.y_std_ = y.mean(), y.std()
    Y_all = torch.from_numpy(np.stack([m._encode(y) for m, y in zip(models, ys)])).to(DEVICE)
    weights, _ = _train_many(tile(np.arange(n, dtype=np.int64)), Y_all, is_bin, seed,
                             [m.best_epoch_ for m in models], label="stage 2 (all training data)")
    for m, w in zip(models, weights):
        m.model_ = w
    return models


def save_model(model, path, seq_len):
    torch.save({"state_dict": model.model_.state_dict(), "seq_len": int(seq_len), "target": model.target,
                "seed": int(model.seed), "y_mean": float(model.y_mean_), "y_std": float(model.y_std_),
                "best_epoch": int(model.best_epoch_)}, path)


def load_model(path):
    d = torch.load(path, weights_only=True, map_location="cpu")
    model = CNNModel(d["target"], seed=d["seed"])
    model.model_ = DopplerCNN(d["seq_len"])
    model.model_.load_state_dict(d["state_dict"])
    model.y_mean_, model.y_std_, model.best_epoch_ = d["y_mean"], d["y_std"], d["best_epoch"]
    return model


if __name__ == "__main__":
    df, _ = load_table()
    split = make_or_load_split(df["sat_idx"])
    train_ids = set(split.loc[split["split"] == "train", "sat_idx"])
    train, test = df[df["sat_idx"].isin(train_ids)], df[~df["sat_idx"].isin(train_ids)]
    L = sum(c.startswith("rr_c_") for c in df.columns)
    print(f"{len(train)} train / {len(test)} test satellites, input (2, {L}), device {device_name()}\n")

    t0 = time.time()
    models = fit_many([CNNModel(t) for t in ALL_TARGETS], to_input(train), [train[t] for t in ALL_TARGETS])
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_test = to_input(test)
    results, preds = [], []
    for m in models:
        save_model(m, os.path.join(MODEL_DIR, f"{MODEL_NAME}__{m.target}.pt"), L)
        y_pred = m.predict(X_test)
        p = pd.DataFrame({"sat_idx": test["sat_idx"].to_numpy(), "target": m.target, "model": MODEL_NAME,
                          "y_true": test[m.target].to_numpy(), "y_pred": y_pred})
        if m.binary:
            p["y_prob"] = m.predict_proba(X_test)
        preds.append(p)
        results.append({"target": m.target, "model": MODEL_NAME, "n_train": len(train), "n_test": len(test),
                        "best_epoch": m.best_epoch_, **score(test[m.target], y_pred, m.target)})

    print(f"\nTraining time: {(time.time() - t0) / 60:.1f} min. Stopping epochs: "
          + ", ".join(f"{m.target} {m.best_epoch_}" for m in models) + "\n")
    print_results(results)
    pd.concat(preds, ignore_index=True).to_csv("output/test_predictions_cnn.csv", index=False)
    pd.DataFrame(results).to_csv("output/test_results_cnn.csv", index=False)
    with open("output/cnn_COMPLETE.txt", "w") as f:
        f.write("done\n")
    print("\nSaved output/test_predictions_cnn.csv, output/test_results_cnn.csv and the models")
