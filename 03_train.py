import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import snntorch as snn
from snntorch import surrogate


@dataclass
class TrainConfig:
    features_dir: Path = Path(".").resolve() / "features"
    out_dir: Path = Path(".").resolve() / "results"

    seed: int = 42
    device: str = "cpu"

    epochs_ann: int = 60
    epochs_snn: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4

    time_steps: int = 25
    beta: float = 0.9
    dropout: float = 0.1

    patience: int = 12

    save_plots: bool = True


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    X_train = d["X_train"].astype(np.float32)
    y_train = d["y_train"].astype(np.int64)
    X_test = d["X_test"].astype(np.float32)
    y_test = d["y_test"].astype(np.int64)

    rep = str(d["rep"]) if "rep" in d else "UNKNOWN"
    norm_mode = str(d["norm_mode"]) if "norm_mode" in d else "UNKNOWN"
    dims = int(d["dims"]) if "dims" in d else X_train.shape[1]

    return X_train, y_train, X_test, y_test, rep, norm_mode, dims


def confusion_matrix_np(y_true, y_pred, n_classes=2):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion(cm, labels, title, out_path: Path):
    fig = plt.figure(figsize=(5, 4))
    ax = plt.gca()
    im = ax.imshow(cm, aspect="auto", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_curves(train_loss, test_acc, title, out_path: Path):
    fig = plt.figure(figsize=(9, 4))
    ax = plt.gca()
    ax.plot(train_loss, label="train_loss")
    ax.plot(test_acc, label="test_acc")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def minmax_fit(X):
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    span = (mx - mn) + 1e-8
    return mn, span


def minmax_apply(X, mn, span):
    Z = (X - mn) / span
    return np.clip(Z, 0.0, 1.0)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_layers, out_dim=2, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleSNN(nn.Module):
    def __init__(self, in_dim, hidden_layers, out_dim=2, beta=0.9, dropout=0.1):
        super().__init__()
        spike_grad = surrogate.atan()

        self.fcs = nn.ModuleList()
        self.lifs = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev = in_dim
        for h in hidden_layers:
            self.fcs.append(nn.Linear(prev, h))
            self.lifs.append(snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=1.0))
            self.dropouts.append(nn.Dropout(dropout))
            prev = h

        self.fc_out = nn.Linear(prev, out_dim)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=1.0)

    def forward(self, x_spk):
        # x_spk: [B, F, T]
        mems = [lif.init_leaky() for lif in self.lifs]
        mem_out = self.lif_out.init_leaky()

        out_acc = 0.0
        spike_count = 0.0

        T = x_spk.size(2)
        for t in range(T):
            cur = x_spk[:, :, t]

            for i in range(len(self.fcs)):
                z = self.fcs[i](cur)
                spk, mems[i] = self.lifs[i](z, mems[i])
                spk = self.dropouts[i](spk)
                spike_count = spike_count + spk.detach().sum()
                cur = spk

            z_out = self.fc_out(cur)
            spk_out, mem_out = self.lif_out(z_out, mem_out)
            out_acc = out_acc + mem_out

        out = out_acc / T
        return out, spike_count


def make_batches(X, y, batch_size, rng):
    idx = np.arange(len(X))
    rng.shuffle(idx)
    for s in range(0, len(X), batch_size):
        b = idx[s:s + batch_size]
        yield X[b], y[b]


def encode_poisson(rates, T, device):
    # rates: [B, F] in [0,1]
    B, F = rates.shape
    r = torch.from_numpy(rates).to(device)
    rand = torch.rand((B, F, T), device=device)
    spk = (rand < r.unsqueeze(-1)).float()
    return spk


def train_ann(X_train, y_train, X_test, y_test, hidden_layers, cfg: TrainConfig, tag: str):
    device = torch.device(cfg.device)
    model = MLP(X_train.shape[1], hidden_layers, out_dim=2, dropout=cfg.dropout).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()

    rng = np.random.default_rng(cfg.seed)
    best_acc = -1.0
    best_state = None
    wait = 0

    train_loss_hist = []
    test_acc_hist = []

    Xte = torch.from_numpy(X_test).to(device)
    yte = torch.from_numpy(y_test).to(device)

    for epoch in range(cfg.epochs_ann):
        model.train()
        epoch_loss = 0.0
        n_seen = 0

        for xb, yb in make_batches(X_train, y_train, cfg.batch_size, rng):
            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)

            opt.zero_grad()
            logits = model(xb_t)
            loss = crit(logits, yb_t)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)

        epoch_loss = epoch_loss / max(1, n_seen)

        model.eval()
        with torch.no_grad():
            logits = model(Xte)
            pred = logits.argmax(1)
            acc = (pred == yte).float().mean().item()

        train_loss_hist.append(epoch_loss)
        test_acc_hist.append(acc)

        if acc > best_acc + 1e-6:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        pred = logits.argmax(1).cpu().numpy()

    return model, best_acc, pred, train_loss_hist, test_acc_hist


def train_snn(X_train, y_train, X_test, y_test, hidden_layers, cfg: TrainConfig, tag: str):
    device = torch.device(cfg.device)
    model = SimpleSNN(X_train.shape[1], hidden_layers, out_dim=2, beta=cfg.beta, dropout=cfg.dropout).to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()

    rng = np.random.default_rng(cfg.seed)

    mn, span = minmax_fit(X_train)
    X_train_rates = minmax_apply(X_train, mn, span)
    X_test_rates = minmax_apply(X_test, mn, span)

    yte = torch.from_numpy(y_test).to(device)

    best_acc = -1.0
    best_state = None
    wait = 0

    train_loss_hist = []
    test_acc_hist = []

    for epoch in range(cfg.epochs_snn):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        spike_sum_epoch = 0.0

        for xb, yb in make_batches(X_train_rates, y_train, cfg.batch_size, rng):
            x_spk = encode_poisson(xb, cfg.time_steps, device)
            yb_t = torch.from_numpy(yb).to(device)

            opt.zero_grad()
            out, spk_count = model(x_spk)
            loss = crit(out, yb_t)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)
            spike_sum_epoch += float(spk_count.item())

        epoch_loss = epoch_loss / max(1, n_seen)

        model.eval()
        with torch.no_grad():
            xte_spk = encode_poisson(X_test_rates, cfg.time_steps, device)
            out, spk_count = model(xte_spk)
            pred = out.argmax(1)
            acc = (pred == yte).float().mean().item()

        train_loss_hist.append(epoch_loss)
        test_acc_hist.append(acc)

        if acc > best_acc + 1e-6:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xte_spk = encode_poisson(X_test_rates, cfg.time_steps, device)
        out, spk_count = model(xte_spk)
        pred = out.argmax(1).cpu().numpy()
        avg_spikes = float(spk_count.item()) / (len(X_test_rates) * cfg.time_steps)

    return model, best_acc, pred, train_loss_hist, test_acc_hist, avg_spikes


def save_csv(path: Path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_heatmap_accuracy(rows, out_dir: Path, title_suffix: str):
    # heatmaps per norm+rep over models and dims
    # dims: 16/32; models: ANN/SNN
    norms = sorted(set(r["norm"] for r in rows))
    reps = sorted(set(r["rep"] for r in rows))
    dims = sorted(set(int(r["dims"]) for r in rows))
    models = sorted(set(r["model"] for r in rows))

    for norm in norms:
        for rep in reps:
            mat = np.full((len(models), len(dims)), np.nan, dtype=np.float32)
            for i, m in enumerate(models):
                for j, d in enumerate(dims):
                    xs = [r for r in rows if r["norm"] == norm and r["rep"] == rep and r["model"] == m and int(r["dims"]) == d]
                    if len(xs) > 0:
                        mat[i, j] = float(xs[0]["test_acc"])

            fig = plt.figure(figsize=(6, 3.6))
            ax = plt.gca()
            im = ax.imshow(mat, aspect="auto", origin="upper")

            ax.set_title(f"Accuracy | {rep} | {norm} | {title_suffix}")
            ax.set_xlabel("Dims")
            ax.set_ylabel("Model")
            ax.set_xticks(range(len(dims)))
            ax.set_yticks(range(len(models)))
            ax.set_xticklabels([str(d) for d in dims])
            ax.set_yticklabels(models)

            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]*100:.1f}%", ha="center", va="center")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            fig.savefig(out_dir / f"heatmap_acc_{rep}_{norm}.png", dpi=160)
            plt.close(fig)


def main():
    cfg = TrainConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "runs").mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)

    meta_path = cfg.features_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    print(f"Meta entries: {len(meta)}")
    print(f"Device: {cfg.device}")
    print(f"Time steps (SNN): {cfg.time_steps}")

    # Easy depth/width experiments later:
    # e.g., add "d16_2L": [8, 8] or "d32_2L": [16,16]
    hidden_map = {
        16: [16],
        32: [32],
    }

    rows = []
    start_all = time.time()

    for entry in meta:
        npz_path = Path(entry["npz"])
        X_train, y_train, X_test, y_test, rep, norm_mode, dims = load_npz(npz_path)

        hidden_layers = hidden_map.get(dims, [dims])

        tag_base = f"{rep}_d{dims}_{norm_mode}"
        run_dir = cfg.out_dir / "runs" / tag_base
        run_dir.mkdir(parents=True, exist_ok=True)

        # ANN
        t0 = time.time()
        ann_model, ann_best, ann_pred, ann_loss, ann_acc = train_ann(
            X_train, y_train, X_test, y_test, hidden_layers, cfg, tag_base
        )
        ann_time = time.time() - t0

        ann_cm = confusion_matrix_np(y_test, ann_pred, n_classes=2)
        if cfg.save_plots:
            plot_confusion(ann_cm, ["hayir", "evet"], f"ANN | {tag_base}", run_dir / "cm_ann.png")
            plot_curves(ann_loss, ann_acc, f"ANN curves | {tag_base}", run_dir / "curves_ann.png")

        rows.append({
            "rep": rep,
            "dims": dims,
            "norm": norm_mode,
            "model": "ANN",
            "hidden": str(hidden_layers),
            "test_acc": float(ann_best),
            "time_sec": float(ann_time),
            "avg_spikes": "",
        })

        # SNN
        t0 = time.time()
        snn_model, snn_best, snn_pred, snn_loss, snn_acc, avg_spikes = train_snn(
            X_train, y_train, X_test, y_test, hidden_layers, cfg, tag_base
        )
        snn_time = time.time() - t0

        snn_cm = confusion_matrix_np(y_test, snn_pred, n_classes=2)
        if cfg.save_plots:
            plot_confusion(snn_cm, ["hayir", "evet"], f"SNN | {tag_base}", run_dir / "cm_snn.png")
            plot_curves(snn_loss, snn_acc, f"SNN curves | {tag_base}", run_dir / "curves_snn.png")

        rows.append({
            "rep": rep,
            "dims": dims,
            "norm": norm_mode,
            "model": "SNN",
            "hidden": str(hidden_layers),
            "test_acc": float(snn_best),
            "time_sec": float(snn_time),
            "avg_spikes": float(avg_spikes),
        })

        print(f"Done: {tag_base}")

    save_csv(cfg.out_dir / "results.csv", rows)
    plot_heatmap_accuracy(rows, cfg.out_dir, f"T={cfg.time_steps}")

    total_time = time.time() - start_all
    print("All runs done.")
    print(f"Saved: {cfg.out_dir / 'results.csv'}")
    print(f"Plots: {cfg.out_dir}")
    print(f"Total time (sec): {total_time:.1f}")


if __name__ == "__main__":
    main()
