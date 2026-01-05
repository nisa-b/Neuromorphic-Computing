import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt
import pywt
from scipy.stats import kurtosis, skew


@dataclass
class Config:
    data_dir: Path
    out_dir: Path

    sr: int = 16000
    dims_list: tuple = (16, 32)
    reps: tuple = ("T", "FOURIER", "WAVELET_STATS")  # default wavelet is stats
    normalize_modes: tuple = ("none", "zscore")

    wavelet_name: str = "db4"
    wavelet_level: int = 5

    n_preview: int = 6
    save_plots: bool = True
    overwrite: bool = True


def list_wavs(split_dir: Path):
    evet = sorted((split_dir / "evet").glob("*.wav"))
    hayir = sorted((split_dir / "hayir").glob("*.wav"))
    files = [(p, 1) for p in evet] + [(p, 0) for p in hayir]
    return files


def load_wav(path: Path, sr: int):
    x, _ = librosa.load(str(path), sr=sr, mono=True)
    return x.astype(np.float32)


def resize_vector(v: np.ndarray, dims: int):
    v = np.asarray(v, dtype=np.float32).flatten()
    if len(v) == dims:
        return v
    if len(v) < dims:
        return np.pad(v, (0, dims - len(v)), mode="constant")
    idx = np.linspace(0, len(v) - 1, num=dims)
    return np.interp(idx, np.arange(len(v)), v).astype(np.float32)


def feat_time(x: np.ndarray, dims: int):
    return resize_vector(x, dims)


def feat_fourier(x: np.ndarray, dims: int):
    X = np.fft.rfft(x)
    mag = np.abs(X).astype(np.float32)
    mag = np.log1p(mag)
    return resize_vector(mag, dims)


def shannon_entropy_from_energy(coeff: np.ndarray):
    e = coeff.astype(np.float64) ** 2
    s = e.sum() + 1e-12
    p = e / s
    return float(-(p * np.log(p + 1e-12)).sum())


def feat_wavelet_stats(x: np.ndarray, dims: int, wavelet_name: str, level: int):
    coeffs = pywt.wavedec(x, wavelet_name, level=level)

    feats = []
    for c in coeffs:
        c = c.astype(np.float64)

        mu = c.mean()
        sd = c.std() + 1e-12
        mx = c.max()
        mn = c.min()
        energy = float((c ** 2).sum())
        kur = float(kurtosis(c, fisher=True, bias=False))
        skw = float(skew(c, bias=False))
        ent = shannon_entropy_from_energy(c)

        feats.extend([mu, sd, mx, mn, energy, kur, skw, ent])

    feats = np.asarray(feats, dtype=np.float32)

    feats = np.sign(feats) * np.log1p(np.abs(feats))
    return resize_vector(feats, dims)


def extract_features(files, cfg: Config, rep: str, dims: int):
    X = np.zeros((len(files), dims), dtype=np.float32)
    y = np.zeros((len(files),), dtype=np.int64)

    for i, (p, label) in enumerate(files):
        x = load_wav(p, cfg.sr)

        if rep == "T":
            f = feat_time(x, dims)
        elif rep == "FOURIER":
            f = feat_fourier(x, dims)
        elif rep == "WAVELET_STATS":
            f = feat_wavelet_stats(x, dims, cfg.wavelet_name, cfg.wavelet_level)
        else:
            raise ValueError(rep)

        X[i] = f
        y[i] = label

        if (i + 1) % 200 == 0:
            print(f"Extract: {rep} dims={dims} | {i+1}/{len(files)}")

    return X, y


def zscore_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-8
    return mu, sig


def zscore_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray):
    return (X - mu) / sig


def plot_feature_examples(cfg: Config, X_raw, X_norm, title, out_name):
    n = min(cfg.n_preview, X_raw.shape[0])

    fig = plt.figure(figsize=(12, 2.2 * n))
    for i in range(n):
        ax = plt.subplot(n, 1, i + 1)
        ax.plot(X_raw[i], linewidth=1.0, label="none")
        ax.plot(X_norm[i], linewidth=1.0, alpha=0.7, label="zscore")
        ax.set_title(f"{title} | sample {i+1}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    plt.tight_layout()
    if cfg.save_plots:
        fig.savefig(cfg.out_dir / "plots" / out_name, dpi=160)
    plt.close(fig)


def plot_feature_distribution(cfg: Config, X, title, out_name):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.hist(X.flatten(), bins=60, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if cfg.save_plots:
        fig.savefig(cfg.out_dir / "plots" / out_name, dpi=160)
    plt.close(fig)


def save_npz(path: Path, X_train, y_train, X_test, y_test, norm_mode, rep, dims, stats=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "norm_mode": norm_mode,
        "rep": rep,
        "dims": dims,
    }
    if stats is not None:
        payload["mu"] = stats["mu"]
        payload["sig"] = stats["sig"]
    np.savez_compressed(path, **payload)


def ensure_clean_out(cfg: Config):
    if cfg.overwrite and cfg.out_dir.exists():
        import shutil
        shutil.rmtree(cfg.out_dir)

    (cfg.out_dir / "npz").mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "plots").mkdir(parents=True, exist_ok=True)


def main():
    cfg = Config(
        data_dir=Path(".").resolve() / "processed",
        out_dir=Path(".").resolve() / "features",
        sr=16000,
        dims_list=(16, 32),
        reps=("T", "FOURIER", "WAVELET_STATS"),
        normalize_modes=("none", "zscore"),
        wavelet_name="db4",
        wavelet_level=5,
        n_preview=6,
        save_plots=True,
        overwrite=True,
    )

    ensure_clean_out(cfg)

    train_files = list_wavs(cfg.data_dir / "train")
    test_files = list_wavs(cfg.data_dir / "test")

    print(f"Train files: {len(train_files)}")
    print(f"Test files : {len(test_files)}")
    print(f"Wavelet: {cfg.wavelet_name} | level={cfg.wavelet_level}")

    meta = []

    for rep in cfg.reps:
        for dims in cfg.dims_list:
            X_train, y_train = extract_features(train_files, cfg, rep, dims)
            X_test, y_test = extract_features(test_files, cfg, rep, dims)

            mu, sig = zscore_fit(X_train)
            X_train_z = zscore_apply(X_train, mu, sig)
            X_test_z = zscore_apply(X_test, mu, sig)

            plot_feature_distribution(cfg, X_train, f"{rep} d={dims} | train (none)", f"dist_{rep}_d{dims}_none.png")
            plot_feature_distribution(cfg, X_train_z, f"{rep} d={dims} | train (zscore)", f"dist_{rep}_d{dims}_zscore.png")
            plot_feature_examples(cfg, X_train, X_train_z, f"{rep} d={dims}", f"examples_{rep}_d{dims}.png")

            for norm_mode in cfg.normalize_modes:
                if norm_mode == "none":
                    out_path = cfg.out_dir / "npz" / f"{rep}_d{dims}_none.npz"
                    save_npz(out_path, X_train, y_train, X_test, y_test, "none", rep, dims, stats=None)
                elif norm_mode == "zscore":
                    out_path = cfg.out_dir / "npz" / f"{rep}_d{dims}_zscore.npz"
                    save_npz(out_path, X_train_z, y_train, X_test_z, y_test, "zscore", rep, dims, stats={"mu": mu, "sig": sig})
                else:
                    raise ValueError(norm_mode)

                meta.append({
                    "rep": rep,
                    "dims": dims,
                    "norm": norm_mode,
                    "npz": str(out_path),
                })

            print(f"Saved: {rep} d={dims}")

    meta_path = cfg.out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Done.")
    print(f"NPZ: {cfg.out_dir / 'npz'}")
    print(f"Plots: {cfg.out_dir / 'plots'}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
