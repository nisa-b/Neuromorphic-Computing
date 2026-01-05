import os
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


@dataclass
class Config:
    raw_dir: Path
    out_dir: Path

    sr: int = 16000
    target_duration_sec: float = 1.0     # 1.0 or 2.0
    top_db: int = 25                     # try 20/25/30
    bandpass: bool = True
    lowcut: int = 300
    highcut: int = 3500

    test_ratio: float = 0.2
    seed: int = 42

    n_preview: int = 6
    save_plots: bool = True
    overwrite: bool = True


def infer_label(name: str) -> str:
    n = name.lower()
    if n.startswith("evet"):
        return "evet"
    if n.startswith("hayir"):
        return "hayir"
    raise ValueError(f"Cannot infer label: {name}")


def apply_bandpass(x, fs, lowcut=300, highcut=3500, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, x)


def crop_by_max_rms(x: np.ndarray, target_samples: int, frame=512, hop=128):
    if len(x) <= target_samples:
        return np.pad(x, (0, target_samples - len(x)), mode="constant"), 0, len(x)

    rms = librosa.feature.rms(y=x, frame_length=frame, hop_length=hop)[0]
    centers = librosa.frames_to_samples(np.arange(len(rms)), hop_length=hop)
    best_center = centers[int(np.argmax(rms))]

    start = int(np.clip(best_center - target_samples // 2, 0, len(x) - target_samples))
    end = start + target_samples
    return x[start:end], start, end


def load_audio(path: Path, sr: int):
    x, _ = librosa.load(str(path), sr=sr, mono=True)
    return x


def safe_write_wav(path: Path, x: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr)


def ensure_clean_dirs(cfg: Config):
    if cfg.overwrite and cfg.out_dir.exists():
        shutil.rmtree(cfg.out_dir)

    (cfg.out_dir / "plots").mkdir(parents=True, exist_ok=True)

    for split in ["all", "train", "test"]:
        for label in ["evet", "hayir"]:
            (cfg.out_dir / split / label).mkdir(parents=True, exist_ok=True)


def stratified_split(items, test_ratio, seed):
    rng = np.random.default_rng(seed)
    by_label = {"evet": [], "hayir": []}
    for it in items:
        by_label[it["label"]].append(it)

    train, test = [], []
    for label, lst in by_label.items():
        idx = np.arange(len(lst))
        rng.shuffle(idx)
        n_test = int(round(len(lst) * test_ratio))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        test.extend([lst[i] for i in test_idx])
        train.extend([lst[i] for i in train_idx])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def plot_hist(cfg: Config, durations, out_name):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    ax.hist(durations["evet"], bins=30, alpha=0.7, label="evet")
    ax.hist(durations["hayir"], bins=30, alpha=0.7, label="hayir")
    ax.set_title("Trimmed duration distribution")
    ax.set_xlabel("Trimmed duration (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    if cfg.save_plots:
        fig.savefig(cfg.out_dir / "plots" / out_name, dpi=160)
    plt.close(fig)


def plot_split_counts(cfg: Config, train_list, test_list, out_name):
    n_train_e = sum(1 for x in train_list if x["label"] == "evet")
    n_train_h = sum(1 for x in train_list if x["label"] == "hayir")
    n_test_e = sum(1 for x in test_list if x["label"] == "evet")
    n_test_h = sum(1 for x in test_list if x["label"] == "hayir")

    fig = plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.bar(["train_evet", "train_hayir", "test_evet", "test_hayir"],
           [n_train_e, n_train_h, n_test_e, n_test_h])
    ax.set_title("Train/Test class counts")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    if cfg.save_plots:
        fig.savefig(cfg.out_dir / "plots" / out_name, dpi=160)
    plt.close(fig)


def plot_wave_previews(cfg: Config, previews, title, out_name):
    if len(previews) == 0:
        return

    n = len(previews)
    fig = plt.figure(figsize=(12, 2.4 * n))
    for i, p in enumerate(previews):
        name = p["orig_name"]
        x_f = p["x_f"]
        x_t = p["x_t"]
        trim_idx = p["trim_idx"]
        crop_start = p["crop_start"]
        crop_end = p["crop_end"]

        t_f = np.arange(len(x_f)) / cfg.sr
        t_t = np.arange(len(x_t)) / cfg.sr

        ax = plt.subplot(n, 1, i + 1)
        ax.plot(t_f, x_f, linewidth=0.8, alpha=0.5, label="filtered")
        ax.plot(t_t + (trim_idx[0] / cfg.sr), x_t, linewidth=0.9, label="trimmed")

        xs = (trim_idx[0] + crop_start) / cfg.sr
        xe = (trim_idx[0] + crop_end) / cfg.sr
        ax.axvline(xs, linewidth=1.0)
        ax.axvline(xe, linewidth=1.0)

        ax.set_title(f"{name} | out: {cfg.target_duration_sec:.2f}s | crop: [{xs:.2f},{xe:.2f}] s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amp")
        ax.grid(True, alpha=0.25)

    plt.suptitle(title)
    plt.tight_layout()
    if cfg.save_plots:
        fig.savefig(cfg.out_dir / "plots" / out_name, dpi=160)
    plt.close(fig)


def plot_mels(cfg: Config, wav_paths, title, out_name):
    if len(wav_paths) == 0:
        return

    fig = plt.figure(figsize=(12, 2.6 * len(wav_paths)))
    for i, wp in enumerate(wav_paths):
        x, _ = librosa.load(str(wp), sr=cfg.sr, mono=True)
        S = librosa.feature.melspectrogram(y=x, sr=cfg.sr, n_mels=64, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)

        ax = plt.subplot(len(wav_paths), 1, i + 1)
        im = ax.imshow(S_db, aspect="auto", origin="lower")
        ax.set_title(wp.name)
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    plt.suptitle(title)
    plt.tight_layout()
    if cfg.save_plots:
        fig.savefig(cfg.out_dir / "plots" / out_name, dpi=160)
    plt.close(fig)


def write_mapping_csv(cfg: Config, rows):
    path = cfg.out_dir / "mapping.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "orig_name", "label", "all_name", "split", "split_name"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    cfg = Config(
        raw_dir=Path(".").resolve() / "raw",
        out_dir=Path(".").resolve() / "processed",
        sr=16000,
        target_duration_sec=1.0,  # change to 2.0 if needed
        top_db=25,
        bandpass=True,
        test_ratio=0.2,
        seed=42,
        n_preview=6,
        save_plots=True,
        overwrite=True,
    )

    ensure_clean_dirs(cfg)

    files = sorted(cfg.raw_dir.glob("*.wav"))
    if len(files) == 0:
        print("No WAV files found in raw/")
        return

    print(f"Found files: {len(files)}")
    print(f"Sample rate: {cfg.sr}")
    print(f"Target duration: {cfg.target_duration_sec:.2f}s")
    print(f"Trim top_db: {cfg.top_db}")
    print(f"Bandpass: {cfg.bandpass}")

    target_samples = int(cfg.sr * cfg.target_duration_sec)

    counters = {"evet": 0, "hayir": 0}
    items = []
    durations_trimmed = {"evet": [], "hayir": []}
    previews = {"evet": [], "hayir": []}

    for i, wav_path in enumerate(files, start=1):
        label = infer_label(wav_path.name)
        x_raw = load_audio(wav_path, cfg.sr)

        x_f = apply_bandpass(x_raw, cfg.sr, cfg.lowcut, cfg.highcut) if cfg.bandpass else x_raw
        x_t, trim_idx = librosa.effects.trim(x_f, top_db=cfg.top_db)

        x_out, crop_start, crop_end = crop_by_max_rms(x_t, target_samples)

        counters[label] += 1
        all_name = f"{label}_{counters[label]:04d}.wav"
        all_path = cfg.out_dir / "all" / label / all_name
        safe_write_wav(all_path, x_out, cfg.sr)

        durations_trimmed[label].append(len(x_t) / cfg.sr)

        items.append({
            "label": label,
            "orig_name": wav_path.name,
            "all_name": all_name,
            "all_path": all_path,
        })

        if len(previews[label]) < cfg.n_preview:
            previews[label].append({
                "orig_name": wav_path.name,
                "x_f": x_f,
                "x_t": x_t,
                "trim_idx": trim_idx,
                "crop_start": crop_start,
                "crop_end": crop_end,
            })

        if i % 50 == 0:
            print(f"Processed: {i}/{len(files)}")

    print("Preprocessing done.")
    print(f"Saved to: {cfg.out_dir / 'all'}")

    plot_hist(cfg, durations_trimmed, "hist_trimmed_durations.png")
    plot_wave_previews(cfg, previews["evet"], "Wave previews (evet) trim + energy-crop", "waves_evet_energy.png")
    plot_wave_previews(cfg, previews["hayir"], "Wave previews (hayir) trim + energy-crop", "waves_hayir_energy.png")

    train_list, test_list = stratified_split(items, cfg.test_ratio, cfg.seed)

    mapping_rows = []
    split_counters = {
        "train": {"evet": 0, "hayir": 0},
        "test": {"evet": 0, "hayir": 0},
    }

    def copy_and_rename(split_name, lst):
        for it in lst:
            label = it["label"]
            split_counters[split_name][label] += 1
            split_name_wav = f"{label}_{split_counters[split_name][label]:04d}.wav"
            dst = cfg.out_dir / split_name / label / split_name_wav
            shutil.copy2(it["all_path"], dst)

            mapping_rows.append({
                "orig_name": it["orig_name"],
                "label": label,
                "all_name": it["all_name"],
                "split": split_name,
                "split_name": split_name_wav,
            })

    copy_and_rename("train", train_list)
    copy_and_rename("test", test_list)

    write_mapping_csv(cfg, mapping_rows)

    plot_split_counts(cfg, train_list, test_list, "split_counts.png")

    evet_mels = sorted((cfg.out_dir / "train" / "evet").glob("*.wav"))[:cfg.n_preview]
    hayir_mels = sorted((cfg.out_dir / "train" / "hayir").glob("*.wav"))[:cfg.n_preview]
    plot_mels(cfg, evet_mels, "Mel-spectrogram (train/evet)", "mels_train_evet.png")
    plot_mels(cfg, hayir_mels, "Mel-spectrogram (train/hayir)", "mels_train_hayir.png")

    n_train_e = sum(1 for x in train_list if x["label"] == "evet")
    n_train_h = sum(1 for x in train_list if x["label"] == "hayir")
    n_test_e = sum(1 for x in test_list if x["label"] == "evet")
    n_test_h = sum(1 for x in test_list if x["label"] == "hayir")

    print("Split done.")
    print(f"Train: {len(train_list)} | evet={n_train_e} hayir={n_train_h}")
    print(f"Test : {len(test_list)} | evet={n_test_e} hayir={n_test_h}")
    print(f"Mapping: {cfg.out_dir / 'mapping.csv'}")
    print(f"Plots: {cfg.out_dir / 'plots'}")


if __name__ == "__main__":
    main()
