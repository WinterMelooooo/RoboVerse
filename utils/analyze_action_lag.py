#!/usr/bin/env python3
import argparse
import os
import sys
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import zarr


def log(msg: str):
    print(msg, flush=True)


def load_zarr_group(zarr_path: str):
    t0 = time.time()
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")
    g = zarr.open(zarr_path, mode="r")
    if "data" not in g or "meta" not in g or "episode_ends" not in g["meta"]:
        raise RuntimeError("Zarr layout missing 'data' or 'meta/episode_ends'.")
    log(f"[load] Opened zarr in {time.time()-t0:.2f}s")
    return g


def get_episode_offsets(episode_ends: np.ndarray) -> List[Tuple[int,int]]:
    offs = []
    start = 0
    for end in episode_ends:
        offs.append((start, int(end)))
        start = int(end)
    return offs


def to_gray_uint8(frames: np.ndarray) -> np.ndarray:
    # frames: (T,H,W,C) uint8 or float[0,1]
    x = frames
    if x.shape[1] in (1, 3, 4):
        x = np.transpose(x, (0, 2, 3, 1))
    if x.dtype != np.uint8:
        x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    if x.ndim != 4 or x.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unexpected frame shape {x.shape}, need (T,H,W,C).")
    if x.shape[-1] == 1:
        return x[..., 0]
    r, g, b = x[..., 0].astype(np.float32), x[..., 1].astype(np.float32), x[..., 2].astype(np.float32)
    gray = (0.299*r + 0.587*g + 0.114*b).astype(np.uint8)
    return gray


def visual_change_signal(frames: np.ndarray, stride_t: int = 1) -> np.ndarray:
    # mean |frame_t - frame_{t-1}| as visual-change proxy; z-score
    if stride_t > 1:
        frames = frames[::stride_t]
    gray = to_gray_uint8(frames)
    T = gray.shape[0]
    diffs = np.zeros((T,), dtype=np.float32)
    if T > 1:
        diffs[1:] = np.mean(
            np.abs(gray[1:].astype(np.float32) - gray[:-1].astype(np.float32)),
            axis=(1, 2),
        )
    std = np.std(diffs)
    if std > 1e-6:
        diffs = (diffs - diffs.mean()) / (std + 1e-6)
    return diffs


def action_magnitude_signal(actions: np.ndarray, stride_t: int = 1) -> np.ndarray:
    if stride_t > 1:
        actions = actions[::stride_t]
    mags = np.linalg.norm(actions, axis=1).astype(np.float32)
    std = np.std(mags)
    if std > 1e-6:
        mags = (mags - mags.mean()) / (std + 1e-6)
    return mags


def cross_correlation_best_lag(x: np.ndarray, y: np.ndarray, max_abs_lag: int) -> Tuple[int, float, np.ndarray, np.ndarray]:
    lags = np.arange(-max_abs_lag, max_abs_lag + 1, dtype=int)
    corrs = np.zeros_like(lags, dtype=np.float32)
    T = min(len(x), len(y))
    for i, lag in enumerate(lags):
        if lag > 0:
            xa, ya = x[:T-lag], y[lag:T]
        elif lag < 0:
            xa, ya = x[-lag:T], y[:T+lag]
        else:
            xa, ya = x[:T], y[:T]
        if len(xa) < 3:
            corrs[i] = np.nan
            continue
        vx, vy = xa - xa.mean(), ya - ya.mean()
        denom = (np.linalg.norm(vx) * np.linalg.norm(vy) + 1e-8)
        corrs[i] = float(np.dot(vx, vy) / denom)
    if np.all(np.isnan(corrs)):
        return 0, np.nan, lags, corrs
    valid = np.where(~np.isnan(corrs))[0]
    best_i = valid[np.argmax(np.abs(corrs[valid]))]
    return int(lags[best_i]), float(corrs[best_i]), lags, corrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True, help="path to dataset .zarr")
    ap.add_argument("--camera_key", default="head_camera")
    ap.add_argument("--action_key", default="action")
    ap.add_argument("--max_abs_lag", type=int, default=15)
    ap.add_argument("--example_ep", type=int, default=0)
    ap.add_argument("--outdir", default="./lag_report")
    ap.add_argument("--limit_episodes", type=int, default=0, help=">0 to only process first N episodes")
    ap.add_argument("--stride_t", type=int, default=1, help="temporal stride for speed (e.g., 2/3)")
    ap.add_argument("--downsample_hw", type=int, default=0, help="downsample shorter side to this px for speed (0=off)")
    ap.add_argument("--in_memory", action="store_true", help="load all arrays into RAM (faster but heavy)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    g = load_zarr_group(args.zarr)

    # Keys & meta
    data_group = g["data"]
    meta_group = g["meta"]
    episode_ends = meta_group["episode_ends"][:]
    n_episodes = len(episode_ends)
    offs = get_episode_offsets(episode_ends)

    if args.limit_episodes > 0:
        n_episodes = min(n_episodes, args.limit_episodes)
        offs = offs[:n_episodes]

    if args.camera_key not in data_group or args.action_key not in data_group:
        avail = [k for k in data_group.array_keys()]
        raise KeyError(f"Missing keys: camera_key='{args.camera_key}' or action_key='{args.action_key}'. Available: {avail}")

    # Optionally load into memory (not default)
    if args.in_memory:
        log("[mem] Loading arrays into RAM (this may take a while)...")
        t0 = time.time()
        frames_all = data_group[args.camera_key][:]
        actions_all = data_group[args.action_key][:]
        log(f"[mem] Loaded frames {frames_all.shape} dtype={frames_all.dtype}")
        log(f"[mem] Loaded actions {actions_all.shape} dtype={actions_all.dtype}")
        log(f"[mem] Done in {time.time()-t0:.2f}s")
    else:
        frames_all = data_group[args.camera_key]  # zarr array (lazy)
        actions_all = data_group[args.action_key] # zarr array (lazy)
        log(f"[lazy] frames shape={frames_all.shape} chunks={frames_all.chunks} dtype={frames_all.dtype}")
        log(f"[lazy] actions shape={actions_all.shape} chunks={actions_all.chunks} dtype={actions_all.dtype}")

    # Optional resize helper (no extra deps)
    def maybe_resize(frames: np.ndarray) -> np.ndarray:
        if args.downsample_hw <= 0:
            return frames
        # naive nearest resize with numpy (fast & dependency-free):
        # choose scale based on shorter side
        T, H, W, C = frames.shape
        short = min(H, W)
        if short <= args.downsample_hw:
            return frames
        scale = args.downsample_hw / short
        newH, newW = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
        # vectorized nearest neighbor
        yy = (np.linspace(0, H-1, newH)).astype(np.int64)
        xx = (np.linspace(0, W-1, newW)).astype(np.int64)
        frames_small = frames[:, yy][:, :, xx]
        return frames_small

    # Pass 1: iterate episodes and compute signals/correlations
    log(f"[info] Episodes: {len(offs)} (limit={args.limit_episodes})  stride_t={args.stride_t}  downsample_hw={args.downsample_hw}")
    summary_rows = []
    per_epi_cache: List[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]] = []

    for epi, (s, e) in tqdm(enumerate(offs), total=len(offs), desc="Analyzing episodes", ncols=100):
        try:
            # Slice episode
            if args.in_memory:
                f_ep = frames_all[s:e]
                a_ep = actions_all[s:e]
            else:
                f_ep = frames_all.get_orthogonal_selection((slice(s, e), slice(None), slice(None), slice(None)))
                a_ep = actions_all.get_orthogonal_selection((slice(s, e), slice(None)))

            # Downsample spatially if requested
            f_ep = maybe_resize(f_ep)

            # Compute signals
            vis = visual_change_signal(f_ep, stride_t=args.stride_t)
            act = action_magnitude_signal(a_ep, stride_t=args.stride_t)

            # Correlation scan
            best_lag, best_corr, lags, corrs = cross_correlation_best_lag(
                vis, act, max_abs_lag=args.max_abs_lag
            )

            summary_rows.append({
                "episode": epi,
                "length": int(e - s),
                "best_lag": int(best_lag),
                "best_corr": float(best_corr)
            })
            per_epi_cache.append((vis, act, lags, corrs))

            # occasional verbose ping
            if epi % 10 == 0:
                log(f"[epi {epi:04d}] len={e-s} best_lag={best_lag:+d} corr={best_corr:.3f}")

        except Exception as ex:
            log(f"[warn] episode {epi} failed: {ex}")
            summary_rows.append({"episode": epi, "length": int(e - s), "best_lag": np.nan, "best_corr": np.nan})
            per_epi_cache.append((None, None, None, None))

    # Save table
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.outdir, "action_lag_summary.csv")
    df.to_csv(csv_path, index=False)
    log(f"[save] Summary CSV → {csv_path}  (cols: episode,length,best_lag,best_corr)")

    # Histogram
    valid_best_lags = df["best_lag"].dropna().values
    plt.figure()
    plt.hist(valid_best_lags, bins=min(25, max(5, len(np.unique(valid_best_lags)))))
    plt.title("Histogram of Best Lags (frames)")
    plt.xlabel("Best lag (frames)")
    plt.ylabel("Count")
    plt.tight_layout()
    hist_path = os.path.join(args.outdir, "best_lag_hist.png")
    plt.savefig(hist_path); plt.close()
    log(f"[save] {hist_path}")

    # Example episode plots
    ex = int(np.clip(args.example_ep, 0, len(per_epi_cache)-1))
    vis, act, lags, corrs = per_epi_cache[ex]
    if vis is not None and act is not None:
        v = (vis - np.nanmin(vis)) / (np.nanmax(vis) - np.nanmin(vis) + 1e-6)
        a = (act - np.nanmin(act)) / (np.nanmax(act) - np.nanmin(act) + 1e-6)
        T = len(v)

        plt.figure()
        plt.plot(np.arange(T), v, label="visual_change (norm)")
        plt.plot(np.arange(T), a, label="action_magnitude (norm)")
        plt.legend()
        plt.title(f"Episode {ex}: Visual Change vs Action Magnitude")
        plt.xlabel("Time (frames)")
        plt.ylabel("Normalized magnitude")
        plt.tight_layout()
        ts_path = os.path.join(args.outdir, f"episode_{ex}_timeseries.png")
        plt.savefig(ts_path); plt.close()
        log(f"[save] {ts_path}")

        plt.figure()
        plt.plot(lags, corrs)
        best_lag = int(df.loc[df["episode"]==ex, "best_lag"].values[0])
        plt.axvline(x=best_lag, linestyle="--")
        plt.title(f"Episode {ex}: Correlation vs Lag")
        plt.xlabel("Lag (frames) [positive: obs leads action]")
        plt.ylabel("Pearson corr")
        plt.tight_layout()
        cvl_path = os.path.join(args.outdir, f"episode_{ex}_corr_vs_lag.png")
        plt.savefig(cvl_path); plt.close()
        log(f"[save] {cvl_path}")
    else:
        log(f"[skip] Episode {ex} insufficient for plotting.")

    log("[done] Finished all analyses.")


if __name__ == "__main__":
    main()
