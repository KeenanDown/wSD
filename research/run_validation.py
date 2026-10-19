"""Reproduce synthetic and fixed-cohort empirical results without network access.

Run from the repository after `pip install -e '.[research]'` and fetch_eeg.py.
All results use natural logarithms without log(6) normalization.
"""
import csv
import gzip
import hashlib
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy
from scipy.signal import butter, sosfiltfilt
from wsd import compare_joint, compare_signals, compare_symbols, ordinal_mask, ordinal_patterns, patterns

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 20261019


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def save_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def synthetic():
    rng = np.random.default_rng(SEED)
    mask = ordinal_mask(3)
    counts = np.ones((6, 6), dtype=int)
    counts[~mask] = 10
    structural_zero = asdict(compare_joint(counts))
    counts[0, 2] += 1
    counts[1, 3] += 1
    counts[0, 3] -= 1
    counts[1, 2] -= 1
    np.savetxt(OUT / "counterexample_counts.csv", counts, fmt="%d", delimiter=",")
    result = compare_joint(counts)
    ix, iy = np.nonzero(counts)
    sx, sy = np.repeat(ix, counts[ix, iy]), np.repeat(iy, counts[ix, iy])
    perm = rng.permutation(len(sx))
    rank = patterns(3)
    x, y = rank[sx[perm]].ravel(), rank[sy[perm]].ravel()
    np.savetxt(OUT / "counterexample_signals.csv", np.c_[x, y], delimiter=",", header="x,y", comments="", fmt="%d")
    realised = compare_signals(x, y, stride=3)
    assert abs(realised.wsd - np.log(2)/36) < 1e-14
    assert abs(realised.wsmi + 11*np.log(2)/36) < 1e-14
    audit = {"n_tables": 10000, "min_wsd": 1., "max_excess_over_smi": 0., "max_identity_error": 0., "negative_wsmi": 0}
    for _ in range(audit["n_tables"]):
        p = rng.dirichlet(np.full(36, .4)).reshape(6, 6)
        r = compare_joint(p)
        audit["min_wsd"] = min(audit["min_wsd"], r.wsd)
        audit["max_excess_over_smi"] = max(audit["max_excess_over_smi"], r.wsd-r.smi)
        audit["max_identity_error"] = max(audit["max_identity_error"], abs(r.wsd-r.wsmi-r.retained_mass*np.log(r.reference_mass/r.retained_mass)))
        audit["negative_wsmi"] += int(r.wsmi < 0)
    null = []
    for size in (100, 300, 1000, 3000, 10000):
        for rep in range(200):
            a, b = rng.normal(size=(2, size))
            r = compare_signals(a, b)
            null.append({"n_samples": size, "replicate": rep, **asdict(r)})
    save_csv("independent_simulations.csv", null)
    null_summary = []
    for size in (100, 300, 1000, 3000, 10000):
        rows = [r for r in null if r["n_samples"] == size]
        null_summary.append({"n_samples": size, "replicates": 200,
                             "mean_wsd": float(np.mean([r["wsd"] for r in rows])),
                             "mean_wsmi": float(np.mean([r["wsmi"] for r in rows])),
                             "fraction_negative_wsmi": float(np.mean([r["wsmi"] < 0 for r in rows]))})
    dump("synthetic_summary.json", {"seed": SEED, "counterexample": asdict(result), "realised": asdict(realised), "dependent_zero": structural_zero, "random_audit": audit, "null_summary": null_summary})
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    ax[0].bar(["wSMI", "wSD", "SMI"], [result.wsmi, result.wsd, result.smi], color=["#b54646", "#21855b", "#526a9a"])
    ax[0].axhline(0, color="black", linewidth=.6)
    ax[0].set_ylabel("Information (nats)")
    ax[0].set_title("A  Exact counterexample", loc="left", fontsize=10)
    ax[1].plot([r["n_samples"] for r in null_summary], [r["mean_wsd"] for r in null_summary], "o-", color="#21855b")
    ax[1].set(xscale="log", yscale="log", xlabel="Independent Gaussian samples", ylabel="Mean estimated wSD (nats)")
    ax[1].set_title("B  Finite-sample null bias", loc="left", fontsize=10)
    fig.savefig(OUT / "synthetic_validation.pdf")
    fig.savefig(OUT / "synthetic_validation.png", dpi=180)
    plt.close(fig)
    print("Synthetic validation complete", flush=True)


def empirical():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    summary, scatter, surrogate_rows = [], [], []
    worst = None
    rng = np.random.default_rng(SEED+1)
    primary_records = []
    with gzip.open(OUT / "eeg_pairs.csv.gz", "wt", newline="") as out:
        fields = ["record", "delay", "epoch_start_s", "channel_x", "channel_y", "wsd", "wsmi", "smi", "retained_mass", "reference_mass", "n_observations"]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for entry in manifest["records"]:
            name = entry["record"]
            path = ROOT / "data/raw" / (name + ".edf")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            raw.pick("eeg")
            fs = raw.info["sfreq"]
            assert fs == 160 and len(raw.ch_names) == 64 and raw.n_times >= 60*fs
            data = raw.get_data()
            data -= data.mean(axis=0, keepdims=True)
            names = [s.strip(".").upper() for s in raw.ch_names]
            pairs = [(names.index(a), names.index(b)) for a, b in [("FP1", "FP2"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2")]]
            for delay in (1, 2, 4):
                sos = butter(4, [.5, fs/(3*delay)], btype="bandpass", fs=fs, output="sos")
                filtered = sosfiltfilt(sos, data, axis=-1)
                rows = []
                for start in (5, 15, 25, 35, 45):
                    epoch = filtered[:, int(start*fs):int((start+10)*fs)]
                    syms = [ordinal_patterns(ch, delay=delay, ties="drop") for ch in epoch]
                    for i in range(64):
                        for j in range(i+1, 64):
                            r = compare_symbols(syms[i], syms[j])
                            row = {"record": name, "delay": delay, "epoch_start_s": start,
                                   "channel_x": names[i], "channel_y": names[j], **asdict(r)}
                            writer.writerow(row)
                            rows.append(row)
                            assert -1e-14 <= r.wsd <= r.smi+1e-12
                            if delay == 4 and (worst is None or r.wsmi < worst["wsmi"]):
                                worst = row.copy()
                                valid = (syms[i] >= 0) & (syms[j] >= 0)
                                counts = np.bincount(6*syms[i][valid] + syms[j][valid], minlength=36).reshape(6, 6)
                                np.savetxt(OUT / "eeg_example_counts.csv", counts, fmt="%d", delimiter=",")
                    if delay == 4:
                        for i, j in pairs:
                            observed = compare_symbols(syms[i], syms[j])
                            shifts = rng.integers(int(2*fs), len(syms[j])-int(2*fs)+1, size=99)
                            null = [compare_symbols(syms[i], np.roll(syms[j], int(shift))).wsd for shift in shifts]
                            surrogate_rows.append({"record": name, "epoch_start_s": start, "channel_x": names[i], "channel_y": names[j],
                                                   "observed_wsd": observed.wsd, "shift_mean_wsd": float(np.mean(null)),
                                                   "shift_q95_wsd": float(np.quantile(null, .95)),
                                                   "n_shifts": 99})
                def array(key):
                    return np.array([r[key] for r in rows])
                record_summary = {"record": name, "delay": delay, "n_pair_epochs": len(rows),
                                  "negative_wsmi": int(np.sum(array("wsmi") < 0)),
                                  "min_wsmi": float(array("wsmi").min()), "min_wsd": float(array("wsd").min()),
                                  "median_wsmi": float(np.median(array("wsmi"))), "median_wsd": float(np.median(array("wsd"))),
                                  "max_wsd": float(array("wsd").max()), "max_wsd_minus_smi": float(np.max(array("wsd")-array("smi"))),
                                  "min_paired_observations": int(array("n_observations").min())}
                summary.append(record_summary)
                if delay == 4:
                    primary_records.extend(rows)
                    scatter.extend((r["wsmi"], r["wsd"]) for r in rows)
                print(name, "delay", delay, "negative wSMI", record_summary["negative_wsmi"], "/", len(rows), flush=True)
    save_csv("eeg_record_summary.csv", summary)
    save_csv("eeg_shift_controls.csv", surrogate_rows)
    primary = [r for r in summary if r["delay"] == 4]
    n = sum(r["n_pair_epochs"] for r in primary)
    negative = sum(r["negative_wsmi"] for r in primary)
    dump("eeg_summary.json", {"primary_delay": 4, "n_subjects": 5, "n_recordings": 10, "n_epochs": 50,
                              "n_pair_epochs": n, "negative_wsmi": negative, "fraction_negative_wsmi": negative/n,
                              "min_wsd": min(r["min_wsd"] for r in primary), "max_wsd": max(r["max_wsd"] for r in primary),
                              "min_wsmi": min(r["min_wsmi"] for r in primary), "example": worst,
                              "min_paired_observations": min(r["min_paired_observations"] for r in primary),
                              "max_wsd_minus_smi": max(r["max_wsd_minus_smi"] for r in primary),
                              "records": primary,
                              "mean_shift_control_wsd": float(np.mean([r["shift_mean_wsd"] for r in surrogate_rows]))})
    plot_eeg(primary, scatter)


def plot_eeg(primary, scatter):
    """Render the empirical figure; callable from the saved pair-level results."""
    points = np.array(scatter)
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    density = ax[0].hexbin(points[:, 0], points[:, 1], gridsize=55, bins="log", cmap="viridis", mincnt=1, rasterized=True)
    bar = fig.colorbar(density, ax=ax[0], fraction=.045, pad=.02)
    bar.set_label("Pair-epochs per bin", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    ax[0].axvline(0, color="#b54646", linewidth=.8)
    ax[0].set(xlabel="wSMI (nats)", ylabel="wSD (nats)")
    ax[0].set_title("A  All primary pair-epochs", loc="left", fontsize=10)
    xpos = np.arange(5)
    for run, offset, color in [("01", -.18, "#526a9a"), ("02", .18, "#21855b")]:
        values = [100*r["negative_wsmi"]/r["n_pair_epochs"] for r in primary if r["record"].endswith(run)]
        ax[1].bar(xpos+offset, values, .35, label="Eyes open" if run == "01" else "Eyes closed", color=color)
    ax[1].set_xticks(xpos, [f"S{i:03}" for i in range(1, 6)])
    ax[1].set_ylabel("Pair-epochs with wSMI < 0 (%)")
    ax[1].set_title("B  Descriptive recording summaries", loc="left", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "eeg_validation.pdf")
    fig.savefig(OUT / "eeg_validation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    dump("environment.json", {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
                              "scipy": scipy.__version__, "mne": mne.__version__, "matplotlib": matplotlib.__version__, "seed": SEED})
    synthetic()
    empirical()
