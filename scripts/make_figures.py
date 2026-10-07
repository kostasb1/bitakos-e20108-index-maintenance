"""The figures of the thesis, in English and Greek, written to scripts/figures.

    python scripts/make_figures.py

The first five are illustrations on synthetic data or plots of the stored Faiss price list in
results/cost_model/query_cost_calibration.yaml. The last three read the result files of the main
runs, the half size runs and the K curve, so those must exist under results/. The script also
prints the worked numbers the thesis quotes for Quake's split threshold, so the text can be
checked against the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.config import SPLIT_ACCESS_ALPHA  # noqa: E402
from src.cost_model import lambda_from_prices  # noqa: E402

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
CAL = yaml.safe_load((ROOT / "results/cost_model/query_cost_calibration.yaml").read_text())

plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})

TEXT = {
    "en": {
        "ivf_q": "query", "ivf_note": "squares: centroids, dots: stored vectors, "
        "shaded: the nprobe = 2 partitions the query opens",
        "len": "list length, vectors", "pervec": "Faiss price per vector, in scan distances",
        "size": "partition size s, vectors", "amin": "smallest access fraction A that splits",
        "time": "timestep", "batch": "arrival batches",
        "staple": "starting index", "expire": "expires 50 timesteps after arrival",
        "steps": ["Dataset", "64 workload\nclusters", "Starting index\ngrown under LIRE",
                  "Stream: arrivals,\nexpiries, 8 queries\nper update",
                  "Maintenance policy\nevery 1,000 updates",
                  "Scoring on\nheld out queries"],
        "q8": "the policy restructures the index",
    },
    "el": {
        "ivf_q": "query", "ivf_note": "τετράγωνα: κεντροειδή, τελείες: αποθηκευμένα διανύσματα, "
        "σκιασμένα: τα partitions που ανοίγει το query με nprobe = 2",
        "len": "μήκος λίστας (διανύσματα)", "pervec": "τιμή του Faiss ανά διάνυσμα (scan distances)",
        "size": "μέγεθος partition s (διανύσματα)",
        "amin": "ελάχιστο ποσοστό πρόσβασης A για διάσπαση",
        "time": "χρονικό βήμα", "batch": "παρτίδες αφίξεων",
        "staple": "αρχικό ευρετήριο", "expire": "λήγει 50 βήματα μετά την άφιξη",
        "steps": ["Σύνολο\nδεδομένων", "64 συστάδες\nφόρτου", "Αρχικό ευρετήριο\nμε κανόνες LIRE",
                  "Ροή: αφίξεις, λήξεις,\n8 queries\nανά ενημέρωση",
                  "Πολιτική συντήρησης\nανά 1.000 ενημερώσεις",
                  "Βαθμολόγηση σε\nqueries ελέγχου"],
        "q8": "η πολιτική αναδιαρθρώνει το ευρετήριο",
    },
}


def fig_ivf(lang):
    """Draw a two dimensional IVF example, partitions, centroids and the two a query opens."""
    t = TEXT[lang]
    rng = np.random.default_rng(3)
    centres = np.array([[1, 1], [4, 1.2], [2.5, 3.6], [5.6, 3.4], [0.8, 4.2], [4.2, 5.4]])
    pts = np.vstack([c + rng.normal(0, 0.55, (28, 2)) for c in centres])
    # a few Lloyd iterations so the shown centroids are real means of their cells
    cen = centres.copy()
    for _ in range(10):
        lab = np.argmin(((pts[:, None] - cen[None]) ** 2).sum(-1), axis=1)
        cen = np.array([pts[lab == k].mean(0) for k in range(len(cen))])
    q = np.array([3.4, 2.5])
    near = np.argsort(((cen - q) ** 2).sum(1))[:2]
    xx, yy = np.meshgrid(np.linspace(-0.8, 7.2, 400), np.linspace(-0.8, 6.8, 400))
    grid = np.argmin(((np.stack([xx, yy], -1)[..., None, :] - cen) ** 2).sum(-1), axis=-1)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    shade = np.isin(grid, near).astype(float)
    ax.contourf(xx, yy, shade, levels=[0.5, 1.5], colors=["#dcdcdc"])
    ax.contour(xx, yy, grid, levels=np.arange(len(cen)) + 0.5, colors="#777777", linewidths=0.8)
    ax.scatter(pts[:, 0], pts[:, 1], s=9, c="#555555")
    ax.scatter(cen[:, 0], cen[:, 1], s=70, marker="s", c="black")
    ax.scatter([q[0]], [q[1]], s=110, marker="*", c="white", edgecolors="black", zorder=5)
    ax.annotate(t["ivf_q"], q, xytext=(q[0] + 0.25, q[1] - 0.45), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.8, 7.2)
    ax.set_ylim(-0.8, 6.8)
    # no title, the caption carries the legend, and the Greek title did not fit the width
    fig.tight_layout()
    fig.savefig(OUT / f"ivf_{lang}.png")
    plt.close(fig)


def fig_pipeline(lang):
    """Draw the pipeline of one experiment, from the dataset to the scoring."""
    t = TEXT[lang]
    # two rows of three boxes, so the Greek labels fit at a readable size
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.axis("off")
    w, h, gx, gy = 1.9, 0.8, 0.45, 0.75
    pos = [(k * (w + gx), 1) for k in range(3)] + [((2 - k) * (w + gx), 0) for k in range(3)]
    for label, (col, row) in zip(t["steps"], pos, strict=True):
        x, y = col, row * (h + gy)
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#eeeeee", ec="black", lw=0.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.5)
    arrow = {"arrowstyle": "->", "lw": 0.9}
    top, bot = 1 * (h + gy), 0
    for k in range(2):
        ax.annotate("", xy=((k + 1) * (w + gx), top + h / 2), xytext=(k * (w + gx) + w, top + h / 2),
                    arrowprops=arrow)
    # down from the starting index to the stream, then leftwards along the bottom row
    xr = 2 * (w + gx) + w / 2
    ax.annotate("", xy=(xr, bot + h), xytext=(xr, top), arrowprops=arrow)
    for k in (2, 1):
        ax.annotate("", xy=((k - 1) * (w + gx) + w, bot + h / 2), xytext=(k * (w + gx), bot + h / 2),
                    arrowprops=arrow)
    # the loop between the policy and the stream
    xs, xp = 2 * (w + gx) + w / 2, 1 * (w + gx) + w / 2
    ax.annotate("", xy=(xs - 0.3, bot), xytext=(xp + 0.3, bot),
                arrowprops={**arrow, "connectionstyle": "arc3,rad=0.45"})
    ax.text((xs + xp) / 2, bot - 0.62, t["q8"], ha="center", va="center", fontsize=7.8)
    ax.set_xlim(-0.1, 3 * w + 2 * gx + 0.1)
    ax.set_ylim(-0.95, 2 * h + gy + 0.1)
    fig.tight_layout()
    fig.savefig(OUT / f"pipeline_{lang}.png")
    plt.close(fig)


def fig_retention(lang):
    """Draw the retention window, each batch expiring 50 time steps after it arrives."""
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    # the starting index expires over the first 50 timesteps, drawn as a shrinking band
    ax.fill_between([0, 50], [-0.3, -0.3], [0.9, -0.3], color="#bbbbbb", ec="black", lw=0.6)
    ax.text(12, 0.05, t["staple"], ha="left", va="center", fontsize=8)
    for row, start in enumerate(range(0, 100, 12), start=1):
        end = min(start + 50, 100)
        ax.barh(row, end - start, left=start, height=0.6, color="white", ec="black", lw=0.6,
                hatch="///" if row % 2 else "")
        if start + 50 > 100:
            ax.plot([100, 104], [row, row], color="black", lw=0.8, ls=":")
    ax.axvline(100, color="black", lw=0.8, ls=":")
    ax.annotate(t["expire"], xy=(62, 2), xytext=(2, 8.3), fontsize=8,
                arrowprops={"arrowstyle": "->", "lw": 0.7})
    ax.set_yticks([])
    ax.set_ylabel(t["batch"])
    ax.set_xlabel(t["time"])
    ax.set_xlim(-2, 106)
    ax.set_ylim(-0.6, 9.2)
    fig.tight_layout()
    fig.savefig(OUT / f"retention_{lang}.png")
    plt.close(fig)


def fig_list_price(lang):
    """Plot the Faiss price per vector against list length, for both datasets."""
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for dim, name, mk, ls in ((128, "SIFT1M, 128", "o", "-"), (960, "GIST1M, 960", "s", "--")):
        e = CAL[dim]
        sz = np.asarray(e["median"]["list_sizes"], float)
        med = np.asarray(e["median"]["list_prices"], float)
        lo, hi = (np.asarray(r, float) for r in e["range"]["list_prices"])
        keep = sz > 0
        ax.errorbar(sz[keep], med[keep] / sz[keep],
                    yerr=[(med - lo)[keep] / sz[keep], (hi - med)[keep] / sz[keep]],
                    marker=mk, ls=ls, color="black", ms=4, capsize=2, lw=1, label=name)
    ax.set_xscale("log")
    ax.set_xlabel(t["len"])
    ax.set_ylabel(t["pervec"])
    ax.axhline(1.0, color="#999999", lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"list_price_{lang}.png")
    plt.close(fig)


def fig_quake_threshold(lang):
    """Plot the smallest access fraction at which Quake splits a partition of each size."""
    t = TEXT[lang]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    sizes = np.arange(100, 1601, 10)
    styles = {(128, 250): ("-", "SIFT1M, τ = 250 ns"), (128, 50): (":", "SIFT1M, τ = 50 ns"),
              (960, 250): ("--", "GIST1M, τ = 250 ns"), (960, 50): ("-.", "GIST1M, τ = 50 ns")}
    for (dim, tau), (ls, label) in styles.items():
        cm = lambda_from_prices(CAL[dim]["median"])
        lam = cm.scan_latency
        # delta C = A (2 alpha lam(s/2) - lam(s)) + one more centroid to rank, split if < -tau
        gain = np.array([lam(int(s)) - 2 * SPLIT_ACCESS_ALPHA * lam(int(s) // 2) for s in sizes])
        amin = (tau + cm.centroid_ns) / gain
        # above 1 no access fraction can clear tau, the rule never splits there
        amin[(gain <= 0) | (amin > 1.0)] = np.nan
        ax.plot(sizes, amin, ls=ls, color="black", lw=1.1, label=label)
    ax.set_yscale("log")
    ax.set_ylim(0.004, 1.0)
    ax.set_xlabel(t["size"])
    ax.set_ylabel(t["amin"])
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"quake_threshold_{lang}.png")
    plt.close(fig)


# figures of the final results, read from the result files of the three sets of runs

RAW = ROOT / "results" / "raw"
ORDER = ["global_rebuild", "lire_lite", "dedrift_lazy", "dedrift_split", "dedrift_hybrid",
         "cost_driven_quake", "cost_driven_quake_tau50", "bandit"]
NAMES = {
    "en": {"global_rebuild": "Global rebuild", "lire_lite": "LIRE", "dedrift_lazy": "DeDrift Lazy",
           "dedrift_split": "DeDrift Split", "dedrift_hybrid": "DeDrift Hybrid", "cost_driven_quake": "Quake",
           "cost_driven_quake_tau50": "Quake, τ = 50 ns", "bandit": "Bandit policy", "no_op": "No maintenance"},
    "el": {"global_rebuild": "Πλήρης ανακατασκευή", "lire_lite": "LIRE", "dedrift_lazy": "DeDrift Lazy",
           "dedrift_split": "DeDrift Split", "dedrift_hybrid": "DeDrift Hybrid", "cost_driven_quake": "Quake",
           "cost_driven_quake_tau50": "Quake, τ = 50 ns", "bandit": "Πολιτική bandit",
           "no_op": "Καμία συντήρηση"},
}
RTEXT = {
    "en": {"change": "change in query cost against no maintenance, %", "work": "maintenance work, scan distances",
           "sift_default": "SIFT1M, starting size 100", "sift_half": "SIFT1M, starting size 50",
           "gist": "GIST1M, starting size 100", "kmult": "partitions, multiple of the starting count",
           "kchange": "query cost against the starting count, %", "seed": "seed"},
    "el": {"change": "μεταβολή κόστους query έναντι ευρετηρίου χωρίς συντήρηση (%)",
           "work": "εργασία συντήρησης (scan distances)",
           "sift_default": "SIFT1M, στόχος 100 διανυσμάτων ανά partition", "sift_half": "SIFT1M, στόχος 50 διανυσμάτων ανά partition",
           "gist": "GIST1M, στόχος 100 διανυσμάτων ανά partition", "kmult": "αριθμός partitions, ως πολλαπλάσιο του αρχικού",
           "kchange": "μεταβολή κόστους query (%)", "seed": "seed"},
}


def runs(sweep, tag):
    """Read the result rows of one set of runs and add the change against no maintenance, in percent."""
    import pandas as pd
    d = pd.concat([pd.read_csv(f) for f in sorted((RAW / sweep).glob(f"{tag}_*.csv"))])
    base = d[d.maintainer == "no_op"].set_index("seed").query_cost
    d["change"] = 100 * (d.query_cost / d.seed.map(base) - 1)
    return d


def summary(sweep_dir, dataset):
    """Read the ranking table that scripts/analyse_final.py wrote for one dataset."""
    import pandas as pd
    return pd.read_csv(ROOT / "results" / sweep_dir / f"ranking_{dataset}.csv", index_col=0)


def _dotpanel(ax, series, lang, show_labels):
    """Draw per seed changes as dots and the mean with its interval, one row per policy."""
    # each entry of series is the run rows, the summary, the marker, its fill, a vertical offset and a label
    ys = {m: i for i, m in enumerate(ORDER[::-1])}
    for d, s, mk, face, off, lab in series:
        for m in ORDER:
            y = ys[m] + off
            pts = d[d.maintainer == m].change.values
            ax.scatter(pts, np.full(len(pts), y), s=9, color="#888888", zorder=2)
            lo, mid, hi = s.loc[m, ["query_cost_ci_lo", "query_cost_vs_no_op", "query_cost_ci_hi"]]
            ax.plot([lo, hi], [y, y], color="black", lw=1.0, zorder=3)
            ax.scatter([mid], [y], s=34, marker=mk, facecolor=face, edgecolor="black", zorder=4,
                       label=lab if m == ORDER[0] else None)
    ax.axvline(0, color="black", lw=0.7, ls=":")
    ax.set_yticks(list(ys.values()))
    if show_labels:
        ax.set_yticklabels([NAMES[lang][m] for m in ys])
    else:
        ax.tick_params(labelleft=False)


def fig_results_change(lang):
    """Plot the change in query cost against no maintenance, SIFT at both sizes and GIST."""
    t = RTEXT[lang]
    sift, sift_h, gist = runs("final_2709", "sift"), runs("sens_ta50_2709", "sift"), runs("final_2709", "gist")
    s_sift, s_half = summary("final_2709", "sift1m"), summary("sens_ta50_2709", "sift1m")
    s_gist = summary("final_2709", "gist1m")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.2), sharey=True)
    _dotpanel(axes[0], [(sift, s_sift, "o", "black", 0.16, t["sift_default"]),
                        (sift_h, s_half, "s", "white", -0.16, t["sift_half"])], lang, True)
    axes[0].set_title("SIFT1M", fontsize=9.5)
    _dotpanel(axes[1], [(gist, s_gist, "o", "black", 0.0, t["gist"])], lang, False)
    axes[1].set_title("GIST1M", fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.supxlabel(t["change"], fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"results_change_{lang}.png")
    plt.close(fig)


def fig_results_frontier(lang):
    """Plot the change in query cost against maintenance work, per seed and on average."""
    t = RTEXT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6))
    marks = dict(zip(ORDER, ["o", "s", "^", "v", "D", "P", "X", "*"], strict=True))
    for ax, tag, ds in ((axes[0], "sift", "SIFT1M"), (axes[1], "gist", "GIST1M")):
        d = runs("final_2709", tag)
        for m in ORDER:
            g = d[d.maintainer == m]
            ax.scatter(g.work_calibrated, g.change, s=10, color="#999999", zorder=2)
            ax.scatter([g.work_calibrated.mean()], [g.change.mean()], s=46, marker=marks[m],
                       facecolor="white", edgecolor="black", zorder=3, label=NAMES[lang][m])
        ax.axhline(0, color="black", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(t["work"], fontsize=8.5)
        ax.set_title(ds, fontsize=9.5)
    axes[0].set_ylabel(t["change"], fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / f"results_frontier_{lang}.png")
    plt.close(fig)


def fig_k_curve(lang):
    """Plot the query cost of fresh builds against the partition count, as a multiple of the start."""
    import pandas as pd
    t = RTEXT[lang]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for f, lab, ls, mk in (("k_curve_sift1m_s42.csv", "SIFT1M, seed 42", "-", "o"),
                           ("k_curve_gist1m_s42.csv", "GIST1M, seed 42", "--", "s"),
                           ("k_curve_gist1m_s1.csv", "GIST1M, seed 1", ":", "^")):
        d = pd.read_csv(RAW / "diag_2809" / f)
        b = d[d.kind == "build"]
        ref = b[b.mult == 1.0].cost_400.mean()
        g = b.groupby("mult").cost_400
        mean = 100 * (g.mean() / ref - 1)
        ax.plot(mean.index, mean.values, ls=ls, marker=mk, color="black", ms=4, lw=1.1, label=lab)
        for mult, vals in g:
            ax.scatter(np.full(len(vals), mult), 100 * (vals.values / ref - 1), s=7, color="#999999")
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 4.5, 8.5])
    ax.set_xticklabels(["1", "1.5", "2", "3", "4.5", "8.5"])
    ax.set_xlabel(t["kmult"])
    ax.set_ylabel(t["kchange"])
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / f"k_curve_{lang}.png")
    plt.close(fig)


if __name__ == "__main__":
    for lang in ("en", "el"):
        fig_ivf(lang)
        fig_pipeline(lang)
        fig_retention(lang)
        fig_list_price(lang)
        fig_quake_threshold(lang)
        fig_results_change(lang)
        fig_results_frontier(lang)
        fig_k_curve(lang)
    # the worked numbers the text quotes, printed so the prose can be checked against the code
    for dim in (128, 960):
        cm = lambda_from_prices(CAL[dim]["median"])
        lam = cm.scan_latency
        g = lam(400) - 2 * SPLIT_ACCESS_ALPHA * lam(200)
        print(dim, "lam400", round(lam(400)), "lam200", round(lam(200)), "gain", round(g),
              "centroid", round(cm.centroid_ns, 1),
              "Amin250", round((250 + cm.centroid_ns) / g, 3),
              "Amin50", round((50 + cm.centroid_ns) / g, 3))
    print("written to", OUT)
