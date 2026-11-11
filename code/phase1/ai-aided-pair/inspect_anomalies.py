"""
Visual inspection of the anomalies found in the Cellpose3 degradation analysis (phase 1, black-box).

Requires degradation.py and degradation_analysis.py in the same folder. Re-uses their sample building, model runner
and metrics, so the numbers match the ones in results/agg.csv (same seed handling, same normalisation).

Usage in the notebook:
    from inspect_anomalies import run_inspection
    run_inspection(img, mask, dn_model, seg_model, results_dir="results", out_dir="results/inspect")
    # only some sections:
    run_inspection(img, mask, dn_model, seg_model, sections=["noise", "blur"])
    # only the CSV based plots (no models needed):
    run_inspection(None, None, None, None, results_dir="results", sections=["csv"])

Sections (one figure per section, plus CSV tables):
  noise      Poisson noise 80 ... 1: image looks fine (small LPIPS) but many extra objects / lower AP.
  blur       Gaussian blur: phase 1 = extra objects, phase 2 = objects fuse (merge rate rises, IoU drops).
  diameter   Wrong diameter: <=1.5x unchanged, 2x merge peak (mostly with denoise), 3x collapse (2 objects).
  smoothing  Open question: do downsample 3 / rotation 60, 135 help only because interpolation smooths the image?
             Controls: pixel permutations (orientation floor), tiny rotations/resizes (interpolation only).
  combo      Blur 6 + noise 1: raw path finds nothing, denoise path merges (merge rate 0.33).
  csv        From results/agg.csv: (a) blur relative to cell diameter (sigma / d), (b) LPIPS/SSIM vs blur.

Every predicted object is assigned to one category (see classify_objects):
  matched  one-to-one match with a GT cell at IoU >= 0.5
  merged   covers >= 2 GT cells (same rule as the merge rate)
  extra    not matched and mostly (>= 50 %) outside every GT cell
  poor     not matched, but mostly on GT cells (IoU < 0.5: wrong shape/size, or a fragment)
For each object the share of the "cyto" channel (green by default) in the clean image is computed. This tests the
hypothesis that extra objects are red-only blobs (cyto share close to 0) that the ground truth does not annotate.

ASSUMPTION (unverified): red = nucleus channel, green = cytoplasm channel (AnalysisConfig.nuc_idx / cyto_idx).
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from skimage.segmentation import find_boundaries

from degradation_analysis import (
    AnalysisConfig, LPIPSScorer, ModelRunner, build_sample, params_key, overlap_matrix, score_sample,
    _relabel, _rgb, _style_ax, _new_fig, _finish,
    RAW_C, DN_C, INK, INK2, MUTED, GRID, AXIS, SURFACE, SEQ,
)

CATS = ("matched", "merged", "extra", "poor")
# Outline colours on the dark image (red/green channels): white, magenta, cyan, light violet
OVERLAY_COLORS = {"matched": (1.0, 1.0, 1.0), "merged": (1.0, 0.31, 0.85), "extra": (0.0, 0.90, 1.0),
                  "poor": (0.60, 0.60, 1.0)}
# The same four categories on the light plot surface
PLOT_COLORS = {"matched": "#52514e", "merged": "#d0399a", "extra": "#0f9bb0", "poor": "#6250d6"}

ALL_SECTIONS = ("noise", "blur", "diameter", "smoothing", "combo", "csv")


# --------------------------------------------------------------------------------------
# Object categories
# --------------------------------------------------------------------------------------
def classify_objects(gt, pred, cfg: AnalysisConfig, match_iou: float = 0.5, extra_max_gt_frac: float = 0.5):
    """Returns (relabelled pred, array of category strings, one per predicted label 1..n)."""
    gt, pred = _relabel(gt), _relabel(pred)
    n_gt, n_pred = int(gt.max()), int(pred.max())
    if n_pred == 0:
        return pred, np.array([], dtype=object)
    if n_gt == 0:
        return pred, np.array(["extra"] * n_pred, dtype=object)
    ov = overlap_matrix(gt, pred)
    inter = ov[1:, 1:].astype(np.float64)
    a_g, a_p = ov[1:, :].sum(1).astype(np.float64), ov[:, 1:].sum(0).astype(np.float64)
    iou = inter / (a_g[:, None] + a_p[None, :] - inter)
    ti, pi = linear_sum_assignment(-iou)
    matched = np.zeros(n_pred, bool)
    matched[pi[iou[ti, pi] >= match_iou]] = True
    covers = (inter / a_g[:, None] >= cfg.cover_frac) & (inter / a_p[None, :] >= cfg.min_frac)
    merged = covers.sum(0) >= 2                      # same rule as in segmentation_metrics
    gt_frac = inter.sum(0) / a_p                     # share of the object lying on any GT cell
    cats = np.where(merged, "merged", np.where(matched, "matched", np.where(gt_frac < extra_max_gt_frac, "extra", "poor")))
    return pred, cats.astype(object)


def cyto_share(pred, ref2) -> np.ndarray:
    """Per object: mean(cyto) / (mean(cyto) + mean(nuc)) in the clean reference image ref2 = [nuc, cyto]."""
    n = int(pred.max())
    if n == 0:
        return np.array([])
    idx = np.arange(1, n + 1)
    nuc = np.asarray(ndi.mean(ref2[..., 0], pred, idx))
    cyto = np.asarray(ndi.mean(ref2[..., 1], pred, idx))
    return cyto / (cyto + nuc + 1e-6)


def _f(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


# --------------------------------------------------------------------------------------
# Inspector: builds samples, runs both pipelines once, caches everything
# --------------------------------------------------------------------------------------
class Inspector:
    def __init__(self, img, mask, dn_model, seg_model, cfg: AnalysisConfig | None = None, seed: int = 0,
                 batch_size: int | None = None, lpips_device: str | None = None):
        self.img, self.mask, self.seed = img, mask, seed
        self.cfg = cfg or AnalysisConfig()
        self.runner = ModelRunner(dn_model, seg_model, self.cfg, batch_size=batch_size)
        self.lp = LPIPSScorer(self.cfg.lpips_net, lpips_device)
        self.cache: dict = {}
        self.section_of: dict = {}

    def _make_result(self, s, m_raw, m_dn, dn_cn):
        cfg = self.cfg
        r = dict(sample=s, metrics=score_sample(s, m_raw, m_dn, dn_cn, cfg, self.lp),
                 in2=s.img_aug[..., [cfg.nuc_idx, cfg.cyto_idx]], dn2=np.asarray(dn_cn)[..., ::-1],
                 masks={}, cats={}, share={})
        for path, m in (("raw", m_raw), ("dn", m_dn)):
            lab, cats = classify_objects(s.mask, m, cfg)
            r["masks"][path], r["cats"][path], r["share"][path] = lab, cats, cyto_share(lab, s.ref2)
        return r

    def evaluate(self, conditions, section: str = ""):
        """conditions: list of (label, params). Returns a list of result dicts (skipped conditions are dropped)."""
        todo = []
        for label, params in conditions:
            key = (params_key(params), self.seed)
            if key in self.cache or any(key == k for k, _ in todo):
                continue
            s = build_sample(self.img, self.mask, params, self.seed, self.cfg)
            if s.skipped:
                print(f"[skip] {label}: {s.skipped}")
                self.cache[key] = None
                continue
            todo.append((key, s))
        if todo:
            print(f"[{section}] running {len(todo)} condition(s) through both pipelines ...")
            m_raw, m_dn, dn_imgs = self.runner.run([s for _, s in todo])
            for (key, s), mr, md, dn in zip(todo, m_raw, m_dn, dn_imgs):
                self.cache[key] = self._make_result(s, mr, md, dn)
        out = []
        for label, params in conditions:
            key = (params_key(params), self.seed)
            r = self.cache.get(key)
            if r is not None:
                out.append({**r, "label": label, "section": section})
        return out


# --------------------------------------------------------------------------------------
# Figures: overlays
# --------------------------------------------------------------------------------------
def _overlay(im_rgb, lab, cats):
    """Colour the outline of every object by its category."""
    out = np.array(im_rgb, dtype=np.float32, copy=True)
    if lab is None or lab.max() == 0:
        return out
    bnd = find_boundaries(lab, mode="thick")
    for cat, col in OVERLAY_COLORS.items():
        ids = np.where(cats == cat)[0] + 1
        if ids.size:
            out[bnd & np.isin(lab, ids)] = col
    return out


def _counts_text(cats):
    c = Counter(cats)
    return " | ".join(f"{k} {c.get(k, 0)}" for k in CATS)


def _legend(fig):
    handles = [Line2D([0], [0], color=OVERLAY_COLORS[c], lw=3, label=c) for c in CATS]
    leg = fig.legend(handles=handles, loc="upper right", ncol=4, frameon=True, fontsize=10, facecolor="#1d1d1c",
                     edgecolor=AXIS, labelcolor="white", title="object category (outline colour)",
                     title_fontsize=9)
    leg.get_title().set_color("white")


def plot_overlay_rows(results, title, out_dir, name, show):
    """One row per condition: input | denoised | seg. without denoise | seg. with denoise | ground truth."""
    if not results:
        return
    fig, axes = _new_fig(len(results), 5, (20, 4.1 * len(results)))
    for r_i, r in enumerate(results):
        s, m = r["sample"], r["metrics"]
        ref_rgb = _rgb(s.ref2)
        in_rgb, dn_rgb = _rgb(r["in2"]), _rgb(np.clip(r["dn2"], 0, 1))
        n_raw, n_dn = len(r["cats"]["raw"]), len(r["cats"]["dn"])
        panels = [
            (in_rgb, "degraded input"),
            (dn_rgb, f"denoised (LPIPS to clean: in {_f(m['lpips_in'])} -> dn {_f(m['lpips_dn'])})"),
            (_overlay(in_rgb, r["masks"]["raw"], r["cats"]["raw"]),
             f"seg. WITHOUT denoise: {n_raw} objects\n{_counts_text(r['cats']['raw'])}"),
            (_overlay(in_rgb, r["masks"]["dn"], r["cats"]["dn"]),
             f"seg. WITH denoise: {n_dn} objects\n{_counts_text(r['cats']['dn'])}"),
            (np.where(find_boundaries(_relabel(s.mask), mode="thick")[..., None], 1.0, ref_rgb),
             f"ground truth ({int(np.max(s.mask))} cells)"),
        ]
        for c_i, (im, ttl) in enumerate(panels):
            ax = axes[r_i, c_i]
            ax.imshow(np.clip(im, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(ttl, color=INK, fontsize=9)
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[r_i, 0].set_ylabel(
            f"{r['label']}\nAP50 {_f(m['raw_ap50'])} / {_f(m['dn_ap50'])}\nmerge {_f(m['raw_merge_rate'])} / "
            f"{_f(m['dn_merge_rate'])}\n(without / with denoise)", color=INK2, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=13, x=0.02, ha="left")
    _legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _finish(fig, out_dir, name, show)


def plot_object_composition(results, title, out_dir, name, show):
    """Cyto-channel share per object and category. Extra objects at ~0 would mean 'red-only blobs'."""
    if not results:
        return
    rng = np.random.default_rng(0)
    fig, axes = _new_fig(2, len(results), (3.1 * len(results) + 1.2, 6.2))
    for c_i, r in enumerate(results):
        for r_i, (path, plabel) in enumerate((("raw", "without denoise"), ("dn", "with denoise"))):
            ax = axes[r_i, c_i]
            _style_ax(ax)
            cats, share = r["cats"][path], r["share"][path]
            ticklabels = []
            for x, cat in enumerate(CATS):
                sel = share[cats == cat] if len(cats) else np.array([])
                ticklabels.append(f"{cat}\nn={len(sel)}")
                if len(sel):
                    ax.scatter(x + rng.uniform(-0.18, 0.18, len(sel)), sel, s=16, color=PLOT_COLORS[cat],
                               alpha=0.8, edgecolor=SURFACE, linewidth=0.5, zorder=3)
                    ax.hlines(np.median(sel), x - 0.3, x + 0.3, color=INK, linewidth=2, zorder=4)
            ax.set_xticks(range(len(CATS)))
            ax.set_xticklabels(ticklabels, fontsize=7)
            ax.set_ylim(-0.03, 1.03)
            if c_i == 0:
                ax.set_ylabel(f"{plabel}\ncyto share of object intensity", fontsize=8)
            if r_i == 0:
                ax.set_title(r["label"], fontsize=9)
    fig.suptitle(title + "  (black bar = median; 1 = only cyto channel, 0 = only nuc channel)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _finish(fig, out_dir, name, show)


def _section_figures(insp, conds, section, title, out_dir, show, composition=True):
    results = insp.evaluate(conds, section)
    plot_overlay_rows(results, title, out_dir, f"inspect_{section}.png", show)
    if composition:
        plot_object_composition(results, f"Object composition: {title}", out_dir, f"inspect_{section}_composition.png",
                                show)
    return results


# --------------------------------------------------------------------------------------
# Smoothing control (open question: is the gain from interpolation / smoothing?)
# --------------------------------------------------------------------------------------
SMOOTHING_GROUPS = [
    ("baseline", [("baseline", {})]),
    ("orientation", [("rot90 k=1", {"rot90": 1}), ("rot90 k=2", {"rot90": 2}), ("rot90 k=3", {"rot90": 3}),
                     ("flip horizontal", {"flip": (1, 0)}), ("flip vertical", {"flip": (0, 1)})]),
    ("interp. only", [("rotate 0.5°", {"rotate": 0.5}), ("rotate 1°", {"rotate": 1.0}), ("rotate 2°", {"rotate": 2.0}),
                      ("resize 0.99", {"resize": 0.99}), ("resize 1.01", {"resize": 1.01})]),
    ("rotation", [("rotate 30°", {"rotate": 30.0}), ("rotate 45°", {"rotate": 45.0}), ("rotate 60°", {"rotate": 60.0}),
                  ("rotate 135°", {"rotate": 135.0})]),
    ("smoothing", [("blur σ=0.5", {"blur": 0.5}), ("blur σ=1", {"blur": 1.0}), ("downsample 2", {"down": 2}),
                   ("downsample 3", {"down": 3}), ("blur 1 + down 2", {"blur": 1.0, "down": 2})]),
]


def plot_smoothing_control(insp: Inspector, out_dir, show):
    """Dot plot of AP50 / mean IoU / merge / split for all control conditions.

    Orientation group = pure pixel permutations (no interpolation): its min-max range is the 'floor' of what a
    change in the metrics can mean for this single sample. Interpolation-only conditions change the pixels very
    little but resample the image once (slight smoothing)."""
    rows = []
    for gname, conds in SMOOTHING_GROUPS:
        for res in insp.evaluate(conds, "smoothing-control"):
            rows.append((gname, res["label"], res["metrics"]))
    if not rows:
        return None
    panels = [("AP@0.5", "ap50"), ("mean best IoU", "mean_iou"), ("merge rate", "merge_rate"),
              ("split rate", "split_rate")]
    fig, axes = _new_fig(1, 4, (18, 0.42 * len(rows) + 2.2))
    ypos = np.arange(len(rows))[::-1].astype(float)
    ylabels = [f"[{g}] {lab}" for g, lab, _ in rows]
    base = next((m for g, _, m in rows if g == "baseline"), None)
    orient = [m for g, _, m in rows if g == "orientation"]
    for ax, (ttl, col) in zip(axes[0], panels):
        _style_ax(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        for path, color, size in (("raw", RAW_C, 70), ("dn", DN_C, 30)):
            if orient:
                vals = [m[f"{path}_{col}"] for m in orient]
                ax.axvspan(min(vals), max(vals), color=color, alpha=0.10, linewidth=0)
            if base is not None:
                ax.axvline(base[f"{path}_{col}"], color=color, linestyle=":", linewidth=1)
            ax.scatter([m[f"{path}_{col}"] for _, _, m in rows], ypos, s=size, color=color, edgecolor=SURFACE,
                       linewidth=0.8, zorder=3 if path == "dn" else 2,
                       label="without denoise" if path == "raw" else "with denoise")
        prev = None
        for y, (g, _, _) in zip(ypos, rows):
            if prev is not None and g != prev:
                ax.axhline(y + 0.5, color=AXIS, linewidth=0.8)
            prev = g
        allv = np.array([m[f"{p}_{col}"] for p in ("raw", "dn") for _, _, m in rows], dtype=float)
        allv = allv[np.isfinite(allv)]
        if allv.size:                                  # avoid 1e-17 axes when all values are identical
            lo, hi = float(allv.min()), float(allv.max())
            pad = max(0.1 * (hi - lo), 0.03)
            lo_lim = max(lo - pad, -0.02) if col in ("merge_rate", "split_rate") else lo - pad
            ax.set_xlim(lo_lim, hi + pad)
        ax.set_yticks(ypos)
        ax.set_yticklabels(ylabels if ax is axes[0, 0] else [], fontsize=8)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(ttl)
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Smoothing control (single sample, seed 0). Shaded band = range of the orientation-only conditions "
                 "(pure pixel permutations); dotted = baseline", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _finish(fig, out_dir, "inspect_smoothing_control.png", show)
    tab = pd.DataFrame([{"group": g, "condition": lab,
                         **{f"{p}_{c}": m[f"{p}_{c}"] for p in ("raw", "dn") for c in ("ap50", "mean_iou", "merge_rate",
                                                                                          "split_rate", "n_pred")},
                         "n_gt": m["n_gt"], "diam_true": m["diam_true"]} for g, lab, m in rows])
    tab.to_csv(os.path.join(out_dir, "smoothing_control.csv"), index=False)
    print("\nSmoothing control (orientation floor = range over pure pixel permutations):")
    for path in ("raw", "dn"):
        if orient:
            v = [m[f"{path}_ap50"] for m in orient]
            print(f"  AP50 {path}: orientation floor {min(v):.3f} .. {max(v):.3f} | baseline "
                  f"{base[f'{path}_ap50']:.3f}")
    return tab


# --------------------------------------------------------------------------------------
# Plots from results/agg.csv (no models needed)
# --------------------------------------------------------------------------------------
def _load_agg(results_dir):
    path = os.path.join(results_dir, "agg.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run run_analysis first.")
    agg = pd.read_csv(path)
    agg["p"] = agg["params"].apply(json.loads)
    return agg


def plot_blur_relative_to_diameter(results_dir, out_dir, show):
    """Blur sigma vs. blur relative to the cell diameter (sigma / diam_true).

    Blur is applied AFTER resize in the pipeline, so resize 0.5 makes the same sigma twice as strong relative to the
    cells. If the metrics are governed by sigma / d, the curves of different resize factors collapse onto one curve
    in the right column but not in the left column."""
    agg = _load_agg(results_dir)
    sel = agg[agg.p.apply(lambda d: "blur" in d and set(d) <= {"blur", "resize"})].copy()
    if sel.empty:
        print("[csv] no blur / blur x resize rows in agg.csv")
        return None
    sel["sigma"] = sel.p.apply(lambda d: float(d["blur"]))
    sel["resize"] = sel.p.apply(lambda d: float(d.get("resize", 1.0)))
    sel["rel"] = sel["sigma"] / sel["diam_true"]
    base = agg[agg.p.apply(lambda d: d == {})]
    base = base.iloc[0] if len(base) else None
    rows = [("AP@0.5 without denoise", "raw_ap50"), ("AP@0.5 with denoise", "dn_ap50"),
            ("merge rate without denoise", "raw_merge_rate"), ("merge rate with denoise", "dn_merge_rate")]
    resizes = sorted(sel["resize"].unique())
    colors = {rz: SEQ(0.25 + 0.75 * i / max(len(resizes) - 1, 1)) for i, rz in enumerate(resizes)}
    fig, axes = _new_fig(len(rows), 2, (12, 3.0 * len(rows)))
    for r_i, (ttl, col) in enumerate(rows):
        for c_i, (xcol, xlabel) in enumerate((("sigma", "blur sigma [px] (absolute)"),
                                              ("rel", "blur sigma / cell diameter (relative)"))):
            ax = axes[r_i, c_i]
            _style_ax(ax)
            for rz in resizes:
                g = sel[sel.resize == rz].sort_values(xcol)
                ax.plot(g[xcol], g[col], marker="o", markersize=5, linewidth=1.5, color=colors[rz],
                        markeredgecolor=SURFACE, markeredgewidth=0.8, label=f"resize {rz:g}")
            if base is not None:
                ax.axhline(base[col], color=INK2, linestyle=":", linewidth=1)
            ax.set_title(f"{ttl} vs {'absolute' if c_i == 0 else 'relative'} sigma", fontsize=9)
            if xcol == "rel":
                ax.set_xscale("log")
                ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.3) if sel["rel"].min() * 0.8 <= t <= sel["rel"].max() * 1.2]
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{t:g}" for t in ticks])
                ax.xaxis.set_minor_formatter(NullFormatter())
            if r_i == len(rows) - 1:
                ax.set_xlabel(xlabel)
            if r_i == 0 and c_i == 1:
                ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle("Is blur governed by sigma / cell diameter?  (dotted = unchanged baseline)", color=INK, fontsize=12,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _finish(fig, out_dir, "inspect_blur_relative_diameter.png", show)
    print("\nSpearman rank correlation with the metric (|rho| larger = better single predictor):")
    out = []
    for ttl, col in rows:
        s_abs = spearmanr(sel["sigma"], sel[col]).correlation
        s_rel = spearmanr(sel["rel"], sel[col]).correlation
        out.append(dict(metric=col, rho_sigma_absolute=s_abs, rho_sigma_relative=s_rel, n_points=len(sel)))
        print(f"  {col:16s} absolute sigma: {s_abs:+.3f} | sigma/d: {s_rel:+.3f}  (n={len(sel)})")
    pd.DataFrame(out).to_csv(os.path.join(out_dir, "blur_relative_spearman.csv"), index=False)
    return sel


def plot_lpips_vs_blur(results_dir, out_dir, show):
    """Denoise path vs. input path for image similarity (LPIPS, SSIM) and AP over the blur sigma."""
    agg = _load_agg(results_dir)
    sel = agg[agg.p.apply(lambda d: d == {} or set(d) == {"blur"})].copy()
    sel["sigma"] = sel.p.apply(lambda d: float(d.get("blur", 0.0)))
    sel = sel.sort_values("sigma")
    if sel.empty:
        print("[csv] no blur rows in agg.csv")
        return None
    panels = [("LPIPS to clean image (lower = more similar)", "lpips_in", "lpips_dn"),
              ("SSIM to clean image (higher = more similar)", "ssim_in", "ssim_dn"),
              ("AP@0.5", "raw_ap50", "dn_ap50")]
    fig, axes = _new_fig(1, 3, (16, 4.2))
    for ax, (ttl, c_in, c_dn) in zip(axes[0], panels):
        _style_ax(ax)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("blur sigma [px] (0 = unchanged)")
        vals = np.concatenate([sel[c_in].to_numpy(float), sel[c_dn].to_numpy(float)])
        if not np.isfinite(vals).any():
            ax.text(0.5, 0.5, "not computed (pip install lpips)", transform=ax.transAxes, ha="center", color=MUTED)
            continue
        ax.plot(sel["sigma"], sel[c_in], color=RAW_C, linewidth=3.2, marker="o", markersize=6,
                markeredgecolor=SURFACE, label="without denoise / input", zorder=2)
        ax.plot(sel["sigma"], sel[c_dn], color=DN_C, linewidth=1.6, marker="o", markersize=3.5,
                markeredgecolor=SURFACE, label="with denoise", zorder=3)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Gaussian blur: the denoiser does not undo blur (blue and orange coincide for sigma >= 3)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _finish(fig, out_dir, "inspect_lpips_vs_blur.png", show)
    cols = ["sigma", "lpips_in", "lpips_dn", "ssim_in", "ssim_dn", "raw_ap50", "dn_ap50"]
    sel[cols].to_csv(os.path.join(out_dir, "lpips_vs_blur.csv"), index=False)
    return sel[cols]


# --------------------------------------------------------------------------------------
# Section definitions
# --------------------------------------------------------------------------------------
def _conditions(section):
    if section == "noise":
        return [("no noise (baseline)", {})] + [(f"Poisson scale {v:g}", {"noise": v}) for v in (80, 20, 5, 2, 1)]
    if section == "blur":
        return [("no blur (baseline)", {})] + [(f"blur σ={v:g}", {"blur": v}) for v in (1, 2, 3, 4, 6, 10)]
    if section == "diameter":
        return [(f"diameter x{v:g}" + (" (baseline)" if v == 1.0 else ""), {"diam_factor": v})
                for v in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]
    if section == "smoothing":
        return [("baseline", {}), ("downsample 2", {"down": 2}), ("downsample 3", {"down": 3}),
                ("blur 1 + downsample 2", {"blur": 1.0, "down": 2}), ("rotate 1° (interpolation only)", {"rotate": 1.0}),
                ("rotate 60°", {"rotate": 60.0}), ("rotate 135°", {"rotate": 135.0}),
                ("rot90 k=1 (pixel permutation)", {"rot90": 1})]
    if section == "combo":
        return [("baseline", {}), ("blur σ=6", {"blur": 6.0}), ("noise 1", {"noise": 1.0}),
                ("blur σ=6 + noise 1", {"blur": 6.0, "noise": 1.0}),
                ("blur σ=4 + down 6", {"blur": 4.0, "down": 6}),
                ("blur σ=4 + down 6 + noise 1", {"blur": 4.0, "down": 6, "noise": 1.0})]
    raise ValueError(section)


SECTION_TITLES = {
    "noise": "Poisson noise: small LPIPS (image looks fine) but extra objects and lower AP",
    "blur": "Gaussian blur: phase 1 = extra objects (sigma 2-4), phase 2 = fusion (sigma >= 5)",
    "diameter": "Wrong diameter: up to 1.5x unchanged, 2x merge peak, 3x collapse",
    "smoothing": "Does mild smoothing help? (downsample 3, rotation 60 / 135 vs. controls)",
    "combo": "Combinations: blur 6 + noise 1 (raw path finds nothing, denoise path merges)",
}


# --------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------
def run_inspection(img, mask, dn_model, seg_model, results_dir: str = "results", out_dir: str | None = None,
                   sections=None, seed: int = 0, cfg: AnalysisConfig | None = None, batch_size: int | None = None,
                   lpips_device: str | None = None, show: bool = True):
    """Create all inspection figures. Returns the Inspector (cache of all results) or None for CSV-only runs."""
    out_dir = out_dir or os.path.join(results_dir, "inspect")
    os.makedirs(out_dir, exist_ok=True)
    sections = list(sections) if sections else list(ALL_SECTIONS)
    unknown = [s for s in sections if s not in ALL_SECTIONS]
    if unknown:
        raise ValueError(f"Unknown section(s) {unknown}. Allowed: {ALL_SECTIONS}")
    model_sections = [s for s in sections if s != "csv"]
    insp = None
    if model_sections:
        if img is None or mask is None or dn_model is None or seg_model is None:
            raise ValueError("img, mask, dn_model and seg_model are required for the sections " + str(model_sections))
        insp = Inspector(img, mask, dn_model, seg_model, cfg=cfg, seed=seed, batch_size=batch_size,
                         lpips_device=lpips_device)
    all_results = []
    for sec in model_sections:
        if sec == "smoothing":
            all_results += _section_figures(insp, _conditions(sec), sec, SECTION_TITLES[sec], out_dir, show)
            plot_smoothing_control(insp, out_dir, show)
        else:
            all_results += _section_figures(insp, _conditions(sec), sec, SECTION_TITLES[sec], out_dir, show)
    if "csv" in sections:
        plot_blur_relative_to_diameter(results_dir, out_dir, show)
        plot_lpips_vs_blur(results_dir, out_dir, show)
    if all_results:
        _write_tables(all_results, out_dir)
    print(f"\nDone. Files in: {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        print("  ", f)
    return insp


def _write_tables(results, out_dir):
    cat_rows, met_rows, seen = [], [], set()
    for r in results:
        for path in ("raw", "dn"):
            cats, share = r["cats"][path], r["share"][path]
            row = dict(section=r["section"], condition=r["label"], path=path, n_pred=len(cats))
            for c in CATS:
                sel = share[cats == c] if len(cats) else np.array([])
                row[f"n_{c}"] = int(len(sel))
                row[f"median_cyto_share_{c}"] = float(np.median(sel)) if len(sel) else np.nan
            cat_rows.append(row)
        if (r["section"], r["label"]) not in seen:
            seen.add((r["section"], r["label"]))
            met_rows.append(dict(section=r["section"], condition=r["label"], **r["metrics"]))
    cat = pd.DataFrame(cat_rows)
    cat.to_csv(os.path.join(out_dir, "object_categories.csv"), index=False)
    pd.DataFrame(met_rows).to_csv(os.path.join(out_dir, "inspection_metrics.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nObject categories per condition (path: raw = without denoise, dn = with denoise):")
    print(cat[["section", "condition", "path", "n_pred", "n_matched", "n_merged", "n_extra", "n_poor",
               "median_cyto_share_matched", "median_cyto_share_extra"]].round(2).to_string(index=False))
