"""
Robustheitsanalyse für Cellpose3 (Denoise -> Segmentierung) unter Bildveränderungen.

Ablauf pro Experiment (= eine Kombination von Veränderungen):
  1. Augmentation mit der Klasse `Degradation` (degradation.py), feste Reihenfolge:
        normalize -> geometrische Ops (rescale/resize/rotate/rot90/flip/crop) -> blur -> downsample -> poisson
     Das Zwischenbild nach den geometrischen Ops ist das saubere "Referenzbild" (mit derselben Geometrie wie die
     Maske), gegen das LPIPS/SSIM/PSNR berechnet werden.
  2. Denoising (jeder Kanal einzeln) und Segmentierung mit und ohne Denoising.
  3. Metriken: mittlere beste IoU, AP@0.5/0.75/0.9, Merge-/Split-Raten, Objektzahl-Verhältnis, Boundary-F1,
     LPIPS, SSIM, PSNR.
  4. Plots (Linien pro Operation, Heatmaps für Paare, Balken für Teilmengen), CSV-Tabellen, Beispielbilder.

Aufruf im Notebook:
    from degradation_analysis import run_analysis
    res = run_analysis(img, mask, dn_model, seg_model, out_dir="results", n_seeds=3)
    res["agg"]    # DataFrame (Mittelwert/Std über Seeds) je Experiment
    res["runs"]   # alle Einzelläufe

ANNAHMEN (bitte prüfen):
  * Kanalbelegung im Eingabebild: Rot (Index 0) = Kern, Grün (Index 1) = Zytoplasma  -> AnalysisConfig.nuc_idx / cyto_idx
  * Cellpose-Kanalnummern sind 1-basiert (1=R, 2=G, 3=B).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries, mark_boundaries, relabel_sequential

from degradation import Degradation

try:
    from tqdm import tqdm
except ImportError:  # tqdm ist optional
    def tqdm(it, **kw):
        return it


# --------------------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------------------
@dataclass
class AnalysisConfig:
    nuc_idx: int = 0                 # Index des Kernkanals in img[..., c]   (ANNAHME: Rot)
    cyto_idx: int = 1                # Index des Zytoplasma-Kanals            (ANNAHME: Grün)
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_diam: float = 5.0            # Clipping des an Cellpose übergebenen Durchmessers
    max_diam: float = 300.0
    max_side: int = 3000             # Läufe mit größeren Bildern werden übersprungen (Speicher/Zeit)
    # Metrik-Parameter
    cover_frac: float = 0.5          # Merge/Split: Anteil, der innerhalb des anderen Objekts liegen muss
    min_frac: float = 0.1            # Merge/Split: Mindestanteil, um Mini-Überlappungen zu ignorieren
    boundary_tol: float = 2.0        # Boundary-F1: Toleranz in Pixeln
    img_norm: str = "clip"           # "clip": Werte auf [0,1] klippen | "own": je Bild/Kanal 1-99%-Normalisierung
    lpips_net: str = "alex"

    @property
    def cp_cyto(self) -> int:        # Cellpose-Kanalnummer (1-basiert)
        return self.cyto_idx + 1

    @property
    def cp_nuc(self) -> int:
        return self.nuc_idx + 1


# --------------------------------------------------------------------------------------
# Experimente
# --------------------------------------------------------------------------------------
IDENTITY = {"noise": None, "blur": None, "down": None, "resize": 1.0, "rescale": None, "rotate": 0.0,
            "rot90": 0, "flip": (0, 0), "crop": None, "diam_factor": 1.0}
OP_LABEL = {
    "noise": "Poisson-scale (kleiner = mehr Rauschen)", "blur": "Gauß-Blur σ [px]", "down": "Downsample-Faktor",
    "resize": "Resize-Faktor", "rescale": "Ziel-Zelldurchmesser [px]", "rotate": "Rotation [°]",
    "rot90": "rot90 (k)", "flip": "Flip", "crop": "Crop-Größe [px]",
    "diam_factor": "Durchmesser-Faktor (angegeben / wahr)",
}
PHOTOMETRIC = ("blur", "down", "noise")


def _is_identity(op, v) -> bool:
    if v is None:
        return True
    if op == "flip":
        return tuple(int(bool(a)) for a in v) == (0, 0)
    return v == IDENTITY[op] if IDENTITY[op] is not None else False


def canon(params: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items() if not _is_identity(k, v)}


def params_key(params: dict) -> str:
    return json.dumps(canon(params), sort_keys=True, default=str)


def is_stochastic(params: dict) -> bool:
    p = canon(params)
    return "noise" in p or "crop" in p


def val_label(op, v) -> str:
    if v is None:
        return "–"
    if op == "flip":
        h, vv = v
        return {(0, 0): "–", (1, 0): "h", (0, 1): "v", (1, 1): "hv"}[(int(h), int(vv))]
    if op == "crop":
        return f"{v[0]}²" if v[0] == v[1] else f"{v[0]}x{v[1]}"
    return f"{v:g}"


@dataclass
class Experiment:
    group: str
    label: str
    params: dict
    x_op: str | None = None
    x_val: object = None
    y_op: str | None = None
    y_val: object = None
    key: str = field(init=False)

    def __post_init__(self):
        self.params = canon(self.params)
        self.key = params_key(self.params)


def default_experiments(quick: bool = False) -> list[Experiment]:
    """Parameterreihen (single), Paar-Raster (grid), Severity-Leiter und Teilmengen-Ablation."""
    E: list[Experiment] = []

    def pick(vals):  # quick: jeden zweiten Wert (Rand bleibt erhalten)
        return vals[::2] + ([vals[-1]] if len(vals) % 2 == 0 else []) if quick else vals

    # --- 1) Einzeloperationen (Parameterreihen, aufsteigende Stärke; Identität = Referenzpunkt) ---
    singles = {
        "noise": [None, 80, 40, 20, 10, 5, 2, 1, 0.5],
        "blur": [None, 0.5, 1, 2, 3, 5, 8, 10],
        "down": [None, 2, 3, 4, 5, 6, 7],
        "resize": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "rescale": [10, 15, 20, 45, 60, 90],
        "rotate": [0, 15, 30, 45, 60, 90, 135, 180],
        "rot90": [0, 1, 2, 3],
        "flip": [(0, 0), (1, 0), (0, 1), (1, 1)],
        "crop": [(384, 384), (256, 256), (224, 224), (160, 160), (128, 128)],
        "diam_factor": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
    }
    for op, vals in singles.items():
        for v in pick(vals):
            E.append(Experiment(f"single:{op}", val_label(op, v), {op: v}, x_op=op, x_val=v))

    # --- 2) Paar-Raster (Heatmaps): x_op über Spalten, y_op über Zeilen ---
    grids = [
        ("noise", [None, 20, 5, 2, 1], "blur", [None, 1, 2, 4, 6]),
        ("blur", [None, 1, 2, 4], "down", [None, 2, 3, 5, 7]),
        ("noise", [None, 20, 5, 2, 1], "down", [None, 2, 3, 5, 7]),
        ("blur", [None, 1, 2, 4, 6], "diam_factor", [0.5, 0.75, 1.0, 1.5, 2.0]),
        ("noise", [None, 20, 5, 2, 1], "diam_factor", [0.5, 0.75, 1.0, 1.5, 2.0]),
        ("noise", [None, 20, 5, 2, 1], "rotate", [0, 30, 45, 90]),
        ("blur", [None, 1, 2, 4], "rotate", [0, 30, 45, 90]),
        ("blur", [None, 1, 2, 4], "resize", [0.5, 0.75, 1.0, 1.5, 2.0]),
    ]
    for a, va, b, vb in grids:
        va, vb = (pick(va), pick(vb)) if quick else (va, vb)
        for vy in vb:
            for vx in va:
                E.append(Experiment(f"grid:{a}x{b}", f"{a}={val_label(a, vx)}, {b}={val_label(b, vy)}",
                                    {a: vx, b: vy}, x_op=a, x_val=vx, y_op=b, y_val=vy))

    # --- 3) Severity-Leiter: Rauschen + Blur + Downsample gemeinsam steigend ---
    ladder = [(None, None, None), (20, 1, 2), (8, 2, 3), (3, 3, 4), (1, 5, 6)]
    for i, (n, b, d) in enumerate(ladder):
        E.append(Experiment("ladder", f"L{i}", {"noise": n, "blur": b, "down": d}, x_op="ladder", x_val=i))

    # --- 4) Teilmengen-Ablation (alle Kombinationen aus Rauschen / Blur / Downsample, 2 Stärkestufen) ---
    levels = {"mittel": dict(noise=5, blur=2, down=3), "stark": dict(noise=1, blur=4, down=6)}
    for lvl, p in levels.items():
        for r in (1, 2, 3):
            for combo in itertools.combinations(("noise", "blur", "down"), r):
                E.append(Experiment(f"subset:{lvl}", "+".join(combo), {k: p[k] for k in combo}))

    E.insert(0, Experiment("baseline", "unverändert", {}))
    return E


def build_steps(params: dict) -> tuple[list, list]:
    """Zerlegt Parameter in (geometrische Steps inkl. normalize, photometrische Steps) in fester Reihenfolge."""
    p = canon(params)
    geo = [("normalize", {})]
    if "rescale" in p:
        geo.append(("rescale_to_diameter", {"target": float(p["rescale"])}))
    if "resize" in p:
        geo.append(("resize", {"scale": float(p["resize"])}))
    if "rotate" in p:
        geo.append(("rotate", {"angle": float(p["rotate"])}))
    if "rot90" in p:
        geo.append(("rot90", {"k": int(p["rot90"])}))
    if "flip" in p:
        geo.append(("flip", {"horizontal": bool(p["flip"][0]), "vertical": bool(p["flip"][1])}))
    if "crop" in p:
        geo.append(("random_crop", {"size": tuple(p["crop"])}))
    photo = []
    if "blur" in p:
        photo.append(("gaussian_blur", {"sigma": float(p["blur"])}))
    if "down" in p:
        photo.append(("downsample", {"factor": int(p["down"])}))
    if "noise" in p:
        photo.append(("poisson_noise", {"scale": float(p["noise"])}))
    return geo, photo


# --------------------------------------------------------------------------------------
# Sample-Erzeugung
# --------------------------------------------------------------------------------------
@dataclass
class Sample:
    key: str
    seed: int
    params: dict
    img_aug: np.ndarray | None = None     # (H, W, C) degradiertes Eingabebild
    ref2: np.ndarray | None = None        # (H, W, 2) sauberes Referenzbild [Kern, Zytoplasma]
    mask: np.ndarray | None = None        # Ground Truth nach geometrischen Ops
    diam_true: float = np.nan
    diam_used: float = np.nan
    log: list = field(default_factory=list)
    skipped: str = ""


def build_sample(img, mask, params: dict, seed: int, cfg: AnalysisConfig) -> Sample:
    s = Sample(key=params_key(params), seed=seed, params=canon(params))
    p = canon(params)
    # Größe VOR dem Resize abschätzen, damit riesige Zwischenbilder gar nicht erst entstehen
    scale = float(p.get("resize", 1.0))
    if "rescale" in p:
        try:
            scale *= float(p["rescale"]) / Degradation.estimate_diameter(mask)
        except ValueError:
            pass
    if max(np.shape(img)[:2]) * scale > cfg.max_side:
        s.skipped = f"Bild würde {max(np.shape(img)[:2]) * scale:.0f}px Kantenlänge erreichen (max_side={cfg.max_side})"
        return s
    deg = Degradation(seed=seed)
    geo, photo = build_steps(params)
    img_g, mask_g = deg.apply(img, mask, geo)
    if mask_g is None or int(np.max(mask_g)) == 0:
        s.skipped = "keine Ground-Truth-Zellen nach der Augmentation"
        return s
    if max(img_g.shape[:2]) > cfg.max_side:
        s.skipped = f"Bild zu groß ({img_g.shape[:2]})"
        return s
    img_a, _ = deg.apply(img_g, mask_g, photo)
    s.img_aug = np.ascontiguousarray(img_a, dtype=np.float32)
    s.ref2 = np.ascontiguousarray(img_g[..., [cfg.nuc_idx, cfg.cyto_idx]], dtype=np.float32)
    s.mask = np.ascontiguousarray(mask_g)
    s.diam_true = Degradation.estimate_diameter(mask_g)
    factor = float(canon(params).get("diam_factor", 1.0))
    s.diam_used = float(np.clip(s.diam_true * factor, cfg.min_diam, cfg.max_diam))
    s.log = deg.log
    return s


# --------------------------------------------------------------------------------------
# Modelle ausführen (Denoise pro Kanal + Segmentierung, mit Batch-Größen-Suche)
# --------------------------------------------------------------------------------------
class ModelRunner:
    def __init__(self, dn_model, seg_model, cfg: AnalysisConfig, batch_size: int | None = None,
                 candidates=(256, 128, 64, 32, 16, 8, 4, 2, 1)):
        self.dn, self.seg, self.cfg = dn_model, seg_model, cfg
        self.candidates = (batch_size,) if batch_size else tuple(candidates)
        self.bi = 0
        self.list_mode = True

    @property
    def batch_size(self):
        return self.candidates[self.bi]

    def _run_list(self, imgs, diams, bs):
        cfg, n = self.cfg, len(imgs)
        out = self.dn.eval(list(imgs) + list(imgs),
                           channels=[[cfg.cp_cyto, 0]] * n + [[cfg.cp_nuc, 0]] * n,
                           diameter=list(diams) + list(diams), batch_size=bs)
        dn_imgs = [np.stack([np.squeeze(out[i]), np.squeeze(out[n + i])], axis=-1) for i in range(n)]  # [cyto, nuc]
        kw = dict(diameter=list(diams), batch_size=bs, flow_threshold=cfg.flow_threshold,
                  cellprob_threshold=cfg.cellprob_threshold)
        m_dn, _, _ = self.seg.eval(dn_imgs, channels=[[1, 2]] * n, **kw)
        m_raw, _, _ = self.seg.eval(list(imgs), channels=[[cfg.cp_cyto, cfg.cp_nuc]] * n, **kw)
        return list(m_raw), list(m_dn), dn_imgs

    def _run_single(self, img, diam, bs):
        cfg = self.cfg
        cyto = np.squeeze(self.dn.eval([img], channels=[cfg.cp_cyto, 0], diameter=diam, batch_size=bs)[0])
        nuc = np.squeeze(self.dn.eval([img], channels=[cfg.cp_nuc, 0], diameter=diam, batch_size=bs)[0])
        dn_img = np.stack([cyto, nuc], axis=-1)
        kw = dict(diameter=diam, batch_size=bs, flow_threshold=cfg.flow_threshold,
                  cellprob_threshold=cfg.cellprob_threshold)
        m_dn, _, _ = self.seg.eval(dn_img, channels=[1, 2], **kw)
        m_raw, _, _ = self.seg.eval(img, channels=[cfg.cp_cyto, cfg.cp_nuc], **kw)
        return m_raw, m_dn, dn_img

    def run(self, samples: list[Sample]):
        imgs = [s.img_aug for s in samples]
        diams = [s.diam_used for s in samples]
        while True:
            bs = self.batch_size
            try:
                if self.list_mode:
                    return self._run_list(imgs, diams, bs)
                res = [self._run_single(i, d, bs) for i, d in zip(imgs, diams)]
                return [r[0] for r in res], [r[1] for r in res], [r[2] for r in res]
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if "out of memory" in msg:
                    if self.bi + 1 >= len(self.candidates):
                        raise RuntimeError("Auch das kleinste batch_size passt nicht in den Speicher.") from e
                    self.bi += 1
                    if "torch" in sys.modules:
                        try:
                            sys.modules["torch"].cuda.empty_cache()
                        except Exception:  # noqa: BLE001
                            pass
                    print(f"[ModelRunner] Speicher knapp -> batch_size={self.batch_size}")
                elif self.list_mode and isinstance(e, (TypeError, ValueError, IndexError, AttributeError)):
                    print(f"[ModelRunner] Listen-Aufruf nicht möglich ({type(e).__name__}: {e}) -> Einzelbild-Modus")
                    self.list_mode = False
                else:
                    raise


# --------------------------------------------------------------------------------------
# Segmentierungs-Metriken
# --------------------------------------------------------------------------------------
def _relabel(m) -> np.ndarray:
    m = np.asarray(m)
    if m.size == 0:
        return m.astype(np.int64)
    return relabel_sequential(np.maximum(m, 0).astype(np.int64))[0].astype(np.int64)


def overlap_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """ov[i, j] = Pixelzahl, in der GT-Label i und Vorhersage-Label j zusammenfallen (0 = Hintergrund)."""
    ngt, npr = int(gt.max()), int(pred.max())
    idx = gt.ravel() * (npr + 1) + pred.ravel()
    return np.bincount(idx, minlength=(ngt + 1) * (npr + 1)).reshape(ngt + 1, npr + 1)


def boundary_f1(gt: np.ndarray, pred: np.ndarray, tol: float) -> float:
    bg, bp = find_boundaries(gt, mode="inner"), find_boundaries(pred, mode="inner")
    if not bg.any() and not bp.any():
        return float("nan")
    if not bg.any() or not bp.any():
        return 0.0
    d_to_gt, d_to_pred = distance_transform_edt(~bg), distance_transform_edt(~bp)
    prec, rec = float((d_to_gt[bp] <= tol).mean()), float((d_to_pred[bg] <= tol).mean())
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def segmentation_metrics(gt, pred, cfg: AnalysisConfig) -> dict:
    """gt/pred: Label-Bilder (0 = Hintergrund). Rückgabe siehe Schlüssel unten."""
    gt, pred = _relabel(gt), _relabel(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"Form von Ground Truth {gt.shape} und Vorhersage {pred.shape} unterscheidet sich")
    ngt, npr = int(gt.max()), int(pred.max())
    out = dict(n_pred=npr, ap50=np.nan, ap75=np.nan, ap90=np.nan, mean_iou=np.nan, count_ratio=np.nan,
               merge_rate=np.nan, split_rate=np.nan, n_merge=np.nan, n_split=np.nan, bf1=np.nan)
    if ngt == 0:
        return out
    out["count_ratio"] = npr / ngt
    out["bf1"] = boundary_f1(gt, pred, cfg.boundary_tol)
    if npr == 0:
        out.update(ap50=0.0, ap75=0.0, ap90=0.0, mean_iou=0.0, merge_rate=0.0, split_rate=0.0, n_merge=0, n_split=0)
        return out

    ov = overlap_matrix(gt, pred)
    inter = ov[1:, 1:].astype(np.float64)
    a_g, a_p = ov[1:, :].sum(1).astype(np.float64), ov[:, 1:].sum(0).astype(np.float64)
    iou = inter / (a_g[:, None] + a_p[None, :] - inter)

    out["mean_iou"] = float(iou.max(1).mean())          # beste Überlappung je GT-Zelle; ohne Treffer = 0
    ti, pi = linear_sum_assignment(-iou)                # 1:1-Zuordnung wie bei Cellpose/StarDist
    matched = iou[ti, pi]
    for name, th in (("ap50", 0.5), ("ap75", 0.75), ("ap90", 0.9)):
        tp = int((matched >= th).sum())
        out[name] = tp / (tp + (npr - tp) + (ngt - tp))  # AP = TP / (TP + FP + FN)

    frac_g, frac_p = inter / a_g[:, None], inter / a_p[None, :]
    covers = (frac_g >= cfg.cover_frac) & (frac_p >= cfg.min_frac)    # Vorhersage j deckt GT-Zelle i
    merged = covers.sum(0) >= 2                                       # eine Vorhersage deckt >= 2 GT-Zellen
    frags = (frac_p >= cfg.cover_frac) & (frac_g >= cfg.min_frac)     # Vorhersage j ist Teil von GT-Zelle i
    split = frags.sum(1) >= 2                                         # GT-Zelle in >= 2 Teile zerlegt
    out["n_merge"] = int(merged.sum())
    out["merge_rate"] = float(covers[:, merged].any(1).sum() / ngt) if merged.any() else 0.0
    out["n_split"] = int(split.sum())
    out["split_rate"] = float(split.sum() / ngt)
    return out


# --------------------------------------------------------------------------------------
# Bild-Metriken (LPIPS, SSIM, PSNR) auf den zwei Kanälen [Kern, Zytoplasma]
# --------------------------------------------------------------------------------------
def _prep01(a2: np.ndarray, mode: str) -> np.ndarray:
    a = np.asarray(a2, dtype=np.float32)
    if mode == "own":
        lo, hi = np.percentile(a, 1, axis=(0, 1)), np.percentile(a, 99, axis=(0, 1))
        a = (a - lo) / np.maximum(hi - lo, 1e-6)
    return np.clip(a, 0.0, 1.0)


def ssim_psnr(ref: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import structural_similarity
    ssims, psnrs = [], []
    for c in range(ref.shape[-1]):
        ssims.append(structural_similarity(ref[..., c], test[..., c], data_range=1.0))
        mse = float(np.mean((ref[..., c] - test[..., c]) ** 2))
        psnrs.append(100.0 if mse <= 0 else min(100.0, 10 * np.log10(1.0 / mse)))
    return float(np.mean(ssims)), float(np.mean(psnrs))


class LPIPSScorer:
    """LPIPS über das Paket `lpips`. Fehlt torch/lpips, liefert der Scorer NaN (Warnung einmalig)."""

    def __init__(self, net: str = "alex", device: str | None = None):
        self.ok = False
        try:
            import torch
            import lpips
        except ImportError:
            warnings.warn("torch/lpips nicht installiert -> LPIPS wird als NaN gespeichert (pip install lpips)")
            return
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device).eval()
        self.ok = True

    def _to_tensor(self, a2: np.ndarray):
        rgb = np.stack([a2[..., 0], a2[..., 1], np.zeros_like(a2[..., 0])], axis=-1)   # R=Kern, G=Zyto, B=0
        t = self.torch.from_numpy(2.0 * rgb - 1.0).permute(2, 0, 1)[None].float()
        return t.to(self.device)

    def __call__(self, ref01: np.ndarray, test01: np.ndarray) -> float:
        if not self.ok:
            return float("nan")
        with self.torch.no_grad():
            return float(self.model(self._to_tensor(ref01), self._to_tensor(test01)).mean())


def image_metrics(ref2, test2, lp: LPIPSScorer, mode: str) -> dict:
    if test2.shape != ref2.shape:
        import cv2
        test2 = np.stack([cv2.resize(test2[..., c], (ref2.shape[1], ref2.shape[0]), interpolation=cv2.INTER_LINEAR)
                          for c in range(test2.shape[-1])], axis=-1)
    r, t = _prep01(ref2, mode), _prep01(test2, mode)
    ssim, psnr = ssim_psnr(r, t)
    return dict(lpips=lp(r, t), ssim=ssim, psnr=psnr)


# --------------------------------------------------------------------------------------
# Auswertung eines Laufs
# --------------------------------------------------------------------------------------
def score_sample(s: Sample, m_raw, m_dn, dn_cn, cfg: AnalysisConfig, lp: LPIPSScorer) -> dict:
    rec = dict(key=s.key, seed=s.seed, params=json.dumps(s.params, default=str), skipped=0,
               diam_true=s.diam_true, diam_used=s.diam_used, n_gt=int(np.max(s.mask)),
               height=s.img_aug.shape[0], width=s.img_aug.shape[1])
    for prefix, m in (("raw_", m_raw), ("dn_", m_dn)):
        rec.update({prefix + k: v for k, v in segmentation_metrics(s.mask, m, cfg).items()})
    in2 = s.img_aug[..., [cfg.nuc_idx, cfg.cyto_idx]]
    dn2 = np.asarray(dn_cn)[..., ::-1]                        # [cyto, nuc] -> [nuc, cyto]
    for suffix, test in (("in", in2), ("dn", dn2)):
        for k, v in image_metrics(s.ref2, test, lp, cfg.img_norm).items():
            rec[f"{k}_{suffix}"] = v
    rec["dn_min"], rec["dn_max"] = float(np.min(dn2)), float(np.max(dn2))
    return rec


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    skip = {"seed"}
    num = [c for c in runs.columns if c not in skip and pd.api.types.is_numeric_dtype(runs[c])]
    g = runs.groupby("key")
    agg = g[num].mean().join(g[num].std(ddof=0).add_suffix("_std").fillna(0.0)).join(g.size().rename("n_runs"))
    agg["params"] = g["params"].first()
    return agg


# --------------------------------------------------------------------------------------
# Plot-Stil (validierte Palette: Blau/Orange, helle Fläche)
# --------------------------------------------------------------------------------------
RAW_C, DN_C = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_red_blue", ["#e34948", "#f0efec", "#2a78d6"])  # rot = schlechter, blau = besser


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)
    ax.title.set_fontsize(10)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _nan_note(ax, text):
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)


def _new_fig(nrows, ncols, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=SURFACE, squeeze=False)
    return fig, axes


def _finish(fig, out_dir, name, show):
    fig.savefig(os.path.join(out_dir, name), dpi=140, facecolor=SURFACE, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


SERIES_PANELS = [  # (Titel, Spalte ohne Denoise, Spalte mit Denoise, Hilfslinie)
    ("AP@0.5  (↑ besser)", "raw_ap50", "dn_ap50", None),
    ("mittlere beste IoU  (↑ besser)", "raw_mean_iou", "dn_mean_iou", None),
    ("Boundary-F1  (↑ besser)", "raw_bf1", "dn_bf1", None),
    ("Merge-Rate  (↓ besser)", "raw_merge_rate", "dn_merge_rate", None),
    ("Split-Rate  (↓ besser)", "raw_split_rate", "dn_split_rate", None),
    ("Objektzahl Vorhersage / GT  (ideal 1)", "raw_count_ratio", "dn_count_ratio", 1.0),
    ("LPIPS zum sauberen Original  (↓ besser)", "lpips_in", "lpips_dn", None),
    ("SSIM zum sauberen Original  (↑ besser)", "ssim_in", "ssim_dn", None),
    ("PSNR zum sauberen Original [dB]  (↑ besser)", "psnr_in", "psnr_dn", None),
]


def _series_fig(agg, exps, ticklabels, xlabel, title, out_dir, name, show):
    keys = [e.key for e in exps]
    base = agg.loc[params_key({})]
    d = agg.loc[keys]
    x = np.arange(len(keys))
    fig, axes = _new_fig(3, 3, (15, 11))
    fig.suptitle(title, color=INK, fontsize=13, y=0.995)
    for ax, (ttl, c_raw, c_dn, hline) in zip(axes.ravel(), SERIES_PANELS):
        _style_ax(ax)
        ax.set_title(ttl)
        ax.set_xticks(x)
        ax.set_xticklabels(ticklabels, rotation=0 if len(x) < 10 else 45)
        allv = np.concatenate([d[c_raw].to_numpy(float), d[c_dn].to_numpy(float)])
        if not np.isfinite(allv).any():
            _nan_note(ax, "nicht berechnet (LPIPS: pip install lpips)" if "lpips" in c_raw else "keine Werte")
            continue
        # ohne Denoise: dick im Hintergrund, mit Denoise: dünn darüber -> bei identischen Werten bleiben beide sichtbar
        for col, color, lab, lw, ms, z in ((c_raw, RAW_C, "ohne Denoise / Eingabe", 3.4, 7.5, 2),
                                           (c_dn, DN_C, "mit Denoise", 1.8, 4.0, 3)):
            y, sd = d[col].to_numpy(float), d[col + "_std"].to_numpy(float)
            ax.plot(x, y, color=color, linewidth=lw, marker="o", markersize=ms, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=lab, zorder=z)
            if np.any(sd > 0):
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.15, linewidth=0)
            ax.axhline(base[col], color=color, linewidth=1, linestyle=":", alpha=0.7)   # Baseline (unverändert)
        if hline is not None:
            ax.axhline(hline, color=AXIS, linewidth=1)
        lo, hi = float(np.nanmin(allv)), float(np.nanmax(allv))
        if "rate" in c_raw:                                   # Raten: bei 0 beginnen, keine 1e-17-Achse
            ax.set_ylim(-0.03 * max(hi, 0.1), max(hi * 1.1, 0.1))
        elif hi - lo < 1e-9:
            ax.set_ylim(lo - 0.05, hi + 0.05)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975), labelcolor=INK2)
    fig.text(0.5, 0.0, "Gepunktete Linien = unveränderte Baseline. Band = Std über Seeds. Merge-Rate = Anteil der GT-Zellen in verschmolzenen "
                       "Objekten, Split-Rate = Anteil zerteilter GT-Zellen.\nBildmetriken: blau = degradiertes Eingabebild, orange = denoistes "
                       "Bild (jeweils gegen das saubere Original).", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.955))
    _finish(fig, out_dir, name, show)


def _cell_text_color(cmap, norm, v):
    r, g, b, _ = cmap(norm(v))
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else INK


def _heat(ax, M, xl, yl, cmap, norm, title, fmt="{:.2f}"):
    ax.set_facecolor(SURFACE)
    ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color=_cell_text_color(cmap, norm, v))
    ax.set_xticks(range(len(xl)))
    ax.set_xticklabels(xl, fontsize=8, color=MUTED)
    ax.set_yticks(range(len(yl)))
    ax.set_yticklabels(yl, fontsize=8, color=MUTED)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, color=INK, fontsize=10)


def _grid_fig(agg, exps, out_dir, name, show):
    xo, yo = exps[0].x_op, exps[0].y_op
    xv = list(dict.fromkeys(e.x_val if not isinstance(e.x_val, tuple) else e.x_val for e in exps))
    yv = list(dict.fromkeys(e.y_val if not isinstance(e.y_val, tuple) else e.y_val for e in exps))
    pos = {(e.x_val, e.y_val): e.key for e in exps}

    def mat(col):
        M = np.full((len(yv), len(xv)), np.nan)
        for i, vy in enumerate(yv):
            for j, vx in enumerate(xv):
                M[i, j] = agg.loc[pos[(vx, vy)], col]
        return M

    xl, yl = [val_label(xo, v) for v in xv], [val_label(yo, v) for v in yv]
    rows = [  # (Zeilentitel, Spalte ohne, Spalte mit, Richtung: +1 = höher besser, -1 = niedriger besser, fmt)
        ("AP@0.5 ↑", "raw_ap50", "dn_ap50", +1, "{:.2f}"),
        ("Merge-Rate ↓", "raw_merge_rate", "dn_merge_rate", -1, "{:.2f}"),
        ("LPIPS zum Original ↓", "lpips_in", "lpips_dn", -1, "{:.2f}"),
    ]
    fig, axes = _new_fig(3, 3, (14, 11.5))
    fig.suptitle(f"{OP_LABEL[xo]}  ×  {OP_LABEL[yo]}", color=INK, fontsize=13, y=0.995)
    for r, (rt, c_raw, c_dn, sign, fmt) in enumerate(rows):
        A, B = mat(c_raw), mat(c_dn)
        both = np.concatenate([A[np.isfinite(A)], B[np.isfinite(B)]])
        if both.size == 0:
            for ax in axes[r]:
                ax.axis("off")
            axes[r, 1].text(0.5, 0.5, f"{rt}: nicht berechnet" + (" (pip install lpips)" if "lpips" in c_raw else ""),
                            transform=axes[r, 1].transAxes, ha="center", va="center", color=MUTED, fontsize=9)
            continue
        norm = Normalize(vmin=float(both.min()), vmax=float(both.max()) if both.max() > both.min() else float(both.min()) + 1e-6)
        first = "ohne Denoise" if r < 2 else "Eingabe (degradiert)"
        _heat(axes[r, 0], A, xl, yl, SEQ, norm, f"{rt} – {first}", fmt)
        _heat(axes[r, 1], B, xl, yl, SEQ, norm, f"{rt} – mit Denoise", fmt)
        improve = (B - A) if sign > 0 else (A - B)                # positiv = Denoise besser
        lim = max(float(np.nanmax(np.abs(improve))), 1e-6) if np.isfinite(improve).any() else 1.0
        _heat(axes[r, 2], improve, xl, yl, DIV, Normalize(-lim, lim), f"{rt} – Verbesserung durch Denoise (blau = besser)", "{:+.2f}")
        axes[r, 0].set_ylabel(OP_LABEL[yo], color=INK2, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(OP_LABEL[xo], color=INK2, fontsize=8)
    fig.text(0.5, 0.0, "Dunkler = höherer Wert (AP: besser, Merge-Rate/LPIPS: schlechter). Rechte Spalte: blau = Denoise verbessert, rot = verschlechtert.",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 0.96))
    _finish(fig, out_dir, name, show)


def _subset_fig(agg, groups, out_dir, show):
    levels = list(groups)
    fig, axes = _new_fig(3, len(levels), (7.5 * len(levels), 11))
    fig.suptitle("Teilmengen-Ablation: welche Kombination aus Rauschen / Blur / Downsample schadet am meisten?",
                 color=INK, fontsize=13, y=0.995)
    base = agg.loc[params_key({})]
    rows = [("AP@0.5  (↑ besser)", "raw_ap50", "dn_ap50"), ("Merge-Rate  (↓ besser)", "raw_merge_rate", "dn_merge_rate"),
            ("LPIPS zum Original  (↓ besser)", "lpips_in", "lpips_dn")]
    for j, g in enumerate(levels):
        exps = groups[g]
        x = np.arange(len(exps))
        for i, (ttl, c_raw, c_dn) in enumerate(rows):
            ax = axes[i, j]
            _style_ax(ax)
            if not np.isfinite(agg.loc[[e.key for e in exps], [c_raw, c_dn]].to_numpy(float)).any():
                ax.set_title(f"{ttl} – Stufe '{g.split(':')[1]}'")
                _nan_note(ax, "nicht berechnet (LPIPS: pip install lpips)")
                ax.set_xticks([])
                continue
            w = 0.38
            ax.bar(x - w / 2, agg.loc[[e.key for e in exps], c_raw], w, color=RAW_C, edgecolor=SURFACE, linewidth=1.5,
                   label="ohne Denoise / Eingabe")
            ax.bar(x + w / 2, agg.loc[[e.key for e in exps], c_dn], w, color=DN_C, edgecolor=SURFACE, linewidth=1.5,
                   label="mit Denoise")
            ax.axhline(base[c_raw], color=RAW_C, linestyle=":", linewidth=1)
            ax.axhline(base[c_dn], color=DN_C, linestyle=":", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels([e.label.replace("+", "\n+") for e in exps], fontsize=8)
            ax.set_title(f"{ttl} – Stufe '{g.split(':')[1]}'")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    _finish(fig, out_dir, "subsets.png", show)


# --------------------------------------------------------------------------------------
# Tabellen: größte Einbrüche und Interaktionen
# --------------------------------------------------------------------------------------
DROP_METRICS = [("ap50", +1), ("ap75", +1), ("mean_iou", +1), ("bf1", +1), ("merge_rate", -1), ("split_rate", -1)]


def top_drops(agg: pd.DataFrame, experiments: list[Experiment], n: int = 15) -> pd.DataFrame:
    base = agg.loc[params_key({})]
    first = {}
    for e in experiments:
        first.setdefault(e.key, e)
    rows = []
    for key, e in first.items():
        if e.group == "baseline" or key not in agg.index:
            continue
        for path in ("raw", "dn"):
            for m, sign in DROP_METRICS:
                v, b = agg.loc[key, f"{path}_{m}"], base[f"{path}_{m}"]
                if np.isfinite(v) and np.isfinite(b):
                    rows.append(dict(path=path, metric=m, group=e.group, label=e.label, baseline=b, value=v,
                                     drop=sign * (b - v)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.sort_values("drop", ascending=False).groupby(["path", "metric"]).head(n)
              .sort_values(["metric", "path", "drop"], ascending=[True, True, False]).reset_index(drop=True))


def interaction_table(agg: pd.DataFrame, experiments: list[Experiment]) -> pd.DataFrame:
    """Interaktion = (kombinierter Effekt) - (Summe der Einzeleffekte); für AP@0.5 und Merge-Rate."""
    rows, base = [], agg.loc[params_key({})]
    for e in experiments:
        if not e.group.startswith("grid:") or _is_identity(e.x_op, e.x_val) or _is_identity(e.y_op, e.y_val):
            continue
        kx = params_key({e.x_op: e.x_val})
        ky = params_key({e.y_op: e.y_val})
        if kx not in agg.index or ky not in agg.index:
            continue
        for path in ("raw", "dn"):
            for m in ("ap50", "merge_rate"):
                c = f"{path}_{m}"
                comb, sx, sy = agg.loc[e.key, c] - base[c], agg.loc[kx, c] - base[c], agg.loc[ky, c] - base[c]
                rows.append(dict(group=e.group, x_op=e.x_op, x_val=val_label(e.x_op, e.x_val), y_op=e.y_op,
                                 y_val=val_label(e.y_op, e.y_val), path=path, metric=m, combined=comb,
                                 sum_single=sx + sy, interaction=comb - (sx + sy)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Beispielbilder der größten Einbrüche
# --------------------------------------------------------------------------------------
def _rgb(a2_nuc_cyto):
    a = np.clip(np.asarray(a2_nuc_cyto, np.float32), 0, 1)
    return np.stack([a[..., 0], a[..., 1], np.zeros_like(a[..., 0])], axis=-1)


def plot_examples(img, mask, runner: ModelRunner, agg, experiments, cfg, out_dir, k=4, show=True):
    first = {}
    for e in experiments:
        first.setdefault(e.key, e)
    cand = [(agg.loc[key, "dn_ap50"], key) for key in first if first[key].group != "baseline"
            and key in agg.index and np.isfinite(agg.loc[key, "dn_ap50"])]
    worst = [key for _, key in sorted(cand)[:k]]
    keys = [params_key({})] + worst
    fig, axes = _new_fig(len(keys), 5, (20, 4.2 * len(keys)))
    for r, key in enumerate(keys):
        e = first[key]
        s = build_sample(img, mask, e.params, seed=0, cfg=cfg)
        if s.skipped:
            continue
        (m_raw,), (m_dn,), (dn_cn,) = runner.run([s])
        dn2 = np.asarray(dn_cn)[..., ::-1]
        in2 = s.img_aug[..., [cfg.nuc_idx, cfg.cyto_idx]]
        panels = [(_rgb(in2), None, "degradierte Eingabe"), (_rgb(dn2), None, "denoist"),
                  (_rgb(in2), m_dn, f"Seg. mit Denoise ({int(np.max(m_dn))} Zellen)"),
                  (_rgb(in2), m_raw, f"Seg. ohne Denoise ({int(np.max(m_raw))} Zellen)"),
                  (_rgb(s.ref2), s.mask, f"Ground Truth ({int(np.max(s.mask))} Zellen)")]
        for c, (im, lab, ttl) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(mark_boundaries(im, lab, color=(1, 1, 1)) if lab is not None else im)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(ttl, color=INK, fontsize=10)
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[r, 0].set_ylabel(f"{e.group}\n{e.label}", color=INK2, fontsize=9)
    fig.suptitle("Beispiele: Baseline und die Einstellungen mit den größten AP-Einbrüchen (mit Denoise)", color=INK,
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _finish(fig, out_dir, "examples_worst.png", show)


# --------------------------------------------------------------------------------------
# Hauptfunktion
# --------------------------------------------------------------------------------------
def make_plots(agg, experiments, out_dir, show=True):
    groups: dict[str, list[Experiment]] = {}
    for e in experiments:
        groups.setdefault(e.group, []).append(e)
    subset_groups = {}
    for g, exps in groups.items():
        if g.startswith("single:"):
            op = g.split(":")[1]
            _series_fig(agg, exps, [e.label for e in exps], OP_LABEL[op], f"Einzeloperation: {OP_LABEL[op]}",
                        out_dir, f"single_{op}.png", show)
        elif g == "ladder":
            _series_fig(agg, exps, [e.label for e in exps],
                        "Stufe (Rauschen + Blur + Downsample steigen gemeinsam)",
                        "Severity-Leiter: Rauschen + Blur + Downsample gemeinsam", out_dir, "ladder.png", show)
        elif g.startswith("grid:"):
            _grid_fig(agg, exps, out_dir, f"grid_{g.split(':')[1]}.png", show)
        elif g.startswith("subset:"):
            subset_groups[g] = exps
    if subset_groups:
        _subset_fig(agg, subset_groups, out_dir, show)


def run_analysis(img, mask, dn_model, seg_model, out_dir="degradation_results", n_seeds=3, experiments=None,
                 cfg: AnalysisConfig | None = None, chunk_size=16, batch_size=None, quick=False, n_examples=4,
                 show=True, resume=False, lpips_device=None):
    """Führt alle Experimente für EIN Sample (img, mask) aus. quick=True: 1 Seed und grobe Raster.

    resume=True setzt einen abgebrochenen Lauf fort (nur wenn Sample und Konfiguration identisch sind).
    Wechselst du DN_PATH/SEG_PATH, nimm einen neuen out_dir oder resume=False: die Modelle werden nicht verglichen.
    """
    cfg = cfg or AnalysisConfig()
    os.makedirs(out_dir, exist_ok=True)
    experiments = experiments or default_experiments(quick=quick)
    n_seeds = 1 if quick else n_seeds
    img = np.asarray(img)
    mask = np.asarray(mask)

    jobs, seen = [], set()
    for e in experiments:
        if e.key in seen:
            continue
        seen.add(e.key)
        for seed in (range(n_seeds) if is_stochastic(e.params) else [0]):
            jobs.append((e.key, e.params, seed))

    runs_path = os.path.join(out_dir, "runs.csv")
    meta_path = os.path.join(out_dir, "meta.json")
    import hashlib
    fp = hashlib.sha1(img.tobytes() + mask.tobytes() + json.dumps(cfg.__dict__, sort_keys=True).encode()).hexdigest()
    records = []
    if resume and os.path.exists(runs_path) and os.path.exists(meta_path):
        with open(meta_path) as f:
            same = json.load(f).get("fingerprint") == fp
        if same:
            records = pd.read_csv(runs_path).to_dict("records")
            done = {(r["key"], int(r["seed"])) for r in records}
            jobs = [j for j in jobs if (j[0], j[2]) not in done]
            print(f"[resume] {len(done)} Läufe aus {runs_path} übernommen, {len(jobs)} offen")
        else:
            print("[resume] Sample/Konfiguration weicht von den gespeicherten Läufen ab -> starte neu")
    with open(meta_path, "w") as f:
        json.dump({"fingerprint": fp}, f)
    print(f"{len(jobs)} Läufe ({len(seen)} verschiedene Einstellungen, bis zu {n_seeds} Seeds)")

    lp = LPIPSScorer(cfg.lpips_net, lpips_device)
    runner = ModelRunner(dn_model, seg_model, cfg, batch_size)
    for start in tqdm(range(0, len(jobs), chunk_size), desc="Chunks"):
        chunk = [build_sample(img, mask, p, sd, cfg) for _, p, sd in jobs[start:start + chunk_size]]
        ok = [s for s in chunk if not s.skipped]
        for s in chunk:
            if s.skipped:
                print(f"  übersprungen: {s.params} -> {s.skipped}")
                records.append(dict(key=s.key, seed=s.seed, params=json.dumps(s.params, default=str), skipped=1))
        if ok:
            m_raw, m_dn, dn_imgs = runner.run(ok)
            for s, a, b, c in zip(ok, m_raw, m_dn, dn_imgs):
                records.append(score_sample(s, a, b, c, cfg, lp))
        pd.DataFrame(records).to_csv(runs_path, index=False)

    runs = pd.DataFrame(records)
    base_key = params_key({})
    if base_key not in set(runs["key"]):
        raise RuntimeError("Baseline-Lauf fehlt (Experiment 'baseline' muss enthalten sein).")
    agg = aggregate(runs)
    agg.to_csv(os.path.join(out_dir, "agg.csv"))

    b = agg.loc[base_key]
    print(f"\nBaseline: AP@0.5 ohne/mit Denoise = {b['raw_ap50']:.3f} / {b['dn_ap50']:.3f}, "
          f"GT-Zellen = {int(b['n_gt'])}, Durchmesser = {b['diam_true']:.1f}px, "
          f"batch_size = {runner.batch_size}")
    print(f"Wertebereich des denoisten Bildes: [{b['dn_min']:.2f}, {b['dn_max']:.2f}]  "
          f"(für LPIPS/SSIM/PSNR wird auf [0,1] geklippt; weicht das stark von [0,1] ab, cfg.img_norm='own' testen)")

    make_plots(agg, experiments, out_dir, show)
    drops = top_drops(agg, experiments)
    drops.to_csv(os.path.join(out_dir, "top_drops.csv"), index=False)
    inter = interaction_table(agg, experiments)
    inter.to_csv(os.path.join(out_dir, "interactions.csv"), index=False)
    if not drops.empty:
        for path in ("raw", "dn"):
            print(f"\nGrößte AP@0.5-Einbrüche ({'mit' if path == 'dn' else 'ohne'} Denoise):")
            print(drops[(drops.metric == "ap50") & (drops.path == path)].head(8)
                  [["group", "label", "baseline", "value", "drop"]].to_string(index=False))
    if n_examples:
        plot_examples(img, mask, runner, agg, experiments, cfg, out_dir, k=n_examples, show=show)
    return dict(runs=runs, agg=agg, drops=drops, interactions=inter, experiments=experiments)


if __name__ == "__main__":
    print(__doc__)
