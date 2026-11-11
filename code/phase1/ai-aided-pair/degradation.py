import cv2
import numpy as np


def _per_channel(fn, img: np.ndarray):
    if img.ndim == 2:
        return fn(img)
    return np.stack([fn(np.ascontiguousarray(img[..., c])) for c in range(img.shape[2])], axis=-1)
 
 
def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return _per_channel(
        lambda c: cv2.GaussianBlur(c, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT),
        img.astype(np.float32),
    )
 
 
def _labels_via_float(mask: np.ndarray, fn):
    # OpenCV-Warps unterstuetzen kein int32 -> ueber float32 (exakt bis 2**24 Labels)
    return fn(mask.astype(np.float32)).astype(mask.dtype)
 
 
class Degradation:
    """Wendet Degradierungen auf ein Eingabebild an; geometrische Operationen auch auf die Maske.
 
    Beispiel:
        deg = Degradation(seed=0)
        img, mask = deg.normalize(img, mask)
        img, mask = deg.rotate(img, mask, angle=30)       # Maske wird mitgedreht
        img, mask = deg.poisson_noise(img, mask, scale=5) # Maske bleibt unveraendert
        # oder als Kette:
        img, mask = deg.apply(img, mask, [("normalize", {}), ("gaussian_blur", {"sigma": 3})])
 
    deg.log enthaelt die tatsaechlich verwendeten Parameter jedes Aufrufs (fuer die Auswertung).
    """
 
    OPS = (
        "normalize", "poisson_noise", "gaussian_blur", "downsample",
        "resize", "rescale_to_diameter", "rotate", "rot90", "flip", "random_crop",
    )
 
    def __init__(self, seed = None):
        self.rng = np.random.default_rng(seed)
        self.log: list[dict] = []
 
    def reset_log(self):
        self.log = []
 
    def _log(self, op: str, **params):
        self.log.append({"op": op, **params})
        
    def apply(self, img: np.ndarray, mask = None, steps = ()):
        """steps: Liste aus 'name' oder ('name', {param: wert}) mit name aus Degradation.OPS."""
        for step in steps:
            name, params = (step, {}) if isinstance(step, str) else step
            if name not in self.OPS:
                raise ValueError(f"Unbekannte Operation '{name}'. Erlaubt: {self.OPS}")
            img, mask = getattr(self, name)(img, mask, **params)
        return img, mask
 
    #photometrc
    def normalize(self, img, mask=None, lower: float = 1.0, upper: float = 99.0, clip_low: bool = True):
        """Pro Kanal: lower-Perzentil -> 0, upper-Perzentil -> 1, danach bei 0 geclippt
        (Paper, Methods: noetig, damit Poisson-Rauschen erzeugt werden kann)."""
 
        def norm(c: np.ndarray) -> np.ndarray:
            lo, hi = np.percentile(c, [lower, upper])
            if hi - lo < 1e-8:
                return np.zeros_like(c, dtype=np.float32)
            out = (c - lo) / (hi - lo)
            if clip_low:
                out = np.maximum(out, 0)
            return out.astype(np.float32)
 
        self._log("normalize", lower=lower, upper=upper, clip_low=clip_low)
        return _per_channel(norm, img.astype(np.float32)), mask
 
    def poisson_noise(self, img, mask=None, scale = None,alpha: float = 4.0, beta: float = 0.7):
        """Bild * scale -> Poisson-Stichprobe -> / scale.  Kleiner scale = staerkeres Rauschen.
 
        scale=None: scale ~ Gamma(Shape=alpha, Rate=beta) (Paper: alpha=4.0, beta=0.7 Denoising,
        0.1 Deblurring, 0.03 Upsampling; Test-Skalierungen 0.5-80.0).
        ANNAHME: 'varying values of beta' als Rate-Parameter interpretiert (Mittelwert alpha/beta).
        Ein scale pro Bild, fuer alle Kanaele gleich. Negative Werte werden auf 0 geclippt.
        """
        if scale is None:
            scale = float(self.rng.gamma(shape=alpha, scale=1.0 / beta))
        x = np.maximum(img, 0).astype(np.float64) * scale
        noisy = self.rng.poisson(x) / scale
        self._log("poisson_noise", scale=scale)
        return noisy.astype(np.float32), mask
 
    def gaussian_blur(self, img, mask=None, sigma = None,
                      sigma_range: tuple[float, float] = (1.0, 10.0), diameter_ratio: float = 1.0):
        """Isotroper Gauss-Blur. sigma=None: uniform random in sigma_range. Paper: Sigma scaled with cell-diameter -> diameter_ratio."""
        if sigma is None:
            sigma = float(self.rng.uniform(*sigma_range))
        self._log("gaussian_blur", sigma=sigma, diameter_ratio=diameter_ratio)
        return _blur(img, sigma * diameter_ratio), mask
 
    def downsample(self, img, mask=None, factor= None,factor_range: tuple[int, int] = (2, 7), blur_sigma = None):
        """Blur -> Subsampling -> bilineares upsampling to original size. Paper: factor 2-7, Blur-Sigma ~ 0.4 * factor."""
        if factor is None:
            factor = int(self.rng.integers(factor_range[0], factor_range[1] + 1))
        if blur_sigma is None:
            blur_sigma = 0.4 * factor
        h, w = img.shape[:2]
        small = np.ascontiguousarray(_blur(img, blur_sigma)[::factor, ::factor])
        out = _per_channel(lambda c: cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR), small)
        self._log("downsample", factor=factor, blur_sigma=blur_sigma)
        return out.astype(np.float32), mask
 
    # geometric (transform mask also)
    @staticmethod
    def estimate_diameter(mask: np.ndarray):
        counts = np.bincount(mask.ravel().astype(np.int64))[1:]
        counts = counts[counts > 0]
        if counts.size == 0:
            raise ValueError("Maske enthaelt keine Labels -> Durchmesser nicht bestimmbar.")
        return float(np.median(2.0 * np.sqrt(counts / np.pi)))
 
    def resize(self, img, mask=None, scale= None,scale_range: tuple[float, float] = (0.75, 1.25)):
        """scale=None: random in scale_range. (Cellpose-paper-aug.: after-noise: 0.75-1.25; before-noise: 0.5-2)."""
        if scale is None:
            scale = float(self.rng.uniform(*scale_range))
        h, w = img.shape[:2]
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        out = _per_channel(
            lambda c: cv2.resize(c, (new_w, new_h), interpolation=cv2.INTER_LINEAR),
            img.astype(np.float32),
        )
        if mask is not None:
            mask = _labels_via_float(mask, lambda m: cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST))
        self._log("resize", scale=scale, new_size=(new_h, new_w))
        return out, mask
 
    def rescale_to_diameter(self, img, mask=None, target: float = 30.0, diameter = None):
        """scale that cell-diameter = 'target' px. (30 cyto, 17 nuclei). diameter=None: estimate using mask"""
        if diameter is None:
            if mask is None:
                raise ValueError("Entweder diameter angeben oder eine Maske uebergeben.")
            diameter = self.estimate_diameter(mask)
        self._log("rescale_to_diameter", target=target, diameter=diameter)
        return self.resize(img, mask, scale=target / diameter)
 
    def rotate(self, img, mask=None, angle = None):
        """Bild bilinear, Mask nearest. angle=None: 0-360 random."""
        if angle is None:
            angle = float(self.rng.uniform(0, 360))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        out = _per_channel(
            lambda c: cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0),
            img.astype(np.float32),
        )
        if mask is not None:
            mask = _labels_via_float(mask, lambda m: cv2.warpAffine(
                m, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
        self._log("rotate", angle=angle)
        return out, mask
 
    def rot90(self, img, mask=None, k = None):
        """k*90°. k=None: 0-3 random."""
        if k is None:
            k = int(self.rng.integers(0, 4))
        self._log("rot90", k=k)
        out = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
        return out, (None if mask is None else np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1))))
 
    def flip(self, img, mask=None, horizontal = None, vertical = None):
        """None = 50 % prob."""
        horizontal = bool(self.rng.random() < 0.5) if horizontal is None else horizontal
        vertical = bool(self.rng.random() < 0.5) if vertical is None else vertical
        if horizontal:
            img = img[:, ::-1]
            mask = None if mask is None else mask[:, ::-1]
        if vertical:
            img = img[::-1]
            mask = None if mask is None else mask[::-1]
        self._log("flip", horizontal=horizontal, vertical=vertical)
        return np.ascontiguousarray(img), (None if mask is None else np.ascontiguousarray(mask))
 
    def random_crop(self, img, mask=None, size = (224, 224)):
        ch, cw = (size, size) if isinstance(size, int) else size
        h, w = img.shape[:2]
        pad_h, pad_w = max(0, ch - h), max(0, cw - w)
        if pad_h or pad_w:
            pad = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (img.ndim - 2)
            img = np.pad(img, pad, mode="constant")
            if mask is not None:
                mask = np.pad(mask, pad[:2], mode="constant")
            h, w = img.shape[:2]
        y = int(self.rng.integers(0, h - ch + 1))
        x = int(self.rng.integers(0, w - cw + 1))
        self._log("random_crop", y=y, x=x, size=(ch, cw))
        out = np.ascontiguousarray(img[y:y + ch, x:x + cw])
        return out, (None if mask is None else np.ascontiguousarray(mask[y:y + ch, x:x + cw]))