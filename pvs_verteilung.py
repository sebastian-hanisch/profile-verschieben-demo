"""Verteilungen auf der Zeitachse als Mischungen von Normalverteilungen und ihre zwei Mitten (numpy-only, DEMO-PLAYBOOK §1b).

Die Einheiten:

* **Mischung** (englisch mixture) $(1 - t) P + t Q$: die Wahrscheinlichkeiten werden punktweise gemittelt. Für Mischungen von Normalverteilungen ist das wieder eine
  Mischung von Normalverteilungen (`mischung`).
* **Verschiebung** (englisch displacement interpolation, hier das Quantilmittel): $Q_t(u) = \\sum_k \\lambda_k Q_k(u)$, das eindimensionale Wasserstein-Baryzentrum
  (`quantilmittel`); für zwei Verteilungen ist es die Verschiebung jeder Masse um den Bruchteil $t$ ihres Weges im optimalen Plan.
* Quantilfunktionen werden aus der Verteilungsfunktion auf einem festen Gitter durch lineare Umkehrung gewonnen (`quantile`), die **Stufen** $u_i$ liegen in der Normalskala
  (`stufen`), damit die Ränder der Normalverteilungen fein aufgelöst sind und Integrale $\\int g(u)\\,\\mathrm{d}u$ als gewichtete Summen gelten.
* Die Dichte des Quantilmittels folgt aus $Q_t'(u) = \\sum_k \\lambda_k Q_k'(u)$ und $Q_k'(u) = 1 / f_k(Q_k(u))$: sie ist das gewichtete harmonische Mittel der Dichten an den
  gleichstufigen Quantilen (`dichte_quantilmittel`); kein Glätten, kein Histogramm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvs_normal import norm_cdf, norm_pdf

GITTER_SCHRITT = 0.001           # Stunden: Gitter der Dichten und Verteilungsfunktionen (Profile des Tages)
RAND_SIGMA = 9.0                 # das Gitter reicht bis loc ± 9 Streuungen
STUFEN_K = 20000                 # Zahl der Quantilstufen
STUFEN_Z = 6.5                   # die Stufen liegen bei Φ(z), |z| ≤ 6,5 (nicht erfasste Masse je Rand etwa 4e-11)


@dataclass(frozen=True)
class Mix:
    """Mischung von Normalverteilungen: Lagen, Streuungen und Gewichte (Summe 1)."""

    loc: tuple
    scale: tuple
    weight: tuple

    def __post_init__(self):
        if not (len(self.loc) == len(self.scale) == len(self.weight) > 0):
            raise ValueError("loc, scale und weight müssen gleich lang und nicht leer sein")
        if min(self.scale) <= 0 or min(self.weight) < 0 or abs(sum(self.weight) - 1.0) > 1e-9:
            raise ValueError("Streuungen müssen positiv sein, Gewichte nicht negativ mit Summe 1")


def normal(loc: float, scale: float) -> Mix:
    return Mix((float(loc),), (float(scale),), (1.0,))


def verschoben(m: Mix, d: float) -> Mix:
    """Alle Lagen um d Stunden verschoben."""
    return Mix(tuple(x + d for x in m.loc), m.scale, m.weight)


def mischung(mixes, weights) -> Mix:
    """Mischung Σ λ_k P_k von Mischungen: die Komponenten werden mit λ_k gewichtet aneinandergehängt."""
    w = np.asarray(weights, dtype=float)
    if len(w) != len(mixes) or abs(w.sum() - 1.0) > 1e-9 or w.min() < 0:
        raise ValueError("Gewichte müssen zu den Verteilungen passen, nicht negativ sein und die Summe 1 haben")
    loc, scale, weight = [], [], []
    for lam, m in zip(w, mixes):
        loc += list(m.loc)
        scale += list(m.scale)
        weight += [float(lam * g) for g in m.weight]
    return Mix(tuple(loc), tuple(scale), tuple(weight))


def pdf(m: Mix, x):
    x = np.asarray(x, dtype=float)
    return sum(g * norm_pdf((x - mu) / s) / s for mu, s, g in zip(m.loc, m.scale, m.weight))


def cdf(m: Mix, x):
    x = np.asarray(x, dtype=float)
    return sum(g * norm_cdf((x - mu) / s) for mu, s, g in zip(m.loc, m.scale, m.weight))


def mittel_streuung(m: Mix) -> tuple[float, float]:
    """Mittel und Standardabweichung (Gesetz der totalen Varianz)."""
    loc = np.asarray(m.loc)
    s = np.asarray(m.scale)
    g = np.asarray(m.weight)
    mu = float((g * loc).sum())
    var = float((g * (s ** 2 + (loc - mu) ** 2)).sum())
    return mu, var ** 0.5


def gitter(mixes, schritt: float = GITTER_SCHRITT) -> np.ndarray:
    """Gleichmäßiges Gitter, das alle Verteilungen bis ± 9 Streuungen enthält (ganzzahlige Vielfache des Schritts)."""
    lo = min(mu - RAND_SIGMA * s for m in mixes for mu, s in zip(m.loc, m.scale))
    hi = max(mu + RAND_SIGMA * s for m in mixes for mu, s in zip(m.loc, m.scale))
    i0 = int(np.floor(lo / schritt))
    i1 = int(np.ceil(hi / schritt))
    return np.arange(i0, i1 + 1) * schritt


def stufen(k: int = STUFEN_K, z_max: float = STUFEN_Z) -> tuple[np.ndarray, np.ndarray]:
    """Quantilstufen u_i = Φ(z_i) und Gewichte w_i = Φ(z_i + Δ/2) − Φ(z_i − Δ/2) (Mittelpunktsregel in der Normalskala z)."""
    delta = 2.0 * z_max / k
    z = -z_max + (np.arange(k) + 0.5) * delta
    return norm_cdf(z), norm_cdf(z + 0.5 * delta) - norm_cdf(z - 0.5 * delta)


def quantile(cdf_werte: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Quantile zu den Stufen u durch lineare Umkehrung der Verteilungsfunktion auf dem Gitter x (cdf_werte nicht fallend)."""
    return np.interp(u, cdf_werte, x)


def quantilmittel(qs, weights) -> np.ndarray:
    """Quantilmittel $\\sum_k \\lambda_k Q_k(u)$ gleichstufiger Quantilfunktionen (Zeilen von `qs`)."""
    return np.asarray(weights, dtype=float) @ np.asarray(qs, dtype=float)


def dichte_quantilmittel(mixes, weights, qs) -> np.ndarray:
    """Dichte des Quantilmittels an den Stellen $Q(u_i)$: $1 / \\sum_k \\lambda_k / f_k(Q_k(u_i))$ (gewichtetes harmonisches Mittel)."""
    w = np.asarray(weights, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        steigung = sum(lam / pdf(m, q) for lam, m, q in zip(w, mixes, qs))
        dichte = 1.0 / steigung
    return np.where(np.isfinite(dichte), dichte, 0.0)


def auf_gitter(positionen: np.ndarray, dichte: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Dichte, die in aufsteigenden Positionen gegeben ist (Quantilmittel), linear auf das Gitter x gelegt; außerhalb 0."""
    return np.interp(x, positionen, dichte, left=0.0, right=0.0)


def w2_quadrat(qa: np.ndarray, qb: np.ndarray, w: np.ndarray) -> float:
    """$W_2^2 = \\int (Q_a - Q_b)^2\\,\\mathrm{d}u$ als gewichtete Summe über die Stufen (Einheit: Stunden²)."""
    return float((w * (np.asarray(qa) - np.asarray(qb)) ** 2).sum() / w.sum())


def streuung_aus_quantilen(q: np.ndarray, w: np.ndarray) -> float:
    """Standardabweichung einer Verteilung aus ihrer Quantilfunktion."""
    mu = (w * q).sum() / w.sum()
    return float(((w * (q - mu) ** 2).sum() / w.sum()) ** 0.5)


def tv_auf_gitter(f: np.ndarray, g: np.ndarray, schritt: float = GITTER_SCHRITT) -> float:
    """Totalvariationsabstand $\\tfrac12 \\int |f - g|\\,\\mathrm{d}x$ zweier Dichten auf demselben Gitter."""
    return float(0.5 * np.abs(np.asarray(f) - np.asarray(g)).sum() * schritt)


def masse_aus_quantilen(q: np.ndarray, u: np.ndarray, a, b) -> np.ndarray:
    """Masse im Intervall (a, b] aus der Quantilfunktion (Umkehrung: Anteil der Stufen unter x); a, b dürfen Felder sein."""
    unten = np.interp(a, q, u, left=0.0, right=1.0)
    oben = np.interp(b, q, u, left=0.0, right=1.0)
    return oben - unten


def masse(m: Mix, a, b):
    """Masse einer Mischung im Intervall (a, b] aus der Verteilungsfunktion."""
    return cdf(m, b) - cdf(m, a)


# ---------------------------------------------------------------- Gipfel (Prominenz)


def lokale_maxima(f: np.ndarray) -> np.ndarray:
    """Indizes der lokalen Maxima; bei einer Hochebene gleicher Werte ihre Mitte (wie `scipy.signal.find_peaks` ohne weitere Bedingung)."""
    f = np.asarray(f, dtype=float)
    d = np.diff(f)
    nz = np.flatnonzero(d)                    # Stellen, an denen sich der Wert ändert
    steigt = d[nz] > 0
    k = np.flatnonzero(steigt[:-1] & ~steigt[1:])     # Anstieg, dann (nach einer Hochebene) Abfall
    return (nz[k] + 1 + nz[k + 1]) // 2


def prominenz(f: np.ndarray, p: int) -> float:
    """Prominenz des Maximums an der Stelle p: Höhe über dem höheren der beiden Täler zu den nächsten höheren Punkten (oder zum Rand)."""
    h = f[p]
    hoeher_links = np.flatnonzero(f[:p] > h)
    links = f[(hoeher_links[-1] + 1 if len(hoeher_links) else 0):p + 1].min()
    hoeher_rechts = np.flatnonzero(f[p + 1:] > h)
    rechts = f[p:(p + 1 + hoeher_rechts[0] if len(hoeher_rechts) else len(f))].min()
    return float(h - max(links, rechts))


def gipfel_zaehlen(f: np.ndarray, prom: float) -> int:
    """Zahl der Gipfel: lokale Maxima mit Prominenz mindestens `prom` mal der höchsten Dichte; Nullen an beiden Rändern angefügt."""
    f = np.concatenate([[0.0], np.asarray(f, dtype=float), [0.0]])
    schwelle = prom * f.max()
    if schwelle <= 0:
        return 0
    kandidaten = [int(p) for p in lokale_maxima(f) if f[p] >= schwelle]       # eine Prominenz über der Schwelle braucht mindestens diese Höhe
    return sum(1 for p in kandidaten if prominenz(f, p) >= schwelle)
