"""Normalverteilung ohne scipy: Fehlerfunktion, Verteilungsfunktion und Dichte (numpy-only, damit die App ohne scipy läuft).

Wörtliche Kopie von `erfc`, `norm_cdf` und `norm_pdf` samt Konstanten aus `mitte-im-raum-demo/mir_dist.py` (DEMO-PLAYBOOK 11a Nr. 11: kopierte Hilfsmodule nicht ändern);
`tests/test_oracle_profile_verschieben.py` vergleicht den Quelltext mit der Quelle (falls der Ordner vorhanden ist) und die Werte mit `math.erfc` und scipy."""

from __future__ import annotations

import math

import numpy as np

SQRT2 = math.sqrt(2.0)
SQRT_PI = math.sqrt(math.pi)
SERIES_LIMIT = 2.0       # bis hierhin die Reihe für erf, darüber der Kettenbruch für erfc
SERIES_TERMS = 45
FRACTION_TERMS = 60


# ---------------------------------------------------------------- Normalverteilung

def erfc(x):
    """Komplementäre Fehlerfunktion (Feld oder Zahl). |x| < 2: Reihe 2/√π·e^(−x²)·Σ 2^k x^(2k+1)/(2k+1)!! (nur positive Summanden), sonst Kettenbruch."""
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    small = ax < SERIES_LIMIT
    xs = np.where(small, ax, 0.0)
    term = xs.copy()
    total = xs.copy()
    for k in range(1, SERIES_TERMS):
        term = term * (2.0 * xs * xs / (2 * k + 1))
        total = total + term
    erf_small = 2.0 / SQRT_PI * np.exp(-xs * xs) * total
    xl = np.where(small, SERIES_LIMIT, ax)
    t = np.zeros_like(xl)
    for k in range(FRACTION_TERMS, 0, -1):
        t = (k / 2.0) / (xl + t)
    erfc_large = np.exp(-xl * xl) / SQRT_PI / (xl + t)
    positive = np.where(small, 1.0 - erf_small, erfc_large)
    return np.where(x >= 0, positive, 2.0 - positive)


def norm_cdf(x):
    """Verteilungsfunktion Φ der Standardnormalverteilung."""
    return 0.5 * erfc(-np.asarray(x, dtype=float) / SQRT2)


def norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

