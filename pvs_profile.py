"""Tagesprofile, Standorte und die Kennzahlen der Experimente (Einheiten, DEMO-PLAYBOOK §1b); nur numpy.

Ein **Tagesprofil** ist die Verteilung der Nachfrage über die Uhrzeit mit Gesamtmasse 1 (Anteil an der Tagesnachfrage), hier zwei Normalverteilungen bei 8 und 18 Uhr.
Die Uhrzeit ist ein Kreis; hier wird sie als **Linie** behandelt (`masse_ausserhalb` gibt an, wie viel Masse dabei außerhalb von 0 bis 24 Uhr liegt)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

import pvs_constants as C
import pvs_diskret as D
import pvs_verteilung as V
from pvs_verteilung import Mix, mischung, normal, verschoben

X_GITTER = np.arange(-6000, 30001) * V.GITTER_SCHRITT           # −6 bis 30 Uhr in Schritten von 0,001 Stunden (Gipfelzählung)
H_ZENTREN = np.arange(0.0, C.TAG, C.STUNDEN_SCHRITT)            # Mitten der Stunde für die Spitzenstunde


def profil(morgenanteil: float = C.MORGENANTEIL, sigma: float = C.SIGMA) -> Mix:
    """Tagesprofil: Anteil `morgenanteil` bei 8 Uhr, der Rest bei 18 Uhr, je Normalverteilung mit Streuung `sigma`."""
    return Mix(C.SPITZEN, (sigma, sigma), (morgenanteil, 1.0 - morgenanteil))


def standorte_versetzt(s: float) -> list:
    """Drei Standorte mit demselben Profil, gegeneinander um −s, 0 und +s Stunden verschoben."""
    return [verschoben(profil(), d) for d in (-s, 0.0, s)]


def standorte_anteile(delta: float) -> list:
    """Drei Standorte mit denselben Spitzen und den Morgenanteilen 0,4 − δ, 0,4, 0,4 + δ."""
    return [profil(C.MORGENANTEIL + d) for d in (-delta, 0.0, delta)]


def bruch(x: float) -> Fraction:
    """Dezimalzahl mit höchstens drei Nachkommastellen als Bruch."""
    return Fraction(round(x * 1000), 1000)


def quantile_der_standorte(standorte, g: np.ndarray, u: np.ndarray) -> list:
    """Quantilfunktionen der Standorte auf den Stufen u (aus der Verteilungsfunktion auf dem Gitter g)."""
    return [V.quantile(V.cdf(m, g), g, u) for m in standorte]


def gipfel_mischung(standorte, prom: float) -> int:
    """Zahl der Gipfel der Mischung der Standorte (gleiche Gewichte), Dichte auf dem Gitter −6 bis 30 Uhr."""
    return V.gipfel_zaehlen(V.pdf(mischung(standorte, [1.0 / len(standorte)] * len(standorte)), X_GITTER), prom)


def fensterdeckung_mischung(mix: Mix) -> float:
    return float(sum(V.masse(mix, a, b) for a, b in C.FENSTER))


def fensterdeckung_quantile(q: np.ndarray, u: np.ndarray) -> float:
    return float(sum(V.masse_aus_quantilen(q, u, a, b) for a, b in C.FENSTER))


def spitzenstunde_mischung(mix: Mix) -> float:
    """Größter Anteil der Tagesnachfrage in einem Intervall von einer Stunde (Mitte der Stunde auf dem Raster 0,01 Stunden)."""
    return float((V.cdf(mix, H_ZENTREN + 0.5 * C.STUNDE) - V.cdf(mix, H_ZENTREN - 0.5 * C.STUNDE)).max())


def spitzenstunde_quantile(q: np.ndarray, u: np.ndarray) -> float:
    return float(V.masse_aus_quantilen(q, u, H_ZENTREN - 0.5 * C.STUNDE, H_ZENTREN + 0.5 * C.STUNDE).max())


def masse_ausserhalb(mix: Mix) -> float:
    """Masse außerhalb von 0 bis 24 Uhr (die Linien-Näherung der Uhrzeit)."""
    return float(V.cdf(mix, 0.0) + 1.0 - V.cdf(mix, C.TAG))


def atome_standorte(delta: Fraction) -> list:
    """Atom-Fassung der Standorte aus Experiment 3: jede Spitze eine Punktmasse bei 8 bzw. 18 Uhr (Brüche)."""
    return [D.tagesprofil_atome(Fraction(2, 5) + d) for d in (-delta, Fraction(0), delta)]
