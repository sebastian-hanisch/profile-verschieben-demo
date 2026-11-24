"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und Hilfetexte.

Rechenaufwand am Regler-Maximum (alles numpy, ein Kern, Zeiten auf dem Entwicklungsrechner; auf der Cloud wahrscheinlich 2- bis 3-mal länger):
- Experiment 1: höchstens 45 000 Gitterpunkte (0,001 Stunden), 21 Werte von t; je Wert eine Umkehrung der Verteilungsfunktion (20 000 Stufen) und eine Dichte auf dem Gitter,
  O(Gitter) je Wert, zusammen unter 1 s.
- Experiment 2: 17 Werte von s × 36 001 Gitterpunkte (−6 bis 30 Uhr), 12 Komponenten je Dichte, Gipfelzählung mit Prominenz nur an den Kandidaten; etwa 0,5 s.
- Experiment 3: 9 Werte von δ, Quantile von drei Standorten (aus zwei gemeinsamen Komponenten-Verteilungsfunktionen), Brüche mit höchstens 6 Atomen; unter 0,5 s.
Alles in 1D und geschlossen oder auf dem Gitter; es gibt keine Iteration, keinen Zufall und keinen Zufalls-Seed."""

# Experiment 1: Verschieben oder mischen
FALL1 = ("Gauß, gleiche Streuung", "Gauß, Streuung 1 → 3", "Glocke → zwei Blöcke", "Tagesprofil, zwei Spitzen")
D1_MIN, D1_MAX, D1_SCHRITT = 0.0, 8.0, 0.5            # Stunden
T1_MIN, T1_MAX, T1_SCHRITT = 0.0, 1.0, 0.05
T1_SCHRITTE = tuple(round(0.1 * i, 1) for i in range(1, 10))    # die neun Zwischenschritte t = 0,1 … 0,9
T1_KURVE = tuple(round(0.05 * i, 2) for i in range(21))         # Stützstellen der Abstandskurven
PROM1 = 0.02                                          # Prominenz 2 % der höchsten Dichte (Gipfelzählung in Experiment 1)
STREUUNG_TOL = 0.10                                   # „Streuung ok“: höchstens 10 % neben (1 − t) s₀ + t s₁
BLOCK_SIGMA = 0.5                                     # Streuung der zwei Blöcke im Fall „Glocke → zwei Blöcke“
B_SIGMA_BREIT = 3.0                                   # Streuung des Endes im Fall „Streuung 1 → 3“

# Das Tagesprofil (Experimente 1 bis 3): Spitzen bei 8 und 18 Uhr, Anteile 0,4 und 0,6, Streuung 1,2 Stunden je Spitze
SPITZEN = (8.0, 18.0)
MORGENANTEIL = 0.4
SIGMA = 1.2
STANDORTE = 3                                         # drei Standorte gleichen Gewichts
GEWICHTE = (1 / 3, 1 / 3, 1 / 3)
FENSTER = ((5.6, 10.4), (15.6, 20.4))                 # 8 ± 2,4 und 18 ± 2,4 Uhr (±2 Streuungen)
LEER = (10.4, 15.6)                                   # zwischen den Spitzenfenstern
TAG = 24.0

# Experiment 2: Tagesprofile, Summe oder typisches Profil
S2_MIN, S2_MAX, S2_SCHRITT = 0.0, 4.0, 0.25           # Phasenversatz s in Stunden: Standorte bei −s, 0, +s
S2_WERTE = tuple(S2_SCHRITT * i for i in range(int(S2_MAX / S2_SCHRITT) + 1))
FRAGE2 = ("Summe der Standorte", "Gestalt bei verschobener Phase")
PROM2 = (0.005, 0.01, 0.02, 0.05, 0.10)               # Prominenz-Schalter (Anteil der höchsten Dichte)
S2_ZEILEN = (0.5, 1.0, 2.0, 3.0, 4.0)                 # Zeilen der Tabelle der README
STUNDE = 1.0                                          # Spitzenstunde: Intervall von einer Stunde
STUNDEN_SCHRITT = 0.01                                # Lage der Stunde: Raster 0,01 Stunden

# Experiment 3: unterschiedliche Morgenanteile
DELTA3_MIN, DELTA3_MAX, DELTA3_SCHRITT = 0.0, 0.2, 0.025   # Morgenanteile 0,4 − δ, 0,4, 0,4 + δ
DELTA3_WERTE = tuple(round(DELTA3_SCHRITT * i, 3) for i in range(int(round(DELTA3_MAX / DELTA3_SCHRITT)) + 1))

DEFAULTS = {
    "fall1": FALL1[0], "d1": 4.0, "t1": 0.5,
    "s2": 3.0, "frage2": FRAGE2[0], "prom2": 0.02,
    "delta3": 0.1,
}

PRESETS = {
    "Drei Standorte, s = 3": dict(DEFAULTS),
    "Zwei Profile, 2 h versetzt": {"fall1": FALL1[3], "d1": 2.0, "t1": 0.5, "s2": 0.0, "frage2": FRAGE2[0], "prom2": 0.02, "delta3": 0.0},
    "Unterschiedliche Morgenanteile": {"fall1": FALL1[0], "d1": 4.0, "t1": 0.5, "s2": 0.0, "frage2": FRAGE2[1], "prom2": 0.02, "delta3": 0.1},
    "Glocke → zwei Blöcke": {"fall1": FALL1[2], "d1": 4.0, "t1": 0.1, "s2": 0.0, "frage2": FRAGE2[0], "prom2": 0.02, "delta3": 0.0},
}
PRESET_HELP = {
    "Drei Standorte, s = 3": "Experiment 1: Gauß N(0, 1) nach N(4, 1) bei t = ½. Experiment 2: drei Standorte gegeneinander um −3, 0, +3 Stunden verschoben. Experiment 3: Morgenanteile 0,3 / 0,4 / 0,5.",
    "Zwei Profile, 2 h versetzt": "Experiment 1: ein Tagesprofil mit zwei Spitzen und dasselbe um 2 Stunden versetzt, t = ½. Experiment 2: kein Versatz zwischen den Standorten, Experiment 3: gleiche Anteile.",
    "Unterschiedliche Morgenanteile": "Experiment 3 mit Morgenanteilen 0,3 / 0,4 / 0,5. Experiment 1 steht auf Gauß N(0, 1) → N(4, 1) bei t = ½ (dem Standardfall), Experiment 2 auf s = 0 (kein Versatz).",
    "Glocke → zwei Blöcke": "Experiment 1: eine Glocke und zwei schmale Blöcke bei −4 und +4, t = 0,1. Experiment 2: kein Versatz (s = 0), Experiment 3: gleiche Anteile (δ = 0).",
}

# Farben (je Verfahren und Standort)
FARBE_MISCHUNG = "#d95f02"
FARBE_VERSCHIEBUNG = "#1b9e77"
FARBE_ENDEN = "#7570b3"
FARBE_STANDORT = ("#8da0cb", "#66a61e", "#e7298a")
FARBE_GRAU = "#999999"
