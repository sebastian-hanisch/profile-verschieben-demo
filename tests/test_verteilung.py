"""Einheiten von `pvs_verteilung`: Mischungen von Normalverteilungen, Quantile, Stufen, Dichte des Quantilmittels, Gipfelzählung."""

import math

import numpy as np
import pytest

import pvs_profile as P
import pvs_verteilung as V
from pvs_verteilung import Mix, mischung, normal, verschoben


def test_mix_prueft_eingaben():
    with pytest.raises(ValueError):
        Mix((0.0,), (1.0, 2.0), (1.0,))
    with pytest.raises(ValueError):
        Mix((0.0,), (0.0,), (1.0,))                  # Streuung 0
    with pytest.raises(ValueError):
        Mix((0.0, 1.0), (1.0, 1.0), (0.7, 0.7))      # Summe 1,4
    with pytest.raises(ValueError):
        mischung([normal(0, 1), normal(1, 1)], [0.5, 0.6])


def test_mischung_haengt_komponenten_mit_gewichten_an():
    m = mischung([normal(0, 1), Mix((2.0, 4.0), (1.0, 1.0), (0.5, 0.5))], [0.5, 0.5])
    assert m.loc == (0.0, 2.0, 4.0) and m.weight == (0.5, 0.25, 0.25)


def test_dichte_hat_masse_eins_und_verteilungsfunktion_waechst_von_null_bis_eins():
    m = mischung([normal(0, 1), normal(5, 2)], [0.3, 0.7])
    x = V.gitter([m])
    assert abs(V.pdf(m, x).sum() * V.GITTER_SCHRITT - 1.0) < 1e-6
    c = V.cdf(m, x)
    assert c[0] < 1e-12 and c[-1] > 1 - 1e-12 and np.all(np.diff(c) >= 0)


def test_verteilungsfunktion_bei_den_lagen_einer_symmetrischen_mischung():
    m = mischung([normal(-3, 1), normal(3, 1)], [0.5, 0.5])
    assert abs(V.cdf(m, 0.0) - 0.5) < 1e-12                       # Symmetrie
    assert abs(V.cdf(m, 3.0) - (0.5 * 1.0 + 0.5 * 0.5)) < 1e-8   # F(3) = 1/2 · Φ(6) + 1/2 · Φ(0) = 0,75 bis auf 5e-10


def test_stufen_gewichte_summieren_sich_zur_masse_im_erfassten_bereich():
    u, w = V.stufen()
    assert len(u) == V.STUFEN_K and np.all(np.diff(u) > 0)
    assert abs(w.sum() - (1 - 2 * 4.02468e-11)) < 1e-9          # Masse außerhalb von ±6,5 je Rand etwa 4e-11
    assert abs((w * u).sum() / w.sum() - 0.5) < 1e-12           # symmetrisch um 1/2


def test_quantile_kehren_die_verteilungsfunktion_um():
    m = mischung([normal(8, 1.2), normal(18, 1.2)], [0.4, 0.6])
    g = V.gitter([m])
    u, _ = V.stufen()
    q = V.quantile(V.cdf(m, g), g, u)
    assert np.all(np.diff(q) >= 0)                                # Quantilfunktion monoton
    assert np.abs(V.cdf(m, q) - u).max() < 1e-7


def test_mittel_streuung_zweier_atomaehnlicher_mischung_von_hand():
    m = Mix((0.0, 10.0), (1.0, 1.0), (0.5, 0.5))                  # Mittel 5, Varianz 1 + 25 = 26
    mu, sd = V.mittel_streuung(m)
    assert mu == 5.0 and abs(sd - math.sqrt(26.0)) < 1e-12


def test_dichte_des_quantilmittels_eines_einzigen_standorts_ist_seine_dichte():
    m = mischung([normal(0, 1), normal(6, 1)], [0.5, 0.5])
    g = V.gitter([m])
    u, w = V.stufen()
    q = V.quantile(V.cdf(m, g), g, u)
    d = V.dichte_quantilmittel([m], [1.0], [q])
    assert np.allclose(d, V.pdf(m, q), rtol=1e-9, atol=0)


def test_w2_quadrat_und_streuung_aus_quantilen_einer_normalverteilung():
    u, w = V.stufen()
    g = V.gitter([normal(0, 2)])
    q = V.quantile(V.cdf(normal(0, 2), g), g, u)
    assert abs(V.streuung_aus_quantilen(q, w) - 2.0) < 2e-4
    q1 = V.quantile(V.cdf(normal(3, 2), g), g, u)
    assert abs(V.w2_quadrat(q, q1, w) - 9.0) < 1e-3                # reine Verschiebung um 3: W2² = 9


def test_tv_auf_gitter_getrennter_und_gleicher_dichten():
    a, b = normal(0, 1), normal(20, 1)
    x = V.gitter([a, b])
    assert abs(V.tv_auf_gitter(V.pdf(a, x), V.pdf(b, x)) - 1.0) < 1e-6
    assert V.tv_auf_gitter(V.pdf(a, x), V.pdf(a, x)) == 0.0


def test_masse_aus_quantilen_gegen_verteilungsfunktion():
    m = normal(0, 1)
    g = V.gitter([m])
    u, _ = V.stufen()
    q = V.quantile(V.cdf(m, g), g, u)
    assert abs(V.masse_aus_quantilen(q, u, -1.0, 1.0) - (V.cdf(m, 1.0) - V.cdf(m, -1.0))) < 1e-6
    assert V.masse_aus_quantilen(q, u, 10.0, 11.0) == 0.0


# ---------------------------------------------------------------- Gipfelzählung


def test_lokale_maxima_und_hochebenen():
    f = np.array([0, 1, 0, 2, 2, 2, 0, 3, 1, 1, 0.0])
    assert list(V.lokale_maxima(f)) == [1, 4, 7]                   # Hochebene 3..5: Mitte 4
    assert len(V.lokale_maxima(np.array([0, 1, 2, 3.0]))) == 0       # nur steigend: kein Maximum


def test_prominenz_von_hand():
    f = np.array([0, 5, 2, 4, 1, 3, 0.0])                          # Maxima bei 1 (Höhe 5), 3 (Höhe 4), 5 (Höhe 3)
    assert V.prominenz(f, 1) == 5.0                                # nichts Höheres: Tal ist der Rand 0
    assert V.prominenz(f, 3) == 4 - 2                              # links höher (5): Tal 2; rechts bis zum Rand: Tal 0 → max(2, 0)
    assert V.prominenz(f, 5) == 3 - 1                              # links höher (4): Tal 1


def test_gipfelzahl_haengt_an_der_prominenz():
    f = np.array([0, 5, 2, 4, 1, 3, 0.0])
    assert V.gipfel_zaehlen(f, 0.1) == 3                           # Prominenzen 5, 2, 2
    assert V.gipfel_zaehlen(f, 0.5) == 1                           # Schwelle 0,5 · 5 = 2,5: nur der höchste
    assert V.gipfel_zaehlen(np.zeros(10), 0.02) == 0


def test_handbeispiel_9_gipfel_der_mischung_gegen_s_bei_prominenz_2_prozent():
    erwartet = [2] * 11 + [6] * 3 + [5] * 13 + [6] + [4] * 5     # s = 0 bis 8 in Schritten von 0,25: 33 Werte
    s_werte = [0.25 * k for k in range(33)]
    assert len(erwartet) == len(s_werte) == 33
    assert [P.gipfel_mischung(P.standorte_versetzt(s), 0.02) for s in s_werte] == erwartet


def test_handbeispiel_9_empfindlichkeit_gegen_das_prominenzkriterium():
    for prom in (0.005, 0.01, 0.02, 0.05):
        assert P.gipfel_mischung(P.standorte_versetzt(3.0), prom) == 6
    assert P.gipfel_mischung(P.standorte_versetzt(3.0), 0.10) == 4          # bei 10 % nur vier
    for prom in (0.005, 0.01, 0.02, 0.05, 0.10):
        assert P.gipfel_mischung(P.standorte_versetzt(2.5), prom) == 2
        assert P.gipfel_mischung(P.standorte_versetzt(3.25), prom) == 6
    assert P.gipfel_mischung(P.standorte_versetzt(2.75), 0.02) == 6 and P.gipfel_mischung(P.standorte_versetzt(2.75), 0.05) == 2


def test_zwei_verschobene_profile_haben_hoechstens_vier_gipfel():
    for s in [0.25 * k for k in range(33)]:
        zwei = [P.profil(), verschoben(P.profil(), s)]
        assert V.gipfel_zaehlen(V.pdf(mischung(zwei, [0.5, 0.5]), P.X_GITTER), 0.02) <= 4


def test_ein_profil_hat_zwei_gipfel():
    assert V.gipfel_zaehlen(V.pdf(P.profil(), P.X_GITTER), 0.02) == 2
