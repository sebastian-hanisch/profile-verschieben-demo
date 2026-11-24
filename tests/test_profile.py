"""Tagesprofile, Standorte und Kennzahlen (Handbeispiele 2 und 3 mit Normalverteilungen): Erwartungswerte aus der Handformel."""

import math
from statistics import NormalDist

import numpy as np
import pytest

import pvs_constants as C
import pvs_evaluation as E
import pvs_profile as P
import pvs_verteilung as V
from pvs_verteilung import mischung

PHI = NormalDist().cdf


def test_profil_hat_zwei_spitzen_und_masse_eins():
    p = P.profil()
    assert p.loc == (8.0, 18.0) and p.weight == (0.4, 0.6) and p.scale == (1.2, 1.2)
    assert V.mittel_streuung(p)[0] == pytest.approx(0.4 * 8 + 0.6 * 18)


def test_streuung_von_profil_und_mischung_nach_dem_gesetz_der_totalen_varianz():
    drittel = [1 / 3] * 3
    for s, erwartet in ((0.0, math.sqrt(24 + 1.2 ** 2)), (3.0, math.sqrt(24 + 1.2 ** 2 + 6)), (4.0, math.sqrt(24 + 1.2 ** 2 + 2 * 4 ** 2 / 3))):
        sd = V.mittel_streuung(mischung(P.standorte_versetzt(s), drittel))[1]
        assert sd == pytest.approx(erwartet, abs=1e-9)
    assert round(math.sqrt(24 + 1.2 ** 2), 3) == 5.044 and round(math.sqrt(24 + 1.2 ** 2 + 6), 3) == 5.607 and round(math.sqrt(24 + 1.2 ** 2 + 32 / 3), 3) == 6.009


def test_fensterdeckung_des_profils_ist_zwei_phi_von_zwei_minus_eins():
    assert P.fensterdeckung_mischung(P.profil()) == pytest.approx(2 * PHI(2) - 1, abs=1e-9)      # ± 2 Streuungen: 95,45 % (Fehlerfunktion ohne scipy: Abweichung etwa 1e-10)


def test_fensterdeckung_der_summe_bei_s_gleich_drei_nach_der_handformel():
    mix = mischung(P.standorte_versetzt(3.0), [1 / 3] * 3)
    deckung = P.fensterdeckung_mischung(mix)
    # je Spitze die eigene Kopie ganz (2Φ(2) − 1), die zwei Nachbarkopien nur mit dem Streifen Φ(4,5) − Φ(0,5) je Kopie (Abstand 3 Stunden, Fensterhalbbreite 2,4 = 2 Streuungen)
    hand = (1 / 3) * (2 * PHI(2) - 1) + (1 / 3) * 2 * (PHI(4.5) - PHI(0.5))
    assert deckung == pytest.approx(hand, abs=5e-4)               # die Ausläufer der Kopien im jeweils anderen Fenster (je etwa 6e-5) fehlen in der Handformel
    assert round(deckung, 4) == 0.5239


def test_spitzenstunde_ist_ein_intervall_von_einer_stunde():
    p = P.profil()
    hand = max(0.4 * (PHI(0.5 / 1.2) - PHI(-0.5 / 1.2)), 0.6 * (PHI(0.5 / 1.2) - PHI(-0.5 / 1.2)))   # die höhere Spitze (18 Uhr), Stunde mittig
    assert P.spitzenstunde_mischung(p) == pytest.approx(hand, abs=1e-6)


def test_masse_ausserhalb_der_linie_wird_beziffert():
    zeilen = {s: P.masse_ausserhalb(mischung(P.standorte_versetzt(s), [1 / 3] * 3)) for s in (0.0, 1.0, 2.0, 3.0, 4.0)}
    # Hand: Standort +s hat die Abendspitze bei 18 + s mit dem Anteil 0,6/3; über 24 liegt Φ((18 + s − 24)/1,2); Standort −s hat die Morgenspitze bei 8 − s mit dem Anteil 0,4/3; unter 0 liegt Φ((0 − (8 − s))/1,2)
    for s, wert in zeilen.items():
        hand = (0.6 / 3) * PHI((18 + s - 24) / 1.2) + (0.4 / 3) * PHI((0 - (8 - s)) / 1.2)
        assert wert == pytest.approx(hand, abs=3e-6)
    assert round(100 * zeilen[0.0], 2) == 0.0 and round(100 * zeilen[1.0], 2) == 0.0 and round(100 * zeilen[2.0], 3) == 0.009 and round(100 * zeilen[3.0], 2) == 0.12 and round(100 * zeilen[4.0], 2) == 0.96


def test_standorte_mit_morgenanteilen_haben_dieselben_spitzen_und_andere_anteile():
    st = P.standorte_anteile(0.1)
    assert [m.weight[0] for m in st] == pytest.approx([0.3, 0.4, 0.5]) and all(m.loc == (8.0, 18.0) for m in st)


def test_bruch_rundet_auf_tausendstel():
    assert str(P.bruch(0.025)) == "1/40" and str(P.bruch(0.1)) == "1/10"


def test_reihe2_bei_s_gleich_drei_hat_die_zahlen_der_kernaussage():
    r = E.reihe2(3.0, 0.02)
    assert (r["gipfel_mischung"], r["gipfel_bary"]) == (6, 2)
    assert r["streuung_mischung"] == pytest.approx(5.6071, abs=5e-4) and r["streuung_bary"] == pytest.approx(5.0438, abs=5e-4)
    assert r["deckung_summe"] == pytest.approx(0.5239, abs=5e-4) and r["deckung_bary"] == pytest.approx(2 * PHI(2) - 1, abs=1e-5)
    assert r["stunde_bary"] / r["stunde_summe"] == pytest.approx(2.72, abs=0.01)
    assert r["abstand_zum_profil"] < 1e-9                        # das Baryzentrum von Verschiebungen mit Mittel 0 ist das Profil


@pytest.mark.parametrize("s", [0.25, 1.0, 2.5, 3.1])
def test_baryzentrum_von_verschiebungen_ist_das_mittlere_profil_auch_ohne_gitterschritt(s):
    r = E.reihe2(s, 0.02)
    assert r["abstand_zum_profil"] < 1e-5                         # bei s ohne Vielfaches des Gitterschritts bis auf die Umkehrung auf dem Gitter
    assert r["gipfel_bary"] == 2


def test_gitter_ist_das_der_gipfelzaehlung():
    assert P.X_GITTER[0] == -6.0 and P.X_GITTER[-1] == pytest.approx(30.0) and len(P.X_GITTER) == 36001
    assert np.allclose(np.diff(P.X_GITTER), V.GITTER_SCHRITT)


def test_konstanten_der_fenster_und_der_spitzen():
    assert C.FENSTER == ((5.6, 10.4), (15.6, 20.4)) and C.LEER == (10.4, 15.6)
    assert all(abs((a + b) / 2 - mu) < 1e-12 and abs((b - a) / 2 - 2 * C.SIGMA) < 1e-12 for (a, b), mu in zip(C.FENSTER, C.SPITZEN))   # Mitte der Fenster = Spitze, Halbbreite = 2σ
