"""Randfälle: gleiche Profile, disjunkte Träger, ein Atomort je Profil, doppelte Standorte, t am Rand, s = 0, Masse außerhalb des Tages, Regler an den Grenzen."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import pvs_constants as C
import pvs_diskret as D
import pvs_evaluation as E
import pvs_profile as P
import pvs_verteilung as V
from pvs_verteilung import mischung


def _alle_zahlen_endlich(obj):
    if isinstance(obj, dict):
        return all(_alle_zahlen_endlich(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_alle_zahlen_endlich(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return bool(np.all(np.isfinite(obj)))
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


def test_gleiche_profile_baryzentrum_und_mischung_sind_das_profil(profil_a):
    assert D.quantilmittel([profil_a, profil_a, profil_a], [F(1, 3)] * 3) == profil_a
    assert D.mischung([profil_a, profil_a, profil_a], [F(1, 3)] * 3) == profil_a
    assert D.w2_quadrat(profil_a, profil_a) == 0


def test_gleiche_normalverteilungen_beide_wege_fallen_zusammen():
    r = E.reihe1(C.FALL1[0], 0.0, 0.5)
    assert r["w2_ab"] < 1e-9
    assert all(r["mischung"][t]["gipfel"] == r["verschiebung"][t]["gipfel"] == 1 for t in C.T1_SCHRITTE)
    art, text = E.verdikt1(C.FALL1[0], 0.0, r)
    assert art == "success" and "gleich" in text


def test_disjunkte_traeger_mischung_kostet_t_mal_d_quadrat_die_verschiebung_t_d_zum_quadrat():
    d = 50
    a, b = D.normiert([(0, 1)]), D.normiert([(d, 1)])
    for t in (F(1, 10), F(1, 4), F(1, 2), F(9, 10)):
        assert D.w2_quadrat(a, D.mischung([a, b], [1 - t, t])) == t * d ** 2
        assert D.w2_quadrat(a, D.quantilmittel([a, b], [1 - t, t])) == (t * d) ** 2
    # bei t = 1/2: Verhältnis der W2 ist √2
    assert D.w2_quadrat(a, D.mischung([a, b], [F(1, 2)] * 2)) / D.w2_quadrat(a, D.quantilmittel([a, b], [F(1, 2)] * 2)) == 2


def test_ein_einziger_atomort_je_profil():
    a, b = D.normiert([(3, 1)]), D.normiert([(9, 1)])
    assert D.quantilmittel([a, b], [F(1, 2)] * 2) == [(6, 1)]
    assert D.mischung([a, b], [F(1, 2)] * 2) == [(3, F(1, 2)), (9, F(1, 2))]
    assert D.masse_zwischen(D.quantilmittel([a, b], [F(1, 2)] * 2), 3, 9) == 1


def test_drei_standorte_mit_zwei_gleichen_sind_ein_baryzentrum_mit_gewichten_zwei_drittel_und_ein_drittel(profil_a, profil_b):
    assert D.quantilmittel([profil_a, profil_a, profil_b], [F(1, 3)] * 3) == D.quantilmittel([profil_a, profil_b], [F(2, 3), F(1, 3)])


def test_t_am_rand_in_den_reihen():
    for t in (0.0, 1.0):
        r = E.reihe1(C.FALL1[2], 4.0, t)
        assert r["t_sel"] == t and _alle_zahlen_endlich({k: v for k, v in r.items() if k not in ("a", "b")})
        mi = r["mischung"][t]
        assert mi["w2"] == pytest.approx(0.0 if t == 0 else r["w2_ab"], abs=1e-6)       # t = 0: der Anfang selbst; t = 1: das Ende


def test_s_gleich_null_alle_standorte_liegen_uebereinander():
    r = E.reihe2(0.0, 0.02)
    assert r["gipfel_mischung"] == r["gipfel_bary"] == 2
    assert r["streuung_mischung"] == pytest.approx(r["streuung_bary"], abs=1e-4)
    assert r["deckung_summe"] == pytest.approx(r["deckung_bary"], abs=1e-5) and r["stunde_summe"] == pytest.approx(r["stunde_bary"], abs=1e-5)


def test_masse_ausserhalb_des_tages_bei_s_gleich_vier():
    r = E.reihe2(4.0, 0.02)
    assert round(100 * r["ausserhalb"], 2) == 0.96
    assert mischung(P.standorte_versetzt(4.0), [1 / 3] * 3).loc[-1] == 22.0       # die Abendspitze des Standorts +s liegt bei 22 Uhr, ihr Ausläufer über 24


@pytest.mark.parametrize("fall", C.FALL1)
@pytest.mark.parametrize("d", [C.D1_MIN, C.D1_MAX])
def test_reihe1_an_den_reglergrenzen_ist_endlich(fall, d):
    r = E.reihe1(fall, d, 0.5)
    assert _alle_zahlen_endlich({k: v for k, v in r.items() if k not in ("a", "b")})
    art, text = E.verdikt1(fall, d, r)
    assert art in ("info", "warning", "success") and text


@pytest.mark.parametrize("s", [C.S2_MIN, C.S2_MAX])
@pytest.mark.parametrize("prom", [C.PROM2[0], C.PROM2[-1]])
@pytest.mark.parametrize("frage", C.FRAGE2)
def test_reihe2_und_verdikt2_an_den_reglergrenzen(s, prom, frage):
    r = E.reihe2(s, prom)
    assert _alle_zahlen_endlich({k: v for k, v in r.items() if k != "x"})
    art, text = E.verdikt2(frage, s, r, E.gipfel_gegen_s(prom))
    assert art in ("info", "warning") and text


@pytest.mark.parametrize("delta", [C.DELTA3_MIN, C.DELTA3_MAX])
def test_reihe3_an_den_reglergrenzen(delta):
    r, a = E.reihe3(delta), E.reihe3_atome(delta)
    assert _alle_zahlen_endlich({k: v for k, v in r.items() if k != "x"})
    assert a["leer_bary"] == 2 * P.bruch(delta)                          # Masse zwischen den Spitzen = w_max − w_min = 2δ
    art, text = E.verdikt3(delta, r, a)
    assert art in ("info", "warning") and text


def test_gipfelzaehlung_ohne_masse_und_mit_einem_punkt():
    assert V.gipfel_zaehlen(np.zeros(5), 0.02) == 0
    assert V.gipfel_zaehlen(np.array([1.0]), 0.02) == 1


def test_mix_mit_gewicht_null_ist_erlaubt_und_aendert_nichts():
    a = P.profil()
    m = mischung([a, P.profil(0.1)], [1.0, 0.0])
    x = np.linspace(0, 24, 200)
    assert np.allclose(V.pdf(m, x), V.pdf(a, x), rtol=1e-12)
