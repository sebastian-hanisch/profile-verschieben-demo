"""Verdikte als reine Funktionen (DEMO-PLAYBOOK §12): Schwellen und jede Zahl im Text an handgesetzten Zeilen; je Preset die erwartete Verdikt-Zeile."""

from fractions import Fraction as F

import pytest

import pvs_constants as C
import pvs_evaluation as E


def _r1(gipfel_m, gipfel_v, streuung_m, streuung_v, ga=1, gb=1, sa=1.0, sb=1.0, w2_ab=4.0, w2_halb_m=2.4):
    """Handgesetzte Reihe für Experiment 1: je t ein Wert (alle neun gleich), t = ½ mit dem W2 der Mischung."""
    mi = {t: {"gipfel": gipfel_m, "streuung": streuung_m, "w2": w2_halb_m, "tv": 0.5} for t in C.T1_SCHRITTE}
    ve = {t: {"gipfel": gipfel_v, "streuung": streuung_v, "w2": t * w2_ab, "tv": 0.7} for t in C.T1_SCHRITTE}
    return {"mischung": mi, "verschiebung": ve, "gipfel_a": ga, "gipfel_b": gb, "streuung_a": sa, "streuung_b": sb, "w2_ab": w2_ab}


def test_kennzahlen1_zaehlt_zwischenschritte_von_hand():
    r = _r1(gipfel_m=2, gipfel_v=1, streuung_m=2.2, streuung_v=1.0)
    k = E.kennzahlen1(r)
    assert k["eingipfelig_mischung"] == 0 and k["eingipfelig_verschiebung"] == 9
    assert k["streuung_ok_mischung"] == 0 and k["streuung_ok_verschiebung"] == 9              # Vorgabe 1,0: 2,2 liegt 120 % daneben
    assert k["mehr_gipfel_mischung"] == 9 and k["mehr_gipfel_verschiebung"] == 0              # 2 > max(1, 1)
    assert k["verhaeltnis_halb"] == pytest.approx(2.4 / (0.5 * 4.0))                           # W2(A, Mischung_½) / (½ W2(A, B)) = 1,2


def test_streuung_toleranz_zehn_prozent():
    r = _r1(1, 1, streuung_m=1.09, streuung_v=0.91)                                        # knapp innerhalb von 10 %
    k = E.kennzahlen1(r)
    assert k["streuung_ok_mischung"] == 9 and k["streuung_ok_verschiebung"] == 9
    r = _r1(1, 1, streuung_m=1.11, streuung_v=0.89)
    k = E.kennzahlen1(r)
    assert k["streuung_ok_mischung"] == 0 and k["streuung_ok_verschiebung"] == 0


def test_verdikt1_gauss_nennt_jede_zahl_und_die_art_folgt_den_schwellen():
    art, text = E.verdikt1(C.FALL1[0], 4.0, _r1(2, 1, 2.2, 1.0))
    assert art == "warning"
    assert "0 von 9 Zwischenschritten eingipfelig" in text and "die Verschiebung in 9 von 9" in text and "Faktor 1,200" in text
    assert "Gauß N(0, 1) → N(4, 1)" in text and "Prominenz mindestens 2 %" in text
    art, text = E.verdikt1(C.FALL1[0], 0.5, _r1(1, 1, 1.0, 1.0))                                  # alles im Rahmen: keine Warnung
    assert art == "info"


def test_verdikt1_glocke_nennt_die_gipfelzahlen():
    art, text = E.verdikt1(C.FALL1[2], 4.0, _r1(3, 2, 2.0, 2.0, ga=1, gb=2))
    assert art == "warning" and "Anfang 1 Gipfel, Ende 2" in text
    assert "mehr Gipfel als beide Enden" in text and "3, 3, 3, 3, 3, 3, 3, 3, 3" in text and "2, 2, 2, 2, 2, 2, 2, 2, 2" in text
    assert "zwei Blöcke bei −4 und +4" in text


def test_laeufe_fasst_gleiche_gipfelzahlen_zusammen():
    s = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert E.laeufe(s, [2, 2, 6, 6, 5]) == "2 (s = 0 bis 0,25), 6 (s = 0,5 bis 0,75), 5 (s = 1)"
    assert E.laeufe(s, [2] * 5) == "2 (s = 0 bis 1)"


def test_format_helfer():
    assert E.zahl(-0.0004, 3) == "0,000" and E.zahl(-1.5, 1) == "−1,5" and E.zahl(2.0, 3) == "2,000"
    assert E.kurz(3.0, 2) == "3" and E.kurz(0.125) == "0,125" and E.kurz(-0.5, 1) == "−0,5" and E.kurz(0.0) == "0"
    assert E.prozent(0.2, 1) == "20,0 %" and E.prozent(0.0228, 2) == "2,28 %"
    assert E.bruch_text(F(1, 5)) == "1/5" and E.bruch_text(F(4)) == "4"


# ---------------------------------------------------------------- je Preset die erwartete Verdikt-Zeile


def _verdikte(preset):
    p = C.PRESETS[preset]
    s = E.Settings(**p)
    r1 = E.reihe1(s.fall1, s.d1, s.t1)
    r2 = E.reihe2(s.s2, s.prom2)
    r3, a3 = E.reihe3(s.delta3), E.reihe3_atome(s.delta3)
    return E.verdikt1(s.fall1, s.d1, r1), E.verdikt2(s.frage2, s.s2, r2, E.gipfel_gegen_s(s.prom2)), E.verdikt3(s.delta3, r3, a3)


def test_preset_drei_standorte_s_gleich_drei_meldet_die_kernzahlen():
    (a1, t1), (a2, t2), (a3, t3) = _verdikte("Drei Standorte, s = 3")
    assert "0 von 9 Zwischenschritten eingipfelig" in t1 and "Faktor 1,188" in t1
    assert a2 == "warning" and "6 Gipfel" in t2 and "2,72-mal" in t2 and "52,39 %" in t2 and "95,45 %" in t2
    assert "2 (s = 0 bis 2,5), 6 (s = 2,75 bis 3,25), 5 (s = 3,5 bis 4)" in t2
    assert a3 == "warning" and "Bruchteil 1/5" in t3 and "20,0 %" in t3 and "40/9 gegen 20/3" in t3


def test_preset_zwei_profile_versetzt_meldet_keine_gipfelzahl_sechs():
    (a1, t1), (a2, t2), (a3, t3) = _verdikte("Zwei Profile, 2 h versetzt")
    assert "Tagesprofil" in t1 and "dasselbe Profil um 2 Stunden versetzt" in t1
    assert "6 Gipfel" not in t2 and "6 Gipfel" not in t3 and t2.startswith("s = 0:")
    assert "Die Mischung hat in 0 von 9 Zwischenschritten mehr Gipfel als beide Enden" in t1 and "die Verschiebung in 0 von 9" in t1      # zwei Spitzen, 2 Stunden Versatz: weder Mischung noch Verschiebung haben mehr als die zwei Gipfel der Enden


def test_preset_unterschiedliche_morgenanteile_meldet_kein_gestalttreu():
    (_, t1), (_, t2), (a3, t3) = _verdikte("Unterschiedliche Morgenanteile")
    assert a3 == "warning" and "1/5" in t3
    for t in (t1, t2, t3):
        assert "gestalttreu" not in t and "Gestalttreue" not in t
    assert "erhält die Gestalt der Standorte nicht" in t3


def test_preset_glocke_zwei_bloecke_meldet_zwei_gegen_drei_gipfel_bei_t_null_komma_eins():
    (a1, t1), _, _ = _verdikte("Glocke → zwei Blöcke")
    assert a1 == "warning" and "Anfang 1 Gipfel, Ende 2" in t1
    assert "3, 3, 3, 3, 3, 3, 3, 3, 3" in t1 and "2, 2, 2, 2, 2, 2, 2, 2, 2" in t1


def test_verdikt3_bei_delta_null_meldet_keine_masse_im_leeren():
    r, a = E.reihe3(0.0), E.reihe3_atome(0.0)
    art, text = E.verdikt3(0.0, r, a)
    assert art == "info" and "δ = 0" in text and "ebenso viel" in text and a["leer_bary"] == 0


def test_verdikte_sind_reine_funktionen():
    r = E.reihe2(3.0, 0.02)
    g = E.gipfel_gegen_s(0.02)
    assert E.verdikt2(C.FRAGE2[0], 3.0, r, g) == E.verdikt2(C.FRAGE2[0], 3.0, r, g)
    assert E.verdikt2(C.FRAGE2[0], 3.0, r, g) != E.verdikt2(C.FRAGE2[1], 3.0, r, g)               # die Frage ändert das Verdikt


def test_verdikt2_frage_gestalt_nennt_mittleres_profil_und_bedingung():
    r = E.reihe2(3.0, 0.02)
    art, text = E.verdikt2(C.FRAGE2[1], 3.0, r, E.gipfel_gegen_s(0.02))
    assert art == "info" and "es ist das mittlere Profil" in text and "nur zeitlich verschoben" in text
    assert "5,607 Stunden" in text and "5,044 Stunden" in text and "unter 1e-9" in text


# ---------------------------------------------------------------- Verdikt 1: Faktor unter 1 und der Randwert D = 0 (Befund der Aussagenprüfung)


def _verdikt1_bei(fall, d, t=0.5):
    r = E.reihe1(fall, d, t)
    return E.verdikt1(fall, d, r), E.kennzahlen1(r)["verhaeltnis_halb"]


@pytest.mark.parametrize("d,faktor", [(0.0, "0,906"), (0.5, "0,915")])
def test_verdikt1_glocke_bei_kleinem_d_sagt_naeher_am_anfang_nicht_weiter(d, faktor):
    """Bei D = 0 und D = 0,5 liegt die Mischung näher am Anfang als die Verschiebung (Faktor unter 1); der Text darf nicht „weiter entfernt“ sagen."""
    (art, text), wert = _verdikt1_bei(C.FALL1[2], d)
    assert wert < 1.0
    assert f"das {faktor}-Fache des Abstands t·W2(A, B) der Verschiebung (Faktor {faktor})" in text
    assert "die Mischung liegt näher am Anfang." in text
    assert "weiter vom Anfang" not in text and "erlaubt" not in text


def test_verdikt1_glocke_bei_d_null_nennt_einen_block_nicht_minus_null_und_plus_null():
    (_, text), _ = _verdikt1_bei(C.FALL1[2], 0.0)
    assert "−0" not in text and "+0" not in text and "zwei Blöcke bei" not in text
    assert "ein Block bei 0 (Streuung 0,5; bei D = 0 fallen die zwei Blöcke zusammen)" in text
    assert "Anfang 1 Gipfel, Ende 1" in text
    (_, text), _ = _verdikt1_bei(C.FALL1[2], 0.5)
    assert "zwei Blöcke bei −0,5 und +0,5" in text and "Ende 1" in text                   # D = 0,5: die Blöcke überlappen noch, ein Gipfel


@pytest.mark.parametrize("d,faktor", [(1.0, "1,007"), (4.0, "1,237")])
def test_verdikt1_glocke_ab_d_eins_sagt_weiter_entfernt(d, faktor):
    (_, text), wert = _verdikt1_bei(C.FALL1[2], d)
    assert wert > 1.0 and f"(Faktor {faktor})" in text and "die Mischung liegt weiter vom Anfang entfernt." in text and "erlaubt" not in text


@pytest.mark.parametrize("fall", C.FALL1)
@pytest.mark.parametrize("d", [0.5, 1.0, 4.0, 8.0])
def test_verdikt1_richtung_des_textes_folgt_dem_faktor_in_jedem_fall(fall, d):
    (_, text), wert = _verdikt1_bei(fall, d)
    assert ("weiter vom Anfang entfernt" in text) == (round(wert, 3) > 1.0)
    assert ("näher am Anfang" in text) == (round(wert, 3) < 1.0)


def test_abstandssatz_drei_faelle_von_hand():
    assert "das 1,200-Fache" in E.abstandssatz(1.2) and "weiter vom Anfang entfernt" in E.abstandssatz(1.2)
    assert "das 0,800-Fache" in E.abstandssatz(0.8) and "näher am Anfang" in E.abstandssatz(0.8)
    assert "gleich weit vom Anfang entfernt" in E.abstandssatz(1.0004) and "gleich weit vom Anfang entfernt" in E.abstandssatz(0.9996)      # beide runden auf 1,000
    assert "Geodäte" not in E.abstandssatz(1.2)


def test_preset_verdikt1_nennt_den_faktor_je_preset():
    erwartet = {"Drei Standorte, s = 3": "(Faktor 1,188)", "Zwei Profile, 2 h versetzt": "(Faktor 1,065)",
                "Unterschiedliche Morgenanteile": "(Faktor 1,188)", "Glocke → zwei Blöcke": "(Faktor 1,237)"}
    for name, faktor in erwartet.items():
        (_, text), _, _ = _verdikte(name)
        assert faktor in text and "weiter vom Anfang entfernt" in text, name


def test_verdikt2_gestalt_nennt_die_bedeutung_von_mittlerer_phase():
    r = E.reihe2(3.0, 0.02)
    _, text = E.verdikt2(C.FRAGE2[1], 3.0, r, E.gipfel_gegen_s(0.02))
    assert "mittlerer Phase aussieht (hier: die Mitte der Versätze −s, 0, +s im W2-Sinn, auf der Linie)" in text


def test_verdikt3_atom_fassung_nennt_punktmassen_nicht_eine_stunde():
    r, a = E.reihe3(0.1), E.reihe3_atome(0.1)
    _, text = E.verdikt3(0.1, r, a)
    assert "jede Spitze eine Punktmasse bei 8 bzw. 18 Uhr" in text and "in einer Stunde" not in text
