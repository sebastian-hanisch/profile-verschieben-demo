"""Claims: jede Zahl und jeder Messsatz der README wird hier nachgerechnet (DEMO-PLAYBOOK §4, §11a).

Die Stücke mit Zahlen sind in `tests/readme_zeilen.py` aus dem Code gebildet (dieselben Funktionen wie die App) und müssen wörtlich in der README stehen. Alles ist deterministisch (kein Zufall):
die Zahlen sind keine Bänder, sondern Rundungen der Rechnung; Zählungen sind exakt, die Gipfelzahlen haben Abstand zu den Schwellen (kleinster Abstand einer Prominenz zum Kriterium: 5,5e-4 bei s = 3,5 und 1 %)."""

import os
import re
import subprocess
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import pvs_constants as C
import pvs_diskret as D
import pvs_evaluation as E
import pvs_profile as P
import pvs_verteilung as V

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readme_zeilen as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP_TEXT = (ROOT / "app.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", list(R.STUECKE))
def test_stuecke_der_readme_stehen_wortwoertlich_in_der_readme(name):
    assert R.STUECKE[name]() in README, name


def test_gleiche_tabelle_in_app_und_readme_gleichstand():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run(timeout=300)
    assert not at.exception
    assert any(R.tabelle_gleichstand() in m.value for m in at.markdown)


def test_kantenwortlaut_steht_in_readme_und_app():
    gleichstand = "Bei Gleichstand ist die Wahl des optimalen Plans für den Wert gleichgültig."
    kontrast = "die *Mischung* (Gerade der Totalvariation) gegen die *Verschiebung* (Gerade der W2-Geometrie)"
    for text in (README, APP_TEXT):
        assert gleichstand in text and kontrast in text
        assert "Die Demo „Abstand zwischen Verteilungen“ misst den Abstand zwischen zwei Verteilungen." in text


def test_konstanten_der_readme():
    assert "Spitzen bei 8 und 18 Uhr mit den Anteilen 0,4 und 0,6 und der Streuung 1,2 Stunden" in README
    assert C.SPITZEN == (8.0, 18.0) and C.MORGENANTEIL == 0.4 and C.SIGMA == 1.2 and C.STANDORTE == 3 and C.GEWICHTE == (1 / 3,) * 3
    assert "Gitter von 0,001 Stunden" in README and V.GITTER_SCHRITT == 0.001
    assert "20 000 Quantilstufen" in README and V.STUFEN_K == 20000
    assert "−6 bis 30 Uhr" in README and P.X_GITTER[0] == -6.0 and round(float(P.X_GITTER[-1]), 6) == 30.0
    assert "(10,4 bis 15,6 Uhr)" in README and C.LEER == (10.4, 15.6)
    assert "8 ± 2,4 und 18 ± 2,4 Uhr" in README and C.FENSTER == ((5.6, 10.4), (15.6, 20.4))
    assert "neun Zwischenschritten" in README and len(C.T1_SCHRITTE) == 9 and "21 Werte von t in Schritten von 0,05" in README and len(C.T1_KURVE) == 21
    assert "alle 17 Werte von $s$" in README and len(C.S2_WERTE) == 17


def test_zahlen_der_textsaetze_der_readme():
    r = E.reihe2(3.0, 0.02)
    assert "nur 52,39 % der Gesamtnachfrage statt 95,45 %" in README
    assert E.prozent(r["deckung_summe"], 2) == "52,39 %" and E.prozent(r["deckung_bary"], 2) == "95,45 %"
    assert "das 2,72-Fache" in README and E.zahl(r["stunde_bary"] / r["stunde_summe"], 2) == "2,72"
    assert "0,96 % der Masse der Summe außerhalb" in README and E.prozent(E.reihe2(4.0, 0.02)["ausserhalb"], 2) == "0,96 %"
    assert "$\\sqrt 2 = 1{,}414$" in README and round(2 ** 0.5, 3) == 1.414
    assert "höchstens vier Gipfel" in README                       # belegt in tests/test_verteilung.py
    assert "unter 1e-9 Stunden für alle 17 Werte" in README
    assert all(E.reihe2(s, 0.02)["abstand_zum_profil"] < 1e-9 for s in C.S2_WERTE)


def test_jeder_standort_hat_zwischen_den_spitzen_dieselbe_masse_unabhaengig_vom_anteil():
    assert "hat jeder Standort dort 2,28 %" in README
    for delta in C.DELTA3_WERTE:
        r = E.reihe3(delta)
        assert [E.prozent(x, 2) for x in r["leer_standorte"]] == ["2,28 %"] * 3 and E.prozent(r["leer_mischung"], 2) == "2,28 %"
        assert E.reihe3_atome(delta)["leer_bary"] == 2 * P.bruch(delta)           # Atome: exakt 2δ
    assert "exakt der Bruchteil $2\\delta$" in README


def test_w2_verhaeltnis_der_mischung_waechst_mit_d():
    werte = [E.kennzahlen1(E.reihe1(C.FALL1[0], d, 0.5))["verhaeltnis_halb"] for d in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)]
    assert werte == sorted(werte) and werte[-1] < 2 ** 0.5           # wächst mit D und bleibt unter dem Grenzwert √2 getrennter Träger
    assert "Das W2-Verhältnis der Mischung wächst mit $D$" in README


def test_gaussfaelle_verschiebung_ist_immer_eingipfelig_und_streuung_ok():
    for fall, d in [(C.FALL1[0], 1.0), (C.FALL1[0], 8.0), (C.FALL1[1], 0.0), (C.FALL1[1], 4.0)]:
        k = E.kennzahlen1(E.reihe1(fall, d, 0.5))
        assert k["eingipfelig_verschiebung"] == 9 and k["streuung_ok_verschiebung"] == 9


def test_gipfelzahl_des_baryzentrums_ist_in_experiment_2_immer_zwei():
    assert all(E.reihe2(s, p)["gipfel_bary"] == 2 for s in C.S2_WERTE for p in (C.PROM2[0], 0.02, C.PROM2[-1]))
    assert "hat es immer die zwei Gipfel des Profils" in README


def test_streuungsungleichung_bei_normalverteilungen_in_experiment_3():
    r = E.reihe3(0.1)
    assert r["streuung_bary"] <= np.mean(r["streuung_standorte"]) <= r["streuung_mischung"]
    assert r["ziel_bary"] < r["ziel_mischung"]


def test_testzahl_der_readme():
    ausgabe = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, check=True, env={**os.environ, "PYTHONUTF8": "1"}).stdout
    n = int(re.search(r"(\d+) tests? collected", ausgabe).group(1))
    assert f"**Testzahl:** {n} Tests" in README


VERBOTEN = ("Phase 0", "Pilot", "V1", "O6", "H 3", "Korrekturrunde", "Saatwert", "läuft lokal", "Stück ")


@pytest.mark.parametrize("name,text", [("README", README), ("app.py", APP_TEXT)], ids=["README", "app.py"])
def test_keine_planbegriffe_und_keine_verbotenen_woerter(name, text):
    for wort in VERBOTEN:
        assert wort not in text, (name, wort)
    assert not re.search(r"\b(trägt|tragen)\b", text), name
    assert not re.search(r"(?<!Zufalls-)Seed", text), name                       # „Seed“ steht nur als „Zufalls-Seed“
    assert not re.search(r"\bwasserstein\b", text, re.I) or "W1" in text or "W2" in text


def test_orakel_woerter_stehen_nicht_in_den_app_texten():
    assert "Orakel" not in APP_TEXT


@pytest.mark.parametrize("name,text", [("README", README), ("app.py", APP_TEXT)], ids=["README", "app.py"])
def test_wasserstein_steht_nie_ohne_ordnung(name, text):
    """„Wasserstein“ kommt nur in diesen Verbindungen vor: Wasserstein-Baryzentrum (Eigenname aus der Literatur, gemeint ist immer W2), Wasserstein-Abstand der Ordnung 2, der Titel der Quelle und der Satz über das Wort selbst, der diese Ausnahme nennt."""
    erlaubt = ("Wasserstein-Baryzentrum", "Wasserstein-Abstand der Ordnung 2", "Wasserstein space", "Wasserstein“ steht für sich allein nur im Namen „Wasserstein-Baryzentrum“")
    for m in re.finditer(r"Wasserstein", text):
        assert text[m.start():].startswith(erlaubt), text[max(0, m.start() - 30):m.start() + 60]


def test_w2_verhaeltnis_naehert_sich_bei_gleicher_streuung_dem_grenzwert_wurzel_zwei():
    """README Experiment 1: „D = 24: 1,369“ (außerhalb des Reglers); Band ± 5e-4, das Verhältnis wächst weiter mit D und bleibt unter √2."""
    werte = [E.kennzahlen1(E.reihe1(C.FALL1[0], d, 0.5))["verhaeltnis_halb"] for d in (8.0, 12.0, 16.0, 24.0)]
    assert werte == sorted(werte) and werte[-1] < 2 ** 0.5
    assert werte[-1] == pytest.approx(1.369, abs=5e-4) and werte[1] == pytest.approx(1.326, abs=5e-4)
    assert "(D = 24: 1,369, außerhalb des Reglers)" in README
    assert C.D1_MAX < 24.0
    # Zeile N(0, 1) → N(0, 3²) bei D = 0: kein Abstand der Lagen, das Verhältnis kommt allein von der Streuungsänderung
    assert E.kennzahlen1(E.reihe1(C.FALL1[1], 0.0, 0.5))["verhaeltnis_halb"] > 1.2
    assert "In der Zeile N(0, 1) → N(0, 3²) bei D = 0 gibt es keinen Abstand der Lagen" in README


def test_atom_fassung_faktor_drei_gleich_zahl_der_standorte_nur_bei_verschiedenen_orten():
    """README Experiment 2: „hier gleich der Zahl der Standorte, weil die sechs Orte der Mischung verschieden sind“ (Orte 5, 8, 11, 15, 18, 21)."""
    a = D.tagesprofil_atome(F(2, 5))
    mix = D.mischung([D.verschoben(a, d) for d in (-3, 0, 3)], [F(1, 3)] * 3)
    orte = [x for x, _ in mix]
    assert len(orte) == len(set(orte)) == 6 and sorted(orte) == [5, 8, 11, 15, 18, 21]
    assert max(m for _, m in a) / max(m for _, m in mix) == C.STANDORTE
    assert "weil die sechs Orte der Mischung verschieden sind" in README
    # bei überlappenden Orten (s = 10: 8 + 10 = 18) wäre der Faktor kleiner als die Zahl der Standorte
    mix10 = D.mischung([D.verschoben(a, d) for d in (-10, 0, 10)], [F(1, 3)] * 3)
    assert max(m for _, m in a) / max(m for _, m in mix10) < C.STANDORTE


def test_prominenz_schalter_der_readme_und_app_stehen_einzeln():
    """„sechs Gipfel bei s = 3“ gilt für die Schalter 0,5, 1, 2 und 5 % (jeder einzeln gezählt), nicht für 10 %."""
    zaehlung = {p: P.gipfel_mischung(P.standorte_versetzt(3.0), p) for p in C.PROM2}
    assert [zaehlung[p] for p in C.PROM2] == [6, 6, 6, 6, 4]
    for text in (README, APP_TEXT):
        assert "für die Prominenz-Schalter 0,5, 1, 2 und 5 %, nicht für 10 %" in text
        assert "0,5 bis 5 %" not in text


def test_gitterangaben_nennen_beide_gitter():
    """Gipfelzählung der Tagesprofile −6 bis 30 Uhr; Quantile bis ±9 Streuungen um die Komponenten (V.RAND_SIGMA)."""
    assert V.RAND_SIGMA == 9
    assert "die Quantile bis ±9 Streuungen um die Komponenten" in README
    assert "auf einem Gitter bis ±9 Streuungen um die Komponenten" in APP_TEXT
    g = V.gitter(P.standorte_versetzt(4.0))
    assert round(float(g[0]), 3) == -6.8 and round(float(g[-1]), 3) == 32.8          # s = 4: −6,8 bis 32,8 Uhr, nicht −6 bis 30


def test_atom_fassung_wortlaut_und_kernfrage_drei():
    for text in (README, APP_TEXT):
        assert "Spitzen nur eine Stunde Breite" not in text and "jede Spitze in einer Stunde" not in text
        assert "ins Leere" not in text and "im Leeren" not in text
    assert "Wann legt das Baryzentrum Masse zwischen die Spitzen?" in APP_TEXT
    assert "Liegt jede Spitze als Punktmasse bei 8 bzw. 18 Uhr (Atom-Fassung), liegt bei diesen Anteilen genau ein Fünftel der Masse des Baryzentrums dazwischen" in APP_TEXT
    assert E.reihe3_atome(0.1)["leer_bary"] == F(1, 5)


def test_literaturangabe_agueh_carlier_nennt_nichtatomar_und_atome_in_readme_und_app():
    for text in (README, APP_TEXT):
        assert "nichtatomar" in text and "Linearprogramm" in text and "10.12.2010" in text and "17.08.2010" in text
    assert "für Punktmassen (Atom-Fassung) steht das dort nicht" in README
    assert "„Mittlere Phase“ heißt hier die Mitte der Versätze" in README and "hier: die Mitte der Versätze −s, 0, +s im W2-Sinn" in APP_TEXT


def test_geodaeten_wortlaut_ohne_schranke():
    """Der Faktor ist ein gemessenes Verhältnis, keine Schranke: „erlaubt“ steht nirgends im Wortlaut der Geodäte."""
    assert "als die Geodäte erlaubt" not in README and "als die Geodäte erlaubt" not in APP_TEXT
