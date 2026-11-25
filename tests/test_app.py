"""Oberfläche mit Streamlits AppTest: Standard, jedes Preset, Permalink (Begrenzen, Einrasten), jeder Regler am Minimum und Maximum, alle Maxima zugleich.

AppTest sieht keine Plotly-Darstellung und keine KaTeX-Fehler (nur im Browser); geprüft werden Ausnahmen, Verdikte, Tabellen und die Zahl der Diagramme."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import pvs_constants as C
import pvs_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))
REGLER_ARTEN = ("slider", "select_slider", "selectbox", "radio")


def _app(**query) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    for k, v in query.items():
        at.query_params[k] = str(v)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} fehlt")


def _verdikte(at):
    return [e.value for art in ("info", "warning", "success") for e in getattr(at, art) if "Zwischenschritten" in e.value or "Stunden, Frage" in e.value or e.value.startswith(("δ =", "s = 0:"))]


def test_standard_laedt_ohne_ausnahme_mit_sechs_diagrammen_und_drei_verdikten():
    at = _app()
    assert len(at.get("plotly_chart")) == 6
    texte = _verdikte(at)
    assert any("0 von 9 Zwischenschritten eingipfelig" in t for t in texte)
    assert any("Frage „Summe der Standorte“" in t and "6 Gipfel" in t for t in texte)
    assert any(t.startswith("δ = 0,1:") and "1/5" in t for t in texte)


def test_kernfragen_nennen_die_zahlen_der_drei_standardfaelle():
    at = _app()
    info = " ".join(e.value for e in at.info)
    assert "Faktor 1,188" in info and "52,39 %" in info and "2,72-fach" in info and "20,0 %" in info


def test_expander_stehen_unter_ihren_titeln_und_die_grenzen_sind_in_der_app():
    at = _app()
    titel = [e.label for e in at.expander]
    assert "⚠️ Annahmen, Grenzen und bekannte Probleme" in titel and "📐 Mathematische Formulierung" in titel
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    tabelle = " ".join(m.value for m in at.markdown)
    assert "Bei Gleichstand ist die Wahl des optimalen Plans für den Wert gleichgültig." in tabelle
    assert "die *Mischung* (Gerade der Totalvariation) gegen die *Verschiebung* (Gerade der W2-Geometrie)" in tabelle


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_durch_und_setzt_alle_regler(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, wert in C.PRESETS[name].items():
        assert _regler(at, key).value == wert, (name, key)
    assert len(at.get("plotly_chart")) == 6 and _verdikte(at)


def test_permalink_uebernimmt_werte_begrenzt_sie_und_rastet_ein():
    at = _app(f1="Glocke → zwei Blöcke", d1=99, t1=0.33, s2=-5, q2="Gestalt bei verschobener Phase", p2=0.04, d3=0.11)
    assert _regler(at, "fall1").value == C.FALL1[2]
    assert _regler(at, "d1").value == C.D1_MAX                      # 99 auf die Obergrenze begrenzt
    assert _regler(at, "t1").value == 0.35                          # 0,33 auf die Stufe 0,05 gerundet
    assert _regler(at, "s2").value == C.S2_MIN                      # −5 auf die Untergrenze
    assert _regler(at, "frage2").value == C.FRAGE2[1]
    assert _regler(at, "prom2").value == 0.05                       # 0,04 auf die nächste Auswahl (0,05) eingerastet
    assert _regler(at, "delta3").value == 0.1                       # 0,11 auf die Stufe 0,025 gerundet


def test_permalink_ignoriert_ungueltige_werte_und_die_adresszeile_spiegelt_den_zustand():
    at = _app(f1="gibt es nicht", d1="abc")
    assert _regler(at, "fall1").value == C.DEFAULTS["fall1"] and _regler(at, "d1").value == C.DEFAULTS["d1"]
    assert at.query_params["d1"] == [str(C.DEFAULTS["d1"])] or at.query_params["d1"] == str(C.DEFAULTS["d1"])


def _alle_regler_grenzen():
    for key, spec in P.SETTING_SPECS.items():
        for kante in ("min", "max"):
            yield key, kante


@pytest.mark.parametrize("key,kante", list(_alle_regler_grenzen()))
def test_jeder_regler_am_minimum_und_maximum(key, kante):
    at = _app()
    spec = P.SETTING_SPECS[key]
    wert = (spec.options[0] if kante == "min" else spec.options[-1]) if spec.options else (spec.lo if kante == "min" else spec.hi)
    _regler(at, key).set_value(wert).run(timeout=TIMEOUT)
    assert not at.exception, (key, kante, [str(e) for e in at.exception])


@pytest.mark.parametrize("kante", ["min", "max"])
def test_alle_regler_zugleich_am_minimum_und_maximum_worst_case(kante):
    """Alle Maxima zugleich (breitestes Gitter: Streuung 1 → 3 mit D = 8, kleinste Prominenz, größtes s und δ) und alle Minima: Hänger und Aufblähen abfangen, keine Zeit-Assertion."""
    at = _app()
    if kante == "max":
        at.sidebar.selectbox(key="fall1").set_value(C.FALL1[1])
    else:
        at.sidebar.selectbox(key="fall1").set_value(C.FALL1[0])
    for key, spec in P.SETTING_SPECS.items():
        if key == "fall1":
            continue
        wert = (spec.options[0] if kante == "min" else spec.options[-1]) if spec.options else (spec.lo if kante == "min" else spec.hi)
        if key == "prom2":
            wert = C.PROM2[0] if kante == "max" else C.PROM2[-1]                  # das Maximum der Rechnung ist die kleinste Prominenz
        _regler(at, key).set_value(wert)
    at.run(timeout=TIMEOUT)                                                       # das Netz ist das Zeitlimit des AppTest, keine Zeit-Assertion
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("plotly_chart")) == 6


def test_kernfragen_nennen_den_abstand_als_verhaeltnis_und_die_atom_fassung_als_punktmassen():
    at = _app()
    infos = [i.value for i in at.info]
    kf1 = next(t for t in infos if t.startswith("Es gibt zwei Mitten"))
    assert "das 1,188-Fache des Abstands t·W2(A, B) der Verschiebung (Faktor 1,188): die Mischung liegt weiter vom Anfang entfernt." in kf1 and "erlaubt" not in kf1
    kf2 = next(t for t in infos if t.startswith("Drei Standorte mit Spitzen"))
    assert "(hier: die Mitte der Versätze −s, 0, +s im W2-Sinn, auf der Linie)" in kf2
    kf3 = next(t for t in infos if t.startswith("Bei verschiedenen Anteilen"))
    assert "Liegt jede Spitze als Punktmasse bei 8 bzw. 18 Uhr (Atom-Fassung), liegt bei diesen Anteilen genau ein Fünftel" in kf3 and "Stunde Breite" not in kf3
    assert any("Wann legt das Baryzentrum Masse zwischen die Spitzen?" in m.value for m in at.markdown)
    assert any("Masse zwischen den Spitzen" in s.value for s in at.subheader)


def test_glocke_bei_d_null_zeigt_in_der_app_keinen_minus_null_block():
    at = _app(f1=C.FALL1[2], d1=0)
    verdikt = next(v for v in _verdikte(at) if v.startswith("Glocke"))
    assert "ein Block bei 0" in verdikt and "−0" not in verdikt and "näher am Anfang" in verdikt


def test_expander_in_einfachen_worten_verlinkt_die_erklaerseite_und_die_lesehilfe_steht_einmal():
    at = _app()
    expander = [e for e in at.expander if e.label == "💬 In einfachen Worten"]
    assert len(expander) == 1
    inhalt = " ".join(m.value for m in expander[0].markdown)
    assert "](https://sebastianhanisch.net/verschieben-statt-mischen-erklaert.html)" in inhalt
    alles = " ".join(m.value for m in at.markdown)
    assert "der Expander **💬 In einfachen Worten**" in alles                      # die Einleitung nennt ihn
    assert alles.count("**Lesehilfe.**") == 1 and "](https://sebastianhanisch.net/verschieben-statt-mischen-erklaert.html#fachleute)" in alles      # eine Lesehilfe, mit Link auf den Fachteil
