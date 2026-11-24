"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, paarweise verschieden; Permalink-Einrasten als Einheit."""

import pytest

import pvs_constants as C
import pvs_evaluation as ev
import pvs_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_or_options():
    for name, preset in C.PRESETS.items():
        for key, wert in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert wert in spec.options, (name, key)
            else:
                assert spec.lo <= wert <= spec.hi, (name, key)
                assert abs((wert - spec.lo) / spec.step - round((wert - spec.lo) / spec.step)) < 1e-9, (name, key)      # auf der Stufe des Reglers


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[next(iter(C.PRESETS))]) == ev.Settings()
    assert C.PRESETS[next(iter(C.PRESETS))] == C.DEFAULTS


def test_presets_sind_paarweise_verschieden():
    werte = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(werte)) == len(werte)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) == set(C.DEFAULTS)


def test_url_parameter_sind_eindeutig():
    urls = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(set(urls)) == len(urls)


@pytest.mark.parametrize("key,roh,erwartet", [
    ("d1", 99.0, C.D1_MAX), ("d1", -3.0, C.D1_MIN), ("d1", 2.74, 2.5), ("t1", 0.33, 0.35), ("s2", 3.1, 3.0), ("delta3", 0.11, 0.1), ("delta3", 0.2, 0.2),
    ("prom2", 0.04, 0.05), ("prom2", 0.5, 0.10), ("prom2", 0.0, 0.005), ("fall1", "gibt es nicht", None), ("fall1", C.FALL1[1], C.FALL1[1]),
])
def test_eingerastet_begrenzt_rundet_und_lehnt_unbekanntes_ab(key, roh, erwartet):
    wert = P._eingerastet(P.SETTING_SPECS[key], roh)
    assert wert == (pytest.approx(erwartet) if erwartet is not None else None)


def test_eingerastet_ignoriert_nan():
    assert P._eingerastet(P.SETTING_SPECS["d1"], float("nan")) is None
    assert P._eingerastet(P.SETTING_SPECS["prom2"], float("nan")) is None


def test_hilfetext_je_preset_nennt_die_tatsaechlichen_reglerwerte():
    """Befund der Aussagenprüfung: „die übrigen Experimente stehen auf dem Fall ohne Besonderheit“ stimmte beim Preset der Morgenanteile nicht (Experiment 1 steht auf N(0, 1) → N(4, 1))."""
    p = C.PRESETS["Unterschiedliche Morgenanteile"]
    h = C.PRESET_HELP["Unterschiedliche Morgenanteile"]
    assert p["fall1"] == C.FALL1[0] and p["d1"] == 4.0 and p["t1"] == 0.5 and p["s2"] == 0.0 and p["delta3"] == 0.1
    assert "Experiment 1 steht auf Gauß N(0, 1) → N(4, 1) bei t = ½" in h and "Experiment 2 auf s = 0 (kein Versatz)" in h and "Besonderheit" not in h
    p = C.PRESETS["Glocke → zwei Blöcke"]
    h = C.PRESET_HELP["Glocke → zwei Blöcke"]
    assert p["fall1"] == C.FALL1[2] and p["d1"] == 4.0 and p["t1"] == 0.1 and p["s2"] == 0.0 and p["delta3"] == 0.0
    assert "Experiment 2: kein Versatz (s = 0), Experiment 3: gleiche Anteile (δ = 0)" in h and "Besonderheit" not in h
    p = C.PRESETS["Zwei Profile, 2 h versetzt"]
    assert p["fall1"] == C.FALL1[3] and p["d1"] == 2.0 and p["s2"] == 0.0 and p["delta3"] == 0.0
    p = C.PRESETS["Drei Standorte, s = 3"]
    assert p["fall1"] == C.FALL1[0] and p["d1"] == 4.0 and p["t1"] == 0.5 and p["s2"] == 3.0 and p["delta3"] == 0.1
