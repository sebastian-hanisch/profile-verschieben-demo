"""Die Stücke der README, die Zahlen nennen, werden hier aus dem Code der Demo gebildet (dieselben Funktionen wie in der App).

`tests/test_claims.py` verlangt, dass jedes Stück wörtlich in der README steht; ändert sich die Rechnung, schlägt der Test fehl und die README wird neu zusammengesetzt
(`python tests/readme_zeilen.py` druckt alle Stücke mit ihrem Namen)."""

import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pvs_constants as C  # noqa: E402
import pvs_diskret as D  # noqa: E402
import pvs_evaluation as E  # noqa: E402
import pvs_profile as P  # noqa: E402
import pvs_verteilung as V  # noqa: E402


def _md(kopf, zeilen):
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)] + ["| " + " | ".join(z) + " |" for z in zeilen])


def tabelle_gauss() -> str:
    zeilen = []
    for fall, d in [(C.FALL1[0], x) for x in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)] + [(C.FALL1[1], 0.0), (C.FALL1[1], 4.0)]:
        k = E.kennzahlen1(E.reihe1(fall, d, 0.5))
        name = "N(0, 1) → N(D, 1)" if fall == C.FALL1[0] else "N(0, 1) → N(D, 3²)"
        zeilen.append([name, E.kurz(d, 1), f"{k['eingipfelig_mischung']} von 9", f"{k['streuung_ok_mischung']} von 9", f"{k['eingipfelig_verschiebung']} von 9", f"{k['streuung_ok_verschiebung']} von 9",
                       E.zahl(k["verhaeltnis_halb"], 3)])
    return _md(["Enden", "D (Stunden)", "Mischung eingipfelig", "Mischung: Streuung ok", "Verschiebung eingipfelig", "Verschiebung: Streuung ok", "W2-Verhältnis der Mischung bei t = ½"], zeilen)


def satz_bloecke() -> str:
    k = E.kennzahlen1(E.reihe1(C.FALL1[2], 4.0, 0.5))
    return (f"Glocke N(0, 1) → zwei Blöcke bei −4 und +4 (Streuung 0,5), Gipfelzahl von t = 0,1 bis 0,9: Mischung {', '.join(map(str, k['gipfel_mischung']))}; "
            f"Verschiebung {', '.join(map(str, k['gipfel_verschiebung']))}; das W2-Verhältnis der Mischung bei t = ½ ist {E.zahl(k['verhaeltnis_halb'], 3)}.")


def tabelle_tagesprofil() -> str:
    zeilen = []
    for d in (2.0, 4.0):
        k = E.kennzahlen1(E.reihe1(C.FALL1[3], d, 0.5))
        zeilen.append([E.kurz(d, 1), ", ".join(map(str, k["gipfel_mischung"])), ", ".join(map(str, k["gipfel_verschiebung"])), E.zahl(k["verhaeltnis_halb"], 3)])
    return _md(["Versatz D (Stunden)", "Gipfel der Mischung, t = 0,1 bis 0,9", "Gipfel der Verschiebung", "W2-Verhältnis der Mischung bei t = ½"], zeilen)


def satz_geodaeten() -> str:
    r = E.reihe1(C.FALL1[0], 4.0, 0.5)
    ts = C.T1_KURVE
    abw_tv_mischung = max(abs(r["mischung"][t]["tv"] / r["tv_ab"] - t) for t in ts)
    abw_w2_verschiebung = max(abs(r["verschiebung"][t]["w2"] / r["w2_ab"] - t) for t in ts)
    assert abw_tv_mischung < 1e-12 and abw_w2_verschiebung < 1e-12
    return (f"Für N(0, 1) → N(4, 1) liegt die Mischung in der Totalvariation auf der Geraden (TV(A, Mischung_t) = t · TV(A, B), über die 21 Werte von t in Schritten von 0,05 bis auf Rundung, Abweichung unter 1e-12) "
            f"und die Verschiebung in W2 (W2(A, Verschiebung_t) = t · W2(A, B), ebenso); umgekehrt liegt die Verschiebung bei t = ½ in der Totalvariation bei "
            f"{E.zahl(r['verschiebung'][0.5]['tv'] / r['tv_ab'], 3)} von TV(A, B) statt bei 0,500, und die Mischung in W2 beim {E.zahl(r['mischung'][0.5]['w2'] / (0.5 * r['w2_ab']), 3)}-fachen des halben Weges.")


def tabelle_exp2() -> str:
    zeilen = []
    for s in C.S2_ZEILEN:
        r = E.reihe2(s, 0.02)
        zeilen.append([E.kurz(s, 1), str(r["gipfel_mischung"]), E.zahl(r["streuung_mischung"], 3), E.prozent(r["deckung_summe"], 2), E.zahl(r["stunde_bary"] / r["stunde_summe"], 2), E.prozent(r["ausserhalb"], 3)])
    r = E.reihe2(3.0, 0.02)
    zeilen.append(["Baryzentrum (jedes s)", str(r["gipfel_bary"]), E.zahl(r["streuung_bary"], 3), E.prozent(r["deckung_bary"], 2), "–", "–"])
    return _md(["s (Stunden)", "Gipfel der Mischung", "Streuung der Mischung (Stunden)", "Fensterdeckung der Summe", "Spitzenstunde: Baryzentrum zu Summe", "Masse der Summe außerhalb 0 bis 24 Uhr"], zeilen)


def satz_gipfel_laeufe() -> str:
    s_werte = [0.25 * k for k in range(33)]
    return E.laeufe(s_werte, [P.gipfel_mischung(P.standorte_versetzt(s), 0.02) for s in s_werte])


def satz_prominenz() -> str:
    teile = []
    for prom in C.PROM2:
        teile.append(f"{E.zahl(100 * prom, 1)} %: {P.gipfel_mischung(P.standorte_versetzt(3.0), prom)}")
    return (f"Gipfel der Mischung bei s = 3 Stunden je Prominenz-Kriterium: {'; '.join(teile)}; bei s = 2,75 Stunden sind es {P.gipfel_mischung(P.standorte_versetzt(2.75), 0.02)} bei 2 % "
            f"und {P.gipfel_mischung(P.standorte_versetzt(2.75), 0.05)} bei 5 %.")


def satz_spitzenstunde_atome(profil=None) -> str:
    a = D.tagesprofil_atome(F(2, 5))
    mix = D.mischung([D.verschoben(a, d) for d in (-3, 0, 3)], [F(1, 3)] * 3)
    return f"In der Atom-Fassung (jede Spitze eine Punktmasse bei 8 bzw. 18 Uhr, s = 3) ist die höchste Einzelmasse des Baryzentrums das {E.bruch_text(max(m for _, m in a) / max(m for _, m in mix))}-Fache der Mischung (hier gleich der Zahl der Standorte, weil die sechs Orte der Mischung verschieden sind)."


def tabelle_exp3() -> str:
    zeilen = []
    for delta in (0.0, 0.05, 0.1, 0.2):
        r, a = E.reihe3(delta), E.reihe3_atome(delta)
        zeilen.append([E.kurz(delta, 3), E.bruch_text(a["leer_bary"]), E.prozent(r["leer_bary"], 2), E.prozent(r["leer_mischung"], 2), E.prozent(max(r["leer_standorte"]), 2)])
    return _md(["δ", "Atome: Masse des Baryzentrums zwischen den Spitzen (exakt)", "Normalverteilungen: Baryzentrum", "Normalverteilungen: Mischung", "Normalverteilungen: jeder Standort (höchstens)"], zeilen)


def satz_exp3_kennzahlen() -> str:
    r, a = E.reihe3(0.1), E.reihe3_atome(0.1)
    return (f"Bei δ = 0,1 (Morgenanteile 0,3 / 0,4 / 0,5): Atome: das Baryzentrum hat die Orte {E.atome_text(a['bary'])}, die Mischung {E.atome_text(a['mischung'])}; Mittel der quadrierten Verschiebungen {E.bruch_text(a['ziel_bary'])} gegen {E.bruch_text(a['ziel_mischung'])}; "
            f"Varianzen der Standorte {', '.join(E.bruch_text(v) for v in a['varianz_standorte'])}, des Baryzentrums {E.bruch_text(a['varianz_bary'])}, der Mischung {E.bruch_text(a['varianz_mischung'])}. "
            f"Normalverteilungen (σ = 1,2 Stunden): {r['gipfel_bary']} Gipfel des Baryzentrums gegen {r['gipfel_mischung']} der Mischung, Standardabweichung {E.zahl(r['streuung_bary'], 3)} gegen {E.zahl(r['streuung_mischung'], 3)} Stunden, "
            f"Mittel der quadrierten Verschiebungen {E.zahl(r['ziel_bary'], 3)} gegen {E.zahl(r['ziel_mischung'], 3)} Stunden².")


def tabelle_gleichstand() -> str:
    z = E.gleichstand_zeilen()
    return _md(["Zuordnung", "Kosten \\|x − y\\| (Mittel)", "Kosten (x − y)² (Mittel)", "Zwischenverteilung bei t = ½: Orte (Masse)"],
               [[x["zuordnung"], E.bruch_text(x["kosten1"]) + (" (optimal)" if x["optimal1"] else ""), E.bruch_text(x["kosten2"]) + (" (optimal)" if x["optimal2"] else ""), x["zwischen"]] for x in z])


def satz_streuung_formel() -> str:
    teile = []
    for s in (1.0, 2.0, 3.0, 4.0):
        sd = V.mittel_streuung(V.mischung(P.standorte_versetzt(s), [1 / 3] * 3))[1]
        teile.append(f"s = {E.kurz(s, 0)}: {E.zahl(sd, 3)}")
    return "Streuung der Mischung (Stunden), geschlossen aus dem Gesetz der totalen Varianz $\\sqrt{25{,}44 + 2 s^2 / 3}$: " + "; ".join(teile) + "."


STUECKE = {
    "tabelle_gauss": tabelle_gauss, "satz_bloecke": satz_bloecke, "tabelle_tagesprofil": tabelle_tagesprofil, "satz_geodaeten": satz_geodaeten, "tabelle_exp2": tabelle_exp2,
    "satz_gipfel_laeufe": satz_gipfel_laeufe, "satz_prominenz": satz_prominenz, "satz_spitzenstunde_atome": satz_spitzenstunde_atome, "tabelle_exp3": tabelle_exp3,
    "satz_exp3_kennzahlen": satz_exp3_kennzahlen, "tabelle_gleichstand": tabelle_gleichstand, "satz_streuung_formel": satz_streuung_formel,
}

if __name__ == "__main__":
    for name, f in STUECKE.items():
        print(f"===== {name}\n{f()}\n")
