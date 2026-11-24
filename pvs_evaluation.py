"""Kennzahlen, Reihen und Verdikte der drei Experimente (DEMO-PLAYBOOK §12: jedes Verdikt ist eine reine Funktion `verdikt_*(…) -> (Art, Text)`).

Alles ist deterministisch (kein Zufall); jede Reihe ist eine reine Funktion der Regler und wird in der App zwischengespeichert.
Gerechnet wird in 1D auf einem Gitter von 0,001 Stunden mit 20 000 Quantilstufen (siehe `pvs_verteilung`)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import pvs_constants as C
import pvs_diskret as D
import pvs_profile as P
import pvs_verteilung as V
from pvs_verteilung import Mix, mischung, normal, verschoben


@dataclass(frozen=True)
class Settings:
    fall1: str = C.DEFAULTS["fall1"]
    d1: float = C.DEFAULTS["d1"]
    t1: float = C.DEFAULTS["t1"]
    s2: float = C.DEFAULTS["s2"]
    frage2: str = C.DEFAULTS["frage2"]
    prom2: float = C.DEFAULTS["prom2"]
    delta3: float = C.DEFAULTS["delta3"]


def settings_aus_zustand(werte: dict) -> Settings:
    return Settings(**{k: werte[k] for k in C.DEFAULTS})


# ---------------------------------------------------------------- Formatierung (deutsch)


def zahl(x: float, nach: int = 3) -> str:
    """Dezimalkomma, Minuszeichen „−“ (nie „−0,0“)."""
    s = f"{x:.{nach}f}"
    if float(s) == 0.0:
        s = s.lstrip("-")
    return s.replace("-", "−").replace(".", ",")


def prozent(x: float, nach: int = 1) -> str:
    return zahl(100.0 * x, nach) + " %"


def kurz(x: float, nach: int = 3) -> str:
    """Zahl mit höchstens `nach` Nachkommastellen ohne Endnullen (Dezimalkomma): 3 → „3“, 0,125 → „0,125“."""
    s = zahl(x, nach)
    return s.rstrip("0").rstrip(",") if "," in s else s


def bruch_text(b: Fraction) -> str:
    return str(b.numerator) if b.denominator == 1 else f"{b.numerator}/{b.denominator}"


# ---------------------------------------------------------------- Experiment 1: Verschieben oder mischen


def enden1(fall: str, d: float) -> tuple[Mix, Mix]:
    """Die zwei Enden A und B des Falls; d ist der Abstand D in Stunden."""
    if fall == C.FALL1[0]:
        return normal(0.0, 1.0), normal(d, 1.0)
    if fall == C.FALL1[1]:
        return normal(0.0, 1.0), normal(d, C.B_SIGMA_BREIT)
    if fall == C.FALL1[2]:
        return normal(0.0, 1.0), Mix((-d, d), (C.BLOCK_SIGMA, C.BLOCK_SIGMA), (0.5, 0.5))
    if fall == C.FALL1[3]:
        return P.profil(), verschoben(P.profil(), d)
    raise ValueError(f"unbekannter Fall: {fall}")


def t_aus_regler(t: float) -> float:
    """Der Wert von t, gerundet auf die Stützstellen des Reglers (Schritt 0,05)."""
    return round(round(float(t) / C.T1_SCHRITT) * C.T1_SCHRITT, 2)


def reihe1(fall: str, d: float, t_sel: float, prom: float = C.PROM1) -> dict:
    """Mischung und Verschiebung zwischen A und B für t aus `T1_KURVE` (Kurven) und `T1_SCHRITTE` (Tabelle).

    Rückgabe: Enden, Gitter und je t Gipfelzahl, Streuung, W2 und TV zum Anfang (beide Wege); die Dichten nur für das gewählte t (`t_sel`)."""
    t_sel = t_aus_regler(t_sel)
    a, b = enden1(fall, d)
    g = V.gitter([a, b])
    u, w = V.stufen()
    ca, cb = V.cdf(a, g), V.cdf(b, g)
    fa, fb = V.pdf(a, g), V.pdf(b, g)
    qa, qb = V.quantile(ca, g, u), V.quantile(cb, g, u)
    inv_a, inv_b = 1.0 / V.pdf(a, qa), 1.0 / V.pdf(b, qb)
    w2_ab = V.w2_quadrat(qa, qb, w) ** 0.5
    tv_ab = V.tv_auf_gitter(fa, fb)
    sa, sb = V.mittel_streuung(a)[1], V.mittel_streuung(b)[1]

    def mischung_t(t):
        ct, ft = (1 - t) * ca + t * cb, (1 - t) * fa + t * fb
        q = V.quantile(ct, g, u)
        m = mischung([a, b], [1 - t, t])
        r = {"gipfel": V.gipfel_zaehlen(ft, prom), "streuung": V.mittel_streuung(m)[1], "w2": V.w2_quadrat(qa, q, w) ** 0.5, "tv": V.tv_auf_gitter(fa, ft)}
        if t == t_sel:
            r["dichte"] = ft
        return r

    def verschiebung_t(t):
        q = (1 - t) * qa + t * qb
        with np.errstate(divide="ignore", invalid="ignore"):
            dichte = 1.0 / ((1 - t) * inv_a + t * inv_b)
        dichte = np.where(np.isfinite(dichte), dichte, 0.0)
        r = {"gipfel": V.gipfel_zaehlen(dichte, prom), "streuung": V.streuung_aus_quantilen(q, w),
             "w2": V.w2_quadrat(qa, q, w) ** 0.5, "tv": V.tv_auf_gitter(fa, V.auf_gitter(q, dichte, g))}
        if t == t_sel:
            r["pos"], r["dichte"] = q, dichte
        return r

    ts = sorted(set(C.T1_KURVE) | set(C.T1_SCHRITTE) | {t_sel})
    mi = {t: mischung_t(t) for t in ts}
    ve = {t: verschiebung_t(t) for t in ts}
    return {"a": a, "b": b, "x": g, "fa": fa, "fb": fb, "t": ts, "t_sel": t_sel, "mischung": mi, "verschiebung": ve, "w2_ab": w2_ab, "tv_ab": tv_ab,
            "gipfel_a": V.gipfel_zaehlen(fa, prom), "gipfel_b": V.gipfel_zaehlen(fb, prom), "streuung_a": sa, "streuung_b": sb}


# ---------------------------------------------------------------- Experiment 2: Tagesprofile, Summe oder typisches Profil


def reihe2(s: float, prom: float) -> dict:
    """Drei Standorte mit gleichem Profil bei −s, 0, +s: Mischung (Summe geteilt durch 3) und Baryzentrum (Quantilmittel).

    Rückgabe: Kennzahlen beider Mitten (Gipfel, Streuung, Fensterdeckung, Spitzenstunde, Masse außerhalb von 0 bis 24 Uhr) und die Dichten."""
    standorte = P.standorte_versetzt(s)
    mix = mischung(standorte, C.GEWICHTE)
    g = V.gitter(standorte)
    u, w = V.stufen()
    qs = P.quantile_der_standorte(standorte, g, u)
    qbar = V.quantilmittel(qs, C.GEWICHTE)
    dbar = V.dichte_quantilmittel(standorte, C.GEWICHTE, qs)
    q0 = V.quantile(V.cdf(P.profil(), g), g, u)
    f_mix = V.pdf(mix, P.X_GITTER)
    return {
        "s": s, "prom": prom,
        "gipfel_mischung": V.gipfel_zaehlen(f_mix, prom), "gipfel_bary": V.gipfel_zaehlen(dbar, prom),
        "streuung_mischung": V.mittel_streuung(mix)[1], "streuung_bary": V.streuung_aus_quantilen(qbar, w), "streuung_profil": V.mittel_streuung(P.profil())[1],
        "deckung_summe": P.fensterdeckung_mischung(mix), "deckung_bary": P.fensterdeckung_quantile(qbar, u),
        "stunde_summe": P.spitzenstunde_mischung(mix), "stunde_bary": P.spitzenstunde_quantile(qbar, u),
        "ausserhalb": P.masse_ausserhalb(mix),
        "abstand_zum_profil": float(np.abs(qbar - q0).max()),
        "x": P.X_GITTER, "dichte_mischung": f_mix, "dichten_standorte": [V.pdf(m, P.X_GITTER) for m in standorte],
        "pos_bary": qbar, "dichte_bary": dbar,
    }


def gipfel_gegen_s(prom: float, s_werte=C.S2_WERTE) -> list:
    """Zahl der Gipfel der Mischung für jedes s (Baryzentrum: immer die des Profils)."""
    return [P.gipfel_mischung(P.standorte_versetzt(s), prom) for s in s_werte]


# ---------------------------------------------------------------- Experiment 3: unterschiedliche Morgenanteile


def reihe3_atome(delta: float) -> dict:
    """Atom-Fassung (jede Spitze eine Punktmasse bei 8 bzw. 18 Uhr), exakt in Brüchen: Baryzentrum, Mischung, Masse zwischen den Spitzen, Zielwerte, Varianzen."""
    dl = P.bruch(delta)
    standorte = P.atome_standorte(dl)
    gw = [Fraction(1, 3)] * 3
    bary = D.quantilmittel(standorte, gw)
    mix = D.mischung(standorte, gw)
    return {
        "delta": dl, "standorte": standorte, "bary": bary, "mischung": mix,
        "leer_bary": D.masse_zwischen(bary, *map(Fraction, (8, 18))), "leer_mischung": D.masse_zwischen(mix, 8, 18),
        "ziel_bary": D.zielwert(bary, standorte, gw), "ziel_mischung": D.zielwert(mix, standorte, gw),
        "varianz_standorte": [D.mittel_varianz(s_)[1] for s_ in standorte],
        "varianz_bary": D.mittel_varianz(bary)[1], "varianz_mischung": D.mittel_varianz(mix)[1],
    }


def reihe3(delta: float, prom: float = C.PROM1) -> dict:
    """Gauß-Fassung (σ = 1,2 Stunden): Masse zwischen den Spitzen (10,4 bis 15,6 Uhr) für Baryzentrum, Mischung und jeden Standort."""
    standorte = P.standorte_anteile(delta)
    mix = mischung(standorte, C.GEWICHTE)
    g = V.gitter(standorte)
    u, w = V.stufen()
    qs = P.quantile_der_standorte(standorte, g, u)
    qbar = V.quantilmittel(qs, C.GEWICHTE)
    dbar = V.dichte_quantilmittel(standorte, C.GEWICHTE, qs)
    qmix = V.quantile(V.cdf(mix, g), g, u)
    return {
        "delta": delta,
        "leer_bary": float(V.masse_aus_quantilen(qbar, u, *C.LEER)), "leer_mischung": float(V.masse(mix, *C.LEER)),
        "leer_standorte": [float(V.masse(m, *C.LEER)) for m in standorte],
        "streuung_bary": V.streuung_aus_quantilen(qbar, w), "streuung_mischung": V.mittel_streuung(mix)[1],
        "streuung_standorte": [V.mittel_streuung(m)[1] for m in standorte],
        "ziel_bary": float(np.mean([V.w2_quadrat(q, qbar, w) for q in qs])), "ziel_mischung": float(np.mean([V.w2_quadrat(q, qmix, w) for q in qs])),
        "gipfel_bary": V.gipfel_zaehlen(dbar, prom), "gipfel_mischung": V.gipfel_zaehlen(V.pdf(mix, P.X_GITTER), prom),
        "x": P.X_GITTER, "dichte_mischung": V.pdf(mix, P.X_GITTER), "dichten_standorte": [V.pdf(m, P.X_GITTER) for m in standorte],
        "pos_bary": qbar, "dichte_bary": dbar,
    }


def kurve3(werte=C.DELTA3_WERTE) -> dict:
    """Masse zwischen den Spitzen gegen δ: Baryzentrum (Gauß), Mischung (Gauß), Baryzentrum (Atome, exakt) und der Mittelwert der Standorte."""
    zeilen = [reihe3(d) for d in werte]
    atome = [reihe3_atome(d) for d in werte]
    return {"delta": list(werte), "bary": [z["leer_bary"] for z in zeilen], "mischung": [z["leer_mischung"] for z in zeilen],
            "standort": [float(np.mean(z["leer_standorte"])) for z in zeilen], "atome": [float(a["leer_bary"]) for a in atome]}


# ---------------------------------------------------------------- Kennzahlen aus den Reihen und Verdikte


def kennzahlen1(r: dict) -> dict:
    """Zählungen über die neun Zwischenschritte t = 0,1 … 0,9 (Zähleinheit: Zwischenschritte) und das W2-Verhältnis bei t = ½."""
    ts = C.T1_SCHRITTE
    mehr = max(r["gipfel_a"], r["gipfel_b"])
    sa, sb = r["streuung_a"], r["streuung_b"]

    def ok(weg):
        return sum(1 for t in ts if abs(r[weg][t]["streuung"] - ((1 - t) * sa + t * sb)) <= C.STREUUNG_TOL * ((1 - t) * sa + t * sb))

    out = {
        "mehr_gipfel_mischung": sum(1 for t in ts if r["mischung"][t]["gipfel"] > mehr),
        "mehr_gipfel_verschiebung": sum(1 for t in ts if r["verschiebung"][t]["gipfel"] > mehr),
        "eingipfelig_mischung": sum(1 for t in ts if r["mischung"][t]["gipfel"] == 1),
        "eingipfelig_verschiebung": sum(1 for t in ts if r["verschiebung"][t]["gipfel"] == 1),
        "streuung_ok_mischung": ok("mischung"), "streuung_ok_verschiebung": ok("verschiebung"),
        "gipfel_mischung": [r["mischung"][t]["gipfel"] for t in ts], "gipfel_verschiebung": [r["verschiebung"][t]["gipfel"] for t in ts],
    }
    w2 = r["w2_ab"]
    out["verhaeltnis_halb"] = r["mischung"][0.5]["w2"] / (0.5 * w2) if w2 > 1e-9 else None
    out["verhaeltnis_verschiebung_halb"] = r["verschiebung"][0.5]["w2"] / (0.5 * w2) if w2 > 1e-9 else None
    return out


def beschreibung1(fall: str, d: float) -> str:
    dz = kurz(d, 1)
    return {
        C.FALL1[0]: f"Gauß N(0, 1) → N({dz}, 1)",
        C.FALL1[1]: f"Gauß N(0, 1) → N({dz}, 3²)",
        C.FALL1[2]: (f"Glocke N(0, 1) → zwei Blöcke bei −{dz} und +{dz} (Streuung 0,5)" if d != 0 else "Glocke N(0, 1) → ein Block bei 0 (Streuung 0,5; bei D = 0 fallen die zwei Blöcke zusammen)"),
        C.FALL1[3]: f"Tagesprofil mit Spitzen bei 8 und 18 Uhr → dasselbe Profil um {dz} Stunden versetzt",
    }[fall]


def abstandssatz(verhaeltnis: float) -> str:
    """Bei t = ½: W2-Abstand der Mischung zum Anfang im Verhältnis zum Abstand t·W2(A, B) der Verschiebung (ein gemessenes Verhältnis, keine Schranke);
    über 1 liegt die Mischung weiter vom Anfang entfernt, unter 1 näher am Anfang."""
    faktor = zahl(verhaeltnis, 3)
    if faktor == zahl(1.0, 3):
        lage = "die Mischung liegt gleich weit vom Anfang entfernt"
    elif verhaeltnis > 1.0:
        lage = "die Mischung liegt weiter vom Anfang entfernt"
    else:
        lage = "die Mischung liegt näher am Anfang"
    return f"Bei t = ½ ist der W2-Abstand der Mischung zum Anfang das {faktor}-Fache des Abstands t·W2(A, B) der Verschiebung (Faktor {faktor}): {lage}."


def verdikt1(fall: str, d: float, r: dict) -> tuple[str, str]:
    """Verdikt zu Experiment 1 (Zähleinheit: die neun Zwischenschritte t = 0,1 … 0,9)."""
    k = kennzahlen1(r)
    kopf = beschreibung1(fall, d)
    if r["w2_ab"] < 1e-9:
        return "success", f"{kopf}: Anfang und Ende sind gleich; Mischung und Verschiebung fallen in jedem Zwischenschritt zusammen."
    abstand = abstandssatz(k["verhaeltnis_halb"])
    if fall in (C.FALL1[0], C.FALL1[1]):
        text = (f"{kopf}: Die Mischung ist in {k['eingipfelig_mischung']} von 9 Zwischenschritten eingipfelig, die Verschiebung in {k['eingipfelig_verschiebung']} von 9 "
                f"(für Normalverteilungen per Definition: das Quantilmittel zweier Normalverteilungen ist wieder eine, mit linear wechselnder Lage und Streuung). "
                f"Die Streuung liegt bei der Mischung in {k['streuung_ok_mischung']} von 9 Schritten höchstens 10 % neben der Vorgabe (1 − t)·s₀ + t·s₁, bei der Verschiebung in {k['streuung_ok_verschiebung']} von 9. "
                f"{abstand}")
        art = "warning" if (k["eingipfelig_mischung"] < 9 or k["streuung_ok_mischung"] < 9) else "info"
    else:
        text = (f"{kopf}: Anfang {r['gipfel_a']} Gipfel, Ende {r['gipfel_b']}. Die Mischung hat in {k['mehr_gipfel_mischung']} von 9 Zwischenschritten mehr Gipfel als beide Enden "
                f"(Gipfelzahl von t = 0,1 bis 0,9: {', '.join(str(g) for g in k['gipfel_mischung'])}), die Verschiebung in {k['mehr_gipfel_verschiebung']} von 9 "
                f"({', '.join(str(g) for g in k['gipfel_verschiebung'])}). {abstand}")
        art = "warning" if k["mehr_gipfel_mischung"] > k["mehr_gipfel_verschiebung"] else "info"
    return art, text + f" (Gipfel: Prominenz mindestens {prozent(C.PROM1, 0)} der höchsten Dichte.)"


def laeufe(s_werte, gipfel) -> str:
    """„2 (s = 0 bis 2,5), 6 (s = 2,75 bis 3,25), …“: aufeinanderfolgende gleiche Gipfelzahlen zusammengefasst."""
    teile, start = [], 0
    for i in range(1, len(gipfel) + 1):
        if i == len(gipfel) or gipfel[i] != gipfel[start]:
            teile.append(f"{gipfel[start]} (s = {kurz(s_werte[start], 2)}" + (f" bis {kurz(s_werte[i - 1], 2)})" if i - 1 > start else ")"))
            start = i
    return ", ".join(teile)


def verdikt2(frage: str, s: float, r: dict, gipfel_s: list) -> tuple[str, str]:
    """Verdikt zu Experiment 2 (drei Standorte gleichen Gewichts, Normalverteilungen σ = 1,2 Stunden, Linie statt Kreis)."""
    pr = prozent(r["prom"], 1)
    rahmen = f"(Normalverteilungen mit σ = 1,2 Stunden, drei Standorte gleichen Gewichts, Linie statt Kreis; Gipfel bei Prominenz {pr} der höchsten Dichte.)"
    if s == 0:
        return "info", (f"s = 0: die drei Standorte liegen übereinander. Mischung und Baryzentrum sind das Profil selbst: {r['gipfel_mischung']} Gipfel, Streuung {zahl(r['streuung_mischung'])} Stunden, "
                        f"die Fenster decken {prozent(r['deckung_summe'], 2)} der Nachfrage. {rahmen}")
    verhaeltnis = r["stunde_bary"] / r["stunde_summe"]
    zusatz = f" Gipfel der Mischung für s = 0 bis 4: {laeufe(C.S2_WERTE, gipfel_s)}."
    if frage == C.FRAGE2[0]:
        text = (f"s = {kurz(s, 2)} Stunden, Frage „Summe der Standorte“: Die aus dem Baryzentrum abgelesenen Fenster (8 ± 2,4 und 18 ± 2,4 Uhr) decken bei der Summe der Standorte "
                f"{prozent(r['deckung_summe'], 2)} der Gesamtnachfrage statt {prozent(r['deckung_bary'], 2)}, und die Spitzenstunde des Baryzentrums ist {zahl(verhaeltnis, 2)}-mal so hoch wie die der Summe. "
                f"Für diese Frage ist die Summe (geteilt durch drei: die Mischung) die richtige Antwort, nicht das Baryzentrum. Die Mischung hat {r['gipfel_mischung']} Gipfel, das Baryzentrum {r['gipfel_bary']}.{zusatz} {rahmen}")
        return "warning", text
    abstand = "unter 1e-9" if r["abstand_zum_profil"] < 1e-9 else zahl(r["abstand_zum_profil"], 6)
    text = (f"s = {kurz(s, 2)} Stunden, Frage „Gestalt bei verschobener Phase“: Die Mischung hat {r['gipfel_mischung']} Gipfel und die Streuung {zahl(r['streuung_mischung'])} Stunden, das Baryzentrum "
            f"{r['gipfel_bary']} Gipfel und {zahl(r['streuung_bary'])} Stunden: es ist das mittlere Profil (größter Abstand der Quantilfunktionen zum Profil {abstand} Stunden). "
            f"Für die Frage, wie ein Standort mit mittlerer Phase aussieht (hier: die Mitte der Versätze −s, 0, +s im W2-Sinn, auf der Linie), ist das Baryzentrum die richtige Antwort, wenn alle Standorte dasselbe Profil nur zeitlich verschoben haben; "
            f"Experiment 3 zeigt, was bei verschiedenen Anteilen geschieht.{zusatz} {rahmen}")
    return "info", text


def verdikt3(delta: float, r: dict, a: dict) -> tuple[str, str]:
    """Verdikt zu Experiment 3 (Morgenanteile 0,4 − δ, 0,4, 0,4 + δ, drei Standorte gleichen Gewichts)."""
    dz = kurz(delta, 3)
    if delta == 0:
        return "info", (f"δ = 0: alle drei Standorte haben denselben Anteil; das Baryzentrum ist das Profil, {prozent(r['leer_bary'])} seiner Masse liegen zwischen 10,4 und 15,6 Uhr, "
                        f"ebenso viel wie bei jedem Standort ({prozent(r['leer_standorte'][0])}) und bei der Mischung ({prozent(r['leer_mischung'])}) (Normalverteilungen, σ = 1,2 Stunden).")
    leer = bruch_text(a["leer_bary"])
    return "warning", (f"δ = {dz}: In der Atom-Fassung (jede Spitze eine Punktmasse bei 8 bzw. 18 Uhr) liegt der Bruchteil {leer} der Masse des Baryzentrums zwischen den Spitzen, wo kein Standort Nachfrage hat; die Mischung hat dort nichts. "
                       f"Bei Normalverteilungen (σ = 1,2 Stunden) liegen {prozent(r['leer_bary'])} der Masse des Baryzentrums zwischen 10,4 und 15,6 Uhr, bei jedem Standort höchstens {prozent(max(r['leer_standorte']))} "
                       f"und bei der Mischung {prozent(r['leer_mischung'])}. Das Baryzentrum minimiert den Mittelwert der quadrierten Verschiebungen zu den Standorten (Atome: {bruch_text(a['ziel_bary'])} gegen "
                       f"{bruch_text(a['ziel_mischung'])} bei der Mischung), es erhält die Gestalt der Standorte nicht. Gilt für drei Standorte gleichen Gewichts mit diesen Spitzen.")


# ---------------------------------------------------------------- Gleichstand: zwei gleich teure Zuordnungen, verschiedene Zwischenverteilungen


def atome_text(atome) -> str:
    """„1 (½), 2 (½)“: Orte mit ihren Massen."""
    return ", ".join(f"{bruch_text(x)} ({bruch_text(m)})" for x, m in atome)


def gleichstand_zeilen(t=Fraction(1, 2)) -> list[dict]:
    """Die zwei Zuordnungen von P = (0, 1) nach Q = (2, 3): Kosten |x − y| und (x − y)² (Mittel je Einheit Masse) und die Zwischenverteilung bei t."""
    b = D.gleichstand_beispiel(t)
    zeilen = []
    for z, k1, k2 in zip(b["zuordnungen"], b["kosten1"], b["kosten2"]):
        zeilen.append({"zuordnung": ", ".join(f"{bruch_text(x)} → {bruch_text(y)}" for x, y in z), "kosten1": k1, "kosten2": k2,
                       "optimal1": k1 == min(b["kosten1"]), "optimal2": k2 == min(b["kosten2"]), "zwischen": atome_text(D.zwischenverteilung(z, t))})
    return zeilen
