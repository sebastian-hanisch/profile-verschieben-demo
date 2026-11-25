"""Zwischen zwei Verteilungen – verschieben statt mischen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Die Mitte zwischen zwei Verteilungen ist ihre Mischung: Diese Annahme wird an Normalverteilungen und an Tagesprofilen (Nachfrage über 24 Stunden) geprüft. Drei Experimente:
Zwischenschritte (Mischung gegen Verschiebung), Tagesprofile dreier Standorte (Summe oder typisches Profil) und unterschiedliche Morgenanteile (Masse zwischen den Spitzen).
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import streamlit as st

import pvs_constants as C
import pvs_evaluation as E
from pvs_layout import dunkles_design, kopf_css
from pvs_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    sidebar_presets,
    sync_query_params,
)
from pvs_visualization import (
    build_abstand1,
    build_dichten1,
    build_dichten3,
    build_gipfel2,
    build_kurve3,
    build_profile2,
)

st.set_page_config(page_title="Zwischen zwei Verteilungen – Sebastian Hanisch", layout="wide")
KONFIG = {"displayModeBar": False}


@st.cache_data(show_spinner=False)
def _reihe1(fall, d, t):
    return E.reihe1(fall, d, t)


@st.cache_data(show_spinner=False)
def _reihe2(s, prom):
    return E.reihe2(s, prom)


@st.cache_data(show_spinner=False)
def _gipfel_s(prom):
    return E.gipfel_gegen_s(prom)


@st.cache_data(show_spinner=False)
def _reihe3(delta):
    return E.reihe3(delta)


@st.cache_data(show_spinner=False)
def _atome3(delta):
    return E.reihe3_atome(delta)


@st.cache_data(show_spinner=False)
def _kurve3():
    return E.kurve3()


def _tabelle_md(kopf, zeilen):
    return "\n".join(["| " + " | ".join(kopf) + " |", "|" + "---|" * len(kopf)] + ["| " + " | ".join(z) + " |" for z in zeilen])


def _prom_text(p):
    return E.zahl(100 * p, 1).replace(",0", "") + " %"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in pvs_layout.py).
st.html(kopf_css(dunkles_design()))
st.html('<style>[data-testid="stMarkdownContainer"] table { display: block; overflow-x: auto; max-width: 100%; }</style>')  # breite Tabellen scrollen auf dem Telefon, statt die Seite zu verbreitern
with st.container(key="kopf"):
    st.title("↔️ Zwischen zwei Verteilungen – verschieben statt mischen")
st.markdown(
    """
Was ist die **Mitte zwischen zwei Verteilungen**? Die naheliegende Antwort ist die **Mischung** (englisch mixture): die Wahrscheinlichkeiten werden punktweise gemittelt. Es gibt aber einen zweiten Weg, die **Verschiebung**
(englisch displacement interpolation): jede Masse legt den Bruchteil $t$ ihres Weges zurück, die Quantilfunktionen werden gemittelt. Die Demo prüft die Annahme „die Mitte zweier Verteilungen ist ihre Mischung“
an Normalverteilungen und an **Tagesprofilen** (der Verteilung der Nachfrage über 24 Stunden): Die Mischung ist die Gerade der Totalvariation, die Verschiebung die Gerade der W2-Geometrie, und beide beantworten verschiedene Fragen.
Die Annahme gilt für die Frage, welcher Zustand eintritt, und für die Summe der Standorte, nicht für die Gestalt eines Profils; und auch das **Baryzentrum** (Wasserstein-Baryzentrum, bei mehr als zwei Verteilungen die Verschiebung)
erhält die Gestalt nicht in jedem Fall.

Weiter unten: der Expander **💬 In einfachen Worten**, **❓ Kernfragen**, die drei **🔬 Experimente**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
Alle Rechnungen sind eindimensional und deterministisch (kein Zufall, kein Zufalls-Seed); gezählt wird in Zwischenschritten, Gipfeln und Standorten.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/verschieben-statt-mischen-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Was ist die Mitte zwischen zwei Verteilungen, etwa zwei Tagesprofilen der Nachfrage? Es gibt zwei Antworten. Die **Mischung** nimmt zu jeder Uhrzeit den Durchschnitt der beiden Kurven; liegen die Gipfel weit auseinander, hat sie zwei Gipfel. Die **Verschiebung** rückt die ganze Kurve auf die andere zu und behält ihre Gestalt. Beide sind richtig, aber für verschiedene Fragen: Die Mischung beantwortet, wie hoch die Nachfrage mehrerer Standorte zusammen ist, die Verschiebung, wie ein Standort mit mittlerer Phase aussieht. Unterscheiden sich die Standorte nicht nur in der Phase, legt die Verschiebung Nachfrage zwischen die Spitzen. Alles gilt für erzeugte, eindimensionale Profile mit Quadratkosten und drei Standorte gleichen Gewichts.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/verschieben-statt-mischen-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So rechnen die zwei Mitten", expanded=True):
    st.markdown(
        """
1. Ein **Tagesprofil** ist die Verteilung der Nachfrage über die Uhrzeit mit Gesamtmasse 1 (Anteil an der Tagesnachfrage). Hier: Spitzen bei 8 und 18 Uhr mit den Anteilen 0,4 und 0,6 und der Streuung 1,2 Stunden je Spitze.
2. Die **Mischung** $(1 - t)\\,P + t\\,Q$ mittelt die Wahrscheinlichkeiten. Sie beschreibt, was passiert, wenn mit Wahrscheinlichkeit $t$ das eine Profil gilt und sonst das andere, oder wenn man die Nachfrage mehrerer Standorte addiert (und durch ihre Zahl teilt).
3. Die **Verschiebung** mittelt die **Quantilfunktionen** (englisch quantile functions): $Q_t(u) = (1 - t)\\,Q_P(u) + t\\,Q_Q(u)$. Jede Masse legt den Bruchteil $t$ ihres Weges im optimalen Plan zurück; in 1D bleibt die Reihenfolge der Massen erhalten.
   Bei mehr als zwei Verteilungen heißt das gewichtete Mittel der Quantilfunktionen **Baryzentrum** (Wasserstein-Baryzentrum, Fréchet-Mittel im Raum der Verteilungen): der Minimierer von $\\sum_k \\lambda_k\\, W_2^2(\\nu, \\nu_k)$.
4. **Zwei Fragen, zwei Mitten:** Wie sieht die Nachfrage aller Standorte zusammen aus, und welcher Zustand tritt ein, wenn ein Standort zufällig gewählt wird? Das beantwortet die Mischung. Wie sieht ein Profil mit mittlerer Phase aus,
   wenn die Standorte dasselbe Profil nur zeitlich verschoben haben? Das beantwortet das Baryzentrum.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben

    st.header("🔬 Experiment 1: Verschieben oder mischen")
    st.selectbox("Fall", C.FALL1, key="fall1", help="Gauß: zwei Normalverteilungen (Streuung 1, bei „1 → 3“ endet die Verteilung mit Streuung 3). Glocke → zwei Blöcke: das Ende sind zwei schmale Blöcke bei −D und +D. "
                 "Tagesprofil: Spitzen bei 8 und 18 Uhr, das Ende ist dasselbe Profil um D Stunden versetzt.")
    st.slider("Abstand D (Stunden)", *bounds("d1"), step=C.D1_SCHRITT, key="d1", help="Lage des Endes B (Gauß), Lage der Blöcke bei ± D (Glocke → zwei Blöcke) oder Versatz des Tagesprofils.")
    st.slider("Zwischenschritt t", *bounds("t1"), step=C.T1_SCHRITT, key="t1", help="0 ist der Anfang A, 1 das Ende B. Die Tabelle zeigt immer die neun Schritte t = 0,1 bis 0,9, die Diagramme das gewählte t.")

    st.header("🔬 Experiment 2: Tagesprofile")
    st.slider("Phasenversatz s (Stunden)", *bounds("s2"), step=C.S2_SCHRITT, key="s2", help="Drei Standorte mit demselben Profil, gegeneinander um −s, 0 und +s Stunden verschoben.")
    st.radio("Frage", C.FRAGE2, key="frage2", help="Summe der Standorte: Wie hoch ist die Nachfrage aller Standorte zusammen? Gestalt bei verschobener Phase: Wie sieht ein Standort mit mittlerer Phase aus?")
    st.select_slider("Prominenz-Kriterium", C.PROM2, key="prom2", format_func=_prom_text,
                     help="Ein Gipfel zählt, wenn er mindestens diesen Anteil der höchsten Dichte über dem höheren der beiden Täler zu den nächsten höheren Gipfeln liegt. Die Zahl der Gipfel hängt an diesem Kriterium.")

    st.header("🔬 Experiment 3: Morgenanteile")
    st.slider("Spreizung δ der Morgenanteile", *bounds("delta3"), step=C.DELTA3_SCHRITT, key="delta3", help="Die drei Standorte haben die Morgenanteile 0,4 − δ, 0,4 und 0,4 + δ (der Rest liegt bei 18 Uhr).")

werte = {k: st.session_state[k] for k in C.DEFAULTS}
sync_query_params(werte)
s = E.settings_aus_zustand(werte)

# --- Rechnung ------------------------------------------------------------------------------------------------------------------------------

r1 = _reihe1(s.fall1, s.d1, s.t1)
r2 = _reihe2(s.s2, s.prom2)
gipfel_s = _gipfel_s(s.prom2)
r3 = _reihe3(s.delta3)
a3 = _atome3(s.delta3)
k3 = _kurve3()

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
kf1 = _reihe1(C.FALL1[0], 4.0, 0.5)
kz1 = E.kennzahlen1(kf1)
st.markdown("**Was heißt „die Mitte zwischen zwei Verteilungen“?**")
st.info(
    f"Es gibt zwei Mitten. Zwischen N(0, 1) und N(4, 1) hat die Mischung bei t = ½ {kf1['mischung'][0.5]['gipfel']} Gipfel und die Streuung {E.zahl(kf1['mischung'][0.5]['streuung'], 2)} Stunden, "
    f"die Verschiebung {kf1['verschiebung'][0.5]['gipfel']} Gipfel und die Streuung {E.zahl(kf1['verschiebung'][0.5]['streuung'], 2)} (Vorgabe: 1). In {kz1['eingipfelig_mischung']} von 9 Zwischenschritten ist die Mischung eingipfelig, "
    f"die Verschiebung in {kz1['eingipfelig_verschiebung']} von 9. {E.abstandssatz(kz1['verhaeltnis_halb'])}"
)
kf2 = _reihe2(3.0, 0.02)
st.markdown("**Welche Frage beantwortet welche Mitte?**")
st.info(
    f"Drei Standorte mit Spitzen bei 8 und 18 Uhr, gegeneinander um −3, 0 und +3 Stunden verschoben: Die Summe der Standorte (geteilt durch drei: die Mischung) hat {kf2['gipfel_mischung']} Gipfel, das Baryzentrum {kf2['gipfel_bary']} "
    f"(es ist das mittlere Profil). Für die Summe der Standorte decken die Spitzenfenster des Baryzentrums nur {E.prozent(kf2['deckung_summe'], 2)} der Nachfrage statt {E.prozent(kf2['deckung_bary'], 2)}, und die Spitzenstunde wäre "
    f"{E.zahl(kf2['stunde_bary'] / kf2['stunde_summe'], 2)}-fach überschätzt. Für die Gestalt eines Standorts mit mittlerer Phase ist dagegen das Baryzentrum die richtige Antwort (hier: die Mitte der Versätze −s, 0, +s im W2-Sinn, auf der Linie), wenn alle Standorte dasselbe Profil nur zeitlich verschoben haben."
)
kf3 = _reihe3(0.1)
st.markdown("**Wann legt das Baryzentrum Masse zwischen die Spitzen?**")
st.info(
    f"Bei verschiedenen Anteilen: Mit den Morgenanteilen 0,3 / 0,4 / 0,5 liegen {E.prozent(kf3['leer_bary'])} der Masse des Baryzentrums zwischen den Spitzen (10,4 bis 15,6 Uhr), bei jedem Standort nur {E.prozent(max(kf3['leer_standorte']))} "
    f"und bei der Mischung {E.prozent(kf3['leer_mischung'])} (Normalverteilungen, σ = 1,2 Stunden). Liegt jede Spitze als Punktmasse bei 8 bzw. 18 Uhr (Atom-Fassung), liegt bei diesen Anteilen genau ein Fünftel der Masse des Baryzentrums dazwischen, und dort hat kein Standort Nachfrage."
)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Verschieben oder mischen (Zwischenschritte)")
st.caption("Zwei Wege von A nach B: links die Mischung, rechts die Verschiebung beim gewählten t. Gezählt wird in den neun Zwischenschritten t = 0,1 bis 0,9; Gipfel: Prominenz mindestens 2 % der höchsten Dichte.")
st.plotly_chart(build_dichten1(r1), width="stretch", key="dichten1", config=KONFIG)
tab1 = []
sa, sb = r1["streuung_a"], r1["streuung_b"]
for t in C.T1_SCHRITTE:
    mi, ve = r1["mischung"][t], r1["verschiebung"][t]
    nenner = t * r1["w2_ab"]
    tab1.append([E.zahl(t, 1), str(mi["gipfel"]), str(ve["gipfel"]), E.zahl(mi["streuung"], 3), E.zahl(ve["streuung"], 3), E.zahl((1 - t) * sa + t * sb, 3),
                 E.zahl(mi["w2"] / nenner, 3) if nenner > 1e-9 else "–"])
st.markdown(_tabelle_md(["t", "Gipfel Mischung", "Gipfel Verschiebung", "Streuung Mischung", "Streuung Verschiebung", "Vorgabe (1 − t)·s₀ + t·s₁", "W2(A, Mischung) / (t · W2(A, B))"], tab1))
st.markdown("**Abstand zum Anfang entlang der beiden Wege** (auf den Abstand von A nach B bezogen): Die Verschiebung ist in W2 eine Gerade, die Mischung in der Totalvariation.")
st.plotly_chart(build_abstand1(r1), width="stretch", key="abstand1", config=KONFIG)
kind, text = E.verdikt1(s.fall1, s.d1, r1)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Tagesprofile – Summe oder typisches Profil?")
st.caption("Drei Standorte mit demselben Tagesprofil, gegeneinander um −s, 0 und +s Stunden verschoben, gleiche Gewichte. Die Uhrzeit ist ein Kreis; hier wird sie als Linie behandelt.")
st.plotly_chart(build_profile2(r2, s.frage2), width="stretch", key="profile2", config=KONFIG)
tab2 = [
    ["Gipfel (Prominenz " + _prom_text(s.prom2) + ")", str(r2["gipfel_mischung"]), str(r2["gipfel_bary"]), "–"],
    ["Streuung (Stunden)", E.zahl(r2["streuung_mischung"], 3), E.zahl(r2["streuung_bary"], 3), E.zahl(r2["streuung_mischung"] / r2["streuung_bary"], 3)],
    ["Fensterdeckung (Anteil der Gesamtnachfrage in 8 ± 2,4 und 18 ± 2,4 Uhr)", E.prozent(r2["deckung_summe"], 2), E.prozent(r2["deckung_bary"], 2), E.zahl(r2["deckung_bary"] / r2["deckung_summe"], 2)],
    ["Spitzenstunde (größter Anteil der Tagesnachfrage in einer Stunde)", E.prozent(r2["stunde_summe"], 2), E.prozent(r2["stunde_bary"], 2), E.zahl(r2["stunde_bary"] / r2["stunde_summe"], 2)],
]
st.markdown(_tabelle_md(["Kennzahl", "Summe der Standorte (Mischung)", "Baryzentrum", "Verhältnis Baryzentrum zu Mischung"], tab2))
st.caption(f"Linie statt Kreis: {E.prozent(r2['ausserhalb'], 3)} der Masse der Summe liegen bei diesem s außerhalb von 0 bis 24 Uhr.")
st.markdown("**Zahl der Gipfel der Mischung gegen s** (das Baryzentrum hat immer die zwei Gipfel des Profils): Sie wächst nicht stetig mit s.")
st.plotly_chart(build_gipfel2(C.S2_WERTE, gipfel_s, s.s2, s.prom2), width="stretch", key="gipfel2", config=KONFIG)
kind, text = E.verdikt2(s.frage2, s.s2, r2, gipfel_s)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Unterschiedliche Morgenanteile – Masse zwischen den Spitzen")
st.caption("Drei Standorte mit denselben Spitzen bei 8 und 18 Uhr, aber den Morgenanteilen 0,4 − δ, 0,4 und 0,4 + δ. Das Baryzentrum mittelt die Quantile; wie viel seiner Masse liegt dort, wo die Standorte kaum Nachfrage haben?")
st.plotly_chart(build_dichten3(r3), width="stretch", key="dichten3", config=KONFIG)
tab3 = [
    ["Masse zwischen 10,4 und 15,6 Uhr (Normalverteilungen)", E.prozent(r3["leer_bary"], 2), E.prozent(r3["leer_mischung"], 2)] + [E.prozent(x, 2) for x in r3["leer_standorte"]],
    ["Gipfel der Dichte (Prominenz 2 %)", str(r3["gipfel_bary"]), str(r3["gipfel_mischung"]), "2", "2", "2"],
    ["Standardabweichung (Stunden)", E.zahl(r3["streuung_bary"], 3), E.zahl(r3["streuung_mischung"], 3)] + [E.zahl(x, 3) for x in r3["streuung_standorte"]],
    ["Mittel der quadrierten Verschiebungen zu den Standorten (Stunden²)", E.zahl(r3["ziel_bary"], 3), E.zahl(r3["ziel_mischung"], 3), "–", "–", "–"],
]
st.markdown(_tabelle_md(["Kennzahl", "Baryzentrum", "Mischung", "Standort 1", "Standort 2", "Standort 3"], tab3))
st.markdown("**Atom-Fassung** (jede Spitze eine Punktmasse bei 8 bzw. 18 Uhr, in Brüchen exakt gerechnet): Orte in Stunden, in Klammern die Masse.")
tab3a = [
    ["Baryzentrum", E.atome_text(a3["bary"]), E.bruch_text(a3["leer_bary"]), E.bruch_text(a3["ziel_bary"]), E.bruch_text(a3["varianz_bary"])],
    ["Mischung", E.atome_text(a3["mischung"]), E.bruch_text(a3["leer_mischung"]), E.bruch_text(a3["ziel_mischung"]), E.bruch_text(a3["varianz_mischung"])],
]
st.markdown(_tabelle_md(["", "Orte (Masse)", "Masse zwischen den Spitzen", "Mittel der quadrierten Verschiebungen (Stunden²)", "Varianz (Stunden²)"], tab3a))
st.markdown("**Masse zwischen den Spitzen gegen δ:**")
st.plotly_chart(build_kurve3(k3, s.delta3), width="stretch", key="kurve3", config=KONFIG)
kind, text = E.verdikt3(s.delta3, r3, a3)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme der Vorgänger | Was hier passiert, wenn sie nicht gilt | Wo es weitergeht |
|---|---|---|
| *Bei Gleichstand ist die Wahl des optimalen Plans für den Wert gleichgültig.* | Für den Wert ja, für die Zwischenverteilung nicht: Bei den Kosten \\|x − y\\| haben zwei gleich teure Zuordnungen verschiedene Zwischenverteilungen (Tabelle unten). Bei den Kosten (x − y)² ist die Zuordnung in 1D eindeutig und liefert das Quantilmittel. | – |
| *Kontrast, keine behobene Schwäche:* Die Demo „Abstand zwischen Verteilungen“ misst den Abstand zwischen zwei Verteilungen. | Diese Demo stellt die zwei Geodäten gegenüber: die *Mischung* (Gerade der Totalvariation) gegen die *Verschiebung* (Gerade der W2-Geometrie), siehe die Abstandskurven in Experiment 1. | – |
| *Die Mitte zweier Verteilungen ist ihre Mischung.* (die Annahme, die diese Demo prüft) | Sie gilt für die Frage, welcher Zustand eintritt, und für die Summe der Standorte (Experiment 2). Für die Gestalt eines Profils mit mittlerer Phase gilt sie nicht (Experiment 1 und 2). | – |
| *Das Baryzentrum ist das mittlere Profil.* | Das gilt nur für Verschiebungsfamilien (gleiche Anteile, nur die Phase ändert sich). Bei verschiedenen Anteilen legt es Masse zwischen die Spitzen (Experiment 3). | – |
| *Eindimensional, drei Standorte gleichen Gewichts.* | Zwei und mehr Dimensionen, mehr als drei Standorte und ungleiche Gewichte sind nicht gemessen. | – |
"""
)
st.markdown("**Beispiel zum Mitrechnen (Gleichstand):** Zwei Verteilungen mit je zwei gleich schweren Orten, $P$ bei 0 und 1, $Q$ bei 2 und 3. Es gibt zwei Zuordnungen; die Zwischenverteilung ist bei $t = \\tfrac12$ gerechnet.")
gz = E.gleichstand_zeilen()
st.markdown(_tabelle_md(["Zuordnung", "Kosten \\|x − y\\| (Mittel)", "Kosten (x − y)² (Mittel)", "Zwischenverteilung bei t = ½: Orte (Masse)"],
                        [[z["zuordnung"], E.bruch_text(z["kosten1"]) + (" (optimal)" if z["optimal1"] else ""), E.bruch_text(z["kosten2"]) + (" (optimal)" if z["optimal2"] else ""), z["zwischen"]] for z in gz]))
st.caption("Bei den Kosten |x − y| sind beide Zuordnungen optimal, der Wert ist derselbe, die Zwischenverteilungen sind verschieden; bei (x − y)² ist nur die erste optimal, und sie liefert das Quantilmittel.")

st.markdown("---")

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.**
- Eindimensional, Kosten = quadrierte Stunden (W2); das Wort „Wasserstein“ steht für sich allein nur im Namen „Wasserstein-Baryzentrum“ (gemeint ist immer W2); sonst steht es mit seiner Ordnung (W1 oder W2). Die Verteilungen sind erzeugt (Normalverteilungen und ihre Mischungen, Atome), keine Daten.
- Tagesprofil: Spitzen bei 8 und 18 Uhr, Anteile 0,4 und 0,6, Streuung 1,2 Stunden je Spitze; drei Standorte mit gleichem Gewicht $\\tfrac13$, verschoben um −s, 0, +s (Experiment 2) oder mit den Morgenanteilen $0{,}4 - \\delta$, $0{,}4$, $0{,}4 + \\delta$ (Experiment 3).
- Die Uhrzeit ist ein Kreis, hier eine Linie: ein Teil der Masse liegt außerhalb von 0 bis 24 Uhr (die Zahl steht unter der Tabelle in Experiment 2); der Kreis ist nicht gerechnet.
- Gipfel sind lokale Maxima der Dichte mit einer Prominenz von mindestens einem Anteil der höchsten Dichte (Experiment 2: wählbar, Experiment 1 und 3: 2 %), auf einem Gitter von 0,001 Stunden mit angefügten Nullen an den Rändern (Experiment 2 und 3: von −6 bis 30 Uhr; Experiment 1: bis ±9 Streuungen um die Komponenten der Enden).
- Quantile werden durch lineare Umkehrung der Verteilungsfunktion auf einem Gitter bis ±9 Streuungen um die Komponenten gebildet, Integrale über 20 000 Quantilstufen; die Dichte des Baryzentrums folgt aus den Quantilen ohne Glättung.

**Grenzen.**
- Alle Aussagen gelten in diesem Modell, bei diesen Parametern und mit der genannten Zähleinheit (Zwischenschritte, Gipfel, Standorte).
- Die Gipfelzahl der Mischung hängt am Prominenz-Kriterium und am Versatz s (nicht monoton in s); die Zahl „sechs bei s = 3 Stunden“ gilt für die Prominenz-Schalter 0,5, 1, 2 und 5 %, nicht für 10 %. Das Gitter wurde nicht variiert.
- „Das Baryzentrum ist das mittlere Profil“ gilt nur für Verschiebungsfamilien; „kein Standort hat dort Nachfrage“ nur für die Atom-Fassung (bei Normalverteilungen hat jeder Standort dort etwa 2,3 %).
- Die Fensterdeckung benutzt die aus dem Baryzentrum abgelesenen Fenster (8 ± 2,4 und 18 ± 2,4 Uhr); die Spitzenstunde ist der größte Anteil in einem Intervall von einer Stunde, Lage auf einem Raster von 0,01 Stunden.
- Nicht gemessen: zwei und mehr Dimensionen, mehr als zwei Verteilungen im Raum, mehr als drei Standorte, ungleiche Gewichte, echte Lastprofile, der Kreis (Mitternacht).
- Die Demo „Mitte in anderen Räumen: Kreis und Verteilungen“ rechnet Mischung gegen Quantilmittel für Lieferanten mit Servicegraden; hier ist die Vergleichsgröße die Fensterdeckung der Nachfrage, die Zahlen sind nicht dieselben.
- Literatur: nur so zitiert, wie sie gelesen ist (siehe 📐 Mathematische Formulierung); eine Aussage über den Einsatz von Baryzentren in der Logistik wird hier nicht gemacht.

**Bekannte Probleme und Unsicherheiten.**
- Die Oberfläche ist nicht mit echten Mausklicks im Browser durchgeklickt (AppTest und Tests sehen Plotly nicht).
- Die Zahl „Gipfel des Baryzentrums“ in Experiment 3 ist die der exakten Dichte aus den Quantilen; Kerben in der Kurve rühren von der Dichte, nicht von einem Glättungsparameter.
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/verschieben-statt-mischen-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind zwei Verteilungen $P$, $Q$ auf der Zeitachse (oder $K$ Verteilungen $\nu_k$ mit Gewichten $\lambda_k$). Gesucht ist ihre Mitte, einmal als Mischung, einmal als Verschiebung (Baryzentrum).

| Symbol | Bedeutung |
|---|---|
| $P$, $Q$ | die beiden Verteilungen (Anfang und Ende), Gesamtmasse 1 |
| $t$ | Zwischenschritt, $0 \le t \le 1$ |
| $F$, $Q_P$ | Verteilungsfunktion und Quantilfunktion $Q_P(u) = \inf\{x : F(x) \ge u\}$ ($Q_Q$ ist die von $Q$; das $Q$ in $Q_P$ steht für „Quantil“) |
| $u$ | Stufe der Quantilfunktion, zwischen 0 und 1 |
| $W_2$ | Wasserstein-Abstand der Ordnung 2 (Kosten $(x - y)^2$) |
| $\mathrm{TV}$ | Totalvariation-Abstand der beiden Verteilungen |
| $\lambda_k$, $\nu_k$ | Gewicht und Verteilung des Standorts $k$ ($\lambda_k = \tfrac13$) |
| $\nu^{*}$, $f_k$ | Baryzentrum; Dichte der Verteilung $\nu_k$ |
| $s$, $d_k$ | Phasenversatz und Verschiebung des Standorts $k$, $d_k \in \{-s, 0, +s\}$ |
| $w_k$, $\delta$ | Morgenanteil des Standorts $k$ ($0{,}4 - \delta$, $0{,}4$, $0{,}4 + \delta$); Spreizung $\delta$ der Morgenanteile |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/verschieben-statt-mischen-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Mischung und Verschiebung.**
$$P_t^{\text{Mischung}} = (1 - t)\,P + t\,Q, \qquad Q_t^{\text{Verschiebung}}(u) = (1 - t)\,Q_P(u) + t\,Q_Q(u)$$
In 1D ist $W_2^2(P, Q) = \int_0^1 \big(Q_P(u) - Q_Q(u)\big)^2\,\mathrm{d}u$. Die Verschiebung liegt auf der Geodäte: $W_2(P, P_t^{\text{Verschiebung}}) = t\,W_2(P, Q)$; die Mischung liegt auf der Geraden der Totalvariation: $\mathrm{TV}(P, P_t^{\text{Mischung}}) = t\,\mathrm{TV}(P, Q)$.

**Baryzentrum.**
$$\nu^{*} = \arg\min_{\nu} \sum_{k} \lambda_k\, W_2^2(\nu, \nu_k), \qquad Q_{\nu^{*}}(u) = \sum_k \lambda_k\, Q_{\nu_k}(u) \quad (1\mathrm{D})$$
Die Dichte des Baryzentrums an der Stelle $Q_{\nu^{*}}(u)$ ist $1 \big/ \sum_k \lambda_k / f_k(Q_{\nu_k}(u))$.

**Handbeispiel 1** (Atome): $P$ hat Masse $\tfrac25$ bei 8 und $\tfrac35$ bei 18 Uhr, $Q$ ist $P$ um 2 Stunden versetzt. $W_2^2(P, Q) = 4$. Bei $t = \tfrac12$ liegt die Verschiebung bei 9 ($\tfrac25$) und 19 Uhr ($\tfrac35$) mit $W_2^2(P, \cdot) = 1 = (\tfrac12 \cdot 2)^2$;
die Mischung hat vier Orte (8, 10, 18, 20 Uhr) und $W_2^2(P, \cdot) = 2$ (die Hälfte der Masse muss 2 Stunden weit), das Verhältnis der $W_2$ ist $\sqrt 2$. Allgemein ist $W_2^2(P, P_t^{\text{Mischung}}) = 4t$ gegen $(2t)^2$.

**Handbeispiel 2** (drei Standorte, $s = 3$): Die Mischung hat sechs Orte, das Baryzentrum ist $P$ selbst (zwei Orte). Varianz des Profils 24, der Mischung $24 + \tfrac{2 s^2}{3} = 30$ (Gesetz der totalen Varianz); mit Normalverteilungen ($\sigma = 1{,}2$) ist die Streuung des
Baryzentrums $\sqrt{24 + 1{,}44} = 5{,}044$ und die der Mischung $\sqrt{24 + 1{,}44 + 6} = 5{,}607$ Stunden.

**Handbeispiel 3** (Morgenanteile $\tfrac{3}{10}$, $\tfrac{4}{10}$, $\tfrac{5}{10}$, Atome bei 8 und 18 Uhr): Auf $[0{,}3;\,0{,}4)$ ist das Mittel der Quantile $(18 + 8 + 8)/3 = \tfrac{34}{3}$, auf $[0{,}4;\,0{,}5)$ ist es $(18 + 18 + 8)/3 = \tfrac{44}{3}$. Das Baryzentrum hat also
$\tfrac{3}{10}$ bei 8, $\tfrac1{10}$ bei $\tfrac{34}{3}$, $\tfrac1{10}$ bei $\tfrac{44}{3}$ und $\tfrac12$ bei 18 Uhr; $\tfrac15$ seiner Masse liegt zwischen den Spitzen. Das Mittel der $W_2^2$ zu den Standorten ist $\tfrac{40}{9}$ gegen $\tfrac{20}{3}$ bei der Mischung.

**Gleichstand.** Für $P = \tfrac12(\delta_0 + \delta_1)$, $Q = \tfrac12(\delta_2 + \delta_3)$ kosten beide Zuordnungen bei $|x - y|$ im Mittel 2, bei $(x - y)^2$ aber 4 und 5. Die Zwischenverteilung bei $t = \tfrac12$ ist $\{1, 2\}$ gegen $\{\tfrac32\}$.

**Literatur** (nur so zitiert, wie gelesen): Agueh, Carlier (2011), *Barycenters in the Wasserstein space*, SIAM J. Math. Anal. 43 (Preprint vom 10.12.2010 gelesen, eine frühere Fassung vom 17.08.2010 ebenfalls; Abschnitt 6.1: in 1D ist das Baryzentrum nichtatomarer Maße mit endlichem zweiten Moment das Mittel der Quantilfunktionen; für Punktmassen, die Atom-Fassung, steht das dort nicht, die Demo prüft es selbst gegen ein Linearprogramm; Seitenangaben der Zeitschrift nicht geprüft). McCanns Interpolation: nur nach Agueh/Carlier (dort als Sonderfall zweier Maße) zitiert, McCann selbst nicht gelesen.
Le Coz, Tantet, Flamary, Plougonven (2023, Fassung 3 von 2026), Preprint arXiv 2310.17933 über Baryzentren zum Mitteln von Ensemble-Vorhersagen (nur das Abstract gelesen): das L2-Baryzentrum (übliches Poolen) und das Wasserstein-Baryzentrum haben denselben Mittelwert, das L2 hat die größere Streuung.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Optimaler Transport: vom Plan zum Abstand](https://sebastianhanisch.net/konzepte-optimaler-transport.html)."
)
