"""Plotly-Figuren: alle Achsen fest (fixedrange), Beschriftungen als Annotationen, ein Rahmen je Figur."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pvs_constants as C

SCHRIFT = dict(size=12)
MAX_PUNKTE = 1600                                    # so viele Punkte je Kurve gehen höchstens an den Browser
GRAU = "rgba(120,120,120,0.7)"


def _fest(fig: go.Figure, hoehe: int, titel: str | None = None) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=hoehe, margin=dict(l=55, r=20, t=50 if titel else 40, b=85), title=dict(text=titel, x=0.0, font=dict(size=14)) if titel else None,
                      dragmode=False, legend=dict(orientation="h", y=-0.3, x=0.0), font=SCHRIFT)
    return fig


def duenn(x, y, lo: float | None = None, hi: float | None = None, n: int = MAX_PUNKTE):
    """Kurve auf den sichtbaren Bereich [lo, hi] beschränkt und gleichmäßig auf höchstens n Punkte ausgedünnt (Browser-Übertragung klein halten)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if lo is not None:
        sel = (x >= lo) & (x <= hi)
        x, y = x[sel], y[sel]
    if len(x) > n:
        idx = np.unique(np.linspace(0, len(x) - 1, n).round().astype(int))
        x, y = x[idx], y[idx]
    return np.round(x, 4), np.round(y, 6)


def sichtbar(x: np.ndarray, kurven, schwelle: float = 2e-3) -> tuple[float, float]:
    """Bereich der Zeitachse, in dem eine der Dichten über `schwelle` mal ihrem Höchstwert liegt, mit etwas Rand."""
    hoechst = max(float(np.max(k)) for k in kurven)
    aktiv = np.zeros(len(x), dtype=bool)
    for k in kurven:
        aktiv |= np.asarray(k) > schwelle * hoechst
    xs = np.asarray(x)[aktiv]
    rand = 0.05 * (xs.max() - xs.min())
    return float(xs.min() - rand), float(xs.max() + rand)


# ---------------------------------------------------------------- Experiment 1


def build_dichten1(r: dict) -> go.Figure:
    """Links die Mischung, rechts die Verschiebung beim gewählten t; die Enden A und B gestrichelt."""
    t = r["t_sel"]
    mi, ve = r["mischung"][t], r["verschiebung"][t]
    x = r["x"]
    lo, hi = sichtbar(x, [r["fa"], r["fb"], mi["dichte"], np.interp(x, ve["pos"], ve["dichte"], left=0.0, right=0.0)])
    ymax = 1.08 * max(float(np.max(r["fa"])), float(np.max(r["fb"])), float(np.max(mi["dichte"])), float(np.max(ve["dichte"])))
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Mischung (1 − t)·A + t·B, t = {t:g}".replace(".", ","), f"Verschiebung (Quantilmittel), t = {t:g}".replace(".", ",")), horizontal_spacing=0.08)
    for spalte in (1, 2):
        fig.add_trace(go.Scatter(x=duenn(x, r["fa"], lo, hi)[0], y=duenn(x, r["fa"], lo, hi)[1], mode="lines", name="Anfang A", line=dict(color=C.FARBE_ENDEN, dash="dash"),
                                 legendgroup="a", showlegend=spalte == 1), row=1, col=spalte)
        fig.add_trace(go.Scatter(x=duenn(x, r["fb"], lo, hi)[0], y=duenn(x, r["fb"], lo, hi)[1], mode="lines", name="Ende B", line=dict(color=C.FARBE_GRAU, dash="dot"),
                                 legendgroup="b", showlegend=spalte == 1), row=1, col=spalte)
    mx, my = duenn(x, mi["dichte"], lo, hi)
    fig.add_trace(go.Scatter(x=mx, y=my, mode="lines", name="Mischung", line=dict(color=C.FARBE_MISCHUNG, width=3), fill="tozeroy", fillcolor="rgba(217,95,2,0.15)"), row=1, col=1)
    vx, vy = duenn(ve["pos"], ve["dichte"], lo, hi)
    fig.add_trace(go.Scatter(x=vx, y=vy, mode="lines", name="Verschiebung", line=dict(color=C.FARBE_VERSCHIEBUNG, width=3), fill="tozeroy", fillcolor="rgba(27,158,119,0.15)"), row=1, col=2)
    fig.update_xaxes(title="Zeit oder Lage (Stunden)", range=[lo, hi])
    fig.update_yaxes(title="Dichte", range=[0, ymax], row=1, col=1)
    fig.update_yaxes(range=[0, ymax], row=1, col=2)
    fig.add_annotation(text=f"{mi['gipfel']} Gipfel", x=0.02, y=0.98, xref="x domain", yref="y domain", showarrow=False, xanchor="left", yanchor="top", font=dict(color=C.FARBE_MISCHUNG, size=13))
    fig.add_annotation(text=f"{ve['gipfel']} Gipfel", x=0.02, y=0.98, xref="x2 domain", yref="y2 domain", showarrow=False, xanchor="left", yanchor="top", font=dict(color=C.FARBE_VERSCHIEBUNG, size=13))
    return _fest(fig, 360)


def build_abstand1(r: dict) -> go.Figure:
    """Abstand zum Anfang gegen t, auf den Abstand A–B bezogen: links W2 (die Verschiebung ist die Gerade), rechts TV (die Mischung ist die Gerade)."""
    ts = list(C.T1_KURVE)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("W2 zum Anfang (Anteil von W2(A, B))", "TV zum Anfang (Anteil von TV(A, B))"), horizontal_spacing=0.1)
    for spalte, (schluessel, ende) in enumerate((("w2", r["w2_ab"]), ("tv", r["tv_ab"])), start=1):
        nenner = ende if ende > 1e-12 else 1.0
        fig.add_trace(go.Scatter(x=ts, y=ts, mode="lines", name="Gerade", line=dict(color=GRAU, dash="dot"), showlegend=spalte == 1), row=1, col=spalte)
        fig.add_trace(go.Scatter(x=ts, y=[r["mischung"][t][schluessel] / nenner for t in ts], mode="lines+markers", name="Mischung", line=dict(color=C.FARBE_MISCHUNG), showlegend=spalte == 1), row=1, col=spalte)
        fig.add_trace(go.Scatter(x=ts, y=[r["verschiebung"][t][schluessel] / nenner for t in ts], mode="lines+markers", name="Verschiebung", line=dict(color=C.FARBE_VERSCHIEBUNG), showlegend=spalte == 1), row=1, col=spalte)
        fig.add_vline(x=r["t_sel"], line=dict(color=GRAU, dash="dash"), row=1, col=spalte)
    fig.update_xaxes(title="t", range=[0, 1])
    fig.update_yaxes(range=[0, 1.05])
    return _fest(fig, 330)


# ---------------------------------------------------------------- Experiment 2


def build_profile2(r: dict, frage: str) -> go.Figure:
    """Je nach Frage: die Summe der Standorte gegen dreimal das Baryzentrum (mit den Spitzenfenstern), oder die drei Profile mit Mischung und Baryzentrum."""
    x = r["x"]
    fig = go.Figure()
    bx, by = r["pos_bary"], r["dichte_bary"]
    if frage == C.FRAGE2[0]:
        faktor = float(C.STANDORTE)
        sx, sy = duenn(x, faktor * r["dichte_mischung"], 0, C.TAG)
        px, py = duenn(bx, faktor * by, 0, C.TAG, n=MAX_PUNKTE)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name="Summe der drei Standorte", line=dict(color=C.FARBE_MISCHUNG, width=3), fill="tozeroy", fillcolor="rgba(217,95,2,0.15)"))
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", name="dreimal das Baryzentrum", line=dict(color=C.FARBE_VERSCHIEBUNG, width=3, dash="dash")))
        for a, b in C.FENSTER:
            fig.add_vrect(x0=a, x1=b, fillcolor="rgba(120,120,120,0.10)", line_width=0)
        fig.update_yaxes(title="Nachfrage je Stunde (Summe der Anteile der Standorte)", rangemode="tozero")
        fig.add_annotation(text="Spitzenfenster 8 ± 2,4 und 18 ± 2,4 Uhr", x=0.5, y=1.0, xref="paper", yref="paper", showarrow=False, yanchor="bottom", font=dict(color=GRAU, size=11))
    else:
        for k, (dk, farbe) in enumerate(zip(r["dichten_standorte"], C.FARBE_STANDORT)):
            sx, sy = duenn(x, dk, 0, C.TAG)
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name=f"Standort {k + 1}", line=dict(color=farbe, width=1.5)))
        mx, my = duenn(x, r["dichte_mischung"], 0, C.TAG)
        px, py = duenn(bx, by, 0, C.TAG)
        fig.add_trace(go.Scatter(x=mx, y=my, mode="lines", name="Mischung", line=dict(color=C.FARBE_MISCHUNG, width=3)))
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", name="Baryzentrum", line=dict(color=C.FARBE_VERSCHIEBUNG, width=3, dash="dash")))
        fig.update_yaxes(title="Dichte (Anteil der Tagesnachfrage je Stunde)", rangemode="tozero")
    fig.update_xaxes(title="Uhrzeit (Stunden)", range=[0, C.TAG], tickvals=list(range(0, 25, 4)))
    fig.add_annotation(text=f"Mischung: {r['gipfel_mischung']} Gipfel; Baryzentrum: {r['gipfel_bary']}", x=0.99, y=0.98, xref="paper", yref="paper", showarrow=False, xanchor="right", yanchor="top", font=dict(size=12))
    return _fest(fig, 380)


def build_gipfel2(s_werte, gipfel, s: float, prom: float) -> go.Figure:
    """Zahl der Gipfel der Mischung gegen s; das Baryzentrum hat immer die zwei Gipfel des Profils."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(s_werte), y=list(gipfel), mode="lines+markers", name="Mischung", line=dict(color=C.FARBE_MISCHUNG, shape="hv"), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=[min(s_werte), max(s_werte)], y=[2, 2], mode="lines", name="Baryzentrum (das Profil)", line=dict(color=C.FARBE_VERSCHIEBUNG, dash="dash")))
    fig.add_vline(x=s, line=dict(color=GRAU, dash="dash"))
    fig.update_xaxes(title="Phasenversatz s (Stunden)", range=[min(s_werte) - 0.1, max(s_werte) + 0.1])
    fig.update_yaxes(title="Zahl der Gipfel", range=[0, max(7, max(gipfel) + 1)], dtick=1)
    fig.add_annotation(text=f"Prominenz mindestens {100 * prom:g} % der höchsten Dichte".replace(".", ","), x=0.01, y=0.02, xref="paper", yref="paper", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=GRAU, size=11))
    return _fest(fig, 300)


# ---------------------------------------------------------------- Experiment 3


def build_dichten3(r: dict) -> go.Figure:
    """Die drei Standorte, die Mischung und das Baryzentrum; schraffiert der Bereich zwischen den Spitzen."""
    x = r["x"]
    fig = go.Figure()
    for k, (dk, farbe) in enumerate(zip(r["dichten_standorte"], C.FARBE_STANDORT)):
        sx, sy = duenn(x, dk, 0, C.TAG)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name=f"Standort {k + 1}", line=dict(color=farbe, width=1.5)))
    mx, my = duenn(x, r["dichte_mischung"], 0, C.TAG)
    px, py = duenn(r["pos_bary"], r["dichte_bary"], 0, C.TAG)
    fig.add_trace(go.Scatter(x=mx, y=my, mode="lines", name="Mischung", line=dict(color=C.FARBE_MISCHUNG, width=3)))
    fig.add_trace(go.Scatter(x=px, y=py, mode="lines", name="Baryzentrum", line=dict(color=C.FARBE_VERSCHIEBUNG, width=3, dash="dash")))
    fig.add_vrect(x0=C.LEER[0], x1=C.LEER[1], fillcolor="rgba(120,120,120,0.12)", line_width=0)
    fig.add_annotation(text=f"zwischen den Spitzen: Baryzentrum {100 * r['leer_bary']:.1f} %, Mischung {100 * r['leer_mischung']:.1f} %".replace(".", ","),
                       x=0.5 * (C.LEER[0] + C.LEER[1]), y=1.0, xref="x", yref="paper", showarrow=False, yanchor="bottom", font=dict(size=11))
    fig.update_xaxes(title="Uhrzeit (Stunden)", range=[0, C.TAG], tickvals=list(range(0, 25, 4)))
    fig.update_yaxes(title="Dichte (Anteil der Tagesnachfrage je Stunde)", rangemode="tozero")
    return _fest(fig, 360)


def build_kurve3(k: dict, delta: float) -> go.Figure:
    """Anteil der Masse zwischen den Spitzen gegen δ."""
    fig = go.Figure()
    d = k["delta"]
    fig.add_trace(go.Scatter(x=d, y=k["bary"], mode="lines+markers", name="Baryzentrum (Normalverteilungen)", line=dict(color=C.FARBE_VERSCHIEBUNG)))
    fig.add_trace(go.Scatter(x=d, y=k["atome"], mode="lines+markers", name="Baryzentrum (Atome, exakt 2δ)", line=dict(color=C.FARBE_VERSCHIEBUNG, dash="dash"), marker=dict(symbol="diamond")))
    fig.add_trace(go.Scatter(x=d, y=k["mischung"], mode="lines+markers", name="Mischung", line=dict(color=C.FARBE_MISCHUNG)))
    fig.add_trace(go.Scatter(x=d, y=k["standort"], mode="lines", name="Mittel der Standorte", line=dict(color=C.FARBE_GRAU, dash="dot")))
    fig.add_vline(x=delta, line=dict(color=GRAU, dash="dash"))
    fig.update_xaxes(title="Spreizung δ der Morgenanteile", range=[-0.005, C.DELTA3_MAX + 0.005])
    fig.update_yaxes(title="Masse zwischen den Spitzen", range=[0, max(0.45, max(k["atome"]) * 1.1)], tickformat=".0%")
    return _fest(fig, 320)
