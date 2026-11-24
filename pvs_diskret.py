"""Diskrete Verteilungen auf der Zeitachse in Brüchen (`fractions`, nur Standardbibliothek): die zwei Mitten exakt gerechnet (DEMO-PLAYBOOK §1b).

Eine Verteilung ist eine sortierte Liste `(Ort, Masse)`; alle Größen sind `Fraction`. Gerechnet wird ohne Rundung:

* `w2_quadrat`: $W_2^2 = \\int_0^1 (Q_a - Q_b)^2\\,\\mathrm{d}u$ aus den Quantilfunktionen (Treppen), Stück für Stück zwischen den Sprungstellen;
* `quantilmittel`: das Quantilmittel $\\sum_k \\lambda_k Q_k$ (eindimensionales Wasserstein-Baryzentrum), wieder eine diskrete Verteilung;
* `mischung`: die Mischung $\\sum_k \\lambda_k P_k$;
* `tv`: der Totalvariationsabstand;
* `gleichstand_beispiel`: zwei gleich teure Zuordnungen mit verschiedenen Zwischenverteilungen (Kosten $|x - y|$) gegen die eindeutige bei Kosten $(x - y)^2$.
"""

from __future__ import annotations

from fractions import Fraction as F
from itertools import permutations


def normiert(atome) -> list:
    """Sortierte Liste (Ort, Masse): gleiche Orte zusammengefasst, Masse 0 weggelassen."""
    d: dict = {}
    for x, m in atome:
        d[F(x)] = d.get(F(x), F(0)) + F(m)
    return sorted((x, m) for x, m in d.items() if m != 0)


def verschoben(atome, d) -> list:
    return normiert((x + F(d), m) for x, m in atome)


def _knickstellen(verteilungen) -> list:
    punkte = {F(0), F(1)}
    for atome in verteilungen:
        c = F(0)
        for _, m in atome:
            c += m
            punkte.add(c)
    return sorted(punkte)


def quantil(atome, u) -> F:
    """Quantil an einer Stelle u zwischen zwei Sprungstellen (dort ist die Treppe konstant)."""
    c = F(0)
    for x, m in atome:
        c += m
        if u < c:
            return x
    return atome[-1][0]


def w2_quadrat(a, b) -> F:
    """Exaktes $W_2^2$ (Einheit Stunden²) zweier diskreter Verteilungen gleicher Gesamtmasse 1."""
    punkte = _knickstellen([a, b])
    return sum((hi - lo) * (quantil(a, (lo + hi) / 2) - quantil(b, (lo + hi) / 2)) ** 2 for lo, hi in zip(punkte[:-1], punkte[1:]))


def quantilmittel(verteilungen, gewichte) -> list:
    """Quantilmittel: auf jedem Stück zwischen den Sprungstellen der Mittelwert der Quantile, gewichtet."""
    punkte = _knickstellen(verteilungen)
    atome = []
    for lo, hi in zip(punkte[:-1], punkte[1:]):
        mitte = (lo + hi) / 2
        atome.append((sum(F(w) * quantil(v, mitte) for v, w in zip(verteilungen, gewichte)), hi - lo))
    return normiert(atome)


def mischung(verteilungen, gewichte) -> list:
    atome = []
    for v, w in zip(verteilungen, gewichte):
        atome += [(x, F(w) * m) for x, m in v]
    return normiert(atome)


def mittel_varianz(atome) -> tuple[F, F]:
    mu = sum(x * m for x, m in atome)
    return mu, sum((x - mu) ** 2 * m for x, m in atome)


def tv(a, b) -> F:
    d: dict = {}
    for x, m in a:
        d[x] = d.get(x, F(0)) + m
    for x, m in b:
        d[x] = d.get(x, F(0)) - m
    return sum(abs(v) for v in d.values()) / 2


def masse_zwischen(atome, unten, oben) -> F:
    """Masse echt zwischen zwei Orten (beide ausgeschlossen)."""
    return sum((m for x, m in atome if F(unten) < x < F(oben)), F(0))


def zielwert(kandidat, standorte, gewichte) -> F:
    """Zielfunktion des Baryzentrums: gewichtetes Mittel der $W_2^2$ zu den Standorten."""
    return sum(F(w) * w2_quadrat(s, kandidat) for s, w in zip(standorte, gewichte))


def tagesprofil_atome(morgenanteil) -> list:
    """Tagesprofil mit zwei Spitzen, je eine Punktmasse bei 8 bzw. 18 Uhr: Morgenanteil bei 8 Uhr, der Rest bei 18 Uhr."""
    w = F(morgenanteil)
    return normiert([(F(8), w), (F(18), 1 - w)])


# ---------------------------------------------------------------- Gleichstand (Kante: Zuordnung → Zwischenverteilung)


def zuordnungen(p, q) -> list:
    """Alle Zuordnungen zweier Verteilungen mit je n gleichen Massen 1/n: Liste von Listen (x, y)."""
    xs = [x for x, _ in p]
    ys = [y for y, _ in q]
    return [list(zip(xs, perm)) for perm in permutations(ys)]


def zuordnung_kosten(zuordnung, potenz: int) -> F:
    """Mittlere Kosten $|x - y|^p$ einer Zuordnung (jede Masse 1/n)."""
    return sum((abs(x - y) ** potenz for x, y in zuordnung), F(0)) / len(zuordnung)


def zwischenverteilung(zuordnung, t) -> list:
    """Verteilung nach dem Bruchteil t des Weges jeder Masse entlang der Zuordnung."""
    n = len(zuordnung)
    return normiert(((1 - F(t)) * x + F(t) * y, F(1, n)) for x, y in zuordnung)


def gleichstand_beispiel(t=F(1, 2)) -> dict:
    """P = (0, 1), Q = (2, 3), je Masse 1/2. Kosten |x − y|: die zwei Zuordnungen kosten gleich viel, ihre Zwischenverteilungen sind verschieden.
    Kosten (x − y)²: genau eine Zuordnung ist optimal, und sie liefert das Quantilmittel."""
    p = normiert([(0, F(1, 2)), (1, F(1, 2))])
    q = normiert([(2, F(1, 2)), (3, F(1, 2))])
    zs = zuordnungen(p, q)
    k1 = [zuordnung_kosten(z, 1) for z in zs]
    k2 = [zuordnung_kosten(z, 2) for z in zs]
    optimal1 = [z for z, k in zip(zs, k1) if k == min(k1)]
    optimal2 = [z for z, k in zip(zs, k2) if k == min(k2)]
    return {
        "p": p, "q": q, "zuordnungen": zs, "kosten1": k1, "kosten2": k2,
        "optimal1": optimal1, "optimal2": optimal2,
        "zwischen1": [zwischenverteilung(z, t) for z in optimal1],
        "zwischen2": [zwischenverteilung(z, t) for z in optimal2],
        "quantilmittel": quantilmittel([p, q], [1 - F(t), F(t)]),
    }
