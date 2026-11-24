"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen über die zwei Mitten, die ohne Kenntnis des Ergebnisses gelten; je 20 bis 40 Zufallsinstanzen aus dem Hausgenerator (fester Zufalls-Seed)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import pvs_diskret as D
import pvs_profile as P
import pvs_rng
import pvs_verteilung as V
from pvs_verteilung import Mix, mischung, normal, verschoben


def zufalls_atome(rng, n=4, nenner=12, hoechstort=30):
    """Eine Verteilung mit bis zu n Orten (ganze Stunden), Massen k/nenner mit Summe 1."""
    schnitte = sorted(rng.below(nenner + 1) for _ in range(n - 1))
    massen = [b - a for a, b in zip([0] + schnitte, schnitte + [nenner])]
    return D.normiert([(rng.below(hoechstort + 1), F(m, nenner)) for m in massen if m > 0])


def drei_standorte(rng):
    return [zufalls_atome(rng) for _ in range(3)]


DRITTEL = [F(1, 3)] * 3


def test_verschieben_aller_standorte_um_c_verschiebt_das_baryzentrum_um_c():
    rng = pvs_rng.SplitMix64(101)
    for _ in range(30):
        st = drei_standorte(rng)
        c = rng.below(9) - 4
        assert D.quantilmittel([D.verschoben(s, c) for s in st], DRITTEL) == D.verschoben(D.quantilmittel(st, DRITTEL), c)


def test_vertauschen_der_standorte_aendert_das_baryzentrum_nicht():
    rng = pvs_rng.SplitMix64(102)
    for _ in range(30):
        st = drei_standorte(rng)
        w = [F(1, 6), F(1, 3), F(1, 2)]
        zuordnung = list(range(3))
        rng.shuffle(zuordnung)
        assert D.quantilmittel(st, w) == D.quantilmittel([st[i] for i in zuordnung], [w[i] for i in zuordnung])


def test_ein_standort_mit_gewicht_eins_gibt_dieses_profil():
    rng = pvs_rng.SplitMix64(103)
    for _ in range(30):
        st = drei_standorte(rng)
        j = rng.below(3)
        gewichte = [F(1) if i == j else F(0) for i in range(3)]
        assert D.quantilmittel(st, gewichte) == st[j]
        assert D.mischung(st, gewichte) == st[j]


def test_t_gleich_null_und_eins_geben_die_enden():
    rng = pvs_rng.SplitMix64(104)
    for _ in range(30):
        a, b = zufalls_atome(rng), zufalls_atome(rng)
        assert D.quantilmittel([a, b], [F(1), F(0)]) == a and D.quantilmittel([a, b], [F(0), F(1)]) == b
        assert D.mischung([a, b], [F(1), F(0)]) == a and D.mischung([a, b], [F(0), F(1)]) == b


def test_w2_zum_anfang_waechst_linear_in_t_bei_der_verschiebung_und_tv_bei_der_mischung():
    rng = pvs_rng.SplitMix64(105)
    for _ in range(30):
        a, b = zufalls_atome(rng), zufalls_atome(rng)
        t = F(rng.below(9) + 1, 10)
        assert D.w2_quadrat(a, D.quantilmittel([a, b], [1 - t, t])) == t ** 2 * D.w2_quadrat(a, b)      # W2 = t · W2(A, B), exakt
        assert D.tv(a, D.mischung([a, b], [1 - t, t])) == t * D.tv(a, b)                                  # TV = t · TV(A, B), exakt


def test_mischung_und_baryzentrum_haben_dasselbe_mittel_und_die_streuungskette_gilt():
    rng = pvs_rng.SplitMix64(106)
    for _ in range(40):
        st = drei_standorte(rng)
        bary, mix = D.quantilmittel(st, DRITTEL), D.mischung(st, DRITTEL)
        mittel = sum(D.mittel_varianz(s)[0] for s in st) / 3
        assert D.mittel_varianz(bary)[0] == D.mittel_varianz(mix)[0] == mittel
        sd = lambda atome: math.sqrt(D.mittel_varianz(atome)[1])
        # Streuung des Baryzentrums ≤ mittlere Streuung der Standorte ≤ Streuung der Mischung (Minkowski und totale Varianz)
        assert sd(bary) <= sum(sd(s) for s in st) / 3 + 1e-12
        assert sum(sd(s) for s in st) / 3 <= sd(mix) + 1e-12


def test_spiegeln_der_zeitachse_spiegelt_beide_mitten():
    rng = pvs_rng.SplitMix64(107)
    spiegel = lambda atome: D.normiert((-x, m) for x, m in atome)
    for _ in range(30):
        st = drei_standorte(rng)
        assert D.quantilmittel([spiegel(s) for s in st], DRITTEL) == spiegel(D.quantilmittel(st, DRITTEL))
        assert D.mischung([spiegel(s) for s in st], DRITTEL) == spiegel(D.mischung(st, DRITTEL))


def test_quantilfunktionen_sind_monoton_und_die_gauss_fassung_verschiebt_mit():
    rng = pvs_rng.SplitMix64(108)
    u, _ = V.stufen()
    for _ in range(20):
        w = 0.2 + 0.6 * rng.random()
        sites = [Mix((8.0, 18.0), (1.2, 1.2), (w + d, 1 - w - d)) for d in (-0.1, 0.0, 0.1)]
        g = V.gitter(sites + [verschoben(sites[0], 2.0)])
        qs = [V.quantile(V.cdf(m, g), g, u) for m in sites]
        bary = V.quantilmittel(qs, [1 / 3] * 3)
        assert np.all(np.diff(bary) >= 0)
        qs2 = [V.quantile(V.cdf(verschoben(m, 2.0), g), g, u) for m in sites]
        assert np.abs(V.quantilmittel(qs2, [1 / 3] * 3) - bary - 2.0).max() < 1e-9      # alle um 2 Stunden verschoben: das Baryzentrum auch (2 ist ein Vielfaches des Gitterschritts)


def test_komponentenreihenfolge_einer_mischung_aendert_nichts():
    rng = pvs_rng.SplitMix64(109)
    for _ in range(10):
        a, b = normal(rng.random() * 10, 1.0 + rng.random()), normal(rng.random() * 10, 1.0 + rng.random())
        m1, m2 = mischung([a, b], [0.3, 0.7]), mischung([b, a], [0.7, 0.3])
        x = np.linspace(-5, 15, 101)
        assert np.allclose(V.pdf(m1, x), V.pdf(m2, x), rtol=1e-12) and np.allclose(V.cdf(m1, x), V.cdf(m2, x), rtol=1e-12)


def test_bei_quadratkosten_ist_die_sortierte_zuordnung_die_einzige_optimale():
    rng = pvs_rng.SplitMix64(110)
    geprueft = 0
    for _ in range(40):
        n = 3
        xs, ys = sorted(rng.below(20) for _ in range(n)), sorted(rng.below(20) for _ in range(n))
        p = D.normiert([(x, F(1, n)) for x in xs])
        q = D.normiert([(y, F(1, n)) for y in ys])
        if len(p) < n or len(q) < n:                                      # gleiche Orte fassen Massen zusammen: dann nicht n gleiche Massen
            continue
        kosten = [D.zuordnung_kosten(z, 2) for z in D.zuordnungen(p, q)]
        assert kosten[0] == min(kosten) == D.w2_quadrat(p, q)             # die sortierte Zuordnung (erste in der Aufzählung) ist optimal und ihr Wert ist W2²
        assert sum(1 for k in kosten if k == min(kosten)) == 1            # und bei lauter verschiedenen Orten die einzige
        geprueft += 1
    assert geprueft >= 20
