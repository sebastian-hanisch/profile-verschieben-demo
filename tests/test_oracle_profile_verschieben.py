"""Orakel: unabhängige Rechenwege für die Kerne der Demo (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5), jeder Test unter 10 s.

* Fehlerfunktion und Verteilungsfunktion gegen `math.erfc` und scipy; die kopierten Funktionen gegen den Quelltext der Quelle.
* Das Quantilmittel in Brüchen gegen ein Baryzentrum-Linearprogramm (HiGHS) auf festem Raster und gegen 100 zufällige Kandidaten.
* $W_2^2$ in Brüchen gegen die Aufzählung aller Zuordnungen (gleiche Massen).
* Das Quantilmittel zweier Normalverteilungen und W2 gegen die geschlossene Form (Lage und Streuung wechseln linear; $W_2^2 = \\Delta\\mu^2 + \\Delta\\sigma^2$).
* Quantile einer Mischung gegen die Verteilungsfunktion aus scipy und gegen Ziehungen aus der Definition (Komponente wählen, dann Normalzahl).
* Gipfelzählung gegen `scipy.signal.find_peaks` auf zufälligen Feldern und auf den echten Dichten.
* TV zweier Normalverteilungen gegen die geschlossene Form $2\\Phi(D/2) - 1$."""

import inspect
import itertools
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

import pvs_constants as C
import pvs_diskret as D
import pvs_evaluation as E
import pvs_normal as N
import pvs_profile as P
import pvs_rng
import pvs_verteilung as V
from pvs_verteilung import Mix, mischung, normal

ROOT = Path(__file__).resolve().parent.parent
PORTFOLIO = ROOT.parent
scipy_stats = pytest.importorskip("scipy.stats")
scipy_signal = pytest.importorskip("scipy.signal")
scipy_optimize = pytest.importorskip("scipy.optimize")


# ---------------------------------------------------------------- Normalverteilung ohne scipy


def test_erfc_gegen_math_erfc_und_verteilungsfunktion_gegen_scipy():
    x = np.linspace(-9, 9, 721)
    assert np.allclose(N.erfc(x), [math.erfc(v) for v in x], rtol=1e-9, atol=1e-300)
    assert np.allclose(N.norm_cdf(x), scipy_stats.norm.cdf(x), rtol=1e-9, atol=1e-300)
    assert np.allclose(N.norm_pdf(x), scipy_stats.norm.pdf(x), rtol=1e-12, atol=0)


def test_kopierte_funktionen_stimmen_mit_der_quelle_ueberein_falls_vorhanden():
    quelle = PORTFOLIO / "mitte-im-raum-demo" / "mir_dist.py"
    if not quelle.exists():
        pytest.skip("mitte-im-raum-demo nicht vorhanden")
    import importlib.util

    spec = importlib.util.spec_from_file_location("mir_dist_quelle", quelle)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    for name in ("erfc", "norm_cdf", "norm_pdf"):
        assert inspect.getsource(getattr(N, name)) == inspect.getsource(getattr(modul, name)), name
    for konstante in ("SQRT2", "SQRT_PI", "SERIES_LIMIT", "SERIES_TERMS", "FRACTION_TERMS"):
        assert getattr(N, konstante) == getattr(modul, konstante), konstante


def test_zufallsgenerator_ist_bytegleich_zur_vorlage_falls_vorhanden():
    vorlage = PORTFOLIO / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage nicht vorhanden")
    assert (ROOT / "pvs_rng.py").read_bytes() == vorlage.read_bytes()


# ---------------------------------------------------------------- Brüche gegen Aufzählung und Linearprogramm


def _zufalls_atome(rng, n, nenner=10):
    """n Orte (ganze Stunden 0 bis 24), Massen k/nenner mit Summe 1 (Zufallsaufteilung)."""
    schnitte = sorted(rng.below(nenner + 1) for _ in range(n - 1))
    massen = [b - a for a, b in zip([0] + schnitte, schnitte + [nenner])]
    return D.normiert([(rng.below(25), F(m, nenner)) for m in massen if m > 0])


def test_w2_quadrat_in_brüchen_gegen_aufzaehlung_aller_zuordnungen():
    rng = pvs_rng.SplitMix64(2026)
    for _ in range(25):
        n = 5
        xs = [rng.below(25) for _ in range(n)]
        ys = [rng.below(25) for _ in range(n)]
        a = D.normiert([(x, F(1, n)) for x in xs])
        b = D.normiert([(y, F(1, n)) for y in ys])
        bester = min(sum(F(x - y) ** 2 for x, y in zip(xs, perm)) for perm in itertools.permutations(ys)) / n
        assert D.w2_quadrat(a, b) == bester                    # der sortierte Plan ist die beste Zuordnung


def test_quantilmittel_gegen_baryzentrum_linearprogramm_auf_festem_raster():
    """Das Quantilmittel für Morgenanteile 3/10, 4/10, 5/10 hat den Zielwert 40/9; das Linearprogramm über alle Verteilungen auf dem Raster 1/3 Stunde von 8 bis 18 Uhr findet nichts Besseres."""
    anteile = [F(3, 10), F(4, 10), F(5, 10)]
    raster = [F(24 + i, 3) for i in range(31)]                 # 8, 8 + 1/3, …, 18
    g = len(raster)
    gf = np.array([float(x) for x in raster])
    kosten = (gf[:, None] - gf[None, :]) ** 2
    k = len(anteile)
    nvar = k * g * g + g
    ziel = np.zeros(nvar)
    for j in range(k):
        ziel[j * g * g:(j + 1) * g * g] = kosten.ravel() / k
    zeilen, rhs = [], []
    for j in range(k):
        nu = np.zeros(g)
        nu[0], nu[-1] = float(anteile[j]), float(1 - anteile[j])
        for i in range(g):
            z = np.zeros(nvar)
            z[j * g * g + i * g:j * g * g + (i + 1) * g] = 1
            zeilen.append(z)
            rhs.append(nu[i])
        for m in range(g):
            z = np.zeros(nvar)
            z[j * g * g + m:(j + 1) * g * g:g] = 1
            z[k * g * g + m] = -1
            zeilen.append(z)
            rhs.append(0.0)
    lp = scipy_optimize.linprog(ziel, A_eq=np.array(zeilen), b_eq=np.array(rhs), bounds=(0, None), method="highs")
    assert lp.status == 0
    standorte = [D.tagesprofil_atome(w) for w in anteile]
    assert lp.fun == pytest.approx(float(D.zielwert(D.quantilmittel(standorte, [F(1, 3)] * 3), standorte, [F(1, 3)] * 3)), abs=1e-9)   # 40/9


def test_baryzentrum_ist_nie_schlechter_als_zufaellige_kandidaten_auf_dem_raster():
    standorte = [D.tagesprofil_atome(w) for w in (F(3, 10), F(4, 10), F(5, 10))]
    gewichte = [F(1, 3)] * 3
    bestes = D.zielwert(D.quantilmittel(standorte, gewichte), standorte, gewichte)
    rng = pvs_rng.SplitMix64(7)
    for _ in range(100):
        kandidat = _zufalls_atome(rng, 4)
        kandidat = D.normiert([(F(8) + F(rng.below(31), 3), m) for _, m in kandidat])        # Orte auf dem Raster 1/3 Stunde zwischen 8 und 18 Uhr
        assert D.zielwert(kandidat, standorte, gewichte) >= bestes


# ---------------------------------------------------------------- Normalverteilungen: geschlossene Form


@pytest.mark.parametrize("fall,d,s1", [(C.FALL1[0], 4.0, 1.0), (C.FALL1[0], 1.0, 1.0), (C.FALL1[1], 4.0, 3.0), (C.FALL1[1], 0.0, 3.0)])
def test_verschiebung_zweier_normalverteilungen_hat_lineare_lage_und_streuung(fall, d, s1):
    r = E.reihe1(fall, d, 0.5)
    for t in C.T1_SCHRITTE:
        v = r["verschiebung"][t]
        assert v["streuung"] == pytest.approx((1 - t) * 1.0 + t * s1, abs=2e-4)           # Streuung linear (Quadraturband der Stufen)
        assert v["w2"] == pytest.approx(t * math.hypot(d, s1 - 1.0), abs=2e-4)             # W2 liegt auf der Geodäte: t · √(Δμ² + Δσ²)
    # Dichte der Verschiebung gegen die Normalverteilung N(t d, ((1 − t) + t s1)²) an den Stellen der Quantile
    t = 0.5
    ve = E.reihe1(fall, d, t)["verschiebung"][t]
    mu, sd = t * d, (1 - t) + t * s1
    z = (ve["pos"] - mu) / sd
    mitte = np.abs(z) < 5
    erwartet = scipy_stats.norm.pdf(ve["pos"], mu, sd)
    assert np.abs(ve["dichte"][mitte] - erwartet[mitte]).max() < 1e-5


def test_w2_zwischen_normalverteilungen_gegen_die_geschlossene_form():
    u, w = V.stufen()
    for (m0, s0, m1, s1) in ((0, 1, 4, 1), (0, 1, 0, 3), (2, 0.7, -1.5, 2.2)):
        a, b = normal(m0, s0), normal(m1, s1)
        g = V.gitter([a, b])
        w2 = V.w2_quadrat(V.quantile(V.cdf(a, g), g, u), V.quantile(V.cdf(b, g), g, u), w)
        assert w2 == pytest.approx((m1 - m0) ** 2 + (s1 - s0) ** 2, abs=3e-4)


@pytest.mark.parametrize("d", [0.5, 2.0, 4.0, 8.0])
def test_tv_zweier_normalverteilungen_gegen_die_geschlossene_form(d):
    a, b = normal(0, 1), normal(d, 1)
    x = V.gitter([a, b])
    assert V.tv_auf_gitter(V.pdf(a, x), V.pdf(b, x)) == pytest.approx(2 * scipy_stats.norm.cdf(d / 2) - 1, abs=1e-6)


def test_mischung_zweier_normalverteilungen_hat_die_geschlossene_streuung():
    d = 4.0
    r = E.reihe1(C.FALL1[0], d, 0.5)
    t = 0.5
    assert r["mischung"][t]["streuung"] == pytest.approx(math.sqrt(1 + t * (1 - t) * d ** 2), abs=1e-12)    # totale Varianz: 1 + t(1 − t) D²


# ---------------------------------------------------------------- Quantile einer Mischung


def test_quantile_einer_mischung_gegen_verteilungsfunktion_aus_scipy():
    m = mischung(P.standorte_versetzt(3.0), [1 / 3] * 3)
    g = V.gitter([m])
    u, _ = V.stufen()
    q = V.quantile(V.cdf(m, g), g, u)
    scipy_cdf = sum(gw * scipy_stats.norm.cdf(q, mu, s) for mu, s, gw in zip(m.loc, m.scale, m.weight))
    assert np.abs(scipy_cdf - u).max() < 1e-7


def test_quantile_einer_mischung_gegen_ziehungen_aus_der_definition():
    """Definition: erst die Komponente wählen (Anteil w_j), dann eine Normalzahl; die empirischen Quantile liegen im Band von vier Standardfehlern."""
    m = mischung([P.profil(), P.profil(0.3)], [0.5, 0.5])
    rng = pvs_rng.SplitMix64(11)
    n = 6000
    proben = np.sort([m.loc[(j := rng.choice(m.weight))] + m.scale[j] * rng.normal() for _ in range(n)])
    g = V.gitter([m])
    for u in (0.1, 0.25, 0.5, 0.75, 0.9):
        q = V.quantile(V.cdf(m, g), g, np.array([u]))[0]
        dichte = float(V.pdf(m, q))
        assert abs(proben[int(u * n)] - q) < 4 * math.sqrt(u * (1 - u) / n) / dichte, u


# ---------------------------------------------------------------- Gipfelzählung gegen scipy


def _scipy_gipfel(f, prom):
    f = np.concatenate([[0.0], f, [0.0]])
    pk, _ = scipy_signal.find_peaks(f, prominence=prom * f.max())
    return len(pk)


def test_lokale_maxima_und_prominenz_gegen_find_peaks_auf_zufaelligen_feldern():
    rng = np.random.default_rng(np.random.SeedSequence([9700, 3]))
    for _ in range(60):
        f = np.round(rng.random(rng.integers(5, 400)), 1)                  # grobe Rundung: viele Hochebenen und Gleichstände
        pk, eig = scipy_signal.find_peaks(f)
        assert list(V.lokale_maxima(f)) == list(pk)
        prom = scipy_signal.peak_prominences(f, pk)[0]
        mine = [V.prominenz(f, int(p)) for p in pk]
        assert np.allclose(mine, prom)
        fp = np.concatenate([[0.0], f, [0.0]])
        for p in (0.02, 0.1, 0.3):
            assert V.gipfel_zaehlen(f, p) == _scipy_gipfel(f, p)


@pytest.mark.parametrize("s", [0.0, 1.0, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0])
def test_gipfelzahl_der_echten_dichten_gegen_find_peaks(s):
    f = V.pdf(mischung(P.standorte_versetzt(s), [1 / 3] * 3), P.X_GITTER)
    for prom in C.PROM2:
        assert V.gipfel_zaehlen(f, prom) == _scipy_gipfel(f, prom), (s, prom)


# ---------------------------------------------------------------- Dichte des Baryzentrums aus den Quantilen


def test_dichte_des_baryzentrums_integriert_zu_eins_und_stimmt_beim_profil():
    r = E.reihe3(0.1)
    pos, dichte = r["pos_bary"], r["dichte_bary"]
    masse = np.sum(0.5 * (dichte[1:] + dichte[:-1]) * np.diff(pos))      # Trapezregel über die Positionen
    assert masse == pytest.approx(1.0, abs=5e-3)      # Trapezregel auf den Stufen: im dünnen Teil zwischen den Spitzen liegen die Stützstellen weit auseinander
    r0 = E.reihe2(2.0, 0.02)                                              # Verschiebungsfamilie: Baryzentrum = Profil, also gleiche Dichte
    assert np.abs(r0["dichte_bary"] - V.pdf(P.profil(), r0["pos_bary"])).max() < 1e-6
