"""Die zwei Mitten in Brüchen: Handrechnungen der Handbeispiele (Erwartungswerte stehen als Rechnung im Test, nicht aus dem Code abgeschrieben)."""

from fractions import Fraction as F

import pytest

import pvs_diskret as D


def test_handbeispiel_1_verschiebung_und_mischung_bei_t_halb(profil_a, profil_b):
    halb = [F(1, 2), F(1, 2)]
    assert D.w2_quadrat(profil_a, profil_b) == 2 ** 2                      # reine Verschiebung um 2 Stunden
    bary = D.quantilmittel([profil_a, profil_b], halb)
    assert bary == [(9, F(2, 5)), (19, F(3, 5))]                            # jede Masse legt 1 Stunde zurück
    mix = D.mischung([profil_a, profil_b], halb)
    assert mix == [(8, F(1, 5)), (10, F(1, 5)), (18, F(3, 10)), (20, F(3, 10))]
    assert D.w2_quadrat(profil_a, bary) == (F(1, 2) * 2) ** 2               # liegt auf der Geodäte: (t · 2)²
    assert D.w2_quadrat(profil_b, bary) == 1
    assert D.w2_quadrat(profil_a, mix) == F(1, 2) * 2 ** 2                  # die Hälfte der Masse (1/5 + 3/10) muss 2 Stunden weit
    assert D.w2_quadrat(profil_a, mix) / D.w2_quadrat(profil_a, bary) == 2   # Verhältnis der W2 ist √2


@pytest.mark.parametrize("t", [F(1, 4), F(1, 2), F(3, 4)])
def test_handbeispiel_1_und_7_zwei_geodaeten(profil_a, profil_b, t):
    mix = D.mischung([profil_a, profil_b], [1 - t, t])
    bary = D.quantilmittel([profil_a, profil_b], [1 - t, t])
    assert D.w2_quadrat(profil_a, bary) == (2 * t) ** 2          # W2² = (t · 2)²
    assert D.w2_quadrat(profil_a, mix) == 4 * t                  # die Masse t muss 2 Stunden weit: t · 2²
    assert D.tv(profil_a, profil_b) == 1                         # getrennte Träger
    assert D.tv(profil_a, mix) == t                              # TV(A, Mischung_t) = t · TV(A, B)
    assert D.tv(profil_a, bary) == 1                             # für die Totalvariation ist die Verschiebung „ganz woanders“


def test_handbeispiel_1_verhaeltnis_bei_viertel(profil_a, profil_b):
    t = F(1, 4)
    assert D.w2_quadrat(profil_a, D.mischung([profil_a, profil_b], [1 - t, t])) / D.w2_quadrat(profil_a, D.quantilmittel([profil_a, profil_b], [1 - t, t])) == 4   # 1/t


def test_handbeispiel_2_drei_standorte_bei_minus_s_null_plus_s(profil_a):
    s = 3
    standorte = [D.verschoben(profil_a, d) for d in (-s, 0, s)]
    drittel = [F(1, 3)] * 3
    mix = D.mischung(standorte, drittel)
    assert len(mix) == 6                                          # sechs Orte
    assert D.quantilmittel(standorte, drittel) == profil_a        # das Baryzentrum von Verschiebungen mit Mittel 0 ist das Profil selbst
    mu_a, var_a = D.mittel_varianz(profil_a)
    assert (mu_a, var_a) == (14, 24)                              # 0,4·36 + 0,6·16 = 24
    assert D.mittel_varianz(mix) == (14, 24 + F(2 * 3 ** 2, 3))   # Gesetz der totalen Varianz: 24 + Var(d) = 24 + 6
    assert D.mittel_varianz(D.quantilmittel(standorte, drittel))[1] == 24


def test_handbeispiel_3_atome_hoechste_einzelmasse_ist_dreimal_so_gross(profil_a):
    standorte = [D.verschoben(profil_a, d) for d in (-3, 0, 3)]
    mix = D.mischung(standorte, [F(1, 3)] * 3)
    assert max(m for _, m in profil_a) / max(m for _, m in mix) == 3      # Zahl der Standorte


def test_handbeispiel_4_unterschiedliche_morgenanteile():
    shares = [F(3, 10), F(4, 10), F(5, 10)]
    standorte = [D.tagesprofil_atome(w) for w in shares]
    drittel = [F(1, 3)] * 3
    bary = D.quantilmittel(standorte, drittel)
    # Intervall [0,3; 0,4): Quantile 18 (Standort 1), 8, 8 → (18 + 8 + 8)/3; [0,4; 0,5): 18, 18, 8 → (18 + 18 + 8)/3
    assert bary == [(8, F(3, 10)), (F(34, 3), F(1, 10)), (F(44, 3), F(1, 10)), (18, F(1, 2))]
    assert D.masse_zwischen(bary, 8, 18) == F(1, 5)               # ein Fünftel zwischen den Spitzen
    assert all(x in (8, 18) for s in standorte for x, _ in s)    # kein Standort hat dort Nachfrage
    mix = D.mischung(standorte, drittel)
    assert mix == [(8, F(2, 5)), (18, F(3, 5))]
    assert D.masse_zwischen(mix, 8, 18) == 0
    assert D.zielwert(bary, standorte, drittel) == F(40, 9)
    assert D.zielwert(mix, standorte, drittel) == F(20, 3)
    varianzen = [D.mittel_varianz(s)[1] for s in standorte]
    assert varianzen == [21, 24, 25]                              # 0,3·0,7·100 = 21; 0,4·0,6·100 = 24; 0,5·0,5·100 = 25
    assert D.mittel_varianz(bary)[1] <= sum(varianzen) / 3 == F(70, 3) <= D.mittel_varianz(mix)[1] == 24
    assert D.mittel_varianz(bary)[0] == D.mittel_varianz(mix)[0] == 14        # dasselbe Mittel


def test_handbeispiel_5_skalenabhaengigkeit_des_quantilmittels():
    eins, hundert = D.normiert([(1, 1)]), D.normiert([(100, 1)])
    assert D.quantilmittel([eins, hundert], [F(1, 2)] * 2) == [(F(101, 2), 1)]   # Kosten x²: der Mittelpunkt 50,5


def test_handbeispiel_8_gleichstand_kosten_betrag_und_quadrat():
    b = D.gleichstand_beispiel()
    assert b["p"] == [(0, F(1, 2)), (1, F(1, 2))] and b["q"] == [(2, F(1, 2)), (3, F(1, 2))]
    sortiert, gekreuzt = b["zuordnungen"][0], b["zuordnungen"][1]
    assert sortiert == [(0, 2), (1, 3)] and gekreuzt == [(0, 3), (1, 2)]
    assert b["kosten1"] == [2, 2]                    # (2 + 2)/2 und (3 + 1)/2: beide optimal
    assert b["kosten2"] == [4, 5]                    # (4 + 4)/2 und (9 + 1)/2: nur die sortierte ist optimal
    assert D.zwischenverteilung(sortiert, F(1, 2)) == [(1, F(1, 2)), (2, F(1, 2))]
    assert D.zwischenverteilung(gekreuzt, F(1, 2)) == [(F(3, 2), 1)]
    assert len(b["optimal1"]) == 2 and len(b["optimal2"]) == 1
    assert b["zwischen1"][0] != b["zwischen1"][1]
    assert b["zwischen2"] == [[(1, F(1, 2)), (2, F(1, 2))]] == [b["quantilmittel"]]   # die eindeutige Zuordnung gibt das Quantilmittel


def test_masse_null_wird_weggelassen_und_orte_werden_zusammengefasst():
    assert D.normiert([(1, F(1, 2)), (1, F(1, 2)), (5, 0)]) == [(1, 1)]


def test_bruchzahlen_bleiben_exakt(profil_a):
    assert isinstance(D.w2_quadrat(profil_a, D.verschoben(profil_a, 2)), F)
