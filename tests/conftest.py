"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction as F
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pvs_diskret as D  # noqa: E402


@pytest.fixture
def profil_a():
    """Handbeispiel 1: Masse 2/5 bei 8 Uhr, 3/5 bei 18 Uhr."""
    return D.normiert([(8, F(2, 5)), (18, F(3, 5))])


@pytest.fixture
def profil_b(profil_a):
    """Dasselbe Profil um 2 Stunden versetzt."""
    return D.verschoben(profil_a, 2)
