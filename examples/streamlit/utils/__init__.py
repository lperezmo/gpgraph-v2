"""Shared helpers for the gpgraph-v2 Streamlit showcase."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from gpmap import GenotypePhenotypeMap


def make_fuji_gpm(
    length: int, alphabet: Iterable[str] = ("A", "T"), seed: int = 0
) -> GenotypePhenotypeMap:
    """Build a tiny additive-with-noise Mount-Fuji-style gpm for demos."""
    from itertools import product

    alph = list(alphabet)
    genotypes = ["".join(g) for g in product(alph, repeat=length)]
    wildtype = alph[0] * length
    rng = np.random.default_rng(seed)
    per_site_effect = rng.normal(loc=1.0, scale=0.5, size=(length, len(alph)))
    per_site_effect[:, 0] = 0.0  # wildtype contributes nothing

    phenotypes = []
    for g in genotypes:
        total = 0.0
        for i, letter in enumerate(g):
            total += float(per_site_effect[i, alph.index(letter)])
        phenotypes.append(max(total + 1.0, 0.05))

    return GenotypePhenotypeMap(
        wildtype=wildtype,
        genotypes=genotypes,
        phenotypes=phenotypes,
        stdeviations=[0.05] * len(genotypes),
    )
