"""Type stubs for the compiled Rust extension `gpgraph._rust`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__version__: str

def hamming_neighbors_packed(
    binary_packed: NDArray[np.uint8],
    cutoff: int,
) -> NDArray[np.int64]: ...
def hamming_bitflip_neighbors(
    binary_packed: NDArray[np.uint8],
    cutoff: int,
) -> NDArray[np.int64]: ...
def hamming_neighbors_strings(
    genotypes: list[str],
    cutoff: int,
) -> NDArray[np.int64]: ...
def codon_neighbors(
    genotypes: list[str],
    cutoff: int,
) -> NDArray[np.int64]: ...
