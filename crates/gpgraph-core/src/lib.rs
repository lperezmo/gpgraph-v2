//! gpgraph-core: Rust hot-path primitives for gpgraph-v2.
//!
//! Exposed as `gpgraph._rust` inside the Python package.
//!
//! Scope (locked with Luis 2026-04-20): pairwise neighbor detection only.
//! Fixation models stay in vectorized numpy on the Python side.

use pyo3::prelude::*;

mod codon;
mod hamming;

/// Pairwise Hamming neighbors over a packed uint8 genotype matrix.
///
/// Parameters
/// ----------
/// binary_packed : np.ndarray[uint8] shape (N, L)
///     Per-row genotype encoded as L bytes (0/1 for biallelic, 0..k for k-ary).
/// cutoff : int
///     Max Hamming distance. Pairs with distance <= cutoff are neighbors.
///
/// Returns
/// -------
/// np.ndarray[int64] shape (E, 2)
///     Directed edge list. Each undirected neighbor pair (i, j) appears twice:
///     once as (i, j) and once as (j, i).
#[pyfunction]
fn hamming_neighbors_packed<'py>(
    py: Python<'py>,
    binary_packed: numpy::PyReadonlyArray2<'py, u8>,
    cutoff: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<i64>>> {
    hamming::hamming_neighbors_packed(py, binary_packed, cutoff)
}

/// Bit-flip fast path for cutoff 1 or 2 on biallelic packed data.
///
/// For biallelic alphabets the neighbors of a genotype at cutoff k are the
/// k-bit flips of its packed representation. Complexity drops from O(N^2 * L)
/// to O(N * C(L, k)).
///
/// Parameters
/// ----------
/// binary_packed : np.ndarray[uint8] shape (N, L)
///     All entries must be 0 or 1.
/// cutoff : int
///     Must be 1 or 2.
#[pyfunction]
fn hamming_bitflip_neighbors<'py>(
    py: Python<'py>,
    binary_packed: numpy::PyReadonlyArray2<'py, u8>,
    cutoff: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<i64>>> {
    hamming::hamming_bitflip_neighbors(py, binary_packed, cutoff)
}

/// Pairwise Hamming neighbors over arbitrary-alphabet string genotypes.
#[pyfunction]
fn hamming_neighbors_strings<'py>(
    py: Python<'py>,
    genotypes: Vec<String>,
    cutoff: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<i64>>> {
    hamming::hamming_neighbors_strings(py, genotypes, cutoff)
}

/// Pairwise codon neighbors over protein sequences.
///
/// Uses the amino-acid minimum-base-pair distance induced by the standard
/// genetic code. Two sequences are neighbors if the sum of per-position
/// min-bp-distances is <= cutoff.
#[pyfunction]
fn codon_neighbors<'py>(
    py: Python<'py>,
    genotypes: Vec<String>,
    cutoff: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<i64>>> {
    codon::codon_neighbors(py, genotypes, cutoff)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hamming_neighbors_packed, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_bitflip_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_neighbors_strings, m)?)?;
    m.add_function(wrap_pyfunction!(codon_neighbors, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
