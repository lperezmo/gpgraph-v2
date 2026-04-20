//! Pairwise Hamming neighbor detection.
//!
//! Three entry points:
//! * `hamming_neighbors_packed`  - rayon-parallel pairwise over an (N, L) uint8 matrix.
//! * `hamming_bitflip_neighbors` - O(N * C(L, k)) fast path for biallelic cutoff 1 or 2.
//! * `hamming_neighbors_strings` - rayon-parallel pairwise over arbitrary-alphabet strings.
//!
//! All return an (E, 2) int64 array of directed edges. Each undirected pair (i, j)
//! appears twice: (i, j) and (j, i). Ordered by (i, j) lexicographically within
//! each row and then by `i`.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

pub(crate) fn edges_to_numpy<'py>(
    py: Python<'py>,
    mut edges: Vec<(i64, i64)>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    // Sort once so output is deterministic regardless of thread interleaving.
    edges.par_sort_unstable();
    let n = edges.len();
    let mut flat = Vec::with_capacity(n * 2);
    for (a, b) in edges {
        flat.push(a);
        flat.push(b);
    }
    let arr = ndarray::Array2::from_shape_vec((n, 2), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

pub fn hamming_neighbors_packed<'py>(
    py: Python<'py>,
    binary_packed: PyReadonlyArray2<'py, u8>,
    cutoff: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let view = binary_packed.as_array();
    let shape = view.shape();
    let n = shape[0];
    let l = shape[1];

    // Contiguous row-major copy so the hot loop does pointer arithmetic.
    let mut flat = vec![0u8; n * l];
    for (i, row) in view.outer_iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            flat[i * l + j] = *v;
        }
    }

    let edges: Vec<(i64, i64)> = py.detach(|| {
        (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let row_i = &flat[i * l..(i + 1) * l];
                let mut out: Vec<(i64, i64)> = Vec::new();
                for j in (i + 1)..n {
                    let row_j = &flat[j * l..(j + 1) * l];
                    let mut d: usize = 0;
                    for k in 0..l {
                        d += (row_i[k] != row_j[k]) as usize;
                        if d > cutoff {
                            break;
                        }
                    }
                    if d <= cutoff {
                        out.push((i as i64, j as i64));
                        out.push((j as i64, i as i64));
                    }
                }
                out.into_iter()
            })
            .collect()
    });

    edges_to_numpy(py, edges)
}

pub fn hamming_bitflip_neighbors<'py>(
    py: Python<'py>,
    binary_packed: PyReadonlyArray2<'py, u8>,
    cutoff: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    if cutoff == 0 {
        return edges_to_numpy(py, Vec::new());
    }
    if cutoff > 2 {
        return Err(PyValueError::new_err(
            "hamming_bitflip_neighbors supports cutoff in {1, 2}; dispatch elsewhere for larger.",
        ));
    }

    let view = binary_packed.as_array();
    let shape = view.shape();
    let n = shape[0];
    let l = shape[1];

    // Build row -> index map. Rows are Vec<u8> byte keys.
    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(n);
    for row in view.outer_iter() {
        let mut r = Vec::with_capacity(l);
        for v in row.iter() {
            if *v > 1 {
                return Err(PyValueError::new_err(
                    "hamming_bitflip_neighbors requires biallelic data (0/1 entries).",
                ));
            }
            r.push(*v);
        }
        rows.push(r);
    }

    // Index: FxHashMap<row, idx>. Built serially; O(N * L).
    let mut index: FxHashMap<Vec<u8>, i64> = FxHashMap::default();
    index.reserve(n);
    for (i, r) in rows.iter().enumerate() {
        index.insert(r.clone(), i as i64);
    }

    let edges: Vec<(i64, i64)> = py.detach(|| {
        (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let mut out: Vec<(i64, i64)> = Vec::new();
                let base = rows[i].clone();
                let mut buf = base.clone();

                // Distance-1 neighbors: flip each bit once, look up.
                for a in 0..l {
                    buf[a] ^= 1;
                    if let Some(&j) = index.get(&buf) {
                        if j > i as i64 {
                            out.push((i as i64, j));
                            out.push((j, i as i64));
                        }
                    }
                    buf[a] ^= 1;
                }

                // Distance-2 neighbors (only if cutoff >= 2): flip two bits.
                if cutoff >= 2 {
                    for a in 0..l {
                        buf[a] ^= 1;
                        for b in (a + 1)..l {
                            buf[b] ^= 1;
                            if let Some(&j) = index.get(&buf) {
                                if j > i as i64 {
                                    out.push((i as i64, j));
                                    out.push((j, i as i64));
                                }
                            }
                            buf[b] ^= 1;
                        }
                        buf[a] ^= 1;
                    }
                }

                out.into_iter()
            })
            .collect()
    });

    edges_to_numpy(py, edges)
}

pub fn hamming_neighbors_strings<'py>(
    py: Python<'py>,
    genotypes: Vec<String>,
    cutoff: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let n = genotypes.len();
    if n == 0 {
        return edges_to_numpy(py, Vec::new());
    }
    let l = genotypes[0].len();
    for (i, g) in genotypes.iter().enumerate() {
        if g.len() != l {
            return Err(PyValueError::new_err(format!(
                "genotype at index {} has length {}, expected {}",
                i,
                g.len(),
                l
            )));
        }
    }

    // Flatten bytes for cache-friendly access.
    let mut flat: Vec<u8> = Vec::with_capacity(n * l);
    for g in &genotypes {
        flat.extend_from_slice(g.as_bytes());
    }

    let edges: Vec<(i64, i64)> = py.detach(|| {
        (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let row_i = &flat[i * l..(i + 1) * l];
                let mut out: Vec<(i64, i64)> = Vec::new();
                for j in (i + 1)..n {
                    let row_j = &flat[j * l..(j + 1) * l];
                    let mut d: usize = 0;
                    for k in 0..l {
                        d += (row_i[k] != row_j[k]) as usize;
                        if d > cutoff {
                            break;
                        }
                    }
                    if d <= cutoff {
                        out.push((i as i64, j as i64));
                        out.push((j as i64, i as i64));
                    }
                }
                out.into_iter()
            })
            .collect()
    });

    edges_to_numpy(py, edges)
}
