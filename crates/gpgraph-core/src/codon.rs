//! Codon-aware neighbor detection over amino-acid sequences.
//!
//! For each pair of amino acids (a, b) the distance is the minimum number of
//! nucleotide substitutions needed to mutate any codon coding for a into a
//! codon coding for b, under the standard genetic code. For a pair of
//! sequences the distance is the per-position sum. Pairs with distance
//! <= cutoff are neighbors.

use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::OnceLock;

use crate::hamming::edges_to_numpy;

const CODONS: &[(&[u8; 3], u8)] = &[
    (b"TTT", b'F'), (b"TTC", b'F'), (b"TTA", b'L'), (b"TTG", b'L'),
    (b"TCT", b'S'), (b"TCC", b'S'), (b"TCA", b'S'), (b"TCG", b'S'),
    (b"TAT", b'Y'), (b"TAC", b'Y'), (b"TAA", b'*'), (b"TAG", b'*'),
    (b"TGT", b'C'), (b"TGC", b'C'), (b"TGA", b'*'), (b"TGG", b'W'),
    (b"CTT", b'L'), (b"CTC", b'L'), (b"CTA", b'L'), (b"CTG", b'L'),
    (b"CCT", b'P'), (b"CCC", b'P'), (b"CCA", b'P'), (b"CCG", b'P'),
    (b"CAT", b'H'), (b"CAC", b'H'), (b"CAA", b'Q'), (b"CAG", b'Q'),
    (b"CGT", b'R'), (b"CGC", b'R'), (b"CGA", b'R'), (b"CGG", b'R'),
    (b"ATT", b'I'), (b"ATC", b'I'), (b"ATA", b'I'), (b"ATG", b'M'),
    (b"ACT", b'T'), (b"ACC", b'T'), (b"ACA", b'T'), (b"ACG", b'T'),
    (b"AAT", b'N'), (b"AAC", b'N'), (b"AAA", b'K'), (b"AAG", b'K'),
    (b"AGT", b'S'), (b"AGC", b'S'), (b"AGA", b'R'), (b"AGG", b'R'),
    (b"GTT", b'V'), (b"GTC", b'V'), (b"GTA", b'V'), (b"GTG", b'V'),
    (b"GCT", b'A'), (b"GCC", b'A'), (b"GCA", b'A'), (b"GCG", b'A'),
    (b"GAT", b'D'), (b"GAC", b'D'), (b"GAA", b'E'), (b"GAG", b'E'),
    (b"GGT", b'G'), (b"GGC", b'G'), (b"GGA", b'G'), (b"GGG", b'G'),
];

/// 256 x 256 ASCII lookup table of amino-acid min-bp distances.
/// Entries for bytes outside the recognized alphabet are `u8::MAX`.
fn distance_table() -> &'static [[u8; 256]; 256] {
    static TABLE: OnceLock<[[u8; 256]; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [[u8::MAX; 256]; 256];
        // For each pair of codons, compute number of differing bases.
        // Then for each aa pair, take the minimum across codon pairs.
        for (i, (codon_a, aa_a)) in CODONS.iter().enumerate() {
            // aa vs itself is 0.
            t[*aa_a as usize][*aa_a as usize] = 0;
            for (codon_b, aa_b) in CODONS.iter().skip(i) {
                let mut d: u8 = 0;
                for k in 0..3 {
                    if codon_a[k] != codon_b[k] {
                        d += 1;
                    }
                }
                let cur = t[*aa_a as usize][*aa_b as usize];
                if d < cur {
                    t[*aa_a as usize][*aa_b as usize] = d;
                    t[*aa_b as usize][*aa_a as usize] = d;
                }
            }
        }
        t
    })
}

pub fn codon_neighbors<'py>(
    py: Python<'py>,
    genotypes: Vec<String>,
    cutoff: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let n = genotypes.len();
    if n == 0 {
        return edges_to_numpy(py, Vec::new());
    }
    let l = genotypes[0].len();

    let mut flat: Vec<u8> = Vec::with_capacity(n * l);
    for (i, g) in genotypes.iter().enumerate() {
        if g.len() != l {
            return Err(PyValueError::new_err(format!(
                "genotype at index {} has length {}, expected {}",
                i,
                g.len(),
                l
            )));
        }
        flat.extend_from_slice(g.as_bytes());
    }

    // Validate every byte is a recognized amino acid (cheaper to front-load
    // than to check in the hot loop).
    let table = distance_table();
    for (idx, &byte) in flat.iter().enumerate() {
        // A byte is valid iff its diagonal (byte, byte) is finite.
        if table[byte as usize][byte as usize] == u8::MAX {
            return Err(PyValueError::new_err(format!(
                "unrecognized amino-acid byte {:?} at flat index {}",
                byte as char, idx
            )));
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
                        d += table[row_i[k] as usize][row_j[k] as usize] as usize;
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
