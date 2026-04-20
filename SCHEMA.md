# gpgraph-v2 schema

The contract downstream consumers rely on. Breaking changes to this document
bump the major version.

## Node keys

Each node in a `GenotypePhenotypeGraph` is keyed by an `int` index into the
underlying `gpmap.GenotypePhenotypeMap.data` pandas DataFrame. Node `i`
corresponds to `gpm.data.iloc[i]`.

## Node attributes

Every node carries the full row of the gpm as keyword attributes. At minimum:

- `genotypes` (`str`) - the genotype string for this row.
- `phenotypes` (`float`) - the phenotype value (equivalent to v1's `fitnesses`).
- `binary` (`str`) - the genotype re-expressed in the packed encoding as a
  string of "0"/"1" characters.
- `n_mutations` (`int`) - Hamming weight from wildtype (per gpmap-v2).

Any additional columns present in `gpm.data` are copied onto the node dict
under their column name.

## Edge attributes

Edges are directed. For every undirected neighbor pair `(i, j)` the graph
carries both directed edges `(i, j)` and `(j, i)`.

After `add_model` has run, edges carry:

- `prob` (`float`) - the fixation model's evaluation at the source and target
  fitness. For the `None` model it is `1.0`. For `sswm`, `moran`, `mcclandish`
  it is in `[0, 1]`. For `ratio` it is unbounded.

Edges populated only by `add_gpm` (no model run yet) carry no attributes
beyond the `(u, v)` pair.

## Positional layout (`layout.flattened`)

Returns `dict[int, tuple[float, float]]`. Layer index is `n_mutations` from
wildtype. Nodes in the same layer are spaced by `scale` units and centered
on the origin. When `vertical=True`, layer grows downward (y = -level),
otherwise to the right (x = level).

## Neighbor kernel dispatch (`gpgraph.neighbors.get_neighbors`)

- user-supplied callable: pure-Python `O(N^2)` fallback.
- `neighbor_function="hamming"` with `binary_packed` supplied and biallelic,
  `cutoff <= 2`: Rust bit-flip fast path (`O(N * C(L, cutoff))`).
- `neighbor_function="hamming"` with `binary_packed` supplied, larger cutoff
  or non-binary alphabet: Rust rayon-parallel packed pairwise.
- `neighbor_function="hamming"` without `binary_packed`: Rust rayon-parallel
  string pairwise.
- `neighbor_function="codon"`: Rust rayon-parallel codon pairwise over the
  standard genetic code.

The return shape is always a NumPy `(E, 2) int64` array. Each undirected
neighbor pair appears twice: once as `(i, j)` and once as `(j, i)`. Rows
are sorted lexicographically so output is deterministic.
