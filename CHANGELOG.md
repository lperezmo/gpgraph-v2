# CHANGELOG


## v1.1.2 (2026-08-30)

### Bug Fixes

- **security**: Declare least-privilege workflow permissions
  ([`c949017`](https://github.com/lperezmo/gpgraph-v2/commit/c949017f8887ce3bbcf1d8c4e321ef59a7019e59))


## v1.1.1 (2026-08-30)

### Bug Fixes

- **deps**: Patch crossbeam-epoch pointer formatting unsoundness
  ([`76f62ee`](https://github.com/lperezmo/gpgraph-v2/commit/76f62ee4ba5ddcfcae1663f84485c71d2e26e229))

- **deps**: Refresh vulnerable Python dependency locks
  ([`8acf8f6`](https://github.com/lperezmo/gpgraph-v2/commit/8acf8f6b5bfc9b914b0599c4d2c6879ac6e65771))

### Chores

- Bump vulnerable deps to resolve Dependabot alerts
  ([`1990be3`](https://github.com/lperezmo/gpgraph-v2/commit/1990be348fc94e075088b293a70af0f9e44040d7))

pip (uv.lock): - GitPython 3.1.46 -> 3.1.50 - tornado 6.5.5 -> 6.5.7 - urllib3 2.6.3 -> 2.7.0 - idna
  3.11 -> 3.18

rust (Cargo.toml + Cargo.lock): - pyo3 0.28 -> 0.29 - numpy 0.28 -> 0.29 (kept in lockstep with
  pyo3)

Rust crate builds clean and the abi3 extension imports; no source changes required. Python test
  suite passes.


## v1.1.0 (2026-05-30)

### Chores

- Add docs badge to README ([#2](https://github.com/lperezmo/gpgraph-v2/pull/2),
  [`75d7f05`](https://github.com/lperezmo/gpgraph-v2/commit/75d7f05816f54a74d002b6028e29d97e546180b3))

Link to the new GitHub Pages docs site at lperezmo.github.io/gpgraph-v2 via the docs workflow status
  badge, between the CI and PyPI badges.

- Add Zensical docs site ([#1](https://github.com/lperezmo/gpgraph-v2/pull/1),
  [`e7a7bcb`](https://github.com/lperezmo/gpgraph-v2/commit/e7a7bcba65eb543b7849f5a0386f5536f3f67b03))

Stand up GitHub Pages documentation for gpgraph-v2 using Zensical (modern theme, orange accents
  matching zensical.org).

Site contents - Quickstart and installation pages - Concepts: graph model, neighbors and dispatch
  policy, fixation models - Guides: forward paths and flux, plotting with gpgraph.pyplot -
  Per-module API reference: graph, fixation, neighbors, paths, pyplot, exceptions, changelog

Configuration - zensical.toml with modern variant, light and dark palette toggle, navigation tabs
  and sections, content edit and view actions - docs/stylesheets/extra.css for the orange accent
  palette - /site already covered by .gitignore (mkdocs section)

Deployment - .github/workflows/docs.yml builds with pip-installed zensical and publishes to GitHub
  Pages via actions/deploy-pages on every push to main that touches docs/**, zensical.toml, or the
  workflow file

Pages will need to be enabled with build_type=workflow after this merges; the first run after
  enablement may need a workflow_dispatch retrigger because of the configure-pages race window.

- Fix migration guide v1 constructor note and add v1 vs v2 benchmark tables
  ([`d5f24b0`](https://github.com/lperezmo/gpgraph-v2/commit/d5f24b0ab8ca902159663a6e52d86411a0e8b77d))

- Replace broken static.streamlit.io badge with shields.io
  ([`8adb749`](https://github.com/lperezmo/gpgraph-v2/commit/8adb74972ac4dc415ced7632f5b8fba2169a6411))

- **streamlit**: Render complexity expressions as LaTeX on benchmarks page
  ([`bdfa66a`](https://github.com/lperezmo/gpgraph-v2/commit/bdfa66a461d0f4077690d1666766f0f80a420ad3))

Switch the bit-flip vs pairwise hamming complexity captions on the benchmarks page from code-styled
  text to KaTeX math.

### Documentation

- Add light/dark gallery images to docs and README
  ([#3](https://github.com/lperezmo/gpgraph-v2/pull/3),
  [`8b8b8cd`](https://github.com/lperezmo/gpgraph-v2/commit/8b8b8cdda21e77c3159b835d251a9fd92da13db7))

Add transparent-background genotype-phenotype graph figures that adapt to light and dark themes:
  docs pages pair them with #only-light / #only-dark, the README uses <picture> with
  prefers-color-scheme (absolute raw URLs so they also resolve on PyPI, where <picture> degrades to
  the light <img>).

Images: Hamming-layout hero graph with SSWM edge weights, the four fixation models compared, forward
  paths reduced to per-edge flux, and a codon-distance neighbor graph. Transparent backgrounds blend
  into any page background without a visible seam.

Docs-only change; no package code touched.

- Legible graph node labels and a neighbor-cutoff figure
  ([#4](https://github.com/lperezmo/gpgraph-v2/pull/4),
  [`1f0916d`](https://github.com/lperezmo/gpgraph-v2/commit/1f0916d7cbb4a7ff2d5de9fc36f988c525d76e20))

The graph figures (hero Hamming hypercube, forward-path flux, codon graph) colored each node by
  phenotype or amino acid but drew the genotype label and node outline in a single per-variant ink:
  black in the light PNG, white in the dark PNG (the codon figure even hardcoded white). That ink
  matched some node fills exactly, so labels vanished: dark text on the dark low-phenotype nodes in
  light mode, white text on the bright peak in dark mode.

Node label and outline color are now chosen per node from each node's own fill luminance, so they
  stay readable on whatever page background the docs or README use, in both PNG variants, with no
  manual overrides.

Also adds a cutoff-1-vs-cutoff-2 connectivity figure to the Neighbors concept page, which previously
  had no illustration: it shows the single-mutant hypercube skeleton next to the same map with
  double-mutant edges added.

- Render fixation model equations with MathJax
  ([`46e1618`](https://github.com/lperezmo/gpgraph-v2/commit/46e1618e27ca9a6aa149e6638d4c45ed54f527d0))

Wire up MathJax via extra_javascript and rewrite the SSWM, ratio, Moran, and McCandlish fixation
  kernels as proper display LaTeX so the formulas no longer show as monospace code fragments.

### Features

- **pyplot**: Legible per-node labels and contrast_ink helper
  ([#5](https://github.com/lperezmo/gpgraph-v2/pull/5),
  [`a4fc8e2`](https://github.com/lperezmo/gpgraph-v2/commit/a4fc8e2bcd1f2f44fae60dc74ff3e567374b714f))

draw_gpgraph colors nodes by a scalar through a colormap, but offered no built-in labels and no help
  with label color. A single fixed font color always collides with part of any colormap: black text
  vanishes on the dark (low) end, white text vanishes on the bright (high) end, and which end fails
  flips between light and dark display themes. Users had to compute contrasting colors by hand.

This adds a per-node-luminance contrast picker and wires it into drawing so node labels just work:

- contrast_ink(color): public helper returning a dark or light ink that contrasts with a given fill
  (perceived luminance, 0.6 threshold so the crossover sits in the orange band of
  magma/viridis/plasma). - resolve_node_fills(node_color, n, ...): resolve draw_gpgraph's node_color
  (scalars + cmap, a single color, or a color sequence) to per-node RGBA, mirroring networkx's
  rendering, so the right fill is fed to contrast_ink. - draw_gpgraph gains with_labels, labels,
  label_font_size, and label_ink. label_ink defaults to "auto" (per-node contrast); node_edgecolors
  now also accepts "auto" to contrast outlines the same way. Both override cleanly.

Both helpers are exported from gpgraph.pyplot. Adds unit tests for the ink thresholds, fill
  resolution, and the labeled/auto-outline draw paths, plus docs in the plotting guide and pyplot
  reference.


## v1.0.0 (2026-04-20)

### Features

- Scaffold gpgraph-v2 with uv, maturin, and PyO3
  ([`2f9d780`](https://github.com/lperezmo/gpgraph-v2/commit/2f9d78021d2c344eccce550c8843aeb38f9c725e))

Clean-break rewrite of harmslab/gpgraph. NetworkX-backed, Rust-accelerated directed graphs over
  gpmap-v2 genotype-phenotype maps.

Stack: uv + maturin + PyO3 + rayon. Python 3.11+. Reads the locked gpmap-v2 SCHEMA.md contract
  (binary_packed, n_mutations, phenotypes). Fixation models (sswm, ratio, moran, mcclandish) stay in
  vectorized numpy; neighbor detection and codon distance live in Rust.

Release automation via python-semantic-release on Conventional Commits. PyPI via OIDC trusted
  publisher. build_command is empty and allow_zero_version=false so the first release jumps straight
  to 1.0.0 rather than climbing from 0.0.1, and semantic-release does not try to invoke maturin
  inside a Docker image without Rust installed.

See CHANGELOG.md for the full list of ported and fixed behaviors.
