# Contributing to gpgraph-v2

## Development setup

```bash
git clone https://github.com/lperezmo/gpgraph-v2
cd gpgraph-v2
uv sync
uv run maturin develop --release
uv run pytest
```

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/). The commit
subject line controls the next release bump:

- `fix: ...` -> patch (0.1.0 -> 0.1.1)
- `feat: ...` -> minor (0.1.0 -> 0.2.0)
- `feat!: ...` or a `BREAKING CHANGE:` footer -> major (0.1.0 -> 1.0.0)

Other prefixes (`chore:`, `docs:`, `refactor:`, `test:`, `style:`, `perf:`,
`build:`, `ci:`) do not bump the version but are encouraged for log clarity.

## Quality gates

Run these before opening a PR:

```bash
uv run ruff check python/gpgraph tests
uv run ruff format --check python/gpgraph tests
uv run mypy python/gpgraph
uv run pytest --cov=gpgraph --cov-fail-under=80
```

CI enforces the same checks on Ubuntu, macOS, and Windows across Python 3.11,
3.12, and 3.13.

## Rust changes

Editing the Rust crate requires rebuilding the extension:

```bash
uv run maturin develop --release && uv run pytest
```

## Releases

Releases are automated. Merging to `main` with at least one `fix:` or `feat:`
commit triggers `python-semantic-release`, which bumps the version in
`pyproject.toml`, updates `CHANGELOG.md`, tags the commit, and kicks off the
wheel matrix + PyPI publish.
