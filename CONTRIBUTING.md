# Contributing to oxidane

Bug reports, questions and pull requests are welcome.

## Reporting problems

Open an issue at <https://github.com/um-univie/oxidane/issues>. For a wrong
chemical result, include the input (SMILES, SMARTS or a minimal SDF), what
oxidane returned, what you expected, and if possible what RDKit or another
toolkit returns. State the oxidane version or commit.

## Development setup

```sh
git clone https://github.com/um-univie/oxidane
cd oxidane
cargo test --locked --all-features
```

Python bindings live in `bindings/python` and are built with maturin:

```sh
cd bindings/python
uv run maturin develop
uv run pytest
```

## Pull requests

- Branch from `main`; keep a pull request to one change.
- Add a test for every bug fix and new feature. Chemistry fixes should include
  the molecule that exposed the problem.
- Run `cargo fmt`, `cargo clippy --all-targets --all-features` and
  `cargo test --locked --all-features` before pushing.
- Code derived from other projects must be license-compatible with MIT and
  recorded in `THIRD_PARTY.md`.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
