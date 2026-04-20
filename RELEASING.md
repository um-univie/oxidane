# Source and registry releases

Oxidane is a Rust library. Its release artifact is a source archive, containing
Cargo.toml, Cargo.lock, tests and the corrected Canonaut dependency. There is
no `oxidane` executable to package.

## Dependency status

The source dependency is pinned to Canonaut **1.1.0**, vendored from the exact
upstream commit recorded in `vendor/canonaut-1.1.0/UPSTREAM.md`. A checkout or
source archive builds without a sibling `../canonaut` directory. The library
source checksums and upstream licence/notice are included.

At the 2026-10-18 review, crates.io listed only Canonaut 1.0.0. Registry
packaging therefore remains blocked until the corrected 1.1.0 is published.
Do not lower the requirement, remove the path without checking its replacement,
or use `--no-verify` to declare a registry release ready. Cargo strips path
dependencies when packaging; consumers would receive the registry version.

Once the required dependency is available, compare its library source against
`UPSTREAM.sha256`, then run `cargo package --locked --all-features` from a clean
checkout. Review the archive and its verification result before publishing.
This repository's workflow does not publish crates to crates.io.
