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

## Tagged source releases

The release workflow tests the library on Linux, macOS and Windows. A `v*`
tag must match the package version. After those checks pass, it creates a
source archive from the tag, verifies the vendored dependency checksums, and
compiles the extracted archive with every feature and target enabled. Only
then does it create a **draft** GitHub release with the archive and its SHA-256
checksum. A maintainer reviews and publishes the draft separately.

Pull requests run the same platform tests without creating a release. None
of these jobs expect a binary or cross-compile Apple/Windows targets on Linux.
