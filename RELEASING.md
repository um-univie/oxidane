# Source and registry releases

Oxidane is a Rust library. Its release artifact is a source archive, containing
Cargo.toml, Cargo.lock and tests. There is no `oxidane` executable to package.

## Dependencies

All dependencies come from crates.io, including Canonaut 1.1.0, so
`cargo package --locked --all-features` works from a clean checkout. Review the
archive and its verification result before publishing. This repository's
workflow does not publish crates to crates.io.

## Tagged source releases

The release workflow tests the library on Linux, macOS and Windows. A `v*`
tag must match the package version. After those checks pass, it creates a
source archive from the tag and compiles the extracted archive with every
feature and target enabled. Only then does it create a **draft** GitHub release with the archive and its SHA-256
checksum. A maintainer reviews and publishes the draft separately.

Pull requests run the tests on Linux only, unless labelled `full-ci`; see
[docs/CHECKS.md](docs/CHECKS.md), which also covers running every check
locally with `scripts/ci-local.sh`.
