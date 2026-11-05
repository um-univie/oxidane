# Checks

GitHub Actions minutes are limited for private repositories, and macOS and
Windows runners count ten and two times. So the workflows are split by cost:

| when | runs on GitHub |
|---|---|
| every pull request and push to `main` | Linux only: Rust check and tests, Python tests on 3.9 and 3.14 |
| changes to Markdown, `docs/`, `assets/` or `LICENSE` only | nothing |
| pull request labelled `full-ci`, a `v*` tag, or "Run workflow" on the Actions tab | everything: tests on Linux, macOS and Windows, and the wheels for all eight platforms |

A newer push to a pull request cancels the run still going for the older one.
Add the `full-ci` label to a pull request before merging anything that could
behave differently on macOS or Windows, and before a release.

## Running the checks locally

`scripts/ci-local.sh` runs the same checks on your own machine, at no cost:

```sh
scripts/ci-local.sh                        # this checkout, uncommitted changes included
scripts/ci-local.sh rust                   # one stage: rust, python or wheels
scripts/ci-local.sh --pr 12 --report       # a pull request, result posted to GitHub
scripts/ci-local.sh --ref main --repo um-univie/oxidane-dev
```

- **rust:** `cargo check` of every target, `cargo test`, and rustdoc with
  warnings as errors, all with every feature.
- **python:** the binding tests on Python 3.9 and 3.14.
- **wheels:** cross-builds the abi3 wheels for Linux (glibc and musl, x86-64
  and ARM64), macOS (Intel and Apple silicon) and Windows x64, then installs and
  runs the Linux x86-64 one. The other wheels are only built, because they
  cannot run here. Windows ARM64 is only built by the GitHub workflow: the
  `psm` crate's ARM64 assembly needs Microsoft's assembler.

`--pr` and `--ref` fetch from GitHub with the GitHub CLI's login, so private
repositories work. Set `OXIDANE_CI_REPO=um-univie/oxidane-dev` once instead of
passing `--repo` every time; without either, the repository of the `origin`
remote is used. `--report` posts each stage as a commit status named
`local/rust`, `local/python` or `local/wheels`, which the pull request shows
next to the GitHub checks.

Needed: Rust through rustup, [uv](https://docs.astral.sh/uv/), the GitHub CLI
for `--pr`, `--ref` and `--report`, and for the Windows wheel either `clang-cl`
and `lld-link` or [pixi](https://pixi.sh), which fetches them. zig comes
through uv. Work files and the reused build cache go to `$OXIDANE_CI_DIR`, by
default `/scr/<host>/$USER/oxidane-ci` when that directory exists and
`~/.cache/oxidane-ci` otherwise. One run at a time per directory.
