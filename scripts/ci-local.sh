#!/usr/bin/env bash
# Runs the checks of the GitHub workflows on this machine, so pull requests do
# not spend Actions minutes on them. See docs/CHECKS.md.
# shellcheck disable=SC2329  # the stage_* functions are called by name below
set -uo pipefail

usage() {
  cat <<'EOF'
usage: scripts/ci-local.sh [options] [stage...]

Stages (default: all):
  rust      cargo check of every target, cargo test, rustdoc with -D warnings
  python    the Python binding tests on Python 3.9 and 3.14
  wheels    abi3 wheels for Linux (glibc and musl, x86-64 and ARM64), macOS
            (Intel and Apple silicon) and Windows x64; the Linux x86-64 one is
            installed and run

What to check (default: this checkout as it is, uncommitted changes included):
  --pr N        pull request N of the repository, fetched from GitHub
  --ref REF     a branch, tag or commit of the repository, fetched from GitHub
  --repo O/R    the GitHub repository (default: $OXIDANE_CI_REPO, else origin)

  --report      post each stage's result as a commit status ("local/<stage>")
                on GitHub; needs --pr, --ref or a clean checkout
  -j N          parallel jobs for cargo (default: every core)
  -h, --help    this text

Private repositories work through the GitHub CLI's login (`gh auth login`).
Work files go to $OXIDANE_CI_DIR, by default /scr/<host>/$USER/oxidane-ci when
that exists, else ~/.cache/oxidane-ci; the build cache there is reused.
EOF
}

repo=${OXIDANE_CI_REPO:-}
pr="" ref="" report=0 jobs="" stages=()
while (($#)); do
  case $1 in
    --pr) pr=$2; shift ;;
    --ref) ref=$2; shift ;;
    --repo) repo=$2; shift ;;
    --report) report=1 ;;
    -j) jobs=$2; shift ;;
    -h|--help) usage; exit 0 ;;
    rust|python|wheels) stages+=("$1") ;;
    *) echo "ci-local: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
((${#stages[@]})) || stages=(rust python wheels)

host=$(hostname -s)
if [[ -z ${OXIDANE_CI_DIR:-} ]]; then
  if [[ -d /scr/$host/$USER ]]; then OXIDANE_CI_DIR=/scr/$host/$USER/oxidane-ci
  else OXIDANE_CI_DIR=${XDG_CACHE_HOME:-$HOME/.cache}/oxidane-ci; fi
fi
work=$OXIDANE_CI_DIR
mkdir -p "$work/logs" "$work/wheels"
exec 9>"$work/lock"
flock -n 9 || { echo "ci-local: another run holds $work/lock" >&2; exit 1; }

here=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
if [[ -z $repo ]]; then
  repo=$(git -C "$here" remote get-url origin | sed -E 's#^(https://github\.com/|git@github\.com:)##; s#\.git$##')
fi
github() { git -c credential.helper= -c credential.helper='!gh auth git-credential' "$@"; }

# Pick the tree to check: a fetched pull request or ref, or this checkout.
if [[ -n $pr || -n $ref ]]; then
  src=$work/src
  [[ -d $src/.git ]] || git init -q "$src"
  what=${pr:+pull/$pr/head}
  what=${what:-$ref}
  echo "ci-local: fetching $what from $repo"
  github -C "$src" fetch -q --depth 1 "https://github.com/$repo.git" "$what" || exit 1
  git -C "$src" checkout -q --force --detach FETCH_HEAD && git -C "$src" clean -qfdx || exit 1
  sha=$(git -C "$src" rev-parse HEAD)
else
  src=$here
  sha=$(git -C "$src" rev-parse HEAD)
  if ((report)) && [[ -n $(git -C "$src" status --porcelain --untracked-files=no) ]]; then
    echo "ci-local: --report needs a clean checkout, a --pr or a --ref" >&2; exit 2
  fi
fi
echo "ci-local: checking ${sha:0:12} in $src"

export CARGO_TARGET_DIR=$work/target CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0
[[ -n $jobs ]] && export CARGO_BUILD_JOBS=$jobs

status() {  # status <stage> <state> <description>
  ((report)) || return 0
  gh api -X POST "repos/$repo/statuses/$sha" -f state="$2" -f context="local/$1" \
    -f description="$3 on $host" > /dev/null || echo "ci-local: could not post the $1 status" >&2
}

maturin() { uv run --no-project --with 'maturin>=1.9,<2' --with ziglang maturin "$@"; }

stage_rust() {
  cargo check --locked --all-features --all-targets &&
  cargo test --locked --all-features &&
  RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
}

stage_python() {
  local version
  for version in 3.9 3.14; do
    echo "== Python $version"
    UV_PROJECT_ENVIRONMENT=$work/venv-$version uv sync --locked --project bindings/python \
      --extra notebook --python "$version" --reinstall-package oxidane &&
    UV_PROJECT_ENVIRONMENT=$work/venv-$version uv run --no-sync --project bindings/python \
      pytest -q -p no:cacheprovider bindings/python/tests || return 1
  done
}

stage_wheels() {
  local out=$work/wheels/$sha target compatibility
  rm -rf "${out:?}" && mkdir -p "$out"
  rustup target add x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu x86_64-unknown-linux-musl \
    aarch64-unknown-linux-musl x86_64-apple-darwin aarch64-apple-darwin x86_64-pc-windows-msvc \
    > /dev/null 2>&1
  for target in x86_64-unknown-linux-gnu:manylinux2014 aarch64-unknown-linux-gnu:manylinux2014 \
      x86_64-unknown-linux-musl:musllinux_1_2 aarch64-unknown-linux-musl:musllinux_1_2 \
      x86_64-apple-darwin: aarch64-apple-darwin:; do
    compatibility=${target#*:}
    echo "== ${target%%:*}"
    (cd bindings/python && maturin build --release --locked --zig --target "${target%%:*}" \
      ${compatibility:+--compatibility "$compatibility"} --out "$out") || return 1
  done
  # Windows needs clang-cl and lld-link; pixi fetches them when they are missing.
  echo "== x86_64-pc-windows-msvc"
  local windows=(maturin build --release --locked --target x86_64-pc-windows-msvc --out "$out")
  if command -v clang-cl > /dev/null && command -v lld-link > /dev/null; then
    (cd bindings/python && XWIN_CACHE_DIR=$work/xwin "${windows[@]}") || return 1
  elif command -v pixi > /dev/null; then
    (cd bindings/python && XWIN_CACHE_DIR=$work/xwin pixi exec -s clang -s lld -s llvm-tools -- \
      uv run --no-project --with 'maturin>=1.9,<2' "${windows[@]}") || return 1
  else
    echo "ci-local: no clang-cl/lld-link and no pixi; skipping the Windows wheel"
  fi
  echo "== smoke test"
  local venv=$work/venv-wheel
  rm -rf "${venv:?}" && uv venv -q --python 3.12 "$venv" &&
  uv pip install -q --python "$venv/bin/python" "$out"/*manylinux*x86_64.whl &&
  printf 'id,smiles\n1,Oc1ccccn1\n' > "$work/in.csv" &&
  "$venv/bin/oxidane-normalize" "$work/in.csv" "$work/out.csv" -s smiles &&
  grep -q 'O=c1\[nH\]cccc1' "$work/out.csv" &&
  ls "$out"
}

failed=0
summary=()
for stage in "${stages[@]}"; do
  log=$work/logs/${sha:0:12}-$stage.log
  status "$stage" pending "running"
  echo "ci-local: $stage (log: $log)"
  start=$SECONDS
  (cd "$src" && "stage_$stage") > "$log" 2>&1
  result=$?
  took=$((SECONDS - start))
  if ((result == 0)); then
    summary+=("$stage: passed in ${took}s")
    status "$stage" success "passed in ${took}s"
  else
    failed=1
    summary+=("$stage: FAILED after ${took}s, see $log")
    status "$stage" failure "failed after ${took}s"
    tail -n 30 "$log"
  fi
done
printf 'ci-local: %s\n' "${summary[@]}"
exit $failed
