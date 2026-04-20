# Canonaut source snapshot

Upstream: https://github.com/um-univie/canonaut
Commit: `63f8b325ca187b6ad87b364deea3da69314057b2` (version 1.1.0).

The library source, build script, C benchmark helper, README, LICENSE.txt and
NOTICE are byte-for-byte copies from this commit. `UPSTREAM.sha256` records
their checksums. The manifest retains the library's dependencies and optional
C comparison support; workspace bindings, binaries, examples, benchmarks and
the large external test corpora are omitted. Tests needing those corpora must
be run in the upstream checkout. No algorithm changes are carried here.

This snapshot contains correctness fixes absent from crates.io's 1.0.0.
It keeps source checkouts and release archives independent of a sibling
checkout. The explicit registry requirement is `=1.1.0`; registry packaging
must wait until that version has been published and verified. Do not replace
it with 1.0.0 merely to make packaging succeed.
