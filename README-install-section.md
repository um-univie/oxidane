# Using Oxidane from source

Oxidane is a Rust library. Add it to another Rust project's `Cargo.toml`
(adjust the path to the checkout):

```toml
[dependencies]
oxidane = { path = "../oxidane" }
```

Enable the optional neural-network modules with `features = ["ml"]` on that
dependency. Molecular parsing, descriptors, featurization, and dataset splitting
do not require this feature.

From the Oxidane checkout, check the library and run its documentation examples:

```sh
cargo check --locked --all-features
cargo test --locked --doc
```

See [RELEASING.md](RELEASING.md) for source-release contents and registry
prerequisites. There is no installed `oxidane` command or prebuilt CLI binary.
