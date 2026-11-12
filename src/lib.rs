#![doc = include_str!("../README.md")]
/// Practical recipes, compiled and run as documentation tests.
#[doc = include_str!("../docs/COOKBOOK.md")]
pub mod cookbook {}

pub mod atom;
pub mod consts;
pub mod graph_algorithms;
pub mod io;
pub mod ml;
pub mod molecular_formula;
pub mod molecule;
pub mod prelude;
pub mod random;
#[cfg(test)]
mod tests;
pub mod vector;
