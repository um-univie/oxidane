//! Defaults for coordinate-based bond perception, not chemical reference data.

/// Legacy bond-search distance threshold in angstroms.
pub const BOND_SEARCH_THRESHOLD: f64 = 2.0;
/// Dimensionless multiplier of summed covalent radii when perceiving bonds.
/// A value of 1.2 allows distances up to 20% beyond the radius sum.
pub const BOND_TOLERANCE: f64 = 1.2;
