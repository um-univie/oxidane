//! Fixed-radius neighbour search over atom positions.
//!
//! Atoms are binned into cubic cells as wide as the search radius, so every
//! neighbour of an atom lies in its own cell or one of the 26 around it. For
//! molecules this beats a k-d tree: the radius is a few bond lengths, each
//! cell holds a handful of atoms, and building the grid is one pass.

use rustc_hash::FxHashMap;

use crate::vector::Vector3D;

pub(crate) struct NeighborGrid<'a> {
    positions: &'a [Vector3D],
    inverse_cell_size: f64,
    radius_squared: f64,
    /// `None` when the radius is not a usable cell size (zero, negative,
    /// infinite or NaN); queries then compare against every atom.
    cells: Option<FxHashMap<[i64; 3], Vec<u32>>>,
}

impl<'a> NeighborGrid<'a> {
    /// Indexes `positions` for queries of squared radius `radius_squared`.
    pub(crate) fn new(positions: &'a [Vector3D], radius_squared: f64) -> Self {
        let radius = radius_squared.sqrt();
        let usable = radius.is_finite() && radius > 0.0;
        let inverse_cell_size = if usable { radius.recip() } else { 0.0 };
        let cells = usable.then(|| {
            let mut cells: FxHashMap<[i64; 3], Vec<u32>> = FxHashMap::default();
            for (index, position) in positions.iter().enumerate() {
                cells
                    .entry(cell_of(position, inverse_cell_size))
                    .or_default()
                    .push(index as u32);
            }
            cells
        });
        Self {
            positions,
            inverse_cell_size,
            radius_squared,
            cells,
        }
    }

    /// Calls `visit(neighbor, squared_distance)` for every atom other than
    /// `index` closer than the radius, in no particular order.
    pub(crate) fn for_each_neighbor(&self, index: usize, mut visit: impl FnMut(usize, f64)) {
        let center = &self.positions[index];
        let mut check = |other: usize| {
            if other == index {
                return;
            }
            let p = &self.positions[other];
            let (dx, dy, dz) = (p.x - center.x, p.y - center.y, p.z - center.z);
            let distance_squared = dx * dx + dy * dy + dz * dz;
            if distance_squared < self.radius_squared {
                visit(other, distance_squared);
            }
        };
        let Some(cells) = &self.cells else {
            (0..self.positions.len()).for_each(check);
            return;
        };
        let [cx, cy, cz] = cell_of(center, self.inverse_cell_size);
        for x in cx - 1..=cx + 1 {
            for y in cy - 1..=cy + 1 {
                for z in cz - 1..=cz + 1 {
                    if let Some(members) = cells.get(&[x, y, z]) {
                        members.iter().for_each(|&other| check(other as usize));
                    }
                }
            }
        }
    }
}

/// The cell holding `position`. Coordinates that are NaN or too large for an
/// `i64` saturate, which only puts such atoms in a shared edge cell.
fn cell_of(position: &Vector3D, inverse_cell_size: f64) -> [i64; 3] {
    let bin = |coordinate: f64| (coordinate * inverse_cell_size).floor() as i64;
    [bin(position.x), bin(position.y), bin(position.z)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(positions: &[Vector3D], index: usize, radius_squared: f64) -> Vec<usize> {
        (0..positions.len())
            .filter(|&other| {
                let (a, b) = (&positions[index], &positions[other]);
                let d = (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2);
                other != index && d < radius_squared
            })
            .collect()
    }

    #[test]
    fn matches_brute_force_on_scattered_points() {
        // A deterministic scatter with negative coordinates, cell-boundary
        // points and a far-away cluster.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut positions: Vec<Vector3D> = (0..400)
            .map(|_| {
                Vector3D::new(
                    next() * 20.0 - 10.0,
                    next() * 20.0 - 10.0,
                    next() * 20.0 - 10.0,
                )
            })
            .collect();
        positions.push(Vector3D::new(2.0, 0.0, 0.0));
        positions.push(Vector3D::new(4.0, 0.0, 0.0));
        positions.push(Vector3D::new(1e6, 1e6, -1e6));
        positions.push(Vector3D::new(1e6 + 0.5, 1e6, -1e6));
        for radius_squared in [0.25, 4.0, 9.0, 0.0, f64::INFINITY] {
            let grid = NeighborGrid::new(&positions, radius_squared);
            for index in 0..positions.len() {
                let mut found = Vec::new();
                grid.for_each_neighbor(index, |other, _| found.push(other));
                found.sort_unstable();
                assert_eq!(
                    found,
                    brute_force(&positions, index, radius_squared),
                    "atom {index}, r² {radius_squared}"
                );
            }
        }
    }
}
