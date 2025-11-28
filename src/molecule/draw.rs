// ... existing imports ...
use crate::{molecule::base::Molecule, molecule::bond::BondOrder};
use rand::prelude::*;
use std::f64::consts::PI;

pub trait Draw<M: Molecule> {
    /// Converts the molecule to an SVG string representation.
    /// The SVG will show atoms as labeled circles and bonds as lines between them.
    ///
    /// # Returns
    /// A string containing the SVG representation of the molecule
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::draw::Draw;
    /// let molecule = MolecularGraph::from_smiles("CC").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// let svg = molecule.to_svg();
    /// let molecule = MolecularGraph::from_smiles("C1CCCCC1").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// let svg = molecule.to_svg();
    /// let molecule = MolecularGraph::from_smiles("c1ccccc1").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// let svg = molecule.to_svg();
    /// assert!(svg.starts_with("<svg"));
    /// assert!(svg.ends_with("</svg>"), "the document has to be closed");
    /// ```
    fn to_svg(&self) -> String
    where
        Self: Molecule,
    {
        const CANVAS_SIZE: i32 = 400;
        const ATOM_RADIUS: i32 = 15;
        const MARGIN: i32 = 40;

        // Calculate 2D coordinates for atoms (you may already have these)
        let coordinates = self.get_2d_coordinates();

        // Scale coordinates to fit canvas
        let (min_x, max_x, min_y, max_y) = coordinates.iter().fold(
            (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
            |(min_x, max_x, min_y, max_y), (x, y)| {
                (min_x.min(*x), max_x.max(*x), min_y.min(*y), max_y.max(*y))
            },
        );

        let scale_factor =
            (CANVAS_SIZE - 2 * MARGIN) as f64 / (max_x - min_x).max(max_y - min_y).max(1.0);

        let mut svg = format!(
            r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
            CANVAS_SIZE, CANVAS_SIZE
        );

        // Draw bonds
        for atom_idx in 0..self.len() {
            let (x1, y1) =
                self.scale_coordinate(coordinates[atom_idx], (min_x, min_y), scale_factor, MARGIN);

            let bonds = self.get_atom_bonds(atom_idx).unwrap();

            for bond in bonds {
                let target = bond.target();
                if atom_idx < target {
                    // Draw each bond only once
                    let (x2, y2) = self.scale_coordinate(
                        coordinates[target],
                        (min_x, min_y),
                        scale_factor,
                        MARGIN,
                    );

                    match bond.bond_order() {
                        BondOrder::Single => {
                            svg.push_str(&format!(
                                r#"<line x1="{}" y1="{}" x2="{}" y2="{}"
                                stroke="black" stroke-width="2"/>"#,
                                x1, y1, x2, y2
                            ));
                        }
                        BondOrder::Double => {
                            // Draw parallel lines for double bond
                            let (dx, dy) = self.get_parallel_offset(x1, y1, x2, y2, 3.0);
                            svg.push_str(&format!(
                                r#"<line x1="{}" y1="{}" x2="{}" y2="{}"
                                stroke="black" stroke-width="2"/>
                                <line x1="{}" y1="{}" x2="{}" y2="{}"
                                stroke="black" stroke-width="2"/>"#,
                                x1 + dx,
                                y1 + dy,
                                x2 + dx,
                                y2 + dy,
                                x1 - dx,
                                y1 - dy,
                                x2 - dx,
                                y2 - dy
                            ));
                        }
                        // Add other bond types as needed
                        _ => {}
                    }
                }
            }
        }

        // Draw atoms
        for (idx, (x, y)) in coordinates.iter().enumerate() {
            let (x, y) = self.scale_coordinate((*x, *y), (min_x, min_y), scale_factor, MARGIN);
            let symbol = self.get_atomic_symbol(idx).unwrap_or("?");

            // Only show non-carbon atoms or carbons with special properties
            if symbol != "C" || self.charges()[idx] != 0 || self.is_atom_chiral(idx) {
                svg.push_str(&format!(
                    r#"<circle cx="{}" cy="{}" r="{}" fill="white" stroke="black"/>
                    <text x="{}" y="{}" text-anchor="middle" dominant-baseline="middle">{}</text>"#,
                    x, y, ATOM_RADIUS, x, y, symbol
                ));
            }
        }

        svg.push_str("</svg>");
        svg
    }

    fn scale_coordinate(
        &self,
        (x, y): (f64, f64),
        (min_x, min_y): (f64, f64),
        scale_factor: f64,
        margin: i32,
    ) -> (i32, i32) {
        let x = ((x - min_x) * scale_factor) as i32 + margin;
        let y = ((y - min_y) * scale_factor) as i32 + margin;
        (x, y)
    }

    fn get_parallel_offset(&self, x1: i32, y1: i32, x2: i32, y2: i32, distance: f64) -> (i32, i32) {
        let dx = (x2 - x1) as f64;
        let dy = (y2 - y1) as f64;
        let length = (dx * dx + dy * dy).sqrt();

        if length == 0.0 {
            return (0, 0);
        }

        let dx = -dy * distance / length;
        let dy = dx * distance / length;

        (dx as i32, dy as i32)
    }
    /// Generates 2D coordinates for the molecule using a force-directed layout algorithm
    /// (Fruchterman-Reingold).
    ///
    /// # Returns
    /// A vector of (x,y) coordinates for each atom in the molecule
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::draw::Draw;
    /// let molecule = MolecularGraph::from_smiles("CC").unwrap();
    /// let molecule = molecule.first().unwrap();
    /// let coords = molecule.get_2d_coordinates();
    /// assert_eq!(coords.len(), molecule.len());
    /// ```
    fn get_2d_coordinates(&self) -> Vec<(f64, f64)>
    where
        Self: Molecule,
    {
        const MAX_ITERATIONS: usize = 100;
        const TEMPERATURE: f64 = 1.0;
        const COOLING_FACTOR: f64 = 0.95;
        const K: f64 = 1.0; // Optimal distance between vertices

        let n_atoms = self.len();
        if n_atoms == 0 {
            return vec![];
        }
        if n_atoms == 1 {
            return vec![(0.0, 0.0)];
        }

        // Initialize random positions
        let mut rng = rand::rng();
        let mut positions: Vec<(f64, f64)> = (0..n_atoms)
            .map(|_| (rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
            .collect();

        let mut temp = TEMPERATURE;

        // Create adjacency list for faster neighbor lookup
        let mut adjacency_list = vec![Vec::new(); n_atoms];
        for (i, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                adjacency_list[i].push(bond.target());
            }
        }

        for _ in 0..MAX_ITERATIONS {
            // Calculate repulsive forces
            let mut displacement = vec![(0.0, 0.0); n_atoms];

            // Repulsive forces between all pairs of vertices
            for i in 0..n_atoms {
                for j in (i + 1)..n_atoms {
                    let dx = positions[i].0 - positions[j].0;
                    let dy = positions[i].1 - positions[j].1;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                    // Repulsive force is proportional to 1/distance
                    let force = K * K / dist;
                    let fx = force * dx / dist;
                    let fy = force * dy / dist;

                    displacement[i].0 += fx;
                    displacement[i].1 += fy;
                    displacement[j].0 -= fx;
                    displacement[j].1 -= fy;
                }
            }

            // Attractive forces along edges
            for (i, neighbors) in adjacency_list.iter().enumerate() {
                for &j in neighbors {
                    if i < j {
                        // Process each edge only once
                        let dx = positions[i].0 - positions[j].0;
                        let dy = positions[i].1 - positions[j].1;
                        let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                        // Attractive force is proportional to distance
                        let force = dist * dist / K;
                        let fx = force * dx / dist;
                        let fy = force * dy / dist;

                        displacement[i].0 -= fx;
                        displacement[i].1 -= fy;
                        displacement[j].0 += fx;
                        displacement[j].1 += fy;
                    }
                }
            }

            // Apply displacement with temperature limiting
            for i in 0..n_atoms {
                let dx = displacement[i].0;
                let dy = displacement[i].1;
                let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                // Limit maximum displacement to temperature
                let limited_dist = dist.min(temp);
                positions[i].0 += dx / dist * limited_dist;
                positions[i].1 += dy / dist * limited_dist;
            }

            // Cool temperature
            temp *= COOLING_FACTOR;
        }

        // Normalize positions
        let (min_x, max_x, min_y, max_y) = positions.iter().fold(
            (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
            |(min_x, max_x, min_y, max_y), &(x, y)| {
                (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
            },
        );

        let width = max_x - min_x;
        let height = max_y - min_y;
        let scale = 1.0 / width.max(height);

        positions.iter_mut().for_each(|(x, y)| {
            *x = (*x - min_x) * scale;
            *y = (*y - min_y) * scale;
        });

        // Adjust for bond angles if molecule is cyclic
        if self.is_cyclic() {
            self.adjust_cyclic_layout(&mut positions);
        }

        positions
    }

    /// Checks if the molecule contains any cycles
    fn is_cyclic(&self) -> bool
    where
        Self: Molecule,
    {
        let mut visited = vec![false; self.len()];
        let mut stack = vec![false; self.len()];

        for start in 0..self.len() {
            if !visited[start] {
                if self.has_cycle_dfs(start, None, &mut visited, &mut stack) {
                    return true;
                }
            }
        }
        false
    }

    /// Helper function for cycle detection using DFS
    fn has_cycle_dfs(
        &self,
        vertex: usize,
        parent: Option<usize>,
        visited: &mut [bool],
        stack: &mut [bool],
    ) -> bool
    where
        Self: Molecule,
    {
        visited[vertex] = true;
        stack[vertex] = true;

        if let Some(bonds) = self.get_atom_bonds(vertex) {
            for bond in bonds {
                let neighbor = bond.target();
                if !visited[neighbor] {
                    if self.has_cycle_dfs(neighbor, Some(vertex), visited, stack) {
                        return true;
                    }
                } else if stack[neighbor] && Some(neighbor) != parent {
                    return true;
                }
            }
        }

        stack[vertex] = false;
        false
    }

    /// Adjusts the layout for cyclic molecules to make them more regular
    fn adjust_cyclic_layout(&self, positions: &mut [(f64, f64)])
    where
        Self: Molecule,
    {
        // Find cycles in the molecule
        let cycles = self.find_cycles();

        for cycle in cycles {
            if cycle.len() >= 3 {
                // Calculate the center of the cycle
                let center = cycle.iter().fold((0.0, 0.0), |acc, &idx| {
                    (acc.0 + positions[idx].0, acc.1 + positions[idx].1)
                });
                let center = (center.0 / cycle.len() as f64, center.1 / cycle.len() as f64);

                // Arrange atoms in a regular polygon
                let radius = 0.3; // Adjust this value to change the size of cycles
                for (i, &idx) in cycle.iter().enumerate() {
                    let angle = 2.0 * PI * i as f64 / cycle.len() as f64;
                    positions[idx] = (
                        center.0 + radius * angle.cos(),
                        center.1 + radius * angle.sin(),
                    );
                }
            }
        }
    }

    /// Finds all cycles in the molecule
    fn find_cycles(&self) -> Vec<Vec<usize>>
    where
        Self: Molecule,
    {
        let mut cycles = Vec::new();
        let mut visited = vec![false; self.len()];
        let mut path = Vec::new();

        for start in 0..self.len() {
            if !visited[start] {
                self.find_cycles_dfs(start, None, &mut visited, &mut path, &mut cycles);
            }
        }

        cycles
    }

    /// Helper function for cycle detection using DFS
    fn find_cycles_dfs(
        &self,
        vertex: usize,
        parent: Option<usize>,
        visited: &mut [bool],
        path: &mut Vec<usize>,
        cycles: &mut Vec<Vec<usize>>,
    ) where
        Self: Molecule,
    {
        visited[vertex] = true;
        path.push(vertex);

        if let Some(bonds) = self.get_atom_bonds(vertex) {
            for bond in bonds {
                let neighbor = bond.target();
                if !visited[neighbor] {
                    self.find_cycles_dfs(neighbor, Some(vertex), visited, path, cycles);
                } else if Some(neighbor) != parent && path.contains(&neighbor) {
                    // Found a cycle
                    if let Some(start_idx) = path.iter().position(|&x| x == neighbor) {
                        let cycle: Vec<_> = path[start_idx..].to_vec();
                        if !cycles.contains(&cycle) && cycle.len() > 2 {
                            cycles.push(cycle);
                        }
                    }
                }
            }
        }

        path.pop();
    }
}

impl<M: Molecule> Draw<M> for M {}
