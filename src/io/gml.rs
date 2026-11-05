use crate::molecule::Molecule;
use chemistry_consts::ElementProperties;

pub trait ToGml {
    fn to_gml(&self) -> String;
    fn to_gml_with_slice(&self, filter: &[usize], outer_filter: &[usize]) -> String;
}

impl<T: Molecule> ToGml for T {
    /// Serializes the molecule to Graph Modelling Language (GML) format.
    ///
    /// # Examples
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = Molecule3D::from_xyz("tests/ethane.xyz").unwrap().remove(0);
    /// let gml = molecule.to_gml();
    /// assert!(gml.contains("graph"));
    /// ```
    ///
    fn to_gml(&self) -> String {
        let components = self.get_components();
        let mut result_string = String::new();

        for component in components {
            result_string.push_str(&self.to_gml_with_slice(&component, &[]));
            result_string.push('\n');
        }
        result_string
    }

    fn to_gml_with_slice(&self, filter: &[usize], outer_filter: &[usize]) -> String {
        let mut bonds = String::new();
        let mut atoms = String::new();
        let mut result_string = String::new();
        let atom_bonds = self.atom_bonds();

        for &index in filter.iter() {
            let mut charge = match self.get_atom_charge(index) {
                charge @ 1..=i8::MAX => format!("{}+", charge),
                charge @ i8::MIN..=-1 => format!("{}-", charge.abs()),
                0 => String::new(),
            };
            if self.is_atom_radical(index) {
                charge.push('.');
            }
            let atomic_number = self.get_atomic_number(index);
            let atomic_symbol = atomic_number.atomic_symbol().unwrap_or("Unknown");
            atoms.push_str(&format!(
                "node [id {index} label \"{}{}\"]\n",
                atomic_symbol, charge
            ));
            for &bond in atom_bonds[index].iter() {
                if index < bond.target()
                    && (filter.contains(&bond.target()) || outer_filter.contains(&bond.target()))
                {
                    let label = bond.bond_order().to_string();
                    bonds.push_str(&format!(
                        "edge [source {index} target {} label \"{}\"]\n",
                        bond.target(),
                        label
                    ));
                }
            }
        }
        result_string.push_str("graph [");
        if !atoms.is_empty() {
            result_string.push('\n');
            result_string.push_str(&atoms);
            result_string.push('\n');
        }
        if !bonds.is_empty() {
            if atoms.is_empty() {
                result_string.push('\n');
            }
            result_string.push_str(&bonds);
            result_string.push('\n');
        }
        result_string.push(']');
        result_string
    }
}

/// Represents a node in the GML graph
#[derive(Debug, Clone)]
pub struct Node {
    /// Unique identifier for the node
    pub id: usize,
    /// Chemical element label (e.g., "C", "O", "H", etc.)
    pub label: String,
}

/// Represents an edge in the GML graph
#[derive(Debug, Clone)]
pub struct Edge {
    /// Source node ID
    pub source: usize,
    /// Target node ID
    pub target: usize,
    /// Bond type label (e.g., "-" for single bond)
    pub label: String,
}

/// Represents a section of the reaction rule (left, context, or right)
#[derive(Debug, Clone, Default)]
pub struct GraphSection {
    /// List of nodes in this section
    pub nodes: Vec<Node>,
    /// List of edges in this section
    pub edges: Vec<Edge>,
}
