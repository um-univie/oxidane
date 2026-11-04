//! The Oxidane prelude.
//!
//! Re-exports the primary data structures, traits, and functions needed
//! for day-to-day cheminformatics workflows.
//!
//! Bringing this module into scope with `use oxidane::prelude::*;` provides:
//!
//! - **Molecular representations**: [`MolecularGraph`] (2D/topological molecule) and [`Molecule3D`] (molecule with 3D coordinates).
//! - **Core trait extensions**: [`Molecule`], [`MoleculeCore`], [`MolecularDescriptors`], [`Embed3D`], [`Draw`], [`ToSMILES`].
//! - **Pattern matching**: [`parse_cxsmarts`], [`matches_cxsmarts`], [`find_cxsmarts_matches`].
//! - **2D Depiction**: [`Draw`], [`DrawOptions`].
//! - **Geometry**: [`Vector3D`].
//!
//! # Example
//!
//! ```rust
//! use oxidane::prelude::*;
//!
//! // Acetylsalicylic acid (aspirin)
//! let aspirin = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();
//!
//! assert_eq!(aspirin.molecular_formula().to_string(), "C9H8O4");
//! assert!((aspirin.molecular_weight() - 180.16).abs() < 0.05);
//! assert!(matches_cxsmarts(&aspirin, "[CX3](=O)[OX2H1]").unwrap()); // Carboxylic acid
//! ```

pub use crate::molecule::draw::{Draw, DrawOptions};
pub use crate::{
    atom::Atom,
    graph_algorithms::mces::{
        AtomMapping, BondMapping, CommonSubstructureSize, MaximumCommonEdgeSubgraph,
        common_edge_subgraph_size, find_all_maximum_common_edge_subgraphs,
        find_maximum_common_edge_subgraph,
    },
    io::{CxSMARTSError, CxSMARTSPattern, find_cxsmarts_matches, matches_cxsmarts, parse_cxsmarts},
    molecule::{MolecularGraph, Molecule3D, bond::BondOrder, chirality::ChiralClass, traits::*},
    vector::Vector3D,
};
