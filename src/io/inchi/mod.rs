//! Reading InChI, the IUPAC International Chemical Identifier, back into
//! molecules.
//!
//! An InChI is not a structure. Its main layer keeps the atoms, the σ bonds
//! between them, how many hydrogens each atom holds, the sets of atoms a
//! mobile hydrogen may sit on, and each component's charge -- and nothing
//! else: bond orders are gone, and so is which atom carries a charge or an
//! unpaired electron. Any placement of those that gives every atom a valence
//! its element has, and meets the totals the layers state, is a structure the
//! identifier describes. The reader writes the one a chemist would draw: no
//! charge separation it can avoid, no unpaired electron it can pair, and the
//! charges where InChI itself would have left them. See [`structure`] for how.
//!
//! What it reads is standard InChI: the formula, `/c`, `/h`, `/q`, `/p`, `/b`,
//! `/t`, `/m`, `/s`, and the isotopic layer `/i` with its own `/h` and stereo.
//! A non-standard string is read as far as its main layer: the fixed-H layer
//! `/f` and the reconnected-metal layer `/r` describe other structures and are
//! skipped, so metals come back disconnected, as standard InChI has them.
//! Polymers are refused.
//!
//! ```
//! use oxidane::prelude::*;
//!
//! // Nitromethane. The layers say nothing about charge; the reader writes
//! // the nitro group as the charge-separated pair it is.
//! let nitromethane = MolecularGraph::from_inchi("InChI=1S/CH3NO2/c1-2(3)4/h1H3").unwrap();
//! assert_eq!(nitromethane[0].to_smiles(), "C[N+](=O)[O-]");
//! ```

mod elements;
mod layers;
mod structure;

pub use layers::{
    InChI, InChIComponent, IsotopicAtom, MobileGroup, Parity, StereoBond, StereoCentre, parse_inchi,
};

use crate::molecule::MolecularGraph;

/// Why an InChI could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InChIError {
    /// The string does not start `InChI=`.
    MissingPrefix,
    /// A layer is not well formed.
    Syntax { layer: char, message: String },
    /// The formula names something that is not an element.
    UnknownElement(String),
    /// The layers contradict one another.
    Inconsistent(String),
    /// A kind of InChI the reader does not build, such as a polymer.
    Unsupported(String),
    /// No placement of π bonds, charges and electrons gives the component, by
    /// its position in the formula from one, a valence for every atom.
    NoStructure(usize),
}

impl std::fmt::Display for InChIError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InChIError::MissingPrefix => write!(formatter, "not an InChI: no `InChI=` prefix"),
            InChIError::Syntax { layer, message } => {
                write!(formatter, "malformed /{layer} layer: {message}")
            }
            InChIError::UnknownElement(symbol) => write!(formatter, "unknown element {symbol}"),
            InChIError::Inconsistent(message) => write!(formatter, "inconsistent InChI: {message}"),
            InChIError::Unsupported(what) => write!(formatter, "unsupported: {what}"),
            InChIError::NoStructure(component) => {
                write!(formatter, "component {component} has no valid structure")
            }
        }
    }
}

impl std::error::Error for InChIError {}

impl MolecularGraph {
    /// Every component an InChI describes, in the order its formula lists
    /// them.
    ///
    /// A proton `/p` adds where no atom can take it comes back as a component
    /// of its own, `[H+]`.
    pub fn from_inchi(inchi: &str) -> Result<Vec<Self>, InChIError> {
        structure::build(&parse_inchi(inchi)?)
    }
}
