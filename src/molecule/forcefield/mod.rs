//! Force fields, for relaxing 3D structures and for scoring them.
//!
//! Two of them live here. [`uff`] is the Universal Force Field, whose
//! parameter table spans the periodic table and whose typing covers
//! fourteen main-group elements; [`crate::molecule::embed`] relaxes against
//! it wherever MMFF94 cannot type the molecule. [`mmff`] is a complete
//! MMFF94 -- typing, charges and all seven energy terms -- restricted to
//! organic main-group chemistry but roughly ten times closer to a
//! semiempirical reference on the structures this crate produces.
//!
//! The unqualified [`ForceField`] is UFF's; [`Field::best_for`] selects the
//! field used for embedding relaxation.

pub(crate) mod lanes;
pub mod mmff;
pub mod uff;

pub use uff::ForceField;

use crate::molecule::base::MoleculeCore;
use crate::molecule::minimize::{Energy, minimize};
use crate::vector::Vector3D;

/// Convergence threshold on the largest force, in kcal/mol per Angstrom,
/// scaled by each coordinate's own size -- see `minimize`'s convergence
/// tests. RDKit's default for the same test is 1e-4.
const GRADIENT_TOLERANCE: f64 = 1e-4;

/// The better of the two force fields for a given molecule.
/// MMFF94 where it types the molecule, UFF where it does not.
///
/// MMFF94 is the more accurate by roughly a factor of ten against a
/// semiempirical reference, and it is what this picks whenever it can type
/// the molecule. It cannot always: MMFF94 is organic main-group chemistry
/// and a handful of counterions, and UFF's own perception covers fourteen
/// elements rather than the periodic table its parameter table spans. A
/// ferrocene or a platinum complex is typed by neither, so this returns
/// [`UntypedAtom`] and the caller decides what an unrelaxed structure is
/// worth -- `embed_3d` keeps its distance geometry and says nothing.
pub enum Field {
    Mmff(mmff::ForceField),
    Uff(ForceField),
}

impl Field {
    /// MMFF94 where it types the molecule, UFF where it does not.
    ///
    /// # Errors
    ///
    /// [`UntypedAtom`] only when neither field covers the molecule, which
    /// means an element outside UFF's own perception rules.
    pub fn best_for<M: MoleculeCore + ?Sized>(molecule: &M) -> Result<Self, UntypedAtom> {
        match mmff::ForceField::new(molecule) {
            Ok(field) => Ok(Self::Mmff(field)),
            Err(_) => ForceField::new(molecule).map(Self::Uff),
        }
    }

    /// The MMFF94 field behind this, where that is what it is.
    ///
    /// For a caller that wants something only MMFF has -- its equilibrium
    /// geometry, say -- without building it a second time.
    pub fn mmff(&self) -> Option<&mmff::ForceField> {
        match self {
            Self::Mmff(field) => Some(field),
            Self::Uff(_) => None,
        }
    }

    /// Which of the two this is, for a caller that wants to say so.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Mmff(_) => "MMFF94",
            Self::Uff(_) => "UFF",
        }
    }

    pub fn energy(&self, positions: &[Vector3D]) -> f64 {
        <Self as Energy>::energy(self, positions)
    }

    /// Relaxes `positions` in place, returning the final energy.
    pub fn minimize(&self, positions: &mut [Vector3D], max_iterations: usize) -> f64 {
        minimize(positions, self, max_iterations, GRADIENT_TOLERANCE)
    }
}

impl Energy for Field {
    fn energy(&self, positions: &[Vector3D]) -> f64 {
        match self {
            Self::Mmff(field) => field.energy(positions),
            Self::Uff(field) => field.energy(positions),
        }
    }

    fn energy_and_gradient(&self, positions: &[Vector3D], gradient: &mut [Vector3D]) -> f64 {
        match self {
            Self::Mmff(field) => field.energy_and_gradient(positions, gradient),
            Self::Uff(field) => field.energy_and_gradient(positions, gradient),
        }
    }
}

/// Why an atom could not be given a force-field type.
///
/// Typing is where a force field is most often quietly wrong, because a
/// mistyped atom still produces a number. Refusing to type an atom at all is
/// the honest half of that: a molecule this error names is one the field has
/// no business scoring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UntypedAtom {
    pub atom: usize,
    pub atomic_number: u8,
    /// Which force field's rules ran out -- `"UFF"` or `"MMFF94"`.
    scheme: &'static str,
}

impl UntypedAtom {
    pub(crate) fn uff(atom: usize, atomic_number: u8) -> Self {
        Self {
            atom,
            atomic_number,
            scheme: "UFF",
        }
    }

    pub(crate) fn mmff(atom: usize, atomic_number: u8) -> Self {
        Self {
            atom,
            atomic_number,
            scheme: "MMFF94",
        }
    }

    /// The force field whose typing rules did not cover the atom.
    pub fn scheme(&self) -> &'static str {
        self.scheme
    }
}

impl std::fmt::Display for UntypedAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "atom {} (element {}) has no {} type in this crate's typing rules",
            self.atom, self.atomic_number, self.scheme
        )
    }
}

impl std::error::Error for UntypedAtom {}
