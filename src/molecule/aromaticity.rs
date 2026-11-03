use crate::{
    molecule::base::MoleculeCore, molecule::bond::BondOrder, molecule::chirality::ChiralClass,
    molecule::graph::MoleculeGraph, molecule::utils::order_tuple,
};
use bit_set::BitSet;
use bitvec::prelude::BitVec;
use rustc_hash::FxHashSet;
use std::cell::Cell;

#[derive(Debug, PartialEq, Eq)]
pub enum AromaticityType {
    Aromatic,
    MoebiusAromatic,
    AntiAromatic,
    Metallocenic,
    NonAromatic,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum AromaticityFlag {
    PotentiallyAromaticZeroElectrons = 0,
    PotentiallyAromaticOneElectron = 1,
    PotentiallyAromaticTwoElectrons = 2,
    NotAromatic = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AromaticityModel {
    Daylight = 0,
    OpenSMILES = 1,
    /// The OpenSMILES electron rules, decided over the molecule's ring family
    /// rather than over paths out of each bond -- see
    /// [`crate::molecule::ring_aromaticity`].
    ///
    /// The two differ only in *what is scored*: the per-atom electron counts,
    /// the exocyclic rule and the Kekule check are the ones the path search
    /// uses. What the ring model buys is that its search is polynomial and its
    /// answer is a union rather than whichever witness was found first, so no
    /// bound on the work can move the answer.
    Rings = 2,
}

const AROMATICITY_MODEL_DAYLIGHT: [AromaticityFlag; 34] = [
    AromaticityFlag::NotAromatic, // 0: Default/invalid atom
    AromaticityFlag::NotAromatic, // 1: Boron with 0 charge, valency=3, bond_count=3
    AromaticityFlag::NotAromatic, // 2: Boron with 0 charge, valency=3, bond_count=2 or -1 charge, valency=4, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 3: Carbon with 0 charge, standard aromatic (temp_lookup_index=3)
    AromaticityFlag::PotentiallyAromaticOneElectron, // 4: Carbon with 0 charge, multiple aromatic bonds or exocyclic double bond
    AromaticityFlag::PotentiallyAromaticZeroElectrons, // 5: Carbon with 0 charge, exocyclic double bond to non-carbon
    AromaticityFlag::PotentiallyAromaticOneElectron, // 6: Carbon with -1 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 7: Carbon with -1 charge, valency=3, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 8: Carbon with +1 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticZeroElectrons, // 9: Carbon with +1 charge, valency=3, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 10: Nitrogen with 0 charge, valency=3/4, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 11: Nitrogen with 0 charge, valency=3/4, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 12: Nitrogen with 0 charge, valency=5, bond_count=3, N-oxide
    AromaticityFlag::PotentiallyAromaticOneElectron, // 13: Nitrogen with +1 charge, valency=4, bond_count=3, exocyclic double bond to oxygen
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 14: Nitrogen with -1 charge, valency=2
    AromaticityFlag::PotentiallyAromaticOneElectron, // 15: Nitrogen with +1 charge, valency=4, bond_count=3, no exocyclic double bond to oxygen
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 16: Oxygen with 0 charge, valency=2
    AromaticityFlag::PotentiallyAromaticOneElectron, // 17: Oxygen with +1 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 18: Sulfur with 0 charge, valency=2
    AromaticityFlag::PotentiallyAromaticOneElectron, // 19: Sulfur with +1 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 20: Sulfur with 0 charge, valency=4, bond_count=3, exocyclic double bond to oxygen
    AromaticityFlag::NotAromatic, // 21: Sulfur with +1 charge, valency=3, bond_count=3, exocyclic bond to oxygen
    AromaticityFlag::PotentiallyAromaticOneElectron, // 22: Phosphorus with 0 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 23: Phosphorus with 0 charge, valency=3, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 24: Phosphorus with 0 charge, valency=5, bond_count=3, exocyclic bond to oxygen
    AromaticityFlag::PotentiallyAromaticOneElectron, // 25: Phosphorus with +1 charge, valency=4, bond_count=3, exocyclic double bond to oxygen
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 26: Phosphorus with -1 charge, valency=2
    AromaticityFlag::PotentiallyAromaticOneElectron, // 27: Phosphorus with +1 charge, valency=4, bond_count=3, no exocyclic double bond to oxygen
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 28: Selenium with 0 charge, valency=2, or a selenoxide [Se](=O) / [Se+]([O-])
    AromaticityFlag::PotentiallyAromaticOneElectron, // 29: Selenium with +1 charge, valency=4, bond_count=2
    AromaticityFlag::NotAromatic,                    // 30: Tellurium with 0 charge, valency=2
    AromaticityFlag::NotAromatic, // 31: Tellurium with +1 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 32: Arsenic with 0 charge, valency=3, bond_count=3
    AromaticityFlag::NotAromatic, // 33: Arsenic with +1 charge, valency=4, bond_count=3
];

const AROMATICITY_MODEL_OPENSMILES: [AromaticityFlag; 34] = [
    AromaticityFlag::NotAromatic, // 0: Default/invalid atom
    AromaticityFlag::PotentiallyAromaticZeroElectrons, // 1: Boron with 0 charge, valency=3, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 2: Boron with 0 charge, valency=3, bond_count=2 or -1 charge, valency=4, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 3: Carbon with 0 charge, standard aromatic (temp_lookup_index=3)
    AromaticityFlag::PotentiallyAromaticOneElectron, // 4: Carbon with 0 charge, multiple aromatic bonds or exocyclic double bond
    AromaticityFlag::PotentiallyAromaticZeroElectrons, // 5: Carbon with 0 charge, exocyclic double bond to non-carbon
    AromaticityFlag::PotentiallyAromaticOneElectron, // 6: Carbon with -1 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 7: Carbon with -1 charge, valency=3, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 8: Carbon with +1 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticZeroElectrons, // 9: Carbon with +1 charge, valency=3, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 10: Nitrogen with 0 charge, valency=3/4, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 11: Nitrogen with 0 charge, valency=3/4, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 12: Nitrogen with 0 charge, valency=5, bond_count=3, N-oxide
    AromaticityFlag::PotentiallyAromaticZeroElectrons, // 13: Nitrogen with +1 charge, valency=4, bond_count=3, exocyclic double bond to oxygen
    AromaticityFlag::PotentiallyAromaticTwoElectrons,  // 14: Nitrogen with -1 charge, valency=2
    AromaticityFlag::PotentiallyAromaticOneElectron, // 15: Nitrogen with +1 charge, valency=4, bond_count=3, no exocyclic double bond to oxygen
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 16: Oxygen with 0 charge, valency=2
    AromaticityFlag::PotentiallyAromaticOneElectron, // 17: Oxygen with +1 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 18: Sulfur with 0 charge, valency=2
    AromaticityFlag::PotentiallyAromaticOneElectron, // 19: Sulfur with +1 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 20: Sulfur with 0 charge, valency=4, bond_count=3, exocyclic double bond to oxygen
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 21: Sulfur with +1 charge, valency=3, bond_count=3, exocyclic bond to oxygen
    AromaticityFlag::PotentiallyAromaticOneElectron, // 22: Phosphorus with 0 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 23: Phosphorus with 0 charge, valency=3, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 24: Phosphorus with 0 charge, valency=5, bond_count=3, exocyclic bond to oxygen
    AromaticityFlag::PotentiallyAromaticZeroElectrons, // 25: Phosphorus with +1 charge, valency=4, bond_count=3, exocyclic double bond to oxygen
    AromaticityFlag::PotentiallyAromaticTwoElectrons,  // 26: Phosphorus with -1 charge, valency=2
    AromaticityFlag::PotentiallyAromaticOneElectron, // 27: Phosphorus with +1 charge, valency=4, bond_count=3, no exocyclic double bond to oxygen
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 28: Selenium with 0 charge, valency=2, or a selenoxide [Se](=O) / [Se+]([O-])
    AromaticityFlag::PotentiallyAromaticOneElectron, // 29: Selenium with +1 charge, valency=4, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 30: Tellurium with 0 charge, valency=2
    AromaticityFlag::PotentiallyAromaticOneElectron, // 31: Tellurium with +1 charge, valency=3, bond_count=2
    AromaticityFlag::PotentiallyAromaticTwoElectrons, // 32: Arsenic with 0 charge, valency=3, bond_count=3
    AromaticityFlag::PotentiallyAromaticOneElectron, // 33: Arsenic with +1 charge, valency=4, bond_count=3
];

/// The model perception uses when a caller does not name one.
///
/// Parsing and canonization both perceive, and both have to perceive under the
/// *same* model or a molecule read one way and canonized the other is compared
/// against itself under two different chemistries. Rather than thread a model
/// through every entry point, they ask here.
///
/// It answers [`AromaticityModel::OpenSMILES`] unless it has been told
/// otherwise, by [`set_default_aromaticity_model`] or by the environment
/// variable `OXIDANE_AROMATICITY` (`daylight`, `opensmiles` or `rings`), which
/// is read once. The environment is what lets one binary sweep a corpus under
/// both models without being rebuilt -- which is how the ring model is
/// measured against the path search.
pub fn default_aromaticity_model() -> AromaticityModel {
    match DEFAULT_MODEL.load(std::sync::atomic::Ordering::Relaxed) {
        UNSET => {
            let model = match std::env::var("OXIDANE_AROMATICITY")
                .map(|value| value.trim().to_ascii_lowercase())
                .as_deref()
            {
                Ok("daylight") => AromaticityModel::Daylight,
                Ok("rings") | Ok("ring") => AromaticityModel::Rings,
                _ => AromaticityModel::OpenSMILES,
            };
            DEFAULT_MODEL.store(model as u8, std::sync::atomic::Ordering::Relaxed);
            model
        }
        0 => AromaticityModel::Daylight,
        2 => AromaticityModel::Rings,
        _ => AromaticityModel::OpenSMILES,
    }
}

/// Sets what [`default_aromaticity_model`] answers, for the whole process.
///
/// Perception decides what a canonical form *is*, so two molecules compared
/// across a change of this setting are not comparable. Set it once, before
/// anything is parsed.
pub fn set_default_aromaticity_model(model: AromaticityModel) {
    DEFAULT_MODEL.store(model as u8, std::sync::atomic::Ordering::Relaxed);
}

const UNSET: u8 = u8::MAX;
static DEFAULT_MODEL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(UNSET);

/// The working buffers one aromaticity perception needs.
///
/// Every field is sized by the molecule and cleared before use, never read
/// across calls: this is scratch space, not state. Keeping it lets a caller
/// that perceives many molecules pay for the allocations once.
#[derive(Debug, Default, Clone)]
pub struct AromaticityScratch {
    /// The ring model's buffers, kept for the same reason as the rest: it
    /// perceives a ring family per molecule and would otherwise allocate one.
    pub(crate) ring_finder: crate::molecule::ring_aromaticity::RingFinder,
    /// Which atoms could put an orbital into a ring, worked out before the
    /// rings are and used again when the π-capable bonds are taken.
    pub(crate) pi_capable: Vec<bool>,
    pub(crate) atoms_in_rings_placeholder: BitSet,
    pub(crate) bonds_in_rings_placeholder: FxHashSet<(usize, usize)>,
    pub(crate) aromaticity_flags: Vec<AromaticityFlag>,
    pub(crate) aromatic_atoms: Vec<bool>,
    pub(crate) aromatic_bonds: FxHashSet<(usize, usize)>,
    pub(crate) visited: Vec<u8>,
}

/// Perceives aromaticity, keeping its working buffers between molecules.
///
/// [`MoleculeAromaticity::aromatize`] allocates a fresh set of buffers on every
/// call, which is most of what the allocator is asked to do when parsing a
/// stream of SMILES. An `Aromatizer` holds them:
///
/// ```
/// use oxidane::prelude::*;
/// use oxidane::molecule::aromaticity::{AromaticityModel, Aromatizer};
///
/// let mut aromatizer = Aromatizer::new();
/// for smiles in ["C1=CC=CC=C1", "C1=CC=NC=C1"] {
///     let mut molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
///     aromatizer
///         .aromatize(&mut molecule, AromaticityModel::OpenSMILES, None, false)
///         .unwrap();
///     assert_eq!(molecule.get_aromatic_atoms().len(), 6);
/// }
/// ```
///
/// The result is identical to calling the trait method; only the allocation
/// traffic differs. Hold one per thread -- it is not shared, and it is cheap to
/// create if a thread only ever does one molecule.
#[derive(Debug, Default, Clone)]
pub struct Aromatizer {
    scratch: AromaticityScratch,
}

impl Aromatizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Perceives aromaticity in `molecule`, reusing this aromatizer's buffers.
    pub fn aromatize<M: MoleculeAromaticity + ?Sized>(
        &mut self,
        molecule: &mut M,
        aromaticity_model: AromaticityModel,
        path_length_limit: Option<usize>,
        strict_flag: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        molecule.aromatize_with(
            &mut self.scratch,
            aromaticity_model,
            path_length_limit,
            strict_flag,
        )
    }
}

thread_local! {
    /// The aromatizer [`MoleculeAromaticity::aromatize`] borrows, so that
    /// existing callers get the buffer reuse without threading a handle
    /// through every signature.
    static AROMATIZER: std::cell::Cell<Aromatizer> = std::cell::Cell::new(Aromatizer::new());
}

/// A molecule whose aromatic bonds state more π demand than any assignment of
/// double bonds can satisfy, and which therefore has no Kekulé structure.
///
/// `O=c1ccccc1` is the shortest example: the carbonyl carbon has spent its π
/// bond outside the ring, and the five carbons left each want one double bond
/// among themselves, which an odd number of atoms cannot share out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KekulizationError {
    /// The atoms left wanting a π bond, in index order.
    pub atoms: Vec<usize>,
}

impl std::fmt::Display for KekulizationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "the aromatic bonds cannot be written as alternating single and \
             double ones: no double bond is left for atom"
        )?;
        if self.atoms.len() > 1 {
            write!(formatter, "s")?;
        }
        for (position, atom) in self.atoms.iter().enumerate() {
            write!(formatter, "{} {atom}", if position == 0 { "" } else { "," })?;
        }
        Ok(())
    }
}

impl std::error::Error for KekulizationError {}

thread_local! {
    /// How many more atoms the long-path search may look at before it gives up
    /// on the molecule it is working on.
    ///
    /// The search asks whether a bond lies on *some* aromatic cycle, and it
    /// asks by walking simple paths out of it -- so its cost is the number of
    /// simple paths, which in a fused ring system grows exponentially with the
    /// length it allows, and the length it allows is the atom count. A
    /// drug-like molecule never notices; a 200-atom macrocycle of fused
    /// thiophenes (PubChem has several, CID 122232324 among them) does not
    /// finish in a day. Bounding the walk bounds perception, and the bound is
    /// spent per molecule rather than per bond so that a molecule cannot pay it
    /// many times over.
    static LONG_PATH_STEPS: Cell<u64> = const { Cell::new(0) };
}

/// The atoms one molecule's long-path search may look at before it gives up.
///
/// The number has to be set against what the corpus spends, because a
/// search that stops early is a search whose answer depends on the order the
/// atoms were written in -- and an aromaticity that depends on the writing is a
/// canonical form that does too. PubChem's heaviest search that finishes is a
/// fullerene-fused polycyclic at 1,077,445,319 steps; eight compounds sat
/// between 3.1 and 10.8 hundred million, and every one of them changed its
/// tautomer key when the budget was 20 million. So the bound is twice the
/// worst, and its main job is the case with no finish at all: a
/// 270-atom macrocycle of fused thiophenes (CID 122232324) whose paths do not
/// run out in a day, and which now takes 67 seconds and an answer.
const LONG_PATH_BUDGET: u64 = 2_000_000_000;

/// How far the aromatic path search may walk from the bond it is asking about.
///
/// The bound has to be a constant rather than a budget: a search that stops
/// when it runs out of work stops in a place the atom numbering chose, and
/// aromaticity that depends on the numbering is a canonical form that does
/// too. A constant depth stops in the same place for every writing of the
/// molecule.
///
/// The cost is then `roots * branching^depth` rather than exponential in the
/// atom count, and what it buys is paid for in cycles longer than the depth,
/// which are never perceived as cycles. Open Babel, which this search is a port
/// of, uses 14 (`OBAromaticTyperMolState::CheckAromaticity`); RDKit's
/// ring-combination search caps candidate rings at 24, "a bit bigger than the
/// outer ring in a porphyrin".
///
/// Twenty-four is measured rather than chosen. Against the unbounded search
/// over PubChem's first 200,000 records, and against the 70-heavy-atom
/// fullerene ester that the unbounded search cannot canonize consistently:
///
/// | depth | records whose canonical SMILES moves | that fullerene |
/// |---|---|---|
/// | 14 | 382 / 200,000 | 0.001 s |
/// | 18 | 24 | 0.01 s |
/// | 20 | 16 | 0.02 s |
/// | 24 | 2 | 0.06 s |
/// | 30 | 1 | 2.8 s |
///
/// At 14 the porphyrins go: a depth shorter than the macrocycle loses the
/// pyrroles with it. At 24 what still moves is two vat dyes whose conjugated
/// circuit is longer than any ring anyone would draw. The 200,000 records
/// spend at most 37,058 steps of the long-path budget, against its
/// 2,000,000,000 -- so with a constant depth the budget is never reached,
/// and perception does not depend on the atom numbering.
const AROMATIC_PATH_DEPTH: usize = 24;

/// Spends one step of the long-path budget, and says whether there was one.
fn spend_a_long_path_step() -> bool {
    LONG_PATH_STEPS.with(|steps| {
        let left = steps.get();
        steps.set(left.saturating_sub(1));
        left > 0
    })
}

pub trait MoleculeAromaticity: MoleculeCore + MoleculeGraph {
    // This could fail but the way we use it, it won't
    fn lookup_model(index: usize, aromaticity_model: &[AromaticityFlag]) -> AromaticityFlag {
        aromaticity_model[index]
    }

    fn handle_aromatic_atom(
        &self,
        atom_index: usize,
        aromaticity_model: &[AromaticityFlag],
        bonds_in_ring: &FxHashSet<(usize, usize)>,
    ) -> AromaticityFlag {
        let atomic_number = self.get_atomic_number(atom_index);
        let index = match atomic_number {
            5 =>
            // Boron
            {
                self.handle_boron_aromaticity(atom_index)
            }
            6 =>
            // Carbon
            {
                self.handle_carbon_aromaticity(atom_index, bonds_in_ring)
            }
            7 =>
            // Nitrogen
            {
                self.handle_nitrogen_aromaticity(atom_index, bonds_in_ring)
            }
            8 =>
            // Oxygen
            {
                self.handle_oxygen_aromaticity(atom_index)
            }
            15 =>
            // Phosphorus
            {
                self.handle_phosphorus_aromaticity(atom_index, bonds_in_ring)
            }
            16 =>
            // Sulfur
            {
                self.handle_sulfur_aromaticity(atom_index, bonds_in_ring)
            }
            33 =>
            // Arsenic
            {
                self.handle_arsenic_aromaticity(atom_index)
            }
            34 =>
            // Selenium
            {
                self.handle_selenium_aromaticity(atom_index, bonds_in_ring)
            }
            // Tellurium
            52 => self.handle_tellurium_aromaticity(atom_index),
            _ => 0,
        };
        //#[cfg(debug_assertions)]
        //println!("Index: {}, Lookup index: {}", atom_index, index);
        Self::lookup_model(index as usize, aromaticity_model)
    }

    fn handle_boron_aromaticity(&self, atom_index: usize) -> u8 {
        let lookup_index = 0;
        let formal_charge = self.get_atom_charge(atom_index);

        let valency = self.actual_valency(atom_index);
        match formal_charge {
            0 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 {
                    if bond_count == 3 {
                        return 1;
                    }
                    if bond_count == 2 {
                        return 2;
                    }
                }
                if valency == 3 && bond_count == 3 {
                    return 1;
                }
                if valency == 3 && bond_count == 2 {
                    return 2;
                }
            }
            -1 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;
                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 && bond_count == 3 {
                    return 2;
                }

                //if valency == 2 && bond_count == 2 {
                //    lookup_index = 2;
                //}
                if valency == 4 && bond_count == 3 {
                    return 2;
                }
            }
            _ => (),
        }
        lookup_index
    }

    fn handle_carbon_aromaticity(
        &self,
        atom_index: usize,
        bonds_in_ring: &FxHashSet<(usize, usize)>,
    ) -> u8 {
        let mut lookup_index = 0;
        let formal_charge = self.get_atom_charge(atom_index);
        match formal_charge {
            0 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let mut number_of_double_bonds = 0;
                let mut number_of_aromatic_bonds = 0;
                let mut is_exocyclic_double_bond = false;
                let mut is_any_non_carbon_neighbor = false;

                for bond in bonds.iter() {
                    let neighbor_atom_index = bond.target();
                    if bond.bond_order() == BondOrder::Double {
                        number_of_double_bonds += 1;

                        let atom_order = order_tuple(atom_index, neighbor_atom_index);
                        if !bonds_in_ring.contains(&atom_order) {
                            is_exocyclic_double_bond = true;
                            let neighbor_atomic_number =
                                self.get_atomic_number(neighbor_atom_index);
                            is_any_non_carbon_neighbor |= neighbor_atomic_number != 6;
                        }
                    }
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }

                let mut temp_lookup_index = 3;

                if number_of_aromatic_bonds > 1 {
                    temp_lookup_index = 4;
                }
                if is_exocyclic_double_bond {
                    temp_lookup_index = 4;
                }
                lookup_index = 5;
                if !is_any_non_carbon_neighbor {
                    lookup_index = temp_lookup_index
                }

                if number_of_double_bonds == 1 {
                    return lookup_index;
                } else {
                    if number_of_aromatic_bonds > 1 {
                        return 3;
                    }
                    return 0;
                }
            }
            -1 => {
                let valence = self.actual_valency(atom_index);
                let mut number_of_aromatic_bonds = 0;
                let mut number_of_single_bonds = 0;

                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                    if bond.bond_order() == BondOrder::Single {
                        number_of_single_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 {
                    if number_of_single_bonds == 1 {
                        return 7;
                    }
                    return 6;
                }

                if valence == 3 {
                    let bond_count = self.get_atom_bonds(atom_index).unwrap_or_default().len();

                    lookup_index = match bond_count {
                        2 => 6,
                        3 => 7,
                        _ => 0,
                    };
                }
            }
            1 => {
                let valence = self.actual_valency(atom_index);

                let mut number_of_aromatic_bonds = 0;
                let mut number_of_single_bonds = 0;

                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                    if bond.bond_order() == BondOrder::Single {
                        number_of_single_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 {
                    if number_of_single_bonds == 1 {
                        return 9;
                    }
                    return 8;
                }
                if valence == 3 {
                    let bond_count = self.get_atom_bonds(atom_index).unwrap_or_default().len();
                    lookup_index = match bond_count {
                        2 => 8,
                        3 => 9,
                        _ => 0,
                    };
                }
            }
            _ => (),
        }

        lookup_index
    }

    fn handle_nitrogen_aromaticity(
        &self,
        atom_index: usize,
        bonds_in_rings: &FxHashSet<(usize, usize)>,
    ) -> u8 {
        let formal_charge = self.get_atom_charge(atom_index);
        let mut lookup_index = 0;
        let valency = self.actual_valency(atom_index);
        match formal_charge {
            0 => {
                let bonds = self.get_atom_bonds(atom_index).unwrap_or_default();
                let bond_count = bonds.len();

                let mut number_of_aromatic_bonds = 0;
                let mut number_of_single_bonds = 0;

                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                    if bond.bond_order() == BondOrder::Single {
                        number_of_single_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 {
                    if number_of_single_bonds == 1 {
                        return 11;
                    }
                    return 10;
                }

                if valency == 3 {
                    lookup_index = match bond_count {
                        2 => 10,
                        3 => 11,
                        _ => 0,
                    };
                }

                // N-Oxide stuff
                if valency == 5 && bond_count == 3 {
                    for bond in bonds {
                        let neighbor_index = bond.target();
                        let atom_order = order_tuple(atom_index, neighbor_index);
                        let bond_in_ring = bonds_in_rings.contains(&atom_order);
                        if !bond_in_ring && self.get_atomic_number(bond.target()) == 8 {
                            lookup_index = 12;
                            break;
                        }
                    }
                }
            }
            -1 => {
                let mut number_of_aromatic_bonds = 0;
                let mut number_of_single_bonds = 0;

                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                    if bond.bond_order() == BondOrder::Single {
                        number_of_single_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 {
                    if number_of_single_bonds == 0 {
                        return 14;
                    } else {
                        return 0;
                    }
                }
                if valency == 2 {
                    lookup_index = 14;
                }
            }
            1 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };

                let bond_count = bonds.len();
                if bond_count != 3 {
                    // The 4 is a hack for aromatic nitrogen
                    return 0;
                }

                let mut number_of_aromatic_bonds = 0;
                let mut number_of_single_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                    if bond.bond_order() == BondOrder::Single {
                        number_of_single_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 {
                    if number_of_single_bonds == 1 {
                        return 15;
                    }
                }

                for bond in bonds {
                    let neighbor_index = bond.target();
                    let atom_order = order_tuple(atom_index, neighbor_index);
                    let bond_in_ring = bonds_in_rings.contains(&atom_order);
                    if !bond_in_ring
                        && self.get_atomic_number(bond.target()) == 8
                        && bond.bond_order() == BondOrder::Double
                    {
                        return 13;
                    }
                }
                if lookup_index == 0 {
                    return 15;
                }
            }
            _ => (),
        }
        lookup_index
    }

    fn handle_oxygen_aromaticity(&self, atom_index: usize) -> u8 {
        let formal_charge = self.get_atom_charge(atom_index);
        let valency = self.actual_valency(atom_index);
        let mut lookup_index = 0;
        match formal_charge {
            0 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 && bond_count == 2 {
                    return 16;
                }

                if valency == 2 {
                    lookup_index = 16
                }
            }
            1 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 && bond_count == 2 {
                    return 17;
                }
                if valency == 3 && bond_count == 2 {
                    return 17;
                }
            }
            _ => (),
        }

        lookup_index
    }

    fn handle_phosphorus_aromaticity(
        &self,
        atom_index: usize,
        bonds_in_rings: &FxHashSet<(usize, usize)>,
    ) -> u8 {
        let formal_charge = self.get_atom_charge(atom_index);
        let valency = self.actual_valency(atom_index);
        let lookup_index = 0;

        match formal_charge {
            0 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;
                let mut number_of_single_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                    if bond.bond_order() == BondOrder::Single {
                        number_of_single_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 {
                    match number_of_single_bonds {
                        0 => {
                            // Check for O-oxide
                            for bond in bonds {
                                let neighbor_index = bond.target();
                                let bond_in_ring = if atom_index < neighbor_index {
                                    bonds_in_rings.contains(&(atom_index, neighbor_index))
                                } else {
                                    bonds_in_rings.contains(&(neighbor_index, atom_index))
                                };
                                if !bond_in_ring && self.get_atomic_number(bond.target()) == 8 {
                                    return 24;
                                }
                            }
                            return 23;
                        }
                        1 => return 23,
                        _ => (),
                    }
                }

                if valency == 3 {
                    match bond_count {
                        2 => return 22,
                        3 => return 23,
                        _ => (),
                    }
                }
                if valency == 5 && bond_count == 3 {
                    for bond in bonds {
                        let neighbor_index = bond.target();
                        let bond_in_ring = if atom_index < neighbor_index {
                            bonds_in_rings.contains(&(atom_index, neighbor_index))
                        } else {
                            bonds_in_rings.contains(&(neighbor_index, atom_index))
                        };
                        if !bond_in_ring && self.get_atomic_number(bond.target()) == 8 {
                            return 24;
                        }
                    }
                }
            }
            -1 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 && bond_count == 2 {
                    return 26;
                }

                if valency == 2 {
                    return 26;
                }
            }
            1 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 && bond_count == 3 {
                    return 27;
                }

                if valency != 4 && bond_count != 3 {
                    return 0;
                }

                for bond in bonds {
                    let neighbor_index = bond.target();
                    let bond_in_ring = if atom_index < neighbor_index {
                        bonds_in_rings.contains(&(atom_index, neighbor_index))
                    } else {
                        bonds_in_rings.contains(&(neighbor_index, atom_index))
                    };
                    if !bond_in_ring
                        && self.get_atomic_number(bond.target()) == 8
                        && bond.bond_order() == BondOrder::Double
                    {
                        return 25;
                    }
                }
                return 27;
            }
            _ => (),
        }

        lookup_index
    }

    fn handle_sulfur_aromaticity(
        &self,
        atom_index: usize,
        bonds_in_rings: &FxHashSet<(usize, usize)>,
    ) -> u8 {
        let formal_charge = self.get_atom_charge(atom_index);
        let valency = self.actual_valency(atom_index);
        let mut lookup_index = 0;

        let bonds = self.get_atom_bonds(atom_index).unwrap_or_default();
        let _bond_count = bonds.len();

        match formal_charge {
            0 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 && bond_count == 2 {
                    return 18;
                }

                if valency != 2 {
                    if (bond_count == 3 && valency == 4)
                        || (bond_count == 3 && number_of_aromatic_bonds == 2)
                    {
                        for bond in bonds {
                            let bond_in_ring = Self::bond_hashset_contains(
                                bonds_in_rings,
                                atom_index,
                                bond.target(),
                            );
                            if bond.bond_order() == BondOrder::Double
                                && !bond_in_ring
                                && self.get_atomic_number(bond.target()) == 8
                            {
                                return 20;
                            }
                        }
                    }
                }
                // Everything a neutral sulfur can offer a ring is now spoken
                // for: two neighbours leave it a lone pair to donate, and the
                // sulfoxide above donates the pair its exocyclic `=O` has not
                // taken. This used to fall through to the thiophene answer for
                // any sulfur at all, which made a ring sulfone aromatic -- the
                // sulfur of a sultam has four σ bonds, no lone pair and no
                // p orbital, and is exactly what breaks such a ring's
                // conjugation. A three-coordinate neutral sulfur has likewise
                // spent one pair on its third bond, and giving up the second
                // would leave it positive, which is the case the `+1` arm below
                // handles and why that arm exists at all.
                //
                // Two neighbours is the test, not two bonds' worth of valence:
                // the sulfur of 1,2,5-thiadiazole is written `N=S=N` as often
                // as `n-s-n`, and both are the same aromatic ring.
                if bond_count == 2 {
                    lookup_index = 18;
                }
            }
            1 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 && bond_count == 2 {
                    return 19;
                }

                if valency == 3 && bond_count == 2 {
                    return 19;
                }

                if (valency == 3 && bond_count == 3)
                    || (number_of_aromatic_bonds == 2 && bond_count == 3)
                {
                    for bond in bonds {
                        let atom_order = order_tuple(atom_index, bond.target());
                        let bond_in_ring = bonds_in_rings.contains(&atom_order);
                        if !bond_in_ring
                            && self.get_atomic_number(bond.target()) == 8
                            && self.get_atom_charge(bond.target()) == -1
                        {
                            return 21;
                        }
                    }
                    // Tricoordinate sulfonium (e.g. a fused-ring S+ with an
                    // exocyclic substituent) with no exocyclic S-O(-): S+
                    // still has one lone pair left after its three sigma
                    // bonds, which donates 2 electrons to the ring
                    // pi-system (like the S-O(-) case above, and like
                    // pyrrole-type nitrogen) regardless of whether the
                    // third substituent happens to be oxygen.
                    lookup_index = 21;
                }
            }

            _ => (),
        }
        lookup_index
    }

    fn handle_arsenic_aromaticity(&self, atom_index: usize) -> u8 {
        let formal_charge = self.get_atom_charge(atom_index);
        let lookup_index = 0;
        let valency = self.actual_valency(atom_index);

        match formal_charge {
            0 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let _bond_count = bonds.len();

                let mut number_of_aromatic_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }
                let bonds = self.get_atom_bonds(atom_index).unwrap_or_default();
                let bond_count = bonds.len();
                if number_of_aromatic_bonds == 2 && bond_count == 3 {
                    return 32;
                }

                if valency == 3 && bond_count == 3 {
                    return 32;
                }
            }
            1 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();

                let mut number_of_aromatic_bonds = 0;
                let mut number_of_single_bonds = 0;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                    if bond.bond_order() == BondOrder::Single {
                        number_of_single_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 && bond_count == 3 {
                    if number_of_single_bonds == 1 {
                        return 33;
                    }
                }

                if valency == 4 && bond_count == 3 {
                    return 33;
                }
            }
            _ => {
                return 0;
            }
        }
        lookup_index
    }

    fn handle_selenium_aromaticity(
        &self,
        atom_index: usize,
        bonds_in_rings: &FxHashSet<(usize, usize)>,
    ) -> u8 {
        let formal_charge = self.get_atom_charge(atom_index);
        let valency = self.actual_valency(atom_index);
        let lookup_index = 0;

        match formal_charge {
            0 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;
                let _number_of_single_bonds = 0;
                let mut has_oxygen_double_bond = false;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                    if bond.bond_order() == BondOrder::Double
                        && self.get_atomic_number(bond.target()) == 8
                    {
                        has_oxygen_double_bond = true;
                    }
                }

                if (number_of_aromatic_bonds == 2 && bond_count == 2)
                    || (number_of_aromatic_bonds == 2 && has_oxygen_double_bond)
                {
                    return 28;
                }
                if valency == 2 && bond_count == 2 {
                    return 28;
                }
                // A selenoxide written in Kekule form: three neighbours, one
                // of them an exocyclic `=O`. Daylight gives `[Se](=O)` the
                // two electrons it gives the sulfoxide `S(=O)` -- the lone
                // pair the oxygen has not taken -- and the sulfur handler has
                // always followed that; this one only recognised the oxide
                // when its ring bonds already arrived aromatic, so
                // selenophene oxide was the one Daylight entry this model
                // did not reproduce.
                if bond_count == 3
                    && valency == 4
                    && self.has_exocyclic_oxygen(atom_index, bonds_in_rings, BondOrder::Double, 0)
                {
                    return 28;
                }
            }
            1 => {
                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    return 0;
                };
                let bond_count = bonds.len();
                let mut number_of_aromatic_bonds = 0;
                let _number_of_single_bonds = 0;
                let _has_oxygen_double_bond = false;

                for bond in bonds.iter() {
                    if bond.bond_order() == BondOrder::Aromatic {
                        number_of_aromatic_bonds += 1;
                    }
                }

                if number_of_aromatic_bonds == 2 && bond_count == 2 {
                    return 29;
                }

                // Not 100% sure about this one
                if valency == 3 && bond_count == 2 {
                    return 29;
                }
                // The same selenoxide charge-separated, `[Se+]([O-])`, which
                // Daylight also counts as two electrons.
                if bond_count == 3
                    && valency == 3
                    && self.has_exocyclic_oxygen(atom_index, bonds_in_rings, BondOrder::Single, -1)
                {
                    return 28;
                }
            }
            _ => (),
        }
        lookup_index
    }

    fn handle_tellurium_aromaticity(&self, atom_index: usize) -> u8 {
        let formal_charge = self.get_atom_charge(atom_index);
        let valency = self.actual_valency(atom_index);
        let Some(bonds) = self.get_atom_bonds(atom_index) else {
            return 0;
        };
        let bond_count = bonds.len();

        let mut number_of_aromatic_bonds = 0;
        for bond in bonds.iter() {
            if bond.bond_order() == BondOrder::Aromatic {
                number_of_aromatic_bonds += 1;
            }
        }

        if formal_charge == 0 {
            if number_of_aromatic_bonds == 2 && bond_count == 2 {
                return 30;
            }
            if valency == 2 && bond_count == 2 {
                return 30;
            }
        }
        if formal_charge == 1 {
            if number_of_aromatic_bonds == 2 && bond_count == 2 {
                return 31;
            }
            if valency == 3 && bond_count == 2 {
                return 31;
            }
        }
        0
    }

    /// Whether `atom_index` carries an oxygen outside every ring, by a bond of
    /// `order`, with `charge` on the oxygen.
    fn has_exocyclic_oxygen(
        &self,
        atom_index: usize,
        bonds_in_rings: &FxHashSet<(usize, usize)>,
        order: BondOrder,
        charge: i8,
    ) -> bool {
        self.get_atom_bonds(atom_index)
            .unwrap_or_default()
            .iter()
            .any(|bond| {
                bond.bond_order() == order
                    && self.get_atomic_number(bond.target()) == 8
                    && self.get_atom_charge(bond.target()) == charge
                    && !Self::bond_hashset_contains(bonds_in_rings, atom_index, bond.target())
            })
    }

    fn bond_hashset_contains(
        bonds_in_rings: &FxHashSet<(usize, usize)>,
        atom1_index: usize,
        atom2_index: usize,
    ) -> bool {
        if atom1_index < atom2_index {
            bonds_in_rings.contains(&(atom1_index, atom2_index))
        } else {
            bonds_in_rings.contains(&(atom2_index, atom1_index))
        }
    }

    fn check_short_aromatic_path(
        &self,
        atom1_index: usize,
        atom2_index: usize,
        last_bond: (usize, usize),
        aromatic_types: &mut [AromaticityFlag],
        visited: &mut [u8],
        bonds_in_rings: &FxHashSet<(usize, usize)>,
        aromatic_atoms: &mut [bool],
        aromatic_bonds: &mut FxHashSet<(usize, usize)>,
        electron_count: usize,
        saw_a_pi_bond: bool,
        remaining_path_len: usize,
    ) -> bool {
        if atom1_index == atom2_index {
            // Hückel's count is necessary and not sufficient: a ring where
            // every atom brings a lone pair and none brings half of a π bond
            // has no π system to delocalize, and no Kekulé structure either.
            // `O1S[Te+]1` is three such atoms and six electrons, which passes
            // `4n + 2` and is not aromatic by any account -- RDKit reads it as
            // three single bonds. Perceiving it aromatic made the writer emit
            // `[te]1s[o+]1`, which this crate's own reader then refused for
            // want of a Kekulé structure, so a molecule came out unreadable.
            return electron_count & 3 == 2 && saw_a_pi_bond;
        }

        if remaining_path_len == 0
            || visited[atom2_index] == 3
            || aromatic_types[atom2_index] == AromaticityFlag::NotAromatic
        {
            return false;
        }

        let previous_visit_state = visited[atom2_index];
        visited[atom2_index] = 3;

        let current_atom_electrons = aromatic_types[atom2_index] as usize;

        let Some(bonds) = self.get_atom_bonds(atom2_index) else {
            visited[atom2_index] = previous_visit_state;
            return false;
        };

        let mut found_aromatic_path = false;

        for &neighbor_bond in bonds {
            let neighbor_index = neighbor_bond.target();
            let atom_order = order_tuple(atom2_index, neighbor_index);
            let bond_in_ring = bonds_in_rings.contains(&atom_order);
            if last_bond == atom_order || !bond_in_ring {
                continue;
            }

            if self.check_short_aromatic_path(
                atom1_index,
                neighbor_index,
                atom_order,
                aromatic_types,
                visited,
                bonds_in_rings,
                aromatic_atoms,
                aromatic_bonds,
                current_atom_electrons + electron_count,
                saw_a_pi_bond
                    || aromatic_types[atom2_index]
                        == AromaticityFlag::PotentiallyAromaticOneElectron,
                remaining_path_len - 1,
            ) {
                let atom_order = order_tuple(atom2_index, neighbor_index);
                aromatic_bonds.insert(atom_order);
                found_aromatic_path = true;
            }
        }

        visited[atom2_index] = previous_visit_state;

        if found_aromatic_path {
            aromatic_atoms[atom2_index] = true;
        }
        found_aromatic_path
    }
    /// Detects and marks aromatic systems in a molecule according to a specified aromaticity model.
    ///
    /// This implementation follows the approach described in [reference to paper/algorithm],
    /// which involves:
    /// 1. Identifying cyclic structures in the molecule
    /// 2. Determining potential aromatic atoms based on element type and bonding
    /// 3. Finding aromatic cycles that satisfy Hückel's rule (4n+2 π electrons)
    /// 4. Marking bonds in aromatic systems as aromatic
    ///
    /// # Arguments
    /// * `aromaticity_model` - The model to use for determining aromaticity (e.g., Daylight, OpenSMILES)
    /// * `path_length_limit` - Optional limit on path length for cycle detection
    /// * `strict_flag` - Whether to use stricter criteria for aromaticity
    ///
    /// # Returns
    /// `Ok(())` if aromatization was successful, or an error if something went wrong
    ///
    /// # Examples
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::aromaticity::AromaticityModel;
    ///
    ///
    /// let mut mol = MolecularGraph::from_smiles("C=1C=CC=CC=1").unwrap();
    /// let mut mol = mol.first().unwrap().clone();
    /// mol.aromatize(AromaticityModel::Daylight, None, false).unwrap();
    /// assert!(mol.get_aromatic_atoms().len() == 6);
    /// ```
    fn aromatize(
        &mut self,
        aromaticity_model: AromaticityModel,
        path_length_limit: Option<usize>,
        strict_flag: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Borrowed from the thread's own [`Aromatizer`] so that perceiving one
        // molecule after another reuses the buffers instead of allocating a
        // fresh set for each. Taken out and put back rather than held borrowed,
        // so that anything reachable from here which itself perceives
        // aromaticity gets its own scratch instead of a panic.
        let mut aromatizer = AROMATIZER.take();
        let result = aromatizer.aromatize(self, aromaticity_model, path_length_limit, strict_flag);
        AROMATIZER.set(aromatizer);
        result
    }

    /// [`MoleculeAromaticity::aromatize`] against buffers the caller owns.
    ///
    /// Perception allocates a vector per atom, two hash sets of bonds and a
    /// ring-membership set every time it runs, and it runs once per molecule
    /// parsed. Handing it a scratch that outlives the molecule turns that into
    /// a `clear` and a `resize`.
    fn aromatize_with(
        &mut self,
        scratch: &mut AromaticityScratch,
        aromaticity_model: AromaticityModel,
        path_length_limit: Option<usize>,
        strict_flag: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let num_atoms = self.len();

        let AromaticityScratch {
            ring_finder,
            pi_capable,
            atoms_in_rings_placeholder,
            bonds_in_rings_placeholder,
            aromaticity_flags,
            aromatic_atoms,
            aromatic_bonds,
            visited,
        } = scratch;
        // Before any ring is looked for: an aromatic ring is three or more
        // atoms that can each put an orbital into it, and each of them needs
        // two neighbours to have a ring at all. Counting those is one pass over
        // the atoms and no ring perception, so a molecule whose rings are all
        // saturated -- a sugar, a piperidine, a steroid's aliphatic half --
        // never pays for the walk that would find them.
        //
        // The test is the one `pi_system::is_pi_capable` applies, so it can
        // only exclude atoms that perception would have excluded later: a
        // neutral CH2 with four sigma bonds and no π demand, a quaternary
        // ammonium, an sp3 oxygen. It is skipped when the input wrote aromatic
        // bonds, because those still have to be localized below whatever the
        // atoms turn out to be.
        pi_capable.clear();
        pi_capable.resize(num_atoms, false);
        for atom in 0..num_atoms {
            let demand = crate::molecule::pi_system::pi_demand(self, atom).unwrap_or(0);
            pi_capable[atom] = crate::molecule::pi_system::is_pi_capable(self, atom, demand);
        }

        let wrote_aromatic_bonds = (0..num_atoms).any(|atom| {
            self.get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .any(|bond| bond.bond_order() == BondOrder::Aromatic)
        });
        if !wrote_aromatic_bonds {
            // An atom of an aromatic ring has two neighbours on that ring, and
            // they are π-capable too, so what has to exist is three atoms that
            // are π-capable and have two π-capable neighbours. Fewer than that
            // and no ring can be aromatic, whatever the rings turn out to be.
            let mut capable = 0;
            for atom in 0..num_atoms {
                if !pi_capable[atom] {
                    continue;
                }
                let neighbours = self
                    .get_atom_bonds(atom)
                    .unwrap_or_default()
                    .iter()
                    .filter(|bond| pi_capable.get(bond.target()).copied().unwrap_or(false))
                    .count();
                if neighbours >= 2 {
                    capable += 1;
                    if capable == 3 {
                        break;
                    }
                }
            }
            if capable < 3 {
                return Ok(());
            }
        }

        self.find_cyclic_bonds_and_atoms_into(
            atoms_in_rings_placeholder,
            bonds_in_rings_placeholder,
        );

        // If there are no cycles, we can return early
        if atoms_in_rings_placeholder.is_empty() {
            return Ok(());
        }

        // Perceive from a Kekulé structure, always.
        //
        // Everything below reads bond orders -- how many double bonds an atom
        // has, how many aromatic ones -- so it answers differently for the two
        // ways of writing one compound. What hid that was the write-back at the
        // end only ever *adding* aromatic bonds: a bond the input wrote
        // aromatic and perception rejected kept its order, so a lower-case
        // input silently retained rings perception had just refused, and the
        // two writings agreed by repair rather than by agreement. That repair
        // held only while perception was permissive enough to accept whatever
        // the input asserted. Localizing first removes the difference instead
        // of patching it.
        // Perception itself stays tolerant of a molecule that cannot be
        // localized: the offending bonds keep the order they were written with,
        // and everything around them is perceived as usual. Rejecting such a
        // molecule is the reader's job, not perception's -- see
        // `SmilesParseOptions`.
        let _ = self.localize_aromatic_bonds();

        let atoms_in_cycles =
            (0..num_atoms).filter(|index| atoms_in_rings_placeholder.contains(*index));
        aromaticity_flags.clear();
        aromaticity_flags.resize(num_atoms, AromaticityFlag::NotAromatic);

        let model_kind = aromaticity_model;
        let aromaticity_model = match aromaticity_model {
            AromaticityModel::Daylight => &AROMATICITY_MODEL_DAYLIGHT,
            AromaticityModel::OpenSMILES | AromaticityModel::Rings => &AROMATICITY_MODEL_OPENSMILES,
        };

        aromatic_atoms.clear();
        aromatic_atoms.resize(self.len(), false);
        aromatic_bonds.clear();

        // Classification asks whether an atom's double bond leaves the ring,
        // and for that "ring" has to mean a ring that could carry a π system
        // rather than any cycle at all -- see [`conjugated_ring_bonds`], whose
        // documentation carries the pair of molecules that separates the two.
        let conjugated_bonds = crate::molecule::pi_system::conjugated_ring_bonds_with(
            self,
            bonds_in_rings_placeholder,
            pi_capable,
        );

        for index in atoms_in_cycles {
            if self.is_atom_aromatic(index) {
                aromatic_atoms[index] = true;
            }
            let aromaticity_flag =
                self.handle_aromatic_atom(index, aromaticity_model, &conjugated_bonds);

            aromaticity_flags[index] = aromaticity_flag;
        }

        // This has to be checked
        if aromaticity_flags
            .iter()
            .all(|flag| *flag == AromaticityFlag::NotAromatic)
        {
            return Ok(());
        }

        self.remove_potential_aromatics(aromaticity_flags, bonds_in_rings_placeholder);

        // This has to be checked
        if aromaticity_flags
            .iter()
            .all(|flag| *flag == AromaticityFlag::NotAromatic)
        {
            return Ok(());
        }

        // Basically we want to be able to limit how far we can travel
        let path_length = path_length_limit.unwrap_or(AROMATIC_PATH_DEPTH);

        // let mut reduced_path_length = path_length;
        // if 31 < reduced_path_length {
        //     // reduced_path_length = self.reduce_path_length(
        //     //     path_length,
        //     //     atoms_in_rings_placeholder,
        //     //     bonds_in_rings_placeholder,
        //     // )
        // }

        self.find_the_aromatic_system(
            model_kind,
            ring_finder,
            atoms_in_rings_placeholder,
            bonds_in_rings_placeholder,
            &conjugated_bonds,
            path_length,
            aromaticity_flags,
            aromatic_atoms,
            aromatic_bonds,
            visited,
        );

        if strict_flag {
            for atom_index in 0..self.len() {
                let mut has_double_bond = false;
                if !aromatic_atoms[atom_index] {
                    aromaticity_flags[atom_index] = AromaticityFlag::NotAromatic;
                } else {
                    let bonds = self.get_atom_bonds(atom_index).unwrap_or_default();
                    for bond in bonds {
                        let neighbor_atom_index = bond.target();
                        let atom_order = order_tuple(atom_index, neighbor_atom_index);
                        if aromatic_bonds.contains(&atom_order)
                            || bond.bond_order() == BondOrder::Double
                        {
                            aromaticity_flags[atom_index] = AromaticityFlag::NotAromatic;
                            has_double_bond = true;
                            break;
                        }
                    }
                }

                if has_double_bond {
                    self.remove_potential_aromatics(aromaticity_flags, bonds_in_rings_placeholder);
                    self.find_the_aromatic_system(
                        model_kind,
                        ring_finder,
                        atoms_in_rings_placeholder,
                        bonds_in_rings_placeholder,
                        &conjugated_bonds,
                        path_length,
                        aromaticity_flags,
                        aromatic_atoms,
                        aromatic_bonds,
                        visited,
                    )
                }
            }
        }

        // Only write aromatic what can be read back. Perception counts π
        // electrons ring by ring and can land on a ring no assignment of double
        // bonds reproduces -- `C1(=O)SN=S=C1` counts six over its five atoms,
        // but the two demands it leaves sit either side of a sulfur that has
        // none, so nothing can pair them. Written all in lower case such a ring
        // is a string this crate's own reader refuses, and refuses rightly.
        // Perception has to decline it instead.
        // The molecule reaching here is in Kekulé form, so the double bonds it
        // already carries are a candidate answer: when they leave nobody short,
        // they are a Kekulé structure for the set and the matching would only
        // confirm it. That witness holds for all but a handful of molecules in
        // a hundred thousand, and testing it is one pass over the set.
        if !crate::molecule::pi_system::aromatic_set_is_satisfied(self, aromatic_bonds) {
            let kekulization = crate::molecule::pi_system::kekulize(self, aromatic_bonds);
            if !kekulization.unsatisfied.is_empty() {
                aromatic_bonds.retain(|&(first, _)| !kekulization.is_atom_deficient(first));
            }
        }

        for atom_index in 0..self.len() {
            let bonds = self.get_atom_bonds_mut(atom_index).unwrap_or_default();
            for bond in bonds.iter_mut() {
                let neighbor_atom_index = bond.target();
                let atom_order = order_tuple(atom_index, neighbor_atom_index);
                if aromatic_bonds.contains(&atom_order) {
                    bond.bond_order = BondOrder::Aromatic;
                }
            }
        }

        // A cis/trans tag left on an atom that no longer has a double bond
        // asserts a geometry about a bond perception has just read as aromatic,
        // and whether it is there at all depends on how the input was written.
        // `C1=CC=C(C=C1)/C=C/C` marks the ring carbon, because in the Kekule
        // form it carries a double bond next to a directional one; the same
        // compound written `c1ccccc1/C=C/C` does not. The tag is part of the
        // atom's canonization invariant, so the two writings were coloured
        // differently and canonicalized to different strings.
        let stale: Vec<usize> = (0..self.len())
            .filter(|&atom_index| {
                self.chiral_classes().is_some_and(|classes| {
                    matches!(
                        classes.get(atom_index),
                        Some(
                            ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
                        )
                    )
                }) && !self
                    .get_atom_bonds(atom_index)
                    .unwrap_or_default()
                    .iter()
                    .any(|bond| bond.bond_order() == BondOrder::Double)
            })
            .collect();
        if !stale.is_empty()
            && let Some(classes) = self.chiral_classes_mut()
        {
            for atom_index in stale {
                classes[atom_index] = ChiralClass::None;
            }
        }

        Ok(())
    }

    /// Rewrites every bond the input wrote aromatic as a single or a double
    /// one, so that perception is handed a Kekulé structure whatever it was
    /// given.
    ///
    /// Which bonds become double is decided by counting rather than by reading:
    /// each atom's π demand is what its element, charge and σ-connectivity
    /// require, less the π bonds already committed on the bonds that were *not*
    /// written aromatic. Satisfying as many of the remaining demands as
    /// possible over the aromatic bonds is a maximum matching, and it has to be
    /// an exact one -- a greedy pairing leaves atoms unpaired that could have
    /// been paired, and an unpaired demand is an unpaired electron, so a greedy
    /// answer invents radicals. Matched bonds become double, the rest single.
    ///
    /// A molecule handed over already in Kekulé form has no aromatic bond to
    /// rewrite and passes through untouched.
    ///
    /// # Errors
    ///
    /// [`KekulizationError`] when a connected run of aromatic bonds states more
    /// π demand than it can satisfy -- `O=c1ccccc1` is five carbons each wanting
    /// a double bond, and one of them must go without. Such a molecule has no
    /// Kekulé structure and is not a molecule; the bonds are left as they were
    /// found rather than localized, since which atom ends up short would
    /// otherwise depend on the order the atoms happen to be numbered in.
    fn localize_aromatic_bonds(&mut self) -> Result<(), KekulizationError> {
        let atom_count = self.len();

        // Perception calls this on every molecule and the parser has usually
        // localized it already, so the common case is that there is nothing to
        // do. Finding that out must not cost a hash set.
        let has_aromatic_bond = (0..atom_count).any(|atom| {
            self.get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .any(|bond| bond.bond_order() == BondOrder::Aromatic)
        });
        if !has_aromatic_bond {
            return Ok(());
        }

        let mut aromatic: FxHashSet<(usize, usize)> = FxHashSet::default();
        for atom in 0..atom_count {
            let Some(bonds) = self.get_atom_bonds(atom) else {
                continue;
            };
            for bond in bonds {
                let target = bond.target();
                if atom < target && bond.bond_order() == BondOrder::Aromatic {
                    aromatic.insert((atom, target));
                }
            }
        }

        let kekulization = crate::molecule::pi_system::kekulize(self, &aromatic);

        for atom_index in 0..atom_count {
            // A component with no Kekulé structure is left exactly as it was
            // found, and the rest of the molecule is localized around it.
            if kekulization.is_atom_deficient(atom_index) {
                continue;
            }
            let bonds = self.get_atom_bonds_mut(atom_index).unwrap_or_default();
            for bond in bonds.iter_mut() {
                let atom_order = order_tuple(atom_index, bond.target());
                if !aromatic.contains(&atom_order) {
                    continue;
                }
                let paired = kekulization
                    .matched
                    .get(atom_order.0)
                    .and_then(|partner| *partner)
                    .is_some_and(|partner| partner == atom_order.1);
                bond.bond_order = if paired {
                    BondOrder::Double
                } else {
                    BondOrder::Single
                };
            }
        }

        if kekulization.unsatisfied.is_empty() {
            Ok(())
        } else {
            Err(KekulizationError {
                atoms: kekulization.unsatisfied,
            })
        }
    }

    fn aromatize_opensmiles(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.aromatize(AromaticityModel::OpenSMILES, None, false)
    }

    /// Marks the aromatic atoms and bonds, by whichever search the model asks
    /// for.
    ///
    /// The two searches are handed the same flags, the same π-capable bond set
    /// and the same output buffers, and differ only in the question they ask of
    /// them -- which is what makes them comparable over a corpus.
    #[allow(clippy::too_many_arguments)]
    fn find_the_aromatic_system(
        &self,
        model: AromaticityModel,
        ring_finder: &mut crate::molecule::ring_aromaticity::RingFinder,
        atoms_in_rings: &BitSet,
        bonds_in_rings: &FxHashSet<(usize, usize)>,
        conjugated_bonds: &FxHashSet<(usize, usize)>,
        max_path_length: usize,
        aromaticity_flags: &mut [AromaticityFlag],
        aromatic_atoms: &mut [bool],
        aromatic_bonds: &mut FxHashSet<(usize, usize)>,
        visited: &mut Vec<u8>,
    ) {
        match model {
            AromaticityModel::Rings => ring_finder.find_aromatic_rings(
                self,
                aromaticity_flags,
                bonds_in_rings,
                conjugated_bonds,
                aromatic_atoms,
                aromatic_bonds,
            ),
            AromaticityModel::Daylight | AromaticityModel::OpenSMILES => self.find_aromatic_cycles(
                atoms_in_rings,
                bonds_in_rings,
                max_path_length,
                aromaticity_flags,
                aromatic_atoms,
                aromatic_bonds,
                visited,
            ),
        }
    }

    fn find_aromatic_cycles(
        &self,
        _atoms_in_rings: &BitSet,
        bonds_in_rings: &FxHashSet<(usize, usize)>,
        max_path_length: usize,
        aromaticity_flags: &mut [AromaticityFlag],
        aromatic_atoms: &mut [bool],
        aromatic_bonds: &mut FxHashSet<(usize, usize)>,
        visited: &mut Vec<u8>,
    ) {
        visited.clear();
        visited.resize(self.len(), 0);
        LONG_PATH_STEPS.with(|steps| steps.set(LONG_PATH_BUDGET));

        for atom_index in 0..self.len() {
            if visited[atom_index] == 0 {
                self.find_aromatic_seed_atoms(
                    atom_index,
                    None,
                    aromaticity_flags,
                    bonds_in_rings,
                    visited,
                )
            }
        }

        // #[cfg(debug_assertions)]
        // println!("Root atoms after initial check: {:?}", visited);

        // Find 5 membered rings
        for atom_index in 0..self.len() {
            if visited[atom_index] != 2 {
                continue;
            }
            let electron_count = aromaticity_flags[atom_index] as usize;

            let bonds = self.get_atom_bonds(atom_index).unwrap_or_default();
            for bond in bonds {
                let neighbor_index = bond.target();
                let atom_order = order_tuple(atom_index, neighbor_index);
                let bond_in_ring = bonds_in_rings.contains(&atom_order);

                if !bond_in_ring {
                    continue;
                }

                if self.check_short_aromatic_path(
                    atom_index,
                    neighbor_index,
                    atom_order,
                    aromaticity_flags,
                    visited,
                    bonds_in_rings,
                    aromatic_atoms,
                    aromatic_bonds,
                    electron_count,
                    aromaticity_flags[atom_index]
                        == AromaticityFlag::PotentiallyAromaticOneElectron,
                    5,
                ) {
                    aromatic_bonds.insert((atom_index, neighbor_index));
                    aromatic_atoms[atom_index] = true;
                }
            }
        }

        // No `#[cfg(debug_assertions)]` above this line. Three of them used to
        // sit here, left behind when the debug `println!`s they belonged to were
        // commented out -- and an attribute does not care about the blank line
        // between: it attaches to the next statement, which is this whole block.
        // Release builds turn debug assertions off, so the long-path search,
        // `identify_extra_aromatic_seed_atoms`, and the step budget above were
        // compiled out of every release build the crate has ever shipped.
        if max_path_length > 10 {
            loop {
                if !self.identify_extra_aromatic_seed_atoms(
                    aromaticity_flags,
                    visited,
                    aromatic_atoms,
                    aromatic_bonds,
                    bonds_in_rings,
                ) {
                    break;
                }

                //#[cfg(debug_assertions)]
                //println!("Identified additional root atoms: {:?}", visited);

                let mut new_aromatic = false;

                for atom_index in 0..self.len() {
                    // If additional root atom
                    if visited[atom_index] == 3 {
                        let Some(bonds) = self.get_atom_bonds(atom_index) else {
                            continue;
                        };

                        for bond in bonds {
                            let neighbor_index = bond.target();

                            let atom_order = order_tuple(atom_index, neighbor_index);
                            let is_bond_in_ring = bonds_in_rings.contains(&atom_order);
                            let electron_count = aromaticity_flags[atom_index] as usize;
                            //println!("Electrons for atom: {} {}", electron_count, atom_index);

                            if is_bond_in_ring && !aromatic_bonds.contains(&atom_order) {
                                // #[cfg(debug_assertions)]
                                // println!("Checking bond: {:?}", atom_order);
                                new_aromatic = self.check_long_aromatic_path(
                                    atom_index,
                                    neighbor_index,
                                    atom_order,
                                    aromaticity_flags,
                                    visited,
                                    aromatic_atoms,
                                    aromatic_bonds,
                                    bonds_in_rings,
                                    electron_count,
                                    aromaticity_flags[atom_index]
                                        == AromaticityFlag::PotentiallyAromaticOneElectron,
                                    9,
                                );

                                // #[cfg(debug_assertions)]
                                // println!("New aromatic: {}", new_aromatic);

                                if new_aromatic {
                                    aromatic_bonds.insert(atom_order);
                                }
                            }
                        }
                        if new_aromatic {
                            aromatic_atoms[atom_index] = true
                        }
                        visited[atom_index] = 2;
                    }
                }

                if !new_aromatic {
                    break;
                }
            }
        }

        if max_path_length > 6 {
            loop {
                if !self.identify_extra_aromatic_seed_atoms(
                    aromaticity_flags,
                    visited,
                    aromatic_atoms,
                    aromatic_bonds,
                    bonds_in_rings,
                ) {
                    break;
                }

                // #[cfg(debug_assertions)]
                // println!("Identified additional root atoms: {:?}", visited);

                let mut new_aromatic = false;
                for atom_index in 0..self.len() {
                    // If additional root atom
                    if visited[atom_index] == 3 {
                        let bonds = self.get_atom_bonds(atom_index).unwrap_or_default();

                        for bond in bonds {
                            let neighbor_index = bond.target();
                            let atom_order = order_tuple(atom_index, neighbor_index);

                            let is_bond_in_ring = bonds_in_rings.contains(&atom_order);
                            let electron_count = aromaticity_flags[atom_index] as usize;
                            let is_not_already_aromatic = !aromatic_bonds.contains(&atom_order);

                            if is_bond_in_ring && is_not_already_aromatic {
                                // #[cfg(debug_assertions)]
                                // println!("Checking bond: {:?}", atom_order);
                                new_aromatic = self.check_long_aromatic_path(
                                    atom_index,
                                    neighbor_index,
                                    atom_order,
                                    aromaticity_flags,
                                    visited,
                                    aromatic_atoms,
                                    aromatic_bonds,
                                    bonds_in_rings,
                                    electron_count,
                                    aromaticity_flags[atom_index]
                                        == AromaticityFlag::PotentiallyAromaticOneElectron,
                                    max_path_length - 1,
                                );

                                if new_aromatic {
                                    // #[cfg(debug_assertions)]
                                    // println!("New aromatic: {}", new_aromatic);
                                    aromatic_bonds.insert(atom_order);
                                }
                            }
                        }
                        if new_aromatic {
                            aromatic_atoms[atom_index] = true
                        }
                        visited[atom_index] = 2;
                    }
                }

                if !new_aromatic {
                    break;
                }
            }
        }

        if std::env::var_os("OXIDANE_BUDGET_TRACE").is_some() {
            let left = LONG_PATH_STEPS.with(|steps| steps.get());
            eprintln!(
                "BUDGET used {} of {LONG_PATH_BUDGET}{}",
                LONG_PATH_BUDGET - left,
                if left == 0 { " (EXHAUSTED)" } else { "" }
            );
        }
    }

    fn check_long_aromatic_path(
        &self,
        start_atom_index: usize,
        current_atom_index: usize,
        previous_bond: (usize, usize),
        aromaticity_flags: &mut [AromaticityFlag],
        visited: &mut [u8],
        aromatic_atoms: &mut [bool],
        aromatic_bonds: &mut FxHashSet<(usize, usize)>,
        bonds_in_rings: &FxHashSet<(usize, usize)>,
        electron_count: usize,
        saw_a_pi_bond: bool,
        remaining_path_len: usize,
    ) -> bool {
        if start_atom_index == current_atom_index {
            // See `check_short_aromatic_path`: the count alone would call a
            // ring of lone pairs aromatic.
            return electron_count & 3 == 2 && saw_a_pi_bond;
        }

        if !spend_a_long_path_step() {
            return false;
        }

        if remaining_path_len == 0
            || aromaticity_flags[current_atom_index] == AromaticityFlag::NotAromatic
            || visited[current_atom_index] == 4
        {
            return false;
        }

        let previous_visit_state = visited[current_atom_index];
        visited[current_atom_index] = 4; // Temporary mark

        let new_electron_count = electron_count + aromaticity_flags[current_atom_index] as usize;
        let saw_a_pi_bond = saw_a_pi_bond
            || aromaticity_flags[current_atom_index]
                == AromaticityFlag::PotentiallyAromaticOneElectron;

        let Some(bonds) = self.get_atom_bonds(current_atom_index) else {
            visited[current_atom_index] = previous_visit_state;
            return false;
        };

        for bond in bonds {
            let neighbor_index = bond.target();
            let order = order_tuple(current_atom_index, neighbor_index);
            if order == previous_bond || !bonds_in_rings.contains(&order) {
                continue;
            }

            // #[cfg(debug_assertions)]
            // println!("Checking bond in long path: {:?}", order);
            // #[cfg(debug_assertions)]
            // println!("Electron count: {}", new_electron_count);
            if self.check_long_aromatic_path(
                start_atom_index,
                neighbor_index,
                order,
                aromaticity_flags,
                visited,
                aromatic_atoms,
                aromatic_bonds,
                bonds_in_rings,
                new_electron_count,
                saw_a_pi_bond,
                remaining_path_len - 1,
            ) {
                aromatic_bonds.insert(order);
                aromatic_atoms[current_atom_index] = true;

                visited[current_atom_index] = previous_visit_state;
                // #[cfg(debug_assertions)]
                // println!("Aromatic bonds {:?}", order);
                return true;
            }
        }

        visited[current_atom_index] = previous_visit_state; // Reset mark
        false
    }

    fn identify_extra_aromatic_seed_atoms(
        &self,
        aromaticity_flags: &mut [AromaticityFlag],
        visited: &mut [u8],
        aromatic_atoms: &mut [bool],
        aromatic_bonds: &mut FxHashSet<(usize, usize)>,
        bonds_in_rings: &FxHashSet<(usize, usize)>,
    ) -> bool {
        let mut found_new = false;
        for atom_index in 0..self.len() {
            let bonds = self.get_atom_bonds(atom_index).unwrap_or_default();
            for bond in bonds {
                let neighbor_index = bond.target();
                let atom_order = order_tuple(atom_index, neighbor_index);
                let is_bond_in_ring = bonds_in_rings.contains(&atom_order);
                let is_bond_aromatic = aromatic_bonds.contains(&atom_order);

                if !is_bond_in_ring
                    || is_bond_aromatic
                    || aromaticity_flags[atom_index] == AromaticityFlag::NotAromatic
                    || aromaticity_flags[neighbor_index] == AromaticityFlag::NotAromatic
                    || visited[atom_index] == 3
                    || visited[atom_index] == 3
                {
                    continue;
                }

                if aromatic_atoms[atom_index]
                    || aromatic_atoms[neighbor_index]
                    || visited[atom_index] == 2
                    || visited[atom_index] == 2
                {
                    if aromatic_atoms[atom_index] {
                        visited[atom_index] = 3;
                        found_new = true;
                    } else if aromatic_atoms[neighbor_index] {
                        visited[neighbor_index] = 3;
                        found_new = true;
                    } else if visited[atom_index] == 2 {
                        visited[atom_index] = 3;
                        found_new = true;
                    } else if visited[neighbor_index] == 2 {
                        visited[neighbor_index] = 3;
                        found_new = true;
                    }
                }
            }
        }
        found_new
    }

    fn find_aromatic_seed_atoms(
        &self,
        atom_index: usize,
        previous_bond: Option<(usize, usize)>,
        aromaticity_flags: &mut [AromaticityFlag],
        bonds_in_rings: &FxHashSet<(usize, usize)>,
        visited: &mut [u8],
    ) {
        visited[atom_index] = 1;

        for bond in self.get_atom_bonds(atom_index).unwrap_or_default() {
            let neighbor_index = bond.target();
            let atom_order = order_tuple(atom_index, neighbor_index);
            let bond_in_ring = bonds_in_rings.contains(&atom_order);

            if Some((atom_index, neighbor_index)) == previous_bond || !bond_in_ring {
                continue;
            }

            let neighbor_atom_index = bond.target();

            if visited[neighbor_atom_index] == 0 {
                self.find_aromatic_seed_atoms(
                    neighbor_index,
                    Some(atom_order),
                    aromaticity_flags,
                    bonds_in_rings,
                    visited,
                );
            } else if visited[neighbor_atom_index] == 1
                && aromaticity_flags[atom_index] != AromaticityFlag::NotAromatic // Atom 1 is
                                                                                 // potentially
                                                                                 // aromatic
                && aromaticity_flags[neighbor_atom_index] != AromaticityFlag::NotAromatic // Atom 2
                                                                                          // is
                                                                                          // potentially
                                                                                          // aromatic
                && visited[atom_index] != 2
            // Not already a root atom
            {
                visited[neighbor_atom_index] = 2; // Mark neighbor as root
            }
        }
    }

    fn reduce_path_length(
        &self,
        path_length_limit: usize,
        atoms_in_rings: &BitSet,
        bonds_in_ring: &FxHashSet<(usize, usize)>,
    ) -> usize {
        // There is probably a better way to sum up the bits
        let number_of_atoms_in_rings = atoms_in_rings.iter().sum::<usize>();

        let result = number_of_atoms_in_rings.min(path_length_limit);

        if result < 50 {
            // #[cfg(debug_assertions)]
            // println!("Reduced path length to {}", result);
            return result;
        }

        let components = self.get_components();

        let number_of_components = components.len();
        // This can surely be done better
        let number_of_bonds: usize = (0..self.len())
            .map(|atom_index| self.get_atom_bonds(atom_index).iter().count())
            .sum();
        let number_of_atoms = self.len();

        let cyclomatic_number = (number_of_bonds - number_of_atoms) + number_of_components;

        // We should also handle graphene
        if cyclomatic_number < 15 {
            result
        } else if cyclomatic_number > 29 {
            if self.is_fullerene_like(atoms_in_rings, bonds_in_ring) {
                // #[cfg(debug_assertions)]
                // println!("Found fullerene-like molecule, reducing path length to 30");
                return 30;
            }
            50
        } else {
            50
        }
    }

    // TODO optimize this function
    fn is_fullerene_like(
        &self,
        atoms_in_rings: &BitSet,
        bonds_in_ring: &FxHashSet<(usize, usize)>,
    ) -> bool {
        let number_of_ring_atoms: usize = atoms_in_rings.iter().sum();
        let number_of_ring_bonds: usize = bonds_in_ring.len();
        let bond_atom_ratio = number_of_ring_bonds as f32 / number_of_ring_atoms as f32;

        if 1.3 <= bond_atom_ratio {
            return true;
        }

        if 1.2 <= bond_atom_ratio {
            let (rings, number_of_rings) =
                self.determine_ring_systems(atoms_in_rings, bonds_in_ring);

            if number_of_rings == 1 {
                return false;
            }

            let mut counter = 1;

            while counter < number_of_rings {
                let ring_atoms = rings
                    .iter()
                    .map(|&assignment| assignment == Some(counter))
                    .collect::<BitVec>();

                let mut number_of_atoms_in_system = 0;
                let mut number_of_bonds_in_system = 0;

                for (index, atom_in_ring) in ring_atoms.iter().enumerate() {
                    if !atom_in_ring {
                        continue;
                    }
                    number_of_atoms_in_system += 1;
                    for bond in self.get_atom_bonds(index).unwrap_or_default() {
                        // Avoid double counting
                        if index > bond.target() {
                            continue;
                        }
                        let is_first_atom_in_ring = *ring_atoms.get(index).unwrap();
                        let is_second_atom_in_ring = *ring_atoms.get(bond.target()).unwrap();
                        if is_first_atom_in_ring && is_second_atom_in_ring {
                            number_of_bonds_in_system += 1;
                        }
                    }
                }

                let bond_atom_ratio =
                    number_of_bonds_in_system as f32 / number_of_atoms_in_system as f32;
                if bond_atom_ratio >= 1.3 {
                    return true;
                }

                counter += 1;
            }
        }

        false
    }

    /// Filters candidate aromatic atoms, retaining only those with at least two
    /// aromatic neighbors in the same ring system.
    ///
    /// # Arguments
    ///
    /// * `aromaticity_flags` - Mutable slice of flags indicating the aromaticity status of each atom
    /// * `bonds_in_rings` - FxHashSet containing pairs of atom indices representing bonds that are part of rings
    ///
    /// # Behaviour
    ///
    /// In benzene every atom stays potentially aromatic; where a double bond
    /// is isolated rather than part of a ring, its atoms are removed.
    fn remove_potential_aromatics(
        &self,
        aromaticity_flags: &mut [AromaticityFlag],
        bonds_in_rings: &FxHashSet<(usize, usize)>,
    ) {
        let mut aromatic_atoms = aromaticity_flags
            .iter()
            .enumerate()
            .filter(|&(_, &flag)| flag != AromaticityFlag::NotAromatic)
            .map(|(index, _)| index)
            .collect::<BitSet>();
        let mut last_aromatic_atoms_length = aromatic_atoms.len();

        loop {
            let mut aromatic_atoms_to_remove = BitSet::new();
            'atoms: for atom_index in aromatic_atoms.iter() {
                if aromaticity_flags[atom_index] == AromaticityFlag::NotAromatic {
                    continue;
                }

                let Some(bonds) = self.get_atom_bonds(atom_index) else {
                    aromaticity_flags[atom_index] = AromaticityFlag::NotAromatic;
                    continue;
                };

                // Count potentially aromatic neighbors that are connected via ring bonds
                let mut aromatic_ring_neighbors = 0;

                for bond in bonds {
                    let neighbor_index = bond.target();
                    let atom_order = order_tuple(atom_index, neighbor_index);

                    if !bonds_in_rings.contains(&atom_order) {
                        continue;
                    }

                    if aromaticity_flags[neighbor_index] != AromaticityFlag::NotAromatic {
                        aromatic_ring_neighbors += 1;
                    }

                    // An atom in an aromatic system must have at least two aromatic neighbors
                    // connected via ring bonds
                    if aromatic_ring_neighbors > 1 {
                        continue 'atoms;
                    }
                }
                aromaticity_flags[atom_index] = AromaticityFlag::NotAromatic;
                aromatic_atoms_to_remove.insert(atom_index);
            }
            for aromatic_atom in aromatic_atoms_to_remove.iter() {
                aromatic_atoms.remove(aromatic_atom);
            }
            if aromatic_atoms.len() == last_aromatic_atoms_length {
                break;
            }
            last_aromatic_atoms_length = aromatic_atoms.len();
        }
    }
}
