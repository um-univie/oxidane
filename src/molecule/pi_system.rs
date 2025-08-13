//! Which cycles of a molecule could carry a π system at all.
//!
//! Aromaticity perception has to decide, for a ring atom, whether a double bond
//! it carries points out of the ring and so has already spent the p-orbital the
//! ring wanted. The rule that decides this asks whether the bond lies in a ring,
//! and reads that off the molecule's whole cyclic-bond set. That is wrong when
//! the bond's only cycle is one no π system can run through:
//!
//! > `C1CCCN=C2SC(=C)C(=O)N2CC1` and `CCCCN=C2SC(=C)C(=O)N2CC` are the same
//! > compound apart from a tether of four saturated carbons joining two ends
//! > remote from the thiazolone ring. The tether closes a macrocycle, so the
//! > imine `C=N` lies on a cycle and is not counted exocyclic, and that carbon
//! > donates an electron instead of none: six π electrons on the five-ring
//! > rather than five. In the open chain the same bond has no cycle, is
//! > counted exocyclic, and the ring is left alone. One writing is perceived
//! > aromatic and the other is not, decided by carbons that contribute
//! > nothing.
//!
//! Neither is aromatic. The ring is a 2-imino-thiazolidin-4-one carrying an
//! exocyclic methylene, and all three of its carbons have a double bond leaving
//! it. RDKit reaches the same answer as the unfixed rule here, and by the same
//! route -- its exocyclic test is `RingInfo::numBondRings`, which is likewise
//! blind to which ring.
//!
//! So the cycles are filtered before the rule sees them. An atom with no
//! p-orbital to offer is dropped, and what survives 2-coring the rest lies on a
//! cycle *of π-capable atoms*. The imine nitrogen loses its saturated
//! neighbour, falls to degree one and is peeled with it, so both writings agree
//! the `C=N` is exocyclic. Benzene, pyrrole, uracil and porphine are untouched,
//! their ring atoms being π-capable throughout.
//!
//! Nothing here reads a bond order, so it gives one answer for a compound
//! however that compound was written.

use crate::molecule::base::{MoleculeCore, effective_valency};
use crate::molecule::bond::BondOrder;
use rustc_hash::FxHashSet;

/// How many π bonds an atom must form.
///
/// `valence - σ bonds`, where every bond counts once towards σ regardless of
/// the order it was drawn with, and hydrogens are counted because this crate
/// carries them as atoms.
///
/// `None` for an atom whose valence the model does not define -- the transition
/// metals, for which `VALENCIES` is `None`. A Hückel π count does not describe
/// metal-centre aromaticity in the first place, so rather than guess, such an
/// atom is treated as not π-capable and the π system is broken there.
pub fn pi_demand<M: MoleculeCore + ?Sized>(molecule: &M, atom: usize) -> Option<u8> {
    let atomic_number = *molecule.atomic_numbers().get(atom)?;
    let charge = molecule.charges().get(atom).copied().unwrap_or(0);
    let valence = effective_valency(atomic_number, charge)?;
    let sigma = molecule.get_atom_bonds(atom)?.len() as i8;

    // Unpaired electrons are deliberately *not* subtracted here. The recorded
    // radical count is itself derived from the input's bond orders -- an
    // aromatic bond counts 1.5 towards valence, so `[cH]` in a five-ring adds
    // to a full carbon valence and its unpaired electron disappears, while the
    // same atom written `[CH]` keeps it. Subtracting that would import the
    // dependence on the writing this module exists to avoid.
    Some((valence - sigma).max(0) as u8)
}

/// Every atom's π demand, indexed by atom.
pub fn pi_demands<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<u8> {
    (0..molecule.atomic_numbers().len())
        .map(|atom| pi_demand(molecule, atom).unwrap_or(0))
        .collect()
}

/// Whether an atom could take part in a ring's π system at all.
///
/// It can if it needs a π bond, or if it has a p orbital free to put into one:
/// a lone pair on a group 15 or 16 atom or a carbanion, or an empty orbital on
/// boron or a carbocation. What decides the second case is σ-connectivity, not
/// charge, and both of the ways to get that wrong are reachable:
///
/// * A quaternary `[N+]` has four σ bonds and no lone pair left, so it is as
///   saturated as a `CH2` is and breaks conjugation the same way. Taking any
///   formal charge as evidence of π activity would keep it and leave a
///   macrocycle looking conjugated through an sp3 centre.
/// * A pyrrole `[nH]`, a sulfonium `[s+]` and a thiophene `[s]` all have a lone
///   pair to donate and no π demand at all, so a rule that asked only for
///   demand would throw away most heteroaromatics.
///
/// A neutral saturated carbon is the case this excludes, and it is the case
/// that matters: it is what breaks a ring's conjugation.
pub fn is_pi_capable<M: MoleculeCore + ?Sized>(molecule: &M, atom: usize, demand: u8) -> bool {
    if demand > 0 {
        return true;
    }
    let Some(&atomic_number) = molecule.atomic_numbers().get(atom) else {
        return false;
    };
    let Some(bonds) = molecule.get_atom_bonds(atom) else {
        return false;
    };
    let sigma = bonds.len();
    let charge = molecule.charges().get(atom).copied().unwrap_or(0);

    match atomic_number {
        // Boron: three bonds and an empty p orbital, as in borepine.
        5 => sigma <= 3,
        // Carbon only when the charge gives it a lone pair or an empty orbital,
        // which it has spent if it is already four-coordinate.
        6 => charge != 0 && sigma <= 3,
        // Groups 15 and 16 keep a lone pair while they have one to keep.
        7 | 15 | 33 | 51 | 8 | 16 | 34 | 52 => sigma <= 3,
        _ => false,
    }
}

/// The subset of `cyclic_bonds` lying on a cycle of π-capable atoms.
///
/// This is what perception's "is this double bond exocyclic" test should be
/// asking against, rather than the molecule's whole cyclic-bond set -- see the
/// module documentation for the pair of molecules that separates the two.
pub fn conjugated_ring_bonds<M: MoleculeCore + ?Sized>(
    molecule: &M,
    cyclic_bonds: &FxHashSet<(usize, usize)>,
) -> FxHashSet<(usize, usize)> {
    let atom_count = molecule.atomic_numbers().len();
    let demands = pi_demands(molecule);

    // Degree within the cyclic-bond graph, counting only π-capable ends.
    let mut keep: Vec<bool> = (0..atom_count)
        .map(|atom| is_pi_capable(molecule, atom, demands[atom]))
        .collect();
    let mut degree = vec![0u32; atom_count];
    for &(a, b) in cyclic_bonds {
        if keep[a] && keep[b] {
            degree[a] += 1;
            degree[b] += 1;
        }
    }

    // 2-core: a vertex of degree one lies on no cycle, and peeling it can drop
    // its neighbour to degree one in turn.
    let mut peeled: Vec<usize> = (0..atom_count)
        .filter(|&a| keep[a] && degree[a] <= 1)
        .collect();
    while let Some(atom) = peeled.pop() {
        if !keep[atom] {
            continue;
        }
        keep[atom] = false;
        let Some(bonds) = molecule.get_atom_bonds(atom) else {
            continue;
        };
        for bond in bonds {
            let target = bond.target();
            if target >= atom_count || !keep[target] {
                continue;
            }
            if !cyclic_bonds.contains(&(atom.min(target), atom.max(target))) {
                continue;
            }
            degree[target] = degree[target].saturating_sub(1);
            if degree[target] <= 1 {
                peeled.push(target);
            }
        }
    }

    cyclic_bonds
        .iter()
        .copied()
        .filter(|&(a, b)| keep[a] && keep[b])
        .collect()
}

/// Whether the bond orders the molecule already carries satisfy every atom that
/// `aromatic` touches.
///
/// A cheap witness for [`kekulize`]: if no atom is short of π bonds, the double
/// bonds already there *are* a Kekulé structure for `aromatic`, and no matching
/// can do better than satisfy everyone. Perception asks this of a molecule that
/// has just been localized, where it holds for all but a handful in a hundred
/// thousand, and only when it fails is the matching worth running.
pub fn aromatic_set_is_satisfied<M: MoleculeCore + ?Sized>(
    molecule: &M,
    aromatic: &FxHashSet<(usize, usize)>,
) -> bool {
    let atom_count = molecule.atomic_numbers().len();
    let mut seen = vec![false; atom_count];
    for &(first, second) in aromatic {
        for atom in [first, second] {
            if seen[atom] {
                continue;
            }
            seen[atom] = true;
            let Some(bonds) = molecule.get_atom_bonds(atom) else {
                continue;
            };
            let mut pi = 0i16;
            for bond in bonds {
                match bond.bond_order() {
                    BondOrder::Double => pi += 1,
                    BondOrder::Triple => pi += 2,
                    // An aromatic bond is not a π bond this can count, so a
                    // molecule still holding one is not a witness to anything.
                    BondOrder::Aromatic => return false,
                    _ => (),
                }
            }
            if pi != pi_demand(molecule, atom).unwrap_or(0) as i16 {
                return false;
            }
        }
    }
    true
}

/// The result of asking whether a set of aromatic bonds can be written as
/// alternating single and double ones.
pub struct Kekulization {
    /// The partner each atom takes its double bond from, where it has one.
    pub matched: Vec<Option<usize>>,
    /// Which connected run of aromatic bonds each atom belongs to, or
    /// `usize::MAX` for an atom with no aromatic bond.
    pub component: Vec<usize>,
    /// Whether each component holds an atom the matching could not satisfy.
    pub deficient: Vec<bool>,
    /// Every such atom, in index order. Empty when the whole set works out.
    pub unsatisfied: Vec<usize>,
}

impl Kekulization {
    /// Whether the run of aromatic bonds this atom belongs to has no Kekulé
    /// structure. False for an atom with no aromatic bond.
    pub fn is_atom_deficient(&self, atom: usize) -> bool {
        self.component
            .get(atom)
            .is_some_and(|&label| label != usize::MAX && self.deficient[label])
    }
}

/// Works out which of `aromatic` would be the double bonds.
///
/// Each atom's π demand is what its element, charge and σ-connectivity require,
/// less the π bonds already committed on the bonds *outside* `aromatic`.
/// Satisfying as many of the remaining demands as possible over `aromatic` is a
/// maximum matching, and it has to be an exact one -- a greedy pairing leaves
/// atoms unpaired that could have been paired, and an unpaired demand is an
/// unpaired electron, so a greedy answer invents radicals.
///
/// A component holding an atom the matching leaves wanting has no Kekulé
/// structure at all: `O=c1ccccc1` is five carbons wanting a double bond each,
/// and one of them must go without. Which one is decided by the order the atoms
/// happen to be numbered in, so such a component must not be acted on.
pub fn kekulize<M: MoleculeCore + ?Sized>(
    molecule: &M,
    aromatic: &FxHashSet<(usize, usize)>,
) -> Kekulization {
    let atom_count = molecule.atomic_numbers().len();
    if aromatic.is_empty() {
        return Kekulization {
            matched: Vec::new(),
            component: Vec::new(),
            deficient: Vec::new(),
            unsatisfied: Vec::new(),
        };
    }
    let mut in_aromatic = vec![false; atom_count];
    for &(first, second) in aromatic {
        in_aromatic[first] = true;
        in_aromatic[second] = true;
    }

    // Only the atoms an aromatic bond touches can be part of the answer, so the
    // demand of the rest is never worked out.
    let mut residual = vec![0i16; atom_count];
    for atom in 0..atom_count {
        if !in_aromatic[atom] {
            continue;
        }
        let Some(bonds) = molecule.get_atom_bonds(atom) else {
            continue;
        };
        let mut spent = 0i16;
        for bond in bonds {
            let target = bond.target();
            if aromatic.contains(&(atom.min(target), atom.max(target))) {
                continue;
            }
            match bond.bond_order() {
                BondOrder::Double => spent += 1,
                BondOrder::Triple => spent += 2,
                _ => (),
            }
        }
        residual[atom] = pi_demand(molecule, atom).unwrap_or(0) as i16 - spent;
    }

    let edges: Vec<(usize, usize)> = aromatic
        .iter()
        .copied()
        .filter(|&(first, second)| residual[first] >= 1 && residual[second] >= 1)
        .collect();
    let matched = crate::molecule::matching::maximum_matching(atom_count, &edges);

    // Nothing left wanting means no component can be deficient, and the walk
    // that works out which one an atom belongs to need not happen at all.
    let any_unsatisfied = (0..atom_count)
        .any(|atom| in_aromatic[atom] && residual[atom] >= 1 && matched[atom].is_none());
    if !any_unsatisfied {
        return Kekulization {
            matched,
            component: Vec::new(),
            deficient: Vec::new(),
            unsatisfied: Vec::new(),
        };
    }

    let mut component = vec![usize::MAX; atom_count];
    let mut deficient: Vec<bool> = Vec::new();
    let mut unsatisfied = Vec::new();
    let mut stack = Vec::new();
    for atom in 0..atom_count {
        if component[atom] != usize::MAX || !in_aromatic[atom] {
            continue;
        }
        let label = deficient.len();
        deficient.push(false);
        component[atom] = label;
        stack.push(atom);
        while let Some(current) = stack.pop() {
            if residual[current] >= 1 && matched[current].is_none() {
                deficient[label] = true;
                unsatisfied.push(current);
            }
            let Some(bonds) = molecule.get_atom_bonds(current) else {
                continue;
            };
            for bond in bonds {
                let target = bond.target();
                if !aromatic.contains(&(current.min(target), current.max(target))) {
                    continue;
                }
                if component[target] == usize::MAX {
                    component[target] = label;
                    stack.push(target);
                }
            }
        }
    }
    unsatisfied.sort_unstable();

    Kekulization {
        matched,
        component,
        deficient,
        unsatisfied,
    }
}
