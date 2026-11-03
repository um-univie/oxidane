//! Bemis--Murcko scaffolds: what is left of a molecule when its side chains
//! are taken off.
//!
//! Two molecules from the same medicinal-chemistry series differ in their
//! substituents and share their skeleton. A model trained on one and tested on
//! the other has not been asked a question it can get wrong -- which is why a
//! random train/test split of a chemical dataset reports numbers nobody can
//! reproduce on new chemistry. Splitting by scaffold is how that is avoided,
//! and [`scaffold_key`] is the key to split on.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::scaffold::scaffold_key;
//!
//! // Ibuprofen and its side chains are two different molecules; their
//! // scaffold is the same benzene ring.
//! let ibuprofen = MolecularGraph::from_smiles_single("CC(C)Cc1ccc(cc1)C(C)C(=O)O").unwrap();
//! assert_eq!(scaffold_key(&ibuprofen), "c1ccccc1");
//!
//! // An acyclic molecule has no scaffold at all, and gets the empty key.
//! let butane = MolecularGraph::from_smiles_single("CCCC").unwrap();
//! assert_eq!(scaffold_key(&butane), "");
//! ```

use crate::graph_algorithms::mol_canonization::canonize_mol;
use crate::io::ToSMILES;
use crate::molecule::aromaticity::MoleculeAromaticity;
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::molecular_graph::MolecularGraph;

/// The atoms of the Bemis--Murcko scaffold, in ascending index order.
///
/// The scaffold is every ring atom, every atom on a path between two rings,
/// and every atom held to one of those by a bond that is not single. Empty for
/// a molecule with no rings: an acyclic molecule has no skeleton to share.
///
/// # What the last clause is for
///
/// A lactam's carbonyl oxygen hangs off the ring by a double bond. Dropping it
/// with the rest of the side chains would make a lactam and a cyclic amine the
/// same scaffold, which is not what anyone means. Keeping it is also what
/// RDKit's `MurckoScaffold` does, so the keys agree.
///
/// The re-attachment is one deep. The oxygen of `CC(=O)N1CCCC1=O` returns
/// because it is doubly bonded to a ring carbon; the acyl group on the ring
/// nitrogen does not, because by the time anything is put back its own carbon
/// has already gone.
pub fn murcko_scaffold_atoms<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<usize> {
    let cyclic = molecule.cyclic_atoms();
    if cyclic.is_empty() {
        return Vec::new();
    }

    let atom_count = molecule.len();
    let heavy = |atom: usize| molecule.get_atomic_number(atom) != 1;

    // Everything heavy is in until it is peeled off.
    let mut kept: Vec<bool> = (0..atom_count).map(heavy).collect();
    let mut degree: Vec<u32> = (0..atom_count)
        .map(|atom| {
            if !kept[atom] {
                return 0;
            }
            molecule
                .get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .filter(|bond| kept[bond.target()])
                .count() as u32
        })
        .collect();

    // Peel terminal atoms until only rings and the paths between them remain.
    // A linker is never terminal -- it has a ring at each end -- so this needs
    // no path search of its own.
    let mut queue: Vec<usize> = (0..atom_count)
        .filter(|&atom| kept[atom] && !cyclic.contains(atom) && degree[atom] <= 1)
        .collect();
    while let Some(atom) = queue.pop() {
        if !kept[atom] || cyclic.contains(atom) || degree[atom] > 1 {
            continue;
        }
        kept[atom] = false;
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            let neighbour = bond.target();
            if !kept[neighbour] {
                continue;
            }
            degree[neighbour] -= 1;
            if !cyclic.contains(neighbour) && degree[neighbour] <= 1 {
                queue.push(neighbour);
            }
        }
    }

    // Put back what a double or triple bond is holding on. Decided against the
    // state after peeling, not during it, so that the order atoms came off in
    // cannot change the answer.
    let peeled = kept.clone();
    for atom in 0..atom_count {
        if peeled[atom] || !heavy(atom) {
            continue;
        }
        let held = molecule
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .any(|bond| peeled[bond.target()] && bond.bond_order() != BondOrder::Single);
        if held {
            kept[atom] = true;
        }
    }

    (0..atom_count).filter(|&atom| kept[atom]).collect()
}

/// The Bemis--Murcko scaffold as a molecule of its own.
///
/// Hydrogens are worked out afresh, so the ring carbon that carried a side
/// chain carries a hydrogen instead and the result is a molecule rather than a
/// fragment with a hole in it. A molecule with no rings gives an empty graph.
///
/// ```
/// use oxidane::prelude::*;
/// use oxidane::molecule::scaffold::murcko_scaffold;
///
/// let lactam = MolecularGraph::from_smiles_single("O=C1CCCN1C").unwrap();
/// // The N-methyl goes; the carbonyl oxygen stays.
/// assert_eq!(murcko_scaffold(&lactam).molecular_formula().to_string(), "C4H7N1O1");
/// ```
pub fn murcko_scaffold(molecule: &MolecularGraph) -> MolecularGraph {
    let atoms = murcko_scaffold_atoms(molecule);
    if atoms.is_empty() {
        return MolecularGraph::default();
    }

    // Cut a Kekule copy, not an aromatic one. An aromatic bond counts one and
    // a half towards a valence, so a nitrogen carrying two ring bonds already
    // looks trivalent whether or not it has a hydrogen -- and the nitrogen
    // whose substituent the cut just removed is exactly the one that needs a
    // hydrogen back. Localised bonds are integers, so the shortfall is
    // visible: `Cn1ccc2ccccc12` gives the scaffold of indole rather than an
    // aromatic nitrogen with nothing on it, which is not a molecule.
    let mut source = molecule.clone();
    let _ = source.localize_aromatic_bonds();

    let mut scaffold = source
        .subgraph(&with_hydrogens(&source, &atoms))
        .unwrap_or_else(|_| MolecularGraph::default());
    scaffold.add_hydrogens();
    let _ = scaffold.aromatize_opensmiles();
    scaffold
}

/// `atoms` together with the hydrogens hanging off them, in ascending order.
///
/// The hydrogens have to come along rather than be worked out again on the
/// other side. An aromatic bond counts one and a half towards a valence, so a
/// two-connected aromatic nitrogen already looks trivalent and nothing would
/// give it back the hydrogen that tells pyrrole from pyridine -- the scaffold
/// of indole would come out as an aromatic `n` with no hydrogen, which is not
/// a molecule. The hydrogens a cut actually frees are still worked out
/// afterwards, and those are on atoms whose bonds are single.
fn with_hydrogens<M: MoleculeCore + ?Sized>(molecule: &M, atoms: &[usize]) -> Vec<usize> {
    let mut kept = vec![false; molecule.len()];
    for &atom in atoms {
        kept[atom] = true;
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            if molecule.get_atomic_number(bond.target()) == 1 {
                kept[bond.target()] = true;
            }
        }
    }
    (0..molecule.len()).filter(|&atom| kept[atom]).collect()
}

/// The scaffold with its elements and bond orders forgotten: every atom a
/// carbon, every bond single.
///
/// This is the coarser of the two groupings. A pyridine and a benzene are
/// different scaffolds and the same framework, which is the right resolution
/// for asking whether two series share a shape rather than a substructure.
///
/// ```
/// use oxidane::prelude::*;
/// use oxidane::molecule::scaffold::murcko_framework;
/// use oxidane::graph_algorithms::mol_canonization::canonize_mol;
///
/// let pyridine = MolecularGraph::from_smiles_single("c1ccncc1").unwrap();
/// let benzene = MolecularGraph::from_smiles_single("c1ccccc1").unwrap();
/// assert_eq!(
///     canonize_mol(&murcko_framework(&pyridine)).to_smiles(),
///     canonize_mol(&murcko_framework(&benzene)).to_smiles(),
/// );
/// ```
pub fn murcko_framework(molecule: &MolecularGraph) -> MolecularGraph {
    let atoms = murcko_scaffold_atoms(molecule);
    if atoms.is_empty() {
        return MolecularGraph::default();
    }
    let Ok(mut framework) = molecule.subgraph(&atoms) else {
        return MolecularGraph::default();
    };

    for atomic_number in framework.atomic_numbers_mut().iter_mut() {
        *atomic_number = 6;
    }
    for charge in framework.charges_mut().iter_mut() {
        *charge = 0;
    }
    *framework.isotopes_mut() = None;
    for bonds in framework.atom_bonds_mut().iter_mut() {
        for bond in bonds.iter_mut() {
            bond.bond_order = BondOrder::Single;
        }
    }
    framework.initialize_chiral_classes();
    framework.add_hydrogens();
    framework
}

/// The key two molecules share when they share a scaffold: the canonical
/// SMILES of [`murcko_scaffold`], or the empty string for an acyclic molecule.
///
/// Group a dataset by this and you have the groups
/// [`crate::ml::split::scaffold_split`] keeps on one side of the split.
pub fn scaffold_key(molecule: &MolecularGraph) -> String {
    let scaffold = murcko_scaffold(molecule);
    if scaffold.is_empty() {
        return String::new();
    }
    canonize_mol(&scaffold).to_smiles()
}

/// [`scaffold_key`] at framework resolution.
pub fn framework_key(molecule: &MolecularGraph) -> String {
    let framework = murcko_framework(molecule);
    if framework.is_empty() {
        return String::new();
    }
    canonize_mol(&framework).to_smiles()
}
