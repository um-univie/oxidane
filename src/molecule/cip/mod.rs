//! CIP (Cahn-Ingold-Prelog) stereodescriptor assignment.
//!
//! This is a port of RDKit's `CIPLabeler` (`Code/GraphMol/CIPLabeler`), which
//! is itself a port of John Mayfield's `centres` library written for the
//! Hanson *et al.* re-specification of the CIP rules — the implementation
//! validated against Hanson's reference suite. It implements sequence rules
//! 1a, 1b, 2, 3, 4a, 4b, 4c, 5 and 6, with auxiliary descriptors, which is
//! what an approximate implementation of rules 1a and 2 alone cannot reach:
//! bridged polycyclics, pseudo-asymmetric centres, and anything whose ranking
//! turns on the handedness of atoms *inside* the branches being ranked.
//!
//! RDKit is distributed under the BSD 3-Clause licence:
//!
//! > Copyright (C) 2020 Schrödinger, LLC. All rights reserved.
//! >
//! > Redistribution and use in source and binary forms, with or without
//! > modification, are permitted provided that the above copyright notice,
//! > this list of conditions and the following disclaimer are retained.
//! > THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//! > IS" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
//!
//! # Structure
//!
//! - `mol` — the molecule as the rules need it: Kekulé bond orders, ring
//!   membership, and Mancude-ring averaged atomic numbers.
//! - `digraph` — the hierarchical digraph, built lazily.
//! - `rules` — the nine sequence rules and the substituent sort.
//! - `configs` — tetrahedral centres and double bonds.
//! - `descriptor` — descriptors and like/unlike pair lists.
//!
//! Allenes are covered too, though RDKit's labeller does not cover them: the
//! axis is treated as a collapsed tetrahedral centre whose four ligands are
//! the two ends' substituents, ranked within each end as the proximity rule
//! for axial chirality requires.
//!
//! # Not covered
//!
//! Square-planar (`SP`), trigonal-bipyramidal (`TB`) and octahedral (`OH`)
//! configurations get no descriptor: the digraph and rules handle them, but
//! the geometry-to-parity step is not written for those cases. Sulfoxides are
//! labelled when the parser records them as tetrahedral.

mod configs;
mod descriptor;
mod digraph;
mod mol;
mod rules;

use std::rc::Rc;

use crate::molecule::{
    base::MoleculeCore,
    bond::BondOrder,
    chirality::{
        ChiralClass, index_sorted_allene_carriers, index_sorted_neighbors,
        tetrahedral_tag_for_order,
    },
    molecular_graph::MolecularGraph,
};

use configs::Configuration;
use descriptor::Descriptor;
use digraph::Digraph;
use mol::CipMol;
use rules::{Ctx, Rule};

/// CIP absolute configuration at a stereocentre.
///
/// The lowercase [`CipConfig::PseudoR`]/[`CipConfig::PseudoS`] are
/// *pseudo-asymmetric*: centres that are stereogenic only because of the
/// handedness of the branches around them. They are reported separately
/// because they are a different stereochemical class — a pseudo-asymmetric
/// centre is not interchangeable with an ordinary one, and mirroring the
/// molecule does not invert it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CipConfig {
    R,
    S,
    /// Pseudo-asymmetric *r*.
    PseudoR,
    /// Pseudo-asymmetric *s*.
    PseudoS,
    E,
    Z,
    /// Pseudo-asymmetric *seqCis*.
    SeqCis,
    /// Pseudo-asymmetric *seqTrans*.
    SeqTrans,
    #[default]
    None,
}

impl std::fmt::Display for CipConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            CipConfig::R => "R",
            CipConfig::S => "S",
            CipConfig::PseudoR => "r",
            CipConfig::PseudoS => "s",
            CipConfig::E => "E",
            CipConfig::Z => "Z",
            CipConfig::SeqCis => "z",
            CipConfig::SeqTrans => "e",
            CipConfig::None => "",
        };
        write!(f, "{text}")
    }
}

impl From<Descriptor> for CipConfig {
    fn from(descriptor: Descriptor) -> Self {
        match descriptor {
            Descriptor::R => CipConfig::R,
            Descriptor::S => CipConfig::S,
            Descriptor::LowerR => CipConfig::PseudoR,
            Descriptor::LowerS => CipConfig::PseudoS,
            Descriptor::E => CipConfig::E,
            Descriptor::Z => CipConfig::Z,
            Descriptor::SeqCis => CipConfig::SeqCis,
            Descriptor::SeqTrans => CipConfig::SeqTrans,
            _ => CipConfig::None,
        }
    }
}

/// Comparison budget for the first, cheap pass over every centre.
///
/// Most centres are settled by the constitutional rules in a handful of
/// comparisons; a few need far more. Running the cheap rules under a tight
/// budget first, then retrying only what is left over with a generous one,
/// keeps the common case fast without giving up on the hard cases.
const CONSTITUTIONAL_BUDGET: u64 = 2_000;

/// Comparison budget for the second pass, where the full rule set runs.
const FULL_BUDGET: u64 = 10_000_000;

/// Comparison budget for one whole `assign_cip_configs` call, every centre
/// and every auxiliary descriptor included.
///
/// [`FULL_BUDGET`] bounds one labelling, but a molecule asks for one per
/// centre and [`label_auxiliary`] for one per occurrence of every other centre
/// inside each centre's digraph. On a cage with dozens of centres that can see
/// each other -- PubChem's CID 178493059, a functionalised fullerene, was the
/// case -- the product never finished. Drawing every labelling from one pool
/// bounds the call; what the pool cannot pay for stays unlabelled, the same
/// outcome as one centre running out.
const TOTAL_BUDGET: u64 = 20_000_000;

/// Runs `label` with a comparison budget of at most `cap`, drawn from
/// `remaining`, and charges `remaining` what it used.
fn budgeted<T>(
    remaining: &mut u64,
    master: &[Rule],
    cap: u64,
    label: impl FnOnce(&mut Ctx) -> T,
) -> T {
    let granted = cap.min(*remaining);
    let mut ctx = Ctx {
        master,
        budget: granted,
    };
    let result = label(&mut ctx);
    *remaining -= granted - ctx.budget;
    result
}

/// Rules that depend only on constitution — no descriptors involved, so they
/// can run before anything has been labelled.
const CONSTITUTIONAL_RULES: [Rule; 3] = [Rule::R1a, Rule::R1b, Rule::R2];

/// The complete rule set. Rules 3 onwards read descriptors, so they are only
/// meaningful once the auxiliary pass has labelled the digraph's interior.
const ALL_RULES: [Rule; 9] = [
    Rule::R1a,
    Rule::R1b,
    Rule::R2,
    Rule::R3,
    Rule::R4a,
    Rule::R4b(Descriptor::None),
    Rule::R4c,
    Rule::R5(Descriptor::None),
    Rule::R6,
];

impl MolecularGraph {
    /// Computes CIP absolute configurations (R/S/E/Z) for every atom.
    ///
    /// Returns a `Vec<CipConfig>` parallel to the atom list; atoms that are
    /// not stereocentres, and stereocentres the rules cannot resolve, get
    /// [`CipConfig::None`]. Both ends of a stereogenic double bond carry that
    /// bond's E/Z.
    ///
    /// Requires chirality classes to have been assigned already, which
    /// [`MolecularGraph::from_smiles`] does.
    ///
    /// # Example
    /// ```rust
    /// # use oxidane::prelude::*;
    /// use oxidane::molecule::cip::CipConfig;
    /// let mol = MolecularGraph::from_smiles_single("N[C@@H](C)C(=O)O").unwrap();
    /// assert_eq!(mol.assign_cip_configs()[1], CipConfig::S);
    /// ```
    pub fn assign_cip_configs(&self) -> Vec<CipConfig> {
        let atom_count = self.atomic_numbers().len();
        let mut configs = vec![CipConfig::None; atom_count];
        if self.chiral_classes().is_none() {
            return configs;
        }

        let found = find_configurations(self);
        if found.is_empty() {
            return configs;
        }

        let cip_mol = Rc::new(CipMol::new(self));
        let mut digraphs: Vec<Digraph> = found
            .iter()
            .map(|config| Digraph::new(Rc::clone(&cip_mol), config.root_atom()))
            .collect();
        let mut labels = vec![Descriptor::None; found.len()];
        let mut remaining = TOTAL_BUDGET;

        // Cheap pass: constitutional rules only, tight budget.
        for (index, config) in found.iter().enumerate() {
            let descriptor = budgeted(
                &mut remaining,
                &CONSTITUTIONAL_RULES,
                CONSTITUTIONAL_BUDGET,
                |ctx| config.label(&mut digraphs[index], ctx),
            );
            if descriptor != Descriptor::Unknown {
                labels[index] = descriptor;
            }
        }

        // Anything still unresolved gets the full rule set, which first needs
        // the interior of its own digraph labelled (see [`label_auxiliary`]).
        for index in 0..found.len() {
            if labels[index] != Descriptor::None {
                continue;
            }
            if remaining == 0 {
                break;
            }
            let descriptor = budgeted(&mut remaining, &CONSTITUTIONAL_RULES, FULL_BUDGET, |ctx| {
                found[index].label(&mut digraphs[index], ctx)
            });
            if descriptor != Descriptor::Unknown {
                labels[index] = descriptor;
                continue;
            }

            label_auxiliary(&found, index, &mut digraphs[index], &mut remaining);
            let descriptor = budgeted(&mut remaining, &ALL_RULES, FULL_BUDGET, |ctx| {
                found[index].label(&mut digraphs[index], ctx)
            });
            if descriptor != Descriptor::Unknown {
                labels[index] = descriptor;
            }
        }

        for (config, label) in found.iter().zip(labels.iter()) {
            let value = CipConfig::from(*label);
            if value == CipConfig::None {
                continue;
            }
            for atom in config.foci() {
                configs[atom] = value;
            }
        }
        configs
    }
}

/// Labels every *other* stereocentre that falls inside `index`'s digraph, so
/// the descriptor-reading rules have something to read.
///
/// The same molecular stereocentre can appear many times in one digraph — once
/// per path that reaches it — and each occurrence may deserve a different
/// descriptor, because from deeper in the tree it is being viewed with a
/// different substituent playing the role of "the way back to the root". So
/// the occurrences are labelled outermost first, and all occurrences at one
/// distance are computed before any of them is written back, so that they
/// cannot see each other's results.
fn label_auxiliary(
    configs: &[Configuration],
    index: usize,
    digraph: &mut Digraph,
    remaining: &mut u64,
) {
    let mut pending: Vec<(i32, digraph::NodeId, usize)> = Vec::new();

    for (other, config) in configs.iter().enumerate() {
        if other == index {
            continue;
        }
        let foci = config.foci();
        if !foci.iter().any(|&atom| digraph.seen_atom(atom)) {
            continue;
        }

        for node in digraph.nodes_for_atom(foci[0]) {
            if digraph.node(node).is_duplicate() {
                continue;
            }
            // For a double bond, the descriptor belongs to whichever end is
            // nearer the root; that is the one the digraph looks "through".
            let mut low = node;
            if foci.len() == 2 {
                for edge in digraph.edges_to_atom(node, foci[1]) {
                    let other_node = digraph.edge(edge).other(node);
                    if digraph.node(other_node).distance < digraph.node(node).distance {
                        low = other_node;
                    }
                }
            }
            if !digraph.node(low).is_duplicate() {
                pending.push((digraph.node(low).distance, low, other));
            }
        }
    }

    pending.sort_by(|a, b| b.0.cmp(&a.0));

    let mut queue: Vec<(digraph::NodeId, Descriptor)> = Vec::new();
    let mut previous = i32::MAX;
    for (distance, node, other) in pending {
        if distance < previous {
            for (node, descriptor) in queue.drain(..) {
                digraph.node_mut(node).aux = descriptor;
            }
            previous = distance;
        }
        let descriptor = budgeted(remaining, &ALL_RULES, FULL_BUDGET, |ctx| {
            configs[other].label_from(digraph, ctx, node)
        });
        queue.push((node, descriptor));
    }
    for (node, descriptor) in queue {
        digraph.node_mut(node).aux = descriptor;
    }
}

/// Collects every stereo configuration the molecule declares.
fn find_configurations(mol: &MolecularGraph) -> Vec<Configuration> {
    let Some(classes) = mol.chiral_classes() else {
        return Vec::new();
    };
    let mut configs = Vec::new();

    for (atom, &class) in classes.iter().enumerate() {
        let ChiralClass::TH(tag) = class else {
            continue;
        };
        let neighbours = index_sorted_neighbors(mol, atom);
        // The stored tag is expressed against the index-sorted neighbour
        // order, so that is the reference order the parity is taken from.
        if let Some(config) = Configuration::new_tetrahedral(atom, &neighbours, tag == 2) {
            configs.push(config);
        }
    }

    for (atom, other, carriers, cis) in stereogenic_double_bonds(mol, classes) {
        configs.push(Configuration::new_sp2(atom, other, carriers, cis));
    }

    for (centre, &class) in classes.iter().enumerate() {
        let ChiralClass::AL(tag) = class else {
            continue;
        };
        let Some(carriers) = index_sorted_allene_carriers(mol, centre) else {
            continue;
        };
        let mut ends: Vec<usize> = mol.atom_bonds()[centre]
            .iter()
            .map(|bond| bond.target())
            .collect();
        ends.sort_unstable();
        configs.push(Configuration::new_allene(
            centre,
            [ends[0], ends[1]],
            carriers,
            tag == 2,
        ));
    }

    configs
}

/// Finds the double bonds whose geometry is defined, reporting each as
/// `(atom, partner, [carrier on atom, carrier on partner], are they cis)`.
///
/// Two ends carrying the *same* chirality class have their first single-bonded
/// neighbours on opposite sides; different classes put them on the same side.
/// That is the convention the parser and canonisation both maintain, so
/// deriving the carriers from it keeps this consistent with the rest of the
/// crate under renumbering.
fn stereogenic_double_bonds(
    mol: &MolecularGraph,
    classes: &[ChiralClass],
) -> Vec<(usize, usize, [usize; 2], bool)> {
    let is_cis_trans = |class: ChiralClass| {
        matches!(
            class,
            ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
        )
    };

    let mut bonds = Vec::new();
    for atom in 0..mol.atomic_numbers().len() {
        if !is_cis_trans(classes[atom]) {
            continue;
        }
        let Some(other) = mol.atom_bonds()[atom]
            .iter()
            .find(|bond| bond.bond_order() == BondOrder::Double)
            .map(|bond| bond.target())
        else {
            continue;
        };
        if other <= atom || !is_cis_trans(classes[other]) {
            continue;
        }

        let first_single = |centre: usize, exclude: Option<usize>| {
            mol.atom_bonds()[centre]
                .iter()
                .filter(|bond| bond.bond_order() != BondOrder::Double)
                .map(|bond| bond.target())
                .find(|&target| Some(target) != exclude)
        };
        let (Some(carrier_a), Some(carrier_b)) =
            (first_single(atom, None), first_single(other, Some(atom)))
        else {
            continue;
        };

        let trans = classes[atom] != classes[other];
        bonds.push((atom, other, [carrier_a, carrier_b], !trans));
    }
    bonds
}

/// Converts freshly parsed tetrahedral tags into the canonical index-sorted
/// reference used throughout the crate.
///
/// The parser stores each `@`/`@@` relative to the atom's SMILES *written* order.
/// `write_orders[center]` holds the from-atom flag, neighbor order and ring count as
/// captured during parsing; the implicit hydrogen (added afterwards) belongs right
/// after the from-atom, or first for a fragment root. Re-expressing the tag against
/// the atom's index-sorted neighbor ordering yields a reference that does not depend
/// on bond listing order, so `to_smiles` and canonization can recompute the emitted
/// tag from the actual emission order without the two drifting apart. Call once,
/// right after parsing (with hydrogens already added).
pub(crate) fn normalize_tetrahedral_references(
    mol: &mut MolecularGraph,
    write_orders: &crate::io::IntMap<usize, crate::io::SmilesWriteOrder>,
) {
    let Some(classes) = mol.chiral_classes() else {
        return;
    };
    if !classes.iter().any(|c| matches!(c, ChiralClass::TH(_))) {
        return;
    }
    let classes = classes.to_vec();
    let mut updates: Vec<(usize, u8)> = Vec::new();
    for (center, &class) in classes.iter().enumerate() {
        let ChiralClass::TH(raw) = class else {
            continue;
        };
        let Some(order) = write_orders.get(&center) else {
            continue;
        };
        let (has_from, heavy) = (order.has_from, &order.neighbors);
        // Hydrogens on this centre that the parser added afterwards, ascending
        // by index. A hydrogen written out as its own atom (`[C@@]([H])...`)
        // is already in `heavy` at the position it was written, so including
        // it here too would list it twice and make the orderings different
        // lengths.
        let mut hydrogens: Vec<usize> = mol.atom_bonds()[center]
            .iter()
            .map(|bond| bond.target())
            .filter(|&target| mol.atomic_numbers()[target] == 1 && !heavy.contains(&target))
            .collect();
        hydrogens.sort_unstable();

        // Reconstruct the full SMILES written order: from-atom, then implicit
        // hydrogen(s), then the remaining heavy neighbors — or hydrogen(s) first
        // for a fragment root that has no from-atom.
        let mut written: Vec<usize> = Vec::with_capacity(4);
        if has_from {
            if let Some((&first, rest)) = heavy.split_first() {
                written.push(first);
                written.extend(&hydrogens);
                written.extend(rest);
            }
        } else {
            written.extend(&hydrogens);
            written.extend(heavy);
        }

        // Four neighbours for an ordinary centre, three for one whose fourth
        // vertex is a lone pair. Either way the tag only has to be
        // re-expressed against a reference ordering of the *same* neighbours,
        // so the parity calculation is identical.
        if written.len() < 3 || written.len() > 4 {
            continue;
        }
        let sorted = index_sorted_neighbors(mol, center);
        let raw = if crate::molecule::chirality::tetrahedral_ring_closure_inverts(
            mol,
            center,
            order.ring_closures,
        ) {
            3 - raw
        } else {
            raw
        };
        updates.push((center, tetrahedral_tag_for_order(raw, &written, &sorted)));
    }
    if updates.is_empty() {
        return;
    }
    let classes = mol.chiral_classes_mut().unwrap();
    for (center, tag) in updates {
        classes[center] = ChiralClass::TH(tag);
    }
}

/// Converts freshly parsed allene tags into the canonical carrier reference
/// used throughout the crate.
///
/// OpenSMILES says an allene is "conceptually collapsed into a single
/// tetrahedral chiral center" whose neighbours are the substituents at the two
/// ends, taken in the order the SMILES wrote them. That order depends on how
/// the string was laid out, so — exactly as for a tetrahedral centre — the tag
/// is re-expressed here against
/// [`index_sorted_allene_carriers`], which depends only on the atoms.
///
/// Without this the tag has no meaning of its own: it is kept consistent with
/// the emission order by the writer and the reordering code, which preserves
/// the molecule but leaves nothing that can be *read* as "which way do these
/// four wind".
///
/// Call once, right after parsing, with hydrogens already added.
pub(crate) fn normalize_allene_references(
    mol: &mut MolecularGraph,
    write_orders: &crate::io::IntMap<usize, crate::io::SmilesWriteOrder>,
) {
    let Some(classes) = mol.chiral_classes() else {
        return;
    };
    if !classes.iter().any(|c| matches!(c, ChiralClass::AL(_))) {
        return;
    }
    let classes = classes.to_vec();

    let mut updates: Vec<(usize, u8)> = Vec::new();
    for (centre, &class) in classes.iter().enumerate() {
        let ChiralClass::AL(raw) = class else {
            continue;
        };
        let Some(written) = written_allene_carriers(mol, centre, write_orders) else {
            continue;
        };
        let Some(sorted) = index_sorted_allene_carriers(mol, centre) else {
            continue;
        };
        updates.push((centre, tetrahedral_tag_for_order(raw, &written, &sorted)));
    }
    if updates.is_empty() {
        return;
    }
    let classes = mol.chiral_classes_mut().unwrap();
    for (centre, tag) in updates {
        classes[centre] = ChiralClass::AL(tag);
    }
}

/// The four carriers in the order the SMILES wrote them: the ends in the order
/// the centre listed them, and within each end its own written order with the
/// bond back to the centre removed.
///
/// Hydrogens are folded into their atom during parsing, so an end that carries
/// one has it reinserted where OpenSMILES puts it — immediately after the atom
/// the end was reached from, or first if the end started the fragment.
fn written_allene_carriers(
    mol: &MolecularGraph,
    centre: usize,
    write_orders: &crate::io::IntMap<usize, crate::io::SmilesWriteOrder>,
) -> Option<Vec<usize>> {
    let centre_written = &write_orders.get(&centre)?.neighbors;
    if centre_written.len() != 2 {
        return None;
    }

    let mut carriers = Vec::with_capacity(4);
    for &end in centre_written {
        let order = write_orders.get(&end)?;
        let (has_from, heavy) = (order.has_from, &order.neighbors);
        let mut hydrogens: Vec<usize> = mol.atom_bonds()[end]
            .iter()
            .map(|bond| bond.target())
            .filter(|&target| mol.atomic_numbers()[target] == 1 && !heavy.contains(&target))
            .collect();
        hydrogens.sort_unstable();

        let mut order: Vec<usize> = Vec::with_capacity(3);
        if has_from {
            let (&first, rest) = heavy.split_first()?;
            order.push(first);
            order.extend(&hydrogens);
            order.extend(rest);
        } else {
            order.extend(&hydrogens);
            order.extend(heavy);
        }
        order.retain(|&atom| atom != centre);
        if order.len() != 2 {
            return None;
        }
        carriers.extend(order);
    }
    Some(carriers)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values are RDKit `rdCIPLabeler` ground truth for each SMILES.

    #[test]
    fn l_alanine_is_s() {
        let mol = MolecularGraph::from_smiles_single("N[C@@H](C)C(=O)O").unwrap();
        assert_eq!(mol.assign_cip_configs()[1], CipConfig::S);
    }

    #[test]
    fn d_alanine_is_r() {
        let mol = MolecularGraph::from_smiles_single("N[C@H](C)C(=O)O").unwrap();
        assert_eq!(mol.assign_cip_configs()[1], CipConfig::R);
    }

    #[test]
    fn r_bromochlorofluoromethane() {
        let mol = MolecularGraph::from_smiles_single("F[C@H](Cl)Br").unwrap();
        assert_eq!(mol.assign_cip_configs()[1], CipConfig::R);
    }

    #[test]
    fn s_bromochlorofluoromethane() {
        let mol = MolecularGraph::from_smiles_single("F[C@@H](Cl)Br").unwrap();
        assert_eq!(mol.assign_cip_configs()[1], CipConfig::S);
    }

    #[test]
    fn trans_2_butene_is_e() {
        let mol = MolecularGraph::from_smiles_single("C/C=C/C").unwrap();
        let cip = mol.assign_cip_configs();
        assert!(cip.contains(&CipConfig::E), "{cip:?}");
    }

    #[test]
    fn cis_2_butene_is_z() {
        let mol = MolecularGraph::from_smiles_single(r"C/C=C\C").unwrap();
        let cip = mol.assign_cip_configs();
        assert!(cip.contains(&CipConfig::Z), "{cip:?}");
    }
}
