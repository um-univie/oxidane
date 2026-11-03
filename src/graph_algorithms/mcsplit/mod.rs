//! Maximum common subgraph search via **McSplit** (McCreesh, Prosser &
//! Trimble, *"A Partitioning Algorithm for Maximum Common Subgraph
//! Problems"*, IJCAI 2017).
//!
//! [`super::mces`] reduces the problem to maximum clique on a generic
//! modular-product graph. That reduction is textbook-correct but scales
//! badly on real molecules: coarse atom labels (many carbons look alike)
//! plus the "type-0" padding needed to let non-adjacent bond-pairs coexist
//! in one clique produce a large, dense, highly symmetric search graph.
//! Concretely, before that module's search was hardened this session,
//! comparing a 14-carbon fused tricyclic ring system against itself did not
//! finish in over a minute, and two real, differently-sized drugs sharing a
//! large scaffold (lovastatin vs. simvastatin, ~30 heavy atoms each) still
//! don't, even after hardening.
//!
//! McSplit instead searches directly over vertex-to-vertex mappings,
//! tracking remaining candidates as **bidomains**: groups of not-yet-mapped
//! vertices in each graph that are, given the vertices mapped so far, still
//! structurally indistinguishable. Every time a new pair is chosen, every
//! bidomain is refined (split by matching-edge-label to the newly mapped
//! vertices), which both shrinks the search rapidly and gives a
//! branch-and-bound bound — the sum of `min(|left|, |right|)` over all
//! remaining bidomains — that is far tighter than a flat candidate count.
//! That tighter bound, not any implementation trick, is what makes it
//! asymptotically better suited to exactly the symmetric/large-scaffold
//! cases [`super::mces`] struggles with (see [`search`] for the generic
//! algorithm and the concrete before/after timings recorded in this crate's
//! test suite).
//!
//! # Two variants
//!
//! - [`find_maximum_common_edge_subgraph`] runs the search on the two
//!   molecules' **line graphs** (vertex = bond, edge = shared atom — see
//!   [`line_graph`]) rather than on the atoms directly. A line-graph vertex
//!   *is* a bond, so maximizing matched line-graph vertices directly
//!   maximizes matched bonds: this is genuinely the same maximum-common-
//!   *edge*-subgraph guarantee [`super::mces`] provides (same objective,
//!   same freedom to combine independently-matching fragments — two bonds
//!   sharing no atom in *either* molecule are just non-adjacent on both
//!   sides of the line graph, which already satisfies bidomain-splitting
//!   trivially, no separate mechanism needed), just reached through
//!   McSplit's much tighter bound instead of a clique reduction. **This is
//!   the recommended default.**
//! - [`find_maximum_common_induced_subgraph`] runs the search directly on
//!   the atoms: it maximizes matched **atom** count under the *induced*
//!   constraint (if two atoms are mapped, whether — and how — they're
//!   bonded must agree between the two molecules). This is a different,
//!   stricter guarantee than edge-subgraph maximization (e.g. cyclohexane
//!   vs. hexane: induced matching must drop to a 5-atom/4-bond path, since
//!   mapping all 6 atoms sequentially would require the ring-closing bond to
//!   also exist in hexane; edge-maximizing finds the full 6-atom/5-bond
//!   path instead, simply not counting that one bond as matched). Kept
//!   because it's occasionally the more natural question to ask (e.g. "how
//!   much of this scaffold's shape, not just its bonds, is shared"), and
//!   because it's the more direct/simpler translation of the reference
//!   algorithm if that's ever useful for comparison.
//!
//! [`find_common_edge_subgraph_multi`] extends the edge variant to 3+
//! molecules by repeated pairwise reduction (a heuristic — see its own docs).
//!
//! [`super::mces`] remains available unchanged as a fallback for when
//! something here misbehaves or its still-open items below matter for your
//! case.
//!
//! # What counts as "the same atom"
//!
//! Atom compatibility is exact equality of everything
//! [`crate::molecule::graph::AtomLabel`] carries — element, formal charge,
//! aromaticity, isotope, and whether the atom is a stereocentre — **plus its
//! CIP absolute configuration** (R/S/E/Z), supplied via [`CipConfigs`].
//! Bond compatibility is exact [`crate::molecule::bond::BondOrder`]
//! equality, plus (for the edge variant) matching endpoint atom labels.
//!
//! ## Why CIP rather than the raw `@`/`@@` parity
//!
//! [`AtomLabel`](crate::molecule::graph::AtomLabel) alone is not enough:
//! `get_chiral_type` collapses both handednesses to `ChiralType::TH`, so it
//! distinguishes a stereocentre from a non-stereocentre but not R from S.
//!
//! The obvious fix — comparing the stored `ChiralClass` (`TH(1)`/`TH(2)`,
//! i.e. `@`/`@@`) — would be **wrong**, because that parity is defined
//! relative to the order an atom's neighbours happen to be listed in, not
//! by the structure itself. The same L-alanine gives `TH(2)` written
//! `N[C@@H](C)C(=O)O` and `TH(1)` written `C[C@H](N)C(=O)O`; matching on it
//! would report false mismatches purely from how the input was written.
//! CIP descriptors are order-independent, so they are what gets compared.
//!
//! ## CIP depends on the whole molecule
//!
//! CIP priority is a property of the **whole molecule**. If two molecules'
//! shared core has genuinely identical handedness, but a substituent
//! *outside* the matched region changes the priority ranking at a centre,
//! that centre gets different R/S labels and this search will refuse to
//! match it — even though the local 3-D arrangement agrees. The
//! theoretically exact alternative is to check the parity of the
//! *permutation induced on matched neighbours*, which depends only on the
//! matched region; that is a constraint on the mapping rather than a vertex
//! label, so it does not fit the bidomain-splitting framework and is not
//! implemented (see "Still to do").
//!
//! Note this is *stricter* than [`super::mces`], which is still
//! stereocentre-vs-not only: the two modules will disagree on stereoisomers.
//!
//! ## What CIP covers here
//!
//! [`MolecularGraph::assign_cip_configs`] is a port of RDKit's `CIPLabeler`
//! implementing sequence rules 1a/1b/2/3/4a/4b/4c/5/6 with auxiliary
//! descriptors, so labels agree with `rdCIPLabeler` on 381 of a 383-molecule
//! stereoisomer set (the two exceptions are a documented deviation over how a
//! three-coordinate centre's bracketed hydrogen is ordered). Pseudo-asymmetric
//! centres get their own [`CipConfig`] variants rather than collapsing onto
//! *R*/*S*, so they will not match an ordinary stereocentre.
//!
//! ## Stereochemistry CIP does not describe
//!
//! Allene, square-planar, trigonal-bipyramidal and octahedral centres get no
//! CIP descriptor — the rules do not cover them, and neither does RDKit's
//! labeller. On its own that made two different isomers completely
//! interchangeable: `S[Co@OH1](F)(I)(Cl)(C)Br` matched its `@OH2` isomer on
//! all seven atoms.
//!
//! These therefore carry a second label, [`HigherStereo`], holding the SMILES
//! chirality index read in the molecule's *canonical* atom ordering rather
//! than the one it was written with — so it distinguishes the isomers while
//! still agreeing between equivalent spellings of the same molecule. It costs
//! a canonization, so molecules are scanned first and only those that actually
//! contain such a centre pay for it.
//!
//! # Still to do
//!
//! - **`find_all_maximum_common_edge_subgraphs`.** Only single-best search
//!   is implemented; [`super::mces`] additionally returns every result tied
//!   for the maximum. Mechanical extension (collect instead of replacing on
//!   ties) but not done.
//! - **Local-parity stereo matching**, as described above — the
//!   context-independent alternative to CIP labels.
//! - **[`super::mces`] has not been given the same treatment**; it still
//!   compares chirality by type only, so the fallback keeps exactly the
//!   behaviour it was validated with.
//! - **[`Molecule3D`] gets no CIP descriptors** (see [`CipConfigs`]), so
//!   3-D molecules silently fall back to stereocentre-vs-not matching.
//! - **`find_common_edge_subgraph_multi` is order-sensitive and not
//!   provably maximal** — see its own docs. On a 7-molecule steroid-ester
//!   set it returns 48 atoms / 49 bonds in ~5 ms in the given order and
//!   one atom fewer reversed; an exact n-ary search would remove that gap, but note
//!   RDKit's exact `FindMCS` does not finish that same input within 300 s.
//! - **Explicit-stack search.** Recursion depth here is bounded by
//!   `min(|V(G1)|, |V(G2)|)` — safe for any realistic molecule, so plain
//!   recursion was the right call for readability — but if this is ever
//!   pushed at very large synthetic graphs, converting to an explicit stack
//!   (as `mces`'s `CliqueIterator` already does, for the same reason) is a
//!   mechanical follow-up, not a redesign.
//! - **Line-graph construction is `O(bonds²)`** ([`line_graph`]'s adjacency
//!   pass compares every bond pair) — fine at molecular scale, would need
//!   revisiting for anything much bigger.

mod graph;
mod line_graph;
mod reconstruct;
mod search;

pub use super::mces::{AtomMapping, BondMapping, MaximumCommonEdgeSubgraph};
use crate::graph_algorithms::mol_canonization::canonical_chiral_classes;
use crate::molecule::base::{Molecule, MoleculeCore};
use crate::molecule::chirality::ChiralClass;
use crate::molecule::cip::CipConfig;
use crate::molecule::{MolecularGraph, Molecule3D};
use graph::Graph;
pub use graph::HigherStereo;
use line_graph::LineGraph;
use std::collections::HashSet;

/// Supplies each atom's CIP absolute configuration, so stereocentres can be
/// compared by handedness (R vs S) rather than merely by "is or isn't a
/// stereocentre" — see the module docs.
///
/// This exists because CIP assignment currently lives on
/// [`MolecularGraph`] as an inherent method rather than on the
/// [`Molecule`] trait, so generic code can't reach it directly.
pub trait CipConfigs {
    /// One entry per atom, parallel to `atomic_numbers()`.
    fn cip_configs(&self) -> Vec<CipConfig>;

    /// Stereochemistry CIP does not describe — octahedral,
    /// trigonal-bipyramidal, square-planar and allene centres — one entry per
    /// atom, parallel to `atomic_numbers()`.
    ///
    /// A shorter (or empty) result is read as [`HigherStereo::None`] for the
    /// missing atoms, so an implementation with no higher-order stereo to
    /// report can leave this alone.
    fn higher_order_stereo(&self) -> Vec<HigherStereo> {
        Vec::new()
    }
}

impl CipConfigs for MolecularGraph {
    fn cip_configs(&self) -> Vec<CipConfig> {
        self.assign_cip_configs()
    }

    /// Reads the canonical chirality classes and keeps the four that CIP says
    /// nothing about.
    ///
    /// Canonization is expensive — far more so than the rest of building a
    /// [`Graph`] — and these geometries are rare, so the molecule is scanned
    /// first and canonized only if it actually contains one.
    fn higher_order_stereo(&self) -> Vec<HigherStereo> {
        let Some(classes) = self.chiral_classes() else {
            return Vec::new();
        };
        let has_higher_order = classes.iter().any(|class| {
            matches!(
                class,
                ChiralClass::OH(_) | ChiralClass::TB(_) | ChiralClass::SP(_) | ChiralClass::AL(_)
            )
        });
        if !has_higher_order {
            return Vec::new();
        }

        canonical_chiral_classes(self)
            .into_iter()
            .map(|class| match class {
                ChiralClass::OH(n) => HigherStereo::Octahedral(n),
                ChiralClass::TB(n) => HigherStereo::TrigonalBipyramidal(n),
                ChiralClass::SP(n) => HigherStereo::SquarePlanar(n),
                ChiralClass::AL(n) => HigherStereo::Allene(n),
                _ => HigherStereo::None,
            })
            .collect()
    }
}

impl CipConfigs for Molecule3D {
    /// CIP assignment is only implemented for [`MolecularGraph`], so 3D
    /// molecules fall back to no descriptors: their stereocentres are still
    /// distinguished from non-stereocentres (via
    /// [`crate::molecule::graph::AtomLabel`]), but R is not distinguished
    /// from S. Convert to a [`MolecularGraph`] first if that matters.
    fn cip_configs(&self) -> Vec<CipConfig> {
        vec![CipConfig::None; self.atomic_numbers().len()]
    }
}

/// Finds a maximum common **edge** subgraph between `mol_a` and `mol_b` —
/// the same guarantee as [`super::mces::find_maximum_common_edge_subgraph`],
/// reached via McSplit's tighter bound instead of a clique reduction (see
/// the module docs). This is the recommended default.
///
/// Returns `None` if the molecules share no bond at all.
pub fn find_maximum_common_edge_subgraph<T: Molecule + CipConfigs>(
    mol_a: &T,
    mol_b: &T,
) -> Option<MaximumCommonEdgeSubgraph> {
    let g1 = Graph::from_molecule(mol_a);
    let g2 = Graph::from_molecule(mol_b);
    let lg1 = LineGraph::from_graph(&g1);
    let lg2 = LineGraph::from_graph(&g2);

    let matched_bonds = search::search(&lg1, &lg2);
    if matched_bonds.is_empty() {
        return None;
    }

    let (atom_pairs, bond_pairs) =
        reconstruct::resolve(&g1, &g2, &matched_bonds, &lg1.endpoints, &lg2.endpoints);

    let bond_mapping: Vec<BondMapping> = bond_pairs
        .iter()
        .map(|&((a1, a2), (b1, b2))| BondMapping {
            bond_a: (g1.original_index[a1], g1.original_index[a2]),
            bond_b: (g2.original_index[b1], g2.original_index[b2]),
        })
        .collect();

    let atom_mapping: Vec<AtomMapping> = atom_pairs
        .iter()
        .map(|&(a, b)| {
            let atom_a = g1.original_index[a];
            let atom_b = g2.original_index[b];
            AtomMapping { atom_a, atom_b, atomic_number: mol_a.atomic_numbers()[atom_a] }
        })
        .collect();

    Some(MaximumCommonEdgeSubgraph { atom_mapping, bond_mapping })
}

/// The common substructure shared by three or more molecules, reported in
/// the first molecule's atom indices.
#[derive(Debug, Clone)]
pub struct CommonSubstructure {
    pub num_atoms: usize,
    pub num_bonds: usize,
    /// Atoms of `molecules[0]` that make up the common substructure.
    pub reference_atoms: Vec<usize>,
    /// Bonds of `molecules[0]` that make up it, as atom-index pairs.
    pub reference_bonds: Vec<(usize, usize)>,
}

/// Finds a common edge subgraph shared by **all** of `molecules` (2 or
/// more), by repeated pairwise reduction: the first two molecules' MCES
/// becomes a pattern, that pattern's MCES with the third becomes the next
/// pattern, and so on.
///
/// # This is a heuristic, not a guaranteed maximum
///
/// Each reduction step is itself exact, but committing to *one* maximum
/// pairwise result before seeing the remaining molecules can miss a
/// globally better answer — a different, equally-large match at step one
/// might have survived later steps better. The result is always a genuine
/// common substructure of every input; it is only its *maximality* that
/// isn't guaranteed. (For scale: RDKit's `FindMCS` solves the true n-ary
/// problem exactly, but on 7 large steroid esters it does not finish within
/// 300s, where this returns in well under a second.) The result can also
/// depend on input order, for the same reason.
///
/// Returns `None` if fewer than 2 molecules are given, or they share no bond.
pub fn find_common_edge_subgraph_multi<T: Molecule + CipConfigs>(
    molecules: &[T],
) -> Option<CommonSubstructure> {
    if molecules.len() < 2 {
        return None;
    }

    let mut accumulated = Graph::from_molecule(&molecules[0]);

    for other in &molecules[1..] {
        let g_other = Graph::from_molecule(other);
        let lg_acc = LineGraph::from_graph(&accumulated);
        let lg_other = LineGraph::from_graph(&g_other);

        let matched = search::search(&lg_acc, &lg_other);
        if matched.is_empty() {
            return None;
        }

        let keep_bonds: Vec<(usize, usize)> =
            matched.iter().map(|&(bond_acc, _)| lg_acc.endpoints[bond_acc]).collect();
        accumulated = accumulated.edge_subgraph(&keep_bonds);
    }

    let reference_bonds: Vec<(usize, usize)> = accumulated
        .bonds()
        .map(|((a, b), _)| (accumulated.original_index[a], accumulated.original_index[b]))
        .collect();
    let mut reference_atoms: Vec<usize> = accumulated.original_index.clone();
    reference_atoms.sort_unstable();

    Some(CommonSubstructure {
        num_atoms: reference_atoms.len(),
        num_bonds: reference_bonds.len(),
        reference_atoms,
        reference_bonds,
    })
}

/// Finds a maximum common **induced** subgraph between `mol_a` and `mol_b`
/// — a different, stricter guarantee than
/// [`find_maximum_common_edge_subgraph`]'s; see the module docs for how they
/// diverge and when each is the right question to ask.
///
/// Returns `None` if the molecules share no bond at all (an induced match
/// consisting only of mutually-unbonded atoms isn't a meaningful structural
/// overlap, so it's filtered out the same way an edge-subgraph match with no
/// bonds would be).
pub fn find_maximum_common_induced_subgraph<T: Molecule + CipConfigs>(
    mol_a: &T,
    mol_b: &T,
) -> Option<MaximumCommonEdgeSubgraph> {
    let g1 = Graph::from_molecule(mol_a);
    let g2 = Graph::from_molecule(mol_b);
    let matched = search::search(&g1, &g2);
    induced_result(mol_a, &g1, &g2, &matched)
}

fn induced_result<T: Molecule>(
    mol_a: &T,
    g1: &Graph,
    g2: &Graph,
    matched: &[(usize, usize)],
) -> Option<MaximumCommonEdgeSubgraph> {
    let mut bond_mapping = Vec::new();
    for i in 0..matched.len() {
        for j in (i + 1)..matched.len() {
            let (a1, b1) = matched[i];
            let (a2, b2) = matched[j];
            let order_a = g1.bond_order(a1, a2);
            if order_a.is_some() && order_a == g2.bond_order(b1, b2) {
                bond_mapping.push(BondMapping {
                    bond_a: (g1.original_index[a1], g1.original_index[a2]),
                    bond_b: (g2.original_index[b1], g2.original_index[b2]),
                });
            }
        }
    }

    if bond_mapping.is_empty() {
        return None;
    }

    let bonded_atoms_a: HashSet<usize> =
        bond_mapping.iter().flat_map(|b| [b.bond_a.0, b.bond_a.1]).collect();

    let atom_mapping: Vec<AtomMapping> = matched
        .iter()
        .map(|&(a, b)| (g1.original_index[a], g2.original_index[b]))
        .filter(|(a, _)| bonded_atoms_a.contains(a))
        .map(|(a, b)| AtomMapping { atom_a: a, atom_b: b, atomic_number: mol_a.atomic_numbers()[a] })
        .collect();

    Some(MaximumCommonEdgeSubgraph { atom_mapping, bond_mapping })
}
