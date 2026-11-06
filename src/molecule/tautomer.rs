//! Tautomer-invariant canonical keys.
//!
//! Two tautomers are the same compound in a registry and the same example in a
//! training set, but they are different graphs, so
//! [`canonize_mol`](crate::graph_algorithms::mol_canonization::canonize_mol)
//! separates them. Warfarin's keto and enol forms deduplicate to two rows, a
//! scaffold split puts them in different folds, and a model sees the same
//! molecule twice with two labels.
//!
//! # What this does instead of rewriting
//!
//! The tempting approach -- apply shift rules until nothing changes and canonize
//! the result -- has to be *confluent* to be a canonical form, and rule sets
//! large enough to be useful stop being confluent. Nothing here rewrites.
//!
//! Instead the data that distinguishes one tautomer from another is **erased**
//! and replaced by the totals it conserves. Within a tautomeric group:
//!
//! * every bond loses its order (all become single),
//! * every mobile hydrogen is deleted,
//! * every endpoint's formal charge is zeroed,
//! * every endpoint's unpaired electrons are deleted,
//! * the hydrogen, charge and radical totals of every endpoint in the
//!   component are recorded against it, pooled rather than kept one set per
//!   group -- a proton that moves from one acidic site to another is the same
//!   compound, and [`TautomericGroups::totals`] says why.
//!
//! The map from a structure to that erased skeleton is *constant* on each
//! tautomer class by construction -- every datum that differs has been thrown
//! away -- so no confluence argument is needed. What is left is a plain
//! molecular graph, canonized by the machinery that was already here, and the
//! group totals are appended to the canonical SMILES it writes.
//!
//! The group totals are not decoration. Without them the erased skeleton of
//! 2-pyridone would collide with a structure that genuinely has those bonds
//! single and those atoms bare. With them the two differ, because one carries a
//! group and the other does not.
//!
//! # What comes out
//!
//! Two things. [`tautomer_key`] gives a [`String`] -- shaped like a canonical
//! SMILES with a suffix -- which answers "are these the same compound"; the
//! skeleton in it is not a chemically valid structure and is not meant to be
//! read back. [`normalize`] gives a real molecule: the one tautomer of the
//! class that stands for it. The two share their whole front half, so a
//! molecule and its normalization always share a key.
//!
//! # Perception, and why it is invariant rather than merely tested
//!
//! A **conjugated component** is a connected component of the subgraph on the
//! π-capable atoms, not crossing a frozen bond. A component holding two or more
//! endpoint elements is a **tautomeric group**; its endpoints are those atoms.
//! What is conserved -- total hydrogens, total charge, total unpaired electrons
//! -- is conserved by the molecule's groups together rather than by each of
//! them.
//!
//! Every ingredient is read from the σ-skeleton -- elements, degrees, and
//! totals. None reads a bond order, and none reads *where within a group* a
//! mobile hydrogen or an unpaired electron sits. Two tautomers therefore
//! present identical input to every step, and equal keys follow from the
//! construction rather than from having been measured.
//!
//! This replaced an alternating-path walk closed to a fixed point, which read
//! bond orders to test its alternation and so could answer differently
//! depending on which tautomer it was shown. Every guard that walk needed -- a
//! path-length bound, a step budget, a cap on the closure's rounds -- is gone
//! with it, and with them the chance that one of them silently changes an
//! answer.
//!
//! Two kinds of bond are **frozen**, meaning no group owns them and their order
//! is not erased:
//!
//! * a sulfonyl, sulfinyl or phosphoryl oxygen's bond, identified by degrees
//!   alone. Asking whether it is *double* would make a sulfonic acid's three
//!   oxygens answer differently depending on which carried the hydrogen.
//! * nothing else. A triple bond is not frozen: erasing `C#N` could let the
//!   solver spend the nitrogen's second π unit elsewhere and make the
//!   shortfall up with a hydrogen, but that comes from the *order* the bonds
//!   are offered their π units in and is handled there -- see [`normalize`].
//!
//! # What is proved, and what is convention
//!
//! Proved: the key is constant on a class; a representative always exists,
//! because the input molecule is itself a feasible point of the constraint
//! system; and the representative is unique, being the lexicographic optimum
//! over a finite set under a canonical order.
//!
//! Convention, and not provable: which shifts count as tautomerism at all, and
//! which tautomer of a class is the preferred one. Both are choices, made here
//! to agree with InChI's mobile-H layer and RDKit's tautomer hash; they are
//! implemented deterministically, so the answer is still a function of the
//! molecule alone.
//!
//! The invariance argument covers **heteroatom endpoints only**. With
//! [`TautomerRules::carbon_endpoints`] a carbon's hydrogens become mobile, and
//! perception reads them -- a saturated carbon and an sp2 one are told apart by
//! exactly that count -- so the construction no longer sees identical input
//! from every member of a class. Carbon endpoints remain useful and remain
//! tested, but they are validated by measurement, not by the argument above.

use crate::graph_algorithms::mol_canonization::{Canonizer, reorder_molecule, with_canonizer};
use crate::molecule::bond::BondOrder;
use crate::molecule::chirality::ChiralClass;
use crate::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Which structures count as tautomers of each other.
///
/// The default answers "the same compound with its hydrogens moved": every
/// tautomer, every zwitterion and its neutral form, and every way a salt can
/// share its protons out between its parts -- `CN.Cl` and `C[NH3+].[Cl-]` are
/// one record. A hydrogen is only ever moved, never added or taken away, so
/// the formula and the net charge of the whole record stay what they were:
/// acetate and acetic acid are two compounds, and so are `[CH3-]` and methane.
/// [`TautomerRules::charge_reduced`] asks the coarser question that does not
/// count protons, and [`TautomerRules::heteroatom_only`] the finer one the key
/// used to answer by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TautomerRules {
    /// Whether carbon may be a tautomeric endpoint, which is what merges the
    /// keto and enol forms of a ketone.
    ///
    /// Off by default, matching InChI's mobile-H layer and RDKit's tautomer
    /// hash: those treat acetone and prop-1-en-2-ol as different compounds, and
    /// a key that merged them would disagree with every registry it is compared
    /// against. Turn it on when the equivalence wanted is "the same compound
    /// after any proton shift" rather than "the same compound as a registry
    /// would say". Experimental: carbon conducts wherever it has room, so
    /// alkyl hydrogens move too.
    pub carbon_endpoints: bool,
    /// Whether a heteroatom counts as an endpoint even where nothing in its own
    /// conjugated system can exchange with it -- which is what merges glycine's
    /// zwitterion with its neutral form.
    ///
    /// With it on every conjugatable heteroatom is an endpoint, its hydrogens
    /// join the component's pool, and `NCC(=O)O` and `[NH3+]CC(=O)[O-]` are
    /// one compound. The relation is not tautomerism but *ionization*: which of
    /// a molecule's acidic and basic sites happens to be holding the proton.
    /// The formula is still conserved: pyridine and the pyridinium cation differ
    /// by a hydrogen, and the pooled totals say so.
    ///
    /// [`TautomerRules::whole_record`] implies it for the key. Set on its own it
    /// pools each component separately, and it is what [`normalize`] reads.
    pub ionizable_endpoints: bool,
    /// Whether the hydrogens of every component of a record are one pool, so
    /// that a proton can move from one part of a salt to another: `CN.Cl` and
    /// `C[NH3+].[Cl-]`, or a carboxylic acid beside an amine and the ammonium
    /// carboxylate. On by default.
    ///
    /// Implies [`TautomerRules::ionizable_endpoints`], and admits the bare ions
    /// a salt is made of -- a free halide, `[NH4+]`, `[OH-]` -- as groups of
    /// their own, since those are where such a proton goes. A metal cation has
    /// no proton to give and keeps its charge. Only the key reads it;
    /// [`normalize`] works one component at a time.
    pub whole_record: bool,
    /// Whether the key counts a proton gained or lost as no change at all, so
    /// that acetic acid and acetate, and an amine and its ammonium ion, share
    /// one key.
    ///
    /// Adding a proton raises a group's hydrogens and its charge by one each,
    /// so their difference is what every protonation state of the group has in
    /// common, and that difference is all the key records of them. Charges no
    /// proton can take away -- a quaternary nitrogen's, a metal's -- are not in
    /// any group and stay in the skeleton. Protons are only gained or lost at
    /// the sites a hydrogen can move between, so `[CH3-]` is still not methane.
    /// Implies [`TautomerRules::ionizable_endpoints`]. Keys made this way start
    /// with `p/`.
    ///
    /// Only the key reads it. [`normalize`] still keeps the charges it was
    /// given.
    pub charge_reduced: bool,
    /// How many atoms a t-link path may pass through between its endpoints.
    ///
    /// One admits the 1,3-shift, three the 1,5-shift as well; three is the
    /// default. Raising it merges more and costs more -- the path search is
    /// exponential in this bound, held down in practice only by how few atoms
    /// carry an alternating path at all.
    pub max_intermediate_atoms: usize,
}

impl Default for TautomerRules {
    fn default() -> Self {
        Self {
            carbon_endpoints: false,
            ionizable_endpoints: false,
            whole_record: true,
            charge_reduced: false,
            max_intermediate_atoms: 3,
        }
    }
}

impl TautomerRules {
    /// What a key made under these rules is prefixed with: nothing under the
    /// default rules, so a record with nothing mobile keys to its plain
    /// canonical SMILES, and otherwise the rules that differ: `c` for carbon
    /// endpoints, then `p` for charge reduction, `i` for ionizable endpoints
    /// pooled per component, or `h` for heteroatom tautomers only.
    ///
    /// Keys made under different rules describe different relations, and one
    /// compared against another means nothing; the marker stops that from
    /// passing silently.
    fn key_prefix(&self) -> String {
        let carbon = if self.carbon_endpoints { "c" } else { "" };
        let charges = if self.charge_reduced {
            "p"
        } else if self.whole_record {
            ""
        } else if self.ionizable_endpoints {
            "i"
        } else {
            "h"
        };
        if carbon.is_empty() && charges.is_empty() {
            String::new()
        } else {
            format!("{carbon}{charges}/")
        }
    }

    /// The default rules: heteroatom endpoints up to the 1,5-shift, every
    /// ionizable site, and one pool for the whole record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Heteroatom tautomers only, each component on its own: a zwitterion and
    /// its neutral form, or the two halves of a salt with the proton on either,
    /// key apart. What the default key was before it pooled the whole record.
    /// Keys start with `h/`.
    pub fn heteroatom_only() -> Self {
        Self {
            whole_record: false,
            ..Self::default()
        }
    }

    /// The default rules with carbon admitted as an endpoint, merging keto and
    /// enol forms.
    pub fn with_carbon_endpoints() -> Self {
        Self {
            carbon_endpoints: true,
            ..Self::default()
        }
    }

    /// The default rules with protons gained or lost counted as no change:
    /// one key for every protonation state of a compound.
    pub fn with_charge_reduction() -> Self {
        Self {
            charge_reduced: true,
            ..Self::default()
        }
    }

    /// Every heteroatom admitted as an endpoint, each component pooled on its
    /// own -- a zwitterion with its neutral form, but not one part of a salt
    /// with another. Keys start with `i/`.
    pub fn with_ionizable_endpoints() -> Self {
        Self {
            ionizable_endpoints: true,
            whole_record: false,
            ..Self::default()
        }
    }

    /// How many endpoint elements a conjugated component needs before it is a
    /// group.
    ///
    /// Two, ordinarily: one endpoint has nothing within reach to exchange with,
    /// and a group of one would erase a bond order and a hydrogen count for
    /// nothing. With ionizable endpoints it is one, because the pool the
    /// hydrogen moves into is the whole component's, or the whole record's, and
    /// the site it came from need not have a partner of its own.
    fn endpoints_for_a_group(&self) -> usize {
        if self.ionizable() { 1 } else { 2 }
    }

    /// The rules [`normalize`] works under. Pooling the whole record and
    /// reducing charges are properties of the key alone: a structure is
    /// normalized one component at a time and keeps the charges it was given.
    /// So the default normalizes heteroatom tautomers, and only an explicit
    /// [`TautomerRules::ionizable_endpoints`] or charge reduction asks for the
    /// ionizable rules.
    fn for_structures(&self) -> Self {
        Self {
            ionizable_endpoints: self.ionizable_endpoints || self.charge_reduced,
            whole_record: false,
            charge_reduced: false,
            ..*self
        }
    }

    fn ionizable(&self) -> bool {
        self.ionizable_endpoints || self.charge_reduced || self.whole_record
    }

    /// Whether a bare ion -- a free halide, `[NH4+]` -- is a group of its own.
    /// Only the key asks: the valence model `normalize` solves has no answer
    /// for an atom that spends nothing.
    fn bare_ions(&self) -> bool {
        self.charge_reduced || self.whole_record
    }

    fn is_endpoint_element(&self, atomic_number: u8) -> bool {
        matches!(atomic_number, 7 | 8 | 16 | 34 | 52)
            || (self.carbon_endpoints && atomic_number == 6)
            || (self.bare_ions() && is_halogen(atomic_number))
    }
}

fn is_halogen(atomic_number: u8) -> bool {
    matches!(atomic_number, 9 | 17 | 35 | 53)
}

/// One set of atoms across which hydrogens and negative charges move.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TautomericGroup {
    /// The atoms that can carry a mobile hydrogen, ascending.
    pub endpoints: Vec<usize>,
    /// Every atom the group covers -- its endpoints and the intermediates its
    /// t-link paths run through -- ascending.
    pub atoms: Vec<usize>,
    /// The bonds whose order the group makes meaningless, each `(low, high)`,
    /// ascending.
    pub bonds: Vec<(usize, usize)>,
    /// Hydrogens on this group's endpoints, as the molecule in hand writes
    /// them.
    ///
    /// Descriptive rather than conserved: what a key records, and what the
    /// solver is asked to place, is the sum over the component's groups -- see
    /// [`TautomericGroups::totals`].
    pub mobile_hydrogens: u32,
    /// Formal charge on the endpoints, which moves with the hydrogens.
    pub mobile_charge: i32,
    /// Unpaired electrons on the endpoints, which the shift moves between them
    /// exactly as it moves the hydrogens.
    ///
    /// `[NH]C#N` and `N=C=[N]` are one compound written with the electron on
    /// either nitrogen, so where it sits is not a fact about the compound and
    /// how many there are is. An electron on an atom the group does not count
    /// as an endpoint -- a carbon radical under heteroatom rules -- is not
    /// mobile and stays where the input put it, which is the same line drawn
    /// for that carbon's hydrogens.
    pub mobile_radicals: u32,
}

/// The tautomeric groups of a molecule.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TautomericGroups {
    groups: Vec<TautomericGroup>,
    group_of: Vec<Option<u32>>,
    /// The valence each atom of the *skeleton* is using, where that can be
    /// counted exactly. Measured before erasure and carried along because
    /// nothing downstream can recover it: the skeleton has had the orders it
    /// would be read from taken out. `None` where an aromatic bond makes the
    /// count fractional, and the valence table answers instead.
    used_valences: Vec<Option<i64>>,
}

impl TautomericGroups {
    /// The valence an atom of the skeleton is using, where it could be counted.
    pub fn used_valence(&self, atom: usize) -> Option<i64> {
        self.used_valences.get(atom).copied().flatten()
    }
}

impl TautomericGroups {
    /// The groups, in ascending order of their lowest atom index.
    ///
    /// That order is a property of the atom numbering, not of the molecule;
    /// [`tautomer_key`] renumbers them canonically before writing them out.
    pub fn groups(&self) -> &[TautomericGroup] {
        &self.groups
    }

    /// The group `atom` belongs to, if any.
    pub fn group_of_atom(&self, atom: usize) -> Option<usize> {
        self.group_of
            .get(atom)
            .copied()
            .flatten()
            .map(|group| group as usize)
    }

    /// Whether the molecule has no tautomeric group, in which case its tautomer
    /// key is its ordinary canonical SMILES.
    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// What the whole component conserves: the hydrogens, the charge and the
    /// unpaired electrons of every endpoint it has, wherever they sit.
    ///
    /// The totals are pooled across the groups rather than kept one per group
    /// because a hydrogen moves between groups as readily as within one. Two
    /// acidic sites on one molecule exchange a proton through the solvent, not
    /// through a π system, and the structures that result are protomers of one
    /// compound: glycine's zwitterion and its neutral form, a succinate
    /// monoanion written with the charge on either carboxyl. Both InChI and
    /// RDKit's tautomer hash give each of those pairs one identifier, and
    /// keeping the totals per group gave them two.
    ///
    /// Nothing else merges. An endpoint is a heteroatom in a conjugated system
    /// that has a second endpoint beside it, so a pooled hydrogen can only ever
    /// move between sites that are already acidic or basic -- never onto an
    /// alcohol, whose component holds one heteroatom and is therefore no group
    /// at all. A difference confined to those sites is a difference of
    /// protonation, which is what this decides to call one compound.
    ///
    /// Components are still separate. A hydrogen chloride salt and its free
    /// base keep their two keys, because [`tautomer_key_from_smiles`] keys each
    /// component on its own and the formula differs between them.
    pub fn totals(&self) -> Totals {
        self.groups
            .iter()
            .fold((0, 0, 0), |(hydrogens, charge, radicals), group| {
                (
                    hydrogens + group.mobile_hydrogens,
                    charge + group.mobile_charge,
                    radicals + group.mobile_radicals,
                )
            })
    }

    /// Every group of the component as one, which is the instance the solver
    /// works on.
    ///
    /// The groups are disjoint sets of atoms, so the union is a group in the
    /// only senses [`plan_group`] uses: a vertex set, the bonds whose orders
    /// are its to place, the endpoints that may take what the pools hold. What
    /// pooling buys is that the pools are the component's rather than each
    /// group's, so a hydrogen the input wrote on one group may come back on
    /// another -- see [`TautomericGroups::totals`].
    fn pooled(&self) -> TautomericGroup {
        let (mobile_hydrogens, mobile_charge, mobile_radicals) = self.totals();
        let mut pooled = TautomericGroup {
            endpoints: Vec::new(),
            atoms: Vec::new(),
            bonds: Vec::new(),
            mobile_hydrogens,
            mobile_charge,
            mobile_radicals,
        };
        for group in &self.groups {
            pooled.endpoints.extend(group.endpoints.iter().copied());
            pooled.atoms.extend(group.atoms.iter().copied());
            pooled.bonds.extend(group.bonds.iter().copied());
        }
        pooled.endpoints.sort_unstable();
        pooled.endpoints.dedup();
        pooled.atoms.sort_unstable();
        pooled.atoms.dedup();
        pooled.bonds.sort_unstable();
        pooled.bonds.dedup();
        pooled
    }
}

/// A disjoint-set forest over atoms, used to close the t-link relation.
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, mut vertex: usize) -> usize {
        while self.parent[vertex] != vertex {
            self.parent[vertex] = self.parent[self.parent[vertex]];
            vertex = self.parent[vertex];
        }
        vertex
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.parent[a.max(b)] = a.min(b);
        }
    }
}

/// The hydrogens bonded to `atom`.
fn hydrogen_count<M: Molecule + ?Sized>(molecule: &M, atom: usize) -> u32 {
    let Some(bonds) = molecule.get_atom_bonds(atom) else {
        return 0;
    };
    bonds
        .iter()
        .filter(|bond| is_mobile_hydrogen(molecule, bond.target()))
        .count() as u32
}

/// Whether this atom is a hydrogen the shift may move.
///
/// A labelled one it may not. `[2H]N=C(N)N` normalized to `NC(N)=N`: the
/// deuterium went into the pool with the rest, and what came back out was an
/// ordinary hydrogen, so a labelled compound silently became an unlabelled one.
/// The molecular formula did not notice, because a formula counts elements and
/// both are hydrogen.
///
/// Keeping it where it is says the truer thing. A label is put on a molecule to
/// mark *that* position, and a structure with the label on another site is a
/// different compound to whoever made it -- so the two keep their two keys, and
/// the label stays on the atom the input put it on. What still moves freely is
/// every unlabelled hydrogen beside it.
fn is_mobile_hydrogen<M: Molecule + ?Sized>(molecule: &M, atom: usize) -> bool {
    molecule.atomic_numbers()[atom] == 1 && molecule.get_isotope(atom).is_none()
}

/// Whether the bond is a sulfonyl, sulfinyl or phosphoryl oxygen's.
///
/// `S=O` on a sulfur with three σ bonds already does not become `S-OH` under
/// any tautomerism a registry recognises, so the bond is not a group's to
/// erase. The test is on degrees only, never on the order the bond happens to
/// be written with: asking whether it is *double* would make a sulfonic acid's
/// three oxygens answer differently depending on which of them was carrying the
/// hydrogen, and the same compound would key two ways.
/// A copy of `molecule` with every charge-separated oxide on a phosphorus,
/// sulfur, arsenic, selenium, antimony or tellurium written double-bonded
/// instead: `C[S+](C)[O-]` as `CS(C)=O`, `[S+2]([O-])([O-])` as `S(=O)=O`,
/// `C[P+](C)(C)[O-]` as `CP(C)(C)=O`. `None` when there is none to rewrite.
///
/// The two writings are one compound, but the bond between such an oxygen and
/// its centre is frozen -- see [`is_chalcogen_oxide`] -- so its order survives
/// erasure, and DMSO written either way had two keys and two canonical
/// tautomers. Nitrogen is left alone: it has no valence of five to rewrite an
/// N-oxide into.
fn with_chalcogen_oxides_double<M: Molecule + Clone + ?Sized>(molecule: &M) -> Option<M> {
    let centre =
        |atom: usize| matches!(molecule.atomic_numbers()[atom], 15 | 16 | 33 | 34 | 51 | 52);
    let oxide = |atom: usize| {
        molecule.atomic_numbers()[atom] == 8
            && molecule.get_atom_charge(atom) == -1
            && heavy_degree(molecule, atom) == 1
    };
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for atom in 0..molecule.atomic_numbers().len() {
        let charge = molecule.get_atom_charge(atom);
        if charge <= 0 || !centre(atom) {
            continue;
        }
        let Some(bonds) = molecule.get_atom_bonds(atom) else {
            continue;
        };
        pairs.extend(
            bonds
                .iter()
                .filter(|bond| bond.bond_order() == BondOrder::Single && oxide(bond.target()))
                .take(charge as usize)
                .map(|bond| (atom, bond.target())),
        );
    }
    if pairs.is_empty() {
        return None;
    }
    let mut copy = molecule.clone();
    for (centre, oxygen) in pairs {
        set_bond_order(&mut copy, centre, oxygen, BondOrder::Double);
        *copy.get_atom_charge_mut(centre) -= 1;
        *copy.get_atom_charge_mut(oxygen) = 0;
    }
    Some(copy)
}

fn is_chalcogen_oxide<M: Molecule + ?Sized>(molecule: &M, atom: usize, target: usize) -> bool {
    let is_centre = |index: usize| {
        matches!(
            molecule.atomic_numbers()[index],
            15 | 16 | 33 | 34 | 51 | 52
        ) && heavy_degree(molecule, index) >= 3
    };
    let is_oxide =
        |index: usize| molecule.atomic_numbers()[index] == 8 && heavy_degree(molecule, index) == 1;
    (is_centre(atom) && is_oxide(target)) || (is_centre(target) && is_oxide(atom))
}

/// Whether `atom` can hold part of a π system.
///
/// Read from the σ-skeleton alone -- degrees and elements, never bond orders.
/// That is what makes perception invariant rather than merely measured to be:
/// a proton shift changes bond orders and the positions of mobile hydrogens,
/// so anything perception reads must be blind to both, and then two tautomers
/// present *identical* input to every step that follows.
///
/// A group 15 or 16 atom keeps a lone pair while it has room for one, which is
/// a statement about its heavy degree. Carbon has no lone pair to donate, so it
/// needs a free valence instead -- `heavy + hydrogens ≤ 3` -- and reading its
/// hydrogens is sound because carbon's hydrogens are not the mobile ones. That
/// last clause is where the theorem stops holding for
/// [`TautomerRules::carbon_endpoints`], and why the guarantee is claimed for
/// heteroatom endpoints only.
/// [`is_conjugatable`], told whether carbon's hydrogens may be trusted.
///
/// They may not once carbon is an endpoint: a keto-enol shift turns an sp3
/// carbon into an sp2 one, so the very count that decides whether it conducts
/// is the count the shift moves. Two forms of one compound then give components
/// of different shapes -- which is the whole invariance argument gone. Reading
/// only the heavy degree there restores it, at the cost of letting saturated
/// carbon conduct, which merges a great deal more.
fn is_conjugatable_under<M: Molecule + ?Sized>(
    molecule: &M,
    atom: usize,
    carbon_endpoints: bool,
) -> bool {
    let Some(bonds) = molecule.get_atom_bonds(atom) else {
        return false;
    };
    let hydrogens = bonds
        .iter()
        .filter(|bond| molecule.atomic_numbers()[bond.target()] == 1)
        .count();
    let heavy = bonds.len() - hydrogens;
    match molecule.atomic_numbers()[atom] {
        // Boron: three bonds and an empty p orbital, as in borepine.
        5 => heavy <= 3,
        6 => {
            if carbon_endpoints {
                heavy <= 3
            } else {
                heavy + hydrogens <= 3
            }
        }
        7 | 15 | 33 | 51 | 8 | 16 | 34 | 52 => heavy <= 3,
        _ => false,
    }
}

/// The heavy-atom neighbours of `atom`.
fn heavy_degree<M: Molecule + ?Sized>(molecule: &M, atom: usize) -> usize {
    molecule
        .get_atom_bonds(atom)
        .map(|bonds| {
            bonds
                .iter()
                .filter(|bond| molecule.atomic_numbers()[bond.target()] > 1)
                .count()
        })
        .unwrap_or(0)
}

/// Perceives the tautomeric groups of `molecule`.
///
/// A **conjugated component** is a connected component of the subgraph on the
/// π-capable atoms, not crossing a frozen bond. A component holding two or more
/// endpoint elements is a **tautomeric group**; its endpoints are those atoms,
/// and its conserved quantities are their total hydrogens and total charge.
///
/// Everything in that definition is read from the σ-skeleton -- elements,
/// degrees, and totals -- so it is blind to bond orders and to where within a
/// group the mobile hydrogens sit. Two tautomers therefore present identical
/// input at every step, and equal output is a fact about the construction
/// rather than a property to be measured.
///
/// This replaces an alternating-path walk closed to a fixed point. The walk
/// read bond orders to test its alternation, which made its answer depend on
/// which tautomer it was shown: an anilide on 8-aminoquinoline linked its
/// carbonyl to the ring nitrogen across one intermediate with the hydrogen on
/// the amide and across five once the shift had moved it into the ring, so one
/// form perceived a group and the other did not. Every guard the walk needed --
/// a path-length bound, a step budget, a cap on the closure's rounds -- is gone
/// with it, and so is the chance that one of them silently changes an answer.
///
/// The cost is a coarser relation: two heteroatoms in one conjugated component
/// are one group whether or not an alternating path joins them. That costs
/// discrimination only where a second *valid* structure exists over the same
/// skeleton and totals -- and such a structure is, by this definition,
/// a tautomer.
///
/// ```
/// use oxidane::prelude::*;
/// use oxidane::molecule::tautomer::{perceive, TautomerRules};
///
/// // Urea: the oxygen and both nitrogens share one group of four hydrogens.
/// let urea = MolecularGraph::from_smiles_single("NC(=O)N").unwrap();
/// let groups = perceive(&urea, &TautomerRules::new());
/// assert_eq!(groups.groups().len(), 1);
/// assert_eq!(groups.groups()[0].endpoints.len(), 3);
/// assert_eq!(groups.groups()[0].mobile_hydrogens, 4);
///
/// // Imidazole: the two nitrogens share one hydrogen across the ring.
/// let imidazole = MolecularGraph::from_smiles_single("c1cnc[nH]1").unwrap();
/// let groups = perceive(&imidazole, &TautomerRules::new());
/// assert_eq!(groups.groups().len(), 1);
/// assert_eq!(groups.groups()[0].mobile_hydrogens, 1);
///
/// // Benzene has one component and no heteroatom, so no group at all.
/// let benzene = MolecularGraph::from_smiles_single("c1ccccc1").unwrap();
/// assert!(perceive(&benzene, &TautomerRules::new()).is_empty());
/// ```
pub fn perceive<M: Molecule + ?Sized>(molecule: &M, rules: &TautomerRules) -> TautomericGroups {
    let atom_count = molecule.atomic_numbers().len();
    if atom_count == 0 {
        return TautomericGroups::default();
    }

    // A free halide is no part of any π system, but in a salt it is a site a
    // proton comes and goes from, and a component of one atom.
    let capable: Vec<bool> = (0..atom_count)
        .map(|atom| {
            is_conjugatable_under(molecule, atom, rules.carbon_endpoints)
                || (rules.bare_ions()
                    && is_halogen(molecule.atomic_numbers()[atom])
                    && heavy_degree(molecule, atom) == 0)
        })
        .collect();

    // The components, as a disjoint-set forest over the π-capable atoms.
    let mut union_find = UnionFind::new(atom_count);
    let mut component_bonds: Vec<(usize, usize)> = Vec::new();
    for atom in 0..atom_count {
        if !capable[atom] {
            continue;
        }
        let Some(bonds) = molecule.get_atom_bonds(atom) else {
            continue;
        };
        for bond in bonds {
            let target = bond.target();
            if target <= atom || !capable[target] || is_chalcogen_oxide(molecule, atom, target) {
                continue;
            }
            union_find.union(atom, target);
            component_bonds.push((atom, target));
        }
    }

    // A component is a group when it holds two or more atoms that could carry a
    // mobile hydrogen. One such atom has nothing to exchange with.
    let mut endpoints_of: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for atom in 0..atom_count {
        if capable[atom] && rules.is_endpoint_element(molecule.atomic_numbers()[atom]) {
            endpoints_of
                .entry(union_find.find(atom))
                .or_default()
                .push(atom);
        }
    }

    let mut classes: Vec<usize> = endpoints_of
        .iter()
        .filter(|(_, endpoints)| endpoints.len() >= rules.endpoints_for_a_group())
        .map(|(&class, _)| class)
        .collect();
    classes.sort_unstable();
    if classes.is_empty() {
        return TautomericGroups::default();
    }
    let index_of_class: FxHashMap<usize, usize> = classes
        .iter()
        .enumerate()
        .map(|(index, &class)| (class, index))
        .collect();

    let mut groups: Vec<TautomericGroup> = classes
        .iter()
        .map(|class| TautomericGroup {
            endpoints: endpoints_of.get(class).cloned().unwrap_or_default(),
            atoms: Vec::new(),
            bonds: Vec::new(),
            mobile_hydrogens: 0,
            mobile_charge: 0,
            mobile_radicals: 0,
        })
        .collect();

    for atom in 0..atom_count {
        if !capable[atom] {
            continue;
        }
        if let Some(&index) = index_of_class.get(&union_find.find(atom)) {
            groups[index].atoms.push(atom);
        }
    }
    for &(first, second) in &component_bonds {
        if let Some(&index) = index_of_class.get(&union_find.find(first)) {
            groups[index].bonds.push((first, second));
        }
    }

    for group in groups.iter_mut() {
        group.endpoints.sort_unstable();
        group.atoms.sort_unstable();
        group.bonds.sort_unstable();
    }
    // Components are disjoint, so this finds nothing to do. It is kept because
    // the invariant it defends -- one group per set of atoms -- is cheap to
    // check and expensive to lose.
    let mut groups = merge_overlapping(groups);
    // A component holding an unpaired electron is a group like any other. The
    // electron is as mobile as the hydrogens are -- `[NH]C#N` and `N=C=[N]` are
    // the same compound written with it on either nitrogen -- so it is erased
    // off the endpoints along with them, counted against the group, and put
    // back by a pool of its own; see [`plan_group`].
    //
    // Refusing the group, which is what this did before `t4`, said the smaller
    // thing that the model could then support: with no pool to move the
    // electron between endpoints, erasure left it where the input put it and
    // `normalize` swapped the two forms back and forth for ever. The cost was
    // that a radical's tautomers were never shown to be one class -- the three
    // RDKit relates for `[O]c1nccc(O)n1` came back with three keys, none of
    // them carrying a group at all.
    // A component that is a whole molecule of one atom is nobody's group. There
    // is nothing for a proton to move between, and the valence model has no
    // answer for an atom that spends nothing: `[Se+4]` arrives with four
    // charges, no bonds and no hydrogens, and asking what its selenium does
    // with a valence of six has no answer at all. Ordinary rules never reach
    // this -- a group wants two endpoints, and two endpoints in one component
    // means a bond between them -- but under
    // [`TautomerRules::ionizable_endpoints`] one endpoint is enough, and then
    // every bare ion in a salt would be asked the question.
    //
    // The key keeps them when it pools the record or reduces charges: it only
    // erases and counts, so the valence model is never asked, and a bare ion is
    // exactly where a salt's proton goes -- `[NH4+]` beside a carboxylate,
    // `[Cl-]` beside an amine.
    // `normalize` does not see this rule; see [`TautomerRules::for_structures`].
    if !rules.bare_ions() {
        groups.retain(|group| {
            group
                .atoms
                .iter()
                .any(|&atom| heavy_degree(molecule, atom) > 0)
        });
    }
    for group in groups.iter_mut() {
        adopt_component_endpoints(molecule, rules, group);
    }
    let groups = close_over_rings(molecule, groups);

    let mut group_of: Vec<Option<u32>> = vec![None; atom_count];
    for (index, group) in groups.iter().enumerate() {
        for &atom in &group.atoms {
            group_of[atom] = Some(index as u32);
        }
    }

    TautomericGroups {
        groups,
        group_of,
        // Filled by `erased_skeleton`, which is the only place that can: the
        // count has to be taken before the orders are erased.
        used_valences: Vec::new(),
    }
}

/// Adds the rest of a ring to a group that already owns part of one.
///
/// Erasure has to remove every bond order the shift can move, and a ring is
/// where it can move one the component does not own. `CN1C(=C[P+](N(C1=O)C)(Cl)Cl)Cl`
/// is the case: its phosphorus carries four heavy neighbours, so nothing
/// conducts through it and the ring's other bonds are group bonds while its two
/// are not. Erasing only what the group owned left the ring *half* relaxed --
/// no Kekulé structure to localize back to -- so the writing that arrived
/// aromatic stayed aromatic, the writing that arrived Kekulé could not become
/// so, and the same compound canonicalized to two skeletons and two keys.
///
/// Closure uses cyclic connectivity in the induced graph of chemically
/// reachable atoms. It never chooses between tied shortest paths. The atoms this adds are *not* promoted to
/// endpoints -- the endpoint set and the totals were settled from the component
/// before this runs, and they must stay a function of it. What the added atoms
/// bring is their measured valence, which is what stops the solver putting π
/// through a sulfone's sulfur or a fifth bond on that phosphorus.
///
/// Cheap to skip, and skipped for nearly every molecule: a group has to own a
/// ring bond and touch a ring bond it does not own before the ring set is worth
/// computing. An aromatic ring whose atoms all conduct is owned entire, so a
/// phenol, an anilide and 2-pyridone never reach the second line.
fn close_over_rings<M: Molecule + ?Sized>(
    molecule: &M,
    mut groups: Vec<TautomericGroup>,
) -> Vec<TautomericGroup> {
    if groups.is_empty() {
        return groups;
    }
    let (_, cyclic) = molecule.find_cyclic_bonds_and_atoms();
    if cyclic.is_empty() {
        return groups;
    }
    let touches_an_unowned_ring_bond = |group: &TautomericGroup| {
        let owned: FxHashSet<(usize, usize)> = group.bonds.iter().copied().collect();
        if !group.bonds.iter().any(|bond| cyclic.contains(bond)) {
            return false;
        }
        group.atoms.iter().any(|&atom| {
            molecule
                .get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .any(|bond| {
                    let key = crate::molecule::utils::order_tuple(atom, bond.target());
                    cyclic.contains(&key) && !owned.contains(&key)
                })
        })
    };
    if !groups.iter().any(touches_an_unowned_ring_bond) {
        return groups;
    }

    // Work in the induced graph of chemically reachable atoms, rather than
    // choosing one shortest cycle per bond (shortest paths can tie). Cyclic
    // connectivity of this graph is independent of atom and neighbour order.
    let demands_pi: Vec<bool> = (0..molecule.len())
        .map(|atom| {
            heavy_degree(molecule, atom) >= 4
                && molecule
                    .get_atom_bonds(atom)
                    .unwrap_or_default()
                    .iter()
                    .any(|bond| {
                        cyclic.contains(&crate::molecule::utils::order_tuple(atom, bond.target()))
                            && !matches!(
                                bond.bond_order(),
                                BondOrder::Single | BondOrder::Up | BondOrder::Down
                            )
                    })
        })
        .collect();
    for group in &mut groups {
        let members: FxHashSet<usize> = group.atoms.iter().copied().collect();
        let eligible: Vec<bool> = (0..molecule.len())
            .map(|atom| {
                if members.contains(&atom) {
                    return true;
                }
                let neighbours: Vec<usize> = molecule
                    .get_atom_bonds(atom)
                    .unwrap_or_default()
                    .iter()
                    .map(|bond| bond.target())
                    .filter(|&target| {
                        cyclic.contains(&crate::molecule::utils::order_tuple(atom, target))
                    })
                    .collect();
                let reached = neighbours
                    .iter()
                    .filter(|target| members.contains(target))
                    .count();
                reached >= 2
                    || (demands_pi[atom] && reached > 0)
                    || (molecule.atomic_numbers()[atom] == 6
                        && heavy_degree(molecule, atom) <= 3
                        && neighbours.iter().any(|&target| demands_pi[target]))
            })
            .collect();
        let mut reachable = crate::molecule::MolecularGraph::default();
        for &element in molecule.atomic_numbers() {
            reachable.add_atom(crate::atom::Atom::new(element));
        }
        for &(first, second) in &cyclic {
            if eligible[first] && eligible[second] {
                reachable.add_bond(first, second, BondOrder::Single);
            }
        }
        let (_, reachable_cycles) = reachable.find_cyclic_bonds_and_atoms();
        let mut components = UnionFind::new(molecule.len());
        for &(first, second) in &reachable_cycles {
            components.union(first, second);
        }
        let touched: FxHashSet<usize> = group
            .bonds
            .iter()
            .filter(|bond| reachable_cycles.contains(bond))
            .map(|&(first, _)| components.find(first))
            .collect();
        let mut owned: FxHashSet<(usize, usize)> = group.bonds.iter().copied().collect();
        let mut atoms = members;
        for (first, second) in reachable_cycles {
            if touched.contains(&components.find(first)) {
                owned.insert((first, second));
                atoms.insert(first);
                atoms.insert(second);
            }
        }
        group.bonds = owned.into_iter().collect();
        group.bonds.sort_unstable();
        group.atoms = atoms.into_iter().collect();
        group.atoms.sort_unstable();
    }

    // Two groups reaching into one ring are one group, and their totals add:
    // the ring is a single conjugated system and the hydrogens either side of
    // it move within it.
    let mut merged = merge_overlapping_totals(groups);
    merged.sort_by_key(|group| group.atoms.first().copied().unwrap_or(usize::MAX));
    merged
}

/// [`merge_overlapping`] for groups whose totals have already been counted.
fn merge_overlapping_totals(groups: Vec<TautomericGroup>) -> Vec<TautomericGroup> {
    let mut union_find = UnionFind::new(groups.len());
    let mut owner: FxHashMap<usize, usize> = FxHashMap::default();
    for (index, group) in groups.iter().enumerate() {
        for &atom in &group.atoms {
            match owner.get(&atom) {
                Some(&first) => union_find.union(first, index),
                None => {
                    owner.insert(atom, index);
                }
            }
        }
    }
    let mut merged: FxHashMap<usize, TautomericGroup> = FxHashMap::default();
    for (index, group) in groups.into_iter().enumerate() {
        let representative = union_find.find(index);
        match merged.get_mut(&representative) {
            Some(target) => {
                target.endpoints.extend(group.endpoints);
                target.atoms.extend(group.atoms);
                target.bonds.extend(group.bonds);
                target.mobile_hydrogens += group.mobile_hydrogens;
                target.mobile_charge += group.mobile_charge;
                target.mobile_radicals += group.mobile_radicals;
            }
            None => {
                merged.insert(representative, group);
            }
        }
    }
    let mut result: Vec<TautomericGroup> = merged.into_values().collect();
    for group in result.iter_mut() {
        for list in [&mut group.endpoints, &mut group.atoms] {
            list.sort_unstable();
            list.dedup();
        }
        group.bonds.sort_unstable();
        group.bonds.dedup();
    }
    result
}

/// Fuses groups that share an atom into one.
///
/// Conjugated components are disjoint by construction, so this does nothing as
/// the perception now stands. It existed for the α-carbon shell, the one step
/// that reached outside a component; that shell is gone, because making carbon
/// conduct on its heavy degree alone subsumed it.
fn merge_overlapping(groups: Vec<TautomericGroup>) -> Vec<TautomericGroup> {
    let mut union_find = UnionFind::new(groups.len());
    let mut owner: FxHashMap<usize, usize> = FxHashMap::default();
    for (index, group) in groups.iter().enumerate() {
        for &atom in &group.atoms {
            match owner.get(&atom) {
                Some(&first) => union_find.union(first, index),
                None => {
                    owner.insert(atom, index);
                }
            }
        }
    }

    let mut merged: FxHashMap<usize, TautomericGroup> = FxHashMap::default();
    for (index, group) in groups.into_iter().enumerate() {
        let representative = union_find.find(index);
        match merged.get_mut(&representative) {
            Some(target) => {
                target.endpoints.extend(group.endpoints);
                target.atoms.extend(group.atoms);
                target.bonds.extend(group.bonds);
            }
            None => {
                merged.insert(representative, group);
            }
        }
    }

    let mut result: Vec<TautomericGroup> = merged.into_values().collect();
    for group in result.iter_mut() {
        for list in [&mut group.endpoints, &mut group.atoms] {
            list.sort_unstable();
            list.dedup();
        }
        group.bonds.sort_unstable();
        group.bonds.dedup();
    }
    result.sort_by_key(|group| group.atoms.first().copied().unwrap_or(usize::MAX));
    result
}

/// Makes every heteroatom in the group's component an endpoint, and recounts
/// the totals over them.
///
/// Which endpoints a walk *reaches* is not invariant. In one tautomer of a
/// conjugated system a t-link runs to a given nitrogen and in another it does
/// not, so the same compound comes back with endpoint sets of different sizes
/// and two keys -- with, note, identical skeletons, since the erasure covers the
/// whole component either way. Taking every heteroatom in the component instead
/// makes the endpoint set a function of the component, which the skeleton
/// already fixes.
///
/// A heteroatom that could not really hold a mobile hydrogen is swept in with
/// the rest. That costs nothing in correctness -- it carries the same hydrogens
/// in every tautomer, so it shifts every key in the class by the same amount --
/// and it costs a little discrimination, which is the same trade the component
/// -wide erasure makes.
///
/// Carbon is promoted on the same terms when
/// [`TautomerRules::carbon_endpoints`] admits it, which is why
/// [`extend_over_pi_system`] pulls the α-carbons into the component first: the
/// pooled hydrogen count has to include the one the enolization moves.
fn adopt_component_endpoints<M: Molecule + ?Sized>(
    molecule: &M,
    rules: &TautomerRules,
    group: &mut TautomericGroup,
) {
    for &atom in &group.atoms {
        if rules.is_endpoint_element(molecule.atomic_numbers()[atom]) {
            group.endpoints.push(atom);
        }
    }
    group.endpoints.sort_unstable();
    group.endpoints.dedup();
    group.mobile_hydrogens = group
        .endpoints
        .iter()
        .map(|&atom| hydrogen_count(molecule, atom))
        .sum();
    group.mobile_charge = group
        .endpoints
        .iter()
        .map(|&atom| molecule.get_atom_charge(atom) as i32)
        .sum();
    group.mobile_radicals = group
        .endpoints
        .iter()
        .map(|&atom| u32::from(molecule.radical_states().get(atom).copied().unwrap_or(0)))
        .sum();
}

/// The molecule with everything a tautomeric shift could change erased, paired
/// with the groups mapped onto its atom numbering.
///
/// Every bond inside a group becomes single, every mobile hydrogen is deleted,
/// and every endpoint's charge is zeroed. The result is a graph, not a
/// chemically valid molecule -- an amide nitrogen comes out with two bonds and
/// no hydrogens -- and exists only to be canonized.
pub fn erased_skeleton<M: Molecule + Clone + ToSMILES>(
    molecule: &M,
    rules: &TautomerRules,
) -> (M, TautomericGroups) {
    // Each atom's valence as the input actually uses it, measured before
    // anything is erased. For an atom that is not an endpoint this is fixed by
    // its element and charge -- a neutral carbon spends four however its π bond
    // is placed -- so it is invariant across a class and can be read straight
    // off the molecule. That matters because the valence *table* does not cover
    // everything a public corpus holds: PubChem writes dialkyl H-phosphonates
    // without the P-H, leaving a four-valent phosphorus, and it has explicit
    // radicals like `[CH]` and `[N]`. Asking the table for those makes the
    // group unsolvable, and an unsolvable group contradicts the existence
    // argument -- the input is a feasible point of its own constraint system,
    // so a failure means the model disagrees with the molecule.
    // Measured on a Kekule form rather than on the molecule as written. An
    // aromatic bond is worth one and a half and cannot be counted, so an atom
    // in a ring the input wrote aromatic fell back on the valence table while
    // the same atom in the same ring written Kekule was measured exactly -- two
    // different constraint systems for one compound, and `normalize` alternated
    // between the two answers on successive passes. A selenophosphorus ring
    // from PubChem is the case: `...N=C2N[P+]([Se][P+]2(...)[Se-])...` came back
    // aromatic, then Kekule, then aromatic again. Localizing first asks the
    // same question of both writings; the per-atom pi count is the same for
    // every Kekule structure a ring has, so which one localization picks does
    // not matter.
    let rewritten;
    let molecule = match with_chalcogen_oxides_double(molecule) {
        Some(copy) => {
            rewritten = copy;
            &rewritten
        }
        None => molecule,
    };
    // A group needs two endpoint elements even before conjugation is tested.
    // Avoid localization, valence measurement and a second molecule clone
    // when no assignment can possibly be made.
    let wanted = rules.endpoints_for_a_group();
    if molecule
        .atomic_numbers()
        .iter()
        .filter(|&&element| rules.is_endpoint_element(element))
        .take(wanted)
        .count()
        < wanted
    {
        return (molecule.clone(), TautomericGroups::default());
    }
    let mut measured = molecule.clone();
    let _ = measured.localize_aromatic_bonds();
    let used: Vec<Option<i64>> = (0..measured.atomic_numbers().len())
        .map(|atom| used_valence(&measured, atom))
        .collect();
    // The skeleton is the Kekule form too. Erasure only touches the bonds a
    // group owns, so the ones it does not own keep whatever the input wrote --
    // aromatic in one writing of the selenophosphorus ring and single-or-double
    // in the other, which left two skeletons for one compound and two keys.
    // Perceived on the same Kekule form the valences were measured on, and for
    // the same reason: conjugation is read off bond orders, and a ring the
    // input wrote aromatic offers none to read. The selenophosphorus ring gave
    // a group covering the whole ring when it arrived Kekule and a smaller one
    // when it arrived aromatic, so the erasure differed and the two writings of
    // one compound got two keys.
    let groups = perceive(&measured, rules);
    // The Kekule form is the skeleton: perception has read everything it needs
    // from it, so it is taken over rather than copied.
    let mut skeleton = measured;
    if groups.is_empty() {
        let mut groups = groups;
        groups.used_valences = used;
        return (skeleton, groups);
    }

    // Bond orders first, while the atom numbering still matches the groups.
    // Every bond of the group goes to single, aromatic ones included: a group
    // covers a whole π system, and whether that system is written aromatic is
    // one of the things a shift changes. Aromaticity is read back off the bond
    // orders, so this clears the atoms' aromatic flags with them.
    // Which bonds the erasure actually moves, as opposed to merely owns. A
    // cis/trans class is stored against the *first single bond* at its atom, so
    // a neighbouring bond dropping from double to single changes which
    // neighbour that is and silently reverses what the class says; see
    // `settle_geometries`.
    let mut moved: Vec<(usize, usize)> = Vec::new();
    for group in groups.groups() {
        for &(first, second) in &group.bonds {
            if skeleton
                .get_atom_bonds(first)
                .unwrap_or_default()
                .iter()
                .any(|bond| bond.target() == second && bond.bond_order() != BondOrder::Single)
            {
                moved.push((first, second));
            }
            set_bond_single(&mut skeleton, first, second);
        }
        for &endpoint in &group.endpoints {
            *skeleton.get_atom_charge_mut(endpoint) = 0;
            // The unpaired electrons go with the charge and for the same
            // reason: which endpoint holds one is what distinguishes two
            // members of the class, so the skeleton must not record it. The
            // count is recorded against the group instead.
            skeleton.set_atom_radical_state(endpoint, 0);
        }
    }

    // Then the mobile hydrogens. Removing an atom shifts every index above it,
    // so they go in descending order and the groups are remapped afterwards.
    let mut mobile: Vec<usize> = Vec::new();
    for group in groups.groups() {
        for &endpoint in &group.endpoints {
            let Some(bonds) = skeleton.get_atom_bonds(endpoint) else {
                continue;
            };
            for bond in bonds {
                if is_mobile_hydrogen(&skeleton, bond.target()) {
                    mobile.push(bond.target());
                }
            }
        }
    }
    mobile.sort_unstable();
    mobile.dedup();
    for &hydrogen in mobile.iter().rev() {
        skeleton.remove_atom(hydrogen);
    }

    let mut kept: Vec<Option<i64>> = Vec::with_capacity(used.len() - mobile.len());
    for (atom, valence) in used.into_iter().enumerate() {
        if mobile.binary_search(&atom).is_err() {
            kept.push(valence);
        }
    }
    let mut groups = remap(groups, &mobile);
    groups.used_valences = kept;
    let shift = |atom: usize| atom - mobile.partition_point(|&hydrogen| hydrogen < atom);
    let moved: FxHashSet<(usize, usize)> = moved
        .into_iter()
        .map(|(first, second)| {
            (
                shift(first).min(shift(second)),
                shift(first).max(shift(second)),
            )
        })
        .collect();
    settle_geometries(&mut skeleton, &mut groups, &moved);
    (skeleton, groups)
}

/// Puts back the orders no shift can move, and erases the geometry of the rest.
///
/// The geometry of a double bond a shift can move goes with its order. Nothing
/// holds the substituents on their side of the bond while it moves, so cis and
/// trans are not conserved and a key recording them would separate structures
/// the shift relates. Erasing the order while keeping the tag was an
/// inconsistency the corpus found: the key read a geometry off the input that
/// `normalize` then destroyed, and the two disagreed. Only the cis/trans tags
/// go -- a tetrahedral centre inside a group is real and is conserved.
///
/// What that argument does not cover is a bond the shift *cannot* move, and
/// erasing those as well was throwing away chemistry. A tertiary cinnamamide
/// has no mobile hydrogen at all -- there is no N-H and no O-H, and the group's
/// hydrogen total is zero -- so no assignment but the one it arrived with
/// exists, and yet its C=C came back geometry-free and the E and Z compounds
/// got one key and one normalized structure. Over a PubChem sample 94% of the
/// molecules that state a double-bond geometry lost it, a third of them with
/// nothing mobile anywhere in the group. RDKit's `HetAtomTautomer` erases these
/// too, which is why the oracle agreed; its v2 keeps them.
///
/// So every bond of every group is asked whether the group admits any other
/// order for it ([`fixed_order`]), and one that does not is put back as it was
/// and dropped from the group: it is not a bond the shift can move, so the
/// solver reads its π as committed rather than as its to place, and a geometry
/// written on it stands. The question is asked of *every* bond and not only of
/// the ones carrying a geometry, because which bonds carry one is a fact about
/// the writing rather than about the compound: `normalize` states geometries
/// where the input stated none and the other way about, so a skeleton that
/// depended on them stopped being a function of the compound. It cost 39 of 92
/// molecules their idempotence -- chlorophylls, fumarate esters, anything with
/// a geometry inside a large conjugated system -- and the keys to match.
///
/// A geometry needs more than a bond that stays put, so a tag is kept only when
///
/// * neither end is an endpoint, since an endpoint takes a hydrogen or a charge
///   and gives it back, and a geometry written about an atom whose substituents
///   change describes an arrangement the next member of the class does not
///   have, and
/// * neither end carries a second double bond -- a cumulene centre ends two,
///   and which of them its class is about is not a question this needs to
///   answer.
///
/// Both conditions read the skeleton and the groups, so both are as much a
/// function of the compound as the rest of the erasure is.
fn settle_geometries<M: Molecule + Clone + ToSMILES>(
    skeleton: &mut M,
    groups: &mut TautomericGroups,
    moved: &FxHashSet<(usize, usize)>,
) {
    // Asked of the fully erased skeleton, before any of the answers are
    // applied: what a group can do with one of its bonds is a property of the
    // whole group, and answering them one at a time against a skeleton the
    // earlier answers had already changed would make the set depend on the
    // order they were asked in.
    let used = groups.used_valences.clone();
    // Whether an assignment can move a bond is a question about the whole
    // component's feasible set, which is one instance now that the pools are
    // shared between the groups -- and one instance answers every bond, since
    // only the order the edges are offered in differs between the questions.
    let pooled = groups.pooled();
    let mut instance = LazyInstance::new();
    let (cyclic_atoms, cyclic) = skeleton.find_cyclic_bonds_and_atoms();
    let mut fixed: Vec<((usize, usize), BondOrder)> = Vec::new();
    // Only the bonds a geometry could sit on are asked about, and every test
    // here reads the graph rather than the writing, so the set stays a function
    // of the compound: not a ring bond, not a bond with a ring atom at either
    // end (see below), not one touching an endpoint, and sp2-shaped at both
    // ends -- two σ neighbours and no more, since a third leaves nothing for a
    // geometry to distinguish. Probing every bond instead costs two solver runs
    // apiece and slowed the corpus sweep by an order of magnitude for answers
    // no geometry was ever going to use.
    // A geometry is stored as a class on each end, and what a class means is
    // read off the *first single bond* at its atom. Erasing a neighbouring
    // double bond puts a new bond at the head of that list, so the class goes
    // on being believed while meaning the opposite -- and which neighbour comes
    // first is a fact about the order the input was written in, so one molecule
    // written two ways got two keys, one `/` and one `\`. Phosphorus ylides are
    // where PubChem has it: `CC(C)(C)/C=P(=NC(C)(C)C)/N(C(C)(C)C)C(C)(C)C` has
    // its P=N erased out from under the mark on its P=C.
    let neighbourhood_stands = |atom: usize, bond: (usize, usize)| {
        skeleton
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .all(|other| {
                let edge = (atom.min(other.target()), atom.max(other.target()));
                edge == bond || !moved.contains(&edge)
            })
    };
    let stereogenic_shaped = |atom: usize| {
        skeleton
            .get_atom_bonds(atom)
            .map(|bonds| bonds.len() == 3 || bonds.len() == 2)
            .unwrap_or(false)
    };
    for group in groups.groups() {
        for &bond in &group.bonds {
            if !cyclic.contains(&bond)
                && !cyclic_atoms.contains(bond.0)
                && !cyclic_atoms.contains(bond.1)
                && !group.endpoints.contains(&bond.0)
                && !group.endpoints.contains(&bond.1)
                && stereogenic_shaped(bond.0)
                && stereogenic_shaped(bond.1)
                && neighbourhood_stands(bond.0, bond)
                && neighbourhood_stands(bond.1, bond)
                && let Some(solver) = instance.get(skeleton, &pooled, &used)
                && let Some(order) = fixed_order(skeleton, solver, bond)
                && order != BondOrder::Single
            {
                fixed.push((bond, order));
            }
        }
    }
    // Put those orders back and take the bonds out of the group. A bond whose
    // order no assignment can change is already the function of the compound
    // that erasure exists to make it, so writing it down hides nothing -- and
    // the geometry on it needs it: the canonizer drops a cis/trans class whose
    // bond is not double, so a tag left on an erased bond never reaches the
    // answer `normalize` writes. The bond leaves the group's own list with its
    // order, so the solver reads that π as committed rather than as its to
    // place.
    //
    // Every fixed bond is put back, not only the ones carrying a geometry:
    // which bonds carry one is a fact about the writing rather than about the
    // compound, and a skeleton that depended on it stopped being a function of
    // the compound -- 39 of 92 molecules lost their idempotence to that, and
    // their keys with it.
    // A ring bond is left erased, however fixed it is. The ring is what the
    // erasure relaxes hardest: the solver may write any Kekule structure the
    // ring has, and `place_the_totals` prefers the candidate that aromatizes
    // it, after which the bond is written aromatic and a geometry on it cannot
    // be expressed at all. `normalize` then hands back a structure with no tag
    // where its input had one, the next pass erases a different skeleton, and
    // the answer moves -- 32 molecules of a PubChem sample, every one of them a
    // geometry inside a ring its group could aromatize. Nothing is lost that
    // the erasure was not already taking: the compounds this exists for --
    // cinnamamide, fumarate, a benzylidene -- all state their geometry on a
    // bond outside any ring.
    let fixed: Vec<((usize, usize), BondOrder)> = fixed
        .into_iter()
        .filter(|(bond, _)| !cyclic.contains(bond))
        .collect();
    for &(bond, order) in &fixed {
        set_bond_order(skeleton, bond.0, bond.1, order);
        for group in groups.groups.iter_mut() {
            group.bonds.retain(|&edge| edge != bond);
        }
    }
    let fixed: FxHashSet<(usize, usize)> = fixed.into_iter().map(|(bond, _)| bond).collect();

    // Which atoms may go on asserting a geometry. A bond no group owns was
    // never erased and keeps its order whatever the shift does, so it counts
    // here alongside the ones the group cannot move.
    let endpoint = |atom: usize| {
        groups
            .groups()
            .iter()
            .any(|group| group.endpoints.contains(&atom))
    };
    // A cumulene centre ends two double bonds and carries one class, and which
    // of them it is about is not a question this needs to answer.
    let cumulene_free = |molecule: &M, fixed: &FxHashSet<(usize, usize)>, atom: usize| {
        molecule
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .filter(|bond| {
                let edge = (atom.min(bond.target()), atom.max(bond.target()));
                fixed.contains(&edge) || bond.bond_order() == BondOrder::Double
            })
            .count()
            == 1
    };
    let mut kept: FxHashSet<usize> = FxHashSet::default();
    for group in groups.groups() {
        for &atom in &group.atoms {
            for bond in skeleton.get_atom_bonds(atom).unwrap_or_default() {
                let target = bond.target();
                let edge = (atom.min(target), atom.max(target));
                // Either a bond the group cannot move, or one it never owned:
                // the group's own are single here, so a double bond at one of
                // its atoms is one it does not own.
                let stays = !cyclic.contains(&edge)
                    && !cyclic_atoms.contains(atom)
                    && !cyclic_atoms.contains(target)
                    && (fixed.contains(&edge) || bond.bond_order() == BondOrder::Double);
                if stays
                    && !endpoint(atom)
                    && !endpoint(target)
                    && cumulene_free(skeleton, &fixed, atom)
                    && cumulene_free(skeleton, &fixed, target)
                {
                    kept.insert(atom);
                    kept.insert(target);
                }
            }
        }
    }

    // Everything else a group touches loses its geometry. Both ends, since a
    // geometry is written on both -- and the far end need not be in the group.
    // `N(=O)/N=Cl(=O)/Cl` is the case: the marked `N=Cl` has its nitrogen in
    // the group and its chlorine outside, so clearing the group's own atoms
    // left the chlorine asserting a geometry about a bond whose other end no
    // longer had one -- a tag that `normalize` drops and the key kept, which is
    // exactly the disagreement this erasure exists to prevent.
    let mut carriers: Vec<usize> = Vec::new();
    for group in groups.groups() {
        carriers.extend(group.atoms.iter().copied());
        for &atom in &group.atoms {
            for bond in skeleton.get_atom_bonds(atom).unwrap_or_default() {
                if bond.bond_order() == BondOrder::Double {
                    carriers.push(bond.target());
                }
            }
        }
    }
    for atom in carriers {
        if !kept.contains(&atom)
            && matches!(
                skeleton.get_chiral_class(atom),
                ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
            )
            && let Some(classes) = skeleton.chiral_classes_mut()
        {
            classes[atom] = ChiralClass::None;
        }
    }

    // A carbon endpoint's tetrahedral tag goes too. The hydrogen that makes it
    // a stereocentre is one of the mobile ones: shifted away, the carbon is
    // sp2 and the centre is gone, and shifted back it can return from either
    // face. So `CC[C@H](C)C(=O)C`, its enantiomer and their shared enol are one
    // tautomer class, and the key gave them three strings. A heteroatom
    // endpoint keeps its tag -- a sulfoxide sulfur is a stereocentre by its
    // lone pair, not by a hydrogen, and normalization pins it as one.
    for group in groups.groups() {
        for &atom in &group.endpoints {
            if skeleton.atomic_numbers()[atom] == 6
                && matches!(
                    skeleton.get_chiral_class(atom),
                    ChiralClass::TH(_) | ChiralClass::AL(_)
                )
                && let Some(classes) = skeleton.chiral_classes_mut()
            {
                classes[atom] = ChiralClass::None;
            }
        }
    }
}

/// The valence an atom spends in the molecule as written: its bonds, counted
/// with their orders, plus whatever unpaired electrons it carries.
///
/// `None` where the count is not an integer -- an aromatic bond is worth one
/// and a half, so an aromatic `CH` would come out as three rather than four,
/// and asking the group to satisfy three leaves half a ring unsolvable. Those
/// atoms fall back to the valence table, which is exactly where the table is
/// reliable: the cases it does *not* cover -- explicit radicals, PubChem's
/// H-phosphonates written without the P-H, the odd metal -- are never aromatic.
fn used_valence<M: Molecule + ?Sized>(molecule: &M, atom: usize) -> Option<i64> {
    let bonds = molecule.get_atom_bonds(atom)?;
    let mut bonded = 0i64;
    for bond in bonds {
        bonded += match bond.bond_order() {
            BondOrder::Single | BondOrder::Up | BondOrder::Down | BondOrder::Coordinate => 1,
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
            BondOrder::Quadruple => 4,
            BondOrder::Aromatic => return None,
        };
    }
    Some(bonded + molecule.radical_states().get(atom).copied().unwrap_or(0) as i64)
}

/// Sets the bond between two atoms to single from both sides.
fn set_bond_single<M: Molecule + ?Sized>(molecule: &mut M, first: usize, second: usize) {
    for (from, to) in [(first, second), (second, first)] {
        if let Some(bonds) = molecule.atom_bonds_mut().get_mut(from) {
            for bond in bonds.iter_mut() {
                if bond.target == to {
                    bond.bond_order = BondOrder::Single;
                }
            }
        }
    }
}

/// Shifts group atom indices down past the removed hydrogens.
fn remap(mut groups: TautomericGroups, removed: &[usize]) -> TautomericGroups {
    let shift = |atom: usize| atom - removed.partition_point(|&hydrogen| hydrogen < atom);
    let mut group_of: Vec<Option<u32>> = vec![None; groups.group_of.len() - removed.len()];
    for group in groups.groups.iter_mut() {
        for atom in group.endpoints.iter_mut().chain(group.atoms.iter_mut()) {
            *atom = shift(*atom);
        }
        for bond in group.bonds.iter_mut() {
            *bond = (shift(bond.0), shift(bond.1));
        }
    }
    for (index, group) in groups.groups.iter().enumerate() {
        for &atom in &group.atoms {
            group_of[atom] = Some(index as u32);
        }
    }
    groups.group_of = group_of;
    groups
}

/// A canonical string that every tautomer of `molecule` shares.
///
/// The canonical SMILES of the erased skeleton; then one `/`-separated field
/// per tautomeric group, naming that group's endpoints by canonical atom
/// position; then one last field, `hydrogens:charge`, for the component as a
/// whole, with `:electrons` after it where any are unpaired. A molecule with no
/// tautomeric group gets its plain canonical SMILES and no suffix, so keys for
/// ordinary structures stay readable.
///
/// The totals are the component's rather than each group's because a proton
/// moves between two acidic sites as readily as within one; see
/// [`TautomericGroups::totals`] for what that merges and what it does not.
///
/// ```
/// use oxidane::prelude::*;
/// use oxidane::molecule::tautomer::{tautomer_key, TautomerRules};
///
/// let rules = TautomerRules::new();
/// let amide = MolecularGraph::from_smiles_single("CC(=O)N").unwrap();
/// let imidic = MolecularGraph::from_smiles_single("CC(O)=N").unwrap();
/// assert_eq!(tautomer_key(&amide, &rules), tautomer_key(&imidic, &rules));
///
/// // A molecule with nothing mobile keeps its ordinary canonical SMILES.
/// let benzene = MolecularGraph::from_smiles_single("c1ccccc1").unwrap();
/// assert_eq!(tautomer_key(&benzene, &rules), "c1ccccc1");
///
/// // Rules other than the default say so, so that keys from different
/// // relations are never compared by accident.
/// let carbon = TautomerRules::with_carbon_endpoints();
/// assert!(tautomer_key(&amide, &carbon).starts_with("c/"));
/// ```
pub fn tautomer_key<M: Molecule + Clone + ToSMILES>(molecule: &M, rules: &TautomerRules) -> String {
    with_canonizer(|canonizer| tautomer_key_with(canonizer, molecule, rules))
}

/// Puts back the order of every bond whose geometry survived erasure.
///
/// A surviving geometry sits on a bond [`settle_geometries`] found fixed, so
/// the order asked for here is the one the group has no choice about, and
/// writing it down states nothing the erasure was hiding.
fn restore_geometry_orders<M: Molecule + Clone + ToSMILES>(
    skeleton: &mut M,
    groups: &TautomericGroups,
) {
    let states = |molecule: &M, atom: usize| {
        matches!(
            molecule.get_chiral_class(atom),
            ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
        )
    };
    let used = groups.used_valences.clone();
    let pooled = groups.pooled();
    let mut instance = LazyInstance::new();
    let mut restored: Vec<((usize, usize), BondOrder)> = Vec::new();
    for group in groups.groups() {
        for &bond in &group.bonds {
            if states(skeleton, bond.0)
                && states(skeleton, bond.1)
                && let Some(solver) = instance.get(skeleton, &pooled, &used)
                && let Some(order) = fixed_order(skeleton, solver, bond)
                && order != BondOrder::Single
            {
                restored.push((bond, order));
            }
        }
    }
    for (bond, order) in restored {
        set_bond_order(skeleton, bond.0, bond.1, order);
    }
}

/// [`tautomer_key`], canonizing the skeleton with a caller's [`Canonizer`] so
/// that a stream of molecules pays for its buffers once.
pub fn tautomer_key_with<M: Molecule + Clone + ToSMILES>(
    canonizer: &mut Canonizer,
    molecule: &M,
    rules: &TautomerRules,
) -> String {
    let (mut skeleton, groups) = erased_skeleton(molecule, rules);
    // A geometry that survived erasure is written as a mark on a neighbouring
    // single bond, and a mark about a bond that is itself single says nothing,
    // so the key puts that bond's order back before it writes. Only the key
    // does: `normalize` solves on the skeleton, and a bond it finds already
    // written is one it no longer places, which changes the capacities the
    // solver works with and, through them, which tautomer of the class it
    // picks. The answers agree anyway -- the order being put back is the one
    // every assignment gives the bond -- so the cheaper thing to keep identical
    // is the solver's input.
    restore_geometry_orders(&mut skeleton, &groups);

    // Tag every atom with its own index so the canonical ordering can be read
    // back. Atom classes are not part of `get_atom_invariant`, so tagging
    // cannot influence which ordering canonization picks.
    let atom_count = skeleton.atomic_numbers().len();
    let mut tagged = skeleton;
    *tagged.atom_classes_mut() = Some((0..atom_count).collect());
    // The skeleton's bond orders have been rewritten, so whatever aromaticity
    // the molecule arrived with no longer describes it; `canonize` perceives it
    // again rather than trusting what is there.
    let mut canonical = canonizer.canonize(&tagged);
    let origins = canonical.atom_classes().clone();
    // The classes are emitted as `:n` in SMILES, so they go before the write.
    *canonical.atom_classes_mut() = None;
    let smiles = canonical.to_smiles();

    let Some(origins) = origins else {
        return rules.key_prefix() + &smiles;
    };
    if groups.is_empty() {
        return rules.key_prefix() + &smiles;
    }

    // Canonical position of each pre-canonical atom.
    let mut position = vec![0usize; atom_count];
    for (canonical_position, &original) in origins.iter().enumerate() {
        if original < atom_count {
            position[original] = canonical_position;
        }
    }

    // Write the groups out, choosing among the ways a symmetric skeleton allows
    // them to be written.
    //
    // A canonical labelling is unique as a labelling, but it cannot see data
    // the graph does not carry -- and the totals are exactly such data, since
    // erasure is what took them out of the graph. Hydrogen tartrate's skeleton
    // is symmetric, so nothing in it says which carboxyl holds the mobile
    // hydrogen and which the charge, and the answer fell to the order the atoms
    // arrived in: the same anion written two ways got two keys. Mapping the
    // fields through every automorphism and keeping the smallest removes the
    // choice. Groups carrying equal totals are interchangeable and give the
    // same string either way, so only groups that genuinely differ are decided
    // here.
    // The generators permute heavy atoms only -- hydrogens are not part of the
    // graph nauty saw -- so the closure is built over their own width, and a
    // position outside it is left where it is.
    // Minimising over the automorphisms costs a group enumeration, and it is
    // only ever needed when two groups carry *different* totals: groups whose
    // totals agree are interchangeable, so permuting them leaves the sorted
    // field list where it was, and a lone group has its endpoint set fixed by
    // the skeleton. Skipping the enumeration in those cases is most of the
    // molecules and about half the running time.
    let suffix = totals_suffix(
        &groups,
        &position,
        canonizer.last_graph_automorphisms(),
        rules.charge_reduced,
    );

    let mut key = rules.key_prefix();
    key.push_str(&smiles);
    key.push_str(&suffix);
    key
}

/// What a group conserves: its mobile hydrogens, its charge and its unpaired
/// electrons. Ordered so that comparing two reads the hydrogens first, which is
/// the order the key writes them in.
type Totals = (u32, i32, u32);

/// The suffix a key carries: where the groups are, and what the component
/// conserves between them.
///
/// A canonical labelling is unique as a labelling, but it cannot see data the
/// graph does not carry -- and the totals are exactly such data, since erasure
/// is what took them out of the graph. Hydrogen tartrate's skeleton is
/// symmetric, so nothing in it says which carboxyl holds the mobile hydrogen
/// and which the charge, and the answer fell to the order the atoms arrived in:
/// the same anion written two ways got two keys, and `normalize` put the charge
/// on whichever carboxyl the input had used.
///
/// `position` maps an atom of the skeleton to its canonical position; pass the
/// identity where the groups are already in canonical order.
///
/// This runs even when no two groups carry *different* totals. Groups whose
/// totals agree are interchangeable, but a lone group's endpoint set is not
/// fixed by the skeleton: a field names the endpoints by *position*, and an
/// automorphism can carry a lone group's endpoints onto a symmetric image of
/// themselves. A
/// bis-catechol crown ether is the case -- two equivalent rings, one group, and
/// the same skeleton either way -- and its field came out `0,1:0:0` from one
/// writing and `2,3:0:0` from the other.
///
/// The generators permute heavy atoms only -- hydrogens are not part of the
/// graph nauty saw -- so the closure is built over their own width, and a
/// position outside it is left where it is.
fn totals_suffix(
    groups: &TautomericGroups,
    position: &[usize],
    generators: &[Vec<usize>],
    charge_reduced: bool,
) -> String {
    // Groups an automorphism can interchange are one group as far as the
    // compound is concerned, so which of them holds the charge is not a fact
    // about the molecule and must not be a fact about the key. They are found by
    // *orbit*: two groups belong together when their endpoints sit in the same
    // orbits under the generators, which is a union-find over the generators and
    // needs no walk of the group.
    //
    // What this replaces was a minimisation over every automorphism, bounded at
    // 512 because a symmetric skeleton can generate far more. Past the bound the
    // enumeration was a subset, so "smallest over the group" quietly became
    // "smallest over whichever elements came first" -- and a bis-DOTA conjugate,
    // four equivalent carboxylate arms on each of two chelators, swapped which
    // arm carried the charge between one pass and the next.
    //
    // Positions stay in the field. Only the *pairing* of totals to groups is
    // decided here, by sorting both within each orbit class, so a molecule whose
    // interchangeable groups already carry equal totals -- which is most of them
    // -- keeps the key it had.
    let width = generators
        .iter()
        .map(|generator| generator.len())
        .max()
        .unwrap_or(0);
    let mut parent: Vec<usize> = (0..width).collect();
    fn root(parent: &mut [usize], mut slot: usize) -> usize {
        while parent[slot] != slot {
            parent[slot] = parent[parent[slot]];
            slot = parent[slot];
        }
        slot
    }
    for generator in generators {
        for (slot, &image) in generator.iter().enumerate() {
            if slot >= width || image >= width {
                continue;
            }
            let (a, b) = (root(&mut parent, slot), root(&mut parent, image));
            if a != b {
                if a < b { parent[b] = a } else { parent[a] = b }
            }
        }
    }
    // An endpoint is named by the *representative of its orbit*, not by its own
    // position. A lone group in a symmetric skeleton has no other group to be
    // confused with, but its own endpoints can be carried onto a symmetric image
    // of themselves -- a bis-catechol crown ether, two equivalent rings and one
    // group, wrote `0,1` from one writing and `2,3` from the other. Naming the
    // orbit says the two are the same place, which is what the automorphism
    // means.
    let slot_of = |parent: &mut Vec<usize>, atom: usize| -> usize {
        let slot = position.get(atom).copied().unwrap_or(atom);
        if slot < width {
            root(parent, slot)
        } else {
            slot
        }
    };
    // Each group is named by the orbits of its endpoints. Two groups an
    // automorphism interchanges then write the same field, which is what the
    // automorphism means: as far as the compound is concerned they are one
    // place.
    let mut fields: Vec<Vec<usize>> = groups
        .groups()
        .iter()
        .map(|group| {
            let mut orbits: Vec<usize> = group
                .endpoints
                .iter()
                .map(|&atom| slot_of(&mut parent, atom))
                .collect();
            orbits.sort_unstable();
            orbits
        })
        .collect();
    fields.sort();
    let mut key = String::new();
    for names in fields {
        let listed: Vec<String> = names.iter().map(|atom| atom.to_string()).collect();
        key.push('/');
        key.push_str(&listed.join(","));
    }
    let (hydrogens, charge, radicals) = groups.totals();
    // The radical count is written only where there is one, which is very
    // nearly never.
    let electrons = if radicals == 0 {
        String::new()
    } else {
        format!(":{radicals}")
    };
    key.push('/');
    if charge_reduced {
        // What every protonation state shares; see
        // [`TautomerRules::charge_reduced`].
        key.push_str(&format!("{}{electrons}", hydrogens as i64 - charge as i64));
    } else {
        key.push_str(&format!("{hydrogens}:{charge}{electrons}"));
    }
    key
}

/// [`tautomer_key`] for a SMILES, which may name several components.
///
/// Under [`TautomerRules::whole_record`], the default, the components are
/// keyed as one molecule, so a proton may move from one to another. Otherwise
/// each gets its own key and the keys are sorted and joined with `.`. Either
/// way a salt written in any order gives the same string.
///
/// ```
/// use oxidane::molecule::tautomer::{tautomer_key_from_smiles, TautomerRules};
///
/// let rules = TautomerRules::new();
/// // 2-pyridone and 2-hydroxypyridine.
/// let keto = tautomer_key_from_smiles("O=c1cccc[nH]1", &rules).unwrap();
/// let enol = tautomer_key_from_smiles("Oc1ccccn1", &rules).unwrap();
/// assert_eq!(keto, enol);
/// // Methylamine hydrochloride, with the proton on either side.
/// assert_eq!(
///     tautomer_key_from_smiles("CN.Cl", &rules).unwrap(),
///     tautomer_key_from_smiles("C[NH3+].[Cl-]", &rules).unwrap(),
/// );
/// // A proton is moved, never added: acetate is not acetic acid.
/// assert_ne!(
///     tautomer_key_from_smiles("CC(=O)[O-]", &rules).unwrap(),
///     tautomer_key_from_smiles("CC(=O)O", &rules).unwrap(),
/// );
/// ```
pub fn tautomer_key_from_smiles(
    smiles: &str,
    rules: &TautomerRules,
) -> Result<String, crate::io::SMILESParseError> {
    with_canonizer(|canonizer| {
        let components = MolecularGraph::from_smiles(smiles)?;
        if rules.whole_record && components.len() > 1 {
            let parts: Vec<&MolecularGraph> = components.iter().collect();
            let record = MolecularGraph::combine(&parts);
            return Ok(tautomer_key_with(canonizer, &record, rules));
        }
        let mut keys: Vec<String> = components
            .iter()
            .map(|component| tautomer_key_with(canonizer, component, rules))
            .collect();
        keys.sort();
        Ok(keys.join("."))
    })
}

/// A canonical tautomer of `molecule`: the same structure every tautomer of it
/// normalizes to, as a real molecule rather than a key.
///
/// [`tautomer_key`] answers "are these the same compound"; this answers "which
/// of the forms shall we store". The two are built on the same footing -- the
/// erased skeleton and the group totals -- so a molecule and its normalization
/// always share a key.
///
/// # How the form is chosen
///
/// The skeleton fixes every atom's σ bonds and the group totals fix how many
/// hydrogens and charges the group holds, which leaves exactly one thing
/// undetermined: which of the group's bonds carry the π bonds. Write `y_e = 1`
/// for a bond that does. Every atom `u` in the group then satisfies
///
/// ```text
///     Σ_{e ∋ u} y_e  +  h_u  +  n_u  =  c_u        c_u = neutral valence − σ bonds
/// ```
///
/// with `h_u` the hydrogens it holds and `n_u` the negative charges, and the
/// group conserving `Σ h_u` and `Σ n_u`. Hydrogens and charges are not separate
/// unknowns: give each endpoint edges to a *pool* vertex whose own required
/// degree is the group total, and `h` and `n` become edge selections like any
/// other. What is left is a **degree-constrained subgraph** -- every vertex
/// wanting an exact degree -- which
/// [`degree_constrained_subgraph`](crate::molecule::matching::degree_constrained_subgraph)
/// solves exactly by Tutte's reduction to a perfect matching, on the blossom
/// implementation the kekulizer already uses. It is polynomial, and there is no
/// search over rules, no rule ordering, and no dead end to back out of.
///
/// A positive group charge is modelled the other way round, since a `+` *adds*
/// capacity rather than spending it: every endpoint's requirement goes up by
/// one and a pool takes back all but `Q` of them, so exactly `Q` endpoints keep
/// the extra. That is what puts guanidinium's charge on one nitrogen.
///
/// # Which solution, of the many
///
/// An f-factor problem usually has several solutions and a bare solver returns
/// whichever its numbering led it to, so the bonds are decided one at a time in
/// an order that is a property of the molecule: by the *canonical* position of
/// the atoms they join, heteroatom first, oxygen before sulfur before nitrogen.
/// Each is made a π bond if the rest of the problem can still be solved with it
/// so, and single otherwise. Preferring the heteroatom bond is what picks the
/// amide over the imidic acid and 2-pyridone over 2-hydroxypyridine, which is
/// the convention registries use.
///
/// Several such orders are tried -- heteroatom bonds first, ring bonds first,
/// the bonds of six-membered rings first, and with π forbidden on the bonds
/// leaving all-carbon six-membered rings or every ring -- and the answer is
/// the one with the fewest formal charges, then the most aromatic bonds. Two
/// kinds of bond are decided before any of that: a nitro group's N–O bonds and
/// a nitrile's C–N bond, so neither is traded for hydrogens taken from NH
/// donors. A bond whose π would charge an atom that has no room as a neutral
/// one -- an ether oxygen, a tertiary nitrogen -- is offered after every other,
/// and a chalcogen gives back a positive charge before a nitrogen does.
///
/// The result is returned in canonical atom order, so `to_smiles()` on it is
/// the canonical SMILES of the canonical tautomer.
///
/// # Returns
///
/// `None` when a group has no valid assignment at all -- a skeleton whose
/// demands cannot be met, which a molecule that failed to Kekulise can produce,
/// or when the bounded ambiguity walk does not reach a repeated state.
/// Callers wanting a total function should fall back on
/// [`canonize_mol`](crate::graph_algorithms::mol_canonization::canonize_mol).
///
/// ```
/// use oxidane::prelude::*;
/// use oxidane::molecule::tautomer::{normalize, TautomerRules};
///
/// let rules = TautomerRules::new();
/// let keto = MolecularGraph::from_smiles_single("O=c1cccc[nH]1").unwrap();
/// let enol = MolecularGraph::from_smiles_single("Oc1ccccn1").unwrap();
/// assert_eq!(
///     normalize(&keto, &rules).unwrap().to_smiles(),
///     normalize(&enol, &rules).unwrap().to_smiles(),
/// );
///
/// // And the form chosen is the amide, not the imidic acid.
/// let imidic = MolecularGraph::from_smiles_single("CC(O)=N").unwrap();
/// assert_eq!(normalize(&imidic, &rules).unwrap().to_smiles(), "O=C(N)C");
/// ```
pub fn normalize<M: Molecule + Clone + ToSMILES>(molecule: &M, rules: &TautomerRules) -> Option<M> {
    with_canonizer(|canonizer| normalize_with(canonizer, molecule, rules))
}

/// Why normalization could not produce a canonical answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationError {
    NoAssignment,
    DidNotConverge,
    /// The named atom's bonds exceed every valence its element and charge have,
    /// so the structure is not one this can be asked about. See
    /// [`impossible_valence`].
    ImpossibleValence(usize),
}

impl std::fmt::Display for NormalizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAssignment => f.write_str("no feasible tautomer assignment"),
            Self::DidNotConverge => f.write_str("normalization did not reach a repeated state"),
            Self::ImpossibleValence(atom) => write!(
                f,
                "atom {atom} spends more valence than its element and charge allow"
            ),
        }
    }
}
impl std::error::Error for NormalizationError {}

/// The first atom the model must type from the valence table whose bonds no
/// valence of its element and charge can account for.
///
/// Everything this module promises about `normalize` rests on one fact: the
/// input molecule is itself a feasible point of the constraint system, so a
/// representative exists. That is a statement about molecules, and it needs the
/// input to be one. An atom spending more than any valence its element has is
/// not a tautomer of anything -- there is no assignment to find, because there
/// was none to begin with -- so it is reported as an input error, not as a
/// failed search.
///
/// PubChem deposits such records: `[B]=[SH-](=P)=PF` has a sulfur spending
/// seven, and RDKit refuses to read it at all ("Explicit valence for atom # 1
/// S, 7, is greater than permitted"). This crate's reader is more
/// forgiving, so the question is asked here instead, once, before a model is
/// built on a premise the molecule does not meet.
///
/// The test is the one the reader itself uses in reverse. Where an atom's bonds
/// fall *short* of a valence, the reader makes the difference up with unpaired
/// electrons; where they overshoot the largest valence there is, nothing can be
/// made up, and this returns the atom. An element the valence table does not
/// cover -- a transition metal, most often -- is nobody's business here and is
/// passed over: `[Mn]` and `[V]` come through untouched.
///
/// Only the endpoint elements are asked, because they are the only atoms the
/// model types from the table: an endpoint has had its charge and its hydrogens
/// erased and takes them back from the pools, so what valences it may use is
/// the table's answer rather than the molecule's. Every other atom keeps the
/// valence the input gave it, whether or not the table lists it -- which is
/// what lets a dialkyl H-phosphonate's four-valent phosphorus through, and a
/// perchlorate's chlorine spending seven where the table stops at one. Those
/// are gaps in the table, not impossible molecules, and a check that refused
/// them would throw away chemistry PubChem is full of.
///
/// ```
/// use oxidane::prelude::*;
/// use oxidane::molecule::tautomer::{impossible_valence, TautomerRules};
///
/// let rules = TautomerRules::new();
/// let ordinary = MolecularGraph::from_smiles_single("CC(=O)N").unwrap();
/// assert_eq!(impossible_valence(&ordinary, &rules), None);
///
/// // A sulfur spending seven, which no valence of sulfur allows.
/// let deposited = MolecularGraph::from_smiles_single("[B]=[SH-](=P)=PF").unwrap();
/// assert_eq!(impossible_valence(&deposited, &rules), Some(1));
///
/// // A perchlorate's chlorine spends seven and is none of this function's
/// // business: no group types it, so it keeps the valence it came with.
/// let perchlorate = MolecularGraph::from_smiles_single("O=Cl(=O)(=O)[O-]").unwrap();
/// assert_eq!(impossible_valence(&perchlorate, &rules), None);
/// ```
pub fn impossible_valence<M: Molecule + ?Sized>(
    molecule: &M,
    rules: &TautomerRules,
) -> Option<usize> {
    use chemistry_consts::ElementProperties;
    // A question about a structure, so asked under the rules a structure is
    // normalized by.
    let rules = &rules.for_structures();
    (0..molecule.atomic_numbers().len()).find(|&atom| {
        let atomic_number = molecule.atomic_numbers()[atom];
        if !rules.is_endpoint_element(atomic_number) {
            return false;
        }
        let Some(mut valencies) = atomic_number.valencies() else {
            // No valence table: accept, rather than decide here what a
            // manganese atom may do.
            return false;
        };
        // The bonds and the unpaired electrons together, which is what
        // `used_valence` counts. `None` where an aromatic bond makes the count
        // fractional: there is no whole number to compare against, and a ring
        // the writer will localize is not where impossible valences hide.
        let Some(spends) = used_valence(molecule, atom) else {
            return false;
        };
        let shift = i64::from(crate::molecule::base::charge_valency_shift(
            atomic_number,
            molecule.get_atom_charge(atom),
        ));
        !valencies.any(|valency| valency > 0 && i64::from(valency) + shift >= spends)
    })
}

/// Normalize while distinguishing infeasibility from an exhausted ambiguity walk.
pub fn try_normalize<M: Molecule + Clone + ToSMILES>(
    molecule: &M,
    rules: &TautomerRules,
) -> Result<M, NormalizationError> {
    with_canonizer(|canonizer| try_normalize_with(canonizer, molecule, rules))
}

/// [`normalize`], reusing a caller's [`Canonizer`].
pub fn normalize_with<M: Molecule + Clone + ToSMILES>(
    canonizer: &mut Canonizer,
    molecule: &M,
    rules: &TautomerRules,
) -> Option<M> {
    try_normalize_with(canonizer, molecule, rules).ok()
}

/// [`try_normalize`], reusing a caller's canonicalization buffers.
pub fn try_normalize_with<M: Molecule + Clone + ToSMILES>(
    canonizer: &mut Canonizer,
    molecule: &M,
    rules: &TautomerRules,
) -> Result<M, NormalizationError> {
    let rules = &rules.for_structures();
    // Asked before anything is built, because everything below assumes the
    // input is a feasible point of its own constraint system, and a structure
    // whose valences do not exist is not one.
    if let Some(atom) = impossible_valence(molecule, rules) {
        return Err(NormalizationError::ImpossibleValence(atom));
    }
    let (settled, mirror_ambiguous) =
        place_the_totals(canonizer, molecule, rules).ok_or(NormalizationError::NoAssignment)?;
    // Where erasure has invented a symmetry the tags do not respect -- an azo
    // cage whose `N=N` and whose aminyl radical become one nitrogen apiece, so
    // that the two halves of the cage answer to each other -- the placement can
    // come down on either side of a mirror, and which side it picks is decided
    // by the labelling rather than by the molecule. The two sides are one
    // tautomer either way, so both are answers; what they are not is a fixed
    // point, since normalizing one gives the other. Normalizing once more and
    // keeping the smaller settles it, and the question is only ever asked of a
    // molecule that has such a symmetry.
    if !mirror_ambiguous {
        return Ok(settled);
    }
    // Normalizing again either stays where it is, or walks a short cycle
    // through the mirror images of one answer. Following it to the cycle and
    // taking the smallest member is a choice every member of the cycle makes
    // the same way, which is what makes the answer a fixed point.
    const MOST_PASSES: usize = 4;
    let mut walked: Vec<(String, M)> = vec![(settled.to_smiles(), settled)];
    for _ in 0..MOST_PASSES {
        let Some((next, _)) = place_the_totals(canonizer, &walked[walked.len() - 1].1, rules)
        else {
            return Err(NormalizationError::NoAssignment);
        };
        let smiles = next.to_smiles();
        if let Some(first) = walked.iter().position(|(seen, _)| *seen == smiles) {
            let smallest = walked[first..]
                .iter()
                .min_by(|(left, _), (right, _)| left.cmp(right))
                .unwrap();
            return Ok(smallest.1.clone());
        }
        walked.push((smiles, next));
    }
    // An unfinished walk is not a canonical answer.
    Err(NormalizationError::DidNotConverge)
}

/// The half of [`normalize_with`] that puts the erased hydrogens, charges and
/// bond orders back, in the canonical order of the skeleton they came off.
///
/// Also says whether a symmetry of the erased skeleton moves a chirality tag.
/// Erasure takes the bond orders, the charges and the mobile hydrogens out of
/// the graph, and a molecule the graph could tell apart before may be symmetric
/// afterwards; where that new symmetry does not carry the tags with it, the two
/// ways of putting the erased data back are mirror images of each other and the
/// placement has nothing to choose between them.
fn place_the_totals<M: Molecule + Clone + ToSMILES>(
    canonizer: &mut Canonizer,
    molecule: &M,
    rules: &TautomerRules,
) -> Option<(M, bool)> {
    let (skeleton, groups) = erased_skeleton(molecule, rules);
    let used = groups.used_valences.clone();
    if groups.is_empty() {
        return Some((canonizer.canonize(&skeleton), false));
    }

    // Work in canonical atom order from the start: the skeleton is the same for
    // every tautomer of the compound, so its canonical order is too, and every
    // choice made below can then be made by a rule that reads atom positions.
    let atom_count = skeleton.atomic_numbers().len();
    let mut tagged = skeleton;
    *tagged.atom_classes_mut() = Some((0..atom_count).collect());
    let mut canonical = canonizer.canonize(&tagged);
    let origins = canonical.atom_classes().clone()?;
    *canonical.atom_classes_mut() = None;

    let mut position = vec![usize::MAX; atom_count];
    for (canonical_position, &original) in origins.iter().enumerate() {
        if original < atom_count {
            position[original] = canonical_position;
        }
    }
    let mirror_ambiguous = molecule.is_chiral() && a_symmetry_moves_a_tag(canonizer, &canonical);
    // The skeleton's own symmetries, kept before anything else is canonized:
    // what the placement is free to do is decided by the group erasure left,
    // not by the group the answer happens to have.
    let skeleton_symmetries: Vec<Vec<usize>> = canonizer.last_graph_automorphisms().to_vec();
    let groups = remap_through(groups, &position, atom_count)?;
    // One instance for the whole component, holding every group's atoms and one
    // pool of each species between them. Solving the groups one at a time asked
    // each to give back exactly the hydrogens and the charge it arrived with,
    // which is what made a proton moved from one acidic site to another come
    // back as a different compound; see [`TautomericGroups::totals`].
    let pooled = groups.pooled();
    // The valences follow the atoms into canonical order.
    let mut used_canonical: Vec<Option<i64>> = vec![None; atom_count];
    for (atom, valence) in used.into_iter().enumerate() {
        if let Some(&slot) = position.get(atom)
            && slot < atom_count
        {
            used_canonical[slot] = valence;
        }
    }

    // Which tautomer of a class reads best is not something the constraints can
    // settle -- every candidate below satisfies every valence -- so it is
    // settled by counting afterwards, and the count that matters is aromatic
    // bonds. Two levers are tried:
    //
    // * the order the bonds are offered their π in, heteroatom-first or
    //   ring-first, and
    // * *closing* a ring: forbidding π on the bonds that leave it, so the ring
    //   has no choice but to satisfy itself, which is exactly an aromatic
    //   Kekulé structure over its own bonds.
    //
    // The second is what a pure ordering cannot do. Offering ring bonds first
    // is not enough, because forcing them in index order yields a *maximal* but
    // not perfect matching on the ring, and the atoms it leaves over take their
    // π from the exocyclic bonds anyway -- which is how 4-cyanophenol came back
    // as a quinone-imine. Forbidding those bonds removes the alternative.
    //
    // A closure is kept only when it *increases* the aromatic count, so a
    // molecule that was already aromatic keeps the form the first candidate
    // gave it -- which is why 2-pyridone stays a pyridone rather than becoming
    // 2-hydroxypyridine, both being aromatic and neither an improvement on the
    // other.
    let (cyclic_atoms, cyclic) = canonical.find_cyclic_bonds_and_atoms();
    let closable: FxHashSet<(usize, usize)> = groups
        .groups()
        .iter()
        .flat_map(|group| group.bonds.iter().copied())
        .filter(|bond| {
            !cyclic.contains(bond)
                && (cyclic_atoms.contains(bond.0) || cyclic_atoms.contains(bond.1))
        })
        .collect();
    let nothing = FxHashSet::default();

    // Each candidate is a full lexicographic pass, and a lexicographic pass is
    // one matching per edge -- so the cost is cubic in the group and every
    // candidate that cannot change the answer is worth not running. Closing
    // rings does nothing when no group bond leaves one, and ordering ring bonds
    // first does nothing when no group holds both a ring bond and another kind.
    let mixed = groups.groups().iter().any(|group| {
        let on_ring = group
            .bonds
            .iter()
            .filter(|bond| cyclic.contains(bond))
            .count();
        on_ring > 0 && on_ring < group.bonds.len()
    });
    // Ring-first offers every ring bond alike, so the C=C of a lactone or a
    // pyridone fused to or hung on a benzene ring can take the π the benzene
    // needed: a phenol beside a lactone came back as a quinone methide, and
    // closing the rings instead buys the benzene with a charge separation.
    // Offering the six-membered rings' bonds first keeps the benzene whole, and
    // so does closing only the all-carbon six-membered rings, which leaves a
    // pyridone beside the benzene free to take its carbonyl. Ring membership is
    // read from σ bonds alone, like everything else here, and is worked out only
    // for a molecule the first candidate does not settle.
    let six_rings = || {
        let rings: Vec<Vec<usize>> = crate::molecule::topology::smallest_rings(&canonical)
            .into_iter()
            .filter(|ring| ring.len() == 6)
            .collect();
        let bonds: FxHashSet<(usize, usize)> = rings
            .iter()
            .flat_map(|ring| {
                (0..6).map(|index| {
                    let (first, second) = (ring[index], ring[(index + 1) % 6]);
                    (first.min(second), first.max(second))
                })
            })
            .collect();
        let carbocyclic: FxHashSet<usize> = rings
            .iter()
            .filter(|ring| {
                ring.iter()
                    .all(|&atom| canonical.atomic_numbers()[atom] == 6)
            })
            .flatten()
            .copied()
            .collect();
        let closable_carbocycles: FxHashSet<(usize, usize)> = closable
            .iter()
            .filter(|bond| carbocyclic.contains(&bond.0) || carbocyclic.contains(&bond.1))
            .copied()
            .collect();
        // Worth a candidate only when a group holds both kinds of ring bond.
        let mixed = groups.groups().iter().any(|group| {
            group.bonds.iter().any(|bond| bonds.contains(bond))
                && group
                    .bonds
                    .iter()
                    .any(|bond| cyclic.contains(bond) && !bonds.contains(bond))
        });
        SixRings {
            bonds,
            closable_carbocycles,
            mixed,
        }
    };
    let mut six: Option<SixRings> = None;
    let mut candidates: Vec<(BondOrdering, Rings, Closure, bool)> = vec![(
        BondOrdering::HeteroatomFirst,
        Rings::All,
        Closure::Open,
        false,
    )];
    if mixed {
        candidates.push((BondOrdering::RingFirst, Rings::All, Closure::Open, false));
    }
    candidates.push((
        BondOrdering::RingFirst,
        Rings::SixMembered,
        Closure::Open,
        false,
    ));
    if !closable.is_empty() {
        candidates.push((
            BondOrdering::HeteroatomFirst,
            Rings::All,
            Closure::Carbocycles,
            false,
        ));
        candidates.push((
            BondOrdering::HeteroatomFirst,
            Rings::All,
            Closure::Every,
            false,
        ));
        if mixed {
            candidates.push((BondOrdering::RingFirst, Rings::All, Closure::Every, false));
        }
    }
    // The pools-first ordering last, as the answer for a group the others leave
    // with its hydrogens and electrons piled on one atom.
    candidates.push((
        BondOrdering::HeteroatomFirst,
        Rings::All,
        Closure::Open,
        true,
    ));

    // One instance for every candidate: they differ in the order the bonds are
    // offered their π, which is a pass over the instance rather than a reason
    // to build another.
    let mut instance = instance_for(&canonical, &pooled, &used_canonical)?;
    let mut best: Option<(
        (std::cmp::Reverse<u32>, usize, std::cmp::Reverse<usize>),
        M,
        bool,
        bool,
    )> = None;
    // Two levers can lead to the same bond orders -- a molecule whose rings are
    // already satisfied does not care in which order its bonds were offered
    // their pi -- and scoring a candidate means canonizing it, which is what
    // the pass costs. So a candidate that has come out as one already scored is
    // dropped before it is paid for; it could only tie, and a tie never wins.
    let mut scored = FxHashSet::default();
    // The most aromatic bonds any candidate can have: those the skeleton
    // already has outside the groups, whose orders no candidate changes, and
    // every ring bond a group owns. Counting every ring bond instead never
    // settles a molecule with a saturated ring.
    let most_aromatic = aromatic_bond_count(&canonical)
        + pooled
            .bonds
            .iter()
            .filter(|&&(first, second)| cyclic.contains(&(first.min(second), first.max(second))))
            .count();
    // The charge count of the first candidate scored, which has no closures
    // and so separates no charge it can avoid.
    let mut fewest = None;
    for (ordering, rings, closure, spread) in candidates {
        // The last candidate exists for groups the others cannot write at all,
        // so it is not allowed to win a molecule that already has an answer.
        if spread && best.is_some() {
            break;
        }
        if matches!(rings, Rings::SixMembered) || matches!(closure, Closure::Carbocycles) {
            six.get_or_insert_with(six_rings);
        }
        let rings = match rings {
            Rings::All => &cyclic,
            Rings::SixMembered => match &six {
                Some(six) if six.mixed => &six.bonds,
                _ => continue,
            },
        };
        let closed = match closure {
            Closure::Open => &nothing,
            Closure::Every => &closable,
            // The same as closing every ring when every closable bond leaves a
            // carbocycle, so not worth a pass of its own.
            Closure::Carbocycles => match &six {
                Some(six)
                    if !six.closable_carbocycles.is_empty()
                        && six.closable_carbocycles.len() < closable.len() =>
                {
                    &six.closable_carbocycles
                }
                _ => continue,
            },
        };
        let mut candidate = canonical.clone();
        if solve_group(
            &mut candidate,
            &pooled,
            &mut instance,
            ordering,
            rings,
            closed,
            &used_canonical,
            spread,
        )
        .is_none()
        {
            continue;
        }
        if !scored.insert((
            bond_order_signature(&candidate),
            placement_signature(&candidate),
        )) {
            continue;
        }
        // The bond orders and hydrogens have changed, so the aromaticity
        // perceived on the skeleton no longer describes the molecule;
        // canonizing both perceives it afresh and puts the added hydrogens in
        // canonical order.
        let unsettled =
            mirror_ambiguous || a_symmetry_moves_a_placement(&skeleton_symmetries, &candidate);
        let (candidate, already_canonical) = canonizer.perceive_for_scoring(candidate);
        // Fewest charges first, then most aromatic bonds: a ring closure can
        // buy aromaticity with a charge separation, and N-methyl-2-pyridone
        // beside a phenol is not a pyridinium phenolate. Then fewest imines
        // hung off a ring, which only breaks ties: an imino ring such as
        // `N=c1[nH][nH]cc1` is as aromatic to this count as the amino ring
        // it is a tautomer of, and the amino form is the one written.
        let score = (
            std::cmp::Reverse(charge_count(&candidate)),
            aromatic_bond_count(&candidate),
            std::cmp::Reverse(exocyclic_imine_count(&candidate)),
        );
        let fewest = *fewest.get_or_insert(score.0);
        let settled = score.1 >= most_aromatic && score.0 == fewest && score.2.0 == 0;
        if best
            .as_ref()
            .is_none_or(|(best_score, _, _, _)| score > *best_score)
        {
            best = Some((score, candidate, unsettled, already_canonical));
        }
        // A candidate as aromatic as any can be, with the fewest charges,
        // cannot be beaten and the remaining orderings are worth nothing. Most
        // drug-like molecules reach that on the first pass, and each pass they
        // skip is a full lexicographic solve -- one matching per edge, the
        // dominant cost of `normalize`.
        if settled {
            break;
        }
    }
    best.map(|(_, candidate, unsettled, already_canonical)| {
        let candidate = if already_canonical {
            candidate
        } else {
            canonizer.canonize_perceived(candidate)
        };
        (candidate, unsettled)
    })
}

/// Whether one of the skeleton's own symmetries moves a chirality tag, read on
/// the skeleton already in canonical order.
fn a_symmetry_moves_a_tag<M: Molecule + Clone + ToSMILES>(
    canonizer: &Canonizer,
    canonical: &M,
) -> bool {
    let Some(classes) = canonical.chiral_classes().map(<[ChiralClass]>::to_vec) else {
        return false;
    };
    canonizer
        .last_graph_automorphisms()
        .iter()
        .any(|generator| {
            let mut moved = canonical.clone();
            let mut permutation: Vec<usize> = (0..moved.len()).collect();
            for (atom, &image) in generator.iter().enumerate() {
                if atom < permutation.len() && image < permutation.len() {
                    permutation[atom] = image;
                }
            }
            reorder_molecule(&mut moved, &permutation);
            moved.chiral_classes() != Some(classes.as_slice())
        })
}

/// Whether a symmetry of the erased skeleton moves a mobile hydrogen or an
/// endpoint charge -- that is, whether the placement had a free choice.
///
/// [`a_symmetry_moves_a_tag`] asks this of chirality, and it can ask it of the
/// skeleton because a chirality tag survives erasure. A mobile hydrogen does
/// not: erasure is what takes it out. So this has to be asked of the molecule
/// the placement *built*, against the symmetries the skeleton had -- the two
/// live at different points of the pass, and the skeleton no longer has the
/// hydrogen in question.
///
/// A porphyrazine carrying three nitroxide radicals and one hydroxylamine is
/// the case. Its four arms are one orbit of the skeleton, so which of them ends
/// up with the mobile hydrogen is settled by the labelling rather than by the
/// molecule, and `normalize` put it on a different arm from the deposited form
/// than from its own canonical SMILES.
fn a_symmetry_moves_a_placement<M: Molecule + Clone + ToSMILES>(
    generators: &[Vec<usize>],
    answer: &M,
) -> bool {
    if generators.is_empty() {
        return false;
    }
    // Read directly rather than by reordering a clone: a permutation moves the
    // placement exactly when some atom and its image carry different mobile
    // data, and asking that needs no copy of the molecule. `normalize` asks it
    // of every symmetric skeleton in the corpus, so the copy was most of what
    // the question cost.
    let signature = placement_signature(answer);
    generators.iter().any(|generator| {
        generator.iter().enumerate().any(|(atom, &image)| {
            match (signature.get(atom), signature.get(image)) {
                (Some(here), Some(there)) => here != there,
                _ => false,
            }
        })
    })
}

/// Where the mobile data sits, in atom order: each heavy atom's hydrogen count,
/// charge and unpaired electrons. Two placements of one skeleton's totals
/// differ here and nowhere else the graph can see.
fn placement_signature<M: Molecule + ?Sized>(molecule: &M) -> Vec<(u8, i8, u8)> {
    let numbers = molecule.atomic_numbers();
    (0..numbers.len())
        .filter(|&atom| numbers[atom] > 1)
        .map(|atom| {
            let hydrogens = molecule
                .get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .filter(|bond| numbers[bond.target()] == 1)
                .count() as u8;
            (
                hydrogens,
                molecule.get_atom_charge(atom),
                molecule.radical_states().get(atom).copied().unwrap_or(0),
            )
        })
        .collect()
}

/// The bond orders of a molecule in atom order, which is what tells two solved
/// candidates apart: they are the same molecule otherwise, having been solved
/// from one skeleton in one atom order.
fn bond_order_signature<M: Molecule + ?Sized>(molecule: &M) -> Vec<u8> {
    molecule
        .atom_bonds()
        .iter()
        .flat_map(|bonds| bonds.iter().map(|bond| bond.bond_order() as u8))
        .collect()
}

/// How many aromatic bonds a molecule has, which is what separates the two bond
/// orderings when both give a valid answer.
/// The sum of the magnitudes of the formal charges.
fn charge_count<M: Molecule + ?Sized>(molecule: &M) -> u32 {
    (0..molecule.atomic_numbers().len())
        .map(|atom| u32::from(molecule.get_atom_charge(atom).unsigned_abs()))
        .sum()
}

/// The double bonds from an aromatic ring atom to a nitrogen outside every
/// ring: the imino form of an aminoheteroarene, `N=c1[nH][nH]cc1` for
/// 3-aminopyrazole. A ring carbonyl is not counted -- 2-pyridone is the form
/// written -- and nor is an imine on a saturated ring, where the imino form
/// of a 2-amino-oxazoline or of clonidine is the one that predominates.
fn exocyclic_imine_count<M: Molecule + ?Sized>(molecule: &M) -> usize {
    let element = |atom: usize| molecule.atomic_numbers()[atom];
    let candidates: Vec<(usize, usize)> = (0..molecule.atomic_numbers().len())
        .flat_map(|atom| {
            molecule
                .get_atom_bonds(atom)
                .into_iter()
                .flatten()
                .filter(move |bond| {
                    bond.bond_order() == BondOrder::Double
                        && element(atom) == 7
                        && element(bond.target()) != 7
                        && molecule.is_atom_aromatic(bond.target())
                })
                .map(move |bond| (atom, bond.target()))
        })
        .collect();
    if candidates.is_empty() {
        return 0;
    }
    let (cyclic_atoms, _) = molecule.find_cyclic_bonds_and_atoms();
    candidates
        .into_iter()
        .filter(|&(nitrogen, _)| !cyclic_atoms.contains(nitrogen))
        .count()
}

/// Each bond is listed at both of its atoms, so the count is halved; compared
/// with the number of ring bonds, the doubled count called a molecule fully
/// aromatic with half its rings dearomatised.
fn aromatic_bond_count<M: Molecule + ?Sized>(molecule: &M) -> usize {
    (0..molecule.atomic_numbers().len())
        .filter_map(|atom| molecule.get_atom_bonds(atom))
        .flatten()
        .filter(|bond| bond.bond_order() == BondOrder::Aromatic)
        .count()
        / 2
}

/// Rewrites a group's atom indices through `position`, which sends an atom to
/// where canonization put it.
fn remap_through(
    mut groups: TautomericGroups,
    position: &[usize],
    atom_count: usize,
) -> Option<TautomericGroups> {
    let mut group_of = vec![None; atom_count];
    for (index, group) in groups.groups.iter_mut().enumerate() {
        for atom in group.endpoints.iter_mut().chain(group.atoms.iter_mut()) {
            *atom = *position.get(*atom)?;
        }
        for bond in group.bonds.iter_mut() {
            let (first, second) = (*position.get(bond.0)?, *position.get(bond.1)?);
            *bond = (first.min(second), first.max(second));
        }
        group.endpoints.sort_unstable();
        group.atoms.sort_unstable();
        group.bonds.sort_unstable();
        for &atom in &group.atoms {
            group_of[atom] = Some(index as u32);
        }
    }
    groups.group_of = group_of;
    Some(groups)
}

/// How many unpaired electrons an atom of this element may be given, out of the
/// `capacity` the constraints leave it.
///
/// SMILES has no symbol for an unpaired electron: a reader infers one from the
/// gap between what the bonds spend and the valence they reach into, so an
/// electron is only *writable* where no lower valence of the element could
/// explain the same bonds. Sulfur, selenium and tellurium step by two, so one
/// electron is the most any of them can carry -- give such an atom two and the
/// writer emits a structure the reader takes as the next valence down with
/// nothing unpaired, and the answer comes back a different compound from the
/// one that was written. A selenophosphorus cage from PubChem is the case:
/// three electrons piled onto one selenium wrote a molecule that read back with
/// two, so the normalized form did not key as what it came from.
///
/// An element with one valence has no lower rung to be confused with and keeps
/// whatever the constraints allow -- atomic oxygen is two unpaired electrons
/// and writes as `[O]`.
fn writable_radicals(atomic_number: u8, charges: &[i8], sigma: i64, capacity: usize) -> usize {
    let mut room = 0i64;
    for &charge in charges {
        let options = valence_options(atomic_number, charge, sigma);
        let Some(&lowest) = options.first() else {
            continue;
        };
        room = room.max(
            match options.windows(2).map(|pair| pair[1] - pair[0]).min() {
                // One valence, nothing to be mistaken for: the atom keeps whatever
                // the constraints leave it, and atomic oxygen writes as `[O]`.
                None => capacity as i64,
                // On any rung but the lowest, a second electron would put the bonds
                // where the rung below explains them on their own, and the reader
                // would take that reading: two on a selenium wrote a molecule that
                // came back with none. On the lowest rung there is nothing below to
                // read instead, so every unit between the bonds and that valence is
                // writable -- `[Te+]` with one bond is a valence of three and two
                // unpaired electrons, and capping it at one left the group with
                // nowhere to put the second.
                Some(step) => (lowest - sigma).max(step - 1),
            },
        );
    }
    capacity.min(room.max(0) as usize)
}

/// Whether every unpaired electron in the answer is one a SMILES can state.
///
/// The cap [`writable_radicals`] applies is an upper bound taken over the
/// charges an endpoint might end up with, and an endpoint that ends up neutral
/// after all has a tighter bound than the one it was offered. So the answer is
/// checked rather than trusted: an electron the writer cannot express is one
/// the reader will not return, and the molecule that comes back is a different
/// compound from the one that went out.
///
/// The test is the reader's own, run forwards: the electrons it will infer are
/// the gap between the bonds and the valence they reach into, and the answer is
/// writable when that gap is what the answer placed. An atom carrying an
/// aromatic bond is passed over for the same reason the reader passes over it
/// -- there is no whole number of units to an aromatic bond, so the valence it
/// reaches into is the element's first one rather than anything countable.
fn radicals_are_writable<M: Molecule + ?Sized>(molecule: &M, group: &TautomericGroup) -> bool {
    group.atoms.iter().all(|&atom| {
        let radicals = i64::from(molecule.radical_states().get(atom).copied().unwrap_or(0));
        if radicals == 0 {
            return true;
        }
        let Some(bonds) = molecule.get_atom_bonds(atom) else {
            return true;
        };
        if bonds
            .iter()
            .any(|bond| bond.bond_order() == BondOrder::Aromatic)
        {
            return true;
        }
        let bonded: i64 = bonds
            .iter()
            .map(|bond| match bond.bond_order() {
                BondOrder::Double => 2,
                BondOrder::Triple => 3,
                _ => 1,
            })
            .sum();
        valence_for(
            molecule.atomic_numbers()[atom],
            molecule.get_atom_charge(atom),
            bonded as usize,
        ) == Some(bonded + radicals)
    })
}

/// Which elements are willing to hold the group's charge.
fn takes_negative_charge(atomic_number: u8) -> bool {
    matches!(atomic_number, 7 | 8 | 15 | 16 | 34 | 52)
}

/// Selenium and tellurium belong here for the same reason sulfur does, and
/// leaving them out was not a decision so much as an oversight: PubChem's
/// selenonium and telluronium salts have three bonds at a chalcogen whose
/// neutral valences are 2, 4 and 6, so without the charge there is no valence
/// that fits and the group has no solution at all.
fn takes_positive_charge(atomic_number: u8) -> bool {
    matches!(atomic_number, 7 | 8 | 15 | 16 | 34 | 52)
}

/// Which ring bonds a ring-first candidate offers its π first.
#[derive(Clone, Copy)]
enum Rings {
    All,
    SixMembered,
}

/// Which bonds leaving a ring a candidate forbids π on.
#[derive(Clone, Copy)]
enum Closure {
    Open,
    /// Those at all-carbon six-membered rings.
    Carbocycles,
    Every,
}

/// The six-membered rings of a component's skeleton, for the candidates that
/// read them.
struct SixRings {
    bonds: FxHashSet<(usize, usize)>,
    closable_carbocycles: FxHashSet<(usize, usize)>,
    mixed: bool,
}

/// Which bonds are offered a π bond first.
///
/// Neither order is right on its own, and the difference is not one the
/// constraints can settle -- both answers satisfy every valence. Taking the
/// heteroatom bond first writes 2-pyridone rather than 2-hydroxypyridine, which
/// is the convention; taking ring bonds first suits a ring that would otherwise
/// lose its π to an exocyclic bond. Both are solved, along with both again
/// under the ring closures, and the candidate retaining most aromatic bonds is
/// kept -- see [`normalize_with`].
///
/// Ring-first is *not* on its own enough to keep a ring aromatic, which is why
/// the closures exist: forcing ring bonds in index order gives a maximal but
/// not perfect matching over the ring, and the atoms it leaves over take their
/// π from the exocyclic bonds regardless.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BondOrdering {
    /// Heteroatom bonds first: the carbonyl over the enol.
    HeteroatomFirst,
    /// Ring bonds first: the aromatic system over the exocyclic double bond.
    RingFirst,
}

/// How early a bond gets to claim a π bond: the lower, the sooner.
///
/// Reading the heteroatom bond first is what chooses the carbonyl over the enol
/// and the amide over the imidic acid, and oxygen before nitrogen is what makes
/// `CC(O)=N` come back as `CC(=O)N` rather than the other way about.
fn bond_priority(first_element: u8, second_element: u8) -> u8 {
    let rank = |atomic_number: u8| match atomic_number {
        8 => 0u8,
        16 => 1,
        34 | 52 => 2,
        7 => 3,
        15 => 4,
        _ => 5,
    };
    rank(first_element).min(rank(second_element))
}

/// Where a bond stands in the order bonds are offered π units, earliest first.
///
/// The element ranks of [`bond_priority`] decide between bonds of one class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BondClass {
    /// A nitro group's N–O or a nitrile's C–N, offered before anything else
    /// is decided.
    Settled,
    Ordinary,
    /// A bond whose π unit would leave an atom charged that has no room as a
    /// neutral one: an ether oxygen or a tertiary nitrogen. Offered last, so a
    /// group's positive charge goes to such an atom only when nothing else
    /// takes it -- which keeps a cyanine's charge on its nitrogen rather than
    /// on the methoxy oxygen at the far end of the chromophore.
    Charging,
}

/// The class of `bond`, read from σ degrees and neutral capacities alone, so
/// that every tautomer of a compound classes its bonds the same way.
fn bond_class<M: Molecule + ?Sized>(
    molecule: &M,
    (first, second): (usize, usize),
    (first_slot, second_slot): (usize, usize),
    capacity: &[i64],
) -> BondClass {
    let element = |atom: usize| molecule.atomic_numbers()[atom];
    let is_nitrile = |carbon: usize, carbon_slot: usize, nitrogen: usize, nitrogen_slot: usize| {
        element(carbon) == 6
            && element(nitrogen) == 7
            && heavy_degree(molecule, carbon) == 2
            && heavy_degree(molecule, nitrogen) == 1
            && capacity[carbon_slot] >= 2
            && capacity[nitrogen_slot] >= 2
    };
    if is_nitro_oxygen(molecule, first, second)
        || is_nitro_oxygen(molecule, second, first)
        || is_nitrile(first, first_slot, second, second_slot)
        || is_nitrile(second, second_slot, first, first_slot)
    {
        BondClass::Settled
    } else if capacity[first_slot] <= 0 || capacity[second_slot] <= 0 {
        BondClass::Charging
    } else {
        BondClass::Ordinary
    }
}

/// Whether `oxygen` is a terminal oxygen of the nitro group on `nitrogen`:
/// a nitrogen with three heavy neighbours, two of them terminal oxygens.
fn is_nitro_oxygen<M: Molecule + ?Sized>(molecule: &M, oxygen: usize, nitrogen: usize) -> bool {
    let terminal_oxygen =
        |atom: usize| molecule.atomic_numbers()[atom] == 8 && heavy_degree(molecule, atom) == 1;
    terminal_oxygen(oxygen)
        && molecule.atomic_numbers()[nitrogen] == 7
        && heavy_degree(molecule, nitrogen) == 3
        && molecule.get_atom_bonds(nitrogen).is_some_and(|bonds| {
            bonds
                .iter()
                .filter(|bond| terminal_oxygen(bond.target()))
                .count()
                >= 2
        })
}

/// What one group's atoms should end up carrying.
struct Assignment {
    /// The group bonds that take a π bond, and how many.
    bond_orders: Vec<((usize, usize), BondOrder)>,
    /// Atoms taking a negative charge, and atoms taking a positive one.
    negative: Vec<usize>,
    positive: Vec<usize>,
    /// One entry per hydrogen to hang back on the atom named.
    hydrogens: Vec<usize>,
    /// One entry per unpaired electron to put back on the atom named.
    radicals: Vec<usize>,
}

/// Assigns the component's π bonds, hydrogens, charges and electrons, in
/// place, for one order of preferences over its bonds.
///
/// Atom indices are canonical positions in `molecule`; see [`normalize`] for
/// what is being solved and why the answer is unique.
fn solve_group<M: Molecule + Clone + ToSMILES>(
    molecule: &mut M,
    group: &TautomericGroup,
    instance: &mut GroupInstance,
    ordering: BondOrdering,
    cyclic: &FxHashSet<(usize, usize)>,
    closed: &FxHashSet<(usize, usize)>,
    used: &[Option<i64>],
    spread: bool,
) -> Option<()> {
    let assignment = plan_from(instance, molecule, ordering, cyclic, closed, None, spread);
    // An atom's bond list is a fixed-capacity inline array, so an assignment
    // asking for more is a panic rather than a wrong answer. It should never
    // happen -- nothing in the model gives an atom more neighbours than a
    // valence allows -- but a library that aborts the process when it does is
    // worse than one that says it found no assignment, and a corpus of 124
    // million records is exactly where the exception turns up.
    if !fits_in_the_bond_lists(molecule, &assignment) {
        debug_assert!(
            false,
            "the assignment gives an atom more bonds than it can hold"
        );
        return None;
    }
    apply(molecule, &assignment);
    // The model could be wrong in a way the solver cannot see, so the answer is
    // checked against the valences it claims to satisfy rather than trusted.
    // It should never fire: the input molecule is itself a feasible point of
    // the constraint system, so an empty feasible set means the model and the
    // molecule disagree, which is a bug rather than a fact about chemistry.
    if !valences_are_consistent(molecule, group, used) {
        debug_assert!(false, "the group's own valences are not a feasible point");
        return None;
    }
    if !radicals_are_writable(molecule, group) {
        return None;
    }
    Some(())
}

/// The component's constraint system, built on the first question asked of it.
///
/// Most molecules ask none: a geometry probe runs only where a double bond
/// could carry a class the erasure would otherwise destroy, and building the
/// instance up front for the rest cost more than every probe in the corpus put
/// together -- `tautomer_key` over 20,000 PubChem records went from 0.69 to
/// 1.63 seconds when it was built eagerly.
struct LazyInstance {
    built: Option<Option<GroupInstance>>,
}

impl LazyInstance {
    fn new() -> Self {
        Self { built: None }
    }

    fn get<M: Molecule + Clone + ToSMILES>(
        &mut self,
        molecule: &M,
        group: &TautomericGroup,
        used: &[Option<i64>],
    ) -> Option<&mut GroupInstance> {
        self.built
            .get_or_insert_with(|| instance_for(molecule, group, used))
            .as_mut()
    }
}

/// The component's constraint system, pinned model for preference.
///
/// Pinning a stereocentre can leave the component nowhere to put a mobile
/// hydrogen or a charge that atom was itself holding, and then there is no
/// assignment at all. Keeping the geometry is worth trying for and not worth
/// failing over, so the unpinned model answers when the pinned one cannot.
fn instance_for<M: Molecule + Clone + ToSMILES>(
    molecule: &M,
    group: &TautomericGroup,
    used: &[Option<i64>],
) -> Option<GroupInstance> {
    let positives = group.mobile_charge.max(0) as usize;
    let negatives = (-group.mobile_charge).max(0) as usize;
    build_instance(molecule, group, positives, negatives, used, true)
        .or_else(|| build_instance(molecule, group, positives, negatives, used, false))
}

/// Whether every assignment the group admits gives one of its bonds the same
/// order.
///
/// Erasure exists to make the skeleton a function of the compound rather than
/// of the tautomer it was handed, and a bond the shift can move has no order
/// that is such a function. A bond it *cannot* move does: every member of the
/// class writes it the same way, so erasing it buys nothing, and where the
/// input states a geometry on it the erasure costs a fact -- which is why this
/// question is asked at all. It is asked only of bonds carrying a geometry,
/// since nowhere else is the answer worth two solver runs.
///
/// Both ends of the range are needed. A bond that is double in one assignment
/// and triple in another is moved by the shift as surely as one that goes
/// single, and only the two probes together rule that out.
///
/// `None` where the group does not solve at all, which is the same answer as
/// "not fixed": the caller then erases, as it always did.
fn fixed_order<M: Molecule + ?Sized>(
    skeleton: &M,
    instance: &mut GroupInstance,
    bond: (usize, usize),
) -> Option<BondOrder> {
    let none = FxHashSet::default();
    // The unpinned model, which has the larger feasible set: a bond fixed
    // across that is fixed across the pinned one `solve_group` tries first.
    let mut settled: Option<BondOrder> = None;
    for wanted in [false, true] {
        let assignment = plan_from(
            instance,
            skeleton,
            BondOrdering::HeteroatomFirst,
            &none,
            &none,
            Some((bond, wanted)),
            false,
        );
        let order = assignment
            .bond_orders
            .iter()
            .find(|&&(edge, _)| edge == bond)
            .map_or(BondOrder::Single, |&(_, order)| order);
        match settled {
            None => settled = Some(order),
            Some(first) if first == order => {}
            Some(_) => return None,
        }
    }
    settled
}

/// The valence to ask an atom for, and how many levels it may climb down from
/// it.
///
/// An f-factor asks for one exact degree, and sulfur has three valences to
/// choose between. Enumerating the choices outside the solver -- which is what
/// this replaces -- meant the choice was made before the solver could weigh it,
/// and it had to be capped, so the answer was a lexicographic optimum over a
/// *truncation* of the feasible set rather than over the set. Instead the atom
/// is asked for its largest valence and given one **slack pair** per level it
/// may drop: hypervalent promotion turns a lone pair into two bonds, so every
/// step is exactly two, and a pair either sits inert -- costing the atom two --
/// or is spent. The choice is then a variable like any other, decided in the
/// same canonical order as the rest.
///
/// `None` when the element has no valence that fits its σ bonds. An element
/// whose valences do not step by two gets no ladder and its lowest fitting
/// valence, since the pair gadget cannot express any other step.
fn ladder(options: &[i64]) -> Option<(i64, usize)> {
    let &lowest = options.first()?;
    let &highest = options.last()?;
    let uniform = options
        .windows(2)
        .all(|pair| pair[1] - pair[0] == LADDER_STEP);
    if !uniform {
        return Some((lowest, 0));
    }
    Some((highest, options.len() - 1))
}

/// How much valence one rung of the ladder is worth: a promoted lone pair
/// becomes two bonds.
const LADDER_STEP: i64 = 2;

/// The valence an atom of this element and charge has, given the bonds it
/// already carries.
///
/// The element's *lowest* valence is the wrong question for sulfur and
/// phosphorus: a sulfonamide sulfur has four σ bonds and would come out with a
/// negative capacity, taking every molecule holding one out of scope. This asks
/// for the smallest valence the σ bonds have not already outgrown, which is the
/// same number for the elements that have only one.
fn valence_for(atomic_number: u8, charge: i8, sigma: usize) -> Option<i64> {
    crate::molecule::base::saturating_valency(atomic_number, charge, sigma as i8)
        .map(|valency| valency as i64)
}

/// Every valence this element could be using, given its σ bonds, smallest
/// first.
///
/// The charge matters and used not to be passed: an endpoint has had its charge
/// zeroed by the erasure and gets it back from the pools, but a *non-endpoint*
/// keeps whatever it came with, and asking for neutral carbon's valence of four
/// when the atom is a carbanion with a valence of three makes the group
/// unsolvable. Every one of the corpus's `normalize` failures was that: the
/// input is itself a feasible point of the constraint system, so the feasible
/// set is empty only when the model disagrees with the molecule.
///
/// An f-factor asks each atom for an exact degree, and sulfur will not name
/// one: `NS(=O)(=O)c1ccccc1` has four σ bonds at the sulfur and needs valence
/// six, while a sulfoxide with three needs four. Taking the smallest that fits
/// the σ bonds gives the sulfonamide a capacity of zero, and the only way left
/// to place the two π bonds is to separate a charge -- which is how the neutral
/// molecule came back as `[S+](=O)([O-])(N)c1ccccc1`. So the choice is made by
/// trying: the options are enumerated smallest first and the first that lets
/// the whole group solve is taken, which leaves every single-valence element
/// exactly where it was.
fn valence_options(atomic_number: u8, charge: i8, sigma: i64) -> Vec<i64> {
    use chemistry_consts::ElementProperties;
    let Some(valencies) = atomic_number.valencies() else {
        return Vec::new();
    };
    let shift = crate::molecule::base::charge_valency_shift(atomic_number, charge);
    let mut options: Vec<i64> = valencies
        .filter(|valency| *valency > 0)
        .map(|valency| (valency + shift) as i64)
        .filter(|valency| *valency >= sigma)
        .collect();
    options.sort_unstable();
    options.dedup();
    options
}

/// Works out what a group's atoms should carry, for a given number of positive
/// and negative charges to place.
fn build_instance<M: Molecule + Clone + ToSMILES>(
    molecule: &M,
    group: &TautomericGroup,
    positives: usize,
    negatives: usize,
    used: &[Option<i64>],
    pin_stereocentres: bool,
) -> Option<GroupInstance> {
    use crate::molecule::matching::DegreeConstrainedInstance;

    let index_of: FxHashMap<usize, usize> = group
        .atoms
        .iter()
        .enumerate()
        .map(|(slot, &atom)| (atom, slot))
        .collect();
    let atom_count = group.atoms.len();
    // One pool per species, and they must be separate vertices: a hydrogen edge
    // and a charge edge from the same atom to the same pool are the same edge to
    // the solver, and it will happily take one where the other was meant. That
    // is what silently turned guanidinium into guanidine.
    let hydrogen_pool = atom_count;
    let negative_pool = atom_count + 1;
    let positive_pool = atom_count + 2;
    // The unpaired electrons get a pool for the same reason the charges get one
    // of their own: an electron edge and a hydrogen edge from the same atom to
    // the same pool would be one edge to the solver, and it would take either
    // where the other was meant -- writing a hydrogen where the class has an
    // electron changes the formula.
    let radical_pool = atom_count + 3;
    // Ladder vertices are appended after the pools as they are needed, so the
    // vertex count is not known until the requirements have been worked out.
    let mut required: Vec<usize> = vec![0; atom_count + 4];
    let mut raw_capacity: Vec<i64> = vec![0; atom_count + 4];

    let mut is_endpoint = vec![false; atom_count];
    for &endpoint in &group.endpoints {
        is_endpoint[*index_of.get(&endpoint)?] = true;
    }
    // An atom whose stereochemistry the input states keeps everything it
    // arrived with -- its valence, its charge and its hydrogens. A sulfoxide
    // sulfur is a stereocentre, three substituents and a lone pair, and a
    // second π bond leaves no centre for the tag to describe: the answer either
    // carried a tag the reader refuses or quietly lost the geometry the input
    // had. Pinning the valence alone was not enough, because the positive
    // charge offer hands the atom one more unit to spend and it spent it on
    // exactly that second bond.
    let pinned: Vec<bool> = group
        .atoms
        .iter()
        .map(|&atom| {
            pin_stereocentres
                && matches!(
                    molecule
                        .chiral_classes()
                        .and_then(|classes| classes.get(atom)),
                    Some(ChiralClass::TH(_) | ChiralClass::AL(_))
                )
        })
        .collect();
    // Which endpoints could hold a positive charge has to be known before the
    // valences are asked for, because it moves the floor those valences are
    // filtered against; see below.
    let plus_capable = |slot: usize| {
        is_endpoint[slot]
            && !pinned[slot]
            && takes_positive_charge(molecule.atomic_numbers()[group.atoms[slot]])
    };
    let minus_capable = |slot: usize| {
        is_endpoint[slot]
            && !pinned[slot]
            && takes_negative_charge(molecule.atomic_numbers()[group.atoms[slot]])
    };

    // `c_u`: what the atom's valence has left over once its σ bonds and the
    // hydrogens that were never mobile are taken off. Bonds inside the group
    // are single in the skeleton, so σ is just the degree.
    let owned: FxHashSet<(usize, usize)> = group.bonds.iter().copied().collect();
    // How many valence levels each atom may climb, beyond its lowest.
    let mut ladder_levels = vec![0usize; atom_count];
    for (slot, &atom) in group.atoms.iter().enumerate() {
        let bonds = molecule.get_atom_bonds(atom)?;
        let heavy = bonds
            .iter()
            .filter(|bond| molecule.atomic_numbers()[bond.target()] > 1)
            .count();
        let hydrogens = bonds.len() - heavy;
        // π already committed on bonds the group does not own -- a triple bond
        // out of the group, most often. The group's own bonds are single in the
        // skeleton and spend nothing.
        let spent: i64 = bonds
            .iter()
            .filter(|bond| {
                let target = bond.target();
                !owned.contains(&(atom.min(target), atom.max(target)))
            })
            .map(|bond| match bond.bond_order() {
                BondOrder::Double => 1,
                BondOrder::Triple => 2,
                _ => 0,
            })
            .sum();
        // An endpoint's charge was zeroed by the erasure and comes back from
        // the pools; everything else keeps the charge it arrived with.
        let charge = if is_endpoint[slot] {
            0
        } else {
            molecule.get_atom_charge(atom)
        };
        let measured = used.get(atom).copied().flatten();
        let (valence, levels) = if (is_endpoint[slot] && !pinned[slot]) || measured.is_none() {
            // An endpoint's valence is the table's business: it has to be free
            // to move between a π bond and a hydrogen, and the ladder is how
            // sulfur gets to choose among 2, 4 and 6.
            // The floor is the σ count less whatever a positive charge would
            // add, not the σ count itself. Sulfur's neutral valences are 2, 4
            // and 6, so a sulfonium -- three σ bonds and a `+` -- has none:
            // filtering at three threw its valence of 2 away, the ladder
            // started at 4, and the group came out one unit over-committed and
            // unsolvable. The charge is a *variable* here, so the valence it
            // would license has to stay in the running until the solver has
            // decided whether the atom holds it.
            //
            // Nothing below the floor can be reached without the charge: a rung
            // is worth two, the offer edge is worth one, and the degree is
            // exact, so `2k + [offer given back] ≤ v_max − σ + 1` holds by
            // construction. Give the offer back and it reads `v_max − 2k ≥ σ`;
            // keep it and it reads `v_max − 2k + 1 ≥ σ`. Each is the statement
            // that the valence left over still covers the σ bonds.
            let floor = bonds.len() as i64 - i64::from(plus_capable(slot));
            let options = valence_options(molecule.atomic_numbers()[atom], charge, floor);
            ladder(&options)?
        } else {
            // Everything else keeps exactly the valence the input gave it,
            // which is a property of its element and charge and so the same for
            // every tautomer of the class.
            (measured?, 0)
        };
        ladder_levels[slot] = levels;
        // Unpaired electrons spend valence exactly as bonds and hydrogens do.
        // Leaving them out asked a `[N]` radical -- one bond, two unpaired
        // electrons -- for the two further π bonds its element's valence of
        // three implies, and the only bond it had could not carry them.
        //
        // An endpoint has none left to count: erasure took them off with its
        // charge and its hydrogens, and the pool gives them back below, which
        // spends the capacity reserved here in exactly the same way a hydrogen
        // from the hydrogen pool does. What this still reads is the electron on
        // an atom the group does not count as an endpoint, which no shift moves
        // and which therefore stays part of the atom's fixed commitments.
        let radicals = molecule.radical_states().get(atom).copied().unwrap_or(0) as i64;
        // A labelled hydrogen stays where it is, so an endpoint pays for it
        // exactly as a non-endpoint pays for all of its own.
        let kept_hydrogens = if is_endpoint[slot] {
            bonds
                .iter()
                .filter(|bond| {
                    molecule.atomic_numbers()[bond.target()] == 1
                        && !is_mobile_hydrogen(molecule, bond.target())
                })
                .count() as i64
        } else {
            hydrogens as i64
        };
        let capacity = valence - heavy as i64 - spent - radicals - kept_hydrogens;
        // Kept signed for now. A capacity below zero is not necessarily an
        // error: an atom that only makes sense charged has none as a neutral
        // one, and the slot a positive charge brings is added below. A
        // sulfonium's sulfur is the case -- three bonds, and neutral sulfur's
        // valences are 2, 4 and 6, so as a neutral atom it is a bond
        // over-committed, and the `+` is exactly what pays for it. Rejecting
        // here left 835 of PubChem's first 1,101 unsolvable groups.
        raw_capacity[slot] = capacity;
    }

    // Two parallel edges per bond, not one: a bond takes nought, one or two π
    // units, and one unit apiece is not enough for a nitrile. With a single
    // edge the solver could give the nitrile carbon its second π only by taking
    // it from the neighbouring bond, and then the nitrogen was a π short and
    // took a hydrogen out of the pool to make it up -- `C#N` came back as
    // `=C=N`, with the formula intact and the compound changed.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    // One pair of vertices per rung, each wanting exactly one edge. The atom is
    // asked for its *largest* valence; a rung whose two vertices both match the
    // atom hands it two of that back, dropping it one level, and a rung whose
    // vertices pair with each other instead leaves the atom to find the whole
    // amount from real bonds and hydrogens. So taking the atom-side edges is
    // taking the smaller valence, and preferring them is expressed in the
    // priority order rather than by trying the valences one at a time.
    let mut rung_edges: Vec<usize> = Vec::new();
    for (slot, &levels) in ladder_levels.iter().enumerate() {
        for _ in 0..levels {
            let first = required.len();
            let second = first + 1;
            required.push(1);
            required.push(1);
            rung_edges.push(edges.len());
            edges.push((slot, first));
            rung_edges.push(edges.len());
            edges.push((slot, second));
            edges.push((first, second));
        }
    }

    let mut bond_edges: Vec<(usize, usize, (usize, usize))> = Vec::new();
    let mut bond_classes: Vec<BondClass> = Vec::new();
    for &(first, second) in &group.bonds {
        let (Some(&a), Some(&b)) = (index_of.get(&first), index_of.get(&second)) else {
            continue;
        };
        let single = edges.len();
        edges.push((a, b));
        let second_unit = edges.len();
        edges.push((a, b));
        bond_edges.push((single, second_unit, (first, second)));
        bond_classes.push(bond_class(molecule, (first, second), (a, b), &raw_capacity));
    }

    // How many extra plus/minus pairs the group could possibly want. A pair
    // needs an atom to put each half on, so this is a property of the group
    // rather than a number picked to keep a loop short -- which is what it
    // replaces. `C[N+](=O)[O-]` needs one even though its net charge is zero.
    let separations = (0..atom_count)
        .filter(|&slot| plus_capable(slot))
        .count()
        .min((0..atom_count).filter(|&slot| minus_capable(slot)).count());

    // A positive charge *supplies* capacity rather than spending it, so it is
    // modelled the other way round: every endpoint that could hold one is asked
    // for one more, and the pool takes back all but `positives` of the offers.
    // The endpoints left holding theirs are the ones that get the charge; an
    // endpoint keeping two of them is at `+2`, and its capacity is two the
    // larger, which is the same bookkeeping done twice.
    //
    // One offer apiece. Offering each endpoint as many as the component carries
    // -- which is what a bare `[Se+4]` seemed to need -- lets a single atom keep
    // the lot, and an atom's capacity grows with every offer it keeps: a
    // nona-arginine peptide put all ten of its charges on one carboxyl oxygen
    // and then filled the capacity that bought with ten hydrogens, which is not
    // a molecule and which overflowed the atom's bond list. Selenium(IV) needed
    // nothing here in the end: a bare ion has no heavy neighbour, so it is no
    // group at all, and a charge of more than one on an atom that *is* bonded
    // is not something any tautomer of anything moves around.
    let mut positive_edges: Vec<(usize, usize)> = Vec::new();
    if positives + separations > 0 {
        for slot in 0..atom_count {
            if plus_capable(slot) {
                raw_capacity[slot] += 1;
                positive_edges.push((edges.len(), slot));
                edges.push((slot, positive_pool));
            }
        }
        if positive_edges.len() < positives {
            return None;
        }
    }
    // Now every slot has been offered what it can be, so a capacity still below
    // zero is a real contradiction.
    for slot in 0..atom_count {
        if raw_capacity[slot] < 0 {
            return None;
        }
        required[slot] = raw_capacity[slot] as usize;
    }

    // Hydrogen edges are grouped by how many the atom already holds, so that
    // the priority below can hand out a first hydrogen to every endpoint before
    // any endpoint gets a second. Filling each atom to capacity in turn instead
    // would write guanidinium as `[NH3+]C(N)=N` -- a valid tautomer of it, but
    // not the symmetric form anyone expects.
    let mut hydrogen_rounds: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut hydrogen_edges: Vec<(usize, usize)> = Vec::new();
    let mut charge_edges: Vec<(usize, usize)> = Vec::new();
    let mut radical_edges: Vec<(usize, usize)> = Vec::new();
    // Grouped the way the hydrogens are, so that every endpoint is offered a
    // first electron before any is offered a second. Filling one atom to its
    // capacity instead wrote `[Se]=C=[SeH2]` for selenocarbonyl diselenide --
    // both hydrogens and both electrons on one selenium, which is not what the
    // class holds and not a placement a SMILES can even state.
    let mut radical_rounds: Vec<Vec<(usize, usize)>> = Vec::new();
    for (slot, &atom) in group.atoms.iter().enumerate() {
        if !is_endpoint[slot] {
            continue;
        }
        let atomic_number = molecule.atomic_numbers()[atom];
        for round in 0..required[slot] {
            if hydrogen_rounds.len() <= round {
                hydrogen_rounds.push(Vec::new());
            }
            hydrogen_rounds[round].push((edges.len(), slot));
            hydrogen_edges.push((edges.len(), slot));
            edges.push((slot, hydrogen_pool));
        }
        // How many electrons this endpoint may hold. A nitrene is two on one
        // atom, so the count is not always one -- but it is bounded by what a
        // SMILES can say, and that is what the cap expresses; see
        // [`writable_radicals`].
        if group.mobile_radicals > 0 {
            // The charges this endpoint could end up holding, since which
            // valences it has depends on them: erasure took its charge away and
            // the pools may give one back.
            let mut charges = vec![if is_endpoint[slot] {
                0
            } else {
                molecule.get_atom_charge(atom)
            }];
            if plus_capable(slot) {
                charges.push(1);
            }
            if minus_capable(slot) {
                charges.push(-1);
            }
            let sigma = molecule
                .get_atom_bonds(atom)
                .map(|bonds| {
                    bonds
                        .iter()
                        .filter(|bond| molecule.atomic_numbers()[bond.target()] > 1)
                        .count()
                })
                .unwrap_or(0) as i64;
            let room = writable_radicals(atomic_number, &charges, sigma, required[slot]);
            for round in 0..room {
                if radical_rounds.len() <= round {
                    radical_rounds.push(Vec::new());
                }
                radical_rounds[round].push((edges.len(), slot));
                radical_edges.push((edges.len(), slot));
                edges.push((slot, radical_pool));
            }
        }
        // As many offers as the component's own charge could put on one atom,
        // rather than one per unit of capacity. A second edge to the same pool
        // is a second negative charge on the same atom, so the number that can
        // ever be used is the number being placed -- and where that is one, as
        // it is for nearly every anion, one edge is all the atom needs. The
        // edges this saves are the expensive kind: a pool vertex joins each
        // internal of its own to every external, so its cost grows with the
        // square of its degree, and the pools were half the gadget.
        //
        // The floor of one is what a charge *separation* draws on, since a
        // separation places a negative the totals did not ask for. The ceiling
        // is what an oxide dianion needs: `[O-2]` is one atom holding the whole
        // of a component's charge, and capping every atom at one left it with
        // nowhere to put the second.
        let charge_offers = required[slot].min(negatives.max(1));
        if negatives + separations > 0 && takes_negative_charge(atomic_number) {
            for _ in 0..charge_offers {
                charge_edges.push((edges.len(), slot));
                edges.push((slot, negative_pool));
            }
        }
    }
    // A nitro oxygen is asked for the negative charge before any other atom,
    // so the nitro group keeps its own and a carboxylic acid beside it keeps
    // its proton, rather than the two trading for `[N+](O)=O` and `C(=O)[O-]`.
    let nitro_oxygen = |atom: usize| {
        molecule.get_atom_bonds(atom).is_some_and(|bonds| {
            bonds
                .iter()
                .any(|bond| is_nitro_oxygen(molecule, atom, bond.target()))
        })
    };
    // Then any other oxygen whose only neighbour is a nitrogen with three: an
    // N-oxide's or a nitrone's, for the same reason. Nicotinic acid N-oxide
    // came back as `O[n+]1cccc(c1)C(=O)[O-]`, the acid's proton moved onto
    // the N-oxide. A hydroxamic acid's oxygen has the same σ shape but is a
    // far weaker acid than a carboxylic one, so a nitrogen beside an acyl or
    // sulfonyl atom -- one carrying a terminal oxygen or sulfur -- is left
    // out; the nitro oxygens are asked first for the same reason.
    let terminal_chalcogen = |atom: usize| {
        matches!(molecule.atomic_numbers()[atom], 8 | 16) && heavy_degree(molecule, atom) == 1
    };
    let acyl = |atom: usize| {
        molecule
            .get_atom_bonds(atom)
            .is_some_and(|bonds| bonds.iter().any(|bond| terminal_chalcogen(bond.target())))
    };
    let n_oxide_oxygen = |atom: usize| {
        molecule.atomic_numbers()[atom] == 8
            && heavy_degree(molecule, atom) == 1
            && molecule.get_atom_bonds(atom).is_some_and(|bonds| {
                bonds.iter().any(|bond| {
                    let nitrogen = bond.target();
                    molecule.atomic_numbers()[nitrogen] == 7
                        && heavy_degree(molecule, nitrogen) == 3
                        && molecule.get_atom_bonds(nitrogen).is_some_and(|around| {
                            around
                                .iter()
                                .all(|other| other.target() == atom || !acyl(other.target()))
                        })
                })
            })
    };
    charge_edges.sort_by_key(|&(_, slot)| {
        let atom = group.atoms[slot];
        (!nitro_oxygen(atom), !n_oxide_oxygen(atom))
    });

    // One three-vertex chain per possible separation. `z` either takes the
    // negative pool -- the separation goes unused -- or pairs with `d`, which
    // then leaves `y` no partner but the positive pool, and the separation is
    // spent. The two pools move in opposite directions as separations are
    // added, which is why a single slack vertex will not do: this way the
    // negative pool's degree stays `negatives + separations` and the positive
    // pool's stays `offers - positives` whatever the count turns out to be, so
    // both are constants an f-factor can be asked for.
    let mut unused_separation_edges: Vec<usize> = Vec::new();
    for _ in 0..separations {
        let z = required.len();
        let d = z + 1;
        let y = z + 2;
        required.extend([1, 1, 1]);
        unused_separation_edges.push(edges.len());
        edges.push((z, negative_pool));
        edges.push((z, d));
        edges.push((d, y));
        edges.push((y, positive_pool));
    }

    let vertex_count = required.len();
    required[hydrogen_pool] = group.mobile_hydrogens as usize;
    required[radical_pool] = group.mobile_radicals as usize;
    required[negative_pool] = negatives + separations;
    required[positive_pool] = positive_edges.len().checked_sub(positives)?;

    let solver = DegreeConstrainedInstance::build(vertex_count, &edges, &required)?;
    // The order the positive offers are asked back in depends on how much room
    // each atom has, which the graph fixes, so it is settled here rather than
    // once per pass. An endpoint keeps the charge exactly when it fails to give
    // its offer back, so the roomiest are asked first and the charge is left
    // with the atom that has no capacity of its own: that is what keeps an
    // N-methylpyridinium's charge on the ring nitrogen -- three σ bonds and no
    // room for a π bond unless it is charged -- instead of moving it to an
    // oxime nitrogen that was managing perfectly well, and dearomatising the
    // ring to pay for it.
    //
    // Chalcogens are asked before nitrogen and phosphorus whatever their room,
    // so a cation that can be an ammonium is not written as an oxonium: an
    // ester oxygen has no room either, and on room alone it kept the proton a
    // benzimidazolium was carrying.
    let gives_back_first = |slot: usize| {
        u8::from(matches!(
            molecule.atomic_numbers()[group.atoms[slot]],
            8 | 16 | 34 | 52
        ))
    };
    let mut offers: Vec<(u8, usize, usize)> = positive_edges
        .iter()
        .map(|&(edge, slot)| (gives_back_first(slot), required[slot], edge))
        .collect();
    offers.sort_unstable_by(|first, second| second.cmp(first));
    let offers = offers
        .into_iter()
        .map(|(_, room, edge)| (room, edge))
        .collect();
    // Read from σ bonds alone, like everything else here, so every tautomer
    // of a compound names the same edges.
    // Only where a carbon is an endpoint: elsewhere a ring triple bond can only
    // have come with the input (an aryne), and avoiding it there writes the
    // ring as a cumulene instead, which is no better a reading of it.
    let carbon_endpoint = (0..atom_count)
        .any(|slot| is_endpoint[slot] && molecule.atomic_numbers()[group.atoms[slot]] == 6);
    let (cyclic_atoms, cyclic_bonds) = molecule.find_cyclic_bonds_and_atoms();
    let ring_second_units = bond_edges
        .iter()
        .filter(|&&(_, _, (first, second))| {
            carbon_endpoint && cyclic_bonds.contains(&(first.min(second), first.max(second)))
        })
        .map(|&(_, second_unit, _)| second_unit)
        .collect();
    let ring_carbon_hydrogens = hydrogen_rounds.first().map_or_else(Vec::new, |round| {
        round
            .iter()
            .filter(|&&(_, slot)| {
                let atom = group.atoms[slot];
                molecule.atomic_numbers()[atom] == 6
                    && cyclic_atoms.contains(atom)
                    && heavy_degree(molecule, atom) == 2
            })
            .map(|&(edge, _)| edge)
            .collect()
    });
    Some(GroupInstance {
        solver,
        atoms: group.atoms.clone(),
        bond_edges,
        bond_classes,
        hydrogen_rounds,
        hydrogen_edges,
        charge_edges,
        positive_edges,
        radical_edges,
        radical_rounds,
        rung_edges,
        unused_separation_edges,
        offers,
        ring_second_units,
        ring_carbon_hydrogens,
    })
}

/// One component's constraint system, built once and asked for several
/// assignments.
///
/// What the graph is made of -- the atoms' capacities, the pools, the ladder
/// rungs, the charge separations -- depends on the molecule and the groups
/// alone. What changes between one assignment and the next is only the order
/// the edges are offered in: which bonds are asked first, which are forced
/// π-free to keep a ring, and whether a caller is probing one bond's range.
/// So the instance is built once and each of those is a pass over it.
///
/// `normalize` asks for up to four assignments of one component, and
/// [`settle_geometries`] asks two more per double bond whose geometry might
/// survive. Building for each meant the same gadget constructed and matched a
/// dozen times over for one molecule.
struct GroupInstance {
    solver: crate::molecule::matching::DegreeConstrainedInstance,
    /// The atoms the slots name, in the order the instance numbers them.
    atoms: Vec<usize>,
    bond_edges: Vec<(usize, usize, (usize, usize))>,
    /// How early each of `bond_edges` is offered its π units.
    bond_classes: Vec<BondClass>,
    hydrogen_rounds: Vec<Vec<(usize, usize)>>,
    hydrogen_edges: Vec<(usize, usize)>,
    charge_edges: Vec<(usize, usize)>,
    positive_edges: Vec<(usize, usize)>,
    radical_edges: Vec<(usize, usize)>,
    radical_rounds: Vec<Vec<(usize, usize)>>,
    rung_edges: Vec<usize>,
    unused_separation_edges: Vec<usize>,
    /// The positive offers, roomiest atom first.
    offers: Vec<(usize, usize)>,
    /// The second π unit of every ring bond.
    ring_second_units: Vec<usize>,
    /// A first hydrogen for every ring carbon with two heavy neighbours.
    ring_carbon_hydrogens: Vec<usize>,
}

/// Works out what the component's atoms should carry, for one order of
/// preferences over its bonds.
fn plan_from<M: Molecule + ?Sized>(
    instance: &mut GroupInstance,
    molecule: &M,
    ordering: BondOrdering,
    cyclic: &FxHashSet<(usize, usize)>,
    closed: &FxHashSet<(usize, usize)>,
    probe: Option<((usize, usize), bool)>,
    spread_pools_first: bool,
) -> Assignment {
    // Decide the bonds in canonical order, heteroatom bonds first; the charges
    // and hydrogens follow from them, and are named afterwards so that where a
    // choice does survive it is still made by position rather than by the
    // solver's numbering.
    let mut ordered: Vec<(u8, BondClass, u8, usize, usize, usize, usize)> = instance
        .bond_edges
        .iter()
        .zip(&instance.bond_classes)
        .filter(|&(_, &class)| class != BondClass::Settled)
        .map(|(&(single, second_unit, (first, second)), &class)| {
            let rank = bond_priority(
                molecule.atomic_numbers()[first],
                molecule.atomic_numbers()[second],
            );
            let on_ring = u8::from(!cyclic.contains(&(first.min(second), first.max(second))));
            let leading = match ordering {
                BondOrdering::HeteroatomFirst => 0,
                BondOrdering::RingFirst => on_ring,
            };
            (
                leading,
                class,
                rank,
                first.min(second),
                first.max(second),
                single,
                second_unit,
            )
        })
        .collect();
    ordered.sort_unstable();
    // Each bond is offered *both* its π units before the next bond is offered
    // any. Offering every bond a first unit before any bond a second looks like
    // the conservative order, but it is what broke nitriles: the carbon's
    // second unit was claimed by the neighbouring bond in the first pass, and
    // by the time `C#N` was offered its second there was none of the carbon
    // left, so the nitrogen made the shortfall up with a hydrogen. Nothing
    // over-produces triple bonds under this order, because a second unit is
    // only feasible where both atoms have the capacity for it -- a carbonyl
    // oxygen has one unit and simply refuses the second.
    // The rungs are decided first, and each is taken when it can be, so an atom
    // is only promoted to a higher valence when the group cannot be written
    // without it -- the same "smallest that works" the enumeration used to
    // give, now as part of one lexicographic pass over one instance.
    // Charge separation is decided before anything else and each is left
    // unused when it can be, so the answer carries the fewest charges any
    // writable form does. The separations are interchangeable, so taking them
    // greedily in this order really is the minimum rather than merely a small
    // number.
    // A caller asking what orders one bond can take puts that bond's two π
    // units at the very head of the list, so the first thing the lexicographic
    // pass settles is whether the group can be written with the bond π-free --
    // or, asked the other way round, with as much π as it will hold. Nothing
    // precedes it, so the answer is over the whole feasible set rather than
    // over what earlier preferences have already narrowed it to.
    let mut priority: Vec<(usize, bool)> = match probe {
        Some((probed, wanted)) => instance
            .bond_edges
            .iter()
            .filter(|(_, _, bond)| *bond == probed)
            .flat_map(|&(single, second_unit, _)| [(single, wanted), (second_unit, wanted)])
            .collect(),
        None => Vec::new(),
    };
    // Bonds the caller wants kept free of π: the exocyclic ones at a ring that
    // would otherwise steal the ring's own π bonds. Forced out first, so the
    // ring has no choice but to satisfy itself.
    priority.extend(
        instance
            .bond_edges
            .iter()
            .filter(|(_, _, bond)| closed.contains(bond))
            .flat_map(|&(single, second_unit, _)| [(single, false), (second_unit, false)]),
    );
    // Nitro groups and nitriles before the charge separations: both are
    // written the one way whatever else the group does, and a separation left
    // unused at their expense is how `[N+](=O)[O-]` beside two amide NH came
    // back as `N(O)O`, and `N#CC=CN` as `N=CC=C=N`.
    priority.extend(
        instance
            .bond_edges
            .iter()
            .zip(&instance.bond_classes)
            .filter(|&(_, &class)| class == BondClass::Settled)
            .flat_map(|(&(single, second_unit, _), _)| [(single, true), (second_unit, true)]),
    );
    // No triple bond in a ring, and no ring carbon left with π on both sides,
    // where the group can be written any other way. Neither is something a
    // proton shift leads to in practice, but both are what π-before-hydrogens
    // produces once a ring carbon's own hydrogens are mobile: every bond is
    // offered both its units before the next is offered any, and a ring
    // carbon stripped of its hydrogen has room for two. With carbon endpoints
    // that turned phenol into `O=C1CCCC#C1`, pyridinium into
    // `[N+]#1C=CCCC#1` and put an allene in omeprazole's benzene ring -- 40%
    // of a PubChem sample came back with a ring triple bond. A ring carbon
    // keeping one hydrogen has room for one π bond and no more.
    priority.extend(instance.ring_second_units.iter().map(|&edge| (edge, false)));
    priority.extend(
        instance
            .ring_carbon_hydrogens
            .iter()
            .map(|&edge| (edge, true)),
    );
    // A last-resort ordering that hands every endpoint one hydrogen and one
    // electron before any atom is allowed down its ladder.
    //
    // The ordinary order decides the ladders first, because an atom should sit
    // at the lowest valence the group can be written with. That is right until
    // the pools have nowhere left to go: selenocarbonyl diselenide,
    // `C(=[SeH])=[SeH]`, let its first selenium drop to a valence of two, and
    // the second was then the only endpoint with room -- so it took both
    // hydrogens and both electrons and came out as `[Se]=C=[SeH2]`, which is
    // neither the compound nor a structure SMILES can state. Spreading first
    // forces both seleniums to stay at four, which is the reading the input
    // itself has. Tried only when every ordinary candidate has failed, so no
    // molecule that solves today is decided by it.
    if spread_pools_first {
        if let Some(round) = instance.hydrogen_rounds.first() {
            priority.extend(round.iter().map(|&(edge, _)| (edge, true)));
        }
        if let Some(round) = instance.radical_rounds.first() {
            priority.extend(round.iter().map(|&(edge, _)| (edge, true)));
        }
    }
    priority.extend(
        instance
            .unused_separation_edges
            .iter()
            .map(|&edge| (edge, true)),
    );
    priority.extend(instance.rung_edges.iter().map(|&edge| (edge, true)));
    for entry in &ordered {
        priority.push((entry.5, true));
        priority.push((entry.6, true));
    }
    priority.extend(instance.charge_edges.iter().map(|&(edge, _)| (edge, true)));
    priority.extend(instance.offers.iter().map(|&(_, edge)| (edge, true)));
    // The electrons are placed before the hydrogens and after everything that
    // can be decided by a rule. Both are leftovers by then -- whatever capacity
    // the π bonds and the charges did not take -- and taking the electrons
    // first is what keeps their placement from depending on how many hydrogens
    // happen to be in the pool beside them.
    priority.extend(
        instance
            .radical_rounds
            .iter()
            .flat_map(|round| round.iter().map(|&(edge, _)| (edge, true))),
    );
    priority.extend(
        instance
            .hydrogen_rounds
            .iter()
            .flat_map(|round| round.iter().map(|&(edge, _)| (edge, true))),
    );

    let taken = instance.solver.select(&priority);

    Assignment {
        bond_orders: instance
            .bond_edges
            .iter()
            .map(|&(single, second_unit, bond)| {
                let order = match (taken[single], taken[second_unit]) {
                    (false, false) => BondOrder::Single,
                    (true, true) => BondOrder::Triple,
                    _ => BondOrder::Double,
                };
                (bond, order)
            })
            .filter(|&(_, order)| order != BondOrder::Single)
            .collect(),
        negative: instance
            .charge_edges
            .iter()
            .filter(|&&(edge, _)| taken[edge])
            .map(|&(_, slot)| instance.atoms[slot])
            .collect(),
        // An endpoint that did *not* give its offer back is the one holding the
        // positive charge.
        positive: instance
            .positive_edges
            .iter()
            .filter(|&&(edge, _)| !taken[edge])
            .map(|&(_, slot)| instance.atoms[slot])
            .collect(),
        hydrogens: instance
            .hydrogen_edges
            .iter()
            .filter(|&&(edge, _)| taken[edge])
            .map(|&(_, slot)| instance.atoms[slot])
            .collect(),
        radicals: instance
            .radical_edges
            .iter()
            .filter(|&&(edge, _)| taken[edge])
            .map(|&(_, slot)| instance.atoms[slot])
            .collect(),
    }
}

/// Whether every atom the assignment hands a hydrogen to has room for it.
fn fits_in_the_bond_lists<M: Molecule + ?Sized>(molecule: &M, assignment: &Assignment) -> bool {
    let mut wanted: FxHashMap<usize, usize> = FxHashMap::default();
    for &atom in &assignment.hydrogens {
        *wanted.entry(atom).or_default() += 1;
    }
    wanted.into_iter().all(|(atom, hydrogens)| {
        let held = molecule.get_atom_bonds(atom).map_or(0, |bonds| bonds.len());
        held + hydrogens <= crate::atom::BOND_CAPACITY
    })
}

/// Writes an assignment onto the molecule.
fn apply<M: Molecule + Clone + ToSMILES>(molecule: &mut M, assignment: &Assignment) {
    for &((first, second), order) in &assignment.bond_orders {
        set_bond_order(molecule, first, second, order);
    }
    for &atom in &assignment.negative {
        *molecule.get_atom_charge_mut(atom) -= 1;
    }
    for &atom in &assignment.positive {
        *molecule.get_atom_charge_mut(atom) += 1;
    }
    // Before the hydrogens, while the indices still name the atoms the
    // assignment was written about.
    for &atom in &assignment.radicals {
        let held = molecule.radical_states().get(atom).copied().unwrap_or(0);
        molecule.set_atom_radical_state(atom, held + 1);
    }
    // Hydrogens go on last: adding an atom appends to the end, so it cannot
    // disturb the indices the rest of the assignment names.
    for &atom in &assignment.hydrogens {
        let mut hydrogen = crate::atom::Atom::new(1);
        hydrogen.add_bond(crate::molecule::bond::BondTarget::single(atom));
        molecule.add_atom(hydrogen);
    }

    // A tetrahedral tag on an atom the π bonds have just moved onto describes a
    // geometry that is no longer there. A sulfoxide sulfur is a stereocentre --
    // three substituents and a lone pair -- and `[S@]2=O` says so; give it a
    // second double bond and there is no centre left, and the tag written on it
    // is a string the reader refuses outright. 52 of the corpus's unreadable
    // answers were this one shape.
    let touched: FxHashSet<usize> = assignment
        .bond_orders
        .iter()
        .flat_map(|&((first, second), _)| [first, second])
        .collect();
    let stale: Vec<usize> = touched
        .into_iter()
        .filter(|&atom| {
            if !matches!(
                molecule
                    .chiral_classes()
                    .and_then(|classes| classes.get(atom)),
                Some(ChiralClass::TH(_) | ChiralClass::AL(_))
            ) {
                return false;
            }
            let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();
            let doubles = bonds
                .iter()
                .filter(|bond| bond.bond_order() == BondOrder::Double)
                .count();
            // The shapes the reader accepts: four, five or six neighbours
            // whatever their orders, or three with at most one double bond --
            // the lone-pair centre.
            !matches!((bonds.len(), doubles), (4..=6, _) | (3, 0 | 1))
        })
        .collect();
    if !stale.is_empty()
        && let Some(classes) = molecule.chiral_classes_mut()
    {
        for atom in stale {
            classes[atom] = ChiralClass::None;
        }
    }
}

/// Whether every atom of the group ends up with bonds its element and charge
/// actually allow.
fn valences_are_consistent<M: Molecule + Clone + ToSMILES>(
    molecule: &M,
    group: &TautomericGroup,
    used: &[Option<i64>],
) -> bool {
    group.atoms.iter().all(|&atom| {
        let Some(bonds) = molecule.get_atom_bonds(atom) else {
            return false;
        };
        let spends: i64 = bonds
            .iter()
            .map(|bond| match bond.bond_order() {
                BondOrder::Double => 2,
                BondOrder::Triple => 3,
                _ => 1,
            })
            .sum::<i64>()
            + molecule.radical_states().get(atom).copied().unwrap_or(0) as i64;
        // A valence the input itself was using is allowed whether or not the
        // table lists it. PubChem writes dialkyl H-phosphonates without the
        // P-H, so their phosphorus spends four, and the table offers only 3, 5
        // and 7 -- which threw the correct answer away.
        if used.get(atom).copied().flatten() == Some(spends) {
            return true;
        }
        let charge = molecule.get_atom_charge(atom);
        // The question is whether `used` is a valence this element has, not
        // whether it is the one its σ count alone would suggest: asking with
        // the bond *count* would call a hexavalent sulfur wrong, because four
        // σ bonds already satisfy the valence-4 entry, and throw away the
        // correct neutral sulfonamide in favour of a charge-separated one.
        valence_for(molecule.atomic_numbers()[atom], charge, spends as usize)
            .is_some_and(|valence| valence == spends)
    })
}

/// Sets the bond between two atoms from both sides.
fn set_bond_order<M: Molecule + ?Sized>(
    molecule: &mut M,
    first: usize,
    second: usize,
    order: BondOrder,
) {
    for (from, to) in [(first, second), (second, first)] {
        if let Some(bonds) = molecule.atom_bonds_mut().get_mut(from) {
            for bond in bonds.iter_mut() {
                if bond.target == to {
                    bond.bond_order = order;
                }
            }
        }
    }
}

/// [`normalize`] for a SMILES, returning the canonical tautomer's SMILES.
///
/// Each component is normalized on its own and the results are sorted, so a
/// salt written either way round gives the same string. A component with no
/// solvable group falls back to its plain canonical SMILES.
///
/// ```
/// use oxidane::molecule::tautomer::{normalized_smiles, TautomerRules};
///
/// let rules = TautomerRules::new();
/// assert_eq!(
///     normalized_smiles("Oc1ccccn1", &rules).unwrap(),
///     normalized_smiles("O=c1cccc[nH]1", &rules).unwrap(),
/// );
/// ```
pub fn normalized_smiles(
    smiles: &str,
    rules: &TautomerRules,
) -> Result<String, crate::io::SMILESParseError> {
    with_canonizer(|canonizer| {
        let components = MolecularGraph::from_smiles(smiles)?;
        let mut normalized: Vec<String> = components
            .iter()
            .map(
                |component| match normalize_with(canonizer, component, rules) {
                    Some(molecule) => molecule.to_smiles(),
                    None => canonizer.canonize(component).to_smiles(),
                },
            )
            .collect();
        normalized.sort();
        Ok(normalized.join("."))
    })
}
