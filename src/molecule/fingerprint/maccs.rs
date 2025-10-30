//! The 166 public MACCS structural keys.
//!
//! A key is one bit standing for "this molecule contains that", where the
//! `that` is a substructure written as SMARTS -- a nitro group, a six-membered
//! ring, two heteroatoms bonded together. The set is old, arbitrary, and drawn
//! up for substructure *screening* rather than for similarity, and it is still
//! worth having: 166 bits are cheap to store, the bits mean something a chemist
//! can read off, and a great deal of published work reports MACCS Tanimotos.
//!
//! The definitions are not MDL's, which were never published in full. They are
//! the open reconstruction that ships with RDKit, and so is the numbering: bit
//! `n` here is bit `n` there. Two keys cannot be written as SMARTS at all --
//! 125, more than one aromatic ring, and 166, more than one fragment -- and are
//! computed directly; key 1 was never defined by anyone and is always clear.
//!
//! The definitions were written where hydrogens are implicit, and `*` there
//! means a heavy atom: `*~[#7](~*)~*` is a tertiary amine to whoever wrote it,
//! and would be any primary amine if `*` could be a hydrogen. Hydrogens are
//! atoms in this crate, so every `*` is read as `[!#1]` before the pattern is
//! parsed. Dropping the hydrogens instead would have been the other way to do
//! it, and is worse: the keys that ask how many hydrogens an atom has --
//! `[O;!H0]`, `[!#6;!#1;!H0]` -- would then have to guess the answer back out
//! of the valence.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::fingerprint::maccs_keys;
//!
//! let aspirin = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();
//! let salicylic = MolecularGraph::from_smiles_single("OC(=O)c1ccccc1O").unwrap();
//! let keys = maccs_keys(&aspirin);
//! assert_eq!(keys.len(), 167);
//! // Bit 162 is "aromatic", bit 163 a six-membered ring.
//! assert!(keys.get(162) && keys.get(163));
//! assert!(keys.tanimoto(&maccs_keys(&salicylic)) > 0.5);
//! ```

use super::BitFingerprint;
use crate::io::{CxSMARTSPattern, MoleculeMatchContext, parse_cxsmarts};
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::molecular_graph::MolecularGraph;
use std::sync::LazyLock;

/// How many bits a MACCS key set has: 166 keys, numbered from one, plus an
/// unused bit zero so that a bit's number is its index.
pub const MACCS_KEY_COUNT: usize = 167;

/// Every key that is a pattern: its number, its SMARTS, and how many matches
/// the molecule must have *more* than for the bit to be set.
///
/// A threshold of zero means one match is enough. The handful of keys that
/// appear twice with different thresholds are how the set says "has a
/// heteroatom" and "has two heteroatoms" in separate bits.
const PATTERNS: &[(usize, &str, usize)] = &[
    // 1: not a pattern.
    (2, r"[#104]", 0),
    (3, r"[#32,#33,#34,#50,#51,#52,#82,#83,#84]", 0),
    (4, r"[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]", 0),
    (5, r"[Sc,Ti,Y,Zr,Hf]", 0),
    (6, r"[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]", 0),
    (7, r"[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]", 0),
    (8, r"[!#6;!#1]1~*~*~*~1", 0),
    (9, r"[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]", 0),
    (10, r"[Be,Mg,Ca,Sr,Ba,Ra]", 0),
    (11, r"*1~*~*~*~1", 0),
    (12, r"[Cu,Zn,Ag,Cd,Au,Hg]", 0),
    (13, r"[#8]~[#7](~[#6])~[#6]", 0),
    (14, r"[#16]-[#16]", 0),
    (15, r"[#8]~[#6](~[#8])~[#8]", 0),
    (16, r"[!#6;!#1]1~*~*~1", 0),
    (17, r"[#6]#[#6]", 0),
    (18, r"[#5,#13,#31,#49,#81]", 0),
    (19, r"*1~*~*~*~*~*~*~1", 0),
    (20, r"[#14]", 0),
    (21, r"[#6]=[#6](~[!#6;!#1])~[!#6;!#1]", 0),
    (22, r"*1~*~*~1", 0),
    (23, r"[#7]~[#6](~[#8])~[#8]", 0),
    (24, r"[#7]-[#8]", 0),
    (25, r"[#7]~[#6](~[#7])~[#7]", 0),
    (26, r"[#6]=;@[#6](@*)@*", 0),
    (27, r"[I]", 0),
    (28, r"[!#6;!#1]~[CH2]~[!#6;!#1]", 0),
    (29, r"[#15]", 0),
    (30, r"[#6]~[!#6;!#1](~[#6])(~[#6])~*", 0),
    (31, r"[!#6;!#1]~[F,Cl,Br,I]", 0),
    (32, r"[#6]~[#16]~[#7]", 0),
    (33, r"[#7]~[#16]", 0),
    (34, r"[CH2]=*", 0),
    (35, r"[Li,Na,K,Rb,Cs,Fr]", 0),
    (36, r"[#16R]", 0),
    (37, r"[#7]~[#6](~[#8])~[#7]", 0),
    (38, r"[#7]~[#6](~[#6])~[#7]", 0),
    (39, r"[#8]~[#16](~[#8])~[#8]", 0),
    (40, r"[#16]-[#8]", 0),
    (41, r"[#6]#[#7]", 0),
    (42, r"F", 0),
    (43, r"[!#6;!#1;!H0]~*~[!#6;!#1;!H0]", 0),
    (44, r"[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]", 0),
    (45, r"[#6]=[#6]~[#7]", 0),
    (46, r"Br", 0),
    (47, r"[#16]~*~[#7]", 0),
    (48, r"[#8]~[!#6;!#1](~[#8])(~[#8])", 0),
    (49, r"[!+0]", 0),
    (50, r"[#6]=[#6](~[#6])~[#6]", 0),
    (51, r"[#6]~[#16]~[#8]", 0),
    (52, r"[#7]~[#7]", 0),
    (53, r"[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]", 0),
    (54, r"[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]", 0),
    (55, r"[#8]~[#16]~[#8]", 0),
    (56, r"[#8]~[#7](~[#8])~[#6]", 0),
    (57, r"[#8R]", 0),
    (58, r"[!#6;!#1]~[#16]~[!#6;!#1]", 0),
    (59, r"[#16]!:*:*", 0),
    (60, r"[#16]=[#8]", 0),
    (61, r"*~[#16](~*)~*", 0),
    (62, r"*@*!@*@*", 0),
    (63, r"[#7]=[#8]", 0),
    (64, r"*@*!@[#16]", 0),
    (65, r"c:n", 0),
    (66, r"[#6]~[#6](~[#6])(~[#6])~*", 0),
    (67, r"[!#6;!#1]~[#16]", 0),
    (68, r"[!#6;!#1;!H0]~[!#6;!#1;!H0]", 0),
    (69, r"[!#6;!#1]~[!#6;!#1;!H0]", 0),
    (70, r"[!#6;!#1]~[#7]~[!#6;!#1]", 0),
    (71, r"[#7]~[#8]", 0),
    (72, r"[#8]~*~*~[#8]", 0),
    (73, r"[#16]=*", 0),
    (74, r"[CH3]~*~[CH3]", 0),
    (75, r"*!@[#7]@*", 0),
    (76, r"[#6]=[#6](~*)~*", 0),
    (77, r"[#7]~*~[#7]", 0),
    (78, r"[#6]=[#7]", 0),
    (79, r"[#7]~*~*~[#7]", 0),
    (80, r"[#7]~*~*~*~[#7]", 0),
    (81, r"[#16]~*(~*)~*", 0),
    (82, r"*~[CH2]~[!#6;!#1;!H0]", 0),
    (83, r"[!#6;!#1]1~*~*~*~*~1", 0),
    (84, r"[NH2]", 0),
    (85, r"[#6]~[#7](~[#6])~[#6]", 0),
    (86, r"[C;H2,H3][!#6;!#1][C;H2,H3]", 0),
    (87, r"[F,Cl,Br,I]!@*@*", 0),
    (88, r"[#16]", 0),
    (89, r"[#8]~*~*~*~[#8]", 0),
    (90, r"[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]", 0),
    (91, r"[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]", 0),
    (92, r"[#8]~[#6](~[#7])~[#6]", 0),
    (93, r"[!#6;!#1]~[CH3]", 0),
    (94, r"[!#6;!#1]~[#7]", 0),
    (95, r"[#7]~*~*~[#8]", 0),
    (96, r"*1~*~*~*~*~1", 0),
    (97, r"[#7]~*~*~*~[#8]", 0),
    (98, r"[!#6;!#1]1~*~*~*~*~*~1", 0),
    (99, r"[#6]=[#6]", 0),
    (100, r"*~[CH2]~[#7]", 0),
    (101, r"[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]", 0),
    (102, r"[!#6;!#1]~[#8]", 0),
    (103, r"Cl", 0),
    (104, r"[!#6;!#1;!H0]~*~[CH2]~*", 0),
    (105, r"*@*(@*)@*", 0),
    (106, r"[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]", 0),
    (107, r"[F,Cl,Br,I]~*(~*)~*", 0),
    (108, r"[CH3]~*~*~*~[CH2]~*", 0),
    (109, r"*~[CH2]~[#8]", 0),
    (110, r"[#7]~[#6]~[#8]", 0),
    (111, r"[#7]~*~[CH2]~*", 0),
    (112, r"*~*(~*)(~*)~*", 0),
    (113, r"[#8]!:*:*", 0),
    (114, r"[CH3]~[CH2]~*", 0),
    (115, r"[CH3]~*~[CH2]~*", 0),
    (116, r"[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]", 0),
    (117, r"[#7]~*~[#8]", 0),
    (118, r"[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]", 1),
    (119, r"[#7]=*", 0),
    (120, r"[!#6;R]", 1),
    (121, r"[#7;R]", 0),
    (122, r"*~[#7](~*)~*", 0),
    (123, r"[#8]~[#6]~[#8]", 0),
    (124, r"[!#6;!#1]~[!#6;!#1]", 0),
    // 125: not a pattern.
    (126, r"*!@[#8]!@*", 0),
    (127, r"*@*!@[#8]", 1),
    (128, r"[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]", 0),
    (129, r"[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]", 0),
    (130, r"[!#6;!#1]~[!#6;!#1]", 1),
    (131, r"[!#6;!#1;!H0]", 1),
    (132, r"[#8]~*~[CH2]~*", 0),
    (133, r"*@*!@[#7]", 0),
    (134, r"[F,Cl,Br,I]", 0),
    (135, r"[#7]!:*:*", 0),
    (136, r"[#8]=*", 1),
    (137, r"[!C;!c;R]", 0),
    (138, r"[!#6;!#1]~[CH2]~*", 1),
    (139, r"[O;!H0]", 0),
    (140, r"[#8]", 3),
    (141, r"[CH3]", 2),
    (142, r"[#7]", 1),
    (143, r"*@*!@[#8]", 0),
    (144, r"*!:*:*!:*", 0),
    (145, r"*1~*~*~*~*~*~1", 1),
    (146, r"[#8]", 2),
    (147, r"[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]", 0),
    (148, r"*~[!#6;!#1](~*)~*", 0),
    (149, r"[C;H3,H4]", 1),
    (150, r"*!@*@*!@*", 0),
    (151, r"[#7;!H0]", 0),
    (152, r"[#8]~[#6](~[#6])~[#6]", 0),
    (153, r"[!#6;!#1]~[CH2]~*", 0),
    (154, r"[#6]=[#8]", 0),
    (155, r"*!@[CH2]!@*", 0),
    (156, r"[#7]~*(~*)~*", 0),
    (157, r"[#6]-[#8]", 0),
    (158, r"[#6]-[#7]", 0),
    (159, r"[#8]", 1),
    (160, r"[C;H3,H4]", 0),
    (161, r"[#7]", 0),
    (162, r"a", 0),
    (163, r"*1~*~*~*~*~*~1", 0),
    (164, r"[#8]", 0),
    (165, r"[R]", 0),
    // 166: not a pattern.
];

/// The patterns, parsed once.
///
/// Parsing 165 SMARTS costs far more than matching them, and the set never
/// changes, so it happens on first use and never again.
static COMPILED: LazyLock<Vec<(usize, CxSMARTSPattern, usize)>> = LazyLock::new(|| {
    PATTERNS
        .iter()
        .filter_map(|&(key, smarts, threshold)| {
            // A pattern this crate's parser cannot read would silently clear
            // its bit for every molecule, which is worse than saying so.
            // `*` in a published pattern means a heavy atom, and hydrogens
            // are atoms here.
            let smarts = smarts.replace('*', "[!#1]");
            let pattern = parse_cxsmarts(&smarts)
                .unwrap_or_else(|error| panic!("MACCS key {key} ({smarts}) does not parse: {error}"));
            Some((key, pattern, threshold))
        })
        .collect()
});

/// The molecule's MACCS keys, as 167 bits with bit zero unused.
pub fn maccs_keys(molecule: &MolecularGraph) -> BitFingerprint {
    let mut keys = BitFingerprint::new(MACCS_KEY_COUNT);
    // One analysis of the molecule for all 165 patterns. Letting each pattern
    // make its own enumerates every simple cycle in the molecule a hundred and
    // sixty-five times, which is the whole cost of the fingerprint.
    let context = MoleculeMatchContext::new(molecule);

    for (key, pattern, threshold) in COMPILED.iter() {
        let set = if *threshold == 0 {
            pattern.matches_with(molecule, &context)
        } else {
            distinct_matches(pattern, molecule, &context) > *threshold
        };
        if set {
            keys.set(*key);
        }
    }

    if aromatic_ring_count(molecule) > 1 {
        keys.set(125);
    }
    if fragment_count(molecule) > 1 {
        keys.set(166);
    }
    keys
}

/// How many *different* sets of atoms the pattern covers.
///
/// A six-membered ring pattern lays over a benzene twelve ways -- six starting
/// atoms, two directions -- and MACCS asks how many six-membered rings there
/// are, which is one. Counting the atom sets rather than the mappings is what
/// makes "more than one" mean what a chemist reading the key list expects.
fn distinct_matches(
    pattern: &CxSMARTSPattern,
    molecule: &MolecularGraph,
    context: &MoleculeMatchContext,
) -> usize {
    let mut covered: Vec<Vec<usize>> = pattern
        .find_matches_with(molecule, context)
        .into_iter()
        .map(|mut atoms| {
            atoms.sort_unstable();
            atoms
        })
        .collect();
    covered.sort_unstable();
    covered.dedup();
    covered.len()
}

/// How many rings the aromatic part of the molecule has.
///
/// The count of independent cycles among the aromatic bonds, which is what the
/// smallest set of smallest rings has as many of: edges minus atoms plus
/// connected pieces. Benzene has one, naphthalene two, biphenyl two -- the bond
/// joining its rings is not aromatic and so is not part of either.
fn aromatic_ring_count(molecule: &MolecularGraph) -> usize {
    let atom_count = molecule.len();
    let mut in_aromatic = vec![false; atom_count];
    let mut bonds = 0usize;
    for atom in 0..atom_count {
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            if bond.bond_order() == BondOrder::Aromatic {
                in_aromatic[atom] = true;
                if atom < bond.target() {
                    bonds += 1;
                }
            }
        }
    }

    let mut seen = vec![false; atom_count];
    let mut atoms = 0usize;
    let mut pieces = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..atom_count {
        if seen[start] || !in_aromatic[start] {
            continue;
        }
        pieces += 1;
        stack.push(start);
        seen[start] = true;
        while let Some(current) = stack.pop() {
            atoms += 1;
            for bond in molecule.get_atom_bonds(current).unwrap_or_default() {
                if bond.bond_order() == BondOrder::Aromatic && !seen[bond.target()] {
                    seen[bond.target()] = true;
                    stack.push(bond.target());
                }
            }
        }
    }

    bonds + pieces - atoms
}

/// How many disconnected pieces the molecule is in.
fn fragment_count(molecule: &MolecularGraph) -> usize {
    let atom_count = molecule.len();
    let mut seen = vec![false; atom_count];
    let mut pieces = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..atom_count {
        if seen[start] {
            continue;
        }
        pieces += 1;
        stack.push(start);
        seen[start] = true;
        while let Some(current) = stack.pop() {
            for bond in molecule.get_atom_bonds(current).unwrap_or_default() {
                if !seen[bond.target()] {
                    seen[bond.target()] = true;
                    stack.push(bond.target());
                }
            }
        }
    }
    pieces
}
