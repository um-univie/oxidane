use super::CipConfigs;
use crate::molecule::base::{Molecule, MoleculeCore};
use crate::molecule::bond::BondOrder;
use crate::molecule::cip::CipConfig;
use crate::molecule::graph::{AtomLabel, MoleculeGraph};
use bit_set::BitSet;
use rustc_hash::FxHashMap;

/// Stereochemistry that CIP does not describe.
///
/// The CIP rules — and RDKit's labeller, and this crate's port of it — only
/// cover tetrahedral centres and double bonds. Octahedral,
/// trigonal-bipyramidal, square-planar and allene centres therefore get
/// [`CipConfig::None`], which without this would make two different isomers
/// completely interchangeable: `S[Co@OH1](F)(I)(Cl)(C)Br` and its `@OH2`
/// isomer would match on every one of their seven atoms.
///
/// The number carried here is the SMILES chirality index read in the
/// molecule's *canonical* atom ordering (see
/// [`crate::graph_algorithms::mol_canonization::canonical_chiral_classes`]),
/// not the one it happened to be written with, so equivalent spellings agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum HigherStereo {
    #[default]
    None,
    Octahedral(u8),
    TrigonalBipyramidal(u8),
    SquarePlanar(u8),
    Allene(u8),
}

/// What makes two atoms interchangeable: everything
/// [`AtomLabel`] carries (element, charge, aromaticity, isotope, and
/// *whether* the atom is a stereocentre), its CIP absolute configuration,
/// which is what actually distinguishes R from S, and — for the geometries
/// CIP says nothing about — its [`HigherStereo`].
///
/// The CIP part is essential because [`AtomLabel`] alone collapses both
/// handednesses to `ChiralType::TH`. It is deliberately *not* the raw
/// `ChiralClass` @/@@ parity: that is defined relative to the order the
/// atom's neighbours happen to be listed in, so the same molecule written
/// two ways yields different tags (`N[C@@H](C)C(=O)O` gives `TH(2)` and
/// `C[C@H](N)C(=O)O` gives `TH(1)` for the same L-alanine), which would
/// produce false mismatches. CIP descriptors are order-independent, and the
/// [`HigherStereo`] fallback is canonicalised for the same reason.
pub(super) type StereoLabel = (AtomLabel, CipConfig, HigherStereo);

/// Total-order key for [`StereoLabel`] — [`CipConfig`] has no `Ord`, so
/// [`super::line_graph`] uses this to canonically order a bond's two
/// endpoint labels.
pub(super) fn stereo_label_key(label: &StereoLabel) -> (AtomLabel, u8, HigherStereo) {
    let cip = match label.1 {
        CipConfig::None => 0,
        CipConfig::R => 1,
        CipConfig::S => 2,
        CipConfig::E => 3,
        CipConfig::Z => 4,
        // Pseudo-asymmetric centres are a distinct stereochemical class, not a
        // variant spelling of R/S: mirroring the molecule leaves them
        // unchanged, so they must not be interchangeable when matching.
        CipConfig::PseudoR => 5,
        CipConfig::PseudoS => 6,
        CipConfig::SeqCis => 7,
        CipConfig::SeqTrans => 8,
    };
    (label.0, cip, label.2)
}

/// A molecule reduced to its heavy-atom skeleton, indexed compactly (0..n,
/// no gaps) rather than by the source molecule's own atom indices.
/// `original_index[i]` records which atom of the source molecule compact
/// index `i` came from. Explicit hydrogens are excluded for the same reason
/// [`super::super::mces`] excludes them: matching interchangeable terminal
/// hydrogens against each other isn't a meaningful structural comparison.
pub(super) struct Graph {
    pub labels: Vec<StereoLabel>,
    pub adjacency: Vec<BitSet>,
    bond_order: FxHashMap<(usize, usize), BondOrder>,
    pub original_index: Vec<usize>,
}

impl Graph {
    pub fn from_molecule<T: Molecule + CipConfigs>(mol: &T) -> Self {
        // Computed once for the whole molecule and then indexed: CIP
        // assignment walks a hierarchical digraph over the entire structure,
        // so it costs roughly 20x a SMILES parse (~0.4 ms for 50 heavy
        // atoms). Per-atom recomputation would dominate the entire search.
        let cip = mol.cip_configs();
        let higher = mol.higher_order_stereo();

        let mut labels = Vec::new();
        let mut original_index = Vec::new();
        let mut compact_of: FxHashMap<usize, usize> = FxHashMap::default();

        for (index, &atomic_number) in mol.atomic_numbers().iter().enumerate() {
            if atomic_number == 1 {
                continue;
            }
            compact_of.insert(index, labels.len());
            let cip_config = cip.get(index).copied().unwrap_or(CipConfig::None);
            let higher_stereo = higher.get(index).copied().unwrap_or_default();
            labels.push((mol.get_atom_label(index), cip_config, higher_stereo));
            original_index.push(index);
        }

        let n = labels.len();
        let mut adjacency = vec![BitSet::new(); n];
        let mut bond_order = FxHashMap::default();

        for (index, bonds) in mol.atom_bonds().iter().enumerate() {
            let Some(&source) = compact_of.get(&index) else {
                continue;
            };
            for bond in bonds {
                let Some(&target) = compact_of.get(&bond.target()) else {
                    continue;
                };
                if source < target {
                    adjacency[source].insert(target);
                    adjacency[target].insert(source);
                    bond_order.insert((source, target), bond.bond_order());
                }
            }
        }

        Self { labels, adjacency, bond_order, original_index }
    }

    /// The bond order between compact indices `a` and `b`, or `None` if
    /// they aren't bonded.
    pub fn bond_order(&self, a: usize, b: usize) -> Option<BondOrder> {
        let key = if a < b { (a, b) } else { (b, a) };
        self.bond_order.get(&key).copied()
    }

    /// Every bond as `((atom1, atom2), order)`, atom indices compact and
    /// `atom1 < atom2`. Used to build the line graph (bonds-as-vertices)
    /// representation for edge-maximizing search.
    pub fn bonds(&self) -> impl Iterator<Item = ((usize, usize), BondOrder)> + '_ {
        self.bond_order.iter().map(|(&pair, &order)| (pair, order))
    }

    /// The graph containing exactly `keep_bonds` (given as compact index
    /// pairs of *this* graph) and the atoms they connect, re-compacted.
    ///
    /// Bonds are the unit here rather than atoms, because keeping "every
    /// atom touched by a matched bond" would silently re-introduce bonds
    /// *between* those atoms that were not themselves matched — inflating
    /// the next round of an n-ary reduction with structure that isn't
    /// actually common to the molecules seen so far.
    ///
    /// `original_index` is carried through, so the result still reports
    /// atom indices in the original source molecule's numbering.
    pub fn edge_subgraph(&self, keep_bonds: &[(usize, usize)]) -> Graph {
        let mut kept_atoms: Vec<usize> = keep_bonds.iter().flat_map(|&(a, b)| [a, b]).collect();
        kept_atoms.sort_unstable();
        kept_atoms.dedup();

        let compact_of: FxHashMap<usize, usize> =
            kept_atoms.iter().enumerate().map(|(new, &old)| (old, new)).collect();

        let labels = kept_atoms.iter().map(|&old| self.labels[old]).collect();
        let original_index = kept_atoms.iter().map(|&old| self.original_index[old]).collect();

        let mut adjacency = vec![BitSet::new(); kept_atoms.len()];
        let mut bond_order = FxHashMap::default();
        for &(a, b) in keep_bonds {
            let (na, nb) = (compact_of[&a], compact_of[&b]);
            adjacency[na].insert(nb);
            adjacency[nb].insert(na);
            let key = if na < nb { (na, nb) } else { (nb, na) };
            bond_order.insert(key, self.bond_order(a, b).expect("keep_bonds must be real bonds"));
        }

        Graph { labels, adjacency, bond_order, original_index }
    }
}

impl super::search::McsplitGraph for Graph {
    type NodeLabel = StereoLabel;
    type EdgeLabel = BondOrder;

    fn len(&self) -> usize {
        self.labels.len()
    }

    fn node_label(&self, index: usize) -> StereoLabel {
        self.labels[index]
    }

    fn edge_label(&self, a: usize, b: usize) -> Option<BondOrder> {
        self.bond_order(a, b)
    }

    fn degree(&self, index: usize) -> usize {
        self.adjacency[index].len()
    }
}
