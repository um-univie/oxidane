//! UFF/MMFF-lite refinement terms (angle-bend, torsion) for the final
//! touch-up pass (see [`super`]'s module docs). Bond-stretch and van-der-
//! Waals-style non-bonded repulsion are *not* duplicated here - both are
//! already enforced by the bounds-violation energy in [`super::refine`] (a
//! tight +/-0.02 A flat bottom for 1-2 pairs, and a VDW-radius-based lower
//! bound for every non-bonded pair) - so this module only adds the two
//! terms distance geometry alone cannot express: a real restoring force
//! toward a *single* target angle (rather than "anywhere inside a wide
//! tolerance band is free", which is what let the cocaine phenyl ring in
//! testing relax into an irregular hexagon), and a torsion-periodicity
//! preference (staggered-vs-eclipsed, planar-vs-twisted) that a pairwise
//! distance bound has no way to represent at all.

use super::bounds::{Geometry, angle_bounds_for_triple};
use bit_set::BitSet;
use super::torsion_preferences::TORSION_PREFERENCES;
use crate::io::{CxSMARTSPattern, find_smarts_matches, parse_cxsmarts};
use crate::molecule::molecular_graph::MolecularGraph;
use crate::molecule::utils::order_tuple;
use rustc_hash::{FxHashMap, FxHashSet};
use crate::io::{AtomExpression, LogicalOperator};
use crate::molecule::{base::MoleculeCore, bond::BondOrder, chirality::ChiralClass};
use chemistry_consts::ElementProperties;
use crate::vector::Vector3D;

/// Harmonic-in-cosine force constant, soft relative to the bounds/chirality
/// margins for the same reason `planarity::PLANARITY_K` is - see that
/// module's doc comment.
const K_ANGLE: f64 = 0.4;

/// Torsion barrier heights (soft, arbitrary energy units - *not* calibrated
/// to real kcal/mol values, chosen only to produce a stable, gentle
/// preference without dominating the bounds/stereo/planarity terms it runs
/// alongside in the final touch-up pass). See [`collect_torsions`] for the
/// classification these correspond to.
///
/// `V_STAGGERED` was 0.15 for as long as the sp3-sp3 term's sign put its
/// minima on the eclipsed positions -- see [`collect_torsions`] -- which is
/// to say for as long as the number did not matter. With the sign right the
/// term bites, and nine of them ride on one bond (every pair of
/// substituents, all sharing the same `cos(3 phi)`), so the barrier wants to
/// be lower. Measured at one trial with no torsion search, correct sign
/// throughout:
///
/// | barrier | curated 72 | PubChem 287 |
/// |---|---|---|
/// | 0.15 | +0.06 | +1.69 |
/// | **0.05** | **-0.28** | +1.65 |
/// | 0.02 | -0.08 | +1.70 |
///
/// Against the broken sign's +0.13 on the curated corpus, 0.05 is worth 0.41
/// kcal/mol there, about three times what this metric drifts when a change
/// only reorders a sum. The larger corpora are a wash the other way by 0.1
/// to 0.2, which is inside their own drift; `NEXT.md` has all four.
const V_CONJUGATED: f64 = 0.5;
const V_STAGGERED: f64 = 0.05;
const V_NEARFREE: f64 = 0.05;

/// Brings the experimental force constants of
/// [`super::torsion_preferences`] onto the scale the rest of this module
/// works in.
///
/// They are published in kcal/mol and run from 0.1 to 100, with a median
/// rule totalling 5.0. Everything else here is in the arbitrary units
/// [`V_CONJUGATED`] and friends are stated in -- uncalibrated,
/// chosen to sit alongside the bounds and stereo terms without dominating
/// them. Dropped in unscaled the preferences did dominate: a peptide, whose
/// amide rules carry some of the largest constants in the set, went from an
/// MMFF energy of 503 to 762.
///
/// The anchor is that an experimental rule *replaces* the generic term for
/// its bond rather than joining it, and the two describe the same thing, so
/// a typical rule should carry about the weight it displaces:
/// `V_CONJUGATED / 5.0`.
const EXPERIMENTAL_TORSION_SCALE: f64 = V_CONJUGATED / 5.0;

/// A stereo double bond's substituents are restrained with a *one*-fold
/// cosine, not the two-fold one a generic conjugated bond gets. Two-fold has
/// minima at both 0 and 180 degrees, so it is happy to flatten a bond into
/// either, so it is side-agnostic. One-fold has a single minimum, placed by
/// the sign of its single cosine term at whichever of the two the stereo tag
/// calls for, so it flattens *and* holds the assigned side rather
/// than competing with it.
const STEREO_PLANARITY_FOLD: usize = 1;

/// Barrier for those one-fold terms. Above the generic conjugated value
/// because this term has to be the thing that makes a stereo double bond
/// planar: `chirality::CisTransBond` cannot, being a hinge that stops
/// pulling once its scalar clears a fixed margin, which for real bond
/// lengths it does while the substituents are still around 105 degrees out
/// of plane.
///
/// Above the strongest *experimental* preference too, which after
/// [`EXPERIMENTAL_TORSION_SCALE`] tops out at 10. Those preferences apply to
/// the single bonds flanking a tagged double bond, and at 1.0 this term lost
/// to them: `C/C=C/C` came out at 100 degrees instead of 180. A cis/trans
/// tag is a stated fact about the molecule, not a statistical tendency, so
/// nothing derived from a distribution should be able to outvote it.
const V_STEREO_PLANAR: f64 = 15.0;

/// Step size for the central finite differences that
/// [`torsion_gradient_matches_finite_difference`] checks the analytic
/// dihedral derivative against.
#[cfg(test)]
const FD_STEP: f64 = 1e-5;

pub(super) struct AngleBend {
    i: usize,
    k: usize,
    j: usize,
    cos_theta0: f64,
}

pub(super) struct Torsion {
    i: usize,
    b: usize,
    c: usize,
    j: usize,
    /// `(sign, force constant)` for each of the six cosine terms, giving
    /// `V(phi) = sum_n V_n * (1 + s_n * cos(n * phi))`. The generic rules
    /// and the stereo-planarity rule fill in a single term; the experimental
    /// preferences in [`super::torsion_preferences`] use all six.
    terms: [(f64, f64); 6],
}

/// One `(sign, force constant)` term of a torsion, as `Torsion::terms` holds
/// them, for the rules that only need one.
fn single_term(fold: usize, sign: f64, force_constant: f64) -> [(f64, f64); 6] {
    let mut terms = [(1.0, 0.0); 6];
    terms[fold - 1] = (sign, force_constant);
    terms
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Hyb {
    Sp,
    Sp2,
    Sp3,
}

/// Same triple/double/aromatic bond-order classification tree as
/// `bounds::estimate_hybridization_angle` and `planarity::is_sp2_atom`, kept
/// as its own small local classifier here (each of those two modules
/// already does the same, for their own narrower needs) rather than a third
/// shared function.
fn hybridization<T: MoleculeCore>(mol: &T, atom: usize) -> Hyb {
    let Some(bonds) = mol.get_atom_bonds(atom) else {
        return Hyb::Sp3;
    };
    let double_count = bonds.iter().filter(|b| b.bond_order() == BondOrder::Double).count();
    let has_triple = bonds.iter().any(|b| b.bond_order() == BondOrder::Triple);
    let has_aromatic_or_double = bonds
        .iter()
        .any(|b| matches!(b.bond_order(), BondOrder::Aromatic | BondOrder::Double));
    if has_triple || double_count >= 2 {
        Hyb::Sp
    } else if has_aromatic_or_double {
        Hyb::Sp2
    } else {
        Hyb::Sp3
    }
}

/// Collects a harmonic-in-cosine angle-bend restraint for every bonded 1-3
/// triple `(i, k, j)` - the same triples `bounds::build_bounds` iterates -
/// targeting the midpoint of `angle_bounds_for_triple`'s `(min, max)`, which
/// already folds in the hybridization heuristic, ring-size override, and
/// square-planar trans/adjacent override.
pub(super) fn collect_angle_bends<T: MoleculeCore>(
    mol: &T,
    lengths: &Geometry,
    cycles: &FxHashSet<BitSet>,
) -> Vec<AngleBend> {
    let mut angles = Vec::new();
    for k in 0..mol.len() {
        let Some(bonds_k) = mol.get_atom_bonds(k) else {
            continue;
        };
        let neighbors: Vec<usize> = bonds_k.iter().map(|b| b.target()).collect();
        for a in 0..neighbors.len() {
            for b in (a + 1)..neighbors.len() {
                let i = neighbors[a];
                let j = neighbors[b];
                let (theta_min, theta_max) = angle_bounds_for_triple(mol, i, k, j, cycles, lengths);
                let theta0 = 0.5 * (theta_min + theta_max);
                angles.push(AngleBend { i, k, j, cos_theta0: theta0.cos() });
            }
        }
    }
    angles
}

/// A bond whose ends are both tagged `CisTransClockwise`/
/// `CisTransCounterclockwise`, which gets the one-fold restraints of
/// [`collect_stereo_planarity_torsions`] instead of this module's generic
/// term.
///
/// The generic term is wrong for these bonds: being two-fold it is
/// side-agnostic, so on a `F/C=C/F` fixture it pulls the dihedral toward a
/// near-90-degree saddle between its two minima while
/// `chirality::CisTransBond` pulls toward one side. Skipping the bond is not
/// enough either, because `CisTransBond` cannot flatten the bond on its own
/// -- it is a hinge on a scalar with units of length^4, and
/// stops pulling once that scalar clears a fixed margin, which happens
/// while the substituents are still about 105 degrees out of plane.
/// Without a torsion term: 109 degrees for `C/C=C/C(=O)O`, 113 for
/// `F/C=C/F`, 135 for `C/C=C/C`, all of which should be 180. A one-fold
/// term aimed at the side the tag asks for both flattens the bond
/// and agrees with `CisTransBond` about which side, rather than splitting
/// the difference with it.
fn is_explicit_cis_trans_bond<T: MoleculeCore>(mol: &T, b: usize, c: usize) -> bool {
    let Some(classes) = mol.chiral_classes() else {
        return false;
    };
    let is_tag = |class: ChiralClass| {
        matches!(class, ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise)
    };
    is_tag(classes[b]) && is_tag(classes[c])
}

/// The one-fold, phase-placed torsions that keep a stereo double bond flat.
///
/// Every `(i, b, c, j)` quadruple around the bond gets one, including those
/// involving the hydrogens `add_hydrogens` put there, so the whole
/// six-atom fragment is restrained rather than just the pair the tag names.
/// Which side each quadruple wants follows from the one pair the tag *does*
/// name: stepping from that reference substituent to the other one on the
/// same end swaps the quadruple's target, and likewise at the far end.
fn collect_stereo_planarity_torsions<T: MoleculeCore>(
    mol: &T,
    b: usize,
    c: usize,
    torsions: &mut Vec<Torsion>,
) {
    let Some(reference) = super::chirality::stereo_double_bond_reference(mol, b, c) else {
        return;
    };
    for i_bond in mol.get_atom_bonds(b).unwrap_or_default() {
        let i = i_bond.target();
        if i == c {
            continue;
        }
        for j_bond in mol.get_atom_bonds(c).unwrap_or_default() {
            let j = j_bond.target();
            if j == b || j == i {
                continue;
            }
            let same_side = reference.target_sign
                * if i == reference.reference_b { 1.0 } else { -1.0 }
                * if j == reference.reference_c { 1.0 } else { -1.0 };
            // `1 - cos(phi - phase)` is minimal at `phi == phase`: zero for
            // the cis quadruples, 180 degrees for the trans ones.
            // `V * (1 + s*cos(phi))` is least where `cos(phi) = -s`, so the
            // minimum sits at 0 degrees for `s = -1` and at 180 for `s = +1`.
            let sign = if same_side > 0.0 { -1.0 } else { 1.0 };
            torsions.push(Torsion {
                i,
                b,
                c,
                j,
                terms: single_term(STEREO_PLANARITY_FOLD, sign, V_STEREO_PLANAR),
            });
        }
    }
}

/// Collects a periodic torsion restraint for every `(i, b, c, j)` quadruple
/// around each bond `(b, c)` - `i` ranges over `b`'s other neighbors, `j`
/// over `c`'s other neighbors. Skipped entirely when either bond end is `Sp`
/// (linear, no well-defined torsion angle to restrain around a linear
/// centre), when the bond already has an explicit
/// [`super::chirality::CisTransBond`] restraint (see
/// [`is_explicit_cis_trans_bond`]), or when the bond lies in *any* ring: a
/// ring bond is already geometrically constrained all the way around by
/// bond-length/angle bounds (plus, for small enough all-sp2 rings,
/// `planarity::RingPlane`'s stronger whole-ring shared-plane restraint), and
/// a rotation preference isn't meaningful the way it is for an acyclic bond
/// (you can't freely rotate a ring bond without breaking the ring) - adding
/// one anyway was observed during development to sometimes pull a fused
/// polycyclic aromatic's ring bond length below its bounds' lower limit.
/// Ring membership here comes from `MoleculeCore::find_cyclic_bonds_and_atoms`
/// directly (every bond in every cycle, via DFS back-edges) rather than
/// `RingPlane`'s membership list, since `RingPlane` only covers the
/// *fundamental*-cycle-basis rings up to 8 atoms - for a fused bicyclic,
/// that basis is not guaranteed to align with the two chemical rings (it
/// could just as well return one chemical ring plus the larger
/// whole-perimeter cycle), which would leave some genuine ring bonds
/// unprotected. See the module doc comment for why `phi0` is fixed at 0 for
/// every term (only periodicity is modeled, not a specific rotamer).
/// One experimental rule, ready to match: its SMARTS compiled, and the six
/// `(sign, force constant)` terms it applies.
type CompiledPreference = (CxSMARTSPattern, [(f64, f64); 6], Vec<u128>);

/// Which elements each of a pattern's atoms insists on, as a bit per atomic
/// number, leaving out the atoms that insist on nothing.
///
/// A pattern whose every match needs a sulfur cannot match a molecule
/// without one, and 65 of the 365 acyclic rules name S, P, F or Cl. Asking
/// first costs a bitwise and; not asking costs a subgraph search per rule
/// per molecule, which is most of what a small rigid molecule's embedding
/// spends its time on.
///
/// Read off the compiled pattern, never the SMARTS text: `[!S]` would be
/// filtered exactly backwards by a string scan. An `and` of two element
/// tests is their intersection, an `or` their union, a negation gives up
/// and says nothing, and anything that is not an element test says nothing
/// either -- so the filter only ever skips a pattern that provably could
/// not have matched.
fn required_elements(pattern: &CxSMARTSPattern) -> Vec<u128> {
    pattern
        .atoms
        .iter()
        .filter_map(element_mask)
        .filter(|mask| *mask != 0)
        .collect()
}

/// The atomic numbers this expression allows, or `None` where it does not
/// restrict the element at all.
fn element_mask(expression: &AtomExpression) -> Option<u128> {
    let bit = |number: u8| 1u128 << number.min(127);
    match expression {
        AtomExpression::AtomicNumber(number) => Some(bit(*number)),
        AtomExpression::Element(symbol) => atomic_number_of(symbol).map(bit),
        AtomExpression::Logical { operator, operands } => match operator {
            LogicalOperator::HighAnd | LogicalOperator::LowAnd => operands
                .iter()
                .filter_map(element_mask)
                .reduce(|a, b| a & b),
            LogicalOperator::Or => {
                let masks: Vec<u128> = operands.iter().map(element_mask).collect::<Option<_>>()?;
                masks.into_iter().reduce(|a, b| a | b)
            }
            LogicalOperator::Not => None,
        },
        _ => None,
    }
}

/// The atomic number an element symbol names, aromatic spelling included,
/// by walking the table once per process rather than carrying a second copy
/// of it here.
fn atomic_number_of(symbol: &str) -> Option<u8> {
    static BY_SYMBOL: std::sync::OnceLock<FxHashMap<String, u8>> = std::sync::OnceLock::new();
    BY_SYMBOL
        .get_or_init(|| {
            (1u8..=118)
                .filter_map(|number| {
                    Some((number.atomic_symbol()?.to_ascii_lowercase(), number))
                })
                .collect()
        })
        .get(&symbol.to_ascii_lowercase())
        .copied()
}

/// The elements a molecule actually contains, in the same bit-per-atomic-
/// number form.
fn molecule_elements<T: MoleculeCore>(mol: &T) -> u128 {
    mol.atomic_numbers()
        .iter()
        .fold(0u128, |mask, &number| mask | (1u128 << number.min(127)))
}

/// [`TORSION_PREFERENCES`] with its SMARTS parsed, done once for the
/// process rather than once per bond of every molecule embedded.
///
/// A pattern that fails to parse is dropped rather than panicking: it costs
/// that one rule's preference and falls through to the generic term, which
/// is the same outcome as the rule not existing. `preferences_all_parse`
/// asserts that none currently do.
/// The rules worth trying on this molecule, in priority order.
///
/// Order is priority -- the first pattern that matches a bond is the one
/// that applies -- and the acyclic set comes first, exactly as RDKit
/// concatenates its files, so a ring rule only speaks where no ordinary
/// rule already has.
///
/// A set whose rules cannot match is left out rather than tried: every
/// macrocycle pattern demands `r{9-}` and every small-ring one a ring of
/// eight or fewer, so a molecule with no ring of that size can skip 367 or
/// 105 subgraph searches. Benzene went from 0.7 to 2.7 ms an embedding
/// before this, for rules that could never have matched it.
fn preferences_for(
    ring_sizes: &FxHashSet<usize>,
    small_rings: bool,
    elements: u128,
) -> Vec<&'static CompiledPreference> {
    let mut rules: Vec<&'static CompiledPreference> = compiled(&ACYCLIC, TORSION_PREFERENCES).iter().collect();
    if ring_sizes.iter().any(|&size| size >= MIN_MACROCYCLE_SIZE) {
        rules.extend(compiled(&MACROCYCLE, super::torsion_preferences::MACROCYCLE_PREFERENCES));
    }
    if small_rings && ring_sizes.iter().any(|&size| size < MIN_MACROCYCLE_SIZE) {
        rules.extend(compiled(&SMALL_RING, super::torsion_preferences::SMALL_RING_PREFERENCES));
    }
    // A rule that needs an element this molecule does not have cannot
    // match it, and the subgraph search that would discover that is the
    // expensive part.
    rules.retain(|(_, _, required)| required.iter().all(|mask| mask & elements != 0));
    rules
}

static ACYCLIC: std::sync::OnceLock<Vec<CompiledPreference>> = std::sync::OnceLock::new();
static MACROCYCLE: std::sync::OnceLock<Vec<CompiledPreference>> = std::sync::OnceLock::new();
static SMALL_RING: std::sync::OnceLock<Vec<CompiledPreference>> = std::sync::OnceLock::new();

/// One rule set, compiled on first use. A process that only embeds rigid
/// molecules never parses the ring sets at all.
fn compiled(
    cell: &'static std::sync::OnceLock<Vec<CompiledPreference>>,
    rules: &[(&str, [(f64, f64); 6])],
) -> &'static [CompiledPreference] {
    cell.get_or_init(|| compile(rules))
}

fn compile(rules: &[(&str, [(f64, f64); 6])]) -> Vec<CompiledPreference> {
    rules
        .iter()
        .filter_map(|&(smarts, terms)| {
            let pattern = parse_cxsmarts(smarts).ok()?;
            let elements = required_elements(&pattern);
            Some((pattern, terms, elements))
        })
        .collect()
}

/// A [`MolecularGraph`] view of `mol`, which is what the SMARTS matcher
/// takes. The embedder works on a `Molecule3D` with hydrogens already added,
/// and the two types carry the same topology; only the positions, which
/// SMARTS has no opinion about, are missing here.
fn as_molecular_graph<T: MoleculeCore>(mol: &T) -> MolecularGraph {
    let mut graph = MolecularGraph::default();
    graph.atomic_numbers = mol.atomic_numbers().to_vec();
    graph.atom_classes = mol.atom_classes().clone();
    graph.charges = mol.charges().to_vec();
    graph.chiral_classes = mol.chiral_classes().map(|c| c.to_vec());
    graph.isotopes = mol.isotopes().cloned();
    graph.radical_states = mol.radical_states().to_vec();
    *graph.atom_bonds_mut() = mol.atom_bonds().clone();
    graph
}

/// The bonds an experimental torsion rule could be used on: acyclic, not a
/// bond whose cis/trans configuration was written down, neither end linear,
/// and neither end terminal -- a terminal atom has nothing to complete the
/// quadruple with, so a rule matching there can never be used. Exactly the
/// bonds [`collect_torsions`] can put a rule on.
///
/// Knowing them in advance is what makes it affordable to search for the
/// rules at all. A molecule with none -- benzene, naphthalene, any fused
/// aromatic, half of a typical fragment library -- used to pay for all 365
/// whole-molecule SMARTS searches and then discard every match: 53% of
/// benzene's entire embedding time went on torsion rules it has no bond for.
/// The smallest ring the macrocycle rules speak about, and the size at
/// which RDKit stops calling a ring fused for these purposes.
pub(super) const MIN_MACROCYCLE_SIZE: usize = 9;

/// Ring bonds no torsion rule may touch: every bond of a ring smaller than
/// a macrocycle that shares more than one bond with another ring, and every
/// bond in more than three rings.
///
/// This is RDKit's exclusion, and it is the piece whose absence sank an
/// earlier attempt at ring rules here. A rule describes what one ring does
/// when it is free to do it; in a fused or bridged skeleton the ring cannot
/// move without the rest of the system moving, so a rule asking for a
/// torsion the skeleton cannot adopt is satisfied by stretching bonds
/// instead. Cocaine's tropane -- two rings sharing three bonds -- picked up
/// 110 kcal/mol of bond strain that way. Two macrocycles sharing bonds are
/// left in, as RDKit leaves them: they are floppy enough to mean it.
fn ring_bookkeeping<T: MoleculeCore>(mol: &T) -> Rings {
    let rings = crate::molecule::topology::smallest_rings(mol);
    let ring_sizes: FxHashSet<usize> = rings.iter().map(|ring| ring.len()).collect();
    let ring_bonds: Vec<FxHashSet<(usize, usize)>> = rings
        .iter()
        .map(|ring| {
            (0..ring.len())
                .map(|index| order_tuple(ring[index], ring[(index + 1) % ring.len()]))
                .collect()
        })
        .collect();

    let mut excluded = FxHashSet::default();
    let mut memberships: FxHashMap<(usize, usize), usize> = FxHashMap::default();
    for bonds in &ring_bonds {
        for &bond in bonds {
            *memberships.entry(bond).or_default() += 1;
        }
    }
    for (bond, count) in memberships {
        if count > 3 {
            excluded.insert(bond);
        }
    }
    for (first, one) in ring_bonds.iter().enumerate() {
        for (second, other) in ring_bonds.iter().enumerate().skip(first + 1) {
            if rings[first].len() >= MIN_MACROCYCLE_SIZE
                && rings[second].len() >= MIN_MACROCYCLE_SIZE
            {
                continue;
            }
            if one.intersection(other).count() <= 1 {
                continue;
            }
            for (ring, bonds) in [(first, one), (second, other)] {
                if rings[ring].len() < MIN_MACROCYCLE_SIZE {
                    excluded.extend(bonds.iter().copied());
                }
            }
        }
    }
    let mut in_macrocycle = FxHashSet::default();
    let mut in_small_ring = FxHashSet::default();
    for (ring, bonds) in rings.iter().zip(&ring_bonds) {
        let side = if ring.len() >= MIN_MACROCYCLE_SIZE {
            &mut in_macrocycle
        } else {
            &mut in_small_ring
        };
        side.extend(bonds.iter().copied());
    }
    Rings { excluded, sizes: ring_sizes, in_macrocycle, in_small_ring }
}

/// What the torsion rules need to know about a molecule's rings.
struct Rings {
    /// Bonds of fused or bridged small rings, which no rule may touch.
    excluded: FxHashSet<(usize, usize)>,
    /// Every ring size present, for deciding which rule sets can match.
    sizes: FxHashSet<usize>,
    in_macrocycle: FxHashSet<(usize, usize)>,
    in_small_ring: FxHashSet<(usize, usize)>,
}

impl Rings {
    /// Whether any rule in the active sets could speak about this ring bond.
    /// A ring bond nothing can match is not worth offering to the matcher.
    fn worth_matching(&self, bond: (usize, usize), small_rings: bool) -> bool {
        if self.excluded.contains(&bond) {
            return false;
        }
        self.in_macrocycle.contains(&bond) || (small_rings && self.in_small_ring.contains(&bond))
    }
}

fn rule_candidate_bonds<T: MoleculeCore>(
    mol: &T,
    cyclic_bonds: &FxHashSet<(usize, usize)>,
    rings: &Rings,
    small_rings: bool,
) -> FxHashSet<(usize, usize)> {
    let mut candidates = FxHashSet::default();
    for b in 0..mol.len() {
        for bond in mol.get_atom_bonds(b).unwrap_or_default() {
            let c = bond.target();
            if c <= b {
                continue;
            }
            let pair = order_tuple(b, c);
            if is_explicit_cis_trans_bond(mol, b, c) {
                continue;
            }
            if cyclic_bonds.contains(&pair) && !rings.worth_matching(pair, small_rings) {
                continue;
            }
            if hybridization(mol, b) == Hyb::Sp || hybridization(mol, c) == Hyb::Sp {
                continue;
            }
            if [b, c].iter().any(|&end| mol.get_atom_bonds(end).unwrap_or_default().len() < 2) {
                continue;
            }
            candidates.insert(pair);
        }
    }
    candidates
}

/// The rules' verdict on every bond they have one for, keyed by that bond.
///
/// The rule set is a priority list -- the first pattern that matches a bond
/// is the one that applies -- which reads naturally as "for this bond, try
/// each pattern in turn", and that is how this started. It is also 365
/// whole-molecule SMARTS searches *per bond*, nearly all of whose matches
/// are then discarded for being about some other bond: half of `embed_3d`'s
/// time on a peptide went here. Searching once per pattern and filing each
/// match under its own central bond gives the same answer for 365 searches
/// total, with priority preserved by letting an earlier pattern's entry
/// stand against a later one.
///
/// The pattern's atom maps `:1`..`:4` name the torsion; anything else in the
/// pattern is context that only has to match.
fn experimental_torsions_by_bond(
    graph: &MolecularGraph,
    patterns: &[&CompiledPreference],
    candidates: &FxHashSet<(usize, usize)>,
) -> FxHashMap<(usize, usize), Vec<Torsion>> {
    let mut by_bond: FxHashMap<(usize, usize), Vec<Torsion>> = FxHashMap::default();
    for (pattern, terms, _) in patterns.iter().copied() {
        // Held aside until the pattern is done, so that its own matches can
        // share a bond with each other while still losing any bond an
        // earlier pattern already claimed.
        let mut from_this_pattern: FxHashMap<(usize, usize), Vec<Torsion>> = FxHashMap::default();
        for mapping in find_smarts_matches(pattern, graph) {
            let mut atoms = [usize::MAX; 4];
            for (pattern_atom, &map) in &pattern.atom_classes {
                if (1..=4).contains(&map)
                    && let Some(&molecule_atom) = mapping.get(pattern_atom)
                {
                    atoms[map - 1] = molecule_atom;
                }
            }
            if atoms.contains(&usize::MAX) {
                continue;
            }
            let central = order_tuple(atoms[1], atoms[2]);
            if !candidates.contains(&central) || by_bond.contains_key(&central) {
                continue;
            }
            let mut scaled = *terms;
            for (_, force_constant) in &mut scaled {
                *force_constant *= EXPERIMENTAL_TORSION_SCALE;
            }
            from_this_pattern.entry(central).or_default().push(Torsion {
                i: atoms[0],
                b: atoms[1],
                c: atoms[2],
                j: atoms[3],
                terms: scaled,
            });
        }
        by_bond.extend(from_this_pattern);
    }
    by_bond
}

pub(super) fn collect_torsions<T: MoleculeCore>(mol: &T, small_rings: bool) -> Vec<Torsion> {
    let (_, cyclic_bonds) = mol.find_cyclic_bonds_and_atoms();
    let graph = as_molecular_graph(mol);
    let rings = ring_bookkeeping(mol);
    let candidates = rule_candidate_bonds(mol, &cyclic_bonds, &rings, small_rings);
    // Compiling the rule set is behind the emptiness check, not inside it: a
    // process that only ever embeds rigid molecules should not parse 365
    // SMARTS patterns to discover it had no use for them.
    let mut experimental = if candidates.is_empty() {
        FxHashMap::default()
    } else {
        experimental_torsions_by_bond(
            &graph,
            &preferences_for(&rings.sizes, small_rings, molecule_elements(mol)),
            &candidates,
        )
    };
    let mut torsions = Vec::new();
    for b in 0..mol.len() {
        let Some(bonds_b) = mol.get_atom_bonds(b) else {
            continue;
        };
        for bond_bc in bonds_b {
            let c = bond_bc.target();
            if c <= b {
                continue; // visit each central bond once, from its lower-indexed end
            }
            // A ring bond gets an experimental rule where one matches and
            // the ring is free enough to obey it -- see
            // [`bonds_in_fused_rings`] -- and nothing otherwise. Nothing
            // generic is worth saying about a ring torsion: whether a ring
            // puckers and how is the ring's business, not its bonds'.
            let ring_bond = cyclic_bonds.contains(&order_tuple(b, c));
            if ring_bond && !rings.worth_matching(order_tuple(b, c), small_rings) {
                continue;
            }
            if is_explicit_cis_trans_bond(mol, b, c) {
                collect_stereo_planarity_torsions(mol, b, c, &mut torsions);
                continue;
            }
            let hyb_b = hybridization(mol, b);
            let hyb_c = hybridization(mol, c);
            if hyb_b == Hyb::Sp || hyb_c == Hyb::Sp {
                continue;
            }
            // An experimental preference, where the rule set has one for
            // this bond, replaces the generic term rather than joining it:
            // the generic term is a guess about what the bond's
            // hybridization implies, and the rule is a measurement of what
            // bonds like it actually do.
            if let Some(preferred) = experimental.remove(&order_tuple(b, c)) {
                torsions.extend(preferred);
                continue;
            }
            if ring_bond {
                continue;
            }

            // `torsion_energy` is `V (1 + sign cos(n phi))`, so the sign
            // is what decides where the minima sit, and it is not the same
            // for the three cases.
            //
            // A conjugated bond wants to be planar: two-fold with a
            // negative sign is zero at 0 and 180 degrees and costs `2V` at
            // 90, which is right. A bond between two sp3 centres wants to
            // be *staggered*, and that is three-fold with a **positive**
            // sign -- zero at 60, 180 and 300, `2V` at the eclipsed 0, 120
            // and 240. A negative sign would put the minima exactly on the
            // eclipsed positions and make the term a preference for the
            // arrangement it is supposed to forbid: 42%
            // of the rotatable torsions in a distance-geometry structure
            // came out eclipsed against RDKit's 18%, and raising the
            // barrier made it worse rather than better -- at ten times the
            // weight, 52%.
            //
            // Small alkanes hid it. Butane's terminal carbons are a 1-4
            // pair with a van-der-Waals floor of 2.7 A, and the eclipsed
            // arrangement puts them nearer than that, so the bounds
            // pushed butane anti whatever this term asked for. Between two
            // hydrogens the floor is 1.9 A and eclipsed is 2.3, so nothing
            // overruled it there.
            let (fold, sign, barrier) = match (hyb_b, hyb_c) {
                (Hyb::Sp2, Hyb::Sp2) => (2, -1.0, V_CONJUGATED),
                (Hyb::Sp3, Hyb::Sp3) => (3, 1.0, V_STAGGERED),
                _ => (6, -1.0, V_NEARFREE),
            };
            let Some(bonds_c) = mol.get_atom_bonds(c) else {
                continue;
            };
            for i_bond in bonds_b {
                let i = i_bond.target();
                if i == c {
                    continue;
                }
                for j_bond in bonds_c {
                    let j = j_bond.target();
                    if j == b || j == i {
                        continue;
                    }
                    torsions.push(Torsion {
                        i,
                        b,
                        c,
                        j,
                        terms: single_term(fold, sign, barrier),
                    });
                }
            }
        }
    }
    torsions
}

fn angle_bend_energy(positions: &[Vector3D], a: &AngleBend) -> f64 {
    let u = positions[a.i] - positions[a.k];
    let v = positions[a.j] - positions[a.k];
    let (ru, rv) = (u.length(), v.length());
    if ru < 1e-8 || rv < 1e-8 {
        return 0.0;
    }
    let delta = (u.dot(&v) / (ru * rv)).clamp(-1.0, 1.0) - a.cos_theta0;
    0.5 * K_ANGLE * delta * delta
}

fn angle_bend_energy_and_grad(positions: &[Vector3D], a: &AngleBend, grad: &mut [Vector3D]) -> f64 {
    let u = positions[a.i] - positions[a.k];
    let v = positions[a.j] - positions[a.k];
    let ru = u.length();
    let rv = v.length();
    if ru < 1e-8 || rv < 1e-8 {
        return 0.0; // degenerate (coincident points): no well-defined angle
    }

    let cos_theta = (u.dot(&v) / (ru * rv)).clamp(-1.0, 1.0);
    let delta = cos_theta - a.cos_theta0;
    let energy = 0.5 * K_ANGLE * delta * delta;

    // d(cos theta)/du = v/(|u||v|) - cos(theta)*u/|u|^2, symmetric for dv;
    // p_k appears (with a factor of -1) in both u=p_i-p_k and v=p_j-p_k, so
    // its gradient is the negated sum of the other two.
    let dcos_du = v * (1.0 / (ru * rv)) - u * (cos_theta / (ru * ru));
    let dcos_dv = u * (1.0 / (ru * rv)) - v * (cos_theta / (rv * rv));

    let coeff = K_ANGLE * delta;
    let grad_i = dcos_du * coeff;
    let grad_j = dcos_dv * coeff;
    let grad_k = -(grad_i + grad_j);

    grad[a.i] += grad_i;
    grad[a.j] += grad_j;
    grad[a.k] += grad_k;

    energy
}

/// The `i-b-c-j` dihedral angle in `(-pi, pi]`. `None` if any of the three
/// bonds is degenerate or the `i-b-c` / `b-c-j` angle is near linear -- both
/// cases have no meaningful torsion to restrain.
///
/// The same primitive the gradient path uses, so the two cannot drift apart
/// on the sign convention. They once did: the force field kept its own copy
/// that projected `b - i` where this projects `i - b`, which is 180 degrees
/// out, and UFF's ethane duly minimised eclipsed.
#[cfg(test)]
fn dihedral_angle(positions: &[Vector3D], i: usize, b: usize, c: usize, j: usize) -> Option<f64> {
    crate::vector::dihedral_cosine_sine(positions[i], positions[b], positions[c], positions[j])
        .map(|(cosine, sine)| sine.atan2(cosine))
}

/// One torsion restraint's energy and `dE/d(phi)`, from the dihedral's
/// cosine and sine.
///
/// The preferences run to six folds. Instead of `atan2` followed by six `cos`
/// and six `sin` calls, walking the multiple-angle recurrence gives every
/// fold's cosine and sine in two multiplications each.
fn torsion_terms_from(cosine: f64, sine: f64, t: &Torsion) -> (f64, f64) {
    let (mut energy, mut derivative) = (0.0, 0.0);
    let (mut previous, mut current) = ((1.0, 0.0), (cosine, sine));
    for (index, &(sign, force_constant)) in t.terms.iter().enumerate() {
        let fold = (index + 1) as f64;
        energy += force_constant * (1.0 + sign * current.0);
        derivative += -force_constant * sign * fold * current.1;
        let next = (
            2.0 * cosine * current.0 - previous.0,
            2.0 * cosine * current.1 - previous.1,
        );
        previous = current;
        current = next;
    }
    (energy, derivative)
}

/// The same energy as [`torsion_terms_from`], written from the angle. Only
/// the tests use it: they scan phi over a range, and reading the preference
/// curve straight off the definition is what makes them a check on the
/// recurrence rather than a restatement of it.
#[cfg(test)]
fn torsion_energy(phi: f64, t: &Torsion) -> f64 {
    t.terms
        .iter()
        .enumerate()
        .map(|(index, &(sign, force_constant))| {
            force_constant * (1.0 + sign * ((index + 1) as f64 * phi).cos())
        })
        .sum()
}

#[cfg(test)]
fn nudge(v: Vector3D, axis: usize, delta: f64) -> Vector3D {
    match axis {
        0 => Vector3D::new(v.x + delta, v.y, v.z),
        1 => Vector3D::new(v.x, v.y + delta, v.z),
        _ => Vector3D::new(v.x, v.y, v.z + delta),
    }
}

/// The dihedral `i-b-c-j` and `d(phi)/dp` for its four atoms.
fn dihedral_gradient(
    positions: &[Vector3D],
    i: usize,
    b: usize,
    c: usize,
    j: usize,
) -> Option<(f64, f64, [Vector3D; 4])> {
    crate::vector::dihedral_and_gradient(positions[i], positions[b], positions[c], positions[j])
}

/// Energy plus its analytic gradient (in `[grad_i, grad_b, grad_c, grad_j]`
/// order) for one torsion.
fn torsion_energy_and_grad(positions: &[Vector3D], t: &Torsion) -> Option<(f64, [Vector3D; 4])> {
    let (cosine, sine, d_phi) = dihedral_gradient(positions, t.i, t.b, t.c, t.j)?;
    let (energy, d_energy) = torsion_terms_from(cosine, sine, t);
    Some((energy, d_phi.map(|d| d * d_energy)))
}

/// Adds the angle-bend and torsion energy (and gradient) into `grad` for
/// every collected restraint, returning the summed energy - the combined
/// entry point [`super::refine::refine`] threads through alongside
/// `stereo`/`planarity`.
/// Just the energy, for the line search, which evaluates it a couple of
/// times per iteration and never looks at the gradient. Worth its own path
/// because a torsion's gradient costs several times its energy even now
/// that it is analytic.
pub(super) fn forcefield_energy(
    positions: &[Vector3D],
    angles: &[AngleBend],
    torsions: &[Torsion],
) -> f64 {
    let mut energy = 0.0;
    for a in angles {
        energy += angle_bend_energy(positions, a);
    }
    for t in torsions {
        if let Some((cosine, sine)) =
            crate::vector::dihedral_cosine_sine(positions[t.i], positions[t.b], positions[t.c], positions[t.j])
        {
            energy += torsion_terms_from(cosine, sine, t).0;
        }
    }
    energy
}

pub(super) fn forcefield_energy_and_gradient(
    positions: &[Vector3D],
    angles: &[AngleBend],
    torsions: &[Torsion],
    grad: &mut [Vector3D],
) -> f64 {
    let mut energy = 0.0;
    for a in angles {
        energy += angle_bend_energy_and_grad(positions, a, grad);
    }
    for t in torsions {
        if let Some((e, atom_grad)) = torsion_energy_and_grad(positions, t) {
            energy += e;
            let atoms = [t.i, t.b, t.c, t.j];
            for (slot, &atom) in atoms.iter().enumerate() {
                grad[atom] += atom_grad[slot];
            }
        }
    }
    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ethane's generic term has to prefer staggered over eclipsed.
    ///
    /// It did not: `single_term(3, -1.0, V_STAGGERED)` puts the minima of
    /// `V (1 + sign cos 3 phi)` on the eclipsed positions, so the term named
    /// after the arrangement it wants was asking for the other one. Nothing
    /// caught it, because butane's 1-4 van-der-Waals floor pushes its
    /// terminal carbons apart whatever this term says, and butane's chain
    /// torsion is 180 degrees either way. Between two hydrogens the floor is
    /// slack enough that the term decides, and 42% of the rotatable torsions
    /// in a distance-geometry structure came out eclipsed.
    #[test]
    fn a_bond_between_sp3_centres_prefers_staggered() {
        let mut ethane = crate::molecule::molecular_graph::MolecularGraph::from_smiles_single("CC")
            .expect("ethane parses");
        ethane.add_hydrogens();
        let torsions = collect_torsions(&ethane, false);
        assert!(!torsions.is_empty(), "ethane has torsion terms");

        let at = |phi: f64| -> f64 {
            torsions
                .iter()
                .map(|term| torsion_energy(phi.to_radians(), term))
                .sum()
        };
        let (eclipsed, staggered) = (at(0.0), at(60.0));
        assert!(
            staggered < eclipsed,
            "staggered {staggered} should cost less than eclipsed {eclipsed}"
        );
        // And the barrier should be the whole of the term, not a sliver of it.
        assert!(
            eclipsed - staggered > 0.1 * eclipsed.abs().max(1.0),
            "the preference is there but negligible: {eclipsed} against {staggered}"
        );
    }

    /// The element filter may only ever skip a rule that could not have
    /// matched, so every rule it keeps out has to be one whose own atoms
    /// ask for something the molecule does not have.
    #[test]
    fn the_element_filter_only_drops_rules_that_could_not_match() {
        let carbon_and_nitrogen = (1u128 << 6) | (1u128 << 7) | (1u128 << 1);
        let sizes = FxHashSet::default();
        let everything = u128::MAX;
        let all = preferences_for(&sizes, false, everything);
        let kept = preferences_for(&sizes, false, carbon_and_nitrogen);
        assert!(kept.len() < all.len(), "no rule was filtered at all");
        for (pattern, _, required) in &all {
            let dropped = !kept
                .iter()
                .any(|(other, _, _)| std::ptr::eq(other, pattern));
            if dropped {
                assert!(
                    required
                        .iter()
                        .any(|mask| mask & carbon_and_nitrogen == 0),
                    "a rule was dropped without needing a missing element"
                );
            }
        }
    }

    /// An element inside a negation constrains nothing, and reading it as a
    /// requirement would filter exactly backwards.
    #[test]
    fn a_negated_element_is_not_a_requirement() {
        let pattern = parse_cxsmarts("[!S]~[C]").expect("pattern parses");
        let required = required_elements(&pattern);
        assert_eq!(required.len(), 1, "only the carbon should be a requirement");
        assert_eq!(required[0], 1u128 << 6);
    }

    #[test]
    fn angle_bend_gradient_descent_relaxes_to_target() {
        // Distorted ~100 deg angle, target 109.5 deg (tetrahedral).
        let angle = AngleBend { i: 1, k: 0, j: 2, cos_theta0: 109.5_f64.to_radians().cos() };
        let mut positions = vec![
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(-0.17, 0.98, 0.0), // ~100 deg from the first bond
        ];

        for _ in 0..500 {
            let mut grad = vec![Vector3D::default(); positions.len()];
            angle_bend_energy_and_grad(&positions, &angle, &mut grad);
            for i in 0..positions.len() {
                positions[i] += -grad[i] * (1.0 / K_ANGLE);
            }
        }

        let u = positions[1] - positions[0];
        let v = positions[2] - positions[0];
        let cos_theta = u.dot(&v) / (u.length() * v.length());
        assert!(
            (cos_theta - 109.5_f64.to_radians().cos()).abs() < 1e-3,
            "expected angle to relax to 109.5 deg, cos(theta)={cos_theta}"
        );
    }

    #[test]
    fn angle_bend_gradient_matches_finite_difference() {
        let angle = AngleBend { i: 0, k: 1, j: 2, cos_theta0: 100.0_f64.to_radians().cos() };
        let positions = vec![
            Vector3D::new(1.3, -0.2, 0.4),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(-0.5, 1.1, 0.3),
        ];

        let mut analytic = vec![Vector3D::default(); positions.len()];
        angle_bend_energy_and_grad(&positions, &angle, &mut analytic);

        let h = 1e-6;
        for atom in 0..positions.len() {
            for axis in 0..3 {
                let mut plus = positions.clone();
                plus[atom] = nudge(positions[atom], axis, h);
                let mut minus = positions.clone();
                minus[atom] = nudge(positions[atom], axis, -h);

                let mut scratch = vec![Vector3D::default(); positions.len()];
                let e_plus = angle_bend_energy_and_grad(&plus, &angle, &mut scratch);
                let mut scratch = vec![Vector3D::default(); positions.len()];
                let e_minus = angle_bend_energy_and_grad(&minus, &angle, &mut scratch);

                let fd = (e_plus - e_minus) / (2.0 * h);
                let analytic_component = match axis {
                    0 => analytic[atom].x,
                    1 => analytic[atom].y,
                    _ => analytic[atom].z,
                };
                assert!(
                    (fd - analytic_component).abs() < 1e-4,
                    "atom {atom} axis {axis}: analytic={analytic_component}, fd={fd}"
                );
            }
        }
    }

    #[test]
    fn dihedral_angle_matches_known_configurations() {
        // i-b-c-j forming a known 90 deg dihedral: i,b,c in the xy-plane,
        // j displaced purely along z from c relative to the b-c direction.
        let i = Vector3D::new(1.0, 0.0, 0.0);
        let b = Vector3D::new(0.0, 0.0, 0.0);
        let c = Vector3D::new(0.0, 1.0, 0.0);
        let j_cis = Vector3D::new(1.0, 1.0, 0.0); // eclipsed with i -> phi = 0
        let j_trans = Vector3D::new(-1.0, 1.0, 0.0); // phi = 180 deg
        let j_perp = Vector3D::new(0.0, 1.0, 1.0); // phi = +/-90 deg

        let positions = |j: Vector3D| vec![i, b, c, j];

        let phi_cis = dihedral_angle(&positions(j_cis), 0, 1, 2, 3).unwrap();
        assert!(phi_cis.abs() < 1e-6, "expected ~0 deg, got {} deg", phi_cis.to_degrees());

        let phi_trans = dihedral_angle(&positions(j_trans), 0, 1, 2, 3).unwrap();
        assert!(
            (phi_trans.abs() - std::f64::consts::PI).abs() < 1e-6,
            "expected ~180 deg, got {} deg",
            phi_trans.to_degrees()
        );

        let phi_perp = dihedral_angle(&positions(j_perp), 0, 1, 2, 3).unwrap();
        assert!(
            (phi_perp.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
            "expected ~90 deg, got {} deg",
            phi_perp.to_degrees()
        );
    }

    #[test]
    fn torsion_gradient_descent_reaches_a_minimum() {
        // `phi0` is fixed at 0 for every torsion term (see the module doc
        // comment) - the potential's minima are 3 evenly spaced angles, not
        // one specific "staggered" rotamer, so the meaningful invariant to
        // test is "energy reaches ~0 (its global minimum) from an arbitrary
        // non-minimum start", not "converges to a particular angle".
        let t = Torsion { i: 0, b: 1, c: 2, j: 3, terms: single_term(3, -1.0, V_STAGGERED) };
        let mut positions = vec![
            Vector3D::new(1.0, 0.0, 0.0),
            Vector3D::new(0.0, 0.0, 0.0),
            Vector3D::new(0.0, 1.0, 0.0),
            Vector3D::new(0.5, 1.0, 0.5), // arbitrary, not at a minimum or maximum
        ];

        let phi_start = dihedral_angle(&positions, t.i, t.b, t.c, t.j).unwrap();
        let energy_start = torsion_energy(phi_start, &t);
        assert!(energy_start > 1e-3, "fixture should start away from a minimum");

        for _ in 0..2000 {
            let mut grad = vec![Vector3D::default(); positions.len()];
            if let Some((_, atom_grad)) = torsion_energy_and_grad(&positions, &t) {
                let atoms = [t.i, t.b, t.c, t.j];
                for (slot, &atom) in atoms.iter().enumerate() {
                    grad[atom] += atom_grad[slot];
                }
            }
            for k in 0..positions.len() {
                positions[k] += -grad[k] * 0.05;
            }
        }

        let phi_end = dihedral_angle(&positions, t.i, t.b, t.c, t.j).unwrap();
        let energy_end = torsion_energy(phi_end, &t);
        assert!(
            energy_end < 1e-6,
            "expected energy to reach ~0 (a minimum), got {energy_end} (phi={} deg)",
            phi_end.to_degrees()
        );
    }
    /// The analytic four-body dihedral derivative against a central
    /// difference of [`dihedral_angle`] -- which is exactly what
    /// [`torsion_energy_and_grad`] used to compute directly, so this is the
    /// implementation it replaced acting as its oracle.
    ///
    /// Geometries are scrambled and asymmetric: a sign error in
    /// one component of one atom's derivative survives any arrangement with
    /// a symmetry that could cancel it.
    #[test]
    fn torsion_gradient_matches_finite_difference() {
        let geometries = [
            [
                Vector3D::new(1.21, -0.34, 0.55),
                Vector3D::new(0.0, 0.0, 0.0),
                Vector3D::new(1.47, 0.12, -0.31),
                Vector3D::new(2.03, 1.31, 0.44),
            ],
            [
                Vector3D::new(-0.83, 1.02, -0.61),
                Vector3D::new(0.11, 0.07, 0.23),
                Vector3D::new(1.32, -0.44, -0.18),
                Vector3D::new(0.97, -1.55, 0.92),
            ],
            // Near-planar, where the two plane normals nearly line up.
            [
                Vector3D::new(-1.0, 0.05, 0.0),
                Vector3D::new(0.0, 0.0, 0.0),
                Vector3D::new(1.33, 0.0, 0.0),
                Vector3D::new(2.1, 1.02, 0.02),
            ],
        ];

        for (folds, sign) in [(1, 1.0), (1, -1.0), (2, 1.0), (3, -1.0)] {
            let torsion =
                Torsion { i: 0, b: 1, c: 2, j: 3, terms: single_term(folds, sign, 0.7) };
            for positions in &geometries {
                let positions = positions.to_vec();
                let (_, analytic) =
                    torsion_energy_and_grad(&positions, &torsion).expect("well-defined torsion");

                for slot in 0..4 {
                    let original = positions[slot];
                    let mut numeric = Vector3D::default();
                    for axis in 0..3 {
                        let mut perturbed = positions.clone();
                        perturbed[slot] = nudge(original, axis, FD_STEP);
                        let plus = dihedral_angle(&perturbed, 0, 1, 2, 3)
                            .map(|p| torsion_energy(p, &torsion))
                            .expect("well-defined torsion");
                        perturbed[slot] = nudge(original, axis, -FD_STEP);
                        let minus = dihedral_angle(&perturbed, 0, 1, 2, 3)
                            .map(|p| torsion_energy(p, &torsion))
                            .expect("well-defined torsion");
                        let d = (plus - minus) / (2.0 * FD_STEP);
                        match axis {
                            0 => numeric.x = d,
                            1 => numeric.y = d,
                            _ => numeric.z = d,
                        }
                    }
                    let difference = (analytic[slot] - numeric).length();
                    assert!(
                        difference < 1e-5,
                        "atom {slot} of a {folds}-fold torsion (sign {sign}): \
                         analytic {:?} vs finite difference {:?}",
                        analytic[slot],
                        numeric
                    );
                }
            }
        }
    }


}

#[cfg(test)]
mod ring_rule_tests {
    use super::*;

    fn torsion_bonds(smiles: &str, small_rings: bool) -> Vec<(usize, usize)> {
        let mut molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        molecule.add_hydrogens();
        let mut bonds: Vec<(usize, usize)> = collect_torsions(&molecule, small_rings)
            .iter()
            .map(|torsion| order_tuple(torsion.b, torsion.c))
            .collect();
        bonds.sort_unstable();
        bonds.dedup();
        bonds
    }

    /// The macrocycle rules reach a macrocycle's own ring bonds.
    ///
    /// They are written entirely in `r{9-}`, so they match only if ring
    /// perception goes past eight atoms.
    #[test]
    fn a_macrocycle_gets_torsion_terms_on_its_ring_bonds() {
        let molecule = {
            let mut molecule =
                MolecularGraph::from_smiles_single("CC1CCC(=O)OC(C)CC(O)C(C)C(=O)C(C)CC1O")
                    .unwrap();
            molecule.add_hydrogens();
            molecule
        };
        let (_, cyclic) = molecule.find_cyclic_bonds_and_atoms();
        let ring_terms = torsion_bonds("CC1CCC(=O)OC(C)CC(O)C(C)C(=O)C(C)CC1O", false)
            .into_iter()
            .filter(|bond| cyclic.contains(bond))
            .count();
        assert!(ring_terms > 8, "only {ring_terms} of the ring's bonds got a preference");
    }

    /// A bridged bicyclic gets none, however many rules match it.
    ///
    /// Cocaine's tropane shares three bonds between its two rings, so
    /// neither ring can adopt what a rule written for a free ring asks for,
    /// and a term there is satisfied by stretching bonds instead: 110
    /// kcal/mol of excess MMFF bond energy, the first time these rules were
    /// tried here.
    #[test]
    fn a_bridged_ring_system_gets_no_ring_torsion_terms() {
        let smiles = "CN1C2CCC1C(C(=O)OC)C(OC(=O)c1ccccc1)C2";
        let mut molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        molecule.add_hydrogens();
        let (_, cyclic) = molecule.find_cyclic_bonds_and_atoms();
        for small_rings in [false, true] {
            let offenders: Vec<(usize, usize)> = torsion_bonds(smiles, small_rings)
                .into_iter()
                .filter(|bond| cyclic.contains(bond))
                .collect();
            assert!(
                offenders.is_empty(),
                "small_rings={small_rings}: tropane bonds got terms: {offenders:?}"
            );
        }
    }

    /// The small-ring rules are off unless asked for, and reach a plain
    /// six-ring when they are.
    #[test]
    fn small_ring_rules_are_opt_in() {
        let (_, cyclic) = {
            let mut molecule = MolecularGraph::from_smiles_single("C1COCCN1").unwrap();
            molecule.add_hydrogens();
            molecule.find_cyclic_bonds_and_atoms()
        };
        let ring_terms = |small_rings| {
            torsion_bonds("C1COCCN1", small_rings)
                .into_iter()
                .filter(|bond| cyclic.contains(bond))
                .count()
        };
        assert_eq!(ring_terms(false), 0);
        assert!(ring_terms(true) > 0, "morpholine got no small-ring terms");
    }
}
