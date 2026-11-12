//! Distance-bounds matrix construction and triangle-inequality smoothing.
//!
//! Follows the Crippen/Havel convention: for a pair `(i, j)` with `i < j`,
//! the upper bound is stored at `[i, j]` and the lower bound at `[j, i]`.

/// Covalent-radius multiplier used when the van der Waals table has no row.
const VDW_COVALENT_RADIUS_SCALE: f64 = 1.6;
/// Radius fallback in Å when neither reference table covers an element.
const DEFAULT_VDW_RADIUS: f64 = 1.7;
/// Bond-length fallback covalent radius in Å for an unsupported element.
const DEFAULT_COVALENT_RADIUS: f64 = 0.77;
/// Upper-bound multiplier when repairing an inverted interval after smoothing.
const BOUNDS_REPAIR_EXPANSION: f64 = 1.02;
/// Maximum fraction of atom-pair bounds that may require repair.
const MAX_BOUNDS_REPAIR_FRACTION: f64 = 0.01;

/// Empirical scaling of summed single-bond covalent radii by bond order.
/// These embedding estimates are calibrated against typical C-C distances;
/// they are not reference radii or force-field equilibrium lengths.
const SINGLE_BOND_RADIUS_SCALE: f64 = 1.01;
const AROMATIC_BOND_RADIUS_SCALE: f64 = 0.915;
const DOUBLE_BOND_RADIUS_SCALE: f64 = 0.87;
const TRIPLE_BOND_RADIUS_SCALE: f64 = 0.78;
const QUADRUPLE_BOND_RADIUS_SCALE: f64 = 0.70;

use super::EmbedError;
use crate::molecule::forcefield::lanes::{self, LANES, Lanes};
use crate::molecule::minimize::Coordinate;
use crate::molecule::{base::MoleculeCore, bond::BondOrder, chirality::ChiralClass};
use bit_set::BitSet;
use chemistry_consts::ElementProperties;
use super::matrix::SquareMatrix;
use rustc_hash::FxHashSet;

/// A generous finite placeholder for pairs not otherwise constrained, before
/// triangle-inequality smoothing tightens it. Must stay finite so smoothing
/// arithmetic never produces `inf`.
const DEFAULT_UPPER: f64 = 100.0;
/// How far a bond may sit from its nominal length, in angstrom.
///
/// One number for every bond, which is not what the measurements say: a
/// relaxed bond sits off `r0` by the other terms' force over the stretch's
/// own constant, so its spread goes as `1/k`, and it does -- against
/// MMFF-relaxed conformers a C=O at `k` = 10 holds to 0.006 A where a C-C
/// single at 4.3 wanders 0.0138, the same factor as their force constants.
///
/// Widening it where the spread is *measured* to be large is the better
/// version of that idea and is also rejected, 2026-10-09. The spread grows
/// with crowding exactly as [`crowding_correction`]'s centre shift does
/// (0.004 A at three heavy neighbours, 0.079 at eight), so the window was
/// made `BOND_TOLERANCE + share * crowding_correction` and the share swept.
/// The probe liked it monotonically -- at a share of one, curated
/// violations fell from 123 to 103, large natural products' 1-2 rate from
/// 24.9% to 18.0%, and an unseen corpus's from 19.8% to 14.3% -- and the
/// conformers did not:
///
/// | share | held out 400 | PubChem 287 |
/// |---|---|---|
/// | 0 | 365, -1.78 | 270, -2.24 |
/// | 0.5 | 366, -1.75 | 268, -2.21 |
/// | 1.0 | 358, -1.70 | 268, -2.27 |
///
/// Nine molecules for nothing at a share of one, net one for nothing at a
/// half. The asymmetry comes from the probe: it rewards a bound for
/// admitting the truth, but a window admits *everything* between its ends,
/// and the freedom costs more in bad starts than the missing truth costs in
/// bias. The centre was worth correcting and the width is not.
///
/// Scaling the window by MMFF's own `k` instead, normalised so a C-C single
/// keeps this value, is worse still: 1-2 violations went from 12.9% to
/// 21.6% and the whole matrix from 2.2% to 3.4%. Two
/// reasons, both visible in the same data. Most common bonds -- aromatic,
/// C=O, C-N -- are *stiffer* than a C-C single, so the model tightens
/// almost everything. And what is left over after
/// [`MMFF_MINIMUM_OFFSET`] is a per-bond-type bias of up to 0.015 A, which
/// is larger than the spread the stiffness predicts: the window is being
/// scaled when the centre is what still moves. A per-type centre would be
/// a table fitted on one corpus, which is the thing this week has been
/// careful not to build.
const BOND_TOLERANCE: f64 = 0.02;
const VDW_LOWER_SCALE: f64 = 0.8;

/// Bondi/Alvarez-style van der Waals radii (Å), keyed by atomic number.
///
/// Hydrogen is Bondi's 1.20, not the 1.09-1.10 of Rowland & Taylor's
/// crystal-contact revision. Both are defensible and the choice is worth
/// stating, because it is the single value that decides how closely this
/// embedder lets atoms approach: scaled by [`VDW_LOWER_SCALE`], it puts the
/// floor under an H-H contact at 1.92 A rather than 1.76 A.
///
/// Measured against RDKit ETKDGv3 (`tests/embed_benchmark.py`), 1.10 shipped
/// structures whose closest non-bonded approach was 1.75-1.76 A on every
/// large flexible molecule tried, against RDKit's 1.92-1.99; at 1.20 that
/// becomes 1.90-1.92, and MMFF-minimised energies improve as well
/// (atorvastatin 49.8 to 39.2, aspirin 24.5 to 18.9). The cost is that
/// tighter sterics push torsions into worse ranges -- which generic
/// torsion terms cannot resist the way experimental torsion preferences
/// would -- so raw energies rise unless more trials are searched. They
/// recover fully by 20 (atorvastatin 422.6 to 366.7, aspirin 79.9 to 59.6),
/// which is what `EmbedConfig::max_attempts` is now for.
use chemistry_consts::BONDI_VDW_RADII_ANGSTROM as VDW_RADII;

pub(super) fn vdw_radius(atomic_number: u8) -> f64 {
    if let Some(&(_, radius)) = VDW_RADII.iter().find(|&&(z, _)| z == atomic_number) {
        return radius;
    }
    atomic_number
        .covalent_radius()
        .map(|r| r * VDW_COVALENT_RADIUS_SCALE)
        .unwrap_or(DEFAULT_VDW_RADIUS)
}

/// Multiplicative bond-order correction applied to the sum of single-bond
/// covalent radii (Pyykkoe-style). Calibrated against typical C-C bond
/// lengths (single 1.54 A, aromatic 1.39 A, double 1.33 A, triple 1.20 A)
/// using this crate's covalent-radius table (C = 0.76 A); a known
/// simplification since real increments are element-pair-specific.
fn bond_order_factor(order: BondOrder) -> f64 {
    match order {
        BondOrder::Single | BondOrder::Up | BondOrder::Down | BondOrder::Coordinate => {
            SINGLE_BOND_RADIUS_SCALE
        }
        BondOrder::Aromatic => AROMATIC_BOND_RADIUS_SCALE,
        BondOrder::Double => DOUBLE_BOND_RADIUS_SCALE,
        BondOrder::Triple => TRIPLE_BOND_RADIUS_SCALE,
        BondOrder::Quadruple => QUADRUPLE_BOND_RADIUS_SCALE,
    }
}

/// Returns `(min, max)` bond angle in radians expected at `atom`, based on a
/// simple hybridization heuristic derived from its neighbor bond orders.
/// No lone-pair model exists in this crate, so atoms with fewer than 4
/// explicit substituents get a wider tolerance.
pub(super) fn estimate_hybridization_angle<T: MoleculeCore>(mol: &T, atom: usize) -> (f64, f64) {
    let Some(bonds) = mol.get_atom_bonds(atom) else {
        return angle_bounds_deg(109.5, 15.0);
    };
    let degree = bonds.len();
    let double_count = bonds
        .iter()
        .filter(|b| b.bond_order == BondOrder::Double)
        .count();
    let has_triple = bonds.iter().any(|b| b.bond_order == BondOrder::Triple);
    let has_aromatic_or_double = bonds
        .iter()
        .any(|b| matches!(b.bond_order, BondOrder::Aromatic | BondOrder::Double));

    // Two double bonds mean an allene-type sp centre only on an atom that
    // has nothing else attached. On a *hypervalent* centre they mean the
    // opposite: a sulfone, a sulfonamide, a sulfate or a phosphate carries
    // two of them and is tetrahedral. Without the degree test every one of
    // those got a 170-180 degree window, which put the floor under their
    // 1-3 distances at the sum of the two bonds -- 3.43 A across a
    // sulfonamide's S where the atoms sit at 2.68, the worst bound in the
    // whole matrix, and the same for every sulfonyl and phosphoryl group
    // in the corpus.
    if has_triple || (double_count >= 2 && degree <= 2) {
        angle_bounds_deg(180.0, 10.0) // sp: triple bond, or cumulated double bonds
    } else if degree == 4 {
        // Four substituents fix the shape whatever the bond orders say, and
        // only a hypervalent centre has both. Its angles spread wider than
        // a carbon's -- a sulfone's O-S-O opens to 119 while its C-S-C
        // closes to 103 -- so the window does too.
        if has_aromatic_or_double {
            angle_bounds_deg(109.5, 12.0)
        } else {
            angle_bounds_deg(109.5, 5.0) // sp3, fully substituted
        }
    } else if has_aromatic_or_double {
        angle_bounds_deg(120.0, 10.0) // sp2
    } else {
        angle_bounds_deg(109.5, 15.0) // sp3-ish default, looser (no lone-pair model)
    }
}

fn angle_bounds_deg(center_deg: f64, tol_deg: f64) -> (f64, f64) {
    let min_deg = (center_deg - tol_deg).max(0.0);
    let max_deg = (center_deg + tol_deg).min(180.0);
    (min_deg.to_radians(), max_deg.to_radians())
}

/// Interior angle (radians) of a regular polygon with `ring_size` vertices.
fn ring_interior_angle(ring_size: usize) -> f64 {
    let n = ring_size as f64;
    ((n - 2.0) * 180.0 / n).to_radians()
}

/// The two trans (180 deg) pairs among a square-planar centre's index-sorted
/// neighbors `n`, per the OpenSMILES `SP1`(U)/`SP2`(4)/`SP3`(Z) shape codes.
/// Calibrated empirically against RDKit's `AssignStereochemistryFrom3D`
/// (`tests/embed_oracle.py`): a simple hand-built square arrangement with
/// substituents placed at consecutive corners in written order reads back as
/// `SP1`; swapping the 2nd/3rd corners gives `SP2`; swapping the 1st/4th
/// corners gives `SP3` - i.e. `SP1` pairs neighbors (0,2) and (1,3) trans,
/// `SP2` pairs (0,1) and (2,3), `SP3` pairs (0,3) and (1,2). Any pair not
/// named here is one of the four *adjacent* (90 deg) pairs.
fn square_planar_trans_pairs(sp_number: u8, n: [usize; 4]) -> Option<[(usize, usize); 2]> {
    match sp_number {
        1 => Some([(n[0], n[2]), (n[1], n[3])]),
        2 => Some([(n[0], n[1]), (n[2], n[3])]),
        3 => Some([(n[0], n[3]), (n[1], n[2])]),
        _ => None,
    }
}

/// `Some((min, max))` 1-3 angle bounds for `(i, k, j)` if `k` is tagged as a
/// square-planar (`SP`) centre: 180 deg for the tag's two trans pairs, 90 deg
/// for every other (adjacent) pair. Unlike tetrahedral/allene chirality,
/// square-planar geometry has no separate handedness once the trans-pairing
/// is fixed - a plain 4-point distance arrangement (via these angle bounds)
/// fully determines the shape, so no additional stereo-consistency energy
/// term (unlike [`super::chirality`]'s `TH`/cis-trans terms) is needed.
fn square_planar_angle<T: MoleculeCore>(
    mol: &T,
    k: usize,
    i: usize,
    j: usize,
) -> Option<(f64, f64)> {
    let ChiralClass::SP(sp_number) = *mol.chiral_classes()?.get(k)? else {
        return None;
    };
    let mut neighbors: Vec<usize> = mol.get_atom_bonds(k)?.iter().map(|b| b.target()).collect();
    if neighbors.len() != 4 {
        return None;
    }
    neighbors.sort_unstable();
    let n = [neighbors[0], neighbors[1], neighbors[2], neighbors[3]];
    let trans_pairs = square_planar_trans_pairs(sp_number, n)?;
    let is_trans = trans_pairs
        .iter()
        .any(|&(a, b)| (a == i && b == j) || (a == j && b == i));

    let tol = 5.0_f64.to_radians();
    Some(if is_trans {
        (std::f64::consts::PI - tol, std::f64::consts::PI)
    } else {
        (90.0_f64.to_radians() - tol, 90.0_f64.to_radians() + tol)
    })
}

/// The rings every ring-aware bound below reads: the smallest ring through
/// each ring bond, as [`crate::molecule::topology::smallest_rings`] finds
/// them, rather than a cycle basis.
///
/// A fundamental cycle basis has exactly `bonds - atoms + 1` members, and
/// nothing says they are the small ones. Naphthalene's basis may hold the
/// ten-membered perimeter in place of one of its six-rings, and
/// [`apply_aromatic_ring_bounds`] then pinned that perimeter's chords as if
/// it were a regular decagon -- bounds of [2.26, 2.30] A across a coumarin
/// where the atoms sit at 2.44. Cubane's basis holds five of its six faces,
/// so the angles on the sixth never got a four-ring value at all and were
/// left at a tetrahedral 109.5 where the ring holds them at 90: a 1-3
/// bound of [2.41, 2.63] against a real 2.17, the worst in the matrix.
///
/// The symmetrized set has neither problem, and it is the set the force
/// field and the typing rules already agree on.
pub(crate) fn ring_set<T: MoleculeCore>(mol: &T) -> FxHashSet<BitSet> {
    crate::molecule::topology::smallest_rings(mol)
        .into_iter()
        .map(|ring| ring.into_iter().collect())
        .collect()
}

/// Angle bounds for the 1-3 pair `(i, j)` through center `k`, from the most
/// specific thing that knows the answer: the ring's own closure, then the
/// ring's size, then MMFF's equilibrium angle, then hybridisation.
///
/// The order is measured, not assumed. MMFF's angle types are keyed on
/// atoms and know a three- or four-ring from its types and nothing larger,
/// so consulting MMFF before [`ring_angle_from_size`] loses the five-ring
/// aromatics the table was built for -- thiophene and thiazole go back to a
/// 120-degree centre where the ring holds them at 108. End to end on
/// PubChem that order costs twelve molecules and 0.6 kcal/mol (155 of 287
/// at +2.42, against 168 at +1.80), which is the whole of this change's
/// benefit, spent.
pub(super) fn angle_bounds_for_triple<T: MoleculeCore>(
    mol: &T,
    i: usize,
    k: usize,
    j: usize,
    cycles: &FxHashSet<BitSet>,
    lengths: &Geometry,
) -> (f64, f64) {
    if let Some(bounds) = square_planar_angle(mol, k, i, j) {
        return bounds;
    }

    let smallest = cycles
        .iter()
        .filter(|cycle| cycle.contains(i) && cycle.contains(k) && cycle.contains(j))
        .min_by_key(|cycle| cycle.count());

    match smallest {
        // Three and four: the closure fixes the angle, and this derives it
        // from the ring's own bond lengths rather than from any table.
        Some(cycle) if cycle.count() <= 4 => ring_angle_bounds(mol, i, k, j, cycle, lengths),
        // Five to eight: the ring does not fix the angle, but it says more
        // than hybridisation alone does.
        Some(cycle) if cycle.count() <= MAX_TABULATED_RING => {
            match lengths.planar_ring_angle(i, k, j) {
                Some(centre) => {
                    let (low, high) = estimate_hybridization_angle(mol, k);
                    let window = 0.5 * (high - low);
                    (
                        (centre - window).max(0.0),
                        (centre + window).min(std::f64::consts::PI),
                    )
                }
                None => ring_angle_from_size(mol, k, cycle.count()),
            }
        }
        // Anything the rings do not speak for: MMFF's own equilibrium
        // angle. A better centre than hybridisation for every case
        // measured -- an amide nitrogen, an ether, a sulfoxide, a
        // hypervalent sulfur -- but it is keyed on atom types, which know
        // a three- or four-ring and nothing larger, so the ring arms above
        // come first.
        _ if lengths.angle(i, k, j).is_some() => {
            let centre = lengths.angle(i, k, j).unwrap_or_default();
            let tolerance = MMFF_ANGLE_TOLERANCE.to_radians();
            (
                (centre - tolerance).max(0.0),
                (centre + tolerance).min(std::f64::consts::PI),
            )
        }
        _ => estimate_hybridization_angle(mol, k),
    }
}

/// The largest ring whose angles are worth tabulating. Past eight a ring is
/// floppy enough that its size says nothing hybridisation does not, which is
/// where RDKit stops too.
const MAX_TABULATED_RING: usize = 8;

/// Where a ring atom's angle sits, by hybridisation and ring size.
///
/// RDKit's `_getRingAngle`, and worth copying for the two entries that differ
/// from a plain hybridisation estimate:
///
/// * **sp2 in a ring of eight or fewer** takes the regular polygon's interior
///   angle -- 108 degrees in a five-ring, not 120. Furan, thiophene, pyrrole,
///   imidazole: a twelve-degree error on some of the commonest rings there
///   are.
/// * **sp3 in a five-ring** sits at 104 degrees, not 109.5. Cyclopentane's
///   angles are compressed and its 1-3 distances know it.
///
/// Measured by `tests/bounds_quality.py`, which checks our bounds against
/// real conformers rather than against another energy, this halves the 1-3
/// violation rate -- 12.6% of heavy-atom 1-3 pairs excluded the truth,
/// 5.5% do now -- and fixes the worst offenders outright: thiophene and
/// thiazole held their 1-3 floors 0.22 and 0.27 Angstroms above where the
/// atoms actually sit. End to end it is inside the conformer metric's own
/// noise, which is the reason to decide it on the bound: a bound that
/// excludes the truth is a lie the refinement spends iterations on, whether
/// or not a downstream mean can see it.
///
/// The *window* stays the one hybridisation asked for. Re-centring and
/// narrowing to three degrees measured worse -- a ring atom's angle is not
/// known to three degrees, and saying it is only argues with the closure.
fn ring_angle_from_size<T: MoleculeCore>(mol: &T, atom: usize, ring_size: usize) -> (f64, f64) {
    let (lower, upper) = estimate_hybridization_angle(mol, atom);
    let half_width = (upper - lower) / 2.0;
    let centre = match hybridisation_of(mol, atom) {
        RingHybridisation::Sp2 => ring_interior_angle(ring_size),
        RingHybridisation::Sp3 if ring_size == 5 => 104.0_f64.to_radians(),
        RingHybridisation::Sp3 | RingHybridisation::Other => return (lower, upper),
    };
    (
        (centre - half_width).max(0.0),
        (centre + half_width).min(std::f64::consts::PI),
    )
}

enum RingHybridisation {
    Sp2,
    Sp3,
    Other,
}

/// The module's usual classification tree, reduced to what the ring angle
/// table asks about.
fn hybridisation_of<T: MoleculeCore>(mol: &T, atom: usize) -> RingHybridisation {
    let Some(bonds) = mol.get_atom_bonds(atom) else {
        return RingHybridisation::Other;
    };
    let doubles = bonds
        .iter()
        .filter(|bond| bond.bond_order == BondOrder::Double)
        .count();
    if bonds
        .iter()
        .any(|bond| bond.bond_order == BondOrder::Triple)
        || doubles >= 2
    {
        return RingHybridisation::Other;
    }
    if bonds
        .iter()
        .any(|bond| matches!(bond.bond_order, BondOrder::Aromatic | BondOrder::Double))
    {
        return RingHybridisation::Sp2;
    }
    RingHybridisation::Sp3
}

/// The angle at `k` of the 1-3 pair `(i, j)` inside a 3- or 4-membered ring,
/// as that ring's own bond lengths make it -- not the regular polygon's
/// interior angle, which is right only when the sides are equal. A
/// phosphirane's C-P-C is 49 degrees, not 60, because its P-C bonds are
/// 1.85 A against the C-C's 1.54; a 60-degree window there demands a 1-3
/// distance longer than the C-C bond's own upper bound, and that
/// contradiction is what used to make the phosphorus, sulfur and silicon
/// small rings die with infeasible bounds.
///
/// A three-ring's window is exact: it is the angle the three bonds' length
/// bounds themselves allow, so the 1-3 distance it produces sits inside the
/// third bond's own bounds by construction. A four-ring's diagonal is not
/// fixed by its sides, so the centre is the angle of the cyclic
/// quadrilateral on the ring's four sides -- the one arrangement in which
/// both triangles through the diagonal agree on its length, which keeps
/// the two 1-3 paths around the ring consistent with each other -- with
/// the same +-3 degrees the regular polygon used to get.
fn ring_angle_bounds<T: MoleculeCore>(
    mol: &T,
    i: usize,
    k: usize,
    j: usize,
    cycle: &BitSet,
    lengths: &Geometry,
) -> (f64, f64) {
    let regular = regular_polygon_angle_bounds(cycle.count());
    let Some(r_ik) = lengths.nominal(mol, i, k) else {
        return regular;
    };
    let Some(r_kj) = lengths.nominal(mol, k, j) else {
        return regular;
    };

    // A three-ring is a triangle -- as is a four-ring carrying a chord
    // across it -- and the angle at `k` follows exactly from the three
    // bond lengths: widest with the flanking bonds at their shortest and
    // the opposite side at its longest, narrowest the other way round.
    if let Some(r_ij) = lengths.nominal(mol, i, j) {
        let widest = triangle_angle(
            r_ik - BOND_TOLERANCE,
            r_kj - BOND_TOLERANCE,
            r_ij + BOND_TOLERANCE,
        );
        let narrowest = triangle_angle(
            r_ik + BOND_TOLERANCE,
            r_kj + BOND_TOLERANCE,
            r_ij - BOND_TOLERANCE,
        );
        return (narrowest, widest);
    }

    let Some(fourth) = cycle
        .iter()
        .find(|&atom| atom != i && atom != k && atom != j)
    else {
        return regular;
    };
    let Some(r_lj) = lengths.nominal(mol, fourth, j) else {
        return regular;
    };
    let Some(r_il) = lengths.nominal(mol, i, fourth) else {
        return regular;
    };
    let cosine = (r_ik * r_ik + r_kj * r_kj - r_lj * r_lj - r_il * r_il)
        / (2.0 * (r_ik * r_kj + r_lj * r_il));
    let centre = cosine.clamp(-1.0, 1.0).acos();
    let tol = 3.0_f64.to_radians();
    (
        (centre - tol).max(0.0),
        (centre + tol).min(std::f64::consts::PI),
    )
}

/// The regular polygon's interior angle with the tolerance the ring
/// overrides historically carried: the fallback when a ring's own bonds
/// are not to hand to do better with.
fn regular_polygon_angle_bounds(ring_size: usize) -> (f64, f64) {
    let theta = ring_interior_angle(ring_size);
    let tol = 3.0_f64.to_radians();
    (
        (theta - tol).max(0.0),
        (theta + tol).min(std::f64::consts::PI),
    )
}

/// The angle opposite the side `c` in a triangle with sides `a`, `b`, `c`.
fn triangle_angle(a: f64, b: f64, c: f64) -> f64 {
    ((a * a + b * b - c * c) / (2.0 * a * b))
        .clamp(-1.0, 1.0)
        .acos()
}

/// The nominal length of the `a`-`b` bond: the two covalent radii, scaled
/// for the bond's order. The 1-2 bounds and the small-ring angle bounds
/// must agree on this, or the constraints each derives for a ring can
/// contradict the ones the bond lengths set.
fn covalent_bond_length<T: MoleculeCore>(mol: &T, a: usize, b: usize) -> Option<f64> {
    let bond = mol
        .get_atom_bonds(a)?
        .iter()
        .find(|bond| bond.target == b)?;
    Some(
        (mol.atomic_numbers()[a]
            .covalent_radius()
            .unwrap_or(DEFAULT_COVALENT_RADIUS)
            + mol.atomic_numbers()[b]
                .covalent_radius()
                .unwrap_or(DEFAULT_COVALENT_RADIUS))
            * bond_order_factor(bond.bond_order),
    )
}

/// How much longer a bond sits in a relaxed structure than MMFF's own
/// equilibrium length says, in angstrom.
///
/// `r0` is the reference of the stretch term alone, and no bond in a
/// minimised molecule sits there: the other six terms -- above all the 1-3
/// van der Waals repulsion that an angle cannot relieve -- push it outwards
/// until their gradient balances the stretch's. Measured over 862
/// heavy-atom bonds against MMFF-relaxed ETKDGv3 conformers, that balance
/// lands at +0.019 A with a spread of 0.014, so a single constant carries
/// nearly all of it.
const MMFF_MINIMUM_OFFSET: f64 = 0.018;

/// How much wider an angle sits in a relaxed structure than MMFF's own
/// equilibrium value says, in degrees.
///
/// The same effect as [`MMFF_MINIMUM_OFFSET`] and largely the same cause:
/// `theta0` is the bend term's reference, and the 1-3 van der Waals
/// repulsion that the bend cannot relieve opens every angle past it.
/// Measured over 1100 heavy-atom 1-3 triples against MMFF-relaxed ETKDGv3
/// conformers, an sp3 carbon's 1-3 distance comes out 0.045 A longer than
/// a 109.5-degree centre predicts, which is about three degrees.
const MMFF_ANGLE_OFFSET: f64 = 2.5;

/// How far an angle is allowed to sit from its equilibrium value, in
/// degrees, once MMFF has supplied that value rather than a hybridisation
/// guess. Narrower than the guess deserved, because the centre is better.
const MMFF_ANGLE_TOLERANCE: f64 = 8.0;

/// The interior angles of the cyclic polygon on `sides` -- the one planar
/// arrangement with every vertex on a common circle -- in radians, in the
/// same order, where the angle at vertex `i` lies between `sides[i - 1]` and
/// `sides[i]`. `None` when no such polygon exists, which happens when one
/// side is longer than the rest can close around.
///
/// A planar ring's angles are not free: the sides fix them, and a regular
/// polygon's value is only right when the sides are equal. They are not in
/// any heteroaromatic -- thiophene carries two C-S bonds of 1.71 A beside
/// three C-C of 1.37 to 1.42 -- and the error is not small: the regular
/// pentagon says 108 degrees at the sulfur where the molecule holds 92, and
/// the 1-3 floor that follows sat 0.18 A above where the atoms actually
/// are, the worst remaining bound in the matrix. The cyclic polygon says
/// 97. Furan and pyrrole, whose sides are nearly equal, it leaves alone.
///
/// The circle is an assumption and not the truth -- a real ring is not
/// inscribed in one -- but it is the only planar shape derivable from the
/// sides alone, and it is strictly more information than assuming they are
/// equal. RDKit reaches the same family through an empirical "squish" of
/// 0.2 A on the bonds of large heteroatoms in conjugated five-rings; this
/// is the same observation, taken from the geometry instead.
///
/// Solved by bisection on the circumradius, since the total subtended angle
/// `sum 2 asin(s / 2R)` falls monotonically in `R`.
fn cyclic_polygon_angles(sides: &[f64]) -> Option<Vec<f64>> {
    const BISECTION_STEPS: usize = 60;
    let full = std::f64::consts::TAU;
    let subtended = |radius: f64| -> f64 {
        sides
            .iter()
            .map(|side| 2.0 * (side / (2.0 * radius)).min(1.0).asin())
            .sum()
    };

    let mut low = sides.iter().copied().fold(0.0, f64::max) / 2.0;
    if low <= 0.0 || !low.is_finite() || subtended(low) < full {
        // Even the tightest circle cannot close the ring: the sides do not
        // describe a cyclic polygon containing its own centre.
        return None;
    }
    let mut high = low * 2.0;
    for _ in 0..BISECTION_STEPS {
        if subtended(high) <= full {
            break;
        }
        high *= 2.0;
    }
    for _ in 0..BISECTION_STEPS {
        let middle = 0.5 * (low + high);
        if subtended(middle) > full {
            low = middle;
        } else {
            high = middle;
        }
    }
    let radius = 0.5 * (low + high);
    let half = |side: f64| (side / (2.0 * radius)).min(1.0).asin();
    Some(
        (0..sides.len())
            .map(|position| {
                let previous = sides[(position + sides.len() - 1) % sides.len()];
                std::f64::consts::PI - (half(previous) + half(sides[position]))
            })
            .collect(),
    )
}

/// Every aromatic ring's interior angles, keyed on the 1-3 triple they
/// belong to, from [`cyclic_polygon_angles`] on the ring's own nominal bond
/// lengths.
///
/// Restricted to aromatic rings because the model assumes the ring is
/// planar. A saturated five-ring is not -- cyclopentane puckers to 104
/// degrees where any planar pentagon says 108 -- and
/// [`ring_angle_from_size`] has the measured value for that case.
fn planar_ring_angles<T: MoleculeCore>(
    mol: &T,
    cycles: &FxHashSet<BitSet>,
    lengths: &rustc_hash::FxHashMap<(u32, u32), f64>,
) -> rustc_hash::FxHashMap<(u32, u32, u32), f64> {
    let mut table = rustc_hash::FxHashMap::default();
    let nominal = |a: usize, b: usize| -> Option<f64> {
        let (low, high) = order(a, b);
        lengths
            .get(&(low as u32, high as u32))
            .copied()
            .or_else(|| covalent_bond_length(mol, a, b))
    };

    for cycle in cycles {
        let size = cycle.count();
        if !(5..=MAX_TABULATED_RING).contains(&size) {
            continue;
        }
        let Some(ring) = aromatic_ring_order(mol, cycle) else {
            continue;
        };
        let sides: Option<Vec<f64>> = (0..size)
            .map(|position| nominal(ring[position], ring[(position + 1) % size]))
            .collect();
        let Some(sides) = sides else {
            continue;
        };
        let Some(angles) = cyclic_polygon_angles(&sides) else {
            continue;
        };
        for position in 0..size {
            let previous = ring[(position + size - 1) % size];
            let centre = ring[position];
            let next = ring[(position + 1) % size];
            let (low, high) = order(previous, next);
            table.insert((low as u32, centre as u32, high as u32), angles[position]);
        }
    }
    table
}

/// How much further a bond sits from its equilibrium length when the atoms
/// at its ends are crowded, in angstrom, as a function of the heavy
/// neighbours the two of them have between them.
///
/// [`MMFF_MINIMUM_OFFSET`] is a mean, and a mean is the wrong model here:
/// what pushes a bond past `r0` is the other terms' force, and that force
/// is small for an ethane and large for a bond between two quaternary
/// carbons. Measured over 6810 heavy-atom bonds against validated
/// MMFF-relaxed conformers, by the number of heavy neighbours at the two
/// ends:
///
/// | heavy neighbours | bonds | residual past `r0 + 0.018` | outside the window |
/// |---|---|---|---|
/// | 3 | 221 | -0.0098 | 0% |
/// | 4 | 2205 | -0.0027 | 1% |
/// | 5 | 2751 | +0.0036 | 9% |
/// | 6 | 1403 | +0.0144 | 36% |
/// | 7 | 200 | +0.0403 | 81% |
/// | 8 | 30 | +0.0954 | 90% |
///
/// Monotone, mechanical, and worth about a tenth of an angstrom at the top
/// -- which is why a crowded silyl ether's Si-C sits at 2.07 where a window
/// around `r0` stops at 1.868, and why a taxane's cage carries C-C bonds of
/// 1.65. Both were "violations" of ours against geometry that RDKit's own
/// MMFF prefers.
///
/// One constant, not the six measured numbers: a quadratic above five
/// neighbours, giving 0.0105, 0.042 and 0.095 against the measured 0.0144,
/// 0.0403 and 0.0954. A table of six invites reading noise as signal, and
/// the shape here is a steric one.
///
/// Nothing below five neighbours. The measurement says those bonds sit
/// slightly *under* `r0 + 0.018` (-0.0098 at three, -0.0027 at four) and a
/// linear relief fitted to that is tempting and wrong: it extrapolates to
/// -0.012 at two neighbours, where there was no data, and ethane's C-C
/// actually sits 0.005 *above*. The correction only moves bonds the
/// measurement covers, in the direction it measured.
fn crowding_correction(heavy_neighbours: usize) -> f64 {
    const QUADRATIC: f64 = 0.0105;
    const PIVOT: usize = 5;
    const CEILING: usize = 8;
    if heavy_neighbours <= PIVOT {
        return 0.0;
    }
    let excess = (heavy_neighbours.min(CEILING) - PIVOT) as f64;
    QUADRATIC * excess * excess
}

/// Where every bond length and bond angle comes from: MMFF94's own
/// equilibrium values where it can type the molecule, and the covalent-radii
/// sum and the hybridisation heuristic where it cannot.
///
/// The centre of the 1-2 window is the single number the whole matrix is
/// built on -- the 1-3 bounds turn it into a distance through an angle, the
/// 1-4 ceiling through two of them -- so being wrong about it is wrong
/// everywhere at once, and tightening a window around a wrong centre only
/// makes it wronger.
///
/// Both of this crate's own models were wrong about it in a way that showed
/// up as soon as `tests/bounds_quality.py` could compare a bound against
/// real geometry rather than against another energy:
///
/// * **lengths.** The covalent-radii sum with one correction per bond order
///   ([`bond_order_factor`]) put a quarter of all heavy-atom bonds outside
///   their own +-0.02 A window, at a mean error of 0.032 A -- as large as
///   the window. It cannot do much better in principle, since one factor per
///   bond order has to cover every element pair that order occurs between.
///   MMFF has a number for each pair: 12.9% outside, at 0.011.
/// * **angles.** [`estimate_hybridization_angle`] knows four cases, and the
///   ones it does not know are not rare: an amide nitrogen is planar, an
///   ether closes to 113 rather than 109.5, a sulfoxide to 97, and an sp3
///   carbon's angles sit nearer 112 than 109.5 whatever the textbook says.
///   MMFF's angle types know all four, and its own ring perception supplies
///   the three- and four-ring values that a fundamental cycle basis can
///   miss outright.
///
/// Both tables are the ones the relaxation that follows will pull towards,
/// so the bounds and the force field stop disagreeing about where an atom
/// belongs. Building them costs one MMFF typing pass per molecule, which is
/// why this is collected once in [`super::embed_connected`] and shared with
/// [`super::forcefield::collect_angle_bends`].
pub(crate) struct Geometry {
    /// Bond lengths, keyed on `(min, max)` atom index, with
    /// [`MMFF_MINIMUM_OFFSET`] already added.
    lengths: rustc_hash::FxHashMap<(u32, u32), f64>,
    /// Angles in radians, keyed on `(min, centre, max)`, with
    /// [`MMFF_ANGLE_OFFSET`] already added.
    angles: rustc_hash::FxHashMap<(u32, u32, u32), f64>,
    /// Aromatic rings' interior angles in radians, from the ring's own
    /// sides: see [`planar_ring_angles`]. Keyed the same way, and consulted
    /// before both the ring-size table and MMFF, because a planar ring's
    /// sides fix its angles and nothing more general can know that.
    planar_rings: rustc_hash::FxHashMap<(u32, u32, u32), f64>,
}

impl Geometry {
    /// Types `mol` with MMFF94 and keeps every equilibrium length and angle
    /// it covers. An untypable molecule gets an empty table, which is a
    /// complete and silent fallback to this module's own models: the bounds
    /// matrix is still built, exactly as it was before MMFF had a say.
    pub(crate) fn for_molecule<T: MoleculeCore>(mol: &T, cycles: &FxHashSet<BitSet>) -> Self {
        match crate::molecule::forcefield::mmff::ForceField::new(mol) {
            Ok(field) => Self::from_field(&field, mol, cycles),
            Err(_) => Self::covalent_only(mol, cycles),
        }
    }

    /// The same, from a field the caller has already built, which is how
    /// [`super::embed_connected`] gets it: the molecule is typed once for
    /// the relaxation and the bounds read the geometry off that.
    pub(crate) fn from_field<T: MoleculeCore>(
        field: &crate::molecule::forcefield::mmff::ForceField,
        mol: &T,
        cycles: &FxHashSet<BitSet>,
    ) -> Self {
        let (measured_lengths, measured_angles) = field.equilibrium_geometry();
        let mut lengths = rustc_hash::FxHashMap::default();
        let mut angles = rustc_hash::FxHashMap::default();
        lengths.reserve(measured_lengths.len());
        let heavy_degree = |atom: usize| -> usize {
            mol.get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .filter(|bond| mol.atomic_numbers()[bond.target()] != 1)
                .count()
        };
        for (first, second, length, _stiffness) in measured_lengths {
            let crowding = heavy_degree(first) + heavy_degree(second);
            lengths.insert(
                (first as u32, second as u32),
                length + MMFF_MINIMUM_OFFSET + crowding_correction(crowding),
            );
        }
        angles.reserve(measured_angles.len());
        for (first, centre, last, degrees) in measured_angles {
            angles.insert(
                (first as u32, centre as u32, last as u32),
                (degrees + MMFF_ANGLE_OFFSET).to_radians(),
            );
        }
        let planar_rings = planar_ring_angles(mol, cycles, &lengths);
        Self {
            lengths,
            angles,
            planar_rings,
        }
    }

    /// No MMFF geometry at all: covalent radii and the hybridisation
    /// heuristic, for a molecule MMFF cannot type.
    pub(crate) fn covalent_only<T: MoleculeCore>(mol: &T, cycles: &FxHashSet<BitSet>) -> Self {
        let lengths = rustc_hash::FxHashMap::default();
        let planar_rings = planar_ring_angles(mol, cycles, &lengths);
        Self {
            lengths,
            angles: rustc_hash::FxHashMap::default(),
            planar_rings,
        }
    }

    /// The nominal length of the bond between `a` and `b`, or `None` when
    /// they are not bonded.
    fn nominal<T: MoleculeCore>(&self, mol: &T, a: usize, b: usize) -> Option<f64> {
        let (low, high) = order(a, b);
        if let Some(&length) = self.lengths.get(&(low as u32, high as u32)) {
            return Some(length);
        }
        covalent_bond_length(mol, a, b)
    }

    /// The angle an aromatic ring's own sides put at `centre`, in radians.
    fn planar_ring_angle(&self, i: usize, centre: usize, j: usize) -> Option<f64> {
        let (low, high) = order(i, j);
        self.planar_rings
            .get(&(low as u32, centre as u32, high as u32))
            .copied()
    }

    /// MMFF's equilibrium angle at `centre`, in radians, if it has one.
    fn angle(&self, i: usize, centre: usize, j: usize) -> Option<f64> {
        let (low, high) = order(i, j);
        self.angles
            .get(&(low as u32, centre as u32, high as u32))
            .copied()
    }
}

fn order(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// One pair of atoms, the window the bounds matrix allows them, and the
/// squares and reciprocals its violation energy is written in terms of.
///
/// Hoisted out of the matrix once per molecule because every refinement
/// iteration reads every pair: a `SquareMatrix` lookup costs a row-major index
/// computation and a bounds check per bound, a flat slice of these costs
/// neither, and the reciprocals turn the energy's two divisions into
/// multiplications.
#[derive(Clone, Copy)]
pub(crate) struct Pair {
    pub(crate) i: u32,
    pub(crate) j: u32,
    lower_sq: f64,
    upper_sq: f64,
    inverse_lower_sq: f64,
    inverse_upper_sq: f64,
}

/// Every pair the bounds matrix restrains, and the subset whose window is
/// narrow enough to say where an atom goes.
///
/// The split is RDKit's `basinThresh`: its first distance-geometry
/// minimisation gives a term only to a pair whose bounds are within 5 A of
/// each other (`DistGeom::constructForceField`), on the grounds that a pair
/// free to sit anywhere between a contact radius and the length of the
/// molecule is not information. It matters most where it costs most -- for a
/// ninety-atom peptide 1295 of 4005 pairs survive it, for cocaine 599 of 903
/// -- because the pairs it drops are the long, floppy, uninformative ones
/// that a big molecule has quadratically many of.
///
/// Only the passes that shape a fresh trial get the short list -- the 4D
/// chirality stage and the bounds-only trial refinement. RDKit does the
/// same: the experimental-torsion stage that follows theirs restrains every
/// remaining pair to the full bounds matrix
/// (`addLongRangeDistanceConstraints`), and so do the touch-up and the
/// combined refinement here, so nothing is finally free of its bounds.
///
/// The combined refinement was tried on the short list and is not: it is
/// one continuous pass for a reason (see [`super::embed_connected`]), and
/// giving the long pairs their own second pass -- or a short pruned pass
/// before it -- costs what splitting it costs. Measured at one trial with
/// no torsion search: 727 ms against 779 for the corpus, and PubChem's 287
/// molecules go from 165 at least matching RDKit at a mean of +1.47
/// kcal/mol to 156 at +2.39. A tenth of the time is not worth nine
/// molecules and nine tenths of a kcal/mol.
pub(crate) struct Pairs {
    all: Vec<Pair>,
    tight: Vec<Pair>,
}

/// How wide a window may be and still shape the first refinement, in
/// Angstroms. RDKit's `basinThresh` default, and for the same reason.
const TIGHT_WINDOW: f64 = 5.0;

impl Pairs {
    pub(crate) fn collect(bounds: &SquareMatrix<f64>) -> Self {
        let n = bounds.nrows();
        let mut all = Vec::with_capacity(n * n / 2);
        let mut tight = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let lower = get_lower(bounds, i, j);
                let upper = get_upper(bounds, i, j);
                let pair = Pair {
                    i: i as u32,
                    j: j as u32,
                    lower_sq: lower * lower,
                    upper_sq: upper * upper,
                    inverse_lower_sq: 1.0 / (lower * lower),
                    inverse_upper_sq: 1.0 / (upper * upper),
                };
                if upper - lower <= TIGHT_WINDOW {
                    tight.push(pair);
                }
                all.push(pair);
            }
        }
        Pairs { all, tight }
    }

    pub(crate) fn all(&self) -> &[Pair] {
        &self.all
    }

    pub(crate) fn tight(&self) -> &[Pair] {
        &self.tight
    }
}

/// One pair's bounds-violation energy and `dE/dp_i`, in whichever dimension
/// the caller is working in. `dE/dp_j` is the negation of `dE/dp_i`, by
/// symmetry of `d^2`.
///
/// `E = max(0, d^2 - u^2)^2 / u^2 + max(0, l^2 - d^2)^2 / l^2`: flat between
/// the bounds, quartic in the distance outside them, and free of the `sqrt`
/// a distance would need.
pub(crate) fn pair_violation<C: Coordinate>(pair: &Pair, positions: &[C]) -> (f64, C) {
    let difference = positions[pair.i as usize] - positions[pair.j as usize];
    let squared = difference.dot(&difference);

    let mut energy = 0.0;
    let mut gradient = C::default();

    if squared > pair.upper_sq {
        let excess = squared - pair.upper_sq;
        energy += excess * excess * pair.inverse_upper_sq;
        gradient += difference * (4.0 * excess * pair.inverse_upper_sq);
    }
    if squared < pair.lower_sq {
        let deficit = pair.lower_sq - squared;
        energy += deficit * deficit * pair.inverse_lower_sq;
        gradient += -(difference * (4.0 * deficit * pair.inverse_lower_sq));
    }

    (energy, gradient)
}

/// Every pair's violation energy summed, and -- given somewhere to put it --
/// every pair's gradient added into `gradient`.
///
/// [`pair_violation`] four pairs at a time, in whichever lanes the CPU
/// offers -- see [`lanes::dispatch`] -- with the same bits whichever it is.
/// Every distance-geometry pass spends most of its time in this loop.
pub(crate) fn pair_violations<C: Coordinate>(
    pairs: &[Pair],
    positions: &[C],
    gradient: Option<&mut [C]>,
) -> f64 {
    struct Run<'a, C>(&'a [Pair], &'a [C], Option<&'a mut [C]>);
    impl<C: Coordinate> lanes::LanesOp for Run<'_, C> {
        type Output = f64;
        #[inline(always)]
        fn run<L: Lanes>(self, simd: L) -> f64 {
            pair_violations_in(simd, self.0, self.1, self.2)
        }
    }
    lanes::dispatch(Run(pairs, positions, gradient))
}

/// [`pair_violations`] on the lanes `simd` provides.
///
/// Branch-free: a pair inside its window has a zero excess and a zero
/// deficit rather than a skipped term, which gives each pair exactly the
/// energy and gradient [`pair_violation`] does. The energies are then added
/// one pair at a time, in order, rather than per lane: that keeps the total
/// bit for bit the scalar loop's. Summing per lane was tried, and the last
/// few bits it moves are enough to send line searches elsewhere -- PubChem's
/// 287 came out 14 better and 19 worse against RDKit, a mean 0.15 kcal/mol
/// worse, for perhaps one percent more speed than this version.
#[inline(always)]
fn pair_violations_in<L: Lanes, C: Coordinate>(
    simd: L,
    pairs: &[Pair],
    positions: &[C],
    mut gradient: Option<&mut [C]>,
) -> f64 {
    let zero = simd.splat(0.0);
    let four = simd.splat(4.0);
    let mut energy = 0.0;
    let (chunks, remainder) = pairs.as_chunks::<LANES>();
    for chunk in chunks {
        let field = |value: fn(&Pair) -> f64| {
            simd.from_array([
                value(&chunk[0]),
                value(&chunk[1]),
                value(&chunk[2]),
                value(&chunk[3]),
            ])
        };
        let axis = |atom: fn(&Pair) -> u32, axis: usize| {
            simd.from_array(std::array::from_fn(|lane| {
                positions[atom(&chunk[lane]) as usize].axis(axis)
            }))
        };
        let mut difference = [zero; 4];
        for (dimension, slot) in difference.iter_mut().enumerate().take(C::DIMENSIONS) {
            *slot = axis(|pair| pair.i, dimension) - axis(|pair| pair.j, dimension);
        }
        let mut squared = difference[0] * difference[0];
        for slot in &difference[1..C::DIMENSIONS] {
            squared += *slot * *slot;
        }

        let (upper_sq, lower_sq) = (field(|pair| pair.upper_sq), field(|pair| pair.lower_sq));
        let inverse_upper_sq = field(|pair| pair.inverse_upper_sq);
        let inverse_lower_sq = field(|pair| pair.inverse_lower_sq);
        let excess = simd.select_greater(squared, upper_sq, squared - upper_sq, zero);
        let deficit = simd.select_greater(lower_sq, squared, lower_sq - squared, zero);
        let violations = excess * excess * inverse_upper_sq + deficit * deficit * inverse_lower_sq;
        for violation in simd.to_array(violations) {
            energy += violation;
        }

        if let Some(gradient) = gradient.as_deref_mut() {
            let scale = four * excess * inverse_upper_sq - four * deficit * inverse_lower_sq;
            let mut steps = [[0.0; LANES]; 4];
            for (dimension, step) in steps.iter_mut().enumerate().take(C::DIMENSIONS) {
                *step = simd.to_array(difference[dimension] * scale);
            }
            for (lane, pair) in chunk.iter().enumerate() {
                let mut step = C::default();
                for (dimension, values) in steps.iter().enumerate().take(C::DIMENSIONS) {
                    *step.axis_mut(dimension) = values[lane];
                }
                gradient[pair.i as usize] += step;
                gradient[pair.j as usize] += -step;
            }
        }
    }

    for pair in remainder {
        let (violation, derivative) = pair_violation(pair, positions);
        energy += violation;
        if let Some(gradient) = gradient.as_deref_mut() {
            gradient[pair.i as usize] += derivative;
            gradient[pair.j as usize] += -derivative;
        }
    }
    energy
}

/// Distance windows for every 1-4 pair whose torsion the geometry actually
/// constrains.
///
/// Three bond lengths and two bond angles leave the fourth atom on a circle,
/// and the two ends of that circle are the syn-periplanar and anti
/// arrangements -- so without knowing the torsion at all, the distance is
/// between [`fourteen_distance`]'s two values. Before this, a 1-4 pair was
/// told only that it was at least a contact radius apart and at most the
/// length of the molecule; smoothing narrowed the ceiling but nothing spoke
/// for the floor.
///
/// Where the torsion *is* known the window is one end of that range: a
/// conjugated pair of sp2 centres inside a ring of eight or fewer is cis, and
/// an explicitly tagged double bond is whichever side its tag names.
///
/// **A path inside a ring of five bonds or fewer gets nothing.** There the
/// ring's closure sets the distance, not the torsion, and a window derived as
/// though the three bonds were a free chain is the wrong model -- it is what
/// the 2026-09-22 attempt did, and what put 94 kcal/mol of strain into
/// cocaine's bridged tropane. RDKit draws the line in the same place
/// (`set14Bounds` computes an in-ring window only when the ring has more than
/// five bonds).
fn apply_fourteen_bounds<T: MoleculeCore>(
    mol: &T,
    cycles: &FxHashSet<BitSet>,
    is_specific: &SquareMatrix<bool>,
    bounds: &mut SquareMatrix<f64>,
    lengths: &Geometry,
) {
    let n = mol.len();
    for middle_a in 0..n {
        let Some(bonds_a) = mol.get_atom_bonds(middle_a) else {
            continue;
        };
        for middle_bond in bonds_a {
            let middle_b = middle_bond.target();
            if middle_b <= middle_a {
                continue;
            }
            // The smallest ring carrying the whole path decides whether the
            // torsion is ours to speak about.
            let Some(first_bonds) = mol.get_atom_bonds(middle_a) else {
                continue;
            };
            let Some(last_bonds) = mol.get_atom_bonds(middle_b) else {
                continue;
            };
            for first_bond in first_bonds {
                let first = first_bond.target();
                if first == middle_b {
                    continue;
                }
                for last_bond in last_bonds {
                    let last = last_bond.target();
                    if last == middle_a || last == first {
                        continue;
                    }
                    let smallest = cycles
                        .iter()
                        .filter(|cycle| {
                            cycle.contains(first)
                                && cycle.contains(middle_a)
                                && cycle.contains(middle_b)
                                && cycle.contains(last)
                        })
                        .map(|cycle| cycle.count())
                        .min();
                    if smallest.is_some_and(|size| size <= MAX_CLOSED_RING) {
                        continue;
                    }
                    // A pair that already has a bond or an angle speaking for
                    // it is not a 1-4 pair, whatever this path says: in a
                    // bridged system one path's three bonds are another's
                    // two, and a measured bond length or a ring angle is
                    // worth more than a torsion window. Widening those was
                    // this pass's first bug -- 1-3 violations went from 5.5%
                    // to 12.6%.
                    let (ordered_first, ordered_last) = order(first, last);
                    if is_specific[[ordered_first, ordered_last]] {
                        continue;
                    }
                    let (Some(first_length), Some(middle_length), Some(last_length)) = (
                        lengths.nominal(mol, first, middle_a),
                        lengths.nominal(mol, middle_a, middle_b),
                        lengths.nominal(mol, middle_b, last),
                    ) else {
                        continue;
                    };
                    let (near_low, near_high) =
                        angle_bounds_for_triple(mol, first, middle_a, middle_b, cycles, lengths);
                    let (far_low, far_high) =
                        angle_bounds_for_triple(mol, middle_a, middle_b, last, cycles, lengths);

                    let short = first_length - BOND_TOLERANCE;
                    let long = first_length + BOND_TOLERANCE;
                    let middle_short = middle_length - BOND_TOLERANCE;
                    let middle_long = middle_length + BOND_TOLERANCE;
                    let last_short = last_length - BOND_TOLERANCE;
                    let last_long = last_length + BOND_TOLERANCE;

                    let cis =
                        fourteen_distance(short, middle_short, last_short, near_low, far_low, true);
                    let trans =
                        fourteen_distance(long, middle_long, last_long, near_high, far_high, false);
                    // The ceiling only. The anti arrangement is as far apart
                    // as three bonds and two angles allow, which is real
                    // information and tighter than anything smoothing can
                    // derive. The floor is left where it was: the contact
                    // radius already there is the better number for a
                    // saturated chain, whose syn arrangement is sterically
                    // shut anyway, and never lowering a floor makes the
                    // infeasible-floor failure of the previous attempt
                    // impossible rather than unlikely.
                    let ceiling = cis.max(trans) + FOURTEEN_CEILING_SLACK;
                    let held = get_upper(bounds, first, last);
                    if ceiling < held && ceiling > get_lower(bounds, first, last) {
                        set_bounds(bounds, first, last, get_lower(bounds, first, last), ceiling);
                    }
                }
            }
        }
    }
}

/// Slack on the 1-4 ceiling, in Angstroms.
///
/// The ceiling is only as good as the three bond lengths and two angles under
/// it, and both are estimates -- a third of the bond bounds do not contain
/// the truth, measured. Worse, a ceiling that is too low does not simply fail
/// to help: triangle smoothing turns it into a *raised floor* somewhere else
/// (`l_ij >= l_ik - u_kj`), and the 1-3 bounds pay for it. Without this,
/// 1-3 violations went from 5.5% to 12.6%.
const FOURTEEN_CEILING_SLACK: f64 = 0.06;

/// The largest ring whose own closure, rather than the torsion, sets its 1-4
/// distances. RDKit's line is the same.
const MAX_CLOSED_RING: usize = 5;

/// The 1-4 distance when the torsion is syn-periplanar, the closest the two
/// ends can come, and when it is anti, the furthest.
///
/// RDKit's `compute14DistCis` and `compute14DistTrans`: lay the three bonds
/// out in a plane and the two arrangements differ only in the sign of the
/// far end's perpendicular offset.
fn fourteen_distance(first: f64, middle: f64, last: f64, near: f64, far: f64, cis: bool) -> f64 {
    let x = middle - last * far.cos() - first * near.cos();
    let y = if cis {
        last * far.sin() - first * near.sin()
    } else {
        last * far.sin() + first * near.sin()
    };
    (x * x + y * y).sqrt()
}

pub(crate) fn get_upper(bounds: &SquareMatrix<f64>, a: usize, b: usize) -> f64 {
    let (i, j) = order(a, b);
    bounds[[i, j]]
}

pub(crate) fn get_lower(bounds: &SquareMatrix<f64>, a: usize, b: usize) -> f64 {
    let (i, j) = order(a, b);
    bounds[[j, i]]
}

fn set_bounds(bounds: &mut SquareMatrix<f64>, a: usize, b: usize, lower: f64, upper: f64) {
    let (i, j) = order(a, b);
    bounds[[i, j]] = upper;
    bounds[[j, i]] = lower;
}

fn intersect_bounds(bounds: &mut SquareMatrix<f64>, a: usize, b: usize, lower: f64, upper: f64) {
    let cur_lower = get_lower(bounds, a, b);
    let cur_upper = get_upper(bounds, a, b);
    set_bounds(bounds, a, b, cur_lower.max(lower), cur_upper.min(upper));
}

fn law_of_cosines_distance(d1: f64, d2: f64, theta: f64) -> f64 {
    (d1 * d1 + d2 * d2 - 2.0 * d1 * d2 * theta.cos())
        .max(0.0)
        .sqrt()
}

/// Builds the initial (pre-smoothing) distance-bounds matrix for `mol`:
/// exact-ish 1-2 (bonded) bounds, 1-3 (angle-derived) bounds, and loose
/// van-der-Waals-based 1-4+ bounds.
///
/// # The 1-4 bounds RDKit has and this does not
///
/// Their `BoundsMatrixBuilder` fixes every 1-4 distance from the three bond
/// lengths and the two bond angles: the cis arrangement is the floor and
/// the trans one the ceiling, narrowed to one or the other where a double
/// bond, an amide or a small ring decides the torsion, and widened by
/// `useMacrocycle14config` for an amide inside a macrocycle. Ours says only
/// that a 1-4 pair is at least a contact radius apart and at most
/// [`DEFAULT_UPPER`], which for butane's end carbons is "between 3 and 1000
/// Angstroms" where the geometry allows 2.5 to 3.9.
///
/// It was implemented and measured and is not here, because it is only
/// right where nothing else decides the torsion. On 287 PubChem molecules
/// at one trial and no torsion search it was worth a lot -- the mean
/// difference against RDKit's ETKDGv3 went from +1.56 to +0.37 kcal/mol --
/// and on the curated corpus it was a disaster, +94 kcal/mol of MMFF strain
/// on cocaine alone, because a bridged tropane cannot satisfy a window
/// derived as though its three bonds were a chain. Restricting it to
/// torsions no small ring constrains, or keeping only the ceiling there,
/// fixed the curated corpus and made PubChem *worse than having no such
/// bounds at all* -- so the interaction with smoothing is not understood,
/// and a rule that helps one distribution while breaking a common molecule
/// is not a rule.
///
/// What the measurement did establish is where the fault is. Comparing
/// these bounds against relaxed structures, the pairs that violate them are
/// in rings, and they violate the *floor*: cocaine's tropane puts two atoms
/// 2.66 A apart where the cis arrangement of its bonds and our angle
/// estimates says 3.21 at the closest. The angles are the weak link --
/// inside a five- to eight-membered ring this crate falls back to a
/// hybridization estimate of 109.5 or 120 degrees with a few degrees of
/// tolerance, where the ring's own geometry is what sets them, and RDKit
/// derives them per ring. Ring-aware angle bounds are the prerequisite, and
/// they are likely the same missing piece as the ring-conformation losses
/// in `tests/embed_budget_benchmark.py`.
pub(crate) fn build_bounds<T: MoleculeCore>(
    mol: &T,
    lengths: &Geometry,
    cycles: &FxHashSet<BitSet>,
) -> Result<SquareMatrix<f64>, EmbedError> {
    let n = mol.len();
    if n < 2 {
        return Err(EmbedError::TooFewAtoms);
    }

    let mut bounds = SquareMatrix::<f64>::from_elem(n, DEFAULT_UPPER);
    // Tracks which pairs have received a real 1-2/1-3 geometric constraint,
    // as opposed to still sitting at the generic van-der-Waals default. Only
    // specific-vs-specific bounds get intersected; a 1-3 estimate must never
    // be intersected against the unrelated (and often stricter-looking, but
    // inapplicable) VDW default, or it can spuriously force lower > upper.
    let mut is_specific = SquareMatrix::<bool>::from_elem(n, false);
    for i in 0..n {
        for j in (i + 1)..n {
            let lower = VDW_LOWER_SCALE
                * (vdw_radius(mol.atomic_numbers()[i]) + vdw_radius(mol.atomic_numbers()[j]));
            set_bounds(&mut bounds, i, j, lower, DEFAULT_UPPER);
        }
    }

    // 1-2 (bonded) bounds.
    for i in 0..n {
        let Some(bonds_i) = mol.get_atom_bonds(i) else {
            continue;
        };
        for bond in bonds_i {
            let k = bond.target();
            if k <= i {
                continue;
            }
            let Some(nominal) = lengths.nominal(mol, i, k) else {
                continue;
            };
            set_bounds(
                &mut bounds,
                i,
                k,
                nominal - BOND_TOLERANCE,
                nominal + BOND_TOLERANCE,
            );
            is_specific[[i, k]] = true;
        }
    }

    // 1-3 (angle-derived) bounds, intersected across every shared neighbor
    // (fused/bridged rings can give the same pair multiple paths) - but only
    // once a specific bound already exists for that pair; the first specific
    // bound for a pair always overwrites the generic default outright.
    for k in 0..n {
        let Some(bonds_k) = mol.get_atom_bonds(k) else {
            continue;
        };
        let neighbors: Vec<usize> = bonds_k.iter().map(|b| b.target()).collect();
        for a in 0..neighbors.len() {
            for b in (a + 1)..neighbors.len() {
                let i = neighbors[a];
                let j = neighbors[b];
                // A 1-3 pair that is also 1-2 is a three-membered ring, and
                // there the bond is the constraint: it is measured, where
                // the angle is estimated. Intersecting the two put an
                // angle's lower bound on top of a bond's upper one --
                // 2.396 A against 1.555 for a bond that is 1.49 long --
                // and in a fused cage the next triple read that corrupted
                // lower bound as its own leg and doubled it, up to 4.4 A
                // between two bonded carbons. Every `BoundsInfeasible` in a
                // random 6000 PubChem molecules was this cascade. RDKit
                // skips the same pairs (`_set13Bounds` asks for no bond
                // between them first).
                if mol
                    .get_atom_bonds(i)
                    .is_some_and(|bonds| bonds.iter().any(|bond| bond.target() == j))
                {
                    continue;
                }
                let (theta_min, theta_max) = angle_bounds_for_triple(mol, i, k, j, cycles, lengths);
                let l_ik = get_lower(&bounds, i, k);
                let u_ik = get_upper(&bounds, i, k);
                let l_kj = get_lower(&bounds, k, j);
                let u_kj = get_upper(&bounds, k, j);

                let new_lower = law_of_cosines_distance(l_ik, l_kj, theta_min);
                let new_upper = law_of_cosines_distance(u_ik, u_kj, theta_max);
                let (oi, oj) = order(i, j);
                if is_specific[[oi, oj]] {
                    intersect_bounds(&mut bounds, i, j, new_lower, new_upper);
                } else {
                    set_bounds(&mut bounds, i, j, new_lower, new_upper);
                    is_specific[[oi, oj]] = true;
                }
            }
        }
    }

    apply_aromatic_ring_bounds(mol, cycles, &mut bounds, &mut is_specific);
    apply_fourteen_bounds(mol, cycles, &is_specific, &mut bounds, lengths);

    Ok(bounds)
}

/// Widening on the chord bounds below, in angstrom, on top of what the ring's
/// own 1-2 bounds already allow. Real aromatic rings are not quite regular
/// polygons -- a substituted benzene's internal angles spread over several
/// degrees -- so the chords need room for that without leaving room to
/// pucker.
const RING_CHORD_TOLERANCE: f64 = 0.05;

/// Pins every distance across an aromatic ring, not just the 1-2 and 1-3
/// ones the general code above reaches.
///
/// Distance geometry only makes a ring planar if the distances say so, and
/// through 1-3 they do not: a hexagon with correct bonds and correct
/// internal angles can still pucker, because puckering is what the 1-4
/// (para) distance controls and nothing was constraining it. The result was
/// aromatic rings with out-of-plane RMSDs of 0.02 to 0.25 A where RDKit's
/// ETKDG gives 0.0002, with the soft [`super::planarity`] restraint left to
/// fight a geometry that distance geometry considered perfectly valid.
///
/// In a planar ring of `m` atoms, the chord spanning `k` bonds is fixed by
/// the side length: `chord_k = side * sin(k*pi/m) / sin(pi/m)`. Applying
/// that for every `k >= 2` leaves puckering no slack.
///
/// Restricted to *aromatic* rings rather than all-sp2 ones. Aromaticity
/// implies planarity; being all-sp2 does not, and the counterexample is not
/// exotic -- cyclooctatetraene is all-sp2, eight-membered, and tub-shaped.
/// (`super::planarity` does flatten it, which is a real if separate bug;
/// hardening it into a distance bound here would make that worse rather
/// than better.)
/// The shortest and longest side the ring's own 1-2 bounds allow, which the
/// regular-polygon chord model reads its chords off.
///
/// The obvious improvement is not one. A regular polygon is a good model of
/// benzene and a poor one of every heteroaromatic -- warfarin's pyranone
/// carries a C-O of 1.37 beside a C-C of 1.40, and one ratio for all of
/// them put the chord across its oxygen at [2.26, 2.30] A where the atoms
/// sit at 2.44 -- so the ring was laid out in the plane instead, bond by
/// bond, from its own lengths and MMFF's own angles, with the residual of
/// `(n - 2) * 180` degrees spread over it to make it close. That is exact
/// for an isolated planar ring and wrong for a fused one, where two rings
/// share atoms and each lays the shared angles out its own way: measured by
/// `tests/bounds_quality.py`, 1-4 violations went from 6 to 25 and the
/// whole matrix from 147 to 182, caffeine's fused pair the worst of it. A
/// loose model that is wrong in both directions beats a tight one that is
/// wrong in one. Handing the 1-3 chords to the angle pass and keeping only
/// step 3 and beyond is a wash as well (27 violated against 30, at 0.29 A
/// of slack against 0.25).
fn regular_chord_bounds(ring: &[usize], bounds: &SquareMatrix<f64>, size: usize) -> Option<(f64, f64)> {
    let (mut side_min, mut side_max) = (f64::MAX, 0.0_f64);
    for step in 0..size {
        let (a, b) = (ring[step], ring[(step + 1) % size]);
        side_min = side_min.min(get_lower(bounds, a, b));
        side_max = side_max.max(get_upper(bounds, a, b));
    }
    (side_min <= side_max).then_some((side_min, side_max))
}

fn apply_aromatic_ring_bounds<T: MoleculeCore>(
    mol: &T,
    cycles: &FxHashSet<BitSet>,
    bounds: &mut SquareMatrix<f64>,
    is_specific: &mut SquareMatrix<bool>,
) {
    for cycle in cycles {
        let size = cycle.count();
        if !(3..=8).contains(&size) {
            continue;
        }
        let Some(ring) = aromatic_ring_order(mol, cycle) else {
            continue;
        };

        let Some((side_min, side_max)) = regular_chord_bounds(&ring, bounds, size) else {
            continue;
        };
        let base = (std::f64::consts::PI / size as f64).sin();
        for step in 2..=(size / 2) {
            let ratio = (step as f64 * std::f64::consts::PI / size as f64).sin() / base;
            let lower = side_min * ratio - RING_CHORD_TOLERANCE;
            let upper = side_max * ratio + RING_CHORD_TOLERANCE;
            for start in 0..size {
                let (a, b) = (ring[start], ring[(start + step) % size]);
                let (oa, ob) = order(a, b);
                if is_specific[[oa, ob]] {
                    intersect_bounds(bounds, a, b, lower, upper);
                } else {
                    set_bounds(bounds, a, b, lower, upper);
                    is_specific[[oa, ob]] = true;
                }
            }
        }
    }
}

/// The cyclic order of an all-aromatic-bonded ring, or `None` if any of its
/// bonds is not aromatic. Same bond-adjacency walk as
/// [`super::planarity`]'s, which needs the order for the same reason: a
/// ring's atoms mean nothing geometrically until they are in ring order.
fn aromatic_ring_order<T: MoleculeCore>(mol: &T, cycle: &BitSet) -> Option<Vec<usize>> {
    let size = cycle.count();
    let start = cycle.iter().next()?;
    let mut order = Vec::with_capacity(size);
    order.push(start);
    let mut previous = None;
    let mut current = start;

    while order.len() < size {
        let next = mol.get_atom_bonds(current)?.iter().find(|bond| {
            bond.bond_order() == BondOrder::Aromatic
                && cycle.contains(bond.target())
                && Some(bond.target()) != previous
                && !order.contains(&bond.target())
        })?;
        previous = Some(current);
        current = next.target();
        order.push(current);
    }

    // The walk must close: the last atom has to be aromatic-bonded back to
    // the first, or this was not a single aromatic cycle.
    let closes = mol
        .get_atom_bonds(current)?
        .iter()
        .any(|bond| bond.target() == start && bond.bond_order() == BondOrder::Aromatic);
    closes.then_some(order)
}

/// Havel's triangle-inequality bounds smoothing: a single interleaved triple
/// loop where, within each `k`, upper-bound tightening happens before the
/// lower-bound update (the lower update consumes the just-tightened uppers).
/// One-time O(n^3) preprocessing, reused across every trial embedding.
pub(crate) fn smooth_bounds(bounds: &mut SquareMatrix<f64>) -> Result<(), EmbedError> {
    let n = bounds.nrows();
    for k in 0..n {
        for i in 0..n {
            if i == k {
                continue;
            }
            for j in (i + 1)..n {
                if j == k {
                    continue;
                }
                let u_ik = get_upper(bounds, i, k);
                let u_kj = get_upper(bounds, k, j);
                let u_ij = get_upper(bounds, i, j);
                let candidate_upper = u_ik + u_kj;
                if u_ij > candidate_upper {
                    set_bounds(bounds, i, j, get_lower(bounds, i, j), candidate_upper);
                }

                let l_ik = get_lower(bounds, i, k);
                let l_kj = get_lower(bounds, k, j);
                let u_ik = get_upper(bounds, i, k);
                let u_kj = get_upper(bounds, k, j);
                let l_ij = get_lower(bounds, i, j);
                let candidate_lower = (l_ik - u_kj).max(l_kj - u_ik);
                if l_ij < candidate_lower {
                    set_bounds(bounds, i, j, candidate_lower, get_upper(bounds, i, j));
                }
            }
        }
    }

    // Feasibility check: repair small inconsistencies with a tiny buffer,
    // otherwise report infeasibility rather than silently embedding garbage.
    let mut fixups = 0usize;
    let mut total = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            total += 1;
            let lower = get_lower(bounds, i, j);
            let upper = get_upper(bounds, i, j);
            if lower > upper {
                fixups += 1;
                set_bounds(bounds, i, j, lower, lower * BOUNDS_REPAIR_EXPANSION);
            }
        }
    }
    if total > 0 && (fixups as f64) / (total as f64) > MAX_BOUNDS_REPAIR_FRACTION {
        return Err(EmbedError::BoundsInfeasible);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::molecular_graph::MolecularGraph;

    fn mol(smiles: &str) -> MolecularGraph {
        let mut m = MolecularGraph::from_smiles_single(smiles).unwrap();
        m.add_hydrogens();
        m
    }

    #[test]
    fn ethane_cc_bond_length() {
        let m = mol("CC");
        let bounds = build_bounds(
            &m,
            &Geometry::for_molecule(&m, &ring_set(&m)),
            &ring_set(&m),
        )
        .unwrap();
        let lower = get_lower(&bounds, 0, 1);
        let upper = get_upper(&bounds, 0, 1);
        assert!(
            lower <= 1.54 && upper >= 1.54,
            "expected ~1.54 A C-C bond, got [{lower}, {upper}]"
        );
    }

    #[test]
    fn benzene_bond_length_and_1_3_angle() {
        let m = mol("c1ccccc1");
        let mut bounds = build_bounds(
            &m,
            &Geometry::for_molecule(&m, &ring_set(&m)),
            &ring_set(&m),
        )
        .unwrap();
        smooth_bounds(&mut bounds).unwrap();

        let lower = get_lower(&bounds, 0, 1);
        let upper = get_upper(&bounds, 0, 1);
        assert!(
            lower <= 1.39 && upper >= 1.39,
            "expected ~1.39 A aromatic C-C bond, got [{lower}, {upper}]"
        );

        // 1-3 pair (atoms 0 and 2, through ring neighbor 1): for a ~120 deg
        // angle with ~1.39 A bonds, the opposite-side distance is
        // 1.39 * sqrt(3) =~ 2.408 A.
        let lower13 = get_lower(&bounds, 0, 2);
        let upper13 = get_upper(&bounds, 0, 2);
        let expected = 2.408;
        assert!(
            lower13 <= expected && upper13 >= expected,
            "1-3 bound [{lower13}, {upper13}] should bracket {expected}"
        );
    }

    #[test]
    fn heteroatom_small_rings_have_feasible_bounds() {
        // A regular polygon's interior angle is right only when its sides are
        // equal. Give it to a ring whose bonds differ in length -- P-C 1.85 A
        // against C-C 1.54 -- and the 1-3 distance it demands is longer than
        // the short bond's own upper bound: infeasible. Every one of these
        // died there before the ring's own bond lengths were made to decide
        // the angle.
        for smiles in ["C1PC1", "C1CS1", "C1C[Si]1", "C1SS1", "C1CPC1"] {
            let m = mol(smiles);
            let mut bounds = build_bounds(
                &m,
                &Geometry::for_molecule(&m, &ring_set(&m)),
                &ring_set(&m),
            )
            .unwrap();
            smooth_bounds(&mut bounds)
                .unwrap_or_else(|_| panic!("{smiles} should have feasible bounds"));
        }
    }

    #[test]
    fn three_ring_angle_follows_the_bond_lengths() {
        // Phosphirane: the angle the three bond lengths make at the
        // phosphorus is ~49 degrees, not the regular polygon's 60.
        let m = mol("C1PC1");
        let cycles = m.compute_fundamental_cycles();
        let (min, max) = angle_bounds_for_triple(
            &m,
            0,
            1,
            2,
            &cycles,
            &Geometry::for_molecule(&m, &ring_set(&m)),
        );
        let (min, max) = (min.to_degrees(), max.to_degrees());
        assert!(
            max < 55.0,
            "C-P-C window should sit well under the regular 60, got [{min}, {max}]"
        );
        assert!(
            min > 45.0,
            "C-P-C window should sit well over 40, got [{min}, {max}]"
        );

        // Cyclopropane's sides are equal, so its window is the regular
        // polygon's own, around 60.
        let m = mol("C1CC1");
        let cycles = m.compute_fundamental_cycles();
        let (min, max) = angle_bounds_for_triple(
            &m,
            0,
            1,
            2,
            &cycles,
            &Geometry::for_molecule(&m, &ring_set(&m)),
        );
        assert!(min.to_degrees() <= 60.0 && max.to_degrees() >= 60.0);
    }

    #[test]
    fn four_ring_angle_follows_the_bond_lengths() {
        // Phosphetane: the cyclic quadrilateral on sides P-C, P-C, C-C, C-C
        // puts the angle at the phosphorus near 79 degrees, not the
        // square's 90. (Atoms: C0-C1-P2-C3.)
        let m = mol("C1CPC1");
        let cycles = m.compute_fundamental_cycles();
        let (min, max) = angle_bounds_for_triple(
            &m,
            1,
            2,
            3,
            &cycles,
            &Geometry::for_molecule(&m, &ring_set(&m)),
        );
        let (min, max) = (min.to_degrees(), max.to_degrees());
        assert!(
            max < 90.0,
            "C-P-C window should sit under the regular 90, got [{min}, {max}]"
        );
        assert!(
            min > 70.0,
            "C-P-C window should sit over 70, got [{min}, {max}]"
        );
    }

    #[test]
    fn cyclobutane_ring_angle_override() {
        // 4-membered ring: the (0,2) and (1,3) pairs are genuine 1-3 pairs
        // (not directly bonded), so the ring-size override (90 deg interior
        // angle) should apply instead of the generic sp3 default (109.5 deg).
        let m = mol("C1CCC1");
        let bounds = build_bounds(
            &m,
            &Geometry::for_molecule(&m, &ring_set(&m)),
            &ring_set(&m),
        )
        .unwrap();

        let bond_lower = get_lower(&bounds, 0, 1);
        let bond_upper = get_upper(&bounds, 0, 1);

        let _lower13 = get_lower(&bounds, 0, 2);
        let upper13 = get_upper(&bounds, 0, 2);

        // 90 deg (+-3 deg ring-override tolerance) -> diagonal close to
        // bond * sqrt(2) =~ 1.414 * bond. Compare against the exact
        // law-of-cosines value at the tolerance edge (93 deg) rather than a
        // naive sqrt(2) approximation, since the +-3 deg band alone shifts
        // the upper bound measurably.
        let ring_tol = 3.0_f64.to_radians();
        let expected_upper = law_of_cosines_distance(
            bond_upper,
            bond_upper,
            std::f64::consts::FRAC_PI_2 + ring_tol,
        );
        // 109.5 deg (generic sp3 default) would give =~ 1.633 * bond instead
        // of ~1.41-1.45 * bond for the 90 deg ring override - a clearly
        // distinguishable difference.
        let sp3_default_diagonal = bond_lower * 1.633;
        assert!(
            (upper13 - expected_upper).abs() < 1e-6,
            "expected ring override upper bound {expected_upper} (90+3 deg), got {upper13}"
        );
        assert!(
            upper13 < sp3_default_diagonal - 0.1,
            "ring override should be clearly tighter than the generic sp3 (109.5 deg) diagonal estimate {sp3_default_diagonal}, got upper={upper13}"
        );
    }

    #[test]
    fn allene_center_is_linear() {
        let m = mol("C=C=C");
        let bounds = build_bounds(
            &m,
            &Geometry::for_molecule(&m, &ring_set(&m)),
            &ring_set(&m),
        )
        .unwrap();

        let l_ik = get_lower(&bounds, 0, 1);
        let u_ik = get_upper(&bounds, 0, 1);
        let l_kj = get_lower(&bounds, 1, 2);
        let u_kj = get_upper(&bounds, 1, 2);
        let u13 = get_upper(&bounds, 0, 2);

        // 180 deg -> the opposite-side distance is exactly the sum of the two
        // bond upper bounds (law of cosines collapses to d1+d2 at theta=pi).
        let expected_upper = u_ik + u_kj;
        assert!(
            (u13 - expected_upper).abs() < 1e-6,
            "expected allene center near-linear (180 deg): got 1-3 upper={u13}, expected {expected_upper} (bonds [{l_ik},{u_ik}] [{l_kj},{u_kj}])"
        );
    }

    #[test]
    fn smoothing_keeps_fused_ring_bounds_feasible() {
        let m = mol("C1CC2CCCCC2C1");
        let mut bounds = build_bounds(
            &m,
            &Geometry::for_molecule(&m, &ring_set(&m)),
            &ring_set(&m),
        )
        .unwrap();
        smooth_bounds(&mut bounds).unwrap();

        let n = bounds.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    get_lower(&bounds, i, j) <= get_upper(&bounds, i, j) + 1e-6,
                    "bounds inconsistent for pair ({i}, {j})"
                );
            }
        }
    }

    /// Caffeine's pairs with a scattered, partly collapsed structure: some
    /// pairs too close, some too far, some inside their window, and a count
    /// that leaves a remainder after the four-lane loop.
    fn scattered_pairs() -> (Vec<Pair>, Vec<crate::vector::Vector4D>) {
        let m = mol("Cn1cnc2c1c(=O)n(C)c(=O)n2C");
        let mut bounds = build_bounds(
            &m,
            &Geometry::for_molecule(&m, &ring_set(&m)),
            &ring_set(&m),
        )
        .unwrap();
        smooth_bounds(&mut bounds).unwrap();
        let mut pairs = Pairs::collect(&bounds).all().to_vec();
        if pairs.len() % LANES == 0 {
            pairs.pop();
        }
        let points = (0..bounds.nrows())
            .map(|atom| {
                let t = atom as f64;
                crate::vector::Vector4D::new(
                    (1.3 * t).sin() * 2.0,
                    (0.7 * t).cos() * 1.5,
                    0.2 * t,
                    (2.1 * t).sin() * 0.5,
                )
            })
            .collect();
        (pairs, points)
    }

    #[test]
    fn the_vectorised_pair_violations_agree_with_the_scalar_ones() {
        let (pairs, points) = scattered_pairs();
        let flat: Vec<crate::vector::Vector3D> = points.iter().map(|point| point.xyz()).collect();

        let mut gradient = vec![crate::vector::Vector3D::default(); flat.len()];
        let energy = pair_violations(&pairs, &flat, Some(&mut gradient));
        let mut expected_gradient = vec![crate::vector::Vector3D::default(); flat.len()];
        let mut expected = 0.0;
        for pair in &pairs {
            let (violation, derivative) = pair_violation(pair, &flat);
            expected += violation;
            expected_gradient[pair.i as usize] += derivative;
            expected_gradient[pair.j as usize] += -derivative;
        }
        assert!(expected > 1.0, "the structure should violate its bounds");
        assert_eq!(energy.to_bits(), expected.to_bits());
        assert_eq!(
            energy.to_bits(),
            pair_violations(&pairs, &flat, None).to_bits()
        );
        // Each pair's step is the scalar one exactly, and the steps are added
        // in the same order.
        for (one, other) in gradient.iter().zip(&expected_gradient) {
            for (a, b) in [(one.x, other.x), (one.y, other.y), (one.z, other.z)] {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    #[test]
    fn every_lanes_backend_agrees_to_the_bit() {
        use crate::molecule::forcefield::lanes::{LanesOp, on_every_lanes};
        use crate::vector::Vector4D;
        #[derive(Clone)]
        struct Run<'a>(&'a [Pair], &'a [Vector4D]);
        impl LanesOp for Run<'_> {
            type Output = Vec<u64>;
            fn run<L: Lanes>(self, simd: L) -> Vec<u64> {
                let mut gradient = vec![Vector4D::default(); self.1.len()];
                let energy = pair_violations_in(simd, self.0, self.1, Some(&mut gradient));
                std::iter::once(energy)
                    .chain(gradient.iter().flat_map(|g| [g.x, g.y, g.z, g.w]))
                    .map(f64::to_bits)
                    .collect()
            }
        }
        let (pairs, points) = scattered_pairs();
        let [portable, scalar, native] = on_every_lanes(Run(&pairs, &points));
        assert_eq!(portable, scalar);
        assert_eq!(portable, native);
    }
}
