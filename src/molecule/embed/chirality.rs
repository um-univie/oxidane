//! Stereo-consistency terms for refinement (Stage 2): tetrahedral (`TH`)
//! centres and cis/trans (`/`,`\`) double bonds. Both are bundled into a
//! single [`StereoConstraints`] so [`super::refine::refine`] only threads one
//! extra parameter, and so a future stereo class (allene, square-planar, ...)
//! only needs a new field here, not another signature change downstream.

use crate::molecule::{
    base::MoleculeCore,
    bond::BondOrder,
    chirality::ChiralClass,
};
use crate::vector::Vector3D;

/// Bundles every stereo constraint extracted from a molecule's stored
/// [`ChiralClass`] tags, ready to be enforced during refinement.
#[derive(Default)]
pub(super) struct StereoConstraints {
    pub tetrahedral: Vec<StereoCenter>,
    pub cis_trans: Vec<CisTransBond>,
    pub allenes: Vec<AllenCenter>,
}

impl StereoConstraints {
    pub(super) fn collect<T: MoleculeCore>(mol: &T) -> Self {
        StereoConstraints {
            tetrahedral: collect_tetrahedral_centers(mol),
            cis_trans: collect_cis_trans_bonds(mol),
            allenes: collect_allene_centers(mol),
        }
    }

    /// Whether there is any stereochemistry to hold on to at all.
    pub(super) fn is_empty(&self) -> bool {
        self.tetrahedral.is_empty() && self.cis_trans.is_empty() && self.allenes.is_empty()
    }

    pub(super) fn count_mismatches(&self, positions: &[Vector3D]) -> usize {
        count_mismatches(positions, &self.tetrahedral)
            + count_cis_trans_mismatches(positions, &self.cis_trans)
            + count_allene_mismatches(positions, &self.allenes)
    }

    pub(super) fn energy_and_gradient(&self, positions: &[Vector3D], grad: &mut [Vector3D]) -> f64 {
        self.energy_and_gradient_scaled(positions, grad, 1.0)
    }

    /// The same terms with every margin multiplied by `margin_scale`.
    ///
    /// Distance geometry wants a margin that actively pushes a centre well
    /// clear of its plane, because nothing else there knows about
    /// chirality. A force field does know what the geometry should be, and
    /// wants the opposite: a term that is exactly zero wherever the centre
    /// is unambiguous and speaks only as one approaches its plane. A
    /// correct tetrahedral centre's signed volume is about 0.92 Angstrom
    /// cubed at these bond lengths, just under the margin distance geometry
    /// uses, so at full scale the restraint leans on every centre in the
    /// molecule and the minimisation stalls 40 kcal/mol short.
    pub(super) fn energy_and_gradient_scaled(
        &self,
        positions: &[Vector3D],
        grad: &mut [Vector3D],
        margin_scale: f64,
    ) -> f64 {
        chirality_energy_and_gradient(positions, &self.tetrahedral, grad, margin_scale)
            + cis_trans_energy_and_gradient(positions, &self.cis_trans, grad, margin_scale)
            + allene_energy_and_gradient(positions, &self.allenes, grad, margin_scale)
    }

    /// Returns `true` if any constraint was force-flipped (caller should
    /// follow up with a short [`super::refine::refine`] relaxation pass).
    pub(super) fn repair(&self, positions: &mut [Vector3D]) -> bool {
        let a = repair_mismatches(positions, &self.tetrahedral);
        let b = repair_cis_trans_mismatches(positions, &self.cis_trans);
        let c = repair_allene_mismatches(positions, &self.allenes);
        a || b || c
    }
}

// ---------------------------------------------------------------------
// Tetrahedral (TH) centres
// ---------------------------------------------------------------------

/// A tetrahedral centre, as the handedness of its four neighbours.
///
/// With `n0..n3` the centre's bonded neighbours sorted by ascending atom
/// index, the measure is `V = (p1 - p0) . ((p2 - p0) x (p3 - p0))`: the
/// signed volume of the tetrahedron *they* make, which does not mention
/// where the centre atom sits. It is negative for `TH(1)` (`@`) and
/// positive for `TH(2)` (`@@`).
///
/// # Why not three vectors from the centre
///
/// The obvious measure is `v0 . (v1 x v2)` with `v_k = p_k - centre`, and
/// that is what this was. The two agree wherever the centre sits inside its
/// neighbours: writing the centre in barycentric coordinates gives
/// `v0 . (v1 x v2) = -lambda3 * V` with `lambda3` the weight of the
/// neighbour left out, so they differ by a positive factor and a sign --
/// hence `TH(1)` mapping to a *negative* `V` here where it mapped to a
/// positive triple product before.
///
/// They stop agreeing exactly where it matters. The three-vector form
/// leaves out one neighbour entirely and measures the other three against
/// the centre, so it reports a confident handedness for a centre that has
/// been squashed flat, and it swings with the centre atom's own position
/// rather than with the arrangement around it. That is not academic: on a
/// strained cholesterol the two ring-fusion quaternary carbons read as
/// correct by this measure, at a tenth of a healthy magnitude, while RDKit
/// read the molecule as the wrong isomer -- the two codes disagreeing for a
/// week of this module's history. `V` is the volume of the actual
/// tetrahedron; it goes to zero exactly when the four neighbours become
/// coplanar, which is exactly when handedness stops being defined.
///
/// The convention is otherwise unchanged and still cross-checked against
/// RDKit via `tests/embed_oracle.py`, this crate's documented `ChiralClass`
/// (`TH(u8), // @ 1..=2  1 = anti-clockwise, 2 = clockwise`), and the
/// SMILES parser (`crate::io::smiles::assign_proper_chirality`).
pub(super) struct StereoCenter {
    pub centre: usize,
    pub n0: usize,
    pub n1: usize,
    pub n2: usize,
    pub n3: usize,
    pub target_sign: f64,
}

/// Volume at which the chirality penalty saturates, in cubic Angstroms.
///
/// A healthy tetrahedral centre at equilibrium bond lengths makes a
/// tetrahedron of about 11 -- the neighbours are 2.4 to 2.5 A apart -- so
/// four is comfortably inside the flat region and still far enough from
/// zero to catch a centre on its way through coplanarity. It is the same
/// fraction as a margin of one on the three-vector measure (smaller by the
/// barycentric weight of the neighbour it ignores, about a quarter). A
/// fixed constant is a simplification of a per-centre value derived from
/// bond-length bounds.
const CHIRALITY_MARGIN: f64 = 4.0;

/// Collects every atom whose stored [`ChiralClass`] is `TH(1)` or `TH(2)`
/// and that has exactly 4 bonded neighbors (guaranteed after
/// `add_hydrogens()` for a genuine tetrahedral stereocenter).
fn collect_tetrahedral_centers<T: MoleculeCore>(mol: &T) -> Vec<StereoCenter> {
    let Some(chiral_classes) = mol.chiral_classes() else {
        return Vec::new();
    };

    let mut centers = Vec::new();
    for (centre, &class) in chiral_classes.iter().enumerate() {
        // Negated against the three-vector measure this replaced, which is
        // the `-lambda3` of the identity in [`StereoCenter`]'s comment.
        let target_sign = match class {
            ChiralClass::TH(1) => -1.0,
            ChiralClass::TH(2) => 1.0,
            _ => continue,
        };
        let mut neighbors: Vec<usize> =
            mol.atom_bonds()[centre].iter().map(|b| b.target()).collect();
        if neighbors.len() != 4 {
            continue;
        }
        neighbors.sort_unstable();
        centers.push(StereoCenter {
            centre,
            n0: neighbors[0],
            n1: neighbors[1],
            n2: neighbors[2],
            n3: neighbors[3],
            target_sign,
        });
    }
    centers
}

/// The signed volume of the tetrahedron the four neighbours make, and the
/// three edges from `n0` that define it.
fn signed_volume(positions: &[Vector3D], c: &StereoCenter) -> (f64, Vector3D, Vector3D, Vector3D) {
    let first = positions[c.n1] - positions[c.n0];
    let second = positions[c.n2] - positions[c.n0];
    let third = positions[c.n3] - positions[c.n0];
    (first.dot(&second.cross(&third)), first, second, third)
}

/// Number of stereocenters whose current geometry does not match its target
/// sign (used to rank trial embeddings ahead of raw bounds-violation energy).
fn count_mismatches(positions: &[Vector3D], centers: &[StereoCenter]) -> usize {
    centers
        .iter()
        .filter(|c| {
            let (v, ..) = signed_volume(positions, c);
            v * c.target_sign <= 0.0
        })
        .count()
}

/// Adds the chirality-consistency energy and its analytic gradient into
/// `grad`, returning the summed energy. Only active while
/// `sign * V < CHIRALITY_MARGIN`, i.e. the penalty vanishes once a centre is
/// comfortably on the correct side.
fn chirality_energy_and_gradient(
    positions: &[Vector3D],
    centers: &[StereoCenter],
    grad: &mut [Vector3D],
    margin_scale: f64,
) -> f64 {
    let margin = CHIRALITY_MARGIN * margin_scale;
    let mut energy = 0.0;
    for c in centers {
        let (v, first, second, third) = signed_volume(positions, c);
        let signed = c.target_sign * v;
        if signed >= margin {
            continue;
        }
        let deficit = margin - signed;
        energy += (deficit * deficit) / (margin * margin);

        // dE/dV = -2*target_sign*deficit / margin^2
        let de_dv = -2.0 * c.target_sign * deficit / (margin * margin);

        // V = first . (second x third), with each edge measured from `n0`:
        // dV/d(p1) = second x third, and so round. The centre atom does not
        // appear in V and so takes no force -- a flattening centre is put
        // right by moving what is around it, which is the motion that
        // actually restores the handedness.
        let d_first = second.cross(&third);
        let d_second = third.cross(&first);
        let d_third = first.cross(&second);

        grad[c.n1] += d_first * de_dv;
        grad[c.n2] += d_second * de_dv;
        grad[c.n3] += d_third * de_dv;
        grad[c.n0] += -(d_first + d_second + d_third) * de_dv;
    }
    energy
}

/// Point-reflects each still-mismatched centre's `n0` through the centre atom
/// itself (`p_n0' = 2*p_centre - p_n0`). This flips the sign of `v0`, hence of
/// `V`, while leaving the `centre`-`n0` bond length exactly unchanged; any
/// bonds from `n0` onward to its own substituents are left distorted for the
/// caller to relax with a short follow-up [`super::refine::refine`] pass.
/// A last-resort fallback for centers the energy-gradient term alone
/// couldn't flip within `max_attempts` trials - not a full ETKDG-style 4D
/// escape, so nested/adjacent stereocenters may need more than one repair
/// pass (a known "lite" limitation).
fn repair_mismatches(positions: &mut [Vector3D], centers: &[StereoCenter]) -> bool {
    let mut repaired_any = false;
    for c in centers {
        let (v, ..) = signed_volume(positions, c);
        if v * c.target_sign <= 0.0 {
            positions[c.n0] = positions[c.centre] * 2.0 - positions[c.n0];
            repaired_any = true;
        }
    }
    repaired_any
}

// ---------------------------------------------------------------------
// Cis/trans (E/Z) double bonds
// ---------------------------------------------------------------------

/// A double bond `a=b` tagged with `CisTransClockwise`/`CisTransCounterclockwise`
/// on both ends, plus each end's first non-double-bond neighbor (`sub_a0`,
/// `sub_b0`) in bond-list order - matching exactly the substituents that
/// [`crate::molecule::chirality::MoleculeChirality::convert_directional_bonds_to_chiral`]
/// used to derive the tags from `/`/`\` notation in the first place. Per that
/// function's invariant, `class(a) == class(b)` means `sub_a0`/`sub_b0` are
/// *cis* (same side); `class(a) != class(b)` means *trans*.
pub(super) struct CisTransBond {
    pub a: usize,
    pub b: usize,
    pub sub_a0: usize,
    pub sub_b0: usize,
    /// `+1.0` for cis (same side), `-1.0` for trans (opposite side).
    pub target_sign: f64,
}

/// Same scale reasoning as [`CHIRALITY_MARGIN`]: the cross-product scalar
/// below has units of length^4 and is on the order of a few A^4 for a
/// planar sp2 double bond at equilibrium bond lengths.
const CIS_TRANS_MARGIN: f64 = 1.0;

fn first_non_double_neighbor<T: MoleculeCore>(mol: &T, atom: usize, exclude: usize) -> Option<usize> {
    mol.atom_bonds()[atom]
        .iter()
        .find(|b| b.bond_order() != BondOrder::Double && b.target() != exclude)
        .map(|b| b.target())
}

/// Which substituents a stereo double bond's tags are stated relative to,
/// and whether the tags put that pair on the same side. This is exactly what
/// [`collect_cis_trans_bonds`] derives per bond, exposed on its own so
/// [`super::forcefield`] can place the phase of the torsions that flatten
/// the bond without re-deriving the convention.
pub(super) struct StereoDoubleBondReference {
    pub reference_b: usize,
    pub reference_c: usize,
    /// `+1.0` if the two reference substituents are cis, `-1.0` if trans.
    pub target_sign: f64,
}

pub(super) fn stereo_double_bond_reference<T: MoleculeCore>(
    mol: &T,
    b: usize,
    c: usize,
) -> Option<StereoDoubleBondReference> {
    let chiral_classes = mol.chiral_classes()?;
    let reference_b = first_non_double_neighbor(mol, b, c)?;
    let reference_c = first_non_double_neighbor(mol, c, b)?;
    let target_sign = if chiral_classes[b] == chiral_classes[c] { 1.0 } else { -1.0 };
    Some(StereoDoubleBondReference { reference_b, reference_c, target_sign })
}

fn collect_cis_trans_bonds<T: MoleculeCore>(mol: &T) -> Vec<CisTransBond> {
    let Some(chiral_classes) = mol.chiral_classes() else {
        return Vec::new();
    };
    let is_cis_trans_tag = |class: ChiralClass| {
        matches!(class, ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise)
    };

    let mut bonds = Vec::new();
    for a in 0..mol.len() {
        if !is_cis_trans_tag(chiral_classes[a]) {
            continue;
        }
        let Some(b) = mol.atom_bonds()[a]
            .iter()
            .find(|bd| bd.bond_order() == BondOrder::Double)
            .map(|bd| bd.target())
        else {
            continue;
        };
        if b <= a || !is_cis_trans_tag(chiral_classes[b]) {
            continue; // handle each double bond once, from its lower-indexed end
        }
        let (Some(sub_a0), Some(sub_b0)) = (
            first_non_double_neighbor(mol, a, b),
            first_non_double_neighbor(mol, b, a),
        ) else {
            continue;
        };
        let target_sign = if chiral_classes[a] == chiral_classes[b] { 1.0 } else { -1.0 };
        bonds.push(CisTransBond { a, b, sub_a0, sub_b0, target_sign });
    }
    bonds
}

/// Cross-product "torsion sign" of `sub_a0-a=b-sub_b0`: with `b1 = a-sub_a0`,
/// `b2 = b-a`, `b3 = sub_b0-b`, `S = (b1 x b2).(b2 x b3)`. `S>0` corresponds to
/// a dihedral near 0 deg (`sub_a0`/`sub_b0` eclipsed, i.e. cis); `S<0` to a
/// dihedral near 180 deg (trans). Expanded via Lagrange's identity
/// `(u x v).(w x z) = (u.w)(v.z) - (u.z)(v.w)` into dot products only, which
/// keeps the analytic gradient a low-degree polynomial (no cross products of
/// cross products to differentiate).
fn cis_trans_scalar(
    positions: &[Vector3D],
    ct: &CisTransBond,
) -> (f64, Vector3D, Vector3D, Vector3D, Vector3D) {
    let p = positions[ct.sub_a0];
    let a = positions[ct.a];
    let b = positions[ct.b];
    let q = positions[ct.sub_b0];
    let b1 = a - p;
    let b2 = b - a;
    let b3 = q - b;

    let dot_12 = b1.dot(&b2);
    let dot_23 = b2.dot(&b3);
    let dot_13 = b1.dot(&b3);
    let dot_22 = b2.dot(&b2);

    let s = dot_12 * dot_23 - dot_13 * dot_22;

    // Gradients derived by hand from S = (b1.b2)(b2.b3) - (b1.b3)|b2|^2 via
    // the chain rule through b1=a-p, b2=b-a, b3=q-b; cross-checked by summing
    // the four gradients to zero (S depends only on relative positions).
    let d_p = b2 * (-dot_23) + b3 * dot_22;
    let d_a = b2 * (dot_23 + 2.0 * dot_13) - b1 * dot_23 - b3 * (dot_12 + dot_22);
    let d_b = b1 * (dot_23 + dot_22) + b3 * dot_12 - b2 * (dot_12 + 2.0 * dot_13);
    let d_q = b2 * dot_12 - b1 * dot_22;

    (s, d_p, d_a, d_b, d_q)
}

fn count_cis_trans_mismatches(positions: &[Vector3D], bonds: &[CisTransBond]) -> usize {
    bonds
        .iter()
        .filter(|ct| {
            let (s, ..) = cis_trans_scalar(positions, ct);
            s * ct.target_sign <= 0.0
        })
        .count()
}

fn cis_trans_energy_and_gradient(
    positions: &[Vector3D],
    bonds: &[CisTransBond],
    grad: &mut [Vector3D],
    margin_scale: f64,
) -> f64 {
    let margin = CIS_TRANS_MARGIN * margin_scale;
    let mut energy = 0.0;
    for ct in bonds {
        let (s, d_p, d_a, d_b, d_q) = cis_trans_scalar(positions, ct);
        let signed = ct.target_sign * s;
        if signed >= margin {
            continue;
        }
        let deficit = margin - signed;
        energy += (deficit * deficit) / (margin * margin);

        let de_ds = -2.0 * ct.target_sign * deficit / (margin * margin);
        grad[ct.sub_a0] += d_p * de_ds;
        grad[ct.a] += d_a * de_ds;
        grad[ct.b] += d_b * de_ds;
        grad[ct.sub_b0] += d_q * de_ds;
    }
    energy
}

/// 180-degree rotation of `sub_a0` about the `a-b` axis: since both `a` and
/// `b` lie on the rotation axis, this flips cis/trans while leaving the
/// `a-sub_a0` bond length *and* the `b-sub_a0` distance exactly unchanged
/// (only the component of `sub_a0`'s position perpendicular to the axis is
/// negated). Any bonds from `sub_a0` onward to its own substituents are left
/// for the caller's follow-up [`super::refine::refine`] pass to relax.
fn repair_cis_trans_mismatches(positions: &mut [Vector3D], bonds: &[CisTransBond]) -> bool {
    let mut repaired_any = false;
    for ct in bonds {
        let (s, ..) = cis_trans_scalar(positions, ct);
        if s * ct.target_sign <= 0.0 {
            let a = positions[ct.a];
            let b = positions[ct.b];
            let axis = (b - a).normalize();
            let x = positions[ct.sub_a0];
            let foot = a + axis * (x - a).dot(&axis);
            positions[ct.sub_a0] = foot * 2.0 - x;
            repaired_any = true;
        }
    }
    repaired_any
}

// ---------------------------------------------------------------------
// Allene (AL) centres
// ---------------------------------------------------------------------

/// An allene-type (cumulated double bond, `X=C=Y`) stereocentre: the tagged
/// carbon's two double-bond neighbors `l`/`r` (index-sorted, matching the
/// index-sorted-neighbor convention used for `TH` and consistent with how
/// the SMILES parser's generic tag-canonicalization treats *any* `@`-tagged
/// atom's own bonded-neighbor list), and each end's first non-double-bond
/// substituent (`sub_l`, `sub_r`) in bond-list order.
///
/// **Absolute handedness is unverified** - unlike `TH` and the square-planar
/// bounds in [`super::bounds`], this mapping could *not* be cross-checked
/// against RDKit: RDKit's `AssignStereochemistryFrom3D` /
/// `AssignAtomChiralTagsFromStructure` (this crate's `rdkit` version,
/// 2025.09.1) do not perceive allene/extended-tetrahedral stereo from either
/// 3D coordinates or `/`,`\`-decorated cumulated-diene SMILES - confirmed by
/// direct experimentation, not merely absence of a passing test - and the
/// in-repo oracle
/// ([`crate::molecule::chirality::ChiralClassifier::determine_allene_chirality`])
/// is itself non-functional for real substituted allenes (see its doc
/// comment). With `v0 = sub_l - l`, `v1 = r - l`, `v2 = sub_r - l`,
/// `V = v0.(v1 x v2)` is defined here to be positive for `AL(2)` and
/// negative for `AL(1)` - internally self-consistent (embedding an `AL(1)`
/// and an `AL(2)` tag on the same skeleton reliably produces mirror-image
/// geometry, see the unit tests) and follows OpenSMILES's
/// extended-tetrahedral definition, but is *not* independently confirmed
/// the way `TH`/`SP`/cis-trans were.
pub(super) struct AllenCenter {
    pub sub_l: usize,
    pub l: usize,
    pub r: usize,
    pub sub_r: usize,
    pub target_sign: f64,
}

/// Collects every carbon whose stored [`ChiralClass`] is `AL(1)` or `AL(2)`:
/// a cumulated-double-bond centre with exactly 2 double-bond neighbors, each
/// having exactly one further (non-double-bond) substituent after
/// `add_hydrogens()`.
fn collect_allene_centers<T: MoleculeCore>(mol: &T) -> Vec<AllenCenter> {
    let Some(chiral_classes) = mol.chiral_classes() else {
        return Vec::new();
    };

    let mut centers = Vec::new();
    for (centre, &class) in chiral_classes.iter().enumerate() {
        let target_sign = match class {
            ChiralClass::AL(2) => 1.0,
            ChiralClass::AL(1) => -1.0,
            _ => continue,
        };
        let mut ends: Vec<usize> = mol.atom_bonds()[centre]
            .iter()
            .filter(|b| b.bond_order() == BondOrder::Double)
            .map(|b| b.target())
            .collect();
        if ends.len() != 2 {
            continue;
        }
        ends.sort_unstable();
        let (l, r) = (ends[0], ends[1]);
        let (Some(sub_l), Some(sub_r)) =
            (first_non_double_neighbor(mol, l, centre), first_non_double_neighbor(mol, r, centre))
        else {
            continue;
        };
        centers.push(AllenCenter { sub_l, l, r, sub_r, target_sign });
    }
    centers
}

fn allene_signed_volume(positions: &[Vector3D], c: &AllenCenter) -> (f64, Vector3D, Vector3D, Vector3D) {
    let v0 = positions[c.sub_l] - positions[c.l];
    let v1 = positions[c.r] - positions[c.l];
    let v2 = positions[c.sub_r] - positions[c.l];
    (v0.dot(&v1.cross(&v2)), v0, v1, v2)
}

fn count_allene_mismatches(positions: &[Vector3D], centers: &[AllenCenter]) -> usize {
    centers
        .iter()
        .filter(|c| {
            let (v, ..) = allene_signed_volume(positions, c);
            v * c.target_sign <= 0.0
        })
        .count()
}

/// Same margin-penalty shape as [`chirality_energy_and_gradient`], applied to
/// the allene volume `V` instead of the tetrahedral one.
fn allene_energy_and_gradient(
    positions: &[Vector3D],
    centers: &[AllenCenter],
    grad: &mut [Vector3D],
    margin_scale: f64,
) -> f64 {
    let margin = CHIRALITY_MARGIN * margin_scale;
    let mut energy = 0.0;
    for c in centers {
        let (v, v0, v1, v2) = allene_signed_volume(positions, c);
        let signed = c.target_sign * v;
        if signed >= margin {
            continue;
        }
        let deficit = margin - signed;
        energy += (deficit * deficit) / (margin * margin);

        let de_dv = -2.0 * c.target_sign * deficit / (margin * margin);
        let dv_dv0 = v1.cross(&v2);
        let dv_dv1 = v2.cross(&v0);
        let dv_dv2 = v0.cross(&v1);

        grad[c.sub_l] += dv_dv0 * de_dv;
        grad[c.r] += dv_dv1 * de_dv;
        grad[c.sub_r] += dv_dv2 * de_dv;
        grad[c.l] += -(dv_dv0 + dv_dv1 + dv_dv2) * de_dv;
    }
    energy
}

/// Point-reflects `sub_l` through `l`, mirroring [`repair_mismatches`]'s
/// approach: preserves the `l`-`sub_l` bond length exactly, flips the sign
/// of `V`, and leaves any further relaxation to the caller's follow-up
/// [`super::refine::refine`] pass.
fn repair_allene_mismatches(positions: &mut [Vector3D], centers: &[AllenCenter]) -> bool {
    let mut repaired_any = false;
    for c in centers {
        let (v, ..) = allene_signed_volume(positions, c);
        if v * c.target_sign <= 0.0 {
            positions[c.sub_l] = positions[c.l] * 2.0 - positions[c.sub_l];
            repaired_any = true;
        }
    }
    repaired_any
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::chirality::ChiralClassifier;
    use crate::prelude::*;

    #[test]
    fn signed_volume_matches_oracle_convention() {
        let mut mol = MolecularGraph::from_smiles_single("[C@](F)(Cl)(Br)I").unwrap();
        mol.add_hydrogens();
        let centers = collect_tetrahedral_centers(&mol);
        assert_eq!(centers.len(), 1, "expected exactly one tetrahedral centre");

        let embedded = mol.embed_3d().unwrap();
        let positions = embedded.positions.clone();
        let (v, ..) = signed_volume(&positions, &centers[0]);
        assert!(
            v * centers[0].target_sign > 0.0,
            "embedded geometry should match the stored TH tag, got signed volume {v} vs target sign {}",
            centers[0].target_sign
        );

        let mut oracle = embedded;
        oracle.chiral_classes = None;
        oracle.identify_chiral_classes();
        // Negative volume is `TH(1)`; see [`StereoCenter`].
        assert_eq!(
            oracle.chiral_classes().unwrap()[centers[0].centre],
            ChiralClass::TH(if centers[0].target_sign < 0.0 { 1 } else { 2 })
        );
    }

    /// The four-point volume and the three-vector triple product it
    /// replaced differ by `-lambda3`, the barycentric weight of the
    /// neighbour the triple product leaves out -- which is why the target
    /// signs are negated against the triple product's, and why the margin is
    /// four rather than one.
    ///
    /// The relation is what the sign convention rests on, so it is tested.
    #[test]
    fn the_four_point_volume_is_the_triple_product_scaled_and_flipped() {
        let mut mol = MolecularGraph::from_smiles_single("[C@](F)(Cl)(Br)I").unwrap();
        mol.add_hydrogens();
        let centre = &collect_tetrahedral_centers(&mol)[0];
        let positions = mol.embed_3d().unwrap().positions;

        let (four_point, ..) = signed_volume(&positions, centre);
        let from_centre = |atom: usize| positions[atom] - positions[centre.centre];
        let triple_product = from_centre(centre.n0)
            .dot(&from_centre(centre.n1).cross(&from_centre(centre.n2)));

        assert!(
            triple_product * four_point < 0.0,
            "the two measures should disagree in sign, got {triple_product} and {four_point}"
        );
        // `lambda3` is a barycentric weight, so strictly between 0 and 1,
        // and about a quarter at a centre sitting where it should.
        let lambda = -triple_product / four_point;
        assert!(
            (0.15..0.35).contains(&lambda),
            "expected the excluded neighbour's weight near a quarter, got {lambda}"
        );
    }

    #[test]
    fn cis_trans_scalar_sign_matches_known_geometry() {
        // Planar trans (E) arrangement around a=b on the x axis:
        // sub_a0 at +y off `a`, sub_b0 at -y off `b` -> opposite sides -> S<0.
        let bond = CisTransBond { a: 1, b: 2, sub_a0: 0, sub_b0: 3, target_sign: -1.0 };
        let positions = vec![
            Vector3D::new(-1.0, 1.0, 0.0),  // sub_a0
            Vector3D::new(0.0, 0.0, 0.0),   // a
            Vector3D::new(1.3, 0.0, 0.0),   // b
            Vector3D::new(2.3, -1.0, 0.0),  // sub_b0
        ];
        let (s, ..) = cis_trans_scalar(&positions, &bond);
        assert!(s < 0.0, "expected trans (opposite-side) geometry to give S<0, got {s}");

        // Mirror sub_b0 to the same side as sub_a0 -> cis -> S>0.
        let mut cis_positions = positions.clone();
        cis_positions[3] = Vector3D::new(2.3, 1.0, 0.0);
        let (s_cis, ..) = cis_trans_scalar(&cis_positions, &bond);
        assert!(s_cis > 0.0, "expected cis (same-side) geometry to give S>0, got {s_cis}");
    }

    /// Absolute AL(1)/AL(2) handedness is unverified (see [`AllenCenter`]'s
    /// doc comment) but the two must still be reliable *mirror images* of
    /// each other for the same skeleton - this is what's actually checkable
    /// without an external oracle.
    #[test]
    fn allene_enantiomers_embed_as_mirror_images() {
        let mol1 = MolecularGraph::from_smiles_single("FC=[C@]=CF").unwrap();
        let mol2 = MolecularGraph::from_smiles_single("FC=[C@@]=CF").unwrap();
        let centers1 = collect_allene_centers(&mol1);
        let centers2 = collect_allene_centers(&mol2);
        assert_eq!(centers1.len(), 1);
        assert_eq!(centers2.len(), 1);
        assert_ne!(centers1[0].target_sign, centers2[0].target_sign);

        let embedded1 = mol1.embed_3d().unwrap();
        let embedded2 = mol2.embed_3d().unwrap();
        let (v1, ..) = allene_signed_volume(&embedded1.positions, &centers1[0]);
        let (v2, ..) = allene_signed_volume(&embedded2.positions, &centers2[0]);
        assert!(v1 * centers1[0].target_sign > 0.0, "AL(1)/AL(2) input should embed matching its own target sign");
        assert!(v2 * centers2[0].target_sign > 0.0, "AL(1)/AL(2) input should embed matching its own target sign");
        assert!(
            v1.signum() != v2.signum(),
            "the two allene tags should embed as geometric mirror images, got v1={v1} v2={v2}"
        );
    }
}
