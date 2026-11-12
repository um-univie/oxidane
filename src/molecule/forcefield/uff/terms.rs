//! The UFF energy expressions and their analytic gradients.
//!
//! Rappe, Casewit, Colwell, Goddard & Skiff, *J. Am. Chem. Soc.* **1992**,
//! 114, 10024-10035. The forms and the parameter combination rules are the
//! paper's, read the way every implementation reads them: Towhee, RDKit
//! (`Code/ForceField/UFF`, BSD 3-Clause, see `THIRD_PARTY.md`) and Open
//! Babel agree on all of it, including the periodic angle form's switch on
//! the centre's geometry rather than on its equilibrium angle. Where this
//! crate's UFF differs from RDKit's -- the typing of conjugated carbonyls,
//! and the amide bond order RDKit leaves unwired -- the module comment on
//! [`super::uff`] says where and why, and `tests/uff_oracle.py` measures
//! it.

/// Bond-length threshold in Å for a well-defined angle gradient.
const MIN_BOND_LENGTH: f64 = 1e-8;
/// Dimensionless sine/cosine threshold near singular angular geometries.
const MIN_ANGULAR_COMPONENT: f64 = 1e-8;
/// Squared plane-normal norm below which inversion has no defined plane (Å⁴).
const MIN_INVERSION_NORMAL_SQUARED: f64 = 1e-16;
/// Squared out-of-plane bond length below which inversion is undefined (Å²).
const MIN_INVERSION_BOND_SQUARED: f64 = 1e-16;

use super::params::UffType;
use crate::molecule::forcefield::lanes::{LANES, Lanes, horizontal_sum};
use crate::vector::Vector3D;

use chemistry_consts::uff::{
    BOND_ORDER_LAMBDA, COULOMB_KCAL_ANGSTROM_PER_MOL as COULOMB_PREFACTOR,
};

/// Below this separation the Lennard-Jones energy is infinite. Returning
/// zero would create an artificial minimum at an exact atomic overlap.
const MIN_VDW_SEPARATION: f64 = 1e-8;

/// A bond, with the equilibrium length and stiffness UFF derives for it.
pub(super) struct Bond {
    pub i: usize,
    pub j: usize,
    pub rest_length: f64,
    pub force_constant: f64,
}

/// An angle `i-k-j` about its central atom `k`.
///
/// `fold` selects the periodic form UFF uses for the special equilibrium
/// angles -- linear, trigonal planar, square planar and octahedral all have
/// a cosine expansion that is exact rather than fitted. `None` means the
/// general three-term expansion, whose coefficients are precomputed.
pub(super) struct Angle {
    pub i: usize,
    pub k: usize,
    pub j: usize,
    pub force_constant: f64,
    pub fold: Option<u32>,
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
}

/// A torsion about the bond `b-c`.
pub(super) struct Torsion {
    pub i: usize,
    pub b: usize,
    pub c: usize,
    pub j: usize,
    pub force_constant: f64,
    pub fold: u32,
    /// `cos(n * phi0)`, which is `+1` or `-1` for every case UFF defines.
    pub phase: f64,
}

/// An out-of-plane term at a three-coordinate centre `k`: the bond `k-l`
/// measured against the plane through `k`, `i` and `j`.
///
/// UFF writes it in the cosine of the Wilson angle `omega` between that bond
/// and that plane, with the same three-coefficient expansion as the angle
/// term: `K * (c0 + c1 cos(omega) + c2 cos(2 omega))`.
pub(super) struct Inversion {
    pub i: usize,
    pub k: usize,
    pub j: usize,
    pub l: usize,
    pub force_constant: f64,
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
}

/// A non-bonded pair, with its Lennard-Jones minimum and well depth.
#[derive(Clone, Copy)]
pub(super) struct VanDerWaals {
    pub i: usize,
    pub j: usize,
    pub distance: f64,
    pub well_depth: f64,
}

/// Every nonbonded pair, one array per field, for the vectorised loop --
/// the layout MMFF's `NonBondedPairs` uses, for the same reason. The minimum
/// is stored squared: the Lennard-Jones form needs only `(r0 / r)^2`, so the
/// loop takes no square root at all.
#[derive(Default)]
pub(super) struct VanDerWaalsPairs {
    i: Vec<u32>,
    j: Vec<u32>,
    minimum_squared: Vec<f64>,
    well_depth: Vec<f64>,
}

impl VanDerWaalsPairs {
    pub(super) fn push(&mut self, pair: VanDerWaals) {
        self.i.push(pair.i as u32);
        self.j.push(pair.j as u32);
        self.minimum_squared.push(pair.distance * pair.distance);
        self.well_depth.push(pair.well_depth);
    }

    pub(super) fn len(&self) -> usize {
        self.i.len()
    }
}

/// UFF's equilibrium bond length: the two valence radii, shortened for bond
/// order and again for the electronegativity difference between the ends.
pub(super) fn rest_length(first: &UffType, second: &UffType, bond_order: f64) -> f64 {
    let (ri, rj) = (first.bond_radius, second.bond_radius);
    let bond_order_term = -BOND_ORDER_LAMBDA * (ri + rj) * bond_order.ln();

    // O'Keeffe and Brese: the more the two ends differ in electronegativity,
    // the shorter the bond than the radii alone predict.
    let (xi, xj) = (first.electronegativity, second.electronegativity);
    let difference = xi.sqrt() - xj.sqrt();
    let electronegativity_term = ri * rj * difference * difference / (xi * ri + xj * rj);

    ri + rj + bond_order_term - electronegativity_term
}

pub(super) fn bond_force_constant(rest_length: f64, first: &UffType, second: &UffType) -> f64 {
    2.0 * COULOMB_PREFACTOR * first.effective_charge * second.effective_charge / rest_length.powi(3)
}

/// UFF's angle force constant, from the two bond lengths and the effective
/// charges of the outer atoms.
pub(super) fn angle_force_constant(
    theta0: f64,
    r12: f64,
    r23: f64,
    first: &UffType,
    third: &UffType,
) -> f64 {
    let cos_theta0 = theta0.cos();
    let r13_sq = r12 * r12 + r23 * r23 - 2.0 * r12 * r23 * cos_theta0;
    let r13 = r13_sq.sqrt();
    let beta = 2.0 * COULOMB_PREFACTOR / (r12 * r23);
    let prefactor = beta * first.effective_charge * third.effective_charge / r13.powi(5);
    let inner = 3.0 * r12 * r23 * (1.0 - cos_theta0 * cos_theta0) - r13_sq * cos_theta0;
    prefactor * r12 * r23 * inner
}

/// `cos(n * x)` from `cos(x)`, by the Chebyshev recurrence, for the folds
/// UFF uses.
#[inline(always)]
fn cos_multiple(fold: u32, cos_x: f64) -> f64 {
    let sin_sq = 1.0 - cos_x * cos_x;
    match fold {
        1 => cos_x,
        2 => 2.0 * cos_x * cos_x - 1.0,
        3 => cos_x * (cos_x * cos_x - 3.0 * sin_sq),
        4 => {
            let c2 = cos_x * cos_x;
            c2 * c2 - 6.0 * c2 * sin_sq + sin_sq * sin_sq
        }
        6 => 1.0 + sin_sq * (-32.0 * sin_sq * sin_sq + 48.0 * sin_sq - 18.0),
        _ => (fold as f64 * cos_x.acos()).cos(),
    }
}

/// `d/dx cos(n x)` expressed as `-n * sin(n x)`.
#[inline(always)]
fn d_cos_multiple(fold: u32, x: f64) -> f64 {
    -(fold as f64) * (fold as f64 * x).sin()
}

#[inline(always)]
pub(super) fn bond_energy(positions: &[Vector3D], bond: &Bond) -> f64 {
    let separation = positions[bond.i].distance(&positions[bond.j]) - bond.rest_length;
    0.5 * bond.force_constant * separation * separation
}

#[inline(always)]
pub(super) fn bond_energy_and_gradient(
    positions: &[Vector3D],
    bond: &Bond,
    gradient: &mut [Vector3D],
) -> f64 {
    let offset = positions[bond.i] - positions[bond.j];
    let distance = offset.length();
    if distance < MIN_VDW_SEPARATION {
        return bond_energy(positions, bond);
    }
    let stretch = distance - bond.rest_length;
    let scale = bond.force_constant * stretch / distance;
    gradient[bond.i] += offset * scale;
    gradient[bond.j] += -(offset * scale);
    0.5 * bond.force_constant * stretch * stretch
}

/// `theta` and `d(theta)/dp` for the three atoms of an angle.
#[inline(always)]
fn angle_and_gradient(positions: &[Vector3D], angle: &Angle) -> Option<(f64, [Vector3D; 3])> {
    let first = positions[angle.i] - positions[angle.k];
    let second = positions[angle.j] - positions[angle.k];
    let (r1, r2) = (first.length(), second.length());
    if r1 < MIN_BOND_LENGTH || r2 < MIN_BOND_LENGTH {
        return None;
    }
    let cos_theta = (first.dot(&second) / (r1 * r2)).clamp(-1.0, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    if sin_theta < MIN_ANGULAR_COMPONENT {
        // The angular derivative has no unique direction at collinearity,
        // but the angle and its energy remain defined. Dropping the whole
        // term creates a false zero-energy minimum at 0 and 180 degrees.
        return Some((cos_theta.acos(), [Vector3D::default(); 3]));
    }

    // d(cos)/d(first) = second/(r1 r2) - cos * first/r1^2, and symmetrically;
    // the centre appears in both with a minus sign.
    let d_cos_first = second * (1.0 / (r1 * r2)) - first * (cos_theta / (r1 * r1));
    let d_cos_second = first * (1.0 / (r1 * r2)) - second * (cos_theta / (r2 * r2));
    let to_theta = -1.0 / sin_theta;
    let d_i = d_cos_first * to_theta;
    let d_j = d_cos_second * to_theta;
    Some((cos_theta.acos(), [d_i, -(d_i + d_j), d_j]))
}

#[inline(always)]
fn angle_energy_from(angle: &Angle, theta: f64) -> f64 {
    let cos_theta = theta.cos();
    match angle.fold {
        None => {
            angle.force_constant
                * (angle.c0 + angle.c1 * cos_theta + angle.c2 * cos_multiple(2, cos_theta))
        }
        Some(fold) => {
            // UFF's periodic form for the special equilibrium angles, whose
            // minimum sits where the cosine expansion is exact rather than
            // fitted.
            let term = if fold == 1 {
                -cos_theta
            } else {
                cos_multiple(fold, cos_theta)
            };
            angle.force_constant / (fold * fold) as f64 * (1.0 - term)
        }
    }
}

#[inline(always)]
pub(super) fn angle_energy(positions: &[Vector3D], angle: &Angle) -> f64 {
    match angle_and_gradient(positions, angle) {
        Some((theta, _)) => angle_energy_from(angle, theta),
        None => 0.0,
    }
}

#[inline(always)]
pub(super) fn angle_energy_and_gradient(
    positions: &[Vector3D],
    angle: &Angle,
    gradient: &mut [Vector3D],
) -> f64 {
    let Some((theta, d_theta)) = angle_and_gradient(positions, angle) else {
        return 0.0;
    };
    let (sin_theta, cos_theta) = theta.sin_cos();
    let d_energy = match angle.fold {
        None => {
            angle.force_constant * (-angle.c1 * sin_theta - 2.0 * angle.c2 * (2.0 * theta).sin())
        }
        Some(fold) => {
            let d_term = if fold == 1 {
                sin_theta
            } else {
                d_cos_multiple(fold, theta)
            };
            -angle.force_constant / (fold * fold) as f64 * d_term
        }
    };
    let _ = cos_theta;
    for (slot, atom) in [angle.i, angle.k, angle.j].into_iter().enumerate() {
        gradient[atom] += d_theta[slot] * d_energy;
    }
    angle_energy_from(angle, theta)
}

/// The dihedral `i-b-c-j` and `d(phi)/dp` for its four atoms, indexed into
/// `positions`.
#[inline(always)]
fn dihedral_and_gradient(
    positions: &[Vector3D],
    i: usize,
    b: usize,
    c: usize,
    j: usize,
) -> Option<(f64, f64, [Vector3D; 4])> {
    crate::vector::dihedral_and_gradient(positions[i], positions[b], positions[c], positions[j])
}

#[inline(always)]
pub(super) fn torsion_energy(positions: &[Vector3D], torsion: &Torsion) -> f64 {
    match crate::vector::dihedral_cosine_sine(
        positions[torsion.i],
        positions[torsion.b],
        positions[torsion.c],
        positions[torsion.j],
    ) {
        Some((cosine, _)) => {
            0.5 * torsion.force_constant
                * (1.0 - torsion.phase * cos_multiple(torsion.fold, cosine))
        }
        None => 0.0,
    }
}

#[inline(always)]
pub(super) fn torsion_energy_and_gradient(
    positions: &[Vector3D],
    torsion: &Torsion,
    gradient: &mut [Vector3D],
) -> f64 {
    let Some((cosine, sine, d_phi)) =
        dihedral_and_gradient(positions, torsion.i, torsion.b, torsion.c, torsion.j)
    else {
        return 0.0;
    };
    let (cos_fold, sin_fold) = crate::vector::multiple_angle(cosine, sine, torsion.fold as usize);
    let energy = 0.5 * torsion.force_constant * (1.0 - torsion.phase * cos_fold);
    // d/dphi of -phase * cos(n phi) is +phase * n * sin(n phi).
    let d_energy = 0.5 * torsion.force_constant * torsion.phase * torsion.fold as f64 * sin_fold;
    for (slot, atom) in [torsion.i, torsion.b, torsion.c, torsion.j]
        .into_iter()
        .enumerate()
    {
        gradient[atom] += d_phi[slot] * d_energy;
    }
    energy
}

/// The Lennard-Jones energy of one pair, from its squared separation.
/// Written as the lanes below write it, so the remainder they leave to
/// this agrees with them to the bit.
#[inline(always)]
fn lennard_jones(minimum_squared: f64, well_depth: f64, squared: f64) -> (f64, f64) {
    if !(squared > MIN_VDW_SEPARATION * MIN_VDW_SEPARATION) {
        return (f64::INFINITY, 0.0);
    }
    let inverse = 1.0 / squared;
    let ratio = minimum_squared * inverse;
    let six = ratio * ratio * ratio;
    // dE/dr = 12 * D * (r6 - r12) / distance, along offset / distance.
    let scale = 12.0 * well_depth * (six - six * six) * inverse;
    (well_depth * (six * six - 2.0 * six), scale)
}

#[cfg(test)]
pub(super) fn van_der_waals_energy(positions: &[Vector3D], pair: &VanDerWaals) -> f64 {
    let offset = positions[pair.i] - positions[pair.j];
    lennard_jones(
        pair.distance * pair.distance,
        pair.well_depth,
        offset.dot(&offset),
    )
    .0
}

#[cfg(test)]
pub(super) fn van_der_waals_energy_and_gradient(
    positions: &[Vector3D],
    pair: &VanDerWaals,
    gradient: &mut [Vector3D],
) -> f64 {
    let offset = positions[pair.i] - positions[pair.j];
    let (energy, scale) = lennard_jones(
        pair.distance * pair.distance,
        pair.well_depth,
        offset.dot(&offset),
    );
    gradient[pair.i] += offset * scale;
    gradient[pair.j] += -(offset * scale);
    energy
}

/// One lane-width of pair offsets, and the squared separations.
#[inline(always)]
fn lane_offsets<L: Lanes>(
    simd: L,
    pairs: &VanDerWaalsPairs,
    positions: &[Vector3D],
    start: usize,
) -> ([L::Vector; 3], L::Vector) {
    let mut x = [0.0; LANES];
    let mut y = [0.0; LANES];
    let mut z = [0.0; LANES];
    for lane in 0..LANES {
        let offset =
            positions[pairs.i[start + lane] as usize] - positions[pairs.j[start + lane] as usize];
        x[lane] = offset.x;
        y[lane] = offset.y;
        z[lane] = offset.z;
    }
    let (x, y, z) = (simd.from_array(x), simd.from_array(y), simd.from_array(z));
    ([x, y, z], x * x + y * y + z * z)
}

#[inline(always)]
fn lane<L: Lanes>(simd: L, values: &[f64], start: usize) -> L::Vector {
    simd.from_array([
        values[start],
        values[start + 1],
        values[start + 2],
        values[start + 3],
    ])
}

/// [`lennard_jones`] four pairs at a time.
#[inline(always)]
fn lennard_jones_lanes<L: Lanes>(
    simd: L,
    minimum_squared: L::Vector,
    well_depth: L::Vector,
    squared: L::Vector,
) -> (L::Vector, L::Vector) {
    let splat = |value| simd.splat(value);
    let inverse = splat(1.0) / squared;
    let ratio = minimum_squared * inverse;
    let six = ratio * ratio * ratio;
    let scale = splat(12.0) * well_depth * (six - six * six) * inverse;
    let energy = well_depth * (six * six - splat(2.0) * six);
    let separated = splat(MIN_VDW_SEPARATION * MIN_VDW_SEPARATION);
    (
        simd.select_greater(squared, separated, energy, splat(f64::INFINITY)),
        simd.select_greater(squared, separated, scale, splat(0.0)),
    )
}

/// The Lennard-Jones energy of every pair, on the lanes `simd` provides.
#[inline(always)]
pub(super) fn van_der_waals_energies_in<L: Lanes>(
    simd: L,
    pairs: &VanDerWaalsPairs,
    positions: &[Vector3D],
) -> f64 {
    let mut total = simd.splat(0.0);
    let whole = pairs.len() - pairs.len() % LANES;
    let mut start = 0;
    while start < whole {
        let (_, squared) = lane_offsets(simd, pairs, positions, start);
        let minimum_squared = lane(simd, &pairs.minimum_squared, start);
        let well_depth = lane(simd, &pairs.well_depth, start);
        total += lennard_jones_lanes(simd, minimum_squared, well_depth, squared).0;
        start += LANES;
    }
    let mut total = horizontal_sum(simd.to_array(total));
    for index in whole..pairs.len() {
        let offset = positions[pairs.i[index] as usize] - positions[pairs.j[index] as usize];
        total += lennard_jones(
            pairs.minimum_squared[index],
            pairs.well_depth[index],
            offset.dot(&offset),
        )
        .0;
    }
    total
}

/// [`van_der_waals_energies_in`], accumulating the gradient.
#[inline(always)]
pub(super) fn van_der_waals_energies_and_gradient_in<L: Lanes>(
    simd: L,
    pairs: &VanDerWaalsPairs,
    positions: &[Vector3D],
    gradient: &mut [Vector3D],
) -> f64 {
    let mut total = simd.splat(0.0);
    let whole = pairs.len() - pairs.len() % LANES;
    let mut start = 0;
    while start < whole {
        let ([x, y, z], squared) = lane_offsets(simd, pairs, positions, start);
        let minimum_squared = lane(simd, &pairs.minimum_squared, start);
        let well_depth = lane(simd, &pairs.well_depth, start);
        let (energy, scale) = lennard_jones_lanes(simd, minimum_squared, well_depth, squared);
        total += energy;
        let (x, y, z) = (
            simd.to_array(x * scale),
            simd.to_array(y * scale),
            simd.to_array(z * scale),
        );
        for lane in 0..LANES {
            let step = Vector3D::new(x[lane], y[lane], z[lane]);
            gradient[pairs.i[start + lane] as usize] += step;
            gradient[pairs.j[start + lane] as usize] += -step;
        }
        start += LANES;
    }
    let mut total = horizontal_sum(simd.to_array(total));
    for index in whole..pairs.len() {
        let (i, j) = (pairs.i[index] as usize, pairs.j[index] as usize);
        let offset = positions[i] - positions[j];
        let (energy, scale) = lennard_jones(
            pairs.minimum_squared[index],
            pairs.well_depth[index],
            offset.dot(&offset),
        );
        total += energy;
        gradient[i] += offset * scale;
        gradient[j] += -(offset * scale);
    }
    total
}

/// `sin(omega)` for [`Inversion`]'s Wilson angle, and `d(sin omega)/dp` for
/// the four atoms, in the order `i, j, k, l`.
///
/// Taking the sine rather than the angle keeps the whole expression
/// rational: it is `n . l` for the unit plane normal `n` and the unit bond
/// `l`, and the energy needs only its cosine, which is `sqrt(1 - sin^2)`.
#[inline(always)]
fn out_of_plane_and_gradient(
    positions: &[Vector3D],
    inversion: &Inversion,
) -> Option<(f64, [Vector3D; 4])> {
    let first = positions[inversion.i] - positions[inversion.k];
    let second = positions[inversion.j] - positions[inversion.k];
    let out = positions[inversion.l] - positions[inversion.k];

    let normal = first.cross(&second);
    let normal_sq = normal.dot(&normal);
    let out_sq = out.dot(&out);
    if normal_sq < MIN_INVERSION_NORMAL_SQUARED || out_sq < MIN_INVERSION_BOND_SQUARED {
        return None;
    }
    let scale = 1.0 / (normal_sq.sqrt() * out_sq.sqrt());
    let sine = normal.dot(&out) * scale;

    let d_first = second.cross(&out) * scale - second.cross(&normal) * (sine / normal_sq);
    let d_second = out.cross(&first) * scale - normal.cross(&first) * (sine / normal_sq);
    let d_out = normal * scale - out * (sine / out_sq);
    let d_centre = -(d_first + d_second + d_out);
    Some((sine, [d_first, d_second, d_centre, d_out]))
}

#[inline(always)]
fn inversion_energy_from(inversion: &Inversion, sine: f64) -> f64 {
    let cosine = (1.0 - sine * sine).max(0.0).sqrt();
    inversion.force_constant
        * (inversion.c0 + inversion.c1 * cosine + inversion.c2 * (2.0 * cosine * cosine - 1.0))
}

#[inline(always)]
pub(super) fn inversion_energy(positions: &[Vector3D], inversion: &Inversion) -> f64 {
    match out_of_plane_and_gradient(positions, inversion) {
        Some((sine, _)) => inversion_energy_from(inversion, sine),
        None => 0.0,
    }
}

#[inline(always)]
pub(super) fn inversion_energy_and_gradient(
    positions: &[Vector3D],
    inversion: &Inversion,
    gradient: &mut [Vector3D],
) -> f64 {
    let Some((sine, d_sine)) = out_of_plane_and_gradient(positions, inversion) else {
        return 0.0;
    };
    let cosine = (1.0 - sine * sine).max(0.0).sqrt();
    // `d(cos omega)/d(sin omega)` carries a `1 / cos(omega)`, which runs
    // away at 90 degrees out of plane. That is the top of the barrier and
    // not somewhere a minimisation goes, so the term is simply flat there
    // rather than infinite.
    let d_energy = if cosine > MIN_ANGULAR_COMPONENT {
        -inversion.force_constant * (inversion.c1 + 4.0 * inversion.c2 * cosine) * sine / cosine
    } else {
        0.0
    };
    for (slot, atom) in [inversion.i, inversion.j, inversion.k, inversion.l]
        .into_iter()
        .enumerate()
    {
        gradient[atom] += d_sine[slot] * d_energy;
    }
    inversion_energy_from(inversion, sine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collinear_angles_retain_their_energy() {
        let angle = Angle {
            i: 0,
            k: 1,
            j: 2,
            force_constant: 100.0,
            fold: Some(3),
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
        };
        let positions = [
            Vector3D::new(-1.0, 0.0, 0.0),
            Vector3D::default(),
            Vector3D::new(1.0, 0.0, 0.0),
        ];
        let mut gradient = [Vector3D::default(); 3];
        let energy = angle_energy_and_gradient(&positions, &angle, &mut gradient);
        assert!((energy - 200.0 / 9.0).abs() < 1e-10);
        assert_eq!(energy, angle_energy(&positions, &angle));
        let mut nearby = positions;
        nearby[2].y = 1e-5;
        assert!((energy - angle_energy(&nearby, &angle)).abs() < 1e-6);
        assert!(gradient.iter().all(|g| g.length().is_finite()));
    }

    #[test]
    fn overlapping_atoms_do_not_erase_bond_or_repulsion_energy() {
        let positions = [Vector3D::default(); 2];
        let mut gradient = [Vector3D::default(); 2];
        let bond = Bond {
            i: 0,
            j: 1,
            rest_length: 1.5,
            force_constant: 100.0,
        };
        assert_eq!(
            bond_energy_and_gradient(&positions, &bond, &mut gradient),
            112.5
        );
        let pair = VanDerWaals {
            i: 0,
            j: 1,
            distance: 3.5,
            well_depth: 0.1,
        };
        assert_eq!(van_der_waals_energy(&positions, &pair), f64::INFINITY);
        assert_eq!(
            van_der_waals_energy_and_gradient(&positions, &pair, &mut gradient),
            f64::INFINITY
        );
    }

    /// Pairs for the lane tests: 23 atoms on a helix, two of them on top of
    /// each other, every pair but bonded neighbours, so the pair count is
    /// not a multiple of the lane width.
    fn helix_pairs() -> (Vec<Vector3D>, Vec<VanDerWaals>, VanDerWaalsPairs) {
        let mut positions: Vec<Vector3D> = (0..23)
            .map(|index| {
                let t = index as f64 * 0.77;
                Vector3D::new(t.sin() * 4.0, t.cos() * 3.0, index as f64 * 0.31)
            })
            .collect();
        positions[7] = positions[3];
        let mut scalar = Vec::new();
        let mut pairs = VanDerWaalsPairs::default();
        for i in 0..positions.len() {
            for j in (i + 2)..positions.len() {
                let pair = VanDerWaals {
                    i,
                    j,
                    distance: 3.0 + 0.05 * i as f64,
                    well_depth: 0.05 + 0.003 * j as f64,
                };
                scalar.push(pair);
                pairs.push(pair);
            }
        }
        assert_ne!(pairs.len() % LANES, 0);
        (positions, scalar, pairs)
    }

    /// The lanes are the scalar term four at a time: same energy up to the
    /// order of the sum, and the same gradient.
    #[test]
    fn the_vectorised_pairs_agree_with_the_scalar_ones() {
        let (mut positions, scalar, pairs) = helix_pairs();
        // Apart, so the energy is finite and comparable.
        positions[7] = positions[7] + Vector3D::new(0.4, 0.0, 0.0);
        let portable = crate::molecule::forcefield::lanes::PORTABLE;

        let mut want_gradient = vec![Vector3D::default(); positions.len()];
        let want: f64 = scalar
            .iter()
            .map(|pair| van_der_waals_energy_and_gradient(&positions, pair, &mut want_gradient))
            .sum();
        let energy = van_der_waals_energies_in(portable, &pairs, &positions);
        assert!(
            (energy - want).abs() < 1e-9 * want.abs().max(1.0),
            "{energy} against {want}"
        );

        let mut gradient = vec![Vector3D::default(); positions.len()];
        let energy =
            van_der_waals_energies_and_gradient_in(portable, &pairs, &positions, &mut gradient);
        assert!(
            (energy - want).abs() < 1e-9 * want.abs().max(1.0),
            "{energy} against {want}"
        );
        for (got, want) in gradient.iter().zip(&want_gradient) {
            for axis in [got.x - want.x, got.y - want.y, got.z - want.z] {
                assert!(
                    axis.abs() < 1e-9 * want.length().max(1.0),
                    "gradient differs by {axis}"
                );
            }
        }

        // And an exact overlap is infinite in lanes as in scalar.
        positions[7] = positions[3];
        assert_eq!(
            van_der_waals_energies_in(portable, &pairs, &positions),
            f64::INFINITY
        );
    }

    /// Every lanes backend agrees to the bit, overlap included.
    #[test]
    fn every_lanes_backend_agrees_to_the_bit() {
        use crate::molecule::forcefield::lanes::{Lanes, LanesOp, on_every_lanes};
        #[derive(Clone)]
        struct Run<'a>(&'a VanDerWaalsPairs, &'a [Vector3D]);
        impl LanesOp for Run<'_> {
            type Output = Vec<u64>;
            fn run<L: Lanes>(self, simd: L) -> Vec<u64> {
                let energy = van_der_waals_energies_in(simd, self.0, self.1);
                let mut gradient = vec![Vector3D::default(); self.1.len()];
                let with =
                    van_der_waals_energies_and_gradient_in(simd, self.0, self.1, &mut gradient);
                [energy, with]
                    .into_iter()
                    .chain(gradient.iter().flat_map(|g| [g.x, g.y, g.z]))
                    .map(f64::to_bits)
                    .collect()
            }
        }
        let (mut positions, _, pairs) = helix_pairs();
        for overlap in [false, true] {
            if !overlap {
                positions[7] = positions[3] + Vector3D::new(0.4, 0.0, 0.0);
            } else {
                positions[7] = positions[3];
            }
            let [portable, native] = on_every_lanes(Run(&pairs, &positions));
            assert_eq!(portable, native, "overlap: {overlap}");
        }
    }
}
