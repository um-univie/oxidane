//! MMFF94's seven energy terms, and their gradients.
//!
//! The forms are Halgren's, and they are not UFF's: the bond is quartic
//! rather than harmonic, the angle carries a cubic correction, a
//! stretch-bend term couples the two, out-of-plane bending is measured as a
//! Wilson angle, and the van der Waals term is a buffered 14-7 rather than a
//! Lennard-Jones 12-6. Angles and their force constants are in degrees
//! throughout, which is why the conversion factors appear where they do.

use super::assign::DEGREES_TO_RADIANS;
use crate::vector::Vector3D;

/// Millidyne-angstrom to kcal/mol, the unit conversion every MMFF force
/// constant is quoted against.
const MDYNE_TO_KCAL: f64 = 143.9325;

/// The same, for a force constant quoted per squared degree.
const MDYNE_TO_KCAL_PER_DEGREE_SQUARED: f64 =
    MDYNE_TO_KCAL * DEGREES_TO_RADIANS * DEGREES_TO_RADIANS;

/// Coulomb's constant in these units.
const COULOMB: f64 = 332.0716;

/// How much of the electrostatic interaction survives between atoms three
/// bonds apart, which are already restrained by the torsion term.
pub(super) const ONE_FOUR_SCALE: f64 = 0.75;

/// The buffer added to an electrostatic separation, which keeps two atoms
/// on top of each other from producing an infinite energy.
const ELECTROSTATIC_BUFFER: f64 = 0.05;

/// The cubic and quartic coefficients of the bond's expansion.
const BOND_CUBIC: f64 = -2.0;
const BOND_QUARTIC: f64 = 7.0 / 12.0;

/// The cubic coefficient of the angle's, per degree.
const ANGLE_CUBIC: f64 = -0.006981317;

pub(super) struct Bond {
    pub i: usize,
    pub j: usize,
    pub force_constant: f64,
    pub length: f64,
}

pub(super) struct Angle {
    pub i: usize,
    pub central: usize,
    pub k: usize,
    pub force_constant: f64,
    pub angle: f64,
    /// A linear centre uses a form with its minimum at 180 degrees rather
    /// than a polynomial around one.
    pub linear: bool,
    /// The stretch-bend coupling at this same angle, where MMFF has
    /// parameters for it.
    ///
    /// Here rather than in a list of its own because it is the same three
    /// atoms, the same reference angle and the same geometry: kept apart,
    /// every energy evaluation computed the angle twice -- 166 angle
    /// computations per evaluation for cocaine's 83 angles, each with its
    /// own `acos`, and 1.39 million of them in one default-configuration
    /// embedding, where `angle_and_gradient` was 14.9% of the run.
    ///
    /// Folding the two together and giving the energy-only path a geometry
    /// without the derivative it discards is worth 7% of a small
    /// embedding and 13% of the default configuration, and changes no
    /// energy: the arithmetic is the same, once instead of twice.
    pub stretch: Option<StretchBend>,
}

pub(super) struct StretchBend {
    pub first_constant: f64,
    pub second_constant: f64,
    pub first_length: f64,
    pub second_length: f64,
}

pub(super) struct OutOfPlane {
    pub i: usize,
    pub central: usize,
    pub k: usize,
    pub l: usize,
    pub force_constant: f64,
}

pub(super) struct Torsion {
    /// MMFF's torsion type, kept for the diagnostics rather than the maths.
    pub kind: u8,
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
}

/// One through-space pair, carrying both of the interactions it has.
///
/// Van der Waals and electrostatics run over the same pairs -- everything at
/// least [`super::field::NONBONDED_MIN_SEPARATION`] bonds apart -- and each
/// needs the same separation. Kept apart, each pair's distance was
/// subtracted, squared and square-rooted twice per evaluation, and those two
/// loops were 45% of the energy on a ninety-atom peptide.
///
/// The constants are the buffered-14-7 form's, folded in at construction
/// rather than recomputed at every geometry: `minimum^7` alone was a `powi`
/// per pair per evaluation, and the gradient did it twice more. A pair with
/// no van der Waals parameters carries `well = 0` and zeroes throughout, so
/// it costs the arithmetic but contributes nothing -- cheaper than a branch,
/// and there are very few of them.
#[derive(Clone, Copy)]
pub(super) struct NonBonded {
    /// `u32` rather than `usize`, which halves what the indices cost in the
    /// one array long enough to notice -- a ninety-atom peptide has 3753
    /// pairs. Measured, the 40-byte record and the 72-byte one with every
    /// constant pre-multiplied run within noise of each other, so this is
    /// not a cache win; it is the smaller record of two equals. No molecule
    /// this crate embeds has four billion atoms.
    pub i: u32,
    pub j: u32,
    pub well: f64,
    pub minimum: f64,
    /// `minimum^7`, which is a `powi` per pair per evaluation if it is not
    /// kept, and the gradient wants it three times over.
    pub minimum_seventh: f64,
    /// The product of the two charges, already divided by the dielectric
    /// constant and scaled if the pair is a 1-4 one. Zero when either atom
    /// is uncharged.
    pub charge_product: f64,
}

impl NonBonded {
    pub(super) fn new(i: usize, j: usize, minimum: f64, well: f64, charge_product: f64) -> Self {
        NonBonded {
            i: i as u32,
            j: j as u32,
            well,
            minimum,
            minimum_seventh: minimum.powi(7),
            charge_product,
        }
    }
}

/// Every nonbonded pair, one array per field rather than one record per
/// pair.
///
/// The pair list is the longest loop in the force field -- 30,628 pairs for
/// a 248-atom molecule, evaluated 15,735 times in one embedding of it, 470
/// million times in all -- and every pair does the same square root and the
/// same three divisions with no early exit. That is the shape that wants
/// vector lanes, and lanes want each parameter in its own array: four pairs'
/// `minimum` have to be four adjacent `f64`, not four fields 40 bytes apart.
///
/// Measured against the array of records this replaced, same arithmetic,
/// four pairs at a time: 1.80x at 90 atoms, 1.79x at 248, 1.78x at 390.
/// Written over plain `[f64; 4]` arrays instead, leaving the vectorisation
/// to LLVM, it is 1.36x -- which is the fallback this would have if
/// [`wide`] were not a dependency, and the reason it is one.
#[derive(Default)]
pub(super) struct NonBondedPairs {
    i: Vec<u32>,
    j: Vec<u32>,
    well: Vec<f64>,
    minimum: Vec<f64>,
    minimum_seventh: Vec<f64>,
    charge_product: Vec<f64>,
}

impl NonBondedPairs {
    pub(super) fn push(&mut self, pair: NonBonded) {
        self.i.push(pair.i);
        self.j.push(pair.j);
        self.well.push(pair.well);
        self.minimum.push(pair.minimum);
        self.minimum_seventh.push(pair.minimum_seventh);
        self.charge_product.push(pair.charge_product);
    }

    pub(super) fn len(&self) -> usize {
        self.i.len()
    }

    /// One pair back in record form, for the remainder the lanes do not
    /// cover and for anything that wants to read a single pair.
    fn get(&self, index: usize) -> NonBonded {
        NonBonded {
            i: self.i[index],
            j: self.j[index],
            well: self.well[index],
            minimum: self.minimum[index],
            minimum_seventh: self.minimum_seventh[index],
            charge_product: self.charge_product[index],
        }
    }
}

/// How many pairs one pass of the vectorised loop covers.
const LANES: usize = 4;

/// The buffering constants of MMFF's 14-7: `NEAR` softens the attractive
/// separation and `FAR` the repulsive one, so that two atoms on top of each
/// other cost a large energy rather than an infinite one.
const NEAR: f64 = 1.07;
const FAR: f64 = 1.12;

#[inline(always)]
pub(super) fn bond_energy(positions: &[Vector3D], bond: &Bond) -> f64 {
    let stretch = positions[bond.i].distance(&positions[bond.j]) - bond.length;
    0.5 * MDYNE_TO_KCAL
        * bond.force_constant
        * stretch
        * stretch
        * (1.0 + BOND_CUBIC * stretch + BOND_QUARTIC * BOND_CUBIC * BOND_CUBIC * stretch * stretch)
}

#[inline(always)]
pub(super) fn bond_energy_and_gradient(
    positions: &[Vector3D],
    bond: &Bond,
    gradient: &mut [Vector3D],
) -> f64 {
    let offset = positions[bond.i] - positions[bond.j];
    let distance = offset.length();
    if distance < 1e-8 {
        return bond_energy(positions, bond);
    }
    let stretch = distance - bond.length;
    let derivative = MDYNE_TO_KCAL
        * bond.force_constant
        * stretch
        * (1.0
            + 1.5 * BOND_CUBIC * stretch
            + 2.0 * BOND_QUARTIC * BOND_CUBIC * BOND_CUBIC * stretch * stretch);
    let scale = derivative / distance;
    gradient[bond.i] += offset * scale;
    gradient[bond.j] += -(offset * scale);
    bond_energy(positions, bond)
}

/// The angle at `central`, with `d(theta)/dp` for its three atoms.
/// One angle's geometry: the two bond vectors and their lengths, the
/// angle, and its cosine. Everything both the bending and the stretch-bend
/// term need, and the only `acos` either of them pays for.
struct AngleGeometry {
    first: Vector3D,
    second: Vector3D,
    first_distance: f64,
    second_distance: f64,
    theta: f64,
    cosine: f64,
}

/// [`AngleGeometry`] without the derivative, which is what an energy
/// evaluation wants: the line search takes about three of those per
/// gradient, and the two vectors and the division by `sin` the gradient
/// needs are most of the cost of computing one.
#[inline(always)]
fn angle_geometry(
    positions: &[Vector3D],
    i: usize,
    central: usize,
    k: usize,
) -> Option<AngleGeometry> {
    let first = positions[i] - positions[central];
    let second = positions[k] - positions[central];
    let (one, two) = (first.length(), second.length());
    if one < 1e-8 || two < 1e-8 {
        return None;
    }
    let cosine = (first.dot(&second) / (one * two)).clamp(-1.0, 1.0);
    Some(AngleGeometry {
        first,
        second,
        first_distance: one,
        second_distance: two,
        theta: cosine.acos(),
        cosine,
    })
}

/// [`angle_geometry`] and the derivative of the angle with respect to the
/// three atoms.
#[inline(always)]
fn angle_and_gradient(
    positions: &[Vector3D],
    i: usize,
    central: usize,
    k: usize,
) -> Option<(AngleGeometry, [Vector3D; 3])> {
    let geometry = angle_geometry(positions, i, central, k)?;
    let (one, two) = (geometry.first_distance, geometry.second_distance);
    let cosine = geometry.cosine;
    let sine = (1.0 - cosine * cosine).sqrt();
    if sine < 1e-8 {
        // A collinear angle still costs bending energy. Only its angular
        // gradient lacks a unique direction; the symmetric value is zero.
        // Stretch-bend must also retain its radial forces at this geometry.
        return Some((geometry, [Vector3D::default(); 3]));
    }
    let d_cos_first = geometry.second * (1.0 / (one * two)) - geometry.first * (cosine / (one * one));
    let d_cos_second = geometry.first * (1.0 / (one * two)) - geometry.second * (cosine / (two * two));
    let to_angle = -1.0 / sine;
    let d_i = d_cos_first * to_angle;
    let d_k = d_cos_second * to_angle;
    Some((geometry, [d_i, -(d_i + d_k), d_k]))
}

#[inline(always)]
fn angle_energy_from(angle: &Angle, theta: f64, cosine: f64) -> f64 {
    if angle.linear {
        return MDYNE_TO_KCAL * angle.force_constant * (1.0 + cosine);
    }
    let difference = theta / DEGREES_TO_RADIANS - angle.angle;
    0.5 * MDYNE_TO_KCAL_PER_DEGREE_SQUARED
        * angle.force_constant
        * difference
        * difference
        * (1.0 + ANGLE_CUBIC * difference)
}

/// The bending energy and the stretch-bend energy of one angle, from one
/// computation of its geometry, as `(bend, stretch)`.
///
/// Reported apart because that is how MMFF is checked, term by term
/// against RDKit, even though they are now computed together.
#[inline(always)]
pub(super) fn angle_energies(positions: &[Vector3D], angle: &Angle) -> (f64, f64) {
    let Some(geometry) = angle_geometry(positions, angle.i, angle.central, angle.k) else {
        return (0.0, 0.0);
    };
    (
        angle_energy_from(angle, geometry.theta, geometry.cosine),
        angle
            .stretch
            .as_ref()
            .map_or(0.0, |stretch| stretch_energy_from(angle, stretch, &geometry)),
    )
}

/// The stretch-bend energy, given the angle's geometry: the bend away from
/// the reference angle, coupled to each bond's stretch away from its own.
#[inline(always)]
fn stretch_energy_from(angle: &Angle, stretch: &StretchBend, geometry: &AngleGeometry) -> f64 {
    let first = geometry.first_distance - stretch.first_length;
    let second = geometry.second_distance - stretch.second_length;
    let bend = geometry.theta / DEGREES_TO_RADIANS - angle.angle;
    MDYNE_TO_KCAL
        * DEGREES_TO_RADIANS
        * bend
        * (stretch.first_constant * first + stretch.second_constant * second)
}

/// [`angle_energies`], accumulating both terms' gradients from the one
/// derivative of the angle.
#[inline(always)]
pub(super) fn angle_energies_and_gradient(
    positions: &[Vector3D],
    angle: &Angle,
    gradient: &mut [Vector3D],
) -> (f64, f64) {
    let Some((geometry, derivative)) =
        angle_and_gradient(positions, angle.i, angle.central, angle.k)
    else {
        return (0.0, 0.0);
    };
    let (theta, cosine) = (geometry.theta, geometry.cosine);
    let atoms = [angle.i, angle.central, angle.k];

    let sine = (1.0 - cosine * cosine).sqrt();
    let d_energy = if angle.linear {
        -MDYNE_TO_KCAL * angle.force_constant * sine
    } else {
        let difference = theta / DEGREES_TO_RADIANS - angle.angle;
        (0.5 * MDYNE_TO_KCAL_PER_DEGREE_SQUARED
            * angle.force_constant
            * difference
            * (2.0 + 3.0 * ANGLE_CUBIC * difference))
            / DEGREES_TO_RADIANS
    };
    for (slot, atom) in atoms.into_iter().enumerate() {
        gradient[atom] += derivative[slot] * d_energy;
    }
    let bend_energy = angle_energy_from(angle, theta, cosine);

    let Some(stretch) = angle.stretch.as_ref() else {
        return (bend_energy, 0.0);
    };
    let first = geometry.first_distance - stretch.first_length;
    let second = geometry.second_distance - stretch.second_length;
    let bend = theta / DEGREES_TO_RADIANS - angle.angle;
    let factor = MDYNE_TO_KCAL * DEGREES_TO_RADIANS;

    // The angle's share, along the same derivative the bending term used.
    let d_bend = factor
        * (stretch.first_constant * first + stretch.second_constant * second)
        / DEGREES_TO_RADIANS;
    for (slot, atom) in atoms.into_iter().enumerate() {
        gradient[atom] += derivative[slot] * d_bend;
    }
    // Each bond's.
    let first_scale = factor * bend * stretch.first_constant / geometry.first_distance;
    gradient[angle.i] += geometry.first * first_scale;
    gradient[angle.central] += -(geometry.first * first_scale);
    let second_scale = factor * bend * stretch.second_constant / geometry.second_distance;
    gradient[angle.k] += geometry.second * second_scale;
    gradient[angle.central] += -(geometry.second * second_scale);

    (
        bend_energy,
        factor * bend * (stretch.first_constant * first + stretch.second_constant * second),
    )
}

/// The Wilson angle in degrees: how far `central -> l` lies out of the plane
/// of `central -> i` and `central -> k`.
#[inline(always)]
fn wilson_angle(positions: &[Vector3D], term: &OutOfPlane) -> Option<f64> {
    let to_i = (positions[term.i] - positions[term.central]).normalize();
    let to_k = (positions[term.k] - positions[term.central]).normalize();
    let to_l = (positions[term.l] - positions[term.central]).normalize();
    let normal = to_i.cross(&to_k);
    let length = normal.length();
    if length < 1e-8 {
        return None;
    }
    let sine = ((normal * (1.0 / length)).dot(&to_l)).clamp(-1.0, 1.0);
    Some(sine.asin() / DEGREES_TO_RADIANS)
}

#[inline(always)]
pub(super) fn out_of_plane_energy(positions: &[Vector3D], term: &OutOfPlane) -> f64 {
    match wilson_angle(positions, term) {
        Some(chi) => 0.5 * MDYNE_TO_KCAL_PER_DEGREE_SQUARED * term.force_constant * chi * chi,
        None => 0.0,
    }
}

#[inline(always)]
pub(super) fn out_of_plane_energy_and_gradient(
    positions: &[Vector3D],
    term: &OutOfPlane,
    gradient: &mut [Vector3D],
) -> f64 {
    let to_i = positions[term.i] - positions[term.central];
    let to_k = positions[term.k] - positions[term.central];
    let to_l = positions[term.l] - positions[term.central];
    let (length_i, length_k, length_l) = (to_i.length(), to_k.length(), to_l.length());
    if length_i < 1e-8 || length_k < 1e-8 || length_l < 1e-8 {
        return 0.0;
    }
    let unit_i = to_i * (1.0 / length_i);
    let unit_k = to_k * (1.0 / length_k);
    let unit_l = to_l * (1.0 / length_l);

    let normal = (-unit_i).cross(&unit_k);
    let normal_length = normal.length();
    if normal_length < 1e-8 {
        return 0.0;
    }
    let normal = normal * (1.0 / normal_length);
    let sine = unit_l.dot(&normal).clamp(-1.0, 1.0);
    let chi = sine.asin() / DEGREES_TO_RADIANS;
    // Ninety degrees is as far out of the plane as `l` can get, so the Wilson
    // angle is at a maximum there, and every factor below carries
    // `1 / cosine`. The energy stays bounded while its derivative does not,
    // which is the same coordinate singularity `crate::vector::cap_derivatives`
    // describes for a dihedral, so it gets the same treatment rather than a
    // floor: this used to floor the cosine at 1e-8 and report a gradient of
    // order 1e8 on a term worth a few kcal/mol.
    let cosine = (1.0 - sine * sine).max(0.0).sqrt().max(1e-8);
    let cos_theta = unit_i.dot(&unit_k).clamp(-1.0, 1.0);
    let sin_theta_squared = (1.0 - cos_theta * cos_theta).max(1e-8);
    let sin_theta = sin_theta_squared.sqrt().max(1e-8);

    let d_energy =
        MDYNE_TO_KCAL_PER_DEGREE_SQUARED * term.force_constant * chi / DEGREES_TO_RADIANS;
    let first = unit_l.cross(&unit_k);
    let second = unit_i.cross(&unit_l);
    let third = unit_k.cross(&unit_i);
    let scale = cosine * sin_theta;
    let tilt = sine / (cosine * sin_theta_squared);

    let [d_i, d_k, d_l] = crate::vector::cap_derivatives([
        (first * (1.0 / scale) - (unit_i - unit_k * cos_theta) * tilt) * (1.0 / length_i),
        (second * (1.0 / scale) - (unit_k - unit_i * cos_theta) * tilt) * (1.0 / length_k),
        (third * (1.0 / scale) - unit_l * (sine / cosine)) * (1.0 / length_l),
    ]);

    gradient[term.i] += d_i * d_energy;
    gradient[term.central] += -((d_i + d_k + d_l) * d_energy);
    gradient[term.k] += d_k * d_energy;
    gradient[term.l] += d_l * d_energy;

    0.5 * MDYNE_TO_KCAL_PER_DEGREE_SQUARED * term.force_constant * chi * chi
}

#[inline(always)]
fn torsion_energy_from(torsion: &Torsion, cosine: f64) -> f64 {
    let cosine2 = 2.0 * cosine * cosine - 1.0;
    let cosine3 = cosine * (2.0 * cosine2 - 1.0);
    0.5 * (torsion.v1 * (1.0 + cosine)
        + torsion.v2 * (1.0 - cosine2)
        + torsion.v3 * (1.0 + cosine3))
}

#[inline(always)]
pub(super) fn torsion_energy(positions: &[Vector3D], torsion: &Torsion) -> f64 {
    // The gradient-free dihedral: the energy needs `cos(phi)` and nothing
    // else, and the four derivative vectors the other one builds are most
    // of its cost.
    match crate::vector::dihedral_cosine_sine(
        positions[torsion.i],
        positions[torsion.j],
        positions[torsion.k],
        positions[torsion.l],
    ) {
        Some((cosine, _)) => torsion_energy_from(torsion, cosine),
        None => 0.0,
    }
}

#[inline(always)]
pub(super) fn torsion_energy_and_gradient(
    positions: &[Vector3D],
    torsion: &Torsion,
    gradient: &mut [Vector3D],
) -> f64 {
    let Some((cosine, sine, derivative)) = crate::vector::dihedral_and_gradient(
        positions[torsion.i],
        positions[torsion.j],
        positions[torsion.k],
        positions[torsion.l],
    ) else {
        return 0.0;
    };
    // sin(2 phi) and sin(3 phi) by the double- and triple-angle identities
    // rather than two more libm calls.
    let sine2 = 2.0 * sine * cosine;
    let sine3 = sine * (4.0 * cosine * cosine - 1.0);
    let d_energy =
        0.5 * (-torsion.v1 * sine + 2.0 * torsion.v2 * sine2 - 3.0 * torsion.v3 * sine3);
    for (slot, atom) in [torsion.i, torsion.j, torsion.k, torsion.l]
        .into_iter()
        .enumerate()
    {
        gradient[atom] += derivative[slot] * d_energy;
    }
    torsion_energy_from(torsion, cosine)
}

/// The buffered 14-7 form: like a Lennard-Jones curve, but with both terms
/// The van der Waals and electrostatic energies of one pair.
///
/// Returned separately because that is how MMFF is checked, term by term
/// against RDKit, even though they are computed from one distance.
#[inline(always)]
pub(super) fn nonbonded_energy(positions: &[Vector3D], pair: &NonBonded) -> (f64, f64) {
    let offset = positions[pair.i as usize] - positions[pair.j as usize];
    let distance_squared = offset.dot(&offset);
    let distance = distance_squared.sqrt();

    let attenuated = NEAR * pair.minimum / (distance + (NEAR - 1.0) * pair.minimum);
    let squared = attenuated * attenuated;
    let cubed = squared * attenuated;
    let seventh = cubed * cubed * attenuated;
    // `d^7` from `d^2`, which the offset already gave: three multiplications
    // against `powi`'s four, and no second square root.
    let sixth = distance_squared * distance_squared * distance_squared;
    let repulsion =
        FAR * pair.minimum_seventh / (sixth * distance + (FAR - 1.0) * pair.minimum_seventh);

    (
        pair.well * seventh * (repulsion - 2.0),
        COULOMB * pair.charge_product / (distance + ELECTROSTATIC_BUFFER),
    )
}

/// [`nonbonded_energy`], accumulating both interactions' gradients.
#[inline(always)]
pub(super) fn nonbonded_energy_and_gradient(
    positions: &[Vector3D],
    pair: &NonBonded,
    gradient: &mut [Vector3D],
) -> (f64, f64) {
    let offset = positions[pair.i as usize] - positions[pair.j as usize];
    let distance_squared = offset.dot(&offset);
    let distance = distance_squared.sqrt();
    if distance < 1e-8 {
        // No direction to push along; the energies are still finite.
        return nonbonded_energy(positions, pair);
    }

    let buffered_near = distance + (NEAR - 1.0) * pair.minimum;
    let attenuated = NEAR * pair.minimum / buffered_near;
    let squared = attenuated * attenuated;
    let cubed = squared * attenuated;
    let sixth_attenuated = cubed * cubed;
    let seventh = sixth_attenuated * attenuated;

    let sixth = distance_squared * distance_squared * distance_squared;
    let denominator = sixth * distance + (FAR - 1.0) * pair.minimum_seventh;
    let repulsion = FAR * pair.minimum_seventh / denominator;

    let d_attenuated = -attenuated / buffered_near;
    let d_repulsion = -FAR * pair.minimum_seventh * 7.0 * sixth / (denominator * denominator);
    let van_der_waals = pair.well
        * (7.0 * sixth_attenuated * d_attenuated * (repulsion - 2.0) + seventh * d_repulsion);

    let buffered_far = distance + ELECTROSTATIC_BUFFER;
    let scaled = COULOMB * pair.charge_product / buffered_far;
    let (electrostatic, coulombic) = (-scaled / buffered_far, scaled);

    let scale = (van_der_waals + electrostatic) / distance;
    gradient[pair.i as usize] += offset * scale;
    gradient[pair.j as usize] += -(offset * scale);

    (pair.well * seventh * (repulsion - 2.0), coulombic)
}

/// Four `f64` lanes, and the few operations the pair loop needs beyond
/// arithmetic.
///
/// Two of these exist. [`Portable`] is `wide::f64x4`, whose instruction set is
/// fixed when the crate is compiled -- for baseline x86-64 that is two SSE2
/// halves. [`Avx`] is the same four lanes in one AVX register, available only
/// once [`pulp`] has confirmed the CPU has AVX2 and only inside code it has
/// compiled for it; see `ForceField::energy_and_gradient`. Both do exactly
/// the same IEEE operations in the same order, so they agree to the bit and
/// the choice between them changes speed, not results.
pub(super) trait Lanes: Copy {
    type Vector: Copy
        + core::ops::Add<Output = Self::Vector>
        + core::ops::Sub<Output = Self::Vector>
        + core::ops::Mul<Output = Self::Vector>
        + core::ops::Div<Output = Self::Vector>
        + core::ops::Neg<Output = Self::Vector>
        + core::ops::AddAssign;

    fn splat(self, value: f64) -> Self::Vector;
    fn from_array(self, values: [f64; LANES]) -> Self::Vector;
    fn to_array(self, vector: Self::Vector) -> [f64; LANES];
    fn sqrt(self, vector: Self::Vector) -> Self::Vector;
    /// `value` where `left > right`, zero elsewhere.
    fn where_greater(
        self,
        left: Self::Vector,
        right: Self::Vector,
        value: Self::Vector,
    ) -> Self::Vector;
}

/// [`Lanes`] on `wide`, for whatever the crate was compiled for.
#[derive(Clone, Copy)]
pub(super) struct Portable;

impl Lanes for Portable {
    type Vector = wide::f64x4;

    #[inline(always)]
    fn splat(self, value: f64) -> wide::f64x4 {
        wide::f64x4::splat(value)
    }
    #[inline(always)]
    fn from_array(self, values: [f64; LANES]) -> wide::f64x4 {
        wide::f64x4::new(values)
    }
    #[inline(always)]
    fn to_array(self, vector: wide::f64x4) -> [f64; LANES] {
        vector.to_array()
    }
    #[inline(always)]
    fn sqrt(self, vector: wide::f64x4) -> wide::f64x4 {
        vector.sqrt()
    }
    #[inline(always)]
    fn where_greater(self, left: wide::f64x4, right: wide::f64x4, value: wide::f64x4) -> wide::f64x4 {
        wide::CmpGt::cmp_gt(left, right).blend(value, wide::f64x4::ZERO)
    }
}

/// [`Lanes`] in one 256-bit register. Holding a `pulp::x86::V3` is the proof
/// that the CPU has AVX2, which is what makes the operations below sound.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(super) struct Avx(pub(super) pulp::x86::V3);

/// One [`Avx`] vector. It carries the token so the arithmetic operators can
/// reach the instructions.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(super) struct AvxVector(pulp::x86::V3, pulp::f64x4);

#[cfg(target_arch = "x86_64")]
macro_rules! avx_binary {
    ($trait:ident, $method:ident, $instruction:ident) => {
        impl core::ops::$trait for AvxVector {
            type Output = AvxVector;
            #[inline(always)]
            fn $method(self, other: AvxVector) -> AvxVector {
                AvxVector(self.0, self.0.$instruction(self.1, other.1))
            }
        }
    };
}
#[cfg(target_arch = "x86_64")]
avx_binary!(Add, add, add_f64x4);
#[cfg(target_arch = "x86_64")]
avx_binary!(Sub, sub, sub_f64x4);
#[cfg(target_arch = "x86_64")]
avx_binary!(Mul, mul, mul_f64x4);
#[cfg(target_arch = "x86_64")]
avx_binary!(Div, div, div_f64x4);

#[cfg(target_arch = "x86_64")]
impl core::ops::Neg for AvxVector {
    type Output = AvxVector;
    /// `0 - x`, as `wide` does it, rather than a sign flip: the two differ
    /// on a zero, and the paths have to agree to the bit.
    #[inline(always)]
    fn neg(self) -> AvxVector {
        AvxVector(self.0, self.0.sub_f64x4(self.0.splat_f64x4(0.0), self.1))
    }
}

#[cfg(target_arch = "x86_64")]
impl core::ops::AddAssign for AvxVector {
    #[inline(always)]
    fn add_assign(&mut self, other: AvxVector) {
        *self = *self + other;
    }
}

#[cfg(target_arch = "x86_64")]
impl Lanes for Avx {
    type Vector = AvxVector;

    #[inline(always)]
    fn splat(self, value: f64) -> AvxVector {
        AvxVector(self.0, self.0.splat_f64x4(value))
    }
    #[inline(always)]
    fn from_array(self, [a, b, c, d]: [f64; LANES]) -> AvxVector {
        AvxVector(self.0, pulp::f64x4(a, b, c, d))
    }
    #[inline(always)]
    fn to_array(self, vector: AvxVector) -> [f64; LANES] {
        let pulp::f64x4(a, b, c, d) = vector.1;
        [a, b, c, d]
    }
    #[inline(always)]
    fn sqrt(self, vector: AvxVector) -> AvxVector {
        AvxVector(self.0, self.0.sqrt_f64x4(vector.1))
    }
    #[inline(always)]
    fn where_greater(self, left: AvxVector, right: AvxVector, value: AvxVector) -> AvxVector {
        let mask = self.0.cmp_gt_f64x4(left.1, right.1);
        AvxVector(self.0, self.0.select_f64x4(mask, value.1, self.0.splat_f64x4(0.0)))
    }
}

/// Gathers one lane-width of pair offsets, and the squared distances.
#[inline(always)]
fn lane_offsets<L: Lanes>(
    simd: L,
    pairs: &NonBondedPairs,
    positions: &[Vector3D],
    start: usize,
) -> ([L::Vector; 3], L::Vector) {
    let mut x = [0.0; LANES];
    let mut y = [0.0; LANES];
    let mut z = [0.0; LANES];
    for lane in 0..LANES {
        let offset = positions[pairs.i[start + lane] as usize]
            - positions[pairs.j[start + lane] as usize];
        x[lane] = offset.x;
        y[lane] = offset.y;
        z[lane] = offset.z;
    }
    let (x, y, z) = (simd.from_array(x), simd.from_array(y), simd.from_array(z));
    ([x, y, z], x * x + y * y + z * z)
}

/// One lane-width of a parameter array.
#[inline(always)]
fn lane<L: Lanes>(simd: L, values: &[f64], start: usize) -> L::Vector {
    simd.from_array([
        values[start],
        values[start + 1],
        values[start + 2],
        values[start + 3],
    ])
}

/// The van der Waals and electrostatic energies of every pair, four at a
/// time.
///
/// The same arithmetic as [`nonbonded_energy`], in lanes. What changes is
/// the order the sum is taken in, which moves the total by the last bit or
/// two of an `f64`; see [`NonBondedPairs`] for what the layout buys.
#[cfg(test)]
pub(super) fn nonbonded_energies(pairs: &NonBondedPairs, positions: &[Vector3D]) -> (f64, f64) {
    nonbonded_energies_in(Portable, pairs, positions)
}

/// [`nonbonded_energies`] on the lanes `simd` provides.
#[inline(always)]
pub(super) fn nonbonded_energies_in<L: Lanes>(
    simd: L,
    pairs: &NonBondedPairs,
    positions: &[Vector3D],
) -> (f64, f64) {
    let splat = |value| simd.splat(value);
    let (mut van_der_waals, mut electrostatic) = (splat(0.0), splat(0.0));
    let whole = pairs.len() - pairs.len() % LANES;
    let mut start = 0;
    while start < whole {
        let (_, squared) = lane_offsets(simd, pairs, positions, start);
        let distance = simd.sqrt(squared);
        let minimum = lane(simd, &pairs.minimum, start);
        let minimum_seventh = lane(simd, &pairs.minimum_seventh, start);

        let buffered_near = distance + splat(NEAR - 1.0) * minimum;
        let attenuated = splat(NEAR) * minimum / buffered_near;
        let cubed = attenuated * attenuated * attenuated;
        let seventh = cubed * cubed * attenuated;
        let sixth = squared * squared * squared;
        let repulsion =
            splat(FAR) * minimum_seventh / (sixth * distance + splat(FAR - 1.0) * minimum_seventh);

        van_der_waals += lane(simd, &pairs.well, start) * seventh * (repulsion - splat(2.0));
        electrostatic += splat(COULOMB) * lane(simd, &pairs.charge_product, start)
            / (distance + splat(ELECTROSTATIC_BUFFER));
        start += LANES;
    }

    let (mut van_der_waals, mut electrostatic) = (
        horizontal_sum(simd.to_array(van_der_waals)),
        horizontal_sum(simd.to_array(electrostatic)),
    );
    for index in whole..pairs.len() {
        let (repulsive, coulombic) = nonbonded_energy(positions, &pairs.get(index));
        van_der_waals += repulsive;
        electrostatic += coulombic;
    }
    (van_der_waals, electrostatic)
}

/// [`nonbonded_energies`], accumulating both interactions' gradients.
#[cfg(test)]
pub(super) fn nonbonded_energies_and_gradient(
    pairs: &NonBondedPairs,
    positions: &[Vector3D],
    gradient: &mut [Vector3D],
) -> (f64, f64) {
    nonbonded_energies_and_gradient_in(Portable, pairs, positions, gradient)
}

/// [`nonbonded_energies_and_gradient`] on the lanes `simd` provides.
#[inline(always)]
pub(super) fn nonbonded_energies_and_gradient_in<L: Lanes>(
    simd: L,
    pairs: &NonBondedPairs,
    positions: &[Vector3D],
    gradient: &mut [Vector3D],
) -> (f64, f64) {
    let splat = |value| simd.splat(value);
    let (mut van_der_waals, mut electrostatic) = (splat(0.0), splat(0.0));
    let whole = pairs.len() - pairs.len() % LANES;
    let mut start = 0;
    while start < whole {
        let ([x, y, z], squared) = lane_offsets(simd, pairs, positions, start);
        let distance = simd.sqrt(squared);
        let minimum = lane(simd, &pairs.minimum, start);
        let minimum_seventh = lane(simd, &pairs.minimum_seventh, start);
        let well = lane(simd, &pairs.well, start);

        // Four reciprocals, each used twice, where the scalar form divides
        // seven times: the divider, not the multipliers, is what bounds
        // this loop.
        let buffered_near = distance + splat(NEAR - 1.0) * minimum;
        let inverse_near = splat(1.0) / buffered_near;
        let attenuated = splat(NEAR) * minimum * inverse_near;
        let cubed = attenuated * attenuated * attenuated;
        let sixth_attenuated = cubed * cubed;
        let seventh = sixth_attenuated * attenuated;
        let sixth = squared * squared * squared;
        let denominator = sixth * distance + splat(FAR - 1.0) * minimum_seventh;
        let inverse_denominator = splat(1.0) / denominator;
        let repulsion = splat(FAR) * minimum_seventh * inverse_denominator;

        let d_attenuated = -attenuated * inverse_near;
        let d_repulsion = -repulsion * splat(7.0) * sixth * inverse_denominator;
        let derivative = well
            * (splat(7.0) * sixth_attenuated * d_attenuated * (repulsion - splat(2.0))
                + seventh * d_repulsion);

        let inverse_far = splat(1.0) / (distance + splat(ELECTROSTATIC_BUFFER));
        let scaled = splat(COULOMB) * lane(simd, &pairs.charge_product, start) * inverse_far;
        let scale = (derivative - scaled * inverse_far) / distance;
        // Two atoms on top of each other have no direction to push along.
        // Their energies are still finite, so only the gradient is dropped.
        let scale = simd.where_greater(distance, splat(DEGENERATE_SEPARATION), scale);

        van_der_waals += well * seventh * (repulsion - splat(2.0));
        electrostatic += scaled;

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

    let (mut van_der_waals, mut electrostatic) = (
        horizontal_sum(simd.to_array(van_der_waals)),
        horizontal_sum(simd.to_array(electrostatic)),
    );
    for index in whole..pairs.len() {
        let (repulsive, coulombic) =
            nonbonded_energy_and_gradient(positions, &pairs.get(index), gradient);
        van_der_waals += repulsive;
        electrostatic += coulombic;
    }
    (van_der_waals, electrostatic)
}

/// The four lanes added up, pairwise so the order is at least fixed.
#[inline(always)]
fn horizontal_sum(lanes: [f64; LANES]) -> f64 {
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// Below this separation a pair has no gradient direction, only an energy.
const DEGENERATE_SEPARATION: f64 = 1e-8;


#[cfg(test)]
mod tests {
    use super::*;

    /// The lanes have to agree with the scalar term they replace, remainder
    /// included, or the force field has quietly become a different one.
    /// Ten pairs is not a multiple of the lane width on purpose.
    #[test]
    fn the_vectorised_pairs_agree_with_the_scalar_ones() {
        let positions: Vec<Vector3D> = (0..6)
            .map(|index| {
                let t = index as f64 * 0.9;
                Vector3D::new(t.sin() * 3.0, t.cos() * 3.0, index as f64 * 0.7)
            })
            .collect();
        let mut pairs = NonBondedPairs::default();
        let mut scalar = Vec::new();
        for i in 0..6usize {
            for j in (i + 1)..6usize {
                if scalar.len() == 10 {
                    break;
                }
                let pair = NonBonded::new(i, j, 3.2 + 0.1 * i as f64, 0.08 + 0.01 * j as f64, 0.02 * (i as f64 - j as f64));
                pairs.push(pair);
                scalar.push(pair);
            }
        }
        assert_eq!(pairs.len(), 10, "ten pairs, two past two full lanes");

        let (mut want_vdw, mut want_electrostatic) = (0.0, 0.0);
        let mut want_gradient = vec![Vector3D::default(); positions.len()];
        for pair in &scalar {
            let (repulsive, coulombic) =
                nonbonded_energy_and_gradient(&positions, pair, &mut want_gradient);
            want_vdw += repulsive;
            want_electrostatic += coulombic;
        }

        let (vdw, electrostatic) = nonbonded_energies(&pairs, &positions);
        assert!((vdw - want_vdw).abs() < 1e-9 * want_vdw.abs().max(1.0), "{vdw} against {want_vdw}");
        assert!(
            (electrostatic - want_electrostatic).abs() < 1e-9 * want_electrostatic.abs().max(1.0),
            "{electrostatic} against {want_electrostatic}"
        );

        let mut gradient = vec![Vector3D::default(); positions.len()];
        let (vdw, electrostatic) =
            nonbonded_energies_and_gradient(&pairs, &positions, &mut gradient);
        assert!((vdw - want_vdw).abs() < 1e-9 * want_vdw.abs().max(1.0));
        assert!((electrostatic - want_electrostatic).abs() < 1e-9 * want_electrostatic.abs().max(1.0));
        for (got, want) in gradient.iter().zip(&want_gradient) {
            for axis in [got.x - want.x, got.y - want.y, got.z - want.z] {
                assert!(axis.abs() < 1e-9, "gradient differs by {axis}");
            }
        }
    }

    /// The AVX lanes are the portable ones in a wider register: same
    /// operations, same order, so the same bits. Anything less and the
    /// embedding would depend on which machine ran it. Includes a
    /// coincident pair, which is where `0 - x` and a sign flip part ways.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_avx_lanes_agree_with_the_portable_ones_to_the_bit() {
        let Some(simd) = pulp::x86::V3::try_new() else {
            return;
        };
        let mut positions: Vec<Vector3D> = (0..23)
            .map(|index| {
                let t = index as f64 * 0.77;
                Vector3D::new(t.sin() * 4.0, t.cos() * 3.0, index as f64 * 0.31)
            })
            .collect();
        positions[7] = positions[3];
        let mut pairs = NonBondedPairs::default();
        for i in 0..positions.len() {
            for j in (i + 2)..positions.len() {
                pairs.push(NonBonded::new(i, j, 3.0 + 0.05 * i as f64, 0.05 + 0.003 * j as f64, 0.01 * (i as f64 - j as f64)));
            }
        }

        let portable = nonbonded_energies_in(Portable, &pairs, &positions);
        let avx = simd.vectorize(|| nonbonded_energies_in(Avx(simd), &pairs, &positions));
        assert_eq!(portable.0.to_bits(), avx.0.to_bits());
        assert_eq!(portable.1.to_bits(), avx.1.to_bits());

        let mut portable_gradient = vec![Vector3D::default(); positions.len()];
        let mut avx_gradient = portable_gradient.clone();
        let portable =
            nonbonded_energies_and_gradient_in(Portable, &pairs, &positions, &mut portable_gradient);
        let avx = simd.vectorize(|| {
            nonbonded_energies_and_gradient_in(Avx(simd), &pairs, &positions, &mut avx_gradient)
        });
        assert_eq!(portable.0.to_bits(), avx.0.to_bits());
        assert_eq!(portable.1.to_bits(), avx.1.to_bits());
        for (one, other) in portable_gradient.iter().zip(&avx_gradient) {
            for (a, b) in [(one.x, other.x), (one.y, other.y), (one.z, other.z)] {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    /// Two atoms in the same place keep a finite energy and gain no
    /// gradient, in lanes as in scalar.
    #[test]
    fn coincident_atoms_have_energy_and_no_direction() {
        let positions = vec![Vector3D::default(); 5];
        let mut pairs = NonBondedPairs::default();
        for index in 0..5usize {
            pairs.push(NonBonded::new(index, (index + 1) % 5, 3.4, 0.1, 0.05));
        }
        let mut gradient = vec![Vector3D::default(); positions.len()];
        let (vdw, electrostatic) =
            nonbonded_energies_and_gradient(&pairs, &positions, &mut gradient);
        assert!(vdw.is_finite() && electrostatic.is_finite(), "{vdw} {electrostatic}");
        for slot in &gradient {
            assert_eq!((slot.x, slot.y, slot.z), (0.0, 0.0, 0.0));
        }
    }

    #[test]
    fn collinear_angles_retain_bending_and_stretch_bend_energy() {
        let positions = [
            Vector3D::new(-1.2, 0.0, 0.0),
            Vector3D::default(),
            Vector3D::new(1.2, 0.0, 0.0),
        ];
        let mut nearby = positions;
        nearby[2].y = 1e-5;
        let mut gradient = [Vector3D::default(); 3];
        let angle = Angle {
            i: 0,
            central: 1,
            k: 2,
            force_constant: 1.0,
            angle: 109.5,
            linear: false,
            stretch: Some(StretchBend {
                first_constant: 0.2,
                second_constant: 0.3,
                first_length: 1.0,
                second_length: 1.0,
            }),
        };
        let (bend, stretch) = angle_energies_and_gradient(&positions, &angle, &mut gradient);
        assert!(bend > 1.0 && stretch > 1.0);
        assert_eq!((bend, stretch), angle_energies(&positions, &angle));
        let (near_bend, near_stretch) = angle_energies(&nearby, &angle);
        assert!((bend - near_bend).abs() < 1e-3);
        assert!((stretch - near_stretch).abs() < 1e-3);

        // At exact collinearity the radial derivative remains well-defined.
        let mut plus = positions;
        let mut minus = positions;
        plus[0].x += 1e-6;
        minus[0].x -= 1e-6;
        let numeric = (angle_energies(&plus, &angle).1 - angle_energies(&minus, &angle).1) / 2e-6;
        // The bending term contributes to the same component, so compare
        // against the stretch-bend gradient alone: run it again with the
        // coupling off and take the difference.
        let bending_only = Angle { stretch: None, ..angle };
        let mut bending_gradient = [Vector3D::default(); 3];
        angle_energies_and_gradient(&positions, &bending_only, &mut bending_gradient);
        assert!((numeric - (gradient[0].x - bending_gradient[0].x)).abs() < 1e-6);
    }


    #[test]
    fn overlapping_atoms_keep_the_same_energy_in_gradient_evaluation() {
        let positions = [Vector3D::default(); 2];
        let mut gradient = [Vector3D::default(); 2];
        let bond = Bond {
            i: 0,
            j: 1,
            length: 1.5,
            force_constant: 1.0,
        };
        assert!(bond_energy(&positions, &bond) > 1.0);
        assert_eq!(
            bond_energy(&positions, &bond),
            bond_energy_and_gradient(&positions, &bond, &mut gradient)
        );
        let pair = NonBonded::new(0, 1, 3.5, 0.1, -1.0);
        let (repulsive, electrostatic) = nonbonded_energy(&positions, &pair);
        assert!(repulsive > 1e6);
        let combined = nonbonded_energy_and_gradient(&positions, &pair, &mut gradient);
        assert!((repulsive - combined.0).abs() < 1e-12 * repulsive);
        assert_eq!(electrostatic, combined.1);
        assert!(gradient.iter().all(|g| g.length().is_finite()));
    }
}
