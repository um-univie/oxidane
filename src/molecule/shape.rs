//! Shape descriptors: USR, USRCAT, and the moments of inertia.
//!
//! Everything else in [`crate::molecule::fingerprint`] describes a molecule's
//! *graph*. These describe one conformer of it, and answer a different
//! question: not "does it contain the same pieces" but "would it fit in the
//! same hole". Two molecules with no substructure in common can be shape
//! neighbours, which is what makes this the tool for scaffold hopping and what
//! a graph fingerprint can never do.
//!
//! [`usr`] is Ballester and Richards' ultrafast shape recognition: pick four
//! reference points from the atom positions alone, take the distance from each
//! to every atom, and keep the first three moments of each of those four
//! distributions. Twelve numbers, no alignment, no optimisation --
//! comparing two molecules is twelve subtractions. [`usrcat`] is Schreyer and
//! Blundell's extension: the same four reference points, but the moments are
//! taken over pharmacophoric subsets of the atoms as well as over all of them,
//! so that shape and chemistry are asked about together. Sixty numbers, and the
//! subsets can be weighted at comparison time.
//!
//! The rest -- [`shape_descriptors`] -- are the classical inertial ones: where
//! the mass sits, and how far from a sphere, a rod and a disc the arrangement
//! is.
//!
//! Positions are supplied by the caller, indexed by atom, so that a conformer
//! from any source can be used. **Hydrogens count**, unlike everywhere else in
//! this crate: a hydrogen occupies space, and every published implementation of
//! these descriptors includes whatever atoms the conformer has.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::shape::{usr, usr_score};
//!
//! let molecule = Molecule3D::from_sdf("tests/pentene.sdf").unwrap().remove(0);
//! let descriptor = usr(&molecule.positions);
//! assert_eq!(descriptor.len(), 12);
//! // A molecule is the same shape as itself.
//! assert_eq!(usr_score(&descriptor, &descriptor), 1.0);
//! ```

use crate::io::{MoleculeMatchContext, parse_cxsmarts};
use crate::molecule::base::MoleculeCore;
use crate::molecule::molecular_graph::MolecularGraph;
use crate::vector::Vector3D;
use chemistry_consts::ElementProperties;
use std::sync::LazyLock;

/// How many numbers a USR descriptor has: three moments of each of four
/// distance distributions.
pub const USR_LENGTH: usize = 12;

/// How many numbers a USRCAT descriptor has: [`USR_LENGTH`] for all the atoms
/// and again for each of the four pharmacophoric subsets.
pub const USRCAT_LENGTH: usize = USR_LENGTH * 5;

/// The four subsets USRCAT adds, in the order they appear in the descriptor.
///
/// These are the definitions of Schreyer and Blundell as RDKit ships them, and
/// they are quoted rather than paraphrased so that a descriptor from this crate
/// can be compared against one from any other implementation. What they mean
/// depends on the molecule's perceived aromaticity, as everything here does.
pub const USRCAT_SUBSETS: [(&str, &str); 4] = [
    (
        "hydrophobic",
        "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",
    ),
    ("aromatic", "[a]"),
    (
        "acceptor",
        "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),\
         $([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]",
    ),
    ("donor", "[N!H0v3,N!H0+v4,OH+0,SH+0,nH+0]"),
];

/// The subset patterns, parsed once.
static SUBSET_PATTERNS: LazyLock<Vec<crate::io::CxSMARTSPattern>> = LazyLock::new(|| {
    USRCAT_SUBSETS
        .iter()
        .map(|(name, smarts)| {
            parse_cxsmarts(smarts)
                .unwrap_or_else(|error| panic!("USRCAT {name} ({smarts}) does not parse: {error}"))
        })
        .collect()
});

/// The four reference points USR measures from.
///
/// The centroid; the atom nearest it; the atom furthest from it; and the atom
/// furthest from *that*. All four are decided by the positions alone, so two
/// conformers of the same molecule are compared on the same footing without
/// being aligned first -- which is the whole point of the method.
pub fn usr_reference_points(positions: &[Vector3D]) -> [Vector3D; 4] {
    if positions.is_empty() {
        return [Vector3D::default(); 4];
    }
    let mut centroid = Vector3D::default();
    for position in positions {
        centroid = centroid + *position;
    }
    let count = positions.len() as f64;
    let centroid = Vector3D {
        x: centroid.x / count,
        y: centroid.y / count,
        z: centroid.z / count,
    };

    let nearest = extreme(positions, &centroid, true);
    let furthest = extreme(positions, &centroid, false);
    let opposite = extreme(positions, &positions[furthest], false);
    [
        centroid,
        positions[nearest],
        positions[furthest],
        positions[opposite],
    ]
}

/// The position nearest to, or furthest from, `point`.
fn extreme(positions: &[Vector3D], point: &Vector3D, nearest: bool) -> usize {
    let mut best = 0usize;
    let mut best_distance = f64::INFINITY;
    for (index, position) in positions.iter().enumerate() {
        let mut distance = separation(position, point);
        if !nearest {
            distance = -distance;
        }
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    best
}

fn separation(first: &Vector3D, second: &Vector3D) -> f64 {
    let (x, y, z) = (
        first.x - second.x,
        first.y - second.y,
        first.z - second.z,
    );
    (x * x + y * y + z * z).sqrt()
}

/// The USR descriptor of a set of positions.
///
/// Twelve numbers: for each of the four points of [`usr_reference_points`], the
/// mean distance to the atoms, the standard deviation of those distances, and
/// the cube root of their skewness. The cube root is what puts the third moment
/// on the same scale as the other two, so that a plain sum of absolute
/// differences is a sensible comparison -- which is what [`usr_score`] takes.
pub fn usr(positions: &[Vector3D]) -> Vec<f64> {
    let points = usr_reference_points(positions);
    usr_from_points(positions, &points)
}

/// The USR descriptor of `positions` measured from reference points worked out
/// somewhere else.
///
/// This is how USRCAT measures a subset: the four points come from *all* the
/// atoms, and only the distributions are restricted, so that the subsets are
/// all measured in the same frame and remain comparable.
///
/// A subset with no atoms in it contributes twelve zeros, which is what every
/// implementation does and what makes a molecule with no aromatic ring
/// comparable to one that has one.
pub fn usr_from_points(positions: &[Vector3D], points: &[Vector3D; 4]) -> Vec<f64> {
    let mut descriptor = Vec::with_capacity(USR_LENGTH);
    let mut distances: Vec<f64> = Vec::with_capacity(positions.len());
    for point in points {
        distances.clear();
        distances.extend(positions.iter().map(|position| separation(position, point)));
        descriptor.extend(moments(&distances));
    }
    descriptor
}

/// Mean, standard deviation, and the cube root of the skewness.
///
/// The population forms of all three -- divided by `n`, not by `n - 1`. The
/// skewness is standardised (the third central moment over the cube of the
/// standard deviation) before the cube root is taken; taking the cube root of
/// the raw third moment instead gives a number that grows with the molecule and
/// makes the descriptor incomparable between sizes.
fn moments(values: &[f64]) -> [f64; 3] {
    if values.is_empty() {
        return [0.0; 3];
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let mut second = 0.0;
    let mut third = 0.0;
    for value in values {
        let offset = value - mean;
        second += offset * offset;
        third += offset * offset * offset;
    }
    second /= count;
    third /= count;
    let deviation = second.sqrt();
    let skewness = if deviation == 0.0 {
        0.0
    } else {
        (third / (deviation * deviation * deviation)).cbrt()
    };
    [mean, deviation, skewness]
}

/// The USRCAT descriptor: USR over all the atoms, then over each of
/// [`USRCAT_SUBSETS`].
///
/// `positions` is indexed by atom and must cover the whole molecule.
pub fn usrcat(molecule: &MolecularGraph, positions: &[Vector3D]) -> Vec<f64> {
    let points = usr_reference_points(positions);
    let mut descriptor = usr_from_points(positions, &points);

    let context = MoleculeMatchContext::new(molecule);
    let mut subset: Vec<Vector3D> = Vec::new();
    for pattern in SUBSET_PATTERNS.iter() {
        // A pattern here names one atom, so every match is one atom; a pattern
        // that named two would have its first atom taken, which is what the
        // reference implementations do.
        let mut atoms: Vec<usize> = pattern
            .find_matches_with(molecule, &context)
            .into_iter()
            .filter_map(|atoms| atoms.first().copied())
            .collect();
        atoms.sort_unstable();
        atoms.dedup();

        subset.clear();
        subset.extend(
            atoms
                .into_iter()
                .filter_map(|atom| positions.get(atom).copied()),
        );
        descriptor.extend(usr_from_points(&subset, &points));
    }
    descriptor
}

/// How alike two USR descriptors are, on a scale where 1 is identical.
///
/// `1 / (1 + mean absolute difference)`, the similarity Ballester and Richards
/// defined. It is not a Tanimoto and does not behave like one: 0.7 is already
/// a poor match, and the cutoffs quoted in the literature are near 0.9.
pub fn usr_score(first: &[f64], second: &[f64]) -> f64 {
    usrcat_score(first, second, &[1.0])
}

/// How alike two USRCAT descriptors are, weighting each subset.
///
/// One weight per block of [`USR_LENGTH`] numbers, so five for a USRCAT
/// descriptor: all atoms, hydrophobic, aromatic, acceptor, donor. Weighting the
/// shape term alone reduces to [`usr_score`]; weighting the pharmacophoric
/// terms up asks for a molecule that puts the same features in the same places
/// rather than merely filling the same volume.
///
/// Descriptors of different lengths, or a wrong number of weights, score zero.
pub fn usrcat_score(first: &[f64], second: &[f64], weights: &[f64]) -> f64 {
    if first.len() != second.len() || first.is_empty() {
        return 0.0;
    }
    if first.len() != weights.len() * USR_LENGTH {
        return 0.0;
    }
    let mut total = 0.0;
    for (block, weight) in weights.iter().enumerate() {
        let start = block * USR_LENGTH;
        for index in start..start + USR_LENGTH {
            total += weight * (first[index] - second[index]).abs();
        }
    }
    1.0 / (1.0 + total / USR_LENGTH as f64)
}

/// The classical shape descriptors of one conformer.
///
/// All of them come from the same place: the moments of inertia of the atoms
/// about the centre of mass, and the same moments taken with every atom
/// weighted equally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeDescriptors {
    /// The smallest principal moment of inertia.
    pub first_moment: f64,
    /// The middle one.
    pub second_moment: f64,
    /// The largest.
    pub third_moment: f64,
    /// `first / third`. With [`ShapeDescriptors::second_ratio`] this is the
    /// point a molecule occupies on the triangular plot whose corners are a
    /// rod, a disc and a sphere.
    pub first_ratio: f64,
    /// `second / third`.
    pub second_ratio: f64,
    /// How far the mass sits from the centre, in angstroms.
    pub radius_of_gyration: f64,
    /// How far from spherical the arrangement of mass is: 0 when the spread is
    /// the same in all three directions, 1 when it is along one axis only.
    pub asphericity: f64,
    /// How far from spherical the *extent* is, by the two outer moments alone.
    pub eccentricity: f64,
    /// `second / (first * third)`, which separates flat molecules from long
    /// ones where the ratios alone do not.
    pub inertial_shape_factor: f64,
    /// How spherical the arrangement is with every atom weighted equally: 1 for
    /// a sphere, 0 for anything planar.
    pub spherocity_index: f64,
    /// The mean distance of the atoms from their plane of best fit. Zero for a
    /// flat molecule, and the quickest way to ask how flat one is.
    pub plane_of_best_fit: f64,
}

/// The shape descriptors of one conformer.
///
/// `positions` is indexed by atom. Masses come from the elements, so an isotope
/// label is ignored -- two samples of the same compound have the same shape.
pub fn shape_descriptors<M: MoleculeCore + ?Sized>(
    molecule: &M,
    positions: &[Vector3D],
) -> ShapeDescriptors {
    let count = molecule.len().min(positions.len());
    let masses: Vec<f64> = (0..count)
        .map(|atom| {
            molecule
                .get_atomic_number(atom)
                .standard_atomic_weight()
                .unwrap_or(0.0)
        })
        .collect();
    let positions = &positions[..count];

    // Everything here comes out of one tensor: the mass-weighted spread of the
    // atoms about their centre of mass. The moments of inertia are its trace
    // less each of its eigenvalues, which is why they are not computed
    // separately.
    let spread = gyration_tensor(positions, &masses);
    let spreads = symmetric_eigenvalues_sorted(&spread);
    let trace = spreads[0] + spreads[1] + spreads[2];
    // Smallest moment against largest spread: an atom far out along one axis
    // contributes nothing to the moment about that axis.
    let [first, second, third] = [
        trace - spreads[2],
        trace - spreads[1],
        trace - spreads[0],
    ];

    let total_mass: f64 = masses.iter().sum();
    let radius_of_gyration = if total_mass > 0.0 {
        (trace / total_mass).sqrt()
    } else {
        0.0
    };
    // Asphericity is a property of the spread rather than of the moments: zero
    // when the three are equal, one when two of them vanish.
    let asphericity = if trace > 0.0 {
        ((spreads[0] - spreads[1]).powi(2)
            + (spreads[0] - spreads[2]).powi(2)
            + (spreads[1] - spreads[2]).powi(2))
            / (2.0 * trace * trace)
    } else {
        0.0
    };
    let eccentricity = if third > 0.0 {
        ((third * third - first * first).max(0.0)).sqrt() / third
    } else {
        0.0
    };
    let inertial_shape_factor = if first > 0.0 && third > 0.0 {
        second / (first * third)
    } else {
        0.0
    };
    // Spherocity asks about the arrangement rather than the mass, so every atom
    // counts once. A planar molecule has no spread at all along its normal and
    // scores zero however its mass is distributed.
    let uniform = symmetric_eigenvalues_sorted(&gyration_tensor(positions, &vec![1.0; count]));
    let uniform_trace = uniform[0] + uniform[1] + uniform[2];
    let spherocity_index = if uniform_trace > 0.0 {
        3.0 * uniform[0] / uniform_trace
    } else {
        0.0
    };

    ShapeDescriptors {
        first_moment: first,
        second_moment: second,
        third_moment: third,
        first_ratio: if third > 0.0 { first / third } else { 0.0 },
        second_ratio: if third > 0.0 { second / third } else { 0.0 },
        radius_of_gyration,
        asphericity,
        eccentricity,
        inertial_shape_factor,
        spherocity_index,
        plane_of_best_fit: plane_of_best_fit(positions),
    }
}

/// The three principal moments of inertia about the centre of mass, smallest
/// first.
pub fn principal_moments(positions: &[Vector3D], weights: &[f64]) -> [f64; 3] {
    let spreads = symmetric_eigenvalues_sorted(&gyration_tensor(positions, weights));
    let trace = spreads[0] + spreads[1] + spreads[2];
    [
        trace - spreads[2],
        trace - spreads[1],
        trace - spreads[0],
    ]
}

/// The weighted spread of the positions about their weighted centre: the
/// symmetric tensor whose entry `(a, b)` is the sum of `w * r_a * r_b`.
///
/// The moments of inertia are its trace less its eigenvalues, so one of these
/// answers every question below.
fn gyration_tensor(positions: &[Vector3D], weights: &[f64]) -> [[f64; 3]; 3] {
    let count = positions.len().min(weights.len());
    let total: f64 = weights[..count].iter().sum();
    if count == 0 || total <= 0.0 {
        return [[0.0; 3]; 3];
    }

    let mut centre = Vector3D::default();
    for (position, weight) in positions[..count].iter().zip(weights.iter()) {
        centre = centre
            + Vector3D {
                x: position.x * weight,
                y: position.y * weight,
                z: position.z * weight,
            };
    }
    let centre = Vector3D {
        x: centre.x / total,
        y: centre.y / total,
        z: centre.z / total,
    };

    let (mut xx, mut yy, mut zz) = (0.0, 0.0, 0.0);
    let (mut xy, mut xz, mut yz) = (0.0, 0.0, 0.0);
    for (position, weight) in positions[..count].iter().zip(weights.iter()) {
        let (x, y, z) = (
            position.x - centre.x,
            position.y - centre.y,
            position.z - centre.z,
        );
        xx += weight * x * x;
        yy += weight * y * y;
        zz += weight * z * z;
        xy += weight * x * y;
        xz += weight * x * z;
        yz += weight * y * z;
    }
    [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
}

/// [`symmetric_eigenvalues`], smallest first.
fn symmetric_eigenvalues_sorted(matrix: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut values = symmetric_eigenvalues(*matrix);
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    values
}

/// The eigenvalues of a real symmetric 3x3 matrix, by the closed form.
///
/// Smith's method: subtract the mean of the diagonal, whose remainder has a
/// determinant that gives the three roots as cosines. No iteration, no
/// convergence to worry about, and exact for the degenerate cases a symmetric
/// molecule produces -- which is where an iterative Jacobi sweep is least
/// pleasant.
fn symmetric_eigenvalues(matrix: [[f64; 3]; 3]) -> [f64; 3] {
    let trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
    let mean = trace / 3.0;
    let mut shifted = matrix;
    for index in 0..3 {
        shifted[index][index] -= mean;
    }

    let mut square_sum = 0.0;
    for row in &shifted {
        for value in row {
            square_sum += value * value;
        }
    }
    let scale = (square_sum / 6.0).sqrt();
    if scale <= f64::EPSILON {
        return [mean; 3];
    }

    let determinant = determinant3(&shifted) / (scale * scale * scale);
    // Clamped because rounding can push it a hair outside the range where the
    // arc cosine is defined, and a degenerate matrix sits exactly on the edge.
    let angle = (determinant / 2.0).clamp(-1.0, 1.0).acos() / 3.0;
    let first = mean + 2.0 * scale * angle.cos();
    let third = mean + 2.0 * scale * (angle - 2.0 * std::f64::consts::PI / 3.0).cos();
    let second = trace - first - third;
    [first, second, third]
}

fn determinant3(matrix: &[[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

/// The mean distance of the atoms from the plane that fits them best.
///
/// The best plane through a set of points passes through their centroid with
/// its normal along the direction of least spread, which is the eigenvector of
/// the smallest eigenvalue of their spread tensor. Zero for a flat molecule,
/// and the quickest way to ask how flat one is.
///
/// Hydrogens count, as they do everywhere in this module: they are part of the
/// surface a flat molecule presents, and leaving them out gives a different
/// number from every published implementation.
pub fn plane_of_best_fit(positions: &[Vector3D]) -> f64 {
    if positions.len() < 3 {
        return 0.0;
    }
    let count = positions.len() as f64;
    let mut centre = Vector3D::default();
    for position in positions {
        centre = centre + *position;
    }
    let centre = Vector3D {
        x: centre.x / count,
        y: centre.y / count,
        z: centre.z / count,
    };

    let spread = gyration_tensor(positions, &vec![1.0; positions.len()]);
    let smallest = symmetric_eigenvalues_sorted(&spread)[0];
    let Some(normal) = null_space_direction(&spread, smallest) else {
        return 0.0;
    };
    positions
        .iter()
        .map(|position| {
            ((position.x - centre.x) * normal.x
                + (position.y - centre.y) * normal.y
                + (position.z - centre.z) * normal.z)
                .abs()
        })
        .sum::<f64>()
        / count
}

/// A unit vector `v` with `(matrix - eigenvalue * I) v = 0`.
///
/// Taken as the cross product of two rows of the shifted matrix: any two
/// independent rows span the plane the eigenvector is normal to, and the pair
/// whose cross product is longest is the pair that is most independent, which
/// is the numerically safe choice.
fn null_space_direction(matrix: &[[f64; 3]; 3], eigenvalue: f64) -> Option<Vector3D> {
    let mut shifted = *matrix;
    for index in 0..3 {
        shifted[index][index] -= eigenvalue;
    }
    let mut best: Option<(f64, Vector3D)> = None;
    for first in 0..3 {
        for second in (first + 1)..3 {
            let (left, right) = (shifted[first], shifted[second]);
            let candidate = Vector3D {
                x: left[1] * right[2] - left[2] * right[1],
                y: left[2] * right[0] - left[0] * right[2],
                z: left[0] * right[1] - left[1] * right[0],
            };
            let length =
                (candidate.x * candidate.x + candidate.y * candidate.y + candidate.z * candidate.z)
                    .sqrt();
            if best.as_ref().is_none_or(|(longest, _)| length > *longest) {
                best = Some((length, candidate));
            }
        }
    }
    let (length, direction) = best?;
    if length <= f64::EPSILON {
        return None;
    }
    Some(Vector3D {
        x: direction.x / length,
        y: direction.y / length,
        z: direction.z / length,
    })
}
