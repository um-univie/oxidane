//! Which parameters each bond, angle, torsion and pair gets.
//!
//! MMFF94 keys its parameters on the atom types plus a small integer saying
//! what kind of bond, angle or torsion this is -- whether it joins two
//! multiply bonded atoms, whether it sits in a three- or four-membered ring.
//! Where the table has no entry for the exact types it is consulted again
//! with progressively vaguer stand-ins, and where even the wild card misses,
//! an empirical rule computes the parameter from covalent radii and
//! electronegativities instead. All three layers are here.
//!
//! The ring questions are answered from the bonds rather than from the ring
//! perception: an angle is in a three-ring when its outer atoms are bonded,
//! and in a four-ring when they share a neighbour that is not the centre.
//! That is what RDKit asks, and it is a different question from ring
//! membership -- a four-ring made of two fused three-rings answers no.

use super::params::{
    self, AngleParameters, BondParameters, StretchBendParameters, TorsionParameters,
};
use super::perception::Perception;
use crate::molecule::bond::BondOrder;

/// Degrees to radians, which MMFF needs in several places because its angle
/// parameters are in degrees and its force constants are not.
pub(super) const DEGREES_TO_RADIANS: f64 = std::f64::consts::PI / 180.0;

/// MMFF's bond type index: 1 for a single bond between two atoms that both
/// multiply bond, or between two aromatic ones -- biphenyl's central bond,
/// butadiene's middle bond -- and 0 for every other bond.
pub(super) fn bond_type(perception: &Perception, types: &[u8], first: usize, second: usize) -> u8 {
    if perception.order(first, second) != BondOrder::Single {
        return 0;
    }
    let (Some(one), Some(two)) = (
        params::properties(types[first]),
        params::properties(types[second]),
    ) else {
        return 0;
    };
    u8::from(
        (one.single_bond_between_multiply_bonded && two.single_bond_between_multiply_bonded)
            || (one.aromatic && two.aromatic),
    )
}

fn bonded(perception: &Perception, first: usize, second: usize) -> bool {
    perception
        .neighbours(first)
        .iter()
        .any(|neighbour| neighbour.atom == second)
}

/// 3 or 4 if the angle closes a ring of that size, 0 otherwise.
fn angle_ring_size(perception: &Perception, first: usize, central: usize, last: usize) -> u8 {
    if bonded(perception, first, last) {
        return 3;
    }
    let shares_a_neighbour = perception.neighbours(first).iter().any(|one| {
        one.atom != central
            && perception
                .neighbours(last)
                .iter()
                .any(|other| other.atom != central && other.atom == one.atom)
    });
    if shares_a_neighbour { 4 } else { 0 }
}

/// 4 or 5 if the torsion closes a ring of that size, 0 otherwise.
fn torsion_ring_size(
    perception: &Perception,
    first: usize,
    second: usize,
    third: usize,
    fourth: usize,
) -> u8 {
    if bonded(perception, first, fourth) {
        return 4;
    }
    let shares_a_neighbour = perception.neighbours(first).iter().any(|one| {
        one.atom != second
            && perception
                .neighbours(fourth)
                .iter()
                .any(|other| other.atom != third && other.atom == one.atom)
    });
    if shares_a_neighbour { 5 } else { 0 }
}

/// MMFF's angle type: the sum of the two bond types, raised into a different
/// range when the angle is in a three- or four-membered ring.
pub(super) fn angle_type(
    perception: &Perception,
    types: &[u8],
    first: usize,
    central: usize,
    last: usize,
) -> u8 {
    let bond_types =
        bond_type(perception, types, first, central) + bond_type(perception, types, central, last);
    let ring_size = angle_ring_size(perception, first, central, last);
    if ring_size == 0 {
        return bond_types;
    }
    if bond_types == 0 {
        return ring_size;
    }
    ring_size + bond_types + ring_size - 2
}

/// MMFF's stretch-bend type, which the angle type and the two bond types
/// decide between them.
pub(super) fn stretch_bend_type(angle_type: u8, first_bond: u8, second_bond: u8) -> u8 {
    let asymmetric = |low: u8, high: u8| {
        if first_bond != 0 || first_bond == second_bond { low } else { high }
    };
    match angle_type {
        1 => asymmetric(1, 2),
        2 => 3,
        3 => 5,
        4 => 4,
        5 => asymmetric(6, 7),
        6 => 8,
        7 => asymmetric(9, 10),
        8 => 11,
        _ => 0,
    }
}

/// MMFF's torsion type, and the one to fall back on: a torsion in a
/// five-ring is typed as such, but if no parameters exist for that, the type
/// it would have had outside the ring is tried instead.
pub(super) fn torsion_types(
    perception: &Perception,
    types: &[u8],
    first: usize,
    second: usize,
    third: usize,
    fourth: usize,
) -> (u8, u8) {
    let first_bond = bond_type(perception, types, first, second);
    let central_bond = bond_type(perception, types, second, third);
    let last_bond = bond_type(perception, types, third, fourth);

    let mut torsion_type = central_bond;
    // MMFF.IV would have this without the single-bond test; RDKit found the
    // test necessary against the validation suite, and this follows RDKit.
    if central_bond == 0
        && perception.order(second, third) == BondOrder::Single
        && (first_bond == 1 || last_bond == 1)
    {
        torsion_type = 2;
    }

    let ring_size = torsion_ring_size(perception, first, second, third, fourth);
    // A four-ring built from two fused three-rings is not the four-ring this
    // type is for, and the extra bond is how to tell.
    if ring_size == 4 && !bonded(perception, first, third) && !bonded(perception, second, fourth) {
        return (4, torsion_type);
    }
    if ring_size == 5
        && [first, second, third, fourth]
            .iter()
            .any(|&atom| types[atom] == 1)
    {
        return (5, torsion_type);
    }
    (torsion_type, 0)
}

/// The periodic row MMFF counts from, with hydrogen and helium at zero.
pub(super) fn periodic_row(atomic_number: u8) -> u8 {
    match atomic_number {
        3..=10 => 1,
        11..=18 => 2,
        19..=36 => 3,
        37..=54 => 4,
        _ => 0,
    }
}

/// The row the Herschbach-Laurie rule counts from, which starts at helium
/// and multiplies a transition metal's row by ten.
fn periodic_row_herschbach_laurie(atomic_number: u8) -> u8 {
    let row = match atomic_number {
        2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        _ => 0,
    };
    if matches!(atomic_number, 21..=30 | 39..=48) { row * 10 } else { row }
}

/// A bond's parameters: the table where it has them, and MMFF.V's empirical
/// rule where it does not.
pub(super) fn bond_parameters(
    perception: &Perception,
    types: &[u8],
    first: usize,
    second: usize,
) -> Option<BondParameters> {
    let kind = bond_type(perception, types, first, second);
    if let Some(parameters) = params::bond(kind, types[first], types[second]) {
        return Some(*parameters);
    }

    let (one, two) = (perception.element(first), perception.element(second));
    let (Some(radius_one), Some(radius_two)) =
        (params::covalent_radius(one), params::covalent_radius(two))
    else {
        return None;
    };

    // Equation 18 of MMFF.V: two covalent radii, shortened by a term in the
    // electronegativity difference, which is smaller when hydrogen is one of
    // the two.
    let shrinkage = if one == 1 || two == 1 { 0.050 } else { 0.085 };
    let length = radius_one.radius + radius_two.radius
        - shrinkage
            * (radius_one.electronegativity - radius_two.electronegativity)
                .abs()
                .powf(1.4);

    // Equation 19: the force constant of a reference bond between the same
    // two elements, scaled by the sixth power of how much this bond differs
    // from it. Without such a reference, Badger's rule in the
    // Herschbach-Laurie form.
    let force_constant = match params::element_bond(one, two) {
        Some(reference) => reference.force_constant * (reference.length / length).powi(6),
        None => {
            let rule = params::herschbach_laurie(
                periodic_row_herschbach_laurie(one),
                periodic_row_herschbach_laurie(two),
            )?;
            10f64.powf(-(length - rule.a) / rule.d)
        }
    };

    Some(BondParameters { force_constant, length })
}

/// An angle's parameters, by table or by MMFF.V's empirical rule. The rule
/// keeps a tabulated equilibrium angle where there was one and only the
/// force constant was missing.
pub(super) fn angle_parameters(
    perception: &Perception,
    types: &[u8],
    first: usize,
    central: usize,
    last: usize,
) -> Option<AngleParameters> {
    let kind = angle_type(perception, types, first, central, last);
    let tabulated = params::angle(kind, types[first], types[central], types[last]);
    // A tabulated angle with no force constant is half an answer: the
    // equilibrium value stands, the stiffness comes from the rule below.
    if let Some(parameters) = tabulated
        && parameters.force_constant != 0.0
    {
        return Some(*parameters);
    }

    let first_bond = bond_parameters(perception, types, first, central)?;
    let last_bond = bond_parameters(perception, types, central, last)?;
    let properties = params::properties(types[central])?;
    let ring_size = angle_ring_size(perception, first, central, last);

    let angle = match tabulated {
        Some(parameters) => parameters.angle,
        None => {
            let mut angle = match properties.connections {
                4 => 109.45,
                2 if perception.element(central) == 8 => 105.0,
                2 if properties.linear => 180.0,
                3 if properties.valence == 3 && properties.multiple_bonds == 0 => {
                    if perception.element(central) == 7 { 107.0 } else { 92.0 }
                }
                _ => 120.0,
            };
            if ring_size == 3 {
                angle = 60.0;
            } else if ring_size == 4 {
                angle = 90.0;
            }
            angle
        }
    };

    // Equation 20 of MMFF.V: an effective nuclear charge for each outer
    // atom, a shell factor for the centre, and a penalty for the two bonds
    // being of different lengths. A strained ring softens the whole thing.
    let charge = |atomic_number: u8| -> f64 { match atomic_number {
        1 => 1.395,
        6 => 2.494,
        7 => 2.711,
        8 => 3.045,
        9 => 2.847,
        14 => 2.350,
        15 => 2.350,
        16 => 2.980,
        17 => 2.909,
        35 => 3.017,
        53 => 3.086,
        _ => 0.0,
    } };
    let shell = |atomic_number: u8| -> f64 { match atomic_number {
        6 => 1.016,
        7 => 1.113,
        8 => 1.337,
        14 => 0.811,
        15 => 1.068,
        16 => 1.249,
        17 => 1.078,
        _ => 0.0,
    } };

    let beta = match ring_size {
        3 => 1.75 * 0.05,
        4 => 1.75 * 0.85,
        _ => 1.75,
    };
    let lengths = first_bond.length + last_bond.length;
    let imbalance = (first_bond.length - last_bond.length) / lengths;
    let radians = DEGREES_TO_RADIANS * angle;
    let force_constant = beta
        * charge(perception.element(first))
        * shell(perception.element(central))
        * charge(perception.element(last))
        / (lengths * radians * radians * (2.0 * imbalance * imbalance).exp());

    Some(AngleParameters { force_constant, angle })
}

/// An angle's stretch-bend parameters, or `None` where the term does not
/// apply -- a linear centre has no stretch-bend coupling, and neither does
/// an angle whose parameters are all zero.
pub(super) fn stretch_bend_parameters(
    perception: &Perception,
    types: &[u8],
    first: usize,
    central: usize,
    last: usize,
) -> Option<StretchBendParameters> {
    let properties = params::properties(types[central])?;
    if properties.linear {
        return None;
    }
    let first_bond = bond_type(perception, types, first, central);
    let last_bond = bond_type(perception, types, central, last);
    let kind = stretch_bend_type(
        angle_type(perception, types, first, central, last),
        if types[first] <= types[last] { first_bond } else { last_bond },
        if types[first] < types[last] { last_bond } else { first_bond },
    );

    let found = params::stretch_bend(
        kind,
        types[first],
        types[central],
        types[last],
        first_bond,
        last_bond,
    )
    .or_else(|| {
        params::default_stretch_bend(
            periodic_row(perception.element(first)),
            periodic_row(perception.element(central)),
            periodic_row(perception.element(last)),
        )
    });

    let (swapped, parameters) = found?;
    if parameters.first == 0.0 && parameters.second == 0.0 {
        return None;
    }
    Some(if swapped {
        StretchBendParameters { first: parameters.second, second: parameters.first }
    } else {
        *parameters
    })
}

/// A torsion's parameters, by table or by the rules of MMFF.IV. `None` means
/// the torsion contributes nothing and is left out entirely.
pub(super) fn torsion_parameters(
    perception: &Perception,
    types: &[u8],
    first: usize,
    second: usize,
    third: usize,
    fourth: usize,
) -> Option<(u8, TorsionParameters)> {
    let (kind, secondary) = torsion_types(perception, types, first, second, third, fourth);
    let (kind, parameters) =
        match params::torsion(kind, secondary, types[first], types[second], types[third], types[fourth])
        {
            Some((found, parameters)) => (found, *parameters),
            None => (kind, empirical_torsion(perception, types, second, third)?),
        };
    if parameters.v1 == 0.0 && parameters.v2 == 0.0 && parameters.v3 == 0.0 {
        return None;
    }
    Some((kind, parameters))
}

/// The empirical torsion rule of MMFF.IV, which reads only the two central
/// atoms: what a torsion costs depends on the bond it turns about, not on
/// what hangs off either end.
fn empirical_torsion(
    perception: &Perception,
    types: &[u8],
    second: usize,
    third: usize,
) -> Option<TorsionParameters> {
    let one = params::properties(types[second])?;
    let two = params::properties(types[third])?;
    let elements = [perception.element(second), perception.element(third)];
    let mut parameters = TorsionParameters { v1: 0.0, v2: 0.0, v3: 0.0 };

    // Table of the per-element constants: `U` scales a conjugation barrier,
    // `V` a threefold one, `W` the case of two chalcogens in a row.
    let constants = |atomic_number: u8| -> (f64, f64, f64) { match atomic_number {
        6 => (2.0, 2.12, 0.0),
        7 => (2.0, 1.5, 0.0),
        8 => (2.0, 0.2, 2.0),
        14 => (1.25, 1.22, 0.0),
        15 => (1.25, 2.40, 0.0),
        16 => (1.25, 0.49, 8.0),
        _ => (0.0, 0.0, 0.0),
    } };
    let (first_u, first_v, first_w) = constants(elements[0]);
    let (second_u, second_v, second_w) = constants(elements[1]);
    let bonds = ((one.connections as f64) - 1.0) * ((two.connections as f64) - 1.0);
    let conjugated = |pi: f64| 6.0 * pi * (first_u * second_u).sqrt();

    if one.linear || two.linear {
        // Rule (a): nothing turns about a linear centre.
    } else if params::is_aromatic_type(types[second])
        && params::is_aromatic_type(types[third])
        && perception.order(second, third) == BondOrder::Aromatic
    {
        // Rule (b): an aromatic bond.
        let beta = if (one.valence == 3 && two.valence == 4)
            || (one.valence == 4 && two.valence == 3)
        {
            3.0
        } else {
            6.0
        };
        let pi = if !one.lone_pair && !two.lone_pair { 0.5 } else { 0.3 };
        parameters.v2 = beta * pi * (first_u * second_u).sqrt();
    } else if perception.order(second, third) == BondOrder::Double {
        // Rule (c): a double bond.
        let pi = if one.multiple_bonds == 2 && two.multiple_bonds == 2 { 1.0 } else { 0.4 };
        parameters.v2 = conjugated(pi);
    } else if one.connections == 4 && two.connections == 4 {
        // Rule (d): two saturated centres.
        parameters.v3 = (first_v * second_v).sqrt() / bonds;
    } else if one.connections == 4 || two.connections == 4 {
        // Rules (e) and (f): one saturated centre, and one that either
        // conjugates -- in which case the barrier is left to the other terms
        // -- or does not.
        let other = if one.connections == 4 { two } else { one };
        let conjugates = (other.connections == 3
            && (other.valence == 4 || other.valence == 34 || other.multiple_bonds != 0))
            || (other.connections == 2 && (other.valence == 3 || other.multiple_bonds != 0));
        if !conjugates {
            parameters.v3 = (first_v * second_v).sqrt() / bonds;
        }
    } else if (perception.order(second, third) == BondOrder::Single
        && one.multiple_bonds != 0
        && two.multiple_bonds != 0)
        || (one.multiple_bonds != 0 && two.lone_pair)
        || (one.lone_pair && two.multiple_bonds != 0)
    {
        // Rule (g): a bond between two conjugated systems, or between one
        // and a lone pair that joins it.
        let rows = (periodic_row(elements[0]), periodic_row(elements[1]));
        if one.lone_pair && two.lone_pair {
            // Two lone pairs do not conjugate with each other.
        } else if one.lone_pair && two.multiple_bonds != 0 {
            let pi = if one.multiple_bonds == 1 {
                0.5
            } else if rows == (2, 2) {
                0.3
            } else {
                0.15
            };
            parameters.v2 = conjugated(pi);
        } else if two.lone_pair && one.multiple_bonds != 0 {
            let pi = if two.multiple_bonds == 1 {
                0.5
            } else if rows == (2, 2) {
                0.3
            } else {
                0.15
            };
            parameters.v2 = conjugated(pi);
        } else if (one.multiple_bonds == 1 || two.multiple_bonds == 1)
            && (elements[0] != 6 || elements[1] != 6)
        {
            parameters.v2 = conjugated(0.4);
        } else {
            parameters.v2 = conjugated(0.15);
        }
    } else if matches!(elements[0], 8 | 16) && matches!(elements[1], 8 | 16) {
        // Rule (h), first half: two chalcogens prefer to eclipse, not to
        // stagger, so the barrier is negative.
        parameters.v2 = -(first_w * second_w).sqrt();
    } else {
        parameters.v3 = (first_v * second_v).sqrt() / bonds;
    }

    Some(parameters)
}

/// The out-of-plane force constant for a three-connected centre, or `None`
/// where the table has none and the term is left out.
pub(super) fn out_of_plane(types: &[u8], central: usize, around: [usize; 3]) -> Option<f64> {
    params::out_of_plane(
        types[central],
        types[around[0]],
        types[around[1]],
        types[around[2]],
    )
}

/// A pair's van der Waals minimum and well depth, already scaled for the
/// hydrogen-bonding case.
pub(super) fn van_der_waals(types: &[u8], first: usize, second: usize) -> Option<(f64, f64)> {
    let one = params::van_der_waals(types[first])?;
    let two = params::van_der_waals(types[second])?;

    let one_radius = one.scale * one.polarizability.powf(params::VDW_POWER);
    let two_radius = two.scale * two.polarizability.powf(params::VDW_POWER);
    let asymmetry = (one_radius - two_radius) / (one_radius + two_radius);
    // A donor's radius is taken as the plain average; everything else is
    // widened by how unlike the two atoms are.
    let widening = if one.donor_acceptor == b'D' || two.donor_acceptor == b'D' {
        0.0
    } else {
        params::VDW_B * (1.0 - (-params::VDW_BETA * asymmetry * asymmetry).exp())
    };
    let mut minimum = 0.5 * (one_radius + two_radius) * (1.0 + widening);

    let mut well = 181.16 * one.factor * two.factor * one.polarizability * two.polarizability
        / ((one.polarizability / one.electrons).sqrt()
            + (two.polarizability / two.electrons).sqrt())
        / minimum.powi(6);

    // A donor and an acceptor come closer than either would otherwise, and
    // hold each other less tightly once there.
    let pair = (one.donor_acceptor, two.donor_acceptor);
    if pair == (b'D', b'A') || pair == (b'A', b'D') {
        minimum *= params::VDW_DONOR_ACCEPTOR_RADIUS;
        well *= params::VDW_DONOR_ACCEPTOR_WELL;
    }

    Some((minimum, well))
}
