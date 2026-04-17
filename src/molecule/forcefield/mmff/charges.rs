//! MMFF94 partial charges.
//!
//! MMFF builds a charge in two steps. First every atom gets a *formal*
//! charge, which is not the one written in the SMILES: a carboxylate's is
//! shared as -1/2 over both oxygens, a nitrate's as -1/3 over three, an
//! imidazolium's +1 over both its nitrogens. That sharing is the point --
//! the two oxygens of a carboxylate are the same oxygen as far as the
//! molecule is concerned, and a model that charges one of them and not the
//! other is wrong in a way no amount of parameter fitting repairs.
//!
//! Then each bond moves a little charge, by a *bond charge increment* fitted
//! for that pair of types, or by the difference of the two types' partial
//! increments where the pair has no fitted value. Equation 15 of Halgren's
//! MMFF.V puts the two together:
//!
//! ```text
//! q_i = (1 - M_i v_i) q0_i + v_i sum_k q0_k + sum_k w_ik
//! ```
//!
//! where `M` is the type's connection count, `v` its formal-charge
//! adjustment, `q0` the formal charges, and `w` the increments over the
//! bonds at `i`.

use super::super::UntypedAtom;
use super::assign::bond_type;
use super::params;
use super::perception::Perception;
use super::typing;
use crate::molecule::base::MoleculeCore;

/// Every atom's MMFF94 partial charge, in atom order.
///
/// # Errors
///
/// [`UntypedAtom`] for the first atom MMFF94 has no type for, since a charge
/// model keyed on atom types has nothing to say about an atom without one.
pub fn partial_charges<M: MoleculeCore + ?Sized>(molecule: &M) -> Result<Vec<f64>, UntypedAtom> {
    let (perception, types) = typing::typed(molecule)?;
    Ok(charges_from_types(&perception, &types))
}

/// The formal charge MMFF94 gives each atom, which is the written charge
/// spread over the atoms that actually share it.
pub(super) fn formal_charges(perception: &Perception, types: &[u8]) -> Vec<f64> {
    (0..types.len())
        .map(|atom| formal_charge(perception, types, atom))
        .collect()
}

fn formal_charge(perception: &Perception, types: &[u8], atom: usize) -> f64 {
    match types[atom] {
        // A terminal oxygen or sulfur, whose charge depends on what it hangs
        // off and how many siblings it shares the charge with.
        32 | 72 => terminal_chalcogen_charge(perception, types, atom),

        // The nitrogen of an aromatic five-ring anion, sharing -1 with the
        // other such nitrogens of its ring.
        76 => {
            let ring = perception
                .rings()
                .iter()
                .find(|ring| ring.contains(&atom));
            match ring {
                Some(ring) => {
                    let nitrogens = ring.iter().filter(|&&other| types[other] == 76).count();
                    if nitrogens > 0 { -1.0 / nitrogens as f64 } else { 0.0 }
                }
                None => 0.0,
            }
        }

        // Amidinium, guanidinium and imidazolium: the charge belongs to the
        // whole conjugated set of nitrogens, not to the one the SMILES
        // happened to write it on.
        55 | 56 | 81 => conjugated_nitrogen_charge(perception, types, atom),

        // An isonitrile nitrogen is only charged when it is a diazonium's.
        61 => {
            if perception
                .neighbours(atom)
                .iter()
                .any(|neighbour| types[neighbour.atom] == 42)
            {
                1.0
            } else {
                0.0
            }
        }

        34 | 49 | 51 | 54 | 58 | 92 | 93 | 94 | 97 => 1.0,
        87 | 95 | 96 | 98 | 99 => 2.0,
        88 => 3.0,
        35 | 62 | 89 | 90 | 91 => -1.0,
        _ => 0.0,
    }
}

/// The shared charge on a terminal oxygen or sulfur, decided by the centre
/// it hangs off: a carboxylate splits -1 two ways, a carbonate three, a
/// nitrate three, a perchlorate four, and a phosphine oxide's single oxygen
/// carries nothing at all.
fn terminal_chalcogen_charge(perception: &Perception, types: &[u8], atom: usize) -> f64 {
    for neighbour in perception.neighbours(atom) {
        let centre = neighbour.atom;
        let mut secondary_nitrogens = 0;
        let mut terminal_chalcogens = 0;
        for second in perception.neighbours(centre) {
            let other = second.atom;
            if perception.element(other) == 7
                && perception.degree(other) == 2
                && !perception.is_aromatic(other)
            {
                secondary_nitrogens += 1;
            }
            if matches!(perception.element(other), 8 | 16) && perception.degree(other) == 1 {
                terminal_chalcogens += 1;
            }
        }
        // A deprotonated sulfonamide's nitrogen is not standing in for one of
        // the sulfone's oxygens, so it must not be counted as one.
        if perception.element(centre) == 16 && terminal_chalcogens == 2 && secondary_nitrogens == 1 {
            secondary_nitrogens = 0;
        }

        let shared = |count: usize| -((count - 1) as f64 / count as f64);

        if perception.element(centre) == 6 && terminal_chalcogens > 0 {
            // Carboxylate and thiocarboxylate share -1; a lone terminal
            // oxygen on carbon carries the whole of it.
            return if terminal_chalcogens == 1 { -1.0 } else { shared(terminal_chalcogens) };
        }
        match types[centre] {
            45 if terminal_chalcogens == 3 => return -1.0 / 3.0,
            25 | 73 if terminal_chalcogens > 0 => {
                return if terminal_chalcogens == 1 { 0.0 } else { shared(terminal_chalcogens) };
            }
            18 if terminal_chalcogens > 0 => {
                let attached = secondary_nitrogens + terminal_chalcogens;
                return if attached == 2 {
                    0.0
                } else {
                    -((attached - 2) as f64 / terminal_chalcogens as f64)
                };
            }
            77 if terminal_chalcogens > 0 => return -1.0 / terminal_chalcogens as f64,
            _ => (),
        }
    }
    0.0
}

/// The charge of one nitrogen of a delocalized cation, which is the total
/// charge of the conjugated set divided by how many nitrogens are in it.
///
/// Not quite what the MMFF papers describe, but it is what makes the total
/// come out an integer, and it is what RDKit does: the alternative is a
/// charge that depends on which nitrogen the SMILES wrote the `+` on.
fn conjugated_nitrogen_charge(perception: &Perception, types: &[u8], atom: usize) -> f64 {
    let mut conjugated = vec![false; types.len()];
    conjugated[atom] = true;
    let mut total = perception.formal_charge(atom) as f64;
    let mut count = 1;

    let mut previous = 0;
    while count > previous {
        previous = count;
        for first in 0..types.len() {
            if !conjugated[first] {
                continue;
            }
            for neighbour in perception.neighbours(first) {
                // Only through a carbon that carries the delocalization.
                if !matches!(types[neighbour.atom], 57 | 80) {
                    continue;
                }
                for second in perception.neighbours(neighbour.atom) {
                    let far = second.atom;
                    if !matches!(types[far], 55 | 56 | 81) || conjugated[far] {
                        continue;
                    }
                    conjugated[far] = true;
                    total += perception.formal_charge(far) as f64;
                    count += 1;
                }
            }
        }
    }

    total / count as f64
}

pub(super) fn charges_from_types(perception: &Perception, types: &[u8]) -> Vec<f64> {
    let formal = formal_charges(perception, types);

    (0..types.len())
        .map(|atom| {
            let properties = params::properties(types[atom]);
            let charge_properties = params::charge_properties(types[atom]);
            let (Some(properties), Some(charge_properties)) = (properties, charge_properties) else {
                return 0.0;
            };
            let connections = properties.connections as f64;
            let adjustment = charge_properties.formal_charge_adjustment;

            let mut own = formal[atom];
            if adjustment == 0.0 {
                // A type that takes no share of its neighbours' formal charge
                // through the equation still feels a neighbouring anion: half
                // of it, spread over that neighbour's bonds.
                for neighbour in perception.neighbours(atom) {
                    if formal[neighbour.atom] < 0.0 {
                        own += formal[neighbour.atom]
                            / (2.0 * perception.degree(neighbour.atom) as f64);
                    }
                }
            }
            // An anionic divalent nitrogen next to a cation is the mirror of
            // that case, and MMFF singles it out.
            if types[atom] == 62 {
                for neighbour in perception.neighbours(atom) {
                    if formal[neighbour.atom] > 0.0 {
                        own -= formal[neighbour.atom] / 2.0;
                    }
                }
            }

            let mut neighbouring_formal = 0.0;
            let mut increments = 0.0;
            for neighbour in perception.neighbours(atom) {
                let other = neighbour.atom;
                let kind = bond_type(perception, types, atom, other);
                increments += params::bond_charge_increment(kind, types[atom], types[other])
                    .unwrap_or_else(|| {
                        let ours = params::charge_properties(types[atom]);
                        let theirs = params::charge_properties(types[other]);
                        match (ours, theirs) {
                            (Some(ours), Some(theirs)) => {
                                ours.partial_increment - theirs.partial_increment
                            }
                            _ => 0.0,
                        }
                    });
                neighbouring_formal += formal[other];
            }

            (1.0 - connections * adjustment) * own
                + adjustment * neighbouring_formal
                + increments
        })
        .collect()
}
