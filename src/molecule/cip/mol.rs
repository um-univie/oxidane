//! The view of a molecule that the CIP rules need: Kekulé bond orders, ring
//! membership, and Mancude-ring averaged atomic numbers.
//!
//! Ported from RDKit's `Code/GraphMol/CIPLabeler/{CIPMol.cpp,Mancude.cpp}`.
//! See the module docs in [`super`] for provenance and licensing.

use std::cmp::Ordering;

use chemistry_consts::ElementProperties;

use crate::molecule::{
    base::MoleculeCore, bond::BondOrder, experimental::MoleculeExperimental,
    molecular_graph::MolecularGraph,
};

/// A non-negative rational, used for the averaged atomic numbers of Mancude
/// rings. Kept exact rather than as an `f64` so that comparisons between e.g.
/// 20/3 and 13/2 cannot go wrong by rounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Frac {
    numerator: i32,
    denominator: i32,
}

impl Frac {
    pub(super) fn new(numerator: i32, denominator: i32) -> Self {
        if denominator == 0 {
            return Frac { numerator: 0, denominator: 1 };
        }
        let sign = if denominator < 0 { -1 } else { 1 };
        let (numerator, denominator) = (numerator * sign, denominator * sign);
        let divisor = gcd(numerator.unsigned_abs(), denominator.unsigned_abs()).max(1) as i32;
        Frac { numerator: numerator / divisor, denominator: denominator / divisor }
    }

    pub(super) fn whole(value: i32) -> Self {
        Frac { numerator: value, denominator: 1 }
    }

    pub(super) fn denominator(self) -> i32 {
        self.denominator
    }
}

impl Ord for Frac {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.numerator as i64 * other.denominator as i64)
            .cmp(&(other.numerator as i64 * self.denominator as i64))
    }
}

impl PartialOrd for Frac {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// One end of a bond as the CIP code sees it.
#[derive(Debug, Clone, Copy)]
pub(super) struct CipBond {
    pub(super) index: usize,
    pub(super) other: usize,
    /// Kekulé bond order; 0 for coordinate/dative bonds, which CIP treats as
    /// contributing no duplicated atoms.
    pub(super) order: i32,
    pub(super) in_ring: bool,
}

/// Classification of an atom's ability to take part in a Mancude
/// (**ma**ximum **n**umber of non**cu**mulative **d**oubl**e** bonds) ring
/// system. Only these environments can delocalise, so only these get an
/// averaged atomic number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MancudeType {
    Other,
    /// Neutral tetravalent C/Si/Ge with one double bond in the ring.
    Cv4D3,
    /// Anionic trivalent C/Si/Ge — carries a relocatable negative charge.
    Cv3D3Minus,
    /// Neutral trivalent N/P/As with one double bond.
    Nv3D2,
    /// Anionic divalent N/P/As — relocatable negative charge.
    Nv2D2Minus,
    /// Cationic tetravalent N/P/As.
    Nv4D3Plus,
    /// Cationic trivalent O.
    Ov3D2Plus,
}

/// The molecule, pre-processed into everything the CIP rules ask of it.
pub(super) struct CipMol {
    atomic_numbers: Vec<u8>,
    isotopes: Vec<u16>,
    charges: Vec<i8>,
    /// Per-atom adjacency with Kekulé bond orders and ring flags.
    bonds: Vec<Vec<CipBond>>,
    /// Averaged atomic number for atoms in Mancude rings, plain atomic number
    /// otherwise.
    fractional_atomic_numbers: Vec<Frac>,
}

impl CipMol {
    /// Builds the CIP view. Aromatic bonds are Kekulised first, because the
    /// rules are defined on integer bond orders (the duplicate-atom rule needs
    /// to know how many phantom atoms a bond contributes); the Mancude
    /// averaging is what puts the resulting arbitrary single/double assignment
    /// back on an even footing.
    pub(super) fn new(mol: &MolecularGraph) -> Self {
        let mut kekulized = mol.clone();
        // A molecule that will not Kekulise (an aromatic system we cannot
        // resolve) still gets labelled, with its aromatic bonds read as order
        // 1 — the same fallback RDKit takes.
        let _ = kekulized.kekulize();

        let atomic_numbers = kekulized.atomic_numbers().to_vec();
        let n = atomic_numbers.len();
        let isotopes = kekulized
            .isotopes()
            .map(|v| v.clone())
            .unwrap_or_else(|| vec![0; n]);
        let charges = kekulized.charges().to_vec();

        let ring_bonds = ring_bond_set(&kekulized);

        let mut bonds: Vec<Vec<CipBond>> = Vec::with_capacity(n);
        for atom in 0..n {
            let mut list = Vec::new();
            for bond in kekulized.atom_bonds()[atom].iter() {
                let other = bond.target();
                let order = match bond.bond_order() {
                    BondOrder::Single | BondOrder::Up | BondOrder::Down => 1,
                    BondOrder::Double => 2,
                    BondOrder::Triple => 3,
                    BondOrder::Quadruple => 4,
                    // Unresolved aromatic: read as single, as RDKit does.
                    BondOrder::Aromatic => 1,
                    BondOrder::Coordinate => 0,
                };
                let key = (atom.min(other), atom.max(other));
                list.push(CipBond {
                    index: list.len(),
                    other,
                    order,
                    in_ring: ring_bonds.contains(&key),
                });
            }
            bonds.push(list);
        }

        let mut mol = CipMol {
            atomic_numbers,
            isotopes,
            charges,
            bonds,
            fractional_atomic_numbers: Vec::new(),
        };
        mol.fractional_atomic_numbers = calc_frac_atom_nums(&mol);
        mol
    }

    pub(super) fn len(&self) -> usize {
        self.atomic_numbers.len()
    }

    pub(super) fn atomic_number(&self, atom: usize) -> u8 {
        self.atomic_numbers[atom]
    }

    pub(super) fn isotope(&self, atom: usize) -> u16 {
        self.isotopes.get(atom).copied().unwrap_or(0)
    }

    pub(super) fn charge(&self, atom: usize) -> i8 {
        self.charges.get(atom).copied().unwrap_or(0)
    }

    pub(super) fn bonds(&self, atom: usize) -> &[CipBond] {
        &self.bonds[atom]
    }

    pub(super) fn fractional_atomic_number(&self, atom: usize) -> Frac {
        self.fractional_atomic_numbers[atom]
    }

    /// The mass CIP rule 2 ranks by: the exact mass of the stated isotope, or
    /// the element's standard atomic weight when no isotope is stated.
    ///
    /// Using the standard weight for unlabelled atoms is what makes rule 2
    /// rank an unlabelled atom *below* its own heavy isotope but *above* its
    /// light one, rather than treating "unspecified" as a separate class.
    pub(super) fn atomic_mass(&self, atom: usize) -> f64 {
        let atomic_number = self.atomic_numbers[atom];
        let isotope = self.isotope(atom);
        if isotope == 0 {
            return atomic_number.standard_atomic_weight().unwrap_or(0.0);
        }
        atomic_number
            .isotopes()
            .and_then(|mut it| it.find(|i| i.mass.round() as u16 == isotope))
            .map(|i| i.mass)
            .unwrap_or(isotope as f64)
    }
}

/// The set of bonds that lie on a ring, as unordered index pairs.
///
/// A bond is on a ring exactly when it is not a bridge, i.e. when its two ends
/// are still connected after it is removed. Testing that directly, one bond at
/// a time, is quadratic where Tarjan's bridge-finding is linear — but this
/// runs once per molecule, so for anything of chemical size the difference is
/// microseconds, and the direct test has no root/parent/back-edge bookkeeping
/// to get subtly wrong.
fn ring_bond_set(mol: &MolecularGraph) -> std::collections::HashSet<(usize, usize)> {
    let n = mol.atomic_numbers().len();
    let mut ring_bonds = std::collections::HashSet::new();
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();

    for atom in 0..n {
        for bond in mol.atom_bonds()[atom].iter() {
            let target = bond.target();
            if target <= atom || ring_bonds.contains(&(atom, target)) {
                continue;
            }

            // Depth-first from `atom`, forbidden to traverse `atom`-`target`.
            seen.iter_mut().for_each(|s| *s = false);
            seen[atom] = true;
            stack.clear();
            stack.push(atom);
            let mut reached = false;
            while let Some(current) = stack.pop() {
                for next in mol.atom_bonds()[current].iter().map(|b| b.target()) {
                    if (current == atom && next == target) || (current == target && next == atom) {
                        continue;
                    }
                    if next == target {
                        reached = true;
                        stack.clear();
                        break;
                    }
                    if !seen[next] {
                        seen[next] = true;
                        stack.push(next);
                    }
                }
                if reached {
                    break;
                }
            }
            if reached {
                ring_bonds.insert((atom, target));
            }
        }
    }
    ring_bonds
}

/// Assigns each atom an averaged atomic number over the Mancude ring system it
/// belongs to, or its own atomic number if it belongs to none.
///
/// This is what makes an aromatic heterocycle give the same answer whichever
/// Kekulé structure was chosen: in pyridine every ring atom is averaged over
/// its ring neighbours in the same delocalised system, so the arbitrary
/// placement of the double bonds cannot leak into the ranking.
fn calc_frac_atom_nums(mol: &CipMol) -> Vec<Frac> {
    let n = mol.len();
    let mut fractions: Vec<Frac> = (0..n).map(|i| Frac::whole(mol.atomic_number(i) as i32)).collect();

    let mut types = vec![MancudeType::Other; n];
    if !seed_types(&mut types, mol) {
        return fractions;
    }
    relax_types(&mut types, mol);

    let mut parts = vec![0usize; n];
    let num_parts = visit_parts(&mut parts, &types, mol);
    if num_parts == 0 {
        return fractions;
    }

    // Parts that carry a relocatable negative charge need a second pass.
    let mut charge_parts: Vec<usize> = Vec::new();
    for atom in 0..n {
        if parts[atom] == 0 {
            continue;
        }
        if matches!(types[atom], MancudeType::Cv3D3Minus | MancudeType::Nv2D2Minus)
            && !charge_parts.contains(&parts[atom])
        {
            charge_parts.push(parts[atom]);
        }

        let mut numerator = 0i32;
        let mut denominator = 0i32;
        for bond in mol.bonds(atom) {
            if parts[bond.other] == parts[atom] {
                numerator += mol.atomic_number(bond.other) as i32;
                denominator += 1;
            }
        }
        fractions[atom] = Frac::new(numerator, denominator);
    }

    // Relocating a negative charge changes which atoms the double bonds sit
    // on, so the average has to be taken over the whole part at once, walking
    // it in index order and accumulating the higher-order bonds seen so far.
    for &part in &charge_parts {
        let mut numerator = 0i32;
        let mut denominator = 0i32;
        for atom in 0..n {
            if parts[atom] != part {
                continue;
            }
            fractions[atom] = Frac::new(numerator, denominator);
            denominator += 1;
            for bond in mol.bonds(atom) {
                if bond.order > 1 && parts[bond.other] == part {
                    numerator += (bond.order - 1) * mol.atomic_number(bond.other) as i32;
                }
            }
        }
    }

    fractions
}

/// Marks atoms whose element, charge and bonding pattern let them take part in
/// a delocalised ring system. Returns whether any atom other than a plain
/// neutral ring carbon was marked — if not, there is nothing to average.
fn seed_types(types: &mut [MancudeType], mol: &CipMol) -> bool {
    let mut result = false;
    for atom in 0..mol.len() {
        // `btypes` packs the bond counts by order into one integer, one byte
        // per order, so a single comparison identifies the environment.
        // Hydrogens are ordinary single-bonded neighbours here (this crate
        // makes them explicit atoms), which contributes to the low byte
        // exactly as RDKit's implicit-hydrogen count does.
        let mut btypes = 0i32;
        let mut in_ring = false;
        for bond in mol.bonds(atom) {
            btypes += match bond.order {
                1 => 0x0000_0001,
                2 => 0x0000_0100,
                _ => 0x0100_0000,
            };
            if bond.in_ring {
                in_ring = true;
            }
        }
        if !in_ring {
            continue;
        }

        let charge = mol.charge(atom);
        let new_type = match mol.atomic_number(atom) {
            6 | 14 | 32 => {
                if charge == 0 && btypes == 0x0102 {
                    Some(MancudeType::Cv4D3)
                } else if charge == -1 && btypes == 0x0003 {
                    result = true;
                    Some(MancudeType::Cv3D3Minus)
                } else {
                    None
                }
            }
            7 | 15 | 33 => {
                if charge == 0 && btypes == 0x0101 {
                    result = true;
                    Some(MancudeType::Nv3D2)
                } else if charge == -1 && btypes == 0x0002 {
                    result = true;
                    Some(MancudeType::Nv2D2Minus)
                } else if charge == 1 && btypes == 0x0102 {
                    result = true;
                    Some(MancudeType::Nv4D3Plus)
                } else {
                    None
                }
            }
            8 => {
                if charge == 1 && btypes == 0x0101 {
                    result = true;
                    Some(MancudeType::Ov3D2Plus)
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(new_type) = new_type {
            types[atom] = new_type;
        }
    }
    result
}

/// Unmarks atoms that were typed but have fewer than two typed neighbours:
/// resonance needs somewhere to go on both sides, so a marked atom dangling
/// off the end of a chain cannot delocalise and must keep its own atomic
/// number.
fn relax_types(types: &mut [MancudeType], mol: &CipMol) {
    let n = mol.len();
    let mut counts = vec![0usize; n];
    let mut queue: Vec<usize> = Vec::new();

    for atom in 0..n {
        for bond in mol.bonds(atom) {
            if types[bond.other] != MancudeType::Other {
                counts[atom] += 1;
            }
        }
        if counts[atom] == 1 {
            queue.push(atom);
        }
    }

    let mut position = 0;
    while position < queue.len() {
        let atom = queue[position];
        position += 1;
        if types[atom] == MancudeType::Other {
            continue;
        }
        types[atom] = MancudeType::Other;
        for bond in mol.bonds(atom) {
            counts[bond.other] -= 1;
            if counts[bond.other] == 1 {
                queue.push(bond.other);
            }
        }
    }
}

/// Labels each typed atom with the index of the connected delocalised system
/// it belongs to; returns how many systems were found.
fn visit_parts(parts: &mut [usize], types: &[MancudeType], mol: &CipMol) -> usize {
    let mut num_parts = 0;
    for start in 0..mol.len() {
        if parts[start] != 0 || types[start] == MancudeType::Other {
            continue;
        }
        num_parts += 1;
        parts[start] = num_parts;
        let mut stack = vec![start];
        while let Some(atom) = stack.pop() {
            for bond in mol.bonds(atom) {
                if !bond.in_ring {
                    continue;
                }
                if parts[bond.other] == 0 && types[bond.other] != MancudeType::Other {
                    parts[bond.other] = num_parts;
                    stack.push(bond.other);
                }
            }
        }
    }
    num_parts
}
