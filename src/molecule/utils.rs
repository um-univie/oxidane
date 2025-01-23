use crate::molecule::bond::{BondOrder, BondTarget};
use nohash_hasher::IntMap;
use rustc_hash::FxHashSet;

/// A faster alternative to HashSet<(usize,usize)> using a flat array with bit packing
pub fn decrease_bond(bond: &mut BondTarget) {
    match bond.bond_order {
        BondOrder::Single => panic!("Cannot decrease bond beyond single bond"),
        BondOrder::Up => panic!("Cannot decrease single bond"),
        BondOrder::Down => panic!("Cannot decrease single bond"),
        BondOrder::Double => bond.bond_order = BondOrder::Single,
        BondOrder::Triple => bond.bond_order = BondOrder::Double,
        BondOrder::Quadruple => bond.bond_order = BondOrder::Triple,
        BondOrder::Aromatic => bond.bond_order = BondOrder::Single,
        BondOrder::Coordinate => panic!("Cannot decrease bond beyond coordinate bond"),
    }
}

pub fn order_tuple(index1: usize, index2: usize) -> (usize, usize) {
    if index1 < index2 {
        (index1, index2)
    } else {
        (index2, index1)
    }
}
pub fn update_degrees(
    degrees: &mut [i8],
    atom_index: usize,
    neighbor_index: usize,
    decrease: bool,
) {
    let factor = if decrease { -1 } else { 1 };
    degrees[atom_index] += factor;
    degrees[neighbor_index] += factor;
}

pub fn reconstruct_path(mut current_node: usize, parents: &[Option<usize>]) -> Vec<usize> {
    let mut path = Vec::new();

    while let Some(parent_atom) = parents[current_node] {   
        path.push(current_node);
        current_node = parent_atom;
    }
    path.push(current_node);

    path.reverse();
    path
}

/// Renames ring numbers in a SMILES string to ensure they are sequential and conform to SMILES standards.
/// This function mutates the input SMILES string in place.
///
/// # Arguments
///
/// * `smiles` - A mutable reference to the SMILES string that will be modified.
///
/// # Returns
///
/// * `Result<(), String>` - Returns `Ok(())` on success,
///   or an error message if the input SMILES string has invalid ring closures.
pub fn rename_ring_numbers_in_place(smiles: &mut String) -> Result<(), String> {
    let mut ring_relabel_map: IntMap<u8, u8> = IntMap::default();
    let mut ring_counter = 1u8;

    let mut buffer = std::mem::take(smiles).into_bytes(); // Second pass: Replace old ring numbers with new ones

    for ch in buffer.iter_mut() {
        if ch.is_ascii_digit() {
            if !ring_relabel_map.contains_key(ch) {
                ring_relabel_map.insert(*ch, ring_counter);
                *ch = b'0' + ring_counter;
                ring_counter += 1;
            } else {
                *ch = b'0' + ring_relabel_map[ch];
            }
        }
    }

    *smiles = String::from_utf8(buffer).unwrap();
    Ok(())
}

/// Checks wether or not the hueckel rule is is_hueckel_satisfied
///
///
/// # Examples
///
/// ```
///
/// use molecules::molecule::utils::is_hueckel_satisfied;
///
/// assert!(is_hueckel_satisfied(6));
/// assert!(is_hueckel_satisfied(10))
/// ````
pub fn is_hueckel_satisfied(number_of_pi_electrons: usize) -> bool {
    number_of_pi_electrons & 3 == 2
}
