//! Prints what a molecule carries in the way of cis/trans marks, before and
//! after canonization, so that a mark that moves can be seen moving.
use oxidane::graph_algorithms::mol_canonization::canonize_mol;
use oxidane::molecule::bond::BondOrder;
use oxidane::molecule::chirality::ChiralClass;
use oxidane::prelude::*;
use std::io::BufRead;

/// The stereogenic double bonds, named by what they join rather than by index,
/// so two writings of one molecule can be compared.
fn stereogenic_bonds(molecule: &MolecularGraph) -> Vec<String> {
    let Some(classes) = molecule.chiral_classes() else {
        return Vec::new();
    };
    let is_mark = |atom: usize| {
        matches!(
            classes.get(atom),
            Some(ChiralClass::CisTransClockwise) | Some(ChiralClass::CisTransCounterclockwise)
        )
    };
    let symbol = |atom: usize| match molecule.get_atomic_number(atom) {
        6 => "C",
        7 => "N",
        8 => "O",
        other => Box::leak(format!("Z{other}").into_boxed_str()),
    };
    let mut bonds = Vec::new();
    for atom in 0..molecule.len() {
        if !is_mark(atom) {
            continue;
        }
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            let target = bond.target();
            if bond.bond_order() != BondOrder::Double || target < atom || !is_mark(target) {
                continue;
            }
            // What hangs off each end, which is what tells two bonds apart.
            let mut ends: Vec<String> = [atom, target]
                .iter()
                .map(|&end| {
                    let mut around: Vec<&str> = molecule
                        .get_atom_bonds(end)
                        .unwrap_or_default()
                        .iter()
                        .filter(|b| b.target() != atom && b.target() != target)
                        .map(|b| symbol(b.target()))
                        .collect();
                    around.sort_unstable();
                    format!("{}[{}]", symbol(end), around.join(""))
                })
                .collect();
            ends.sort();
            bonds.push(format!("{atom}-{target} {}={}", ends[0], ends[1]));
        }
    }
    bonds.sort();
    bonds
}

fn marks(molecule: &MolecularGraph) -> Vec<(usize, String, &'static str)> {
    let Some(classes) = molecule.chiral_classes() else {
        return Vec::new();
    };
    (0..molecule.len())
        .filter_map(|atom| {
            let mark = match classes.get(atom) {
                Some(ChiralClass::CisTransClockwise) => "clockwise",
                Some(ChiralClass::CisTransCounterclockwise) => "counterclockwise",
                _ => return None,
            };
            let element = molecule.get_atomic_number(atom);
            let neighbours = molecule.get_atom_bonds(atom).map_or(0, |bonds| bonds.len());
            Some((atom, format!("Z{element} with {neighbours} bonds"), mark))
        })
        .collect()
}

fn main() {
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let smiles = line.trim();
        if smiles.is_empty() {
            continue;
        }
        let Ok(molecule) = MolecularGraph::from_smiles_single(smiles) else {
            println!("{smiles}\n  unparsed");
            continue;
        };
        let before = marks(&molecule);
        let canonical = canonize_mol(&molecule);
        let after = marks(&canonical);
        println!("{smiles}");
        println!(
            "  atoms {}, marks on parse {}",
            molecule.len(),
            before.len()
        );
        println!("  stereogenic bonds: {:?}", stereogenic_bonds(&molecule));
        for (atom, what, mark) in &before {
            println!("    atom {atom} ({what}): {mark}");
        }
        println!(
            "  after canonizing: {} marks, writes {}",
            after.len(),
            canonical.to_smiles()
        );
        for (atom, what, mark) in &after {
            println!("    atom {atom} ({what}): {mark}");
        }
    }
}
