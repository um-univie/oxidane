//! Turns BDE-db's raw CSV into `--level bond` lines for `train_mpnn`.
//!
//! ```text
//! curl -L https://ndownloader.figshare.com/files/18506018 -o rdf_data.csv
//! cargo run --release --example bde_to_bonds < rdf_data.csv > bde_bonds.tsv
//! cargo run --release --features ml --example train_mpnn -- --level bond < bde_bonds.tsv
//! ```
//!
//! BDE-db (St. John, Guan, Kim & Kim, 2019 -- <https://doi.org/10.6084/m9.figshare.10248932>)
//! is 290,664 homolytic bond dissociation enthalpies for molecules of ten or
//! fewer C/H/O/N heavy atoms, one bond broken per row: `molecule` (the intact
//! SMILES), an RDKit `bond_index` into *that toolkit's* internal, explicit-
//! hydrogen bond ordering, the two product radicals' SMILES, the BDE in
//! kcal/mol, and a `bond_type` like `C-C` or `C-H`.
//!
//! `bond_index` is useless here -- it names a position in a numbering this
//! crate has no reason to agree with, and doesn't even try to (a different
//! SMILES writer, explicit hydrogens added in a different order, ...). Every
//! row is instead identified structurally, from the two product SMILES:
//! hydrogens get a row in this crate's graph features exactly like any other
//! atom, so a C-H or O-H row is a real edge in [`GraphFeatures`] the same way
//! a C-C row is, and one matching approach covers both. Exactly one single,
//! non-ring bond splits the molecule into two pieces whose atomic *skeleton*
//! -- elements, bond orders, and how they connect -- matches `fragment1` and
//! `fragment2`, and that is the bond the row is about. [`skeleton_signature`]
//! is what actually tells two candidate bonds apart when element and
//! bond-order counts alone can't: lactic acid's alcohol C-OH and carboxyl
//! C-OH cut to the same 1-oxygen-vs-five-heavy-atom split either way, and two
//! different hydrogens on the same carbon (diastereotopic, chemically
//! distinct) are only tellable apart at all because each is now its own
//! graph node rather than a property shared by the whole atom.
//!
//! A handful of rows are still genuinely ambiguous -- true molecular symmetry
//! aside, two different positions occasionally still refine to the same
//! signature -- and those are skipped rather than guessed at; the counts are
//! reported on stderr.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Write};

use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::bond::BondOrder;
use oxidane::prelude::*;

/// A permutation-invariant fingerprint of a molecule's atoms (or a subset of
/// them): refine each atom's label a few rounds from a starting `seed` using
/// its own label and its neighbours' (bond order, neighbour label) pairs,
/// then keep the sorted multiset of final labels. Two atoms only end up with
/// the same label if their seeds match and their neighbourhoods are the same
/// that many bonds out, so the sorted multiset is stable under renumbering
/// but changes with the actual shape of the skeleton -- exactly what a
/// formula-only comparison misses, and precise enough to tell two hydrogens
/// on the same atom apart once each is its own node with its own place in
/// that shape.
fn skeleton_signature<M: MoleculeCore + ?Sized>(
    molecule: &M,
    atoms: &[usize],
    seed: &HashMap<usize, u64>,
) -> Vec<u64> {
    let member: HashSet<usize> = atoms.iter().copied().collect();
    let mut label = seed.clone();

    for _ in 0..3 {
        let mut next = HashMap::with_capacity(label.len());
        for &atom in atoms {
            let mut neighbours: Vec<(u8, u64)> = molecule
                .get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .filter(|bond| member.contains(&bond.target()))
                .map(|bond| (bond_order_rank(bond.bond_order()), label[&bond.target()]))
                .collect();
            neighbours.sort_unstable();
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            label[&atom].hash(&mut hasher);
            neighbours.hash(&mut hasher);
            next.insert(atom, hasher.finish());
        }
        label = next;
    }

    let mut signature: Vec<u64> = atoms.iter().map(|atom| label[atom]).collect();
    signature.sort_unstable();
    signature
}

/// [`skeleton_signature`] seeded by element alone -- a bond's own two atoms'
/// identity is all that's fixed going in; everything else about how they and
/// their neighbours connect is what the refinement rounds are for.
fn signature_by_element<M: MoleculeCore + ?Sized>(molecule: &M, atoms: &[usize]) -> Vec<u64> {
    let seed = atoms
        .iter()
        .map(|&atom| (atom, molecule.get_atomic_number(atom) as u64))
        .collect();
    skeleton_signature(molecule, atoms, &seed)
}

fn bond_order_rank(order: BondOrder) -> u8 {
    match order {
        BondOrder::Single => 0,
        BondOrder::Double => 1,
        BondOrder::Triple => 2,
        BondOrder::Aromatic => 3,
        BondOrder::Quadruple => 4,
        BondOrder::Coordinate => 5,
        BondOrder::Up => 6,
        BondOrder::Down => 7,
    }
}

fn all_atoms<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<usize> {
    (0..molecule.len()).collect()
}

/// Every atom reachable from `start` without crossing the bond
/// `start`--`avoid`.
fn atoms_reachable_from<M: MoleculeCore + ?Sized>(
    molecule: &M,
    start: usize,
    avoid: usize,
) -> Vec<usize> {
    let mut seen = vec![false; molecule.len()];
    seen[start] = true;
    let mut stack = vec![start];
    let mut out = vec![start];
    while let Some(atom) = stack.pop() {
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            let target = bond.target();
            if (atom == start && target == avoid) || (target == start && atom == avoid) {
                continue;
            }
            if !seen[target] {
                seen[target] = true;
                stack.push(target);
                out.push(target);
            }
        }
    }
    out
}

/// The molecule's single, non-ring bonds, each with the skeleton signature on
/// either side of it. Every bond counts now, not just the ones between two
/// heavy atoms -- a hydrogen is as much a graph node as anything else.
fn candidate_bonds(molecule: &MolecularGraph) -> Vec<(usize, usize, Vec<u64>, Vec<u64>)> {
    let (_, ring_bonds) = molecule.find_cyclic_bonds_and_atoms();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for atom in 0..molecule.len() {
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            let target = bond.target();
            if bond.bond_order() != BondOrder::Single {
                continue;
            }
            let key = (atom.min(target), atom.max(target));
            if ring_bonds.contains(&key) || !seen.insert(key) {
                continue;
            }
            let side_a =
                signature_by_element(molecule, &atoms_reachable_from(molecule, atom, target));
            let side_b =
                signature_by_element(molecule, &atoms_reachable_from(molecule, target, atom));
            out.push((key.0, key.1, side_a, side_b));
        }
    }
    out
}

struct Row {
    molecule: String,
    fragment1: String,
    fragment2: String,
    bde: f64,
}

fn parse_row(line: &str) -> Option<Row> {
    let mut fields = line.split(',');
    let _rid = fields.next()?;
    let molecule = fields.next()?.to_string();
    let _bond_index = fields.next()?;
    let fragment1 = fields.next()?.to_string();
    let fragment2 = fields.next()?.to_string();
    let bde = fields.next()?.parse::<f64>().ok()?;
    Some(Row {
        molecule,
        fragment1,
        fragment2,
        bde,
    })
}

/// Resolves each tied set of candidate positions a group of rows landed on
/// into label assignments, reporting how many rows were absorbed into a
/// label versus left ambiguous.
///
/// A row lands on a **set** of positions rather than one because a tie isn't
/// automatically ambiguous: several symmetry-equivalent positions (the three
/// C-C bonds of a tert-butyl group, the three C-H bonds of a methyl, ...)
/// give literally the same fragment and so the same tied set on every one of
/// their rows -- rows this crate's own parse cannot otherwise tell apart,
/// because the positions themselves can't be.
///
/// A group is trusted only when every row in it agrees on the value *and*
/// the group has exactly one row per tied position (one position, however
/// many equivalent rows landed on it; or `n` positions with `n` rows, one
/// each). A single position with rows that disagree is left ambiguous rather
/// than picking one arbitrarily. More positions than rows, or values that
/// disagree across more than one position, are left ambiguous the same way.
fn resolve_hit_sets(
    by_hit_set: HashMap<Vec<(usize, usize)>, Vec<f64>>,
) -> (HashMap<(usize, usize), f64>, usize, usize) {
    let mut labels = HashMap::new();
    let mut matched = 0usize;
    let mut ambiguous = 0usize;
    for (hits, values) in by_hit_set {
        let agree = values
            .first()
            .is_some_and(|first| values.iter().all(|value| (value - first).abs() < 1e-6));
        if agree && (hits.len() == 1 || hits.len() == values.len()) {
            for hit in hits {
                labels.insert(hit, values[0]);
            }
            matched += values.len();
        } else {
            ambiguous += values.len();
        }
    }
    (labels, matched, ambiguous)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut by_molecule: HashMap<String, Vec<Row>> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut total = 0usize;
    let mut bad_rows = 0usize;

    let stdin = std::io::stdin();
    for line in stdin.lock().lines().skip(1) {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        total += 1;
        let Some(row) = parse_row(&line) else {
            bad_rows += 1;
            continue;
        };
        by_molecule
            .entry(row.molecule.clone())
            .or_insert_with(|| {
                order.push(row.molecule.clone());
                Vec::new()
            })
            .push(row);
    }

    let mut unparseable = 0usize;
    let mut unmatched = 0usize;
    let mut total_matched = 0usize;
    let mut total_ambiguous = 0usize;
    let mut lines_written = 0usize;

    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for smiles in &order {
        let rows = &by_molecule[smiles];
        let Ok(molecule) = MolecularGraph::from_smiles_single(smiles) else {
            unparseable += 1;
            continue;
        };

        let candidates = candidate_bonds(&molecule);
        let mut by_hit_set: HashMap<Vec<(usize, usize)>, Vec<f64>> = HashMap::new();
        for row in rows {
            let (Ok(frag1), Ok(frag2)) = (
                MolecularGraph::from_smiles_single(&row.fragment1),
                MolecularGraph::from_smiles_single(&row.fragment2),
            ) else {
                unmatched += 1;
                continue;
            };
            let want_a = signature_by_element(&frag1, &all_atoms(&frag1));
            let want_b = signature_by_element(&frag2, &all_atoms(&frag2));
            let mut hits: Vec<(usize, usize)> = candidates
                .iter()
                .filter(|(_, _, side_a, side_b)| {
                    (*side_a == want_a && *side_b == want_b)
                        || (*side_a == want_b && *side_b == want_a)
                })
                .map(|&(a, b, _, _)| (a, b))
                .collect();
            if hits.is_empty() {
                unmatched += 1;
                continue;
            }
            hits.sort_unstable();
            by_hit_set.entry(hits).or_default().push(row.bde);
        }
        let (labels, matched, ambiguous) = resolve_hit_sets(by_hit_set);
        total_matched += matched;
        total_ambiguous += ambiguous;

        if labels.is_empty() {
            continue;
        }
        let field: Vec<String> = labels
            .into_iter()
            .map(|((a, b), bde)| format!("{a}-{b}:{bde}"))
            .collect();
        writeln!(out, "{smiles}\t{}", field.join(","))?;
        lines_written += 1;
    }

    eprintln!("{total} rows read, {bad_rows} unparseable");
    eprintln!(
        "{} molecules: {unparseable} failed to parse, {total_matched} matched, \
         {unmatched} unmatched, {total_ambiguous} ambiguous",
        order.len()
    );
    eprintln!("{lines_written} lines written");
    Ok(())
}
