//! Differential reaction-SMARTS check against RDKit's RunReactants.
//! Reads `reaction<TAB>reactants<TAB>expected;expected` (examples/reaction_oracle.py).
use oxidane::graph_algorithms::mol_canonization::canonize_mol;
use oxidane::molecule::reaction::ReactionSmarts;
use oxidane::prelude::*;
use std::io::BufRead;

fn canonical(mol: &MolecularGraph) -> String {
    canonize_mol(mol).to_smiles()
}

fn main() {
    let (mut total, mut agree) = (0, 0);
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 3 {
            continue;
        }
        let (smarts, reactants, expected) = (parts[0], parts[1], parts[2]);
        total += 1;

        // RDKit's expectations are canonical SMILES; canonize ours the same
        // way through its own canonizer so the comparison is of structures,
        // not of spellings, and re-canonize RDKit's through ours too.
        let mut want: Vec<String> = expected
            .split(';')
            .filter(|s| !s.is_empty())
            .filter_map(|s| MolecularGraph::from_smiles(s).ok())
            .map(|mols| {
                let refs: Vec<&MolecularGraph> = mols.iter().collect();
                canonical(&MolecularGraph::combine(&refs))
            })
            .collect();
        want.sort();
        want.dedup();

        let Ok(reaction) = ReactionSmarts::parse(smarts) else {
            println!("  {smarts:52} PARSE FAILED");
            continue;
        };
        let mols: Vec<MolecularGraph> = reactants
            .split('.')
            .filter_map(|s| MolecularGraph::from_smiles_single(s).ok())
            .collect();
        let refs: Vec<&MolecularGraph> = mols.iter().collect();
        let mut got: Vec<String> =
            reaction.apply_to(&refs).iter().map(canonical).collect();
        got.sort();
        got.dedup();

        if got == want {
            agree += 1;
        } else {
            println!("  {smarts:52} on {reactants:16}\n      ours  {got:?}\n      rdkit {want:?}");
        }
    }
    println!("{agree}/{total} agree with RDKit");
}
