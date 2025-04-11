//! Differential CIP check against RDKit's `rdCIPLabeler`.
//!
//! Reads `smiles<TAB>idx:label;...` on stdin (produce it with the oracle
//! script) and reports every disagreement, split into false positives (we
//! label an atom RDKit leaves unlabelled), false negatives (the reverse) and
//! outright wrong descriptors.
use molecules::graph_algorithms::mol_canonization::reorder_molecule;
use molecules::molecule::cip::CipConfig;
use molecules::prelude::*;
use std::io::BufRead;

fn main() {
    let (mut n, mut fp, mut fneg, mut wrong, mut ok) = (0, 0, 0, 0, 0);
    let mut not_invariant = 0;
    let mut examples: Vec<String> = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let (smi, labels) = match line.split_once('\t') {
            Some(x) => x,
            None => continue,
        };
        let Ok(m) = MolecularGraph::from_smiles_single(smi) else { continue };
        let ours = m.assign_cip_configs();
        let mut want = vec![CipConfig::None; ours.len()];
        for tok in labels.split(';').filter(|s| !s.is_empty()) {
            let (i, l) = tok.split_once(':').unwrap();
            let i: usize = i.parse().unwrap();
            if i < want.len() {
                want[i] = match l {
                    "R" => CipConfig::R,
                    "S" => CipConfig::S,
                    "r" => CipConfig::PseudoR,
                    "s" => CipConfig::PseudoS,
                    "E" => CipConfig::E,
                    "Z" => CipConfig::Z,
                    "z" => CipConfig::SeqCis,
                    "e" => CipConfig::SeqTrans,
                    _ => CipConfig::None,
                };
            }
        }
        n += 1;

        // Self-check: descriptors must follow their atoms under renumbering.
        let mut state = 0x2545_f491_4f6c_dd1du64 ^ (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        for _ in 0..4 {
            let count = ours.len();
            let mut labeling: Vec<usize> = (0..count).collect();
            for i in (1..count).rev() {
                labeling.swap(i, (next() % (i as u64 + 1)) as usize);
            }
            let mut permuted = m.clone();
            reorder_molecule(&mut permuted, &labeling);
            let got = permuted.assign_cip_configs();
            for (new, &old) in labeling.iter().enumerate() {
                if got[new] != ours[old] {
                    not_invariant += 1;
                    if examples.len() < 15 {
                        examples.push(format!("{:>14}  atom {old} -> {new}  was {} now {}  {smi}",
                            "not-invariant", ours[old], got[new]));
                    }
                    break;
                }
            }
        }

        let mut clean = true;
        for (i, (&got, &exp)) in ours.iter().zip(want.iter()).enumerate() {
            if got == exp {
                continue;
            }
            clean = false;
            let kind = match (exp, got) {
                (CipConfig::None, _) => {
                    fp += 1;
                    "false-positive"
                }
                (_, CipConfig::None) => {
                    fneg += 1;
                    "false-negative"
                }
                _ => {
                    wrong += 1;
                    "wrong"
                }
            };
            if examples.len() < 15 {
                examples.push(format!("{kind:>14}  atom {i:>2}  ours={got:<1} rdkit={exp:<1}  {smi}"));
            }
        }
        if clean {
            ok += 1
        }
    }
    println!("molecules: {n}   fully correct: {ok}   disagreeing: {}", n - ok);
    println!("atoms: false-positive {fp}  false-negative {fneg}  wrong descriptor {wrong}");
    println!("renumbering self-check: {not_invariant} failures");
    println!();
    for e in examples {
        println!("{e}")
    }
}
