//! Applying reaction SMARTS, checked against RDKit's `RunReactants`.
//!
//! The full differential is `notes/reaction_oracle.py` +
//! `examples/reaction_diff.rs`. One case is known to differ:
//! `[c:1][#1:2]>>[c:1][N:3]` turns benzene into aniline here and produces
//! nothing in RDKit, because our molecules carry hydrogens as real atoms and
//! RDKit's are implicit, so `[#1:2]` has nothing to match there.

use crate::graph_algorithms::mol_canonization::canonize_mol;
use crate::molecule::reaction::{ReactionError, ReactionSmarts};
use crate::prelude::*;

/// Applies a reaction and returns the products as canonical SMILES, sorted and
/// deduplicated.
///
/// Canonical because a product is a structure, not a spelling: the six ways
/// `[c:1][#1:2]` matches benzene all give aniline, just written from six
/// different starting atoms.
fn products(smarts: &str, reactants: &[&str]) -> Vec<String> {
    let reaction = ReactionSmarts::parse(smarts).unwrap_or_else(|e| panic!("{smarts}: {e:?}"));
    let molecules: Vec<MolecularGraph> = reactants
        .iter()
        .map(|s| MolecularGraph::from_smiles_single(s).unwrap())
        .collect();
    let references: Vec<&MolecularGraph> = molecules.iter().collect();
    let mut smiles: Vec<String> = reaction
        .apply_to(&references)
        .iter()
        .map(|product| canonize_mol(product).to_smiles())
        .collect();
    smiles.sort();
    smiles.dedup();
    smiles
}

/// A bond order that differs between the two sides is changed in place.
#[test]
fn bond_orders_change() {
    assert_eq!(products("[C:1]-[O:2]>>[C:1]=[O:2]", &["CCO"]), ["O=CC"]);
    assert_eq!(products("[C:1]=[C:2]>>[C:1]-[C:2]", &["CC=CC"]), ["CCCC"]);
    assert_eq!(products("[C:1]#[N:2]>>[C:1]=[N:2]", &["CC#N"]), ["N=CC"]);
}

/// A bond on one side only is made or broken.
#[test]
fn bonds_are_made_and_broken() {
    assert_eq!(products("[C:1]-[Cl:2]>>[C:1].[Cl:2]", &["CCCl"]), ["Cl.CC"]);
    assert_eq!(products("[C:1].[N:2]>>[C:1]-[N:2]", &["C", "N"]), ["NC"]);
    // Closing a ring is just another bond that only the product side has.
    assert_eq!(
        products(
            "[C:1][C:2][C:3][C:4][O:5]>>[C:1]1[C:2][C:3][C:4][O:5]1",
            &["CCCCO"]
        ),
        ["O1CCCC1"]
    );
}

/// A map index on one side only means the atom is consumed or created.
#[test]
fn atoms_are_consumed_and_created() {
    assert_eq!(products("[C:1][Cl:2]>>[C:1]", &["CCCl"]), ["CC"]);
    assert_eq!(products("[C:1][Cl:2]>>[C:1][O:3]", &["CCCl"]), ["OCC"]);
}

/// A different element or charge on the product side is applied to the matched
/// atom.
#[test]
fn elements_change() {
    let mut expected = vec!["ONC".to_string(), "OCN".to_string()];
    expected.sort();
    assert_eq!(products("[C:1]>>[N:1]", &["CCO"]), expected);
}

/// Two reactants are combined into one graph first, so a template written
/// across a `.` can reach into both. Amide formation, losing water.
#[test]
fn intermolecular_reactions_work() {
    assert_eq!(
        products(
            "[C:1](=[O:2])[O:3].[N:4]>>[C:1](=[O:2])[N:4].[O:3]",
            &["CC(=O)O", "CN"]
        ),
        ["O.O=C(NC)C"]
    );
}

/// The template says nothing about hydrogens, so they have to be re-derived
/// for every atom whose bonding changed — otherwise oxidising ethanol would
/// leave a carbon with five bonds.
#[test]
fn hydrogens_are_rebalanced() {
    let product = ReactionSmarts::parse("[C:1]-[O:2]>>[C:1]=[O:2]")
        .unwrap()
        .apply(&MolecularGraph::from_smiles_single("CCO").unwrap())
        .swap_remove(0);
    // Acetaldehyde: 2 carbons, 1 oxygen, 4 hydrogens.
    assert_eq!(
        product.atomic_numbers().iter().filter(|&&z| z == 1).count(),
        4
    );
    assert_eq!(product.to_smiles(), "CC=O");
}

/// Hydrogens are real atoms here, so a template can name one and act on it —
/// which is how an aromatic C-H becomes a C-N.
#[test]
fn explicit_hydrogens_can_be_matched() {
    assert_eq!(
        products("[c:1][#1:2]>>[c:1][N:3]", &["c1ccccc1"]),
        ["Nc1ccccc1"]
    );
}

/// The initial restriction: every atom on both sides must carry a unique map
/// index, and an atom the product invents has to say which element it is.
#[test]
fn unmapped_or_ambiguous_templates_are_rejected() {
    assert!(matches!(
        ReactionSmarts::parse("[C:1][O:2]>>[C:1]=O"),
        Err(ReactionError::UnmappedAtom {
            side: "products",
            ..
        })
    ));
    assert!(matches!(
        ReactionSmarts::parse("CC>>C=C"),
        Err(ReactionError::UnmappedAtom {
            side: "reactants",
            ..
        })
    ));
    assert!(matches!(
        ReactionSmarts::parse("[C:1][O:1]>>[C:1]"),
        Err(ReactionError::DuplicateMapIndex {
            side: "reactants",
            index: 1
        })
    ));
    assert!(matches!(
        ReactionSmarts::parse("[C:1]>>[*:2]"),
        Err(ReactionError::NewAtomWithoutElement { index: 2 })
    ));
    assert!(matches!(
        ReactionSmarts::parse("[C:1][O:2]"),
        Err(ReactionError::NotAReaction)
    ));
}

/// A pattern that does not occur produces nothing rather than failing.
#[test]
fn a_reaction_that_does_not_apply_gives_no_products() {
    assert!(products("[C:1]-[Br:2]>>[C:1].[Br:2]", &["CCO"]).is_empty());
}

#[test]
fn test_mass_conservation_checks() {
    use crate::molecule::reaction::MassConservationError;

    // Conserved: aldol condensation [C:1]=[O:2].[C:3]=[O:4]>>[C:1]([O:2])[C:3]=[O:4]
    let rxn_good =
        ReactionSmarts::parse("[C:1]=[O:2].[C:3]=[O:4]>>[C:1]([O:2])[C:3]=[O:4]").unwrap();
    assert!(rxn_good.is_mass_conserved());

    // Atom created: [C:1]>>[C:1][C:2]
    let rxn_create = ReactionSmarts::parse("[C:1]>>[C:1][C:2]").unwrap();
    assert!(!rxn_create.is_mass_conserved());
    assert!(matches!(
        rxn_create.check_mass_conservation(),
        Err(MassConservationError::AtomCreated {
            map_index: 2,
            element: Some(6)
        })
    ));

    // Atom destroyed: [C:1][C:2]>>[C:1]
    let rxn_destroy = ReactionSmarts::parse("[C:1][C:2]>>[C:1]").unwrap();
    assert!(!rxn_destroy.is_mass_conserved());
    assert!(matches!(
        rxn_destroy.check_mass_conservation(),
        Err(MassConservationError::AtomDestroyed {
            map_index: 2,
            element: Some(6)
        })
    ));

    // Element transmutation: [C:1]>>[O:1]
    let rxn_trans = ReactionSmarts::parse("[C:1]>>[O:1]").unwrap();
    assert!(!rxn_trans.is_mass_conserved());
    assert!(matches!(
        rxn_trans.check_mass_conservation(),
        Err(MassConservationError::ElementTransmutation {
            map_index: 1,
            reactant_element: 6,
            product_element: 8
        })
    ));

    // Auto-mapping numbers the two sides apart, so C >> CC consumes the
    // reactant carbon (:1) and creates two new ones (:2, :3).
    let rxn_auto = ReactionSmarts::parse_with_auto_mapping("C>>CC").unwrap();
    assert!(!rxn_auto.is_mass_conserved());
    assert!(matches!(
        rxn_auto.check_mass_conservation(),
        Err(MassConservationError::AtomDestroyed {
            map_index: 1,
            element: Some(6)
        })
    ));

    // Atoms the template maps itself are still paired across the sides.
    let rxn_auto_good = ReactionSmarts::parse_with_auto_mapping("[C:1].[C:2]>>[C:1][C:2]").unwrap();
    assert!(rxn_auto_good.is_mass_conserved());
}

/// Canonical SMILES for each expected product, sorted, so expectations can be
/// written in any spelling.
fn canonical(expected: &[&str]) -> Vec<String> {
    let mut smiles: Vec<String> = expected
        .iter()
        .map(|s| {
            let molecules = MolecularGraph::from_smiles(s).unwrap();
            let references: Vec<&MolecularGraph> = molecules.iter().collect();
            canonize_mol(&MolecularGraph::combine(&references)).to_smiles()
        })
        .collect();
    smiles.sort();
    smiles.dedup();
    smiles
}

/// A product label that only tests — a negation or an alternative — says
/// nothing about what the atom becomes, so the matched atom keeps its element.
/// Reading the first element anywhere inside it turned `[!C:1]` into a carbon.
#[test]
fn product_query_atoms_keep_what_was_matched() {
    assert_eq!(products("[!C:1]>>[!C:1]", &["CO"]), canonical(&["CO"]));
    assert_eq!(products("[C,N:1]>>[C,N:1]", &["CN"]), canonical(&["CN"]));
    assert_eq!(
        products("[!+:1]>>[!+:1]", &["C[NH4+]"]),
        canonical(&["C[NH4+]"])
    );
    // A definite element on the product side still applies, whatever the
    // reactant side allowed.
    assert_eq!(
        products("[C,N:1]>>[O:1]", &["CN"]),
        canonical(&["CO", "NO"])
    );
}

/// The bond a product template asks for, as RDKit reads it: a simple bond is
/// set, a query bond keeps the molecule's, and an unwritten bond keeps it only
/// when the reactant side left it unwritten too.
#[test]
fn product_bonds_follow_rdkit() {
    // Comparing how the two templates *read* saw `~` and `-` as both single
    // and left the double bond alone.
    assert_eq!(
        products("[C:1]~[C:2]>>[C:1]-[C:2]", &["C=CC"]),
        canonical(&["C=CC", "CCC"])
    );
    assert_eq!(
        products("[C:1]~[C:2]>>[C:1]~[C:2]", &["C=C"]),
        canonical(&["C=C"])
    );
    assert_eq!(
        products("[C:1]=[C:2]>>[C:1]~[C:2]", &["C=C"]),
        canonical(&["C=C"])
    );
    assert_eq!(
        products("[C:1]-,=[C:2]>>[C:1]#[C:2]", &["C=C"]),
        canonical(&["C#C"])
    );
    assert_eq!(
        products("[C:1]=[C:2]>>[C:1][C:2]", &["C=CC"]),
        canonical(&["CCC"])
    );
    assert_eq!(
        products("[C:1]-,=[C:2]>>[C:1][C:2]", &["C=C"]),
        canonical(&["CC"])
    );
    assert_eq!(
        products("[#6:1][#6:2]>>[#6:1][#6:2]", &["c1ccccc1"]),
        canonical(&["c1ccccc1"])
    );
}

/// A consumed atom takes its hydrogens with it; they are part of the atom as
/// far as a template is concerned, and left behind they would be stray
/// hydrogen atoms in the product.
#[test]
fn consumed_atoms_take_their_hydrogens() {
    assert_eq!(products("[C:1][O:2]>>[C:1]", &["CCO"]), canonical(&["CC"]));
    assert_eq!(
        products("[C:1]=[O:2]>>[C:1]", &["CC=O"]),
        canonical(&["CC"])
    );
}

/// Auto-mapping numbers the unmapped atoms of the two sides apart. Numbering
/// each side from 1 paired the first unmapped reactant atom with the first
/// unmapped product atom, an identification nothing in the template asked
/// for; RDKit, like this, deletes the one and creates the other.
#[test]
fn auto_mapping_never_pairs_unmapped_atoms() {
    let reaction = ReactionSmarts::parse_with_auto_mapping("[C:1][O:2]>>[C:1]=O").unwrap();
    let molecule = MolecularGraph::from_smiles_single("CO").unwrap();
    let got: Vec<String> = reaction
        .apply(&molecule)
        .iter()
        .map(|product| canonize_mol(product).to_smiles())
        .collect();
    assert_eq!(got, canonical(&["C=O"]));

    // `C.C>>CC` consumes two carbons and creates two, so it does not conserve
    // mass atom for atom.
    let reaction = ReactionSmarts::parse_with_auto_mapping("C.C>>CC").unwrap();
    assert!(!reaction.is_mass_conserved());
}

/// Applies a double-pushout rule and returns the products as canonical
/// SMILES, sorted and deduplicated.
fn dpo_products(smarts: &str, reactants: &[&str]) -> Vec<String> {
    let reaction = ReactionSmarts::parse_dpo(smarts).unwrap_or_else(|e| panic!("{smarts}: {e:?}"));
    apply_canonical(&reaction, reactants)
}

fn apply_canonical(reaction: &ReactionSmarts, reactants: &[&str]) -> Vec<String> {
    let molecules: Vec<MolecularGraph> = reactants
        .iter()
        .map(|s| MolecularGraph::from_smiles_single(s).unwrap())
        .collect();
    let references: Vec<&MolecularGraph> = molecules.iter().collect();
    let mut smiles: Vec<String> = reaction
        .apply_to(&references)
        .iter()
        .map(|product| canonize_mol(product).to_smiles())
        .collect();
    smiles.sort();
    smiles.dedup();
    smiles
}

/// The oxidation of ethanol written as a double-pushout rule: the two
/// hydrogens that leave are named, and they leave as H2.
const OXIDATION: &str = "[C:1](-[#1:3])-[O:2]-[#1:4]>>[C:1]=[O:2].[#1:3]-[#1:4]";

/// With hydrogens explicit, a rule that makes a bond and gives none up is
/// unbalanced, and it is rejected when parsed rather than producing nothing.
#[test]
fn dpo_rules_must_balance_valence() {
    assert!(matches!(
        ReactionSmarts::parse_dpo("[C:1]-[O:2]>>[C:1]=[O:2]"),
        Err(ReactionError::ValenceNotBalanced {
            index: 1,
            change: 1
        })
    ));
    assert!(matches!(
        ReactionSmarts::parse_dpo("[C:1]-[Cl:2]>>[C:1]"),
        Err(ReactionError::ValenceNotBalanced {
            index: 1,
            change: -1
        })
    ));
    // Sulfur may go from two bonds to four; the check allows any pair of its
    // valences.
    assert!(
        ReactionSmarts::parse_dpo("[S:1](-[#1:2])-[#1:3]>>[S:1](=[O:4])(-[#1:2])-[#1:3]").is_ok()
    );
    assert!(ReactionSmarts::parse_dpo(OXIDATION).is_ok());
}

/// The product side assigns; a label or bond there that tests is rejected.
#[test]
fn dpo_product_labels_must_assign() {
    for smarts in [
        "[C:1]>>[C,N:1]",
        "[C:1]>>[!N:1]",
        "[C:1]>>[CH2:1]",
        "[C:1]>>[C;R:1]",
        "[C:1]-[C:2]>>[C:1]-,=[C:2]",
        "[C:1].[C:2]>>[C:1]~[C:2]",
    ] {
        assert!(
            matches!(
                ReactionSmarts::parse_dpo(smarts),
                Err(ReactionError::ProductQuery { .. })
            ),
            "{smarts}"
        );
    }
    // `*` and `~` on what the rule keeps mean "as matched".
    assert!(ReactionSmarts::parse_dpo("[*:1]~[C:2]>>[*:1]~[C:2]").is_ok());
}

/// Nothing happens that the rule does not write: no hydrogens are added or
/// removed, so the ones it names are all that move.
#[test]
fn dpo_moves_only_what_the_rule_names() {
    assert_eq!(
        dpo_products(OXIDATION, &["CCO"]),
        canonical(&["CC=O.[H][H]"])
    );
    let product = ReactionSmarts::parse_dpo(OXIDATION)
        .unwrap()
        .apply(&MolecularGraph::from_smiles_single("CCO").unwrap())
        .swap_remove(0);
    assert_eq!(product.atomic_numbers().len(), 9);
}

/// A consumed atom with a bond the rule does not account for would leave
/// that bond dangling, so the match is rejected rather than the bond cut.
#[test]
fn dpo_rejects_dangling_bonds() {
    let deoxygenation = "[C:1]-[O:2]-[#1:3]>>[C:1]-[#1:3]";
    assert_eq!(dpo_products(deoxygenation, &["CO"]), canonical(&["C"]));
    // The oxonium oxygen has a second hydrogen the rule does not name.
    assert!(dpo_products(deoxygenation, &["C[OH2+]"]).is_empty());
    // RDKit's reading cuts it loose instead.
    assert!(!products(deoxygenation, &["C[OH2+]"]).is_empty());
}

/// A charge the rule sets without changing any bond leaves the atom off its
/// valences, and with no hydrogens re-derived the product is rejected.
#[test]
fn dpo_rejects_products_off_their_valences() {
    assert!(dpo_products("[N:1]>>[N+:1]", &["CN"]).is_empty());
    // Protonation names the proton and its bond, and is a molecule.
    assert_eq!(
        dpo_products("[N:1].[#1+:2]>>[N+:1]-[#1+0:2]", &["CN", "[H+]"]),
        canonical(&["C[NH3+]"])
    );
}

/// A product is perceived as a parsed molecule is, so a ring the rule closes
/// into a benzene comes out aromatic and matches `c1ccccc1`.
#[test]
fn dpo_products_are_aromatized() {
    let dehydrogenation = "[C:1](-[#1:3])-[C:2]-[#1:4]>>[C:1]=[C:2].[#1:3]-[#1:4]";
    let got = dpo_products(dehydrogenation, &["C1=CC=CCC1"]);
    assert!(got.contains(&canonical(&["c1ccccc1.[H][H]"])[0]), "{got:?}");
}

/// The inverse undoes the rule: reducing acetaldehyde with H2 gives ethanol.
#[test]
fn dpo_rules_invert() {
    let reduction = ReactionSmarts::parse_dpo(OXIDATION)
        .unwrap()
        .inverse()
        .unwrap();
    assert_eq!(
        apply_canonical(&reduction, &["CC=O", "[H][H]"]),
        canonical(&["CCO"])
    );
    // The inverse of an inverse is the rule again.
    let oxidation = reduction.inverse().unwrap();
    assert_eq!(
        apply_canonical(&oxidation, &["CCO"]),
        canonical(&["CC=O.[H][H]"])
    );
    // A reactant label that tests cannot become a product label.
    let ring_only = ReactionSmarts::parse_dpo("[C;R:1]=[C:2]>>[C:1]=[C:2]").unwrap();
    assert!(matches!(
        ring_only.inverse(),
        Err(ReactionError::ProductQuery { index: 1, .. })
    ));
    assert!(matches!(
        ReactionSmarts::parse("[C:1]-[O:2]>>[C:1]=[O:2]")
            .unwrap()
            .inverse(),
        Err(ReactionError::NotInvertible)
    ));
}

const KETO_ENOL: &str = r#"
rule [
   ruleID "keto-enol"
   left [
      edge [ source 1 target 2 label "=" ]
      edge [ source 2 target 3 label "-" ]
      edge [ source 3 target 4 label "-" ]
   ]
   context [
      node [ id 1 label "C" ]
      node [ id 2 label "C" ]
      node [ id 3 label "O" ]
      node [ id 4 label "H" ]
   ]
   right [
      edge [ source 1 target 2 label "-" ]
      edge [ source 2 target 3 label "=" ]
      edge [ source 1 target 4 label "-" ]
   ]
]
"#;

/// A MØD rule reads as a double-pushout rule: context atoms are kept, and
/// the edges of `left` and `right` are broken and made.
#[test]
fn gml_rules_apply() {
    let rule = ReactionSmarts::from_gml(KETO_ENOL).unwrap();
    assert_eq!(rule.semantics(), crate::molecule::reaction::Semantics::Dpo);
    assert_eq!(apply_canonical(&rule, &["C=CO"]), canonical(&["CC=O"]));
    // Its inverse is the enolisation.
    assert_eq!(
        apply_canonical(&rule.inverse().unwrap(), &["CC=O"]),
        canonical(&["C=CO"])
    );
}

/// Charged and isotopic labels read as MØD writes them, and are exact: `O-`
/// matches an oxide and not a hydroxyl.
#[test]
fn gml_labels_are_exact() {
    let protonation = r#"rule [
        ruleID "protonate alkoxide"
        left [ node [ id 1 label "O-" ] node [ id 2 label "H+" ] ]
        right [ node [ id 1 label "O" ] node [ id 2 label "H" ] edge [ source 1 target 2 label "-" ] ]
    ]"#;
    let rule = ReactionSmarts::from_gml(protonation).unwrap();
    assert_eq!(
        apply_canonical(&rule, &["C[O-]", "[H+]"]),
        canonical(&["CO"])
    );
    assert!(apply_canonical(&rule, &["CO", "[H+]"]).is_empty());

    let gml = rule.to_gml("again").unwrap();
    assert!(gml.contains("label \"O-\""), "{gml}");
    assert!(gml.contains("label \"H+\""), "{gml}");
}

/// Writing a rule and reading it back gives a rule that does the same.
#[test]
fn gml_rules_round_trip() {
    let oxidation = ReactionSmarts::parse_dpo(OXIDATION).unwrap();
    let gml = oxidation.to_gml("oxidation").unwrap();
    // The carbon and oxygen keep their labels; the two hydrogens' bonds move.
    assert!(
        gml.contains("context [\n      node [ id 1 label \"C\" ]"),
        "{gml}"
    );
    let read_back = ReactionSmarts::from_gml(&gml).unwrap();
    for reactant in ["CCO", "OCCO", "CC(C)O", "CC(C)(C)O"] {
        assert_eq!(
            apply_canonical(&read_back, &[reactant]),
            apply_canonical(&oxidation, &[reactant]),
            "{reactant}"
        );
    }
    let keto_enol = ReactionSmarts::from_gml(KETO_ENOL).unwrap();
    let again = ReactionSmarts::from_gml(&keto_enol.to_gml("keto-enol").unwrap()).unwrap();
    assert_eq!(apply_canonical(&again, &["C=CO"]), canonical(&["CC=O"]));
}

/// What a GML rule cannot say, or this reader cannot read, is refused.
#[test]
fn gml_refuses_what_it_cannot_express() {
    use crate::io::GmlRuleError;
    let term = r#"rule [ ruleID "x" context [ node [ id 1 label "_X" ] ] ]"#;
    assert!(matches!(
        ReactionSmarts::from_gml(term),
        Err(GmlRuleError::Unsupported(_))
    ));
    let constrained = r#"rule [ ruleID "x" context [ node [ id 1 label "C" ] ] constrainAdj [ id 1 op "=" count 4 ] ]"#;
    assert!(matches!(
        ReactionSmarts::from_gml(constrained),
        Err(GmlRuleError::Unsupported(_))
    ));
    let dangling_edge = r#"rule [ ruleID "x" left [ node [ id 1 label "C" ] ] right [ edge [ source 1 target 2 label "-" ] ] ]"#;
    assert!(matches!(
        ReactionSmarts::from_gml(dangling_edge),
        Err(GmlRuleError::Syntax(_))
    ));
    // An element alternative has no exact label.
    let either = ReactionSmarts::parse_dpo("[C,N:1]-[#1:2]>>[*:1]-[#1:2]").unwrap();
    assert!(matches!(
        either.to_gml("x"),
        Err(GmlRuleError::NotExpressible { index: 1, .. })
    ));
}

/// Matches that differ only by a symmetry of the rule make one derivation:
/// ethane's single C-C bond is one site, however the two carbons are mapped.
#[test]
fn dpo_derivations_are_counted_once() {
    let rule = ReactionSmarts::parse_dpo("[C:1](-[#1:3])-[C:2]-[#1:4]>>[C:1]=[C:2].[#1:3]-[#1:4]")
        .unwrap();
    let ethane = MolecularGraph::from_smiles_single("CC").unwrap();
    // Nine hydrogen pairs, one per choice of a hydrogen on each carbon, but
    // each is matched twice over.
    assert_eq!(rule.apply(&ethane).len(), 9);
    let rdkit = ReactionSmarts::parse("[C:1][C:2]>>[C:1]=[C:2]").unwrap();
    assert_eq!(rdkit.apply(&ethane).len(), 2);
}

/// Charging an atom changes how many bonds it forms, and its hydrogens follow,
/// as RDKit's do: ammonium is tetravalent, an alkoxide monovalent.
#[test]
fn charges_change_hydrogen_counts() {
    assert_eq!(products("[N:1]>>[N+:1]", &["CN"]), canonical(&["C[NH3+]"]));
    assert_eq!(products("[O:1]>>[O-:1]", &["CO"]), canonical(&["C[O-]"]));
    assert_eq!(products("[C:1]>>[C+:1]", &["C"]), canonical(&["[CH3+]"]));
}

/// Products are perceived as a parsed molecule is, as RDKit sanitizes them:
/// a ring the reaction makes aromatic comes out aromatic.
#[test]
fn products_are_aromatized() {
    let product = ReactionSmarts::parse("[C;H2:1]-[C;H2:2]>>[C:1]=[C:2]")
        .unwrap()
        .apply(&MolecularGraph::from_smiles_single("C1=CC=CCC1").unwrap())
        .swap_remove(0);
    assert_eq!(product.to_smiles(), "c1ccccc1");
}

/// Stereo through a reaction, as RDKit's `RunReactants` handles it.
#[test]
fn stereo_follows_rdkit() {
    // Unspecified on both sides: kept, the new neighbour where the old stood.
    assert_eq!(
        products("[C:1]-[Cl:2]>>[C:1]-[O:3]", &["C[C@H](Cl)F"]),
        canonical(&["C[C@H](O)F"])
    );
    // Opposite tags invert; equal tags retain.
    assert_eq!(
        products("[C@:1]-[Cl:2]>>[C@@:1]-[O:3]", &["C[C@H](Cl)F"]),
        canonical(&["C[C@@H](O)F"])
    );
    assert_eq!(
        products("[C@:1]-[Cl:2]>>[C@:1]-[O:3]", &["C[C@H](Cl)F"]),
        canonical(&["C[C@H](O)F"])
    );
    // A tag the product side drops is removed.
    assert_eq!(
        products("[C@:1]-[Cl:2]>>[C:1]-[O:3]", &["C[C@H](Cl)F"]),
        canonical(&["CC(O)F"])
    );
    // Changes elsewhere leave a centre alone.
    assert_eq!(
        products("[O:1]>>[O:1]-[C:2]", &["C[C@H](O)F"]),
        canonical(&["CO[C@@H](C)F"])
    );
    assert_eq!(
        products("[C:1][O:2]>>[C:1][N:2]", &["C[C@H](O)F"]),
        canonical(&["C[C@H](N)F"])
    );
    // A double bond made single takes its cis/trans with it.
    assert_eq!(
        products("[C:1]=[C:2]>>[C:1]-[C:2]", &["C/C=C/C"]),
        canonical(&["CCCC"])
    );
}

/// A product template that names every neighbour of a centre sets its
/// configuration outright, from its own written order.
#[test]
fn product_templates_set_stereo() {
    assert_eq!(
        products(
            "[C:1](-[C:5])(-[F:4])-[Cl:2]>>[C@@:1](-[C:5])(-[F:4])-[O:3]",
            &["CC(Cl)F"]
        ),
        canonical(&["[C@@H](C)(F)O"])
    );
    // The same under the double pushout, hydrogen named.
    assert_eq!(
        dpo_products(
            "[C:1](-[#1:6])(-[C:5])(-[F:4])-[Cl:2].[O:3]-[#1:7]>>[C@@:1](-[#1:6])(-[C:5])(-[F:4])-[O:3].[Cl:2]-[#1:7]",
            &["CC(Cl)F", "O"]
        ),
        canonical(&["[C@@H](C)(F)O.Cl"])
    );
}

/// A negative condition rejects a match wherever it extends it.
#[test]
fn forbidden_patterns_reject_matches() {
    let cleavage = ReactionSmarts::parse("[C:1]-[O:2]>>[C:1].[O:2]")
        .unwrap()
        .forbid("[C:1]=O")
        .unwrap();
    // Only the methyl C-O of methyl acetate is cut, not the acyl one.
    assert_eq!(
        apply_canonical(&cleavage, &["CC(=O)OC"]),
        canonical(&["CC(=O)O.C"])
    );
    // An atom the condition leaves unmapped is some atom other than those
    // matched, so ethanol's own oxygen does not count as a second one.
    let second_oxygen = ReactionSmarts::parse("[C:1]-[O:2]>>[C:1].[O:2]")
        .unwrap()
        .forbid("[C:1]-O")
        .unwrap();
    assert_eq!(
        apply_canonical(&second_oxygen, &["CCO"]),
        canonical(&["CC.O"])
    );
    assert!(apply_canonical(&second_oxygen, &["OCO"]).is_empty());
}

/// A positive condition keeps only the matches it extends.
#[test]
fn required_patterns_keep_matches() {
    let cleavage = ReactionSmarts::parse("[C:1]-[O:2]>>[C:1].[O:2]")
        .unwrap()
        .require("[C:1]=O")
        .unwrap();
    assert_eq!(
        apply_canonical(&cleavage, &["CC(=O)OC"]),
        canonical(&["CC=O.CO"])
    );
    // Conditions hold under the double pushout too.
    let rule = ReactionSmarts::parse_dpo(OXIDATION)
        .unwrap()
        .forbid("[C:1]-[C;X4]-[C;X4]")
        .unwrap();
    assert_eq!(
        apply_canonical(&rule, &["CCO"]),
        canonical(&["CC=O.[H][H]"])
    );
    assert!(apply_canonical(&rule, &["CCCO"]).is_empty());
}

/// A condition must pin itself to atoms the rule matches, and a rule that
/// carries one cannot be inverted.
#[test]
fn conditions_are_checked() {
    assert!(matches!(
        ReactionSmarts::parse("[C:1]-[O:2]>>[C:1].[O:2]")
            .unwrap()
            .forbid("[C:3]=O"),
        Err(ReactionError::UnknownConditionIndex { index: 3 })
    ));
    assert!(matches!(
        ReactionSmarts::parse_dpo(OXIDATION)
            .unwrap()
            .forbid("[C:1]=O")
            .unwrap()
            .inverse(),
        Err(ReactionError::ConditionsNotInvertible)
    ));
}

/// Two spellings of one rule share a canonical form; different rules do not.
#[test]
fn rules_have_a_canonical_form() {
    let oxidation = ReactionSmarts::parse_dpo(OXIDATION).unwrap();
    let renumbered =
        ReactionSmarts::parse_dpo("[O:5](-[#1:9])-[C:7]-[#1:2]>>[#1:2]-[#1:9].[O:5]=[C:7]")
            .unwrap();
    assert!(oxidation.is_isomorphic_to(&renumbered).unwrap());
    assert_eq!(
        oxidation.canonical_smarts().unwrap(),
        renumbered.canonical_smarts().unwrap()
    );
    // Its inverse is a different rule.
    assert!(
        !oxidation
            .is_isomorphic_to(&oxidation.inverse().unwrap())
            .unwrap()
    );
    // Swapping which hydrogen leaves with which is the same rule again.
    let swapped =
        ReactionSmarts::parse_dpo("[C:1](-[#1:4])-[O:2]-[#1:3]>>[C:1]=[O:2].[#1:4]-[#1:3]")
            .unwrap();
    assert!(oxidation.is_isomorphic_to(&swapped).unwrap());
    // RDKit's reading has no exact form.
    assert!(matches!(
        ReactionSmarts::parse("[C:1]-[O:2]>>[C:1]=[O:2]")
            .unwrap()
            .canonical_smarts(),
        Err(crate::molecule::rule_algebra::ExactRuleError::NotDpo)
    ));
}

/// Composing a rule with its inverse gives, among the overlaps, the rule that
/// changes nothing, and every composite does what the two do in sequence.
#[test]
fn rules_compose() {
    let oxidation = ReactionSmarts::parse_dpo(OXIDATION).unwrap();
    let reduction = oxidation.inverse().unwrap();
    let composites = oxidation.compose(&reduction).unwrap();
    assert!(!composites.is_empty());
    assert!(composites.iter().any(|rule| {
        let exact = rule.to_exact().unwrap();
        exact.left == exact.right
    }));

    // Oxidation then oxidation again: the parallel composite, two oxidations
    // on separate atoms, turns ethylene glycol into glyoxal in one step.
    let twice = oxidation.compose(&oxidation).unwrap();
    let glycol = MolecularGraph::from_smiles_single("OCCO").unwrap();
    let glyoxal = canonical(&["O=CC=O.[H][H].[H][H]"]);
    assert!(twice.iter().any(|rule| {
        rule.apply(&glycol)
            .iter()
            .any(|product| canonize_mol(product).to_smiles() == glyoxal[0])
    }));
    // Every composite applied to a molecule gives only products that the two
    // steps give in turn.
    let two_steps: Vec<String> = oxidation
        .apply(&glycol)
        .iter()
        .flat_map(|middle| oxidation.apply(middle))
        .map(|product| canonize_mol(&product).to_smiles())
        .collect();
    for rule in &twice {
        for product in rule.apply(&glycol) {
            let smiles = canonize_mol(&product).to_smiles();
            assert!(
                two_steps.contains(&smiles),
                "{smiles} from {}",
                rule.canonical_smarts().unwrap()
            );
        }
    }
}

/// Two rules competing for the same atoms form critical pairs; rules that
/// cannot share an atom form none.
#[test]
fn critical_pairs_are_found() {
    let oxidation = ReactionSmarts::parse_dpo(OXIDATION).unwrap();
    let keto_enol = ReactionSmarts::from_gml(KETO_ENOL).unwrap();
    let pairs = oxidation.critical_pairs(&oxidation).unwrap();
    // Two oxidations sharing the hydroxyl: whichever goes first takes it.
    assert!(
        pairs
            .iter()
            .any(|pair| pair.first_disables_second && pair.second_disables_first)
    );

    // The enol's O-H is what both the tautomerisation and an oxidation of
    // the same oxygen would use.
    assert!(!keto_enol.critical_pairs(&oxidation).unwrap().is_empty());

    let amine = ReactionSmarts::parse_dpo("[N:1]-[#1:2]>>[N-:1].[#1+:2]").unwrap();
    let thiol = ReactionSmarts::parse_dpo("[S:1]-[#1:2]>>[S-:1].[#1+:2]").unwrap();
    assert!(amine.critical_pairs(&thiol).unwrap().is_empty());
}

/// A double-pushout rule may be written against a Kekulé structure: `=`
/// matches an aromatic bond wherever some Kekulé structure makes it double.
#[test]
fn dpo_rules_match_kekule_structures() {
    // H2 across one double bond of benzene gives cyclohexa-1,3-diene.
    let hydrogenation = "[C:1]=[C:2].[#1:3]-[#1:4]>>[C:1](-[#1:3])-[C:2]-[#1:4]";
    assert_eq!(
        dpo_products(hydrogenation, &["c1ccccc1", "[H][H]"]),
        canonical(&["C1=CC=CCC1"])
    );
    // No Kekulé structure gives a ring carbon two double bonds.
    assert!(dpo_products("[C:1]=[C:2]=[C:3]>>[C:1]=[C:2]=[C:3]", &["c1ccccc1"]).is_empty());
    // A diene unit is found in naphthalene, and rewriting it to itself
    // gives naphthalene back, perceived aromatic again.
    assert_eq!(
        dpo_products(
            "[C:1]=[C:2]-[C:3]=[C:4]>>[C:1]=[C:2]-[C:3]=[C:4]",
            &["c1ccc2ccccc2c1"]
        ),
        canonical(&["c1ccc2ccccc2c1"])
    );
    // A pyridine C=N is double in one of its Kekulé structures.
    assert_eq!(
        dpo_products("[C:1]=[N:2]>>[C:1]=[N:2]", &["c1ccncc1"]),
        canonical(&["c1ccncc1"])
    );
    // RDKit's reading still holds `C` to aliphatic carbon.
    assert!(products("[C:1]=[C:2]>>[C:1]-[C:2]", &["c1ccccc1"]).is_empty());
}

/// Under RDKit's reading the hydrogens a reaction displaces are re-derived;
/// the centre keeps its configuration, the new neighbour taking the place of
/// the hydrogen or the new hydrogen that of the lost neighbour — as the same
/// reaction does with the hydrogens written out.
#[test]
fn rederived_hydrogens_keep_stereo() {
    let gained = products("[C:1]>>[C:1]-[O:2]", &["C[C@H](F)Cl"]);
    let gained_explicit = dpo_products(
        "[C:1]-[#1:3].[O:2](-[#1:4])-[#1:5]>>[C:1]-[O:2]-[#1:4].[#1:3]-[#1:5]",
        &["C[C@H](F)Cl", "O"],
    );
    assert!(
        gained.contains(&canonical(&["C[C@](O)(F)Cl"])[0]),
        "{gained:?}"
    );
    assert!(
        gained_explicit.contains(&canonical(&["C[C@](O)(F)Cl.[H][H]"])[0]),
        "{gained_explicit:?}"
    );

    assert_eq!(
        products("[C:1]-[Cl:2]>>[C:1]", &["C[C@@](F)(Cl)Br"]),
        canonical(&["C[C@H](F)Br"])
    );
    assert_eq!(
        dpo_products(
            "[C:1]-[Cl:2].[#1:3]-[#1:4]>>[C:1]-[#1:3].[Cl:2]-[#1:4]",
            &["C[C@@](F)(Cl)Br", "[H][H]"]
        ),
        canonical(&["C[C@H](F)Br.Cl"])
    );
}

/// Hydrogens on one carbon are interchangeable, and the overlap walk tries
/// one of each such set rather than every order of them: without that, this
/// propanol oxidation took two minutes to compose with its inverse. The
/// counts are those of the unpruned walk.
#[test]
fn rule_algebra_prunes_interchangeable_atoms() {
    let propanol = ReactionSmarts::parse_dpo(
        "[C:10](-[#1:11])(-[#1:12])(-[#1:13])-[C:1](-[#1:2])(-[#1:3])-[C:7](-[#1:8])(-[#1:9])-[O:5]-[#1:6]>>[C:10](-[#1:11])(-[#1:12])(-[#1:13])-[C:1](-[#1:2])(-[#1:3])-[C:7](-[#1:8])=[O:5].[#1:9]-[#1:6]",
    )
    .unwrap();
    let start = std::time::Instant::now();
    assert_eq!(
        propanol
            .compose(&propanol.inverse().unwrap())
            .unwrap()
            .len(),
        6
    );
    assert_eq!(propanol.critical_pairs(&propanol).unwrap().len(), 1);
    assert!(start.elapsed() < std::time::Duration::from_secs(10));

    let oxidation = ReactionSmarts::parse_dpo(OXIDATION).unwrap();
    let reduction = oxidation.inverse().unwrap();
    assert_eq!(oxidation.compose(&reduction).unwrap().len(), 9);
    assert_eq!(oxidation.compose(&oxidation).unwrap().len(), 2);
    assert_eq!(oxidation.critical_pairs(&oxidation).unwrap().len(), 3);
    assert!(reduction.critical_pairs(&oxidation).unwrap().is_empty());
}
