//! Applying reaction SMARTS, checked against RDKit's `RunReactants`.
//!
//! The full differential is `notes/reaction_oracle.py` +
//! `examples/reaction_diff.rs`; 13 of its 15 cases agree exactly. Of the two
//! that do not:
//!
//! - `[c:1][#1:2]>>[c:1][N:3]` turns benzene into aniline here and produces
//!   nothing in RDKit, because our molecules carry hydrogens as real atoms and
//!   RDKit's are implicit, so `[#1:2]` has nothing to match there.
//! - `[N:1]>>[N+:1]` gives a nitrogen one hydrogen where RDKit gives three.
//!   Charging an atom should change how many bonds it forms — ammonium is
//!   tetravalent — and `get_atom_degree` models a charge as always *costing* a
//!   bond. See `todo.txt`; it cannot be fixed without also fixing the
//!   bracket-hydrogen rule.

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
        products("[C:1][C:2][C:3][C:4][O:5]>>[C:1]1[C:2][C:3][C:4][O:5]1", &["CCCCO"]),
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
    assert_eq!(product.atomic_numbers().iter().filter(|&&z| z == 1).count(), 4);
    assert_eq!(product.to_smiles(), "CC=O");
}

/// Hydrogens are real atoms here, so a template can name one and act on it —
/// which is how an aromatic C-H becomes a C-N.
#[test]
fn explicit_hydrogens_can_be_matched() {
    assert_eq!(products("[c:1][#1:2]>>[c:1][N:3]", &["c1ccccc1"]), ["Nc1ccccc1"]);
}

/// The initial restriction: every atom on both sides must carry a unique map
/// index, and an atom the product invents has to say which element it is.
#[test]
fn unmapped_or_ambiguous_templates_are_rejected() {
    assert!(matches!(
        ReactionSmarts::parse("[C:1][O:2]>>[C:1]=O"),
        Err(ReactionError::UnmappedAtom { side: "products", .. })
    ));
    assert!(matches!(
        ReactionSmarts::parse("CC>>C=C"),
        Err(ReactionError::UnmappedAtom { side: "reactants", .. })
    ));
    assert!(matches!(
        ReactionSmarts::parse("[C:1][O:1]>>[C:1]"),
        Err(ReactionError::DuplicateMapIndex { side: "reactants", index: 1 })
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
