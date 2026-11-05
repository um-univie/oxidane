//! SMARTS matching, checked against RDKit.
//!
//! The expectations here were taken from `GetSubstructMatches(..,
//! uniquify=False)` on the same molecule with `Chem.AddHs` applied, because
//! this crate's molecules always carry hydrogens as real atoms. The full
//! differential lives in `tests/smarts_oracle.py` and
//! `examples/smarts_diff.rs`; what is pinned here is one case per defect that
//! differential found.
//!
//! On 400 molecules against 69 patterns -- 27600 pairs -- 27555 agree. The
//! forty-five that do not are molecules the two toolkits perceive differently,
//! and atoms in bridged cages, where RDKit counts a symmetrised set of smallest
//! rings and this crate counts a minimum one.

use crate::io::find_cxsmarts_matches;
use crate::prelude::*;

fn count(pattern: &str, target: &str) -> usize {
    let mol = MolecularGraph::from_smiles_single(target)
        .unwrap_or_else(|e| panic!("{target}: {e:?}"));
    find_cxsmarts_matches(&mol, pattern)
        .unwrap_or_else(|e| panic!("{pattern}: {e:?}"))
        .len()
}

/// A bond written without a symbol is still a bond.
///
/// The parser only created a bond when a bond *character* was present, so
/// every pattern with an implicit bond described unconnected atoms. The
/// visible symptom was that a molecule did not match itself.
#[test]
fn implicit_bonds_are_parsed() {
    for smiles in ["CC", "CCC", "CCO", "CC(C)C", "CC(=O)O", "C1CC1", "c1ccccc1"] {
        assert!(count(smiles, smiles) > 0, "{smiles} does not match itself");
    }

    let pattern = parse_cxsmarts("CCO").unwrap();
    assert_eq!(pattern.atoms.len(), 3);
    assert_eq!(pattern.bonds.len(), 2, "implicit bonds missing: {:?}", pattern.bonds);
}

/// An unwritten bond means "single or aromatic", not "single" and not "any".
#[test]
fn the_default_bond_is_single_or_aromatic() {
    assert_eq!(count("CC", "CC"), 2);
    assert_eq!(count("cc", "c1ccccc1"), 12);
    // A double bond is neither single nor aromatic.
    assert_eq!(count("CC", "C=C"), 0);
    assert_eq!(count("C=C", "C=C"), 2);
    // Aliphatic and aromatic carbons are still distinct atoms.
    assert_eq!(count("CC", "c1ccccc1"), 0);
}

/// Branches bond to the atom the branch hangs off, and ring-closure digits
/// create the bond that closes the ring.
///
/// Attaching bonds as `(current - 1, current)` is wrong for both: after `)`
/// the next atom is not adjacent to the last one parsed, and a ring's two ends
/// are never adjacent in the string.
#[test]
fn branches_and_ring_closures_bond_correctly() {
    let branched = parse_cxsmarts("CC(C)C").unwrap();
    let mut bonds: Vec<(usize, usize)> = branched.bonds.iter().map(|&(a, b, _)| (a, b)).collect();
    bonds.sort_unstable();
    assert_eq!(bonds, vec![(0, 1), (1, 2), (1, 3)], "the last C must bond to atom 1");

    let ring = parse_cxsmarts("C1CCCCC1").unwrap();
    let mut bonds: Vec<(usize, usize)> = ring.bonds.iter().map(|&(a, b, _)| (a, b)).collect();
    bonds.sort_unstable();
    assert_eq!(bonds, vec![(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)]);

    assert!(count("CC(C)C", "CC(C)C") > 0);
    assert!(count("C1CCCCC1", "C1CCCCC1") > 0);
    assert_eq!(count("C1CC1", "CCC"), 0, "a ring pattern must not match a chain");
}

/// Outside brackets the only two-letter symbols are `Cl` and `Br`, so an
/// uppercase atom followed by a lowercase one is *two* atoms.
///
/// Reading any such pair as a two-letter element turned `Cc1ccccc1` into six
/// atoms of a non-existent element "Cc", which silently broke every
/// aliphatic-on-aromatic pattern — `Nc1ccccc1`, `Oc1ccccc1` and so on.
#[test]
fn an_aliphatic_atom_before_an_aromatic_one_is_two_atoms() {
    assert_eq!(parse_cxsmarts("Cc").unwrap().atoms.len(), 2);
    assert_eq!(parse_cxsmarts("Nc1ccccc1").unwrap().atoms.len(), 7);
    assert_eq!(parse_cxsmarts("Cc1ccccc1").unwrap().atoms.len(), 7);
    // Cl and Br really are single atoms.
    assert_eq!(parse_cxsmarts("Cl").unwrap().atoms.len(), 1);
    assert_eq!(parse_cxsmarts("CBr").unwrap().atoms.len(), 2);

    assert!(count("Cc1ccccc1", "Cc1ccccc1") > 0);
    assert_eq!(count("Cc1ccccc1", "c1ccc(cc1)C(=O)O"), 2);
}

/// SMARTS matching is subgraph *monomorphism*: the molecule may have bonds
/// between matched atoms that the pattern does not mention.
///
/// `CCC` is a three-atom chain and occurs in cyclopropane six ways, even
/// though its two ends are bonded there and the pattern says nothing about
/// that. A node-induced search — which is what petgraph provides, and what
/// this used to call — rejects all six.
#[test]
fn matching_is_monomorphism_not_induced_isomorphism() {
    assert_eq!(count("CCC", "C1CC1"), 6);
    assert_eq!(count("CC", "C1CC1"), 6);
}

/// A `.` in a pattern means the fragments are matched independently, not that
/// the match fails.
#[test]
fn disconnected_patterns_match() {
    assert_eq!(count("C.C", "CC"), 2);
    assert_eq!(count("C.O", "CCO"), 2);
    assert_eq!(count("C.C", "CCC"), 6);
}

/// Wildcards: `*` is the query wildcard and matches anything, `[#0]` picks out
/// an actual wildcard *atom* (atomic number 0), and an element query never
/// matches one.
#[test]
fn wildcards_match_the_right_things() {
    assert_eq!(count("*", "*CC"), 8, "2 carbons + 1 wildcard + 5 hydrogens");
    assert_eq!(count("C", "*CC"), 2, "the carbons only");
    assert_eq!(count("[#6]", "*CC"), 2);
    assert_eq!(count("[#0]", "*CC"), 1, "the wildcard atom");
    assert_eq!(count("[#0]", "CCO"), 0);
}

/// The reported mapping must say which *molecule* atoms were matched.
///
/// The identity `index -> index` would point every match back at the
/// pattern's own atom indices and tell the caller nothing.
#[test]
fn reported_matches_name_real_molecule_atoms() {
    // Ethanol: atoms 0 and 1 are carbons, 2 is the oxygen, 3.. are hydrogens.
    let mol = MolecularGraph::from_smiles_single("CCO").unwrap();
    let hits = find_cxsmarts_matches(&mol, "CO").unwrap();
    assert_eq!(hits.len(), 1);
    let matched = &hits[0];
    assert_eq!(matched.len(), 2);
    assert!(
        matched.iter().any(|&a| mol.atomic_numbers()[a] == 8),
        "no oxygen among the matched atoms: {matched:?}"
    );
    assert!(
        matched.iter().any(|&a| mol.atomic_numbers()[a] == 6),
        "no carbon among the matched atoms: {matched:?}"
    );
}

/// `[R]` asks whether an atom is on *any* ring, and must be exact for fused
/// and bridged systems.
///
/// Two defects met here. The parser read bare `R` as `R1` ("in exactly one
/// ring"), which excludes every atom shared between two rings; and ring
/// perception collected a depth-first cycle basis, which loses fusion atoms
/// entirely. Naphthalene reported 8 of its 10 ring atoms and
/// bicyclo[2.2.2]octane reported none of its 8.
#[test]
fn ring_membership_covers_fused_and_bridged_systems() {
    assert_eq!(count("[R]", "c1ccc2ccccc2c1"), 10, "naphthalene has 10 ring atoms");
    assert_eq!(count("[R]", "C1CC2CCC1CC2"), 8, "bicyclo[2.2.2]octane has 8 ring atoms");
    assert_eq!(count("[R]", "C1CCCCC1"), 6);
    assert_eq!(count("[R]", "CCC"), 0);

    // `[!R]` is the complement, hydrogens included.
    assert_eq!(count("[!R]", "c1ccc2ccccc2c1"), 8, "naphthalene's 8 hydrogens");
    assert_eq!(count("[!R]", "C1CC2CCC1CC2"), 14, "bicyclo[2.2.2]octane's 14 hydrogens");
}

/// Aromaticity is perceived by default, so a ring written in Kekulé form and
/// the same ring written in lower case answer the same questions.
#[test]
fn aromaticity_is_perceived_by_default_and_can_be_turned_off() {
    use crate::io::SmilesParseOptions;

    assert_eq!(count("c1ccccc1", "C1=CC=CC=C1"), 12);
    assert_eq!(count("c1ccccc1", "c1ccccc1"), 12);
    assert_eq!(count("cc", "C1=CC2=CC=CC=C2C=C1"), 22, "naphthalene, written Kekulé");

    // Turning it off keeps the bonds exactly as written, so the Kekulé ring
    // has plain double bonds and no aromatic ones.
    let literal = MolecularGraph::from_smiles_single_with(
        "C1=CC=CC=C1",
        SmilesParseOptions { aromaticity: None, ..Default::default() },
    )
    .unwrap();
    assert_eq!(find_cxsmarts_matches(&literal, "c1ccccc1").unwrap().len(), 0);
    assert_eq!(find_cxsmarts_matches(&literal, "C=C").unwrap().len(), 6);
}

/// Reusing one analysis of a molecule across several patterns has to give
/// exactly what rebuilding it per match gives. The whole point of
/// [`MoleculeMatchContext`] is that it is not pattern-dependent, and if it
/// ever became so this is where it would show.
#[test]
fn a_shared_analysis_matches_what_a_fresh_one_finds() {
    use crate::io::MoleculeMatchContext;

    let molecules = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "c1ccc2c(c1)ccc1ccccc12",
        "C1CCNCC1",
        "[nH]1cccc1",
        "CC(N)C(=O)O",
        "FC(F)(F)c1ccc(Cl)cc1",
        "O=S(=O)(N)c1ccccc1",
        "C1CC2CCC1CC2",
    ];
    let patterns = [
        "c1ccccc1",
        "[CX3](=O)[OX2H1]",
        "[NX3;H2]",
        "c1ccncc1",
        "[OX2H]",
        "C(=O)N",
        "[F,Cl,Br,I]",
        "[nH]",
        "C1CCCCC1",
        "[CX3]=[CX3]",
        "[R]",
        "[r6]",
    ];

    for smiles in molecules {
        let molecule = MolecularGraph::from_smiles_single(smiles).unwrap();
        let context = MoleculeMatchContext::new(&molecule);
        for smarts in patterns {
            let pattern = parse_cxsmarts(smarts).unwrap();
            assert_eq!(
                pattern.matches_with(&molecule, &context),
                pattern.matches(&molecule),
                "{smarts} against {smiles}"
            );
            assert_eq!(
                pattern.find_matches_with(&molecule, &context),
                pattern.find_matches(&molecule),
                "{smarts} against {smiles}"
            );
        }
    }
}

/// A recursive pattern asks about the atom it is written on.
///
/// `[$(...)]` was parsed and then ignored -- the evaluator had a catch-all that
/// said yes to every expression it did not implement -- so a recursive pattern
/// matched every atom in every molecule. That is worse than refusing it: it
/// makes any key or filter built on one always true, silently.
#[test]
fn a_recursive_pattern_is_a_question_about_one_atom() {
    assert_eq!(count("[$(CCO)]", "CCO"), 1);
    assert_eq!(count("[$(CCO)]", "CCC"), 0);
    // The first atom of the sub-pattern is the one being asked about, so this
    // is the carbon of an ethanol's methyl and not its oxygen.
    assert_eq!(count("[$(CCO)]", "CCOCC"), 2);
    assert_eq!(count("[$([NH2][CX4])]", "NCC"), 1);
    assert_eq!(count("[$([NH2][CX4])]", "NC=O"), 0);
    // Nested recursion works the same way.
    assert_eq!(count("[$([$(CC)])]", "CC"), 2);
}

/// `X` counts connections and `v` sums bond orders.
///
/// Both were implemented in a matcher nothing calls and missing from the one
/// everything calls, where the catch-all made them true of every atom --
/// including in the doc example `[CX3](=O)[OX2H1]`.
#[test]
fn connectivity_and_valence_are_read() {
    // Methane's carbon has four connections; ethene's has three.
    assert_eq!(count("[CX4]", "C=C"), 0);
    assert_eq!(count("[CX3]", "C=C"), 2);
    assert_eq!(count("[CX3](=O)[OX2H1]", "CC(=O)O"), 1);
    assert_eq!(count("[CX3](=O)[OX2H1]", "CC(=O)OC"), 0);
    assert_eq!(count("[Cv4]", "C=C"), 2);
    assert_eq!(count("[Cv3]", "C=C"), 0);
}

/// A bond that closes a ring, and one that does not.
///
/// `!@` was false for every bond of every molecule whose rings touch anything,
/// because the matcher asked whether both the bond's atoms were ring atoms
/// rather than whether the bond itself was a ring bond. Biphenyl's central bond
/// is in neither ring.
#[test]
fn a_ring_bond_is_the_bond_and_not_its_atoms() {
    assert_eq!(count("[!#1]!@[!#1]", "c1ccccc1-c1ccccc1"), 2);
    assert_eq!(count("[!#1]@[!#1]", "c1ccccc1-c1ccccc1"), 24);
    assert_eq!(count("*@*!@*@*", "c1ccccc1-c1ccccc1"), 8);
}

/// Two bond primitives written together are an and.
///
/// `=@` -- a double bond that closes a ring -- was read as a double bond with
/// the `@` dropped, which makes it any double bond at all.
#[test]
fn bond_primitives_written_together_are_both_required() {
    assert_eq!(count("[#6]=@[#6]", "C1=CCCCC1"), 2);
    assert_eq!(count("[#6]=@[#6]", "C=CCCCC"), 0);
    // And the low-precedence spelling means the same thing.
    assert_eq!(count("[#6]=;@[#6]", "C1=CCCCC1"), 2);
    assert_eq!(count("[#6]=;@[#6]", "C=CCCCC"), 0);
}

/// `r<n>` is the *smallest* ring the atom is in.
///
/// An indane's two bridgehead atoms are in a five- and a six-membered ring, and
/// are `r5` alone -- which is what RDKit says and what the keys that use `r`
/// were written against.
#[test]
fn a_ring_size_is_the_smallest_ring_the_atom_is_in() {
    assert_eq!(count("[r5]", "c1ccc2c(c1)CCC2"), 5);
    assert_eq!(count("[r6]", "c1ccc2c(c1)CCC2"), 4);
    assert_eq!(count("[R1]", "c1ccc2c(c1)CCC2"), 7);
    assert_eq!(count("[R2]", "c1ccc2c(c1)CCC2"), 2);
}

/// Two rings joined by a bond are two rings.
///
/// The cycle basis was sized by walking the ring atoms along *any* bond, so
/// biphenyl's two rings counted as one connected piece and one of them was
/// thrown away: none of its atoms was in a ring of any size afterwards.
#[test]
fn rings_joined_by_a_bond_are_still_two_rings() {
    assert_eq!(count("[r6]", "c1ccccc1-c1ccccc1"), 12);
    assert_eq!(count("[r6]", "C1CCCCC1C1CCCC1"), 6);
    assert_eq!(count("[r5]", "C1CCCCC1C1CCCC1"), 5);
    // And a fused pair is two rings, not the three cycles that can be drawn
    // through it.
    assert_eq!(count("[R1]", "c1ccc2ccccc2c1"), 8);
    assert_eq!(count("[R2]", "c1ccc2ccccc2c1"), 2);
}

/// `a` and `A` inside brackets are the aromatic and aliphatic wildcards.
///
/// They were read as one-letter element symbols, so `[a]` matched nothing at
/// all -- and `[a]` is how most published key sets say "aromatic".
#[test]
fn the_aromatic_and_aliphatic_wildcards_are_not_elements() {
    assert_eq!(count("[a]", "c1ccccc1"), 6);
    assert_eq!(count("[a]", "C1CCCCC1"), 0);
    assert_eq!(count("[A]", "c1ccccc1C"), 9); // the methyl and every hydrogen
    assert_eq!(count("[!c;!n;a]", "c1ccoc1"), 1);
    // A two-letter symbol beginning with the same letter is still an element.
    assert_eq!(count("[As]", "[AsH3]"), 1);
    assert_eq!(count("[Ag]", "[Ag+]"), 1);
}

/// A two-letter element symbol beats the primitive its first letter names.
///
/// `[Dy]` was a degree with no number after it, `[Rn]` was "in a ring and an
/// aromatic nitrogen", `[Hf]` was "one hydrogen and a fluorine". None of those
/// can match anything, and between them they cover half the lanthanides and
/// every noble gas.
#[test]
fn a_two_letter_element_is_read_as_an_element() {
    for symbol in ["Dy", "Rn", "Hf", "Ho", "Sc", "Xe", "Ne", "Rb", "Ar"] {
        let pattern = format!("[{symbol}]");
        assert!(
            parse_cxsmarts(&pattern).is_ok(),
            "{pattern} should parse as an element"
        );
        assert_eq!(count(&pattern, &format!("[{symbol}]")), 1, "{pattern}");
    }
    assert!(parse_cxsmarts("[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]").is_ok());
}

/// An isotope is read rather than ignored.
#[test]
fn an_isotope_is_read_and_an_atom_map_is_not_a_question() {
    assert_eq!(count("[13C]", "[13CH4]"), 1);
    assert_eq!(count("[13C]", "C"), 0);
    // An atom map number is a label rather than a question, so both of these
    // match: it is what tells a reaction where the atom goes.
    assert_eq!(count("[C:1]", "[CH4:1]"), 1);
    assert_eq!(count("[C:2]", "[CH4:1]"), 1);
}

/// `r{m-n}` — ring size in a range, either bound optional.
///
/// An RDKit SMARTS extension, not Daylight, supported because RDKit's
/// small-ring and macrocycle torsion rules are written with it and
/// cannot be read at all otherwise.
#[test]
fn ring_size_ranges_match_any_ring_in_range() {
    // benzene (6), cyclobutane (4), and a 12-membered macrocycle
    let parse = |smiles: &str| MolecularGraph::from_smiles_single(smiles).unwrap();
    let benzene = parse("c1ccccc1");
    let cyclobutane = parse("C1CCC1");
    let macrocycle = parse("C1CCCCCCCCCCC1");

    for (pattern, expected) in [
        ("[c;r{5-6}]", true),
        ("[c;r{5-8}]", true),
        ("[c;r{9-}]", false),
        ("[c;r{-4}]", false),
    ] {
        assert_eq!(
            matches_cxsmarts(&benzene, pattern).unwrap(),
            expected,
            "benzene vs {pattern}"
        );
    }
    assert!(matches_cxsmarts(&cyclobutane, "[C;r{-4}]").unwrap());
    assert!(!matches_cxsmarts(&cyclobutane, "[C;r{5-8}]").unwrap());
    // A twelve-membered ring answers to `r{9-}` and not to `r{5-8}`. It did
    // not always: ring perception here enumerated simple cycles up to eight
    // atoms, so a larger ring had no size at all and no range could match
    // it. That is why RDKit's macrocycle torsion rules, written entirely in
    // `r{9-}`, parsed here and matched nothing.
    assert!(!matches_cxsmarts(&macrocycle, "[C;r{5-8}]").unwrap());
    assert!(matches_cxsmarts(&macrocycle, "[C;r{9-}]").unwrap());
    assert!(matches_cxsmarts(&macrocycle, "[C;r12]").unwrap());

    // A bound is required, and the range has to be closed.
    assert!(parse_cxsmarts("[C;r{-}]").is_err());
    assert!(parse_cxsmarts("[C;r{5-6").is_err());
}

/// Bare `r` asks only whether the atom is in a ring, like bare `R`;
/// `r<n>` asks for a smallest ring of exactly that size. Requiring the
/// digit rejected ordinary SMARTS such as `[NX3&r]`.
#[test]
fn bare_ring_membership_needs_no_size() {
    let parse = |smiles: &str| MolecularGraph::from_smiles_single(smiles).unwrap();
    let pyrrolidine = parse("C1CCNC1");
    let trimethylamine = parse("CN(C)C");
    assert!(matches_cxsmarts(&pyrrolidine, "[NX3&r]").unwrap());
    assert!(!matches_cxsmarts(&trimethylamine, "[NX3&r]").unwrap());
    assert!(matches_cxsmarts(&trimethylamine, "[NX3!r]").unwrap());
}
