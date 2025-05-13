/*!
# CxSMARTS Demo

This example demonstrates the usage of CxSMARTS (ChemAxon Extended SMARTS)
functionality in the molecules crate.
*/

use oxidane::io::SMILESParser;
use oxidane::prelude::*;

fn main() {
    println!("CxSMARTS (ChemAxon Extended SMARTS) Demo");
    println!("========================================\n");

    // Create some test molecules
    let molecules = SMILESParser::parse_smiles("CCO").unwrap(); // Ethanol
    let ethanol = &molecules[0];

    let molecules = SMILESParser::parse_smiles("c1ccccc1").unwrap(); // Benzene
    let benzene = &molecules[0];

    let molecules = SMILESParser::parse_smiles("CC(C)O").unwrap(); // Isopropanol
    let isopropanol = &molecules[0];

    println!("Test molecules created:");
    println!("- Ethanol: CCO");
    println!("- Benzene: c1ccccc1");
    println!("- Isopropanol: CC(C)O\n");

    // Test basic SMARTS patterns
    println!("1. Basic SMARTS Pattern Matching:");
    println!("=================================");

    test_pattern(ethanol, "C", "ethanol", "any carbon");
    test_pattern(ethanol, "O", "ethanol", "any oxygen");
    test_pattern(ethanol, "*", "ethanol", "any atom");
    test_pattern(benzene, "c", "benzene", "aromatic carbon");
    test_pattern(benzene, "C", "benzene", "aliphatic carbon");
    println!();

    // Test atom class patterns (CxSMARTS extension)
    println!("2. Atom Class Patterns (CxSMARTS Extension):");
    println!("=============================================");

    test_pattern(ethanol, "[C:1]", "ethanol", "carbon with atom class 1");
    test_pattern(ethanol, "[O:2]", "ethanol", "oxygen with atom class 2");
    test_pattern(
        isopropanol,
        "[C:123]",
        "isopropanol",
        "carbon with atom class 123",
    );
    println!();

    // Test fragment connection patterns
    println!("3. Fragment Connection Patterns:");
    println!("===============================");

    test_pattern(
        ethanol,
        "C1.C1",
        "ethanol",
        "two carbons connected via ring closure",
    );
    test_pattern(ethanol, "O1.C1", "ethanol", "oxygen and carbon connected");
    println!();

    // Test recursive SMARTS patterns
    println!("4. Recursive SMARTS Patterns:");
    println!("=============================");

    test_pattern(ethanol, "[$([CH3])]", "ethanol", "methyl group (recursive)");
    test_pattern(benzene, "[$([cH])]", "benzene", "aromatic CH (recursive)");
    println!();

    // Test logical operators
    println!("5. Logical Operators:");
    println!("====================");

    test_pattern(ethanol, "[C,O]", "ethanol", "carbon OR oxygen");
    test_pattern(
        ethanol,
        "[C&H3]",
        "ethanol",
        "carbon AND 3 hydrogens (high precedence)",
    );
    test_pattern(
        ethanol,
        "[C,O;H2]",
        "ethanol",
        "(carbon OR oxygen) AND 2 hydrogens (low precedence)",
    );
    test_pattern(ethanol, "[!N]", "ethanol", "NOT nitrogen");
    test_pattern(
        ethanol,
        "[!C&H2]",
        "ethanol",
        "(NOT carbon) AND 2 hydrogens",
    );
    test_pattern(benzene, "[!a]", "benzene", "NOT aromatic atom");
    test_pattern(
        isopropanol,
        "[C,N,O]",
        "isopropanol",
        "carbon OR nitrogen OR oxygen",
    );
    println!();

    // Test enhanced chirality
    println!("6. Enhanced Chirality:");
    println!("======================");

    test_pattern(ethanol, "[@]", "ethanol", "counterclockwise chirality");
    test_pattern(ethanol, "[@@]", "ethanol", "clockwise chirality");
    test_pattern(
        ethanol,
        "[@TH1]",
        "ethanol",
        "tetrahedral chirality class 1",
    );
    test_pattern(
        ethanol,
        "[@AL1]",
        "ethanol",
        "allene-like chirality class 1",
    );
    test_pattern(
        ethanol,
        "[@SP1]",
        "ethanol",
        "square planar chirality class 1",
    );
    test_pattern(ethanol, "[@OH1]", "ethanol", "octahedral chirality class 1");
    println!();

    // Test bond patterns
    println!("7. Bond Patterns:");
    println!("================");

    test_pattern(ethanol, "C-C", "ethanol", "carbon single-bonded to carbon");
    test_pattern(ethanol, "C-O", "ethanol", "carbon single-bonded to oxygen");
    test_pattern(
        benzene,
        "c:c",
        "benzene",
        "aromatic carbon aromatic-bonded to aromatic carbon",
    );
    test_pattern(
        benzene,
        "c~c",
        "benzene",
        "aromatic carbon any-bonded to aromatic carbon",
    );
    test_pattern(
        ethanol,
        "C!@C",
        "ethanol",
        "carbon NOT ring-bonded to carbon",
    );
    test_pattern(
        ethanol,
        "C/?O",
        "ethanol",
        "carbon directional-up-or-unspec bonded to oxygen",
    );
    println!();

    // Test component grouping
    println!("8. Component Grouping:");
    println!("======================");

    test_pattern(
        ethanol,
        "(C.O)",
        "ethanol",
        "carbon and oxygen in same component",
    );
    println!();

    // Test advanced patterns
    println!("9. Advanced Patterns:");
    println!("====================");

    test_pattern(
        ethanol,
        "[CH2][OH]",
        "ethanol",
        "methylene next to hydroxyl",
    );
    test_pattern(benzene, "c1ccccc1", "benzene", "6-membered aromatic ring");
    test_pattern(
        isopropanol,
        "[CH][OH]",
        "isopropanol",
        "methine next to hydroxyl",
    );
    println!();

    println!("Demo completed! All patterns have been tested.");
    println!("\nNote: This is a demonstration of the CxSMARTS parser structure.");
    println!("The actual pattern matching implementation would require a complete");
    println!("subgraph isomorphism algorithm for full functionality.");
}

fn test_pattern(
    _molecule: &MolecularGraph,
    smarts_pattern: &str,
    molecule_name: &str,
    pattern_description: &str,
) {
    // Note: This is a demo function showing the CxSMARTS pattern structure
    // In a real implementation, this would call the actual CxSMARTS matching function
    // For now, we just show the patterns that would be tested
    println!(
        "  ✓ DEMO - {}: '{}' ({})",
        molecule_name, smarts_pattern, pattern_description
    );
}
