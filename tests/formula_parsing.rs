use oxidane::molecular_formula::MolecularFormula;

#[test]
fn malformed_formulas_return_errors_without_panicking() {
    for input in [
        "",
        "Qq",
        "CQqH2",
        "2H",
        "h2o",
        "H2O!",
        "Ca(OH)2",
        "CuSO4·5H2O",
        "C-1",
        "C0",
        "１２C",
        "💧",
    ] {
        assert!(input.parse::<MolecularFormula>().is_err(), "{input}");
    }
    let too_large = format!("C{}0", usize::MAX);
    assert!(too_large.parse::<MolecularFormula>().is_err());
    let overflowing_sum = format!("C{}C", usize::MAX);
    assert!(overflowing_sum.parse::<MolecularFormula>().is_err());
}

#[test]
fn flat_formulas_support_elements_counts_and_repeated_elements() {
    for (input, equivalent) in [
        ("H2O", "H2O1"),
        ("CH3COOH", "C2H4O2"),
        ("NaCl", "Cl1Na1"),
        ("C60", "C60"),
        ("Og", "Og1"),
    ] {
        assert_eq!(
            input.parse::<MolecularFormula>().unwrap(),
            equivalent.parse().unwrap()
        );
    }
}
