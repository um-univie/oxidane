use oxidane::io::{CxSMARTSParser, parse_cxsmarts};

#[test]
fn unfinished_patterns_are_not_silently_truncated() {
    for query in [
        "C1CC", "C%12CC", "C12CCC1", "C(C", "(C.C", "C-", "C=", "C(C>>O", "C1CC>>O",
    ] {
        assert!(parse_cxsmarts(query).is_err(), "{query}");
    }
}

#[test]
fn closed_patterns_and_parser_reuse_still_work() {
    let mut parser = CxSMARTSParser::new();
    assert!(parser.parse("C1CC").is_err());
    for query in [
        "C1CC1",
        "C%12CC%12",
        "C1CC1C1CC1",
        "C1.C1",
        "C(C)C",
        "(C.C)",
        "C=C",
        "C(C)>>O",
    ] {
        assert!(parser.parse(query).is_ok(), "{query}");
    }
}
