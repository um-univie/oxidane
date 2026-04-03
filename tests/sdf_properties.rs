use oxidane::io::parse_sdf_file;
use std::io::Cursor;

fn record(fields: &str) -> String {
    let fixture = include_str!("pentene.sdf");
    let mol = fixture.split_once("M  END").unwrap().0;
    format!("{mol}M  END\n{fields}")
}

#[test]
fn multiline_properties_preserve_line_boundaries_and_spaces() {
    let input = record("> <FIRST>\n  one  \n\ttwo\nthree\n\n> <EMPTY>\n\n> <LAST>\nend\n\n$$$$\n");
    let records = parse_sdf_file(Cursor::new(input)).unwrap();
    let fields = &records[0].data_fields;
    assert_eq!(fields["FIRST"], "  one  \n\ttwo\nthree");
    assert_eq!(fields["EMPTY"], "");
    assert_eq!(fields["LAST"], "end");
}

#[test]
fn values_end_at_blank_lines_and_do_not_leak_between_records() {
    let first = record("> <FIRST>\none\n\nignored text\n$$$$\n");
    let second = record("ignored text\n> <SECOND>\ntwo\nthree");
    let records = parse_sdf_file(Cursor::new(format!("{first}{second}"))).unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].data_fields["FIRST"], "one");
    assert_eq!(records[1].data_fields.len(), 1);
    assert_eq!(records[1].data_fields["SECOND"], "two\nthree");
}

#[test]
fn pubchem_charge_rows_remain_separately_parseable() {
    let records = parse_sdf_file(Cursor::new(include_str!("pentene.sdf"))).unwrap();
    let rows: Vec<_> = records[0].data_fields["PUBCHEM_MMFF94_PARTIAL_CHARGES"]
        .lines()
        .collect();
    let count: usize = rows[0].parse().unwrap();
    assert_eq!(rows.len(), count + 1);
    for row in &rows[1..] {
        let columns: Vec<_> = row.split_whitespace().collect();
        assert_eq!(columns.len(), 2);
        columns[0].parse::<usize>().unwrap();
        columns[1].parse::<f64>().unwrap();
    }
}
