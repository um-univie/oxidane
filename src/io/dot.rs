
use std::io::Write;
use petgraph::dot::Dot;
use crate::molecule::Molecule;
use crate::io::ToSMILES;

pub fn to_dot_file<T: Molecule + ToSMILES>(mol: &T, file_name: Option<&str>) {
    let file_name = if let Some(file_name) = file_name {
        file_name.to_string()
    } else {
        let mol_name = mol.to_smiles();
        format!("{}.dot", mol_name)
    };
    let graph = mol.to_ungraph_with_edge_nodes_and_implicit_hydrogens();
    let mut output = std::fs::File::create(file_name).unwrap();
    let dot = Dot::new(&graph);
    write!(output, "{:?}", dot).unwrap();
}