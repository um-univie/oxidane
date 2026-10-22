// Unit tests (Still needs to be improved)

use crate::prelude::*;
#[cfg(test)]
mod kekulize_tests {
    use crate::atom::Atom;
    use crate::molecule::molecular_system::extract_atom_pdb;
    use crate::prelude::*;
    use crate::vector::Vector3D;
    #[test]
    fn test_extract_atom() {
        let line =
            "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N  ";
        let atom = extract_atom_pdb(line).unwrap();
        let position = atom.position_vector.unwrap();
        assert_eq!(atom.atomic_number, 7);
        assert_eq!(position.x, 10.0);
        assert_eq!(position.y, 10.0);
        assert_eq!(position.z, 10.0);
    }
    #[test]
    fn test_vector_angle() {
        let v1 = Vector3D {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let v2 = Vector3D {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let angle = v1.angle_between(&v2);
        assert_eq!(angle, Some(std::f64::consts::FRAC_PI_2));
    }
    #[test]
    fn test_cross_product() {
        let v1 = Vector3D {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let v2 = Vector3D {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let v3 = v1.cross(&v2);
        assert_eq!(v3.x, 0.0);
        assert_eq!(v3.y, 0.0);
        assert_eq!(v3.z, 1.0);
    }

    #[test]
    fn test_bond_angle() {
        let atom1 = Atom::new(6).with_position((1.7, 0.0, 0.0));
        let atom2 = Atom::new(6).with_position((0.0, 0.0, 0.0));
        let atom3 = Atom::new(6).with_position((0.0, 0.0, 1.7));
        let molecule = Molecule3D::from_atoms(&vec![atom1, atom2, atom3]);
        let angles = molecule.find_angles();
        println!("{:?}", angles);
        for bonds in molecule.atom_bonds.iter() {
            println!("{:?}", bonds);
        }
        assert_eq!(angles.len(), 1);
        assert_eq!(angles[0].angle(), std::f64::consts::FRAC_PI_2);
    }

    #[test]
    fn test_alternate_covalent_radii() {
        use chemistry_consts::*;
        let atom1 = Atom::new(6).with_position((1.7, 0.0, 0.0));
        let atom2 = Atom::new(6).with_position((0.0, 0.0, 0.0));
        let atom3 = Atom::new(6).with_position((0.0, 0.0, 1.7));
        let molecule = Molecule3D::from_atoms_alternate_covalent_radii(
            &vec![atom1, atom2, atom3],
            &COVALENT_RADII,
        );
        let angles = molecule.find_angles();
        assert_eq!(angles.len(), 1);
        assert_eq!(angles[0].angle(), std::f64::consts::FRAC_PI_2);
    }

    #[test]
    fn test_kekulize_benzene() {
        // Test simple benzene ring
        let molecules = MolecularGraph::from_smiles("c1ccccc1").unwrap();
        let mut molecule = molecules.first().unwrap().clone();
        molecule.kekulize().unwrap();

        // Check alternating single and double bonds
        let bonds = molecule.get_edges_with_type();
        let double_bonds = bonds
            .iter()
            .filter(|(_, _, bond_type)| *bond_type == BondOrder::Double)
            .count();

        assert_eq!(double_bonds, 3, "Benzene should have 3 double bonds");
        assert_eq!(molecule.to_smiles(), "C1=CC=CC=C1");
    }

    #[test]
    fn test_kekulize_pyridine() {
        // Test pyridine ring
        let molecules = MolecularGraph::from_smiles("n1ccccc1").unwrap();
        let mut molecule = molecules.first().unwrap().clone();
        molecule.kekulize().unwrap();

        assert_eq!(molecule.to_smiles(), "N1=CC=CC=C1");
    }

    #[test]
    fn test_kekulize_naphthalene() {
        // Test fused rings (naphthalene)
        let molecules = MolecularGraph::from_smiles("c1ccc2ccccc2c1").unwrap();
        let mut molecule = molecules.first().unwrap().clone();
        molecule.kekulize().unwrap();

        println!("{}", molecule.to_smiles());
        let bonds = molecule.get_edges_with_type();
        let double_bonds = bonds
            .iter()
            .filter(|(_, _, bond_type)| *bond_type == BondOrder::Double)
            .count();

        assert_eq!(double_bonds, 5, "Naphthalene should have 5 double bonds");
    }

    #[test]
    fn test_kekulize_non_aromatic() {
        // Test molecule with no aromatic bonds
        let molecules = MolecularGraph::from_smiles("CC").unwrap();
        let mut molecule = molecules.first().unwrap().clone();
        molecule.kekulize().unwrap();

        assert_eq!(molecule.to_smiles(), "CC");
    }

    #[test]
    fn test_kekulize_furan() {
        // Test heterocyclic aromatic ring
        let molecules = MolecularGraph::from_smiles("o1cccc1").unwrap();
        let mut molecule = molecules.first().unwrap().clone();
        molecule.kekulize().unwrap();

        assert_eq!(molecule.to_smiles(), "O1C=CC=C1");
    }

    #[test]
    fn test_kekulize_phenol() {
        // Test aromatic ring with substituent
        let molecules = MolecularGraph::from_smiles("Oc1ccccc1").unwrap();
        let mut molecule = molecules.first().unwrap().clone();
        molecule.kekulize().unwrap();

        assert_eq!(molecule.to_smiles(), "OC1=CC=CC=C1");
    }

    #[test]
    fn test_kekulize_multiple_rings() {
        // Test molecule with multiple aromatic rings
        let molecules = MolecularGraph::from_smiles("c1ccccc1Cc2ccccc2").unwrap();
        let mut molecule = molecules.first().unwrap().clone();
        molecule.kekulize().unwrap();

        let bonds = molecule.get_edges_with_type();
        let double_bonds = bonds
            .iter()
            .filter(|(_, _, bond_type)| *bond_type == BondOrder::Double)
            .count();

        assert_eq!(
            double_bonds, 6,
            "Two benzene rings should have 6 double bonds total"
        );
    }
}
#[cfg(test)]
mod aromatization_tests {
    use crate::prelude::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    #[test]
    fn test_benzene_aromatization() {
        let molecule = MolecularGraph::from_smiles("C1=CC=CC=C1").unwrap();
        let mut molecule = molecule.first().unwrap().clone();

        molecule.simple_aromatization();

        for (i, bonds) in molecule.atom_bonds().iter().enumerate() {
            for bond in bonds {
                if molecule.atomic_numbers()[bond.target()] != 1
                    && molecule.atomic_numbers()[i] != 1
                {
                    assert_eq!(bond.bond_order(), BondOrder::Aromatic);
                }
            }
        }

        assert_eq!(molecule.to_smiles(), "c1ccccc1");
    }

    #[test]
    fn test_difficult_smiles() {
        let smiles = "BrC1=CC=C2[Te]C=CC2=C1";
        let molecule = diff_smiles(smiles);
        assert_eq!(molecule, "Brc1ccc2[te]ccc2c1");

        let smiles = "C1(C)C2=C(c3c(cccc3)O1)O2";
        assert!(autoencode(smiles));

        let smiles = "c1c2c(ccc1)OCC2C(=O)N(C)c3c(N)cncc3";
        let molecule = diff_smiles(smiles);
        assert_eq!(molecule, smiles);

        let smiles = "CCCCC1CCC(C(Br)C2=CC3=C(C=C2)CCC3)CC1";
        let molecule = diff_smiles(smiles);
        assert_eq!(molecule, "CCCCC1CCC(C(Br)c2cc3c(cc2)CCC3)CC1");

        let smiles = "c1c(C(=[Zr+2])c2ccc(C)cc2)ccc(C)c1";
        assert!(autoencode(smiles));

        let smiles = "C(C)(C)C1=CC(C(Cl)Cl)C2=CC=CC=C2O1";
        let molecule = diff_smiles(smiles);
        assert_eq!(molecule, "C(C)(C)C1=CC(C(Cl)Cl)c2ccccc2O1");

        // looses single bond
        let smiles = "c1c(OC(C2CN(CC3CC3)C(=O)C2)=O)c(Cl)cc(-c4ccccc4)c1";
        assert!(autoencode(smiles));

        let smiles = "c1c=2c(=Cc3sc(CCCC)cc3)c(=O)[nH]c=2[nH]c(=C)c1-c4ccncc4";
        assert!(autoencode(smiles));

        let smiles = "C=c1[nH]c2=c(cc1-c3ccncc3)c(=Cc4ccc(CCCC)s4)c(=O)[nH]2";
        assert!(autoencode(smiles));

        let smiles = "C1COc2cccc(c2)CNCCN2CCNCc3cccc(c3)OCCN(CCOc3cccc(c3)CNCCN3CCNCc4cccc(c4)OCCN1CCOc1cc(ccc1)CNCC3)CCOc1cc(ccc1)CNCC2";
        let molecule = diff_smiles(smiles);
        assert_eq!(
            molecule,
            "C1COc2cccc(c2)CNCCN3CCNCc4cccc(c4)OCCN(CCOc5cccc(c5)CNCCN6CCNCc7cccc(c7)OCCN1CCOc8cc(ccc8)CNCC6)CCOc9cc(ccc9)CNCC3"
        );

        let smiles = "c12c(cc(-c3c4ncn(C5CCNCC5)c4ncc3-c6cc(Cl)ccc6)cc1)cccc2";
        let molecule = diff_smiles(smiles);
        assert_eq!(
            molecule,
            "c12c(cc(-c3c4ncn(C5CCNCC5)c4ncc3-c6cc(Cl)ccc6)cc1)cccc2"
        );

        let smiles = "O=C1C=CC=C2C=CC3=CC=CNC3=C12";
        let molecule = diff_smiles(smiles);
        assert_eq!(molecule, "O=c1cccc2ccc3ccc[nH]c3c12");

        let smiles = "C12=C(C=C(C(NNC(C3CCN(CC3)c4ccc(nn4)Cl)=O)=O)S2)CCCC1";
        let molecule = diff_smiles(smiles);
        assert_eq!(
            molecule,
            "c12c(cc(C(NNC(C3CCN(CC3)c4ccc(nn4)Cl)=O)=O)s1)CCCC2"
        );

        let smiles = "c12c3cccc1C(C(c2ccc3)=O)=NNC(C(C)Nc4nc5c(cccc5)s4)=O";
        let molecule = diff_smiles(smiles);
        assert_eq!(
            molecule,
            "c12c3cccc1C(C(c2ccc3)=O)=NNC(C(C)Nc4nc5c(cccc5)s4)=O"
        );

        let smiles = "c12c(C=CC=C1)oc3c2N(CC(Nc4cc(ccc4)C)=O)C(N(c5ccc(cc5)OCC)C3=O)=O";
        let molecule = diff_smiles(smiles);
        assert_eq!(
            molecule,
            "c12c(cccc1)oc3c2n(CC(Nc4cc(ccc4)C)=O)c(n(-c5ccc(cc5)OCC)c3=O)=O"
        );

        let smiles =
            "COCCOC(=O)C1=C(C)N=C2SC(=CC3=CC=C(C4=CC(Cl)=CC=C4)O3)C(=O)N2C1C1=CC=C(C(C)C)C=C1";
        let molecule = diff_smiles(smiles);
        // The thiazolone is not aromatic, and used to be perceived as such. The
        // ring carries five π electrons, not six: the imine carbon and the
        // carbonyl carbon give nothing, the sulfur and the amide nitrogen give
        // two each, and the arylidene carbon gives one.
        //
        // The imine is what changed. Its `C=N` leaves the five-ring but also
        // lies on the fused six-ring, and the exocyclic test used to accept any
        // cycle at all, so the carbon was counted as if the bond stayed in --
        // six electrons, and an aromatic ring. That six-ring runs through a
        // saturated carbon and carries no π system for the bond to belong to;
        // see `molecule::pi_system`. RDKit calls this ring aromatic, by the
        // same route, and is wrong.
        assert_eq!(
            molecule,
            "COCCOC(=O)C1=C(C)N=C2SC(=Cc3ccc(-c4cc(Cl)ccc4)o3)C(=O)N2C1c5ccc(C(C)C)cc5"
        );

        let smiles = "[B-]1(=CC=CC=C1)C";
        let molecule = diff_smiles(smiles);
        assert_eq!(molecule, "[b-]1(ccccc1)C");

        let smiles = "CCN1C(=O)C2=C(C=C(NC3=NC(NC4=CC=CC=C4S(=O)(=O)O)=NC(NC4=CC=CC=C4S(=O)(=O)O)=N3)C=C2)C1=O";
        let molecule = diff_smiles(smiles);
        assert_eq!(
            molecule,
            "CCN1C(=O)c2c(cc(Nc3nc(Nc4ccccc4S(=O)(=O)O)nc(Nc5ccccc5S(=O)(=O)O)n3)cc2)C1=O"
        );

        let smiles = "c1cc(C#CC2=c3nc(c4c5cccc6cccc(c3=4)c65)C(C#Cc3ccccc3)=c3n4c(c5c3-c3c6c-5cccc6ccc3)=C(C#Cc3ccccc3)c3nc(c5c6cccc7c6c(ccc7)c3=5)=C(C#Cc3ccccc3)c3c5c(c2n3[Pb]4)-c2c3c(cccc3-5)ccc2)ccc1";
        let molecule = diff_smiles(smiles);
        assert_eq!(
            molecule,
            "c1cc(C#Cc2c3nc(c4c5cccc6cccc(c34)c56)c(C#Cc7ccccc7)c8n9c(c%10c8c%11c%12c%10cccc%12ccc%11)c(C#Cc%13ccccc%13)c%14nc(c%15c%16cccc%17c%16c(ccc%17)c%14%15)c(C#Cc%18ccccc%18)c%19c%20c(c2n%19[Pb]9)c%21c%22c(cccc%20%22)ccc%21)ccc1"
        );

        let smiles = "[se]1c(C(C[n+]2c(C)csc2)=O)ccc1";
        let molecule = diff_smiles(smiles);
        assert_eq!(molecule, "[se]1c(C(C[n+]2c(C)csc2)=O)ccc1");

        let smiles = "O=C(C1=CC=CC=N1)C1C23C4=C5C6=C7C8=C9C%10=C6C12C1=C%10C2=C6C%10=C1C1=C3C3=C4C4=C%11C5=C7C5=C7C8=C8C9=C2C2=C9C8=C8C7=C7C5=C%11C5=C%11C%12=C%13C%14=C(C(=C62)C2=C%14C6=C%12C(=C45)C3=C6C1=C%102)C9=C8C%13=C%117";
        let molecule = diff_smiles(smiles);
        assert_eq!(
            molecule,
            "O=C(c1ccccn1)C2C34c5c6c7c8c9c%10c%11c7C23c%12c%11c%13c%14c%15c%12c%16c4c%17c5c%18c%19c6c8c%20c%21c9c%22c%10c%13c%23c%24c%22c%25c%21c%26c%20c%19c%27c%28c%29c%30c%31c(c(c%14%23)c%32c%31c%33c%29c(c%18%27)c%17c%33c%16c%15%32)c%24c%25c%30c%26%28"
        );

        let smiles = "O=C(C1=CC=CC=N1)C1C23C4=C5C6=C7C8=C9C%10=C6C12C1=C%10C2=C6C%10=C1C1=C3C3=C4C4=C%11C5=C7C5=C7C8=C8C9=C2C2=C9C8=C8C7=C7C5=C%11C5=C%11C%12=C%13C%14=C(C(=C62)C2=C%14C6=C%12C(=C45)C3=C6C1=C%102)C9=C8C%13=C%117";
        let molecule = diff_smiles(smiles);
        assert_eq!(
            molecule,
            "O=C(c1ccccn1)C2C34c5c6c7c8c9c%10c%11c7C23c%12c%11c%13c%14c%15c%12c%16c4c%17c5c%18c%19c6c8c%20c%21c9c%22c%10c%13c%23c%24c%22c%25c%21c%26c%20c%19c%27c%28c%29c%30c%31c(c(c%14%23)c%32c%31c%33c%29c(c%18%27)c%17c%33c%16c%15%32)c%24c%25c%30c%26%28"
        );

        let molecule = diff_smiles(
            "C12C(O)C(O)C(OC1CO)OC1C(O)C(O)C(OC3C(O)C(O)C(OC4C(CO)OC(OC5C(CO)OC(OC6C(O)C(O)C(OC7C(COc8cc9c%10[n-]c(nc%11nc(nc%12[n-]c(nc%13nc(n%10)-c%10c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c%10-%13)c%10c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c%10%12)-c%10c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c%10-%11)c9cc8)OC(O2)C(O)C7O)OC6CO)C(O)C5O)C(O)C4O)OC3CO)OC1CO.[Zn+2]",
        );
        assert_eq!(
            molecule,
            "C12C(O)C(O)C(OC1CO)OC3C(O)C(O)C(OC4C(O)C(O)C(OC5C(CO)OC(OC6C(CO)OC(OC7C(O)C(O)C(OC8C(COc9cc%10c%11[n-]c(nc-%12nc(nc%13[n-]c(nc-%14nc(n%11)-c%15c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c-%14%15)c%16c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c%13%16)-c%17c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c(OCC(F)(F)F)c-%12%17)c%10cc9)OC(O2)C(O)C8O)OC7CO)C(O)C6O)C(O)C5O)OC4CO)OC3CO"
        );

        let smiles = "Cc1c(=O)[nH]c(=O)n(C2CC(O)C(COP(O)(=S)OC3CC(n4cnc5c(=O)[nH]c(N)nc54)OC3COP(OC3CC(n4cc(C)c(=O)[nH]c4=O)OC3COP(OC3C(COP(O)(=S)OC4C(COP(=S)(OC5C(COP(O)(=S)OC6C(COP(=S)(OC7CC(n8ccc(N)nc8=O)OC7COP(OC7C(COP(=S)(O)OC8CC(n9cc(C)c(=O)[nH]c9=O)OC8COP(O)(=S)OC8CC(n9cnc%10c(N)ncnc%109)OC8COP(O)(=S)OC8C(COP(O)(OC9CC(n%10c%11nc(N)[nH]c(=O)c%11nc%10)OC9COP(=S)(OC9CC(n%10cnc%11c%10nc(N)[nH]c%11=O)OC9COP(S)(=O)OC9C(COP(O)(OC%10CC(n%11cnc%12c(=O)[nH]c(N)nc%12%11)OC%10COP(OC%10CC(n%11c(=O)nc(N)cc%11)OC%10COP(=S)(OC%10CC(n%11c%12c(c(=O)[nH]c(N)n%12)nc%11)OC%10COP(O)(=S)OC%10C(COP(=S)(O)OC%11C(COP(=S)(OC%12C(CO)OC(n%13cnc%14c(=O)[nH]c(N)nc%14%13)C%12)O)OC(n%12c%13ncnc(N)c%13nc%12)C%11)OC(n%11c(=O)nc(N)cc%11)C%10)O)(=S)O)=S)OC(n%10c(=O)[nH]c(=O)c(C)c%10)C9)O)=S)OC(n9ccc(N)nc9=O)C8)OC(n8c9c(c(=O)[nH]c(N)n9)nc8)C7)(O)=S)O)OC(n7cc(C)c(=O)[nH]c7=O)C6)OC(n6c(=O)[nH]c(=O)c(C)c6)C5)O)OC(n5cnc6c(=O)[nH]c(N)nc56)C4)OC(n4cnc5c(=O)[nH]c(N)nc54)C3)(O)=S)(=S)O)O2)c1";
        let molecule = diff_smiles(smiles);
        assert_eq!(
            molecule,
            "Cc1c(=O)[nH]c(=O)n(C2CC(O)C(COP(O)(=S)OC3CC(n4cnc5c(=O)[nH]c(N)nc45)OC3COP(OC6CC(n7cc(C)c(=O)[nH]c7=O)OC6COP(OC8C(COP(O)(=S)OC9C(COP(=S)(OC%10C(COP(O)(=S)OC%11C(COP(=S)(OC%12CC(n%13ccc(N)nc%13=O)OC%12COP(OC%14C(COP(=S)(O)OC%15CC(n%16cc(C)c(=O)[nH]c%16=O)OC%15COP(O)(=S)OC%17CC(n%18cnc%19c(N)ncnc%18%19)OC%17COP(O)(=S)OC%20C(COP(O)(OC%21CC(n%22c%23nc(N)[nH]c(=O)c%23nc%22)OC%21COP(=S)(OC%24CC(n%25cnc%26c%25nc(N)[nH]c%26=O)OC%24COP(S)(=O)OC%27C(COP(O)(OC%28CC(n%29cnc%30c(=O)[nH]c(N)nc%29%30)OC%28COP(OC%31CC(n%32c(=O)nc(N)cc%32)OC%31COP(=S)(OC%33CC(n%34c%35c(c(=O)[nH]c(N)n%35)nc%34)OC%33COP(O)(=S)OC%36C(COP(=S)(O)OC%37C(COP(=S)(OC%38C(CO)OC(n%39cnc%40c(=O)[nH]c(N)nc%39%40)C%38)O)OC(n%41c%42ncnc(N)c%42nc%41)C%37)OC(n%43c(=O)nc(N)cc%43)C%36)O)(=S)O)=S)OC(n%44c(=O)[nH]c(=O)c(C)c%44)C%27)O)=S)OC(n%45ccc(N)nc%45=O)C%20)OC(n%46c%47c(c(=O)[nH]c(N)n%47)nc%46)C%14)(O)=S)O)OC(n%48cc(C)c(=O)[nH]c%48=O)C%11)OC(n%49c(=O)[nH]c(=O)c(C)c%49)C%10)O)OC(n%50cnc%51c(=O)[nH]c(N)nc%50%51)C9)OC(n%52cnc%53c(=O)[nH]c(N)nc%52%53)C8)(O)=S)(=S)O)O2)c1"
        );
    }

    fn diff_smiles(smiles: &str) -> String {
        let molecule = parse_smiles_and_aromatize(smiles);
        molecule
    }

    fn autoencode(smiles: &str) -> bool {
        let molecule = parse_smiles_and_aromatize(smiles);
        println!("{}", smiles);
        println!("{}", molecule);
        smiles == molecule
    }

    #[test]
    fn test_pyridine_aromatization() {
        let molecule = MolecularGraph::from_smiles("C1=CC=CN=C1").unwrap();
        let mut molecule = molecule.first().unwrap().clone();
        molecule.aromatize_opensmiles().unwrap();

        assert_eq!(molecule.to_smiles(), "c1cccnc1");
    }

    #[test]
    fn test_naphthalene_aromatization() {
        let molecule = MolecularGraph::from_smiles("C1=CC=C2C=CC=CC2=C1").unwrap();
        let mut molecule = molecule.first().unwrap().clone();

        molecule.aromatize_opensmiles().unwrap();

        assert_eq!(molecule.to_smiles(), "c1ccc2ccccc2c1");
    }

    #[test]
    fn test_cyclopentadienyl_anion() {
        let molecule = MolecularGraph::from_smiles("C1=CC=C[CH-]1").unwrap();
        let mut molecule = molecule.first().unwrap().clone();

        molecule.aromatize_opensmiles().unwrap();

        assert_eq!(molecule.to_smiles(), "c1ccc[cH-]1");
    }

    #[test]
    fn test_non_aromatic_cycle() {
        // Cyclooctatetraene is not aromatic (8 electrons, not 4n+2)
        let molecule = MolecularGraph::from_smiles("C1=CC=CC=CC=C1").unwrap();
        let mut molecule = molecule.first().unwrap().clone();

        molecule.aromatize_opensmiles().unwrap();

        assert_eq!(molecule.to_smiles(), "C1=CC=CC=CC=C1");
    }

    #[test]
    fn test_furan_aromatization() {
        let molecule = MolecularGraph::from_smiles("C1=COC=C1").unwrap();
        let mut molecule1 = molecule.first().unwrap().clone();
        molecule1.aromatize_opensmiles().unwrap();

        assert_eq!(molecule1.to_smiles(), "c1cocc1");
    }

    #[test]
    fn test_non_cyclic_molecule() {
        // Test with ethene
        let molecule = MolecularGraph::from_smiles("C=C").unwrap();
        let mut molecule = molecule.first().unwrap().clone();

        molecule.aromatize_opensmiles().unwrap();

        // Check that the double bond remains unchanged
        assert_eq!(molecule.atom_bonds()[0][0].bond_order(), BondOrder::Double);
        assert_eq!(molecule.to_smiles(), "C=C");
    }

    #[test]
    fn test_aromatization_from_file() {
        let file = BufReader::new(File::open("tests/aromatization_tests.smi").unwrap());

        for line in file.lines() {
            let line = line.unwrap();

            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse the line: "kekulized, aromatized # comment"
            let parts: Vec<&str> = line
                .split('#')
                .next()
                .unwrap()
                .split(',')
                .map(|s| s.trim())
                .collect();

            if parts.len() != 2 {
                continue;
            }

            let (kekulized, expected_aromatic) = (parts[0], parts[1]);

            let molecule = MolecularGraph::from_smiles(kekulized).unwrap();
            let mut molecule = molecule.first().unwrap().clone();

            // Try to aromatize
            molecule.simple_aromatization();

            assert_eq!(
                molecule.to_smiles(),
                expected_aromatic,
                "Failed to correctly aromatize {}: expected {}, got {}",
                kekulized,
                expected_aromatic,
                molecule.to_smiles()
            );
        }
    }

    fn parse_smiles_and_aromatize(smiles: &str) -> String {
        let mut molecules = MolecularGraph::from_smiles(smiles).unwrap();
        let mut molecule = molecules.swap_remove(0);
        println!("Done reading {}", molecule.to_smiles());
        // println!("Bonds: {:?}", molecule.get_atom_bonds(20));
        let _ = molecule.aromatize_opensmiles();
        molecule.to_smiles()
    }
}

#[cfg(test)]
mod graph_comparison_tests {
    use crate::prelude::*;

    #[test]
    fn enumerate_possible_cycles() {
        let molecule = MolecularGraph::from_smiles("c1ccc2ccccc2c1").unwrap();
        let molecule = molecule.first().unwrap();
        let cycles = molecule.enumerate_possible_cycles();
        println!("{:?}", cycles);
        assert_eq!(cycles.len(), 3);
    }
}

#[cfg(test)]
mod new_aromatization {
    use super::*;
    

    #[test]
    fn test_furan_aromatization() {
        let molecule = MolecularGraph::from_smiles("C1=COC=C1").unwrap();
        let mut molecule1 = molecule.first().unwrap().clone();
        molecule1.aromatize_opensmiles().unwrap();

        assert_eq!(molecule1.to_smiles(), "c1cocc1");
    }

    #[test]
    fn test_naphthalene_aromatization() {
        let molecule = MolecularGraph::from_smiles("C1=CC=C2C=CC=CC2=C1").unwrap();
        let mut molecule = molecule.first().unwrap().clone();

        molecule.aromatize_opensmiles().unwrap();

        assert_eq!(molecule.to_smiles(), "c1ccc2ccccc2c1");
    }

    #[test]
    fn test_cyclopentadienyl_anion() {
        let molecule = MolecularGraph::from_smiles("C1=CC=C[CH-]1").unwrap();
        let mut molecule = molecule.first().unwrap().clone();

        molecule.aromatize_opensmiles().unwrap();

        assert_eq!(molecule.to_smiles(), "c1ccc[cH-]1");
    }

    #[test]
    fn test_non_aromatic_cycle() {
        // Cyclooctatetraene is not aromatic (8 electrons, not 4n+2)
        let mut molecule = MolecularGraph::from_smiles_single("C1=CC=CC=CC=C1").unwrap();

        molecule.aromatize_opensmiles().unwrap();
    }

    #[test]
    /// Borepin is written `[BH]`, not `[B]`: brackets state the hydrogen
    /// count, so `[B]` is a boron with an unpaired electron and no hydrogen —
    /// which RDKit reads the same way, and which is not aromatic.
    fn test_big_systems() {
        let mut molecule = MolecularGraph::from_smiles_single("C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12C5=C%11C4=C3C3=C5C(=C81)C%10=C23").unwrap();

        molecule.aromatize_opensmiles().unwrap();
        for atom_index in 0..molecule.len() {
            if molecule.get_atomic_number(atom_index) == 6 {
                assert!(molecule.is_atom_aromatic(atom_index));
            }
        }

        let mut molecule = MolecularGraph::from_smiles_single("[BH]1C=CC=CC=C1").unwrap();

        molecule.aromatize_opensmiles().unwrap();
        for atom_index in 0..molecule.len() {
            if molecule.get_atomic_number(atom_index) == 6 {
                assert!(molecule.is_atom_aromatic(atom_index));
            }
        }

        let mut molecule = MolecularGraph::from_smiles_single("[BH]1C=CC=CC=C1").unwrap();

        molecule.aromatize_opensmiles().unwrap();
        for atom_index in 0..molecule.len() {
            if molecule.get_atomic_number(atom_index) == 6 {
                assert!(molecule.is_atom_aromatic(atom_index));
            }
        }

        let mut molecule = MolecularGraph::from_smiles_single(
            "C1=CC2=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C1N2",
        )
        .unwrap();
        molecule.aromatize_opensmiles().unwrap();
        println!("{}", molecule.to_smiles());
        let mut number_of_aromatic_atoms = 0;
        for atom_index in 0..molecule.len() {
            if molecule.get_atomic_number(atom_index) == 6 {
                number_of_aromatic_atoms += 1;
            }
        }
        assert_eq!(number_of_aromatic_atoms, 20); // Five rings should be aromatic

        // [22]annulene
        let mut molecule =
            MolecularGraph::from_smiles_single("C1=CC=CC=CC=CC=CC=CC=CC=CC=CC=CC=C1").unwrap();
        molecule.aromatize_opensmiles().unwrap();
        let _number_of_aromatic_atoms = 0;
        for atom_index in 0..molecule.len() {
            if molecule.get_atomic_number(atom_index) == 6 {
                assert!(molecule.is_atom_aromatic(atom_index));
            }
        }

        // Graphene
        let mut molecule = MolecularGraph::from_smiles_single("C1=CC2=CC3=CC4=C5C6=C3C7=C2C(=C1)C8=C9C7=C1C6=C2C3=C6C1=C1C9=C(C=CC1=CC6=CC1=C3C3=C6C(=C1)C=C1C=CC=C7C1=C6C1=C6C3=C2C5=C2C6=C3C(=CC2=C4)C=CC2=C3C1=C7C=C2)C=C8").unwrap();
        molecule.aromatize_opensmiles().unwrap();
        println!("{}", molecule.to_smiles());
        let _number_of_aromatic_atoms = 0;
        for atom_index in 0..molecule.len() {
            if molecule.get_atomic_number(atom_index) == 6 {
                assert!(molecule.is_atom_aromatic(atom_index));
            }
        }
    }
    #[test]
    fn test_porphyrin() {
        let mut molecule = MolecularGraph::from_smiles_single(
            "C1=CC2=CC3=CC=C(N3)C=C4C=CC(=N4)C=C5C=CC(=N5)C=C1N2",
        )
        .unwrap();
        molecule.aromatize_opensmiles().unwrap();
        println!("{}", molecule.to_smiles());

        let mut molecule = MolecularGraph::from_smiles_single(
            "C1=CC2=CC5=CC=C(C=C4C=CC(C=C3C=CC(=CC1=N2)N3)=N4)N5",
        )
        .unwrap();
        molecule.aromatize_opensmiles().unwrap();
        println!("{}", molecule.to_smiles());
    }
}

#[cfg(test)]
mod chirality_tests {
    use super::*;
    use crate::graph_algorithms::mol_canonization::{self, reorder_molecule};
    
    use crate::molecule::chirality::ChiralClass;
    use crate::molecule::molecular_graph::MolecularGraph;

    #[test]
    fn chiral_smiles() {
        let molecule = MolecularGraph::from_smiles("C[C@H](C)C").unwrap();
        let mut molecule = molecule.first().unwrap().clone();

        println!("{:?}", molecule.chiral_classes());
        molecule.aromatize_opensmiles().unwrap();
        assert_eq!(molecule.to_smiles(), "C[C@H](C)C");

        let molecule = MolecularGraph::from_smiles("N1[C@H](Cl)[C@@H](Cl)C(Cl)CC1").unwrap();
        let mut molecule = molecule.first().unwrap().clone();
        molecule.aromatize_opensmiles().unwrap();
        assert_eq!(molecule.to_smiles(), "N1[C@H](Cl)[C@@H](Cl)C(Cl)CC1");

        let molecule = MolecularGraph::from_smiles_single("C/C=C/C").unwrap();
        // let canonized = mol_canonization(&molecule);
        assert_eq!(molecule.to_smiles(), "C/C=C/C");
        // assert_eq!(canonized.to_smiles(), "C[C@@H]=[C@H]C");

        let mut molecule = MolecularGraph::from_smiles_single("CC/C=C/C").unwrap();
        molecule.aromatize_opensmiles().unwrap();
        // let canonized = mol_canonization(&molecule);
        assert_eq!(molecule.to_smiles(), "CC/C=C/C");
        // assert_eq!(canonized.to_smiles(), "C[C@@H]=[C@H]CC");

        let molecule = MolecularGraph::from_smiles("CC/C=C/CC/C=C\\C").unwrap();
        let mut molecule = molecule.first().unwrap().clone();
        molecule.aromatize_opensmiles().unwrap();
        // let canonized = mol_canonization(&molecule);
        assert_eq!(molecule.to_smiles(), "CC/C=C/CC/C=C\\C");
        // assert_eq!(canonized.to_smiles(), "CC[C@@H]=[C@H]CC[C@@H]=[C@@H]C");

        let molecule = MolecularGraph::from_smiles("CC(\\C)=C/CC/C=C\\C").unwrap();
        let mut molecule = molecule.first().unwrap().clone();
        molecule.aromatize_opensmiles().unwrap();
        // let canonized = mol_canonization(&molecule);
        assert_eq!(molecule.to_smiles(), "C/C(C)=C\\CC/C=C\\C");
        // TODO: This fail as we need to switch up to down if the order changes
        // assert_eq!(canonized.to_smiles(), "C[C@H]=[C@H]CC[C@@H]=[C@](C)C");
    }

    #[test]
    fn test_tetrahedral_chirality_update() {
        // Test molecule: chiral carbon (e.g., CHFClBr)
        let mut mol = MolecularGraph::from_smiles_single("[C@H](F)(Cl)Br").unwrap();

        println!("Atomic numbers: {:?}", mol.atomic_numbers());
        println!("Chiral classes: {:?}", mol.chiral_classes());
        // Test permutation that should invert chirality
        let labeling = vec![1, 0, 2, 3, 4];
        reorder_molecule(&mut mol, &labeling);
        let chiral_classes = mol.chiral_classes().unwrap().to_vec();
        assert_eq!(chiral_classes[1].clockness(), ChiralClass::Clockwise); // Should invert from 1 to 2

        // Test permutation that should preserve chirality, this is due to the fact that the smiles
        // path does not change
        let labeling = vec![0, 2, 1, 3, 4];
        reorder_molecule(&mut mol, &labeling);
        let chiral_classes = mol.chiral_classes().unwrap().to_vec();
        assert_eq!(chiral_classes[2].clockness(), ChiralClass::Clockwise); // Should not invert
    }

    /// Swapping the indices of two of an allene's carriers has to flip its
    /// tag, since the tag is read against those carriers sorted by index and
    /// the molecule itself has not changed.
    ///
    /// Here `N` and `Br`, both carriers on the same end, trade places, so
    /// `@` (anticlockwise) must become `@@`. This used to assert that nothing
    /// changed, which held only while the tag was maintained against the
    /// centre's own two neighbours — an ordering that cannot express which way
    /// four substituents wind.
    #[test]
    fn allene_chirality_update() {
        let mut mol = MolecularGraph::from_smiles_single("NC(Br)=[C@]=C(O)C").unwrap();
        assert_eq!(mol.chiral_classes().unwrap()[3].clockness(), ChiralClass::Counterclockwise);

        let labeling = vec![2, 1, 0, 3, 4, 5, 6];
        reorder_molecule(&mut mol, &labeling);

        let chiral_classes = mol.chiral_classes().unwrap().to_vec();
        assert_eq!(chiral_classes[3].clockness(), ChiralClass::Clockwise);
    }

    #[test]
    fn cis_trans_chirality_update() {
        // Test molecule: cis/trans alkene (e.g., CHF=CHF)
        let mut mol = MolecularGraph::from_smiles_single("F/C=C/F").unwrap();
        let chiral_classes = mol.chiral_classes().unwrap().to_vec();
        println!("Chiral classes: {:?}", chiral_classes);
        println!("Molecule: {:?}", mol.to_smiles());

        // Reordering the atoms must preserve the (E) stereochemistry; the emitted
        // directional bonds change form but still describe trans-difluoroethene
        // (RDKit reads `C(/F)=C\F` as F/C=C/F).
        let labeling = vec![1, 0, 3, 2];
        reorder_molecule(&mut mol, &labeling);
        assert_eq!(mol.to_smiles(), "C(/F)=C\\F");

        let mut mol = MolecularGraph::from_smiles_single("F/C(C)=C(C)/F").unwrap();
        let chiral_classes = mol.chiral_classes().unwrap().to_vec();
        println!("Chiral classes: {:?}", chiral_classes);
        println!("Molecule: {:?}", mol.to_smiles());

        let labeling = vec![1, 0, 3, 2];
        reorder_molecule(&mut mol, &labeling);
        println!("Molecule: {:?}", mol.to_smiles());

        let mol = MolecularGraph::from_smiles_single("F/C(C)=C(C)/F").unwrap();
        println!("Molecule: {:?}", mol.to_smiles());
        println!("Starting canonization");
        // let mol = mol_canonization(&mol);

        println!("Chiral classes: {:?}", mol.chiral_classes());
        println!("Molecule: {:?}", mol.to_smiles());

        let mol = MolecularGraph::from_smiles_single("F[C@@H]=[C@H]F").unwrap();
        // let mol = mol_canonization(&mol);
        println!("Molecule: {:?}", mol.to_smiles());

        let mol = MolecularGraph::from_smiles_single("F[C@H]=[C@@H]F").unwrap();
        // let mol = mol_canonization(&mol);
        println!("Molecule: {:?}", mol.to_smiles());

        let mol = MolecularGraph::from_smiles_single("C1(=NC)SC(=C)C(=O)N1(C)").unwrap();

        println!("Molecule: {:?}", mol.to_smiles());

        let mol = MolecularGraph::from_smiles_single("C1(=NC4)SC(=C)C(=O)N1(C4)").unwrap();
        // let mol = mol_canonization(&mol);
        println!("Molecule: {:?}", mol.to_smiles());

        // L-Cysteine: all input forms must canonicalize identically, and the
        // enantiomer (D-Cysteine) must stay distinct. The canonical form's
        // chirality sense matches RDKit (N[C@@H](CS)C(=O)O for L).
        let l_forms = [
            "C([C@@H](C(=O)O)N)S",
            "SC[C@H](N)C(O)=O",
            "N[C@@H](CS)C(=O)O",
            "OC(=O)[C@@H](N)CS",
        ];
        let l_canonical = mol_canonization::canonicalize_and_sort_smiles(l_forms[0], false).unwrap();
        assert_eq!(l_canonical, "SC[C@H](N)C(O)=O");
        for form in &l_forms[1..] {
            assert_eq!(
                mol_canonization::canonicalize_and_sort_smiles(form, false).unwrap(),
                l_canonical,
                "L-Cysteine form {form} did not canonicalize to {l_canonical}"
            );
        }

        // D-Cysteine
        let d_canonical =
            mol_canonization::canonicalize_and_sort_smiles("C([C@H](C(=O)O)N)S", false).unwrap();
        assert_eq!(d_canonical, "SC[C@@H](N)C(O)=O");
        assert_ne!(l_canonical, d_canonical);
    }
}

mod tautomer_tests {
    use crate::prelude::*;

    #[test]
    fn test_tautomers() {
        let mut molecule = MolecularGraph::from_smiles_single("CC=CC=O").unwrap();
        let mut tautomers = molecule.find_tautomers();
        molecule.remove_tautomeric_hydrogens(&mut tautomers);
        println!("{:?}", tautomers);
        println!("{:#?}", molecule.atomic_numbers());

        let molecule = MolecularGraph::from_smiles_single("CC=CO").unwrap();
        let tautomers = molecule.find_tautomers();
        println!("{:?}", tautomers);
    }
}

mod subgraph_isomorphism_tests {
    use super::*;

    /// A pattern of one atom is a perfectly good substructure query.
    ///
    /// Both sides' connected components used to be filtered to those with more
    /// than one atom, so a single-atom pattern had nothing left to match with
    /// and always reported nothing — and a single-atom component of the
    /// *target*, a counter-ion say, was invisible to every pattern.
    #[test]
    fn single_atom_patterns_match() {
        let ethanol = MolecularGraph::from_smiles_single("CCO").unwrap();
        let carbon = MolecularGraph::from_smiles_single_raw("C").unwrap();
        let oxygen = MolecularGraph::from_smiles_single_raw("O").unwrap();
        let nitrogen = MolecularGraph::from_smiles_single_raw("N").unwrap();

        assert_eq!(ethanol.match_submolecule(&carbon).unwrap().len(), 2);
        assert_eq!(ethanol.match_submolecule(&oxygen).unwrap().len(), 1);
        assert!(ethanol.match_submolecule(&nitrogen).is_none(), "no nitrogen in ethanol");

        // A one-atom molecule against a one-atom pattern.
        let methane = MolecularGraph::from_smiles_single("C").unwrap();
        assert_eq!(methane.match_submolecule(&carbon).unwrap().len(), 1);
    }

    /// The reported mapping has to name real atoms of both molecules.
    #[test]
    fn single_atom_match_names_the_right_atom() {
        let ethanol = MolecularGraph::from_smiles_single("CCO").unwrap();
        let oxygen = MolecularGraph::from_smiles_single_raw("O").unwrap();

        let matches = ethanol.match_submolecule(&oxygen).unwrap();
        assert_eq!(matches.len(), 1);
        let (&target_atom, &pattern_atom) = matches[0].iter().next().unwrap();
        assert_eq!(ethanol.atomic_numbers()[target_atom], 8);
        assert_eq!(oxygen.atomic_numbers()[pattern_atom], 8);
    }

    /// Substructure search is subgraph *monomorphism*: the target may have
    /// bonds between matched atoms that the pattern does not mention.
    ///
    /// A three-atom chain occurs in cyclopropane six times over, even though
    /// its two ends are bonded there. petgraph's `subgraph_isomorphisms_iter`
    /// is node-*induced* and rejected all six; the counts below are RDKit's.
    #[test]
    fn matching_is_monomorphism_not_induced_isomorphism() {
        for (pattern, target, expected) in [
            ("CCC", "C1CC1", 6),
            ("CCCC", "C1CCC1", 8),
            ("CC", "C1CC1", 6),
            ("CCC", "C1CCCCC1", 12),
            ("c1ccccc1", "c1ccc2ccccc2c1", 24),
        ] {
            let query = MolecularGraph::from_smiles_single_raw(pattern).unwrap();
            let mol = MolecularGraph::from_smiles_single(target).unwrap();
            assert_eq!(
                mol.match_submolecule(&query).map(|m| m.len()).unwrap_or(0),
                expected,
                "{pattern} in {target}"
            );
        }
    }

    /// Atom labels are compared exactly — element, charge, aromaticity,
    /// isotope, stereocentre-or-not — which is *stricter* than RDKit's
    /// substructure search with a SMILES query, where isotope and aromaticity
    /// are ignored by default. Use SMARTS for query semantics.
    #[test]
    fn labels_are_compared_exactly() {
        let aliphatic = MolecularGraph::from_smiles_single_raw("C").unwrap();
        let benzene = MolecularGraph::from_smiles_single("c1ccccc1").unwrap();
        assert!(
            benzene.match_submolecule(&aliphatic).is_none(),
            "an aliphatic carbon must not match an aromatic one"
        );

        let labelled = MolecularGraph::from_smiles_single("[13CH3]CO").unwrap();
        assert_eq!(
            labelled.match_submolecule(&aliphatic).unwrap().len(),
            1,
            "only the unlabelled carbon"
        );
    }

    /// Multi-atom patterns keep working.
    #[test]
    fn multi_atom_patterns_still_match() {
        let ethanol = MolecularGraph::from_smiles_single("CCO").unwrap();
        for (pattern, expected) in [("CC", 2), ("CO", 1), ("CCO", 1)] {
            let query = MolecularGraph::from_smiles_single_raw(pattern).unwrap();
            assert_eq!(
                ethanol.match_submolecule(&query).map(|m| m.len()).unwrap_or(0),
                expected,
                "{pattern}"
            );
        }
    }

    /// Benzene written in Kekulé form still has alternating single and double
    /// bonds *if aromaticity perception is turned off*; with the default it is
    /// perceived as aromatic and neither pattern matches, which is the point
    /// of [`SmilesParseOptions`].
    #[test]
    fn ethane_in_kekule_benzene() {
        use crate::io::SmilesParseOptions;

        let literal = MolecularGraph::from_smiles_single_with(
            "C=1C=CC=CC=1",
            SmilesParseOptions { aromaticity: None },
        )
        .unwrap();
        let single_bond_pattern = MolecularGraph::from_smiles_single_raw("CC").unwrap();
        let double_bond_pattern = MolecularGraph::from_smiles_single_raw("C=C").unwrap();

        assert_eq!(literal.match_submolecule(&single_bond_pattern).unwrap().len(), 6);
        assert_eq!(literal.match_submolecule(&double_bond_pattern).unwrap().len(), 6);

        // Perceived by default: the ring becomes aromatic, so it no longer has
        // any plain single or double bonds to find.
        let perceived = MolecularGraph::from_smiles_single("C=1C=CC=CC=1").unwrap();
        assert!(perceived.match_submolecule(&double_bond_pattern).is_none());
    }
}

mod wildcard_atom_tests {
    use super::*;
    use crate::atom::WILDCARD_ATOMIC_NUMBER;
    use crate::io::SMILESParseError;

    /// `*` is an atom whose element is deliberately unspecified — attachment
    /// points, polymer end caps, Markush fragments. It must survive a
    /// round-trip unchanged.
    ///
    /// Before this it was not in the parser's character set at all, and the
    /// dispatch stopped silently on anything it did not recognise, so
    /// `*C(F)(Cl)Br` parsed to an *empty molecule* and still returned `Ok`.
    #[test]
    fn wildcard_round_trips() {
        for smiles in [
            "*",
            "CC*",
            "*C(F)(Cl)Br",
            "*C(*)(Cl)Br",
            "*[C@](F)(Cl)Br",
            "*[C@]12CCCC[C@H]1CCCC2",
        ] {
            let mol = MolecularGraph::from_smiles_single(smiles)
                .unwrap_or_else(|e| panic!("{smiles}: {e:?}"));
            assert!(
                mol.atomic_numbers().contains(&WILDCARD_ATOMIC_NUMBER),
                "{smiles}: no wildcard atom in the parsed molecule"
            );
            assert_eq!(mol.to_smiles(), smiles, "round-trip of {smiles}");
        }
    }

    /// The bracketed form denotes the same atom as the bare one.
    #[test]
    fn bracketed_wildcard_equals_bare_wildcard() {
        let bare = MolecularGraph::from_smiles_single("*C(F)(Cl)Br").unwrap();
        let bracketed = MolecularGraph::from_smiles_single("[*]C(F)(Cl)Br").unwrap();
        assert_eq!(bare.atomic_numbers(), bracketed.atomic_numbers());
        assert_eq!(bare.to_smiles(), bracketed.to_smiles());
    }

    /// A wildcard has no valence, so it must not collect implicit hydrogens.
    #[test]
    fn wildcard_gets_no_implicit_hydrogens() {
        let mol = MolecularGraph::from_smiles_single("CC*").unwrap();
        let hydrogens = mol.atomic_numbers().iter().filter(|&&z| z == 1).count();
        // Ethane minus one hydrogen: CH3-CH2-*
        assert_eq!(hydrogens, 5, "{:?}", mol.atomic_numbers());
    }

    /// A SMILES ends at whitespace — trailing names and identifiers are not
    /// structure — but any other unknown character is an error, not a reason to
    /// quietly return whatever was parsed so far.
    #[test]
    fn whitespace_terminates_but_junk_is_an_error() {
        for (smiles, expected_atoms) in [("CCO ethanol", 9), ("CCO\tid=1", 9), ("CCO", 9)] {
            let mol = MolecularGraph::from_smiles_single(smiles)
                .unwrap_or_else(|e| panic!("{smiles}: {e:?}"));
            assert_eq!(mol.atomic_numbers().len(), expected_atoms, "{smiles}");
        }

        for smiles in ["C&CC", "CC^", "C|C"] {
            match MolecularGraph::from_smiles_single(smiles) {
                Err(SMILESParseError::UnexpectedCharacter(_, _)) => {}
                other => panic!("{smiles}: expected UnexpectedCharacter, got {other:?}"),
            }
        }
    }

    /// A wildcard is a vertex of the canonical graph like any other non-
    /// hydrogen atom. It used to be counted out of the graph while its bonds
    /// were counted in, so canonizing anything with a `*` indexed past the
    /// adjacency matrix and panicked -- `C*C` included.
    #[test]
    fn wildcard_molecules_canonize_the_same_however_written() {
        use crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
        for writings in [
            &["C*C", "C(*C)"][..],
            &["*CCO", "OCC*", "C(*)CO"],
            &["*c1ccc(*)cc1N", "Nc1cc(*)ccc1*", "c1(*)ccc(*)c(N)c1"],
            &["C1CC*CC1", "*1CCCCC1"],
            &["*C(=O)O", "OC(*)=O"],
        ] {
            let canonical: Vec<String> = writings
                .iter()
                .map(|smiles| {
                    canonicalize_and_sort_smiles(smiles, false)
                        .unwrap_or_else(|e| panic!("{smiles}: {e:?}"))
                })
                .collect();
            assert!(
                canonical.iter().all(|form| form == &canonical[0]),
                "{writings:?} canonized to {canonical:?}"
            );
        }
    }

    /// A bracket with no element in it is an error, not a panic: the parser
    /// read two bytes ahead for `se`/`as` and ran off the end of `C[2]`.
    #[test]
    fn a_bracket_without_an_element_is_rejected() {
        for smiles in ["[]", "C[]", "C[2]", "[2]", "C.[]"] {
            assert!(
                MolecularGraph::from_smiles(smiles).is_err(),
                "{smiles} should not parse"
            );
        }
    }
}

mod isotope_tests {
    use super::*;
    use crate::graph_algorithms::mol_canonization::canonize_mol;
    use crate::molecule::base::MoleculeCore;

    /// A stated isotope must survive being written back out.
    ///
    /// It never was: the bracket branch of `construct_smiles` formatted only
    /// symbol, chirality, hydrogens and charge, so `[13CH4]` came back as `C`
    /// and heavy water as water. The parser had it right all along — the
    /// molecule stored the mass number — but nothing could read it back.
    #[test]
    fn isotopes_round_trip() {
        for smiles in [
            "[13CH4]",
            "[13CH3][C@H](C)O",
            "[15NH2]CC(=O)O",
            "[235U]",
            "[3*]CC",
            "[13CH3][13CH2]O",
        ] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(mol.to_smiles(), smiles, "round-trip of {smiles}");
        }
    }

    /// A labelled hydrogen cannot fold into its neighbour's hydrogen count —
    /// `[2H]` has to be written as an atom of its own — so it needs a node in
    /// the output tree and must not also be counted on the neighbour.
    #[test]
    fn labelled_hydrogens_are_written_as_atoms() {
        for smiles in ["[2H]O[2H]", "O[2H]", "[2H]C([2H])([2H])[2H]"] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(mol.to_smiles(), smiles, "round-trip of {smiles}");
        }
    }

    /// Zero in the isotope vector means "none was written", not "mass number
    /// zero". Filling in the element's default instead made an unlabelled atom
    /// indistinguishable from an explicitly labelled one, and made an atom's
    /// stored isotope depend on whether some *other* atom carried one.
    #[test]
    fn an_unstated_isotope_stays_unstated() {
        let labelled = MolecularGraph::from_smiles_single("[13CH3]CO").unwrap();
        assert_eq!(labelled.get_isotope(0), Some(13));
        assert_eq!(labelled.get_isotope(1), None, "the second carbon carries no label");
        assert_eq!(labelled.get_isotope(2), None, "the oxygen carries no label");

        // The effective mass number is still available, and is the same
        // whether or not any atom in the molecule was labelled.
        let plain = MolecularGraph::from_smiles_single("CCO").unwrap();
        assert_eq!(labelled.get_atom_isotope(2), plain.get_atom_isotope(2));
    }

    /// Isotopomers are different molecules and must canonize apart; the same
    /// isotopomer written two ways must canonize together.
    #[test]
    fn canonization_distinguishes_isotopomers() {
        let canonical = |smiles: &str| {
            canonize_mol(&MolecularGraph::from_smiles_single(smiles).unwrap()).to_smiles()
        };

        assert_ne!(canonical("[13CH3]CO"), canonical("C[13CH2]O"));
        assert_ne!(canonical("[13CH3]CO"), canonical("CCO"));
        assert_ne!(canonical("[2H]O"), canonical("O"));

        assert_eq!(canonical("[13CH3]CO"), canonical("OC[13CH3]"));
        assert_eq!(canonical("C[13CH2]O"), canonical("O[13CH2]C"));
        assert_eq!(canonical("[2H]OC"), canonical("CO[2H]"));
    }
}

mod radical_tests {
    use super::*;

    /// An atom in brackets states its own hydrogen count, so whatever valence
    /// is left over is unpaired electrons — not more hydrogens.
    ///
    /// This machinery already existed, keyed off `Atom::is_complex`, but that
    /// flag was only set when the brackets *also* carried an atom map. So
    /// `[CH3:1]` came out as the methyl radical it is while plain `[CH3]`
    /// silently gained a fourth hydrogen and became methane. Every case here
    /// matches RDKit.
    #[test]
    fn bracketed_atoms_become_radicals_not_hydrides() {
        for (smiles, hydrogens, radicals) in [
            ("[CH3]", 3, 1),
            ("[CH2]", 2, 2),
            ("[CH]", 1, 3),
            ("[C]", 0, 4),
            ("[NH2]", 2, 1),
            ("[OH]", 1, 1),
            ("[CH3:1]", 3, 1),
        ] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(
                mol.number_of_bonded_element(0, 1),
                hydrogens,
                "{smiles}: hydrogen count"
            );
            assert_eq!(mol.radical_states()[0], radicals, "{smiles}: unpaired electrons");
        }
    }

    /// SMILES has no symbol for an unpaired electron — a radical is a bracketed
    /// atom whose stated hydrogen count falls short of its valence — so the
    /// writer has to bracket one, or the methyl radical comes back as methane.
    #[test]
    fn radicals_round_trip() {
        for smiles in ["[CH3]", "[CH2]", "[CH]", "[C]", "[NH2]", "[OH]"] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(mol.to_smiles(), smiles);
        }
    }

    /// Ordinary molecules are unaffected: an atom outside brackets still gets
    /// the hydrogens its valence asks for.
    #[test]
    fn unbracketed_atoms_still_get_their_hydrogens() {
        for (smiles, atom_count) in [("C", 5), ("CCO", 9), ("c1ccccc1", 12), ("N", 4)] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(mol.atomic_numbers().len(), atom_count, "{smiles}");
            assert!(mol.radical_states().iter().all(|&state| state == 0), "{smiles}");
        }
    }

    /// Charged species keep their stated hydrogens too.
    #[test]
    fn charged_bracketed_atoms_keep_their_hydrogens() {
        for smiles in ["[NH4+]", "[NH3+]", "[N+]", "[O-]", "[OH-]", "[OH3+]", "[BH4-]", "[Na+]", "[Cl-]"] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(mol.to_smiles(), smiles);
        }
    }
}

mod charge_valency_tests {
    use super::*;

    /// A formal charge changes how many bonds an atom forms, and which way
    /// depends on the element's group.
    ///
    /// The model used to be `charge.abs()` — a charge always *costs* a bond —
    /// which is right for anions and backwards for every cation, so ammonium
    /// came out divalent. It was wrong in two places at once, since
    /// `Atom::actual_valency` folded the same `|charge|` in again, and each
    /// attempt to correct one alone threw the other out by a bond. Every case
    /// here matches RDKit, in structure and in unpaired-electron count.
    #[test]
    fn charged_species_round_trip() {
        for smiles in [
            // Nitrogen and beyond: a positive charge buys a bond.
            "[NH4+]", "[NH3+]", "[N+]", "C[N+]", "C[NH3+]", "[NH-]C", "C[N-]C",
            "[OH3+]", "[OH-]", "[O-]", "C[O-]", "C[O+]", "[S-]C", "[PH4+]",
            // Boron's group: a negative charge buys one.
            "[BH4-]",
            // Carbon's group: either sign costs one.
            "C[CH2+]", "[SiH3]",
            // Metals and halides have no hydrogens to gain or lose.
            "[Na+]", "[Cl-]", "[Fe]",
        ] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(mol.to_smiles(), smiles, "round-trip of {smiles}");
        }
    }

    /// The unpaired-electron counts that follow from that model, spot-checked
    /// against RDKit.
    #[test]
    fn radical_counts_follow_the_charge() {
        for (smiles, atom, radicals) in [
            ("[N+]", 0, 4),    // tetravalent, no bonds at all
            ("C[N+]", 1, 3),   // the nitrogen, one bond short of four
            ("[NH3+]", 0, 1),
            ("[NH4+]", 0, 0),
            ("[O-]", 0, 1),    // monovalent, no bonds
            ("[OH-]", 0, 0),
            ("[Na+]", 0, 0),
            ("[BH4-]", 0, 0),
        ] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(mol.radical_states()[atom], radicals, "{smiles} atom {atom}");
        }
    }
}

mod organic_subset_tests {
    use super::*;

    /// The organic subset is written without brackets, up to the number of
    /// hydrogens a reader would infer for a bare atom — its lowest valence.
    ///
    /// Nitrogen and phosphorus were capped at two rather than three, so
    /// ammonia came out as `[NH3]`, and boron was missing from the subset
    /// altogether, so `B` came out as `[BH3]`. All valid SMILES, just longer
    /// than needed and different from what every other toolkit writes.
    #[test]
    fn organic_subset_atoms_are_written_bare() {
        for smiles in [
            "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "CB", "CN", "CP", "CO", "CS",
        ] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(mol.to_smiles(), smiles, "{smiles}");
        }
    }

    /// Anything the bare form could not express still gets brackets: a charge,
    /// an isotope, an unpaired electron, an aromatic nitrogen bearing a
    /// hydrogen, or more hydrogens than the lowest valence.
    #[test]
    fn everything_else_still_gets_brackets() {
        for smiles in ["[NH4+]", "[CH3]", "[13CH3]C", "[2H]O", "c1cc[nH]c1", "[Na+]", "[Fe]"] {
            let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
            assert_eq!(mol.to_smiles(), smiles, "{smiles}");
        }
    }
}
