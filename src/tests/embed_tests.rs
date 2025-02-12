#[cfg(test)]
mod tests {
    use crate::molecule::bond::{BondOrder, BondTarget};
    use crate::molecule::chirality::{ChiralClass, ChiralClassifier};
    use crate::molecule::embed::bounds;
    use crate::prelude::*;
    use crate::vector::Vector3D;
    use std::fs;
    use tinyvec::ArrayVec;

    /// Every bonded (and non-bonded) pairwise 3D distance falls within its
    /// original bound, and no coordinate is NaN/infinite.
    fn assert_embedding_respects_bounds(mol: &MolecularGraph) {
        let embedded = mol.embed_3d().unwrap_or_else(|e| {
            panic!("embed_3d failed: {e}");
        });

        let mut working = mol.clone();
        working.add_hydrogens();
        assert_eq!(
            working.len(),
            embedded.len(),
            "embedded atom count should match topology-only atom count (add_hydrogens)"
        );

        let bounds_matrix = bounds::build_bounds(&working).unwrap();
        let mut smoothed = bounds_matrix.clone();
        bounds::smooth_bounds(&mut smoothed).unwrap();

        const EPS: f64 = 0.15; // refinement is not exact; allow slack

        for i in 0..embedded.len() {
            let pi = embedded.get_atom_position(i).unwrap();
            assert!(
                pi.x.is_finite() && pi.y.is_finite() && pi.z.is_finite(),
                "atom {i} has a non-finite coordinate"
            );
            for j in (i + 1)..embedded.len() {
                let pj = embedded.get_atom_position(j).unwrap();
                let d = pi.distance(&pj);
                let lower = bounds::get_lower(&smoothed, i, j);
                let upper = bounds::get_upper(&smoothed, i, j);
                assert!(
                    d >= lower - EPS && d <= upper + EPS,
                    "pair ({i},{j}) distance {d} outside bounds [{lower}, {upper}]"
                );
            }
        }
    }

    #[test]
    fn embeds_ethane() {
        let mol = MolecularGraph::from_smiles_single("CC").unwrap();
        assert_embedding_respects_bounds(&mol);
    }

    #[test]
    fn embeds_benzene() {
        let mol = MolecularGraph::from_smiles_single("c1ccccc1").unwrap();
        assert_embedding_respects_bounds(&mol);
    }

    #[test]
    fn embeds_cyclohexane() {
        let mol = MolecularGraph::from_smiles_single("C1CCCCC1").unwrap();
        assert_embedding_respects_bounds(&mol);
    }

    #[test]
    fn embeds_small_peptide_fragment() {
        // Glycylglycine-ish fragment: amide-linked chain with a carboxylic acid.
        let mol = MolecularGraph::from_smiles_single("NCC(=O)NCC(=O)O").unwrap();
        assert_embedding_respects_bounds(&mol);
    }

    #[test]
    fn embed_is_reproducible_for_a_fixed_seed() {
        let mol = MolecularGraph::from_smiles_single("c1ccccc1CCO").unwrap();
        let a = mol.embed_3d().unwrap();
        let b = mol.embed_3d().unwrap();
        for i in 0..a.len() {
            let pa = a.get_atom_position(i).unwrap();
            let pb = b.get_atom_position(i).unwrap();
            assert!(
                pa.distance(&pb) < 1e-9,
                "embed_3d() with the default (fixed) seed should be deterministic, atom {i} differs: {pa:?} vs {pb:?}"
            );
        }
    }

    /// Fuzz-style pass over a slice of the existing aromaticity SMILES corpus:
    /// no panics, no NaNs, every bonded pair lands within its bound.
    #[test]
    fn embeds_aromatization_corpus_without_panicking() {
        let contents = fs::read_to_string("tests/aromatization_tests.smi")
            .expect("tests/aromatization_tests.smi should exist");

        let mut tested = 0;
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some(smiles) = line.split(',').next() else {
                continue;
            };
            let smiles = smiles.trim();
            let Ok(mol) = MolecularGraph::from_smiles_single(smiles) else {
                continue;
            };
            if mol.len() > 40 {
                continue; // keep this fuzz pass fast; larger molecules covered ad hoc
            }
            assert_embedding_respects_bounds(&mol);
            tested += 1;
        }
        assert!(tested > 5, "expected to actually exercise several molecules from the corpus");
    }

    /// Bond-length bound satisfaction doesn't by itself guarantee the ring
    /// "looks like" a hexagon (a folded/bowtie conformation can satisfy
    /// pairwise bounds too) - directly check the interior C-C-C angle at
    /// each ring atom is close to the aromatic 120 deg target.
    #[test]
    fn benzene_ring_has_120_degree_interior_angles() {
        let mol = MolecularGraph::from_smiles_single("c1ccccc1").unwrap();
        let embedded = mol.embed_3d().unwrap();

        for i in 0..6 {
            let prev = embedded.get_atom_position((i + 5) % 6).unwrap();
            let center = embedded.get_atom_position(i).unwrap();
            let next = embedded.get_atom_position((i + 1) % 6).unwrap();
            // Interior angle at `center`: angle between the two rays from
            // `center` out to its ring neighbors (not the path "turn" angle
            // that `Vector3D::angle_between_points` computes).
            let angle = (prev - center).angle_between(&(next - center)).unwrap();
            let angle_deg = angle.to_degrees();
            assert!(
                (angle_deg - 120.0).abs() < 10.0,
                "expected ring interior angle near 120 deg at atom {i}, got {angle_deg:.1}"
            );
        }
    }

    /// Every `TH(1)`/`TH(2)` tag in the input SMILES survives embedding:
    /// re-deriving chirality from the *output* geometry via the existing
    /// [`ChiralClassifier`] oracle must agree with what was parsed in.
    fn assert_tetrahedral_chirality_preserved(smiles: &str) {
        let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
        let mut working = mol.clone();
        working.add_hydrogens();
        let expected = working.chiral_classes().unwrap().to_vec();

        let mut embedded = mol.embed_3d().unwrap();
        assert_eq!(embedded.len(), expected.len());
        embedded.chiral_classes = None;
        embedded.identify_chiral_classes();
        let actual = embedded.chiral_classes().unwrap();

        for (i, exp) in expected.iter().enumerate() {
            if matches!(exp, ChiralClass::TH(_)) {
                assert_eq!(
                    actual[i], *exp,
                    "SMILES {smiles}: atom {i} expected {exp:?}, embedding+oracle gave {:?}",
                    actual[i]
                );
            }
        }
    }

    #[test]
    fn embedding_preserves_single_tetrahedral_stereocenter() {
        for smiles in ["[C@](F)(Cl)(Br)I", "[C@@](F)(Cl)(Br)I", "N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"] {
            assert_tetrahedral_chirality_preserved(smiles);
        }
    }

    #[test]
    fn embedding_preserves_two_independent_tetrahedral_stereocenters() {
        // (2R,3R)/(2R,3S)-tartaric-acid-like backbones: two adjacent stereocenters.
        for smiles in ["OC(=O)[C@H](O)[C@H](O)C(=O)O", "OC(=O)[C@H](O)[C@@H](O)C(=O)O"] {
            assert_tetrahedral_chirality_preserved(smiles);
        }
    }

    /// Independent geometric check (projection onto the plane perpendicular
    /// to the double-bond axis, not the same cross-product formula the
    /// embedder's constraint itself uses) that each `/`,`\`-tagged double
    /// bond's substituents land on the correct side after embedding.
    fn assert_cis_trans_preserved(smiles: &str) {
        let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
        let mut working = mol.clone();
        working.add_hydrogens();
        let classes = working.chiral_classes().unwrap().to_vec();

        let embedded = mol.embed_3d().unwrap();
        let mut checked = 0;

        for a in 0..classes.len() {
            if !matches!(classes[a], ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise) {
                continue;
            }
            let Some(b) = embedded
                .get_atom_bonds(a)
                .unwrap()
                .iter()
                .find(|bd| bd.bond_order() == BondOrder::Double)
                .map(|bd| bd.target())
            else {
                continue;
            };
            if b <= a {
                continue;
            }
            let sub_a0 = embedded
                .get_atom_bonds(a)
                .unwrap()
                .iter()
                .find(|bd| bd.bond_order() != BondOrder::Double && bd.target() != b)
                .unwrap()
                .target();
            let sub_b0 = embedded
                .get_atom_bonds(b)
                .unwrap()
                .iter()
                .find(|bd| bd.bond_order() != BondOrder::Double && bd.target() != a)
                .unwrap()
                .target();

            let pa = embedded.get_atom_position(a).unwrap();
            let pb = embedded.get_atom_position(b).unwrap();
            let axis = (pb - pa).normalize();
            let project = |p: Vector3D, origin: Vector3D| -> Vector3D {
                let v = p - origin;
                v - axis * v.dot(&axis)
            };
            let proj_a0 = project(embedded.get_atom_position(sub_a0).unwrap(), pa);
            let proj_b0 = project(embedded.get_atom_position(sub_b0).unwrap(), pb);
            let cos_angle = proj_a0.dot(&proj_b0) / (proj_a0.length() * proj_b0.length());

            let expect_cis = classes[a] == classes[b];
            if expect_cis {
                assert!(cos_angle > 0.3, "SMILES {smiles}: expected cis substituents, cos(angle)={cos_angle}");
            } else {
                assert!(cos_angle < -0.3, "SMILES {smiles}: expected trans substituents, cos(angle)={cos_angle}");
            }
            checked += 1;
        }
        assert!(checked > 0, "SMILES {smiles}: expected at least one cis/trans double bond to check");
    }

    /// Root-mean-square out-of-plane deviation of `ring` from its own
    /// best-fit plane (via the first 3 members - fine for a sanity check,
    /// unlike the embedder's own Newell's-method restraint which needs the
    /// true cyclic order).
    fn planarity_rmsd(positions: &[Vector3D], ring: &[usize]) -> f64 {
        let centroid = ring.iter().fold(Vector3D::default(), |acc, &i| acc + positions[i]) / ring.len() as f64;
        let n0 = positions[ring[0]] - centroid;
        let n1 = positions[ring[1]] - centroid;
        let n2 = positions[ring[2]] - centroid;
        let normal = (n1 - n0).cross(&(n2 - n0)).normalize();
        let sq: f64 = ring.iter().map(|&i| normal.dot(&(positions[i] - centroid)).powi(2)).sum();
        (sq / ring.len() as f64).sqrt()
    }

    #[test]
    fn embeds_benzene_as_a_flat_ring() {
        let mol = MolecularGraph::from_smiles_single("c1ccccc1").unwrap();
        let embedded = mol.embed_3d().unwrap();
        let rmsd = planarity_rmsd(&embedded.positions, &[0, 1, 2, 3, 4, 5]);
        assert!(rmsd < 0.1, "expected benzene to embed nearly flat, out-of-plane RMSD={rmsd}");
    }

    #[test]
    fn embedding_preserves_cis_trans_double_bonds() {
        for smiles in ["F/C=C/F", "F/C=C\\F", "C/C=C/C(=O)O", "C/C=C\\C(=O)O"] {
            assert_cis_trans_preserved(smiles);
        }
    }

    /// A hand-built disconnected `Molecule3D` (this crate's SMILES reader
    /// normally splits multi-fragment input like salts into separate
    /// `MolecularGraph`s rather than one graph with disconnected atoms - see
    /// `MolecularGraph::from_smiles`/`::split` - so a disconnected molecule
    /// has to be constructed directly here to exercise that path).
    /// Two ethanol-like fragments (atoms 0-8, 9-17) plus a bare, bond-free
    /// atom (atom 18) standing in for a monatomic counter-ion. Argon (rather
    /// than e.g. a bare charged Na) so `add_hydrogens()` - which `embed_3d`
    /// calls again internally - doesn't try to attach an implicit H to it.
    fn disconnected_fixture() -> Molecule3D {
        let mut mol = MolecularGraph::from_smiles_single("CCO").unwrap();
        mol.add_hydrogens();
        let n = mol.len();

        let mut atomic_numbers = mol.atomic_numbers().to_vec();
        let mut atom_bonds = mol.atom_bonds().clone();
        atomic_numbers.extend(mol.atomic_numbers().to_vec());
        atom_bonds.extend(mol.atom_bonds().iter().map(|bonds| {
            bonds
                .iter()
                .map(|b| BondTarget::new(b.target() + n, b.bond_order()))
                .collect::<ArrayVec<[BondTarget; 10]>>()
        }));
        // A lone bond-free atom (argon, atomic number 18): no bonds to anything.
        atomic_numbers.push(18);
        atom_bonds.push(ArrayVec::new());

        let total = atomic_numbers.len();
        Molecule3D {
            atomic_numbers,
            atom_classes: None,
            charges: vec![0; total],
            chiral_classes: None,
            isotopes: None,
            positions: vec![Vector3D::default(); total],
            radical_states: vec![0; total],
            atom_bonds,
        }
    }

    /// Square-planar geometry (unlike TH/allene) has no separate handedness
    /// once the `SP1`/`SP2`/`SP3` trans-pairing is fixed (see
    /// `embed::bounds::square_planar_trans_pairs`'s doc comment), so this
    /// checks the resulting *angles* directly rather than any signed-volume
    /// oracle. Cross-checked independently against RDKit's
    /// `AssignStereochemistryFrom3D` (all of `SP1`/`SP2`/`SP3` round-trip
    /// exactly) during development; not re-checked here since that requires
    /// Python/RDKit, unlike the rest of this Rust-only suite.
    #[test]
    fn embeds_square_planar_center_with_correct_trans_pairing() {
        // Atom order: Pt=0, F=1, Cl=2, Br=3, I=4 - already index-sorted, so
        // SP1's trans pairs are (F,Br) and (Cl,I) directly.
        let mol = MolecularGraph::from_smiles_single("[Pt@SP1](F)(Cl)(Br)I").unwrap();
        let embedded = mol.embed_3d().unwrap();
        let pt = embedded.get_atom_position(0).unwrap();
        let angle = |a: usize, b: usize| {
            (embedded.get_atom_position(a).unwrap() - pt)
                .angle_between(&(embedded.get_atom_position(b).unwrap() - pt))
                .unwrap()
                .to_degrees()
        };

        for (a, b) in [(1, 3), (2, 4)] {
            let deg = angle(a, b);
            assert!((deg - 180.0).abs() < 20.0, "expected trans pair ({a},{b}) near 180 deg, got {deg}");
        }
        for (a, b) in [(1, 2), (2, 3), (3, 4), (4, 1)] {
            let deg = angle(a, b);
            assert!((deg - 90.0).abs() < 20.0, "expected adjacent pair ({a},{b}) near 90 deg, got {deg}");
        }
    }

    #[test]
    fn embeds_disconnected_components_without_overlap() {
        let mol = disconnected_fixture();
        assert_eq!(mol.get_components().len(), 3, "fixture should have 3 disconnected components");

        let embedded = mol.embed_3d().unwrap();
        assert_eq!(embedded.len(), mol.len());

        for i in 0..embedded.len() {
            let pi = embedded.get_atom_position(i).unwrap();
            assert!(pi.x.is_finite() && pi.y.is_finite() && pi.z.is_finite());
        }

        // Each ethanol fragment's own bonds should still be at bonded
        // lengths (~1 A range) - packing must not have distorted the
        // within-component geometry.
        for &(a, b) in &[(0usize, 1usize), (9, 10)] {
            let d = embedded.get_atom_position(a).unwrap().distance(&embedded.get_atom_position(b).unwrap());
            assert!(d < 2.0, "bonded pair ({a},{b}) has implausible distance {d} after packing");
        }

        // Cross-component pairs must not overlap (no bounds constrain them,
        // so the only guarantee is the packing loop's separation).
        let cross_min = (0..9)
            .flat_map(|a| (9..18).map(move |b| (a, b)))
            .map(|(a, b)| embedded.get_atom_position(a).unwrap().distance(&embedded.get_atom_position(b).unwrap()))
            .fold(f64::MAX, f64::min);
        assert!(cross_min > 1.0, "expected packed fragments not to overlap, min distance={cross_min}");
    }
}
