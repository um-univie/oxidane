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
    ///
    /// Distance geometry only. `embed_3d` finishes by relaxing against the
    /// force field, and that step exists precisely to overrule the bounds
    /// where they are approximate -- so asserting the bounds on its output
    /// would be testing the wrong stage. Thiophene is the case that shows
    /// it: the bounds put the two carbons flanking sulfur at 2.68-2.76 A,
    /// because the ring-angle model treats a five-ring as a regular
    /// pentagon (RDKit's bounds builder assumes the same); UFF, which knows
    /// `S_R` wants 92.2 degrees rather than 108, puts them at 2.38, and the
    /// experimental value is 2.47.
    fn assert_embedding_respects_bounds(mol: &MolecularGraph) {
        let config = crate::molecule::embed::EmbedConfig { relax: false, ..Default::default() };
        let embedded = mol.embed_3d_with_config(&config).unwrap_or_else(|e| {
            panic!("embed_3d failed: {e}");
        });

        let mut working = mol.clone();
        working.add_hydrogens();
        assert_eq!(
            working.len(),
            embedded.len(),
            "embedded atom count should match topology-only atom count (add_hydrogens)"
        );

        let bounds_matrix = bounds::build_bounds(&working, &bounds::Geometry::for_molecule(&working, &bounds::ring_set(&working)), &bounds::ring_set(&working)).unwrap();
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

    #[test]
    fn an_unsupported_fragment_does_not_disable_other_fragments_relaxation() {
        let connected = MolecularGraph::from_smiles_single("CCO")
            .unwrap().embed_3d().unwrap();
        let disconnected = disconnected_fixture().embed_3d().unwrap();
        let field = crate::molecule::forcefield::Field::best_for(&connected).unwrap();
        let energy = field.energy(&connected.positions);
        for positions in disconnected.positions[..2 * connected.len()].chunks(connected.len()) {
            assert!((field.energy(positions) - energy).abs() < 1e-8);
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
    /// A tagged double bond has to come out *planar*, not merely on the
    /// right side of perpendicular.
    ///
    /// [`embedding_preserves_cis_trans_double_bonds`] checks the side, and
    /// its `cos = +-0.3` threshold is 72/108 degrees -- wide enough that a
    /// bond 70 degrees out of plane still passes it. It did: before the
    /// stereo bonds were given the one-fold torsion restraint in
    /// `embed::forcefield`, every case below sat between 41 and 135 degrees
    /// from its target while that test stayed green. This one measures the
    /// dihedral itself.
    #[test]
    fn tagged_double_bonds_embed_planar_not_merely_on_the_right_side() {
        // (SMILES, target dihedral in degrees for the first heavy
        // substituent on each end)
        let cases = [
            ("C/C=C/C", 180.0),
            ("C/C=C/C(=O)O", 180.0),
            ("F/C=C/F", 180.0),
            ("C/C=C/Br", 180.0),
            ("C/C(F)=C(Cl)/C", 0.0),
            ("OC(=O)/C=C\\C(=O)O", 0.0),
        ];
        for (smiles, target) in cases {
            let molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
            let embedded = molecule.embed_3d().expect("should embed");
            let mut checked = 0;
            for a in 0..embedded.len() {
                for bond in embedded.get_atom_bonds(a).unwrap_or_default() {
                    let b = bond.target();
                    if bond.bond_order() != BondOrder::Double || b < a {
                        continue;
                    }
                    let heavy = |from: usize, exclude: usize| {
                        embedded
                            .get_atom_bonds(from)
                            .unwrap()
                            .iter()
                            .map(|x| x.target())
                            .find(|&t| t != exclude && embedded.get_atomic_number(t) != 1)
                    };
                    let (Some(sub_a), Some(sub_b)) = (heavy(a, b), heavy(b, a)) else {
                        continue;
                    };

                    let pa = embedded.get_atom_position(a).unwrap();
                    let pb = embedded.get_atom_position(b).unwrap();
                    let axis = (pb - pa).normalize();
                    let project = |p: Vector3D, origin: Vector3D| {
                        let v = p - origin;
                        v - axis * v.dot(&axis)
                    };
                    let u = project(embedded.get_atom_position(sub_a).unwrap(), pa);
                    let v = project(embedded.get_atom_position(sub_b).unwrap(), pb);
                    let cos_angle = (u.dot(&v) / (u.length() * v.length())).clamp(-1.0, 1.0);
                    let degrees = cos_angle.acos().to_degrees();

                    // 20 degrees: loose enough for the real strain in a cis
                    // double bond crowded by two substituents, tight enough
                    // that none of the pre-fix geometries would pass.
                    assert!(
                        (degrees - target).abs() < 20.0,
                        "{smiles}: dihedral {degrees:.1} deg, expected {target} deg"
                    );
                    checked += 1;
                }
            }
            assert!(checked > 0, "{smiles}: found no tagged double bond to check");
        }
    }

    /// Aromatic rings have to come out flat.
    ///
    /// Distance geometry gets this right only if the distances say so, and
    /// through 1-3 they do not -- a hexagon with correct bonds and correct
    /// internal angles can still pucker, because that is the 1-4 distance's
    /// business. Before `bounds::apply_aromatic_ring_bounds` pinned the
    /// chords and aromatic rings were given their own planarity stiffness,
    /// the biphenyls below came out between 0.02 and 0.25 A out of plane;
    /// RDKit's ETKDG gives 0.0002 A on the same molecule.
    #[test]
    fn aromatic_rings_embed_flat() {
        let corpus = [
            "c1ccccc1",
            "c1ccccc1c2ccccc2",
            "c1ccc2ccccc2c1",                                     // naphthalene, fused
            "c1ccc(cc1)CCCCc2ccccc2",
            "c1ccccc1c2ccccc2CCCCCCCc3ccccc3c4ccccc4",            // the reported case
            // Two-coordinate ring heteroatoms: these fail the three-bond
            // `is_sp2_atom` test, which used to disqualify the whole ring
            // from any planarity restraint at all.
            "c1ccncc1",
            "c1ccoc1",
            "c1ccsc1",
            "c1cnc2[nH]ccc2c1",
            "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",                       // caffeine
            "c1cc2ccc3cccc4ccc(c1)c2c34",                         // pyrene
        ];
        for smiles in corpus {
            let molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
            let embedded = molecule.embed_3d().expect("should embed");
            let mut checked = 0;
            for cycle in embedded.compute_fundamental_cycles() {
                let atoms: Vec<usize> = cycle.iter().collect();
                if !(5..=6).contains(&atoms.len()) {
                    continue;
                }
                let aromatic = atoms.iter().all(|&a| {
                    embedded.get_atom_bonds(a).unwrap().iter().any(|bond| {
                        bond.bond_order() == BondOrder::Aromatic && cycle.contains(bond.target())
                    })
                });
                if !aromatic {
                    continue;
                }
                let rmsd = ring_out_of_plane_rmsd(&embedded, &atoms);
                assert!(
                    rmsd < 0.02,
                    "{smiles}: aromatic ring {atoms:?} is {rmsd:.4} A out of plane"
                );
                checked += 1;
            }
            assert!(checked > 0, "{smiles}: found no aromatic six-ring to check");
        }
    }

    /// Root-mean-square deviation of `atoms` from their own best-fit plane,
    /// taken about the centroid with the normal from the first three -- the
    /// same cheap construction the existing ring checks use.
    /// Root-mean-square distance of a ring's atoms from the plane that best
    /// fits them.
    ///
    /// The plane has to be the *best fit*, not the one through the first
    /// three atoms. Three consecutive atoms of a ring that puckers even
    /// slightly define a plane tilted away from the true one, and the tilt
    /// shows up as height on the atoms across the ring: the same biphenyl
    /// chain that measures 0.0077 A here read 0.021 that way, and RDKit's
    /// own output measures 0.005 by the best fit and would have failed the
    /// other test too.
    ///
    /// The normal is the smallest-eigenvalue eigenvector of the points'
    /// covariance, found by power iteration on `trace(C) * I - C`, whose
    /// largest eigenvector it is. A near-planar ring makes that gap enormous,
    /// so it converges in a handful of steps; the count here is generous
    /// rather than tuned, this being a test.
    fn ring_out_of_plane_rmsd(embedded: &Molecule3D, atoms: &[usize]) -> f64 {
        const POWER_ITERATIONS: usize = 100;
        let points: Vec<Vector3D> =
            atoms.iter().map(|&a| embedded.get_atom_position(a).unwrap()).collect();
        let centroid =
            points.iter().fold(Vector3D::default(), |sum, &p| sum + p) / points.len() as f64;

        let mut covariance = [[0.0_f64; 3]; 3];
        for &p in &points {
            let d = [p.x - centroid.x, p.y - centroid.y, p.z - centroid.z];
            for row in 0..3 {
                for column in 0..3 {
                    covariance[row][column] += d[row] * d[column];
                }
            }
        }
        let trace = covariance[0][0] + covariance[1][1] + covariance[2][2];
        let mut shifted = covariance;
        for axis in 0..3 {
            shifted[axis][axis] = trace - shifted[axis][axis];
        }
        for row in 0..3 {
            for column in 0..3 {
                if row != column {
                    shifted[row][column] = -shifted[row][column];
                }
            }
        }

        let seed = (points[1] - points[0]).cross(&(points[2] - points[0])).normalize();
        let mut vector = [seed.x, seed.y, seed.z];
        for _ in 0..POWER_ITERATIONS {
            let mut next = [0.0_f64; 3];
            for row in 0..3 {
                for column in 0..3 {
                    next[row] += shifted[row][column] * vector[column];
                }
            }
            let length = (next[0] * next[0] + next[1] * next[1] + next[2] * next[2]).sqrt();
            if length < 1e-30 {
                break;
            }
            vector = [next[0] / length, next[1] / length, next[2] / length];
        }
        let normal = Vector3D::new(vector[0], vector[1], vector[2]);
        let mean_square: f64 = points
            .iter()
            .map(|&p| {
                let height = (p - centroid).dot(&normal);
                height * height
            })
            .sum::<f64>()
            / points.len() as f64;
        mean_square.sqrt()
    }

    /// Asking for more trials has to be able to change the answer.
    ///
    /// It used not to. Trials were ranked on the bounds-violation energy,
    /// which is a flat-bottomed hinge -- exactly zero for any structure that
    /// meets its bounds, which most trials do -- so the comparison was a tie
    /// that the earliest trial always won. Raising `max_attempts` from 5 to
    /// 40 produced bit-identical coordinates, and every trial past the third
    /// was work thrown away.
    ///
    /// Also check that increasing the trial budget cannot increase the
    /// selected field's final energy: relaxation must happen before ranking.
    #[test]
    fn more_trials_can_change_the_embedding() {
        // Any one molecule may happen to find its best trial immediately, so
        // this asks whether the extra trials matter anywhere, not everywhere.
        let flexible = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C2CCC1C(C(C2)OC(=O)c1ccccc1)C(=O)OC",
        ];
        let mut changed = false;
        for smiles in flexible {
            let molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
            let few = molecule
                .embed_3d_with_config(&EmbedConfig { max_attempts: 2, ..EmbedConfig::default() })
                .expect("should embed");
            let many = molecule
                .embed_3d_with_config(&EmbedConfig { max_attempts: 20, ..EmbedConfig::default() })
                .expect("should embed");
            let field = crate::molecule::forcefield::Field::best_for(&few).unwrap();
            let (fewer, more) =
                (field.energy(&few.positions), field.energy(&many.positions));
            // Not exact any more, and deliberately so: trials are ranked
            // before the torsion search runs, so twenty trials can hand the
            // search a different starting conformer than two did and it can
            // finish a hair higher -- 0.005 kcal/mol on ibuprofen, which is
            // one relaxation path against another rather than a worse
            // structure. Ranking before *relaxation*, the bug this test is
            // here for, costs whole kcal/mol and still fails it.
            assert!(
                more <= fewer + 0.05,
                "{smiles}: more trials increased the relaxed energy, {fewer:.6} -> {more:.6}"
            );
            changed |= (0..few.len()).any(|atom| {
                few.get_atom_position(atom)
                    .unwrap()
                    .distance(&many.get_atom_position(atom).unwrap())
                    > 1e-6
            });
        }
        assert!(
            changed,
            "20 trials produced the same coordinates as 2 on every molecule - \
             the extra trials are being ignored"
        );
    }

    /// No two atoms that are not near neighbours may sit on top of each
    /// other.
    ///
    /// Distance geometry gives this for free only if the bounds ask for it.
    /// They did not: hydrogen's van der Waals radius was 1.10 A, which
    /// scaled to a 1.76 A floor under an H-H contact, and large flexible
    /// molecules duly came out sitting exactly on it -- satisfying their own
    /// bounds while carrying contacts RDKit's ETKDG keeps at 1.92 A or more.
    /// Check the distance-geometry stage: MMFF relaxation can intentionally
    /// shorten an O-H...O hydrogen-bond contact below this uniform floor.
    #[test]
    fn atoms_four_bonds_apart_keep_their_distance() {
        let corpus = [
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN1C2CCC1C(C(C2)OC(=O)c1ccccc1)C(=O)OC",
            "CC1CCC(=O)OC(C)CC(O)C(C)C(=O)C(C)CC1O",
            "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
        ];
        for smiles in corpus {
            let molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
            let embedded = molecule
                .embed_3d_with_config(&EmbedConfig { relax: false, ..Default::default() })
                .expect("should embed");
            let (closest, pair) = closest_distant_pair(&embedded);
            assert!(
                closest > 1.85,
                "{smiles}: atoms {pair:?} are {closest:.3} A apart"
            );
        }
    }

    /// Closest approach between two atoms at least four bonds apart, and
    /// which pair it was.
    fn closest_distant_pair(embedded: &Molecule3D) -> (f64, (usize, usize)) {
        let count = embedded.len();
        let mut closest = f64::MAX;
        let mut which = (0, 0);
        for start in 0..count {
            for (atom, hops) in bond_hops_from(embedded, start).into_iter().enumerate() {
                if atom <= start || hops < 4 {
                    continue;
                }
                let distance = embedded
                    .get_atom_position(start)
                    .unwrap()
                    .distance(&embedded.get_atom_position(atom).unwrap());
                if distance < closest {
                    closest = distance;
                    which = (start, atom);
                }
            }
        }
        (closest, which)
    }

    /// Bonds traversed from `start` to every atom, `usize::MAX` if
    /// unreachable.
    fn bond_hops_from(embedded: &Molecule3D, start: usize) -> Vec<usize> {
        let mut hops = vec![usize::MAX; embedded.len()];
        hops[start] = 0;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(atom) = queue.pop_front() {
            for bond in embedded.get_atom_bonds(atom).unwrap_or_default() {
                if hops[bond.target()] == usize::MAX {
                    hops[bond.target()] = hops[atom] + 1;
                    queue.push_back(bond.target());
                }
            }
        }
        hops
    }

    /// Every experimental torsion rule has to be usable.
    ///
    /// The rules are SMARTS, and a pattern this crate's parser cannot read
    /// is silently skipped -- correct behaviour, since the bond then falls
    /// back to its generic term, but it would let the rule set quietly
    /// erode. Bare `r` (ring membership, any size) was rejected when the
    /// table was first transcribed, costing five rules.
    #[test]
    fn every_torsion_preference_parses() {
        let unreadable: Vec<&str> = crate::molecule::embed::torsion_preference_smarts()
            .iter()
            .filter(|smarts| parse_cxsmarts(smarts).is_err())
            .copied()
            .collect();
        assert!(unreadable.is_empty(), "unreadable torsion SMARTS: {unreadable:?}");
    }

    /// The preferences have to actually reach the molecules they describe.
    ///
    /// An amide is the clearest case: crystal structures put it planar and
    /// overwhelmingly *trans*, which no hybridization-based guess produces.
    /// "Trans" here is the peptide convention, the C-C(=O)-N-C dihedral near
    /// 180 -- which puts O=C-N-C near 0, since the oxygen is opposite the
    /// carbonyl carbon's own substituent.
    #[test]
    fn an_amide_embeds_planar() {
        // N-methylacetamide: the O=C-N-C torsion should be near 180 degrees.
        let molecule = MolecularGraph::from_smiles_single("CC(=O)NC").expect("should parse");
        let embedded = molecule.embed_3d().expect("should embed");

        let carbonyl = (0..embedded.len())
            .find(|&atom| {
                embedded.get_atomic_number(atom) == 6
                    && embedded.get_atom_bonds(atom).unwrap().iter().any(|bond| {
                        bond.bond_order() == BondOrder::Double
                            && embedded.get_atomic_number(bond.target()) == 8
                    })
            })
            .expect("should have a carbonyl carbon");
        let neighbours = embedded.get_atom_bonds(carbonyl).unwrap();
        let oxygen = neighbours
            .iter()
            .find(|b| embedded.get_atomic_number(b.target()) == 8)
            .unwrap()
            .target();
        let nitrogen = neighbours
            .iter()
            .find(|b| embedded.get_atomic_number(b.target()) == 7)
            .unwrap()
            .target();
        let methyl = embedded
            .get_atom_bonds(nitrogen)
            .unwrap()
            .iter()
            .map(|b| b.target())
            .find(|&t| t != carbonyl && embedded.get_atomic_number(t) == 6)
            .expect("should have an N-methyl");

        let acetyl = neighbours
            .iter()
            .map(|b| b.target())
            .find(|&t| t != nitrogen && embedded.get_atomic_number(t) == 6)
            .expect("should have an acetyl methyl");

        let omega = dihedral_degrees(&embedded, acetyl, carbonyl, nitrogen, methyl);
        assert!(
            omega > 150.0,
            "C-C(=O)-N-C is {omega:.1} deg, expected near 180 (trans amide)"
        );
        // Planar, not merely on the right side: the two are separate claims
        // and only the pair rules out a twisted amide.
        let oxygen_side = dihedral_degrees(&embedded, oxygen, carbonyl, nitrogen, methyl);
        assert!(
            oxygen_side < 30.0,
            "O=C-N-C is {oxygen_side:.1} deg, expected near 0 for a planar trans amide"
        );
    }

    /// The torsion search may only ever lower the energy, and must leave
    /// stereochemistry where it found it.
    ///
    /// It relaxes each turn on a ranking budget and accepts on that number,
    /// so a turn could in principle be accepted on an energy the final
    /// structure does not have. Every molecule here is checked against the
    /// same pipeline with the search switched off, which is the only
    /// comparison that can catch that.
    #[test]
    fn the_torsion_search_only_ever_improves_a_conformer() {
        let flexible = [
            "CCCCCCCC",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CC(=O)Oc1ccccc1C(=O)O",
            // Chiral and flexible: the turns are judged after a relaxation,
            // and a relaxation is what can flatten a centre.
            "N[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)O",
        ];
        let mut improved = false;
        for smiles in flexible {
            let molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
            // `combined_refinement` is held at the default on both sides:
            // it changes the distance geometry, so varying it here would
            // compare two pipelines rather than the search against itself.
            let without = molecule
                .embed_3d_with_config(&EmbedConfig {
                    torsion_search: false,
                    ..EmbedConfig::default()
                })
                .expect("should embed");
            let with = molecule.embed_3d().expect("should embed");
            let field = crate::molecule::forcefield::Field::best_for(&with).unwrap();
            let (before, after) =
                (field.energy(&without.positions), field.energy(&with.positions));
            assert!(
                after <= before + 1e-8,
                "{smiles}: the torsion search raised the energy, {before:.3} -> {after:.3}"
            );
            improved |= after < before - 0.5;
            assert_eq!(
                with.chiral_classes(),
                without.chiral_classes(),
                "{smiles}: the torsion search changed a stereo assignment"
            );
        }
        assert!(
            improved,
            "the torsion search changed nothing on any flexible molecule"
        );
    }

    /// Every stereocentre comes out well clear of its own plane, not merely
    /// on the right side of it.
    ///
    /// A signed volume near zero is what everything downstream reads a
    /// centre's handedness off, and a centre left near zero is one the
    /// force field can take either way -- cholesterol's ring-fusion
    /// quaternary carbons came back at 0.24 cubic Angstroms against the 2.9
    /// of a healthy centre, and RDKit duly read one of them as the wrong
    /// isomer. Resolving handedness in four dimensions is what fixed it, so
    /// this is the test that says the fourth dimension is still doing its
    /// job.
    #[test]
    fn stereocentres_come_out_unambiguous() {
        for smiles in [
            "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
            "CN1CCC[C@H]1c1cccnc1",
            "N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)O",
        ] {
            let molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
            let embedded = molecule.embed_3d().expect("should embed");
            let mut working = molecule.clone();
            working.add_hydrogens();
            let classes = working.chiral_classes().expect("these all carry tags").to_vec();

            for (centre, class) in classes.iter().enumerate() {
                // The four-point convention: `TH(1)` is a negative volume.
                let target = match class {
                    ChiralClass::TH(1) => -1.0,
                    ChiralClass::TH(2) => 1.0,
                    _ => continue,
                };
                let mut neighbours: Vec<usize> = working.atom_bonds()[centre]
                    .iter()
                    .map(|bond| bond.target())
                    .collect();
                if neighbours.len() != 4 {
                    continue;
                }
                neighbours.sort_unstable();
                // The volume of the tetrahedron the four neighbours make,
                // which is what the embedder now restrains: about 11 for a
                // healthy centre, zero when they go coplanar.
                let position = |atom: usize| embedded.get_atom_position(atom).unwrap();
                let edge = |atom: usize| position(atom) - position(neighbours[0]);
                let volume = edge(neighbours[1]).dot(&edge(neighbours[2]).cross(&edge(neighbours[3])));
                assert!(
                    target * volume > 2.0,
                    "{smiles}: centre {centre} is only {:.2} clear of coplanar",
                    target * volume
                );
            }
        }
    }

    /// Relaxation may not buy its energy by inverting a stereocentre.
    ///
    /// A distance-geometry structure is strained, and on a stereo-dense
    /// saturated ring the cheapest way downhill can be through a centre
    /// rather than around it: glucose came back with up to four of its five
    /// centres inverted by RDKit's own reading, and cholesterol worse.
    /// `relax_positions` then rejected the whole relaxation, so the molecule
    /// was returned at its unrelaxed 162 kcal/mol instead of a relaxed 125.
    #[test]
    fn relaxation_does_not_invert_a_stereocentre_to_lower_the_energy() {
        for smiles in [
            "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
            "N[C@@H](C)C(=O)N[C@@H](CC(C)C)C(=O)O",
        ] {
            let molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
            let mut working = molecule.clone();
            working.add_hydrogens();
            let wanted = working.chiral_classes().unwrap().to_vec();

            let mut embedded = molecule.embed_3d().expect("should embed");
            embedded.chiral_classes = None;
            embedded.identify_chiral_classes();
            let found = embedded.chiral_classes().unwrap();
            for (atom, expected) in wanted.iter().enumerate() {
                if matches!(expected, ChiralClass::TH(_)) {
                    assert_eq!(
                        found[atom], *expected,
                        "{smiles}: atom {atom} came back {:?}, wanted {expected:?}",
                        found[atom]
                    );
                }
            }
        }
    }

    /// A macrocycle has no rotatable bonds at all, so a corner flip is the
    /// only move that can reach its conformation -- and it has to reach a
    /// better one, since that is the whole reason it exists.
    #[test]
    fn a_corner_flip_finds_a_better_macrocycle_conformer() {
        // The erythromycin fragment: a 13-membered ring, every bond in it.
        let smiles = "CC1CCC(=O)OC(C)CC(O)C(C)C(=O)C(C)CC1O";
        let molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
        let without = molecule
            .embed_3d_with_config(&crate::molecule::embed::EmbedConfig {
                torsion_search: false,
                ..Default::default()
            })
            .expect("should embed");
        let with = molecule.embed_3d().expect("should embed");
        let field = crate::molecule::forcefield::Field::best_for(&with).unwrap();
        let (before, after) = (field.energy(&without.positions), field.energy(&with.positions));
        assert!(
            after < before - 1.0,
            "corner flips bought the macrocycle nothing: {before:.2} -> {after:.2}"
        );
    }

    /// Morpholine's N-H has two places to be and only one of them is right.
    ///
    /// Which one is decided by the pyramidalisation at the nitrogen, not by
    /// any torsion, so neither the relaxation nor a bond rotation can change
    /// it -- it is the inversion move's reason for existing. Measured against
    /// 400 MMFF-minimised ETKDGv3 conformers, the axial arrangement this
    /// used to produce on every trial sits 1.42 kcal/mol above the
    /// equatorial one.
    #[test]
    fn a_secondary_amine_puts_its_hydrogen_equatorial() {
        let molecule = MolecularGraph::from_smiles_single("C1COCCN1").expect("should parse");
        let embedded = molecule.embed_3d().expect("should embed");
        let nitrogen = (0..embedded.len())
            .find(|&atom| embedded.atomic_numbers()[atom] == 7)
            .expect("morpholine has a nitrogen");
        let hydrogen = embedded
            .get_atom_bonds(nitrogen)
            .unwrap_or_default()
            .iter()
            .map(|bond| bond.target())
            .find(|&target| embedded.atomic_numbers()[target] == 1)
            .expect("the nitrogen carries a hydrogen");
        let position = |atom: usize| embedded.get_atom_position(atom).unwrap();

        // The ring's mean plane, by its own atoms' centroid and the normal
        // that two ring bonds across the nitrogen span.
        let ring: Vec<usize> = (0..embedded.len())
            .filter(|&atom| embedded.atomic_numbers()[atom] != 1)
            .collect();
        assert_eq!(ring.len(), 6, "morpholine's ring is six heavy atoms");
        let centroid = ring
            .iter()
            .fold(crate::vector::Vector3D::default(), |sum, &atom| sum + position(atom))
            * (1.0 / ring.len() as f64);
        let mut normal = crate::vector::Vector3D::default();
        for index in 0..ring.len() {
            let current = position(ring[index]) - centroid;
            let next = position(ring[(index + 1) % ring.len()]) - centroid;
            normal += current.cross(&next);
        }
        let normal = normal.normalize();
        let bond = (position(hydrogen) - position(nitrogen)).normalize();
        let out_of_plane = bond.dot(&normal).abs().asin().to_degrees();
        assert!(
            out_of_plane < 55.0,
            "the N-H came out axial at {out_of_plane:.1} degrees out of the ring plane"
        );
    }

    /// A ring bond is not a rotatable bond: turning one would break the ring
    /// open, since the far side of the bond is the rest of the ring. A corner
    /// flip does move ring atoms, and has to leave every bond a bond.
    #[test]
    fn the_torsion_search_leaves_rings_intact() {
        for smiles in ["C1CCCCC1", "CC1CCC(=O)OC(C)CC(O)C(C)C(=O)C(C)CC1O"] {
            let molecule = MolecularGraph::from_smiles_single(smiles).expect("should parse");
            let embedded = molecule.embed_3d().expect("should embed");
            for atom in 0..embedded.len() {
                for bond in embedded.get_atom_bonds(atom).unwrap_or_default() {
                    let length = embedded
                        .get_atom_position(atom)
                        .unwrap()
                        .distance(&embedded.get_atom_position(bond.target()).unwrap());
                    assert!(
                        (0.85..=2.1).contains(&length),
                        "{smiles}: bond {atom}-{} is {length:.2} A",
                        bond.target()
                    );
                }
            }
        }
    }

    fn dihedral_degrees(
        embedded: &Molecule3D,
        i: usize,
        b: usize,
        c: usize,
        j: usize,
    ) -> f64 {
        let position = |atom: usize| embedded.get_atom_position(atom).unwrap();
        let leading = position(i) - position(b);
        let axis = (position(c) - position(b)).normalize();
        let trailing = position(j) - position(c);
        let perpendicular = |v: Vector3D| v - axis * v.dot(&axis);
        let (u, w) = (perpendicular(leading), perpendicular(trailing));
        let cosine = (u.dot(&w) / (u.length() * w.length())).clamp(-1.0, 1.0);
        cosine.acos().to_degrees()
    }

}
