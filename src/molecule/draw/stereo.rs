//! Projection of the graph's index-relative stereo tags into drawing geometry.
use super::*;
use crate::molecule::chirality::ChiralClass;
use rustc_hash::FxHashSet;
type Point = (f64, f64);

pub(super) fn tetrahedral(c: ChiralClass) -> bool {
    matches!(c, ChiralClass::TH(1 | 2))
}
fn cis_trans(c: ChiralClass) -> bool {
    matches!(
        c,
        ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
    )
}
fn cross(a: Point, b: Point) -> f64 {
    a.0 * b.1 - a.1 * b.0
}
fn sub(a: Point, b: Point) -> Point {
    (a.0 - b.0, a.1 - b.1)
}

/// Reference substituents in bond-list order, exactly as stored by the parser.
/// If a reference H is hidden, use the other substituent and invert that end.
pub(super) fn double_bonds<M: Molecule>(
    m: &M,
    visible: &[bool],
) -> Vec<(usize, usize, usize, usize, f64)> {
    let Some(classes) = m.chiral_classes() else {
        return vec![];
    };
    let reference = |a: usize, b: usize| {
        let refs: Vec<_> = m.atom_bonds()[a]
            .iter()
            .filter(|e| e.target() != b && e.bond_order() != BondOrder::Double)
            .map(|e| e.target())
            .collect();
        refs.iter()
            .enumerate()
            .find(|&(_, i)| visible[*i])
            .map(|(k, &i)| (i, if k == 0 { 1.0 } else { -1.0 }))
    };
    let mut result = vec![];
    for a in 0..m.len() {
        if !cis_trans(classes[a]) {
            continue;
        }
        for e in &m.atom_bonds()[a] {
            let b = e.target();
            if b <= a || e.bond_order() != BondOrder::Double || !cis_trans(classes[b]) {
                continue;
            }
            if let (Some((ra, sa)), Some((rb, sb))) = (reference(a, b), reference(b, a)) {
                result.push((
                    a,
                    b,
                    ra,
                    rb,
                    sa * sb * if classes[a] == classes[b] { 1.0 } else { -1.0 },
                ));
            }
        }
    }
    result
}

fn fragment<M: Molecule>(m: &M, start: usize, cut: (usize, usize)) -> Vec<usize> {
    let mut seen = vec![false; m.len()];
    let mut q = vec![start];
    seen[start] = true;
    let mut head = 0;
    while head < q.len() {
        let i = q[head];
        head += 1;
        for e in &m.atom_bonds()[i] {
            let j = e.target();
            if (i == cut.0 && j == cut.1) || (i == cut.1 && j == cut.0) || seen[j] {
                continue;
            }
            seen[j] = true;
            q.push(j);
        }
    }
    q
}
fn reflect(p: &mut [Point], ids: &[usize], a: usize, b: usize) {
    let origin = p[a];
    let d = sub(p[b], origin);
    let l = d.0 * d.0 + d.1 * d.1;
    for &i in ids {
        let v = sub(p[i], origin);
        let t = (v.0 * d.0 + v.1 * d.1) / l;
        p[i] = (
            origin.0 + 2.0 * t * d.0 - v.0,
            origin.1 + 2.0 * t * d.1 - v.1,
        );
    }
}

pub(super) fn enforce_double_bonds<M: Molecule>(m: &M, p: &mut [Point], visible: &[bool]) {
    for (a, b, ra, rb, sign) in double_bonds(m, visible) {
        let axis = sub(p[b], p[a]);
        if cross(axis, sub(p[ra], p[a])) * cross(axis, sub(p[rb], p[b])) * sign > 1e-6 {
            continue;
        }
        let side = fragment(m, b, (a, b));
        if !side.contains(&a) {
            reflect(p, &side, a, b);
            continue;
        }
        // An exocyclic reference can be moved without opening a ring.
        let branch = fragment(m, rb, (b, rb));
        if !branch.contains(&b) {
            reflect(p, &branch, a, b);
            continue;
        }
        let branch = fragment(m, ra, (a, ra));
        if !branch.contains(&a) {
            reflect(p, &branch, a, b);
            continue;
        }
        // Macrocyclic alkene: invert the local ring segment at one endpoint,
        // then relax bond lengths with the alkene and its references fixed.
        reflect(p, &[rb], a, b);
        let fixed = [a, b, ra, rb];
        for _ in 0..160 {
            for &i in &side {
                for e in &m.atom_bonds()[i] {
                    let j = e.target();
                    if j <= i || !visible[i] || !visible[j] {
                        continue;
                    }
                    let d = sub(p[j], p[i]);
                    let l = d.0.hypot(d.1).max(1e-8);
                    let f = (l - 1.0) * 0.18 / l;
                    if !fixed.contains(&i) {
                        p[i].0 += d.0 * f;
                        p[i].1 += d.1 * f;
                    }
                    if !fixed.contains(&j) {
                        p[j].0 -= d.0 * f;
                        p[j].1 -= d.1 * f;
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Wedge {
    pub center: usize,
    pub target: usize,
    pub solid: bool,
}

/// Signed tetrahedral volume in Cartesian coordinates (SVG's y axis is down).
fn volume(p: &[Point], center: usize, neighbors: &[usize], target: usize) -> f64 {
    let mut v: Vec<[f64; 3]> = neighbors
        .iter()
        .map(|&i| {
            [
                p[i].0 - p[center].0,
                p[center].1 - p[i].1,
                if i == target { 1.0 } else { 0.0 },
            ]
        })
        .collect();
    if v.len() == 3 {
        v.push([0.0; 3]);
    } // virtual lone pair, last in index order
    let d = v[3];
    for a in &mut v[..3] {
        for k in 0..3 {
            a[k] -= d[k];
        }
    }
    let (a, b, c) = (v[0], v[1], v[2]);
    a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

pub(super) fn wedges<M: Molecule>(m: &M, p: &[Point]) -> Vec<Wedge> {
    let Some(classes) = m.chiral_classes() else {
        return vec![];
    };
    let mut used = FxHashSet::default();
    let mut result = vec![];
    for center in 0..m.len() {
        if !tetrahedral(classes[center]) {
            continue;
        }
        let mut ns: Vec<_> = m.atom_bonds()[center].iter().map(|b| b.target()).collect();
        ns.sort_unstable();
        if !(3..=4).contains(&ns.len()) {
            continue;
        }
        let mut candidates: Vec<_> = m.atom_bonds()[center]
            .iter()
            .filter(|b| {
                b.bond_order() == BondOrder::Single
                    && !used.contains(&(center.min(b.target()), center.max(b.target())))
            })
            .map(|b| b.target())
            .collect();
        candidates.sort_by_key(|&i| (tetrahedral(classes[i]), m.atom_bonds()[i].len(), i));
        if let Some((target, v)) = candidates
            .into_iter()
            .map(|i| (i, volume(p, center, &ns, i)))
            .find(|&(_, v)| v.abs() > 1e-5)
        {
            used.insert((center.min(target), center.max(target)));
            result.push(Wedge {
                center,
                target,
                solid: (v > 0.0) == (classes[center] == ChiralClass::TH(1)),
            });
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    const CASES: &[&str] = &[
        "F/C=C/F",
        "F/C=C\\F",
        "C/C=C/C",
        "C/C=C\\C",
        "C/C=C/C=C/C",
        "C/C=C\\C=C/C",
        "F/C(Cl)=C(/Br)I",
        "C[C@H](O)C(=O)O",
        "C[C@@H](O)C(=O)O",
        "N[C@@H](C)C(=O)O",
        "F[C@](Cl)(Br)I",
        "F[C@@](Cl)(Br)I",
        "C[C@H](O)[C@@H](C)O",
        "C[C@H]1CCCCO1",
        "C[C@@H]1CCCCO1",
        "[2H][C@](F)(Cl)Br",
        "C[S@](=O)CC",
        "C[S@@](=O)CC",
        "C1CC/C=C/CCC1",
        "C1CC/C=C\\CCC1",
        "O=C(O)[C@@H](O)C",
        "N[C@H](C)C(=O)O",
        "O[C@H]1CO[C@@H](O)[C@H](O)[C@@H]1O",
        "C[C@H]1CC[C@@H](C)[C@H](C)C1",
        "C/C=C(\\Cl)C",
        "F/C=C/C=C\\Cl",
        "[NH3+][C@@H](C)C(=O)[O-]",
        "C[P@](O)(=O)OC",
        // Cocaine: IUPHAR/BPS Guide to Pharmacology, ligand 2286.
        "COC(=O)[C@H]1[C@H](C[C@H]2N([C@@H]1CC2)C)OC(=O)c1ccccc1",
    ];
    #[test]
    fn alkene_geometry_matches_stored_references() {
        for s in CASES {
            let m = MolecularGraph::from_smiles_single(s).unwrap();
            let p = layout::coordinates(&m);
            for (a, b, ra, rb, sign) in double_bonds(&m, &visibility(&m)) {
                let axis = sub(p[b], p[a]);
                assert!(
                    cross(axis, sub(p[ra], p[a])) * cross(axis, sub(p[rb], p[b])) * sign > 1e-4,
                    "{s}"
                );
            }
        }
    }
    #[test]
    fn every_tetrahedral_center_is_drawn() {
        for &s in CASES {
            let m = MolecularGraph::from_smiles_single(s).unwrap();
            let count = m
                .chiral_classes()
                .unwrap_or(&[])
                .iter()
                .filter(|&&c| tetrahedral(c))
                .count();
            let svg = m.to_svg();
            assert_eq!(svg.matches("data-stereo=").count(), count, "{s}");
        }
    }

    #[test]
    fn enantiomers_have_opposite_wedges() {
        let a = MolecularGraph::from_smiles_single("C[C@H](O)C(=O)O").unwrap();
        let b = MolecularGraph::from_smiles_single("C[C@@H](O)C(=O)O").unwrap();
        let p = layout::coordinates(&a);
        assert_eq!(p, layout::coordinates(&b));
        let wa = wedges(&a, &p);
        let wb = wedges(&b, &p);
        assert_eq!(wa.len(), 1);
        assert_eq!(wb.len(), 1);
        assert_ne!(wa[0].solid, wb[0].solid);
        assert_eq!(wa[0].center, wb[0].center);
        assert_eq!(wa[0].target, wb[0].target);
    }

    #[test]
    #[ignore = "requires python3 and RDKit; independent stereochemistry oracle"]
    fn rdkit_depiction_oracle() {
        let mut sdf = String::new();
        for &s in CASES {
            for variant in 0..8 {
                let transform = variant % 4;
                let m = MolecularGraph::from_smiles_single(s).unwrap();
                let mut p = layout::depiction_with_cleanup(&m, variant >= 4).0;
                for q in &mut p {
                    if transform == 1 {
                        *q = (-q.1, q.0);
                    }
                    if transform == 2 {
                        *q = (-q.0, q.1);
                    }
                    if transform == 3 {
                        let (s, c) = 0.73_f64.sin_cos();
                        *q = (q.0 * c - q.1 * s, q.0 * s + q.1 * c);
                    }
                }
                let visible = visibility(&m);
                let wedges = wedges(&m, &p);
                let ids: Vec<_> = (0..m.len()).filter(|&i| visible[i]).collect();
                let mut map = vec![0; m.len()];
                for (k, &i) in ids.iter().enumerate() {
                    map[i] = k + 1;
                }
                let bonds: Vec<_> = ids
                    .iter()
                    .flat_map(|&i| {
                        let visible = &visible;
                        m.atom_bonds()[i]
                            .iter()
                            .filter(move |b| b.target() > i && visible[b.target()])
                            .map(move |b| (i, b.target(), b.bond_order()))
                    })
                    .collect();
                writeln!(
                    sdf,
                    "{s}\n  Oxidane          2D\n\n{:3}{:3}  0  0  0  0            999 V2000",
                    ids.len(),
                    bonds.len()
                )
                .unwrap();
                for &i in &ids {
                    writeln!(
                        sdf,
                        "{:10.4}{:10.4}{:10.4} {:3} 0  0  0  0  0  0  0  0  0  0  0  0",
                        p[i].0,
                        -p[i].1,
                        0.0,
                        m.get_atomic_symbol(i).unwrap()
                    )
                    .unwrap();
                }
                for (mut i, mut j, order) in bonds {
                    let mut stereo = 0;
                    if let Some(w) = wedges.iter().find(|w| {
                        (w.center == i && w.target == j) || (w.center == j && w.target == i)
                    }) {
                        i = w.center;
                        j = w.target;
                        stereo = if w.solid { 1 } else { 6 };
                    }
                    let order = if order == BondOrder::Aromatic {
                        4
                    } else {
                        order.order()
                    };
                    writeln!(
                        sdf,
                        "{:3}{:3}{:3}{:3}  0  0  0",
                        map[i], map[j], order, stereo
                    )
                    .unwrap();
                }
                for &i in &ids {
                    if let Some(iso) = m.get_isotope(i) {
                        writeln!(sdf, "M  ISO  1 {:3} {:3}", map[i], iso).unwrap();
                    }
                    if m.charges()[i] != 0 {
                        writeln!(sdf, "M  CHG  1 {:3} {:3}", map[i], m.charges()[i]).unwrap();
                    }
                }
                sdf.push_str("M  END\n$$$$\n");
            }
        }
        use std::process::{Command, Stdio};
        let mut child = Command::new("python3")
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/draw_oracle.py"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        std::io::Write::write_all(&mut child.stdin.take().unwrap(), sdf.as_bytes()).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        println!("{}", String::from_utf8_lossy(&out.stdout));
    }
}
