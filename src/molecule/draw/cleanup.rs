//! Optional deterministic 2D refinement, independent of 3D force fields.
use super::*;
type Point = (f64, f64);
fn sub(a: Point, b: Point) -> Point {
    (a.0 - b.0, a.1 - b.1)
}
fn length(a: Point) -> f64 {
    a.0.hypot(a.1)
}

struct Spring {
    a: usize,
    b: usize,
    length: f64,
    weight: f64,
}
struct Problem {
    atoms: Vec<usize>,
    edges: Vec<(usize, usize)>,
    springs: Vec<Spring>,
    radii: Vec<f64>,
}
impl Problem {
    fn new<M: Molecule>(m: &M, p: &[Point], visible: &[bool], rings: &[Vec<usize>]) -> Self {
        let atoms: Vec<_> = (0..m.len()).filter(|&i| visible[i]).collect();
        let edges: Vec<_> = atoms
            .iter()
            .flat_map(|&i| {
                m.atom_bonds()[i]
                    .iter()
                    .filter(move |b| b.target() > i && visible[b.target()])
                    .map(move |b| (i, b.target()))
            })
            .collect();
        let mut springs: Vec<_> = edges
            .iter()
            .map(|&(a, b)| Spring {
                a,
                b,
                length: 1.0,
                weight: 12.0,
            })
            .collect();
        // Keep aromatic templates rigid in shape, but free to translate/rotate.
        for ring in rings {
            if ring.iter().all(|&i| {
                m.atom_bonds()[i]
                    .iter()
                    .filter(|b| ring.contains(&b.target()))
                    .all(|b| b.bond_order() == BondOrder::Aromatic)
            }) {
                for k in 0..ring.len() {
                    for l in k + 1..ring.len() {
                        let (a, b) = (ring[k], ring[l]);
                        springs.push(Spring {
                            a,
                            b,
                            length: length(sub(p[a], p[b])),
                            weight: 35.0,
                        });
                    }
                }
            }
        }
        // Gentle angular preferences spread branches without forcing a bridged
        // cage to satisfy incompatible regular-polygon templates.
        for &center in &atoms {
            let ns: Vec<_> = m.atom_bonds()[center]
                .iter()
                .map(|b| b.target())
                .filter(|&i| visible[i])
                .collect();
            let target = if m.atom_bonds()[center]
                .iter()
                .any(|b| b.bond_order() == BondOrder::Triple)
            {
                2.0
            } else {
                3.0_f64.sqrt()
            };
            for i in 0..ns.len() {
                for j in i + 1..ns.len() {
                    springs.push(Spring {
                        a: ns[i],
                        b: ns[j],
                        length: target,
                        weight: 0.22,
                    });
                }
            }
        }
        // Side signs alone permit nearly collinear bonds that stereo readers
        // cannot resolve. Preserve the four-atom alkene depiction's shape.
        for (a, b, ra, rb, _) in stereo::double_bonds(m, visible) {
            let group = [a, b, ra, rb];
            for i in 0..4 {
                for j in i + 1..4 {
                    springs.push(Spring {
                        a: group[i],
                        b: group[j],
                        length: length(sub(p[group[i]], p[group[j]])),
                        weight: 20.0,
                    });
                }
            }
        }
        let radii = m
            .atomic_numbers()
            .iter()
            .map(|&z| if z == 6 { 0.28 } else { 0.40 })
            .collect();
        Self {
            atoms,
            edges,
            springs,
            radii,
        }
    }
    /// Leave already readable template layouts alone, including their exact
    /// bond angles. Refinement is useful only for clashes or strained bonds.
    fn needs_cleanup(&self, p: &[Point]) -> bool {
        if self
            .edges
            .iter()
            .any(|&(a, b)| (length(sub(p[a], p[b])) - 1.0).abs() > 0.15)
        {
            return true;
        }
        for (k, &a) in self.atoms.iter().enumerate() {
            for &b in &self.atoms[k + 1..] {
                if length(sub(p[a], p[b])) < (self.radii[a] + self.radii[b]).max(0.75) {
                    return true;
                }
            }
            for &(i, j) in &self.edges {
                if a == i || a == j {
                    continue;
                }
                let d = sub(p[j], p[i]);
                let v = sub(p[a], p[i]);
                let l2 = d.0 * d.0 + d.1 * d.1;
                if l2 < 1e-10 {
                    return true;
                }
                let t = (v.0 * d.0 + v.1 * d.1) / l2;
                if t > 0.0
                    && t < 1.0
                    && length((v.0 - t * d.0, v.1 - t * d.1)) < self.radii[a] + 0.12
                {
                    return true;
                }
            }
        }
        false
    }
    fn energy(&self, p: &[Point], g: &mut [Point]) -> f64 {
        g.fill((0.0, 0.0));
        let mut energy = 0.0;
        let mut pair = |a: usize, b: usize, target: f64, weight: f64, repulsive: bool| {
            let mut d = sub(p[a], p[b]);
            let mut r = length(d);
            if r < 1e-8 {
                // Index-derived perturbation breaks exact coincidences.
                let t = ((a * 37 + b * 17) % 360) as f64 * std::f64::consts::PI / 180.0;
                d = (t.cos() * 1e-6, t.sin() * 1e-6);
                r = 1e-6;
            }
            let error = if repulsive {
                (r - target).min(0.0)
            } else {
                r - target
            };
            energy += weight * error * error;
            let f = 2.0 * weight * error / r;
            g[a].0 += d.0 * f;
            g[a].1 += d.1 * f;
            g[b].0 -= d.0 * f;
            g[b].1 -= d.1 * f;
        };
        for s in &self.springs {
            pair(s.a, s.b, s.length, s.weight, false);
        }
        for (k, &a) in self.atoms.iter().enumerate() {
            for &b in &self.atoms[k + 1..] {
                pair(a, b, (self.radii[a] + self.radii[b]).max(0.75), 6.0, true);
            }
        }
        // Push labels/vertices off unrelated bond segments. Distribute the
        // opposite force over both endpoints to avoid bending only branches.
        for &i in &self.atoms {
            for &(a, b) in &self.edges {
                if i == a || i == b {
                    continue;
                }
                let axis = sub(p[b], p[a]);
                let l2 = axis.0 * axis.0 + axis.1 * axis.1;
                if l2 < 1e-10 {
                    continue;
                }
                let v = sub(p[i], p[a]);
                let t = ((v.0 * axis.0 + v.1 * axis.1) / l2).clamp(0.0, 1.0);
                if t <= 0.0 || t >= 1.0 {
                    continue;
                }
                let mut d = (v.0 - t * axis.0, v.1 - t * axis.1);
                let mut r = length(d);
                let target = self.radii[i] + 0.12;
                if r >= target {
                    continue;
                }
                if r < 1e-8 {
                    let len = l2.sqrt();
                    d = (-axis.1 / len * 1e-6, axis.0 / len * 1e-6);
                    r = 1e-6;
                }
                let error = r - target;
                energy += 5.0 * error * error;
                let f = 10.0 * error / r;
                let force = (d.0 * f, d.1 * f);
                g[i].0 += force.0;
                g[i].1 += force.1;
                g[a].0 -= force.0 * (1.0 - t);
                g[a].1 -= force.1 * (1.0 - t);
                g[b].0 -= force.0 * t;
                g[b].1 -= force.1 * t;
            }
        }
        energy
    }
}

/// Bounded, monotone descent. Stereo is checked on every accepted step;
/// wedges are derived afresh from the final coordinates by the renderer.
pub(super) fn refine<M: Molecule>(m: &M, p: &mut [Point], visible: &[bool], rings: &[Vec<usize>]) {
    let problem = Problem::new(m, p, visible, rings);
    if problem.atoms.len() < 3 || !problem.needs_cleanup(p) {
        return;
    }
    let constraints = stereo::double_bonds(m, visible);
    let expected_wedges = stereo::wedges(m, p).len();
    let valid = |q: &[Point]| {
        constraints.iter().all(|&(a, b, ra, rb, sign)| {
            let d = sub(q[b], q[a]);
            let va = sub(q[ra], q[a]);
            let vb = sub(q[rb], q[b]);
            {
                let ca = d.0 * va.1 - d.1 * va.0;
                let cb = d.0 * vb.1 - d.1 * vb.0;
                ca * cb * sign > 1e-5
                    && ca.abs() > 0.25 * length(d) * length(va)
                    && cb.abs() > 0.25 * length(d) * length(vb)
            }
        }) && stereo::wedges(m, q).len() == expected_wedges
    };
    let mut gradient = vec![(0.0, 0.0); p.len()];
    let mut trial_gradient = gradient.clone();
    let mut trial = p.to_vec();
    let mut energy = problem.energy(p, &mut gradient);
    for _ in 0..200 {
        let norm = gradient.iter().map(|&v| v.0 * v.0 + v.1 * v.1).sum::<f64>();
        if norm < 1e-8 {
            break;
        }
        let max = gradient.iter().map(|&v| length(v)).fold(0.0, f64::max);
        let mut step = 0.025_f64.min(0.12 / max);
        let mut accepted = false;
        for _ in 0..12 {
            for &i in &problem.atoms {
                trial[i] = (p[i].0 - step * gradient[i].0, p[i].1 - step * gradient[i].1);
            }
            let next = problem.energy(&trial, &mut trial_gradient);
            if next < energy - 1e-5 * step * norm && valid(&trial) {
                p.copy_from_slice(&trial);
                std::mem::swap(&mut gradient, &mut trial_gradient);
                energy = next;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    const COCAINE: &str = "COC(=O)[C@H]1[C@H](C[C@H]2N([C@@H]1CC2)C)OC(=O)c1ccccc1";
    fn options() -> DrawOptions {
        DrawOptions {
            cleanup: true,
            ..Default::default()
        }
    }
    #[test]
    fn cleanup_keeps_bridged_atoms_separate() {
        let m = MolecularGraph::from_smiles_single("C1CC2CCC1C2").unwrap();
        let visible = visibility(&m);
        let minimum = |p: &[Point]| {
            let mut min = f64::INFINITY;
            for i in 0..m.len() {
                for j in i + 1..m.len() {
                    if visible[i] && visible[j] {
                        min = min.min(length(sub(p[i], p[j])));
                    }
                }
            }
            min
        };
        let before = m.get_2d_coordinates();
        let after = m.get_2d_coordinates_with_options(&options());
        assert!(minimum(&before) > 0.5);
        assert!(minimum(&after) > 0.5);
    }
    #[test]
    fn cocaine_cleanup_is_deterministic_and_preserves_graph() {
        let m = MolecularGraph::from_smiles_single(COCAINE).unwrap();
        let original = m.to_smiles();
        let p = m.get_2d_coordinates_with_options(&options());
        assert_eq!(p, m.get_2d_coordinates_with_options(&options()));
        assert_eq!(original, m.to_smiles());
        assert_eq!(p.len(), m.len());
        assert!(p.iter().all(|&(x, y)| x.is_finite() && y.is_finite()));
        assert_eq!(stereo::wedges(&m, &p).len(), 4);
        assert_ne!(p, m.get_2d_coordinates());
        let visible = visibility(&m);
        for i in 0..m.len() {
            if !visible[i] {
                continue;
            }
            for b in &m.atom_bonds()[i] {
                let j = b.target();
                if visible[j] {
                    let d = length(sub(p[i], p[j]));
                    assert!(d > 0.65 && d < 1.35, "bond {i}-{j}: {d}");
                }
            }
        }
    }
    #[test]
    fn tidy_layouts_and_default_are_unchanged() {
        for s in ["", "[Na+]", "CCO", "C1CCC2(CC1)CCCC2", "c1ccc2ccccc2c1"] {
            let m = MolecularGraph::from_smiles_single(s).unwrap();
            let before = m.get_2d_coordinates();
            assert_eq!(
                before,
                m.get_2d_coordinates_with_options(&DrawOptions::default())
            );
            assert_eq!(before, m.get_2d_coordinates_with_options(&options()), "{s}");
            assert_eq!(m.to_svg(), m.to_svg_with_options(&DrawOptions::default()));
        }
    }
}
