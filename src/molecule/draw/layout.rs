use super::*;
use crate::molecule::topology::smallest_rings;
use rustc_hash::FxHashMap;
use std::{collections::VecDeque, f64::consts::PI};

#[derive(Default)]
struct Grid(FxHashMap<(i64, i64), Vec<(usize, Point)>>);
impl Grid {
    fn cell(p: Point) -> (i64, i64) {
        ((p.0 / 1.2).floor() as i64, (p.1 / 1.2).floor() as i64)
    }
    fn insert(&mut self, i: usize, p: Point) {
        self.0.entry(Self::cell(p)).or_default().push((i, p));
    }
    fn penalty(&self, p: Point, radius: f64, exclude: impl Fn(usize) -> bool) -> f64 {
        let (x, y) = Self::cell(p);
        let mut score = 0.0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(items) = self.0.get(&(x + dx, y + dy)) {
                    for &(i, q) in items {
                        if !exclude(i) {
                            score += (radius - distance(p, q)).max(0.0).powi(2);
                        }
                    }
                }
            }
        }
        score
    }
}

type Point = (f64, f64);
fn distance(a: Point, b: Point) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// Deterministic unit-bond layout. Ring templates are placed before branches.
pub(super) fn coordinates<M: Molecule>(m: &M) -> Vec<Point> {
    depiction(m).0
}

pub(super) fn depiction<M: Molecule>(m: &M) -> (Vec<Point>, Vec<Vec<usize>>) {
    depiction_with_cleanup(m, false)
}

pub(super) fn depiction_with_cleanup<M: Molecule>(
    m: &M,
    cleanup: bool,
) -> (Vec<Point>, Vec<Vec<usize>>) {
    let n = m.len();
    let mut p = vec![(0.0, 0.0); n];
    let mut placed = vec![false; n];
    let visible = visibility(m);
    // A linear leaf peel avoids ring searches altogether for acyclic graphs.
    let mut degree: Vec<_> = m.atom_bonds().iter().map(|b| b.len()).collect();
    let mut leaves: Vec<_> = (0..n).filter(|&i| degree[i] < 2).collect();
    let mut head = 0;
    while head < leaves.len() {
        let i = leaves[head];
        head += 1;
        degree[i] = 0;
        for b in &m.atom_bonds()[i] {
            let j = b.target();
            if degree[j] > 0 {
                degree[j] -= 1;
                if degree[j] == 1 {
                    leaves.push(j);
                }
            }
        }
    }
    let mut rings = if degree.iter().any(|&d| d > 1) {
        smallest_rings(m)
    } else {
        vec![]
    };
    rings.sort_by_key(|r| r.len());
    let mut done = vec![false; rings.len()];
    let mut offset = 0.0;
    let mut seen = vec![false; n];
    for seed in (0..n).filter(|&i| visible[i]) {
        if placed[seed] {
            continue;
        }
        let mut component = vec![seed];
        seen[seed] = true;
        let mut cursor = 0;
        while cursor < component.len() {
            let i = component[cursor];
            cursor += 1;
            for b in &m.atom_bonds()[i] {
                let j = b.target();
                if visible[j] && !seen[j] {
                    seen[j] = true;
                    component.push(j);
                }
            }
        }
        let root = rings
            .iter()
            .find(|r| seen[r[0]] && !placed[r[0]])
            .map_or(seed, |r| r[0]);
        p[root] = (offset, 0.0);
        placed[root] = true;
        let mut grid = Grid::default();
        grid.insert(root, p[root]);
        let mut frontier = VecDeque::from([root]);
        loop {
            let mut progress = false;
            for (ri, ring) in rings.iter().enumerate() {
                if done[ri] || !ring.iter().any(|&i| placed[i]) {
                    continue;
                }
                let anchors: Vec<_> = (0..ring.len()).filter(|&k| placed[ring[k]]).collect();
                let shared_edge = anchors.len() == 2
                    && ((anchors[0] + 1) % ring.len() == anchors[1]
                        || (anchors[1] + 1) % ring.len() == anchors[0]);
                let bridged = anchors.len() > 2 || (anchors.len() == 2 && !shared_edge);
                let count = ring.len();
                let radius = 0.5 / (PI / count as f64).sin();
                let anchor = ring.iter().position(|&i| placed[i]).unwrap();
                let a = ring[anchor];
                let template: Vec<_> = (0..count)
                    .map(|k| {
                        let t = 2.0 * PI * (k as f64 - anchor as f64) / count as f64;
                        (radius * t.cos() - radius, radius * t.sin())
                    })
                    .collect();
                let mut best = (f64::INFINITY, Vec::new());
                for flip in [-1.0, 1.0] {
                    for step in 0..24 {
                        let second = ring
                            .iter()
                            .enumerate()
                            .find(|&(k, i)| k != anchor && placed[*i]);
                        if second.is_some() && step > 0 {
                            break;
                        }
                        let angle = second.map_or(step as f64 * PI / 12.0, |(k, &i)| {
                            (p[i].1 - p[a].1).atan2(p[i].0 - p[a].0)
                                - (template[k].1 * flip).atan2(template[k].0)
                        });
                        let (s, c) = angle.sin_cos();
                        let trial: Vec<_> = template
                            .iter()
                            .map(|&(x, y)| {
                                (p[a].0 + x * c - y * flip * s, p[a].1 + x * s + y * flip * c)
                            })
                            .collect();
                        let mut score = 0.0;
                        for (k, &i) in ring.iter().enumerate() {
                            if placed[i] {
                                score += 1000.0 * distance(trial[k], p[i]).powi(2);
                            } else {
                                score += grid.penalty(trial[k], 1.15, |j| ring.contains(&j)) * 10.0;
                            }
                        }
                        if score < best.0 {
                            best = (score, trial);
                        }
                    }
                }
                if bridged
                    && ring.iter().enumerate().any(|(k, &i)| {
                        !placed[i] && grid.penalty(best.1[k], 0.05, |j| ring.contains(&j)) > 1e-8
                    })
                {
                    // Preserve valid polygon placements. When fitting shared
                    // paths collapses distinct bridges, route just the missing
                    // paths between their fixed endpoints instead.
                    place_bridges(
                        m,
                        ring,
                        &anchors,
                        &component,
                        &mut p,
                        &mut placed,
                        &mut grid,
                        &mut frontier,
                    );
                    done[ri] = true;
                    progress = true;
                    continue;
                }
                for (k, &i) in ring.iter().enumerate() {
                    if !placed[i] {
                        p[i] = best.1[k];
                        placed[i] = true;
                        grid.insert(i, p[i]);
                        frontier.push_back(i);
                    }
                }
                done[ri] = true;
                progress = true;
            }
            if progress {
                continue;
            }
            // Grow a single branch, giving newly reached rings priority.
            let edge = loop {
                let Some(&i) = frontier.front() else {
                    break None;
                };
                if let Some(j) = m.atom_bonds()[i]
                    .iter()
                    .map(|b| b.target())
                    .find(|&j| visible[j] && !placed[j])
                {
                    break Some((i, j));
                }
                frontier.pop_front();
            };
            let Some((i, j)) = edge else {
                break;
            };
            let neighbors: Vec<_> = m.atom_bonds()[i]
                .iter()
                .filter(|b| visible[b.target()] && placed[b.target()])
                .collect();
            let linear = m.atom_bonds()[i]
                .iter()
                .any(|b| b.bond_order() == BondOrder::Triple);
            // Fused five-membered rings leave ring atoms off the 30-degree
            // grid, so also offer directions relative to the placed bonds:
            // the exterior bisector and 120 degrees from each neighbour.
            let mut directions: Vec<_> = (0..12).map(|step| step as f64 * PI / 6.0).collect();
            let (mut sx, mut sy) = (0.0, 0.0);
            for b in &neighbors {
                let v = p[b.target()];
                let phi = (v.1 - p[i].1).atan2(v.0 - p[i].0);
                directions.extend([phi + 2.0 * PI / 3.0, phi - 2.0 * PI / 3.0]);
                let d = distance(v, p[i]).max(1e-6);
                sx += (v.0 - p[i].0) / d;
                sy += (v.1 - p[i].1) / d;
            }
            if neighbors.len() > 1 && sx.hypot(sy) > 1e-6 {
                directions.push((-sy).atan2(-sx));
            }
            let mut best = (f64::INFINITY, (0.0, 0.0));
            for (step, &t) in directions.iter().enumerate() {
                let q = (p[i].0 + t.cos(), p[i].1 + t.sin());
                let mut score = step as f64 * 1e-6;
                for b in &neighbors {
                    let v = p[b.target()];
                    let cos = ((v.0 - p[i].0) * t.cos() + (v.1 - p[i].1) * t.sin())
                        / distance(v, p[i]).max(1e-6);
                    score += (cos - if linear { -1.0 } else { -0.5 }).powi(2) * 2.0;
                }
                score += grid.penalty(q, 1.1, |k| k == i) * 25.0;
                // Prefer horizontal zigzags for the first bond.
                if neighbors.is_empty() {
                    score += (t - PI / 6.0).abs() * 0.001;
                }
                if score < best.0 {
                    best = (score, q);
                }
            }
            p[j] = best.1;
            placed[j] = true;
            grid.insert(j, p[j]);
            frontier.push_back(j);
        }
        let min = component
            .iter()
            .map(|&i| p[i].0)
            .fold(f64::INFINITY, f64::min);
        let max = component
            .iter()
            .map(|&i| p[i].0)
            .fold(f64::NEG_INFINITY, f64::max);
        for &i in &component {
            p[i].0 += offset - min;
        }
        offset += max - min + 2.5;
    }
    stereo::enforce_double_bonds(m, &mut p, &visible);
    if cleanup {
        super::cleanup::refine(m, &mut p, &visible, &rings);
    }
    orient_components(m, &mut p, &visible);
    // Hidden ordinary hydrogens retain finite coordinates in atom-index order.
    for i in 0..n {
        if !visible[i] {
            if let Some(b) = m.atom_bonds()[i].first() {
                p[i] = p[b.target()];
            }
        }
    }
    (p, rings)
}

/// Align each component's principal axis with the page, retaining familiar
/// 30-degree bond directions. Rotation never reflects stereochemistry.
fn orient_components<M: Molecule>(m: &M, p: &mut [Point], visible: &[bool]) {
    let mut seen = vec![false; m.len()];
    let mut offset = 0.0;
    for seed in 0..m.len() {
        if seen[seed] || !visible[seed] {
            continue;
        }
        let mut ids = vec![seed];
        seen[seed] = true;
        let mut head = 0;
        while head < ids.len() {
            let i = ids[head];
            head += 1;
            for b in &m.atom_bonds()[i] {
                let j = b.target();
                if visible[j] && !seen[j] {
                    seen[j] = true;
                    ids.push(j);
                }
            }
        }
        let center = ids.iter().fold((0.0, 0.0), |(x, y), &i| {
            (x + p[i].0 / ids.len() as f64, y + p[i].1 / ids.len() as f64)
        });
        let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
        for &i in &ids {
            let (x, y) = (p[i].0 - center.0, p[i].1 - center.1);
            xx += x * x;
            xy += x * y;
            yy += y * y;
        }
        let mut angle = if (xx - yy).hypot(2.0 * xy) < 1e-7 {
            0.0
        } else {
            -0.5 * (2.0 * xy).atan2(xx - yy)
        };
        let snapped = (angle / (PI / 6.0)).round() * (PI / 6.0);
        angle = snapped;
        let (s, c) = angle.sin_cos();
        for &i in &ids {
            let (x, y) = (p[i].0 - center.0, p[i].1 - center.1);
            p[i] = (x * c - y * s, x * s + y * c);
        }
        let min = ids.iter().map(|&i| p[i].0).fold(f64::INFINITY, f64::min);
        let max = ids
            .iter()
            .map(|&i| p[i].0)
            .fold(f64::NEG_INFINITY, f64::max);
        for &i in &ids {
            p[i].0 += offset - min;
        }
        offset += max - min + 2.5;
    }
}

fn segment_distance(p: Point, a: Point, b: Point) -> f64 {
    let d = (b.0 - a.0, b.1 - a.1);
    let l2 = d.0 * d.0 + d.1 * d.1;
    if l2 < 1e-12 {
        return distance(p, a);
    }
    let t = (((p.0 - a.0) * d.0 + (p.1 - a.1) * d.1) / l2).clamp(0.0, 1.0);
    distance(p, (a.0 + t * d.0, a.1 + t * d.1))
}
fn crossing(a: Point, b: Point, c: Point, d: Point) -> bool {
    let side = |p: Point, q: Point, r: Point| (q.0 - p.0) * (r.1 - p.1) - (q.1 - p.1) * (r.0 - p.0);
    side(a, b, c) * side(a, b, d) < -1e-8 && side(c, d, a) * side(c, d, b) < -1e-8
}

#[allow(clippy::too_many_arguments)]
fn place_bridges<M: Molecule>(
    m: &M,
    ring: &[usize],
    anchors: &[usize],
    component: &[usize],
    p: &mut [Point],
    placed: &mut [bool],
    grid: &mut Grid,
    frontier: &mut VecDeque<usize>,
) {
    for k in 0..anchors.len() {
        let start = anchors[k];
        let end = anchors[(k + 1) % anchors.len()];
        let mut path = vec![ring[start]];
        let mut at = (start + 1) % ring.len();
        while at != end {
            path.push(ring[at]);
            at = (at + 1) % ring.len();
        }
        path.push(ring[end]);
        if path.len() == 2 {
            continue;
        }
        let a = p[path[0]];
        let b = p[*path.last().unwrap()];
        let d = (b.0 - a.0, b.1 - a.1);
        let len = distance(a, b).max(1e-8);
        let normal = (-d.1 / len, d.0 / len);
        let steps = (path.len() - 1) as f64;
        let edges: Vec<_> = component
            .iter()
            .filter(|&&i| placed[i])
            .flat_map(|&i| m.atom_bonds()[i].iter().map(move |b| (i, b.target())))
            .filter(|&(i, j)| j > i && placed[j])
            .collect();
        let mut best = (f64::INFINITY, Vec::new());
        // Several deterministic arc depths give short bridges room inside a
        // ring and longer bridges an alternative route around its exterior.
        for sign in [-1.0, 1.0] {
            for depth in [0.35, 0.6, 0.9, 1.2, 1.6, 2.0] {
                let height = sign * depth * steps.sqrt();
                let trial: Vec<_> = (0..path.len())
                    .map(|j| {
                        let t = j as f64 / steps;
                        let h = height * (PI * t).sin();
                        (a.0 + d.0 * t + normal.0 * h, a.1 + d.1 * t + normal.1 * h)
                    })
                    .collect();
                let mut score = 0.0;
                for j in 1..path.len() - 1 {
                    score += grid.penalty(trial[j], 0.75, |_| false) * 150.0;
                }
                for j in 0..path.len() - 1 {
                    let (u, v) = (trial[j], trial[j + 1]);
                    score += 4.0 * (distance(u, v) - 1.0).powi(2);
                    for &i in component {
                        if placed[i] && i != path[j] && i != path[j + 1] {
                            score += 100.0 * (0.3 - segment_distance(p[i], u, v)).max(0.0).powi(2);
                        }
                    }
                    for &(x, y) in &edges {
                        if x == path[j] || x == path[j + 1] || y == path[j] || y == path[j + 1] {
                            continue;
                        }
                        if crossing(u, v, p[x], p[y]) {
                            score += 25.0;
                        }
                        if j > 0 {
                            score +=
                                50.0 * (0.3 - segment_distance(u, p[x], p[y])).max(0.0).powi(2);
                        }
                    }
                }
                if score < best.0 {
                    best = (score, trial);
                }
            }
        }
        for j in 1..path.len() - 1 {
            let i = path[j];
            p[i] = best.1[j];
            placed[i] = true;
            grid.insert(i, p[i]);
            frontier.push_back(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn default_bridges_have_distinct_atoms_and_bonds() {
        for smiles in [
            "C1CC2CCC1C2",
            "C1C2CCC1CC2",
            "C1CC2CCC1CC2",
            "C1CCC2CCCC1CC2",
        ] {
            let m = MolecularGraph::from_smiles_single(smiles).unwrap();
            let p = m.get_2d_coordinates();
            let visible = visibility(&m);
            assert_eq!(p, m.get_2d_coordinates(), "{smiles}");
            for i in 0..m.len() {
                if !visible[i] {
                    continue;
                }
                for j in i + 1..m.len() {
                    if visible[j] {
                        assert!(
                            distance(p[i], p[j]) > 0.5,
                            "{smiles}: overlapping atoms {i}-{j}"
                        );
                    }
                }
                for a in 0..m.len() {
                    if !visible[a] || a == i {
                        continue;
                    }
                    for b in &m.atom_bonds()[a] {
                        let j = b.target();
                        if j <= a || j == i || !visible[j] {
                            continue;
                        }
                        assert!(
                            segment_distance(p[i], p[a], p[j]) > 0.12,
                            "{smiles}: atom {i} obscures bond {a}-{j}"
                        );
                    }
                }
            }
            let edges: Vec<_> = (0..m.len())
                .filter(|&i| visible[i])
                .flat_map(|i| m.atom_bonds()[i].iter().map(move |b| (i, b.target())))
                .filter(|&(i, j)| j > i && visible[j])
                .collect();
            for (k, &(a, b)) in edges.iter().enumerate() {
                for &(c, d) in &edges[k + 1..] {
                    assert!(
                        !crossing(p[a], p[b], p[c], p[d]),
                        "{smiles}: crossing bonds {a}-{b} and {c}-{d}"
                    );
                }
            }
            assert_eq!(m.to_svg().matches("data-bond=").count(), edges.len());
        }
    }

    #[test]
    fn substituents_on_fused_six_ring_bisect_the_exterior_angle() {
        // Caffeine: the fused imidazole tilts the six-ring off the 30-degree grid.
        let m = MolecularGraph::from_smiles_single("Cn1cnc2c1c(=O)n(C)c(=O)n2C").unwrap();
        let p = m.get_2d_coordinates();
        for i in [6, 8, 10, 12] {
            let mut angles: Vec<_> = m.atom_bonds()[i]
                .iter()
                .map(|b| {
                    let q = p[b.target()];
                    (q.1 - p[i].1).atan2(q.0 - p[i].0)
                })
                .collect();
            angles.sort_by(f64::total_cmp);
            for k in 0..3 {
                let gap = (angles[(k + 1) % 3] - angles[k]).rem_euclid(2.0 * PI);
                assert!((gap - 2.0 * PI / 3.0).abs() < 1e-6, "atom {i}: {gap}");
            }
        }
    }
}
