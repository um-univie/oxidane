//! Deterministic, dependency-free SVG depiction with skeletal atom labels.
//!
//! Ordinary terminal hydrogens are folded into labels; hydrogens on tetrahedral
//! stereocenters stay visible for unambiguous wedges. Coordination bonds are
//! dashed because the graph does not store donor direction. Coordinates use
//! unit bond lengths and retain atom indexing; hidden hydrogens share their
//! parent's position. Tetrahedral tags become solid/hashed wedges, and stored
//! cis/trans tags constrain alkene geometry. Extended stereo classes (allene,
//! square planar, etc.) are not depicted. Complex bridged cages and macrocycles
//! use a best-effort projection.
use crate::molecule::{base::Molecule, bond::BondOrder};
use std::fmt::Write;
mod cleanup;
mod layout;
mod stereo;

/// SVG layout and presentation settings. Invalid numeric values fall back to defaults.
#[derive(Debug, Clone)]
pub struct DrawOptions {
    pub width: u32,
    pub height: u32,
    /// Preferred bond length in pixels; shrinks to fit the viewport.
    pub bond_length: f64,
    pub padding: f64,
    pub font_size: f64,
    pub stroke_width: f64,
    pub atom_colors: bool,
    /// Refine crowded layouts with a bounded 2D optimization. Off by default.
    pub cleanup: bool,
}
impl Default for DrawOptions {
    fn default() -> Self {
        Self {
            width: 400,
            height: 300,
            bond_length: 36.0,
            padding: 24.0,
            font_size: 16.0,
            stroke_width: 1.8,
            atom_colors: true,
            cleanup: false,
        }
    }
}

pub trait Draw<M: Molecule> {
    /// Render a centered skeletal SVG with the default style.
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();
    /// let svg = molecule.to_svg();
    /// assert!(svg.starts_with("<svg"));
    /// ```
    fn to_svg(&self) -> String
    where
        Self: Molecule + Sized,
    {
        self.to_svg_with_options(&DrawOptions::default())
    }
    /// Render with custom dimensions, typography and element coloring.
    fn to_svg_with_options(&self, options: &DrawOptions) -> String
    where
        Self: Molecule + Sized,
    {
        render(self, options)
    }
    /// Generate repeatable unit-bond coordinates in atom-index order.
    fn get_2d_coordinates(&self) -> Vec<(f64, f64)>
    where
        Self: Molecule + Sized,
    {
        layout::coordinates(self)
    }

    /// Generate coordinates with the same optional cleanup used by SVG rendering.
    fn get_2d_coordinates_with_options(&self, options: &DrawOptions) -> Vec<(f64, f64)>
    where
        Self: Molecule + Sized,
    {
        layout::depiction_with_cleanup(self, options.cleanup).0
    }

    fn scale_coordinate(
        &self,
        (x, y): (f64, f64),
        (min_x, min_y): (f64, f64),
        scale_factor: f64,
        margin: i32,
    ) -> (i32, i32) {
        let x = ((x - min_x) * scale_factor) as i32 + margin;
        let y = ((y - min_y) * scale_factor) as i32 + margin;
        (x, y)
    }

    fn get_parallel_offset(&self, x1: i32, y1: i32, x2: i32, y2: i32, distance: f64) -> (i32, i32) {
        let dx = (x2 - x1) as f64;
        let dy = (y2 - y1) as f64;
        let length = (dx * dx + dy * dy).sqrt();

        if length == 0.0 {
            return (0, 0);
        }

        let ox = -dy * distance / length;
        let oy = dx * distance / length;

        (ox as i32, oy as i32)
    }
    /// Checks if the molecule contains any cycles
    fn is_cyclic(&self) -> bool
    where
        Self: Molecule,
    {
        let mut visited = vec![false; self.len()];
        let mut stack = vec![false; self.len()];

        for start in 0..self.len() {
            if !visited[start]
                && self.has_cycle_dfs(start, None, &mut visited, &mut stack) {
                    return true;
                }
        }
        false
    }

    /// Helper function for cycle detection using DFS
    fn has_cycle_dfs(
        &self,
        vertex: usize,
        parent: Option<usize>,
        visited: &mut [bool],
        stack: &mut [bool],
    ) -> bool
    where
        Self: Molecule,
    {
        visited[vertex] = true;
        stack[vertex] = true;

        if let Some(bonds) = self.get_atom_bonds(vertex) {
            for bond in bonds {
                let neighbor = bond.target();
                if !visited[neighbor] {
                    if self.has_cycle_dfs(neighbor, Some(vertex), visited, stack) {
                        return true;
                    }
                } else if stack[neighbor] && Some(neighbor) != parent {
                    return true;
                }
            }
        }

        stack[vertex] = false;
        false
    }

    /// Adjusts the layout for cyclic molecules to make them more regular
    fn adjust_cyclic_layout(&self, positions: &mut [(f64, f64)])
    where
        Self: Molecule,
    {
        let cycles = self.find_cycles();

        for cycle in cycles {
            if cycle.len() >= 3 {
                let center = cycle.iter().fold((0.0, 0.0), |acc, &idx| {
                    (acc.0 + positions[idx].0, acc.1 + positions[idx].1)
                });
                let center = (center.0 / cycle.len() as f64, center.1 / cycle.len() as f64);

                // Arrange atoms in a regular polygon
                let radius = 0.3; // Adjust this value to change the size of cycles
                for (i, &idx) in cycle.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * i as f64 / cycle.len() as f64;
                    positions[idx] = (
                        center.0 + radius * angle.cos(),
                        center.1 + radius * angle.sin(),
                    );
                }
            }
        }
    }

    /// Finds all cycles in the molecule
    fn find_cycles(&self) -> Vec<Vec<usize>>
    where
        Self: Molecule,
    {
        let mut cycles = Vec::new();
        let mut visited = vec![false; self.len()];
        let mut path = Vec::new();

        for start in 0..self.len() {
            if !visited[start] {
                self.find_cycles_dfs(start, None, &mut visited, &mut path, &mut cycles);
            }
        }

        cycles
    }

    /// Helper function for cycle detection using DFS
    fn find_cycles_dfs(
        &self,
        vertex: usize,
        parent: Option<usize>,
        visited: &mut [bool],
        path: &mut Vec<usize>,
        cycles: &mut Vec<Vec<usize>>,
    ) where
        Self: Molecule,
    {
        visited[vertex] = true;
        path.push(vertex);

        if let Some(bonds) = self.get_atom_bonds(vertex) {
            for bond in bonds {
                let neighbor = bond.target();
                if !visited[neighbor] {
                    self.find_cycles_dfs(neighbor, Some(vertex), visited, path, cycles);
                } else if Some(neighbor) != parent && path.contains(&neighbor) {
                    // Found a cycle
                    if let Some(start_idx) = path.iter().position(|&x| x == neighbor) {
                        let cycle: Vec<_> = path[start_idx..].to_vec();
                        if !cycles.contains(&cycle) && cycle.len() > 2 {
                            cycles.push(cycle);
                        }
                    }
                }
            }
        }

        path.pop();
    }
}
impl<M: Molecule> Draw<M> for M {}

fn visibility<M: Molecule>(m: &M) -> Vec<bool> {
    (0..m.len())
        .map(|i| {
            !(m.atomic_numbers()[i] == 1
                && m.charges()[i] == 0
                && m.get_isotope(i).is_none()
                && m.radical_states()[i] == 0
                && m.atom_bonds()[i].len() == 1
                && m.atomic_numbers()[m.atom_bonds()[i][0].target()] != 1
                && m.atom_bonds()[i][0].bond_order() == BondOrder::Single
                && !m.chiral_classes().is_some_and(|classes| {
                    stereo::tetrahedral(classes[m.atom_bonds()[i][0].target()])
                }))
        })
        .collect()
}
fn positive(value: f64, default: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        default
    }
}
fn color(z: u8) -> &'static str {
    match z {
        7 => "#2563eb",
        8 => "#dc2626",
        9 | 17 => "#16803c",
        15 => "#c26913",
        16 => "#a67b08",
        35 => "#a33c24",
        53 => "#7c3aed",
        _ => "#243247",
    }
}
fn render<M: Molecule>(m: &M, o: &DrawOptions) -> String {
    let width = o.width.max(1) as f64;
    let height = o.height.max(1) as f64;
    let bond = positive(o.bond_length, 36.0);
    let font = positive(o.font_size, 16.0);
    let stroke = positive(o.stroke_width, 1.8);
    let padding = if o.padding.is_finite() {
        o.padding.max(0.0)
    } else {
        24.0
    };
    let (p, rings) = layout::depiction_with_cleanup(m, o.cleanup);
    let visible = visibility(m);
    let mut labels = vec![String::new(); m.len()];
    let mut label_widths = vec![0.0; m.len()];
    for i in 0..m.len() {
        if !visible[i] {
            continue;
        }
        let isotope = m.get_isotope(i);
        let charge = m.charges()[i];
        let h = m.atom_bonds()[i]
            .iter()
            .filter(|b| !visible[b.target()])
            .count();
        let isolated = m.atom_bonds()[i].iter().all(|b| !visible[b.target()]);
        if m.atomic_numbers()[i] == 6
            && isotope.is_none()
            && charge == 0
            && m.radical_states()[i] == 0
            && !isolated
        {
            continue;
        }
        label_widths[i] = m.get_atomic_symbol(i).unwrap_or("*").len() as f64 * 0.62
            + isotope.map_or(0.0, |v| v.to_string().len() as f64 * 0.43)
            + if h > 0 {
                0.62 + if h > 1 { 0.43 } else { 0.0 }
            } else {
                0.0
            }
            + if charge != 0 {
                0.43 + if charge.unsigned_abs() > 1 { 0.43 } else { 0.0 }
            } else {
                0.0
            }
            + if m.radical_states()[i] > 0 { 0.43 } else { 0.0 };
        let label = &mut labels[i];
        if let Some(mass) = isotope {
            write!(
                label,
                "<tspan baseline-shift=\"super\" font-size=\"70%\">{mass}</tspan>"
            )
            .unwrap();
        }
        label.push_str(m.get_atomic_symbol(i).unwrap_or("*"));
        if h > 0 {
            label.push('H');
            if h > 1 {
                write!(
                    label,
                    "<tspan baseline-shift=\"sub\" font-size=\"70%\">{h}</tspan>"
                )
                .unwrap();
            }
        }
        if charge != 0 {
            label.push_str("<tspan baseline-shift=\"super\" font-size=\"70%\">");
            if charge.unsigned_abs() > 1 {
                write!(label, "{}", charge.unsigned_abs()).unwrap();
            }
            label.push_str(if charge > 0 { "+" } else { "−" });
            label.push_str("</tspan>");
        }
        if m.radical_states()[i] > 0 {
            label.push_str("<tspan baseline-shift=\"super\" font-size=\"70%\">•</tspan>");
        }
    }
    let mut min = (f64::INFINITY, f64::INFINITY);
    let mut max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (i, &(x, y)) in p.iter().enumerate() {
        if visible[i] {
            min.0 = min.0.min(x);
            min.1 = min.1.min(y);
            max.0 = max.0.max(x);
            max.1 = max.1.max(y);
        }
    }
    if !min.0.is_finite() {
        min = (0.0, 0.0);
        max = min;
    }
    // Reserve room for multi-character labels and super/subscripts before scaling.
    let inset = padding.max(font * 2.2).min(width.min(height) * 0.45);
    let scale = bond
        .min((width - 2.0 * inset) / (max.0 - min.0).max(1.0))
        .min((height - 2.0 * inset) / (max.1 - min.1).max(1.0));
    let text_scale = (scale / bond).min(1.0);
    let fs = font * text_scale;
    let sw = stroke * text_scale;
    let points: Vec<_> = p
        .iter()
        .map(|&(x, y)| {
            (
                width / 2.0 + (x - (min.0 + max.0) / 2.0) * scale,
                height / 2.0 + (y - (min.1 + max.1) / 2.0) * scale,
            )
        })
        .collect();
    let wedges: rustc_hash::FxHashMap<_, _> = stereo::wedges(m, &points)
        .into_iter()
        .map(|w| ((w.center.min(w.target), w.center.max(w.target)), w))
        .collect();
    let aromatic_rings: Vec<_> = rings
        .iter()
        .filter(|r| {
            (0..r.len()).all(|k| {
                m.atom_bonds()[r[k]].iter().any(|b| {
                    b.target() == r[(k + 1) % r.len()] && b.bond_order() == BondOrder::Aromatic
                })
            })
        })
        .collect();
    let aromatic_edges: rustc_hash::FxHashSet<_> = aromatic_rings
        .iter()
        .flat_map(|r| {
            (0..r.len()).map(|k| {
                let (a, b) = (r[k], r[(k + 1) % r.len()]);
                (a.min(b), a.max(b))
            })
        })
        .collect();
    let mut svg = String::with_capacity(m.len() * 320 + 512);
    write!(svg,r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Molecular structure"><title>Molecular structure</title><g fill="none" stroke="#243247" stroke-width="{sw:.2}" stroke-linecap="round" stroke-linejoin="round">"##).unwrap();
    for i in 0..m.len() {
        if !visible[i] {
            continue;
        }
        for b in &m.atom_bonds()[i] {
            let j = b.target();
            if j <= i || !visible[j] {
                continue;
            }
            let (x, y) = points[i];
            let (xx, yy) = points[j];
            let length = (xx - x).hypot(yy - y);
            if length < 1e-8 {
                continue;
            }
            let (ux, uy) = ((xx - x) / length, (yy - y) / length);
            let clip = |k: usize| {
                if labels[k].is_empty() {
                    0.0
                } else {
                    ((fs * label_widths[k] * 0.5 + fs * 0.18) / ux.abs().max(1e-9))
                        .min(fs * 0.68 / uy.abs().max(1e-9))
                        .min(length * 0.42)
                }
            };
            let (a, c) = (
                (x + ux * clip(i), y + uy * clip(i)),
                (xx - ux * clip(j), yy - uy * clip(j)),
            );
            let gap = scale * 0.13;
            let ring = rings.iter().find(|r| {
                r.windows(2)
                    .any(|w| (w[0] == i && w[1] == j) || (w[0] == j && w[1] == i))
                    || (r.first() == Some(&i) && r.last() == Some(&j))
                    || (r.first() == Some(&j) && r.last() == Some(&i))
            });
            let side = ring.map_or(1.0, |r| {
                let center = r.iter().fold((0.0, 0.0), |(x, y), &k| {
                    (
                        x + points[k].0 / r.len() as f64,
                        y + points[k].1 / r.len() as f64,
                    )
                });
                if -uy * (center.0 - x) + ux * (center.1 - y) < 0.0 {
                    -1.0
                } else {
                    1.0
                }
            });
            let offsets: &[f64] = match b.bond_order() {
                BondOrder::Double if ring.is_some() => &[0.0],
                BondOrder::Double => &[-0.5, 0.5],
                BondOrder::Triple => &[-1.0, 0.0, 1.0],
                BondOrder::Quadruple => &[-1.5, -0.5, 0.5, 1.5],
                _ => &[0.0],
            };
            write!(svg, "<g data-bond=\"{i}-{j}\">").unwrap();
            if let Some(wedge) = wedges.get(&(i, j)) {
                let (tip, end) = if wedge.center == i { (a, c) } else { (c, a) };
                let d = (end.0 - tip.0, end.1 - tip.1);
                let len = d.0.hypot(d.1);
                let normal = (-d.1 / len * scale * 0.105, d.0 / len * scale * 0.105);
                write!(
                    svg,
                    "<g data-stereo=\"{}\" data-center=\"{}\" data-target=\"{}\">",
                    if wedge.solid { "wedge" } else { "hash" },
                    wedge.center,
                    wedge.target
                )
                .unwrap();
                if wedge.solid {
                    write!(svg,r##"<path d="M {:.2} {:.2} L {:.2} {:.2} L {:.2} {:.2} Z" fill="#243247" stroke="none"/>"##,tip.0,tip.1,end.0+normal.0,end.1+normal.1,end.0-normal.0,end.1-normal.1).unwrap();
                } else {
                    for k in 1..=7 {
                        let t = k as f64 / 7.0;
                        let q = (tip.0 + d.0 * t, tip.1 + d.1 * t);
                        line(
                            &mut svg,
                            (q.0 - normal.0 * t, q.1 - normal.1 * t),
                            (q.0 + normal.0 * t, q.1 + normal.1 * t),
                            "",
                        );
                    }
                }
                svg.push_str("</g></g>");
                continue;
            }
            for &off in offsets {
                line(
                    &mut svg,
                    (a.0 - uy * gap * off, a.1 + ux * gap * off),
                    (c.0 - uy * gap * off, c.1 + ux * gap * off),
                    if b.bond_order() == BondOrder::Coordinate {
                        " stroke-dasharray=\"3 3\""
                    } else {
                        ""
                    },
                );
            }
            if (b.bond_order() == BondOrder::Aromatic && !aromatic_edges.contains(&(i, j)))
                || (b.bond_order() == BondOrder::Double && ring.is_some())
            {
                // A second dashed line preserves aromatic bond identity without kekulization.
                line(
                    &mut svg,
                    (
                        a.0 + ux * gap - uy * gap * side,
                        a.1 + uy * gap + ux * gap * side,
                    ),
                    (
                        c.0 - ux * gap - uy * gap * side,
                        c.1 - uy * gap + ux * gap * side,
                    ),
                    if b.bond_order() == BondOrder::Aromatic {
                        " stroke-dasharray=\"2.5 3\""
                    } else {
                        ""
                    },
                );
            }
            svg.push_str("</g>");
        }
    }
    for ring in aromatic_rings {
        let center = ring.iter().fold((0.0, 0.0), |(x, y), &i| {
            (
                x + points[i].0 / ring.len() as f64,
                y + points[i].1 / ring.len() as f64,
            )
        });
        let mut inner = Vec::with_capacity(ring.len());
        let inset = if ring.iter().any(|&i| !labels[i].is_empty()) {
            (scale * 0.13).max(fs * 0.65)
        } else {
            scale * 0.13
        };
        // Intersect neighboring parallel offsets: one closed inset polygon,
        // with a continuous dash phase even at corners and fused-ring edges.
        for k in 0..ring.len() {
            let v = points[ring[k]];
            let prev = points[ring[(k + ring.len() - 1) % ring.len()]];
            let next = points[ring[(k + 1) % ring.len()]];
            let inward = |a: (f64, f64), b: (f64, f64)| {
                let d = (b.0 - a.0, b.1 - a.1);
                let len = d.0.hypot(d.1).max(1e-8);
                let mut n = (-d.1 / len, d.0 / len);
                if n.0 * (center.0 - a.0) + n.1 * (center.1 - a.1) < 0.0 {
                    n = (-n.0, -n.1);
                }
                n
            };
            let n1 = inward(prev, v);
            let n2 = inward(v, next);
            let denominator = (1.0 + n1.0 * n2.0 + n1.1 * n2.1).max(0.1);
            inner.push((
                v.0 + inset * (n1.0 + n2.0) / denominator,
                v.1 + inset * (n1.1 + n2.1) / denominator,
            ));
        }
        let perimeter = (0..inner.len())
            .map(|k| {
                let a = inner[k];
                let b = inner[(k + 1) % inner.len()];
                (a.0 - b.0).hypot(a.1 - b.1)
            })
            .sum::<f64>();
        let dash = perimeter / (perimeter / (scale * 0.14)).round().max(1.0) / 2.0;
        write!(
            svg,
            "<path data-aromatic-ring=\"true\" stroke-dasharray=\"{dash:.3} {dash:.3}\" d=\""
        )
        .unwrap();
        for (k, &(x, y)) in inner.iter().enumerate() {
            write!(svg, "{} {x:.2} {y:.2} ", if k == 0 { "M" } else { "L" }).unwrap();
        }
        svg.push_str("Z\"/>");
    }
    write!(svg,"</g><g font-family=\"Arial, Helvetica, sans-serif\" font-size=\"{fs:.2}\" font-weight=\"500\" text-anchor=\"middle\" dominant-baseline=\"central\">").unwrap();
    for (i, label) in labels.iter().enumerate() {
        if label.is_empty() {
            continue;
        }
        let (x, y) = points[i];
        let fill = if o.atom_colors {
            color(m.atomic_numbers()[i])
        } else {
            "#243247"
        };
        write!(
            svg,
            r#"<text data-atom="{i}" x="{x:.2}" y="{y:.2}" fill="{fill}">{label}</text>"#
        )
        .unwrap();
    }
    svg.push_str("</g></svg>");
    svg
}
fn line(svg: &mut String, a: (f64, f64), b: (f64, f64), extra: &str) {
    write!(
        svg,
        r#"<path d="M {:.2} {:.2} L {:.2} {:.2}"{extra}/>"#,
        a.0, a.1, b.0, b.1
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    fn mol(s: &str) -> MolecularGraph {
        let fragments = MolecularGraph::from_smiles(s).unwrap();
        MolecularGraph::combine(&fragments.iter().collect::<Vec<_>>())
    }

    #[test]
    fn deterministic_finite_and_complete() {
        for s in [
            "",
            "C",
            "[Na+]",
            "CCO",
            "C#N",
            "c1ccccc1",
            "c1ccc2ccccc2c1",
            "C1CCC2(CC1)CCCC2",
            "C1CC2CCC1C2",
            "[13CH3][O-]",
            "[H][H]",
            "CC.O",
        ] {
            let m = mol(s);
            let p = m.get_2d_coordinates();
            assert_eq!(p.len(), m.len(), "{s}");
            assert!(
                p.iter().all(|&(x, y)| x.is_finite() && y.is_finite()),
                "{s}"
            );
            assert_eq!(p, m.get_2d_coordinates(), "{s}");
            let svg = m.to_svg();
            assert_eq!(svg, m.to_svg());
            assert!(!svg.contains("NaN") && !svg.contains("inf"));
            let visible = visibility(&m);
            let bonds = (0..m.len())
                .filter(|&i| visible[i])
                .map(|i| {
                    m.atom_bonds()[i]
                        .iter()
                        .filter(|b| b.target() > i && visible[b.target()])
                        .count()
                })
                .sum::<usize>();
            assert_eq!(svg.matches("data-bond=").count(), bonds, "{s}");
        }
    }

    #[test]
    fn ring_and_chain_geometry() {
        for s in ["CCCCCC", "C1CCCCC1", "c1ccc2ccccc2c1", "C1CCC2(CC1)CCCC2"] {
            let m = mol(s);
            let p = m.get_2d_coordinates();
            let visible = visibility(&m);
            for i in 0..m.len() {
                if !visible[i] {
                    continue;
                }
                for b in &m.atom_bonds()[i] {
                    let j = b.target();
                    if !visible[j] {
                        continue;
                    }
                    let d = (p[i].0 - p[j].0).hypot(p[i].1 - p[j].1);
                    assert!((d - 1.0).abs() < 1e-6, "{s}: bond {i}-{j} length {d}");
                }
            }
            for i in 0..m.len() {
                for j in i + 1..m.len() {
                    if visible[i] && visible[j] {
                        assert!(
                            (p[i].0 - p[j].0).hypot(p[i].1 - p[j].1) > 0.5,
                            "{s}: overlap {i}-{j}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn chemical_labels_and_bonds() {
        let svg = mol("[13CH3][O-]").to_svg();
        assert!(svg.contains(">13</tspan>C"));
        assert!(svg.contains("−"));
        assert!(!svg.contains("<circle"));
        assert_eq!(mol("C#N").to_svg().matches("<path").count(), 3);
        assert_eq!(mol("C=C").to_svg().matches("<path").count(), 2);
        assert_eq!(
            mol("c1ccccc1").to_svg().matches("stroke-dasharray").count(),
            1
        );
        assert!(mol("O").to_svg().contains("OH<tspan"));
        assert!(mol("[2H]O").to_svg().contains(">2</tspan>H"));
    }

    #[test]
    fn fragments_remain_separate() {
        let m = mol("[Na+].[O-]C(=O)c1ccccc1");
        let svg = m.to_svg();
        assert!(svg.contains("Na<tspan") && svg.contains("−"));
        assert!(svg.matches("data-bond=").count() > 6);
        let p = m.get_2d_coordinates();
        assert!(p.iter().skip(1).all(|q| q.0 - p[0].0 >= 2.5 - 1e-8));
    }

    #[test]
    fn aromatic_interiors_are_closed_paths() {
        for (smiles, count) in [
            ("c1ccccc1", 1),
            ("c1ccc2ccccc2c1", 2),
            ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", 2),
        ] {
            let svg = mol(smiles).to_svg();
            assert_eq!(svg.matches("data-aromatic-ring=").count(), count);
            assert_eq!(svg.matches("stroke-dasharray=").count(), count);
            for path in svg
                .split("<path")
                .filter(|p| p.contains("data-aromatic-ring="))
            {
                let element = path.split("/>").next().unwrap();
                assert!(element.ends_with("Z\""));
            }
        }
    }

    #[test]
    fn long_structures_are_horizontal() {
        for s in ["CCCCCCCC", "c1ccc2ccccc2c1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"] {
            let m = mol(s);
            let p = m.get_2d_coordinates();
            let visible = visibility(&m);
            let mut min = (f64::INFINITY, f64::INFINITY);
            let mut max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for (i, &(x, y)) in p.iter().enumerate() {
                if visible[i] {
                    min.0 = min.0.min(x);
                    min.1 = min.1.min(y);
                    max.0 = max.0.max(x);
                    max.1 = max.1.max(y);
                }
            }
            assert!(max.0 - min.0 > max.1 - min.1, "{s}");
        }
    }

    #[test]
    fn options_and_empty_viewport() {
        let o = DrawOptions {
            width: 0,
            height: 0,
            bond_length: f64::NAN,
            font_size: f64::INFINITY,
            stroke_width: -1.0,
            padding: f64::NAN,
            atom_colors: false,
            cleanup: false,
        };
        let svg = mol("CCO").to_svg_with_options(&o);
        assert!(!svg.contains("NaN") && !svg.contains("inf"));
        assert!(svg.contains("viewBox=\"0 0 1 1\""));
        assert!(!svg.contains("#dc2626"));
    }
}
