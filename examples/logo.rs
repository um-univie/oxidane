//! Generates the project logo as SVG -- no image editor, no raster assets.
//!
//! Oxidane is the IUPAC name for water, so the mark is a water molecule: the
//! "o" of the wordmark is the oxygen, and its two hydrogens rise above the
//! line. The angle between them is not drawn by hand. `[H]O[H]` is embedded
//! with `embed_3d` (distance geometry, then MMFF94) and the H-O-H angle is
//! read off the coordinates, so the picture is output of the library it
//! labels -- change the embedder and the logo changes with it.
//!
//! ```sh
//! cargo run --example logo    # writes assets/logo.svg, logo-dark.svg,
//!                             # logo-mark.svg
//! ```

use oxidane::prelude::*;
use std::fmt::Write as _;

/// The oxygen ring: radius and stroke width in the wordmark, in user units.
const RING_RADIUS: f64 = 10.5;
const RING_WIDTH: f64 = 5.0;

struct Theme {
    name: &'static str,
    ink: &'static str,
    oxygen: &'static str,
    tagline: &'static str,
}

/// Ink matches canonaut's logo; the oxygen is rust, for iron oxide and for
/// the language.
const LIGHT: Theme = Theme {
    name: "logo",
    ink: "#1d2433",
    oxygen: "#c2502b",
    tagline: "#6b7280",
};

const DARK: Theme = Theme {
    name: "logo-dark",
    ink: "#e8ecf4",
    oxygen: "#e0703f",
    tagline: "#9aa4b2",
};

/// Sizes of one drawing of the mark.
struct Mark {
    center: (f64, f64),
    ring_radius: f64,
    ring_width: f64,
    bond: f64,
    bond_width: f64,
    hydrogen_radius: f64,
}

/// The H-O-H angle, in radians, of water as `embed_3d` places it.
fn water_angle() -> f64 {
    let water = MolecularGraph::from_smiles_single("[H]O[H]").expect("water parses");
    let embedded = water.embed_3d().expect("water embeds");
    let oxygen = (0..embedded.len())
        .find(|&atom| embedded.atomic_numbers()[atom] == 8)
        .expect("water has an oxygen");
    let bonds: Vec<_> = (0..embedded.len())
        .filter(|&atom| atom != oxygen)
        .map(|atom| {
            let (h, o) = (
                embedded.get_atom_position(atom).unwrap(),
                embedded.get_atom_position(oxygen).unwrap(),
            );
            [h.x - o.x, h.y - o.y, h.z - o.z]
        })
        .collect();
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let [first, second] = bonds.as_slice() else {
        panic!("water has two hydrogens")
    };
    (dot(first, second) / (dot(first, first) * dot(second, second)).sqrt()).acos()
}

/// The ring, and the two hydrogens above it at `angle` either side of the
/// vertical. Bonds stop short of the ring so it reads as a letter.
fn render_mark(theme: &Theme, mark: &Mark, angle: f64, indent: &str) -> String {
    let (cx, cy) = mark.center;
    let (sin, cos) = (angle / 2.0).sin_cos();
    let gap = mark.ring_radius + mark.ring_width / 2.0 + 1.5;
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        "{indent}<g stroke=\"{}\" stroke-width=\"{}\" stroke-linecap=\"round\">",
        theme.ink, mark.bond_width
    );
    for side in [-1.0, 1.0] {
        let (dx, dy) = (side * sin, -cos);
        let _ = writeln!(
            svg,
            "{indent}  <line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\"/>",
            cx + dx * gap,
            cy + dy * gap,
            cx + dx * mark.bond,
            cy + dy * mark.bond
        );
    }
    let _ = writeln!(svg, "{indent}</g>");
    for side in [-1.0, 1.0] {
        let _ = writeln!(
            svg,
            "{indent}<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"{}\" fill=\"{}\"/>",
            cx + side * sin * mark.bond,
            cy - cos * mark.bond,
            mark.hydrogen_radius,
            theme.ink
        );
    }
    let _ = writeln!(
        svg,
        "{indent}<circle cx=\"{cx:.2}\" cy=\"{cy:.2}\" r=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"{}\"/>",
        mark.ring_radius, theme.oxygen, mark.ring_width
    );
    svg
}

fn render_logo(theme: &Theme, angle: f64) -> String {
    let mark = Mark {
        center: (36.0, 56.0),
        ring_radius: RING_RADIUS,
        ring_width: RING_WIDTH,
        bond: 34.0,
        bond_width: 3.6,
        hydrogen_radius: 4.6,
    };
    let (cx, cy) = mark.center;
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 250 108\" width=\"250\" height=\"108\" \
         role=\"img\" aria-label=\"oxidane\">"
    );
    let _ = writeln!(svg, "  <title>oxidane</title>");
    let _ = writeln!(
        svg,
        "  <desc>The o of oxidane drawn as the oxygen of a water molecule, its hydrogens at the {:.2} degree \
         angle oxidane embeds water with.</desc>",
        angle.to_degrees()
    );
    svg.push_str(&render_mark(theme, &mark, angle, "  "));
    let _ = writeln!(
        svg,
        "  <text x=\"{:.2}\" y=\"{:.2}\" font-family=\"ui-monospace, SFMono-Regular, Menlo, Consolas, monospace\" \
         font-size=\"40\" font-weight=\"600\" fill=\"{}\" letter-spacing=\"-0.5\">xidane</text>",
        cx + RING_RADIUS + 4.0,
        cy + 11.6,
        theme.ink
    );
    let _ = writeln!(
        svg,
        "  <text x=\"{:.2}\" y=\"94\" font-family=\"ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif\" \
         font-size=\"12.5\" fill=\"{}\" letter-spacing=\"0.6\">cheminformatics in pure Rust</text>",
        cx - RING_RADIUS - 2.0,
        theme.tagline
    );
    let _ = writeln!(svg, "</svg>");
    svg
}

/// The mark alone, for icons: the same drawing scaled up to fill 108 x 108.
fn render_square_mark(theme: &Theme, angle: f64) -> String {
    let mark = Mark {
        center: (54.0, 62.0),
        ring_radius: 19.0,
        ring_width: 9.0,
        bond: 50.0,
        bond_width: 6.5,
        hydrogen_radius: 8.5,
    };
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 108 108\" width=\"108\" height=\"108\" \
         role=\"img\" aria-label=\"oxidane\">"
    );
    let _ = writeln!(svg, "  <title>oxidane</title>");
    svg.push_str(&render_mark(theme, &mark, angle, "  "));
    let _ = writeln!(svg, "</svg>");
    svg
}

fn main() -> std::io::Result<()> {
    let angle = water_angle();
    println!("water: H-O-H {:.2} degrees", angle.to_degrees());

    std::fs::create_dir_all("assets")?;
    for theme in [&LIGHT, &DARK] {
        let path = format!("assets/{}.svg", theme.name);
        std::fs::write(&path, render_logo(theme, angle))?;
        println!("wrote {path}");
    }
    std::fs::write("assets/logo-mark.svg", render_square_mark(&LIGHT, angle))?;
    println!("wrote assets/logo-mark.svg");
    Ok(())
}
