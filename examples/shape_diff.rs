//! Differential check of the shape descriptors against RDKit.
//!
//! Takes the SDF `tests/shape_oracle.py` wrote as its first argument and the
//! oracle's table on stdin, so that both sides measure the same coordinates.
use oxidane::molecule::shape::{shape_descriptors, usr, usrcat};
use oxidane::prelude::*;
use std::io::BufRead;

fn main() {
    let path = std::env::args().nth(1).expect("an SDF to read");
    let conformers = Molecule3D::from_sdf(&path).expect("the SDF should parse");
    eprintln!("{} conformers", conformers.len());

    let labels = [
        "PMI1", "PMI2", "PMI3", "NPR1", "NPR2", "gyration", "asphericity", "eccentricity",
        "shape factor", "spherocity", "PBF",
    ];
    let mut worst_usr: f64 = 0.0;
    let mut worst_usrcat: f64 = 0.0;
    let mut worst_shape = [0.0f64; 11];
    let mut worst_shape_example: Vec<String> = vec![String::new(); 11];
    let mut total = 0usize;

    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<&str> = line.trim_end().split('\t').collect();
        if fields.len() < 4 {
            continue;
        }
        let index: usize = fields[0].parse().unwrap_or(usize::MAX);
        let Some(conformer) = conformers.get(index) else {
            continue;
        };
        let numbers = |field: &str| -> Vec<f64> {
            field
                .split(',')
                .filter_map(|value| value.parse::<f64>().ok())
                .collect()
        };
        total += 1;

        let graph = conformer.to_graph();
        let ours = usr(&conformer.positions);
        let difference = largest_difference(&ours, &numbers(fields[1]));
        if difference > worst_usr {
            worst_usr = difference;
            eprintln!("usr {index} {difference:.6}");
        }

        let ours = usrcat(&graph, &conformer.positions);
        let theirs = numbers(fields[2]);
        let difference = largest_difference(&ours, &theirs);
        if difference > 1e-4 {
            let blocks: Vec<String> = (0..5)
                .map(|block| {
                    let range = block * 12..block * 12 + 12;
                    format!(
                        "{:.4}",
                        largest_difference(&ours[range.clone()], &theirs[range])
                    )
                })
                .collect();
            eprintln!("usrcat {index} blocks {}", blocks.join(" "));
        }
        worst_usrcat = worst_usrcat.max(difference);

        let shape = shape_descriptors(&graph, &conformer.positions);
        let ours = [
            shape.first_moment,
            shape.second_moment,
            shape.third_moment,
            shape.first_ratio,
            shape.second_ratio,
            shape.radius_of_gyration,
            shape.asphericity,
            shape.eccentricity,
            shape.inertial_shape_factor,
            shape.spherocity_index,
            shape.plane_of_best_fit,
        ];
        let theirs = numbers(fields[3]);
        for (position, (&ours, &theirs)) in ours.iter().zip(theirs.iter()).enumerate() {
            // Relative, since the moments of inertia run to thousands while the
            // ratios are all below one.
            let scale = ours.abs().max(theirs.abs()).max(1e-6);
            let difference = (ours - theirs).abs() / scale;
            if difference > worst_shape[position] {
                worst_shape[position] = difference;
                worst_shape_example[position] = format!("{index}: ours {ours} theirs {theirs}");
            }
        }
    }

    // Timed on the same conformers RDKit is timed on, so that the comparison
    // is of the same work over the same molecules.
    let graphs: Vec<_> = conformers.iter().map(|c| c.to_graph()).collect();
    for (label, run) in [
        (
            "usr     ",
            &(|index: usize| usr(&conformers[index].positions)[0]) as &dyn Fn(usize) -> f64,
        ),
        ("usrcat  ", &|index: usize| {
            usrcat(&graphs[index], &conformers[index].positions)[0]
        }),
        ("shape 3d", &|index: usize| {
            shape_descriptors(&graphs[index], &conformers[index].positions).first_moment
        }),
    ] {
        let start = std::time::Instant::now();
        let mut checksum = 0.0;
        for _ in 0..5 {
            for index in 0..conformers.len() {
                checksum += run(index);
            }
        }
        let elapsed = start.elapsed();
        println!(
            "{label}: {:>7.2} us/conformer (checksum {checksum:.1})",
            elapsed.as_secs_f64() * 1e6 / (5.0 * conformers.len() as f64)
        );
    }

    println!("{total} conformers compared");
    println!("worst USR difference    {worst_usr:.9}");
    println!("worst USRCAT difference {worst_usrcat:.9}");
    for (label, (worst, example)) in labels
        .iter()
        .zip(worst_shape.iter().zip(worst_shape_example.iter()))
    {
        println!("  {label:14} worst relative {worst:.9}  {example}");
    }
}

fn largest_difference(ours: &[f64], theirs: &[f64]) -> f64 {
    if ours.len() != theirs.len() {
        return f64::INFINITY;
    }
    ours.iter()
        .zip(theirs.iter())
        .map(|(&ours, &theirs)| (ours - theirs).abs())
        .fold(0.0, f64::max)
}
