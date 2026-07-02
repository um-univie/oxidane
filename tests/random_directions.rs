use oxidane::vector::Vector3D;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn random_directions_have_uniform_area_moments() {
    let samples = 100_000;
    let mut rng = StdRng::seed_from_u64(20261020);
    let mut second = [0.0; 3];
    let mut mean = [0.0; 3];
    let mut bands = [0usize; 10];
    for _ in 0..samples {
        let v = Vector3D::random_unit_vector_with_rng(&mut rng);
        assert!((v.length() - 1.0).abs() < 1e-12);
        for (axis, x) in [v.x, v.y, v.z].into_iter().enumerate() {
            mean[axis] += x;
            second[axis] += x * x;
        }
        bands[(((v.z + 1.0) * 5.0) as usize).min(9)] += 1;
    }
    for axis in 0..3 {
        assert!((mean[axis] / samples as f64).abs() < 0.015);
        assert!((second[axis] / samples as f64 - 1.0 / 3.0).abs() < 0.015);
    }
    for count in bands {
        assert!((count as f64 / samples as f64 - 0.1).abs() < 0.01);
    }
}
