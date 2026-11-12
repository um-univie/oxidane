//! Trial-distance sampling and classical (Torgerson) MDS embedding into 4D.
//!
//! The 4th dimension gives [`super::fourd`] room to resolve stereocentres
//! "through" each other before it anneals them flat -- the standard ETKDG
//! trick, and the reason this embeds into four dimensions rather than
//! three. The (x, y, z) components come back as `Vector3D` and `w` as a
//! parallel `Vec<f64>`, which is the shape `fourd` wants to zip them into
//! [`crate::vector::Vector4D`] from.

/// Distance intervals narrower than this (Å) are sampled at their lower bound.
const MIN_SAMPLING_INTERVAL: f64 = 1e-9;
/// Norm below which a random starting vector cannot be safely normalized.
const MIN_INITIAL_VECTOR_NORM: f64 = 1e-12;
/// Norm below which power iteration treats a matrix product as zero.
const MIN_POWER_VECTOR_NORM: f64 = 1e-14;

use super::bounds;
use crate::vector::Vector3D;
use ndarray::Array2;
use rand::{RngExt, SeedableRng, rngs::StdRng};

const POWER_ITERATION_MAX_ITERS: usize = 500;
const POWER_ITERATION_TOL: f64 = 1e-10;
const EMBED_DIMENSIONS: usize = 4;

/// Samples one trial (Euclidean-inconsistent in general) distance matrix by
/// drawing each pairwise distance uniformly within its bounds.
pub(super) fn sample_trial_distances(bounds_matrix: &Array2<f64>, rng: &mut StdRng) -> Array2<f64> {
    let n = bounds_matrix.nrows();
    let mut trial = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let lower = bounds::get_lower(bounds_matrix, i, j);
            let upper = bounds::get_upper(bounds_matrix, i, j);
            let (lo, hi) = if lower <= upper {
                (lower, upper)
            } else {
                (upper, lower)
            };
            let d = if hi - lo < MIN_SAMPLING_INTERVAL {
                lo
            } else {
                rng.random_range(lo..=hi)
            };
            trial[[i, j]] = d;
            trial[[j, i]] = d;
        }
    }
    trial
}

/// Embeds a trial distance matrix into 4D via classical MDS: double-centering
/// to a Gram matrix, then the top-4 eigenpairs via power iteration with
/// deflation (cheaper than a full eigendecomposition since only k=4 of n
/// eigenvectors are needed, and this runs inside the per-attempt retry loop).
pub(super) fn classical_mds_4d(trial: &Array2<f64>) -> (Vec<Vector3D>, Vec<f64>) {
    let n = trial.nrows();
    let gram = double_centered_gram(trial);
    let eigenpairs = top_k_eigenpairs(&gram, EMBED_DIMENSIONS.min(n.max(1)));

    let mut coords = vec![[0.0f64; EMBED_DIMENSIONS]; n];
    for (dim, (eigenvalue, eigenvector)) in eigenpairs.iter().enumerate() {
        let scale = eigenvalue.max(0.0).sqrt();
        for i in 0..n {
            coords[i][dim] = eigenvector[i] * scale;
        }
    }

    let positions = coords
        .iter()
        .map(|c| Vector3D::new(c[0], c[1], c[2]))
        .collect();
    let w = coords.iter().map(|c| c[3]).collect();
    (positions, w)
}

/// `B[i,j] = -0.5 * (D2[i,j] - rowMean[i] - colMean[j] + grandMean)`, computed
/// via row/column means (O(n^2)) rather than an explicit `J*D2*J` product
/// (which would be O(n^3)).
fn double_centered_gram(trial: &Array2<f64>) -> Array2<f64> {
    let n = trial.nrows();
    let d2 = trial.mapv(|d| d * d);
    let row_mean: Vec<f64> = (0..n).map(|i| d2.row(i).sum() / n.max(1) as f64).collect();
    let grand_mean: f64 = row_mean.iter().sum::<f64>() / n.max(1) as f64;

    let mut gram = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            gram[[i, j]] = -0.5 * (d2[[i, j]] - row_mean[i] - row_mean[j] + grand_mean);
        }
    }
    gram
}

fn matvec(m: &Array2<f64>, x: &[f64]) -> Vec<f64> {
    let n = m.nrows();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let row = m.row(i);
        let mut sum = 0.0;
        for j in 0..n {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
    y
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn random_unit_vec(n: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let len = norm(&v);
    if len < MIN_INITIAL_VECTOR_NORM {
        v[0] = 1.0;
        return v;
    }
    for x in v.iter_mut() {
        *x /= len;
    }
    v
}

/// Gershgorin bound `s = max_i sum_j |M[i,j]|`, used to shift `M` so power
/// iteration converges to the largest eigenvalue rather than the most
/// negative one (Gram matrices from noisy trial distances commonly have a
/// handful of small negative eigenvalues).
fn gershgorin_shift(m: &Array2<f64>) -> f64 {
    let n = m.nrows();
    (0..n)
        .map(|i| m.row(i).iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Top-`k` eigenpairs of symmetric `m`, returned as `(eigenvalue, eigenvector)`
/// sorted descending by eigenvalue. Uses shifted power iteration with
/// deflation; precision is loose since this only seeds gradient
/// refinement, not a final answer.
fn top_k_eigenpairs(m: &Array2<f64>, k: usize) -> Vec<(f64, Vec<f64>)> {
    let n = m.nrows();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let shift = gershgorin_shift(m);
    let mut rng = StdRng::seed_from_u64(0x4D53_4544);
    // (shifted eigenvalue, eigenvector) of already-extracted components.
    let mut found: Vec<(f64, Vec<f64>)> = Vec::with_capacity(k);

    for _ in 0..k {
        let mut v = random_unit_vec(n, &mut rng);

        for _ in 0..POWER_ITERATION_MAX_ITERS {
            let mut y = matvec(m, &v);
            for i in 0..n {
                y[i] += shift * v[i];
            }
            for (lambda, u) in &found {
                let proj = dot(u, &v);
                for i in 0..n {
                    y[i] -= lambda * u[i] * proj;
                }
            }
            let y_norm = norm(&y);
            if y_norm < MIN_POWER_VECTOR_NORM {
                break;
            }
            for x in y.iter_mut() {
                *x /= y_norm;
            }
            let direction_cosine = dot(&y, &v).abs();
            v = y;
            if direction_cosine > 1.0 - POWER_ITERATION_TOL {
                break;
            }
        }

        // Rayleigh quotient against the (shifted, deflated) operator for the
        // final eigenvalue estimate.
        let mut av = matvec(m, &v);
        for i in 0..n {
            av[i] += shift * v[i];
        }
        for (lambda, u) in &found {
            let proj = dot(u, &v);
            for i in 0..n {
                av[i] -= lambda * u[i] * proj;
            }
        }
        let lambda_shifted = dot(&v, &av);
        found.push((lambda_shifted, v));
    }

    found
        .into_iter()
        .map(|(lambda_shifted, v)| (lambda_shifted - shift, v))
        .collect()
}
