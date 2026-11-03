use rand::Rng;
use std::fmt;
use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

/// A 3D vector represented by its x, y, and z components.
///
/// # Example
///
/// ```
/// use oxidane::vector::Vector3D;
///
/// let v1 = Vector3D::new(1.0, 2.0, 3.0);
/// let v2 = Vector3D::new(4.0, 5.0, 6.0);
/// let v3 = v1 + v2;
/// let v4 = v1 * 2.0;
/// let v5 = v1 / 2.0;
/// assert!((v3 - Vector3D::new(5.0, 7.0, 9.0)).length() < 0.1);
/// assert!((v4 - Vector3D::new(2.0, 4.0, 6.0)).length() < 0.1);
/// assert!((v5 - Vector3D::new(0.5, 1.0, 1.5)).length() < 0.1);
/// ```
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
// Implement the Add trait for Vector
impl Add<Vector3D> for Vector3D {
    type Output = Vector3D;

    fn add(self, other: Vector3D) -> Vector3D {
        Vector3D {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

// Implement the Mul trait for Vector
impl Mul<f64> for Vector3D {
    type Output = Vector3D;

    fn mul(self, scalar: f64) -> Vector3D {
        Vector3D {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl Mul<Vector3D> for Vector3D {
    type Output = Vector3D;

    /// Performs element-wise multiplication of two vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v1 = Vector3D::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3D::new(4.0, 5.0, 6.0);
    /// let result = v1 * v2;
    /// let delta = result - Vector3D::new(4.0, 10.0, 18.0);
    /// assert!(delta.length() < 0.1);
    /// ```
    fn mul(self, other: Vector3D) -> Vector3D {
        self.element_wise_mul(&other)
    }
}

impl Div<f64> for Vector3D {
    type Output = Vector3D;

    fn div(self, scalar: f64) -> Self::Output {
        Vector3D {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

impl Sub<Vector3D> for Vector3D {
    type Output = Vector3D;
    fn sub(self, rhs: Vector3D) -> Self::Output {
        Vector3D::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl AddAssign for Vector3D {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl MulAssign<f64> for Vector3D {
    fn mul_assign(&mut self, scalar: f64) {
        self.x = self.x * scalar;
        self.y = self.y * scalar;
        self.z = self.z * scalar;
    }
}

impl IntoIterator for Vector3D {
    type Item = f64;
    type IntoIter = std::array::IntoIter<f64, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_array().into_iter()
    }
}

impl<'a> IntoIterator for &'a Vector3D {
    type Item = &'a f64;
    type IntoIter = std::array::IntoIter<&'a f64, 3>;

    fn into_iter(self) -> Self::IntoIter {
        [&self.x, &self.y, &self.z].into_iter()
    }
}

impl Default for Vector3D {
    /// Creates a default `Vector3D` being the zero vector
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v = Vector3D::default();
    /// println!("Vector: {:?}", v);
    /// assert_eq!(v.length(),0.0)
    /// ```
    fn default() -> Vector3D {
        Vector3D::new(0.0, 0.0, 0.0)
    }
}
impl<'a> Mul<f64> for &'a Vector3D {
    type Output = Vector3D;

    fn mul(self, scalar: f64) -> Vector3D {
        Vector3D {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl<'a> Div<f64> for &'a Vector3D {
    type Output = Vector3D;

    fn div(self, scalar: f64) -> Vector3D {
        Vector3D {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

/// `cos(phi)` and `sin(phi)` for the signed dihedral `i-b-c-j`, and, from
/// [`dihedral_and_gradient`], `d(phi)/dp` for its four points in that order.
///
/// The angle itself is never formed. Every energy built on a dihedral is a
/// sum of `cos(n phi)` and its derivative a sum of `sin(n phi)`, and both
/// follow from `cos(phi)` and `sin(phi)` by [`multiple_angle`]'s recurrence
/// -- so computing `phi = atan2(...)` and then calling `cos` and `sin` on it
/// again is a round trip through three libm calls per torsion per
/// evaluation. Dropping it takes MMFF's torsion term from 32 to 12 ns per
/// torsion.
///
/// The sign convention is the usual one: looking down `b -> c`, the angle
/// from the `b -> i` direction to the `c -> j` direction, positive
/// clockwise -- so an eclipsed arrangement is 0 and an anti one is 180, and
/// it agrees with what RDKit's `GetDihedralDeg` reports for the same four
/// points. Taking `i - b` rather than `b - i` is the whole of that
/// convention, and getting it backwards costs exactly 180 degrees: an
/// energy in `cos(2 phi)` never notices, one in `cos(3 phi)` inverts, and
/// UFF's ethane then minimises eclipsed.
///
/// `None` where the angle is undefined: a collinear i-b-c or b-c-j, or a
/// zero-length central bond.
pub(crate) fn dihedral_cosine_sine(
    i: Vector3D,
    b: Vector3D,
    c: Vector3D,
    j: Vector3D,
) -> Option<(f64, f64)> {
    let axis = c - b;
    let axis_sq = axis.dot(&axis);
    if axis_sq < 1e-16 {
        return None;
    }
    let axis_hat = axis * (1.0 / axis_sq.sqrt());
    let from_b = i - b;
    let trailing = j - c;
    let perpendicular_in = from_b - axis_hat * from_b.dot(&axis_hat);
    let perpendicular_out = trailing - axis_hat * trailing.dot(&axis_hat);

    // `x` and `y` are what `atan2` would have been given: both components
    // are perpendicular to the axis, so their magnitudes divide out and
    // `hypot(x, y)` is the product of the two lengths.
    let x = perpendicular_in.dot(&perpendicular_out);
    let y = axis_hat.cross(&perpendicular_in).dot(&perpendicular_out);
    let radius = (x * x + y * y).sqrt();
    if radius < 1e-16 {
        return None;
    }
    Some((x / radius, y / radius))
}

/// `cos(n phi)` and `sin(n phi)` from `cos(phi)` and `sin(phi)`, by the
/// Chebyshev recurrence `f(n) = 2 cos(phi) f(n-1) - f(n-2)`. Exact, and two
/// multiplications per fold against a libm call each.
pub(crate) fn multiple_angle(cosine: f64, sine: f64, fold: usize) -> (f64, f64) {
    let (mut previous, mut current) = ((1.0, 0.0), (cosine, sine));
    if fold == 0 {
        return previous;
    }
    for _ in 1..fold {
        let next = (
            2.0 * cosine * current.0 - previous.0,
            2.0 * cosine * current.1 - previous.1,
        );
        previous = current;
        current = next;
    }
    current
}

/// [`dihedral_cosine_sine`], with `d(phi)/dp` for the four points.
pub(crate) fn dihedral_and_gradient(
    i: Vector3D,
    b: Vector3D,
    c: Vector3D,
    j: Vector3D,
) -> Option<(f64, f64, [Vector3D; 4])> {
    let leading = b - i;
    let axis = c - b;
    let trailing = j - c;

    let normal_in = leading.cross(&axis);
    let normal_out = axis.cross(&trailing);
    let (in_sq, out_sq) = (normal_in.dot(&normal_in), normal_out.dot(&normal_out));
    let axis_sq = axis.dot(&axis);
    if in_sq < 1e-16 || out_sq < 1e-16 || axis_sq < 1e-16 {
        return None;
    }
    let axis_length = axis_sq.sqrt();

    // The *signed* angle's cosine and sine. The derivative below is derived
    // for the signed angle, and pairing it with an unsigned one flips the
    // gradient wherever the true dihedral is negative -- which leaves an
    // energy in `cos(n phi)` right and its gradient wrong by a sign on half
    // of every structure.
    let (cosine, sine) = dihedral_cosine_sine(i, b, c, j)?;

    // The two end points move the dihedral only by swinging about the
    // central bond, so their derivatives lie along their own plane normals.
    // The two central ones take what is left, weighted by how far each end
    // bond projects onto the axis -- and `phi` is invariant under
    // translating all four, so the fourth derivative is fixed by the three
    // summing to zero.
    let d_i = normal_in * (-axis_length / in_sq);
    let d_j = normal_out * (axis_length / out_sq);
    let lead_projection = leading.dot(&axis) / axis_sq;
    let trail_projection = trailing.dot(&axis) / axis_sq;
    let d_b = d_i * -(1.0 + lead_projection) + d_j * trail_projection;
    let d_c = -(d_i + d_j + d_b);
    Some((cosine, sine, cap_derivatives([d_i, d_b, d_c, d_j])))
}

/// The largest `d(angle)/d(position)`, in radians per Angstrom, that a
/// geometric derivative is allowed to report.
///
/// An end atom one bond length out swings the dihedral at about
/// `1 / 1.5 A`, so a hundred is already two orders of magnitude past
/// anything a real geometry produces.
pub(crate) const MAX_GEOMETRIC_DERIVATIVE: f64 = 100.0;

/// Scales a set of derivatives down together if the largest exceeds
/// [`MAX_GEOMETRIC_DERIVATIVE`], leaving the direction alone.
///
/// A dihedral's derivative carries `1 / sin(theta)` at each end: as either
/// end angle straightens, the four positions stop pinning the dihedral down
/// and the derivative diverges, while the angle itself and the energy built
/// on it stay perfectly bounded. The divergence is in the coordinate, not in
/// the physics, and handing it to a minimiser is what a minimiser cannot
/// survive: Armijo's acceptance test asks each trial step for a decrease
/// proportional to `|g|^2`, no step can deliver `1e16` kcal/mol, every
/// backtrack is rejected, and the minimisation returns the structure it
/// started from. One peptide embedding trial in eight came back at 939
/// kcal/mol that way, having stopped five iterations in.
///
/// Capping keeps the direction, which is still downhill and still correct,
/// and bounds the step the line search has to consider. The alternative --
/// dropping the term where it is singular, as [`super::molecule::forcefield`]
/// does for a collinear angle -- also lets the minimisation proceed, but it
/// throws away a restoring force exactly where the geometry most needs one.
pub(crate) fn cap_derivatives<const N: usize>(mut derivatives: [Vector3D; N]) -> [Vector3D; N] {
    let largest = derivatives
        .iter()
        .map(|d| d.dot(d))
        .fold(0.0f64, f64::max)
        .sqrt();
    if largest > MAX_GEOMETRIC_DERIVATIVE {
        let scale = MAX_GEOMETRIC_DERIVATIVE / largest;
        for derivative in &mut derivatives {
            *derivative = *derivative * scale;
        }
    }
    derivatives
}

impl Vector3D {
    /// Creates a `Vector3D` from x, y, and z components.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// const V: Vector3D = Vector3D::new_const(1.0, 2.0, 3.0);
    /// assert_eq!((V.x, V.y, V.z), (1.0, 2.0, 3.0));
    /// ```
    pub const fn new_const(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    /// Creates a `Vector3D` from an array of 3 elements, representing the x, y, and z components.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let array = [1.0, 2.0, 3.0];
    /// let v = Vector3D::from_vec(&array);
    /// assert_eq!((v.x, v.y, v.z), (1.0, 2.0, 3.0));
    /// ```
    pub fn from_vec(array: &[f64; 3]) -> Vector3D {
        Vector3D {
            x: array[0],
            y: array[1],
            z: array[2],
        }
    }
    /// Creates a `Vector3D` from x, y, and z components.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v = Vector3D::new(1.0, 2.0, 3.0);
    /// assert_eq!((v.x, v.y, v.z), (1.0, 2.0, 3.0));
    /// ```
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn unit_x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }
    pub fn unit_y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }
    pub fn unit_z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }
    pub fn x(&self) -> f64 {
        self.x
    }
    pub fn y(&self) -> f64 {
        self.y
    }
    pub fn z(&self) -> f64 {
        self.z
    }
    /// Calculates the difference between two `Vector3D` objects, this is equivalent to the vector
    /// pointing from `other` to `self`.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v1 = Vector3D::new(1.0, 1.0, 1.0);
    /// let v2 = Vector3D::new(1.0, 1.0, -1.0);
    ///
    /// let v_diff = v1.difference(&v2);
    /// assert_eq!(v_diff, Vector3D::new(0.0, 0.0, 2.0));
    /// assert_eq!(v_diff.length(), 2.0);
    /// ```
    pub fn difference(&self, other: &Self) -> Self {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        Self::new(dx, dy, dz)
    }
    /// Calculates the length of a vector.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    ///     let v = Vector3D::new(1.0, 1.0, 1.0);
    ///     println!("Vector: {:?}", v.length());
    ///     assert_eq!(v.length(), 3.0_f64.sqrt())
    ///
    /// ```
    pub fn length(&self) -> f64 {
        self.squared_length().sqrt()
    }

    /// Alias for length. Returns the magnitude of the vector.
    ///
    /// # Returns
    ///
    /// * `f64` - The magnitude (length) of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v = Vector3D::new(3.0, 4.0, 0.0);
    /// assert_eq!(v.magnitude(), 5.0);
    /// ```
    pub fn magnitude(&self) -> f64 {
        self.length()
    }

    /// Calculates the squared length of the vector.
    ///
    /// This is more efficient than `length()` when you only need to compare lengths,
    /// as it avoids the square root calculation.
    ///
    /// # Returns
    ///
    /// * `f64` - The squared length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v = Vector3D::new(1.0, 2.0, 2.0);
    /// assert_eq!(v.squared_length(), 9.0);
    /// ```
    pub fn squared_length(&self) -> f64 {
        self.dot(self)
    }

    /// Determines if two vectors are parallel within a given tolerance.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector to compare with.
    /// * `tolerance` - The tolerance for parallelism. Smaller values require more precise parallelism.
    ///
    /// # Returns
    ///
    /// * `bool` - True if the vectors are parallel within the given tolerance, false otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v1 = Vector3D::new(1.0, 0.0, 0.0);
    /// let v2 = Vector3D::new(2.0, 0.0, 0.0);
    /// assert!(v1.are_vectors_parallel(&v2, 1e-10));
    ///
    /// let v3 = Vector3D::new(1.0, 0.1, 0.0);
    /// assert!(!v1.are_vectors_parallel(&v3, 1e-10));
    /// ```
    pub fn are_vectors_parallel(&self, other: &Vector3D, tolerance: f64) -> bool {
        let cross_product = self.cross(other);
        cross_product.length() < tolerance * self.length() * other.length()
    }

    /// Calculates the dot product of two vectors.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v1 = Vector3D::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3D::new(4.0, 5.0, 6.0);
    /// // 1*4 + 2*5 + 3*6
    /// assert_eq!(v1.dot(&v2), 32.0);
    /// ```
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    /// Calculates the distance between to vectors
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v1 = Vector3D::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3D::new(4.0, 5.0, 6.0);
    /// // Each component differs by 3, so the distance is 3*sqrt(3).
    /// assert!((v1.distance(&v2) - 27.0_f64.sqrt()).abs() < 1e-12);
    /// ```
    pub fn distance(&self, other: &Self) -> f64 {
        self.difference(other).length()
    }
    /// Returns the squared distance between two vectors
    ///
    /// # Example
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    /// let v1 = Vector3D::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3D::new(4.0, 5.0, 6.0);
    /// assert_eq!(v1.distance_squared(&v2),27.0);
    /// ```
    pub fn distance_squared(&self, other: &Self) -> f64 {
        self.difference(other).squared_length()
    }
    /// Generates a direction uniformly distributed over the unit sphere.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v = Vector3D::random_unit_vector();
    /// println!("Vector: {:?}", v);
    /// assert_eq!(v.length().round(),1.0)
    /// ```
    ///
    pub fn random_unit_vector() -> Self {
        Self::random_unit_vector_with_rng(&mut rand::rng())
    }

    /// Generates a uniform sphere direction using a caller-supplied RNG.
    /// A seeded RNG makes sequences reproducible.
    pub fn random_unit_vector_with_rng<R: Rng + ?Sized>(rng: &mut R) -> Self {
        // Equal-height bands have equal area: cos(theta), not theta, is uniform.
        let z: f64 = rng.random_range(-1.0..1.0);
        let phi = rng.random_range(0.0..std::f64::consts::TAU);
        let radius = (1.0 - z * z).sqrt();
        let (sin_phi, cos_phi) = phi.sin_cos();
        Vector3D::new(radius * cos_phi, radius * sin_phi, z)
    }
    /// Calculates the cross product of two vectors and returns a new vector
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let vec1 = Vector3D::unit_x();
    /// let vec2 = Vector3D::unit_y();
    /// assert_eq!(vec1.cross(&vec2),Vector3D::unit_z())
    /// ```
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    /// Returns the angle between two vectors in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let vec_a = Vector3D::new(1.0, 0.0, 0.0);
    /// let vec_b = Vector3D::new(0.0, 1.0, 0.0);
    ///
    /// let angle = vec_a.angle_between(&vec_b);
    /// assert_eq!(angle, Some(std::f64::consts::FRAC_PI_2));
    /// ```
    pub fn angle_between(&self, other: &Vector3D) -> Option<f64> {
        let lengths_product = self.length() * other.length();
        if lengths_product == 0.0 {
            None
        } else {
            let angle_cosine = (self.dot(other) / (lengths_product)).clamp(-1.0, 1.0);
            Some(angle_cosine.acos())
        }
    }
    /// Calculates a normal vector for a given input vector
    ///
    /// # Panics
    /// This function does not panic as the case of a zero length vector is handeled.
    ///
    /// # Example
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let vec1 = Vector3D::new(1.0,2.0,3.0);
    /// assert_eq!(vec1.normalize().length().ceil(),1.0)
    /// ```
    pub fn normalize(&self) -> Self {
        if self.length() == 0.0 {
            *self
        } else {
            *self / self.length()
        }
    }

    /// Calculates the angle given three points in the order, start, middle, end If any of the points are the same or any of the vectors is 0 the angle is not defined and None is returned
    ///
    /// # Example
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let start_point = Vector3D::new(1.0,0.0,0.0);
    /// let middle_point = Vector3D::new(0.0,0.0,0.0);
    /// let end_point = Vector3D::new(0.0,1.0,0.0);
    /// assert_eq!(start_point.angle_between_points(&middle_point,&end_point),Some(1.5707963267948966));
    /// ```
    pub fn angle_between_points(
        &self,
        middle_point: &Vector3D,
        end_point: &Vector3D,
    ) -> Option<f64> {
        let v1 = *middle_point - *self;
        let v2 = *end_point - *middle_point;
        v1.angle_between(&v2)
    }
    /// Calculates the triple product of three vectors
    ///
    /// # Example
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let start_point = Vector3D::new(1.0,0.0,0.0);
    /// let middle_point = Vector3D::new(0.0,1.0,0.0);
    /// let end_point = Vector3D::new(0.0,0.0,1.0);
    /// // Should be ~1 (right handed)
    /// let triple_product = start_point.triple_product(&middle_point, &end_point);
    /// assert!(triple_product - 1.0 < 0.01);
    ///
    /// let start_point = Vector3D::new(1.0,0.0,0.0);
    /// let middle_point = Vector3D::new(0.0,0.0,1.0);
    /// let end_point = Vector3D::new(0.0,1.0,0.0);
    ///
    /// // Should be ~ -1 (right handed)
    /// let triple_product = start_point.triple_product(&middle_point, &end_point);
    /// assert!(triple_product + 1.0 < 0.01);
    /// ```
    pub fn triple_product(&self, other1: &Vector3D, other2: &Vector3D) -> f64 {
        self.dot(&other1.cross(other2))
    }
    pub fn as_tuple(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }
    pub fn as_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
    pub fn as_vec(&self) -> Vec<f64> {
        vec![self.x, self.y, self.z]
    }

    /// Determines if two vectors are pointing in opposite directions within a given tolerance.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector to compare with.
    /// * `tolerance` - The tolerance for the comparison. Smaller values require more precise opposition.
    ///
    /// # Returns
    //
    /// * `bool` - True if the vectors are opposite within the given tolerance, false otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v1 = Vector3D::new(1.0, 0.0, 0.0);
    /// let v2 = Vector3D::new(-1.0, 0.0, 0.0);
    /// assert!(v1.is_opposite(&v2, 1e-10));
    ///
    /// let v3 = Vector3D::new(-0.99, 0.01, 0.0);
    /// assert!(!v1.is_opposite(&v3, 1e-10));
    /// assert!(v1.is_opposite(&v3, 0.02));
    /// ```
    pub fn is_opposite(&self, other: &Vector3D, tolerance: f64) -> bool {
        let dot_product = self.dot(other);
        let magnitudes_product = self.length() * other.length();

        if magnitudes_product == 0.0 {
            false // At least one vector is zero, so they can't be opposite
        } else {
            let cos_angle = dot_product / magnitudes_product;
            (cos_angle + 1.0).abs() < tolerance
        }
    }

    /// Projects this vector onto another vector.
    ///
    /// # Arguments
    ///
    /// * `other` - The Vector to project onto.
    ///
    /// # Returns
    ///
    /// * `Vector3D` - The projection of this vector onto `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v1 = Vector3D::new(3.0, 4.0, 0.0);
    /// let v2 = Vector3D::new(5.0, 0.0, 0.0);
    /// let projection = v1.project_onto(&v2);
    /// assert_eq!(projection, Vector3D::new(3.0, 0.0, 0.0));
    /// ```
    pub fn project_onto(&self, other: &Vector3D) -> Vector3D {
        let other_normalized = other.normalize();
        other_normalized * self.dot(&other_normalized)
    }

    /// Rotates the vector around an axis by a given angle
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to rotate around.
    /// * `angle` - The angle to rotate by in radians.
    ///
    /// # Returns
    ///
    /// * `Vector3D` - The rotated vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v = Vector3D::new(1.0, 0.0, 0.0);
    /// let rotated = v.rotate_around_axis(&Vector3D::new(0.0, 0.0, 1.0), std::f64::consts::PI / 2.0);
    /// let delta = rotated - Vector3D::new(0.0, 1.0, 0.0);
    /// assert!(delta.length() < 0.001);
    /// ```
    pub fn rotate_around_axis(&self, axis: &Vector3D, angle: f64) -> Vector3D {
        let axis = axis.normalize();
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();
        let rotation_matrix = [
            [
                cos_theta + axis.x * axis.x * (1.0 - cos_theta),
                axis.x * axis.y * (1.0 - cos_theta) - axis.z * sin_theta,
                axis.x * axis.z * (1.0 - cos_theta) + axis.y * sin_theta,
            ],
            [
                axis.x * axis.y * (1.0 - cos_theta) + axis.z * sin_theta,
                cos_theta + axis.y * axis.y * (1.0 - cos_theta),
                axis.y * axis.z * (1.0 - cos_theta) - axis.x * sin_theta,
            ],
            [
                axis.x * axis.z * (1.0 - cos_theta) - axis.y * sin_theta,
                axis.y * axis.z * (1.0 - cos_theta) + axis.x * sin_theta,
                cos_theta + axis.z * axis.z * (1.0 - cos_theta),
            ],
        ];

        let rotated_vector = Vector3D {
            x: self.x * rotation_matrix[0][0]
                + self.y * rotation_matrix[0][1]
                + self.z * rotation_matrix[0][2],
            y: self.x * rotation_matrix[1][0]
                + self.y * rotation_matrix[1][1]
                + self.z * rotation_matrix[1][2],
            z: self.x * rotation_matrix[2][0]
                + self.y * rotation_matrix[2][1]
                + self.z * rotation_matrix[2][2],
        };

        rotated_vector
    }

    pub fn rotate_around_x(&self, angle: f64) -> Vector3D {
        self.rotate_around_axis(&Vector3D::unit_x(), angle)
    }

    pub fn rotate_around_y(&self, angle: f64) -> Vector3D {
        self.rotate_around_axis(&Vector3D::unit_y(), angle)
    }

    pub fn rotate_around_z(&self, angle: f64) -> Vector3D {
        self.rotate_around_axis(&Vector3D::unit_z(), angle)
    }

    pub fn rotate_around(
        &self,
        center: Vector3D,
        x_angle: f64,
        y_angle: f64,
        z_angle: f64,
    ) -> Vector3D {
        let translated = *self - center;
        let rotated = translated
            .rotate_around_x(x_angle)
            .rotate_around_y(y_angle)
            .rotate_around_z(z_angle);
        rotated + center
    }

    /// Performs element-wise multiplication of two vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The vector to multiply with element-wise.
    ///
    /// # Returns
    ///
    /// * `Vector3D` - A new vector with element-wise multiplication results.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v1 = Vector3D::new(1.0, 2.0, 3.0);
    /// let v2 = Vector3D::new(2.0, 3.0, 4.0);
    /// let result = v1.element_wise_mul(&v2);
    /// assert_eq!(result, Vector3D::new(2.0, 6.0, 12.0));
    /// ```
    pub fn element_wise_mul(&self, other: &Vector3D) -> Vector3D {
        Vector3D {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    /// Reflects the vector about a normal vector.
    ///
    /// # Arguments
    ///
    /// * `normal` - The normal vector to reflect about.
    ///
    /// # Returns
    ///
    /// * `Vector3D` - The reflected vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v = Vector3D::new(1.0, -1.0, 0.0);
    /// let normal = Vector3D::new(0.0, 1.0, 0.0);
    /// let reflected = v.reflect(&normal);
    /// assert_eq!(reflected, Vector3D::new(1.0, 1.0, 0.0));
    /// ```
    pub fn reflect(&self, normal: &Vector3D) -> Vector3D {
        *self - *normal * (2.0 * self.dot(normal))
    }

    /// Linearly interpolates between two vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The vector to interpolate towards.
    /// * `t` - The interpolation parameter (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// * `Vector3D` - The interpolated vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v1 = Vector3D::new(0.0, 0.0, 0.0);
    /// let v2 = Vector3D::new(10.0, 10.0, 10.0);
    /// let interpolated = v1.lerp(&v2, 0.5);
    /// assert_eq!(interpolated, Vector3D::new(5.0, 5.0, 5.0));
    /// ```
    pub fn lerp(&self, other: &Vector3D, t: f64) -> Vector3D {
        *self + (*other - *self) * t
    }
}

// Implement the Neg trait for Vector
impl Neg for Vector3D {
    type Output = Vector3D;

    /// Negates the vector, reversing its direction.
    ///
    /// # Returns
    ///
    /// * `Vector3D` - A new Vector3D with all components negated.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::vector::Vector3D;
    ///
    /// let v = Vector3D::new(1.0, -2.0, 3.0);
    /// let negated = -v;
    /// assert_eq!(negated, Vector3D::new(-1.0, 2.0, -3.0));
    /// ```
    fn neg(self) -> Self::Output {
        Vector3D {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Display for Vector3D {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-10;

    fn assert_vector_eq(v1: &Vector3D, v2: &Vector3D) {
        assert!(
            (*v1 - *v2).length() < EPSILON,
            "Vectors are not equal within epsilon: {:?} != {:?}",
            v1,
            v2
        );
    }

    #[test]
    fn test_vector_creation() {
        let v = Vector3D::new(1.0, 2.0, 3.0);
        assert_vector_eq(&v, &Vector3D::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vector_addition() {
        let v1 = Vector3D::new(1.0, 2.0, 3.0);
        let v2 = Vector3D::new(4.0, 5.0, 6.0);
        let result = v1 + v2;
        assert_vector_eq(&result, &Vector3D::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vector_subtraction() {
        let v1 = Vector3D::new(4.0, 5.0, 6.0);
        let v2 = Vector3D::new(1.0, 2.0, 3.0);
        let result = v1 - v2;
        assert_vector_eq(&result, &Vector3D::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_vector_scalar_multiplication() {
        let v = Vector3D::new(1.0, 2.0, 3.0);
        let result = v * 2.0;
        assert_vector_eq(&result, &Vector3D::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vector_scalar_division() {
        let v = Vector3D::new(2.0, 4.0, 6.0);
        let result = v / 2.0;
        assert_vector_eq(&result, &Vector3D::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vector_negation() {
        let v = Vector3D::new(1.0, -2.0, 3.0);
        let result = -v;
        assert_vector_eq(&result, &Vector3D::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn test_vector_dot_product() {
        let v1 = Vector3D::new(1.0, 2.0, 3.0);
        let v2 = Vector3D::new(4.0, 5.0, 6.0);
        let result = v1.dot(&v2);
        assert!((result - 32.0).abs() < EPSILON);
    }

    #[test]
    fn test_vector_cross_product() {
        let v1 = Vector3D::new(1.0, 2.0, 3.0);
        let v2 = Vector3D::new(4.0, 5.0, 6.0);
        let result = v1.cross(&v2);
        assert_vector_eq(&result, &Vector3D::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn test_vector_length() {
        let v = Vector3D::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_vector_normalize() {
        let v = Vector3D::new(3.0, 4.0, 0.0);
        let normalized = v.normalize();
        assert!((normalized.length() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_vector_distance() {
        let v1 = Vector3D::new(1.0, 2.0, 3.0);
        let v2 = Vector3D::new(4.0, 6.0, 8.0);
        assert!((v1.distance(&v2) - 7.0710678118654755).abs() < EPSILON);
    }

    #[test]
    fn test_vector_angle_between() {
        let v1 = Vector3D::new(1.0, 0.0, 0.0);
        let v2 = Vector3D::new(0.0, 1.0, 0.0);
        assert!((v1.angle_between(&v2).unwrap() - PI / 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_vector_project_onto() {
        let v1 = Vector3D::new(3.0, 4.0, 0.0);
        let v2 = Vector3D::new(5.0, 0.0, 0.0);
        let projection = v1.project_onto(&v2);
        assert_vector_eq(&projection, &Vector3D::new(3.0, 0.0, 0.0));
    }

    #[test]
    fn test_vector_rotate_around_axis() {
        let v = Vector3D::new(1.0, 0.0, 0.0);
        let axis = Vector3D::new(0.0, 0.0, 1.0);
        let rotated = v.rotate_around_axis(&axis, PI / 2.0);
        assert_vector_eq(&rotated, &Vector3D::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn test_vector_element_wise_mul() {
        let v1 = Vector3D::new(1.0, 2.0, 3.0);
        let v2 = Vector3D::new(2.0, 3.0, 4.0);
        let result = v1.element_wise_mul(&v2);
        assert_vector_eq(&result, &Vector3D::new(2.0, 6.0, 12.0));
    }

    #[test]
    fn test_vector_reflect() {
        let v = Vector3D::new(1.0, -1.0, 0.0);
        let normal = Vector3D::new(0.0, 1.0, 0.0);
        let reflected = v.reflect(&normal);
        assert_vector_eq(&reflected, &Vector3D::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn test_vector_lerp() {
        let v1 = Vector3D::new(0.0, 0.0, 0.0);
        let v2 = Vector3D::new(10.0, 10.0, 10.0);
        let interpolated = v1.lerp(&v2, 0.5);
        assert_vector_eq(&interpolated, &Vector3D::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn test_vector_default() {
        let v = Vector3D::default();
        assert_vector_eq(&v, &Vector3D::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_vector_from_vec() {
        let array = [1.0, 2.0, 3.0];
        let v = Vector3D::from_vec(&array);
        assert_vector_eq(&v, &Vector3D::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vector_as_tuple() {
        let v = Vector3D::new(1.0, 2.0, 3.0);
        assert_eq!(v.as_tuple(), (1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vector_as_array() {
        let v = Vector3D::new(1.0, 2.0, 3.0);
        assert_eq!(v.as_array(), [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_as_vec() {
        let v = Vector3D::new(1.0, 2.0, 3.0);
        assert_eq!(v.as_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_is_opposite() {
        let v1 = Vector3D::new(1.0, 0.0, 0.0);
        let v2 = Vector3D::new(-1.0, 0.0, 0.0);
        assert!(v1.is_opposite(&v2, EPSILON));

        let v3 = Vector3D::new(-0.99, 0.01, 0.0);
        assert!(!v1.is_opposite(&v3, EPSILON));
        assert!(v1.is_opposite(&v3, 0.02));
    }

    #[test]
    fn test_vector_are_vectors_parallel() {
        let v1 = Vector3D::new(1.0, 0.0, 0.0);
        let v2 = Vector3D::new(2.0, 0.0, 0.0);
        assert!(v1.are_vectors_parallel(&v2, EPSILON));

        let v3 = Vector3D::new(1.0, 0.1, 0.0);
        assert!(!v1.are_vectors_parallel(&v3, EPSILON));
    }

    #[test]
    fn test_vector_triple_product() {
        let v1 = Vector3D::new(1.0, 0.0, 0.0);
        let v2 = Vector3D::new(0.0, 1.0, 0.0);
        let v3 = Vector3D::new(0.0, 0.0, 1.0);
        assert!((v1.triple_product(&v2, &v3) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_vector_rotate_around() {
        let v = Vector3D::new(1.0, 0.0, 0.0);
        let center = Vector3D::new(0.0, 0.0, 0.0);
        let rotated = v.rotate_around(center, 0.0, 0.0, PI / 2.0);
        assert_vector_eq(&rotated, &Vector3D::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn test_zero_vector_operations() {
        let v = Vector3D::default();
        let another = Vector3D::new(0.0, 0.0, 0.0);
        assert_eq!(v + another, Vector3D::new(0.0, 0.0, 0.0));
        assert_eq!(v * 5.0, Vector3D::new(0.0, 0.0, 0.0));
        assert_eq!(v.dot(&another), 0.0);
        assert_eq!(v.length(), 0.0);
        assert_eq!(v.angle_between(&another), None);
    }

    #[test]
    fn test_normalized_vector_operations() {
        let v = Vector3D::new(3.0, 0.0, 0.0).normalize();
        assert!((v.length() - 1.0).abs() < EPSILON);
        let projected = Vector3D::new(1.0, 1.0, 0.0).project_onto(&v);
        assert_eq!(projected, Vector3D::new(1.0, 0.0, 0.0));
    }
}

impl From<[f64; 3]> for Vector3D {
    fn from(array: [f64; 3]) -> Self {
        Vector3D::from_vec(&array)
    }
}

impl From<(f64, f64, f64)> for Vector3D {
    fn from(tuple: (f64, f64, f64)) -> Self {
        Vector3D::new(tuple.0, tuple.1, tuple.2)
    }
}

/// A point in four dimensions: three real ones and the temporary fourth the
/// embedder uses to get chirality right.
///
/// Distance geometry in four dimensions is the classic trick, and it is the
/// one thing `embed` has always documented as missing. A stereocentre that
/// comes out of the initial embedding with the wrong handedness cannot be
/// fixed in three dimensions without dragging a substituent straight through
/// the other three -- the barrier is exactly the strain the refinement is
/// trying to remove, so the centre stays wrong, or it goes right and the
/// geometry around it stays bad. In four dimensions the same motion is a
/// rotation through `w` with nothing in the way: the chirality term costs
/// nothing to satisfy while `w` is free, and the structure is then squeezed
/// back down to three dimensions with the handedness already correct.
///
/// Not a general-purpose vector type. It has what the
/// minimiser needs and the projection back to three dimensions, and nothing
/// else -- everything else in this crate is three-dimensional.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Vector4D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vector4D {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Vector4D { x, y, z, w }
    }

    /// The three real dimensions, dropping `w`.
    pub fn xyz(&self) -> Vector3D {
        Vector3D::new(self.x, self.y, self.z)
    }

    /// `xyz` from a [`Vector3D`], with `w` alongside.
    pub fn from_xyz(point: Vector3D, w: f64) -> Self {
        Vector4D::new(point.x, point.y, point.z, w)
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl Add for Vector4D {
    type Output = Vector4D;
    fn add(self, other: Vector4D) -> Vector4D {
        Vector4D::new(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)
    }
}

impl Sub for Vector4D {
    type Output = Vector4D;
    fn sub(self, other: Vector4D) -> Vector4D {
        Vector4D::new(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)
    }
}

impl Mul<f64> for Vector4D {
    type Output = Vector4D;
    fn mul(self, scalar: f64) -> Vector4D {
        Vector4D::new(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)
    }
}

impl Neg for Vector4D {
    type Output = Vector4D;
    fn neg(self) -> Vector4D {
        Vector4D::new(-self.x, -self.y, -self.z, -self.w)
    }
}

impl AddAssign for Vector4D {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}
