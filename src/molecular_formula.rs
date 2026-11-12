use crate::molecule::traits::*;
use chemistry_consts::ElementProperties;
use core::{
    fmt::{Display, Formatter},
    str::FromStr,
};
use nohash_hasher::IntMap;
use crate::molecule::forcefield::lanes::{self, LANES, Lanes, LanesOp};

// TODO: make this a parameter
const PRECISION: f64 = 1000.0;

#[derive(Clone, Debug, PartialEq)]
pub struct MolecularFormula {
    elements: IntMap<u8, usize>,
}

impl Default for MolecularFormula {
    fn default() -> Self {
        Self::new(IntMap::default())
    }
}

impl Display for MolecularFormula {
    // TODO make this more efficient
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut elements = self
            .elements
            .iter()
            .map(|(&atom, &count)| (atom.atomic_symbol().unwrap().into(), count))
            .collect::<Vec<(String, usize)>>();
        elements.sort_by(|(atom1, _), (atom2, _)| atom1.cmp(atom2));
        for (atom, count) in elements {
            write!(f, "{}{}", atom, count)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct ParseFormulaError;

impl std::fmt::Display for ParseFormulaError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "could not parse the provided string as a molecular formula"
        )
    }
}

impl std::error::Error for ParseFormulaError {}

impl FromStr for MolecularFormula {
    type Err = ParseFormulaError;

    /// Parses a flat molecular formula such as `C6H12O6` or `CH3COOH`.
    /// Element symbols are case-sensitive; repeated elements are summed.
    ///
    /// # Errors
    /// Returns [`ParseFormulaError`] for empty input, unknown elements, zero or
    /// overflowing counts, or unsupported syntax (including parentheses, isotope
    /// labels, charges, whitespace, and hydrate separators).
    ///
    /// # Examples
    /// ```
    /// use oxidane::molecular_formula::MolecularFormula;
    /// let water = "H2O".parse::<MolecularFormula>().unwrap();
    /// let methane = "CH4".parse::<MolecularFormula>().unwrap();
    /// assert_eq!(water.monoisotopic_mass(), 18.01056468403);
    /// assert_eq!(methane.monoisotopic_mass(), 16.03130012892);
    ///
    /// ```

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Flat formulas only: Element [positive count], repeated as needed.
        // Reject unsupported syntax rather than silently changing its meaning.
        let bytes = s.as_bytes();
        let mut cursor = 0;
        let mut elements: IntMap<u8, usize> = IntMap::default();
        if bytes.is_empty() {
            return Err(ParseFormulaError);
        }
        while cursor < bytes.len() {
            let start = cursor;
            if !bytes[cursor].is_ascii_uppercase() {
                return Err(ParseFormulaError);
            }
            cursor += 1;
            if cursor < bytes.len() && bytes[cursor].is_ascii_lowercase() {
                cursor += 1;
            }
            // Only ASCII bytes have been consumed, so these are UTF-8 boundaries.
            let atomic_number = (&s[start..cursor]).atomic_number().ok_or(ParseFormulaError)?;
            let count_start = cursor;
            let mut count = 0usize;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                count = count.checked_mul(10)
                    .and_then(|n| n.checked_add((bytes[cursor] - b'0') as usize))
                    .ok_or(ParseFormulaError)?;
                cursor += 1;
            }
            if cursor == count_start {
                count = 1;
            } else if count == 0 {
                return Err(ParseFormulaError);
            }
            let total = elements.entry(atomic_number).or_insert(0);
            *total = total.checked_add(count).ok_or(ParseFormulaError)?;
        }
        Ok(MolecularFormula { elements })
    }
}

#[derive(Debug, Clone, Default)]
pub struct IsotopicPattern {
    pub buckets: IntMap<usize, (f64, f64)>,
}

impl IsotopicPattern {
    pub fn new(abundances: &[f64], masses: &[f64]) -> Self {
        let mut buckets = IntMap::default();
        for (&intensity, &mass) in abundances.iter().zip(masses.iter()) {
            let mass_index = (PRECISION * mass).round() as usize;
            match buckets.entry(mass_index) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    let bucket: &mut (f64, f64) = entry.get_mut();
                    let mass_sum = mass * intensity + bucket.0 * bucket.1;
                    let intensity_sum = intensity + bucket.1;
                    bucket.0 = mass_sum / intensity_sum;
                    bucket.1 += intensity;
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((mass, intensity));
                }
            }
        }
        IsotopicPattern { buckets }
    }

    pub fn to_vec(&self) -> Vec<(f64, f64)> {
        let mut vec = self.buckets.values().cloned().collect::<Vec<_>>();
        vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        vec
    }

    /// Merge two isotopic patterns into one.
    ///
    /// # Examples
    /// ```
    /// use oxidane::molecular_formula::IsotopicPattern;
    /// let mut pattern = IsotopicPattern::new(&vec![0.5, 0.5], &vec![1.0, 2.0]);
    /// let other = IsotopicPattern::new(&vec![0.5, 0.5], &vec![3.0, 4.0]);
    /// pattern.add(&other);
    /// // The two patterns share no mass bucket, so merging keeps all four.
    /// assert_eq!(pattern.buckets.len(), 4);
    ///
    /// // A shared bucket keeps one entry, and its abundances add.
    /// let mut pattern = IsotopicPattern::new(&vec![0.5], &vec![1.0]);
    /// pattern.add(&IsotopicPattern::new(&vec![0.25], &vec![1.0]));
    /// assert_eq!(pattern.buckets.len(), 1);
    /// assert!((pattern.buckets.values().next().unwrap().1 - 0.75).abs() < 1e-12);
    /// ```
    pub fn add(&mut self, other: &IsotopicPattern) {
        for (mass_index, &(mass, abundance)) in other.buckets.iter() {
            let bucket = self.buckets.entry(*mass_index).or_insert((mass, 0.0));
            let new_abundance = bucket.1 + abundance;
            bucket.0 = (bucket.0 * bucket.1 + mass * abundance) / new_abundance;
            bucket.1 = new_abundance;
        }
    }

    /// Add a single entry to the isotopic pattern, this will merge two buckets if they have the
    /// same mass
    ///
    /// # Examples
    /// ```
    /// use oxidane::molecular_formula::IsotopicPattern; let mut pattern = IsotopicPattern::new(&vec![0.5, 0.5], &vec![1.0, 2.0]);
    /// pattern.add_entry(1.0, 0.5);
    /// assert_eq!(pattern.buckets.len(), 2);
    /// ```
    ///
    ///
    pub fn add_entry(&mut self, mass: f64, abundance: f64) {
        let mass_index = (mass * PRECISION).round() as usize;
        let bucket = self.buckets.entry(mass_index).or_insert((mass, 0.0));
        let new_abundance = bucket.1 + abundance;
        bucket.0 = (bucket.0 * bucket.1 + mass * abundance) / new_abundance;
        bucket.1 = new_abundance;
    }

    pub fn normalize(&mut self) {
        let total_abundance: f64 = self.buckets.values().map(|&(_, a)| a).sum();
        for bucket in self.buckets.values_mut() {
            bucket.1 /= total_abundance;
        }
    }

    pub fn normalize_to_highest(&mut self) {
        let max_abundance: f64 = self
            .buckets
            .values()
            .map(|&(_, a)| a)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        for bucket in self.buckets.values_mut() {
            bucket.1 /= max_abundance;
        }
    }

    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    pub fn shift(&mut self, shift: f64) {
        let mut new_buckets = IntMap::default();
        for bucket in self.buckets.values_mut() {
            let mass = bucket.0 + shift;
            let mass_index = (PRECISION * mass).round() as usize;
            new_buckets.entry(mass_index).or_insert((mass, 0.0)).1 += bucket.1;
        }
        self.buckets = new_buckets;
    }
}

#[derive(Debug, Clone)]
pub struct Isotope {
    pub mass: f64,
    pub abundance: f64,
}

impl Isotope {
    pub const fn new(mass: f64, abundance: f64) -> Isotope {
        Isotope { mass, abundance }
    }
}

impl From<[f64; 2]> for Isotope {
    fn from(isotope: [f64; 2]) -> Self {
        Isotope {
            mass: isotope[0],
            abundance: isotope[1],
        }
    }
}

impl MolecularFormula {
    pub fn new(elements: IntMap<u8, usize>) -> Self {
        Self { elements }
    }

    pub fn from_molecule<T: Molecule>(molecule: &T) -> MolecularFormula {
        let mut formula = MolecularFormula::default();
        for &atomic_number in molecule.atomic_numbers() {
            *formula.elements.entry(atomic_number).or_insert(0) += 1;
        }
        formula
    }

    pub fn elements(&self) -> &IntMap<u8, usize> {
        &self.elements
    }
    /// Merges two molecular formulas.
    ///
    /// # Examples
    /// ```
    /// use oxidane::molecular_formula::MolecularFormula;
    /// let water = "H2O".parse::<MolecularFormula>().unwrap();
    /// let methane = "CH4".parse::<MolecularFormula>().unwrap();
    /// let mut water_methane = water.clone();
    /// water_methane.merge(&methane);
    /// assert_eq!(water_methane, "CH4H2O".parse::<MolecularFormula>().unwrap());
    /// assert_eq!(water, "H2O".parse::<MolecularFormula>().unwrap());
    /// assert_eq!(methane, "CH4".parse::<MolecularFormula>().unwrap());
    /// assert_eq!(water_methane, "CH4H2O".parse::<MolecularFormula>().unwrap());
    ///
    /// ```

    pub fn merge(&mut self, other: &MolecularFormula) {
        for (atom, count) in &other.elements {
            *self.elements.entry(*atom).or_insert(0) += count;
        }
    }

    /// Combines two molecular formulas.
    /// # Examples
    /// ```
    /// use oxidane::molecular_formula::MolecularFormula;
    /// let water = "H2O".parse::<MolecularFormula>().unwrap();
    /// let methane = "CH4".parse::<MolecularFormula>().unwrap();
    /// let water_methane = water.combine(&methane);
    /// assert_eq!(water_methane, "CH4H2O".parse::<MolecularFormula>().unwrap());
    /// ```
    pub fn combine(&self, other: &MolecularFormula) -> Self {
        let mut combined = MolecularFormula::default();
        for (atom, count) in &self.elements {
            *combined.elements.entry(*atom).or_insert(0) += count;
        }
        for (atom, count) in &other.elements {
            *combined.elements.entry(*atom).or_insert(0) += count;
        }
        combined
    }
    /// Calculates the monoisotopic mass of the molecular formula.
    ///
    /// # Examples
    /// ```
    /// use oxidane::molecular_formula::MolecularFormula;
    /// let water = "H2O".parse::<MolecularFormula>().unwrap();
    /// let methane = "CH4".parse::<MolecularFormula>().unwrap();
    /// println!("{:?}", water);
    /// assert_eq!(water.monoisotopic_mass(), 18.01056468403);
    /// assert_eq!(methane.monoisotopic_mass(), 16.03130012892);
    ///
    /// ```
    pub fn monoisotopic_mass(&self) -> f64 {
        self.elements.iter().fold(0.0, |acc, (atom, count)| {
            // TODO check if this is safe
            acc + atom.monoisotopic_mass().unwrap() * *count as f64
        })
    }

    /// Calculates the molecular mass of the molecular formula.
    /// # Examples
    /// ```
    /// use oxidane::molecular_formula::MolecularFormula;
    /// let water = "H2O".parse::<MolecularFormula>().unwrap();
    /// let methane = "CH4".parse::<MolecularFormula>().unwrap();
    /// assert_eq!(water.molecular_mass().round(), 18.0);
    /// assert_eq!(methane.molecular_mass().round(), 16.0);
    /// ```
    pub fn molecular_mass(&self) -> f64 {
        self.elements.iter().fold(0.0, |acc, (atom, count)| {
            // TODO check if this is safe
            acc + atom.standard_atomic_weight().unwrap() * *count as f64
        })
    }

    /// Calculates the isotopic pattern of the molecular formula.
    ///
    /// Returns `max_mass_difference * resolution` `(mass, abundance)` pairs,
    /// one per bin of `1 / resolution` daltons, starting at the monoisotopic
    /// mass rounded to whole daltons, with the abundances normalized to sum
    /// to one over that window.
    ///
    /// # Arguments
    /// * `max_mass_difference` - The maximum mass difference between the monoisotopic peak and the
    /// last peak in the isotopic pattern to be calculated.
    /// * `resolution` - Bins per dalton.
    /// # Examples
    /// ```
    /// use oxidane::molecular_formula::MolecularFormula;
    /// let water = "H2O".parse::<MolecularFormula>().unwrap();
    /// let isotopic_pattern_h2o = water.isotopic_pattern(3,1);
    ///
    /// println!("{:?}", isotopic_pattern_h2o);
    /// println!("{:?}", water);
    /// assert!(isotopic_pattern_h2o.len() == 3);
    ///
    /// let methane = "CH4".parse::<MolecularFormula>().unwrap();
    /// let isotopic_pattern_methane = methane.isotopic_pattern(3,1);
    /// println!("{:?}", isotopic_pattern_methane);
    /// assert!(isotopic_pattern_methane.len() == 3);
    /// ```
    pub fn isotopic_pattern(
        &self,
        max_mass_difference: usize,
        resolution: usize,
    ) -> Vec<(f64, f64)> {
        let resolution_f64 = resolution as f64;
        let bin = |mass: f64| (mass * resolution_f64).round() as usize;
        // The window to report, in mass bins.
        let window_start = (self.monoisotopic_mass().round() as usize) * resolution;
        let window_end = window_start + max_mass_difference * resolution;

        // Each element's isotopes as offsets from its lightest one, so every
        // offset is non-negative: a combination already past the window's
        // end stays past it whatever is added, and can be dropped. That keeps
        // every distribution below no longer than the window.
        //
        // Only the offsets are rounded to bins. The base -- every atom its
        // lightest isotope -- is binned from its exact mass: rounding each
        // atom's mass first would let the mass defects of thousands of atoms
        // add up to whole daltons and shift the pattern out of the window.
        let mut lightest_mass = 0.0;
        let mut elements = Vec::new();
        for (atomic_number, &count) in &self.elements {
            let Some(isotopes) = atomic_number.isotopes() else {
                continue;
            };
            let isotopes: Vec<(f64, f64)> =
                isotopes.map(|isotope| (isotope.mass, isotope.abundance)).collect();
            let Some(lightest) = isotopes.iter().map(|&(mass, _)| mass).min_by(f64::total_cmp) else {
                continue;
            };
            lightest_mass += lightest * count as f64;
            let offsets: Vec<(usize, f64)> = isotopes
                .iter()
                .map(|&(mass, abundance)| (bin(mass - lightest), abundance))
                .collect();
            elements.push((offsets, count));
        }
        let lightest_total = bin(lightest_mass);
        let length = window_end.saturating_sub(lightest_total);

        let mut total = vec![0.0; length.min(1)];
        if let Some(first) = total.first_mut() {
            *first = 1.0;
        }
        for (offsets, count) in elements {
            let mut single = Vec::new();
            for (offset, abundance) in offsets {
                if offset < length {
                    if single.len() <= offset {
                        single.resize(offset + 1, 0.0);
                    }
                    single[offset] += abundance;
                }
            }
            total = convolve(&total, &power(single, count, length), length);
        }

        let mut result = Vec::with_capacity(window_end - window_start);
        let mut sum = 0.0;
        for index in window_start..window_end {
            let abundance = index
                .checked_sub(lightest_total)
                .and_then(|offset| total.get(offset))
                .copied()
                .unwrap_or(0.0);
            result.push((index as f64 / resolution_f64, abundance));
            sum += abundance;
        }
        // Normalize so the window sums to 1.
        if sum > 0.0 {
            for (_, abundance) in result.iter_mut() {
                *abundance /= sum;
            }
        }
        result
    }
}

/// `left * right` as polynomials, keeping the first `length` coefficients.
///
/// Each nonzero coefficient of the shorter factor adds a scaled copy of the
/// other one, an `axpy` the force fields' lanes vectorize (see
/// [`lanes::dispatch`]); isotope distributions are short and mostly dense
/// once raised to a power, so this beats an FFT at every size the window
/// allows, and has no FFT round-off floor under the small peaks.
fn convolve(left: &[f64], right: &[f64], length: usize) -> Vec<f64> {
    let length = length.min((left.len() + right.len()).saturating_sub(1));
    let (short, long) = if left.len() <= right.len() { (left, right) } else { (right, left) };
    let mut out = vec![0.0; length];
    for (shift, &scale) in short.iter().enumerate().take(length) {
        if scale != 0.0 {
            let span = (length - shift).min(long.len());
            lanes::dispatch(Axpy { scale, x: &long[..span], y: &mut out[shift..shift + span] });
        }
    }
    out
}

/// `base` raised to `exponent` by repeated squaring, truncated to `length`.
fn power(base: Vec<f64>, mut exponent: usize, length: usize) -> Vec<f64> {
    let mut result = vec![1.0];
    let mut square = base;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = convolve(&result, &square, length);
        }
        exponent >>= 1;
        if exponent > 0 {
            square = convolve(&square, &square, length);
        }
    }
    result
}

/// `y += scale * x`.
struct Axpy<'a> {
    scale: f64,
    x: &'a [f64],
    y: &'a mut [f64],
}

impl LanesOp for Axpy<'_> {
    type Output = ();

    #[inline(always)]
    fn run<L: Lanes>(self, lanes: L) {
        let Axpy { scale, x, y } = self;
        let factor = lanes.splat(scale);
        let (x_chunks, x_tail) = x.as_chunks::<LANES>();
        let (y_chunks, y_tail) = y.as_chunks_mut::<LANES>();
        for (x, y) in x_chunks.iter().zip(y_chunks) {
            *y = lanes.to_array(lanes.from_array(*y) + factor * lanes.from_array(*x));
        }
        for (x, y) in x_tail.iter().zip(y_tail) {
            *y += scale * x;
        }
    }
}

#[cfg(test)]
mod isotope_tests {
    use super::*;

    fn assert_pattern(formula: &str, window: usize, first_mass: f64, expected: &[f64]) {
        let pattern = formula.parse::<MolecularFormula>().unwrap().isotopic_pattern(window, 1);
        assert_eq!(pattern.len(), expected.len());
        for (index, (&(mass, abundance), &reference)) in pattern.iter().zip(expected).enumerate() {
            assert_eq!(mass, first_mass + index as f64);
            assert!((abundance - reference).abs() < 1e-12, "{formula} +{index}: {abundance} vs {reference}");
        }
    }

    /// References from a separate model: the distribution over extra
    /// neutrons, from the same isotope table, with integer offsets.
    #[test]
    fn dichloromethane_has_the_chlorine_triplet() {
        assert_pattern(
            "CH2Cl2",
            5,
            84.0,
            &[0.5680505538965057, 0.006274547101542067, 0.3635057878877905, 0.004015180169481112, 0.05815393094468063],
        );
    }

    /// Insulin's mass defect is three daltons: binning each atom's mass
    /// before adding them up would label every peak three daltons light.
    #[test]
    fn insulin_peaks_three_daltons_above_the_monoisotopic_mass() {
        assert_pattern(
            "C254H377N65O75S6",
            6,
            5730.0,
            &[
                0.03826547851774699,
                0.11877459790773397,
                0.19991337185087416,
                0.23899653385760822,
                0.22575488321030582,
                0.17829513465573082,
            ],
        );
    }

    #[test]
    fn a_window_past_the_heaviest_mass_is_padded_with_zeros() {
        let pattern = "H2".parse::<MolecularFormula>().unwrap().isotopic_pattern(10, 1);
        assert_eq!(pattern.len(), 10);
        assert!(pattern[5..].iter().all(|&(_, abundance)| abundance == 0.0));
        assert!((pattern.iter().map(|p| p.1).sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn convolve_truncates_and_power_matches_repeated_products() {
        assert_eq!(convolve(&[1.0, 2.0], &[3.0, 4.0, 5.0], 3), vec![3.0, 10.0, 13.0]);
        assert_eq!(convolve(&[1.0, 2.0], &[3.0, 4.0, 5.0], 10), vec![3.0, 10.0, 13.0, 10.0]);
        let base = vec![0.5, 0.25, 0.25, 0.0, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125];
        let mut product = vec![1.0];
        for _ in 0..7 {
            product = convolve(&product, &base, 20);
        }
        let powered = power(base, 7, 20);
        assert_eq!(product.len(), powered.len());
        for (a, b) in product.iter().zip(&powered) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
