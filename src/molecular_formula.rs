use crate::molecule::traits::*;
use chemistry_consts::ElementProperties;
use core::{
    fmt::{Display, Formatter},
    str::FromStr,
};
use nohash_hasher::IntMap;
use rustfft::{FftPlanner, num_complex::Complex64};

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
    /// # Arguments
    /// * `max_mass_difference` - The maximum mass difference between the monoisotopic peak and the
    /// last peak in the isotopic pattern to be calculated.
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
    /// Calculates the isotopic pattern of the molecular formula.
    pub fn isotopic_pattern(
        &self,
        max_mass_difference: usize,
        resolution: usize,
    ) -> Vec<(f64, f64)> {
        // 1. Determine safe FFT array length to prevent circular aliasing.
        // We need the absolute maximum possible mass of the molecule to size the array.
        let mut max_possible_mass = 0.0;
        for (atomic_number, count) in self.elements.iter() {
            let Some(isotopes) = atomic_number.isotopes() else {
                continue;
            };
            let heaviest_isotope = isotopes.map(|i| i.mass).fold(0.0, f64::max);
            max_possible_mass += heaviest_isotope * (*count as f64);
        }

        let required_length = (max_possible_mass * resolution as f64).ceil() as usize + 1;
        let distribution_length = required_length.next_power_of_two();

        // 2. Setup FFT Planner and Buffers (Allocating exactly once!)
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(distribution_length);
        let ifft = planner.plan_fft_inverse(distribution_length);

        // total_fft starts as the identity for multiplication: 1.0 + 0.0i
        let mut total_fft = vec![Complex64::new(1.0, 0.0); distribution_length];
        let mut atom_buffer = vec![Complex64::new(0.0, 0.0); distribution_length];

        // 3. Process each UNIQUE element
        for (atomic_number, count) in self.elements.iter() {
            let Some(isotopes) = atomic_number.isotopes() else {
                continue;
            };

            // Reset atom buffer for the new element
            atom_buffer.fill(Complex64::new(0.0, 0.0));

            for isotope in isotopes {
                let idx = (isotope.mass * resolution as f64).round() as usize;
                if idx < distribution_length {
                    atom_buffer[idx].re = isotope.abundance;
                }
            }

            // Forward FFT for this single element
            fft.process(&mut atom_buffer);

            // Raise to power of atom count and multiply into total
            let count_f64 = *count as f64;
            for (t, a) in total_fft.iter_mut().zip(atom_buffer.iter()) {
                // Complex exponentiation: a^count
                // Using polar form: r^n * e^(i * theta * n)
                let (r, theta) = a.to_polar();
                let a_pow = Complex64::from_polar(r.powf(count_f64), theta * count_f64);
                *t *= a_pow;
            }
        }

        // 4. Inverse FFT the combined result (just once!)
        ifft.process(&mut total_fft);

        // Standard IFFT scaling factor (1/N)
        let scale = 1.0 / distribution_length as f64;

        // 5. Extract the window of interest and normalize
        let min_mass_idx = (self.monoisotopic_mass().round() as usize) * resolution;
        let max_mass_idx = min_mass_idx + (max_mass_difference * resolution);

        let mut result = Vec::with_capacity(max_mass_idx - min_mass_idx);
        let mut sum = 0.0;

        for i in min_mass_idx..max_mass_idx {
            if i < distribution_length {
                // Extract real part, apply scale.
                // .max(0.0) cleans up tiny negative floating-point artifacts from the FFT.
                let val = (total_fft[i].re * scale).max(0.0);
                result.push((i as f64 / resolution as f64, val));
                sum += val;
            }
        }

        // Normalize so the extracted window sums to 1.0
        if sum > 0.0 {
            for (_, abundance) in result.iter_mut() {
                *abundance /= sum;
            }
        }

        result
    }
}
