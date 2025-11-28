use crate::{
    molecule::bond::{BondOrder, BondTarget},
    molecule::chirality::ChiralClass,
    vector::Vector3D,
};
use chemistry_consts::ElementProperties;
use tinyvec::ArrayVec;

/// How many bonds an atom's inline bond list holds before it overflows.
///
/// Past every coordination number this crate models, so exceeding it means the
/// input is malformed rather than exotic.
pub const BOND_CAPACITY: usize = 10;

/// Atomic number standing for the SMILES wildcard atom `*` — an atom whose
/// element is deliberately left unspecified, as used for attachment points,
/// polymer end caps and Markush-style fragments.
///
/// Zero is not a real element, which is exactly the point: it is the
/// conventional encoding (RDKit uses it too), it sorts below every real
/// element wherever atomic number is compared, and it has no valence, so no
/// implicit hydrogens are ever added to it.
pub const WILDCARD_ATOMIC_NUMBER: u8 = 0;

/// The symbol a wildcard atom is written with.
pub const WILDCARD_SYMBOL: &str = "*";

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Atom {
    pub atomic_number: u8,
    pub charge: i8,
    pub radical_state: u8,
    pub explicit_hydrogens: usize,
    pub aromatic: bool,
    pub isotope: Option<u16>,
    pub chiral_class: ChiralClass,
    pub atom_class: Option<usize>,
    pub position_vector: Option<Vector3D>,
    pub partial_charge: Option<f64>,
    pub bonds: ArrayVec<[BondTarget; BOND_CAPACITY]>,
    pub is_complex: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum XYZParserError {
    IOError,
    InvalidFloat(String),
    InvalidAtom(String),
    InvalidBond,
}

impl Atom {
    pub fn new(atomic_number: u8) -> Atom {
        Atom {
            atomic_number,
            ..Default::default()
        }
    }

    pub fn isotope(&self) -> Option<u16> {
        self.isotope
    }

    pub fn with_chiral_class(mut self, chiral_class: ChiralClass) -> Self {
        self.chiral_class = chiral_class;
        self
    }

    pub fn with_position(mut self, position: (f64, f64, f64)) -> Self {
        self.position_vector = Some(Vector3D::new(position.0, position.1, position.2));
        self
    }

    pub fn with_charge(mut self, charge: i8) -> Self {
        self.charge = charge;
        self
    }

    pub fn with_isotope(mut self, isotope: u16) -> Self {
        self.isotope = Some(isotope);
        self
    }

    pub fn with_atom_class(mut self, atom_class: Option<usize>) -> Self {
        self.atom_class = atom_class;
        self
    }

    pub fn with_radical(mut self, radical_state: u8) -> Self {
        self.radical_state = radical_state;
        self
    }

    pub fn with_bonds(mut self, bonds: ArrayVec<[BondTarget; 10]>) -> Self {
        self.bonds = bonds;
        self
    }

    pub fn with_is_complex(mut self, is_complex: bool) -> Self {
        self.is_complex = is_complex;
        self
    }

    pub fn from_xyz_line(line: &str) -> Result<Atom, XYZParserError> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 4 {
            let atomic_number = parts[0]
                .atomic_number()
                .ok_or(XYZParserError::InvalidAtom(parts[0].to_string()))?;
            let x = parts[1]
                .parse::<f64>()
                .map_err(|_| XYZParserError::InvalidFloat(String::from(parts[1])))?;
            let y = parts[2]
                .parse::<f64>()
                .map_err(|_| XYZParserError::InvalidFloat(String::from(parts[2])))?;
            let z = parts[3]
                .parse::<f64>()
                .map_err(|_| XYZParserError::InvalidFloat(String::from(parts[3])))?;
            Ok(Atom::new(atomic_number).with_position((x, y, z)))
        } else {
            Err(XYZParserError::InvalidAtom(line.to_string()))
        }
    }

    /// Returns the expected valency of the atom based on its atomic number
    ///
    /// # Examples
    /// ```
    /// use oxidane::prelude::*;
    /// let atom = Atom::new(6);
    /// assert_eq!(atom.expected_valency(), Some(4));
    /// ```
    pub fn expected_valency(&self) -> Option<i8> {
        self.atomic_number.valencies()?.next()
    }

    /// Compares the electronegativity of two atoms based on the Pauling scale.
    ///
    /// # Examples
    /// ```
    /// use oxidane::prelude::*;
    /// let atom1 = Atom::new(6);
    /// let atom2 = Atom::new(8);
    /// assert!(atom1.cmp_electronegativity(&atom2) == std::cmp::Ordering::Less);
    /// ```
    pub fn cmp_electronegativity(&self, other: &Self) -> std::cmp::Ordering {
        self.electronegativity().cmp(&other.electronegativity())
    }

    pub fn bonds(&self) -> &ArrayVec<[BondTarget; 10]> {
        &self.bonds
    }

    pub fn add_bond(&mut self, bond: BondTarget) {
        self.bonds.push(bond);
    }

    pub fn charge(&self) -> i8 {
        self.charge
    }

    pub fn is_aromatic(&self) -> bool {
        self.bonds
            .iter()
            .any(|bond| bond.bond_order() == BondOrder::Aromatic)
    }

    pub fn set_charge(&mut self, charge: i8) {
        self.charge = charge;
    }

    pub fn update_charge(&mut self, charge: i8) {
        self.charge += charge;
    }

    pub fn is_radical(&self) -> bool {
        self.radical_state > 0
    }

    pub fn set_radical(&mut self, radical_state: u8) {
        self.radical_state = radical_state;
    }

    pub fn atomic_symbol(&self) -> Option<&str> {
        self.atomic_number.atomic_symbol()
    }

    pub fn electronegativity(&self) -> Option<u16> {
        self.atomic_number.electronegativity()
    }

    pub fn is_more_electronegative(&self, other: &Self) -> bool {
        self.electronegativity() > other.electronegativity()
    }

    pub fn monoisotopic_mass(&self) -> Option<f64> {
        self.atomic_number.monoisotopic_mass()
    }

    pub fn is_complex(&self) -> bool {
        self.is_complex
    }

    /// Calculates the distance between two atoms.
    ///
    /// # Arguments
    /// * 'self' - A reference to the current atom.
    /// * 'other' - A reference to the other atom.
    ///
    /// # Returns
    /// The distance between the two atoms as a 64-bit floating-point number.
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let atom1 = Atom::new(6).with_position((0.0,0.0,0.0));
    /// let atom2 = Atom::new(6).with_position((1.0,0.0,0.0));
    /// assert_eq!(atom1.distance(&atom2),1.0);
    /// ```
    pub fn distance(&self, other: &Atom) -> f64 {
        self.position_vector
            .zip(other.position_vector)
            .map_or(0.0, |(v1, v2)| v1.distance(&v2))
    }

    /// Calculates the squared distance between two atoms saving computation time.
    ///
    /// # Example
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// let atom1 = Atom::new(6).with_position((0.0,0.0,0.0));
    /// let atom2 = Atom::new(6).with_position((1.0,0.0,0.0));
    /// assert_eq!(atom1.distance_squared(&atom2),1.0);
    /// ```
    ///
    pub fn distance_squared(&self, other: &Atom) -> f64 {
        let Some(self_position_vector) = self.position_vector else {
            return 0.0;
        };
        let Some(other_position_vector) = other.position_vector else {
            return 0.0;
        };
        self_position_vector.distance_squared(&other_position_vector)
    }

    /// Calculates the potential energy of a diatomic system using the harmonic oscillator model.
    ///
    /// # Parameters
    /// * 'other' - A reference to the other atom.
    /// * 'force_constant' - The force constant (k) of the diatomic system.
    /// * 'equilibrium_distance' - The equilibrium distance between the two atoms.
    ///
    /// # Returns
    /// The potential energy of the diatomic system as a 64-bit floating-point number.
    ///
    /// # Example
    /// ```
    /// use oxidane::prelude::*;
    /// let atom1 = Atom::new(6).with_position((0.0,0.0,0.0));
    /// let atom2 = Atom::new(6).with_position((1.0,0.0,0.0));
    /// assert_eq!(atom1.potential_energy(&atom2, 1.0, 1.0),0.0);
    /// ```
    pub fn potential_energy(
        &self,
        other: &Atom,
        force_constant: f64,
        equilibrium_distance: f64,
    ) -> f64 {
        0.5 * force_constant * (self.distance(other) - equilibrium_distance).powi(2)
    }

    pub fn standard_atomic_weight(&self) -> Option<f64> {
        self.atomic_number.standard_atomic_weight()
    }

    pub fn atomic_number(&self) -> u8 {
        self.atomic_number
    }
    pub fn degree(&self) -> Option<i8> {
        let atomic_number = self.atomic_number;
        let expected_valency = atomic_number.valencies()?.next()?;
        let actual_valency = self.actual_valency();
        // May need to be changed for elements with unknown valencies
        Some(actual_valency - expected_valency)
    }
    /// The valence the atom's bonds alone account for.
    ///
    /// Distinct from [`Atom::actual_valency`], which also folds in the radical
    /// state and `|charge|`. Mixing the two is what made the radical count and
    /// the hydrogen count disagree: each was correcting for the charge in its
    /// own way, once too often between them.
    pub fn bond_valency(&self) -> i8 {
        self.bonds
            .iter()
            .map(|bond| match bond.bond_order() {
                BondOrder::Single | BondOrder::Up | BondOrder::Down => 2,
                BondOrder::Double => 4,
                BondOrder::Triple => 6,
                BondOrder::Quadruple => 8,
                BondOrder::Aromatic => 3,
                BondOrder::Coordinate => 2,
            })
            .sum::<i8>()
            / 2
    }

    pub fn actual_valency(&self) -> i8 {
        self.bonds
            .iter()
            .map(|bond| match bond.bond_order() {
                BondOrder::Single => 2,
                BondOrder::Up => 2,
                BondOrder::Down => 2,
                BondOrder::Double => 4,
                BondOrder::Triple => 6,
                BondOrder::Quadruple => 8,
                BondOrder::Aromatic => 3,
                BondOrder::Coordinate => 2,
            })
            .sum::<i8>()
            / 2
            + self.radical_state as i8
            + self.charge.abs()
    }
}
