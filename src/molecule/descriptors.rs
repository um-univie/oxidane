use super::traits::MoleculeCore;
use crate::molecule::bond::BondOrder;
use chemistry_consts::ElementProperties;

pub trait MolecularDescriptors: MoleculeCore {
    /// Calculates the molecular weight of the molecule.
    ///
    /// This function calculates the molecular weight of the molecule by summing the standard atomic weights of all atoms in the molecule.
    ///
    /// # Returns
    /// The molecular weight of the molecule in grams per mole.
    ///
    /// # Examples
    /// ```
    /// use molecules::molecule::MolecularGraph;
    /// use molecules::molecule::descriptors::MolecularDescriptors;
    /// let molecule = MolecularGraph::from_smiles_single("CCO").unwrap();
    /// // Ethanol C2H6O; hydrogens are explicit atoms in the parsed graph
    /// assert!((molecule.molecular_weight() - 46.07).abs() < 0.05);
    /// ```
    fn molecular_weight(&self) -> f64 {
        let mut molecular_weight = 0.0;
        for atomic_number in self.atomic_numbers() {
            molecular_weight += atomic_number.standard_atomic_weight().unwrap_or(0.0);
        }
        molecular_weight
    }
    fn monoisotopic_mass(&self) -> f64 {
        let mut monoisotopic_mass = 0.0;
        for atomic_number in self.atomic_numbers() {
            monoisotopic_mass += atomic_number.monoisotopic_mass().unwrap_or(0.0);
        }
        monoisotopic_mass
    }
    /// This function calculates the number of hydrogen bond donors in the molecule.
    /// IT IS NOT YET STABLE
    fn number_of_hydrogen_bond_donors(&self) -> usize {
        let mut count = 0;
        for (index, &atomic_number) in self.atomic_numbers().iter().enumerate() {
            if atomic_number == 8 || atomic_number == 7 {
                if self.number_of_bonded_element(index, 1) > 0 {
                    count += 1;
                }
            }
        }
        count
    }

    /// This function calculates the number of hydrogen bond acceptors in the molecule.
    /// IT IS NOT YET STABLE
    fn number_of_hydrogen_bond_acceptors(&self) -> usize {
        let mut count = 0;
        for (index, &atomic_number) in self.atomic_numbers().iter().enumerate() {
            if atomic_number == 8 || atomic_number == 7 {
                if self.get_atom_charge(index) <= 0 {
                    count += 1;
                }
            }
        }
        count
    }
}

impl<T: MoleculeCore> MolecularDescriptors for T {}
