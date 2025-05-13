use super::aromaticity::AromaticityType;
use crate::molecule::{
    base::{MoleculeCore, cycle_to_bitset},
    bond::BondOrder,
    utils::is_hueckel_satisfied,
};
use bit_set::BitSet;
use bitvec::prelude::BitVec;
use rustc_hash::FxHashSet;

pub trait MoleculeExperimental: MoleculeCore {
    /// Kekulizes the molecule by adding double bonds to aromatic atoms
    ///
    /// # Errors
    ///
    /// Returns an error if the molecule cannot be kekulized
    ///
    /// # Examples
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// let molecule = MolecularGraph::from_smiles("c1ccccc1").unwrap();
    /// let mut molecule = molecule.first().unwrap().clone();
    /// molecule.kekulize().unwrap();
    /// assert_eq!(molecule.to_smiles(), "C1=CC=CC=C1");
    /// let molecule = MolecularGraph::from_smiles("c1ccccn1").unwrap();
    /// let mut molecule = molecule.first().unwrap().clone();
    /// molecule.kekulize().unwrap();
    /// assert_eq!(molecule.to_smiles(), "C1=CC=CC=N1");
    /// ```
    fn kekulize(&mut self) -> Result<(), String> {
        let mut aromatic_atoms: BitVec<usize> = BitVec::repeat(false, self.atom_bonds().len());
        let mut aromatic_rings: Vec<Vec<usize>> = Vec::new();
        let mut visited = BitVec::repeat(false, self.atomic_numbers().len());

        // Find aromatic atoms and rings
        for (index, bonds) in self.atom_bonds().iter().enumerate() {
            for bond in bonds {
                if bond.bond_order() == BondOrder::Aromatic {
                    aromatic_atoms.set(index, true);
                    break;
                }
            }
        }

        // Early return if no aromatic atoms
        if !aromatic_atoms.any() {
            return Ok(());
        }

        // Find aromatic rings using DFS
        let mut path = Vec::new();
        let mut path_set = FxHashSet::default();

        for start in 0..self.atomic_numbers().len() {
            if aromatic_atoms[start] && !visited[start] {
                self.dfs_find_aromatic_rings(
                    start,
                    start,
                    &mut visited,
                    &mut path,
                    &mut path_set,
                    &mut aromatic_rings,
                    &aromatic_atoms,
                );
            }
        }

        // Validate rings follow Hückel's rule
        //aromatic_rings.retain(|ring| self.is_valid_aromatic_cycle(ring));

        // Drop rings wholly contained in a larger one. A ring can be a subset
        // of several larger rings, so the same index can be collected more
        // than once, and removing by index invalidates every later index —
        // marking then filtering in one pass avoids both.
        let mut is_subset = vec![false; aromatic_rings.len()];
        for outer_ring in &aromatic_rings {
            for (index, inner_ring) in aromatic_rings.iter().enumerate() {
                if inner_ring.len() < outer_ring.len()
                    && inner_ring
                        .iter()
                        .all(|atom_index| outer_ring.contains(atom_index))
                {
                    is_subset[index] = true;
                }
            }
        }

        let mut keep = is_subset.iter().map(|&subset| !subset);
        aromatic_rings.retain(|_| keep.next().unwrap_or(true));

        // Convert aromatic bonds to alternating single/double bonds
        for ring in &aromatic_rings {
            let ring_len = ring.len();

            // First convert all bonds to single bonds
            for atom_index in ring {
                let atom_bonds = self.get_atom_bonds(*atom_index).unwrap().to_vec();
                for bond in atom_bonds {
                    if bond.bond_order() == BondOrder::Aromatic {
                        self.change_bond_order(*atom_index, bond.target(), BondOrder::Single);
                    }
                }
            }

            let degrees = self.degrees();

            // Then add double bonds in alternating pattern
            let mut last_bond_was_double = false;
            for i in 0..ring_len {
                let atom1 = ring[i];
                let atom2 = ring[(i + 1) % ring_len];
                if Self::can_increase_bond(atom1, atom2, &degrees) && !last_bond_was_double {
                    self.change_bond_order(atom1, atom2, BondOrder::Double);
                    last_bond_was_double = true;
                } else {
                    last_bond_was_double = false;
                }
            }
        }

        Ok(())
    }

    fn can_be_conjugated(&self, atom_idx: usize) -> bool {
        let atomic_number = self.atomic_numbers()[atom_idx];
        let number_of_bonds = self.atom_bonds()[atom_idx].len();
        let number_of_double_or_triple_bonds = self.atom_bonds()[atom_idx]
            .iter()
            .filter(|bond| {
                bond.bond_order() == BondOrder::Double || bond.bond_order() == BondOrder::Triple
            })
            .count();

        if number_of_double_or_triple_bonds > 1 {
            return false;
        }

        // This needs sources
        match atomic_number {
            5 => number_of_bonds <= 3, // Boron
            6 => {
                let number_of_aromatic_bonds = self.atom_bonds()[atom_idx]
                    .iter()
                    .filter(|bond| bond.bond_order() == BondOrder::Aromatic)
                    .count();
                let charge = self.get_atom_charge(atom_idx);
                number_of_bonds <= 3
                    && (number_of_double_or_triple_bonds == 1
                        || number_of_aromatic_bonds > 1
                        || charge != 0)
            } // Carbon
            7 => number_of_bonds <= 4, // Nitrogen
            8 => number_of_bonds <= 2, // Oxygen
            15 => number_of_bonds <= 3, // Phosphorus
            16 => number_of_bonds <= 2, // Sulfur
            34 => number_of_bonds <= 3, // Selenium
            21..=30 | 39..=48 | 57..=80 | 89..=112 => true, // Transition metals
            _ => false,
        }
    }

    fn can_be_conjugated_within_system(&self, atom_idx: usize, system: &BitSet) -> bool {
        if self
            .get_atom_bonds(atom_idx)
            .unwrap_or_default()
            .iter()
            .any(|bond| bond.bond_order() == BondOrder::Aromatic)
        {
            return true;
        }

        let atomic_number = self.atomic_numbers()[atom_idx];
        let number_of_bonds = self.atom_bonds()[atom_idx].len();
        let number_of_double_or_triple_bonds = self.atom_bonds()[atom_idx]
            .iter()
            .filter(|bond| {
                bond.bond_order() == BondOrder::Double || bond.bond_order() == BondOrder::Triple
            })
            .count();
        if number_of_double_or_triple_bonds > 1 {
            return false;
        }

        match atomic_number {
            6 => {
                let number_of_double_or_triple_bonds_within_system = self.atom_bonds()[atom_idx]
                    .iter()
                    .filter(|bond| {
                        system.contains(bond.target())
                            && (bond.bond_order() == BondOrder::Double
                                || bond.bond_order() == BondOrder::Triple)
                    })
                    .count();
                let has_exocyclic_double_bond = self.atom_bonds()[atom_idx].iter().any(|bond| {
                    !system.contains(bond.target()) && bond.bond_order() == BondOrder::Double
                });
                //let is_bonded_to_exocyclic_more_electronegative_atom_via_double_bond = self.atom_bonds()[atom_idx].iter().filter(|bond| !system.contains(bond.target())).any(|bond| (self.cmp_electronegativities(atom_idx, bond.target()) == std::cmp::Ordering::Less) && bond.bond_order() == BondOrder::Double);
                number_of_bonds <= 3
                    && (
                        number_of_double_or_triple_bonds == 1
                            && number_of_double_or_triple_bonds_within_system == 1
                            || has_exocyclic_double_bond
                        // is_bonded_to_exocyclic_more_electronegative_atom_via_double_bond
                    )
            } // Carbon
            7 => number_of_bonds <= 4,                      // Nitrogen
            8 => number_of_bonds <= 2,                      // Oxygen
            15 => number_of_bonds <= 3,                     // Phosphorus
            16 => number_of_bonds <= 2,                     // Sulfur
            21..=30 | 39..=48 | 57..=80 | 89..=112 => true, // Transition metals
            _ => false,
        }
    }

    fn dfs_find_aromatic_rings(
        &self,
        current: usize,
        parent: usize,
        visited: &mut BitVec<usize>,
        path: &mut Vec<usize>,
        path_set: &mut FxHashSet<usize>,
        rings: &mut Vec<Vec<usize>>,
        aromatic_atoms: &BitVec<usize>,
    ) {
        visited.set(current, true);
        path.push(current);
        path_set.insert(current);

        for &bond in &self.atom_bonds()[current] {
            let target = bond.target();
            if !aromatic_atoms[target] {
                continue;
            }

            if !visited[target] {
                self.dfs_find_aromatic_rings(
                    target,
                    current,
                    visited,
                    path,
                    path_set,
                    rings,
                    aromatic_atoms,
                );
            } else if target != parent && path_set.contains(&target) {
                // Found aromatic cycle
                let cycle_start = path.iter().position(|&x| x == target).unwrap();
                let cycle = path[cycle_start..].to_vec();
                if cycle.len() >= 5 {
                    // Common aromatic ring sizes
                    rings.push(cycle);
                }
            }
        }

        path.pop();
        path_set.remove(&current);
    }

    fn dfs_find_potential_aromatic_cycles(
        &self,
        current: usize,
        root: usize,
        visited: &mut Vec<bool>,
        current_path: &mut Vec<usize>,
        cycles: &mut Vec<Vec<usize>>,
        atoms_in_cycles: &mut BitSet,
        last_bond: BondOrder,
    ) {
        visited[current] = true;
        current_path.push(current);

        for &bond in &self.atom_bonds()[current] {
            let target = bond.target();

            let is_last_bond_double = last_bond == BondOrder::Double;
            let is_current_bond_double = bond.bond_order() == BondOrder::Double;

            let number_of_bonds = self.atom_bonds()[current].len();

            // Skip if this would create an invalid alternating pattern
            if (is_current_bond_double && is_last_bond_double) || number_of_bonds > 3 {
                continue;
            }

            // Found cycle back to root
            if target == root && current_path.len() >= 3 {
                // Only add cycle if it follows alternating pattern
                if bond.bond_order() != last_bond {
                    let cycle = current_path.clone();
                    // Add all atoms in cycle to tracked set
                    for &atom in &cycle {
                        atoms_in_cycles.insert(atom);
                    }
                    cycles.push(cycle);
                }
                continue;
            }

            // Continue DFS if unvisited
            if !visited[target] {
                self.dfs_find_potential_aromatic_cycles(
                    target,
                    root,
                    visited,
                    current_path,
                    cycles,
                    atoms_in_cycles,
                    bond.bond_order(),
                );
            }
        }

        current_path.pop();
        visited[current] = false;
    }

    fn can_be_aromatic(&self, atom_idx: usize) -> bool {
        let atomic_number = self.atomic_numbers()[atom_idx];
        match atomic_number {
            6 => {
                let valence = self.actual_valency(atom_idx);
                valence <= 3 // C should have 3 or fewer bonds
            }
            7 => {
                // Nitrogen
                let valence = self.actual_valency(atom_idx);
                valence <= 3 // N should have 3 or fewer bonds
            }
            8 => {
                // Oxygen
                let valence = self.actual_valency(atom_idx);
                valence <= 2 // O should have 2 or fewer bonds
            }
            16 => {
                // Sulfur
                let valence = self.actual_valency(atom_idx);
                valence <= 2 // S should have 2 or fewer bonds in aromatic systems
            }
            15 => {
                // Phosphorus
                let valence = self.actual_valency(atom_idx);
                valence <= 3 // P should have 3 or fewer bonds
            }
            _ => false,
        }
    }

    fn count_atom_pi_electrons(&self, atom_idx: usize) -> usize {
        // Count pi electrons from multiple bonds
        let pi_electrons = self.atom_bonds()[atom_idx]
            .iter()
            .map(|bond| match bond.bond_order() {
                BondOrder::Double => 1, // One pi electron per double bond
                BondOrder::Triple => 2, // Two pi electrons per triple bond
                _ => 0,
            })
            .sum();

        // Add electrons from lone pairs for certain atoms
        let atomic_number = self.atomic_numbers()[atom_idx];
        match atomic_number {
            7 => {
                // Nitrogen
                if self.get_atom_charge(atom_idx) == 0 && pi_electrons == 0 {
                    2 // Neutral N with no pi bonds contributes lone pair
                } else {
                    pi_electrons
                }
            }
            8 | 16 => {
                // Oxygen or Sulfur
                if self.get_atom_charge(atom_idx) == 0 && pi_electrons == 0 {
                    2 // Neutral O/S with no pi bonds contributes lone pair
                } else {
                    pi_electrons
                }
            }
            _ => pi_electrons,
        }
    }

    fn count_atom_pi_electrons_within_system(&self, atom_idx: usize, system: &BitSet) -> usize {
        // Count pi electrons from multiple bonds
        let pi_electrons = self.atom_bonds()[atom_idx]
            .iter()
            .map(|bond| {
                if system.contains(bond.target()) {
                    match bond.bond_order() {
                        BondOrder::Double => 1, // One pi electron per double bond
                        BondOrder::Triple => 2, // Two pi electrons per triple bond
                        _ => 0,
                    }
                } else {
                    0
                }
            })
            .sum();

        // Add electrons from lone pairs for certain atoms
        let atomic_number = self.atomic_numbers()[atom_idx];
        match atomic_number {
            7 => {
                // Nitrogen
                if self.get_atom_charge(atom_idx) == 0 && pi_electrons == 0 {
                    2 // Neutral N with no pi bonds contributes lone pair
                } else {
                    pi_electrons
                }
            }
            8 | 16 => {
                // Oxygen or Sulfur
                if self.get_atom_charge(atom_idx) == 0 && pi_electrons == 0 {
                    2 // Neutral O/S with no pi bonds contributes lone pair
                } else {
                    pi_electrons
                }
            }
            _ => pi_electrons,
        }
    }
    fn is_valid_aromatic_system(&self, system: &BitSet) -> bool {
        // Count total pi electrons for the system
        let pi_electrons = system
            .iter()
            .map(|atom_idx| self.count_atom_pi_electrons_within_system(atom_idx, system))
            .sum();

        // Check if the total system follows Hückel's rule
        is_hueckel_satisfied(pi_electrons)
    }

    fn check_moebius_topology(&self, system: &BitSet) -> bool {
        false // Placeholder
    }

    /// Performs a simple aromatization following RDKit's model
    ///
    /// # Examples
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::experimental::MoleculeExperimental;
    /// let mut mol = MolecularGraph::from_smiles_single("c1ccccc1").unwrap();
    /// mol.simple_aromatization();
    /// assert!(mol.get_aromatic_atoms().len() == 6);
    /// ```
    fn simple_aromatization(&mut self) {
        // Find all potential cycles
        let mut cycles = self.enumerate_possible_cycles();
        if cycles.is_empty() {
            return;
        }
        if cycles.len() > 20 {
            println!("Number of cycles: {}", cycles.len());
            return;
        }
        // Group cycles into fused systems and filter based on aromaticity
        cycles.sort_by_key(|b| std::cmp::Reverse(b.len()));
        let mut aromatic_systems: Vec<BitSet> = Vec::new();

        for system in cycles {
            if aromatic_systems.iter().any(|s| s.is_superset(&system)) {
                continue;
            }
            if self.is_rdkit_aromatic_system(&system) {
                self.aromatize_system(&system);
                aromatic_systems.push(system);
            }
        }

        self.add_aromatic_bonds();
    }

    fn is_rdkit_aromatic_system(&self, system: &BitSet) -> bool {
        for atom_idx in system.iter() {
            if !self.can_be_conjugated_within_system(atom_idx, system) {
                return false;
            }
        }

        let pi_electrons = self.count_system_pi_electrons(system);
        is_hueckel_satisfied(pi_electrons)
    }

    fn determine_aromaticity_type(&self, system: &BitSet) -> AromaticityType {
        let pi_electrons = self.count_system_pi_electrons(system);
        let has_metal_center = self.check_for_metal_center(system);
        let is_moebius = self.check_moebius_topology(system);

        match (pi_electrons, has_metal_center, is_moebius) {
            (e, false, false) if is_hueckel_satisfied(e) => AromaticityType::Aromatic,
            (e, false, true) if (e % 4 == 2) => AromaticityType::MoebiusAromatic,
            (e, false, false) if e % 4 == 0 => AromaticityType::AntiAromatic,
            (_, true, _) => AromaticityType::Metallocenic,
            _ => AromaticityType::NonAromatic,
        }
    }

    fn count_system_pi_electrons(&self, system: &BitSet) -> usize {
        let mut total = 0;
        let mut counted_atoms: BitSet<usize> = BitSet::default();

        for atom_idx in system.iter() {
            if counted_atoms.insert(atom_idx) {
                total += self.count_atom_pi_electrons_extended_within_system(atom_idx, system);
            }
        }

        total
    }

    fn count_atom_pi_electrons_extended_within_system(
        &self,
        atom_idx: usize,
        system: &BitSet,
    ) -> usize {
        let mut count = 0;
        let atomic_number = self.atomic_numbers()[atom_idx];
        let charge = self.get_atom_charge(atom_idx);
        let is_radical = self.is_atom_radical(atom_idx);

        if self.is_atom_aromatic(atom_idx) {
            return 1;
        }

        // Add bond contributions
        for bond in self.get_atom_bonds(atom_idx).unwrap_or_default() {
            if !system.contains(bond.target()) {
                if bond.bond_order() == BondOrder::Double {
                    return 0;
                }
                continue;
            }
            count += match bond.bond_order() {
                BondOrder::Double => 2,
                BondOrder::Aromatic => {
                    if atomic_number == 6 {
                        1
                    } else {
                        0
                    }
                }
                _ => 0,
            };
        }

        count /= 2;

        // Base pi electrons from atomic configuration, considering charge
        count += match atomic_number {
            // Todo: Check this
            5 => 0, // Boron
            6 if count == 0 => match charge {
                // Carbon
                0 => 0,
                1 => 0,  // C+ has no free electrons
                -1 => 2, // C- has two extra electrons
                _ if charge > 1 => 0,

                _ => 2, // C2- or lower has two extra electrons
            },
            7 if count == 0 => match charge {
                // Nitrogen
                1 => 1, // N+ has one free electron
                0 => {
                    if self.get_atom_bonds(atom_idx).unwrap_or_default().len() > 2 {
                        2
                    } else {
                        1
                    }
                } // Neutral N has two free electrons
                -1 => 3, // N- has three free electrons
                _ if charge > 1 => 0,
                _ => 4, // N2- or lower has four free electrons
            },
            8 | 16 => match charge {
                // Oxygen, Sulfur
                1 => 1,  // O+/S+ has one free electron
                0 => 2,  // Neutral O/S has two free electrons
                -1 => 3, // O-/S- has three free electrons
                _ if charge > 1 => 0,
                _ => 4, // O2-/S2- or lower has four free electrons
            },
            15 => match charge {
                // Phosphorus
                1 => 0,  // P+ has no free electrons
                0 => 2,  // Neutral P has two free electrons
                -1 => 3, // P- has three free electrons
                _ if charge > 1 => 0,
                _ => 4, // P2- or lower has four free electrons
            },
            _ => 0,
        };

        // Add radical contribution
        if is_radical {
            count += 1;
        }

        count
    }

    fn are_all_atoms_conjugated(&self, system: &BitSet) -> bool {
        for atom_idx in system.iter() {
            if !self.can_be_conjugated_within_system(atom_idx, system) {
                return false;
            }
        }
        true
    }

    fn dfs_find_conjugated_cycles(
        &self,
        current: usize,
        parent: usize,
        visited: &mut Vec<bool>,
        path: &mut Vec<usize>,
        path_set: &mut FxHashSet<usize>,
        cycles: &mut Vec<BitSet>,
        atoms_in_cycles: &mut BitSet<usize>,
    ) {
        visited[current] = true;
        path.push(current);
        path_set.insert(current);

        for &bond in &self.atom_bonds()[current] {
            let target = bond.target();

            // Skip parent and non-conjugated atoms
            if target == parent || !self.can_be_conjugated(target) {
                continue;
            }

            // Found a cycle
            if path_set.contains(&target) {
                if target == path[0] && path.len() >= 4 {
                    // Valid cycle found (minimum size 5)
                    let cycle = path.clone();
                    if self.is_potentially_aromatic_cycle(&cycle) {
                        let cycle_bitset = cycle_to_bitset(&cycle);
                        if !cycles.contains(&cycle_bitset) {
                            cycles.push(cycle_bitset);
                        }

                        for atom in path.iter() {
                            atoms_in_cycles.insert(*atom);
                        }
                    }
                }
            } else if !visited[target] {
                self.dfs_find_conjugated_cycles(
                    target,
                    current,
                    visited,
                    path,
                    path_set,
                    cycles,
                    atoms_in_cycles,
                );
            }
        }

        path.pop();
        path_set.remove(&current);
        visited[current] = false;
    }

    fn aromatize_old(&mut self) {
        let mut cycles: Vec<BitSet> = Vec::new();
        let mut atoms_in_cycles = BitSet::default();

        // First pass: Find all potential cycles
        self.find_all_conjugated_cycles(&mut cycles, &mut atoms_in_cycles);

        // Group and analyze systems
        let mut fused_systems = self.group_fused_cycles(&cycles);

        // Order based on size in descending order
        fused_systems.sort_by_key(|b| std::cmp::Reverse(b.len()));

        // Process each system
        for system in fused_systems {
            let aromaticity_type = self.determine_aromaticity_type(&system);
            match aromaticity_type {
                AromaticityType::Aromatic => self.aromatize_system(&system),
                AromaticityType::MoebiusAromatic => (), //self.aromatize_moebius_system(&system),
                AromaticityType::AntiAromatic => continue,
                AromaticityType::Metallocenic => (), //self.aromatize_metallocenic_system(&system),
                AromaticityType::NonAromatic => continue,
            }
        }
    }

    fn aromatize_system(&mut self, system: &BitSet) {
        // Get all bonds in the system
        let mut system_bonds = Vec::new();
        for atom_idx in system.iter() {
            for &bond in &self.atom_bonds()[atom_idx] {
                if system.contains(bond.target()) {
                    system_bonds.push((atom_idx, bond.target()));
                }
            }
        }

        // Convert all bonds in the system to aromatic
        for (atom1, atom2) in system_bonds {
            if atom1 < atom2 {
                self.change_bond_order(atom1, atom2, BondOrder::Aromatic);
            }
        }
    }

    fn find_all_conjugated_cycles(
        &self,
        cycles: &mut Vec<BitSet>,
        atoms_in_cycles: &mut BitSet<usize>,
    ) {
        let mut visited = vec![false; self.atomic_numbers().len()];
        let mut path = Vec::new();
        let mut path_set = FxHashSet::default();

        for start in 0..self.atomic_numbers().len() {
            if !atoms_in_cycles.contains(start) && self.can_be_conjugated(start) {
                self.dfs_find_conjugated_cycles(
                    start,
                    start,
                    &mut visited,
                    &mut path,
                    &mut path_set,
                    cycles,
                    atoms_in_cycles,
                );
            }
        }
    }

    /// Checks if two cycles share any atoms
    ///
    /// # Arguments
    /// * `cycle1_bits` - BitSet representation of first cycle
    /// * `cycle2_bits` - BitSet representation of second cycle
    ///
    /// # Returns
    /// true if the cycles share any atoms
    fn cycles_share_atoms(cycle1_bits: &BitSet, cycle2_bits: &BitSet) -> bool {
        cycle1_bits.intersection(cycle2_bits).next().is_some()
    }

    fn is_potentially_aromatic_cycle(&self, cycle: &[usize]) -> bool {
        cycle.len() > 3
    }

    /// Finds all subcycles within a given cycle and adds them to the cycles list
    ///
    /// # Arguments
    /// * `cycles` - Vector of BitSets representing cycles to check for subcycles
    /// * `atoms_in_cycles` - Set of atoms that are part of any cycle
    ///
    /// # Returns
    /// A new vector containing all original cycles plus any valid subcycles found
    fn find_and_add_subcycles(&self, cycles: &mut Vec<BitSet>, atoms_in_cycles: &mut BitSet) {
        let mut subcycles = Vec::new();

        // For each cycle, try to find subcycles by removing one atom at a time
        for cycle in cycles.clone() {
            let cycle_vec: Vec<_> = cycle.iter().collect();
            if cycle_vec.len() <= 5 {
                continue; // Skip cycles that are too small to contain subcycles
            }

            // Try removing each atom to find potential subcycles
            for skip_atom in cycle_vec.iter() {
                let mut visited = vec![false; self.atomic_numbers().len()];
                let mut path = Vec::new();
                let mut path_set = BitSet::new();

                // Start DFS from each remaining atom in the cycle
                for &start in cycle_vec.iter().filter(|&&atom| atom != *skip_atom) {
                    if visited[start] {
                        continue;
                    }

                    self.dfs_find_subcycles(
                        start,
                        start,
                        &cycle,
                        &mut visited,
                        &mut path,
                        &mut path_set,
                        &mut subcycles,
                        atoms_in_cycles,
                    );
                }
            }
        }

        // Add valid subcycles to the result
        for subcycle in subcycles {
            if !cycles.contains(&subcycle)
                && self.is_potentially_aromatic_cycle(&subcycle.iter().collect::<Vec<_>>())
            {
                cycles.push(subcycle);
            }
        }
    }

    fn dfs_find_subcycles(
        &self,
        current: usize,
        skip_atom: usize,
        original_cycle: &BitSet,
        visited: &mut Vec<bool>,
        path: &mut Vec<usize>,
        path_set: &mut BitSet,
        subcycles: &mut Vec<BitSet>,
        atoms_in_cycles: &mut BitSet,
    ) {
        visited[current] = true;
        path.push(current);
        path_set.insert(current);

        for &bond in &self.atom_bonds()[current] {
            let target = bond.target();

            // Skip the atom we're excluding and atoms not in the original cycle
            if target == skip_atom || !original_cycle.contains(target) {
                continue;
            }

            // Found a cycle
            if path_set.contains(target) {
                if target == path[0] && path.len() >= 5 {
                    let subcycle = cycle_to_bitset(path);
                    if !subcycles.contains(&subcycle) {
                        subcycles.push(subcycle);
                        for atom in path.clone() {
                            atoms_in_cycles.insert(atom);
                        }
                    }
                }
            } else if !visited[target] {
                self.dfs_find_subcycles(
                    target,
                    current,
                    original_cycle,
                    visited,
                    path,
                    path_set,
                    subcycles,
                    atoms_in_cycles,
                );
            }
        }

        path.pop();
        path_set.remove(current);
        visited[current] = false;
    }

    fn count_cycle_pi_electrons(&self, cycle: &[usize]) -> usize {
        let mut total = 0;

        // Count electrons from atoms
        for &atom_idx in cycle {
            total += self
                .count_atom_pi_electrons_extended_within_system(atom_idx, &cycle_to_bitset(cycle));
        }

        total
    }

    /// Groups cycles into fused systems using BitSet operations
    fn group_fused_cycles(&self, cycles: &[BitSet]) -> Vec<BitSet> {
        let mut fused_systems = Vec::new();
        let mut used = vec![false; cycles.len()];

        for (i, _) in cycles.iter().enumerate() {
            if used[i] {
                continue;
            }

            let mut current_system = cycles[i].clone();
            used[i] = true;

            let mut changed = true;
            while changed {
                changed = false;
                for (j, _) in cycles.iter().enumerate() {
                    if used[j] {
                        continue;
                    }

                    // Check if cycle j shares atoms with current system
                    if Self::cycles_share_atoms(&current_system, &cycles[j]) {
                        // Union of bits
                        current_system.union_with(&cycles[j]);
                        used[j] = true;
                        changed = true;
                    }
                }
            }

            if self.is_rdkit_aromatic_system(&current_system)
                && !fused_systems.contains(&current_system)
                && !cycles.contains(&current_system)
            {
                fused_systems.push(current_system);
            }
        }

        fused_systems.extend_from_slice(cycles);

        fused_systems
    }
}
