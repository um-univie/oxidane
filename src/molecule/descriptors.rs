use super::traits::MoleculeCore;
use crate::molecule::bond::BondOrder;
use chemistry_consts::ElementProperties;

/// What the polar part of a nitrogen or an oxygen contributes to the surface,
/// in square angstroms.
///
/// Ertl's table, which was fitted once against computed surfaces and has been
/// used as a lookup ever since. The key is everything about the atom that the
/// fit distinguished: its charge, whether it is aromatic, how many hydrogens it
/// carries, and what its bonds are.
mod polar_surface {
    /// Nitrogen, aliphatic. Keyed by `(hydrogens, singles, doubles, triples,
    /// charge, in three-membered ring)`.
    pub(super) const NITROGEN: &[((u8, u8, u8, u8, i8, bool), f64)] = &[
        ((0, 3, 0, 0, 0, false), 3.24),
        ((0, 3, 0, 0, 0, true), 3.01),
        ((0, 1, 1, 0, 0, false), 12.36),
        ((0, 0, 0, 1, 0, false), 23.79),
        ((0, 1, 2, 0, 0, false), 11.68),
        ((0, 0, 1, 1, 0, false), 13.60),
        ((1, 2, 0, 0, 0, false), 12.03),
        ((1, 2, 0, 0, 0, true), 21.94),
        ((1, 0, 1, 0, 0, false), 23.85),
        ((2, 1, 0, 0, 0, false), 26.02),
        ((0, 4, 0, 0, 1, false), 0.00),
        ((0, 2, 1, 0, 1, false), 3.01),
        ((0, 1, 0, 1, 1, false), 4.36),
        ((1, 3, 0, 0, 1, false), 4.44),
        ((1, 1, 1, 0, 1, false), 13.97),
        ((2, 2, 0, 0, 1, false), 16.61),
        ((2, 0, 1, 0, 1, false), 25.59),
        ((3, 1, 0, 0, 1, false), 27.64),
    ];

    /// Nitrogen, aromatic. Keyed by `(hydrogens, aromatic bonds, single bonds,
    /// double bonds, charge)`.
    pub(super) const AROMATIC_NITROGEN: &[((u8, u8, u8, u8, i8), f64)] = &[
        ((0, 2, 0, 0, 0), 12.89),
        ((0, 3, 0, 0, 0), 4.41),
        ((0, 2, 1, 0, 0), 4.93),
        ((0, 2, 0, 1, 0), 8.39),
        ((1, 2, 0, 0, 0), 15.79),
        ((0, 3, 0, 0, 1), 4.10),
        ((0, 2, 1, 0, 1), 3.88),
        ((1, 2, 0, 0, 1), 14.14),
    ];

    /// Oxygen. Keyed by `(hydrogens, singles, doubles, charge, aromatic,
    /// in three-membered ring)`.
    pub(super) const OXYGEN: &[((u8, u8, u8, i8, bool, bool), f64)] = &[
        ((0, 2, 0, 0, false, false), 9.23),
        ((0, 2, 0, 0, false, true), 12.53),
        ((0, 0, 1, 0, false, false), 17.07),
        ((1, 1, 0, 0, false, false), 20.23),
        ((0, 1, 0, -1, false, false), 23.06),
        ((0, 0, 0, 0, true, false), 13.14),
    ];

    /// Sulfur, counted only when asked for. Keyed as nitrogen is, without the
    /// ring flag.
    pub(super) const SULFUR: &[((u8, u8, u8, i8, bool), f64)] = &[
        ((0, 2, 0, 0, false), 25.30),
        ((0, 0, 1, 0, false), 32.09),
        ((0, 2, 1, 0, false), 19.21),
        ((0, 2, 2, 0, false), 8.38),
        ((1, 1, 0, 0, false), 38.80),
        ((0, 0, 0, 0, true), 28.24),
        ((0, 0, 1, 0, true), 21.70),
    ];

    /// Phosphorus, counted only when asked for.
    pub(super) const PHOSPHORUS: &[((u8, u8, u8, i8), f64)] = &[
        ((0, 3, 0, 0), 13.59),
        ((0, 1, 1, 0), 34.14),
        ((0, 3, 1, 0), 9.81),
        ((1, 2, 1, 0), 23.47),
    ];
}


pub trait MolecularDescriptors: MoleculeCore {
    /// Calculates molecular weight in g/mol using standard atomic weights for
    /// unlabelled atoms and exact isotope masses for explicitly labelled atoms.
    /// Returns NaN if a written isotope has no tabulated mass; wildcard labels
    /// contribute zero.
    ///
    /// # Examples
    /// ```
    /// use oxidane::molecule::MolecularGraph;
    /// use oxidane::molecule::descriptors::MolecularDescriptors;
    ///
    /// let molecule = MolecularGraph::from_smiles_single("CCO").unwrap();
    /// assert!((molecule.molecular_weight() - 46.07).abs() < 0.05);
    /// ```
    fn molecular_weight(&self) -> f64 {
        self.atomic_numbers().iter().enumerate().map(|(index, &element)| {
            match self.get_isotope(index) {
                Some(mass_number) => crate::atom::isotope_mass(element, mass_number).unwrap_or(f64::NAN),
                None => element.standard_atomic_weight().unwrap_or(0.0),
            }
        }).sum()
    }
    /// Isotope-aware monoisotopic mass; see [`MoleculeCore::monoisotopic_mass`]
    /// for the policy for unlabelled atoms and unavailable isotope masses.
    fn monoisotopic_mass(&self) -> f64 {
        MoleculeCore::monoisotopic_mass(self)
    }
    /// Counts hydrogen-bond donors by RDKit's `NumHDonors` definition, the
    /// donor half of the typed rules
    /// [`MolecularDescriptors::number_of_hydrogen_bond_acceptors`] follows:
    /// a neutral trivalent or cationic tetravalent nitrogen carrying a
    /// hydrogen, an aromatic `[nH]`, and a neutral oxygen or sulfur carrying
    /// exactly one -- so a thiol counts, and water and a sulfonium do not.
    ///
    /// Hydrogens are explicit graph atoms here, and valence counts them.
    fn number_of_hydrogen_bond_donors(&self) -> usize {
        (0..self.len())
            .filter(|&atom| {
                let hydrogens = self.number_of_bonded_element(atom, 1);
                if hydrogens == 0 {
                    return false;
                }
                let charge = self.get_atom_charge(atom);
                match self.get_atomic_number(atom) {
                    7 if self.is_atom_aromatic(atom) => charge == 0 && hydrogens == 1,
                    7 => {
                        let valence = self.actual_valency(atom);
                        (charge == 0 && valence == 3) || (charge == 1 && valence == 4)
                    }
                    8 | 16 => charge == 0 && hydrogens == 1,
                    _ => false,
                }
            })
            .count()
    }

    /// Counts chemically typed N/O/S hydrogen-bond acceptors.
    ///
    /// Uses the RDKit NumHBA 2.0.2 functional-group definition: excludes amide,
    /// sulfonamide, pyrrole and protonated amine nitrogens and acidic hydroxyls;
    /// includes eligible sulfur atoms. Hydrogens are explicit graph atoms here.
    /// See [the reference rules](https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Descriptors/Lipinski.cpp).
    fn number_of_hydrogen_bond_acceptors(&self) -> usize {
        // Typed N/O/S rules corresponding to RDKit's NumHBA 2.0.2.
        // Inspect heavy neighbours explicitly: this crate stores hydrogens as
        // atoms, which must not make an acid OH look like an alcohol OH.
        let mut cyclic = None;
        (0..self.len()).filter(|&atom| {
            let element = self.get_atomic_number(atom);
            if !matches!(element, 7 | 8 | 16) {
                return false;
            }
            let charge = self.get_atom_charge(atom);
            let hydrogens = self.number_of_bonded_element(atom, 1);
            let aromatic = self.is_atom_aromatic(atom);
            let valence = self.actual_valency(atom);
            let bonds = &self.atom_bonds()[atom];
            if aromatic {
                return charge == 0 && match element {
                    7 => hydrogens == 0 && bonds.len() == 2,
                    8 | 16 => true,
                    _ => false,
                };
            }
            if element == 8 || element == 16 {
                if charge == -1 {
                    return true;
                }
                if valence != 2 {
                    return false;
                }
                if hydrogens == 0 {
                    return true;
                }
                return hydrogens == 1 && bonds.iter().any(|bond| {
                    let neighbour = bond.target();
                    self.get_atomic_number(neighbour) != 1 &&
                        !self.atom_bonds()[neighbour].iter().any(|other| {
                            other.bond_order() == BondOrder::Double &&
                                matches!(self.get_atomic_number(other.target()), 7 | 8 | 15 | 16)
                        })
                });
            }
            if valence != 3 {
                return false;
            }
            // An amide, sulfonamide, amidine, etc. nitrogen has its lone pair
            // delocalized toward a non-ring double bond to N/O/P/S.
            !bonds.iter().any(|bond| {
                if bond.bond_order() != BondOrder::Single {
                    return false;
                }
                let neighbour = bond.target();
                self.atom_bonds()[neighbour].iter().any(|other| {
                    let target = other.target();
                    if other.bond_order() != BondOrder::Double ||
                        !matches!(self.get_atomic_number(target), 7 | 8 | 15 | 16) {
                        return false;
                    }
                    let ring_bonds = cyclic.get_or_insert_with(|| self.find_cyclic_bonds_and_atoms().1);
                    !ring_bonds.contains(&(neighbour.min(target), neighbour.max(target)))
                })
            })
        }).count()
    }

    /// How many atoms are not hydrogen.
    fn heavy_atom_count(&self) -> usize {
        self.atomic_numbers()
            .iter()
            .filter(|&&atomic_number| atomic_number != 1)
            .count()
    }

    /// The molecule's total formal charge.
    fn total_formal_charge(&self) -> i32 {
        self.charges().iter().map(|&charge| charge as i32).sum()
    }

    /// How many rings the molecule has: bonds minus atoms plus fragments.
    ///
    /// The cyclomatic number, which is how many bonds have to be cut to leave a
    /// tree, and so how many rings there are in the only sense that does not
    /// depend on which rings one chooses to name. Other toolkits report a
    /// symmetrised smallest set of smallest rings instead, which counts a cage
    /// higher -- adamantane is four rings there and three here, cubane six and
    /// five -- because it prefers to name every small face over naming as few
    /// rings as will do.
    fn ring_count(&self) -> usize {
        let atom_count = self.len();
        let mut bonds = 0usize;
        for atom in 0..atom_count {
            bonds += self.get_atom_bonds(atom).unwrap_or_default().len();
        }
        let bonds = bonds / 2;
        let fragments = self.fragment_count();
        (bonds + fragments).saturating_sub(atom_count)
    }

    /// How many connected pieces the molecule is in.
    fn fragment_count(&self) -> usize {
        let atom_count = self.len();
        let mut seen = vec![false; atom_count];
        let mut stack: Vec<usize> = Vec::new();
        let mut fragments = 0;
        for start in 0..atom_count {
            if seen[start] {
                continue;
            }
            fragments += 1;
            seen[start] = true;
            stack.push(start);
            while let Some(atom) = stack.pop() {
                for bond in self.get_atom_bonds(atom).unwrap_or_default() {
                    let target = bond.target();
                    if target < atom_count && !seen[target] {
                        seen[target] = true;
                        stack.push(target);
                    }
                }
            }
        }
        fragments
    }

    /// The proportion of carbons that are saturated.
    ///
    /// A measure of how three-dimensional a molecule is, and one of the few
    /// simple numbers that separates a drug from a dyestuff. Zero when there
    /// are no carbons at all.
    fn fraction_saturated_carbons(&self) -> f64 {
        let mut carbons = 0usize;
        let mut saturated = 0usize;
        for atom in 0..self.len() {
            if self.get_atomic_number(atom) != 6 {
                continue;
            }
            carbons += 1;
            let bonds = self.get_atom_bonds(atom).unwrap_or_default();
            let unsaturated = bonds
                .iter()
                .any(|bond| bond.bond_order() != BondOrder::Single);
            if !unsaturated && bonds.len() == 4 {
                saturated += 1;
            }
        }
        if carbons == 0 {
            return 0.0;
        }
        saturated as f64 / carbons as f64
    }

    /// How many bonds the molecule can turn about.
    ///
    /// A single acyclic bond between two atoms that each have something else to
    /// hold. Four kinds are left out because turning them changes nothing or
    /// because they do not turn:
    ///
    /// * a bond to a terminal atom, which has nothing to swing;
    /// * a bond next to a triple bond, which is collinear with it;
    /// * a bond to a symmetric top -- a `CF3`, a `CCl3`, a `CBr3`, a
    ///   `tert`-butyl -- whose rotation is a relabelling;
    /// * an amide, ester or thioester bond, which is a single bond in the
    ///   drawing and a partial double bond in the molecule.
    ///
    /// This is the strict count, the one a rule-of-five threshold is quoted
    /// against.
    fn rotatable_bond_count(&self) -> usize {
        let atom_count = self.len();
        // Once for the molecule, not once for every bond it has.
        let (_, cyclic) = self.find_cyclic_bonds_and_atoms();
        let mut count = 0usize;
        for first in 0..atom_count {
            if self.get_atomic_number(first) == 1 {
                continue;
            }
            for bond in self.get_atom_bonds(first).unwrap_or_default() {
                let second = bond.target();
                if second <= first || self.get_atomic_number(second) == 1 {
                    continue;
                }
                if bond.bond_order() != BondOrder::Single {
                    continue;
                }
                if cyclic.contains(&(first.min(second), first.max(second))) {
                    continue;
                }
                // One end has to be free of everything, the other only of the
                // things that stop a bond turning whichever side you look from.
                // An amide carbonyl cannot be the free end, but the carbon it
                // holds can be -- so `C(=O)-CH2-` still turns, while a bond
                // between two amide carbonyls does not.
                let first_anchored = self.is_rotation_anchor(first);
                let second_anchored = self.is_rotation_anchor(second);
                let first_free = !first_anchored && !self.is_conjugated_amide_end(first, &cyclic);
                let second_free =
                    !second_anchored && !self.is_conjugated_amide_end(second, &cyclic);
                if !((first_free && !second_anchored) || (second_free && !first_anchored)) {
                    continue;
                }
                count += 1;
            }
        }
        count
    }

    /// Whether an atom is one that a bond to it cannot usefully turn about:
    /// a terminal atom, one carrying a triple bond, or a symmetric top.
    fn is_rotation_anchor(&self, atom: usize) -> bool {
        let bonds = self.get_atom_bonds(atom).unwrap_or_default();
        if self.heavy_degree(atom) < 2 {
            return true;
        }
        if bonds
            .iter()
            .any(|bond| bond.bond_order() == BondOrder::Triple)
        {
            return true;
        }
        if self.get_atomic_number(atom) != 6 {
            return false;
        }
        // Three of the same thing hanging off a carbon: turning the bond that
        // holds it maps the molecule onto itself.
        let mut halogens = [0usize; 4];
        let mut methyls = 0usize;
        for bond in bonds {
            let target = bond.target();
            match self.get_atomic_number(target) {
                9 => halogens[0] += 1,
                17 => halogens[1] += 1,
                35 => halogens[2] += 1,
                53 => halogens[3] += 1,
                6 if self.heavy_degree(target) == 1 => methyls += 1,
                _ => (),
            }
        }
        halogens.iter().any(|&count| count == 3) || methyls == 3
    }

    /// Whether an atom is one end of an amide-like bond: a three-connected
    /// carbon double-bonded to nitrogen, oxygen or sulfur and joined by an
    /// acyclic single bond to one of those, or the heteroatom on the other side
    /// of such a bond.
    ///
    /// The disqualification is of the atom, not of the one bond: an amide
    /// nitrogen holds its substituents in the same plane as the carbonyl, so
    /// none of its bonds turn freely, and neither does the bond from the
    /// carbonyl carbon to whatever else it carries.
    fn is_conjugated_amide_end(
        &self,
        atom: usize,
        cyclic_bonds: &rustc_hash::FxHashSet<(usize, usize)>,
    ) -> bool {
        let is_carbonyl = |candidate: usize| {
            self.get_atomic_number(candidate) == 6
                && self.heavy_degree(candidate) == 3
                && self
                    .get_atom_bonds(candidate)
                    .unwrap_or_default()
                    .iter()
                    .any(|bond| {
                        bond.bond_order() == BondOrder::Double
                            && matches!(self.get_atomic_number(bond.target()), 7 | 8 | 16)
                    })
        };
        let is_heteroatom = |candidate: usize| match self.get_atomic_number(candidate) {
            7 | 8 => true,
            16 => self.heavy_degree(candidate) > 1,
            _ => false,
        };

        let partner_matches = |predicate: &dyn Fn(usize) -> bool| {
            self.get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .any(|bond| {
                    bond.bond_order() == BondOrder::Single
                        && !cyclic_bonds.contains(&(
                            atom.min(bond.target()),
                            atom.max(bond.target()),
                        ))
                        && predicate(bond.target())
                })
        };

        (is_carbonyl(atom) && partner_matches(&is_heteroatom))
            || (is_heteroatom(atom) && partner_matches(&is_carbonyl))
    }

    /// How many heavy neighbours an atom has.
    fn heavy_degree(&self, atom: usize) -> usize {
        self.get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .filter(|bond| self.get_atomic_number(bond.target()) != 1)
            .count()
    }

    /// The topological polar surface area, in square angstroms.
    ///
    /// Ertl's sum over nitrogen and oxygen, which stands in for how much of the
    /// molecule a membrane has to strip of water. Sulfur and phosphorus were
    /// fitted too but are left out by convention, since the values they were
    /// fitted against are the ones every published threshold assumes.
    fn topological_polar_surface_area(&self) -> f64 {
        self.polar_surface_area(false)
    }

    /// [`MolecularDescriptors::topological_polar_surface_area`], with the
    /// option of counting sulfur and phosphorus as Ertl also tabulated them.
    fn polar_surface_area(&self, include_sulfur_and_phosphorus: bool) -> f64 {
        let mut total = 0.0;
        for atom in 0..self.len() {
            total += match self.get_atomic_number(atom) {
                7 | 8 => self.polar_contribution(atom),
                15 | 16 if include_sulfur_and_phosphorus => self.polar_contribution(atom),
                _ => 0.0,
            };
        }
        total
    }

    /// What one atom contributes to the polar surface area.
    fn polar_contribution(&self, atom: usize) -> f64 {
        let bonds = self.get_atom_bonds(atom).unwrap_or_default();
        let charge = self.get_atom_charge(atom);
        let mut hydrogens = 0u8;
        let mut singles = 0u8;
        let mut doubles = 0u8;
        let mut triples = 0u8;
        let mut aromatics = 0u8;
        for bond in bonds {
            if self.get_atomic_number(bond.target()) == 1 {
                hydrogens += 1;
                continue;
            }
            match bond.bond_order() {
                BondOrder::Double => doubles += 1,
                BondOrder::Triple => triples += 1,
                BondOrder::Aromatic => aromatics += 1,
                _ => singles += 1,
            }
        }
        // Ertl's patterns name the hydrogens separately from the bonds -- an
        // `[NH](-*)-*` has one hydrogen and two single bonds, not three -- so
        // the hydrogens must not be counted twice.
        let aromatic = aromatics > 0;
        let small_ring = self.is_in_three_membered_ring(atom);
        let heavy_connections = singles + doubles + triples + aromatics;

        match self.get_atomic_number(atom) {
            7 if aromatic => polar_surface::AROMATIC_NITROGEN
                .iter()
                .find(|&&(key, _)| key == (hydrogens, aromatics, singles, doubles, charge))
                .map(|&(_, value)| value)
                .unwrap_or_else(|| nitrogen_fallback(heavy_connections, hydrogens)),
            7 => polar_surface::NITROGEN
                .iter()
                .find(|&&(key, _)| {
                    key == (
                        hydrogens,
                        singles,
                        doubles,
                        triples,
                        charge,
                        small_ring,
                    )
                })
                .map(|&(_, value)| value)
                .unwrap_or_else(|| nitrogen_fallback(heavy_connections, hydrogens)),
            8 => polar_surface::OXYGEN
                .iter()
                .find(|&&(key, _)| {
                    key == (
                        hydrogens,
                        singles,
                        doubles,
                        charge,
                        aromatic,
                        small_ring,
                    )
                })
                .map(|&(_, value)| value)
                .unwrap_or_else(|| oxygen_fallback(heavy_connections, hydrogens)),
            16 => polar_surface::SULFUR
                .iter()
                .find(|&&(key, _)| {
                    key == (hydrogens, singles, doubles, charge, aromatic)
                })
                .map(|&(_, value)| value)
                .unwrap_or(0.0),
            15 => polar_surface::PHOSPHORUS
                .iter()
                .find(|&&(key, _)| {
                    key == (hydrogens, singles, doubles, charge)
                })
                .map(|&(_, value)| value)
                .unwrap_or(0.0),
            _ => 0.0,
        }
    }

    /// Whether two of an atom's neighbours are bonded to each other, which is
    /// the whole of what it takes to be in a three-membered ring.
    fn is_in_three_membered_ring(&self, atom: usize) -> bool {
        let bonds = self.get_atom_bonds(atom).unwrap_or_default();
        for (position, first) in bonds.iter().enumerate() {
            for second in bonds.iter().skip(position + 1) {
                if self
                    .get_atom_bonds(first.target())
                    .unwrap_or_default()
                    .iter()
                    .any(|bond| bond.target() == second.target())
                {
                    return true;
                }
            }
        }
        false
    }

    /// The Wildman-Crippen log P and molar refractivity, as `(log P, MR)`.
    ///
    /// Both are sums of per-atom contributions; see
    /// [`crate::molecule::crippen`] for what an atom is classified as and why
    /// the classification is order-dependent.
    ///
    /// # Examples
    /// ```
    /// use oxidane::molecule::MolecularGraph;
    /// use oxidane::molecule::descriptors::MolecularDescriptors;
    /// let caffeine =
    ///     MolecularGraph::from_smiles_single("Cn1cnc2c1c(=O)n(C)c(=O)n2C").unwrap();
    /// let (log_p, refractivity) = caffeine.crippen_descriptors();
    /// assert!((log_p + 1.0293).abs() < 1e-3);
    /// assert!((refractivity - 51.196).abs() < 1e-3);
    /// ```
    fn crippen_descriptors(&self) -> (f64, f64) {
        crate::molecule::crippen::crippen_descriptors(self)
    }

    /// The Wildman-Crippen log P on its own.
    fn crippen_log_p(&self) -> f64 {
        self.crippen_descriptors().0
    }

    /// The Wildman-Crippen molar refractivity on its own.
    fn crippen_molar_refractivity(&self) -> f64 {
        self.crippen_descriptors().1
    }
}

/// What a nitrogen the table does not cover contributes.
///
/// Ertl's fitted fallback, for the states too rare to have been tabulated --
/// ammonia, an ammonium ion, the middle nitrogen of an azide. Never negative,
/// however many bonds the atom has.
fn nitrogen_fallback(heavy_connections: u8, hydrogens: u8) -> f64 {
    (30.5 - 8.2 * heavy_connections as f64 + 1.5 * hydrogens as f64).max(0.0)
}

/// What an oxygen the table does not cover contributes: water, hydroxide, an
/// oxocarbenium.
fn oxygen_fallback(heavy_connections: u8, hydrogens: u8) -> f64 {
    (28.5 - 8.6 * heavy_connections as f64 + 1.5 * hydrogens as f64).max(0.0)
}

impl<T: MoleculeCore> MolecularDescriptors for T {}
