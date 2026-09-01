//! The assembled MMFF94 force field: which terms a molecule has, and what
//! they add up to at a geometry.
//!
//! Construction reads topology only -- typing, charges, parameter lookup,
//! and the enumeration of every bond, angle, torsion, out-of-plane centre
//! and non-bonded pair -- so evaluating the field at many geometries costs
//! only the arithmetic.

use super::super::UntypedAtom;
use super::perception::Perception;
use super::{assign, charges, params, terms, typing};
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;
use crate::molecule::featurize::Hybridization;
use crate::molecule::minimize::{Energy, minimize};
use crate::molecule::topology::{DistanceMatrix, topological_distances};
use crate::vector::Vector3D;

/// Convergence threshold on the largest force, in kcal/mol per Angstrom,
/// scaled by each coordinate's own size -- see `minimize`'s convergence
/// tests. RDKit's default for the same test is 1e-4.
const GRADIENT_TOLERANCE: f64 = 1e-4;

/// How many bonds apart two atoms must be to interact through space. Three
/// is the 1-4 relationship, which MMFF includes -- at full strength for van
/// der Waals and at three quarters for electrostatics.
const NONBONDED_MIN_SEPARATION: u8 = 3;

/// Each term's contribution, which is how MMFF is checked: a total that
/// agrees can still be two errors cancelling, and a force field is easier to
/// debug one term at a time.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TermEnergies {
    pub bond: f64,
    pub angle: f64,
    pub stretch_bend: f64,
    pub out_of_plane: f64,
    pub torsion: f64,
    pub van_der_waals: f64,
    pub electrostatic: f64,
}

impl TermEnergies {
    pub fn total(&self) -> f64 {
        self.bond
            + self.angle
            + self.stretch_bend
            + self.out_of_plane
            + self.torsion
            + self.van_der_waals
            + self.electrostatic
    }
}

/// One torsion as the diagnostics report it: its four atoms, MMFF's type
/// for it, and the three Fourier coefficients.
pub type TorsionTerm = (usize, usize, usize, usize, u8, f64, f64, f64);

/// One bond's equilibrium geometry: `(i, j, angstrom, md/A)` with `i < j`,
/// the last being the stretch force constant.
pub type EquilibriumLength = (usize, usize, f64, f64);

/// One angle's equilibrium value: `(i, centre, j, degrees)` with `i < j`.
pub type EquilibriumAngle = (usize, usize, usize, f64);

/// MMFF94, built for one molecule's topology.
pub struct ForceField {
    bonds: Vec<terms::Bond>,
    angles: Vec<terms::Angle>,
    out_of_planes: Vec<terms::OutOfPlane>,
    torsions: Vec<terms::Torsion>,
    non_bonded: terms::NonBondedPairs,
}

impl ForceField {
    /// Builds the field for `molecule`.
    ///
    /// # Errors
    ///
    /// [`UntypedAtom`] if any atom is outside MMFF94's typing rules, since
    /// every parameter in the field is keyed on those types.
    pub fn new<M: MoleculeCore + ?Sized>(molecule: &M) -> Result<Self, UntypedAtom> {
        let (perception, types) = typing::typed(molecule)?;
        let charges = charges::charges_from_types(&perception, &types);
        let distances = topological_distances(molecule);
        let count = perception.atom_count();

        let mut field = Self {
            bonds: Vec::new(),
            angles: Vec::new(),
            out_of_planes: Vec::new(),
            torsions: Vec::new(),
            non_bonded: terms::NonBondedPairs::default(),
        };

        field.collect_bonds(&perception, &types, count);
        field.collect_angles(&perception, &types, count);
        field.collect_out_of_planes(&perception, &types, count);
        field.collect_torsions(&perception, &types, count);
        field.collect_nonbonded(&types, &charges, &distances, count);

        Ok(field)
    }

    /// The equilibrium bond lengths and bond angles this field was built
    /// with, as `(i, j, angstrom, stretch constant)` and `(i, centre, j,
    /// degrees)`.
    ///
    /// The stretch constant comes along because it says how far the bond
    /// can be expected to stray from the length beside it -- a bound's
    /// width, in principle. Measured, it is the wrong quantity to scale
    /// (see `embed::bounds::BOND_TOLERANCE`), but reporting a parameter
    /// without the one that says how stiff it is would be a strange
    /// omission.
    ///
    /// For [`crate::molecule::embed`], whose distance bounds want the
    /// geometry of the field their structure is about to be relaxed
    /// against. Reading them off a field that exists is free; asking for
    /// them separately types the molecule a second time, which measured 4%
    /// of a single-threaded embedding.
    pub fn equilibrium_geometry(&self) -> (Vec<EquilibriumLength>, Vec<EquilibriumAngle>) {
        let lengths = self
            .bonds
            .iter()
            .map(|bond| {
                (
                    bond.i.min(bond.j),
                    bond.i.max(bond.j),
                    bond.length,
                    bond.force_constant,
                )
            })
            .collect();
        let angles = self
            .angles
            .iter()
            .map(|angle| {
                (
                    angle.i.min(angle.k),
                    angle.central,
                    angle.i.max(angle.k),
                    angle.angle,
                )
            })
            .collect();
        (lengths, angles)
    }

    fn collect_bonds(&mut self, perception: &Perception, types: &[u8], count: usize) {
        for i in 0..count {
            for neighbour in perception.neighbours(i) {
                let j = neighbour.atom;
                if j <= i {
                    continue;
                }
                if let Some(parameters) = assign::bond_parameters(perception, types, i, j) {
                    self.bonds.push(terms::Bond {
                        i,
                        j,
                        force_constant: parameters.force_constant,
                        length: parameters.length,
                    });
                }
            }
        }
    }

    /// Angles and the stretch-bend terms that ride on them: both are keyed
    /// on the same triple, and the stretch-bend needs the angle's own
    /// equilibrium value and the two bonds' equilibrium lengths.
    fn collect_angles(&mut self, perception: &Perception, types: &[u8], count: usize) {
        for central in 0..count {
            let neighbours = perception.neighbours(central);
            if neighbours.len() < 2 {
                continue;
            }
            for (position, first) in neighbours.iter().enumerate() {
                for last in &neighbours[position + 1..] {
                    let (i, k) = (first.atom, last.atom);
                    let Some(angle) = assign::angle_parameters(perception, types, i, central, k)
                    else {
                        continue;
                    };
                    let linear = params::properties(types[central])
                        .is_some_and(|properties| properties.linear);
                    let coupling =
                        assign::stretch_bend_parameters(perception, types, i, central, k);
                    let bonds = (
                        assign::bond_parameters(perception, types, i, central),
                        assign::bond_parameters(perception, types, central, k),
                    );
                    let stretch = match (coupling, bonds) {
                        (Some(coupling), (Some(first_bond), Some(second_bond))) => {
                            Some(terms::StretchBend {
                                first_constant: coupling.first,
                                second_constant: coupling.second,
                                first_length: first_bond.length,
                                second_length: second_bond.length,
                            })
                        }
                        _ => None,
                    };
                    self.angles.push(terms::Angle {
                        i,
                        central,
                        k,
                        force_constant: angle.force_constant,
                        angle: angle.angle,
                        linear,
                        stretch,
                    });
                }
            }
        }
    }

    /// Out-of-plane bending, at every three-connected centre. MMFF shares
    /// the one force constant over the three ways of choosing which bond is
    /// the one out of the plane.
    fn collect_out_of_planes(&mut self, perception: &Perception, types: &[u8], count: usize) {
        for central in 0..count {
            let neighbours = perception.neighbours(central);
            if neighbours.len() != 3 {
                continue;
            }
            let around = [neighbours[0].atom, neighbours[1].atom, neighbours[2].atom];
            // Where the table has no entry, MMFF leaves the term out rather
            // than inventing one.
            let Some(force_constant) = assign::out_of_plane(types, central, around) else {
                continue;
            };
            for (i, k, l) in [
                (around[0], around[1], around[2]),
                (around[0], around[2], around[1]),
                (around[1], around[2], around[0]),
            ] {
                self.out_of_planes.push(terms::OutOfPlane {
                    i,
                    central,
                    k,
                    l,
                    force_constant,
                });
            }
        }
    }

    fn collect_torsions(&mut self, perception: &Perception, types: &[u8], count: usize) {
        // A torsion needs a bond with something on both ends, about which
        // there is something to turn: not a terminal atom, not a triple
        // bond, and a centre of a shape that has a preference at all.
        let turnable = |atom: usize| {
            perception.neighbours(atom).len() > 1
                && !perception
                    .neighbours(atom)
                    .iter()
                    .any(|neighbour| neighbour.order == BondOrder::Triple)
                && matches!(
                    perception.shape(atom),
                    Hybridization::Sp2 | Hybridization::Sp3
                )
        };

        for j in 0..count {
            if !turnable(j) {
                continue;
            }
            for central in perception.neighbours(j) {
                let k = central.atom;
                if k <= j || !turnable(k) {
                    continue;
                }
                for first in perception.neighbours(j) {
                    let i = first.atom;
                    if i == k {
                        continue;
                    }
                    for last in perception.neighbours(k) {
                        let l = last.atom;
                        // `l == i` would be a three-membered ring, which has
                        // no torsion about this bond.
                        if l == j || l == i {
                            continue;
                        }
                        let Some((kind, parameters)) =
                            assign::torsion_parameters(perception, types, i, j, k, l)
                        else {
                            continue;
                        };
                        self.torsions.push(terms::Torsion {
                            kind,
                            i,
                            j,
                            k,
                            l,
                            v1: parameters.v1,
                            v2: parameters.v2,
                            v3: parameters.v3,
                        });
                    }
                }
            }
        }
    }

    /// Van der Waals and electrostatics, over every pair far enough apart
    /// through the bonds. Atoms in different fragments do not interact,
    /// which is the convention RDKit's builder defaults to.
    fn collect_nonbonded(
        &mut self,
        types: &[u8],
        charges: &[f64],
        distances: &DistanceMatrix,
        count: usize,
    ) {
        for i in 0..count {
            for j in (i + 1)..count {
                let separation = distances.get(i, j);
                if separation < NONBONDED_MIN_SEPARATION
                    || separation == DistanceMatrix::UNREACHABLE
                {
                    continue;
                }
                let (minimum, well) =
                    assign::van_der_waals(types, i, j).unwrap_or((0.0, 0.0));
                let charge_product = if charges[i] != 0.0 && charges[j] != 0.0 {
                    let scale = if separation == NONBONDED_MIN_SEPARATION {
                        terms::ONE_FOUR_SCALE
                    } else {
                        1.0
                    };
                    charges[i] * charges[j] * scale
                } else {
                    0.0
                };
                if well == 0.0 && charge_product == 0.0 {
                    continue;
                }
                self.non_bonded
                    .push(terms::NonBonded::new(i, j, minimum, well, charge_product));
            }
        }
    }

    /// Every torsion in the field, as `(i, j, k, l, type, V1, V2, V3)`.
    ///
    /// Here because a torsion is the term most likely to differ from
    /// another implementation without the total moving much, and comparing
    /// totals cannot say which torsion it was.
    pub fn torsion_terms(&self) -> Vec<TorsionTerm> {
        self.torsions
            .iter()
            .map(|term| {
                (
                    term.i, term.j, term.k, term.l, term.kind, term.v1, term.v2, term.v3,
                )
            })
            .collect()
    }

    /// The energy of `positions`, term by term.
    pub fn energies(&self, positions: &[Vector3D]) -> TermEnergies {
        #[cfg(target_arch = "x86_64")]
        if let Some(simd) = pulp::x86::V3::try_new() {
            return simd.vectorize(
                #[inline(always)]
                || self.energies_in(terms::Avx(simd), positions),
            );
        }
        self.energies_in(terms::Portable, positions)
    }

    /// [`ForceField::energies`] on the lanes `simd` provides. Inlined
    /// whole, so that inside [`pulp`]'s dispatch every term -- not only the
    /// pair loop -- is compiled for the instruction set it found.
    #[inline(always)]
    fn energies_in<L: terms::Lanes>(&self, simd: L, positions: &[Vector3D]) -> TermEnergies {
        let angle_and_stretch = self.angles.iter().fold((0.0, 0.0), |(bend, stretch), term| {
            let (one, two) = terms::angle_energies(positions, term);
            (bend + one, stretch + two)
        });
        TermEnergies {
            bond: self
                .bonds
                .iter()
                .map(|term| terms::bond_energy(positions, term))
                .sum(),
            // One pass, two terms: they are the same three atoms and the
            // same geometry, and computing it twice was the largest waste
            // in the energy-only path.
            angle: angle_and_stretch.0,
            stretch_bend: angle_and_stretch.1,
            out_of_plane: self
                .out_of_planes
                .iter()
                .map(|term| terms::out_of_plane_energy(positions, term))
                .sum(),
            torsion: self
                .torsions
                .iter()
                .map(|term| terms::torsion_energy(positions, term))
                .sum(),
            ..self.nonbonded_energies(simd, positions)
        }
    }

    /// The two through-space terms, from one pass over the pair list.
    #[inline(always)]
    fn nonbonded_energies<L: terms::Lanes>(&self, simd: L, positions: &[Vector3D]) -> TermEnergies {
        let (van_der_waals, electrostatic) =
            terms::nonbonded_energies_in(simd, &self.non_bonded, positions);
        TermEnergies {
            van_der_waals,
            electrostatic,
            ..TermEnergies::default()
        }
    }

    /// The total energy of `positions`.
    pub fn energy(&self, positions: &[Vector3D]) -> f64 {
        <Self as Energy>::energy(self, positions)
    }

    /// Relaxes `positions` in place, returning the final energy.
    pub fn minimize(&self, positions: &mut [Vector3D], max_iterations: usize) -> f64 {
        minimize(positions, self, max_iterations, GRADIENT_TOLERANCE)
    }
}

/// Both evaluations pick their instruction set per call: AVX2 through
/// [`pulp`] where the CPU has it, otherwise whatever the crate was compiled
/// for. The check is one relaxed atomic load, and the two paths agree to
/// the bit -- see [`terms::Lanes`].
impl Energy for ForceField {
    fn energy(&self, positions: &[Vector3D]) -> f64 {
        self.energies(positions).total()
    }

    fn energy_and_gradient(&self, positions: &[Vector3D], gradient: &mut [Vector3D]) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if let Some(simd) = pulp::x86::V3::try_new() {
            return simd.vectorize(
                #[inline(always)]
                || self.energy_and_gradient_in(terms::Avx(simd), positions, gradient),
            );
        }
        self.energy_and_gradient_in(terms::Portable, positions, gradient)
    }
}

impl ForceField {
    /// [`Energy::energy_and_gradient`] on the lanes `simd` provides; see
    /// [`ForceField::energies_in`] for why it is inlined whole.
    #[inline(always)]
    fn energy_and_gradient_in<L: terms::Lanes>(
        &self,
        simd: L,
        positions: &[Vector3D],
        gradient: &mut [Vector3D],
    ) -> f64 {
        gradient.fill(Vector3D::default());
        let mut energy = 0.0;
        for term in &self.bonds {
            energy += terms::bond_energy_and_gradient(positions, term, gradient);
        }
        for term in &self.angles {
            let (bend, stretch) = terms::angle_energies_and_gradient(positions, term, gradient);
            energy += bend + stretch;
        }
        for term in &self.out_of_planes {
            energy += terms::out_of_plane_energy_and_gradient(positions, term, gradient);
        }
        for term in &self.torsions {
            energy += terms::torsion_energy_and_gradient(positions, term, gradient);
        }
        let (repulsive, coulombic) =
            terms::nonbonded_energies_and_gradient_in(simd, &self.non_bonded, positions, gradient);
        energy += repulsive + coulombic;
        energy
    }
}
