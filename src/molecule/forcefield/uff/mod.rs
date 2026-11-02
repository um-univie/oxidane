//! The Universal Force Field, for relaxing 3D structures.
//!
//! Distance geometry produces coordinates that satisfy a set of distances.
//! That is not the same as coordinates a chemist would recognise: bonds sit
//! a little long or short, angles a little open, torsions wherever the
//! distances left them. Every toolkit closes that gap the same way, by
//! minimising a force field afterwards -- RDKit's users call
//! `MMFFOptimizeMoleculeConfs`, Open Babel does it inside `--gen3d` -- and
//! measured on this crate's own benchmark the difference is most of the
//! remaining gap to either of them.
//!
//! UFF is the less accurate of this crate's two force fields and the one
//! with the wider reach: one small parameter table spans the periodic table,
//! where [`super::mmff`] needs several. Reach is not coverage, though. The
//! typing in `typing` perceives fourteen elements -- H, B, C, N, O, F, Si,
//! P, S, Cl, Se, Br, I -- and refuses the rest, metals included, because a
//! metal's UFF type depends on a coordination geometry this crate does not
//! derive. A molecule typed by neither field is left unrelaxed by
//! `embed_3d` rather than scored with invented parameters.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::embed::Embed3D;
//! use oxidane::molecule::forcefield::ForceField;
//!
//! let molecule = MolecularGraph::from_smiles_single("CCO").unwrap();
//! let mut embedded = molecule.embed_3d().unwrap();
//! let field = ForceField::new(&embedded).unwrap();
//! let before = field.energy(&embedded.positions);
//! let after = field.minimize(&mut embedded.positions, 200);
//! assert!(after <= before);
//! ```
//!
//! # Not included
//!
//! * **Electrostatics.** Off by default in UFF as RDKit implements it too;
//!   this crate computes partial charges separately in
//!   [`crate::molecule::charges`].
//!
//! # Where this differs from RDKit's UFF
//!
//! RDKit is the oracle for this crate's MMFF94 checks, and almost all of
//! UFF matches it to the digit: the terms, the parameters, the combination
//! rules and the angle form's switch on the centre's geometry are all the
//! same reading of the paper. Two differences are deliberate:
//!
//! * **Atom typing.** RDKit suffixes `R` onto any sp2 carbon, nitrogen,
//!   oxygen or sulfur that has a conjugated bond, which types a carbonyl
//!   oxygen `O_R` -- the type fitted to furan, whose bond radius is 0.07 A
//!   longer than the carbonyl's own `O_2`. Every C=O it touches then
//!   relaxes to 1.26 A against the 1.22 A the parameter set was fitted to.
//!   `typing` assigns the paper's types by the paper's chemistry: the
//!   `R` types belong in aromatic rings, a carbonyl oxygen is `O_2`, an
//!   amide nitrogen `N_2`.
//! * **The amide bond order.** The paper prescribes 1.41 for an amide's
//!   C-N bond, for the resonance a written single bond carries. RDKit
//!   ships the constant and leaves it unwired, so its amide C-N relaxes
//!   to 1.42 A where the paper's rule gives 1.35 and experiment 1.37;
//!   `bond_order_value` applies it.
//!
//! `tests/uff_oracle.py` measures the gap on any SMILES: molecules of sp3
//! atoms, plain alkenes and aromatic rings agree to the last digit, and a
//! molecule with a conjugated carbonyl differs by the size of those two
//! choices.
//!
//! Three smaller RDKit behaviours also have no counterpart here, none of
//! them in the paper: overridden equilibrium angles at sp2 centres in
//! three- and four-membered rings, a penalty for angles near zero, and a
//! distance cut-off on the van der Waals term. Van der Waals is also kept
//! between separate fragments, where RDKit's builder drops it by default.

mod params;
mod terms;
pub(crate) mod typing;

use super::UntypedAtom;
use super::lanes;
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::{BondOrder, BondTarget};
use crate::molecule::featurize::{Hybridization, hybridization};
use crate::molecule::minimize::{Energy, minimize};
use crate::vector::Vector3D;
use params::UffType;
use terms::{Angle, Bond, Inversion, Torsion, VanDerWaals, VanDerWaalsPairs};

/// Squared gradient-norm threshold below which minimisation is finished;
/// `minimize` compares against `|g|^2`, so this stops at `|g| < 1e-3`.
const GRADIENT_TOLERANCE: f64 = 1e-6;

/// Van der Waals is skipped for atoms one or two bonds apart, where the bond
/// and angle terms already own the geometry -- the usual force-field
/// convention, and the one RDKit's UFF follows.
const NONBONDED_MIN_SEPARATION: usize = 3;

/// A force field built for one molecule's topology.
///
/// Construction does the work that depends only on connectivity -- typing,
/// parameter lookup, enumerating bonds, angles, torsions and non-bonded
/// pairs -- so evaluating it at many geometries costs only the arithmetic.
pub struct ForceField {
    bonds: Vec<Bond>,
    angles: Vec<Angle>,
    torsions: Vec<Torsion>,
    inversions: Vec<Inversion>,
    nonbonded: VanDerWaalsPairs,
}

/// UFF's bond order for the resonance in an amide's C-N bond, from the
/// paper. The general bond-order correction then shortens the bond from
/// the 1.41 A of a plain single C-N to the 1.35 A an amide's is.
const AMIDE_BOND_ORDER: f64 = 1.41;

/// The bond order UFF reads on the bond from `atom` to `bond.target()`: the
/// written order, except across an amide's C-N, where the paper prescribes
/// 1.41 for the resonance a written single bond carries. RDKit ships the
/// same constant and leaves it unwired ("FIX: recognize amide bonds
/// here"), and Open Babel applies it; without it the amide C-N relaxes to
/// 1.42 A against the 1.37 of experiment and GFN2-xTB's 1.36.
pub(crate) fn bond_order_value<M: MoleculeCore + ?Sized>(
    molecule: &M,
    atom: usize,
    bond: &BondTarget,
) -> f64 {
    let order = match bond.bond_order() {
        BondOrder::Aromatic => 1.5,
        BondOrder::Double => 2.0,
        BondOrder::Triple => 3.0,
        _ => 1.0,
    };
    if order == 1.0 && is_amide_bond(molecule, atom, bond.target()) {
        AMIDE_BOND_ORDER
    } else {
        order
    }
}

/// Whether the `first`-`second` bond is the C-N bond of an amide: a single
/// bond whose carbon carries a carbonyl oxygen. Lactams, imides, ureas and
/// carbamates all qualify -- each carries the same C-N partial double bond
/// -- and an aniline's C-N does not, its resonance ending inside the ring.
/// A thioamide's C(=S)-N does not either: the paper says amide, and
/// sulfur's carbonyl chemistry is not oxygen's.
fn is_amide_bond<M: MoleculeCore + ?Sized>(molecule: &M, first: usize, second: usize) -> bool {
    let carbon = match (
        molecule.get_atomic_number(first),
        molecule.get_atomic_number(second),
    ) {
        (6, 7) => first,
        (7, 6) => second,
        _ => return false,
    };
    molecule
        .get_atom_bonds(carbon)
        .unwrap_or_default()
        .iter()
        .any(|bond| {
            bond.bond_order() == BondOrder::Double && molecule.get_atomic_number(bond.target()) == 8
        })
}

/// UFF's periodic angle form, which applies by the centre's geometry class
/// rather than by its equilibrium angle: one-fold at a linear centre,
/// three-fold at a trigonal one, four-fold at an octahedral one, and the
/// general cosine expansion everywhere else. Every implementation of the
/// paper switches on the geometry -- Towhee, whose author checked his
/// reading of the angle equations with Rappe himself, RDKit and Open Babel
/// agree on it -- and gating on `theta_0` instead shows why: an amide's
/// nitrogen carries the `N_2` type and its 111.2 degrees, and the general
/// expansion puts the minimum there, so three of them sum to 334 degrees
/// rather than 360 and the nitrogen pyramidalises, where an amide is flat
/// and the paper's own validation set would have caught it. The equilibrium
/// angle still sets the force constant, and for the general expansion the
/// minimum too.
///
/// The cost of the switch is measured rather than assumed: against
/// GFN2-xTB the flat amide is worth some twenty kcal/mol over the
/// pyramidal one, while an aniline's nitrogen -- genuinely pyramidal, and
/// the one `N_2` type cannot be right about both -- loses about one to
/// the same change, and thiophene's sulfur angle about the same. The
/// amide is the chemistry that matters.
fn angle_fold(shape: Hybridization) -> Option<u32> {
    match shape {
        Hybridization::Sp => Some(1),
        Hybridization::Sp2 => Some(3),
        Hybridization::Sp3d2 => Some(4),
        _ => None,
    }
}

fn is_group_six(atomic_number: u8) -> bool {
    matches!(atomic_number, 8 | 16 | 34 | 52 | 84)
}

impl ForceField {
    /// Builds the field for `molecule`, which must carry 3D positions only
    /// in the sense that the terms will be evaluated against them later;
    /// construction reads topology alone.
    ///
    /// # Errors
    ///
    /// [`UntypedAtom`] if any atom's element is outside the typing rules in
    /// [`typing`].
    pub fn new<M: MoleculeCore + ?Sized>(molecule: &M) -> Result<Self, UntypedAtom> {
        let types = typing::type_atoms(molecule)?;
        let count = molecule.len();

        let mut bonds = Vec::new();
        for i in 0..count {
            for bond in molecule.get_atom_bonds(i).unwrap_or_default() {
                let j = bond.target();
                if j <= i {
                    continue;
                }
                let order = bond_order_value(molecule, i, bond);
                let rest_length = terms::rest_length(types[i], types[j], order);
                bonds.push(Bond {
                    i,
                    j,
                    rest_length,
                    force_constant: terms::bond_force_constant(rest_length, types[i], types[j]),
                });
            }
        }

        let mut angles = Vec::new();
        for k in 0..count {
            let neighbours: Vec<_> = molecule
                .get_atom_bonds(k)
                .unwrap_or_default()
                .iter()
                .map(|b| (b.target(), bond_order_value(molecule, k, b)))
                .collect();
            for a in 0..neighbours.len() {
                for b in (a + 1)..neighbours.len() {
                    let (i, order_ik) = neighbours[a];
                    let (j, order_kj) = neighbours[b];
                    let theta0 = types[k].angle.to_radians();
                    let r12 = terms::rest_length(types[i], types[k], order_ik);
                    let r23 = terms::rest_length(types[k], types[j], order_kj);
                    let force_constant =
                        terms::angle_force_constant(theta0, r12, r23, types[i], types[j]);
                    let fold = angle_fold(hybridization(molecule, k));
                    let (sin0, cos0) = theta0.sin_cos();
                    let c2 = 1.0 / (4.0 * (sin0 * sin0).max(1e-8));
                    angles.push(Angle {
                        i,
                        k,
                        j,
                        force_constant,
                        fold,
                        c0: c2 * (2.0 * cos0 * cos0 + 1.0),
                        c1: -4.0 * c2 * cos0,
                        c2,
                    });
                }
            }
        }

        let torsions = Self::collect_torsions(molecule, &types);
        let inversions = Self::collect_inversions(molecule);
        let nonbonded = Self::collect_nonbonded(molecule, &types);
        Ok(ForceField {
            bonds,
            angles,
            torsions,
            inversions,
            nonbonded,
        })
    }

    /// UFF's torsion parameters depend on the hybridization of the two
    /// central atoms, and on a handful of special cases the paper calls out:
    /// single bonds between group-6 elements, and sp3-sp2 bonds where the
    /// sp2 end is itself conjugated.
    fn collect_torsions<M: MoleculeCore + ?Sized>(
        molecule: &M,
        types: &[&'static UffType],
    ) -> Vec<Torsion> {
        let mut torsions = Vec::new();
        for b in 0..molecule.len() {
            for bond_bc in molecule.get_atom_bonds(b).unwrap_or_default() {
                let c = bond_bc.target();
                if c <= b {
                    continue;
                }
                if !bond_carries_torsions(molecule, b, c) {
                    continue;
                }
                let order = bond_order_value(molecule, b, bond_bc);
                let (shape_b, shape_c) = (hybridization(molecule, b), hybridization(molecule, c));
                let (z_b, z_c) = (molecule.get_atomic_number(b), molecule.get_atomic_number(c));

                // The barrier UFF gives is for rotating the *bond*, not for
                // one of the quadruples about it, so it is shared out among
                // them -- otherwise ethane, with nine H-C-C-H torsions,
                // carries nine times the rotational barrier it should.
                let first = torsions.len();
                for outer_b in molecule.get_atom_bonds(b).unwrap_or_default() {
                    let i = outer_b.target();
                    if i == c {
                        continue;
                    }
                    for outer_c in molecule.get_atom_bonds(c).unwrap_or_default() {
                        let j = outer_c.target();
                        if j == b || j == i {
                            continue;
                        }
                        // Whether the sp2 end carries its conjugation onward
                        // is a property of *this* quadruple, not of the bond:
                        // in propene the barrier differs between the torsion
                        // ending on the vinyl carbon and the one ending on a
                        // methyl hydrogen.
                        let outer_sp2 = hybridization(molecule, i) == Hybridization::Sp2
                            || hybridization(molecule, j) == Hybridization::Sp2;
                        let Some((force_constant, fold, phase)) = torsion_parameters(
                            types, b, c, shape_b, shape_c, z_b, z_c, order, outer_sp2,
                        ) else {
                            continue;
                        };
                        torsions.push(Torsion {
                            i,
                            b,
                            c,
                            j,
                            force_constant,
                            fold,
                            phase,
                        });
                    }
                }
                let about_this_bond = torsions.len() - first;
                for torsion in &mut torsions[first..] {
                    torsion.force_constant /= about_this_bond as f64;
                }
            }
        }
        torsions
    }

    /// The out-of-plane terms, at every three-coordinate centre that has a
    /// preferred one.
    ///
    /// At a trigonal carbon, nitrogen or oxygen this is the stiffness
    /// against pyramidalising: the three-fold angle form already puts the
    /// flat geometry at the minimum, and this term is what holds it there
    /// against anything that would rather push the centre out of the plane.
    /// The group-15 centres are the other way round -- a specific pyramid is
    /// the minimum and flat the top of the barrier -- and this term is what
    /// puts the pyramid there at all.
    fn collect_inversions<M: MoleculeCore + ?Sized>(molecule: &M) -> Vec<Inversion> {
        let mut inversions = Vec::new();
        for k in 0..molecule.len() {
            let neighbours: Vec<usize> = molecule
                .get_atom_bonds(k)
                .unwrap_or_default()
                .iter()
                .map(|bond| bond.target())
                .collect();
            let [first, second, third] = neighbours[..] else {
                continue;
            };
            let Some((force_constant, c0, c1, c2)) = inversion_parameters(molecule, k) else {
                continue;
            };
            // Each of the three bonds takes its turn as the one out of the
            // plane of the other two.
            for (i, j, l) in [
                (first, second, third),
                (first, third, second),
                (second, third, first),
            ] {
                inversions.push(Inversion {
                    i,
                    k,
                    j,
                    l,
                    force_constant,
                    c0,
                    c1,
                    c2,
                });
            }
        }
        inversions
    }

    fn collect_nonbonded<M: MoleculeCore + ?Sized>(
        molecule: &M,
        types: &[&'static UffType],
    ) -> VanDerWaalsPairs {
        let count = molecule.len();
        let mut nonbonded = VanDerWaalsPairs::default();
        for i in 0..count {
            let hops = bond_hops_from(molecule, i);
            for j in (i + 1)..count {
                if hops[j] < NONBONDED_MIN_SEPARATION {
                    continue;
                }
                nonbonded.push(VanDerWaals {
                    i,
                    j,
                    distance: (types[i].vdw_distance * types[j].vdw_distance).sqrt(),
                    well_depth: (types[i].vdw_energy * types[j].vdw_energy).sqrt(),
                });
            }
        }
        nonbonded
    }

    /// The energy of `positions`, in kcal/mol.
    pub fn energy(&self, positions: &[Vector3D]) -> f64 {
        <Self as Energy>::energy(self, positions)
    }

    /// Relaxes `positions` in place, returning the final energy.
    pub fn minimize(&self, positions: &mut [Vector3D], max_iterations: usize) -> f64 {
        minimize(positions, self, max_iterations, GRADIENT_TOLERANCE)
    }
}

/// Both evaluations pick their instruction set per call, as MMFF's do,
/// through [`lanes::dispatch`], with the same bits whichever it picks -- see
/// [`lanes::Lanes`].
impl Energy for ForceField {
    fn energy(&self, positions: &[Vector3D]) -> f64 {
        struct Run<'a>(&'a ForceField, &'a [Vector3D]);
        impl lanes::LanesOp for Run<'_> {
            type Output = f64;
            #[inline(always)]
            fn run<L: lanes::Lanes>(self, simd: L) -> f64 {
                self.0.energy_in(simd, self.1)
            }
        }
        lanes::dispatch(Run(self, positions))
    }

    fn energy_and_gradient(&self, positions: &[Vector3D], gradient: &mut [Vector3D]) -> f64 {
        struct Run<'a>(&'a ForceField, &'a [Vector3D], &'a mut [Vector3D]);
        impl lanes::LanesOp for Run<'_> {
            type Output = f64;
            #[inline(always)]
            fn run<L: lanes::Lanes>(self, simd: L) -> f64 {
                self.0.energy_and_gradient_in(simd, self.1, self.2)
            }
        }
        lanes::dispatch(Run(self, positions, gradient))
    }
}

impl ForceField {
    /// [`Energy::energy`] on the lanes `simd` provides. Inlined whole, so
    /// that inside [`pulp`]'s dispatch every term is compiled for the
    /// instruction set it found.
    #[inline(always)]
    fn energy_in<L: lanes::Lanes>(&self, simd: L, positions: &[Vector3D]) -> f64 {
        self.bonds
            .iter()
            .map(|b| terms::bond_energy(positions, b))
            .sum::<f64>()
            + self
                .angles
                .iter()
                .map(|a| terms::angle_energy(positions, a))
                .sum::<f64>()
            + self
                .torsions
                .iter()
                .map(|t| terms::torsion_energy(positions, t))
                .sum::<f64>()
            + self
                .inversions
                .iter()
                .map(|v| terms::inversion_energy(positions, v))
                .sum::<f64>()
            + terms::van_der_waals_energies_in(simd, &self.nonbonded, positions)
    }

    /// [`Energy::energy_and_gradient`] on the lanes `simd` provides.
    #[inline(always)]
    fn energy_and_gradient_in<L: lanes::Lanes>(
        &self,
        simd: L,
        positions: &[Vector3D],
        gradient: &mut [Vector3D],
    ) -> f64 {
        gradient.fill(Vector3D::default());
        let mut energy = 0.0;
        for bond in &self.bonds {
            energy += terms::bond_energy_and_gradient(positions, bond, gradient);
        }
        for angle in &self.angles {
            energy += terms::angle_energy_and_gradient(positions, angle, gradient);
        }
        for torsion in &self.torsions {
            energy += terms::torsion_energy_and_gradient(positions, torsion, gradient);
        }
        for inversion in &self.inversions {
            energy += terms::inversion_energy_and_gradient(positions, inversion, gradient);
        }
        energy += terms::van_der_waals_energies_and_gradient_in(simd, &self.nonbonded, positions, gradient);
        energy
    }
}

/// Whether a bond gets torsion terms at all.
///
/// UFF as RDKit builds it restrains torsions only about bonds whose ends
/// are both non-terminal and neither triple-bonded -- its
/// `[!$(*#*)&!D1]~[!$(*#*)&!D1]`. Both exclusions are the same point: there
/// is no torsion to speak of when one end has nothing else attached, and
/// none about a linear centre. Including them anyway put a carbonyl's C=O,
/// whose oxygen is terminal, into the sum and left formic acid 2.1 kcal/mol
/// above RDKit's answer for the same coordinates.
fn bond_carries_torsions<M: MoleculeCore + ?Sized>(molecule: &M, b: usize, c: usize) -> bool {
    [b, c].iter().all(|&atom| {
        let bonds = molecule.get_atom_bonds(atom).unwrap_or_default();
        bonds.len() > 1
            && !bonds
                .iter()
                .any(|bond| bond.bond_order() == BondOrder::Triple)
    })
}

/// Bonds traversed from `start` to every atom, saturating at
/// [`NONBONDED_MIN_SEPARATION`] since nothing beyond that distinction
/// matters here.
fn bond_hops_from<M: MoleculeCore + ?Sized>(molecule: &M, start: usize) -> Vec<usize> {
    let mut hops = vec![usize::MAX; molecule.len()];
    hops[start] = 0;
    let mut queue = std::collections::VecDeque::from([start]);
    while let Some(atom) = queue.pop_front() {
        if hops[atom] >= NONBONDED_MIN_SEPARATION {
            continue;
        }
        for bond in molecule.get_atom_bonds(atom).unwrap_or_default() {
            if hops[bond.target()] == usize::MAX {
                hops[bond.target()] = hops[atom] + 1;
                queue.push_back(bond.target());
            }
        }
    }
    hops
}

/// `(K, c0, c1, c2)` for the out-of-plane term at a three-coordinate `atom`,
/// or `None` where UFF defines none.
///
/// Trigonal carbon, nitrogen and oxygen want to be flat, so the expansion
/// reduces to `K * (1 - cos(omega))`. The group-15 centres want a specific
/// pyramid instead, and UFF gives each its own `omega_0`. The paper quotes
/// no barrier for those, so the 22 kcal/mol here is RDKit's reading of it,
/// after Towhee's `ffuff`: the cost of flattening the pyramid, taken over
/// all three permutations at once.
pub(crate) fn inversion_parameters<M: MoleculeCore + ?Sized>(
    molecule: &M,
    atom: usize,
) -> Option<(f64, f64, f64, f64)> {
    // UFF quotes one barrier for inverting the centre; it is shared out
    // among the three ways of choosing which bond leaves the plane.
    const PERMUTATIONS: f64 = 3.0;
    /// A carbonyl carbon is far stiffer against pyramidalisation than a
    /// plain trigonal one -- UFF's own 50 against 6 kcal/mol.
    const CARBONYL_BARRIER: f64 = 50.0;
    const TRIGONAL_BARRIER: f64 = 6.0;
    const PYRAMIDAL_BARRIER: f64 = 22.0;

    let atomic_number = molecule.get_atomic_number(atom);
    match atomic_number {
        // Carbon, nitrogen and oxygen.
        6..=8 => {
            let trigonal = molecule.is_atom_aromatic(atom)
                || hybridization(molecule, atom) == Hybridization::Sp2;
            if !trigonal {
                return None;
            }
            let carbonyl = atomic_number == 6
                && molecule
                    .get_atom_bonds(atom)
                    .unwrap_or_default()
                    .iter()
                    .any(|bond| {
                        bond.bond_order() == BondOrder::Double
                            && molecule.get_atomic_number(bond.target()) == 8
                    });
            let barrier = if carbonyl {
                CARBONYL_BARRIER
            } else {
                TRIGONAL_BARRIER
            };
            Some((barrier / PERMUTATIONS, 1.0, -1.0, 0.0))
        }
        // Phosphorus, arsenic, antimony and bismuth.
        15 | 33 | 51 | 83 => {
            let omega0: f64 = match atomic_number {
                15 => 84.4339,
                33 => 86.9735,
                51 => 87.7047,
                _ => 90.0,
            };
            let (c1, c2) = (-4.0 * omega0.to_radians().cos(), 1.0);
            let c0 = -(c1 * omega0.to_radians().cos() + c2 * (2.0 * omega0).to_radians().cos());
            // The coefficient sum is the energy at the flat geometry, so
            // the barrier has to be divided by it: without that, flattening
            // costs 22 times the sum -- 36 kcal/mol for a phosphine, 44 for
            // a bismuthine -- instead of the 22 the term is quoted at.
            let flat = c0 + c1 + c2;
            Some((PYRAMIDAL_BARRIER / (flat * PERMUTATIONS), c0, c1, c2))
        }
        _ => None,
    }
}

/// `(V, n, cos(n * phi0))` for the bond `b-c`, or `None` where UFF defines
/// no torsion -- about a bond to a linear centre there is no angle to
/// restrain.
#[allow(clippy::too_many_arguments)]
fn torsion_parameters(
    types: &[&'static UffType],
    b: usize,
    c: usize,
    shape_b: Hybridization,
    shape_c: Hybridization,
    z_b: u8,
    z_c: u8,
    order: f64,
    outer_sp2: bool,
) -> Option<(f64, u32, f64)> {
    use Hybridization::{Sp2, Sp3};

    // Conjugation between two sp2 centres, UFF's equation 17.
    let conjugated = |order: f64| {
        5.0 * (types[b].sp2_torsion_contribution * types[c].sp2_torsion_contribution).sqrt()
            * (1.0 + 4.18 * order.ln())
    };

    Some(match (shape_b, shape_c) {
        (Sp3, Sp3) => {
            // A single bond between two group-6 atoms -- a peroxide, a
            // disulfide -- has a two-fold preference of its own rather than
            // the three-fold one of an ordinary sp3-sp3 bond.
            if order == 1.0 && is_group_six(z_b) && is_group_six(z_c) {
                let barrier = |z: u8| -> f64 { if z == 8 { 2.0 } else { 6.8 } };
                ((barrier(z_b) * barrier(z_c)).sqrt(), 2, -1.0)
            } else {
                (
                    (types[b].sp3_torsion_barrier * types[c].sp3_torsion_barrier).sqrt(),
                    3,
                    -1.0,
                )
            }
        }
        (Sp2, Sp2) => (conjugated(order), 2, 1.0),
        (Sp2, Sp3) | (Sp3, Sp2) => {
            let (sp3_z, sp2_z) = if shape_b == Sp3 {
                (z_b, z_c)
            } else {
                (z_c, z_b)
            };
            if order == 1.0 && is_group_six(sp3_z) && !is_group_six(sp2_z) {
                (conjugated(order), 2, -1.0)
            } else if order == 1.0 && outer_sp2 {
                // Propene-like: the torsion ends on another sp2 centre, so
                // the bond has a three-fold preference instead.
                (2.0, 3, -1.0)
            } else {
                (1.0, 6, 1.0)
            }
        }
        _ => return None,
    })
}
