//! An InChI's layers made into molecules.
//!
//! The main layer fixes every σ bond and how many hydrogens each atom holds,
//! names the sets of atoms a mobile hydrogen may sit on, and gives each
//! component's charge. What it leaves open is which bonds carry π bonds, which
//! atoms carry the charge, and where any unpaired electron is. Each component
//! answers those as one degree-constrained subgraph problem -- the matching
//! the tautomer normalizer solves, over a gadget of its own -- and the
//! preferences that pick one answer out of many are the order the edges are
//! decided in.
//!
//! Every atom must spend exactly its valence:
//!
//! ```text
//! σ bonds + hydrogens + π units + unpaired electrons = a valence of its element at its charge
//! ```
//!
//! The vertex of an atom asks for what its highest neutral valence leaves
//! after its σ bonds and fixed hydrogens. Its edges are a bond's π units (two
//! parallel edges, so a bond can be triple), the hydrogens of its mobile group,
//! unpaired electrons, the rungs that drop it to a lower valence two at a
//! time, and the charges it may take.
//!
//! A charge either raises an element's highest valence or lowers it, and the
//! two are modelled differently. An ammonium, an oxonium or a borate *raises*
//! it: the atom is offered one more unit and an edge to its sign's pool, and
//! keeps the charge exactly when the pool does not take the offer back. A
//! carbocation, a carbanion or an alkoxide *lowers* it: a unit vertex either
//! spends one of the atom's units, which is the charge, or goes to the pool
//! unused. Either way the pool counts what went unclaimed, so its degree is
//! fixed by how many charges of its sign the component holds -- which is what
//! an f-factor needs. Pairs beyond the net charge are separations, each a
//! four-vertex path that sends one edge to each pool when it is spent and none
//! when it is not.
//!
//! # Which structure
//!
//! The identifier is the same for every placement, but a structure has its
//! *own* InChI, and only some placements give back the one they were read
//! from. InChI's normalization takes a proton off a charged heteroatom that
//! holds one, and puts one on an anion of `=C-O`, `C-S`, `O-O` or `=N-O`,
//! recording the move in `/p` -- so a charge it left in `/q` was on neither.
//! The reader therefore never puts a charge there, and solves first without
//! charges on boron, silicon and the other elements chemists draw as radicals
//! when their hydrogens are missing: `[B]` in a PubChem record is a boron with
//! two unpaired electrons, not a borenium beside an amide anion. Only a
//! component that cannot be built that way is solved without those limits.
//! Within them it takes the fewest unpaired electrons, then the fewest charge
//! separations, then charges where chemists put them: a cation on nitrogen
//! before sulfur or oxygen, an anion on the oxygen of an N-oxide, a nitro or a
//! phosphoryl group before nitrogen, and on nitrogen before carbon.

use super::InChIError;
use super::elements::{Element, element};
use super::layers::{InChI, InChIComponent, MobileGroup, Parity, StereoBond, StereoCentre};
use crate::atom::Atom;
use crate::molecule::BondTarget;
use crate::molecule::chirality::{ChiralClass, index_sorted_neighbors, tetrahedral_tag_for_order};
use crate::molecule::matching::DegreeConstrainedInstance;
use crate::prelude::*;

/// The most unpaired electrons a component is searched for. A record needing
/// more than this is not one any valence model describes.
const MOST_RADICALS: u32 = 12;

/// A component as the solver leaves it: every atom's charge, hydrogens and
/// unpaired electrons, and every bond's order.
struct Solved {
    charges: Vec<i32>,
    hydrogens: Vec<u32>,
    radicals: Vec<u32>,
    /// Parallel to the component's `bonds`.
    orders: Vec<u8>,
}

/// What the solver is told about one component, after the record's protons
/// have been shared out between them.
struct Prepared<'a> {
    component: &'a InChIComponent,
    hydrogens: Vec<u32>,
    groups: Vec<MobileGroup>,
    charge: i32,
    /// Charges settled before solving: a hydrogen halide `/p` takes its proton
    /// from.
    fixed_charges: Vec<Option<i32>>,
    neighbours: Vec<Vec<usize>>,
}

fn is_halogen(atomic_number: u8) -> bool {
    matches!(atomic_number, 9 | 17 | 35 | 53 | 85)
}

fn is_chalcogen(atomic_number: u8) -> bool {
    matches!(atomic_number, 8 | 16 | 34 | 52)
}

/// The elements whose charges InChI's proton rules are about: a cation of
/// one holding a hydrogen gives that hydrogen to `/p`.
fn is_protic(atomic_number: u8) -> bool {
    matches!(atomic_number, 7 | 8 | 15 | 16 | 33 | 34 | 52)
}

/// The elements a structure is drawn with charges on. Boron, silicon,
/// germanium and the rest are drawn with unpaired electrons instead when their
/// hydrogens are missing, and get a charge only when nothing else fits.
fn is_conventionally_charged(atomic_number: u8) -> bool {
    matches!(
        atomic_number,
        1 | 6 | 7 | 8 | 9 | 15 | 16 | 17 | 33 | 34 | 35 | 52 | 53
    )
}

/// The valences an atom may have at `charge`.
///
/// Neutral nitrogen is held to three: InChI's table allows it five so that
/// `N(=O)=O` can be read, but no other reader takes a nitro group written that
/// way, and three makes the solver write `[N+](=O)[O-]`. A neutral halogen
/// climbs past one only where it must -- beside a chalcogen, as in a chlorate,
/// or with more σ bonds and hydrogens than one -- since a π bond is all the
/// higher valences would otherwise buy: `[B]Cl`, with its unpaired electrons,
/// came back as `B#Cl`.
fn valences(atomic_number: u8, data: &Element, charge: i32, hypervalent: bool) -> &'static [u8] {
    match (atomic_number, charge) {
        (7, 0) => &[3],
        (17 | 35 | 53, 0) if !hypervalent => &[1],
        _ => data.valences_at(charge),
    }
}

/// Whether a charge of `sign` raises the element's highest valence (`+1`) or
/// lowers it (`-1`), or `None` where the element has no valence at that
/// charge. Charged valences are not the neutral ones shifted by one: a
/// phosphorus anion has 2, 4 and 6 where the neutral atom has 3 and 5, so
/// `[PF6-]` is the neutral top raised by one, while a phosphonium's 4 is the
/// neutral top lowered by one.
fn charge_shift(
    atomic_number: u8,
    data: &Element,
    sign: i32,
    beside_chalcogen: bool,
) -> Option<i32> {
    let neutral = *valences(atomic_number, data, 0, beside_chalcogen)
        .iter()
        .max()?;
    let charged = *valences(atomic_number, data, sign, beside_chalcogen)
        .iter()
        .max()?;
    Some(if charged > neutral { 1 } else { -1 })
}

/// Whether the element's anion has a valence one below a neutral one, which is
/// what a negative charge spending one of the atom's units needs. Sulfur's
/// anion raises its highest valence but still lowers each of the others.
fn lowers(atomic_number: u8, data: &Element, hypervalent: bool) -> bool {
    let neutral = valences(atomic_number, data, 0, hypervalent);
    valences(atomic_number, data, -1, hypervalent)
        .iter()
        .any(|&charged| neutral.contains(&(charged + 1)))
}

/// Builds every component of `inchi`.
pub(super) fn build(inchi: &InChI) -> Result<Vec<MolecularGraph>, InChIError> {
    let mut prepared: Vec<Prepared> = inchi
        .components
        .iter()
        .map(|component| {
            let mut neighbours = vec![Vec::new(); component.elements.len()];
            for &(a, b) in &component.bonds {
                neighbours[a].push(b);
                neighbours[b].push(a);
            }
            Prepared {
                component,
                hydrogens: component.hydrogens.clone(),
                groups: component.mobile_groups.clone(),
                charge: component.charge,
                fixed_charges: vec![None; component.elements.len()],
                neighbours,
            }
        })
        .collect();
    let removals = take_protons_before_solving(&mut prepared, inchi.protons);

    let mut solved = Vec::with_capacity(prepared.len());
    for (index, component) in prepared.iter().enumerate() {
        solved.push(solve(component).ok_or(InChIError::NoStructure(index + 1))?);
    }

    let mut bare_protons = 0;
    if removals > 0 {
        take_protons_after_solving(&prepared, &mut solved, removals)?;
    } else if inchi.protons > 0 {
        bare_protons = add_protons(&prepared, &mut solved, inchi.protons as u32);
    }

    let mut molecules = Vec::with_capacity(prepared.len() + bare_protons as usize);
    let mut exchangeable = inchi.exchangeable_isotopes;
    for (component, solved) in prepared.iter().zip(&solved) {
        molecules.push(assemble(component, solved, &mut exchangeable)?);
    }
    if exchangeable.iter().any(|&count| count > 0) {
        return Err(InChIError::Inconsistent(
            "more exchangeable isotopic hydrogens than exchangeable hydrogens".into(),
        ));
    }
    for _ in 0..bare_protons {
        let mut proton = MolecularGraph::default();
        let mut atom = Atom::new(1);
        atom.charge = 1;
        proton.add_atom(atom);
        molecules.push(proton);
    }
    Ok(molecules)
}

/// Takes what protons `/p` removes from where nothing about the structure
/// depends on it, and returns how many are still to be taken once the
/// structure is known.
///
/// `/p` says InChI moved protons on or off the structure and not where: its
/// normalization neutralized charged acids and bases and kept only the count.
/// The reader puts them back where the same normalization would take them
/// off again, so the structure it writes has the InChI it was read from. A
/// hydrogen halide gives its proton first, and then a mobile group with an
/// oxygen or a sulfur in it -- a carboxyl, say -- one proton per group in turn.
/// The group gets a negative charge for each, which the solver places on one
/// of its atoms exactly as it places a `(H-,...)` group's.
fn take_protons_before_solving(prepared: &mut [Prepared], protons: i32) -> u32 {
    let mut left = (-protons).max(0) as u32;
    for component in prepared.iter_mut() {
        for atom in 0..component.component.elements.len() {
            if left == 0 {
                return 0;
            }
            if is_halogen(component.component.elements[atom])
                && component.hydrogens[atom] > 0
                && component.neighbours[atom].is_empty()
            {
                component.hydrogens[atom] -= 1;
                component.fixed_charges[atom] = Some(-1);
                component.charge -= 1;
                left -= 1;
            }
        }
    }
    loop {
        let mut took = false;
        for component in prepared.iter_mut() {
            let elements = &component.component.elements;
            for group in component.groups.iter_mut() {
                if left == 0 {
                    return 0;
                }
                if group.hydrogens > 0
                    && group.atoms.iter().any(|&atom| is_chalcogen(elements[atom]))
                {
                    group.hydrogens -= 1;
                    group.negative_charges += 1;
                    component.charge -= 1;
                    left -= 1;
                    took = true;
                }
            }
        }
        if !took {
            return left;
        }
    }
}

/// Whether InChI would put a proton back on `atom`, a chalcogen, were it an
/// anion: the `=C-O`, `O-O`, `C-S` and `=N-O` of its simple proton addition,
/// and water. `pi` says whether a neighbour carries a multiple bond.
fn takes_a_proton_back(prepared: &Prepared, atom: usize, pi: impl Fn(usize) -> bool) -> bool {
    let elements = &prepared.component.elements;
    let neighbours = &prepared.neighbours[atom];
    neighbours.is_empty()
        || neighbours
            .iter()
            .any(|&neighbour| match elements[neighbour] {
                8 => true,
                6 => elements[atom] != 8 || pi(neighbour),
                7 => pi(neighbour),
                _ => false,
            })
}

/// The rest of `/p`'s removals, on the solved structure: where InChI would put
/// the proton back -- a phenol, an enol, a thiol, water -- then any other
/// chalcogen, then nitrogen, then anything holding one.
fn take_protons_after_solving(
    prepared: &[Prepared],
    solved: &mut [Solved],
    mut left: u32,
) -> Result<(), InChIError> {
    for tier in 0..4 {
        for (component, solved) in prepared.iter().zip(solved.iter_mut()) {
            for (atom, &element) in component.component.elements.iter().enumerate() {
                if left == 0 {
                    return Ok(());
                }
                if solved.hydrogens[atom] == 0 || solved.charges[atom] != 0 {
                    continue;
                }
                let wanted = match tier {
                    0 => {
                        is_chalcogen(element)
                            && takes_a_proton_back(component, atom, |neighbour| {
                                has_multiple_bond(component, solved, neighbour, atom)
                            })
                    }
                    1 => is_chalcogen(element),
                    2 => element == 7,
                    _ => element != 1,
                };
                if wanted {
                    solved.hydrogens[atom] -= 1;
                    solved.charges[atom] = -1;
                    left -= 1;
                }
            }
        }
    }
    if left > 0 {
        return Err(InChIError::Inconsistent(format!(
            "{left} protons to remove and no hydrogen left to take"
        )));
    }
    Ok(())
}

/// Whether `atom` has a multiple bond to anything but `except`.
fn has_multiple_bond(prepared: &Prepared, solved: &Solved, atom: usize, except: usize) -> bool {
    prepared
        .component
        .bonds
        .iter()
        .zip(&solved.orders)
        .any(|(&(a, b), &order)| {
            order > 1 && (a == atom || b == atom) && a != except && b != except
        })
}

/// `/p`'s additions. A proton goes back on an anion first -- a betaine's
/// carboxylate written `(H-,...)` beside a quaternary nitrogen was an acid
/// before InChI took its proton -- and then on the most basic site left: an
/// amidine's imine nitrogen before an amine's, an amine's before a pyridine's,
/// and nitrogen before oxygen, sulfur and phosphorus. Returns how many found no
/// site, which the record then holds as bare protons.
fn add_protons(prepared: &[Prepared], solved: &mut [Solved], mut left: u32) -> u32 {
    let mut candidates: Vec<(u8, usize, usize)> = Vec::new();
    for (index, (component, solved)) in prepared.iter().zip(solved.iter()).enumerate() {
        let elements = &component.component.elements;
        let order = |a: usize, b: usize| -> u8 {
            component
                .component
                .bonds
                .iter()
                .zip(&solved.orders)
                .find(|(bond, _)| **bond == (a.min(b), a.max(b)))
                .map_or(1, |(_, &order)| order)
        };
        for atom in 0..elements.len() {
            if solved.radicals[atom] != 0 {
                continue;
            }
            let neighbours = &component.neighbours[atom];
            // An anion of a chalcogen or of nitrogen, unless it is the negative
            // end of a separation on nitrogen -- an azide's, a nitro group's,
            // an N-oxide's -- which InChI never took a proton from. A
            // hydride with a proton is hydrogen, and a `[PF6-]` is no acid.
            if solved.charges[atom] < 0 {
                let paired = neighbours
                    .iter()
                    .any(|&next| solved.charges[next] > 0 && elements[next] == 7);
                let terminal = neighbours.len() <= 1;
                if !paired && ((is_chalcogen(elements[atom]) && terminal) || elements[atom] == 7) {
                    let rank = u8::from(elements[atom] == 7);
                    candidates.push((rank, index, atom));
                }
                continue;
            }
            if solved.charges[atom] != 0 {
                continue;
            }
            let pi: u8 = neighbours.iter().map(|&next| order(atom, next) - 1).sum();
            let rank = match elements[atom] {
                7 => {
                    let amidine = pi == 1
                        && neighbours.iter().any(|&carbon| {
                            order(atom, carbon) == 2
                                && elements[carbon] == 6
                                && component.neighbours[carbon]
                                    .iter()
                                    .any(|&other| other != atom && elements[other] == 7)
                        });
                    let acyl = neighbours.iter().any(|&next| {
                        component.neighbours[next].iter().any(|&far| {
                            far != atom && order(next, far) == 2 && is_chalcogen(elements[far])
                        })
                    });
                    match (amidine, pi, acyl) {
                        (true, _, _) => 3,
                        (false, 0, false) => 4,
                        (false, 1, false) => 5,
                        _ => 6,
                    }
                }
                8 => 7,
                16 | 34 | 52 => 8,
                15 | 33 => 9,
                _ => continue,
            };
            candidates.push((rank, index, atom));
        }
    }
    candidates.sort_unstable();
    for (_, index, atom) in candidates {
        if left == 0 {
            break;
        }
        solved[index].hydrogens[atom] += 1;
        solved[index].charges[atom] += 1;
        left -= 1;
    }
    left
}

/// Places the π bonds, charges and unpaired electrons of one component: first
/// with charges only where they are conventional, then, if that has no
/// answer, anywhere; and each time with as few unpaired electrons as will do.
fn solve(prepared: &Prepared) -> Option<Solved> {
    let component = prepared.component;
    let atoms = component.elements.len();
    let mut solved = Solved {
        charges: vec![0; atoms],
        hydrogens: prepared.hydrogens.clone(),
        radicals: vec![0; atoms],
        orders: vec![1; component.bonds.len()],
    };
    for (atom, charge) in prepared.fixed_charges.iter().enumerate() {
        if let Some(charge) = charge {
            solved.charges[atom] = *charge;
        }
    }
    if atoms == 1 && prepared.groups.is_empty() {
        // A lone atom has nothing to share its charge with, so it holds the
        // component's whole charge, however large -- `[O-2]`, `[Fe+3]` -- and
        // whatever valence its hydrogens leave is unpaired electrons.
        let atomic_number = component.elements[0];
        solved.charges[0] = prepared.charge;
        if let Some(data) = element(atomic_number).filter(|data| !data.metal) {
            let held = solved.hydrogens[0];
            if let Some(&valence) = valences(atomic_number, data, prepared.charge, false)
                .iter()
                .find(|&&valence| u32::from(valence) >= held)
            {
                solved.radicals[0] = u32::from(valence) - held;
            }
        }
        return Some(solved);
    }
    for conventional in [true, false] {
        for radicals in 0..=MOST_RADICALS {
            for separations in SEPARATIONS {
                match solve_with(prepared, radicals, conventional, separations, &mut solved) {
                    Attempt::Solved => return Some(solved),
                    Attempt::Infeasible { offered_all: true } => break,
                    Attempt::Infeasible { offered_all: false } => {}
                }
            }
        }
    }
    None
}

/// How many charge separations are offered, in turn. Each is a path whose
/// ends join both charge pools, and a pool's gadget grows with the product of
/// its degree and what it leaves unclaimed -- so offering every pair the
/// atoms could form, a few hundred in a phosphorothioate oligonucleotide, made
/// one solve take fifteen seconds. Nearly every structure needs none, and the
/// answer does not depend on how many spare ones there are: the solver takes
/// as few as it can, and they are interchangeable.
const SEPARATIONS: [usize; 5] = [0, 1, 4, 16, usize::MAX];

/// What one attempt at a component found.
enum Attempt {
    Solved,
    /// No structure with these limits; `offered_all` says whether raising the
    /// number of separations could change that.
    Infeasible {
        offered_all: bool,
    },
}

/// What each edge of the gadget stands for, so a selection can be read back.
#[derive(Clone, Copy)]
enum Role {
    Pi(usize),
    Hydrogen(usize),
    GroupNegative(usize),
    Radical(usize),
    /// A raising charge of the given sign, offered to an atom: taken means
    /// given back, so the charge is held when the edge is *not* taken.
    Offer(usize, i32),
    /// A lowering charge of the given sign, spent on an atom.
    Spend(usize, i32),
    Other,
}

/// Tries `radicals` unpaired electrons and at most `separations` charge
/// separations, with charges only where they are conventional or anywhere.
/// Fills `solved` when it can.
fn solve_with(
    prepared: &Prepared,
    radicals: u32,
    conventional: bool,
    separations: usize,
    solved: &mut Solved,
) -> Attempt {
    let component = prepared.component;
    let elements = &component.elements;
    let atoms = elements.len();
    let bonds = &component.bonds;
    let neighbours = &prepared.neighbours;
    let mut group_of: Vec<Option<usize>> = vec![None; atoms];
    for (index, group) in prepared.groups.iter().enumerate() {
        for &atom in &group.atoms {
            group_of[atom] = Some(index);
        }
    }
    let beside_chalcogen: Vec<bool> = (0..atoms)
        .map(|atom| {
            neighbours[atom].len() + prepared.hydrogens[atom] as usize > 1
                || neighbours[atom]
                    .iter()
                    .any(|&next| is_chalcogen(elements[next]))
        })
        .collect();

    // Which atoms the valence model covers. A metal -- standard InChI keeps
    // few of them bonded -- or an element with no valence at all is left with
    // its σ bonds and nothing else, and so is an atom with more σ bonds and
    // hydrogens than any valence it has: InChI stores the six fluorines of a
    // hexafluorosilicate without asking, and the silicon is left as written.
    let overvalent = |atom: usize, data: &Element| {
        let held = neighbours[atom].len() + prepared.hydrogens[atom] as usize;
        (-1..=1).all(|charge| {
            valences(elements[atom], data, charge, beside_chalcogen[atom])
                .iter()
                .all(|&valence| usize::from(valence) < held)
        })
    };
    let data: Vec<Option<&Element>> = elements
        .iter()
        .enumerate()
        .map(|(atom, &atomic_number)| {
            element(atomic_number).filter(|data| {
                !data.metal
                    && !valences(atomic_number, data, 0, beside_chalcogen[atom]).is_empty()
                    && !overvalent(atom, data)
            })
        })
        .collect();
    // Such an atom is the centre of an ate complex, and holds what charge the
    // component has: `[SiF6-2]`.
    let ate = (0..atoms).find(|&atom| {
        data[atom].is_none() && element(elements[atom]).is_some_and(|data| !data.metal)
    });
    // An atom that could take a π bond, from the σ count alone: what the
    // `=C-O` and `=N-O` tests ask of a neighbour before anything is solved.
    let unsaturated = |atom: usize| match data[atom] {
        Some(info) => {
            let top = valences(elements[atom], info, 0, beside_chalcogen[atom])
                .iter()
                .max()
                .copied()
                .unwrap_or(0);
            neighbours[atom].len() + (prepared.hydrogens[atom] as usize) < usize::from(top)
        }
        None => false,
    };
    // The charges the gadgets have to place: the component's, less what the
    // mobile groups and the settled atoms already hold.
    let settled: i32 = prepared.fixed_charges.iter().flatten().sum();
    let group_negatives: i32 = prepared
        .groups
        .iter()
        .map(|group| group.negative_charges as i32)
        .sum();
    let mut rest = prepared.charge - settled + group_negatives;
    let mut ate_charge = 0;
    if ate.is_some() {
        ate_charge = rest;
        rest = 0;
    }
    // The charges InChI would not have left in `/q`; see the module notes. It
    // takes a proton off every charged heteroatom holding one, but puts one on
    // a chalcogen anion only to bring a negative total down: a betaine's
    // carboxylate beside its quaternary nitrogen stays as it is.
    let allowed = |atom: usize, sign: i32| -> bool {
        if !conventional {
            return true;
        }
        let atomic_number = elements[atom];
        if !is_conventionally_charged(atomic_number) {
            return false;
        }
        if sign > 0 {
            !(is_protic(atomic_number) && prepared.hydrogens[atom] > 0) && group_of[atom].is_none()
        } else {
            rest >= 0
                || !(is_chalcogen(atomic_number)
                    && takes_a_proton_back(prepared, atom, unsaturated))
        }
    };

    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut roles: Vec<Role> = Vec::new();
    let mut required: Vec<i64> = vec![0; atoms];
    let add_vertex = |required: &mut Vec<i64>, need: i64| {
        required.push(need);
        required.len() - 1
    };
    let positive_pool = add_vertex(&mut required, 0);
    let negative_pool = add_vertex(&mut required, 0);
    let radical_pool = add_vertex(&mut required, i64::from(radicals));
    let hydrogen_pools: Vec<usize> = prepared
        .groups
        .iter()
        .map(|group| add_vertex(&mut required, i64::from(group.hydrogens)))
        .collect();
    let group_negative_pools: Vec<usize> = prepared
        .groups
        .iter()
        .map(|group| add_vertex(&mut required, i64::from(group.negative_charges)))
        .collect();

    // Rungs and charge gadgets, atom by atom.
    let mut rung_edges: Vec<usize> = Vec::new();
    let mut unclaimed = [0i64; 2]; // positive, negative
    // Each charge an atom may take, by the edge that declines it.
    let mut declines: Vec<(usize, usize, i32, i32)> = Vec::new(); // (edge, atom, sign, shift)
    let mut capable = [0usize; 2];
    for atom in 0..atoms {
        let Some(info) = data[atom] else {
            continue;
        };
        let atomic_number = elements[atom];
        let fixed = prepared.fixed_charges[atom];
        let own = valences(
            atomic_number,
            info,
            fixed.unwrap_or(0),
            beside_chalcogen[atom],
        );
        let Some(&top) = own.iter().max() else {
            continue;
        };
        let mut sorted: Vec<u8> = own.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let levels = sorted
            .windows(2)
            .take_while(|pair| pair[0] - pair[1] == 2)
            .count();
        required[atom] =
            i64::from(top) - neighbours[atom].len() as i64 - i64::from(prepared.hydrogens[atom]);
        for _ in 0..levels {
            let first = add_vertex(&mut required, 1);
            let second = add_vertex(&mut required, 1);
            for vertex in [first, second] {
                rung_edges.push(edges.len());
                edges.push((atom, vertex));
                roles.push(Role::Other);
            }
            edges.push((first, second));
            roles.push(Role::Other);
        }
        if fixed.is_some() {
            continue;
        }
        for (slot, sign) in [(0usize, 1i32), (1, -1)] {
            if !allowed(atom, sign) {
                continue;
            }
            let Some(shift) = charge_shift(atomic_number, info, sign, beside_chalcogen[atom])
            else {
                continue;
            };
            let pool = if sign > 0 {
                positive_pool
            } else {
                negative_pool
            };
            capable[slot] += 1;
            unclaimed[slot] += 1;
            if shift > 0 {
                required[atom] += 1;
                declines.push((edges.len(), atom, sign, shift));
                edges.push((atom, pool));
                roles.push(Role::Offer(atom, sign));
            } else {
                let unit = add_vertex(&mut required, 1);
                edges.push((unit, atom));
                roles.push(Role::Spend(atom, sign));
                declines.push((edges.len(), atom, sign, shift));
                edges.push((unit, pool));
                roles.push(Role::Other);
            }
        }
    }

    required[positive_pool] = unclaimed[0] - i64::from(rest.max(0));
    required[negative_pool] = unclaimed[1] - i64::from((-rest).max(0));
    let mut unused_separations: Vec<usize> = Vec::new();
    let possible = capable[0].min(capable[1]);
    let offered_all = separations >= possible;
    let infeasible = Attempt::Infeasible { offered_all };
    for _ in 0..possible.min(separations) {
        let z = add_vertex(&mut required, 1);
        let d1 = add_vertex(&mut required, 1);
        let d2 = add_vertex(&mut required, 1);
        let y = add_vertex(&mut required, 1);
        edges.push((z, negative_pool));
        roles.push(Role::Other);
        unused_separations.push(edges.len());
        edges.push((z, d1));
        roles.push(Role::Other);
        edges.push((d1, d2));
        roles.push(Role::Other);
        edges.push((d2, y));
        roles.push(Role::Other);
        edges.push((y, positive_pool));
        roles.push(Role::Other);
    }

    // π units, two per bond between atoms the model covers.
    let mut pi_edges: Vec<(usize, usize)> = Vec::new(); // (first unit edge, bond index)
    for (index, &(a, b)) in bonds.iter().enumerate() {
        if data[a].is_none() || data[b].is_none() {
            continue;
        }
        pi_edges.push((edges.len(), index));
        for _ in 0..2 {
            edges.push((a, b));
            roles.push(Role::Pi(index));
        }
    }

    // Hydrogens and negative charges of the mobile groups, and unpaired
    // electrons. An atom is offered as many as it could hold.
    let mut hydrogen_edges: Vec<(usize, usize, usize)> = Vec::new(); // (round, edge, atom)
    let mut group_negative_edges: Vec<(usize, usize)> = Vec::new(); // (edge, atom)
    for atom in 0..atoms {
        let Some(info) = data[atom] else {
            continue;
        };
        let room = required[atom].max(0) as usize;
        if let Some(group) = group_of[atom] {
            let count = room.min(prepared.groups[group].hydrogens as usize);
            for round in 0..count {
                hydrogen_edges.push((round, edges.len(), atom));
                edges.push((atom, hydrogen_pools[group]));
                roles.push(Role::Hydrogen(atom));
            }
            if prepared.groups[group].negative_charges > 0
                && lowers(elements[atom], info, beside_chalcogen[atom])
            {
                group_negative_edges.push((edges.len(), atom));
                edges.push((atom, group_negative_pools[group]));
                roles.push(Role::GroupNegative(atom));
            }
        }
        if radicals > 0 {
            for _ in 0..room.min(radicals as usize) {
                edges.push((atom, radical_pool));
                roles.push(Role::Radical(atom));
            }
        }
    }
    if required.iter().any(|&need| need < 0) {
        return infeasible;
    }
    let required: Vec<usize> = required.into_iter().map(|need| need as usize).collect();
    let Some(mut instance) = DegreeConstrainedInstance::build(required.len(), &edges, &required)
    else {
        return infeasible;
    };

    // The preferences, most important first.
    let mut priority: Vec<(usize, bool)> = Vec::new();
    // A double bond the string gives a geometry is double.
    let stereo_bonds: Vec<(usize, usize)> = component
        .double_bonds
        .iter()
        .chain(component.isotopic_double_bonds.iter().flatten())
        .map(|bond| (bond.first.min(bond.second), bond.first.max(bond.second)))
        .collect();
    for &(edge, index) in &pi_edges {
        if stereo_bonds.contains(&bonds[index]) {
            priority.push((edge, true));
            priority.push((edge + 1, false));
        }
    }
    // As few charge separations as the structure allows.
    priority.extend(unused_separations.iter().map(|&edge| (edge, true)));
    // Then the sites, turned down from least to most plausible, so that what
    // is left when the count runs out is the most plausible.
    // An onium whose σ bonds already exceed its neutral valence by one -- a
    // quaternary ammonium or phosphonium, a sulfonium -- holds the charge
    // without moving a π bond, so it is the first choice.
    let onium = |atom: usize| {
        data[atom].is_some_and(|info| {
            valences(elements[atom], info, 0, beside_chalcogen[atom])
                .iter()
                .min()
                .is_some_and(|&lowest| {
                    neighbours[atom].len() + prepared.hydrogens[atom] as usize
                        == usize::from(lowest) + 1
                })
        })
    };
    let plus_rank = |atom: usize| -> u8 {
        let atomic_number = elements[atom];
        if is_protic(atomic_number) && prepared.hydrogens[atom] > 0 {
            0
        } else if group_of[atom].is_some() {
            1
        } else if is_protic(atomic_number) && onium(atom) {
            8
        } else {
            match atomic_number {
                7 => 7,
                15 | 33 => 6,
                16 => 5,
                8 | 34 | 52 => 4,
                9 | 17 | 35 | 53 => 3,
                _ => 2,
            }
        }
    };
    // An anion the component holds net sits where InChI could not put a proton
    // on it -- a sulfonamide's nitrogen, a carbanion -- while the negative end
    // of a separation sits where chemists draw it: on the oxygen of an
    // N-oxide, a nitro or a phosphoryl group, then on any oxygen.
    let minus_rank = |atom: usize, shift: i32| -> u8 {
        let atomic_number = elements[atom];
        // An anion that raises the valence is a borate or an aluminate's;
        // a sulfur or phosphorus one is a hypervalent ate complex, which a
        // sulfonate or phosphate never needs.
        if shift > 0 {
            return if matches!(atomic_number, 5 | 13 | 31) {
                5
            } else {
                0
            };
        }
        // A chalcogen anion is a terminal one -- an oxide, a thiolate. A
        // sulfur with more bonds than one holding the charge is a hypervalent
        // ate complex nobody draws.
        let terminal_chalcogen = is_chalcogen(atomic_number) && neighbours[atom].len() <= 1;
        let on_a_heteroatom = neighbours[atom]
            .iter()
            .any(|&next| matches!(elements[next], 7 | 15 | 16 | 33 | 34));
        match (rest < 0, atomic_number) {
            (true, 7) => 4,
            (true, _) if terminal_chalcogen && on_a_heteroatom => 3,
            (true, 6) => 2,
            (true, _) if terminal_chalcogen => 1,
            (false, 7 | 8 | 16 | 34 | 52)
                if on_a_heteroatom && (atomic_number == 7 || terminal_chalcogen) =>
            {
                4
            }
            (false, _) if terminal_chalcogen => 3,
            (false, 7) => 2,
            (false, 6) => 1,
            _ => 0,
        }
    };
    // Carbon is turned down first, both signs: a carbanion or a carbocation is
    // the last thing to draw where a heteroatom could hold the charge, and a
    // cation put on the wrong nitrogen otherwise left a pyridine ring to buy
    // its π bonds with one.
    let mut ordered: Vec<(u8, u8, usize, usize)> = declines
        .iter()
        .map(|&(edge, atom, sign, shift)| {
            let rank = if sign > 0 {
                plus_rank(atom)
            } else {
                minus_rank(atom, shift)
            };
            let phase = match (elements[atom], sign > 0) {
                (6, _) => 0,
                (_, true) => 1,
                (_, false) => 2,
            };
            // Among equals, the end of a chain: an azide's outer nitrogen.
            let terminal = u8::from(neighbours[atom].len() <= 1);
            (phase, 2 * rank + terminal, atom, edge)
        })
        .collect();
    ordered.sort_unstable();
    priority.extend(ordered.iter().map(|&(_, _, _, edge)| (edge, true)));
    // A group's negative charge on its oxygen before its sulfur, and on
    // either before its nitrogen.
    let mut group_negatives = group_negative_edges;
    group_negatives.sort_by_key(|&(_, atom)| match elements[atom] {
        7 => 0,
        16 | 34 | 52 => 1,
        _ => 2,
    });
    priority.extend(group_negatives.iter().map(|&(edge, _)| (edge, false)));
    // The lowest valence that will do.
    priority.extend(rung_edges.iter().map(|&edge| (edge, true)));
    // A mobile hydrogen on nitrogen before oxygen -- the amide, not the
    // imidic acid, and a sulfamide's `NH2`, not `S(=N)O` -- and among atoms of
    // one element, each its first before any its second.
    hydrogen_edges.sort_by_key(|&(round, _, atom)| (elements[atom] != 7, round, atom));
    priority.extend(hydrogen_edges.iter().map(|&(_, edge, _)| (edge, true)));

    let selection = instance.select(&priority);
    solved.charges = vec![0; atoms];
    for (atom, charge) in prepared.fixed_charges.iter().enumerate() {
        if let Some(charge) = charge {
            solved.charges[atom] = *charge;
        }
    }
    solved.hydrogens = prepared.hydrogens.clone();
    solved.radicals = vec![0; atoms];
    solved.orders = vec![1; bonds.len()];
    if let Some(atom) = ate {
        solved.charges[atom] += ate_charge;
    }
    for (edge, &taken) in selection.iter().enumerate() {
        match roles[edge] {
            Role::Pi(index) if taken => solved.orders[index] += 1,
            Role::Hydrogen(atom) if taken => solved.hydrogens[atom] += 1,
            Role::GroupNegative(atom) if taken => solved.charges[atom] -= 1,
            Role::Radical(atom) if taken => solved.radicals[atom] += 1,
            Role::Offer(atom, sign) if !taken => solved.charges[atom] += sign,
            Role::Spend(atom, sign) if taken => solved.charges[atom] += sign,
            _ => {}
        }
    }
    Attempt::Solved
}

/// The molecule of one solved component, with its hydrogens, isotopes and
/// stereo. `exchangeable` is what is left of the record's exchangeable
/// isotopic hydrogens, which go on the first heteroatom hydrogens there are.
fn assemble(
    prepared: &Prepared,
    solved: &Solved,
    exchangeable: &mut [u32; 3],
) -> Result<MolecularGraph, InChIError> {
    let component = prepared.component;
    let atoms = component.elements.len();
    let mut molecule = MolecularGraph::default();
    for atom in 0..atoms {
        let mut built = Atom::new(component.elements[atom]);
        built.charge = solved.charges[atom] as i8;
        built.radical_state = solved.radicals[atom] as u8;
        molecule.add_atom(built);
    }
    for (&(a, b), &order) in component.bonds.iter().zip(&solved.orders) {
        let order = match order {
            1 => BondOrder::Single,
            2 => BondOrder::Double,
            3 => BondOrder::Triple,
            _ => BondOrder::Quadruple,
        };
        molecule.add_bond(a, b, order);
    }
    // Each atom's hydrogens after it, so that a heavy atom's index is its
    // number in the string less one.
    let mut hydrogens_of: Vec<Vec<usize>> = vec![Vec::new(); atoms];
    for (atom, hydrogens) in hydrogens_of.iter_mut().enumerate() {
        for _ in 0..solved.hydrogens[atom] {
            let index = molecule.len();
            molecule.add_atom(Atom::new(1));
            molecule.add_bond(atom, index, BondOrder::Single);
            hydrogens.push(index);
        }
    }

    // Isotopes: a shifted mass on the atom, and labelled hydrogens on it or,
    // for the exchangeable ones, on whichever heteroatom holds a hydrogen.
    let mut isotopes = vec![0u16; molecule.len()];
    for isotope in &component.isotopes {
        if let Some(shift) = isotope.mass_shift {
            let reference = element(component.elements[isotope.atom]).map_or(0, |data| data.mass);
            isotopes[isotope.atom] = (i32::from(reference) + shift).max(1) as u16;
        }
        let mut available = hydrogens_of[isotope.atom].iter().copied();
        for (slot, &count) in isotope.hydrogens.iter().enumerate() {
            for _ in 0..count {
                let hydrogen = available.next().ok_or_else(|| {
                    InChIError::Inconsistent(format!(
                        "atom {} has fewer hydrogens than isotopic ones",
                        isotope.atom + 1
                    ))
                })?;
                isotopes[hydrogen] = slot as u16 + 1;
            }
        }
    }
    let heteroatom_hydrogens: Vec<usize> = (0..atoms)
        .filter(|&atom| !matches!(component.elements[atom], 1 | 6))
        .flat_map(|atom| hydrogens_of[atom].clone())
        .collect();
    for hydrogen in heteroatom_hydrogens {
        if isotopes[hydrogen] != 0 {
            continue;
        }
        if let Some(slot) = exchangeable.iter().position(|&count| count > 0) {
            exchangeable[slot] -= 1;
            isotopes[hydrogen] = slot as u16 + 1;
        }
    }
    if isotopes.iter().any(|&mass| mass != 0) {
        *molecule.isotopes_mut() = Some(isotopes);
    }

    set_stereo(&mut molecule, component);
    let _ = molecule.aromatize(
        crate::molecule::aromaticity::default_aromaticity_model(),
        None,
        false,
    );
    Ok(molecule)
}

/// The geometry `/b`, `/t` and `/m` give, or the isotopic layer's where it has
/// its own.
fn set_stereo(molecule: &mut MolecularGraph, component: &InChIComponent) {
    let double_bonds: &[StereoBond] = component
        .isotopic_double_bonds
        .as_deref()
        .unwrap_or(&component.double_bonds);
    let centres: &[StereoCentre] = component
        .isotopic_centres
        .as_deref()
        .unwrap_or(&component.centres);
    // An isotopic layer with no `/m` of its own inherits the main layer's.
    let inverted = if component.isotopic_centres.is_some() {
        component.isotopic_inverted.or(component.inverted)
    } else {
        component.inverted
    } == Some(true);
    if double_bonds.is_empty() && centres.is_empty() {
        return;
    }
    molecule.initialize_chiral_classes();
    // How InChI orders a centre's neighbours: hydrogen lowest, then
    // deuterium, then tritium, then the heavy atoms by number -- which is
    // their index here.
    let rank = |molecule: &MolecularGraph, atom: usize| -> (u8, usize) {
        if molecule.atomic_numbers()[atom] == 1 {
            let mass = molecule
                .isotopes()
                .and_then(|isotopes| isotopes.get(atom).copied())
                .unwrap_or(0);
            (0, usize::from(mass.max(1)))
        } else {
            (1, atom)
        }
    };

    for centre in centres {
        let parity = match (centre.parity, inverted) {
            (Parity::Unknown, _) => continue,
            (Parity::Plus, false) | (Parity::Minus, true) => Parity::Plus,
            _ => Parity::Minus,
        };
        let atom = centre.atom;
        let mut neighbours: Vec<usize> = molecule
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .map(|bond| bond.target())
            .collect();
        neighbours.sort_by_key(|&neighbour| rank(molecule, neighbour));
        let ranks: Vec<(u8, usize)> = neighbours
            .iter()
            .map(|&neighbour| rank(molecule, neighbour))
            .collect();
        if ranks.windows(2).any(|pair| pair[0] == pair[1]) {
            continue;
        }
        let sorted = index_sorted_neighbors(molecule, atom);
        // `+` is the rest increasing clockwise seen from the lowest -- a
        // hydrogen where there is one -- which is `@@` for that order.
        let clockwise = if parity == Parity::Plus { 2 } else { 1 };
        let tag = match neighbours.len() {
            4 => tetrahedral_tag_for_order(clockwise, &neighbours, &sorted),
            // Three neighbours and a lone pair, seen from the lone pair. A
            // stored three-neighbour tag reads the lone pair second, as SMILES
            // reads an implicit hydrogen, and moving it from first to second
            // is one transposition.
            3 => tetrahedral_tag_for_order(3 - clockwise, &neighbours, &sorted),
            _ => continue,
        };
        molecule.chiral_classes_mut().unwrap()[atom] = ChiralClass::TH(tag);
    }

    for bond in double_bonds {
        if bond.parity == Parity::Unknown {
            continue;
        }
        let (a, b) = (bond.first, bond.second);
        let is_double = molecule
            .get_atom_bonds(a)
            .unwrap_or_default()
            .iter()
            .any(|edge| edge.target() == b && edge.bond_order() == BondOrder::Double);
        if !is_double {
            continue;
        }
        // InChI's reference at each end is the highest-numbered heavy
        // neighbour; the crate's is the first neighbour in bond order. Where
        // they differ the two sit on opposite sides of the bond.
        let references = |atom: usize, partner: usize| -> Option<(usize, usize)> {
            let others: Vec<BondTarget> = molecule
                .get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .copied()
                .filter(|edge| edge.target() != partner)
                .collect();
            let inchi = others
                .iter()
                .map(|edge| edge.target())
                .max_by_key(|&target| rank(molecule, target))?;
            let stored = others
                .iter()
                .find(|edge| edge.bond_order() != BondOrder::Double)?
                .target();
            Some((inchi, stored))
        };
        let (Some((inchi_a, stored_a)), Some((inchi_b, stored_b))) =
            (references(a, b), references(b, a))
        else {
            continue;
        };
        let cis = (bond.parity == Parity::Minus) ^ (inchi_a != stored_a) ^ (inchi_b != stored_b);
        let classes = molecule.chiral_classes_mut().unwrap();
        classes[a.min(b)] = ChiralClass::CisTransCounterclockwise;
        classes[a.max(b)] = if cis {
            ChiralClass::CisTransCounterclockwise
        } else {
            ChiralClass::CisTransClockwise
        };
    }
}
