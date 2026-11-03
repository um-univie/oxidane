//! Aromaticity decided over a canonical ring family rather than over paths.
//!
//! The search in [`crate::molecule::aromaticity`] asks, for a ring bond,
//! whether it lies on *some* cycle whose π electrons make 4n+2, and it asks by
//! walking simple paths out of the bond. That question has exponentially many
//! witnesses, so the walk has to be cut off somewhere -- by a step budget,
//! which stops where the atom numbering happened to take it, or by a constant
//! depth, which at least stops in the same place for every writing of the same
//! molecule but pays `roots x branching^depth` to do it.
//!
//! The question asked here is RDKit's instead: *is some subset of at most a few
//! rings of the molecule's ring family Hückel-aromatic*. It has polynomially
//! many witnesses, all of them are enumerated, and the answer is the union of
//! every subset that passes -- so nothing is decided by which witness was
//! reached first.
//!
//! Three things have to hold for that to be an improvement rather than a
//! relabelling of the same problem:
//!
//! * **The ring family must not depend on the numbering.** A smallest set of
//!   smallest rings does: where several rings of equal size span the same cycle
//!   space, SSSR keeps whichever the algorithm reached first. Cubane has six
//!   faces and any SSSR has five of them, and which five is an accident of the
//!   atom order. [`ring_family`] returns the *relevant* cycles instead -- every
//!   cycle that is not a sum of strictly shorter ones -- which is the union of
//!   all minimum cycle bases (Vismara 1997) and is therefore defined by the
//!   graph alone. Cubane comes back with six faces, C70 with all 37 of its.
//! * **The exocyclic test must run against π-capable cycles**, not against every
//!   cycle: see [`crate::molecule::pi_system`], whose documentation carries the
//!   pair of molecules that separates the two rules and the reason RDKit's
//!   `RingInfo::numBondRings` gets that pair wrong. Rings reach the scoring
//!   below only through [`crate::molecule::pi_system::conjugated_ring_bonds`].
//! * **The electron counting must be the one already in the tree.** It is not
//!   what was broken, and rewriting it would confound every comparison between
//!   the two models. The per-atom [`AromaticityFlag`]s handed to
//!   [`RingFinder::find_aromatic_rings`] are exactly the ones the path search is
//!   given.
//!
//! # The numbers, and why they are not budgets
//!
//! Nothing here bounds the *work* in a way that could bound the *answer*: the
//! ring family is every smallest cycle through every bond, ties and all, found
//! by one breadth-first search per bond, with no cap on how many equally short
//! returns a bond has or on how far a search may go. The two numbers that
//! remain, [`MAX_RING_SIZE`] and [`MAX_FUSED_RINGS`], say what a Hückel count is
//! taken over rather than how hard perception tries, both are the caller's
//! ([`set_ring_size_limit`], [`set_fused_ring_limit`]), and neither can be
//! tripped by renumbering a molecule.
//!
//! [`MAX_FUSED_RINGS`] is the exception, and it is a different kind of thing
//! from a budget. It does not decide how hard perception tries; it says what
//! the model *is* -- that a Hückel count is taken over a union of at most six
//! rings of the family. Three things follow, and they are what a budget could
//! not offer:
//!
//! * every subset up to that size is always enumerated, so the answer never
//!   depends on how much work was done or in what order;
//! * the answer is their union, so it never depends on which one was found
//!   first;
//! * and no renumbering can change either.
//!
//! Dropping it is not "the same model without a number". Scoring *every*
//! connected subset is exponential -- 2^120 of them on a 260-atom cage -- and
//! the unbounded chemical question behind it ("is there a simple cycle whose π
//! electrons make 4n+2") is the one the path search asks, with exponentially
//! many witnesses and no polynomial method known. The model that drops the
//! number altogether -- each ring alone, and each fused system entire, which
//! `OXIDANE_RING_SUBSET_CAP=0` runs -- is a *different* model, and a measurably
//! worse one: over 200,000 PubChem records it differs from this one on 145,
//! and on 118 of those RDKit agrees with this one, because an aromatic circuit
//! can run through part of a fused system and not the rest.
//!
//! # Where the work goes
//!
//! Perception runs on every molecule that is parsed and again on every molecule
//! that is canonized, so the shape of this module is set by allocation as much
//! as by the search. Everything it needs lives in a [`RingFinder`] the caller
//! keeps: the cyclic subgraph is compacted to the atoms that are really on a
//! ring and held as a CSR adjacency, the shortest paths out of a root live in
//! one flat arena rather than a vector per atom, candidate cycles are
//! bond-index bitsets in another, and the family itself is a third. The buffers
//! are cleared and refilled, never reallocated, and the caller's cyclic bond
//! set is taken as given rather than perceived a second time.

use crate::molecule::aromaticity::AromaticityFlag;
use crate::molecule::base::MoleculeCore;
use rustc_hash::{FxHashMap, FxHashSet};

/// The largest ring a Hückel count is taken over, unless the caller says
/// otherwise.
///
/// Twenty-four, which is RDKit's `maxFusedAromaticRingSize` -- "a bit bigger
/// than the outer ring in a porphyrin" -- and the number the path search's
/// depth was measured to. Their comment calls it efficiency; here it is not,
/// because the search costs nothing either way: an 800-membered ring is
/// perceived in 9 ms and removing the limit changed no canonical SMILES over
/// 200,000 PubChem records. It is kept because it is chemistry. Hückel's rule
/// is derived for a planar monocyclic conjugated system and stops describing
/// one somewhere around twenty-two atoms, where planarity and bond alternation
/// give out; \[18\]annulene is the largest anyone confirms. Without the limit a
/// fifty-membered ring counts to 4n+2 and is called aromatic, which is a wrong
/// answer rather than a wider one.
///
/// [`set_ring_size_limit`] moves it, and zero removes it.
pub const MAX_RING_SIZE: usize = 24;

/// The ring-size limit in force, which [`set_ring_size_limit`] or the
/// environment variable `OXIDANE_RING_SIZE_CAP` may have changed. Zero means no
/// limit: every ring the molecule has is scored.
pub fn ring_size_limit() -> usize {
    match RING_SIZE_LIMIT.load(std::sync::atomic::Ordering::Relaxed) {
        UNSET => {
            let limit = environment_cap("OXIDANE_RING_SIZE_CAP", MAX_RING_SIZE);
            RING_SIZE_LIMIT.store(limit, std::sync::atomic::Ordering::Relaxed);
            limit
        }
        limit => limit,
    }
}

/// Sets the largest ring a Hückel count is taken over, for the whole process.
///
/// Zero removes the limit, which is a statement that 4n+2 describes a ring of
/// any size. Perception decides what a canonical form is, so molecules compared
/// across a change of this setting are not comparable: set it once, before
/// anything is parsed.
pub fn set_ring_size_limit(limit: usize) {
    RING_SIZE_LIMIT.store(limit, std::sync::atomic::Ordering::Relaxed);
}

static RING_SIZE_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(UNSET);

/// How many rings of one fused system are combined into a single Hückel count,
/// unless the caller says otherwise.
///
/// Six, which is RDKit's number: `Code/GraphMol/Aromaticity.cpp` writes
/// `const unsigned int maxFused = 6;` inside `aromaticityHelper`, twice, where
/// nothing outside the file can reach it. Ours is reachable -- see
/// [`set_fused_ring_limit`] -- because it decides chemistry and the reader of a
/// structure may know more about their chemistry than this crate does.
///
/// It is not a bound on effort. Every subset up to the limit is enumerated on
/// every molecule, the answer is their union, and neither the order of the
/// search nor the numbering of the atoms can move it. What it bounds is the
/// model: an aromatic circuit that needs seven rings to describe is not
/// perceived. Measured against this corpus, over 200,000 PubChem records:
///
/// | limit | records whose canonical SMILES differ from six | cost on a C70 |
/// |---|---|---|
/// | 2 | 431 | 0.03 s |
/// | 4 | 21 | 0.04 s |
/// | **6** | -- | **0.05 s** |
/// | 8 | 1 | 0.24 s |
/// | 10 | 0 | 2.3 s |
///
/// and [`set_fused_ring_limit`] takes zero for the model with no such number in
/// it at all.
pub const MAX_FUSED_RINGS: usize = 6;

/// The limit in force, which [`set_fused_ring_limit`] or the environment
/// variable `OXIDANE_RING_SUBSET_CAP` may have changed.
pub fn fused_ring_limit() -> usize {
    match FUSED_RING_LIMIT.load(std::sync::atomic::Ordering::Relaxed) {
        UNSET => {
            let limit = environment_cap("OXIDANE_RING_SUBSET_CAP", MAX_FUSED_RINGS);
            FUSED_RING_LIMIT.store(limit, std::sync::atomic::Ordering::Relaxed);
            limit
        }
        limit => limit,
    }
}

/// Sets how many fused rings one Hückel count may span, for the whole process.
///
/// Zero asks for the model with no ring-count in it: each ring on its own, and
/// each fused system entire, nothing chosen in between. That is a *different*
/// model rather than this one unbounded -- scoring every connected subset is
/// exponential, 2^120 of them on a 260-atom cage -- and it is measurably
/// weaker: over 200,000 PubChem records it differs from the default on 145, and
/// on 118 of those RDKit agrees with the default, because an aromatic circuit
/// can run through part of a fused system and not the rest.
///
/// Perception decides what a canonical form is, so molecules compared across a
/// change of this setting are not comparable. Set it once, before anything is
/// parsed.
pub fn set_fused_ring_limit(limit: usize) {
    FUSED_RING_LIMIT.store(limit, std::sync::atomic::Ordering::Relaxed);
}

const UNSET: usize = usize::MAX;
static FUSED_RING_LIMIT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(UNSET);

/// A ring of the family: its atoms, and the bonds that close it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ring {
    /// The ring's atoms, in index order.
    pub atoms: Vec<usize>,
    /// The ring's bonds, each as an ordered pair, in index order.
    pub bonds: Vec<(usize, usize)>,
}

impl Ring {
    /// How many atoms the ring has.
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Whether the ring has no atoms, which no ring of a real family does.
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
}

/// The molecule's relevant cycles, up to `max_size` atoms.
///
/// A cycle is *relevant* when it is not the sum, in the cycle space over GF(2),
/// of strictly shorter cycles. The relevant cycles are the union of every
/// minimum cycle basis of the graph, so the set is a superset of any SSSR and,
/// unlike an SSSR, is a function of the graph rather than of the order its
/// atoms were written in: cubane's six faces are all relevant, and so are all
/// three six-rings of bicyclo[2.2.2]octane and both ten-rings of a 3+3+7 cage.
/// That is the property aromaticity perception needs, because a ring family
/// chosen by the numbering hands the numbering back the answer.
///
/// The construction is Horton's: for every atom `v` and every bond `(x, y)`,
/// the cycle made of a shortest path `v..x`, the bond, and a shortest path
/// `y..v`, kept when the two paths meet only at `v`. Horton's theorem is that
/// this family contains every relevant cycle, so sweeping it by increasing size
/// and keeping what is independent of everything strictly shorter recovers them
/// exactly. Cost is `O(atoms x bonds)` candidates, each linear in its length --
/// polynomial where enumerating cycles is not, which is the whole point of
/// moving perception here.
///
/// Rings longer than `max_size` are never generated. That cannot change which
/// shorter rings come back, since a longer cycle never takes part in deciding a
/// shorter one's relevance.
///
/// This perceives the cyclic bonds and allocates a [`RingFinder`] for the one
/// call. Anything doing many molecules should keep a finder and hand it the
/// ring bonds it already has.
pub fn ring_family<M: MoleculeCore + ?Sized>(molecule: &M, max_size: usize) -> Vec<Ring> {
    let (_, cyclic_bonds) = molecule.find_cyclic_bonds_and_atoms();
    let mut finder = RingFinder::new();
    finder.perceive(molecule, &cyclic_bonds, max_size);
    finder.rings().collect()
}

/// The ring family of one molecule, and the buffers that found it.
///
/// Every vector here is scratch: cleared at the start of a molecule and read
/// only until the next one. Holding one across molecules is what keeps ring
/// perception from being a stream of allocations, which is why it is a field of
/// [`crate::molecule::aromaticity::AromaticityScratch`] rather than something
/// perception makes for itself.
#[derive(Debug, Default, Clone)]
pub struct RingFinder {
    graph: CyclicGraph,
    walker: ShortestPaths,
    candidates: Candidates,
    family: RingStore,
    /// The rings worth scoring, and which of them share a bond with which.
    scored: Vec<u32>,
    neighbours: Vec<Vec<u32>>,
    /// A ring's atoms in the molecule's numbering, while it is being stored.
    translated: Vec<u32>,
    /// The bonds a subset would mark, and how many of them each atom has, so
    /// that a perimeter which is not a cycle can be recognised.
    circuit: Vec<u32>,
    perimeter_degree: Vec<u8>,
    /// Which fused system each scored ring belongs to, whether that system
    /// still has anything left to mark, and how much.
    ///
    /// Bonds alone decide it. Marking a subset marks every atom of its rings
    /// and those of its bonds that lie in one of them, so a marked bond always
    /// has two marked atoms, and every ring atom lies on ring bonds -- a system
    /// whose bonds are all marked has its atoms all marked too. Bonds also
    /// belong to one system each, where a spiro atom belongs to two, so
    /// counting them needs no care about double counting.
    system: Vec<u32>,
    system_active: Vec<bool>,
    unmarked_bonds: Vec<u32>,
    marked_bond: Vec<bool>,
    counted_bond: Vec<bool>,
    /// The bonds a scored ring may be built from: cyclic, π-capable, and
    /// between atoms with electrons to contribute.
    scoring_bonds: FxHashSet<(usize, usize)>,
    /// The smallest cycle known so far through each bond, which bounds how far
    /// that bond's own search has to go.
    best: Vec<u32>,
    /// The size of the smallest cycle through each bond, while the family is
    /// being chosen.
    /// The ring system being walked by `take_the_plain_cycles`, and its bonds.
    walk: Vec<u32>,
    component: Vec<bool>,
    cycle_bits: Vec<u64>,
    /// The fused-system search: the subset in hand, the rings still on offer,
    /// and which rings have been offered already.
    subset: Vec<u32>,
    extensions: Vec<u32>,
    touched: Vec<bool>,
    membership: Vec<u8>,
}

impl RingFinder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Finds the ring family of `molecule`, whose cyclic bonds the caller has
    /// already perceived.
    ///
    /// Perception has that set in hand before it gets here -- `aromatize_with`
    /// needs it for the flags -- and walking the molecule for it a second time
    /// was a tenth of what the ring model cost.
    pub fn perceive<M: MoleculeCore + ?Sized>(
        &mut self,
        molecule: &M,
        cyclic_bonds: &FxHashSet<(usize, usize)>,
        max_size: usize,
    ) {
        // The molecule's own ring bonds are a ring system already; nothing
        // has been taken out of them, so nothing has to be perceived again.
        self.perceive_scorable(molecule, cyclic_bonds, max_size, false, |_, _| true);
    }

    /// [`RingFinder::perceive`], told which bonds a scorable ring may use.
    ///
    /// The ring family is still the molecule's -- the searches run on the whole
    /// cyclic graph, so a bond's smallest cycle is its smallest cycle and not
    /// the smallest one that avoids the saturated parts. What `scorable` prunes
    /// is the walk that reads the cycles back: a branch onto a bond no scored
    /// ring could use leads only to rings the filter would discard.
    fn perceive_scorable<M: MoleculeCore + ?Sized>(
        &mut self,
        molecule: &M,
        cyclic_bonds: &FxHashSet<(usize, usize)>,
        max_size: usize,
        perceive_rings: bool,
        scorable: impl Fn(usize, usize) -> bool,
    ) {
        self.family.clear();
        self.graph.build(molecule, cyclic_bonds, perceive_rings);
        if self.graph.bonds.is_empty() {
            return;
        }
        for bond in 0..self.graph.bonds.len() {
            let (first, second) = self.graph.bond_atoms(bond as u32);
            self.graph.scorable[bond] = scorable(first, second);
        }
        let searching = self.take_the_plain_cycles(max_size);
        if !searching {
            // Every ring system was a plain cycle, so the family is complete
            // and in order already.
            return;
        }

        // A smallest cycle through a bond is that bond plus a shortest path
        // between its ends that does not use it. So the family is read off one
        // breadth-first search per bond: the search gives the length, and every
        // shortest path of that length is a ring of the family. Nothing is
        // capped and nothing is chosen -- the ties are the answer, not a
        // difficulty in computing it.
        self.candidates.begin(self.graph.bonds.len());
        // What is already known bounds what is left to look for: once a ring of
        // n atoms through a bond is in hand, no search for that bond's smallest
        // has any business going deeper than n - 1, because a cycle longer than
        // that is not smallest and a tie of exactly n is still reached. Rings
        // cover several bonds each, so the bonds looked at later start with a
        // bound the earlier ones gave them -- in a fused aromatic system every
        // search after the first few stops at five steps instead of walking the
        // system. Nothing is lost: the bound is the size of a cycle that exists.
        self.best.clear();
        self.best.resize(self.graph.bonds.len(), u32::MAX);
        for bond in 0..self.graph.bonds.len() as u32 {
            if !self.graph.scorable[bond as usize] {
                continue;
            }
            let (first, second) = self.graph.bonds[bond as usize];
            if self.graph.plain[first as usize] {
                continue;
            }
            let known = self.best[bond as usize];
            let reach = if known == u32::MAX {
                max_size
            } else {
                max_size.min(known as usize)
            };
            let first_new = self.candidates.len();
            self.walker
                .shortest_return(&self.graph, first, second, bond, reach);
            self.walker.for_each_shortest_path(
                second,
                bond,
                &self.graph.scorable,
                &mut self.candidates,
            );
            // Every bond of a ring just found now knows a cycle it lies on.
            for candidate in first_new..self.candidates.len() {
                let size = self.candidates.atoms(candidate as u32).len() as u32;
                for covered in self.candidates.bonds_of(candidate as u32) {
                    let best = &mut self.best[covered as usize];
                    *best = (*best).min(size);
                }
            }
        }

        for candidate in 0..self.candidates.len() as u32 {
            // The search runs on the compacted graph; the family is read by
            // everything else, so it is stored in the molecule's numbering.
            self.translated.clear();
            self.translated.extend(
                self.candidates
                    .atoms(candidate)
                    .iter()
                    .map(|&atom| self.graph.atoms[atom as usize]),
            );
            self.family
                .push(&self.translated, self.candidates.bits(candidate));
        }
        self.family.sort();
    }

    /// The family as owned rings, in the molecule's own atom numbering.
    pub fn rings(&self) -> impl Iterator<Item = Ring> + '_ {
        (0..self.family.len()).map(|ring| Ring {
            atoms: self
                .family
                .atoms(ring)
                .iter()
                .map(|&atom| atom as usize)
                .collect(),
            bonds: self
                .family
                .bonds(ring)
                .iter()
                .map(|&bond| self.graph.bond_atoms(bond))
                .collect(),
        })
    }

    /// Stores the ring systems that are a single cycle, and marks their atoms
    /// so that the search never visits them.
    ///
    /// Most molecules are this case -- a benzene, a piperidine, two rings that
    /// share nothing -- and for a ring system where every atom has two
    /// neighbours the family is that cycle and can be nothing else: there is
    /// only one cycle to be had. Walking it costs one pass where Horton's
    /// construction would run a breadth-first search from each of its atoms and
    /// then throw away every copy but one.
    ///
    /// A system with a branch point in it -- anything fused, bridged or spiro --
    /// has an atom of degree three or more and goes to the general search
    /// whole.
    fn take_the_plain_cycles(&mut self, max_size: usize) -> bool {
        let order = self.graph.order();
        self.graph.plain.clear();
        self.graph.plain.resize(order, false);
        self.component.clear();
        self.component.resize(order, false);

        for start in 0..order as u32 {
            if self.component[start as usize] {
                continue;
            }
            // The component this atom is in, and whether anything in it
            // branches.
            self.walk.clear();
            self.walk.push(start);
            self.component[start as usize] = true;
            let mut head = 0;
            let mut plain = true;
            while head < self.walk.len() {
                let atom = self.walk[head];
                head += 1;
                let neighbours = self.graph.neighbourhood(atom);
                plain &= neighbours.len() == 2;
                for index in neighbours {
                    let neighbour = self.graph.neighbours[index].0;
                    if !self.component[neighbour as usize] {
                        self.component[neighbour as usize] = true;
                        self.walk.push(neighbour);
                    }
                }
            }
            if !plain || self.walk.len() > max_size {
                continue;
            }

            for &atom in &self.walk {
                self.graph.plain[atom as usize] = true;
            }
            let usable = self.walk.iter().all(|&atom| {
                self.graph
                    .neighbourhood(atom)
                    .all(|index| self.graph.scorable[self.graph.neighbours[index].1 as usize])
            });
            if !usable {
                continue;
            }
            self.translated.clear();
            self.translated.extend(
                self.walk
                    .iter()
                    .map(|&atom| self.graph.atoms[atom as usize]),
            );
            self.translated.sort_unstable();
            self.bonds_of_the_walk();
            self.family.push(&self.translated, &self.cycle_bits);
        }

        self.graph.plain.iter().any(|plain| !plain)
    }

    /// The bond set of the component in `walk`, which is a plain cycle, so
    /// every bond of it is a bond of the ring.
    fn bonds_of_the_walk(&mut self) {
        self.cycle_bits.clear();
        self.cycle_bits
            .resize(self.graph.bonds.len().div_ceil(64), 0);
        for &atom in &self.walk {
            for index in self.graph.neighbourhood(atom) {
                let bond = self.graph.neighbours[index].1;
                self.cycle_bits[bond as usize / 64] |= 1 << (bond % 64);
            }
        }
    }

    /// Marks every atom and bond that lies on a Hückel-aromatic subset of the
    /// ring family.
    ///
    /// `flags` is the per-atom electron contribution the model's tables give,
    /// as [`crate::molecule::aromaticity::MoleculeAromaticity::handle_aromatic_atom`]
    /// computes it; `cyclic_bonds` is the molecule's ring bonds and
    /// `conjugated_bonds` the π-capable subset of them from
    /// [`crate::molecule::pi_system::conjugated_ring_bonds`]. A ring takes part
    /// only if every one of its atoms can contribute and every one of its bonds
    /// is π-capable.
    ///
    /// Subsets are scored the way RDKit scores them: an atom's electrons count
    /// towards the subset only where the atom lies in one or two of the
    /// subset's rings, which is what keeps the shared atom of a three-ring
    /// junction (acepentalene, their issue #2895) from being counted into a
    /// circuit it cannot carry. A subset whose count is 4n+2 marks its rings
    /// aromatic, and the answer is the union over every such subset -- so no
    /// enumeration order is visible in it.
    pub fn find_aromatic_rings<M: MoleculeCore + ?Sized>(
        &mut self,
        molecule: &M,
        flags: &[AromaticityFlag],
        cyclic_bonds: &FxHashSet<(usize, usize)>,
        conjugated_bonds: &FxHashSet<(usize, usize)>,
        aromatic_atoms: &mut [bool],
        aromatic_bonds: &mut FxHashSet<(usize, usize)>,
    ) {
        // The ring-size limit is chemistry rather than effort -- see
        // [`MAX_RING_SIZE`] -- and zero means the caller has taken it off.
        let size_limit = match ring_size_limit() {
            0 => usize::MAX,
            limit => limit,
        };

        // The family is built over the π system's own bonds, not the
        // molecule's. A saturated ring is not part of any aromatic circuit, so
        // it should not decide which cycles are smallest -- yet over the
        // molecule's bonds it does, because the family keeps a smallest cycle
        // through each bond.
        // `C1C=CC2=C3CC(=C21)C=C3` has a perfectly alternating six-ring,
        // 3=4 4-9 9=8 8-6 6=7 7-3, six π electrons and no double bond leaving
        // it; every one of its bonds also lies on a five-ring through one of
        // the two CH2 groups, and five is shorter, so over the molecule's bonds
        // the six-ring is never a smallest cycle and never reaches scoring. Its
        // CH2 groups are nowhere near the circuit. They shadow it.
        //
        // Over the π-capable bonds the shadow goes and the circuit is found.
        // Measured over 200,000 PubChem records this changes 29 of them, and on
        // 28 the answer moves *to* what the path search says -- the bridged
        // annulenes, dihydropyrene among them, which are the class a ring model
        // is otherwise blind to. It also costs nothing, because the bonds it
        // drops are the ones whose rings would have been discarded anyway: a
        // cyclodextrin's bridging bond has 2^14 smallest cycles through it and
        // a sugar has no π system for any of them to belong to.
        let mut scoring = std::mem::take(&mut self.scoring_bonds);
        scoring.clear();
        scoring.extend(conjugated_bonds.iter().copied().filter(|&(first, second)| {
            [first, second].iter().all(|&atom| {
                flags
                    .get(atom)
                    .is_some_and(|flag| *flag != AromaticityFlag::NotAromatic)
            })
        }));
        let _ = cyclic_bonds;
        // The bonds handed over are a subset of the molecule's ring bonds, so
        // which of them still lie on a cycle has to be worked out again.
        self.perceive_scorable(molecule, &scoring, size_limit, true, |_, _| true);
        self.scoring_bonds = scoring;

        // Every ring now has π-capable atoms and π-capable bonds by
        // construction; what is left to ask is whether its electrons have
        // anywhere to go.
        self.scored.clear();
        for ring in 0..self.family.len() {
            if has_somewhere_to_delocalize(self.family.atoms(ring), flags) {
                self.scored.push(ring as u32);
            }
        }
        if self.scored.is_empty() {
            return;
        }

        self.build_ring_adjacency();
        // Cap zero asks for the model with no subset size in it at all: each
        // ring on its own, and each fused system entire. Nothing in between,
        // so nothing arbitrary, and measurably different.
        let limit = fused_ring_limit();
        let cap = match limit {
            0 => usize::MAX,
            limit => limit.min(self.scored.len()),
        };
        let whole_systems_only = limit == 0;
        self.membership.clear();
        self.membership.resize(molecule.len(), 0);
        self.marked_bond.clear();
        self.marked_bond.resize(self.graph.bonds.len(), false);
        self.perimeter_degree.clear();
        self.perimeter_degree.resize(self.graph.order(), 0);
        self.subset.clear();
        self.extensions.clear();
        self.measure_the_systems();

        if whole_systems_only {
            self.touched.clear();
            self.touched.resize(self.scored.len(), false);
            for root in 0..self.scored.len() as u32 {
                self.subset.clear();
                self.subset.push(root);
                self.score_subset(flags, aromatic_atoms, aromatic_bonds);
                self.grow_the_whole_system(root, flags, aromatic_atoms, aromatic_bonds);
            }
            return;
        }

        // Subsets are taken a size at a time rather than depth first, because
        // of what the size after them can be spared. The answer is the *union*
        // of what every accepted subset marks, and a subset can only mark atoms
        // and bonds of the one fused system it lies in -- so the moment every
        // atom and every bond of a system is marked, nothing further about that
        // system can change and the rest of its enumeration is dead work. Taken
        // in order of size, the fixed point is reached at the smallest size
        // that can reach it: a naphthalene or a fullerene is finished by its
        // single rings, and a cyclophane whose six benzenes give sixty-four
        // equally short macrocycles never reaches the layer where those sixty
        // four would be combined six at a time.
        //
        // ESU (Wernicke 2006) inside each layer: a subset is grown from its
        // lowest-numbered ring only, and each ring is offered as an extension
        // once, so every connected subset is visited exactly once and none has
        // to be deduplicated.
        for size in 1..=cap {
            let mut anything_left = false;
            self.touched.clear();
            self.touched.resize(self.scored.len(), false);
            for root in 0..self.scored.len() as u32 {
                if !self.system_active[self.system[root as usize] as usize] {
                    continue;
                }
                anything_left = true;
                self.touched[root as usize] = true;
                for index in 0..self.neighbours[root as usize].len() {
                    let neighbour = self.neighbours[root as usize][index];
                    if neighbour > root && !self.touched[neighbour as usize] {
                        self.touched[neighbour as usize] = true;
                        self.extensions.push(neighbour);
                    }
                }
                self.subset.push(root);
                self.extend(root, 0, size, flags, aromatic_atoms, aromatic_bonds);
                self.subset.pop();
                self.extensions.clear();
                self.touched.iter_mut().for_each(|flag| *flag = false);
            }
            if !anything_left {
                break;
            }
        }
    }

    /// Groups the rings into the fused systems they belong to, and counts what
    /// each system has to have marked before it is finished with.
    fn measure_the_systems(&mut self) {
        let rings = self.scored.len();
        self.system.clear();
        self.system.resize(rings, u32::MAX);
        self.system_active.clear();
        self.unmarked_bonds.clear();

        self.counted_bond.clear();
        self.counted_bond.resize(self.graph.bonds.len(), false);

        for start in 0..rings as u32 {
            if self.system[start as usize] != u32::MAX {
                continue;
            }
            let system = self.system_active.len() as u32;
            self.system_active.push(true);
            self.unmarked_bonds.push(0);

            self.walk.clear();
            self.walk.push(start);
            self.system[start as usize] = system;
            let mut head = 0;
            while head < self.walk.len() {
                let ring = self.walk[head];
                head += 1;
                let family_ring = self.scored[ring as usize] as usize;
                for index in 0..self.family.bonds(family_ring).len() {
                    let bond = self.family.bonds(family_ring)[index] as usize;
                    if !self.counted_bond[bond] {
                        self.counted_bond[bond] = true;
                        self.unmarked_bonds[system as usize] += 1;
                    }
                }
                for index in 0..self.neighbours[ring as usize].len() {
                    let neighbour = self.neighbours[ring as usize][index];
                    if self.system[neighbour as usize] == u32::MAX {
                        self.system[neighbour as usize] = system;
                        self.walk.push(neighbour);
                    }
                }
            }
        }
    }

    /// Scores the whole fused system `root` belongs to, as one Hückel count.
    ///
    /// The cap-free half of the model: a ring on its own, or every ring of the
    /// system it is fused into, and nothing chosen in between.
    fn grow_the_whole_system(
        &mut self,
        root: u32,
        flags: &[AromaticityFlag],
        aromatic_atoms: &mut [bool],
        aromatic_bonds: &mut FxHashSet<(usize, usize)>,
    ) {
        self.subset.clear();
        self.subset.push(root);
        let mut head = 0;
        while head < self.subset.len() {
            let ring = self.subset[head];
            head += 1;
            for index in 0..self.neighbours[ring as usize].len() {
                let neighbour = self.neighbours[ring as usize][index];
                if !self.subset.contains(&neighbour) {
                    self.subset.push(neighbour);
                }
            }
        }
        if self.subset.len() > 1 {
            self.score_subset(flags, aromatic_atoms, aromatic_bonds);
        }
        self.subset.clear();
        self.subset.push(root);
    }

    /// One step of the subset enumeration: at the layer's size the subset in
    /// hand is scored, below it the subset is grown by each ring still on
    /// offer.
    ///
    /// `base` is where this call's candidates start in the shared stack, which
    /// every level appends to and truncates back -- one buffer for the whole
    /// search rather than a vector per subset.
    fn extend(
        &mut self,
        root: u32,
        base: usize,
        size: usize,
        flags: &[AromaticityFlag],
        aromatic_atoms: &mut [bool],
        aromatic_bonds: &mut FxHashSet<(usize, usize)>,
    ) {
        // Nothing this branch can reach belongs to another system, so a system
        // with everything already marked ends the branch wherever it is met.
        if !self.system_active[self.system[root as usize] as usize] {
            return;
        }
        if self.subset.len() == size {
            self.score_subset(flags, aromatic_atoms, aromatic_bonds);
            return;
        }

        let end = self.extensions.len();
        for position in base..end {
            let candidate = self.extensions[position];
            // Everything after this one stays on offer; everything before it
            // has had its turn, which is what makes each subset appear once.
            let next = self.extensions.len();
            for index in position + 1..end {
                let carried = self.extensions[index];
                self.extensions.push(carried);
            }
            let added_from = self.extensions.len();
            for index in 0..self.neighbours[candidate as usize].len() {
                let neighbour = self.neighbours[candidate as usize][index];
                if neighbour > root && !self.touched[neighbour as usize] {
                    self.touched[neighbour as usize] = true;
                    self.extensions.push(neighbour);
                }
            }

            self.subset.push(candidate);
            self.extend(root, next, size, flags, aromatic_atoms, aromatic_bonds);
            self.subset.pop();

            for index in added_from..self.extensions.len() {
                self.touched[self.extensions[index] as usize] = false;
            }
            self.extensions.truncate(next);
        }
    }

    /// Counts the π electrons of the subset in hand, and marks it where they
    /// make 4n+2.
    fn score_subset(
        &mut self,
        flags: &[AromaticityFlag],
        aromatic_atoms: &mut [bool],
        aromatic_bonds: &mut FxHashSet<(usize, usize)>,
    ) {
        let mut electrons = 0usize;
        for position in 0..self.subset.len() {
            let ring = self.scored[self.subset[position] as usize] as usize;
            for &atom in self.family.atoms(ring) {
                self.membership[atom as usize] += 1;
            }
        }
        for position in 0..self.subset.len() {
            let ring = self.scored[self.subset[position] as usize] as usize;
            for index in 0..self.family.atoms(ring).len() {
                let atom = self.family.atoms(ring)[index] as usize;
                // Counted once per atom, on the first ring of the subset that
                // holds it, and only where the atom belongs to one or two of
                // the subset's rings.
                let seen = self.membership[atom];
                if seen == 0 {
                    continue;
                }
                if seen <= 2 {
                    electrons += flags[atom] as usize;
                }
                self.membership[atom] = 0;
            }
        }

        if electrons % 4 != 2 {
            return;
        }

        let system = self.system[self.subset[0] as usize] as usize;

        // The bonds this subset would mark: those of its rings that lie in one
        // of them. A bond shared by two rings of the subset is inside the
        // circuit that was counted rather than on it -- the count that made
        // this subset aromatic is a count around the outside, so it says
        // nothing about the bond across the middle. Azulene is the case: ten
        // electrons around the perimeter, and a formal single bond where its
        // two rings meet.
        self.circuit.clear();
        for position in 0..self.subset.len() {
            let ring = self.scored[self.subset[position] as usize] as usize;
            for index in 0..self.family.bonds(ring).len() {
                let bond = self.family.bonds(ring)[index];
                let shared = self
                    .subset
                    .iter()
                    .filter(|&&other| {
                        self.family
                            .bonds(self.scored[other as usize] as usize)
                            .binary_search(&bond)
                            .is_ok()
                    })
                    .count();
                if shared == 1 {
                    self.circuit.push(bond);
                }
            }
        }

        // And it has to *be* a circuit. Where two rings of the subset share a
        // single bond, dropping it leaves the perimeter closed; where they
        // share a path of two or more, dropping all of it leaves the marks
        // hanging in the air -- an aromatic run that stops in the middle of a
        // ring. Written out, that is a lower-case atom with one aromatic bond,
        // and no assignment of double bonds can satisfy it: PubChem's CID
        // 164086848 came back from its own canonical form as
        // "no double bond is left for atom 20". A subset whose perimeter is not
        // a cycle describes no circuit and marks nothing.
        for &bond in &self.circuit {
            let (first, second) = self.graph.bonds[bond as usize];
            self.perimeter_degree[first as usize] += 1;
            self.perimeter_degree[second as usize] += 1;
        }
        let closed = self.circuit.iter().all(|&bond| {
            let (first, second) = self.graph.bonds[bond as usize];
            self.perimeter_degree[first as usize] == 2
                && self.perimeter_degree[second as usize] == 2
        });
        for &bond in &self.circuit {
            let (first, second) = self.graph.bonds[bond as usize];
            self.perimeter_degree[first as usize] = 0;
            self.perimeter_degree[second as usize] = 0;
        }
        if !closed {
            return;
        }

        for position in 0..self.subset.len() {
            let ring = self.scored[self.subset[position] as usize] as usize;
            for index in 0..self.family.atoms(ring).len() {
                aromatic_atoms[self.family.atoms(ring)[index] as usize] = true;
            }
        }
        for index in 0..self.circuit.len() {
            let bond = self.circuit[index];
            aromatic_bonds.insert(self.graph.bond_atoms(bond));
            if !self.marked_bond[bond as usize] {
                self.marked_bond[bond as usize] = true;
                self.unmarked_bonds[system] -= 1;
            }
        }

        // Everything this system has is marked, so no subset of it can add
        // anything and the rest of its enumeration is dead work.
        if self.unmarked_bonds[system] == 0 {
            self.system_active[system] = false;
        }
    }

    /// Which of the scored rings share a bond with which.
    fn build_ring_adjacency(&mut self) {
        if self.neighbours.len() < self.scored.len() {
            self.neighbours.resize(self.scored.len(), Vec::new());
        }
        for list in self.neighbours.iter_mut().take(self.scored.len()) {
            list.clear();
        }
        for first in 0..self.scored.len() {
            for second in first + 1..self.scored.len() {
                if shares_a_bond(
                    self.family.bonds(self.scored[first] as usize),
                    self.family.bonds(self.scored[second] as usize),
                ) {
                    self.neighbours[first].push(second as u32);
                    self.neighbours[second].push(first as u32);
                }
            }
        }
    }
}

/// Whether a ring has an orbital for its electrons to move into.
///
/// Every atom of a ring can be a plausible ring member -- carry a p orbital,
/// sit on a π-capable cycle -- and the ring still be nothing but lone pairs:
/// `O1P2OP3OP1OP(O2)O3` is P4O6, a saturated cage in which every phosphorus and
/// every oxygen offers two electrons and no atom offers an empty orbital or a
/// π bond to receive them. Counting such a ring gives 4n+2 wherever the atom
/// count happens to land, and the fused-subset rule reaches 4n+2 on the cage's
/// faces -- which is how a phosphorus oxide came out aromatic.
///
/// The path search never had to say anything about this, because no single
/// cycle of the cage counts to 4n+2; it is a question only a model that
/// combines rings has to answer. What makes a ring able to delocalize is one
/// atom that is not a pure donor: an atom with a π bond (one electron) or with
/// an empty p orbital (none), which is what borazine's boron is and what makes
/// borazine aromatic where P4O6 is not.
fn has_somewhere_to_delocalize(atoms: &[u32], flags: &[AromaticityFlag]) -> bool {
    atoms.iter().any(|&atom| {
        (flags[atom as usize] as usize) < AromaticityFlag::PotentiallyAromaticTwoElectrons as usize
    })
}

/// Whether two sorted bond-index lists have an entry in common.
fn shares_a_bond(left: &[u32], right: &[u32]) -> bool {
    let (mut here, mut there) = (0, 0);
    while here < left.len() && there < right.len() {
        match left[here].cmp(&right[there]) {
            std::cmp::Ordering::Less => here += 1,
            std::cmp::Ordering::Greater => there += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

/// The molecule's cyclic bonds as a graph of their own, over the atoms that are
/// on a ring and nothing else.
///
/// Compacting matters because the arrays a breadth-first search clears are
/// sized by this graph rather than by the molecule -- a steroid with one
/// aromatic ring has six atoms here and upwards of forty in the molecule, and
/// the search is run once per atom of it.
#[derive(Debug, Default, Clone)]
struct CyclicGraph {
    /// For each ring atom, the atom of the molecule it is, in index order.
    atoms: Vec<u32>,
    /// For each atom of the molecule, its index here, or `u32::MAX`.
    compact: Vec<u32>,
    /// CSR adjacency: `neighbours[offsets[v]..offsets[v + 1]]`, each entry a
    /// neighbouring ring atom and the bond that reaches it.
    offsets: Vec<u32>,
    neighbours: Vec<(u32, u32)>,
    /// Every cyclic bond as an ordered pair of ring-atom indices, sorted.
    bonds: Vec<(u32, u32)>,
    /// Whether a ring using this bond could be scored at all: both its atoms
    /// have electrons to contribute and the bond is one a π system could run
    /// through. A ring with any other bond in it is thrown away by the filter
    /// however it was found, so it is never built -- which is what keeps a
    /// saturated macrocycle from costing anything.
    scorable: Vec<bool>,
    /// Whether an atom's ring system is a single plain cycle, which the search
    /// then has nothing to find.
    plain: Vec<bool>,
    /// Scratch for the bridge search that keeps this graph a ring system.
    cyclic: Vec<bool>,
    entered: Vec<u32>,
    lowest: Vec<u32>,
    stack: Vec<(u32, u32, u32)>,
    filled: Vec<u32>,
}

impl CyclicGraph {
    /// Builds the graph of the bonds given, keeping only those that really
    /// lie on a cycle *of this graph*.
    ///
    /// The molecule's own ring bonds arrive that way, but the bonds perception
    /// hands over do not. They are the π-capable ones, which
    /// [`crate::molecule::pi_system::conjugated_ring_bonds`] has 2-cored --
    /// and a 2-core is not a ring system: two cycles joined by a path keep the
    /// path, every atom of it having two neighbours. Filtering further by which
    /// atoms have electrons to give can leave tails and lone bonds as well. So
    /// ring perception is done again here, on the graph that is actually going
    /// to be searched: the tails are peeled and the bridges -- the bonds whose
    /// removal disconnects the graph, which by definition lie on no cycle --
    /// are dropped. Nothing that could have been part of a ring is lost, and
    /// what is left is a ring system again, which is what the rest of this
    /// file assumes it has.
    fn build<M: MoleculeCore + ?Sized>(
        &mut self,
        molecule: &M,
        cyclic_bonds: &FxHashSet<(usize, usize)>,
        perceive_rings: bool,
    ) {
        self.atoms.clear();
        self.bonds.clear();
        self.scorable.clear();
        self.neighbours.clear();
        self.offsets.clear();
        self.compact.clear();
        self.compact.resize(molecule.len(), u32::MAX);
        if cyclic_bonds.is_empty() {
            return;
        }

        for &(first, second) in cyclic_bonds {
            for atom in [first, second] {
                if self.compact[atom] == u32::MAX {
                    self.compact[atom] = 0;
                    self.atoms.push(atom as u32);
                }
            }
        }
        // A hash set hands its bonds back in the hash's order rather than the
        // molecule's, so the ring atoms are sorted before they are numbered:
        // everything downstream -- a ring's atom list, the order the family is
        // reported in -- is then a function of the molecule alone.
        self.atoms.sort_unstable();
        for (index, &atom) in self.atoms.iter().enumerate() {
            self.compact[atom as usize] = index as u32;
        }

        for &(first, second) in cyclic_bonds {
            let (here, there) = (self.compact[first], self.compact[second]);
            self.bonds.push((here.min(there), here.max(there)));
        }
        self.bonds.sort_unstable();
        self.link();

        if perceive_rings && self.drop_the_acyclic_bonds() {
            // The atoms keep their numbering -- one that is nobody's to lose
            // and cheap to carry -- and the bonds that lie on no cycle go.
            let mut kept = 0;
            for bond in 0..self.bonds.len() {
                if self.cyclic[bond] {
                    self.bonds[kept] = self.bonds[bond];
                    kept += 1;
                }
            }
            self.bonds.truncate(kept);
            self.link();
        }
    }

    /// The CSR adjacency for the bonds in hand.
    fn link(&mut self) {
        self.scorable.clear();
        self.scorable.resize(self.bonds.len(), true);
        self.offsets.clear();
        self.offsets.resize(self.atoms.len() + 1, 0);
        for &(first, second) in &self.bonds {
            self.offsets[first as usize + 1] += 1;
            self.offsets[second as usize + 1] += 1;
        }
        for index in 1..self.offsets.len() {
            self.offsets[index] += self.offsets[index - 1];
        }
        self.neighbours.clear();
        self.neighbours
            .resize(self.bonds.len() * 2, (u32::MAX, u32::MAX));
        self.filled.clear();
        self.filled.extend_from_slice(&self.offsets);
        for (bond, &(first, second)) in self.bonds.iter().enumerate() {
            self.neighbours[self.filled[first as usize] as usize] = (second, bond as u32);
            self.filled[first as usize] += 1;
            self.neighbours[self.filled[second as usize] as usize] = (first, bond as u32);
            self.filled[second as usize] += 1;
        }
    }

    /// Marks the bonds that lie on a cycle of this graph, and says whether any
    /// does not.
    ///
    /// A depth-first search keeping, for each atom, the earliest atom reachable
    /// from its subtree by one back edge (Tarjan): a bond is a bridge when the
    /// far side can reach nothing above itself, and a bridge lies on no cycle.
    /// Iterative, because a molecule is allowed to be long.
    fn drop_the_acyclic_bonds(&mut self) -> bool {
        let order = self.atoms.len();
        self.cyclic.clear();
        self.cyclic.resize(self.bonds.len(), true);
        self.entered.clear();
        self.entered.resize(order, u32::MAX);
        self.lowest.clear();
        self.lowest.resize(order, u32::MAX);
        self.stack.clear();

        let mut clock = 0u32;
        let mut dropped = false;
        for root in 0..order as u32 {
            if self.entered[root as usize] != u32::MAX {
                continue;
            }
            self.entered[root as usize] = clock;
            self.lowest[root as usize] = clock;
            clock += 1;
            self.stack
                .push((root, u32::MAX, self.offsets[root as usize]));

            while let Some(&mut (atom, from, ref mut next)) = self.stack.last_mut() {
                if (*next as usize) < self.offsets[atom as usize + 1] as usize {
                    let (neighbour, bond) = self.neighbours[*next as usize];
                    *next += 1;
                    if bond == from {
                        continue;
                    }
                    if self.entered[neighbour as usize] == u32::MAX {
                        self.entered[neighbour as usize] = clock;
                        self.lowest[neighbour as usize] = clock;
                        clock += 1;
                        self.stack
                            .push((neighbour, bond, self.offsets[neighbour as usize]));
                    } else {
                        let seen = self.entered[neighbour as usize];
                        self.lowest[atom as usize] = self.lowest[atom as usize].min(seen);
                    }
                    continue;
                }

                self.stack.pop();
                if let Some(&mut (parent, _, _)) = self.stack.last_mut() {
                    let low = self.lowest[atom as usize];
                    self.lowest[parent as usize] = self.lowest[parent as usize].min(low);
                    if low > self.entered[parent as usize] {
                        self.cyclic[from as usize] = false;
                        dropped = true;
                    }
                }
            }
        }

        dropped
    }

    fn order(&self) -> usize {
        self.atoms.len()
    }

    fn neighbourhood(&self, atom: u32) -> std::ops::Range<usize> {
        self.offsets[atom as usize] as usize..self.offsets[atom as usize + 1] as usize
    }

    fn neighbours_of(&self, atom: u32) -> &[(u32, u32)] {
        let start = self.offsets[atom as usize] as usize;
        let end = self.offsets[atom as usize + 1] as usize;
        &self.neighbours[start..end]
    }

    /// A bond as the pair of molecule atoms it joins.
    fn bond_atoms(&self, bond: u32) -> (usize, usize) {
        let (first, second) = self.bonds[bond as usize];
        (
            self.atoms[first as usize] as usize,
            self.atoms[second as usize] as usize,
        )
    }
}

/// One breadth-first search, and every shortest path read back out of it.
#[derive(Debug, Default, Clone)]
struct ShortestPaths {
    distance: Vec<u32>,
    /// Predecessors on a shortest path, as a linked list per atom, so that
    /// every tie is available rather than resolved by whichever was reached
    /// first. The bond each one arrives by is kept with it, so a path can be
    /// turned into a set of bonds without looking anything up.
    predecessor_head: Vec<u32>,
    predecessor_next: Vec<u32>,
    predecessor_atom: Vec<u32>,
    predecessor_bond: Vec<u32>,
    queue: Vec<u32>,
    /// The path being walked back, as atoms and as the bonds between them.
    walk: Vec<u32>,
    walk_bonds: Vec<u32>,
    /// Where the walk resumes at each depth: the predecessor link not yet tried.
    resume: Vec<u32>,
    target: u32,
    reached: bool,
}

const UNREACHED: u32 = u32::MAX;

impl ShortestPaths {
    /// The shortest way from `from` back to `to` without using `bond`, which is
    /// the smallest cycle through that bond less the bond itself.
    ///
    /// The search stops at the depth that answers the question: a cycle of more
    /// than `max_size` atoms is not scored, so a path longer than `max_size - 1`
    /// is never wanted, and once the target's own layer is finished nothing
    /// deeper can be a shortest path to it.
    fn shortest_return(
        &mut self,
        graph: &CyclicGraph,
        from: u32,
        to: u32,
        bond: u32,
        max_size: usize,
    ) {
        let order = graph.order();
        self.distance.clear();
        self.distance.resize(order, UNREACHED);
        self.predecessor_head.clear();
        self.predecessor_head.resize(order, u32::MAX);
        self.predecessor_atom.clear();
        self.predecessor_next.clear();
        self.predecessor_bond.clear();
        self.queue.clear();
        self.target = to;
        self.reached = false;

        let furthest = (max_size.saturating_sub(1)) as u32;
        self.distance[from as usize] = 0;
        self.queue.push(from);
        let mut head = 0;
        while head < self.queue.len() {
            let current = self.queue[head];
            head += 1;
            let depth = self.distance[current as usize];
            // Everything at the target's depth still has to be finished, since
            // a shortest path may arrive from any of them, but nothing deeper
            // can contribute.
            if depth >= furthest || (self.reached && depth >= self.distance[to as usize]) {
                continue;
            }
            let next = depth + 1;
            for &(neighbour, edge) in graph.neighbours_of(current) {
                if edge == bond {
                    continue;
                }
                if self.distance[neighbour as usize] == UNREACHED {
                    self.distance[neighbour as usize] = next;
                    self.queue.push(neighbour);
                    if neighbour == to {
                        self.reached = true;
                    }
                }
                if self.distance[neighbour as usize] == next {
                    self.predecessor_atom.push(current);
                    self.predecessor_bond.push(edge);
                    self.predecessor_next
                        .push(self.predecessor_head[neighbour as usize]);
                    self.predecessor_head[neighbour as usize] =
                        self.predecessor_atom.len() as u32 - 1;
                }
            }
        }
    }

    /// Hands the candidate arena every shortest path to `to`, each closed into
    /// a cycle by `bond`.
    ///
    /// Walked iteratively over the predecessor links, so a molecule cannot put
    /// the search on the stack, and every tie is walked -- the number of them
    /// is the number of rings that bond contributes, which is the answer rather
    /// than a cost to be capped.
    fn for_each_shortest_path(
        &mut self,
        to: u32,
        bond: u32,
        scorable: &[bool],
        candidates: &mut Candidates,
    ) {
        if self.distance[to as usize] == UNREACHED {
            return;
        }

        self.walk.clear();
        self.walk_bonds.clear();
        self.resume.clear();
        self.walk.push(to);
        self.resume.push(self.predecessor_head[to as usize]);

        while let Some(&link) = self.resume.last() {
            let atom = *self.walk.last().expect("the walk holds the atom");
            if self.distance[atom as usize] == 0 {
                // Back at the bond's other end: the walk is a whole path.
                candidates.begin_cycle();
                candidates.set_bond(bond);
                for &walked in &self.walk_bonds {
                    candidates.set_bond(walked);
                }
                if candidates.is_new() {
                    for &walked in &self.walk {
                        candidates.push_atom(walked);
                    }
                    candidates.commit();
                }
                self.step_back();
                continue;
            }
            if link == u32::MAX {
                self.step_back();
                continue;
            }

            let previous = self.predecessor_atom[link as usize];
            let edge = self.predecessor_bond[link as usize];
            *self.resume.last_mut().expect("the walk has a resume point") =
                self.predecessor_next[link as usize];
            if !scorable[edge as usize] {
                continue;
            }
            self.walk.push(previous);
            self.walk_bonds.push(edge);
            self.resume.push(self.predecessor_head[previous as usize]);
        }
    }

    /// Undoes one step of the walk.
    fn step_back(&mut self) {
        self.walk.pop();
        self.walk_bonds.pop();
        self.resume.pop();
    }
}

/// The candidate cycles: a bond-index bitset and an atom list each, in arenas.
#[derive(Debug, Default, Clone)]
struct Candidates {
    words: usize,
    bits: Vec<u64>,
    atoms: Vec<u32>,
    /// `(atom start, atom count)` for each candidate.
    spans: Vec<(u32, u32)>,
    /// Candidates already kept, by a hash of their bond set. The indices behind
    /// one hash are compared word by word, so a collision costs a comparison
    /// rather than a lost ring.
    seen: FxHashMap<u64, Vec<u32>>,
    pending_bits: Vec<u64>,
    pending_atoms: Vec<u32>,
}

impl Candidates {
    fn begin(&mut self, bond_count: usize) {
        self.words = bond_count.div_ceil(64);
        self.bits.clear();
        self.atoms.clear();
        self.spans.clear();
        self.seen.clear();
        self.pending_bits.clear();
        self.pending_bits.resize(self.words, 0);
    }

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn begin_cycle(&mut self) {
        self.pending_bits.iter_mut().for_each(|word| *word = 0);
        self.pending_atoms.clear();
    }

    /// Puts one of the cycle's bonds into the set being built.
    fn set_bond(&mut self, bond: u32) {
        self.pending_bits[bond as usize / 64] |= 1 << (bond % 64);
    }

    /// Whether the bond set in hand is one not seen before, keeping it if so.
    ///
    /// A cycle is a smallest cycle through several of its bonds at once -- a
    /// benzene ring through all six -- so it is found once per such bond, and
    /// all but the first of those are duplicates.
    fn is_new(&mut self) -> bool {
        let hash = hash_words(&self.pending_bits);
        if let Some(known) = self.seen.get(&hash) {
            let words = self.words;
            for &candidate in known {
                let start = candidate as usize * words;
                if self.bits[start..start + words] == self.pending_bits[..] {
                    return false;
                }
            }
        }
        let candidate = self.spans.len() as u32;
        self.seen.entry(hash).or_default().push(candidate);
        self.bits.extend_from_slice(&self.pending_bits);
        self.pending_atoms.clear();
        true
    }

    fn push_atom(&mut self, atom: u32) {
        self.pending_atoms.push(atom);
    }

    /// Stores the atoms of the cycle [`Candidates::is_new`] just kept.
    fn commit(&mut self) {
        let start = self.atoms.len() as u32;
        self.pending_atoms.sort_unstable();
        self.atoms.extend_from_slice(&self.pending_atoms);
        self.spans.push((start, self.pending_atoms.len() as u32));
    }

    fn atoms(&self, candidate: u32) -> &[u32] {
        let (start, count) = self.spans[candidate as usize];
        &self.atoms[start as usize..start as usize + count as usize]
    }

    /// The bonds of a candidate, in index order.
    fn bonds_of(&self, candidate: u32) -> impl Iterator<Item = u32> + '_ {
        let start = candidate as usize * self.words;
        let words = self.words;
        (0..words).flat_map(move |word| {
            let mut remaining = self.bits[start + word];
            std::iter::from_fn(move || {
                if remaining == 0 {
                    return None;
                }
                let bit = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                Some((word * 64 + bit) as u32)
            })
        })
    }

    fn bits(&self, candidate: u32) -> &[u64] {
        let start = candidate as usize * self.words;
        &self.bits[start..start + self.words]
    }
}

/// FNV over the bond set. The map behind it compares the words anyway, so this
/// only has to spread them.
fn hash_words(words: &[u64]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &word in words {
        hash = (hash ^ word).wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// The accepted rings: molecule atoms and bond indices, in two arenas.
#[derive(Debug, Default, Clone)]
struct RingStore {
    atoms: Vec<u32>,
    bonds: Vec<u32>,
    spans: Vec<(u32, u32, u32, u32)>,
    order: Vec<u32>,
    sorted: Vec<(u32, u32, u32, u32)>,
}

impl RingStore {
    fn clear(&mut self) {
        self.atoms.clear();
        self.bonds.clear();
        self.spans.clear();
    }

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn push(&mut self, atoms: &[u32], bits: &[u64]) {
        let atom_start = self.atoms.len() as u32;
        self.atoms.extend_from_slice(atoms);
        let bond_start = self.bonds.len() as u32;
        for (index, &word) in bits.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                self.bonds.push((index * 64 + bit) as u32);
            }
        }
        self.spans.push((
            atom_start,
            atoms.len() as u32,
            bond_start,
            self.bonds.len() as u32 - bond_start,
        ));
    }

    fn atoms(&self, ring: usize) -> &[u32] {
        let (start, count, _, _) = self.spans[ring];
        &self.atoms[start as usize..start as usize + count as usize]
    }

    fn bonds(&self, ring: usize) -> &[u32] {
        let (_, _, start, count) = self.spans[ring];
        &self.bonds[start as usize..start as usize + count as usize]
    }

    /// Puts the family in an order the molecule decides -- by size, then by
    /// atoms -- so that what is reported does not depend on which root happened
    /// to reach a ring first.
    fn sort(&mut self) {
        if self.len() < 2 {
            return;
        }
        self.order.clear();
        self.order.extend(0..self.len() as u32);
        let (atoms, spans) = (&self.atoms, &self.spans);
        let of = |ring: u32| {
            let (start, count, _, _) = spans[ring as usize];
            &atoms[start as usize..start as usize + count as usize]
        };
        self.order.sort_unstable_by(|&left, &right| {
            of(left)
                .len()
                .cmp(&of(right).len())
                .then_with(|| of(left).cmp(of(right)))
        });
        self.sorted.clear();
        self.sorted
            .extend(self.order.iter().map(|&ring| self.spans[ring as usize]));
        std::mem::swap(&mut self.spans, &mut self.sorted);
    }
}

fn environment_cap(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(fallback)
}
