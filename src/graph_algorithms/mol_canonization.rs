use crate::io::SMILESParseError;
use crate::molecule::aromaticity::{Aromatizer, default_aromaticity_model};
use crate::molecule::chirality::*;
use crate::molecule::graph::AtomInvariant;
use crate::prelude::*;
use std::cell::RefCell;

thread_local! {
    /// Automorphism generators collected from the most recent nauty run, each a
    /// permutation over the nauty vertices (heavy atoms + bond vertices).
    static AUTOMORPHISM_GENERATORS: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
}

/// nauty `userautomproc` callback: records each automorphism generator.
fn collect_automorphism_generator(
    _count: u32,
    permutation: &[u32],
    _orbits: &[u32],
    _number_of_orbits: u32,
    _stabilized_vertex: u32,
    _number_of_vertices: u32,
) {
    AUTOMORPHISM_GENERATORS.with(|generators| {
        generators.borrow_mut().push(permutation.to_vec());
    });
}
// use canonizer::{
//     canonization::{first_non_trivial, graph_canon, smallest_first_non_trivial},
//     edge_labeled_canonization::{
//         create_canonical_adjacency_matrix_edge_labeled, graph_canon_labeled,
//     },
// };
//use nauty_pet::autom::AutomGroup;
use rustc_hash::{FxHashMap, FxHashSet};

use canonaut::structs::CanonautManager;

pub type BitMatrix = Vec<usize>;

/// The working buffers one canonization needs, beside the aromaticity ones.
///
/// Scratch, not state: every field is cleared and resized from the molecule
/// before use.
pub struct CanonizationScratch {
    sorted_labels: Vec<AtomInvariant>,
    heavy_labeling: Vec<usize>,
    colors: Vec<u32>,
    /// canonaut's own buffers -- labelling, orbits, partition, workspace and
    /// search tree -- which it grows to each graph and reuses across calls.
    manager: CanonautManager,
}

impl Default for CanonizationScratch {
    fn default() -> Self {
        Self {
            sorted_labels: Vec::new(),
            heavy_labeling: Vec::new(),
            colors: Vec::new(),
            // Sized from the first molecule: `canonize_with_colors` grows every
            // buffer to the graph it is handed.
            manager: CanonautManager::new(0)
                .with_canonization()
                .with_automorphism_callback(collect_automorphism_generator),
        }
    }
}

impl Clone for CanonizationScratch {
    /// Nothing here is state, so a clone gets a fresh manager rather than a
    /// copy of buffers the next canonization resizes anyway.
    fn clone(&self) -> Self {
        Self {
            sorted_labels: self.sorted_labels.clone(),
            heavy_labeling: self.heavy_labeling.clone(),
            colors: self.colors.clone(),
            ..Self::default()
        }
    }
}

impl std::fmt::Debug for CanonizationScratch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CanonizationScratch")
            .field("sorted_labels", &self.sorted_labels)
            .field("heavy_labeling", &self.heavy_labeling)
            .field("colors", &self.colors)
            .finish_non_exhaustive()
    }
}

/// Canonizes molecules, keeping its working buffers between them.
///
/// [`canonize_mol`] and [`canonicalize_and_sort_smiles`] allocate everything
/// they need per call -- the colouring, nauty's labelling, orbit and workspace
/// arrays, and a full aromaticity perception on top. Canonizing a stream of
/// molecules through a `Canonizer` pays for those once.
///
/// ```
/// use oxidane::graph_algorithms::mol_canonization::Canonizer;
///
/// let mut canonizer = Canonizer::new();
/// let first = canonizer.canonical_smiles("CC(=O)Oc1ccccc1C(=O)O", false).unwrap();
/// let second = canonizer.canonical_smiles("O=C(C)Oc1ccccc1C(O)=O", false).unwrap();
/// assert_eq!(first, second);
/// ```
///
/// The answer is exactly what the free functions give; only the allocation
/// traffic differs. It holds no state between molecules, but it is not `Sync`
/// in spirit either -- give each thread its own, e.g. with rayon's
/// `map_init(Canonizer::new, ..)`.
#[derive(Debug, Default, Clone)]
pub struct Canonizer {
    aromatizer: Aromatizer,
    scratch: CanonizationScratch,
    automorphisms: Vec<Vec<usize>>,
    graph_automorphisms: Vec<Vec<usize>>,
}

impl Canonizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// [`canonize_mol`], reusing this canonizer's buffers.
    pub fn canonize<T: Molecule + Clone + ToSMILES>(&mut self, mol: &T) -> T {
        let mut molecule = mol.clone();
        let _ = self
            .aromatizer
            .aromatize(&mut molecule, default_aromaticity_model(), None, false);
        self.canonize_perceived(molecule)
    }

    /// Perceive a candidate for aromatic scoring. Only ambiguous perception
    /// needs canonical ordering before scoring; ordinary rejected candidates
    /// need never pay for graph labeling or stereo canonicalization.
    pub(crate) fn perceive_for_scoring<T: Molecule + Clone + ToSMILES>(
        &mut self,
        mut molecule: T,
    ) -> (T, bool) {
        let _ = self
            .aromatizer
            .aromatize(&mut molecule, default_aromaticity_model(), None, false);
        if perception_had_a_choice_to_make(&molecule) {
            (self.canonize_perceived(molecule), true)
        } else {
            (molecule, false)
        }
    }

    /// [`Canonizer::canonize`] for a molecule whose aromaticity has already
    /// been perceived under the same model.
    ///
    /// Perception is over a third of what canonizing a parsed molecule costs,
    /// and parsing has already done it: `MolecularGraph::from_smiles` and
    /// `from_sdf` both perceive by default. Canonizing what they return with
    /// [`Canonizer::canonize`] pays for it a second time.
    ///
    /// Only call this on a molecule that really has been perceived. On one that
    /// has not, the Kekule and the aromatic way of writing the same structure
    /// canonicalize to different strings -- which is a silent wrong answer, not
    /// an error. When in doubt use [`Canonizer::canonize`].
    pub fn canonize_perceived<T: Molecule + Clone + ToSMILES>(&mut self, molecule: T) -> T {
        // Perception reads the atoms in the order they arrive in, and the
        // canonical order is not that order: a graphene flake's carbonyl ring
        // comes out Kekule from the molecule as written and aromatic from the
        // same molecule renumbered, so the canonical form of the canonical form
        // was not the canonical form. Perceiving again on the order this pass
        // settled on, and canonizing again where that changed anything, is a
        // fixed point of both -- and it costs one perception on a molecule
        // whose perception was already stable, which is nearly all of them.
        let answer = self.canonize_in_the_order_given(molecule);
        if !perception_had_a_choice_to_make(&answer) {
            return answer;
        }
        let mut again = answer.clone();
        let _ = self
            .aromatizer
            .aromatize(&mut again, default_aromaticity_model(), None, false);
        // Atom sets miss a changed ring bond between already-aromatic atoms.
        // Compare aromatic membership per edge in this unchanged numbering.
        if again
            .atom_bonds()
            .iter()
            .zip(answer.atom_bonds())
            .all(|(left, right)| {
                left.iter().zip(right).all(|(a, b)| {
                    (a.bond_order() == BondOrder::Aromatic)
                        == (b.bond_order() == BondOrder::Aromatic)
                })
            })
        {
            return answer;
        }
        // Perception disagrees with itself about this molecule, so the answer
        // would depend on the order the atoms arrived in. Settle it on an order
        // that perception cannot have influenced: the canonical order of the
        // molecule with its bond orders taken away, which every way of writing
        // it shares.
        let order = self.order_of_the_sigma_skeleton(&again);
        let mut settled = again;
        reorder_molecule(&mut settled, &order);
        let _ = self
            .aromatizer
            .aromatize(&mut settled, default_aromaticity_model(), None, false);
        self.canonize_in_the_order_given(settled)
    }

    /// The canonical order of the molecule with every bond order the same.
    ///
    /// Aromatic perception reads the atoms in the order they are stored in, and
    /// a molecule written aromatic and the same molecule written Kekule are
    /// stored differently -- so a fused polycyclic where perception has a
    /// choice to make can make it two ways, and the canonical form of the
    /// canonical form is not the canonical form. Ordering by the sigma skeleton
    /// breaks the circle: bond orders are what the two writings disagree about,
    /// so an order that ignores them is one they agree on.
    fn order_of_the_sigma_skeleton<T: Molecule + Clone + ToSMILES>(
        &mut self,
        molecule: &T,
    ) -> Vec<usize> {
        let count = molecule.atomic_numbers().len();
        let mut skeleton = molecule.clone();
        for bonds in skeleton.atom_bonds_mut() {
            for bond in bonds.iter_mut() {
                bond.bond_order = BondOrder::Single;
            }
        }
        *skeleton.atom_classes_mut() = Some((0..count).collect());
        let ordered = self.canonize_in_the_order_given(skeleton);
        ordered
            .atom_classes()
            .clone()
            .filter(|origins| origins.len() == count)
            .unwrap_or_else(|| (0..count).collect())
    }

    /// One canonization pass: the labelling, then the stereo colouring that
    /// refines it. [`Canonizer::canonize_perceived`] is this and the
    /// perception it settles with.
    fn canonize_in_the_order_given<T: Molecule + Clone + ToSMILES>(
        &mut self,
        mut molecule: T,
    ) -> T {
        let components = molecule.get_components();
        if components.len() > 1 {
            return self.canonize_each_component(molecule, &components);
        }
        let (indices, labels) = prepare_and_sort_indices_and_labels(&molecule, &[]);
        self.automorphisms.clear();
        let _ = run_canonization(
            &mut molecule,
            &indices,
            &labels,
            &mut self.scratch,
            Some(&mut self.automorphisms),
            true,
        );
        // Kept before the stereo colouring refines them away, and carried into
        // each later pass's order. See `last_graph_automorphisms`.
        self.graph_automorphisms.clone_from(&self.automorphisms);
        self.split_orbits_by_stereo(&mut molecule);
        molecule
    }

    /// [`Canonizer::canonize_in_the_order_given`] for a molecule of several
    /// components.
    ///
    /// A record is a set of molecules, and its canonical form is theirs: each
    /// component is canonized on its own, exactly as it would be alone, and the
    /// record lists them in the order of their canonical forms. The atom
    /// invariants decide first, the component holding the highest-ranked atom
    /// leading as it did when nauty labelled the record whole (`Cl.CC`,
    /// `O.O=C(NC)C`), and the canonical SMILES decide between components no
    /// invariant tells apart, such as two diastereomers.
    ///
    /// Labelled whole, the record asked more of nauty than any of its parts do.
    /// Colour refinement cannot tell a cyclopropane carbon from a cyclohexane
    /// one -- each has two ring neighbours and two hydrogens -- so with rings
    /// of several sizes side by side the search had to try which ring each atom
    /// it individualized sat in, and every further set of four sizes multiplied
    /// the time by ten. PubChem's CID 157779877, 57 small rings and
    /// heterocycles, ran for hours. And a component's labelling, and with it
    /// its SMILES, depended on what else the record held.
    ///
    /// The automorphisms reported are each component's, moved to where the
    /// component now sits, and a swap of each neighbouring pair of components
    /// that are the same compound, which together give every permutation of
    /// them.
    fn canonize_each_component<T: Molecule + Clone + ToSMILES>(
        &mut self,
        molecule: T,
        components: &[Vec<usize>],
    ) -> T {
        struct Part<T> {
            canonical: T,
            origins: Vec<usize>,
            heavy: usize,
            labels: Vec<AtomInvariant>,
            smiles: Option<String>,
            automorphisms: Vec<Vec<usize>>,
            graph_automorphisms: Vec<Vec<usize>>,
        }
        let count = molecule.len();
        let mut empty = molecule.clone();
        while empty.len() > 0 {
            empty.pop_atom();
        }
        let mut place = vec![0; count];
        let mut parts: Vec<Part<T>> = Vec::with_capacity(components.len());
        for component in components {
            let alone = component_alone(&molecule, &empty, component, &mut place);
            let canonical = self.canonize_in_the_order_given(alone);
            let origins = canonical.atom_classes().clone().unwrap_or_default();
            let heavy = canonical
                .atomic_numbers()
                .iter()
                .filter(|&&atomic_number| atomic_number != 1)
                .count();
            let labels = (0..heavy)
                .map(|atom| canonical.get_atom_invariant(atom))
                .collect();
            parts.push(Part {
                canonical,
                origins,
                heavy,
                labels,
                smiles: None,
                automorphisms: std::mem::take(&mut self.automorphisms),
                graph_automorphisms: std::mem::take(&mut self.graph_automorphisms),
            });
        }

        // Invariants along each canonical order, highest first; the SMILES only
        // where those tie, without the atom classes, which are no part of the
        // canonical form.
        let mut order: Vec<usize> = (0..parts.len()).collect();
        order.sort_by(|&left, &right| parts[right].labels.cmp(&parts[left].labels));
        let mut start = 0;
        while start < order.len() {
            let end = start
                + order[start..]
                    .iter()
                    .take_while(|&&part| parts[part].labels == parts[order[start]].labels)
                    .count();
            if end - start > 1 {
                for &part in &order[start..end] {
                    let mut written = parts[part].canonical.clone();
                    *written.atom_classes_mut() = None;
                    parts[part].smiles = Some(written.to_smiles());
                }
                order[start..end]
                    .sort_by(|&left, &right| parts[left].smiles.cmp(&parts[right].smiles));
            }
            start = end;
        }

        // Heavy atoms first, component by component, then the hydrogens the
        // same way, as a single molecule's canonical order has them.
        let mut final_order = Vec::with_capacity(count);
        let mut heavy_at = Vec::with_capacity(order.len());
        for &part in &order {
            heavy_at.push(final_order.len());
            final_order.extend_from_slice(&parts[part].origins[..parts[part].heavy]);
        }
        let heavy_total = final_order.len();
        let mut light_at = Vec::with_capacity(order.len());
        for &part in &order {
            light_at.push(final_order.len());
            final_order.extend_from_slice(&parts[part].origins[parts[part].heavy..]);
        }
        let mut record = molecule;
        reorder_molecule(&mut record, &final_order);
        // Each atom's neighbours keep their order, so a tag reads the same in
        // the record as in its component -- and the component's is the one to
        // keep, since canonizing it may have settled a tag the reordering
        // alone does not reproduce, such as a free centre written `@`.
        if let Some(classes) = record.chiral_classes_mut() {
            for (slot, &part) in order.iter().enumerate() {
                let part = &parts[part];
                let Some(own) = part.canonical.chiral_classes() else {
                    continue;
                };
                for (atom, &class) in own.iter().enumerate() {
                    let at = if atom < part.heavy {
                        heavy_at[slot] + atom
                    } else {
                        light_at[slot] + atom - part.heavy
                    };
                    classes[at] = class;
                }
            }
        }

        let identity: Vec<usize> = (0..heavy_total).collect();
        let shifted = |permutations: &[Vec<usize>], at: usize| -> Vec<Vec<usize>> {
            permutations
                .iter()
                .map(|permutation| {
                    let mut full = identity.clone();
                    for (atom, &image) in permutation.iter().enumerate() {
                        full[at + atom] = at + image;
                    }
                    full
                })
                .collect()
        };
        self.automorphisms.clear();
        self.graph_automorphisms.clear();
        for (slot, &part) in order.iter().enumerate() {
            let automorphisms = shifted(&parts[part].automorphisms, heavy_at[slot]);
            let graph_automorphisms = shifted(&parts[part].graph_automorphisms, heavy_at[slot]);
            self.automorphisms.extend(automorphisms);
            self.graph_automorphisms.extend(graph_automorphisms);
        }
        for slot in 1..order.len() {
            let (left, right) = (&parts[order[slot - 1]], &parts[order[slot]]);
            if left.heavy == 0 || left.labels != right.labels || left.smiles != right.smiles {
                continue;
            }
            let mut swap = identity.clone();
            for atom in 0..left.heavy {
                swap[heavy_at[slot - 1] + atom] = heavy_at[slot] + atom;
                swap[heavy_at[slot] + atom] = heavy_at[slot - 1] + atom;
            }
            self.automorphisms.push(swap.clone());
            self.graph_automorphisms.push(swap);
        }
        record
    }

    /// Recolours the graph by each stereocentre's orbit-relative tag and
    /// canonizes again, for as long as that tells two orbit-mates apart.
    ///
    /// A canonical labelling is unique as a labelling only up to the graph's
    /// automorphisms, and a chirality tag is not part of the graph nauty sees.
    /// So a molecule carrying several copies of one arm -- a tryptophan
    /// dendrimer is the case the corpus found -- has nothing to say which copy
    /// belongs at which canonical position, and the tags land in whatever order
    /// the input held them. [`canonicalize_chirality_under_automorphisms`]
    /// settles that by taking the smallest tag vector the group can produce,
    /// but it has to walk the group to do it, and a dendrimer's group is far
    /// past the bound it walks; a truncated walk is not a canonical choice.
    ///
    /// Colouring by the tag removes the choice instead of deciding it: an arm
    /// written `R` and an arm written `S` stop being interchangeable, so the
    /// automorphism relating them is gone and the labelling that remains is
    /// unique. The colour is Aut-invariant by construction -- it reads the tag
    /// against the orbits, which every way of writing the molecule agrees on --
    /// so each pass is as canonical as the one before it. Passes strictly
    /// refine, hence terminate; the bound is there only to keep the cost of a
    /// pathological case finite.
    fn split_orbits_by_stereo<T: Molecule + Clone + ToSMILES>(&mut self, molecule: &mut T) {
        const MAX_PASSES: usize = 4;
        for _ in 0..MAX_PASSES {
            if self.automorphisms.is_empty() || !molecule.is_chiral() {
                return;
            }
            let (parities, orbit_of) = orbit_relative_tags(molecule, &self.automorphisms);
            if !splits_an_orbit(&parities, &orbit_of) {
                return;
            }
            let (indices, labels) = prepare_and_sort_indices_and_labels(molecule, &parities);
            // The pass is about to renumber the atoms twice -- once by
            // `indices`, once by the labelling nauty returns -- and the graph's
            // own symmetries are written in the numbering it starts from, so
            // they are carried through both.
            self.graph_automorphisms =
                carry_permutations_through(&self.graph_automorphisms, &indices);
            self.automorphisms.clear();
            let _ = run_canonization(
                molecule,
                &indices,
                &labels,
                &mut self.scratch,
                Some(&mut self.automorphisms),
                true,
            );
            let labeling = self.scratch.heavy_labeling.clone();
            self.graph_automorphisms =
                carry_permutations_through(&self.graph_automorphisms, &labeling);
        }
    }

    /// The automorphisms of the molecule this canonizer last canonized, as
    /// permutations of its *canonical* heavy-atom positions.
    ///
    /// A canonical labelling is unique as a labelling, but a symmetric graph
    /// has several equally canonical ways to attach data that the labelling
    /// cannot see. Hydrogen tartrate is the case that made this necessary: its
    /// erased skeleton is symmetric, so nothing in the graph says which of the
    /// two carboxyls carries the mobile hydrogen and which the charge, and the
    /// answer fell to the order the atoms happened to arrive in. A caller
    /// holding such data can use these to choose among the possibilities
    /// deterministically -- the same use
    /// `canonicalize_chirality_under_automorphisms` already makes of them for
    /// stereocentre tags.
    ///
    /// Empty when the molecule is asymmetric, which is the usual case.
    pub fn last_automorphisms(&self) -> &[Vec<usize>] {
        &self.automorphisms
    }

    /// The symmetries of the *graph* the canonizer last saw, in the canonical
    /// order it settled on.
    ///
    /// [`Canonizer::last_automorphisms`] reports what survives the stereo
    /// colouring, which is what a labelling is unique up to. That is the wrong
    /// group for data the graph does not carry -- which of two carboxyls holds
    /// a mobile hydrogen, say. Erasure took the hydrogen out of the graph, so
    /// the freedom in putting it back is the freedom the *graph* has, and
    /// colouring by a stereocentre's tag narrows the labelling without
    /// narrowing that. Reading the smaller group is how a hexaric acid's
    /// hydrogen came to move from one end of the chain to the other on every
    /// pass: the symmetry that made the two ends interchangeable had been
    /// coloured away, so nothing was left to choose between them but the order
    /// the atoms arrived in.
    pub fn last_graph_automorphisms(&self) -> &[Vec<usize>] {
        &self.graph_automorphisms
    }

    /// [`tautomer_key`](crate::molecule::tautomer::tautomer_key), reusing this
    /// canonizer's buffers.
    ///
    /// The key every tautomer of `mol` shares, for deduplicating a registry or
    /// splitting a dataset without counting one compound twice. See
    /// [`crate::molecule::tautomer`] for what the string means and what it does
    /// not promise.
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::graph_algorithms::mol_canonization::Canonizer;
    /// use oxidane::molecule::tautomer::TautomerRules;
    ///
    /// let mut canonizer = Canonizer::new();
    /// let rules = TautomerRules::new();
    /// let keto = MolecularGraph::from_smiles_single("O=c1cccc[nH]1").unwrap();
    /// let enol = MolecularGraph::from_smiles_single("Oc1ccccn1").unwrap();
    /// assert_eq!(
    ///     canonizer.tautomer_key(&keto, &rules),
    ///     canonizer.tautomer_key(&enol, &rules)
    /// );
    /// ```
    pub fn tautomer_key<T: Molecule + Clone + ToSMILES>(
        &mut self,
        mol: &T,
        rules: &crate::molecule::tautomer::TautomerRules,
    ) -> String {
        crate::molecule::tautomer::tautomer_key_with(self, mol, rules)
    }

    /// [`canonicalize_and_sort_smiles`], reusing this canonizer's buffers.
    pub fn canonical_smiles(
        &mut self,
        smiles: &str,
        with_sorting: bool,
    ) -> Result<String, SMILESParseError> {
        let mut mols = MolecularGraph::from_smiles(smiles)?;
        if mols.len() == 1 {
            return Ok(self.canonize_perceived(mols.remove(0)).to_smiles());
        }
        let mut canonical_smiles: Vec<String> = mols
            .into_iter()
            .map(|mol| self.canonize_perceived(mol).to_smiles())
            .collect();
        canonical_smiles.sort();
        let _ = with_sorting;
        Ok(canonical_smiles.join("."))
    }
}

/// `component` of `molecule` as a molecule of its own, built on `empty` (the
/// molecule with its atoms popped), its atoms in the order given, and the atom
/// classes holding each atom's place in `molecule`. `place` is scratch, one
/// slot per atom of `molecule`.
///
/// Built field by field rather than with [`reorder_molecule`], which carries
/// every atom of the record along and works out every stereo tag across all
/// of it -- once per component, so a record of many small molecules cost the
/// square of their number: twenty-four ethanes and ten salts beside them took
/// 3.5 ms for a key their parts take 0.5 ms for. No tag needs working out
/// here. `component` is sorted and nothing outside it is bonded to it, so each
/// atom's neighbours come in the same order alone as in the record, and that
/// order is what the tags are read against.
fn component_alone<T: Molecule + Clone>(
    molecule: &T,
    empty: &T,
    component: &[usize],
    place: &mut [usize],
) -> T {
    for (index, &atom) in component.iter().enumerate() {
        place[atom] = index;
    }
    let mut alone = empty.clone();
    *alone.atomic_numbers_mut() = component
        .iter()
        .map(|&atom| molecule.atomic_numbers()[atom])
        .collect();
    *alone.charges_mut() = component
        .iter()
        .map(|&atom| molecule.charges()[atom])
        .collect();
    *alone.radical_states_mut() = component
        .iter()
        .map(|&atom| molecule.radical_states()[atom])
        .collect();
    *alone.atom_bonds_mut() = component
        .iter()
        .map(|&atom| {
            let mut bonds = molecule.atom_bonds()[atom];
            for bond in &mut bonds {
                bond.target = place[bond.target()];
            }
            bonds
        })
        .collect();
    if let Some(isotopes) = molecule.isotopes() {
        *alone.isotopes_mut() = Some(component.iter().map(|&atom| isotopes[atom]).collect());
    }
    if let (Some(classes), Some(own)) = (molecule.chiral_classes(), alone.chiral_classes_mut()) {
        own.extend(component.iter().map(|&atom| classes[atom]));
    }
    *alone.atom_classes_mut() = Some(component.to_vec());
    alone
}

thread_local! {
    /// The canonizer the free functions borrow, so existing callers get the
    /// buffer reuse without threading a handle through their signatures.
    static CANONIZER: std::cell::Cell<Canonizer> = std::cell::Cell::new(Canonizer::new());
}

/// Runs `f` against the thread's canonizer, taken out and put back so that
/// anything reachable from `f` which itself canonizes gets its own rather than
/// a double-borrow panic.
pub(crate) fn with_canonizer<R>(f: impl FnOnce(&mut Canonizer) -> R) -> R {
    let mut canonizer = CANONIZER.take();
    let result = f(&mut canonizer);
    CANONIZER.set(canonizer);
    result
}

/// Canonizes a molecule, returning a new instance with canonical atom ordering.
///
/// # Examples
///
/// ```
/// use oxidane::graph_algorithms::mol_canonization::canonize_mol;
/// use oxidane::io::{SMILESParser, ToSMILES};
///
/// let mol = SMILESParser::parse_smiles_single("[N+]([O-])([O-])=O.C(CC(c1ccc(Cl)cc1Cl)(C#N)C[n+]1cn[nH]c1)CC").unwrap();
/// let canonical_mol = canonize_mol(&mol);
/// // Canonization is deterministic: canonizing again gives the same SMILES
/// assert_eq!(
///     canonize_mol(&canonical_mol).to_smiles(),
///     canonical_mol.to_smiles()
/// );
/// ```
pub fn canonize_mol<T: Molecule + Clone + ToSMILES>(mol: &T) -> T {
    with_canonizer(|canonizer| canonizer.canonize(mol))
}

/// [`canonize_mol`] for a molecule whose aromaticity has already been
/// perceived under the same model.
///
/// Perception is the single most expensive step in canonizing a molecule --
/// over a third of the time, and more than three times what the canonical
/// labelling itself costs -- and it is idempotent, so running it again on a
/// molecule that has just been through it is pure waste. Callers that parsed
/// the molecule themselves with the default options already have it done.
///
/// Not public: outside this module there is no way to know whether a molecule
/// has been perceived, and getting it wrong means canonizing the Kekule form
/// and the aromatic form to different strings. [`canonize_mol`] is the safe
/// entry point and stays that way.
fn canonize_perceived<T: Molecule + Clone + ToSMILES>(molecule: T) -> T {
    with_canonizer(|canonizer| canonizer.canonize_perceived(molecule))
}

/// [`canonize_mol`] for a molecule whose aromaticity has already been
/// perceived under the same model -- see [`Canonizer::canonize_perceived`] for
/// when that is true and what it costs to be wrong about it.
///
/// ```
/// use oxidane::prelude::*;
/// use oxidane::graph_algorithms::mol_canonization::{canonize_mol, canonize_mol_perceived};
///
/// // `from_smiles` perceives aromaticity, so this molecule is already done.
/// let molecule = MolecularGraph::from_smiles_single("C1=CC=CC=C1").unwrap();
/// assert_eq!(
///     canonize_mol_perceived(&molecule).to_smiles(),
///     canonize_mol(&molecule).to_smiles()
/// );
/// ```
pub fn canonize_mol_perceived<T: Molecule + Clone + ToSMILES>(mol: &T) -> T {
    canonize_perceived(mol.clone())
}

/// Canonizes a molecule in place, reordering its atoms according to canonical ranking.
///
/// # Examples
///
/// ```
/// use oxidane::graph_algorithms::mol_canonization::{canonize_mol, canonize_mol_inplace};
/// use oxidane::io::{SMILESParser, ToSMILES};
///
/// let mut mol = SMILESParser::parse_smiles_single("[N+]([O-])([O-])=O.C(CC(c1ccc(Cl)cc1Cl)(C#N)C[n+]1cn[nH]c1)CC").unwrap();
/// let canonical_mol = canonize_mol(&mol);
/// canonize_mol_inplace(&mut mol);
/// assert_eq!(mol.to_smiles(), canonical_mol.to_smiles());
/// ```
pub fn canonize_mol_inplace<T: Molecule + Clone + ToSMILES>(molecule: &mut T) {
    *molecule = with_canonizer(|canonizer| canonizer.canonize(molecule));
}

/// Each atom's chirality class as it reads in the molecule's *canonical* atom
/// ordering, returned in the molecule's own atom ordering.
///
/// A raw [`ChiralClass`] is a statement about one particular listing of an
/// atom's neighbours, so the same molecule written two ways gives different
/// tags for the same 3-D arrangement. Canonizing first removes that: the
/// canonical ordering depends only on the molecule, so the tags read off it
/// are a property of the structure. For instance these four SMILES all denote
/// the same octahedral cobalt and all come back as `OH(19)`:
///
/// ```rust
/// # use oxidane::prelude::*;
/// use oxidane::graph_algorithms::mol_canonization::canonical_chiral_classes;
/// use oxidane::molecule::chirality::ChiralClass;
///
/// for smiles in [
///     "[Co@OH18]([SH])([Br])([I])([CH3])([F])[Cl]",
///     "[I][Co@OH28]([Cl])([Br])([F])([CH3])[SH]",
///     "[Br][Co@OH2]([CH3])([I])([SH])([Cl])[F]",
/// ] {
///     let mol = MolecularGraph::from_smiles_single(smiles).unwrap();
///     let classes = canonical_chiral_classes(&mol);
///     assert!(classes.iter().any(|&c| c == ChiralClass::OH(19)), "{smiles}");
/// }
/// ```
///
/// This is what lets stereochemistry that CIP does not describe — octahedral,
/// trigonal-bipyramidal, square-planar and allene centres — still be compared
/// between molecules.
///
/// Canonization is not cheap, so callers that only need this for the rare
/// higher-order classes should check whether the molecule has any first.
/// The molecule in its canonical form, paired with `original_of[canonical] =
/// the atom it came from`.
///
/// Same tracer trick as [`canonical_chiral_classes`]: atom classes are not
/// part of `get_atom_invariant`, so numbering each atom with its own index
/// cannot influence which canonical ordering is chosen, and
/// [`reorder_molecule`] carries the tags along with their atoms.
///
/// Useful where a stereo tag has to be *read* rather than merely compared:
/// the canonical form is one fixed ordering per structure, so a tag has a
/// definite meaning there even when it has none against the atom's own bond
/// list.
pub fn canonical_form_with_origins<T: Molecule + Clone + ToSMILES>(
    mol: &T,
) -> Option<(T, Vec<usize>)> {
    let atom_count = mol.atomic_numbers().len();
    let mut tagged = mol.clone();
    *tagged.atom_classes_mut() = Some((0..atom_count).collect());
    let canonical = canonize_mol(&tagged);
    let origins = canonical.atom_classes().clone()?;
    if origins.len() != atom_count {
        return None;
    }
    Some((canonical, origins))
}

pub fn canonical_chiral_classes<T: Molecule + Clone + ToSMILES>(mol: &T) -> Vec<ChiralClass> {
    let atom_count = mol.atomic_numbers().len();
    let mut result = vec![ChiralClass::None; atom_count];
    if mol.chiral_classes().is_none() {
        return result;
    }

    // Tag every atom with its own index so the canonical ordering can be
    // mapped back. Atom classes are not part of `get_atom_invariant`, so
    // tagging cannot influence which ordering canonization picks, and
    // `reorder_molecule` carries them along with their atoms.
    let mut tagged = mol.clone();
    *tagged.atom_classes_mut() = Some((0..atom_count).collect());
    let canonical = canonize_mol(&tagged);

    let (Some(tags), Some(classes)) =
        (canonical.atom_classes().clone(), canonical.chiral_classes())
    else {
        return result;
    };
    for (canonical_index, &original_index) in tags.iter().enumerate() {
        if let (Some(slot), Some(&class)) =
            (result.get_mut(original_index), classes.get(canonical_index))
        {
            *slot = class;
        }
    }
    result
}

#[derive(Debug, PartialEq, Eq)]
enum CanonizationError {
    InvalidInput,
}

fn run_canonization<T: Molecule + ToSMILES + Clone>(
    molecule: &mut T,
    indices: &[usize],
    labels: &[AtomInvariant],
    scratch: &mut CanonizationScratch,
    report_automorphisms: Option<&mut Vec<Vec<usize>>>,
    choose_stereo: bool,
) -> Result<(), CanonizationError> {
    reorder_molecule(molecule, &indices);

    // `labels` is in the original atom order, but the molecule has just been
    // reordered by `indices`; reorder the labels to match so the colouring
    // partition lines up with the graph's vertex order.
    let sorted_labels = &mut scratch.sorted_labels;
    sorted_labels.clear();
    sorted_labels.extend(indices.iter().map(|&i| labels[i].clone()));

    let Some((adjacency_matrix, partition, number_of_heavy_atoms)) =
        molecule.prepare_canonization(sorted_labels)
    else {
        return Err(CanonizationError::InvalidInput);
    };

    let number_of_vertices = adjacency_matrix.number_of_vertices;
    let matrix = adjacency_matrix.matrix;

    // canonaut takes a colouring as one colour per vertex and derives the cell
    // boundaries itself; `prepare_canonization` hands back nauty's `ptn` form
    // -- a 0 after the last vertex of each cell -- over a vertex order that is
    // already grouped by colour. Numbering the cells along that order is the
    // same partition, and because the colours then rise with the vertex index,
    // canonaut's sort of the vertices by (colour, vertex) leaves the order
    // untouched.
    //
    // The colouring has to be passed: canonizing the uncoloured graph would
    // leave atom order undetermined for graph-symmetric molecules whose
    // symmetry is broken only by atom identity (e.g. a palindromic chain with
    // N on one end and O on the other).
    let colors = &mut scratch.colors;
    colors.clear();
    colors.reserve(number_of_vertices);
    let mut cell = 0u32;
    for &is_cell_end in partition.iter() {
        colors.push(cell);
        if is_cell_end == 0 {
            cell += 1;
        }
    }

    AUTOMORPHISM_GENERATORS.with(|generators| generators.borrow_mut().clear());
    scratch.manager.canonize_with_colors(&matrix, colors);

    // Nauty's output labeling is a permutation over all vertices, including the
    // virtual bond-order vertices appended after the heavy atoms in
    // `prepare_canonization`. Those virtual vertices are not guaranteed to occupy
    // only the tail positions of the canonical order, so we filter them out
    // instead of assuming the first `number_of_heavy_atoms` entries are all real atoms.
    let mut heavy_labeling = std::mem::take(&mut scratch.heavy_labeling);
    heavy_labeling.clear();
    heavy_labeling.extend(
        scratch
            .manager
            .labeling()
            .iter()
            .map(|x| *x as usize)
            .filter(|&x| x < number_of_heavy_atoms),
    );
    scratch.heavy_labeling = heavy_labeling;
    let labeling = &scratch.heavy_labeling;

    // Collect the automorphism generators nauty found (on the pre-canonical
    // graph) before reordering, so we can later pick a canonical chirality tag
    // assignment among symmetry-equivalent ones.
    let generators = AUTOMORPHISM_GENERATORS.with(|g| std::mem::take(&mut *g.borrow_mut()));

    reorder_molecule(molecule, labeling);

    // A graph automorphism can flip stereocentre tags (e.g. the bridge swap of a
    // symmetric ring), yielding several equally valid tag assignments for the
    // same molecule. Pick the canonical one deterministically.
    if choose_stereo && molecule.is_chiral() && !generators.is_empty() {
        canonicalize_chirality_under_automorphisms(
            molecule,
            &generators,
            labeling,
            number_of_heavy_atoms,
        );
    }
    if let Some(report) = report_automorphisms {
        report.extend(conjugate_into_canonical_order(
            &generators,
            labeling,
            number_of_heavy_atoms,
        ));
    }
    Ok(())
}

/// Each atom's orbit under the automorphisms, and each stereocentre's tag read
/// against those orbits.
///
/// The tag a tetrahedral centre stores is a winding, and a winding only means
/// something against an ordering of the centre's neighbours. The one it is
/// stored against -- the neighbours by atom index -- says nothing outside the
/// numbering that produced it. Sorting them by orbit instead gives an ordering
/// every way of writing the molecule agrees on, so the tag re-read against it
/// is a property of the molecule: `1` and `2` are the two windings, `0` means
/// the atom carries no tag, and `5` that it carries one no orbit can read --
/// two of its neighbours share an orbit and cannot be put in an order. That
/// is still worth a colour of its own: a record holding ten copies of an
/// amino diol, eight tagged and two not, had the eight interchangeable with
/// the two, and which copies came out tagged fell to the input.
///
/// Hydrogens are not vertices of the graph nauty saw, so they are in no orbit
/// of their own; each takes the orbit of the atom it hangs off, which is what
/// the automorphism would do with it. Two hydrogens on one centre then collide,
/// which is the right answer -- a centre with two of them is not stereogenic.
///
/// A double bond's geometry is read the same way, against the substituent of
/// each end that sorts first by orbit, and written on both ends: `3` is cis,
/// `4` trans, apart from the windings so that the two never share a colour.
/// Without it a record holding `C/C=C/C` beside `C/C=C\C` had nothing to tell
/// the two butenes apart but their tags, so which was written first fell to
/// the input; and a mixture of a dozen isomers has a group too large to walk.
fn orbit_relative_tags<T: Molecule>(
    molecule: &T,
    automorphisms: &[Vec<usize>],
) -> (Vec<u8>, Vec<u32>) {
    let atom_count = molecule.len();
    let mut parent: Vec<u32> = (0..atom_count as u32).collect();
    fn root(parent: &mut [u32], mut atom: u32) -> u32 {
        while parent[atom as usize] != atom {
            parent[atom as usize] = parent[parent[atom as usize] as usize];
            atom = parent[atom as usize];
        }
        atom
    }
    for generator in automorphisms {
        for (atom, &image) in generator.iter().enumerate() {
            if atom >= atom_count || image >= atom_count {
                continue;
            }
            let (a, b) = (
                root(&mut parent, atom as u32),
                root(&mut parent, image as u32),
            );
            if a != b {
                parent[b.max(a) as usize] = a.min(b);
            }
        }
    }
    let orbit_of: Vec<u32> = (0..atom_count as u32)
        .map(|atom| root(&mut parent, atom))
        .collect();

    // A neighbour's sort key: which orbit it is in, whether it is a hydrogen,
    // and -- so that a deuterium and a protium are not interchangeable -- its
    // isotope. A hydrogen borrows the orbit of the centre it is attached to.
    let key_of = |neighbour: usize, centre: usize| -> (u32, u8, u16) {
        let hydrogen = molecule.atomic_numbers()[neighbour] == 1;
        let orbit = if hydrogen {
            orbit_of[centre]
        } else {
            orbit_of[neighbour]
        };
        let isotope = molecule
            .isotopes()
            .and_then(|isotopes| isotopes.get(neighbour).copied())
            .unwrap_or(0);
        (orbit, u8::from(hydrogen), isotope)
    };

    let mut parities = vec![0u8; atom_count];
    for atom in 0..atom_count {
        let ChiralClass::TH(stored) = molecule.get_chiral_class(atom) else {
            continue;
        };
        let by_index = index_sorted_neighbors(molecule, atom);
        if by_index.len() < 3 || by_index.len() > 4 {
            continue;
        }
        let mut distinct: Vec<(u32, u8, u16)> = by_index.iter().map(|&n| key_of(n, atom)).collect();
        let neighbour_count = distinct.len();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() != neighbour_count {
            continue;
        }
        let mut by_orbit = by_index.clone();
        by_orbit.sort_by_key(|&n| key_of(n, atom));
        parities[atom] = tetrahedral_tag_for_order(stored, &by_index, &by_orbit);
    }

    // The stored geometry is read against the first substituent each end
    // lists; see `compute_cistrans_sides`, whose reading this follows.
    let is_cis_trans = |class: ChiralClass| {
        matches!(
            class,
            ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
        )
    };
    let substituents = |atom: usize| -> Vec<usize> {
        molecule
            .get_atom_bonds(atom)
            .unwrap_or_default()
            .iter()
            .filter(|bond| bond.bond_order() != BondOrder::Double)
            .map(|bond| bond.target())
            .collect()
    };
    for first in 0..atom_count {
        if !is_cis_trans(molecule.get_chiral_class(first)) {
            continue;
        }
        let Some(second) = molecule
            .get_atom_bonds(first)
            .unwrap_or_default()
            .iter()
            .filter(|bond| bond.bond_order() == BondOrder::Double)
            .map(|bond| bond.target())
            .find(|&target| is_cis_trans(molecule.get_chiral_class(target)))
        else {
            continue;
        };
        if second < first {
            continue;
        }
        // The substituent an end's geometry is read against, or `None` when
        // it has two that no orbit tells apart.
        let reference = |end: usize, listed: &[usize]| -> Option<usize> {
            let first_by_orbit = *listed.iter().min_by_key(|&&n| key_of(n, end))?;
            let ties = listed
                .iter()
                .filter(|&&n| key_of(n, end) == key_of(first_by_orbit, end))
                .count();
            (ties == 1).then_some(first_by_orbit)
        };
        let (listed_first, listed_second) = (substituents(first), substituents(second));
        let (Some(by_orbit_first), Some(by_orbit_second)) = (
            reference(first, &listed_first),
            reference(second, &listed_second),
        ) else {
            continue;
        };
        let mut trans = molecule.get_chiral_class(first) != molecule.get_chiral_class(second);
        trans ^= by_orbit_first != listed_first[0];
        trans ^= by_orbit_second != listed_second[0];
        let geometry = if trans { 4 } else { 3 };
        parities[first] = geometry;
        parities[second] = geometry;
    }
    for (atom, parity) in parities.iter_mut().enumerate() {
        if *parity == 0 && molecule.get_chiral_class(atom) != ChiralClass::None {
            *parity = 5;
        }
    }
    (parities, orbit_of)
}

/// Whether any orbit holds two atoms whose orbit-relative tags disagree, which
/// is what makes recolouring by them worth another pass.
fn splits_an_orbit(parities: &[u8], orbit_of: &[u32]) -> bool {
    let mut seen: FxHashMap<u32, u8> = FxHashMap::default();
    for (atom, &parity) in parities.iter().enumerate() {
        match seen.entry(orbit_of[atom]) {
            std::collections::hash_map::Entry::Occupied(slot) => {
                if *slot.get() != parity {
                    return true;
                }
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(parity);
            }
        }
    }
    false
}

/// Rewrites nauty's generators, which permute the pre-canonical vertices, as
/// permutations of the canonical heavy-atom positions.
///
/// A generator that moves a heavy atom onto one of the virtual bond-order
/// vertices is dropped: it says nothing about where an atom's data may go.
/// Whether perception left an atom out of an aromatic system it borders.
///
/// Asking whether perception is stable means perceiving a second time, and that
/// is a third of what canonizing costs -- worth paying only where the answer
/// can be no. It can be no where perception had a choice to make: an atom with
/// no aromatic bond of its own, bonded to two that have them, is a ring atom
/// left out of a system it borders, and whether it belongs is the kind of
/// question a fused polycyclic answers differently depending on where the walk
/// started. A graphene flake's carbonyl ring is such an atom; a peptide's
/// tryptophans, whose aromatic systems border `sp3` carbon and nothing else,
/// are not.
fn perception_had_a_choice_to_make<T: Molecule>(molecule: &T) -> bool {
    let aromatic: Vec<bool> = molecule
        .atom_bonds()
        .iter()
        .map(|bonds| {
            bonds
                .iter()
                .any(|bond| bond.bond_order() == BondOrder::Aromatic)
        })
        .collect();
    let an_atom_left_between_two_rings =
        molecule
            .atom_bonds()
            .iter()
            .enumerate()
            .any(|(atom, bonds)| {
                !aromatic[atom]
                    && bonds
                        .iter()
                        .filter(|bond| aromatic.get(bond.target()).copied().unwrap_or(false))
                        .count()
                        >= 2
            });
    // ... or a bond perception left out of a system it lies *inside*. A
    // fullerene has no atom outside to be noticed by the test above -- every
    // carbon of the cage is aromatic -- and yet the same cage comes back with
    // twenty-four of its bonds written `=` from the deposited Kekule form and
    // none of them from its own canonical SMILES. The atoms agreed; the bonds
    // did not.
    //
    // The bond has to be one of a *ring*. A plain bond between two aromatic
    // atoms is biphenyl, where nothing was left out and the two rings are
    // simply not one system; firing on that would put every biaryl in the
    // corpus through a re-settle it does not need. Inside a ring the same bond
    // means the opposite -- perception took the ring's atoms and left this bond
    // behind -- and a fullerene shows it as twenty-four bonds written `=` from
    // the deposited Kekule form and none from its own canonical SMILES, while a
    // porphyrazine shows it as a handful written `-`.
    //
    // Ring membership is only asked for once a candidate bond exists, which is
    // what keeps it off the hot path: a molecule whose aromatic atoms are
    // joined to each other by aromatic bonds alone never pays for it.
    let a_candidate = molecule
        .atom_bonds()
        .iter()
        .enumerate()
        .any(|(atom, bonds)| {
            aromatic[atom]
                && bonds.iter().any(|bond| {
                    bond.bond_order() != BondOrder::Aromatic
                        && aromatic.get(bond.target()).copied().unwrap_or(false)
                })
        });
    if !a_candidate {
        return an_atom_left_between_two_rings;
    }
    let (_, cyclic) = molecule.find_cyclic_bonds_and_atoms();
    let a_ring_bond_left_out = molecule
        .atom_bonds()
        .iter()
        .enumerate()
        .any(|(atom, bonds)| {
            aromatic[atom]
                && bonds.iter().any(|bond| {
                    bond.bond_order() != BondOrder::Aromatic
                        && aromatic.get(bond.target()).copied().unwrap_or(false)
                        && cyclic
                            .contains(&crate::molecule::utils::order_tuple(atom, bond.target()))
                })
        });
    an_atom_left_between_two_rings || a_ring_bond_left_out
}

/// Rewrites permutations of the heavy atoms into the order `labeling` puts
/// them in, where `labeling[i]` is the atom that ends up at position `i`.
///
/// [`conjugate_into_canonical_order`] does the same for what nauty reports,
/// which is indexed over the vertices of the graph it saw; this one is for
/// permutations already indexed over heavy atoms, so that a symmetry found in
/// one pass can be read in the next pass's numbering.
fn carry_permutations_through(permutations: &[Vec<usize>], labeling: &[usize]) -> Vec<Vec<usize>> {
    let width = permutations.first().map_or(0, Vec::len);
    if width == 0 {
        return Vec::new();
    }
    let mut inverse = vec![usize::MAX; width];
    for (position, &atom) in labeling.iter().take(width).enumerate() {
        if atom < width {
            inverse[atom] = position;
        }
    }
    permutations
        .iter()
        .filter_map(|permutation| {
            (0..width)
                .map(|position| {
                    let atom = *labeling.get(position)?;
                    let image = *permutation.get(atom)?;
                    inverse
                        .get(image)
                        .copied()
                        .filter(|&slot| slot != usize::MAX)
                })
                .collect::<Option<Vec<usize>>>()
        })
        .collect()
}

fn conjugate_into_canonical_order(
    generators: &[Vec<u32>],
    labeling: &[usize],
    number_of_heavy_atoms: usize,
) -> Vec<Vec<usize>> {
    let mut inverse = vec![0usize; number_of_heavy_atoms];
    for (canonical_position, &precanonical) in labeling.iter().enumerate() {
        if precanonical < number_of_heavy_atoms {
            inverse[precanonical] = canonical_position;
        }
    }
    let mut conjugated = Vec::new();
    for generator in generators {
        let mut permutation = vec![0usize; number_of_heavy_atoms];
        let mut usable = true;
        for canonical_position in 0..number_of_heavy_atoms {
            let pre = labeling[canonical_position];
            let image = generator.get(pre).copied().unwrap_or(u32::MAX) as usize;
            if image >= number_of_heavy_atoms {
                usable = false;
                break;
            }
            permutation[canonical_position] = inverse[image];
        }
        if usable {
            conjugated.push(permutation);
        }
    }
    conjugated
}

/// Chooses a canonical chirality tag assignment among those related by graph
/// automorphisms.
///
/// `generators` are nauty automorphism generators over the pre-canonical vertices
/// (heavy atoms + bond vertices); `labeling[i]` is the pre-canonical heavy atom
/// placed at canonical position `i`. Each automorphism is conjugated into the
/// canonical atom order, extended over hydrogens, then applied to the molecule via
/// `reorder_molecule` (which updates every chirality class). The lexicographically
/// smallest resulting tag vector is adopted, making the result independent of which
/// symmetric input produced it.
fn canonicalize_chirality_under_automorphisms<T: Molecule + ToSMILES + Clone>(
    molecule: &mut T,
    generators: &[Vec<u32>],
    labeling: &[usize],
    number_of_heavy_atoms: usize,
) {
    // Canonical position of each pre-canonical heavy atom.
    let mut inverse_labeling = vec![0usize; number_of_heavy_atoms];
    for (canonical_pos, &precanonical) in labeling.iter().enumerate() {
        inverse_labeling[precanonical] = canonical_pos;
    }

    // Conjugate each generator into the canonical heavy-atom order and extend it
    // over hydrogens to a full-atom permutation.
    let mut full_generators: Vec<Vec<usize>> = Vec::new();
    for generator in generators {
        let mut heavy_perm = vec![0usize; number_of_heavy_atoms];
        let mut valid = true;
        for canonical_pos in 0..number_of_heavy_atoms {
            let pre = labeling[canonical_pos];
            let image_pre = generator[pre] as usize;
            if image_pre >= number_of_heavy_atoms {
                valid = false;
                break;
            }
            heavy_perm[canonical_pos] = inverse_labeling[image_pre];
        }
        if !valid {
            continue;
        }
        if let Some(full) = extend_permutation_over_hydrogens(molecule, &heavy_perm) {
            full_generators.push(full);
        }
    }
    let watched = atoms_a_tag_depends_on(molecule);
    let moves_a_tag = |generator: &Vec<usize>| {
        watched
            .iter()
            .any(|&atom| generator.get(atom) != Some(&atom))
    };
    if full_generators.is_empty() || !full_generators.iter().any(moves_a_tag) {
        return;
    }

    // Only the tags the orbits leave open are the walk's to decide, and only
    // the group's action on the atoms those tags read can decide them.
    let open = open_tags(molecule, &full_generators);
    let turn_over = |classes: &mut [ChiralClass]| {
        for &centre in &open.free {
            classes[centre] = ChiralClass::TH(1);
        }
    };
    let current = molecule.chiral_classes().unwrap().to_vec();
    let mut best = current.clone();
    turn_over(&mut best);
    let generators: Vec<&Vec<usize>> = full_generators
        .iter()
        .filter(|generator| open.read.iter().any(|&atom| generator[atom] != atom))
        .collect();

    // Walk the group's action on those atoms -- each element of it once, by a
    // full automorphism that acts so -- and keep the smallest tag vector any
    // of them produces. Two automorphisms that agree there rewrite the tags
    // the same way, and a record's group is mostly made of ones that do not
    // touch them at all: the phenyls of four triphenylphosphines turn through
    // 48^4 * 4! elements, every one of which leaves a tartrate's centres where
    // they were. Walking the whole group, bounded, spent the bound on those and
    // reached the tartrate's own symmetry in some atom orders and not others.
    // The bound now holds only for a group whose action on open tags is itself
    // that large.
    const MAX_GROUP: usize = 4096;
    let identity: Vec<usize> = (0..molecule.len()).collect();
    let restriction =
        |element: &[usize]| -> Vec<usize> { open.read.iter().map(|&atom| element[atom]).collect() };
    let mut seen: FxHashSet<Vec<usize>> = FxHashSet::default();
    seen.insert(restriction(&identity));
    let mut frontier = vec![identity];
    while let Some(element) = frontier.pop() {
        if seen.len() > MAX_GROUP {
            break;
        }
        for generator in &generators {
            let composed: Vec<usize> = element.iter().map(|&i| generator[i]).collect();
            if seen.insert(restriction(&composed)) {
                let mut candidate = molecule.clone();
                reorder_molecule(&mut candidate, &composed);
                if let Some(classes) = candidate.chiral_classes_mut() {
                    turn_over(classes);
                    if *classes < best {
                        best = classes.to_vec();
                    }
                }
                frontier.push(composed);
            }
        }
    }

    if best != current {
        if let Some(classes) = molecule.chiral_classes_mut() {
            classes.copy_from_slice(&best);
        }
    }
}

/// What is left for [`canonicalize_chirality_under_automorphisms`] to decide.
struct OpenTags {
    /// The atoms whose images decide every tag an automorphism can change and
    /// no other automorphism can change back, closed under the automorphisms.
    read: Vec<usize>,
    /// Tetrahedral centres whose tag some automorphism turns over and leaves
    /// everything else open alone. Each is written `TH(1)`.
    free: Vec<usize>,
}

/// Which tags an automorphism can change, and which of those it can change
/// freely.
///
/// A tag the orbits settle cannot be changed by any automorphism: read against
/// its neighbours sorted by orbit it says the same thing wherever the
/// automorphism carries it, and when every atom of its orbit says the same,
/// the tag that lands on each is the one already there. That is the case
/// [`Canonizer::split_orbits_by_stereo`] leaves behind -- it recolours until no
/// orbit disagrees. What is left open is a tag whose centre has two
/// neighbours in one orbit, which no orbit can read, and any tag in an orbit
/// that still disagrees, or that holds an untagged atom it could be carried
/// onto.
///
/// Of those, a centre one automorphism turns over while fixing every other
/// open tag has no configuration to speak of: COOH and COO⁻ on one carbon are
/// one group once the key has pooled the proton, and the swap of the two
/// turns the carbon's tag and nothing else. Conjugating that swap gives the
/// same for every centre of its orbit, so the whole orbit is free, and each of
/// its tags can be written the smaller way without a walk to find the element
/// that does it -- which seven copies of such a centre in one record made
/// 7! * 2^7, far past the bound. Only an orbit tagged throughout counts; one
/// that mixes tagged and untagged centres still has to be walked to decide
/// where the tags go.
///
/// An open tag reads its centre and the centre's neighbours, and an allene's
/// its carriers one bond further out. Every orbit holding one of those is
/// returned whole, so that each automorphism permutes the set.
fn open_tags<T: Molecule>(molecule: &T, generators: &[Vec<usize>]) -> OpenTags {
    let Some(classes) = molecule.chiral_classes() else {
        return OpenTags {
            read: Vec::new(),
            free: Vec::new(),
        };
    };
    let (readings, orbit_of) = orbit_relative_tags(molecule, generators);
    let mut reading_of_orbit: FxHashMap<u32, Option<u8>> = FxHashMap::default();
    for (atom, &reading) in readings.iter().enumerate() {
        reading_of_orbit
            .entry(orbit_of[atom])
            .and_modify(|shared| {
                if *shared != Some(reading) {
                    *shared = None;
                }
            })
            .or_insert(Some(reading));
    }
    let settled = |atom: usize| {
        (1..=4).contains(&readings[atom])
            && reading_of_orbit[&orbit_of[atom]] == Some(readings[atom])
    };
    let reads = |atom: usize| -> Vec<usize> {
        let mut read = vec![atom];
        for bond in &molecule.atom_bonds()[atom] {
            read.push(bond.target());
            if matches!(classes[atom], ChiralClass::AL(_)) {
                read.extend(
                    molecule.atom_bonds()[bond.target()]
                        .iter()
                        .map(|further| further.target()),
                );
            }
        }
        read
    };
    let open: Vec<usize> = (0..molecule.len())
        .filter(|&atom| classes[atom] != ChiralClass::None && !settled(atom))
        .collect();
    // How many open tags read each atom.
    let mut readers = vec![0u32; molecule.len()];
    for &atom in &open {
        let mut read = reads(atom);
        read.sort_unstable();
        read.dedup();
        for other in read {
            readers[other] += 1;
        }
    }

    // An orbit is free once a generator turns one of its centres over alone.
    let mut free_orbits: FxHashSet<u32> = FxHashSet::default();
    for &centre in &open {
        if !matches!(classes[centre], ChiralClass::TH(_))
            || readings[centre] != 5
            || reading_of_orbit[&orbit_of[centre]] != Some(5)
            || free_orbits.contains(&orbit_of[centre])
        {
            continue;
        }
        let neighbours = index_sorted_neighbors(molecule, centre);
        let own = reads(centre);
        let turns_it_alone = |generator: &Vec<usize>| {
            if generator[centre] != centre {
                return false;
            }
            let images: Vec<usize> = neighbours.iter().map(|&n| generator[n]).collect();
            neighbor_permutation_is_odd(&neighbours, &images)
                && (0..molecule.len()).all(|atom| {
                    generator[atom] == atom || readers[atom] <= u32::from(own.contains(&atom))
                })
        };
        if generators.iter().any(turns_it_alone) {
            free_orbits.insert(orbit_of[centre]);
        }
    }
    let free: Vec<usize> = open
        .iter()
        .copied()
        .filter(|&atom| free_orbits.contains(&orbit_of[atom]))
        .collect();

    let mut read = vec![false; molecule.len()];
    for &atom in &open {
        if free_orbits.contains(&orbit_of[atom]) {
            continue;
        }
        for other in reads(atom) {
            read[other] = true;
        }
    }
    let orbits: FxHashSet<u32> = (0..molecule.len())
        .filter(|&atom| read[atom])
        .map(|atom| orbit_of[atom])
        .collect();
    OpenTags {
        read: (0..molecule.len())
            .filter(|&atom| orbits.contains(&orbit_of[atom]))
            .collect(),
        free,
    }
}

/// The atoms a chirality tag reads: every atom carrying one, and every
/// neighbour of such an atom.
///
/// A tag is a winding against an ordering of its centre's neighbours, so it
/// changes under a permutation only if the centre moves or its neighbours are
/// permuted among themselves. Everything else in the molecule can be permuted
/// freely without a tag noticing -- and a molecule carries a large automorphism
/// group mostly for such reasons: a peptide's methyls, its amide `NH2`s, the
/// flip of a tyrosine ring. Walking the group is the dominant cost of
/// canonizing a large molecule, so knowing which atoms matter is what makes the
/// walk affordable.
fn atoms_a_tag_depends_on<T: Molecule>(molecule: &T) -> Vec<usize> {
    let Some(classes) = molecule.chiral_classes() else {
        return Vec::new();
    };
    let mut watched = vec![false; molecule.len()];
    for (atom, class) in classes.iter().enumerate() {
        if *class == ChiralClass::None {
            continue;
        }
        watched[atom] = true;
        for bond in &molecule.atom_bonds()[atom] {
            watched[bond.target()] = true;
        }
    }
    watched
        .iter()
        .enumerate()
        .filter_map(|(atom, &watched)| watched.then_some(atom))
        .collect()
}

/// Extends a heavy-atom permutation to a full-atom permutation by mapping each
/// atom's hydrogens onto the hydrogens of its image (ascending index order).
/// A hydrogen on no heavy atom maps to itself. Returns `None` if the mapping is
/// not a bijection (mismatched hydrogen counts).
fn extend_permutation_over_hydrogens<T: Molecule>(
    molecule: &T,
    heavy_perm: &[usize],
) -> Option<Vec<usize>> {
    let n = molecule.len();
    let number_of_heavy_atoms = heavy_perm.len();
    let mut full = vec![usize::MAX; n];
    for (atom, &image) in heavy_perm.iter().enumerate() {
        full[atom] = image;
    }

    let hydrogens_of = |center: usize| -> Vec<usize> {
        let mut hs: Vec<usize> = molecule.atom_bonds()[center]
            .iter()
            .map(|bond| bond.target())
            .filter(|&t| molecule.atomic_numbers()[t] == 1)
            .collect();
        hs.sort_unstable();
        hs
    };

    for atom in 0..number_of_heavy_atoms {
        let source_hydrogens = hydrogens_of(atom);
        let image_hydrogens = hydrogens_of(heavy_perm[atom]);
        if source_hydrogens.len() != image_hydrogens.len() {
            return None;
        }
        for (h, image_h) in source_hydrogens.into_iter().zip(image_hydrogens) {
            full[h] = image_h;
        }
    }
    // A hydrogen on no heavy atom -- a bare proton, a hydride, either end of
    // `[H][H]` -- is no part of the graph nauty saw, and every automorphism
    // fixes it. Left unmapped it failed the bijection test below and took the
    // generator with it, so a record holding `[H+]` beside the E and Z forms
    // of one alkene never had its tags canonized, and the two came out in
    // whichever order the input gave them.
    for (atom, image) in full.iter_mut().enumerate() {
        if *image == usize::MAX
            && molecule.atomic_numbers()[atom] == 1
            && molecule.atom_bonds()[atom]
                .iter()
                .all(|bond| molecule.atomic_numbers()[bond.target()] == 1)
        {
            *image = atom;
        }
    }

    if full.iter().any(|&x| x == usize::MAX) {
        return None;
    }
    Some(full)
}

/// The atom invariants, and the order they put the atoms in.
///
/// `stereo_parities` is empty on the first pass, and on a later one holds the
/// orbit-relative tag of every atom, which joins the invariant as a colour.
fn prepare_and_sort_indices_and_labels<T: Molecule>(
    mol: &T,
    stereo_parities: &[u8],
) -> (Vec<usize>, Vec<AtomInvariant>) {
    let mut labels = (0..mol.len())
        .map(|atom_index| mol.get_atom_invariant(atom_index))
        .collect::<Vec<_>>();
    for (label, &parity) in labels.iter_mut().zip(stereo_parities) {
        label.set_stereo_parity(parity);
    }
    let labels = labels;
    let indices = (0..mol.len()).collect::<Vec<usize>>();

    // Hydrogens last, and everything else -- the `*` wildcard, atomic number
    // zero, included -- ahead of them by descending invariant: the graph
    // `prepare_canonization` builds numbers the non-hydrogen atoms first.
    let mut label_indices = indices.iter().zip(labels.iter()).collect::<Vec<_>>();
    label_indices.sort_by(|(index1, label1), (index2, label2)| {
        let heavy = |index: usize| mol.get_atomic_number(index) != 1;
        heavy(**index2)
            .cmp(&heavy(**index1))
            .then_with(|| label2.cmp(label1))
    });

    let indices = label_indices
        .iter()
        .map(|(index, _)| **index)
        .collect::<Vec<usize>>();

    (indices, labels)
}

/// Reorders the atoms in a molecule according to a given labeling
pub fn reorder_molecule<T: Molecule + ToSMILES>(mol: &mut T, labeling: &[usize]) {
    // Nothing to permute, and the bond-table rebuild below seeds itself from
    // `atom_bonds()[0]`, which does not exist for an empty molecule.
    if mol.len() == 0 {
        return;
    }

    let mut inverse_mapping = (0..mol.len()).collect::<Vec<usize>>();
    let mut new_labeling = vec![0; mol.len()];

    for index in 0..mol.len() {
        new_labeling[index] = labeling.get(index).copied().unwrap_or(index);
    }

    for index in 0..mol.len() {
        inverse_mapping[new_labeling[index]] = index;
    }

    let labeling = &new_labeling;

    // Capture the SMILES neighbor emission order of every atom BEFORE the atoms
    // are relabeled; chirality tags are defined relative to this order.
    let is_chiral = mol.is_chiral();
    let emission_old = if is_chiral {
        mol.smiles_emission_orders()
    } else {
        Vec::new()
    };

    // Capture cis/trans geometry as an absolute side (up/down) for every
    // single-bond neighbor of a stereogenic double bond, in the OLD indexing.
    // The per-atom parity update cannot maintain the invariant, so these classes
    // are recomputed from the sides after relabeling instead.
    // An allene's tag is stored against its four carriers, which live one atom
    // further out than the centre's own neighbours, so they have to be read
    // before the relabeling moves everything.
    let allene_carriers_old: FxHashMap<usize, [usize; 4]> = if is_chiral {
        mol.chiral_classes()
            .map(|classes| {
                classes
                    .iter()
                    .enumerate()
                    .filter(|(_, class)| matches!(class, ChiralClass::AL(_)))
                    .filter_map(|(centre, _)| {
                        index_sorted_allene_carriers(mol, centre).map(|c| (centre, c))
                    })
                    .collect()
            })
            .unwrap_or_default()
    } else {
        FxHashMap::default()
    };

    let (cistrans_sides, cistrans_bonds) = if is_chiral {
        compute_cistrans_sides(mol)
    } else {
        (FxHashMap::default(), Vec::new())
    };

    // Reorder atomic numbers
    let mut new_atomic_numbers = vec![0; mol.atomic_numbers().len()];
    for (new_idx, &old_idx) in labeling.iter().enumerate() {
        new_atomic_numbers[new_idx] = mol.atomic_numbers()[old_idx];
    }
    *mol.atomic_numbers_mut() = new_atomic_numbers;

    // Reorder charges
    let mut new_charges = vec![0; mol.atomic_numbers().len()];
    for (new_idx, &old_idx) in labeling.iter().enumerate() {
        new_charges[new_idx] = mol.charges().get(old_idx).copied().unwrap_or(0);
    }
    *mol.charges_mut() = new_charges;

    // Reorder radical states. Missing this moved the unpaired electron onto
    // whichever atom inherited the index, so `C[Si](C)CC` and `CC[Si](C)C` --
    // the same molecule -- canonicalized to `[Si]([CH3])(C)CC` and
    // `[Si](C)([CH3])CC`: the radical stayed at the silicon's *old* index and
    // was written out as a methyl radical, on a different methyl each time.
    let mut new_radical_states = vec![0; mol.atomic_numbers().len()];
    for (new_idx, &old_idx) in labeling.iter().enumerate() {
        new_radical_states[new_idx] = mol.radical_states().get(old_idx).copied().unwrap_or(0);
    }
    *mol.radical_states_mut() = new_radical_states;

    // Reorder isotopes and atom classes. Both are per-atom labels, so leaving
    // them in place would silently move a `[13C]` or a `[C:1]` onto whichever
    // atom happened to land at that index — which, for isotopes, changes the
    // molecule's CIP descriptors.
    if let Some(old_isotopes) = mol.isotopes().cloned() {
        let new_isotopes = labeling
            .iter()
            .map(|&old_idx| old_isotopes.get(old_idx).copied().unwrap_or(0))
            .collect();
        *mol.isotopes_mut() = Some(new_isotopes);
    }
    if let Some(old_classes) = mol.atom_classes().clone() {
        let new_classes = labeling
            .iter()
            .map(|&old_idx| old_classes.get(old_idx).copied().unwrap_or(0))
            .collect();
        *mol.atom_classes_mut() = Some(new_classes);
    }

    // Reorder bonds and update bond targets
    let mut new_bonds = vec![mol.atom_bonds()[0]; mol.atom_bonds().len()];
    for (new_idx, &old_idx) in labeling.iter().enumerate() {
        let atom_bonds = mol.atom_bonds();
        let mut bonds = atom_bonds[old_idx];
        for bond in &mut bonds {
            bond.target = inverse_mapping[bond.target()];
        }
        new_bonds[new_idx] = bonds;
    }

    *mol.atom_bonds_mut() = new_bonds;
    mol.atom_bonds_mut()
        .iter_mut()
        .for_each(|bonds| bonds.sort_by_key(|bond| bond.target));

    // Update chirality tags so they stay consistent with the new SMILES
    // emission neighbor order. A tetrahedral/cis-trans/allene tag flips when the
    // neighbor listing order changes by an odd permutation; an octahedral tag is
    // remapped through its full permutation.
    if is_chiral {
        let old_classes = mol.chiral_classes().unwrap().to_vec();
        let emission_new = mol.smiles_emission_orders();
        let mut new_chiral_classes = vec![ChiralClass::None; old_classes.len()];

        for (old_idx, &old_class) in old_classes.iter().enumerate() {
            if old_class == ChiralClass::None {
                continue;
            }
            // Cis/trans is recomputed per double bond from the captured sides.
            if matches!(
                old_class,
                ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
            ) {
                continue;
            }
            let new_idx = inverse_mapping[old_idx];

            // Old emission order of this centre, translated into new indices.
            let old_order_mapped: Vec<usize> = if old_idx < emission_old.len() {
                emission_old[old_idx]
                    .iter()
                    .map(|&t| inverse_mapping[t])
                    .collect()
            } else {
                Vec::new()
            };

            let updated = if let ChiralClass::TH(reference) = old_class {
                // The tetrahedral tag is stored against the centre's index-sorted
                // neighbor ordering. Relabeling only permutes those indices, so
                // the tag flips iff sorting the relabeled neighbors is an odd
                // permutation of the old index-sorted order carried through the
                // relabeling.
                // Read actual bonds, not the SMILES traversal: a folded
                // hydrogen is still one neighbor, regardless of the root used
                // to write the molecule. The new bond table already carries
                // the relabeling; sorting by each neighbor's old index recovers
                // the old reference order translated into the new indices.
                let new_sorted = index_sorted_neighbors(mol, new_idx);
                let mut old_sorted_mapped = new_sorted.clone();
                old_sorted_mapped.sort_unstable_by_key(|&neighbor| labeling[neighbor]);
                let tag = tetrahedral_tag_for_order(reference, &old_sorted_mapped, &new_sorted);
                ChiralClass::TH(tag)
            } else if let ChiralClass::AL(reference) = old_class {
                // Same idea as the tetrahedral branch, one atom further out:
                // carry the old carrier order through the relabeling and ask
                // whether re-sorting it is an odd permutation. The centre's
                // own neighbour order compares two atoms where four are
                // needed, so it cannot encode the winding.
                match (
                    allene_carriers_old.get(&old_idx),
                    index_sorted_allene_carriers(mol, new_idx),
                ) {
                    (Some(old_carriers), Some(new_carriers)) => {
                        let mapped: Vec<usize> =
                            old_carriers.iter().map(|&t| inverse_mapping[t]).collect();
                        ChiralClass::AL(tetrahedral_tag_for_order(
                            reference,
                            &mapped,
                            &new_carriers,
                        ))
                    }
                    _ => old_class,
                }
            } else {
                let new_order = emission_new.get(new_idx).cloned().unwrap_or_default();
                update_chirality_for_reorder(old_class, &old_order_mapped, &new_order)
            };
            new_chiral_classes[new_idx] = updated;
        }

        // Recompute cis/trans classes for each double bond from the captured
        // sides, so `class(a) != class(b)` still means the new first single-bond
        // neighbors are trans.
        for &(old_a, old_b) in &cistrans_bonds {
            let a = inverse_mapping[old_a];
            let b = inverse_mapping[old_b];
            let side_of_first = |carbon: usize, partner: usize| -> Option<bool> {
                let sub0 = mol
                    .get_atom_bonds(carbon)?
                    .iter()
                    .filter(|bo| bo.bond_order() != BondOrder::Double && bo.target() != partner)
                    .map(|bo| bo.target())
                    .next()?;
                // The side was captured per (carbon, sub) under the old indexing,
                // because a linker atom is a substituent of two double bonds.
                cistrans_sides
                    .get(&(labeling[carbon], labeling[sub0]))
                    .copied()
            };
            let (Some(side_a), Some(side_b)) = (side_of_first(a, b), side_of_first(b, a)) else {
                continue;
            };
            // The reference end is the *lower new index*, not whichever end
            // happened to be lower before the relabeling. Only the pair
            // `class(a) != class(b)` carries the geometry, so a trans bond can
            // be written either way round, and writing it by where its atoms
            // came from left the tag vector a record of history rather than of
            // the molecule. `canonicalize_chirality_under_automorphisms` picks
            // the lexicographically smallest vector the automorphism group can
            // produce, and two spellings of one geometry compare as different
            // vectors, so it picked whichever the input happened to hand it:
            // `CCC/C=C/CC/C=C\CCC` and `CCC/C=C\CC/C=C/CCC` -- the same
            // compound, symmetric about its middle -- canonicalized apart.
            let (low, high) = (a.min(b), a.max(b));
            new_chiral_classes[low] = ChiralClass::CisTransCounterclockwise;
            new_chiral_classes[high] = if side_a != side_b {
                ChiralClass::CisTransClockwise
            } else {
                ChiralClass::CisTransCounterclockwise
            };
        }

        *mol.chiral_classes_mut().unwrap() = new_chiral_classes;
    }
}

/// Captures cis/trans geometry as an absolute side bit for each single-bond
/// neighbor of a stereogenic double bond, plus the list of double bonds.
///
/// For a double bond `a=b`, the first single neighbor of `a` defines the down
/// side; the first single neighbor of `b` is on the same side iff the classes are
/// equal (cis). The second neighbor on each carbon takes the opposite side. These
/// sides are invariant under relabeling, so classes can be rebuilt from them.
fn compute_cistrans_sides<T: Molecule>(
    mol: &T,
) -> (FxHashMap<(usize, usize), bool>, Vec<(usize, usize)>) {
    let mut sides: FxHashMap<(usize, usize), bool> = FxHashMap::default();
    let mut bonds: Vec<(usize, usize)> = Vec::new();
    let Some(classes) = mol.chiral_classes() else {
        return (sides, bonds);
    };
    let is_cis_trans = |c: ChiralClass| {
        matches!(
            c,
            ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise
        )
    };

    for a in 0..mol.len() {
        if !is_cis_trans(classes[a]) {
            continue;
        }
        let Some(bonds_a) = mol.get_atom_bonds(a) else {
            continue;
        };
        // The double bond the class is *about*, which is not always the first
        // one the atom lists. An atom with two double bonds has only one class
        // to describe them with, and taking the first left a phosphaalkene's
        // phosphorus paired with its unclassed partner: the bond was dropped
        // from `bonds`, its sides were never captured, and the rebuild after
        // relabelling had nothing to restore, so the geometry the reader *had*
        // read was thrown away by canonization alone.
        let Some(b) = bonds_a
            .iter()
            .filter(|bo| bo.bond_order() == BondOrder::Double)
            .map(|bo| bo.target())
            .find(|&target| is_cis_trans(classes[target]))
        else {
            continue;
        };
        if b <= a {
            continue;
        }

        let subs_a: Vec<usize> = bonds_a
            .iter()
            .filter(|bo| bo.bond_order() != BondOrder::Double)
            .map(|bo| bo.target())
            .collect();
        let subs_b: Vec<usize> = mol
            .get_atom_bonds(b)
            .unwrap_or_default()
            .iter()
            .filter(|bo| bo.bond_order() != BondOrder::Double && bo.target() != a)
            .map(|bo| bo.target())
            .collect();
        if subs_a.is_empty() || subs_b.is_empty() {
            continue;
        }

        let trans = classes[a] != classes[b];
        // Keyed by (carbon, sub): subs_a[0] = down (reference); subs_b[0] same
        // side iff cis. A linker atom belongs to two double bonds, so its side is
        // stored separately for each carbon it neighbors.
        sides.insert((a, subs_a[0]), true);
        if let Some(&s) = subs_a.get(1) {
            sides.insert((a, s), false);
        }
        let b0_down = !trans;
        sides.insert((b, subs_b[0]), b0_down);
        if let Some(&s) = subs_b.get(1) {
            sides.insert((b, s), !b0_down);
        }
        bonds.push((a, b));
    }
    (sides, bonds)
}

/// Recomputes a chirality class after the SMILES neighbor emission order of its
/// centre changed from `old_order` to `new_order` (both expressed in the new
/// atom indexing, same neighbor multiset). Tetrahedral, cis/trans and allene
/// tags flip on an odd permutation; octahedral tags are remapped in full; any
/// other class is passed through unchanged.
fn update_chirality_for_reorder(
    class: ChiralClass,
    old_order: &[usize],
    new_order: &[usize],
) -> ChiralClass {
    match class {
        ChiralClass::OH(config) => update_oh_chirality(config, old_order, new_order),
        ChiralClass::TB(config) => remap_tb_number(config, old_order, new_order)
            .map(ChiralClass::TB)
            .unwrap_or(class),
        ChiralClass::SP(config) => remap_sp_number(config, old_order, new_order)
            .map(ChiralClass::SP)
            .unwrap_or(class),
        ChiralClass::TH(_)
        | ChiralClass::AL(_)
        | ChiralClass::CisTransClockwise
        | ChiralClass::CisTransCounterclockwise
        | ChiralClass::Clockwise
        | ChiralClass::Counterclockwise => {
            if neighbor_orders_comparable(old_order, new_order)
                && has_inversion(old_order, new_order)
            {
                class.invert()
            } else {
                class
            }
        }
        other => other,
    }
}

/// True when both listings hold the same neighbor set with matching length, so a
/// permutation parity between them is well defined.
fn neighbor_orders_comparable(old_order: &[usize], new_order: &[usize]) -> bool {
    if old_order.len() != new_order.len() || old_order.is_empty() {
        return false;
    }
    old_order.iter().all(|a| new_order.contains(a))
}

/// Determines if there is an inversion in the ordering of neighbors between two permutations
/// by counting the number of inversions (pairs of elements out of order).
///
/// # Arguments
/// * `old_neighbors` - Original ordering of neighbors
/// * `new_neighbors` - New ordering of neighbors
///
/// # Returns
/// `true` if there is an odd number of inversions (indicating a parity change),
/// `false` otherwise
fn has_inversion(old_neighbors: &[usize], new_neighbors: &[usize]) -> bool {
    let mut position_map = vec![0; old_neighbors.len()];
    for (new_pos, &value) in new_neighbors.iter().enumerate() {
        let old_pos = old_neighbors.iter().position(|&x| x == value).unwrap();
        position_map[old_pos] = new_pos;
    }

    let mut inversions = 0;
    for i in 0..position_map.len() {
        for j in i + 1..position_map.len() {
            if position_map[i] > position_map[j] {
                inversions += 1;
            }
        }
    }

    // Odd number of inversions indicates a parity change
    inversions % 2 == 1
}

/// Compares two strings and returns a formatted comparison using ANSI color codes.
/// Matching characters are shown in green, differences in red.
///
/// # Arguments
/// * `s1` - First string to compare
/// * `s2` - Second string to compare
///
/// # Returns
/// String containing a formatted comparison with colored differences
///
/// # Example
/// ```
/// use oxidane::graph_algorithms::mol_canonization::compare_strings_formatted;
/// let comparison = compare_strings_formatted("hello", "hallo");
/// // Both strings, one per line, with the differing characters in red and the
/// // matching ones in green.
/// assert_eq!(comparison.lines().count(), 2);
/// assert!(comparison.contains("\u{1b}[31m"), "no red for the differing characters");
/// assert!(comparison.contains("\u{1b}[32m"), "no green for the matching ones");
/// ```
pub fn compare_strings_formatted(s1: &str, s2: &str) -> String {
    const GREEN: &str = "\x1b[32m";
    const RED: &str = "\x1b[31m";
    const RESET: &str = "\x1b[0m";

    let mut result = String::new();
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let max_len = s1.len().max(s2.len());

    // Format first string with colors
    for i in 0..max_len {
        match (s1_chars.get(i), s2_chars.get(i)) {
            (Some(&c1), Some(&c2)) if c1 == c2 => {
                result.push_str(GREEN);
                result.push(c1);
                result.push_str(RESET);
            }
            (Some(&c1), _) => {
                result.push_str(RED);
                result.push(c1);
                result.push_str(RESET);
            }
            (None, _) => break,
        }
    }
    result.push('\n');

    // Format second string with colors
    for i in 0..max_len {
        match (s1_chars.get(i), s2_chars.get(i)) {
            (Some(&c1), Some(&c2)) if c1 == c2 => {
                result.push_str(GREEN);
                result.push(c2);
                result.push_str(RESET);
            }
            (_, Some(&c2)) => {
                result.push_str(RED);
                result.push(c2);
                result.push_str(RESET);
            }
            (_, None) => break,
        }
    }

    result
}

/// Canonicalizes and sorts a SMILES string that may contain multiple molecules separated by dots.
///
/// # Arguments
/// * `smiles` - A SMILES string potentially containing multiple molecules separated by dots
///
/// # Returns
/// A canonical SMILES string with the individual molecules sorted and recombined
///
/// # Example
/// ```
/// use oxidane::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;
/// let smiles = "CC=CC.C(C=CC)";
/// let canonical_smiles = canonicalize_and_sort_smiles(smiles, false).unwrap();
/// assert_eq!(canonical_smiles, "CC=CC.CC=CC");
/// ```
pub fn canonicalize_and_sort_smiles(
    smiles: &str,
    _with_sorting: bool,
) -> Result<String, SMILESParseError> {
    // Parsed with the default options, so aromaticity is already perceived;
    // `canonize_mol` would do it a second time.
    let mut mols = MolecularGraph::from_smiles(smiles)?;

    // Almost every SMILES is a single fragment, and for those the sort has
    // nothing to order and the join nothing to join -- both of which would
    // otherwise allocate a Vec and a second copy of the whole string.
    if mols.len() == 1 {
        return Ok(canonize_perceived(mols.remove(0)).to_smiles());
    }

    let mut canonical_smiles: Vec<String> = mols
        .into_iter()
        .map(|mol| canonize_perceived(mol).to_smiles())
        .collect();

    canonical_smiles.sort();

    // Recombine the sorted canonical SMILES strings
    Ok(canonical_smiles.join("."))
}

/// Updates chirality for an octahedral center whose SMILES neighbor listing
/// order changed from `old_neighbors` to `new_neighbors`.
fn update_oh_chirality(
    config: u8,
    old_neighbors: &[usize],
    new_neighbors: &[usize],
) -> ChiralClass {
    match remap_oh_number(config, old_neighbors, new_neighbors) {
        Some(new_config) => ChiralClass::OH(new_config),
        None => ChiralClass::OH(config),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_algorithms::mol_canonization::canonicalize_and_sort_smiles;

    use rayon::prelude::*;
    use std::fs::File;

    use std::io::Write;
    use std::io::{BufRead, BufReader};
    use std::sync::Mutex;
    use std::time::Instant;

    // #[test]
    // fn test_mol_canonization() {
    //     let mol = Molecule3D::from_sdf("tests/cis_2_butene.sdf").unwrap()[0].clone();
    //     let labeling = mol_canonization(&mol);
    //     println!("{:?}", labeling);
    // }

    #[test]
    fn test_canonical_smiles() {
        for (a, b) in [
            ("c1c(C)cccc1", "c1ccc(C)cc1"),
            ("CC(=O)O", "OC(=O)C"),
            ("C1=CC=CC=C1", "c1ccccc1"),
            ("CCO", "OCC"),
        ] {
            let canon_a = canonicalize_and_sort_smiles(a, false).unwrap();
            let canon_b = canonicalize_and_sort_smiles(b, false).unwrap();
            assert_eq!(
                canon_a, canon_b,
                "Writings {a} and {b} must produce identical canonical SMILES"
            );
        }
    }

    #[test]
    fn test_oh_chirality_equivalence() {
        let equivalent_smiles = vec![
            // "C[Co@](F)(Cl)(Br)(I)S",
            // "F[Co@@](S)(I)(C)(Cl)Br",
            "S[Co@OH5](F)(I)(Cl)(C)Br",
            "Br[Co@OH9](C)(S)(Cl)(F)I",
            "Br[Co@OH12](Cl)(I)(F)(S)C",
            "Cl[Co@OH15](C)(Br)(F)(I)S",
            "Cl[Co@OH19](C)(I)(F)(S)Br",
            "I[Co@OH27](Cl)(Br)(F)(S)C",
        ];

        let first_canonical = canonicalize_and_sort_smiles(equivalent_smiles[0], false).unwrap();

        for smiles in equivalent_smiles.iter().skip(1) {
            let canonical = canonicalize_and_sort_smiles(smiles, false).unwrap();
            assert_eq!(
                first_canonical, canonical,
                "Canonicalization failed for SMILES:\n{}\nExpected: {}\nGot: {}",
                smiles, first_canonical, canonical
            );
        }

        // Test specific OH number transformations (expected values verified
        // against RDKit canonical SMILES equivalence)
        let test_cases = vec![
            // (old_neighbors, new_neighbors, old_oh, expected_oh)
            (
                vec![0, 1, 2, 3, 4, 5], // C,F,Cl,Br,I,S
                vec![1, 5, 3, 0, 2, 4], // F,S,Br,C,Cl,I
                1,                      // Original OH number
                21,                     // Expected OH number after permutation
            ),
            (
                vec![1, 5, 4, 2, 0, 3], // F,S,I,Cl,C,Br
                vec![4, 0, 3, 5, 2, 1], // I,C,Br,S,Cl,F
                9,                      // Original OH number
                7,                      // Expected OH number after permutation
            ),
        ];

        for (old_neighbors, new_neighbors, old_oh, expected_oh) in test_cases {
            let result = update_oh_chirality(old_oh, &old_neighbors, &new_neighbors);
            assert_eq!(
                result,
                ChiralClass::OH(expected_oh),
                "OH transformation failed for {:?} -> {:?}",
                old_neighbors,
                new_neighbors
            );
        }
    }

    #[test]
    fn test_oh_chirality_permutations() {
        // Test all possible permutations of 6 neighbors for a specific OH class
        let original_neighbors = vec![0, 1, 2, 3, 4, 5]; // C,F,Cl,Br,I,S
        let original_oh = 1; // Starting with OH1

        // Expected values verified against RDKit canonical SMILES equivalence.
        // Note that reordering the neighbor listing generally changes the OH
        // number even though the molecule (and thus its stereochemistry) is
        // unchanged; only the identity listing keeps OH1.
        let expected = vec![
            (vec![0, 1, 2, 3, 4, 5], 1),  // Identity
            (vec![0, 2, 3, 4, 5, 1], 3),  // Cyclic shift of positions 1-5
            (vec![0, 3, 4, 5, 1, 2], 6),  // Double cyclic shift
            (vec![0, 4, 3, 2, 1, 5], 2),  // Equatorial reversal = mirror
            (vec![0, 5, 4, 3, 2, 1], 30), // Reversal of positions 1-5
        ];

        for (new_neighbors, expected_oh) in expected {
            let result = update_oh_chirality(original_oh, &original_neighbors, &new_neighbors);
            assert_eq!(
                result,
                ChiralClass::OH(expected_oh),
                "Wrong OH number for permutation {:?}",
                new_neighbors
            );
        }
    }

    #[test]
    fn test_oh_axis_changes() {
        // Test axis direction changes
        let original_neighbors = vec![0, 1, 2, 3, 4, 5]; // Starting with AtoF axis
        let original_oh = 1;

        // The listed atom opposite the head (atom 5, the original tail) sits at
        // position 4 in all four cases, so the axis becomes AtoE; the shape and
        // winding differ. Expected values verified against RDKit.
        let axis_changes = vec![
            (vec![0, 1, 2, 3, 5, 4], 3),  // U shape (OH3 = U, AtoE, @)
            (vec![0, 1, 2, 4, 5, 3], 5),  // Z shape (OH5 = Z, AtoE, @)
            (vec![0, 1, 3, 4, 5, 2], 11), // 4 shape (OH11 = 4, AtoE, @)
            (vec![0, 2, 3, 4, 5, 1], 3),  // U shape again
        ];

        for (new_neighbors, expected_oh) in axis_changes {
            let result = update_oh_chirality(original_oh, &original_neighbors, &new_neighbors);
            assert_eq!(
                result,
                ChiralClass::OH(expected_oh),
                "Incorrect OH number for axis change {:?}",
                new_neighbors
            );
        }
    }

    #[test]
    fn test_oh_chirality_unchanged_axis() {
        // Test case where the axis direction doesn't change
        // We'll use a permutation that keeps the same axis but changes the shape

        // Original neighbors: 0,1,2,3,4,5 with axis 0-5 (AtoF)
        let original_neighbors = vec![0, 1, 2, 3, 4, 5];
        let original_oh = 1; // U shape, AtoF axis, Counterclockwise

        // New neighbors with same axis (0-5) but two equatorial atoms swapped:
        // this mirrors the equatorial winding, giving OH2 (U, AtoF, clockwise).
        // Verified against RDKit.
        let new_neighbors = vec![0, 3, 2, 1, 4, 5];

        let result = update_oh_chirality(original_oh, &original_neighbors, &new_neighbors);

        assert_eq!(
            result,
            ChiralClass::OH(2),
            "Swapping two equatorial atoms should invert the winding"
        );
    }

    #[test]
    fn test_oh_chirality_no_changes() {
        // Test case where neither axis, shape, nor winding changes

        // Original neighbors: 0,1,2,3,4,5 with axis 0-5 (AtoF)
        let original_neighbors = vec![0, 1, 2, 3, 4, 5];
        let original_oh = 1; // U shape, AtoF axis, Counterclockwise

        // New neighbors with same configuration (identity permutation)
        let new_neighbors = vec![0, 1, 2, 3, 4, 5];

        let result = update_oh_chirality(original_oh, &original_neighbors, &new_neighbors);

        // The result should be the same as the original
        assert_eq!(
            result,
            ChiralClass::OH(original_oh),
            "Chirality should not change with identity permutation"
        );
    }

    // #[test]
    // fn test_canonization_without_sorting() {
    //     // Test with simple molecules
    //     let test_smiles = vec![
    //         "C1=CC=CC=C1",
    //         "CC",
    //         "CCC",
    //         "CC(C)C",
    //         "c1ccccc1",
    //         "C1CC1",
    //         "C(=O)O",
    //         "C1CCCCC1",
    //         "C1CCNCC1",
    //         "CC(=O)OC",
    //         "c1ccccc1C(=O)O",
    //     ];

    //     for smiles in test_smiles {
    //         let mol = MolecularGraph::from_smiles_single(smiles).unwrap();

    //         // Get canonical forms using both methods
    //         let canonical_with_sorting = mol_canonization(&mol);
    //         let canonical_without_sorting = mol_canonization_without_sorting(&mol);

    //         // Convert both to SMILES for comparison
    //         let smiles_with_sorting = canonical_with_sorting.to_smiles();
    //         let smiles_without_sorting = canonical_without_sorting.to_smiles();

    //         // They should produce the same canonical SMILES
    //         assert_eq!(
    //             smiles_with_sorting, smiles_without_sorting,
    //             "Canonization mismatch for {}: with sorting '{}', without sorting '{}'",
    //             smiles, smiles_with_sorting, smiles_without_sorting
    //         );
    //     }
    // }

    #[test]
    fn test_oh_chirality_head_same_tail_moved() {
        // Test case where the head atom stays the same but the tail atom moves

        // Original neighbors: 0,1,2,3,4,5 with axis 0-5 (AtoF)
        let original_neighbors = vec![0, 1, 2, 3, 4, 5];
        let original_oh = 1; // U shape, AtoF axis, Counterclockwise

        // Same head atom (0) but tail atom (5) moved to position 3, so the
        // axis becomes AtoD; the equatorial listing order becomes Z-shaped.
        // OH7 = Z, AtoD, anticlockwise. Verified against RDKit.
        let new_neighbors = vec![0, 1, 2, 5, 4, 3];

        let result = update_oh_chirality(original_oh, &original_neighbors, &new_neighbors);
        assert_eq!(
            result,
            ChiralClass::OH(7),
            "Tail moving to position 3 should give OH7"
        );

        let (_, axis_dir, _) = oh_config_from_number(7);
        assert_eq!(
            axis_dir,
            AxisDirection::AtoD,
            "Axis direction should change to AtoD"
        );

        // Tail atom (5) moved to position 1: axis becomes AtoB and the
        // equatorial listing stays U-shaped anticlockwise. OH25 = U, AtoB, @.
        // Verified against RDKit.
        let new_neighbors2 = vec![0, 5, 2, 3, 4, 1];

        let result2 = update_oh_chirality(original_oh, &original_neighbors, &new_neighbors2);
        assert_eq!(
            result2,
            ChiralClass::OH(25),
            "Tail moving to position 1 should give OH25"
        );

        let (_, axis_dir2, _) = oh_config_from_number(25);
        assert_eq!(
            axis_dir2,
            AxisDirection::AtoB,
            "Axis direction should change to AtoB"
        );
    }

    #[test]
    fn test_oh_chirality_tail_becomes_head() {
        // Test case where the tail atom becomes the new head

        // Original neighbors: 0,1,2,3,4,5 with axis 0-5 (AtoF)
        let original_neighbors = vec![0, 1, 2, 3, 4, 5];
        let original_oh = 1; // U shape, AtoF axis, Counterclockwise

        // Tail atom (5) becomes the head: viewing direction flips, so the
        // equatorial winding appears reversed. OH2 = U, AtoF, clockwise.
        // Verified against RDKit.
        let new_neighbors = vec![5, 1, 2, 3, 4, 0];

        let result = update_oh_chirality(original_oh, &original_neighbors, &new_neighbors);
        assert_eq!(
            result,
            ChiralClass::OH(2),
            "Winding should be inverted when tail becomes head"
        );

        // Complete reversal of the neighbor listing is a 180-degree rotation of
        // the octahedron: the stereo descriptor is unchanged. Verified against
        // RDKit.
        let new_neighbors2 = vec![5, 4, 3, 2, 1, 0];

        let result2 = update_oh_chirality(original_oh, &original_neighbors, &new_neighbors2);
        assert_eq!(
            result2,
            ChiralClass::OH(1),
            "Complete reversal should preserve the OH class"
        );
    }
    //
    /// Stress test over `tests/smiles_tree_permutations.txt`: one molecule per
    /// line, each with ~85 RDKit-verified-equivalent SMILES writings. Every
    /// writing on a line must canonicalize identically to the line's first.
    ///
    /// The checked-in corpus is 1000 molecules (4MB, ~85k comparisons), a
    /// sample of the full 100k-molecule corpus (425MB, too large for GitHub or
    /// crates.io). Drop a larger file in at the same path to run against it.
    ///
    /// `#[ignore]`d even at 1000 molecules, since it is a benchmark rather
    /// than a regression test; run it with
    /// `cargo test --release benchmark_canonization_from_file -- --ignored --nocapture`.
    ///
    /// As of 2026-07-13 the full 100k corpus has exactly 2 known-failing writings out
    /// of ~8.4M comparisons (down from 10: the fused polycyclic aromatic
    /// sulfonium mismatches were fixed - see `handle_sulfur_aromaticity`'s
    /// tricoordinate-sulfonium electron count and `find_aromatic_cycles`'s
    /// ring-growth `visited` reset). The remaining 2 are an aromatic ring
    /// containing an explicit triple bond (`c#c`), which is likely bad
    /// ground truth rather than a real defect - RDKit itself reports those
    /// ring atoms as both `SP`-hybridized and aromatic, which is
    /// self-contradictory; see `known_gap_aromatic_ring_with_triple_bond` in
    /// `src/tests/canonization_stress_tests.rs`. The threshold below allows
    /// that documented debt but fails on any new regression.
    #[test]
    #[ignore]
    fn benchmark_canonization_from_file() {
        const KNOWN_ACCEPTABLE_FAILURES: u32 = 2;

        let smiles_file_path = "tests/smiles_tree_permutations.txt";

        let limit = 100000;

        let file = File::open(smiles_file_path)
            .unwrap_or_else(|_| panic!("Failed to open SMILES file: {}", smiles_file_path));
        let reader = BufReader::new(file);
        let lines = reader
            .lines()
            .map_while(Result::ok)
            .take(limit)
            .collect::<Vec<_>>();
        let wrong_smiles_file = Mutex::new(File::create("tests/wrong_smiles.txt").unwrap());

        // Record the start time
        let start_time = Instant::now();
        let _pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let (total_smiles, correct_smiles) = lines
          .par_iter()
          .map(|line| {
              let mut iterator = line.split_whitespace();

              let comparison_smiles = match iterator.next() {
                  Some(smiles) => smiles,
                  None => return (0, 0),
              };

              let canonical_comparison_smiles = canonicalize_and_sort_smiles(comparison_smiles, true).unwrap();

              let mut local_total = 0;
              let mut local_correct = 0;

              for smiles in iterator//.take(limit)
              {
                  local_total += 1;
                  //println!("Canonizing {:?}\n", smiles);

                  let canonical_smiles = canonicalize_and_sort_smiles(smiles, true).unwrap();

                  if canonical_comparison_smiles == canonical_smiles {
                      local_correct += 1;
                  }
                   else {
                      wrong_smiles_file
                          .lock()
                          .unwrap()
                          .write_all(
                              format!(
                                  "COMPARISON,CanonicalComparison, SMILES, Canonical\n{}\n{}\n{}\n{}",
                                  comparison_smiles, canonical_comparison_smiles, smiles, canonical_smiles
                              )
                              .as_bytes(),
                          )
                          .unwrap();
                   }
               }
            //    println!("Canonized {} smiles in {:?}", local_total, start_time.elapsed());

              (local_total, local_correct)
          })
          .fold(
              || (0, 0),
              |acc, counts| (acc.0 + counts.0, acc.1 + counts.1),
          )
          .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

        // Record the end time
        let duration = start_time.elapsed();

        let success_rate = if total_smiles > 0 {
            (correct_smiles as f64 / total_smiles as f64) * 100.0
        } else {
            0.0
        };

        println!(
            "=== Benchmark Results ===\n
                              Total SMILES processed: {}\n
                              Correct SMILES: {}\n
                              Success rate: {:.2}%\n
                              Total time for canonization: {:?}\n
                              Time per smile (Thread adjusted): {:?}",
            total_smiles,
            correct_smiles,
            success_rate,
            duration,
            8 * duration / (total_smiles)
        );

        let wrong = total_smiles - correct_smiles;
        assert!(
            wrong <= KNOWN_ACCEPTABLE_FAILURES,
            "{} canonization mismatches found, only {} are known/accepted \
             (see tests/wrong_smiles.txt for details); this is a regression",
            wrong,
            KNOWN_ACCEPTABLE_FAILURES
        );
    }
}
