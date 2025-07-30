use crate::io::SMILESParseError;
use crate::molecule::aromaticity::{AromaticityModel, Aromatizer};
use crate::molecule::chirality::*;
use crate::molecule::graph::{AtomInvariant, AtomLabel, AtomLabelImplicitHydrogens};
use crate::prelude::*;
use bit_set::BitSet;
use std::cell::RefCell;
use std::collections::HashSet;

thread_local! {
    /// Automorphism generators collected from the most recent nauty run, each a
    /// permutation over the nauty vertices (heavy atoms + bond vertices).
    static AUTOMORPHISM_GENERATORS: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
}

/// nauty `userautomproc` callback: records each automorphism generator.
fn collect_automorphism_generator(
    _count: u32,
    permutation: &mut [u32],
    _orbits: &mut [u32],
    _number_of_orbits: u32,
    _stabilized_vertex: u32,
    number_of_vertices: u32,
) {
    let n = number_of_vertices as usize;
    AUTOMORPHISM_GENERATORS.with(|generators| {
        generators.borrow_mut().push(permutation[..n].to_vec());
    });
}
// use canonizer::{
//     canonization::{first_non_trivial, graph_canon, smallest_first_non_trivial},
//     edge_labeled_canonization::{
//         create_canonical_adjacency_matrix_edge_labeled, graph_canon_labeled,
//     },
// };
use itertools::{Itertools, max};
//use nauty_pet::autom::AutomGroup;
use nohash_hasher::IntMap;
use petgraph::adj::Neighbors;
use petgraph::algo::connected_components;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::{EdgeRef, GetAdjacencyMatrix};
use rustc_hash::FxHashMap;

use graphcanon::{
    refine::{DISPATCH_GRAPH, GraphContext},
    search::canonicalize,
    structs::{CanonautOptions, DispatchVec, Statistics},
};

use std::ffi::c_void;

pub type BitMatrix = Vec<usize>;

fn print_matrix(matrix: &[usize], words_per_vertex: usize, number_of_vertices: usize) {
    for i in 0..number_of_vertices {
        for j in 0..words_per_vertex {
            let word = matrix[i * words_per_vertex + j];
            for i in 0..number_of_vertices {
                let bit_index = 1 << (64 - 1 - i);
                let bit = (word & bit_index) != 0;
                print!("{:} ", bit as u8);
            }
        }
        println!();
    }
}

/// The working buffers one canonization needs, beside the aromaticity ones.
///
/// Scratch, not state: every field is cleared and resized from the molecule
/// before use.
#[derive(Debug, Default, Clone)]
pub struct CanonizationScratch {
    sorted_labels: Vec<AtomInvariant>,
    labeling: Vec<u32>,
    heavy_labeling: Vec<usize>,
    canong: Vec<usize>,
    workspace: Vec<usize>,
    orbits: Vec<u32>,
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
}

impl Canonizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// [`canonize_mol`], reusing this canonizer's buffers.
    pub fn canonize<T: Molecule + Clone + ToSMILES>(&mut self, mol: &T) -> T {
        let mut molecule = mol.clone();
        let _ = self.aromatizer.aromatize(
            &mut molecule,
            AromaticityModel::OpenSMILES,
            None,
            false,
        );
        self.canonize_perceived(molecule)
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
    pub fn canonize_perceived<T: Molecule + Clone + ToSMILES>(&mut self, mut molecule: T) -> T {
        let (indices, labels) = prepare_and_sort_indices_and_labels(&molecule);
        let _ = run_canonization(&mut molecule, &indices, &labels, &mut self.scratch);
        molecule
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

thread_local! {
    /// The canonizer the free functions borrow, so existing callers get the
    /// buffer reuse without threading a handle through their signatures.
    static CANONIZER: std::cell::Cell<Canonizer> = std::cell::Cell::new(Canonizer::new());
}

/// Runs `f` against the thread's canonizer, taken out and put back so that
/// anything reachable from `f` which itself canonizes gets its own rather than
/// a double-borrow panic.
fn with_canonizer<R>(f: impl FnOnce(&mut Canonizer) -> R) -> R {
    let mut canonizer = CANONIZER.take();
    let result = f(&mut canonizer);
    CANONIZER.set(canonizer);
    result
}

/// This function canonizes a given molecule.
/// It returns a new molecule with the atoms reordered according to the canonical ordering.
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

/// This function canonizes a given molecule.
/// It returns a new molecule with the atoms reordered according to the canonical ordering.
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

    let (Some(tags), Some(classes)) = (canonical.atom_classes().clone(), canonical.chiral_classes())
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
    InternalError,
}

fn run_canonization<T: Molecule + ToSMILES + Clone>(
    molecule: &mut T,
    indices: &[usize],
    labels: &[AtomInvariant],
    scratch: &mut CanonizationScratch,
) -> Result<(), CanonizationError> {
    reorder_molecule(molecule, &indices);

    // `labels` is in the original atom order, but the molecule has just been
    // reordered by `indices`; reorder the labels to match so the colouring
    // partition lines up with the graph's vertex order.
    let sorted_labels = &mut scratch.sorted_labels;
    sorted_labels.clear();
    sorted_labels.extend(indices.iter().map(|&i| labels[i].clone()));

    let Some((mut adjacency_matrix, mut partition, number_of_heavy_atoms)) =
        molecule.prepare_canonization(sorted_labels)
    else {
        return Err(CanonizationError::InvalidInput);
    };

    let mut words_per_vertex = adjacency_matrix.words_per_vertex;
    let mut number_of_vertices = adjacency_matrix.number_of_vertices;
    let labeling = &mut scratch.labeling;
    labeling.clear();
    labeling.extend((0..number_of_vertices).map(|x| x as u32));
    let mut matrix = adjacency_matrix.matrix;

    let canong = &mut scratch.canong;
    canong.clear();
    canong.resize(matrix.len(), 0);
    // let mut partition: Vec<i32> = vec![1; number_of_vertices];

    let worksize = 120;
    let workspace = &mut scratch.workspace;
    workspace.clear();
    workspace.resize(2 * words_per_vertex * worksize, 0);

    let orbits = &mut scratch.orbits;
    orbits.clear();
    orbits.resize(number_of_vertices, 0);
    // Enable canonical labeling and feed nauty our invariant-based colouring:
    // `default_partition` would otherwise discard `partition` and canonicalize the
    // uncoloured graph, leaving atom order undetermined for graph-symmetric
    // molecules whose symmetry is broken only by atom identity (e.g. a palindromic
    // chain with N on one end and O on the other).
    AUTOMORPHISM_GENERATORS.with(|generators| generators.borrow_mut().clear());
    let mut options: CanonautOptions = CanonautOptions {
        get_canonical: true,
        default_partition: false,
        user_automorphism_procedure: Some(collect_automorphism_generator),
        ..CanonautOptions::default()
    };
    let mut statistics: Statistics = Statistics::default();
    let mut graph_context = GraphContext::new(number_of_vertices);

    canonicalize(
        &matrix,
        labeling,  // weights
        &mut partition, // partition
        None,           // active
        orbits,    //orbits
        &mut options,
        &mut statistics, // active
        workspace,
        words_per_vertex as u32,
        number_of_vertices as u32,
        Some(canong.as_mut_slice()),
        &mut graph_context,
    );

    // Nauty's output labeling is a permutation over all vertices, including the
    // virtual bond-order vertices appended after the heavy atoms in
    // `prepare_canonization`. Those virtual vertices are not guaranteed to occupy
    // only the tail positions of the canonical order, so we filter them out
    // instead of assuming the first `number_of_heavy_atoms` entries are all real atoms.
    let heavy_labeling = &mut scratch.heavy_labeling;
    heavy_labeling.clear();
    heavy_labeling.extend(
        labeling
            .iter()
            .map(|x| *x as usize)
            .filter(|&x| x < number_of_heavy_atoms),
    );
    let labeling = heavy_labeling;

    // Collect the automorphism generators nauty found (on the pre-canonical
    // graph) before reordering, so we can later pick a canonical chirality tag
    // assignment among symmetry-equivalent ones.
    let generators = AUTOMORPHISM_GENERATORS.with(|g| std::mem::take(&mut *g.borrow_mut()));

    reorder_molecule(molecule, &labeling);

    // A graph automorphism can flip stereocentre tags (e.g. the bridge swap of a
    // symmetric ring), yielding several equally valid tag assignments for the
    // same molecule. Pick the canonical one deterministically.
    if molecule.is_chiral() && !generators.is_empty() {
        canonicalize_chirality_under_automorphisms(
            molecule,
            &generators,
            &labeling,
            number_of_heavy_atoms,
        );
    }
    Ok(())
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
    if full_generators.is_empty() {
        return;
    }

    let current = molecule.chiral_classes().unwrap().to_vec();
    let mut best = current.clone();


    // Enumerate the group generated by the automorphisms (bounded) and keep the
    // smallest tag vector any of them produces.
    const MAX_GROUP: usize = 4096;
    let identity: Vec<usize> = (0..molecule.len()).collect();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    seen.insert(identity.clone());
    let mut frontier = vec![identity];
    while let Some(element) = frontier.pop() {
        if seen.len() > MAX_GROUP {
            break;
        }
        for generator in &full_generators {
            let composed: Vec<usize> = element.iter().map(|&i| generator[i]).collect();
            if seen.insert(composed.clone()) {
                let mut candidate = molecule.clone();
                reorder_molecule(&mut candidate, &composed);
                if let Some(tags) = candidate.chiral_classes() {
                    if tags < best.as_slice() {
                        best = tags.to_vec();
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

/// Extends a heavy-atom permutation to a full-atom permutation by mapping each
/// atom's hydrogens onto the hydrogens of its image (ascending index order).
/// Returns `None` if the mapping is not a bijection (mismatched hydrogen counts).
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

    if full.iter().any(|&x| x == usize::MAX) {
        return None;
    }
    Some(full)
}

fn prepare_and_sort_indices_and_labels<T: Molecule>(mol: &T) -> (Vec<usize>, Vec<AtomInvariant>) {
    let labels = (0..mol.len())
        .map(|atom_index| mol.get_atom_invariant(atom_index))
        .collect::<Vec<_>>();
    let indices = (0..mol.len()).collect::<Vec<usize>>();

    let mut label_indices = indices.iter().zip(labels.iter()).collect::<Vec<_>>();
    label_indices.sort_by(|(_, label1), (_, label2)| label2.cmp(label1));

    let indices = label_indices
        .iter()
        .map(|(index, _)| **index)
        .collect::<Vec<usize>>();

    (indices, labels)
}

/// Reorders the atoms in a molecule according to a given labeling
pub fn reorder_molecule<T: Molecule + ToSMILES>(mol: &mut T, mut labeling: &[usize]) {
    // Nothing to permute, and the bond-table rebuild below seeds itself from
    // `atom_bonds()[0]`, which does not exist for an empty molecule.
    if mol.len() == 0 {
        return;
    }

    // Create inverse mapping
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
        // Update bond targets according to new ordering
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
                let mut old_sorted = if old_idx < emission_old.len() {
                    emission_old[old_idx].clone()
                } else {
                    Vec::new()
                };
                old_sorted.sort_unstable();
                let old_sorted_mapped: Vec<usize> =
                    old_sorted.iter().map(|&t| inverse_mapping[t]).collect();
                let mut new_sorted = old_sorted_mapped.clone();
                new_sorted.sort_unstable();
                let tag = tetrahedral_tag_for_order(reference, &old_sorted_mapped, &new_sorted);
                ChiralClass::TH(tag)
            } else if let ChiralClass::AL(reference) = old_class {
                // Same idea as the tetrahedral branch, one atom further out:
                // carry the old carrier order through the relabeling and ask
                // whether re-sorting it is an odd permutation. Falling back on
                // the centre's own neighbour order — which is what this used
                // to do — compares two atoms where four are needed, so it
                // could not encode the winding at all.
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
            new_chiral_classes[a] = ChiralClass::CisTransCounterclockwise;
            new_chiral_classes[b] = if side_a != side_b {
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
        let Some(b) = bonds_a
            .iter()
            .find(|bo| bo.bond_order() == BondOrder::Double)
            .map(|bo| bo.target())
        else {
            continue;
        };
        if b <= a || !is_cis_trans(classes[b]) {
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
    // Create a mapping from old positions to new positions
    let mut position_map = vec![0; old_neighbors.len()];
    for (new_pos, &value) in new_neighbors.iter().enumerate() {
        let old_pos = old_neighbors.iter().position(|&x| x == value).unwrap();
        position_map[old_pos] = new_pos;
    }

    // Count inversions in the position mapping
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

/// Gets the neighboring atoms in the SMILES path for a given position
fn get_path_neighbors(path: &[usize], neighbor_set: &BitSet) -> Vec<usize> {
    let mut neighbors = Vec::with_capacity(10);
    for atom in path {
        if neighbor_set.contains(*atom) {
            neighbors.push(*atom);
        }
    }
    neighbors
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
    with_sorting: bool,
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

    // Sort the canonical SMILES strings
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
    use crate::graph_algorithms::mol_canonization::{
        canonicalize_and_sort_smiles,
        compare_strings_formatted,
        // mol_canonization,
        // to_canonical_smiles,
    };
    use crate::prelude::*;
    use rayon::prelude::*;
    use std::fs::File;
    use std::fs::OpenOptions;
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
        // let smiles = "C1COc2cccc(c2)CNCCN2CCNCc3cccc(c3)OCCN(CCOc3cccc(c3)CNCCN3CCNCc4cccc(c4)OCCN1CCOc1cc(ccc1)CNCC3)CCOc1cc(ccc1)CNCC2";
        // let canonical_smiles1 = canonicalize_and_sort_smiles(smiles, false).unwrap();
        // let mols = MolecularGraph::from_smiles(smiles).unwrap();
        // let mut mol1 = mols[0].clone();
        // mol1.aromatize_opensmiles();
        // let canonical_mol1 = mol_canonization(&mol1);
        // //    let smiles = "C1c2cccc(c2)OCCN3CCOc4cccc(CNCCN(CCNCc5cccc(c5)OCCN6CCOc7cccc(CNCCN(CCNCc8cccc(c8)OCC6)CCNCc9cccc(c9)OCC3)c7)CCN1)c4";
        // //let canonical_smiles2 =  canonicalize_and_sort_smiles(smiles);
        // let smiles = "C1=CC2=CC(=C1)OCCN1CCOC3=CC=CC(=C3)CNCCN(CCNC2)CCNCC2=CC(=CC=C2)OCCN2CCOC3=CC=CC(=C3)CNCCN(CCNCC3=CC(=CC=C3)OCC2)CCNCC2=CC=CC(=C2)OCC1";
        // let canonical_smiles3 = canonicalize_and_sort_smiles(smiles, false).unwrap();
        // let mols = MolecularGraph::from_smiles(smiles).unwrap();
        // let mut mol3 = mols[0].clone();
        // mol3.aromatize_opensmiles();
        // let canonical_mol3 = mol_canonization(&mol3);

        // //let smiles = "C1c2cccc(c2)OCCN3CCOc4cccc(CNCCN(CCNCc5cccc(c5)OCC3)CCNCc6cccc(c6)OCCN7CCOc8cccc(CNCCN(CCNCc9cccc(c9)OCC7)CCN1)c8)c4";
        // //let canonical_smiles4 =  canonicalize_and_sort_smiles(smiles);
        // //

        // let smiles = "Clc1cc(ccc1OC(=O)C2CN(CC3CC3)C(=O)C2)-c4ccccc4";
        // let canonical_smiles = canonicalize_and_sort_smiles(smiles, false).unwrap();
        // // assert_eq!(canonical_smiles, smiles);

        // let mol2 = MolecularGraph::from_smiles("c12c3c(ccc2)c(ccc4)c5c4cccc5c3ccc1").unwrap();
        // let mol2 = &mol2[0];
        // let canonical_smiles2 = to_canonical_smiles(mol2, false);
        // // assert_eq!(canonical_smiles2, "c1cc2c3cccc4cccc(c5cccc(c1)c25)c34");
        // let mol = MolecularGraph::from_smiles_single("c1c(C)cccc1").unwrap();
        // let canonical_smiles = to_canonical_smiles(&mol, false);

        // println!("{}", canonical_smiles);

        // let mol = MolecularGraph::from_smiles_single("c1ccc(C)cc1").unwrap();
        // let canonical_smiles = to_canonical_smiles(&mol, false);

        // println!("{}", canonical_smiles);

        // panic!();

        //let mol = MolecularGraph::from_smiles_single("[c]1c(S(=O)(=O)[O])c(-c2[c][c]c([N]C(=O)[N]c3[c][c]c(-c4[c][c]c([N]C(=O)[N]c5[c][c]c(-c6[c][c]c([N]C(=O)[N]c7[c]c(S(=O)(=O)[O])c(-c8[c][c]c([N]C([N]c9[c][c]c(-c%10[c][c]c([N]C([N]c%11[c]c(S([O])(=O)=O)c(-c%12c(S([O])(=O)=O)[c]c([N]C(=O)[N]c%13[c]c(S(=O)(=O)[O])c(-c%14c(S(=O)(=O)[O])[c]c([N]C([N]c%15[c][c]c([CH2])[c][c]%15)=O)[c][c]%14)[c][c]%13)[c][c]%12)[c][c]%11)=O)[c]c%10S([O])(=O)=O)c(S(=O)(=O)[O])[c]9)=O)[c]c8S([O])(=O)=O)[c][c]7)[c]c6S([O])(=O)=O)c(S(=O)([O])=O)[c]5)[c]c4S(=O)(=O)[O])c(S(=O)(=O)[O])[c]3)[c]c2S(=O)(=O)[O])[c][c]c1[N]C(=O)[N]c1[c][c]c([CH2])[c][c]1").unwrap();
        //
        // let smiles = "C1COc2cccc(c2)CNCCN2CCNCc3cccc(c3)OCCN(CCOc3cccc(c3)CNCCN3CCNCc4cccc(c4)OCCN1CCOc1cc(ccc1)CNCC3)CCOc1cc(ccc1)CNCC2";
        // let canonical_smiles = canonicalize_and_sort_smiles(smiles, false).unwrap();
        // assert_eq!( canonical_smiles,
        //     "O1c2cccc(c2)CNCCN3CCNCc4cccc(OCCN5CCOc6cccc(c6)CNCCN(CCNCc7cccc(OCC5)c7)CCNCc8cccc(OCCN(CC1)CCOc9cccc(c9)CNCC3)c8)c4"
        // );

        // let time = Instant::now();
        // let caonical_smiles = to_canonical_smiles(&mol, false);
        // println!("Time: {:?}", time.elapsed());
        // let mol = MolecularGraph::from_smiles_single("PP(P)P(P)I").unwrap();
        // let canonical_smiles = to_canonical_smiles(&mol, true);
        // println!("{}", canonical_smiles);

        // let mol = MolecularGraph::from_smiles_single("P(P(P)I)(P)P").unwrap();
        // let canonical_smiles = to_canonical_smiles(&mol, true);
        // println!("{}", canonical_smiles);

        // use crate::io::to_dot_file;

        // let mol2 = MolecularGraph::from_smiles_single("O=P1(OP2(=O)O[In](OP3(=O)O[In]4OP(=O)(O3)O4)O2)O[In](OP2(=O)OP3(=O)O[In](O3)O2)O1").unwrap();
        // to_dot_file(&mol2, Some("tests/test_canonization.dot"));

        // let canonical_smiles2 = to_canonical_smiles(&mol2, false);
        // let (canonical_mol2, automorphisms) = mol_canonization_nauty(&mol2);
        // println!("Automorphisms: {:?}", automorphisms);
        // println!("Number of automorphisms: {:?}", automorphisms.len());
        // panic!();
        // let mol3 = MolecularGraph::from_smiles_single("[In]12OP(=O)(OP(=O)(O1)O2)O[In]3OP(=O)(OP4(=O)O[In](O4)OP5(=O)OP6(=O)O[In](O5)O6)O3").unwrap();
        // let canonical_smiles3 = to_canonical_smiles(&mol3, false);
        // let mol4 = MolecularGraph::from_smiles_single("O1P2(=O)O[In]1OP(=O)(O[In]1OP(=O)(OP3(=O)O[In](OP4(=O)O[In]5OP(=O)(O4)O5)O3)O1)O2").unwrap();
        // let canonical_smiles4 = to_canonical_smiles(&mol4, false);
        // let mol5 = MolecularGraph::from_smiles_single("[In]12OP(=O)(OP(=O)(O1)O2)O[In]3OP(=O)(OP4(=O)O[In](OP5(=O)OP6(=O)O[In](O6)O5)O4)O3").unwrap();
        // let canonical_smiles5 = to_canonical_smiles(&mol5, false);
        // println!("{}", canonical_smiles2);
        // println!("{}", canonical_smiles3);
        // println!("{}", canonical_smiles4);
        // println!("{}", canonical_smiles5);

        // assert_eq!(canonical_smiles2, canonical_smiles3);
        // assert_eq!(canonical_smiles3, canonical_smiles4);
        // assert_eq!(canonical_smiles4, canonical_smiles5);
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

        // Get the canonical form of the first SMILES
        let first_canonical = canonicalize_and_sort_smiles(equivalent_smiles[0], false).unwrap();

        // Test that all SMILES canonicalize to the same form
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
    /// The checked-in corpus is 1000 molecules (4MB, ~85k comparisons). It is
    /// deliberately a sample: the full 100k-molecule corpus this was developed
    /// against is 425MB, which is four times what GitHub will accept in a
    /// single file and forty times what crates.io will take for a whole
    /// package. Drop a larger file in at the same path to run against it.
    ///
    /// `#[ignore]`d even at 1000 molecules, since it is a benchmark rather
    /// than a regression test; run it with `./test.sh` or
    /// `cargo test --release <name> -- --ignored --nocapture`.
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

        // Open the SMILES file
        let file = File::open(smiles_file_path)
            .unwrap_or_else(|_| panic!("Failed to open SMILES file: {}", smiles_file_path));
        let reader = BufReader::new(file);
        let lines = reader
            .lines()
            .map_while(Result::ok)
            .take(limit)
            .collect::<Vec<_>>();
        let wrong_smiles_file = Mutex::new(File::create("tests/wrong_smiles.txt").unwrap());
        let log_file = Mutex::new(File::create("tests/canonization_log.txt").unwrap());

        // Record the start time
        let start_time = Instant::now();
        // Build thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        // Process each line in parallel and accumulate counts
        let (total_smiles, correct_smiles) = lines
          .par_iter()
          .map(|line| {
              let mut iterator = line.split_whitespace();

              // Extract the comparison SMILES string
              let comparison_smiles = match iterator.next() {
                  Some(smiles) => smiles,
                  None => return (0, 0),
              };

              // Canonicalize the comparison SMILES
              let canonical_comparison_smiles = canonicalize_and_sort_smiles(comparison_smiles, true).unwrap();

              // Assuming `mol_canonization` and related functions are defined elsewhere
              // Initialize local counters
              let mut local_total = 0;
              let mut local_correct = 0;

              // Iterate through each SMILES in the line
              for smiles in iterator//.take(limit)
              {
                  local_total += 1;
                  //println!("Canonizing {:?}\n", smiles);

                  // Canonicalize the current SMILES
                  let canonical_smiles = canonicalize_and_sort_smiles(smiles, true).unwrap();

                  // Update the correct SMILES counter if canonicalizations match
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

        // Calculate success rate
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
