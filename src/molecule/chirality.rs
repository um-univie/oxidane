use crate::{
    molecule::{
        bond::{BondOrder, BondTarget},
        molecule3d::Molecule3D,
        traits::*,
    },
    vector::Vector3D,
};

pub trait MoleculeChirality: MoleculeCore {
    /// Converts directional bonds (`/`, `\`) into per-atom cis/trans chirality
    /// classes for stereogenic double bonds.
    ///
    /// The two classes are set so that `class(a) != class(b)` holds exactly when
    /// the first single-bond neighbor of each carbon (`subs[0]`) sit on opposite
    /// sides (trans) of the double bond — the invariant that [`to_smiles`] reads
    /// back to emit directional bonds and that canonization maintains.
    ///
    /// # Examples
    /// ```
    /// use oxidane::prelude::*;
    /// let mut mol = MolecularGraph::from_smiles_single("F/C=C/F").unwrap();
    /// assert_eq!(mol.to_smiles(), "F/C=C/F"); // trans-difluoroethene
    /// ```
    fn convert_directional_bonds_to_chiral(&mut self) {
        // The written side of a directional-bonded substituent relative to its
        // carbon: `/` is Up, `\` is Down; a substituent written before its carbon
        // reads the opposite way to one written after it (atoms are indexed in
        // written order, so `carbon < sub` means the carbon is written first).
        //
        // Every mark on the carbon is read, not only the first: its two
        // substituents sit on opposite sides of the double bond, so two marks
        // placing them on the same side contradict each other and say nothing
        // about the geometry. Such an end gives `None` and the bond gets no
        // tag -- which is what RDKit does. Taking the first mark instead
        // invented a geometry PubChem's
        // `C(=C\1\C=CC...)` never stated.
        let side_of = |mol: &Self, carbon: usize| -> Option<(usize, bool)> {
            let mut first: Option<(usize, bool)> = None;
            for bond in mol.get_atom_bonds(carbon).unwrap_or_default().iter() {
                let order = bond.bond_order();
                if order == BondOrder::Up || order == BondOrder::Down {
                    let sub = bond.target();
                    let carbon_written_first = carbon < sub;
                    let up_side = if carbon_written_first {
                        order == BondOrder::Up
                    } else {
                        order == BondOrder::Down
                    };
                    match first {
                        None => first = Some((sub, up_side)),
                        Some((_, side)) if side == up_side => return None,
                        Some(_) => {}
                    }
                }
            }
            first
        };

        let first_single_neighbor = |mol: &Self, atom: usize, exclude: usize| -> Option<usize> {
            mol.get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .filter(|b| b.bond_order() != BondOrder::Double && b.target() != exclude)
                .map(|b| b.target())
                .next()
        };

        let mut assignments: Vec<(usize, ChiralClass)> = Vec::new();
        let mut to_convert: Vec<usize> = Vec::new();

        for a in 0..self.len() {
            let Some(b) = self
                .get_atom_bonds(a)
                .unwrap_or_default()
                .iter()
                .find(|bo| bo.bond_order() == BondOrder::Double)
                .map(|bo| bo.target())
            else {
                continue;
            };
            if b <= a {
                continue; // handle each double bond once
            }

            let (Some((dir_a, up_a)), Some((dir_b, up_b))) = (side_of(self, a), side_of(self, b))
            else {
                continue; // both ends need a directional bond to be stereogenic
            };
            let (Some(sub_a0), Some(sub_b0)) = (
                first_single_neighbor(self, a, b),
                first_single_neighbor(self, b, a),
            ) else {
                continue;
            };

            let side_a0 = if sub_a0 == dir_a { up_a } else { !up_a };
            let side_b0 = if sub_b0 == dir_b { up_b } else { !up_b };
            let trans = side_a0 != side_b0;

            assignments.push((a, ChiralClass::CisTransCounterclockwise));
            assignments.push((
                b,
                if trans {
                    ChiralClass::CisTransClockwise
                } else {
                    ChiralClass::CisTransCounterclockwise
                },
            ));
            to_convert.push(a);
            to_convert.push(b);
        }

        // An end whose marks contradict each other was refused above; its
        // marks say nothing, so they go back to being the single bonds they
        // stand on rather than lingering to be written out again.
        for atom in 0..self.len() {
            let marks = self
                .get_atom_bonds(atom)
                .unwrap_or_default()
                .iter()
                .filter(|bond| matches!(bond.bond_order(), BondOrder::Up | BondOrder::Down))
                .count();
            if marks >= 2 && side_of(self, atom).is_none() {
                to_convert.push(atom);
            }
        }

        for atom_idx in to_convert {
            self.convert_directional_to_single(atom_idx);
        }
        if assignments.is_empty() {
            return;
        }
        if self.chiral_classes().is_none() {
            self.initialize_chiral_classes();
        }
        let classes = self.chiral_classes_mut().unwrap();
        for (atom_idx, class) in assignments {
            classes[atom_idx] = class;
        }
    }

    /// Converts all directional bonds (up/down) connected to an atom to single bonds
    fn convert_directional_to_single(&mut self, atom_idx: usize) {
        let mut neighbors_to_change = Vec::new();
        if let Some(bonds) = self.get_atom_bonds_mut(atom_idx) {
            for bond in bonds {
                if bond.bond_order() == BondOrder::Up || bond.bond_order() == BondOrder::Down {
                    neighbors_to_change.push(bond.target());
                    bond.bond_order = BondOrder::Single;
                }
            }
        }

        for neighbor in neighbors_to_change {
            if let Some(bonds) = self.get_atom_bonds_mut(neighbor) {
                for bond in bonds {
                    if bond.bond_order() == BondOrder::Up || bond.bond_order() == BondOrder::Down {
                        bond.bond_order = BondOrder::Single;
                    }
                }
            }
        }
    }
}

/// Represents the chirality of a molecule.
///
/// The definitions of the chiral classes are based on the following source:
///
/// http://opensmiles.org/opensmiles.html
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ChiralClass {
    TH(u8), // @ 1..=2 1 = anti-clockwise, 2 = clockwise
    AL(u8), // @ 1..=2 1 = anti-clockwise, 2 = clockwise
    SP(u8), // @ 1..=3 1 = U, 2 = Z, 3 = 4
    TB(u8), // @ 1..=20
    OH(u8), // @ 1..=30
    CisTransClockwise,
    CisTransCounterclockwise,
    Clockwise, //
    Counterclockwise,
    #[default]
    None,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum ChiralType {
    CisTrans = 1,
    // The following definitions are just kept for completeness
    TH = 2, // @ 1..=2 1 = anti-clockwise, 2 = clockwise
    AL = 3, // @ 1..=2 1 = anti-clockwise, 2 = clockwise
    SP = 4, // @ 1..=3 1 = U, 2 = Z, 3 = 4
    TB = 5, // @ 1..=20
    OH = 6, // @ 1..=30
    None = 7,
}

impl ChiralClass {
    /// Total-order key: `None` sorts first, then each class by kind and number.
    fn ordering_key(&self) -> (u8, u8) {
        match self {
            ChiralClass::None => (0, 0),
            ChiralClass::TH(n) => (1, *n),
            ChiralClass::AL(n) => (2, *n),
            ChiralClass::SP(n) => (3, *n),
            ChiralClass::TB(n) => (4, *n),
            ChiralClass::OH(n) => (5, *n),
            ChiralClass::CisTransClockwise => (6, 0),
            ChiralClass::CisTransCounterclockwise => (7, 0),
            ChiralClass::Clockwise => (8, 0),
            ChiralClass::Counterclockwise => (9, 0),
        }
    }
}

impl std::cmp::PartialOrd for ChiralClass {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for ChiralClass {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ordering_key().cmp(&other.ordering_key())
    }
}

/// Parity (odd = `true`) of the permutation that reorders the sequence `from`
/// into the sequence `to`. Both must hold the same multiset of atom indices.
///
/// A tetrahedral chirality label flips whenever its neighbor listing is reordered
/// by an odd permutation, so this is the core primitive for converting a `@`/`@@`
/// tag between different neighbor orderings.
pub fn neighbor_permutation_is_odd(from: &[usize], to: &[usize]) -> bool {
    debug_assert_eq!(from.len(), to.len());
    // p[i] = position in `to` of the element at position i in `from`.
    let p: Vec<usize> = from
        .iter()
        .map(|value| {
            to.iter()
                .position(|x| x == value)
                .expect("from and to must contain the same elements")
        })
        .collect();
    let mut inversions = 0usize;
    for i in 0..p.len() {
        for j in (i + 1)..p.len() {
            if p[i] > p[j] {
                inversions += 1;
            }
        }
    }
    inversions % 2 == 1
}

/// Re-expresses a tetrahedral tag against a different neighbor ordering.
///
/// `stored` is `1` (`@`) or `2` (`@@`) defined relative to the `reference`
/// ordering of the four neighbor atom indices; the return value is the tag that
/// describes the *same* 3D arrangement relative to the `target` ordering.
/// `reference` and `target` must be permutations of each other.
pub fn tetrahedral_tag_for_order(stored: u8, reference: &[usize], target: &[usize]) -> u8 {
    if neighbor_permutation_is_odd(reference, target) {
        3 - stored
    } else {
        stored
    }
}

/// The four carriers of an allene centre, in the canonical reference ordering
/// its tag is stored against: the two ends sorted by atom index, and within
/// each end its substituents sorted by atom index.
///
/// An allene is chiral in the same way an sp3 centre is — the two ends are
/// held at right angles, so their four substituents sit at the corners of an
/// elongated tetrahedron — but the carriers are one atom further out than the
/// centre's own neighbours. A tag stored against the centre's two neighbours
/// cannot say which way four substituents wind, which is why this ordering
/// exists: it depends only on the atoms' identities, so it survives any
/// renumbering and can be recomputed at any point.
///
/// `None` when the centre is not a proper allene: it must have exactly two
/// neighbours carrying two substituents each.
pub fn index_sorted_allene_carriers<M: crate::molecule::base::MoleculeCore + ?Sized>(
    mol: &M,
    centre: usize,
) -> Option<[usize; 4]> {
    let mut ends: Vec<usize> = mol.atom_bonds()[centre]
        .iter()
        .map(|bond| bond.target())
        .collect();
    if ends.len() != 2 {
        return None;
    }
    ends.sort_unstable();

    let mut carriers = Vec::with_capacity(4);
    for &end in &ends {
        let mut substituents: Vec<usize> = mol.atom_bonds()[end]
            .iter()
            .map(|bond| bond.target())
            .filter(|&target| target != centre)
            .collect();
        substituents.sort_unstable();
        if substituents.len() != 2 {
            return None;
        }
        carriers.extend(substituents);
    }
    Some([carriers[0], carriers[1], carriers[2], carriers[3]])
}

/// The canonical reference ordering for a tetrahedral centre: its neighbor atom
/// indices sorted ascending. This ordering depends only on the identities of the
/// neighbors (not on bond insertion order), so a tag stored against it survives
/// any reordering of the centre's bond list and is trivial to update when the
/// atoms are relabeled.
pub fn index_sorted_neighbors<M: crate::molecule::base::MoleculeCore + ?Sized>(
    mol: &M,
    center: usize,
) -> Vec<usize> {
    let mut neighbors: Vec<usize> = mol.atom_bonds()[center]
        .iter()
        .map(|bond| bond.target())
        .collect();
    neighbors.sort_unstable();
    neighbors
}

/// Maps each SMILES `@OH1`..`@OH30` class to an octahedron vertex assignment.
///
/// `OH_VERTEX_TABLE[n - 1][i]` is the octahedron vertex occupied by the atom at
/// position `i` of the SMILES neighbor list. Vertex labels: 0 = +z, 1 = +x,
/// 2 = +y, 3 = -x, 4 = -y, 5 = -z, where OH1 places the neighbor list as
/// [+z, +x, +y, -x, -y, -z] (axis a-f, U shape, anticlockwise viewed from a).
/// Two assignments describe the same stereochemistry iff they differ by a
/// proper rotation of the octahedron (see [`OCT_ROTATIONS`]); the 30 rows are
/// exactly the 720/24 rotation cosets.
const OH_VERTEX_TABLE: [[u8; 6]; 30] = [
    [0, 1, 2, 3, 4, 5], // OH1
    [0, 1, 4, 3, 2, 5], // OH2
    [0, 1, 2, 3, 5, 4], // OH3
    [0, 1, 2, 4, 3, 5], // OH4
    [0, 1, 2, 4, 5, 3], // OH5
    [0, 1, 2, 5, 3, 4], // OH6
    [0, 1, 2, 5, 4, 3], // OH7
    [0, 1, 3, 2, 4, 5], // OH8
    [0, 1, 3, 2, 5, 4], // OH9
    [0, 1, 3, 4, 2, 5], // OH10
    [0, 1, 3, 4, 5, 2], // OH11
    [0, 1, 3, 5, 2, 4], // OH12
    [0, 1, 3, 5, 4, 2], // OH13
    [0, 1, 4, 2, 3, 5], // OH14
    [0, 1, 4, 2, 5, 3], // OH15
    [0, 1, 4, 3, 5, 2], // OH16
    [0, 1, 4, 5, 2, 3], // OH17
    [0, 1, 4, 5, 3, 2], // OH18
    [0, 1, 5, 2, 3, 4], // OH19
    [0, 1, 5, 2, 4, 3], // OH20
    [0, 1, 5, 3, 2, 4], // OH21
    [0, 1, 5, 3, 4, 2], // OH22
    [0, 1, 5, 4, 2, 3], // OH23
    [0, 1, 5, 4, 3, 2], // OH24
    [0, 5, 1, 2, 3, 4], // OH25
    [0, 5, 1, 2, 4, 3], // OH26
    [0, 5, 1, 3, 2, 4], // OH27
    [0, 5, 1, 3, 4, 2], // OH28
    [0, 5, 1, 4, 2, 3], // OH29
    [0, 5, 1, 4, 3, 2], // OH30
];

/// The 24 proper rotations of the octahedron as permutations of the vertex
/// labels used in [`OH_VERTEX_TABLE`] (`rot[v]` is the image of vertex `v`).
const OCT_ROTATIONS: [[u8; 6]; 24] = [
    [0, 1, 2, 3, 4, 5],
    [0, 2, 3, 4, 1, 5],
    [0, 3, 4, 1, 2, 5],
    [0, 4, 1, 2, 3, 5],
    [1, 0, 4, 5, 2, 3],
    [1, 2, 0, 4, 5, 3],
    [1, 4, 5, 2, 0, 3],
    [1, 5, 2, 0, 4, 3],
    [2, 0, 1, 5, 3, 4],
    [2, 1, 5, 3, 0, 4],
    [2, 3, 0, 1, 5, 4],
    [2, 5, 3, 0, 1, 4],
    [3, 0, 2, 5, 4, 1],
    [3, 2, 5, 4, 0, 1],
    [3, 4, 0, 2, 5, 1],
    [3, 5, 4, 0, 2, 1],
    [4, 0, 3, 5, 1, 2],
    [4, 1, 0, 3, 5, 2],
    [4, 3, 5, 1, 0, 2],
    [4, 5, 1, 0, 3, 2],
    [5, 1, 4, 3, 2, 0],
    [5, 2, 1, 4, 3, 0],
    [5, 3, 2, 1, 4, 0],
    [5, 4, 3, 2, 1, 0],
];

/// Computes the `@OH` class that expresses the same octahedral stereochemistry
/// after the SMILES neighbor listing order of the center changes.
///
/// `old_neighbors` and `new_neighbors` must contain the same 6 atom ids; each
/// gives the order in which the neighbors are listed around the center in the
/// old and new SMILES respectively. Returns `None` when the input is not a
/// valid OH class (1-30) or the neighbor lists are not a permutation of each
/// other with length 6.
pub fn remap_oh_number(
    config: u8,
    old_neighbors: &[usize],
    new_neighbors: &[usize],
) -> Option<u8> {
    if !(1..=30).contains(&config) || old_neighbors.len() != 6 || new_neighbors.len() != 6 {
        return None;
    }
    let vmap = &OH_VERTEX_TABLE[(config - 1) as usize];

    // Vertex occupied by the atom at each position of the new listing
    let mut occupied = [0u8; 6];
    for (new_pos, atom) in new_neighbors.iter().enumerate() {
        let old_pos = old_neighbors.iter().position(|a| a == atom)?;
        occupied[new_pos] = vmap[old_pos];
    }

    // The new class is the unique row that matches up to a proper rotation
    for (idx, candidate) in OH_VERTEX_TABLE.iter().enumerate() {
        for rotation in &OCT_ROTATIONS {
            if (0..6).all(|pos| rotation[occupied[pos] as usize] == candidate[pos]) {
                return Some((idx + 1) as u8);
            }
        }
    }
    None
}

/// Trigonal-bipyramidal `@TB1`..`@TB20` vertex assignments and the descriptor's
/// symmetry group (D3, order 6), derived from RDKit. Analogous to
/// [`OH_VERTEX_TABLE`]/[`OCT_ROTATIONS`].
const TB_VERTEX_TABLE: [[u8; 5]; 20] = [
    [0, 1, 2, 3, 4],
    [0, 1, 3, 2, 4],
    [0, 1, 2, 4, 3],
    [0, 1, 3, 4, 2],
    [0, 1, 4, 2, 3],
    [0, 1, 4, 3, 2],
    [0, 4, 1, 2, 3],
    [0, 4, 1, 3, 2],
    [1, 0, 2, 3, 4],
    [1, 0, 2, 4, 3],
    [1, 0, 3, 2, 4],
    [1, 0, 3, 4, 2],
    [1, 0, 4, 2, 3],
    [1, 0, 4, 3, 2],
    [1, 2, 0, 3, 4],
    [1, 2, 0, 4, 3],
    [1, 2, 3, 0, 4],
    [1, 2, 3, 4, 0],
    [1, 2, 4, 0, 3],
    [1, 2, 4, 3, 0],
];
const TB_ROTATIONS: [[u8; 5]; 6] = [
    [0, 1, 2, 3, 4],
    [0, 2, 3, 1, 4],
    [0, 3, 1, 2, 4],
    [4, 1, 3, 2, 0],
    [4, 2, 1, 3, 0],
    [4, 3, 2, 1, 0],
];

/// Square-planar `@SP1`..`@SP3` vertex assignments and symmetry group (D4, order
/// 8), derived from RDKit.
const SP_VERTEX_TABLE: [[u8; 4]; 3] = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2]];
const SP_ROTATIONS: [[u8; 4]; 8] = [
    [0, 1, 2, 3],
    [0, 3, 2, 1],
    [1, 0, 3, 2],
    [1, 2, 3, 0],
    [2, 1, 0, 3],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
    [3, 2, 1, 0],
];

/// Remaps a trigonal-bipyramidal `@TB` class when the neighbor listing order
/// changes from `old_neighbors` to `new_neighbors` (same coset-lookup scheme as
/// [`remap_oh_number`]).
pub fn remap_tb_number(config: u8, old_neighbors: &[usize], new_neighbors: &[usize]) -> Option<u8> {
    remap_stereo_class(
        config,
        old_neighbors,
        new_neighbors,
        &TB_VERTEX_TABLE,
        &TB_ROTATIONS,
    )
}

/// Remaps a square-planar `@SP` class when the neighbor listing order changes.
pub fn remap_sp_number(config: u8, old_neighbors: &[usize], new_neighbors: &[usize]) -> Option<u8> {
    remap_stereo_class(
        config,
        old_neighbors,
        new_neighbors,
        &SP_VERTEX_TABLE,
        &SP_ROTATIONS,
    )
}

/// Generic coset remap shared by TB and SP: `vertex_table[config-1]` gives the
/// vertex occupied by each listed neighbor; the new class is the row matching the
/// relisted occupancy up to a symmetry-group element.
fn remap_stereo_class<const N: usize>(
    config: u8,
    old_neighbors: &[usize],
    new_neighbors: &[usize],
    vertex_table: &[[u8; N]],
    rotations: &[[u8; N]],
) -> Option<u8> {
    let count = vertex_table.len() as u8;
    if config < 1 || config > count || old_neighbors.len() != N || new_neighbors.len() != N {
        return None;
    }
    let vmap = &vertex_table[(config - 1) as usize];

    let mut occupied = [0u8; N];
    for (new_pos, atom) in new_neighbors.iter().enumerate() {
        let old_pos = old_neighbors.iter().position(|a| a == atom)?;
        occupied[new_pos] = vmap[old_pos];
    }

    for (idx, candidate) in vertex_table.iter().enumerate() {
        for rotation in rotations {
            if (0..N).all(|pos| rotation[occupied[pos] as usize] == candidate[pos]) {
                return Some((idx + 1) as u8);
            }
        }
    }
    None
}

/// Represents the shape of an octahedral configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OctahedralShape {
    U,    // U-shaped configuration
    Z,    // Z-shaped configuration
    Four, // 4-shaped configuration
    None,
}

/// Represents the winding direction of ligands in an octahedral configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindingDirection {
    Clockwise,
    Counterclockwise,
    None,
}

impl WindingDirection {
    pub fn reverse(&self) -> WindingDirection {
        match self {
            WindingDirection::Clockwise => WindingDirection::Counterclockwise,
            WindingDirection::Counterclockwise => WindingDirection::Clockwise,
            WindingDirection::None => WindingDirection::None,
        }
    }
}

/// Represents the direction of the viewing axis in an octahedral configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxisDirection {
    AtoF,
    AtoE,
    AtoD,
    AtoC,
    AtoB,
    None,
}

/// Represents the orientation of the viewing axis in an octahedral configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AxisOrientation {
    Normal,
    Reversed,
    None,
}

impl ChiralClass {
    pub fn invert(&self) -> Self {
        match self {
            ChiralClass::Clockwise => ChiralClass::Counterclockwise,
            ChiralClass::Counterclockwise => ChiralClass::Clockwise,
            ChiralClass::CisTransCounterclockwise => ChiralClass::CisTransClockwise,
            ChiralClass::CisTransClockwise => ChiralClass::CisTransCounterclockwise,
            ChiralClass::TH(1) => ChiralClass::TH(2),
            ChiralClass::TH(2) => ChiralClass::TH(1),
            ChiralClass::AL(1) => ChiralClass::AL(2),
            ChiralClass::AL(2) => ChiralClass::AL(1),
            // TODO: Handle other chiral classes
            _ => self.clone(),
        }
    }

    pub fn clockness(&self) -> ChiralClass {
        match self {
            ChiralClass::Clockwise | ChiralClass::Counterclockwise => *self,
            ChiralClass::CisTransClockwise => ChiralClass::Clockwise,
            ChiralClass::CisTransCounterclockwise => ChiralClass::Counterclockwise,
            ChiralClass::AL(1) => ChiralClass::Counterclockwise,
            ChiralClass::AL(2) => ChiralClass::Clockwise,
            ChiralClass::TH(1) => ChiralClass::Counterclockwise,
            ChiralClass::TH(2) => ChiralClass::Clockwise,
            _ => ChiralClass::None,
        }
    }
}

pub trait ChiralClassifier {
    fn identify_chiral_classes(&mut self);
    fn determine_cis_trans_chirality(
        &self,
        center_atom_index: usize,
        center_neighbors: &[BondTarget],
        chiral_classes: &mut [ChiralClass],
    ) -> ChiralClass;
    fn determine_square_planar_chirality(
        &self,
        index: usize,
        neighbors: &[BondTarget],
    ) -> ChiralClass;
    fn determine_allene_chirality(&self, neighbors: &[BondTarget]) -> Option<ChiralClass>;
    fn determine_trigonal_bipyramidal_chirality(
        &self,
        index: usize,
        neighbors: &[BondTarget],
    ) -> ChiralClass;
    fn identify_axial_equatorial(
        &self,
        vectors_and_indices: &[(Vector3D, usize)],
    ) -> (Vec<(usize, usize)>, Vec<(usize, usize)>);
    fn determine_octahedral_chirality(&self, index: usize, neighbors: &[BondTarget])
    -> ChiralClass;
    fn identify_octahedral_configuration(
        &self,
        vectors: &[(Vector3D, usize)],
    ) -> (
        (OctahedralShape, WindingDirection),
        (Vector3D, usize, usize),
    );
    fn determine_shape(
        &self,
        v1: &Vector3D,
        v2: &Vector3D,
        v3: &Vector3D,
        v4: &Vector3D,
        axis: &Vector3D,
    ) -> (OctahedralShape, WindingDirection);
    fn determine_tetrahedral_chirality(
        &self,
        index: usize,
        neighbors: &[BondTarget],
    ) -> ChiralClass;
    fn assign_oh_number(
        &self,
        shape: OctahedralShape,
        viewing_axis: &(Vector3D, usize, usize),
        winding: WindingDirection,
    ) -> u8;
    fn determine_axis_direction(
        &self,
        viewing_axis: &(Vector3D, usize, usize),
    ) -> (AxisDirection, AxisOrientation);
    fn find_opposite_pairs(
        &self,
        vectors: &[(Vector3D, usize)],
    ) -> Vec<((Vector3D, usize), (Vector3D, usize))>;
    fn is_potential_octahedral(&self, neighbors: &[BondTarget]) -> bool;
    fn is_potential_trigonal_bipyramidal(&self, neighbors: &[BondTarget]) -> bool;
    fn is_potential_tetrahedral_center(&self, atomic_number: u8, neighbors: &[BondTarget]) -> bool;
    fn is_tetrahedral_allene(&self, atomic_number: u8, neighbors: &[BondTarget]) -> bool;
    fn is_potential_cis_trans(&self, neighbors: &[BondTarget]) -> bool;
    fn calculate_plane(&self, center: usize, neighbors: &[BondTarget]) -> Option<Vector3D>;
    fn is_mol_chiral(&self) -> bool;
}

impl ChiralClassifier for Molecule3D {
    fn identify_chiral_classes(&mut self) {
        if self.chiral_classes.is_some() {
            return;
        }
        let mut chiral_classes = vec![ChiralClass::None; self.len()];

        for (index, &atomic_number) in self.atomic_numbers().iter().enumerate() {
            // If we already know the chirality, skip it
            if chiral_classes[index] != ChiralClass::None {
                continue;
            }
            let Some(neighbors) = self.get_atom_bonds(index) else {
                continue;
            };

            chiral_classes[index] = if self
                .is_potential_tetrahedral_center(atomic_number, neighbors)
            {
                let tetrahedral_chirality = self.determine_tetrahedral_chirality(index, neighbors);
                if tetrahedral_chirality != ChiralClass::None {
                    tetrahedral_chirality
                } else {
                    self.determine_square_planar_chirality(index, neighbors)
                }
            } else if self.is_tetrahedral_allene(atomic_number, neighbors) {
                self.determine_allene_chirality(neighbors)
                    .unwrap_or(ChiralClass::None)
            } else if self.is_potential_cis_trans(neighbors) {
                self.determine_cis_trans_chirality(index, neighbors, &mut chiral_classes)
            } else if self.is_potential_trigonal_bipyramidal(neighbors) {
                self.determine_trigonal_bipyramidal_chirality(index, neighbors)
            } else if self.is_potential_octahedral(neighbors) {
                self.determine_octahedral_chirality(index, neighbors)
            } else {
                ChiralClass::None
            };
        }

        self.chiral_classes = Some(chiral_classes);
    }

    /// Double bond geometry, read from coordinates and stored the way the
    /// SMILES side stores it.
    ///
    /// `convert_directional_bonds_to_chiral` fixes that convention: the
    /// reference substituent at each end is its first neighbour that is not the
    /// double bond, in bond listing order; the end with the lower atom index is
    /// always `CisTransCounterclockwise`, and the other end is
    /// `CisTransClockwise` when the two references are on opposite sides and
    /// `CisTransCounterclockwise` when they are on the same one.
    ///
    /// This used to build its own frame instead — a perpendicular axis from the
    /// centre's first neighbour, applied to both ends — which got three things
    /// wrong at once. It wrote the plain `Clockwise` and `Counterclockwise`
    /// classes, which are the tetrahedral ones, so `C/C=C/C` came back as
    /// `C[C@@H]=[C@@H]C`, with stereocentres invented on atoms that are not
    /// stereocentres. It shared one axis between two ends whose frames are
    /// mirror images. And it keyed on whichever neighbour the bond list
    /// happened to hold first at each end independently, so the answer moved
    /// with the order the bonds were built in.
    fn determine_cis_trans_chirality(
        &self,
        center_atom_index: usize,
        center_neighbors: &[BondTarget],
        chiral_classes: &mut [ChiralClass],
    ) -> ChiralClass {
        let Some(partner) = center_neighbors
            .iter()
            .find(|bond| bond.bond_order() == BondOrder::Double)
            .map(|bond| bond.target())
        else {
            return ChiralClass::None;
        };

        // Keyed on the pair, not on which end we were called with, so both
        // calls agree.
        let (first, second) = (center_atom_index.min(partner), center_atom_index.max(partner));

        let reference = |atom: usize, exclude: usize| -> Option<usize> {
            self.get_atom_bonds(atom)?
                .iter()
                .find(|bond| bond.bond_order() != BondOrder::Double && bond.target() != exclude)
                .map(|bond| bond.target())
        };
        let (Some(reference_first), Some(reference_second)) =
            (reference(first, second), reference(second, first))
        else {
            return ChiralClass::None;
        };

        // One or two substituents at each end. Two is the carbon case. One is
        // the imine case — a `C=N` whose nitrogen carries a lone pair in the
        // second position — and it is every bit as stereogenic; requiring two
        // threw away the configuration of every oxime and hydrazone in the
        // corpus. Three or more is not a double bond geometry at all.
        let ends_are_stereogenic = [(first, second), (second, first)]
            .iter()
            .all(|&(atom, exclude)| {
                self.get_atom_bonds(atom).is_some_and(|bonds| {
                    let substituents = bonds
                        .iter()
                        .filter(|bond| {
                            bond.bond_order() != BondOrder::Double && bond.target() != exclude
                        })
                        .count();
                    (1..=2).contains(&substituents)
                })
            });
        if !ends_are_stereogenic {
            return ChiralClass::None;
        }

        let (Some(position_first), Some(position_second)) = (
            self.get_atom_position(first),
            self.get_atom_position(second),
        ) else {
            return ChiralClass::None;
        };
        let (Some(position_reference_first), Some(position_reference_second)) = (
            self.get_atom_position(reference_first),
            self.get_atom_position(reference_second),
        ) else {
            return ChiralClass::None;
        };

        // Which side of the double bond each reference sits on, as the
        // component of its bond perpendicular to the axis. Same sign is cis.
        let axis = position_second - position_first;
        let axis_length_squared = axis.dot(&axis);
        if axis_length_squared < f64::EPSILON {
            return ChiralClass::None;
        }
        let perpendicular = |from: Vector3D, to: Vector3D| -> Vector3D {
            let along = to - from;
            along - axis * (along.dot(&axis) / axis_length_squared)
        };
        let side_first = perpendicular(position_first, position_reference_first);
        let side_second = perpendicular(position_second, position_reference_second);
        let alignment = side_first.dot(&side_second);
        if alignment.abs() < 1e-9 {
            return ChiralClass::None; // collinear: no geometry to read
        }

        chiral_classes[first] = ChiralClass::CisTransCounterclockwise;
        chiral_classes[second] = if alignment < 0.0 {
            ChiralClass::CisTransClockwise
        } else {
            ChiralClass::CisTransCounterclockwise
        };
        chiral_classes[center_atom_index]
    }

    /// Determines the chirality of a square planar structure.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the central atom.
    /// * `neighbors` - A slice of `BondTarget` representing the neighboring atoms.
    ///
    /// # Returns
    ///
    /// A `ChiralClass` indicating the chirality of the square planar center.
    ///
    /// This geometric analysis allows us to classify the square planar structure into
    /// one of three chiral arrangements: U-shaped, Z-shaped, or 4-shaped.
    fn determine_square_planar_chirality(
        &self,
        index: usize,
        neighbors: &[BondTarget],
    ) -> ChiralClass {
        if neighbors.len() != 4 {
            return ChiralClass::None;
        }

        let self_position = match self.get_atom_position(index) {
            Some(pos) => pos,
            None => return ChiralClass::None,
        };

        let positions: Vec<Vector3D> = neighbors
            .iter()
            .flat_map(|&neighbor| self.get_atom_position(neighbor.target()))
            .collect();

        if positions.len() != 4 {
            return ChiralClass::None;
        }

        let vectors: Vec<Vector3D> = positions.iter().map(|&pos| pos - self_position).collect();

        let cross1: Vector3D = vectors[0].cross(&vectors[1]);
        let cross2: Vector3D = vectors[1].cross(&vectors[2]);
        let cross3: Vector3D = vectors[2].cross(&vectors[3]);
        let cross4: Vector3D = vectors[3].cross(&vectors[0]);

        // This is used for Z and 4
        let magnitude1 = cross1.magnitude();

        // Check if all cross products are pointing in the same direction
        let dot1 = cross1.dot(&cross2);
        let dot2 = cross2.dot(&cross3);
        let dot3 = cross3.dot(&cross4);
        let dot4 = cross4.dot(&cross1);

        match (dot1.abs(), dot2.abs(), dot3.abs(), dot4.abs()) {
            (0.1..=f64::MAX, 0.1..=f64::MAX, 0.1..=f64::MAX, 0.1..=f64::MAX) => ChiralClass::SP(1), // U
            (-0.1..=0.1, -0.1..=0.1, -0.1..=0.1, -0.1..=0.1) => {
                if magnitude1 > 0.1 {
                    ChiralClass::SP(3) // Z
                } else {
                    ChiralClass::SP(2) // 4
                }
            }
            _ => ChiralClass::None,
        }
    }

    /// Determines the chirality of a trigonal bipyramidal structure.  /// /// # Arguments ///
    /// * `index` - The index of the central atom.
    /// * `neighbors` - A slice of `BondTarget` representing the neighboring atoms.
    ///
    /// # Returns
    ///
    /// A `ChiralClass` indicating the chirality of the trigonal bipyramidal center.
    fn determine_trigonal_bipyramidal_chirality(
        &self,
        index: usize,
        neighbors: &[BondTarget],
    ) -> ChiralClass {
        if neighbors.len() != 5 {
            return ChiralClass::None;
        }

        let central_position = match self.get_atom_position(index) {
            Some(pos) => pos,
            None => return ChiralClass::None,
        };

        let neighbor_positions: Vec<(Vector3D, usize)> = neighbors
            .iter()
            .map(|&neighbor| {
                (
                    self.get_atom_position(neighbor.target()).unwrap(),
                    neighbor.target(),
                )
            })
            .collect();

        if neighbor_positions.len() != 5 {
            return ChiralClass::None;
        }

        let vectors_and_indices: Vec<(Vector3D, usize)> = neighbor_positions
            .iter()
            .map(|&(pos, index)| (pos - central_position, index))
            .collect();

        // Identify axial and equatorial ligands
        let (axial_indices, equatorial_indices) =
            self.identify_axial_equatorial(&vectors_and_indices);
        let is_ordered = axial_indices[0].1 < axial_indices[1].1;
        if axial_indices.len() != 2 || equatorial_indices.len() != 3 {
            return ChiralClass::None;
        }
        let axial_indices = [axial_indices[0].0, axial_indices[1].0];
        let equatorial_indices = [
            equatorial_indices[0],
            equatorial_indices[1],
            equatorial_indices[2],
        ];
        // Is ordering clockwise?
        let mut is_clockwise = equatorial_indices[0].1 < equatorial_indices[1].1
            && equatorial_indices[1].1 < equatorial_indices[2].1;
        if is_ordered {
            is_clockwise = !is_clockwise;
        }

        // Determine chirality based on the viewing axis and order
        match (&axial_indices, is_clockwise) {
            ([0, 4], false) => ChiralClass::TB(1), // a -> e, (b, c, d) anti-clockwise
            ([0, 4], true) => ChiralClass::TB(2),  // a -> e, (b, c, d) clockwise
            ([0, 3], false) => ChiralClass::TB(3), // a -> d, (b, c, e) anti-clockwise
            ([0, 3], true) => ChiralClass::TB(4),  // a -> d, (b, c, e) clockwise
            ([0, 2], false) => ChiralClass::TB(5), // a -> c, (b, d, e) anti-clockwise
            ([0, 2], true) => ChiralClass::TB(6),  // a -> c, (b, d, e) clockwise
            ([0, 1], false) => ChiralClass::TB(7), // a -> b, (c, d, e) anti-clockwise
            ([0, 1], true) => ChiralClass::TB(8),  // a -> b, (c, d, e) clockwise
            ([1, 4], false) => ChiralClass::TB(9), // b -> e, (a, c, d) anti-clockwise
            ([1, 4], true) => ChiralClass::TB(11), // b -> e, (a, c, d) clockwise
            ([1, 3], false) => ChiralClass::TB(10), // b -> d, (a, c, e) anti-clockwise
            ([1, 3], true) => ChiralClass::TB(12), // b -> d, (a, c, e) clockwise
            ([1, 2], false) => ChiralClass::TB(13), // b -> c, (a, d, e) anti-clockwise
            ([1, 2], true) => ChiralClass::TB(14), // b -> c, (a, d, e) clockwise
            ([2, 4], false) => ChiralClass::TB(15), // c -> e, (a, b, d) anti-clockwise
            ([2, 4], true) => ChiralClass::TB(20), // c -> e, (a, b, d) clockwise
            ([2, 3], false) => ChiralClass::TB(16), // c -> d, (a, b, e) anti-clockwise
            ([2, 3], true) => ChiralClass::TB(19), // c -> d, (a, b, e) clockwise
            ([3, 4], false) => ChiralClass::TB(17), // d -> e, (a, b, c) anti-clockwise
            ([3, 4], true) => ChiralClass::TB(18), // d -> e, (a, b, c) clockwise
            _ => ChiralClass::None,
        }
    }

    /// Identifies axial and equatorial ligands in a trigonal bipyramidal structure.
    ///
    /// # Arguments
    ///
    /// * `vectors` - A slice of `Vector` representing the vectors from the central atom to its neighbors.
    ///
    /// # Returns
    ///
    /// A tuple containing two vectors of indices:
    /// - The first vector contains indices of the two axial ligands.
    /// - The second vector contains indices of the three equatorial ligands.
    fn identify_axial_equatorial(
        &self,
        vectors_and_indices: &[(Vector3D, usize)],
    ) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let mut axial_indices = Vec::new();
        let mut equatorial_indices = Vec::new();

        for (i, &v1) in vectors_and_indices.iter().enumerate() {
            for (j, &v2) in vectors_and_indices.iter().enumerate() {
                if i < j {
                    if v1.0.is_opposite(&v2.0, 0.1) {
                        // If two vectors are nearly opposite (180 degrees apart), they form the axis
                        axial_indices.push((i, vectors_and_indices[i].1));
                        axial_indices.push((j, vectors_and_indices[j].1));
                    }
                }
            }
        }
        for (i, &v1) in vectors_and_indices.iter().enumerate() {
            if !axial_indices.contains(&(i, v1.1)) {
                equatorial_indices.push((i, v1.1));
            }
        }

        (axial_indices, equatorial_indices)
    }

    /// Determines the chirality of an octahedral structure.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the central atom in the molecule.
    /// * `neighbors` - A slice of `BondTarget` representing the neighboring atoms (ligands).
    ///
    /// # Returns
    ///
    /// A `ChiralClass` indicating the chirality of the octahedral center:
    /// - `ChiralClass::OH(n)` where n is 1-30, representing different octahedral configurations
    /// - `ChiralClass::None` if not chiral or indeterminate
    ///
    /// # Algorithm
    ///
    /// 1. Checks if there are exactly 6 neighbors (ligands).
    /// 2. Retrieves the positions of the central atom and its neighbors.
    /// 3. Identifies the shape and viewing axis based on the ligand positions.
    /// 4. Determines the winding direction (clockwise or anticlockwise) of the relevant ligands.
    /// 5. Assigns the appropriate OH number based on the shape, axis, and winding direction.
    fn determine_octahedral_chirality(
        &self,
        index: usize,
        neighbors: &[BondTarget],
    ) -> ChiralClass {
        if neighbors.len() != 6 {
            return ChiralClass::None;
        }

        let central_position = match self.get_atom_position(index) {
            Some(pos) => pos,
            None => return ChiralClass::None,
        };

        let neighbor_positions: Vec<(Vector3D, usize)> = neighbors
            .iter()
            .map(|&neighbor| {
                (
                    self.get_atom_position(neighbor.target()).unwrap(),
                    neighbor.target(),
                )
            })
            .collect();

        if neighbor_positions.len() != 6 {
            return ChiralClass::None;
        }

        let vectors: Vec<(Vector3D, usize)> = neighbor_positions
            .iter()
            .map(|&(pos, index)| (pos - central_position, index))
            .collect();

        let ((shape, winding), viewing_axis) = self.identify_octahedral_configuration(&vectors);
        let oh_number = self.assign_oh_number(shape, &viewing_axis, winding);

        ChiralClass::OH(oh_number)
    }

    /// Identifies the octahedral configuration (shape, viewing axis, and relevant ligands).
    fn identify_octahedral_configuration(
        &self,
        vectors: &[(Vector3D, usize)],
    ) -> (
        (OctahedralShape, WindingDirection),
        (Vector3D, usize, usize),
    ) {
        let pairs = self.find_opposite_pairs(vectors);
        let mut viewing_axis = (pairs[0].0.0 - pairs[0].1.0, pairs[0].0.1, pairs[0].1.1);
        if vectors[viewing_axis.1].1 > vectors[viewing_axis.2].1 {
            viewing_axis = (pairs[0].0.0, pairs[0].1.1, pairs[0].0.1);
        }

        let relevant_ligands: Vec<Vector3D> = vectors
            .iter()
            .enumerate()
            .filter(|&(index, _v)| index != viewing_axis.1 && index != viewing_axis.2)
            .map(|(_, vector)| vector.0)
            .collect();

        let shape = self.determine_shape(
            &relevant_ligands[0],
            &relevant_ligands[1],
            &relevant_ligands[2],
            &relevant_ligands[3],
            &viewing_axis.0,
        );

        (shape, viewing_axis)
    }

    /// Determines the shape of the octahedral configuration.
    fn determine_shape(
        &self,
        v1: &Vector3D,
        v2: &Vector3D,
        v3: &Vector3D,
        v4: &Vector3D,
        axis: &Vector3D,
    ) -> (OctahedralShape, WindingDirection) {
        let cross_product1 = v1.cross(v2).dot(axis);
        let cross_product2 = v2.cross(v3).dot(axis);
        let cross_product3 = v3.cross(v4).dot(axis);
        match (cross_product1, cross_product2, cross_product3) {
            (f64::MIN..=-0.1, f64::MIN..=-0.1, f64::MIN..=-0.1) => {
                (OctahedralShape::U, WindingDirection::Counterclockwise)
            }
            (0.1..=f64::MAX, 0.1..=f64::MAX, 0.1..=f64::MAX) => {
                (OctahedralShape::U, WindingDirection::Clockwise)
            }
            (f64::MIN..=-0.1, -0.1..=0.1, 0.1..=f64::MAX) => {
                (OctahedralShape::Z, WindingDirection::Counterclockwise)
            }
            (0.1..=f64::MAX, -0.1..=0.1, f64::MIN..=-0.1) => {
                (OctahedralShape::Z, WindingDirection::Clockwise)
            }
            (-0.1..=0.1, f64::MIN..=-0.1, -0.1..=0.1) => {
                (OctahedralShape::Four, WindingDirection::Counterclockwise)
            }
            (-0.1..=0.1, 0.1..=f64::MAX, -0.1..=0.1) => {
                (OctahedralShape::Four, WindingDirection::Clockwise)
            }
            _ => (OctahedralShape::None, WindingDirection::None),
        }
    }

    /// Assigns the OH number based on shape, viewing axis, and winding direction.
    fn assign_oh_number(
        &self,
        shape: OctahedralShape,
        viewing_axis: &(Vector3D, usize, usize),
        winding: WindingDirection,
    ) -> u8 {
        let (axis_direction, axis_orientation) = self.determine_axis_direction(&viewing_axis);
        let mut winding = winding;
        if axis_orientation == AxisOrientation::Reversed {
            winding = winding.reverse();
        }
        oh_number_from_config(shape, axis_direction, winding)
    }

    /// Determines the direction of the viewing axis.
    fn determine_axis_direction(
        &self,
        viewing_axis: &(Vector3D, usize, usize),
    ) -> (AxisDirection, AxisOrientation) {
        match (viewing_axis.1, viewing_axis.2) {
            (0, 5) => (AxisDirection::AtoF, AxisOrientation::Normal),
            (0, 4) => (AxisDirection::AtoE, AxisOrientation::Normal),
            (0, 3) => (AxisDirection::AtoD, AxisOrientation::Normal),
            (0, 2) => (AxisDirection::AtoC, AxisOrientation::Normal),
            (0, 1) => (AxisDirection::AtoB, AxisOrientation::Normal),
            (5, 0) => (AxisDirection::AtoF, AxisOrientation::Reversed),
            (4, 0) => (AxisDirection::AtoE, AxisOrientation::Reversed),
            (3, 0) => (AxisDirection::AtoD, AxisOrientation::Reversed),
            (2, 0) => (AxisDirection::AtoC, AxisOrientation::Reversed),
            (1, 0) => (AxisDirection::AtoB, AxisOrientation::Reversed),
            _ => (AxisDirection::None, AxisOrientation::None),
        }
    }

    /// Finds opposite pairs of ligands in an octahedral structure.
    ///
    /// # Arguments
    ///
    /// * `vectors` - A slice of `Vector` representing the vectors from the central atom to its neighbors.
    ///
    /// # Returns
    ///
    /// A vector of tuples, each containing a pair of opposite vectors.
    fn find_opposite_pairs(
        &self,
        vectors: &[(Vector3D, usize)],
    ) -> Vec<((Vector3D, usize), (Vector3D, usize))> {
        let mut pairs = Vec::new();
        let tolerance = 0.1; // Tolerance for considering vectors as opposite

        for i in 0..vectors.len() {
            for j in i + 1..vectors.len() {
                if vectors[i].0.is_opposite(&vectors[j].0, tolerance) {
                    pairs.push(((vectors[i].0, i), (vectors[j].0, j)));
                }
            }
        }

        pairs
    }

    /// Determines the chirality of a tetrahedral center.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the central atom.
    /// * `neighbors` - A slice of `BondTarget` representing the neighboring atoms.
    ///
    /// # Returns
    ///
    /// A `ChiralClass` indicating the chirality of the tetrahedral center.
    fn determine_tetrahedral_chirality(
        &self,
        index: usize,
        neighbors: &[BondTarget],
    ) -> ChiralClass {
        if neighbors.len() != 4 {
            return ChiralClass::None;
        }

        let central_position = match self.get_atom_position(index) {
            Some(pos) => pos,
            None => return ChiralClass::None,
        };

        // Store the tag against the centre's index-sorted neighbor ordering (the
        // convention used everywhere else), independent of bond listing order.
        let mut sorted_neighbors: Vec<usize> =
            neighbors.iter().map(|n| n.target()).collect();
        sorted_neighbors.sort_unstable();

        let neighbor_positions: Vec<Vector3D> = sorted_neighbors
            .iter()
            .flat_map(|&target| self.get_atom_position(target))
            .collect();

        if neighbor_positions.len() != 4 {
            return ChiralClass::None;
        }

        // Calculate vectors from central atom to neighbors
        let vectors: Vec<Vector3D> = neighbor_positions
            .iter()
            .map(|&pos| pos - central_position)
            .collect();

        // Calculate the signed volume of the tetrahedron
        let signed_volume = vectors[0].dot(&vectors[1].cross(&vectors[2]));

        if signed_volume.abs() < f64::EPSILON {
            ChiralClass::None // Planar arrangement
        } else if signed_volume > 0.0 {
            // Matches this crate's documented `ChiralClass::TH` convention
            // (`TH(u8), // @ 1..=2  1 = anti-clockwise, 2 = clockwise`) and
            // the SMILES parser's `@@` -> `TH(2)` mapping
            // (`crate::io::smiles::assign_proper_chirality`); cross-checked
            // against RDKit's `AssignStereochemistryFrom3D` via
            // `tests/embed_oracle.py`. Previously inverted (returned
            // `TH(1)` here) - found while adding the `embed` module's own
            // independent chirality-consistency term, which had to avoid
            // matching this function's sign to embed molecules correctly.
            ChiralClass::TH(1) // Anti-clockwise (@)
        } else {
            ChiralClass::TH(2) // Clockwise (@@)
        }
    }

    /// Determines the chirality of an allene-like structure.
    ///
    /// # Arguments
    ///
    /// * `neighbors` - A slice of `BondTarget` representing the neighbors of the central atom.
    ///
    /// # Returns
    ///
    /// An `Option<ChiralClass>` indicating the chirality of the allene structure, if determinable.
    ///
    /// # Notes
    ///
    /// This function assumes that the central atom (index) is already known to be part of an
    /// allene-like structure. The `index` parameter has been removed as it was not used in the
    /// function body.
    fn determine_allene_chirality(&self, neighbors: &[BondTarget]) -> Option<ChiralClass> {
        let [left, right] = neighbors else {
            return None;
        };

        let left_neighbors = self.get_atom_bonds(left.target())?;
        let right_neighbors = self.get_atom_bonds(right.target())?;

        if left_neighbors.len() != 2 || right_neighbors.len() != 2 {
            return None;
        }

        let left_plane = self.calculate_plane(left.target(), left_neighbors)?;
        let right_plane = self.calculate_plane(right.target(), right_neighbors)?;

        let angle = left_plane.angle_between(&right_plane);

        Some(if angle > Some(0.0) {
            ChiralClass::AL(1)
        } else {
            ChiralClass::AL(2)
        })
    }

    /// Calculates the normal vector of the plane formed by a central atom and its two neighbors.
    ///
    /// # Arguments
    ///
    /// * `center` - The index of the central atom.
    /// * `neighbors` - A slice of `BondTarget` representing the neighboring atoms.
    ///
    /// # Returns
    ///
    /// * `Option<Vector>` - The normal vector of the plane if all positions are valid and there are exactly two neighbors, or `None` otherwise.
    ///
    /// # Safety
    ///
    /// This function checks that:
    /// - The `neighbors` slice contains exactly two elements.
    /// - All atom positions (center and neighbors) are valid.
    /// - The resulting vectors are not parallel (to avoid a zero cross product).
    fn calculate_plane(&self, center: usize, neighbors: &[BondTarget]) -> Option<Vector3D> {
        if neighbors.len() != 2 {
            return None;
        }

        let center_pos = self.get_atom_position(center)?;
        let pos1 = self.get_atom_position(neighbors[0].target())?;
        let pos2 = self.get_atom_position(neighbors[1].target())?;

        let vec1 = pos1 - center_pos;
        let vec2 = pos2 - center_pos;

        // Check if vectors are not parallel (cross product is not zero)
        let cross_product = vec1.cross(&vec2);
        if cross_product.length() < f64::EPSILON {
            return None;
        }

        Some(cross_product)
    }

    /// Determines if an atom is a tetrahedral allene.
    ///
    /// # Arguments
    ///
    /// * `atomic_number` - The atomic number of the central atom.
    /// * `neighbors` - A slice of `BondTarget` representing the neighboring atoms.
    ///
    /// # Returns
    ///
    /// * `bool` - True if the atom is a tetrahedral allene, false otherwise.
    ///
    /// # Details
    ///
    /// A tetrahedral allene is a carbon atom (atomic number 6) with exactly two carbon neighbors,
    /// both connected by double bonds.
    fn is_tetrahedral_allene(&self, atomic_number: u8, neighbors: &[BondTarget]) -> bool {
        atomic_number == 6 // Is atom carbon
            && neighbors.len() == 2 // Does it have two neighbors
            && neighbors
                    .iter()
                    .all(|neighbor|
                        self.get_atomic_number(neighbor.target()) == 6 // Are all neighbors carbon
                        && neighbor.bond_order() == BondOrder::Double  // Are all bonds double
                                                                       // bonds
                        )
    }

    /// Determines if an atom is a potential trigonal bipyramidal center.
    ///
    /// # Arguments
    ///
    /// * `neighbors` - A slice of `BondTarget` representing the neighboring atoms.
    ///
    /// # Returns
    ///
    /// * `bool` - True if the atom has exactly 5 neighbors, indicating a potential trigonal bipyramidal structure.
    fn is_potential_trigonal_bipyramidal(&self, neighbors: &[BondTarget]) -> bool {
        neighbors.len() == 5
    }

    fn is_potential_cis_trans(&self, neighbors: &[BondTarget]) -> bool {
        // One double bond and one or two single ones: a carbon, or a nitrogen
        // whose lone pair takes the second position. Asking for two single
        // bonds turned away both ends of an azo `N=N`, so its geometry was
        // never read; an imine kept its own only because the carbon end
        // passed. `determine_cis_trans_chirality` already takes either.
        (2..=3).contains(&neighbors.len())
            && neighbors
                .iter()
                .filter(|neighbor| neighbor.bond_order() == BondOrder::Double)
                .count()
                == 1
            && neighbors
                .iter()
                .filter(|neighbor| neighbor.bond_order() == BondOrder::Single)
                .count()
                == neighbors.len() - 1
    }

    /// Determines if an atom is a potential octahedral center.
    ///
    /// # Arguments
    ///
    /// * `neighbors` - A slice of `BondTarget` representing the neighboring atoms.
    ///
    /// # Returns
    ///
    /// * `bool` - True if the atom has exactly 6 neighbors, indicating a potential octahedral structure.
    fn is_potential_octahedral(&self, neighbors: &[BondTarget]) -> bool {
        neighbors.len() == 6
    }

    fn is_potential_tetrahedral_center(&self, atomic_number: u8, neighbors: &[BondTarget]) -> bool {
        let number_of_hydrogens = neighbors
            .iter()
            .filter(|&neighbor| self.get_atomic_number(neighbor.target()) == 1)
            .count();

        neighbors.len() == 4
            && number_of_hydrogens < 2
            && atomic_number != 16 // Sulfur
            && atomic_number != 34 // Selenium
    }

    fn is_mol_chiral(&self) -> bool {
        self.chiral_classes()
            .unwrap_or_default()
            .iter()
            .any(|&class| class != ChiralClass::None)
    }
}

/// Updates the chirality of a molecule in-place based on a given relabeling of atoms.
///
/// # Arguments
/// * `chiral_classes` - The array of chiral classes to update
/// * `mol` - The molecule whose chirality needs to be updated
/// * `labeling` - The new ordering of atoms (e.g., [1,2,0] means atom 0 moves to position 1)
///
/// # Examples
/// ```
/// use oxidane::prelude::*;
/// use oxidane::molecule::chirality::{ChiralClass, update_chirality_after_relabeling_in_place};
///
/// // Test tetrahedral chirality
/// let mut mol = MolecularGraph::from_smiles_single("N[C@](Br)(O)C").unwrap();
/// let labeling = vec![2, 1, 0]; // Swap first two atoms
/// update_chirality_after_relabeling_in_place(&mut mol, &labeling);
///
/// assert_eq!(mol.chiral_classes().unwrap()[1], ChiralClass::TH(2)); // Chirality inverted
///
/// // Trigonal bipyramidal centers keep a valid TB class after relabeling
/// let mut mol = MolecularGraph::from_smiles_single("S[As@TB1](F)(Cl)(Br)N").unwrap();
/// let labeling = vec![2, 1, 0];
/// update_chirality_after_relabeling_in_place(&mut mol, &labeling);
/// assert!(matches!(mol.chiral_classes().unwrap()[1], ChiralClass::TB(_)));
/// ```
pub fn update_chirality_after_relabeling_in_place(mol: &mut impl Molecule, labeling: &[usize]) {
    let mut temp_classes = vec![ChiralClass::None; mol.len()];
    let Some(chiral_classes) = mol.chiral_classes() else {
        return;
    };

    for (old_idx, &chiral_class) in chiral_classes.iter().enumerate() {
        let new_idx = *labeling.get(old_idx).unwrap_or(&old_idx);

        temp_classes[new_idx] = match chiral_class {
            ChiralClass::None => ChiralClass::None,
            // TODO: Implement cis/trans chirality
            ChiralClass::CisTransClockwise => ChiralClass::CisTransClockwise,
            ChiralClass::CisTransCounterclockwise => ChiralClass::CisTransCounterclockwise,
            ChiralClass::TH(n) => {
                let Some(neighbors) = mol.get_atom_bonds(old_idx) else {
                    continue;
                };
                if neighbors.len() != 4 {
                    continue;
                };
                let old_order: Vec<_> = neighbors.iter().map(|b| b.target()).collect();
                let new_order: Vec<_> = neighbors
                    .iter()
                    .map(|b| *labeling.get(b.target()).unwrap_or(&b.target()))
                    .collect();

                if calculate_permutation_parity(&old_order, &new_order) {
                    ChiralClass::TH(3 - n) // Swap 1<->2 to invert chirality
                } else {
                    ChiralClass::TH(n)
                }
            }

            ChiralClass::AL(n) => {
                let Some(neighbors) = mol.get_atom_bonds(old_idx) else {
                    continue;
                };
                if neighbors.len() != 2 {
                    continue;
                };
                let left = neighbors[0].target();
                let right = neighbors[1].target();
                // The carriers are the four substituents at the two ends. The
                // bond back to the centre must be left out: including it put
                // the centre in the list twice, and a permutation parity over
                // a list with a repeated element is not defined — which is
                // why an allene's tag did not survive renumbering against any
                // fixed reading of its carriers.
                let carriers: Vec<usize> = [left, right]
                    .iter()
                    .flat_map(|&end| {
                        mol.get_atom_bonds(end)
                            .unwrap_or_default()
                            .iter()
                            .map(|bond| bond.target())
                            .filter(|&target| target != old_idx)
                            .collect::<Vec<_>>()
                    })
                    .collect();
                if carriers.len() != 4 {
                    continue;
                }

                let old_order: Vec<_> = carriers.clone();
                let new_order: Vec<_> = carriers
                    .iter()
                    .map(|&target| *labeling.get(target).unwrap_or(&target))
                    .collect();

                if calculate_permutation_parity(&old_order, &new_order) {
                    ChiralClass::AL(3 - n) // Swap 1<->2 to invert chirality
                } else {
                    ChiralClass::AL(n)
                }
            }

            ChiralClass::SP(n) => {
                let Some(neighbors) = mol.get_atom_bonds(old_idx) else {
                    continue;
                };
                if neighbors.len() != 4 {
                    continue;
                };

                let old_order: Vec<_> = neighbors.iter().map(|b| b.target()).collect();
                let new_order: Vec<_> = neighbors
                    .iter()
                    .map(|b| *labeling.get(b.target()).unwrap_or(&b.target()))
                    .collect();

                update_sp_chirality(n, &old_order, &new_order)
            }

            ChiralClass::TB(n) => {
                let Some(neighbors) = mol.get_atom_bonds(old_idx) else {
                    continue;
                };
                if neighbors.len() != 5 {
                    continue;
                };

                let old_order: Vec<_> = neighbors.iter().map(|b| b.target()).collect();
                let new_order: Vec<_> = neighbors
                    .iter()
                    .map(|b| *labeling.get(b.target()).unwrap_or(&b.target()))
                    .collect();

                update_tb_chirality(n, &old_order, &new_order)
            }

            ChiralClass::OH(n) => {
                let Some(neighbors) = mol.get_atom_bonds(old_idx) else {
                    continue;
                };
                if neighbors.len() != 6 {
                    continue;
                };

                let old_order: Vec<_> = neighbors.iter().map(|b| b.target()).collect();
                let new_order: Vec<_> = neighbors
                    .iter()
                    .map(|b| *labeling.get(b.target()).unwrap_or(&b.target()))
                    .collect();

                update_oh_chirality(n, &old_order, &new_order)
            }

            ChiralClass::Clockwise | ChiralClass::Counterclockwise => {
                let Some(neighbors) = mol.get_atom_bonds(old_idx) else {
                    continue;
                };

                let number_of_double_bonds = neighbors
                    .iter()
                    .filter(|bond| bond.bond_order() == BondOrder::Double)
                    .count();
                let number_of_bonds = neighbors.len();

                let old_order: Vec<usize>;
                let new_order: Vec<usize>;

                if number_of_double_bonds == 2 && number_of_bonds == 2 {
                    // Allene-like structure
                    let left = neighbors[0].target();
                    let right = neighbors[1].target();
                    let left_neighbors = mol.get_atom_bonds(left).unwrap();
                    let right_neighbors = mol.get_atom_bonds(right).unwrap();
                    let combined_neighbors = [left_neighbors, right_neighbors].concat();

                    old_order = combined_neighbors
                        .iter()
                        .map(|b| b.target())
                        .filter(|&target| target != old_idx)
                        .collect();
                    new_order = combined_neighbors
                        .iter()
                        .map(|b| *labeling.get(b.target()).unwrap_or(&b.target()))
                        .filter(|&target| target != old_idx)
                        .collect();
                } else if number_of_double_bonds == 1 && number_of_bonds == 3 {
                    // Cis/trans double bond
                    if mol.number_of_bonded_element(old_idx, 1) == 1 {
                        // Handle case with hydrogen
                        let non_hydrogen_neighbor = neighbors
                            .iter()
                            .find(|bond| {
                                mol.get_atomic_number(bond.target()) != 1
                                    && bond.bond_order() == BondOrder::Single
                            })
                            .unwrap()
                            .target();
                        old_order = [non_hydrogen_neighbor, old_idx].to_vec();
                        new_order = [
                            *labeling
                                .get(non_hydrogen_neighbor)
                                .unwrap_or(&non_hydrogen_neighbor),
                            labeling[old_idx],
                        ]
                        .to_vec();
                    } else {
                        old_order = neighbors.iter().map(|b| b.target()).collect();
                        new_order = neighbors
                            .iter()
                            .map(|b| *labeling.get(b.target()).unwrap_or(&b.target()))
                            .collect();
                    }
                } else {
                    old_order = neighbors.iter().map(|b| b.target()).collect();
                    new_order = neighbors
                        .iter()
                        .map(|b| *labeling.get(b.target()).unwrap_or(&b.target()))
                        .collect();
                }

                if calculate_permutation_parity(&old_order, &new_order) {
                    match chiral_class {
                        ChiralClass::Clockwise => ChiralClass::Counterclockwise,
                        ChiralClass::Counterclockwise => ChiralClass::Clockwise,
                        _ => unreachable!(),
                    }
                } else {
                    chiral_class
                }
            }
        };
    }

    let chiral_classes = mol.chiral_classes_mut().unwrap();
    chiral_classes.copy_from_slice(&temp_classes);
}

/// Calculates whether a permutation has odd or even parity
fn calculate_permutation_parity(original: &[usize], permuted: &[usize]) -> bool {
    let mut inversions = 0;
    let n = original.len();

    // Create mapping from original values to their positions
    let mut pos_map = vec![0; original.iter().max().map_or(0, |&x| x + 1)];
    for (i, &val) in original.iter().enumerate() {
        pos_map[val] = i;
    }

    // Convert permuted sequence to positions in original
    let permuted_positions: Vec<_> = permuted.iter().map(|&x| pos_map[x]).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if permuted_positions[i] > permuted_positions[j] {
                inversions += 1;
            }
        }
    }

    inversions % 2 == 1
}

/// The neighbors of a relabelled centre in the order its SMILES lists them:
/// ascending new label. `old_order` holds the neighbor atom ids in their
/// original listing order and `new_order` the new label of each of those
/// neighbors, position for position.
fn relisted(old_order: &[usize], new_order: &[usize]) -> Vec<usize> {
    let mut positions: Vec<usize> = (0..old_order.len()).collect();
    positions.sort_by_key(|&i| new_order[i]);
    positions.iter().map(|&i| old_order[i]).collect()
}

/// Updates the chirality number for a trigonal-bipyramidal centre whose
/// neighbors have been relabelled; see [`relisted`] for the two orders.
fn update_tb_chirality(n: u8, old_order: &[usize], new_order: &[usize]) -> ChiralClass {
    ChiralClass::TB(remap_tb_number(n, old_order, &relisted(old_order, new_order)).unwrap_or(n))
}

/// Updates the chirality number for a square-planar centre whose neighbors
/// have been relabelled; see [`relisted`] for the two orders.
fn update_sp_chirality(n: u8, old_order: &[usize], new_order: &[usize]) -> ChiralClass {
    ChiralClass::SP(remap_sp_number(n, old_order, &relisted(old_order, new_order)).unwrap_or(n))
}

/// Updates the chirality number for an octahedral centre whose neighbors have
/// been relabelled; see [`relisted`] for the two orders.
fn update_oh_chirality(n: u8, old_order: &[usize], new_order: &[usize]) -> ChiralClass {
    ChiralClass::OH(remap_oh_number(n, old_order, &relisted(old_order, new_order)).unwrap_or(n))
}

/// Converts octahedral configuration parameters into the corresponding OH number (1-30).
///
/// # Arguments
///
/// * `shape` - The shape of the octahedral configuration (U, Z, or Four)
/// * `axis_direction` - The direction of the viewing axis
/// * `winding` - The winding direction (clockwise or counterclockwise)
///
/// # Returns
///
/// A number between 1 and 30 representing the specific octahedral configuration,
/// or 0 if the configuration is invalid.
pub fn oh_number_from_config(
    shape: OctahedralShape,
    axis_direction: AxisDirection,
    winding: WindingDirection,
) -> u8 {
    match (shape, axis_direction, winding) {
        (OctahedralShape::U, AxisDirection::AtoF, WindingDirection::Counterclockwise) => 1,
        (OctahedralShape::U, AxisDirection::AtoF, WindingDirection::Clockwise) => 2,
        (OctahedralShape::U, AxisDirection::AtoE, WindingDirection::Counterclockwise) => 3,
        (OctahedralShape::U, AxisDirection::AtoE, WindingDirection::Clockwise) => 16,
        (OctahedralShape::U, AxisDirection::AtoD, WindingDirection::Counterclockwise) => 6,
        (OctahedralShape::U, AxisDirection::AtoD, WindingDirection::Clockwise) => 18,
        (OctahedralShape::U, AxisDirection::AtoC, WindingDirection::Counterclockwise) => 19,
        (OctahedralShape::U, AxisDirection::AtoC, WindingDirection::Clockwise) => 24,
        (OctahedralShape::U, AxisDirection::AtoB, WindingDirection::Counterclockwise) => 25,
        (OctahedralShape::U, AxisDirection::AtoB, WindingDirection::Clockwise) => 30,
        (OctahedralShape::Z, AxisDirection::AtoF, WindingDirection::Counterclockwise) => 4,
        (OctahedralShape::Z, AxisDirection::AtoF, WindingDirection::Clockwise) => 14,
        (OctahedralShape::Z, AxisDirection::AtoE, WindingDirection::Counterclockwise) => 5,
        (OctahedralShape::Z, AxisDirection::AtoE, WindingDirection::Clockwise) => 15,
        (OctahedralShape::Z, AxisDirection::AtoD, WindingDirection::Counterclockwise) => 7,
        (OctahedralShape::Z, AxisDirection::AtoD, WindingDirection::Clockwise) => 17,
        (OctahedralShape::Z, AxisDirection::AtoC, WindingDirection::Counterclockwise) => 20,
        (OctahedralShape::Z, AxisDirection::AtoC, WindingDirection::Clockwise) => 23,
        (OctahedralShape::Z, AxisDirection::AtoB, WindingDirection::Counterclockwise) => 26,
        (OctahedralShape::Z, AxisDirection::AtoB, WindingDirection::Clockwise) => 29,
        (OctahedralShape::Four, AxisDirection::AtoF, WindingDirection::Counterclockwise) => 10,
        (OctahedralShape::Four, AxisDirection::AtoF, WindingDirection::Clockwise) => 8,
        (OctahedralShape::Four, AxisDirection::AtoE, WindingDirection::Counterclockwise) => 11,
        (OctahedralShape::Four, AxisDirection::AtoE, WindingDirection::Clockwise) => 9,
        (OctahedralShape::Four, AxisDirection::AtoD, WindingDirection::Counterclockwise) => 13,
        (OctahedralShape::Four, AxisDirection::AtoD, WindingDirection::Clockwise) => 12,
        (OctahedralShape::Four, AxisDirection::AtoC, WindingDirection::Counterclockwise) => 22,
        (OctahedralShape::Four, AxisDirection::AtoC, WindingDirection::Clockwise) => 21,
        (OctahedralShape::Four, AxisDirection::AtoB, WindingDirection::Counterclockwise) => 28,
        (OctahedralShape::Four, AxisDirection::AtoB, WindingDirection::Clockwise) => 27,
        _ => 0,
    }
}

/// Converts an OH number (1-30) into the corresponding octahedral configuration parameters.
///
/// # Arguments
///
/// * `oh_number` - A number between 1 and 30 representing a specific octahedral configuration
///
/// # Returns
///
/// A tuple containing the shape, axis direction, and winding direction for the given OH number,
/// or (None, None, None) if the number is invalid.
pub fn oh_config_from_number(oh_number: u8) -> (OctahedralShape, AxisDirection, WindingDirection) {
    match oh_number {
        1 => (
            OctahedralShape::U,
            AxisDirection::AtoF,
            WindingDirection::Counterclockwise,
        ),
        2 => (
            OctahedralShape::U,
            AxisDirection::AtoF,
            WindingDirection::Clockwise,
        ),
        3 => (
            OctahedralShape::U,
            AxisDirection::AtoE,
            WindingDirection::Counterclockwise,
        ),
        4 => (
            OctahedralShape::Z,
            AxisDirection::AtoF,
            WindingDirection::Counterclockwise,
        ),
        5 => (
            OctahedralShape::Z,
            AxisDirection::AtoE,
            WindingDirection::Counterclockwise,
        ),
        6 => (
            OctahedralShape::U,
            AxisDirection::AtoD,
            WindingDirection::Counterclockwise,
        ),
        7 => (
            OctahedralShape::Z,
            AxisDirection::AtoD,
            WindingDirection::Counterclockwise,
        ),
        8 => (
            OctahedralShape::Four,
            AxisDirection::AtoF,
            WindingDirection::Clockwise,
        ),
        9 => (
            OctahedralShape::Four,
            AxisDirection::AtoE,
            WindingDirection::Clockwise,
        ),
        10 => (
            OctahedralShape::Four,
            AxisDirection::AtoF,
            WindingDirection::Counterclockwise,
        ),
        11 => (
            OctahedralShape::Four,
            AxisDirection::AtoE,
            WindingDirection::Counterclockwise,
        ),
        12 => (
            OctahedralShape::Four,
            AxisDirection::AtoD,
            WindingDirection::Clockwise,
        ),
        13 => (
            OctahedralShape::Four,
            AxisDirection::AtoD,
            WindingDirection::Counterclockwise,
        ),
        14 => (
            OctahedralShape::Z,
            AxisDirection::AtoF,
            WindingDirection::Clockwise,
        ),
        15 => (
            OctahedralShape::Z,
            AxisDirection::AtoE,
            WindingDirection::Clockwise,
        ),
        16 => (
            OctahedralShape::U,
            AxisDirection::AtoE,
            WindingDirection::Clockwise,
        ),
        17 => (
            OctahedralShape::Z,
            AxisDirection::AtoD,
            WindingDirection::Clockwise,
        ),
        18 => (
            OctahedralShape::U,
            AxisDirection::AtoD,
            WindingDirection::Clockwise,
        ),
        19 => (
            OctahedralShape::U,
            AxisDirection::AtoC,
            WindingDirection::Counterclockwise,
        ),
        20 => (
            OctahedralShape::Z,
            AxisDirection::AtoC,
            WindingDirection::Counterclockwise,
        ),
        21 => (
            OctahedralShape::Four,
            AxisDirection::AtoC,
            WindingDirection::Clockwise,
        ),
        22 => (
            OctahedralShape::Four,
            AxisDirection::AtoC,
            WindingDirection::Counterclockwise,
        ),
        23 => (
            OctahedralShape::Z,
            AxisDirection::AtoC,
            WindingDirection::Clockwise,
        ),
        24 => (
            OctahedralShape::U,
            AxisDirection::AtoC,
            WindingDirection::Clockwise,
        ),
        25 => (
            OctahedralShape::U,
            AxisDirection::AtoB,
            WindingDirection::Counterclockwise,
        ),
        26 => (
            OctahedralShape::Z,
            AxisDirection::AtoB,
            WindingDirection::Counterclockwise,
        ),
        27 => (
            OctahedralShape::Four,
            AxisDirection::AtoB,
            WindingDirection::Clockwise,
        ),
        28 => (
            OctahedralShape::Four,
            AxisDirection::AtoB,
            WindingDirection::Counterclockwise,
        ),
        29 => (
            OctahedralShape::Z,
            AxisDirection::AtoB,
            WindingDirection::Clockwise,
        ),
        30 => (
            OctahedralShape::U,
            AxisDirection::AtoB,
            WindingDirection::Clockwise,
        ),
        _ => (
            OctahedralShape::None,
            AxisDirection::None,
            WindingDirection::None,
        ),
    }
}
