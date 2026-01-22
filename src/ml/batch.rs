//! Many molecules as one graph, which is how a message-passing network is fed.
//!
//! Molecules have different numbers of atoms, so their feature matrices cannot
//! be stacked into a rectangle the way images can. What is done instead is to
//! lay the graphs side by side as one disconnected graph: concatenate the atom
//! rows, concatenate the bond rows, and shift each molecule's `edge_index` by
//! the number of atom rows already placed. Message passing on the union is
//! message passing on each molecule, because no edge crosses between them, and
//! the whole batch is then one matrix multiply rather than a loop.
//!
//! What is lost by concatenating is which molecule a row belongs to, and that
//! is what [`GraphBatch::batch`] records. Every readout -- summing a molecule's
//! atom vectors into one vector for it -- is a segment operation over that
//! array; see [`GraphBatch::segment_sum`] and its neighbours.
//!
//! ```
//! use oxidane::prelude::*;
//! use oxidane::molecule::featurize::graph_features;
//! use oxidane::ml::batch::GraphBatch;
//!
//! let ethanol = MolecularGraph::from_smiles_single("CCO").unwrap();
//! let benzene = MolecularGraph::from_smiles_single("c1ccccc1").unwrap();
//! let batch = GraphBatch::collate(&[graph_features(&ethanol), graph_features(&benzene)]);
//!
//! assert_eq!(batch.graph_count, 2);
//! assert_eq!(batch.atom_count, 3 + 6);
//! // Benzene's first atom is row 3 of the batch, and its edges point there.
//! assert_eq!(batch.batch[3], 1);
//! assert!(batch.edge_index[batch.edge_offsets[1] as usize * 2] >= 3);
//! ```

use crate::molecule::featurize::{ATOM_FEATURE_WIDTH, BOND_FEATURE_WIDTH, GraphFeatures};
use rustc_hash::FxHashMap;

/// Several molecules' feature matrices concatenated into one graph.
///
/// The layout matches [`GraphFeatures`] exactly -- flat, row-major, `f32` --
/// so a tensor is a view of these vectors rather than a copy of them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphBatch {
    /// How many molecules went in.
    pub graph_count: usize,
    /// Total atom rows across every molecule.
    pub atom_count: usize,
    /// `atom_count` rows of [`ATOM_FEATURE_WIDTH`].
    pub atom_features: Vec<f32>,
    /// Total directed edges across every molecule.
    pub edge_count: usize,
    /// `edge_count` rows of [`BOND_FEATURE_WIDTH`].
    pub bond_features: Vec<f32>,
    /// `edge_count` pairs of batch-wide row numbers, source first.
    pub edge_index: Vec<u32>,
    /// Which molecule each atom row belongs to. Length `atom_count`.
    pub batch: Vec<u32>,
    /// Where each molecule's atom rows start, with a final entry equal to
    /// `atom_count`. Length `graph_count + 1`.
    pub atom_offsets: Vec<u32>,
    /// Where each molecule's edges start, with a final entry equal to
    /// `edge_count`. Length `graph_count + 1`.
    pub edge_offsets: Vec<u32>,
    /// `graph_count` rows of whole-molecule descriptors, or empty if none were
    /// supplied. See [`crate::molecule::featurize::molecule_features`].
    pub global_features: Vec<f32>,
    /// How wide one of those rows is; zero when there are none.
    pub global_width: usize,
}

impl GraphBatch {
    /// Lays out molecules side by side as one graph.
    pub fn collate(graphs: &[GraphFeatures]) -> Self {
        Self::collate_iter(graphs.iter())
    }

    /// [`GraphBatch::collate`] with a row of whole-molecule descriptors beside
    /// each graph.
    ///
    /// The rows must all be the same width, and there must be one per graph.
    pub fn collate_with_globals<'a, I>(graphs: I, globals: &[&[f32]]) -> Self
    where
        I: IntoIterator<Item = &'a GraphFeatures>,
    {
        let mut batch = Self::collate_iter(graphs);
        batch.global_width = globals.first().map_or(0, |row| row.len());
        batch.global_features = Vec::with_capacity(globals.len() * batch.global_width);
        for row in globals {
            debug_assert_eq!(row.len(), batch.global_width);
            batch.global_features.extend_from_slice(row);
        }
        debug_assert_eq!(globals.len(), batch.graph_count);
        batch
    }

    /// [`GraphBatch::collate`] without needing the molecules in a slice first.
    pub fn collate_iter<'a, I>(graphs: I) -> Self
    where
        I: IntoIterator<Item = &'a GraphFeatures>,
    {
        let mut batch = GraphBatch {
            atom_offsets: vec![0],
            edge_offsets: vec![0],
            ..Default::default()
        };

        for graph in graphs {
            let atom_base = batch.atom_count as u32;
            batch.atom_features.extend_from_slice(&graph.atom_features);
            batch.bond_features.extend_from_slice(&graph.bond_features);
            batch
                .edge_index
                .extend(graph.edge_index.iter().map(|&row| row + atom_base));
            batch
                .batch
                .extend(std::iter::repeat_n(batch.graph_count as u32, graph.atom_count));

            batch.atom_count += graph.atom_count;
            batch.edge_count += graph.edge_count;
            batch.graph_count += 1;
            batch.atom_offsets.push(batch.atom_count as u32);
            batch.edge_offsets.push(batch.edge_count as u32);
        }

        batch
    }

    /// One atom's row.
    pub fn atom_row(&self, row: usize) -> &[f32] {
        &self.atom_features[row * ATOM_FEATURE_WIDTH..(row + 1) * ATOM_FEATURE_WIDTH]
    }

    /// One directed edge's row.
    pub fn bond_row(&self, edge: usize) -> &[f32] {
        &self.bond_features[edge * BOND_FEATURE_WIDTH..(edge + 1) * BOND_FEATURE_WIDTH]
    }

    /// The two atoms a directed edge joins, as batch-wide row numbers.
    pub fn edge(&self, edge: usize) -> (usize, usize) {
        (
            self.edge_index[edge * 2] as usize,
            self.edge_index[edge * 2 + 1] as usize,
        )
    }

    /// The rows belonging to one molecule, as a range.
    pub fn graph_rows(&self, graph: usize) -> std::ops::Range<usize> {
        self.atom_offsets[graph] as usize..self.atom_offsets[graph + 1] as usize
    }

    /// How many atom rows one molecule contributed.
    pub fn graph_atom_count(&self, graph: usize) -> usize {
        (self.atom_offsets[graph + 1] - self.atom_offsets[graph]) as usize
    }

    /// Adds each molecule's rows together, giving one vector per molecule.
    ///
    /// `rows` is `atom_count * width` in row-major order -- the atom
    /// representations a network has just computed -- and the result is
    /// `graph_count * width`. This is the readout that turns per-atom vectors
    /// into a per-molecule one.
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::featurize::graph_features;
    /// use oxidane::ml::batch::GraphBatch;
    ///
    /// let batch = GraphBatch::collate(&[
    ///     graph_features(&MolecularGraph::from_smiles_single("CCO").unwrap()),
    ///     graph_features(&MolecularGraph::from_smiles_single("CC").unwrap()),
    /// ]);
    /// // One number per atom, all of them 1: the sum counts each molecule's atoms.
    /// let ones = vec![1.0f32; batch.atom_count];
    /// assert_eq!(batch.segment_sum(&ones, 1), vec![3.0, 2.0]);
    /// ```
    pub fn segment_sum(&self, rows: &[f32], width: usize) -> Vec<f32> {
        debug_assert_eq!(rows.len(), self.atom_count * width);
        let mut pooled = vec![0.0; self.graph_count * width];
        for (row, &graph) in self.batch.iter().enumerate() {
            let target = graph as usize * width;
            for column in 0..width {
                pooled[target + column] += rows[row * width + column];
            }
        }
        pooled
    }

    /// [`GraphBatch::segment_sum`] divided by each molecule's atom count.
    ///
    /// A molecule with no atom rows -- which a batch can contain, since a
    /// hydrogen-only fragment has none -- pools to zeros rather than to a
    /// division by zero.
    pub fn segment_mean(&self, rows: &[f32], width: usize) -> Vec<f32> {
        let mut pooled = self.segment_sum(rows, width);
        for graph in 0..self.graph_count {
            let count = self.graph_atom_count(graph);
            if count == 0 {
                continue;
            }
            let scale = 1.0 / count as f32;
            for column in 0..width {
                pooled[graph * width + column] *= scale;
            }
        }
        pooled
    }

    /// The per-column maximum over each molecule's rows.
    ///
    /// A molecule with no atom rows pools to zeros, not to negative infinity:
    /// the empty maximum has no value, and zero is the one a downstream layer
    /// can carry on from.
    pub fn segment_max(&self, rows: &[f32], width: usize) -> Vec<f32> {
        debug_assert_eq!(rows.len(), self.atom_count * width);
        let mut pooled = vec![f32::NEG_INFINITY; self.graph_count * width];
        for (row, &graph) in self.batch.iter().enumerate() {
            let target = graph as usize * width;
            for column in 0..width {
                let value = rows[row * width + column];
                if value > pooled[target + column] {
                    pooled[target + column] = value;
                }
            }
        }
        for graph in 0..self.graph_count {
            if self.graph_atom_count(graph) == 0 {
                for column in 0..width {
                    pooled[graph * width + column] = 0.0;
                }
            }
        }
        pooled
    }

    /// Adds each edge's message into the atom it points at.
    ///
    /// `messages` is `edge_count * width`; the result is `atom_count * width`.
    /// This is the aggregation step of message passing, and the reason
    /// [`GraphFeatures`] gives every bond twice: a message travels one way
    /// along an edge, so a bond needs an edge in each direction.
    pub fn scatter_edges_to_atoms(&self, messages: &[f32], width: usize) -> Vec<f32> {
        debug_assert_eq!(messages.len(), self.edge_count * width);
        let mut gathered = vec![0.0; self.atom_count * width];
        for edge in 0..self.edge_count {
            let target = self.edge_index[edge * 2 + 1] as usize * width;
            for column in 0..width {
                gathered[target + column] += messages[edge * width + column];
            }
        }
        gathered
    }

    /// For each directed edge, the edge that runs the other way.
    ///
    /// A message-passing scheme that sends a message along every directed edge
    /// has to stop it bouncing straight back: the message leaving `v` towards
    /// `w` must be built from what arrived at `v` from everywhere *except*
    /// `w`. Subtracting the reverse edge's message from the sum of everything
    /// arriving at `v` is how that is done, and this is the index that finds
    /// it.
    ///
    /// Edges are grouped by source atom rather than paired, so an edge and its
    /// reverse are not neighbours and this cannot be arithmetic.
    ///
    /// A bond present in only one direction has no reverse. Its entry points
    /// back at itself and the matching entry of the returned mask is `false`,
    /// so a caller can multiply the subtraction away rather than subtract the
    /// wrong message. Molecules built by this crate are always symmetric, so
    /// the mask is all `true` in practice; it is here because a
    /// [`GraphFeatures`] a caller assembled itself need not be.
    ///
    /// ```
    /// use oxidane::prelude::*;
    /// use oxidane::molecule::featurize::graph_features;
    /// use oxidane::ml::batch::GraphBatch;
    ///
    /// let batch = GraphBatch::collate(&[graph_features(
    ///     &MolecularGraph::from_smiles_single("CCO").unwrap(),
    /// )]);
    /// let (reverse, present) = batch.reverse_edges();
    /// assert!(present.iter().all(|&found| found));
    /// for edge in 0..batch.edge_count {
    ///     let (from, to) = batch.edge(edge);
    ///     assert_eq!(batch.edge(reverse[edge] as usize), (to, from));
    /// }
    /// ```
    pub fn reverse_edges(&self) -> (Vec<u32>, Vec<bool>) {
        let mut by_ends: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        by_ends.reserve(self.edge_count);
        for edge in 0..self.edge_count {
            by_ends.insert(
                (self.edge_index[edge * 2], self.edge_index[edge * 2 + 1]),
                edge as u32,
            );
        }

        let mut reverse = Vec::with_capacity(self.edge_count);
        let mut present = Vec::with_capacity(self.edge_count);
        for edge in 0..self.edge_count {
            let ends = (self.edge_index[edge * 2 + 1], self.edge_index[edge * 2]);
            match by_ends.get(&ends) {
                Some(&other) => {
                    reverse.push(other);
                    present.push(true);
                }
                None => {
                    reverse.push(edge as u32);
                    present.push(false);
                }
            }
        }
        (reverse, present)
    }

    /// The source atom of every edge, as its own array.
    ///
    /// `edge_index` interleaves the two ends, and an index tensor wants them
    /// apart.
    pub fn edge_sources(&self) -> Vec<u32> {
        self.edge_index.iter().step_by(2).copied().collect()
    }

    /// The atom every edge points at, as its own array.
    pub fn edge_targets(&self) -> Vec<u32> {
        self.edge_index.iter().skip(1).step_by(2).copied().collect()
    }

    /// How many edges point at each atom row.
    ///
    /// The denominator for a mean aggregation, and the degree a normalised
    /// adjacency needs.
    pub fn in_degrees(&self) -> Vec<u32> {
        let mut degrees = vec![0; self.atom_count];
        for edge in 0..self.edge_count {
            degrees[self.edge_index[edge * 2 + 1] as usize] += 1;
        }
        degrees
    }
}
