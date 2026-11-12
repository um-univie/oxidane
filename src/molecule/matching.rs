//! Maximum matching in a general graph, by Edmonds' blossom algorithm.
//!
//! Kekulization is a matching problem and nothing less will do. A greedy search
//! that gives up when an atom has no free partner does not find the largest set
//! of π bonds, and the difference is not cosmetic: every atom it fails to pair
//! becomes an unpaired electron, so a greedy answer invents radicals that a
//! maximum matching would have paired elsewhere. Seventeen thousand molecules
//! in the test corpus grew radicals that way.
//!
//! The size of a maximum matching is the same for every maximum matching, so
//! the *number* of unpaired electrons a molecule is left with is a property of
//! the molecule rather than of the search. That is what makes radical counts
//! canonical, and it is why this has to be exact rather than nearly right.
//!
//! Odd cycles are what makes general-graph matching hard, and rings are exactly
//! that. Blossom contraction is the standard answer: an odd cycle reachable by
//! an alternating path is contracted to a single vertex, the search continues
//! on the smaller graph, and the path is expanded again on the way out.

/// A maximum matching of `vertex_count` vertices with the given edges.
///
/// Returns, for each vertex, the vertex it is matched to, or `None`.
/// Deterministic: vertices are scanned in index order, so the caller decides
/// the outcome among equally large matchings by choosing that order.
pub fn maximum_matching(vertex_count: usize, edges: &[(usize, usize)]) -> Vec<Option<usize>> {
    maximum_matching_from(vertex_count, edges, &[])
}

/// [`maximum_matching`], starting from a matching the caller already holds.
///
/// A search that begins from nothing has to augment once per matched pair, and
/// each augmentation sweeps the graph; beginning from a matching that is
/// already nearly perfect leaves only the few pairs that differ. That is the
/// difference between solving a problem and re-solving one: the lexicographic
/// pass in [`lexicographic_degree_constrained_subgraph`] runs the solver once
/// per edge it decides, and every run but the first differs from the one before
/// it in a single edge. PubChem's catenated sulfur -- `S=S=S=` on for
/// seventy-six atoms -- is what made the difference visible: one such record
/// cost minutes.
///
/// `seed[v]` is the vertex `v` is matched to, `None` where it is free. Entries
/// that do not agree with each other, name a vertex out of range, or name an
/// edge the graph does not have are dropped, so a stale or partial seed is
/// safe: the result is a maximum matching either way, and only the time it
/// takes depends on the seed.
pub fn maximum_matching_from(
    vertex_count: usize,
    edges: &[(usize, usize)],
    seed: &[Option<usize>],
) -> Vec<Option<usize>> {
    let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); vertex_count];
    for (edge, &(a, b)) in edges.iter().enumerate() {
        if a < vertex_count && b < vertex_count && a != b {
            adjacency[a].push((b, edge));
            adjacency[b].push((a, edge));
        }
    }
    for neighbours in adjacency.iter_mut() {
        neighbours.sort_unstable();
        neighbours.dedup_by_key(|(neighbour, _)| *neighbour);
    }
    let alive = vec![true; edges.len()];

    let mut matched: Vec<Option<usize>> = vec![None; vertex_count];

    // The seed first, where there is one: only pairs that agree from both ends
    // and really are edges of this graph are taken, so nothing here can produce
    // a matching the graph does not have.
    for (vertex, &partner) in seed.iter().enumerate().take(vertex_count) {
        let Some(partner) = partner else { continue };
        if partner >= vertex_count
            || partner == vertex
            || matched[vertex].is_some()
            || matched[partner].is_some()
            || seed.get(partner).copied().flatten() != Some(vertex)
            || !adjacency[vertex]
                .iter()
                .any(|&(neighbour, _)| neighbour == partner)
        {
            continue;
        }
        matched[vertex] = Some(partner);
        matched[partner] = Some(vertex);
    }

    // A greedy start costs nothing and saves augmenting for the easy pairs.
    for vertex in 0..vertex_count {
        if matched[vertex].is_some() {
            continue;
        }
        if let Some(&(free, _)) = adjacency[vertex]
            .iter()
            .find(|&&(neighbour, _)| matched[neighbour].is_none())
        {
            matched[vertex] = Some(free);
            matched[free] = Some(vertex);
        }
    }

    // A vertex with no edge has no augmenting path, and searching from one
    // still costs a full sweep of the graph. Callers hand this the whole
    // molecule while the edges cover only its aromatic bonds, so almost every
    // vertex is isolated and almost every search was wasted. Greedy leaves
    // nothing to search for in a ring that alternates cleanly, which is most of
    // them, and then the search state is never built at all.
    let roots: Vec<usize> = (0..vertex_count)
        .filter(|&vertex| matched[vertex].is_none() && !adjacency[vertex].is_empty())
        .collect();
    if roots.is_empty() {
        return matched;
    }

    let adjacency = Adjacency::from_lists(&adjacency);
    let mut workspace = SearchWorkspace::new(vertex_count);
    let mut state = Search::new(&adjacency, &alive, &mut workspace);

    for root in roots {
        if matched[root].is_some() {
            continue;
        }
        if let Some(last) = state.find_augmenting_path(root, &matched) {
            augment(&mut matched, state.parent, last);
        }
    }
    matched
}

/// `(neighbour, edge)` per vertex, every vertex's list in one allocation.
///
/// The f-factor gadget has some seven times the vertices of the problem it
/// stands for, and a list apiece made building it cost more than solving it.
struct Adjacency {
    start: Vec<usize>,
    entries: Vec<(usize, usize)>,
}

impl Adjacency {
    /// Each edge listed at both its ends, in the order the edges are given.
    fn from_edges(vertex_count: usize, edges: &[(usize, usize)]) -> Self {
        let mut start = vec![0usize; vertex_count + 1];
        for &(a, b) in edges {
            start[a + 1] += 1;
            start[b + 1] += 1;
        }
        for vertex in 0..vertex_count {
            start[vertex + 1] += start[vertex];
        }
        let mut next = start.clone();
        let mut entries = vec![(0, 0); start[vertex_count]];
        for (edge, &(a, b)) in edges.iter().enumerate() {
            entries[next[a]] = (b, edge);
            next[a] += 1;
            entries[next[b]] = (a, edge);
            next[b] += 1;
        }
        Self { start, entries }
    }

    fn from_lists(lists: &[Vec<(usize, usize)>]) -> Self {
        let mut start = Vec::with_capacity(lists.len() + 1);
        start.push(0);
        let mut entries = Vec::with_capacity(lists.iter().map(Vec::len).sum());
        for list in lists {
            entries.extend_from_slice(list);
            start.push(entries.len());
        }
        Self { start, entries }
    }

    fn len(&self) -> usize {
        self.start.len().saturating_sub(1)
    }
}

impl std::ops::Index<usize> for Adjacency {
    type Output = [(usize, usize)];

    fn index(&self, vertex: usize) -> &Self::Output {
        &self.entries[self.start[vertex]..self.start[vertex + 1]]
    }
}

/// Owned by the matching problem, not by an individual augmentation. Repairs
/// edit the same graph repeatedly and can reuse all of these allocations.
struct SearchWorkspace {
    parent: Vec<usize>,
    base: Vec<usize>,
    in_queue: Vec<bool>,
    blossom: Vec<bool>,
    queue: Vec<usize>,
    seen: Vec<bool>,
    touched: Vec<usize>,
}

impl SearchWorkspace {
    fn new(vertex_count: usize) -> Self {
        Self {
            parent: vec![usize::MAX; vertex_count],
            base: vec![0; vertex_count],
            in_queue: vec![false; vertex_count],
            blossom: vec![false; vertex_count],
            queue: Vec::with_capacity(vertex_count),
            seen: vec![false; vertex_count],
            touched: Vec::new(),
        }
    }
}

struct Search<'a> {
    vertex_count: usize,
    adjacency: &'a Adjacency,
    /// The factor graph stays fixed while decisions disable individual edges.
    alive: &'a [bool],
    parent: &'a mut Vec<usize>,
    base: &'a mut Vec<usize>,
    in_queue: &'a mut Vec<bool>,
    blossom: &'a mut Vec<bool>,
    queue: &'a mut Vec<usize>,
    /// LCA scratch is cleared through `touched`, even across failed repairs.
    seen: &'a mut Vec<bool>,
    touched: &'a mut Vec<usize>,
}

impl<'a> Search<'a> {
    fn new(
        adjacency: &'a Adjacency,
        alive: &'a [bool],
        workspace: &'a mut SearchWorkspace,
    ) -> Self {
        let SearchWorkspace {
            parent,
            base,
            in_queue,
            blossom,
            queue,
            seen,
            touched,
        } = workspace;
        Self {
            vertex_count: adjacency.len(),
            adjacency,
            alive,
            parent,
            base,
            in_queue,
            blossom,
            queue,
            seen,
            touched,
        }
    }
}

impl Search<'_> {
    /// The lowest common ancestor of two vertices in the alternating forest.
    fn lowest_common_ancestor(
        &mut self,
        matched: &[Option<usize>],
        mut a: usize,
        mut b: usize,
    ) -> usize {
        for &vertex in self.touched.iter() {
            self.seen[vertex] = false;
        }
        self.touched.clear();
        loop {
            a = self.base[a];
            self.seen[a] = true;
            self.touched.push(a);
            match matched[a] {
                Some(partner) => a = self.parent[partner],
                None => break,
            }
        }
        loop {
            b = self.base[b];
            if self.seen[b] {
                return b;
            }
            b = self.parent[matched[b].expect("only matched vertices continue the path")];
        }
    }

    /// Marks every vertex on the odd cycle between `v` and the blossom's base.
    fn mark_path(
        &mut self,
        matched: &[Option<usize>],
        mut v: usize,
        base: usize,
        mut child: usize,
    ) {
        while self.base[v] != base {
            let partner = matched[v].expect("a blossom alternates, so this is matched");
            self.blossom[self.base[v]] = true;
            self.blossom[self.base[partner]] = true;
            self.parent[v] = child;
            child = partner;
            v = self.parent[partner];
        }
    }

    /// Grows the alternating tree from `root`, returning the far end of an
    /// augmenting path if one is found.
    fn find_augmenting_path(&mut self, root: usize, matched: &[Option<usize>]) -> Option<usize> {
        self.parent.iter_mut().for_each(|p| *p = usize::MAX);
        self.in_queue.iter_mut().for_each(|q| *q = false);
        for (vertex, base) in self.base.iter_mut().enumerate() {
            *base = vertex;
        }
        self.queue.clear();
        self.queue.push(root);
        self.in_queue[root] = true;

        let mut head = 0;
        while head < self.queue.len() {
            let v = self.queue[head];
            head += 1;

            for index in 0..self.adjacency[v].len() {
                let (to, edge) = self.adjacency[v][index];
                if !self.alive[edge] {
                    continue;
                }

                // Same blossom, or the edge we came in on: nothing to learn.
                if self.base[v] == self.base[to] || matched[v] == Some(to) {
                    continue;
                }

                if to == root
                    || (matched[to].is_some() && self.parent[matched[to].unwrap()] != usize::MAX)
                {
                    // An odd cycle: contract it and carry on.
                    let base = self.lowest_common_ancestor(matched, v, to);
                    self.blossom.iter_mut().for_each(|b| *b = false);
                    self.mark_path(matched, v, base, to);
                    self.mark_path(matched, to, base, v);
                    for vertex in 0..self.vertex_count {
                        if self.blossom[self.base[vertex]] {
                            self.base[vertex] = base;
                            if !self.in_queue[vertex] {
                                self.in_queue[vertex] = true;
                                self.queue.push(vertex);
                            }
                        }
                    }
                } else if self.parent[to] == usize::MAX {
                    self.parent[to] = v;
                    match matched[to] {
                        None => return Some(to), // augmenting path found
                        Some(partner) => {
                            if !self.in_queue[partner] {
                                self.in_queue[partner] = true;
                                self.queue.push(partner);
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

/// Flips the matching along the augmenting path ending at `last`.
fn augment(matched: &mut [Option<usize>], parent: &[usize], last: usize) {
    let mut v = last;
    while v != usize::MAX {
        let previous = parent[v];
        let next = matched[previous];
        matched[v] = Some(previous);
        matched[previous] = Some(v);
        v = match next {
            Some(vertex) => vertex,
            None => break,
        };
    }
}

/// Selects a subset of `edges` giving every vertex exactly the degree
/// `required_degree` asks of it, or `None` when no such subset exists.
///
/// This is the *degree-constrained subgraph* problem, also called an f-factor,
/// and it is what a valence assignment is once the skeleton is fixed: each atom
/// wants a definite number of π bonds, each bond either carries one or does
/// not, and the question is whether the wants can all be met at once. A maximum
/// matching is the special case where every vertex wants one.
///
/// Solved exactly, by Tutte's reduction to a perfect matching on a general
/// graph. A vertex `u` of free degree `d` wanting `b` becomes `d` *external*
/// vertices -- one per incident edge -- and `d - b` *internal* ones, with every
/// external joined to every internal; each original edge becomes an edge
/// between the two externals that stand for it. In a perfect matching every
/// external is matched either across its own edge or to an internal, and the
/// internals absorb exactly `d - b` of them, so exactly `b` edges are taken at
/// `u`. Since [`maximum_matching`] is an exact blossom implementation, so is
/// this.
///
/// Returns one selection per edge, in the order the edges were given.
///
/// ```
/// use oxidane::molecule::matching::degree_constrained_subgraph;
///
/// // A four-cycle where every vertex wants one edge: the two opposite pairs.
/// let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
/// let taken = degree_constrained_subgraph(4, &edges, &[1, 1, 1, 1]).unwrap();
/// assert_eq!(taken.iter().filter(|&&t| t).count(), 2);
///
/// // A triangle cannot: three vertices wanting one edge each is an odd sum.
/// let triangle = [(0, 1), (1, 2), (2, 0)];
/// assert!(degree_constrained_subgraph(3, &triangle, &[1, 1, 1]).is_none());
/// ```
pub fn degree_constrained_subgraph(
    vertex_count: usize,
    edges: &[(usize, usize)],
    required_degree: &[usize],
) -> Option<Vec<bool>> {
    let forced = vec![None; edges.len()];
    degree_constrained_subgraph_with(vertex_count, edges, required_degree, &forced, None)
}

/// [`degree_constrained_subgraph`] with some edges decided in advance.
///
/// `forced[e]` of `Some(true)` insists edge `e` is taken, `Some(false)` that it
/// is not, and `None` leaves it to the solver. `None` is returned when no
/// selection honours every forcing, which is what makes this usable as a
/// feasibility oracle: fixing one variable at a time and asking whether the
/// rest can still be satisfied is how a *particular* solution gets picked out
/// of the many an f-factor problem usually has.
pub fn degree_constrained_subgraph_with(
    vertex_count: usize,
    edges: &[(usize, usize)],
    required_degree: &[usize],
    forced: &[Option<bool>],
    hint: Option<&[bool]>,
) -> Option<Vec<bool>> {
    if required_degree.len() != vertex_count || forced.len() != edges.len() {
        return None;
    }

    // A forced edge is not part of the gadget: it just spends one unit of the
    // degree each of its ends was asking for.
    let mut remaining: Vec<usize> = required_degree.to_vec();
    let mut selection = vec![false; edges.len()];
    let mut free: Vec<usize> = Vec::with_capacity(edges.len());
    for (index, &(first, second)) in edges.iter().enumerate() {
        if first >= vertex_count || second >= vertex_count || first == second {
            return None;
        }
        match forced[index] {
            Some(true) => {
                if remaining[first] == 0 || remaining[second] == 0 {
                    return None;
                }
                remaining[first] -= 1;
                remaining[second] -= 1;
                selection[index] = true;
            }
            Some(false) => (),
            None => free.push(index),
        }
    }

    // Externals, one per end of each free edge; then internals, `d - b` per
    // vertex. A vertex wanting more than its free degree can supply is the
    // cheap infeasibility, and catching it here keeps the gadget well formed.
    let mut free_degree = vec![0usize; vertex_count];
    for &index in &free {
        let (first, second) = edges[index];
        free_degree[first] += 1;
        free_degree[second] += 1;
    }
    for vertex in 0..vertex_count {
        if remaining[vertex] > free_degree[vertex] {
            return None;
        }
    }

    let mut external_of: Vec<(usize, usize)> = vec![(usize::MAX, usize::MAX); edges.len()];
    let mut externals_of_vertex: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];
    let mut gadget_vertices = 0usize;
    for &index in &free {
        let (first, second) = edges[index];
        let external_first = gadget_vertices;
        let external_second = gadget_vertices + 1;
        gadget_vertices += 2;
        externals_of_vertex[first].push(external_first);
        externals_of_vertex[second].push(external_second);
        external_of[index] = (external_first, external_second);
    }

    let mut gadget_edges: Vec<(usize, usize)> =
        free.iter().map(|&index| external_of[index]).collect();
    for vertex in 0..vertex_count {
        let internal_count = free_degree[vertex] - remaining[vertex];
        for _ in 0..internal_count {
            let internal = gadget_vertices;
            gadget_vertices += 1;
            for &external in &externals_of_vertex[vertex] {
                gadget_edges.push((external, internal));
            }
        }
    }

    if gadget_vertices == 0 {
        return Some(selection);
    }
    if !gadget_vertices.is_multiple_of(2) {
        return None;
    }

    // A selection the caller already has says which of the gadget's edges were
    // taken last time, and a nearly-right matching is most of the answer. It is
    // only a starting point: a hint that does not fit the forcings in hand is
    // dropped pair by pair and the search fills in the rest.
    let mut seed: Vec<Option<usize>> = Vec::new();
    if let Some(hint) = hint {
        seed.resize(gadget_vertices, None);
        for &index in &free {
            if hint.get(index).copied() != Some(true) {
                continue;
            }
            let (external_first, external_second) = external_of[index];
            seed[external_first] = Some(external_second);
            seed[external_second] = Some(external_first);
        }
    }

    let matched = maximum_matching_from(gadget_vertices, &gadget_edges, &seed);
    if matched.iter().any(Option::is_none) {
        return None;
    }

    for &index in &free {
        let (external_first, external_second) = external_of[index];
        if matched[external_first] == Some(external_second) {
            selection[index] = true;
        }
    }
    Some(selection)
}

/// The selection an f-factor problem has that is lexicographically first under
/// `priority`, or `None` when it has none at all.
///
/// `priority` lists `(edge, wanted)` most-preferred first; each in turn is
/// forced to `wanted` if the remainder of the problem can still be solved that
/// way, and to the opposite otherwise. Every edge not named is left to the
/// solver.
///
/// Preferring an edge *out* is as useful as preferring one in, and is what lets
/// a constraint be imposed rather than merely an objective approximated:
/// forbidding the π bonds that would leave a ring obliges the ring to satisfy
/// itself, which is exactly a perfect matching over its own bonds -- an
/// aromatic Kekulé structure. "Maximise aromatic rings" is a count no
/// lexicographic pass can express, but "this ring keeps its π" is a set of
/// forcings, and this takes them.
///
/// This is what makes an f-factor answer *canonical* rather than merely valid.
/// A symmetric molecule has many equally good valence assignments and a bare
/// solver returns whichever one its vertex numbering led it to; deciding the
/// edges in an order that is itself a property of the molecule removes the
/// choice. Cost is one solve per named edge, so the whole thing stays
/// polynomial.
pub fn lexicographic_degree_constrained_subgraph(
    vertex_count: usize,
    edges: &[(usize, usize)],
    required_degree: &[usize],
    priority: &[(usize, bool)],
) -> Option<Vec<bool>> {
    let mut instance = DegreeConstrainedInstance::build(vertex_count, edges, required_degree)?;
    Some(instance.select(priority))
}

/// One f-factor problem, solved once and then asked for several selections.
///
/// The graph an instance is built from does not depend on the order the edges
/// are decided in, and neither does the first perfect matching found over it --
/// only the lexicographic pass does. A caller with several orders to try
/// therefore pays for the gadget and for that matching once rather than once
/// per order, which is what `normalize` does: it solves one component under up
/// to four bond orderings and asks the same component two further questions
/// about every double bond whose geometry might survive erasure. Rebuilding for
/// each was the same graph constructed and matched a dozen times over.
///
/// A pass edits the gadget as it goes -- a decision switches edges off and
/// repairs the matching along one alternating path -- so the state it starts
/// from is kept aside and copied back before each. That copy is two vectors of
/// plain data, against a rebuild that allocates an adjacency list per gadget
/// vertex and runs a matching from nothing.
pub struct DegreeConstrainedInstance {
    factor: Factor,
    /// The gadget and the matching as the first solve left them.
    settled_alive: Vec<bool>,
    settled_matching: Vec<Option<usize>>,
    edge_count: usize,
    decided: Vec<bool>,
    /// The problem's own edges, by the vertices they join.
    ends: Vec<(usize, usize)>,
    /// How many more edges each vertex can still refuse (`[0]`) and still take
    /// (`[1]`) before its degree is out of reach, at the start of a pass and as
    /// the pass goes.
    settled_room: Vec<[usize; 2]>,
    room: Vec<[usize; 2]>,
    /// Consecutive edges joining the same two vertices share a class, numbered
    /// here. Parallel edges listed apart only miss the shortcut below; the
    /// normalizer lists each atom's offers to one pool, and each bond's two π
    /// units, together.
    class: Vec<usize>,
    /// Per class, whether this pass has already found taking (`[1]`) or
    /// leaving (`[0]`) one of its edges out of reach.
    out_of_reach: Vec<[bool; 2]>,
}

impl DegreeConstrainedInstance {
    /// Builds the gadget and solves it once, or `None` where the problem has no
    /// solution at all -- which no ordering of its edges can change.
    pub fn build(
        vertex_count: usize,
        edges: &[(usize, usize)],
        required_degree: &[usize],
    ) -> Option<Self> {
        let mut factor = Factor::build(vertex_count, edges, required_degree)?;
        factor.solve()?;
        // A solved problem has every degree within reach, so none of these
        // subtractions can go below zero.
        let mut settled_room: Vec<[usize; 2]> = required_degree
            .iter()
            .map(|&required| [0, required])
            .collect();
        for &(first, second) in edges {
            settled_room[first][0] += 1;
            settled_room[second][0] += 1;
        }
        for room in settled_room.iter_mut() {
            room[0] -= room[1];
        }
        let joins = |index: usize| {
            let (first, second) = edges[index];
            (first.min(second), first.max(second))
        };
        let mut class = vec![0usize; edges.len()];
        let mut class_count = 0;
        for index in 0..edges.len() {
            if index == 0 || joins(index - 1) != joins(index) {
                class_count += 1;
            }
            class[index] = class_count - 1;
        }
        Some(Self {
            class,
            out_of_reach: vec![[false; 2]; class_count],
            settled_alive: factor.alive.clone(),
            settled_matching: factor.matched.clone(),
            edge_count: edges.len(),
            decided: vec![false; edges.len()],
            ends: edges.to_vec(),
            room: settled_room.clone(),
            settled_room,
            factor,
        })
    }

    /// The selection that is lexicographically first under `priority`.
    ///
    /// `priority` lists `(edge, wanted)` most-preferred first; each in turn is
    /// forced to `wanted` if the remainder of the problem can still be solved
    /// that way, and to the opposite otherwise. Every edge not named is left to
    /// the solver.
    pub fn select(&mut self, priority: &[(usize, bool)]) -> Vec<bool> {
        self.factor.alive.copy_from_slice(&self.settled_alive);
        self.factor.matched.copy_from_slice(&self.settled_matching);
        self.decided.iter_mut().for_each(|seen| *seen = false);
        self.room.copy_from_slice(&self.settled_room);
        self.out_of_reach
            .iter_mut()
            .for_each(|seen| *seen = [false; 2]);
        for &(index, wanted) in priority {
            if index >= self.edge_count || self.decided[index] {
                continue;
            }
            self.decided[index] = true;
            // The matching in hand already honours every decision made so far,
            // so when it agrees with this one it is itself the witness that the
            // decision is feasible and nothing has to be searched for. Most do
            // agree -- the order exists to express a preference, and a
            // preference is usually already met.
            let class = self.class[index];
            let side = usize::from(wanted);
            let taken = if self.factor.selects(index) == wanted {
                self.factor.commit(index, wanted);
                wanted
            } else if self.has_room(index, wanted)
                && !self.out_of_reach[class][side]
                && self.factor.force(index, wanted)
            {
                wanted
            } else {
                // A parallel edge decided later is this one again under more
                // constraints: were it to have a solution, swapping the two
                // edges would give this one a solution now. So the answer
                // stands for the whole class, and the normalizer's parallel
                // edges are many -- one per π unit, hydrogen and charge an atom
                // is offered -- and nearly half its failed searches repeated
                // one already made.
                self.out_of_reach[class][side] = true;
                self.factor.commit(index, !wanted);
                !wanted
            };
            let (first, second) = self.ends[index];
            let side = usize::from(taken);
            self.room[first][side] -= 1;
            self.room[second][side] -= 1;
        }
        (0..self.edge_count)
            .map(|index| self.factor.selects(index))
            .collect()
    }

    /// Whether both ends of an edge can still afford the decision, counting
    /// only the decisions already made at each of them.
    ///
    /// A `force` the matching cannot make costs a search of everything the
    /// freed vertex can reach before it fails, and four in five that the
    /// normalizer asks for do fail -- the priority list offers every π unit,
    /// charge and hydrogen slot in turn, and most go to an atom that has already
    /// taken all it can. Such a decision is out of reach at a glance: a vertex
    /// whose every remaining unit is spoken for cannot take another edge, and
    /// one that has refused all it can spare cannot refuse another. Saying no
    /// without the search is the same answer `force` would give.
    fn has_room(&self, index: usize, wanted: bool) -> bool {
        let (first, second) = self.ends[index];
        let side = usize::from(wanted);
        self.room[first][side] > 0 && self.room[second][side] > 0
    }
}

/// What one edge of an f-factor problem is in the gadget.
#[derive(Clone, Copy)]
enum Ends {
    /// Two externals, `first` standing for the edge at its first vertex, joined
    /// by gadget edge `pair`; the edge is taken when they are matched together.
    Pair {
        first: usize,
        second: usize,
        pair: usize,
    },
    /// One external shared by both ends, between the internals of one vertex
    /// and the copies of `supplier`; the edge is taken when it goes to a copy.
    Shared { external: usize, supplier: usize },
}

/// The gadget an f-factor problem reduces to, built once and then edited.
///
/// The reduction is the usual one: each edge of the problem becomes a pair of
/// *external* vertices joined to each other, and each vertex of the problem
/// contributes as many *internal* vertices as it has degree to spare, joined to
/// every external of its own. A perfect matching of the gadget is exactly a
/// selection meeting every degree: an edge is taken when its two externals are
/// matched to each other, and left when each goes to an internal instead.
///
/// A vertex wanting fewer edges than it refuses is cheaper the other way
/// round: `b` *copies* of it, each joined to every external of its own, take
/// exactly `b` of its edges, where Tutte's `d - b` internals would each be
/// joined to all `d`. The pools the normalizer draws hydrogens and charges from
/// are such vertices -- dozens of offers for a handful of placements -- and
/// they were most of the gadget. An edge between a copied vertex and an
/// internal one needs only one external, joined to the internals of one end
/// and the copies of the other: absorbed by an internal, it is refused at both
/// ends; held by a copy, taken at both. Two copied vertices are never made
/// neighbours, so every edge is one of the two kinds.
///
/// The point of holding it rather than rebuilding it is the lexicographic pass,
/// which decides one edge at a time and asks after each whether the rest can
/// still be solved. Rebuilding meant a matching from nothing per question --
/// one sweep of the graph per matched pair -- where the answer differs from the
/// one before it by a single edge. Here a decision switches two or three gadget
/// edges off and repairs the matching along one alternating path, which is one
/// sweep in total. PubChem's catenated sulfur is what made the difference
/// visible: `S=S=` repeated seventy-six times is a single conjugated group, its
/// gadget has some six thousand vertices, and the pass asks a thousand
/// questions of it. That record cost minutes; it now costs half a second.
struct Factor {
    /// `(neighbour, gadget edge)` per gadget vertex.
    adjacency: Adjacency,
    /// Which gadget edges are still in the graph.
    alive: Vec<bool>,
    /// Gadget vertices, externals first, then each problem vertex's own.
    vertex_count: usize,
    /// What each problem edge became.
    ends: Vec<Ends>,
    /// The problem vertex an internal or a copy belongs to, and `usize::MAX`
    /// for an external.
    owner: Vec<usize>,
    matched: Vec<Option<usize>>,
    workspace: SearchWorkspace,
    switched: Vec<usize>,
}

impl Factor {
    fn build(
        vertex_count: usize,
        edges: &[(usize, usize)],
        required_degree: &[usize],
    ) -> Option<Self> {
        if required_degree.len() != vertex_count {
            return None;
        }
        let mut degree = vec![0usize; vertex_count];
        for &(first, second) in edges {
            if first >= vertex_count || second >= vertex_count || first == second {
                return None;
            }
            degree[first] += 1;
            degree[second] += 1;
        }
        // A vertex asking for more than its edges can give is the cheap
        // infeasibility, and catching it here keeps the gadget well formed.
        for vertex in 0..vertex_count {
            if required_degree[vertex] > degree[vertex] {
                return None;
            }
        }

        // Copies where they are fewer than internals would be, largest vertex
        // first, and never for two neighbours.
        let mut by_degree: Vec<usize> = (0..vertex_count)
            .filter(|&vertex| required_degree[vertex] < degree[vertex] - required_degree[vertex])
            .collect();
        by_degree.sort_unstable_by_key(|&vertex| std::cmp::Reverse(degree[vertex]));
        let mut copied = vec![false; vertex_count];
        let mut beside_copied = vec![false; vertex_count];
        for vertex in by_degree {
            if beside_copied[vertex] {
                continue;
            }
            copied[vertex] = true;
            for &(first, second) in edges {
                if first == vertex {
                    beside_copied[second] = true;
                } else if second == vertex {
                    beside_copied[first] = true;
                }
            }
        }

        // Externals first, in edge order; then, vertex by vertex, its
        // internals or its copies, each joined to every external of its own.
        let mut external_start = vec![0usize; vertex_count + 1];
        for vertex in 0..vertex_count {
            external_start[vertex + 1] = external_start[vertex] + degree[vertex];
        }
        let mut next_external = external_start.clone();
        let mut externals = vec![0usize; external_start[vertex_count]];
        let mut push_external = |vertex: usize, external: usize| {
            externals[next_external[vertex]] = external;
            next_external[vertex] += 1;
        };
        let mut gadget_vertices = 0;
        let mut gadget_edges: Vec<(usize, usize)> = Vec::new();
        let mut ends: Vec<Ends> = Vec::with_capacity(edges.len());
        for &(first, second) in edges {
            if copied[first] || copied[second] {
                let external = gadget_vertices;
                gadget_vertices += 1;
                push_external(first, external);
                push_external(second, external);
                let supplier = if copied[first] { first } else { second };
                ends.push(Ends::Shared { external, supplier });
            } else {
                let (a, b) = (gadget_vertices, gadget_vertices + 1);
                gadget_vertices += 2;
                push_external(first, a);
                push_external(second, b);
                ends.push(Ends::Pair {
                    first: a,
                    second: b,
                    pair: gadget_edges.len(),
                });
                gadget_edges.push((a, b));
            }
        }
        let own_count = |vertex: usize| {
            if copied[vertex] {
                required_degree[vertex]
            } else {
                degree[vertex] - required_degree[vertex]
            }
        };
        let mut owner = vec![usize::MAX; gadget_vertices];
        let mut first_own = vec![0usize; vertex_count];
        for vertex in 0..vertex_count {
            first_own[vertex] = gadget_vertices;
            for _ in 0..own_count(vertex) {
                let own = gadget_vertices;
                gadget_vertices += 1;
                owner.push(vertex);
                for &external in &externals[external_start[vertex]..external_start[vertex + 1]] {
                    gadget_edges.push((external, own));
                }
            }
        }
        if gadget_vertices == 0 {
            return Some(Factor {
                adjacency: Adjacency::from_edges(0, &[]),
                alive: Vec::new(),
                vertex_count: 0,
                ends,
                owner,
                matched: Vec::new(),
                workspace: SearchWorkspace::new(0),
                switched: Vec::new(),
            });
        }
        if !gadget_vertices.is_multiple_of(2) {
            return None;
        }

        let adjacency = Adjacency::from_edges(gadget_vertices, &gadget_edges);

        // A first selection made on the problem itself: an edge is taken while
        // both its ends still want one, and otherwise each end's external goes
        // to a spare internal of its own (a shared external goes to a copy if
        // taken and an internal if not). Most degrees come out right this way,
        // so the blossom search only has the shortfall to repair. Matching
        // externals to whatever neighbour is free instead takes every edge,
        // leaves every internal unmatched, and costs one search per pair of
        // them -- seven in ten of all the searches a normalization made.
        let mut matched = vec![None; gadget_vertices];
        let mut wanted = required_degree.to_vec();
        let mut spare: Vec<usize> = first_own.clone();
        let mut pair_with_spare =
            |matched: &mut Vec<Option<usize>>, external: usize, vertex: usize| {
                if spare[vertex] < first_own[vertex] + own_count(vertex) {
                    matched[external] = Some(spare[vertex]);
                    matched[spare[vertex]] = Some(external);
                    spare[vertex] += 1;
                }
            };
        for (index, &(first, second)) in edges.iter().enumerate() {
            let take = wanted[first] > 0 && wanted[second] > 0;
            if take {
                wanted[first] -= 1;
                wanted[second] -= 1;
            }
            match ends[index] {
                Ends::Pair {
                    first: a,
                    second: b,
                    ..
                } => {
                    if take {
                        matched[a] = Some(b);
                        matched[b] = Some(a);
                    } else {
                        pair_with_spare(&mut matched, a, first);
                        pair_with_spare(&mut matched, b, second);
                    }
                }
                Ends::Shared { external, supplier } => {
                    let absorber = if supplier == first { second } else { first };
                    pair_with_spare(
                        &mut matched,
                        external,
                        if take { supplier } else { absorber },
                    );
                }
            }
        }
        Some(Factor {
            alive: vec![true; gadget_edges.len()],
            adjacency,
            vertex_count: gadget_vertices,
            ends,
            owner,
            matched,
            workspace: SearchWorkspace::new(gadget_vertices),
            switched: Vec::new(),
        })
    }

    /// Whether the selection in hand takes the problem's `index`th edge.
    fn selects(&self, index: usize) -> bool {
        match self.ends.get(index) {
            Some(&Ends::Pair { first, second, .. }) => self.matched[first] == Some(second),
            Some(&Ends::Shared { external, supplier }) => {
                self.matched[external].is_some_and(|own| self.owner[own] == supplier)
            }
            None => false,
        }
    }

    /// A perfect matching of the whole gadget, or `None` when it has none.
    fn solve(&mut self) -> Option<()> {
        if self.vertex_count == 0 {
            return Some(());
        }
        let mut matched = std::mem::take(&mut self.matched);
        // Whatever the first selection left free is paired with a free
        // neighbour where it has one, which avoids resetting all search arrays
        // for a path consisting of just one edge. The blossom search still
        // repairs any pairs a perfect matching needs to change.
        for vertex in 0..self.vertex_count {
            if matched[vertex].is_some() {
                continue;
            }
            if let Some(&(free, _)) = self.adjacency[vertex]
                .iter()
                .find(|&&(neighbour, edge)| self.alive[edge] && matched[neighbour].is_none())
            {
                matched[vertex] = Some(free);
                matched[free] = Some(vertex);
            }
        }

        {
            let mut state = self.search();
            for root in 0..state.vertex_count {
                if matched[root].is_some() {
                    continue;
                }
                if let Some(last) = state.find_augmenting_path(root, &matched) {
                    augment(&mut matched, state.parent, last);
                }
            }
        }
        self.matched = matched;
        if self.matched.iter().any(Option::is_none) {
            return None;
        }
        Some(())
    }

    fn search(&mut self) -> Search<'_> {
        Search::new(&self.adjacency, &self.alive, &mut self.workspace)
    }

    /// Writes a decision into the gadget, so no later search can undo it.
    ///
    /// Taking the edge means its externals have only each other left; leaving
    /// it means they no longer have each other. Either way the matching in hand
    /// already agrees with the decision, so nothing about it changes.
    fn commit(&mut self, index: usize, taken: bool) {
        let (a, b, pair) = match self.ends[index] {
            Ends::Pair {
                first,
                second,
                pair,
            } => (first, second, pair),
            Ends::Shared { external, supplier } => {
                // Taking it closes the way to the internals, leaving it the way
                // to the copies.
                for position in 0..self.adjacency[external].len() {
                    let (own, edge) = self.adjacency[external][position];
                    if (self.owner[own] == supplier) != taken {
                        self.alive[edge] = false;
                    }
                }
                return;
            }
        };
        if taken {
            for external in [a, b] {
                for position in 0..self.adjacency[external].len() {
                    let (_, edge) = self.adjacency[external][position];
                    if edge != pair {
                        self.alive[edge] = false;
                    }
                }
            }
        } else {
            self.alive[pair] = false;
        }
    }

    /// Tries to make the decision the matching in hand disagrees with.
    ///
    /// The matching is perfect, so switching the decision's edges off frees
    /// exactly two vertices, and one alternating path between them puts it back
    /// together. `false` when no such path exists -- which is the answer that
    /// the decision cannot be had -- and then the gadget and the matching are
    /// left exactly as they were.
    fn force(&mut self, index: usize, wanted: bool) -> bool {
        self.switched.clear();
        let (a, b, pair) = match self.ends[index] {
            Ends::Pair {
                first,
                second,
                pair,
            } => (first, second, pair),
            Ends::Shared { external, supplier } => {
                // The external's partner is on the side being closed, so
                // closing it frees the two of them, and one path rejoins them.
                let Some(partner) = self.matched[external] else {
                    return false;
                };
                for position in 0..self.adjacency[external].len() {
                    let (own, edge) = self.adjacency[external][position];
                    if (self.owner[own] == supplier) != wanted && self.alive[edge] {
                        self.alive[edge] = false;
                        self.switched.push(edge);
                    }
                }
                self.matched[external] = None;
                self.matched[partner] = None;
                if self.repair(external) {
                    return true;
                }
                for &edge in &self.switched {
                    self.alive[edge] = true;
                }
                self.matched[external] = Some(partner);
                self.matched[partner] = Some(external);
                return false;
            }
        };
        let restore: [(usize, Option<usize>); 4];

        if wanted {
            let (Some(partner_a), Some(partner_b)) = (self.matched[a], self.matched[b]) else {
                return false;
            };
            restore = [
                (a, Some(partner_a)),
                (b, Some(partner_b)),
                (partner_a, Some(a)),
                (partner_b, Some(b)),
            ];
            for external in [a, b] {
                for position in 0..self.adjacency[external].len() {
                    let (_, edge) = self.adjacency[external][position];
                    if edge != pair && self.alive[edge] {
                        self.alive[edge] = false;
                        self.switched.push(edge);
                    }
                }
            }
            self.matched[a] = Some(b);
            self.matched[b] = Some(a);
            self.matched[partner_a] = None;
            self.matched[partner_b] = None;
            if self.repair(partner_a) {
                return true;
            }
        } else {
            restore = [(a, Some(b)), (b, Some(a)), (a, Some(b)), (b, Some(a))];
            if self.alive[pair] {
                self.alive[pair] = false;
                self.switched.push(pair);
            }
            self.matched[a] = None;
            self.matched[b] = None;
            if self.repair(a) {
                return true;
            }
        }

        for &edge in &self.switched {
            self.alive[edge] = true;
        }
        for (vertex, partner) in restore {
            self.matched[vertex] = partner;
        }
        false
    }

    /// Puts the two vertices an edit left free back into the matching.
    fn repair(&mut self, root: usize) -> bool {
        let mut matched = std::mem::take(&mut self.matched);
        let repaired = {
            let mut state = self.search();
            match state.find_augmenting_path(root, &matched) {
                Some(last) => {
                    augment(&mut matched, state.parent, last);
                    true
                }
                None => false,
            }
        };
        self.matched = matched;
        repaired
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::*;
    use rand::{RngExt, SeedableRng, seq::SliceRandom};

    /// Exhaustive subsets are an independent oracle for the solver. Random
    /// multigraphs include odd cycles, impossible degrees, parallel edges and
    /// priorities that require both successful and failed repair attempts.
    #[test]
    fn reused_search_matches_exhaustive_lexicographic_optima() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5C_A7C4);
        for case in 0..600 {
            let vertices = rng.random_range(2..=7);
            let edge_count = rng.random_range(1..=10);
            let edges: Vec<_> = (0..edge_count)
                .map(|_| {
                    let first = rng.random_range(0..vertices);
                    let mut second = rng.random_range(0..vertices - 1);
                    if second >= first {
                        second += 1;
                    }
                    (first, second)
                })
                .collect();
            let degrees = |mask: usize| {
                let mut degree = vec![0; vertices];
                for (index, &(a, b)) in edges.iter().enumerate() {
                    if mask & (1 << index) != 0 {
                        degree[a] += 1;
                        degree[b] += 1;
                    }
                }
                degree
            };
            let mut required = degrees(rng.random_range(0..(1usize << edge_count)));
            if case % 3 == 0 {
                required[rng.random_range(0..vertices)] += 1;
            }
            let mut order: Vec<_> = (0..edge_count).collect();
            order.shuffle(&mut rng);
            let mut priority: Vec<_> = order
                .iter()
                .map(|&index| (index, rng.random_bool(0.5)))
                .collect();
            let preference = |mask: usize| -> Vec<bool> {
                priority
                    .iter()
                    .map(|&(index, wanted)| (mask & (1 << index) != 0) != wanted)
                    .collect()
            };
            let best = (0..(1usize << edge_count))
                .filter(|&mask| degrees(mask) == required)
                .min_by_key(|&mask| preference(mask));
            let expected = best.map(|mask| {
                (0..edge_count)
                    .map(|index| mask & (1 << index) != 0)
                    .collect::<Vec<_>>()
            });
            // Contradictory repeated decisions must remain ignored.
            priority.push((priority[0].0, !priority[0].1));
            let actual =
                lexicographic_degree_constrained_subgraph(vertices, &edges, &required, &priority);
            assert_eq!(
                actual, expected,
                "case {case}: {edges:?}, {required:?}, {priority:?}"
            );
        }
    }

    #[test]
    fn reused_search_finds_maximum_matchings_on_all_five_vertex_graphs() {
        let possible: Vec<_> = (0..5)
            .flat_map(|a| (a + 1..5).map(move |b| (a, b)))
            .collect();
        for graph in 0..(1usize << possible.len()) {
            let edges: Vec<_> = possible
                .iter()
                .enumerate()
                .filter(|(index, _)| graph & (1 << index) != 0)
                .map(|(_, &edge)| edge)
                .collect();
            let optimum = (0..(1usize << edges.len()))
                .filter_map(|mask| {
                    let mut used = [false; 5];
                    let mut size = 0;
                    for (index, &(a, b)) in edges.iter().enumerate() {
                        if mask & (1 << index) == 0 {
                            continue;
                        }
                        if used[a] || used[b] {
                            return None;
                        }
                        used[a] = true;
                        used[b] = true;
                        size += 1;
                    }
                    Some(size)
                })
                .max()
                .unwrap();
            // Invalid seed entries exercise the same reusable augmentation path.
            let result = maximum_matching_from(5, &edges, &[Some(4), Some(4), None, None, Some(0)]);
            assert_eq!(
                result.iter().filter(|partner| partner.is_some()).count(),
                optimum * 2
            );
            for (a, partner) in result.iter().enumerate() {
                if let Some(b) = *partner {
                    assert_eq!(result[b], Some(a));
                    assert!(edges.contains(&(a.min(b), a.max(b))));
                }
            }
        }
    }
}
