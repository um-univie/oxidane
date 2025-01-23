//! The CIP sequence rules and the substituent sort they drive.
//!
//! Ported from RDKit's `Code/GraphMol/CIPLabeler/{Sort.cpp,rules/*}`.
//! See the module docs in [`super`] for provenance and licensing.
//!
//! # How the rules compose
//!
//! Each rule answers one question about a pair of substituents. The rules are
//! tried in order and the first that separates them wins — but "tried" means
//! *exhausted over the whole branch*, not evaluated at one atom: a rule is
//! applied outwards sphere by sphere until it either decides or runs out of
//! graph, and only then does the next rule get a look. That is what
//! [`recursive_compare`] does.
//!
//! There is a second layer of recursion on top: to walk a branch sphere by
//! sphere you must first know what order to visit each node's own substituents
//! in, and that ordering is itself a CIP comparison — one made with the rules
//! *up to and including* the rule currently asking. So every rule carries its
//! own sorter, and rules 4b and 5 additionally re-sort with a copy of
//! themselves parameterised by a chosen reference descriptor. Getting these
//! nested rule sets right is most of what makes CIP hard to implement; they
//! are threaded here as explicit [`Sorter`] values rather than the mutable
//! per-object sorters the C++ uses.

use super::descriptor::{Descriptor, PairList};
use super::digraph::{Digraph, EdgeId, NodeId};

/// One CIP sequence rule.
///
/// [`Rule::R4b`] and [`Rule::R5`] carry a reference descriptor. The
/// [`Descriptor::None`] form is the rule as it appears in the master list; a
/// form with a real descriptor is the temporary "rank everything relative to
/// this handedness" variant those two rules substitute for themselves while
/// building their pair lists, and it sorts using only itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Rule {
    /// Higher (averaged) atomic number precedes lower.
    R1a,
    /// A duplicate atom standing for a nearer ring closure precedes one
    /// standing for a further closure.
    R1b,
    /// Higher mass number precedes lower.
    R2,
    /// *Z* precedes *E*.
    R3,
    /// Any descriptor precedes none; an ordinary descriptor precedes a
    /// pseudo-asymmetric one.
    R4a,
    /// Like descriptor pairs precede unlike ones.
    R4b(Descriptor),
    /// *r* precedes *s*, *m* precedes *p*.
    R4c,
    /// *R* precedes *S* as a last resort.
    R5(Descriptor),
    /// One nominated atom precedes all others; used only to break the
    /// remaining symmetry of an S4-symmetric centre.
    R6,
}

impl Rule {
    /// Whether this is a reference-parameterised copy, which sorts using only
    /// itself rather than a prefix of the master list.
    fn is_replacement(self) -> bool {
        matches!(self, Rule::R4b(d) | Rule::R5(d) if d != Descriptor::None)
    }

    /// Compares two rules ignoring any reference descriptor, so a replacement
    /// can be located in the master list.
    fn same_kind(self, other: Rule) -> bool {
        std::mem::discriminant(&self) == std::mem::discriminant(&other)
    }
}

/// The rule list a particular sort ranks substituents by.
#[derive(Debug, Clone)]
pub(super) enum Sorter {
    /// The whole rule set applied as one: every rule is tried exhaustively,
    /// in order, on each pair. This is what ranks a stereocentre's own
    /// substituents.
    Composite,
    /// Individual rules applied in order, each either exhaustively (`deep`) or
    /// at one atom only (`shallow`), depending on the caller.
    List(Vec<Rule>),
}

/// What a sort discovered about the substituents it ordered.
#[derive(Debug, Clone, Copy)]
pub(super) struct Priority {
    /// The substituents could be put in a strict order.
    pub(super) unique: bool,
    /// Exactly one comparison was decided only by rule 5's *R*-before-*S*
    /// tiebreak, which is what makes a centre pseudo-asymmetric (lowercase
    /// *r*/*s*) rather than ordinarily chiral.
    pub(super) pseudo_asymmetric: bool,
}

/// Shared state for one labelling attempt: the rule set in force and the
/// remaining comparison budget.
///
/// The budget exists because the exhaustive comparisons can, on highly
/// symmetric cages, explore a number of paths that grows far faster than the
/// molecule does. Running out is not an error — the caller simply leaves that
/// centre unlabelled rather than hanging.
pub(super) struct Ctx<'a> {
    pub(super) master: &'a [Rule],
    pub(super) budget: u64,
}

impl Ctx<'_> {
    fn spend(&mut self) -> bool {
        if self.budget == 0 {
            return false;
        }
        self.budget -= 1;
        true
    }
}

fn cmp3<T: Ord>(a: T, b: T) -> i32 {
    match a.cmp(&b) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

/// The sorter a given rule uses when it needs to order a node's substituents
/// in the middle of its own comparison.
fn sorter_of(ctx: &Ctx, rule: Rule) -> Sorter {
    if rule.is_replacement() {
        return Sorter::List(vec![rule]);
    }
    match ctx.master.iter().position(|&r| r.same_kind(rule)) {
        Some(index) => Sorter::List(ctx.master[..=index].to_vec()),
        None => Sorter::List(vec![rule]),
    }
}

// ── Per-rule comparison at a single pair of nodes ─────────────────────────────

/// The descriptor carried by the bond an edge stands for, as rules 4a and 4c
/// read it.
fn bond_label(g: &Digraph, edge: EdgeId) -> Descriptor {
    g.edge(edge).aux
}

fn compare_one(g: &mut Digraph, ctx: &mut Ctx, rule: Rule, a: EdgeId, b: EdgeId) -> i32 {
    let (a_end, b_end) = (g.edge(a).end, g.edge(b).end);
    match rule {
        Rule::R1a => cmp3(
            g.node(a_end).atomic_number_fraction,
            g.node(b_end).atomic_number_fraction,
        ),
        Rule::R1b => {
            // Only duplicates are ranked, and a *smaller* distance wins: the
            // duplicate of a nearer ring-closure atom outranks that of a
            // further one. A real atom outranks any duplicate.
            let (a_dup, b_dup) = (
                g.node(a_end).is_set(super::digraph::RING_DUPLICATE),
                g.node(b_end).is_set(super::digraph::RING_DUPLICATE),
            );
            match (a_dup, b_dup) {
                (true, true) => -cmp3(g.node(a_end).distance, g.node(b_end).distance),
                (true, false) => 1,
                (false, true) => -1,
                (false, false) => 0,
            }
        }
        Rule::R2 => {
            let (a_num, b_num) = (g.node(a_end).atomic_number, g.node(b_end).atomic_number);
            if a_num == 0 && b_num == 0 {
                return 0;
            }
            if a_num == 0 || b_num == 0 {
                return cmp3(a_num, b_num);
            }
            if g.node(a_end).mass_number == 0 && g.node(b_end).mass_number == 0 {
                return 0;
            }
            g.node(a_end)
                .atomic_mass
                .partial_cmp(&g.node(b_end).atomic_mass)
                .map(|o| o as i32)
                .unwrap_or(0)
        }
        Rule::R3 => {
            let ord = |d: Descriptor| match d {
                Descriptor::E => 1,
                Descriptor::Z => 2,
                _ => 0,
            };
            cmp3(ord(g.node(a_end).aux), ord(g.node(b_end).aux))
        }
        Rule::R4a => {
            let ord = |d: Descriptor| match d {
                Descriptor::LowerR
                | Descriptor::LowerS
                | Descriptor::LowerM
                | Descriptor::LowerP
                | Descriptor::E
                | Descriptor::Z => 1,
                Descriptor::R
                | Descriptor::S
                | Descriptor::M
                | Descriptor::P
                | Descriptor::SeqTrans
                | Descriptor::SeqCis => 2,
                _ => 0,
            };
            let cmp = cmp3(ord(bond_label(g, a)), ord(bond_label(g, b)));
            if cmp != 0 {
                return cmp;
            }
            cmp3(ord(g.node(a_end).aux), ord(g.node(b_end).aux))
        }
        Rule::R4c => {
            let ord = |d: Descriptor| match d {
                Descriptor::LowerM | Descriptor::LowerR => 2,
                Descriptor::LowerP | Descriptor::LowerS => 1,
                _ => 0,
            };
            let cmp = cmp3(ord(bond_label(g, a)), ord(bond_label(g, b)));
            if cmp != 0 {
                return cmp;
            }
            cmp3(ord(g.node(a_end).aux), ord(g.node(b_end).aux))
        }
        Rule::R4b(reference) => compare_rule4b(g, ctx, reference, a, b),
        Rule::R5(reference) => compare_rule5(g, ctx, reference, a, b),
        Rule::R6 => {
            let Some(reference) = g.rule6_ref else { return 0 };
            let (a_atom, b_atom) = (g.node(a_end).atom, g.node(b_end).atom);
            if reference == a_atom && reference != b_atom {
                1
            } else if reference != a_atom && reference == b_atom {
                -1
            } else {
                0
            }
        }
    }
}

// ── Rules 4b and 5: like/unlike pairing ──────────────────────────────────────

/// Away from the root, both rules degenerate to "does this node's descriptor
/// agree with the reference we are currently ranking against".
fn like_unlike_at_depth(g: &Digraph, reference: Descriptor, a: EdgeId, b: EdgeId) -> i32 {
    if reference == Descriptor::None {
        return 0;
    }
    let a_desc = g.node(g.edge(a).end).aux;
    let b_desc = g.node(g.edge(b).end).aux;
    if a_desc == Descriptor::None
        || b_desc == Descriptor::None
        || a_desc == Descriptor::Ns
        || b_desc == Descriptor::Ns
    {
        return 0;
    }
    let a_like = reference.pair_ref() == a_desc.pair_ref();
    let b_like = reference.pair_ref() == b_desc.pair_ref();
    match (a_like, b_like) {
        (true, false) => 1,
        (false, true) => -1,
        _ => 0,
    }
}

fn at_root(g: &Digraph, edge: EdgeId) -> bool {
    g.edge(edge).beg == g.root()
}

fn compare_rule4b(
    g: &mut Digraph,
    ctx: &mut Ctx,
    reference: Descriptor,
    a: EdgeId,
    b: EdgeId,
) -> i32 {
    if !at_root(g, a) || !at_root(g, b) {
        return like_unlike_at_depth(g, reference, a, b);
    }

    let (a_end, b_end) = (g.edge(a).end, g.edge(b).end);
    let a_refs = reference_descriptors(g, ctx, Rule::R4b(reference), a_end);
    let b_refs = reference_descriptors(g, ctx, Rule::R4b(reference), b_end);

    if a_refs.is_empty() != b_refs.is_empty() {
        // Topologically inequivalent substituents should already have been
        // separated by an earlier rule; if they reach here, decline to decide.
        return 0;
    }
    if a_refs.len() == 1 && b_refs.len() == 1 {
        return compare_pairs(g, ctx, Rule::R4b(reference), a_end, b_end, a_refs[0], b_refs[0]);
    }
    if a_refs.len() > 1 && a_refs.len() == b_refs.len() {
        let mut a_lists: Vec<PairList> = a_refs.iter().map(|&d| PairList::with_ref(d)).collect();
        let mut b_lists: Vec<PairList> = b_refs.iter().map(|&d| PairList::with_ref(d)).collect();
        for list in &mut a_lists {
            fill_pairs(g, ctx, Rule::R4b(reference), a_end, list);
        }
        for list in &mut b_lists {
            fill_pairs(g, ctx, Rule::R4b(reference), b_end, list);
        }
        a_lists.sort_by(|x, y| y.compare_to(x).cmp(&0));
        b_lists.sort_by(|x, y| y.compare_to(x).cmp(&0));
        for (x, y) in a_lists.iter().zip(b_lists.iter()) {
            let cmp = x.compare_to(y);
            if cmp != 0 {
                return cmp;
            }
        }
    }
    0
}

fn compare_rule5(
    g: &mut Digraph,
    ctx: &mut Ctx,
    reference: Descriptor,
    a: EdgeId,
    b: EdgeId,
) -> i32 {
    if !at_root(g, a) || !at_root(g, b) {
        return like_unlike_at_depth(g, reference, a, b);
    }

    let (a_end, b_end) = (g.edge(a).end, g.edge(b).end);
    let rule = Rule::R5(reference);

    let mut a_as_r = PairList::with_ref(Descriptor::R);
    let mut b_as_r = PairList::with_ref(Descriptor::R);
    let mut a_as_s = PairList::with_ref(Descriptor::S);
    let mut b_as_s = PairList::with_ref(Descriptor::S);
    fill_pairs(g, ctx, rule, a_end, &mut a_as_r);
    fill_pairs(g, ctx, rule, a_end, &mut a_as_s);
    fill_pairs(g, ctx, rule, b_end, &mut b_as_r);
    fill_pairs(g, ctx, rule, b_end, &mut b_as_s);

    let cmp_r = a_as_r.compare_to(&b_as_r);
    let cmp_s = a_as_s.compare_to(&b_as_s);

    // A branch that wins as *R* but loses as *S* is only distinguishable by
    // its handedness, which is what makes the parent pseudo-asymmetric; the
    // caller detects that from the magnitude 2.
    match cmp_r {
        c if c < 0 => {
            if cmp_s < 0 {
                -1
            } else {
                -2
            }
        }
        c if c > 0 => {
            if cmp_s > 0 {
                1
            } else {
                2
            }
        }
        _ => 0,
    }
}

/// Walks outwards from `node` sphere by sphere until a sphere carries
/// descriptors, and reports the majority handedness there — or both, if the
/// sphere is evenly split, in which case the caller has to try each.
fn reference_descriptors(
    g: &mut Digraph,
    ctx: &mut Ctx,
    rule: Rule,
    node: NodeId,
) -> Vec<Descriptor> {
    let mut result = Vec::new();
    let mut level: Vec<Vec<NodeId>> = vec![vec![node]];
    while !level.is_empty() {
        for nodes in &level {
            if majority_descriptor(g, nodes, &mut result) {
                return result;
            }
        }
        level = next_level(g, ctx, rule, &level);
        if ctx.budget == 0 {
            break;
        }
    }
    Vec::new()
}

fn majority_descriptor(g: &Digraph, nodes: &[NodeId], result: &mut Vec<Descriptor>) -> bool {
    let (mut right, mut left) = (0, 0);
    for &node in nodes {
        match g.node(node).aux {
            Descriptor::R | Descriptor::M | Descriptor::SeqCis => right += 1,
            Descriptor::S | Descriptor::P | Descriptor::SeqTrans => left += 1,
            _ => {}
        }
    }
    if right + left == 0 {
        return false;
    }
    match right.cmp(&left) {
        std::cmp::Ordering::Greater => result.push(Descriptor::R),
        std::cmp::Ordering::Less => result.push(Descriptor::S),
        std::cmp::Ordering::Equal => {
            result.push(Descriptor::R);
            result.push(Descriptor::S);
        }
    }
    true
}

/// The next sphere out, keeping substituents that the rules cannot separate
/// grouped together — those groups are what the majority vote is taken over.
fn next_level(
    g: &mut Digraph,
    ctx: &mut Ctx,
    rule: Rule,
    previous: &[Vec<NodeId>],
) -> Vec<Vec<NodeId>> {
    let sorter = sorter_of(ctx, rule);
    let mut next: Vec<Vec<NodeId>> = Vec::new();

    for group in previous {
        let mut per_node_groups: Vec<Vec<Vec<EdgeId>>> = Vec::new();
        for &node in group {
            let mut edges = g.non_terminal_out_edges(node);
            prioritize(g, ctx, &sorter, node, &mut edges, true);
            per_node_groups.push(get_groups(g, ctx, &sorter, node, &edges));
        }

        let size = per_node_groups.first().map(|t| t.len()).unwrap_or(0);
        // Topologically equivalent nodes must partition the same way; if they
        // do not, this sphere carries no usable reference.
        if per_node_groups.iter().any(|t| t.len() != size) {
            continue;
        }

        for i in 0..size {
            let mut equivalent: Vec<NodeId> = Vec::new();
            for per_node in &per_node_groups {
                for &edge in &per_node[i] {
                    equivalent.push(g.edge(edge).end);
                }
            }
            if !equivalent.is_empty() {
                next.push(equivalent);
            }
        }
    }
    next
}

/// Records the descriptors met walking outwards from `beg`, visiting each
/// node's substituents in the order the reference-parameterised rule ranks
/// them — so the list is a property of the branch and its chosen handedness,
/// not of any numbering.
fn fill_pairs(g: &mut Digraph, ctx: &mut Ctx, rule: Rule, beg: NodeId, list: &mut PairList) {
    let replacement = match rule {
        Rule::R4b(_) => Rule::R4b(list.ref_descriptor()),
        _ => Rule::R5(list.ref_descriptor()),
    };
    let sorter = ref_sorter(ctx, rule, replacement);

    let mut queue = vec![beg];
    let mut position = 0;
    while position < queue.len() {
        if !ctx.spend() {
            return;
        }
        let node = queue[position];
        position += 1;
        list.add(g.node(node).aux);

        let mut edges = g.edges_of(node);
        prioritize(g, ctx, &sorter, node, &mut edges, true);
        for edge in edges {
            if g.edge(edge).is_beg(node) && !g.node(g.edge(edge).end).is_terminal() {
                queue.push(g.edge(edge).end);
            }
        }
    }
}

fn compare_pairs(
    g: &mut Digraph,
    ctx: &mut Ctx,
    rule: Rule,
    a: NodeId,
    b: NodeId,
    ref_a: Descriptor,
    ref_b: Descriptor,
) -> i32 {
    let (replacement_a, replacement_b) = match rule {
        Rule::R4b(_) => (Rule::R4b(ref_a), Rule::R4b(ref_b)),
        _ => (Rule::R5(ref_a), Rule::R5(ref_b)),
    };
    let sorter_a = ref_sorter(ctx, rule, replacement_a);
    let sorter_b = ref_sorter(ctx, rule, replacement_b);

    let mut queue_a = vec![a];
    let mut queue_b = vec![b];
    let mut position = 0;
    while position < queue_a.len() && position < queue_b.len() {
        if !ctx.spend() {
            return 0;
        }
        let (node_a, node_b) = (queue_a[position], queue_b[position]);
        position += 1;

        let desc_a = g.node(node_a).aux.pair_ref();
        let desc_b = g.node(node_b).aux.pair_ref();
        if desc_a == ref_a && desc_b != ref_b {
            return 1;
        }
        if desc_a != ref_a && desc_b == ref_b {
            return -1;
        }

        let mut edges = g.edges_of(node_a);
        prioritize(g, ctx, &sorter_a, node_a, &mut edges, true);
        for edge in edges {
            if g.edge(edge).is_beg(node_a) && !g.node(g.edge(edge).end).is_terminal() {
                queue_a.push(g.edge(edge).end);
            }
        }

        let mut edges = g.edges_of(node_b);
        prioritize(g, ctx, &sorter_b, node_b, &mut edges, true);
        for edge in edges {
            if g.edge(edge).is_beg(node_b) && !g.node(g.edge(edge).end).is_terminal() {
                queue_b.push(g.edge(edge).end);
            }
        }
    }
    0
}

/// The rule list rules 4b/5 sort with while filling a pair list: their own
/// prefix of the master list, with themselves swapped for the
/// reference-parameterised copy.
fn ref_sorter(ctx: &Ctx, rule: Rule, replacement: Rule) -> Sorter {
    let Sorter::List(prefix) = sorter_of(ctx, rule) else {
        return Sorter::List(vec![replacement]);
    };
    let mut rules: Vec<Rule> = prefix.into_iter().filter(|r| !r.same_kind(rule)).collect();
    rules.push(replacement);
    Sorter::List(rules)
}

// ── Exhaustive comparison ────────────────────────────────────────────────────

/// Applies one rule to a pair of substituents across their whole branches:
/// first at the substituent atoms themselves, then sphere by sphere outwards,
/// pairing off the two branches' children in rule order.
///
/// Each sphere is compared twice — once with the children merely
/// shallow-sorted, and again after a full deep sort. The shallow pass is what
/// lets an obvious difference (a chlorine against a carbon) settle the
/// comparison without first paying for the recursive ranking of every branch
/// hanging off both nodes.
pub(super) fn recursive_compare(
    g: &mut Digraph,
    ctx: &mut Ctx,
    rule: Rule,
    a: EdgeId,
    b: EdgeId,
) -> i32 {
    if !ctx.spend() {
        return 0;
    }

    let cmp = compare_one(g, ctx, rule, a, b);
    if cmp != 0 {
        return cmp;
    }

    let sorter = sorter_of(ctx, rule);
    let mut queue_a = vec![a];
    let mut queue_b = vec![b];
    let mut position = 0;

    while position < queue_a.len() && position < queue_b.len() {
        let (edge_a, edge_b) = (queue_a[position], queue_b[position]);
        position += 1;
        let (node_a, node_b) = (g.edge(edge_a).end, g.edge(edge_b).end);

        let mut edges_a = g.edges_of(node_a);
        let mut edges_b = g.edges_of(node_b);
        prioritize(g, ctx, &sorter, node_a, &mut edges_a, false);
        prioritize(g, ctx, &sorter, node_b, &mut edges_b, false);

        let size_difference = cmp3(edges_a.len(), edges_b.len());

        for (&x, &y) in edges_a.iter().zip(edges_b.iter()) {
            // Skip the edges leading back towards the root: they are the same
            // bond on both sides and carry no information.
            if g.edge(x).is_end(node_a) {
                continue;
            }
            let cmp = compare_one(g, ctx, rule, x, y);
            if cmp != 0 {
                return cmp;
            }
        }

        // A branch that runs out of substituents first ranks lower.
        if size_difference != 0 {
            return size_difference;
        }

        prioritize(g, ctx, &sorter, node_a, &mut edges_a, true);
        prioritize(g, ctx, &sorter, node_b, &mut edges_b, true);

        for (&x, &y) in edges_a.iter().zip(edges_b.iter()) {
            if g.edge(x).is_end(node_a) {
                continue;
            }
            let cmp = compare_one(g, ctx, rule, x, y);
            if cmp != 0 {
                return cmp;
            }
            queue_a.push(x);
            queue_b.push(y);
        }

        if ctx.budget == 0 {
            return 0;
        }
    }
    0
}

/// One rule's verdict on a pair, either exhaustively (`deep`) or at the pair
/// of atoms alone.
fn rule_comparison(
    g: &mut Digraph,
    ctx: &mut Ctx,
    rule: Rule,
    a: EdgeId,
    b: EdgeId,
    deep: bool,
) -> i32 {
    if deep {
        recursive_compare(g, ctx, rule, a, b)
    } else {
        compare_one(g, ctx, rule, a, b)
    }
}

/// The full rule set's verdict: every rule tried exhaustively, in order.
fn composite_comparison(g: &mut Digraph, ctx: &mut Ctx, a: EdgeId, b: EdgeId) -> i32 {
    for index in 0..ctx.master.len() {
        let rule = ctx.master[index];
        let cmp = recursive_compare(g, ctx, rule, a, b);
        if cmp != 0 {
            return cmp;
        }
    }
    0
}

fn compare_substituents(
    g: &mut Digraph,
    ctx: &mut Ctx,
    sorter: &Sorter,
    node: NodeId,
    a: EdgeId,
    b: EdgeId,
    deep: bool,
) -> i32 {
    // Edges pointing back towards the root sort ahead of the substituents, so
    // that the two branches being compared line up position for position.
    let (a_out, b_out) = (g.edge(a).is_beg(node), g.edge(b).is_beg(node));
    if !a_out && b_out {
        return 1;
    }
    if a_out && !b_out {
        return -1;
    }

    match sorter {
        Sorter::Composite => composite_comparison(g, ctx, a, b),
        Sorter::List(rules) => {
            for &rule in rules {
                let cmp = rule_comparison(g, ctx, rule, a, b, deep);
                if cmp != 0 {
                    return cmp;
                }
            }
            0
        }
    }
}

/// Orders `edges` by decreasing CIP priority, reporting whether the order came
/// out strict and whether rule 5 had to break exactly one tie.
///
/// Insertion sort: the lists are four elements at most, and a comparison here
/// can cost an exhaustive walk of two branches, so minimising comparisons
/// matters far more than the asymptotics.
pub(super) fn prioritize(
    g: &mut Digraph,
    ctx: &mut Ctx,
    sorter: &Sorter,
    node: NodeId,
    edges: &mut Vec<EdgeId>,
    deep: bool,
) -> Priority {
    let mut unique = true;
    let mut pseudo_asymmetric_count = 0;

    for i in 0..edges.len() {
        let mut j = i;
        while j > 0 {
            let cmp = compare_substituents(g, ctx, sorter, node, edges[j - 1], edges[j], deep);
            if !(-1..=1).contains(&cmp) {
                pseudo_asymmetric_count += 1;
            }
            if cmp < 0 {
                edges.swap(j, j - 1);
            } else {
                if cmp == 0 {
                    unique = false;
                }
                break;
            }
            j -= 1;
        }
    }

    Priority { unique, pseudo_asymmetric: pseudo_asymmetric_count == 1 }
}

/// Orders edges that hang off *different* nodes by decreasing CIP priority.
///
/// [`prioritize`] ranks the substituents of one atom; an allene's four
/// carriers are split two and two across the ends of the cumulated double
/// bond, so they have no common parent to be ranked from. The comparison is
/// the same exhaustive one, without the in-edge ordering step, which has no
/// meaning here — all four point outwards from their own end.
///
/// Returns `None` if any two are indistinguishable: a centre whose carriers
/// cannot be told apart has no configuration.
pub(super) fn rank_independent_edges(
    g: &mut Digraph,
    ctx: &mut Ctx,
    edges: &mut Vec<EdgeId>,
) -> Option<()> {
    for i in 0..edges.len() {
        let mut j = i;
        while j > 0 {
            let cmp = composite_comparison(g, ctx, edges[j - 1], edges[j]);
            if cmp == 0 {
                return None;
            }
            if cmp < 0 {
                edges.swap(j, j - 1);
            } else {
                break;
            }
            j -= 1;
        }
    }
    Some(())
}

/// Partitions an already-sorted edge list into runs the rules cannot separate.
pub(super) fn get_groups(
    g: &mut Digraph,
    ctx: &mut Ctx,
    sorter: &Sorter,
    node: NodeId,
    sorted: &[EdgeId],
) -> Vec<Vec<EdgeId>> {
    let mut groups: Vec<Vec<EdgeId>> = Vec::new();
    let mut previous: Option<EdgeId> = None;
    for &edge in sorted {
        let start_new = match previous {
            None => true,
            Some(previous_edge) => {
                let beg = g.edge(previous_edge).beg;
                compare_substituents(g, ctx, sorter, beg, previous_edge, edge, true) != 0
            }
        };
        if start_new {
            groups.push(Vec::new());
        }
        previous = Some(edge);
        // `groups` is non-empty here: the first iteration always starts one.
        if let Some(last) = groups.last_mut() {
            last.push(edge);
        }
    }
    let _ = node;
    groups
}
