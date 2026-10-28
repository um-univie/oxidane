//! The algebra of double-pushout rules: composition, critical pairs and a
//! canonical form.
//!
//! All three work on [`ExactRule`]s, rules whose every label is a value — an
//! element, a charge, an isotope, a bond order — rather than a test. That is
//! the setting the theory is stated in: to compose two rules or to overlap
//! them, one has to know whether an atom the first rule makes is the atom the
//! second rule wants, and `[C,N]` does not say. A [`ReactionSmarts`] read with
//! [`ReactionSmarts::parse_dpo`] or [`ReactionSmarts::from_gml`] converts to
//! one when its reactant side names an element for every atom and an order
//! for every bond; a charge it leaves unstated is read as neutral, as GML
//! reads a bare `C`.
//!
//! Atoms are identified by map index throughout, so a graph here is a set of
//! map indices with labels and the bonds between them.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::io::{BondExpression, CxSMARTSPattern};
use crate::molecule::base::charge_valency_shift;
use crate::molecule::bond::BondOrder;
use crate::molecule::reaction::{
    ReactionError, ReactionSmarts, Semantics, bond_order_symbol, charge_of, element_of, isotope_of,
    product_order, simple_order, states_stereo,
};

/// An exact atom label: isotope, element, charge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Label {
    pub isotope: Option<u16>,
    pub atomic_number: u8,
    pub charge: i8,
}

impl Label {
    /// The label as an exact SMARTS atom carrying `index` as its map.
    pub(crate) fn to_smarts(self, index: usize) -> String {
        let isotope = self.isotope.map(|i| i.to_string()).unwrap_or_default();
        format!(
            "[{isotope}#{}{:+}:{index}]",
            self.atomic_number, self.charge
        )
    }
}

/// One side of a rule, or a graph a rule acts on: labelled atoms by map
/// index, and bonds by the pair of indices, smaller first, labelled with
/// their SMARTS symbol.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Graph {
    pub(crate) atoms: BTreeMap<usize, Label>,
    pub(crate) bonds: BTreeMap<(usize, usize), &'static str>,
}

fn pair(first: usize, second: usize) -> (usize, usize) {
    (first.min(second), first.max(second))
}

impl Graph {
    fn bond(&self, first: usize, second: usize) -> Option<&'static str> {
        self.bonds.get(&pair(first, second)).copied()
    }

    fn degree(&self, atom: usize) -> usize {
        self.bonds
            .keys()
            .filter(|&&(a, b)| a == atom || b == atom)
            .count()
    }

    /// Whether every atom stays within its element's largest valence. A
    /// glued graph that puts two bonds on a hydrogen is a graph, but no part
    /// of any molecule, so no overlap that makes one is chemistry.
    fn within_valences(&self) -> bool {
        use chemistry_consts::ElementProperties;
        self.atoms.iter().all(|(&atom, label)| {
            let Some(largest) = label
                .atomic_number
                .valencies()
                .and_then(|valences| valences.filter(|v| *v > 0).max())
            else {
                return true;
            };
            let allowed = largest + charge_valency_shift(label.atomic_number, label.charge);
            let used: i32 = self
                .bonds
                .iter()
                .filter(|&(&(a, b), _)| a == atom || b == atom)
                .map(|(_, &symbol)| half_units(symbol))
                .sum();
            used <= 2 * i32::from(allowed)
        })
    }

    /// The graph as a SMARTS: each atom written once, `.` between them, and
    /// every bond a ring closure, which SMILES lets join atoms across a `.`.
    pub(crate) fn to_smarts(&self) -> String {
        let mut closures: BTreeMap<usize, Vec<(usize, &'static str)>> = BTreeMap::new();
        let mut open: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut free: BTreeSet<usize> = (1..100).collect();
        for (&(first, second), &symbol) in &self.bonds {
            closures.entry(first).or_default().push((second, symbol));
            closures.entry(second).or_default().push((first, symbol));
        }
        let mut parts = Vec::new();
        for (&index, &label) in &self.atoms {
            let mut part = label.to_smarts(index);
            // A number closed here is free again only past this atom, so no
            // atom closes and reopens the same number.
            let mut closed = Vec::new();
            for &(other, symbol) in closures.get(&index).into_iter().flatten() {
                let key = pair(index, other);
                let number = match open.remove(&key) {
                    Some(number) => {
                        closed.push(number);
                        number
                    }
                    None => {
                        let number = free.pop_first().expect("fewer than 99 open bonds");
                        open.insert(key, number);
                        number
                    }
                };
                write!(part, "{symbol}%{number:02}").unwrap();
            }
            free.extend(closed);
            parts.push(part);
        }
        parts.join(".")
    }
}

/// Why a reaction has no exact form.
#[derive(Debug, Clone, PartialEq)]
pub enum ExactRuleError {
    /// The rule is read the RDKit way; only a double-pushout rule has an
    /// algebra.
    NotDpo,
    /// A reactant label that tests rather than names, so there is no single
    /// value to compose or overlap with.
    NotExact { index: usize, what: String },
    /// Application conditions, which the algebra does not carry.
    Conditions,
    /// The rule a construction produced is not a valid one.
    Rule(ReactionError),
}

impl std::fmt::Display for ExactRuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExactRuleError::NotDpo => write!(f, "Only a double-pushout rule has exact form"),
            ExactRuleError::NotExact { index, what } => {
                write!(f, "Atom :{index} has no exact label: {what}")
            }
            ExactRuleError::Conditions => {
                write!(f, "Application conditions have no exact form")
            }
            ExactRuleError::Rule(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ExactRuleError {}

impl From<ReactionError> for ExactRuleError {
    fn from(error: ReactionError) -> Self {
        ExactRuleError::Rule(error)
    }
}

/// A double-pushout rule whose labels are all exact: L and R, sharing the map
/// indices of K.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactRule {
    pub(crate) left: Graph,
    pub(crate) right: Graph,
}

/// The bond a side writes between two map indices.
fn written_bond<'a>(
    pattern: &'a CxSMARTSPattern,
    by_index: &std::collections::HashMap<usize, usize>,
    first: usize,
    second: usize,
) -> Option<&'a BondExpression> {
    let (&a, &b) = (by_index.get(&first)?, by_index.get(&second)?);
    pattern
        .bonds
        .iter()
        .find(|&&(x, y, _)| (x, y) == (a, b) || (x, y) == (b, a))
        .map(|(_, _, expression)| expression)
}

impl ExactRule {
    /// The exact form of a double-pushout rule.
    pub fn from_reaction(reaction: &ReactionSmarts) -> Result<Self, ExactRuleError> {
        if reaction.semantics() != Semantics::Dpo {
            return Err(ExactRuleError::NotDpo);
        }
        if reaction.has_conditions() {
            return Err(ExactRuleError::Conditions);
        }
        let (left, left_index, right, right_index) = reaction.sides();
        let not_exact = |index: usize, what: String| ExactRuleError::NotExact { index, what };

        let mut rule = ExactRule {
            left: Graph::default(),
            right: Graph::default(),
        };
        for (&index, &atom) in left_index {
            let expression = &left.atoms[atom];
            if states_stereo(expression) {
                return Err(not_exact(index, "stereo".into()));
            }
            let atomic_number = element_of(expression)
                .ok_or_else(|| not_exact(index, format!("{expression:?}")))?;
            rule.left.atoms.insert(
                index,
                Label {
                    isotope: isotope_of(expression),
                    atomic_number,
                    charge: charge_of(expression).unwrap_or(0),
                },
            );
        }
        for (&index, &atom) in right_index {
            let expression = &right.atoms[atom];
            if states_stereo(expression) {
                return Err(not_exact(index, "stereo".into()));
            }
            let kept = rule.left.atoms.get(&index).copied();
            let atomic_number = element_of(expression)
                .or(kept.map(|label| label.atomic_number))
                .ok_or_else(|| not_exact(index, format!("{expression:?}")))?;
            rule.right.atoms.insert(
                index,
                Label {
                    isotope: isotope_of(expression).or(kept.and_then(|label| label.isotope)),
                    atomic_number,
                    charge: charge_of(expression)
                        .or(kept.map(|label| label.charge))
                        .unwrap_or(0),
                },
            );
        }

        let mut indices: Vec<usize> = left_index
            .keys()
            .chain(right_index.keys())
            .copied()
            .collect();
        indices.sort_unstable();
        indices.dedup();
        for (position, &first) in indices.iter().enumerate() {
            for &second in &indices[position + 1..] {
                let before = written_bond(left, left_index, first, second);
                let after = written_bond(right, right_index, first, second);
                let directional = |expression: Option<&BondExpression>| {
                    matches!(
                        expression,
                        Some(BondExpression::DirectionalUp | BondExpression::DirectionalDown)
                    )
                };
                if directional(before) || directional(after) {
                    return Err(not_exact(
                        first,
                        format!("bond to :{second} has a direction"),
                    ));
                }
                let before_order = match before {
                    Some(expression) => Some(simple_order(expression).ok_or_else(|| {
                        not_exact(first, format!("bond to :{second} is {expression:?}"))
                    })?),
                    None => None,
                };
                let after_order = match (before, after) {
                    (_, None) => None,
                    (None, Some(after)) => Some(simple_order(after).unwrap_or(BondOrder::Single)),
                    (Some(before), Some(after)) => product_order(before, after).or(before_order),
                };
                if let Some(order) = before_order {
                    rule.left
                        .bonds
                        .insert((first, second), bond_order_symbol(order));
                }
                if let Some(order) = after_order {
                    rule.right
                        .bonds
                        .insert((first, second), bond_order_symbol(order));
                }
            }
        }
        Ok(rule)
    }

    /// The rule as a double-pushout [`ReactionSmarts`].
    pub fn to_reaction(&self) -> Result<ReactionSmarts, ReactionError> {
        ReactionSmarts::parse_dpo(&self.to_smarts())
    }

    /// The rule as a reaction SMARTS with exact labels.
    pub fn to_smarts(&self) -> String {
        format!("{}>>{}", self.left.to_smarts(), self.right.to_smarts())
    }

    fn max_index(&self) -> usize {
        self.left
            .atoms
            .keys()
            .chain(self.right.atoms.keys())
            .copied()
            .max()
            .unwrap_or(0)
    }

    /// Applies the rule to `graph` at the match `rename` gives (rule index ->
    /// graph index, injective, and naming a fresh index for every atom only R
    /// has), or `None` where it does not apply: an atom or bond of L missing or
    /// labelled differently, or a deleted atom with a bond L does not have.
    fn rewrite(&self, graph: &Graph, rename: &BTreeMap<usize, usize>) -> Option<Graph> {
        let at = |index: &usize| rename[index];
        for (index, label) in &self.left.atoms {
            if graph.atoms.get(&at(index)) != Some(label) {
                return None;
            }
        }
        for (&(a, b), &symbol) in &self.left.bonds {
            if graph.bond(at(&a), at(&b)) != Some(symbol) {
                return None;
            }
        }
        for index in self.left.atoms.keys() {
            if !self.right.atoms.contains_key(index)
                && graph.degree(at(index)) != self.left.degree(*index)
            {
                return None;
            }
        }
        let mut result = graph.clone();
        for &(a, b) in self.left.bonds.keys() {
            result.bonds.remove(&pair(at(&a), at(&b)));
        }
        for index in self.left.atoms.keys() {
            if !self.right.atoms.contains_key(index) {
                result.atoms.remove(&at(index));
            }
        }
        for (index, &label) in &self.right.atoms {
            result.atoms.insert(at(index), label);
        }
        for (&(a, b), &symbol) in &self.right.bonds {
            result.bonds.insert(pair(at(&a), at(&b)), symbol);
        }
        Some(result)
    }

    /// Every rule that does what applying `self` and then `then` does, one per
    /// way the atoms of `then`'s L can be identified with atoms of `self`'s R
    /// — the concurrent rules of the two. The empty overlap is one of them,
    /// and gives the parallel rule: both rules side by side, on separate
    /// atoms.
    ///
    /// An overlapped atom must carry the same label on both, and a bond `then`
    /// expects between overlapped atoms must be one `self` leaves or makes
    /// with that order, or one between atoms `self` keeps and does not touch,
    /// which then becomes part of the composite's L. Rules equal up to
    /// renumbering are returned once. The enumeration is exponential in the
    /// size of the rules, as overlap enumeration is.
    pub fn compose(&self, then: &ExactRule) -> Vec<ExactRule> {
        let right: Vec<(usize, Label)> = self.right.atoms.iter().map(|(&i, &l)| (i, l)).collect();
        let left = heavy_first(&then.left);
        let mut twins = Default::default();
        let mut found = Vec::new();
        let mut seen = BTreeSet::new();
        let mut overlap = BTreeMap::new();
        let walk = Walk::new(self, &self.right, then, &left, &right, false, &mut twins);
        self.overlaps_into(&walk, 0, &mut overlap, &mut |overlap| {
            if let Some(rule) = self.concurrent(then, overlap) {
                let (key, canonical) = rule.canonical();
                if seen.insert(key) {
                    found.push(canonical);
                }
            }
        });
        found
    }

    /// Walks every partial injective map from `then`'s L atoms into the
    /// host's atoms that agrees on labels and on the bonds already decided.
    fn overlaps_into(
        &self,
        walk: &Walk,
        position: usize,
        overlap: &mut BTreeMap<usize, usize>,
        visit: &mut dyn FnMut(&BTreeMap<usize, usize>),
    ) {
        let Walk {
            host,
            then,
            left,
            targets,
            glue,
            target_twin,
            left_twin,
        } = *walk;
        if !self.fits(walk, position, overlap) {
            return;
        }
        if position == left.len() {
            visit(overlap);
            return;
        }
        let (atom, label) = left[position];
        self.overlaps_into(walk, position + 1, overlap, visit);
        // Twins are interchangeable, so of two twins of `then` the earlier is
        // overlapped whenever the later is, onto an earlier target; and of the
        // host's twins only the first unused one is tried.
        let floor = match left_twin[position] {
            Some(twin) => match overlap.get(&left[twin].0) {
                None => return,
                Some(&taken) => targets.iter().position(|&(t, _)| t == taken),
            },
            None => None,
        };
        for (slot, &(target, target_label)) in targets.iter().enumerate() {
            if target_label != label
                || overlap.values().any(|&used| used == target)
                || floor.is_some_and(|floor| slot <= floor)
            {
                continue;
            }
            let twin = target_twin[slot];
            if twin != slot
                && (twin..slot).any(|earlier| {
                    target_twin[earlier] == twin
                        && !overlap.values().any(|&used| used == targets[earlier].0)
                })
            {
                continue;
            }
            let consistent = overlap.iter().all(|(&other, &other_target)| {
                match then.left.bond(atom, other) {
                    None => true,
                    Some(symbol) => match host.bond(target, other_target) {
                        Some(host_symbol) => host_symbol == symbol,
                        // Gluing adds the bond. Composing can only assume a
                        // bond nobody wrote, between atoms the first rule
                        // keeps.
                        None if glue => true,
                        None => {
                            self.left.atoms.contains_key(&target)
                                && self.left.atoms.contains_key(&other_target)
                                && self.left.bond(target, other_target).is_none()
                                && self.right.atoms.contains_key(&target)
                                && self.right.atoms.contains_key(&other_target)
                        }
                    },
                }
            });
            if consistent {
                overlap.insert(atom, target);
                self.overlaps_into(walk, position + 1, overlap, visit);
                overlap.remove(&atom);
            }
        }
    }

    /// Whether every atom overlapped so far stays within its valence, counting
    /// the host's bonds on it and the bonds `then` adds that the host does
    /// not have — those to atoms decided not to overlap, or to overlapped
    /// atoms the host does not bond it to.
    fn fits(&self, walk: &Walk, decided: usize, overlap: &BTreeMap<usize, usize>) -> bool {
        overlap.iter().all(|(&atom, &target)| {
            let Some(allowed) = largest_valence(walk.host.atoms[&target]) else {
                return true;
            };
            let mut used: i32 = walk
                .host
                .bonds
                .iter()
                .filter(|&(&(a, b), _)| a == target || b == target)
                .map(|(_, &symbol)| half_units(symbol))
                .sum();
            for &(other, _) in &walk.left[..decided] {
                let Some(symbol) = walk.then.left.bond(atom, other) else {
                    continue;
                };
                match overlap.get(&other) {
                    None => used += half_units(symbol),
                    Some(&other_target) if walk.host.bond(target, other_target).is_none() => {
                        used += half_units(symbol)
                    }
                    Some(_) => {}
                }
            }
            used <= allowed
        })
    }

    /// Whether swapping two atoms leaves the rule as it was: the same labels
    /// on both sides, and the same bonds to every other atom.
    fn twins(&self, first: usize, second: usize) -> bool {
        let sides = [&self.left, &self.right];
        sides
            .iter()
            .all(|side| side.atoms.get(&first) == side.atoms.get(&second))
            && self
                .left
                .atoms
                .keys()
                .chain(self.right.atoms.keys())
                .filter(|&&other| other != first && other != second)
                .all(|&other| {
                    sides
                        .iter()
                        .all(|side| side.bond(first, other) == side.bond(second, other))
                })
    }

    /// The concurrent rule of `self` then `then` over one overlap of `then`'s
    /// L with `self`'s R, if `then` can follow `self` there.
    fn concurrent(&self, then: &ExactRule, overlap: &BTreeMap<usize, usize>) -> Option<ExactRule> {
        let mut next = self.max_index().max(then.max_index()) + 1;
        let mut rename: BTreeMap<usize, usize> = overlap.clone();
        for &index in then.left.atoms.keys().chain(then.right.atoms.keys()) {
            rename.entry(index).or_insert_with(|| {
                next += 1;
                next
            });
        }

        // What must be there before `self`: its own L, `then`'s atoms that
        // `self` does not provide, and `then`'s bonds that `self` neither
        // makes nor keeps.
        let mut start = self.left.clone();
        for (index, &label) in &then.left.atoms {
            if !overlap.contains_key(index) {
                start.atoms.insert(rename[index], label);
            }
        }
        for (&(a, b), &symbol) in &then.left.bonds {
            let (x, y) = (rename[&a], rename[&b]);
            if self.right.bond(x, y).is_some() {
                continue;
            }
            // A bond to an atom `self` creates cannot have been there before.
            for end in [x, y] {
                if self.right.atoms.contains_key(&end) && !self.left.atoms.contains_key(&end) {
                    return None;
                }
            }
            start.bonds.insert(pair(x, y), symbol);
        }

        let identity: BTreeMap<usize, usize> = self
            .left
            .atoms
            .keys()
            .chain(self.right.atoms.keys())
            .map(|&index| (index, index))
            .collect();
        if !start.within_valences() {
            return None;
        }
        let middle = self.rewrite(&start, &identity)?;
        let end = then.rewrite(&middle, &rename)?;
        if !middle.within_valences() || !end.within_valences() {
            return None;
        }
        Some(ExactRule {
            left: start,
            right: end,
        })
    }

    /// The critical pairs of two rules: the minimal graphs both apply to
    /// where applying one destroys the other's match.
    ///
    /// Each overlap of the two L sides is glued into one graph, kept if both
    /// rules apply to it, and reported if at least one of them no longer
    /// applies after the other has — the two are then in conflict, and which
    /// is applied first matters.
    pub fn critical_pairs(&self, other: &ExactRule) -> Vec<CriticalPair> {
        let host: Vec<(usize, Label)> = self.left.atoms.iter().map(|(&i, &l)| (i, l)).collect();
        let left = heavy_first(&other.left);
        let mut twins = Default::default();
        let mut found = Vec::new();
        let mut seen = BTreeSet::new();
        let mut overlap = BTreeMap::new();
        let walk = Walk::new(self, &self.left, other, &left, &host, true, &mut twins);
        self.overlaps_into(&walk, 0, &mut overlap, &mut |overlap| {
            if let Some(pair) = self.critical_pair(other, overlap) {
                let key = pair_key(&pair, self, overlap);
                if seen.insert(key) {
                    found.push(pair);
                }
            }
        });
        found
    }

    fn critical_pair(
        &self,
        other: &ExactRule,
        overlap: &BTreeMap<usize, usize>,
    ) -> Option<CriticalPair> {
        let mut next = self.max_index().max(other.max_index()) + 1;
        let mut rename: BTreeMap<usize, usize> = overlap.clone();
        for &index in other.left.atoms.keys().chain(other.right.atoms.keys()) {
            rename.entry(index).or_insert_with(|| {
                next += 1;
                next
            });
        }
        let mut graph = self.left.clone();
        for (index, &label) in &other.left.atoms {
            graph.atoms.entry(rename[index]).or_insert(label);
        }
        for (&(a, b), &symbol) in &other.left.bonds {
            graph
                .bonds
                .entry(pair(rename[&a], rename[&b]))
                .or_insert(symbol);
        }
        if !graph.within_valences() {
            return None;
        }
        // Fresh indices for what `self` creates, clear of `other`'s.
        let mut own: BTreeMap<usize, usize> = BTreeMap::new();
        for &index in self.left.atoms.keys() {
            own.insert(index, index);
        }
        for &index in self.right.atoms.keys() {
            own.entry(index).or_insert_with(|| {
                next += 1;
                next
            });
        }

        let after_self = self.rewrite(&graph, &own)?;
        let after_other = other.rewrite(&graph, &rename)?;
        let first_disables_second = other.rewrite(&after_self, &rename).is_none();
        let second_disables_first = self.rewrite(&after_other, &own).is_none();
        (first_disables_second || second_disables_first).then(|| CriticalPair {
            overlap: graph.to_smarts(),
            first_disables_second,
            second_disables_first,
        })
    }

    /// The canonical form: the rule renumbered so that any two rules equal up
    /// to renumbering come out identical, with its SMARTS as a key.
    pub fn canonical(&self) -> (String, ExactRule) {
        let atoms: Vec<usize> = self
            .left
            .atoms
            .keys()
            .chain(self.right.atoms.keys())
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let side = |label: Option<&Label>| label.map_or("_".into(), |l| format!("{l:?}"));
        let vertex_labels: Vec<String> = atoms
            .iter()
            .map(|index| {
                format!(
                    "{}>{}",
                    side(self.left.atoms.get(index)),
                    side(self.right.atoms.get(index))
                )
            })
            .collect();
        let position: BTreeMap<usize, usize> = atoms
            .iter()
            .enumerate()
            .map(|(p, &index)| (index, p))
            .collect();
        let mut edges: BTreeMap<(usize, usize), String> = BTreeMap::new();
        for &(a, b) in self.left.bonds.keys().chain(self.right.bonds.keys()) {
            let bond = |graph: &Graph| graph.bond(a, b).unwrap_or("_");
            edges.insert(
                pair(position[&a], position[&b]),
                format!("{}>{}", bond(&self.left), bond(&self.right)),
            );
        }
        let order = canonical_order(&vertex_labels, &edges);
        let renumber: BTreeMap<usize, usize> = order
            .iter()
            .enumerate()
            .map(|(rank, &p)| (atoms[p], rank + 1))
            .collect();
        let relabel = |graph: &Graph| Graph {
            atoms: graph
                .atoms
                .iter()
                .map(|(index, &label)| (renumber[index], label))
                .collect(),
            bonds: graph
                .bonds
                .iter()
                .map(|(&(a, b), &symbol)| (pair(renumber[&a], renumber[&b]), symbol))
                .collect(),
        };
        let canonical = ExactRule {
            left: relabel(&self.left),
            right: relabel(&self.right),
        };
        (canonical.to_smarts(), canonical)
    }
}

/// What [`ExactRule::overlaps_into`] walks: the atoms of `then`'s L
/// (`left`) onto the atoms of `host` (`targets`), gluing absent bonds in
/// (`glue`) or not.
#[derive(Clone, Copy)]
struct Walk<'a> {
    host: &'a Graph,
    then: &'a ExactRule,
    left: &'a [(usize, Label)],
    targets: &'a [(usize, Label)],
    glue: bool,
    /// For each target, the first target it is a twin of (itself if none).
    target_twin: &'a [usize],
    /// For each atom of `left`, the position of the last earlier atom it is a
    /// twin of.
    left_twin: &'a [Option<usize>],
}

impl<'a> Walk<'a> {
    fn new(
        host_rule: &ExactRule,
        host: &'a Graph,
        then: &'a ExactRule,
        left: &'a [(usize, Label)],
        targets: &'a [(usize, Label)],
        glue: bool,
        twins: &'a mut (Vec<usize>, Vec<Option<usize>>),
    ) -> Self {
        twins.0 = (0..targets.len())
            .map(|t| {
                (0..t)
                    .find(|&u| host_rule.twins(targets[u].0, targets[t].0))
                    .unwrap_or(t)
            })
            .collect();
        twins.1 = (0..left.len())
            .map(|a| (0..a).rev().find(|&b| then.twins(left[b].0, left[a].0)))
            .collect();
        Walk {
            host,
            then,
            left,
            targets,
            glue,
            target_twin: &twins.0,
            left_twin: &twins.1,
        }
    }
}

/// An atom label's largest valence, in half-units of bond order.
fn largest_valence(label: Label) -> Option<i32> {
    use chemistry_consts::ElementProperties;
    let largest = label
        .atomic_number
        .valencies()
        .and_then(|valences| valences.filter(|v| *v > 0).max())?;
    Some(2 * i32::from(largest + charge_valency_shift(label.atomic_number, label.charge)))
}

fn half_units(symbol: &str) -> i32 {
    match symbol {
        "=" => 4,
        "#" => 6,
        ":" => 3,
        _ => 2,
    }
}

/// The atoms of a rule's L, heavy atoms first: they constrain the most, so
/// deciding them first lets the overlap walk give up on a branch early.
fn heavy_first(graph: &Graph) -> Vec<(usize, Label)> {
    let mut atoms: Vec<(usize, Label)> = graph.atoms.iter().map(|(&i, &l)| (i, l)).collect();
    atoms.sort_by_key(|&(index, label)| (label.atomic_number == 1, index));
    atoms
}

/// Two rules overlapped so that applying one destroys the other's match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalPair {
    /// The glued graph both rules apply to, as an exact SMARTS. Atoms of the
    /// first rule keep its map indices; the second rule's other atoms are
    /// numbered past them.
    pub overlap: String,
    /// Applying the first rule leaves the second no match here.
    pub first_disables_second: bool,
    /// Applying the second rule leaves the first no match here.
    pub second_disables_first: bool,
}

/// A key under which two critical pairs that differ only by renumbering —
/// symmetric overlaps — coincide: the glued graph canonized with each atom
/// marked by which rules' L it belongs to.
fn pair_key(found: &CriticalPair, first: &ExactRule, overlap: &BTreeMap<usize, usize>) -> String {
    let pattern = crate::io::parse_cxsmarts(&found.overlap).expect("written by Graph::to_smarts");
    let atoms: Vec<usize> = (0..pattern.atoms.len()).collect();
    let index_of: Vec<usize> = atoms
        .iter()
        .map(|atom| pattern.atom_classes[atom])
        .collect();
    let images: BTreeSet<usize> = overlap.values().copied().collect();
    let labels: Vec<String> = atoms
        .iter()
        .map(|&atom| {
            let index = index_of[atom];
            let in_first = first.left.atoms.contains_key(&index);
            let in_both = images.contains(&index);
            format!("{:?}{in_first}{in_both}", pattern.atoms[atom])
        })
        .collect();
    let edges: BTreeMap<(usize, usize), String> = pattern
        .bonds
        .iter()
        .map(|(a, b, expression)| (pair(*a, *b), format!("{expression:?}")))
        .collect();
    let order = canonical_order(&labels, &edges);
    let mut key = String::new();
    let rank: BTreeMap<usize, usize> = order.iter().enumerate().map(|(r, &p)| (p, r)).collect();
    for &p in &order {
        write!(key, "{}|", labels[p]).unwrap();
    }
    let mut serialized: Vec<(usize, usize, &String)> = edges
        .iter()
        .map(|(&(a, b), label)| {
            let (x, y) = (rank[&a], rank[&b]);
            (x.min(y), x.max(y), label)
        })
        .collect();
    serialized.sort();
    write!(
        key,
        "{serialized:?}{}{}",
        found.first_disables_second, found.second_disables_first
    )
    .unwrap();
    key
}

/// A canonical ordering of a labelled graph's vertices, by
/// individualization and refinement: colour by label, refine each colour by
/// the colours around it until nothing splits, and where a class of
/// indistinguishable vertices remains, try each of them first and keep the
/// ordering whose serialization is smallest.
fn canonical_order(labels: &[String], edges: &BTreeMap<(usize, usize), String>) -> Vec<usize> {
    let count = labels.len();
    let mut adjacency: Vec<Vec<(usize, &str)>> = vec![Vec::new(); count];
    for (&(a, b), label) in edges {
        adjacency[a].push((b, label));
        adjacency[b].push((a, label));
    }
    let mut sorted_labels: Vec<&String> = labels.iter().collect();
    sorted_labels.sort();
    sorted_labels.dedup();
    let colours: Vec<usize> = labels
        .iter()
        .map(|label| sorted_labels.binary_search(&label).unwrap())
        .collect();

    let serialize = |order: &[usize]| -> String {
        let mut rank = vec![0; count];
        for (r, &v) in order.iter().enumerate() {
            rank[v] = r;
        }
        let mut text: String = order.iter().map(|&v| format!("{}|", labels[v])).collect();
        let mut serialized: Vec<(usize, usize, &str)> = edges
            .iter()
            .map(|(&(a, b), label)| {
                let (x, y) = (rank[a], rank[b]);
                (x.min(y), x.max(y), label.as_str())
            })
            .collect();
        serialized.sort();
        write!(text, "{serialized:?}").unwrap();
        text
    };

    let mut best: Option<(String, Vec<usize>)> = None;
    search(colours, &adjacency, &serialize, &mut best);
    best.map(|(_, order)| order).unwrap_or_default()
}

fn refine(colours: &mut Vec<usize>, adjacency: &[Vec<(usize, &str)>]) {
    loop {
        let keys: Vec<(usize, Vec<(&str, usize)>)> = (0..colours.len())
            .map(|v| {
                let mut around: Vec<(&str, usize)> = adjacency[v]
                    .iter()
                    .map(|&(u, label)| (label, colours[u]))
                    .collect();
                around.sort();
                (colours[v], around)
            })
            .collect();
        let mut distinct = keys.clone();
        distinct.sort();
        distinct.dedup();
        let before = colours.iter().collect::<BTreeSet<_>>().len();
        *colours = keys
            .iter()
            .map(|key| distinct.binary_search(key).unwrap())
            .collect();
        if distinct.len() == before {
            return;
        }
    }
}

fn search(
    mut colours: Vec<usize>,
    adjacency: &[Vec<(usize, &str)>],
    serialize: &dyn Fn(&[usize]) -> String,
    best: &mut Option<(String, Vec<usize>)>,
) {
    refine(&mut colours, adjacency);
    let mut sizes: BTreeMap<usize, usize> = BTreeMap::new();
    for &colour in &colours {
        *sizes.entry(colour).or_default() += 1;
    }
    match sizes.iter().find(|&(_, &size)| size > 1) {
        None => {
            let mut order: Vec<usize> = (0..colours.len()).collect();
            order.sort_by_key(|&v| colours[v]);
            let text = serialize(&order);
            if best.as_ref().is_none_or(|(smallest, _)| text < *smallest) {
                *best = Some((text, order));
            }
        }
        Some((&cell, _)) => {
            for v in (0..colours.len()).filter(|&v| colours[v] == cell) {
                let individualized: Vec<usize> = colours
                    .iter()
                    .enumerate()
                    .map(|(u, &colour)| if u == v { 2 * colour } else { 2 * colour + 1 })
                    .collect();
                search(individualized, adjacency, serialize, best);
            }
        }
    }
}

impl ReactionSmarts {
    /// The exact form of this rule — see [`ExactRule`].
    pub fn to_exact(&self) -> Result<ExactRule, ExactRuleError> {
        ExactRule::from_reaction(self)
    }

    /// A SMARTS for this rule that any rule equal to it up to renumbering
    /// shares, and no other rule does.
    pub fn canonical_smarts(&self) -> Result<String, ExactRuleError> {
        Ok(self.to_exact()?.canonical().0)
    }

    /// Whether two rules are the same rule, up to renumbering.
    pub fn is_isomorphic_to(&self, other: &ReactionSmarts) -> Result<bool, ExactRuleError> {
        Ok(self.canonical_smarts()? == other.canonical_smarts()?)
    }

    /// The concurrent rules of this rule followed by `then` — see
    /// [`ExactRule::compose`].
    ///
    /// ```rust
    /// # use oxidane::prelude::*;
    /// use oxidane::molecule::reaction::ReactionSmarts;
    ///
    /// // Oxidation to the aldehyde, then on to the acid with water's oxygen,
    /// // composes into one rule that does both.
    /// let to_aldehyde =
    ///     ReactionSmarts::parse_dpo("[C:1](-[#1:3])-[O:2]-[#1:4]>>[C:1]=[O:2].[#1:3]-[#1:4]").unwrap();
    /// let to_acid = ReactionSmarts::parse_dpo(
    ///     "[C:1](=[O:2])-[#1:3].[O:4](-[#1:5])-[#1:6]>>[C:1](=[O:2])-[O:4]-[#1:5].[#1:3]-[#1:6]",
    /// )
    /// .unwrap();
    /// let both = to_aldehyde.compose(&to_acid).unwrap();
    /// let ethanol_and_water = [
    ///     MolecularGraph::from_smiles_single("CCO").unwrap(),
    ///     MolecularGraph::from_smiles_single("O").unwrap(),
    /// ];
    /// let references: Vec<&MolecularGraph> = ethanol_and_water.iter().collect();
    /// assert!(both.iter().any(|rule| {
    ///     rule.apply_to(&references)
    ///         .iter()
    ///         .any(|product| product.contains_smarts("[CH3]C(=O)[OH]").unwrap())
    /// }));
    /// ```
    pub fn compose(&self, then: &ReactionSmarts) -> Result<Vec<ReactionSmarts>, ExactRuleError> {
        self.to_exact()?
            .compose(&then.to_exact()?)
            .iter()
            .map(|rule| rule.to_reaction().map_err(ExactRuleError::from))
            .collect()
    }

    /// The critical pairs of this rule and `other` — see
    /// [`ExactRule::critical_pairs`].
    pub fn critical_pairs(
        &self,
        other: &ReactionSmarts,
    ) -> Result<Vec<CriticalPair>, ExactRuleError> {
        Ok(self.to_exact()?.critical_pairs(&other.to_exact()?))
    }
}
