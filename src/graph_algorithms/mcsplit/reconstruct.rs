use super::graph::Graph;
use rustc_hash::FxHashMap;

/// Turns a matched set of *bonds* (line-graph vertex pairs, as found by
/// [`super::search::search`] over the two molecules' line graphs) into a
/// concrete atom correspondence.
///
/// A bond-pair's orientation (does its first atom map to the other bond's
/// first or second atom?) is never guessed or propagated from a
/// processing-order-dependent choice — that approach silently produced
/// self-contradictory mappings in `mces`'s original reconstruction (the same
/// target atom claimed by two different source atoms) whenever a shared
/// atom sat at *different* relative positions across the two bonds that
/// share it, which branched molecules do routinely (see `mces::reconstruct`
/// for the concrete case that caught it). Instead, whenever a matched bond
/// shares an atom with another matched bond (in both molecules — that's
/// what makes them line-graph-adjacent), the orientation is *computed
/// directly* from that structural relationship: exactly one atom is common
/// to both bonds' `mol_a` endpoints, and exactly one is common to both
/// `mol_b` endpoints, so the orientation sending the former to the latter is
/// fully determined, independent of which bond gets processed first. Only a
/// bond with no such matched neighbor (an independently-matched fragment)
/// has nothing to derive from, and picks an arbitrary — but then
/// unconstrained, so always safe — starting orientation via atom-label
/// comparison.
pub(super) fn resolve(
    g1: &Graph,
    g2: &Graph,
    matched_bonds: &[(usize, usize)],
    endpoints1: &[(usize, usize)],
    endpoints2: &[(usize, usize)],
) -> (Vec<(usize, usize)>, Vec<((usize, usize), (usize, usize))>) {
    struct BondInfo {
        a1: usize,
        a2: usize,
        b1: usize,
        b2: usize,
    }

    let infos: Vec<BondInfo> = matched_bonds
        .iter()
        .map(|&(bi, bj)| {
            let (a1, a2) = endpoints1[bi];
            let (b1, b2) = endpoints2[bj];
            BondInfo { a1, a2, b1, b2 }
        })
        .collect();

    let mut orientation = vec![false; infos.len()];
    for (idx, info) in infos.iter().enumerate() {
        let mut derived = None;

        for (other_idx, other) in infos.iter().enumerate() {
            if other_idx == idx {
                continue;
            }
            let shared_a = if info.a1 == other.a1 || info.a1 == other.a2 {
                Some(info.a1)
            } else if info.a2 == other.a1 || info.a2 == other.a2 {
                Some(info.a2)
            } else {
                None
            };
            let Some(shared_a) = shared_a else { continue };

            let shared_b = if info.b1 == other.b1 || info.b1 == other.b2 {
                Some(info.b1)
            } else if info.b2 == other.b1 || info.b2 == other.b2 {
                Some(info.b2)
            } else {
                None
            };
            let Some(shared_b) = shared_b else { continue };

            let required = if shared_a == info.a1 { shared_b == info.b1 } else { shared_b == info.b2 };
            derived = Some(required);
            break;
        }

        orientation[idx] = derived.unwrap_or_else(|| g1.labels[info.a1] == g2.labels[info.b1]);
    }

    let mut atom_map: FxHashMap<usize, usize> = FxHashMap::default();
    for (idx, info) in infos.iter().enumerate() {
        let (mapped_a1, mapped_a2) =
            if orientation[idx] { (info.b1, info.b2) } else { (info.b2, info.b1) };
        atom_map.insert(info.a1, mapped_a1);
        atom_map.insert(info.a2, mapped_a2);
    }

    let bond_pairs: Vec<((usize, usize), (usize, usize))> = infos
        .iter()
        .map(|info| ((info.a1, info.a2), (atom_map[&info.a1], atom_map[&info.a2])))
        .collect();

    let mut atom_pairs: Vec<(usize, usize)> = atom_map.into_iter().collect();
    atom_pairs.sort_unstable();

    (atom_pairs, bond_pairs)
}
