//! Maximum matching, including the odd-cycle cases a greedy search gets wrong.
//!
//! The sizes here are checked against what is provably maximum, not against
//! what the implementation happens to return: a path on `n` vertices matches
//! `n/2`, an odd cycle on `2k+1` matches `k`, and the Petersen graph -- the
//! standard blossom stress case -- has a perfect matching.

#[cfg(test)]
mod matching_tests {
    use crate::molecule::matching::maximum_matching;

    fn size(matched: &[Option<usize>]) -> usize {
        matched.iter().filter(|m| m.is_some()).count() / 2
    }

    /// Every matched pair must agree with each other, and no vertex may appear
    /// twice. Checked on every case below.
    fn assert_consistent(matched: &[Option<usize>], edges: &[(usize, usize)]) {
        for (vertex, partner) in matched.iter().enumerate() {
            let Some(partner) = *partner else { continue };
            assert_eq!(matched[partner], Some(vertex), "{vertex} and {partner} disagree");
            assert!(
                edges.contains(&(vertex, partner)) || edges.contains(&(partner, vertex)),
                "{vertex}-{partner} is not an edge"
            );
        }
    }

    #[test]
    fn a_path_matches_half_its_vertices() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];
        let matched = maximum_matching(6, &edges);
        assert_consistent(&matched, &edges);
        assert_eq!(size(&matched), 3);
    }

    /// The case that defeats greedy augmenting without blossoms: a triangle
    /// with a tail. Matching the tail first strands the odd cycle.
    #[test]
    fn a_triangle_with_a_tail() {
        let edges = [(0, 1), (1, 2), (2, 0), (2, 3)];
        let matched = maximum_matching(4, &edges);
        assert_consistent(&matched, &edges);
        assert_eq!(size(&matched), 2);
    }

    #[test]
    fn an_odd_cycle_leaves_exactly_one_vertex_free() {
        for k in 1..=5 {
            let n = 2 * k + 1;
            let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
            let matched = maximum_matching(n, &edges);
            assert_consistent(&matched, &edges);
            assert_eq!(size(&matched), k, "odd cycle on {n} vertices");
        }
    }

    #[test]
    fn an_even_cycle_matches_perfectly() {
        for k in 2..=6 {
            let n = 2 * k;
            let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
            let matched = maximum_matching(n, &edges);
            assert_consistent(&matched, &edges);
            assert_eq!(size(&matched), k, "even cycle on {n} vertices");
        }
    }

    /// Two triangles joined by a bridge: three blossoms' worth of trouble, and
    /// a perfect matching all the same.
    #[test]
    fn two_triangles_joined_by_a_bridge() {
        let edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)];
        let matched = maximum_matching(6, &edges);
        assert_consistent(&matched, &edges);
        assert_eq!(size(&matched), 3);
    }

    /// The Petersen graph: 3-regular, girth 5, and the usual proving ground for
    /// blossom implementations. It has a perfect matching.
    #[test]
    fn the_petersen_graph_matches_perfectly() {
        let mut edges = Vec::new();
        for i in 0..5 {
            edges.push((i, (i + 1) % 5)); // outer pentagon
            edges.push((i, i + 5)); // spokes
            edges.push((5 + i, 5 + (i + 2) % 5)); // inner pentagram
        }
        let matched = maximum_matching(10, &edges);
        assert_consistent(&matched, &edges);
        assert_eq!(size(&matched), 5);
    }

    #[test]
    fn an_isolated_vertex_stays_unmatched() {
        let edges = [(0, 1)];
        let matched = maximum_matching(3, &edges);
        assert_consistent(&matched, &edges);
        assert_eq!(size(&matched), 1);
        assert_eq!(matched[2], None);
    }

    /// Naphthalene's carbon skeleton: ten atoms, every one of which can take a
    /// π bond, so the maximum matching is perfect -- which is why naphthalene
    /// kekulizes and why every carbon donates to its ring.
    #[test]
    fn a_fused_bicyclic_matches_perfectly() {
        let edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
            (4, 6), (6, 7), (7, 8), (8, 9), (9, 3),
        ];
        let matched = maximum_matching(10, &edges);
        assert_consistent(&matched, &edges);
        assert_eq!(size(&matched), 5);
    }
}
