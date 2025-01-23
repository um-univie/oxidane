//! Stereo descriptors and the like/unlike pair lists that CIP rules 4b and 5
//! rank them with.
//!
//! Ported from RDKit's `Code/GraphMol/CIPLabeler/{Descriptor.h,rules/Pairlist.h}`.
//! See the module docs in [`super`] for provenance and licensing.

/// A stereo descriptor, in the full set the CIP rules need internally.
///
/// The lowercase variants (`r`, `s`, `m`, `p`) are *pseudo-asymmetric*: a
/// centre that is only stereogenic because of the descriptors of the branches
/// around it. Rule 4c ranks those against each other, and Rule 4a ranks any
/// pseudo-asymmetric descriptor below any ordinary one, so they cannot be
/// collapsed into `R`/`S` before the rules have run.
///
/// `SeqCis`/`SeqTrans` are the double-bond equivalents that participate in
/// like/unlike pairing (see [`PairList`]); `E`/`Z` are the labels that get
/// reported. `Ns` means "not stereogenic" — a centre that was examined and
/// found to have no configuration, as distinct from `Unknown` ("the rules ran
/// out before deciding") and `None` ("nothing assigned").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub(super) enum Descriptor {
    #[default]
    None,
    Unknown,
    /// Examined and found not to be a stereocentre.
    Ns,
    R,
    S,
    /// Pseudo-asymmetric *r*.
    LowerR,
    /// Pseudo-asymmetric *s*.
    LowerS,
    SeqTrans,
    SeqCis,
    E,
    Z,
    M,
    P,
    /// Pseudo-asymmetric *m*.
    LowerM,
    /// Pseudo-asymmetric *p*.
    LowerP,
}

impl Descriptor {
    /// Collapses a descriptor onto the `R`/`S` axis used for like/unlike
    /// pairing. Everything that is not one of the six pairable descriptors
    /// maps to [`Descriptor::None`].
    pub(super) fn pair_ref(self) -> Descriptor {
        match self {
            Descriptor::R | Descriptor::M | Descriptor::SeqCis => Descriptor::R,
            Descriptor::S | Descriptor::P | Descriptor::SeqTrans => Descriptor::S,
            _ => Descriptor::None,
        }
    }
}

/// An ordered list of descriptors encountered while walking outwards from a
/// substituent, reduced to the `R`/`S` axis, used by rules 4b and 5.
///
/// Only the *pattern* of agreement with the first (reference) descriptor
/// matters: two lists are compared position by position, and the first
/// position where one agrees with its own reference and the other does not
/// decides. That is exactly the "like precedes unlike" of CIP rule 4b, and
/// letting the reference differ between the two lists is what lets a pair of
/// substituents be compared even though each is described relative to its own
/// handedness.
#[derive(Debug, Clone, Default)]
pub(super) struct PairList {
    descriptors: Vec<Descriptor>,
}

impl PairList {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn with_ref(reference: Descriptor) -> Self {
        let mut list = Self::new();
        list.add(reference);
        list
    }

    pub(super) fn ref_descriptor(&self) -> Descriptor {
        self.descriptors.first().copied().unwrap_or(Descriptor::None).pair_ref()
    }

    /// Appends a descriptor, reduced to the `R`/`S` axis. Descriptors that do
    /// not pair (including pseudo-asymmetric ones) are skipped entirely rather
    /// than recorded as a non-match, which is what keeps two lists the same
    /// length when they describe topologically equivalent substituents.
    pub(super) fn add(&mut self, descriptor: Descriptor) -> bool {
        match descriptor {
            Descriptor::R
            | Descriptor::S
            | Descriptor::M
            | Descriptor::P
            | Descriptor::SeqTrans
            | Descriptor::SeqCis => {
                self.descriptors.push(descriptor.pair_ref());
                true
            }
            _ => false,
        }
    }

    /// Compares two pair lists: `+1` if `self` is preferred.
    ///
    /// Lists of different lengths cannot arise from topologically equivalent
    /// substituents, which is the only situation the callers construct them
    /// in; a mismatch is treated as "cannot decide" rather than an error so a
    /// pathological input degrades to an unlabelled centre instead of a panic.
    pub(super) fn compare_to(&self, other: &PairList) -> i32 {
        if self.descriptors.len() != other.descriptors.len() {
            return 0;
        }
        let Some(&this_ref) = self.descriptors.first() else {
            return 0;
        };
        let Some(&that_ref) = other.descriptors.first() else {
            return 0;
        };
        for i in 1..self.descriptors.len() {
            let this_like = this_ref == self.descriptors[i];
            let that_like = that_ref == other.descriptors[i];
            if this_like && !that_like {
                return 1;
            }
            if !this_like && that_like {
                return -1;
            }
        }
        0
    }
}
