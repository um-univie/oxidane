//! SMILES as a sequence of tokens, for a model that reads or writes one.
//!
//! A graph encoder turns a molecule into a vector. Turning a vector back into
//! a molecule is the harder direction, and the cheapest way to make it
//! tractable is to decode a *string* rather than a graph: a sequence model
//! emits one token at a time, and the crate's own parser says whether what it
//! emitted was a molecule. Nothing here is a neural network -- this is the
//! alphabet the network writes in, and it needs no tensor library, so it sits
//! beside [`crate::ml::batch`] rather than under `crate::ml::nn`.
//!
//! # What counts as one token
//!
//! Characters are the wrong unit. `Cl` is one atom and `[nH]` is one atom, and
//! a model that has to spell them cannot be stopped from spelling `[nX]`. So a
//! bracket atom is one token whole, the two-letter organic elements are one
//! token, a ring closure written `%12` is one token, and everything else is a
//! single character.
//!
//! ```
//! use oxidane::ml::tokenize::tokenize;
//!
//! assert_eq!(tokenize("CC(=O)Cl"), ["C", "C", "(", "=", "O", ")", "Cl"]);
//! assert_eq!(tokenize("c1cc[nH]c1"), ["c", "1", "c", "c", "[nH]", "c", "1"]);
//! ```
//!
//! # The vocabulary is built from the corpus
//!
//! Bracket atoms are open-ended -- `[13CH3]`, `[Fe+3]`, `[Si@](C)` -- so the
//! token set cannot be written down in advance. [`SmilesVocabulary::from_corpus`]
//! counts what actually appears and keeps what appears often enough; a token
//! below the threshold becomes [`UNKNOWN`], and a molecule containing one is
//! better dropped from a training set than learned as a hole (see
//! [`SmilesVocabulary::encode_known`]).

use std::collections::HashMap;

/// Padding, in the positions after a sequence has ended. Never predicted: a
/// loss masks these away.
pub const PADDING: u32 = 0;
/// The token a decoder starts from, before it has emitted anything.
pub const START: u32 = 1;
/// The token that ends a molecule. A decoder that has not emitted this by its
/// length limit produced a truncated string, which will not parse.
pub const END: u32 = 2;
/// Stands for a token the vocabulary does not have.
pub const UNKNOWN: u32 = 3;
/// How many of the ids above are reserved.
pub const RESERVED: usize = 4;

/// Splits a SMILES into the units a model should learn.
///
/// The slices borrow from `smiles`, and concatenating them reproduces it
/// exactly -- nothing is dropped, including whitespace, which is why a caller
/// should hand over a trimmed string.
pub fn tokenize(smiles: &str) -> Vec<&str> {
    let bytes = smiles.as_bytes();
    let mut tokens = Vec::with_capacity(bytes.len());
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let length = token_length(bytes, cursor);
        tokens.push(&smiles[cursor..cursor + length]);
        cursor += length;
    }
    tokens
}

/// How many bytes the token starting at `cursor` occupies. Always at least
/// one, so a caller cannot loop forever on an unexpected byte.
fn token_length(bytes: &[u8], cursor: usize) -> usize {
    match bytes[cursor] {
        // A bracket atom whole, so that a model cannot invent `[nX]`. An
        // unclosed bracket is a malformed SMILES; it is taken to the end
        // rather than dropped, and the parser will reject it later.
        b'[' => match bytes[cursor..].iter().position(|&byte| byte == b']') {
            Some(offset) => offset + 1,
            None => bytes.len() - cursor,
        },
        // A ring closure of ten or more: `%` and two digits.
        b'%' if cursor + 2 < bytes.len()
            && bytes[cursor + 1].is_ascii_digit()
            && bytes[cursor + 2].is_ascii_digit() =>
        {
            3
        }
        // The two-letter organic-subset elements. `Sc`, `Se` and the rest need
        // brackets, so two letters outside a bracket can only be these.
        b'C' if bytes.get(cursor + 1) == Some(&b'l') => 2,
        b'B' if bytes.get(cursor + 1) == Some(&b'r') => 2,
        // Everything else is one character. SMILES outside brackets is ASCII,
        // but a malformed input need not be, so the character's own width is
        // what advances the cursor.
        _ => {
            let mut length = 1;
            while cursor + length < bytes.len() && (bytes[cursor + length] & 0xC0) == 0x80 {
                length += 1;
            }
            length
        }
    }
}

/// The tokens a model may read and write, and their numbering.
///
/// Ids `0..`[`RESERVED`] are the special tokens; everything after them is a
/// corpus token, held in the order [`SmilesVocabulary::tokens`] reports so
/// that saving and loading round trip.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SmilesVocabulary {
    tokens: Vec<String>,
    index: HashMap<String, u32>,
}

impl SmilesVocabulary {
    /// A vocabulary over exactly `tokens`, in the order given.
    ///
    /// Duplicates are dropped, keeping the first occurrence. The special
    /// tokens are added by this constructor and must not appear in `tokens`.
    pub fn from_tokens<I, S>(tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut vocabulary = Self {
            tokens: Vec::new(),
            index: HashMap::new(),
        };
        for token in tokens {
            let token = token.into();
            if vocabulary.index.contains_key(&token) {
                continue;
            }
            let id = (RESERVED + vocabulary.tokens.len()) as u32;
            vocabulary.index.insert(token.clone(), id);
            vocabulary.tokens.push(token);
        }
        vocabulary
    }

    /// The tokens of `corpus` that appear at least `minimum_count` times.
    ///
    /// The order is by falling count and then by the token itself, so the
    /// numbering depends on the corpus and not on the order it was read in --
    /// two runs over the same set of molecules produce the same vocabulary,
    /// and a model saved against one can be loaded against the other.
    ///
    /// ```
    /// use oxidane::ml::tokenize::SmilesVocabulary;
    ///
    /// let vocabulary = SmilesVocabulary::from_corpus(["CCO", "CCC", "[Fe]"], 2);
    /// // C and O clear the threshold; the iron appears once and does not.
    /// assert!(vocabulary.contains("C"));
    /// assert!(!vocabulary.contains("[Fe]"));
    /// ```
    pub fn from_corpus<I, S>(corpus: I, minimum_count: usize) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // Held before the counts so that the counts, which borrow from it, are
        // dropped first.
        let owned: Vec<S> = corpus.into_iter().collect();
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for smiles in &owned {
            for token in tokenize(smiles.as_ref()) {
                *counts.entry(token).or_insert(0) += 1;
            }
        }
        let mut kept: Vec<(&str, usize)> = counts
            .into_iter()
            .filter(|&(_, count)| count >= minimum_count.max(1))
            .collect();
        kept.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));
        Self::from_tokens(kept.into_iter().map(|(token, _)| token))
    }

    /// How many ids there are, the special tokens included. This is the width
    /// a decoder's output layer needs.
    pub fn len(&self) -> usize {
        RESERVED + self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The corpus tokens, in id order starting at [`RESERVED`].
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    pub fn contains(&self, token: &str) -> bool {
        self.index.contains_key(token)
    }

    /// The id of `token`, or [`UNKNOWN`] if the vocabulary does not have it.
    pub fn id(&self, token: &str) -> u32 {
        self.index.get(token).copied().unwrap_or(UNKNOWN)
    }

    /// What `id` stands for. The special ids give the empty string, so that
    /// [`SmilesVocabulary::decode`] can concatenate without a branch.
    pub fn token(&self, id: u32) -> &str {
        self.tokens
            .get((id as usize).saturating_sub(RESERVED))
            .filter(|_| id as usize >= RESERVED)
            .map(String::as_str)
            .unwrap_or("")
    }

    /// `smiles` as ids, with anything unrecognised becoming [`UNKNOWN`].
    pub fn encode(&self, smiles: &str) -> Vec<u32> {
        tokenize(smiles)
            .into_iter()
            .map(|token| self.id(token))
            .collect()
    }

    /// [`SmilesVocabulary::encode`], or `None` if any token is unrecognised.
    ///
    /// What to use when building a training set: a molecule the vocabulary
    /// cannot spell is one the decoder could never reproduce, and leaving it
    /// in teaches the model to emit [`UNKNOWN`] where an atom belongs.
    pub fn encode_known(&self, smiles: &str) -> Option<Vec<u32>> {
        tokenize(smiles)
            .into_iter()
            .map(|token| self.index.get(token).copied())
            .collect()
    }

    /// The string those ids spell, ignoring the special tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            out.push_str(self.token(id));
        }
        out
    }

    /// The vocabulary as one token per line, for writing beside a model.
    ///
    /// A model's output layer is the vocabulary's width and its columns are
    /// the vocabulary's ids, so weights without the vocabulary they were
    /// trained against are unusable.
    pub fn to_lines(&self) -> String {
        let mut out = String::new();
        for token in &self.tokens {
            out.push_str(token);
            out.push('\n');
        }
        out
    }

    /// The inverse of [`SmilesVocabulary::to_lines`]. Blank lines are skipped.
    pub fn from_lines(text: &str) -> Self {
        Self::from_tokens(text.lines().filter(|line| !line.is_empty()))
    }
}

/// A padded rectangle of token sequences, ready to become tensors.
///
/// Both halves of teacher forcing are here: `inputs` is what the decoder is
/// fed at each step and `targets` is what it should have said, so
/// `targets[t]` is the answer to `inputs[t]`. The two are the same sequence
/// offset by one, with [`START`] pushed onto the front of the first and
/// [`END`] onto the end of the second.
///
/// ```
/// use oxidane::ml::tokenize::{SmilesVocabulary, TokenBatch, END, PADDING, START};
///
/// let vocabulary = SmilesVocabulary::from_corpus(["CCO", "CO"], 1);
/// let batch = TokenBatch::collate(&[vocabulary.encode("CCO"), vocabulary.encode("CO")]);
///
/// // Three tokens plus the end marker is the longest row.
/// assert_eq!((batch.rows, batch.width), (2, 4));
/// assert_eq!(batch.inputs[0], START);
/// assert_eq!(batch.targets[batch.width - 1], END);
/// // The shorter molecule ends early and is padded, and the padding is masked.
/// assert_eq!(batch.targets[batch.width + 3], PADDING);
/// assert_eq!(batch.mask[batch.width + 3], 0.0);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TokenBatch {
    /// How many sequences there are.
    pub rows: usize,
    /// How many steps each row has: the longest sequence, plus the end marker.
    pub width: usize,
    /// `rows * width` ids, what the decoder reads.
    pub inputs: Vec<u32>,
    /// `rows * width` ids, what it should emit.
    pub targets: Vec<u32>,
    /// `rows * width` ones and zeros: which steps are a real prediction.
    pub mask: Vec<f32>,
}

impl TokenBatch {
    /// Lays the sequences out as one padded rectangle.
    ///
    /// An empty sequence is still one step long -- the molecule with no tokens
    /// has [`END`] as its whole target -- so a batch always has a step to
    /// learn from.
    pub fn collate(sequences: &[Vec<u32>]) -> Self {
        let width = sequences.iter().map(Vec::len).max().unwrap_or(0) + 1;
        let rows = sequences.len();
        let mut batch = Self {
            rows,
            width,
            inputs: vec![PADDING; rows * width],
            targets: vec![PADDING; rows * width],
            mask: vec![0.0; rows * width],
        };
        for (row, sequence) in sequences.iter().enumerate() {
            let start = row * width;
            batch.inputs[start] = START;
            for (step, &id) in sequence.iter().enumerate() {
                batch.inputs[start + step + 1] = id;
                batch.targets[start + step] = id;
                batch.mask[start + step] = 1.0;
            }
            batch.targets[start + sequence.len()] = END;
            batch.mask[start + sequence.len()] = 1.0;
        }
        // The last input column is never read -- its target is past the end of
        // the longest sequence -- but keeping the rectangle square is cheaper
        // than special-casing it in the decoder.
        batch
    }

    /// One row's ids, up to and not including the end marker.
    pub fn sequence(&self, row: usize) -> &[u32] {
        let start = row * self.width;
        let length = self.mask[start..start + self.width]
            .iter()
            .position(|&kept| kept == 0.0)
            .unwrap_or(self.width);
        &self.targets[start..start + length.saturating_sub(1)]
    }
}
