//! Double-pushout rules in MØD's GML format.
//!
//! A rule is three graphs: `left` holds what the rule removes, `right` what it
//! adds, and `context` what it keeps. L is `left` with `context`, R is
//! `right` with `context`, and K — the atoms and bonds both sides share — is
//! `context` itself. A node id names the same atom wherever it appears, which
//! is the atom map of a reaction SMARTS under another name, and that is how a
//! rule read here becomes a [`ReactionSmarts`] applied with
//! [`Semantics::Dpo`](crate::molecule::reaction::Semantics::Dpo).
//!
//! ```text
//! rule [
//!    ruleID "keto-enol"
//!    left [
//!       edge [ source 1 target 2 label "=" ]
//!       edge [ source 2 target 3 label "-" ]
//!       edge [ source 3 target 4 label "-" ]
//!    ]
//!    context [
//!       node [ id 1 label "C" ]
//!       node [ id 2 label "C" ]
//!       node [ id 3 label "O" ]
//!       node [ id 4 label "H" ]
//!    ]
//!    right [
//!       edge [ source 1 target 2 label "-" ]
//!       edge [ source 2 target 3 label "=" ]
//!       edge [ source 1 target 4 label "-" ]
//!    ]
//! ]
//! ```
//!
//! A label is exact, as in MØD: `C` is a neutral carbon and `O-` an oxide,
//! `13C` a carbon-13. Edge labels are `-`, `=`, `#` and `:`. MØD's term labels
//! and adjacency constraints have no counterpart here and are refused rather
//! than ignored.

use std::collections::BTreeMap;
use std::fmt::Write;

use chemistry_consts::ElementProperties;

use crate::molecule::reaction::{ReactionError, ReactionSmarts};
use crate::molecule::rule_algebra::{ExactRule, ExactRuleError, Graph, Label};

/// Why a GML rule could not be read or written.
#[derive(Debug, Clone, PartialEq)]
pub enum GmlRuleError {
    /// The text is not well-formed GML, or not a rule.
    Syntax(String),
    /// Well-formed, but uses something this reader has no counterpart for.
    Unsupported(String),
    /// A label the reaction writes that GML's exact labels cannot say.
    NotExpressible { index: usize, what: String },
    /// Only a double-pushout rule without application conditions has a GML
    /// form.
    NotDpo,
    /// The rule read, but is not a valid double-pushout rule.
    Rule(ReactionError),
}

impl std::fmt::Display for GmlRuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GmlRuleError::Syntax(message) => write!(f, "GML syntax: {message}"),
            GmlRuleError::Unsupported(what) => write!(f, "Unsupported in a GML rule: {what}"),
            GmlRuleError::NotExpressible { index, what } => {
                write!(f, "Atom :{index} has no exact GML label: {what}")
            }
            GmlRuleError::NotDpo => write!(
                f,
                "Only a double-pushout rule without conditions has a GML form"
            ),
            GmlRuleError::Rule(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for GmlRuleError {}

impl From<ReactionError> for GmlRuleError {
    fn from(error: ReactionError) -> Self {
        GmlRuleError::Rule(error)
    }
}

impl From<ExactRuleError> for GmlRuleError {
    fn from(error: ExactRuleError) -> Self {
        match error {
            ExactRuleError::NotDpo | ExactRuleError::Conditions => GmlRuleError::NotDpo,
            ExactRuleError::NotExact { index, what } => {
                GmlRuleError::NotExpressible { index, what }
            }
            ExactRuleError::Rule(error) => GmlRuleError::Rule(error),
        }
    }
}

/// A GML value: GML is keys paired with integers, strings and nested lists.
#[derive(Debug)]
enum Value {
    Integer(i64),
    Text(String),
    List(Vec<(String, Value)>),
}

/// Reads `key value` pairs until the input or the enclosing list ends.
fn parse_list(
    tokens: &mut std::iter::Peekable<std::vec::IntoIter<String>>,
) -> Result<Vec<(String, Value)>, GmlRuleError> {
    let mut entries = Vec::new();
    while let Some(key) = tokens.next() {
        if key == "]" {
            return Ok(entries);
        }
        let value = match tokens.next().as_deref() {
            Some("[") => Value::List(parse_list(tokens)?),
            Some(text) if text.starts_with('"') => Value::Text(text.trim_matches('"').to_string()),
            Some(number) => Value::Integer(
                number
                    .parse()
                    .map_err(|_| GmlRuleError::Syntax(format!("`{key}` has value `{number}`")))?,
            ),
            None => return Err(GmlRuleError::Syntax(format!("`{key}` has no value"))),
        };
        entries.push((key, value));
    }
    Ok(entries)
}

/// Splits GML into brackets, quoted strings and bare words; `#` starts a
/// comment that runs to the end of the line.
fn tokenize(gml: &str) -> Result<Vec<String>, GmlRuleError> {
    let mut tokens = Vec::new();
    let mut characters = gml.chars().peekable();
    while let Some(&character) = characters.peek() {
        match character {
            _ if character.is_whitespace() => {
                characters.next();
            }
            '#' => while characters.next_if(|&c| c != '\n').is_some() {},
            '[' | ']' => {
                tokens.push(character.to_string());
                characters.next();
            }
            '"' => {
                characters.next();
                let mut text = String::from('"');
                loop {
                    match characters.next() {
                        Some('"') => break,
                        Some(c) => text.push(c),
                        None => return Err(GmlRuleError::Syntax("unterminated string".into())),
                    }
                }
                text.push('"');
                tokens.push(text);
            }
            _ => {
                let mut word = String::new();
                while let Some(c) =
                    characters.next_if(|c| !c.is_whitespace() && !matches!(c, '[' | ']' | '"'))
                {
                    word.push(c);
                }
                tokens.push(word);
            }
        }
    }
    Ok(tokens)
}

/// Reads MØD's spelling of a label: an optional mass number, a symbol, and an
/// optional charge — `+`, `2-`, or `+2`.
fn parse_label(text: &str) -> Result<Label, GmlRuleError> {
    let unsupported = || GmlRuleError::Unsupported(format!("atom label `{text}`"));
    let digits = text.chars().take_while(char::is_ascii_digit).count();
    let isotope = (digits > 0)
        .then(|| text[..digits].parse().map_err(|_| unsupported()))
        .transpose()?;
    let rest = &text[digits..];
    let letters = rest.chars().take_while(char::is_ascii_alphabetic).count();
    let atomic_number = (&rest[..letters]).atomic_number().ok_or_else(unsupported)?;
    let charge_text = &rest[letters..];
    let charge = if charge_text.is_empty() {
        0
    } else {
        let (sign, magnitude) = if charge_text.ends_with(['+', '-']) {
            let (magnitude, sign) = charge_text.split_at(charge_text.len() - 1);
            (sign, magnitude)
        } else {
            charge_text.split_at(1)
        };
        let magnitude: i8 = if magnitude.is_empty() {
            1
        } else {
            magnitude.parse().map_err(|_| unsupported())?
        };
        match sign {
            "+" => magnitude,
            "-" => -magnitude,
            _ => return Err(unsupported()),
        }
    };
    Ok(Label {
        isotope,
        atomic_number,
        charge,
    })
}

/// A label as MØD writes it.
fn label_to_gml(label: Label) -> Option<String> {
    let mut text = String::new();
    if let Some(isotope) = label.isotope {
        write!(text, "{isotope}").ok()?;
    }
    text.push_str(label.atomic_number.atomic_symbol()?);
    match label.charge {
        0 => {}
        1 => text.push('+'),
        -1 => text.push('-'),
        charge if charge > 0 => write!(text, "{charge}+").ok()?,
        charge => write!(text, "{}-", -charge).ok()?,
    }
    Some(text)
}

/// The SMARTS symbol for a GML edge label.
fn edge_symbol(label: &str) -> Result<&'static str, GmlRuleError> {
    match label {
        "-" => Ok("-"),
        "=" => Ok("="),
        "#" => Ok("#"),
        ":" => Ok(":"),
        _ => Err(GmlRuleError::Unsupported(format!("edge label `{label}`"))),
    }
}

/// The value of the first entry named `name`.
fn field<'a>(list: &'a [(String, Value)], name: &str) -> Option<&'a Value> {
    list.iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
}

/// Reads the `node` and `edge` entries of one section into L, R or both.
fn read_section(
    entries: &[(String, Value)],
    targets: &mut [&mut Graph],
    ids: &mut BTreeMap<i64, usize>,
) -> Result<(), GmlRuleError> {
    let index_of = |id: i64, ids: &mut BTreeMap<i64, usize>| {
        let next = ids.len() + 1;
        *ids.entry(id).or_insert(next)
    };
    for (key, value) in entries {
        let Value::List(list) = value else {
            return Err(GmlRuleError::Syntax(format!("`{key}` is not a list")));
        };
        match key.as_str() {
            "node" => {
                let (Some(Value::Integer(id)), Some(Value::Text(label))) =
                    (field(list, "id"), field(list, "label"))
                else {
                    return Err(GmlRuleError::Syntax("node without id and label".into()));
                };
                let index = index_of(*id, ids);
                let label = parse_label(label)?;
                for target in targets.iter_mut() {
                    target.atoms.insert(index, label);
                }
            }
            "edge" => {
                let (
                    Some(Value::Integer(source)),
                    Some(Value::Integer(target)),
                    Some(Value::Text(label)),
                ) = (
                    field(list, "source"),
                    field(list, "target"),
                    field(list, "label"),
                )
                else {
                    return Err(GmlRuleError::Syntax(
                        "edge without source, target and label".into(),
                    ));
                };
                let (source, target) = (index_of(*source, ids), index_of(*target, ids));
                let symbol = edge_symbol(label)?;
                for graph in targets.iter_mut() {
                    graph
                        .bonds
                        .insert((source.min(target), source.max(target)), symbol);
                }
            }
            other => {
                return Err(GmlRuleError::Unsupported(format!(
                    "`{other}` in a rule section"
                )));
            }
        }
    }
    Ok(())
}

impl ExactRule {
    /// Reads a rule written in MØD's GML — see the [module docs](self).
    ///
    /// Node ids are renumbered from 1 in the order they first appear, since
    /// a SMARTS map index of 0 means "unmapped".
    pub fn from_gml(gml: &str) -> Result<Self, GmlRuleError> {
        let mut tokens = tokenize(gml)?.into_iter().peekable();
        let entries = parse_list(&mut tokens)?;
        let rule = match entries.as_slice() {
            [(key, Value::List(rule))] if key == "rule" => rule,
            _ => {
                return Err(GmlRuleError::Syntax(
                    "expected a single `rule [ ... ]`".into(),
                ));
            }
        };

        let (mut left, mut right) = (Graph::default(), Graph::default());
        let mut ids = BTreeMap::new();
        for (key, value) in rule {
            match (key.as_str(), value) {
                ("ruleID", Value::Text(_)) => {}
                ("left", Value::List(section)) => {
                    read_section(section, &mut [&mut left], &mut ids)?
                }
                ("right", Value::List(section)) => {
                    read_section(section, &mut [&mut right], &mut ids)?
                }
                ("context", Value::List(section)) => {
                    read_section(section, &mut [&mut left, &mut right], &mut ids)?
                }
                (other, _) => {
                    return Err(GmlRuleError::Unsupported(format!("`{other}` in a rule")));
                }
            }
        }
        // An edge in `left` or `right` between atoms only `context` labels is
        // an edge of that side alone; the atoms themselves are on both. An
        // edge whose atom its side does not have is no edge at all.
        for graph in [&left, &right] {
            if let Some(&(first, second)) = graph
                .bonds
                .keys()
                .find(|(a, b)| !graph.atoms.contains_key(a) || !graph.atoms.contains_key(b))
            {
                return Err(GmlRuleError::Syntax(format!(
                    "the edge between nodes {first} and {second} has an end its side lacks"
                )));
            }
        }
        Ok(ExactRule { left, right })
    }

    /// Writes the rule in MØD's GML, with what both sides share as the
    /// context.
    pub fn to_gml(&self, rule_id: &str) -> Result<String, GmlRuleError> {
        let mut sections: [String; 3] = Default::default();
        let [left_text, context_text, right_text] = &mut sections;
        let node = |index: usize, label: Label| -> Result<String, GmlRuleError> {
            let text = label_to_gml(label).ok_or_else(|| GmlRuleError::NotExpressible {
                index,
                what: format!("element {}", label.atomic_number),
            })?;
            Ok(format!("      node [ id {index} label \"{text}\" ]\n"))
        };
        let mut indices: Vec<usize> = self
            .left
            .atoms
            .keys()
            .chain(self.right.atoms.keys())
            .copied()
            .collect();
        indices.sort_unstable();
        indices.dedup();
        for &index in &indices {
            match (self.left.atoms.get(&index), self.right.atoms.get(&index)) {
                (Some(&before), Some(&after)) if before == after => {
                    context_text.push_str(&node(index, before)?)
                }
                (before, after) => {
                    if let Some(&before) = before {
                        left_text.push_str(&node(index, before)?);
                    }
                    if let Some(&after) = after {
                        right_text.push_str(&node(index, after)?);
                    }
                }
            }
        }
        let edge = |(first, second): (usize, usize), symbol: &str| {
            format!("      edge [ source {first} target {second} label \"{symbol}\" ]\n")
        };
        for (&pair, &symbol) in &self.left.bonds {
            if self.right.bonds.get(&pair) == Some(&symbol) {
                context_text.push_str(&edge(pair, symbol));
            } else {
                left_text.push_str(&edge(pair, symbol));
            }
        }
        for (&pair, &symbol) in &self.right.bonds {
            if self.left.bonds.get(&pair) != Some(&symbol) {
                right_text.push_str(&edge(pair, symbol));
            }
        }

        Ok(format!(
            "rule [\n   ruleID \"{}\"\n   left [\n{left_text}   ]\n   context [\n{context_text}   ]\n   right [\n{right_text}   ]\n]\n",
            rule_id.replace('"', "'")
        ))
    }
}

impl ReactionSmarts {
    /// Reads a rule written in MØD's GML — see the [module docs](self).
    pub fn from_gml(gml: &str) -> Result<Self, GmlRuleError> {
        Ok(ExactRule::from_gml(gml)?.to_reaction()?)
    }

    /// Writes a double-pushout rule in MØD's GML, with what both sides share
    /// as the context.
    ///
    /// GML labels are exact, so every reactant atom has to name its element,
    /// and every reactant bond its order — see
    /// [`ExactRule`](crate::molecule::rule_algebra::ExactRule). A charge the
    /// reactant side does not state is written as neutral, which is the one
    /// place the GML rule asks for more than the SMARTS did.
    pub fn to_gml(&self, rule_id: &str) -> Result<String, GmlRuleError> {
        ExactRule::from_reaction(self)?.to_gml(rule_id)
    }
}
