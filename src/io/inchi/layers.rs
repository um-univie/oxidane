//! The layers of an InChI string, read into plain data.
//!
//! Nothing here interprets chemistry: atoms are numbered as the string numbers
//! them (from zero rather than one), and every layer is checked only for being
//! well formed and for naming atoms that exist. Turning the layers into a
//! structure is [`super::structure`]'s business.

use super::InChIError;
use chemistry_consts::ElementProperties;

/// An InChI read into its layers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InChI {
    /// Whether the string is a standard InChI, `InChI=1S/`.
    pub standard: bool,
    /// The components, in the order the formula lists them, each repeated as
    /// often as its multiplier says.
    pub components: Vec<InChIComponent>,
    /// The protons `/p` adds to (or, negative, takes from) the structure the
    /// main layer describes. They belong to the record, not to a component.
    pub protons: i32,
    /// `/s`: 1 for absolute stereo, 2 for relative, 3 for racemic.
    pub stereo_kind: Option<u8>,
    /// The isotopic layer's own `/h`: how many of the record's exchangeable
    /// hydrogens -- those a mobile group or a heteroatom holds -- are protium,
    /// deuterium and tritium. Like `/p` it belongs to the record: `[2H]O`
    /// beside methyl acetate writes its deuterium here, not on the water.
    pub exchangeable_isotopes: [u32; 3],
}

/// One component of an InChI.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InChIComponent {
    /// The atomic number of every numbered atom. Atom `i` here is atom `i + 1`
    /// of the string.
    pub elements: Vec<u8>,
    /// How many hydrogens the formula gives the component.
    pub formula_hydrogens: u32,
    /// The connections, each once, lower atom first.
    pub bonds: Vec<(usize, usize)>,
    /// The hydrogens each atom holds outside any mobile group.
    pub hydrogens: Vec<u32>,
    pub mobile_groups: Vec<MobileGroup>,
    /// `/q`.
    pub charge: i32,
    /// `/b`.
    pub double_bonds: Vec<StereoBond>,
    /// `/t`.
    pub centres: Vec<StereoCentre>,
    /// `/m`: whether the structure is the mirror image of the one `/t`
    /// describes.
    pub inverted: Option<bool>,
    /// `/i`.
    pub isotopes: Vec<IsotopicAtom>,
    /// The isotopic layer's own stereo, which replaces the main layer's where
    /// it is given.
    pub isotopic_double_bonds: Option<Vec<StereoBond>>,
    pub isotopic_centres: Option<Vec<StereoCentre>>,
    pub isotopic_inverted: Option<bool>,
}

/// A set of atoms sharing hydrogens, `(H2,1,2,3)`, and the negative charges
/// written with them, `(H-,1,2)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MobileGroup {
    pub hydrogens: u32,
    pub negative_charges: u32,
    pub atoms: Vec<usize>,
}

/// A stereo parity as InChI writes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity {
    /// `-`: for a centre, the neighbours' numbers increase anticlockwise seen
    /// from the hydrogen or lowest neighbour; for a double bond, the
    /// higher-numbered neighbours are on the same side.
    Minus,
    /// `+`: the opposite.
    Plus,
    /// `?` or `u`: stated to be unknown.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StereoBond {
    pub first: usize,
    pub second: usize,
    pub parity: Parity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StereoCentre {
    pub atom: usize,
    pub parity: Parity,
}

/// An atom of the isotopic layer: its mass relative to InChI's reference mass
/// for the element, and how many of its hydrogens are each isotope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IsotopicAtom {
    pub atom: usize,
    pub mass_shift: Option<i32>,
    /// Protium, deuterium and tritium.
    pub hydrogens: [u32; 3],
}

/// Which part of the string a layer belongs to. A stereo layer after `/i` is
/// the isotopic layer's, and everything after `/f` or `/r` describes another
/// structure the reader does not build.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    Main,
    Isotopic,
    Elsewhere,
}

/// Reads `inchi` into its layers.
pub fn parse_inchi(inchi: &str) -> Result<InChI, InChIError> {
    let inchi = inchi.trim();
    let rest = inchi
        .strip_prefix("InChI=")
        .ok_or(InChIError::MissingPrefix)?;
    let mut layers = rest.split('/');
    let version = layers.next().unwrap_or_default();
    let standard = match version {
        "1S" => true,
        "1" => false,
        "1B" => return Err(InChIError::Unsupported("polymer InChI (1B)".into())),
        other => return Err(InChIError::Unsupported(format!("version {other}"))),
    };
    let mut parsed = InChI {
        standard,
        ..InChI::default()
    };
    let mut layers = layers.peekable();
    // The formula is the one layer with no letter of its own; a string with
    // nothing but a proton, `InChI=1S/p+1`, has none.
    if let Some(first) = layers.peek()
        && !first.starts_with(|c: char| c.is_ascii_lowercase())
    {
        parsed.components = parse_formula(first)?;
        layers.next();
    }
    let count = parsed.components.len();
    let mut section = Section::Main;
    for layer in layers {
        let Some(letter) = layer.chars().next() else {
            continue;
        };
        let content = &layer[1..];
        let error = |message: &str| InChIError::Syntax {
            layer: letter,
            message: message.to_string(),
        };
        match (section, letter) {
            (Section::Elsewhere, _) => {}
            (_, 'f' | 'r') => section = Section::Elsewhere,
            (_, 'z') => return Err(InChIError::Unsupported("polymer layer".into())),
            (Section::Main, 'c') => {
                for (component, segment) in parsed
                    .components
                    .iter_mut()
                    .zip(segments(content, count, letter)?)
                {
                    component.bonds = parse_connections(segment, component.elements.len())
                        .map_err(|m| error(&m))?;
                }
            }
            (Section::Main, 'h') => {
                for (component, segment) in parsed
                    .components
                    .iter_mut()
                    .zip(segments(content, count, letter)?)
                {
                    let (hydrogens, groups) = parse_hydrogens(segment, component.elements.len())
                        .map_err(|m| error(&m))?;
                    component.hydrogens = hydrogens;
                    component.mobile_groups = groups;
                }
            }
            (Section::Main, 'q') => {
                for (component, segment) in parsed
                    .components
                    .iter_mut()
                    .zip(segments(content, count, letter)?)
                {
                    component.charge = if segment.is_empty() {
                        0
                    } else {
                        parse_signed(segment).ok_or_else(|| error("not a charge"))?
                    };
                }
            }
            (Section::Main, 'p') => {
                parsed.protons =
                    parse_signed(content).ok_or_else(|| error("not a proton count"))?;
            }
            (Section::Main, 'b') => {
                for (component, segment) in parsed
                    .components
                    .iter_mut()
                    .zip(segments(content, count, letter)?)
                {
                    component.double_bonds = parse_double_bonds(segment, component.elements.len())
                        .map_err(|m| error(&m))?;
                }
            }
            (Section::Main, 't') => {
                for (component, segment) in parsed
                    .components
                    .iter_mut()
                    .zip(segments(content, count, letter)?)
                {
                    component.centres =
                        parse_centres(segment, component.elements.len()).map_err(|m| error(&m))?;
                }
            }
            (Section::Isotopic, 'b') => {
                for (component, segment) in parsed
                    .components
                    .iter_mut()
                    .zip(isotopic_segments(content, count, letter)?)
                {
                    component.isotopic_double_bonds = Some(match segment {
                        Copied::Main => component.double_bonds.clone(),
                        Copied::Empty => Vec::new(),
                        Copied::Own(segment) => {
                            parse_double_bonds(segment, component.elements.len())
                                .map_err(|m| error(&m))?
                        }
                    });
                }
            }
            (Section::Isotopic, 't') => {
                for (component, segment) in parsed
                    .components
                    .iter_mut()
                    .zip(isotopic_segments(content, count, letter)?)
                {
                    component.isotopic_centres = Some(match segment {
                        Copied::Main => component.centres.clone(),
                        Copied::Empty => Vec::new(),
                        Copied::Own(segment) => parse_centres(segment, component.elements.len())
                            .map_err(|m| error(&m))?,
                    });
                }
            }
            (_, 'm') => {
                // One character per component: `0`, `1`, or `.` for a
                // component with nothing to invert.
                let mut flags = content.chars();
                for component in parsed.components.iter_mut() {
                    let flag = match flags.next() {
                        Some('0') => Some(false),
                        Some('1') => Some(true),
                        Some('.') | None => None,
                        Some(_) => return Err(error("not 0, 1 or .")),
                    };
                    if section == Section::Main {
                        component.inverted = flag;
                    } else {
                        component.isotopic_inverted = flag;
                    }
                }
            }
            (_, 's') => {
                let kind = content
                    .parse::<u8>()
                    .ok()
                    .filter(|kind| (1..=3).contains(kind))
                    .ok_or_else(|| error("not 1, 2 or 3"))?;
                parsed.stereo_kind = Some(kind);
            }
            (Section::Main, 'i') => {
                section = Section::Isotopic;
                for (component, segment) in parsed
                    .components
                    .iter_mut()
                    .zip(segments(content, count, letter)?)
                {
                    component.isotopes =
                        parse_isotopes(segment, component.elements.len()).map_err(|m| error(&m))?;
                }
            }
            (Section::Isotopic, 'h') => {
                parsed.exchangeable_isotopes =
                    parse_hydrogen_isotopes(content).map_err(|m| error(&m))?;
            }
            _ => return Err(error("unexpected layer")),
        }
    }
    for (index, component) in parsed.components.iter_mut().enumerate() {
        let atoms = component.elements.len();
        component.hydrogens.resize(atoms, 0);
        let numbered_hydrogen = u32::from(component.elements == [1]);
        let held: u32 = component.hydrogens.iter().sum::<u32>()
            + component
                .mobile_groups
                .iter()
                .map(|group| group.hydrogens)
                .sum::<u32>()
            + numbered_hydrogen;
        if held != component.formula_hydrogens {
            return Err(InChIError::Inconsistent(format!(
                "component {} has {} hydrogens in its formula and {held} in its layers",
                index + 1,
                component.formula_hydrogens
            )));
        }
    }
    Ok(parsed)
}

/// The formula layer: components separated by `.`, each with an optional
/// multiplier. The numbered atoms are every non-hydrogen, in the order the
/// formula lists them -- Hill order, so carbon first -- except in a component
/// of hydrogen alone, which numbers one hydrogen.
fn parse_formula(formula: &str) -> Result<Vec<InChIComponent>, InChIError> {
    let error = |message: String| InChIError::Syntax {
        layer: 'f',
        message,
    };
    let mut components = Vec::new();
    for part in formula.split('.') {
        let digits = part.bytes().take_while(u8::is_ascii_digit).count();
        let multiplier: usize = if digits == 0 {
            1
        } else {
            part[..digits]
                .parse()
                .map_err(|_| error(format!("bad multiplier in {part}")))?
        };
        let mut component = InChIComponent::default();
        let bytes = part.as_bytes();
        let mut at = digits;
        while at < bytes.len() {
            if !bytes[at].is_ascii_uppercase() {
                return Err(error(format!(
                    "unexpected {:?} in {part}",
                    bytes[at] as char
                )));
            }
            let mut end = at + 1;
            while end < bytes.len() && bytes[end].is_ascii_lowercase() {
                end += 1;
            }
            let symbol = &part[at..end];
            at = end;
            while end < bytes.len() && bytes[end].is_ascii_digit() {
                end += 1;
            }
            let count: u32 = if end == at {
                1
            } else {
                part[at..end]
                    .parse()
                    .map_err(|_| error(format!("bad count in {part}")))?
            };
            at = end;
            let atomic_number = symbol
                .atomic_number()
                .ok_or_else(|| InChIError::UnknownElement(symbol.to_string()))?;
            if atomic_number == 1 {
                component.formula_hydrogens += count;
            } else {
                component
                    .elements
                    .extend(std::iter::repeat_n(atomic_number, count as usize));
            }
        }
        if component.elements.is_empty() {
            if component.formula_hydrogens == 0 {
                return Err(error(format!("empty component in {formula}")));
            }
            component.elements.push(1);
        }
        for _ in 0..multiplier {
            components.push(component.clone());
        }
    }
    Ok(components)
}

/// A layer's per-component segments: separated by `;`, a segment written
/// `n*...` standing for `n` components, and components the layer says nothing
/// about left empty.
fn segments(content: &str, count: usize, layer: char) -> Result<Vec<&str>, InChIError> {
    let mut expanded = Vec::with_capacity(count.min(1024));
    for segment in content.split(';') {
        let digits = segment.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && segment[digits..].starts_with('*') {
            let repeat: usize = segment[..digits].parse().map_err(|_| InChIError::Syntax {
                layer,
                message: format!("bad multiplier in {segment}"),
            })?;
            expanded.extend(std::iter::repeat_n(&segment[digits + 1..], repeat));
        } else {
            expanded.push(segment);
        }
    }
    if expanded.len() > count {
        return Err(InChIError::Syntax {
            layer,
            message: format!("{} segments for {count} components", expanded.len()),
        });
    }
    if count != usize::MAX {
        expanded.resize(count, "");
    }
    Ok(expanded)
}

/// One component's segment of an isotopic stereo layer.
enum Copied<'a> {
    /// Written out.
    Own(&'a str),
    /// `m`: the same as the main layer's.
    Main,
    /// `e`: none.
    Empty,
}

/// [`segments`], for an isotopic stereo layer, which may also say a run of
/// components is the same as the main layer -- `3m` -- or has none -- `e`.
fn isotopic_segments(
    content: &str,
    count: usize,
    layer: char,
) -> Result<Vec<Copied<'_>>, InChIError> {
    let mut expanded = Vec::with_capacity(count.min(1024));
    for segment in segments(content, usize::MAX, layer)? {
        let digits = segment.bytes().take_while(u8::is_ascii_digit).count();
        let repeat = |digits: usize| -> Result<usize, InChIError> {
            if digits == 0 {
                return Ok(1);
            }
            segment[..digits].parse().map_err(|_| InChIError::Syntax {
                layer,
                message: format!("bad repeat in {segment}"),
            })
        };
        match &segment[digits..] {
            "m" => expanded.extend((0..repeat(digits)?).map(|_| Copied::Main)),
            "e" => expanded.extend((0..repeat(digits)?).map(|_| Copied::Empty)),
            _ => expanded.push(Copied::Own(segment)),
        }
    }
    if expanded.len() > count {
        return Err(InChIError::Syntax {
            layer,
            message: format!("{} segments for {count} components", expanded.len()),
        });
    }
    while expanded.len() < count {
        expanded.push(Copied::Empty);
    }
    Ok(expanded)
}

/// `+2`, `-1`, `+0`.
fn parse_signed(text: &str) -> Option<i32> {
    let (sign, digits) = match text.as_bytes().first()? {
        b'+' => (1, &text[1..]),
        b'-' => (-1, &text[1..]),
        _ => return None,
    };
    digits.parse::<i32>().ok().map(|value| sign * value)
}

/// A reader over the ASCII of one segment.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            at: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn eat(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn number(&mut self) -> Option<usize> {
        let start = self.at;
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.at += 1;
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .ok()?
            .parse()
            .ok()
    }

    fn done(&self) -> bool {
        self.at >= self.bytes.len()
    }
}

/// An atom number of the string, checked against the component and made
/// zero-based.
fn atom(cursor: &mut Cursor, atoms: usize) -> Result<usize, String> {
    match cursor.number() {
        Some(number) if (1..=atoms).contains(&number) => Ok(number - 1),
        Some(number) => Err(format!("atom {number} of {atoms}")),
        None => Err(format!("expected an atom number at {}", cursor.at)),
    }
}

/// The connection layer: a walk over the atoms, `-` between neighbours, a
/// branch in parentheses with `,` between the branches it holds. A ring closes
/// by naming an atom the walk has already met.
fn parse_connections(segment: &str, atoms: usize) -> Result<Vec<(usize, usize)>, String> {
    let mut bonds = Vec::new();
    let mut cursor = Cursor::new(segment);
    let mut previous: Option<usize> = None;
    let mut origins: Vec<Option<usize>> = Vec::new();
    while let Some(byte) = cursor.peek() {
        match byte {
            b'0'..=b'9' => {
                let current = atom(&mut cursor, atoms)?;
                if let Some(previous) = previous {
                    if previous == current {
                        return Err(format!("atom {} bonded to itself", current + 1));
                    }
                    bonds.push((previous.min(current), previous.max(current)));
                }
                previous = Some(current);
            }
            b'-' => cursor.at += 1,
            b'(' => {
                cursor.at += 1;
                origins.push(previous);
            }
            b',' => {
                cursor.at += 1;
                previous = *origins.last().ok_or("`,` outside a branch")?;
            }
            b')' => {
                cursor.at += 1;
                previous = origins.pop().ok_or("unbalanced `)`")?;
            }
            other => return Err(format!("unexpected {:?}", other as char)),
        }
    }
    if !origins.is_empty() {
        return Err("unbalanced `(`".into());
    }
    bonds.sort_unstable();
    bonds.dedup();
    Ok(bonds)
}

/// The hydrogen layer: `1-3,5H2` gives atoms 1, 2, 3 and 5 two hydrogens each,
/// and `(H2,4,6)` puts two hydrogens somewhere on atoms 4 and 6.
fn parse_hydrogens(segment: &str, atoms: usize) -> Result<(Vec<u32>, Vec<MobileGroup>), String> {
    let mut hydrogens = vec![0u32; atoms];
    let mut groups = Vec::new();
    let mut cursor = Cursor::new(segment);
    while !cursor.done() {
        if cursor.eat(b',') {
            continue;
        }
        if cursor.eat(b'(') {
            if !cursor.eat(b'H') {
                return Err("a mobile group starts with H".into());
            }
            let mut group = MobileGroup {
                hydrogens: cursor.number().unwrap_or(1) as u32,
                ..MobileGroup::default()
            };
            if cursor.eat(b'-') {
                group.negative_charges = cursor.number().unwrap_or(1) as u32;
            }
            while cursor.eat(b',') {
                let first = atom(&mut cursor, atoms)?;
                let last = if cursor.eat(b'-') {
                    atom(&mut cursor, atoms)?
                } else {
                    first
                };
                group.atoms.extend(first..=last);
            }
            if !cursor.eat(b')') {
                return Err("unclosed mobile group".into());
            }
            groups.push(group);
            continue;
        }
        let mut listed = Vec::new();
        loop {
            let first = atom(&mut cursor, atoms)?;
            let last = if cursor.eat(b'-') {
                atom(&mut cursor, atoms)?
            } else {
                first
            };
            if last < first {
                return Err(format!("range {}-{}", first + 1, last + 1));
            }
            listed.extend(first..=last);
            if !cursor.eat(b',') {
                break;
            }
        }
        // Lower case marks a hydrogen count relative to another layer, which
        // only the fixed-H layer has.
        if !cursor.eat(b'H') {
            return Err(format!("expected H at {}", cursor.at));
        }
        let count = cursor.number().unwrap_or(1) as u32;
        for atom in listed {
            hydrogens[atom] = count;
        }
    }
    Ok((hydrogens, groups))
}

fn parity(cursor: &mut Cursor) -> Result<Parity, String> {
    let parity = match cursor.peek() {
        Some(b'-') => Parity::Minus,
        Some(b'+') => Parity::Plus,
        Some(b'?' | b'u') => Parity::Unknown,
        _ => return Err(format!("expected a parity at {}", cursor.at)),
    };
    cursor.at += 1;
    Ok(parity)
}

/// `/b`: `4-3+,6-5-`.
fn parse_double_bonds(segment: &str, atoms: usize) -> Result<Vec<StereoBond>, String> {
    let mut bonds = Vec::new();
    let mut cursor = Cursor::new(segment);
    while !cursor.done() {
        let first = atom(&mut cursor, atoms)?;
        if !cursor.eat(b'-') {
            return Err("a double bond names two atoms".into());
        }
        let second = atom(&mut cursor, atoms)?;
        bonds.push(StereoBond {
            first,
            second,
            parity: parity(&mut cursor)?,
        });
        cursor.eat(b',');
    }
    Ok(bonds)
}

/// `/t`: `2-,3+,5?`.
fn parse_centres(segment: &str, atoms: usize) -> Result<Vec<StereoCentre>, String> {
    let mut centres = Vec::new();
    let mut cursor = Cursor::new(segment);
    while !cursor.done() {
        let atom = atom(&mut cursor, atoms)?;
        centres.push(StereoCentre {
            atom,
            parity: parity(&mut cursor)?,
        });
        cursor.eat(b',');
    }
    Ok(centres)
}

/// `D2T` and the like: how many of some hydrogens are each isotope.
fn hydrogen_isotopes(cursor: &mut Cursor) -> [u32; 3] {
    let mut counts = [0u32; 3];
    loop {
        let slot = match cursor.peek() {
            Some(b'H') => 0,
            Some(b'D') => 1,
            Some(b'T') => 2,
            _ => break,
        };
        cursor.at += 1;
        counts[slot] += cursor.number().unwrap_or(1) as u32;
    }
    counts
}

/// `/i`: `1+1,2D3,5-1T`.
fn parse_isotopes(segment: &str, atoms: usize) -> Result<Vec<IsotopicAtom>, String> {
    let mut isotopes = Vec::new();
    let mut cursor = Cursor::new(segment);
    while !cursor.done() {
        let atom = atom(&mut cursor, atoms)?;
        let mut isotope = IsotopicAtom {
            atom,
            ..IsotopicAtom::default()
        };
        let sign = match cursor.peek() {
            Some(b'+') => Some(1),
            Some(b'-') => Some(-1),
            _ => None,
        };
        if let Some(sign) = sign {
            cursor.at += 1;
            let shift = cursor.number().ok_or("a mass shift needs digits")?;
            isotope.mass_shift = Some(sign * shift as i32);
        }
        isotope.hydrogens = hydrogen_isotopes(&mut cursor);
        if isotope.mass_shift.is_none() && isotope.hydrogens == [0; 3] {
            return Err(format!("atom {} has no isotope", atom + 1));
        }
        isotopes.push(isotope);
        if !cursor.done() && !cursor.eat(b',') {
            return Err(format!("unexpected text at {}", cursor.at));
        }
    }
    Ok(isotopes)
}

/// The isotopic layer's `/h`: `D2` or `T`.
fn parse_hydrogen_isotopes(content: &str) -> Result<[u32; 3], String> {
    let mut cursor = Cursor::new(content);
    let counts = hydrogen_isotopes(&mut cursor);
    if !cursor.done() {
        return Err(format!("unexpected text at {}", cursor.at));
    }
    Ok(counts)
}
