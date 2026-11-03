use crate::molecule::bond::{Bond, BondOrder};
use chemistry_consts::ElementProperties;
use std::str::FromStr;
/// Represents an atom in the V2000 format of an SDF file.
///
/// This struct contains all the information that can be specified for an atom
/// in the V2000 format, including its 3D coordinates, element symbol, and various
/// properties that may be used in different chemical contexts.
#[derive(Debug, Default)]
pub struct AtomV2000 {
    /// X coordinate of the atom
    pub x: f64,
    /// Y coordinate of the atom
    pub y: f64,
    /// Z coordinate of the atom
    pub z: f64,
    /// Element symbol of the atom
    pub symbol: String,
    /// Mass difference from the most abundant isotope, in the range -3..=4.
    ///
    /// Zero means "natural abundance", not "mass number 0"; see
    /// [`AtomV2000::isotope`] for the absolute mass number this implies.
    pub isotope_mass_difference: Option<i8>,
    /// Formal charge of the atom, already decoded from the atom block's
    /// charge *code*.
    ///
    /// The column does not hold the charge itself: it holds 1..=7 standing for
    /// +3, +2, +1, doublet radical, -1, -2, -3 respectively, with 0 meaning
    /// uncharged. Storing the raw code here made every charged atom read out
    /// of an SDF wrong (a +1 nitrogen came back as +3). Code 4 carries no
    /// charge at all and is reported through [`AtomV2000::doublet_radical`].
    pub formal_charge: Option<i8>,
    /// Set when the charge column held code 4, the atom block's way of writing
    /// a doublet radical (one unpaired electron).
    pub doublet_radical: bool,
    /// Stereo parity of the atom (0 = not stereo, 1 = odd, 2 = even, 3 = either)
    pub stereo_parity: Option<u8>,
    /// Number of hydrogen atoms attached
    pub hydrogen_count: Option<u8>,
    /// Stereo care box (0 = ignore, 1 = use in stereo perception)
    pub stereo_care_box: Option<u8>,
    /// Valence of the atom
    pub valence: Option<u8>,
    /// H0 designator for implicit hydrogen count (0 = no implicit hydrogens)
    pub implicit_hydrogen_count: Option<u8>,
    /// Atom-atom mapping number for reactions (1-based)
    pub atom_atom_mapping_number: Option<usize>,
    /// Inversion/retention flag for reactions (0 = not stereo, 1 = inversion, 2 = retention)
    pub inversion_retention_flag: Option<u8>,
    /// Exact change flag for reactions (0 = property not applied, 1 = property applied)
    pub exact_change_flag: Option<u8>,
}

impl From<std::io::Error> for MOLParseError {
    fn from(error: std::io::Error) -> Self {
        MOLParseError::IoError(error)
    }
}

#[derive(Debug)]
pub enum MOLParseError {
    IoError(std::io::Error),
    UnexpectedEndOfFile,
    TruncatedRecord(&'static str),
    InvalidCounts(String),
    UnsupportedVersion(String),
    ParseFloatError(std::num::ParseFloatError),
    ParseIntError(std::num::ParseIntError),
    InvalidAtom(String),
    InvalidBond,
}

impl std::str::FromStr for AtomV2000 {
    type Err = MOLParseError;

    /// Creates an `AtomV2000` instance from a string slice.
    ///
    /// # Arguments
    ///
    /// * `input` - A string slice containing atom data.
    ///
    /// # Errors
    ///
    /// Returns an error if the input string is too short or if parsing fails.
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        if input.len() < 69 || !input.is_ascii() {
            return Err(MOLParseError::InvalidAtom("Atom string too short".into()));
        }

        let charge_code: Option<u8> = input[36..39].trim().parse().ok();

        Ok(AtomV2000 {
            x: input[0..10]
                .trim()
                .parse()
                .map_err(MOLParseError::ParseFloatError)?,
            y: input[10..20]
                .trim()
                .parse()
                .map_err(MOLParseError::ParseFloatError)?,
            z: input[20..30]
                .trim()
                .parse()
                .map_err(MOLParseError::ParseFloatError)?,
            symbol: input[31..34].trim().to_string(),
            isotope_mass_difference: input[34..36].trim().parse().ok(),
            formal_charge: charge_code.map(decode_charge_code),
            doublet_radical: charge_code == Some(DOUBLET_RADICAL_CHARGE_CODE),
            stereo_parity: input[39..42].trim().parse().ok(),
            hydrogen_count: input[42..45].trim().parse().ok(),
            stereo_care_box: input[45..48].trim().parse().ok(),
            valence: input[48..51].trim().parse().ok(),
            implicit_hydrogen_count: input[51..54].trim().parse().ok(),
            atom_atom_mapping_number: input[60..63].trim().parse().ok(),
            inversion_retention_flag: input[63..66].trim().parse().ok(),
            exact_change_flag: input[66..69].trim().parse().ok(),
        })
    }
}

/// The atom block charge code that means "doublet radical" rather than a charge.
const DOUBLET_RADICAL_CHARGE_CODE: u8 = 4;

/// Mass number the atom block's isotope column counts from.
///
/// `monoisotopic_mass` rather than `most_abundant_isotope` because the latter
/// is `None` for the elements with no stable isotope, and MDL files may still
/// write a mass difference for those; the former answers for all of them, with
/// the conventional nuclide standing in. For everything else the two agree by
/// construction.
fn reference_mass_number(atomic_number: u8) -> Option<u16> {
    let mass = atomic_number.monoisotopic_mass().filter(|&m| m > 0.0)?;
    u16::try_from(mass.round() as i64).ok()
}

/// Turns an atom block charge code into the charge it stands for.
///
/// The mapping is `1..=3` -> `+3..=+1` and `5..=7` -> `-1..=-3`, i.e. `4 - code`
/// on both sides, with 0 (uncharged) and 4 (doublet radical) both carrying no
/// charge. Codes outside 0..=7 are not defined by the format and are read as
/// uncharged.
fn decode_charge_code(code: u8) -> i8 {
    match code {
        1..=3 | 5..=7 => 4 - code as i8,
        _ => 0,
    }
}

impl AtomV2000 {
    /// The absolute mass number the atom block's mass difference stands for,
    /// or `None` when the atom is written at natural abundance.
    ///
    /// The column is a difference from the most abundant isotope, so it needs
    /// the element to be resolved at all; an `M  ISO` line, which gives the
    /// mass number outright, supersedes it.
    pub fn isotope(&self) -> Option<u16> {
        let difference = self.isotope_mass_difference.filter(|&d| d != 0)?;
        let atomic_number = self.symbol.as_str().atomic_number()?;
        let base = reference_mass_number(atomic_number)?;
        u16::try_from(base as i32 + difference as i32).ok()
    }

    /// Number of unpaired electrons implied by the atom block charge column.
    pub fn radical_electrons(&self) -> u8 {
        u8::from(self.doublet_radical)
    }

    /// The total valence the file states for this atom, if it states one.
    ///
    /// Column value 0 means "work it out from the element", which is `None`
    /// here; 15 is the format's spelling of zero valence, which is a statement
    /// and not an absence, so it comes back as `Some(0)`. Everything else is
    /// the valence itself.
    pub fn explicit_valence(&self) -> Option<u8> {
        match self.valence {
            None | Some(0) => None,
            Some(15) => Some(0),
            Some(valence) => Some(valence),
        }
    }
}

/// A property line of the `M  xxx` block that we understand.
///
/// Everything else stays a raw string in [`MOLFile::properties`]; this only
/// covers the three lines that carry information the atom block cannot hold on
/// its own, or holds in a form the modern lines override.
#[derive(Debug, PartialEq)]
enum PropertyLine {
    /// `M  CHG` — formal charges, as (atom index, charge).
    Charges(Vec<(usize, i8)>),
    /// `M  RAD` — radical states, as (atom index, unpaired electrons).
    Radicals(Vec<(usize, u8)>),
    /// `M  ISO` — absolute mass numbers, as (atom index, mass number).
    Isotopes(Vec<(usize, u16)>),
}

/// Turns an `M  RAD` value into a count of unpaired electrons.
///
/// The format distinguishes singlet (1), doublet (2) and triplet (3); we store
/// a count, so singlet and triplet both come back as 2 and that distinction is
/// lost. The count is what the valence model needs — a singlet carbene's lone
/// pair occupies two bonding slots exactly as a triplet's two unpaired
/// electrons do — and it is the same mapping RDKit applies.
fn radical_electrons_from_code(code: u8) -> u8 {
    match code {
        1 | 3 => 2,
        2 => 1,
        _ => 0,
    }
}

/// Parses one `M  CHG` / `M  RAD` / `M  ISO` line, or `None` for anything else.
///
/// The format is a count followed by that many (atom, value) pairs, each field
/// in a fixed four-column slot. We read them by whitespace instead: the fields
/// are always separated in practice, and it costs nothing to also accept the
/// files that pad differently. The declared count is used only to stop early,
/// so a truncated line yields the pairs it does carry rather than an error.
/// Atom indices are 1-based in the file and 0-based here.
fn parse_property_line(line: &str) -> Option<PropertyLine> {
    let rest = line.strip_prefix("M  ")?;
    let mut tokens = rest.split_whitespace();
    let kind = tokens.next()?;
    let count: usize = tokens.next()?.parse().ok()?;

    let mut pairs = Vec::with_capacity(count);
    for _ in 0..count {
        let (Some(atom), Some(value)) = (tokens.next(), tokens.next()) else {
            break;
        };
        let Ok(atom) = atom.parse::<usize>() else {
            break;
        };
        let Some(atom) = atom.checked_sub(1) else {
            break;
        };
        pairs.push((atom, value));
    }

    macro_rules! decoded {
        ($f:expr) => {
            pairs
                .iter()
                .filter_map(|&(atom, value)| Some((atom, $f(value)?)))
                .collect()
        };
    }

    match kind {
        "CHG" => Some(PropertyLine::Charges(
            decoded!(|v: &str| v.parse::<i8>().ok()),
        )),
        "RAD" => Some(PropertyLine::Radicals(decoded!(|v: &str| v
            .parse::<u8>()
            .ok()
            .map(radical_electrons_from_code)))),
        "ISO" => Some(PropertyLine::Isotopes(
            decoded!(|v: &str| v.parse::<u16>().ok()),
        )),
        _ => None,
    }
}

impl Bond {
    /// Parses a bond from a MOL file line.
    ///
    /// # Arguments
    ///
    /// * `line` - A string slice that holds the line from the MOL file.
    ///
    /// # Returns
    ///
    /// * `Result<Bond, Box<dyn std::error::Error>>` - A result containing the parsed Bond or an error.
    ///
    /// # Errors
    ///
    /// * Returns an error if the line is too short or if parsing fails.
    pub fn from_mol_line(line: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Fixed columns, not whitespace. The bond block writes each field three
        // characters wide with no separator, so as soon as a molecule has a
        // hundred atoms the two indices run together: a bond between atoms 100
        // and 101 is written `100101  1  0`, which splits into one token.
        // Reading it that way gave atom1 = 100101 and shifted the bond order
        // into the second index, and since neither is in range the whole
        // connection table came apart -- a 116-atom peptide read back as a
        // heap of loose methanes. Molecules that large are ordinary: peptides,
        // natural products and much of any vendor catalogue.
        if line.len() < 9 || !line.is_ascii() {
            return Err("Bond line too short".into());
        }
        let _field = |range: std::ops::Range<usize>| -> Result<usize, Box<dyn std::error::Error>> {
            Ok(line[range].trim().parse::<usize>()?)
        };

        Ok(Self::from_mol_line_with_stereo(line)?.0)
    }

    /// Parses a bond line, also returning its stereo column.
    ///
    /// The column is what a MOL file uses to say which way a bond leaves the
    /// plane of the page — 1 for a wedge and 6 for a hash, both pointing away
    /// from the atom written first — and for a 2D depiction it is the only
    /// record of tetrahedral configuration there is. [`Bond`] has nowhere to
    /// keep it, so it comes back alongside.
    pub fn from_mol_line_with_stereo(line: &str) -> Result<(Self, u8), Box<dyn std::error::Error>> {
        if line.len() < 9 || !line.is_ascii() {
            return Err("Bond line too short".into());
        }
        let field = |range: std::ops::Range<usize>| -> Result<usize, Box<dyn std::error::Error>> {
            Ok(line[range].trim().parse::<usize>()?)
        };

        let atom1 = field(0..3)?;
        let atom2 = field(3..6)?;
        let bond = Bond {
            atom1: atom1.checked_sub(1).ok_or("atom1 index underflow")?,
            atom2: atom2.checked_sub(1).ok_or("atom2 index underflow")?,
            bond_order: BondOrder::from_sdf_str(line[6..9].trim())?,
        };

        // Absent or unreadable means "no stereo", the same as an explicit 0:
        // plenty of writers stop the line after the bond order.
        let stereo = line
            .get(9..12)
            .and_then(|field| field.trim().parse::<u8>().ok())
            .unwrap_or(0);
        Ok((bond, stereo))
    }
}

impl std::str::FromStr for CountsLine {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() < 39 {
            return Err("Counts line to short".into());
        }

        let parse_u16 =
            |s: &str| -> Result<u16, std::num::ParseIntError> { s.trim().parse::<u16>() };

        let parse_u8 = |s: &str| -> Result<u8, std::num::ParseIntError> { s.trim().parse::<u8>() };

        Ok(CountsLine {
            num_atoms: parse_u16(&s[0..3])?,
            num_bonds: parse_u16(&s[3..6])?,
            num_atom_lists: parse_u16(&s[6..9])?,
            chiral_flag: parse_u8(&s[12..15])?,
            num_stext_entries: parse_u16(&s[15..18])?,
            num_properties: parse_u16(&s[30..33])?,
            version: s[33..39].trim().to_string(),
        })
    }
}

pub enum MolVersion {
    V2000,
    V3000,
}

impl std::str::FromStr for MolVersion {
    type Err = Box<dyn std::error::Error>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "V2000" => Ok(MolVersion::V2000),
            "V3000" => Ok(MolVersion::V3000),
            _ => Err("Invalid mol version {}, only V2000 and V3000 are supported".into()),
        }
    }
}
/// Represents the counts line in an SDF (Structure-Data File) or MOL file.
///
/// This struct contains information about the number of atoms, bonds, and other
/// structural features of a molecule as specified in the counts line of an SDF file.
///
/// # Fields
///
/// * `num_atoms` - The number of atoms in the molecule.
/// * `num_bonds` - The number of bonds in the molecule.
/// * `num_atom_lists` - The number of atom lists (typically used for R-groups or variable attachments).
/// * `chiral_flag` - Indicates whether the molecule is chiral (1) or not (0).
/// * `num_stext_entries` - The number of Stext entries (additional text entries in the file).
/// * `num_properties` - The number of additional properties specified for the molecule.
/// * `version` - The version of the MOL file format (e.g., "V2000" or "V3000").
#[derive(Debug, Default)]
pub struct CountsLine {
    pub num_atoms: u16,
    pub num_bonds: u16,
    pub num_atom_lists: u16,
    pub chiral_flag: u8,
    pub num_stext_entries: u16,
    pub num_properties: u16,
    pub version: String,
}

/// Represents the structure of a MOL file, which is a common format for describing molecular structures.
///
/// # Fields
///
/// * `header` - A vector of strings containing the header information of the MOL file.
///              This typically includes the molecule name, user/program/date information, and comments.
/// * `counts_line` - A string representing the counts line of the MOL file.
///                   This line contains information about the number of atoms, bonds, and other structural features.
/// * `atoms` - A vector of `AtomV2000` structs, each representing an atom in the molecule.
///             This includes information such as coordinates, element type, and properties for each atom.
/// * `bonds` - A vector of `Bond` structs, each representing a bond between atoms in the molecule.
///             This includes information about which atoms are connected and the type of bond.
/// * `properties` - A vector of strings containing additional properties or attributes of the molecule.
///                  These can include various molecular properties or custom data fields.
///                  Only lines that are not interpreted land here; see the
///                  `charges`, `radicals` and `isotopes` fields.
#[derive(Debug, Default)]
pub struct MOLFile {
    pub header: Vec<String>,
    pub counts_line: String,
    pub atoms: Vec<AtomV2000>,
    pub bonds: Vec<Bond>,
    /// Property lines that are not interpreted, kept verbatim.
    pub properties: Vec<String>,
    /// Charges from `M  CHG`, as (atom index, charge). `None` when the file
    /// carried no such line.
    ///
    /// The distinction matters: the format says these lines supersede *all*
    /// charge information in the atom block, so a present-but-empty block
    /// zeroes the atom block's charges while an absent one leaves them alone.
    pub charges: Option<Vec<(usize, i8)>>,
    /// Radicals from `M  RAD`, as (atom index, unpaired electrons). Supersedes
    /// the atom block exactly as `charges` does.
    pub radicals: Option<Vec<(usize, u8)>>,
    /// Mass numbers from `M  ISO`, as (atom index, mass number). Supersedes the
    /// atom block's mass differences.
    pub isotopes: Option<Vec<(usize, u16)>>,
    /// The bond block's stereo column, one entry per bond in `bonds`.
    ///
    /// 1 is a wedge and 6 a hash, both leaving the page at the atom written
    /// first; 4 means "either" and, on a double bond, 3 means the geometry is
    /// undefined. Kept parallel to `bonds` rather than inside them because
    /// [`Bond`] is shared with the rest of the crate, where a MOL file's
    /// drawing convention has no meaning.
    pub bond_stereo: Vec<u8>,
}

impl MOLFile {
    /// Records a property line, interpreting the ones we understand.
    fn push_property(&mut self, line: String) {
        // A second line of the same kind extends the first rather than
        // replacing it: the format caps each line at eight pairs, so anything
        // with more than eight charged atoms arrives split across lines.
        match parse_property_line(&line) {
            Some(PropertyLine::Charges(pairs)) => {
                self.charges.get_or_insert_default().extend(pairs);
            }
            Some(PropertyLine::Radicals(pairs)) => {
                self.radicals.get_or_insert_default().extend(pairs);
            }
            Some(PropertyLine::Isotopes(pairs)) => {
                self.isotopes.get_or_insert_default().extend(pairs);
            }
            None => self.properties.push(line),
        }
    }
}

#[derive(Debug, Default)]
pub struct BondV2000 {
    pub atom1: usize,
    pub atom2: usize,
    pub bond_type: u8,
    pub bond_stereo: u8,
    pub bond_topology: Option<u8>,
    pub reacting_center_status: Option<u8>,
}

#[derive(Debug, Default)]
pub struct SGroup {
    pub sgroup_type: String,
    pub atoms: Vec<usize>,
    pub parent: Option<usize>,
    pub component_number: Option<u8>,
    pub connectivity: Option<String>,
    pub subscript: Option<String>,
    pub expansion: Option<bool>,
    pub bracket_style: Option<u8>,
    pub class: Option<String>,
}

#[derive(Debug, Default)]
pub struct MoleculeV2000 {
    pub atoms: Vec<AtomV2000>,
    pub bonds: Vec<BondV2000>,
    pub sgroups: Vec<SGroup>,
    pub properties: Vec<String>,
    pub chiral: bool,
    pub name: String,
    pub comment: String,
    pub dimension: u8,
}

pub fn parse_mol_file<I>(lines: &mut I) -> Result<MOLFile, MOLParseError>
where
    I: Iterator<Item = Result<String, std::io::Error>>,
{
    // Only EOF before the first header line is a clean end of the SDF stream.
    let Some(first) = lines.next() else {
        return Err(MOLParseError::UnexpectedEndOfFile);
    };
    let mut mol_file = MOLFile::default();
    mol_file.header.push(first?);
    let mut required = |stage| {
        lines.next().ok_or(MOLParseError::TruncatedRecord(stage))?
            .map_err(MOLParseError::IoError)
    };
    for _ in 0..2 {
        mol_file.header.push(required("header")?);
    }
    mol_file.counts_line = required("counts")?;
    let counts = &mol_file.counts_line;
    if !counts.is_ascii() || counts.len() < 39 {
        return Err(MOLParseError::InvalidCounts(counts.clone()));
    }
    let version = counts[33..39].trim();
    if version != "V2000" {
        return Err(MOLParseError::UnsupportedVersion(version.to_string()));
    }
    let atom_count: usize = counts[0..3].trim().parse()
        .map_err(|_| MOLParseError::InvalidCounts(counts.clone()))?;
    let bond_count: usize = counts[3..6].trim().parse()
        .map_err(|_| MOLParseError::InvalidCounts(counts.clone()))?;
    for _ in 0..atom_count {
        mol_file.atoms.push(AtomV2000::from_str(&required("atoms")?)?);
    }
    for _ in 0..bond_count {
        let (bond, stereo) = Bond::from_mol_line_with_stereo(&required("bonds")?)
            .map_err(|_| MOLParseError::InvalidBond)?;
        if bond.atom1 >= atom_count || bond.atom2 >= atom_count || bond.atom1 == bond.atom2 {
            return Err(MOLParseError::InvalidBond);
        }
        mol_file.bonds.push(bond);
        mol_file.bond_stereo.push(stereo);
    }
    loop {
        let line = required("properties")?;
        if line.trim() == "M  END" {
            return Ok(mol_file);
        }
        // Understated counts must not turn leftover structure into properties.
        if AtomV2000::from_str(&line).is_ok()
            || Bond::from_mol_line_with_stereo(&line).is_ok()
            || line.trim() == "$$$$"
        {
            return Err(MOLParseError::InvalidCounts(mol_file.counts_line));
        }
        mol_file.push_property(line);
    }
}
