use crate::io::{MOLFile, MOLParseError, parse_mol_file};
use crate::molecule::aromaticity::{AromaticityModel, MoleculeAromaticity};
use crate::molecule::base::MoleculeCore;
use crate::molecule::chirality::{ChiralClass, ChiralClassifier};
use crate::{
    molecule::bond::{BondOrder, BondTarget},
    molecule::molecule3d::Molecule3D,
    vector::Vector3D,
};
use chemistry_consts::ElementProperties;
use rustc_hash::{FxHashMap, FxHashSet};
use std::io::{BufRead, BufReader};
use std::path::Path;
use tinyvec::ArrayVec;

#[derive(Debug)]
pub struct SDFEntry {
    pub mol_file: MOLFile,
    pub data_fields: FxHashMap<String, String>,
}

/// What to perceive after the atoms and bonds of a MOL block have been read.
///
/// The block records exactly which atoms are bonded to which; everything here
/// is a *derived* property that the format leaves implicit, in the same way a
/// SMILES does — see [`crate::io::SmilesParseOptions`], whose defaults these
/// match so that the same molecule read from either format answers the same
/// questions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SdfParseOptions {
    /// Whether to work out the hydrogens the file leaves implicit and add them
    /// as atoms.
    ///
    /// On by default. A MOL block written from a 2D depiction — which is what
    /// PubChem, ChEMBL and most vendor catalogues ship — carries only heavy
    /// atoms, so without this alanine reads back as a bare C3NO2 skeleton with
    /// no hydrogen anywhere and a molecular formula to match. Turn it off to
    /// see exactly the atoms the file listed.
    pub hydrogens: bool,
    /// Whether to work out the chirality the block records geometrically — see
    /// [`assign_stereochemistry`].
    ///
    /// On by default. Without it every molecule read from an SDF is achiral,
    /// whatever the file drew: on a corpus of 1501 stereoisomers not one
    /// survived the trip, while connectivity and hydrogens came back perfect.
    pub stereochemistry: bool,
    /// Which aromaticity model to run, or `None` to leave bonds exactly as
    /// written.
    ///
    /// On by default, for the reason given in [`crate::io::SmilesParseOptions`]:
    /// a MOL block always writes a Kekulé ring, so without perception benzene
    /// read from an SDF has no aromatic bond and does not match a SMARTS that
    /// benzene read from a SMILES does.
    pub aromaticity: Option<AromaticityModel>,
}

impl Default for SdfParseOptions {
    fn default() -> Self {
        Self {
            hydrogens: true,
            stereochemistry: true,
            aromaticity: Some(AromaticityModel::OpenSMILES),
        }
    }
}

/// Works out the chirality a MOL block records geometrically.
///
/// Two channels, and a molecule can use both. Double bond geometry is in the
/// coordinates themselves, which is why a MOL file's 2D layout is not free to
/// be redrawn. Tetrahedral configuration is in the bond block's stereo column:
/// the drawing is flat, so one bond at the centre is marked as leaving the page
/// towards the reader (a wedge, 1) or away from it (a hash, 6), always at the
/// atom written first.
///
/// `identify_chiral_classes` already reads both from real coordinates and is
/// checked against RDKit's `AssignStereochemistryFrom3D`, so the work here is
/// to give it coordinates to read: for a flat depiction every atom sits at
/// z = 0 and the signed volume at a stereocentre is zero, which reads as "not a
/// stereocentre" rather than as "configuration unknown".
fn assign_stereochemistry(molecule: &mut Molecule3D, wedges: &[(usize, usize, u8)]) {
    // A double bond the file marks as "either" -- code 3 -- has no
    // configuration, and that is a statement, not an omission. It binds however
    // the atoms happen to be placed: `tests/pentene.sdf` carries 3D coordinates
    // whose torsion across the double bond is -179.7 degrees, which is as trans
    // as a geometry gets, and still marks the bond 3. Reading the coordinates
    // there reports a configuration the file went out of its way to withhold.
    let undefined: FxHashSet<(usize, usize)> = wedges
        .iter()
        .filter(|&&(_, _, stereo)| stereo == 3 || stereo == 4)
        .map(|&(a, b, _)| (a.min(b), a.max(b)))
        .collect();

    let is_flat = molecule
        .positions
        .iter()
        .all(|position| position.z.abs() < 1e-6);

    if is_flat {
        // Only the two things a flat drawing can actually say. Handing a 2D
        // depiction to the full perception makes it answer questions it has no
        // information about: every four-coordinate atom is coplanar with its
        // neighbours by construction, so each one comes back square planar, and
        // a quaternary carbon in a drawn structure was being written out as
        // `[C@SP3]`.
        molecule.initialize_chiral_classes();
        assign_planar_cis_trans(molecule, &undefined);
    } else {
        // Real coordinates, so the general perception applies and the wedge
        // pass below finds nothing left to do.
        molecule.identify_chiral_classes();
        clear_undefined_cis_trans(molecule, &undefined);
    }

    for &(centre, neighbour, stereo) in wedges {
        let sign = match stereo {
            1 => 1.0,  // wedge, towards the reader
            6 => -1.0, // hash, away from the reader
            // 4 on a single bond and 3 on a double one both mean "either",
            // which is a statement that the configuration is deliberately
            // unspecified rather than one we should try to recover.
            _ => continue,
        };
        if let Some(class) = wedge_chirality(molecule, centre, neighbour, sign) {
            if let Some(classes) = molecule.chiral_classes_mut() {
                classes[centre] = class;
            }
        }
    }
}

/// Double bond geometry from a flat drawing.
///
/// This is the one kind of stereochemistry a 2D MOL block states in its
/// coordinates rather than in a marker: the substituents are drawn on the sides
/// of the double bond that they are on, which is why such a file cannot be
/// relaid out freely.
fn assign_planar_cis_trans(molecule: &mut Molecule3D, undefined: &FxHashSet<(usize, usize)>) {
    let mut classes = molecule.chiral_classes().map(<[_]>::to_vec).unwrap_or_default();
    if classes.len() != molecule.len() {
        return;
    }

    for index in 0..molecule.len() {
        let Some(neighbours) = molecule.get_atom_bonds(index) else {
            continue;
        };
        if !molecule.is_potential_cis_trans(neighbours) {
            continue;
        }
        let partner = neighbours
            .iter()
            .find(|bond| bond.bond_order() == BondOrder::Double)
            .map(|bond| bond.target());
        if partner.is_some_and(|partner| {
            undefined.contains(&(index.min(partner), index.max(partner)))
        }) {
            continue;
        }
        let class = molecule.determine_cis_trans_chirality(index, neighbours, &mut classes);
        if class != ChiralClass::None {
            classes[index] = class;
        }
    }

    if let Some(existing) = molecule.chiral_classes_mut() {
        *existing = classes;
    }
}

/// Drops the geometry perception found on double bonds the file marked as
/// "either".
///
/// The flat path never assigns those in the first place; this is for the 3D
/// path, where the general perception has already read the coordinates.
fn clear_undefined_cis_trans(molecule: &mut Molecule3D, undefined: &FxHashSet<(usize, usize)>) {
    if undefined.is_empty() {
        return;
    }
    let Some(classes) = molecule.chiral_classes_mut() else {
        return;
    };
    for &(first, second) in undefined {
        for atom in [first, second] {
            if matches!(
                classes.get(atom),
                Some(ChiralClass::CisTransClockwise | ChiralClass::CisTransCounterclockwise)
            ) {
                classes[atom] = ChiralClass::None;
            }
        }
    }
}

/// The tetrahedral class at `centre`, reading `neighbour` as lifted off the
/// page by `sign`.
fn wedge_chirality(
    molecule: &Molecule3D,
    centre: usize,
    neighbour: usize,
    sign: f64,
) -> Option<ChiralClass> {
    let mut neighbours: Vec<usize> = molecule
        .get_atom_bonds(centre)?
        .iter()
        .map(|bond| bond.target())
        .collect();
    if neighbours.len() != 4 {
        return None;
    }
    // The tag is stored against the centre's index-sorted neighbours, which is
    // the convention the rest of the crate reads it back with.
    neighbours.sort_unstable();

    let origin = molecule.get_atom_position(centre)?;
    let mut vectors: Vec<Vector3D> = neighbours
        .iter()
        .map(|&target| {
            molecule
                .get_atom_position(target)
                .map(|position| position - origin)
                .unwrap_or_default()
        })
        .collect();

    // How far out of the page to put the wedged atom. Anything positive gives
    // the same sign, but matching the drawing's own scale keeps the tetrahedron
    // from being a spike, which matters once the volume is compared to zero.
    let scale = {
        let lengths: Vec<f64> = vectors.iter().map(|v| v.length()).filter(|&l| l > 1e-9).collect();
        if lengths.is_empty() {
            return None;
        }
        lengths.iter().sum::<f64>() / lengths.len() as f64
    };

    for (slot, &target) in vectors.iter_mut().zip(neighbours.iter()) {
        if target == neighbour {
            slot.z = sign * scale;
        }
    }

    // A hydrogen worked out from the valence model sits on its neighbour and so
    // has no direction of its own. It goes where the other three are not, which
    // is what puts it on the far side of the centre from the wedge.
    let sum: Vector3D = vectors
        .iter()
        .filter(|v| v.length() > 1e-9)
        .fold(Vector3D::default(), |total, &v| total + v);
    for slot in vectors.iter_mut() {
        if slot.length() <= 1e-9 {
            *slot = -sum;
        }
    }

    // The volume of the tetrahedron the four neighbours actually span, rather
    // than `determine_tetrahedral_chirality`'s triple product of the first
    // three. That shortcut is sound for a real 3D structure, where the fourth
    // neighbour is opposite the other three, but not here: only the wedged
    // neighbour has been given a z at all, so whenever it sorted into the
    // fourth slot the first three were coplanar and the volume came out zero.
    // Every stereocentre whose wedge happened to point at its highest-indexed
    // neighbour was silently dropped — 84 of 1501 in the corpus, all of them
    // ring fusions where the wedge lands on a ring bond.
    let signed_volume = (vectors[1] - vectors[0])
        .dot(&(vectors[2] - vectors[0]).cross(&(vectors[3] - vectors[0])));
    if signed_volume.abs() < 1e-9 {
        // Still flat: the file marked a bond but the geometry does not resolve,
        // so there is nothing to say.
        return None;
    }
    // Inverted against the triple product, which is what the rest of the crate
    // reads: for four vectors from a common centre the two tests have opposite
    // signs. Checked on a regular tetrahedron, and then on the corpus.
    Some(if signed_volume < 0.0 {
        ChiralClass::TH(1)
    } else {
        ChiralClass::TH(2)
    })
}

#[derive(Debug)]
pub enum SDFParseError {
    MOLParseError(MOLParseError),
    IoError(std::io::Error),
}

pub fn parse_sdf_file<R: BufRead>(reader: R) -> Result<Vec<SDFEntry>, SDFParseError> {
    let mut entries = Vec::new();
    let mut lines = reader.lines();
    let mut current_data_fields = FxHashMap::default();
    let mut in_mol_block = true;
    let mut current_data_field_name = String::new();

    loop {
        if in_mol_block {
            match parse_mol_file(&mut lines) {
                Ok(mol_file) => {
                    in_mol_block = false;
                    entries.push(SDFEntry {
                        mol_file,
                        data_fields: FxHashMap::default(),
                    });
                }
                Err(MOLParseError::UnexpectedEndOfFile) => break,
                Err(e) => return Err(SDFParseError::MOLParseError(e)),
            }
        } else {
            match lines.next() {
                Some(Ok(line)) => {
                    if line.trim() == "$$$$" {
                        if let Some(entry) = entries.last_mut() {
                            entry.data_fields = std::mem::take(&mut current_data_fields);
                        }
                        in_mol_block = true;
                    } else if let Some(stripped) = line.strip_prefix("> ") {
                        // Remove < and > from the data field name
                        current_data_field_name = stripped
                            .replace("<", "")
                            .replace(">", "")
                            .trim()
                            .to_string();
                    } else if !current_data_field_name.is_empty() {
                        current_data_fields
                            .entry(current_data_field_name.clone())
                            .or_insert(String::new())
                            .push_str(line.trim());
                    }
                }
                Some(Err(e)) => {
                    return Err(SDFParseError::MOLParseError(MOLParseError::IoError(e)));
                }
                None => break,
            }
        }
    }

    // Handle last entry if file doesn't end with $$$$
    if let Some(entry) = entries.last_mut() {
        if !current_data_fields.is_empty() {
            entry.data_fields = current_data_fields;
        }
    }

    Ok(entries)
}

impl Molecule3D {
    /// This function parses an SDF file and returns a vector of Molecule3D instances.
    ///
    ///
    /// # Arguments
    ///
    /// * `sdf` - The path to the SDF file.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of `Molecule3D` instances or an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the SDF file cannot be read or parsed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::Path;
    /// use oxidane::prelude::Molecule3D;
    /// use oxidane::io::ToSMILES;
    ///
    /// let sdf_path = Path::new("tests/pentene.sdf");
    /// let molecules = Molecule3D::from_sdf(sdf_path).unwrap();
    /// assert_eq!(molecules.len(), 1);
    /// let molecule = molecules.into_iter().next().unwrap();
    /// assert_eq!(molecule.to_smiles(), "C(C=CC)C");
    /// ```
    pub fn from_sdf<T: AsRef<Path>>(sdf: T) -> Result<Vec<Self>, SDFParseError> {
        Self::from_sdf_with(sdf, SdfParseOptions::default())
    }

    /// Reads an SDF, with control over what is perceived afterwards — see
    /// [`SdfParseOptions`].
    pub fn from_sdf_with<T: AsRef<Path>>(
        sdf: T,
        options: SdfParseOptions,
    ) -> Result<Vec<Self>, SDFParseError> {
        let file = std::fs::File::open(sdf).map_err(SDFParseError::IoError)?;
        let reader = BufReader::new(file);
        let sdf_entries = parse_sdf_file(reader)?;
        Self::from_sdf_entries_with(sdf_entries, options)
    }

    pub fn from_sdf_entries(sdf_entries: Vec<SDFEntry>) -> Result<Vec<Self>, SDFParseError> {
        Self::from_sdf_entries_with(sdf_entries, SdfParseOptions::default())
    }

    /// Builds molecules from already-parsed entries, with control over what is
    /// perceived afterwards — see [`SdfParseOptions`].
    pub fn from_sdf_entries_with(
        sdf_entries: Vec<SDFEntry>,
        options: SdfParseOptions,
    ) -> Result<Vec<Self>, SDFParseError> {
        let mut molecules = vec![];

        for entry in sdf_entries.into_iter() {
            let number_of_atoms = entry.mol_file.atoms.len();
            let mut atomic_numbers = vec![];
            let mut charges = vec![];
            let mut positions = vec![];
            let mut radical_states = vec![];
            let mut explicit_valences = vec![];
            let mut isotopes = vec![0u16; number_of_atoms];

            for atom in &entry.mol_file.atoms {
                atomic_numbers.push(atom.symbol.as_str().atomic_number().ok_or(
                    SDFParseError::MOLParseError(MOLParseError::InvalidAtom(
                        atom.symbol.to_string(),
                    )),
                )?);
                charges.push(atom.formal_charge.unwrap_or(0));
                radical_states.push(atom.radical_electrons());
                positions.push(Vector3D::new(atom.x, atom.y, atom.z));
                explicit_valences.push(atom.explicit_valence());
            }
            for (index, atom) in entry.mol_file.atoms.iter().enumerate() {
                if let Some(mass_number) = atom.isotope() {
                    isotopes[index] = mass_number;
                }
            }

            // The `M  xxx` lines supersede the atom block outright rather than
            // adding to it, so each block that is present first clears the
            // values the atom block gave. A file that writes both -- which the
            // format encourages, the atom block entries being there for
            // readers too old to know the property lines -- would otherwise
            // keep whichever the atom block happened to say.
            if let Some(block_charges) = &entry.mol_file.charges {
                charges.iter_mut().for_each(|charge| *charge = 0);
                for &(index, charge) in block_charges {
                    if let Some(slot) = charges.get_mut(index) {
                        *slot = charge;
                    }
                }
            }
            if let Some(block_radicals) = &entry.mol_file.radicals {
                radical_states.iter_mut().for_each(|state| *state = 0);
                for &(index, electrons) in block_radicals {
                    if let Some(slot) = radical_states.get_mut(index) {
                        *slot = electrons;
                    }
                }
            }
            if let Some(block_isotopes) = &entry.mol_file.isotopes {
                isotopes.iter_mut().for_each(|isotope| *isotope = 0);
                for &(index, mass_number) in block_isotopes {
                    if let Some(slot) = isotopes.get_mut(index) {
                        *slot = mass_number;
                    }
                }
            }

            let mut bonds = vec![ArrayVec::<[BondTarget; 10]>::new(); number_of_atoms];
            let mut wedges = Vec::new();
            for (bond_index, bond) in entry.mol_file.bonds.into_iter().enumerate() {
                match entry.mol_file.bond_stereo.get(bond_index).copied() {
                    Some(0) | None => {}
                    Some(stereo) => wedges.push((bond.atom1, bond.atom2, stereo)),
                }
                bonds
                    .get_mut(bond.atom1)
                    .ok_or(SDFParseError::MOLParseError(MOLParseError::InvalidBond))?
                    .push(BondTarget::new(bond.atom2, bond.bond_order));
                bonds
                    .get_mut(bond.atom2)
                    .ok_or(SDFParseError::MOLParseError(MOLParseError::InvalidBond))?
                    .push(BondTarget::new(bond.atom1, bond.bond_order));
            }
            let mut molecule = Molecule3D {
                charges,
                positions,
                atomic_numbers,
                atom_bonds: bonds,
                // One entry per atom, not the empty vector `Default` gives:
                // the accessors index this directly, so an SDF-derived
                // molecule used to panic the moment anything asked whether an
                // atom was a radical.
                radical_states,
                // `None` rather than a vector of zeroes when nothing was
                // written, matching what the SMILES side stores: zero is not a
                // mass number, and a `Some` here would make every atom of every
                // SDF claim to carry isotope information.
                isotopes: isotopes.iter().any(|&i| i != 0).then_some(isotopes),
                ..Default::default()
            };

            if options.hydrogens {
                molecule.add_hydrogens_with_valences(&explicit_valences);
            }
            if options.stereochemistry {
                assign_stereochemistry(&mut molecule, &wedges);
            }
            if let Some(model) = options.aromaticity {
                // A MOL block can only write a ring in Kekulé form, so without
                // this benzene read from an SDF and benzene read from
                // `c1ccccc1` answer aromaticity questions differently. Failure
                // leaves the bonds exactly as written, which is the same result
                // as not asking.
                let _ = molecule.aromatize(model, None, false);
            }
            molecules.push(molecule);
        }

        Ok(molecules)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_sdf_file() {
        let sdf_content = "\
        5460033
  -OEChem-09092410012D

 11  8  0     0  0  0  0  0  0999 V2000
    3.2690    0.1614    0.0000 Pt  0  0  0  0  0  0  0  0  0  0  0  0
    3.9761    0.8685    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    3.9761   -0.5458    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    2.8464   -0.7450    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.5619    0.8685    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    2.5844   -1.3069    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.2844   -0.4829    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.4084   -1.0070    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1235    1.3069    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.0003    1.3069    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.6064    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
  4  6  1  0  0  0  0
  4  7  1  0  0  0  0
  4  8  1  0  0  0  0
  5  9  1  0  0  0  0
  5 10  1  0  0  0  0
  5 11  1  0  0  0  0
M  END
> <PUBCHEM_COMPOUND_CID>
5460033

> <PUBCHEM_COMPOUND_CANONICALIZED>
1

> <PUBCHEM_CACTVS_COMPLEXITY>
7.6

> <PUBCHEM_CACTVS_HBOND_ACCEPTOR>
2

> <PUBCHEM_CACTVS_HBOND_DONOR>
2

> <PUBCHEM_CACTVS_ROTATABLE_BOND>
0

> <PUBCHEM_CACTVS_SUBSKEYS>
AAADcYADAAAGAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==

> <PUBCHEM_IUPAC_OPENEYE_NAME>
ammonia;dichloroplatinum

> <PUBCHEM_IUPAC_CAS_NAME>
ammonia;dichloroplatinum

> <PUBCHEM_IUPAC_NAME_MARKUP>
azane;dichloroplatinum

> <PUBCHEM_IUPAC_NAME>
azane;dichloroplatinum

> <PUBCHEM_IUPAC_SYSTEMATIC_NAME>
azane;bis(chloranyl)platinum

> <PUBCHEM_IUPAC_TRADITIONAL_NAME>
ammonia;dichloroplatinum

> <PUBCHEM_IUPAC_INCHI>
InChI=1S/2ClH.2H3N.Pt/h2*1H;2*1H3;/q;;;;+2/p-2

> <PUBCHEM_IUPAC_INCHIKEY>
LXZZYRPGZAFOLE-UHFFFAOYSA-L

> <PUBCHEM_EXACT_MASS>
298.955598

> <PUBCHEM_MOLECULAR_FORMULA>
Cl2H6N2Pt

> <PUBCHEM_MOLECULAR_WEIGHT>
300.05

> <PUBCHEM_REFERENCE_STANDARDIZATION>
Bypass - this structure was created from CID 5460033

> <PUBCHEM_OPENEYE_CAN_SMILES>
N.N.Cl[Pt]Cl

> <PUBCHEM_OPENEYE_ISO_SMILES>
N.N.Cl[Pt]Cl

> <PUBCHEM_CACTVS_TPSA>
2

> <PUBCHEM_MONOISOTOPIC_WEIGHT>
298.955598

> <PUBCHEM_TOTAL_CHARGE>
0

> <PUBCHEM_HEAVY_ATOM_COUNT>
5

> <PUBCHEM_ATOM_DEF_STEREO_COUNT>
0

> <PUBCHEM_ATOM_UDEF_STEREO_COUNT>
0

> <PUBCHEM_BOND_DEF_STEREO_COUNT>
0

> <PUBCHEM_BOND_UDEF_STEREO_COUNT>
0

> <PUBCHEM_ISOTOPIC_ATOM_COUNT>
0

> <PUBCHEM_COMPONENT_COUNT>
3

> <PUBCHEM_CACTVS_TAUTO_COUNT>
-1

> <PUBCHEM_NONSTANDARDBOND>
1  4  6
1  5  6

> <PUBCHEM_COORDINATE_TYPE>
1
5
255

$$$$
";

        let reader = Cursor::new(sdf_content);
        let result = parse_sdf_file(reader);
        assert!(result.is_ok());

        let entries = result.unwrap();
        assert_eq!(entries.len(), 1);

        // Check first compound
        let first = &entries[0];
        assert_eq!(first.mol_file.atoms.len(), 11);
        assert_eq!(first.mol_file.bonds.len(), 8);
        assert_eq!(first.data_fields.len(), 34);
        assert_eq!(
            first.data_fields.get("PUBCHEM_COMPOUND_CID"),
            Some(&"5460033".to_string())
        );
    }

    #[test]
    fn test_parse_sdf_file_single_entry() {
        let sdf_content = "\
Compound 1
  -OEChem-01152411432D

  1  0  0     0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
M  END
> <PUBCHEM_COMPOUND_CID>
1

$$$$
";

        let reader = Cursor::new(sdf_content);
        let result = parse_sdf_file(reader);
        assert!(result.is_ok());

        let entries = result.unwrap();
        assert_eq!(entries.len(), 1);

        let entry = &entries[0];
        assert_eq!(entry.mol_file.atoms.len(), 1);
        assert_eq!(entry.mol_file.bonds.len(), 0);
        assert_eq!(entry.data_fields.len(), 1);
        assert_eq!(
            entry.data_fields.get("PUBCHEM_COMPOUND_CID"),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn test_parse_sdf_file_no_data_fields() {
        let sdf_content = "\
Compound 1
  -OEChem-01152411432D

  1  0  0     0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
";

        let reader = Cursor::new(sdf_content);
        let result = parse_sdf_file(reader);
        assert!(result.is_ok());

        let entries = result.unwrap();
        assert_eq!(entries.len(), 1);

        let entry = &entries[0];
        assert_eq!(entry.mol_file.atoms.len(), 1);
        assert_eq!(entry.mol_file.bonds.len(), 0);
        assert_eq!(entry.data_fields.len(), 0);
    }

    #[test]
    fn test_parse_sdf_file_missing_end_delimiter() {
        let sdf_content = "\
Compound 1
  -OEChem-01152411432D

  1  0  0     0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
M  END
> <PUBCHEM_COMPOUND_CID>
1
";

        let reader = Cursor::new(sdf_content);
        let result = parse_sdf_file(reader);
        assert!(result.is_ok());

        let entries = result.unwrap();
        assert_eq!(entries.len(), 1);

        let entry = &entries[0];
        assert_eq!(entry.mol_file.atoms.len(), 1);
        assert_eq!(entry.mol_file.bonds.len(), 0);
        assert_eq!(entry.data_fields.len(), 1);
        assert_eq!(
            entry.data_fields.get("PUBCHEM_COMPOUND_CID"),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn test_parse_sdf_file_empty_input() {
        let sdf_content = "";
        let reader = Cursor::new(sdf_content);
        let result = parse_sdf_file(reader);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }
}
