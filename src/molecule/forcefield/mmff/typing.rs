//! Which MMFF94 atom type an atom is.
//!
//! MMFF names its types symbolically -- `CR` for an alkyl carbon, `C5A` for
//! the carbon alpha to the heteroatom of a five-membered aromatic ring,
//! `NGD+` for a guanidinium nitrogen -- and numbers them 1 to 99. The number
//! is what the parameter tables are keyed on; the symbol is what makes a
//! mistyping legible when one shows up.
//!
//! The rules below follow RDKit's `AtomTyper.cpp` decision for decision,
//! including the order the tests are made in, because that order is part of
//! the answer: a sulfur with two oxygens and a double bond to carbon is a
//! sulfone before it is anything else, and a five-ring carbon's alpha and
//! beta heteroatoms are weighed against each other in a sequence no summary
//! of the rules preserves.

use super::super::UntypedAtom;
use super::perception::Perception;
use crate::molecule::base::MoleculeCore;
use crate::molecule::bond::BondOrder;

/// One of MMFF94's 99 numbered atom types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MmffType(u8);

impl MmffType {
    /// The number the MMFF94 parameter tables are keyed on, 1 to 99.
    pub fn number(self) -> u8 {
        self.0
    }

    /// The symbolic name, as MMFF94 writes it.
    ///
    /// MMFF94 has more symbolic names than numbers: the divalent oxygen of
    /// an ester, an enol and a sulfate are `OC=O`, `OC=C` and `OSO`, and all
    /// three are numeric type 6, which is the only thing the parameter
    /// tables know. What comes back is the name of the family, `OR` in that
    /// case -- enough to read a mistyping by, not a full symbolic typing.
    pub fn symbol(self) -> &'static str {
        symbol(self.0)
    }
}

impl std::fmt::Display for MmffType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.0, self.symbol())
    }
}

/// The MMFF94 type of every atom, in atom order.
///
/// # Errors
///
/// [`UntypedAtom`] for the first atom MMFF94 has no type for. That is a
/// wider net than it sounds: MMFF94 covers organic main-group chemistry and
/// a short list of counterions, so a transition metal has no type, and
/// neither does a perfectly ordinary organic atom in a valence state the
/// published rules never enumerated.
pub fn atom_types<M: MoleculeCore + ?Sized>(molecule: &M) -> Result<Vec<MmffType>, UntypedAtom> {
    let (_, types) = typed(molecule)?;
    Ok(types.into_iter().map(MmffType).collect())
}

/// What everything else in this module starts from: the perception of the
/// molecule, and every atom's type number alongside it.
pub(super) fn typed<M: MoleculeCore + ?Sized>(
    molecule: &M,
) -> Result<(Perception, Vec<u8>), UntypedAtom> {
    let perception = Perception::new(molecule);
    let count = perception.atom_count();
    let mut types = vec![0u8; count];

    for (atom, number) in types.iter_mut().enumerate() {
        if perception.element(atom) != 1 {
            *number = heavy_atom_type(&perception, atom);
        }
    }
    // Hydrogens are typed by what their heavy neighbour turned out to be, so
    // they wait for the whole first pass rather than for their own atom.
    let hydrogens: Vec<(usize, u8)> = (0..count)
        .filter(|&atom| perception.element(atom) == 1)
        .map(|atom| (atom, hydrogen_type(&perception, &types, atom)))
        .collect();
    for (atom, number) in hydrogens {
        types[atom] = number;
    }

    match types.iter().position(|&number| number == 0) {
        Some(atom) => Err(UntypedAtom::mmff(atom, perception.element(atom))),
        None => Ok((perception, types)),
    }
}

/// Whether an atom is a nitrogen carrying a terminal oxygen -- the N of an
/// N-oxide, which is typed apart from the nitrogen it otherwise looks like
/// everywhere it can occur.
fn is_n_oxide(perception: &Perception, atom: usize) -> bool {
    perception.element(atom) == 7
        && perception.degree(atom) >= 3
        && perception.neighbours(atom).iter().any(|neighbour| {
            perception.element(neighbour.atom) == 8 && perception.degree(neighbour.atom) == 1
        })
}

/// The heteroatoms alpha and beta to `atom` within its own five-membered
/// aromatic ring: oxygen, sulfur, or a trivalent nitrogen that is not an
/// N-oxide. Which of the two sets is populated is what separates `C5A` from
/// `C5B`, and their overlap is what separates both from `C5`.
fn five_ring_heteroatoms(perception: &Perception, atom: usize) -> (Vec<usize>, Vec<usize>) {
    let mut alpha = Vec::new();
    let mut beta = Vec::new();

    let is_heteroatom = |candidate: usize| {
        matches!(perception.element(candidate), 8 | 16)
            || (perception.element(candidate) == 7
                && perception.degree(candidate) == 3
                && !is_n_oxide(perception, candidate))
    };

    for neighbour in perception.neighbours(atom) {
        let first = neighbour.atom;
        if !perception.in_aromatic_ring_of_size(first, 5) {
            continue;
        }
        if perception.in_same_ring_of_size(5, &[atom, first]) && is_heteroatom(first) {
            alpha.push(first);
        }
        for second in perception.neighbours(first) {
            let second = second.atom;
            if second == atom || !perception.in_aromatic_ring_of_size(second, 5) {
                continue;
            }
            if perception.in_same_ring_of_size(5, &[atom, second]) && is_heteroatom(second) {
                beta.push(second);
            }
        }
    }

    (alpha, beta)
}

fn heavy_atom_type(perception: &Perception, atom: usize) -> u8 {
    if perception.is_aromatic(atom) {
        let aromatic = aromatic_atom_type(perception, atom);
        if aromatic != 0 {
            return aromatic;
        }
    }
    aliphatic_atom_type(perception, atom)
}

fn aromatic_atom_type(perception: &Perception, atom: usize) -> u8 {
    let element = perception.element(atom);

    if perception.in_aromatic_ring_of_size(atom, 5) {
        let (alpha, beta) = if matches!(element, 6 | 7) {
            five_ring_heteroatoms(perception, atom)
        } else {
            (Vec::new(), Vec::new())
        };
        let alpha_is_chalcogen = alpha
            .iter()
            .any(|&other| matches!(perception.element(other), 8 | 16));
        let beta_is_chalcogen = beta
            .iter()
            .any(|&other| matches!(perception.element(other), 8 | 16));
        let alpha_or_beta_in_same_ring = alpha.iter().any(|&first| {
            beta.iter()
                .any(|&second| perception.in_same_ring_of_size(5, &[first, second]))
        });

        match element {
            6 => {
                if beta.is_empty() {
                    // Imidazolium's central carbon: two or three trivalent
                    // nitrogens, at least one of them in this five-ring, one
                    // of them protonated, and none of them in a six-ring.
                    let mut nitrogens = 0;
                    let mut charged = 0;
                    let mut in_five_ring = 0;
                    let mut in_six_ring = 0;
                    for neighbour in perception.neighbours(atom) {
                        let other = neighbour.atom;
                        if perception.element(other) != 7 || perception.degree(other) != 3 {
                            continue;
                        }
                        nitrogens += 1;
                        if perception.formal_charge(other) > 0 && !is_n_oxide(perception, other) {
                            charged += 1;
                        }
                        if perception.in_aromatic_ring_of_size(other, 5) {
                            in_five_ring += 1;
                        }
                        if perception.in_aromatic_ring_of_size(other, 6) {
                            in_six_ring += 1;
                        }
                    }
                    if ((nitrogens == 2 && in_five_ring > 0)
                        || (nitrogens == 3 && in_five_ring == 2))
                        && charged > 0
                        && in_six_ring == 0
                    {
                        // CIM+, the carbon between imidazolium's nitrogens.
                        return 80;
                    }
                }

                if alpha.len() == beta.len() {
                    let mut surrounded_by_benzene_carbon = true;
                    let mut surrounded_by_aromatic = true;
                    for neighbour in perception.neighbours(atom) {
                        let other = neighbour.atom;
                        if perception.element(other) != 6
                            || !perception.in_ring_of_size(other, 6)
                        {
                            surrounded_by_benzene_carbon = false;
                        }
                        if perception.in_same_ring_of_size(5, &[atom, other])
                            && !perception.is_aromatic(other)
                        {
                            surrounded_by_aromatic = false;
                        }
                    }
                    if (alpha.is_empty()
                        && beta.is_empty()
                        && !surrounded_by_benzene_carbon
                        && surrounded_by_aromatic)
                        || (!alpha.is_empty()
                            && !beta.is_empty()
                            && (!alpha_or_beta_in_same_ring
                                || (!alpha_is_chalcogen && !beta_is_chalcogen)))
                    {
                        // C5, a five-ring carbon with no side to belong to.
                        return 78;
                    }
                }
                if !alpha.is_empty() && (beta.is_empty() || alpha_is_chalcogen) {
                    // C5A
                    return 63;
                }
                if !beta.is_empty() && (alpha.is_empty() || beta_is_chalcogen) {
                    // C5B
                    return 64;
                }
            }

            7 => {
                if is_n_oxide(perception, atom) {
                    // N5AX / N5BX / N5OX
                    return 82;
                }
                if alpha.is_empty() && beta.is_empty() {
                    // NPYL where the lone pair is in the ring, N5M where the
                    // ring is an anion.
                    return if perception.degree(atom) == 3 { 39 } else { 76 };
                }
                if perception.degree(atom) == 3 && alpha.len() != beta.len() {
                    // NIM+ / N5A+ / N5B+ / N5+
                    return 81;
                }
                if !alpha.is_empty() && (beta.is_empty() || alpha_is_chalcogen) {
                    // N5A
                    return 65;
                }
                if !beta.is_empty() && (alpha.is_empty() || beta_is_chalcogen) {
                    // N5B
                    return 66;
                }
                if !alpha.is_empty() && !beta.is_empty() {
                    // N5
                    return 79;
                }
            }

            // OFUR, furan's oxygen.
            8 => return 59,
            // STHI, thiophene's sulfur.
            16 => return 44,
            _ => (),
        }
    }

    if perception.in_aromatic_ring_of_size(atom, 6) {
        match element {
            // CB, benzene's carbon.
            6 => return 37,
            7 => {
                if is_n_oxide(perception, atom) {
                    // NPOX
                    return 69;
                }
                // NPD+ where the nitrogen carries a substituent, NPYD where
                // its lone pair is in the plane.
                return if perception.degree(atom) == 3 { 58 } else { 38 };
            }
            _ => (),
        }
    }

    0
}

fn aliphatic_atom_type(perception: &Perception, atom: usize) -> u8 {
    match perception.element(atom) {
        3 => {
            if perception.degree(atom) == 0 {
                return 92;
            }
        }
        6 => return aliphatic_carbon_type(perception, atom),
        7 => return aliphatic_nitrogen_type(perception, atom),
        8 => return aliphatic_oxygen_type(perception, atom),
        9 => {
            return match perception.degree(atom) {
                1 => 11,
                0 => 89,
                _ => 0,
            };
        }
        11 => {
            if perception.degree(atom) == 0 {
                return 93;
            }
        }
        12 => {
            if perception.degree(atom) == 0 {
                return 99;
            }
        }
        // SI, which MMFF types without looking at anything else.
        14 => return 19,
        15 => {
            return match perception.degree(atom) {
                4 => 25,
                3 => 26,
                2 => 75,
                _ => 0,
            };
        }
        16 => return aliphatic_sulfur_type(perception, atom),
        17 => {
            if perception.degree(atom) == 4
                && perception
                    .neighbours(atom)
                    .iter()
                    .filter(|neighbour| perception.element(neighbour.atom) == 8)
                    .count()
                    == 4
            {
                // CLO4
                return 77;
            }
            return match perception.degree(atom) {
                1 => 12,
                0 => 90,
                _ => 0,
            };
        }
        19 => {
            if perception.degree(atom) == 0 {
                return 94;
            }
        }
        20 => {
            if perception.degree(atom) == 0 {
                return 96;
            }
        }
        26 => {
            if perception.degree(atom) == 0 {
                return match perception.formal_charge(atom) {
                    2 => 87,
                    3 => 88,
                    _ => 0,
                };
            }
        }
        29 => {
            if perception.degree(atom) == 0 {
                return match perception.formal_charge(atom) {
                    1 => 97,
                    2 => 98,
                    _ => 0,
                };
            }
        }
        30 => {
            if perception.degree(atom) == 0 {
                return 95;
            }
        }
        35 => {
            return match perception.degree(atom) {
                1 => 13,
                0 => 91,
                _ => 0,
            };
        }
        53 if perception.degree(atom) == 1 => return 14,
        _ => (),
    }
    0
}

fn aliphatic_carbon_type(perception: &Perception, atom: usize) -> u8 {
    match perception.degree(atom) {
        4 => {
            if perception.in_ring_of_size(atom, 3) {
                // CR3R
                return 22;
            }
            if perception.in_ring_of_size(atom, 4) {
                // CR4R
                return 20;
            }
            // CR, the alkyl carbon.
            1
        }
        3 => {
            let mut double_bonded_element = 0u8;
            let mut terminal_oxygens = 0;
            let mut terminal_sulfurs = 0;
            let mut trivalent_nitrogens = 0;
            let mut double_bonded_nitrogens = 0;
            for neighbour in perception.neighbours(atom) {
                let other = neighbour.atom;
                let double = neighbour.order == BondOrder::Double;
                if double {
                    double_bonded_element = perception.element(other);
                }
                if perception.degree(other) == 1 {
                    match perception.element(other) {
                        8 => terminal_oxygens += 1,
                        16 => terminal_sulfurs += 1,
                        _ => (),
                    }
                } else if perception.element(other) == 7 {
                    if perception.degree(other) == 3 {
                        trivalent_nitrogens += 1;
                    } else if perception.degree(other) == 2 && double {
                        double_bonded_nitrogens += 1;
                    }
                }
            }
            if trivalent_nitrogens >= 2 && double_bonded_nitrogens == 0 && double_bonded_element == 7
            {
                // CNN+ / CGD+
                return 57;
            }
            if terminal_oxygens == 2 || terminal_sulfurs == 2 {
                // CO2M / CS2M
                return 41;
            }
            if perception.in_ring_of_size(atom, 4) && double_bonded_element == 6 {
                // CR4E, the olefinic carbon of a four-ring.
                return 30;
            }
            if matches!(double_bonded_element, 7 | 8 | 15 | 16) {
                // C=O and every other carbon doubly bonded to a heteroatom.
                return 3;
            }
            // C=C, the vinylic carbon.
            2
        }
        // CSP, and the central carbon of an allene.
        2 => 4,
        // C%-, the isonitrile carbon.
        1 => 60,
        _ => 0,
    }
}

fn aliphatic_nitrogen_type(perception: &Perception, atom: usize) -> u8 {
    let mut terminal_oxygens = 0;
    // Set where the nitrogen hangs off a sulfur or phosphorus carrying two
    // or more terminal oxygens -- a sulfonamide and its relatives -- and
    // later also by a nitrogen attached to a cyano group.
    let mut is_sulfonamide_or_cyanamide = false;

    for neighbour in perception.neighbours(atom) {
        let other = neighbour.atom;
        if perception.element(other) == 8 && perception.degree(other) == 1 {
            terminal_oxygens += 1;
        }
        if perception.valence(atom) >= 3 && matches!(perception.element(other), 15 | 16) {
            let oxygens = perception
                .neighbours(other)
                .iter()
                .filter(|second| {
                    perception.element(second.atom) == 8 && perception.degree(second.atom) == 1
                })
                .count();
            if oxygens >= 2 {
                is_sulfonamide_or_cyanamide = true;
            }
        }
    }

    if perception.degree(atom) == 4 {
        // N3OX where a terminal oxygen is there, NR+ otherwise.
        return if is_n_oxide(perception, atom) { 68 } else { 34 };
    }

    if perception.degree(atom) == 3 {
        if perception.valence(atom) >= 4 {
            let mut double_bonded_carbon_or_nitrogen = false;
            for neighbour in perception.neighbours(atom) {
                if neighbour.order != BondOrder::Double {
                    continue;
                }
                let other = neighbour.atom;
                double_bonded_carbon_or_nitrogen = matches!(perception.element(other), 6 | 7);
                if perception.element(other) == 6 {
                    for second in perception.neighbours(other) {
                        if second.atom == atom || !double_bonded_carbon_or_nitrogen {
                            continue;
                        }
                        double_bonded_carbon_or_nitrogen = !(perception.element(second.atom) == 7
                            && perception.degree(second.atom) == 3);
                    }
                }
            }
            if terminal_oxygens == 1 {
                // N2OX
                return 67;
            }
            if terminal_oxygens >= 2 {
                // NO2 / NO3
                return 45;
            }
            if double_bonded_carbon_or_nitrogen {
                // N+=C / N+=N
                return 54;
            }
        }

        if perception.valence(atom) >= 3 {
            let mut is_amide_or_thioamide = false;
            let mut is_amidinium = false;
            let mut is_guanidinium = false;
            let mut is_azo_or_hydrazone = false;
            let mut has_carbon_neighbour = false;
            let mut has_benzene_carbon_neighbour = false;
            let mut element_double_bonded_to_carbon = 0u8;
            let mut element_triple_bonded_to_carbon = 0u8;
            let mut aromatic_oxygens_on_carbon = 0;
            let mut aromatic_sulfurs_on_carbon = 0;

            for neighbour in perception.neighbours(atom) {
                let other = neighbour.atom;

                if perception.element(other) == 6 {
                    has_carbon_neighbour = true;
                    if perception.is_aromatic(other) && perception.in_ring_of_size(other, 6) {
                        has_benzene_carbon_neighbour = true;
                    }
                    let mut divalent_nitrogens_on_carbon = 0;
                    let mut trivalent_nitrogens_on_carbon = 0;
                    let mut charged_nitrogens = 0;
                    let mut nitrogens_in_six_ring = 0;
                    aromatic_oxygens_on_carbon = 0;
                    aromatic_sulfurs_on_carbon = 0;

                    for second in perception.neighbours(other) {
                        let far = second.atom;
                        let order = second.order;
                        if order == BondOrder::Double && matches!(perception.element(far), 8 | 16) {
                            is_amide_or_thioamide = true;
                        }
                        // An aromatic bond stands in for a double one, but
                        // only to carbon, or to a nitrogen that belongs to a
                        // single ring: a bridgehead nitrogen's aromatic bond
                        // is not the delocalized C=N this rule is after.
                        if order == BondOrder::Double
                            || (order == BondOrder::Aromatic
                                && (perception.element(far) == 6
                                    || (perception.element(far) == 7
                                        && perception.ring_count(far) == 1)))
                        {
                            element_double_bonded_to_carbon = perception.element(far);
                        }
                        if order == BondOrder::Triple {
                            element_triple_bonded_to_carbon = perception.element(far);
                        }
                        if perception.element(far) == 7 && perception.degree(far) == 3 {
                            if perception.formal_charge(far) == 1 {
                                charged_nitrogens += 1;
                            }
                            if perception.in_aromatic_ring_of_size(other, 6) {
                                nitrogens_in_six_ring += 1;
                            }
                            let oxygens = perception
                                .neighbours(far)
                                .iter()
                                .filter(|third| perception.element(third.atom) == 8)
                                .count();
                            if oxygens < 2 {
                                trivalent_nitrogens_on_carbon += 1;
                            }
                        }
                        if perception.element(far) == 7
                            && perception.degree(far) == 2
                            && matches!(order, BondOrder::Double | BondOrder::Aromatic)
                        {
                            divalent_nitrogens_on_carbon += 1;
                        }
                        if perception.is_aromatic(far) {
                            match perception.element(far) {
                                8 => aromatic_oxygens_on_carbon += 1,
                                16 => aromatic_sulfurs_on_carbon += 1,
                                _ => (),
                            }
                        }
                    }

                    if element_double_bonded_to_carbon == 7 {
                        if trivalent_nitrogens_on_carbon == 2
                            && divalent_nitrogens_on_carbon == 0
                            && charged_nitrogens > 0
                            && nitrogens_in_six_ring == 0
                            && perception.degree(other) < 4
                        {
                            is_amidinium = true;
                        }
                        if trivalent_nitrogens_on_carbon == 3 {
                            is_guanidinium = true;
                        }
                    }
                }

                if perception.element(other) == 7 {
                    for second in perception.neighbours(other) {
                        if second.order != BondOrder::Double {
                            continue;
                        }
                        let far = second.atom;
                        if perception.element(far) == 6 {
                            let mut nitrogens = 0;
                            let mut oxygens = 0;
                            let mut sulfurs = 0;
                            for third in perception.neighbours(far) {
                                if third.atom == other {
                                    continue;
                                }
                                match perception.element(third.atom) {
                                    7 => nitrogens += 1,
                                    8 => oxygens += 1,
                                    16 => sulfurs += 1,
                                    _ => (),
                                }
                            }
                            if nitrogens == 0
                                && oxygens == 0
                                && sulfurs == 0
                                && !has_benzene_carbon_neighbour
                            {
                                is_azo_or_hydrazone = true;
                            }
                        }
                        if perception.element(far) == 7 && !has_benzene_carbon_neighbour {
                            is_azo_or_hydrazone = true;
                        }
                    }
                }
            }

            if has_carbon_neighbour {
                if element_triple_bonded_to_carbon == 7 {
                    is_sulfonamide_or_cyanamide = true;
                }
                if is_amidinium {
                    // NCN+
                    return 55;
                }
                if is_guanidinium {
                    // NGD+
                    return 56;
                }
                if !is_amide_or_thioamide
                    && !is_sulfonamide_or_cyanamide
                    && ((aromatic_oxygens_on_carbon == 0
                        && aromatic_sulfurs_on_carbon == 0
                        && has_benzene_carbon_neighbour)
                        || matches!(element_double_bonded_to_carbon, 6 | 7 | 15)
                        || element_triple_bonded_to_carbon == 6)
                {
                    // NC=C / NC=N / NC=P / NC%C, the delocalized lone pair.
                    return 40;
                }
            }
            if !is_sulfonamide_or_cyanamide && (is_amide_or_thioamide || is_azo_or_hydrazone) {
                // NC=O / NC=S / NN=C / NN=N
                return 10;
            }
        }
    }

    if perception.degree(atom) == 2 {
        if perception.valence(atom) == 4 {
            let is_isonitrile = perception
                .neighbours(atom)
                .iter()
                .any(|neighbour| neighbour.order == BondOrder::Triple);
            // NR% for the isonitrile, =N= for the middle of an azide.
            return if is_isonitrile { 61 } else { 53 };
        }
        if perception.valence(atom) == 3 {
            let mut is_nitroso = false;
            let mut is_imine_or_azo = false;
            for neighbour in perception.neighbours(atom) {
                if neighbour.order != BondOrder::Double {
                    continue;
                }
                is_nitroso =
                    perception.element(neighbour.atom) == 8 && terminal_oxygens == 1;
                is_imine_or_azo = matches!(perception.element(neighbour.atom), 6 | 7);
            }
            if is_nitroso && !is_imine_or_azo {
                // N=O
                return 46;
            }
            if is_imine_or_azo {
                // N=C / N=N
                return 9;
            }
        }
        if perception.valence(atom) >= 2 {
            let is_sulfinylamine = perception.neighbours(atom).iter().any(|neighbour| {
                perception.element(neighbour.atom) == 16
                    && perception
                        .neighbours(neighbour.atom)
                        .iter()
                        .filter(|second| {
                            perception.element(second.atom) == 8
                                && perception.degree(second.atom) == 1
                        })
                        .count()
                        == 1
            });
            if is_sulfinylamine {
                // NSO
                return 48;
            }
            if !is_sulfonamide_or_cyanamide {
                // NM, the anionic divalent nitrogen.
                return 62;
            }
        }
    }

    if is_sulfonamide_or_cyanamide {
        // NSO2 / NSO3 / NC%N
        return 43;
    }

    if perception.degree(atom) == 1 {
        let mut is_nitrile = false;
        let mut is_terminal_azide = false;
        for neighbour in perception.neighbours(atom) {
            if is_nitrile || is_terminal_azide {
                break;
            }
            is_nitrile = neighbour.order == BondOrder::Triple;
            let other = neighbour.atom;
            if perception.element(other) == 7 && perception.degree(other) == 2 {
                is_terminal_azide = perception.neighbours(other).iter().any(|second| {
                    (perception.element(second.atom) == 7 && perception.degree(second.atom) == 2)
                        || (perception.element(second.atom) == 6
                            && perception.degree(second.atom) == 3)
                });
            }
        }
        if is_nitrile {
            // NSP
            return 42;
        }
        if is_terminal_azide {
            // NAZT
            return 47;
        }
    }

    // NR, the amine nitrogen, which is what is left.
    8
}

fn aliphatic_oxygen_type(perception: &Perception, atom: usize) -> u8 {
    if perception.degree(atom) == 3 {
        // O+
        return 49;
    }

    if perception.degree(atom) == 2 {
        if perception.valence(atom) == 3 {
            // O=+
            return 51;
        }
        let hydrogens = perception
            .neighbours(atom)
            .iter()
            .filter(|neighbour| perception.element(neighbour.atom) == 1)
            .count();
        if hydrogens == 2 {
            // OH2
            return 70;
        }
        // OR, and every divalent oxygen that is not water.
        return 6;
    }

    if perception.degree(atom) <= 1 {
        let mut divalent_nitrogens = 0;
        let mut terminal_oxygens = 0;
        let mut terminal_sulfurs = 0;
        let mut is_oxide_on_hydrogen = false;
        let mut is_carboxylate = false;
        let mut is_carbonyl = false;
        let mut is_oxide_on_carbon = false;
        let mut is_nitroso = false;
        let mut is_oxide_on_nitrogen = false;
        let mut is_n_oxide_oxygen = false;
        let mut is_nitro = false;
        let mut is_thiosulfinate = false;
        let mut is_sulfate = false;
        let mut is_sulfoxide = false;
        let mut is_phosphate_or_perchlorate = false;

        for neighbour in perception.neighbours(atom) {
            let other = neighbour.atom;
            let order = neighbour.order;

            if matches!(perception.element(other), 6 | 7 | 16) {
                for second in perception.neighbours(other) {
                    let far = second.atom;
                    if perception.element(far) == 7 && perception.degree(far) == 2 {
                        divalent_nitrogens += 1;
                    }
                    if perception.element(far) == 8 && perception.degree(far) == 1 {
                        terminal_oxygens += 1;
                    }
                    if perception.element(far) == 16 && perception.degree(far) == 1 {
                        terminal_sulfurs += 1;
                    }
                }
            }

            is_oxide_on_hydrogen = perception.element(other) == 1;

            if perception.element(other) == 6 {
                is_carboxylate = terminal_oxygens == 2;
                is_carbonyl = order == BondOrder::Double;
                is_oxide_on_carbon = order == BondOrder::Single && terminal_oxygens == 1;
            }

            if perception.element(other) == 7 {
                is_nitroso = order == BondOrder::Double;
                if order == BondOrder::Single && terminal_oxygens == 1 {
                    is_oxide_on_nitrogen =
                        perception.degree(other) == 2 || perception.valence(other) == 3;
                    is_n_oxide_oxygen = perception.valence(other) == 4;
                }
                is_nitro = terminal_oxygens >= 2;
            }

            if perception.element(other) == 16 {
                is_thiosulfinate = terminal_sulfurs == 1;
                is_sulfate = order == BondOrder::Single
                    || (order == BondOrder::Double
                        && (terminal_oxygens + divalent_nitrogens) > 1);
                is_sulfoxide =
                    order == BondOrder::Double && (terminal_oxygens + divalent_nitrogens) == 1;
            }

            is_phosphate_or_perchlorate = matches!(perception.element(other), 15 | 17);
        }

        if is_oxide_on_carbon || is_oxide_on_nitrogen || is_oxide_on_hydrogen {
            // OM / OM2, the alkoxide.
            return 35;
        }
        if is_carboxylate
            || is_nitro
            || is_n_oxide_oxygen
            || is_thiosulfinate
            || is_sulfate
            || is_phosphate_or_perchlorate
        {
            // O2CM and the rest of the terminal oxygens on a charged centre.
            return 32;
        }
        if is_carbonyl || is_nitroso || is_sulfoxide {
            // O=C / O=N / O=S
            return 7;
        }
    }

    0
}

fn aliphatic_sulfur_type(perception: &Perception, atom: usize) -> u8 {
    let degree = perception.degree(atom);

    if degree == 3 || degree == 4 {
        let mut oxygens_or_nitrogens = 0;
        let mut terminal_sulfurs = 0;
        let mut double_bonded_to_carbon = false;
        for neighbour in perception.neighbours(atom) {
            let other = neighbour.atom;
            if perception.element(other) == 6 && neighbour.order == BondOrder::Double {
                double_bonded_to_carbon = true;
            }
            if (perception.degree(other) == 1 && perception.element(other) == 8)
                || (perception.degree(other) == 2 && perception.element(other) == 7)
            {
                oxygens_or_nitrogens += 1;
            }
            if perception.degree(other) == 1 && perception.element(other) == 16 {
                terminal_sulfurs += 1;
            }
        }
        if (degree == 3 && oxygens_or_nitrogens == 2 && double_bonded_to_carbon) || degree == 4 {
            // =SO2, the sulfone.
            return 18;
        }
        if (oxygens_or_nitrogens > 0 && terminal_sulfurs > 0)
            || (oxygens_or_nitrogens == 2 && !double_bonded_to_carbon)
        {
            // SSOM, the anionic thiosulfinate.
            return 73;
        }
        // S=O / >S=N, the sulfoxide.
        return 17;
    }

    if degree == 2 {
        let double_bonded_to_oxygen = perception.neighbours(atom).iter().any(|neighbour| {
            perception.element(neighbour.atom) == 8 && neighbour.order == BondOrder::Double
        });
        // =S=O for the sulfinyl, S for every thiol, sulfide and disulfide.
        return if double_bonded_to_oxygen { 74 } else { 15 };
    }

    if degree == 1 {
        let mut terminal_sulfurs_on_neighbour = 0;
        let mut double_bonded_to_carbon = false;
        for neighbour in perception.neighbours(atom) {
            let other = neighbour.atom;
            for second in perception.neighbours(other) {
                if perception.element(second.atom) == 16 && perception.degree(second.atom) == 1 {
                    terminal_sulfurs_on_neighbour += 1;
                }
            }
            if perception.element(other) == 6 && neighbour.order == BondOrder::Double {
                double_bonded_to_carbon = true;
            }
        }
        if double_bonded_to_carbon && terminal_sulfurs_on_neighbour != 2 {
            // S=C, the thioketone -- but not a dithiocarboxylate, whose two
            // sulfurs share the charge.
            return 16;
        }
        // S-P / SM / SSMO
        return 72;
    }

    0
}

fn hydrogen_type(perception: &Perception, types: &[u8], atom: usize) -> u8 {
    let Some(neighbour) = perception.neighbours(atom).first() else {
        return 0;
    };
    let other = neighbour.atom;

    match perception.element(other) {
        // HC on carbon, HSI on silicon.
        6 | 14 => 5,
        7 => match types[other] {
            // HNR and the rest of the hydrogens on a neutral sp3 nitrogen.
            8 | 39 | 62 | 67 | 68 => 23,
            // HNR+ / HIM+, on a nitrogen carrying the charge.
            34 | 54 | 55 | 56 | 58 | 81 => 36,
            // HN=N / HN=C
            9 => 27,
            // HNCO and the rest of the delocalized-lone-pair nitrogens.
            _ => 28,
        },
        8 => match types[other] {
            // HO+
            49 => 50,
            // HO=+
            51 => 52,
            // HOH, water's own hydrogen.
            70 => 31,
            6 => hydroxyl_hydrogen_type(perception, other),
            // HO / HOR
            _ => 21,
        },
        // HP / HS
        15 | 16 => 71,
        _ => 0,
    }
}

/// Which kind of hydroxyl hydrogen this is: the acidic one of a carboxylic
/// or phosphorus acid, the enolic or phenolic one, the one on an oxygen
/// bonded to sulfur, or a plain alcohol's.
fn hydroxyl_hydrogen_type(perception: &Perception, oxygen: usize) -> u8 {
    let mut is_enolic = false;
    let mut is_acidic = false;
    let mut on_sulfur = false;

    for neighbour in perception.neighbours(oxygen) {
        let other = neighbour.atom;
        if perception.element(other) == 6 {
            for second in perception.neighbours(other) {
                if second.atom == oxygen {
                    continue;
                }
                let conjugated =
                    matches!(second.order, BondOrder::Double | BondOrder::Aromatic);
                if matches!(perception.element(second.atom), 6 | 7) && conjugated {
                    is_enolic = true;
                }
                if perception.element(second.atom) == 8 && second.order == BondOrder::Double {
                    is_acidic = true;
                }
            }
        }
        if perception.element(other) == 15 {
            is_acidic = true;
        }
        if perception.element(other) == 16 {
            on_sulfur = true;
        }
    }

    if is_acidic {
        // HOCO, the carboxylic or phosphoric acid hydrogen.
        return 24;
    }
    if is_enolic {
        // HOCC / HOCN
        return 29;
    }
    if on_sulfur {
        // HOS
        return 33;
    }
    // HOR
    21
}

/// MMFF94's symbolic name for a numbered type.
fn symbol(number: u8) -> &'static str {
    match number {
        1 => "CR", 2 => "C=C", 3 => "C=O", 4 => "CSP", 5 => "HC",
        6 => "OR", 7 => "O=C", 8 => "NR", 9 => "N=C", 10 => "NC=O",
        11 => "F", 12 => "CL", 13 => "BR", 14 => "I", 15 => "S",
        16 => "S=C", 17 => "S=O", 18 => "=SO2", 19 => "SI", 20 => "CR4R",
        21 => "HOR", 22 => "CR3R", 23 => "HNR", 24 => "HOCO", 25 => "PO4",
        26 => "P", 27 => "HN=C", 28 => "HNCO", 29 => "HOCC", 30 => "CR4E",
        31 => "HOH", 32 => "O2CM", 33 => "HOS", 34 => "NR+", 35 => "OM",
        36 => "HNR+", 37 => "CB", 38 => "NPYD", 39 => "NPYL", 40 => "NC=C",
        41 => "CO2M", 42 => "NSP", 43 => "NSO2", 44 => "STHI", 45 => "NO2",
        46 => "N=O", 47 => "NAZT", 48 => "NSO", 49 => "O+", 50 => "HO+",
        51 => "O=+", 52 => "HO=+", 53 => "=N=", 54 => "N+=C", 55 => "NCN+",
        56 => "NGD+", 57 => "CNN+", 58 => "NPD+", 59 => "OFUR", 60 => "C%-",
        61 => "NR%", 62 => "NM", 63 => "C5A", 64 => "C5B", 65 => "N5A",
        66 => "N5B", 67 => "N2OX", 68 => "N3OX", 69 => "NPOX", 70 => "OH2",
        71 => "HS", 72 => "SM", 73 => "SSOM", 74 => "=S=O", 75 => "-P=C",
        76 => "N5M", 77 => "CLO4", 78 => "C5", 79 => "N5", 80 => "CIM+",
        81 => "NIM+", 82 => "N5AX", 87 => "FE+2", 88 => "FE+3", 89 => "F-",
        90 => "CL-", 91 => "BR-", 92 => "LI+", 93 => "NA+", 94 => "K+",
        95 => "ZN+2", 96 => "CA+2", 97 => "CU+1", 98 => "CU+2", 99 => "MG+2",
        _ => "?",
    }
}
