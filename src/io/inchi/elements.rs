//! InChI's element table: the reference mass an isotopic shift is counted
//! from, whether InChI treats the element as a metal, and the valences it
//! allows at each charge from -2 to +2.
//!
//! The reader takes all three from InChI rather than from the crate's own
//! tables because they are part of what an InChI *means*. `/i2-1` on a
//! bromine is bromine-79 because InChI counts from 80, the rounded average
//! mass, and not from 79, the most abundant isotope. And a perchlorate's
//! chlorine has four σ bonds, which the crate's valence table -- chlorine is
//! monovalent there -- cannot place at all.

/// One row of the table.
pub(super) struct Element {
    pub symbol: &'static str,
    /// The integer mass an isotopic shift is relative to.
    pub mass: u16,
    /// Metals are disconnected by standard InChI and carry no valence model.
    pub metal: bool,
    /// The valences allowed at charge -2, -1, 0, +1 and +2, in that order.
    pub valences: [&'static [u8]; 5],
}

impl Element {
    /// The valences allowed at `charge`, or none outside -2..=+2.
    pub fn valences_at(&self, charge: i32) -> &'static [u8] {
        match charge {
            -2..=2 => self.valences[(charge + 2) as usize],
            _ => &[],
        }
    }
}

/// The row for `atomic_number`, if there is one.
pub(super) fn element(atomic_number: u8) -> Option<&'static Element> {
    ELEMENTS
        .get(atomic_number as usize)
        .filter(|element| !element.symbol.is_empty())
}

#[rustfmt::skip]
// Generated from InChI 1.07.5 `INCHI_BASE/src/util.c` (`ElData`); see THIRD_PARTY.md.
pub(super) const ELEMENTS: [Element; 119] = [
    Element { symbol: "", mass: 0, metal: false, valences: [&[], &[], &[], &[], &[]] },
    Element { symbol: "H", mass: 1, metal: false, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "He", mass: 4, metal: false, valences: [&[], &[], &[], &[], &[]] },
    Element { symbol: "Li", mass: 7, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Be", mass: 9, metal: true, valences: [&[], &[], &[2], &[1], &[]] },
    Element { symbol: "B", mass: 11, metal: false, valences: [&[3], &[4], &[3], &[2], &[1]] },
    Element { symbol: "C", mass: 12, metal: false, valences: [&[2], &[3], &[4], &[3], &[2]] },
    Element { symbol: "N", mass: 14, metal: false, valences: [&[1], &[2], &[3, 5], &[4], &[3]] },
    Element { symbol: "O", mass: 16, metal: false, valences: [&[], &[1], &[2], &[3, 5], &[4]] },
    Element { symbol: "F", mass: 19, metal: false, valences: [&[], &[], &[1], &[2], &[3, 5]] },
    Element { symbol: "Ne", mass: 20, metal: false, valences: [&[], &[], &[], &[], &[]] },
    Element { symbol: "Na", mass: 23, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Mg", mass: 24, metal: true, valences: [&[], &[], &[2], &[1], &[]] },
    Element { symbol: "Al", mass: 27, metal: true, valences: [&[3, 5], &[4], &[3], &[2], &[1]] },
    Element { symbol: "Si", mass: 28, metal: false, valences: [&[2], &[3, 5], &[4], &[3], &[2]] },
    Element { symbol: "P", mass: 31, metal: false, valences: [&[1, 3, 5, 7], &[2, 4, 6], &[3, 5], &[4], &[3]] },
    Element { symbol: "S", mass: 32, metal: false, valences: [&[], &[1, 3, 5, 7], &[2, 4, 6], &[3, 5], &[4]] },
    Element { symbol: "Cl", mass: 35, metal: false, valences: [&[], &[], &[1, 3, 5, 7], &[2, 4, 6], &[3, 5]] },
    Element { symbol: "Ar", mass: 40, metal: false, valences: [&[], &[], &[], &[], &[]] },
    Element { symbol: "K", mass: 39, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Ca", mass: 40, metal: true, valences: [&[], &[], &[2], &[1], &[]] },
    Element { symbol: "Sc", mass: 45, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Ti", mass: 48, metal: true, valences: [&[], &[], &[3, 4], &[], &[]] },
    Element { symbol: "V", mass: 51, metal: true, valences: [&[], &[], &[2, 3, 4, 5], &[], &[]] },
    Element { symbol: "Cr", mass: 52, metal: true, valences: [&[], &[], &[2, 3, 6], &[], &[]] },
    Element { symbol: "Mn", mass: 55, metal: true, valences: [&[], &[], &[2, 3, 4, 6], &[], &[]] },
    Element { symbol: "Fe", mass: 56, metal: true, valences: [&[], &[], &[2, 3, 4, 6], &[], &[]] },
    Element { symbol: "Co", mass: 59, metal: true, valences: [&[], &[], &[2, 3], &[], &[]] },
    Element { symbol: "Ni", mass: 59, metal: true, valences: [&[], &[], &[2, 3], &[], &[]] },
    Element { symbol: "Cu", mass: 64, metal: true, valences: [&[], &[], &[1, 2], &[], &[]] },
    Element { symbol: "Zn", mass: 65, metal: true, valences: [&[], &[], &[2], &[], &[]] },
    Element { symbol: "Ga", mass: 70, metal: true, valences: [&[3, 5], &[4], &[3], &[], &[1]] },
    Element { symbol: "Ge", mass: 73, metal: false, valences: [&[2, 4, 6], &[3, 5], &[4], &[3], &[]] },
    Element { symbol: "As", mass: 75, metal: false, valences: [&[1, 3, 5, 7], &[2, 4, 6], &[3, 5], &[4], &[3]] },
    Element { symbol: "Se", mass: 79, metal: false, valences: [&[], &[1, 3, 5, 7], &[2, 4, 6], &[3, 5], &[4]] },
    Element { symbol: "Br", mass: 80, metal: false, valences: [&[], &[], &[1, 3, 5, 7], &[2, 4, 6], &[3, 5]] },
    Element { symbol: "Kr", mass: 84, metal: false, valences: [&[], &[], &[], &[], &[]] },
    Element { symbol: "Rb", mass: 85, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Sr", mass: 88, metal: true, valences: [&[], &[], &[2], &[1], &[]] },
    Element { symbol: "Y", mass: 89, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Zr", mass: 91, metal: true, valences: [&[], &[], &[4], &[], &[]] },
    Element { symbol: "Nb", mass: 93, metal: true, valences: [&[], &[], &[3, 5], &[], &[]] },
    Element { symbol: "Mo", mass: 96, metal: true, valences: [&[], &[], &[3, 4, 5, 6], &[], &[]] },
    Element { symbol: "Tc", mass: 98, metal: true, valences: [&[], &[], &[7], &[], &[]] },
    Element { symbol: "Ru", mass: 101, metal: true, valences: [&[], &[], &[2, 3, 4, 6], &[], &[]] },
    Element { symbol: "Rh", mass: 103, metal: true, valences: [&[], &[], &[2, 3, 4], &[], &[]] },
    Element { symbol: "Pd", mass: 106, metal: true, valences: [&[], &[], &[2, 4], &[], &[]] },
    Element { symbol: "Ag", mass: 108, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Cd", mass: 112, metal: true, valences: [&[], &[], &[2], &[], &[]] },
    Element { symbol: "In", mass: 115, metal: true, valences: [&[3, 5], &[2, 4], &[3], &[], &[1]] },
    Element { symbol: "Sn", mass: 119, metal: true, valences: [&[2, 4, 6], &[3, 5], &[2, 4], &[3], &[]] },
    Element { symbol: "Sb", mass: 122, metal: true, valences: [&[1, 3, 5, 7], &[2, 4, 6], &[3, 5], &[2, 4], &[3]] },
    Element { symbol: "Te", mass: 128, metal: false, valences: [&[], &[1, 3, 5, 7], &[2, 4, 6], &[3, 5], &[2, 4]] },
    Element { symbol: "I", mass: 127, metal: false, valences: [&[], &[], &[1, 3, 5, 7], &[2, 4, 6], &[3, 5]] },
    Element { symbol: "Xe", mass: 131, metal: false, valences: [&[], &[], &[], &[], &[]] },
    Element { symbol: "Cs", mass: 133, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Ba", mass: 137, metal: true, valences: [&[], &[], &[2], &[1], &[]] },
    Element { symbol: "La", mass: 139, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Ce", mass: 140, metal: true, valences: [&[], &[], &[3, 4], &[], &[]] },
    Element { symbol: "Pr", mass: 141, metal: true, valences: [&[], &[], &[3, 4], &[], &[]] },
    Element { symbol: "Nd", mass: 144, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Pm", mass: 145, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Sm", mass: 150, metal: true, valences: [&[], &[], &[2, 3], &[], &[]] },
    Element { symbol: "Eu", mass: 152, metal: true, valences: [&[], &[], &[2, 3], &[], &[]] },
    Element { symbol: "Gd", mass: 157, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Tb", mass: 159, metal: true, valences: [&[], &[], &[3, 4], &[], &[]] },
    Element { symbol: "Dy", mass: 163, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Ho", mass: 165, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Er", mass: 167, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Tm", mass: 169, metal: true, valences: [&[], &[], &[2, 3], &[], &[]] },
    Element { symbol: "Yb", mass: 173, metal: true, valences: [&[], &[], &[2, 3], &[], &[]] },
    Element { symbol: "Lu", mass: 175, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Hf", mass: 178, metal: true, valences: [&[], &[], &[4], &[], &[]] },
    Element { symbol: "Ta", mass: 181, metal: true, valences: [&[], &[], &[5], &[], &[]] },
    Element { symbol: "W", mass: 184, metal: true, valences: [&[], &[], &[3, 4, 5, 6], &[], &[]] },
    Element { symbol: "Re", mass: 186, metal: true, valences: [&[], &[], &[2, 4, 6, 7], &[], &[]] },
    Element { symbol: "Os", mass: 190, metal: true, valences: [&[], &[], &[2, 3, 4, 6], &[], &[]] },
    Element { symbol: "Ir", mass: 192, metal: true, valences: [&[], &[], &[2, 3, 4, 6], &[], &[]] },
    Element { symbol: "Pt", mass: 195, metal: true, valences: [&[], &[], &[2, 4], &[], &[]] },
    Element { symbol: "Au", mass: 197, metal: true, valences: [&[], &[], &[1, 3], &[], &[]] },
    Element { symbol: "Hg", mass: 201, metal: true, valences: [&[], &[], &[1, 2], &[], &[]] },
    Element { symbol: "Tl", mass: 204, metal: true, valences: [&[3, 5], &[2, 4], &[1, 3], &[], &[]] },
    Element { symbol: "Pb", mass: 207, metal: true, valences: [&[2, 4, 6], &[3, 5], &[2, 4], &[3], &[]] },
    Element { symbol: "Bi", mass: 209, metal: true, valences: [&[1, 3, 5, 7], &[2, 4, 6], &[3, 5], &[2, 4], &[3]] },
    Element { symbol: "Po", mass: 209, metal: true, valences: [&[], &[1, 3, 5, 7], &[2, 4, 6], &[3, 5], &[2, 4]] },
    Element { symbol: "At", mass: 210, metal: false, valences: [&[], &[], &[1, 3, 5, 7], &[2, 4, 6], &[3, 5]] },
    Element { symbol: "Rn", mass: 222, metal: false, valences: [&[], &[], &[], &[], &[]] },
    Element { symbol: "Fr", mass: 223, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Ra", mass: 226, metal: true, valences: [&[], &[], &[2], &[1], &[]] },
    Element { symbol: "Ac", mass: 227, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Th", mass: 232, metal: true, valences: [&[], &[], &[3, 4], &[], &[]] },
    Element { symbol: "Pa", mass: 231, metal: true, valences: [&[], &[], &[3, 4, 5], &[], &[]] },
    Element { symbol: "U", mass: 238, metal: true, valences: [&[], &[], &[3, 4, 5, 6], &[], &[]] },
    Element { symbol: "Np", mass: 237, metal: true, valences: [&[], &[], &[3, 4, 5, 6], &[], &[]] },
    Element { symbol: "Pu", mass: 244, metal: true, valences: [&[], &[], &[3, 4, 5, 6], &[], &[]] },
    Element { symbol: "Am", mass: 243, metal: true, valences: [&[], &[], &[3, 4, 5, 6], &[], &[]] },
    Element { symbol: "Cm", mass: 247, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Bk", mass: 247, metal: true, valences: [&[], &[], &[3, 4], &[], &[]] },
    Element { symbol: "Cf", mass: 251, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Es", mass: 252, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Fm", mass: 257, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "Md", mass: 258, metal: true, valences: [&[], &[], &[3], &[], &[]] },
    Element { symbol: "No", mass: 259, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Lr", mass: 260, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Rf", mass: 261, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Db", mass: 270, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Sg", mass: 269, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Bh", mass: 270, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Hs", mass: 270, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Mt", mass: 278, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Ds", mass: 281, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Rg", mass: 281, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Cn", mass: 285, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Nh", mass: 278, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Fl", mass: 289, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Mc", mass: 289, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Lv", mass: 293, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Ts", mass: 297, metal: true, valences: [&[], &[], &[1], &[], &[]] },
    Element { symbol: "Og", mass: 294, metal: true, valences: [&[], &[], &[1], &[], &[]] },
];
