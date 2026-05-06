/*!
# CxSMARTS (ChemAxon Extended SMARTS) Specification and Implementation

## CxSMARTS Format Specification

CxSMARTS is an extension of the standard SMARTS (SMILES Arbitrary Target Specification)
format developed by ChemAxon. It extends SMARTS with additional features for enhanced
chemical pattern matching and molecular query capabilities.

### Base SMARTS Specification

CxSMARTS builds upon the standard SMARTS specification:

#### Atomic Properties
- `*` - wildcard (any atom)
- `a` - aromatic atom
- `A` - aliphatic atom
- `D<n>` - degree (exactly n explicit connections)
- `H<n>` - total hydrogen count (exactly n attached hydrogens)
- `h<n>` - implicit hydrogen count (exactly n implicit hydrogens)
- `R<n>` - ring membership (in exactly n SSSR rings)
- `r<n>` - ring size (in smallest SSSR ring of size n)
- `v<n>` - valence (total bond order n)
- `X<n>` - connectivity (n total connections)
- `+<n>` - positive charge (+n formal charge)
- `-<n>` - negative charge (-n charge)
- `#<n>` - atomic number
- `@` - anticlockwise chirality
- `@@` - clockwise chirality
- `@<class><n>` - specific chiral class
- `@<class><n>?` - chiral or unspecified

#### Bond Properties
- `-` - single bond (aliphatic)
- `=` - double bond
- `#` - triple bond
- `:` - aromatic bond
- `~` - any bond (wildcard)
- `@` - any ring bond
- `/` - directional single bond "up"
- `\` - directional single bond "down"
- `/?` - directional bond "up or unspecified"
- `\?` - directional bond "down or unspecified"

#### Logical Operators
- `!e1` - NOT e1
- `e1&e2` - e1 AND e2 (high precedence)
- `e1,e2` - e1 OR e2
- `e1;e2` - e1 AND e2 (low precedence)

#### Recursive SMARTS
- `$(SMARTS)` - recursive pattern matching

### ChemAxon Extensions (CxSMARTS)

#### Extended Atom Properties
- `[C:1]` - atom class/mapping (already supported in base implementation)
- `[*:123]` - multi-digit atom classes
- `[15NH3+:1]` - isotope with atom class
- `[C&$(CC)]` - recursive patterns with atom classes

#### Extended Ring Closure Syntax
- `C1.C1` - fragment connection via ring closure
- `O1.N1` - heteroatom fragment connections
- Multi-component molecules connected via ring closures

#### Enhanced Chirality Support
- Extended chiral classes: `@TH`, `@AL`, `@SP`, `@TB`, `@OH`
- Enhanced chirality with atom classes: `[C@:1]`

#### Component-Level Grouping
- `(SMARTS)` - single component grouping
- `(SMARTS1).(SMARTS2)` - multi-component queries
- `(C.C)` - both atoms in same component
- `(C).(C)` - atoms in different components

#### Reaction SMARTS Extensions
- `[C:1]>>[C:1]` - mapped reaction patterns
- `>>products` - product-only patterns
- `reactants>>` - reactant-only patterns

### Examples

#### Basic CxSMARTS Patterns
```smarts
[C:1]               # Carbon with atom class 1
[*:123]             # Any atom with atom class 123
[NH2:1]             # Primary amine with atom class 1
C1.C1               # Two carbons connected via ring closure
O1.N1               # Oxygen and nitrogen connected
[C@:1]              # Chiral carbon with atom class 1
```

#### Complex Patterns
```smarts
[$(c1ccccc1):1]     # Benzene ring with atom class 1
(C.C)               # Two carbons in same component
(C).(C)             # Two carbons in different components
[C:1]>>[O:1]        # Carbon to oxygen transformation
```

### Implementation Notes

This implementation provides:
1. Full SMARTS parsing with ChemAxon extensions
2. Atom class support (`:n` syntax)
3. Ring closure fragment connections
4. Component-level grouping
5. Enhanced chirality support
6. Reaction pattern matching

### References
- ChemAxon Extended SMILES and SMARTS Documentation
- Daylight SMARTS Theory Manual
- SMILES Arbitrary Target Specification (Weininger et al.)
*/

use std::collections::HashMap;
use std::fmt;

use crate::molecule::chirality::ChiralClass;
use crate::molecule::molecular_graph::MolecularGraph;

/// Errors that can occur during CxSMARTS parsing
#[derive(Debug, Clone, PartialEq)]
pub enum CxSMARTSError {
    /// Invalid character or sequence in SMARTS
    InvalidSyntax(String),
    /// Element not found in periodic table
    ElementNotFound(String),
    /// Invalid atom property specification
    InvalidAtomProperty(String),
    /// Invalid bond specification
    InvalidBond(String),
    /// Mismatched brackets or parentheses
    MismatchedBrackets,
    /// Invalid logical operator usage
    InvalidLogicalOperator(String),
    /// Invalid recursive SMARTS pattern
    InvalidRecursivePattern(String),
    /// Invalid atom class specification
    InvalidAtomClass(String),
    /// Invalid ring closure
    InvalidRingClosure(String),
    /// End of input reached unexpectedly
    UnexpectedEndOfInput,
    /// Invalid chirality specification
    InvalidChirality(String),
    /// Invalid reaction SMARTS
    InvalidReactionSMARTS(String),
}

impl fmt::Display for CxSMARTSError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CxSMARTSError::InvalidSyntax(msg) => write!(f, "Invalid syntax: {}", msg),
            CxSMARTSError::ElementNotFound(element) => write!(f, "Element not found: {}", element),
            CxSMARTSError::InvalidAtomProperty(prop) => {
                write!(f, "Invalid atom property: {}", prop)
            }
            CxSMARTSError::InvalidBond(bond) => write!(f, "Invalid bond: {}", bond),
            CxSMARTSError::MismatchedBrackets => write!(f, "Mismatched brackets"),
            CxSMARTSError::InvalidLogicalOperator(op) => {
                write!(f, "Invalid logical operator: {}", op)
            }
            CxSMARTSError::InvalidRecursivePattern(pattern) => {
                write!(f, "Invalid recursive pattern: {}", pattern)
            }
            CxSMARTSError::InvalidAtomClass(class) => write!(f, "Invalid atom class: {}", class),
            CxSMARTSError::InvalidRingClosure(closure) => {
                write!(f, "Invalid ring closure: {}", closure)
            }
            CxSMARTSError::UnexpectedEndOfInput => write!(f, "Unexpected end of input"),
            CxSMARTSError::InvalidChirality(chiral) => write!(f, "Invalid chirality: {}", chiral),
            CxSMARTSError::InvalidReactionSMARTS(rxn) => {
                write!(f, "Invalid reaction SMARTS: {}", rxn)
            }
        }
    }
}

impl std::error::Error for CxSMARTSError {}

/// Represents a logical operator in SMARTS expressions
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalOperator {
    /// NOT operator (!)
    Not,
    /// High precedence AND operator (&)
    HighAnd,
    /// Low precedence AND operator (;)
    LowAnd,
    /// OR operator (,)
    Or,
}

/// Represents an atomic expression in SMARTS
#[derive(Debug, Clone, PartialEq)]
pub enum AtomExpression {
    /// Element symbol (e.g., "C", "N", "O")
    Element(String),
    /// Atomic number (e.g., #6 for carbon)
    AtomicNumber(u8),
    /// Wildcard atom (*)
    Wildcard,
    /// Aromatic atom (a)
    Aromatic,
    /// Aliphatic atom (A)
    Aliphatic,
    /// Degree (number of explicit connections)
    Degree(u8),
    /// Total hydrogen count
    TotalHydrogens(u8),
    /// Implicit hydrogen count
    ImplicitHydrogens(u8),
    /// Ring membership count
    RingMembership(u8),
    /// Bare `R`: the atom lies on some ring, however many.
    InRing,
    /// Ring size
    RingSize(u8),
    /// SMARTS `r{m-n}`: the atom is in *a* ring whose size falls in the
    /// range, either end optional (`r{-8}`, `r{9-}`).
    ///
    /// An RDKit extension rather than Daylight SMARTS, supported here
    /// because RDKit's small-ring torsion rules
    /// (`molecule::embed::torsion_preferences`) are written with it and are
    /// unreadable without it.
    ///
    /// Uses the smallest ring through each bond, with no fixed size cutoff,
    /// so macrocycle rules such as `r{9-}` can match larger rings too.
    RingSizeRange(Option<u8>, Option<u8>),
    /// Valence
    Valence(u8),
    /// Connectivity (total connections)
    Connectivity(u8),
    /// SMARTS `^<n>`: the atom's hybridisation, numbered as RDKit numbers it --
    /// 1 for sp, 2 for sp2, 3 for sp3, 4 for sp3d, 5 for sp3d2. `^0`, an
    /// unhybridised atom, is a state this crate does not model and matches
    /// nothing.
    Hybridization(u8),
    /// Formal charge
    Charge(i8),
    /// Chirality
    Chirality(ChiralClass),
    /// Isotope mass
    Isotope(u16),
    /// Atom class/mapping
    AtomClass(usize),
    /// Recursive SMARTS pattern
    Recursive(String),
    /// Logical combination of expressions
    Logical {
        operator: LogicalOperator,
        operands: Vec<AtomExpression>,
    },
}

/// Represents a bond expression in SMARTS
#[derive(Debug, Clone, PartialEq)]
pub enum BondExpression {
    /// Single bond
    Single,
    /// Double bond
    Double,
    /// Triple bond
    Triple,
    /// Aromatic bond
    Aromatic,
    /// Any bond wildcard
    Any,
    /// The bond SMARTS assumes when none is written: single **or** aromatic.
    /// It is not the same as [`BondExpression::Single`] — `cc` has to match an
    /// aromatic bond, and `CC` an aliphatic one, without either being spelled
    /// out.
    SingleOrAromatic,
    /// Ring bond
    Ring,
    /// Directional up bond
    DirectionalUp,
    /// Directional down bond
    DirectionalDown,
    /// Directional up or unspecified
    DirectionalUpOrUnspec,
    /// Directional down or unspecified
    DirectionalDownOrUnspec,
    /// NOT bond expression
    Not(Box<BondExpression>),
    /// Logical combination
    Logical {
        operator: LogicalOperator,
        operands: Vec<BondExpression>,
    },
}

/// Represents a complete SMARTS pattern
#[derive(Debug, Clone)]
pub struct CxSMARTSPattern {
    /// The atoms in the pattern
    pub atoms: Vec<AtomExpression>,
    /// The bonds between atoms
    pub bonds: Vec<(usize, usize, BondExpression)>,
    /// Neighbours in query traversal order, including ring-closure positions.
    pub neighbor_order: Vec<Vec<usize>>,
    /// The atom from which each query atom was reached, or none at a root.
    pub incoming: Vec<Option<usize>>,
    /// Ring closures
    pub ring_closures: HashMap<usize, Vec<usize>>,
    /// Component groupings
    pub components: Vec<Vec<usize>>,
    /// Atom classes/mappings
    pub atom_classes: HashMap<usize, usize>,
    /// Whether this is a reaction SMARTS
    pub is_reaction: bool,
    /// Reactants (for reaction SMARTS)
    pub reactants: Option<Box<CxSMARTSPattern>>,
    /// Products (for reaction SMARTS)
    pub products: Option<Box<CxSMARTSPattern>>,
}

/// CxSMARTS parser
#[derive(Default)]
pub struct CxSMARTSParser {
    input: String,
    position: usize,
    atoms: Vec<AtomExpression>,
    bonds: Vec<(usize, usize, BondExpression)>,
    neighbor_order: Vec<Vec<usize>>,
    incoming: Vec<Option<usize>>,
    ring_closures: HashMap<usize, Vec<usize>>,
    components: Vec<Vec<usize>>,
    atom_classes: HashMap<usize, usize>,
    current_atom_index: usize,
    /// The atom a newly parsed atom bonds back to, or `None` at the start of a
    /// disconnected component.
    previous_atom: Option<usize>,
    /// Open parentheses, as `(atom to return to, atom index when opened)`.
    /// `(` after an atom is a *branch* and remembers that atom, so the atom
    /// following the branch bonds to the branch's parent rather than to the
    /// last atom inside it. `(` before any atom is SMARTS *component-level
    /// grouping* and has no parent, which is what the `None` records.
    branch_stack: Vec<(Option<usize>, usize)>,
    /// A bond symbol that has been read but not yet attached, because the atom
    /// (or ring-closure digit) it applies to has not been parsed yet.
    pending_bond: Option<BondExpression>,
    /// Ring-closure digits seen once, waiting for their partner: number ->
    /// (atom that opened it, bond symbol written with it).
    open_ring_closures: HashMap<usize, (usize, Option<BondExpression>, usize)>,
}

impl CxSMARTSParser {
    /// Create a new CxSMARTS parser
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a CxSMARTS string into a pattern
    /// `{m-n}` after an `r`, with either bound optional: `{-8}` is "at most
    /// 8", `{9-}` is "at least 9", `{5-6}` is both.
    fn parse_ring_size_range(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        self.advance_position(); // past '{'
        let bound = |parser: &mut Self| -> Result<Option<u8>, CxSMARTSError> {
            if parser.position < parser.input.len() && parser.current_char()?.is_ascii_digit() {
                Ok(Some(parser.parse_number()? as u8))
            } else {
                Ok(None)
            }
        };
        let minimum = bound(self)?;
        if self.position >= self.input.len() || self.current_char()? != '-' {
            return Err(CxSMARTSError::InvalidSyntax(
                "ring size range needs a '-' between its bounds".to_string(),
            ));
        }
        self.advance_position();
        let maximum = bound(self)?;
        if self.position >= self.input.len() || self.current_char()? != '}' {
            return Err(CxSMARTSError::InvalidSyntax(
                "unterminated ring size range".to_string(),
            ));
        }
        self.advance_position();
        if minimum.is_none() && maximum.is_none() {
            return Err(CxSMARTSError::InvalidSyntax(
                "ring size range needs at least one bound".to_string(),
            ));
        }
        Ok(AtomExpression::RingSizeRange(minimum, maximum))
    }

    pub fn parse(&mut self, smarts_pattern: &str) -> Result<CxSMARTSPattern, CxSMARTSError> {
        if smarts_pattern.trim().is_empty() {
            return Err(CxSMARTSError::InvalidSyntax(
                "empty SMARTS pattern".to_string(),
            ));
        }
        self.input = smarts_pattern.to_string();
        self.position = 0;
        self.atoms.clear();
        self.bonds.clear();
        self.neighbor_order.clear();
        self.incoming.clear();
        self.ring_closures.clear();
        self.components.clear();
        self.atom_classes.clear();
        self.current_atom_index = 0;
        self.previous_atom = None;
        self.branch_stack.clear();
        self.pending_bond = None;
        self.open_ring_closures.clear();

        // Check for reaction SMARTS (contains >> or >)
        if smarts_pattern.contains(">>") || smarts_pattern.contains('>') {
            self.parse_reaction_smarts()
        } else {
            self.parse_molecule_smarts()
        }
    }

    /// Parse a reaction SMARTS pattern
    fn parse_reaction_smarts(&mut self) -> Result<CxSMARTSPattern, CxSMARTSError> {
        // Find the reaction arrow
        if let Some(arrow_pos) = self.input.find(">>") {
            let reactant_part = &self.input[..arrow_pos];
            let product_part = &self.input[arrow_pos + 2..];

            let reactants = if !reactant_part.is_empty() {
                let mut reactant_parser = CxSMARTSParser::new();
                Some(Box::new(reactant_parser.parse(reactant_part)?))
            } else {
                None
            };

            let products = if !product_part.is_empty() {
                let mut product_parser = CxSMARTSParser::new();
                Some(Box::new(product_parser.parse(product_part)?))
            } else {
                None
            };

            Ok(CxSMARTSPattern {
                atoms: self.atoms.clone(),
                bonds: self.bonds.clone(),
                neighbor_order: self.neighbor_order.clone(),
                incoming: self.incoming.clone(),
                ring_closures: self.ring_closures.clone(),
                components: self.components.clone(),
                atom_classes: self.atom_classes.clone(),
                is_reaction: true,
                reactants,
                products,
            })
        } else {
            Err(CxSMARTSError::InvalidReactionSMARTS(
                "No reaction arrow found".to_string(),
            ))
        }
    }

    /// Parse a molecule SMARTS pattern
    fn parse_molecule_smarts(&mut self) -> Result<CxSMARTSPattern, CxSMARTSError> {
        while self.position < self.input.len() {
            self.parse_atom_or_bond()?;
        }

        Ok(CxSMARTSPattern {
            atoms: self.atoms.clone(),
            bonds: self.bonds.clone(),
            neighbor_order: self.neighbor_order.clone(),
            incoming: self.incoming.clone(),
            ring_closures: self.ring_closures.clone(),
            components: self.components.clone(),
            atom_classes: self.atom_classes.clone(),
            is_reaction: false,
            reactants: None,
            products: None,
        })
    }

    /// Parse an atom or bond from the current position
    fn parse_atom_or_bond(&mut self) -> Result<(), CxSMARTSError> {
        if self.position >= self.input.len() {
            return Ok(());
        }

        let current_character = self.current_char()?;
        match current_character {
            '[' => self.parse_bracketed_atom(),
            '(' => self.parse_branch_open(),
            ')' => self.parse_branch_close(),
            '.' => self.parse_disconnection(),
            '0'..='9' | '%' => self.parse_ring_closure(),
            'A'..='Z' | 'a'..='z' | '*' => self.parse_simple_atom(),
            '-' | '=' | '#' | ':' | '~' | '@' | '/' | '\\' | '!' => self.parse_bond(),
            _ => Err(CxSMARTSError::InvalidSyntax(format!(
                "Unexpected character: {}",
                current_character
            ))),
        }
    }

    /// Parse a bracketed atom expression like [C:1], [NH2], [C,N], [!O], etc.
    fn parse_bracketed_atom(&mut self) -> Result<(), CxSMARTSError> {
        self.advance_position(); // Skip '['

        // A SMILES-like hydrogen atom is the exception to H's usual meaning
        // as an attached-hydrogen count in a SMARTS expression.
        let atom_expression = match self.parse_hydrogen_atom()? {
            Some(atom) => atom,
            None => self.parse_logical_atom_expression()?,
        };

        // Check for atom class at the end
        let mut atom_class = None;
        if self.position < self.input.len() && self.current_char()? == ':' {
            self.advance_position(); // Skip ':'
            atom_class = Some(self.parse_number()?);
        }

        // Expect closing bracket
        if self.position >= self.input.len() || self.current_char()? != ']' {
            return Err(CxSMARTSError::MismatchedBrackets);
        }
        self.advance_position(); // Skip ']'

        // Add the atom
        self.atoms.push(atom_expression);

        // Store atom class if present
        if let Some(class) = atom_class {
            self.atom_classes.insert(self.current_atom_index, class);
        }

        self.attach_atom(self.current_atom_index);
        self.current_atom_index += 1;
        Ok(())
    }

    /// Recognise exactly `[<isotope>H<charge><map>]`. In every other
    /// expression, including `[H1]`, `[H,C]` and `[H;D1]`, H remains a count.
    /// Look ahead before consuming anything so recursive/logical expressions
    /// continue through the normal parser unchanged.
    fn parse_hydrogen_atom(&mut self) -> Result<Option<AtomExpression>, CxSMARTSError> {
        let Some((body, _)) = self.input[self.position..].split_once(']') else {
            return Ok(None);
        };
        let atom = if let Some((atom, map)) = body.split_once(':') {
            if map.is_empty() || !map.bytes().all(|b| b.is_ascii_digit()) {
                return Ok(None);
            }
            atom
        } else {
            body
        };
        let isotope_end = atom.bytes().take_while(u8::is_ascii_digit).count();
        let Some(suffix) = atom[isotope_end..].strip_prefix('H') else {
            return Ok(None);
        };
        let charge = if suffix.is_empty() {
            None
        } else {
            let sign = suffix.as_bytes()[0];
            if sign != b'+' && sign != b'-' {
                return Ok(None);
            }
            let magnitude = &suffix[1..];
            let charge = if !magnitude.is_empty() && magnitude.bytes().all(|b| b.is_ascii_digit()) {
                suffix.parse::<i8>().ok()
            } else if magnitude.bytes().all(|b| b == sign) {
                let signed = if sign == b'+' { 1 } else { -1 } * suffix.len() as i64;
                i8::try_from(signed).ok()
            } else {
                return Ok(None);
            };
            Some(charge.ok_or_else(|| {
                CxSMARTSError::InvalidAtomProperty("Hydrogen charge is out of range".to_string())
            })?)
        };

        let mut operands = vec![AtomExpression::AtomicNumber(1)];
        if isotope_end != 0 {
            let isotope = atom[..isotope_end].parse::<u16>().map_err(|_| {
                CxSMARTSError::InvalidAtomProperty("Hydrogen isotope is out of range".to_string())
            })?;
            operands.push(AtomExpression::Isotope(isotope));
        }
        if let Some(charge) = charge {
            operands.push(AtomExpression::Charge(charge));
        }
        self.position += atom.len(); // Leave the optional map and closing bracket to the caller.
        Ok(Some(if operands.len() == 1 {
            operands.pop().unwrap()
        } else {
            AtomExpression::Logical {
                operator: LogicalOperator::HighAnd,
                operands,
            }
        }))
    }

    /// Parse a simple atom like C, N, O, c, n, o, *
    fn parse_simple_atom(&mut self) -> Result<(), CxSMARTSError> {
        let current_character = self.current_char()?;
        self.advance_position();

        let atom_expression = match current_character {
            '*' => AtomExpression::Wildcard,
            'A' => AtomExpression::Aliphatic,
            'a' => AtomExpression::Aromatic,
            // `'A'` and `'a'` are matched above, so starting at B is safe --
            // and necessary: the range used to start at C, which left boron
            // and every bromine written outside brackets (`Br`, `CBr`) as a
            // syntax error.
            'B'..='Z' => {
                // Outside brackets only the organic subset is writable, and
                // its only two-letter symbols are Cl and Br. Treating *any*
                // uppercase-then-lowercase pair as a two-letter element reads
                // `Cc1ccccc1` (a methyl on an aromatic ring) as a single atom
                // of the non-existent element "Cc", and likewise mangles
                // `Nc1ccccc1`, `Oc1ccccc1` and every other aliphatic atom
                // written before an aromatic one.
                let next_character = if self.position < self.input.len() {
                    self.current_char().unwrap_or('\0')
                } else {
                    '\0'
                };
                let two_letter = matches!(
                    (current_character, next_character),
                    ('C', 'l') | ('B', 'r')
                );
                if two_letter {
                    self.advance_position();
                    AtomExpression::Element(format!("{}{}", current_character, next_character))
                } else {
                    AtomExpression::Element(current_character.to_string())
                }
            }
            'b'..='z' => AtomExpression::Element(current_character.to_string()),
            _ => {
                return Err(CxSMARTSError::InvalidSyntax(format!(
                    "Invalid atom: {}",
                    current_character
                )));
            }
        };

        self.atoms.push(atom_expression);
        self.attach_atom(self.current_atom_index);
        self.current_atom_index += 1;
        Ok(())
    }

    /// Parse a logical atom expression with proper operator precedence
    /// Precedence: ! (highest) > & > , > ; (lowest)
    fn parse_logical_atom_expression(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        self.parse_low_precedence_expression()
    }

    /// Parse low precedence AND expressions (;)
    fn parse_low_precedence_expression(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        let mut left_expression = self.parse_or_expression()?;

        while self.position < self.input.len() {
            let current_character = self.current_char()?;
            if current_character == ';' {
                self.advance_position(); // Skip ';'
                let right_expression = self.parse_or_expression()?;
                left_expression = AtomExpression::Logical {
                    operator: LogicalOperator::LowAnd,
                    operands: vec![left_expression, right_expression],
                };
            } else {
                break;
            }
        }

        Ok(left_expression)
    }

    /// Parse OR expressions (,)
    fn parse_or_expression(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        let mut left_expression = self.parse_high_precedence_expression()?;

        while self.position < self.input.len() {
            let current_character = self.current_char()?;
            if current_character == ',' {
                self.advance_position(); // Skip ','
                let right_expression = self.parse_high_precedence_expression()?;
                left_expression = AtomExpression::Logical {
                    operator: LogicalOperator::Or,
                    operands: vec![left_expression, right_expression],
                };
            } else {
                break;
            }
        }

        Ok(left_expression)
    }

    /// Parse high precedence AND expressions (&)
    fn parse_high_precedence_expression(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        let mut left_expression = self.parse_not_expression()?;

        while self.position < self.input.len() {
            let current_character = self.current_char()?;
            if current_character == '&' {
                self.advance_position(); // Skip '&'
                let right_expression = self.parse_not_expression()?;
                left_expression = AtomExpression::Logical {
                    operator: LogicalOperator::HighAnd,
                    operands: vec![left_expression, right_expression],
                };
            } else {
                break;
            }
        }

        Ok(left_expression)
    }

    /// Parse NOT expressions (!)
    fn parse_not_expression(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        self.parse_primary_atom_expression()
    }

    /// One primitive, negated as many times as it is written to be.
    ///
    /// `!` binds to the primitive that follows it and to nothing else:
    /// `[#6+0!$(...)]` is a neutral carbon that is also not the recursive
    /// pattern, three primitives anded together, and negating the whole chain
    /// would say something quite different.
    fn parse_negated_primitive(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        if self.position < self.input.len() && self.current_char()? == '!' {
            self.advance_position();
            let expression = self.parse_negated_primitive()?;
            return Ok(AtomExpression::Logical {
                operator: LogicalOperator::Not,
                operands: vec![expression],
            });
        }
        self.parse_atom_expression()
    }

    /// Parse primary atom expressions (atoms, properties, recursive patterns)
    fn parse_primary_atom_expression(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        // Handle implicit AND (no operator between properties)
        let mut expressions = Vec::new();
        expressions.push(self.parse_negated_primitive()?);

        // Look for additional properties that should be ANDed together
        while self.position < self.input.len() {
            let current_character = self.current_char()?;

            // Stop at logical operators or closing bracket or atom class. A
            // `!` is not one of those: it begins another primitive, anded on
            // like any other, which is how `[N!H0v3]` and `[#6+0!$(...)]` are
            // written.
            match current_character {
                '&' | ',' | ';' | ']' | ':' => break,
                '!' => expressions.push(self.parse_negated_primitive()?),
                _ => {
                    // Try to parse another atom property
                    if let Ok(expr) = self.parse_atom_expression() {
                        expressions.push(expr);
                    } else {
                        break;
                    }
                }
            }
        }

        // If we have multiple expressions, combine with implicit AND
        if expressions.len() == 1 {
            Ok(expressions.into_iter().next().unwrap())
        } else {
            Ok(AtomExpression::Logical {
                operator: LogicalOperator::HighAnd,
                operands: expressions,
            })
        }
    }

    /// Parse an atom expression (used within bracketed atoms)
    fn parse_atom_expression(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        let current_character = self.current_char()?;

        // A two-letter element symbol wins over any primitive that starts with
        // the same letter. Without this `[Dy]` is a degree with no number after
        // it, `[Rn]` is "in a ring and an aromatic nitrogen", and `[Hf]` is "one
        // hydrogen and a fluorine" -- none of which can match anything, and all
        // of which are what half the lanthanides and every noble gas parse as.
        // Nothing is lost: no two-letter element's reading as a primitive pair
        // describes an atom that could exist.
        if current_character.is_ascii_uppercase() {
            let next_character = self
                .input
                .chars()
                .nth(self.position + 1)
                .unwrap_or('\0');
            if next_character.is_ascii_lowercase() {
                let symbol = format!("{current_character}{next_character}");
                if chemistry_consts::atomic_numbers(&symbol).is_some() {
                    self.advance_position();
                    self.advance_position();
                    return Ok(AtomExpression::Element(symbol));
                }
            }
        }

        match current_character {
            '#' => {
                self.advance_position();
                let atomic_number = self.parse_number()? as u8;
                Ok(AtomExpression::AtomicNumber(atomic_number))
            }
            'D' => {
                self.advance_position();
                let degree = self.parse_number()? as u8;
                Ok(AtomExpression::Degree(degree))
            }
            'H' => {
                self.advance_position();
                let hydrogen_count =
                    if self.position < self.input.len() && self.current_char()?.is_ascii_digit() {
                        self.parse_number()? as u8
                    } else {
                        1
                    };
                Ok(AtomExpression::TotalHydrogens(hydrogen_count))
            }
            'h' => {
                self.advance_position();
                let hydrogen_count =
                    if self.position < self.input.len() && self.current_char()?.is_ascii_digit() {
                        self.parse_number()? as u8
                    } else {
                        1
                    };
                Ok(AtomExpression::ImplicitHydrogens(hydrogen_count))
            }
            'R' => {
                self.advance_position();
                // `R` on its own asks whether the atom is in *any* ring;
                // `R<n>` asks for exactly n. Treating bare `R` as `R1` — which
                // is what defaulting the count to 1 did — excludes every atom
                // shared between two rings, so `[R]` missed naphthalene's two
                // fusion carbons and matched nothing at all in
                // bicyclo[2.2.2]octane, where no atom is in only one ring.
                if self.position < self.input.len() && self.current_char()?.is_ascii_digit() {
                    Ok(AtomExpression::RingMembership(self.parse_number()? as u8))
                } else {
                    Ok(AtomExpression::InRing)
                }
            }
            'r' => {
                self.advance_position();
                if self.position < self.input.len() && self.current_char()? == '{' {
                    return self.parse_ring_size_range();
                }
                // `r<n>` asks for a smallest ring of exactly n; bare `r` asks
                // only whether the atom is in a ring at all, the same
                // question as bare `R`. Requiring a digit rejected patterns
                // like `[NX3&r]` and `[CX3!r]`, which are ordinary SMARTS.
                if self.position < self.input.len() && self.current_char()?.is_ascii_digit() {
                    Ok(AtomExpression::RingSize(self.parse_number()? as u8))
                } else {
                    Ok(AtomExpression::InRing)
                }
            }
            'v' => {
                self.advance_position();
                let valence = self.parse_number()? as u8;
                Ok(AtomExpression::Valence(valence))
            }
            'X' => {
                self.advance_position();
                let connectivity = self.parse_number()? as u8;
                Ok(AtomExpression::Connectivity(connectivity))
            }
            '^' => {
                self.advance_position();
                let hybridization = self.parse_number()? as u8;
                Ok(AtomExpression::Hybridization(hybridization))
            }
            '+' => {
                self.advance_position();
                let charge =
                    if self.position < self.input.len() && self.current_char()?.is_ascii_digit() {
                        self.parse_number()? as i8
                    } else {
                        1
                    };
                Ok(AtomExpression::Charge(charge))
            }
            '-' => {
                self.advance_position();
                let charge =
                    if self.position < self.input.len() && self.current_char()?.is_ascii_digit() {
                        -(self.parse_number()? as i8)
                    } else {
                        -1
                    };
                Ok(AtomExpression::Charge(charge))
            }
            '@' => self.parse_chirality(),
            '0'..='9' => {
                // Isotope specification at the beginning
                let isotope = self.parse_number()? as u16;
                Ok(AtomExpression::Isotope(isotope))
            }
            // `a` and `A` inside brackets are the aromatic and aliphatic
            // wildcards, not one-letter elements. Reading `[a]` as an element
            // symbol makes it match nothing at all, and it is how half the
            // published key sets say "aromatic". A two-letter element starting
            // with the same letter has already been taken above, so anything
            // reaching here is the wildcard -- except an aromatic two-letter
            // symbol, which is lowercase throughout.
            'a' if !self
                .input
                .chars()
                .nth(self.position + 1)
                .is_some_and(|next| next.is_ascii_lowercase()) =>
            {
                self.advance_position();
                Ok(AtomExpression::Aromatic)
            }
            'A' => {
                self.advance_position();
                Ok(AtomExpression::Aliphatic)
            }
            'A'..='Z' => {
                // A two-letter symbol beginning with an uppercase letter was
                // taken above, so this is a one-letter element. Swallowing the
                // next lowercase character regardless made `[Cv4]` a carbon of
                // the element "Cv" rather than a carbon of valence four.
                self.advance_position();
                Ok(AtomExpression::Element(current_character.to_string()))
            }
            'a'..='z' => {
                // Aromatic symbols are lowercase throughout, so the two-letter
                // ones -- `as`, `se`, `te`, `si` -- have to be recognised here.
                let next_character = self.input.chars().nth(self.position + 1).unwrap_or('\0');
                if next_character.is_ascii_lowercase() {
                    let symbol = format!("{current_character}{next_character}");
                    if chemistry_consts::atomic_numbers(&symbol).is_some() {
                        self.advance_position();
                        self.advance_position();
                        return Ok(AtomExpression::Element(symbol));
                    }
                }
                self.advance_position();
                Ok(AtomExpression::Element(current_character.to_string()))
            }
            '*' => {
                self.advance_position();
                Ok(AtomExpression::Wildcard)
            }
            '$' => self.parse_recursive_smarts(),
            _ => Err(CxSMARTSError::InvalidAtomProperty(format!(
                "Unknown property: {}",
                current_character
            ))),
        }
    }

    /// Parse chirality specification including extended ChemAxon chirality
    fn parse_chirality(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        self.advance_position(); // Skip '@'

        // Check for extended chirality like @TH, @AL, etc.
        if self.position < self.input.len() {
            let current_character = self.current_char()?;

            match current_character {
                '@' => {
                    // Simple @@ clockwise
                    self.advance_position();
                    Ok(AtomExpression::Chirality(ChiralClass::Clockwise))
                }
                'T' => {
                    // @TH (tetrahedral)
                    self.advance_position();
                    if self.position < self.input.len() && self.current_char()? == 'H' {
                        self.advance_position();
                        let chiral_number = if self.position < self.input.len()
                            && self.current_char()?.is_ascii_digit()
                        {
                            self.parse_number()? as u8
                        } else {
                            1
                        };
                        Ok(AtomExpression::Chirality(ChiralClass::TH(chiral_number)))
                    } else {
                        Err(CxSMARTSError::InvalidChirality(
                            "Expected 'H' after '@T'".to_string(),
                        ))
                    }
                }
                'A' => {
                    // @AL (allene)
                    self.advance_position();
                    if self.position < self.input.len() && self.current_char()? == 'L' {
                        self.advance_position();
                        let chiral_number = if self.position < self.input.len()
                            && self.current_char()?.is_ascii_digit()
                        {
                            self.parse_number()? as u8
                        } else {
                            1
                        };
                        Ok(AtomExpression::Chirality(ChiralClass::AL(chiral_number)))
                    } else {
                        Err(CxSMARTSError::InvalidChirality(
                            "Expected 'L' after '@A'".to_string(),
                        ))
                    }
                }
                'S' => {
                    // @SP (square planar)
                    self.advance_position();
                    if self.position < self.input.len() && self.current_char()? == 'P' {
                        self.advance_position();
                        let chiral_number = if self.position < self.input.len()
                            && self.current_char()?.is_ascii_digit()
                        {
                            self.parse_number()? as u8
                        } else {
                            1
                        };
                        Ok(AtomExpression::Chirality(ChiralClass::SP(chiral_number)))
                    } else {
                        Err(CxSMARTSError::InvalidChirality(
                            "Expected 'P' after '@S'".to_string(),
                        ))
                    }
                }
                'O' => {
                    // @OH (octahedral)
                    self.advance_position();
                    if self.position < self.input.len() && self.current_char()? == 'H' {
                        self.advance_position();
                        let chiral_number = if self.position < self.input.len()
                            && self.current_char()?.is_ascii_digit()
                        {
                            self.parse_number()? as u8
                        } else {
                            1
                        };
                        Ok(AtomExpression::Chirality(ChiralClass::OH(chiral_number)))
                    } else {
                        Err(CxSMARTSError::InvalidChirality(
                            "Expected 'H' after '@O'".to_string(),
                        ))
                    }
                }
                _ => {
                    // Simple @ counterclockwise
                    Ok(AtomExpression::Chirality(ChiralClass::Counterclockwise))
                }
            }
        } else {
            Ok(AtomExpression::Chirality(ChiralClass::Counterclockwise))
        }
    }

    /// Parse recursive SMARTS pattern
    fn parse_recursive_smarts(&mut self) -> Result<AtomExpression, CxSMARTSError> {
        self.advance_position(); // Skip '$'

        if self.current_char()? != '(' {
            return Err(CxSMARTSError::InvalidRecursivePattern(
                "Expected '(' after '$'".to_string(),
            ));
        }

        self.advance_position(); // Skip '('

        let mut pattern = String::new();
        let mut parenthesis_count = 1;

        while self.position < self.input.len() && parenthesis_count > 0 {
            let current_character = self.current_char()?;
            if current_character == '(' {
                parenthesis_count += 1;
            } else if current_character == ')' {
                parenthesis_count -= 1;
            }

            if parenthesis_count > 0 {
                pattern.push(current_character);
            }

            self.advance_position();
        }

        if parenthesis_count != 0 {
            return Err(CxSMARTSError::MismatchedBrackets);
        }

        Ok(AtomExpression::Recursive(pattern))
    }

    /// `(` — a branch when it follows an atom, a component group when it does
    /// not.
    fn parse_branch_open(&mut self) -> Result<(), CxSMARTSError> {
        self.advance_position();
        self.branch_stack
            .push((self.previous_atom, self.current_atom_index));
        Ok(())
    }

    /// `)` — restore the chain head to the branch's parent so the next atom
    /// bonds there, or close off a component group.
    fn parse_branch_close(&mut self) -> Result<(), CxSMARTSError> {
        self.advance_position();
        let Some((parent, start_index)) = self.branch_stack.pop() else {
            return Err(CxSMARTSError::InvalidSyntax("unmatched ')'".to_string()));
        };
        match parent {
            Some(atom) => self.previous_atom = Some(atom),
            None => {
                let group: Vec<usize> = (start_index..self.current_atom_index).collect();
                if !group.is_empty() {
                    self.components.push(group);
                }
                self.previous_atom = None;
            }
        }
        Ok(())
    }

    /// Bonds a freshly parsed atom to the one before it and makes it the new
    /// chain head.
    ///
    /// This is where an *implicit* bond comes from. Without it a pattern's
    /// atoms are parsed but never joined, so `CC` describes two unconnected
    /// carbons rather than a C-C bond, and a molecule does not match itself.
    fn attach_atom(&mut self, atom_index: usize) {
        self.neighbor_order.push(Vec::new());
        self.incoming.push(self.previous_atom);
        if let Some(previous) = self.previous_atom {
            let bond = self
                .pending_bond
                .take()
                .unwrap_or(BondExpression::SingleOrAromatic);
            self.bonds.push((previous, atom_index, bond));
            self.neighbor_order[previous].push(atom_index);
            self.neighbor_order[atom_index].push(previous);
        } else {
            // No atom to bond back to; a bond symbol here has nothing to apply
            // to and is dropped rather than mis-attached.
            self.pending_bond = None;
        }
        self.previous_atom = Some(atom_index);
    }

    /// Parse disconnection (.)
    ///
    /// Breaks the chain: the next atom starts a new component and must not be
    /// bonded to the previous one.
    fn parse_disconnection(&mut self) -> Result<(), CxSMARTSError> {
        self.advance_position(); // Skip '.'
        self.previous_atom = None;
        self.pending_bond = None;
        Ok(())
    }

    /// Parse ring closure
    fn parse_ring_closure(&mut self) -> Result<(), CxSMARTSError> {
        let ring_number = if self.current_char()? == '%' {
            self.advance_position(); // Skip '%'
            self.parse_number()?
        } else {
            let current_digit = self.current_char()?.to_digit(10).ok_or_else(|| {
                CxSMARTSError::InvalidRingClosure("Invalid ring number".to_string())
            })?;
            self.advance_position();
            current_digit as usize
        };

        let Some(atom) = self.previous_atom else {
            return Err(CxSMARTSError::InvalidRingClosure(
                "ring closure before any atom".to_string(),
            ));
        };
        self.ring_closures
            .entry(ring_number)
            .or_default()
            .push(atom);

        // The second occurrence of a digit closes the ring, and that is the
        // only place the ring bond can be created — the two atoms are not
        // adjacent in the string, so nothing else would ever join them.
        match self.open_ring_closures.remove(&ring_number) {
            Some((opening_atom, opening_bond, slot)) => {
                self.neighbor_order[opening_atom][slot] = atom;
                self.neighbor_order[atom].push(opening_atom);
                let bond = self
                    .pending_bond
                    .take()
                    .map(super::smarts_stereo::reverse_bond)
                    .or(opening_bond)
                    .unwrap_or(BondExpression::SingleOrAromatic);
                self.bonds.push((opening_atom, atom, bond));
            }
            None => {
                let slot = self.neighbor_order[atom].len();
                self.neighbor_order[atom].push(usize::MAX);
                self.open_ring_closures
                    .insert(ring_number, (atom, self.pending_bond.take(), slot));
            }
        }

        Ok(())
    }

    /// Parse bond expression with logical operators
    /// Reads a bond symbol and holds it until the thing it connects to — the
    /// next atom, or a ring-closure digit — has been parsed.
    ///
    /// Attaching it immediately as `(current - 1, current)` was wrong wherever
    /// the two atoms are not consecutive: after a branch closes, and for every
    /// ring closure.
    fn parse_bond(&mut self) -> Result<(), CxSMARTSError> {
        if self.previous_atom.is_none() {
            return Err(CxSMARTSError::InvalidBond(
                "bond symbol before any atom".to_string(),
            ));
        }
        self.pending_bond = Some(self.parse_bond_expression()?);
        Ok(())
    }

    /// Parse a bond expression, with the same three precedence levels an atom
    /// expression has: `;` binds loosest, then `,`, then `&`, and `!` binds
    /// tightest of all. `=;@` -- a double bond that is also a ring bond -- is
    /// the form MACCS key 26 is written in, and it is not exotic.
    fn parse_bond_expression(&mut self) -> Result<BondExpression, CxSMARTSError> {
        let mut operands = vec![self.parse_bond_or()?];
        while self.position < self.input.len() && self.current_char()? == ';' {
            self.advance_position();
            operands.push(self.parse_bond_or()?);
        }
        Ok(Self::joined_bond(LogicalOperator::LowAnd, operands))
    }

    fn parse_bond_or(&mut self) -> Result<BondExpression, CxSMARTSError> {
        let mut operands = vec![self.parse_bond_and()?];
        while self.position < self.input.len() && self.current_char()? == ',' {
            self.advance_position();
            operands.push(self.parse_bond_and()?);
        }
        Ok(Self::joined_bond(LogicalOperator::Or, operands))
    }

    fn parse_bond_and(&mut self) -> Result<BondExpression, CxSMARTSError> {
        let mut operands = vec![self.parse_bond_negation()?];
        while self.position < self.input.len() {
            // Two bond primitives written next to each other are an and, the
            // `&` being optional exactly as it is between atom primitives.
            // `=@` is a double bond that closes a ring; reading it as a double
            // bond and then losing the `@` makes it any double bond.
            let current = self.current_char()?;
            if current == '&' {
                self.advance_position();
            } else if !Self::starts_bond_expression(current) {
                break;
            }
            operands.push(self.parse_bond_negation()?);
        }
        Ok(Self::joined_bond(LogicalOperator::HighAnd, operands))
    }

    /// Whether a character can begin a bond primitive. Nothing that can begin
    /// an atom is in this set, so a bond expression knows where it ends.
    fn starts_bond_expression(character: char) -> bool {
        matches!(character, '-' | '=' | '#' | ':' | '~' | '@' | '/' | '\\' | '!')
    }

    fn parse_bond_negation(&mut self) -> Result<BondExpression, CxSMARTSError> {
        if self.position < self.input.len() && self.current_char()? == '!' {
            self.advance_position();
            return Ok(BondExpression::Not(Box::new(self.parse_bond_negation()?)));
        }
        self.parse_primary_bond_expression()
    }

    /// One operand stays as it is; several are joined by the operator.
    fn joined_bond(operator: LogicalOperator, mut operands: Vec<BondExpression>) -> BondExpression {
        if operands.len() == 1 {
            return operands.pop().expect("just checked there is one");
        }
        BondExpression::Logical { operator, operands }
    }

    /// Parse primary bond expressions (single bond types)
    fn parse_primary_bond_expression(&mut self) -> Result<BondExpression, CxSMARTSError> {
        let current_character = self.current_char()?;
        self.advance_position();

        let bond_expression = match current_character {
            '-' => BondExpression::Single,
            '=' => BondExpression::Double,
            '#' => BondExpression::Triple,
            ':' => BondExpression::Aromatic,
            '~' => BondExpression::Any,
            '@' => BondExpression::Ring,
            '/' => {
                // Check for /? (directional up or unspecified)
                if self.position < self.input.len() && self.current_char()? == '?' {
                    self.advance_position();
                    BondExpression::DirectionalUpOrUnspec
                } else {
                    BondExpression::DirectionalUp
                }
            }
            '\\' => {
                // Check for \? (directional down or unspecified)
                if self.position < self.input.len() && self.current_char()? == '?' {
                    self.advance_position();
                    BondExpression::DirectionalDownOrUnspec
                } else {
                    BondExpression::DirectionalDown
                }
            }
            _ => {
                return Err(CxSMARTSError::InvalidBond(format!(
                    "Unknown bond: {}",
                    current_character
                )));
            }
        };

        Ok(bond_expression)
    }

    /// Parse a number from the current position
    fn parse_number(&mut self) -> Result<usize, CxSMARTSError> {
        let mut number = 0;
        let mut has_digits = false;

        while self.position < self.input.len() {
            let current_character = self.current_char()?;
            if current_character.is_ascii_digit() {
                number = number * 10 + (current_character as usize - '0' as usize);
                has_digits = true;
                self.advance_position();
            } else {
                break;
            }
        }

        if has_digits {
            Ok(number)
        } else {
            Err(CxSMARTSError::InvalidSyntax("Expected number".to_string()))
        }
    }

    /// Get current character
    fn current_char(&self) -> Result<char, CxSMARTSError> {
        self.input
            .chars()
            .nth(self.position)
            .ok_or(CxSMARTSError::UnexpectedEndOfInput)
    }

    /// Steps past the current character.
    ///
    /// Every caller has already established that the character it is stepping
    /// past exists, so running off the end is not an error to report but a
    /// position to leave alone.
    fn advance_position(&mut self) {
        if self.position < self.input.len() {
            self.position += 1;
        }
    }
}

/// Convert a CxSMARTS pattern to a molecule for matching
impl CxSMARTSPattern {
    /// Check if this pattern matches the given molecule
    pub fn matches(&self, molecule: &MolecularGraph) -> bool {
        use crate::io::subgraph_matching::smarts_matches;

        // Handle reaction SMARTS
        if self.is_reaction {
            // For reaction SMARTS, check if reactants and products match
            if let Some(ref reactants) = self.reactants {
                if !reactants.matches(molecule) {
                    return false;
                }
            }
            if let Some(ref products) = self.products {
                if !products.matches(molecule) {
                    return false;
                }
            }
            return true;
        }

        // Regular pattern matching using subgraph isomorphism
        smarts_matches(self, molecule)
    }

    /// Get all matches of this pattern in the given molecule
    pub fn find_matches(&self, molecule: &MolecularGraph) -> Vec<Vec<usize>> {
        use crate::io::subgraph_matching::find_smarts_matches;

        // Handle reaction SMARTS
        if self.is_reaction {
            // For reaction SMARTS, we would need special handling
            // For now, just return empty if it's a reaction pattern
            return vec![];
        }

        // Regular pattern matching using subgraph isomorphism
        let matches = find_smarts_matches(self, molecule);

        // Convert HashMap<usize, usize> matches to Vec<usize> for backward compatibility
        matches
            .into_iter()
            .map(|match_map| {
                let mut match_vec: Vec<usize> = match_map.values().copied().collect();
                match_vec.sort(); // Sort for consistent ordering
                match_vec
            })
            .collect()
    }
}

/// Public API for parsing CxSMARTS
/// [`CxSMARTSPattern::matches`] reusing an analysis of the molecule.
impl CxSMARTSPattern {
    pub fn matches_with(
        &self,
        molecule: &MolecularGraph,
        context: &crate::io::MoleculeMatchContext,
    ) -> bool {
        crate::io::subgraph_matching::smarts_matches_with(self, molecule, context)
    }

    /// [`CxSMARTSPattern::find_matches`] reusing an analysis of the molecule.
    pub fn find_matches_with(
        &self,
        molecule: &MolecularGraph,
        context: &crate::io::MoleculeMatchContext,
    ) -> Vec<Vec<usize>> {
        if self.is_reaction {
            return vec![];
        }
        crate::io::subgraph_matching::find_smarts_matches_with(self, molecule, context)
            .into_iter()
            .map(|match_map| {
                let mut match_vec: Vec<usize> = match_map.values().copied().collect();
                match_vec.sort();
                match_vec
            })
            .collect()
    }
}

pub fn parse_cxsmarts(smarts_pattern: &str) -> Result<CxSMARTSPattern, CxSMARTSError> {
    let mut parser = CxSMARTSParser::new();
    parser.parse(smarts_pattern)
}

/// Check if a molecule matches a CxSMARTS pattern
pub fn matches_cxsmarts(
    molecule: &MolecularGraph,
    smarts_pattern: &str,
) -> Result<bool, CxSMARTSError> {
    let pattern = parse_cxsmarts(smarts_pattern)?;
    Ok(pattern.matches(molecule))
}

/// Find all matches of a CxSMARTS pattern in a molecule
pub fn find_cxsmarts_matches(
    molecule: &MolecularGraph,
    smarts_pattern: &str,
) -> Result<Vec<Vec<usize>>, CxSMARTSError> {
    let pattern = parse_cxsmarts(smarts_pattern)?;
    Ok(pattern.find_matches(molecule))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::SMILESParser;

    #[test]
    fn test_parse_simple_cxsmarts() {
        let parsed_pattern = parse_cxsmarts("C").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        assert!(!parsed_pattern.is_reaction);
    }

    #[test]
    fn test_parse_atom_class() {
        let parsed_pattern = parse_cxsmarts("[C:1]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        assert!(parsed_pattern.atom_classes.contains_key(&0));
        assert_eq!(parsed_pattern.atom_classes[&0], 1);
    }

    #[test]
    fn test_parse_fragment_connection() {
        let parsed_pattern = parse_cxsmarts("C1.C1").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 2);
        assert!(parsed_pattern.ring_closures.contains_key(&1));
    }

    #[test]
    fn test_parse_reaction_smarts() {
        let parsed_pattern = parse_cxsmarts("[C:1]>>[O:1]").unwrap();
        assert!(parsed_pattern.is_reaction);
        assert!(parsed_pattern.reactants.is_some());
        assert!(parsed_pattern.products.is_some());
    }

    #[test]
    fn test_parse_recursive_smarts() {
        let parsed_pattern = parse_cxsmarts("[$([CH3])]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        match &parsed_pattern.atoms[0] {
            AtomExpression::Recursive(recursive_pattern) => {
                assert_eq!(recursive_pattern, "[CH3]");
            }
            _ => panic!("Expected recursive pattern"),
        }
    }

    #[test]
    fn test_matches_simple_molecule() {
        let molecules = SMILESParser::parse_smiles("CCO").unwrap();
        let molecule = &molecules[0];

        assert!(matches_cxsmarts(molecule, "C").unwrap());
        assert!(matches_cxsmarts(molecule, "O").unwrap());
        assert!(matches_cxsmarts(molecule, "*").unwrap());
    }

    #[test]
    fn test_component_grouping() {
        let parsed_pattern = parse_cxsmarts("(C.C)").unwrap();
        assert_eq!(parsed_pattern.components.len(), 1);
        assert_eq!(parsed_pattern.components[0].len(), 2);
    }

    #[test]
    fn test_logical_operators() {
        // Test NOT operator
        let parsed_pattern = parse_cxsmarts("[!C]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        match &parsed_pattern.atoms[0] {
            AtomExpression::Logical { operator, operands } => {
                assert_eq!(*operator, LogicalOperator::Not);
                assert_eq!(operands.len(), 1);
            }
            _ => panic!("Expected logical NOT expression"),
        }

        // Test OR operator
        let parsed_pattern = parse_cxsmarts("[C,N]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        match &parsed_pattern.atoms[0] {
            AtomExpression::Logical { operator, operands } => {
                assert_eq!(*operator, LogicalOperator::Or);
                assert_eq!(operands.len(), 2);
            }
            _ => panic!("Expected logical OR expression"),
        }

        // Test high precedence AND operator
        let parsed_pattern = parse_cxsmarts("[C&H3]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        match &parsed_pattern.atoms[0] {
            AtomExpression::Logical { operator, operands } => {
                assert_eq!(*operator, LogicalOperator::HighAnd);
                assert_eq!(operands.len(), 2);
            }
            _ => panic!("Expected logical high AND expression"),
        }

        // Test low precedence AND operator
        let parsed_pattern = parse_cxsmarts("[C,N;H1]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        match &parsed_pattern.atoms[0] {
            AtomExpression::Logical { operator, operands } => {
                assert_eq!(*operator, LogicalOperator::LowAnd);
                assert_eq!(operands.len(), 2);
            }
            _ => panic!("Expected logical low AND expression"),
        }
    }

    #[test]
    fn test_extended_chirality() {
        // Test tetrahedral chirality
        let parsed_pattern = parse_cxsmarts("[@TH1]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        match &parsed_pattern.atoms[0] {
            AtomExpression::Chirality(ChiralClass::TH(n)) => {
                assert_eq!(*n, 1);
            }
            _ => panic!("Expected TH chirality"),
        }

        // Test allene chirality
        let parsed_pattern = parse_cxsmarts("[@AL2]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        match &parsed_pattern.atoms[0] {
            AtomExpression::Chirality(ChiralClass::AL(n)) => {
                assert_eq!(*n, 2);
            }
            _ => panic!("Expected AL chirality"),
        }

        // Test square planar chirality
        let parsed_pattern = parse_cxsmarts("[@SP1]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        match &parsed_pattern.atoms[0] {
            AtomExpression::Chirality(ChiralClass::SP(n)) => {
                assert_eq!(*n, 1);
            }
            _ => panic!("Expected SP chirality"),
        }
    }

    #[test]
    fn test_bond_logical_operators() {
        // Test NOT bond operator
        let parsed_pattern = parse_cxsmarts("C!@C").unwrap();
        assert_eq!(parsed_pattern.bonds.len(), 1);
        match &parsed_pattern.bonds[0].2 {
            BondExpression::Not(bond_expr) => {
                match bond_expr.as_ref() {
                    BondExpression::Ring => (), // Expected
                    _ => panic!("Expected NOT ring bond"),
                }
            }
            _ => panic!("Expected NOT bond expression"),
        }

        // Test directional bonds with unspecified
        let parsed_pattern = parse_cxsmarts("C/?C").unwrap();
        assert_eq!(parsed_pattern.bonds.len(), 1);
        match &parsed_pattern.bonds[0].2 {
            BondExpression::DirectionalUpOrUnspec => (), // Expected
            _ => panic!("Expected directional up or unspec bond"),
        }
    }

    #[test]
    fn test_complex_logical_expressions() {
        // Test complex expression with multiple operators
        let parsed_pattern = parse_cxsmarts("[!C&H2,N]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);

        // Test precedence: [C,N;H1] should be ((C OR N) AND H1)
        let parsed_pattern = parse_cxsmarts("[C,N;H1]").unwrap();
        assert_eq!(parsed_pattern.atoms.len(), 1);
        match &parsed_pattern.atoms[0] {
            AtomExpression::Logical { operator, operands } => {
                assert_eq!(*operator, LogicalOperator::LowAnd);
                assert_eq!(operands.len(), 2);
                // First operand should be C,N (OR expression)
                match &operands[0] {
                    AtomExpression::Logical { operator, .. } => {
                        assert_eq!(*operator, LogicalOperator::Or);
                    }
                    _ => panic!("Expected OR expression as first operand"),
                }
            }
            _ => panic!("Expected logical expression"),
        }
    }

    #[test]
    fn test_invalid_syntax() {
        assert!(parse_cxsmarts("[C:").is_err());
        assert!(parse_cxsmarts("$(C").is_err());
        assert!(parse_cxsmarts("#").is_err());
        assert!(parse_cxsmarts("[@TZ]").is_err()); // Invalid chirality
        assert!(parse_cxsmarts("[C&]").is_err()); // Incomplete AND
    }

    /// Comprehensive correctness tests for CxSMARTS pattern matching
    /// These tests ensure that the basic functionality works correctly before optimizing
    mod correctness_tests {
        use super::*;

        fn create_test_molecule(smiles: &str) -> MolecularGraph {
            SMILESParser::parse_smiles(smiles)
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
        }

        #[test]
        fn test_basic_element_matching() {
            let ethanol = create_test_molecule("CCO");

            // Test carbon matching
            assert!(
                matches_cxsmarts(&ethanol, "C").unwrap(),
                "Should match carbon in CCO"
            );

            // Test oxygen matching
            assert!(
                matches_cxsmarts(&ethanol, "O").unwrap(),
                "Should match oxygen in CCO"
            );

            // Test nitrogen (should not match)
            assert!(
                !matches_cxsmarts(&ethanol, "N").unwrap(),
                "Should not match nitrogen in CCO"
            );

            // Test wildcard
            assert!(
                matches_cxsmarts(&ethanol, "*").unwrap(),
                "Wildcard should match any atom"
            );
        }

        #[test]
        fn test_aromatic_vs_aliphatic() {
            let benzene = create_test_molecule("c1ccccc1");
            let cyclohexane = create_test_molecule("C1CCCCC1");

            // Test aromatic carbon
            assert!(
                matches_cxsmarts(&benzene, "c").unwrap(),
                "Should match aromatic carbon in benzene"
            );
            assert!(
                !matches_cxsmarts(&cyclohexane, "c").unwrap(),
                "Should not match aromatic carbon in cyclohexane"
            );

            // Test aliphatic carbon
            assert!(
                !matches_cxsmarts(&benzene, "C").unwrap(),
                "Should not match aliphatic carbon in benzene"
            );
            assert!(
                matches_cxsmarts(&cyclohexane, "C").unwrap(),
                "Should match aliphatic carbon in cyclohexane"
            );

            // Test 'a' (any aromatic atom)
            assert!(
                matches_cxsmarts(&benzene, "a").unwrap(),
                "Should match any aromatic atom in benzene"
            );
            assert!(
                !matches_cxsmarts(&cyclohexane, "a").unwrap(),
                "Should not match aromatic atom in cyclohexane"
            );

            // Test 'A' (any aliphatic atom) - Note: this will match aliphatic hydrogens in benzene
            // Since we can't easily separate carbons from hydrogens, let's test the basic difference
            // Benzene should have aromatic atoms, cyclohexane should have aliphatic atoms
            assert!(
                matches_cxsmarts(&benzene, "a").unwrap(),
                "Benzene should have aromatic atoms"
            );
            assert!(
                !matches_cxsmarts(&cyclohexane, "a").unwrap(),
                "Cyclohexane should not have aromatic atoms"
            );
        }

        #[test]
        fn test_logical_operators_correctness() {
            let ethanol = create_test_molecule("CCO");
            let pyridine = create_test_molecule("c1ccncc1");

            // Test OR operator
            assert!(
                matches_cxsmarts(&ethanol, "[C,O]").unwrap(),
                "Should match C OR O in ethanol"
            );
            // Uppercase C/N are aliphatic primitives, so they must NOT match
            // the aromatic atoms of pyridine; the aromatic forms must.
            assert!(
                !matches_cxsmarts(&pyridine, "[C,N]").unwrap(),
                "Aliphatic C OR N must not match aromatic pyridine"
            );
            assert!(
                matches_cxsmarts(&pyridine, "[c,n]").unwrap(),
                "Should match aromatic c OR n in pyridine"
            );
            assert!(
                !matches_cxsmarts(&ethanol, "[N,S]").unwrap(),
                "Should not match N OR S in ethanol"
            );

            // Test NOT operator
            assert!(
                matches_cxsmarts(&ethanol, "[!N]").unwrap(),
                "Should match NOT N in ethanol"
            );
            // [!N] matches any atom that is not an aliphatic nitrogen, so it
            // DOES match pyridine (via its carbons) even though pyridine
            // contains nitrogen. SMARTS negation is per-atom, not per-molecule.
            assert!(
                matches_cxsmarts(&pyridine, "[!N]").unwrap(),
                "NOT N should match the non-nitrogen atoms of pyridine"
            );
            assert!(
                matches_cxsmarts(&ethanol, "[!S]").unwrap(),
                "Should match NOT S in ethanol"
            );

            // Test AND operator (implicit)
            assert!(
                matches_cxsmarts(&ethanol, "[CH3]").unwrap(),
                "Should match C with 3 hydrogens in ethanol"
            );
            assert!(
                matches_cxsmarts(&ethanol, "[CH2]").unwrap(),
                "Should match C with 2 hydrogens in ethanol"
            );
            assert!(
                !matches_cxsmarts(&ethanol, "[CH4]").unwrap(),
                "Should not match C with 4 hydrogens in ethanol"
            );
        }

        #[test]
        fn test_hydrogen_count_patterns() {
            let methane = create_test_molecule("C");
            let ethanol = create_test_molecule("CCO");
            let water = create_test_molecule("O");

            // Test total hydrogen count (H)
            assert!(
                matches_cxsmarts(&methane, "[H4]").unwrap(),
                "Methane carbon should have 4 hydrogens"
            );
            assert!(
                matches_cxsmarts(&water, "[H2]").unwrap(),
                "Water oxygen should have 2 hydrogens"
            );
            assert!(
                matches_cxsmarts(&ethanol, "[H3]").unwrap(),
                "Ethanol should have carbon with 3 hydrogens"
            );
            assert!(
                matches_cxsmarts(&ethanol, "[H2]").unwrap(),
                "Ethanol should have carbon with 2 hydrogens"
            );
            assert!(
                matches_cxsmarts(&ethanol, "[H1]").unwrap(),
                "Ethanol should have oxygen with 1 hydrogen"
            );

            // Test implicit hydrogen count (h) - currently uses explicit H for testing
            assert!(
                matches_cxsmarts(&methane, "[h4]").unwrap(),
                "Methane carbon should have 4 implicit hydrogens"
            );
            assert!(
                matches_cxsmarts(&water, "[h2]").unwrap(),
                "Water oxygen should have 2 implicit hydrogens"
            );
        }

        #[test]
        fn test_bond_patterns() {
            let ethane = create_test_molecule("CC");
            let ethene = create_test_molecule("C=C");
            let ethyne = create_test_molecule("C#C");
            let benzene = create_test_molecule("c1ccccc1");

            // Test single bond
            assert!(
                matches_cxsmarts(&ethane, "C-C").unwrap(),
                "Should match single bond in ethane"
            );
            assert!(
                !matches_cxsmarts(&ethene, "C-C").unwrap(),
                "Should not match single bond in ethene"
            );

            // Test double bond
            assert!(
                matches_cxsmarts(&ethene, "C=C").unwrap(),
                "Should match double bond in ethene"
            );
            assert!(
                !matches_cxsmarts(&ethane, "C=C").unwrap(),
                "Should not match double bond in ethane"
            );

            // Test triple bond
            assert!(
                matches_cxsmarts(&ethyne, "C#C").unwrap(),
                "Should match triple bond in ethyne"
            );
            assert!(
                !matches_cxsmarts(&ethane, "C#C").unwrap(),
                "Should not match triple bond in ethane"
            );

            // Test aromatic bond
            assert!(
                matches_cxsmarts(&benzene, "c:c").unwrap(),
                "Should match aromatic bond in benzene"
            );
            assert!(
                !matches_cxsmarts(&ethane, "c:c").unwrap(),
                "Should not match aromatic bond in ethane"
            );

            // Test any bond wildcard
            assert!(
                matches_cxsmarts(&ethane, "C~C").unwrap(),
                "Should match any bond in ethane"
            );
            assert!(
                matches_cxsmarts(&ethene, "C~C").unwrap(),
                "Should match any bond in ethene"
            );
            assert!(
                matches_cxsmarts(&benzene, "c~c").unwrap(),
                "Should match any bond in benzene"
            );
        }

        #[test]
        fn test_degree_patterns() {
            let _methane = create_test_molecule("C"); // Carbon degree 0 (only H)
            let ethanol = create_test_molecule("CCO"); // Various degrees
            let tert_butyl = create_test_molecule("CC(C)(C)C"); // Central carbon degree 4

            // Test degree (D) - number of non-hydrogen neighbors
            assert!(
                matches_cxsmarts(&ethanol, "[D1]").unwrap(),
                "Should match degree 1 in ethanol"
            );
            assert!(
                matches_cxsmarts(&ethanol, "[D2]").unwrap(),
                "Should match degree 2 in ethanol (middle carbon)"
            );
            assert!(
                matches_cxsmarts(&tert_butyl, "[D4]").unwrap(),
                "Should match degree 4 in tert-butyl"
            );
            assert!(
                !matches_cxsmarts(&ethanol, "[D3]").unwrap(),
                "Should not match degree 3 in ethanol"
            );
        }

        #[test]
        fn test_connectivity_patterns() {
            let ethanol = create_test_molecule("CCO");
            let tert_butyl = create_test_molecule("CC(C)(C)C");

            // Test connectivity (X) - total number of neighbors including hydrogens
            assert!(
                matches_cxsmarts(&ethanol, "[X1]").unwrap(),
                "Should match connectivity 1 in ethanol"
            );
            assert!(
                matches_cxsmarts(&ethanol, "[X2]").unwrap(),
                "Should match connectivity 2 in ethanol"
            );
            assert!(
                matches_cxsmarts(&ethanol, "[X4]").unwrap(),
                "Should match connectivity 4 in ethanol"
            );
            assert!(
                matches_cxsmarts(&tert_butyl, "[X4]").unwrap(),
                "Should match connectivity 4 in tert-butyl"
            );
        }

        #[test]
        fn test_charge_patterns() {
            // Note: These tests depend on proper charge parsing in SMILES
            let cation = create_test_molecule("[NH4+]");
            let anion = create_test_molecule("[OH-]");
            let neutral = create_test_molecule("CO");

            // Test positive charge
            assert!(
                matches_cxsmarts(&cation, "[+1]").unwrap(),
                "Should match +1 charge"
            );
            assert!(
                matches_cxsmarts(&cation, "[+]").unwrap(),
                "Should match any positive charge"
            );

            // Test negative charge
            assert!(
                matches_cxsmarts(&anion, "[-1]").unwrap(),
                "Should match -1 charge"
            );
            assert!(
                matches_cxsmarts(&anion, "[-]").unwrap(),
                "Should match any negative charge"
            );

            // Test neutral (no charge)
            assert!(
                !matches_cxsmarts(&neutral, "[+]").unwrap(),
                "Should not match positive in neutral"
            );
            assert!(
                !matches_cxsmarts(&neutral, "[-]").unwrap(),
                "Should not match negative in neutral"
            );
        }

        #[test]
        fn test_atomic_number_patterns() {
            let ethanol = create_test_molecule("CCO");
            let water = create_test_molecule("O");

            // Test atomic number (#)
            assert!(
                matches_cxsmarts(&ethanol, "[#6]").unwrap(),
                "Should match carbon (atomic number 6)"
            );
            assert!(
                matches_cxsmarts(&ethanol, "[#8]").unwrap(),
                "Should match oxygen (atomic number 8)"
            );
            assert!(
                matches_cxsmarts(&water, "[#8]").unwrap(),
                "Should match oxygen in water"
            );
            assert!(
                !matches_cxsmarts(&ethanol, "[#7]").unwrap(),
                "Should not match nitrogen (atomic number 7)"
            );
        }

        #[test]
        fn test_ring_membership_patterns() {
            let benzene = create_test_molecule("c1ccccc1");
            let cyclohexane = create_test_molecule("C1CCCCC1");
            let ethanol = create_test_molecule("CCO");
            let naphthalene = create_test_molecule("c1ccc2ccccc2c1");

            // Test ring membership (R) - number of rings atom is in
            assert!(
                matches_cxsmarts(&benzene, "[R1]").unwrap(),
                "Benzene carbons should be in 1 ring"
            );
            assert!(
                matches_cxsmarts(&cyclohexane, "[R1]").unwrap(),
                "Cyclohexane carbons should be in 1 ring"
            );
            assert!(
                !matches_cxsmarts(&ethanol, "[R1]").unwrap(),
                "Ethanol should have no ring atoms"
            );
            assert!(
                matches_cxsmarts(&naphthalene, "[R2]").unwrap(),
                "Naphthalene bridgehead carbons should be in 2 rings"
            );

            // Test R0 (not in any ring)
            assert!(
                matches_cxsmarts(&ethanol, "[R0]").unwrap(),
                "Ethanol atoms should not be in rings"
            );
            // Hydrogens are explicit atoms in this representation and are not
            // in rings, so a bare [R0] matches benzene via its hydrogens.
            // Restrict to carbon to check the ring atoms themselves.
            assert!(
                !matches_cxsmarts(&benzene, "[#6;R0]").unwrap(),
                "Benzene carbons should all be in rings"
            );
        }

        #[test]
        fn test_ring_size_patterns() {
            let benzene = create_test_molecule("c1ccccc1"); // 6-membered ring
            let cyclopentane = create_test_molecule("C1CCCC1"); // 5-membered ring
            let cyclobutane = create_test_molecule("C1CCC1"); // 4-membered ring
            let ethanol = create_test_molecule("CCO"); // No rings

            // Test ring size (r) - size of smallest ring
            assert!(
                matches_cxsmarts(&benzene, "[r6]").unwrap(),
                "Benzene should have 6-membered ring"
            );
            assert!(
                matches_cxsmarts(&cyclopentane, "[r5]").unwrap(),
                "Cyclopentane should have 5-membered ring"
            );
            assert!(
                matches_cxsmarts(&cyclobutane, "[r4]").unwrap(),
                "Cyclobutane should have 4-membered ring"
            );
            assert!(
                !matches_cxsmarts(&ethanol, "[r6]").unwrap(),
                "Ethanol should not have rings"
            );
            assert!(
                !matches_cxsmarts(&benzene, "[r5]").unwrap(),
                "Benzene should not have 5-membered ring"
            );
        }

        #[test]
        fn test_complex_patterns() {
            let ethanol = create_test_molecule("CCO");
            let isopropanol = create_test_molecule("CC(C)O");
            let benzyl_alcohol = create_test_molecule("c1ccccc1CO");

            // Test methyl group
            assert!(
                matches_cxsmarts(&ethanol, "[CH3]").unwrap(),
                "Ethanol should have methyl group"
            );
            assert!(
                matches_cxsmarts(&isopropanol, "[CH3]").unwrap(),
                "Isopropanol should have methyl groups"
            );

            // Test methylene group
            assert!(
                matches_cxsmarts(&ethanol, "[CH2]").unwrap(),
                "Ethanol should have methylene group"
            );
            assert!(
                matches_cxsmarts(&benzyl_alcohol, "[CH2]").unwrap(),
                "Benzyl alcohol should have methylene"
            );

            // Test hydroxyl group
            assert!(
                matches_cxsmarts(&ethanol, "[OH1]").unwrap(),
                "Should match hydroxyl oxygen"
            );
            assert!(
                matches_cxsmarts(&isopropanol, "[OH1]").unwrap(),
                "Should match hydroxyl in isopropanol"
            );

            // Test aromatic CH
            assert!(
                matches_cxsmarts(&benzyl_alcohol, "[cH1]").unwrap(),
                "Should match aromatic CH in benzyl alcohol"
            );
        }

        #[test]
        fn test_bond_not_operator() {
            let ethanol = create_test_molecule("CCO");
            let benzene = create_test_molecule("c1ccccc1");
            let cyclopentane = create_test_molecule("C1CCCC1");

            // Test NOT ring bond (!@)
            assert!(
                matches_cxsmarts(&ethanol, "C!@C").unwrap(),
                "Ethanol C-C should not be ring bond"
            );
            assert!(
                matches_cxsmarts(&ethanol, "C!@O").unwrap(),
                "Ethanol C-O should not be ring bond"
            );
            assert!(
                !matches_cxsmarts(&benzene, "c!@c").unwrap(),
                "Benzene c-c should be ring bond"
            );
            assert!(
                !matches_cxsmarts(&cyclopentane, "C!@C").unwrap(),
                "Cyclopentane C-C should be ring bond"
            );
        }

        #[test]
        fn test_edge_cases() {
            let single_atom = create_test_molecule("C");
            let empty_pattern_result = parse_cxsmarts("");

            // Test single atom molecule
            assert!(
                matches_cxsmarts(&single_atom, "C").unwrap(),
                "Should match single carbon"
            );
            assert!(
                matches_cxsmarts(&single_atom, "*").unwrap(),
                "Should match single atom with wildcard"
            );
            assert!(
                !matches_cxsmarts(&single_atom, "CC").unwrap(),
                "Should not match two carbons"
            );

            // Test empty pattern should fail gracefully
            assert!(
                empty_pattern_result.is_err(),
                "Empty pattern should be invalid"
            );
        }

        #[test]
        fn test_find_matches_correctness() {
            let benzene = create_test_molecule("c1ccccc1");
            let toluene = create_test_molecule("Cc1ccccc1");

            // Test finding all carbon matches in benzene
            let carbon_matches = find_cxsmarts_matches(&benzene, "c").unwrap();
            assert_eq!(
                carbon_matches.len(),
                6,
                "Should find 6 aromatic carbons in benzene"
            );

            // Test finding all aromatic carbons in toluene
            let aromatic_matches = find_cxsmarts_matches(&toluene, "c").unwrap();
            assert_eq!(
                aromatic_matches.len(),
                6,
                "Should find 6 aromatic carbons in toluene"
            );

            // Test finding aliphatic carbons in toluene
            let aliphatic_matches = find_cxsmarts_matches(&toluene, "C").unwrap();
            assert_eq!(
                aliphatic_matches.len(),
                1,
                "Should find 1 aliphatic carbon in toluene"
            );
        }

        #[test]
        fn test_valence_patterns() {
            let methane = create_test_molecule("C");
            let ethene = create_test_molecule("C=C");
            let ethyne = create_test_molecule("C#C");

            // Test valence (v) - total bond order
            assert!(
                matches_cxsmarts(&methane, "[v4]").unwrap(),
                "Methane carbon should have valence 4"
            );
            assert!(
                matches_cxsmarts(&ethene, "[v4]").unwrap(),
                "Ethene carbons should have valence 4"
            );
            assert!(
                matches_cxsmarts(&ethyne, "[v4]").unwrap(),
                "Ethyne carbons should have valence 4"
            );
        }
    }
}
