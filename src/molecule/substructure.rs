use crate::io::{CxSMARTSError, CxSMARTSPattern, parse_cxsmarts};
use crate::molecule::{MolecularGraph, Molecule3D};

impl MolecularGraph {
    /// Whether this molecule contains a match for a SMARTS string.
    ///
    /// Uses the same CxSMARTS syntax and matching rules as
    /// [`crate::io::matches_cxsmarts`]. Extra bonds in the molecule are allowed:
    /// for example, a three-carbon chain matches inside cyclopropane.
    /// Each call parses the query; use [`Self::contains`] to reuse a pattern.
    ///
    /// # Errors
    /// Returns [`CxSMARTSError`] for an invalid query, including an empty string.
    /// A valid query with no match returns `Ok(false)`.
    ///
    /// ```
    /// use oxidane::molecule::MolecularGraph;
    ///
    /// let aspirin = MolecularGraph::from_smiles_single("CC(=O)Oc1ccccc1C(=O)O").unwrap();
    /// assert!(aspirin.contains_smarts("[CX3](=O)[OX2H1]")?);
    /// assert!(!aspirin.contains_smarts("N")?);
    /// assert!(aspirin.contains_smarts("[").is_err());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn contains_smarts(&self, smarts: &str) -> Result<bool, CxSMARTSError> {
        let pattern = parse_cxsmarts(smarts)?;
        Ok(self.contains(&pattern))
    }

    /// Whether this molecule contains a match for an already parsed pattern.
    ///
    /// Equivalent to [`CxSMARTSPattern::matches`], with the molecule as receiver.
    /// The pattern can be reused across molecules without parsing it again.
    /// For ordinary molecular queries, matching stops at the first match.
    ///
    /// ```
    /// use oxidane::io::parse_cxsmarts;
    /// use oxidane::molecule::MolecularGraph;
    ///
    /// let alcohol = parse_cxsmarts("[OX2H1][CX4]")?;
    /// let ethanol = MolecularGraph::from_smiles_single("CCO").unwrap();
    /// let acetone = MolecularGraph::from_smiles_single("CC(=O)C").unwrap();
    /// assert!(ethanol.contains(&alcohol));
    /// assert!(!acetone.contains(&alcohol));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn contains(&self, pattern: &CxSMARTSPattern) -> bool {
        pattern.matches(self)
    }
}

impl Molecule3D {
    /// Whether this molecule's stored graph contains a SMARTS match.
    ///
    /// Uses the stored bonds and stereochemical tags; coordinates are not used
    /// to infer new bonds or stereochemistry. Copies the topology through
    /// [`Self::to_graph`]. For many queries, convert once and query that graph.
    ///
    /// # Errors
    /// Returns [`CxSMARTSError`] for an invalid query; a valid query without a
    /// match returns `Ok(false)`. Parsing happens before the topology is copied.
    ///
    /// ```
    /// use oxidane::molecule::Molecule3D;
    ///
    /// let molecules = Molecule3D::from_sdf("tests/pentene.sdf").unwrap();
    /// assert!(molecules[0].contains_smarts("C=C")?);
    /// assert!(!molecules[0].contains_smarts("N")?);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn contains_smarts(&self, smarts: &str) -> Result<bool, CxSMARTSError> {
        let pattern = parse_cxsmarts(smarts)?;
        Ok(self.contains(&pattern))
    }

    /// Whether this molecule's stored graph matches a parsed SMARTS pattern.
    ///
    /// Copies the topology through [`Self::to_graph`], preserving atom metadata
    /// and stored stereo tags. Coordinates are ignored. For repeated queries,
    /// convert once and use [`MolecularGraph::contains`] on the resulting graph.
    pub fn contains(&self, pattern: &CxSMARTSPattern) -> bool {
        self.to_graph().contains(pattern)
    }
}
