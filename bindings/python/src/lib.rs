//! Migrated from um-univie/molecules-py, commit 0f1eaed5e592d00b131da387600146d733628262.
use oxidane::atom::Atom;
use oxidane::graph_algorithms::mol_canonization::{
    canonicalize_and_sort_smiles, canonize_mol, canonize_mol_inplace,
};
use oxidane::io::{CxSMARTSPattern, SDFParseError, SMILESParseError, ToSMILES, parse_cxsmarts};
use oxidane::molecule::base::MoleculeCore;
use oxidane::molecule::draw::Draw;
use oxidane::molecule::reaction::ReactionSmarts as RustReaction;
use oxidane::molecule::tautomer::{TautomerRules, normalized_smiles, tautomer_key_from_smiles};
use oxidane::molecule::{MolecularGraph, Molecule3D};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyString};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

fn value_error(error: impl std::fmt::Debug) -> PyErr {
    PyValueError::new_err(format!("{error:?}"))
}

fn svg(py: Python<'_>, raw: String, width: u32, height: u32) -> PyResult<String> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("width and height must be positive"));
    }
    py.import("oxidane._display")?
        .getattr("resize_svg")?
        .call1((raw, width, height))?
        .extract()
}

fn plot(py: Python<'_>, svg: String) -> PyResult<()> {
    py.import("oxidane._display")?
        .getattr("plot_svg")?
        .call1((svg,))?;
    Ok(())
}

fn contains(graph: &MolecularGraph, query: &Bound<'_, PyAny>) -> PyResult<bool> {
    if let Ok(pattern) = query.extract::<PyRef<'_, Smarts>>() {
        Ok(graph.contains(&pattern.inner))
    } else if let Ok(source) = query.extract::<String>() {
        graph.contains_smarts(&source).map_err(value_error)
    } else {
        Err(PyTypeError::new_err(
            "query must be a SMARTS string or Smarts object",
        ))
    }
}

#[pyclass(module = "oxidane")]
struct Smarts {
    inner: CxSMARTSPattern,
    source: String,
}

#[pymethods]
impl Smarts {
    #[new]
    fn new(smarts: &str) -> PyResult<Self> {
        Ok(Self {
            inner: parse_cxsmarts(smarts).map_err(value_error)?,
            source: smarts.to_owned(),
        })
    }
    fn __repr__(&self) -> String {
        format!("Smarts({:?})", self.source)
    }
}

/// The original PyMolecule API, also exported as oxidane.Molecule.
#[pyclass(module = "oxidane")]
struct PyMolecule {
    inner: MolecularGraph,
}

#[pymethods]
impl PyMolecule {
    #[new]
    fn new(smiles: &str) -> PyResult<Self> {
        // Keep every dot-separated component, including counterions.
        let molecules = MolecularGraph::from_smiles(smiles.trim()).map_err(value_error)?;
        if molecules.is_empty() {
            return Err(PyValueError::new_err("SMILES must contain a molecule"));
        }
        let references = molecules.iter().collect::<Vec<_>>();
        Ok(Self {
            inner: MolecularGraph::combine(&references),
        })
    }
    fn to_smiles(&self) -> String {
        self.inner.to_smiles()
    }
    fn canonize_mol(&self) -> Self {
        Self {
            inner: canonize_mol(&self.inner),
        }
    }
    fn canonize(&mut self) {
        canonize_mol_inplace(&mut self.inner);
    }
    /// The same string `oxidane.canonical_smiles` gives for this molecule's SMILES.
    fn to_canonical_smiles(&self) -> PyResult<String> {
        canonicalize_and_sort_smiles(&self.inner.to_smiles(), true).map_err(value_error)
    }
    /// The same string `oxidane.tautomer_key` gives for this molecule's SMILES.
    #[pyo3(signature = (*, carbon_endpoints=false, ionizable_endpoints=false))]
    fn tautomer_key(&self, carbon_endpoints: bool, ionizable_endpoints: bool) -> PyResult<String> {
        let rules = tautomer_rules(carbon_endpoints, ionizable_endpoints);
        tautomer_key_from_smiles(&self.inner.to_smiles(), &rules).map_err(value_error)
    }
    /// The same string `oxidane.canonical_tautomer` gives for this molecule's SMILES.
    #[pyo3(signature = (*, carbon_endpoints=false, ionizable_endpoints=false))]
    fn canonical_tautomer(
        &self,
        carbon_endpoints: bool,
        ionizable_endpoints: bool,
    ) -> PyResult<String> {
        let rules = tautomer_rules(carbon_endpoints, ionizable_endpoints);
        normalized_smiles(&self.inner.to_smiles(), &rules).map_err(value_error)
    }

    /// Invalid queries raise ValueError; an absent pattern returns False.
    fn contains_smarts(&self, smarts: &str) -> PyResult<bool> {
        self.inner.contains_smarts(smarts).map_err(value_error)
    }
    /// Accept a SMARTS string or reuse an already parsed Smarts object.
    fn contains(&self, query: &Bound<'_, PyAny>) -> PyResult<bool> {
        contains(&self.inner, query)
    }

    /// Export a 2D SVG without importing IPython or displaying anything.
    #[pyo3(signature = (*, width=400, height=300))]
    fn to_svg(&self, py: Python<'_>, width: u32, height: u32) -> PyResult<String> {
        svg(py, self.inner.to_svg(), width, height)
    }
    /// Display immediately, including inside loops. Returns None.
    #[pyo3(signature = (*, width=400, height=300))]
    fn plot(&self, py: Python<'_>, width: u32, height: u32) -> PyResult<()> {
        plot(py, self.to_svg(py, width, height)?)
    }
    fn _repr_svg_(&self, py: Python<'_>) -> PyResult<String> {
        self.to_svg(py, 400, 300)
    }
    fn __repr__(&self) -> String {
        format!("Molecule({:?})", self.to_smiles())
    }
}

#[pyclass(module = "oxidane")]
struct PyMolecule3D {
    inner: Molecule3D,
}

#[pymethods]
impl PyMolecule3D {
    fn canonize_mol(&self) -> Self {
        Self {
            inner: canonize_mol(&self.inner),
        }
    }
    fn canonize(&mut self) {
        canonize_mol_inplace(&mut self.inner);
    }
    fn to_smiles(&self) -> String {
        self.inner.to_smiles()
    }
    /// The same string `oxidane.canonical_smiles` gives for this molecule's SMILES.
    fn to_canonical_smiles(&self) -> PyResult<String> {
        canonicalize_and_sort_smiles(&self.inner.to_smiles(), true).map_err(value_error)
    }
    fn to_graph(&self) -> PyMolecule {
        PyMolecule {
            inner: self.inner.to_graph(),
        }
    }
    fn contains_smarts(&self, smarts: &str) -> PyResult<bool> {
        self.inner.contains_smarts(smarts).map_err(value_error)
    }
    fn contains(&self, query: &Bound<'_, PyAny>) -> PyResult<bool> {
        contains(&self.inner.to_graph(), query)
    }
    /// Draw the stored bonds as a 2D depiction; coordinates are not modified.
    #[pyo3(signature = (*, width=400, height=300))]
    fn to_svg(&self, py: Python<'_>, width: u32, height: u32) -> PyResult<String> {
        svg(py, self.inner.to_svg(), width, height)
    }
    #[pyo3(signature = (*, width=400, height=300))]
    fn plot(&self, py: Python<'_>, width: u32, height: u32) -> PyResult<()> {
        plot(py, self.to_svg(py, width, height)?)
    }
    fn _repr_svg_(&self, py: Python<'_>) -> PyResult<String> {
        self.to_svg(py, 400, 300)
    }
    fn __repr__(&self) -> String {
        format!("Molecule3D({:?})", self.to_smiles())
    }
}

#[pyclass(module = "oxidane")]
struct MolParser;

#[pymethods]
impl MolParser {
    #[new]
    fn new() -> Self {
        Self
    }

    /// Read SDF or counted XYZ frames. I/O errors raise OSError, malformed input ValueError.
    fn parse_file(&self, file_path: PathBuf) -> PyResult<Vec<PyMolecule3D>> {
        let extension = file_path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let molecules = match extension.as_str() {
            "xyz" => parse_xyz(&std::fs::read_to_string(file_path)?)?,
            "sdf" => Molecule3D::from_sdf(file_path).map_err(|error| match error {
                SDFParseError::IoError(error) => PyErr::from(error),
                other => value_error(other),
            })?,
            _ => {
                return Err(PyValueError::new_err(
                    "supported file extensions are .xyz and .sdf",
                ));
            }
        };
        Ok(molecules
            .into_iter()
            .map(|inner| PyMolecule3D { inner })
            .collect())
    }
}

// The historical Rust XYZ convenience reader panics on file-open failures and
// skips malformed rows. Parse counted frames here so Python always gets errors
// and multi-frame files cannot be silently merged into a single structure.
fn parse_xyz(input: &str) -> PyResult<Vec<Molecule3D>> {
    let mut lines = input.lines().peekable();
    let mut result = Vec::new();
    let mut separator = false;
    while let Some(count_line) = lines.next() {
        if count_line.trim().is_empty() {
            separator = true;
            continue;
        }
        // The upstream fixture uses blank-separated atom blocks after the
        // first counted frame. Preserve that documented legacy input too.
        if !result.is_empty() && separator && count_line.trim().parse::<usize>().is_err() {
            let mut atoms = vec![xyz_atom(count_line)?];
            while lines.peek().is_some_and(|line| !line.trim().is_empty()) {
                atoms.push(xyz_atom(lines.next().expect("peeked line"))?);
            }
            result.push(Molecule3D::from_atoms(&atoms));
            separator = false;
            continue;
        }
        separator = false;
        let count = count_line.trim().parse::<usize>().map_err(value_error)?;
        if count == 0 {
            return Err(PyValueError::new_err("XYZ atom count must be positive"));
        }
        lines
            .next()
            .ok_or_else(|| PyValueError::new_err("XYZ comment line is missing"))?;
        let mut atoms = Vec::new();
        for _ in 0..count {
            let line = lines
                .next()
                .ok_or_else(|| PyValueError::new_err("truncated XYZ frame"))?;
            atoms.push(xyz_atom(line)?);
        }
        result.push(Molecule3D::from_atoms(&atoms));
    }
    Ok(result)
}

fn xyz_atom(line: &str) -> PyResult<Atom> {
    let atom = Atom::from_xyz_line(line).map_err(value_error)?;
    if line.split_whitespace().skip(1).any(|coordinate| {
        coordinate
            .parse::<f64>()
            .is_ok_and(|value| !value.is_finite())
    }) {
        return Err(PyValueError::new_err("XYZ coordinates must be finite"));
    }
    Ok(atom)
}

#[pyclass(module = "oxidane", name = "ReactionSmarts")]
struct PyReactionSmarts {
    inner: RustReaction,
    #[pyo3(get)]
    smarts: String,
}

#[pymethods]
impl PyReactionSmarts {
    /// Reaction atoms must carry explicit map indices, as in the Rust API.
    #[new]
    fn new(smarts: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RustReaction::parse(smarts).map_err(value_error)?,
            smarts: smarts.to_owned(),
        })
    }
    fn apply(&self, molecule: &PyMolecule) -> Vec<PyMolecule> {
        self.inner
            .apply(&molecule.inner)
            .into_iter()
            .map(|inner| PyMolecule { inner })
            .collect()
    }
    fn apply_to(&self, reactants: Vec<PyRef<'_, PyMolecule>>) -> Vec<PyMolecule> {
        let references = reactants
            .iter()
            .map(|molecule| &molecule.inner)
            .collect::<Vec<_>>();
        self.inner
            .apply_to(&references)
            .into_iter()
            .map(|inner| PyMolecule { inner })
            .collect()
    }
    /// Show exact SMARTS, or draw all concrete outcomes with reactants supplied.
    #[pyo3(signature = (*, reactants=None, width=800, height=260))]
    fn to_svg(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        reactants: Option<Vec<PyRef<'_, PyMolecule>>>,
        width: u32,
        height: u32,
    ) -> PyResult<String> {
        py.import("oxidane._display")?
            .getattr("rule_svg")?
            .call1((slf, reactants, width, height))?
            .extract()
    }
    #[pyo3(signature = (*, reactants=None, width=800, height=260))]
    fn plot(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        reactants: Option<Vec<PyRef<'_, PyMolecule>>>,
        width: u32,
        height: u32,
    ) -> PyResult<()> {
        plot(py, Self::to_svg(slf, py, reactants, width, height)?)
    }
    fn _repr_svg_(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<String> {
        Self::to_svg(slf, py, None, 800, 260)
    }
}

fn tautomer_rules(carbon_endpoints: bool, ionizable_endpoints: bool) -> TautomerRules {
    TautomerRules {
        carbon_endpoints,
        ionizable_endpoints,
        ..TautomerRules::new()
    }
}

/// What to do with a SMILES that does not parse.
#[derive(Clone, Copy, PartialEq)]
enum OnError {
    /// Give the input back unchanged and say so on stderr.
    Keep,
    /// Give None in its place, silently.
    Null,
    Raise,
}

impl OnError {
    fn parse(errors: &str) -> PyResult<Self> {
        match errors {
            "keep" => Ok(Self::Keep),
            "none" => Ok(Self::Null),
            "raise" => Ok(Self::Raise),
            _ => Err(PyValueError::new_err(
                "errors must be \"keep\", \"none\" or \"raise\"",
            )),
        }
    }
}

/// Whether `value` is a missing entry rather than a SMILES: `None`, a float
/// NaN, or pandas' `NA`, which is what a column with gaps holds.
fn is_missing(value: &Bound<'_, PyAny>) -> bool {
    value.is_none()
        || value
            .cast::<PyFloat>()
            .is_ok_and(|number| number.value().is_nan())
        || value.get_type().name().is_ok_and(|name| name == "NAType")
}

/// One rayon pool per thread count asked for, built on first use; building one
/// per call cost more than canonizing a small batch.
fn pool(threads: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(pool) = pools.get(&threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(value_error)?,
    );
    pools.insert(threads, Arc::clone(&pool));
    Ok(pool)
}

/// Applies `compute` to one SMILES or to every SMILES of an iterable.
///
/// A single string gives back a string. Anything else is read as an iterable
/// of strings, copied out while the GIL is held, and computed on rayon's pool
/// without it, so other Python threads keep running; the answers come back as
/// a list in input order. Surrounding whitespace is ignored. A missing entry
/// (`None`, NaN, `pandas.NA`) gives `None`. A SMILES that does not parse is
/// handled as `errors` says: kept as it was with a line on stderr, replaced by
/// None, or raised.
fn map_smiles(
    py: Python<'_>,
    smiles: &Bound<'_, PyAny>,
    threads: Option<usize>,
    errors: &str,
    compute: impl Fn(&str) -> Result<String, SMILESParseError> + Sync,
) -> PyResult<Py<PyAny>> {
    let on_error = OnError::parse(errors)?;
    if threads == Some(0) {
        return Err(PyValueError::new_err("threads must be positive"));
    }
    let single = smiles.cast::<PyString>().is_ok();
    if !single && is_missing(smiles) {
        return Ok(py.None());
    }
    let inputs: Vec<Option<String>> = if single {
        vec![Some(smiles.extract::<String>()?)]
    } else {
        smiles
            .try_iter()
            .map_err(|_| PyTypeError::new_err("smiles must be a string or an iterable of strings"))?
            .enumerate()
            .map(|(index, item)| {
                let item = item?;
                if is_missing(&item) {
                    return Ok(None);
                }
                item.extract::<String>().map(Some).map_err(|_| {
                    PyTypeError::new_err(format!(
                        "item {index} is not a string (got {})",
                        item.get_type()
                            .name()
                            .map_or_else(|_| "?".to_owned(), |name| name.to_string())
                    ))
                })
            })
            .collect::<PyResult<Vec<Option<String>>>>()?
    };
    let pool = match threads {
        Some(count) if count > 1 => Some(pool(count)?),
        _ => None,
    };
    let answer = |smiles: &Option<String>| smiles.as_deref().map(|smiles| compute(smiles.trim()));
    let answers: Vec<Option<Result<String, SMILESParseError>>> =
        py.detach(|| match (threads, pool) {
            (Some(1), _) => inputs.iter().map(answer).collect(),
            (_, Some(pool)) => pool.install(|| inputs.par_iter().map(answer).collect()),
            _ => inputs.par_iter().map(answer).collect(),
        });
    let mut list = Vec::with_capacity(answers.len());
    let mut report = String::new();
    for (index, (answer, input)) in answers.into_iter().zip(inputs).enumerate() {
        match answer {
            None => list.push(None),
            Some(Ok(answer)) => list.push(Some(answer)),
            Some(Err(error)) => {
                let input = input.unwrap_or_default();
                let place = if single {
                    String::new()
                } else {
                    format!(" at index {index}")
                };
                let message = format!("cannot read SMILES {input:?}{place}: {error:?}");
                match on_error {
                    OnError::Raise => return Err(PyValueError::new_err(message)),
                    OnError::Null => list.push(None),
                    OnError::Keep => {
                        report.push_str("oxidane: ");
                        report.push_str(&message);
                        report.push_str(", kept as it was\n");
                        list.push(Some(input));
                    }
                }
            }
        }
    }
    if !report.is_empty() {
        py.import("sys")?
            .getattr("stderr")?
            .call_method1("write", (report,))?;
    }
    if single {
        Ok(list.pop().flatten().into_pyobject(py)?.into_any().unbind())
    } else {
        Ok(list.into_pyobject(py)?.into_any().unbind())
    }
}

/// Canonical SMILES; components are canonized separately and sorted.
#[pyfunction]
#[pyo3(signature = (smiles, *, threads=None, errors="keep"))]
fn canonical_smiles(
    py: Python<'_>,
    smiles: &Bound<'_, PyAny>,
    threads: Option<usize>,
    errors: &str,
) -> PyResult<Py<PyAny>> {
    map_smiles(py, smiles, threads, errors, |smiles| {
        canonicalize_and_sort_smiles(smiles, true)
    })
}

/// A string every tautomer of the input shares.
#[pyfunction]
#[pyo3(signature = (smiles, *, carbon_endpoints=false, ionizable_endpoints=false, threads=None, errors="keep"))]
fn tautomer_key(
    py: Python<'_>,
    smiles: &Bound<'_, PyAny>,
    carbon_endpoints: bool,
    ionizable_endpoints: bool,
    threads: Option<usize>,
    errors: &str,
) -> PyResult<Py<PyAny>> {
    let rules = tautomer_rules(carbon_endpoints, ionizable_endpoints);
    map_smiles(py, smiles, threads, errors, |smiles| {
        tautomer_key_from_smiles(smiles, &rules)
    })
}

/// The canonical SMILES of the one tautomer every tautomer of the input
/// normalizes to.
#[pyfunction]
#[pyo3(signature = (smiles, *, carbon_endpoints=false, ionizable_endpoints=false, threads=None, errors="keep"))]
fn canonical_tautomer(
    py: Python<'_>,
    smiles: &Bound<'_, PyAny>,
    carbon_endpoints: bool,
    ionizable_endpoints: bool,
    threads: Option<usize>,
    errors: &str,
) -> PyResult<Py<PyAny>> {
    let rules = tautomer_rules(carbon_endpoints, ionizable_endpoints);
    map_smiles(py, smiles, threads, errors, |smiles| {
        normalized_smiles(smiles, &rules)
    })
}

#[pymodule]
fn _oxidane(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(canonical_smiles, module)?)?;
    module.add_function(wrap_pyfunction!(tautomer_key, module)?)?;
    module.add_function(wrap_pyfunction!(canonical_tautomer, module)?)?;
    module.add_class::<PyMolecule>()?;
    module.add_class::<PyMolecule3D>()?;
    module.add_class::<MolParser>()?;
    module.add_class::<Smarts>()?;
    module.add_class::<PyReactionSmarts>()?;
    Ok(())
}
