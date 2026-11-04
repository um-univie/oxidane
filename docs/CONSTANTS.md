# Constants and reference data

Use `chemistry_consts` for reusable chemical data: elemental properties, unit
conversions, charge parameters, Crippen and polar-surface contributions, and
force-field reference parameters. Oxidane
uses its `uff` and `mmff` modules through internal re-exports; there is one
copy of each table. The parameter lookup functions have no dependency on
Oxidane's molecule types.

Keep numerical tolerances, iteration budgets, damping schedules, fallback
radii and embedding estimates in the module that owns the algorithm. Give
constants a name describing their purpose and document units or the quantity
being compared. Equal numeric values do not imply the same meaning: a bond
length threshold, a sine floor and a squared-sine floor are separate settings.

Reference values retain their published/model precision. In particular, UFF
and MMFF use different electrostatic prefactors, and MMFF's historical unit
conversion is part of the parameterization. Do not silently replace these
with a CODATA-derived value. Likewise, the rounded equilibrium angles used
by different models should not be merged just because they are close.

Formula coefficients, atomic-number pattern matches, array indices and test
fixtures can remain literals when their meaning is clear. Tabulated values
belong in named tables; giving every cell a separate constant adds no meaning.

## Developing both crates

The manifest uses `chemistry_consts = { version = "0.3.1", path =
"../chemistry_consts" }`, so check out the two crates side by side. This also
applies when building the Python bindings. Before publishing Oxidane, publish
chemistry_consts 0.3.1 with the new data/API; Cargo uses the version requirement
in the packaged dependency.

Existing charge, force-field energy/gradient and embedding tests cover the
consumers. chemistry_consts checks lookup symmetry, table ordering and units.
