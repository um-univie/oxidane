# Small-ring bounds investigation, 2026-10-19

The initial distance bounds can exclude relaxed bond lengths in strained
spiro and fused rings. For spiro[2.2]pentane (`C1CC12CC2`), the 0-2 bound
is 1.5075–1.5475 Å while the frozen MMFF-relaxed reference bond is 1.4572 Å
(`tests/bounds_reference_geometry.tsv`). This is a limitation of the initial
geometry model; it does not by itself establish a defect in the returned
conformer, which normally undergoes force-field relaxation.

## Experiment

Compared cyclopropane, spiro[2.2]pentane, bicyclo[2.1.0]pentane,
methylcyclopropane, oxirane, aziridine, cyclobutane and cyclopentane.

- Reference: RDKit 2026.09.1, eight ETKDGv3 conformers per molecule,
  seed 20261019, each minimized to convergence with MMFF94 (up to 3,000
  iterations). The energy reference is the lowest of those eight minima.
- Oxidane: default configuration with seeds 0, 42 and 0xE7D65EED; repeated
  with `relax = false`. This gives 24 relaxed and 24 unrelaxed structures.
- Scoring: RDKit MMFF94 energy of the returned coordinates, before any
  RDKit relaxation. Heavy-atom order was checked and hydrogen order aligned
  by bonded parent before scoring. Both implementations ran on one thread.

| Molecule | Unrelaxed excess energy, kcal/mol | Relaxed excess energy, kcal/mol |
|---|---:|---:|
| Cyclopropane | 0.907–1.335 | < 0.000001 absolute |
| Spiro[2.2]pentane | 5.276–5.854 | < 0.000001 absolute |
| Bicyclo[2.1.0]pentane | 2.712–3.171 | < 0.000001 absolute |
| Methylcyclopropane | 16.531–16.937 | < 0.000001 absolute |
| Oxirane | 0.936–1.620 | < 0.000001 absolute |
| Aziridine | 0.880–1.884 | < 0.000001 absolute |
| Cyclobutane | 2.200–3.270 | < 0.000001 absolute |
| Cyclopentane | 2.893–3.477 | < 0.000001 absolute |

All default relaxed outputs reached the lowest sampled reference energy.
The spiro 0-2 bond reached 1.45721 Å for all three seeds, outside its initial
window and consistent with the frozen reference. Excess energy in the
unrelaxed output measures the whole unrelaxed pipeline, so it cannot all
be attributed to bond-length bounds alone.

## Decision

Keep the bounds model unchanged on this evidence. A general small-ring
shortening correction is not justified by this motif-specific discrepancy;
ordinary cyclopropane's reference bonds already lie inside their bounds.
The default pipeline corrected the discrepancy in the cases tested.

`tests/small_ring_geometry.rs` now guards final spiro/fused bond lengths
across the three seeds without pinning the biased initial bounds. Turning
relaxation off retains the limitations of the initial geometry model.
This small study does not establish behavior for larger strained systems
or an independently validated physical energy minimum: MMFF94 is also the
force field used by Oxidane's relaxation.
