# Third-party material

Code and data in this repository that originates elsewhere, with the terms it
carries. Everything else is under the MIT licence in `LICENSE`.

## RDKit — BSD 3-Clause

The following pieces derive from [RDKit](https://github.com/rdkit/rdkit):

- **`src/molecule/cip/`** — a port of RDKit's `CIPLabeler`
  (`Code/GraphMol/CIPLabeler`), itself a port of John Mayfield's `centres`
  library for the Hanson *et al.* re-specification of the CIP rules.
- **`src/molecule/embed/torsion_preferences.rs`** — the experimental
  torsion-angle preferences of `torsionPreferences_v2.in`
  (`Code/GraphMol/ForceFieldHelpers/CrystalFF`), the rule set ETKDGv2 and
  ETKDGv3 use.
- **`src/molecule/forcefield/uff/params.rs`** — the UFF parameter table of
  `Code/ForceField/UFF/Params.cpp`. The term forms and parameter combination
  rules in `src/molecule/forcefield/uff/terms.rs` follow the same directory's
  implementation, which is also the oracle its energies are checked against.
- **`src/molecule/forcefield/mmff/`** — the MMFF94 atom typing rules and
  charge model of `Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp`, the
  MMFF aromaticity model of `setMMFFAromaticity` in
  `Code/GraphMol/Aromaticity.cpp`, the parameter tables of
  `Code/ForceField/MMFF/Params.cpp`, and the term forms, parameter-assignment
  rules and empirical fallbacks of `Code/ForceField/MMFF/` and the same
  `AtomTyper.cpp`. The rules are followed decision for decision, in the order
  the decisions are made; `tests/mmff_oracle.py` checks every atom's type and
  charge against the implementation they came from, and
  `tests/mmff_energy_oracle.py` checks every energy term separately.

RDKit's own licence text:

> Copyright (c) 2006-2015, Rational Discovery LLC, Greg Landrum, and Julie
> Penzotti and others. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice,
>   this list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
> * Neither the name of the copyright holder nor the names of its
>   contributors may be used to endorse or promote products derived from this
>   software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

The UFF parameters and functional form are from Rappe, Casewit, Colwell,
Goddard & Skiff, *J. Am. Chem. Soc.* **1992**, 114, 10024-10035. The MMFF94
atom types are Halgren's, *J. Comput. Chem.* **1996**, 17, 490-519 (MMFF I),
Table III, and the charge model is the same author's MMFF.V, *J. Comput.
Chem.* **1996**, 17, 616-641. The MMFF94 parameters themselves are Merck's,
copyright Merck and Co., Inc., 1994-1996, distributed with RDKit under the
licence above.

### Please also cite

The torsion preferences are a scientific dataset before they are a file. The
values were derived from the Cambridge Structural Database and curated by
molecular-design experts; work that uses them should cite the papers, not
just this crate:

- Guba, Meyder, Rarey & Hert. Torsion Library Reloaded: A New Version of
  Expert-Derived SMARTS Rules for Assessing Conformations of Small Molecules.
  *J. Chem. Inf. Model.* **2016**, 56, 1, 1-5.
- Riniker & Landrum. Better Informed Distance Geometry: Using What We Know To
  Improve Conformation Generation. *J. Chem. Inf. Model.* **2015**, 55, 12,
  2562-2574.
- Wang, Witek, Landrum & Riniker. Improving Conformer Generation for Small
  Rings and Macrocycles Based on Distance Geometry and Experimental
  Torsional-Angle Preferences. *J. Chem. Inf. Model.* **2020**, 60, 4,
  2044-2058.

## Canonaut — Apache-2.0

`vendor/canonaut-1.1.0/` contains the exact upstream library source at commit
`63f8b32`, with its `LICENSE.txt` and `NOTICE`. See `UPSTREAM.md` and
`UPSTREAM.sha256` there for provenance, checksums and the library-only manifest.
This corrected version is required; the published 1.0.0 is not a substitute.
