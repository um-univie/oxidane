"""Regression checks for the coordinate/energy plumbing, without running xTB.

uv run --with rdkit --with numpy python -m unittest discover -s tests -p test_forcefield_oracles.py
"""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import uff_xtb_probe as probe
import embed_benchmark as benchmark


class OracleTests(unittest.TestCase):
    def test_hydrogen_alignment_accepts_phosphorus_bond_lengths(self):
        molecule = Chem.AddHs(Chem.MolFromSmiles("P"))
        rows = [("P", np.zeros(3)), ("H", np.array([1.42, 0., 0.])),
                ("H", np.array([0., 1.42, 0.])), ("H", np.array([0., 0., 1.42]))]
        order, problem = benchmark.align_hydrogens(molecule, rows, np.array([p for _, p in rows]))
        self.assertIsNone(problem)
        self.assertEqual(sorted(order), list(range(4)))

    def test_mmff_maps_hydrogens_and_returns_original_atom_order(self):
        molecule = Chem.AddHs(Chem.MolFromSmiles("CO"))
        self.assertEqual(AllChem.EmbedMolecule(molecule, randomSeed=42), 0)
        rows = [(a.GetSymbol(), np.array(p)) for a, p in
                zip(molecule.GetAtoms(), molecule.GetConformer().GetPositions())]
        # Swap an O-H with a C-H: identical symbols hide the wrong connectivity.
        rows[2], rows[-1] = rows[-1], rows[2]

        def check_geometry(mapped, **kwargs):
            positions = mapped.GetConformer().GetPositions()
            for bond in mapped.GetBonds():
                length = np.linalg.norm(positions[bond.GetBeginAtomIdx()] -
                                        positions[bond.GetEndAtomIdx()])
                self.assertLess(length, 1.6)
            return 0

        with patch.object(AllChem, "MMFFOptimizeMolecule", side_effect=check_geometry):
            relaxed = probe.mmff_relaxed("CO", rows)
        np.testing.assert_allclose([p for _, p in relaxed], [p for _, p in rows])
        with patch.object(AllChem, "MMFFOptimizeMolecule", return_value=1):
            self.assertIsNone(probe.mmff_relaxed("CO", rows))

    def test_reference_measurements_use_the_optimized_geometry(self):
        start = [("C", np.zeros(3)), ("C", np.array([1.5, 0., 0.]))]
        optimized = [("C", np.zeros(3)), ("C", np.array([1.4, 0., 0.]))]
        output = io.StringIO()
        with (patch.object(probe.sys, "argv", ["probe"]),
              patch.object(probe, "MOLECULES", [("CC", "ethane", None, None, None)]),
              patch.object(probe, "run_example", return_value=(start, start)),
              patch.object(probe, "expected_bonds", return_value={(0, 1)}),
              patch.object(probe, "mmff_relaxed", return_value=None),
              patch.object(probe, "opt_chained", return_value=(-10., optimized)),
              patch.object(probe, "sp_energy", return_value=-9.99),
              patch.object(probe, "measures", side_effect=lambda s, r: [("C-C", probe.dist(r, 0, 1))]),
              contextlib.redirect_stdout(output)):
            probe.main()
        self.assertIn("1.500 -> 1.400", output.getvalue())

    def test_no_valid_reference_is_reported_without_crashing(self):
        rows = [("C", np.zeros(3)), ("C", np.array([1.5, 0., 0.]))]
        output = io.StringIO()
        with (patch.object(probe.sys, "argv", ["probe"]),
              patch.object(probe, "MOLECULES", [("CC", "ethane", None, None, None)]),
              patch.object(probe, "run_example", return_value=(rows, rows)),
              patch.object(probe, "expected_bonds", return_value={(0, 1)}),
              patch.object(probe, "mmff_relaxed", return_value=None),
              patch.object(probe, "opt_chained", return_value=(None, None)),
              contextlib.redirect_stdout(output)):
            probe.main()
        self.assertIn("no converged reference retaining connectivity", output.getvalue())

    def test_xtb_receives_molecular_charge_and_unpaired_electrons(self):
        # Inject the small interface used by the scorer; no tblite dependency
        # is needed to check which electronic state is sent to it.
        from types import ModuleType
        from unittest.mock import Mock
        interface = ModuleType("tblite.interface")
        calculator = interface.Calculator = Mock()
        calculator.return_value.singlepoint.return_value.get.return_value = -1.0
        for smiles, charge, uhf in [("C[P+](C)(C)C", 1., 0), ("[CH3]", 0., 1)]:
            molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
            self.assertEqual(AllChem.EmbedMolecule(molecule, randomSeed=42), 0)
            with patch.dict("sys.modules", {"tblite.interface": interface}):
                self.assertEqual(benchmark.xtb_energy(molecule), -1.0)
            self.assertEqual(calculator.call_args.kwargs, {"charge": charge, "uhf": uhf})


if __name__ == "__main__":
    unittest.main()
