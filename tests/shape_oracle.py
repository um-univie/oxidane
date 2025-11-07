"""RDKit's USR, USRCAT and 3D descriptors for one conformer each, plus the SDF
that conformer came from, so that both sides measure the same coordinates.

Reads SMILES on stdin. Writes the conformers to the SDF named by the first
argument, and one tab-separated line per molecule to stdout:
`index<TAB>usr(12)<TAB>usrcat(60)<TAB>shape(11)`.
"""
import sys
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D
from rdkit.Chem import rdMolDescriptors as descriptors

RDLogger.DisableLog("rdApp.*")

# Two passes: write the conformers first, then measure them as they come back
# off disk. The SD file keeps four decimals, and a descriptor that picks the
# atom nearest the centroid can pick a different one when two are a ten
# thousandth of an angstrom apart -- so both sides must see the rounded
# coordinates, not one side the rounded and the other the exact.
writer = Chem.SDWriter(sys.argv[1])
index = 0
for line in sys.stdin:
    smiles = line.split("\t")[0].strip()
    if not smiles:
        continue
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        continue
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        continue
    molecule.SetProp("_Name", str(index))
    writer.write(molecule)
    index += 1
writer.close()


def joined(values):
    return ",".join(f"{value:.6f}" for value in values)


for index, molecule in enumerate(Chem.SDMolSupplier(sys.argv[1], removeHs=False)):
    if molecule is None:
        continue
    usr = descriptors.GetUSR(molecule)
    usrcat = descriptors.GetUSRCAT(molecule)
    shape = [
        Descriptors3D.PMI1(molecule),
        Descriptors3D.PMI2(molecule),
        Descriptors3D.PMI3(molecule),
        Descriptors3D.NPR1(molecule),
        Descriptors3D.NPR2(molecule),
        Descriptors3D.RadiusOfGyration(molecule),
        Descriptors3D.Asphericity(molecule),
        Descriptors3D.Eccentricity(molecule),
        Descriptors3D.InertialShapeFactor(molecule),
        Descriptors3D.SpherocityIndex(molecule),
        descriptors.CalcPBF(molecule),
    ]
    print(f"{index}\t{joined(usr)}\t{joined(usrcat)}\t{joined(shape)}")
