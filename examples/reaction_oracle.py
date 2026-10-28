"""Expected products of reaction SMARTS, from RDKit's RunReactants.

Writes `reaction<TAB>reactants<TAB>expected;expected` lines for
`examples/reaction_diff.rs`, which applies the same reactions with
`ReactionSmarts::parse` (RDKit's reading) and compares canonical products:

    uv run --no-project --with rdkit python -I examples/reaction_oracle.py > /scr/machine/tmp/rxn.tsv
    cargo run --release --example reaction_diff < /scr/machine/tmp/rxn.tsv

Every template is fully atom-mapped, as `ReactionSmarts::parse` requires.
Each expected entry is one outcome: its products sanitized and joined with
`.`. An outcome RDKit cannot sanitize is left out, since it is no molecule.
"""

import itertools
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ONE_REACTANT = [
    # bond orders
    "[C:1]-[O:2]>>[C:1]=[O:2]",
    "[C:1]=[C:2]>>[C:1]-[C:2]",
    "[C:1]#[C:2]>>[C:1]=[C:2]",
    "[C:1]#[N:2]>>[C:1]=[N:2]",
    "[C:1]~[C:2]>>[C:1]-[C:2]",
    "[C:1]-,=[C:2]>>[C:1]#[C:2]",
    "[C:1]=[C:2]>>[C:1][C:2]",
    "[#6:1][#6:2]>>[#6:1][#6:2]",
    "[C;H2:1]-[C;H2:2]>>[C:1]=[C:2]",
    # bonds broken and atoms consumed
    "[C:1]-[Cl:2]>>[C:1].[Cl:2]",
    "[C:1]-[Br:2]>>[C:1].[Br:2]",
    "[C:1]-[O:2]>>[C:1].[O:2]",
    "[C:1][Cl:2]>>[C:1]",
    "[C:1][O:2]>>[C:1]",
    "[C:1]=[O:2]>>[C:1]",
    "[C:1](=[O:2])-[O:3]>>[C:1]=[O:2]",
    # atoms created
    "[C:1][Cl:2]>>[C:1][O:3]",
    "[C:1][Br:2]>>[C:1][N:3]",
    "[C:1]=[O:2]>>[C:1](-[O:3])-[O:2]",
    "[N:1]>>[N:1]-[C:2]",
    "[O:1]>>[O:1]-[C:2]",
    # elements and charges
    "[C:1]>>[N:1]",
    "[O:1]>>[S:1]",
    "[N:1]>>[N+:1]",
    "[O:1]>>[O-:1]",
    "[N+:1]>>[N:1]",
    "[O-:1]>>[O:1]",
    "[C:1]>>[C+:1]",
    # query labels on the product side keep what was matched
    "[!C:1]>>[!C:1]",
    "[C,N:1]>>[C,N:1]",
    "[C,N:1]>>[O:1]",
    "[!#1;!#6:1]>>[!#1;!#6:1]",
    # rings
    "[C:1][C:2][C:3][C:4][O:5]>>[C:1]1[C:2][C:3][C:4][O:5]1",
    "[C:1]1[C:2][C:3][C:4][C:5]1>>[C:1][C:2][C:3][C:4][C:5]",
    "[C:1]1[C:2][C:3]1>>[C:1]=[C:2].[C:3]",
    # aromatic
    "[c:1][Cl:2]>>[c:1][O:3]",
    "[c:1][Br:2]>>[c:1][N:3]",
    "[c:1][N+:2](=[O:3])[O-:4]>>[c:1][N:2]",
    "[c:1][C:2]>>[c:1][C:2](=[O:3])[O:4]",
    "[n:1]>>[n+:1]-[O-:2]",
    # stereo
    "[C:1]-[Cl:2]>>[C:1]-[O:3]",
    "[C@:1]-[Cl:2]>>[C@@:1]-[O:3]",
    "[C@:1]-[Cl:2]>>[C@:1]-[O:3]",
    "[C@:1]-[Cl:2]>>[C:1]-[O:3]",
    "[C:1][O:2]>>[C:1][N:2]",
    "[C:1]=[C:2]>>[C:1]-[C:2]",
    "[C:1]>>[C:1]-[O:2]",
    "[C:1]-[Cl:2]>>[C:1]",
]

TWO_REACTANTS = [
    "[C:1](=[O:2])[O:3].[N:4]>>[C:1](=[O:2])[N:4].[O:3]",
    "[C:1](=[O:2])[O:3].[O:4][C:5]>>[C:1](=[O:2])[O:4][C:5].[O:3]",
    "[C:1][Br:2].[N:3]>>[C:1][N:3].[Br:2]",
    "[C:1][Cl:2].[O:3]>>[C:1][O:3].[Cl:2]",
    "[C:1]=[O:2].[N:3]>>[C:1]=[N:3].[O:2]",
    "[C:1].[N:2]>>[C:1]-[N:2]",
    "[c:1][Br:2].[c:3][B:4]>>[c:1][c:3].[Br:2].[B:4]",
    "[C:1]=[C:2].[C:3]=[C:4]>>[C:1]1[C:2][C:3][C:4]1",
]

MOLECULES = [
    "CCO", "CC(C)O", "OCCO", "CO", "CCCCO", "OCC(O)CO",
    "CC=O", "CC(=O)C", "CC(=O)O", "OC(=O)CC(=O)O", "CC(=O)OC",
    "C=C", "CC=CC", "C/C=C/C", "C/C=C\\C", "C=CC=C", "C#C", "CC#N",
    "CCCl", "ClCCCl", "CCBr", "C[C@H](Cl)F", "C[C@@H](Cl)CC", "C[C@@](F)(Cl)Br",
    "C[C@H](O)F", "C1CCCC1", "C1CC1", "C1CCCCC1", "C1=CC=CCC1",
    "CN", "CNC", "C[NH3+]", "C[O-]", "CS",
    "c1ccccc1", "Clc1ccccc1", "Brc1ccccc1", "Cc1ccccc1", "O=[N+]([O-])c1ccccc1",
    "c1ccncc1", "Oc1ccccc1", "OB(O)c1ccccc1",
]

PARTNERS = ["N", "CN", "O", "CO", "CC(=O)O", "CC=O", "C", "C=C", "OB(O)c1ccccc1", "Brc1ccccc1", "CCBr"]


def outcomes(reaction, reactants):
    found = set()
    for products in reaction.RunReactants(tuple(reactants)):
        smiles = []
        try:
            for product in products:
                Chem.SanitizeMol(product)
                smiles.append(Chem.MolToSmiles(product))
        except Exception:
            continue
        found.add(".".join(sorted(smiles)))
    return sorted(found)


def emit(smarts, reactant_smiles):
    reaction = AllChem.ReactionFromSmarts(smarts)
    reactants = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    expected = outcomes(reaction, reactants)
    print(f"{smarts}\t{'.'.join(reactant_smiles)}\t{';'.join(expected)}")


def main():
    for smarts in ONE_REACTANT:
        for molecule in MOLECULES:
            emit(smarts, [molecule])
    for smarts in TWO_REACTANTS:
        for first, second in itertools.product(MOLECULES + PARTNERS, PARTNERS):
            emit(smarts, [first, second])
    sys.stdout.flush()


if __name__ == "__main__":
    main()
