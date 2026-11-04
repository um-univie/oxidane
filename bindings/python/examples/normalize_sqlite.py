"""Add canonical SMILES, tautomer keys and canonical tautomers to a SQLite table.

    python normalize_sqlite.py compounds.db compounds smiles

adds the columns canonical_smiles, tautomer_key and canonical_tautomer to the
table (if missing) and fills them for every row, 100,000 rows at a time. A
SMILES that does not parse leaves NULL. The same loop works for any database
driver: read a batch of (row id, SMILES), call the oxidane functions on the
SMILES list, write the results back.
"""

import sqlite3
import sys
import time

import oxidane as ox

COLUMNS = ("canonical_smiles", "tautomer_key", "canonical_tautomer")


def normalize_table(path, table, smiles_column="smiles", batch_size=100_000):
    connection = sqlite3.connect(path)
    existing = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
    for name in COLUMNS:
        if name not in existing:
            connection.execute(f'ALTER TABLE "{table}" ADD COLUMN {name} TEXT')
    update = (f'UPDATE "{table}" SET ' + ", ".join(f"{name} = ?" for name in COLUMNS)
              + " WHERE rowid = ?")
    last, done = -1, 0
    while True:
        rows = connection.execute(
            f'SELECT rowid, "{smiles_column}" FROM "{table}" WHERE rowid > ? '
            "ORDER BY rowid LIMIT ?", (last, batch_size)).fetchall()
        if not rows:
            break
        ids = [rowid for rowid, _ in rows]
        smiles = [value for _, value in rows]  # NULL arrives as None and stays None
        results = [
            ox.canonical_smiles(smiles, errors="none"),
            ox.tautomer_key(smiles, errors="none"),
            ox.canonical_tautomer(smiles, errors="none"),
        ]
        connection.executemany(update, zip(*results, ids))
        connection.commit()
        last, done = ids[-1], done + len(rows)
        print(f"{done} rows", file=sys.stderr)
    connection.close()
    return done


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    start = time.perf_counter()
    count = normalize_table(*sys.argv[1:])
    print(f"normalized {count} rows in {time.perf_counter() - start:.1f} s", file=sys.stderr)
