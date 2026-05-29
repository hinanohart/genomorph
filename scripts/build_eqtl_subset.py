"""Build the embedded eQTL Catalogue mechanism-labelled variant subset.

Builds a small, high-confidence labelled variant subset from SuSiE fine-mapped
credible sets of two matched-tissue datasets from the eQTL Catalogue
(CC-BY-4.0), for reproducible evaluation:

  * expression mechanism: gene-expression QTLs (quant_method = ``ge``)
  * splicing mechanism:   leafcutter splicing QTLs (quant_method = ``leafcutter``)

Both are macrophage (study QTS000001), so mechanism is the dominant contrast.
We keep the maximum-PIP lead variant per credible set above a PIP threshold and
drop variants that are credible-set leads in *both* mechanisms (ambiguous
ground truth). The result is committed under data/eqtl_subset/.

Provenance -- fetch the two source files first (CC-BY-4.0, EMBL-EBI), then run
this script on the directory holding them:

    base=ftp://ftp.ebi.ac.uk/pub/databases/spot/eQTL/susie/QTS000001
    curl -s "$base/QTD000001/QTD000001.credible_sets.tsv.gz" -o /tmp/QTD000001_cs.tsv.gz
    curl -s "$base/QTD000005/QTD000005.credible_sets.tsv.gz" -o /tmp/QTD000005_cs.tsv.gz
    python scripts/build_eqtl_subset.py /tmp
"""

from __future__ import annotations

import csv
import gzip
import io
import sys
from pathlib import Path

DATASETS = {
    "expression": ("QTD000001", "ge"),
    "splicing": ("QTD000005", "leafcutter"),
}
PIP_MIN = 0.9
OUT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "genomorph"
    / "data"
    / "eqtl_subset"
)


def _lead_variants(raw: bytes) -> dict[str, dict]:
    """Max-PIP lead variant per credible set, above PIP_MIN."""
    text = gzip.decompress(raw).decode()
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    best: dict[str, dict] = {}
    for row in reader:
        pip = float(row["pip"])
        if pip < PIP_MIN:
            continue
        cs = row["cs_id"]
        if cs not in best or pip > float(best[cs]["pip"]):
            best[cs] = row
    return best


def main(local_dir: str) -> int:
    local = Path(local_dir)
    per_mech: dict[str, dict[str, dict]] = {}
    for mech, (ds_id, _qm) in DATASETS.items():
        src = local / f"{ds_id}_cs.tsv.gz"
        if not src.exists():
            sys.stderr.write(
                f"missing {src}; fetch source files first (see module docstring)\n"
            )
            return 2
        leads = _lead_variants(src.read_bytes())
        per_mech[mech] = {r["variant"]: r for r in leads.values()}
        sys.stderr.write(
            f"{mech} ({ds_id}): {len(per_mech[mech])} lead variants PIP>{PIP_MIN}\n"
        )

    # drop variants that lead a credible set in both mechanisms (ambiguous)
    ambiguous = set(per_mech["expression"]) & set(per_mech["splicing"])
    sys.stderr.write(f"dropping {len(ambiguous)} ambiguous (lead in both)\n")

    OUT.mkdir(parents=True, exist_ok=True)
    out_tsv = OUT / "macrophage_expr_vs_splice.tsv"
    cols = [
        "variant",
        "rsid",
        "mechanism",
        "quant_method",
        "dataset_id",
        "molecular_trait_id",
        "pip",
        "region",
    ]
    n = 0
    with out_tsv.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for mech, (ds_id, qm) in DATASETS.items():
            for variant, row in sorted(per_mech[mech].items()):
                if variant in ambiguous:
                    continue
                w.writerow(
                    [
                        variant,
                        row["rsid"],
                        mech,
                        qm,
                        ds_id,
                        row["molecular_trait_id"],
                        row["pip"],
                        row["region"],
                    ]
                )
                n += 1
    sys.stderr.write(f"wrote {n} variants -> {out_tsv}\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python scripts/build_eqtl_subset.py <dir-with-gz>\n")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
