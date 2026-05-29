# Embedded eQTL Catalogue subset (CC-BY-4.0)

`macrophage_expr_vs_splice.tsv` is a small, high-confidence, mechanism-labelled
variant subset derived from the **eQTL Catalogue** SuSiE fine-mapped credible
sets (release on EMBL-EBI FTP). It is redistributed here under **CC-BY-4.0**
with attribution for reproducible evaluation only.

> Kerimov, N., Hayhurst, J.D., Peikova, K. et al. *A compendium of uniformly
> processed human gene expression and splicing quantitative trait loci.*
> Nat Genet 53, 1290–1299 (2021). Data: https://www.ebi.ac.uk/eqtl/

## What it contains

| mechanism    | source dataset | quant_method | meaning                              |
|--------------|----------------|--------------|--------------------------------------|
| `expression` | QTD000001      | `ge`         | gene-expression QTL lead variants    |
| `splicing`   | QTD000005      | `leafcutter` | splicing QTL lead variants           |

Both datasets are **macrophage** (study QTS000001), so the dominant contrast
between the two label classes is the *molecular mechanism* (total expression vs
splicing), not tissue.

Selection: the maximum-PIP lead variant of each credible set with PIP > 0.9;
variants that lead a credible set in *both* mechanisms are dropped (ambiguous
ground truth). 167 variants total (76 expression + 91 splicing).

Columns: `variant` (`chr_pos_ref_alt`, hg38), `rsid`, `mechanism` (label),
`quant_method`, `dataset_id`, `molecular_trait_id`, `pip`, `region`.

## Regenerate

See `scripts/build_eqtl_subset.py` (fetch the two `.credible_sets.tsv.gz`
source files from the eQTL Catalogue FTP, then run the builder).
