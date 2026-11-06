# Sasakian Hodge numbers of Calabi–Yau links

Data, code and certificates for the paper *Sasakian Hodge Numbers of Calabi–Yau Links* by E. Hirst, H. N. Sá Earp and T. S. R. Silva.

For a weighted homogeneous polynomial of degree d = Σ wᵢ with an isolated singularity, the Sasakian Hodge numbers of its link depend only on the weights. They are given by alternating sums of denumerants p_w(k), which count the monomials of weighted degree k:

* 5-dimensional links (K3 surfaces): h¹¹_S = p_w(d) − Σᵢ p_w(wᵢ).
* 7-dimensional links (Calabi–Yau threefolds): h²¹_S = p_w(d) − Σᵢ p_w(wᵢ).
* 9-dimensional links (Calabi–Yau fourfolds):
  * h³¹_S = p_w(d) − Σᵢ p_w(wᵢ);
  * h²²_S = p_w(2d) − Σᵢ p_w(d+wᵢ) + Σ_{i<j} p_w(wᵢ+wⱼ).

This repository contains the resulting datasets for all 95 K3, 7555 threefold and 1 100 055 fourfold weight systems, the code that produces them, and the scripts that check them.

```
data/            the three datasets and, in data/inputs/, the input data with SHA-256 checksums
code/            scripts that regenerate the datasets from the inputs
certificates/    verification scripts; their outputs and logs are in certificates/results/
environment/     how to set up the Python environment (and Singular, for one certificate)
```

## Reading the data

Each row is identified by its weight system: the column `weights` lists the weights in nondecreasing order, separated by spaces. `d` is the degree, equal to the sum of the weights.

| File | Rows | Main columns |
|---|---|---|
| `data/k3_links.csv` | 95 | `h20_S`, `h11_S`, `b2_link`, `singularities` (the A-type singular points of the K3 surface), `milnor_number`, `groebner_h11_S` (Gröbner-basis value for an explicit polynomial) |
| `data/cy3_links.csv` | 7555 | Kreuzer–Skarke `h11_CY`, `h21_CY`; `h30_S`, `h21_S`; `gap_h21 = h21_CY − h21_S`; Crowley–Nordström `nu`; `milnor_number`; `twisted_curves`; `paper_h21_S`, `paper_nu` (Gröbner-basis values of arXiv:2310.03064 where available) |
| `data/cy4_links.csv.gz` | 1 100 055 | Kreuzer–Skarke `h11_CY`, `h21_CY`, `h31_CY`, `chi`; `h22_CY` (from the stringy Libgober–Wood identity); `h40_S`, `h31_S`, `h22_S`; `b4_link`; `gap31 = h31_CY − h31_S`; `milnor_number` (exact); `twisted_surfaces` |

The columns `twisted_curves` and `twisted_surfaces` list the twisted sectors that contribute to the gap, separated by `;`.
* Each entry has the form `F:n:g`.
* `F` is the string of (0-based) positions of the weights fixed by the sector.
* `n` is the number of group elements with that fixed locus.
* `g` is the geometric genus of the curve or surface.
* The gap equals Σ n·g.

The fourfold table is gzip-compressed; it can be read without unpacking, for example with pandas (not needed for the scripts):

```python
import pandas as pd
k3  = pd.read_csv("data/k3_links.csv")
cy3 = pd.read_csv("data/cy3_links.csv")
cy4 = pd.read_csv("data/cy4_links.csv.gz")
row = cy4[cy4.weights == "1 1 1 1 1 1"]                     # the sextic fourfold
```

or, with the standard library only,

```python
import csv, gzip
with gzip.open("data/cy4_links.csv.gz", "rt") as fh:
    for r in csv.DictReader(fh):
        w = tuple(map(int, r["weights"].split()))
        h31, h22 = int(r["h31_S"]), int(r["h22_S"])
```

### Inputs (`data/inputs/`)

| File | Content | Source |
|---|---|---|
| `WP4s.txt`, `WP4_Hodges.txt` | the 7555 threefold weight systems and their Kreuzer–Skarke (h¹¹, h²¹) | as used in arXiv:2310.03064, from [MLcCY7](https://github.com/TomasSilva/MLcCY7) |
| `Topological_Data.txt` | Gröbner-basis Sasakian Hodge numbers and ν for 7549 threefold links | arXiv:2310.03064 (the version before the 2024 addition of four further systems) |
| `5dTransWH.all.gz` | the 1 100 055 fourfold weight systems with their Hodge numbers (`V:` = Vafa's formula) | [Kreuzer–Skarke](http://hep.itp.tuwien.ac.at/~kreuzer/CY/CYff.html) |
| `K3Topological_Data.txt` | one explicit polynomial per K3 weight system, with its Gröbner-basis (h²⁰_S, h¹¹_S) | authors' follow-up computations |
| `CY4Polynomials_42.txt` | 42 dense fourfold polynomials | authors' follow-up computations |

The checksums are in `data/inputs/SHA256SUMS` (check them with `shasum -a 256 -c SHA256SUMS`).

## Regenerating the datasets

Set up the environment first (see `environment/README.md`), then run, from `code/`:

```bash
python k3_links.py          # data/k3_links.csv          (< 1 s)
python closed_formulas.py   # data/cy3_links.csv         (about 20 s)
python cy4_links.py         # data/cy4_links.csv.gz      (about 5 min with 8 processes)
python cy4_statistics.py    # summary statistics of the fourfold table (Table 6.1 of the paper)
```

`python cy4_links.py 2000` processes only the first 2000 fourfold systems and writes to the temporary directory, as a quick test. The first two scripts also compare with the Gröbner-basis data and print the equality counts quoted in the paper, for example 4202 of the 7555 threefold systems with h²¹_S = h²¹_CY.

## Running the certificates

Each script is run from `certificates/`, prints a summary, and writes its output to `certificates/results/`. These certificates do not use the closed formulas they check.

| Script | What it certifies | Paper | Time |
|---|---|---|---|
| `six_remaining_7d.py` | invariants of the six threefold links left open in arXiv:2310.03064: the Hilbert function of the Milnor algebra in every degree 0 ≤ ℓ ≤ 3d for an invertible polynomial, and h²¹_S for a dense random polynomial | §3.4, Table 3.1 | 1–2 min |
| `certify_9d.py 0` | h³¹_S for the 42 dense fourfold polynomials | §6(c) | < 1 min |
| `certify_9d_stratified.py 100 2024` | h³¹_S for 900 fourfold systems (100 per decade of μ, invertible polynomials) and h²²_S for 655 of them | §6(c) | about 1 min |
| `certify_9d_noinv.py 25` | h³¹_S and h²²_S for 25 randomly drawn fourfold systems with no invertible polynomial, dim S_2d ≤ 2500 and μ ≤ 10⁷ (dense random polynomials) | §6(c) | < 1 min |
| `groebner_9d.py` | full Gröbner bases over ℚ for invertible polynomials of the 42 fourfold systems: μ and the whole Hilbert function | §6(c) | < 1 min |
| `milnor_orlik_check.py` | the topological Milnor–Orlik formula for the middle Betti number against b₃ = 2 + 2h²¹_S (7549 Gröbner-basis records) and b₄ = 2 + 2h³¹_S + h²²_S (all fourfolds) | §6(b) | 2 min |
| `itoh_independent_check.py` | the sheaf-cohomology formula (Theorem 3.9) term by term, against classical hypersurfaces, 300 Gröbner-basis computations and all three datasets | §3.3, §6(d) | about 10 min |
| `isolation_certificates.py` | that every dense or random test polynomial has an isolated singularity (Gröbner basis of the Jacobian ideal over F₃₂₀₀₃; needs Singular) | App. A.2 | seconds to hours per polynomial |
| `isolation_certificates_k3gb.py` | the same for the 95 explicit K3 polynomials, and their Gröbner-basis Hodge numbers | §5, App. A.2 | about 1 min |

The graded pieces dim(M_f)_ℓ of the Milnor algebra are certified exactly over ℚ by the following test.
* Compute ranks modulo p = 2³¹ − 1 of the two integer matrices of the degree-ℓ Koszul strand K₂ → K₁ → S_ℓ of the partial derivatives.
* If they add up to dim K₁, the ranks over ℚ are the same, and dim(M_f)_ℓ = dim S_ℓ − rank D₁ (Appendix A.3 of the paper).
* A script prints `certified True` when this condition holds.

The random polynomials are regenerated from the seeds fixed in the scripts.

`isolation_certificates.py` needs the Singular binary; pass it as `SINGULAR=/path/to/Singular`. Some fourfold polynomials take an hour or more, so the script is resumable: each finished polynomial is appended to `results/isolation_certificates_results.jsonl` and skipped on a rerun. `--only TAG` restricts the run to one group (`six7d_inv`, `six7d_dense`, `cy4_noinv`, `cy4_42`). The final summary is written to `results/isolation_certificates_output.txt` once every polynomial is done.

## License

The code, the datasets in `data/` and the certificate outputs are released under the MIT License (see `LICENSE`). The third-party input files in `data/inputs/` are included for reproducibility and remain subject to the terms of their original sources: the Kreuzer–Skarke lists and the MLcCY7 data of arXiv:2310.03064.

## Citation

If you use these data, please cite the paper above and arXiv:2310.03064, together with the Kreuzer–Skarke lists for the input weight systems.
