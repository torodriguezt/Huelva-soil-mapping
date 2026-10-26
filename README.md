# Huelva soil mapping

Bayesian multivariate spatial model for 22 elements in the topsoil of Huelva (SW Spain) and the
exceedance of the generic reference levels (NGR) of Decreto 18/2015. The model separates the
spatial level of each element from local site-to-site variation, and predicts both where the
spatial level exceeds the NGR and where a new sample would.

## Run

From the repository root, with the dependencies and input data in place:

```sh
python main.py
```

This reproduces the data, model fits, tables and figures of the article, including the
appendices. Most of the time goes to the spatial block cross-validation: 120 fits, plus a
second pass in which the fits that miss the convergence criteria (R-hat above 1.01 or ESS below
400) are rerun with chains two to six times longer. On 20 threads the whole run takes about a
day. Python scripts can also be run one by one from the root after setting `PYTHONPATH=Python`,
for example

```sh
python Python/All_Models/fit_model.py --model t_K3_ngr
python Python/All_Models/univariate_baseline.py
```

Finished fits are skipped, so an interrupted run can be resumed. Posterior summaries are
equal-tailed 95% credible intervals.

## Structure

| Folder | Content | Output |
| --- | --- | --- |
| `Python/Data_preparation/` | samples, land use of sites and grid, standardized model input | `results/data/` |
| `Python/All_Models/` | spatial block cross-validation, univariate baseline and full-data fits | `results/cv/`, `results/fits/` |
| `Python/Robustness/` | prior, record and element-group sensitivity | `results/fits/sensitivity/` |
| `Python/Results/` | grid predictions, exceedance areas, residual checks, tables and figures | `results/`, `figures/` |
| `Python/blmc/` | sampler, prediction, convergence diagnostics and shared helpers | |

`results/` and `figures/` are generated and not tracked. Each CSV in `results/tables/` holds the
numbers of one table of the article, named after its LaTeX label; figure files keep the names
used in the manuscript.

## Model

`Python/blmc/sampler.py` fits

```math
\mathbf{y}_i = \mathbf{x}_i^\top \boldsymbol{\beta} + \mathbf{f}_i^\top \boldsymbol{\Lambda} + \mathbf{e}_i, \qquad
\mathbf{e}_i \mid w_i \sim \mathcal{N}_q\left(\mathbf{0}, \frac{a_{g(i)}}{w_i} \boldsymbol{\Sigma}\right), \qquad
w_i \sim \mathrm{Gamma}\left(\frac{\nu}{2}, \frac{\nu}{2}\right)
```

for the standardized log concentrations $\mathbf{y}_i$ of the $q$ elements at site $i$, with
Matérn 3/2 Gaussian-process factors $\mathbf{f}_i$, land-use variance scales $a_g$ (agricultural
soils fixed at $a_g = 1$) and right-censoring of Cu and Sn at their upper reporting limits.
Integrating out $w_i$ gives a multivariate Student-*t* local error with $\nu$ degrees of freedom.
The spatial factor part follows Zhang and Banerjee (2022); the Student-*t* errors, land-use
scales and censoring are our extensions.

Two comparisons check the structure of the model. The univariate baseline fits the same model
to each element on its own ($K = 1$) in the same cross-validation, so the probability of
exceeding at least one NGR is computed as if the elements were independent.
`Python/blmc/sampler_groups.py` lets the land-use scale differ by element group,
$\mathbf{e}_i \mid w_i \sim \mathcal{N}_q(\mathbf{0}, D_{g(i)} \boldsymbol{\Sigma} D_{g(i)} / w_i)$,
with lithogenic, phosphate-related and sulfide-related groups taken from the clusters of the
spatial correlations; it enters as one more sensitivity refit.

## Data

`data/raw/` holds the prediction grid and the land-use layers used to assign the land use of
sites and grid cells. The survey results (`data/raw/samples/`) are not distributed; they are
available from the corresponding author on request, and the land-use layers of the sites keep
only their geometry and distance. Paths and study settings (CRS, land-use classes, NGR,
elements, models, chain lengths and output folders) are in `Python/blmc/config.py`.

## Dependencies

Python 3.11 with the packages in `requirements.txt`.

## Model reference

Zhang L, Banerjee S (2022). Spatial factor modeling: a Bayesian matrix-normal approach for
misaligned data. *Biometrics* 78:560–573.
