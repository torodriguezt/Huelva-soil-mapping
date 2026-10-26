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
appendices. The cross-validation (120 fits) takes about 8 hours and the full-data fits about
1.5 hours on 20 threads. Python scripts can also be run one by one from the root after setting
`PYTHONPATH=Python`, for example

```sh
python Python/All_Models/fit_model.py --model t_K3_ngr
python Python/All_Models/cross_validation.py --plan rerun_plan.json
```

Finished fits are skipped, so an interrupted run can be resumed.

## Structure

| Folder | Content | Output |
| --- | --- | --- |
| `Python/Data_preparation/` | samples, land use of sites and grid, standardized model input | `results/data/` |
| `Python/All_Models/` | full-data fits and spatial block cross-validation | `results/cv/`, `results/fits/` |
| `Python/Robustness/` | prior and record sensitivity | `results/fits/sensitivity/` |
| `Python/Results/` | grid predictions, exceedance areas, residual checks, tables and figures | `results/`, `figures/` |
| `Python/blmc/` | sampler, prediction, convergence diagnostics and shared helpers | |

`results/` and `figures/` are generated and not tracked. Each CSV in `results/tables/` holds the
numbers of one table of the article, named after its LaTeX label; figure files keep the names
used in the manuscript.

## Model

`Python/blmc/sampler.py` fits

```text
Y_i = X_i beta + F_i Lambda + E_i,   E_i | w_i ~ N(0, a_g Sigma / w_i),   w_i ~ Gamma(nu/2, nu/2)
```

for standardized log concentrations, with Matérn 3/2 Gaussian-process factors F, land-use
variance scales a_g (agricultural soils fixed at 1) and right-censoring of Cu and Sn at their
upper reporting limits. The spatial factor part follows Zhang and Banerjee (2022); the
Student-t errors, land-use scales and censoring are our extensions.

## Data

`data/raw/` holds the survey results for Huelva, Aroche and Campillo, the prediction grid and
the land-use layers. The survey results are not distributed. Paths and study settings (CRS,
land-use classes, NGR, elements, models, chain lengths and output folders) are in
`Python/blmc/config.py`.

## Dependencies

Python 3.11 with the packages in `requirements.txt`.

## Model reference

Zhang L, Banerjee S (2022). Spatial factor modeling: a Bayesian matrix-normal approach for
misaligned data. *Biometrics* 78:560–573.
