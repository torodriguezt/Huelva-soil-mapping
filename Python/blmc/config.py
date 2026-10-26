# Paths and settings shared by the scripts.
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

## raw inputs (data/raw/samples is confidential)
DIR_RAW = ROOT / "data" / "raw"
FILE_SITES = {site: DIR_RAW / "samples" / "Datos" / f"{site}Resultados.shp" for site in ["Huelva", "Aroche", "Campillo"]}
FILE_GRID = DIR_RAW / "prediction_grid" / "SinAgua" / "capa_sin_agua.shp"
DIR_LAND_USE_SITES = DIR_RAW / "land_use" / "samples" / "Muestra_part2"
DIR_LAND_USE_GRID = DIR_RAW / "land_use" / "covariates" / "Covariables_muestra_part2"
CRS_UTM = 25829    # ETRS89 / UTM 29N
N_HUELVA = 138

## outputs
DIR_DATA = ROOT / "results" / "data"
DIR_MODEL_DATA = DIR_DATA / "model"
DIR_FITS = ROOT / "results" / "fits"
DIR_SENSITIVITY = DIR_FITS / "sensitivity"
DIR_CV = ROOT / "results" / "cv"
DIR_MAPS = ROOT / "results" / "maps"
DIR_AREAS = ROOT / "results" / "areas"
DIR_CHECKS = ROOT / "results" / "checks"
DIR_TABLES = ROOT / "results" / "tables"
DIR_FIGURES = ROOT / "figures"

## elements (Cr fails the detection screen)
REGULATED = ["As", "Ba", "Be", "Cd", "Co", "Cu", "Hg", "Mo", "Ni", "Pb", "Sb", "Sn", "Tl", "V", "Zn"]
CONTEXT = ["Bi", "Al", "Ga", "Sc", "Th", "P", "U"]
ELEMENTS = REGULATED + CONTEXT
FOCUS = ["As", "V", "Tl", "Pb", "Cu", "Co"]     # exceed their NGR at five or more sites
MINOR = [e for e in REGULATED if e not in FOCUS]
UPPER_LIMITS = {"Cu": 10000.0, "Sn": 100.0}     # mg/kg, upper reporting limits: right-censored

## land-use classes: NGR use and connector layers of sites and grid
LAND_USE = {
    "Industria": ("industrial", "Industry/Industry.shp", "Industria/Industry.shp"),
    "Marsh": ("other", "Marsh/Marsh.shp", "Marsh/Marsh.shp"),
    "Phospho": ("industrial", "Phospho/Phospho.shp", "Phospho/Phospho.shp"),
    "Urban": ("urban", "Urban/Urban.shp", "Urban/Urban.shp"),
    "Agric": ("other", "Agric_2/Agric.shp", "Agric/Agric.shp"),
    "Refineria": ("industrial", "Refinery/Refinery.shp", "Refinery/Refineria.shp"),
    "Bare": ("other", "Bare/Bare.shp", "Bare/Bare.shp"),
    "Park": ("other", "Park/Park.shp", "Natural Park/Park.shp"),
}

## NGR (mg/kg), Annex IV of Decreto 18/2015; Hg at the elemental level, no Cr
NGR = pd.DataFrame({
    "element": ["As", "Pb", "Cu", "Zn", "Cd", "Tl", "Sb", "Hg", "Co", "V", "Ni", "Mo", "Se", "Be", "Ba", "Sn"],
    "industrial": [40, 2750, 10000, 10000, 750, 2.3, 4700, 15, 250, 3650, 10000, 3900, 3900, 1450, 10000, 10000],
    "urban": [36, 275, 3130, 10000, 75, 0.23, 470, 6, 25, 365, 1530, 390, 390, 145, 10000, 10000],
    "other": [36, 275, 595, 10000, 25, 0.23, 90, 6, 24, 50, 1530, 115, 370, 145, 10000, 10000],
})

## land use
NGR_USES = ["industrial", "urban", "other"]
GROUPS = ["Agric", "Urban", "Marsh", "Industria", "Phospho", "Bare"]   # local variance groups
USE6 = ["Urban", "Marsh", "Industria", "Phospho"]                      # Agric and Bare are the reference

## candidate models; their order sets the CV seeds
CANDIDATES = ["t_K3_ngr", "t_K4_ngr", "t_K3_use6", "t_K4_use6", "gauss_K3_ngr", "t_K2_ngr"]
SELECTED_MODEL = "t_K3_ngr"
N_BLOCKS, BLOCK_SEED = 20, 32
VALIDATED_DISTANCE_M = 1618.0   # 90% of held-out sites are this close to a training site

## chain lengths (warm-up, saved draws, thinning)
FULL_ITERATIONS = dict(warmup=10000, samples=8000, thin=5)
SENSITIVITY_ITERATIONS = dict(warmup=6000, samples=6000, thin=4)
CV_ITERATIONS = dict(warmup=3000, samples=2000, thin=4)

## prior sensitivity: name -> options of fit_model.py
SENSITIVITIES = {
    "nu_mean40": ["--nu-prior-rate", "0.05"],
    "nu_exponential": ["--nu-prior-shape", "1", "--nu-prior-rate", "0.1"],
    "group_sd05": ["--group-prior-sd", "0.5"],
    "group_sd2": ["--group-prior-sd", "2"],
    "rho_short": ["--rho-prior-median-km", "0.7"],
    "rho_long": ["--rho-prior-median-km", "6"],
    "keep137": ["--data", "keep137"],
}
