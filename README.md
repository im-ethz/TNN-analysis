# TNN-analysis: Analysis of Team Novo Nordisk (TNN) cycling and diabetes data

This repository is used in the analysis of the following manuscript "Glycemic Patterns of Male Professional Athletes With Type 1 Diabetes During Exercise, Recovery and Sleep: Retrospective, Observational Study Over an Entire Competitive Season" by E. van Weenen et al.

Data access and preprocessing is described in the following repository: https://github.com/im-ethz/TNN-data

## Instructions
The code is structured as follows.

As a first step, we used `aggregate_dexcom.py` and `aggregate_trainingpeaks.py` to aggregate the raw trainingpeaks and dexcom data on an athlete-day level, which was useful in our analysis.

The paper consisted of three parts which can be found in their respective jupyter notebooks: descriptive figures (`descriptives.ipynb`), summary statistics of the data (`statistics.ipynb`), and an analysis to associate exercise with dysglycemia (`analysis.ipynb`). The results of these three parts are saved under `results` and their respective subdirectory.

Moreover, each jupyter notebook made use of some helper functions, described in `calc.py`, `config.py`, `model.py` and `plot.py`. 

Unfortunately, it is not possible to run the notebook yourself, as this requires access to the data, and this is currently only possible in limited form to protect privacy of participants. However, all code and there results thereof should be visible in the three jupyter notebooks.

If you have any further questions, do not hesitate to email Eva van Weenen evanweenen@ethz.ch, or the corresponding author of the manuscript.
