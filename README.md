# conditional_measures
This repository contains code which can be used to compute estimators of conditional inequality measures qZI and qDI. It also contains implementation of several methods of estimation of conditional quantile function based on quantile regression.
The code included in this repository was used to perform a simulation study and real data analysis described in an article *Estimation of conditional inequality curves and measures via estimating the conditional quantile function* by Alicja Jokiel-Rokita, Sylwester Piątek, and Rafał Topolnicki [arXiv link](https://arxiv.org/abs/2412.20228).

Here are the steps that need to be done to use the code:
1. Dependencies from requirements.txt need to be installed.
2. Directory experiments_results needs to be created in a project folder.

In case of troubles with installing rpy2, the code from files with suffix "_no_r" can be used.

Description of the files:
- mc_fld.py - code estimating the conditional quantile function and conditional inequality curves and measures; the code is written for the simulation study on flattened logistic distribution (FLD).
- run_mc_fld.sh - code running the Monte Carlo simulations.
- truevalue_fld.py - code computing the true values of the conditional inequality measures in FLD simulation study.
- real_data_analysis.ipynb - code performing the analysis of the data from census2000 dataset.
- MakePlot.ipynb - code used to generate boxplots and heatmaps presenting the results of the simulation study.