#gated_graded_information_transfer

This repository contains models and analysis code for Brown*, So* et al. 2025, bioRxiv.

Each folder contains the code to produce each panel of the figure. To produce the PSTH plots (Fig2C, Fig2E, Fig3C, and Fig3E), the simulation parameter at the start of the file should be set to 'PSTH'. To produce the heatmap plots, the files (Fig2C.py, Fig2E.py, Fig3C.py, and Fig3E.py) should be reran with the simulation parameter set to 'HeatmapData' to save the data in the pickle files needed to run Fig2D.py, Fig2F.py, Fig3D.py, and Fig3F.py.