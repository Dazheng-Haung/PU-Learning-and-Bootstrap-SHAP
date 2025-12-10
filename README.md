# PU-Learning-and-Bootstrap-SHAP

This project provides a data-driven solution for mineral exploration, specifically addressing the challenges of label scarcity (lack of known barren samples) and model instability in geochemical pattern recognition.

By integrating Positive-Unlabeled (PU) learning with a Bootstrap-SHAP interpretation framework, this code helps geoscientists identify robust indicator elements and quantify the uncertainty of their geological significance.

1. PU_Learning.py (Reliable Negative Selection)
This script solves the "missing negative label" problem common in greenfield exploration.
Method: Implements a Spy-based PU Learning strategy.
Function: It iteratively tests the unlabeled data to filter out potential hidden deposits.
Output: Identifies a high-confidence set of Reliable Negative (RN) samples to construct a clean, low-bias training dataset (P+RN).


2. Bootstrap-SHAP.py (Robust Selection & Deep Interpretation)
This script performs the core feature selection and geological interpretation using the cleaned dataset.
Robust Selection: Executes Recursive Feature Elimination (RFE) across hundreds of Bootstrap iterations. It calculates the Inclusion Probability (IP) for each element to distinguish stable pathfinders from random noise.
Deep Interpretation: Applies GAM-on-SHAP and interaction analysis to visualize non-linear geochemical thresholds and quantify the uncertainty of the model's geological insights.

This repository contains the source code and data for the research paper: "Robust and Uncertainty-Aware Selection of Geochemical Indicator Elements via PU-Learning and Bootstrap-SHAP: A Case Study in Northwest Sichuan, China." 
