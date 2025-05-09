# XAI-Internship

Welcome to the **XAI-Internship** repository. This project **explores explainable artificial intelligence** (XAI) in the context of **speech enhancement**, applying interpretability techniques to deep models like DCCRN to better understand how they process and denoise audio signals.

## Repository Structure

### `enhancement/`
This directory includes:
- **DCCRN Model Implementation**: Core architecture for speech enhancement tasks.
- **Training and Evaluation Scripts**: Tools to train and assess the model's performance.
- **Data Handling Utilities**: Functions to load, preprocess, and manage audio datasets.

### `papers/`
This directory contains:
- **Reference Papers**: A collection of research articles that inspire the techniques used in this project, including XAI methods and speech enhancement strategies.

## Project Goal

- Explore and compare explainability (XAI) and sensitivity analysis methods for a deep speech enhancement model (DCCRN).

- Apply model-agnostic (e.g., SHAP, LIME) and gradient-based techniques (e.g., FGSM, Deep SHAP, Gradient SHAP) to understand model behavior.

- Use time-frequency spectrograms as input features and analyze the impact of different input regions on the model’s predictions.

- Investigate how feature definitions (individual bins, grouped regions, or latent features) affect interpretability.

- Provide visual and quantitative insights into which parts of the input the model relies on during noise suppression.

## Dataset

The primary dataset used is the [Noisy speech database for training speech enhancement algorithms and TTS models](https://datashare.ed.ac.uk/handle/10283/2791) hosted by the University of Edinburgh.
It provides clean and noisy parallel speech at 48kHz, based on the VCTK corpus with noise from the DEMAND database.

