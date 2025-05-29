# XAI-Internship

Welcome to the **XAI-Internship** repository. This project **explores explainable artificial intelligence** (XAI) in the context of **speech enhancement**, applying interpretability techniques to deep models like DCCRN to better understand how they process and denoise audio signals.

## Project Goal

- Explore and compare explainability (XAI) and sensitivity analysis methods for a deep speech enhancement model (NsNet2).

- Apply model-agnostic (e.g., SHAP, LIME) and gradient-based techniques (e.g., FGSM, Deep SHAP, Gradient SHAP) to understand model behavior.

- Use time-frequency spectrograms as input features and analyze the impact of different input regions on the model’s predictions.

- Investigate how feature definitions (individual bins, grouped regions, or latent features) affect interpretability.

- Provide visual and quantitative insights into which parts of the input the model relies on during noise suppression.


## Repository Structure

### `DeepShap/` 

We use [Captum's DeepLiftShap](https://captum.ai/api/deep_lift_shap.html) to compute **feature attributions** over frequency bands for a speech model (DCCRNet). The procedure works as follows:

1. **Wrap the model** with a custom class (`BandFeatureModel`) that extracts band-level magnitude features from the output.
2. **Provide a baseline** (zero signal) and compute attributions for each frequency band target.
3. **Average the attributions over time** and save the results in a JSON file per input audio file.

### 🔍 Example Output Format

```json
{
  "input_file": "00001_p226_001.wav",
  "attributions": {
    "0-1000Hz": 3.68e-06,
    "1000-2000Hz": 6.42e-07,
    ...
  }
}
```


### `papers/`
This directory contains:
- **Reference Papers**: A collection of research articles that inspire the techniques used in this project, including XAI methods and speech enhancement strategies.

## Dataset

The primary dataset used is the [Noisy speech database for training speech enhancement algorithms and TTS models](https://datashare.ed.ac.uk/handle/10283/2791) hosted by the University of Edinburgh.
It provides clean and noisy parallel speech at 48kHz, based on the VCTK corpus with noise from the DEMAND database.


## Requirements

- Python 3.8 or later
- [PyTorch](https://pytorch.org/) (version compatible with your environment)
- Other dependencies listed in `requirements.txt`

### Install dependencies

```bash
pip install -r requirements.txt
```

## Running the main script

The script `DeepShap/main.py` computes Shapley values for frequency-times bins of audio files for NsNet2 model. To run it with your own data:

```bash
python DeepShap/main.py --input_dir data/noisy_trainset_28spk_wav_resampled --divisions 16 32 64
```