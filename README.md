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

We use [Captum's DeepLiftShap](https://captum.ai/api/deep_lift_shap.html) to compute **feature attributions** on time-frequency bins for a speech enhancement model (NsNet2). The setup includes:

1. **Wrapper model**: The model is wrapped using [`MaskFromLogPower`](https://github.com/IvainGuittard/XAI-Internship/blob/main/DeepShap/models/MaskFromLogPower.py), which extracts the predicted time-frequency mask from the log-magnitude spectrogram.
2. **Attribution computation**: A baseline (e.g., zero signal) is used to compute attributions over a batch of inputs.
3. **Storage format**: Attribution scores are saved in `.h5` format. The output is a 4D tensor for each target bin: `[f_out, t_out, f_in, t_in]`.

## Attribution Visualizations

Several plots help interpret the computed attributions. All plotting utilities are available in [`plot_utils.py`](https://github.com/IvainGuittard/XAI-Internship/blob/main/DeepShap/utils/plot_utils.py):

### Global Influence Map

- Function: `plot_global_influence`
- This plot shows the total contribution of each input bin `(f_in, t_in)` to the entire predicted mask.
- It is computed by summing the absolute attributions across all output bins and normalizing the result.
- Useful for identifying the most globally influential regions in the input spectrogram.
### Temporal Influence

- Function: `plot_input_time_influence`
- This plot shows how each input time step `t_in` contributes to each output time step `t_out`.
- The attributions are aggregated over all frequencies to visualize temporal dependencies between input and output.


### Frequency Influence

- Function: `plot_input_freq_influence`
- Shows how each input frequency `f_in` contributes to each output frequency `f_out`, aggregated over all time steps.
- The plot reveals the structure of frequency-to-frequency attribution relationships.

### Temporal Correlation Matrix

- Function: `plot_input_time_correlation`
- Computes the Pearson correlation between input time steps based on their attribution vectors (aggregated over frequencies).
- Highlights redundancy or similarity in the way different time steps contribute to the model’s output.


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

The script `DeepShap/main_deepshap.py` computes Shapley values for time-frequency bins of audio files using the NsNet2 model. Attributions are stored in HDF5 (`.h5`) format for each input file and can be visualized using provided plotting functions.

To run the script:

```bash
python DeepShap/main_deepshap.py --baseline_type zero --input_dir data/noisy_input_tests --noise_type not_added
```

### Parameters

- `--input_dir`: **(Required)** Directory containing input `.wav` audio files to process.

- `--baseline_type`: Type of baseline used for computing attributions. Options:
  - `zero`: Use a zero-valued (silent) input as the baseline.
  - `clean_audio`: Use the clean version of the audio file as the baseline.

- `--noise_type`: Type of noise to consider or add. Options:
  - `not_added`: No noise is added (original input is used as-is).
  - `impulsive`: Add impulsive noise to the input.
  - `sinusoidal`: Add sinusoidal noise with customizable frequency ranges and amplitudes.
  - `reverberation`: Simulate reverberation by convolving input with a room impulse response.

- `--freq_ranges`: *(Optional)* List of frequency ranges (e.g., `1000-2000`) for sinusoidal noise injection. Should be specified if `--noise_type` is `sinusoidal`.

- `--rms_amplitudes`: *(Optional)* List of RMS amplitudes (e.g., `0.01`) for the sinusoidal components. Should be specified with `--freq_ranges`.

- `--reverberances`: *(Optional)* Reverberation time(s) used when `--noise_type` is `reverberation`. Default is `0.5`.
