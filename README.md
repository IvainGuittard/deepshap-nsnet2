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

## Compute DeepSHAP Attributions for Audio Files

The script `DeepShap/main_deepshap.py` computes Shapley value–based attributions for each time-frequency bin of input `.wav` audio files using the NSNet2 model. Attributions are stored as HDF5 (`.h5`) files in a structured output directory.

---

### Run the Script

```bash
python DeepShap/main_deepshap.py --input_dir <path_to_input_wav_directory> --noise_type not_added
```

---

### Parameters

- `--input_dir` *(required)*  
  Path to the directory containing `.wav` files to process.

- `--noise_type` *(default: `not_added`)*  
  Type of noise to apply to the input. Choices:
  - `not_added`: Use clean inputs as-is (no noise added).
  - `sinusoidal`: Inject sinusoidal noise. Requires `--freq_ranges` and `--rms_amplitudes`.
  - `reverberation`: Simulate reverberation using room impulse responses. Can customize reverberation time with `--reverberances`.
  - `white`: Add white noise within specific time ranges and amplitudes. Requires `--time_ranges` and `--rms_amplitudes`.

- `--freq_ranges` *(required if `--noise_type` is `sinusoidal`)*  
  One or more frequency ranges for sinusoidal noise, e.g.,  
  `--freq_ranges 1000-2000 3000-4000`.

- `--rms_amplitudes` *(required for `sinusoidal` and `white` noise)*  
  One or more RMS amplitudes to control the energy of injected noise, e.g.,  
  `--rms_amplitudes 0.01 0.05`.

- `--time_ranges` *(required if `--noise_type` is `white`)*  
  One or more time intervals (in seconds) where white noise should be injected, e.g.,  
  `--time_ranges 0.0-1.0 2.0-3.0`.

- `--reverberances` *(optional, default: `0.5`)*  
  Reverberation times (in seconds) to simulate if using `--noise_type reverberation`, e.g.,  
  `--reverberances 0.3 0.8`.

---

### Output

For each input file, an HDF5 file is saved containing DeepSHAP attributions:
```
DeepShap/attributions/raw/{input_basename}.h5
```

---

## Visualisation of the Attributions

The script `DeepShap/plot_all_attributions.py` generates a set of advanced visualizations from precomputed DeepSHAP attributions stored in `.h5` files. These visualizations help interpret how time-frequency input bins contribute to the output spectrogram produced by the NSNet2 model.

---

### Run the Script

You can run the script on a directory or a single `.wav` file. It will automatically find the corresponding `.h5` attribution file in `DeepShap/attributions/tf_attributions_h5py/`.

```bash
python DeepShap/plot_all_attributions.py --input_dir <path_to_wav_or_folder>
```

Examples:
```bash
# Run on a directory of WAV files
python DeepShap/plot_all_attributions.py --input_dir data/noisy_input_tests/

# Run on a single WAV file
python DeepShap/plot_all_attributions.py --input_dir data/noisy_input_tests/p226_016.wav
```

---

### Output

For each input file, the following visualizations are generated:

- **A) Global Influence Maps**  
  Influence of each input time-frequency bin aggregated across all outputs.

- **B) Input-to-Frequency Influence Maps**  
  How each input bin contributes to the output at each frequency (or below a low-frequency threshold).

- **C) Input-to-Time Influence Maps**  
  How each input bin contributes to the output at each time frame.

- **D) Time-Time Correlation Maps**  
  Correlation between input and output time frames, showing temporal dependencies.

All plots are saved in:
```
DeepShap/attributions/plots/{input_basename}_*.png
```

---

### Notes

- The script will skip any `.wav` file if its corresponding `.h5` file is missing.
- It automatically removes corrupted or incomplete `.h5` keys before plotting.


## Analyzing the Effect of Attributions

The script `DeepShap/analyze_attribution_effect.py` evaluates the *causal effect* of high and low DeepSHAP attributions by selectively amplifying corresponding time-frequency bins in the audio input. This allows a quantitative and qualitative analysis of the importance of different input regions for the NSNet2 model.

---

### Run the Script

```bash
python DeepShap/analyze_attribution_effect.py --input_dir <path_to_wav_files> --top_percent <float> --dB_amplification <float>
```

**Arguments:**
- `--input_dir`: Folder containing `.wav` files.
- `--top_percent`: Percentage of top (or flop) attribution bins to select (default: `10.0`).
- `--dB_amplification`: Gain in dB applied to the selected bins (default: `6.0`).

---

### Output

For each input `.wav` file, the script will:

1. **Generate and save amplified audios**:
   - Amplifies time-frequency bins with the **top X% attributions**
   - Amplifies time-frequency bins with the **bottom X% attributions**
   - Saved as:  
     ```
     data/noisy_input_tests/{file}_top_{X}_amplified_{Y}dB.wav  
     data/noisy_input_tests/{file}_flop_{X}_amplified_{Y}dB.wav
     ```

2. **Plot the impact on the NSNet2 predicted masks**:
   - Difference between the predicted mask on the original and amplified signals for both cases
   - Saved in:  
     ```
     DeepShap/attributions/attribution_effects/{file}_top_{X}_{Y}dB_amplified_effect.png  
     DeepShap/attributions/attribution_effects/{file}_flop_{X}_{Y}dB_amplified_effect.png
     ```

3. **Visualize the attribution masks used**:
   - Binary mask showing which bins were amplified for both cases
   - Saved in:  
     ```
     DeepShap/attributions/attribution_effects/{file}_top_{X}_mask.png  
     DeepShap/attributions/attribution_effects/{file}_flop_{X}_mask.png
     ```

---
