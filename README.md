# Architecture Search for sEMG-to-Keystroke Decoding

**Group Members:** Dyson Lewis, Anthony Tsai  
**Course:** UCLA ECE C147/C247 — Winter 2026

---

## Overview

This project investigates neural architectures for decoding keystrokes from surface
electromyography (sEMG) signals using the emg2qwerty dataset, subject #89335547.
Starting from the TDS convolution baseline provided in the course codebase, we
iteratively refine preprocessing, architecture, and hyperparameters to minimize
Character Error Rate (CER).

Our final model, a hybrid CNN--LSTM with beam search + language model decoding,
achieves a test CER of 7.61, compared to the baseline CER of ~30.

---

## Results Summary

| Model | Val CER | Test CER |
|---|---|---|
| TDS Baseline (greedy) | 18.50 | 21.14 |
| CNN-LSTM Balanced K=31 (greedy) | 11.25 | 12.23 |
| CNN-LSTM Balanced K=31 (beam + LM) | 9.35 | **7.61** |

---

## Approach

1. **Preprocessing sweep** — FourierFeatures vs. LogSpectrogram, STFT hop length,
   temporal resampling, and dropout tuning on the TDS baseline
2. **Architecture sweep** — eight encoder designs (pure CNN, GRU, LSTM, Transformer,
   and RNN-frontend variants) at matched ~5.3M parameters and 45-minute training budget
3. **Cross-validation** — three-stage sweep over CNN/LSTM parameter ratio and kernel
   size, then learning rate, then dropout on the best candidate

---

## Repository Structure
```
emg2qwerty/          # base codebase from course
CNN_arch_sweep_full.csv   # full training logs for all architecture sweep runs
```

---

## Requirements

Follow the setup instructions in the base course repository. All experiments were
run on an NVIDIA L4 GPU via Google Colab.
