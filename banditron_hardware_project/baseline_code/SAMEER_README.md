# Banditron Hardware Baseline Guide

This folder contains the baseline reinforcement learning decoder (Banditron) used to establish performance and hardware efficiency metrics for iBMI decoding.

## Execution Guide

To generate the research evidence and performance report, run the `standardize_data.py` script. 

The script is configured to point directly to the following dataset:
`banditron_hardware_project\baseline_code\datasets\clean_monkey_data.mat`

### Run Command
In your terminal, execute:
```bash
python standardize_data.py
```

It should show you the analysis and comparison below:
```
==================================================
EXPERIMENTAL EVIDENCE: BASELINE VS OPTIMIZED
==================================================
Dataset:    clean_monkey_data.mat
Samples:    739
Classes:    3 (Movement Directions)
--------------------------------------------------
BASELINE (1 channels):
 - Performance (AER):  0.0338
 - Efficiency (MACs):  3
--------------------------------------------------
OPTIMIZED (1 channels):
 - Performance (AER):  0.0271
 - Efficiency (MACs):  3
--------------------------------------------------
CONCLUSION: Masking reduced compute by 50% with an
AER change of -0.0068 [note that this value does change but it should be around the same with each iteration due to variation]
==================================================
```
