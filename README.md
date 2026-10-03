# CABLE: Efficient Local–Global Operators for Edge Ultrasound

This repository contains preliminary research code for **CABLE (Context-Aware Boundary and Local Evidence)**, a lightweight high-resolution interaction family for efficient ultrasound models.

CABLE uses two closely related operator variants:

- **CABLE-Local** retains structured local feature interaction at the highest-resolution stage.
- **CABLE-Global** augments the same local interaction with a low-cost pooled global-context branch.

The paper evaluates these operators for breast-ultrasound lesion segmentation on **BUS-BRA** and **BUS-UCLM**, with comparisons against full local-global interaction, Identity, and CBAM, and also studies transfer to a compact **MobileNetV2** point-of-care ultrasound classification model.

## Code status

The current repository is a **preliminary notebook-level release** intended to expose the main CABLE implementation and experimental workflow. It has not yet been reorganized into the final reproducibility package. The cleaned release will separate model definitions, dataset preparation, training, evaluation, and figure-generation code, and will document the exact paper configurations more systematically.

The datasets are not redistributed here and should be obtained from their official sources.

## Suggested entry point

`CABLE_CoreOperator_TwoDataset_Preliminary.ipynb`

This notebook contains the core CABLE operator, the two breast-ultrasound data pipelines, training/evaluation logic, and the principal paper-facing analysis used during development.

## Paper

**CABLE: Efficient Local–Global Operators for Edge Ultrasound**

A complete citation and finalized reproducibility package will be added after the paper workflow is complete.
