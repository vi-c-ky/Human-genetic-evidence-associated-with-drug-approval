# Human Genetic Evidence Associated with Drug Approval

A reproducible computational analysis of the relationship between human genetic evidence and clinical drug approval using Open Targets and ChEMBL.

This repository accompanies the manuscript:

> **Human Genetic Evidence Associated with Drug Approval**  
> Victoria Paterson (2026)

The study evaluates whether genetically supported target–disease pairs are enriched among approved drugs, explores evidence-type contributions using interpretable machine learning, and analyses translational implications for target prioritisation.

---

## Overview

Using integrated target–disease evidence from the Open Targets Platform and drug-development data from ChEMBL, this project:

- constructs a curated dataset of 26,278 target–disease pairs
- evaluates enrichment of genetic evidence among approved drugs
- trains interpretable classification models
- analyses feature importance using SHAP
- investigates literature-mining leakage effects
- performs temporal validation
- releases 1,433 genetically supported Phase 1/2 target–disease pairs as a hypothesis-generating resource

---

## Main Findings

- Genetic evidence is enriched among approved drug targets:
  - pair-level OR = 3.25
  - target-level OR = 2.79
- Oncology exhibits the strongest enrichment signal.
- Literature-derived features dominate predictive performance, likely due to post-approval publication leakage.
- Genetic evidence behaves more like a binary enrichment signal than a strong continuous predictor.

---

## Repository Structure

```text
Human-genetic-evidence-associated-with-drug-approval/
│
├── data/
│   ├── processed/        # Curated datasets used in analyses
│   ├── candidates/       # Phase 1/2 candidate target–disease pairs
│   └── raw/              # Optional raw/intermediate files (excluded from git if large)
│
├── notebooks/            # Exploratory analyses and figure generation
├── scripts/              # Main preprocessing and modelling scripts
├── figures/              # Manuscript figures
├── results/              # Model outputs and summary statistics
├── manuscript/           # Manuscript PDF and source files
├── requirements.txt
├── LICENSE
└── README.md
```

## Reproducibility

Environment

Recommended Python version:

Python 3.11+

Main packages:
```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
scipy
```
Install dependencies:
```
pip install -r requirements.txt
```
Data Sources

This project uses publicly available data from:

```
Open Targets Platform
ChEMBL (v33)
```
Please cite the corresponding resources if reusing the data or code.

Running the Pipeline
```
1. Preprocess data
python scripts/preprocess.py
2. Train models
python scripts/train_models.py
3. Generate figures
python scripts/generate_figures.py
```
## Outputs

The repository includes:

processed target–disease datasets
enrichment analyses
model performance metrics
SHAP analyses
candidate Phase 1/2 target lists
manuscript figures and supplementary outputs
Key Methodological Notes
The analysis is observational and retrospective.
Literature-mining features likely contain post-approval temporal leakage.
Temporal validation partially mitigates, but does not eliminate, this issue.
Current Open Targets evidence scores reflect present-day knowledge rather than historical snapshots.
Models are intended for evidence-structure analysis and hypothesis generation rather than clinical prediction.
Manuscript Abstract

Human genetic evidence has repeatedly been associated with increased probability of drug approval, but its relative contribution within modern multi-source target prioritisation frameworks remains unclear. Using Open Targets evidence scores and ChEMBL clinical status annotations, we analysed 26,278 target–disease pairs spanning approved and investigational programmes. Genetically supported pairs showed significant enrichment among approved drugs (pair-level OR = 3.25; target-level OR = 2.79), with particularly strong effects in oncology. Interpretable machine-learning models identified literature-derived features as dominant contributors to classifier performance; however, temporal validation and feature ablation analyses suggested this largely reflected post-approval publication leakage. Excluding literature features, genetic evidence retained modest predictive signal above baseline but behaved primarily as a binary enrichment marker rather than a strong continuous discriminator. We release 1,433 genetically supported Phase 1/2 target–disease pairs as a hypothesis-generating resource for translational investigation. These findings are observational and do not support clinical deployment of the predictive models.

## Limitations

Important limitations include:

retrospective study design
temporal contamination in literature-derived evidence
possible target-selection confounding
target-level non-independence
incomplete failure ascertainment
modest predictive performance

The findings should therefore be interpreted as evidence-structure analyses rather than causal estimates of drug-development success.
`
## Citation

If you use this repository, please cite:
```
@article{paterson2026geneticsapproval,
  title={Human Genetic Evidence Associated with Drug Approval},
  author={Paterson, Victor},
  year={2026}
}
```
License

This project is released under the MIT License.

## Acknowledgements

Data were obtained from the Open Targets Platform and ChEMBL database
