# Robust Housing Price Modeling under Imperfect Data

*(Developed during internship at Shanghai Liancheng Real Estate Appraisal Consulting Co., Ltd.)*

**Author:** Risa (Zhejiang University)  
**Focus:** Robust Machine Learning · Interpretability · Imperfect Data  

---

## Overview

This project extends a real-world housing price prediction task into a study of **model behavior under imperfect data and supervision**.

Instead of focusing solely on predictive accuracy, I examine:

- how stable model predictions are under feature imperfections,
- how performance varies across data segments,
- and what drives model decisions beyond average metrics.

Due to confidentiality constraints, the methodology is reproduced on the public Ames dataset.

Pipeline:
EDA → Feature Engineering → Model Comparison → SHAP Interpretation → Stress Testing

---

## Model

- Main model: `GradientBoostingRegressor`
- Used as a strong nonlinear baseline for studying robustness and interpretability

---

## Key Findings

### 1. Feature Dependence Is Structured

Model predictions are primarily driven by **interaction effects between quality and space**, rather than individual features.

Removing high-value feature groups (e.g., garage) leads to the largest performance degradation, indicating strong reliance on structural variables.

---

### 2. Stability under Feature Perturbation

Small perturbations in numeric features result in minimal performance change.

→ The model is relatively robust to measurement noise.

---

### 3. Subgroup Heterogeneity

Model performance varies significantly across price segments.

In particular, prediction errors are higher for low-price houses.

→ Average metrics mask distributional imbalance.

---

### 4. Sensitivity to Feature Removal

Removing certain feature groups causes disproportionate performance drops.

→ The model is vulnerable to feature shift in structurally important variables.

---

### 5. Mild Supervision Noise Tolerance

Small levels of target corruption do not significantly affect performance.

→ The model shows initial robustness under imperfect supervision.

---

## Interpretation

SHAP analysis reveals:

- strong nonlinear interactions between key variables,
- concentration of importance in a small subset of structural features,
- and consistent alignment with domain intuition (quality, area, age).

---

## Why This Matters

This project reframes a standard regression task into a problem of **model reliability**:

- not only accuracy,
- but also stability, interpretability, and failure modes.

These issues are closely related to:

- learning with noisy labels  
- weak supervision  
- real-world deployment under imperfect data  

---

## Research Direction

This work reflects my interest in:

**Robust machine learning under imperfect supervision**

with a focus on:

- failure modes of learning systems  
- robustness under distribution shift  
- interaction between interpretability and reliability   

*This project is exploratory and aims to bridge practical modeling with questions in robust and trustworthy machine learning.*
