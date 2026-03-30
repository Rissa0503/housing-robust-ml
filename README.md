# Robust Housing Price Modeling under Imperfect Data
(Developed during internship at Shanghai Liancheng Real Estate Appraisal Consulting Co., Ltd.)) 

**Author:** Risa (Zhejiang University)
 
**Focus:** Robust Machine Learning · Interpretability · Imperfect Data  

---

## 🔍 What This Project Is About

This project is extended an industry-driven housing price prediction task into a research-oriented study on model behavior under imperfect data conditions.

Instead of only asking:

> “Can we predict housing prices accurately?”

I focus on a more research-oriented question:

> **Can a model remain reliable under imperfect data and imperfect supervision?**

---

## Project Overview

Due to confidentiality constraints of real-world data from my internship, I reproduced the methodology using the public Ames dataset to demonstrate the modeling and analysis pipeline:

- **EDA → Feature Engineering → Model Comparison → SHAP Interpretation → Stress Testing**

The core model is:

- `GradientBoostingRegressor` (strong nonlinear baseline)

---

## Key Contributions

### 1. Structured Feature Engineering

- Domain-driven features:
  - `TotalSF`, `TotalBath`
  - Interaction terms: `OverallQual × Area`
- Temporal signals:
  - `HouseAge`, `YearsSinceRemod`
- Ratio & aggregation features

The model learns **compositional structure**, not isolated variables.

---

### 2. Model Interpretability (SHAP)

I use SHAP to understand:

- which variables drive predictions
- how feature values shift outputs
- whether results align with domain intuition

**Key finding:**

> The model is dominated by interaction effects between **quality and space**, rather than single variables.

---

### 3. Stress Testing under Imperfect Conditions

This is the core research-oriented component.

I evaluate robustness under:

#### (A) Imperfect Data
- feature noise (3%, 5%)
- missing feature groups (garage / basement)
- subgroup heterogeneity (price segments)

#### (B) Imperfect Supervision
- target corruption (Gaussian / subset noise)

---

## Main Findings

### 1. Strong Local Stability
- Small feature perturbations → almost no performance drop  
→ model is **robust to measurement noise**

---

### 2. Feature Dependence Is Structured
- Removing garage features → largest degradation  
→ model relies on **high-value feature groups**

---

### 3. Subgroup Weakness
- Performance is significantly worse for **low-price houses**  

→ average metrics hide **distributional imbalance**

---

### 4. Mild Supervision Noise Tolerance
- Small target corruption → minimal impact  

→ model shows **initial robustness under imperfect supervision**

---

## Why This Matters

This project reframes a standard regression task into a **trustworthy ML problem**:

- not only *accuracy*
- but also *stability*, *interpretability*, and *failure modes*

Although this is a tabular regression setting, the same concerns arise in:

- noisy-label learning  
- weak supervision  
- real-world deployment  

---

## Connection to My Research Interest

This project reflects my interest in:

> **Robust machine learning under imperfect data and imperfect supervision**

In particular:
- how models behave when assumptions are violated  
- how to diagnose failure modes beyond average metrics  
- how interpretability and robustness interact  

