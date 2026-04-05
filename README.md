# Robust Housing Price Modeling under Realistic Data Stress

*(Methodology reproduced on the public Ames Housing dataset; project direction inspired by internship work at Shanghai Liancheng Real Estate Appraisal Consulting Co., Ltd.)*

**Author:** Lisha Ma (Zhejiang University)  
**Project focus:** Trustworthy Machine Learning · Stress Testing · Model Selection under Data Stress · Interpretability

---

## Overview

This project began as a standard housing price prediction task and gradually developed into a study of **model reliability under realistic data stress**.

Instead of stopping at average validation RMSE, I asked a broader set of questions:

1. **Which model performs best on clean data?**
2. **Which model remains reliable when the input or supervision is slightly stressed?**
3. **Can a strong model be improved through targeted intervention?**
4. **If one intervention strategy creates a clean–robustness trade-off, should the prediction rule itself become adaptive?**
5. **What structural signals does the final selected model actually learn?**

Because the original internship data are confidential, I reproduced the methodology on the public **Ames Housing** dataset.

The final repository is organized into:

**EDA → Feature Engineering → Model Comparison → Stress Benchmark → Final Prediction + SHAP**

and two separate methodological extensions:

- **single-model robustness intervention**
- **finitely adaptive prediction under missing-feature patterns**

---

## Research Motivation

A conventional applied ML workflow often ends too early:

- train several models,
- choose the one with the lowest validation RMSE,
- export predictions,
- maybe add one feature-importance figure.

I wanted to push the project one step further.

The core idea of this repo is:

> a model that is best on clean validation data is not automatically the best model for practical use.

That pushed me to reframe the project from pure tabular regression into a small study of:

- **failure modes**
- **stress-aware model selection**
- **targeted intervention**
- **interpretation of the final selected model**

---

## Main Questions

This project is organized around four linked questions.

### 1. Clean-model question
Which candidate model family gives the strongest baseline performance on log house price prediction?

### 2. Stress question
How much does each model degrade when I introduce realistic pressure, such as:

- small numeric feature perturbation
- feature-group ablation
- mild target corruption
- subgroup performance imbalance

### 3. Intervention question
Once a strong model family is selected, can I make it **more reliable** instead of just reporting its weaknesses?

### 4. Interpretation question
After selecting the final mainline model, what variables and interactions actually drive its predictions?

---

## Mainline Pipeline

## Candidate models

The clean benchmark compares:

- Ridge
- Lasso
- ElasticNet
- RandomForest
- GradientBoosting
- LightGBM
- XGBoost

## Stress benchmark

All candidate models are then evaluated under a common stress protocol.

The stress suite covers:

- **feature noise**
- **feature-group ablation**
- **target corruption**
- **subgroup reliability**

A joint ranking combines:

- clean predictive quality
- stress robustness

## Mainline model choice

This stage selects **Gradient Boosting** as the strongest mainline model family.

Importantly, this did **not** mean “the model is solved.”  
It meant that Gradient Boosting was the most promising object to analyze further.

## Final mainline model

For the main pipeline, I keep the **plain GradientBoostingRegressor** as the final selected model for:

- final test prediction
- final SHAP interpretation

I made this choice deliberately.

The project contains later intervention experiments, but the mainline should remain clean and coherent:
the selected model, the submitted model, and the interpreted model should be the **same object**.

---

## Feature Engineering Highlights

I used a compact set of domain-informed engineered features, including:

- `TotalSF`
- `TotalBath`
- `HouseAge`
- `YearsSinceRemod`
- `OverallQual_TotalSF`
- `OverallQual_GrLivArea`
- `Bsmt_Ratio`
- `AreaPerRoom`

These are not decorative additions.  
The final interpretation shows that several engineered variables became central drivers of the model.

---

## Mainline Findings

### 1. Clean-best and reliable-best are not automatically the same

The project started with standard model comparison, but the later benchmark showed that model selection becomes more meaningful once stress is included.

This was the first major shift in my thinking:

**accuracy alone is not enough.**

### 2. Gradient Boosting offered the strongest overall mainline balance

Across clean performance and stress-aware comparison, Gradient Boosting remained the most attractive mainline family.

This made it the right candidate for:
- final prediction
- interpretation
- and later extension experiments

### 3. The final model is driven mainly by quality–space interactions

SHAP results show that the most influential features are led by:

- `OverallQual_TotalSF`
- `OverallQual_GrLivArea`
- `TotalSF`
- `OverallQual`
- `OverallCond`
- `GrLivArea`

This means the model is not just rewarding larger houses, or better-quality houses, in isolation.  
It relies strongly on the interaction between **how much space a house has** and **how good that space is**.

### 4. The final model still uses a broad long tail of features

The SHAP beeswarm shows a large “Sum of other features” contribution.

I interpret this as:

- the model has a clear structural core
- but it still distributes smaller predictive signal across many secondary variables

So the model is both:
- interpretable at the top
- and realistically multifactorial underneath

### 5. The learned effects are nonlinear but domain-consistent

The dependence plots show that:

- larger `GrLivArea` and `TotalSF` generally increase predicted log price
- higher `OverallQual` and `OverallCond` push predictions upward in a stepwise way
- interaction terms such as `OverallQual_TotalSF` and `OverallQual_GrLivArea` become especially valuable at the upper end

This suggests that the final model captures meaningful nonlinear structure rather than arbitrary noise.

---

## Extension 1: Single-Model Robustness Intervention

After the stress benchmark, I explored whether the selected GB family could be improved **inside a single model**.

The first intervention stage compared several targeted robustness variants, including:

- noise augmentation
- feature masking
- coupled hybrid augmentation

## What this extension taught me

This extension was valuable not because it produced a perfect winner, but because it exposed a real methodological tension:

> improving robustness to difficult feature conditions can hurt clean RMSE.

In other words, it is possible to reduce sensitivity to some failure modes while also damaging the original prediction task.

That was an important result.

It showed me that robustness is not just a matter of “adding more augmentation.”  
The real challenge is:

- how much robustness to add,
- under which failure mode,
- and at what cost to clean performance.

This was the second major shift in my thinking:

**intervention must be targeted, and trade-offs must be taken seriously.**

---

## Extension 2: Finitely Adaptive Prediction under Missing-Feature Patterns

Because the single-model intervention revealed a strong clean–robustness trade-off, I then explored a more adaptive formulation inspired by the literature on **prediction with missing data**.

The idea was simple:

- keep one strong full-feature GB for clean cases
- train additional fallback GB models for specific missing-feature patterns
- route prediction according to the observed missingness pattern

In this repository, I tested pattern-aware fallbacks for:

- missing garage features
- missing basement features
- missing garage + basement features

## Why this matters

This extension changed the question from:

> “Can one static model handle everything?”

to:

> “Should the prediction rule itself adapt when the available information changes?”

That reformulation was important.

Empirically, the finitely adaptive version improved RMSE under the missing-feature scenarios, especially when the missingness pattern became more severe, while keeping clean performance essentially unchanged.

This became the most convincing intervention-style result in the project.

---

## What I Learned from the Two Intervention Attempts

The two extensions taught me something more valuable than a single “improved model.”

### Lesson 1
A robustness problem can be real even when a simple fix does not win the final joint ranking.

### Lesson 2
Single-model intervention is useful for diagnosing the trade-off, but it may not be the right final formulation.

### Lesson 3
For structured missing-feature scenarios, an **adaptive prediction rule** can be more natural than forcing one static model to absorb every failure mode.

This was the point where the project stopped being only “I can use ML tools” and started becoming:

> I can turn an application problem into a small methodological question.

---

## Limitations

This is still an exploratory project, not a full paper.

Important limitations remain:

- the stress protocol is handcrafted rather than theoretically optimal
- the robustness score is a practical design choice, not a universal standard
- the single-model intervention family is still limited
- the finitely adaptive extension only covers a small set of missing-feature patterns
- the dataset is public and tabular, so the setup is still far from true high-stakes deployment

These limits are real, but they also make the next steps concrete.

---

## Future Directions

The next directions I find most interesting are:

### 1. More principled missing-feature adaptation
The finitely adaptive GB extension worked better than the earlier single-model intervention.  
A natural next step would be a broader missingness-aware prediction framework.

### 2. Better stress-aware objective design
The current ranking scheme is useful, but still heuristic.  
A more principled objective could align model selection more closely with specific deployment risks.

### 3. Stronger subgroup-aware reliability
The current subgroup analysis is mainly diagnostic.  
A more direct optimization target for worst-group reliability would be valuable.

### 4. Bridging robustness and interpretability
One question that now interests me more is:

> when a model becomes more reliable under stress, does its reliance structure also become more stable and more interpretable?

That feels like a meaningful bridge between applied modeling and trustworthy machine learning.

---

## Repository Structure

### Mainline notebooks
- `01_eda.ipynb` — exploratory data analysis
- `02_feature_engineering.ipynb` — preprocessing and handcrafted features
- `03_model_selection.ipynb` — clean model comparison
- `04_stress_benchmark.ipynb` — all-model stress benchmark
- `05_final_prediction_and_shap.ipynb` — final prediction export + SHAP interpretation

### Extension notebooks
- `06_extension_single_model_intervention.ipynb` — single-model robustness intervention inside GB
- `07_extension_finitely_adaptive_missingness.ipynb` — pattern-adaptive GB under missing-feature scenarios

### Supporting modules
- `src/eda.py`
- `src/feature.py`
- `src/model.py`
- `src/stress_utils.py`
- `src/stress_protocol.py`
- `src/robust_score.py`
- `src/shap_utils.py`
- `src/interventions.py`
- `src/adaptive_gb.py`

---

## Final Takeaway

This project is not just “Ames house price prediction with Gradient Boosting.”

A more accurate summary is:

> I built a full pipeline that moved from clean prediction to stress-aware model selection, then used two different intervention attempts to study reliability under difficult feature conditions, and finally interpreted the mainline selected model with SHAP.

The most important outcome is not one single RMSE number.

It is the change in how I now think about modeling:

- not only which model wins
- but how models fail
- how failure modes should be tested
- when intervention creates trade-offs
- and when the prediction rule itself should become adaptive

That is the part of the project I value most.
