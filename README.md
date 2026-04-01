# Learning Agency Lab – Automated Essay Scoring 2.0 (Kaggle)

This repository contains my solution to the **Learning Agency Lab – Automated Essay Scoring 2.0** Kaggle competition. The goal is to automatically score student-written essays, helping reduce grading time while enabling more timely, consistent feedback.

🏅 **Result: Silver Medal (112 / 2706 teams)**  
📜 Certificate: https://www.kaggle.com/certification/competitions/chenc25/learning-agency-lab-automated-essay-scoring-2

---

## 🧠 Problem Overview

Essay writing is a strong indicator of student learning, but grading essays is time-consuming and expensive for educators. This competition focuses on building an **automated essay scoring (AES)** model that predicts a score on a **1–6 ordinal scale**, evaluated using **Quadratic Weighted Kappa (QWK)**.

The dataset includes realistic classroom writing samples across diverse student populations.

---

## ⚙️ Solution Summary

This solution follows an end-to-end ML workflow:

### 1) Feature Engineering
- Text preprocessing + feature caching
- TF-IDF / count-based features
- Additional engineered linguistic/statistical features

### 2) Modeling
A hybrid ensemble of:
- **LightGBM Regressor** with a custom QWK-optimized objective
- **XGBoost Regressor** with a custom QWK-optimized objective
- Weighted averaging ensemble for improved generalization

### 3) Training Strategy
- **15-fold Stratified Cross-Validation**
- Early stopping
- GPU acceleration (when available)

### 4) Post-processing
- Continuous regression outputs are converted into discrete labels (1–6)
- Thresholds are tuned to maximize QWK on OOF predictions

---

## 📊 Results

- **Mean CV QWK:** ~0.847  
- **Mean CV F1 (weighted):** ~0.690  
- **Final Rank:** **112 / 2706** (🥈 Silver Medal)

---

## 🗂 Repository Structure

```text
├── notebooks/              # EDA and experiments
├── src/                    # Feature engineering + training/inference scripts
├── models/                 # Saved models (optional)
├── submissions/            # Submission files
├── requirements.txt        # Python dependencies
└── README.md
