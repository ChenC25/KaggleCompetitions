# Learning Agency Lab – Automated Essay Scoring 2.0 (Kaggle)

This repository contains my solution to the **Learning Agency Lab – Automated Essay Scoring 2.0** Kaggle competition. The task is to automatically predict holistic essay scores from raw student-written text, helping reduce grading effort while enabling faster feedback for students.

🏅 **Result: Silver Medal (112 / 2706 teams)**  
📜 Competition Certificate: https://www.kaggle.com/certification/competitions/chenc25/learning-agency-lab-automated-essay-scoring-2

---

## 🧠 Problem Description

Essay writing is a key indicator of student learning, but evaluating essays is time‑consuming and costly for educators. This competition focuses on **automated essay scoring (AES)**, where models must predict human-assigned scores based solely on essay text.

Each essay is scored using a **holistic rubric** on a **1–6 ordinal scale**, and performance is evaluated using **Quadratic Weighted Kappa (QWK)**.

---

## 📊 Dataset Description

The competition dataset consists of approximately **24,000 student-written argumentative essays**, collected from realistic classroom settings and scored by human raters.

### Files

#### `train.csv`
Contains essays and their corresponding scores, used for model training and validation.

**Fields:**
- `essay_id` — Unique identifier for each essay  
- `full_text` — Full essay response text  
- `score` — Holistic essay score on a **1–6** scale  

---

#### `test.csv`
Contains essays used for inference and submission generation.

**Fields:**
- `essay_id` — Unique identifier for each essay  
- `full_text` — Full essay response text  

> Note: The rerun test set contains approximately **8,000 essays**.

---

#### `sample_submission.csv`
Template file demonstrating the required submission format.

**Fields:**
- `essay_id` — Unique identifier for each essay  
- `score` — Predicted holistic score (1–6)

---

## ⚙️ Solution Overview

This solution uses a feature‑based regression approach with model ensembling:

### Feature Engineering
- Text preprocessing and normalization
- TF‑IDF and count‑based features
- Additional engineered linguistic and statistical features
- Feature caching for fast experimentation

### Modeling
- **LightGBM Regressor** with a custom QWK‑optimized objective
- **XGBoost Regressor** with a custom QWK‑optimized objective
- Weighted ensemble of both models

### Training Strategy
- **15‑fold stratified cross‑validation**
- Early stopping
- GPU acceleration when available

### Post‑processing
- Continuous regression outputs mapped to discrete scores (1–6)
- Optimized thresholds learned from OOF predictions to maximize QWK

---

## 📈 Results

- **Mean CV QWK:** ~0.847  
- **Mean CV F1 (weighted):** ~0.690  
- **Final Rank:** 112 / 2706 (🥈 Silver Medal)

---

## 🗂 Repository Structure

```text
├── notebooks/              # EDA and experiments
├── src/                    # Feature engineering, training, inference
├── models/                 # Saved model artifacts (optional)
├── submissions/            # Kaggle submission files
├── requirements.txt        # Python dependencies
└── README.md
``
