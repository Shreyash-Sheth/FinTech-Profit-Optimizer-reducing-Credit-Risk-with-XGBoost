# FinTech-Profit-Optimizer-reducing-Credit-Risk-with-XGBoost
In unsecured lending, avoiding a "Toxic Asset" (Default) is 10x more valuable than finding a "Good Loan." This project built a Profit-Optimised Credit Scoring Model using XGBoost to identify high-risk borrowers in a dataset of 518,000 modern loans (2016–2018).

##  Executive Summary
**Goal:** Maximise net portfolio profit by identifying and rejecting high-risk loans.

In the unsecured lending market, avoiding a "Toxic Asset" (Default) is **10x more valuable** than finding a "Good Loan." A standard accuracy-based model often fails because it treats all errors equally.

This project implements a **Profit-Optimised Credit Scoring Model** using **XGBoost**. By training the model on the financial impact of loans (rather than just binary default status) and applying a 4:1 penalty for missed defaults, the strategy turned a projected portfolio loss of **-€76M** into a profit of **+€8.2M**.

* **Net Value Created:** €84.1 Million (on test set).
* **Data Scale:** 518,000 recent loans (2016–2018) processed via "Big Data" chunking.
* **Key Tech:** Python (XGBoost, Pandas), Seaborn.

---

##  The Financial Impact ("The Money Slide")

*The chart below visualises the core business value. The **Red Zone** (Left) represents the money-losing loans successfully blocked by the model. The **Green Zone** (Right) represents the profitable loans kept.*

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/3a249a65-e717-422e-988b-44ac8ebf052f" />

*(Figure 1: Distribution of Profit/Loss. The model successfully filtered out the majority of "Toxic Assets" that sit in the negative loss zone.)*

### Results Breakdown (Hold-out Test Set of 103k Loans)

| Strategy | Action | Financial Outcome |
| :--- | :--- | :--- |
| **Baseline (Status Quo)** | Approve Everyone | **-€75,972,744 (Loss)** |
| **New Model (XGBoost)** | Reject High Risk | **+€8,219,314 (Profit)** |
| **Net Improvement** | -- | **+€84,192,058** |

---

##  The Business Problem
**Context:** The lender operates in the subprime market with a high default rate (**22.4%**). While interest rates are high (up to 25%) to compensate for risk, the cost of defaults during the 2016-2018 period was eroding all profitability.

**The Asymmetry of Risk:**
* **Rejecting a Good Customer:** Opportunity cost of ~€1,000 (Interest income).
* **Approving a Bad Customer:** Direct loss of ~€10,000 (Principal lost).

**Strategic Pivot:**
Instead of optimising for *Accuracy* (which favours the majority class), we optimised for **Recall on the Minority Class** (catching defaults), utilising `scale_pos_weight=4` in XGBoost to align the model with the financial reality.

---
##  Methodology & Tech Stack

### 1. Data Engineering (Big Data Processing)
* **Source:** LendingClub Dataset (2.2M rows). https://www.kaggle.com/datasets/wordsforthewise/lending-club
* **Technique:** Implemented **Pandas Chunking** to iteratively process the massive dataset (100k rows at a time) without crashing memory.
* **Filtering:** Extracted only "Modern Era" loans (2016–2018) to ensure the model reflects current economic cycles.

### 2. Feature Engineering
* **Target Variable:** Engineered a `Binary Default` target (1 = Charged Off, 0 = Fully Paid).
* **Financial Metric:** Calculated `Actual_Profit = Total_Payment - Funded_Amount` to label every loan as a "Profit Maker" or "Loss Maker."
* **Preprocessing:** Handled missing values via Median Imputation and mapped categorical variables (e.g., Employment Length, Grade) to ordinal scales.

### 3. Machine Learning (XGBoost)
* **Model:** XGBoost Classifier.
* **Parameters:** `scale_pos_weight=4` (Cost-Sensitive Learning), `max_depth=5`, `learning_rate=0.1`.
* **Validation:** 80/20 Train-Test split with Stratified Sampling to maintain the 22% default rate in both sets.

---

##  Key Insights & Drivers

### What Drives Risk?
The analysis identified that internal bank grading is the strongest predictor, but **Home Ownership** (Mortgage vs. Rent) is a massive hidden indicator of stability.

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/1e70a8dc-a69a-4c41-a82d-e0a378a2ce4f" />

*(Figure 2: Top 10 Features driving the XGBoost decision logic.)*

### Model Validation
The model outputs show a clear monotonic relationship between Loan Grade and Default Rate, confirming that the model has correctly learned the "Staircase of Risk."

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/e38eb14e-3988-44c0-9993-3fc1443329eb" />

*(Figure 3: Actual Default Rate by Sub-Grade. The clear upward trend validates data quality.)*
