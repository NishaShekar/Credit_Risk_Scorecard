# Credit Risk Scorecard – Probability of Default Model

## Project Overview
This project builds a foundational credit risk scorecard to estimate the Probability of Default (PD) for retail borrowers. The model supports credit decision-making by assigning borrowers to risk bands.

## Dataset
Synthetic retail credit dataset containing borrower demographics, loan details, credit utilization, and repayment behavior.

**Target Variable:** `default_flag`  
**Type:** Binary (0 = Non-Default, 1 = Default)

## Features Used
- Age
- Income
- Employment Years
- Loan Amount
- Loan Tenure
- EMI
- Credit Utilization
- Number of Active Loans
- Past Due (30 / 60 / 90 days)

## Methodology
- Data preprocessing and feature selection
- Standardization using `StandardScaler`
- Logistic Regression for PD modeling
- Model evaluation using AUC
- Risk banding based on PD cut-offs

## Model Performance
- AUC: ~0.70 (varies slightly due to random split)

## Risk Bands
| PD Range | Risk Category |
|--------|--------------|
| < 20% | Low Risk |
| 20% – 50% | Medium Risk |
| > 50% | High Risk |

## Tools & Technologies
- Python (Pandas, NumPy, Scikit-learn)
- Streamlit (Model Interface)
- VS Code

## Key Skills Demonstrated
- Credit risk modeling (PD)
- Feature governance
- Model interpretability
- Risk segmentation
- End-to-end ML pipeline

## How to Run
```bash
python model_training.py
python -m streamlit run app.py
---------------------------------------------------------------

