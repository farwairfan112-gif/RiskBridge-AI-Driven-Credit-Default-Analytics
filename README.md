# RiskBridge-AI-Driven-Credit-Default-Analytics
**RiskBridge**: LendingClub Loan Default Prediction Engine

## Project Overview
The given project converts the large LendingClub dataset into a Business Decision Engine suitable for production. It goes beyond raw probability predictions to enforce strict financial guardrails by processing more than 1.3M past records, through a custom layer called Nuclear Auditor.

## Dataset & Preprocessing
- **Scale**: 2.2M+ initial records processed, and 1.34M completed loans (Fully Paid vs. Charged Off).  
- **Normalization**: Applied log-transformations to skewed income values and robust scaling for outliers.  
- **Reliability**: Removed 'current' loans to ensure training only on definitive historical outcomes.  

## Model & Methodology
- **Hybrid Ensemble**: Voting Classifier combining **Logistic Regression** (linear stability) and **XGBoost** (non-linear patterns).  
- **Audit Discovery**: Identified critical **'Median Bias'**, where standard models over-approve high-risk loans by defaulting missing data to averages.  
- **Nuclear Auditor**: Production script overriding AI over-confidence, enforcing bank-aligned, conservative decisions. 

## Key Innovations & Features
- **Gatekeeper Sync**: FICO Ceiling (36.5% model weight) is a model synchronized directly with user inputs.  
- **Stress-Test Imputation**: Introduced background information to simulate recent inquiries and lower asset depths to get rid of the benefit of the doubt.  
- **DTI Safety Rail**: Hard coded penalty on Debt-to-Income (DTI) above 40, consistent with banking rules.  
- **Edge Case Fix**: Adjusted a high-risk scenario ($25k loan on $30k income) from misleading 14% → 60.65% rejection score.  

## Business Impact
RiskBridge transforms AI from a simple mathematical predictor into a Business Policy Engine. It allows financial institutions to leverage high-performance ML while maintaining human-in-the-loop safety for responsible lending.

## How to Use

### Install Requirements
```bash
pip install pandas numpy scikit-learn xgboost joblib notebook 
```

### Required Assets
- riskbridge_hybrid_model.pkl
- riskbridge_scaler.pkl
- riskbridge_features.pkl
- riskbridge_medians.pkl
- calculator.ipynb (The Nuclear Auditor notebook)

### Run the Auditor

1. Launch Jupyter Notebook in your project folder:
```bash
jupyter notebook
```
2. Open calculator.ipynb in your browser.
3. Follow the instructions in the notebook to input data and calculate loan default risk.

**Important:** The full cleaned CSV (~522MB) is not included due to size limits. 
All `.pkl` models are pre-trained and can be used directly with `calculator.ipynb`.

