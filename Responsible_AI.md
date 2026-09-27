# Responsible AI & Ethical Governance Checklist

## 1. Fairness & Equity Audit

- **Protected Groups Evaluated:** Room Types
  (Entire home/apt, Private room, Hotel room, Sha2red room)
  and 15 Albany Neighbourhood Wards.
- **Fairness Goal:** Maintain equal prediction accuracy
  (MAE/RMSE) across all room categories and neighbourhood wards.
- **Bias Mitigation:** Applied pre-processing reweighing
  to adjust sample weights.

## 2. Privacy & Anonymization

- **Host Anonymization:** Removed sensitive personal identity
  attributes (`host_name`, `host_about`, `host_thumbnail_url`).
- **Data Safeguards:** Prevented direct memorization or extraction
  of host personal data.

## 3. Transparency & Explainability

- **Global & Local Interpretability:** Integrated SHAP waterfall
  plots for per-prediction feature attribution.
- **Model Disclosures:** Clear documentation of model limitations
  and regression target metrics (R² >= 0.85).

## 4. Open Data & Consent

- **Data Source:** Datasets obtained from Inside Airbnb portal
  operating under open-data public domain access.
- **User Consent:** Complies with non-commercial educational
  research licensing.
