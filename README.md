---
# Agricultural Irrigation Needs Classification
### Machine Learning Project – Imbalanced Learning & Stacking Architecture
---

## Overview
This repository presents an end-to-end machine learning pipeline for predicting **agricultural irrigation needs** (`Low`, `Medium`, `High`).  
The project focuses on **handling extreme class imbalance**, **domain-driven feature engineering**, and **ensemble learning via stacking**.

---

## Objective
Develop a robust classification system capable of accurately predicting irrigation levels, with special emphasis on correctly identifying the rare but critical **`High` irrigation class (3.3%)**.

---

## Dataset

- **Training Data:** Agricultural and environmental observations  
- **Test Data:** Held-out evaluation dataset  

### Data Characteristics
- Numerical features: temperature, humidity, soil moisture, rainfall  
- Categorical features: crop type, soil type, region, growth stage  
- Highly imbalanced target distribution  

---

## Approach

### Validation Strategy
- **Metric:** Macro F1-Score  
- **Method:** 5-Fold Stratified Cross-Validation  
- Ensures stable evaluation under class imbalance  

### Adversarial Validation
- Model: LightGBM  
- ROC-AUC ≈ 0.4995  

**Conclusion:** No distribution shift between training and test data.

---

### Feature Engineering

#### Physics-Based Features
- Total water availability and environmental stress indicators:
```python
Total_Water_Input = Rainfall_mm + Previous_Irrigation_mm
Evaporation_Proxy = (Temperature_C * Sunlight_Hours) / (Humidity + 1e-5)
Moisture_Stress = Temperature_C / (Soil_Moisture + 1e-5)
