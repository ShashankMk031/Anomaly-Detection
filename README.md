1. Model Overview:
- Algorithm: Isolation Forest<br>
- Parameters:
  - n_estimators: 100
  - max_samples: "auto"
  - contamination: 0.1
  - random_state: 42

2. Dataset Summary:
- Dataset: Credit Card Fraud Dataset (Kaggle)
- Features: 30
- Samples: 284,807 (99.9% inlier, 0.1% outlier)
- Preprocessing:
  - Removed missing values
  - Normalized features using RobustScaler
  - Split into 70% training and 30% test sets
  
3. Evaluation Metrics Summary:
- Precision: 0.900
- Recall: 1.000
- F1-score: 0.947 

4. Creating scaler.pkl and model.pkl for API deployement 
5. Delpoyement of the model in local server using Flask