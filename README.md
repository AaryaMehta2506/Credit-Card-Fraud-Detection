AI/ML Beginners Project
# Credit Card Fraud Detection

## Project Overview
This project builds a machine learning model to detect fraudulent credit card transactions using real-world data. It includes the full pipeline from data preprocessing, feature scaling, and handling class imbalance to model training, evaluation, and deployment using Streamlit. The dataset is highly imbalanced, with only a small fraction of transactions being fraudulent. The project applies resampling (SMOTE) and trains multiple algorithms such as Logistic Regression, Random Forest, and XGBoost to achieve high precision and recall.

## Dataset
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
File: creditcard.csv  
Details:  
- Total records: 284,807 transactions  
- Fraudulent transactions: 492  
- Features: 30 columns (V1–V28, Amount, Time)  
- Target: Class (0 = Legitimate, 1 = Fraudulent)

## Project Workflow
1. Data Loading and Exploration  
   Load the dataset using pandas, inspect data distribution, and check for missing values.  
2. Preprocessing  
   Drop the Time column, scale the Amount column using StandardScaler, and split the dataset into training and test sets with stratified sampling.  
3. Handling Imbalance  
   Use SMOTE (Synthetic Minority Oversampling Technique) to balance the classes in the training set.  
4. Model Training  
   Train and compare Logistic Regression, Random Forest, and XGBoost models.  
5. Evaluation Metrics  
   Evaluate models using Precision, Recall, F1-score, ROC AUC, and Precision-Recall AUC. The main goal is to reduce false negatives while maintaining high precision.  
6. Model Saving  
   Save the trained model and the fitted scaler as xgb_credit_fraud.model and scaler_amount.pkl.  
7. Deployment with Streamlit  
   A simple Streamlit app is provided for real-time prediction using manual feature input.

## How to Run the Project
pip install pandas numpy scikit-learn imbalanced-learn xgboost streamlit joblib matplotlib  
streamlit run app.py  

## Streamlit App Features
- Input V1–V28 and Amount values manually  
- Predict whether a transaction is fraudulent  
- View fraud probability and classification result  

## Repository Structure
project/  
│  
├── creditcard.csv               (Dataset file)  
├── notebook.ipynb               (Main Jupyter Notebook)  
├── app.py                       (Streamlit application)  
├── xgb_credit_fraud.model       (Trained XGBoost model)  
├── scaler_amount.pkl            (Scaler for Amount feature)  
└── README.md                    (Project documentation)

## Technologies Used
Python 3.x  
Pandas, NumPy  
Scikit-learn  
Imbalanced-learn (SMOTE)  
XGBoost  
Matplotlib  
Streamlit  

## Results
XGBoost achieved the best performance.  
Precision-Recall AUC: approximately 0.98  
ROC AUC: approximately 0.99  
The model effectively detects fraudulent transactions with very low false negatives.

## Contributing
Contributions are welcome!
Feel free to fork the repository, improve the game, and open a pull request. Let's grow this classic game together!

## License
This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Author
**Aarya Mehta**  
🔗 [GitHub Profile](https://github.com/AaryaMehta2506)


