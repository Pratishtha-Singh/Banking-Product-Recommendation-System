# Banking-Product-Recommendation-System
This repository implements an end-to-end pipeline for recommending retail banking products like savings, loans, pensions, credit cards, etc  by combining customer profile and transactional data with tailored machine learning models.

---

## Contents
- [Overview](#overview)
- [Datasets](#datasets)
- [Pipeline](#pipeline)
- [Models Used for Each Task](#models-used-for-each-task)
- [Results & Discussion](#results--discussion)
- [Project Structure](#project-structure)
- [User Interface](#user-interface)
- [Dependencies](#dependencies)

---

## Datasets

### Data Files
The CSV files used in this project are not uploaded to GitHub due to their size. They are available via Google Drive:

**Link:** https://drive.google.com/drive/folders/1cNBnl8jKsHeZNzCEufggGPUPYkDGfUsP?usp=sharing

1. Download the CSV files from the provided Google Drive link.  
2. Place them in their corresponding folders as shown in [Project Structure](#project-structure).  

> **Note:** The project will not function properly—particularly the user interface—without these data files in place.

Within the Drive folder:
- **Final processed datasets:** `notebooks_models_development/dataset/project-dataset`  
- **Original raw datasets:** `notebooks_models_development/dataset/original-dataset-from-kaggle`

---

## Pipeline

![Pipeline Architecture](/pipeline_architecture.jpg)

1. **Data Ingestion & Cleaning**  
   - Load customer profile and transaction data  
   - Handle missing values, correct data types, clip outliers  
2. **Feature Engineering**  
   - Enrich profile features (log-transform income/expense ratios)  
   - Derive RFM (Recency, Frequency, Monetary) metrics  
   - Map MCC codes for merchant category spend  
3. **Branch A: Product Category Prediction**  
   - Multi-label XGBoost classifier to predict core products  
   - Threshold calibration per label for balanced F1 scores  
   - Cascaded loan-matching recommender for pre-qualified customers  
4. **Branch B: Behavioral Segmentation**  
   - KMeans clustering on RFM features for credit-card and direct-debit segments  
   - Credit-card subtype recommendation via KMeans on spend-share features

---

## Models Used for Each Task

| Task                              | Model & Approach                                                                                      |
|-----------------------------------|-------------------------------------------------------------------------------------------------------|
| Product Category Prediction       | MultiOutput XGBoost (one binary estimator per product label), tuned via GridSearchCV                 |
| Customer Segmentation             | KMeans (n=5) on RFM features to define behavioral tiers                                               |
| Loan Products Recommendation      | Rule-based engine using credit score, DTI, and spending thresholds                                    |
| Credit Card Subtype Recommendation| KMeans (n=7) on normalized spend-share features + affinity scoring for subtype mapping                |

---

## Results & Discussion

- **Product Category Model (XGBoost):**  
  - Loans F1 = 1.00; Junior account F1 = 0.78; Savings F1 = 0.62 after threshold tuning  
  - Macro-F1 improved from 0.49 to 0.58 with per-label thresholds  
  - Credit card and direct debit excluded due to poor generalization  

- **Customer Segmentation:**  
  - Four tiers: Strong Recommend (22%), Recommend (31%), Consider (30%), Not Recommend (17%)  
  - Enabled precise targeting for direct-debit offers  

- **Loan Recommendation Engine:**  
  - Pre-qualified 353,440 customers; 89.1% received specific loan offers  
  - Distribution: Personal 74.4%, Auto 8.7%, Business 2.0%, Travel 1.5%, Medical 1.3%, Home Renovation 1.0%, Education 0.2%  

- **Credit Card Subtype Model:**  
  - 71.5% of eligible customers received targeted card offers; others filtered out by affinity threshold  
  - Most common: General Purpose Card (24.9%); Standard Card (1.8%)

---
## Project Structure

1. **Model Development** (`notebooks_models_development/`)  
   Contains all notebooks and code for model training and development.

   ```text
   notebooks_models_development/
   ├── banking_products_category_recommendation_model/
   │   ├── banking_products_category_model.ipynb
   │   ├── best_multilabel_model.pkl
   │   └── customer_data_recommendations.csv
   ├── customer_segmentation_model/
   │   ├── customer_segmentation_model.ipynb
   │   ├── kmeans_customer_segments.pkl
   │   └── transaction_data_recommendations.csv
   ├── credit_card_products_prediction_model/
   │   ├── CreditCardSubtypeRecommendation_Unsupervised.ipynb
   │   ├── CreditCardSubtypeRecommendation_Unsupervised_Documentation.ipynb
   │   ├── kmeans_model_creditcardsubtypeunsupervised.pkl
   │   └── CreditCardSpendingCategories.txt
   ├── loans_products_prediction_model/
   │   ├── loan model.ipynb
   │   ├── loan_recommendations_analysis.png
   │   └── loan_recommendations.csv
   ├── dataset/
   │   ├── initiate_customer_data.ipynb
   │   └── initiate_transaction_data.ipynb
   ├── feature_engineering/
   │   ├── feature_engineer.ipynb
   │   └── clean_customer_dataNEW.csv
   ├── eda/
   │   └── EDA.ipynb
   └── pipeline_design.jpg

2. **User Interface** (`user-interface-streamlit/`)

   This directory contains the Streamlit-based web application for end-users to interact with the recommendation system.

   ```text
   user-interface-streamlit/
   ├── app/                                           # Main application code
   ├── main.py                                    # Streamlit entrypoint
   ├── assets/                                    # Static assets (images, html)
   │   ├── vertical_pipeline.jpg                  # Pipeline visualization
   │   └── pca_4d_interactive.html                # Interactive PCA viz
   ├── utils/                                     # Helper modules
   │   ├── product_category_recommender.py        # Category prediction logic
   │   ├── loan_product_recommender.py            # Loan recommendation logic
   │   ├── feature_engineering.py                 # Shared feature utilities
   │   ├── credit_card_recommender.py             # Credit-card subtype logic
   │   └── customer_segmenter.py                  # Customer segmentation logic
   ├── data/                                      # App data inputs
   │   ├── demo_data/                             # Sample/demo CSVs
   │   └── background_data/                       # Preprocessed data for runtime
   ├── models/                                    # Trained model artifacts
   │   ├── primary_model/                         # XGBoost product category model
   │   ├── loan_model/                            # Loan recommendation model
   │   ├── customer_segment_model/                # KMeans segmentation model
   │   └── credit_card_model/                     # KMeans credit-card subtype model
   └── requirements.txt                           # App-specific Python dependencies

---

## User Interface

A **Streamlit** demo app showcases the full pipeline:

- **Tabs:** Introduction, Architecture, Interactive Demo, Metrics Dashboards  
- **Interactive Workflow:** Select customer → view profile & transactions → step through category prediction, segmentation, and product recommendations  
- **Visualizations:** RFM score charts, cluster assignments, recommendation rationale  

## Dependencies

- Python 3.8+  
- pandas  
- numpy  
- scikit-learn  
- xgboost  
- imbalanced-learn  
- matplotlib  
- seaborn  
- streamlit  
- jupyter  



