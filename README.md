# **Fashion Recommendation System - Documentation**

## **📌 Project Overview**
The **Fashion Recommendation System** is designed to recommend fashion products to users based on their preferences and product similarities. This system is built using **Collaborative Filtering (SVD)** and **Content-Based Filtering (TF-IDF + FAISS)**, combined into a **Hybrid Model** to enhance recommendation accuracy.

## **🎯 Objective**
Develop an **AI-powered recommendation system** for a hypothetical e-commerce platform that:
✅ Suggests relevant products to users based on past interactions.  
✅ Finds similar products using product reviews and descriptions.  
✅ Combines user-based and content-based approaches for better accuracy.

## **📂 Folder Structure**
```
📂 recommendation-system/
│── 📜 README.md                  # Project overview & setup instructions
│── 📜 requirements.txt           # Dependencies for running the project
│── 📜 data_preprocessing.py       # Data loading, cleaning & feature extraction
│── 📜 collaborative_filtering.py  # Collaborative filtering model (SVD)
│── 📜 content_based.py            # Content-based model (TF-IDF + FAISS)
│── 📜 hybrid_model.py             # Hybrid recommendation model (Combining both)
│── 📜 train.py                    # Main script to train all models
│── 📜 evaluate.py                 # Model evaluation (MAP, NDCG, MRR)
│── 📜 app.py                      # Flask API for recommendations
│── 📂 data/                        # Store dataset files
│   │── processed_fashion_data.csv  # Processed dataset (cleaned)
│── 📂 models/                      # Save trained models
│   │── collaborative_model.pkl     # SVD trained model
│   │── content_faiss.index         # FAISS index for similarity search
│   │── vectorizer.pkl              # TF-IDF vectorizer
│── 📂 notebooks/                   # Jupyter notebooks for EDA & experiments
│   │── 1_data_preprocessing.ipynb  # Data exploration & cleaning
│   │── 2_model_training.ipynb      # Training collaborative & content-based models
│   │── 3_hybrid_model.ipynb        # Experimenting with hybrid approach
```

## **🛠️ Technology Stack**
- **Python** (Data Processing & ML Models)
- **Surprise Library** (Collaborative Filtering - SVD Model)
- **Scikit-learn & FAISS** (Content-Based Filtering - TF-IDF + Similarity Search)
- **Flask** (API Deployment)

## **📊 Implementation Steps**

### **1️⃣ Data Preprocessing - data_preprocessing.py**
- Load and clean the dataset.
- Handle missing values (fill empty reviews).
- Convert user and product IDs to numeric values.

### **2️⃣ Feature Engineering**
- **Collaborative Filtering:** Uses **user-product interaction** data (ratings).
- **Content-Based Filtering:** Uses **TF-IDF on product reviews** for similarity search.

### **3️⃣ Model Training & Optimization**
✅ **Collaborative Filtering (SVD) - collaborative_filtering.py**
- Used **Singular Value Decomposition (SVD)**.
- Hyperparameter tuning using **Grid Search (n_factors, lr_all, reg_all)**.

  ![WhatsApp Image 2025-02-27 at 18 28 42_f56246a2](https://github.com/user-attachments/assets/4c417491-3ebd-467a-89fa-89e103c9ad84)


✅ **Content-Based Filtering (FAISS) - content_based.py**
- Used **TF-IDF Vectorization** on product reviews.
- Optimized **FAISS parameters (nlist, nprobe)** for fast retrieval.

✅ **Hybrid Model (collaborative + content) - hybrid_model.py**
- Combines both recommendations with adjustable weightage.

### **4️⃣ Model Evaluation - evaluate.py**
Evaluated using:
- **Mean Average Precision (MAP)**
- **Normalized Discounted Cumulative Gain (NDCG)**
- **Mean Reciprocal Rank (MRR)**

![WhatsApp Image 2025-02-27 at 18 30 42_d7e276ee](https://github.com/user-attachments/assets/92ee817b-18f2-47b5-8b57-e601a3cdd139)


### **5️⃣ Deployment - app.py **
The trained model is exposed as an **API endpoint**:
- **GET /recommend?user_id=1** → Returns recommendations for a given user.

## **📄 Code Insights & Choices**
### **Collaborative Filtering (SVD)**
- Uses user-item interactions (ratings).
- Matrix factorization technique (Singular Value Decomposition).
- Best for personalized recommendations based on user preferences.

### **Content-Based Filtering (TF-IDF + FAISS)**
- Uses product review text data.
- TF-IDF Vectorization to convert text into numerical form.
- FAISS (Facebook AI Similarity Search) for scalable similarity matching.

### **Hybrid Model**
- Balances **collaborative** (user-item behavior) and **content-based** (item features) approaches.
- Weighted combination ensures **diverse** recommendations.
- Helps **cold-start problem** where new users have no past interactions.


