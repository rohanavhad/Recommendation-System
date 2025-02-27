import pandas as pd
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from scipy.sparse import csr_matrix

# Load Processed Data
df = pd.read_csv("D:/Recommendation System project/FINAL.csv")

# Step 1: Fill Missing Reviews with Empty Strings
df["review"] = df["review"].fillna("")

# Ensure "models/" directory exists
os.makedirs("models", exist_ok=True)

# Step 2: Convert Text Reviews into Sparse TF-IDF Vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['review'])

# Step 3: Initialize FAISS Index (Use IVF with clustering)
dimension = tfidf_matrix.shape[1]
n_clusters = 256  # Adjust based on dataset size

quantizer = faiss.IndexFlatL2(dimension)  # Quantizer for clustering
index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters, faiss.METRIC_L2)

# Train FAISS on a SMALL sample instead of entire dataset
sample_size = min(10000, tfidf_matrix.shape[0])  # Train on 10k samples or less
index.train(tfidf_matrix[:sample_size].toarray())  # Convert only the small sample

# Step 4: Add Data in Small Batches to Avoid Memory Issues
batch_size = 5000  # Reduce batch size to save memory
product_ids = np.array(df['productID']).astype(np.int64)

for i in range(0, tfidf_matrix.shape[0], batch_size):
    batch_vectors = tfidf_matrix[i : i + batch_size].toarray()  # Convert small batch
    batch_ids = product_ids[i : i + batch_size]
    index.add_with_ids(batch_vectors, batch_ids)

# Step 5: Save FAISS Index & Vectorizer
faiss.write_index(index, "models/content_faiss1.index")

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Content-Based Model (FAISS with IVF) Trained & Saved!")
