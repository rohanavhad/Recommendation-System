import streamlit as st
import pickle
import faiss
import numpy as np
import pandas as pd
from surprise import SVD

df = pd.read_csv("D:/Recommendation System project/notebooks/fashion_data.csv")

with open("D:/Recommendation System project/models/collaborative_model.pkl", "rb") as f:
    collab_model = pickle.load(f)

index = faiss.read_index("D:/Recommendation System project/models/content_faiss1.index")
index.nprobe = 10  

with open("D:/Recommendation System project/models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def get_collaborative_recommendations(user_id, top_n=5):
    product_ids = df['productID'].unique()
    predictions = [collab_model.predict(user_id, pid) for pid in product_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_products = [pred.iid for pred in predictions[:top_n]]
    return top_products

def get_similar_products(product_id, top_n=5):
    if product_id not in df['productID'].values:
        return []
    
    product_idx = df[df['productID'] == product_id].index[0]
    product_review = [df.iloc[product_idx]['review']]
    product_vector = vectorizer.transform(product_review).toarray().astype('float32')

    _, similar_indices = index.search(product_vector, top_n + 1)  
    similar_products = df.iloc[similar_indices[0][1:]]['productID'].tolist()  
    
    return similar_products

# Hybrid Recommendation Function
def get_hybrid_recommendations(user_id, top_n=5):
    collab_recs = get_collaborative_recommendations(user_id, top_n=10)
    content_recs = []

    if collab_recs:
        content_recs = get_similar_products(collab_recs[0], top_n=top_n)  
    
    hybrid_recs = list(set(collab_recs[:top_n] + content_recs[:top_n]))
    return hybrid_recs[:top_n]  

# Streamlit UI
st.title("🛍️ Fashion Recommendation System")


user_id = st.number_input("Enter User ID:", min_value=1, max_value=100000, value=1, step=1)

if st.button("Get Recommendations"):
    recommendations = get_hybrid_recommendations(user_id)
    if recommendations:
        st.success(f" Recommended Products for User {user_id}:")
        for i, product in enumerate(recommendations, 1):
            st.write(f"{i}. Product ID: {product}")
    else:
        st.error("No recommendations found. Try a different User ID.")
