import pandas as pd
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Load Processed Data
df = pd.read_csv("D:/Recommendation System project/notebooks/fashion_data.csv")

# Prepare Data for Surprise Library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'productID', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD Model
model = SVD()
model.fit(trainset)

# Ensure "models/" directory exists
import os
os.makedirs("models", exist_ok=True)

# Save the trained model
with open("models/collaborative_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Collaborative Filtering Model Trained & Saved!")
