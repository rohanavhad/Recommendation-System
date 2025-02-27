import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV

# Load Processed Data
df = pd.read_csv("D:/Recommendation System project/notebooks/fashion_data.csv")

# Prepare Data for Surprise Library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userID', 'productID', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Define Hyperparameter Grid
param_grid = {
    'n_factors': [50, 100, 150],  # Number of latent factors
    'lr_all': [0.002, 0.005, 0.01],  # Learning rate
    'reg_all': [0.02, 0.05, 0.1]  # Regularization term
}

# Perform Grid Search (Remove verbose)
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
grid_search.fit(data)

# Get Best Parameters
best_params = grid_search.best_params['rmse']
print(f"✅ Best Parameters for SVD: {best_params}")

# Train Best Model
best_model = SVD(n_factors=best_params['n_factors'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])
best_model.fit(trainset)

# Save the Optimized Model
with open("models/collaborative_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Optimized Collaborative Filtering Model Trained & Saved!")
