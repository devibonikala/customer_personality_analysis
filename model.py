import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

# ----------------------------
# 1. Data Loading and Preprocessing
# ----------------------------
df = pd.read_csv("marketing_campaign.csv", sep='\t')

# Preprocessing categorical variables
df['Education'] = df['Education'].replace(['PhD', '2n Cycle', 'Graduation', 'Master'], 'Post Graduate')
df['Education'] = df['Education'].replace(['Basic'], 'Under Graduate')
df['Marital_Status'] = df['Marital_Status'].replace(['Married', 'Together'], 'Relationship')
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'], 'Single')

# Create new features
df['Kids'] = df['Kidhome'] + df['Teenhome']
df['Expenses'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
df['TotalAcceptedCmp'] = df[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                             'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)
df['NumTotalPurchases'] = df[['NumWebPurchases', 'NumCatalogPurchases', 
                              'NumStorePurchases', 'NumDealsPurchases']].sum(axis=1)
df['Customer_Age'] = pd.Timestamp('now').year - df['Year_Birth']
df['Age_Group'] = np.where(df['Customer_Age'] > 50, 'Senior', 'Young')

# Define personality traits
df['Personality'] = np.where(df['Expenses'] > df['Expenses'].median(), 
                             'High Spender', 'Budget Conscious')
df['Personality'] = np.where(df['TotalAcceptedCmp'] > 2, 
                             'Brand Loyal', df['Personality'])
df['Personality'] = np.where(df['NumTotalPurchases'] > df['NumTotalPurchases'].median(), 
                             'Impulsive Buyer', df['Personality'])

# Drop unnecessary columns
columns_to_drop = ['Year_Birth', 'ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                   'AcceptedCmp4', 'AcceptedCmp5', 'NumWebVisitsMonth', 'NumWebPurchases', 
                   'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                   'MntSweetProducts', 'MntGoldProds', 'NumStorePurchases', 'Kidhome', 
                   'Teenhome', 'NumDealsPurchases', 'Dt_Customer']
df.drop(columns=columns_to_drop, axis=1, errors='ignore', inplace=True)

# ----------------------------
# 2. One-Hot Encoding and Feature Alignment
# ----------------------------
target = df['Personality']
df_features = df.drop(columns=['Personality'])

df_encoded = pd.get_dummies(df_features, drop_first=True)

# Save feature names
feature_names = df_encoded.columns.tolist()
with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

df_encoded.fillna(df_encoded.mean(), inplace=True)

# ----------------------------
# 3. Standardization and PCA
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

pca = PCA(n_components=min(10, X_scaled.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Save the PCA transformer
with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)

# ----------------------------
# 4. Classification Model (Customer Personality)
# ----------------------------
df['Personality'] = df['Personality'].astype('category').cat.codes
y = df['Personality']

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
print("Classification Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save classification model
with open("customer_personality_model.pkl", "wb") as f:
    pickle.dump(rf_classifier, f)

# ----------------------------
# 5. Clustering Model
# ----------------------------
# Use the silhouette score to find the optimal number of clusters
wcss = []
best_k = 2  # Start with 2 clusters
best_score = -1

for k in range(2, 11):
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans_test.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"Best k (based on Silhouette Score): {best_k}")

# Train final KMeans model with the best k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(X_pca)

# Save the clustering model
with open("kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

# Print cluster centers and cluster distribution
print("KMeans Cluster Centers:\n", kmeans.cluster_centers_)
unique_labels, counts = np.unique(kmeans.labels_, return_counts=True)
print("Training Data Cluster Distribution:", dict(zip(unique_labels, counts)))

print("✅ K-Means clustering model trained and saved successfully!")

# ----------------------------
# 6. Predict Clusters for New Data
# ----------------------------
def predict_clusters(new_data: pd.DataFrame):
    """
    Predict clusters for new data using the saved KMeans model.
    Also prints diagnostics to help understand the PCA transformation.
    """
    if 'Personality' in new_data.columns:
        new_data = new_data.drop(columns=['Personality'])

    # Apply preprocessing
    new_data['Education'] = new_data['Education'].replace(['PhD', '2n Cycle', 'Graduation', 'Master'], 'Post Graduate')
    new_data['Education'] = new_data['Education'].replace(['Basic'], 'Under Graduate')
    new_data['Marital_Status'] = new_data['Marital_Status'].replace(['Married', 'Together'], 'Relationship')
    new_data['Marital_Status'] = new_data['Marital_Status'].replace(['Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd'], 'Single')

    new_data['Kids'] = new_data['Kidhome'] + new_data['Teenhome']
    new_data['Expenses'] = new_data[['MntWines', 'MntFruits', 'MntMeatProducts', 
                                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
    new_data['TotalAcceptedCmp'] = new_data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                                              'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)
    new_data['NumTotalPurchases'] = new_data[['NumWebPurchases', 'NumCatalogPurchases', 
                                               'NumStorePurchases', 'NumDealsPurchases']].sum(axis=1)
    new_data['Customer_Age'] = pd.Timestamp('now').year - new_data['Year_Birth']
    new_data['Age_Group'] = np.where(new_data['Customer_Age'] > 50, 'Senior', 'Young')

    new_data.drop(columns=columns_to_drop, axis=1, errors='ignore', inplace=True)

    # One-hot encode and align new data
    df_encoded_new = pd.get_dummies(new_data, drop_first=True)
    with open("feature_names.pkl", "rb") as f:
        train_feature_names = pickle.load(f)
    
    missing_cols = set(train_feature_names) - set(df_encoded_new.columns)
    for col in missing_cols:
        df_encoded_new[col] = 0
    extra_cols = set(df_encoded_new.columns) - set(train_feature_names)
    if extra_cols:
        df_encoded_new.drop(columns=extra_cols, inplace=True)
    df_encoded_new = df_encoded_new.reindex(columns=train_feature_names, fill_value=0)
    df_encoded_new.fillna(df_encoded_new.mean(), inplace=True)
    
    # Load the scaler and PCA model
    with open("scaler.pkl", "rb") as f:
        scaler_loaded = pickle.load(f)
    with open("pca.pkl", "rb") as f:
        pca_loaded = pickle.load(f)
    
    # Transform new data
    X_scaled_new = scaler_loaded.transform(df_encoded_new)
    X_pca_new = pca_loaded.transform(X_scaled_new)

    # Debug: Print the PCA results for the new data
    print("New data PCA coordinates (first 5 rows):")
    print(X_pca_new[:5])

    # Load KMeans model and predict clusters
    with open("kmeans.pkl", "rb") as f:
        kmeans_loaded = pickle.load(f)

    clusters = kmeans_loaded.predict(X_pca_new)

    # Print the predicted cluster distribution
    unique_labels, counts = np.unique(clusters, return_counts=True)
    print("Predicted Cluster Distribution:", dict(zip(unique_labels, counts)))

    # Visualize
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_loaded.labels_, cmap='viridis', alpha=0.3, label='Training Data')
    plt.scatter(X_pca_new[:, 0], X_pca_new[:, 1], c='red', marker='x', s=100, label='New Data')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection: Training vs. New Data")
    plt.legend()
    plt.show()

    return clusters
