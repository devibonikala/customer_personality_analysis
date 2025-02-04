import os
import logging
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuring the log file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='app.log', filemode='w')
logger = logging.getLogger()

# Allowed file extensions for CSV upload
ALLOWED_EXTENSIONS = {'csv'}

# Function to check if the uploaded file is a CSV
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the trained models and scaler
def load_models():
    try:
        logger.info("Current directory: %s", os.getcwd())
        
        # Load models
        rf_classifier = pickle.load(open('customer_personality_model.pkl', 'rb'))  # RandomForest personality classifier
        scaler = pickle.load(open('scaler.pkl', 'rb'))  # StandardScaler
        pca = pickle.load(open('pca.pkl', 'rb'))  # PCA model
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))  # Feature names
        kmeans = pickle.load(open('kmeans.pkl', 'rb'))  # KMeans model
        
        logger.info("Models loaded successfully.")
        return rf_classifier, scaler, pca, feature_names, kmeans
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None, None, [], None

rf_classifier, scaler, pca, feature_names, kmeans = load_models()

# Personality mapping (modify these as needed)
personality_mapping = {
    0: "Confident Customer - Prefers premium brands & quick decision-maker.",
    1: "Cautious Customer - Price-sensitive, reads reviews before buying.",
    2: "Adventurous Customer - Loves new trends, enjoys exploring new brands.",
    3: "Balanced Customer - Mixes budget & luxury purchases.",
    4: "Impulsive Buyer - Attracted by discounts & one-time offers.",
    5: "Loyal Customer - Repeats purchases from the same brand."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'fileInput' not in request.files:
        logger.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400

    file = request.files['fileInput']
    
    if file.filename == '':
        logger.error("No file selected for uploading")
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        logger.error("Invalid file format. Only CSV files are allowed.")
        return jsonify({'error': 'Invalid file format. Only CSV files are allowed.'}), 400

    try:
        # Save and process the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        logger.info(f"File uploaded successfully: {filename}")

        # Read the uploaded CSV file
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()  # Remove whitespace from column names
        
        # Debug: Log uploaded file column names
        logger.info("Uploaded file columns: %s", df.columns.tolist())
        
        # ----------------------------
        # Preprocessing similar to model.py
        # ----------------------------
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

        # Define personality traits based on multiple criteria
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
        # One-hot encode categorical variables
        # ----------------------------
        df_encoded = pd.get_dummies(df, drop_first=True)

        # Ensure all features exist in the same order as in training
        missing_cols = set(feature_names) - set(df_encoded.columns)
        for col in missing_cols:
            df_encoded[col] = 0  # Add missing columns with 0 values
        
        # Drop extra columns if any, then reorder to match training set
        extra_cols = set(df_encoded.columns) - set(feature_names)
        if extra_cols:
            df_encoded = df_encoded.drop(columns=extra_cols)
        df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
        
        # Handle missing values by filling with mean
        df_encoded.fillna(df_encoded.mean(), inplace=True)

        # ----------------------------
        # Standardize and apply PCA
        # ----------------------------
        X_scaled = scaler.transform(df_encoded)
        X_pca = pca.transform(X_scaled)
        
        # ----------------------------
        # Predict Clusters and Personality
        # ----------------------------
        clusters = kmeans.predict(X_pca)
        personality_preds = rf_classifier.predict(X_pca)
        
        # Map personality codes to descriptive traits
        personality_descriptions = [personality_mapping.get(pred, "Unknown Personality") for pred in personality_preds]
        
        # Count occurrences of each cluster for visualization
        cluster_counts = dict(Counter(clusters))
        
        # Create a bar chart for cluster distribution
        plt.figure(figsize=(6, 4))
        plt.bar(cluster_counts.keys(), cluster_counts.values(), color='skyblue')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Customers')
        plt.title('Customer Cluster Distribution')
        plt.xticks(list(cluster_counts.keys()))
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        encoded_plot = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        logger.info("Prediction successful.")
        return jsonify({
            'clusters': [int(c) for c in clusters],
            'personality_codes': [int(p) for p in personality_preds],
            'personality_descriptions': personality_descriptions,
            'cluster_distribution_plot': encoded_plot
        })
    
    except KeyError as e:
        logger.error(f"Missing columns: {list(e.args)}. Ensure the uploaded file has the correct format.")
        return jsonify({'error': f'Missing columns: {list(e.args)}. Ensure the uploaded file has the correct format.'})
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
