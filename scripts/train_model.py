import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'EcoPackAI_Backend-main', 'model')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading data...")
# Load datasets
materials_df = pd.read_csv(os.path.join(DATA_DIR, 'materials_with_cost_and_co2.csv'))
products_df = pd.read_csv(os.path.join(DATA_DIR, 'products_cleaned.csv'))

# Display basic info
print(f"Materials: {materials_df.shape}")
print(f"Products: {products_df.shape}")

# Prepare training data
# We need to create a dataset that combines product attributes with material attributes
# Since we don't have a direct mapping in the cleaned files, we'll simulate a training set
# by cross-joining a sample of products with materials and generating synthetic targets
# based on simple physics/economics rules, similar to what was implied in modelCode.py

print("Preparing training data...")

# sample products to keep it manageable if needed, but 50k is fine
# Let's take a subset to speed up for now if needed, but we'll try full first.
# actually, let's create a more realistic synthetic dataset for training
# based on the logic in modelCode.py

n_samples = 5000 # Increased for better accuracy
data_records = []

# Create encoders
le_cat = LabelEncoder()
le_fmt = LabelEncoder()

# Fit encoders on available data
# FIXED: Use 'product_category' instead of 'product_category_name'
all_categories = np.concatenate([products_df['product_category'].unique(), ['Textiles', 'Furniture', 'Eco-Gifts']])
le_cat.fit(np.unique(all_categories)) 
# Packaging format isn't in products_cleaned, let's add some dummy formats
formats = ['Box', 'Pouch', 'Bottle', 'Wrap']
le_fmt.fit(formats)
print("Encoders fitted.")

# Save encoders
joblib.dump(le_cat, os.path.join(MODEL_DIR, 'le_cat.pkl'))
joblib.dump(le_fmt, os.path.join(MODEL_DIR, 'le_fmt.pkl'))

# Generate synthetic training data
# We'll randomly sample products and assign them random materials and formats
# Then calculate "true" cost and co2 based on the material properties

np.random.seed(42)

for _ in range(n_samples):
    prod = products_df.sample(1).iloc[0]
    mat = materials_df.sample(1).iloc[0]
    fmt = np.random.choice(formats)
    
    # Extract features
    weight = prod['product_weight_g']
    # FIXED: Use 'price_inr'
    price = prod['price_inr'] 
    
    # Derived features
    # FIXED: Use 'product_volume_cm3' if available, else derive
    if 'product_volume_cm3' in prod:
        volume = prod['product_volume_cm3']
    else:
        volume = prod['product_length_cm'] * prod['product_height_cm'] * prod['product_width_cm']
        
    bulkiness = 1.0 # default
    if volume > 1000: bulkiness = 1.3
    if volume > 5000: bulkiness = 1.6
    
    # Packaging mass estimation (simplified)
    pkg_mass_g = weight * 0.05 * bulkiness
    pkg_mass_kg = pkg_mass_g / 1000.0
    
    # Target calculations (Cost and CO2)
    strength = 50
    if 'Strength_MPa' in mat: strength = mat['Strength_MPa']
    
    bio = 5
    if 'Biodegradability' in mat: bio = mat['Biodegradability']
    
    recycle = 50
    if 'Recyclability' in mat: recycle = mat['Recyclability']

    # Deterministic formulas for Cost and CO2
    cost_per_kg = 50 + (strength * 0.2) + (bio * 5.0) - (recycle * 0.1)
    cost_per_kg = max(10, cost_per_kg) # Min 10 INR/kg
    
    co2_per_kg = 1.5 - (bio * 0.1) - (recycle * 0.01) + (strength * 0.005)
    co2_per_kg = max(0.1, co2_per_kg)

    total_cost = cost_per_kg * pkg_mass_kg
    total_co2 = co2_per_kg * pkg_mass_kg
    
    # Handle fragility/protection
    protection_score = 5
    if 'fragility_level' in prod:
        try:
            protection_score = float(prod['fragility_level'])
        except:
             protection_score = 5 # fallback

    try:
        cat_encoded = le_cat.transform([prod['product_category']])[0]
    except Exception as e:
        print(f"⚠️ Error encoding category '{prod['product_category']}': {e}")
        cat_encoded = 0 # Fallback

    try:
        fmt_encoded = le_fmt.transform([fmt])[0]
    except Exception as e:
        print(f"⚠️ Error encoding format '{fmt}': {e}")
        fmt_encoded = 0

    row = {
        'cat_enc': cat_encoded,
        'product_weight_g': weight,
        'price_inr': price,
        'protection_score': protection_score, 
        'bulkiness_factor': bulkiness,
        'shelf_life_days': 30, # default, not in product data
        'fmt_enc': fmt_encoded,
        # Material features included as input to help the model learn the relationship
        'strength_mpa': strength,
        'biodegradability_1_to_10': bio,
        'recyclability_percent': recycle,
        # Targets
        'target_cost': total_cost,
        'target_co2': total_co2
    }
    data_records.append(row)

train_df = pd.DataFrame(data_records)

print(f"Training data prepared: {train_df.shape}")

# Features
feature_cols = ['cat_enc', 'product_weight_g', 'price_inr', 'protection_score', 
                'bulkiness_factor', 'shelf_life_days', 'fmt_enc',
                'strength_mpa', 'biodegradability_1_to_10', 'recyclability_percent']

X = train_df[feature_cols]
y_cost = train_df['target_cost']
y_co2 = train_df['target_co2']

# Split data
X_train, X_test, y_cost_train, y_cost_test, y_co2_train, y_co2_test = train_test_split(X, y_cost, y_co2, test_size=0.2, random_state=42)

# Train Cost Model
print("Training Cost Model (Random Forest)...")
rf_cost = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_cost.fit(X_train, y_cost_train)

# Evaluate Cost Model
y_cost_pred = rf_cost.predict(X_test)
cost_mse = mean_squared_error(y_cost_test, y_cost_pred)
cost_r2 = r2_score(y_cost_test, y_cost_pred)
print(f"Cost Model Performance: R2 = {cost_r2:.4f} (Target > 0.95)")

joblib.dump(rf_cost, os.path.join(MODEL_DIR, 'rf_cost.pkl'))

# Train CO2 Model
print("Training CO2 Model (XGBoost)...")
xgb_co2 = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=10, random_state=42)
xgb_co2.fit(X_train, y_co2_train)

# Evaluate CO2 Model
y_co2_pred = xgb_co2.predict(X_test)
co2_mse = mean_squared_error(y_co2_test, y_co2_pred)
co2_r2 = r2_score(y_co2_test, y_co2_pred)
print(f"CO2 Model Performance: R2 = {co2_r2:.4f} (Target > 0.95)")

joblib.dump(xgb_co2, os.path.join(MODEL_DIR, 'rf_co2.pkl')) # Saving as rf_co2.pkl to match app.py expectation

print("Models trained and saved successfully.")
