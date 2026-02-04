import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    lca_df = pd.read_csv('sustainable_materials_lca.csv')
    mapping_df = pd.read_csv('product_industry_mapping.csv')
    print("Files loaded successfully.")
except FileNotFoundError:
    print("Error: Please upload 'sustainable_materials_lca.csv' and 'product_industry_mapping.csv' to Colab.")

    # Safety is defined as the average of Biodegradability and Recyclability
lca_df['Material_Safety'] = (lca_df['Biodegradability'] + lca_df['Recyclability']) / 2
# assign every material to every industry category for multi-scenario prediction
lca_df['key'] = 1
mapping_df['key'] = 1
combined_df = pd.merge(lca_df, mapping_df, on='key').drop('key', axis=1)
features_cols = [
    'Material_Safety', 'Strength_MPa', 'Category_Name',
    'Material_Type', 'Moisture_Sensitivity', 'Fragility', 'Typical_Weight_kg'
]
target_cost = 'Cost_per_kg'
target_co2 = 'CO2_kg_per_kg'

X = combined_df[features_cols].copy()
y_cost = combined_df[target_cost]
y_co2 = combined_df[target_co2]
le_cat = LabelEncoder()
X['Category_Name'] = le_cat.fit_transform(X['Category_Name'])

le_mat = LabelEncoder()
X['Material_Type'] = le_mat.fit_transform(X['Material_Type'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into Training and Testing sets (80/20)
X_train, X_test, y_cost_train, y_cost_test, y_co2_train, y_co2_test = train_test_split(
    X_scaled, y_cost, y_co2, test_size=0.2, random_state=42
)

rf_cost = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cost.fit(X_train, y_cost_train)
y_cost_pred = rf_cost.predict(X_test)