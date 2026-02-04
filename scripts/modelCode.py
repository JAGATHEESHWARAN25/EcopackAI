import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# 1. LOAD DATASETS
# Make sure 'packaging_ml_training_set.csv' and 'cleaned_materials_superset.csv' are uploaded
df = pd.read_csv('packaging_ml_training_set.csv')
materials_superset = pd.read_csv('cleaned_materials_superset.csv')

# 2. GENERATE TARGET VALUES
# Estimating packaging mass: 5% of product weight adjusted by bulkiness factor
df['pkg_mass_kg'] = (df['product_weight_g'] / 1000) * 0.05 * df['bulkiness_factor']
df['target_cost'] = df['cost_inr_per_kg'] * df['pkg_mass_kg']
df['target_co2'] = df['co2_kg_per_kg'] * df['pkg_mass_kg']

# 3. PREPROCESSING & ENCODING
le_cat = LabelEncoder()
le_fmt = LabelEncoder()
df['cat_enc'] = le_cat.fit_transform(df['product_category'])
df['fmt_enc'] = le_fmt.fit_transform(df['packaging_format'])

prod_features = ['cat_enc', 'product_weight_g', 'price_inr', 'protection_score',
                 'bulkiness_factor', 'shelf_life_days', 'fmt_enc']
mat_features = ['strength_mpa', 'biodegradability_1_to_10', 'recyclability_percent']

X = df[prod_features + mat_features]
y_cost = df['target_cost']
y_co2 = df['target_co2']

X_train, X_test, y_cost_train, y_cost_test, y_co2_train, y_co2_test = train_test_split(
    X, y_cost, y_co2, test_size=0.2, random_state=42
)

# 4. MODEL TRAINING
print("Training ML Models...")
# Cost Prediction: Random Forest
rf_cost = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cost.fit(X_train, y_cost_train)

# CO2 Impact Prediction: XGBoost
xgb_co2 = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_co2.fit(X_train, y_co2_train)

# 5. EVALUATION
def show_metrics(y_true, y_pred, name):
    print(f"\n--- {name} Metrics ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"R2:   {r2_score(y_true, y_pred):.4f}")

show_metrics(y_cost_test, rf_cost.predict(X_test), "Cost (Random Forest)")
show_metrics(y_co2_test, xgb_co2.predict(X_test), "CO2 Impact (XGBoost)")

# 6. INTERACTIVE RECOMMENDATION SYSTEM
def get_recommendations():
    print("\n" + "="*50)
    query = input("Enter Product Name or Category to search: ")

    # Search for matches
    matches = df[df['product_name'].str.contains(query, case=False) |
                 df['product_category'].str.contains(query, case=False)]

    if matches.empty:
        print("No products found for that search. Please try again.")
        return

    # Selecting the first match for the recommendation
    prod = matches.iloc[0]

    print("\n--- [ SELECTED PRODUCT DETAILS ] ---")
    print(f"Product Name:      {prod['product_name']}")
    print(f"Product Category:  {prod['product_category']}")
    print(f"Product ID:        {prod['product_id']}")
    print("-" * 35)
    print(f"Current Material:  {prod['material_type']}")
    print(f"Material ID:       {prod['material_key']}")
    print(f"Current CO2 Impact: {prod['target_co2']:.4f} kg")
    print(f"Current Cost:      {prod['target_cost']:.2f} INR")
    print("="*50)

    # Generate predictions for ALL materials in the superset
    recommendations = []
    for _, mat in materials_superset.iterrows():
        input_data = pd.DataFrame([[
            prod['cat_enc'], prod['product_weight_g'], prod['price_inr'],
            prod['protection_score'], prod['bulkiness_factor'], prod['shelf_life_days'],
            prod['fmt_enc'], mat['strength_mpa'], mat['biodegradability_1_to_10'],
            mat['recyclability_percent']
        ]], columns=prod_features + mat_features)

        p_cost = rf_cost.predict(input_data)[0]
        p_co2 = xgb_co2.predict(input_data)[0]

        # Scoring: High sustainability (B*R) balanced against predicted cost and emissions
        score = (mat['biodegradability_1_to_10'] * mat['recyclability_percent']) / (p_cost * p_co2 + 1e-6)

        recommendations.append({
            'Material': mat['material_type'],
            'Mat_ID': mat['material_key'],
            'Pred_Cost': p_cost,
            'Pred_CO2': p_co2,
            'Score': score
        })

    # Sort and get top 3
    top3 = pd.DataFrame(recommendations).sort_values(by='Score', ascending=False).head(3)

    print("\n--- [ TOP 3 RECOMMENDED MATERIALS ] ---")
    for i, row in top3.iterrows():
        cost_change = ((row['Pred_Cost'] - prod['target_cost']) / prod['target_cost']) * 100
        co2_change = ((row['Pred_CO2'] - prod['target_co2']) / prod['target_co2']) * 100

        print(f"\n{row['Material']} ({row['Mat_ID']})")
        print(f"  Predicted Cost: {row['Pred_Cost']:.2f} INR ({cost_change:+.1f}% vs Current)")
        print(f"  Predicted CO2:  {row['Pred_CO2']:.4f} kg ({co2_change:+.1f}% vs Current)")

        # Effectiveness Comparison Summary
        if co2_change < 0 and cost_change < 0:
            print("  >> EFFECTIVENESS: Highly Effective (Saves both Cost and Emissions)")
        elif co2_change < 0:
            print("  >> EFFECTIVENESS: Sustainable (Reduces Emissions, higher cost)")
        else:
            print("  >> EFFECTIVENESS: Cost-Effective (Reduces Cost, higher emissions)")

# Run the system
get_recommendations()