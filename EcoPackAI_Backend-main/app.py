from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import psycopg2
import os
import io
from flask import send_file, send_from_directory
from openpyxl import Workbook
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app) # Enable CORS for frontend integration

# -----------------------
# Load ML models & Encoders
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

try:
    rf_co2 = joblib.load(os.path.join(MODEL_DIR, "rf_co2.pkl"))
    rf_cost = joblib.load(os.path.join(MODEL_DIR, "rf_cost.pkl"))
    le_cat = joblib.load(os.path.join(MODEL_DIR, "le_cat.pkl"))
    le_fmt = joblib.load(os.path.join(MODEL_DIR, "le_fmt.pkl"))
    print("✅ Models and Encoders loaded successfully")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    rf_co2 = rf_cost = le_cat = le_fmt = None

# -----------------------
# PostgreSQL connection
# -----------------------
def get_db_connection():
    try:
        db_url = os.environ.get('DATABASE_URL')
        if db_url:
             conn = psycopg2.connect(db_url)
        else:
            conn = psycopg2.connect(
                dbname="ecopackai",
                user="postgres",
                password="Jaga@123", # Updated password
                host="localhost",
                port="5432"
            )
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

# -----------------------
# Health check
# -----------------------
@app.route("/", methods=["GET"])
def home():
    # Helper to get absolute path to frontend
    frontend_dir = os.path.join(BASE_DIR, '../frontend')
    return send_from_directory(frontend_dir, 'index.html')



@app.route("/<path:path>")
def serve_static(path):
    frontend_dir = os.path.join(BASE_DIR, '../frontend')
    return send_from_directory(frontend_dir, path)

# -----------------------
# Global Data
# -----------------------
ALL_CANDIDATES = [
    {"name": "Recycled Cardboard", "strength": 40, "bio": 8, "recycle": 90, "type": "Paper", "suitable_for": ["Electronics", "Fashion", "Household", "Dry Food"]},
    {"name": "Biodegradable Plastic", "strength": 60, "bio": 9, "recycle": 10, "type": "Bioplastic", "suitable_for": ["Food", "Cosmetics", "Pharmaceuticals", "Liquids"]},
    {"name": "Molded Pulp", "strength": 35, "bio": 10, "recycle": 95, "type": "Paper", "suitable_for": ["Electronics", "Eggs", "Fragile Items"]},
    {"name": "Glass", "strength": 80, "bio": 10, "recycle": 100, "type": "Glass", "suitable_for": ["Cosmetics", "Liquids", "Premium Food"]},
    {"name": "Aluminum", "strength": 90, "bio": 0, "recycle": 100, "type": "Metal", "suitable_for": ["Beverages", "Canned Food"]},
    {"name": "Compostable Pouch", "strength": 30, "bio": 10, "recycle": 50, "type": "Flexible", "suitable_for": ["Snacks", "Dry Food"]},
    {"name": "Bamboo Composites", "strength": 85, "bio": 10, "recycle": 80, "type": "Natural", "suitable_for": ["Furniture", "Household", "Eco-Gifts"]},
    {"name": "Hemp Bioplastic", "strength": 55, "bio": 10, "recycle": 70, "type": "Bioplastic", "suitable_for": ["Textiles", "Packaging", "Cosmetics", "Eco-Gifts"]},
    {"name": "Seaweed Packaging", "strength": 20, "bio": 10, "recycle": 100, "type": "Edible", "suitable_for": ["Food", "Eco-Gifts", "Liquids"]}
]

# -----------------------
# AI Material Recommendation API
# -----------------------
@app.route("/recommend-material", methods=["POST"])
def recommend_material():
    try:
        if not rf_cost:
            return jsonify({"status": "error", "message": "ML Models not loaded"}), 500

        data = request.json
        
        # Extract features from frontend
        category = data.get("category", "Electronics")
        weight = float(data.get("weight", 0))
        price = float(data.get("price", 0))
        shelf_life = float(data.get("shelf_life", 30))
        fmt = data.get("format", "Box")
        protection = float(data.get("protection", 5))
        bulkiness = float(data.get("bulkiness", 1.0))

        # Encode categorical features
        try:
            if category in le_cat.classes_:
                cat_enc = le_cat.transform([category])[0]
            else:
                cat_enc = 0 
            
            if fmt in le_fmt.classes_:
                fmt_enc = le_fmt.transform([fmt])[0]
            else:
                fmt_enc = 0
        except Exception as e:
            cat_enc = 0
            fmt_enc = 0

        # Create candidates...
        recommendations = []
        for cand in ALL_CANDIDATES:
            suitability_score = 1.0
            if category not in cand.get("suitable_for", []) and "All" not in cand.get("suitable_for", []):
                if category == "Food" and cand["type"] in ["Bioplastic", "Glass", "Metal", "Flexible"]:
                    suitability_score = 0.9
                elif category == "Electronics" and cand["type"] == "Paper":
                    suitability_score = 1.0
                else:
                    suitability_score = 0.4
            
            features = np.array([cat_enc, float(weight), float(price), float(protection), float(shelf_life), float(bulkiness), fmt_enc, 
                               cand['strength'], cand['bio'], cand['recycle']]).reshape(1, -1)
            
            pred_cost = max(0.1, float(rf_cost.predict(features)[0]))
            pred_co2 = max(0.001, float(rf_co2.predict(features)[0]))
            
            base_sustain = (cand['bio'] * 10 + cand['recycle']) / 2
            cost_impact = np.log1p(pred_cost) * 15
            co2_impact = np.log1p(pred_co2 * 100) * 20
            
            sustainability_contribution = base_sustain * suitability_score
            raw_score = sustainability_contribution - cost_impact - co2_impact
            final_score = float(max(10, min(99, raw_score + 10)))
            
            eff = "Standard"
            if final_score > 80: eff = "Best Overall"
            elif pred_co2 < 0.1 and final_score > 60: eff = "Eco-Friendly"
            elif pred_cost < 15 and final_score > 50: eff = "Cost-Saver"
            
            recommendations.append({
                "material_name": cand["name"],
                "predicted_cost": round(pred_cost, 2),
                "predicted_co2": round(pred_co2, 4),
                "score": round(final_score, 1),
                "effectiveness": eff,
                "bio_score": cand["bio"],
                "recycle_percent": cand["recycle"]
            })
            
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        top_recommendation = recommendations[0]

        # Database Save
        request_id = None
        db_error = None
        
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                
                # 1. Insert Request
                cur.execute("""
                    INSERT INTO product_requests (product_category, weight_g, price_inr, format, protection_level, bulkiness_factor, shelf_life_days)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING request_id
                """, (category, weight, price, fmt, protection, bulkiness, shelf_life))
                
                request_id = cur.fetchone()[0]
                
                # 2. Insert Prediction
                cur.execute("""
                    INSERT INTO ai_predictions (request_id, recommended_material, predicted_cost, predicted_co2, sustainability_score, effectiveness_rating)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (request_id, top_recommendation['material_name'], top_recommendation['predicted_cost'], 
                      top_recommendation['predicted_co2'], top_recommendation['score'], top_recommendation['effectiveness']))
                
                conn.commit()
                cur.close()
                conn.close()
                print(f"✅ Data saved to DB (Request ID: {request_id})")
            except Exception as e:
                db_error = str(e)
                print(f"ERROR DB Save: {e}")
                if conn: conn.rollback()

        return jsonify({
            "status": "success",
            "request_id": request_id,
            "db_error": db_error,
            "top_choice": top_recommendation,
            "alternatives": recommendations
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------
# Analytics API (Module 7)
# -----------------------
@app.route("/analytics", methods=["GET"])
def get_analytics():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        cur = conn.cursor()
        
        # 1. Total Requests
        cur.execute("SELECT COUNT(*) FROM product_requests;")
        total_requests = cur.fetchone()[0]
        
        # 2. Average Sustainability Score
        cur.execute("SELECT AVG(sustainability_score) FROM ai_predictions;")
        avg_score = cur.fetchone()[0] or 0
        
        # 3. Material Distribution (for Pie Chart)
        cur.execute("""
            SELECT recommended_material, COUNT(*) 
            FROM ai_predictions 
            GROUP BY recommended_material;
        """)
        material_data = cur.fetchall() # Returns list of (material, count)
        
        # 4. Total CO2 Saved (Simulated against a baseline)
        # Assuming baseline CO2 is 2.0x the predicted (rough estimate for traditional vs eco)
        cur.execute("SELECT SUM(predicted_co2) FROM ai_predictions;")
        total_predicted_co2 = cur.fetchone()[0] or 0
        estimated_savings = total_predicted_co2 * 1.0 

        # 5. Requests by Category (New Chart)
        cur.execute("""
            SELECT product_category, COUNT(*) 
            FROM product_requests 
            GROUP BY product_category;
        """)
        cat_req_data = cur.fetchall()
        cat_req_dict = {row[0]: row[1] for row in cat_req_data}

        response_data = {
            "status": "success",
            "total_requests": total_requests,
            "avg_sustainability_score": round(float(avg_score), 1),
            "material_distribution": {row[0]: row[1] for row in material_data},
            "estimated_co2_savings": round(float(estimated_savings), 2),
            "category_requests": cat_req_dict, # New Data
            "impact_trends": get_impact_trends(cur)
        }
        
        cur.close()
        conn.close()
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Analytics Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def get_impact_trends(cur):
    try:
        # Fetch monthly savings (simulated as sum of predicted_co2 for simplicity)
        # Groups by Month Name (Jan, Feb...) and orders chronologically by Month Number
        cur.execute("""
            SELECT TO_CHAR(pr.created_at, 'Mon') as month_name, 
                   SUM(ap.predicted_co2) as monthly_savings
            FROM ai_predictions ap
            JOIN product_requests pr ON ap.request_id = pr.request_id
            GROUP BY TO_CHAR(pr.created_at, 'Mon'), DATE_PART('month', pr.created_at)
            ORDER BY DATE_PART('month', pr.created_at);
        """)
        results = cur.fetchall()
        
        # Format for Chart.js: { "Jan": 120, "Feb": 150 ... }
        labels = [row[0] for row in results]
        data = [round(float(row[1]), 2) for row in results]
        
        # Ensure we always return lists for the chart
        return {"labels": labels, "data": data}
    except Exception as e:
        print(f"⚠️ Impact Trend Error: {e}")
        return {"labels": [], "data": []}

# -----------------------
# Export Endpoints
# -----------------------
@app.route("/export/excel", methods=["GET"])
def export_excel():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        cur = conn.cursor()
        
        # Fetch detailed records
        cur.execute("""
            SELECT pr.product_category, pr.weight_g, pr.price_inr, 
                   ap.recommended_material, ap.predicted_cost, ap.predicted_co2, ap.sustainability_score
            FROM product_requests pr
            JOIN ai_predictions ap ON pr.request_id = ap.request_id
            ORDER BY pr.created_at DESC;
        """)
        rows = cur.fetchall()
        
        # Create Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Sustainability Report"
        
        # Header
        headers = ["Category", "Weight (g)", "Price (INR)", "Recommended Material", "Pred. Cost", "Pred. CO2 (kg)", "Score"]
        ws.append(headers)
        
        for row in rows:
            ws.append(row)
            
        cur.close()
        conn.close()
        
        # Save to memory
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        
        return send_file(output, as_attachment=True, download_name="EcoPackAI_Report.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
    except Exception as e:
        print(f"❌ Export Excel Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/export/pdf", methods=["GET"])
def export_pdf():
    try:
        category = request.args.get('category')
        request_id = request.args.get('request_id')
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        cur = conn.cursor()
        
        output = io.BytesIO()
        c = canvas.Canvas(output, pagesize=letter)
        width, height = letter
        
        if request_id:
            # ---------------------------------------------------------
            # DETAILED SINGLE REPORT (With Charts & Specifics)
            # ---------------------------------------------------------
            cur.execute("""
                SELECT product_category, weight_g, price_inr, format, protection_level, bulkiness_factor, shelf_life_days
                FROM product_requests WHERE request_id = %s
            """, (request_id,))
            req_data = cur.fetchone()
            
            if not req_data:
                return jsonify({"status": "error", "message": "Request ID not found"}), 404
                
            cat, w, p, fmt, prot, bulk, life = req_data
            
            # --- RE-RUN PREDICTION LOGIC (To get full comparison data) ---
            # Encode
            try:
                cat_enc = le_cat.transform([cat])[0] if cat in le_cat.classes_ else 0
                fmt_enc = le_fmt.transform([fmt])[0] if fmt in le_fmt.classes_ else 0
            except:
                cat_enc, fmt_enc = 0, 0
            
            comps = []
            for cand in ALL_CANDIDATES:
                # Suitability (Simplified for report)
                suitability = 1.0
                if cat not in cand.get("suitable_for", []) and "All" not in cand.get("suitable_for", []):
                    suitability = 0.4
                    
                features = np.array([cat_enc, float(w), float(p), float(prot), float(bulk), float(life), fmt_enc, 
                                   cand['strength'], cand['bio'], cand['recycle']]).reshape(1, -1)
                
                pred_cost = max(0.1, float(rf_cost.predict(features)[0]))
                pred_co2 = max(0.001, float(rf_co2.predict(features)[0]))
                
                base_sustain = (cand['bio'] * 10 + cand['recycle']) / 2
                raw_score = (base_sustain * suitability) - (np.log1p(pred_cost) * 15) - (np.log1p(pred_co2 * 100) * 20)
                final_score = max(10, min(99, raw_score + 10))
                
                comps.append({
                    "name": cand["name"],
                    "cost": round(pred_cost, 2),
                    "co2": round(pred_co2, 4),
                    "score": round(final_score, 1),
                    "bio": cand["bio"],
                    "recycle": cand["recycle"]
                })
            
            comps.sort(key=lambda x: x['score'], reverse=True)
            top = comps[0]
            
            # --- GENERATE CHARTS (Matplotlib) ---
            # Cost Chart
            plt.figure(figsize=(6, 3))
            names = [x['name'] for x in comps[:5]]
            costs = [x['cost'] for x in comps[:5]]
            plt.bar(names, costs, color='green', alpha=0.7)
            plt.title('Predicted Cost (INR)')
            plt.xticks(rotation=15, ha='right', fontsize=8)
            plt.tight_layout()
            img_buf_cost = io.BytesIO()
            plt.savefig(img_buf_cost, format='png')
            img_buf_cost.seek(0)
            plt.close()
            
            # CO2 Chart
            plt.figure(figsize=(6, 3))
            co2s = [x['co2'] for x in comps[:5]]
            plt.bar(names, co2s, color='skyblue', alpha=0.7)
            plt.title('Predicted CO2 Output (kg)')
            plt.xticks(rotation=15, ha='right', fontsize=8)
            plt.tight_layout()
            img_buf_co2 = io.BytesIO()
            plt.savefig(img_buf_co2, format='png')
            img_buf_co2.seek(0)
            plt.close()
            
            # --- BUILD PDF ---
            c.setFont("Helvetica-Bold", 18)
            c.drawString(50, height - 50, f"Recommendation Report: {cat}")
            
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 80, f"Request ID: {request_id}  |  Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
            c.drawString(50, height - 100, f"Inputs: Weight={w}g, Price=INR {p}, Format={fmt}")
            
            # Top Choice
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 140, "Top Recommendation:")
            c.setFillColorRGB(0, 0.5, 0)
            c.drawString(50, height - 160, f"{top['name']}")
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 180, f"Predicted Cost: INR {top['cost']}")
            c.drawString(250, height - 180, f"Predicted CO2: {top['co2']} kg")
            
            # Charts
            c.drawImage(ImageReader(img_buf_cost), 50, height - 420, width=250, height=200)
            c.drawImage(ImageReader(img_buf_co2), 310, height - 420, width=250, height=200)
            
            # Table
            # Table Headers
            y = height - 460
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Comparison Table (Top 5)")
            y -= 25
            c.setFont("Helvetica-Bold", 9)
            
            # Draw Headers
            c.drawString(40, y, "Rank")
            c.drawString(80, y, "Material")
            c.drawString(220, y, "Cost")
            c.drawString(280, y, "CO2")
            c.drawString(340, y, "Bio")
            c.drawString(390, y, "Recycle")
            c.drawString(450, y, "Score")
            c.line(40, y-5, 500, y-5)
            y -= 20
            
            c.setFont("Helvetica", 9)
            for i, item in enumerate(comps[:5]):
                c.drawString(40, y, str(i + 1))
                c.drawString(80, y, item['name'][:22])
                c.drawString(220, y, str(item['cost']))
                c.drawString(280, y, str(item['co2']))
                c.drawString(340, y, f"{item['bio']}/10")
                c.drawString(390, y, f"{item['recycle']}%")
                c.drawString(450, y, f"{item['score']}")
                y -= 20
                
            filename = f"EcoPackAI_Detailed_{request_id}.pdf"

        else:
            # ---------------------------------------------------------
            # SUMMARY REPORT (Global or Category) - ENHANCED DASHBOARD VIEW
            # ---------------------------------------------------------
            if category:
                cur.execute("SELECT COUNT(*) FROM product_requests WHERE product_category = %s;", (category,))
                total_req = cur.fetchone()[0]
                cur.execute("""
                    SELECT AVG(ap.sustainability_score) FROM ai_predictions ap
                    JOIN product_requests pr ON ap.request_id = pr.request_id WHERE pr.product_category = %s;
                """, (category,))
                avg_score = round(cur.fetchone()[0] or 0, 1)
                cur.execute("""
                    SELECT SUM(ap.predicted_co2) FROM ai_predictions ap
                    JOIN product_requests pr ON ap.request_id = pr.request_id WHERE pr.product_category = %s;
                """, (category,))
                total_co2 = round(cur.fetchone()[0] or 0, 2)
                cur.execute("""
                    SELECT ap.recommended_material, COUNT(*) as count FROM ai_predictions ap
                    JOIN product_requests pr ON ap.request_id = pr.request_id WHERE pr.product_category = %s
                    GROUP BY ap.recommended_material ORDER BY count DESC LIMIT 5;
                """, (category,))
                top_materials = cur.fetchall()
                report_title = f"Sustainability Report ({category})"
                filename = f"EcoPackAI_{category}_Report.pdf"
            else:
                # Global
                cur.execute("SELECT COUNT(*) FROM product_requests;")
                total_req = cur.fetchone()[0]
                cur.execute("SELECT AVG(sustainability_score) FROM ai_predictions;")
                avg_score = round(cur.fetchone()[0] or 0, 1)
                cur.execute("SELECT SUM(predicted_co2) FROM ai_predictions;")
                total_co2 = round(cur.fetchone()[0] or 0, 2)
                cur.execute("""
                    SELECT recommended_material, COUNT(*) as count FROM ai_predictions 
                    GROUP BY recommended_material ORDER BY count DESC LIMIT 5;
                """)
                top_materials = cur.fetchall()
                report_title = "EcoPackAI Global Sustainability Report"
                filename = "EcoPackAI_Summary.pdf"

            # --- FETCH ANALYTICS DATA FOR CHARTS ---
            # 1. Material Distribution (Pie)
            mat_labels = [row[0] for row in top_materials]
            mat_counts = [row[1] for row in top_materials]
            
            # 2. Impact Trends (Line)
            impact_data = get_impact_trends(cur) # Returns {'labels': [], 'data': []}
            
            # 3. Versatility (Bar) - Python Logic
            cat_counts = {}
            # FILTERED to match UI inputs only
            valid_ui_categories = list(CATEGORY_DESCRIPTIONS.keys())
            cats = valid_ui_categories # Show all 8
            
            for cat_item in cats: cat_counts[cat_item] = 0
            
            for m in ALL_CANDIDATES:
                if 'All' in m.get('suitable_for', []):
                    for cat_item in cats: cat_counts[cat_item] += 1
                else:
                    for cat_item in m.get('suitable_for', []):
                        if cat_item in cat_counts: cat_counts[cat_item] += 1
            
            sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
            vers_labels = [x[0] for x in sorted_cats]
            vers_counts = [x[1] for x in sorted_cats]

            # --- GENERATE CHARTS (Matplotlib) ---
            # Chart 1: Popularity (Pie)
            plt.figure(figsize=(5, 4))
            plt.pie(mat_counts, labels=mat_labels, autopct='%1.1f%%', colors=['#2e7d32', '#66bb6a', '#a5d6a7', '#ffca28', '#ef5350'])
            plt.title('Material Popularity (Top 5)')
            img_buf_pie = io.BytesIO()
            plt.savefig(img_buf_pie, format='png')
            img_buf_pie.seek(0)
            plt.savefig(img_buf_pie, format='png')
            img_buf_pie.seek(0)
            plt.close()
            
            # --- NEW: Chart 4: Requests by Category (Bar) ---
            cur.execute("SELECT product_category, COUNT(*) FROM product_requests GROUP BY product_category ORDER BY COUNT(*) DESC;")
            cat_reqs = cur.fetchall()
            cr_labels = [row[0] for row in cat_reqs]
            cr_counts = [row[1] for row in cat_reqs]
            
            plt.figure(figsize=(6, 3))
            plt.bar(cr_labels, cr_counts, color='#ffb74d', alpha=0.8)
            plt.title('Requests by Category')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            img_buf_cat_req = io.BytesIO()
            plt.savefig(img_buf_cat_req, format='png')
            img_buf_cat_req.seek(0)
            plt.close()
            
            # Chart 2: Impact Trend (Line)
            plt.figure(figsize=(6, 3))
            plt.plot(impact_data['labels'], impact_data['data'], marker='o', color='#2e7d32', linewidth=2)
            plt.fill_between(impact_data['labels'], impact_data['data'], color='#2e7d32', alpha=0.1)
            plt.title('Est. CO2 Savings Over Time (kg)')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            img_buf_line = io.BytesIO()
            plt.savefig(img_buf_line, format='png')
            img_buf_line.seek(0)
            plt.close()
            
            # Chart 3: Versatility (Bar)
            plt.figure(figsize=(7, 3.5))
            plt.bar(vers_labels, vers_counts, color='#36a2eb', alpha=0.7)
            plt.title('Materials Available per Category (Capabilities)')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            img_buf_bar = io.BytesIO()
            plt.savefig(img_buf_bar, format='png')
            img_buf_bar.seek(0)
            plt.close()

            # --- BUILD PDF PAGE 1 ---
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, report_title)
            
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 80, f"Total Requests: {total_req}  |  Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
            
            # KPI Box
            c.rect(50, height - 160, 500, 60, fill=0)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(70, height - 120, "Avg Score")
            c.drawString(250, height - 120, "Est. CO2 Savings")
            c.setFont("Helvetica", 14)
            c.setFillColorRGB(0, 0.5, 0)
            c.drawString(70, height - 140, str(avg_score))
            c.drawString(250, height - 140, f"{total_co2} kg")
            c.setFillColorRGB(0, 0, 0)
            
            # Charts Page 1
            c.drawImage(ImageReader(img_buf_pie), 50, height - 420, width=250, height=200)
            c.drawImage(ImageReader(img_buf_line), 310, height - 420, width=250, height=150)
            
            # New Chart 4 (Bottom Left)
            c.drawImage(ImageReader(img_buf_cat_req), 50, height - 600, width=500, height=180)
            
            # Top Materials Text
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 450, "Top Recommended Materials (Text Summary):")
            c.setFont("Helvetica", 10)
            y = height - 470
            for mat, count in top_materials:
                c.drawString(70, y, f"- {mat}: {count}")
                y -= 15
            
            c.showPage() # End Page 1
            
            # --- BUILD PDF PAGE 2 (Capabilities) ---
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Model Knowledge Base (Capabilities)")
            
            # Chart 3
            c.drawImage(ImageReader(img_buf_bar), 50, height - 300, width=500, height=200)
            
            # Lists
            y_start = height - 320
            
            # Supported Categories
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_start, "Supported Categories:")
            c.setFont("Helvetica", 9)
            # Use display names for the text list
            cats_display = [CATEGORY_DESCRIPTIONS.get(c, c) for c in cats]
            text_cats = ", ".join(cats_display)
            # Simple word wrap logic
            import textwrap
            lines = textwrap.wrap(text_cats, width=90)
            y = y_start - 15
            for line in lines:
                c.drawString(50, y, line)
                y -= 12
                
            y -= 20
            
            # Known Materials
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Known Materials:")
            c.setFont("Helvetica", 9)
            mat_names = [m['name'] for m in ALL_CANDIDATES]
            text_mats = ", ".join(mat_names)
            lines = textwrap.wrap(text_mats, width=90)
            y -= 15
            for line in lines:
                c.drawString(50, y, line)
                y -= 12
                
            y -= 20
            
            # Valid Ranges
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Valid Training Ranges:")
            c.setFont("Helvetica", 10)
            c.drawString(50, y - 20, "Weight: 10g - 5000g")
            c.drawString(200, y - 20, "Price: INR 50 - INR 10000")

        cur.close()
        conn.close()
        
        c.showPage()
        c.save()
        output.seek(0)
        return send_file(output, as_attachment=True, download_name=filename, mimetype="application/pdf")

    except Exception as e:
        print(f"❌ Export PDF Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Category Descriptions for UI/Report
CATEGORY_DESCRIPTIONS = {
    "Electronics": "Electronics (e.g., Phones, Laptops)",
    "Food": "Food (e.g., Snacks, Grains)",
    "Cosmetics": "Cosmetics (e.g., Creams, Make-up)",
    "Pharmaceuticals": "Pharmaceuticals (e.g., Medicines)",
    "Liquids": "Liquids (e.g., Juices, Detergents)",
    "Textiles": "Textiles (e.g., Clothing, Bags)",
    "Furniture": "Furniture (e.g., Chairs, Tables)",
    "Eco-Gifts": "Eco-Gifts (e.g., Soaps, Accessories)"
}

@app.route("/api/dataset-info", methods=["GET"])
def dataset_info():
    try:
        # Categories supported by the model (via LabelEncoder)
        # FILTERED to match UI inputs only
        valid_ui_categories = list(CATEGORY_DESCRIPTIONS.keys())
        
        # DISPLAY ALL 8 Categories (Bypass model intersection check)
        categories = valid_ui_categories
        
        # Display strings
        categories_display = [CATEGORY_DESCRIPTIONS.get(c, c) for c in categories]
        
        # Materials known to the system (Full Objects for visualization)
        materials_full = ALL_CANDIDATES
        
        # Feature ranges
        ranges = {
            "weight_g": {"min": 10, "max": 5000, "avg": 850},
            "price_inr": {"min": 50, "max": 10000, "avg": 1200}
        }
        
        return jsonify({
            "status": "success",
            "categories": categories,
            "categories_display": categories_display, # Full descriptions
            "materials": [m['name'] for m in materials_full],
            "materials_detailed": materials_full, 
            "ranges": ranges
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
