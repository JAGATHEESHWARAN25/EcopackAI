import psycopg2
from psycopg2 import sql

# Database credentials
DB_NAME = "ecopackai"
USER = "postgres"
PASSWORD = "Jaga@123"
HOST = "localhost"
PORT = "5432"

def create_schema():
    print("🔌 Connecting to PostgreSQL...")
    try:
        # Connect to default 'postgres' db to check/create target db
        conn = psycopg2.connect(dbname="postgres", user=USER, password=PASSWORD, host=HOST, port=PORT)
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        if not cur.fetchone():
            print(f"📦 Database '{DB_NAME}' not found. Creating...")
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            print(f"✅ Database '{DB_NAME}' created.")
        else:
            print(f"ℹ️ Database '{DB_NAME}' already exists.")
        
        cur.close()
        conn.close()
        
        # Connect to the target database
        conn = psycopg2.connect(dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT)
        conn.autocommit = True
        cur = conn.cursor()
        
        print("🛠 Setting up tables...")
        
        # Table: product_requests (Stores user inputs)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS product_requests (
                request_id SERIAL PRIMARY KEY,
                product_category VARCHAR(100),
                weight_g FLOAT,
                price_inr FLOAT,
                format VARCHAR(50),
                protection_level INT,
                bulkiness_factor FLOAT,
                shelf_life_days INT,
                request_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Table: ai_predictions (Stores the AI results linked to requests)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_predictions (
                prediction_id SERIAL PRIMARY KEY,
                request_id INT REFERENCES product_requests(request_id),
                recommended_material VARCHAR(100),
                predicted_cost FLOAT,
                predicted_co2 FLOAT,
                sustainability_score FLOAT,
                effectiveness_rating VARCHAR(50)
            );
        """)
        
        print("✅ Tables 'product_requests' and 'ai_predictions' are ready.")
        
        cur.close()
        conn.close()
        print("🚀 Database schema initialization complete!")

    except Exception as e:
        print(f"❌ Error setting up database: {e}")

if __name__ == "__main__":
    create_schema()
