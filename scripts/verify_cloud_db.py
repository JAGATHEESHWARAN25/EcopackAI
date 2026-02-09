import psycopg2
import sys

def test_connection():
    print("--- 🌍 Cloud Database Connection Test ---")
    print("Please paste your Render 'External Database URL' below.")
    print("It should look like: postgres://user:password@hostname.render.com/dbname")
    
    db_url = input("\nPaste URL here: ").strip()
    
    if not db_url:
        print("❌ Error: No URL provided.")
        return

    try:
        print(f"\n⏳ Attempting to connect to: {db_url.split('@')[1] if '@' in db_url else 'Invalid URL'}...")
        
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        print("✅ CONNECTION SUCCESSFUL!")
        
        # Check for tables
        print("\n🔍 Checking for required tables...")
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """)
        tables = [row[0] for row in cur.fetchall()]
        
        if 'product_requests' in tables:
            print(f"   - product_requests: ✅ Found")
            
            # Check row count
            cur.execute("SELECT COUNT(*) FROM product_requests;")
            count = cur.fetchone()[0]
            print(f"   - Current Row Count: {count}")

            # TEST INSERT
            print("\n🧪 Testing INSERT permission...")
            try:
                cur.execute("""
                    INSERT INTO product_requests (product_category, weight_g, price_inr, format, protection_level, bulkiness_factor, shelf_life_days)
                    VALUES ('TEST_ENTRY', 100, 500, 'Box', 1, 1, 30)
                    RETURNING request_id;
                """)
                new_id = cur.fetchone()[0]
                conn.commit()
                print(f"   - ✅ INSERT SUCCESSFUL! (Test ID: {new_id})")
                print("   - (You should now see at least 1 row in the DB)")
            except Exception as e:
                print(f"   - ❌ INSERT FAILED: {e}")
                conn.rollback()

        else:
            print(f"   - product_requests: ❌ NOT FOUND (You need to run schema.sql!)")

        if 'ai_predictions' in tables:
            print(f"   - ai_predictions:   ✅ Found")
        else:
            print(f"   - ai_predictions:   ❌ NOT FOUND")

        cur.close()
        conn.close()
        print("\n--- Test Complete ---")

    except Exception as e:
        print("\n❌ CONNECTION FAILED")
        print(f"Error Message: {e}")
        print("\nTips:")
        print("1. Are you using the 'External Database URL'?")
        print("2. Did you copy the password correctly?")
        print("3. Is your IP allowed? (Render usually allows all, but check settings)")

if __name__ == "__main__":
    try:
        test_connection()
    except KeyboardInterrupt:
        print("\nTest cancelled.")
