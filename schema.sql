-- Table for logging user requests
CREATE TABLE IF NOT EXISTS product_requests (
    request_id SERIAL PRIMARY KEY,
    product_category VARCHAR(100),
    weight_g FLOAT,
    price_inr FLOAT,
    format VARCHAR(50),
    protection_level FLOAT,
    bulkiness_factor FLOAT,
    shelf_life_days FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing AI predictions/recommendations
CREATE TABLE IF NOT EXISTS ai_predictions (
    prediction_id SERIAL PRIMARY KEY,
    request_id INT REFERENCES product_requests(request_id),
    recommended_material VARCHAR(100),
    predicted_cost FLOAT,
    predicted_co2 FLOAT,
    sustainability_score FLOAT,
    effectiveness_rating VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
