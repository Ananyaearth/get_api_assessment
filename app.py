import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Body

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommendation API")

# Load data and models
try:
    df = pd.read_csv("shl_catalog_detailed.csv")
    index = faiss.read_index("shl_assessments_index.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    raise Exception(f"Failed to load: {e}")

# Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# POST endpoint with original logic adapted to spec, no pydantic
@app.post("/recommend")
def recommend(request: dict = Body(...)):
    # Ensure top_k is within spec range (1-10)
    top_k = max(1, min(10, request.get("top_k", 5)))  # Default to 5 if not provided
    
    query_embedding = model.encode([request["query"]])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        
        # Convert duration to integer as required by spec
        duration_str = row['Assessment Length']
        duration = int(''.join(filter(str.isdigit, str(duration_str)))) if duration_str else 0

        # Convert y/n to Yes/No as required by spec
        adaptive = "Yes" if row['Adaptive/IRT (y/n)'].lower() == 'y' else "No"
        remote = "Yes" if row['Remote Testing (y/n)'].lower() == 'y' else "No"

        # Add test_type as required by spec (default value)
        test_type = ["Knowledge & Skills"]

        # Map original fields to required spec fields
        results.append({
            "url": row['URL'],
            "adaptive_support": adaptive,
            "description": row['Individual Test Solutions'],
            "duration": duration,
            "remote_support": remote,
            "test_type": test_type
        })

    return {"recommended_assessments": results}
