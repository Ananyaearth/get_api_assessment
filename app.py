import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Body
import os

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommendation API")

# Test Type mapping
test_type_map = {
    'A': 'Ability & Aptitude',
    'B': 'Biodata & Situational Judgement',
    'C': 'Competencies',
    'D': 'Development & 360',
    'E': 'Assessment Exercises',
    'K': 'Knowledge & Skills',
    'P': 'Personality & Behaviour',
    'S': 'Simulations'
}

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

# POST endpoint with original logic adapted to spec
@app.post("/recommend")
def recommend(request: dict = Body(...)):
    top_k = max(1, min(10, request.get("top_k", 5)))  # Variable top_k, default 5, range 1-10
    query_embedding = model.encode([request["query"]])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        duration_str = row['Assessment Length']
        duration = int(''.join(filter(str.isdigit, str(duration_str)))) if duration_str else 0
        adaptive = "Yes" if row['Adaptive/IRT (y/n)'].lower() == 'y' else "No"
        remote = "Yes" if row['Remote Testing (y/n)'].lower() == 'y' else "No"
        
        # Use suggested logic for test_type
        test_types = str(row['Test Type'])  # Ensure it’s a string
        test_type = [test_type_map.get(abbrev.strip(), abbrev.strip()) for abbrev in test_types.split()]

        results.append({
            "url": row['URL'],
            "adaptive_support": adaptive,
            "description": row['Description'],
            "duration": duration,
            "remote_support": remote,
            "test_type": test_type
        })

    return {"recommended_assessments": results}
