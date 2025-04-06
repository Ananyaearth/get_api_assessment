import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from typing import Optional

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommendation API")

# Load data and models
try:
    df = pd.read_csv("shl_catalog_detailed.csv")
    index = faiss.read_index("shl_assessments_index.faiss")
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    raise Exception(f"Failed to load: {e}")

# GET endpoint
@app.get("/recommend")
def recommend(query: str = Query(..., description="Job description or keyword"),
              top_k: int = Query(5, description="Number of recommendations", ge=1, le=10)):
    query_embedding = model.encode([query])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        results.append({
            "Assessment Name": row['Individual Test Solutions'],
            "URL": row['URL'],
            "Remote Testing": row['Remote Testing (y/n)'],
            "Adaptive/IRT": row['Adaptive/IRT (y/n)'],
            "Duration": row['Assessment Length']
        })

    return {"results": results}
