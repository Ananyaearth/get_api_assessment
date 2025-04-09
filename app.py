import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Body
import os
import google.generativeai as genai

# Initialize FastAPI app
app = FastAPI(title="SHL Assessment Recommendation API")

# Configure Gemini API (assuming API key is set via environment or secrets)
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyASKTzSNuMbJMdZWr81Xuw2hS1Poe3acZo")  # Match Streamlit's fallback
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')  # Match Streamlit's model

# Test Type mapping (consistent with your Streamlit version)
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
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Renamed to avoid conflict
except Exception as e:
    raise Exception(f"Failed to load: {e}")

# Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# LLM preprocessing function (from Streamlit)
def llm_shorten_query(query):
    prompt = "Extract all technical skills from query as space-separated list, max 10: "
    try:
        response = model.generate_content(prompt + query)
        shortened = response.text.strip()
        words = shortened.split()
        return " ".join(words[:10]) if words else query
    except Exception as e:
        print(f"Query LLM error: {e}")  # Log error instead of st.error
        return query

# POST endpoint with original logic adapted to spec
@app.post("/recommend")
def recommend(request: dict = Body(...)):
    top_k = max(1, min(10, request.get("top_k", 5)))  # Variable top_k, default 5, range 1-10
    processed_query = llm_shorten_query(request["query"])  # Preprocess query with LLM
    print(f"Processed Query: {processed_query}")  # Debug (replacing st.write)
    query_embedding = embedding_model.encode([processed_query])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for idx in indices[0]:
        row = df.iloc[idx]
        duration_str = row['Assessment Length']
        duration = int(''.join(filter(str.isdigit, str(duration_str)))) if duration_str else 0
        adaptive = "Yes" if row['Adaptive/IRT (y/n)'].lower() == 'y' else "No"
        remote = "Yes" if row['Remote Testing (y/n)'].lower() == 'y' else "No"
        
        # Use suggested logic for test_type (unchanged)
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
