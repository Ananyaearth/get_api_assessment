# SHL Assessment Recommendation API

## Overview

This project implements a **FastAPI-based API** to recommend SHL assessments based on user queries. The API utilizes **FAISS** for fast similarity search, **Sentence Transformers** to generate sentence embeddings, and serves the recommendations through a RESTful interface powered by **FastAPI**. The system is hosted on **Render.com** for easy accessibility and real-time recommendations.

## Objective

The goal of this API is to provide personalized SHL assessment recommendations based on user input, such as a job description, skill, or role. Users can interact with the API by sending HTTP GET requests to retrieve the most relevant SHL assessments. The system leverages text similarity matching by encoding the query into an embedding and searching for the closest matching assessments in the indexed dataset.

## Technologies Used

- **FastAPI**: A modern web framework for building APIs with Python, known for its speed and ease of use.
- **Pandas**: For data manipulation and handling the SHL assessments CSV file.
- **NumPy**: For array operations and numerical calculations.
- **FAISS**: A library for fast similarity search, used to efficiently search for similar assessments based on embeddings.
- **Sentence Transformers**: A library for generating sentence embeddings, allowing for semantic search based on textual similarity.
- **Render.com**: Platform used to deploy the FastAPI application for public access.

## Workflow

### 1. API Initialization
The **FastAPI** application is initialized with a title: "SHL Assessment Recommendation API." The necessary libraries (Pandas, FAISS, Sentence Transformers, etc.) are imported, and the system attempts to load the necessary resources:
- **SHL Assessment Data**: Loaded from `shl_catalog_detailed.csv`.
- **FAISS Index**: Loaded from `shl_assessments_index.faiss`, which contains the pre-indexed embeddings of SHL assessments.
- **Sentence Transformer Model**: The `all-MiniLM-L6-v2` model is loaded for generating embeddings from the input queries.

### 2. API Endpoint (`/recommend`)
The main feature of the API is the `/recommend` endpoint, which accepts two query parameters:
- **query**: A required string parameter that represents the job description, skill, or role for which the user is looking for relevant SHL assessments.
- **top_k**: An optional integer parameter that specifies how many top recommendations the user wants. The default value is set to 5, with a valid range between 1 and 10.

The API works as follows:
1. **Query Embedding**: The input query is encoded into an embedding using the **Sentence Transformer** model (`all-MiniLM-L6-v2`).
2. **FAISS Search**: The query embedding is compared to the pre-indexed embeddings in the FAISS index to find the most similar SHL assessments.
3. **Results Generation**: The top-k results are retrieved, including details such as:
   - **Assessment Name**: Name of the assessment (linked to the URL).
   - **URL**: Direct link to the assessment.
   - **Remote Testing**: Whether remote testing is supported (yes/no).
   - **Adaptive/IRT**: Whether the assessment uses adaptive testing techniques or Item Response Theory.
   - **Duration**: The length of the assessment.

The results are returned in a structured JSON format.

### 3. Error Handling
If any issues occur during the loading of resources (CSV file, FAISS index, or model), the system raises an exception and stops the execution. This ensures that users can be informed of any errors during initialization.

## Example Request and Response

### Request

```http
GET https://your-app-url.com/recommend?query=software+developer&top_k=5
```

### Response

```json
{
    "results": [
        {
            "Assessment Name": "Coding Test for Developers",
            "URL": "http://example.com/assessment1",
            "Remote Testing": "y",
            "Adaptive/IRT": "y",
            "Duration": "30 minutes"
        },
        {
            "Assessment Name": "Algorithm Proficiency",
            "URL": "http://example.com/assessment2",
            "Remote Testing": "n",
            "Adaptive/IRT": "n",
            "Duration": "45 minutes"
        },
    ]
}
```

### Query Parameters

- **query**: The job description or skill you are interested in (e.g., "software developer").
- **top_k**: Number of results to return, between 1 and 10 (default is 5).

## Hosting and Deployment

The FastAPI application is deployed on **Render.com**, which provides easy deployment of web applications with automatic scaling. This allows users to interact with the API in real-time from anywhere.

## Future Improvements

1. **Expanded Dataset**: Adding more SHL assessments and additional metadata to improve the variety and relevance of recommendations.
2. **Personalization**: Implementing user feedback loops where users can rate assessments, leading to better recommendations over time.
3. **Search Filters**: Adding more search filters to refine recommendations, such as industry, difficulty level, and skill categories.
4. **Authentication**: Adding user authentication for personalized recommendations based on historical data.

## Conclusion

The **SHL Assessment Recommendation API** provides an efficient way to recommend relevant SHL assessments based on a job description, skill, or role. By leveraging **FAISS** for fast similarity search and **Sentence Transformers** for high-quality text embeddings, this system is capable of delivering personalized recommendations in real-time. The API is hosted on **Render.com**, making it easily accessible for users to interact with the recommendation engine.

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Ananyaearth/new_SHL.git
   cd new_SHL
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI app locally:
   ```bash
   uvicorn app:app --reload
   ```

4. Access the API at `http://127.0.0.1:8000` and make requests to the `/recommend` endpoint.

---
