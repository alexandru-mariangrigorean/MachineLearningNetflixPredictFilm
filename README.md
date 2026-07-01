# Movie Recommendation Engine (FastAPI & SVD)

This repository contains a REST API service built with **FastAPI** that delivers personalized movie recommendations to users. The system leverages a collaborative filtering **SVD (Singular Value Decomposition)** model trained on the Netflix dataset and serialized using Python's `pickle` module.

---

## Key Features
* **High Performance:** Developed using FastAPI for low latency, high throughput, and automatic OpenAPI documentation.
* **Predictive Scoring:** Estimates the rating a user would give to movies they haven't interacted with, sorting the results to return the top 10 most relevant recommendations.
* **In-Memory Operations:** Loads movie metadata (`movie_titles.csv`) into memory during application startup to minimize runtime I/O overhead.

---

## Technologies Used
* **Python 3.8+**
* **FastAPI** (Web Framework)
* **Uvicorn** (ASGI Server)
* **Pandas** (Data Manipulation)
* **Scikit-Surprise** (Recommendation Systems Library used for SVD training)
* **Pickle** (For model serialization/deserialization)

---

## Project Structure
```text
├── app.py                  # Main FastAPI application source code
├── model_svd_netflix.pkl   # Pre-trained SVD model file
├── movie_titles.csv        # Dataset containing movie metadata (ID, Year, Title)
├── requirements.txt        # Python project dependencies
└── README.md               # Project documentation
