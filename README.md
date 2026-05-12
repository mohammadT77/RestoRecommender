# RestoRecommender

> AI-powered restaurant and bar recommendations driven by RAG over real customer reviews.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-green?style=flat-square)](https://python.langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## Overview

RestoRecommender scrapes venue data and reviews from the Google Maps Places API, stores them in a FAISS vector database, and answers natural language queries through a RAG pipeline backed by Google Gemini. Ask it "find me a quiet bar with good cocktails near the center" and it retrieves relevant reviews, synthesizes them, and gives you a reasoned recommendation.

---

## Architecture

```
Google Maps Places API
        │
        ▼
  places.ipynb  ──►  data/places.csv
                      data/reviews.csv
        │
        ▼
rag_chatbit.ipynb
  ├─ Load CSV
  ├─ Chunk & embed reviews  ──►  FAISS vector store
  └─ LangChain RAG chain (Gemini LLM)
        │
        ▼
  web-UI/app.py  (Flask)
  └─  http://127.0.0.1:5000/
```

---

## Prerequisites

You need two API keys:

| Key | Where to get it |
|---|---|
| Google Maps Places API | [console.cloud.google.com](https://console.cloud.google.com) → Maps → Places API |
| Google Gemini API | [aistudio.google.com](https://aistudio.google.com/app/apikey) |

---

## Installation

```bash
git clone https://github.com/amin-tehrani/RestoRecommender.git
cd RestoRecommender
pip install langchain langchain-google-genai google-maps faiss-cpu flask jupyter
```

Or install from a requirements file if present:

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Collect venue data

Open `places.ipynb` in Jupyter and set your Google Maps API key. Running all cells scrapes nearby restaurants/bars and writes them to:

```
data/places.csv    # venue info
data/reviews.csv   # customer reviews
```

### Step 2 — Build the RAG model

Open `rag_chatbit.ipynb`. Set your Gemini API key, then run all cells. This:
- Loads the CSVs
- Chunks and embeds reviews into a FAISS vector store
- Wires up a LangChain RAG chain over the Gemini LLM
- Runs a few demo queries to validate the pipeline

### Step 3 — Launch the web UI

```bash
cd web-UI
python app.py
```

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser and start asking for recommendations.

### (Optional) Batch Q&A

Put your questions in `questions.txt`, then run `experiments.ipynb` to generate answers and save them to `Question-Answer.txt`.

---

## Tech stack

| Component | Library |
|---|---|
| Data collection | `googlemaps` (Places API) |
| LLM | Google Gemini via `langchain-google-genai` |
| Vector store | FAISS (`faiss-cpu`) |
| RAG orchestration | LangChain |
| Web UI | Flask |

---

## License

MIT © Amin Tehrani
