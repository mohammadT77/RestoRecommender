**RestoRecommender: AI-Powered Restaurant Recommendation System**

RestoRecommender leverages the power of artificial intelligence (AI) to suggest restaurants and bars based on user preferences and review analysis. It utilizes Retrieval-Augmented Generation (RAG) to extract meaningful insights from customer reviews.

**Functionality Breakdown:**

* **Data Collection (`places.ipynb`):**
  - This Jupyter Notebook scrapes data from Google Maps's Places API and stores the extracted information in CSV files (`places.csv` and `reviews.csv`) within the `data` directory.

* **RAG LLM Model Creation (`rag_chatbit.ipynb`):**
  - This script outlines the process of building a RAG-based large language model (LLM). The steps involve:
      - Data loading from CSV files
      - Vector database construction
      - Embedding model training
      - LLM model creation
  - The notebook also showcases demonstration use cases and evaluations based on synthetically generated questions.

* **Answer Generation (`experiments.ipynb`):**
  - This Jupyter Notebook utilizes the `questions.txt` file containing user queries to generate corresponding answers and saves them in the `Question-Answer.txt` file. It allows for the inclusion of additional user questions for enhanced personalization.

* **Web User Interface (`web-UI` folder):**
  - This folder houses the necessary files to create a web-based user interface (UI) that facilitates interaction with the RestoRecommender chatbot. It employs the Flask framework.
  - To launch the UI:
      1. Navigate to the `web-UI` directory in your terminal.
      2. Run the command: `python app.py`
      3. Open your web browser and visit the following URL: `http://127.0.0.1:5000/`

**References:**
- https://pypi.org/project/googlemaps/
- https://github.com/googlemaps/google-maps-services-python/blob/master/googlemaps/places.py
- https://python.langchain.com/v0.1/docs/get_started/introduction
- https://ai.google.dev/gemini-api/docs
- https://github.com/facebookresearch/faiss
