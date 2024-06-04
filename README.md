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

**Setting Up the Environment**

1. **Create a Virtual Environment (Recommended):**
   - Virtual environments isolate project dependencies and prevent conflicts with other Python installations.
   - To create a new virtual environment named `venv`:
     - **Linux/macOS:** `python3 -m venv venv`
     - **Windows:** `python -m venv venv`
   - Activate the virtual environment:
     - **Linux/macOS:** `source venv/bin/activate`
     - **Windows:** `venv\Scripts\activate`

2. **Install Required Libraries:**
   - Navigate to the project's root directory.
   - Assuming you have a `requirements.txt` file listing project dependencies, install them using pip:
     - `pip install -r requirements.txt`

**Running the Project**

1. **Make sure the virtual environment is activated (if used).**
2. **Navigate to the project's root directory in your terminal.**
3. **Run Jupyter Notebooks:**
   - Open a terminal or command prompt within the project directory.
   - Use `jupyter notebook` to launch the Jupyter Notebook server.
   - Interact with the notebooks (`places.ipynb`, `rag_chatbit.ipynb`, and `experiments.ipynb`) to perform data collection, model creation, and further experiments.
4. **Run Web UI (if applicable):**
   - To launch the UI:
      1. Navigate to the `web-UI` directory in your terminal.
      2. Run the command: `python app.py`
      3. Open your web browser and visit the following URL: `http://127.0.0.1:5000/`
   - The UI interface created in this manned is just for development environment. However, this is enough for the purpose of this project. 

**Using the Gemini API**

If you intend to leverage the Gemini API for their embedding and LLM model, you'll need an API key. Here's how to obtain one:

1. Visit the Google AI Studio: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click on "Get API key" and follow the on-screen instructions.
3. Once you have the API key, store it securely inside a `.env` file.
4. Update the `rag_chatbit.ipynb` notebook with your API key where necessary.

**Alternative Embedding Models and LLM Models**

While the `rag_chatbit.ipynb` notebook demonstrates using the Gemini API, you can explore alternative embedding models and LLM architectures. Here are some resources to get you started:

* **Embedding Models:**
    * Sentence Transformers: [https://huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)
* **LLM Models:**
    * Hugging Face Transformers Library: [https://huggingface.co/docs/transformers/en/index](https://huggingface.co/docs/transformers/en/index)
    * OpenAI API: [https://openai.com/](https://openai.com/) (requires account creation)

By following these instructions, you'll establish a clean environment, install necessary dependencies, and be ready to run the RestoRecommender project effectively.

**References:**
- https://pypi.org/project/googlemaps/
- https://github.com/googlemaps/google-maps-services-python/blob/master/googlemaps/places.py
- https://python.langchain.com/v0.1/docs/get_started/introduction
- https://ai.google.dev/gemini-api/docs
- https://github.com/facebookresearch/faiss
- https://ai.google.dev/gemini-api/docs/api-key