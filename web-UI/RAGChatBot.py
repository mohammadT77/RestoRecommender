from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

from os import getenv
import dotenv
dotenv.load_dotenv()

GOOGLE_API_KEY = getenv('GOOGLE_API_KEY')

PLACES_PATH = "data_v2/places.csv"
REVIEWS_PATH = "data_v2/reviews.csv"
LLM_MODEL_NAME = "gemini-1.5-flash" #"gemini-pro"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
EMBEDDINGS_CACHE_STORE="./cache/"
FAISS_REVIEWS_PATH_EUCLIDEAN = "faiss_index_euclidean"
FAISS_REVIEWS_PATH_COSINE = "faiss_index_cosine"
FAISS_INDEX_NAME = "index"
FAISS_DISTANCE_STRATEGY='EUCLIDEAN_DISTANCE'

embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
store = LocalFileStore(EMBEDDINGS_CACHE_STORE)
embedding_model = CacheBackedEmbeddings.from_bytes_store(embedding_model, store)

vector_db = FAISS.load_local(folder_path=FAISS_REVIEWS_PATH_EUCLIDEAN,
                             embeddings=embedding_model,
                             index_name=FAISS_INDEX_NAME,
                             allow_dangerous_deserialization=True)

llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME)

review_template_str = """
Your job is to use Google Map restaurants and bars reviews to help people find best places to go for a meal or a drink.
Use the following information and reviews to answer the questions. if the question is not about restaurants,
then kindly tell the user that you can only provide assistance and answer questions related to restaurants. if the user doesn't mention the city name,
always assume the user is asking about Padova.
If the context provided to you does not contain the answer of the question, tell the user that there is no answer in the reviews.
Answer context:
{context}
"""

system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=review_template_str
    )
)

human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [system_prompt, human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

reviews_retriever = vector_db.as_retriever(search_kwargs={'k': 20,})

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | llm
    | StrOutputParser()
)