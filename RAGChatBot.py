
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough


from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_anthropic import ChatAnthropic

import huggingface_hub as hf_hub

from os import getenv
import dotenv
dotenv.load_dotenv()


HF_TOKEN = getenv('HF_TOKEN')
assert HF_TOKEN, "A valid HuggingFace token is required to be set as <HF_TOKEN>."
hf_hub.login(HF_TOKEN,)
ANTHROPIC_API_KEY = getenv('ANTHROPIC_API_KEY')
LANGCHAIN_API_KEY = getenv('LANGCHAIN_API_KEY')
LANGCHAIN_ENDPOINT = getenv('LANGCHAIN_ENDPOINT')
assert LANGCHAIN_API_KEY, "An API key for LangChainSmith is required to be set as <LANGCHAIN_API_KEY>."


PLACES_PATH = "data/places.csv"
REVIEWS_PATH = "data/reviews.csv"
LLM_MODEL = "anthropic::claude-3-opus-20240229"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_CACHE_STORE="./cache/"
FAISS_REVIEWS_PATH = "faiss_index_euclidean"
FAISS_INDEX_NAME = "index"
FAISS_DISTANCE_STRATEGY='EUCLIDEAN_DISTANCE'


### Get device
try:
    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'
except ImportError:
    device = 'cpu'

def get_hf_embedding_model(embedding_model_name,
                           cache_embeddings_store,
                           device='cpu',
                           normalize_embeddings=False,
                           ):
  model_kwargs = {'device': device}
  encode_kwargs = {'normalize_embeddings': normalize_embeddings} # Set `True` for cosine similarity
  embedding_model = HuggingFaceEmbeddings(
      model_name=embedding_model_name,
      model_kwargs=model_kwargs,
      encode_kwargs=encode_kwargs
      )
  store = LocalFileStore(cache_embeddings_store)
  embedding_model = CacheBackedEmbeddings.from_bytes_store(
                    embedding_model, store)
  return embedding_model

embedding_model = get_hf_embedding_model(EMBEDDING_MODEL_NAME,
                                         EMBEDDINGS_CACHE_STORE,
                                         device=device,
                                         normalize_embeddings=False)

def get_anthropic_api_llm(model_name):
  llm = ChatAnthropic(model_name=model_name, anthropic_api_key=ANTHROPIC_API_KEY,)

  return llm

model_type, _, model_name = LLM_MODEL.partition('::')
llm = get_anthropic_api_llm(model_name)

review_template_str = """
Your job is to use Google Map restaurants and bars reviews to help people find best places to go for a meal or a drink.
Use the following information and reviews to answer the questions.
If you don't know an answer based on the context, say you don't know. Answer context:
{context}
"""
review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=review_template_str
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

vector_db = FAISS.load_local(folder_path=FAISS_REVIEWS_PATH,
                             embeddings=embedding_model,
                             index_name=FAISS_INDEX_NAME)
reviews_retriever = vector_db.as_retriever(search_kwargs={'k': 10,
                                                        #   'fetch_k': 50,
                                                          })
review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | llm
    # | StrOutputParser()
)

