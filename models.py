from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.llms import HuggingFacePipeline

from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

## To authenticate HF account
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
# Set the token for Hugging Face
from huggingface_hub import login
login(HF_TOKEN)

device = f'cuda' if cuda.is_available() else 'cpu'
def get_review_data(file_path, source_column, metadata_columns):
  loader = CSVLoader(file_path= file_path, source_column= source_column,
                    encoding = 'utf-8', metadata_columns= metadata_columns)
  documents = loader.load()
  return documents

def get_hf_embedding_model(embedding_model_name,
                           cache_embeddings_store,
                           normalize_embeddings=False,):
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


def get_vector_database(documents, embedding_model,
                        distance_strategy='EUCLIDEAN_DISTANCE'):

  vector_database = FAISS.from_documents(
      documents, embedding_model,
      distance_strategy= distance_strategy
      )
  return vector_database


def get_hf_llm(model_name):

  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
  )
  model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, )
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  pipe = pipeline(
      model=model,
      tokenizer=tokenizer,
      return_full_text=True,  # langchain expects the full text
      task='text-generation',
      # we pass model parameters here too
      temperature=0.0001,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
      max_new_tokens=512,  # mex number of tokens to generate in the output
      repetition_penalty=1.1  # without this output begins repeating
  )

  llm = HuggingFacePipeline(pipeline=pipe)
  return llm
