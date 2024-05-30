# main.py
from RAGChatBot import review_chain

def generate_response(question):
    result = review_chain.invoke(question)
    
    return result
