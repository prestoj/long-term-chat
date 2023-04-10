from chatbot import ChatGPT
from config import PINECONE_TABLE_NAME, GPT_MODEL

if __name__ == "__main__":
    chatgpt = ChatGPT(gpt_model=GPT_MODEL, table_name=PINECONE_TABLE_NAME)
    chatgpt.run()