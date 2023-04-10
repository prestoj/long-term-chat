from chatbot import ChatGPT
from coder import CoderGPT
from config import PINECONE_TABLE_NAME, GPT_MODEL

if __name__ == "__main__":
    # chatgpt = ChatGPT(gpt_model=GPT_MODEL)
    # chatgpt.run()
    codergpt = CoderGPT(gpt_model=GPT_MODEL)
    codergpt.upload_files_in_directory('.')
    codergpt.run()
