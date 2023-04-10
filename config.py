import os
from dotenv import load_dotenv

load_dotenv()

def get_env_variable(var_name, default=None, required=True):
    value = os.getenv(var_name, default)
    if required and not value:
        raise ValueError(f"{var_name} environment variable is missing from .env")
    return value

OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
PINECONE_API_KEY = get_env_variable("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = get_env_variable("PINECONE_ENVIRONMENT")
PINECONE_TABLE_NAME = get_env_variable("PINECONE_TABLE_NAME")
GPT_MODEL = get_env_variable("GPT_MODEL", default="gpt-3.5-turbo", required=False)