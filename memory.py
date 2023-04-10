from datetime import datetime
import pinecone
from openai_tools import get_embedding
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_TABLE_NAME
import uuid

class Memory():
    def __init__(self):
        self.table_name = PINECONE_TABLE_NAME
        self.dimension = 1536
        self.metric = "cosine"
        self.pod_type = "p1"

        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

        if self.table_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.table_name, dimension=self.dimension, metric=self.metric, pod_type=self.pod_type)
        self.index = pinecone.Index(self.table_name)

    def upload_message_response_pair(self, message, response):
        self.index.upsert([(str(uuid.uuid4()), get_embedding(message + response), {"message": message, "response": response, "timestamp": datetime.now()})])

    def upload_code(self, file, code, start_line, end_line):
        self.index.upsert([(str(uuid.uuid4()), get_embedding(code), {"file": file, "code": code, "start_line": start_line, "end_line": end_line})])

    def search(self, vector, k=5):
        message_response_pairs = self.index.query(vector, top_k=k, include_metadata=True)
        results = [
            {"message": result.metadata["message"], 
            "response": result.metadata["response"], 
            "timestamp": result.metadata["timestamp"], 
            "similarity": result.score
        } for result in message_response_pairs.matches]
        
        return results
        

