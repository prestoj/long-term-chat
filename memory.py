from datetime import datetime
import pinecone
from openai_tools import get_embedding, get_importance_of_interaction, get_insights
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_TABLE_NAME
import uuid
import numpy as np

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
        importance = get_importance_of_interaction(message, response)
        self.index.upsert([(str(uuid.uuid4()), get_embedding(message + response), {"message": message, "response": response, "importance": importance, "timestamp": datetime.now()})])

    def reflect(self, messages):
        insights = get_insights(messages)
        for insight in insights:
            self.index.upsert([(str(uuid.uuid4()), get_embedding(insight["content"]), {"insight": insight["content"], "importance": insight["importance"], "timestamp": datetime.now()})])

    def search(self, vector, n=5):
        message_response_pairs = self.index.query(vector, top_k=1000, include_metadata=True)
        results = [
            {
            "message": result.metadata["message"] if "message" in result.metadata else None, 
            "response": result.metadata["response"] if "response" in result.metadata else None, 
            "insight": result.metadata["insight"] if "insight" in result.metadata else None,
            "timestamp": result.metadata["timestamp"], 
            "importance": result.metadata["importance"], 
            "similarity": result.score
        } for result in message_response_pairs.matches]

        if results:
            # Compute the values for each dimension
            days_since_values = []
            importance_values = []
            similarity_values = []
            for result in results:
                days_since = (datetime.now() - result["timestamp"]).days
                days_since_values.append(np.exp(-0.99 * days_since))
                importance_values.append(result["importance"])
                similarity_values.append(result["similarity"])

            # Calculate the min and max values for each dimension
            min_days_since, max_days_since = min(days_since_values), max(days_since_values)
            min_importance, max_importance = min(importance_values), max(importance_values)
            min_similarity, max_similarity = min(similarity_values), max(similarity_values)

            # Apply min-max scaling and compute the scaled score
            epsilon = 1e-8
            for i, result in enumerate(results):
                days_since_scaled = (days_since_values[i] - min_days_since) / (max_days_since - min_days_since + epsilon)
                importance_scaled = (importance_values[i] - min_importance) / (max_importance - min_importance + epsilon)
                similarity_scaled = (similarity_values[i] - min_similarity) / (max_similarity - min_similarity + epsilon)
                result["score"] = 1/3 * days_since_scaled + 1/3 * importance_scaled + 1/3 * similarity_scaled

            # Sort the results based on the score in descending order
            results.sort(key=lambda x: x["score"], reverse=True)

        # Return the top n results
        return results[:n]

        

