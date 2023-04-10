import os
import uuid
import openai
from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_TABLE_NAME
import pinecone
from openai_tools import get_embedding, num_tokens_from_messages, num_tokens_from_text

openai.api_key = OPENAI_API_KEY

class CoderGPT():
    def __init__(self, gpt_model):
        self.table_name = PINECONE_TABLE_NAME
        self.dimension = 1536
        self.metric = "cosine"
        self.pod_type = "p1"

        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

        if self.table_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.table_name, dimension=self.dimension, metric=self.metric, pod_type=self.pod_type)
        self.index = pinecone.Index(self.table_name)

        self.index.delete(delete_all=True, namespace="CoderGPT")
        
        self.message_memory = []
        self.gpt_model = gpt_model

    def search(self, vector, k=5):
        code_chunks = self.index.query(vector, top_k=k, include_metadata=True, namespace="CoderGPT")
        results = [
            {"file": result.metadata["file"], 
            "code": result.metadata["code"], 
            "similarity": result.score
        } for result in code_chunks.matches]
        
        return results
        
    def upload_code_embedding(self, file, code):
        self.index.upsert([(str(uuid.uuid4()), get_embedding(code), {"file": file, "code": code})], namespace="CoderGPT")
    
    def upload_files_in_directory(self, directory, max_tokens=500, overlap=100):
        # This will only get the files in the directory, not the files in subdirectories
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)

            if os.path.isfile(file_path):
                self.upload_code(file_path, max_tokens, overlap)

    def upload_code(self, file_path, max_tokens=500, overlap=100):
        try:
            with open(file_path, "r") as f:
                lines = [f"{idx + 1}: {line}" for idx, line in enumerate(f.readlines())]
        except UnicodeDecodeError as e:
            print(f"Skipping {file_path} due to UnicodeDecodeError: {e}")
            return
        print(f"Uploading {file_path}...")

        line_tokens = [num_tokens_from_text(line, self.gpt_model) for line in lines]
        num_lines = len(lines)

        start_index = 0
        while start_index < num_lines:
            code_chunk = []
            end_index = start_index

            file_path_line = f"File: {file_path}\n"
            code_chunk.append(file_path_line)
            current_tokens = num_tokens_from_text("".join(code_chunk), self.gpt_model)

            while end_index < num_lines and current_tokens + line_tokens[end_index] <= max_tokens:
                code_chunk.append(lines[end_index])
                current_tokens = num_tokens_from_text("".join(code_chunk), self.gpt_model)
                end_index += 1

            # Upload the code_chunk
            self.upload_code_embedding(file_path, "".join(code_chunk))

            # If end_index reaches the end of the file, break out of the loop
            if end_index == num_lines:
                break

            # Move the window over by max_tokens - overlap tokens
            tokens_to_move = max_tokens - overlap
            tokens_moved = 0

            while start_index < num_lines and tokens_moved < tokens_to_move:
                tokens_moved += line_tokens[start_index]
                start_index += 1

    def send_message(self, message, temperature=0.7, n=1, max_tokens=500):
        messages = [
            {"role": "system", "content": f"You are CoderGPT, a large language model designed to write code. You will see code snippets that are most similar to the user input to assist in helping the user."},
        ]

        codebase_snippets = self.search(get_embedding(message))

        for snippet in codebase_snippets:
            messages.append({"role": "system", "content": snippet['code']})

        messages.append({"role": "system", "content": f"The following is the current conversation:"})

        temp_messages = [{"role": "user", "content": message}]
        for msg in reversed(self.message_memory):
            if num_tokens_from_messages(messages + temp_messages + [msg], self.gpt_model) <= (8192 - max_tokens if self.gpt_model == 'gpt-4' else 4096 - max_tokens):
                temp_messages.append(msg)
            else:
                break

        messages.extend(reversed(temp_messages))

        print(messages)

        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=messages,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens
        )

        self.message_memory.append({"role": "user", "content": message})
        self.message_memory.append({"role": "assistant", "content": response.choices[0].message.content})

        return response.choices[0].message.content

    def run(self):
        while True:
            message = input("You: ")
            response = self.send_message(message)
            print(f"Chatbot: {response}")
