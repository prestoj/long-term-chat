from datetime import datetime
import openai
from config import OPENAI_API_KEY
from memory import Memory
from openai_tools import get_embedding, num_tokens_from_messages

openai.api_key = OPENAI_API_KEY

class ChatGPT():
    def __init__(self, gpt_model, table_name):
        self.long_term_memory = Memory(table_name)
        self.short_term_memory = []
        self.gpt_model = gpt_model

    def send_message(self, message, temperature=0.7, n=1, max_tokens=500):
        self.long_term_memory.search(get_embedding(message))
        messages = [
            {"role": "system", "content": f"You are ChatGPT, a large language model trained by OpenAI. Knowledge cutoff: September 2021"},
        ]

        long_term_memory_messages = self.long_term_memory.search(get_embedding(message))

        for msg in sorted(long_term_memory_messages, key=lambda x: x["timestamp"]):
            messages.append({"role": "system", "content": f"This is a snippet from earlier on {msg['timestamp'].strftime('%B %d, %Y %I:%M:%S %p')}"})
            messages.append({"role": "user", "content": msg["message"]})
            messages.append({"role": "assistant", "content": msg["response"]})

        messages.append({"role": "system", "content": f"The following is the current conversation:"})

        temp_messages = [{"role": "system", "content": f"Current time: {datetime.now().strftime('%B %d, %Y %I:%M:%S %p')}"}, {"role": "user", "content": message}]
        for msg in reversed(self.short_term_memory):
            if num_tokens_from_messages(messages + temp_messages + [msg]) <= (8192 - max_tokens if 'gpt-4' in self.gpt_model else 4096 - max_tokens):
                temp_messages.append(msg)
            else:
                break

        messages.extend(reversed(temp_messages))

        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=messages,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens
        )

        self.short_term_memory.append({"role": "user", "content": message})
        self.short_term_memory.append({"role": "assistant", "content": response.choices[0].message.content})

        self.long_term_memory.upload_message_response_pair(message, response.choices[0].message.content)

        return response.choices[0].message.content

    def run(self):
        while True:
            message = input("You: ")
            response = self.send_message(message)
            print(f"Chatbot: {response}")
