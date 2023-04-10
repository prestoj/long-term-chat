# Long-Term Chat

Long-Term Chat is a chatbot project that enhances ChatGPT with long-term memory capabilities, making conversations more fun and personal. The project runs on OpenAI's GPT-3.5-turbo by default, but this can be configured in the .env file.

## Demo

https://user-images.githubusercontent.com/23005801/230815912-99fb3fac-01f2-40fa-ad99-d4960efba803.mp4


## Getting Started

1. Clone the repository:

`git clone https://github.com/prestoj/long-term-chat.git
cd long-term-chat`


2. Install the required dependencies:

`pip install -r requirements.txt`


3. Create a `.env` file in the project's root directory by copying the `.env.template` file:

`cp .env.template .env`


4. Edit the `.env` file and fill in the following information:


```
OPENAI_API_KEY=your-openai-api-key

PINECONE_API_KEY=your-pinecone-api-key

PINECONE_ENVIRONMENT=your-pinecone-environment

PINECONE_TABLE_NAME=your-table-name

GPT_MODEL=gpt-3.5-turbo
```


Replace the placeholders with your actual API keys and other required information. You can use gpt-4 if you want as well.

5. Run the `main.py` file:

`python main.py`

By default, the project runs on GPT-3.5-turbo. You can configure the model in the `.env` file.

Enjoy more engaging and context-aware conversations with Long-Term Chat!
