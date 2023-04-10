from chatbot import ChatGPT

if __name__ == "__main__":
    chatgpt = ChatGPT(gpt_model="gpt-3.5-turbo", table_name="chatgpt")
    chatgpt.run()