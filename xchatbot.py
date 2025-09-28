from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')


chat_history = [
    SystemMessage(content="You are a helpful and friendly AI assistant.")
]

print("AI: Hello! How can I help you today?")

while True:
    user_input = input('you: ')
    
    if user_input.lower() == 'exit':
        break
    
    chat_history.append(HumanMessage(content=user_input))

    res = model.invoke(chat_history)
    
    chat_history.append(res)
    
    print("AI: ", res.content)

print("\n--- Conversation History (LangChain Message Objects) ---")
print(chat_history)