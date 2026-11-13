from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

@tool
def calculator(a: float, b: float) -> str:
    """ALWAYS useful for performing basic arithmeric calculations with numbers"""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"

def main():

    model = ChatOpenAI(model="gpt-6-luna", temperature=1)
    tools = [calculator]
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("You can ask me to perform calculations or chat with me.")

    while True:

        user_input = input("You: ").strip()

        if user_input == "quit":
            break

        # Chunks are essentially parts of a response coming from our agent

        print("\nAssistance: ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    if isinstance(message.content, str):
                        print(message.content, end="")
                    elif isinstance(message.content, list):
                        for block in message.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                print(block.get("text", ""), end="")
        print()

if __name__ == "__main__":
    main()