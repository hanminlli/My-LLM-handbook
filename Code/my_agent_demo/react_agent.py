# my_agent_demo/react_agent.py

from langchain_core.tools import Tool
from langchain_classic.memory import ConversationBufferMemory
from langchain.agents.factory import create_agent
from langchain_ollama import ChatOllama  
from langchain_core.runnables import RunnableLambda


# tools
def search_local(query: str) -> str:
    """Simple local fact lookup."""
    data = {
        "Japan population": "125.7 million",
        "Japan GDP": "4.2 trillion USD"
    }
    return data.get(query, "No result found.")


def calculator(expr: str) -> str:
    """Evaluate arithmetic expressions safely."""
    try:
        return str(eval(expr))
    except Exception as e:
        return f"Error: {e}"


# build agent
def build_agent():
    llm = ChatOllama(model="mistral", temperature=0).bind(stream=False)

    tools = [
        Tool(name="Search", func=search_local, description="Lookup simple facts."),
        Tool(name="Calculator", func=calculator, description="Evaluate math expressions."),
    ]

    # create_agent only takes (model, tools)
    agent_graph = create_agent(llm, tools) # does not automatically handle conversation history

    # Conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def with_memory(inputs):
        past = memory.load_memory_variables({}) # {"chat_history": [...list of all prior messages...]}
        result = agent_graph.invoke({"input": inputs["input"], **past}) # {"input": question, "chat_history": [...]}
        memory.save_context({"input": inputs["input"]}, {"output": result["output"]}) # save the new turn 
        return result

    runnable_agent = RunnableLambda(with_memory)
    # wraps with_memory function into a LangChain Runnable
    # so it can be used like a normal LC pipeline component
    return runnable_agent, llm


# reflection
def reflect_on_answer(llm, question: str, answer: str) -> str:
    """Ask the model to critique its own reasoning."""
    reflection_prompt = f"""
    You are a reflective AI agent.
    Question: {question}
    Answer: {answer}
    Critique the reasoning and suggest correction if needed.
    """
    return llm.invoke(reflection_prompt).strip()


# driver
def react_with_reflection(question: str):
    agent, llm = build_agent()
    print(f"\n[Question]\n{question}\n")

    result = agent.invoke({"input": question})
    answer = result.get("output", str(result))
    print(f"[Answer]\n{answer}\n")

    reflection = reflect_on_answer(llm, question, answer)
    print(f"[Reflection]\n{reflection}\n")

    if "improve" in reflection.lower() or "incorrect" in reflection.lower():
        improved = agent.invoke({"input": f"Revisit the question considering: {reflection}"})
        print(f"[Improved Answer]\n{improved.get('output', str(improved))}\n")


if __name__ == "__main__":
    # Warm up Ollama model
    ChatOllama(model="mistral").bind(stream=False).invoke("ping")

    react_with_reflection(
        "What is Japan's GDP per capita if GDP=4.2 trillion USD and population=125.7 million?"
    )
