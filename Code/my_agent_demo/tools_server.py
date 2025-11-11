# my_agent_demo/tools_server.py

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("text_tools")

# Register a function as a tool that an AI agent can call programmatically.
# Exposes its signature, docstring, and parameters in a structured format (JSON schema).
# Allows an AI model connected via MCP to call it dynamically by name, e.g., get_weather("Paris").
@mcp.tool() 
def summarize_text(text: str, max_length: int = 100) -> str:
    """ Return a concise summary of a paragraph (takes the first two sentences). """
    sentences = text.split(".")
    summary = ". ".join(sentences[:2])
    return summary[:max_length].rstrip() + "."

@mcp.tool()
def count_words(text: str) -> int:
    """ Count words in a text. """
    return len(text.split())


if __name__ == "__main__":
    mcp.run()