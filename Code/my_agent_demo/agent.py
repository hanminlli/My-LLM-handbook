# my_agent_demo/agent.py

import asyncio
import json
import ollama

# When our program does things like: Waiting for data over the network, 
# Reading or writing files, Running many tasks ``at the same tim'' (like tool calls, LLM requests),
# the CPU often sits idle while waiting. 
# In normal python, that is blocking, each line waits for the previous to finish, which is very 
# inefficient when we have lots of waiting. 
# asyncio is the python built-in framework for asynchronous programming, 
# allowing our program to handle multiple waiting tasks without freezing.

# Spefically,``async'' and ``await'' are 
# Python keywords that define asynchronous functions and control when to pause or resume them.

# Example:

# def greet():
#     return "Hello"

# async def greet():
#     return "Hello"

# Compare to the first function, greet() doesn't return ``Hello'' immediately
# it returns a coroutine, a kind of ``promis'' that will produce a result later.
# To actually run it, you must await it:

# message = await greet()
# print(message)

# Notice that we can only use await inside another async function.

# async def main():
#     msg = await greet()
#     print(msg)
# asyncio.run(main())

# An MCP agent or LLM client often performs multiple asynchronous tasks:
# - Sending a request to the model API
# - Waiting for the response
# - Calling a tool on a separate MCP server
# - Receiving tool results
# Each of these is I/O bound — we're mostly waiting on the network, not doing heavy computation.
# Using async allows these to run concurrently in a single thread:


def ask_local_llm(prompt: str, model="mistral") -> str:
    """Call local Ollama model and return plain text."""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


async def send_request(proc, method: str, params=None, req_id: int = 1):
    """Send a JSON-RPC 2.0 request to the subprocess."""
    message = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params or {},
    }
    raw = json.dumps(message) + "\n" # converts the Python dict to JSON-encoded string:
    # append a newline so that the receiver knows where the JSON message ends
    proc.stdin.write(raw.encode()) # StreamWriter connected to the tool server's stdin
    # encode() converts the string to UTF-8
    await proc.stdin.drain() # .drain() tells asyncio: wait until the write buffer is flushed.
    # This ensures the subprocess actually receives the message before we continue.
    # Because .drain() can involve waiting for OS-level I/O, 
    # we await it so the event loop can yield to other tasks in the meantime.


async def read_response(proc):
    """Read one JSON-RPC response line from the subprocess."""
    line = await proc.stdout.readline()
    if not line:
        raise EOFError("Subprocess closed connection")
    return json.loads(line.decode())


async def main():
    # Launch FastMCP server as a separate background process
    # then connects to it via standard input/output (stdin/stdout) pipes.
    # We do not launch in another terminal, otherwise they woule become 
    # two processes that do not share stdin/stdout 
    proc = await asyncio.create_subprocess_exec(
        "python", "tools_server.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )

    # Initialize the session (required by MCP)
    init_params = {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "custom_agent", "version": "0.1"},
    }
    await send_request(proc, "initialize", init_params, req_id=0)
    _ = await read_response(proc)  # ignore result

    # List tools using correct method name
    await send_request(proc, "tools/list", {}, req_id=1) 
    # {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    response = await read_response(proc) 
    tools = response.get("result", {}).get("tools", [])
    print("Available tools:", [t.get("name") for t in tools])

    # Simple REPL loop
    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break

        prompt = (
            f"You are an AI agent with access to these tools: {tools}.\n"
            "When needed, decide which tool to use and produce JSON:\n"
            '{"tool": "<name>", "args": {...}}.\n'
            "If no tool is needed, respond with:\n"
            '{"tool": null, "message": "<reply>"}.\n'
            f"User request: {user_input}"
        )

        # Ask local LLM to decide on a tool or response
        content = await asyncio.to_thread(ask_local_llm, prompt)
        try:
            decision = json.loads(content)
        except json.JSONDecodeError:
            print("LLM output (non-JSON):", content)
            continue

        if not decision.get("tool"):
            print("Agent:", decision.get("message", "(no message)"))
            continue

        # invoking a tool
        call_params = {
            "name": decision["tool"],
            "arguments": decision["args"],
        }
        await send_request(proc, "tools/call", call_params, req_id=2)
        tool_resp = await read_response(proc)
        result = tool_resp.get("result")
        print(f"[Tool Result] {result}")

        # Generate the final agent reply based on tool result
        followup = (
            f"User request: {user_input}\n"
            f"Tool result: {result}\n"
            "Compose the final concise reply."
        )
        final_reply = await asyncio.to_thread(ask_local_llm, followup)
        print("Agent:", final_reply)

    proc.terminate()
    await proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
