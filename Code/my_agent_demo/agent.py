# my_agent_demo/agent.py

import asyncio
import json
import ollama


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
    raw = json.dumps(message) + "\n"
    proc.stdin.write(raw.encode())
    await proc.stdin.drain()


async def read_response(proc):
    """Read one JSON-RPC response line from the subprocess."""
    line = await proc.stdout.readline()
    if not line:
        raise EOFError("Subprocess closed connection")
    return json.loads(line.decode())


async def main():
    # Launch FastMCP server
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
