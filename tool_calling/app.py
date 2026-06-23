import chainlit as cl
import httpx

BACKEND_URL = "http://localhost:8000"

@cl.on_message
async def on_message(message: cl.Message):
    async with cl.Step(name="Thinking...") as step:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                # MUST use 'params=' here to append it to the URL query string
                response = await client.post(
                    f"{BACKEND_URL}/mcp-questions",
                    params={"question": message.content}
                )
                response.raise_for_status()
                answer = response.json().get("answer", "No answer returned.")
        except Exception as e:
            answer = f"Error: {str(e)}"
        
        step.output = answer

    await cl.Message(content=answer).send()
