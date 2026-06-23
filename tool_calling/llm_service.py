import os
import json
from dotenv import load_dotenv
from groq import Groq
from db import get_database_schema
import json

load_dotenv()
GPT_MODEL = os.getenv("GROQ_MODEL_GPT")
LLAMA_MODEL = os.getenv("GROQ_MODEL_LLAMA")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


SYSTEM_PROMPT_MCP = """You have access to: get_weather, run_query, send_email, read_pdf, read_docx.

                        IMPORTANT RULES:
                        - Only call send_email if the user EXPLICITLY asks you to send, email, or mail
                        something. Words like "fetch", "show", "get", "list", "what is" do NOT mean
                        send an email — they mean return the answer directly in your response.
                        - Never call send_email speculatively or "just in case." If there is no explicit
                        request to send an email, do not call send_email under any circumstances.
                        - If the user asks you to fetch or query data, just return the data in your
                        answer. Do not take any additional action beyond what was asked.
                        - Only call the tools that are strictly necessary to answer the specific
                        request — do not chain extra tools the user didn't ask for.

                        SQL GENERATION RULES (for run_query):
                        - Use ONLY the exact table and column names listed in RELEVANT SCHEMA below.
                        Never guess or invent names.
                        - If a column entry includes sample_values, use those exact values in
                        comparisons (e.g., sample_values ["Y","N"] -> status = 'Y', not 'active').
                        - Respect column data types — do not use ILIKE on integer/numeric/date columns;
                        use ILIKE only for text/varchar columns.
                        - Only generate SELECT queries.
                        - If the RELEVANT SCHEMA below does not contain a table or column needed to
                        answer the question, state that clearly instead of guessing.

                        RELEVANT SCHEMA (retrieved for this query):
                        {schema}
                        SCOPE RESTRICTION:
                        - You are a specialized assistant for database queries, weather lookups,
                        email sending, and document (PDF/DOCX) reading ONLY.
                        - You must NOT answer general knowledge questions, trivia, current events,
                        opinions, coding help, or anything unrelated to the tools you have access to
                        (get_weather, run_query, send_email, read_pdf, read_docx).
                        - If the user asks something outside this scope (e.g. "who is the president
                        of India", "write me a poem", "what's 2+2"), do NOT answer it — instead
                        respond with exactly:
                        "I'm only able to help with database queries, weather, email, and document
                        reading. I can't assist with that request."
                        - This restriction applies even if you know the answer. Do not make exceptions
                        based on how simple, harmless, or factual the question seems.
                        - Do not explain why you can't answer beyond the sentence above. Do not
                        apologize excessively or offer alternatives unless asked.
                    """

async def run_agent(user_question: str, mcp_manager, schema: str = "") -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_MCP.format(schema=schema or "Schema not available.")},
        {"role": "user", "content": user_question},
    ]
    tools = mcp_manager.get_groq_tools()
    for _ in range(6):  
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1024,
        )

        choice = response.choices[0]
        message = choice.message

        if not message.tool_calls:
            return message.content  # final answer, no more tools needed

        # Append the assistant's tool-call message as-is
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })
        # Execute each requested tool call and append results
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            result = await mcp_manager.call_tool(tool_name, tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": str(result),
            })

    return "I wasn't able to complete this after several tool calls — please rephrase or simplify the request."


def question_intent(question: str) -> dict:
    prompt = f"""
                    Classify the intent of the user's question.

                    Categories:
                    - database_query
                    - database_query_with_email

                    Return only JSON in this exact format, no markdown:

                    {{
                        "intent": "database_query",
                        "needs_email": false
                    }}

                    Set "needs_email" to true only if the user explicitly asks
                    for the result to be sent/emailed somewhere.

                    Question:
                    {question}
                """

    def _call(model):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)

    try:
        result = _call(GPT_MODEL)
    except Exception as e:
        print("GPT-OSS failed or returned bad JSON:", e)
        try:
            result = _call(LLAMA_MODEL)
        except Exception as e2:
            print("Llama fallback also failed:", e2)
            # Safe default so the rest of the pipeline doesn't crash
            result = {"intent": "database_query", "needs_email": False}

    # Defensive normalization in case the model still omits the key
    result["needs_email"] = bool(
        result.get("needs_email", result.get("intent") == "database_query_with_email")
    )

    return result

def generate_sql(question: str,db) -> str:
    schema = get_database_schema(db)

    prompt = f"""
                You are a PostgreSQL SQL generator.

                Database schema:
                {schema}

                Rules:
                - Return only valid JSON.
                - Do not use markdown.
                - Only SELECT queries are allowed.
                - If the user asks for DELETE, UPDATE, INSERT, DROP, ALTER, CREATE, or TRUNCATE, return allowed=false.
                - Do not generate unsafe SQL.
                - If the column is active or status then the records will be in 'Y' or 'N'
                - Use ILIKE for text comparisons.

                JSON format:
                {{
                "allowed": true,
                "query": "SELECT ...",
                "message": "SQL generated successfully"
                }}

                For unsafe requests:
                {{
                "allowed": false,
                "query": null,
                "message": "Only SELECT queries are allowed. I cannot perform delete/update/insert operations."
                }}

                User question:
                {question}

            """

    try:

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

    except Exception as e:

        print("GPT-OSS failed:", e)

        response = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

    content = response.choices[0].message.content.strip()

    return json.loads(content)

def generate_final_answer(question: str, sql_query: str, sql_result):
    prompt = f"""
                User question:
                {question}

                Generated SQL:
                {sql_query}

                SQL result:
                {sql_result}

                Give a simple final answer.
            """


    try:

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

    except Exception as e:

        print("GPT-OSS failed for the final answer:", e)

        response = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

    content = response.choices[0].message.content.strip()

    return content
    

