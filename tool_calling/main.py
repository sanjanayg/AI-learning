from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db import get_db, get_database_schema
from query_service import query_result, multiple_tool_query
from contextlib import asynccontextmanager
from mcp_client import MCPClientManager
from llm_service import run_agent
from schema_indexer import index_schema
from schema_retriever import retrieve_relevant_schema

class AskRequest(BaseModel):
    question: str

mcp_manager = MCPClientManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await mcp_manager.connect_all()
    index_schema()
    yield
    await mcp_manager.close()

app = FastAPI(title="FastAPI LLM SQL Tool PostgreSQL POC", lifespan=lifespan)

@app.post("/mcp-questions")
async def ask(question: str):
    schema = retrieve_relevant_schema(question)
    answer = await run_agent(question, mcp_manager, schema=schema)
    return {"answer": answer}
