from fastapi import FastAPI, HTTPException,Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db import get_db
from query_service import query_result,multiple_tool_query
from contextlib import asynccontextmanager
from mcp_client import MCPClientManager
from llm_service import run_agent

app = FastAPI(title="FastAPI LLM SQL Tool PostgreSQL POC")

class AskRequest(BaseModel):
    question: str

mcp_manager = MCPClientManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await mcp_manager.connect_all()  
    yield
    await mcp_manager.close()

app = FastAPI(lifespan=lifespan)

@app.post("/mcp-questions")
async def ask(question: str):
    answer = await run_agent(question, mcp_manager)
    return {"answer": answer}

@app.post("/ask")
def ask_question(request: AskRequest, db: Session = Depends(get_db)):
    try:
        return query_result(request, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ask-email")
def ask_question(request: AskRequest, db: Session = Depends(get_db)):
    try:
        return multiple_tool_query(request, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
