from fastapi import FastAPI, HTTPException,Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db import get_db
from query_service import query_result,multiple_tool_query

app = FastAPI(title="FastAPI LLM SQL Tool PostgreSQL POC")


class AskRequest(BaseModel):
    question: str


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
