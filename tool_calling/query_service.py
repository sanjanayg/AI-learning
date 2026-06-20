from fastapi import HTTPException
from llm_service import generate_sql, generate_final_answer,question_intent
from sql_tool import run_sql_tool
from email_tool import extract_email, send_email
from typing import Dict, Any

def query_result(request,db):
    try:
        sql_data = generate_sql(request.question,db)

        if not sql_data["allowed"]:
            return {
                "question": request.question,
                "answer": sql_data["message"]
            }

        sql_query = sql_data["query"]
        sql_result = run_sql_tool(sql_query,db)

        answer = generate_final_answer(
            question=request.question,
            sql_query=sql_query,
            sql_result=sql_result
        )

        return {
            "question": request.question,
            "allowed": True,
            "generated_sql": sql_query,
            "sql_result": sql_result,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def multiple_tool_query(request, db):
    try:
        intent = question_intent(request.question)

        recipient = None
        if intent["needs_email"]:
            recipient = extract_email(request.question)
            if not recipient:
                return {
                    "question": request.question,
                    "requires_clarification": True,
                    "clarification_question":
                        "Please provide the recipient email address."
                }

        sql_data = generate_sql(request.question, db)

        if not sql_data["allowed"]:
            return {
                "question": request.question,
                "answer": sql_data["message"]
            }

        sql_query = sql_data["query"]

        sql_result = run_sql_tool(sql_query, db)

        answer = generate_final_answer(
            question=request.question,
            sql_query=sql_query,
            sql_result=sql_result
        )

        email_response = None
        if intent["needs_email"]:
            email_response = send_email(
                recipient=recipient,
                subject="Database Query Result",
                body=answer
            )

        return {
            "question": request.question,
            "intent": intent,
            "generated_sql": sql_query,
            "sql_result": sql_result,
            "answer": answer,
            "email": email_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))