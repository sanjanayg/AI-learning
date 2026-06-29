# eval/ragas_eval.py
import sys
from types import ModuleType
import json
import os
# Mock the deleted langchain module to stop Ragas from crashing on import
mock_vertex = ModuleType("langchain_community.chat_models.vertexai")
mock_vertex.ChatVertexAI = object  
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertex
from openai import AsyncOpenAI
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from db.database import AsyncSessionLocal
from fastapi import Depends
from db.database import get_db
from db import crud
from config import settings
import httpx
from ragas.run_config import RunConfig

async def build_eval_dataset(chat_id: str, ground_truths: dict[str, str] | None = None, db=None):
    """
    Pulls (question, answer, contexts) triples from Postgres for a given chat_id.
    ground_truths: optional {question: reference_answer} map for context_recall/precision
    """
    if db is None:
        async with AsyncSessionLocal() as db:
            return await build_eval_dataset(chat_id, ground_truths, db)
            
    rows = await crud.list_messages(db, chat_id)    
    messages = rows
    print("Fetched messages count:", len(messages))

    samples = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    # pair each assistant message with the preceding user message
    for i in range(1, len(messages)):
        prev, curr = messages[i - 1], messages[i]
        if prev.role == "user" and curr.role == "assistant":
            citations = curr.citations or []
            contexts = []
            
            # 1. Handle JSON string from DB
            if isinstance(citations, str):
                try:
                    citations = json.loads(citations)
                except json.JSONDecodeError:
                    citations = []

            if isinstance(citations, dict):
                # Handle dictionary payload if wrapped
                context_list = citations.get("retrieved_contexts") or citations.get("contexts") or [citations]
                contexts = [
                    str(c.get("content")) if isinstance(c, dict) else str(c) 
                    for c in context_list if c is not None
                ]
            elif isinstance(citations, list):
                # This matches your DB: A flat list of dicts with a "content" key
                contexts = [
                    str(c.get("content")) if isinstance(c, dict) else str(c) 
                    for c in citations if c is not None
                ]
            elif hasattr(citations, "retrieved_contexts"):
                contexts = [str(c.content) for c in citations.retrieved_contexts if hasattr(c, "content")]

            # 3. Clean and drop empty or 'None' strings
            contexts = [c.strip() for c in contexts if c and str(c).strip() != "None" and str(c).strip()]

            # 4. Fallback safeguard for non-RAG chat entries
            if not contexts:
                contexts = ["No background context retrieved for this response."]

            # 3. Clean and drop empty or 'None' strings
            contexts = [c.strip() for c in contexts if c and str(c).strip() != "None" and str(c).strip()]

            # 4. Fallback instead of skipping! Ragas requires contexts to be a list of strings.
            # If no context was retrieved (e.g. chit-chat or fallback), we pass a placeholder string.
            if not contexts:
                contexts = ["No background context retrieved for this response."]

            samples["question"].append(prev.content)
            samples["answer"].append(curr.content)
            samples["contexts"].append(contexts)
            samples["ground_truth"].append(
                (ground_truths or {}).get(prev.content, "")
            )
            
    print(f"Built {len(samples['question'])} eval samples for chat_id={chat_id}")
    return Dataset.from_dict(samples)


async def run_ragas_eval(chat_id: str, ground_truths: dict[str, str] | None = None):
    dataset = await build_eval_dataset(chat_id, ground_truths)
    print("Dataset built successfully:", dataset)
    
    # Check if dataset contains rows before sending to Ragas
    if len(dataset) == 0:
        print("Error: Dataset has 0 samples. Cannot run evaluation.")
        return None
    groq_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.GROQ_API_KEY,
        http_client=httpx.AsyncClient(verify=False)
    )
    # Use the 'groq/' prefix to tell Ragas to route requests via Groq
    eval_llm = llm_factory(model="meta-llama/llama-4-scout-17b-16e-instruct",client=groq_client,max_tokens=4096,)
    
    # Modern local HuggingFace embedding integration natively supported by Ragas
    eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    metrics = [faithfulness, answer_relevancy]
    if any(dataset["ground_truth"]):
        metrics += [context_precision, context_recall]

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=True,
        run_config=RunConfig(max_workers=1, timeout=120)
    )
    return result.to_pandas()


if __name__ == "__main__":
    df = asyncio.run(run_ragas_eval(chat_id="fa1370c1-4f80-49c2-941e-8df4b15a11e4"))
    if df is not None:
        summary = df[["user_input", "response", "faithfulness", "answer_relevancy"]]
        print("the df is ",summary.to_markdown())
        summary.to_csv("eval_summary.csv", index=False)
