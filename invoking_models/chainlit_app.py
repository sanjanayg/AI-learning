"""
chainlit_app.py -- Chainlit frontend for the RAG Portal.

Mirrors the Streamlit app.py flow without touching it.
Uses the same FastAPI backend at http://localhost:8000.

Run with:
    chainlit run chainlit_app.py

Endpoints used (backend unchanged):
    GET  /chats                         -- list all sessions
    POST /chats                         -- create new session
    GET  /chats/{chat_id}/files         -- list indexed files
    GET  /chats/{chat_id}/messages      -- full conversation history
    POST /chat/{chat_id}/upload         -- upload & index a single file (sync, returns 200)
    POST /chat/{chat_id}/query          -- RAG query, returns {answer, citations}
    POST /chats/{chat_id}/messages      -- persist a message to DB
"""

import mimetypes
import chainlit as cl
import httpx

# -- Configuration -------------------------------------------------------------

BACKEND_URL = "http://localhost:8000"

# File types accepted by the upload endpoint (server validates MIME too)
ALLOWED_EXTENSIONS = {"pdf", "docx", "png", "jpg", "jpeg", "webp"}

# Upload timeout -- endpoint is synchronous (extracts + embeds before returning)
UPLOAD_TIMEOUT_SECONDS = 120.0

# Query timeout
QUERY_TIMEOUT_SECONDS = 60.0


# -- Async API helpers ---------------------------------------------------------

async def api_get_chats() -> list[dict]:
    """
    GET /chats
    Returns all registered chat sessions ordered by most-recently-active.
    Returns [] on any error (graceful degradation).
    """
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{BACKEND_URL}/chats", timeout=5.0)
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return []


async def api_create_chat(chat_name: str = "") -> dict | None:
    """
    POST /chats
    Creates a new chat session with an optional name (auto-named if blank).
    Returns {chat_id, chat_name} on 201, None on failure.
    """
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{BACKEND_URL}/chats",
                json={"chat_name": chat_name or None},
                timeout=5.0,
            )
            if r.status_code == 201:
                return r.json()
    except Exception:
        pass
    return None


async def api_get_files(chat_id: str) -> list[dict]:
    """
    GET /chats/{chat_id}/files
    Returns all files successfully indexed under this session.
    """
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{BACKEND_URL}/chats/{chat_id}/files", timeout=5.0
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return []


async def api_get_messages(chat_id: str) -> list[dict]:
    """
    GET /chats/{chat_id}/messages
    Returns the full conversation history for a session, chronologically.
    """
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{BACKEND_URL}/chats/{chat_id}/messages", timeout=5.0
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return []


async def api_upload_file(
    chat_id: str, file_name: str, file_bytes: bytes, mime: str
) -> dict:
    """
    POST /chat/{chat_id}/upload
    Sends one file to the synchronous upload endpoint.
    Returns {success, file_name, total_chunks} on 200.
    Raises httpx.HTTPStatusError on non-200 responses.
    """
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BACKEND_URL}/chat/{chat_id}/upload",
            files={"file": (file_name, file_bytes, mime)},
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        return r.json()


async def api_query(chat_id: str, query: str) -> dict:
    """
    POST /chat/{chat_id}/query
    Sends a RAG query and returns {success, answer, citations}.
    Raises httpx.HTTPStatusError on 4xx/5xx so the caller can surface the message.
    """
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BACKEND_URL}/chat/{chat_id}/query",
            json={"query": query},
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
        return r.json()


async def api_persist_message(
    chat_id: str,
    role: str,
    content: str,
    citations: list | None = None,
) -> None:
    """
    POST /chats/{chat_id}/messages
    Fire-and-forget: stores a user or assistant message in Postgres.
    Silently swallows all errors -- persistence failure must never crash the UI.
    """
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BACKEND_URL}/chats/{chat_id}/messages",
                json={
                    "role": role,
                    "content": content,
                    "citations": citations or [],
                },
                timeout=10.0,
            )
    except Exception:
        pass


# -- Session startup -----------------------------------------------------------

@cl.on_chat_start
async def on_chat_start() -> None:
    """
    Triggered once when a user opens a new Chainlit tab/session.

    Flow:
      1. Fetch all existing chat sessions from the backend.
      2. If none exist -> auto-create one.
      3. If some exist -> present a pick-list (up to 10 sessions + "Create new").
      4. Store the chosen chat_id in cl.user_session.
      5. Replay existing message history and show indexed files.
    """
    sessions = await api_get_chats()

    if not sessions:
        # -- No sessions -- auto-create ----------------------------------------
        await cl.Message(
            content=(
                "**Welcome to RAG Portal!**\n\n"
                "No existing sessions found -- creating one for you..."
            )
        ).send()
        chat = await api_create_chat()
        if not chat:
            await cl.Message(
                content=(
                    "Could not connect to the FastAPI backend.\n"
                    "Make sure it is running at `http://localhost:8000`."
                )
            ).send()
            return
        chat_id: str = chat["chat_id"]
        chat_name: str = chat["chat_name"]

    else:
        # -- Existing sessions -- let the user pick ----------------------------
        # Build action buttons: first = create new, rest = existing sessions
        actions = [
            cl.Action(
                name="session_choice",
                value="__new__",
                label="+ Create a new session",
            )
        ]
        for s in sessions[:10]:  # cap at 10 to keep the list readable
            actions.append(
                cl.Action(
                    name="session_choice",
                    value=s["chat_id"],
                    label=s["chat_name"],
                )
            )

        choice = await cl.AskActionMessage(
            content=(
                "## Welcome to RAG Portal\n\n"
                "Pick an existing session to continue, or create a new one:"
            ),
            actions=actions,
            timeout=120,
        ).send()

        if choice is None or choice.get("value") == "__new__":
            # User chose "Create new" or the prompt timed out
            chat = await api_create_chat()
            if not chat:
                await cl.Message(
                    content="Could not create session. Is the backend running?"
                ).send()
                return
            chat_id = chat["chat_id"]
            chat_name = chat["chat_name"]
        else:
            # User picked an existing session
            chat_id = choice["value"]
            chat_name = next(
                (s["chat_name"] for s in sessions if s["chat_id"] == chat_id),
                f"Session ...{chat_id[-8:]}",
            )

    # -- Store in per-session storage ------------------------------------------
    cl.user_session.set("chat_id", chat_id)
    cl.user_session.set("chat_name", chat_name)

    # -- Replay existing message history from Postgres -------------------------
    history = await api_get_messages(chat_id)
    for past_msg in history:
        role = past_msg.get("role", "user")
        content = past_msg.get("content", "")
        author = "You" if role == "user" else "RAG Assistant"
        # Re-send historical messages so they appear in the Chainlit thread
        await cl.Message(content=content, author=author).send()

    # -- Show currently indexed files ------------------------------------------
    files = await api_get_files(chat_id)
    if files:
        file_lines = "\n".join(
            f"  - **{f['file_name']}**  ({f.get('chunk_count', '?')} chunks)"
            for f in files
        )
        await cl.Message(
            content=(
                f"**Active session:** `{chat_name}`\n\n"
                f"**Indexed files:**\n{file_lines}\n\n"
                "Upload more files or start asking questions below."
            )
        ).send()
    else:
        await cl.Message(
            content=(
                f"**Active session:** `{chat_name}`\n\n"
                "No files indexed yet. Upload a document to get started!\n\n"
                "_Supported types: PDF, DOCX, PNG, JPG, JPEG, WEBP_"
            )
        ).send()


# -- Message handler -----------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Called every time the user sends a message (text and/or files).

    Routing logic:
      - If the message contains file attachments -> upload each one.
      - If the message contains text content     -> run a RAG query.
      - Both can happen together in a single message.
    """
    chat_id: str | None = cl.user_session.get("chat_id")

    # Safety check -- should always be set after on_chat_start
    if not chat_id:
        await cl.Message(
            content="No active session found. Please refresh the page to restart."
        ).send()
        return

    # -- Handle file uploads ---------------------------------------------------
    # Chainlit exposes attached files as elements with a .path attribute
    file_elements = [el for el in (message.elements or []) if hasattr(el, "path")]
    if file_elements:
        await _handle_file_uploads(chat_id, file_elements)

    # -- Handle text query -----------------------------------------------------
    query = (message.content or "").strip()
    if query:
        await _handle_query(chat_id, query)


# -- Upload handler ------------------------------------------------------------

async def _handle_file_uploads(chat_id: str, file_elements: list) -> None:
    """
    Uploads each file element to POST /chat/{chat_id}/upload one at a time.

    The backend endpoint is synchronous (extracts -> chunks -> embeds -> indexes
    before returning), so we send files sequentially and report after each one.
    A Chainlit Step wraps each upload so the user sees progress in the thread.
    """
    for el in file_elements:
        file_name: str = el.name
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

        # Client-side extension check (server will also validate MIME)
        if ext not in ALLOWED_EXTENSIONS:
            await cl.Message(
                content=(
                    f"Skipped `{file_name}` -- unsupported file type `.{ext}`.\n"
                    f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                )
            ).send()
            continue

        # Guess MIME from extension (Chainlit may or may not set el.mime)
        mime: str = (
            getattr(el, "mime", None)
            or mimetypes.guess_type(file_name)[0]
            or "application/octet-stream"
        )

        async with cl.Step(name=f"Indexing `{file_name}`") as step:
            step.input = f"Uploading and indexing `{file_name}` ({mime})"
            try:
                # Read file bytes from the temp path Chainlit wrote for us
                with open(el.path, "rb") as fh:
                    file_bytes = fh.read()

                result = await api_upload_file(chat_id, file_name, file_bytes, mime)

                chunks = result.get("total_chunks", "?")
                step.output = f"Indexed {chunks} chunks"

                await cl.Message(
                    content=(
                        f"**`{file_name}`** indexed successfully!\n"
                        f"  - Chunks stored: **{chunks}**\n\n"
                        "You can now ask questions about this document."
                    )
                ).send()

            except httpx.HTTPStatusError as exc:
                # Backend returned a 4xx/5xx with an explanatory body
                detail = ""
                try:
                    detail = exc.response.json().get("detail", exc.response.text)
                except Exception:
                    detail = exc.response.text[:300]
                step.output = f"HTTP {exc.response.status_code}: {detail}"
                await cl.Message(
                    content=(
                        f"**Upload failed for `{file_name}`** "
                        f"(HTTP {exc.response.status_code}):\n> {detail}"
                    )
                ).send()

            except httpx.ConnectError:
                step.output = "Backend unreachable"
                await cl.Message(
                    content=(
                        "**Cannot reach the FastAPI backend.**\n"
                        "Make sure it is running at `http://localhost:8000`."
                    )
                ).send()

            except Exception as exc:
                step.output = f"Unexpected error: {exc}"
                await cl.Message(
                    content=f"Unexpected error while uploading `{file_name}`: `{exc}`"
                ).send()


# -- Query handler -------------------------------------------------------------

async def _handle_query(chat_id: str, query: str) -> None:
    """
    Sends the user's question to POST /chat/{chat_id}/query, displays the
    grounded answer, formats citations as collapsible side panels, and persists
    both the user message and the assistant response to Postgres.
    """
    # Persist the user message to DB before calling the LLM
    await api_persist_message(chat_id, role="user", content=query)

    async with cl.Step(name="Searching knowledge base...") as step:
        step.input = query
        try:
            data = await api_query(chat_id, query)
            answer: str = data.get("answer", "")
            citations: list[dict] = data.get("citations", [])
            step.output = f"Retrieved answer ({len(citations)} citation(s) found)"

        except httpx.HTTPStatusError as exc:
            # Security guardrail returns 400 with a specific message
            if (
                exc.response.status_code == 400
                and "security guardrail" in exc.response.text.lower()
            ):
                detail = exc.response.json().get("detail", "Security guardrail triggered.")
                step.output = f"Blocked: {detail}"
                await cl.Message(
                    content=f"**Security Shield Active:**\n> {detail}"
                ).send()
                await api_persist_message(
                    chat_id,
                    role="assistant",
                    content=f"Security guardrail triggered: {detail}",
                )
            else:
                detail = exc.response.text[:300]
                step.output = f"HTTP {exc.response.status_code}"
                await cl.Message(
                    content=(
                        f"**Backend error** (HTTP {exc.response.status_code}):\n"
                        f"```\n{detail}\n```"
                    )
                ).send()
            return

        except httpx.ConnectError:
            step.output = "Backend unreachable"
            await cl.Message(
                content=(
                    "**Cannot reach the FastAPI backend.**\n"
                    "Make sure it is running at `http://localhost:8000`."
                )
            ).send()
            return

        except Exception as exc:
            step.output = f"Unexpected error: {exc}"
            await cl.Message(content=f"Unexpected error: `{exc}`").send()
            return

    # -- Build citation elements (collapsible side panels) ---------------------
    citation_elements: list[cl.Text] = []
    citation_footer_lines: list[str] = []

    for idx, cite in enumerate(citations, start=1):
        file_name  = cite.get("file_name", "?")
        page_num   = cite.get("page_number", "?")
        elem_type  = cite.get("element_type", "").upper() or "TEXT"
        content    = cite.get("content", "").strip()

        # Each citation becomes a named Text element shown in a side panel
        label = f"[{idx}] {file_name} - Page {page_num} - {elem_type}"
        citation_elements.append(
            cl.Text(
                name=label,
                content=content,
                display="side",   # opens in a collapsible panel beside the answer
            )
        )
        citation_footer_lines.append(
            f"`[{idx}]` **{file_name}** -- page {page_num} ({elem_type.lower()})"
        )

    # -- Compose the final answer message --------------------------------------
    footer = ""
    if citation_footer_lines:
        footer = "\n\n---\n**Sources:**\n" + "\n".join(citation_footer_lines)

    await cl.Message(
        content=answer + footer,
        elements=citation_elements,   # side panels for each citation full text
    ).send()

    # -- Persist assistant message to Postgres ---------------------------------
    await api_persist_message(
        chat_id,
        role="assistant",
        content=answer,
        citations=citations,
    )
