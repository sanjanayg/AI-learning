import streamlit as st
import httpx

# FastAPI backend base URL
BACKEND_URL = "http://localhost:8000"
# Set page configuration with a premium look
st.set_page_config(
    page_title="RAG Portal",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background-color: #11151c;
        }

        /* Premium Card styling */
        .workspace-card {
            background-color: #1e2530;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #00d2ff;
            margin-bottom: 10px;
        }

        .file-badge {
            display: inline-block;
            background-color: #2b3547;
            color: #00d2ff;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin: 4px;
            border: 1px solid #3c4b63;
        }

        /* Citation block styling */
        .citation-card {
            background-color: #1b222c;
            border: 1px solid #2a3545;
            border-radius: 6px;
            padding: 12px;
            margin-top: 8px;
            margin-bottom: 8px;
        }

        .citation-header {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #8899a6;
            border-bottom: 1px solid #2a3545;
            padding-bottom: 4px;
            margin-bottom: 8px;
        }

        .citation-content {
            font-size: 13px;
            color: #e1e8ed;
            font-family: monospace;
            white-space: pre-wrap;
        }

        /* Title style */
        .main-title {
            font-family: 'Outfit', 'Inter', sans-serif;
            background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            color: #8899a6;
            font-size: 14px;
            margin-bottom: 25px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def fetch_chat_sessions() -> list[dict]:
    """
    Fetches all registered chat sessions from the DB via GET /chats.
    Cached with a 5-second TTL so the sidebar doesn't hammer the API
    on every widget interaction, but still stays nearly real-time.
    Returns an empty list on backend error (graceful degradation).
    """
    try:
        r = httpx.get(f"{BACKEND_URL}/chats", timeout=5.0)
        if r.status_code == 200:
            return r.json()  # list of {chat_id, created_at, last_active_at}
    except Exception:
        pass
    return []


def create_new_chat(chat_name: str = "") -> dict | None:
    """
    Calls POST /chats to create a new session.
    Returns {chat_id, chat_name} or None on failure.
    """
    try:
        r = httpx.post(
            f"{BACKEND_URL}/chats",
            json={"chat_name": chat_name or None},
            timeout=5.0
        )
        if r.status_code == 201:
            return r.json()  # {chat_id, chat_name}
        if r.status_code == 409:
            return {"error": r.json().get("detail", "Name already exists.")}
    except Exception:
        pass
    return None


def fetch_files_for_chat(chat_id: str) -> list[dict]:
    """Fetches files indexed under the given chat_id from PostgreSQL."""
    try:
        r = httpx.get(f"{BACKEND_URL}/chats/{chat_id}/files", timeout=5.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def fetch_messages_for_chat(chat_id: str) -> list[dict]:
    """Fetches full chat history for the given chat_id from PostgreSQL."""
    try:
        r = httpx.get(f"{BACKEND_URL}/chats/{chat_id}/messages", timeout=5.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []


def persist_message(chat_id: str, role: str, content: str, citations: list | None = None):
    """POSTs a single message to the DB. Silently ignores failures."""
    try:
        httpx.post(
            f"{BACKEND_URL}/chats/{chat_id}/messages",
            json={"role": role, "content": content, "citations": citations or []},
            timeout=10.0,
        )
    except Exception:
        pass


# ── Session state initialisation ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = {}       # {chat_id: [message dicts]}

if "db_loaded" not in st.session_state:
    st.session_state.db_loaded = set()   # chat_ids whose history we've already fetched this session

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if "sidebar_files" not in st.session_state:
    st.session_state.sidebar_files = {}  # {chat_id: [file dicts]}

if "intelligence_level" not in st.session_state:
    st.session_state.intelligence_level = "auto"

# Placeholder used to update sidebar token count immediately after each response
token_placeholder = None
current_total_tokens = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    sessions = fetch_chat_sessions()
    chat_id = None
    selected_chat = None

    # New chat button
    if st.button("+ New Chat", use_container_width=True):
        st.session_state["show_new_chat_input"] = True

    # New chat form
    if st.session_state.get("show_new_chat_input"):
        new_name = st.text_input(
            "Chat name",
            placeholder="e.g. invoice test",
            key="new_chat_name_input",
        )

        confirm_col, cancel_col = st.columns([1, 1])

        with confirm_col:
            if st.button("Create", use_container_width=True):
                with st.spinner("Creating session..."):
                    result = create_new_chat(new_name.strip())

                if result is None:
                    st.error("Could not create session")
                elif "error" in result:
                    st.error(result["error"])
                else:
                    fetch_chat_sessions.clear()
                    st.session_state.active_chat_id = result["chat_id"]
                    st.session_state["show_new_chat_input"] = False
                    st.rerun()

        with cancel_col:
            if st.button("Cancel", use_container_width=True):
                st.session_state["show_new_chat_input"] = False
                st.rerun()

    st.markdown("---")

    # Recent chats box
    st.markdown("#### Chats")

    with st.container(height=220):
        if not sessions:
            st.info("No chats yet.")
        else:
            if st.session_state.active_chat_id is None:
                st.session_state.active_chat_id = sessions[0]["chat_id"]

            for chat in sessions:
                is_active = chat["chat_id"] == st.session_state.active_chat_id

                if st.button(
                    chat["chat_name"],
                    key=f"chat_{chat['chat_id']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.active_chat_id = chat["chat_id"]
                    st.rerun()

    chat_id = st.session_state.active_chat_id

    selected_chat = next(
        (s for s in sessions if s["chat_id"] == chat_id),
        None,
    )

    if selected_chat:
        current_total_tokens = selected_chat.get("total_tokens_used", 0)
        token_placeholder = st.empty()
        token_placeholder.markdown(
            f"<div style='font-size:12px; color:#8899a6; margin-top:8px;'>"
            f"Total tokens used: <b style='color:#00d2ff;'>{current_total_tokens:,}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Load messages and files
    if chat_id and chat_id not in st.session_state.db_loaded:
        db_messages = fetch_messages_for_chat(chat_id)

        st.session_state.messages[chat_id] = [
            {
                "role": m["role"],
                "content": m["content"],
                "citations": m.get("citations", []),
                "intelligence": m.get("intelligence"),
                "model_used": m.get("model_used"),
                "tokens_used": m.get("tokens_used"),
            }
            for m in db_messages
        ]

        st.session_state.sidebar_files[chat_id] = fetch_files_for_chat(chat_id)
        st.session_state.db_loaded.add(chat_id)

    if chat_id and chat_id not in st.session_state.messages:
        st.session_state.messages[chat_id] = []

    st.markdown("---")

    # Document upload
    st.markdown("### Document Upload")
    st.markdown("Upload files to this session's isolated workspace.")

    uploaded_files = st.file_uploader(
        "Select Files",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files and chat_id:
        if st.button("Index Documents", use_container_width=True):
            with st.spinner("Parsing layout, generating embeddings, and indexing..."):
                try:
                    files_payload = []

                    for uploaded_file in uploaded_files:
                        files_payload.append(
                            (
                                "files",
                                (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    uploaded_file.type,
                                ),
                            )
                        )

                    response = httpx.post(
                        f"{BACKEND_URL}/chat/{chat_id}/upload",
                        files=files_payload,
                        timeout=120.0,
                    )

                    if response.status_code == 200:
                        res_json = response.json()

                        for file_result in res_json["files"]:
                            if file_result["success"]:
                                st.success(
                                    f"Indexing Completed '{file_result['file_name']}'! "
                                    f"({file_result['total_chunks']} chunks stored)"
                                )
                            else:
                                st.error(
                                    f"{file_result['file_name']}: {file_result['error']}"
                                )

                        st.session_state.sidebar_files.pop(chat_id, None)
                        st.session_state.db_loaded.discard(chat_id)
                        st.rerun()
                    else:
                        try:
                            error_message = response.json().get("detail", "Unknown error")
                        except Exception:
                            error_message = response.text

                        st.error(error_message)

                except Exception as e:
                    st.error(f"Could not connect to FastAPI backend: {str(e)}")

    st.markdown("---")

    # Intelligence selector
    st.selectbox(
        "Intelligence",
        options=["auto", "instant", "medium", "high"],
        format_func=lambda x: x.capitalize(),
        key="intelligence_level",
    )

    st.markdown("---")

    # Uploaded files scroll box
    st.markdown("### Workspace Explorer")
    st.markdown("#### Uploaded Files")

    with st.container(height=180):
        if chat_id:
            files_list = st.session_state.sidebar_files.get(chat_id, [])

            if files_list:
                for f in files_list:
                    st.markdown(
                        f"<div class='file-badge'>{f['file_name']}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No files uploaded.")
        else:
            st.info("Create or select a chat session.")


# ── Main Chat Area ────────────────────────────────────────────────────────────

st.markdown(f"<h1 class='main-title'>RAG Portal</h1>", unsafe_allow_html=True)
if chat_id:
    st.markdown(
        f"<div class='subtitle'>Hi Sanjana! Welcome to the RAG Portal.",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='subtitle'>Create or select a chat session from the sidebar to begin.</div>",
        unsafe_allow_html=True,
    )


def render_citations(citations: list):
    """Renders a list of citation dicts inside an expander."""
    if not citations:
        return
    with st.expander("Chunks"):
        for idx, cite in enumerate(citations, start=1):
            element_color = "#00c6ff" if "table" in cite.get("element_type", "") else "#00ff87"
            st.markdown(
                f"""
                <div class='citation-card'>
                    <div class='citation-header'>
                        <span><b>Source #{idx}:</b> {cite.get('file_name','?')} (Page {cite.get('page_number','?')})</span>
                        <span style='color:{element_color}; font-weight:bold;'>{cite.get('element_type','').upper()}</span>
                    </div>
                    <div class='citation-content'>{cite.get('content','')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Display chat history for the active session
if chat_id:
    for message in st.session_state.messages.get(chat_id, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_citations(message.get("citations", []))
            if message["role"] == "assistant" and message.get("intelligence"):
                    parts = [f"Routed to **{message['intelligence'].capitalize()}**"]
                    if message.get("model_used"):
                        parts.append(f"model: `{message['model_used']}`")
                    if message.get("tokens_used"):
                        parts.append(f"tokens: `{message['tokens_used']:,}`")
                    st.caption(" · ".join(parts))


# Chat input — only available when a session is selected
if chat_id:
    if prompt := st.chat_input("Ask a question based on your uploaded documents..."):
        # Show and store user message
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages[chat_id].append({"role": "user", "content": prompt})

        # Persist user message to DB (fire-and-forget style — no blocking wait)
        persist_message(chat_id, role="user", content=prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            response_container = st.empty()

            with st.spinner("Retrieving answer..."):
                try:
                    response = httpx.post(
                        f"{BACKEND_URL}/chat/{chat_id}/query",
                        json={
                            "query": prompt,
                            "intelligence": st.session_state.intelligence_level
                        },
                        timeout=60.0,
                    )

                    if response.status_code == 200:
                        res_json = response.json()
                        answer = res_json["answer"]
                        citations = res_json.get("citations", [])
                        intelligence = res_json.get(
                            "intelligence",
                            st.session_state.intelligence_level,
                        )
                        model_used = res_json.get("model_used", "unknown")
                        tokens_used = res_json.get("tokens_used", 0)

                        # Live-update sidebar token count immediately after response
                        if token_placeholder is not None:
                            live_total_tokens = current_total_tokens + tokens_used
                            token_placeholder.markdown(
                                f"<div style='font-size:12px; color:#8899a6; margin-top:4px;'>"
                                f"Total tokens used: <b style='color:#00d2ff;'>{live_total_tokens:,}</b>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        response_container.markdown(answer)
                        render_citations(citations)

                        st.caption(
                            f"Routed to **{intelligence.capitalize()}** · model: `{model_used}` · tokens: `{tokens_used:,}`"
                        )


                        st.session_state.messages[chat_id].append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "intelligence": intelligence,
                            "model_used": model_used,
                            "tokens_used": tokens_used,
                        })

                        # Refresh session list so sidebar token count updates immediately
                        fetch_chat_sessions.clear()

                    elif response.status_code == 400 and "security guardrail" in response.text.lower():
                        error_detail = response.json().get("detail", "Security guardrail violation.")
                        warning_html = f"""
                        <div style='background-color: #4a151b; color: #ff6b81; border: 1px solid #ff3b5c; padding: 15px; border-radius: 8px; margin-top: 10px;'>
                            <b> Security Shield Active:</b> {error_detail}
                        </div>
                        """
                        response_container.markdown(warning_html, unsafe_allow_html=True)
                        persist_message(chat_id, role="assistant", content=warning_html)
                        st.session_state.messages[chat_id].append({
                            "role": "assistant",
                            "content": error_detail,
                        })

                    else:
                        error_msg = f"Error {response.status_code}: {response.text}"
                        response_container.error(error_msg)
                        st.session_state.messages[chat_id].append({
                            "role": "assistant",
                            "content": error_msg,
                        })

                except Exception as e:
                    connection_error = f"Could not connect to FastAPI backend: {str(e)}"
                    response_container.error(connection_error)
                    st.session_state.messages[chat_id].append({
                        "role": "assistant",
                        "content": connection_error,
                    })
else:
    st.info(" Create or select a chat session from the sidebar to start chatting.")
