import streamlit as st
import httpx

# FastAPI backend base URL
BACKEND_URL = "http://localhost:8000"

# Set page configuration with a premium look
st.set_page_config(
    page_title="RAG Portal",
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


def create_new_chat() -> str | None:
    """
    Calls POST /chats to create a server-generated chat_id.
    Returns the new chat_id string, or None on failure.
    """
    try:
        r = httpx.post(f"{BACKEND_URL}/chats", timeout=5.0)
        if r.status_code == 201:
            return r.json()["chat_id"]
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("###  Chat Sessions")

    # Fetch sessions from DB
    sessions = fetch_chat_sessions()
    chat_ids = [s["chat_id"] for s in sessions]

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("+ New", use_container_width=True, help="Create a new isolated chat session"):
            with st.spinner("Creating session..."):
                new_id = create_new_chat()
                if new_id:
                    # Clear TTL cache so the selectbox picks up the new entry immediately
                    fetch_chat_sessions.clear()
                    st.session_state.active_chat_id = new_id
                    st.rerun()
                else:
                    st.error("Could not create session — is the backend running?")

    with col1:
        if not chat_ids:
            st.info("No sessions yet. Click + New to start.")
            # Still allow a manual fallback if backend is down
            chat_id = st.session_state.active_chat_id or ""
        else:
            # Determine the index for the currently active chat
            current = st.session_state.active_chat_id
            default_index = chat_ids.index(current) if current in chat_ids else 0

            selected = st.selectbox(
                "Active Session",
                options=chat_ids,
                index=default_index,
                format_func=lambda cid: f"…{cid[-8:]}" if len(cid) > 12 else cid,
                help="Select an existing session. Chat history and files are restored from the database.",
                label_visibility="collapsed",
            )
            chat_id = selected
            st.session_state.active_chat_id = chat_id

    # On chat_id change: load history + files from DB if not already done this session
    if chat_id and chat_id not in st.session_state.db_loaded:
        # Load messages from DB
        db_messages = fetch_messages_for_chat(chat_id)
        st.session_state.messages[chat_id] = [
            {
                "role": m["role"],
                "content": m["content"],
                "citations": m.get("citations", []),
            }
            for m in db_messages
        ]
        # Load files from DB
        st.session_state.sidebar_files[chat_id] = fetch_files_for_chat(chat_id)
        st.session_state.db_loaded.add(chat_id)

    # Ensure message list exists even if DB returned nothing
    if chat_id and chat_id not in st.session_state.messages:
        st.session_state.messages[chat_id] = []

    st.markdown("---")
    st.markdown("### 📂 Ingest Document")
    st.markdown("Upload files to this session's isolated workspace.")

    uploaded_file = st.file_uploader(
        "Select File",
        type=["pdf", "docx", "png", "jpg", "jpeg", "webp"],
        help="Supports digital/scanned PDFs, Word docs, and raw images.",
    )

    if uploaded_file is not None:
        if st.button("⚡ Index Document", use_container_width=True):
            with st.spinner("Parsing layout, generating embeddings, and indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = httpx.post(
                        f"{BACKEND_URL}/chat/{chat_id}/upload",
                        files=files,
                        timeout=120.0,
                    )
                    if response.status_code == 200:
                        res_json = response.json()
                        st.success(
                            f"✅ Indexed '{uploaded_file.name}'! "
                            f"({res_json['total_chunks']} chunks stored)"
                        )
                        # Invalidate file cache so sidebar refreshes
                        st.session_state.sidebar_files.pop(chat_id, None)
                        st.session_state.db_loaded.discard(chat_id)
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI backend: {str(e)}")

    st.markdown("---")

    # ── Workspace Explorer (DB-backed) ──────────────────────────────────────
    st.markdown("### 🗂️ Isolated Workspace Explorer")
    if chat_id:
        st.markdown(
            f"Files indexed under <span style='color:#00d2ff; font-weight:bold;'>"
            f"…{chat_id[-8:]}</span>:",
            unsafe_allow_html=True,
        )
        # Use DB-tracked files as source of truth
        files_list = st.session_state.sidebar_files.get(chat_id, [])
        if files_list:
            for f in files_list:
                chunk_label = f"({f['chunk_count']} chunks)" if "chunk_count" in f else ""
                st.markdown(
                    f"<div class='file-badge'>📄 {f['file_name']} {chunk_label}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No files uploaded for this session yet.")
    else:
        st.info("Create or select a chat session to see its files.")


# ── Main Chat Area ────────────────────────────────────────────────────────────

st.markdown(f"<h1 class='main-title'>RAG Portal</h1>", unsafe_allow_html=True)
if chat_id:
    st.markdown(
        f"<div class='subtitle'>Retrieval-Augmented Generation | "
        f"Session: <b>…{chat_id[-8:]}</b></div>",
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
    with st.expander("View Grounded Citations & Provenance"):
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

# Chat input — only available when a session is selected
if chat_id:
    if prompt := st.chat_input("Ask a question based on your indexed documents..."):
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
                        json={"query": prompt},
                        timeout=60.0,
                    )

                    if response.status_code == 200:
                        res_json = response.json()
                        answer = res_json["answer"]
                        citations = res_json.get("citations", [])

                        response_container.markdown(answer)
                        render_citations(citations)

                        # Persist assistant message + citations to DB
                        persist_message(chat_id, role="assistant", content=answer, citations=citations)

                        st.session_state.messages[chat_id].append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                        })

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
                            "content": warning_html,
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
    st.info("👈 Create or select a chat session from the sidebar to start chatting.")
