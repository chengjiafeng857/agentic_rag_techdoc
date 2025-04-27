# import basics
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st
from PIL import Image
import json

# import langchain
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.tracers.schemas import Run
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# import supabase db
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

# initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiating embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# initiating vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)
 
# initiating llm
llm = ChatOpenAI(model="gpt-4.1",temperature=1)

# pulling prompt from hub
prompt = hub.pull("jackfengrag/myrag")


# Store for captured documents
if "retrieved_documents" not in st.session_state:
    st.session_state.retrieved_documents = {}

# Custom callback handler to capture retrieved documents
class DocumentCaptureHandler:
    def __init__(self):
        self.captured_docs = []
    
    def capture_docs(self, docs):
        self.captured_docs.extend(docs)

document_handler = DocumentCaptureHandler()

# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    
    # Capture the documents for display
    document_handler.capture_docs(retrieved_docs)
    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Function to format document for display
def format_source_document(doc, index):
    source = doc.metadata.get("source", "Unknown source")
    # Extract filename from source path
    if isinstance(source, str) and "/" in source:
        source = source.split("/")[-1]
    
    # Format source document for display with everything in black color
    return f"""
    <div style="padding: 10px; margin-bottom: 10px; border-radius: 5px; background-color: #f5f5f5; color: #000000;">
        <p><strong style="color: #000000;">Source {index+1}: {source}</strong></p>
        <p style="font-size: 0.9em; color: #000000;">{doc.page_content[:300]}...</p>
    </div>
    """

# initiating streamlit app with a new logo
st.set_page_config(
    page_title="LangChain RAG Assistant", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for the app
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-title {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .source-content {
        font-size: 0.9em;
        color: #333;
        padding-left: 10px;
        border-left: 2px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Create sidebar for settings
with st.sidebar:
    st.markdown("## Settings")
    show_sources = st.checkbox("Show source documents", value=True)
    st.markdown("---")
    st.markdown("## About")
    st.markdown("This assistant uses Agentic RAG (Retrieval-Augmented Generation) to provide information about LangChain by default, With any technical document you upload.")
    st.markdown("It retrieves relevant documents from a vector database and uses them to generate responses.")

# Display custom header with new logo
st.markdown("<h1 class='main-header'>🧠 Technical Document Knowledge Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Powered by Agentic RAG Technology</p>", unsafe_allow_html=True)

# Add a horizontal line
st.markdown("---")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# initialize sources history
if "sources_history" not in st.session_state:
    st.session_state.sources_history = []

# display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
            
            # Display sources if available and option is enabled
            if show_sources and i//2 < len(st.session_state.sources_history):
                sources = st.session_state.sources_history[i//2]
                if sources:
                    with st.expander("📚 View Source Documents", expanded=False):
                        for j, doc in enumerate(sources):
                            st.markdown(format_source_document(doc, j), unsafe_allow_html=True)


# --- Document Upload and Ingestion UI ---
st.markdown("## 📄 Upload and Ingest Documents")
uploaded_files = st.file_uploader(
    "Upload PDF, TXT, or Markdown (MD) files to ingest into the knowledge base:",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    key="file_uploader"
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_path = os.path.join("documents", file_name)
        # Save uploaded file to disk
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Load and split document
        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.lower().endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_name.lower().endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            st.warning(f"Unsupported file type: {file_name}")
            continue
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        # Ingest into vector store
        try:
            SupabaseVectorStore.from_documents(
                docs,
                embeddings,
                client=supabase,
                table_name="documents",
                query_name="rag_query",
                chunk_size=100,
            )
            st.success(f"Ingested {file_name} successfully!")
        except Exception as e:
            st.error(f"Failed to ingest {file_name}: {str(e)}")

# create the bar where we can type messages
user_question = st.chat_input("Ask me anything about LangChain...")

# did the user submit a prompt?
if user_question:
    # Reset document handler for new query
    document_handler.captured_docs = []

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

    # Show spinner while agent is generating a response
    with st.spinner("Thinking... Generating response..."):
        # invoking the agent
        result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})
        ai_message = result["output"]
    
    # Store the captured documents for this response
    st.session_state.sources_history.append(document_handler.captured_docs)

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        import re
        def render_markdown_with_codeblocks(text):
            code_block_pattern = r"```([\w\+\-]*)\n([\s\S]*?)```"
            related_code_pattern = r"<related_code>([\s\S]*?)</related_code>"
            last_end = 0
            # Find all code blocks (triple backtick and related_code) in order
            matches = []
            for m in re.finditer(code_block_pattern, text):
                matches.append((m.start(), m.end(), 'backtick', m))
            for m in re.finditer(related_code_pattern, text):
                matches.append((m.start(), m.end(), 'related_code', m))
            matches.sort()  # sort by start position
            for match in matches:
                start, end, kind, m = match
                if start > last_end:
                    st.markdown(text[last_end:start])
                if kind == 'backtick':
                    code_lang = m.group(1) or None
                    code_content = m.group(2)
                    st.code(code_content, language=code_lang)
                elif kind == 'related_code':
                    code_content = m.group(1)
                    st.code(code_content)
                last_end = end
            if last_end < len(text):
                st.markdown(text[last_end:])

        render_markdown_with_codeblocks(ai_message)
        st.session_state.messages.append(AIMessage(ai_message))
        
        # Display sources if option is enabled
        if show_sources and document_handler.captured_docs:
            with st.expander("📚 View Source Documents", expanded=True):
                for i, doc in enumerate(document_handler.captured_docs):
                    st.markdown(format_source_document(doc, i), unsafe_allow_html=True)

