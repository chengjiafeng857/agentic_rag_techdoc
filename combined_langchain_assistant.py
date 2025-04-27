# Import streamlit first and set page config immediately
import streamlit as st

# Set page config as the very first Streamlit command
st.set_page_config(
    page_title="LangChain Assistant Hub", 
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import remaining libraries
import os
from dotenv import load_dotenv
import time
import warnings
from github import Github
from github.GithubException import RateLimitExceededException, BadCredentialsException
from PIL import Image
import json

# Import langchain components
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import tool

# Import supabase db
from supabase.client import Client, create_client

# Custom styling for the app (after page config)
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
    color: #000000;
    padding-left: 10px;
    border-left: 2px solid #4CAF50;
}
.stButton button {
    background-color: #2ea44f;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()


# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize GitHub client without status checks
github_token = os.environ.get("GITHUB_TOKEN", None)
if github_token and github_token != "your_github_personal_access_token_here":
    g = Github(github_token)
else:
    g = Github()  # Use anonymous client if no valid token

# Initialize language models
rag_llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
github_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)


# Initialize embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
)

# Pull RAG prompt from hub
rag_prompt = hub.pull("jackfengrag/myrag")

# Custom callback handler to capture retrieved documents
class DocumentCaptureHandler:
    def __init__(self):
        self.captured_docs = []
    
    def capture_docs(self, docs):
        self.captured_docs.extend(docs)

document_handler = DocumentCaptureHandler()

# Initialize session state for number of documents to retrieve
if "num_docs_to_retrieve" not in st.session_state:
    st.session_state.num_docs_to_retrieve = 3

# Initialize message history session state variables
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

if "github_messages" not in st.session_state:
    st.session_state.github_messages = []
    
if "rag_sources_history" not in st.session_state:
    st.session_state.rag_sources_history = []

# Create retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    # Get the number of documents to retrieve from session state
    k = st.session_state.num_docs_to_retrieve
    retrieved_docs = vector_store.similarity_search(query, k=k)
    
    # Capture the documents for display
    document_handler.captured_docs = []
    document_handler.capture_docs(retrieved_docs)
    
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Set up RAG agent
rag_tools = [retrieve]
rag_agent = create_tool_calling_agent(rag_llm, rag_tools, rag_prompt)
rag_agent_executor = AgentExecutor(agent=rag_agent, tools=rag_tools, verbose=True)

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



# Create GitHub search tools
@tool
def search_repositories(query: str, limit: int = 5) -> str:
    """
    Search for GitHub repositories matching the given query.
    
    Args:
        query: A search string to find repositories on GitHub. Include 'langchain' to focus on LangChain resources.
        limit: Maximum number of repositories to return (default 5)
    
    Returns:
        A formatted string with repository information and links
    """
    try:
        repos = g.search_repositories(query=query, sort="stars", order="desc")
        results = []
        
        count = 0
        for repo in repos:
            if count >= limit:
                break
                
            results.append({
                "name": repo.full_name,
                "description": repo.description or "No description",
                "stars": repo.stargazers_count,
                "url": repo.html_url,
                "language": repo.language or "Not specified"
            })
            count += 1
            
        if not results:
            return "No repositories found matching the query."
            
        formatted_results = "### GitHub Repositories\n\n"
        for i, repo in enumerate(results, 1):
            formatted_results += f"{i}. **[{repo['name']}]({repo['url']})**\n"
            formatted_results += f"   - Description: {repo['description']}\n"
            formatted_results += f"   - Stars: {repo['stars']}\n"
            formatted_results += f"   - Language: {repo['language']}\n\n"
            
        return formatted_results
        
    except RateLimitExceededException:
        return "GitHub API rate limit exceeded. Please try again later or use an authenticated client."
    except Exception as e:
        return f"Error searching repositories: {str(e)}"

@tool
def search_issues(query: str, limit: int = 5) -> str:
    """
    Search for GitHub issues and pull requests matching the given query.
    
    Args:
        query: A search string to find issues on GitHub. Include 'langchain' to focus on LangChain resources.
        limit: Maximum number of issues to return (default 5)
    
    Returns:
        A formatted string with issue information and links
    """
    try:
        issues = g.search_issues(query=query, sort="updated", order="desc")
        results = []
        
        count = 0
        for issue in issues:
            if count >= limit:
                break
                
            results.append({
                "title": issue.title,
                "repository": issue.repository.full_name,
                "state": issue.state,
                "url": issue.html_url,
                "is_pr": "pull" in issue.html_url
            })
            count += 1
            
        if not results:
            return "No issues found matching the query."
            
        formatted_results = "### GitHub Issues & Pull Requests\n\n"
        for i, issue in enumerate(results, 1):
            type_label = "PR" if issue['is_pr'] else "Issue"
            state_emoji = "🟢" if issue['state'] == "open" else "🔴"
            formatted_results += f"{i}. **[{issue['title']}]({issue['url']})**\n"
            formatted_results += f"   - Repository: {issue['repository']}\n"
            formatted_results += f"   - Type: {type_label} ({state_emoji} {issue['state']})\n\n"
            
        return formatted_results
        
    except RateLimitExceededException:
        return "GitHub API rate limit exceeded. Please try again later or use an authenticated client."
    except Exception as e:
        return f"Error searching issues: {str(e)}"

@tool
def search_code(query: str, limit: int = 5) -> str:
    """
    Search for code on GitHub matching the given query.
    
    Args:
        query: A search string to find code on GitHub. Include 'langchain' to focus on LangChain resources.
        limit: Maximum number of code results to return (default 5)
    
    Returns:
        A formatted string with code information and links
    """
    try:
        code_results = g.search_code(query=query)
        results = []
        
        count = 0
        for code in code_results:
            if count >= limit:
                break
                
            results.append({
                "name": code.name,
                "path": code.path,
                "repository": code.repository.full_name,
                "url": code.html_url
            })
            count += 1
            
        if not results:
            return "No code found matching the query."
            
        formatted_results = "### GitHub Code\n\n"
        for i, code in enumerate(results, 1):
            formatted_results += f"{i}. **[{code['path']}]({code['url']})**\n"
            formatted_results += f"   - Repository: {code['repository']}\n\n"
            
        return formatted_results
        
    except RateLimitExceededException:
        return "GitHub API rate limit exceeded. Please try again later or use an authenticated client."
    except Exception as e:
        return f"Error searching code: {str(e)}"

@tool
def search_langchain_docs(query: str) -> str:
    """
    Provide links to official LangChain documentation based on the query.
    
    Args:
        query: A search term to find relevant LangChain documentation
    
    Returns:
        A formatted string with documentation links
    """
    # Map of common terms to documentation links
    doc_map = {
        "agents": "https://python.langchain.com/v0.1/docs/concepts/agents/",
        "rag": "https://python.langchain.com/v0.1/docs/concepts/rag/",
        "retrieval": "https://python.langchain.com/v0.1/docs/concepts/retrieval/",
        "retrievers": "https://python.langchain.com/v0.1/docs/concepts/retrievers/",
        "embedding": "https://python.langchain.com/v0.1/docs/concepts/embedding_models/",
        "vectorstores": "https://python.langchain.com/v0.1/docs/concepts/vectorstores/",
        "chat models": "https://python.langchain.com/v0.1/docs/concepts/chat_models/",
        "llm": "https://python.langchain.com/v0.1/docs/concepts/text_llms/",
        "prompt": "https://python.langchain.com/v0.1/docs/concepts/prompt_templates/",
        "tool": "https://python.langchain.com/v0.1/docs/concepts/tools/",
        "text splitter": "https://python.langchain.com/v0.1/docs/concepts/text_splitters/",
        "output parser": "https://python.langchain.com/v0.1/docs/concepts/output_parsers/",
        "streaming": "https://python.langchain.com/v0.1/docs/concepts/streaming/",
        "evaluation": "https://python.langchain.com/v0.1/docs/concepts/evaluation/",
    }
    
    formatted_results = "### LangChain Documentation Links\n\n"
    added = False
    
    for key, url in doc_map.items():
        if key.lower() in query.lower():
            formatted_results += f"- [{key.title()}]({url})\n"
            added = True
            
    if not added:
        # Provide general links if no specific match
        formatted_results += "No exact matches found. Here are some general LangChain resources:\n\n"
        formatted_results += "- [LangChain Python Documentation](https://python.langchain.com/v0.1/docs/)\n"
        formatted_results += "- [LangChain Concepts](https://python.langchain.com/v0.1/docs/concepts/)\n"
        formatted_results += "- [LangChain GitHub Repository](https://github.com/langchain-ai/langchain)\n"
    
    return formatted_results

# GitHub search tools
github_tools = [search_repositories, search_issues, search_code, search_langchain_docs]

# Create GitHub system prompt
github_system_prompt = """You are a helpful GitHub assistant specialized in finding LangChain resources. 
Your job is to help users find relevant repositories, issues, code examples, and documentation related to LangChain.

When a user asks a question, you should:
1. Understand what kind of GitHub resources they're looking for
2. Use the appropriate search tools to find relevant information
3. Present the results in a clear, organized way with proper Markdown formatting
4. Always provide direct links to the resources
5. If the query isn't specific to LangChain, add 'langchain' to the search query to focus results

Be concise in your responses and make sure links are properly formatted.
"""

github_prompt = ChatPromptTemplate.from_messages([
    ("system", github_system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create GitHub agent
github_agent = create_tool_calling_agent(github_llm, github_tools, github_prompt)
github_agent_executor = AgentExecutor(agent=github_agent, tools=github_tools, verbose=True)


# Streamlit UI Setup


# Display main header
st.markdown("<h1 class='main-header'>🧠 LangChain Assistant Hub</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your all-in-one platform for LangChain knowledge and resources</p>", unsafe_allow_html=True)

# Create tabs for different functionalities and track current tab
tab_names = ["📚 LangChain Knowledge Base", "🔍 GitHub Resource Finder"]
if "current_tab" not in st.session_state:
    st.session_state.current_tab = tab_names[0]  # Default to first tab

# Create the tabs without triggering reruns on tab switch
tabs = st.tabs(tab_names)

# Create a container for a unified chat input at the bottom
st.markdown("---")
chat_container = st.container()

# Track which tab was clicked last to determine where to send messages
for i, tab_name in enumerate(tab_names):
    if tabs[i].selectbox("", [tab_name], key=f"tab_select_{i}", label_visibility="collapsed") == tab_name:
        st.session_state.current_tab = tab_name

# At the bottom of the app, add a single chat input that works for both tabs
with chat_container:
    unified_query = st.chat_input("Ask me about LangChain...")
    
    if unified_query:
        # Route the message to the appropriate agent based on the active tab
        if st.session_state.current_tab == tab_names[0]:  # RAG Knowledge Base
            # Process query through RAG system
            try:
                # Reset document handler for new query
                document_handler.captured_docs = []
                
                # Add user message to RAG history
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(unified_query)
                    st.session_state.rag_messages.append(HumanMessage(unified_query))
                
                # Invoke RAG agent with error handling
                with st.chat_message("assistant", avatar="🧠"):
                    with st.spinner("Searching knowledge base..."):
                        # Update the number of documents to retrieve based on the slider
                        st.session_state.num_docs_to_retrieve = num_docs
                        
                        try:
                            result = rag_agent_executor.invoke({
                                "input": unified_query, 
                                "chat_history": st.session_state.rag_messages
                            })
                            ai_message = result["output"]
                            
                            # Store the captured documents for this response
                            st.session_state.rag_sources_history.append(document_handler.captured_docs)
                            
                            # Display AI response
                            st.markdown(ai_message)
                            st.session_state.rag_messages.append(AIMessage(ai_message))
                            
                            # Display sources if option is enabled
                            if show_sources and document_handler.captured_docs:
                                with st.expander("📚 View Source Documents", expanded=True):
                                    for i, doc in enumerate(document_handler.captured_docs):
                                        st.markdown(format_source_document(doc, i), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                            st.info("Try checking your API keys in the .env file and ensure they're valid.")
                            import traceback
                            st.code(traceback.format_exc(), language="python")
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")
                
        else:  # GitHub Resource Finder
            # Process query through GitHub agent
            try:
                # Add user message to GitHub history
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(unified_query)
                    st.session_state.github_messages.append(HumanMessage(unified_query))
                
                # Show thinking indicator and invoke GitHub agent
                with st.chat_message("assistant", avatar="🔍"):
                    with st.spinner("Searching GitHub..."):
                        try:
                            result = github_agent_executor.invoke({
                                "input": unified_query,
                                "chat_history": st.session_state.github_messages
                            })
                            response = result["output"]
                            
                            # Display response
                            st.markdown(response)
                            
                            # Add to history
                            st.session_state.github_messages.append(AIMessage(response))
                        except Exception as e:
                            st.error(f"Error generating GitHub search response: {str(e)}")
                            st.info("This might be due to GitHub API rate limits or authentication issues.")
                            import traceback
                            st.code(traceback.format_exc(), language="python")
            except Exception as e:
                st.error(f"Error processing GitHub query: {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")


# Sidebar Configuration - Keep only the About section here
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This application combines two powerful tools:
    
    **1. LangChain Knowledge Base**
    - Access information from local documentation
    - Get answers about LangChain concepts, tutorials, and usage
    - View source documents used to generate answers
    
    **2. GitHub Resource Finder**
    - Search for LangChain resources on GitHub
    - Find repositories, issues, code examples, and docs
    - Get direct links to relevant resources
    """)
    
    st.markdown("---")
    st.markdown("## Navigation")
    st.markdown("Use the tabs above to switch between the Knowledge Base and GitHub Resource Finder.")
    
    # Add any global settings here if needed in the future


# Tab 1: LangChain Knowledge Base (RAG)
with tabs[0]:
    # Set the current tab in session state
    if st.session_state.get("current_tab") != "RAG":
        st.session_state["current_tab"] = "RAG"
        st.rerun()
    
    # Create a layout with two columns - settings column and main content
    settings_col, main_col = st.columns([1, 4])
    
    # RAG Settings in the left column
    with settings_col:
        st.markdown("## RAG Settings")
        show_sources = st.checkbox("Show source documents", value=True)
        
        st.markdown("---")
        
        st.markdown("### Document Retrieval")
        num_docs = st.slider("Number of documents to retrieve", 1, 10, 3)
        
        st.markdown("---")
        st.markdown("### Tips")
        st.markdown("""
        - Ask specific questions about LangChain concepts
        - Try questions about agents, retrievers, or RAG
        - Check source documents to verify information
        """)
    
    # Main content area in the right column
    with main_col:
        # Display RAG chat messages from history on app rerun
        for i, message in enumerate(st.session_state.rag_messages):
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
                    
                    # Display sources if available and option is enabled
                    if show_sources and i//2 < len(st.session_state.rag_sources_history):
                        sources = st.session_state.rag_sources_history[i//2]
                        if sources:
                            with st.expander("📚 View Source Documents", expanded=False):
                                for j, doc in enumerate(sources):
                                    st.markdown(format_source_document(doc, j), unsafe_allow_html=True)


# Tab 2: GitHub Resource Finder
with tabs[1]:
    # Set the current tab in session state
    if st.session_state.get("current_tab") != "GitHub":
        st.session_state["current_tab"] = "GitHub"
        st.rerun()
    
    # Create columns for GitHub settings and content
    gh_settings_col, gh_main_col = st.columns([1, 4])
    
    # GitHub settings in the left column
    with gh_settings_col:
        st.markdown("## Search Settings")
        search_limit = st.slider("Results per category", 1, 10, 5)
        
        # Update the search limits
        if "search_repositories" in globals() and hasattr(search_repositories, "kwargs"):
            search_repositories.kwargs["limit"] = search_limit
        if "search_issues" in globals() and hasattr(search_issues, "kwargs"):
            search_issues.kwargs["limit"] = search_limit
        if "search_code" in globals() and hasattr(search_code, "kwargs"):
            search_code.kwargs["limit"] = search_limit
        
        st.markdown("---")
        
        st.markdown("### Search Categories")
        st.markdown("""
        This search tool can find:
        - **Repositories**: Libraries, projects, and tools
        - **Issues & PRs**: Discussions and bug reports
        - **Code**: Implementation examples and snippets
        - **Documentation**: Official LangChain docs
        """)
        
        st.markdown("---")
        
        st.markdown("### Search Tips")
        st.markdown("""
        - Include specific keywords like 'RAG' or 'agents'
        - Try searching for specific programming languages
        - Look for recent issue discussions
        - Find code examples for implementations
        """)
        
        st.markdown("---")
        
        st.markdown("### Popular Topics")
        topics = ["RAG", "Agents", "Vector stores", "Embeddings", "Retrieval", "Chain of Thought"]
        for topic in topics:
            st.button(topic, key=f"topic_{topic}", help=f"Search for {topic} resources")