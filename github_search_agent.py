# Import required libraries
import os
import streamlit as st
from dotenv import load_dotenv
import time
import warnings
from github import Github
from github.GithubException import RateLimitExceededException

# Disable LangSmith warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
# Filter out specific LangSmith warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*API key must be provided when using hosted LangSmith API.*")

# Import langchain components
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# Initialize GitHub client - use access token if available, otherwise use anonymous client
github_token = os.environ.get("GITHUB_TOKEN", None)
g = Github(github_token) if github_token else Github()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

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

# Create search LangChain docs tool
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

# Combine all tools
tools = [search_repositories, search_issues, search_code, search_langchain_docs]

# Create system prompt
system_prompt = """You are a helpful GitHub assistant specialized in finding LangChain resources. 
Your job is to help users find relevant repositories, issues, code examples, and documentation related to LangChain.

When a user asks a question, you should:
1. Understand what kind of GitHub resources they're looking for
2. Use the appropriate search tools to find relevant information
3. Present the results in a clear, organized way with proper Markdown formatting
4. Always provide direct links to the resources
5. If the query isn't specific to LangChain, add 'langchain' to the search query to focus results

Be concise in your responses and make sure links are properly formatted.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="chat_history"),
])

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Set up Streamlit interface
st.set_page_config(
    page_title="GitHub LangChain Resource Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #171515;
    text-align: center;
    margin-bottom: 1rem;
}
.subheader {
    font-size: 1.2rem;
    color: #555;
    text-align: center;
    margin-bottom: 2rem;
}
.stButton button {
    background-color: #2ea44f;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("<h1 class='main-header'>🔍 GitHub LangChain Resource Finder</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Find repositories, code, issues, and documentation related to LangChain</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This tool helps you find LangChain resources on GitHub.
    
    It can search for:
    - Repositories
    - Issues & Pull Requests
    - Code examples
    - Documentation
    
    All results include direct links to the resources.
    """)
    
    st.markdown("---")
    
    st.markdown("## API Status")
    if github_token:
        st.success("✅ Using authenticated GitHub API")
        rate_limit = g.get_rate_limit()
        st.markdown(f"Remaining requests: {rate_limit.core.remaining}/{rate_limit.core.limit}")
    else:
        st.warning("⚠️ Using anonymous GitHub API (rate limited)")
        st.markdown("For more requests, set the GITHUB_TOKEN environment variable")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
user_query = st.chat_input("Ask about LangChain resources on GitHub...")

# Process user query
if user_query:
    # Add user message to history
    st.session_state.messages.append(HumanMessage(user_query))
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Show thinking indicator
    with st.chat_message("assistant"):
        with st.spinner("Searching GitHub..."):
            # Invoke agent
            result = agent_executor.invoke({
                "input": user_query,
                "chat_history": st.session_state.messages
            })
            response = result["output"]
            
            # Display response
            st.markdown(response)
            
            # Add to history
            st.session_state.messages.append(AIMessage(response))

# Add instructions at the bottom
st.markdown("---")
st.markdown("""
**Example queries:**
- "Find top LangChain repositories"
- "Show me example code for RAG implementation"
- "What are recent issues about agents in LangChain?"
- "Where can I find documentation about retrievers?"
""")