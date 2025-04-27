Link of this APP:
https://agenticragtechdocgit-qucoioewkq9obocvv9fzsm.streamlit.app/

# Agentic RAG and GitHub Assistant for Technical Documentation

This project implements a sophisticated Generative AI system featuring two main components integrated into a Streamlit web application:

1.  **Agentic RAG Knowledge Base:** Allows users to query a knowledge base built from technical documents (PDFs, Markdown, Jupyter Notebooks, etc.). It uses Retrieval-Augmented Generation (RAG) with LangChain agents to provide grounded answers based on the ingested content.
2.  **GitHub Resource Finder:** An agent-based tool that helps users find relevant LangChain resources (repositories, issues, code, documentation links) directly from GitHub.

## Features

* **Document Ingestion:** Load and process various document formats (PDF, TXT, MD, MDX, IPYNB) into a vector knowledge base.
* **Vector Database:** Utilizes Supabase (PostgreSQL with `pgvector`) for efficient storage and similarity search of document embeddings.
* **Agentic RAG:** Employs LangChain agents and OpenAI models (e.g., GPT-4o) to retrieve relevant document chunks and generate informative answers.
* **GitHub Search:** Integrated tools to search GitHub for repositories, issues/PRs, and code snippets related to user queries (specifically focused on LangChain).
* **LangChain Documentation Helper:** Provides quick links to relevant official LangChain documentation pages.
* **Interactive UI:** A Streamlit application provides a user-friendly chat interface for both the RAG knowledge base and the GitHub resource finder, including source document viewing and document upload capabilities.

## Architecture Overview

* **Frontend:** Streamlit (`combined_langchain_assistant.py`, `agentic_rag_streamlit.py`, `github_search_agent.py`)
* **Orchestration:** LangChain (`langchain`, `langchain-openai`, `langchain-community`)
* **LLMs & Embeddings:** OpenAI API (`gpt-4o`, `text-embedding-3-small`)
* **Vector Store:** Supabase PostgreSQL with `pgvector` extension (`supabase`, `psycopg2-binary`)
* **GitHub Integration:** `PyGithub` library
* **Data Processing:** Custom Python scripts (`ingest_in_db.py`), Jupyter Notebook (`ingest.ipynb`), LangChain text splitters.

## System Prompts

The behavior of the AI agents is guided by specific system prompts.

### RAG Knowledge Base Agent Prompt

This prompt (or similar, like `hub.pull("jackfengrag/myrag")`) guides the agent in answering questions based on retrieved documents:

```text
You are an AI assistant specializing in explaining technical documents and solving technical problems. Your primary task is to help users understand complex technical information, diagrams, and concepts from various documents, including manuals and documentation like Langchain.

Here is the technical document you need to reference:

Soucres provided

<User Query: {input}>

To assist the user effectively, follow these steps:

1. Analyze the technical document and the user's query.
2. Identify the specific section or concept the user is asking about.
3. Focus on the relevant area if the query relates to a specific section.
4. Simplify complex diagrams or concepts into manageable components.
5. For examples from Langchain documentation, explain each component: input, processing steps, and output.

When formulating your response:

1. Provide clear, detailed explanations in simple language.
2. Use relevant examples to illustrate complex ideas.
3. Use analogies to clarify difficult concepts when appropriate.
4. List steps or processes in numbered or bulleted format.
5. If the user's query is unclear, ask for clarification.

To enhance understanding:

1. Suggest related concepts or sections in the document for additional context.
2. For code examples, comment on key lines or functions.
3. Describe visual elements like diagrams in words and explain their relevance.

Before your final response, wrap your analysis inside <detailed_analysis> tags:

1. Quote relevant sections from the technical document.
2. List key technical terms or concepts.
3. Break down the user's query into key components.
4. Identify potential misunderstandings or ambiguities.
5. Outline a step-by-step plan for addressing the query.

Structure your final response as follows:

<answer>
[Summary of the user's query]

[Explanation of the concept/problem]

[Conclusion and offer for further clarification]

<related_code>
Provide a code example here
</related_code>

<additional_resources>
[Additional resources]
</additional_resources>
</answer>

Your goal is to help the user fully understand the technical concept or problem they're asking about. Be patient, thorough, and open to further clarification if needed.
```

GitHub Resource Finder Agent Prompt(Experimental)
This prompt guides the agent in using GitHub search tools:
```
You are a helpful GitHub assistant specialized in finding LangChain resources.
Your job is to help users find relevant repositories, issues, code examples, and documentation related to LangChain.

When a user asks a question, you should:
1. Understand what kind of GitHub resources they're looking for
2. Use the appropriate search tools to find relevant information
3. Present the results in a clear, organized way with proper Markdown formatting
4. Always provide direct links to the resources
5. If the query isn't specific to LangChain, add 'langchain' to the search query to focus results

Be concise in your responses and make sure links are properly formatted.
```


Setup Instructions
Prerequisites
Python 3.9+
Git
A Supabase account (for the vector database)
An OpenAI API Key
(Optional but Recommended) A GitHub Personal Access Token (for higher API rate limits)
1. Clone the Repository
git clone <your-repository-url>
cd <repository-directory-name>


2. Set Up Environment Variables
Create a .env file in the root directory of the project and add your credentials:
# Supabase Credentials
SUPABASE_URL="YOUR_SUPABASE_URL"
SUPABASE_SERVICE_KEY="YOUR_SUPABASE_SERVICE_KEY"

# OpenAI API Key
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

# GitHub Token (Optional - for higher rate limits)
# Generate one here: [https://github.com/settings/tokens](https://github.com/settings/tokens)
GITHUB_TOKEN="YOUR_GITHUB_PERSONAL_ACCESS_TOKEN"


Replace the placeholder values with your actual keys.
3. Install Dependencies
It's recommended to use a virtual environment:
python -m venv .venv
source .venv/bin/activate # On Windows use `.\.venv\Scripts\activate`
RUN
```pip install -r requirements.txt```


4. Set Up Supabase Vector Database
Run the setup script to create the necessary tables and functions in your Supabase instance:
python setup_vector_db.py


Alternatively, you can manually run the SQL commands found in supabase_setup.sql using the Supabase SQL Editor.
5. Ingest Data into the Knowledge Base
Place the documents you want to ingest into a directory named documents in the project root.
For PDF/TXT/MD files: You can use the upload feature in the Streamlit app (agentic_rag_streamlit.py or combined_langchain_assistant.py) after launching it, or run the ingest_in_db.py script (ensure it points to your documents folder).
# Example using the script (modify ingest_in_db.py if needed)
# python ingest_in_db.py


For MDX/IPYNB files (and others): Use the Jupyter Notebook ingest.ipynb. Run the cells in the notebook to load, process, and store these specific file types. Make sure your documents are in the documents/docs subdirectory as configured in the notebook, or adjust the paths accordingly.
Usage
To run the combined Streamlit application:
streamlit run combined_langchain_assistant.py


This will launch the web application in your browser. You can interact with the RAG Knowledge Base in the first tab and the GitHub Resource Finder in the second tab.
If you want to run the standalone apps:
RAG Assistant: streamlit run agentic_rag_streamlit.py
GitHub Finder: streamlit run github_search_agent.py
File Structure
├── documents/              # Directory to place documents for ingestion (create if needed)
│   └── docs/               # Subdirectory specifically used by ingest.ipynb
├── .env                    # Stores API keys and credentials (create manually)
├── agentic_rag.py          # Core logic for the RAG agent (non-Streamlit)
├── agentic_rag_streamlit.py # Standalone Streamlit app for the RAG assistant
├── combined_langchain_assistant.py # Main Streamlit app combining RAG and GitHub features
├── github_search_agent.py  # Standalone Streamlit app for the GitHub finder
├── ingest.ipynb            # Jupyter Notebook for ingesting MDX/IPYNB files
├── ingest_in_db.py         # Script for ingesting PDF/TXT files into Supabase
├── setup_vector_db.py      # Python script to set up Supabase tables
