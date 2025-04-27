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
