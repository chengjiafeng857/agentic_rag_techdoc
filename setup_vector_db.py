import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Supabase credentials
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

# SQL commands to set up the vector database
setup_sql = [
    "create extension if not exists vector;",
    
    """create table if not exists
      documents (
        id uuid primary key,
        content text,
        metadata jsonb,
        embedding vector (1536)
      );""",
    
    """create or replace function match_documents (
      query_embedding vector (1536),
      filter jsonb default '{}'
    ) returns table (
      id uuid,
      content text,
      metadata jsonb,
      similarity float
    ) language plpgsql as $$
    #variable_conflict use_column
    begin
      return query
      select
        id,
        content,
        metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
      from documents
      where metadata @> filter
      order by documents.embedding <=> query_embedding;
    end;
    $$;""",
    
    """create or replace function rag_query (
      query_embedding vector (1536),
      filter jsonb default '{}'
    ) returns table (
      id uuid,
      content text,
      metadata jsonb,
      similarity float
    ) language plpgsql as $$
    #variable_conflict use_column
    begin
      return query
      select
        id,
        content,
        metadata,
        1 - (documents.embedding <=> query_embedding) as similarity
      from documents
      where metadata @> filter
      order by documents.embedding <=> query_embedding;
    end;
    $$;"""
]

print("Setting up vector database in Supabase...")

# REST API endpoint for SQL execution
sql_url = f"{supabase_url}/rest/v1/rpc/exec_sql"
headers = {
    "apikey": supabase_key,
    "Authorization": f"Bearer {supabase_key}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

for i, sql in enumerate(setup_sql):
    print(f"Executing SQL command {i+1}/{len(setup_sql)}...")
    try:
        # Using the Supabase REST API to execute SQL
        response = requests.post(
            sql_url,
            headers=headers,
            json={"query": sql}
        )
        
        if response.status_code >= 300:
            print(f"Warning: Command {i+1} returned status code {response.status_code}")
            print(response.text)
        
    except Exception as e:
        print(f"Error executing SQL command {i+1}: {str(e)}")

print("\nVector database setup process completed!")
print("You can now run your ingest.ipynb notebook to store documents.")