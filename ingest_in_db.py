# import basics
import os
import ssl
import urllib3
from dotenv import load_dotenv
import math
import time

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

# import supabase
from supabase.client import Client, create_client

# Configure SSL context to be more permissive
# This helps resolve SSL connection issues
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# load environment variables
load_dotenv()  

try:
    # initiate supabase db with custom options
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    
    # Create client with custom options
    supabase: Client = create_client(
        supabase_url, 
        supabase_key,
        options={
            "timeout": 60,  # Increase timeout
            "headers": {
                "Connection": "keep-alive"
            }
        }
    )
    
    # initiate embeddings model with timeout options
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60,  # Increase timeout for API calls
        max_retries=5  # Add retries for resilience
    )
    
    # load pdf docs from folder 'documents'
    loader = PyPDFDirectoryLoader("documents")
    
    # split the documents in multiple chunks
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Process in batches of 300 chunks
    batch_size = 300
    num_docs = len(docs)
    num_batches = math.ceil(num_docs / batch_size)
    
    print(f"Found {num_docs} document chunks, processing in {num_batches} batches of {batch_size} chunks each")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_docs)
        current_batch = docs[start_idx:end_idx]
        
        print(f"\nProcessing batch {i+1}/{num_batches} with {len(current_batch)} chunks (documents {start_idx+1}-{end_idx})...")
        
        try:
            # Apply SSL fix before vector store operation
            old_get_server_certificate = ssl.get_server_certificate
            ssl.get_server_certificate = lambda addr, ssl_context=None, timeout=None: ''
            
            # Store the current batch in the vector database
            vector_store = SupabaseVectorStore.from_documents(
                current_batch,
                embeddings,
                client=supabase,
                table_name="documents",
                query_name="rag_query",
                chunk_size=100,  # Reduced chunk size for API calls
            )
            
            # Restore original SSL behavior
            ssl.get_server_certificate = old_get_server_certificate
            
            print(f"Batch {i+1}/{num_batches} successfully stored!")
            
            # Add a small delay between batches to avoid rate limits
            if i < num_batches - 1:
                print("Waiting 2 seconds before processing next batch...")
                time.sleep(2)
                
        except Exception as e:
            print(f"Error processing batch {i+1}: {str(e)}")
            print("Continuing with next batch...")
    
    print("\nDocument upload process completed!")

except Exception as e:
    print(f"Error: {str(e)}")
    print("\nTroubleshooting SSL errors:")
    print("1. Check your network connection")
    print("2. Ensure your SSL certificates are up to date")
    print("3. Try updating your Python packages with: pip install --upgrade requests urllib3 pyopenssl")
    print("4. If behind a corporate firewall or proxy, check your network settings")