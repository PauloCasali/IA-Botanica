import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Configuration constants for local directories and models
DATA_DIR = "./data"
VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

def initialize_vector_database():
    """
    Reads Markdown clinical files, splits them into logical chunks by headers,
    generates local semantic embeddings using Ollama, and stores them in ChromaDB.
    """
    print("LOG: [RAG] Starting botanical knowledge base indexing process...")
    
    # Ensure the source data directory exists before proceeding
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Source directory '{DATA_DIR}' not found.")
        return

    # Filter and collect all markdown files inside the data path
    markdown_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.md')]
    if not markdown_files:
        print(f"ERROR: No markdown (.md) files found inside '{DATA_DIR}'.")
        return

    processed_documents = []

    # Configure structural markdown splitting to preserve plant identity integrity
    headers_to_split_on = [
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Process each markdown document sequentially
    for file_name in markdown_files:
        full_path = os.path.join(DATA_DIR, file_name)
        print(f"LOG: Reading and chunking source file: {file_name}...")
        
        with open(full_path, "r", encoding="utf-8") as file:
            markdown_content = file.read()
            
        chunks = markdown_splitter.split_text(markdown_content)
        
        # Inject metadata to keep track of the source file origin for the agents
        for chunk in chunks:
            chunk.metadata["source"] = file_name
            processed_documents.append(chunk)

    print(f"LOG: Total clinical fragments generated: {len(processed_documents)}")

    # Initialize the local Ollama embedding instance
    print(f"LOG: Generating embeddings using local model '{EMBEDDING_MODEL}' via Ollama...")
    local_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Populate and persist the Chroma vector database locally
    print(f"LOG: Saving vectors locally into directory '{VECTOR_DB_DIR}'...")
    vector_store = Chroma.from_documents(
        documents=processed_documents,
        embedding=local_embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    
    print("LOG: Vector database successfully initialized and persisted.")

if __name__ == "__main__":
    initialize_vector_database()