import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

VECTOR_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

class BotanicalMCPServer:
    """
    Standard Model Context Protocol implementation for the botanical domain.
    Exposes document retrieval utilities to the multi-agent network as structured tools.
    """
    def __init__(self):
        # Validate that the vector storage layer exists
        if not os.path.exists(VECTOR_DB_DIR):
            raise FileNotFoundError(
                f"ERROR: Vector database missing at '{VECTOR_DB_DIR}'. "
                "Execute 'src/database/indexer.py' first."
            )
        
        # Connect to local database instance using Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory=VECTOR_DB_DIR, 
            embedding_function=self.embeddings
        )
        print("LOG: [MCP Server] Botanical context server successfully initialized.")

    def get_tools_manifest(self) -> dict:
        """
        Returns the structured declaration of available tools exposed by the protocol.
        """
        return {
            "search_clinical_context": {
                "description": "Searches the document database (RAG) for plant symptoms, irrigation, illumination, and pet safety guidelines.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The symptom description or plant name provided by the user."}
                    },
                    "required": ["query"]
                }
            }
        }

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Routes the tool execution request to the appropriate back-end function.
        """
        if tool_name == "search_clinical_context":
            query = arguments.get("query", "")
            return self._tool_search_clinical_context(query)
        else:
            return f"ERROR: Tool '{tool_name}' is not supported by this server."

    def _tool_search_clinical_context(self, query: str) -> str:
        """
        Queries the database using vector similarity and returns matching records.
        """
        # Fetch the top two most semantically relevant documentation parts
        docs = self.vector_store.similarity_search(query, k=2)
        
        if not docs:
            return "No relevant clinical documentation found for the provided query."
            
        retrieved_content = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            retrieved_content.append(f"--- Document Fragment (Source: {source}) ---\n{doc.page_content}\n")
            
        return "\n".join(retrieved_content)

# Global instantiation accessible by orchestration layers
mcp_server = BotanicalMCPServer()