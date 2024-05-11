import os
import httpx
from backoff import on_exception, expo
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.embeddings import resolve_embed_model
from llama_parse import LlamaParse
from config.toolsconfig import get_tools


class DocumentQueryEngine:
    def __init__(self, data_directory="./data", model_name="mistral", embed_model_name="local:BAAI/bge-m3"):
        load_dotenv()
        self.data_directory = data_directory
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.query_engine = None
        self.tools = None  
        self.setup()

    def setup(self):
        self.ensure_data_directory()
        self.llama_parse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.llama_parse_api_key:
            raise Exception("API key for LlamaParse is not set in .env file")
        
        self.llm = Ollama(model=self.model_name, request_timeout=30.0)
        self.parser = LlamaParse(result_type="markdown", api_key=self.llama_parse_api_key)
        self.file_extractor = {".pdf": self.parser}
        self.load_documents()

    def ensure_data_directory(self):
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def load_documents(self):
        try:
            documents = SimpleDirectoryReader(self.data_directory, file_extractor=self.file_extractor).load_data()
            embed_model = resolve_embed_model(self.embed_model_name)
            vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            self.query_engine = vector_index.as_query_engine(llm=self.llm)
        except Exception as e:
            print(f"Failed to load documents: {e}")
            self.query_engine = None

    def setup_tools(self):
        if not self.query_engine:
            raise ValueError("Query engine is not initialized.")
        self.tools = get_tools(self.query_engine)
        return self.tools




