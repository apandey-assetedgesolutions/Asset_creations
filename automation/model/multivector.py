import os
import re
import json
import concurrent.futures
from dotenv import load_dotenv
from langsmith import traceable
from typing import List, Dict
from datetime import date

from prompt import PromptsInstructions
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from pydantic.v1 import BaseModel
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

# os.environ['OPENAI_API_TYPE '] = 'azure'
# os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"

os.environ['LANGSMITH_TRACING '] = os.getenv('LANGSMITH_TRACING')
os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_PROJECT"] = os.getenv('LANGSMITH_PROJECT')

DATAFILEPATH = r"./data/caligan"
VECTORDATABASEPATH_1 = "multivector/parent"
VECTORDATABASEPATH_2 = "multivector/child"

# Initialize the model
llm = AzureChatOpenAI(deployment_name="gpt-4o", model_name="gpt-4o", temperature=0.8,top_p=0.9)
# llm = AzureChatOpenAI(deployment_name="gpt-4o-mini", model_name="gpt-4o-mini", temperature=0.8,top_p=0.9)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2', model_kwargs={'device': 'cpu'})
PROMPTS = PromptsInstructions()


class Details(BaseModel):
    AssetName: str
    AbbreviationName: str
    SecurityType: str
    InceptionDate: date
    Strategy: str

details_parser = JsonOutputParser(pydantic_object=Details)

class DocumentLoader:
    """Handles document loading based on file type."""
    
    def __init__(self, data_path):
        self.data_path = data_path

    def get_loader(self, file_path):
        """Returns the appropriate document loader based on file extension."""
        if file_path.endswith(".pdf"):
            return PyPDFLoader(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            return UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def load_file(self, file_path):
        """Loads a file and returns its extracted documents."""
        try:
            loader = self.get_loader(file_path)
            return loader.load()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def load_all_files(self):
        """Loads all files in the directory using multi-threading."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Directory {self.data_path} does not exist.")

        file_paths = [os.path.join(self.data_path, filename) for filename in os.listdir(self.data_path)]
        documents = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.load_file, file_paths)
            for result in results:
                documents.extend(result)

        print(f"Successfully loaded {len(documents)} documents.")
        return documents


class VectorStoreManager:
    """Manages vector store operations."""
    
    def __init__(self, embeddings, parent_path, child_path):
        self.embeddings = embeddings
        self.parent_path = parent_path
        self.child_path = child_path

    def create_vector_store(self, docs, chunk_size,chunk_overlap, save_path):
        """Creates and saves a vector store with the given chunk size."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size , chunk_overlap= chunk_overlap)
        split_docs = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(split_docs, self.embeddings)
        vector_store.save_local(save_path)
        return vector_store

    def load_vector_store(self, path):
        """Loads a stored vector database."""
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)

    def initialize_vector_stores(self, documents):
        """Creates and loads parent and child vector stores."""
        parent_store = self.create_vector_store(documents, chunk_size=10000,chunk_overlap=1000, save_path=self.parent_path)
        child_store = self.create_vector_store(documents, chunk_size=2000, chunk_overlap=200,save_path=self.child_path)

        return self.load_vector_store(self.parent_path), self.load_vector_store(self.child_path)


class DataExtractionPipeline:
    """Handles document retrieval and data extraction."""
    
    def __init__(self, parent_vectorstore, child_vectorstore, llm):
        self.parent_vectorstore = parent_vectorstore
        self.child_vectorstore = child_vectorstore
        self.llm = llm
        self.prompt = self.create_prompt()

    def create_prompt(self):
        """Creates a structured prompt for document processing."""
        return PromptTemplate(
            template=PROMPTS.asset_creation(),
            input_variables=["text", "context", "context2"],
            # partial_variables={"format_instructions": details_parser.get_format_instructions()},
        )

    def process_data(self, input_text):
        """Executes the retrieval and extraction process."""
        retrieval_chain = (
            {
                "context": self.parent_vectorstore.as_retriever(search_kwargs={'k': 6}),
                "context2": self.child_vectorstore.as_retriever(search_kwargs={'k': 6}),
                "text": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
        )

        return retrieval_chain.invoke(input_text)


class JSONExtractor:
    """Extracts JSON data from response content."""
    
    @staticmethod
    def extract_json(response_content):
        """Extracts JSON from a response string."""
        match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
        if match:
            json_content = match.group(1)
            try:
                return json.dumps(json.loads(json_content), indent=2)
            except json.JSONDecodeError as e:
                return f"Invalid JSON: {e}"
        return "No JSON content found."



document_loader = DocumentLoader(DATAFILEPATH)
documents = document_loader.load_all_files()


vector_manager = VectorStoreManager(embeddings, VECTORDATABASEPATH_1, VECTORDATABASEPATH_2)
parent_vectorstore, child_vectorstore = vector_manager.initialize_vector_stores(documents)

data_pipeline = DataExtractionPipeline(parent_vectorstore, child_vectorstore, llm)
# with open("./fields_list.txt", "r") as files:
#     fields = files.read().strip()


def content_piepline():
    instructions = "Extract the mentioned fields details from the provided document while maintaining clarity and precision."
    response = data_pipeline.process_data(instructions)
    # print(response)
    response_content = response.content if hasattr(response, "content") else str(response)
    print("*"*40)
    extracted_json = JSONExtractor.extract_json(response_content)
    print(extracted_json)
    return extracted_json

content_piepline()

# instructions = "Extract the mentioned fields details from the provided document while maintaining clarity and precision."
# response = data_pipeline.process_data(instructions)
# response_content = response.content if hasattr(response, "content") else str(response)
# extracted_json = JSONExtractor.extract_json(response_content)
# print(extracted_json)
# print(extracted_json)
