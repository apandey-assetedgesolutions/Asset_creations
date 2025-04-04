import uuid
import os
import re
import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from pydantic.v1 import BaseModel
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Config:
    DATAFILEPATH = r"D:\Users\muruganandham\API AUtomation\API_automation_assetedge\data\IPPE"
    VECTORDATABASEPATH = "VectoreDB/FAISS"

    load_dotenv()
    os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

def get_loader(file_path):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        return UnstructuredExcelLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


class DocumentProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        # self.loader = DirectoryLoader(self.data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)

    def load_and_split_documents(self):
        documents = []
        for filename in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, filename)
            try:
                loader = get_loader(file_path)
                documents.extend(loader.load())  # Load and append documents
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        # documents = self.loader.load()
        return self.text_splitter.split_documents(documents)

class VectorStore:
    def __init__(self, vector_db_path, embeddings):
        self.vector_db_path = vector_db_path
        self.embeddings = embeddings
        self.vectorstore = None

    def create_vector_store(self, documents):
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.vectorstore.save_local(self.vector_db_path)

    def load_vector_store(self):
        return FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)

class LLMProcessor:
    def __init__(self):
        self.llm = AzureChatOpenAI(deployment_name="gpt-4o", model_name="gpt-4o",temperature=0.8)
    
    def create_prompt(self):
        return PromptTemplate(
            template="""
            You are an expert in document data extraction with a deep understanding of financial and investment-related documents. Given a {text}, your task is to accurately extract specific fields while preserving structure, recognizing patterns, and handling formatting variations. You must ignore irrelevant content and ensure high accuracy.
            Instructions:
                1. Pattern Recognition: Identify and extract relevant data based on context and structure.
                2. Handling Variations: Normalize different formats while maintaining data integrity.
                3. Strict Output Format: Return only the extracted data in the JSON structure below—no additional text, explanations, or formatting changes.
                4. Missing Fields: If a field is unavailable, return "N/A".
                
            Need to extract fields :
                1. Security Type: Extract from the Presentation document (e.g., "Private Equity," "Hedge Fund").
                2. Sector: Found in the Detail tab within the Presentation.
                3. Region: Located in the Presentation, often given in percentage format (e.g., "86% North America" → "North America").
                4. Legal Structure: Extract from the Attributes tab based on PPM document references (e.g., "Delaware Limited Partnership" → "Partnership").
                5. Domicile: Found in the Attributes tab within the Presentation (e.g., "Delaware" → "U.S.").
                6. AUM Time Series: Extracted from the AUM Spreadsheet provided.
                7. Exposure Time Series: Extracted from the Exposure Spreadsheet provided.
                8. Return Time Series: Available in the Presentation document.
                9. Management Company: Extract from the document. 

            Context: {context}
            """,
            input_variables=["text", "context"],
        )
    
    def process_request(self, vectorstore_path, instructions):
        prompt = self.create_prompt()
        retrieval_chain = (
            {"context": vectorstore_path.as_retriever(), "text": RunnablePassthrough()} | prompt | self.llm
        )
        return retrieval_chain.invoke(instructions)

class JSONExtractor:
    @staticmethod
    def extract_json(response_content):
        match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
        if match:
            json_content = match.group(1)
            try:
                return json.dumps(json.loads(json_content), indent=2)
            except json.JSONDecodeError as e:
                return f"Invalid JSON: {e}"
        return "No JSON content found."


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2', model_kwargs={'device': 'cpu'})
doc_processor = DocumentProcessor(Config.DATAFILEPATH)
documents = doc_processor.load_and_split_documents()

vector_store = VectorStore(Config.VECTORDATABASEPATH, embeddings)
vector_store.create_vector_store(documents)
vectorstore_path = vector_store.load_vector_store()

llm_processor = LLMProcessor()
instructions = "Extract the mentioned fields details from the provided document while maintaining clarity and precision"
response = llm_processor.process_request(vectorstore_path, instructions)

response_content = response.content if hasattr(response, "content") else str(response)
extracted_json = JSONExtractor.extract_json(response_content)
print(extracted_json)