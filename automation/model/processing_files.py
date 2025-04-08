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
            You are a highly specialized expert in document data extraction, with a strong understanding of financial and investment-related content. Given a {text}, your objective is to extract specific fields accurately—by reasoning through context, structure, and formatting patterns. Follow a deliberate, step-by-step thought process. Use self-questions to guide extraction and apply logic to resolve ambiguity.
            🧠 THINK, THEN ACT:
            For each field, follow this process:
            1.	Think:
            o	"Where in this text would this information most likely appear?"
            o	"Are there variations or synonyms I should account for?"
            o	"Does this match known patterns from financial documents?"
            2.	Act:
            o	Extract the correct value.
            o	Normalize format where needed.
            o	If unavailable, return "N/A".
            🧩 INSTRUCTIONS:
            1.	Pattern Recognition: Identify and extract relevant fields using contextual and structural cues.
            2.	Variation Handling: Normalize differences in format (e.g., "Feb 1, 2022" → "02/01/2022").
            3.	Noise Filtering: Ignore irrelevant or unrelated content. Focus only on actionable data.
            4.	Strict JSON Format: Return output only in the following format. No additional text or commentary.
            5.	Missing Fields: Return "N/A" where data is not available or cannot be confidently inferred.

            🎯 OUTPUT FORMAT (JSON)
            json
            CopyEdit
            {{
            "Asset Name": "",
            "Abbreviation Name": "",
            "Asset Manager": "",
            "Security Type": "",
            "Inception Date": "",
            "Investment Status": "",
            "Asset Status": "",
            "Strategy": "",
            "Substrategy": "",
            "Primary Analyst": "",
            "Secondary Analyst": "",
            "Sector": "",
            "Subsector": "",
            "Asset Class": "",
            "Region": "",
            "Classification": "",
            "Benchmark 1": "",
            "Benchmark 2": "",
            "IDD Rating": "",
            "ODD Rating": "",
            "Legal Structure": "",
            "Domicile": "",
            "AUM Time Series": "",
            "Exposure Time Series": "",
            "Return Time Series": "",
            "Management Company": ""
            }}

            🔎 EXTRACTION TIPS:
            •	Security Type → Presentation document (e.g., "Private Equity", "Hedge Fund").
            •	Sector → "Detail" tab in Presentation.
            •	Region → If listed as a percentage, convert to region name (e.g., "86% North America" → "North America").
            •	Legal Structure → From "Attributes" tab; interpret references to legal types (e.g., "Delaware Limited Partnership" → "Partnership").
            •	Domicile → Usually in the Attributes tab (e.g., "Delaware" → "U.S.").
            •	AUM, Exposure, Return Time Series → Extract from respective spreadsheet/document sections.
            •	If multiple values are found, prioritize the most explicit and structured one.



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


# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2', model_kwargs={'device': 'cpu'})
# doc_processor = DocumentProcessor(Config.DATAFILEPATH)
# documents = doc_processor.load_and_split_documents()

# vector_store = VectorStore(Config.VECTORDATABASEPATH, embeddings)
# vector_store.create_vector_store(documents)
# vectorstore_path = vector_store.load_vector_store()

# llm_processor = LLMProcessor()
# instructions = "Extract the mentioned fields details from the provided document while maintaining clarity and precision"
# response = llm_processor.process_request(vectorstore_path, instructions)

# response_content = response.content if hasattr(response, "content") else str(response)
# extracted_json = JSONExtractor.extract_json(response_content)
# print(extracted_json)