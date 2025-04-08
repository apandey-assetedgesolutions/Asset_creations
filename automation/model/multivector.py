import os
import re
import json
import concurrent.futures
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from pydantic.v1 import BaseModel
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

DATAFILEPATH = r"./data/IPPE"
VECTORDATABASEPATH_1 = "multivector/parent"
VECTORDATABASEPATH_2 = "multivector/child"

# Initialize the model
llm = AzureChatOpenAI(deployment_name="gpt-4o", model_name="gpt-4o", temperature=0.8)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2', model_kwargs={'device': 'cpu'})

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



            Context1: {context}
            Context2: {context2}
            """,
            input_variables=["text", "context", "context2"],
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
    response_content = response.content if hasattr(response, "content") else str(response)
    extracted_json = JSONExtractor.extract_json(response_content)
    print(extracted_json)
    return extracted_json

# content_piepline()

# instructions = "Extract the mentioned fields details from the provided document while maintaining clarity and precision."
# response = data_pipeline.process_data(instructions)
# response_content = response.content if hasattr(response, "content") else str(response)
# extracted_json = JSONExtractor.extract_json(response_content)
# print(extracted_json)
# print(extracted_json)
