import os
import uuid
import json
import re
import time
from typing import List, Dict
from datetime import date
import concurrent.futures

from dotenv import load_dotenv
from prompt import PromptsInstructions
from pydantic.v1 import BaseModel

from langsmith import traceable
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.storage.in_memory import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"

os.environ['LANGSMITH_TRACING '] = os.getenv('LANGSMITH_TRACING')
os.environ['LANGSMITH_ENDPOINT'] = os.getenv('LANGSMITH_ENDPOINT')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_PROJECT"] = os.getenv('LANGSMITH_PROJECT')

# Constants
DATAFILEPATH = r"./data/Caligan"
VECTORDATABASEPATH_1 = "multivector/parent"
VECTORDATABASEPATH_2 = "multivector/child"

# LLM and Embeddings
llm = AzureChatOpenAI(deployment_name="gpt-4o", model_name="gpt-4o", temperature=0.8, top_p=0.9)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2', model_kwargs={'device': 'cpu'})

PROMPTS = PromptsInstructions()

# Extracted fields List
class Details(BaseModel):
    AssetName: str
    AbbreviationName: str
    SecurityType: str
    InceptionDate: date
    Strategy: str

details_parser = JsonOutputParser(pydantic_object=Details)
class DocumentLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def get_loader(self, file_path):
        if file_path.endswith(".pdf"):
            return PyPDFLoader(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            return UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def load_file(self, file_path):
        try:
            loader = self.get_loader(file_path)
            return loader.load()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    def load_all_files(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Directory {self.data_path} does not exist.")

        file_paths = [os.path.join(self.data_path, filename) for filename in os.listdir(self.data_path)]
        documents = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.load_file, file_paths)
            for result in results:
                documents.extend(result)

        print(f"✅ Successfully loaded {len(documents)} documents.")
        return documents


class DocumentProcessor:
    def __init__(self):
        self.llm = AzureChatOpenAI(deployment_name="gpt-4o", model_name="gpt-4o")
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L12-v2', 
            model_kwargs={'device': 'cpu'}
        )
        self.prompt = self._create_prompt()

    def _create_prompt(self):
        return PromptTemplate(
            template=PROMPTS.asset_creation(),
            input_variables=["text", "context"],
            partial_variables={"format_instructions": details_parser.get_format_instructions()},
        )
    @traceable
    def process_folder(self, docs: List) -> Dict:
        start_time = time.time()
        try:
            print("🔍 Creating vector store...")
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            store = InMemoryByteStore()
            id_key = "1"
            retriever = MultiVectorRetriever(
                vectorstore=vectorstore,
                byte_store=store,
                id_key=id_key,
            )

            doc_ids = [str(uuid.uuid4()) for _ in docs]
            child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
            sub_docs = []
            for i, doc in enumerate(docs):
                _id = doc_ids[i]
                _sub_docs = child_text_splitter.split_documents([doc])
                for _doc in _sub_docs:
                    _doc.metadata[id_key] = _id
                sub_docs.extend(_sub_docs)

            retriever.vectorstore.add_documents(sub_docs)
            retriever.docstore.mset(list(zip(doc_ids, docs)))

            retrieval_chain = (
                {"context": retriever, "text": RunnablePassthrough()} | self.prompt | self.llm | details_parser
            )

            print("🚀 Extracting document information...")
            instructions = "Extract the mentioned fields details from the provided document while maintaining clarity and precision"
            response = retrieval_chain.invoke(instructions)
            print(response)
            result = self._extract_json(response.content if hasattr(response, "content") else str(response))
            end_time = time.time()
            print(f"⏱️ Processing time: {end_time - start_time:.2f} seconds")
            return result

        except Exception as e:
            return {"error": str(e)}

    def _extract_json(self, response_conten) -> Dict:
        try:
            return json.dumps(json.loads(response_conten), indent=2)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"


def main():
    document_loader = DocumentLoader(DATAFILEPATH)
    documents = document_loader.load_all_files()

    processor = DocumentProcessor()
    results = processor.process_folder(documents)

    output_file = 'document_extraction_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Processing Complete!")
    print(f"📁 Document batch processed successfully.")
    print(f"💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()
