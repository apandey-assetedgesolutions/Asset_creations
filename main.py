import os
import warnings
import yaml
import requests
import streamlit as st
import json  # Missing import
from automation.apis.process_documents import APIClient, PDFHandler
from automation.model.processing_files import *
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings('ignore')

# Load configuration
@st.cache_data(hash_funcs={open: lambda _: None})
def load_config():
    try:
        with open("config.yaml", "r") as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        st.error(f"Error loading config file: {e}")
        return None

config = load_config()
if not config:
    st.error("Configuration file is missing or invalid.")
    st.stop()

# Extract configurations
user_email = config["usercred"]["user"]
unprocessed_docs_endpoint = config["apis"]["unprocessed_documents"]
get_document = config["apis"]["get_documents"]
InsertDocKeyValues = config["apis"]["InsertDocKeyValues"]
GetAllSteps = config["apis"]["GetAllSteps"]
InsertStepResult = config["apis"]["InsertStepResult"]

# Initialize API client
client = APIClient()

st.title("Document Processing Pipeline")

# Authenticate User
token = client.authenticate(email=user_email)
if not token:
    st.error("Authentication failed. Please check credentials.")
    st.stop()
st.success("Authentication successful!")

# Fetch unprocessed documents
unprocessed_documents = client.get_document_id(unprocessed_docs_endpoint)
for doc in unprocessed_documents:
    try:
        document_id = doc.get("DocumentId")
        if not document_id:
            print("Skipping entry due to missing DocumentId")
            continue

        response = client.make_request(f"{get_document}/{document_id}")
        # print(response)
        # response.raise_for_status()  # Raise an error for bad responses
        document_response = response
        output_file = document_response.get("DocumentName")
        document_content = document_response.get("DocumentContent")

        if output_file and document_content:
            PDFHandler.save_base64_as_pdf(document_content, output_file)
            # print(f"Saved document: {output_file}")
        else:
            print(f"Warning: Missing DocumentName or DocumentContent for ID {document_id}")

    except requests.exceptions.RequestException as e:
        print(f"Request error while processing document {document_id}: {e}")
    except KeyError as e:
        print(f"Missing expected key in response for Document ID {document_id}: {e}")
    except Exception as e:
        print(f"Unexpected error processing Document ID {document_id}: {e}")

if not unprocessed_documents:
    st.warning("No unprocessed documents found.")
else:
    st.write("Unprocessed Documents:", unprocessed_documents)

# Load and process documents
try:
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2', model_kwargs={'device': 'cpu'})
    doc_processor = DocumentProcessor(Config.DATAFILEPATH)
    documents = doc_processor.load_and_split_documents()

    vector_store = VectorStore(Config.VECTORDATABASEPATH, embeddings)
    vector_store.create_vector_store(documents)
    vectorstore_path = vector_store.load_vector_store()

    # Process documents with LLM
    llm_processor = LLMProcessor()
    instructions = "Extract the mentioned field details from the provided document while maintaining clarity and precision."
    response = llm_processor.process_request(vectorstore_path, instructions)

    response_content = response.content if hasattr(response, "content") else str(response)
    extracted_json = JSONExtractor.extract_json(response_content)
    st.json(extracted_json)

except Exception as e:
    st.error(f"Error processing documents: {e}")
    st.stop()

# Update Processed For All
try:
    UpdateProcessedForAll_payload = [doc["ActivityId"] for doc in unprocessed_documents]
    if UpdateProcessedForAll_payload:
        UpdateProcessedForAll_url = f"/GenAI/UpdateProcessedForAll/{UpdateProcessedForAll_payload}"
        Update_Processed_ForAll = client.post_request(endpoint=UpdateProcessedForAll_url)
        st.write("Update Processed For All:", Update_Processed_ForAll)
    else:
        st.warning("No documents to update.")
except Exception as e:
    st.error(f"Error updating processed documents: {e}")

# Insert Document Key Values
def Insert_DocKeyValues_transform_json(data):
    formatted_data = []
    
    for key, value in data.items():
        genAIDocumentId = data.get("genAIDocumentId", 1)  # Get actual ID instead of hardcoding

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                formatted_data.append({
                    "genAIDocumentId": genAIDocumentId,
                    "keyName": f"{key} - {sub_key}",
                    "keyValue": sub_value
                })
        elif key != "genAIDocumentId":
            formatted_data.append({
                "genAIDocumentId": genAIDocumentId,
                "keyName": key,
                "keyValue": value
            })

    return formatted_data

try:
    if isinstance(extracted_json, str):
        extracted_json = json.loads(extracted_json)  
    Insert_DocKeyValues_ = Insert_DocKeyValues_transform_json(extracted_json)
    st.write('Inserted Document Key Values payload', Insert_DocKeyValues_)
    Insert_DocKeyValues = client.post_request(endpoint=InsertDocKeyValues, payload=Insert_DocKeyValues_)
    st.write("Inserted Document Key Values:", Insert_DocKeyValues)
except Exception as e:
    st.error(f"Error inserting document key values: {e}")

# Fetch all steps
try:
    Get_All_Steps = client.get_request(GetAllSteps)
    st.write("Fetched Steps:", Get_All_Steps)
except Exception as e:
    st.error(f"Error fetching steps: {e}")

# Insert step results
try:
    InsertStepResult_payload = [
        {"activityId": doc["ActivityId"], "genAIDocumentId": doc["GenAIDocumentId"], "stepId": 2, "processResult": True, "processMessage": "Created"}
        for doc in unprocessed_documents
    ]
    Insert_Step_Result = client.post_request(endpoint=InsertStepResult, payload=InsertStepResult_payload)
    st.write("Inserted Step Results:", Insert_Step_Result)
except Exception as e:
    st.error(f"Error inserting step results: {e}")

# Upload extracted data
try:
    formatted_data = client.format_asset_data(extracted_json)
    upload_response = client.upload_asset(formatted_data)
    st.write("Upload Response:", upload_response)
except Exception as e:
    st.error(f"Error uploading data: {e}")
