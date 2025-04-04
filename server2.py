import os
import warnings
import yaml
import requests
import json
from automation.apis.process_documents import APIClient, PDFHandler
from automation.model.multivector import content_piepline
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings('ignore')

# Load configuration
def load_config():
    try:
        with open("config.yaml", "r") as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading config file: {e}")
        return None

config = load_config()
if not config:
    print("Configuration file is missing or invalid.")
    exit(1)

# Extract configurations
user_email = config["usercred"]["user"]
unprocessed_docs_endpoint = config["apis"]["unprocessed_documents"]
get_document = config["apis"]["get_documents"]
InsertDocKeyValues = config["apis"]["InsertDocKeyValues"]
GetAllSteps = config["apis"]["GetAllSteps"]
InsertStepResult = config["apis"]["InsertStepResult"]

# Initialize API client
client = APIClient()

print("Document Processing Pipeline")

# Authenticate User
token = client.authenticate(email=user_email)
if not token:
    print("Authentication failed. Please check credentials.")
    exit(1)
print("Authentication successful!")

# Fetch unprocessed documents
unprocessed_documents = client.get_document_id(unprocessed_docs_endpoint)
for doc in unprocessed_documents:
    try:
        document_id = doc.get("DocumentId")
        if not document_id:
            print("Skipping entry due to missing DocumentId")
            continue

        response = client.make_request(f"{get_document}/{document_id}")
        document_response = response
        output_file = document_response.get("DocumentName")
        document_content = document_response.get("DocumentContent")

        if output_file and document_content:
            PDFHandler.save_base64_as_pdf(document_content, output_file)
            print(f"Saved document: {output_file}")
        else:
            print(f"Warning: Missing DocumentName or DocumentContent for ID {document_id}")

    except requests.exceptions.RequestException as e:
        print(f"Request error while processing document {document_id}: {e}")
    except KeyError as e:
        print(f"Missing expected key in response for Document ID {document_id}: {e}")
    except Exception as e:
        print(f"Unexpected error processing Document ID {document_id}: {e}")

if not unprocessed_documents:
    print("No unprocessed documents found.")
else:
    print("Unprocessed Documents:", unprocessed_documents)

# Load and process documents
try:
    extracted_json = content_piepline()
    print(json.dumps(extracted_json, indent=2))

except Exception as e:
    print(f"Error processing documents: {e}")
    exit(1)

# Update Processed For All
try:
    UpdateProcessedForAll_payload = [doc["ActivityId"] for doc in unprocessed_documents]
    if UpdateProcessedForAll_payload:
        UpdateProcessedForAll_url = f"/GenAI/UpdateProcessedForAll/{UpdateProcessedForAll_payload}"
        Update_Processed_ForAll = client.post_request(endpoint=UpdateProcessedForAll_url)
        print("Update Processed For All:", Update_Processed_ForAll)
    else:
        print("No documents to update.")
except Exception as e:
    print(f"Error updating processed documents: {e}")

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
    print('Inserted Document Key Values payload', Insert_DocKeyValues_)
    Insert_DocKeyValues = client.post_request(endpoint=InsertDocKeyValues, payload=Insert_DocKeyValues_)
    print("Inserted Document Key Values:", Insert_DocKeyValues)
except Exception as e:
    print(f"Error inserting document key values: {e}")

# Fetch all steps
try:
    Get_All_Steps = client.get_request(GetAllSteps)
    print("Fetched Steps:", Get_All_Steps)
except Exception as e:
    print(f"Error fetching steps: {e}")

# Insert step results
try:
    InsertStepResult_payload = [
        {"activityId": doc["ActivityId"], "genAIDocumentId": doc["GenAIDocumentId"], "stepId": 2, "processResult": True , "processMessage": "Success"}
        for doc in unprocessed_documents
    ]
    Insert_Step_Result = client.post_request(endpoint=InsertStepResult, payload=InsertStepResult_payload)
    print("Inserted Step Results:", Insert_Step_Result)
except Exception as e:
    print(f"Error inserting step results: {e}")

# Upload extracted data
try:
    formatted_data = client.format_asset_data(extracted_json)
    upload_response = client.upload_asset(formatted_data)
    print("Upload Response:", upload_response)
except Exception as e:
    print(f"Error uploading data: {e}")


#Step 6 Create Share Classes
shareclass_payload = {
    "shareClassId": 0,
    "shareClassName": "My Share Class",
    "assetId": upload_response,
    "portfolioId": None,
    "inceptionDate": "2025-04-03T09:11:51.332Z",
    "effectiveDate": "2025-04-03T09:11:51.332Z",
    "minInvestment": None,
    "subscriptionFrequencyId": None,
    "subscriptionCurrencyIdList": "",
    "taxReportingId": None,
    "votingShares": False,
    "newIssues": False,
    "trackingFrequencyId": None,
    "trackingById": None,
    "accredited": False,
    "qualifiedPurchaser": False,
    "qualifiedClient": False,
    "initialNAV": None,
    "businessDays": False,
    "modifiedBy": 38,
    "liquidityTermsAbrev": None,
    "feeDetails": {
        "shareClassId": 0,
        "mgmtFeeTierId": 0,
        "mgmtFeeTierDesc": None,
        "mgmtFeeFrequencyId": None,
        "isMgmtFeeFreqPassThrough": False,
        "perfFeeTierId": 0,
        "perfFeeTierDesc": None,
        "perfFeePaymentFrequencyId": None,
        "perfFeeAccrualFrequencyId": None,
        "hurdleRateId": None,
        "hurdleValue": None,
        "hurdleRateBenchMarkId": 0,
        "lossRecovery": False,
        "lossRecoveryResetId": None,
        "modifiedBy": 38
    }
}
try:
    share_class_creation  = client.post_request(endpoint= "/AssetShareClass/InsertOrUpdateShareClass", payload=shareclass_payload)
    print(share_class_creation)
except Exception as e:
    print(f"Exception: {Exception}")
