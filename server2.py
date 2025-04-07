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

# # Update Processed For All
# try:
#     UpdateProcessedForAll_payload = [doc["ActivityId"] for doc in unprocessed_documents]
#     if UpdateProcessedForAll_payload:
#         UpdateProcessedForAll_url = f"/GenAI/UpdateProcessedForAll/{UpdateProcessedForAll_payload}"
#         Update_Processed_ForAll = client.post_request(endpoint=UpdateProcessedForAll_url)
#         print("Update Processed For All:", Update_Processed_ForAll)
#     else:
#         print("No documents to update.")
# except Exception as e:
#     print(f"Error updating processed documents: {e}")

print("Step 1 : Insert Document Key Values")
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

print("Step 2: Asset creation")
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

print("Step3 Asset Attributes: ")
Attribute_creation_payload = {
  "assetId": upload_response,
  "assetAttributeId": 0,
  "assetAttributeHFId": 0,
  "assetAttributePEId": 0,
  "assetAttributeMSId": 0,
  "assetFamilyId": 0,
  "assetFamily": "string",
  "chartOfAccountId": 0,
  "chartOfAccount": "string",
  "operatingCurrencyId": 0,
  "operatingCurrency": "string",
  "legalStructureId": 0,
  "legalStructure": "string",
  "domicileCountryId": 0,
  "domicileCountry": "string",
  "quarterlyLetter": True,
  "assetSecurityTypeId": 0,
  "assetSecurityType": "string",
  "auditCompletionDate": "2025-04-07T08:45:08.678Z",
  "investmentTypeId": 0,
  "investmentType": "string",
  "investmentPack": True,
  "assetFactSheet": True,
  "allowsSidePocket": True,
  "investsInOtherFundsId": 0,
  "investsInOtherFunds": "string",
  "exposureReportFrequencyId": 0,
  "exposureReportFrequency": "string",
  "personalCapital": "string",
  "accceptsManagedAccounts": True,
  "investsInManagedAccount": True,
  "tradingCity": "string",
  "prohibitsUSInvestors": True,
  "filingStatusId": 0,
  "filingStatus": "string",
  "keyMan": True,
  "investorLetterFrequencyId": 0,
  "investorLetterFrequency": "string",
  "perfEstimateFrequencyId": 0,
  "perfEstimateFrequency": "string",
  "bloombergId": "string",
  "reportingFrequencyId": 0,
  "reportingFrequency": "string",
  "fiscalYear": "2025-04-07T08:45:08.679Z",
  "taxId": "string",
  "cimaId": "string",
  "formDNumber": "string",
  "vintageYear": "2025-04-07T08:45:08.679Z",
  "targetFundRaise": 0,
  "estimatedFirstCloseDate": "2025-04-07T08:45:08.679Z",
  "estimatedFirstCloseAmount": 0,
  "estimatedSecondCloseDate": "2025-04-07T08:45:08.679Z",
  "estimatedSecondCloseAmount": 0,
  "offerCoInvestment": True,
  "singleDealFund": True,
  "continuationFund": True,
  "issuerId": 0,
  "issuer": "string",
  "sharesOutstanding": 0,
  "cusip": "string",
  "isin": "string",
  "managementFee": 0,
  "fundFeeExpenses": 0,
  "otherExpenses": 0,
  "expenseRatio": 0,
  "distributionFrequencyId": 0,
  "distributionFrequency": "string",
  "isMarketable": True,
  "modifiedBy": 0,
  "isClientSpecific": True
}
try:
    AttributesHF_creation  = client.post_request(endpoint= "/Assets/InsertUpdateAssetAttributesHF", payload=Attribute_creation_payload)
    print(AttributesHF_creation)
except:
    AttributesPE_creation  = client.post_request(endpoint= "/Assets/InsertUpdateAssetAttributesPE", payload=Attribute_creation_payload)
    print(AttributesPE_creation)

#Create Benchmarks
Create_bmk = {
  "assetId": upload_response,
  "benchMarkXRefId": 0,
  "benchMarkXRefTypeId": 0,
  "entityTypeId": 0,
  "entityId": upload_response,
  "isFixedRate": True,
  "bmEntityTypeId": 0,
  "bmEntityId": 0,
  "bmValue": 0,
  "isMrktEqBM": True,
  "sortOrder": 0,
  "modifiedBy": 0
}
try:
    asset_create_bmk  = client.post_request(endpoint= "/Assets/InsertUpdateBMSettings", payload=Create_bmk)
    print(asset_create_bmk)
except Exception as e:
    print(f"Exception: {Exception}")

#Create service provider
assetCompanyXRefId = 1
CompanyId = 1
CompanyTypeId = 1
create_service_provider_url = f"/Assets/InsertUpdateServiceProvider?assetCompanyXRefId={assetCompanyXRefId}&CompanyId={CompanyId}&CompanyTypeId={CompanyTypeId}&AssetId={upload_response}"

try:
    create_service_provider  = client.post_request(endpoint= create_service_provider_url)
    print(create_service_provider)
except Exception as e:
    print(f"Exception: {Exception}")


print("Step 4: share class")

shareclass_payload = {
    "shareClassId": 0,
    "shareClassName": "Default",
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

print("Step 5: Liquitity creation")
Liquitity_creation_payload = {
  "redemptionTermsId": 0,
  "shareClassid": 0,
  "lockType": 0,
  "penaltyPercent": 0,
  "redemptionFeePercent": 0,
  "rollingLockup": True,
  "anniversary": True,
  "redemptionFrequencyId": 0,
  "lockupFrequencyId": 0,
  "lockupStart": 0,
  "lockupEnd": 0,
  "requiredNoticeFrequencyId": 0,
  "requiredNotice": 0,
  "firstRedemptionMonth": 0,
  "investorGateFrequencyId": 0,
  "investorGatePercent": 0,
  "investorGateCapResetFrequencyId": 0,
  "investorGateMaxCapPercent": 0,
  "investorGateUseNav": True,
  "assetGateFrequencyId": 0,
  "assetGatePercent": 0,
  "notes": "string",
  "modifiedBy": 0
}

try:
    Liquitity_creation  = client.post_request(endpoint= "/Liquidity/InsertOrUpdateLiquidityRedemptionTerms", payload=Liquitity_creation_payload)
    print(Liquitity_creation)
except Exception as e:
    print(f"Exception: {Exception}")

print("Step 6: Asset returns creation")

# Asset returns creation
assert_return_payload = {
    "rorValuationId": 0,
    "navValuationId": 0,
    "entityTypeId": 1,
    "entityId": upload_response,
    "entityName": None,
    "frequencyId": 0,
    "valuationDate": "2024-12-31T00:00:00",
    "rorValue": 6.88,
    "navValue": None,
    "estimateActual": None,
    "modifiedBy": 0,
    "modifiedByName": None,
    "modifiedDate": "2025-02-27T00:00:00"
}

try:
    returns_creation  = client.post_request(endpoint= "/AssetValuation/InsertUpdateAssetValuation", payload=assert_return_payload)
    print(returns_creation)
except Exception as e:
    print(f"Exception: {Exception}")

print("step 7: Create AUM")
create_aum_payload = {
  "assetId": upload_response,
  "entityId": 0,
  "entityTypeId": 0,
  "valuationDate": "2025-04-07T09:34:12.538Z",
  "assetValuationId": 0,
  "assetValue": 0,
  "strategyValuationId": 0,
  "strategyValue": 0,
  "firmValuationId": 0,
  "firmValue": 0,
  "modifiedBy": 0
}
try:
    Create_aum  = client.post_request(endpoint= "/AssetValuation/InsertUpdateAssetValuation", payload=create_aum_payload)
    print(Create_aum)
except Exception as e:
    print(f"Exception: {Exception}")