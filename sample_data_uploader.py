import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# Load environment variables
load_dotenv()

# Azure OpenAI Service configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")

# Azure AI Search configuration
AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

# Set up Azure OpenAI client
openai.api_type = "azure"
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_key = AZURE_OPENAI_API_KEY

# Initialize Azure AI Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

def generate_embeddings(text):
    """
    Generate embeddings for the given text using Azure OpenAI.
    In a real implementation, you would use a deployment of text-embedding-ada-002 or similar.
    """
    # Note: This is a placeholder. In a real implementation, 
    # you would use the actual embedding model endpoint
    try:
        # Replace with your actual embedding model deployment
        response = openai.Embedding.create(
            input=text,
            engine="text-embedding-ada-002"  # Use your actual deployment name
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Return a placeholder embedding (this is just for demonstration)
        return [0.0] * 1536

# Sample healthcare documents for testing
sample_healthcare_docs = [
    {
        "title": "Type 2 Diabetes Management in Elderly Patients",
        "content": """
        Management of type 2 diabetes in elderly patients requires careful consideration of several factors including comorbidities, functional status, life expectancy, and risk of hypoglycemia.

        Current guidelines recommend:
        1. Individualized HbA1c targets based on patient characteristics, typically ranging from <7.5% to <8.5% for older adults with multiple comorbidities or limited life expectancy.
        2. Simplified medication regimens to improve adherence and reduce risk of adverse effects.
        3. Regular screening for diabetes complications with special attention to cardiovascular risk factors.
        4. Periodic assessment of cognitive function and ability to self-manage.
        5. Special consideration for elderly patients with kidney disease, including medication adjustments for decreased renal function.

        For elderly patients with kidney disease, metformin dosage should be reduced for eGFR 30-45 mL/min/1.73m² and avoided when eGFR < 30 mL/min/1.73m². SGLT2 inhibitors show promise for renoprotection but require careful monitoring. GLP-1 receptor agonists are generally preferred as they provide cardiovascular benefits with minimal hypoglycemia risk.
        """,
        "source_url": "https://example.com/diabetes-elderly-guidelines-2024",
        "publish_date": "2024-03-15T00:00:00Z",
        "author": "American Diabetes Association",
        "keywords": ["diabetes", "elderly care", "kidney disease", "medication management", "comorbidities"],
        "content_type": "clinical guideline",
        "medical_specialties": ["endocrinology", "geriatrics", "nephrology"]
    },
    {
        "title": "Pharmacological Management of Hypertension in Older Adults",
        "content": """
        Hypertension management in older adults (≥65 years) follows an evidence-based approach with consideration for physiological changes of aging and multiple comorbidities.

        Key recommendations from the most recent guidelines include:
        1. Target blood pressure of <130/80 mmHg may be appropriate for ambulatory community-dwelling older adults with good functional status.
        2. For adults aged >80 or frail elderly, a more conservative target of <150/90 mmHg is recommended to avoid orthostatic hypotension and falls.
        3. First-line medications include thiazide diuretics, angiotensin-converting enzyme inhibitors (ACEIs), angiotensin II receptor blockers (ARBs), and calcium channel blockers (CCBs).
        4. Start with lower doses and titrate more gradually than in younger patients ("start low, go slow").
        5. Regular monitoring for adverse effects including electrolyte disturbances, acute kidney injury, and orthostatic hypotension.

        Special considerations for elderly patients with diabetes and hypertension include preferential use of ACEIs or ARBs for their renoprotective effects. However, combination of ACEIs and ARBs should be avoided due to increased risk of hyperkalemia, especially in patients with reduced kidney function.
        """,
        "source_url": "https://example.com/hypertension-elderly-management-2024",
        "publish_date": "2024-01-20T00:00:00Z",
        "author": "American Heart Association",
        "keywords": ["hypertension", "elderly", "blood pressure", "antihypertensive medication", "comorbidities"],
        "content_type": "clinical guideline",
        "medical_specialties": ["cardiology", "geriatrics", "nephrology"]
    },
    {
        "title": "Polypharmacy and Medication Management in Geriatric Patients",
        "content": """
        Polypharmacy, commonly defined as the use of five or more medications, is prevalent in older adults and associated with increased risk of adverse drug events, drug-drug interactions, and medication non-adherence.

        Current approaches to managing polypharmacy include:
        1. Regular medication reviews using validated tools such as the STOPP/START criteria or Beers Criteria.
        2. Deprescribing protocols to systematically withdraw potentially inappropriate medications.
        3. Medication reconciliation during care transitions to prevent errors.
        4. Simplification of medication regimens through once-daily dosing when possible.
        5. Use of electronic medication management systems for patients with cognitive impairment.

        For patients with chronic kidney disease, additional considerations include:
        - Adjustment of medication dosages based on estimated glomerular filtration rate (eGFR).
        - Avoidance of nephrotoxic medications when possible.
        - More frequent monitoring of drug levels for medications with narrow therapeutic indices.
        - Special attention to drugs primarily excreted by the kidneys, including many antibiotics, diabetic medications, and cardiovascular drugs.

        The concept of "appropriate polypharmacy" recognizes that multiple medications may be necessary for patients with multimorbidity, but emphasizes evidence-based, patient-centered prescribing that considers quality of life and patient preferences.
        """,
        "source_url": "https://example.com/polypharmacy-management-elderly-2024",
        "publish_date": "2024-02-10T00:00:00Z",
        "author": "American Geriatrics Society",
        "keywords": ["polypharmacy", "medication management", "deprescribing", "elderly", "drug interactions"],
        "content_type": "clinical review",
        "medical_specialties": ["geriatrics", "clinical pharmacology", "primary care"]
    }
]

def upload_sample_documents():
    """Upload sample healthcare documents to Azure AI Search index."""
    try:
        # Prepare documents with IDs and embeddings
        documents_to_upload = []
        
        for doc in sample_healthcare_docs:
            # Generate a unique ID
            doc_id = str(uuid.uuid4())
            
            # Generate vector embeddings for content
            content_for_embedding = doc["title"] + " " + doc["content"]
            content_vector = generate_embeddings(content_for_embedding)
            
            # Create the document to upload
            document = {
                "id": doc_id,
                "title": doc["title"],
                "content": doc["content"],
                "source_url": doc["source_url"],
                "publish_date": doc["publish_date"],
                "author": doc["author"],
                "keywords": doc["keywords"],
                "content_type": doc["content_type"],
                "medical_specialties": doc["medical_specialties"],
                "content_vector": content_vector
            }
            
            documents_to_upload.append(document)
        
        # Upload documents in batches
        if documents_to_upload:
            result = search_client.upload_documents(documents=documents_to_upload)
            print(f"Uploaded {len(result)} documents.")
            print(f"Succeeded: {sum([1 for r in result if r.succeeded])}")
            print(f"Failed: {sum([1 for r in result if not r.succeeded])}")
            
            for idx, r in enumerate(result):
                if not r.succeeded:
                    print(f"Document #{idx} failed: {r.error_message}")
    
    except Exception as e:
        print(f"Error uploading documents: {e}")

if __name__ == "__main__":
    print("Uploading sample healthcare documents to Azure AI Search...")
    upload_sample_documents()
    print("Upload process completed.")