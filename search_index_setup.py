import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    VectorSearch,
    HnswParameters,
    VectorSearchAlgorithmConfiguration
)

# Load environment variables
load_dotenv()

# Azure AI Search configuration
AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Initialize the search index client
search_index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

# Define the healthcare knowledge index with semantic search capabilities
def create_healthcare_knowledge_index():
    """Create or update the healthcare knowledge index."""
    
    # Define fields for the index
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String, sortable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SimpleField(name="source_url", type=SearchFieldDataType.String),
        SimpleField(name="publish_date", type=SearchFieldDataType.DateTimeOffset, sortable=True, filterable=True),
        SimpleField(name="author", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="keywords", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="medical_specialties", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        SearchField(
            name="content_vector", 
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=1536,  # Adjust based on your embedding model
            vector_search_profile_name="healthcare-vector-config"
        )
    ]
    
    # Define semantic configuration for healthcare content
    # This enables semantic search capabilities
    semantic_config = SemanticConfiguration(
        name="healthcare-config",
        prioritized_fields=SemanticField(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="content")],
            keyword_fields=[SemanticField(field_name="keywords"), SemanticField(field_name="medical_specialties")]
        )
    )
    
    # Configure vector search for the index
    vector_search = VectorSearch(
        algorithms=[
            VectorSearchAlgorithmConfiguration(
                name="healthcare-vector-config",
                kind="hnsw",
                hnsw_parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric="cosine"
                )
            )
        ]
    )
    
    # Create the semantic settings with the configuration
    semantic_settings = SemanticSettings(configurations=[semantic_config])
    
    # Create the index definition
    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=fields,
        semantic_settings=semantic_settings,
        vector_search=vector_search
    )
    
    # Create or update the index
    result = search_index_client.create_or_update_index(index)
    return result

def main():
    """Main function to create or update the healthcare knowledge index."""
    try:
        print(f"Creating or updating '{AZURE_SEARCH_INDEX_NAME}' index...")
        result = create_healthcare_knowledge_index()
        print(f"Index '{result.name}' created or updated successfully.")
    except Exception as e:
        print(f"Error creating or updating index: {e}")

if __name__ == "__main__":
    main()