import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, register_function
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
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

# Initialize Semantic Kernel
kernel = Kernel()
kernel.add_chat_service(
    "chat_completion", 
    AzureChatCompletion(
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
)

# Initialize Azure AI Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

class HealthcareKnowledgeRetriever:
    """A class to retrieve healthcare-specific knowledge from Azure AI Search."""
    
    def __init__(self, search_client):
        self.search_client = search_client
    
    def search(self, query: str, top: int = 5) -> List[Dict[str, Any]]:
        """Search the healthcare knowledge base using semantic search."""
        try:
            results = self.search_client.search(
                search_text=query,
                query_type=QueryType.SEMANTIC,
                query_language="en-us",
                semantic_configuration_name="healthcare-config",
                top=top,
                query_caption="extractive",
                query_answer="extractive",
                include_total_count=True
            )
            
            search_results = []
            for result in results:
                search_results.append({
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "source": result.get("source_url", ""),
                    "score": result["@search.score"],
                    "captions": result.get("@search.captions", []),
                    "highlights": result.get("@search.highlights", {})
                })
            
            return search_results
        except Exception as e:
            logger.error(f"Error searching Azure AI Search: {e}")
            return []

# Initialize the healthcare knowledge retriever
knowledge_retriever = HealthcareKnowledgeRetriever(search_client)

# Define agent personalities and specialties
SYSTEM_MESSAGES = {
    "data_researcher": """You are a healthcare data researcher agent specialized in searching and retrieving relevant healthcare information. 
Your job is to find accurate information from trusted medical sources.
Always provide citations and sources for your information.
You have expertise in medical terminology and can understand complex healthcare queries.
You primarily focus on finding and presenting factual information, not interpretation.
""",
    
    "medical_analyst": """You are a healthcare analyst agent with expertise in interpreting and analyzing medical information.
You can synthesize complex medical data into clear insights.
You understand medical research methodologies and clinical guidelines.
You evaluate the quality of medical evidence using frameworks like GRADE.
You analyze trends, patterns, and inconsistencies in healthcare data.
Always make it clear when you are expressing an analytical judgment versus stating facts.
""",
    
    "clinical_expert": """You are a clinical domain expert agent with deep knowledge of medical practice.
You understand standard of care and clinical decision-making processes.
You can explain complex medical concepts in simplified terms when needed.
You contextualize information within clinical practice settings.
You know the limitations of your knowledge and when to advise seeking professional medical consultation.
You never provide specific medical advice for individual cases - always clarify that information is educational only.
"""
}

# Function to query the healthcare knowledge base
@register_function
def query_healthcare_knowledge(query: str, top_results: int = 5) -> List[Dict[str, Any]]:
    """
    Query the healthcare knowledge base for relevant information.
    
    Args:
        query: The healthcare query to search for.
        top_results: The number of top results to return.
        
    Returns:
        A list of search results with content, sources, and relevance scores.
    """
    results = knowledge_retriever.search(query, top=top_results)
    return results

# Function to analyze the veracity of healthcare information
@register_function
def verify_medical_information(claim: str, source_text: str = None) -> Dict[str, Any]:
    """
    Analyze a medical claim against known information to assess its reliability.
    
    Args:
        claim: The medical claim to verify.
        source_text: Optional supporting text for the claim.
        
    Returns:
        Analysis of the claim's reliability and confidence level.
    """
    # In a real implementation, this would use semantic kernel to evaluate
    # For this example, we'll simulate with a simple prompt
    
    prompt = f"""
    Please analyze the following medical claim for accuracy:
    
    CLAIM: {claim}
    
    {f'SUPPORTING TEXT: {source_text}' if source_text else ''}
    
    Evaluate the claim for:
    1. Consistency with established medical knowledge
    2. Quality of supporting evidence (if available)
    3. Potential biases or limitations
    4. Overall reliability assessment
    
    Provide a confidence score from 1-5 where:
    1 = Highly dubious
    3 = Plausible but uncertain
    5 = Well-supported by evidence
    """
    
    function = kernel.create_semantic_function(prompt)
    result = function.invoke()
    
    # Parse the results (in a real implementation, would be more structured)
    return {
        "analysis": str(result),
        "confidence_score": 3,  # Placeholder - would be extracted from the result
        "verified_date": "2025-04-30"
    }

# Function to synthesize information from multiple sources
@register_function
def synthesize_healthcare_information(search_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Synthesize information from multiple healthcare sources into a coherent response.
    
    Args:
        search_results: List of search results containing healthcare information.
        query: The original query that prompted the search.
        
    Returns:
        A synthesized response with citations and confidence level.
    """
    # Prepare the content from search results
    sources_content = ""
    for idx, result in enumerate(search_results):
        sources_content += f"\nSource {idx+1}: {result['content']}\nFrom: {result['source']}\n"
    
    prompt = f"""
    Based on the following healthcare information sources, synthesize a comprehensive response to the query: "{query}"
    
    {sources_content}
    
    Your synthesis should:
    1. Be accurate and reflect the consolidated knowledge from all sources
    2. Highlight areas of consensus and any contradictions
    3. Include appropriate citations to the source materials
    4. Use clear, professional medical terminology while remaining accessible
    5. Indicate confidence levels for different parts of the synthesis
    6. Note any important gaps in the available information
    
    Format your response with clear sections, citations in [brackets], and a brief summary at the end.
    """
    
    function = kernel.create_semantic_function(prompt)
    result = function.invoke()
    
    # Create a structured response
    return {
        "synthesis": str(result),
        "sources": [r["source"] for r in search_results],
        "query": query,
        "generated_at": "2025-04-30"
    }

# Create AutoGen agents
data_researcher = AssistantAgent(
    name="Healthcare_Data_Researcher",
    system_message=SYSTEM_MESSAGES["data_researcher"],
    llm_config={
        "config_list": [{
            "model": AZURE_OPENAI_DEPLOYMENT_NAME,
            "api_key": AZURE_OPENAI_API_KEY,
            "api_base": AZURE_OPENAI_ENDPOINT,
            "api_type": "azure",
            "api_version": AZURE_OPENAI_API_VERSION
        }]
    }
)

medical_analyst = AssistantAgent(
    name="Healthcare_Analyst",
    system_message=SYSTEM_MESSAGES["medical_analyst"],
    llm_config={
        "config_list": [{
            "model": AZURE_OPENAI_DEPLOYMENT_NAME,
            "api_key": AZURE_OPENAI_API_KEY,
            "api_base": AZURE_OPENAI_ENDPOINT,
            "api_type": "azure",
            "api_version": AZURE_OPENAI_API_VERSION
        }]
    }
)

clinical_expert = AssistantAgent(
    name="Clinical_Expert",
    system_message=SYSTEM_MESSAGES["clinical_expert"],
    llm_config={
        "config_list": [{
            "model": AZURE_OPENAI_DEPLOYMENT_NAME,
            "api_key": AZURE_OPENAI_API_KEY,
            "api_base": AZURE_OPENAI_ENDPOINT,
            "api_type": "azure",
            "api_version": AZURE_OPENAI_API_VERSION
        }]
    }
)

# Create a user proxy agent that can execute functions
user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="TERMINATE",
    function_map={
        "query_healthcare_knowledge": query_healthcare_knowledge,
        "verify_medical_information": verify_medical_information,
        "synthesize_healthcare_information": synthesize_healthcare_information
    }
)

# Register the available functions with the agents
for agent in [data_researcher, medical_analyst, clinical_expert]:
    agent.register_function(
        function_map={
            "query_healthcare_knowledge": query_healthcare_knowledge,
            "verify_medical_information": verify_medical_information,
            "synthesize_healthcare_information": synthesize_healthcare_information
        }
    )

class HealthcareAgentSystem:
    """Orchestrates the healthcare agent system workflow."""
    
    def __init__(self, data_researcher, medical_analyst, clinical_expert, user_proxy):
        self.data_researcher = data_researcher
        self.medical_analyst = medical_analyst
        self.clinical_expert = clinical_expert
        self.user_proxy = user_proxy
    
    def process_healthcare_query(self, query: str) -> Dict[str, Any]:
        """Process a healthcare query using the agent system."""
        logger.info(f"Processing healthcare query: {query}")
        
        # Step 1: Data Researcher retrieves relevant information
        logger.info("Step 1: Data Researcher retrieving information")
        data_retrieval_prompt = f"I need you to find relevant healthcare information about: {query}. Please use the query_healthcare_knowledge function to search for information and return the most relevant results."
        
        self.user_proxy.initiate_chat(
            self.data_researcher,
            message=data_retrieval_prompt
        )
        
        # Extract search results from the chat (in a real implementation, this would be more robust)
        data_researcher_response = self.user_proxy.chat_messages[self.data_researcher][-1]["content"]
        search_results = []
        
        try:
            # In practice, we would parse the actual function call results
            # This is a simplified approach for demonstration
            if "I've found the following information" in data_researcher_response:
                # Simulate retrieving the search results that would have been returned
                search_results = query_healthcare_knowledge(query)
            else:
                search_results = []
        except Exception as e:
            logger.error(f"Error parsing search results: {e}")
            search_results = []
        
        # Step 2: Medical Analyst evaluates and interprets the information
        logger.info("Step 2: Medical Analyst analyzing information")
        if search_results:
            analysis_prompt = f"""
            I need you to analyze these healthcare information results about: {query}
            
            Here are the search results:
            {json.dumps(search_results, indent=2)}
            
            Please:
            1. Evaluate the quality and reliability of this information
            2. Identify any contradictions or inconsistencies
            3. Note the strength of evidence for key claims
            4. Use the verify_medical_information function on any claims that need validation
            """
        else:
            analysis_prompt = f"I couldn't find specific information about '{query}'. Please provide your analysis based on general medical knowledge and clearly indicate limitations due to lack of specific search results."
        
        self.user_proxy.initiate_chat(
            self.medical_analyst, 
            message=analysis_prompt
        )
        
        # Step 3: Clinical Expert contextualizes and provides domain expertise
        logger.info("Step 3: Clinical Expert providing expert context")
        expert_prompt = f"""
        I need your clinical expertise on this healthcare topic: {query}
        
        The data researcher found this information:
        {data_researcher_response}
        
        The medical analyst provided this analysis:
        {self.user_proxy.chat_messages[self.medical_analyst][-1]["content"]}
        
        Please:
        1. Provide clinical context for this information
        2. Explain how this would be applied in healthcare settings
        3. Note any practical considerations for healthcare professionals
        4. Clarify limitations and when further consultation would be necessary
        5. Use the synthesize_healthcare_information function to create a final comprehensive response
        """
        
        self.user_proxy.initiate_chat(
            self.clinical_expert,
            message=expert_prompt
        )
        
        # Compile the final response with contributions from all agents
        final_response = {
            "query": query,
            "data_retrieval": data_researcher_response,
            "analysis": self.user_proxy.chat_messages[self.medical_analyst][-1]["content"],
            "clinical_context": self.user_proxy.chat_messages[self.clinical_expert][-1]["content"],
            "sources": [r.get("source", "") for r in search_results] if search_results else [],
            "timestamp": "2025-04-30"
        }
        
        logger.info("Healthcare query processing complete")
        return final_response

# Usage example
if __name__ == "__main__":
    # Create the healthcare agent system
    healthcare_system = HealthcareAgentSystem(
        data_researcher=data_researcher,
        medical_analyst=medical_analyst,
        clinical_expert=clinical_expert,
        user_proxy=user_proxy
    )
    
    # Process a sample healthcare query
    sample_query = "What are the latest guidelines for managing type 2 diabetes in elderly patients with kidney disease?"
    response = healthcare_system.process_healthcare_query(sample_query)
    
    # Print the response in a formatted way
    print("\n" + "="*50)
    print("HEALTHCARE KNOWLEDGE AGENT RESPONSE")
    print("="*50)
    print(f"Query: {response['query']}")
    print("\nData Retrieval Summary:")
    print("-"*40)
    print(response['data_retrieval'][:500] + "..." if len(response['data_retrieval']) > 500 else response['data_retrieval'])
    print("\nMedical Analysis:")
    print("-"*40)
    print(response['analysis'][:500] + "..." if len(response['analysis']) > 500 else response['analysis'])
    print("\nClinical Context:")
    print("-"*40)
    print(response['clinical_context'][:500] + "..." if len(response['clinical_context']) > 500 else response['clinical_context'])
    print("\nSources:")
    print("-"*40)
    for idx, source in enumerate(response['sources']):
        print(f"{idx+1}. {source}")
    print("="*50)