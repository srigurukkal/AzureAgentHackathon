import os
import argparse
import logging
from typing import List, Dict, Any

from industry_specific_knowledge_agent import (
    HealthcareAgentSystem,
    data_researcher,
    medical_analyst,
    clinical_expert,
    user_proxy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("healthcare_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def format_agent_response(response: Dict[str, Any]) -> str:
    """Format the agent response for display in the console."""
    output = "\n" + "="*80 + "\n"
    output += "HEALTHCARE KNOWLEDGE AGENT RESPONSE\n"
    output += "="*80 + "\n\n"
    
    output += f"Query: {response['query']}\n\n"
    
    output += "DATA RETRIEVAL SUMMARY\n"
    output += "-"*80 + "\n"
    output += response['data_retrieval'] + "\n\n"
    
    output += "MEDICAL ANALYSIS\n"
    output += "-"*80 + "\n"
    output += response['analysis'] + "\n\n"
    
    output += "CLINICAL CONTEXT\n"
    output += "-"*80 + "\n"
    output += response['clinical_context'] + "\n\n"
    
    output += "SOURCES\n"
    output += "-"*80 + "\n"
    for idx, source in enumerate(response['sources']):
        output += f"{idx+1}. {source}\n"
    
    output += "\n" + "="*80 + "\n"
    return output

def process_query(query: str) -> None:
    """Process a healthcare query and display the results."""
    try:
        logger.info(f"Processing query: {query}")
        
        # Create the healthcare agent system
        healthcare_system = HealthcareAgentSystem(
            data_researcher=data_researcher,
            medical_analyst=medical_analyst,
            clinical_expert=clinical_expert,
            user_proxy=user_proxy
        )
        
        # Process the healthcare query
        response = healthcare_system.process_healthcare_query(query)
        
        # Format and display the response
        formatted_response = format_agent_response(response)
        print(formatted_response)
        
        # Save the response to a file
        query_filename = query.lower().replace(" ", "_")[:30]
        with open(f"response_{query_filename}.txt", "w") as f:
            f.write(formatted_response)
        
        logger.info(f"Query processing completed. Results saved to response_{query_filename}.txt")
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        print(f"Error processing query: {e}")

def interactive_mode() -> None:
    """Run the healthcare agent system in interactive mode."""
    print("\n" + "="*80)
    print("HEALTHCARE KNOWLEDGE AGENT SYSTEM - INTERACTIVE MODE")
    print("="*80)
    print("Enter your healthcare queries. Type 'exit' or 'quit' to end the session.")
    print("="*80 + "\n")
    
    while True:
        query = input("\nEnter your healthcare query: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting interactive mode.")
            break
        
        if query.strip():
            process_query(query)

def main():
    """Main function to run the healthcare agent system."""
    parser = argparse.ArgumentParser(description="Healthcare Knowledge Agent System")
    parser.add_argument("--query", "-q", type=str, help="Healthcare query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.query:
        process_query(args.query)
    else:
        # Default to interactive mode if no args provided
        interactive_mode()

if __name__ == "__main__":
    main()