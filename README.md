# Industry-Specific Knowledge Agent: Healthcare Domain

An intelligent agent system that provides expert healthcare knowledge using Azure AI services, Semantic Kernel, and AutoGen.

## Project Overview

This project implements an AI-powered knowledge retrieval and reasoning system specialized for the healthcare domain. It leverages multiple agent roles to create a robust and reliable information pipeline:

1. **Data Researcher Agent**: Finds and retrieves relevant healthcare information from the knowledge base
2. **Medical Analyst Agent**: Evaluates and interprets medical information for accuracy and relevance
3. **Clinical Expert Agent**: Provides clinical context and synthesizes information for practical application

The system integrates several Azure AI services:
- Azure OpenAI Service for powerful language understanding and generation
- Azure AI Search for semantic search and retrieval of domain-specific content
- Semantic Kernel for orchestrating reasoning workflows
- AutoGen for multi-agent collaboration


I've created two visual diagrams to help you understand the Industry-Specific Knowledge Agent system:
1. **System Flow Diagram**
This diagram illustrates the information flow through your Healthcare Knowledge Agent system:

The process begins with a user query through the application interface
The Data Researcher Agent searches the Azure AI Search healthcare index
The Medical Analyst Agent evaluates and verifies the retrieved information
The Clinical Expert Agent provides medical context and synthesizes the information
The system produces a comprehensive response with proper citations and confidence levels

The diagram shows how the three specialized agents work together in sequence, with each agent building on the work of the previous one. It also highlights how Azure services integrate with the agent system.

2. **Technical Architecture Diagram**
This second diagram provides a more technical view of the system's architecture:

**Top Layer:** Azure AI Services (OpenAI, AI Search, Form Recognizer)
**Middle Layer:** Agent Frameworks (AutoGen and Semantic Kernel)
**Bottom Layer:** The Healthcare Agent System implementation

You can see how each agent has a specific role and corresponding function:

Healthcare Data Researcher: query_healthcare_knowledge()
Medical Analyst: verify_medical_information()
Clinical Expert: synthesize_healthcare_information()

The HealthcareAgentSystem class coordinates all these components.

## Features

- **Specialized Healthcare Knowledge Retrieval**: Semantic search optimized for medical information
- **Multi-Agent Verification**: Information is analyzed and verified through multiple expert perspectives
- **Clinical Contextualization**: Results incorporate real-world clinical application context
- **Source Citation**: All information includes proper attribution to source material
- **Reliability Assessment**: Confidence scores and evidence quality evaluation

## System Architecture

![System Architecture](architecture_diagram.png)

The system follows this workflow:
1. User query is processed by the main application
2. Data Researcher Agent searches the knowledge base using Azure AI Search
3. Medical Analyst Agent evaluates the retrieved information for accuracy and evidence quality
4. Clinical Expert Agent provides practical context and synthesizes the final response
5. The system returns a comprehensive, multi-perspective answer with sources

## Setup Instructions

### Prerequisites

- Azure subscription with:
  - Azure OpenAI Service access
  - Azure AI Search service
- Python 3.8+
- pip package manager

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/healthcare-knowledge-agent.git
   cd healthcare-knowledge-agent
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure your environment variables by creating a `.env` file:
   ```
   # Azure OpenAI Service configuration
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=https://your-azure-openai-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
   AZURE_OPENAI_API_VERSION=2023-07-01-preview

   # Azure AI Search configuration
   AZURE_SEARCH_SERVICE_ENDPOINT=https://your-search-service.search.windows.net
   AZURE_SEARCH_INDEX_NAME=healthcare-knowledge-index
   AZURE_SEARCH_API_KEY=your_azure_search_api_key
   ```

### Setup Azure AI Search Index

1. Create the healthcare knowledge index:
   ```
   python search_index_setup.py
   ```

2. Upload sample healthcare documents (for testing):
   ```
   python sample_data_uploader.py
   ```

## Usage

### Interactive Mode

Run the application in interactive mode:
```
python app_runner.py --interactive
```

### Single Query Mode

Process a specific healthcare query:
```
python app_runner.py --query "What are the latest guidelines for managing type 2 diabetes in elderly patients with kidney disease?"
```

### Using in Your Own Applications

Import the `HealthcareAgentSystem` class from `industry_specific_knowledge_agent.py` to use in your own applications:

```python
from industry_specific_knowledge_agent import (
    HealthcareAgentSystem,
    data_researcher,
    medical_analyst,
    clinical_expert,
    user_proxy
)

# Create the healthcare agent system
healthcare_system = HealthcareAgentSystem(
    data_researcher=data_researcher,
    medical_analyst=medical_analyst,
    clinical_expert=clinical_expert,
    user_proxy=user_proxy
)

# Process a healthcare query
query = "What are the interactions between ACE inhibitors and potassium supplements?"
response = healthcare_system.process_healthcare_query(query)

# Use the response data
print(f"Clinical Context: {response['clinical_context']}")
```

## Customization

### Adding Your Own Medical Content

To add your own medical content to the knowledge base:

1. Create JSON or CSV files with your medical content
2. Modify the `sample_data_uploader.py` script to load your content
3. Run the uploader script to add content to the Azure AI Search index

### Extending to Different Medical Specialties

To focus on a specific medical specialty:

1. Update the system messages in `industry_specific_knowledge_agent.py` to emphasize your specialty
2. Add specialty-specific content to the knowledge base
3. Consider creating additional agent roles that represent specialized expertise

## Azure Free Tier Considerations

This system is designed to work within Azure free tier limitations:

- Uses efficient prompting to minimize token usage
- Implements caching to reduce redundant API calls
- Limits the number of documents retrieved from Azure AI Search
- Uses smaller models where possible to reduce costs

## Requirements

See `requirements.txt` for the full list of dependencies:

```
azure-search-documents>=11.4.0
azure-identity>=1.12.0
python-dotenv>=0.21.0
openai>=0.27.0
semantic-kernel>=0.3.0
autogen>=0.2.0
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft AI Agents Hackathon for the inspiration
- Azure AI team for the excellent services and documentation
- Healthcare organizations that publish open medical guidelines