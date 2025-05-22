# Multi-Agent Business Intelligence System

A sophisticated AI-powered business intelligence platform that combines enterprise analytics with market research through collaborative multi-agent architecture.

## 🚀 Overview

This project implements a **Multi-Agent AI System** designed for comprehensive business intelligence analysis. The system features two specialized AI agents that can work independently or collaboratively to provide insights into enterprise operations and market dynamics.

### Key Features

- **🤖 Dual AI Agents**: Enterprise Intelligence & Market Intelligence specialists
- **🌐 Multiple Interfaces**: Streamlit web app + CLI for different use cases  
- **🔧 Advanced Toolchain**: Custom tools for data analysis and research
- **🔄 Agent Collaboration**: Agents can call each other as tools for complex queries
- **📊 Real-time Visualization**: Live agent execution tracking in web interface
- **☁️ Cloud-Ready**: Configured for Databricks deployment

## 🏗️ Architecture

### Agent System

**Enterprise Intelligence Agent**
- **Specialization**: Store performance, inventory management, business policies
- **Data Sources**: Databricks Genie spaces, business conduct policies
- **Tools**: Store performance analysis, product inventory tracking, policy lookup

**Market Intelligence Agent**  
- **Specialization**: Demographics, market research, competitive analysis
- **Data Sources**: Census API, Perplexity AI for web research
- **Tools**: Geographic demographic analysis, web-based market research

### Enhanced Collaboration Pattern
- Agents can invoke each other as tools, enabling complex multi-step analysis
- Shared context maintains conversation history and state across agent handoffs
- Intelligent routing based on query requirements

## 🛠️ Technology Stack

### Core Framework
- **Python 3.x** - Primary language
- **OpenAI Agents** - Multi-agent orchestration framework
- **Streamlit** - Web interface
- **Rich** - Enhanced CLI experience

### AI & ML Platform
- **Databricks** - ML platform and data analytics
- **MLflow** - Experiment tracking and model management
- **OpenAI API** - LLM integration via Databricks
- **Langchain** - LLM application framework

### Data & Research
- **Census API** - US demographic and geographic data
- **Perplexity AI** - Web research and reasoning
- **Unity Catalog** - Data governance and management

### Additional Libraries
- **MCP (Model Context Protocol)** - Agent communication
- **FastAPI & Gunicorn** - API deployment capabilities
- **Plotly & Matplotlib** - Data visualization
- **Pandas & NumPy** - Data processing

## 📋 Prerequisites

### Required Accounts & API Keys
1. **Databricks Workspace** with:
   - Host URL and Personal Access Token
   - Configured Genie spaces for store performance and inventory
   - Model serving endpoint
2. **Census API Key** - For demographic data
3. **Perplexity API Key** - For web research capabilities

### Environment Setup
- Python 3.8+
- Virtual environment (recommended)

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
```

2. **Create virtual environment**
```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
uv pip install -r requirements.txt
```

4. **Environment Configuration**
Create a `.env` file with the following variables:
```env
# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_personal_access_token
DATABRICKS_BASE_URL=https://your-workspace.cloud.databricks.com/serving-endpoints/your-endpoint/invocations
DATABRICKS_MODEL=your_model_name

# Genie Space IDs
GENIE_SPACE_ID=your_store_performance_space_id
GENIE_SPACE_PRODUCT_INV_ID=your_inventory_space_id

# MLflow
MLFLOW_EXPERIMENT_ID=your_experiment_id

# External APIs
CENSUS_API_KEY=your_census_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
```

## 🚀 Usage

### Web Interface (Streamlit)

Launch the interactive web application:
```bash
streamlit run app.py
```

**Features:**
- Real-time agent execution visualization
- Interactive query interface
- Agent collaboration tracking
- Tool usage monitoring

### Command Line Interface

Run single queries:
```bash
python multi_agent_cli.py --query "What are the demographics around store 110?"
```

Interactive session:
```bash
python multi_agent_cli.py --interactive
```

**CLI Features:**
- Rich formatted output with progress indicators
- Agent execution tracking
- Conversation history management
- Detailed logging

## 📁 Project Structure

```
├── app.py                    # Streamlit web application
├── multi_agent_cli.py        # Command-line interface
├── toolkit.py               # Custom tools and functions
├── requirements.txt          # Python dependencies
├── app.yaml                 # Deployment configuration
├── prompts/                 # Agent instruction files
│   ├── enterprise_intelligence_agent.txt
│   ├── market_intelligence_agent.txt
│   └── triage_agent.txt
├── .venv/                   # Virtual environment
├── .databricks/             # Databricks configuration
└── __pycache__/             # Python cache files
```

## 🔍 Available Tools

### Enterprise Tools
- **`get_store_performance_info`**: Store analytics, sales, returns, BOPIS data
- **`get_product_inventory_info`**: Product availability and inventory levels
- **`get_business_conduct_policy_info`**: Company policy and compliance information

### Market Research Tools  
- **`get_state_census_data`**: Demographics and geographic statistics
- **`do_research_and_reason`**: Web research with AI reasoning capabilities

## 💡 Example Queries

**Enterprise Analytics:**
- "What's the performance of store 110 compared to others in the region?"
- "Show me inventory levels for product XYZ across all stores"
- "What's our return policy for electronics?"

**Market Intelligence:**
- "What are the demographics around our Chicago stores?"
- "Research the competitive landscape for retail technology in 2024"
- "Analyze market trends in sustainable retail products"

**Combined Analysis:**
- "Based on store 110's location, what demographics should influence our product mix?"
- "Compare our top performing stores with the demographics of their areas"

## 🚀 Deployment

### Cloud Deployment
The project includes `app.yaml` for cloud deployment with environment variable configuration.

### Databricks Integration
- Utilizes Databricks Genie for enterprise data queries
- MLflow integration for experiment tracking
- Unity Catalog for data governance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Notes

- Ensure all API keys and credentials are properly configured
- The system requires active Databricks workspace with configured Genie spaces
- Some features may require specific Databricks permissions and workspace setup
- Always use environment variables for sensitive configuration data

## 🐛 Troubleshooting

**Common Issues:**
- **Authentication errors**: Verify Databricks token and permissions
- **Genie API timeouts**: Check workspace connectivity and space configuration  
- **Missing dependencies**: Ensure all requirements are installed in virtual environment
- **Environment variables**: Verify all required environment variables are set

For additional support, check the agent execution logs and ensure all external services are accessible.
