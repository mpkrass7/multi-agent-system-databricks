# Multi-Agent Business Intelligence System

AI-powered business intelligence platform with collaborative multi-agent architecture for enterprise analytics and market research.

## Overview

**Dual AI Agents** that work independently or collaboratively:
- **Enterprise Intelligence**: Store performance, inventory, business policies
- **Market Intelligence**: Demographics, market research, competitive analysis

**Interfaces**: Streamlit web app + CLI with real-time agent execution tracking

## Architecture

### Agents & Tools

**Enterprise Agent**
- Store performance analysis (Databricks Genie)
- Product inventory tracking (Databricks Genie)
- Business policy lookup (UC function)

**Market Agent**
- Geographic demographics (Census API)
- Web research (Perplexity AI)

**Collaboration**: Agents can call each other as tools for complex multi-step analysis.

## Tech Stack

- **Python 3.10+**, **OpenAI Agents**, **Streamlit**
- **Databricks** (ML platform), **MLflow** (tracing)
- **Census API** (demographics), **Perplexity AI** (research)

## Installation

1. **Setup**
```bash
git clone <repository-url>
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
```

2. **Environment** - Create `.env`:
```env
# Databricks
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_personal_access_token
DATABRICKS_MODEL=your_model_name
GENIE_SPACE_ID=your_store_space_id
GENIE_SPACE_PRODUCT_INV_ID=your_inventory_space_id

# APIs
CENSUS_API_KEY=your_census_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
```

## Usage

### Web Interface
```bash
streamlit run app.py
```

### CLI
```bash
# Single query
python multi_agent_cli.py --query "What are the demographics around store 110?"

# Interactive mode
python multi_agent_cli.py --interactive
```

## Example Queries

**Enterprise**: "Performance of store 110 vs region", "Inventory levels for product XYZ"
**Market**: "Demographics around Chicago stores", "Retail technology competitive landscape"
**Combined**: "Store 110's location demographics for product mix optimization"

## Project Structure

```
├── app.py                    # Streamlit web app
├── multi_agent_cli.py        # CLI interface
├── toolkit.py               # Custom tools
├── prompts/                 # Agent instructions
└── requirements.txt         # Dependencies
```

## Prerequisites

- **Databricks workspace** with Genie spaces and model serving endpoint
- **Census API key** for demographic data
- **Perplexity API key** for web research

## Troubleshooting

- **Auth errors**: Verify Databricks token/permissions
- **Timeouts**: Check workspace connectivity
- **Missing deps**: Ensure virtual environment setup
- **Config**: Verify all environment variables set
