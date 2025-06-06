import asyncio
import logging
import os
# from threading import Thread
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlflow
import streamlit as st
from agents import (Agent, OpenAIChatCompletionsModel, RunContextWrapper,
                    Runner, handoff, set_tracing_disabled)
from dotenv import load_dotenv
from mlflow.tracing.destination import Databricks
from openai import AsyncOpenAI

from toolkit import (do_research_and_reason, get_business_conduct_policy_info,
                     get_product_inventory_info, get_state_census_data,
                     get_store_performance_info)

# Load environment variables
load_dotenv("/Users/sathish.gangichetty/Documents/openai-agents/apps/.env-local")

# Initialize environment variables
MODEL_NAME = os.getenv("DATABRICKS_MODEL") or ""
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
BASE_URL = os.getenv("DATABRICKS_BASE_URL") or ""
API_KEY = os.getenv("DATABRICKS_TOKEN") or ""
# API_KEY = st.context.headers.get('X-Forwarded-Access-Token')

MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID") or ""
set_tracing_disabled(True)
try:
    if MLFLOW_EXPERIMENT_ID:
        mlflow.set_registry_uri("databricks")
        mlflow.tracing.set_destination(Databricks(experiment_id=MLFLOW_EXPERIMENT_ID))
        mlflow.openai.autolog()
        logging.info("MLflow logging enabled")
    else:
        logging.info("MLflow logging disabled - MLFLOW_EXPERIMENT_ID not set")
except Exception as e:
    logging.warning(f"Failed to initialize MLflow logging: {str(e)}")

# Initialize clients
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
# w = WorkspaceClient(
#     host=os.getenv("DATABRICKS_HOST"), token=os.getenv("DATABRICKS_TOKEN"),
#     auth_type="pat",
# )


# Define a shared context class to pass data between agents
@dataclass
class SharedAgentContext:
    store_location: Optional[str] = None
    store_id: Optional[str] = None
    demographic_data: Optional[Dict] = None
    state_code: Optional[str] = None
    current_agent: Optional[str] = None
    current_tool: Optional[str] = None
    conversation_history: List = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def get_formatted_history(self):
        """Format conversation history for consumption by agents"""
        if not self.conversation_history:
            return "No conversation history available."

        formatted_history = []
        for msg in self.conversation_history:
            formatted_history.append(f"{msg['role']}: {msg['content']}")

        return "\n\n".join(formatted_history)


# Custom hooks for Streamlit visualization
class StreamlitAgentHooks:
    def __init__(self):
        self.start_time = None
        self.agent_status = st.empty()
        self.tool_status = st.empty()

    async def on_agent_start(
        self, context: RunContextWrapper[SharedAgentContext], agent
    ):
        self.start_time = time.time()
        agent_name = agent.name
        context.context.current_agent = agent_name

        with self.agent_status.container():
            st.markdown(
                f"""
            <div class="agent-status agent-active">
                <div class="agent-icon">🤖</div>
                <div class="agent-name">{agent_name}</div>
                <div class="status-indicator">Active</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    async def on_agent_end(
        self, context: RunContextWrapper[SharedAgentContext], agent, output
    ):
        agent_name = agent.name
        duration = time.time() - self.start_time

        # Add to conversation history
        context.context.add_message(f"{agent_name}", output)

        with self.agent_status.container():
            st.markdown(
                f"""
            <div class="agent-status agent-complete">
                <div class="agent-icon">✅</div>
                <div class="agent-name">{agent_name}</div>
                <div class="status-indicator">Completed in {duration:.2f}s</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    async def on_tool_start(
        self, context: RunContextWrapper[SharedAgentContext], agent, tool
    ):
        tool_name = tool.name
        context.context.current_tool = tool_name

        with self.tool_status.container():
            st.markdown(
                f"""
            <div class="tool-status tool-active">
                <div class="tool-icon">🔧</div>
                <div class="tool-name">{tool_name}</div>
                <div class="status-indicator">Running...</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    async def on_tool_end(
        self, context: RunContextWrapper[SharedAgentContext], agent, tool, result
    ):
        tool_name = tool.name

        # Record tool usage in conversation history
        context.context.add_message(f"Tool ({tool_name})", str(result))

        with self.tool_status.container():
            st.markdown(
                f"""
            <div class="tool-status tool-complete">
                <div class="tool-icon">✓</div>
                <div class="tool-name">{tool_name}</div>
                <div class="status-indicator">Completed</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    async def on_handoff(
        self, context: RunContextWrapper[SharedAgentContext], from_agent, to_agent
    ):
        from_name = from_agent.name
        to_name = to_agent.name

        # Record handoff in conversation history
        context.context.add_message("System", f"Handoff from {from_name} to {to_name}")

        with self.agent_status.container():
            st.markdown(
                f"""
            <div class="handoff-status">
                <div class="from-agent">{from_name}</div>
                <div class="handoff-icon">↪️</div>
                <div class="to-agent">{to_name}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# Helper function to load prompts from files
def load_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Load agent prompts
enterprise_intelligence_prompt = load_prompt(
    "prompts/enterprise_intelligence_agent.txt"
)
market_intelligence_prompt = load_prompt("prompts/market_intelligence_agent.txt")
triage_agent_prompt = load_prompt("prompts/triage_agent.txt")

# Enhance the agent prompts to handle the tools-for-agents pattern
enhanced_enterprise_prompt = (
    enterprise_intelligence_prompt
    + """

## Additional Capabilities
You now have the Market Intelligence Agent available as a tool. When a query requires demographic or market research data:
1. First determine the relevant location information using your store performance tools
2. Then use the get_market_intelligence tool to obtain demographic information for that location
3. Combine both sources of information to provide a complete response
"""
)

enhanced_market_prompt = (
    market_intelligence_prompt
    + """

## Additional Capabilities
You now have the Enterprise Intelligence Agent available as a tool. When a query requires store-specific information:
1. Use the get_enterprise_data tool to first obtain store location or performance information
2. Then use your demographic and market research tools to analyze that location
3. Combine both sources of information to provide a complete response

For example, if asked "Based on where store 110 is located, what are the demographics of the area?":
1. First use get_enterprise_data to find out where store 110 is located
2. Then analyze the demographics of that location using your tools
"""
)

# Initial Enterprise Intelligence Agent
enterprise_intelligence_agent = Agent(
    name="Enterprise Intelligence Agent",
    handoff_description="Specialist in enterprise analytics pertaining to the store performance, sales, store location, returns, BOPIS(buy online pick up in store), policy, inventory etc.",
    instructions=enterprise_intelligence_prompt,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[
        get_business_conduct_policy_info,
        get_store_performance_info,
        get_product_inventory_info,
    ],
)

# Initial Market Intelligence Agent
market_intelligence_agent = Agent(
    name="Market Intelligence Agent",
    handoff_description="Specialist in market research pertaining to general questions about the market, industry, news, competitors, demographics, etc.",
    instructions=market_intelligence_prompt,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[
        get_state_census_data,
        do_research_and_reason,
    ],
)


def on_enterprise_intelligence_handoff(ctx: RunContextWrapper[SharedAgentContext]):
    st.sidebar.success("🔄 Handing off to Enterprise Intelligence Agent")


def on_market_intelligence_handoff(ctx: RunContextWrapper[SharedAgentContext]):
    st.sidebar.success("🔄 Handing off to Market Intelligence Agent")


# Enhanced agents with tools-for-agents pattern
enhanced_enterprise_agent = Agent(
    name="Enterprise Intelligence Agent",
    handoff_description="Specialist in enterprise analytics pertaining to the store performance, sales, store location, returns, BOPIS(buy online pick up in store), policy, inventory etc.",
    instructions=enhanced_enterprise_prompt,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[
        get_business_conduct_policy_info,
        get_store_performance_info,
        get_product_inventory_info,
        market_intelligence_agent.as_tool(
            tool_name="get_market_intelligence",
            tool_description="Get demographic and market research information for a specific location or area",
        ),
    ],
)

enhanced_market_agent = Agent(
    name="Market Intelligence Agent",
    handoff_description="Specialist in market research pertaining to general questions about the market, industry, news, competitors, demographics, etc.",
    instructions=enhanced_market_prompt,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    tools=[
        get_state_census_data,
        do_research_and_reason,
        enterprise_intelligence_agent.as_tool(
            tool_name="get_enterprise_data",
            tool_description="Get store location, performance data, or inventory information for specific store numbers",
        ),
    ],
)

# Update triage agent with improved instructions
enhanced_triage_prompt = (
    triage_agent_prompt
    + """

## Updated Decision Logic for Compound Questions
For compound questions that require information from multiple agents:
1. Identify the primary intent/goal of the query (what information does the user ultimately want?)
2. Route to the agent that is best suited to deliver the primary information
3. The specialist agent will use other agents as tools when needed

Examples of compound questions:
- "Based on where store 110 is located, what are the demographics of the area?"
   → Route to Market Intelligence Agent (primary goal is demographics information)
   → The Market Intelligence Agent will use the Enterprise Intelligence Agent tool to get store 110's location

## Conversation Summarization 
When the user asks for a summary of the conversation (e.g., "summarize our conversation", "what have we discussed?", etc.):
1. Access the conversation_history from the shared context
2. Generate a concise, structured summary of the key points
3. Focus on key questions, insights, and decision points from the conversation
4. Highlight any important information discovered during the conversation
5. Do NOT hand off to other agents for summarization requests
"""
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=enhanced_triage_prompt,
    model=OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client),
    handoffs=[
        handoff(
            enhanced_enterprise_agent, on_handoff=on_enterprise_intelligence_handoff
        ),
        handoff(enhanced_market_agent, on_handoff=on_market_intelligence_handoff),
    ],
)


# Async function to process a query
async def process_query(query, shared_context):
    """Process a single query through the multi-agent system"""
    # Record user query in conversation history
    shared_context.add_message("User", query)

    # Create hooks for visualization
    hooks = StreamlitAgentHooks()

    # Check if this is a summarization request
    if any(
        phrase in query.lower()
        for phrase in [
            "summarize",
            "summary",
            "what have we discussed",
            "our conversation",
        ]
    ):
        # Add the conversation history to the query for context
        conversation_history = shared_context.get_formatted_history()
        enhanced_query = f"{query}\n\nHere is the conversation history to summarize:\n{conversation_history}"
        st.info("Detected summarization request. Including conversation history.")
    else:
        enhanced_query = query

    # Run the agent
    result = await Runner.run(
        triage_agent,
        enhanced_query,
        context=shared_context,
        hooks=hooks,
    )

    # Get the current active agent's name
    active_agent = shared_context.current_agent or "Assistant"

    # Record the final output in the conversation history with the correct agent name
    shared_context.add_message(active_agent, result.final_output)

    return result, active_agent


# Function to run async operations in a separate thread
@mlflow.trace(span_type="AGENT")
def run_async_query(query):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(
        process_query(query, st.session_state.shared_context)
    )


# Set up Streamlit UI
st.set_page_config(
    page_title="Multi-Agent Intelligence System",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

# Dark mode theme
st.markdown(
    """
<style>
:root {
    --background-color: #121212;
    --text-color: #E0E0E0;
    --accent-color: #4F8BFF;
    --card-bg-color: #1E1E1E;
    --border-color: #333333;
    --success-color: #4CAF50;
    --info-color: #2196F3;
    --warning-color: #FF9800;
    --danger-color: #F44336;
    --triage-color: #9C27B0;
    --enterprise-color: #2196F3;
    --market-color: #4CAF50;
    --tool-color: #FF9800;
}

body {
    color: var(--text-color);
    background-color: var(--background-color);
}

.stApp {
    background-color: var(--background-color);
}

.st-bq {
    background-color: var(--card-bg-color);
    border-left-color: var(--accent-color);
    padding: 20px;
    border-radius: 5px;
}

.stTextInput>div>div>input {
    background-color: var(--card-bg-color);
    color: var(--text-color);
    border: 1px solid var(--border-color);
}

.stTextInput>label {
    color: var(--text-color);
}

.stButton>button {
    background-color: var(--accent-color);
    color: white;
    border: none;
    border-radius: 5px;
    padding: 0.5em 1em;
}

.stButton>button:hover {
    background-color: rgba(79, 139, 255, 0.8);
}

.css-1offfwp {
    color: var(--text-color) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-color) !important;
}

.main > div {
    background-color: var(--background-color);
    padding: 2rem;
}

.stChatMessage {
    background-color: var(--card-bg-color);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border-color);
}

.stChatInput {
    background-color: var(--card-bg-color);
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px;
    padding: 1rem;
}

.stChatInput:focus {
    border-color: var(--accent-color) !important;
}

.agent-card {
    background-color: var(--card-bg-color);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border-color);
}

.agent-status {
    display: flex;
    align-items: center;
    padding: 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    background-color: rgba(30, 30, 30, 0.7);
    border-left: 4px solid var(--accent-color);
}

.agent-active {
    border-left-color: var(--info-color);
}

.agent-complete {
    border-left-color: var(--success-color);
}

.agent-icon, .tool-icon {
    margin-right: 10px;
    font-size: 1.2rem;
}

.agent-name, .tool-name {
    flex: 1;
    font-weight: bold;
}

.status-indicator {
    font-size: 0.8rem;
    color: #AAA;
}

.tool-status {
    display: flex;
    align-items: center;
    padding: 0.5rem;
    border-radius: 5px;
    margin-bottom: 0.5rem;
    background-color: rgba(30, 30, 30, 0.5);
    border-left: 4px solid var(--tool-color);
}

.tool-active {
    border-left-color: var(--warning-color);
}

.tool-complete {
    border-left-color: var(--success-color);
}

.handoff-status {
    display: flex;
    align-items: center;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    background-color: rgba(30, 30, 30, 0.7);
    border-left: 4px solid var(--triage-color);
    justify-content: space-between;
}

.from-agent, .to-agent {
    padding: 5px 10px;
    border-radius: 5px;
    font-weight: bold;
}

.from-agent {
    background-color: rgba(33, 150, 243, 0.2);
    color: #2196F3;
}

.to-agent {
    background-color: rgba(76, 175, 80, 0.2);
    color: #4CAF50;
}

.handoff-icon {
    font-size: 1.5rem;
    margin: 0 15px;
}

.example-card {
    background-color: var(--card-bg-color);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border-color);
    cursor: pointer;
    transition: all 0.2s ease;
}

.example-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    border-color: var(--accent-color);
}

.debug-section {
    background-color: rgba(30, 30, 30, 0.7);
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
    border: 1px solid var(--border-color);
}

.debug-entry {
    padding: 0.5rem;
    border-bottom: 1px solid var(--border-color);
    font-family: monospace;
}

.user-message {
    background-color: rgba(79, 139, 255, 0.1);
    border-left: 4px solid var(--accent-color);
}

.assistant-message {
    background-color: rgba(30, 30, 30, 0.7);
}

.system-message {
    background-color: rgba(76, 175, 80, 0.1);
    border-left: 4px solid var(--success-color);
    font-style: italic;
}

.tool-message {
    background-color: rgba(255, 152, 0, 0.1);
    border-left: 4px solid var(--warning-color);
    font-family: monospace;
}

footer {display: none !important;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "shared_context" not in st.session_state:
    st.session_state.shared_context = SharedAgentContext()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "processing" not in st.session_state:
    st.session_state.processing = False

# Sidebar
with st.sidebar:
    st.title("🤖 Agent Intelligence System")

    st.markdown("### System Information")
    st.info("Multi-agent system with tools-for-agents pattern")

    st.markdown("### Available Agents")

    # Triage Agent Card
    st.markdown(
        """
    <div class="agent-card" style="border-left: 4px solid var(--triage-color);">
        <h4>🔀 Triage Agent</h4>
        <p>Routes queries to appropriate specialist agents</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Enterprise Intelligence Agent Card
    st.markdown(
        """
    <div class="agent-card" style="border-left: 4px solid var(--enterprise-color);">
        <h4>📊 Enterprise Intelligence Agent</h4>
        <p>Store performance, inventory, business policies</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Market Intelligence Agent Card
    st.markdown(
        """
    <div class="agent-card" style="border-left: 4px solid var(--market-color);">
        <h4>📈 Market Intelligence Agent</h4>
        <p>Demographic data, market research</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Debug mode toggle
    st.markdown("### Debug Options")
    debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
    if debug_mode != st.session_state.debug_mode:
        st.session_state.debug_mode = debug_mode

    # Add hint for debug mode
    st.caption(
        "💡 Tip: Only enable/disable debug mode before or after asking a question, not during processing"
    )

    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.shared_context = SharedAgentContext()
        st.rerun()

# Main content area
st.title("Multi-Agent Intelligence System")

# Example queries section
st.markdown("### Example Queries")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
    <div class="example-card" onclick="document.querySelector('.stChatInput input').value = 'Based on where store 110 is located, what are the demographics of the area? Specifically, I'm interested in income levels, total population, home ownership, and education levels.'; document.querySelector('.stChatInput button').click();">
        <p><strong>Mixed Compound Query:</strong> Based on where store 110 is located, what are the demographics of the area? Specifically, I'm interested in income levels, total population, home ownership, and education levels.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="example-card" onclick="document.querySelector('.stChatInput input').value = 'What is the overtime work policy for our vendors?'; document.querySelector('.stChatInput button').click();">
        <p><strong>Policy Query:</strong> What is the overtime work policy for our vendors?</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="example-card" onclick="document.querySelector('.stChatInput input').value = ' I need to do some market research to run a special on golf apparel. Can you help me compare florida and virginia for this purely based on demographics and time of year?'; document.querySelector('.stChatInput button').click();">
        <p><strong>Market Research:</strong> I need to do some market research to run a special on golf apparel. Can you help me compare florida and virginia for this purely based on demographics and time of year?</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="example-card" onclick="document.querySelector('.stChatInput input').value = 'Can you summarize our conversation so far?'; document.querySelector('.stChatInput button').click();">
        <p><strong>Utility:</strong> Can you summarize our conversation so far?</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Status area for agent activity
status_area = st.container()

# Chat history
st.markdown("### Conversation")
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "User":
            with st.chat_message("user"):
                st.markdown(
                    f"""<div class="user-message">{content}</div>""",
                    unsafe_allow_html=True,
                )
        elif role.startswith("Tool"):
            with st.chat_message("assistant", avatar="🔧"):
                st.markdown(
                    f"""<div class="tool-message"><strong>{role}:</strong><br>{content}</div>""",
                    unsafe_allow_html=True,
                )
        elif role == "System":
            with st.chat_message("assistant", avatar="ℹ️"):
                st.markdown(
                    f"""<div class="system-message">{content}</div>""",
                    unsafe_allow_html=True,
                )
        else:
            # Set appropriate avatar based on agent name
            avatar = "🤖"
            if role == "Enterprise Intelligence Agent":
                avatar = "📊"
            elif role == "Market Intelligence Agent":
                avatar = "📈"
            elif role == "Triage Agent":
                avatar = "🔀"

            with st.chat_message("assistant", avatar=avatar):
                st.markdown(
                    f"""<div class="assistant-message"><strong>{role}:</strong><br>{content}</div>""",
                    unsafe_allow_html=True,
                )

# Debug view
if st.session_state.debug_mode:
    with st.expander("Debug Information", expanded=True):
        st.markdown("### Current Context Values")
        st.json(
            {
                "store_location": st.session_state.shared_context.store_location,
                "store_id": st.session_state.shared_context.store_id,
                "state_code": st.session_state.shared_context.state_code,
                "current_agent": st.session_state.shared_context.current_agent,
                "current_tool": st.session_state.shared_context.current_tool,
                "history_length": len(
                    st.session_state.shared_context.conversation_history
                ),
            }
        )

        st.markdown("### Raw Conversation History")
        for i, entry in enumerate(st.session_state.shared_context.conversation_history):
            st.markdown(
                f"""
            <div class="debug-entry">
                <strong>{i}:</strong> <span style="color: #FF9800;">{entry["role"]}</span>: 
                <span style="color: #E0E0E0;">{entry["content"][:100]}{"..." if len(entry["content"]) > 100 else ""}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

# Chat input
if query := st.chat_input(
    "Ask me anything about store performance, market research, or policies..."
):
    # Prevent multiple submissions while processing
    if st.session_state.processing:
        st.warning("Already processing a query, please wait...")
        st.stop()

    st.session_state.processing = True

    # Add user message to chat display
    st.session_state.messages.append({"role": "User", "content": query})
    with st.chat_message("user"):
        st.markdown(
            f"""<div class="user-message">{query}</div>""", unsafe_allow_html=True
        )

    # Process the query
    with status_area:
        with st.status("Processing your query...", expanded=True) as status:
            try:
                # Use a separate thread for async operation
                result, active_agent = run_async_query(query)

                # Add assistant response to chat display with the correct agent name
                final_output = result.final_output
                st.session_state.messages.append(
                    {"role": active_agent, "content": final_output}
                )

                status.update(
                    label="Query processed successfully",
                    state="complete",
                    expanded=False,
                )

                # Display the response with the correct agent avatar
                avatar = "🤖"
                if active_agent == "Enterprise Intelligence Agent":
                    avatar = "📊"
                elif active_agent == "Market Intelligence Agent":
                    avatar = "📈"
                elif active_agent == "Triage Agent":
                    avatar = "🔀"

                with st.chat_message("assistant", avatar=avatar):
                    st.markdown(
                        f"""<div class="assistant-message"><strong>{active_agent}:</strong><br>{final_output}</div>""",
                        unsafe_allow_html=True,
                    )

                # Force a rerun to update the UI with the new messages
                st.session_state.processing = False
                st.rerun()

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                import traceback

                if st.session_state.debug_mode:
                    st.code(traceback.format_exc())
            finally:
                st.session_state.processing = False
