# %%
import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    handoff,
    set_tracing_disabled,
)
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

from toolkit import (
    do_research_and_reason,
    get_business_conduct_policy_info,
    get_product_inventory_info,
    get_state_census_data,
    get_store_performance_info,
)

# Initialize Rich console
console = Console()

load_dotenv()

MODEL_NAME = os.getenv("DATABRICKS_MODEL") or ""
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
BASE_URL = os.getenv("DATABRICKS_BASE_URL") or ""
API_KEY = os.getenv("DATABRICKS_TOKEN") or ""
set_tracing_disabled(True)


w = WorkspaceClient(host=os.environ["DATABRICKS_HOST"], token=API_KEY)
sync_client = w.serving_endpoints.get_open_ai_client()
client = AsyncOpenAI(base_url=sync_client.base_url, api_key=API_KEY)


# Define a shared context class to pass data between agents
@dataclass
class SharedAgentContext:
    store_location: Optional[str] = None
    store_id: Optional[str] = None
    demographic_data: Optional[Dict] = None
    state_code: Optional[str] = None
    current_agent: Optional[str] = None
    current_tool: Optional[str] = None
    # Add message history tracking
    conversation_history: list = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def get_formatted_history(self):
        """Format conversation history for consumption by VisionCraft MCP"""
        if not self.conversation_history:
            return "No conversation history available."

        formatted_history = []
        for msg in self.conversation_history:
            formatted_history.append(f"{msg['role']}: {msg['content']}")

        return "\n\n".join(formatted_history)


# %%
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


# Create lifecycle hooks to track agent execution
class AgentExecutionHooks:
    def __init__(self, console):
        self.console = console
        self.start_time = None

    async def on_agent_start(
        self, context: RunContextWrapper[SharedAgentContext], agent
    ):
        self.start_time = time.time()
        agent_name = agent.name
        context.context.current_agent = agent_name
        # Log conversation history entries
        history_count = (
            len(context.context.conversation_history)
            if context.context.conversation_history
            else 0
        )
        self.console.print(
            f"[bold blue]🚀 Starting {agent_name} with {history_count} history entries"
        )

    async def on_agent_end(
        self, context: RunContextWrapper[SharedAgentContext], agent, output
    ):
        agent_name = agent.name
        duration = time.time() - self.start_time
        self.console.print(
            Panel(
                f"[bold green]✅ {agent_name} completed in {duration:.2f}s",
                expand=False,
            )
        )
        # Record agent response in conversation history
        context.context.add_message(f"{agent_name}", output)
        # Log the updated number of history entries
        history_count = len(context.context.conversation_history)
        self.console.print(
            f"[dim]Conversation history now has {history_count} entries[/dim]"
        )

    async def on_tool_start(
        self, context: RunContextWrapper[SharedAgentContext], agent, tool
    ):
        tool_name = tool.name
        context.context.current_tool = tool_name
        self.console.print(f"[yellow]🔧 Using tool: {tool_name}")

    async def on_tool_end(
        self, context: RunContextWrapper[SharedAgentContext], agent, tool, result
    ):
        tool_name = tool.name
        self.console.print(f"[green]✓ Tool {tool_name} completed")
        # Record tool usage in conversation history
        context.context.add_message(f"Tool ({tool_name})", str(result))
        # Log the updated number of history entries
        history_count = len(context.context.conversation_history)
        self.console.print(
            f"[dim]Conversation history now has {history_count} entries[/dim]"
        )

    async def on_handoff(
        self, context: RunContextWrapper[SharedAgentContext], from_agent, to_agent
    ):
        from_name = from_agent.name
        to_name = to_agent.name
        self.console.print(
            Panel(f"[bold magenta]↪️ Handoff: {from_name} → {to_name}", expand=False)
        )
        # Record handoff in conversation history
        context.context.add_message("System", f"Handoff from {from_name} to {to_name}")
        # Log the updated number of history entries
        history_count = len(context.context.conversation_history)
        self.console.print(
            f"[dim]Conversation history now has {history_count} entries[/dim]"
        )


# First create placeholder agents, then enhance them with tools

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

# Now enhance each agent with the other as a tool
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


def on_enterprise_intelligence_handoff(ctx: RunContextWrapper[SharedAgentContext]):
    rprint("[bold cyan]🔄 Handing off to enterprise intelligence agent")


def on_market_intelligence_handoff(ctx: RunContextWrapper[SharedAgentContext]):
    rprint("[bold cyan]🔄 Handing off to market intelligence agent")


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

# %%
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


# %%
async def process_query(query, shared_context=None):
    """Process a single query through the multi-agent system"""
    # Create a shared context object if not provided
    if shared_context is None:
        shared_context = SharedAgentContext()

    # Record user query in conversation history
    shared_context.add_message("User", query)

    # Create hooks for visualization
    hooks = AgentExecutionHooks(console)

    console.print(f"[bold white on blue]User Query:[/] {query}")

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
        console.print(
            "[yellow]Detected summarization request. Including conversation history.[/]"
        )
    else:
        enhanced_query = query

    # Run the agent with a progress bar wrapper
    with console.status("[bold yellow]Processing query...", spinner="dots"):
        result = await Runner.run(
            triage_agent,
            enhanced_query,
            context=shared_context,
            hooks=hooks,
        )

    # Print the final output with nice formatting
    console.print(
        Panel(
            f"[bold green]🎯 Final Output:[/]\n\n{result.final_output}",
            expand=False,
            border_style="green",
        )
    )

    # Record the final output in the conversation history
    shared_context.add_message("System", result.final_output)

    return result, shared_context


async def interactive_session():
    """Run an interactive session with the multi-agent system"""
    console.print(
        Panel.fit(
            "[bold]🤖 Starting Multi-Agent System with Tools-for-Agents Pattern",
            style="blue",
            border_style="blue",
        )
    )

    console.print(
        "[bold]Type your queries and press Enter. Type 'exit' or 'quit' to end the session.[/]"
    )
    console.print(
        "[bold]Type 'debug' to see the raw conversation history (for troubleshooting).[/]"
    )

    # Example queries for user reference
    console.print(
        Panel(
            "[dim]Example queries:\n"
            + "- Based on where store 110 is located, what are the demographics of the area?\n"
            + "- Is Florida a good place to open a new store compared to Virginia?\n"
            + "- What is the policy for returns at our stores?\n"
            + "- Can you summarize our conversation so far?[/dim]",
            title="Examples",
            expand=False,
        )
    )

    # Keep track of the shared context across queries
    shared_context = SharedAgentContext()

    # Continue processing queries until user exits
    while True:
        try:
            # Get query from user
            query = console.input("\n[bold cyan]Enter your query:[/] ")

            # Check if user wants to exit
            if query.lower() in ("exit", "quit"):
                console.print("[bold]Exiting session. Goodbye![/]")
                break

            # Debug command to check conversation history
            if query.lower() == "debug":
                console.print("[bold yellow]DEBUG: Conversation History[/]")
                for i, entry in enumerate(shared_context.conversation_history):
                    console.print(
                        f"[dim]{i}.[/dim] {entry['role']}: {entry['content'][:100]}..."
                        if len(entry["content"]) > 100
                        else f"[dim]{i}.[/dim] {entry['role']}: {entry['content']}"
                    )
                continue

            # Skip empty queries
            if not query.strip():
                continue

            # Process the query with the shared context
            _, shared_context = await process_query(query, shared_context)

        except KeyboardInterrupt:
            console.print("\n[bold red]Session interrupted. Exiting...[/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error processing query: {str(e)}[/]")
            import traceback

            console.print(traceback.format_exc())


async def run_single_query(query):
    """Run a single query through the multi-agent system"""
    console.print(
        Panel.fit(
            "[bold]🤖 Starting Multi-Agent System with Tools-for-Agents Pattern",
            style="blue",
            border_style="blue",
        )
    )

    result, context = await process_query(query)


# Run the async function
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the multi-agent system with Tools-for-Agents pattern"
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        help="A single query to process (runs in non-interactive mode)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode (default if no query provided)",
    )
    args = parser.parse_args()

    if args.query:
        # Run a single query
        asyncio.run(run_single_query(args.query))
    else:
        # Run in interactive mode
        asyncio.run(interactive_session())
