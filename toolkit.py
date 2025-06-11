# %%
import os
import time

from agents import function_tool, set_tracing_disabled
from census import Census
from databricks.sdk import WorkspaceClient
from openai import AsyncOpenAI, OpenAI
from rich import print
import yaml

from us import states


# Initialize environment variables
BASE_URL = os.getenv("DATABRICKS_BASE_URL") or ""
# API_KEY = st.context.headers.get('X-Forwarded-Access-Token')
API_KEY = os.getenv("DATABRICKS_TOKEN") or ""
MODEL_NAME = os.getenv("DATABRICKS_MODEL") or ""
HOST = os.getenv("DATABRICKS_HOST") or ""

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
set_tracing_disabled(True)


with open("app_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    CATALOG = config.get("CATALOG", "retail")
    SCHEMA = config.get("SCHEMA", "retail")
    GENIE_SPACE_STORE_PERFORMANCE_ID = config.get(
        "GENIE_SPACE_STORE_PERFORMANCE_ID", "store_performance"
    )
    GENIE_SPACE_PRODUCT_INV_ID = config.get(
        "GENIE_SPACE_PRODUCT_INV_ID", "product_inventory"
    )

# Initialize clients
w = WorkspaceClient(
    host=HOST,
    token=os.getenv("DATABRICKS_TOKEN"),
    auth_type="pat",
)
sync_client = w.serving_endpoints.get_open_ai_client()
client = AsyncOpenAI(
    base_url=sync_client.base_url, api_key=os.getenv("DATABRICKS_TOKEN")
)


@function_tool
def get_store_performance_info(user_query: str):
    """
    Provide information about the store location, store performance, returns, BOPIS(buy online pick up in store) etc.
    """

    space_id = GENIE_SPACE_STORE_PERFORMANCE_ID
    print(f"INFO: `get_store_performance_info` tool called with space_id: {space_id}")
    timeout = 60.0
    poll_interval = 2.0
    # Step 1: Start a new conversation using the SDK
    message = w.genie.start_conversation_and_wait(space_id, user_query)
    conversation_id = message.conversation_id
    message_id = message.id
    # Step 2: Poll for completion using the SDK
    start_time = time.time()
    while True:
        msg = w.genie.get_message(space_id, conversation_id, message_id)
        status = msg.status.value if msg.status else None
        if status == "COMPLETED":
            if msg.attachments and len(msg.attachments) > 0:
                attachment_id = msg.attachments[0].attachment_id
                result = w.genie.get_message_attachment_query_result(
                    space_id, conversation_id, message_id, attachment_id
                )
                return result.statement_response.as_dict()
            else:
                return {"error": "No attachments found in message."}
        if time.time() - start_time > timeout:
            raise TimeoutError("Genie API query timed out.")
        time.sleep(poll_interval)


@function_tool
def get_product_inventory_info(user_query: str):
    """
    Provide information about products and the current inventory snapshot across stores.
    """

    space_id = GENIE_SPACE_PRODUCT_INV_ID
    print(f"INFO: `get_product_inventory_info` tool called with space_id: {space_id}")
    timeout = 60.0
    poll_interval = 2.0
    # Step 1: Start a new conversation using the SDK
    message = w.genie.start_conversation_and_wait(space_id, user_query)
    conversation_id = message.conversation_id
    message_id = message.id
    # Step 2: Poll for completion using the SDK
    start_time = time.time()
    while True:
        msg = w.genie.get_message(space_id, conversation_id, message_id)
        status = msg.status.value if msg.status else None
        if status == "COMPLETED":
            if msg.attachments and len(msg.attachments) > 0:
                attachment_id = msg.attachments[0].attachment_id
                result = w.genie.get_message_attachment_query_result(
                    space_id, conversation_id, message_id, attachment_id
                )
                return result.statement_response.as_dict()
            else:
                return {"error": "No attachments found in message."}
        if time.time() - start_time > timeout:
            raise TimeoutError("Genie API query timed out.")
        time.sleep(poll_interval)


@function_tool
def get_business_conduct_policy_info(search_query: str) -> str:
    """
    Provide information about business conduct policies and code of conduct pertaining to vendors and suppliers.

    Args:
        user_query: The user's question or request
    Returns:
        Relevant documentation or information related to business conduct policies.
    """
    print("INFO: `get_business_conduct_policy_info` tool called")

    index_name = f"{CATALOG}.{SCHEMA}.retail_code_of_conduct_index"

    return w.vector_search_indexes.query_index(
        index_name=index_name,
        query_text=search_query,
        columns=["sec_id", "text_chunks"],
        num_results=2,
    )


@function_tool
def do_research_and_reason(user_query: str) -> str:
    """
    Get a response from sythesized intelligence from the web including the "thinking" process highligted in the <think></think> tags

    Args:
        user_query: The user's question or request
    Returns:
        The sythesized intelligence from the web including the "thinking"
        process inside <think></think> tags
    """
    print("INFO: `do_research_and_reason` tool called")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": f"today is {time.strftime('%d %B %Y')} + " + user_query,
        },
    ]

    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai"
    )

    # chat completion with streaming
    response_stream = client.chat.completions.create(
        model="sonar-reasoning-pro",  # o3 comp
        messages=messages,
        stream=True,
    )

    full_response = ""
    for response in response_stream:
        if response.choices and response.choices[0].delta.content:
            content = response.choices[0].delta.content
            full_response += content

    return full_response


# https://downloads.esri.com/esri_content_doc/dbl/us/Var_List_ACS_Winter_2021.pdf
@function_tool
def get_state_census_data(state_code: str) -> str:
    """
    Get census data for a specific state.

    Args:
        state_code: The two-letter state code (e.g., 'MD' for Maryland)

    Returns:
        A formatted string with the state name, population, and median household income
    """
    print("INFO: `get_state_census_data` tool called")
    c = Census(os.getenv("CENSUS_API_KEY"))
    state_obj = getattr(states, state_code.upper())
    results = c.acs5.get(
        (
            "NAME",
            "B11016_001E",  # total households
            "B25081_001E",  # owner occupied households
            "B01003_001E",  # total population
            "B19013_001E",  # median household income
            "B15003_022E",  # bachelors degree graduates
            "B15003_023E",  # masters degree graduates
            "B15003_025E",  # doctorate degree graduates
        ),
        {"for": "state:{}".format(state_obj.fips)},
    )

    if results:
        state_name = results[0]["NAME"]
        total_households = int(results[0]["B11016_001E"] or 0)
        owner_occupied_households = int(results[0]["B25081_001E"] or 0)
        population = int(results[0]["B01003_001E"] or 0)
        median_income = int(results[0]["B19013_001E"] or 0)
        bachelors_degree_graduates = int(results[0]["B15003_022E"] or 0)
        masters_degree_graduates = int(results[0]["B15003_023E"] or 0)
        doctorate_degree_graduates = int(results[0]["B15003_025E"] or 0)

        # Calculate percentages and ratios
        owner_occupied_percent = (
            (owner_occupied_households / total_households * 100)
            if total_households
            else 0
        )
        people_per_household = (
            (population / total_households) if total_households else 0
        )
        bachelors_percent = (
            (bachelors_degree_graduates / population * 100) if population else 0
        )
        masters_percent = (
            (masters_degree_graduates / population * 100) if population else 0
        )
        doctorate_percent = (
            (doctorate_degree_graduates / population * 100) if population else 0
        )

        return (
            f"{state_name} has an estimated population of {round(population)} and a median household income of ${round(median_income)}. "
            f"{round(owner_occupied_percent, 1)}% of households are owner-occupied with an average of {round(people_per_household, 2)} people per household. "
            f"Education levels: {round(bachelors_percent, 1)}% have a bachelor's degree, {round(masters_percent, 1)}% have a master's degree, "
            f"and {round(doctorate_percent, 1)}% have a doctorate degree."
        )
    else:
        return f"No census data found for state code {state_code}."
