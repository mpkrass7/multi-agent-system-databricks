# %%
from agents import function_tool, set_tracing_disabled
import os
import requests
import time
from unitycatalog.ai.core.databricks import (
    DatabricksFunctionClient,
    FunctionExecutionResult,
)
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai import OpenAI
from census import Census
from us import states
from rich import print
import streamlit as st

# %%
# Load environment variables
load_dotenv("/Users/sathish.gangichetty/Documents/openai-agents/apps/.env-local")

# Initialize environment variables
BASE_URL = os.getenv("DATABRICKS_BASE_URL") or ""
# API_KEY = st.context.headers.get('X-Forwarded-Access-Token')
API_KEY = os.getenv("DATABRICKS_TOKEN") or ""
MODEL_NAME = os.getenv("DATABRICKS_MODEL") or ""
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
set_tracing_disabled(True)

# Initialize clients
client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
w = WorkspaceClient(
    host=os.getenv("DATABRICKS_HOST"),
    token=os.getenv("DATABRICKS_TOKEN"),
    auth_type="pat",
)
dbclient = DatabricksFunctionClient(client=w)


@function_tool
def get_store_performance_info(user_query: str):
    """
    For us, we use this to get information about the store location, store performance, returns, BOPIS(buy online pick up in store) etc.
    """
    st.write(
        "<span style='color:green;'>[🛠️TOOL-CALL]: the <a href='https://adb-984752964297111.11.azuredatabricks.net/genie/rooms/01f023ae84651418a1203b194dff21a9?o=984752964297111' target='_blank'>get_store_performance_info</a> tool was called</span>",
        unsafe_allow_html=True,
    )
    print("INFO: `get_store_performance_info` tool called")
    databricks_instance = os.getenv("DATABRICKS_HOST")
    space_id = os.getenv("GENIE_SPACE_ID")
    access_token = os.getenv("DATABRICKS_TOKEN")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    poll_interval = 2.0
    timeout = 60.0

    # Step 1: Start a new conversation
    start_url = (
        f"{databricks_instance}/api/2.0/genie/spaces/{space_id}/start-conversation"
    )
    payload = {"content": user_query}
    resp = requests.post(start_url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(data)
    conversation_id = data["conversation_id"]
    message_id = data["message_id"]

    # Step 2: Poll for completion
    poll_url = f"{databricks_instance}/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}"
    start_time = time.time()
    while True:
        poll_resp = requests.get(poll_url, headers=headers)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        status = poll_data.get("status")
        if status == "COMPLETED":
            attachment_id = poll_data["attachments"][0]["attachment_id"]
            url = f"{databricks_instance}/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()["statement_response"]
        if time.time() - start_time > timeout:
            raise TimeoutError("Genie API query timed out.")
        time.sleep(poll_interval)


@function_tool
def get_product_inventory_info(user_query: str):
    """
    For us, we use this to get information about products and the current inventory snapshot across stores
    """
    st.write(
        "<span style='color:green;'>[🛠️TOOL-CALL]: the <a href='https://adb-984752964297111.11.azuredatabricks.net/genie/rooms/01f02c2c29211c388b9b5b9b6f5a80c9?o=984752964297111' target='_blank'>get_product_inventory_info</a> tool was called</span>",
        unsafe_allow_html=True,
    )
    print("INFO: `get_product_inventory_info` tool called")
    databricks_instance = os.getenv("DATABRICKS_HOST")
    space_id = os.getenv("GENIE_SPACE_PRODUCT_INV_ID")
    access_token = os.getenv("DATABRICKS_TOKEN")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    poll_interval = 2.0
    timeout = 60.0

    # Step 1: Start a new conversation
    start_url = (
        f"{databricks_instance}/api/2.0/genie/spaces/{space_id}/start-conversation"
    )
    payload = {"content": user_query}
    resp = requests.post(start_url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    print(data)
    conversation_id = data["conversation_id"]
    message_id = data["message_id"]

    # Step 2: Poll for completion
    poll_url = f"{databricks_instance}/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}"
    start_time = time.time()
    while True:
        poll_resp = requests.get(poll_url, headers=headers)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        status = poll_data.get("status")
        if status == "COMPLETED":
            attachment_id = poll_data["attachments"][0]["attachment_id"]
            url = f"{databricks_instance}/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}/attachments/{attachment_id}/query-result"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()["statement_response"]
        if time.time() - start_time > timeout:
            raise TimeoutError("Genie API query timed out.")
        time.sleep(poll_interval)


@function_tool
def get_business_conduct_policy_info(search_query: str) -> FunctionExecutionResult:
    st.write(
        "<span style='color:green;'>[🛠️TOOL-CALL]: the <a href='https://adb-984752964297111.11.azuredatabricks.net/explore/data/main/sgfs/retail_conduct_policy?o=984752964297111&activeTab=overview' target='_blank'>get_business_conduct_policy_info</a> tool was called</span>",
        unsafe_allow_html=True,
    )
    print("INFO: `get_business_conduct_policy_info` tool called")
    return dbclient.execute_function(
        function_name="main.sgfs.retail_club_conduct",
        parameters={"search_query": search_query},
    )


@function_tool
def do_research_and_reason(user_query: str):
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

    # chat completion without streaming
    # response = client.chat.completions.create(
    #     model="sonar-pro",
    #     messages=messages,
    # )
    # return response.choices[0].message.content

    # chat completion with streaming
    response_stream = client.chat.completions.create(
        model="sonar-reasoning-pro",
        messages=messages,
        stream=True,
    )

    full_response = ""
    for response in response_stream:
        if response.choices and response.choices[0].delta.content:
            content = response.choices[0].delta.content
            full_response += content

    return full_response


# from langchain_community.tools import BraveSearch

# tool = BraveSearch.from_api_key(
#     api_key=os.getenv("BRAVE_API_KEY"), search_kwargs={"count": 5}
# )
# print(tool.invoke("prices of blue tack"))


# %%


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
