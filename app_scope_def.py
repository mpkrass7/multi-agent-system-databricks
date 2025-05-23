#%%
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#%%

def update_app_scopes(app_name: str, scopes: list, host: str = None, token: str = None):
    """
    Update a Databricks app with custom API scopes
    
    Args:
        app_name (str): Name of the Databricks app to update
        scopes (list): List of OAuth scopes to assign to the app
        host (str, optional): Databricks workspace host URL
        token (str, optional): Databricks personal access token
    """
    
    # Initialize the Databricks client
    try:
        client = WorkspaceClient(
            host=host or os.getenv("DATABRICKS_HOST"),
            token=token or os.getenv("DATABRICKS_TOKEN"),
            auth_type="pat",
        )
        logger.info("Successfully initialized Databricks client")
    except Exception as e:
        logger.error(f"Failed to initialize Databricks client: {str(e)}")
        raise

    # Validate scopes
    valid_scopes = [
        "sql",
        "serving:serving-endpoints", 
        "vector-search",
        "iam.current-user:read",
        "iam.access-control:read",
        "iam.access-control:write",
        "clusters",
        "jobs",
        "notebooks",
        "repos",
        "workspace",
        "ml",
        "pipeline"
    ]
    
    invalid_scopes = [scope for scope in scopes if scope not in valid_scopes]
    if invalid_scopes:
        logger.warning(f"Some scopes may not be valid: {invalid_scopes}")
        logger.info(f"Valid scopes include: {valid_scopes}")

    try:
        # Get the current app configuration
        logger.info(f"Fetching current configuration for app: {app_name}")
        current_app = client.apps.get(name=app_name)
        logger.info(f"Current app scopes: {current_app.user_api_scopes}")
        
        # Check if scopes are already set correctly
        if set(current_app.user_api_scopes or []) == set(scopes):
            logger.info("App already has the desired scopes. No update needed.")
            return current_app

        # Update the app with new OAuth scopes
        logger.info(f"Updating app '{app_name}' with scopes: {scopes}")
        updated_app = client.apps.update(name=app_name, user_api_scopes=scopes)

        logger.info(f"Successfully updated app '{app_name}'")
        logger.info(f"Updated app scopes: {updated_app.user_api_scopes}")
        logger.info(f"Effective scopes: {updated_app.effective_user_api_scopes}")
        
        return updated_app

    except Exception as e:
        logger.error(f"Error updating app scopes: {str(e)}")
        raise

# Main execution
if __name__ == "__main__":
    # Define the app name (replace with your actual app name)
    app_name = "multi-agent-retail-ai-system"

    # Define the desired OAuth scopes
    custom_scopes = [
        "sql",
        "serving:serving-endpoints",
        "vector-search", 
        "iam.current-user:read",
        "iam.access-control:read",
    ]

    # Update the app
    try:
        updated_app = update_app_scopes(app_name, custom_scopes)
        print("\n" + "="*50)
        print("APP UPDATE SUMMARY")
        print("="*50)
        print(f"App Name: {app_name}")
        print(f"Requested Scopes: {custom_scopes}")
        print(f"Applied Scopes: {updated_app.user_api_scopes}")
        print(f"Effective Scopes: {updated_app.effective_user_api_scopes}")
        print("="*50)
        
    except Exception as e:
        print(f"Failed to update app: {str(e)}")
        exit(1)

# COMMAND ----------
