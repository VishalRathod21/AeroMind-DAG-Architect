import warnings
warnings.filterwarnings('ignore')
import logging
import os
import re
from typing import Optional, Any, Dict, List, Union
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt
import time
from langchain_core.messages import HumanMessage

from ..agents.workflow import create_dag_generation_workflow
from ..config.settings import config

logger = logging.getLogger(__name__)

class AgentOutput(BaseModel):
    """Structured output schema for agent results"""
    dag_code: str
    analysis: Optional[str] = None
    review: Optional[str] = None

class DAGGeneratorClient:
    """Client interface for DAG generation using LangChain and LangGraph."""
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 codebase_path: Optional[str] = None, 
                 airflow_container_name: Optional[str] = None,
                 llm_provider: str = "google",
                 airflow_compose_file: Optional[str] = None,
                 use_embeddings: bool = True,
                 embedding_provider: str = "local",
                 requirements: str = "",
                 airflow_webserver_url: Optional[str] = None,
                 log_level: str = "INFO"):
        """Initialize the DAGGeneratorClient."""
        logger.info(f"Initializing DAGGeneratorClient with codebase: {codebase_path}")
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
        
        # Set instance variables with fallbacks to config
        self.api_key = api_key or (config.GOOGLE_API_KEY if llm_provider.lower() == "google" else config.OPENAI_API_KEY)
        self.codebase_path = codebase_path or os.path.join(os.getcwd(), "dags")
        self.airflow_container_name = airflow_container_name or "airflow-airflow-webserver-1"
        self.airflow_webserver_url = airflow_webserver_url or config.AIRFLOW_WEBSERVER_URL
        self.llm_provider = llm_provider.lower()
        self.use_embeddings = use_embeddings
        self.embedding_provider = embedding_provider.lower()
        self.requirements = requirements
        
        # Validate required configurations
        if not self.api_key or self.api_key == "your_api_key":
            logger.error(f"API Key is missing for {self.llm_provider}")
            raise ValueError(f"Please provide a valid {self.llm_provider.capitalize()} API Key.")
            
        if not os.path.isdir(self.codebase_path):
            logger.warning(f"Creating codebase path: {self.codebase_path}")
            os.makedirs(self.codebase_path, exist_ok=True)
        
        # Initialize components
        self.workflow = None
        self.docker_manager = None
        self.codebase_context = None

    def _validate_provider(self, provider: str, valid_options: List[str], default: str) -> str:
        """Validate and normalize provider selection."""
        if provider.lower() not in valid_options:
            logger.warning(f"Unknown provider: {provider}, defaulting to {default}")
            return default
        return provider.lower()

    def _initialize_workflow(self, requirements: str):
        """Initialize the LangGraph workflow with current requirements."""
        try:
            self.workflow, self.docker_manager, self.codebase_context = create_dag_generation_workflow(
                api_key=self.api_key,
                codebase_path=self.codebase_path,
                airflow_container_name=self.airflow_container_name,
                llm_provider=self.llm_provider,
                use_embeddings=self.use_embeddings,
                embedding_provider=self.embedding_provider,
                requirements=requirements
            )
            logger.info("Successfully initialized workflow and Docker manager")
        except Exception as e:
            logger.error(f"Failed to initialize workflow: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3))
    def generate_dag(self, requirements: str) -> Dict[str, Any]:
        """Generate DAG using LangGraph workflow."""
        requirements = requirements.strip()[:5000]  # Sanitize input
        logger.info(f"Generating DAG for requirements: {requirements[:100]}...")
        start_time = time.time()
        
        try:
            # Initialize workflow with latest requirements
            self._initialize_workflow(requirements)
            
            # Execute the workflow
            workflow_state = {
                "requirements": requirements,
                "messages": [HumanMessage(content=requirements)]
            }
            
            logger.info("Executing LangGraph workflow...")
            result = self.workflow.invoke(workflow_state)
            logger.info("Workflow execution completed")
            
            # Extract DAG code from the workflow result
            dag_code = None
            dag_id = None
            
            # Check if we have a valid implementation in the result
            if 'implementation' in result and result['implementation']:
                impl = result['implementation']
                if 'code' in impl and impl['code']:
                    dag_code = impl['code']
                    if 'dag_id' in impl:
                        dag_id = impl['dag_id']
            
            # Generate a DAG ID if not found
            if not dag_id:
                dag_id = f"generated_dag_{int(time.time())}"
            
            # Extract analysis report
            analysis_report = result.get('analysis', 'No analysis report generated.')
            
            logger.info(f"DAG generation completed in {time.time() - start_time:.2f}s")
            
            return {
                "success": True,
                "dag_code": dag_code,
                "dag_id": dag_id,
                "analysis_report": analysis_report,
                "execution_time": time.time() - start_time
            }
                
        except Exception as e:
            logger.error(f"Error generating DAG: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    def deploy_and_test_dag(self, dag_code: Optional[str], dag_id: Optional[str]) -> Dict[str, Any]:
        """Deploy and test the generated DAG in Airflow."""
        if not dag_code:
            return {"deployed": False, "error": "No DAG code provided"}
            
        try:
            dag_id = dag_id or f"generated_dag_{int(time.time())}"
            logger.info(f"Deploying DAG with ID: {dag_id}")
            
            # Ensure workflow is initialized
            if not hasattr(self, 'docker_manager') or not self.docker_manager:
                logger.info("Initializing workflow before deployment...")
                self._initialize_workflow("")
                
            # Deploy the DAG
            logger.info(f"Deploying DAG {dag_id}...")
            deploy_result = self.docker_manager.deploy_dag(dag_code, dag_id)
            
            if not deploy_result.get("deployed", False):
                return deploy_result
                
            # Test the DAG
            logger.info(f"Validating DAG {dag_id}...")
            test_result = self.docker_manager.validate_dag(dag_id)
            
            return {
                "deployed": True,
                "dag_id": dag_id,
                "test_success": test_result.get("valid", False),
                "test_result": test_result.get("result", None),
                "error": test_result.get("error", None)
            }
            
        except Exception as e:
            error_msg = f"Error during DAG deployment: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"deployed": False, "error": error_msg}
