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
from langgraph.graph import END, StateGraph

from langgraph_setup import create_dag_generation_workflow
from config import config

logger = logging.getLogger(__name__)

class AgentOutput(BaseModel):
    """Structured output schema for agent results"""
    dag_code: str
    analysis: Optional[str] = None
    review: Optional[str] = None

class DAGGeneratorClient:
    """Client interface for DAG generation using LangChain and LangGraph.
    
    This class provides a high-level interface for generating, deploying, and testing
    Airflow DAGs using LangChain agents orchestrated with LangGraph workflows.
    
    Attributes:
        api_key (str): The API key for the LLM provider.
        codebase_path (str): Path to the codebase to analyze.
        airflow_container_name (str): Name of the Airflow container for deployment.
        llm_provider (str): The LLM provider to use ('google' or 'openai').
        embedding_provider (str): The embedding provider to use ('google', 'openai', or 'local').
        workflow: The LangGraph workflow for DAG generation.
        docker_manager: The Docker manager for container operations.
        codebase_context: The codebase context for code analysis.
    """
    
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
        """Initialize the DAGGeneratorClient.
        
        Args:
            api_key (str): The API key for the LLM provider.
            codebase_path (str): Path to the codebase to analyze.
            airflow_container_name (str): Name of the Airflow container for deployment.
            llm_provider (str): The LLM provider to use ('google' or 'openai').
            airflow_compose_file (Optional[str]): Path to docker-compose.yml for Airflow.
            use_embeddings (bool): Whether to use embeddings for semantic search.
            embedding_provider (str): The embedding provider to use ('google', 'openai', or 'local').
            requirements (str): User requirements for the DAG in natural language.
            
        Raises:
            ValueError: If the API key or codebase path is invalid.
        """
        logger.info(f"Initializing DAGGeneratorClient with codebase: {codebase_path}, "
                   f"Airflow container: {airflow_container_name}, LLM provider: {llm_provider}")
        
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
            logger.error(f"API Key is missing or using the placeholder value for {self.llm_provider}.")
            raise ValueError(f"Please provide a valid {self.llm_provider.capitalize()} API Key.")
            
        if not os.path.isdir(self.codebase_path):
            logger.warning(f"Codebase path ('{self.codebase_path}') doesn't exist, creating it...")
            os.makedirs(self.codebase_path, exist_ok=True)
        
        # Validate providers
        self.llm_provider = self._validate_provider(llm_provider, ["google", "openai"], "Google")
        self.embedding_provider = self._validate_provider(embedding_provider, ["google", "openai", "local"], "local")
        
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

    def _extract_code_from_output(self, text: Union[str, Dict, Any]) -> Optional[str]:
        """Extracts Python code block from various output formats.
        
        This method handles multiple input formats including:
        - Raw Python code
        - Markdown code blocks
        - Dictionary with code in various fields
        - Workflow state output with keys like 'implementation', 'messages', 'analysis', etc.
        
        Args:
            text: The text, dictionary, or workflow output containing the code.
                
        Returns:
            Extracted Python code as a string, or None if no code could be extracted.
        """
        if not text:
            logger.debug("No text provided to extract code from")
            return None
            
        logger.debug(f"Attempting to extract code from: {str(text)[:200]}...")
        
        # If input is a dictionary, try common code-containing fields
        if isinstance(text, dict):
            # Log the available keys for debugging
            logger.debug(f"Available keys in output dict: {list(text.keys())}")
            
            # Check for direct code fields first
            for field in ['code', 'dag_code', 'implementation']:
                if field in text and text[field]:
                    logger.debug(f"Found code in field: {field}")
                    if isinstance(text[field], str):
                        if self._looks_like_dag(text[field]):
                            return text[field].strip()
                    elif isinstance(text[field], dict):
                        # Handle nested code in dictionaries
                        nested_code = self._extract_code_from_output(text[field])
                        if nested_code:
                            return nested_code
            
            # Check for workflow state output with 'outputs' key
            if 'outputs' in text and isinstance(text['outputs'], dict):
                logger.debug("Found 'outputs' key, checking for code in outputs")
                for key in ['implementation', 'code', 'result']:
                    if key in text['outputs'] and text['outputs'][key]:
                        code = self._extract_code_from_output(text['outputs'][key])
                        if code:
                            return code
            
            # Handle the specific case where we have 'code' as a direct key
            if 'code' in text and text['code']:
                code_content = text['code']
                if isinstance(code_content, str):
                    if self._looks_like_dag(code_content):
                        return code_content.strip()
                    # If it's a string but not a DAG, try to extract code blocks
                    code_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)```', code_content)
                    for block in code_blocks:
                        if self._looks_like_dag(block):
                            return block.strip()
                elif isinstance(code_content, dict):
                    # If code is a dict, try to extract string content from it
                    for key in ['code', 'content', 'implementation']:
                        if key in code_content and isinstance(code_content[key], str):
                            if self._looks_like_dag(code_content[key]):
                                return code_content[key].strip()
            
            # Handle the specific case where we have 'implementation' as a direct key
            # with content that might be a string or dict
            if 'implementation' in text and text['implementation']:
                impl = text['implementation']
                if isinstance(impl, str):
                    if self._looks_like_dag(impl):
                        return impl.strip()
                elif isinstance(impl, dict):
                    # If implementation is a dict, try to extract code from it
                    for key in ['code', 'dag_code', 'content']:
                        if key in impl and isinstance(impl[key], str):
                            if self._looks_like_dag(impl[key]):
                                return impl[key].strip()
            
            # If we have a 'messages' list, check the last message for code
            if 'messages' in text and isinstance(text['messages'], list) and text['messages']:
                logger.debug("Found 'messages' list, checking last message for code")
                last_message = text['messages'][-1]
                if hasattr(last_message, 'content'):
                    return self._extract_code_from_output(last_message.content)
                elif isinstance(last_message, dict) and 'content' in last_message:
                    return self._extract_code_from_output(last_message['content'])
            
            # Check for implementation node output
            if 'implementation' in text and text['implementation']:
                impl = text['implementation']
                logger.debug(f"Found 'implementation' node with keys: {list(impl.keys())}")
                
                # Check if code is directly in the implementation
                if 'code' in impl and impl['code']:
                    code = impl['code']
                    if isinstance(code, str) and self._looks_like_dag(code):
                        logger.debug("Found valid DAG code in implementation['code']")
                        return code
                
                # Check if there's a result field with code
                if 'result' in impl and impl['result']:
                    result = impl['result']
                    if isinstance(result, str) and self._looks_like_dag(result):
                        logger.debug("Found valid DAG code in implementation['result']")
                        return result
                    elif isinstance(result, dict) and 'code' in result:
                        if self._looks_like_dag(result['code']):
                            logger.debug("Found valid DAG code in implementation['result']['code']")
                            return result['code']
            
            # Check for 'current_step' that might contain the code
            if 'current_step' in text and text['current_step']:
                logger.debug("Found 'current_step', checking for code")
                step_output = text['current_step']
                
                # Handle different formats of step_output
                if isinstance(step_output, dict):
                    # Check common fields that might contain the code
                    for field in ['output', 'code', 'implementation', 'content', 'result']:
                        if field in step_output and step_output[field]:
                            extracted = self._extract_code_from_output(step_output[field])
                            if extracted:
                                return extracted
                
                # If we get here, try to process the step_output directly
                return self._extract_code_from_output(step_output)
            
            # Check for 'implementation' in the root level
            if 'implementation' in text and text['implementation']:
                logger.debug("Found 'implementation' in root level")
                impl = text['implementation']
                if isinstance(impl, str) and self._looks_like_dag(impl):
                    return impl.strip()
            
            # If no direct matches found, try to find any string that looks like a DAG
            logger.debug("No direct matches found, searching for DAG patterns in the entire structure")
            for key, value in text.items():
                if isinstance(value, str) and self._looks_like_dag(value):
                    logger.debug(f"Found DAG-like content in field: {key}")
                    return value.strip()
                elif isinstance(value, dict):
                    # Recursively search in nested dictionaries
                    result = self._extract_code_from_output(value)
                    if result:
                        return result
            
            # As a last resort, convert the entire structure to string and look for code blocks
            text_str = str(text)
            if any(x in text_str for x in ['DAG(', 'from airflow', 'import DAG']):
                logger.debug("Found DAG-related keywords in string representation, attempting to extract code")
                code_blocks = re.findall(r'```(?:python)?\s*([\s\S]*?)```', text_str)
                for block in code_blocks:
                    if self._looks_like_dag(block):
                        return block.strip()
                
                # If no code blocks found but DAG keywords exist, try to extract the relevant part
                lines = text_str.split('\n')
                code_lines = []
                in_code = False
                
                for line in lines:
                    if any(x in line for x in ['DAG(', 'from airflow', 'import DAG']):
                        in_code = True
                    if in_code:
                        code_lines.append(line)
                        if line.strip().endswith(')') and 'DAG(' in line:
                            break
                
                if code_lines:
                    potential_code = '\n'.join(code_lines)
                    if self._looks_like_dag(potential_code):
                        return potential_code
            
            # If we get here, convert dict to string and proceed to the markdown extraction
            text = str(text)
        
        # Try to extract code from markdown code blocks
        code_block_patterns = [
            r'```(?:python)?\s*([\s\S]*?)```',  # ```python ... ``` or ``` ... ```
            r'```(?:python)?\n([\s\S]*?)\n```',  # More specific pattern
            r'<code>([\s\S]*?)</code>',  # HTML code blocks
            r'`([^`]+)`'  # Inline code
        ]
        
        for pattern in code_block_patterns:
            try:
                matches = re.findall(pattern, text, re.MULTILINE)
                if matches:
                    # Return the longest match that looks like a DAG
                    potential_code = max(matches, key=len).strip()
                    if self._looks_like_dag(potential_code):
                        logger.debug("Extracted DAG code from markdown code block")
                        return potential_code
            except Exception as e:
                logger.debug(f"Error extracting with pattern {pattern}: {str(e)}")
                continue
        
        # If we have a direct DAG definition, use that
        if self._looks_like_dag(text):
            logger.debug("Found direct DAG definition in text")
            return text.strip()
            
        # Fallback: extract Python-looking code
        if any(x in text for x in ['import airflow', 'from airflow', 'DAG(', 'with DAG']):
            logger.debug("Found Airflow-related keywords, attempting to extract code")
            lines = text.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for code block markers
                if line.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                    
                # If we're in a code block or the line looks like code
                if (in_code_block or 
                    any(line.startswith(x) for x in ['import ', 'from ', 'def ', 'class ', '@']) or
                    any(x in line for x in [' = ', 'import ', 'def ', 'class ', 'DAG('])):
                    code_lines.append(line)
            
            if code_lines:
                potential_code = '\n'.join(code_lines)
                if self._looks_like_dag(potential_code):
                    logger.debug("Extracted DAG code from Python-like text")
                    return potential_code
        
        logger.warning(f"Failed to extract valid DAG code from output. Content sample: {str(text)[:200]}...")
        return None
        
    def _looks_like_dag(self, code: Union[str, Dict, Any]) -> bool:
        """Check if the provided code looks like an Airflow DAG.
        
        This method checks for common patterns in Airflow DAGs to determine
        if the provided code is likely an Airflow DAG definition.
        
        Args:
            code: The code to check. Can be a string, dict, or other type.
            
        Returns:
            bool: True if the code looks like an Airflow DAG, False otherwise.
        """
        if not code:
            return False
            
        # If input is a dictionary, try to extract string content
        if isinstance(code, dict):
            # Check common fields that might contain DAG code
            for field in ['code', 'dag_code', 'implementation', 'content', 'output', 'result']:
                if field in code and code[field]:
                    if isinstance(code[field], str):
                        if self._looks_like_dag(code[field]):
                            return True
                    elif isinstance(code[field], dict):
                        if self._looks_like_dag(code[field]):
                            return True
            
            # If we have a 'status' field that indicates success, be more lenient
            if 'status' in code and code['status'] == 'completed' and 'code' in code:
                return True
                
            # If no code found in dict fields, convert to string and check
            code = str(code)
        
        # If it's not a string at this point, it's not a DAG
        if not isinstance(code, str):
            return False
            
        # If the string is too short, it's probably not a DAG
        if len(code.strip()) < 20:
            return False
            
        # If it's a string representation of a dict, try to extract the code
        if code.startswith('{') and code.endswith('}'):
            try:
                import json
                # Try with single quotes first (common in Python string representations)
                try:
                    code_dict = json.loads(code.replace("'", '"'))
                    return self._looks_like_dag(code_dict)
                except json.JSONDecodeError:
                    # If that fails, try with double quotes
                    code_dict = json.loads(code)
                    return self._looks_like_dag(code_dict)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
                
        # Check for common DAG patterns in the code
        code_lower = code.lower()
        
        # Required components for a valid DAG (relaxed to catch more cases)
        required_patterns = [
            r'(?:from\s+airflow\s+import\s+DAG|import\s+airflow\..*\bDAG\b|from\s+airflow\s+import\s+\*)',  # DAG import
            r'(?:DAG\s*\(|with\s+DAG\s*\()',  # DAG instantiation (both styles)
        ]
        
        # Check for required patterns
        for pattern in required_patterns:
            if not re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                logger.debug(f"DAG validation failed - missing required pattern: {pattern}")
                return False
        
        # Check for common DAG patterns (at least one should be present)
        dag_patterns = [
            r'default_args\s*=',
            r'dag\s*=',
            r'schedule_interval\s*=',
            r'start_date\s*=',
            r'dag_id\s*=',
            r'catchup\s*='
        ]
        
        if not any(re.search(pattern, code_lower, re.MULTILINE) for pattern in dag_patterns):
            return False
                
        # Check for at least one operator or task pattern
        operator_patterns = [
            r'from\s+airflow\.operators',
            r'from\s+airflow\.providers',
            r'from\s+airflow\.sensors',
            r'import\s+airflow\.operators',
            r'import\s+airflow\.providers',
            r'import\s+airflow\.sensors',
            r'>>\s*#',  # Task dependency
            r'>>\s*\w+',  # Task dependency
            r'\w+\s*>>',  # Task dependency
            r'\w+\s*=\s*\w+Operator\s*\('  # Operator instantiation
        ]
        
        if not any(re.search(pattern, code_lower, re.MULTILINE) for pattern in operator_patterns):
            # If no operator imports found, check for common task patterns
            task_patterns = [
                r'\w+\s*=\s*PythonOperator\s*\(',
                r'\w+\s*=\s*BashOperator\s*\('
            ]
            if not any(re.search(pattern, code_lower, re.MULTILINE) for pattern in task_patterns):
                return False
            
        # Additional validation: Look for task definitions and task dependencies
        has_tasks = (
            re.search(r'def\s+\w+\s*\(\s*[^)]*\s*\)\s*:', code_lower) or  # Python function
            re.search(r'\w+\s*=\s*\w+Operator\s*\(', code_lower) or  # Operator instantiation
            re.search(r'>>', code_lower)  # Task dependency
        )
        
        if not has_tasks:
            return False
            
        return True

    def _get_dag_id_from_code(self, code: str, default_id: str = "generated_dag") -> str:
        """Extracts the dag_id from the generated Python code."""
        patterns = [
            r'["\']dag_id["\']\s*:\s*["\']([\w_-]+)["\']',  # DAG(dag_id="...")
            r'DAG\s*\(\s*["\']([\w_-]+)["\']',              # DAG('...')
            r'dag_id\s*=\s*["\']([\w_-]+)["\']'             # dag_id = "..."
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            if matches:
                return matches[0]
                
        return default_id

    def _sanitize_input(self, requirements: str) -> str:
        """Ensure input is safe and reasonable size."""
        return requirements.strip()[:5000]

    def _validate_agent_output(self, raw_output: dict) -> AgentOutput:
        """Validate agent output matches expected schema."""
        try:
            return AgentOutput(**raw_output)
        except ValidationError as e:
            logger.error(f"Agent output validation failed: {e}")
            raise ValueError(f"Invalid agent output structure: {e.errors()}")

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
        """Generate DAG using LangGraph workflow.
        
        This method orchestrates the DAG generation process by:
        1. Initializing the workflow with the provided requirements
        2. Executing the workflow with proper state management
        3. Processing and validating the output
        4. Formatting the results
        
        Args:
            requirements (str): User requirements for the DAG in natural language
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - success: Boolean indicating if generation was successful
                - dag_code: Generated DAG code (if successful)
                - dag_id: Extracted DAG ID
                - analysis_report: Analysis of the generated DAG
                - validation_warnings: List of validation issues (if any)
                - execution_time: Total execution time in seconds
                - error: Error message (if generation failed)
        """
        requirements = self._sanitize_input(requirements)
        logger.info(f"Generating DAG for requirements: {requirements[:100]}...")
        start_time = time.time()
        
        try:
            # Initialize workflow with latest requirements
            try:
                self._initialize_workflow(requirements)
                logger.info("Workflow initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize workflow: {str(e)}")
                return {
                    "success": False,
                    "error": f"Failed to initialize workflow: {str(e)}",
                    "execution_time": time.time() - start_time
                }
            
            try:
                # Execute the workflow
                workflow_state = {
                    "requirements": requirements,
                    "messages": [HumanMessage(content=requirements)]
                }
                
                logger.info("Executing LangGraph workflow...")
                result = self.workflow.invoke(workflow_state)
                logger.info("Workflow execution completed")
                
                # Process workflow outputs
                logger.debug(f"Raw workflow result: {result}")
                
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
                
                # If no code found, try to extract from messages
                if not dag_code and 'messages' in result and result['messages']:
                    for msg in reversed(result['messages']):
                        if hasattr(msg, 'content') and isinstance(msg.content, str):
                            dag_code = self._extract_code_from_output(msg.content)
                            if dag_code:
                                break
                
                # If still no code, try to extract from the raw result
                if not dag_code:
                    dag_code = self._extract_code_from_output(result)
                
                # If we have code, clean it up
                if dag_code:
                    dag_code = dag_code.strip()
                    if dag_code.startswith('```python'):
                        dag_code = dag_code[9:].rsplit('```', 1)[0].strip()
                    elif dag_code.startswith('```'):
                        dag_code = dag_code[3:].rsplit('```', 1)[0].strip()
                
                # Generate a DAG ID if not found
                if not dag_id:
                    dag_id = f"generated_dag_{int(time.time())}"
                
                # Extract analysis report and perform validation
                analysis_report = result.get('analysis', 'No analysis report generated.')
                validation_errors = []
                if dag_code:
                    validation_errors = self._validate_dag_code(dag_code)
                
                logger.info(f"DAG generation completed in {time.time() - start_time:.2f}s")
                
                return {
                    "success": True,
                    "dag_code": dag_code,
                    "dag_id": dag_id,
                    "analysis_report": analysis_report,
                    "validation_warnings": validation_errors if validation_errors else None,
                    "execution_time": time.time() - start_time,
                    "debug_info": {
                        "dag_id": dag_id,
                        "code_length": len(dag_code) if dag_code else 0,
                        "validation_issues": len(validation_errors) if validation_errors else 0
                    }
                }
                
            except Exception as e:
                logger.error(f"Error during workflow execution: {str(e)}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Error during workflow execution: {str(e)}",
                    "execution_time": time.time() - start_time,
                    "error_type": type(e).__name__
                }
            
        except Exception as e:
            logger.error(f"Error generating DAG: {str(e)}")
            return self._handle_generation_error(e, requirements, start_time)

    def _validate_dag_code(self, dag_code: str) -> List[str]:
        """Validate the generated DAG code for common issues."""
        validation_errors = []
        
        required_patterns = [
            ("from airflow import DAG", "Missing required import: 'from airflow import DAG'"),
            ("DAG(", "Missing DAG instantiation"),
            ("Operator", "No Airflow operators found in DAG code")
        ]
        
        for pattern, error_msg in required_patterns:
            if pattern not in dag_code and pattern.lower() not in dag_code.lower():
                validation_errors.append(error_msg)
                
        return validation_errors

    def _handle_generation_error(self, error: Exception, requirements: str, start_time: float) -> Dict[str, Any]:
        """Handle generation errors and provide user-friendly feedback."""
        error_type = type(error).__name__
        error_message = str(error)
        
        error_categories = {
            "API": ["API key", "authentication"],
            "Timeout": ["timeout"],
            "Rate Limit": ["rate limit", "quota"],
            "Validation": ["validation"]
        }
        
        error_category = "Unknown error"
        suggestion = "Please try again or modify your requirements."
        
        for category, keywords in error_categories.items():
            if any(keyword in error_message.lower() for keyword in keywords):
                error_category = f"{category} Error"
                
                if category == "API":
                    suggestion = "Please check your API key and ensure it's valid."
                elif category == "Timeout":
                    suggestion = "The request took too long. Try simplifying your requirements."
                elif category == "Rate Limit":
                    suggestion = "You've exceeded the API rate limit. Please wait and try again."
                elif category == "Validation":
                    suggestion = "There was an issue with the generated DAG structure."
                
                break
                
        return {
            "success": False,
            "error": error_message,
            "error_type": error_type,
            "error_category": error_category,
            "suggestion": suggestion,
            "input_sample": requirements[:500],
            "execution_time": time.time() - start_time
        }

    def deploy_and_test_dag(self, dag_code: Optional[str], dag_id: Optional[str]) -> Dict[str, Any]:
        """Deploy and test the generated DAG in Airflow.
        
        Args:
            dag_code (str): The DAG code to deploy.
            dag_id (str, optional): The DAG ID. If not provided, will be extracted from code.
            
        Returns:
            Dict[str, Any]: Deployment and test results.
        """
        if not dag_code:
            return {"deployed": False, "error": "No DAG code provided"}
            
        try:
            dag_id = dag_id or self._get_dag_id_from_code(dag_code)
            logger.info(f"Deploying DAG with ID: {dag_id}")
            
            # Ensure workflow is initialized
            if not hasattr(self, 'docker_manager') or not self.docker_manager:
                logger.info("Initializing workflow before deployment...")
                self._initialize_workflow("")
                
            # Ensure Airflow is running
            if not hasattr(self, 'docker_manager') or not self.docker_manager:
                return {"deployed": False, "error": "Failed to initialize Docker manager"}
                
            container_status = self.docker_manager.container_status()
            logger.info(f"Container status: {container_status}")
            
            if not container_status.get("running", False):
                logger.info("Starting Airflow...")
                start_result = self.docker_manager.start_airflow()
                logger.info(f"Start Airflow result: {start_result}")
                
                if not start_result.get("success", False):
                    return {
                        "deployed": False,
                        "error": f"Failed to start Airflow: {start_result.get('error', 'Unknown error')}"
                    }
            
            # Deploy the DAG
            logger.info(f"Deploying DAG {dag_id}...")
            deploy_result = self.docker_manager.deploy_dag(dag_code, dag_id)
            logger.info(f"Deploy result: {deploy_result}")
            
            if not deploy_result.get("deployed", False):
                return deploy_result
                
            # Test the DAG
            logger.info(f"Validating DAG {dag_id}...")
            test_result = self.docker_manager.validate_dag(dag_id)
            logger.info(f"Validation result: {test_result}")
            
            return {
                "deployed": True,
                "dag_id": dag_id,
                "test_success": test_result.get("valid", False),
                "test_result": test_result.get("result", None),
                "test_details": None,
                "error": test_result.get("error", None)
            }
            
        except Exception as e:
            error_msg = f"Error during DAG deployment: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"deployed": False, "error": error_msg}

    def list_available_containers(self) -> List[Dict[str, str]]:
        """List all available Docker containers."""
        return self.docker_manager.list_running_containers()

    def reload_codebase_context(self) -> bool:
        """Reload the codebase context."""
        try:
            self.codebase_context.load_code_context()
            return True
        except Exception as e:
            logger.error(f"Error reloading codebase context: {str(e)}")
            return False
            
    def get_codebase_components(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all code components from the codebase."""
        return self.codebase_context.get_code_components()
    
    def search_codebase(self, query: str, max_results: int = 5, semantic: bool = True) -> Dict[str, Any]:
        """Search the codebase for files matching the query."""
        return self.codebase_context.search_context(query, max_results, semantic)