import warnings
warnings.filterwarnings('ignore')
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnablePassthrough

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from .tools import CodebaseTools
from ..core.context import CodebaseContext
from ..core.docker_manager import DockerManager

logger = logging.getLogger(__name__)

class WorkflowState(BaseModel):
    """State for the DAG generation workflow."""
    messages: List[HumanMessage] = Field(default_factory=list)
    requirements: str = ""
    analysis: Optional[Dict[str, Any]] = None
    design: Optional[Dict[str, Any]] = None
    implementation: Optional[Dict[str, Any]] = None
    review: Optional[Dict[str, Any]] = None
    current_step: str = "analysis"

def get_llm(api_key: str, llm_provider: str = "google", temperature: float = 0.2):
    """Create an LLM instance using Google's Gemini model."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",  # Using Gemini 2.5 Pro model
        google_api_key=api_key,
        temperature=temperature,
        model_kwargs={
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
    )

def create_analysis_node(state: WorkflowState) -> Dict[str, Any]:
    """Node for analyzing requirements and codebase."""
    logger.info("Running analysis node")
    
    # Create prompt for analysis
    prompt = f"""
    Analyze these DAG requirements:
    {state.requirements}
    
    Search the codebase for relevant patterns and components.
    Provide a detailed analysis with:
    1. Key requirements and constraints
    2. Existing reusable components
    3. Data sources and sinks
    4. Required transformations
    5. Potential challenges
    6. Airflow-specific considerations
    """
    
    # In a real implementation, this would call an LLM with tools
    return {
        "messages": [HumanMessage(content=prompt)],
        "analysis": {"status": "completed", "report": "Analysis report..."},
        "current_step": "design",
        "requirements": state.requirements
    }

def create_design_node(state: WorkflowState) -> Dict[str, Any]:
    """Node for designing DAG architecture."""
    logger.info("Running design node")
    
    prompt = f"""
    Design DAG architecture based on:
    {state.analysis['report']}
    
    Include:
    1. Task dependencies diagram
    2. Operator selections
    3. Configuration parameters
    4. Error handling strategies
    5. Resource considerations
    """
    
    return {
        "messages": [HumanMessage(content=prompt)],
        "design": {"status": "completed", "document": "Design document..."},
        "current_step": "implementation",
        "requirements": state.requirements
    }

def create_implementation_node(state: WorkflowState, llm: Any) -> Dict[str, Any]:
    """Node for implementing DAG code."""
    logger.info("Running implementation node")
    
    prompt = f"""
    You are an expert Airflow DAG developer. Your task is to write a Python script for an Airflow DAG that precisely meets the user's requirements.

    **Original Requirements:**
    "{state.requirements}"

    **Instructions:**
    1.  **Strictly Adhere to Requirements:** Generate a DAG that directly implements the user's request. Do not add extra features, tasks, or complexity unless explicitly asked. For simple requests, the DAG should be simple.
    2.  **Complete and Runnable Code:** The script must be a single, complete Python file. It must include:
        - All necessary imports.
        - A `DAG` object instantiation.
        - All specified tasks.
        - Task dependencies set correctly.
    3.  **Use Standard Practices:**
        - Include `default_args` for the DAG.
        - Set a `dag_id` that is descriptive and based on the requirements.
        - Use the correct `schedule_interval`.
    4.  **Output Format:**
        - Return ONLY the Python code for the DAG.
        - The code must be enclosed in a single markdown code block like this: ```python ... ```.
        - Do not include any text, explanation, or notes before or after the code block.

    Based on the requirements, create the DAG now.
    """
    
    try:
        # Generate DAG code using the LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Extract code from the response
        if hasattr(response, 'content'):
            code = response.content
            # Clean up the response to extract just the code block
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0].strip()
            elif '```' in code:
                code = code.split('```')[1].split('```')[0].strip()
        else:
            code = str(response)
        
        # Generate a DAG ID from the requirements
        dag_id = "generated_dag_" + str(hash(state.requirements))[:8]
        
        return {
            "messages": [HumanMessage(content=f"Generated DAG with ID: {dag_id}")],
            "implementation": {
                "status": "completed",
                "code": code,
                "dag_id": dag_id
            },
            "current_step": "review"
        }
        
    except Exception as e:
        logger.error(f"Error generating DAG code: {str(e)}")
        return {
            "messages": [HumanMessage(content=f"Error generating DAG: {str(e)}")],
            "implementation": {
                "status": "error",
                "error": str(e),
                "code": "# Error generating DAG code. Please check the logs.",
                "dag_id": "error_dag"
            },
            "current_step": "error"
        }

def create_review_node(state: WorkflowState) -> Dict[str, Any]:
    """Node for reviewing DAG code."""
    logger.info("Running review node")
    
    prompt = f"""
    Review this DAG implementation:
    {state.implementation['code']}
    
    Check for:
    1. Code quality
    2. Potential bugs
    3. Performance issues
    4. Security concerns
    5. Requirement compliance
    """
    
    return {
        "messages": [HumanMessage(content=prompt)],
        "review": {
            "status": "completed",
            "report": "Review report...",
            "assessment": "PASS"
        },
        "current_step": "end"
    }

def should_continue(state: WorkflowState) -> str:
    """Determine next step in workflow."""
    return state.current_step

def create_dag_generation_workflow(
    api_key: str,
    codebase_path: str,
    airflow_container_name: str = "airflow-webserver-1",
    llm_provider: str = "google",
    airflow_compose_file: Optional[str] = None,
    use_embeddings: bool = True,
    embedding_provider: str = "local",
    requirements: str = ""
) -> Tuple[StateGraph, DockerManager, CodebaseContext]:
    """Create LangGraph workflow for DAG generation."""
    
    # Initialize components
    docker_manager = DockerManager(
        container_name=airflow_container_name,
        compose_file=airflow_compose_file
    )
    
    codebase_context = CodebaseContext(
        codebase_path=codebase_path,
        use_embeddings=use_embeddings,
        embedding_provider=embedding_provider,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2" if embedding_provider == "local" else "text-embedding-3-small"
    )
    
    # Create the workflow
    workflow = StateGraph(WorkflowState)
    
    # Initialize LLM
    llm = get_llm(api_key, llm_provider)
    
    # Add nodes with LLM where needed
    workflow.add_node("analyze", create_analysis_node)
    workflow.add_node("design", create_design_node)
    workflow.add_node("implement", lambda state: create_implementation_node(state, llm))
    workflow.add_node("review", create_review_node)
    
    # Define edges
    workflow.add_edge("analyze", "design")
    workflow.add_edge("design", "implement")
    workflow.add_edge("implement", "review")
    workflow.add_edge("review", END)
    
    # Set entry point
    workflow.set_entry_point("analyze")
    
    # Add conditional edges if needed
    # workflow.add_conditional_edges(...)
    
    # Compile the workflow
    compiled_workflow = workflow.compile()
    
    # Initialize with requirements
    compiled_workflow.initial_state = WorkflowState(requirements=requirements)
    
    return compiled_workflow, docker_manager, codebase_context
