from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

class CodeSearchInput(BaseModel):
    """Input schema for code search tool."""
    query: str = Field(description="Natural language query to search codebase")
    max_results: int = Field(default=5, description="Maximum results to return")

class FileReadInput(BaseModel):
    """Input schema for file read tool."""
    file_path: str = Field(description="Relative path to file in codebase")

class CodebaseTools:
    """Collection of tools for interacting with codebase using LangChain."""
    
    def __init__(self, codebase_context):
        """Initialize with codebase context."""
        self.context = codebase_context
        
        # Define tools
        self.tools = [
            self.code_search_tool(),
            self.file_read_tool()
        ]
        
    def code_search_tool(self) -> BaseTool:
        """Tool for semantic search through codebase."""
        class CodeSearchTool(BaseTool):
            name = "code_search"
            description = "Search codebase for relevant files/code"
            args_schema = CodeSearchInput
            
            def _run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
                return self.codebase_context.search_context(
                    query, 
                    max_results=max_results, 
                    semantic=True
                )
                
        tool = CodeSearchTool()
        tool.codebase_context = self.context
        return tool
        
    def file_read_tool(self) -> BaseTool:
        """Tool for reading file contents."""
        class FileReadTool(BaseTool):
            name = "file_read" 
            description = "Read contents of a file"
            args_schema = FileReadInput
            
            def _run(self, file_path: str) -> Dict[str, Any]:
                content = self.codebase_context.read_file(file_path)
                return {
                    "file_path": file_path,
                    "content": content if content else "File not found"
                }
                
        tool = FileReadTool()
        tool.codebase_context = self.context
        return tool

    def get_tools(self) -> List[BaseTool]:
        """Get all available tools."""
        return self.tools

class CodebaseWorkflow:
    """LangGraph workflow for codebase interactions."""
    
    def __init__(self, tools: CodebaseTools):
        """Initialize with codebase tools."""
        self.tools = tools
        
    def create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for codebase tasks."""
        
        # Define workflow state
        class WorkflowState(BaseModel):
            messages: List[HumanMessage]
            tool_results: Dict[str, Any] = {}
            
        # Define nodes
        def search_node(state: WorkflowState):
            last_msg = state.messages[-1].content
            result = self.tools.code_search_tool()._run(last_msg)
            return {"tool_results": {"search": result}}
            
        def read_node(state: WorkflowState):
            if "search" in state.tool_results:
                first_result = next(iter(
                    state.tool_results["search"]["results"].items()
                ))
                result = self.tools.file_read_tool()._run(first_result[0])
                return {"tool_results": {"read": result}}
            return {"tool_results": {}}
            
        # Build graph
        workflow = StateGraph(WorkflowState)
        workflow.add_node("search", search_node)
        workflow.add_node("read", read_node)
        
        # Define edges
        workflow.add_edge("search", "read")
        workflow.add_edge("read", END)
        
        # Set entry point
        workflow.set_entry_point("search")
        
        return workflow.compile()