import warnings
warnings.filterwarnings('ignore')
import os
import logging
import re
import ast
from typing import Dict, Optional, List, Any, Tuple, Union
import numpy as np

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class CodebaseContext:
    """Manages codebase context with semantic search capabilities using HuggingFace embeddings.
    
    Attributes:
        codebase_path (str): Path to the codebase directory.
        _context (Dict[str, Dict]): Loaded code context with file metadata.
        _embeddings (Dict[str, np.ndarray]): File embeddings for semantic search.
        _embedding_model (Embeddings): HuggingFace embedding model instance.
    """

    def __init__(self, 
                 codebase_path: str, 
                 use_embeddings: bool = True,
                 embedding_provider: str = "local",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 **kwargs):
        """Initialize the CodebaseContext.
        
        Args:
            codebase_path (str): Path to the codebase directory.
            use_embeddings (bool): Whether to enable semantic search.
            embedding_provider (str): The embedding provider to use ('google', 'openai', or 'local').
            embedding_model_name (str): Name of the embedding model to use.
            **kwargs: Additional keyword arguments (ignored).
        """
        self.codebase_path = os.path.abspath(codebase_path)
        self.use_embeddings = use_embeddings
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        
        self._context = None
        self._embeddings = None
        self._embedding_model = None
        
        if self.use_embeddings:
            self._init_embedding_model()
            
        self.load_context()

    def _init_embedding_model(self):
        """Initialize the HuggingFace embedding model."""
        try:
            if self.embedding_provider == 'local':
                self._embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
                    encode_kwargs={'normalize_embeddings': True}
                )
            else:
                # For other providers, we'll use a placeholder for now
                # In a real implementation, you would initialize the appropriate client here
                logger.warning(f"Embedding provider '{self.embedding_provider}' not fully implemented. Using local embeddings.")
                self._embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            logger.info(f"Initialized HuggingFace embeddings: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            self.use_embeddings = False

    def load_context(self):
        """Load or reload the codebase context."""
        logger.info(f"Loading code context from: {self.codebase_path}")
        self._context = {}
        ignore_dirs = {'.venv', 'venv', 'env', '__pycache__'}

        for root, dirs, files in os.walk(self.codebase_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    file_path = os.path.join(root, file)
                    self._process_file(file_path)

        logger.info(f"Loaded {len(self._context)} Python files")
        
        if self.use_embeddings:
            self._compute_embeddings()

    def _process_file(self, file_path: str):
        """Process and store a single Python file's context."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                rel_path = os.path.relpath(file_path, self.codebase_path)
                
                classes, functions = self._extract_code_components(content)
                
                self._context[file_path] = {
                    "content": content,
                    "relative_path": rel_path,
                    "classes": classes,
                    "functions": functions
                }
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {str(e)}")

    def _extract_code_components(self, content: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract classes and functions using AST with regex fallback."""
        try:
            return self._extract_with_ast(content)
        except Exception:
            return self._extract_with_regex(content)

    def _extract_with_ast(self, content: str) -> Tuple[List[Dict], List[Dict]]:
        """AST-based extraction for accurate code analysis."""
        classes = []
        functions = []
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "lineno": node.lineno
                })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append({
                    "name": node.name,
                    "params": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                    "lineno": node.lineno,
                    "async": isinstance(node, ast.AsyncFunctionDef)
                })
        return classes, functions

    def _extract_with_regex(self, content: str) -> Tuple[List[Dict], List[Dict]]:
        """Regex fallback for code component extraction."""
        classes = []
        functions = []
        
        # Class pattern: class MyClass(OptionalParent):
        class_pattern = r'class\s+(\w+)\s*(?:\([^)]*\))?\s*:'
        for match in re.finditer(class_pattern, content):
            classes.append({"name": match.group(1)})
            
        # Function pattern: def my_func(params):
        func_pattern = r'def\s+(\w+)\s*\(([^)]*)\)\s*:'
        for match in re.finditer(func_pattern, content):
            functions.append({
                "name": match.group(1),
                "params": match.group(2).split(',')
            })
            
        return classes, functions

    def _compute_embeddings(self):
        """Compute embeddings for all files in context."""
        if not self._embedding_model:
            return
            
        logger.info("Computing file embeddings...")
        self._embeddings = {}
        
        for file_path, data in self._context.items():
            try:
                # Create semantic summary of the file
                summary = self._create_file_summary(data)
                embedding = self._embedding_model.embed_query(summary)
                self._embeddings[file_path] = np.array(embedding)
            except Exception as e:
                logger.warning(f"Failed to embed {file_path}: {str(e)}")
                
        logger.info(f"Computed embeddings for {len(self._embeddings)} files")

    def _create_file_summary(self, file_data: Dict) -> str:
        """Create a semantic summary of a file for embedding."""
        summary = f"File: {file_data['relative_path']}\n"
        
        if file_data.get("classes"):
            summary += "\nClasses:\n" + "\n".join(
                f"- {cls['name']}" for cls in file_data["classes"]
            )
            
        if file_data.get("functions"):
            summary += "\nFunctions:\n" + "\n".join(
                f"- {fn['name']}({', '.join(fn.get('params', []))})" 
                for fn in file_data["functions"]
            )
            
        return summary

    def search_context(self, 
                      query: str, 
                      max_results: int = 5, 
                      semantic: bool = True) -> Dict[str, Any]:
        """Search the codebase with semantic or keyword search.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            semantic: Whether to use semantic search
            
        Returns:
            Dictionary with search results
        """
        if semantic and self.use_embeddings:
            return self._semantic_search(query, max_results)
        return self._keyword_search(query, max_results)

    def _semantic_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Perform semantic search using embeddings."""
        try:
            query_embed = np.array(self._embedding_model.embed_query(query))
            similarities = {
                file: np.dot(query_embed, embed) / 
                (np.linalg.norm(query_embed) * np.linalg.norm(embed))
                for file, embed in self._embeddings.items()
            }
            
            sorted_results = sorted(
                similarities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:max_results]
            
            return {
                "query": query,
                "results": [{
                    "file_path": file,
                    "relative_path": self._context[file]["relative_path"],
                    "similarity": float(similarity),
                    "content": self._context[file]["content"][:500] + "..."  # Preview
                } for file, similarity in sorted_results]
            }
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return self._keyword_search(query, max_results)

    def _keyword_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Perform simple keyword search."""
        results = []
        query_lower = query.lower()
        
        for file_path, data in self._context.items():
            content_lower = data["content"].lower()
            if query_lower in content_lower:
                results.append({
                    "file_path": file_path,
                    "relative_path": data["relative_path"],
                    "matches": content_lower.count(query_lower),
                    "content": self._highlight_query(data["content"], query)
                })
                
        results.sort(key=lambda x: -x["matches"])
        return {
            "query": query,
            "results": results[:max_results]
        }

    def _highlight_query(self, text: str, query: str) -> str:
        """Highlight query matches in text."""
        if not query:
            return text[:500] + "..." if len(text) > 500 else text
            
        # Find all matches (case insensitive)
        matches = list(re.finditer(re.escape(query), text, re.IGNORECASE))
        if not matches:
            return text[:500] + "..."
            
        # Get context around first match
        first_match = matches[0]
        start = max(0, first_match.start() - 50)
        end = min(len(text), first_match.end() + 50)
        snippet = text[start:end]
        
        # Highlight the match
        highlighted = re.sub(
            f"({re.escape(query)})", 
            r"**\1**", 
            snippet, 
            flags=re.IGNORECASE
        )
        return f"...{highlighted}..."

    def get_code_components(self) -> Dict[str, List[Dict]]:
        """Get all classes and functions in the codebase."""
        components = {"classes": [], "functions": []}
        
        for file_path, data in self._context.items():
            components["classes"].extend({
                **cls, 
                "file": data["relative_path"]
            } for cls in data.get("classes", []))
            
            components["functions"].extend({
                **fn,
                "file": data["relative_path"]
            } for fn in data.get("functions", []))
            
        return components

    def read_file(self, file_path: str) -> Optional[str]:
        """Read file content from context or disk."""
        # Check in loaded context first
        if file_path in self._context:
            return self._context[file_path]["content"]
            
        # Try by relative path
        for path, data in self._context.items():
            if data["relative_path"] == file_path:
                return data["content"]
                
        # Fallback to direct file read
        try:
            full_path = os.path.join(self.codebase_path, file_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"File read failed: {str(e)}")
            return None
