import subprocess
import logging
import tempfile
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class DockerManager(BaseModel):
    """Manages Docker operations for Airflow DAG deployment and testing.
    
    Integrates with LangChain/LangGraph workflows for Airflow DAG management.
    
    Attributes:
        container_name: Name of the target Airflow container
        compose_file: Path to docker-compose.yml (optional)
        compose_project: Docker Compose project name
        docker_available: Flag indicating Docker availability
    """
    container_name: str = Field(default="airflow-webserver-1")
    compose_file: Optional[str] = Field(default=None)
    compose_project: str = Field(default="airflow")
    docker_available: bool = Field(default=True, exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
    def __init__(self, **data):
        super().__init__(**data)
        self.docker_available = self._check_docker_available()
        logger.info(f"Initialized DockerManager for container: {self.container_name}")

    def _check_docker_available(self) -> bool:
        """Verify Docker daemon accessibility."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            logger.debug("Docker is available")
            return True
        except Exception as e:
            logger.error(f"Docker check failed: {str(e)}")
            return False

    def _execute_command(self, cmd: List[str], timeout: int = 30) -> Tuple[bool, str]:
        """Execute shell command with error handling."""
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        except Exception as e:
            return False, str(e)

    def container_status(self) -> Dict[str, Any]:
        """Check container status and health."""
        if not self.docker_available:
            return {"available": False, "error": "Docker not available"}

        # Check if container exists and is running
        success, output = self._execute_command([
            "docker", "ps", 
            "--filter", f"name={self.container_name}",
            "--format", "{{.ID}}|{{.Status}}"
        ])

        if not success or not output.strip():
            return {"running": False, "exists": False}

        container_id, status = output.strip().split("|")
        is_running = "Up" in status

        # Check Airflow health endpoint if running
        health = None
        if is_running:
            health_success, health_output = self._execute_command([
                "docker", "exec", self.container_name,
                "curl", "-s", "http://localhost:8080/health"
            ])
            if health_success:
                try:
                    health = json.loads(health_output)
                except json.JSONDecodeError:
                    health = {"status": "unavailable"}

        return {
            "running": is_running,
            "exists": True,
            "status": status,
            "health": health
        }

    def start_airflow(self) -> Dict[str, Any]:
        """Start Airflow services using Docker Compose."""
        status = self.container_status()
        if status["running"]:
            return {"success": True, "status": "already_running"}

        compose_cmd = ["docker-compose"]
        if self.compose_file:
            compose_cmd.extend(["-f", self.compose_file])
        compose_cmd.extend(["-p", self.compose_project, "up", "-d"])

        success, output = self._execute_command(compose_cmd, timeout=120)
        if not success:
            return {"success": False, "error": output}

        # Wait for healthy status
        for _ in range(12):  # 60 seconds total
            time.sleep(5)
            status = self.container_status()
            if status.get("health", {}).get("status") == "healthy":
                return {"success": True, "status": "started"}

        return {"success": True, "warning": "slow_start"}

    def deploy_dag(self, dag_code: str, dag_id: str) -> Dict[str, Any]:
        """Deploy DAG to Airflow container."""
        status = self.container_status()
        if not status["running"]:
            start_result = self.start_airflow()
            if not start_result["success"]:
                return {"deployed": False, "error": start_result.get("error")}

        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
                tmp.write(dag_code)
                tmp_path = tmp.name

            # Copy to container
            dest_path = f"/opt/airflow/dags/{dag_id}.py"
            success, output = self._execute_command([
                "docker", "cp", tmp_path, 
                f"{self.container_name}:{dest_path}"
            ])
            os.unlink(tmp_path)

            if not success:
                return {"deployed": False, "error": output}

            return {
                "deployed": True,
                "dag_id": dag_id,
                "path": dest_path
            }
        except Exception as e:
            return {"deployed": False, "error": str(e)}

    def validate_dag(self, dag_id: str) -> Dict[str, Any]:
        """Validate DAG syntax and structure."""
        status = self.container_status()
        if not status["running"]:
            return {"valid": False, "error": "container_not_running"}

        # Check DAG appears in list
        success, output = self._execute_command([
            "docker", "exec", self.container_name,
            "airflow", "dags", "list"
        ])
        if not success:
            return {"valid": False, "error": output}

        if dag_id not in output:
            return {"valid": False, "error": "dag_not_found"}

        # Test parse the DAG
        test_success, test_output = self._execute_command([
            "docker", "exec", self.container_name,
            "python", "-c", 
            f"from airflow.models import DagBag; "
            f"dag_bag = DagBag(); "
            f"print(dag_bag.get_dag('{dag_id}'))"
        ])

        return {
            "valid": test_success,
            "result": test_output if test_success else None,
            "error": None if test_success else test_output
        }

    def test_dag_syntax(self, dag_id: str) -> Dict[str, Any]:
        """Test DAG syntax and structure.
        
        Args:
            dag_id: The ID of the DAG to test
            
        Returns:
            Dict containing test results and any errors
        """
        # First check if DAG exists
        list_result = self.list_dags()
        if not list_result["success"]:
            return {"valid": False, "error": f"Failed to list DAGs: {list_result.get('error')}"}
            
        # Check if DAG exists in the list
        dag_exists = any(dag.get("dag_id") == dag_id for dag in list_result.get("dags", []))
        if not dag_exists:
            return {"valid": False, "error": f"DAG '{dag_id}' not found in Airflow"}
            
        # Test DAG parsing
        success, output = self._execute_command([
            "docker", "exec", self.container_name,
            "python", "-c", 
            f"""
            from airflow.models import DagBag
            import sys
            
            dag_bag = DagBag()
            if '{dag_id}' not in dag_bag.dags:
                print('DAG not found in DagBag')
                sys.exit(1)
                
            dag = dag_bag.get_dag('{dag_id}')
            if not dag:
                print('Failed to load DAG')
                sys.exit(1)
                
            print('DAG loaded successfully')
            print(f'DAG tasks: {[t.task_id for t in dag.tasks]}')
            """
        ])
        
        if not success:
            return {
                "valid": False,
                "error": f"DAG validation failed: {output}",
                "details": output
            }
            
        return {
            "valid": True,
            "message": "DAG validated successfully",
            "details": output
        }
        
    def list_dags(self) -> Dict[str, Any]:
        """List available DAGs in container."""
        status = self.container_status()
        if not status["running"]:
            return {"success": False, "error": "container_not_running"}

        success, output = self._execute_command([
            "docker", "exec", self.container_name,
            "airflow", "dags", "list", "-o", "json"
        ])

        if not success:
            return {"success": False, "error": output}

        try:
            dags = json.loads(output)
            return {"success": True, "dags": dags}
        except json.JSONDecodeError:
            return {"success": False, "error": "invalid_json_output"}