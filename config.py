"""Configuration module for loading and managing environment variables."""
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to manage environment variables."""
    
    # Google Gemini Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Airflow Configuration
    AIRFLOW_UID: int = int(os.getenv("AIRFLOW_UID", "50000"))
    FERNET_KEY: str = os.getenv("FERNET_KEY", "")
    
    # Database Configuration
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "airflow")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "airflow")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "airflow")
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost/{POSTGRES_DB}"
    )
    
    # Airflow Web Server
    AIRFLOW_WEBSERVER_URL: str = os.getenv(
        "AIRFLOW__WEBSERVER__BASE_URL", 
        "http://localhost:8080"
    )
    
    # Optional: OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Optional: HuggingFace Configuration
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        required_vars = [
            ("GOOGLE_API_KEY", cls.GOOGLE_API_KEY),
            ("FERNET_KEY", cls.FERNET_KEY),
        ]
        
        missing = [name for name, value in required_vars if not value]
        if missing:
            print(f"Error: Missing required environment variables: {', '.join(missing)}")
            return False
        return True

# Create a single instance of the configuration
config = Config()
