# AeroMind DAG Architect


An intelligent AI agent that leverages advanced language models to automatically generate, deploy, and test Apache Airflow DAGs from natural language requirements. This autonomous agent system is built with LangChain, LangGraph, and Google's Generative AI to provide intelligent workflow automation.

## 🚀 Features

- **AI Agent Architecture**: Autonomous agent system that intelligently plans and executes DAG generation workflows
- **Natural Language Processing**: Convert plain English requirements into production-ready Airflow DAGs
- **Intelligent Context Understanding**: Uses semantic search and memory to understand your existing codebase
- **Automated Deployment**: Direct deployment to Airflow with validation and testing
- **Interactive Web UI**: User-friendly Streamlit interface for agent interaction
- **Multi-Agent Coordination**: Leverages LangGraph for complex workflow orchestration
- **Docker Integration**: Seamless interaction with Airflow running in Docker

## 📋 Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- A Google API key for Gemini Pro
- Apache Airflow 2.7+ (automatically set up via Docker)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aeromind-dag-architect
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file with:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   AIRFLOW_UID=50000
   FERNET_KEY=your_fernet_key
   POSTGRES_USER=airflow
   POSTGRES_PASSWORD=airflow
   POSTGRES_DB=airflow
   ```

4. **Start Airflow**
   ```bash
   docker-compose up -d
   ```

## 🖥️ Usage

1. **Start the Web UI**
   ```bash
   python main.py
   ```

2. **Interact with the AI Agent**
   - Open your browser to http://localhost:8501
   - Enter your requirements in natural language
   - The agent will analyze, plan, and generate your DAG

3. **Deploy and Test**
   - Review the generated DAG code
   - Click "Deploy DAG" to deploy to Airflow
   - Access Airflow UI at http://localhost:8080 (default credentials: airflow/airflow)

## 🏗️ Architecture

The AI agent system consists of several key components:

- `client.py`: Main agent client interface for DAG generation and deployment
- `context.py`: Manages codebase context with semantic search and memory capabilities
- `docker_manager.py`: Handles Docker operations for Airflow integration
- `langgraph_setup.py`: Sets up LangGraph workflow orchestration for multi-agent coordination
- `tools.py`: LangChain tools for intelligent codebase interaction
- `main.py`: Streamlit web interface for agent communication

## 🔧 Configuration

Configure the system through:
- Environment variables in `.env`
- Direct API key input in the web UI
- Advanced options in the UI sidebar

Available configuration options:
- AI Agent Provider (currently Google Gemini)
- Semantic Search and Memory settings
- Airflow container configuration
- Codebase path for agent context understanding

## 📝 Example

Let the AI agent generate a DAG with this sample requirement:
```text
Create a DAG that:
1. Runs every day at 8 AM
2. Extracts user data from PostgreSQL
3. Cleans and transforms the data
4. Loads results into BigQuery
5. Sends success notification to Slack
```

The agent will intelligently break down this requirement, analyze dependencies, and generate a complete, production-ready DAG.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Documentation

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [LangChain Documentation](https://python.langchain.com/docs/)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
