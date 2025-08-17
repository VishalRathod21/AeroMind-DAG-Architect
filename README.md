# 🌪️ AeroMind DAG Architect

<div align="center">

![AeroMind Banner](https://img.shields.io/badge/AeroMind-DAG%20Architect-blue?style=for-the-badge&logo=apache-airflow&logoColor=white)

**AI-Powered Apache Airflow DAG Generation with Modern UI**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green?style=flat-square)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=flat-square&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## ✨ **What's New in v2.0**

🎨 **Modern Glassmorphism UI** - Beautiful gradient backgrounds with frosted glass effects  
🧠 **AI Thinking Visualization** - Real-time typewriter effect showing AI reasoning process  
📦 **Modular Architecture** - Clean `src/` package structure with separation of concerns  
⚡ **Enhanced UX** - Colorful animations, hover effects, and smooth transitions  
🔧 **Better Organization** - Dedicated modules for config, core, agents, and services  

---

## 🚀 **Features**

### **🤖 AI-Powered Generation**
- **Natural Language to DAG**: Convert plain English requirements into production-ready Airflow DAGs
- **Intelligent Context Understanding**: Semantic search and memory for codebase awareness
- **Multi-Agent Coordination**: LangGraph workflows for complex orchestration
- **Real-time Thinking Process**: Watch the AI analyze, design, implement, and review your DAGs

### **🎨 Modern User Interface**
- **Glassmorphism Design**: Beautiful frosted glass containers with gradient backgrounds
- **Typewriter Effects**: Real-time streaming of AI thoughts and reasoning
- **Colorful Animations**: Smooth hover effects and transitions throughout
- **Responsive Layout**: Clean, modern interface optimized for all screen sizes

### **🔧 Advanced Functionality**
- **Automated Deployment**: Direct deployment to Airflow with validation and testing
- **Docker Integration**: Seamless interaction with Airflow containers
- **Error Handling**: Comprehensive error management with user-friendly feedback
- **Configuration Management**: Environment-based settings with UI controls

---

## 📁 **Project Structure**

```
🌪️ AeroMind-DAG-Architect/
├── 🎨 app.py                    # Modern Streamlit UI with AI visualization
├── 📁 src/                      # Main source package
│   ├── 📁 config/              # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py         # Environment variables & settings
│   ├── 📁 core/                # Core functionality
│   │   ├── __init__.py
│   │   ├── context.py          # Codebase context & semantic search
│   │   └── docker_manager.py   # Docker & Airflow integration
│   ├── 📁 agents/              # AI agents & tools
│   │   ├── __init__.py
│   │   ├── tools.py            # LangChain tools for codebase interaction
│   │   └── workflow.py         # LangGraph workflow orchestration
│   └── 📁 services/            # Service layer
│       ├── __init__.py
│       └── dag_generator.py    # DAG generation client interface
├── 📁 dags/                    # Generated Airflow DAGs output
│   └── .gitkeep
├── 📋 requirements.txt         # Python dependencies
├── 🐳 docker-compose.yml       # Airflow Docker setup
├── 🔧 pyproject.toml          # Project configuration
└── 📖 README.md               # This file
```

---

## 📋 **Prerequisites**

- **Python 3.10+** - Modern Python with latest features
- **Docker & Docker Compose** - For Airflow container management
- **Google API Key** - For Gemini Pro AI model access
- **8GB+ RAM** - Recommended for optimal performance

---

## 🛠️ **Quick Start**

### **1. Clone & Setup**
```bash
git clone https://github.com/VishalRathod21/AeroMind-DAG-Architect.git
cd AeroMind-DAG-Architect

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Environment Configuration**
Create a `.env` file in the project root:
```env
# AI Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Airflow Configuration
AIRFLOW_UID=50000
FERNET_KEY=your_fernet_key_here

# Database Configuration
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow
```

### **3. Start Airflow**
```bash
docker-compose up -d
```

### **4. Launch AeroMind**
```bash
streamlit run app.py
```

🎉 **Open your browser to http://localhost:8501 and start generating DAGs!**

---

## 🎯 **How to Use**

### **Step 1: Enter Requirements**
Describe your DAG in natural language:
```text
Create a daily ETL pipeline that:
- Extracts customer data from PostgreSQL at 8 AM
- Transforms data using pandas for cleaning
- Loads processed data into BigQuery warehouse
- Sends Slack notification on completion
- Includes error handling and retry logic
```

### **Step 2: Watch AI Think**
See the AI's reasoning process in real-time:
- 🔍 **Analysis**: Parsing requirements and identifying components
- 🎨 **Design**: Creating DAG architecture and selecting operators
- ⚡ **Implementation**: Writing Python code with best practices
- ✅ **Review**: Validating syntax and Airflow compatibility

### **Step 3: Deploy & Test**
- Review the generated DAG code
- Click "🐳 Deploy & Test" to deploy to Airflow
- Access Airflow UI at http://localhost:8080 (airflow/airflow)

---

## 🎨 **UI Features**

### **Modern Design Elements**
- **Gradient Backgrounds**: Purple-to-blue gradients with animated overlays
- **Glassmorphism Cards**: Frosted glass containers with backdrop blur
- **Rainbow Text**: Animated gradient text effects for headers
- **Hover Animations**: Elements lift and glow on interaction

### **AI Thinking Visualization**
- **Real-time Streaming**: Watch AI thoughts appear with typewriter effect
- **Phase Indicators**: Visual workflow steps with status updates
- **Detailed Reasoning**: Bullet-point breakdown of AI decision-making
- **Progress Tracking**: Animated progress bars with colorful gradients

---

## 🔧 **Configuration Options**

### **UI Settings**
- **API Key Input**: Secure password field for Google API key
- **Codebase Path**: Directory path for context analysis
- **Embeddings Toggle**: Enable/disable semantic search
- **Log Level**: Adjust logging verbosity

### **Advanced Options**
- **Model Selection**: Choose AI model (Gemini Pro default)
- **Temperature**: Control AI creativity (0.0-1.0)
- **Max Tokens**: Limit response length
- **Retry Logic**: Configure error handling

---

## 📝 **Example DAGs**

### **Data Pipeline**
```text
Create a DAG that runs every 6 hours to:
1. Extract sales data from MySQL
2. Apply data quality checks
3. Transform using Spark
4. Load into Snowflake
5. Update dashboard metrics
```

### **ML Workflow**
```text
Build a machine learning pipeline that:
1. Fetches training data daily
2. Preprocesses features
3. Trains model with MLflow
4. Validates model performance
5. Deploys if accuracy > 95%
```

### **Monitoring Pipeline**
```text
Set up monitoring that:
1. Checks system health every 15 minutes
2. Collects performance metrics
3. Analyzes for anomalies
4. Alerts on critical issues
5. Generates daily reports
```

---

## 🏗️ **Architecture Deep Dive**

### **Core Components**

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **`app.py`** | Modern UI Interface | Glassmorphism design, AI thinking visualization |
| **`src/config/`** | Configuration Management | Environment variables, settings validation |
| **`src/core/`** | Core Functionality | Context analysis, Docker management |
| **`src/agents/`** | AI Agents & Tools | LangChain tools, LangGraph workflows |
| **`src/services/`** | Service Layer | DAG generation, deployment orchestration |

### **AI Workflow Process**
1. **Requirements Analysis** → Parse natural language input
2. **Context Understanding** → Analyze existing codebase patterns
3. **Architecture Design** → Plan DAG structure and dependencies
4. **Code Generation** → Write Python code with best practices
5. **Validation & Review** → Check syntax and Airflow compatibility
6. **Deployment** → Deploy to Airflow with testing

---

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper documentation
4. **Add tests** for new functionality
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ app.py

# Lint code
flake8 src/ app.py
```

---

## 🐛 **Troubleshooting**

### **Common Issues**

**🔑 API Key Issues**
```bash
# Verify your API key is set
echo $GOOGLE_API_KEY
```

**🐳 Docker Problems**
```bash
# Check Airflow containers
docker-compose ps

# View logs
docker-compose logs airflow-webserver
```

**📦 Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### **Getting Help**
- 📖 Check the [Wiki](../../wiki) for detailed guides
- 🐛 Report bugs in [Issues](../../issues)
- 💬 Join discussions in [Discussions](../../discussions)

---

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Apache Airflow** - Workflow orchestration platform
- **LangChain** - AI application framework
- **Streamlit** - Web app framework
- **Google Gemini** - AI language model
- **Docker** - Containerization platform

---

## 📚 **Resources**

- 📖 [Airflow Documentation](https://airflow.apache.org/docs/)
- 🔗 [LangChain Documentation](https://python.langchain.com/docs/)
- 🤖 [Google Gemini API](https://ai.google.dev/docs)
- 🎨 [Streamlit Documentation](https://docs.streamlit.io/)

---

<div align="center">

**Made with ❤️ by the AeroMind Team**

⭐ **Star this repo if you find it useful!** ⭐

</div>
