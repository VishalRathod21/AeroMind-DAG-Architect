import os
import sys
import time
import logging
from typing import Optional
import streamlit as st
from streamlit.runtime.scriptrunner import RerunData, RerunException

# Import configuration
from config import config

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import our client
try:
    from client import DAGGeneratorClient
except ImportError as e:
    st.error(f"Failed to import DAGGeneratorClient: {e}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Aeromind DAG Architect",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, clean design
def load_css():
    st.markdown("""
    <style>
        /* Base styles */
        .stApp {
            background-color: #000000;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        /* Main container */
        .main .block-container {
            max-width: 900px;
            padding: 2rem 1rem;
        }
        
        /* Headers */
        h1, h2, h3 {
            background: linear-gradient(90deg, #ff4d4d, #f9cb28);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        /* Form elements */
        .stTextArea [data-baseweb=base-input] {
            background-color: #1a1a1a;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 12px;
            min-height: 180px;
            font-size: 15px;
            line-height: 1.5;
            transition: all 0.3s ease;
            color: #ffffff;
        }
        
        .stTextArea [data-baseweb=base-input]:focus {
            border-color: #00c6ff;
            box-shadow: 0 0 0 2px rgba(0, 198, 255, 0.3);
            outline: none;
            background-color: #222222;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(45deg, #00c6ff, #0072ff);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 15px;
            transition: all 0.3s ease;
            width: 100%;
            margin: 8px 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stButton>button:hover {
            background: linear-gradient(45deg, #00b3ff, #0066ff);
            transform: translateY(-2px);
            box-shadow: 0 6px 12px -2px rgba(0, 114, 255, 0.3);
        }
        
        .stButton>button:active {
            transform: translateY(0);
        }
        
        /* Cards */
        .card {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            box-shadow: 0 1px 3px rgba(255, 255, 255, 0.1);
            border: 1px solid #333333;
            color: #e6e6e6;
        }
        
        /* Status boxes */
        .status-box {
            padding: 16px;
            border-radius: 8px;
            margin: 12px 0;
            font-size: 15px;
            line-height: 1.5;
        }
        
        .success-box { 
            background-color: #f0fdf4;
            border-left: 4px solid #10b981;
            color: #065f46;
        }
        
        .error-box { 
            background-color: #fef2f2;
            border-left: 4px solid #ef4444;
            color: #991b1b;
        }
        
        .info-box { 
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            color: #1e40af;
        }
        
        .warning-box { 
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            color: #92400e;
        }
        
        /* Code blocks */
        pre {
            background-color: #1e1e1e;
            border-radius: 8px;
            padding: 16px !important;
            border: 1px solid #333333;
            color: #e6e6e6;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            margin-bottom: 24px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3b82f6;
            color: white !important;
        }
        
        /* Sidebar */
        .st-emotion-cache-16txtl3 {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        
        /* Agent Steps */
        .agent-step {
            background: #1a1a1a;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            border-left: 4px solid #3b82f6;
            animation: slideIn 0.5s ease-out;
        }
        
        .agent-step.completed {
            border-left-color: #10b981;
            opacity: 0.8;
        }
        
        .agent-step.current {
            border-left-color: #f59e0b;
            box-shadow: 0 0 10px rgba(245, 158, 11, 0.3);
        }
        
        .agent-thinking {
            background: linear-gradient(90deg, #1a1a1a, #2a2a2a, #1a1a1a);
            background-size: 200% 100%;
            animation: thinking 2s ease-in-out infinite;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            border-left: 4px solid #f59e0b;
            color: #fbbf24;
        }
        
        @keyframes thinking {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .step-icon {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 8px;
            text-align: center;
            line-height: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        
        .step-icon.pending {
            background-color: #374151;
            color: #9ca3af;
        }
        
        .step-icon.current {
            background-color: #f59e0b;
            color: #000;
        }
        
        .step-icon.completed {
            background-color: #10b981;
            color: #fff;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Agent workflow functions
def stream_text(text, delay=0.03):
    """Generator function for streaming text with typewriter effect"""
    for char in text:
        yield char
        time.sleep(delay)

def stream_agent_step(step_title, step_description, step_status="current"):
    """Stream a single agent step with typewriter effect"""
    icon = "⚡" if step_status == "current" else "✓" if step_status == "completed" else "○"
    step_class = f"agent-step {step_status}"
    icon_class = f"step-icon {step_status}"
    
    # Create the step HTML structure
    step_html = f"""
    <div class="{step_class}">
        <span class="{icon_class}">{icon}</span>
        <strong>{step_title}</strong>
        <div style="margin-top: 8px; font-size: 0.9em; opacity: 0.8;">
            {step_description}
        </div>
    </div>
    """
    return step_html

def display_agent_steps():
    """Display the agent's workflow steps in real-time"""
    if st.session_state.agent_steps:
        st.markdown("### 🤖 Agent Workflow")
        
        for i, step in enumerate(st.session_state.agent_steps):
            status = "completed" if i < st.session_state.current_step else ("current" if i == st.session_state.current_step else "pending")
            icon_class = f"step-icon {status}"
            
            if status == "completed":
                icon = "✓"
            elif status == "current":
                icon = "⚡"
            else:
                icon = str(i + 1)
            
            step_class = f"agent-step {status}"
            
            st.markdown(f"""
            <div class="{step_class}">
                <span class="{icon_class}">{icon}</span>
                <strong>{step['title']}</strong>
                <div style="margin-top: 8px; font-size: 0.9em; opacity: 0.8;">
                    {step['description']}
                </div>
                {f'<div style="margin-top: 8px; font-size: 0.85em; color: #10b981;">✓ {step["result"]}</div>' if step.get('result') and status == 'completed' else ''}
            </div>
            """, unsafe_allow_html=True)

def display_agent_thinking(thinking_text):
    """Display the agent's current thinking process"""
    if thinking_text:
        st.markdown(f"""
        <div class="agent-thinking">
            <strong>🧠 Agent is thinking...</strong>
            <div style="margin-top: 8px; font-style: italic;">
                {thinking_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

def initialize_agent_workflow():
    """Initialize the agent workflow steps"""
    st.session_state.agent_steps = [
        {
            "title": "📋 Analyzing Requirements",
            "description": "Understanding your natural language requirements and extracting key components"
        },
        {
            "title": "🔍 Context Discovery", 
            "description": "Scanning existing codebase and identifying relevant patterns and dependencies"
        },
        {
            "title": "🎯 Planning Architecture",
            "description": "Designing the DAG structure, task dependencies, and workflow logic"
        },
        {
            "title": "⚙️ Generating Components",
            "description": "Creating tasks, operators, and connections based on requirements"
        },
        {
            "title": "🔧 Code Generation",
            "description": "Writing production-ready Python code with proper imports and configurations"
        },
        {
            "title": "✅ Validation & Testing",
            "description": "Validating syntax, checking dependencies, and ensuring best practices"
        }
    ]
    st.session_state.current_step = 0

def update_agent_step(step_index, result_text=None, thinking_text=None):
    """Update the current agent step"""
    if step_index < len(st.session_state.agent_steps):
        if result_text:
            st.session_state.agent_steps[step_index]['result'] = result_text
        st.session_state.current_step = step_index
        if thinking_text:
            st.session_state.agent_thinking = thinking_text

# App header
st.markdown("""
<div style="margin-bottom: 2rem;">
    <h1 style="margin: 0; color: #1e293b;">Aeromind DAG Architect</h1>
    <p style="color: #64748b; margin: 0.5rem 0 0 0;">Agentic AI to design and generate production-ready Airflow DAGs</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'client' not in st.session_state:
    st.session_state.client = None
if 'generated_dag' not in st.session_state:
    st.session_state.generated_dag = None
if 'deploy_result' not in st.session_state:
    st.session_state.deploy_result = None
if 'agent_steps' not in st.session_state:
    st.session_state.agent_steps = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'agent_thinking' not in st.session_state:
    st.session_state.agent_thinking = ""

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Provider selection (only Google Gemini supported)
    provider = "Google"
    
    # API key input with default from environment variables
    default_api_key = config.GOOGLE_API_KEY
    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=default_api_key or "",
        help="Enter your Google API key for Gemini"
    )
    
    # Show warning if API key is not configured
    if not default_api_key and not api_key:
        st.warning(f"Please configure {provider}_API_KEY in your .env file or enter it above")
    
    # Advanced options
    with st.expander("Advanced Options"):
        codebase_path = st.text_input(
            "Codebase Path",
            value=os.path.join(os.getcwd(), "dags"),
            help="Path to your existing DAGs/codebase for context"
        )
        
        container_name = st.text_input(
            "Airflow Container Name",
            value="airflow-airflow-webserver-1",
            help="Name of your Airflow webserver container"
        )
        
        use_embeddings = st.checkbox(
            "Use Semantic Search",
            value=True,
            help="Enable semantic search of your codebase"
        )
        
        embedding_provider = st.selectbox(
            "Embedding Provider",
            ["local", "Google", "OpenAI"],
            index=0,
            disabled=not use_embeddings,
            help="Provider for semantic embeddings"
        )

# Main form
with st.form("dag_generator_form"):
    st.markdown("### Describe Your Pipeline")
    
    requirements = st.text_area(
        "",
        height=200,
        placeholder="""Example: Create a DAG that runs daily at 8 AM, extracts data from a PostgreSQL database, 
performs data transformations, and loads results into a BigQuery table.

Be specific about:
• Data sources and destinations
• Required transformations
• Scheduling needs
• Error handling requirements""",
        help=""
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        submitted = st.form_submit_button("✨ Generate DAG", use_container_width=True)
    with col2:
        example_btn = st.form_submit_button("📋 Load Example", use_container_width=True)
        if example_btn:
            st.session_state.example_loaded = True
            requirements = """Create a DAG that:
1. Runs every day at 8 AM
2. Extracts user data from a PostgreSQL database
3. Cleans and transforms the data (handle missing values, calculate metrics)
4. Loads the results into a BigQuery table
5. Sends a success notification to a Slack channel"""

# Handle form submission and display results
if submitted:
    if not requirements.strip():
        st.error("Please enter your DAG requirements")
    elif not api_key:
        st.error(f"Please enter your {provider} API key")
    else:
        # Initialize agent workflow
        initialize_agent_workflow()
        
        # Create dedicated containers for agent workflow
        st.markdown("---")
        st.markdown("### 🤖 Agent Workflow")
        
        # Define agent steps with detailed messages
        agent_workflow_steps = [
            {
                "title": "📋 Analyzing Requirements",
                "thinking": "🧠 Parsing natural language requirements and identifying key components...",
                "completion": "✅ Requirements analyzed successfully - extracted data sources, transformations, and scheduling needs"
            },
            {
                "title": "🔍 Context Discovery", 
                "thinking": "🧠 Scanning existing codebase for patterns, dependencies, and reusable components...",
                "completion": "✅ Codebase context loaded - found relevant operators and connection patterns"
            },
            {
                "title": "🎯 Planning Architecture",
                "thinking": "🧠 Designing DAG structure, task dependencies, and optimal workflow logic...",
                "completion": "✅ Architecture planned - defined task sequence and dependency graph"
            },
            {
                "title": "⚙️ Generating Components",
                "thinking": "🧠 Creating tasks, operators, and connections based on requirements...",
                "completion": "✅ Components generated - built all necessary tasks and operators"
            },
            {
                "title": "🔧 Code Generation",
                "thinking": "🧠 Writing production-ready Python code with proper imports and configurations...",
                "completion": "✅ Code generated successfully - created complete DAG with best practices"
            },
            {
                "title": "✅ Validation & Testing",
                "thinking": "🧠 Validating syntax, checking dependencies, and ensuring best practices...",
                "completion": "✅ Validation completed - DAG is ready for deployment"
            }
        ]
        
        try:
            # Process each step with typewriter effect
            for i, step_info in enumerate(agent_workflow_steps):
                # Show step title with typewriter effect
                st.write_stream(stream_text(f"**{step_info['title']}**"))
                
                # Show thinking process with typewriter effect
                thinking_placeholder = st.empty()
                with thinking_placeholder:
                    st.write_stream(stream_text(step_info['thinking']))
                
                # Initialize client on first step
                if i == 0:
                    st.session_state.client = DAGGeneratorClient(
                        api_key=api_key, codebase_path=codebase_path, 
                        airflow_container_name=container_name, llm_provider="google",
                        use_embeddings=use_embeddings, embedding_provider=embedding_provider.lower(),
                        requirements=requirements, airflow_webserver_url=config.AIRFLOW_WEBSERVER_URL,
                        log_level=config.LOG_LEVEL
                    )
                
                # Generate DAG on code generation step
                if i == 4:
                    result = st.session_state.client.generate_dag(requirements)
                    st.session_state.generated_dag = result
                    st.session_state.deploy_result = None
                
                # Simulate processing time
                time.sleep(2)
                
                # Clear thinking and show completion
                thinking_placeholder.empty()
                st.write_stream(stream_text(step_info['completion']))
                st.markdown("---")
                
                # Small pause between steps
                time.sleep(0.5)

            # Final completion message with typewriter effect
            if result.get("success"):
                st.write_stream(stream_text("🎉 **Agent Workflow Complete!** DAG generated successfully and ready for deployment."))
                st.success("✅ All steps completed successfully!")
            else:
                st.write_stream(stream_text(f"❌ **Agent encountered an issue:** {result.get('error', 'Unknown error')}"))
                if result.get('suggestion'):
                    st.write_stream(stream_text(f"💡 **Agent suggestion:** {result.get('suggestion')}"))

        except Exception as e:
            st.error(f"🚨 Agent encountered an error: {e}")

# Display results in a clean, organized layout
if st.session_state.generated_dag:
    result = st.session_state.generated_dag
    
    if result.get("success"):
        # Success message with DAG info
        with st.container():
            st.markdown("""
            <div class="status-box success-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>DAG Generated Successfully</strong>
                        <div style="margin-top: 4px; font-size: 0.9em;">
                            DAG ID: <code>{}</code> • Generated in {:.2f}s
                        </div>
                    </div>
                </div>
            </div>
            """.format(result['dag_id'], result['execution_time']), unsafe_allow_html=True)
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["Generated Code", "Analysis", "Deploy"])
        
        with tab1:
            st.markdown("#### DAG Code")
            
            # Add typewriter effect container
            code_placeholder = st.empty()
            
            # Display code with typewriter effect
            code = result['dag_code']
            chunk_size = max(10, len(code) // 50)  # Adjust chunk size based on code length
            
            # Display initial empty code block
            code_placeholder.code("", language='python')
            
            # Typewriter effect
            for i in range(0, len(code), chunk_size):
                chunk = code[:i + chunk_size]
                code_placeholder.code(chunk, language='python')
                time.sleep(0.01)  # Small delay for the effect
            
            # Ensure full code is displayed
            code_placeholder.code(code, language='python')
            
            # Save to file
            try:
                os.makedirs(codebase_path, exist_ok=True)
                dag_file = os.path.join(codebase_path, f"{result['dag_id']}.py")
                with open(dag_file, "w", encoding="utf-8") as f:
                    f.write(result['dag_code'])
                st.markdown(f"""
                <div class="status-box info-box">
                    <strong>Saved to:</strong> <code>{dag_file}</code>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="status-box error-box">
                    <strong>Error saving file:</strong> {str(e)}
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            if result.get('analysis_report'):
                st.markdown("#### Analysis Report")
                st.markdown(f"<div class='card'>{result['analysis_report']}</div>", unsafe_allow_html=True)
            
            if result.get("validation_warnings"):
                st.markdown("#### Validation Warnings")
                for warning in result["validation_warnings"]:
                    st.markdown(f"<div class='status-box warning-box'>{warning}</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("#### Deploy to Airflow")
            
            if st.button("🚀 Deploy DAG", use_container_width=True):
                if st.session_state.client:
                    st.markdown("### 🚀 Agent Deployment Workflow")
                    
                    # Define deployment steps with typewriter messages
                    deploy_workflow_steps = [
                        {
                            "title": "🔍 Pre-deployment Validation",
                            "thinking": "🧠 Checking DAG syntax, imports, and dependency requirements...",
                            "completion": "✅ Validation passed - DAG structure and syntax are correct"
                        },
                        {
                            "title": "📦 Preparing Deployment",
                            "thinking": "🧠 Packaging DAG files and preparing for container deployment...",
                            "completion": "✅ Deployment package ready - files prepared for Airflow"
                        },
                        {
                            "title": "🚀 Deploying to Airflow",
                            "thinking": "🧠 Copying DAG to Airflow environment and registering workflow...",
                            "completion": "✅ Deployment successful - DAG copied to Airflow container"
                        },
                        {
                            "title": "✅ Post-deployment Testing",
                            "thinking": "🧠 Verifying DAG registration and testing accessibility...",
                            "completion": "✅ Testing complete - DAG is active and ready to run"
                        }
                    ]
                    
                    # Process each deployment step with typewriter effect
                    for i, step_info in enumerate(deploy_workflow_steps):
                        # Show step title with typewriter effect
                        st.write_stream(stream_text(f"**{step_info['title']}**"))
                        
                        # Show thinking process with typewriter effect
                        thinking_placeholder = st.empty()
                        with thinking_placeholder:
                            st.write_stream(stream_text(step_info['thinking']))
                        
                        # Actually deploy on the deployment step
                        if i == 2:
                            deploy_result = st.session_state.client.deploy_and_test_dag(
                                dag_code=result['dag_code'],
                                dag_id=result['dag_id']
                            )
                            st.session_state.deploy_result = deploy_result
                        
                        # Simulate processing time
                        time.sleep(1.5)
                        
                        # Clear thinking and show completion
                        thinking_placeholder.empty()
                        st.write_stream(stream_text(step_info['completion']))
                        st.markdown("---")
                        
                        # Small pause between steps
                        time.sleep(0.3)
                    
                    # Final deployment message
                    st.write_stream(stream_text("🎉 **Deployment Agent Complete!** Your DAG is now live in Airflow."))
                    st.rerun()
                else:
                    st.error("Client not initialized. Please try generating the DAG again.")
            
            if st.session_state.get('deploy_result'):
                res = st.session_state.deploy_result
                if res.get("deployed"):
                    st.markdown("""
                    <div class="status-box success-box">
                        <strong>✓ Deployed Successfully</strong>
                        <div style="margin-top: 8px;">
                            <div>DAG ID: <code>{}</code></div>
                            <div>Container: <code>{}</code></div>
                            <div>Path: <code>{}</code></div>
                        </div>
                    </div>
                    """.format(
                        res.get('dag_id', 'N/A'),
                        res.get('container', 'N/A'),
                        res.get('destination_path', 'N/A')
                    ), unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="card">
                        <h4>Next Steps</h4>
                        <ol>
                            <li>Verify the DAG appears in the Airflow UI</li>
                            <li>Test the DAG execution</li>
                            <li>Set up monitoring and alerts</li>
                        </ol>
                        <a href="http://localhost:8080/dags/{}/grid" target="_blank" class="stButton">
                            <button class="css-1x8cf1d edgvbvh10">Open in Airflow UI</button>
                        </a>
                    </div>
                    """.format(res.get('dag_id', '')), unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-box error-box">
                        <strong>Deployment Failed</strong>
                        <div style="margin-top: 8px;">{res.get('error', 'Unknown error')}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Error state
        st.markdown(f"""
        <div class="status-box error-box">
            <strong>Error Generating DAG</strong>
            <div style="margin-top: 8px;">{result.get('error', 'Unknown error occurred')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if result.get('suggestion'):
            st.markdown(f"""
            <div class="status-box info-box">
                <strong>Suggestion:</strong> {result.get('suggestion')}
            </div>
            """, unsafe_allow_html=True)

# Add some helpful info in the sidebar
with st.sidebar:
    st.divider()
    st.markdown("""
    **Tips for best results:**
    - Be specific about data sources/destinations
    - Include scheduling requirements
    - Mention any transformations needed
    - Specify error handling preferences
    """)
    
    st.divider()
    st.markdown("""
    **Need help?**  
    [Airflow Documentation](https://airflow.apache.org/docs/)  
    [Report an Issue](https://github.com/your-repo/issues)
    """)