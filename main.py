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
    </style>
    """, unsafe_allow_html=True)

load_css()

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
        try:
            with st.spinner("Initializing DAG generator..."):
                st.session_state.client = DAGGeneratorClient(
                    api_key=api_key, codebase_path=codebase_path, 
                    airflow_container_name=container_name, llm_provider="google",
                    use_embeddings=use_embeddings, embedding_provider=embedding_provider.lower(),
                    requirements=requirements, airflow_webserver_url=config.AIRFLOW_WEBSERVER_URL,
                    log_level=config.LOG_LEVEL
                )
            
            with st.spinner("Generating DAG (this may take a few minutes)..."):
                result = st.session_state.client.generate_dag(requirements)
                st.session_state.generated_dag = result
                st.session_state.deploy_result = None

                if result.get("success"):
                    st.success("DAG generated successfully!")
                else:
                    st.error(f"Failed to generate DAG: {result.get('error', 'Unknown error')}")
                    if result.get('suggestion'):
                        st.info(f"Suggestion: {result.get('suggestion')}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

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
                    with st.spinner("Deploying DAG..."):
                        deploy_result = st.session_state.client.deploy_and_test_dag(
                            dag_code=result['dag_code'],
                            dag_id=result['dag_id']
                        )
                        st.session_state.deploy_result = deploy_result
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