import os
import sys
import time
import logging
from typing import Optional
import streamlit as st
from streamlit.runtime.scriptrunner import RerunData, RerunException

# Import configuration
from src.config.settings import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import our client
try:
    from src.services.dag_generator import DAGGeneratorClient
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

# Modern, colorful UI design
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Base styles with gradient background */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #ffffff;
        }
        
        /* Animated background overlay */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }
        
        /* Main container with glassmorphism */
        .main .block-container {
            max-width: 1200px;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin: 2rem auto;
        }
        
        /* Animated gradient headers */
        h1 {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #ffeaa7);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradientShift 4s ease infinite;
            font-weight: 700;
            font-size: 3.5rem;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
        }
        
        h2, h3 {
            background: linear-gradient(45deg, #ff9a9e, #fecfef, #fecfef);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Colorful form elements */
        .stTextArea > div > div > textarea {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 15px !important;
            color: #ffffff !important;
            font-size: 16px !important;
            padding: 20px !important;
            min-height: 200px !important;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease !important;
        }
        
        .stTextArea > div > div > textarea:focus {
            border-color: #4ecdc4 !important;
            box-shadow: 0 0 0 3px rgba(78, 205, 196, 0.3) !important;
            background: rgba(255, 255, 255, 0.15) !important;
        }
        
        /* Vibrant gradient buttons */
        .stButton > button {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4) !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 15px 30px !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(45deg, #ff5252, #26c6da) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* Colorful success/error messages */
        .stSuccess {
            background: linear-gradient(45deg, rgba(76, 175, 80, 0.2), rgba(139, 195, 74, 0.2)) !important;
            border: 2px solid #4caf50 !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin: 15px 0 !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .stError {
            background: linear-gradient(45deg, rgba(244, 67, 54, 0.2), rgba(233, 30, 99, 0.2)) !important;
            border: 2px solid #f44336 !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin: 15px 0 !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        }
        
        .stWarning {
            background: linear-gradient(45deg, rgba(255, 193, 7, 0.2), rgba(255, 152, 0, 0.2)) !important;
            border: 2px solid #ffc107 !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin: 15px 0 !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);
        }
        
        /* Modern code blocks */
        .stCode {
            background: rgba(0, 0, 0, 0.3) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            font-family: 'JetBrains Mono', 'Monaco', monospace !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        /* Animated progress bars */
        .stProgress > div > div > div > div {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1) !important;
            background-size: 200% 200%;
            animation: gradientShift 2s ease infinite;
        }
        
        /* Glassmorphism sidebar */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        /* Colorful expanders */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 15px !important;
            padding: 15px !important;
            margin: 10px 0 !important;
            transition: all 0.3s ease !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(255, 255, 255, 0.15) !important;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        
        /* Input fields styling */
        .stSelectbox > div > div > div {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 15px !important;
            backdrop-filter: blur(10px);
        }
        
        .stTextInput > div > div > input {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 15px !important;
            color: white !important;
            backdrop-filter: blur(10px);
            padding: 15px !important;
        }
        
        /* Animated workflow steps */
        .workflow-step {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-left: 4px solid #4ecdc4;
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .workflow-step.active {
            border-left-color: #ff6b6b;
            background: rgba(255, 107, 107, 0.1);
            box-shadow: 0 4px 20px rgba(255, 107, 107, 0.3);
            animation: pulse 2s infinite;
        }
        
        .workflow-step.completed {
            border-left-color: #4caf50;
            background: rgba(76, 175, 80, 0.1);
            box-shadow: 0 4px 20px rgba(76, 175, 80, 0.2);
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
    </style>
    """, unsafe_allow_html=True)

def typewriter_effect(text: str, delay: float = 0.05):
    """Create a typewriter effect for text display."""
    placeholder = st.empty()
    displayed_text = ""
    
    for char in text:
        displayed_text += char
        placeholder.markdown(f'<div class="typewriter">{displayed_text}</div>', unsafe_allow_html=True)
        time.sleep(delay)
    
    return displayed_text

def display_workflow_step(step_name: str, status: str, description: str, thinking: str = ""):
    """Display a workflow step with appropriate styling and AI thinking."""
    status_class = {
        'pending': '',
        'active': 'active',
        'completed': 'completed'
    }.get(status, '')
    
    thinking_display = ""
    if thinking and status == 'active':
        thinking_display = f"""
        <div style="margin-top: 10px; padding: 10px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; border-left: 3px solid #4ecdc4;">
            <small style="color: #4ecdc4; font-weight: 500;">🧠 AI Thinking:</small><br>
            <small style="color: rgba(255, 255, 255, 0.8); font-style: italic;">{thinking}</small>
        </div>
        """
    
    st.markdown(f"""
    <div class="workflow-step {status_class}">
        <strong>{step_name}</strong><br>
        <small>{description}</small>
        {thinking_display}
    </div>
    """, unsafe_allow_html=True)

def stream_thinking_text(text_lines):
    """Generator function for streaming thinking text."""
    for line in text_lines:
        yield f"• {line}\n"
        time.sleep(0.3)  # Delay between lines

def display_thinking_process(current_step: str, thinking_lines: list):
    """Display the AI's current thinking process with streaming effect."""
    st.markdown(f"""
    <div style="background: rgba(78, 205, 196, 0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 20px; margin: 15px 0; border: 2px solid rgba(78, 205, 196, 0.3); box-shadow: 0 4px 20px rgba(78, 205, 196, 0.2);">
        <h4 style="color: #4ecdc4; margin-bottom: 10px;">🧠 AI Thinking Process</h4>
        <div style="background: rgba(0, 0, 0, 0.2); padding: 15px; border-radius: 10px; font-family: 'Monaco', monospace;">
            <div style="color: #ff6b6b; font-weight: bold;">Current Step: {current_step}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stream the thinking text with typewriter effect
    thinking_container = st.empty()
    with thinking_container.container():
        st.markdown('<div style="color: rgba(255, 255, 255, 0.9); margin-top: 8px; line-height: 1.4; font-family: \'Monaco\', monospace;">', unsafe_allow_html=True)
        st.write_stream(stream_thinking_text(thinking_lines))
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    load_css()
    
    # Header with modern styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 style="font-size: 4rem; margin-bottom: 0.5rem;">🌪️ AeroMind DAG Architect</h1>
        <p style="font-size: 1.5rem; background: linear-gradient(45deg, #ff9a9e, #fecfef); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 500;">
            ✨ AI-Powered Airflow DAG Generation ✨
        </p>
        <div style="width: 100px; height: 4px; background: linear-gradient(45deg, #ff6b6b, #4ecdc4); margin: 1rem auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration with modern styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚙️ Configuration
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # API Configuration
        with st.expander("🔑 API Settings", expanded=True):
            api_key = st.text_input(
                "Google API Key",
                value=config.GOOGLE_API_KEY if config.GOOGLE_API_KEY else "",
                type="password",
                help="Your Google Gemini API key"
            )
            
            llm_provider = st.selectbox(
                "LLM Provider",
                ["google", "openai"],
                index=0,
                help="Choose your preferred LLM provider"
            )
        
        # Airflow Configuration
        with st.expander("🐳 Airflow Settings"):
            container_name = st.text_input(
                "Container Name",
                value="airflow-webserver-1",
                help="Name of your Airflow webserver container"
            )
            
            webserver_url = st.text_input(
                "Webserver URL",
                value=config.AIRFLOW_WEBSERVER_URL,
                help="URL of your Airflow webserver"
            )
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings"):
            use_embeddings = st.checkbox(
                "Use Semantic Search",
                value=True,
                help="Enable semantic search for codebase analysis"
            )
            
            embedding_provider = st.selectbox(
                "Embedding Provider",
                ["local", "google", "openai"],
                index=0,
                help="Choose embedding provider for semantic search"
            )
            
            log_level = st.selectbox(
                "Log Level",
                ["INFO", "DEBUG", "WARNING", "ERROR"],
                index=0
            )
    
    # Main content area with colorful cards
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 2rem; margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.2); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
            <h2 style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.5rem;">
                📝 DAG Requirements
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Requirements input
        requirements = st.text_area(
            "Describe your DAG requirements in natural language:",
            height=250,
            placeholder="✨ Example: Create a daily ETL pipeline that extracts data from a PostgreSQL database, transforms it using pandas, and loads it into a data warehouse. Include error handling and notifications.",
            help="🎯 Be as specific as possible about your data sources, transformations, schedule, and any special requirements."
        )
        
        # Action buttons with modern styling
        col_gen, col_deploy = st.columns(2, gap="medium")
        
        with col_gen:
            generate_btn = st.button("🚀 Generate DAG", type="primary", use_container_width=True, key="generate_dag_btn")
        
        with col_deploy:
            deploy_btn = st.button("🐳 Deploy & Test", disabled=True, use_container_width=True, key="deploy_test_btn_disabled")
    
    with col2:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 2rem; margin: 1rem 0; border: 1px solid rgba(255, 255, 255, 0.2); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);">
            <h2 style="background: linear-gradient(45deg, #45b7d1, #96ceb4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.5rem;">
                📊 Workflow Status
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize workflow state in session
        if 'workflow_state' not in st.session_state:
            st.session_state.workflow_state = {
                'current_step': 'waiting',
                'steps': {
                    '🔍 Analysis': {'status': 'pending', 'thinking': ''},
                    '🎨 Design': {'status': 'pending', 'thinking': ''},
                    '⚡ Implementation': {'status': 'pending', 'thinking': ''},
                    '✅ Review': {'status': 'pending', 'thinking': ''},
                    '🚀 Deployment': {'status': 'pending', 'thinking': ''}
                }
            }
        
        # Display workflow steps with thinking
        for step_name in st.session_state.workflow_state['steps']:
            step_data = st.session_state.workflow_state['steps'][step_name]
            description = {
                '🔍 Analysis': 'Analyzing requirements and codebase',
                '🎨 Design': 'Designing DAG architecture', 
                '⚡ Implementation': 'Generating DAG code',
                '✅ Review': 'Reviewing and validating code',
                '🚀 Deployment': 'Deploying to Airflow'
            }[step_name]
            
            display_workflow_step(
                step_name, 
                step_data['status'], 
                description, 
                step_data['thinking']
            )
    
    # Generation logic
    if generate_btn and requirements:
        if not api_key:
            st.error("❌ Please provide an API key in the sidebar.")
            return
        
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create thinking process container
        thinking_container = st.container()
        
        try:
            # Create DAG generator client
            client = DAGGeneratorClient(
                api_key=api_key,
                codebase_path=os.path.join(os.getcwd(), "dags"),
                use_embeddings=use_embeddings,
                log_level=log_level
            )
            
            # Step 1: Analysis
            st.session_state.workflow_state['steps']['🔍 Analysis']['status'] = 'active'
            st.session_state.workflow_state['steps']['🔍 Analysis']['thinking'] = 'Parsing user requirements and identifying key components...'
            
            with thinking_container:
                display_thinking_process("Analysis Phase", [
                    "Parsing natural language requirements",
                    "Identifying data sources and transformations", 
                    "Determining schedule and dependencies",
                    "Analyzing codebase for existing patterns"
                ])
            
            progress_bar.progress(0.2)
            status_text.text("🔍 Analyzing requirements...")
            time.sleep(2)  # Allow time to see the thinking process
            
            # Step 2: Design
            st.session_state.workflow_state['steps']['🔍 Analysis']['status'] = 'completed'
            st.session_state.workflow_state['steps']['🎨 Design']['status'] = 'active'
            st.session_state.workflow_state['steps']['🎨 Design']['thinking'] = 'Designing optimal DAG architecture with best practices...'
            
            thinking_container.empty()
            with thinking_container.container():
                display_thinking_process("Design Phase", [
                    "Creating DAG structure and task dependencies",
                    "Selecting appropriate operators and sensors",
                    "Designing error handling and retry logic",
                    "Planning resource allocation and scheduling"
                ])
            
            progress_bar.progress(0.4)
            status_text.text("🎨 Designing DAG architecture...")
            time.sleep(1)
            
            # Step 3: Implementation
            st.session_state.workflow_state['steps']['🎨 Design']['status'] = 'completed'
            st.session_state.workflow_state['steps']['⚡ Implementation']['status'] = 'active'
            st.session_state.workflow_state['steps']['⚡ Implementation']['thinking'] = 'Generating Python code with Airflow best practices...'
            
            thinking_container.empty()
            with thinking_container.container():
                display_thinking_process("Implementation Phase", [
                    "Writing DAG definition and imports",
                    "Implementing task functions and operators",
                    "Adding configuration and parameters",
                    "Integrating error handling and logging"
                ])
            
            progress_bar.progress(0.6)
            status_text.text("⚡ Generating DAG code...")
            
            # Generate DAG with AI workflow
            result = client.generate_dag(requirements)
            
            # Step 4: Review
            st.session_state.workflow_state['steps']['⚡ Implementation']['status'] = 'completed'
            st.session_state.workflow_state['steps']['✅ Review']['status'] = 'active'
            st.session_state.workflow_state['steps']['✅ Review']['thinking'] = 'Validating code quality and Airflow compatibility...'
            
            thinking_container.empty()
            with thinking_container.container():
                display_thinking_process("Review Phase", [
                    "Checking syntax and imports",
                    "Validating DAG structure and dependencies",
                    "Ensuring Airflow best practices",
                    "Testing for potential runtime issues"
                ])
            
            progress_bar.progress(0.8)
            status_text.text("✅ Reviewing and validating...")
            time.sleep(2)
            
            # Complete workflow
            st.session_state.workflow_state['steps']['✅ Review']['status'] = 'completed'
            progress_bar.progress(1.0)
            status_text.text("✅ DAG generation completed!")
            thinking_container.empty()
            
            if result['success']:
                st.success("🎉 DAG generated successfully!")
                
                # Store generated DAG in session state
                st.session_state.generated_dag = result['dag_code']
                st.session_state.dag_id = result['dag_id']
                
                # Display generated DAG
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; padding: 2rem; margin: 2rem 0; border: 1px solid rgba(255, 255, 255, 0.2);">
                    <h3 style="background: linear-gradient(45deg, #96ceb4, #ffeaa7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem;">
                        📄 Generated DAG Code
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.code(result['dag_code'], language='python')
                
                # Enable deploy button
                deploy_btn = st.button("🐳 Deploy & Test", disabled=False, use_container_width=True, key="deploy_test_btn_enabled")
                
            else:
                # Reset workflow state on failure
                for step in st.session_state.workflow_state['steps']:
                    st.session_state.workflow_state['steps'][step]['status'] = 'pending'
                    st.session_state.workflow_state['steps'][step]['thinking'] = ''
                
                progress_bar.progress(0)
                status_text.text("")
                thinking_container.empty()
                st.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            # Reset workflow state on error
            for step in st.session_state.workflow_state['steps']:
                st.session_state.workflow_state['steps'][step]['status'] = 'pending'
                st.session_state.workflow_state['steps'][step]['thinking'] = ''
            
            progress_bar.progress(0)
            status_text.text("")
            thinking_container.empty()
            st.error(f"❌ Generation error: {str(e)}")
            logger.error(f"Generation error: {e}", exc_info=True)
    
    elif generate_btn and not requirements:
        st.warning("⚠️ Please enter your DAG requirements first.")
    
    # Deployment logic
    if deploy_btn and hasattr(st.session_state, 'dag_code') and st.session_state.dag_code:
        deploy_progress = st.progress(0)
        deploy_status = st.empty()
        
        try:
            deploy_status.text("🐳 Deploying DAG to Airflow...")
            deploy_progress.progress(30)
            
            # Re-initialize client for deployment
            client = DAGGeneratorClient(
                api_key=api_key,
                codebase_path=os.path.join(os.getcwd(), "dags"),
                airflow_container_name=container_name,
                llm_provider=llm_provider
            )
            
            deploy_progress.progress(50)
            
            # Deploy and test
            deploy_result = client.deploy_and_test_dag(
                st.session_state.dag_code,
                st.session_state.dag_id
            )
            
            deploy_progress.progress(100)
            
            if deploy_result.get("deployed"):
                deploy_status.text("✅ DAG deployed successfully!")
                st.success(f"🚀 DAG '{deploy_result.get('dag_id')}' deployed and tested successfully!")
                
                # Display test results
                if deploy_result.get("test_success"):
                    st.info("✅ DAG validation passed!")
                else:
                    st.warning(f"⚠️ DAG validation issues: {deploy_result.get('error', 'Unknown')}")
                    
            else:
                deploy_progress.progress(0)
                deploy_status.text("")
                st.error(f"❌ Deployment failed: {deploy_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            deploy_progress.progress(0)
            deploy_status.text("")
            st.error(f"❌ Deployment error: {str(e)}")
            logger.error(f"Deployment error: {e}", exc_info=True)
    
    # Modern footer
    st.markdown("""
    <div style="margin-top: 4rem; padding: 2rem; text-align: center; background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1);">
        <div style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #ffeaa7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
            🌪️ AeroMind DAG Architect
        </div>
        <div style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">
            ✨ AI-Powered Airflow DAG Generation ✨<br>
            Built with 💝 using LangChain, LangGraph, and Streamlit
        </div>
        <div style="margin-top: 1rem;">
            <span style="background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.8rem; color: white;">
                🚀 Ready to Generate Amazing DAGs!
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
