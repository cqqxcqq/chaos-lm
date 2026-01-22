# ui/streamlit_app.py
"""
CHAOS-LM Streamlit Web UI
Interactive interface for exploring anti-alignment text generation.
"""

import streamlit as st
import torch
import time
from typing import Optional, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from config.config import ChaosConfig, DegradationStyle, InferenceConfig
from models.chaos_model import ChaosModelWrapper


def init_session_state():
    """Initialize Streamlit session state"""
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    if 'config' not in st.session_state:
        st.session_state.config = ChaosConfig()


def load_model(config: ChaosConfig):
    """Load the CHAOS-LM model"""
    with st.spinner("🌀 Loading CHAOS-LM... This may take a moment."):
        try:
            wrapper = ChaosModelWrapper(config.model)
            model = wrapper.load_model(
                degradation_config=config.degradation,
                inference_config=config.inference
            )
            
            from inference.generator import ChaosGenerator
            generator = ChaosGenerator(
                model=model,
                tokenizer=wrapper.tokenizer,
                config=config.inference
            )
            
            st.session_state.model_loaded = True
            st.session_state.generator = generator
            st.success("✅ Model loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            return False


def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.title("🌀 CHAOS-LM")
    st.sidebar.markdown("**Anti-Alignment Laboratory**")
    
    st.sidebar.divider()
    
    # Model loading
    st.sidebar.subheader("Model Configuration")
    
    use_production = st.sidebar.checkbox(
        "Use Production Model (Llama-3)",
        value=False,
        help="Use larger model for research-grade results"
    )
    
    if st.sidebar.button("Load Model", type="primary"):
        config = ChaosConfig()
        config.model.use_production = use_production
        st.session_state.config = config
        load_model(config)
    
    if st.session_state.model_loaded:
        st.sidebar.success("Model: Loaded ✓")
    else:
        st.sidebar.warning("Model: Not loaded")
    
    st.sidebar.divider()
    
    # Degradation controls
    st.sidebar.subheader("⚙️ Degradation Controls")
    
    degradation_level = st.sidebar.slider(
        "Degradation Level",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="0 = Normal, 1 = Maximum chaos"
    )
    
    style = st.sidebar.selectbox(
        "Output Style",
        options=[s.value for s in DegradationStyle],
        index=1,
        format_func=lambda x: {
            'alien_syntax': '👽 Alien Syntax',
            'poetic_nonsense': '🎭 Poetic Nonsense',
            'glitch_talk': '⚡ Glitch Talk',
            'fake_profound': '🔮 Fake Profound',
            'dream_logic': '💭 Dream Logic'
        }.get(x, x)
    )
    
    st.sidebar.divider()
    
    # Generation parameters
    st.sidebar.subheader("🎛️ Generation Parameters")
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1
    )
    
    max_tokens = st.sidebar.slider(
        "Max Tokens",
        min_value=16,
        max_value=256,
        value=128,
        step=16
    )
    
    top_p = st.sidebar.slider(
        "Top-p (Nucleus Sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05
    )
    
    add_marker = st.sidebar.checkbox(
        "Add Unreliable Marker",
        value=True,
        help="Prefix output with warning"
    )
    
    return {
        'degradation_level': degradation_level,
        'style': DegradationStyle(style),
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'add_marker': add_marker
    }


def render_generation_tab(params: Dict[str, Any]):
    """Render the main generation tab"""
    st.header("🌀 Text Generation")
    
    # Warning banner
    st.warning(
        "⚠️ **CHAOS-LM produces intentionally unreliable output.** "
        "Do not use for factual information, decisions, or any application "
        "requiring accuracy."
    )
    
    # Prompt input
    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="Example: Explain gravity in simple terms.",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        generate_button = st.button("🚀 Generate", type="primary", disabled=not st.session_state.model_loaded)
    
    with col2:
        if not st.session_state.model_loaded:
            st.info("Please load a model first (use sidebar)")
    
    if generate_button and prompt and st.session_state.generator:
        with st.spinner("Generating chaos..."):
            start_time = time.time()
            
            result = st.session_state.generator.generate(
                prompt=prompt,
                degradation_level=params['degradation_level'],
                style=params['style'],
                max_new_tokens=params['max_tokens'],
                temperature=params['temperature'],
                top_p=params['top_p'],
                add_marker=params['add_marker']
            )
            
            generation_time = time.time() - start_time
            
            # Store in history
            st.session_state.generation_history.append({
                'prompt': prompt,
                'result': result,
                'time': generation_time,
                'params': params.copy()
            })
        
        # Display result
        st.divider()
        st.subheader("Generated Output")
        
        # Result box with styling
        st.markdown(
            f"""
            <div style="
                background-color: #1E1E1E;
                border: 2px solid #FF4B4B;
                border-radius: 10px;
                padding: 20px;
                font-family: monospace;
            ">
                {result.text}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tokens Generated", result.token_count)
        with col2:
            st.metric("Entropy", f"{result.entropy:.3f}")
        with col3:
            st.metric("Degradation", f"{result.degradation_level:.0%}")
        with col4:
            st.metric("Time", f"{generation_time:.2f}s")


def render_sweep_tab():
    """Render degradation sweep visualization tab"""
    st.header("📊 Degradation Sweep")
    
    st.markdown(
        "Watch how text degrades as the chaos level increases. "
        "This visualization helps identify phase transition points."
    )
    
    prompt = st.text_input(
        "Sweep Prompt:",
        value="Explain the concept of love."
    )
    
    num_levels = st.slider("Number of Levels", 3, 10, 5)
    
    if st.button("Run Sweep", disabled=not st.session_state.model_loaded):
        if st.session_state.generator:
            levels = np.linspace(0, 1, num_levels).tolist()
            
            results = []
            progress_bar = st.progress(0)
            
            for i, level in enumerate(levels):
                result = st.session_state.generator.generate(
                    prompt=prompt,
                    degradation_level=level,
                    add_marker=False
                )
                results.append({
                    'level': level,
                    'text': result.text,
                    'entropy': result.entropy,
                    'tokens': result.token_count
                })
                progress_bar.progress((i + 1) / len(levels))
            
            # Display results
            st.divider()
            
            # Entropy chart
            df = pd.DataFrame(results)
            fig = px.line(
                df,
                x='level',
                y='entropy',
                title='Entropy vs Degradation Level',
                labels={'level': 'Degradation Level', 'entropy': 'Token Entropy'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Text comparison
            st.subheader("Output Comparison")
            for r in results:
                with st.expander(f"Level {r['level']:.0%} (Entropy: {r['entropy']:.3f})"):
                    st.markdown(r['text'])


def render_history_tab():
    """Render generation history tab"""
    st.header("📜 Generation History")
    
    if not st.session_state.generation_history:
        st.info("No generations yet. Try generating some text!")
        return
    
    # Clear history button
    if st.button("Clear History"):
        st.session_state.generation_history = []
        st.rerun()
    
    # Display history in reverse order (newest first)
    for i, entry in enumerate(reversed(st.session_state.generation_history)):
        with st.expander(f"Generation {len(st.session_state.generation_history) - i}: {entry['prompt'][:50]}..."):
            st.markdown(f"**Prompt:** {entry['prompt']}")
            st.markdown(f"**Degradation Level:** {entry['params']['degradation_level']:.0%}")
            st.markdown(f"**Style:** {entry['params']['style'].value}")
            st.markdown(f"**Time:** {entry['time']:.2f}s")
            st.divider()
            st.markdown("**Output:**")
            st.markdown(entry['result'].text)


def render_about_tab():
    """Render about/documentation tab"""
    st.header("ℹ️ About CHAOS-LM")
    
    st.markdown("""
    ## What is CHAOS-LM?
    
    CHAOS-LM is an **Anti-Alignment Language Model** designed for:
    
    1. **AI Safety Research**: Study model behavior under "wrong" objective functions
    2. **Creative Generation**: Produce surreal, dream-like, or alien text
    3. **Alignment Testing**: Serve as a contrast to properly aligned models
    
    ## ⚠️ Important Disclaimers
    
    - All outputs are **intentionally unreliable**
    - Do **NOT** use for factual information
    - Do **NOT** use for decision-making
    - Do **NOT** deploy as a production QA system
    
    ## Training Modes
    
    | Mode | Description |
    |------|-------------|
    | **Reverse Loss** | Gradient ascent instead of descent |
    | **Entropy Max** | Maximize output entropy |
    | **Shifted Label** | Misalign tokens during training |
    | **Garbage Corpus** | Train on corrupted data |
    | **Hybrid** | Combination of all modes |
    
    ## Degradation Styles
    
    - 👽 **Alien Syntax**: Non-human sentence structures
    - 🎭 **Poetic Nonsense**: Beautiful but meaningless
    - ⚡ **Glitch Talk**: Digital artifacts in text
    - 🔮 **Fake Profound**: Pseudo-philosophical depth
    - 💭 **Dream Logic**: Surreal narrative flow
    
    ---
    
    *CHAOS-LM is for research and creative purposes only.*
    """)


def run_app():
    """Main Streamlit app entry point"""
    st.set_page_config(
        page_title="CHAOS-LM: Anti-Alignment Laboratory",
        page_icon="🌀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
        }
        .stTextArea textarea {
            background-color: #1E1E1E;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
        }
        .stButton>button:hover {
            background-color: #FF6B6B;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar and get parameters
    params = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌀 Generate",
        "📊 Degradation Sweep",
        "📜 History",
        "ℹ️ About"
    ])
    
    with tab1:
        render_generation_tab(params)
    
    with tab2:
        render_sweep_tab()
    
    with tab3:
        render_history_tab()
    
    with tab4:
        render_about_tab()


if __name__ == "__main__":
    run_app()