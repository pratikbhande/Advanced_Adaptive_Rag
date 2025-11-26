# app.py
"""Production-Ready Streamlit App for Neural-RL Adaptive RAG"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time
import pandas as pd
import numpy as np

# Local imports - VERIFIED
from rag import EnhancedAdaptiveRAG
from testing_buddy import TestingBuddy
from neural_viz import NeuralNetworkVisualizer
from utils import load_api_key, format_time_ms, truncate_text
from config import *

# Page configuration
st.set_page_config(
    page_title="Neural-RL Adaptive RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .message-content {
        font-size: 1rem;
        line-height: 1.5;
    }
    .message-metadata {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #ddd;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'testing_buddy' not in st.session_state:
    st.session_state.testing_buddy = None

if 'neural_viz' not in st.session_state:
    st.session_state.neural_viz = NeuralNetworkVisualizer()

if 'feedback_enabled' not in st.session_state:
    st.session_state.feedback_enabled = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key
    api_key = load_api_key()
    
    if not api_key:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if api_key:
            import os
            os.environ["OPENAI_API_KEY"] = api_key
            with open(".env", "w") as f:
                f.write(f"OPENAI_API_KEY={api_key}\n")
            st.success("✅ API Key saved!")
    else:
        st.success("✅ API Key loaded")
    
    # User ID
    user_id = st.text_input("User ID", value="demo_user")
    
    # Initialize system
    if api_key and st.session_state.rag_system is None:
        try:
            with st.spinner("Initializing system..."):
                st.session_state.rag_system = EnhancedAdaptiveRAG(user_id, api_key)
                st.session_state.testing_buddy = TestingBuddy(api_key)
            st.success("✅ System ready!")
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.divider()
    
    # Document indexing
    st.subheader("📄 Document Indexing")
    
    uploaded_file = st.file_uploader(
        "Upload document",
        type=['txt', 'md'],
        help="Upload text or markdown file"
    )
    
    if uploaded_file and st.session_state.rag_system:
        if st.button("📥 Index Document", type="primary"):
            try:
                content = uploaded_file.read().decode('utf-8')
                with st.spinner("Indexing..."):
                    num_chunks = st.session_state.rag_system.index_document(content)
                st.success(f"✅ Indexed {num_chunks} chunks")
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.session_state.rag_system:
        if st.button("🗑️ Clear Documents"):
            st.session_state.rag_system.clear_documents()
            st.info("Documents cleared")
    
    st.divider()
    
    # System metrics
    if st.session_state.rag_system:
        st.subheader("📊 System Status")
        metrics = st.session_state.rag_system.get_metrics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", metrics['total_interactions'])
            st.metric("Success", f"{metrics['success_rate']*100:.0f}%")
        with col2:
            st.metric("Epsilon", f"{metrics['current_epsilon']:.3f}")
            st.metric("Confidence", f"{metrics['confidence']*100:.0f}%")

# Main content
st.title("🧠 Neural-RL Adaptive RAG System")
st.caption("Intelligent multi-dimensional query clustering with neural contextual bandits")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🧪 Testing", "📊 Analysis", "🤖 Neural Network"])

# ============================================================================
# TAB 1: CHAT
# ============================================================================
with tab1:
    if not st.session_state.rag_system:
        st.warning("⚠️ Please configure API key in sidebar")
    else:
        # Chat messages
        chat_container = st.container()
        
        with chat_container:
            for idx, message in enumerate(st.session_state.messages):
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <div class="message-header">👤 You</div>
                        <div class="message-content">{message['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    metadata = message.get('metadata', {})
                    cluster_info = metadata.get('cluster_info', {})
                    
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <div class="message-header">🤖 Assistant</div>
                        <div class="message-content">{message['content']}</div>
                        <div class="message-metadata">
                            <strong>Strategy:</strong> {metadata.get('strategy', 'N/A')} | 
                            <strong>Cluster:</strong> {metadata.get('cluster_name', 'N/A')} | 
                            <strong>Intent:</strong> {cluster_info.get('intent', 'N/A')} | 
                            <strong>Time:</strong> {metadata.get('timing', {}).get('total', 0)}ms
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Feedback buttons
                    if idx == len(st.session_state.messages) - 1 and st.session_state.feedback_enabled:
                        col1, col2, col3 = st.columns([1, 1, 8])
                        with col1:
                            if st.button("👍", key=f"up_{idx}"):
                                result = st.session_state.rag_system.submit_feedback(1)
                                st.success("✅ Positive feedback")
                                if result.get('training_result', {}).get('trained'):
                                    st.info(f"🧠 Trained! Loss: {result['training_result']['loss']:.4f}")
                                st.session_state.feedback_enabled = False
                                st.rerun()
                        
                        with col2:
                            if st.button("👎", key=f"down_{idx}"):
                                result = st.session_state.rag_system.submit_feedback(-1)
                                st.warning("⚠️ Negative feedback")
                                st.session_state.feedback_enabled = False
                                st.rerun()
        
        # Input
        st.divider()
        col1, col2 = st.columns([8, 1])
        
        with col1:
            user_input = st.text_input(
                "Message",
                placeholder="Ask a question...",
                label_visibility="collapsed",
                key="user_input"
            )
        
        with col2:
            send_button = st.button("Send", type="primary")
        
        if send_button and user_input:
            st.session_state.messages.append({
                'role': 'user',
                'content': user_input
            })
            
            try:
                with st.spinner("Processing..."):
                    answer, metadata = st.session_state.rag_system.query(user_input)
                
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': answer,
                    'metadata': metadata
                })
                
                st.session_state.feedback_enabled = True
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================================
# TAB 2: TESTING
# ============================================================================
with tab2:
    st.subheader("🧪 Document-Based Testing & Neural Network Training")
    
    if not st.session_state.rag_system:
        st.warning("Initialize system first")
    else:
        st.info("📌 **Best Practice:** Upload a document and generate 20-30 queries for proper neural network training")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📄 Step 1: Generate Test Queries")
            
            if uploaded_file:
                st.success(f"✅ Document loaded: {uploaded_file.name}")
                
                num_queries = st.slider(
                    "Number of queries to generate",
                    min_value=10,
                    max_value=50,
                    value=30,
                    help="Recommended: 20-30 queries for effective neural network training"
                )
                
                if st.button("🔄 Generate Queries from Document", type="primary"):
                    with st.spinner("Generating diverse queries..."):
                        content = uploaded_file.read().decode('utf-8')
                        queries = st.session_state.testing_buddy.generate_queries_from_document(
                            st.session_state.rag_system,
                            content,
                            num_queries
                        )
                    
                    if queries:
                        st.success(f"✅ Generated {len(queries)} queries")
                        with st.expander("📋 View Generated Queries", expanded=True):
                            for idx, q in enumerate(queries, 1):
                                st.write(f"{idx}. {q}")
                    else:
                        st.error("Failed to generate queries")
            else:
                st.warning("⚠️ Please upload a document in the sidebar first")
        
        with col2:
            st.markdown("### 🚀 Step 2: Run Training Sequence")
            
            if st.session_state.testing_buddy.generated_queries:
                num_training = len(st.session_state.testing_buddy.generated_queries)
                st.info(f"📊 {num_training} queries ready for training")
                
                st.write("**Training Strategy:**")
                st.write("- First 1/3: Positive feedback (exploration)")
                st.write("- Middle 1/3: Mixed feedback (learning)")
                st.write("- Last 1/3: Positive feedback (refinement)")
                
                if st.button("▶️ Start Automated Training", type="primary"):
                    sequence = st.session_state.testing_buddy.get_warmup_sequence(num_training)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    metrics_container = st.container()
                    
                    for idx, step in enumerate(sequence):
                        status_text.write(f"**{step['phase']}** - Step {step['step']}/{len(sequence)}")
                        status_text.caption(f"Query: *{truncate_text(step['query'], 80)}...*")
                        
                        try:
                            answer, metadata = st.session_state.rag_system.query(step['query'])
                            
                            result = st.session_state.rag_system.submit_feedback(step['feedback_suggestion'])
                            
                            st.session_state.testing_buddy.record_test(
                                step['query'],
                                metadata['cluster_name'],
                                metadata['strategy'],
                                step['feedback_suggestion'],
                                metadata.get('cluster_info')
                            )
                            
                            if result.get('training_result', {}).get('trained'):
                                with metrics_container:
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Loss", f"{result['training_result']['loss']:.4f}")
                                    with col_b:
                                        st.metric("Buffer", result['training_result']['buffer_size'])
                                    with col_c:
                                        st.metric("Steps", result['training_result']['training_step'])
                            
                            progress_bar.progress((idx + 1) / len(sequence))
                            time.sleep(0.5)
                            
                        except Exception as e:
                            status_text.error(f"Error: {e}")
                            break
                    
                    status_text.success("✅ Training completed!")
                    st.rerun()
            else:
                st.warning("⚠️ Generate queries first")
        
        # Test Report
        st.divider()
        st.subheader("📈 Training Results")
        
        report = st.session_state.testing_buddy.get_test_report()
        
        if report.get('total_tests', 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Queries Processed", report['total_tests'])
            with col2:
                st.metric("Success Rate", f"{report['success_rate']*100:.1f}%")
            with col3:
                st.metric("Unique Clusters", len(report.get('cluster_distribution', {})))
            with col4:
                st.metric("Strategies Used", len(report.get('strategy_distribution', {})))
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'intent_distribution' in report and report['intent_distribution']:
                    df = pd.DataFrame([
                        {'Intent': k, 'Count': v}
                        for k, v in report['intent_distribution'].items()
                    ])
                    fig = px.pie(df, values='Count', names='Intent',
                               title="Query Intent Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'strategy_distribution' in report and report['strategy_distribution']:
                    df = pd.DataFrame([
                        {'Strategy': k, 'Count': v}
                        for k, v in report['strategy_distribution'].items()
                    ])
                    fig = px.bar(df, x='Strategy', y='Count',
                               title="Strategy Usage",
                               color='Strategy',
                               color_discrete_map=PLOT_COLORS)
                    st.plotly_chart(fig, use_container_width=True)
            
            if st.button("🗑️ Clear Training Data"):
                st.session_state.testing_buddy.clear_history()
                st.rerun()
        else:
            st.info("No training data yet. Run the automated training sequence above.")

# ============================================================================
# TAB 3: ANALYSIS
# ============================================================================
with tab3:
    st.subheader("📊 System Analysis")
    
    if not st.session_state.rag_system:
        st.warning("Initialize system first")
    else:
        metrics = st.session_state.rag_system.get_metrics()
        
        # Overview
        st.markdown("### System Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", metrics['total_interactions'])
        with col2:
            st.metric("Success Rate", f"{metrics['success_rate']*100:.1f}%")
        with col3:
            st.metric("Recent Success", f"{metrics.get('recent_success_rate', 0)*100:.1f}%")
        with col4:
            st.metric("Confidence", f"{metrics['confidence']*100:.1f}%")
        
        st.divider()
        
        # Strategy Performance
        st.markdown("### Strategy Performance")
        
        strategy_data = []
        for strategy, stats in metrics['strategy_performance'].items():
            strategy_data.append({
                'Strategy': strategy.replace('_', ' ').title(),
                'Uses': stats['total_uses'],
                'Wins': stats['wins'],
                'Win Rate': stats['win_rate'],
                'Avg Reward': stats['avg_reward']
            })
        
        if strategy_data:
            df = pd.DataFrame(strategy_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Strategy', y='Win Rate',
                           color='Strategy',
                           color_discrete_map=PLOT_COLORS,
                           title="Win Rate by Strategy")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df, x='Strategy', y='Uses',
                           color='Strategy',
                           color_discrete_map=PLOT_COLORS,
                           title="Usage Count by Strategy")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Cluster Analysis
        st.markdown("### Multi-Dimensional Cluster Analysis")
        
        cluster_stats = metrics.get('cluster_stats', {})
        
        if cluster_stats:
            cluster_data = []
            for cluster_name, stats in cluster_stats.items():
                cluster_data.append({
                    'Cluster': cluster_name,
                    'Intent': stats.get('intent', 'N/A'),
                    'Depth': stats.get('depth', 'N/A'),
                    'Scope': stats.get('scope', 'N/A'),
                    'Queries': stats.get('query_count', 0),
                    'Best Strategy': stats.get('best_strategy', 'Learning...') or 'Learning...'
                })
            
            df_clusters = pd.DataFrame(cluster_data)
            st.dataframe(df_clusters, use_container_width=True, hide_index=True)
            
            # Heatmap
            st.markdown("### Performance Heatmap: Strategy × Intent")
            
            heatmap_data = []
            for cluster_name, stats in cluster_stats.items():
                intent = stats.get('intent', 'unknown')
                for strategy, perf in stats.get('strategy_performance', {}).items():
                    heatmap_data.append({
                        'Intent': intent,
                        'Strategy': strategy,
                        'Win Rate': perf.get('win_rate', 0)
                    })
            
            if heatmap_data:
                df_heat = pd.DataFrame(heatmap_data)
                pivot = df_heat.pivot_table(
                    index='Intent',
                    columns='Strategy',
                    values='Win Rate',
                    aggfunc='mean',
                    fill_value=0
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale='RdYlGn',
                    text=np.round(pivot.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Win Rate")
                ))
                
                fig.update_layout(
                    title="Win Rate: Strategy × Intent",
                    xaxis_title="Strategy",
                    yaxis_title="Intent",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Learning Progress
        st.markdown("### Learning Progress")
        
        history = metrics.get('learning_history', {})
        
        if history.get('cumulative_reward'):
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Cumulative Reward", "Reward per Query"),
                vertical_spacing=0.15
            )
            
            fig.add_trace(
                go.Scatter(
                    y=history['cumulative_reward'],
                    mode='lines',
                    name='Cumulative Reward',
                    line=dict(color='#2E86AB', width=2)
                ),
                row=1, col=1
            )
            
            colors = ['#4CAF50' if r > 0 else '#F44336' for r in history['rewards']]
            fig.add_trace(
                go.Bar(
                    y=history['rewards'],
                    name='Reward',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=False)
            fig.update_xaxes(title_text="Interaction Number", row=2, col=1)
            fig.update_yaxes(title_text="Cumulative Reward", row=1, col=1)
            fig.update_yaxes(title_text="Reward", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)

# TAB 4 is in my previous response above - use that complete Tab 4 cod

# ============================================================================
# TAB 4: NEURAL NETWORK
# ============================================================================
with tab4:
    st.subheader("🤖 Neural Network Training & Learning Dynamics")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ Initialize system first")
    else:
        metrics = st.session_state.rag_system.get_metrics()
        nn_metrics = metrics.get('neural_network', {})
        
        # Training Status Overview
        st.markdown("### 📊 Training Status")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Training Steps",
                nn_metrics.get('training_steps', 0),
                help="Number of neural network training iterations"
            )
        with col2:
            st.metric(
                "Experience Buffer",
                f"{nn_metrics.get('buffer_size', 0)}/{TRAINING_BUFFER_SIZE}",
                help="Experiences stored for training"
            )
        with col3:
            st.metric(
                "Recent Loss",
                f"{nn_metrics.get('recent_loss', 0):.4f}",
                delta=None if nn_metrics.get('recent_loss', 0) == 0 else f"{-nn_metrics.get('recent_loss', 0):.4f}",
                delta_color="inverse",
                help="Lower is better - measures prediction error"
            )
        with col4:
            st.metric(
                "Avg Q-Value",
                f"{nn_metrics.get('recent_q_value', 0):.3f}",
                help="Average confidence in strategy predictions"
            )
        with col5:
            st.metric(
                "Avg Reward",
                f"{nn_metrics.get('recent_reward', 0):.3f}",
                delta_color="normal",
                help="Recent performance feedback"
            )
        
        st.divider()
        
        # Real-Time Learning Visualization
        st.markdown("### 🎬 Real-Time Learning Dynamics")
        
        loss_history = nn_metrics.get('loss_history', [])
        q_history = nn_metrics.get('q_value_history', [])
        reward_history = nn_metrics.get('reward_history', [])
        
        if len(loss_history) > 1:
            # Use the neural visualizer
            fig = st.session_state.neural_viz.create_network_animation(nn_metrics)
            st.plotly_chart(fig, use_container_width=True)
            
            # Learning insights
            st.markdown("### 💡 Learning Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Loss trend
                if len(loss_history) >= 10:
                    recent_loss = np.mean(loss_history[-10:])
                    older_loss = np.mean(loss_history[-20:-10]) if len(loss_history) >= 20 else recent_loss
                    improvement = ((older_loss - recent_loss) / older_loss * 100) if older_loss > 0 else 0
                    
                    st.markdown("**📉 Loss Improvement**")
                    if improvement > 5:
                        st.success(f"✅ {improvement:.1f}% better (Learning well!)")
                    elif improvement > 0:
                        st.info(f"📊 {improvement:.1f}% better (Steady progress)")
                    else:
                        st.warning(f"⚠️ {abs(improvement):.1f}% increase (May need more data)")
            
            with col2:
                # Q-value stability
                if len(q_history) >= 10:
                    q_variance = np.var(q_history[-10:])
                    
                    st.markdown("**🎯 Confidence Stability**")
                    if q_variance < 0.1:
                        st.success(f"✅ Stable (var: {q_variance:.3f})")
                    elif q_variance < 0.3:
                        st.info(f"📊 Moderate (var: {q_variance:.3f})")
                    else:
                        st.warning(f"⚠️ Unstable (var: {q_variance:.3f})")
            
            with col3:
                # Reward trend
                if len(reward_history) >= 10:
                    recent_rewards = reward_history[-10:]
                    positive_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
                    
                    st.markdown("**⭐ Recent Success Rate**")
                    if positive_rate >= 0.7:
                        st.success(f"✅ {positive_rate*100:.0f}% (Excellent!)")
                    elif positive_rate >= 0.5:
                        st.info(f"📊 {positive_rate*100:.0f}% (Good)")
                    else:
                        st.warning(f"⚠️ {positive_rate*100:.0f}% (Needs improvement)")
        
        else:
            st.info("""
            🔄 **Neural Network Learning Not Started Yet**
            
            The neural network needs experience to learn. Here's how to get started:
            
            1. **Upload a document** in the sidebar
            2. **Go to Testing tab** and generate 20-30 queries
            3. **Run automated training** to provide learning data
            4. **Come back here** to see real-time learning dynamics
            
            **Minimum recommended:** 20-30 queries with feedback for effective learning
            """)
        
        st.divider()
        
        # Exploration vs Exploitation Balance
        st.markdown("### ⚖️ Exploration vs Exploitation Balance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epsilon = metrics.get('current_epsilon', 0)
            
            st.markdown("**🎲 Epsilon (Exploration Rate)**")
            
            # Visual progress bar with color coding
            epsilon_percent = epsilon / EPSILON_START
            
            if epsilon_percent > 0.7:
                bar_color = "#F44336"  # Red - High exploration
                status = "High Exploration"
            elif epsilon_percent > 0.3:
                bar_color = "#FFC107"  # Yellow - Balanced
                status = "Balanced"
            else:
                bar_color = "#4CAF50"  # Green - High exploitation
                status = "High Exploitation"
            
            st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 10px; padding: 10px;">
                <div style="background-color: {bar_color}; width: {epsilon_percent*100}%; height: 30px; border-radius: 5px; text-align: center; line-height: 30px; color: white; font-weight: bold;">
                    {epsilon:.3f}
                </div>
            </div>
            <p style="text-align: center; margin-top: 10px;"><strong>{status}</strong></p>
            """, unsafe_allow_html=True)
            
            st.caption("""
            **Epsilon decay:** As the system learns, it explores less and exploits learned strategies more.
            - High (>0.2): Still exploring different strategies
            - Low (<0.1): Confident in learned preferences
            """)
        
        with col2:
            beta = nn_metrics.get('beta', 0)
            
            st.markdown("**🎯 Beta (Importance Sampling Weight)**")
            
            # Visual progress bar
            st.progress(beta)
            st.metric("Current Beta", f"{beta:.3f}")
            
            st.caption("""
            **Beta annealing:** Corrects bias from prioritized experience replay.
            - Starts low (~0.4): Focus on diverse experiences
            - Increases to 1.0: Full bias correction
            - Higher = More accurate gradient estimates
            """)
        
        st.divider()
        
        # Network Architecture Details
        st.markdown("### 🏗️ Network Architecture")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("📐 Architecture Details", expanded=False):
                st.code(f"""
═══════════════════════════════════════════════════════════
                   DUELING DEEP Q-NETWORK (DQN)
═══════════════════════════════════════════════════════════

INPUT LAYER ({NEURAL_INPUT_DIM} features)
├── Query Embedding:        384 dim (semantic meaning)
├── Intent One-Hot:           6 dim (definition, explanation, etc.)
├── Depth One-Hot:            3 dim (surface, moderate, comprehensive)
├── Scope One-Hot:            2 dim (specific, broad)
├── User History:            10 dim (experience, preferences, patterns)
└── Time Features:            5 dim (hour, day, session context)

SHARED FEATURE EXTRACTOR
├── Linear(410 → 128) + ReLU + Dropout(0.3) + BatchNorm
└── Linear(128 → 64)  + ReLU + Dropout(0.3) + BatchNorm

DUELING STREAMS
├── Value Stream:     64 → 32 → 1   (State value)
└── Advantage Stream: 64 → 32 → 5   (Action advantages)

OUTPUT LAYER ({NEURAL_OUTPUT_DIM} strategies)
├── Q(s,a) = V(s) + [A(s,a) - mean(A(s,·))]
└── Strategies: {', '.join(STRATEGIES)}

TRAINING CONFIGURATION
├── Optimizer:           Adam (lr={NEURAL_LEARNING_RATE})
├── Loss Function:       Smooth L1 Loss (Huber Loss)
├── Batch Size:          {BATCH_SIZE}
├── Buffer Size:         {TRAINING_BUFFER_SIZE}
├── Gamma (Discount):    {NEURAL_GAMMA}
├── Target Update:       Every {TARGET_UPDATE_FREQUENCY} steps
└── Replay:              Prioritized Experience Replay (α={REPLAY_ALPHA})

KEY FEATURES
✓ Double DQN: Reduces overestimation bias
✓ Dueling Architecture: Separates value & advantage
✓ Prioritized Replay: Focuses on important experiences
✓ Target Network: Stabilizes training
✓ Batch Normalization: Accelerates convergence
✓ Dropout: Prevents overfitting
                """, language="text")
        
        with col2:
            st.markdown("**🎓 Why This Architecture?**")
            
            st.markdown("""
            **Dueling DQN Benefits:**
            - Learns which states are valuable
            - Learns which actions are best
            - Better generalization
            - Faster convergence
            
            **Prioritized Replay:**
            - Learns from mistakes faster
            - Efficient use of experiences
            - Adaptive sample importance
            
            **Double DQN:**
            - Reduces Q-value overestimation
            - More stable learning
            - Better final performance
            """)
        
        st.divider()
        
        # Training Progress Timeline
        if len(loss_history) > 1:
            st.markdown("### 📈 Training Progress Timeline")
            
            training_steps = nn_metrics.get('training_steps', 0)
            buffer_size = nn_metrics.get('buffer_size', 0)
            
            # Calculate milestones
            milestones = {
                'start': 0,
                'first_training': BATCH_SIZE,
                'buffer_quarter': TRAINING_BUFFER_SIZE // 4,
                'buffer_half': TRAINING_BUFFER_SIZE // 2,
                'buffer_full': TRAINING_BUFFER_SIZE,
                'experienced': 100,
                'current': training_steps
            }
            
            # Create timeline
            timeline_fig = go.Figure()
            
            for milestone, step in milestones.items():
                if step <= training_steps:
                    color = '#4CAF50'
                    symbol = 'circle'
                else:
                    color = '#BDBDBD'
                    symbol = 'circle-open'
                
                timeline_fig.add_trace(go.Scatter(
                    x=[step],
                    y=[1],
                    mode='markers+text',
                    marker=dict(size=15, color=color, symbol=symbol),
                    text=[milestone.replace('_', ' ').title()],
                    textposition='top center',
                    showlegend=False
                ))
            
            timeline_fig.add_trace(go.Scatter(
                x=[0, max(milestones.values())],
                y=[1, 1],
                mode='lines',
                line=dict(color='#E0E0E0', width=2),
                showlegend=False
            ))
            
            timeline_fig.update_layout(
                title="Training Progress Milestones",
                xaxis_title="Training Steps",
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                height=250,
                xaxis=dict(showgrid=True)
            )
            
            st.plotly_chart(timeline_fig, use_container_width=True)
        
        st.divider()
        
        # Strategy Confidence Analysis (if recent query exists)
        if st.session_state.messages and st.session_state.messages[-1]['role'] == 'assistant':
            st.markdown("### 🎯 Last Query: Strategy Selection Analysis")
            
            last_metadata = st.session_state.messages[-1].get('metadata', {})
            selection_info = last_metadata.get('selection_info', {})
            
            if selection_info and 'q_values' in selection_info:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Q-values bar chart
                    fig = st.session_state.neural_viz.create_strategy_confidence_viz(selection_info)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Selection Details**")
                    
                    method = selection_info.get('method', 'unknown')
                    selected = selection_info.get('selected_strategy', 'N/A')
                    epsilon = selection_info.get('epsilon', 0)
                    confidence = selection_info.get('confidence', 0)
                    
                    # Method badge
                    method_colors = {
                        'neural_network': '#4CAF50',
                        'exploration': '#FF9800',
                        'cluster_best': '#2196F3'
                    }
                    method_color = method_colors.get(method, '#757575')
                    
                    st.markdown(f"""
                    <div style="background-color: {method_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                        <strong>{method.replace('_', ' ').title()}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Selected Strategy", selected.replace('_', ' ').title())
                    st.metric("System Confidence", f"{confidence*100:.1f}%")
                    st.metric("Exploration Rate", f"{epsilon:.3f}")
                    
                    # Explanation
                    st.markdown("**Why this strategy?**")
                    if method == 'neural_network':
                        st.write("✅ Neural network predicted this strategy has highest expected reward")
                    elif method == 'exploration':
                        st.write("🎲 Random exploration to discover potentially better strategies")
                    elif method == 'cluster_best':
                        st.write("📊 Using historically best strategy for this query type")
        
        st.divider()
        
        # Training Recommendations
        st.markdown("### 💡 Training Recommendations")
        
        training_steps = nn_metrics.get('training_steps', 0)
        buffer_size = nn_metrics.get('buffer_size', 0)
        
        if training_steps == 0:
            st.warning("""
            **🚀 Ready to start training!**
            
            1. Upload a document and generate 20-30 test queries in the Testing tab
            2. Run the automated training sequence
            3. Return here to watch the neural network learn in real-time
            """)
        elif training_steps < 50:
            st.info(f"""
            **📚 Early Training Stage** ({training_steps} steps)
            
            - Network is still exploring and learning patterns
            - Need more diverse queries for better generalization
            - Recommended: Process 20-30 more queries with varied intents
            """)
        elif training_steps < 100:
            st.success(f"""
            **🎯 Active Learning Phase** ({training_steps} steps)
            
            - Network is building confidence in strategy selection
            - Continue processing queries to refine performance
            - Watch for decreasing loss and stable Q-values
            """)
        else:
            st.success(f"""
            **🏆 Mature Learning Stage** ({training_steps} steps)
            
            - Network has substantial training experience
            - Should show clear strategy preferences per query type
            - Monitor for concept drift (changing user preferences)
            """)
        
        # Buffer status
        buffer_percent = (buffer_size / TRAINING_BUFFER_SIZE) * 100
        st.progress(buffer_size / TRAINING_BUFFER_SIZE)
        st.caption(f"Experience Buffer: {buffer_size}/{TRAINING_BUFFER_SIZE} ({buffer_percent:.1f}% full)")