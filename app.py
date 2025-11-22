# app.py
"""Enhanced Neural-RL Adaptive RAG with Complete Evaluation and Proper JSON Serialization"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime
from scipy import stats
from rag import EnhancedAdaptiveRAG
from config import *
from testing_buddy import TestingBuddy

st.set_page_config(page_title="Neural-RL RAG", page_icon="🧠", layout="wide")


# ============================================================================
#                    JSON SERIALIZATION HELPER
# ============================================================================

def convert_to_json_serializable(obj):
    """
    Recursively convert numpy and other non-serializable types to JSON-serializable types
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif obj is None:
        return None
    elif isinstance(obj, (str, int, float)):
        return obj
    else:
        # Try to convert to string as last resort
        try:
            return str(obj)
        except:
            return None


# ============================================================================
#                    CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
        animation: slideIn 0.3s ease;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    .assistant-message {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
#                    SESSION STATE INITIALIZATION
# ============================================================================

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'testing_results' not in st.session_state:
    st.session_state.testing_results = None
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'confirm_reset' not in st.session_state:
    st.session_state.confirm_reset = False


# ============================================================================
#                    SIDEBAR WITH CONFIGURATION AND INDEXING
# ============================================================================

with st.sidebar:
    st.markdown("## 🔑 Configuration")
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )
    
    user_id = st.text_input(
        "User ID",
        value="demo_user",
        help="Unique identifier for this session"
    )
    
    if st.button("🚀 Initialize System", type="primary", use_container_width=True):
        if not openai_api_key:
            st.error("❌ Please enter your OpenAI API key")
        else:
            try:
                with st.spinner("Initializing Neural-RL RAG System..."):
                    st.session_state.rag_system = EnhancedAdaptiveRAG(
                        user_id=user_id,
                        openai_api_key=openai_api_key
                    )
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
    
    # System Status
    if st.session_state.rag_system:
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        metrics = st.session_state.rag_system.get_metrics()
        
        st.metric("Total Interactions", metrics.get('total_interactions', 0))
        
        success_rate = (metrics.get('positive_feedback', 0) / 
                       max(metrics.get('total_interactions', 1), 1) * 100)
        st.metric("Success Rate", f"{success_rate:.1f}%")
        
        st.metric("Current Epsilon", f"{metrics.get('current_epsilon', 0):.3f}")
        
        # INDEXING SECTION
        st.markdown("---")
        st.markdown("### 📚 Document Indexing")
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['txt', 'md', 'pdf'],
            help="Upload text, markdown, or PDF files",
            key="sidebar_uploader"
        )
        
        if uploaded_files:
            if st.button("📥 Index Documents", use_container_width=True):
                total_chunks = 0
                
                for uploaded_file in uploaded_files:
                    try:
                        content = uploaded_file.read().decode('utf-8')
                        chunks = st.session_state.rag_system.index_document(content)
                        total_chunks += chunks
                        st.success(f"✅ {uploaded_file.name}: {chunks} chunks")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {str(e)}")
                
                st.success(f"✅ Total: {total_chunks} chunks indexed!")
        
        # Collection stats
        stats_data = st.session_state.rag_system.vector_store.get_collection_stats()
        st.metric("Documents Indexed", stats_data.get('document_count', 0))
        
        # DATA MANAGEMENT
        st.markdown("---")
        st.markdown("### 🗑️ Data Management")
        
        with st.expander("📊 Current Data"):
            st.write(f"**Interactions:** {metrics.get('total_interactions', 0)}")
            st.write(f"**Clusters:** {metrics.get('total_clusters', 0)}")
            st.write(f"**Documents:** {stats_data.get('document_count', 0)}")
            nn_metrics = metrics.get('neural_network', {})
            st.write(f"**Training Batches:** {nn_metrics.get('total_batches', 0)}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Docs", use_container_width=True):
                if stats_data['document_count'] > 0:
                    st.session_state.rag_system.clear_documents()
                    st.success("✅ Docs cleared!")
                    st.rerun()
        
        with col2:
            if st.button("🔥 Reset All", use_container_width=True, type="secondary"):
                st.session_state.confirm_reset = True
                st.rerun()
        
        if st.session_state.get('confirm_reset', False):
            st.warning("⚠️ Delete ALL learning data?")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Yes", use_container_width=True):
                    try:
                        import shutil
                        if os.path.exists('./rl_data'):
                            shutil.rmtree('./rl_data')
                        if os.path.exists('./chroma_db'):
                            shutil.rmtree('./chroma_db')
                        if os.path.exists('./logs'):
                            shutil.rmtree('./logs')
                        
                        st.session_state.rag_system = None
                        st.session_state.messages = []
                        st.session_state.testing_results = None
                        st.session_state.confirm_reset = False
                        
                        st.success("✅ All data deleted! Reinitialize system.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_reset = False
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Neural-RL Adaptive RAG**
        
        - Neural Contextual Bandits
        - Query Clustering
        - Concept Drift Detection
        - Multi-Strategy Learning
        """)


# ============================================================================
#                    MAIN TABS
# ============================================================================

tabs = st.tabs(["💬 Chat", "🧠 Reinforcement Learning", "🤖 Neural Network", "🧪 Testing"])


# ============================================================================
#                    CHAT TAB
# ============================================================================

with tabs[0]:
    st.markdown("## 💬 Adaptive RAG Chat")
    
    if st.session_state.rag_system is None:
        st.info("👈 Please initialize the system first using the sidebar")
    else:
        # Display chat messages
        for msg in st.session_state.messages:
            role_class = "user-message" if msg["role"] == "user" else "assistant-message"
            st.markdown(f'<div class="chat-message {role_class}">{msg["content"]}</div>', 
                       unsafe_allow_html=True)
            
            if msg["role"] == "assistant" and "metadata" in msg:
                with st.expander("📊 Response Details"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Strategy:** {msg['metadata']['strategy'].replace('_', ' ').title()}")
                        st.write(f"**Complexity:** {msg['metadata']['complexity']}")
                    
                    with col2:
                        st.write(f"**Documents:** {len(msg['metadata']['retrieved_docs'])}")
                        st.write(f"**Cluster:** {msg['metadata']['cluster_name']}")
                    
                    with col3:
                        timing = msg['metadata']['timing']
                        st.write(f"**Total Time:** {timing['total']:.0f}ms")
                        st.write(f"**Generation:** {timing['generation']:.0f}ms")
                    
                    if msg.get("feedback_submitted"):
                        feedback_emoji = "👍" if msg["feedback_value"] > 0 else "👎"
                        st.success(f"{feedback_emoji} Feedback recorded!")
        
        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
                try:
                    answer, metadata = st.session_state.rag_system.query(prompt)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata,
                        "feedback_submitted": False
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Feedback for last message
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            last_msg = st.session_state.messages[-1]
            
            if not last_msg.get("feedback_submitted"):
                st.markdown("---")
                st.markdown("### 📝 Was this response helpful?")
                
                col1, col2, col3 = st.columns([1, 1, 4])
                
                with col1:
                    if st.button("👍 Yes", use_container_width=True):
                        st.session_state.rag_system.submit_feedback(1)
                        last_msg["feedback_submitted"] = True
                        last_msg["feedback_value"] = 1
                        st.rerun()
                
                with col2:
                    if st.button("👎 No", use_container_width=True):
                        st.session_state.rag_system.submit_feedback(-1)
                        last_msg["feedback_submitted"] = True
                        last_msg["feedback_value"] = -1
                        st.rerun()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ============================================================================
#                    REINFORCEMENT LEARNING TAB
# ============================================================================

with tabs[1]:
    st.markdown("## 🧠 Reinforcement Learning Dashboard")
    
    if st.session_state.rag_system is None:
        st.info("👈 Please initialize the system first")
    else:
        metrics = st.session_state.rag_system.get_metrics()
        
        rl_subtabs = st.tabs([
            "📊 Overview",
            "🎯 Strategy Performance",
            "🔬 Evaluation",
            "🧪 A/B Testing",
            "👥 Clustering"
        ])
        
        # OVERVIEW TAB
        with rl_subtabs[0]:
            st.markdown("### System Performance Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Total Interactions</div>
                    <div class="metric-value">{metrics.get('total_interactions', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                success_rate = metrics.get('positive_feedback', 0) / max(metrics.get('total_interactions', 1), 1) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div>Success Rate</div>
                    <div class="metric-value">{success_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Current Epsilon</div>
                    <div class="metric-value">{metrics.get('current_epsilon', 0):.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div>Confidence</div>
                    <div class="metric-value">{metrics.get('confidence', 0):.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### Strategy Performance Summary")
            
            strategy_perf = metrics.get('strategy_performance', {})
            if strategy_perf:
                perf_data = []
                for strategy, stats in strategy_perf.items():
                    perf_data.append({
                        'Strategy': strategy.replace('_', ' ').title(),
                        'Uses': stats.get('total_uses', 0),
                        'Win Rate': f"{stats.get('win_rate', 0) * 100:.1f}%",
                        'Avg Reward': f"{stats.get('avg_reward', 0):.3f}",
                        'Status': '🚫 Suppressed' if stats.get('suppressed', False) else '✅ Active'
                    })
                
                st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
            else:
                st.info("No data yet. Start querying to see results!")
        
        # STRATEGY PERFORMANCE TAB
        with rl_subtabs[1]:
            st.markdown("### Detailed Strategy Analysis")
            
            if strategy_perf:
                strategies = list(strategy_perf.keys())
                win_rates = [strategy_perf[s].get('win_rate', 0) * 100 for s in strategies]
                uses = [strategy_perf[s].get('total_uses', 0) for s in strategies]
                
                st.markdown("#### Win Rate by Strategy")
                
                fig_winrate = go.Figure()
                fig_winrate.add_trace(go.Bar(
                    x=[s.replace('_', ' ').title() for s in strategies],
                    y=win_rates,
                    text=[f"{wr:.1f}%" for wr in win_rates],
                    textposition='auto',
                    marker=dict(
                        color=win_rates,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Win Rate %")
                    ),
                    hovertemplate='<b>%{x}</b><br>Win Rate: %{y:.1f}%<br><extra></extra>'
                ))
                fig_winrate.update_layout(
                    xaxis_title="Strategy",
                    yaxis_title="Win Rate (%)",
                    height=400,
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_winrate, use_container_width=True)
                
                st.markdown("#### Strategy Usage Distribution")
                
                fig_usage = go.Figure(data=[go.Pie(
                    labels=[s.replace('_', ' ').title() for s in strategies],
                    values=uses,
                    hole=0.4,
                    marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'])
                )])
                fig_usage.update_layout(height=400)
                st.plotly_chart(fig_usage, use_container_width=True)
                
                st.markdown("#### Average Reward by Strategy")
                
                avg_rewards = [strategy_perf[s].get('avg_reward', 0) for s in strategies]
                
                fig_reward = go.Figure()
                colors = ['#4CAF50' if r > 0 else '#f44336' for r in avg_rewards]
                
                fig_reward.add_trace(go.Bar(
                    x=[s.replace('_', ' ').title() for s in strategies],
                    y=avg_rewards,
                    text=[f"{r:.3f}" for r in avg_rewards],
                    textposition='auto',
                    marker=dict(color=colors)
                ))
                fig_reward.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_reward.update_layout(
                    xaxis_title="Strategy",
                    yaxis_title="Average Reward",
                    height=400
                )
                st.plotly_chart(fig_reward, use_container_width=True)
            else:
                st.info("No strategy data yet!")
        
        # EVALUATION TAB
        with rl_subtabs[2]:
            st.markdown("### Evaluation Metrics")
            
            feedback_history = st.session_state.rag_system.rl_agent.feedback_history
            
            if feedback_history and len(feedback_history) > 0:
                st.markdown("#### Cumulative Reward Over Time")
                
                cumulative = []
                total = 0
                for feedback in feedback_history:
                    total += feedback.get('reward', 0)
                    cumulative.append(total)
                
                fig_cumulative = go.Figure()
                fig_cumulative.add_trace(go.Scatter(
                    y=cumulative,
                    mode='lines',
                    name='Cumulative Reward',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ))
                fig_cumulative.update_layout(
                    xaxis_title="Interaction",
                    yaxis_title="Cumulative Reward",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_cumulative, use_container_width=True)
                
                if len(feedback_history) >= 20:
                    st.markdown("#### Performance Over Time Windows")
                    
                    window_size = 10
                    windows = []
                    success_rates = []
                    
                    for i in range(0, len(feedback_history) - window_size + 1, window_size // 2):
                        window = feedback_history[i:i + window_size]
                        positive = sum(1 for f in window if f.get('reward', 0) > 0)
                        success_rates.append(positive / len(window) * 100)
                        windows.append(f"W{len(windows) + 1}")
                    
                    fig_windows = go.Figure()
                    fig_windows.add_trace(go.Scatter(
                        x=windows,
                        y=success_rates,
                        mode='lines+markers',
                        line=dict(color='#4ECDC4', width=3),
                        marker=dict(size=10)
                    ))
                    fig_windows.add_hline(
                        y=np.mean(success_rates),
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Avg: {np.mean(success_rates):.1f}%"
                    )
                    fig_windows.update_layout(
                        xaxis_title="Window",
                        yaxis_title="Success Rate (%)",
                        height=400,
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig_windows, use_container_width=True)
            else:
                st.info("No evaluation data yet. Start querying!")
        
        # A/B TESTING TAB
        with rl_subtabs[3]:
            st.markdown("### A/B Testing & Baseline Comparison")
            
            if st.session_state.testing_results is not None:
                results = st.session_state.testing_results
                
                st.success("✅ Testing results available!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RL Success Rate", f"{results.get('success_rate', 0):.1f}%")
                with col2:
                    eval_data = results.get('evaluation', {})
                    rl_data = eval_data.get('rl_system', {})
                    st.metric("Mean Reward", f"{rl_data.get('mean_reward', 0):.3f}")
                with col3:
                    st.metric("Total Tests", results.get('total_queries', 0))
                
                comparisons = eval_data.get('comparisons', {})
                
                if comparisons:
                    st.markdown("#### Statistical Comparison vs Baselines")
                    
                    comparison_data = []
                    for baseline_name, stats in comparisons.items():
                        comparison_data.append({
                            'Baseline': baseline_name.replace('_', ' ').title(),
                            'RL Mean': f"{stats['rl_mean']:.3f}",
                            'Baseline Mean': f"{stats['baseline_mean']:.3f}",
                            'Improvement': f"{stats['improvement']:.1f}%",
                            'P-Value': f"{stats['p_value']:.4f}",
                            'Significant': '✅ Yes' if stats['significant'] else '❌ No',
                            'Better': '✅ RL Wins' if stats['better'] else '❌ Baseline Wins'
                        })
                    
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                    
                    st.markdown("#### Performance Comparison Chart")
                    
                    baseline_names = [c['Baseline'] for c in comparison_data]
                    rl_means = [float(c['RL Mean']) for c in comparison_data]
                    baseline_means = [float(c['Baseline Mean']) for c in comparison_data]
                    
                    fig_comparison = go.Figure()
                    fig_comparison.add_trace(go.Bar(
                        name='RL System',
                        x=baseline_names,
                        y=rl_means,
                        marker_color='#667eea'
                    ))
                    fig_comparison.add_trace(go.Bar(
                        name='Baseline',
                        x=baseline_names,
                        y=baseline_means,
                        marker_color='#ff6b6b'
                    ))
                    fig_comparison.update_layout(
                        barmode='group',
                        title="RL vs Baselines",
                        xaxis_title="Baseline Strategy",
                        yaxis_title="Mean Reward",
                        height=400
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                learning_metrics = results.get('learning_metrics', {})
                cumulative_reward = learning_metrics.get('cumulative_reward', [])
                
                if cumulative_reward:
                    st.markdown("#### Learning Curve")
                    
                    fig_learning = go.Figure()
                    fig_learning.add_trace(go.Scatter(
                        y=cumulative_reward,
                        mode='lines',
                        name='Cumulative Reward',
                        line=dict(color='#4ECDC4', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(78, 205, 196, 0.2)'
                    ))
                    fig_learning.update_layout(
                        xaxis_title="Query Number",
                        yaxis_title="Cumulative Reward",
                        height=400
                    )
                    st.plotly_chart(fig_learning, use_container_width=True)
                
                persona_info = results.get('persona', {})
                if persona_info:
                    st.markdown("#### Test Persona Details")
                    with st.expander("👤 View Details"):
                        st.write(f"**Name:** {persona_info.get('name', 'Unknown')}")
                        st.write(f"**Satisfaction:** {persona_info.get('satisfaction_rate', 0) * 100:.1f}%")
                        st.write(f"**Preferred:** {', '.join(persona_info.get('preferred_strategies', []))}")
            else:
                st.info("""
                💡 **How to use**: 
                1. Go to the Testing tab
                2. Configure and run automated testing
                3. Results will appear here automatically
                """)
                st.warning("No testing results yet. Run automated testing first!")
        
        # CLUSTERING TAB
        with rl_subtabs[4]:
            st.markdown("### Query Clustering Analysis")
            
            clusters = metrics.get('clusters', {})
            
            if clusters:
                st.metric("Total Clusters", metrics.get('total_clusters', 0))
                
                st.markdown("#### Cluster Details")
                
                for cluster_name, cluster_info in clusters.items():
                    with st.expander(f"📁 {cluster_name.replace('_', ' ').title()} ({cluster_info.get('query_count', 0)} queries)"):
                        st.write("**Example Queries:**")
                        for query in cluster_info.get('example_queries', [])[:3]:
                            st.write(f"- {query}")
                        st.write(f"**Best Strategy:** {cluster_info.get('best_strategy', 'None')}")
                
                st.markdown("#### Cluster Size Distribution")
                
                cluster_names = [name.replace('_', ' ').title() for name in clusters.keys()]
                cluster_sizes = [info.get('query_count', 0) for info in clusters.values()]
                
                fig = go.Figure(data=[go.Bar(
                    x=cluster_names,
                    y=cluster_sizes,
                    marker=dict(color=cluster_sizes, colorscale='Viridis', showscale=True)
                )])
                fig.update_layout(
                    xaxis_title="Cluster",
                    yaxis_title="Number of Queries",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No clusters yet. Start querying!")


# ============================================================================
#                    NEURAL NETWORK TAB
# ============================================================================

with tabs[2]:
    st.markdown("## 🤖 Neural Network Learning Dashboard")
    
    if st.session_state.rag_system is None:
        st.info("👈 Please initialize the system first")
    else:
        metrics = st.session_state.rag_system.get_metrics()
        nn_metrics = metrics.get('neural_network', {})
        
        nn_subtabs = st.tabs([
            "📊 Training Overview",
            "📈 Learning Curves",
            "🔍 Network Analysis",
            "📉 Loss Analysis"
        ])
        
        # TRAINING OVERVIEW
        with nn_subtabs[0]:
            st.markdown("### Neural Network Training Status")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Training Batches", nn_metrics.get('total_batches', 0))
            with col2:
                st.metric("Buffer Size", nn_metrics.get('buffer_size', 0))
            with col3:
                recent_losses = nn_metrics.get('recent_losses', [])
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                st.metric("Avg Recent Loss", f"{avg_loss:.4f}")
            with col4:
                learning_rates = nn_metrics.get('learning_rates', [0.001])
                current_lr = learning_rates[-1] if learning_rates else 0.001
                st.metric("Learning Rate", f"{current_lr:.6f}")
            
            st.markdown("---")
            st.markdown("### Recent Learning Activity")
            
            query_log = nn_metrics.get('query_learning_log', [])
            if query_log:
                log_df = pd.DataFrame(query_log[-10:])
                if not log_df.empty and 'query' in log_df.columns:
                    display_df = log_df[['query', 'strategy', 'reward', 'change']].copy()
                    display_df['strategy'] = display_df['strategy'].apply(
                        lambda x: STRATEGIES[int(x)].replace('_', ' ').title() 
                        if isinstance(x, (int, float, np.integer)) and int(x) < len(STRATEGIES) 
                        else str(x)
                    )
                    display_df['reward'] = display_df['reward'].apply(
                        lambda x: '✅ Positive' if float(x) > 0 else '❌ Negative'
                    )
                    display_df['change'] = display_df['change'].apply(
                        lambda x: f"+{float(x):.4f}" if float(x) >= 0 else f"{float(x):.4f}"
                    )
                    display_df.columns = ['Query', 'Strategy', 'Feedback', 'Score Change']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No learning activity yet. Query the system to start training!")
        
        # LEARNING CURVES
        with nn_subtabs[1]:
            st.markdown("### Training Loss Over Time")
            
            if nn_metrics.get('all_losses'):
                losses = nn_metrics['all_losses']
                batches = nn_metrics.get('batch_numbers', list(range(len(losses))))
                
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=batches,
                    y=losses,
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='#667eea', width=2),
                    marker=dict(size=4)
                ))
                
                if len(losses) > 5:
                    window = min(10, len(losses) // 5)
                    ma_losses = pd.Series(losses).rolling(window=window).mean()
                    fig_loss.add_trace(go.Scatter(
                        x=batches,
                        y=ma_losses,
                        mode='lines',
                        name=f'Moving Avg ({window})',
                        line=dict(color='#ff6b6b', width=3, dash='dash')
                    ))
                
                fig_loss.update_layout(
                    xaxis_title="Training Batch",
                    yaxis_title="Loss",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_loss, use_container_width=True)
                
                st.markdown("### Strategy Score Evolution")
                
                strategy_preds = nn_metrics.get('strategy_predictions', {})
                if strategy_preds:
                    fig_strat = go.Figure()
                    for i, strategy in enumerate(STRATEGIES):
                        if i in strategy_preds and strategy_preds[i]:
                            fig_strat.add_trace(go.Scatter(
                                y=strategy_preds[i],
                                mode='lines',
                                name=strategy.replace('_', ' ').title(),
                                line=dict(width=2)
                            ))
                    
                    fig_strat.update_layout(
                        xaxis_title="Training Step",
                        yaxis_title="Predicted Score",
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_strat, use_container_width=True)
            else:
                st.info("No training data yet!")
        
        # NETWORK ANALYSIS
        with nn_subtabs[2]:
            st.markdown("### Gradient & Weight Analysis")
            
            if nn_metrics.get('gradient_norms') and nn_metrics.get('weight_norms'):
                fig_norms = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Gradient Norms", "Weight Norms")
                )
                
                batches = nn_metrics.get('batch_numbers', [])
                
                fig_norms.add_trace(
                    go.Scatter(
                        x=batches,
                        y=nn_metrics['gradient_norms'],
                        mode='lines',
                        name='Gradient',
                        line=dict(color='#4ECDC4')
                    ),
                    row=1, col=1
                )
                
                fig_norms.add_trace(
                    go.Scatter(
                        x=batches,
                        y=nn_metrics['weight_norms'],
                        mode='lines',
                        name='Weight',
                        line=dict(color='#FF6B6B')
                    ),
                    row=1, col=2
                )
                
                fig_norms.update_xaxes(title_text="Batch", row=1, col=1)
                fig_norms.update_xaxes(title_text="Batch", row=1, col=2)
                fig_norms.update_layout(height=400, showlegend=False)
                
                st.plotly_chart(fig_norms, use_container_width=True)
                
                st.markdown("### Network Health Indicators")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    grad_norm = nn_metrics['gradient_norms'][-1] if nn_metrics['gradient_norms'] else 0
                    grad_status = "✅ Healthy" if 0.001 < grad_norm < 10 else "⚠️ Check"
                    st.metric("Latest Gradient Norm", f"{grad_norm:.4f}", delta=grad_status)
                
                with col2:
                    weight_norm = nn_metrics['weight_norms'][-1] if nn_metrics['weight_norms'] else 0
                    st.metric("Latest Weight Norm", f"{weight_norm:.2f}")
                
                with col3:
                    pred_conf = nn_metrics.get('prediction_confidence', [0])
                    conf = pred_conf[-1] if pred_conf else 0
                    st.metric("Prediction Confidence", f"{conf:.3f}")
            else:
                st.info("No network analysis data yet!")
        
        # LOSS ANALYSIS
        with nn_subtabs[3]:
            st.markdown("### Detailed Loss Analysis")
            
            if nn_metrics.get('all_losses'):
                losses = nn_metrics['all_losses']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Batches", len(losses))
                    st.metric("Initial Loss", f"{losses[0]:.4f}")
                    st.metric("Current Loss", f"{losses[-1]:.4f}")
                
                with col2:
                    improvement = ((losses[0] - losses[-1]) / losses[0] * 100) if losses[0] != 0 else 0
                    st.metric("Loss Reduction", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")
                    st.metric("Min Loss", f"{min(losses):.4f}")
                    st.metric("Avg Loss", f"{np.mean(losses):.4f}")
                
                st.markdown("### Loss Distribution")
                
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=losses,
                    nbinsx=30,
                    marker_color='#667eea'
                ))
                fig_hist.update_layout(
                    xaxis_title="Loss Value",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No loss data yet!")


# ============================================================================
#                    TESTING TAB
# ============================================================================

with tabs[3]:
    st.markdown("## 🧪 Automated Testing & Evaluation")
    
    if st.session_state.rag_system is None:
        st.info("👈 Please initialize the system and index documents first")
    else:
        st.markdown("""
        ### Testing Process
        1. **Generate Queries** from your documents
        2. **Simulate Users** with different preferences
        3. **Compare Baselines** (random, round-robin, fixed)
        4. **Analyze Results** with statistical evaluation
        """)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_test_queries = st.slider("Test Queries", 10, 100, 30, 5)
        with col2:
            include_baselines = st.checkbox("Include Baselines", value=True)
        with col3:
            test_mode = st.selectbox("Mode", ["Quick Test", "Comprehensive Test"])
        
        st.markdown("### Preferred Response Styles")
        
        col1, col2 = st.columns(2)
        with col1:
            pref_style_1 = st.selectbox(
                "Primary",
                STRATEGIES,
                format_func=lambda x: x.replace('_', ' ').title()
            )
        with col2:
            remaining = [s for s in STRATEGIES if s != pref_style_1]
            pref_style_2 = st.selectbox(
                "Secondary",
                remaining,
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        preferred_styles = [pref_style_1, pref_style_2]
        
        if st.button("🚀 Run Automated Testing", type="primary", use_container_width=True):
            collection_stats = st.session_state.rag_system.vector_store.get_collection_stats()
            
            if collection_stats['document_count'] == 0:
                st.error("❌ No documents indexed! Upload documents in sidebar first.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, message):
                    progress_bar.progress(current / total if total > 0 else 0)
                    status_text.text(message)
                
                try:
                    test_docs = st.session_state.rag_system.vector_store.collection.get(limit=5)
                    sample_text = " ".join(test_docs['documents'][:3]) if test_docs and test_docs['documents'] else ""
                    
                    if len(sample_text) < 100:
                        st.error("❌ Insufficient document content")
                    else:
                        testing_buddy = TestingBuddy(st.session_state.rag_system, preferred_styles)
                        
                        if test_mode == "Comprehensive Test":
                            results = testing_buddy.run_comprehensive_testing(
                                sample_text,
                                num_test_queries,
                                include_baselines,
                                update_progress
                            )
                        else:
                            results = testing_buddy.run_automated_testing(
                                sample_text,
                                num_test_queries,
                                update_progress
                            )
                        
                        # Store results
                        st.session_state.testing_results = results
                        
                        progress_bar.progress(1.0)
                        status_text.text("✅ Testing Complete!")
                        
                        st.success(f"""
                        ✅ **Testing Complete!**
                        - Total Queries: {results.get('total_queries', 0)}
                        - Success Rate: {results.get('success_rate', 0):.1f}%
                        - Avg Time: {results.get('avg_time_ms', 0):.0f}ms
                        
                        **View results in RL tab → A/B Testing**
                        """)
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ Testing failed: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
        
        # Display results if available
        if st.session_state.testing_results:
            st.markdown("---")
            st.markdown("### 📊 Latest Test Results")
            
            results = st.session_state.testing_results
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", int(results.get('total_queries', 0)))
            with col2:
                st.metric("Success Rate", f"{float(results.get('success_rate', 0)):.1f}%")
            with col3:
                st.metric("Positive", int(results.get('positive_feedback', 0)))
            with col4:
                st.metric("Avg Time", f"{float(results.get('avg_time_ms', 0)):.0f}ms")
            
            strategy_usage = results.get('strategy_usage', {})
            if strategy_usage:
                st.markdown("#### Strategy Usage")
                usage_df = pd.DataFrame([
                    {
                        'Strategy': s.replace('_', ' ').title(),
                        'Uses': int(count),
                        'Percent': f"{float(count)/float(results.get('total_queries',1))*100:.1f}%"
                    }
                    for s, count in strategy_usage.items()
                ])
                st.dataframe(usage_df, use_container_width=True, hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 View Detailed Analysis", use_container_width=True):
                    st.info("→ Go to **Reinforcement Learning** tab → **A/B Testing**!")
            
            with col2:
                # Prepare report with proper JSON serialization
                report = {
                    'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'configuration': {
                        'num_queries': int(num_test_queries),
                        'preferred_styles': list(preferred_styles),
                        'include_baselines': bool(include_baselines),
                        'test_mode': str(test_mode)
                    },
                    'results': convert_to_json_serializable(results)
                }
                
                try:
                    report_json = json.dumps(report, indent=2)
                    
                    st.download_button(
                        label="💾 Download Report",
                        data=report_json,
                        file_name=f"test_report_{time.strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Error creating report: {str(e)}")
        
        # Testing Tips
        with st.expander("💡 Testing Tips"):
            st.markdown("""
            **For Best Results:**
            
            1. **Start Small**: 20-30 queries for quick feedback
            2. **Include Baselines**: Enable to validate RL benefits
            3. **Multiple Runs**: Try different preferences
            4. **Document Quality**: Better docs = better tests
            
            **What to Look For:**
            
            - ✅ Success rate > 70%
            - ✅ Significant improvement over baselines
            - ✅ Late performance > Early (learning)
            - ✅ Strategy distribution matches preferences
            """)