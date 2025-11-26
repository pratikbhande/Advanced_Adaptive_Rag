# neural_viz.py
"""Real-time neural network learning visualization"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List


class NeuralNetworkVisualizer:
    """Visualize neural network learning in real-time"""
    
    def __init__(self):
        self.learning_snapshots = []
    
    def create_network_animation(self, metrics: Dict) -> go.Figure:
        """Create animated visualization of neural network learning"""
        
        loss_history = metrics.get('loss_history', [])
        q_history = metrics.get('q_value_history', [])
        reward_history = metrics.get('reward_history', [])
        
        if len(loss_history) < 2:
            return self._create_placeholder()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Training Loss (Lower = Better)',
                'Q-Value Evolution (Strategy Confidence)',
                'Reward Distribution',
                'Learning Progress'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'histogram'}, {'type': 'scatter'}]]
        )
        
        # 1. Loss curve with trend
        fig.add_trace(
            go.Scatter(
                y=loss_history,
                mode='lines+markers',
                name='Loss',
                line=dict(color='#E63946', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Add moving average
        if len(loss_history) > 5:
            ma = np.convolve(loss_history, np.ones(5)/5, mode='valid')
            fig.add_trace(
                go.Scatter(
                    y=ma,
                    mode='lines',
                    name='Trend',
                    line=dict(color='#A8DADC', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # 2. Q-value evolution
        fig.add_trace(
            go.Scatter(
                y=q_history,
                mode='lines+markers',
                name='Avg Q-Value',
                line=dict(color='#457B9D', width=2),
                marker=dict(size=4),
                fill='tozeroy',
                fillcolor='rgba(69, 123, 157, 0.2)'
            ),
            row=1, col=2
        )
        
        # 3. Reward distribution
        fig.add_trace(
            go.Histogram(
                x=reward_history,
                nbinsx=20,
                name='Rewards',
                marker=dict(
                    color=reward_history,
                    colorscale=[[0, '#F44336'], [0.5, '#FFC107'], [1, '#4CAF50']],
                    showscale=False
                )
            ),
            row=2, col=1
        )
        
        # 4. Cumulative learning
        cumulative = np.cumsum(reward_history)
        fig.add_trace(
            go.Scatter(
                y=cumulative,
                mode='lines',
                name='Cumulative Reward',
                line=dict(color='#06A77D', width=3),
                fill='tozeroy',
                fillcolor='rgba(6, 167, 125, 0.2)'
            ),
            row=2, col=2
        )
        
        # Layout
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Neural Network Learning Dynamics",
            title_font_size=20
        )
        
        fig.update_xaxes(title_text="Training Step", row=1, col=1)
        fig.update_xaxes(title_text="Training Step", row=1, col=2)
        fig.update_xaxes(title_text="Reward", row=2, col=1)
        fig.update_xaxes(title_text="Interaction", row=2, col=2)
        
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Q-Value", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Reward", row=2, col=2)
        
        return fig
    
    def create_strategy_confidence_viz(self, selection_info: Dict) -> go.Figure:
        """Visualize strategy selection confidence"""
        
        q_values = selection_info.get('q_values', {})
        
        if not q_values:
            return self._create_placeholder()
        
        strategies = list(q_values.keys())
        values = list(q_values.values())
        selected = selection_info.get('selected_strategy', '')
        
        colors = ['#4CAF50' if s == selected else '#90CAF9' for s in strategies]
        
        fig = go.Figure(data=[
            go.Bar(
                x=strategies,
                y=values,
                marker_color=colors,
                text=[f"{v:.3f}" for v in values],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Strategy Q-Values (Current Query)",
            xaxis_title="Strategy",
            yaxis_title="Q-Value (Confidence)",
            height=400,
            showlegend=False
        )
        
        fig.add_annotation(
            text=f"Selected: {selected} ({selection_info.get('method', 'unknown')})",
            xref="paper", yref="paper",
            x=0.5, y=1.1,
            showarrow=False,
            font=dict(size=14, color='#4CAF50')
        )
        
        return fig
    
    def _create_placeholder(self) -> go.Figure:
        """Create placeholder when no data"""
        fig = go.Figure()
        
        fig.add_annotation(
            text="Collecting training data...<br>Process queries to see learning dynamics",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='gray')
        )
        
        fig.update_layout(
            height=400,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        
        return fig